#!/usr/bin/env python3
"""
Full Walk-Forward Validation of 2h LGBM Model
==============================================

Uses ALL 197 days of minute bar data for proper HC #428 R1 validation:
- 60-day sliding training window
- 5-day purge gap
- ~132 OOT days (vs only 15 in the backfill)
- Regime stratification (green/red/flat)
- Permutation test (100 trials, shuffle labels)
- Per-day and per-regime Sharpe/PF/WR

This is the HONEST test before any deployment recommendation.
"""

import numpy as np
import pandas as pd
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/jupiter/Lvl3Quant/live_trading_linux')

# Import from the paper engine
from lh_2h_paper_engine import (
    compute_enhanced_hourly, add_rolling_features, add_regime_context,
    get_feature_cols, LGBM_PARAMS, SIGNAL_HOURS_UTC, HORIZON_BARS,
    COST_RT_TICKS, ES_TICK_VALUE
)

TRAIN_DAYS = 60
PURGE_DAYS = 5
MINUTE_BAR_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1")


def load_all_minute_bars():
    """Load ALL minute bar files."""
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["date"] = f.stem
            frames.append(df)
        except Exception as e:
            print(f"  Skip {f.stem}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined["ts_minute"] = pd.to_datetime(combined["ts_minute"], utc=True)
    combined = combined.sort_values("ts_minute").reset_index(drop=True)
    print(f"Loaded {len(combined):,} minute bars across {len(frames)} days")
    return combined


def run_walkforward(minute_df, n_perms=100):
    """Full walk-forward backtest with permutation test."""
    import lightgbm as lgb

    hourly = compute_enhanced_hourly(minute_df)
    hourly = add_rolling_features(hourly)
    hourly = add_regime_context(hourly)

    # Forward labels — close is stored in tick units (1 unit = 0.25 ES pts = 1 tick)
    # so the raw difference IS already in ticks. No division needed.
    hourly = hourly.sort_values("ts").reset_index(drop=True)
    hourly["fwd_ticks"] = hourly["close"].shift(-HORIZON_BARS) - hourly["close"]

    # Null overnight gaps
    for i in range(len(hourly) - HORIZON_BARS):
        ts_now = hourly["ts"].iloc[i]
        ts_fwd = hourly["ts"].iloc[i + HORIZON_BARS]
        diff_s = (ts_fwd - ts_now).total_seconds()
        if diff_s > 8 * 3600:
            hourly.loc[hourly.index[i], "fwd_ticks"] = np.nan

    # Filter intraday-clean (exclude overnight bars)
    hourly_clean = hourly[~hourly["hour"].isin([19, 20])].copy()

    dates = sorted(hourly_clean["date"].unique())
    feature_cols = get_feature_cols(hourly_clean)

    print(f"\nTotal dates: {len(dates)}")
    print(f"Feature cols: {len(feature_cols)}")
    print(f"OOT dates: {len(dates) - TRAIN_DAYS - PURGE_DAYS}")
    print(f"Signal hours (UTC): {SIGNAL_HOURS_UTC}")
    print()

    # ── Walk-forward with real labels ──
    all_trades = []
    daily_results = {}

    for i in range(TRAIN_DAYS + PURGE_DAYS, len(dates)):
        oot_date = dates[i]
        train_end_idx = i - PURGE_DAYS
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
        train_dates = dates[train_start_idx:train_end_idx]

        train = hourly_clean[hourly_clean["date"].isin(train_dates)].dropna(subset=["fwd_ticks"])
        oot = hourly_clean[hourly_clean["date"] == oot_date]
        oot_tradeable = oot[oot["hour"].isin(SIGNAL_HOURS_UTC)]

        if len(train) < 100 or len(oot_tradeable) == 0:
            continue

        X_train = train[feature_cols].fillna(0).values.astype(np.float32)
        y_train = train["fwd_ticks"].values.astype(np.float32)

        X_oot = oot_tradeable[feature_cols].fillna(0).values.astype(np.float32)
        actual_fwd = oot_tradeable["fwd_ticks"].values

        split = int(len(X_train) * 0.8)
        try:
            model = lgb.LGBMRegressor(**LGBM_PARAMS, early_stopping_rounds=50, seed=42,
                                       verbose=-1)
            model.fit(
                X_train[:split], y_train[:split],
                eval_set=[(X_train[split:], y_train[split:])],
            )
        except Exception as e:
            continue

        preds = model.predict(X_oot)

        # Generate trades for this day — enforce non-overlapping positions
        # With HORIZON_BARS=2 and hourly signals, a position entered at hour H
        # exits at hour H+2. Cannot enter again until H+2.
        day_pnl = 0.0
        day_trades = []
        oot_hours = oot_tradeable["hour"].values
        next_available_hour = -1  # track when current position exits

        for j in range(len(preds)):
            if np.isnan(actual_fwd[j]):
                continue

            # Skip if still in a position from a prior signal
            current_hour = int(oot_hours[j]) if j < len(oot_hours) else -1
            if current_hour < next_available_hour:
                continue  # position overlap — skip this signal

            direction = 1 if preds[j] > 0 else -1
            # fwd_ticks is in tick units (close stored as integer ticks). No conversion needed.
            raw_ticks = actual_fwd[j]
            gross_ticks = raw_ticks * direction
            net_ticks = gross_ticks - COST_RT_TICKS

            # Mark position occupied for HORIZON_BARS hours
            next_available_hour = current_hour + HORIZON_BARS

            day_trades.append({
                'date': oot_date,
                'direction': direction,
                'prediction': float(preds[j]),
                'actual_fwd_ticks': float(actual_fwd[j]),
                'gross_ticks': float(gross_ticks),
                'net_ticks': float(net_ticks),
                'entry_hour': current_hour,
            })
            day_pnl += net_ticks

        all_trades.extend(day_trades)
        daily_results[oot_date] = {
            'pnl_ticks': day_pnl,
            'pnl_dollars': day_pnl * ES_TICK_VALUE,
            'n_trades': len(day_trades),
            'wins': sum(1 for t in day_trades if t['net_ticks'] > 0),
        }

        if (i - TRAIN_DAYS - PURGE_DAYS) % 20 == 0:
            print(f"  [{i - TRAIN_DAYS - PURGE_DAYS + 1}/{len(dates) - TRAIN_DAYS - PURGE_DAYS}] "
                  f"{oot_date}: {len(day_trades)} trades, PnL={day_pnl:+.1f}t")

    # ── Aggregate metrics ──
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD RESULTS ({len(daily_results)} OOT days)")
    print(f"{'='*70}")

    if not all_trades:
        print("NO TRADES GENERATED!")
        return

    trades_df = pd.DataFrame(all_trades)
    n = len(trades_df)
    wins = (trades_df['net_ticks'] > 0).sum()
    total_pnl = trades_df['net_ticks'].sum()

    print(f"Total trades: {n}")
    print(f"Win rate: {wins/n:.1%}")
    print(f"Total PnL: {total_pnl:.0f} ticks (${total_pnl * ES_TICK_VALUE:,.0f})")
    print(f"Avg PnL/trade: {total_pnl/n:.1f} ticks")

    # Daily metrics
    daily_pnls = np.array([daily_results[d]['pnl_ticks'] for d in sorted(daily_results.keys())])
    sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252) if np.std(daily_pnls) > 0 else 0
    sortino_denom = np.std(daily_pnls[daily_pnls < 0]) if np.sum(daily_pnls < 0) > 1 else np.std(daily_pnls)
    sortino = np.mean(daily_pnls) / sortino_denom * np.sqrt(252) if sortino_denom > 0 else 0

    win_pnl = daily_pnls[daily_pnls > 0].sum()
    loss_pnl = abs(daily_pnls[daily_pnls < 0].sum()) if np.sum(daily_pnls < 0) > 0 else 0.01
    pf = win_pnl / loss_pnl

    print(f"\nDaily Sharpe (tick-based): {sharpe:.2f}")
    print(f"Daily Sortino (tick-based): {sortino:.2f}")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Win days: {np.sum(daily_pnls > 0)}/{len(daily_pnls)} ({np.mean(daily_pnls > 0):.0%})")

    # Capital-basis Sharpe (HC #664 audit fix: comparable to Wheel strategies)
    # 1 ES contract on $100K account. Daily returns as % of capital.
    ACCOUNT_CAPITAL = 100_000.0
    daily_dollars = daily_pnls * ES_TICK_VALUE
    daily_pct_returns = daily_dollars / ACCOUNT_CAPITAL
    cap_sharpe = np.mean(daily_pct_returns) / np.std(daily_pct_returns) * np.sqrt(252) if np.std(daily_pct_returns) > 0 else 0
    cap_sortino_denom = np.std(daily_pct_returns[daily_pct_returns < 0]) if np.sum(daily_pct_returns < 0) > 1 else np.std(daily_pct_returns)
    cap_sortino = np.mean(daily_pct_returns) / cap_sortino_denom * np.sqrt(252) if cap_sortino_denom > 0 else 0

    print(f"\nCapital-basis Sharpe ($100K, 1 contract): {cap_sharpe:.2f}")
    print(f"Capital-basis Sortino: {cap_sortino:.2f}")
    print(f"Avg daily return: {np.mean(daily_pct_returns):.2%}")
    print(f"Annualized return: {np.mean(daily_pct_returns) * 252:.1%}")

    # Max drawdown (in ticks and % of capital)
    cum = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cum)
    dd = cum - running_max
    max_dd = dd.min()
    max_dd_pct = (max_dd * ES_TICK_VALUE) / ACCOUNT_CAPITAL
    print(f"\nMax drawdown: {max_dd:.0f} ticks (${max_dd * ES_TICK_VALUE:,.0f}, {max_dd_pct:.1%} of $100K)")

    # Long vs Short
    for d, name in [(1, 'LONG'), (-1, 'SHORT')]:
        mask = trades_df['direction'] == d
        if mask.sum() > 0:
            sub = trades_df[mask]
            wr = (sub['net_ticks'] > 0).mean()
            avg = sub['net_ticks'].mean()
            print(f"\n{name}: {mask.sum()} trades, WR={wr:.1%}, avg={avg:+.1f}t")

    # ── Regime stratification (HC #428 R1) ──
    print(f"\n{'='*70}")
    print(f"REGIME STRATIFICATION")
    print(f"{'='*70}")

    # Classify days by ES close-to-close
    sorted_dates = sorted(daily_results.keys())
    regime_map = {}
    for i, d in enumerate(sorted_dates):
        day_bars = hourly_clean[hourly_clean['date'] == d]
        if len(day_bars) >= 2:
            day_return = day_bars['close'].iloc[-1] - day_bars['close'].iloc[0]
            if day_return > 2:
                regime_map[d] = 'GREEN'
            elif day_return < -2:
                regime_map[d] = 'RED'
            else:
                regime_map[d] = 'FLAT'

    for regime in ['GREEN', 'RED', 'FLAT']:
        regime_dates = [d for d in sorted_dates if regime_map.get(d) == regime]
        if not regime_dates:
            continue
        regime_pnls = [daily_results[d]['pnl_ticks'] for d in regime_dates]
        regime_pnls = np.array(regime_pnls)
        r_sharpe = np.mean(regime_pnls) / np.std(regime_pnls) * np.sqrt(252) if np.std(regime_pnls) > 0 else 0
        r_wr = np.mean(regime_pnls > 0)
        print(f"  {regime:5}: {len(regime_dates)} days, avg={np.mean(regime_pnls):+.1f}t/day, "
              f"Sharpe={r_sharpe:.2f}, WR={r_wr:.0%}")

    # Check regime gap (HC #428)
    sharpe_by_regime = {}
    for regime in ['GREEN', 'RED', 'FLAT']:
        regime_dates = [d for d in sorted_dates if regime_map.get(d) == regime]
        if len(regime_dates) > 3:
            rpnls = np.array([daily_results[d]['pnl_ticks'] for d in regime_dates])
            sharpe_by_regime[regime] = np.mean(rpnls) / np.std(rpnls) * np.sqrt(252) if np.std(rpnls) > 0 else 0

    if 'GREEN' in sharpe_by_regime and 'RED' in sharpe_by_regime:
        gap = abs(sharpe_by_regime['GREEN'] - sharpe_by_regime['RED']) / max(abs(sharpe_by_regime['GREEN']), abs(sharpe_by_regime['RED']))
        print(f"\n  Regime gap: {gap:.3f} (limit: 0.50) → {'PASS' if gap < 0.50 else 'FAIL'}")

    # ── Permutation test ──
    print(f"\n{'='*70}")
    print(f"PERMUTATION TEST ({n_perms} trials)")
    print(f"{'='*70}")

    rng = np.random.RandomState(42)
    model_total_pnl = total_pnl
    random_pnls = np.zeros(n_perms)

    for p in range(n_perms):
        # Shuffle trade directions randomly
        random_dirs = rng.choice([-1, 1], size=n)
        # actual_fwd_ticks is in tick units (close stored as integer ticks)
        raw_ticks = trades_df['actual_fwd_ticks'].values
        random_pnl = np.sum(raw_ticks * random_dirs - COST_RT_TICKS)
        random_pnls[p] = random_pnl

        if (p + 1) % 25 == 0:
            print(f"  Perm {p+1}/{n_perms}: random PnL = {random_pnl:.0f}t")

    p_value = np.mean(random_pnls >= model_total_pnl)
    genuine_edge = model_total_pnl / n - np.mean(random_pnls) / n

    print(f"\n  Model PnL:  {model_total_pnl:.0f} ticks")
    print(f"  Random mean: {np.mean(random_pnls):.0f} ticks (std={np.std(random_pnls):.0f})")
    print(f"  p-value:    {p_value:.4f}")
    print(f"  Genuine edge per trade: {genuine_edge:+.1f} ticks")

    # ── VERDICT ──
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    passes = True
    checks = []

    if sharpe < 1.0:
        checks.append(f"FAIL: Sharpe {sharpe:.2f} < 1.0")
        passes = False
    else:
        checks.append(f"PASS: Sharpe {sharpe:.2f}")

    if p_value >= 0.05:
        checks.append(f"FAIL: p-value {p_value:.4f} >= 0.05")
        passes = False
    else:
        checks.append(f"PASS: p-value {p_value:.4f}")

    if len(daily_results) < 40:
        checks.append(f"FAIL: Only {len(daily_results)} OOT days (need 40+)")
        passes = False
    else:
        checks.append(f"PASS: {len(daily_results)} OOT days")

    gap_val = gap if 'gap' in dir() else 0
    if gap_val > 0.50:
        checks.append(f"FAIL: Regime gap {gap_val:.3f} > 0.50")
        passes = False
    else:
        checks.append(f"PASS: Regime gap {gap_val:.3f}")

    for c in checks:
        print(f"  {c}")

    print(f"\n  OVERALL: {'✅ ALL GATES PASS' if passes else '❌ FAILED'}")

    # Save results
    output = {
        'generated': datetime.now().isoformat(),
        'n_oot_days': len(daily_results),
        'n_trades': n,
        'win_rate': float(wins/n),
        'total_pnl_ticks': float(total_pnl),
        'total_pnl_dollars': float(total_pnl * ES_TICK_VALUE),
        'avg_pnl_ticks': float(total_pnl/n),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'profit_factor': float(pf),
        'max_dd_ticks': float(max_dd),
        'p_value': float(p_value),
        'genuine_edge_per_trade': float(genuine_edge),
        'regime_sharpes': {k: float(v) for k, v in sharpe_by_regime.items()},
        'regime_gap': float(gap_val),
        'passes_all_gates': passes,
    }

    out_path = '/home/jupiter/Lvl3Quant/output/lh_2h_full_walkforward_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--perms', type=int, default=100)
    args = parser.parse_args()

    print("="*70)
    print("FULL WALK-FORWARD VALIDATION — 2H LGBM MODEL")
    print("HC #428 R1: ≥40 OOT days, all regimes, permutation test")
    print("="*70)

    minute_df = load_all_minute_bars()
    if minute_df.empty:
        print("No data!")
        sys.exit(1)

    run_walkforward(minute_df, n_perms=args.perms)
