#!/usr/bin/env python3
"""
2h Directional Model — INTRADAY-CLEAN Labels
==============================================

FIXED version: forward labels never cross overnight gaps.
Any bar where close[t+2] would be on a different trading date is DROPPED
from both training and OOT sets.

Audit showed:
  IC with all bars:       0.516
  IC excluding cross-day: 0.642
  Cross-day bars alone:   0.133

This script uses ONLY same-day labels → cleaner signal, no noise dilution.

Based on lh_2h_enhanced_ic_push.py — same features, same LGBM params.
Adds: regime stratification, per-month breakdown, long/short analysis,
Sortino ratio, profit factor, trade simulation with market order costs.

Author: Claude (HC #658 / HC #659)
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Import data loading and feature engineering from the original script
sys.path.insert(0, str(ROOT / "scripts"))
from lh_2h_enhanced_ic_push import (
    load_minute_bars,
    compute_enhanced_hourly,
    add_rolling_features,
    add_regime_context,
    get_feature_cols,
    train_lgbm,
    TRAIN_DAYS,
    HORIZON_BARS,
)

OUTPUT_DIR = ROOT / "output" / "lh_2h_intraday_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / "lh_2h_intraday_clean.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [LH-CLEAN] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
log = logging.getLogger('LH-CLEAN')

# ── Constants ──
PURGE_DAYS = 5
N_PERMUTATIONS = 100
ES_TICK_VALUE = 12.50
COST_MARKET_RT_TICKS = 1.376  # AMP/Rithmic canonical

# =============================================================================
# INTRADAY-CLEAN LABEL (THE FIX)
# =============================================================================

def add_intraday_forward_labels(df, horizon_bars=2):
    """
    Compute fwd_ticks ONLY within the same trading day.
    Drop any bar where close[t + horizon_bars] would be on a different date.
    This means the last `horizon_bars` bars of each day are dropped.
    """
    df = df.copy()
    df['fwd_ticks'] = np.nan

    for date_val, grp in df.groupby('date'):
        idx = grp.index
        n = len(grp)
        # Only bars 0..n-horizon_bars-1 have valid same-day forward labels
        for j in range(n - horizon_bars):
            src_idx = idx[j]
            tgt_idx = idx[j + horizon_bars]
            df.loc[src_idx, 'fwd_ticks'] = df.loc[tgt_idx, 'close'] - df.loc[src_idx, 'close']

    before = len(df)
    df = df.dropna(subset=['fwd_ticks'])
    after = len(df)
    dropped = before - after
    log.info(f"Intraday label: kept {after}/{before} bars, dropped {dropped} cross-day bars ({dropped/before*100:.1f}%)")
    return df


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

def run_walkforward(hourly, feature_cols, shuffle_labels=False, purge_days=0):
    """Sliding walk-forward with LGBM. Returns IC, predictions, actuals, dates."""
    dates = sorted(hourly['date'].unique())

    all_preds = []
    all_actuals = []
    all_dates = []
    all_hours = []

    for i in range(TRAIN_DAYS + purge_days, len(dates)):
        oot_date = dates[i]
        train_end_idx = i - purge_days
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
        train_dates = dates[train_start_idx:train_end_idx]

        train = hourly[hourly['date'].isin(train_dates)]
        oot = hourly[hourly['date'] == oot_date]

        if len(train) < 100 or len(oot) == 0:
            continue

        X_train = train[feature_cols].fillna(0).values
        y_train = train['fwd_ticks'].values
        X_oot = oot[feature_cols].fillna(0).values
        y_oot = oot['fwd_ticks'].values

        if shuffle_labels:
            y_train = np.random.permutation(y_train)

        try:
            split = int(len(X_train) * 0.8)
            model = train_lgbm(X_train[:split], y_train[:split],
                               X_train[split:], y_train[split:])
            preds = model.predict(X_oot)

            all_preds.extend(preds)
            all_actuals.extend(y_oot)
            all_dates.extend([oot_date] * len(y_oot))
            all_hours.extend(oot['hour'].values.tolist())

        except Exception as e:
            log.warning(f"Fold {oot_date} failed: {e}")
            continue

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    dates_arr = np.array(all_dates)
    hours_arr = np.array(all_hours)

    if len(preds) < 50:
        return 0.0, preds, actuals, dates_arr, hours_arr

    ic = float(stats.spearmanr(preds, actuals)[0])
    return ic, preds, actuals, dates_arr, hours_arr


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def classify_day_regime(daily_returns):
    """Classify each date as GREEN (>+0.2%), RED (<-0.2%), FLAT."""
    regimes = {}
    for date, ret in daily_returns.items():
        if ret > 0.002:
            regimes[date] = 'GREEN'
        elif ret < -0.002:
            regimes[date] = 'RED'
        else:
            regimes[date] = 'FLAT'
    return regimes


def compute_trade_simulation(preds, actuals, dates, cost_ticks=COST_MARKET_RT_TICKS):
    """Full trade simulation: direction = sign(pred), market orders."""
    directions = np.sign(preds)
    gross_ticks = directions * actuals
    net_ticks = gross_ticks - cost_ticks

    # Per-trade metrics
    n_trades = len(net_ticks)
    winners = net_ticks > 0
    losers = net_ticks < 0
    win_rate = np.mean(winners) if n_trades > 0 else 0

    # Total
    total_net = net_ticks.sum()
    avg_net = net_ticks.mean() if n_trades > 0 else 0
    avg_gross = gross_ticks.mean() if n_trades > 0 else 0

    # Profit factor
    gross_wins = net_ticks[winners].sum() if winners.any() else 0
    gross_losses = abs(net_ticks[losers].sum()) if losers.any() else 1e-9
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Daily P&L for Sharpe/Sortino
    unique_dates = sorted(set(dates))
    daily_pnl = []
    for d in unique_dates:
        mask = dates == d
        daily_pnl.append(net_ticks[mask].sum())
    daily_pnl = np.array(daily_pnl)

    # Sharpe (annualized, 252 trading days)
    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino (annualized, downside deviation only)
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) > 0:
        downside_std = np.sqrt(np.mean(downside**2))
        sortino = (daily_pnl.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0
    else:
        sortino = float('inf') if daily_pnl.mean() > 0 else 0.0

    # Long vs Short
    long_mask = directions > 0
    short_mask = directions < 0

    long_stats = {}
    if long_mask.any():
        long_net = net_ticks[long_mask]
        long_stats = {
            'n_trades': int(long_mask.sum()),
            'avg_net_ticks': float(long_net.mean()),
            'win_rate': float(np.mean(long_net > 0)),
            'total_net_ticks': float(long_net.sum()),
        }

    short_stats = {}
    if short_mask.any():
        short_net = net_ticks[short_mask]
        short_stats = {
            'n_trades': int(short_mask.sum()),
            'avg_net_ticks': float(short_net.mean()),
            'win_rate': float(np.mean(short_net > 0)),
            'total_net_ticks': float(short_net.sum()),
        }

    return {
        'n_trades': n_trades,
        'win_rate': float(win_rate),
        'avg_gross_ticks': float(avg_gross),
        'avg_net_ticks': float(avg_net),
        'total_net_ticks': float(total_net),
        'total_pnl_usd': float(total_net * ES_TICK_VALUE),
        'profit_factor': float(profit_factor),
        'daily_sharpe': float(sharpe),
        'daily_sortino': float(sortino),
        'daily_pnl_mean': float(daily_pnl.mean()),
        'daily_pnl_std': float(daily_pnl.std()),
        'n_days': len(daily_pnl),
        'long': long_stats,
        'short': short_stats,
    }


def per_day_ic(preds, actuals, dates):
    """IC per OOT date."""
    unique_dates = sorted(set(dates))
    day_ics = {}
    for d in unique_dates:
        mask = dates == d
        p = preds[mask]
        a = actuals[mask]
        if len(p) >= 3 and p.std() > 0 and a.std() > 0:
            day_ics[d] = float(stats.spearmanr(p, a)[0])
        else:
            day_ics[d] = np.nan
    return day_ics


def per_month_breakdown(preds, actuals, dates, cost_ticks=COST_MARKET_RT_TICKS):
    """Per-month trade stats."""
    directions = np.sign(preds)
    net_ticks = directions * actuals - cost_ticks

    months = {}
    for i, d in enumerate(dates):
        m = d[:7]  # YYYY-MM
        if m not in months:
            months[m] = []
        months[m].append(net_ticks[i])

    breakdown = {}
    for m in sorted(months.keys()):
        arr = np.array(months[m])
        breakdown[m] = {
            'n_trades': len(arr),
            'total_net_ticks': float(arr.sum()),
            'avg_net_ticks': float(arr.mean()),
            'win_rate': float(np.mean(arr > 0)),
            'total_pnl_usd': float(arr.sum() * ES_TICK_VALUE),
        }
    return breakdown


def regime_stratification(preds, actuals, dates, hourly_df, cost_ticks=COST_MARKET_RT_TICKS):
    """Stratify IC and trade stats by day regime (GREEN/RED/FLAT)."""
    # Compute daily close-to-close returns for regime classification
    daily_data = hourly_df.groupby('date').agg(
        day_open=('open', 'first'),
        day_close=('close', 'last'),
    )
    daily_returns = ((daily_data['day_close'] - daily_data['day_open']) / daily_data['day_open']).to_dict()
    regimes = classify_day_regime(daily_returns)

    regime_results = {}
    for regime_label in ['GREEN', 'RED', 'FLAT']:
        regime_dates = {d for d, r in regimes.items() if r == regime_label}
        mask = np.array([d in regime_dates for d in dates])

        if mask.sum() < 10:
            regime_results[regime_label] = {'n_bars': int(mask.sum()), 'ic': np.nan}
            continue

        p = preds[mask]
        a = actuals[mask]

        ic = float(stats.spearmanr(p, a)[0]) if p.std() > 0 and a.std() > 0 else 0.0

        directions = np.sign(p)
        net = directions * a - cost_ticks
        wr = float(np.mean(net > 0))
        avg_net = float(net.mean())

        regime_results[regime_label] = {
            'n_bars': int(mask.sum()),
            'ic': float(ic),
            'win_rate': wr,
            'avg_net_ticks': avg_net,
            'total_net_ticks': float(net.sum()),
        }

    # Regime-agnostic check (HC #428 R1)
    green_ic = regime_results.get('GREEN', {}).get('ic', 0) or 0
    red_ic = regime_results.get('RED', {}).get('ic', 0) or 0
    max_ic = max(abs(green_ic), abs(red_ic))
    if max_ic > 0:
        regime_gap = abs(green_ic - red_ic) / max_ic
    else:
        regime_gap = 0

    regime_results['_regime_gap'] = float(regime_gap)
    regime_results['_regime_agnostic_pass'] = regime_gap <= 0.50

    return regime_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("2h Directional Model — INTRADAY-CLEAN Labels")
    log.info("Fix: labels never cross overnight gaps")
    log.info(f"Purge days: {PURGE_DAYS}, Permutations: {N_PERMUTATIONS}")
    log.info(f"Cost assumption: {COST_MARKET_RT_TICKS} ticks RT (market orders)")
    log.info("=" * 60)

    # ── Load and process ──
    log.info("Loading minute bars...")
    minutes = load_minute_bars()

    log.info("Computing enhanced hourly features...")
    hourly = compute_enhanced_hourly(minutes)
    hourly = add_rolling_features(hourly)
    hourly = add_regime_context(hourly)

    del minutes
    gc.collect()

    # ── INTRADAY-CLEAN LABELS (THE FIX) ──
    log.info("Applying intraday-clean forward labels (same-day only)...")
    hourly = add_intraday_forward_labels(hourly, horizon_bars=HORIZON_BARS)

    feature_cols = get_feature_cols(hourly)
    log.info(f"Feature count: {len(feature_cols)}")
    log.info(f"Sample count: {len(hourly)}")
    log.info(f"Date range: {hourly['date'].min()} → {hourly['date'].max()}")
    log.info(f"Unique dates: {hourly['date'].nunique()}")

    # ── Walk-forward (real) ──
    log.info("\nRunning REAL walk-forward (LGBM, intraday-clean labels)...")
    t0 = time.time()
    real_ic, preds, actuals, dates_arr, hours_arr = run_walkforward(
        hourly, feature_cols, shuffle_labels=False, purge_days=PURGE_DAYS
    )
    elapsed = time.time() - t0
    log.info(f"Real IC: {real_ic:.4f} (took {elapsed:.0f}s, {len(preds)} samples)")

    # ── Permutation test (100 shuffles, 5-day purge) ──
    log.info(f"\nRunning {N_PERMUTATIONS} permutation tests (5-day purge)...")
    shuf_ics = []
    for trial in range(N_PERMUTATIONS):
        shuf_ic, _, _, _, _ = run_walkforward(
            hourly, feature_cols, shuffle_labels=True, purge_days=PURGE_DAYS
        )
        shuf_ics.append(shuf_ic)
        if (trial + 1) % 10 == 0:
            log.info(f"  Permutation {trial+1}/{N_PERMUTATIONS}: mean_shuf_ic={np.mean(shuf_ics):.4f}")

    mean_shuf = np.mean(shuf_ics)
    std_shuf = np.std(shuf_ics)
    genuine_ic = real_ic - mean_shuf
    p_value = np.mean([s >= real_ic for s in shuf_ics])

    log.info(f"\n{'='*40}")
    log.info("IC RESULTS")
    log.info(f"{'='*40}")
    log.info(f"  Real IC:        {real_ic:.4f}")
    log.info(f"  Shuffle IC:     {mean_shuf:.4f} ± {std_shuf:.4f}")
    log.info(f"  GENUINE IC:     {genuine_ic:.4f}")
    log.info(f"  p-value:        {p_value:.3f}")

    # ── Per-day IC ──
    log.info(f"\n{'='*40}")
    log.info("PER-DAY IC BREAKDOWN")
    log.info(f"{'='*40}")
    day_ics = per_day_ic(preds, actuals, dates_arr)
    valid_ics = [v for v in day_ics.values() if not np.isnan(v)]
    for d, ic_val in sorted(day_ics.items()):
        log.info(f"  {d}: IC={ic_val:.4f}" if not np.isnan(ic_val) else f"  {d}: IC=N/A")
    log.info(f"  Median day IC: {np.median(valid_ics):.4f}")
    log.info(f"  Mean day IC:   {np.mean(valid_ics):.4f}")
    log.info(f"  Days with IC>0: {sum(1 for v in valid_ics if v > 0)}/{len(valid_ics)}")

    # ── Trade simulation ──
    log.info(f"\n{'='*40}")
    log.info("TRADE SIMULATION (market orders, cost={COST_MARKET_RT_TICKS} ticks RT)")
    log.info(f"{'='*40}")
    sim = compute_trade_simulation(preds, actuals, dates_arr)
    log.info(f"  Trades:         {sim['n_trades']}")
    log.info(f"  Win rate:       {sim['win_rate']:.1%}")
    log.info(f"  Avg gross:      {sim['avg_gross_ticks']:.3f} ticks")
    log.info(f"  Avg net:        {sim['avg_net_ticks']:.3f} ticks")
    log.info(f"  Total net:      {sim['total_net_ticks']:.1f} ticks (${sim['total_pnl_usd']:,.0f})")
    log.info(f"  Profit factor:  {sim['profit_factor']:.2f}")
    log.info(f"  Daily Sharpe:   {sim['daily_sharpe']:.2f}")
    log.info(f"  Daily Sortino:  {sim['daily_sortino']:.2f}")

    if sim['long']:
        log.info(f"\n  LONG trades:    {sim['long']['n_trades']}, WR={sim['long']['win_rate']:.1%}, avg_net={sim['long']['avg_net_ticks']:.3f}")
    if sim['short']:
        log.info(f"  SHORT trades:   {sim['short']['n_trades']}, WR={sim['short']['win_rate']:.1%}, avg_net={sim['short']['avg_net_ticks']:.3f}")

    # ── Regime stratification ──
    log.info(f"\n{'='*40}")
    log.info("REGIME STRATIFICATION (GREEN/RED/FLAT)")
    log.info(f"{'='*40}")
    regime_stats = regime_stratification(preds, actuals, dates_arr, hourly)
    for regime in ['GREEN', 'RED', 'FLAT']:
        r = regime_stats.get(regime, {})
        ic_val = r.get('ic', 'N/A')
        ic_str = f"{ic_val:.4f}" if isinstance(ic_val, float) and not np.isnan(ic_val) else 'N/A'
        wr = r.get('win_rate', 0)
        avg = r.get('avg_net_ticks', 0)
        n = r.get('n_bars', 0)
        log.info(f"  {regime:5s}: n={n:4d}, IC={ic_str}, WR={wr:.1%}, avg_net={avg:.3f} ticks")

    gap = regime_stats.get('_regime_gap', 0)
    agnostic = regime_stats.get('_regime_agnostic_pass', False)
    log.info(f"  Regime gap: {gap:.2f} ({'PASS' if agnostic else 'FAIL'} HC#428 ≤0.50)")

    # ── Per-month breakdown ──
    log.info(f"\n{'='*40}")
    log.info("PER-MONTH BREAKDOWN")
    log.info(f"{'='*40}")
    monthly = per_month_breakdown(preds, actuals, dates_arr)
    for m, ms in monthly.items():
        log.info(f"  {m}: trades={ms['n_trades']:3d}, net={ms['total_net_ticks']:+7.1f} ticks, "
                 f"WR={ms['win_rate']:.1%}, PnL=${ms['total_pnl_usd']:+,.0f}")

    # ── Feature importance ──
    log.info(f"\n{'='*40}")
    log.info("FEATURE IMPORTANCE (single full-data fit)")
    log.info(f"{'='*40}")
    import lightgbm as lgb
    X_all = hourly[feature_cols].fillna(0).values
    y_all = hourly['fwd_ticks'].values
    temp_model = lgb.LGBMRegressor(
        num_leaves=15, max_depth=4, n_estimators=200,
        feature_fraction=0.5, min_child_samples=50,
        lambda_l1=1.0, lambda_l2=5.0, verbosity=-1
    )
    temp_model.fit(X_all, y_all)
    importances = temp_model.feature_importances_
    fi_pairs = sorted(zip(feature_cols, importances.tolist()), key=lambda x: -x[1])
    for name, imp in fi_pairs[:20]:
        log.info(f"  {name:35s} {imp:6d}")

    # ── Save results ──
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'description': '2h directional model with intraday-clean labels (no cross-day forward labels)',
        'fix': 'Dropped bars where fwd_ticks label crosses overnight gap',
        'ic': {
            'real_ic': float(real_ic),
            'mean_shuffle_ic': float(mean_shuf),
            'std_shuffle_ic': float(std_shuf),
            'genuine_ic': float(genuine_ic),
            'p_value': float(p_value),
            'n_permutations': N_PERMUTATIONS,
            'purge_days': PURGE_DAYS,
        },
        'per_day_ic': {k: (float(v) if not np.isnan(v) else None) for k, v in day_ics.items()},
        'per_day_ic_summary': {
            'median': float(np.median(valid_ics)),
            'mean': float(np.mean(valid_ics)),
            'pct_positive': float(sum(1 for v in valid_ics if v > 0) / len(valid_ics)),
            'n_days': len(valid_ics),
        },
        'trade_simulation': sim,
        'regime_stratification': {k: v for k, v in regime_stats.items()},
        'per_month': monthly,
        'feature_importance_top20': [(n, int(i)) for n, i in fi_pairs[:20]],
        'n_features': len(feature_cols),
        'n_samples': len(preds),
        'date_range': [hourly['date'].min(), hourly['date'].max()],
        'cost_assumption_ticks': COST_MARKET_RT_TICKS,
    }

    output_file = OUTPUT_DIR / "results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_file}")

    # Save predictions for further analysis
    pred_df = pd.DataFrame({
        'date': dates_arr,
        'hour': hours_arr,
        'prediction': preds,
        'actual': actuals,
    })
    pred_file = OUTPUT_DIR / "predictions.parquet"
    pred_df.to_parquet(pred_file, index=False)
    log.info(f"Predictions saved to {pred_file}")

    # ── Final verdict ──
    log.info(f"\n{'='*60}")
    log.info("FINAL VERDICT")
    log.info(f"{'='*60}")
    log.info(f"  Genuine IC:     {genuine_ic:.4f}")
    log.info(f"  p-value:        {p_value:.3f}")
    log.info(f"  Daily Sharpe:   {sim['daily_sharpe']:.2f}")
    log.info(f"  Daily Sortino:  {sim['daily_sortino']:.2f}")
    log.info(f"  Profit factor:  {sim['profit_factor']:.2f}")
    log.info(f"  Win rate:       {sim['win_rate']:.1%}")
    log.info(f"  Avg net/trade:  {sim['avg_net_ticks']:.3f} ticks")
    profitable = sim['avg_net_ticks'] > 0
    log.info(f"  Market order profitable: {'YES' if profitable else 'NO'}")
    log.info(f"  Regime-agnostic:         {'PASS' if agnostic else 'FAIL'}")

    return results


if __name__ == '__main__':
    main()
