"""
Advanced Strategy Test on EventTransformer OOS Predictions
==========================================================
Tests three strategies that address the core problem:
  Signal is real (IC=0.094) but per-trade edge (0.29 ticks) < ES costs (1.24 ticks).

Strategy 1: Confidence Gating — only trade top N% most extreme predictions
Strategy 2: Vol-Gated Transformer — combine pred confidence with trailing vol
Strategy 3: Hold Until Flip — hold until prediction changes sign (THE KEY TEST)

OOS methodology:
  - Days 1-40: parameter optimization
  - Days 41-94: TRUE OOS with fixed best config
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── Constants ──
TICK = 0.25
TICK_VAL = 12.50
ES_SPREAD_TICKS = 1.0
ES_COMM_TICKS = 0.24  # $3.00 RT / $12.50
SPY_SPREAD_TICKS = 0.40  # $5.00 RT / $12.50 for 500sh equivalent
BARS_PER_SEC = 10  # 100ms bars
TRAIN_DAYS = 40

# Cost profiles: (spread_ticks, comm_ticks, label)
COST_PROFILES = [
    (ES_SPREAD_TICKS, ES_COMM_TICKS, 'ES_futures'),
    (SPY_SPREAD_TICKS, 0.0, 'SPY_500sh'),
    (0.0, 0.0, 'zero_cost'),
]

LVL3_ROOT = Path(__file__).parent.parent
FEAT_CACHE = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oos_predictions_event_20260228_123832.npz'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'


def load_data():
    """Load predictions and MBO features, return aligned day data."""
    print("Loading OOS predictions...")
    pred_data = np.load(str(PRED_FILE), allow_pickle=True)
    pred_dates = sorted(set(
        k.replace('_preds', '').replace('_targets', '')
        for k in pred_data.keys()
    ))
    print(f"  {len(pred_dates)} prediction days")

    print("Loading MBO features (mid, spread)...")
    days = {}
    for date in pred_dates:
        mbo_path = FEAT_CACHE / f'{date}_mbo_features.npz'
        if not mbo_path.exists():
            continue

        preds = pred_data[f'{date}_preds']
        targets = pred_data[f'{date}_targets']

        mbo = np.load(str(mbo_path))
        feats = mbo['mbo_features']
        mid = feats[:, 0].astype(np.float32)
        spread = feats[:, 1].astype(np.float32)

        # Align lengths
        n = min(len(mid), len(preds))
        raw_preds = preds[:n].copy()

        # Z-score predictions per day for threshold comparison
        # Raw preds are ~[-0.35, +0.30] with std~0.085
        pred_mean = np.mean(raw_preds)
        pred_std = np.std(raw_preds)
        if pred_std > 0:
            z_preds = (raw_preds - pred_mean) / pred_std
        else:
            z_preds = np.zeros_like(raw_preds)

        days[date] = {
            'mid': mid[:n].copy(),
            'spread': spread[:n].copy(),
            'preds': raw_preds,       # raw for confidence gating (percentile-based)
            'z_preds': z_preds,        # z-scored for threshold-based strategies
            'targets': targets[:n].copy(),
            'n_bars': n,
        }
        del feats, mbo

    all_dates = sorted(days.keys())
    print(f"  Loaded {len(all_dates)} days ({all_dates[0]} to {all_dates[-1]})")

    # Print z-score stats
    sample_date = all_dates[0]
    zp = days[sample_date]['z_preds']
    print(f"  Z-score check ({sample_date}): mean={np.mean(zp):.4f}, std={np.std(zp):.4f}, "
          f"max_abs={np.max(np.abs(zp)):.2f}")
    all_z = np.concatenate([days[d]['z_preds'] for d in all_dates])
    print(f"  Z-score overall: >1.0={np.mean(np.abs(all_z)>1.0)*100:.1f}%, "
          f">2.0={np.mean(np.abs(all_z)>2.0)*100:.1f}%, "
          f">3.0={np.mean(np.abs(all_z)>3.0)*100:.1f}%")

    return days, all_dates


def compute_trailing_rvol(mid, window=500):
    """Compute trailing realized vol (std of returns) over window bars. No lookahead."""
    n = len(mid)
    rvol = np.zeros(n, dtype=np.float32)
    # Returns: mid[i] - mid[i-1]
    returns = np.diff(mid)
    # Trailing std of returns
    for i in range(window, n):
        rvol[i] = np.std(returns[i-window:i])
    return rvol


def compute_trailing_rvol_fast(mid, window=500):
    """Fast trailing rvol using cumulative sums."""
    n = len(mid)
    rvol = np.zeros(n, dtype=np.float32)
    returns = np.diff(mid)
    if len(returns) < window:
        return rvol

    # Use sliding window with cumsum trick
    ret2 = returns ** 2
    cumsum = np.cumsum(returns)
    cumsum2 = np.cumsum(ret2)

    # For index i in rvol, use returns[i-window:i]
    for i in range(window, n):
        s = cumsum[i-1] - (cumsum[i-1-window] if i-1-window >= 0 else 0)
        s2 = cumsum2[i-1] - (cumsum2[i-1-window] if i-1-window >= 0 else 0)
        var = s2 / window - (s / window) ** 2
        rvol[i] = np.sqrt(max(var, 0))

    return rvol


# ── Strategy 1: Confidence Gating ──

def sim_confidence_gating(mid, spread, preds, percentile_thresh, hold_bars=600,
                          cooldown=100, spread_cost_ticks=0.0, comm_ticks=0.24):
    # HC #231(A): spread cost deleted — fill price already encodes side
    """
    Only trade when |pred| is in the top N% for that day.
    percentile_thresh: e.g. 0.01 = top 1%.
    """
    n = len(mid)
    abs_preds = np.abs(preds)

    # Dynamic threshold: top N% of absolute predictions
    cutoff = np.percentile(abs_preds[abs_preds > 0], (1.0 - percentile_thresh) * 100)
    if cutoff <= 0:
        return 0.0, 0, 0, [], 0.0

    total_pnl = 0.0
    trades = 0
    wins = 0
    trade_pnls = []
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    cost_per_trade = spread_cost_ticks + comm_ticks

    for i in range(n):
        if in_pos:
            elapsed = i - entry_bar
            if elapsed >= hold_bars:
                curr_pnl = direction * (mid[i] - entry_price) / TICK
                pnl = curr_pnl - cost_per_trade
                total_pnl += pnl
                trade_pnls.append(pnl)
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i
        elif i - last_exit >= cooldown and abs_preds[i] >= cutoff:
            entry_price = mid[i]
            direction = 1 if preds[i] > 0 else -1
            entry_bar = i
            in_pos = True
            entry_price += direction * spread[i] / 2

    edge = np.mean(trade_pnls) if trade_pnls else 0.0
    return total_pnl * TICK_VAL, trades, wins, trade_pnls, edge


# ── Strategy 2: Vol-Gated Transformer ──

def sim_vol_gated(mid, spread, preds, z_preds, rvol, thresh, vol_percentile,
                  hold_bars=600, cooldown=100,
                  spread_cost_ticks=0.0, comm_ticks=0.24):
    # HC #231(A): spread cost deleted — fill price already encodes side
    """
    Trade only when BOTH |z_pred| > thresh AND rvol is in top N% for the day.
    Direction from raw preds sign.
    """
    n = len(mid)
    abs_z = np.abs(z_preds)

    # Vol gate: only nonzero rvol values
    nonzero_rvol = rvol[rvol > 0]
    if len(nonzero_rvol) == 0:
        return 0.0, 0, 0, [], 0.0
    vol_cutoff = np.percentile(nonzero_rvol, (1.0 - vol_percentile) * 100)

    total_pnl = 0.0
    trades = 0
    wins = 0
    trade_pnls = []
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    cost_per_trade = spread_cost_ticks + comm_ticks

    for i in range(n):
        if in_pos:
            elapsed = i - entry_bar
            if elapsed >= hold_bars:
                curr_pnl = direction * (mid[i] - entry_price) / TICK
                pnl = curr_pnl - cost_per_trade
                total_pnl += pnl
                trade_pnls.append(pnl)
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i
        elif (i - last_exit >= cooldown
              and abs_z[i] > thresh
              and rvol[i] >= vol_cutoff):
            entry_price = mid[i]
            direction = 1 if preds[i] > 0 else -1
            entry_bar = i
            in_pos = True
            entry_price += direction * spread[i] / 2

    edge = np.mean(trade_pnls) if trade_pnls else 0.0
    return total_pnl * TICK_VAL, trades, wins, trade_pnls, edge


# ── Strategy 3: Hold Until Flip ──

def sim_hold_until_flip(mid, spread, preds, z_preds, thresh,
                        min_hold=100, max_hold=18000, cooldown=100,
                        spread_cost_ticks=0.0, comm_ticks=0.24):
    # HC #231(A): spread cost deleted — fill price already encodes side
    """
    Entry: when |z_pred| > thresh, enter in direction of pred.
    Exit: when raw pred changes sign (prediction flips) or max_hold reached.
    Min hold: 100 bars (10s) to avoid whipsaws.
    Uses z_preds for entry threshold, raw preds for direction and flip detection.
    """
    n = len(mid)
    abs_z = np.abs(z_preds)

    total_pnl = 0.0
    trades = 0
    wins = 0
    trade_pnls = []
    hold_times = []
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0
    cost_per_trade = spread_cost_ticks + comm_ticks

    for i in range(n):
        if in_pos:
            elapsed = i - entry_bar

            # Check exit conditions
            do_exit = False

            # Max hold safety
            if elapsed >= max_hold:
                do_exit = True
            # After min hold, check for sign flip on raw preds
            elif elapsed >= min_hold:
                if direction == 1 and preds[i] < 0:
                    do_exit = True
                elif direction == -1 and preds[i] > 0:
                    do_exit = True

            if do_exit:
                curr_pnl = direction * (mid[i] - entry_price) / TICK
                pnl = curr_pnl - cost_per_trade
                total_pnl += pnl
                trade_pnls.append(pnl)
                hold_times.append(elapsed)
                trades += 1
                if pnl > 0:
                    wins += 1
                in_pos = False
                last_exit = i

        elif i - last_exit >= cooldown and abs_z[i] > thresh:
            entry_price = mid[i]
            direction = 1 if preds[i] > 0 else -1
            entry_bar = i
            in_pos = True
            entry_price += direction * spread[i] / 2

    avg_hold = np.mean(hold_times) if hold_times else 0.0
    edge = np.mean(trade_pnls) if trade_pnls else 0.0
    return total_pnl * TICK_VAL, trades, wins, trade_pnls, edge, avg_hold, hold_times


def get_month(date_str):
    return date_str[:7]


def monthly_breakdown(dates, day_pnls):
    monthly = defaultdict(lambda: {'pnl': 0.0, 'days': 0})
    for date, pnl in zip(dates, day_pnls):
        m = get_month(date)
        monthly[m]['pnl'] += pnl
        monthly[m]['days'] += 1
    return dict(monthly)


def run_strategy_1(days, all_dates, train_dates, test_dates):
    """Strategy 1: Confidence Gating."""
    print("\n" + "=" * 70)
    print("STRATEGY 1: CONFIDENCE GATING (Top N% most extreme predictions)")
    print("=" * 70)

    percentiles = [0.01, 0.02, 0.05, 0.10, 0.20]
    hold_bars = 600  # 1 minute

    results = {}

    for cost_spread, cost_comm, cost_label in COST_PROFILES:
        print(f"\n--- Cost profile: {cost_label} (spread={cost_spread} + comm={cost_comm} = {cost_spread+cost_comm:.2f} ticks RT) ---")

        best_train_pnl = -np.inf
        best_pct = None
        config_results = []

        for pct in percentiles:
            # Train phase
            train_pnl = 0.0
            train_trades = 0
            train_day_pnls = []

            for date in train_dates:
                d = days[date]
                pnl, trades, wins, _, edge = sim_confidence_gating(
                    d['mid'], d['spread'], d['preds'], pct, hold_bars,
                    cooldown=100, spread_cost_ticks=cost_spread, comm_ticks=cost_comm
                )
                train_pnl += pnl
                train_trades += trades
                train_day_pnls.append(pnl)

            avg_daily = np.mean(train_day_pnls) if train_day_pnls else 0
            std_daily = np.std(train_day_pnls) if train_day_pnls else 1
            sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0
            trades_per_day = train_trades / len(train_dates) if train_dates else 0

            print(f"  TRAIN top {pct*100:5.1f}%: ${train_pnl:+9,.0f} | {trades_per_day:5.1f} trades/day | Sharpe={sharpe:+.2f}")

            config_results.append({
                'percentile': pct,
                'train_pnl': round(train_pnl),
                'train_trades': train_trades,
                'train_sharpe': round(sharpe, 2),
                'trades_per_day': round(trades_per_day, 1),
            })

            if train_pnl > best_train_pnl:
                best_train_pnl = train_pnl
                best_pct = pct

        # OOS with best config
        print(f"\n  BEST TRAIN: top {best_pct*100:.1f}% (${best_train_pnl:+,.0f})")
        print(f"  Testing OOS with fixed top {best_pct*100:.1f}%...")

        oos_pnl = 0.0
        oos_trades = 0
        oos_wins = 0
        oos_day_pnls = []
        all_edges = []

        for date in test_dates:
            d = days[date]
            pnl, trades, wins, trade_pnls, edge = sim_confidence_gating(
                d['mid'], d['spread'], d['preds'], best_pct, hold_bars,
                cooldown=100, spread_cost_ticks=cost_spread, comm_ticks=cost_comm
            )
            oos_pnl += pnl
            oos_trades += trades
            oos_wins += wins
            oos_day_pnls.append(pnl)
            if edge != 0:
                all_edges.append(edge)

        avg_oos = np.mean(oos_day_pnls) if oos_day_pnls else 0
        std_oos = np.std(oos_day_pnls) if oos_day_pnls else 1
        sharpe_oos = (avg_oos / std_oos * np.sqrt(252)) if std_oos > 0 else 0
        wr = oos_wins / oos_trades if oos_trades > 0 else 0
        avg_edge = np.mean(all_edges) if all_edges else 0
        trades_per_day = oos_trades / len(test_dates) if test_dates else 0
        monthly = monthly_breakdown(test_dates, oos_day_pnls)

        print(f"\n  OOS RESULT ({cost_label}):")
        print(f"    PnL: ${oos_pnl:+,.0f}")
        print(f"    Sharpe: {sharpe_oos:+.2f}")
        print(f"    Trades: {oos_trades} ({trades_per_day:.1f}/day)")
        print(f"    Win rate: {wr:.1%}")
        print(f"    Avg edge/trade: {avg_edge:+.2f} ticks")
        print(f"    Monthly:")
        for m in sorted(monthly.keys()):
            print(f"      {m}: ${monthly[m]['pnl']:+8,.0f} ({monthly[m]['days']} days)")

        results[cost_label] = {
            'best_percentile': best_pct,
            'best_train_pnl': round(best_train_pnl),
            'oos_pnl': round(oos_pnl),
            'oos_sharpe': round(sharpe_oos, 2),
            'oos_trades': oos_trades,
            'oos_trades_per_day': round(trades_per_day, 1),
            'oos_win_rate': round(wr, 3),
            'oos_avg_edge_ticks': round(avg_edge, 3),
            'monthly': {m: {'pnl': round(v['pnl']), 'days': v['days']} for m, v in monthly.items()},
            'config_results': config_results,
        }

    return results


def run_strategy_2(days, all_dates, train_dates, test_dates, rvol_cache):
    """Strategy 2: Vol-Gated Transformer."""
    print("\n" + "=" * 70)
    print("STRATEGY 2: VOL-GATED TRANSFORMER")
    print("=" * 70)

    thresholds = [1.0, 2.0, 3.0]
    vol_gates = [0.10, 0.20, 0.30]  # top 10%, 20%, 30%
    hold_bars = 600

    results = {}

    for cost_spread, cost_comm, cost_label in COST_PROFILES:
        print(f"\n--- Cost profile: {cost_label} ({cost_spread+cost_comm:.2f} ticks RT) ---")

        best_train_pnl = -np.inf
        best_config = None
        config_results = []

        for thresh in thresholds:
            for vol_gate in vol_gates:
                train_pnl = 0.0
                train_trades = 0

                for date in train_dates:
                    d = days[date]
                    rvol = rvol_cache[date]
                    pnl, trades, wins, _, edge = sim_vol_gated(
                        d['mid'], d['spread'], d['preds'], d['z_preds'], rvol,
                        thresh, vol_gate, hold_bars,
                        cooldown=100, spread_cost_ticks=cost_spread, comm_ticks=cost_comm
                    )
                    train_pnl += pnl
                    train_trades += trades

                trades_per_day = train_trades / len(train_dates) if train_dates else 0
                print(f"  TRAIN t={thresh:.1f} vol_top={vol_gate*100:.0f}%: ${train_pnl:+9,.0f} | {trades_per_day:5.1f} trades/day")

                config_results.append({
                    'thresh': thresh, 'vol_gate': vol_gate,
                    'train_pnl': round(train_pnl), 'train_trades': train_trades,
                })

                if train_pnl > best_train_pnl:
                    best_train_pnl = train_pnl
                    best_config = (thresh, vol_gate)

        b_thresh, b_vol = best_config
        print(f"\n  BEST TRAIN: t={b_thresh}, vol_top={b_vol*100:.0f}% (${best_train_pnl:+,.0f})")

        # OOS
        oos_pnl = 0.0
        oos_trades = 0
        oos_wins = 0
        oos_day_pnls = []
        all_edges = []

        for date in test_dates:
            d = days[date]
            rvol = rvol_cache[date]
            pnl, trades, wins, trade_pnls, edge = sim_vol_gated(
                d['mid'], d['spread'], d['preds'], d['z_preds'], rvol,
                b_thresh, b_vol, hold_bars,
                cooldown=100, spread_cost_ticks=cost_spread, comm_ticks=cost_comm
            )
            oos_pnl += pnl
            oos_trades += trades
            oos_wins += wins
            oos_day_pnls.append(pnl)
            if edge != 0:
                all_edges.append(edge)

        avg_oos = np.mean(oos_day_pnls) if oos_day_pnls else 0
        std_oos = np.std(oos_day_pnls) if oos_day_pnls else 1
        sharpe_oos = (avg_oos / std_oos * np.sqrt(252)) if std_oos > 0 else 0
        wr = oos_wins / oos_trades if oos_trades > 0 else 0
        avg_edge = np.mean(all_edges) if all_edges else 0
        trades_per_day = oos_trades / len(test_dates) if test_dates else 0
        monthly = monthly_breakdown(test_dates, oos_day_pnls)

        print(f"\n  OOS RESULT ({cost_label}):")
        print(f"    PnL: ${oos_pnl:+,.0f}")
        print(f"    Sharpe: {sharpe_oos:+.2f}")
        print(f"    Trades: {oos_trades} ({trades_per_day:.1f}/day)")
        print(f"    Win rate: {wr:.1%}")
        print(f"    Avg edge/trade: {avg_edge:+.2f} ticks")
        print(f"    Monthly:")
        for m in sorted(monthly.keys()):
            print(f"      {m}: ${monthly[m]['pnl']:+8,.0f} ({monthly[m]['days']} days)")

        results[cost_label] = {
            'best_thresh': b_thresh,
            'best_vol_gate': b_vol,
            'best_train_pnl': round(best_train_pnl),
            'oos_pnl': round(oos_pnl),
            'oos_sharpe': round(sharpe_oos, 2),
            'oos_trades': oos_trades,
            'oos_trades_per_day': round(trades_per_day, 1),
            'oos_win_rate': round(wr, 3),
            'oos_avg_edge_ticks': round(avg_edge, 3),
            'monthly': {m: {'pnl': round(v['pnl']), 'days': v['days']} for m, v in monthly.items()},
            'config_results': config_results,
        }

    return results


def run_strategy_3(days, all_dates, train_dates, test_dates):
    """Strategy 3: Hold Until Flip (THE KEY TEST)."""
    print("\n" + "=" * 70)
    print("STRATEGY 3: HOLD UNTIL PREDICTION FLIPS (THE KEY TEST)")
    print("=" * 70)

    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    min_hold = 100   # 10 seconds
    max_hold = 18000  # 30 minutes
    cooldown = 100

    results = {}

    for cost_spread, cost_comm, cost_label in COST_PROFILES:
        print(f"\n--- Cost profile: {cost_label} ({cost_spread+cost_comm:.2f} ticks RT) ---")

        best_train_pnl = -np.inf
        best_thresh = None
        config_results = []

        for thresh in thresholds:
            train_pnl = 0.0
            train_trades = 0
            train_day_pnls = []
            all_hold_times = []

            for date in train_dates:
                d = days[date]
                pnl, trades, wins, trade_pnls, edge, avg_hold, hold_times = sim_hold_until_flip(
                    d['mid'], d['spread'], d['preds'], d['z_preds'], thresh,
                    min_hold=min_hold, max_hold=max_hold, cooldown=cooldown,
                    spread_cost_ticks=cost_spread, comm_ticks=cost_comm
                )
                train_pnl += pnl
                train_trades += trades
                train_day_pnls.append(pnl)
                all_hold_times.extend(hold_times)

            trades_per_day = train_trades / len(train_dates) if train_dates else 0
            avg_hold_s = np.mean(all_hold_times) / BARS_PER_SEC if all_hold_times else 0
            med_hold_s = np.median(all_hold_times) / BARS_PER_SEC if all_hold_times else 0

            avg_daily = np.mean(train_day_pnls) if train_day_pnls else 0
            std_daily = np.std(train_day_pnls) if train_day_pnls else 1
            sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

            print(f"  TRAIN t={thresh:.1f}: ${train_pnl:+9,.0f} | {trades_per_day:5.1f} trades/day | "
                  f"Sharpe={sharpe:+.2f} | avg_hold={avg_hold_s:.0f}s med={med_hold_s:.0f}s")

            config_results.append({
                'thresh': thresh,
                'train_pnl': round(train_pnl),
                'train_trades': train_trades,
                'train_sharpe': round(sharpe, 2),
                'trades_per_day': round(trades_per_day, 1),
                'avg_hold_s': round(avg_hold_s, 1),
                'med_hold_s': round(med_hold_s, 1),
            })

            if train_pnl > best_train_pnl:
                best_train_pnl = train_pnl
                best_thresh = thresh

        print(f"\n  BEST TRAIN: t={best_thresh} (${best_train_pnl:+,.0f})")
        print(f"  Testing ALL thresholds OOS (for comparison)...")

        # OOS: test ALL thresholds (report best-train fixed, but show all)
        oos_by_thresh = {}
        for thresh in thresholds:
            oos_pnl = 0.0
            oos_trades = 0
            oos_wins = 0
            oos_day_pnls = []
            all_edges = []
            all_hold_times = []
            all_trade_pnls = []

            for date in test_dates:
                d = days[date]
                pnl, trades, wins, trade_pnls, edge, avg_hold, hold_times = sim_hold_until_flip(
                    d['mid'], d['spread'], d['preds'], d['z_preds'], thresh,
                    min_hold=min_hold, max_hold=max_hold, cooldown=cooldown,
                    spread_cost_ticks=cost_spread, comm_ticks=cost_comm
                )
                oos_pnl += pnl
                oos_trades += trades
                oos_wins += wins
                oos_day_pnls.append(pnl)
                all_hold_times.extend(hold_times)
                all_trade_pnls.extend(trade_pnls)
                if edge != 0:
                    all_edges.append(edge)

            avg_oos = np.mean(oos_day_pnls) if oos_day_pnls else 0
            std_oos = np.std(oos_day_pnls) if oos_day_pnls else 1
            sharpe_oos = (avg_oos / std_oos * np.sqrt(252)) if std_oos > 0 else 0
            wr = oos_wins / oos_trades if oos_trades > 0 else 0
            avg_edge = np.mean(all_trade_pnls) if all_trade_pnls else 0
            trades_per_day = oos_trades / len(test_dates) if test_dates else 0
            avg_hold_s = np.mean(all_hold_times) / BARS_PER_SEC if all_hold_times else 0
            med_hold_s = np.median(all_hold_times) / BARS_PER_SEC if all_hold_times else 0
            monthly = monthly_breakdown(test_dates, oos_day_pnls)

            is_best = (thresh == best_thresh)
            marker = " <<<< BEST TRAIN" if is_best else ""
            print(f"\n  OOS t={thresh:.1f} ({cost_label}):{marker}")
            print(f"    PnL: ${oos_pnl:+,.0f}")
            print(f"    Sharpe: {sharpe_oos:+.2f}")
            print(f"    Trades: {oos_trades} ({trades_per_day:.1f}/day)")
            print(f"    Win rate: {wr:.1%}")
            print(f"    Avg edge/trade: {avg_edge:+.3f} ticks")
            print(f"    Avg hold: {avg_hold_s:.0f}s | Median: {med_hold_s:.0f}s")
            print(f"    Monthly:")
            for m in sorted(monthly.keys()):
                print(f"      {m}: ${monthly[m]['pnl']:+8,.0f} ({monthly[m]['days']} days)")

            oos_by_thresh[thresh] = {
                'oos_pnl': round(oos_pnl),
                'oos_sharpe': round(sharpe_oos, 2),
                'oos_trades': oos_trades,
                'oos_trades_per_day': round(trades_per_day, 1),
                'oos_win_rate': round(wr, 3),
                'oos_avg_edge_ticks': round(avg_edge, 3),
                'avg_hold_s': round(avg_hold_s, 1),
                'med_hold_s': round(med_hold_s, 1),
                'monthly': {m: {'pnl': round(v['pnl']), 'days': v['days']} for m, v in monthly.items()},
                'is_best_train': is_best,
            }

        results[cost_label] = {
            'best_train_thresh': best_thresh,
            'best_train_pnl': round(best_train_pnl),
            'oos_by_threshold': {str(k): v for k, v in oos_by_thresh.items()},
            'config_results': config_results,
        }

    return results


def both_halves_test(days, all_dates):
    """Test Strategy 3 on both halves independently (no optimization)."""
    print("\n" + "=" * 70)
    print("BOTH-HALVES TEST (Strategy 3: Hold Until Flip)")
    print("=" * 70)

    mid_point = len(all_dates) // 2
    half1 = all_dates[:mid_point]
    half2 = all_dates[mid_point:]
    thresh = 2.0  # Fixed middle-of-road threshold
    results = {}

    for cost_spread, cost_comm, cost_label in COST_PROFILES:
        for label, dates in [("Half 1", half1), ("Half 2", half2)]:
            total_pnl = 0.0
            total_trades = 0
            total_wins = 0
            day_pnls = []

            for date in dates:
                d = days[date]
                pnl, trades, wins, _, edge, avg_hold, _ = sim_hold_until_flip(
                    d['mid'], d['spread'], d['preds'], d['z_preds'], thresh,
                    min_hold=100, max_hold=18000, cooldown=100,
                    spread_cost_ticks=cost_spread, comm_ticks=cost_comm
                )
                total_pnl += pnl
                total_trades += trades
                total_wins += wins
                day_pnls.append(pnl)

            avg_d = np.mean(day_pnls) if day_pnls else 0
            std_d = np.std(day_pnls) if day_pnls else 1
            sharpe = (avg_d / std_d * np.sqrt(252)) if std_d > 0 else 0
            wr = total_wins / total_trades if total_trades > 0 else 0
            pct_pos = np.mean([p > 0 for p in day_pnls]) * 100 if day_pnls else 0

            print(f"  {label} ({cost_label}): ${total_pnl:+,.0f} | {total_trades} trades | "
                  f"Sharpe={sharpe:+.2f} | WR={wr:.1%} | {pct_pos:.0f}% days+")

            results[f'{label}_{cost_label}'] = {
                'pnl': round(total_pnl),
                'trades': total_trades,
                'sharpe': round(sharpe, 2),
                'win_rate': round(wr, 3),
                'pct_positive_days': round(pct_pos, 1),
                'dates': f'{dates[0]} to {dates[-1]}',
            }

    return results


def print_comparison_table(s1_results, s2_results, s3_results):
    """Print a clear comparison table of all strategies."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)

    header = f"{'Strategy':<35} {'ES PnL':>10} {'SPY PnL':>10} {'Zero PnL':>10} {'ES Sharpe':>10} {'Trades/d':>10}"
    print(header)
    print("-" * len(header))

    # Strategy 1
    es = s1_results.get('ES_futures', {})
    spy = s1_results.get('SPY_500sh', {})
    zc = s1_results.get('zero_cost', {})
    print(f"{'S1: Confidence Gate':<35} "
          f"${es.get('oos_pnl',0):>+9,} "
          f"${spy.get('oos_pnl',0):>+9,} "
          f"${zc.get('oos_pnl',0):>+9,} "
          f"{es.get('oos_sharpe',0):>+10.2f} "
          f"{es.get('oos_trades_per_day',0):>10.1f}")

    # Strategy 2
    es = s2_results.get('ES_futures', {})
    spy = s2_results.get('SPY_500sh', {})
    zc = s2_results.get('zero_cost', {})
    print(f"{'S2: Vol-Gated':<35} "
          f"${es.get('oos_pnl',0):>+9,} "
          f"${spy.get('oos_pnl',0):>+9,} "
          f"${zc.get('oos_pnl',0):>+9,} "
          f"{es.get('oos_sharpe',0):>+10.2f} "
          f"{es.get('oos_trades_per_day',0):>10.1f}")

    # Strategy 3: show best-train threshold
    for cost_label in ['ES_futures', 'SPY_500sh', 'zero_cost']:
        if cost_label in s3_results:
            best_t = s3_results[cost_label]['best_train_thresh']
            break
    else:
        best_t = 2.0

    es = s3_results.get('ES_futures', {}).get('oos_by_threshold', {}).get(str(best_t), {})
    spy = s3_results.get('SPY_500sh', {}).get('oos_by_threshold', {}).get(str(best_t), {})
    zc = s3_results.get('zero_cost', {}).get('oos_by_threshold', {}).get(str(best_t), {})
    print(f"{'S3: Hold-Until-Flip (best)':<35} "
          f"${es.get('oos_pnl',0):>+9,} "
          f"${spy.get('oos_pnl',0):>+9,} "
          f"${zc.get('oos_pnl',0):>+9,} "
          f"{es.get('oos_sharpe',0):>+10.2f} "
          f"{es.get('oos_trades_per_day',0):>10.1f}")

    # Also show all S3 thresholds at ES cost
    print(f"\n{'--- Strategy 3 All Thresholds (ES costs) ---':^80}")
    print(f"{'Threshold':<12} {'PnL':>10} {'Sharpe':>8} {'Trades/d':>10} {'WR':>8} {'Edge':>8} {'Avg Hold':>10}")
    print("-" * 68)
    s3_es = s3_results.get('ES_futures', {}).get('oos_by_threshold', {})
    for t in sorted(s3_es.keys(), key=float):
        r = s3_es[t]
        print(f"  t={float(t):.1f}      "
              f"${r['oos_pnl']:>+9,} "
              f"{r['oos_sharpe']:>+8.2f} "
              f"{r['oos_trades_per_day']:>10.1f} "
              f"{r['oos_win_rate']:>7.1%} "
              f"{r['oos_avg_edge_ticks']:>+8.3f} "
              f"{r['avg_hold_s']:>8.0f}s")

    # Profitability check
    print(f"\n{'--- PROFITABILITY CHECK ---':^80}")
    profitable = []
    for name, res in [("S1:ConfGate", s1_results), ("S2:VolGate", s2_results)]:
        for cl in ['ES_futures', 'SPY_500sh']:
            r = res.get(cl, {})
            if r.get('oos_pnl', 0) > 0:
                profitable.append(f"{name} @ {cl}: ${r['oos_pnl']:+,}")

    for cl in ['ES_futures', 'SPY_500sh']:
        s3_data = s3_results.get(cl, {}).get('oos_by_threshold', {})
        for t, r in s3_data.items():
            if r.get('oos_pnl', 0) > 0:
                profitable.append(f"S3:Flip(t={t}) @ {cl}: ${r['oos_pnl']:+,}")

    if profitable:
        print("  PROFITABLE CONFIGS:")
        for p in profitable:
            print(f"    >> {p}")
    else:
        print("  NO PROFITABLE CONFIGS AT ES OR SPY COSTS.")


def main():
    t0 = time.time()
    print("=" * 70)
    print("ADVANCED STRATEGY TEST ON EVENT TRANSFORMER OOS PREDICTIONS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    days, all_dates = load_data()

    train_dates = all_dates[:TRAIN_DAYS]
    test_dates = all_dates[TRAIN_DAYS:]
    print(f"\nTRAIN: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"TEST:  {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    # Quick signal quality check
    print("\n--- Signal Quality Check ---")
    ics = []
    for date in all_dates:
        d = days[date]
        ic = np.corrcoef(d['preds'], d['targets'])[0, 1]
        ics.append(ic)
    print(f"  Overall IC: mean={np.mean(ics):.4f}, median={np.median(ics):.4f}")
    print(f"  Train IC:  mean={np.mean(ics[:TRAIN_DAYS]):.4f}")
    print(f"  Test IC:   mean={np.mean(ics[TRAIN_DAYS:]):.4f}")
    print(f"  Pct positive IC: {np.mean([ic > 0 for ic in ics])*100:.1f}%")

    # Precompute trailing rvol for Strategy 2
    print("\nPrecomputing trailing realized vol (500-bar window)...")
    rvol_cache = {}
    for i, date in enumerate(all_dates):
        d = days[date]
        rvol_cache[date] = compute_trailing_rvol_fast(d['mid'], window=500)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(all_dates)} days processed")
    print(f"  Done. ({time.time()-t0:.1f}s elapsed)")

    # Run strategies
    s1_results = run_strategy_1(days, all_dates, train_dates, test_dates)
    print(f"\n[Strategy 1 complete. {time.time()-t0:.0f}s elapsed]")

    s2_results = run_strategy_2(days, all_dates, train_dates, test_dates, rvol_cache)
    print(f"\n[Strategy 2 complete. {time.time()-t0:.0f}s elapsed]")

    s3_results = run_strategy_3(days, all_dates, train_dates, test_dates)
    print(f"\n[Strategy 3 complete. {time.time()-t0:.0f}s elapsed]")

    # Both-halves test
    both_halves = both_halves_test(days, all_dates)

    # Final comparison
    print_comparison_table(s1_results, s2_results, s3_results)

    # Both halves summary
    print(f"\n{'--- BOTH-HALVES TEST (S3 @ t=2.0, fixed) ---':^80}")
    for k, v in both_halves.items():
        print(f"  {k}: ${v['pnl']:+,} | Sharpe={v['sharpe']:+.2f} | WR={v['win_rate']:.1%} | {v['dates']}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Total elapsed: {elapsed:.0f}s")
    print(f"{'='*70}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'advanced_strategy_test_{timestamp}.json'

    all_results = {
        'timestamp': timestamp,
        'n_days': len(all_dates),
        'train_days': TRAIN_DAYS,
        'test_days': len(test_dates),
        'train_range': f'{train_dates[0]} to {train_dates[-1]}',
        'test_range': f'{test_dates[0]} to {test_dates[-1]}',
        'signal_quality': {
            'overall_ic': round(float(np.mean(ics)), 4),
            'train_ic': round(float(np.mean(ics[:TRAIN_DAYS])), 4),
            'test_ic': round(float(np.mean(ics[TRAIN_DAYS:])), 4),
        },
        'strategy_1_confidence_gating': s1_results,
        'strategy_2_vol_gated': s2_results,
        'strategy_3_hold_until_flip': s3_results,
        'both_halves_test': both_halves,
        'elapsed_s': round(elapsed, 1),
    }

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
