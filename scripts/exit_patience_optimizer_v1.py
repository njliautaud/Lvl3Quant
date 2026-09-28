#!/usr/bin/env python3
"""
Exit Patience Optimizer v1
==========================
Finds optimal patience window for passive exit after 5s hold expires.

Strategy:
  1. Signal fires -> passive sell limit at ask (entry)
  2. Hold 5s. Stop at 2-tick adverse -> market exit immediately.
  3. After 5s: place passive buy limit at bid (1-tick profit target).
  4. Wait patience_window seconds for passive fill.
  5. If filled -> P&L = +1 tick - 0.376 commission = +0.624 ticks (fixed).
     If NOT filled -> market exit at current mid:
       P&L = (-label_at_total_hold) - 0.376 commission - 1.0 spread

Also sweeps breakeven exit (passive at entry price, higher fill rate).

Labels are in TICKS (verified: std ~2.5 for 1s, ~5.5 for 5s).
Short P&L = -label (positive when price drops).
Stop: label_1s >= 2.0 ticks.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# === CONSTANTS ===
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
STOP_LOSS_TOTAL = STOP_TICKS + STOP_SLIPPAGE  # 3 ticks realized on stop
SHORT_PERCENTILE = 3  # top 3% short signals

PRED_DIR = Path('/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2/')
MBO_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/')
OUT_DIR = Path('/home/jupiter/Lvl3Quant/output/exit_patience_v1/')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Patience windows to sweep (seconds after the 5s hold)
PATIENCE_WINDOWS = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60]

# Retrace rates: probability of 1-tick retrace within patience window
RETRACE_ANCHORS_1TICK = {
    0: 0.00, 1: 0.25, 3: 0.45, 5: 0.63,
    10: 0.73, 25: 0.84, 55: 0.88,
}

# Breakeven retrace (higher fill rates)
RETRACE_ANCHORS_BREAKEVEN = {
    0: 0.00, 1: 0.35, 3: 0.55, 5: 0.72,
    10: 0.80, 25: 0.89, 55: 0.93,
}


def interpolate_retrace_rate(patience_sec, anchors):
    """Linearly interpolate retrace rate from anchor points."""
    sorted_pts = sorted(anchors.items())
    if patience_sec <= sorted_pts[0][0]:
        return sorted_pts[0][1]
    if patience_sec >= sorted_pts[-1][0]:
        return min(sorted_pts[-1][1], 0.95)
    for i in range(len(sorted_pts) - 1):
        x0, y0 = sorted_pts[i]
        x1, y1 = sorted_pts[i + 1]
        if x0 <= patience_sec <= x1:
            frac = (patience_sec - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return sorted_pts[-1][1]


def interpolate_label(label_1s, label_5s, label_10s, label_30s, total_hold_sec):
    """Interpolate label at arbitrary horizon from available horizons."""
    if total_hold_sec <= 1:
        return label_1s.copy()
    elif total_hold_sec <= 5:
        frac = (total_hold_sec - 1) / 4.0
        return label_1s + frac * (label_5s - label_1s)
    elif total_hold_sec <= 10:
        frac = (total_hold_sec - 5) / 5.0
        return label_5s + frac * (label_10s - label_5s)
    elif total_hold_sec <= 30:
        frac = (total_hold_sec - 10) / 20.0
        return label_10s + frac * (label_30s - label_10s)
    else:
        # Extrapolate from 10->30 slope
        slope = (label_30s - label_10s) / 20.0
        return label_30s + slope * (total_hold_sec - 30)


def load_day_labels(pred_file):
    """Load one day: predictions + aligned labels only (memory efficient)."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str = str(pred_data['date'])
    preds = pred_data['predictions']  # (n_windows, 3)
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not mbo_file.exists():
        return None

    # Load only label arrays (not full events)
    mbo = np.load(mbo_file, allow_pickle=True)
    l1s = mbo['labels_1s']
    l5s = mbo['labels_5s']
    l10s = mbo['labels_10s']
    l30s = mbo['labels_30s']

    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(l1s), len(l5s), len(l10s), len(l30s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    # Extract aligned labels
    al1 = l1s[indices]
    al5 = l5s[indices]
    al10 = l10s[indices]
    al30 = l30s[indices]

    # Remove NaN rows
    all_valid = ~(np.isnan(al1) | np.isnan(al5) | np.isnan(al10) | np.isnan(al30))
    preds = preds[all_valid]
    al1 = al1[all_valid]
    al5 = al5[all_valid]
    al10 = al10[all_valid]
    al30 = al30[all_valid]

    if len(preds) == 0:
        return None

    return {
        'date': date_str,
        'pred_5s': preds[:, 1],
        'label_1s': al1,
        'label_5s': al5,
        'label_10s': al10,
        'label_30s': al30,
    }


def simulate_patience_day(day, patience_sec, exit_mode='1tick'):
    """Simulate exit patience for one day (pre-extracted labels)."""
    pred_5s = day['pred_5s']
    if len(pred_5s) == 0:
        return None

    # Top 3% short signals
    threshold = np.percentile(pred_5s, SHORT_PERCENTILE)
    short_mask = pred_5s <= threshold
    n_trades = short_mask.sum()
    if n_trades == 0:
        return None

    label_1s = day['label_1s'][short_mask]
    label_5s = day['label_5s'][short_mask]
    label_10s = day['label_10s'][short_mask]
    label_30s = day['label_30s'][short_mask]

    total_hold = 5 + patience_sec
    label_total = interpolate_label(label_1s, label_5s, label_10s, label_30s, total_hold)

    anchors = RETRACE_ANCHORS_1TICK if exit_mode == '1tick' else RETRACE_ANCHORS_BREAKEVEN
    fill_rate = interpolate_retrace_rate(patience_sec, anchors)

    pnl = np.empty(n_trades)

    # Stopped trades
    stop_hit = label_1s >= STOP_TICKS
    pnl[stop_hit] = -STOP_LOSS_TOTAL - COMMISSION_RT_TICKS  # -3.376

    # Non-stopped trades
    non_stop = ~stop_hit
    n_non_stop = non_stop.sum()

    if n_non_stop > 0:
        non_stop_label_total = label_total[non_stop]
        n_passive = int(round(n_non_stop * fill_rate))

        # Sort: most favorable for shorts (most negative label) get passive fill first
        sort_idx = np.argsort(non_stop_label_total)
        non_stop_pnl = np.empty(n_non_stop)

        if exit_mode == '1tick':
            # Passive fill: +1 tick profit - commission
            non_stop_pnl[sort_idx[:n_passive]] = 1.0 - COMMISSION_RT_TICKS  # +0.624
        else:
            # Breakeven: just avoid spread cost
            non_stop_pnl[sort_idx[:n_passive]] = 0.0 - COMMISSION_RT_TICKS  # -0.376

        # Market exits
        if n_passive < n_non_stop:
            market_labels = non_stop_label_total[sort_idx[n_passive:]]
            non_stop_pnl[sort_idx[n_passive:]] = -market_labels - COMMISSION_RT_TICKS - SPREAD_TICKS

        pnl[non_stop] = non_stop_pnl

    return {
        'pnl': pnl,
        'n_trades': int(n_trades),
        'n_stopped': int(stop_hit.sum()),
        'n_passive_fill': int(round(n_non_stop * fill_rate)) if n_non_stop > 0 else 0,
        'n_market_exit': int(n_non_stop - round(n_non_stop * fill_rate)) if n_non_stop > 0 else 0,
    }


def compute_metrics(daily_results, fill_rate):
    """Compute aggregate metrics across all days."""
    all_pnl = np.concatenate([d['pnl'] for d in daily_results])
    total_trades = sum(d['n_trades'] for d in daily_results)
    total_stopped = sum(d['n_stopped'] for d in daily_results)
    total_passive = sum(d['n_passive_fill'] for d in daily_results)
    total_market = sum(d['n_market_exit'] for d in daily_results)
    n_days = len(daily_results)

    daily_pnl = np.array([d['pnl'].sum() for d in daily_results])
    daily_mean = daily_pnl.mean()
    daily_std = daily_pnl.std(ddof=1) if n_days > 1 else 1.0
    daily_downside = np.sqrt(np.mean(np.minimum(daily_pnl, 0) ** 2))

    mean_pnl = all_pnl.mean()
    wins = all_pnl > 0
    losses = all_pnl < 0
    wr = wins.sum() / len(all_pnl) * 100

    gross_win = all_pnl[wins].sum() if wins.any() else 0
    gross_loss = abs(all_pnl[losses].sum()) if losses.any() else 1
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    green_days = (daily_pnl > 0).sum()
    green_pct = green_days / n_days * 100

    sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0
    sortino = (daily_mean / daily_downside * np.sqrt(252)) if daily_downside > 0 else 0

    return {
        'mean_pnl_ticks': round(float(mean_pnl), 4),
        'total_pnl_ticks': round(float(all_pnl.sum()), 2),
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'win_rate_pct': round(float(wr), 1),
        'profit_factor': round(float(pf), 3),
        'green_day_pct': round(float(green_pct), 1),
        'n_days': n_days,
        'total_trades': int(total_trades),
        'trades_per_day': round(total_trades / n_days, 1),
        'total_stopped': int(total_stopped),
        'stop_rate_pct': round(total_stopped / total_trades * 100, 1),
        'total_passive_fill': int(total_passive),
        'total_market_exit': int(total_market),
        'fill_rate_pct': round(float(fill_rate * 100), 1),
        'mean_daily_pnl_ticks': round(float(daily_mean), 2),
        'pnl_per_trade_dollars': round(float(mean_pnl * 12.50), 2),
    }


def main():
    # Load all OOT prediction files (exclude stale backups)
    pred_files = sorted([
        f for f in PRED_DIR.glob('*_predictions.npz')
        if '_stale_' not in str(f)
    ])
    print(f"Found {len(pred_files)} prediction files", flush=True)

    # Load all days (memory-efficient: only labels, not full events)
    all_days = []
    for i, pf in enumerate(pred_files):
        if i % 10 == 0:
            print(f"  Loading day {i+1}/{len(pred_files)}...", flush=True)
        day = load_day_labels(pf)
        if day is not None:
            all_days.append(day)
    print(f"Loaded {len(all_days)} OOT days with valid data\n", flush=True)

    results = {}

    for exit_mode, mode_label in [('1tick', '1-TICK PROFIT TARGET'), ('breakeven', 'BREAKEVEN')]:
        print("=" * 85)
        print(f"EXIT MODE: {mode_label}")
        print("=" * 85)
        print(f"{'Patience':>10} {'Fill%':>6} {'MeanPnL':>8} {'WR%':>6} {'PF':>6} "
              f"{'Sharpe':>7} {'Sortino':>8} {'Green%':>7} {'$/trade':>8}")
        print("-" * 85)

        mode_key = f'{exit_mode}_target'
        results[mode_key] = {}
        anchors = RETRACE_ANCHORS_1TICK if exit_mode == '1tick' else RETRACE_ANCHORS_BREAKEVEN

        for pw in PATIENCE_WINDOWS:
            fill_rate = interpolate_retrace_rate(pw, anchors)
            daily_results = []
            for day in all_days:
                r = simulate_patience_day(day, pw, exit_mode=exit_mode)
                if r is not None:
                    daily_results.append(r)

            if not daily_results:
                continue

            m = compute_metrics(daily_results, fill_rate)
            results[mode_key][str(pw)] = m

            print(f"{pw:>8}s {m['fill_rate_pct']:>5.0f}% {m['mean_pnl_ticks']:>+7.3f} "
                  f"{m['win_rate_pct']:>5.1f} {m['profit_factor']:>5.2f} "
                  f"{m['sharpe']:>+6.2f} {m['sortino']:>+7.2f} {m['green_day_pct']:>6.1f} "
                  f"{m['pnl_per_trade_dollars']:>+7.2f}")

        print()

    # === Find optimal ===
    print("=" * 85)
    print("OPTIMAL PATIENCE WINDOWS")
    print("=" * 85)

    best_overall = None
    for mode_key in ['1tick_target', 'breakeven_target']:
        best_sharpe = -999
        best_pw = None
        for pw_str, m in results[mode_key].items():
            if m['sharpe'] > best_sharpe:
                best_sharpe = m['sharpe']
                best_pw = pw_str
        if best_pw:
            m = results[mode_key][best_pw]
            label = "1-tick profit" if '1tick' in mode_key else "Breakeven"
            print(f"\n{label} target:")
            print(f"  Best patience: {best_pw}s")
            print(f"  Sharpe: {m['sharpe']:+.2f}, Sortino: {m['sortino']:+.2f}")
            print(f"  Mean P&L: {m['mean_pnl_ticks']:+.3f} ticks ({m['pnl_per_trade_dollars']:+.2f}/trade)")
            print(f"  WR: {m['win_rate_pct']:.1f}%, PF: {m['profit_factor']:.2f}")
            print(f"  Fill rate: {m['fill_rate_pct']:.0f}%, Green days: {m['green_day_pct']:.1f}%")
            print(f"  Trades/day: {m['trades_per_day']:.0f}, Stop rate: {m['stop_rate_pct']:.1f}%")
            if best_overall is None or m['sharpe'] > best_overall[1]:
                best_overall = (mode_key, m['sharpe'], best_pw)

    # === Comparison ===
    print("\n" + "=" * 85)
    print("COMPARISON: ALL MARKET EXIT vs BEST PATIENCE")
    print("=" * 85)
    m0 = results['1tick_target'].get('0')
    if m0 and best_overall:
        mk, _, bp = best_overall
        mb = results[mk][bp]
        print(f"  All-market (patience=0): {m0['mean_pnl_ticks']:+.3f} ticks/trade, Sharpe {m0['sharpe']:+.2f}")
        print(f"  Best ({mk} patience={bp}s): {mb['mean_pnl_ticks']:+.3f} ticks/trade, Sharpe {mb['sharpe']:+.2f}")
        improvement = mb['mean_pnl_ticks'] - m0['mean_pnl_ticks']
        print(f"  Improvement: {improvement:+.3f} ticks/trade ({improvement * 12.50:+.2f}/trade)")

    # Save
    out_file = OUT_DIR / 'results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
