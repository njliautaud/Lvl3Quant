#!/usr/bin/env python3
"""
Regime-Stratified Validation for Confluence Meta-Model v2
==========================================================
HC #428 compliance: per-day Sharpe/PF/WR, regime stratification (green/red/flat),
regime-imbalance rejection test, day-concentration cap (HC #344).

Loads per-fold predictions from oot_predictions.npz (sliced by n_test per fold),
reconstructs per-date P&L, classifies days by avg(labels_30s), and validates
each meta-filter level.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

# === PATHS ===
BASE = Path('/home/jupiter/Lvl3Quant')
META_DIR = BASE / 'output' / 'confluence_meta_v2'
PRED_DIR = BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = BASE / 'data' / 'processed' / 'mbo_events_smart_v3'

# === CONSTANTS (from CLAUDE.md) ===
COMMISSION_RT_TICKS = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0
SHORT_PERCENTILE = 3  # top 3%

# Regime thresholds: avg labels_30s in ticks
# NOTE: OOT window (Mar-Apr 2026) was range-bound, max avg_l30s ~0.77 ticks.
# Fixed ±1.0 thresholds leave all days as "flat". Use empirical tertile splits:
# bottom 33% of dates by avg_l30s = "red" (price drifted down = good for shorts)
# top 33% = "green" (bad for shorts), middle = "flat"
# This is set dynamically after loading all dates (see compute_regime_thresholds).
GREEN_THRESH = None  # set dynamically
RED_THRESH = None    # set dynamically

# HC #344: day concentration cap
DAY_CONC_CAP = 0.70

# HC #428: regime imbalance rejection
REGIME_IMBALANCE_THRESH = 0.50

# Meta-filter levels (percentile TOP of predictions = highest predicted P&L)
FILTER_LEVELS = [100, 50, 30, 20, 10]


def load_fold_data():
    """Load meta predictions and per-fold metadata."""
    with open(META_DIR / 'results.json') as f:
        results = json.load(f)

    data = np.load(META_DIR / 'oot_predictions.npz')
    predictions = data['predictions']  # meta-model predicted P&L
    actuals = data['actuals']          # realized P&L (ground truth)

    return results['per_fold'], predictions, actuals


def load_daily_labels30s(date_str):
    """Load avg labels_30s for a date (regime proxy)."""
    mbo_path = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not mbo_path.exists():
        return None

    mbo = np.load(mbo_path)
    pred_path = PRED_DIR / f'{date_str}_predictions.npz'
    if not pred_path.exists():
        return None

    pred_data = np.load(pred_path)
    n_windows = int(pred_data['n_windows'])
    stride = int(pred_data['stride'])
    window_size = int(pred_data['window_size'])

    l30s = mbo['labels_30s']
    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = len(l30s) - 1
    valid = indices <= max_idx
    indices = indices[valid]

    l30s_aligned = l30s[indices]
    valid_mask = ~np.isnan(l30s_aligned)
    if valid_mask.sum() == 0:
        return None

    return float(np.mean(l30s_aligned[valid_mask]))


def compute_regime_thresholds(all_avgs):
    """Compute empirical tertile thresholds from list of avg_l30s values."""
    valid = [a for a in all_avgs if a is not None]
    red_thresh = float(np.percentile(valid, 33))
    green_thresh = float(np.percentile(valid, 67))
    return green_thresh, red_thresh


def classify_regime(avg_l30s, green_thresh, red_thresh):
    """Classify day as green/red/flat using empirical tertile thresholds."""
    if avg_l30s is None:
        return 'unknown'
    if avg_l30s > green_thresh:
        return 'green'
    elif avg_l30s < red_thresh:
        return 'red'
    else:
        return 'flat'


def sharpe(pnl_series):
    """Annualized daily Sharpe (sqrt(252))."""
    if len(pnl_series) < 2:
        return np.nan
    mu = np.mean(pnl_series)
    sd = np.std(pnl_series, ddof=1)
    if sd < 1e-10:
        return np.nan
    return (mu / sd) * np.sqrt(252)


def profit_factor(pnl_series):
    """Profit factor: sum(wins) / abs(sum(losses))."""
    wins = sum(p for p in pnl_series if p > 0)
    losses = abs(sum(p for p in pnl_series if p < 0))
    if losses < 1e-10:
        return np.inf
    return wins / losses


def win_rate(pnl_series):
    if len(pnl_series) == 0:
        return np.nan
    return 100.0 * sum(1 for p in pnl_series if p > 0) / len(pnl_series)


def check_regime_imbalance(sharpe_g, sharpe_r):
    """HC #428 imbalance check. Returns (ratio, pass/fail)."""
    if np.isnan(sharpe_g) or np.isnan(sharpe_r):
        return np.nan, 'SKIP (insufficient data)'
    denom = max(abs(sharpe_g), abs(sharpe_r))
    if denom < 1e-10:
        return 0.0, 'PASS'
    ratio = abs(sharpe_g - sharpe_r) / denom
    status = 'PASS' if ratio <= REGIME_IMBALANCE_THRESH else 'FAIL (HC#428 REJECT)'
    return ratio, status


def day_concentration(day_pnls):
    """HC #344: max single-day P&L / total. Returns (ratio, pass/fail)."""
    total = sum(day_pnls)
    if abs(total) < 1e-10:
        return np.nan, 'SKIP'
    max_day = max(day_pnls)
    ratio = max_day / total if total > 0 else max_day / abs(total)
    status = 'PASS' if ratio <= DAY_CONC_CAP else 'FAIL (HC#344 REJECT)'
    return ratio, status


def run_validation():
    print("=" * 70)
    print("REGIME-STRATIFIED VALIDATION — Meta-Model v2 (HC #428 / HC #344)")
    print("=" * 70)

    per_fold, meta_preds, actuals = load_fold_data()

    # Build per-date slices
    per_date = []
    idx = 0
    for fold in per_fold:
        n = fold['n_test']
        date = fold['date']
        fold_preds = meta_preds[idx:idx + n]
        fold_actuals = actuals[idx:idx + n]
        idx += n
        per_date.append({
            'date': date,
            'meta_preds': fold_preds,
            'actuals': fold_actuals,
        })

    print(f"\nLoaded {len(per_date)} dates, {len(actuals)} total top-3% trades.")

    # Load regime classification for each date
    print("\nClassifying regimes via avg(labels_30s) — empirical tertile splits...")
    for entry in per_date:
        avg_l30s = load_daily_labels30s(entry['date'])
        entry['avg_l30s'] = avg_l30s

    # Compute thresholds from actual distribution
    all_avgs = [e['avg_l30s'] for e in per_date]
    green_thresh, red_thresh = compute_regime_thresholds(all_avgs)
    print(f"Tertile thresholds: RED < {red_thresh:.4f}, FLAT [{red_thresh:.4f}, {green_thresh:.4f}], GREEN > {green_thresh:.4f}")
    print(f"  (p33={red_thresh:.4f}, p67={green_thresh:.4f} of daily avg_l30s)")

    for entry in per_date:
        entry['regime'] = classify_regime(entry['avg_l30s'], green_thresh, red_thresh)

    regime_counts = {}
    for entry in per_date:
        r = entry['regime']
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"Regime breakdown: {regime_counts}")

    # Run for each filter level
    summary = []
    for top_pct in FILTER_LEVELS:
        label = f"Top {top_pct}%" if top_pct < 100 else "All (100%)"

        # Per-date P&L under this filter
        day_records = []
        all_trade_pnls = []

        for entry in per_date:
            preds = entry['meta_preds']
            acts = entry['actuals']

            if top_pct == 100:
                sel = np.ones(len(preds), dtype=bool)
            else:
                threshold = np.percentile(preds, 100 - top_pct)
                sel = preds >= threshold

            if sel.sum() == 0:
                continue

            day_pnl_arr = acts[sel]
            day_total_pnl = float(day_pnl_arr.mean())
            n_trades = int(sel.sum())

            day_records.append({
                'date': entry['date'],
                'regime': entry['regime'],
                'mean_pnl': day_total_pnl,
                'sum_pnl': float(day_pnl_arr.sum()),
                'n_trades': n_trades,
                'wr': win_rate(day_pnl_arr),
                'pf': profit_factor(day_pnl_arr),
            })
            all_trade_pnls.extend(day_pnl_arr.tolist())

        day_mean_pnls = [d['mean_pnl'] for d in day_records]
        day_sum_pnls = [d['sum_pnl'] for d in day_records]

        # Overall metrics
        overall_mean = np.mean(all_trade_pnls) if all_trade_pnls else np.nan
        overall_wr = win_rate(all_trade_pnls)
        overall_pf = profit_factor(all_trade_pnls)
        overall_sharpe = sharpe(day_mean_pnls)

        # Per-regime Sharpe (on per-day mean P&L series)
        for regime in ['green', 'red', 'flat']:
            regime_days = [d['mean_pnl'] for d in day_records if d['regime'] == regime]

        green_days = [d['mean_pnl'] for d in day_records if d['regime'] == 'green']
        red_days = [d['mean_pnl'] for d in day_records if d['regime'] == 'red']
        flat_days = [d['mean_pnl'] for d in day_records if d['regime'] == 'flat']

        sharpe_green = sharpe(green_days)
        sharpe_red = sharpe(red_days)
        sharpe_flat = sharpe(flat_days)

        # HC #428 imbalance
        imbalance_ratio, imbalance_status = check_regime_imbalance(sharpe_green, sharpe_red)

        # HC #344 day concentration
        positive_day_sums = [s for s in day_sum_pnls if s > 0]
        conc_ratio, conc_status = day_concentration(positive_day_sums) if positive_day_sums else (np.nan, 'SKIP')

        # Compile per-regime WR/PF
        green_acts = [p for entry in per_date if entry['regime'] == 'green'
                      for p in _get_filtered(entry, top_pct)]
        red_acts = [p for entry in per_date if entry['regime'] == 'red'
                    for p in _get_filtered(entry, top_pct)]
        flat_acts = [p for entry in per_date if entry['regime'] == 'flat'
                     for p in _get_filtered(entry, top_pct)]

        overall_verdict = 'PASS'
        if 'FAIL' in imbalance_status:
            overall_verdict = 'FAIL'
        if 'FAIL' in conc_status:
            overall_verdict = 'FAIL'

        summary.append({
            'label': label,
            'n_trades': len(all_trade_pnls),
            'n_days': len(day_records),
            'mean_pnl': overall_mean,
            'wr': overall_wr,
            'pf': overall_pf,
            'sharpe_all': overall_sharpe,
            'sharpe_green': sharpe_green,
            'sharpe_red': sharpe_red,
            'sharpe_flat': sharpe_flat,
            'imbalance_ratio': imbalance_ratio,
            'imbalance_status': imbalance_status,
            'conc_ratio': conc_ratio,
            'conc_status': conc_status,
            'overall_verdict': overall_verdict,
            'green_days': len(green_days),
            'red_days': len(red_days),
            'flat_days': len(flat_days),
            'wr_green': win_rate(green_acts),
            'wr_red': win_rate(red_acts),
            'wr_flat': win_rate(flat_acts),
            'pf_green': profit_factor(green_acts),
            'pf_red': profit_factor(red_acts),
            'pf_flat': profit_factor(flat_acts),
            'mean_pnl_green': np.mean(green_acts) if green_acts else np.nan,
            'mean_pnl_red': np.mean(red_acts) if red_acts else np.nan,
            'mean_pnl_flat': np.mean(flat_acts) if flat_acts else np.nan,
        })

    # Print results
    print("\n" + "=" * 70)
    print("OVERALL RESULTS BY FILTER LEVEL")
    print("=" * 70)
    print(f"{'Filter':<12} {'N':>6} {'Days':>5} {'AvgPnL':>8} {'WR%':>6} {'PF':>5} {'Sharpe':>7} {'Verdict'}")
    print("-" * 70)
    for s in summary:
        print(f"{s['label']:<12} {s['n_trades']:>6} {s['n_days']:>5} "
              f"{s['mean_pnl']:>+8.3f} {s['wr']:>6.1f} {s['pf']:>5.2f} "
              f"{s['sharpe_all']:>+7.3f}  {s['overall_verdict']}")

    print("\n" + "=" * 70)
    print("REGIME-STRATIFIED SHARPE (HC #428)")
    print(f"Regime thresholds (empirical tertile): GREEN>{green_thresh:.4f}, RED<{red_thresh:.4f} avg_l30s")
    print("=" * 70)
    header = f"{'Filter':<12} {'Sharpe_G':>9} {'Sharpe_R':>9} {'Sharpe_F':>9} {'Imbal':>7} {'Status'}"
    print(header)
    print("-" * 70)
    for s in summary:
        sg = f"{s['sharpe_green']:>+9.3f}" if not np.isnan(s['sharpe_green']) else "     N/A"
        sr = f"{s['sharpe_red']:>+9.3f}" if not np.isnan(s['sharpe_red']) else "     N/A"
        sf = f"{s['sharpe_flat']:>+9.3f}" if not np.isnan(s['sharpe_flat']) else "     N/A"
        ir = f"{s['imbalance_ratio']:>7.3f}" if not np.isnan(s['imbalance_ratio']) else "    N/A"
        print(f"{s['label']:<12} {sg} {sr} {sf} {ir}  {s['imbalance_status']}")

    print("\n" + "=" * 70)
    print("PER-REGIME MEAN P&L / WR% / PF")
    print("=" * 70)
    print(f"{'Filter':<12} {'G:AvgPnL':>9} {'G:WR':>6} {'G:PF':>5}  "
          f"{'R:AvgPnL':>9} {'R:WR':>6} {'R:PF':>5}  "
          f"{'F:AvgPnL':>9} {'F:WR':>6} {'F:PF':>5}")
    print("-" * 70)
    for s in summary:
        print(f"{s['label']:<12} "
              f"{s['mean_pnl_green']:>+9.3f} {s['wr_green']:>6.1f} {s['pf_green']:>5.2f}  "
              f"{s['mean_pnl_red']:>+9.3f} {s['wr_red']:>6.1f} {s['pf_red']:>5.2f}  "
              f"{s['mean_pnl_flat']:>+9.3f} {s['wr_flat']:>6.1f} {s['pf_flat']:>5.2f}")

    print("\n" + "=" * 70)
    print("DAY CONCENTRATION CAP (HC #344, cap=0.70)")
    print("=" * 70)
    print(f"{'Filter':<12} {'Conc':>7}  {'Status'}")
    print("-" * 40)
    for s in summary:
        cr = f"{s['conc_ratio']:>7.3f}" if not np.isnan(s['conc_ratio']) else "    N/A"
        print(f"{s['label']:<12} {cr}  {s['conc_status']}")

    print("\n" + "=" * 70)
    print("FINAL VERDICT SUMMARY")
    print("=" * 70)
    for s in summary:
        verdict_str = f"  *** HC #428 REGIME IMBALANCE: {s['imbalance_status']}" if 'FAIL' in s['imbalance_status'] else ""
        verdict_str2 = f"  *** HC #344 DAY CONC: {s['conc_status']}" if 'FAIL' in s['conc_status'] else ""
        print(f"{s['label']:<12}  {s['overall_verdict']}{verdict_str}{verdict_str2}")

    print("\nDone.")
    return summary


def _get_filtered(entry, top_pct):
    """Get filtered actuals for a date at given top_pct filter level."""
    preds = entry['meta_preds']
    acts = entry['actuals']
    if top_pct == 100:
        return acts.tolist()
    threshold = np.percentile(preds, 100 - top_pct)
    sel = preds >= threshold
    return acts[sel].tolist()


if __name__ == '__main__':
    run_validation()
