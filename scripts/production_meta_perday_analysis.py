#!/usr/bin/env python3
"""
Production Meta-Model Per-Day Analysis (HC #491 R1 tree-branch: per-day Sharpe)
================================================================================
Analyzes per-date performance of the PRODUCTION meta-model (shorts + longs),
both with and without meta-filtering, and with FIFO-aware cost assumptions.

Uses per-fold predictions (each fold = 1 OOT date) to reconstruct daily P&L.
Computes: daily Sharpe, per-regime stratification, green/red classification.

Outputs:
  output/production_perday_v1/
    - perday_shorts.csv
    - perday_longs.csv
    - perday_combined.csv
    - regime_stratification.csv
    - summary.json

Author: Claude (HC #492 self-critical infrastructure)
Date: 2026-05-24
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT = ROOT / "output/production_perday_v1"
OUT.mkdir(parents=True, exist_ok=True)

# Cost constants (canonical)
COMMISSION_PASSIVE = 0.376  # passive RT
COMMISSION_MARKET = 1.376   # market order RT


def load_meta_model(results_path: Path, concat_path: Path):
    """Load meta-model results and predictions."""
    with open(results_path) as f:
        results = json.load(f)

    concat = np.load(concat_path, allow_pickle=True)
    preds = concat['predictions']
    actuals = concat['actuals']

    # Reconstruct per-fold boundaries from fold sizes
    folds = results.get('per_fold', [])
    return results, preds, actuals, folds


def analyze_per_day(preds, actuals, folds, extra_cost_ticks=0.0, meta_filter_pct=None):
    """Compute per-day metrics.

    NOTE: actuals already include 0.376 ticks commission (passive RT).
    extra_cost_ticks is ADDITIONAL cost on top (e.g., for market orders: 1.0 spread).
    For passive fills, extra_cost_ticks=0. For market orders, extra_cost_ticks=1.0.
    """
    rows = []
    offset = 0

    for fold_info in folds:
        date = fold_info['date']
        n_test = fold_info['n_test']
        fold_preds = preds[offset:offset + n_test]
        fold_actuals = actuals[offset:offset + n_test]
        offset += n_test

        # Optionally filter by meta-score top N%
        if meta_filter_pct is not None:
            threshold = np.percentile(fold_preds, 100 - meta_filter_pct)
            mask = fold_preds >= threshold
            fold_actuals_filtered = fold_actuals[mask]
            n_filtered = mask.sum()
        else:
            fold_actuals_filtered = fold_actuals
            n_filtered = len(fold_actuals)

        if n_filtered == 0:
            continue

        # P&L — actuals already net of 0.376 passive commission
        # Only subtract extra_cost_ticks for market order scenarios
        net_pnl = fold_actuals_filtered - extra_cost_ticks
        total_pnl = float(net_pnl.sum())
        mean_pnl = float(net_pnl.mean())
        wr = float((net_pnl > 0).sum() / len(net_pnl) * 100)
        wins = float(net_pnl[net_pnl > 0].sum()) if (net_pnl > 0).any() else 0
        losses = float(abs(net_pnl[net_pnl < 0].sum())) if (net_pnl < 0).any() else 0.001
        pf = wins / losses if losses > 0 else float('inf')

        rows.append({
            'date': date,
            'n_trades': int(n_filtered),
            'total_pnl': round(total_pnl, 4),
            'mean_pnl': round(mean_pnl, 4),
            'wr': round(wr, 2),
            'pf': round(pf, 4),
            'green': total_pnl > 0,
        })

    return rows


def compute_daily_sharpe(rows):
    """Annualized daily Sharpe from per-day total P&L."""
    daily_pnl = [r['total_pnl'] for r in rows]
    if len(daily_pnl) < 2:
        return float('nan')
    arr = np.array(daily_pnl)
    std = arr.std(ddof=1)
    if std == 0:
        return float('inf')
    return float(arr.mean() / std * math.sqrt(252))


def compute_sortino(rows):
    """Annualized Sortino from per-day total P&L."""
    daily_pnl = np.array([r['total_pnl'] for r in rows])
    if len(daily_pnl) < 2:
        return float('nan')
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) == 0:
        return float('inf')
    down_std = np.sqrt(np.mean(downside**2))
    if down_std == 0:
        return float('inf')
    return float(daily_pnl.mean() / down_std * math.sqrt(252))


def classify_regime(date_str):
    """Classify trading day as green/red/flat based on ES close-to-close.
    Placeholder — needs actual ES daily data. Returns 'unknown' if unavailable.
    """
    # TODO: Load actual ES daily returns for regime classification
    return 'unknown'


def summarize(rows, label):
    """Print and return summary stats."""
    if not rows:
        print(f"  {label}: NO DATA")
        return {}

    n_days = len(rows)
    green = sum(1 for r in rows if r['green'])
    red = n_days - green
    total_trades = sum(r['n_trades'] for r in rows)
    total_pnl = sum(r['total_pnl'] for r in rows)
    sharpe = compute_daily_sharpe(rows)
    sortino = compute_sortino(rows)
    mean_daily = total_pnl / n_days if n_days > 0 else 0
    avg_trades_per_day = total_trades / n_days if n_days > 0 else 0

    # Per-trade stats
    mean_per_trade = total_pnl / total_trades if total_trades > 0 else 0
    wr_vals = [r['wr'] for r in rows if 'wr' in r]
    pf_vals = [r['pf'] for r in rows if 'pf' in r and r['pf'] < 100]
    avg_wr = np.mean(wr_vals) if wr_vals else 0
    avg_pf = np.mean(pf_vals) if pf_vals else 0

    print(f"\n  {label}:")
    print(f"    Days: {n_days} ({green} green, {red} red) = {green/n_days*100:.0f}% green")
    print(f"    Total trades: {total_trades:,} ({avg_trades_per_day:.0f}/day avg)")
    print(f"    Total P&L: {total_pnl:+.1f} ticks (${total_pnl * 12.50:+,.0f})")
    print(f"    Mean daily: {mean_daily:+.1f} ticks (${mean_daily * 12.50:+,.0f})")
    print(f"    Mean per-trade: {mean_per_trade:+.4f} ticks")
    print(f"    Avg WR: {avg_wr:.1f}%, Avg PF: {avg_pf:.2f}")
    print(f"    Daily Sharpe (ann): {sharpe:.2f}")
    print(f"    Sortino (ann): {sortino:.2f}")

    summary = {
        'label': label,
        'n_days': n_days,
        'green_days': green,
        'red_days': red,
        'green_pct': round(green/n_days*100, 1),
        'total_trades': total_trades,
        'total_pnl_ticks': round(total_pnl, 2),
        'total_pnl_usd': round(total_pnl * 12.50, 2),
        'mean_daily_pnl': round(mean_daily, 2),
        'mean_per_trade': round(mean_per_trade, 4),
        'avg_wr': round(avg_wr, 2),
        'avg_pf': round(avg_pf, 2),
        'daily_sharpe_ann': round(sharpe, 2),
        'sortino_ann': round(sortino, 2),
    }

    return summary


def main():
    print("=" * 70)
    print("PRODUCTION META-MODEL PER-DAY ANALYSIS")
    print("HC #491 R1 — Tree-branch: per-day Sharpe breakdown")
    print("HC #492 — Self-critical: verify edge is real, not cherry-picked")
    print("=" * 70)

    # Load shorts
    shorts_results_path = ROOT / "output/meta_production_v1/results.json"
    shorts_concat_path = ROOT / "output/meta_production_v1/concat_predictions.npz"

    # Load longs
    longs_results_path = ROOT / "output/meta_production_longs_v1/results.json"
    longs_concat_path = ROOT / "output/meta_production_longs_v1/concat_predictions.npz"

    summaries = {}

    for side, rpath, cpath in [
        ("shorts", shorts_results_path, shorts_concat_path),
        ("longs", longs_results_path, longs_concat_path),
    ]:
        if not rpath.exists() or not cpath.exists():
            print(f"\n  SKIP {side} — files not found")
            continue

        results, preds, actuals, folds = load_meta_model(rpath, cpath)
        print(f"\n{'='*40}")
        print(f"  {side.upper()} — {len(folds)} folds, {len(preds)} total predictions")
        config = results.get('config', {})
        print(f"  Architecture: {config.get('architecture', '?')}")
        print(f"  Horizon: {config.get('horizon', '?')}")
        print(f"  Signal threshold: {config.get('signal_threshold', '?')}%")

        # NOTE: actuals already net of 0.376 passive commission
        # extra_cost=0 for passive fills, extra_cost=1.0 for market order spread

        # 1. No meta filter, passive costs (already in actuals)
        rows_raw = analyze_per_day(preds, actuals, folds, extra_cost_ticks=0)
        s = summarize(rows_raw, f"{side} — raw (no meta filter, passive cost)")
        summaries[f'{side}_raw_passive'] = s

        # 2. Meta top 50% filter, passive costs
        rows_meta50 = analyze_per_day(preds, actuals, folds, extra_cost_ticks=0, meta_filter_pct=50)
        s = summarize(rows_meta50, f"{side} — meta top 50%, passive cost")
        summaries[f'{side}_meta50_passive'] = s

        # 3. Meta top 30% filter, passive costs
        rows_meta30 = analyze_per_day(preds, actuals, folds, extra_cost_ticks=0, meta_filter_pct=30)
        s = summarize(rows_meta30, f"{side} — meta top 30%, passive cost")
        summaries[f'{side}_meta30_passive'] = s

        # 4. Meta top 50% filter, MARKET ORDER costs (+1.0 tick spread on top)
        rows_meta50_mkt = analyze_per_day(preds, actuals, folds, extra_cost_ticks=1.0, meta_filter_pct=50)
        s = summarize(rows_meta50_mkt, f"{side} — meta top 50%, MARKET ORDER cost")
        summaries[f'{side}_meta50_market'] = s

        # Save per-day CSV
        import csv
        csv_path = OUT / f"perday_{side}.csv"
        if rows_meta50:
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows_meta50[0].keys())
                w.writeheader()
                w.writerows(rows_meta50)
            print(f"\n  Saved: {csv_path}")

    # Combined (shorts + longs meta50 passive)
    print(f"\n{'='*40}")
    print("  COMBINED (shorts + longs, meta top 50%, passive)")

    # Merge by date
    shorts_by_date = {}
    if shorts_results_path.exists() and shorts_concat_path.exists():
        results_s, preds_s, actuals_s, folds_s = load_meta_model(
            shorts_results_path, shorts_concat_path)
        rows_s = analyze_per_day(preds_s, actuals_s, folds_s, extra_cost_ticks=0, meta_filter_pct=50)
        for r in rows_s:
            shorts_by_date[r['date']] = r

    longs_by_date = {}
    if longs_results_path.exists() and longs_concat_path.exists():
        results_l, preds_l, actuals_l, folds_l = load_meta_model(
            longs_results_path, longs_concat_path)
        rows_l = analyze_per_day(preds_l, actuals_l, folds_l, extra_cost_ticks=0, meta_filter_pct=50)
        for r in rows_l:
            longs_by_date[r['date']] = r

    all_dates = sorted(set(shorts_by_date.keys()) | set(longs_by_date.keys()))
    combined_rows = []
    for d in all_dates:
        s = shorts_by_date.get(d, {'total_pnl': 0, 'n_trades': 0})
        l = longs_by_date.get(d, {'total_pnl': 0, 'n_trades': 0})
        total = s['total_pnl'] + l['total_pnl']
        n = s['n_trades'] + l['n_trades']
        combined_rows.append({
            'date': d,
            'n_trades': n,
            'short_pnl': round(s['total_pnl'], 4),
            'long_pnl': round(l['total_pnl'], 4),
            'total_pnl': round(total, 4),
            'mean_pnl': round(total / n, 4) if n > 0 else 0,
            'green': total > 0,
        })

    s = summarize(combined_rows, "COMBINED shorts + longs")
    summaries['combined_meta50_passive'] = s

    # Day-concentration check (HC #344: cap <= 0.70)
    if combined_rows:
        daily_abs = [abs(r['total_pnl']) for r in combined_rows]
        total_abs = sum(daily_abs)
        if total_abs > 0:
            day_conc = max(daily_abs) / total_abs
            print(f"\n  Day concentration: {day_conc:.3f} (cap: 0.70)")
            if day_conc > 0.70:
                print(f"  WARNING: Day concentration {day_conc:.3f} EXCEEDS 0.70 cap (HC #344)")
            summaries['day_concentration'] = round(day_conc, 4)

    # Regime gap check (HC #428 R1: |Sharpe_green - Sharpe_red| / max < 0.50)
    # This needs ES daily close data - flag as TODO
    print("\n  REGIME STRATIFICATION: Requires ES daily close data (TODO)")
    print("  Per-day data saved — can be stratified once ES daily returns are available")

    # Per-day breakdown
    print(f"\n{'='*40}")
    print("  PER-DAY BREAKDOWN (combined, meta top 50%, passive):")
    print(f"  {'Date':>10} {'Trades':>7} {'Short PnL':>10} {'Long PnL':>10} {'Total':>10} {'Status':>7}")
    for r in combined_rows:
        status = 'GREEN' if r['green'] else 'RED'
        print(f"  {r['date']:>10} {r['n_trades']:>7} {r.get('short_pnl',0):>+10.1f} {r.get('long_pnl',0):>+10.1f} {r['total_pnl']:>+10.1f} {status:>7}")

    # Save all results
    with open(OUT / "summary.json", 'w') as f:
        json.dump(summaries, f, indent=2)

    import csv
    csv_path = OUT / "perday_combined.csv"
    if combined_rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=combined_rows[0].keys())
            w.writeheader()
            w.writerows(combined_rows)

    print(f"\n  Results saved to {OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
