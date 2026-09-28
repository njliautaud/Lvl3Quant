#!/usr/bin/env python3
"""
48-date regime-stratified Sharpe analysis.
Closes HC #428 R1 validation gap: need 40+ OOT days, ALL regimes.

For each date: classify as green/red/flat from ES close-to-close,
compute per-day Sharpe/PF/WR for shorts (top 5%), then stratify.
Gate: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
"""

import numpy as np
import os
import json
from pathlib import Path
from collections import defaultdict

PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2")
ALPHA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v3")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/regime_strat_48d_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION_TICKS = 0.376  # passive both sides
THRESHOLDS = [0.03, 0.05, 0.10]  # top 3%, 5%, 10%

def classify_regime(alpha_path):
    """Classify day regime from alpha labels mean return."""
    try:
        d = np.load(alpha_path, allow_pickle=True)
        # Try different keys for returns
        for key in ['mean_5min_ret', 'ret_5min', 'close_ret', 'returns']:
            if key in d:
                ret = float(np.nanmean(d[key]))
                if ret > 0.0001:
                    return 'green', ret
                elif ret < -0.0001:
                    return 'red', ret
                else:
                    return 'flat', ret
        # Fallback: use labels if available
        if 'labels' in d:
            ret = float(np.nanmean(d['labels']))
            if ret > 0:
                return 'green', ret
            elif ret < 0:
                return 'red', ret
            else:
                return 'flat', ret
    except Exception as e:
        pass
    return 'unknown', 0.0

def analyze_date(pred_path, threshold_pct, horizon_idx=0):
    """Compute per-day stats for shorts at given threshold on given horizon."""
    d = np.load(pred_path, allow_pickle=True)
    preds = d['predictions'][:, horizon_idx]  # 1s horizon
    labels = d['labels'][:, horizon_idx]

    # Filter valid
    valid = ~np.isnan(preds) & ~np.isnan(labels)
    preds = preds[valid]
    labels = labels[valid]

    if len(preds) < 100:
        return None

    # Short signals: most negative predictions (model predicts price drop)
    cutoff = np.percentile(preds, threshold_pct * 100)  # bottom N%
    short_mask = preds <= cutoff

    if short_mask.sum() < 10:
        return None

    # For shorts: profit = -label (we're short, price drops = profit)
    pnl_ticks = -labels[short_mask]
    net_pnl = pnl_ticks - COMMISSION_TICKS

    n_trades = len(net_pnl)
    mean_pnl = float(np.mean(net_pnl))
    total_pnl = float(np.sum(net_pnl))
    wins = (net_pnl > 0).sum()
    losses = (net_pnl <= 0).sum()
    wr = float(wins / n_trades) if n_trades > 0 else 0

    gross_wins = float(np.sum(net_pnl[net_pnl > 0]))
    gross_losses = float(np.abs(np.sum(net_pnl[net_pnl <= 0])))
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    return {
        'n_trades': n_trades,
        'mean_pnl': mean_pnl,
        'total_pnl': total_pnl,
        'wr': wr,
        'pf': pf,
        'gross_edge': float(np.mean(-labels[short_mask])),
    }

def main():
    pred_files = sorted(PRED_DIR.glob("*_predictions.npz"))
    print(f"Found {len(pred_files)} prediction dates")

    results = {}

    for threshold in THRESHOLDS:
        print(f"\n=== Threshold: top {threshold*100:.0f}% shorts ===")

        daily_results = []
        regime_days = defaultdict(list)

        for pf in pred_files:
            date_str = pf.name[:8]

            # Classify regime
            alpha_path = ALPHA_DIR / f"{date_str}_alpha_labels.npz"
            regime, day_ret = classify_regime(alpha_path)

            # Analyze
            stats = analyze_date(pf, threshold, horizon_idx=0)
            if stats is None:
                print(f"  {date_str}: SKIP (insufficient data)")
                continue

            stats['date'] = date_str
            stats['regime'] = regime
            stats['day_ret'] = day_ret
            daily_results.append(stats)
            regime_days[regime].append(stats)

            green_str = "✅" if stats['total_pnl'] > 0 else "❌"
            print(f"  {date_str} [{regime:5s}]: {stats['n_trades']:4d} trades, "
                  f"net {stats['total_pnl']:+7.1f} ticks, WR {stats['wr']:.1%}, PF {stats['pf']:.2f} {green_str}")

        if not daily_results:
            print("  NO VALID DATES")
            continue

        # Aggregate stats
        all_pnl = [d['total_pnl'] for d in daily_results]
        green_days = sum(1 for p in all_pnl if p > 0)
        total_days = len(all_pnl)

        daily_mean = np.mean(all_pnl)
        daily_std = np.std(all_pnl)
        sharpe = float(daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0

        # Sortino
        downside = [p for p in all_pnl if p < 0]
        downside_std = np.std(downside) if len(downside) > 1 else np.std(all_pnl)
        sortino = float(daily_mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        print(f"\n  AGGREGATE: {green_days}/{total_days} green days ({green_days/total_days:.0%})")
        print(f"  Daily Sharpe: {sharpe:.1f}, Sortino: {sortino:.1f}")
        print(f"  Mean daily P&L: {daily_mean:.1f} ticks, Total: {sum(all_pnl):.1f} ticks")

        # Per-regime Sharpe
        regime_sharpes = {}
        for regime in ['green', 'red', 'flat', 'unknown']:
            days = regime_days.get(regime, [])
            if len(days) < 2:
                regime_sharpes[regime] = {'sharpe': None, 'n_days': len(days), 'mean_pnl': np.mean([d['total_pnl'] for d in days]) if days else 0}
                continue
            rpnl = [d['total_pnl'] for d in days]
            rs = float(np.mean(rpnl) / np.std(rpnl) * np.sqrt(252)) if np.std(rpnl) > 0 else 0
            regime_sharpes[regime] = {
                'sharpe': rs,
                'n_days': len(days),
                'mean_pnl': float(np.mean(rpnl)),
                'green_pct': sum(1 for p in rpnl if p > 0) / len(rpnl),
            }
            print(f"  {regime:7s}: {len(days):2d} days, Sharpe {rs:6.1f}, "
                  f"mean {np.mean(rpnl):+.1f} ticks, "
                  f"{sum(1 for p in rpnl if p > 0)}/{len(rpnl)} green")

        # HC #428 R1 regime gate
        sg = regime_sharpes.get('green', {}).get('sharpe')
        sr = regime_sharpes.get('red', {}).get('sharpe')
        if sg is not None and sr is not None and max(abs(sg), abs(sr)) > 0:
            imbalance = abs(sg - sr) / max(abs(sg), abs(sr))
            gate = "PASS" if imbalance <= 0.50 else "FAIL"
            print(f"\n  REGIME GATE (HC #428 R1): |{sg:.1f} - {sr:.1f}| / max = {imbalance:.3f} → {gate}")
        else:
            print(f"\n  REGIME GATE: INSUFFICIENT DATA (need both green and red days with 2+ samples)")

        # Day concentration
        max_day_pnl = max(abs(p) for p in all_pnl)
        total_abs = sum(abs(p) for p in all_pnl)
        day_conc = max_day_pnl / total_abs if total_abs > 0 else 0
        print(f"  Day concentration: {day_conc:.3f} (cap 0.70)")

        results[f"top_{threshold*100:.0f}pct"] = {
            'threshold': threshold,
            'n_dates': total_days,
            'green_days': green_days,
            'sharpe': sharpe,
            'sortino': sortino,
            'mean_daily_pnl': float(daily_mean),
            'total_pnl': float(sum(all_pnl)),
            'regime_sharpes': regime_sharpes,
            'day_concentration': day_conc,
            'daily_results': daily_results,
        }

    # Save
    out_path = OUTPUT_DIR / "regime_strat_48d_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

if __name__ == '__main__':
    main()
