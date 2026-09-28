#!/usr/bin/env python3
"""
HC #428 R1 Regime-Agnostic OOT Validation
CNN-Mamba v2 — Top 5% Short Signals — Regime-Stratified Sharpe Analysis

Uses 48 OOT prediction files. Classifies each day as GREEN/RED/FLAT
based on net 1s label sum (proxy for ES open-to-close direction).
Computes per-day PnL for top 5% short signals with passive entry cost.
"""

import numpy as np
import json
import glob
import os
from pathlib import Path

# Constants
COST_TICKS = 0.376  # passive both sides (commission only)
SHORT_PERCENTILE = 5  # top 5% most negative predictions = short signals
HORIZON_IDX = 0  # 1s horizon

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output'
PRED_DIR = f'{OUTPUT_DIR}/cnn_mamba_v2_bulk_oot_v2'
OUTPUT_FILE = f'{OUTPUT_DIR}/regime_stratification_v1.json'

def load_prediction_files():
    """Load all 48 OOT prediction files."""
    files = sorted(glob.glob(f'{PRED_DIR}/*_predictions.npz'))
    print(f"Found {len(files)} prediction files")

    daily_data = []
    for fpath in files:
        fname = os.path.basename(fpath)
        date_str = fname.split('_')[0]

        data = np.load(fpath, allow_pickle=True)
        preds = data['predictions'][:, HORIZON_IDX]  # 1s horizon predictions
        labels = data['labels'][:, HORIZON_IDX]  # 1s horizon actuals (ticks)

        # Filter valid (non-NaN)
        valid_mask = ~np.isnan(preds) & ~np.isnan(labels)
        preds = preds[valid_mask]
        labels = labels[valid_mask]

        if len(preds) == 0:
            print(f"  {date_str}: no valid predictions, skipping")
            continue

        # Daily direction: sum of 1s returns = net daily move in ticks
        daily_net_ticks = labels.sum()

        daily_data.append({
            'date': date_str,
            'predictions': preds,
            'labels': labels,
            'daily_net_ticks': float(daily_net_ticks),
            'n_samples': len(preds),
        })

    return daily_data


def classify_regime(daily_data, flat_threshold=50.0):
    """Classify each day as GREEN, RED, or FLAT based on net daily move."""
    for d in daily_data:
        net = d['daily_net_ticks']
        if net > flat_threshold:
            d['regime'] = 'GREEN'
        elif net < -flat_threshold:
            d['regime'] = 'RED'
        else:
            d['regime'] = 'FLAT'

    regimes = [d['regime'] for d in daily_data]
    print(f"\nRegime classification (flat threshold = {flat_threshold} ticks):")
    for r in ['GREEN', 'RED', 'FLAT']:
        count = regimes.count(r)
        print(f"  {r}: {count} days")

    return daily_data


def compute_short_signal_pnl(daily_data, percentile=SHORT_PERCENTILE):
    """For each day, select top percentile short signals and compute PnL."""
    for d in daily_data:
        preds = d['predictions']
        labels = d['labels']

        # Short signal = most negative prediction
        # Top 5% most bearish predictions
        threshold = np.percentile(preds, percentile)
        short_mask = preds <= threshold

        # PnL for shorts: we predict price goes DOWN, so PnL = -label (we sold)
        # minus cost
        short_labels = labels[short_mask]
        # Short trade PnL in ticks: -actual_return - cost
        trade_pnls = -short_labels - COST_TICKS

        d['n_trades'] = int(short_mask.sum())
        d['trade_pnls'] = trade_pnls
        d['daily_pnl_ticks'] = float(trade_pnls.sum())
        d['daily_mean_pnl'] = float(trade_pnls.mean()) if len(trade_pnls) > 0 else 0.0
        d['daily_wr'] = float((trade_pnls > 0).mean()) if len(trade_pnls) > 0 else 0.0

    return daily_data


def compute_metrics(daily_data, label="ALL"):
    """Compute Sharpe, PF, WR from per-day PnL."""
    if len(daily_data) == 0:
        return {'sharpe': np.nan, 'pf': np.nan, 'wr': np.nan, 'n_days': 0,
                'total_pnl_ticks': 0, 'avg_daily_pnl': 0, 'n_trades': 0}

    daily_pnls = np.array([d['daily_pnl_ticks'] for d in daily_data])
    all_trades = np.concatenate([d['trade_pnls'] for d in daily_data])

    # Sharpe: annualized from daily
    if daily_pnls.std() > 0:
        sharpe = (daily_pnls.mean() / daily_pnls.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Profit Factor: gross profit / gross loss
    gross_profit = all_trades[all_trades > 0].sum()
    gross_loss = abs(all_trades[all_trades < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Win Rate: trade-level
    wr = (all_trades > 0).mean()

    # Sortino
    downside = daily_pnls[daily_pnls < 0]
    downside_std = downside.std() if len(downside) > 1 else daily_pnls.std()
    sortino = (daily_pnls.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'pf': float(pf),
        'wr': float(wr),
        'n_days': len(daily_data),
        'total_pnl_ticks': float(daily_pnls.sum()),
        'avg_daily_pnl': float(daily_pnls.mean()),
        'avg_trades_per_day': float(np.mean([d['n_trades'] for d in daily_data])),
        'n_trades': int(len(all_trades)),
    }


def main():
    # Load and process
    daily_data = load_prediction_files()
    daily_data = classify_regime(daily_data)
    daily_data = compute_short_signal_pnl(daily_data)

    # Split by regime
    green_days = [d for d in daily_data if d['regime'] == 'GREEN']
    red_days = [d for d in daily_data if d['regime'] == 'RED']
    flat_days = [d for d in daily_data if d['regime'] == 'FLAT']

    # Compute metrics
    overall = compute_metrics(daily_data, "ALL")
    green = compute_metrics(green_days, "GREEN")
    red = compute_metrics(red_days, "RED")
    flat = compute_metrics(flat_days, "FLAT")

    # HC #428 R1 test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)
    s_green = green['sharpe']
    s_red = red['sharpe']
    max_abs = max(abs(s_green), abs(s_red))
    regime_divergence = abs(s_green - s_red) / max_abs if max_abs > 0 else 0.0
    regime_pass = regime_divergence <= 0.50

    # Print results
    print("\n" + "="*70)
    print("HC #428 R1 — REGIME-AGNOSTIC OOT VALIDATION")
    print("CNN-Mamba v2 — Top 5% Short Signals — Passive Entry (0.376 ticks cost)")
    print("="*70)

    for label, metrics in [("OVERALL", overall), ("GREEN DAYS", green),
                            ("RED DAYS", red), ("FLAT DAYS", flat)]:
        print(f"\n--- {label} ---")
        print(f"  Days: {metrics['n_days']}, Trades: {metrics['n_trades']}, "
              f"Avg trades/day: {metrics['avg_trades_per_day']:.0f}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}")
        print(f"  PF: {metrics['pf']:.3f}, WR: {metrics['wr']:.1%}")
        print(f"  Total PnL: {metrics['total_pnl_ticks']:.1f} ticks, "
              f"Avg daily: {metrics['avg_daily_pnl']:.1f} ticks")

    print(f"\n{'='*70}")
    print(f"REGIME DIVERGENCE TEST:")
    print(f"  Sharpe_green = {s_green:.3f}")
    print(f"  Sharpe_red   = {s_red:.3f}")
    print(f"  |S_g - S_r| / max(|S_g|, |S_r|) = {regime_divergence:.3f}")
    print(f"  Threshold: 0.50")
    print(f"  RESULT: {'PASS' if regime_pass else 'REJECT'}")
    print(f"{'='*70}")

    # Per-day detail
    print("\nPer-day breakdown:")
    print(f"{'Date':>10} {'Regime':>6} {'Trades':>7} {'PnL':>8} {'WR':>6} {'Net(ticks)':>10}")
    for d in daily_data:
        print(f"{d['date']:>10} {d['regime']:>6} {d['n_trades']:>7} "
              f"{d['daily_pnl_ticks']:>8.1f} {d['daily_wr']:>6.1%} "
              f"{d['daily_net_ticks']:>10.0f}")

    # Save JSON
    result = {
        'analysis': 'HC #428 R1 Regime-Agnostic OOT Validation',
        'model': 'CNN-Mamba v2',
        'signal': 'Top 5% short signals',
        'cost_ticks': COST_TICKS,
        'n_oot_days': len(daily_data),
        'overall': overall,
        'green_days': green,
        'red_days': red,
        'flat_days': flat,
        'regime_divergence': float(regime_divergence),
        'regime_divergence_threshold': 0.50,
        'regime_test_pass': regime_pass,
        'per_day': [{
            'date': d['date'],
            'regime': d['regime'],
            'n_trades': d['n_trades'],
            'daily_pnl_ticks': d['daily_pnl_ticks'],
            'daily_wr': d['daily_wr'],
            'daily_net_ticks': d['daily_net_ticks'],
        } for d in daily_data],
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
