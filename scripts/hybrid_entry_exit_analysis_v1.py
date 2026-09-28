#!/usr/bin/env python3
"""
Hybrid Entry/Exit Analysis v1
Analyzes whether passive-entry + market-exit can make short signals profitable.

Cost scenarios:
  A) PURE MARKET: 1.376 ticks RT (0.376 commission + 1.0 spread both sides)
  B) HYBRID: 0.876 ticks (passive entry saves spread, market exit)
  C) AGGRESSIVE HYBRID: 0.688 ticks (passive entry + passive exit, best case)
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

# === CONSTANTS ===
COMMISSION_RT_TICKS = 0.376   # $4.70 / $12.50
SPREAD_TICKS = 1.0            # 1 tick wide book during RTH

COST_PURE_MARKET = COMMISSION_RT_TICKS + SPREAD_TICKS      # 1.376 (cross spread both sides)
COST_HYBRID = COMMISSION_RT_TICKS + 0.5 * SPREAD_TICKS     # 0.876 (passive entry, market exit)
COST_AGGRESSIVE = COMMISSION_RT_TICKS + 0.0 * SPREAD_TICKS # 0.376 ... wait
# Actually: passive entry = no spread on entry, passive exit = no spread on exit
# But commission is always 0.376 RT. So aggressive = 0.376 commission only.
# Let me re-read the task...
# "AGGRESSIVE HYBRID: 0.688 ticks (passive entry + passive exit, best case)"
# 0.688 = 0.376 + 0.312? That doesn't make sense. Let me use 0.376 for full passive.
# Actually the user specified 0.688 explicitly. Let's use that.
COST_AGGRESSIVE_HYBRID = 0.688

# For passive entry: you save 0.5 ticks (half spread) on entry
# For market exit: you pay 0.5 ticks (half spread) on exit
# Total: 0.376 + 0.5 = 0.876. Correct.
# For passive both: 0.376 + 0 = 0.376, but user says 0.688.
# 0.688 = 0.376 + 0.312... maybe accounting for partial fills/slippage.
# Just use what the user specified.

PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2/'
MBO_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/'
OUT_DIR = '/home/jupiter/Lvl3Quant/output/hybrid_entry_exit_v1/'

HORIZONS = ['1s', '5s', '10s']
HORIZON_IDX = {h: i for i, h in enumerate(HORIZONS)}
PERCENTILES = [1, 2, 3, 5, 10]

def load_day(pred_file):
    """Load predictions and map to MBO labels for one day."""
    pred_data = np.load(pred_file, allow_pickle=True)
    date_str = str(pred_data['date'])
    preds = pred_data['predictions']  # (n_windows, 3)
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    # Load corresponding MBO file
    mbo_file = os.path.join(MBO_DIR, f'{date_str}_mbo_events.npz')
    if not os.path.exists(mbo_file):
        print(f'  WARNING: MBO file not found for {date_str}')
        return None

    mbo = np.load(mbo_file, allow_pickle=True)

    # Map prediction indices to event indices
    indices = np.arange(n_windows) * stride + (window_size - 1)

    # Clip to valid range
    max_idx = len(mbo['labels_1s']) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[valid]

    # Extract labels (in ticks)
    result = {
        'date': date_str,
        'pred_1s': preds[:, 0],
        'pred_5s': preds[:, 1],
        'pred_10s': preds[:, 2],
        'label_1s': mbo['labels_1s'][indices],
        'label_5s': mbo['labels_5s'][indices],
        'label_10s': mbo['labels_10s'][indices],
    }

    # Try to get 30s labels if available
    if 'labels_30s' in mbo:
        result['label_30s'] = mbo['labels_30s'][indices]

    return result


def analyze_short_signals(all_data, horizon='10s'):
    """Analyze short signals at various confidence percentiles."""
    hi = HORIZON_IDX[horizon]

    # Concatenate all days
    all_preds = np.concatenate([d[f'pred_{horizon}'] for d in all_data])
    all_labels = np.concatenate([d[f'label_{horizon}'] for d in all_data])

    # For shorts: pred < 0 means model predicts price will fall
    # realized_ticks for short = -label (if label is forward return in ticks, short profits when price drops)
    short_mask = all_preds < 0
    short_preds = all_preds[short_mask]
    short_labels = all_labels[short_mask]

    # Remove NaNs
    valid = ~np.isnan(short_labels)
    short_preds = short_preds[valid]
    short_labels = short_labels[valid]

    # Realized ticks for SHORT position: price goes down = profit
    # Labels are forward returns in ticks (positive = price went up)
    # Short profit = -label
    short_realized = -short_labels

    print(f'\n=== SHORT SIGNALS, {horizon} horizon ===')
    print(f'Total predictions: {len(all_preds):,}')
    print(f'Short predictions (pred < 0): {len(short_preds):,} ({100*len(short_preds)/len(all_preds):.1f}%)')
    print(f'Short realized ticks: mean={short_realized.mean():.4f} std={short_realized.std():.4f}')

    results = []

    for pct in PERCENTILES:
        # Top pct% = most negative predictions (strongest short signals)
        threshold = np.percentile(short_preds, pct)
        mask = short_preds <= threshold
        n_signals = mask.sum()
        realized = short_realized[mask]

        avg_realized = realized.mean()
        std_realized = realized.std()

        for cost_name, cost in [('PURE_MARKET', COST_PURE_MARKET),
                                 ('HYBRID', COST_HYBRID),
                                 ('AGGRESSIVE_HYBRID', COST_AGGRESSIVE_HYBRID)]:
            net = realized - cost
            net_mean = net.mean()
            net_std = net.std()
            sharpe = net_mean / net_std * np.sqrt(252) if net_std > 0 else 0
            win_rate = (realized > cost).mean()

            results.append({
                'horizon': horizon,
                'percentile': f'top_{pct}pct',
                'n_signals': n_signals,
                'avg_realized_ticks': avg_realized,
                'cost_scenario': cost_name,
                'cost_ticks': cost,
                'net_ticks': net_mean,
                'net_std': net_std,
                'sharpe_annualized': sharpe,
                'win_rate': win_rate,
                'pred_threshold': threshold,
            })

        print(f'\n  Top {pct}% (n={n_signals:,}, threshold={threshold:.4f}):')
        print(f'    Avg realized: {avg_realized:.4f} ticks')
        print(f'    PURE MARKET (cost={COST_PURE_MARKET}): net={avg_realized - COST_PURE_MARKET:.4f}, WR={((realized > COST_PURE_MARKET).mean()*100):.1f}%')
        print(f'    HYBRID     (cost={COST_HYBRID}): net={avg_realized - COST_HYBRID:.4f}, WR={((realized > COST_HYBRID).mean()*100):.1f}%')
        print(f'    AGG HYBRID (cost={COST_AGGRESSIVE_HYBRID}): net={avg_realized - COST_AGGRESSIVE_HYBRID:.4f}, WR={((realized > COST_AGGRESSIVE_HYBRID).mean()*100):.1f}%')

    return results


def per_day_analysis(all_data, horizon='10s'):
    """Per-day net ticks under each cost scenario."""
    results = []

    for d in all_data:
        preds = d[f'pred_{horizon}']
        labels = d[f'label_{horizon}']

        # Top 1% short signals for this day
        short_mask = preds < 0
        if short_mask.sum() < 10:
            continue

        short_preds = preds[short_mask]
        short_labels = labels[short_mask]
        valid = ~np.isnan(short_labels)
        short_preds = short_preds[valid]
        short_labels = short_labels[valid]
        short_realized = -short_labels

        if len(short_realized) < 100:
            continue

        for pct in [1, 3, 5]:
            threshold = np.percentile(short_preds, pct)
            mask = short_preds <= threshold
            n = mask.sum()
            if n == 0:
                continue
            realized = short_realized[mask]
            avg_r = realized.mean()

            for cost_name, cost in [('PURE_MARKET', COST_PURE_MARKET),
                                     ('HYBRID', COST_HYBRID),
                                     ('AGGRESSIVE_HYBRID', COST_AGGRESSIVE_HYBRID)]:
                net = avg_r - cost
                results.append({
                    'date': d['date'],
                    'percentile': f'top_{pct}pct',
                    'n_signals': n,
                    'avg_realized_ticks': avg_r,
                    'cost_scenario': cost_name,
                    'cost_ticks': cost,
                    'net_ticks': net,
                    'profitable': net > 0,
                })

    return pd.DataFrame(results)


def fill_rate_sensitivity(all_data, horizon='10s'):
    """Analyze adverse selection under partial fill scenarios."""

    all_preds = np.concatenate([d[f'pred_{horizon}'] for d in all_data])
    all_labels = np.concatenate([d[f'label_{horizon}'] for d in all_data])

    short_mask = all_preds < 0
    short_preds = all_preds[short_mask]
    short_labels = all_labels[short_mask]
    valid = ~np.isnan(short_labels)
    short_preds = short_preds[valid]
    short_labels = short_labels[valid]
    short_realized = -short_labels

    print(f'\n=== FILL RATE SENSITIVITY ({horizon}) ===')

    results = []
    for pct in [1, 3, 5]:
        threshold = np.percentile(short_preds, pct)
        mask = short_preds <= threshold
        realized = short_realized[mask]
        n = len(realized)

        # Sort realized to simulate adverse selection
        sorted_realized = np.sort(realized)  # ascending = worst first

        # Scenario 1: Random 50% fill (expected value = same as full set)
        random_50_mean = realized.mean()  # E[random subset] = E[full set]

        # Scenario 2: Worst 50% fill (adverse selection)
        worst_half = sorted_realized[:n // 2]
        worst_50_mean = worst_half.mean()

        # Scenario 3: Best 50% fill (favorable selection, unlikely but informative)
        best_half = sorted_realized[n // 2:]
        best_50_mean = best_half.mean()

        # Scenario 4: Bottom quartile (extreme adverse selection, 25% fill)
        worst_25 = sorted_realized[:n // 4]
        worst_25_mean = worst_25.mean()

        print(f'\n  Top {pct}% short signals (n={n:,}):')
        print(f'    Full set avg realized: {realized.mean():.4f} ticks')
        print(f'    Random 50% fill:       {random_50_mean:.4f} ticks (net hybrid: {random_50_mean - COST_HYBRID:.4f})')
        print(f'    Worst 50% fill:        {worst_50_mean:.4f} ticks (net hybrid: {worst_50_mean - COST_HYBRID:.4f})')
        print(f'    Best 50% fill:         {best_50_mean:.4f} ticks (net hybrid: {best_50_mean - COST_HYBRID:.4f})')
        print(f'    Worst 25% fill:        {worst_25_mean:.4f} ticks (net hybrid: {worst_25_mean - COST_HYBRID:.4f})')

        for scenario, mean_val in [('random_50pct', random_50_mean),
                                    ('worst_50pct', worst_50_mean),
                                    ('best_50pct', best_50_mean),
                                    ('worst_25pct', worst_25_mean)]:
            results.append({
                'horizon': horizon,
                'percentile': f'top_{pct}pct',
                'fill_scenario': scenario,
                'avg_realized_ticks': mean_val,
                'net_hybrid_ticks': mean_val - COST_HYBRID,
                'net_pure_market_ticks': mean_val - COST_PURE_MARKET,
            })

    return pd.DataFrame(results)


def main():
    print('Loading OOT predictions...')
    pred_files = sorted(Path(PRED_DIR).glob('*_predictions.npz'))
    print(f'Found {len(pred_files)} prediction files')

    all_data = []
    for f in pred_files:
        d = load_day(str(f))
        if d is not None:
            all_data.append(d)

    print(f'Loaded {len(all_data)} days successfully')

    # === MAIN ANALYSIS ===
    all_results = []
    for h in HORIZONS:
        results = analyze_short_signals(all_data, horizon=h)
        all_results.extend(results)

    sweep_df = pd.DataFrame(all_results)
    sweep_df.to_csv(os.path.join(OUT_DIR, 'hybrid_analysis.csv'), index=False)
    print(f'\nSaved sweep results to {OUT_DIR}hybrid_analysis.csv')

    # === PER-DAY ANALYSIS ===
    day_df = per_day_analysis(all_data, horizon='10s')
    day_df.to_csv(os.path.join(OUT_DIR, 'per_day_analysis.csv'), index=False)

    # Summary of profitable days
    print('\n=== PER-DAY PROFITABLE DAYS ===')
    for pct in [1, 3, 5]:
        for cost_name in ['PURE_MARKET', 'HYBRID', 'AGGRESSIVE_HYBRID']:
            sub = day_df[(day_df['percentile'] == f'top_{pct}pct') & (day_df['cost_scenario'] == cost_name)]
            n_profitable = sub['profitable'].sum()
            n_total = len(sub)
            frac = n_profitable / n_total if n_total > 0 else 0
            avg_net = sub['net_ticks'].mean()
            print(f'  Top {pct}% {cost_name}: {n_profitable}/{n_total} days profitable ({frac*100:.0f}%), avg net={avg_net:.4f} ticks')

    # === FILL RATE SENSITIVITY ===
    fill_df = fill_rate_sensitivity(all_data, horizon='10s')
    fill_df.to_csv(os.path.join(OUT_DIR, 'fill_sensitivity.csv'), index=False)

    # === GENERATE REPORT ===
    generate_report(sweep_df, day_df, fill_df, all_data)

    print(f'\nAll outputs saved to {OUT_DIR}')


def generate_report(sweep_df, day_df, fill_df, all_data):
    """Generate REPORT.md with findings."""

    lines = ['# Hybrid Entry/Exit Analysis v1', '']
    lines.append(f'**OOT Days Analyzed:** {len(all_data)}')
    lines.append(f'**Date Range:** {all_data[0]["date"]} to {all_data[-1]["date"]}')
    lines.append('')

    lines.append('## Cost Scenarios')
    lines.append(f'- PURE MARKET: {COST_PURE_MARKET} ticks (spread both sides + commission)')
    lines.append(f'- HYBRID: {COST_HYBRID} ticks (passive entry, market exit)')
    lines.append(f'- AGGRESSIVE HYBRID: {COST_AGGRESSIVE_HYBRID} ticks (passive entry + passive exit)')
    lines.append('')

    # Key results table for 10s horizon
    lines.append('## Short Signal Results (10s horizon)')
    lines.append('')
    lines.append('| Percentile | Avg Realized | Cost Scenario | Net Ticks | Win Rate | Sharpe |')
    lines.append('|------------|-------------|---------------|-----------|----------|--------|')

    h10 = sweep_df[sweep_df['horizon'] == '10s']
    for _, row in h10.iterrows():
        lines.append(f"| {row['percentile']} | {row['avg_realized_ticks']:.3f} | {row['cost_scenario']} | {row['net_ticks']:.3f} | {row['win_rate']*100:.1f}% | {row['sharpe_annualized']:.2f} |")

    lines.append('')

    # Per-day summary
    lines.append('## Per-Day Profitability (10s horizon)')
    lines.append('')
    for pct in [1, 3, 5]:
        for cost_name in ['HYBRID']:
            sub = day_df[(day_df['percentile'] == f'top_{pct}pct') & (day_df['cost_scenario'] == cost_name)]
            n_profitable = sub['profitable'].sum()
            n_total = len(sub)
            frac = n_profitable / n_total if n_total > 0 else 0
            lines.append(f'- Top {pct}% HYBRID: {n_profitable}/{n_total} days profitable ({frac*100:.0f}%)')

    lines.append('')

    # Fill sensitivity
    lines.append('## Fill Rate Sensitivity (10s, HYBRID cost)')
    lines.append('')
    for _, row in fill_df[fill_df['horizon'] == '10s'].iterrows():
        lines.append(f"- {row['percentile']} {row['fill_scenario']}: realized={row['avg_realized_ticks']:.3f}, net={row['net_hybrid_ticks']:.3f}")

    lines.append('')

    report = '\n'.join(lines)
    with open(os.path.join(OUT_DIR, 'REPORT.md'), 'w') as f:
        f.write(report)

    print(f'\nReport saved to {OUT_DIR}REPORT.md')


if __name__ == '__main__':
    main()
