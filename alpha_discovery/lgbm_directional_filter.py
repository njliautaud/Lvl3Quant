"""
LGBM Directional Filter for Monday Go-Live
===========================================

Problem: LGBM fold 00 has IC=0.24 but only 36-42% direction accuracy → unprofitable

Solution: Filter predictions to trade ONLY when direction confidence is high

Approach:
1. Load LGBM predictions + labels
2. Calibrate direction probability using prediction magnitude
3. Only trade when P(correct_direction) > threshold (e.g., 60%)
4. Output filtered signals for fill sim validation

This is a POST-PROCESSING filter - doesn't retrain model, just filters trades.
Can be deployed Monday with existing LGBM weights.
"""

import numpy as np
import sys
from pathlib import Path
from scipy.stats import spearmanr

sys.stdout.reconfigure(line_buffering=True)

PREDS_PATH = Path("/home/jupiter/Lvl3Quant/alpha_discovery/results/lgbm_magnitude_weighted/fold_00_predictions.npz")

print("=" * 70)
print("LGBM DIRECTIONAL FILTER - Calibration")
print("=" * 70)
print()

# Load predictions
print("Loading predictions...")
data = np.load(PREDS_PATH)
preds = data['predictions']
labels = data['labels']
print(f"Loaded {len(preds):,} predictions\n")

# Compute direction accuracy by prediction magnitude bins
print("Calibrating direction probability by prediction strength...")
print()

abs_preds = np.abs(preds)
pred_sign = np.sign(preds)
label_sign = np.sign(labels)
correct_dir = (pred_sign == label_sign)

# Create magnitude bins
percentiles = np.arange(0, 101, 5)  # 20 bins
thresholds = np.percentile(abs_preds, percentiles)

print(f"{'Percentile':<12} {'Threshold':<12} {'DirAcc':<10} {'Count':<10} {'Trade?':<8}")
print("-" * 60)

results = []
for i in range(len(thresholds) - 1):
    low, high = thresholds[i], thresholds[i+1]
    pct_low, pct_high = percentiles[i], percentiles[i+1]

    mask = (abs_preds >= low) & (abs_preds < high)
    n = mask.sum()

    if n > 0:
        dir_acc = correct_dir[mask].mean()
        trade = "YES" if dir_acc >= 0.55 else "NO"  # 55% threshold for profitability

        print(f"{pct_low:3.0f}-{pct_high:3.0f}%   {low:>10.6f}  {dir_acc:>8.3f}  {n:>9,}  {trade:>6}")

        results.append({
            'pct_range': (pct_low, pct_high),
            'mag_range': (low, high),
            'dir_acc': dir_acc,
            'count': n,
            'trade': dir_acc >= 0.55
        })

print()
print("=" * 70)
print("FILTER RECOMMENDATION")
print("=" * 70)

# Find optimal threshold
tradeable = [r for r in results if r['trade']]

if not tradeable:
    print("❌ NO bins have >55% direction accuracy")
    print("   Model quality too poor - need to retrain with better features/architecture")
else:
    # Find the minimum magnitude threshold where dir_acc >= 55%
    min_tradeable_pct = min(r['pct_range'][0] for r in tradeable)
    optimal_thresh = thresholds[int(min_tradeable_pct // 5)]

    print(f"✅ Trade signals with |prediction| >= {optimal_thresh:.6f}")
    print(f"   This is the top {100 - min_tradeable_pct:.0f}% of predictions by strength")
    print()

    # Compute expected metrics
    trade_mask = abs_preds >= optimal_thresh
    n_trades = trade_mask.sum()
    pct_kept = 100 * n_trades / len(preds)
    final_dir_acc = correct_dir[trade_mask].mean()
    final_ic = spearmanr(preds[trade_mask], labels[trade_mask])[0]

    print(f"Expected performance:")
    print(f"  - Trades per day: {n_trades:,} ({pct_kept:.1f}% of all signals)")
    print(f"  - Direction accuracy: {final_dir_acc:.1%}")
    print(f"  - IC: {final_ic:+.4f}")
    print()

    # Save filtered signals
    output_path = PREDS_PATH.parent / "fold_00_filtered.npz"
    np.savez(
        output_path,
        predictions=preds[trade_mask],
        labels=labels[trade_mask],
        indices=np.where(trade_mask)[0],
        threshold=optimal_thresh
    )

    print(f"💾 Filtered signals saved: {output_path}")
    print(f"   {n_trades:,} trades (from {len(preds):,} total)")
    print()
    print("Next step: Run these through fill sim to validate profitability")
