"""
Analyze IC vs Profitability Gap
================================
Why does LGBM fold 00 have IC=0.24 but $2.63/day P&L?

Diagnostic checks:
1. Prediction distribution - are signals too weak?
2. Top percentile analysis - does strength correlate with profitability?
3. Direction accuracy at different confidence levels
4. Realized returns vs predicted returns
"""

import numpy as np
import sys

sys.stdout.reconfigure(line_buffering=True)

# Load predictions
print("Loading LGBM fold 00 predictions...")
data = np.load('/home/jupiter/Lvl3Quant/alpha_discovery/results/lgbm_magnitude_weighted/fold_00_predictions.npz')
preds = data['predictions']
labels = data['labels']

print(f"Loaded {len(preds):,} predictions\n")

# 1. Prediction distribution
print("=== PREDICTION DISTRIBUTION ===")
print(f"Mean: {preds.mean():.6f}")
print(f"Std:  {preds.std():.6f}")
print(f"Min:  {preds.min():.6f}")
print(f"Max:  {preds.max():.6f}")
print(f"Median: {np.median(preds):.6f}")
print()

# 2. Top percentile analysis
print("=== TOP PERCENTILE ANALYSIS ===")
abs_preds = np.abs(preds)
percentiles = [50, 75, 90, 95, 99]

for pct in percentiles:
    thresh = np.percentile(abs_preds, pct)
    mask = abs_preds >= thresh

    # IC in this bucket
    corr = np.corrcoef(preds[mask], labels[mask])[0, 1]

    # Direction accuracy
    pred_dir = np.sign(preds[mask])
    label_dir = np.sign(labels[mask])
    dir_acc = (pred_dir == label_dir).mean()

    # Average magnitude
    avg_pred_mag = abs_preds[mask].mean()
    avg_label_mag = np.abs(labels[mask]).mean()

    print(f"Top {100-pct:2d}% (>{thresh:.6f}):")
    print(f"  IC:       {corr:+.4f}")
    print(f"  DirAcc:   {dir_acc:.3f}")
    print(f"  Avg|pred|: {avg_pred_mag:.6f}")
    print(f"  Avg|label|: {avg_label_mag:.6f}")
    print(f"  Count:    {mask.sum():,}")
    print()

# 3. Win rate by confidence bucket
print("=== WIN RATE BY CONFIDENCE BUCKET ===")
buckets = [
    (0, 25, "Bottom 25%"),
    (25, 50, "25-50%"),
    (50, 75, "50-75%"),
    (75, 90, "75-90%"),
    (90, 95, "90-95%"),
    (95, 99, "95-99%"),
    (99, 100, "Top 1%")
]

for low, high, label in buckets:
    low_thresh = np.percentile(abs_preds, low)
    high_thresh = np.percentile(abs_preds, high)
    mask = (abs_preds >= low_thresh) & (abs_preds < high_thresh)

    pred_dir = np.sign(preds[mask])
    label_dir = np.sign(labels[mask])
    win_rate = (pred_dir == label_dir).mean()

    avg_ret = (preds[mask] * labels[mask]).mean()

    print(f"{label:12s}: WinRate={win_rate:.3f}, AvgRet={avg_ret:.6f}, N={mask.sum():,}")

print("\n=== DIAGNOSIS ===")
print("If top percentiles have good IC/DirAcc but fill sim fails:")
print("  → Problem is EXECUTION (queue position, adverse selection)")
print("  → Solution: execution-aware filtering or retrain with fill-aware loss")
print()
print("If top percentiles have poor IC/DirAcc:")
print("  → Problem is MODEL QUALITY")
print("  → Solution: better features, architecture, or training")
