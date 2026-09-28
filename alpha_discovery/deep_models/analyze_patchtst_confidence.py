#!/usr/bin/env python3
"""
Confidence-tier analysis for PatchTST (or any model) OOT predictions.

Loads fold_*_oot_predictions.npz files, concatenates them, and computes
IC (Spearman rank correlation) and DA (directional accuracy) at various
confidence tiers based on abs(prediction).

Usage:
    python analyze_patchtst_confidence.py [output_dir]

Default: /home/jupiter/Lvl3Quant/output/patchtst_sliding60d_smart_v2
"""

import sys
import glob
import numpy as np
from scipy.stats import spearmanr

def load_concat_predictions(output_dir):
    """Load and concatenate all fold predictions."""
    files = sorted(glob.glob(f"{output_dir}/fold_*_oot_predictions.npz"))
    if not files:
        print(f"ERROR: No fold_*_oot_predictions.npz files found in {output_dir}")
        sys.exit(1)

    all_preds = []
    all_labels = []

    for f in files:
        d = np.load(f)
        all_preds.append(d['predictions'])
        all_labels.append(d['labels'])
        fold_name = f.split('/')[-1]
        n = d['predictions'].shape[0]
        print(f"  Loaded {fold_name}: {n:,} samples")

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"  Total concat: {preds.shape[0]:,} samples across {len(files)} folds\n")
    return preds, labels, files


def compute_ic(preds, labels):
    """Spearman rank IC, handling edge cases."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return np.nan
    corr, _ = spearmanr(preds[mask], labels[mask])
    return corr


def compute_da(preds, labels):
    """Directional accuracy: fraction where sign(pred) == sign(label), excluding zeros."""
    mask = (preds != 0) & (labels != 0) & np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return np.nan
    return np.mean(np.sign(preds[mask]) == np.sign(labels[mask]))


def analyze_confidence_tiers(preds, labels, horizons=['1s', '5s', '10s']):
    """Run confidence-tier analysis across horizons."""
    tiers = [
        ('All',    0.0),
        ('Top50%', 0.50),
        ('Top25%', 0.75),
        ('Top10%', 0.90),
        ('Top5%',  0.95),
        ('Top1%',  0.99),
    ]

    # Determine horizon columns
    n_horizons = preds.shape[1] if preds.ndim > 1 else 1
    if n_horizons == 1:
        horizons = ['pred']
    else:
        horizons = horizons[:n_horizons]

    for h_idx, h_name in enumerate(horizons):
        if preds.ndim > 1:
            p = preds[:, h_idx]
            l = labels[:, h_idx]
        else:
            p = preds
            l = labels

        confidence = np.abs(p)

        print(f"{'='*70}")
        print(f"  HORIZON: {h_name}")
        print(f"{'='*70}")
        print(f"  {'Tier':<10} {'N':>8} {'IC':>10} {'DA':>10} {'Mean|Pred|':>12} {'Mean|Label|':>12}")
        print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")

        for tier_name, quantile in tiers:
            if quantile > 0:
                threshold = np.quantile(confidence, quantile)
                mask = confidence >= threshold
            else:
                mask = np.ones(len(p), dtype=bool)

            n = mask.sum()
            if n < 10:
                print(f"  {tier_name:<10} {n:>8} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>12}")
                continue

            ic = compute_ic(p[mask], l[mask])
            da = compute_da(p[mask], l[mask])
            mean_pred = np.mean(np.abs(p[mask]))
            mean_label = np.mean(np.abs(l[mask]))

            print(f"  {tier_name:<10} {n:>8,} {ic:>10.4f} {da:>10.4f} {mean_pred:>12.6f} {mean_label:>12.6f}")

        print()

    # Summary: concat IC across all data (same as "All" tier)
    print(f"{'='*70}")
    print(f"  CONCAT IC SUMMARY (All data)")
    print(f"{'='*70}")
    for h_idx, h_name in enumerate(horizons):
        if preds.ndim > 1:
            ic = compute_ic(preds[:, h_idx], labels[:, h_idx])
        else:
            ic = compute_ic(preds, labels)
        print(f"  {h_name}: IC = {ic:.4f}")
    print()


def per_fold_ic(output_dir, horizons=['1s', '5s', '10s']):
    """Show per-fold IC for context."""
    files = sorted(glob.glob(f"{output_dir}/fold_*_oot_predictions.npz"))

    n_horizons = 3  # default
    first = np.load(files[0])
    n_horizons = first['predictions'].shape[1] if first['predictions'].ndim > 1 else 1
    horizons = horizons[:n_horizons]

    print(f"{'='*70}")
    print(f"  PER-FOLD IC")
    print(f"{'='*70}")
    print(f"  {'Fold':<10}", end='')
    for h in horizons:
        print(f" {'IC_'+h:>10}", end='')
    print(f" {'N':>8}")
    print(f"  {'-'*10}", end='')
    for _ in horizons:
        print(f" {'-'*10}", end='')
    print(f" {'-'*8}")

    for f in files:
        d = np.load(f)
        fold_name = f.split('/')[-1].replace('_oot_predictions.npz', '')
        n = d['predictions'].shape[0]
        print(f"  {fold_name:<10}", end='')
        for h_idx in range(len(horizons)):
            if d['predictions'].ndim > 1:
                ic = compute_ic(d['predictions'][:, h_idx], d['labels'][:, h_idx])
            else:
                ic = compute_ic(d['predictions'], d['labels'])
            print(f" {ic:>10.4f}", end='')
        print(f" {n:>8,}")
    print()


def main():
    default_dir = "/home/jupiter/Lvl3Quant/output/patchtst_sliding60d_smart_v2"
    output_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir

    print(f"\n{'#'*70}")
    print(f"  CONFIDENCE-TIER ANALYSIS")
    print(f"  Source: {output_dir}")
    print(f"{'#'*70}\n")

    preds, labels, files = load_concat_predictions(output_dir)

    # Per-fold IC first
    per_fold_ic(output_dir)

    # Main confidence tier analysis
    analyze_confidence_tiers(preds, labels)


if __name__ == "__main__":
    main()
