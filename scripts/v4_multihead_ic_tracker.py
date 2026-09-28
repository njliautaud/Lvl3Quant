"""
V4 Multihead IC Tracker
========================
Pulls latest OOT predictions from Neptune, computes running concat IC
for all heads (directional + pressure). Designed to run periodically
to track whether pressure heads develop signal as more folds complete.
"""

import os
import sys
import glob
import subprocess
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict


def sync_predictions():
    """Pull latest predictions from Neptune."""
    local_dir = '/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1'
    remote_dir = 'nick@neptune:/home/nick/Lvl3Quant/output/v4_multihead_pressure_v1/fold_*_oot_predictions.npz'
    os.makedirs(local_dir, exist_ok=True)
    try:
        subprocess.run(
            f'scp -o ConnectTimeout=10 {remote_dir} {local_dir}/',
            shell=True, capture_output=True, timeout=120
        )
    except Exception as e:
        print(f"Sync failed: {e}")


def compute_concat_ic(pred_dir):
    """Compute concat IC across all available folds."""
    files = sorted(glob.glob(os.path.join(pred_dir, 'fold_*_oot_predictions.npz')))
    if not files:
        print("No prediction files found.")
        return

    print(f"Found {len(files)} fold predictions")
    print()

    # Collect per-fold ICs
    heads = {
        'dir': ['1s', '5s', '10s'],
        'ntps': ['1s', '5s', '10s', '30s'],
        'eofi': ['1s', '5s', '10s', '30s'],
        'pdi': ['1s', '5s', '10s', '30s'],
        'tia': ['1s', '5s', '10s', '30s'],
    }

    per_fold_ics = defaultdict(list)
    all_preds = defaultdict(list)
    all_labels = defaultdict(list)
    fold_nums = []

    for f in files:
        d = np.load(f, allow_pickle=True)
        fold = int(d['fold'])
        fold_nums.append(fold)

        for head, horizons in heads.items():
            for h in horizons:
                key = f'ic_{head}_{h}'
                if key in d:
                    per_fold_ics[f'{head}_{h}'].append(float(d[key]))

                # Also collect raw predictions for concat IC
                pred_key = f'preds_{head}'
                label_key = f'labels_{head}'
                if pred_key in d and label_key in d:
                    preds = d[pred_key]
                    labels = d[label_key]
                    h_idx = horizons.index(h)
                    if h_idx < preds.shape[1]:
                        p = preds[:, h_idx]
                        l = labels[:, h_idx]
                        # Skip if labels are NaN
                        valid = ~np.isnan(l)
                        if valid.sum() > 0:
                            all_preds[f'{head}_{h}'].append(p[valid])
                            all_labels[f'{head}_{h}'].append(l[valid])

    # Print per-fold average ICs
    print(f"{'Head':>8} {'Hz':>4} {'Mean_IC':>8} {'Std_IC':>8} {'Min':>8} {'Max':>8} {'Sign%':>6} {'N':>4}")
    print("-" * 55)
    for head, horizons in heads.items():
        for h in horizons:
            key = f'{head}_{h}'
            ics = per_fold_ics.get(key, [])
            if ics:
                mean_ic = np.mean(ics)
                std_ic = np.std(ics)
                min_ic = np.min(ics)
                max_ic = np.max(ics)
                # Sign consistency: % of folds with same sign as mean
                if mean_ic != 0:
                    sign_pct = np.mean([1 for ic in ics if ic * mean_ic > 0])
                else:
                    sign_pct = 0.5
                print(f"{head:>8} {h:>4} {mean_ic:+8.4f} {std_ic:8.4f} {min_ic:+8.4f} {max_ic:+8.4f} {sign_pct:5.0%} {len(ics):4d}")
        print()

    # Compute concat IC (pool all predictions across folds)
    print("\nCONCAT IC (pooled across all folds):")
    print(f"{'Head':>8} {'Hz':>4} {'Concat_IC':>10} {'N_samples':>10}")
    print("-" * 40)
    for head, horizons in heads.items():
        for h in horizons:
            key = f'{head}_{h}'
            if key in all_preds and all_preds[key]:
                p = np.concatenate(all_preds[key])
                l = np.concatenate(all_labels[key])
                if len(p) > 100 and np.std(p) > 1e-8 and np.std(l) > 1e-8:
                    ic, _ = spearmanr(p, l)
                    print(f"{head:>8} {h:>4} {ic:+10.4f} {len(p):10d}")
                else:
                    print(f"{head:>8} {h:>4} {'N/A':>10} {len(p):10d}")
        print()

    print(f"\nFold range: {min(fold_nums)} - {max(fold_nums)} ({len(fold_nums)} folds)")


def main():
    pred_dir = '/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1'

    if '--sync' in sys.argv:
        print("Syncing predictions from Neptune...")
        sync_predictions()
        print()

    compute_concat_ic(pred_dir)


if __name__ == '__main__':
    main()
