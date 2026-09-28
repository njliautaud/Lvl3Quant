#!/usr/bin/env python3
"""
Compute MFE/MAE labels within multiple horizons for each CNN-Mamba v2 prediction event.
Output saved per date to /home/jupiter/Lvl3Quant/output/mfe_mae_labels_v1/
"""
import os
import numpy as np
from pathlib import Path
from glob import glob

PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_all_oot")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/mfe_mae_labels_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pred_files = sorted(glob(str(PRED_DIR / "*_predictions.npz")))
print(f"Found {len(pred_files)} prediction files")

all_mfe = {h: [] for h in ['1s','5s','10s','30s']}
all_mae = {h: [] for h in ['1s','5s','10s','30s']}

processed = 0
skipped = 0

for idx, pf in enumerate(pred_files):
    date = Path(pf).stem.replace("_predictions", "")
    mbo_path = MBO_DIR / f"{date}_mbo_events.npz"

    if not mbo_path.exists():
        skipped += 1
        continue

    # Load predictions
    pd = np.load(pf, allow_pickle=True)
    predictions = pd['predictions']  # (n_windows, 3)
    n_windows = int(pd['n_windows'])
    window_size = int(pd['window_size'])
    stride = int(pd['stride'])

    # Load MBO events
    md = np.load(mbo_path, allow_pickle=True)
    labels_1s  = md['labels_1s']
    labels_5s  = md['labels_5s']
    labels_10s = md['labels_10s']
    labels_30s = md['labels_30s']
    N_events = len(labels_1s)

    # Build aligned event indices
    event_indices = np.array([i * stride + window_size - 1 for i in range(n_windows)])

    # Filter valid indices (within MBO event array)
    valid_mask = event_indices < N_events
    event_indices = event_indices[valid_mask]
    preds_valid = predictions[valid_mask]
    n_valid = len(event_indices)

    if n_valid == 0:
        skipped += 1
        continue

    # Extract labels at aligned positions
    l1  = labels_1s[event_indices].astype(np.float32)
    l5  = labels_5s[event_indices].astype(np.float32)
    l10 = labels_10s[event_indices].astype(np.float32)
    l30 = labels_30s[event_indices].astype(np.float32)

    # MFE = max favorable excursion for SHORT trades (negative label = price dropped = good for short)
    # MAE = max adverse excursion for SHORT trades (positive label = price rose = bad for short)
    # NaN labels (boundary/missing events) are treated as 0 (no info = no excursion)
    def safe_max(*arrays):
        """Element-wise max treating NaN as 0."""
        stacked = np.stack(arrays, axis=0)
        return np.nanmax(stacked, axis=0).astype(np.float32)

    zeros = np.zeros(n_valid, dtype=np.float32)

    mfe_1s  = safe_max(zeros, -l1)
    mfe_5s  = safe_max(zeros, -l1, -l5)
    mfe_10s = safe_max(zeros, -l1, -l5, -l10)
    mfe_30s = safe_max(zeros, -l1, -l5, -l10, -l30)

    mae_1s  = safe_max(zeros, l1)
    mae_5s  = safe_max(zeros, l1, l5)
    mae_10s = safe_max(zeros, l1, l5, l10)
    mae_30s = safe_max(zeros, l1, l5, l10, l30)

    # Save
    out_path = OUT_DIR / f"{date}_mfe_mae.npz"
    np.savez_compressed(
        out_path,
        mfe_1s=mfe_1s, mfe_5s=mfe_5s, mfe_10s=mfe_10s, mfe_30s=mfe_30s,
        mae_1s=mae_1s, mae_5s=mae_5s, mae_10s=mae_10s, mae_30s=mae_30s,
        predictions=preds_valid,
        date=date,
        n_valid=n_valid
    )

    # Accumulate for summary
    for key, arr in [('1s',mfe_1s),('5s',mfe_5s),('10s',mfe_10s),('30s',mfe_30s)]:
        all_mfe[key].append(arr)
    for key, arr in [('1s',mae_1s),('5s',mae_5s),('10s',mae_10s),('30s',mae_30s)]:
        all_mae[key].append(arr)

    processed += 1
    if processed % 10 == 0:
        print(f"  [{processed}] processed {processed} dates so far (last: {date}, n_valid={n_valid})")

print(f"\nDone. Processed={processed}, Skipped={skipped}")
print("\n=== Summary Stats (across all dates, SHORT perspective, in TICKS) ===")
print(f"{'Horizon':<10} {'MFE mean':>10} {'MFE p50':>10} {'MFE p90':>10} | {'MAE mean':>10} {'MAE p50':>10} {'MAE p90':>10}")
print("-" * 80)
for h in ['1s', '5s', '10s', '30s']:
    mfe_all = np.concatenate(all_mfe[h])
    mae_all = np.concatenate(all_mae[h])
    print(f"{h:<10} {mfe_all.mean():>10.4f} {np.median(mfe_all):>10.4f} {np.percentile(mfe_all,90):>10.4f} | "
          f"{mae_all.mean():>10.4f} {np.median(mae_all):>10.4f} {np.percentile(mae_all,90):>10.4f}")

print(f"\nTotal predictions analyzed: {sum(len(x) for x in all_mfe['1s']):,}")
print(f"Output directory: {OUT_DIR}")
