#!/usr/bin/env python3
"""
Merge per-date v3.4.2 predictions into a single NPZ for RL training.

The CanonicalReplayEnv expects a single NPZ with:
  - oot_dates: array of date strings
  - n_samples: total sample count
  - pred_*/target_*/mask_* arrays: concatenated across all dates

The v3.4.2 per-date predictions have all 32 PRED_HEADS plus targets/masks.
This script merges them into env-compatible format, giving the RL 34 days
instead of the 5 from fold_00.

NOT MALWARE. Merge utility for RL training data.
"""

import os
import sys
import numpy as np
from pathlib import Path

PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
LABEL_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
OUTPUT = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/merged_34day_for_rl.npz")

# All arrays to concatenate (must exist in per-date NPZs)
ARRAY_KEYS = [
    # 32 PRED_HEADS
    "pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s",
    "pred_log_ret_30s", "pred_log_ret_60s", "pred_log_ret_5min",
    "pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s", "pred_p_up_60s",
    "pred_log_ret_10s_q10", "pred_log_ret_10s_q50", "pred_log_ret_10s_q90",
    "pred_log_ret_30s_q10", "pred_log_ret_30s_q50", "pred_log_ret_30s_q90",
    "pred_log_ret_60s_q10", "pred_log_ret_60s_q50", "pred_log_ret_60s_q90",
    "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks",
    "pred_pred_mfe_60s_ticks", "pred_pred_mae_60s_ticks",
    "pred_pred_time_to_mfe_secs",
    "pred_p_reversal_15s", "pred_p_reversal_30s", "pred_p_reversal_60s",
    "pred_pred_realized_vol_30s_ticks",
    "pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net",
    "pred_fifo_tp4sl3_hit_tp", "pred_fifo_tp8sl5_hit_tp",
    # Targets + masks needed by env
    "target_log_ret_30s", "mask_log_ret_30s",
    "target_log_ret_5s", "mask_log_ret_5s",
    "target_log_ret_1s", "mask_log_ret_1s",
    # Additional targets the env might reference
    "pred_log_ret_5s",  # already in PRED_HEADS but listed for clarity
]

# Deduplicate
ARRAY_KEYS = list(dict.fromkeys(ARRAY_KEYS))


def main():
    # Find matched dates
    pred_dates = set()
    for f in os.listdir(PRED_DIR):
        if f.startswith("oot_") and f.endswith(".npz"):
            pred_dates.add(f.replace("oot_", "").replace(".npz", ""))

    label_dates = set()
    for f in os.listdir(LABEL_DIR):
        if f.endswith("_fifo_labels.npz"):
            label_dates.add(f.replace("_fifo_labels.npz", ""))

    matched = sorted(pred_dates & label_dates)
    print(f"Matched dates: {len(matched)}")

    # Load and concatenate
    buffers = {k: [] for k in ARRAY_KEYS}
    n_total = 0

    valid_dates = []
    for date_str in matched:
        npz_path = PRED_DIR / f"oot_{date_str}.npz"
        data = np.load(npz_path, allow_pickle=True)

        # Skip empty/weekend dates
        if ARRAY_KEYS[0] not in data:
            print(f"  {date_str}: SKIPPED (no prediction arrays — weekend/holiday)")
            continue

        n_samples = data[ARRAY_KEYS[0]].shape[0]
        if n_samples < 100:
            print(f"  {date_str}: SKIPPED ({n_samples} samples — too few)")
            continue

        n_total += n_samples
        valid_dates.append(date_str)

        for key in ARRAY_KEYS:
            if key in data:
                buffers[key].append(data[key])
            else:
                print(f"  WARNING: {key} missing from {date_str}, using zeros")
                buffers[key].append(np.zeros(n_samples, dtype=np.float32))

        print(f"  {date_str}: {n_samples:,} samples")

    matched = valid_dates

    # Concatenate
    merged = {}
    for key in ARRAY_KEYS:
        merged[key] = np.concatenate(buffers[key], axis=0)
        print(f"  {key}: {merged[key].shape}")

    merged["oot_dates"] = np.array(matched)
    merged["n_samples"] = np.int64(n_total)

    # Verify
    print(f"\nTotal: {n_total:,} samples across {len(matched)} dates")
    print(f"oot_dates: {merged['oot_dates']}")

    # Save
    np.savez_compressed(OUTPUT, **merged)
    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    print(f"\nSaved: {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
