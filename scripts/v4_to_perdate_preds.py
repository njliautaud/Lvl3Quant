#!/usr/bin/env python3
"""
Convert V4 multihead per-fold predictions to per-date format
compatible with the tick_level_replay engine.

The v4 model saves per-fold NPZ with:
  preds_dir: (N, 3)  → directional predictions at 1s, 5s, 10s
  labels_dir: (N, 3)  → actual returns at 1s, 5s, 10s

We need per-date files with:
  pred_log_ret_1s: (N,)  → 1s prediction (for signal direction)
  pred_log_ret_10s: (N,) → 10s prediction (optional)
  pred_confidence: (N,)  → |prediction| magnitude (for top-5% filter)

The stride/window used in v4 training:
  stride=2000, seq_len=100
  vs v3.4.2: stride=250, window=1500

This means v4 has ~8x fewer predictions per day.
The tick replay will need to be adapted for the sparser signal.
"""

import numpy as np
from pathlib import Path
import sys

ROOT = Path("/home/jupiter/Lvl3Quant")
V4_DIR = ROOT / "output" / "v4_multihead_pressure_v1"
OUT_DIR = ROOT / "output" / "v4_perdate_preds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fold_files = sorted(V4_DIR.glob("fold_*_oot_predictions.npz"))
print(f"Found {len(fold_files)} fold files")

for fp in fold_files:
    data = np.load(str(fp), allow_pickle=True)
    fold = int(data['fold'])
    oot_file = str(data['oot_files'][0])
    date_str = Path(oot_file).stem.split('_')[0]

    preds_dir = data['preds_dir']  # (N, 3)
    labels_dir = data['labels_dir']  # (N, 3)

    # Use 10s horizon predictions as the primary signal (best net edge)
    pred_1s = preds_dir[:, 0]
    pred_5s = preds_dir[:, 1] if preds_dir.shape[1] > 1 else np.full(len(preds_dir), np.nan)
    pred_10s = preds_dir[:, 2] if preds_dir.shape[1] > 2 else np.full(len(preds_dir), np.nan)

    # Confidence = |pred_10s| (since 10s has best net edge)
    confidence = np.abs(pred_10s)

    out_path = OUT_DIR / f"oot_{date_str}.npz"
    np.savez(str(out_path),
        pred_log_ret_1s=pred_1s.astype(np.float32),
        pred_log_ret_5s=pred_5s.astype(np.float32),
        pred_log_ret_10s=pred_10s.astype(np.float32),
        pred_confidence=confidence.astype(np.float32),
        label_1s=labels_dir[:, 0].astype(np.float32),
        label_5s=labels_dir[:, 1].astype(np.float32) if labels_dir.shape[1] > 1 else np.array([]),
        label_10s=labels_dir[:, 2].astype(np.float32) if labels_dir.shape[1] > 2 else np.array([]),
        model='v4_multihead',
        stride=2000,
        seq_len=100,
    )
    print(f"  {date_str}: {len(pred_1s)} predictions → {out_path.name}")

print(f"\nDone. {len(fold_files)} files written to {OUT_DIR}")
