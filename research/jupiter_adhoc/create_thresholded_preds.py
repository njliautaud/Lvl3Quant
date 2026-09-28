#!/usr/bin/env python3
"""
Create thresholded per-day NPZ prediction files for Rust fill_sim_cli.
Applies signal threshold to reduce trade count to reasonable levels.

Run on Jupiter: python3 /home/jupiter/create_thresholded_preds.py
"""
import numpy as np
import os
from pathlib import Path

PRED_DIR = Path("/home/jupiter/lvl3quant/data/processed/rust_predictions")
OUT_BASE = Path("/home/jupiter/lvl3quant/data/processed/rust_predictions_thresh")

# Thresholds to test: matching the Saturn sweep (0.3, 0.5, 0.7, 0.9)
# Plus some finer ones based on actual pred distribution
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]

def apply_threshold(preds, thresh):
    """Zero out predictions below threshold. Keep sign."""
    result = preds.copy()
    result[np.abs(result) <= thresh] = 0.0
    return result

# Process each date
pred_files = sorted(PRED_DIR.glob("*_predictions.npz"))
print(f"Found {len(pred_files)} prediction files")

for thresh in THRESHOLDS:
    out_dir = OUT_BASE / f"thresh_{thresh:.1f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_signals = 0
    total_bars = 0
    for fpath in pred_files:
        d = np.load(str(fpath))
        preds = d["predictions"]
        p10 = d.get("predictions_10s", d["predictions"])

        # Apply threshold to primary (30s) predictions
        preds_thresh = apply_threshold(preds, thresh)
        p10_thresh = apply_threshold(p10, thresh)

        n_signals = int((preds_thresh != 0).sum())
        total_signals += n_signals
        total_bars += len(preds_thresh)

        date_str = fpath.name[:10]
        out_path = out_dir / fpath.name
        np.savez_compressed(
            str(out_path),
            predictions=preds_thresh,
            predictions_10s=p10_thresh,
        )

    avg_signals_per_day = total_signals / len(pred_files)
    print(f"thresh={thresh:.1f}: {total_signals} total signals, "
          f"{avg_signals_per_day:.0f}/day avg "
          f"({100*total_signals/total_bars:.1f}% of bars)")

print(f"\nThresholded prediction files ready in {OUT_BASE}/")
print("Directories:")
for thresh in THRESHOLDS:
    d = OUT_BASE / f"thresh_{thresh:.1f}"
    count = len(list(d.glob("*.npz")))
    print(f"  {d}: {count} files")
