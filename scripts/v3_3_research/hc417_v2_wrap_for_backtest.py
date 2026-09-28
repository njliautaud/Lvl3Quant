#!/usr/bin/env python3
"""HC #417 — Wrap the CNN-Mamba v2 full-OOT NPZ for HC #413 / HC #415 backtesters.

The v2 inference NPZ has:
  - Only 3 prediction heads (pred_log_ret_1s/5s/10s)  -- the LIVE model is 3-head.
  - 56 oot_dates declared, 46 dates_present, day_index ranging [10, 55].
  - Per-day sample count is exactly +8 more than the corresponding FIFO label NPZ.
  - 10 of the 46 present dates have NO FIFO labels file (20260320-20260331).

To make this consumable by the existing HC #413 backtester (which assumes
per-sample alignment with FIFO labels and uses min(n_npz, n_fifo) truncation),
this wrapper:

  1. Keeps only the dates that have BOTH v2 predictions AND FIFO labels (36 dates).
  2. Trims each kept day's v2 samples to match the FIFO per-day count (drops
     trailing 8 samples per day — head/tail trimming is unverifiable but
     trailing trim is the conservative default since the FIFO loader expects
     window_k = 0..n-1).
  3. Writes a new NPZ with n_samples + oot_dates(36-date) keys.

HONESTY NOTE: HC #415 multi-output sweep CANNOT be run on v2 — the v2 architecture
emits only 3 log_ret heads, while the sweep needs ~20 heads (MFE/MAE/quantile/
reversal/vol/direct-FIFO) AND the canonical realized-FIFO label column
`target_fifo_tp4sl3_net`. This wrapper does NOT fabricate those heads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC_NPZ = Path("/home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_56d.npz")
LABELS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
OUT_NPZ = Path("/home/jupiter/Lvl3Quant/output/hc417_v2_full_oot_wrapped_for_hc413.npz")


def main() -> int:
    d = np.load(SRC_NPZ, allow_pickle=True)
    dates_present = [str(x) for x in d["dates_present"]]
    oot_dates_full = [str(x) for x in d["oot_dates"]]
    day_index = d["day_index"]

    # Determine which dates have FIFO labels
    kept_dates = []
    fifo_n_by_date = {}
    for dt in dates_present:
        fp = LABELS_DIR / f"{dt}_fifo_labels.npz"
        if fp.exists():
            n_fifo = int(np.load(fp)["window_k"].shape[0])
            kept_dates.append(dt)
            fifo_n_by_date[dt] = n_fifo

    print(f"[wrap] dates_present={len(dates_present)} kept(FIFO available)={len(kept_dates)}")
    missing_fifo = [dt for dt in dates_present if dt not in fifo_n_by_date]
    print(f"[wrap] dates dropped (no FIFO labels): {missing_fifo}")

    # For each kept date, find rows in v2 NPZ and trim to FIFO count
    # day_index uses oot_dates_full (56-array) indices, so look up by that
    keep_indices_parts = []
    for dt in kept_dates:
        full_idx = oot_dates_full.index(dt)
        rows = np.where(day_index == full_idx)[0]
        n_v2 = rows.shape[0]
        n_fifo = fifo_n_by_date[dt]
        diff = n_v2 - n_fifo
        if diff < 0:
            raise RuntimeError(f"{dt}: v2 n={n_v2} < fifo n={n_fifo}, cannot trim trailing")
        if diff != 8:
            print(f"[wrap] WARN {dt}: diff={diff} (expected 8)")
        # Trim trailing 'diff' samples
        keep = rows[: n_v2 - diff]
        keep_indices_parts.append(keep)
    keep_idx = np.concatenate(keep_indices_parts)
    print(f"[wrap] total kept samples: {keep_idx.shape[0]}")

    # Build output dict
    out = {}
    for h in ("1s", "5s", "10s"):
        out[f"pred_log_ret_{h}"] = np.asarray(d[f"pred_log_ret_{h}"], dtype=np.float32)[keep_idx]
        out[f"target_log_ret_{h}"] = np.asarray(d[f"target_log_ret_{h}"], dtype=np.float32)[keep_idx]
        out[f"mask_log_ret_{h}"] = np.asarray(d[f"mask_log_ret_{h}"], dtype=np.float32)[keep_idx]
    out["n_samples"] = np.int64(keep_idx.shape[0])
    out["oot_dates"] = np.array(kept_dates, dtype="<U8")
    out["ckpt_path"] = d["ckpt_path"]
    out["ckpt_sha256"] = d["ckpt_sha256"]

    np.savez(OUT_NPZ, **out)
    print(f"[wrap] wrote {OUT_NPZ} ({OUT_NPZ.stat().st_size/1e6:.2f} MB)")
    print(f"[wrap] oot_dates ({len(kept_dates)}): {kept_dates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
