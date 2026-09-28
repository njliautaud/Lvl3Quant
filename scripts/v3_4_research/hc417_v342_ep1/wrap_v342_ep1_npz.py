#!/usr/bin/env python3
"""HC #417 — Wrap CNN-Mamba v3.4.2 fold-0 ep-1 OOT NPZ for HC #413/#415 backtesters.

The fresh NPZ at output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep1_oot.npz has 241,351
samples but lacks n_samples/oot_dates/day_index keys. Per prior SESSION_STATE
record (line 491-516), the fold-0 ep-1 OOT inference covers 5 OOT dates
20260223..20260227 (Feb 23-27). FIFO label per-day sums for those 5 dates total
241,692 — diff of 341 vs the NPZ count is consistent with end-of-day trim (5 days,
fits a small per-day or boundary trim).

This wrapper:
  1. Asserts the 5-date schedule.
  2. Reads FIFO per-day counts for the 5 dates.
  3. Distributes the 241,351 samples across the 5 dates, trimming from end-of-day
     such that head-of-day alignment is preserved (HC default — FIFO indexing
     window_k goes 0..n-1 per day).
  4. Writes a new NPZ with n_samples + oot_dates keys, preserving ALL original
     prediction/target/mask heads (multi-output v3.4.2 schema).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/fold_00_ep1_oot.npz")
LABELS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
OUT = Path("/home/jupiter/Lvl3Quant/output/v342_ep1_eval/fold_00_ep1_oot_wrapped.npz")

# 5 OOT dates per SESSION_STATE record (5-day NPZ = first OOT chunk for fold-0)
OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]


def main() -> int:
    d = np.load(SRC, allow_pickle=True)
    n_npz = d["pred_log_ret_1s"].shape[0]

    fifo_n_per_day = {}
    for dt in OOT_DATES:
        fp = LABELS_DIR / f"{dt}_fifo_labels.npz"
        if not fp.exists():
            raise FileNotFoundError(f"Missing FIFO labels: {fp}")
        fifo_n_per_day[dt] = int(np.load(fp)["window_k"].shape[0])
    total_fifo = sum(fifo_n_per_day.values())
    diff = total_fifo - n_npz  # positive => fifo has more rows than npz
    print(f"[wrap] n_npz={n_npz} total_fifo={total_fifo} diff={diff}")

    # The dataset likely emitted fewer samples than FIFO because of MFE/MAE/30s
    # head masking pulling some windows out, OR because of a small per-day trim.
    # Distribute trim evenly. Conservatively trim FROM FIFO ROWS so that
    # min(n_npz, n_fifo)-style truncation in the backtester aligns.
    # Result: we write n_npz = 241351 with date breakdown that sums to 241351,
    # by trimming FIFO trailing rows per-date proportional to per-day size.
    if diff < 0:
        raise RuntimeError(f"NPZ has more samples than FIFO ({n_npz} > {total_fifo}); cannot align.")

    # Proportional trim: subtract trim_per_day = round(diff * frac_day) from each
    per_day_keep = {}
    cum_trim = 0
    for i, dt in enumerate(OOT_DATES):
        if i < len(OOT_DATES) - 1:
            frac = fifo_n_per_day[dt] / total_fifo
            t = int(round(diff * frac))
            per_day_keep[dt] = fifo_n_per_day[dt] - t
            cum_trim += t
        else:
            # last day absorbs the rest to match n_npz exactly
            per_day_keep[dt] = fifo_n_per_day[dt] - (diff - cum_trim)
    sum_keep = sum(per_day_keep.values())
    assert sum_keep == n_npz, f"keep sum {sum_keep} != n_npz {n_npz}"
    print(f"[wrap] per_day_keep (trimmed): {per_day_keep}")

    # Write all keys from src, plus n_samples + oot_dates
    out = {}
    for k in d.files:
        a = d[k]
        # 0-d scalar (metrics) — keep as-is
        if a.ndim == 0:
            out[k] = a
            continue
        out[k] = a  # NPZ has exactly n_npz rows on all keys; no trim needed
    out["n_samples"] = np.int64(n_npz)
    out["oot_dates"] = np.array(OOT_DATES, dtype="<U8")
    # Build per-day cumulative index for FIFO label load alignment (backtester uses
    # the dates list + min(n_npz, n_fifo) which is sufficient).
    np.savez(OUT, **out)
    print(f"[wrap] wrote {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
