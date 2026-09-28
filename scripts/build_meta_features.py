#!/usr/bin/env python3
"""
build_meta_features.py — Assemble a flat meta-features parquet from CNN-Mamba v4
per-day OOT NPZ files. Rows = events; columns = 32 prediction heads + targets +
date + pred_k.

Output: /home/jupiter/Lvl3Quant/output/razer_meta/meta_features.parquet
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
PERDAY_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR = ROOT / "output/razer_meta"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = [
    "target_log_ret_1s",
    "target_log_ret_5s",
    "target_log_ret_10s",
    "target_log_ret_30s",
]


def load_day(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    pred_keys = sorted([k for k in d.files if k.startswith("pred_")])
    if not pred_keys:
        return None
    n = d[pred_keys[0]].shape[0]
    cols = {k: d[k] for k in pred_keys}
    for tc in TARGET_COLS:
        if tc in d.files:
            cols[tc] = d[tc]
        else:
            cols[tc] = np.full(n, np.nan, dtype=np.float32)
    date_str = npz_path.stem.replace("oot_", "")
    cols["date"] = np.full(n, date_str, dtype=object)
    cols["pred_k"] = np.arange(n, dtype=np.int64)
    return pd.DataFrame(cols)


def main():
    npz_files = sorted(PERDAY_DIR.glob("oot_*.npz"))
    print(f"[build_meta_features] {len(npz_files)} per-day NPZs found.")
    frames = []
    for i, p in enumerate(npz_files, start=1):
        df = load_day(p)
        if df is None:
            print(f"  SKIP {p.name} (no pred_ keys)")
            continue
        frames.append(df)
        if i % 5 == 0 or i == len(npz_files):
            print(f"  loaded {i}/{len(npz_files)} days. last={p.stem} rows={len(df)}")
    big = pd.concat(frames, ignore_index=True)
    print(f"[build_meta_features] concat shape={big.shape}")
    pred_cols = sorted([c for c in big.columns if c.startswith("pred_")])
    print(f"[build_meta_features] {len(pred_cols)} pred_* columns.")
    target_present = [c for c in TARGET_COLS if c in big.columns]
    print(f"[build_meta_features] target_present={target_present}")
    out_path = OUT_DIR / "meta_features.parquet"
    big.to_parquet(out_path, index=False)
    print(f"[build_meta_features] wrote {out_path} size={out_path.stat().st_size/1e6:.1f} MB")

    # Sanity
    nan_targets = {c: int(big[c].isna().sum()) for c in target_present}
    print(f"[build_meta_features] NaN counts in targets: {nan_targets}")
    print(f"[build_meta_features] unique dates: {big['date'].nunique()}")
    print(f"[build_meta_features] events per date (head): {big.groupby('date').size().head()}")


if __name__ == "__main__":
    main()
