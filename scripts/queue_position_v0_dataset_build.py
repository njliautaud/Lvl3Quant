#!/usr/bin/env python3
"""
queue_position_v0_dataset_build.py — HC #471 R5 / HC #469 R5(a) — queue-position model dataset.

For each fill in surviving_canonical_fifo_fills.parquet (20939 rows, 16 days),
extract a WINDOW of pre-signal book features ending at ts_signal_ns, and
pair it with the realized queue_ahead label from the fill record.

Window: last K=200 book events before signal time (~50 ms at typical activity).
Features: 30 book features per event (5-level bid/ask × price+size + 10 derived).

Output:
  output/queue_position_v0/dataset.npz
    X: float32 [N, K, F=30]
    y: float32 [N]                  (queue_ahead, raw count)
    y_log: float32 [N]              (log(1 + queue_ahead) — model target)
    dates: <U8 [N]                  (split key)
    directions: <U6 [N]
    pred_strengths: float32 [N]     (auxiliary)
    feature_names: <U22 [F]
  output/queue_position_v0/manifest.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
FILLS = LVL3 / "output/stream_backtest_v2/surviving_canonical_fifo_fills.parquet"
BOOK_DIR = LVL3 / "data/processed/mbo_book_features"
OUT_DIR = LVL3 / "output/queue_position_v0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K = 200  # event window length
MIN_QUEUE = 1


def main():
    t0 = time.time()
    fills = pd.read_parquet(FILLS)
    print(f"[build] fills loaded: {len(fills)}")
    fills = fills[fills["queue_ahead"] >= MIN_QUEUE].copy()
    print(f"[build] post queue>=1 filter: {len(fills)}")

    dates = sorted(fills["date"].unique())
    print(f"[build] dates: {dates}")

    X_list, y_list, date_list, dir_list, ps_list = [], [], [], [], []
    feature_names = None

    for di, d in enumerate(dates):
        npz_path = BOOK_DIR / f"{d}_book_features.npz"
        if not npz_path.exists():
            print(f"[build] [{di+1}/{len(dates)}] {d} — MISSING book NPZ, skip")
            continue
        ar = np.load(npz_path, allow_pickle=True)
        ts = ar["timestamps"]  # int64 ns
        feats = ar["features"]  # [M, 30]
        if feature_names is None:
            feature_names = list(ar["feature_names"])

        day_fills = fills[fills["date"] == d]
        n_day = 0
        for _, row in day_fills.iterrows():
            t_sig = int(row["ts_signal_ns"])
            # right-search for first ts > t_sig, then take K events BEFORE that
            idx = int(np.searchsorted(ts, t_sig, side="right"))
            if idx < K:
                continue
            window = feats[idx - K: idx]  # [K, 30]
            if window.shape != (K, 30):
                continue
            X_list.append(window.astype(np.float32))
            y_list.append(float(row["queue_ahead"]))
            date_list.append(d)
            dir_list.append(str(row["direction"]))
            ps_list.append(float(row.get("pred_strength", 0.0)))
            n_day += 1
        print(f"[build] [{di+1}/{len(dates)}] {d} — {n_day} rows extracted")
        del ar, ts, feats

    X = np.stack(X_list, axis=0)  # [N, K, F]
    y = np.array(y_list, dtype=np.float32)
    y_log = np.log1p(y)
    dates_arr = np.array(date_list, dtype="<U8")
    dirs_arr = np.array(dir_list, dtype="<U6")
    ps_arr = np.array(ps_list, dtype=np.float32)

    print(f"[build] FINAL: X={X.shape} y={y.shape} y range=[{y.min()},{y.max()}] mean={y.mean():.2f}")

    np.savez_compressed(
        OUT_DIR / "dataset.npz",
        X=X, y=y, y_log=y_log,
        dates=dates_arr, directions=dirs_arr, pred_strengths=ps_arr,
        feature_names=np.array(feature_names, dtype="<U22"),
    )
    manifest = {
        "n_samples": int(X.shape[0]),
        "window_K": K,
        "n_features": int(X.shape[2]),
        "dates": dates,
        "y_stats": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "median": float(np.median(y)),
            "p25": float(np.percentile(y, 25)),
            "p75": float(np.percentile(y, 75)),
            "p95": float(np.percentile(y, 95)),
        },
        "wall_s": time.time() - t0,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "dataset.DONE").write_text(f"completed {time.strftime('%Y-%m-%d %H:%M:%S ET')}\n")
    print(f"[build] DONE in {time.time() - t0:.1f}s -> {OUT_DIR/'dataset.npz'}")


if __name__ == "__main__":
    main()
