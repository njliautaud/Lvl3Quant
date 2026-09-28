#!/usr/bin/env python3
"""
build_stream_native_labels.py — HC #470 R1 Phase A relabeling engine.

For each champion-overlap OOT day with v4 multi-head predictions, derive
the four stream-native label families that are computable from the
existing pred/target arrays (no new MBO inference needed):

  (a) stream_life_k       int     # of consecutive future strides where
                                  sign(pred_t+k) == sign(pred_t) until first flip
  (b) stream_int_return   float   sum of target_log_ret_1s from t to t+stream_life_k
  (c) sign_flip_latency   int     same as stream_life_k (in strides; convert
                                  to seconds with stride_seconds if available)
  (d) stream_mfe          float   max of cumulative target_log_ret_1s over
                                  [t, t+stream_life_k]
      stream_mae          float   min of same

Reference signal: pred_log_ret_1s (the 1s entry-trigger head).
Optional: also run with pred_log_ret_5s as reference for ablation.

Inputs:
  output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_<date>.npz

Outputs:
  output/stream_native_labels/labels_<date>.npz
  output/stream_native_labels/stream_labels_summary.json

Pure NumPy. Causal — every label at t uses only data at t and after.
No leakage relative to inference (the model has access to t's prediction
at decision time, the LABEL uses future ground truth which is standard
supervised training).

Per HC #470 R1, snapshot labels stay — these are NEW heads, not replacements.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
IN_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUT_DIR = ROOT / "output/stream_native_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_HEADS = ["pred_log_ret_1s", "pred_log_ret_5s"]
MAX_HORIZON_STRIDES = 240  # cap stream-life at 240 strides (= 60s at 4Hz)


def build_for_day(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    date_str = str(d["oot_dates"][0]) if "oot_dates" in d.files else npz_path.stem.replace("oot_", "")
    target_1s = d["target_log_ret_1s"].astype(np.float64)
    valid_mask = ~np.isnan(target_1s)
    out: dict[str, np.ndarray] = {"date": np.array([date_str], dtype="<U8")}
    summary = {"date": date_str, "n_events": int(len(target_1s)), "ref_heads": {}}

    for ref in REF_HEADS:
        if ref not in d.files:
            continue
        pred = d[ref].astype(np.float64)
        n = len(pred)
        sgn = np.sign(pred)

        # (a) stream_life_k: how many consecutive future strides keep same sign?
        # vectorized: scan forward up to MAX_HORIZON_STRIDES
        # For each t, find smallest k>=1 such that sgn[t+k] != sgn[t] OR k==MAX.
        # Build via a forward sweep using a stack of "last flip index" — O(n).
        # We approximate by capped forward scan (O(n * cap)) — n=60k, cap=240 -> 14M ops, fine.
        stream_life = np.zeros(n, dtype=np.int32)
        # Use cumulative-mismatch trick: for each t compute first index where sgn != sgn[t]
        # Efficient pass:
        # We iterate from end to start, maintaining run-length-ahead.
        rle = np.zeros(n, dtype=np.int32)
        # rle[i] = number of consecutive future strides with same sign as sgn[i], starting at i+1
        for i in range(n - 2, -1, -1):
            if sgn[i + 1] == sgn[i] and sgn[i] != 0:
                rle[i] = min(rle[i + 1] + 1, MAX_HORIZON_STRIDES)
            else:
                rle[i] = 0
        stream_life = rle

        # (b) stream_int_return: sum of target_log_ret_1s over [t+1, t+stream_life_k]
        # Use prefix sums of target_1s (set NaNs to 0 for summation; track mask).
        tgt_zeroed = np.where(np.isnan(target_1s), 0.0, target_1s)
        prefix = np.concatenate([[0.0], np.cumsum(tgt_zeroed)])
        # sum(target_1s[t+1 : t+1+k]) = prefix[t+1+k] - prefix[t+1]
        k = stream_life
        end_idx = np.minimum(np.arange(n) + 1 + k, n)
        start_idx = np.minimum(np.arange(n) + 1, n)
        stream_int = prefix[end_idx] - prefix[start_idx]

        # (c) sign_flip_latency: same as stream_life in stride units (we don't have ts_ns to convert to seconds — left as stride count)
        sign_flip_lat = stream_life.copy()

        # (d) stream_mfe / stream_mae: max / min of cumulative target return within stream-life
        # For each t, compute max & min of cumsum(target_1s[t+1 : t+1+k]) - using cumulative max/min slabs.
        # Simple approach: for each t, iterate up to k strides (capped 240). Same O(n*cap) cost.
        stream_mfe = np.zeros(n, dtype=np.float64)
        stream_mae = np.zeros(n, dtype=np.float64)
        # Precompute cumsum per starting point using rolling — but cap=240 so direct loop is fine for now.
        # Vectorize via numpy: build an (n, cap) window via stride tricks, mask by k, take max/min.
        # For memory safety n=60k x 240 = 14.4M floats = 115MB — acceptable.
        cap = MAX_HORIZON_STRIDES
        # Pad target_1s_zeroed with zeros at the end
        padded = np.concatenate([tgt_zeroed, np.zeros(cap, dtype=np.float64)])
        # Build cumulative-sum-from-t+1 matrix: cum[t, j] = sum(target[t+1 : t+1+j+1])
        # Equivalent: cumsum along axis-1 of windowed view starting at t+1.
        # Use stride trick:
        from numpy.lib.stride_tricks import sliding_window_view
        if n + cap > cap:
            windows = sliding_window_view(padded[1:], cap)[:n]  # shape (n, cap)
        else:
            windows = np.zeros((n, cap), dtype=np.float64)
        cum_windows = np.cumsum(windows, axis=1)
        # Mask rows beyond stream_life
        col_idx = np.arange(cap)[None, :]
        mask_k = col_idx < stream_life[:, None]
        # Where mask_k is False, set to NaN so they don't affect max/min
        cum_masked_max = np.where(mask_k, cum_windows, -np.inf)
        cum_masked_min = np.where(mask_k, cum_windows,  np.inf)
        stream_mfe = np.where(stream_life > 0, np.max(cum_masked_max, axis=1), 0.0)
        stream_mae = np.where(stream_life > 0, np.min(cum_masked_min, axis=1), 0.0)

        prefix_key = ref.replace("pred_", "")  # e.g. "log_ret_1s"
        out[f"stream_life_{prefix_key}"] = stream_life.astype(np.int32)
        out[f"stream_int_return_{prefix_key}"] = stream_int.astype(np.float32)
        out[f"sign_flip_latency_{prefix_key}"] = sign_flip_lat.astype(np.int32)
        out[f"stream_mfe_{prefix_key}"] = stream_mfe.astype(np.float32)
        out[f"stream_mae_{prefix_key}"] = stream_mae.astype(np.float32)
        out[f"ref_sign_{prefix_key}"] = sgn.astype(np.int8)
        out[f"valid_{prefix_key}"] = valid_mask.astype(np.bool_)

        summary["ref_heads"][ref] = {
            "mean_stream_life_strides": float(np.mean(stream_life)),
            "p50_stream_life_strides": float(np.median(stream_life)),
            "p90_stream_life_strides": float(np.percentile(stream_life, 90)),
            "mean_stream_int_return": float(np.nanmean(stream_int)),
            "frac_positive_stream_int": float(np.mean(stream_int > 0)),
            "mean_stream_mfe": float(np.nanmean(stream_mfe)),
            "mean_stream_mae": float(np.nanmean(stream_mae)),
        }

    out_path = OUT_DIR / f"labels_{date_str}.npz"
    np.savez_compressed(out_path, **out)
    return summary


def main():
    t0 = time.time()
    print("=" * 78)
    print("HC #470 R1 — STREAM-NATIVE LABEL BUILDER (Phase A)")
    print("=" * 78)

    npz_paths = sorted(IN_DIR.glob("oot_*.npz"))
    print(f"[setup] found {len(npz_paths)} v4 OOT NPZs at {IN_DIR}")

    all_summaries = []
    for i, npz in enumerate(npz_paths):
        try:
            s = build_for_day(npz)
            all_summaries.append(s)
            ref_summaries = ", ".join(
                f"{ref}: meanK={s['ref_heads'][ref]['mean_stream_life_strides']:.1f} pctPos={s['ref_heads'][ref]['frac_positive_stream_int']:.2%}"
                for ref in s.get("ref_heads", {})
            )
            print(f"  [{i+1}/{len(npz_paths)}] {s['date']}  n={s['n_events']:,}  {ref_summaries}")
        except Exception as ex:
            print(f"  [{i+1}/{len(npz_paths)}] {npz.name}  ERROR: {ex}")

    summary_path = OUT_DIR / "stream_labels_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "hc": "HC #470 R1",
                "ref_heads": REF_HEADS,
                "max_horizon_strides": MAX_HORIZON_STRIDES,
                "days": all_summaries,
                "n_days": len(all_summaries),
            },
            f,
            indent=2,
        )
    print(f"\n[done] {len(all_summaries)} days in {time.time()-t0:.1f}s -> {OUT_DIR}")
    print(f"[done] summary -> {summary_path}")
    (OUT_DIR / "stream_native_labels.DONE").touch()


if __name__ == "__main__":
    main()
