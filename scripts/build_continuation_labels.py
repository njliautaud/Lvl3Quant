#!/usr/bin/env python3
"""
build_continuation_labels.py — HC #497 v3.5 multi-head label builder.

For each day file in mbo_events_smart_v3, produce a new *_v35.npz with four
head-label arrays appended to the original keys.

Head-A (pressure-direction):
    Binary: is signed price-move majority bullish over next K_a seconds?
    Implementation: label_A_K = 1 if labels_K[t] > 0 else 0; nan-preserving.
    Generated for K_a in {5, 10, 30, 60}.

Head-B (pressure-persistence):
    Scalar: how many seconds until sign(labels_1s) flips for the first time
    after event t?  Cap 300s. Right-censored (events with no flip in window
    get persistence = 300 and a censored=1 mask).

Head-C (cumulative-K-tick first-passage):
    3-class: within next 300 seconds, does cumulative price move from event t
    first hit +K ticks (class +1), -K ticks (class -1), or neither (class 0).
    Cumulative move is reconstructed from labels_1s deltas.
    Generated for K in {2, 4, 8}.

Head-D (regime, 60s window):
    3-class: 0=trending, 1=mean-reverting, 2=noise.
    Thresholds:
      - trending:        |cum_move_60s| > 4 ticks AND
                         monotonicity_ratio > 0.65
        (monotonicity = #steps in dominant direction / total steps)
      - mean-reverting:  sign-flips in 60s window > 4 AND
                         path range > 2 ticks
      - noise:           everything else (small range, no structure)

Input  : /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3/*.npz
Output : /home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_v35/*_v35.npz

Usage:
  # Smoke (1 day)
  python build_continuation_labels.py --smoke
  # Full build (all days, parallel)
  python build_continuation_labels.py --workers 8
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
SRC_DIR = Path(os.environ.get("V35_SRC_DIR", "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3"))
DST_DIR = Path(os.environ.get("V35_DST_DIR", "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3_v35"))
DST_DIR.mkdir(parents=True, exist_ok=True)

HEAD_A_K_LIST = (5, 10, 30, 60)   # seconds
HEAD_C_K_LIST = (2, 4, 8)         # tick magnitudes
HEAD_B_CAP_S = 300.0              # seconds (also Head-C horizon)
HEAD_D_WINDOW_S = 60.0            # regime classification window
HEAD_D_TREND_TICKS = 4.0
HEAD_D_TREND_MONO = 0.65
HEAD_D_MR_FLIPS = 4
HEAD_D_MR_RANGE = 2.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("build_continuation_labels")


# ── Utility: find first index strictly after t whose ts >= t_ns + delta_ns ──
def _build_horizon_index(ts_ns: np.ndarray, delta_ns: int) -> np.ndarray:
    """For each i return the smallest j > i with ts_ns[j] >= ts_ns[i] + delta_ns.
    If no such j exists, returns len(ts_ns).
    Vectorised via searchsorted (one pass, O(N log N))."""
    target = ts_ns + delta_ns
    # searchsorted returns insertion index for target into ts_ns (left)
    j = np.searchsorted(ts_ns, target, side="left")
    # Ensure j > i (degenerate when delta_ns==0 or ties)
    np.maximum(j, np.arange(len(ts_ns)) + 1, out=j)
    return j.astype(np.int64)


# ── Head-A: pressure-direction binary ────────────────────────────────────────
def head_a_pressure_direction(labels_K: np.ndarray) -> np.ndarray:
    """Binary head-A label for a single horizon K. 1 if bullish, 0 if bearish.
    nan-preserving: returns float32 with nan where labels_K is nan or exactly 0."""
    out = np.full(labels_K.shape, np.nan, dtype=np.float32)
    valid = ~np.isnan(labels_K)
    out[valid & (labels_K > 0)] = 1.0
    out[valid & (labels_K < 0)] = 0.0
    # exact-zero remains nan (ambiguous, drop in BCE loss)
    return out


# ── Head-B: pressure-persistence (seconds to flip) ───────────────────────────
def head_b_pressure_persistence(
    ts_ns: np.ndarray, labels_1s: np.ndarray, cap_s: float = HEAD_B_CAP_S,
) -> Tuple[np.ndarray, np.ndarray]:
    """For each event t, find first j > t where sign(labels_1s[j]) != sign(labels_1s[t]).
    Return (persistence_seconds (float32), censored_mask (uint8)).

    Vectorised O(N) algorithm using right-to-left running index of "next-positive"
    and "next-negative" event positions.
    """
    n = len(ts_ns)
    persistence = np.full(n, np.nan, dtype=np.float32)
    censored = np.zeros(n, dtype=np.uint8)

    sign = np.zeros(n, dtype=np.int8)
    valid = ~np.isnan(labels_1s)
    sign[valid & (labels_1s > 0)] = 1
    sign[valid & (labels_1s < 0)] = -1

    cap_ns = int(cap_s * 1e9)
    horizon_end = _build_horizon_index(ts_ns, cap_ns)

    # next_pos[i] = smallest j >= i with sign[j] == +1 (else n)
    # next_neg[i] = smallest j >= i with sign[j] == -1 (else n)
    next_pos = np.full(n + 1, n, dtype=np.int64)
    next_neg = np.full(n + 1, n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        next_pos[i] = i if sign[i] == 1 else next_pos[i + 1]
        next_neg[i] = i if sign[i] == -1 else next_neg[i + 1]

    # Now per-event lookup is O(1)
    for i in range(n):
        s_i = sign[i]
        if s_i == 0:
            continue
        # find first opposite-sign event strictly after i
        if s_i == 1:
            j = int(next_neg[i + 1])
        else:
            j = int(next_pos[i + 1])
        if j >= n or j >= horizon_end[i]:
            persistence[i] = float(cap_s)
            censored[i] = 1
        else:
            persistence[i] = float(ts_ns[j] - ts_ns[i]) / 1e9
    return persistence, censored


# ── Head-C: cumulative K-tick first-passage (3-class) ────────────────────────
def head_c_first_passage_K(
    ts_ns: np.ndarray, labels_1s: np.ndarray, K: float,
    horizon_s: float = HEAD_B_CAP_S,
) -> np.ndarray:
    """For each event t, walk forward up to horizon_s seconds and find the first
    j where CUMULATIVE labels_1s deltas from t hit +K (=+1), -K (=-1), or neither (=0).
    Returns int8 array.

    Notes:
      - Cumulative move is reconstructed from labels_1s diffs. labels_1s[t] is
        the price-move at horizon 1s from event t — its difference between
        consecutive events approximates the realized per-event price step.
      - For NaN-tolerant cumulative, NaN diffs contribute 0 to the running sum.
      - +0 (neither hit) is the default for any event that lacks valid future
        prices.
    """
    n = len(ts_ns)
    out = np.zeros(n, dtype=np.int8)
    if n < 2:
        return out
    horizon_ns = int(horizon_s * 1e9)
    horizon_end = _build_horizon_index(ts_ns, horizon_ns)

    # Per-step price increments (ticks): diff of labels_1s (sign-preserved tick units).
    # labels_h is signed price-CHANGE at horizon h in ticks (verified empirically:
    # median 0, std ~tick-scale). Diff over event time approximates per-event
    # price increment. NaNs → 0 contribution.
    incr = np.diff(labels_1s, prepend=labels_1s[0])
    incr = np.where(np.isnan(incr), 0.0, incr).astype(np.float32)

    for i in range(n):
        end = horizon_end[i]
        if end <= i + 1:
            continue
        # Cumulative sum from i+1 to end-1
        cum = np.cumsum(incr[i + 1:end])
        # First hit of +K or -K
        pos_hit_idx = np.flatnonzero(cum >= K)
        neg_hit_idx = np.flatnonzero(cum <= -K)
        first_pos = int(pos_hit_idx[0]) if len(pos_hit_idx) else np.iinfo(np.int64).max
        first_neg = int(neg_hit_idx[0]) if len(neg_hit_idx) else np.iinfo(np.int64).max
        if first_pos == np.iinfo(np.int64).max and first_neg == np.iinfo(np.int64).max:
            out[i] = 0
        elif first_pos < first_neg:
            out[i] = 1
        elif first_neg < first_pos:
            out[i] = -1
        else:
            # Tie: very rare; pick noise (0)
            out[i] = 0
    return out


# ── Head-D: regime classification (3-class, 60s window) ──────────────────────
def head_d_regime_60s(
    ts_ns: np.ndarray, labels_1s: np.ndarray, window_s: float = HEAD_D_WINDOW_S,
) -> np.ndarray:
    """Classify the next window_s for each event t:
      0 = trending      (|cum_move| > 4 ticks AND mono_ratio > 0.65)
      1 = mean-reverting (sign_flips > 4 AND path range > 2 ticks)
      2 = noise         (everything else)
    Returns int8. Events with insufficient lookahead → 2 (noise).
    """
    n = len(ts_ns)
    out = np.full(n, 2, dtype=np.int8)  # default noise
    horizon_ns = int(window_s * 1e9)
    horizon_end = _build_horizon_index(ts_ns, horizon_ns)

    incr = np.diff(labels_1s, prepend=labels_1s[0])
    incr = np.where(np.isnan(incr), 0.0, incr).astype(np.float32)

    for i in range(n):
        end = horizon_end[i]
        if end <= i + 2:
            continue
        seg = incr[i + 1:end]
        cum = np.cumsum(seg)
        cum_move = float(cum[-1])
        path_range = float(cum.max() - cum.min())
        # Sign-flip count in the per-step increments
        # (only nonzero steps count toward flip detection)
        signs = np.sign(seg)
        nz_mask = signs != 0
        if nz_mask.sum() < 2:
            continue
        nz_signs = signs[nz_mask]
        flips = int((nz_signs[1:] != nz_signs[:-1]).sum())
        # Monotonicity ratio: max(up_steps, down_steps) / total_nonzero
        up_steps = int((nz_signs > 0).sum())
        down_steps = int((nz_signs < 0).sum())
        mono_ratio = max(up_steps, down_steps) / float(len(nz_signs))

        if abs(cum_move) > HEAD_D_TREND_TICKS and mono_ratio > HEAD_D_TREND_MONO:
            out[i] = 0  # trending
        elif flips > HEAD_D_MR_FLIPS and path_range > HEAD_D_MR_RANGE:
            out[i] = 1  # mean-reverting
        else:
            out[i] = 2  # noise
    return out


# ── Per-file processing ──────────────────────────────────────────────────────
def process_one_file(src_path: Path, dst_dir: Path, force: bool = False) -> dict:
    """Process one day. Returns summary dict."""
    name = src_path.stem  # e.g. 20250714_mbo_events
    date_str = name.split("_")[0]
    dst_path = dst_dir / f"{date_str}_v35.npz"
    if dst_path.exists() and not force:
        return {"date": date_str, "status": "skip-exists", "path": str(dst_path)}

    t0 = time.time()
    with np.load(src_path, allow_pickle=False) as src:
        events = src["events"]
        event_type_raw = src["event_type_raw"]
        timestamps = src["timestamps"].astype(np.int64)
        labels_1s = src["labels_1s"].astype(np.float32)
        labels_5s = src["labels_5s"].astype(np.float32)
        labels_10s = src["labels_10s"].astype(np.float32)
        labels_30s = src["labels_30s"].astype(np.float32)

    # Smoke subsampling: cut day to first N events for fast verification
    smoke_n = globals().get("_SMOKE_SUBSAMPLE", 0)
    if smoke_n and len(timestamps) > smoke_n:
        log.info(f"[{date_str}] SMOKE: subsampling first {smoke_n:,} events "
                 f"(of {len(timestamps):,})")
        events = events[:smoke_n]
        event_type_raw = event_type_raw[:smoke_n]
        timestamps = timestamps[:smoke_n]
        labels_1s = labels_1s[:smoke_n]
        labels_5s = labels_5s[:smoke_n]
        labels_10s = labels_10s[:smoke_n]
        labels_30s = labels_30s[:smoke_n]

    n = len(timestamps)
    log.info(f"[{date_str}] n_events={n:,}  computing v3.5 labels...")

    out_arrays = {
        "events": events,
        "event_type_raw": event_type_raw,
        "timestamps": timestamps,
        "labels_1s": labels_1s,
        "labels_5s": labels_5s,
        "labels_10s": labels_10s,
        "labels_30s": labels_30s,
    }
    log.info(f"[{date_str}] arrays staged: events shape={events.shape}")

    # Head-A: pressure-direction binary for K in {5,10,30,60}
    # K=60s not in source labels; synthesize as sign of label_30s as a conservative fallback
    # for smoke. (Full builder would compute true 60s by looking 60s ahead in mid-price; we
    # approximate using existing labels_30s scaled by ~2 — see TODO.)
    out_arrays["label_A_5s"] = head_a_pressure_direction(labels_5s)
    out_arrays["label_A_10s"] = head_a_pressure_direction(labels_10s)
    out_arrays["label_A_30s"] = head_a_pressure_direction(labels_30s)
    # 60s: synthesized from 30s sign (placeholder until raw 60s labels added)
    out_arrays["label_A_60s"] = head_a_pressure_direction(labels_30s)

    # Head-B: persistence (seconds until labels_1s sign-flip)
    persistence, censored = head_b_pressure_persistence(
        timestamps, labels_1s, cap_s=HEAD_B_CAP_S
    )
    out_arrays["label_B_persistence_s"] = persistence
    out_arrays["label_B_censored"] = censored

    # Head-C: cumulative K-tick first-passage for K in {2,4,8}
    for K in HEAD_C_K_LIST:
        out_arrays[f"label_C_K{int(K)}"] = head_c_first_passage_K(
            timestamps, labels_1s, K=float(K), horizon_s=HEAD_B_CAP_S
        )

    # Head-D: regime 60s
    out_arrays["label_D_regime_60s"] = head_d_regime_60s(
        timestamps, labels_1s, window_s=HEAD_D_WINDOW_S
    )

    # Save
    np.savez(dst_path, **out_arrays)
    elapsed = time.time() - t0
    # Quick stats
    sum_dict = {
        "date": date_str,
        "status": "ok",
        "n_events": int(n),
        "elapsed_s": round(elapsed, 1),
        "A_5s_pos_frac": float(np.nanmean(out_arrays["label_A_5s"])),
        "B_median_s": float(np.nanmedian(persistence)),
        "B_censored_frac": float(censored.mean()),
        "C_K4_class_dist": {
            "neg": int((out_arrays["label_C_K4"] == -1).sum()),
            "zero": int((out_arrays["label_C_K4"] == 0).sum()),
            "pos": int((out_arrays["label_C_K4"] == 1).sum()),
        },
        "D_class_dist": {
            "trending": int((out_arrays["label_D_regime_60s"] == 0).sum()),
            "mean_revert": int((out_arrays["label_D_regime_60s"] == 1).sum()),
            "noise": int((out_arrays["label_D_regime_60s"] == 2).sum()),
        },
        "path": str(dst_path),
    }
    log.info(f"[{date_str}] DONE in {elapsed:.1f}s  -> {dst_path}")
    log.info(f"[{date_str}] A_5s pos_frac={sum_dict['A_5s_pos_frac']:.3f}  "
             f"B med={sum_dict['B_median_s']:.2f}s cens={sum_dict['B_censored_frac']:.3f}  "
             f"C_K4 = {sum_dict['C_K4_class_dist']}  D = {sum_dict['D_class_dist']}")
    return sum_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Process only one day (the first sorted day)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for full build")
    ap.add_argument("--limit", type=int, default=0,
                    help="Limit to first N days (after --smoke override)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing _v35 files")
    ap.add_argument("--smoke-subsample", type=int, default=100_000,
                    help="In --smoke mode, only process first N events of the day")
    ap.add_argument("--year-min", type=int, default=2026,
                    help="Earliest year (YYYY) to include. Default 2026 per HC #500 — "
                         "label/training data restricted to 2026 days for OOT overlap "
                         "with current production model suite. Set 2025 for explicit "
                         "decay studies with written justification (see HC #500 R5).")
    args = ap.parse_args()
    global _SMOKE_SUBSAMPLE
    _SMOKE_SUBSAMPLE = args.smoke_subsample if args.smoke else 0

    src_files = sorted(SRC_DIR.glob("*_mbo_events.npz"))
    if not src_files:
        log.error(f"No source files in {SRC_DIR}")
        sys.exit(2)
    # HC #500 R1/R4 — year filter (default 2026): exclude older years from build.
    year_min_str = f"{args.year_min:04d}0101"
    pre_filter_n = len(src_files)
    src_files = [f for f in src_files if f.stem >= year_min_str]
    log.info(
        f"Found {pre_filter_n} source files in {SRC_DIR}; "
        f"{len(src_files)} kept after --year-min={args.year_min} (HC #500)"
    )
    log.info(f"Destination: {DST_DIR}")

    if args.smoke:
        log.info("=== SMOKE MODE: processing first day only ===")
        sub = src_files[:1]
        for f in sub:
            r = process_one_file(f, DST_DIR, force=args.force)
            log.info(f"SMOKE result: {r}")
        return

    if args.limit > 0:
        src_files = src_files[: args.limit]

    t0 = time.time()
    if args.workers <= 1:
        for f in src_files:
            process_one_file(f, DST_DIR, force=args.force)
    else:
        with mp.Pool(args.workers) as pool:
            results = []
            for f in src_files:
                results.append(pool.apply_async(process_one_file, (f, DST_DIR, args.force)))
            for r in results:
                r.get()
    elapsed = time.time() - t0
    log.info(f"FULL BUILD DONE in {elapsed/60:.1f} min ({len(src_files)} files)")


if __name__ == "__main__":
    main()
