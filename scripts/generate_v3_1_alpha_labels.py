"""
Generate v3.1 ALPHA-FIRST labels per HC #293(B).

Per signal step (event index), compute path-aware targets within a 30s forward window
from the per-event mid-price trajectory:

  - log_ret_30s_ticks         (mid[j_30s] - mid[i]) in ticks
  - p_up_5s/10s/30s           sign of forward delta (binary, 0/1)
  - pred_mfe_30s_ticks        max(mid[k] - mid[i]) for k in (i, j_30s]
  - pred_mae_30s_ticks        min(mid[k] - mid[i]) for k in (i, j_30s]
  - pred_time_to_mfe_secs     (ts[k_at_mfe] - ts[i]) / 1e9
  - p_reversal_15s/30s        binary: did sign(mid[k]-mid[i]) flip back vs sign(label_5s)?
  - pred_realized_vol_30s_ticks  std of 1-sec-bucketed log-returns × sqrt(30) in ticks

Inputs:
  /data/processed/mbo_events_smart_v3/<date>_mbo_events.npz   (events + labels + ts)
  /data/processed/mbo_book_features/<date>_book_features.npz   (bid_price_1, ask_price_1)

Output:
  /data/processed/mbo_events_smart_v3_alpha_labels/<date>_alpha_labels.npz
"""

import os
import sys
import time
import argparse
from pathlib import Path
from collections import deque
import numpy as np

try:
    import numba
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(f=None, **kwargs):
        if f is None:
            return lambda x: x
        return f

# Constants
TICK_SIZE = 0.25  # ES tick (not used directly; bid/ask already in tick units)
SEC_NS = 1_000_000_000
WIN_NS_5S = 5 * SEC_NS
WIN_NS_10S = 10 * SEC_NS
WIN_NS_15S = 15 * SEC_NS
WIN_NS_30S = 30 * SEC_NS
# HC #477 R3 fix (2026-05-21): add 60s + 5min horizons. Six model heads
# (log_ret_60s, log_ret_5min, mfe_60s, mae_60s, p_up_60s, p_reversal_60s)
# previously trained as no-ops because their target columns were never produced.
WIN_NS_60S = 60 * SEC_NS
WIN_NS_5MIN = 300 * SEC_NS

DEFAULT_EVENTS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
DEFAULT_BOOK_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_book_features")
# HC #477 R3 fix: bump output dir to _v2 so old 124 files preserved for rollback.
DEFAULT_OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_alpha_labels_v3")


def _two_pointer_indices(ts: np.ndarray, horizon_ns: int) -> np.ndarray:
    """For each i, return smallest j such that ts[j] >= ts[i] + horizon_ns,
    or N if no such j exists. Fully vectorized via searchsorted."""
    targets = ts + horizon_ns
    # searchsorted returns index where target would be inserted to keep sorted
    # using side='left' -> first index where ts[j] >= target
    out = np.searchsorted(ts, targets, side='left').astype(np.int64)
    return out


@njit(cache=True, fastmath=False)
def _windowed_minmax_argmax_njit(values, j_ends):
    """O(N) sliding window min, max, argmax using monotonic deques.
    Window for i is (i, j_ends[i]). j_ends must be monotonically non-decreasing.
    Returns (mins, maxs, argmaxs)."""
    N = values.shape[0]
    mins = np.full(N, np.nan, dtype=np.float64)
    maxs = np.full(N, np.nan, dtype=np.float64)
    argmaxs = np.full(N, -1, dtype=np.int64)

    # Manual ring buffers for deques. Use index arrays.
    max_dq = np.empty(N, dtype=np.int64)
    min_dq = np.empty(N, dtype=np.int64)
    max_head = 0; max_tail = 0  # empty when head == tail
    min_head = 0; min_tail = 0

    pushed_up_to = 0

    for i in range(N):
        j_end = j_ends[i]
        if j_end <= i + 1:
            continue
        start = pushed_up_to if pushed_up_to > i + 1 else i + 1
        for k in range(start, j_end):
            v = values[k]
            # max deque (monotonic decreasing from front)
            while max_tail > max_head and values[max_dq[max_tail - 1]] <= v:
                max_tail -= 1
            max_dq[max_tail] = k
            max_tail += 1
            # min deque
            while min_tail > min_head and values[min_dq[min_tail - 1]] >= v:
                min_tail -= 1
            min_dq[min_tail] = k
            min_tail += 1
        if pushed_up_to < j_end:
            pushed_up_to = j_end

        # Pop entries that are out of window (<=i or >=j_end)
        while max_head < max_tail and max_dq[max_head] <= i:
            max_head += 1
        while min_head < min_tail and min_dq[min_head] <= i:
            min_head += 1
        while max_head < max_tail and max_dq[max_head] >= j_end:
            max_head += 1
        while min_head < min_tail and min_dq[min_head] >= j_end:
            min_head += 1

        if max_head < max_tail:
            maxs[i] = values[max_dq[max_head]]
            argmaxs[i] = max_dq[max_head]
        if min_head < min_tail:
            mins[i] = values[min_dq[min_head]]

    return mins, maxs, argmaxs


def _compute_min_max_windowed(values: np.ndarray, j_ends: np.ndarray,
                              return_argmax: bool = False):
    """For each i, compute min(values[i+1..j_ends[i]-1]) and max(values[i+1..j_ends[i]-1]).
    j_ends[i] is exclusive."""
    # Ensure contiguous arrays of the right dtypes for numba
    vals = np.ascontiguousarray(values, dtype=np.float64)
    jends = np.ascontiguousarray(j_ends, dtype=np.int64)
    mins, maxs, argmaxs = _windowed_minmax_argmax_njit(vals, jends)
    if return_argmax:
        return mins, maxs, argmaxs
    return mins, maxs


def compute_realized_vol_30s(mids: np.ndarray, ts_ns: np.ndarray, j_30s: np.ndarray,
                              j_starts: np.ndarray = None) -> np.ndarray:
    """
    For each i, compute realized vol over next 30s = std(1s-bucket log-returns) * sqrt(30).
    Since we're in ticks (small integers), use raw tick differences instead of log-returns.
    Returns vol in ticks.

    For each i: find j_1s, j_2s, ..., j_30s — get mid at each second boundary, compute std of diffs.
    Approximation: use j_30s window endpoints and compute std of mid changes
    between 30 equally-spaced timestamps.
    Vectorized approach: precompute mid at each 1s boundary across the day, then for each
    signal i, slice the next 30 boundaries.
    """
    N = len(mids)
    out = np.full(N, np.nan, dtype=np.float32)

    if N == 0 or len(ts_ns) == 0:
        return out
    t0 = int(ts_ns[0])
    t_end = int(ts_ns[-1])

    # Build 1-second boundary array spanning the full day
    boundary_count = (t_end - t0) // SEC_NS + 1
    if boundary_count < 2:
        return out
    boundaries = t0 + np.arange(boundary_count, dtype=np.int64) * SEC_NS

    # For each boundary, find the first event index >= boundary (searchsorted)
    bnd_event_idx = np.searchsorted(ts_ns, boundaries, side='left')
    # Clip to N-1 (last event)
    bnd_event_idx = np.clip(bnd_event_idx, 0, N - 1)
    # Mid at each 1s boundary
    mids_1s = mids[bnd_event_idx]
    # 1s tick returns at boundaries
    diff_1s = np.diff(mids_1s)  # length boundary_count - 1
    # Treat NaN diffs as 0 (boundary was pre-RTH / no valid mid) — they don't contribute to vol.
    # Also keep a validity mask: a window is valid only if at least 50% of its diffs are finite.
    finite_mask = np.isfinite(diff_1s).astype(np.float32)
    diff_1s = np.nan_to_num(diff_1s, nan=0.0).astype(np.float64)

    # For each signal i, find its boundary index b_i (the boundary that aligns to ts[i])
    # b_i = floor((ts[i] - t0) / SEC_NS)
    b_idx = ((ts_ns - t0) // SEC_NS).astype(np.int64)
    b_idx = np.clip(b_idx, 0, boundary_count - 1)

    # For each i, vol = std(diff_1s[b_idx[i]:b_idx[i]+30]) * sqrt(30)
    # Vectorize via cumulative sums:
    # We need a windowed std of 30 elements starting at b_idx[i].
    # Precompute cumsum and cumsum-of-squares of diff_1s.
    W = 30
    n_diffs = len(diff_1s)
    if n_diffs < W + 1:
        return out
    cs = np.concatenate(([0.0], np.cumsum(diff_1s)))
    cs_sq = np.concatenate(([0.0], np.cumsum(diff_1s ** 2)))
    cs_fin = np.concatenate(([0.0], np.cumsum(finite_mask.astype(np.float64))))

    starts = b_idx
    ends = starts + W
    valid = (starts >= 0) & (ends <= n_diffs)
    s = starts[valid]
    e = ends[valid]
    sums = cs[e] - cs[s]
    sums_sq = cs_sq[e] - cs_sq[s]
    n_finite = cs_fin[e] - cs_fin[s]
    # Require at least half the window to be finite
    enough = n_finite >= (W // 2)
    means = sums / W
    variances = np.maximum((sums_sq / W) - means ** 2, 0.0)
    stds = np.sqrt(variances)
    # Final vol assignment, NaN where insufficient coverage
    vol_vals = (stds * np.sqrt(W)).astype(np.float32)
    vol_vals[~enough] = np.nan
    out[valid] = vol_vals
    return out


def process_one_date(date_str: str, events_dir: Path, book_dir: Path, out_dir: Path,
                     overwrite: bool = False) -> dict:
    out_path = out_dir / f"{date_str}_alpha_labels.npz"
    if out_path.exists() and not overwrite:
        return {"date": date_str, "status": "skipped_exists", "out": str(out_path)}

    events_path = events_dir / f"{date_str}_mbo_events.npz"
    book_path = book_dir / f"{date_str}_book_features.npz"
    if not events_path.exists() or not book_path.exists():
        return {"date": date_str, "status": "missing_inputs",
                "events_ok": events_path.exists(), "book_ok": book_path.exists()}

    t0 = time.time()
    ev = np.load(events_path, allow_pickle=True)
    ts_ns = ev["timestamps"].astype(np.int64)
    lab1 = ev["labels_1s"].astype(np.float32)
    lab5 = ev["labels_5s"].astype(np.float32)
    lab10 = ev["labels_10s"].astype(np.float32)
    lab30 = ev["labels_30s"].astype(np.float32) if "labels_30s" in ev else None
    N = len(ts_ns)

    bk = np.load(book_path, allow_pickle=True)
    feats = bk["features"]
    # cols: 0=bid_price_1, 5=ask_price_1
    bid = feats[:, 0].astype(np.float64)
    ask = feats[:, 5].astype(np.float64)
    mids = (bid + ask) / 2.0  # in ticks

    # Sanity: book features and events must have same N
    if len(mids) != N:
        return {"date": date_str, "status": "length_mismatch", "N_events": N, "N_book": len(mids)}

    # HC #485 fix (2026-05-22): the OLD gate `(bid<=0)|(ask<=0)|(ask<bid)`
    # was WRONG. `bid_price_1` / `ask_price_1` in mbo_book_features are
    # RELATIVE OFFSETS around an unknown session anchor (see
    # output/stream_backtest_v2/label_coverage_diagnosis.md), so they go
    # negative for the majority of each session. The old gate falsely
    # invalidated 47-100% of events per day, which cascaded into NaN labels
    # for the 60s/5min/MFE-60s heads added in HC #477 R3.
    # The signed mids are still fine for log_ret/MFE/MAE because those are
    # DIFFERENCES (mid[j]-mid[i]) — the anchor cancels.
    # Correct gate: keep when spread is a sane positive number of ticks.
    spread = ask - bid
    invalid_mask = (spread <= 0) | (spread >= 20.0) | ~np.isfinite(spread)
    # For invalid events, mid value is meaningless — replace with NaN to propagate
    mids_clean = mids.copy()
    mids_clean[invalid_mask] = np.nan

    # ===== Two-pointer indices at various horizons =====
    j_5s = _two_pointer_indices(ts_ns, WIN_NS_5S)
    j_10s = _two_pointer_indices(ts_ns, WIN_NS_10S)
    j_15s = _two_pointer_indices(ts_ns, WIN_NS_15S)
    j_30s = _two_pointer_indices(ts_ns, WIN_NS_30S)
    # HC #477 R3 fix: 60s + 5min horizons (previously missing).
    j_60s = _two_pointer_indices(ts_ns, WIN_NS_60S)
    j_5min = _two_pointer_indices(ts_ns, WIN_NS_5MIN)

    # ===== log_ret_30s (in ticks) =====
    if lab30 is not None:
        log_ret_30s = lab30  # already in ticks
    else:
        # Fallback: mid[j_30s] - mid[i]
        log_ret_30s = np.full(N, np.nan, dtype=np.float32)
        valid = j_30s < N
        idx = np.where(valid)[0]
        delta = mids_clean[j_30s[idx]] - mids_clean[idx]
        log_ret_30s[idx] = delta.astype(np.float32)

    # ===== log_ret_60s / log_ret_5min (HC #477 R3 fix, in ticks) =====
    # No upstream `labels_60s` / `labels_5min` exist — compute from mids directly.
    # End-of-day samples where horizon exceeds session get NaN (correctly propagated
    # through trainer mask path — no zero-fill leak).
    log_ret_60s = np.full(N, np.nan, dtype=np.float32)
    valid60 = j_60s < N
    idx60 = np.where(valid60)[0]
    log_ret_60s[idx60] = (mids_clean[j_60s[idx60]] - mids_clean[idx60]).astype(np.float32)

    log_ret_5min = np.full(N, np.nan, dtype=np.float32)
    valid5m = j_5min < N
    idx5m = np.where(valid5m)[0]
    log_ret_5min[idx5m] = (mids_clean[j_5min[idx5m]] - mids_clean[idx5m]).astype(np.float32)

    # ===== p_up_*s (binary) =====
    p_up_5s = (lab5 > 0).astype(np.float32)
    p_up_10s = (lab10 > 0).astype(np.float32)
    p_up_30s = (log_ret_30s > 0).astype(np.float32)
    # HC #477 R3 fix: p_up_60s
    p_up_60s = (log_ret_60s > 0).astype(np.float32)
    # Mask: NaN labels -> output NaN
    p_up_5s[np.isnan(lab5)] = np.nan
    p_up_10s[np.isnan(lab10)] = np.nan
    p_up_30s[np.isnan(log_ret_30s)] = np.nan
    p_up_60s[np.isnan(log_ret_60s)] = np.nan

    # ===== MFE/MAE 30s + time-to-MFE =====
    # Need max/min of mid in window (i, j_30s).
    # Use windowed monotonic deque on mids_clean (NaN-safe — replace nans with -inf for max, +inf for min)
    mids_for_max = np.where(np.isnan(mids_clean), -np.inf, mids_clean)
    mids_for_min = np.where(np.isnan(mids_clean), np.inf, mids_clean)

    # Compute max + argmax over 30s window (use -inf for NaN so max ignores them)
    _, max_in_win, argmax_in_win = _compute_min_max_windowed(mids_for_max, j_30s, return_argmax=True)
    # Compute min over 30s window (use +inf for NaN so min ignores them)
    min_proper, _ = _compute_min_max_windowed(mids_for_min, j_30s, return_argmax=False)

    # MFE/MAE in ticks (vs signal mid)
    mfe_30s = max_in_win - mids_clean  # may be NaN
    mae_30s = min_proper - mids_clean  # negative
    # Where signal mid is NaN, result is NaN
    mfe_30s = np.where(np.isnan(mids_clean), np.nan, mfe_30s)
    mae_30s = np.where(np.isnan(mids_clean), np.nan, mae_30s)
    # Where max is -inf (no valid mid in window), set NaN
    mfe_30s = np.where(np.isinf(mfe_30s), np.nan, mfe_30s).astype(np.float32)
    mae_30s = np.where(np.isinf(mae_30s), np.nan, mae_30s).astype(np.float32)

    # ===== MFE/MAE 60s (HC #477 R3 fix) =====
    _, max_in_win_60s, _ = _compute_min_max_windowed(mids_for_max, j_60s, return_argmax=True)
    min_proper_60s, _ = _compute_min_max_windowed(mids_for_min, j_60s, return_argmax=False)
    mfe_60s = max_in_win_60s - mids_clean
    mae_60s = min_proper_60s - mids_clean
    mfe_60s = np.where(np.isnan(mids_clean), np.nan, mfe_60s)
    mae_60s = np.where(np.isnan(mids_clean), np.nan, mae_60s)
    mfe_60s = np.where(np.isinf(mfe_60s), np.nan, mfe_60s).astype(np.float32)
    mae_60s = np.where(np.isinf(mae_60s), np.nan, mae_60s).astype(np.float32)

    # time-to-MFE in seconds — only valid where MFE itself is valid (signal mid valid + argmax in window)
    valid_argmax = (argmax_in_win >= 0) & np.isfinite(mfe_30s)
    time_to_mfe = np.full(N, np.nan, dtype=np.float32)
    if valid_argmax.any():
        idxs = np.where(valid_argmax)[0]
        dt_ns = ts_ns[argmax_in_win[idxs]] - ts_ns[idxs]
        time_to_mfe[idxs] = (dt_ns / SEC_NS).astype(np.float32)

    # ===== p_reversal_15s / p_reversal_30s =====
    # Definition: did the mid cross BACK through signal_mid within the window?
    # Use min_in_15s and max_in_15s. If min < signal_mid < max → mid crossed in both directions
    # within the window → that's "reversal".
    # Compute min/max for 15s window separately
    min_15s, _ = _compute_min_max_windowed(mids_for_min, j_15s)
    _, max_15s_proper = _compute_min_max_windowed(mids_for_max, j_15s)
    # signal_mid: if signal direction was up (lab5>0), reversal = min went below signal_mid;
    # if signal was down (lab5<0), reversal = max went above signal_mid.
    # Simpler symmetric def: reversal happened if min < signal_mid AND max > signal_mid.
    reversal_15s = ((min_15s < mids_clean) & (max_15s_proper > mids_clean)).astype(np.float32)
    reversal_15s[np.isnan(mids_clean) | np.isinf(min_15s) | np.isinf(max_15s_proper)] = np.nan

    reversal_30s = ((min_proper < mids_clean) & (max_in_win > mids_clean)).astype(np.float32)
    reversal_30s[np.isnan(mids_clean) | np.isinf(min_proper) | np.isinf(max_in_win)] = np.nan

    # ===== p_reversal_60s (HC #477 R3 fix) =====
    reversal_60s = ((min_proper_60s < mids_clean) & (max_in_win_60s > mids_clean)).astype(np.float32)
    reversal_60s[np.isnan(mids_clean) | np.isinf(min_proper_60s) | np.isinf(max_in_win_60s)] = np.nan

    # ===== realized_vol_30s =====
    realized_vol_30s = compute_realized_vol_30s(mids_clean, ts_ns, j_30s)

    # ===== Save =====
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        log_ret_30s=log_ret_30s.astype(np.float32),
        # HC #477 R3 fix: 60s + 5min log-returns (previously missing).
        log_ret_60s=log_ret_60s.astype(np.float32),
        log_ret_5min=log_ret_5min.astype(np.float32),
        p_up_5s=p_up_5s.astype(np.float32),
        p_up_10s=p_up_10s.astype(np.float32),
        p_up_30s=p_up_30s.astype(np.float32),
        # HC #477 R3 fix: p_up_60s (previously missing).
        p_up_60s=p_up_60s.astype(np.float32),
        mfe_30s_ticks=mfe_30s,
        mae_30s_ticks=mae_30s,
        # HC #477 R3 fix: 60s MFE/MAE (previously missing).
        mfe_60s_ticks=mfe_60s,
        mae_60s_ticks=mae_60s,
        time_to_mfe_secs=time_to_mfe,
        p_reversal_15s=reversal_15s,
        p_reversal_30s=reversal_30s,
        # HC #477 R3 fix: p_reversal_60s (previously missing).
        p_reversal_60s=reversal_60s,
        realized_vol_30s_ticks=realized_vol_30s,
    )
    elapsed = time.time() - t0
    valid_mfe = np.isfinite(mfe_30s).sum()
    return {
        "date": date_str, "status": "ok", "N": N, "elapsed_s": elapsed,
        "valid_mfe": int(valid_mfe), "out": str(out_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=None, help="Optional date subset; default = all dates >= 2025-11-01 with full coverage")
    ap.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    ap.add_argument("--book-dir", type=Path, default=DEFAULT_BOOK_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--min-date", type=str, default="20251101")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.dates:
        dates = sorted(args.dates)
    else:
        import re
        smart = {re.match(r"(\d{8})_mbo_events.npz", f).group(1)
                 for f in os.listdir(args.events_dir)
                 if re.match(r"\d{8}_mbo_events.npz", f)}
        book = {re.match(r"(\d{8})_book_features.npz", f).group(1)
                for f in os.listdir(args.book_dir)
                if re.match(r"\d{8}_book_features.npz", f)}
        dates = sorted([d for d in (smart & book) if d >= args.min_date])

    print(f"[label-gen] {len(dates)} dates to process", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_t = time.time()
    results = []
    if args.workers > 1:
        from multiprocessing import Pool
        from functools import partial
        fn = partial(process_one_date, events_dir=args.events_dir, book_dir=args.book_dir,
                     out_dir=args.out_dir, overwrite=args.overwrite)
        with Pool(args.workers) as pool:
            for r in pool.imap_unordered(fn, dates):
                results.append(r)
                print(f"  {r['date']}: {r}", flush=True)
    else:
        for d in dates:
            r = process_one_date(d, args.events_dir, args.book_dir, args.out_dir, overwrite=args.overwrite)
            results.append(r)
            print(f"  {d}: {r}", flush=True)

    elapsed = time.time() - total_t
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skipped_exists")
    fail = len(results) - ok - skip
    print(f"[label-gen] DONE: ok={ok} skipped={skip} fail={fail} elapsed={elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
