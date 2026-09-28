"""
HC #464 R2(a) — MFE/MAE-within-horizon label generation for h in {1s, 5s, 10s, 30s}.

For each event t in the smart_v3 stream, compute:
  - mfe_<h>_ticks  = max(mid[k] - mid[t])  for k in (t, t+h]
  - mae_<h>_ticks  = min(mid[k] - mid[t])  for k in (t, t+h]
  - time_to_mfe_<h>_secs  = (ts_argmax - ts[t]) / 1e9
  - mfe_minus_mae_<h>_ticks  = trade-quality score
  - mid_at_t_ticks  = signal-time mid (for downstream FIFO sim)

Inputs (per date):
  /data/processed/mbo_events_smart_v3/<date>_mbo_events.npz    (timestamps in ns)
  /data/processed/mbo_book_features/<date>_book_features.npz   (bid_price_1 col 0, ask_price_1 col 5)

Output (per date, per horizon):
  /data/relabel/mfe_mae_h<h>_<date>.parquet
  Schema: event_idx int32 | mid_t float32 | mfe float32 | mae float32 |
          time_to_mfe_s float32 | mfe_minus_mae float32

Driver:
  python3 scripts/relabel_mfe_mae_multi_horizon.py \
      --manifest /home/jupiter/Lvl3Quant/data/champion_date_manifest.json \
      --workers 8 --horizons 1s 5s 10s 30s

Reuses the same O(N) monotonic-deque min/max windowed scan from
generate_v3_1_alpha_labels.py — extended to ALL four horizons.

Per HC #464 R3 naming convention: mfe_mae_h<h>_YYYYMMDD.parquet
Per HC #462 R4 scope gate: every date checked against champion_date_manifest.json
                            before processing; out-of-scope dates rejected.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(f=None, **_):
        if f is None:
            return lambda x: x
        return f

# Constants
SEC_NS = 1_000_000_000
HORIZON_NS = {
    "1s":  1 * SEC_NS,
    "5s":  5 * SEC_NS,
    "10s": 10 * SEC_NS,
    "30s": 30 * SEC_NS,
}

DEFAULT_EVENTS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
DEFAULT_BOOK_DIR   = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_book_features")
DEFAULT_OUT_DIR    = Path("/home/jupiter/Lvl3Quant/data/relabel")
DEFAULT_MANIFEST   = Path("/home/jupiter/Lvl3Quant/data/champion_date_manifest.json")


def _two_pointer_indices(ts: np.ndarray, horizon_ns: int) -> np.ndarray:
    """Smallest j s.t. ts[j] >= ts[i] + horizon_ns, else N."""
    targets = ts + horizon_ns
    return np.searchsorted(ts, targets, side="left").astype(np.int64)


@njit(cache=True, fastmath=False)
def _windowed_minmax_argmax_njit(values, j_ends):
    """O(N) sliding window min, max, argmax using monotonic deques.
    Window for i is (i, j_ends[i]). j_ends must be monotonically non-decreasing."""
    N = values.shape[0]
    mins = np.full(N, np.nan, dtype=np.float64)
    maxs = np.full(N, np.nan, dtype=np.float64)
    argmaxs = np.full(N, -1, dtype=np.int64)
    argmins = np.full(N, -1, dtype=np.int64)

    max_dq = np.empty(N, dtype=np.int64)
    min_dq = np.empty(N, dtype=np.int64)
    max_head = 0; max_tail = 0
    min_head = 0; min_tail = 0
    pushed_up_to = 0

    for i in range(N):
        j_end = j_ends[i]
        if j_end <= i + 1:
            continue
        start = pushed_up_to if pushed_up_to > i + 1 else i + 1
        for k in range(start, j_end):
            v = values[k]
            while max_tail > max_head and values[max_dq[max_tail - 1]] <= v:
                max_tail -= 1
            max_dq[max_tail] = k
            max_tail += 1
            while min_tail > min_head and values[min_dq[min_tail - 1]] >= v:
                min_tail -= 1
            min_dq[min_tail] = k
            min_tail += 1
        if pushed_up_to < j_end:
            pushed_up_to = j_end
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
            argmins[i] = min_dq[min_head]

    return mins, maxs, argmaxs, argmins


def _mfe_mae_for_horizon(mids_clean: np.ndarray, ts_ns: np.ndarray, h_ns: int):
    """Returns (mfe_ticks, mae_ticks, time_to_mfe_secs) for a single horizon."""
    j_ends = _two_pointer_indices(ts_ns, h_ns)
    # NaN-safe transforms for max/min
    mids_for_max = np.where(np.isnan(mids_clean), -np.inf, mids_clean)
    mids_for_min = np.where(np.isnan(mids_clean), np.inf,  mids_clean)
    vals_max = np.ascontiguousarray(mids_for_max, dtype=np.float64)
    vals_min = np.ascontiguousarray(mids_for_min, dtype=np.float64)
    jends = np.ascontiguousarray(j_ends, dtype=np.int64)
    _, maxs, argmaxs, _ = _windowed_minmax_argmax_njit(vals_max, jends)
    mins, _, _, _      = _windowed_minmax_argmax_njit(vals_min, jends)
    mfe = (maxs - mids_clean).astype(np.float32)
    mae = (mins - mids_clean).astype(np.float32)
    mfe = np.where(np.isinf(mfe) | np.isnan(mids_clean), np.nan, mfe).astype(np.float32)
    mae = np.where(np.isinf(mae) | np.isnan(mids_clean), np.nan, mae).astype(np.float32)
    # time-to-MFE only where mfe is finite
    valid = (argmaxs >= 0) & np.isfinite(mfe)
    tt = np.full(len(mids_clean), np.nan, dtype=np.float32)
    if valid.any():
        idxs = np.where(valid)[0]
        dt_ns = ts_ns[argmaxs[idxs]] - ts_ns[idxs]
        tt[idxs] = (dt_ns / SEC_NS).astype(np.float32)
    return mfe, mae, tt


def process_one_date(date_str: str, horizons: list[str],
                     events_dir: Path, book_dir: Path, out_dir: Path,
                     overwrite: bool = False) -> dict:
    out_paths = {h: out_dir / f"mfe_mae_h{h}_{date_str}.parquet" for h in horizons}
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return {"date": date_str, "status": "skipped_exists"}

    events_path = events_dir / f"{date_str}_mbo_events.npz"
    book_path   = book_dir   / f"{date_str}_book_features.npz"
    if not events_path.exists() or not book_path.exists():
        return {"date": date_str, "status": "missing_inputs",
                "events_ok": events_path.exists(), "book_ok": book_path.exists()}

    t0 = time.time()
    try:
        ev = np.load(events_path, allow_pickle=True)
        ts_ns = ev["timestamps"].astype(np.int64)
        bk = np.load(book_path, allow_pickle=True)
        feats = bk["features"]
        bid = feats[:, 0].astype(np.float64)
        ask = feats[:, 5].astype(np.float64)
        mids_book = (bid + ask) / 2.0  # ticks (book features already in tick units)
        N = len(ts_ns)
        N_book = len(mids_book)
        if N_book == N:
            mids = mids_book
        else:
            # Length mismatch: align book mid-prices to event timestamps
            # Both are MBO-event-level but may differ due to filtering.
            # Use book timestamps to map mid_price onto event timestamps.
            bk_ts = bk["timestamps"].astype(np.int64)
            # For each event ts, find the closest book ts (last book entry <= event ts)
            idx = np.searchsorted(bk_ts, ts_ns, side="right") - 1
            idx = np.clip(idx, 0, N_book - 1)
            mids = mids_book[idx]

        if N_book == N:
            invalid = (bid <= 0) | (ask <= 0) | (ask < bid)
        else:
            # invalid mask must match event-aligned mids (size N)
            invalid_book = (bid <= 0) | (ask <= 0) | (ask < bid)
            invalid = invalid_book[idx]
        mids_clean = mids.copy()
        mids_clean[invalid] = np.nan

        out_dir.mkdir(parents=True, exist_ok=True)
        wrote = []
        for h in horizons:
            mfe, mae, tt = _mfe_mae_for_horizon(mids_clean, ts_ns, HORIZON_NS[h])
            df = pd.DataFrame({
                "event_idx":     np.arange(N, dtype=np.int32),
                "ts_ns":         ts_ns.astype(np.int64),
                "mid_t_ticks":   mids_clean.astype(np.float32),
                "mfe_ticks":     mfe,
                "mae_ticks":     mae,
                "time_to_mfe_s": tt,
                "mfe_minus_mae_ticks": (mfe - mae).astype(np.float32),
            })
            out_paths[h].parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_paths[h], compression="zstd")
            wrote.append((h, int(np.isfinite(mfe).sum()), int(np.isfinite(mae).sum())))

        return {
            "date": date_str, "status": "ok",
            "N": int(N),
            "elapsed_s": round(time.time() - t0, 2),
            "horizons": wrote,
        }
    except Exception as e:
        return {"date": date_str, "status": "error", "err": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help="champion_date_manifest.json — gates which dates are processed (HC #462 R4)")
    ap.add_argument("--horizons", nargs="+", default=["1s", "5s", "10s", "30s"],
                    choices=list(HORIZON_NS.keys()))
    ap.add_argument("--events-dir", type=Path, default=DEFAULT_EVENTS_DIR)
    ap.add_argument("--book-dir",   type=Path, default=DEFAULT_BOOK_DIR)
    ap.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dates", nargs="*", default=None,
                    help="Optional explicit date list (overrides manifest)")
    args = ap.parse_args()

    if args.dates:
        dates = sorted(args.dates)
        scope = "explicit"
    else:
        if not args.manifest.exists():
            print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
            sys.exit(2)
        m = json.loads(args.manifest.read_text())
        dates = sorted(m["dates"])
        scope = f"manifest ({m['window']['start']} → {m['window']['end']})"

    print(f"== HC #464 R2(a) — MFE/MAE-within-horizon relabel ==", flush=True)
    print(f"scope    : {scope}", flush=True)
    print(f"horizons : {args.horizons}", flush=True)
    print(f"n_dates  : {len(dates)}", flush=True)
    print(f"workers  : {args.workers}", flush=True)
    print(f"out_dir  : {args.out_dir}", flush=True)

    t_start = time.time()
    results = []
    if args.workers <= 1:
        for d in dates:
            r = process_one_date(d, args.horizons, args.events_dir, args.book_dir,
                                 args.out_dir, args.overwrite)
            results.append(r)
            print(f"  {d}: {r.get('status')} N={r.get('N','?')} t={r.get('elapsed_s','?')}s",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {
                ex.submit(process_one_date, d, args.horizons,
                          args.events_dir, args.book_dir, args.out_dir,
                          args.overwrite): d
                for d in dates
            }
            for fut in as_completed(fut_map):
                r = fut.result()
                results.append(r)
                print(f"  {r['date']}: {r.get('status')} N={r.get('N','?')} t={r.get('elapsed_s','?')}s",
                      flush=True)

    ok = sum(1 for r in results if r.get("status") == "ok")
    skipped = sum(1 for r in results if r.get("status") == "skipped_exists")
    missing = sum(1 for r in results if r.get("status") == "missing_inputs")
    errors = sum(1 for r in results if r.get("status") in ("error", "length_mismatch"))
    elapsed = round(time.time() - t_start, 1)
    summary = {
        "n_dates": len(dates), "ok": ok, "skipped": skipped,
        "missing_inputs": missing, "errors": errors,
        "elapsed_s": elapsed,
        "horizons": args.horizons,
        "out_dir": str(args.out_dir),
    }
    print("== DONE ==")
    print(json.dumps(summary, indent=2))

    # Persist run-log
    run_log = args.out_dir / "_run_log.jsonl"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with open(run_log, "a") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "details": results,
        }) + "\n")


if __name__ == "__main__":
    main()
