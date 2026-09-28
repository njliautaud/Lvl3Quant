#!/usr/bin/env python3
"""
expand_mfe_mae_full_session.py — full-session MFE/MAE-within-horizon labels.

Diagnosis (HC #466/#467 stream backtest follow-up):
  The existing `relabel_mfe_mae_multi_horizon.py` produces parquets covering
  only ~6% of each session (~12 minutes). Root cause is the validity gate
  `(bid > 0) & (ask > 0)` inside `process_one_date()` — the book_features
  bid/ask columns are encoded as RELATIVE OFFSETS (around a session anchor)
  that go negative for most of the day. A spread-based gate (ask - bid > 0,
  ask - bid < SPREAD_MAX) admits ~99.9% of events for every date checked.

Strategy here:
  - Reuse the proven O(N) monotonic-deque scan from `relabel_mfe_mae_multi_horizon.py`
    (import via sys.path) — same correctness, just a different validity gate.
  - Output to `/data/relabel_full/` (parallel to `/data/relabel/`) so the existing
    sparse parquets stay intact.
  - Smoke test: 3 dates (20260223, 20260301, 20260401). Coverage should jump 10-20x
    (3,100 valid pred-step rows -> 49,500 valid pred-step rows).

Outputs (per date, per horizon):
  /data/relabel_full/mfe_mae_h<h>_<date>.parquet
  Schema: event_idx int32 | ts_ns int64 | mid_t_ticks float32 | mfe_ticks float32 |
          mae_ticks float32 | time_to_mfe_s float32 | mfe_minus_mae_ticks float32

Read-only on existing relabel parquets and source NPZs.
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

# Import O(N) windowed scan and constants from the existing relabel script (sibling import)
sys.path.insert(0, str(Path(__file__).parent))
from relabel_mfe_mae_multi_horizon import (
    _two_pointer_indices,
    _windowed_minmax_argmax_njit,
    SEC_NS,
    HORIZON_NS,
)

EVENTS_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
BOOK_DIR   = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_book_features")
OUT_DIR    = Path("/home/jupiter/Lvl3Quant/data/relabel_full")

# Bid/ask are stored as RELATIVE offsets (around an unknown session anchor). The original
# script's `bid > 0 & ask > 0` gate is wrong. A spread-based gate is the correct check:
# ES book is 1-2 ticks wide ~99% of the time during RTH. We admit 0 < spread < SPREAD_MAX.
SPREAD_MAX_TICKS = 20.0


def _mfe_mae_for_horizon(mids_clean: np.ndarray, ts_ns: np.ndarray, h_ns: int):
    """Returns (mfe_ticks, mae_ticks, time_to_mfe_secs) for one horizon."""
    j_ends = _two_pointer_indices(ts_ns, h_ns)
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
    valid = (argmaxs >= 0) & np.isfinite(mfe)
    tt = np.full(len(mids_clean), np.nan, dtype=np.float32)
    if valid.any():
        idxs = np.where(valid)[0]
        dt_ns = ts_ns[argmaxs[idxs]] - ts_ns[idxs]
        tt[idxs] = (dt_ns / SEC_NS).astype(np.float32)
    return mfe, mae, tt


def process_one_date(date_str: str, horizons: list[str], out_dir: Path,
                     overwrite: bool = False) -> dict:
    out_paths = {h: out_dir / f"mfe_mae_h{h}_{date_str}.parquet" for h in horizons}
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return {"date": date_str, "status": "skipped_exists"}

    events_path = EVENTS_DIR / f"{date_str}_mbo_events.npz"
    book_path   = BOOK_DIR   / f"{date_str}_book_features.npz"
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
        N = len(ts_ns)
        if len(bid) != N:
            return {"date": date_str, "status": "length_mismatch",
                    "N_events": N, "N_book": len(bid)}

        spread = ask - bid
        # CORRECTED validity gate: spread strictly positive AND realistic (<20 ticks)
        invalid = (spread <= 0) | (spread >= SPREAD_MAX_TICKS) | ~np.isfinite(spread)
        mids = (bid + ask) / 2.0
        mids_clean = mids.copy()
        mids_clean[invalid] = np.nan
        n_valid = int(np.isfinite(mids_clean).sum())

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
            "N": int(N), "n_valid_mids": n_valid,
            "valid_pct": round(100.0 * n_valid / N, 2),
            "elapsed_s": round(time.time() - t0, 2),
            "horizons": wrote,
        }
    except Exception as e:
        return {"date": date_str, "status": "error", "err": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--horizons", nargs="+", default=["1s", "5s", "10s", "30s"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    print(f"== expand_mfe_mae_full_session ==", flush=True)
    print(f"dates    : {len(args.dates)} ({args.dates[0]}..{args.dates[-1]})", flush=True)
    print(f"horizons : {args.horizons}", flush=True)
    print(f"workers  : {args.workers}", flush=True)
    print(f"out_dir  : {args.out_dir}", flush=True)

    t_start = time.time()
    results = []
    if args.workers <= 1:
        for d in args.dates:
            r = process_one_date(d, args.horizons, args.out_dir, args.overwrite)
            results.append(r)
            print(f"  {d}: {r}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {ex.submit(process_one_date, d, args.horizons, args.out_dir, args.overwrite): d for d in args.dates}
            for fut in as_completed(fut_map):
                r = fut.result()
                results.append(r)
                print(f"  {r.get('date')}: status={r.get('status')} N={r.get('N','?')} valid_pct={r.get('valid_pct','?')} t={r.get('elapsed_s','?')}s", flush=True)

    ok = sum(1 for r in results if r.get("status") == "ok")
    elapsed = round(time.time() - t_start, 1)
    summary = {"n_dates": len(args.dates), "ok": ok, "elapsed_s": elapsed,
               "horizons": args.horizons, "out_dir": str(args.out_dir)}
    print("== DONE ==")
    print(json.dumps(summary, indent=2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_log = args.out_dir / "_run_log.jsonl"
    with open(run_log, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": summary, "details": results}) + "\n")


if __name__ == "__main__":
    main()
