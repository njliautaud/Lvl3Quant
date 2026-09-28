#!/usr/bin/env python3
"""
HC #451 R5 — Pressure-style label cache builder.

For every NPZ in data/processed/mbo_events_smart_v3/<date>_mbo_events.npz
compute 5 derived label arrays and write to
data/processed/mbo_events_smart_v3_pressure_labels/<date>_pressure.npz.

Labels computed (length == n_events per source NPZ):
  - persistence_1s_10s  (int8)  in {-1, 0, +1}
  - persistence_1s_30s  (int8)  in {-1, 0, +1}
  - persistence_5s_30s  (int8)  in {-1, 0, +1}
  - mfe_minus_mae_10s   (float32, in same units as source labels = ticks)
  - pressure_score      (float32) in [-1, +1]

Persistence rule (with neutrality band b, default 0.5 in label units):
    sign_a = +1 if a > b else (-1 if a < -b else 0)
    sign_b = +1 if b > b else (-1 if b < -b else 0)
    if either sign == 0 -> 0  (neutral)
    elif sign_a == sign_b -> +1
    else -> -1

NaN handling:
    Any input NaN -> persistence = 0, mfe_minus_mae_10s = NaN, pressure_score = NaN.

Usage:
    python hc451_build_pressure_labels.py [--dates 20250714,20250715] [--force]
        [--output-dir ...] [--input-dir ...] [--neutrality-bps 0.5] [--workers 8]

NOTE on units: the existing NPZ labels are stored in TICKS (not bps) per
alpha_discovery/evaluation/mfe_mae_path_analysis.py header. The --neutrality-bps
flag name is kept for compatibility with the HC #451 task brief, but the value
is applied in the same units as the source label arrays (ticks).
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DEFAULT_INPUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3"
DEFAULT_OUTPUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_pressure_labels"
DATE_RE = re.compile(r"^(\d{8})_mbo_events\.npz$")


def _classify_sign(arr: np.ndarray, neutrality: float) -> np.ndarray:
    """Return int8 signs: +1 if x>b, -1 if x<-b, 0 otherwise (NaN -> 0)."""
    out = np.zeros(arr.shape, dtype=np.int8)
    valid = ~np.isnan(arr)
    out[valid & (arr > neutrality)] = 1
    out[valid & (arr < -neutrality)] = -1
    return out


def _persistence(arr_a: np.ndarray, arr_b: np.ndarray, neutrality: float) -> np.ndarray:
    """+1 if signs agree (both non-neutral), -1 if disagree, 0 if either neutral."""
    sa = _classify_sign(arr_a, neutrality)
    sb = _classify_sign(arr_b, neutrality)
    out = np.zeros(arr_a.shape, dtype=np.int8)
    both_active = (sa != 0) & (sb != 0)
    agree = both_active & (sa == sb)
    disagree = both_active & (sa != sb)
    out[agree] = 1
    out[disagree] = -1
    # NaN propagation: if either input NaN, force 0 (already default)
    nan_mask = np.isnan(arr_a) | np.isnan(arr_b)
    out[nan_mask] = 0
    return out


def _mfe_minus_mae_10s(l1, l5, l10) -> np.ndarray:
    """Approximate MFE - MAE within 10s using 3 sample points (1s, 5s, 10s).

    Best-path = max over [l1, l5, l10] (highest favorable excursion seen)
    Worst-path = min over [l1, l5, l10] (worst adverse excursion seen)
    Result    = best - worst (always >= 0 on valid data)
    NaN if any input NaN.
    """
    stk = np.stack([l1, l5, l10], axis=0)  # (3, N)
    # Any NaN at a row -> nan output
    nan_mask = np.isnan(stk).any(axis=0)
    best = np.nanmax(stk, axis=0)
    worst = np.nanmin(stk, axis=0)
    out = (best - worst).astype(np.float32)
    out[nan_mask] = np.nan
    return out


def _pressure_score(l1, l5, l10, l30, neutrality: float) -> np.ndarray:
    """tanh(2 * mean_sign) across the 4 horizons. NaN if any input NaN."""
    s1 = _classify_sign(l1, neutrality).astype(np.float32)
    s5 = _classify_sign(l5, neutrality).astype(np.float32)
    s10 = _classify_sign(l10, neutrality).astype(np.float32)
    s30 = _classify_sign(l30, neutrality).astype(np.float32)
    agreement = (s1 + s5 + s10 + s30) / 4.0
    out = np.tanh(2.0 * agreement).astype(np.float32)
    nan_mask = np.isnan(l1) | np.isnan(l5) | np.isnan(l10) | np.isnan(l30)
    out[nan_mask] = np.nan
    return out


def build_one(args_tuple):
    """Worker: process one date file. Returns (date, n_events, ok, msg)."""
    src_path, dst_path, neutrality, force = args_tuple
    date = Path(src_path).stem.replace("_mbo_events", "")
    if os.path.exists(dst_path) and not force:
        try:
            with np.load(dst_path) as d:
                n = len(d["pressure_score"])
            return (date, n, True, "skip-exists")
        except Exception:
            pass  # rebuild on error
    try:
        with np.load(src_path) as d:
            l1 = d["labels_1s"].astype(np.float32, copy=False)
            l5 = d["labels_5s"].astype(np.float32, copy=False)
            l10 = d["labels_10s"].astype(np.float32, copy=False)
            l30 = d["labels_30s"].astype(np.float32, copy=False)
    except Exception as e:
        return (date, 0, False, f"load-fail: {e}")

    n = len(l1)
    p1_10 = _persistence(l1, l10, neutrality)
    p1_30 = _persistence(l1, l30, neutrality)
    p5_30 = _persistence(l5, l30, neutrality)
    mma = _mfe_minus_mae_10s(l1, l5, l10)
    ps = _pressure_score(l1, l5, l10, l30, neutrality)

    tmp = dst_path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        persistence_1s_10s=p1_10,
        persistence_1s_30s=p1_30,
        persistence_5s_30s=p5_30,
        mfe_minus_mae_10s=mma,
        pressure_score=ps,
        neutrality=np.float32(neutrality),
    )
    os.replace(tmp, dst_path)
    return (date, n, True, "built")


def discover_dates(input_dir: str):
    out = []
    for f in sorted(os.listdir(input_dir)):
        m = DATE_RE.match(f)
        if m:
            out.append(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--dates", default="all",
                    help="comma-separated YYYYMMDD list, or 'all'")
    ap.add_argument("--neutrality-bps", type=float, default=0.5,
                    help="Neutrality band (applied in source label units = ticks). Default 0.5.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_dates = discover_dates(args.input_dir)
    if args.dates.lower() == "all":
        dates = all_dates
    else:
        wanted = set(args.dates.split(","))
        dates = [d for d in all_dates if d in wanted]

    if not dates:
        print("No dates to process.", file=sys.stderr)
        return 1

    jobs = []
    for d in dates:
        src = os.path.join(args.input_dir, f"{d}_mbo_events.npz")
        dst = os.path.join(args.output_dir, f"{d}_pressure.npz")
        jobs.append((src, dst, args.neutrality_bps, args.force))

    print(f"Processing {len(jobs)} dates with {args.workers} workers, "
          f"neutrality={args.neutrality_bps} (ticks)", flush=True)
    t0 = time.time()
    built = skipped = failed = 0
    total_events = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            date, n, ok, msg = fut.result()
            total_events += n
            if not ok:
                failed += 1
                print(f"[{i}/{len(jobs)}] {date} FAIL: {msg}", flush=True)
            elif msg == "skip-exists":
                skipped += 1
            else:
                built += 1
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] built={built} skipped={skipped} "
                      f"failed={failed} events={total_events:,} "
                      f"elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"DONE: built={built} skipped={skipped} failed={failed} "
          f"total_events={total_events:,} in {time.time()-t0:.1f}s")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
