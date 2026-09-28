#!/usr/bin/env python3
"""spy_walkforward_smoke.py — Ingest 1 day of SPY MBO into the WF harness shape.

This is an INGEST-ONLY smoke test. It does NOT launch training. Per directive,
training requires explicit user OK. We only verify:
  1. SPY NPZs load with the same numpy keys the WF harness expects
  2. The 6-col events array passes the dtype/shape checks
  3. A SLIDING-window iterator over a 60-day train / 1-day OOT slice works
     (synthesized timeline since we only have 1 fixture day)
  4. Mid-price reconstruction works using SPY tick size, not ES

Usage:
    python feeds/spy_walkforward_smoke.py \
        --data-dir /home/jupiter/Lvl3Quant/data/processed/spy_mbo_events

Exit code 0 = pipeline ready for SPY training when user authorizes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make our cost constants importable
LVL3 = Path("/home/jupiter/Lvl3Quant")
if str(LVL3) not in sys.path:
    sys.path.insert(0, str(LVL3))
import cost_constants_spy as costs

EXPECTED_KEYS = ["events", "timestamps", "labels_1s", "labels_5s",
                 "labels_10s", "labels_30s", "metadata"]
EXPECTED_FEATURE_NAMES = ["time_delta_log", "event_type_id", "side_id",
                          "price_rel_ticks", "qty_log", "spread_ticks"]


def check_file(path: Path) -> dict:
    """Validate one NPZ. Returns a report dict."""
    rep = {"path": str(path), "ok": False, "issues": []}
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        rep["issues"].append(f"FAILED TO LOAD: {e}")
        return rep

    for k in EXPECTED_KEYS:
        if k not in d.files:
            rep["issues"].append(f"missing key: {k}")
    if rep["issues"]:
        return rep

    events = d["events"]; ts = d["timestamps"]
    if events.dtype != np.float32:
        rep["issues"].append(f"events dtype {events.dtype} != float32")
    if ts.dtype != np.int64:
        rep["issues"].append(f"timestamps dtype {ts.dtype} != int64")
    if events.ndim != 2 or events.shape[1] != 6:
        rep["issues"].append(f"events shape {events.shape} != (N, 6)")
    if events.shape[0] != ts.shape[0]:
        rep["issues"].append(f"events N={events.shape[0]} != ts N={ts.shape[0]}")

    # Monotonic-ish ts (allow ties)
    if ts.size > 1:
        diffs = np.diff(ts)
        n_neg = int((diffs < 0).sum())
        if n_neg > 0:
            rep["issues"].append(f"non-monotonic ts: {n_neg} negative deltas")

    # Spread sanity at SPY tick size (col 5)
    spr = events[:, 5]
    if spr.size:
        rep["spread_ticks_mean"] = float(spr.mean())
        rep["spread_ticks_max"]  = float(spr.max())

    # Metadata sanity
    try:
        meta = json.loads(str(d["metadata"][0]))
    except Exception as e:
        rep["issues"].append(f"metadata not parseable: {e}")
        meta = {}
    if meta.get("tick_size") != costs.TICK_SIZE:
        rep["issues"].append(
            f"metadata tick_size {meta.get('tick_size')} != "
            f"cost_constants_spy.TICK_SIZE {costs.TICK_SIZE}")
    if meta.get("feature_names") != EXPECTED_FEATURE_NAMES:
        rep["issues"].append("feature_names mismatch")

    rep["n_events"] = int(events.shape[0])
    rep["symbol"] = meta.get("symbol")
    rep["source"] = meta.get("source")
    rep["ok"] = len(rep["issues"]) == 0
    return rep


def reconstruct_mid_prices(events: np.ndarray, seed_mid: float = 580.0) -> np.ndarray:
    """Recover absolute mid prices from price_rel_ticks (col 3).

    This is the same step `walkforward_oot_lean.py` does (with ES tick=0.25).
    Returns float32 array of length N.
    """
    return (seed_mid + events[:, 3] * costs.TICK_SIZE).astype(np.float32)


def simulate_sliding_window_indexing(n_days: int = 1,
                                     train_days: int = 60) -> dict:
    """The HC #0 sliding window: 60 train, 1 OOT, drop oldest each fold.

    Even though we only have 1 fixture day, we PROVE the indexing math is
    correct so when 61 days of SPY data accumulate, the harness can step
    forward without modification.
    """
    available = list(range(train_days + 5))  # pretend we have 65 days
    folds = []
    for fold_id in range(len(available) - train_days):
        train = available[fold_id:fold_id + train_days]
        oot = available[fold_id + train_days]
        folds.append({"fold": fold_id, "train_first": train[0],
                      "train_last": train[-1], "oot": oot})
    return {"n_folds": len(folds), "first_fold": folds[0],
            "last_fold": folds[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",
                    default="/home/jupiter/Lvl3Quant/data/processed/spy_mbo_events")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)

    files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not files:
        print(f"NO SPY NPZ FOUND in {data_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(files)} SPY NPZ file(s) in {data_dir}")
    all_ok = True
    for f in files:
        rep = check_file(f)
        status = "OK " if rep["ok"] else "FAIL"
        print(f"  [{status}] {f.name}  n={rep.get('n_events','?')}  "
              f"sym={rep.get('symbol')}  src={rep.get('source')}")
        if rep["issues"]:
            for i in rep["issues"]:
                print(f"        - {i}")
            all_ok = False
        if "spread_ticks_mean" in rep:
            print(f"        spread_ticks mean={rep['spread_ticks_mean']:.3f} "
                  f"max={rep['spread_ticks_max']:.1f}")

    # Mid-price reconstruction smoke test
    d = np.load(files[0], allow_pickle=True)
    mid = reconstruct_mid_prices(d["events"])
    print(f"\nMid-price reconstruction: N={len(mid)}  "
          f"min=${mid.min():.2f}  max=${mid.max():.2f}  mean=${mid.mean():.2f}")
    if not (mid.min() > 0 and mid.max() < 10_000):
        print("  WARN: reconstructed mid prices look off-range")
        all_ok = False

    # Sliding-window math
    wf = simulate_sliding_window_indexing()
    print(f"\nSLIDING window WF math (60 train / 1 OOT, oldest-day-drop):")
    print(f"  total folds at 65-day horizon: {wf['n_folds']}")
    print(f"  fold 0:    train days {wf['first_fold']['train_first']}..{wf['first_fold']['train_last']}  OOT={wf['first_fold']['oot']}")
    print(f"  last fold: train days {wf['last_fold']['train_first']}..{wf['last_fold']['train_last']}  OOT={wf['last_fold']['oot']}")

    print("\n=== RESULT ===")
    print("READY for SPY walk-forward (pending: 60+ days of SPY captures and user GO)"
          if all_ok else "BLOCKED — fix issues above before launching training")
    print("NOTE: training intentionally NOT launched (requires user authorization).")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
