#!/usr/bin/env python3
"""
Process raw .dbn.zst MBO files into _mbo_events.npz format for training.

Scans data/raw/mbo/ for unprocessed dates and converts each to the 6-feature
event format with forward-looking mid-price labels at 1s/5s/10s/30s horizons.

Usage:
    python process_missing_mbo.py                    # process all missing
    python process_missing_mbo.py --workers 4        # parallel
    python process_missing_mbo.py --force 20260313   # reprocess specific date
"""

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw" / "mbo"
OUT_DIR = ROOT / "data" / "processed" / "mbo_events"

TICK_SIZE = 0.25
HORIZONS_NS = {"1s": 1_000_000_000, "5s": 5_000_000_000,
               "10s": 10_000_000_000, "30s": 30_000_000_000}
ACTION_MAP = {"A": 0, "C": 1, "M": 2, "T": 3, "F": 4}
SIDE_MAP = {"B": 0, "A": 1, "N": 2}
PRICE_CLIP = 50

# ES contract rollover (cutoff_date, instrument_id)
ES_CONTRACTS = [
    ("2025-09-19", 14160),       # ESU5
    ("2025-12-19", 294973),      # ESZ5
    ("2026-03-20", 42140878),    # ESH6
    ("2026-06-19", None),        # ESM6
]


def get_instrument_id(date_str: str):
    d = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    for cutoff, iid in ES_CONTRACTS:
        if d < cutoff:
            return iid
    return ES_CONTRACTS[-1][1]


def auto_detect_front_month(df, date_str: str):
    """When instrument_id is None, auto-detect the front-month ES contract.

    Strategy: pick the instrument_id with the most trade events (action=T).
    This handles rollover periods where both old and new contracts are active.
    Falls back to the instrument with the most total events if no trades found.
    """
    if "instrument_id" not in df.columns:
        return None

    # Try trade events first (most reliable indicator of front month)
    trades = df[df["action"] == "T"] if "action" in df.columns else df
    if len(trades) > 0:
        iid = trades["instrument_id"].value_counts().idxmax()
        count = trades["instrument_id"].value_counts().max()
        total = len(trades)
        logger.info(f"  Auto-detected front-month instrument_id={iid} "
                     f"({count}/{total} trades, {count/total*100:.0f}%)")
        return int(iid)

    # Fallback: most events overall
    iid = df["instrument_id"].value_counts().idxmax()
    logger.info(f"  Auto-detected instrument_id={iid} (most events)")
    return int(iid)


def find_missing() -> list[tuple[Path, str]]:
    existing = {f.stem.split("_")[0] for f in OUT_DIR.glob("*_mbo_events.npz")}
    missing = []
    for f in sorted(RAW_DIR.glob("*.dbn.zst")):
        m = re.search(r"(\d{8})", f.name)
        if m and m.group(1) not in existing:
            missing.append((f, m.group(1)))
    return missing


def track_bbo(prices, actions, sides, n):
    """Single-pass BBO tracker. Returns mid_prices and spread arrays."""
    mid_arr = np.full(n, np.nan, dtype=np.float64)
    spread_arr = np.zeros(n, dtype=np.float64)
    bb, ba = np.nan, np.nan

    for i in range(n):
        p, act, side = prices[i], actions[i], sides[i]
        if not np.isnan(p):
            if act == "T" or act == "F":
                if side == "A":
                    bb = p
                elif side == "B":
                    ba = p
            elif act == "A":
                if side == "B" and (np.isnan(bb) or p > bb):
                    bb = p
                elif side == "A" and (np.isnan(ba) or p < ba):
                    ba = p
            elif act == "C":
                if side == "B" and not np.isnan(bb) and p >= bb:
                    bb = p - TICK_SIZE
                elif side == "A" and not np.isnan(ba) and p <= ba:
                    ba = p + TICK_SIZE

        if not np.isnan(bb) and not np.isnan(ba):
            mid_arr[i] = (bb + ba) / 2.0
            spread_arr[i] = max(0.0, (ba - bb) / TICK_SIZE)
        elif i > 0:
            mid_arr[i] = mid_arr[i - 1]
            spread_arr[i] = spread_arr[i - 1]

    # Forward-fill remaining NaNs
    for i in range(1, n):
        if np.isnan(mid_arr[i]):
            mid_arr[i] = mid_arr[i - 1]
    return mid_arr, spread_arr


def compute_labels(timestamps, mid_prices, n):
    """Two-pointer label computation for each horizon."""
    labels = {}
    for name, hz_ns in HORIZONS_NS.items():
        lbl = np.full(n, np.nan, dtype=np.float32)
        j = 0
        for i in range(n):
            target = timestamps[i] + hz_ns
            while j < n - 1 and timestamps[j] < target:
                j += 1
            if timestamps[j] >= target and not np.isnan(mid_prices[j]) and not np.isnan(mid_prices[i]):
                lbl[i] = np.float32((mid_prices[j] - mid_prices[i]) / TICK_SIZE)
            # Don't reset j — timestamps are sorted, so next i needs j >= current j
        labels[name] = lbl
    return labels


def process_file(raw_path: str, date_str: str, out_dir: str = None) -> tuple[str, int, str]:
    """Convert one .dbn.zst to _mbo_events.npz."""
    raw_path = Path(raw_path)
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    try:
        import databento as db
    except ImportError:
        return date_str, 0, "databento not installed"

    t0 = time.time()
    try:
        store = db.DBNStore.from_file(str(raw_path))
        df = store.to_df()
        df.columns = [c.lower() for c in df.columns]

        iid = get_instrument_id(date_str)
        if iid is None:
            # ESM6 or unknown contract — auto-detect from data
            iid = auto_detect_front_month(df, date_str)
        if iid and "instrument_id" in df.columns:
            df = df[df["instrument_id"] == iid]

        n_total = len(df)
        if n_total == 0:
            return date_str, 0, "no events after instrument filter"

        # Price scaling check
        if "price" in df.columns and len(df) > 0:
            sample = df["price"].dropna().iloc[0] if len(df["price"].dropna()) > 0 else 0
            if sample > 1e6:
                df["price"] = df["price"] * 1e-9

        if "ts_event" not in df.columns:
            return date_str, 0, "no ts_event column"

        df = df.sort_values("ts_event").reset_index(drop=True)
        df = df[df["action"].isin(["A", "C", "M", "T", "F"])].reset_index(drop=True)
        if len(df) == 0:
            return date_str, 0, "no valid events"

        n = len(df)
        timestamps = df["ts_event"].values.astype("int64")
        prices = df["price"].values.astype(np.float64)
        sizes = df["size"].values.astype(np.float64)
        actions_str = df["action"].values
        sides_str = df["side"].values

        # BBO tracking
        mid_prices, spread_ticks = track_bbo(prices, actions_str, sides_str, n)

        # Build 6-feature event array
        events = np.zeros((n, 6), dtype=np.float32)

        # F0: time_delta_log
        dt_ns = np.diff(timestamps, prepend=timestamps[0]).astype(np.float64)
        dt_ns = np.maximum(dt_ns, 0)
        # log(dt_seconds), clipped to [0, 10]
        dt_sec = dt_ns / 1e9
        events[:, 0] = np.clip(np.log(np.maximum(dt_sec, 1e-15)), 0.0, 10.0).astype(np.float32)
        events[0, 0] = 0.0

        # F1: event_type_id
        events[:, 1] = np.array([ACTION_MAP.get(a, 0) for a in actions_str], dtype=np.float32)

        # F2: side_id
        events[:, 2] = np.array([SIDE_MAP.get(s, 2) for s in sides_str], dtype=np.float32)

        # F3: price_rel_ticks
        rel = (prices - mid_prices) / TICK_SIZE
        events[:, 3] = np.clip(np.nan_to_num(rel, 0.0), -PRICE_CLIP, PRICE_CLIP).astype(np.float32)

        # F4: qty_log (min size=2 so log >= 0.693)
        events[:, 4] = np.log(np.maximum(sizes, 2.0)).astype(np.float32)

        # F5: spread_ticks
        events[:, 5] = np.clip(spread_ticks, 0.0, 20.0).astype(np.float32)

        # Labels
        labels = compute_labels(timestamps, mid_prices, n)

        # Metadata
        metadata = json.dumps({
            "date": date_str,
            "instrument_id": int(iid) if iid else None,
            "tick_size": TICK_SIZE,
            "n_events": n,
            "n_total_records": n_total,
            "feature_names": ["time_delta_log", "event_type_id", "side_id",
                              "price_rel_ticks", "qty_log", "spread_ticks"],
            "action_encoding": {"A": 0, "C": 1, "M": 2, "T": 3, "F": 4},
            "side_encoding": {"B": 0, "A": 1, "N": 2},
            "label_horizons": list(HORIZONS_NS.keys()),
        })

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{date_str}_mbo_events.npz"
        np.savez_compressed(
            str(out_path),
            events=events, labels_1s=labels["1s"], labels_5s=labels["5s"],
            labels_10s=labels["10s"], labels_30s=labels["30s"],
            timestamps=timestamps, metadata=np.array([metadata]),
        )

        elapsed = time.time() - t0
        return date_str, n, f"ok ({elapsed:.0f}s)"
    except Exception as e:
        import traceback
        return date_str, 0, f"{e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser(description="Process raw .dbn.zst MBO files to NPZ")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--force", nargs="*", help="Force reprocess dates (YYYYMMDD)")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.force:
        tasks = []
        for d in args.force:
            matches = list(raw_dir.glob(f"*{d}*.dbn.zst"))
            if matches:
                tasks.append((str(matches[0]), d))
            else:
                logger.warning(f"No raw file for date {d}")
    else:
        existing = {f.stem.split("_")[0] for f in out_dir.glob("*_mbo_events.npz")}
        tasks = []
        for f in sorted(raw_dir.glob("*.dbn.zst")):
            m = re.search(r"(\d{8})", f.name)
            if m and m.group(1) not in existing:
                tasks.append((str(f), m.group(1)))

    if not tasks:
        logger.info("Nothing to process — all dates already converted.")
        return

    logger.info(f"Processing {len(tasks)} files: {raw_dir} -> {out_dir}")

    results = []
    od = str(out_dir)
    if args.workers <= 1:
        for raw_path, date_str in tasks:
            logger.info(f"Processing {date_str}...")
            r = process_file(raw_path, date_str, od)
            results.append(r)
            logger.info(f"  {r[0]}: {r[1]:,} events — {r[2]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(process_file, p, d, od): d for p, d in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                logger.info(f"  {r[0]}: {r[1]:,} events — {r[2]}")

    ok = [r for r in results if r[2].startswith("ok")]
    fail = [r for r in results if not r[2].startswith("ok")]
    logger.info(f"\nDone: {len(ok)}/{len(tasks)} succeeded, {sum(r[1] for r in ok):,} total events")
    if fail:
        for d, _, msg in fail:
            logger.warning(f"  FAILED {d}: {msg}")


if __name__ == "__main__":
    main()
