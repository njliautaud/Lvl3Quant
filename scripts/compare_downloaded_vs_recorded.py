#!/usr/bin/env python3
"""
Compare downloaded Databento .dbn.zst files with processed NPZ files.

Validates that our processing pipeline produces correct output from raw
Databento data, and that live-recorded NPZ files match the same schema.

This proves we can generate our own data files independently.

Usage:
    python scripts/compare_downloaded_vs_recorded.py
    python scripts/compare_downloaded_vs_recorded.py --dates 20260301,20260302
    python scripts/compare_downloaded_vs_recorded.py --sample 10  # random 10 dates
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "mbo"
NPZ_DIR = ROOT / "data" / "processed" / "mbo_events"

# Expected NPZ schema
EXPECTED_KEYS = {"events", "timestamps", "labels_1s", "labels_5s", "labels_10s", "labels_30s"}
EXPECTED_EVENT_COLS = 6  # time_delta, event_type, side, price_rel, qty_log, spread
EXPECTED_DTYPES = {
    "events": np.float32,
    "timestamps": np.int64,
    "labels_1s": np.float32,
    "labels_5s": np.float32,
    "labels_10s": np.float32,
    "labels_30s": np.float32,
}

# ES contract mapping (same as process_missing_mbo.py)
ES_CONTRACTS = [
    ("2025-09-19", 14160),
    ("2025-12-19", 294973),
    ("2026-03-20", 42140878),
    ("2026-06-19", None),
]

ACTION_MAP = {"A": 0, "C": 1, "M": 2, "T": 3, "F": 4}
SIDE_MAP = {"B": 0, "A": 1, "N": 2}


def get_instrument_id(date_str):
    d = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    for cutoff, iid in ES_CONTRACTS:
        if d < cutoff:
            return iid
    return ES_CONTRACTS[-1][1]


def validate_npz_schema(npz_path):
    """Validate NPZ file has correct schema, dtypes, and structure."""
    issues = []
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        return {"valid": False, "issues": [f"Cannot load: {e}"], "stats": {}}

    keys = set(data.files)

    # Check required keys
    missing = EXPECTED_KEYS - keys
    if missing:
        issues.append(f"Missing keys: {missing}")

    extra = keys - EXPECTED_KEYS - {"metadata"}
    if extra:
        issues.append(f"Unexpected keys: {extra}")

    stats = {}

    # Check events array
    if "events" in keys:
        events = data["events"]
        stats["n_events"] = len(events)
        stats["event_shape"] = events.shape

        if events.dtype != np.float32:
            issues.append(f"events dtype={events.dtype}, expected float32")
        if len(events.shape) != 2 or events.shape[1] != EXPECTED_EVENT_COLS:
            issues.append(f"events shape={events.shape}, expected (N, {EXPECTED_EVENT_COLS})")

        # Check value ranges
        if len(events) > 0:
            stats["event_type_unique"] = sorted(np.unique(events[:, 1]).tolist())
            stats["side_unique"] = sorted(np.unique(events[:, 2]).tolist())
            stats["price_rel_range"] = (float(np.nanmin(events[:, 3])), float(np.nanmax(events[:, 3])))
            stats["spread_range"] = (float(np.nanmin(events[:, 5])), float(np.nanmax(events[:, 5])))

    # Check timestamps
    if "timestamps" in keys:
        ts = data["timestamps"]
        stats["n_timestamps"] = len(ts)
        if ts.dtype != np.int64:
            issues.append(f"timestamps dtype={ts.dtype}, expected int64")
        if len(ts) > 1:
            if not np.all(np.diff(ts) >= 0):
                issues.append("timestamps NOT monotonically increasing")
            stats["ts_range_ns"] = (int(ts[0]), int(ts[-1]))
            # Convert to human readable
            from datetime import datetime, timezone
            stats["ts_start"] = datetime.fromtimestamp(ts[0] / 1e9, tz=timezone.utc).isoformat()
            stats["ts_end"] = datetime.fromtimestamp(ts[-1] / 1e9, tz=timezone.utc).isoformat()

        # Check alignment with events
        if "events" in keys and len(ts) != len(data["events"]):
            issues.append(f"timestamps len={len(ts)} != events len={len(data['events'])}")

    # Check labels
    for h in ["1s", "5s", "10s", "30s"]:
        key = f"labels_{h}"
        if key in keys:
            lbl = data[key]
            if lbl.dtype != np.float32:
                issues.append(f"{key} dtype={lbl.dtype}, expected float32")
            if "events" in keys and len(lbl) != len(data["events"]):
                issues.append(f"{key} len={len(lbl)} != events len={len(data['events'])}")

            n_nan = int(np.isnan(lbl).sum())
            n_total = len(lbl)
            pct_valid = (n_total - n_nan) / n_total * 100 if n_total > 0 else 0
            stats[f"{key}_nan_pct"] = round(100 - pct_valid, 2)
            stats[f"{key}_range"] = (float(np.nanmin(lbl)), float(np.nanmax(lbl))) if n_total - n_nan > 0 else None

    # Check metadata
    if "metadata" in keys:
        try:
            meta = data["metadata"].item()
            if isinstance(meta, str):
                meta = json.loads(meta)
            stats["metadata"] = meta
        except Exception:
            stats["metadata"] = "unparseable"

    return {"valid": len(issues) == 0, "issues": issues, "stats": stats}


def compare_raw_vs_npz(date_str):
    """Compare a raw .dbn.zst file with its corresponding NPZ file."""
    raw_path = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    npz_path = NPZ_DIR / f"{date_str}_mbo_events.npz"

    result = {
        "date": date_str,
        "has_raw": raw_path.exists(),
        "has_npz": npz_path.exists(),
        "match": False,
        "issues": [],
        "raw_stats": {},
        "npz_stats": {},
    }

    if not npz_path.exists():
        result["issues"].append("NPZ file missing")
        return result

    # Validate NPZ schema
    npz_val = validate_npz_schema(npz_path)
    result["npz_stats"] = npz_val["stats"]
    result["issues"].extend(npz_val["issues"])

    if not raw_path.exists():
        result["issues"].append("Raw .dbn.zst file missing (live-recorded only)")
        result["match"] = npz_val["valid"]  # Can only validate schema
        return result

    # Load raw file and compare
    try:
        import databento as db
        store = db.DBNStore.from_file(str(raw_path))
        df = store.to_df()
        df.columns = [c.lower() for c in df.columns]

        result["raw_stats"]["n_total_records"] = len(df)

        # Filter by instrument
        iid = get_instrument_id(date_str)
        if iid is None:
            # Auto-detect
            if "instrument_id" in df.columns:
                trades = df[df["action"] == "T"] if "action" in df.columns else df
                if len(trades) > 0:
                    iid = int(trades["instrument_id"].value_counts().idxmax())

        if iid and "instrument_id" in df.columns:
            df = df[df["instrument_id"] == iid]
            result["raw_stats"]["instrument_id"] = int(iid)

        # Filter valid actions
        df = df[df["action"].isin(["A", "C", "M", "T", "F"])]
        result["raw_stats"]["n_filtered_events"] = len(df)

        # Compare event counts
        npz_events = result["npz_stats"].get("n_events", 0)
        raw_events = len(df)
        result["raw_stats"]["n_events"] = raw_events

        if npz_events != raw_events:
            pct_diff = abs(npz_events - raw_events) / max(raw_events, 1) * 100
            if pct_diff > 1.0:
                result["issues"].append(
                    f"Event count mismatch: raw={raw_events}, npz={npz_events} "
                    f"({pct_diff:.1f}% diff)"
                )
            else:
                result["issues"].append(
                    f"Minor event count diff: raw={raw_events}, npz={npz_events} "
                    f"({pct_diff:.2f}% diff, likely filtering)"
                )

        # Compare event type distribution
        if "action" in df.columns:
            raw_action_counts = df["action"].value_counts().to_dict()
            result["raw_stats"]["action_dist"] = raw_action_counts

        # Compare side distribution
        if "side" in df.columns:
            raw_side_counts = df["side"].value_counts().to_dict()
            result["raw_stats"]["side_dist"] = raw_side_counts

        # Compare timestamp range
        if "ts_event" in df.columns and len(df) > 0:
            df_sorted = df.sort_values("ts_event")
            raw_ts_start = int(df_sorted["ts_event"].iloc[0].value)
            raw_ts_end = int(df_sorted["ts_event"].iloc[-1].value)
            result["raw_stats"]["ts_range_ns"] = (raw_ts_start, raw_ts_end)

            npz_ts_range = result["npz_stats"].get("ts_range_ns")
            if npz_ts_range:
                if abs(raw_ts_start - npz_ts_range[0]) > 1_000_000:  # 1ms tolerance
                    result["issues"].append(
                        f"Start timestamp mismatch: raw={raw_ts_start}, npz={npz_ts_range[0]}"
                    )
                if abs(raw_ts_end - npz_ts_range[1]) > 1_000_000:
                    result["issues"].append(
                        f"End timestamp mismatch: raw={raw_ts_end}, npz={npz_ts_range[1]}"
                    )

        # If no critical issues, mark as match
        critical = [i for i in result["issues"] if "mismatch" in i.lower() and "minor" not in i.lower()]
        result["match"] = len(critical) == 0

    except ImportError:
        result["issues"].append("databento library not installed — cannot compare raw")
    except Exception as e:
        result["issues"].append(f"Error loading raw file: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare downloaded vs recorded MBO data")
    parser.add_argument("--dates", type=str, help="Comma-separated dates to check")
    parser.add_argument("--sample", type=int, help="Random sample of N dates")
    parser.add_argument("--all", action="store_true", help="Check all dates")
    parser.add_argument("--schema-only", action="store_true", help="Only validate NPZ schema (no raw comparison)")
    args = parser.parse_args()

    # Discover all dates
    raw_dates = set()
    for f in RAW_DIR.glob("*.dbn.zst"):
        m = re.search(r"(\d{8})", f.name)
        if m:
            raw_dates.add(m.group(1))

    npz_dates = set()
    for f in NPZ_DIR.glob("*_mbo_events.npz"):
        npz_dates.add(f.stem.split("_")[0])

    all_dates = sorted(raw_dates | npz_dates)

    logger.info(f"Found {len(raw_dates)} raw .dbn.zst files")
    logger.info(f"Found {len(npz_dates)} processed NPZ files")
    logger.info(f"Overlap (both exist): {len(raw_dates & npz_dates)} dates")
    logger.info(f"NPZ-only (live recorded): {len(npz_dates - raw_dates)} dates")
    logger.info(f"Raw-only (unprocessed): {len(raw_dates - npz_dates)} dates")

    # Select dates to check
    if args.dates:
        check_dates = args.dates.split(",")
    elif args.sample:
        overlap = sorted(raw_dates & npz_dates)
        check_dates = list(np.random.choice(overlap, min(args.sample, len(overlap)), replace=False))
    elif args.all:
        check_dates = all_dates
    else:
        # Default: check 5 random overlapping dates + all NPZ-only dates
        overlap = sorted(raw_dates & npz_dates)
        sample_overlap = list(np.random.choice(overlap, min(5, len(overlap)), replace=False))
        npz_only = sorted(npz_dates - raw_dates)
        check_dates = sample_overlap + npz_only

    logger.info(f"\nChecking {len(check_dates)} dates...\n")

    results = []
    n_match = 0
    n_issues = 0
    n_schema_ok = 0

    for i, date in enumerate(sorted(check_dates)):
        logger.info(f"[{i+1}/{len(check_dates)}] Processing {date}...")
        t0 = time.time()

        if args.schema_only:
            npz_path = NPZ_DIR / f"{date}_mbo_events.npz"
            if npz_path.exists():
                val = validate_npz_schema(npz_path)
                r = {"date": date, "has_raw": (RAW_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst").exists(),
                     "has_npz": True, "match": val["valid"], "issues": val["issues"],
                     "npz_stats": val["stats"]}
            else:
                r = {"date": date, "has_raw": False, "has_npz": False, "match": False,
                     "issues": ["NPZ not found"]}
        else:
            r = compare_raw_vs_npz(date)

        elapsed = time.time() - t0
        results.append(r)

        status = "✅ MATCH" if r["match"] else "❌ ISSUES"
        n_events = r.get("npz_stats", {}).get("n_events", "?")
        logger.info(f"  {status} | {n_events} events | {elapsed:.1f}s")
        if r["issues"]:
            for issue in r["issues"]:
                logger.info(f"    ⚠️  {issue}")

        if r["match"]:
            n_match += 1
        if r["issues"]:
            n_issues += 1
        if r.get("npz_stats", {}).get("n_events", 0) > 0:
            n_schema_ok += 1

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total dates checked:  {len(results)}")
    logger.info(f"Clean matches:        {n_match}")
    logger.info(f"Dates with issues:    {n_issues}")
    logger.info(f"Valid NPZ schema:     {n_schema_ok}")

    # Categorize
    raw_only = [r for r in results if r["has_raw"] and not r["has_npz"]]
    npz_only = [r for r in results if not r["has_raw"] and r["has_npz"]]
    both = [r for r in results if r["has_raw"] and r["has_npz"]]

    if both:
        logger.info(f"\nDownloaded + NPZ (comparison possible): {len(both)}")
        for r in both:
            status = "✅" if r["match"] else "❌"
            logger.info(f"  {status} {r['date']} — {r.get('npz_stats', {}).get('n_events', '?')} events")

    if npz_only:
        logger.info(f"\nNPZ-only (live recorded, schema validation only): {len(npz_only)}")
        for r in npz_only:
            status = "✅" if r["match"] else "❌"
            logger.info(f"  {status} {r['date']} — {r.get('npz_stats', {}).get('n_events', '?')} events")

    if raw_only:
        logger.info(f"\nRaw-only (not yet processed!): {len(raw_only)}")
        for r in raw_only:
            logger.info(f"  ⚠️  {r['date']}")

    # Save detailed results
    out_path = ROOT / "output" / "data_comparison_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def to_json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_json_safe(i) for i in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(to_json_safe(results), f, indent=2, default=str)
    logger.info(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
