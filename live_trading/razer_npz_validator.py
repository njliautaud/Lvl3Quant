#!/usr/bin/env python3
"""razer_npz_validator.py — Daily NPZ file validator for Razer MBO recordings.

Validates the daily MBO events NPZ file for:
  - Expected keys and dtypes
  - Timestamp monotonicity
  - Event count thresholds
  - NaN/Inf detection in events array
  - Feature dimension consistency
  - Metadata integrity

Usage:
  python razer_npz_validator.py                  # validate today's file (UTC)
  python razer_npz_validator.py --date 20260429  # validate specific date
  python razer_npz_validator.py --path /path/to/file.npz  # validate specific file

Exit codes: 0 = pass, 1 = fail
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\claude\Lvl3Quant\data\processed\mbo_events")
LOG_DIR = Path(r"C:\Users\claude\Lvl3Quant\live_trading\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Expected NPZ array keys
REQUIRED_KEYS = {"events", "timestamps", "metadata"}
LABEL_KEYS = {"labels_1s", "labels_5s", "labels_10s", "labels_30s"}
ALL_EXPECTED_KEYS = REQUIRED_KEYS | LABEL_KEYS

# Expected dtypes
EXPECTED_DTYPES = {
    "events": np.float32,
    "timestamps": np.int64,
    "labels_1s": np.float32,
    "labels_5s": np.float32,
    "labels_10s": np.float32,
    "labels_30s": np.float32,
}

# Expected event feature names (6 columns)
EXPECTED_FEATURES = [
    "time_delta_log", "event_type_id", "side_id",
    "price_rel_ticks", "qty_log", "spread_ticks",
]
EXPECTED_N_FEATURES = len(EXPECTED_FEATURES)

# Minimum events for a full trading session (~6.5 hrs for NQ)
MIN_EVENTS_FULL_DAY = 50_000
# Warning threshold for low event count (partial day / early close)
MIN_EVENTS_WARNING = 10_000

# Metadata expected keys
EXPECTED_META_KEYS = {"date", "instrument_id", "tick_size", "n_events",
                      "feature_names", "action_encoding", "side_encoding"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ValidationResult:
    """Accumulates pass/fail checks with messages."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.checks: list[dict] = []
        self.passed = True
        self.warnings: list[str] = []

    def check(self, name: str, ok: bool, detail: str = ""):
        status = "PASS" if ok else "FAIL"
        self.checks.append({"check": name, "status": status, "detail": detail})
        if not ok:
            self.passed = False

    def warn(self, msg: str):
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {
            "file": self.file_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall": "PASS" if self.passed else "FAIL",
            "checks": self.checks,
            "warnings": self.warnings,
        }


def validate_npz(npz_path: Path) -> ValidationResult:
    """Run all validation checks on an NPZ file."""
    result = ValidationResult(str(npz_path))

    # --- File existence ---
    result.check("file_exists", npz_path.exists(),
                 f"Path: {npz_path}")
    if not npz_path.exists():
        return result

    # --- Load file ---
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        result.check("file_readable", False, f"Failed to load: {exc}")
        return result
    result.check("file_readable", True, f"Size: {npz_path.stat().st_size:,} bytes")

    # --- Key checks ---
    actual_keys = set(data.files)
    missing_required = REQUIRED_KEYS - actual_keys
    result.check("required_keys_present",
                 len(missing_required) == 0,
                 f"Missing: {missing_required}" if missing_required
                 else f"Keys: {sorted(actual_keys)}")

    missing_labels = LABEL_KEYS - actual_keys
    if missing_labels:
        result.warn(f"Missing label keys (OK for live data): {missing_labels}")

    unexpected = actual_keys - ALL_EXPECTED_KEYS
    if unexpected:
        result.warn(f"Unexpected keys (not necessarily wrong): {unexpected}")

    if missing_required:
        return result  # Can't continue without required keys

    # --- Events array ---
    events = data["events"]
    result.check("events_dtype", events.dtype == np.float32,
                 f"Expected float32, got {events.dtype}")
    result.check("events_2d", events.ndim == 2,
                 f"Expected 2D, got {events.ndim}D shape={events.shape}")
    if events.ndim == 2:
        result.check("events_n_features",
                     events.shape[1] == EXPECTED_N_FEATURES,
                     f"Expected {EXPECTED_N_FEATURES} features, got {events.shape[1]}")

    n_events = events.shape[0]
    result.check("events_count_minimum",
                 n_events >= MIN_EVENTS_WARNING,
                 f"Event count: {n_events:,} (min warning: {MIN_EVENTS_WARNING:,})")
    if n_events < MIN_EVENTS_FULL_DAY:
        result.warn(f"Event count {n_events:,} below full-day threshold "
                    f"({MIN_EVENTS_FULL_DAY:,}). Partial day or early close?")

    # --- NaN / Inf in events ---
    nan_count = int(np.isnan(events).sum())
    inf_count = int(np.isinf(events).sum())
    result.check("events_no_nan", nan_count == 0,
                 f"NaN count: {nan_count}")
    result.check("events_no_inf", inf_count == 0,
                 f"Inf count: {inf_count}")

    # --- Timestamps ---
    timestamps = data["timestamps"]
    result.check("timestamps_dtype", timestamps.dtype == np.int64,
                 f"Expected int64, got {timestamps.dtype}")
    result.check("timestamps_length", len(timestamps) == n_events,
                 f"Timestamps: {len(timestamps)}, Events: {n_events}")

    if len(timestamps) > 1:
        diffs = np.diff(timestamps)
        non_monotonic = int((diffs < 0).sum())
        result.check("timestamps_monotonic", non_monotonic == 0,
                     f"Non-monotonic transitions: {non_monotonic}")

        if non_monotonic > 0:
            # Find first violation for debugging
            first_idx = int(np.argmax(diffs < 0))
            result.warn(f"First monotonicity violation at index {first_idx}: "
                        f"ts[{first_idx}]={timestamps[first_idx]}, "
                        f"ts[{first_idx+1}]={timestamps[first_idx+1]}")

        # Sanity: timestamps should be nanoseconds (2026 range)
        # 2026-01-01 in ns ~ 1.767e18
        ts_min, ts_max = int(timestamps[0]), int(timestamps[-1])
        reasonable_range = 1_700_000_000_000_000_000 < ts_min < 2_000_000_000_000_000_000
        result.check("timestamps_nanosecond_range", reasonable_range,
                     f"Range: {ts_min} - {ts_max}")

    # --- Label arrays (if present) ---
    for lbl_key in LABEL_KEYS:
        if lbl_key in actual_keys:
            lbl = data[lbl_key]
            result.check(f"{lbl_key}_dtype", lbl.dtype == np.float32,
                         f"Expected float32, got {lbl.dtype}")
            result.check(f"{lbl_key}_length", len(lbl) == n_events,
                         f"Labels: {len(lbl)}, Events: {n_events}")

    # --- Metadata ---
    meta_arr = data["metadata"]
    try:
        meta = json.loads(str(meta_arr[0]) if meta_arr.ndim > 0 else str(meta_arr))
        result.check("metadata_parseable", True, "JSON metadata parsed OK")

        missing_meta = EXPECTED_META_KEYS - set(meta.keys())
        result.check("metadata_keys",
                     len(missing_meta) == 0,
                     f"Missing: {missing_meta}" if missing_meta
                     else f"Keys: {sorted(meta.keys())}")

        # Check feature_names match expected
        if "feature_names" in meta:
            result.check("metadata_feature_names",
                         meta["feature_names"] == EXPECTED_FEATURES,
                         f"Got: {meta['feature_names']}")

        # Check n_events matches actual
        if "n_events" in meta:
            meta_n = meta["n_events"]
            result.check("metadata_n_events_consistent",
                         meta_n == n_events,
                         f"Metadata says {meta_n}, actual {n_events}")

    except (json.JSONDecodeError, IndexError, TypeError) as exc:
        result.check("metadata_parseable", False, f"Failed: {exc}")

    data.close()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate daily MBO events NPZ file")
    parser.add_argument("--date", type=str, default=None,
                        help="Date string YYYYMMDD (default: today UTC)")
    parser.add_argument("--path", type=str, default=None,
                        help="Direct path to NPZ file (overrides --date)")
    args = parser.parse_args()

    if args.path:
        npz_path = Path(args.path)
        date_str = npz_path.stem.split("_")[0]
    else:
        date_str = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
        npz_path = DATA_DIR / f"{date_str}_mbo_events.npz"

    print(f"Validating: {npz_path}")

    result = validate_npz(npz_path)

    # Write output JSON
    out_path = LOG_DIR / f"npz_validation_{date_str}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"Result written to: {out_path}")

    # Summary
    n_pass = sum(1 for c in result.checks if c["status"] == "PASS")
    n_fail = sum(1 for c in result.checks if c["status"] == "FAIL")
    total = len(result.checks)

    print(f"\n{'='*50}")
    print(f"OVERALL: {'PASS' if result.passed else 'FAIL'}  "
          f"({n_pass}/{total} checks passed, {n_fail} failed)")
    if result.warnings:
        print(f"Warnings: {len(result.warnings)}")
        for w in result.warnings:
            print(f"  - {w}")
    if not result.passed:
        print("\nFailed checks:")
        for c in result.checks:
            if c["status"] == "FAIL":
                print(f"  FAIL: {c['check']} — {c['detail']}")
    print(f"{'='*50}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
