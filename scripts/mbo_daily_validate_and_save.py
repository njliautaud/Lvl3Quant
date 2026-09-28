#!/usr/bin/env python3
"""
mbo_daily_validate_and_save.py — Automated MBO data validation + save pipeline.

Fetches the daily MBO NPZ file from Razer (live recorder), validates it against
the canonical schema, checks for gaps, and promotes it to the permanent data repo.

Usage:
    python3 scripts/mbo_daily_validate_and_save.py                  # today (UTC)
    python3 scripts/mbo_daily_validate_and_save.py --date 20260429  # specific date
    python3 scripts/mbo_daily_validate_and_save.py --local /path/to/file.npz  # skip fetch
    python3 scripts/mbo_daily_validate_and_save.py --dry-run        # validate only, don't move
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────

RAZER_HOST = "razer"
RAZER_USER = "claude"
RAZER_PASS = os.environ.get("CLUSTER_SSH_PASSWORD", "")
RAZER_MBO_DIR = r"C:\Users\claude\Lvl3Quant\data\processed\mbo_events"

STAGING_DIR = Path("/tmp/mbo_staging")
DEST_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")

# Canonical NPZ schema
REQUIRED_KEYS = {"events", "labels_1s", "labels_5s", "labels_10s", "labels_30s",
                 "timestamps", "metadata"}
EXPECTED_DTYPES = {
    "events": np.float32,
    "labels_1s": np.float32,
    "labels_5s": np.float32,
    "labels_10s": np.float32,
    "labels_30s": np.float32,
    "timestamps": np.int64,
}
EVENTS_NCOLS = 6
MIN_EVENTS_FULL_DAY = 500_000
MAX_GAP_NS = 2 * 60 * 1_000_000_000  # 2 minutes in nanoseconds

# RTH hours in Eastern Time (9:30 AM - 4:00 PM ET)
# We use a loose UTC window that covers both EST and EDT:
#   EDT (summer): RTH = 13:30-20:00 UTC
#   EST (winter): RTH = 14:30-21:00 UTC
# Conservative: use the wider window 13:30-21:00 UTC
RTH_START_UTC_H, RTH_START_UTC_M = 13, 30
RTH_END_UTC_H, RTH_END_UTC_M = 21, 0


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def ok(msg: str) -> str:
    return f"  {Colors.GREEN}[PASS]{Colors.RESET} {msg}"


def fail(msg: str) -> str:
    return f"  {Colors.RED}[FAIL]{Colors.RESET} {msg}"


def warn(msg: str) -> str:
    return f"  {Colors.YELLOW}[WARN]{Colors.RESET} {msg}"


def info(msg: str) -> str:
    return f"  {Colors.CYAN}[INFO]{Colors.RESET} {msg}"


def ns_to_utc(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)


def format_ns(ts_ns: int) -> str:
    return ns_to_utc(ts_ns).strftime("%Y-%m-%d %H:%M:%S.%f UTC")


# ──────────────────────────────────────────────
#  STEP 1: FETCH FROM RAZER
# ──────────────────────────────────────────────

def fetch_from_razer(date_str: str) -> Path:
    """SCP the daily NPZ file from Razer to local staging directory."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{date_str}_mbo_events.npz"
    remote_path = f"{RAZER_MBO_DIR}\\{filename}"
    local_path = STAGING_DIR / filename

    print(f"\n{Colors.BOLD}=== STEP 1: Fetch from Razer ==={Colors.RESET}")
    print(info(f"Remote: {RAZER_USER}@{RAZER_HOST}:{remote_path}"))
    print(info(f"Local:  {local_path}"))

    # First check if file exists on Razer
    check_cmd = [
        "sshpass", "-p", RAZER_PASS,
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        f"{RAZER_USER}@{RAZER_HOST}",
        f'if exist "{remote_path}" (echo EXISTS) else (echo MISSING)',
    ]
    try:
        result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout.strip()
        if "MISSING" in output:
            print(fail(f"File not found on Razer: {remote_path}"))
            # Try to list what's there
            list_cmd = [
                "sshpass", "-p", RAZER_PASS,
                "ssh", "-o", "StrictHostKeyChecking=no",
                f"{RAZER_USER}@{RAZER_HOST}",
                f'dir "{RAZER_MBO_DIR}\\*mbo*" /b 2>nul || echo NO_FILES',
            ]
            lr = subprocess.run(list_cmd, capture_output=True, text=True, timeout=30)
            if lr.stdout.strip() and "NO_FILES" not in lr.stdout:
                print(info(f"Available files on Razer:\n{lr.stdout.strip()}"))
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(fail("SSH to Razer timed out (10s). Is Razer online?"))
        sys.exit(1)
    except Exception as e:
        print(warn(f"Could not check remote file existence: {e}"))
        print(info("Proceeding with SCP anyway..."))

    # SCP the file
    scp_cmd = [
        "sshpass", "-p", RAZER_PASS,
        "scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        f"{RAZER_USER}@{RAZER_HOST}:{remote_path}",
        str(local_path),
    ]
    try:
        result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            # SCP with Windows paths can be tricky — try forward slashes
            remote_path_fwd = remote_path.replace("\\", "/")
            scp_cmd[-2] = f"{RAZER_USER}@{RAZER_HOST}:'{remote_path_fwd}'"
            result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(fail(f"SCP failed: {result.stderr.strip()}"))
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print(fail("SCP transfer timed out (5 min)"))
        sys.exit(1)

    if not local_path.exists() or local_path.stat().st_size == 0:
        print(fail("File was not transferred or is empty"))
        sys.exit(1)

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(ok(f"Fetched {filename} ({size_mb:.1f} MB)"))
    return local_path


# ──────────────────────────────────────────────
#  STEP 2: VALIDATE
# ──────────────────────────────────────────────

def validate_npz(npz_path: Path, date_str: str) -> tuple[bool, dict]:
    """
    Validate an MBO NPZ file against the canonical schema.
    Returns (all_passed, stats_dict).
    """
    print(f"\n{Colors.BOLD}=== STEP 2: Validate ==={Colors.RESET}")
    print(info(f"File: {npz_path}"))

    errors = []
    warnings = []
    stats = {}

    # --- Load ---
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(fail(f"Cannot load NPZ: {e}"))
        return False, {"error": str(e)}

    # --- Check keys ---
    keys = set(data.keys())
    missing = REQUIRED_KEYS - keys
    extra = keys - REQUIRED_KEYS
    if missing:
        errors.append(f"Missing keys: {missing}")
        print(fail(f"Missing keys: {missing}"))
    else:
        print(ok(f"All required keys present: {sorted(REQUIRED_KEYS)}"))
    if extra:
        print(warn(f"Extra keys (ignored): {extra}"))

    if missing:
        return False, {"errors": errors}

    events = data["events"]
    timestamps = data["timestamps"]

    # --- Check filename pattern ---
    expected_name = f"{date_str}_mbo_events.npz"
    actual_name = npz_path.name
    if actual_name == expected_name:
        print(ok(f"Filename matches expected: {expected_name}"))
    else:
        errors.append(f"Filename mismatch: got {actual_name}, expected {expected_name}")
        print(fail(f"Filename mismatch: got {actual_name}, expected {expected_name}"))

    # --- Check dtypes ---
    dtype_ok = True
    for key, expected_dtype in EXPECTED_DTYPES.items():
        actual_dtype = data[key].dtype
        if actual_dtype != expected_dtype:
            errors.append(f"{key}: dtype={actual_dtype}, expected {expected_dtype}")
            print(fail(f"{key}: dtype={actual_dtype}, expected {expected_dtype}"))
            dtype_ok = False
    if dtype_ok:
        print(ok(f"All dtypes match canonical schema"))

    # --- Check events shape ---
    if events.ndim == 2 and events.shape[1] == EVENTS_NCOLS:
        print(ok(f"Events shape: {events.shape} (N x {EVENTS_NCOLS})"))
    else:
        errors.append(f"Events shape: {events.shape}, expected (N, {EVENTS_NCOLS})")
        print(fail(f"Events shape: {events.shape}, expected (N, {EVENTS_NCOLS})"))

    n_events = events.shape[0]
    stats["n_events"] = n_events

    # --- Check label array lengths match ---
    label_keys = ["labels_1s", "labels_5s", "labels_10s", "labels_30s"]
    lengths_ok = True
    for lk in label_keys:
        if data[lk].shape[0] != n_events:
            errors.append(f"{lk} length {data[lk].shape[0]} != events length {n_events}")
            print(fail(f"{lk} length mismatch"))
            lengths_ok = False
    if lengths_ok:
        print(ok(f"All label arrays match event count ({n_events:,})"))

    # --- Check timestamps length ---
    if timestamps.shape[0] != n_events:
        errors.append(f"Timestamps length {timestamps.shape[0]} != events {n_events}")
        print(fail(f"Timestamps length mismatch"))
    else:
        print(ok(f"Timestamps length matches ({n_events:,})"))

    # --- Check event count ---
    if n_events >= MIN_EVENTS_FULL_DAY:
        print(ok(f"Event count: {n_events:,} (>= {MIN_EVENTS_FULL_DAY:,} threshold)"))
    else:
        errors.append(f"Only {n_events:,} events (< {MIN_EVENTS_FULL_DAY:,} for full day)")
        print(fail(f"Only {n_events:,} events (< {MIN_EVENTS_FULL_DAY:,} minimum for full day)"))

    # --- Timestamps: range ---
    ts_min, ts_max = int(timestamps.min()), int(timestamps.max())
    stats["ts_min"] = ts_min
    stats["ts_max"] = ts_max
    stats["time_start"] = format_ns(ts_min)
    stats["time_end"] = format_ns(ts_max)
    duration_hours = (ts_max - ts_min) / 3.6e12
    stats["duration_hours"] = round(duration_hours, 2)
    print(info(f"Time range: {format_ns(ts_min)} -> {format_ns(ts_max)}"))
    print(info(f"Duration: {duration_hours:.2f} hours"))

    # --- Timestamps: monotonically increasing ---
    if n_events > 1:
        diffs = np.diff(timestamps)
        n_decreasing = int((diffs < 0).sum())
        if n_decreasing == 0:
            print(ok("Timestamps are monotonically non-decreasing"))
        else:
            errors.append(f"Timestamps not monotonic: {n_decreasing:,} decreasing steps")
            print(fail(f"Timestamps not monotonic: {n_decreasing:,} decreasing steps"))
        stats["n_decreasing_ts"] = n_decreasing

    # --- Timestamps: date check ---
    ts_date = ns_to_utc(ts_min).strftime("%Y%m%d")
    ts_date_end = ns_to_utc(ts_max).strftime("%Y%m%d")
    # For live recorder, data is saved by UTC date. The file date should
    # match the UTC date of the timestamps (or span midnight by a small margin).
    if date_str == ts_date or date_str == ts_date_end:
        print(ok(f"Timestamps date matches file date ({date_str})"))
    else:
        # Allow for recorder files that span UTC midnight (market hours cross midnight)
        warnings.append(f"Timestamp dates ({ts_date}-{ts_date_end}) don't exactly match file date ({date_str})")
        print(warn(f"Timestamp dates ({ts_date}-{ts_date_end}) vs file date ({date_str}) — may span midnight"))

    # --- RTH gap analysis ---
    print(f"\n  {Colors.BOLD}RTH Gap Analysis (09:30-16:00 ET / ~13:30-21:00 UTC):{Colors.RESET}")

    # Compute RTH boundaries for the target date
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    rth_start_utc = datetime(year, month, day, RTH_START_UTC_H, RTH_START_UTC_M,
                             tzinfo=timezone.utc)
    rth_end_utc = datetime(year, month, day, RTH_END_UTC_H, RTH_END_UTC_M,
                           tzinfo=timezone.utc)
    rth_start_ns = int(rth_start_utc.timestamp() * 1e9)
    rth_end_ns = int(rth_end_utc.timestamp() * 1e9)

    # Filter to RTH timestamps
    rth_mask = (timestamps >= rth_start_ns) & (timestamps <= rth_end_ns)
    rth_ts = timestamps[rth_mask]
    n_rth = len(rth_ts)
    stats["n_rth_events"] = n_rth
    print(info(f"RTH events: {n_rth:,} / {n_events:,} total"))

    if n_rth < 2:
        if n_events > 0:
            warnings.append("No RTH events found — may be pre/post market only or wrong date")
            print(warn("No RTH events found — cannot check for gaps"))
        stats["rth_gaps"] = []
    else:
        rth_diffs = np.diff(rth_ts)
        gap_mask = rth_diffs > MAX_GAP_NS
        n_gaps = int(gap_mask.sum())
        stats["n_rth_gaps_over_2min"] = n_gaps

        if n_gaps == 0:
            print(ok(f"No gaps > 2 minutes during RTH"))
        else:
            errors.append(f"{n_gaps} gaps > 2 minutes during RTH")
            print(fail(f"{n_gaps} gap(s) > 2 minutes during RTH:"))

        # Report top gaps regardless
        gap_indices = np.where(gap_mask)[0] if n_gaps > 0 else np.array([], dtype=int)
        gap_details = []
        for idx in gap_indices[:10]:  # Show up to 10 largest
            gap_ns = int(rth_diffs[idx])
            gap_start = int(rth_ts[idx])
            gap_end = int(rth_ts[idx + 1])
            gap_sec = gap_ns / 1e9
            gap_details.append({
                "start": format_ns(gap_start),
                "end": format_ns(gap_end),
                "duration_sec": round(gap_sec, 1),
            })
            print(f"    Gap: {format_ns(gap_start)} -> {format_ns(gap_end)} ({gap_sec:.1f}s)")
        stats["rth_gaps"] = gap_details

        # Also show largest gap even if under threshold
        if n_gaps == 0 and len(rth_diffs) > 0:
            max_gap_sec = float(rth_diffs.max()) / 1e9
            stats["max_rth_gap_sec"] = round(max_gap_sec, 1)
            print(info(f"Largest RTH gap: {max_gap_sec:.1f}s"))

    # --- Metadata check ---
    try:
        meta_arr = data["metadata"]
        meta_str = str(meta_arr[0]) if meta_arr.size > 0 else ""
        meta = json.loads(meta_str)
        stats["metadata"] = meta
        print(ok(f"Metadata is valid JSON with {len(meta)} fields"))
        if "feature_names" in meta:
            expected_features = ["time_delta_log", "event_type_id", "side_id",
                                 "price_rel_ticks", "qty_log", "spread_ticks"]
            if meta["feature_names"] == expected_features:
                print(ok("Feature names match canonical spec"))
            else:
                warnings.append(f"Feature names differ: {meta['feature_names']}")
                print(warn(f"Feature names differ from canonical"))
    except Exception as e:
        warnings.append(f"Metadata parse error: {e}")
        print(warn(f"Could not parse metadata: {e}"))

    # --- Events sanity: check value ranges ---
    if events.shape[0] > 0 and events.shape[1] == EVENTS_NCOLS:
        # event_type_id should be in {0,1,2,3,4}
        etypes = np.unique(events[:, 1])
        valid_etypes = {0.0, 1.0, 2.0, 3.0, 4.0}
        if set(etypes.tolist()).issubset(valid_etypes):
            print(ok(f"Event types valid: {sorted(etypes.tolist())}"))
        else:
            bad = set(etypes.tolist()) - valid_etypes
            warnings.append(f"Unexpected event types: {bad}")
            print(warn(f"Unexpected event types: {bad}"))

        # side_id should be in {0,1,2}
        sides = np.unique(events[:, 2])
        valid_sides = {0.0, 1.0, 2.0}
        if set(sides.tolist()).issubset(valid_sides):
            print(ok(f"Side IDs valid: {sorted(sides.tolist())}"))
        else:
            bad = set(sides.tolist()) - valid_sides
            warnings.append(f"Unexpected side IDs: {bad}")
            print(warn(f"Unexpected side IDs: {bad}"))

        # price_rel_ticks should be clipped to [-50, 50]
        price_min = float(events[:, 3].min())
        price_max = float(events[:, 3].max())
        stats["price_rel_range"] = (round(price_min, 2), round(price_max, 2))
        if -50.0 <= price_min and price_max <= 50.0:
            print(ok(f"Price rel ticks in range [{price_min:.1f}, {price_max:.1f}]"))
        else:
            warnings.append(f"Price rel ticks outside +-50: [{price_min:.1f}, {price_max:.1f}]")
            print(warn(f"Price rel ticks outside +-50: [{price_min:.1f}, {price_max:.1f}]"))

    # --- NaN check on labels (live recorder uses NaN, pipeline uses real labels) ---
    for lk in label_keys:
        n_nan = int(np.isnan(data[lk]).sum())
        n_total = data[lk].shape[0]
        stats[f"{lk}_nan_pct"] = round(100.0 * n_nan / max(1, n_total), 1)
        if n_nan == n_total:
            print(info(f"{lk}: all NaN (live recorder — labels not yet computed)"))
        elif n_nan == 0:
            print(ok(f"{lk}: no NaN values (labels computed)"))
        else:
            pct = 100.0 * n_nan / n_total
            print(info(f"{lk}: {pct:.1f}% NaN ({n_nan:,}/{n_total:,})"))

    # --- Summary ---
    all_passed = len(errors) == 0
    stats["errors"] = errors
    stats["warnings"] = warnings
    stats["passed"] = all_passed

    return all_passed, stats


# ──────────────────────────────────────────────
#  STEP 3: PROMOTE TO REPO
# ──────────────────────────────────────────────

def promote_to_repo(staged_path: Path, date_str: str, dry_run: bool = False) -> Path:
    """Move validated file from staging to permanent data repo."""
    print(f"\n{Colors.BOLD}=== STEP 3: Promote to Data Repo ==={Colors.RESET}")

    dest_path = DEST_DIR / f"{date_str}_mbo_events.npz"

    if dest_path.exists():
        existing_size = dest_path.stat().st_size
        new_size = staged_path.stat().st_size
        if new_size > existing_size:
            print(warn(f"Overwriting existing file ({existing_size:,} -> {new_size:,} bytes)"))
        else:
            print(warn(f"Existing file is same/larger ({existing_size:,} bytes vs new {new_size:,} bytes)"))
            print(warn("Overwriting anyway — new file passed validation"))

    if dry_run:
        print(info(f"DRY RUN: Would move {staged_path} -> {dest_path}"))
        return dest_path

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(staged_path), str(dest_path))
    # Verify the copy
    if dest_path.exists() and dest_path.stat().st_size == staged_path.stat().st_size:
        # Remove staging copy
        staged_path.unlink()
        print(ok(f"Promoted to: {dest_path}"))
        print(ok(f"Staging file cleaned up"))
    else:
        print(fail("Copy verification failed!"))
        sys.exit(1)

    return dest_path


# ──────────────────────────────────────────────
#  SUMMARY
# ──────────────────────────────────────────────

def print_summary(date_str: str, passed: bool, stats: dict, dest_path: Path | None):
    """Print final PASS/FAIL summary."""
    print(f"\n{'=' * 60}")
    if passed:
        print(f"  {Colors.BOLD}{Colors.GREEN}PASS{Colors.RESET} — {date_str}_mbo_events.npz")
    else:
        print(f"  {Colors.BOLD}{Colors.RED}FAIL{Colors.RESET} — {date_str}_mbo_events.npz")
    print(f"{'=' * 60}")

    print(f"  Events:        {stats.get('n_events', '?'):>12,}")
    print(f"  RTH events:    {stats.get('n_rth_events', '?'):>12,}")
    print(f"  Time start:    {stats.get('time_start', '?')}")
    print(f"  Time end:      {stats.get('time_end', '?')}")
    print(f"  Duration:      {stats.get('duration_hours', '?')} hours")
    print(f"  RTH gaps >2m:  {stats.get('n_rth_gaps_over_2min', '?')}")
    if stats.get("max_rth_gap_sec"):
        print(f"  Max RTH gap:   {stats['max_rth_gap_sec']}s")
    print(f"  Errors:        {len(stats.get('errors', []))}")
    print(f"  Warnings:      {len(stats.get('warnings', []))}")
    if dest_path:
        print(f"  Saved to:      {dest_path}")
    if stats.get("errors"):
        print(f"\n  Errors:")
        for e in stats["errors"]:
            print(f"    - {e}")
    if stats.get("warnings"):
        print(f"\n  Warnings:")
        for w in stats["warnings"]:
            print(f"    - {w}")
    print()


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate and save daily MBO event data from Razer recorder"
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Date string YYYYMMDD (default: today UTC)")
    parser.add_argument("--local", type=str, default=None,
                        help="Path to a local NPZ file (skip fetch from Razer)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate only, don't move to data repo")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Assume file is already in staging dir")
    args = parser.parse_args()

    # Determine date
    if args.date:
        date_str = args.date
        # Validate date format
        try:
            datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(fail(f"Invalid date format: {date_str} (expected YYYYMMDD)"))
            sys.exit(1)
    else:
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    print(f"{Colors.BOLD}MBO Daily Validate & Save Pipeline{Colors.RESET}")
    print(f"Date: {date_str}")
    print(f"Dest: {DEST_DIR}")

    # Step 1: Get the file
    if args.local:
        npz_path = Path(args.local)
        if not npz_path.exists():
            print(fail(f"Local file not found: {npz_path}"))
            sys.exit(1)
        print(info(f"Using local file: {npz_path}"))
    elif args.skip_fetch:
        npz_path = STAGING_DIR / f"{date_str}_mbo_events.npz"
        if not npz_path.exists():
            print(fail(f"Expected staged file not found: {npz_path}"))
            sys.exit(1)
        print(info(f"Using staged file: {npz_path}"))
    else:
        npz_path = fetch_from_razer(date_str)

    # Step 2: Validate
    passed, stats = validate_npz(npz_path, date_str)

    # Step 3: Promote if passed
    dest_path = None
    if passed and not args.dry_run:
        dest_path = promote_to_repo(npz_path, date_str)
    elif passed and args.dry_run:
        print(info("DRY RUN: Skipping promotion"))
    else:
        print(fail("Validation FAILED — file NOT promoted"))

    # Summary
    print_summary(date_str, passed, stats, dest_path)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
