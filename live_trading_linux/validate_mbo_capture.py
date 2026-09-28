#!/usr/bin/env python3
"""validate_mbo_capture.py — Daily integrity validator for live MBO capture.

Per DIRECTIVES.md item 5b, this runs after each market close (cron schedules
it 4:15 PM ET on weekdays) and verifies that the most recent NPZ file in
/home/jupiter/Lvl3Quant/data/processed/mbo_events/ is healthy.

Checks:
  (a) File exists and is loadable (np.load).
  (b) Schema — events, timestamps, labels_{1,5,10,30}s, metadata all present.
  (c) Event count >= threshold (RTH day vs overnight-only).
  (d) Timestamps non-decreasing (monotonic).
  (e) No gap > 60s between consecutive events during 13:30-20:00 UTC RTH window.
  (f) events.shape[1] == 6.
  (g) No NaN/Inf in events; values within physically plausible ranges.
  (h) Append SHA256 + size + event count to mbo_checksums.log.

On failure: writes to validation_errors.log AND appends a structured alert to
monitor_messages.jsonl (which the persistent_monitor / Claude session reads).

On success: appends OK line to mbo_checksums.log and validation_status.log.

Exit codes:
  0  — all checks passed
  1  — one or more checks failed
  2  — could not even find a file to validate (worse — recorder may be down)
  3  — internal error in the validator itself
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths (all absolute — this script is invoked by cron with no PWD guarantees)
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
LOG_DIR = Path("/home/jupiter/Lvl3Quant/live_trading_linux/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

CHECKSUM_LOG = LOG_DIR / "mbo_checksums.log"
ERROR_LOG = LOG_DIR / "validation_errors.log"
STATUS_LOG = LOG_DIR / "validation_status.log"
MONITOR_JSONL = Path("/home/jupiter/teleclaude-main/monitor_messages.jsonl")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
RTH_DAY_MIN_EVENTS = 100_000      # Normal weekday RTH
OVERNIGHT_MIN_EVENTS = 1_000      # Holiday or Sun-night-only session
RTH_GAP_MAX_SECONDS = 60.0        # Gap longer than this inside RTH = outage
EXPECTED_NUM_COLS = 6

# Timestamp monotonicity tolerance.  Empirically, even healthy Rithmic feeds
# show a small number of ~1 microsecond "rewinds" between consecutive events
# (likely from same-nanosecond ordering at the wire level).  A genuine
# ordering bug shows up as either (a) a backwards step >= 1 ms, or (b) a
# large fraction of events out of order.  The thresholds below were chosen
# after auditing several known-good historical days.
TS_BACKWARDS_HARD_NS = 1_000_000          # 1 ms — any single jump >= this fails
TS_BACKWARDS_FRACTION_MAX = 0.001         # > 0.1% of events out of order fails
PRICE_REL_TICKS_ABS_MAX = 1e6     # Sanity bound on price_rel_ticks
QTY_LOG_MAX = 30.0                # log(max sane qty) — log(1e13) ~ 30
TIME_DELTA_LOG_MAX = 30.0         # log1p(microseconds); 1 day ~ log(8.6e10) ~ 25

REQUIRED_KEYS = (
    "events",
    "timestamps",
    "labels_1s",
    "labels_5s",
    "labels_10s",
    "labels_30s",
    "metadata",
)

# RTH window in UTC: 9:30 ET = 13:30 UTC (EDT) / 14:30 UTC (EST). We use the
# permissive 13:30-20:00 UTC window so the check works year-round during DST.
RTH_START_UTC = dt.time(13, 30)
RTH_END_UTC = dt.time(20, 0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def latest_npz() -> Path | None:
    if not DATA_DIR.is_dir():
        return None
    files = sorted(DATA_DIR.glob("*_mbo_events.npz"))
    return files[-1] if files else None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(line.rstrip("\n") + "\n")


def emit_alert(severity: str, summary: str, detail: dict) -> None:
    """Write to validation_errors.log AND monitor_messages.jsonl.

    The persistent_monitor uses monitor_messages.jsonl as its audit/inbox;
    appending a Claude-targeted entry here ensures the next session-start /
    deep-check picks it up. Severity is 'critical' or 'warning'.
    """
    payload = {
        "ts": now_iso(),
        "source": "mbo_validator",
        "severity": severity,
        "summary": summary,
        "detail": detail,
    }
    json_line = json.dumps(payload, default=str)
    append_line(ERROR_LOG, json_line)
    try:
        append_line(MONITOR_JSONL, json_line)
    except Exception as exc:  # don't fail validation just because alert write failed
        append_line(ERROR_LOG, json.dumps({
            "ts": now_iso(),
            "source": "mbo_validator",
            "severity": "warning",
            "summary": "could not append to monitor_messages.jsonl",
            "detail": {"error": str(exc)},
        }))


def is_full_rth_day(file_date: dt.date) -> bool:
    """Heuristic: weekday and not a known US market holiday => full RTH expected.

    We don't ship a holiday calendar; weekday-only is a reasonable approximation.
    Sunday night sessions and holidays drop into the lower threshold automatically
    via the file's date being a non-weekday.
    """
    return file_date.weekday() < 5  # Mon-Fri


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def run_validation(npz_path: Path) -> tuple[bool, list[str], dict]:
    failures: list[str] = []
    info: dict = {"file": str(npz_path)}

    # (a) loadable
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        failures.append(f"NPZ load failed: {exc!r}")
        return False, failures, info

    # (b) schema keys
    keys = set(data.files)
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        failures.append(f"Missing schema keys: {missing}")

    if "events" not in keys or "timestamps" not in keys:
        # Without these we can't run any further numeric checks
        return False, failures, info

    events = data["events"]
    timestamps = data["timestamps"]
    info["n_events"] = int(events.shape[0])
    info["events_shape"] = list(events.shape)

    # (f) column count
    if events.ndim != 2 or events.shape[1] != EXPECTED_NUM_COLS:
        failures.append(
            f"Unexpected events shape {events.shape}; expected (N, {EXPECTED_NUM_COLS})"
        )

    # (c) event count threshold
    file_date_str = npz_path.stem.split("_")[0]  # "20260428"
    try:
        file_date = dt.datetime.strptime(file_date_str, "%Y%m%d").date()
    except ValueError:
        file_date = None
        failures.append(f"Could not parse date from filename {npz_path.name}")

    is_rth = is_full_rth_day(file_date) if file_date else True
    threshold = RTH_DAY_MIN_EVENTS if is_rth else OVERNIGHT_MIN_EVENTS
    info["expected_min_events"] = threshold
    info["is_rth_day"] = is_rth
    if events.shape[0] < threshold:
        failures.append(
            f"Event count {events.shape[0]} below threshold {threshold} "
            f"({'RTH' if is_rth else 'overnight'} day)"
        )

    # (d) timestamp monotonicity
    if timestamps.ndim != 1 or timestamps.shape[0] != events.shape[0]:
        failures.append(
            f"timestamps shape {timestamps.shape} does not match events {events.shape}"
        )
    else:
        diffs = np.diff(timestamps.astype(np.int64))
        bad = np.where(diffs < 0)[0]
        info["ts_backwards_count"] = int(bad.size)
        if bad.size:
            worst_step = int(diffs[bad].min())  # most negative
            info["ts_worst_backwards_ns"] = worst_step
            # Hard fail: any single rewind >= 1 ms = real ordering bug
            if -worst_step >= TS_BACKWARDS_HARD_NS:
                first_big = int(bad[np.argmin(diffs[bad])])
                failures.append(
                    f"Timestamp rewind >= {TS_BACKWARDS_HARD_NS} ns: "
                    f"worst={worst_step} ns at index {first_big}"
                )
            # Hard fail: too many rewinds in aggregate (recorder/feed broken)
            frac = bad.size / max(1, diffs.size)
            if frac > TS_BACKWARDS_FRACTION_MAX:
                failures.append(
                    f"Timestamp monotonicity violations exceed tolerance: "
                    f"{bad.size}/{diffs.size} ({frac*100:.3f}% > "
                    f"{TS_BACKWARDS_FRACTION_MAX*100:.3f}%)"
                )

        # (e) RTH gap check — only if we have a parsed date and any events
        if file_date is not None and timestamps.size >= 2:
            ts_seconds = timestamps.astype(np.float64) / 1e9
            ts_dt = np.array(
                [dt.datetime.fromtimestamp(s, tz=dt.timezone.utc) for s in ts_seconds]
            )
            in_rth = np.array(
                [RTH_START_UTC <= t.time() < RTH_END_UTC for t in ts_dt]
            )
            rth_idx = np.where(in_rth)[0]
            if rth_idx.size >= 2:
                rth_ts = ts_seconds[rth_idx]
                rth_diffs = np.diff(rth_ts)
                # A gap is "real" only if both endpoints fall in RTH consecutively
                # (we already filtered by in_rth, so consecutive rth_idx values
                # may include gaps where intermediate events were outside RTH).
                # Use only adjacent rth_idx pairs that are themselves adjacent
                # in the original array.
                consecutive = np.where(np.diff(rth_idx) == 1)[0]
                if consecutive.size:
                    big_gaps = []
                    for i in consecutive:
                        gap = rth_diffs[i]
                        if gap > RTH_GAP_MAX_SECONDS:
                            big_gaps.append((int(rth_idx[i]), float(gap)))
                    info["rth_gaps_over_60s"] = len(big_gaps)
                    if big_gaps:
                        worst = max(big_gaps, key=lambda p: p[1])
                        failures.append(
                            f"{len(big_gaps)} RTH gaps > {RTH_GAP_MAX_SECONDS}s "
                            f"(worst: {worst[1]:.1f}s at event index {worst[0]})"
                        )
            info["rth_event_count"] = int(rth_idx.size)

    # (g) NaN/Inf and physical sanity on events
    if events.size:
        if not np.isfinite(events).all():
            n_bad = int((~np.isfinite(events)).sum())
            failures.append(f"events contains {n_bad} non-finite values (NaN/Inf)")
        # Column 0: time_delta_log
        col_td = events[:, 0]
        if col_td.size and (col_td.min() < 0 or col_td.max() > TIME_DELTA_LOG_MAX):
            failures.append(
                f"time_delta_log out of range [0, {TIME_DELTA_LOG_MAX}]: "
                f"min={float(col_td.min())}, max={float(col_td.max())}"
            )
        # Column 3: price_rel_ticks
        if events.shape[1] >= 4:
            col_pr = events[:, 3]
            if col_pr.size and np.abs(col_pr).max() > PRICE_REL_TICKS_ABS_MAX:
                failures.append(
                    f"price_rel_ticks abs max={float(np.abs(col_pr).max())} "
                    f"exceeds {PRICE_REL_TICKS_ABS_MAX}"
                )
        # Column 4: qty_log
        if events.shape[1] >= 5:
            col_q = events[:, 4]
            if col_q.size and (col_q.min() < 0 or col_q.max() > QTY_LOG_MAX):
                failures.append(
                    f"qty_log out of range [0, {QTY_LOG_MAX}]: "
                    f"min={float(col_q.min())}, max={float(col_q.max())}"
                )

    return (len(failures) == 0), failures, info


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    try:
        npz = latest_npz()
        if npz is None:
            emit_alert(
                "critical",
                "MBO validator: no NPZ files found in capture directory",
                {"data_dir": str(DATA_DIR)},
            )
            return 2

        size_bytes = npz.stat().st_size
        digest = sha256_of(npz)
        ok, failures, info = run_validation(npz)
        info["sha256"] = digest
        info["size_bytes"] = size_bytes

        timestamp = now_iso()
        n_events = info.get("n_events", "?")

        if ok:
            line = (
                f"{timestamp} OK file={npz.name} events={n_events} "
                f"size={size_bytes} sha256={digest}"
            )
            append_line(CHECKSUM_LOG, line)
            append_line(STATUS_LOG, line)
            return 0

        # Failure path
        line = (
            f"{timestamp} FAIL file={npz.name} events={n_events} "
            f"size={size_bytes} sha256={digest} failures={len(failures)}"
        )
        append_line(CHECKSUM_LOG, line)
        append_line(STATUS_LOG, line)
        emit_alert(
            "critical",
            f"MBO validator FAIL on {npz.name}: {len(failures)} issue(s)",
            {**info, "failures": failures},
        )
        return 1

    except Exception:
        tb = traceback.format_exc()
        emit_alert(
            "critical",
            "MBO validator crashed (internal error)",
            {"traceback": tb},
        )
        sys.stderr.write(tb)
        return 3


if __name__ == "__main__":
    sys.exit(main())
