#!/usr/bin/env python3
"""
HC #31 enforcement — MBO recorder zombie-connection detector.

Reads today's MBO NPZ file, computes events-in-last-60-seconds, and:
  - if zero events for >=120s during expected-active hours → restart mbo-recorder + Discord alert
  - if recovery within next watchdog cycle → Discord recovery alert
  - emits state to /tmp/mbo_gap_watchdog.state for cross-invocation continuity

Companion to mbo_recorder.py — does NOT modify the recorder; only watches its output.

Hours when zero events is suspicious (ET, Mon-Fri):
  - 09:00-16:00 (regular session)         — gap >120s = ALERT
  - 16:00-17:00 (post-market liquidation) — gap >180s = ALERT
  - 17:00-17:30 (CME daily maintenance)   — KNOWN DEAD ZONE, no alert (verify resume by 17:30)
  - 17:30-23:59 (after-hours)             — gap >300s = ALERT
  - 00:00-08:59 (overnight, lower volume) — gap >600s = ALERT

Run via cron every minute: `* 7-23 * * 1-5 jupiter_exec_research_watchdog ...`
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
STATE_FILE = Path("/tmp/mbo_gap_watchdog.state")
LOG_FILE = Path("/home/jupiter/Lvl3Quant/logs/mbo_gap_watchdog.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

INJECT_URL = "http://127.0.0.1:7731/inject"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with LOG_FILE.open("a") as fh:
        fh.write(line + "\n")


def gap_threshold_seconds(now_et: datetime) -> tuple[int, str]:
    """Return (gap_seconds_threshold, regime_label). 0 == ignore (maintenance window)."""
    h = now_et.hour
    m = now_et.minute
    if now_et.weekday() >= 5:  # Sat=5, Sun=6 — weekends, looser
        return (1800, "weekend")
    if 9 <= h < 16:
        return (120, "regular_session")
    if h == 16:
        return (180, "post_market_close")
    if h == 17 and m < 30:
        return (0, "cme_maintenance")  # known dead zone
    if (h == 17 and m >= 30) or (18 <= h <= 23):
        return (300, "after_hours")
    if 0 <= h < 9:
        return (600, "overnight")
    return (300, "default")


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {"last_alert_ts": 0, "last_event_ts": 0, "last_restart_ts": 0, "consecutive_gaps": 0}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"last_alert_ts": 0, "last_event_ts": 0, "last_restart_ts": 0, "consecutive_gaps": 0}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state))


def get_latest_event_ts_ns() -> tuple[int, int] | None:
    """Return (last_event_ts_ns, total_events) from today's NPZ; None if file missing.

    Recorder names files by UTC date (mbo_recorder.py line 76 uses
    datetime.now(timezone.utc)). After the UTC date roll (~20:00 ET in EDT),
    the recorder switches to the next day's file. The watchdog must match
    that logic or it reads a stale file post-roll. Fall back to today-ET
    if the UTC file is missing (handles the brief moment around the roll).
    """
    today_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
    today_et = datetime.now().strftime("%Y%m%d")
    f = DATA_DIR / f"{today_utc}_mbo_events.npz"
    if not f.exists() and today_et != today_utc:
        f = DATA_DIR / f"{today_et}_mbo_events.npz"
    if not f.exists():
        return None
    try:
        d = np.load(f, allow_pickle=True)
        ts = d.get("timestamps")
        if ts is None or len(ts) == 0:
            return (0, 0)
        return (int(ts.max()), int(len(ts)))
    except Exception as e:
        log(f"NPZ read error: {e}")
        return None


def _resolve_pm2_bin() -> str:
    """Find pm2 binary (cron PATH is stripped — must use absolute path)."""
    candidates = [
        "/home/jupiter/.npm-global/bin/pm2",
        "/usr/local/bin/pm2",
        "/usr/bin/pm2",
        os.path.expanduser("~/.npm-global/bin/pm2"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return "pm2"  # last-resort PATH lookup


def restart_recorder() -> bool:
    pm2_bin = _resolve_pm2_bin()
    try:
        result = subprocess.run(
            [pm2_bin, "restart", "mbo-recorder"],
            check=True, timeout=30, capture_output=True, text=True,
        )
        log(f"mbo-recorder PM2 restart issued via {pm2_bin}")
        return True
    except Exception as e:
        log(f"PM2 restart FAILED (bin={pm2_bin}): {e}")
        return False


def discord_alert(message: str) -> None:
    try:
        import urllib.request
        req = urllib.request.Request(
            INJECT_URL,
            data=json.dumps({"message": message}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10).read()
        log(f"Discord alert dispatched: {message[:80]}...")
    except Exception as e:
        log(f"Discord inject failed: {e}")


def main() -> int:
    now_local = datetime.now()
    threshold_s, regime = gap_threshold_seconds(now_local)

    if threshold_s == 0:
        log(f"In {regime} window — skipping check")
        return 0

    latest = get_latest_event_ts_ns()
    if latest is None:
        log(f"NPZ for today missing (regime={regime})")
        # If during regular session and no NPZ at all → that's catastrophic, alert
        if regime == "regular_session":
            state = load_state()
            now_unix = int(time.time())
            if now_unix - state.get("last_alert_ts", 0) > 600:
                discord_alert(
                    f"**🚨 MBO RECORDER FAILURE — no NPZ file for today during regular session.** "
                    f"Regime: {regime}. Restarting mbo-recorder. (HC #31 watchdog)"
                )
                restart_recorder()
                state["last_alert_ts"] = now_unix
                state["last_restart_ts"] = now_unix
                save_state(state)
        return 1

    last_event_ns, total_events = latest
    now_unix = int(time.time())
    last_event_unix = last_event_ns // 1_000_000_000
    seconds_since_last = now_unix - last_event_unix

    state = load_state()
    log(f"regime={regime} threshold={threshold_s}s last_event={seconds_since_last}s ago total={total_events}")

    # Recovery detection: if we previously alerted and now getting fresh data
    if state.get("consecutive_gaps", 0) > 0 and seconds_since_last < threshold_s:
        discord_alert(
            f"**✅ MBO RECORDER RECOVERED.** Fresh events arriving "
            f"(last event {seconds_since_last}s ago). Total events today: {total_events:,}. "
            f"Was in gap state for {state['consecutive_gaps']} watchdog cycles. (HC #31)"
        )
        state["consecutive_gaps"] = 0
        save_state(state)
        return 0

    # Gap detection
    if seconds_since_last >= threshold_s:
        state["consecutive_gaps"] = state.get("consecutive_gaps", 0) + 1
        save_state(state)

        # Rate-limit alerts to once per 5 min while in gap, but always log
        time_since_last_alert = now_unix - state.get("last_alert_ts", 0)
        time_since_last_restart = now_unix - state.get("last_restart_ts", 0)

        # First alert + restart: as soon as gap detected
        if state["consecutive_gaps"] == 1:
            discord_alert(
                f"**🚨 HC #31: MBO RECORDER GAP DETECTED.** "
                f"No events for {seconds_since_last}s during {regime} (threshold {threshold_s}s). "
                f"Total events today: {total_events:,}. Restarting mbo-recorder PM2 process."
            )
            if restart_recorder():
                state["last_restart_ts"] = now_unix
            state["last_alert_ts"] = now_unix
            save_state(state)
        # Subsequent alerts: every 5 min if still in gap, restart every 5 min
        elif time_since_last_alert >= 300:
            discord_alert(
                f"**🚨 HC #31 ONGOING GAP.** Still no events for {seconds_since_last}s "
                f"during {regime}. Cycle #{state['consecutive_gaps']}. Restarting again."
            )
            if time_since_last_restart >= 180 and restart_recorder():
                state["last_restart_ts"] = now_unix
            state["last_alert_ts"] = now_unix
            save_state(state)
        return 2

    # Normal operation
    state["consecutive_gaps"] = 0
    save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
