#!/usr/bin/env python3
"""
live_stack_watchdog.py — HC #422 Rule 1 mandatory active-heartbeat watchdog.

Runs on the LIVE host (Razer) next to the paper-trading + MBO recorder + shadow trader
processes. Polls per-process status files and per-process JSONLs every 60s. Posts to
Discord webhook + writes local rolling JSONL when ANY heartbeat is broken.

A heartbeat is "broken" if any of the following hold for > STALL_MIN_RTH min during RTH:
  - process is dead (psutil.pid_exists == False)
  - events_counter not advancing (status file)
  - preds_counter not advancing (status file)
  - passed_gate cumulative is 0 across the full RTH session (status file)
  - JSONL file mtime > STALL_MIN_RTH min old

Run on Razer:
    pythonw.exe C:\\Users\\claude\\Lvl3Quant\\live_trading_linux\\live_stack_watchdog.py

Launch via Win32_Process.Create pattern (HC #401) so it survives SSH disconnect.

Self-heartbeat: writes own status to `C:\\Users\\claude\\Lvl3Quant\\logs\\watchdog\\watchdog_status.json`
every cycle. A separate Jupiter-side cron reads this to verify the watchdog ITSELF is alive.
"""
from __future__ import annotations

import json
import os
import sys
import time
import socket
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, error as urlerr

try:
    import psutil
except ImportError:
    print("ERROR: psutil required. pip install psutil", file=sys.stderr)
    sys.exit(1)


# ---------- Configuration ----------
# On Razer (Windows). On Linux dry-run, these are still valid string paths (existence-checked).
IS_WINDOWS = os.name == "nt"
LVL3_ROOT = (
    Path("C:/Users/claude/Lvl3Quant") if IS_WINDOWS else Path("/home/jupiter/Lvl3Quant")
)
LOG_ROOT = LVL3_ROOT / "logs"
WATCHDOG_LOG_DIR = LOG_ROOT / "watchdog"
WATCHDOG_LOG_DIR.mkdir(parents=True, exist_ok=True)
WATCHDOG_STATUS_FILE = WATCHDOG_LOG_DIR / "watchdog_status.json"
WATCHDOG_EVENTS_JSONL = WATCHDOG_LOG_DIR / "watchdog_events.jsonl"

POLL_INTERVAL_SEC = 60
STALL_MIN_RTH = 10  # heartbeat broken if any counter stale > N min during RTH
STALL_MIN_OFFHOURS = 60  # more lenient when market closed

# RTH: 09:30 - 16:00 America/New_York
import zoneinfo

NY_TZ = zoneinfo.ZoneInfo("America/New_York") if hasattr(zoneinfo, "ZoneInfo") else None


def is_rth_now() -> bool:
    if NY_TZ is None:
        return True
    now = datetime.now(NY_TZ)
    if now.weekday() >= 5:
        return False
    open_ = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_ <= now <= close_


# ---------- Discord webhook ----------
# Source the same .env Razer paper traders use. NEVER hard-code the URL.
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_LIVE_STACK", "")
if not DISCORD_WEBHOOK:
    # Try loading from .env on Razer
    env_path = LVL3_ROOT / "live_trading_linux" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("DISCORD_WEBHOOK_LIVE_STACK="):
                DISCORD_WEBHOOK = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
            if line.startswith("DISCORD_WEBHOOK=") and not DISCORD_WEBHOOK:
                DISCORD_WEBHOOK = line.split("=", 1)[1].strip().strip('"').strip("'")


def post_discord(msg: str, mention_user: bool = False) -> bool:
    if not DISCORD_WEBHOOK:
        return False
    if mention_user:
        msg = "@everyone " + msg  # user can convert to role mention later
    payload = {"content": msg[:1900]}
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        DISCORD_WEBHOOK,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "live_stack_watchdog/1.0"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except urlerr.URLError as e:
        log_event({"type": "discord_post_fail", "error": str(e)})
        return False
    except Exception as e:
        log_event({"type": "discord_post_fail", "error": repr(e)})
        return False


def log_event(event: Dict[str, Any]) -> None:
    event = dict(event)
    event.setdefault("ts", datetime.now(timezone.utc).isoformat())
    event.setdefault("host", socket.gethostname())
    try:
        with WATCHDOG_EVENTS_JSONL.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass


# ---------- Process registry ----------
# Each entry: name, expected_substring_in_cmdline, status_file (optional), jsonl_glob,
# counters_to_check (list of dict-key names in status file), critical (bool).
PROCESS_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "mbo_recorder",
        # mbo_recorder.py runs as `python.exe ... mbo_recorder.py --symbol ESM6 ...`
        "cmdline_match": ["mbo_recorder.py"],
        "status_file": None,  # No status file — only PID + log mtime
        "jsonl_glob": "logs/mbo_recorder*.log",
        "counters": [],
        "critical": True,
    },
    {
        "name": "legacy_paper_top5",
        # legacy is `paper_trading_mamba_v2.py --min-tier Top5%`
        "cmdline_match": ["paper_trading_mamba_v2.py"],
        "status_file": None,  # Legacy has no JSON status file; track log mtime
        "jsonl_glob": "live_trading/logs/paper_mamba_v2_*.log",
        "counters": [],
        "critical": True,
    },
    {
        "name": "shadow_v2_1s_short_top05",
        # shadow runs as `pythonw.exe paper_trading_v2_1s_short_top05.py --shadow ...`
        "cmdline_match": ["paper_trading_v2_1s_short_top05.py"],
        "status_file": LVL3_ROOT / "output" / "v2_1s_short_top05_heartbeat.json",
        "jsonl_glob": "live_trading_linux/logs/v2_1s_short_top05_paper_*.jsonl",
        # Heartbeat JSON schema fields:
        # n_signals_total — total preds processed (must advance)
        # n_signals_passed_gate — gate trips (0 cumulative during RTH = ALARM per HC #422 Rule 1)
        # fills_today — fills (allowed to be 0; just informational)
        # session_uptime_s — uptime (monotonic, validates clock)
        "counters": ["n_signals_total", "n_signals_passed_gate"],
        # Special: passed_gate=0 during RTH is the user-painful alarm
        "passed_gate_field": "n_signals_passed_gate",
        "critical": True,
    },
]


# ---------- Heartbeat state (persists across cycles in-memory; reset on launch) ----------
class HeartbeatState:
    def __init__(self) -> None:
        # name -> {"pid": int, "first_seen": iso, "last_advance": {counter_name: (value, iso_ts)}, "jsonl_last_mtime": float}
        self.state: Dict[str, Dict[str, Any]] = {}
        # Alarms already fired (don't spam): name -> last_alarm_iso
        self.last_alarm: Dict[str, str] = {}
        self.cycle_count = 0


def find_process(cmdline_substrs: List[str]) -> Optional[psutil.Process]:
    for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or []).lower()
            if any(s.lower() in cmd for s in cmdline_substrs):
                return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def read_status_file(p: Optional[Path]) -> Optional[Dict[str, Any]]:
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def newest_jsonl(glob_pattern: str) -> Optional[Path]:
    matches = sorted(LVL3_ROOT.glob(glob_pattern), key=lambda x: x.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def evaluate_process(entry: Dict[str, Any], state: HeartbeatState) -> List[str]:
    """Return list of alarm strings for this entry. Empty = healthy."""
    alarms: List[str] = []
    name = entry["name"]
    now_iso = datetime.now(timezone.utc).isoformat()
    proc = find_process(entry["cmdline_match"])

    if proc is None:
        alarms.append(f"DEAD: no process matching {entry['cmdline_match']}")
        state.state.setdefault(name, {})["pid"] = None
        return alarms

    # Track PID — alarm if it changed (process restart we didn't initiate)
    prev_pid = state.state.get(name, {}).get("pid")
    state.state.setdefault(name, {})["pid"] = proc.pid
    if prev_pid and prev_pid != proc.pid:
        alarms.append(f"PID-CHANGE: was {prev_pid}, now {proc.pid} (silent restart?)")

    # Status file counter advancement
    status = read_status_file(entry.get("status_file"))
    if status is None and entry["counters"]:
        # Tolerated if file simply doesn't exist yet (first launch); only alarm if we've
        # previously seen it
        if state.state[name].get("had_status"):
            alarms.append(f"STATUS-FILE-DISAPPEARED: {entry['status_file']}")
    elif status is not None:
        state.state[name]["had_status"] = True
        passed_gate_field = entry.get("passed_gate_field")
        for c in entry["counters"]:
            val = status.get(c)
            if val is None:
                continue
            prev = state.state[name].get(f"counter_{c}")
            stall_min = STALL_MIN_RTH if is_rth_now() else STALL_MIN_OFFHOURS
            if prev is None:
                state.state[name][f"counter_{c}"] = {"v": val, "since": now_iso}
                # First-cycle passed_gate=0 during RTH: arm a separate timer immediately
                if c == passed_gate_field and val == 0 and is_rth_now():
                    state.state[name][f"passed_gate_zero_since"] = now_iso
            else:
                if val != prev["v"]:
                    state.state[name][f"counter_{c}"] = {"v": val, "since": now_iso}
                    # passed_gate moved off 0 — clear the alarm timer
                    if c == passed_gate_field and val > 0:
                        state.state[name].pop("passed_gate_zero_since", None)
                else:
                    # Counter unchanged; check stall window
                    since_dt = datetime.fromisoformat(prev["since"])
                    age_min = (datetime.now(timezone.utc) - since_dt).total_seconds() / 60
                    if age_min > stall_min:
                        if c == passed_gate_field and val == 0:
                            # passed_gate stuck at 0 during RTH = the exact failure mode user is angry about
                            scope = "RTH" if is_rth_now() else "off-hours"
                            alarms.append(
                                f"GATE-STARVED: {c}=0 cumulative for {age_min:.1f}min during {scope}"
                                f" (preds advancing? check n_signals_total)"
                            )
                        else:
                            alarms.append(
                                f"STALL: {c} frozen at {val} for {age_min:.1f}min (>{stall_min}min threshold)"
                            )

    # JSONL mtime check
    jsonl = newest_jsonl(entry["jsonl_glob"])
    if jsonl is None:
        if state.state[name].get("had_jsonl"):
            alarms.append(f"JSONL-DISAPPEARED: {entry['jsonl_glob']}")
    else:
        state.state[name]["had_jsonl"] = True
        mtime = jsonl.stat().st_mtime
        age_min = (time.time() - mtime) / 60
        stall_min = STALL_MIN_RTH if is_rth_now() else STALL_MIN_OFFHOURS
        if age_min > stall_min:
            alarms.append(
                f"JSONL-STALE: {jsonl.name} not written for {age_min:.1f}min (>{stall_min}min)"
            )
        state.state[name]["jsonl_last_mtime"] = mtime
        state.state[name]["jsonl_path"] = str(jsonl)

    return alarms


def write_watchdog_status(state: HeartbeatState, all_alarms: Dict[str, List[str]]) -> None:
    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "rth": is_rth_now(),
        "cycle": state.cycle_count,
        "processes": {name: state.state.get(name, {}) for name in [e["name"] for e in PROCESS_REGISTRY]},
        "alarms": all_alarms,
    }
    tmp = WATCHDOG_STATUS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    tmp.replace(WATCHDOG_STATUS_FILE)


def should_fire_discord(name: str, alarms: List[str], state: HeartbeatState) -> bool:
    """Throttle: same process can only fire 1 alarm message per 15 min, except DEAD which fires immediately and re-fires every 5 min."""
    if not alarms:
        # If was alarming and now clean -> fire recovery
        if state.last_alarm.get(name):
            state.last_alarm[name] = ""  # clear; recovery msg below
            return True
        return False
    last = state.last_alarm.get(name, "")
    if not last:
        state.last_alarm[name] = datetime.now(timezone.utc).isoformat()
        return True
    last_dt = datetime.fromisoformat(last)
    age_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
    is_dead = any("DEAD" in a for a in alarms)
    threshold_min = 5 if is_dead else 15
    if age_min >= threshold_min:
        state.last_alarm[name] = datetime.now(timezone.utc).isoformat()
        return True
    return False


def main() -> None:
    print(f"[watchdog] starting. host={socket.gethostname()} poll={POLL_INTERVAL_SEC}s "
          f"stall_rth={STALL_MIN_RTH}min stall_off={STALL_MIN_OFFHOURS}min "
          f"webhook_set={'yes' if DISCORD_WEBHOOK else 'no'}", flush=True)
    log_event({"type": "startup", "webhook_set": bool(DISCORD_WEBHOOK), "rth_now": is_rth_now()})
    post_discord(f":heart: live_stack_watchdog STARTED on {socket.gethostname()} — "
                 f"polling every {POLL_INTERVAL_SEC}s, alert threshold {STALL_MIN_RTH}min RTH / {STALL_MIN_OFFHOURS}min off-hours.")

    state = HeartbeatState()
    while True:
        state.cycle_count += 1
        all_alarms: Dict[str, List[str]] = {}
        for entry in PROCESS_REGISTRY:
            try:
                alarms = evaluate_process(entry, state)
            except Exception as e:
                alarms = [f"EVAL-EXCEPTION: {e!r}"]
                log_event({"type": "eval_exception", "process": entry["name"],
                           "tb": traceback.format_exc()})
            all_alarms[entry["name"]] = alarms

            if should_fire_discord(entry["name"], alarms, state):
                if alarms:
                    icon = ":rotating_light:" if any("DEAD" in a for a in alarms) else ":warning:"
                    body = (
                        f"{icon} **WATCHDOG ALARM** [{entry['name']}] on {socket.gethostname()}\n"
                        + "\n".join(f"• {a}" for a in alarms)
                    )
                    post_discord(body)
                    log_event({"type": "alarm", "process": entry["name"], "alarms": alarms})
                else:
                    # Recovery
                    post_discord(f":white_check_mark: **WATCHDOG RECOVERY** [{entry['name']}] now healthy on {socket.gethostname()}.")
                    log_event({"type": "recovery", "process": entry["name"]})

        try:
            write_watchdog_status(state, all_alarms)
        except Exception as e:
            log_event({"type": "status_write_fail", "error": repr(e)})

        # Brief stdout for nohup-tail visibility (one line per cycle)
        rth = "RTH" if is_rth_now() else "OFF"
        compact = " | ".join(
            f"{e['name']}={'OK' if not all_alarms[e['name']] else len(all_alarms[e['name']])}a"
            for e in PROCESS_REGISTRY
        )
        print(f"[watchdog] cycle={state.cycle_count} {rth} {compact}", flush=True)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
