#!/usr/bin/env python3
"""
razer_watchdog_alerter.py — HC #422 Rule 1 — Jupiter-side companion to
live_stack_watchdog.py on Razer.

Reads the Razer watchdog status JSON via SSH every N min. If any process has
alarms OR the status file itself is stale (> 3 min during RTH, > 30 min off-hours),
posts a Discord alert via the discord MCP webhook (or HTTP if configured).

Why this exists: Razer's .env has no Discord webhook URL (only Rithmic creds).
Rather than putting secrets on Razer, the alerter lives on Jupiter where the
existing Discord webhook is already configured for monitoring infrastructure.

Run via cron (CronCreate "*/3 * * * *") or as a long-running daemon:
    python3 /home/jupiter/Lvl3Quant/ops/razer_watchdog_alerter.py --once
    python3 /home/jupiter/Lvl3Quant/ops/razer_watchdog_alerter.py --daemon --interval 180

The discord_webhook_url is sourced (in priority order):
    1. --webhook-url CLI arg
    2. DISCORD_WEBHOOK_LIVE_STACK env var
    3. /home/jupiter/Lvl3Quant/ops/.discord_webhook (single-line file)
    4. Fallback: write to /home/jupiter/Lvl3Quant/logs/alerter/alerter_pending.jsonl
       so the parent Claude session can pick up alerts via JSONL tail
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request, error as urlerr


# ---------- Config ----------
RAZER_HOST = "claude@razer"
RAZER_STATUS_PATH = r"C:\Users\claude\Lvl3Quant\logs\watchdog\watchdog_status.json"
JUPITER_OPS_DIR = Path("/home/jupiter/Lvl3Quant/ops")
JUPITER_LOG_DIR = Path("/home/jupiter/Lvl3Quant/logs/alerter")
JUPITER_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_STATUS_SNAPSHOT = JUPITER_LOG_DIR / "razer_watchdog_latest.json"
PENDING_ALERTS_JSONL = JUPITER_LOG_DIR / "alerter_pending.jsonl"
HISTORY_JSONL = JUPITER_LOG_DIR / "alerter_history.jsonl"

# Status file stall thresholds (NOT the watchdog's own counter thresholds —
# these check whether the watchdog ITSELF is alive)
STATUS_STALL_RTH_MIN = 3
STATUS_STALL_OFF_MIN = 30

# Per-alarm Discord throttle
ALERT_DEDUP_MIN = 15  # don't re-fire same alarm-key within N min
DEAD_DEDUP_MIN = 5  # death alarms re-fire faster


def now_ny():
    try:
        import zoneinfo
        return datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()


def is_weekend_or_holiday() -> bool:
    """Return True if markets are closed (weekend). Paper trader expected to be off."""
    n = now_ny()
    return n.weekday() >= 5  # Saturday=5, Sunday=6


def is_rth_now() -> bool:
    n = now_ny()
    if n.weekday() >= 5:
        return False
    return (n.hour, n.minute) >= (9, 30) and (n.hour, n.minute) <= (16, 0)


# ---------- Webhook resolution ----------
def resolve_webhook() -> Optional[str]:
    env = os.environ.get("DISCORD_WEBHOOK_LIVE_STACK", "").strip()
    if env:
        return env
    p = JUPITER_OPS_DIR / ".discord_webhook"
    if p.exists():
        return p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    return None


def post_discord_webhook(url: str, msg: str) -> bool:
    payload = {"content": msg[:1900]}
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "razer_watchdog_alerter/1.0"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except (urlerr.URLError, Exception) as e:
        write_history({"type": "webhook_fail", "error": repr(e)})
        return False


def write_pending_alert(msg: str, alarm_key: str) -> None:
    """Fallback: write to JSONL so parent Claude session can read + post."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "key": alarm_key,
        "message": msg,
    }
    with PENDING_ALERTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def write_history(rec: Dict[str, Any]) -> None:
    rec = dict(rec)
    rec.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with HISTORY_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------- Razer status fetch ----------
def fetch_razer_status(timeout_s: int = 10) -> Optional[Dict[str, Any]]:
    """SSH to Razer + read the watchdog status JSON."""
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        RAZER_HOST,
        # Windows-side: type the file
        f'cmd /c type "{RAZER_STATUS_PATH}"',
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if r.returncode != 0:
            write_history({"type": "ssh_fail", "rc": r.returncode, "stderr": r.stderr[:200]})
            return None
        return json.loads(r.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        write_history({"type": "fetch_error", "error": repr(e)})
        return None
    except Exception as e:
        write_history({"type": "fetch_error", "error": repr(e)})
        return None


# ---------- Alarm evaluation ----------
def is_status_stale(status: Dict[str, Any]) -> Optional[str]:
    """Return stale reason if watchdog hasn't updated status in too long; else None.

    HC #492: Skip stale-check entirely on weekends — paper trader and Razer
    watchdog are intentionally off when markets are closed. Alerting on expected
    downtime was spamming Discord with 'RAZER DEAD' every 5-15 min all weekend.
    """
    # Weekend: paper trader intentionally off, don't alert
    if is_weekend_or_holiday():
        return None

    ts_iso = status.get("ts")
    if not ts_iso:
        return "no ts field"
    try:
        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except Exception:
        return f"unparseable ts: {ts_iso}"
    age_min = (datetime.now(timezone.utc) - ts).total_seconds() / 60
    threshold = STATUS_STALL_RTH_MIN if is_rth_now() else STATUS_STALL_OFF_MIN
    if age_min > threshold:
        return f"status JSON {age_min:.1f}min stale (>{threshold}min threshold)"
    return None


# In-memory dedup (resets on alerter restart; that's OK for cron mode since each
# invocation re-reads HISTORY for "recently-fired" lookback)
def already_fired_recently(alarm_key: str, dead: bool) -> bool:
    if not HISTORY_JSONL.exists():
        return False
    threshold_min = DEAD_DEDUP_MIN if dead else ALERT_DEDUP_MIN
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=threshold_min)
    # Tail-read last 200 lines
    lines = HISTORY_JSONL.read_text(encoding="utf-8").splitlines()[-200:]
    for raw in reversed(lines):
        try:
            r = json.loads(raw)
        except Exception:
            continue
        if r.get("type") != "alert_fired":
            continue
        if r.get("key") != alarm_key:
            continue
        try:
            t = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
        except Exception:
            continue
        if t > cutoff:
            return True
    return False


def evaluate_and_alert(status: Dict[str, Any], webhook: Optional[str]) -> List[str]:
    """Return list of alarm-keys that fired this cycle."""
    fired: List[str] = []
    host = status.get("host", "razer")
    cycle = status.get("cycle")
    rth = status.get("rth")

    # 1. Watchdog self-liveness — is the status JSON stale?
    stale_reason = is_status_stale(status)
    if stale_reason:
        key = "watchdog_self_stale"
        if not already_fired_recently(key, dead=False):
            msg = f":rotating_light: **WATCHDOG SELF-STALE** on {host}: {stale_reason}"
            push_alert(msg, key, webhook)
            fired.append(key)

    # 2. Per-process alarms from the watchdog
    for proc_name, alarms in (status.get("alarms") or {}).items():
        if not alarms:
            # If this proc had a fired alarm and is now clean, fire recovery
            key = f"{proc_name}::active"
            if already_fired_recently(key, dead=False):
                rec_key = f"{proc_name}::recovery"
                if not already_fired_recently(rec_key, dead=False):
                    push_alert(
                        f":white_check_mark: **WATCHDOG RECOVERY** [{proc_name}] now healthy on {host}.",
                        rec_key, webhook,
                    )
                    fired.append(rec_key)
            continue
        # Active alarm(s)
        dead = any("DEAD" in a or "GATE-STARVED" in a for a in alarms)
        key = f"{proc_name}::active"
        if already_fired_recently(key, dead=dead):
            continue
        icon = ":rotating_light:" if dead else ":warning:"
        bullets = "\n".join(f"• {a}" for a in alarms)
        msg = f"{icon} **WATCHDOG ALARM** [{proc_name}] on {host} (cycle {cycle}, rth={rth})\n{bullets}"
        push_alert(msg, key, webhook)
        fired.append(key)

    return fired


def push_alert(msg: str, alarm_key: str, webhook: Optional[str]) -> None:
    """Try webhook first, fall back to pending JSONL. Always record to history."""
    delivered = False
    if webhook:
        delivered = post_discord_webhook(webhook, msg)
    if not delivered:
        write_pending_alert(msg, alarm_key)
    write_history({
        "type": "alert_fired",
        "key": alarm_key,
        "delivered_via_webhook": delivered,
        "message_first_line": msg.split("\n")[0],
    })


# ---------- Main loop ----------
def run_once(webhook: Optional[str]) -> int:
    """Returns count of alarms fired this cycle."""
    status = fetch_razer_status()
    if status is None:
        # Couldn't fetch — alarm on that itself
        key = "fetch_failed"
        if not already_fired_recently(key, dead=False):
            push_alert(
                ":rotating_light: **ALERTER CANNOT REACH RAZER**: SSH or status file read failed. Razer may be offline OR watchdog status path missing.",
                key, webhook,
            )
            return 1
        return 0
    # Save snapshot
    try:
        LOCAL_STATUS_SNAPSHOT.write_text(json.dumps(status, indent=2), encoding="utf-8")
    except Exception:
        pass
    fired = evaluate_and_alert(status, webhook)
    return len(fired)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="Run a single check and exit (for cron)")
    ap.add_argument("--daemon", action="store_true", help="Loop forever")
    ap.add_argument("--interval", type=int, default=180, help="Daemon mode poll interval (seconds)")
    ap.add_argument("--webhook-url", default=None, help="Override webhook URL")
    args = ap.parse_args()

    webhook = args.webhook_url or resolve_webhook()
    if webhook is None:
        sys.stderr.write(
            "[alerter] WARN: no DISCORD_WEBHOOK_LIVE_STACK configured. Alarms will be "
            f"written to {PENDING_ALERTS_JSONL} for the parent Claude session to relay.\n"
        )

    if args.daemon:
        while True:
            try:
                run_once(webhook)
            except Exception as e:
                write_history({"type": "loop_exception", "error": repr(e)})
            time.sleep(args.interval)
    else:
        n = run_once(webhook)
        print(f"[alerter] fired {n} alarm(s). webhook={'yes' if webhook else 'no'} rth={is_rth_now()}")


if __name__ == "__main__":
    main()
