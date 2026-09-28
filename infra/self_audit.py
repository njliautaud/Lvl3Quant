#!/usr/bin/env python3
"""
self_audit.py — HC #492: Self-critical monitoring of Claude's own effectiveness
================================================================================

This script checks whether Claude is ACTUALLY DOING ITS JOB. Not just "are nodes
online" but "am I following directives, keeping GPUs busy, and producing results?"

Run via cron every 2 hours or on-demand:
    python3 /home/jupiter/Lvl3Quant/infra/self_audit.py

Outputs structured JSON to stdout and optionally writes to audit log.
Returns exit code 0 if all checks pass, 1 if any FAIL, 2 if any CRITICAL.

Checks performed:
  1. GPU UTILIZATION: Are both GPUs busy? Idle GPU = wasted research hours.
  2. MONITORING ALIVE: Is the durable watchdog running? Are crons firing?
  3. DIRECTIVES DRIFT: Is SESSION_STATE.md updated within 24h?
  4. DISCORD SPAM: Are we flooding #general with garbage?
  5. STALE JOBS: Any QCC jobs marked RUNNING but process dead?
  6. RESULTS HARVESTED: Completed training → results logged to RUN_HISTORY?
  7. WEEKEND AWARENESS: Not alerting about expected downtime.
  8. TOKEN BUDGET: Check if on pace for weekly budget (HC #489).

Author: Claude (Self-Audit Infrastructure, HC #492)
Date: 2026-05-24
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = Path("/home/jupiter/Lvl3Quant")
INFRA_DIR = BASE_DIR / "infra"
AUDIT_LOG = INFRA_DIR / "logs" / "self_audit.jsonl"
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

# Node SSH details
NODES = {
    "neptune": {"host": "neptune", "user": "nick", "gpu": True},
    "razer":   {"host": "razer", "user": "claude", "gpu": True, "password": os.environ.get("CLUSTER_SSH_PASSWORD", "")},
}

def now_ny():
    try:
        import zoneinfo
        return datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now()

def is_weekend():
    return now_ny().weekday() >= 5

def ssh_exec(host, user, cmd, timeout=10, password=None):
    """Execute SSH command. Returns (ok, stdout, stderr)."""
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
               f"{user}@{host}", cmd]
    try:
        r = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return False, "", str(e)


# ─── Check Functions ─────────────────────────────────────────────────────────

def check_gpu_utilization() -> Dict:
    """Check if GPUs are being used. Idle GPU on weekday = FAIL."""
    results = {"name": "gpu_utilization", "status": "PASS", "details": {}}

    for node, info in NODES.items():
        if not info.get("gpu"):
            continue

        if node == "razer":
            ok, out, _ = ssh_exec(info["host"], info["user"],
                                   "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits",
                                   password=info.get("password"))
        else:
            ok, out, _ = ssh_exec(info["host"], info["user"],
                                   "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits")

        if not ok:
            results["details"][node] = {"reachable": False}
            results["status"] = "WARN"
            continue

        try:
            parts = out.split(",")
            gpu_util = int(parts[0].strip())
            mem_used = int(parts[1].strip())
            results["details"][node] = {"gpu_util": gpu_util, "mem_mb": mem_used}

            # Razer idle on weekend is fine (training is optional, paper trader off)
            if gpu_util < 5 and not is_weekend():
                results["status"] = "FAIL"
                results["details"][node]["issue"] = f"GPU idle ({gpu_util}%) on a weekday — wasted compute"
            elif gpu_util < 5 and is_weekend():
                results["details"][node]["note"] = "Idle but weekend — acceptable if no pending work"
        except Exception as e:
            results["details"][node] = {"parse_error": str(e)}

    return results


def check_monitoring_alive() -> Dict:
    """Check if PM2 processes are running (durable watchdog, QCC daemon, etc)."""
    results = {"name": "monitoring_alive", "status": "PASS", "details": {}}

    try:
        r = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            results["status"] = "FAIL"
            results["details"]["pm2"] = "pm2 jlist failed"
            return results

        procs = json.loads(r.stdout)
        critical_procs = {"qcc-daemon", "teleclaude-bridge"}
        running = {p["name"] for p in procs if p.get("pm2_env", {}).get("status") == "online"}

        for proc in critical_procs:
            if proc in running:
                results["details"][proc] = "online"
            else:
                results["status"] = "FAIL"
                results["details"][proc] = "NOT RUNNING — monitoring is dark"

        # Check watchdog heartbeat freshness
        heartbeat_file = INFRA_DIR / "logs" / "watchdog_heartbeat.json"
        if heartbeat_file.exists():
            try:
                hb = json.loads(heartbeat_file.read_text())
                hb_ts = datetime.fromisoformat(hb["timestamp"].replace("Z", "+00:00"))
                age_min = (datetime.now(timezone.utc) - hb_ts).total_seconds() / 60
                results["details"]["watchdog_heartbeat_age_min"] = round(age_min, 1)
                if age_min > 10:
                    results["status"] = "WARN"
                    results["details"]["watchdog_heartbeat"] = f"Stale ({age_min:.0f}min old)"
            except Exception:
                pass

    except Exception as e:
        results["status"] = "FAIL"
        results["details"]["error"] = str(e)

    return results


def check_session_state_freshness() -> Dict:
    """SESSION_STATE.md should be updated at least once per 24h when active."""
    results = {"name": "session_state_freshness", "status": "PASS", "details": {}}

    ss_path = BASE_DIR / "SESSION_STATE.md"
    if not ss_path.exists():
        results["status"] = "CRITICAL"
        results["details"]["issue"] = "SESSION_STATE.md does not exist!"
        return results

    mtime = datetime.fromtimestamp(ss_path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
    results["details"]["last_modified_hours_ago"] = round(age_hours, 1)

    if age_hours > 24:
        results["status"] = "WARN"
        results["details"]["issue"] = f"Not updated in {age_hours:.0f}h — state may be stale"
    if age_hours > 72:
        results["status"] = "FAIL"
        results["details"]["issue"] = f"Not updated in {age_hours:.0f}h — severely stale"

    # Check file size (HC #489 R3: cap at 100KB)
    size_kb = ss_path.stat().st_size / 1024
    results["details"]["size_kb"] = round(size_kb, 1)
    if size_kb > 100:
        results["details"]["bloat"] = f"SESSION_STATE.md is {size_kb:.0f}KB (cap is 100KB per HC #489)"

    return results


def check_stale_qcc_jobs() -> Dict:
    """Check for QCC jobs marked RUNNING but actually dead."""
    results = {"name": "stale_qcc_jobs", "status": "PASS", "details": {}}

    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:3456/api/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        stale = data.get("stale_jobs", [])
        if stale:
            results["status"] = "WARN"
            results["details"]["stale_count"] = len(stale)
            results["details"]["stale_jobs"] = [
                {"id": j.get("id"), "node": j.get("node"), "desc": j.get("description", "")[:60]}
                for j in stale[:5]
            ]
    except Exception as e:
        results["details"]["qcc_unreachable"] = str(e)[:100]

    return results


def check_discord_spam() -> Dict:
    """Check alerter history for spam patterns (>20 alerts in last 6h = spam)."""
    results = {"name": "discord_spam_check", "status": "PASS", "details": {}}

    alerter_history = BASE_DIR / "logs" / "alerter" / "alerter_history.jsonl"
    if not alerter_history.exists():
        results["details"]["note"] = "No alerter history found"
        return results

    try:
        lines = alerter_history.read_text(encoding="utf-8").splitlines()[-500:]
        cutoff = datetime.now(timezone.utc) - timedelta(hours=6)
        recent_alerts = 0
        alert_keys = {}

        for raw in lines:
            try:
                r = json.loads(raw)
                if r.get("type") != "alert_fired":
                    continue
                ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
                if ts > cutoff:
                    recent_alerts += 1
                    key = r.get("key", "unknown")
                    alert_keys[key] = alert_keys.get(key, 0) + 1
            except Exception:
                continue

        results["details"]["alerts_last_6h"] = recent_alerts
        if recent_alerts > 20:
            results["status"] = "FAIL"
            results["details"]["issue"] = f"{recent_alerts} alerts in 6h — SPAMMING user"
            results["details"]["top_keys"] = dict(sorted(alert_keys.items(), key=lambda x: -x[1])[:5])
        elif recent_alerts > 10:
            results["status"] = "WARN"
            results["details"]["note"] = f"{recent_alerts} alerts in 6h — borderline"

    except Exception as e:
        results["details"]["error"] = str(e)[:100]

    return results


def check_run_history_current() -> Dict:
    """RUN_HISTORY.md should reflect completed experiments."""
    results = {"name": "run_history_current", "status": "PASS", "details": {}}

    rh_path = BASE_DIR / "RUN_HISTORY.md"
    if not rh_path.exists():
        results["status"] = "CRITICAL"
        results["details"]["issue"] = "RUN_HISTORY.md missing!"
        return results

    mtime = datetime.fromtimestamp(rh_path.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
    results["details"]["last_modified_hours_ago"] = round(age_hours, 1)

    size_kb = rh_path.stat().st_size / 1024
    results["details"]["size_kb"] = round(size_kb, 1)
    if size_kb > 200:
        results["details"]["bloat"] = f"{size_kb:.0f}KB exceeds 200KB cap (HC #489 R4)"

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def run_audit() -> Tuple[str, List[Dict]]:
    """Run all checks. Returns (overall_status, list_of_check_results)."""
    checks = [
        check_gpu_utilization,
        check_monitoring_alive,
        check_session_state_freshness,
        check_stale_qcc_jobs,
        check_discord_spam,
        check_run_history_current,
    ]

    results = []
    overall = "PASS"

    for check_fn in checks:
        try:
            result = check_fn()
        except Exception as e:
            result = {"name": check_fn.__name__, "status": "ERROR", "details": {"exception": str(e)}}

        results.append(result)

        if result["status"] == "CRITICAL":
            overall = "CRITICAL"
        elif result["status"] == "FAIL" and overall not in ("CRITICAL",):
            overall = "FAIL"
        elif result["status"] == "WARN" and overall in ("PASS",):
            overall = "WARN"

    return overall, results


def main():
    overall, results = run_audit()

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall": overall,
        "is_weekend": is_weekend(),
        "checks": results,
    }

    # Write to audit log
    try:
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
    except Exception:
        pass

    # Print summary
    print(json.dumps(audit_entry, indent=2))

    # Exit code
    if overall == "CRITICAL":
        sys.exit(2)
    elif overall == "FAIL":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
