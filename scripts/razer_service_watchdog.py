#!/usr/bin/env python3
"""
razer_service_watchdog.py — Monitors MBO recorder + live inference on Razer.
Restarts crashed processes, alerts Discord on failures.
Run every 60s via Windows scheduled task.
"""
import subprocess
import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")
LVL3 = Path(r"C:\Users\claude\Lvl3Quant")
LOGS = LVL3 / "live_trading" / "logs"
PYTHON = r"C:\Python311\python.exe"

SERVICES = {
    "mbo_recorder": {
        "search": "mbo_recorder.py",
        "cmd": [PYTHON, str(LVL3 / "live_trading" / "mbo_recorder.py"), "--symbol", "NQM6", "--exchange", "CME"],
        "log": str(LOGS / "mbo_recorder_out.log"),
    },
    "live_inference": {
        "search": "paper_trading_mamba_v2.py",
        "cmd": [PYTHON, str(LVL3 / "live_trading" / "paper_trading_mamba_v2.py"),
                "--weights", str(LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt"),
                "--stats", str(LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_09_feature_stats.npz"),
                "--device", "cuda"],
        "log": str(LOGS / "live_inference_out.log"),
    },
}

STATE_FILE = LVL3 / "live_trading" / "logs" / "watchdog_state.json"


def discord_alert(msg: str):
    """Send alert to Discord via webhook if configured."""
    if not DISCORD_WEBHOOK:
        return
    try:
        data = json.dumps({"content": msg}).encode()
        req = Request(DISCORD_WEBHOOK, data=data, headers={"Content-Type": "application/json"})
        urlopen(req, timeout=10)
    except Exception as e:
        print(f"Discord alert failed: {e}")


def is_process_running(search_str: str) -> bool:
    """Check if a process with search_str in its command line is running."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=10
        )
        # tasklist won't show command line, use wmic
        result = subprocess.run(
            ["wmic", "process", "where", "name='python.exe'", "get", "commandline"],
            capture_output=True, text=True, timeout=10
        )
        return search_str in result.stdout
    except Exception:
        # Fallback: just check if ANY python.exe is running
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True, text=True, timeout=10
            )
            return "python.exe" in result.stdout
        except Exception:
            return False


def restart_service(name: str, svc: dict):
    """Restart a service by launching it detached."""
    print(f"[{datetime.now()}] Restarting {name}...")
    try:
        log_file = open(svc["log"], "a")
        proc = subprocess.Popen(
            svc["cmd"],
            stdout=log_file, stderr=subprocess.STDOUT,
            cwd=str(LVL3 / "live_trading"),
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
        )
        print(f"  Started PID {proc.pid}")
        discord_alert(f"**RAZER WATCHDOG**: Restarted `{name}` (PID {proc.pid})")
        return proc.pid
    except Exception as e:
        print(f"  FAILED: {e}")
        discord_alert(f"**RAZER WATCHDOG CRITICAL**: Failed to restart `{name}`: {e}")
        return None


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"restart_counts": {}, "last_check": None}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def main():
    state = load_state()
    now = datetime.now().isoformat()
    state["last_check"] = now

    for name, svc in SERVICES.items():
        running = is_process_running(svc["search"])
        if running:
            print(f"[{now}] {name}: OK")
            state["restart_counts"][name] = 0
        else:
            count = state["restart_counts"].get(name, 0)
            if count < 5:  # Max 5 restarts before giving up
                print(f"[{now}] {name}: DOWN — restarting (attempt {count + 1}/5)")
                restart_service(name, svc)
                state["restart_counts"][name] = count + 1
            else:
                print(f"[{now}] {name}: DOWN — max restarts reached, alerting only")
                discord_alert(f"**RAZER WATCHDOG CRITICAL**: `{name}` has crashed 5 times. Manual intervention needed.")

    save_state(state)


if __name__ == "__main__":
    main()
