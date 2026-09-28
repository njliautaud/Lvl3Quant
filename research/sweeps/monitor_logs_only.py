#!/usr/bin/env python3
"""
Logs-only monitoring system (HC #194)
Monitors training and live trading by reading logs, NOT GPU metrics.

Tracks:
- Neptune: Split DQN epoch/fold progress, training metrics
- Razer: Paper trader activity, trades, errors
- Detects: Stalls (no log output >5 min), crashes, errors
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta
import re

# ===== CONFIG =====
NEPTUNE_LOG = Path("/home/nick/Lvl3Quant/split_dqn_v1_r14_neptune.log")
RAZER_LOGS = [
    Path("C:\\Users\\claude\\Lvl3Quant\\logs\\paper_mamba_v2_*.log"),
    Path("C:\\Users\\claude\\Lvl3Quant\\logs\\mamba_v2_signals_ESU6_*.jsonl"),
]

STALL_THRESHOLD = 300  # 5 minutes
CHECK_INTERVAL = 60    # Check every 60 seconds

# ===== MONITORING STATE =====
last_log_output = {}
last_check = {}

def ssh_exec(host, user, cmd):
    """Execute command on remote host via SSH"""
    ssh_cmd = f"ssh -o ConnectTimeout=5 {user}@{host} \"{cmd}\""
    try:
        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except:
        return None

def tail_file_ssh(host, user, filepath, lines=20):
    """Tail a file on remote host"""
    cmd = f"tail -{lines} {filepath}" if ":" not in filepath else f"type {filepath}"
    return ssh_exec(host, user, cmd)

def check_neptune():
    """Check Neptune training log"""
    print(f"\n[{datetime.now().isoformat()}] CHECKING NEPTUNE...")

    log_content = ssh_exec("neptune", "nick", f"tail -50 {NEPTUNE_LOG}")

    if not log_content:
        print("  ❌ Cannot read Neptune log (host unreachable?)")
        return {"status": "unreachable"}

    # Parse for progress
    matches = re.findall(r"Fold (\d+).*Ep (\d+).*(E=\d+.*X=\d+)", log_content, re.DOTALL)
    if matches:
        fold, epoch, updates = matches[-1]
        print(f"  ✅ Training: Fold {fold} Ep {epoch} | GPU updates: {updates}")
        last_log_output["neptune"] = datetime.now()
        return {"status": "training", "fold": fold, "epoch": epoch}

    # Check for errors
    if "error" in log_content.lower() or "exception" in log_content.lower():
        print(f"  ⚠️  ERROR in log: {log_content[-200:]}")
        return {"status": "error"}

    # Check for stall
    if last_log_output.get("neptune"):
        elapsed = (datetime.now() - last_log_output["neptune"]).total_seconds()
        if elapsed > STALL_THRESHOLD:
            print(f"  ⚠️  STALL DETECTED: No log output for {elapsed:.0f}s")
            return {"status": "stalled"}

    return {"status": "unknown"}

def check_razer():
    """Check Razer paper trader logs"""
    print(f"\n[{datetime.now().isoformat()}] CHECKING RAZER...")

    # Get latest paper trader log
    log_list = ssh_exec("razer", "claude",
        "powershell -Command \"Get-ChildItem 'C:\\\\Users\\\\claude\\\\Lvl3Quant\\\\logs' -Filter 'paper_mamba_v2*.log' -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName\"")

    if not log_list:
        print("  ❌ No paper trader log found (not running?)")
        return {"status": "not_running"}

    # Read latest log
    log_content = ssh_exec("razer", "claude", f"type {log_list.strip()}")

    if not log_content or log_content == "0":
        print(f"  ⚠️  Paper trader log empty or crashed")
        return {"status": "crashed"}

    # Check for trading activity
    if "Entered" in log_content or "Trade" in log_content:
        trades = log_content.count("Entered")
        print(f"  ✅ Paper trading active: {trades} trades executed")
        last_log_output["razer"] = datetime.now()
        return {"status": "trading", "trades": trades}

    if "signal" in log_content.lower() or "prediction" in log_content.lower():
        print(f"  ✅ Paper trader running (generating signals)")
        last_log_output["razer"] = datetime.now()
        return {"status": "running"}

    if "error" in log_content.lower():
        print(f"  ❌ ERROR: {log_content[-300:]}")
        return {"status": "error"}

    print(f"  ⚠️  Unknown status (check logs manually)")
    return {"status": "unknown"}

def main():
    print("=" * 70)
    print("LOGS-ONLY MONITORING (HC #194)")
    print("Monitoring by log content, NOT GPU metrics")
    print("=" * 70)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'=' * 70}")
        print(f"CHECK #{iteration} — {datetime.now().isoformat()}")
        print(f"{'=' * 70}")

        neptune_status = check_neptune()
        razer_status = check_razer()

        print(f"\n📊 SUMMARY:")
        print(f"  Neptune: {neptune_status.get('status', '?').upper()}")
        print(f"  Razer:   {razer_status.get('status', '?').upper()}")

        print(f"\n⏱️  Sleeping {CHECK_INTERVAL}s until next check...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
