#!/usr/bin/env python3
"""
Auto-Dispatch: Checks experiment queue, launches next job on idle GPUs.
Run from PM2 or cron. Sends Discord webhook on launch/completion.

Usage:
    python3 auto_dispatch.py [--dry-run] [--node neptune|razer]
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

QUEUE_FILE = Path("/home/jupiter/Lvl3Quant/infra/experiment_queue.json")
DISPATCH_LOG = Path("/home/jupiter/Lvl3Quant/infra/logs/dispatch.log")
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Node SSH configs
NODES = {
    "neptune": {
        "ssh_user": "nick",
        "ssh_host": "neptune",
        "gpu_check": "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
    },
    "razer": {
        "ssh_user": "claude",
        "ssh_host": "razer",
        "gpu_check": "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
    },
}


def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    DISPATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DISPATCH_LOG, "a") as f:
        f.write(line + "\n")


def check_gpu_idle(node: str, threshold: int = 10) -> bool:
    """Check if GPU utilization is below threshold."""
    cfg = NODES.get(node)
    if not cfg:
        log(f"Unknown node: {node}")
        return False

    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", f"{cfg['ssh_user']}@{cfg['ssh_host']}", cfg["gpu_check"]],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            log(f"GPU check failed for {node}: {result.stderr.strip()}")
            return False

        util = int(result.stdout.strip().split()[0])
        log(f"{node} GPU utilization: {util}%")
        return util < threshold
    except Exception as e:
        log(f"GPU check error for {node}: {e}")
        return False


def check_training_running(node: str) -> bool:
    """Check if any training process is running on the node."""
    cfg = NODES.get(node)
    if not cfg:
        return False

    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", f"{cfg['ssh_user']}@{cfg['ssh_host']}",
             "ps aux | grep -E 'python.*train' | grep -v grep | wc -l"],
            capture_output=True, text=True, timeout=15
        )
        count = int(result.stdout.strip())
        return count > 0
    except:
        return False


def launch_experiment(node: str, experiment: dict) -> bool:
    """Launch an experiment on a node via SSH."""
    cfg = NODES.get(node)
    if not cfg:
        return False

    cmd = experiment["command"]
    log_file = experiment.get("log_file", f"output/{experiment['id']}.log")

    ssh_cmd = f"cd /home/{cfg['ssh_user']}/Lvl3Quant && nohup {cmd} > {log_file} 2>&1 &"

    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", f"{cfg['ssh_user']}@{cfg['ssh_host']}", ssh_cmd],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log(f"LAUNCHED {experiment['id']} on {node}: {experiment['description']}")
            return True
        else:
            log(f"LAUNCH FAILED {experiment['id']}: {result.stderr.strip()}")
            return False
    except Exception as e:
        log(f"LAUNCH ERROR {experiment['id']}: {e}")
        return False


def send_discord(msg: str):
    """Send notification to Discord via webhook."""
    if not DISCORD_WEBHOOK:
        return
    try:
        import urllib.request
        data = json.dumps({"content": msg}).encode()
        req = urllib.request.Request(DISCORD_WEBHOOK, data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except:
        pass


def main():
    dry_run = "--dry-run" in sys.argv
    target_node = None
    for arg in sys.argv[1:]:
        if arg.startswith("--node="):
            target_node = arg.split("=")[1]
        elif arg in NODES:
            target_node = arg

    if not QUEUE_FILE.exists():
        log("No queue file found")
        return

    with open(QUEUE_FILE) as f:
        queue_data = json.load(f)

    queued = [e for e in queue_data["queue"] if e["status"] == "queued"]
    if not queued:
        log("No queued experiments")
        return

    for exp in queued:
        node = exp["node"]
        if target_node and node != target_node:
            continue

        if node == "razer":
            log(f"Skipping {exp['id']} — Razer is LIVE HOST, no training")
            continue

        # Check if GPU is idle and no training running
        if not check_gpu_idle(node):
            log(f"{node} GPU busy, skipping {exp['id']}")
            continue

        if check_training_running(node):
            log(f"{node} has training running, skipping {exp['id']}")
            continue

        log(f"{'[DRY RUN] Would launch' if dry_run else 'Launching'} {exp['id']} on {node}")

        if not dry_run:
            if launch_experiment(node, exp):
                exp["status"] = "running"
                exp["launched_at"] = datetime.now().isoformat()
                with open(QUEUE_FILE, "w") as f:
                    json.dump(queue_data, f, indent=2)
                send_discord(f"🚀 Auto-dispatched **{exp['id']}** on {node}: {exp['description']}")
                break  # One at a time per node
            else:
                log(f"Failed to launch {exp['id']}")

    log("Dispatch check complete")


if __name__ == "__main__":
    main()
