#!/usr/bin/env python3
"""Auto-dispatcher: checks GPU nodes, launches untested experiments from queue."""

import json
import subprocess
import sys
import time
from pathlib import Path

QUEUE_PATH = Path("/home/jupiter/Lvl3Quant/data/research_queue_persistent.json")
WEBHOOK_CMD = ["node", "/home/jupiter/teleclaude-main/utils/webhook_notifier.js"]
SCRIPTS_DIR = "alpha_discovery/deep_models"

NODES = {
    "neptune": {
        "ssh": "nick@neptune",
        "root": "/home/nick/Lvl3Quant",
        "launch_prefix": (
            "source ~/.bashrc && conda activate py311-train && "
            "cd /home/nick/Lvl3Quant/{scripts_dir} && "
            "DISABLE_MLFLOW=1 WF_WINDOW_DAYS=60 nohup python"
        ),
    },
    "razer": {
        "ssh": "claude@razer",
        "root": r"C:\Users\claude\Lvl3Quant",
        "launch_prefix": (
            'cd /d C:\\Users\\claude\\Lvl3Quant\\{scripts_dir} && '
            'set DISABLE_MLFLOW=1 && set WF_WINDOW_DAYS=60 && '
            'start /B C:\\Python311\\python.exe'
        ),
    },
}

# Map experiment names to their training scripts
SCRIPT_MAP = {
    "Event Mamba SSM": "train_event_mamba.py",
    "Wider EventCNN1D": "train_event_cnn1d.py",
    "Event 3D CNN": "train_event_3dcnn.py",
    "Hawkes Temporal Point Process": "train_hawkes_tpp.py",
    "Neural ODE event stream": "train_neural_ode.py",
    "Variable-resolution event encoder": "train_varres_encoder.py",
    "Event Transformer": "train_event_transformer.py",
}


def notify(msg: str):
    """Send Discord notification via webhook_notifier.js."""
    try:
        subprocess.run(WEBHOOK_CMD + [msg], capture_output=True, timeout=15)
    except Exception as e:
        print(f"[WARN] Webhook failed: {e}", file=sys.stderr)


def check_gpu(ssh_target: str) -> float:
    """SSH into node and return GPU utilization %. Returns -1 on failure."""
    cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
           ssh_target, "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception as e:
        print(f"[WARN] GPU check failed for {ssh_target}: {e}", file=sys.stderr)
    return -1.0


def check_training_procs(ssh_target: str, is_windows: bool = False) -> int:
    """Check if training python processes exist on remote node."""
    if is_windows:
        remote_cmd = 'tasklist /FI "IMAGENAME eq python.exe" /NH 2>nul | find /c "python"'
    else:
        remote_cmd = "ps aux | grep -E 'python.*train' | grep -v grep | wc -l"
    cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "BatchMode=yes", ssh_target, remote_cmd]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return -1  # unknown


def load_queue() -> dict:
    with open(QUEUE_PATH) as f:
        return json.load(f)


def save_queue(data: dict):
    from datetime import datetime, timezone
    data["last_updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(QUEUE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Queue saved to {QUEUE_PATH}")


def find_script(experiment_name: str) -> str | None:
    """Match experiment name to a script via prefix matching."""
    for prefix, script in SCRIPT_MAP.items():
        if experiment_name.startswith(prefix):
            return script
    return None


def launch_on_node(node_name: str, experiment: dict) -> bool:
    """Launch experiment on node via SSH. Returns True on success."""
    node = NODES[node_name]
    script = find_script(experiment["name"])
    if not script:
        print(f"[SKIP] No script mapping for: {experiment['name']}")
        notify(f"[auto_dispatch] No script for '{experiment['name']}' — skipping")
        return False

    prefix = node["launch_prefix"].format(scripts_dir=SCRIPTS_DIR)

    if node_name == "neptune":
        remote_cmd = f"{prefix} {script} > /tmp/{script}.log 2>&1 &"
        cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
               node["ssh"], remote_cmd]
    else:
        # Razer is Windows — use cmd /c
        remote_cmd = f'{prefix} {script} > C:\\tmp\\{script}.log 2>&1'
        cmd = ["ssh", "-o", "ConnectTimeout=10",
               node["ssh"], f'cmd /c "{remote_cmd}"']

    print(f"[LAUNCH] {experiment['name']} on {node_name}")
    print(f"  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0 and result.stderr.strip():
            print(f"[WARN] SSH stderr: {result.stderr.strip()}")
        return True
    except Exception as e:
        print(f"[ERROR] Launch failed: {e}", file=sys.stderr)
        return False


def main():
    print("=== Auto-Dispatcher ===")
    print(f"Checking GPU nodes...\n")

    # Also check queue for what's already marked as running on each node
    queue_data = load_queue()
    running_on = {e["node"] for e in queue_data["queue"] if e["status"] == "running" and e["node"]}

    idle_nodes = []
    for name, cfg in NODES.items():
        # Skip if queue says something is running here
        if name in running_on:
            print(f"  {name}: BUSY (queue says experiment running)")
            continue

        util1 = check_gpu(cfg["ssh"])
        if util1 < 0:
            print(f"  {name}: UNREACHABLE")
            continue

        # Check for actual training processes (GPU can be low during data loading)
        is_win = name == "razer"
        procs = check_training_procs(cfg["ssh"], is_windows=is_win)
        if procs > 0:
            print(f"  {name}: ACTIVE ({util1}% GPU, {procs} training procs)")
            continue

        # Double-check after 5s to confirm idle
        if util1 < 10:
            time.sleep(5)
            util2 = check_gpu(cfg["ssh"])
            procs2 = check_training_procs(cfg["ssh"], is_windows=is_win)
            if util2 < 10 and procs2 == 0:
                print(f"  {name}: IDLE ({util1}% -> {util2}%, 0 procs)")
                idle_nodes.append(name)
            else:
                print(f"  {name}: ACTIVE ({util1}% -> {util2}%, {procs2} procs)")
        else:
            print(f"  {name}: ACTIVE ({util1}%)")

    if not idle_nodes:
        print("\nNo idle nodes. Nothing to dispatch.")
        return

    untested = [e for e in queue_data["queue"] if e["status"] == "untested"]

    if not untested:
        print("\nNo untested experiments in queue.")
        return

    print(f"\n{len(untested)} untested experiments, {len(idle_nodes)} idle node(s)\n")

    for node_name in idle_nodes:
        if not untested:
            break
        experiment = untested.pop(0)
        ok = launch_on_node(node_name, experiment)
        if ok:
            experiment["status"] = "running"
            experiment["node"] = node_name
            save_queue(queue_data)
            notify(f"[auto_dispatch] Launched '{experiment['name']}' on {node_name}")
            print(f"[OK] Dispatched '{experiment['name']}' -> {node_name}\n")
        else:
            notify(f"[auto_dispatch] FAILED to launch '{experiment['name']}' on {node_name}")


if __name__ == "__main__":
    main()
