#!/usr/bin/env python3
"""
Autonomous Training Coordinator
================================
Keeps GPUs busy with experiments from research queue.
Monitors progress, kills underperformers, launches new experiments.

Run as: python3 utils/autonomous_trainer.py &
"""

import time
import subprocess
import json
import psutil
from pathlib import Path
from datetime import datetime

# Research queue - UNTESTED experiments (priority order)
RESEARCH_QUEUE = [
    {
        "name": "Event_3D_CNN",
        "script": "alpha_discovery/deep_models/train_event_3d_cnn.py",
        "args": "--n_files 60 --batch_size 16 --epochs_per_fold 8 --n_folds 5",
        "min_ic": 0.08,
        "needs_gpu": True
    },
    {
        "name": "Wider_EventCNN",
        "script": "alpha_discovery/deep_models/train_event_cnn_1d.py",
        "args": "--n_files 60 --batch_size 256 --epochs_per_fold 10 --n_folds 5 --cnn_channels 256 --cnn_layers 8",
        "min_ic": 0.12,
        "needs_gpu": False
    },
    {
        "name": "Event_Transformer_Large",
        "script": "alpha_discovery/deep_models/train_event_transformer_fast.py",
        "args": "--n_files 60 --batch_size 64 --epochs_per_fold 8 --n_folds 5 --window_size 1000",
        "min_ic": 0.10,
        "needs_gpu": True
    },
    {
        "name": "LGBM_Simple_Features",
        "script": "alpha_discovery/lgbm_confidence_ic.py",
        "args": "",
        "min_ic": 0.10,
        "needs_gpu": False
    }
]

ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_FILE = ROOT / "data" / "autonomous_trainer_state.json"

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"completed": [], "running": {}, "last_check": None}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def check_process_alive(pid):
    try:
        return psutil.pid_exists(pid)
    except:
        return False

def get_idle_resources():
    """Check which compute resources are idle"""
    # Check CPU usage
    cpu_pct = psutil.cpu_percent(interval=1)
    cpu_idle = cpu_pct < 50

    # Check GPU (if available)
    gpu_idle = True  # TODO: Add nvidia-smi check

    return {"cpu": cpu_idle, "gpu": gpu_idle}

def launch_experiment(exp):
    """Launch an experiment from queue"""
    cmd = f"cd {ROOT} && nohup python3 -u {exp['script']} {exp['args']} > /tmp/{exp['name']}_{datetime.now().strftime('%Y%m%d_%H%M')}.log 2>&1 &"

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    time.sleep(2)

    # Find the actual python PID
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in p.info['name'] and exp['script'] in ' '.join(p.info['cmdline'] or []):
                return p.info['pid']
        except:
            continue

    return None

def main():
    print("🤖 AUTONOMOUS TRAINER STARTING")
    print("=" * 60)

    state = load_state()

    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking cluster status...")

        # Clean up dead processes
        for name, pid in list(state["running"].items()):
            if not check_process_alive(pid):
                print(f"  ⚰️  {name} (PID {pid}) died - removing from running")
                del state["running"][name]

        # Check idle resources
        idle = get_idle_resources()
        print(f"  Resources: CPU={'IDLE' if idle['cpu'] else 'BUSY'}, GPU={'IDLE' if idle['gpu'] else 'BUSY'}")

        # Launch new experiments if resources available
        for exp in RESEARCH_QUEUE:
            if exp["name"] in state["completed"]:
                continue
            if exp["name"] in state["running"]:
                continue

            # Check if we have resources for this experiment
            can_launch = False
            if exp["needs_gpu"] and idle["gpu"]:
                can_launch = True
            elif not exp["needs_gpu"] and idle["cpu"]:
                can_launch = True

            if can_launch:
                print(f"  🚀 Launching: {exp['name']}")
                pid = launch_experiment(exp)
                if pid:
                    state["running"][exp["name"]] = pid
                    print(f"     PID: {pid}")
                    save_state(state)
                    break  # Only launch one at a time

        # Status report
        print(f"  Running: {len(state['running'])} experiments")
        print(f"  Completed: {len(state['completed'])} experiments")

        state["last_check"] = datetime.now().isoformat()
        save_state(state)

        # Check every 15 minutes
        time.sleep(900)

if __name__ == "__main__":
    main()
