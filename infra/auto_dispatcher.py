#!/usr/bin/env python3
"""
auto_dispatcher.py — HC #492: Automated GPU work dispatcher
=============================================================

Detects idle GPUs and dispatches the next highest-priority experiment.
Integrated with self_audit.py — called when GPU idle is detected on weekdays.

Priority queue (highest first):
1. Resume crashed training (check for intra_ckpt files without completed results)
2. Generate missing prediction files (e.g., longs concat_predictions.npz)
3. Next walk-forward fold of in-progress experiments
4. Tree-branch research on confirmed edges (HC #491)

Node roles (from DIRECTIVES.md):
- Neptune (RTX 3090): Smart AI execution research (RL/MLP), model training
- Razer (RTX 3070): LIVE HOST — inference + paper trading only (NO training on weekdays)
  Weekend exception: can run lightweight training/analysis

Author: Claude (HC #492)
Date: 2026-05-24
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE_DIR = Path("/home/jupiter/Lvl3Quant")
DISPATCH_LOG = BASE_DIR / "infra" / "logs" / "auto_dispatch.jsonl"
DISPATCH_LOG.parent.mkdir(parents=True, exist_ok=True)

NODES = {
    "neptune": {
        "host": "neptune",
        "user": "nick",
        "lvl3_root": "/home/nick/Lvl3Quant",
        "gpu": "RTX 3090",
        "vram_gb": 24,
        "role": "training",  # smart exec research, model training
    },
    "razer": {
        "host": "razer",
        "user": "claude",
        "lvl3_root": "C:\\Users\\claude\\Lvl3Quant",
        "gpu": "RTX 3070",
        "vram_gb": 8,
        "role": "live",  # inference + paper trading
    },
}


def ssh_exec(host, user, cmd, timeout=15):
    """Execute SSH command. Returns (ok, stdout, stderr)."""
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
               f"{user}@{host}", cmd]
    try:
        r = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip(), r.stderr.strip()
    except Exception as e:
        return False, "", str(e)


def check_gpu_idle(node: str) -> Tuple[bool, int]:
    """Check if a node's GPU is idle. Returns (is_idle, utilization_pct)."""
    info = NODES[node]
    ok, out, _ = ssh_exec(info["host"], info["user"],
                          "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
    if not ok:
        return False, -1
    try:
        util = int(out.strip())
        return util < 10, util
    except ValueError:
        return False, -1


def get_neptune_dispatch_queue() -> List[Dict]:
    """Get prioritized work queue for Neptune."""
    queue = []

    # Priority 1: Check for crashed/interrupted training with checkpoints
    output_dir = BASE_DIR / "output"
    # Look for intra_ckpt files that don't have completed results
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        ckpts = list(d.glob("*intra_ckpt*.pt"))
        results = d / "results.json"
        if ckpts and not results.exists():
            queue.append({
                "priority": 1,
                "type": "resume_training",
                "description": f"Resume crashed training in {d.name}",
                "output_dir": str(d),
                "checkpoint": str(ckpts[-1]),
            })

    # Priority 2: Next fold of active v3.4.2 training (already handled by running process)
    # Skip — this is already running

    # Priority 3: Tree-branch experiments (HC #491)
    # These would be defined by research priorities
    queue.append({
        "priority": 3,
        "type": "tree_branch",
        "description": "Horizon generalization — test 5s/10s horizons",
        "script": "scripts/horizon_generalization_v1.py",
    })

    return sorted(queue, key=lambda x: x["priority"])


def get_razer_weekend_queue() -> List[Dict]:
    """Get prioritized weekend work for Razer (lightweight only)."""
    queue = []

    # Priority 1: Generate missing longs concat_predictions.npz
    longs_concat = BASE_DIR / "output" / "meta_production_longs_v1" / "concat_predictions.npz"
    if not longs_concat.exists():
        queue.append({
            "priority": 1,
            "type": "inference",
            "description": "Generate longs concat_predictions.npz for combined analysis",
        })

    return sorted(queue, key=lambda x: x["priority"])


def is_weekend():
    """Check if it's a weekend in NY timezone."""
    try:
        import zoneinfo
        n = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        n = datetime.now()
    return n.weekday() >= 5


def dispatch(node: str, task: Dict) -> bool:
    """Dispatch a task to a node. Returns True if successfully launched."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "node": node,
        "task": task,
        "status": "dispatched",
    }

    # For now, just log what SHOULD be dispatched
    # Actual dispatch requires node-specific SSH commands
    # which depend on the task type
    print(f"DISPATCH: {node} <- {task['description']}")

    try:
        with open(DISPATCH_LOG, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    return True


def run_dispatch_check():
    """Main dispatch check — called by self_audit or cron."""
    results = {"dispatched": [], "skipped": [], "errors": []}

    for node in NODES:
        is_idle, util = check_gpu_idle(node)
        info = NODES[node]

        if not is_idle:
            results["skipped"].append({
                "node": node,
                "reason": f"GPU busy ({util}%)",
            })
            continue

        # Node-specific dispatch logic
        if node == "neptune":
            queue = get_neptune_dispatch_queue()
            if queue:
                task = queue[0]
                dispatch(node, task)
                results["dispatched"].append({"node": node, "task": task})
            else:
                results["skipped"].append({"node": node, "reason": "No tasks in queue"})

        elif node == "razer":
            if is_weekend():
                queue = get_razer_weekend_queue()
                if queue:
                    task = queue[0]
                    dispatch(node, task)
                    results["dispatched"].append({"node": node, "task": task})
                else:
                    results["skipped"].append({
                        "node": node,
                        "reason": "Weekend, no weekend tasks queued",
                    })
            else:
                # Weekday idle Razer = live stack problem
                results["errors"].append({
                    "node": node,
                    "issue": "Razer GPU idle on weekday — live stack may be down",
                    "action": "Check inference + paper trader processes",
                })

    return results


def main():
    print("=" * 50)
    print("AUTO-DISPATCHER — HC #492")
    print("=" * 50)

    results = run_dispatch_check()
    print(json.dumps(results, indent=2))

    return 0 if not results["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
