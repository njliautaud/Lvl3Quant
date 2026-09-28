#!/usr/bin/env python3
"""enqueue.py — Add a task to the persistent queue (atomic via fcntl.flock).

Usage:
    python enqueue.py --command "python foo.py" --node-preference jupiter \
        --priority 3 --gpu-required false --max-runtime-s 7200 \
        --working-dir /home/jupiter/Lvl3Quant --tag rules_sweep

    python enqueue.py --from-json /path/to/task.json
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from _lib import (
    new_task_id,
    now_iso,
    queue_lock,
    read_all_tasks,
    validate_task,
    write_all_tasks,
)


def build_task(args: argparse.Namespace) -> dict:
    if args.from_json:
        with Path(args.from_json).open("r", encoding="utf-8") as fh:
            spec = json.load(fh)
    else:
        spec = {}

    task = {
        "id": spec.get("id", new_task_id()),
        "created": spec.get("created", now_iso()),
        "created_by": spec.get("created_by", f"{socket.gethostname()}:{Path(sys.argv[0]).name}"),
        "node_preference": spec.get("node_preference", args.node_preference or ["jupiter"]),
        "gpu_required": spec.get("gpu_required", args.gpu_required),
        "priority": spec.get("priority", args.priority),
        "command": spec.get("command", args.command or ""),
        "working_dir": spec.get("working_dir", args.working_dir or "/home/jupiter/Lvl3Quant"),
        "env": spec.get("env", {}),
        "max_runtime_s": spec.get("max_runtime_s", args.max_runtime_s),
        "depends_on": spec.get("depends_on", args.depends_on or []),
        "tags": spec.get("tags", args.tag or []),
        "status": "pending",
        "claimed_by": None,
        "claimed_at": None,
        "started_at": None,
        "finished_at": None,
        "exit_code": None,
        "log_path": None,
        "notes": spec.get("notes", args.notes or ""),
    }
    errs = validate_task(task)
    if errs:
        print("ERROR: invalid task:", *errs, sep="\n  ", file=sys.stderr)
        sys.exit(2)
    return task


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--command", help="Shell command to execute")
    p.add_argument(
        "--node-preference",
        nargs="+",
        choices=["jupiter", "neptune", "saturn", "razer"],
        help="Ordered list of preferred nodes",
    )
    p.add_argument("--gpu-required", type=lambda s: s.lower() == "true", default=False)
    p.add_argument("--priority", type=int, default=5, help="1=highest, 10=lowest")
    p.add_argument("--max-runtime-s", type=int, default=7200)
    p.add_argument("--working-dir", default="/home/jupiter/Lvl3Quant")
    p.add_argument("--depends-on", nargs="*", default=[], help="task IDs that must be done first")
    p.add_argument("--tag", nargs="*", default=[])
    p.add_argument("--notes", default="")
    p.add_argument("--from-json", help="Read task spec from a JSON file (overrides flags)")
    args = p.parse_args()

    if not args.command and not args.from_json:
        p.error("--command (or --from-json) is required")

    task = build_task(args)
    with queue_lock(exclusive=True):
        tasks = read_all_tasks()
        if any(t.get("id") == task["id"] for t in tasks):
            print(f"ERROR: task id collision: {task['id']}", file=sys.stderr)
            return 2
        tasks.append(task)
        write_all_tasks(tasks)

    print(json.dumps({"enqueued": task["id"], "priority": task["priority"],
                      "node_preference": task["node_preference"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
