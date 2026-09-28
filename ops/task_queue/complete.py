#!/usr/bin/env python3
"""complete.py — Mark a claimed/running task done (or failed).

Usage:
    python complete.py --id <task_id> --exit-code 0 [--log-path /path/to/log]
    python complete.py --id <task_id> --exit-code 137 --status failed \
        --error "OOM killed"

Mutates the task record under flock; appends an event to completed.jsonl;
removes the per-claim sidecar.
"""
from __future__ import annotations

import argparse
import json
import sys

from _lib import (
    append_completed,
    claimed_sidecar_path,
    now_iso,
    queue_lock,
    read_all_tasks,
    write_all_tasks,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--exit-code", type=int, default=0)
    p.add_argument("--status", choices=["done", "failed"], default=None,
                   help="Override status; default infers from exit-code (0=done, else failed)")
    p.add_argument("--log-path", default=None)
    p.add_argument("--error", default=None)
    args = p.parse_args()

    new_status = args.status or ("done" if args.exit_code == 0 else "failed")

    with queue_lock(exclusive=True):
        tasks = read_all_tasks()
        idx = next((i for i, t in enumerate(tasks) if t["id"] == args.id), None)
        if idx is None:
            print(f"ERROR: task not found: {args.id}", file=sys.stderr)
            return 2
        t = tasks[idx]
        t["status"] = new_status
        t["finished_at"] = now_iso()
        t["exit_code"] = args.exit_code
        if args.log_path:
            t["log_path"] = args.log_path
        if args.error:
            t["notes"] = (t.get("notes", "") + f"\nERROR: {args.error}").strip()
        tasks[idx] = t
        write_all_tasks(tasks)

        append_completed({
            "id": t["id"], "event": new_status, "ts": t["finished_at"],
            "exit_code": args.exit_code, "log_path": args.log_path,
            "error": args.error,
        })

        sidecar = claimed_sidecar_path(t["id"])
        if sidecar.exists():
            sidecar.unlink()

    print(json.dumps({"id": t["id"], "status": new_status, "exit_code": args.exit_code}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
