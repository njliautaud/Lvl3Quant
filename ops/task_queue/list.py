#!/usr/bin/env python3
"""list.py — Read-only view over the task queue.

Examples:
    python list.py                      # counts by status
    python list.py --status pending     # full JSON of pending tasks
    python list.py --node jupiter
    python list.py --json               # all tasks as JSON array
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from _lib import queue_lock, read_all_tasks


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--status", choices=["pending", "claimed", "running", "done", "failed"])
    p.add_argument("--node", choices=["jupiter", "neptune", "saturn", "razer"])
    p.add_argument("--json", action="store_true", help="Emit raw JSON instead of summary")
    p.add_argument("--limit", type=int, default=50)
    args = p.parse_args()

    with queue_lock(exclusive=False):
        tasks = read_all_tasks()

    if args.node:
        tasks = [
            t for t in tasks
            if (not t.get("node_preference")) or args.node in t.get("node_preference", [])
        ]
    if args.status:
        tasks = [t for t in tasks if t.get("status") == args.status]

    if args.json:
        print(json.dumps(tasks[: args.limit], indent=2))
        return 0

    counts = Counter(t.get("status", "?") for t in tasks)
    print("=== task_queue summary ===")
    for s in ("pending", "claimed", "running", "done", "failed"):
        print(f"  {s:8s} {counts.get(s, 0)}")
    print(f"  total    {len(tasks)}")
    print()
    print("=== top by priority (pending, first 20) ===")
    pending = sorted(
        (t for t in tasks if t.get("status") == "pending"),
        key=lambda t: (int(t.get("priority", 5)), t.get("created", "")),
    )
    for t in pending[:20]:
        prefs = ",".join(t.get("node_preference") or ["any"])
        print(f"  P{t.get('priority'):>2}  {t['id']}  [{prefs:<20s}]  "
              f"gpu={'Y' if t.get('gpu_required') else 'N'}  "
              f"{(t.get('command') or '')[:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
