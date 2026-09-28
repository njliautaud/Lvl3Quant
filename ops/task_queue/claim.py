#!/usr/bin/env python3
"""claim.py — Atomically claim the next eligible pending task for a node.

Eligibility rules (evaluated under exclusive flock):
  * status == "pending"
  * node in node_preference  (OR node_preference is empty == any)
  * gpu_required is False, OR caller passed --has-gpu
  * all task IDs in depends_on are status == "done"

Tie-break order: priority asc, created asc.

Output on success: prints the full task JSON to stdout. Sidecar file
`claimed/<id>.json` is created with claimer + claim_ts (so a crashed claimer
can be detected and the task released).

Exit 0 on claim, 1 on nothing-eligible, 2 on error.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

from _lib import (
    append_completed,
    claimed_sidecar_path,
    now_iso,
    queue_lock,
    read_all_tasks,
    write_all_tasks,
)


def eligible(task: dict, node: str, has_gpu: bool, done_ids: set[str]) -> bool:
    if task.get("status") != "pending":
        return False
    prefs = task.get("node_preference") or []
    if prefs and node not in prefs:
        return False
    if task.get("gpu_required") and not has_gpu:
        return False
    for dep in task.get("depends_on") or []:
        if dep not in done_ids:
            return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--node", required=True, choices=["jupiter", "neptune", "saturn", "razer"])
    p.add_argument("--has-gpu", action="store_true", help="Caller advertises GPU availability")
    p.add_argument("--worker-id", default=None,
                   help="Optional worker identifier (default: hostname:pid)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be claimed but do not mutate state")
    args = p.parse_args()

    worker_id = args.worker_id or f"{socket.gethostname()}:{__import__('os').getpid()}"

    with queue_lock(exclusive=True):
        tasks = read_all_tasks()
        done_ids = {t["id"] for t in tasks if t.get("status") == "done"}
        candidates = [t for t in tasks if eligible(t, args.node, args.has_gpu, done_ids)]
        if not candidates:
            print(json.dumps({"claimed": None, "reason": "no eligible tasks"}))
            return 1

        candidates.sort(key=lambda t: (int(t.get("priority", 5)), t.get("created", "")))
        chosen = candidates[0]

        if args.dry_run:
            print(json.dumps({"would_claim": chosen["id"], "task": chosen}))
            return 0

        chosen["status"] = "claimed"
        chosen["claimed_by"] = worker_id
        chosen["claimed_at"] = now_iso()

        # Rewrite tasks
        for i, t in enumerate(tasks):
            if t["id"] == chosen["id"]:
                tasks[i] = chosen
                break
        write_all_tasks(tasks)

        # Sidecar
        sidecar = claimed_sidecar_path(chosen["id"])
        with sidecar.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "id": chosen["id"],
                    "claimed_by": worker_id,
                    "claimed_at": chosen["claimed_at"],
                    "node": args.node,
                },
                fh,
                indent=2,
            )

        append_completed({
            "id": chosen["id"], "event": "claimed",
            "ts": chosen["claimed_at"], "by": worker_id, "node": args.node,
        })

    print(json.dumps({"claimed": chosen["id"], "task": chosen}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
