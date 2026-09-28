#!/usr/bin/env python3
"""
task_queue.py — Persistent, on-disk, append-only task queue for cross-session
autonomy. Survives Claude context resets, PM2 restarts, machine reboots.

Storage:
  /home/jupiter/Lvl3Quant/ops/task_queue.jsonl  (append-only JSON Lines)

Each line is one task event/state record:
  {
    "id":           "<uuid4-hex>",
    "created_at":   "<iso8601-utc>",
    "node":         "jupiter|neptune|saturn|razer",
    "kind":         "training|backtest|audit|data|exec|scaffold|...",
    "cmd":          "<shell command to run>",
    "priority":     1-10  (lower = higher priority),
    "status":       "pending|running|done|failed|cancelled",
    "started_at":   "<iso8601-utc>" or null,
    "finished_at":  "<iso8601-utc>" or null,
    "log_path":     "<path to stdout/stderr log>" or null,
    "error":        "<error string>" or null,
    "tags":         [...]
  }

Because the file is append-only, mutations are written as REPLACEMENT records
keyed by `id`. The current state of any task is the LAST line with that id.

All multi-step mutations (claim-next / atomic claim) use fcntl.flock for
cross-process safety. Multiple node-side pullers may safely race.

CLI:
  task_queue.py enqueue --node N --kind K --cmd "..." [--priority P] [--tag t1 t2]
  task_queue.py next    --node N            # atomically claim next pending for node N
  task_queue.py done    --id ID [--log PATH]
  task_queue.py fail    --id ID --error "..."
  task_queue.py list    [--node N] [--status S]
  task_queue.py show    --id ID

Exit codes:
  0 success
  1 nothing-to-do (e.g. `next` with empty queue)
  2 error
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

OPS_DIR = Path("/home/jupiter/Lvl3Quant/ops")
QUEUE_FILE = OPS_DIR / "task_queue.jsonl"
LOCK_FILE = OPS_DIR / "task_queue.lock"
LOG_DIR = OPS_DIR / "logs"

VALID_NODES = {"jupiter", "neptune", "saturn", "razer", "any"}
VALID_STATUS = {"pending", "running", "done", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not QUEUE_FILE.exists():
        QUEUE_FILE.touch()
    if not LOCK_FILE.exists():
        LOCK_FILE.touch()


# ──────────────────────────────────────────────────────────────────────────────
# Locking
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def _flock() -> Iterator[None]:
    """Exclusive cross-process lock on LOCK_FILE."""
    _ensure_dirs()
    fd = os.open(str(LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ──────────────────────────────────────────────────────────────────────────────
# Read / Write
# ──────────────────────────────────────────────────────────────────────────────

def _append_record(rec: dict) -> None:
    _ensure_dirs()
    line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
    with open(QUEUE_FILE, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _read_all_states() -> dict[str, dict]:
    """Return current state for every task (last record per id)."""
    _ensure_dirs()
    states: dict[str, dict] = {}
    with open(QUEUE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = rec.get("id")
            if not tid:
                continue
            states[tid] = rec
    return states


# ──────────────────────────────────────────────────────────────────────────────
# Core API
# ──────────────────────────────────────────────────────────────────────────────

def enqueue(
    node: str,
    kind: str,
    cmd: str,
    priority: int = 5,
    tags: list[str] | None = None,
) -> dict:
    if node not in VALID_NODES:
        raise ValueError(f"node must be one of {VALID_NODES}, got {node!r}")
    if not (1 <= priority <= 10):
        raise ValueError("priority must be in [1,10]")
    tid = uuid.uuid4().hex[:12]
    rec = {
        "id":           tid,
        "created_at":   _now(),
        "node":         node,
        "kind":         kind,
        "cmd":          cmd,
        "priority":     int(priority),
        "status":       "pending",
        "started_at":   None,
        "finished_at":  None,
        "log_path":     None,
        "error":        None,
        "tags":         list(tags or []),
    }
    with _flock():
        _append_record(rec)
    return rec


def claim_next(node: str) -> dict | None:
    """
    Atomically claim the highest-priority pending task for `node`.
    A task with node=='any' is also eligible for any caller.
    Returns the claimed record (status=running) or None if nothing pending.
    """
    if node not in VALID_NODES:
        raise ValueError(f"node must be one of {VALID_NODES}, got {node!r}")
    with _flock():
        states = _read_all_states()
        candidates = [
            r for r in states.values()
            if r.get("status") == "pending"
            and (r.get("node") == node or r.get("node") == "any")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda r: (r.get("priority", 5), r.get("created_at", "")))
        chosen = dict(candidates[0])
        chosen["status"] = "running"
        chosen["started_at"] = _now()
        chosen["claimed_by_node"] = node
        chosen["claimed_pid"] = os.getpid()
        _append_record(chosen)
        return chosen


def mark_done(task_id: str, log_path: str | None = None) -> dict:
    with _flock():
        states = _read_all_states()
        if task_id not in states:
            raise KeyError(f"task {task_id!r} not found")
        rec = dict(states[task_id])
        rec["status"] = "done"
        rec["finished_at"] = _now()
        if log_path:
            rec["log_path"] = log_path
        _append_record(rec)
        return rec


def mark_failed(task_id: str, error: str, log_path: str | None = None) -> dict:
    with _flock():
        states = _read_all_states()
        if task_id not in states:
            raise KeyError(f"task {task_id!r} not found")
        rec = dict(states[task_id])
        rec["status"] = "failed"
        rec["finished_at"] = _now()
        rec["error"] = error
        if log_path:
            rec["log_path"] = log_path
        _append_record(rec)
        return rec


def cancel(task_id: str) -> dict:
    with _flock():
        states = _read_all_states()
        if task_id not in states:
            raise KeyError(f"task {task_id!r} not found")
        rec = dict(states[task_id])
        rec["status"] = "cancelled"
        rec["finished_at"] = _now()
        _append_record(rec)
        return rec


def list_tasks(node: str | None = None, status: str | None = None) -> list[dict]:
    states = _read_all_states()
    out = list(states.values())
    if node:
        out = [r for r in out if r.get("node") == node]
    if status:
        out = [r for r in out if r.get("status") == status]
    out.sort(key=lambda r: (
        0 if r.get("status") == "pending" else
        1 if r.get("status") == "running" else
        2 if r.get("status") == "failed" else
        3,
        r.get("priority", 5),
        r.get("created_at", ""),
    ))
    return out


def show(task_id: str) -> dict | None:
    states = _read_all_states()
    return states.get(task_id)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_row(r: dict) -> str:
    return (
        f"{r.get('id','????????'):<12} "
        f"p{r.get('priority',5):<2} "
        f"{r.get('node',''):<8} "
        f"{r.get('kind',''):<12} "
        f"{r.get('status',''):<10} "
        f"{(r.get('cmd','') or '')[:80]}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="task_queue", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)

    pe = sub.add_parser("enqueue", help="Append a new pending task")
    pe.add_argument("--node", required=True, choices=sorted(VALID_NODES))
    pe.add_argument("--kind", required=True)
    pe.add_argument("--cmd", required=True)
    pe.add_argument("--priority", type=int, default=5)
    pe.add_argument("--tag", nargs="*", default=[])

    pn = sub.add_parser("next", help="Atomically claim next pending task for a node")
    pn.add_argument("--node", required=True, choices=sorted(VALID_NODES))

    pd = sub.add_parser("done", help="Mark a task done")
    pd.add_argument("--id", required=True)
    pd.add_argument("--log", default=None)

    pf = sub.add_parser("fail", help="Mark a task failed")
    pf.add_argument("--id", required=True)
    pf.add_argument("--error", required=True)
    pf.add_argument("--log", default=None)

    pc = sub.add_parser("cancel", help="Cancel a task")
    pc.add_argument("--id", required=True)

    pl = sub.add_parser("list", help="List tasks")
    pl.add_argument("--node", default=None, choices=sorted(VALID_NODES))
    pl.add_argument("--status", default=None, choices=sorted(VALID_STATUS))

    ps = sub.add_parser("show", help="Show full record for one task id")
    ps.add_argument("--id", required=True)

    args = p.parse_args(argv)

    try:
        if args.action == "enqueue":
            rec = enqueue(args.node, args.kind, args.cmd,
                          priority=args.priority, tags=args.tag)
            print(json.dumps(rec, indent=2, default=str))
            return 0

        if args.action == "next":
            rec = claim_next(args.node)
            if rec is None:
                print("", end="")
                return 1
            print(json.dumps(rec, indent=2, default=str))
            return 0

        if args.action == "done":
            rec = mark_done(args.id, log_path=args.log)
            print(json.dumps(rec, indent=2, default=str))
            return 0

        if args.action == "fail":
            rec = mark_failed(args.id, args.error, log_path=args.log)
            print(json.dumps(rec, indent=2, default=str))
            return 0

        if args.action == "cancel":
            rec = cancel(args.id)
            print(json.dumps(rec, indent=2, default=str))
            return 0

        if args.action == "list":
            rows = list_tasks(node=args.node, status=args.status)
            if not rows:
                print("(empty)")
                return 0
            print(f"{'id':<12} {'pri':<3} {'node':<8} {'kind':<12} {'status':<10} cmd")
            for r in rows:
                print(_fmt_row(r))
            return 0

        if args.action == "show":
            rec = show(args.id)
            if rec is None:
                print(f"task {args.id!r} not found", file=sys.stderr)
                return 2
            print(json.dumps(rec, indent=2, default=str))
            return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    sys.exit(main())
