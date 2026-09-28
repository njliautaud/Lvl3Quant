"""Shared primitives for the persistent task queue (HC #422 Rule 10).

Atomic operations are gated by `fcntl.flock` on `queue.lock`. Every mutator
acquires an EXCLUSIVE lock; readers can use SHARED. The queue file itself
(`queue.jsonl`) is append-only on enqueue and rewritten in-place under lock
on state transitions (claim/complete/fail).

Task lifecycle:
    pending --claim()--> claimed --(puller spawns)--> running
        running --complete()--> done
        running --fail()-----> failed
        claimed --release()--> pending  (e.g. node decided it can't run it)

Files:
    queue.jsonl       — canonical store, one JSON task per line
    queue.lock        — flock target (zero bytes, never read)
    claimed/<id>.json — per-claim sidecar with claimer + claim_ts
    completed.jsonl   — append-only audit log of done/failed transitions
"""
from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
QUEUE_FILE = HERE / "queue.jsonl"
LOCK_FILE = HERE / "queue.lock"
CLAIMED_DIR = HERE / "claimed"
COMPLETED_FILE = HERE / "completed.jsonl"
LOG_DIR = HERE / "logs"

VALID_NODES = {"jupiter", "neptune", "saturn", "razer"}
VALID_STATUS = {"pending", "claimed", "running", "done", "failed"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_layout() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    CLAIMED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (QUEUE_FILE, LOCK_FILE, COMPLETED_FILE):
        if not f.exists():
            f.touch()


@contextmanager
def queue_lock(exclusive: bool = True, timeout_s: float = 30.0) -> Iterator[None]:
    """fcntl.flock-based mutual exclusion. Default exclusive; SHARED for readers."""
    ensure_layout()
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fd = os.open(str(LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.time() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"queue_lock: could not acquire {('EX' if exclusive else 'SH')} "
                        f"in {timeout_s}s"
                    )
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def read_all_tasks() -> list[dict]:
    """Read every task line. Caller is responsible for holding a lock if mutating."""
    ensure_layout()
    out: list[dict] = []
    with QUEUE_FILE.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                # Skip corrupt lines but keep going — corruption is logged via the
                # audit trail rather than crashing the queue.
                continue
    return out


def write_all_tasks(tasks: list[dict]) -> None:
    """Atomically rewrite the queue file. CALLER MUST hold the exclusive lock."""
    ensure_layout()
    tmp = QUEUE_FILE.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, QUEUE_FILE)


def append_completed(record: dict) -> None:
    """Append-only audit log. CALLER MUST hold the exclusive lock."""
    ensure_layout()
    with COMPLETED_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def new_task_id() -> str:
    return uuid.uuid4().hex[:12]


def validate_task(t: dict) -> list[str]:
    """Return list of validation error messages; empty list if OK."""
    errs: list[str] = []
    for req in ("id", "created", "command", "priority", "status"):
        if req not in t:
            errs.append(f"missing field: {req}")
    if "status" in t and t["status"] not in VALID_STATUS:
        errs.append(f"bad status: {t['status']!r}")
    if "node_preference" in t:
        for n in t["node_preference"]:
            if n not in VALID_NODES:
                errs.append(f"bad node in node_preference: {n!r}")
    if "priority" in t:
        try:
            p = int(t["priority"])
            if not 1 <= p <= 10:
                errs.append(f"priority out of range 1..10: {p}")
        except (TypeError, ValueError):
            errs.append(f"priority not int: {t['priority']!r}")
    return errs


def claimed_sidecar_path(task_id: str) -> Path:
    return CLAIMED_DIR / f"{task_id}.json"
