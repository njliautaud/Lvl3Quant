#!/usr/bin/env python3
"""Persistent SQLite task queue for the orchestrator (HC #594 P1-1).

Work items survive session restarts. Use from Python or via CLI:

    python3 task_queue.py list
    python3 task_queue.py add "task title" --priority 1 --payload '{"k":"v"}'
    python3 task_queue.py claim
    python3 task_queue.py done <id>
    python3 task_queue.py fail <id> [--error "msg"]

States: pending -> claimed -> done | failed (failed retries with backoff
until max_retries, then state=dead).
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "task_queue.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT NOT NULL,
    priority    INTEGER NOT NULL DEFAULT 5,          -- 0 = highest
    payload     TEXT,                                -- JSON blob
    state       TEXT NOT NULL DEFAULT 'pending',     -- pending|claimed|done|failed|dead
    retries     INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 5,
    not_before  REAL NOT NULL DEFAULT 0,             -- epoch; backoff gate
    last_error  TEXT,
    created_at  REAL NOT NULL,
    claimed_at  REAL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_tasks_state_prio ON tasks(state, priority, id);
"""

BACKOFF_BASE_S = 60  # 1m, 2m, 4m, 8m, ...


def _conn():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def enqueue(task, priority=5, payload=None, max_retries=5, dedupe=True):
    """Add a task. Returns task id. If dedupe, skip if an identical
    pending/claimed task title already exists (returns existing id)."""
    with _conn() as con:
        if dedupe:
            row = con.execute(
                "SELECT id FROM tasks WHERE task=? AND state IN ('pending','claimed','failed')",
                (task,),
            ).fetchone()
            if row:
                return row["id"]
        cur = con.execute(
            "INSERT INTO tasks (task, priority, payload, max_retries, created_at) VALUES (?,?,?,?,?)",
            (task, priority, json.dumps(payload) if payload is not None else None,
             max_retries, time.time()),
        )
        return cur.lastrowid


def claim():
    """Atomically claim the highest-priority eligible task. Returns dict or None."""
    now = time.time()
    with _conn() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT * FROM tasks WHERE state IN ('pending','failed') AND not_before<=? "
            "ORDER BY priority ASC, id ASC LIMIT 1",
            (now,),
        ).fetchone()
        if not row:
            return None
        con.execute(
            "UPDATE tasks SET state='claimed', claimed_at=? WHERE id=?",
            (now, row["id"]),
        )
        d = dict(row)
        d["state"] = "claimed"
        d["claimed_at"] = now
        if d.get("payload"):
            try:
                d["payload"] = json.loads(d["payload"])
            except (ValueError, TypeError):
                pass
        return d


def complete(task_id):
    """Mark a task done."""
    with _conn() as con:
        con.execute(
            "UPDATE tasks SET state='done', finished_at=? WHERE id=?",
            (time.time(), task_id),
        )


def fail(task_id, error=None):
    """Mark a task failed; schedules retry with exponential backoff.
    After max_retries the task goes to state='dead'."""
    with _conn() as con:
        row = con.execute("SELECT retries, max_retries FROM tasks WHERE id=?",
                          (task_id,)).fetchone()
        if not row:
            return None
        retries = row["retries"] + 1
        if retries > row["max_retries"]:
            con.execute(
                "UPDATE tasks SET state='dead', retries=?, last_error=?, finished_at=? WHERE id=?",
                (retries, error, time.time(), task_id),
            )
            return "dead"
        backoff = BACKOFF_BASE_S * (2 ** (retries - 1))
        con.execute(
            "UPDATE tasks SET state='failed', retries=?, last_error=?, not_before=? WHERE id=?",
            (retries, error, time.time() + backoff, task_id),
        )
        return "retry_in_%ds" % backoff


def list_pending(include_claimed=True):
    """Return open tasks (pending/failed, optionally claimed) as list of dicts."""
    states = ("pending", "failed", "claimed") if include_claimed else ("pending", "failed")
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM tasks WHERE state IN (%s) ORDER BY priority ASC, id ASC"
            % ",".join("?" * len(states)),
            states,
        ).fetchall()
    return [dict(r) for r in rows]


def seed_backlog():
    """Seed with known open backlog items from SESSION_STATE.md (idempotent)."""
    items = [
        ("P3-1 durable crons — make monitoring/dispatch crons survive reboots "
         "(persist definitions, restore on session start)", 3,
         {"source": "SESSION_STATE.md BACKLOG", "tag": "P3-1"}),
        ("P4-2 EVENT_TRIGGER game-filter leak — GPU busy/idle triggers fire on "
         "desktop/gaming GPU noise (Neptune Steam, Razer WebView); add process-aware "
         "filter before alerting", 4,
         {"source": "SESSION_STATE.md BACKLOG", "tag": "P4-2"}),
    ]
    ids = [enqueue(t, p, pl) for t, p, pl in items]
    return ids


def _cli():
    ap = argparse.ArgumentParser(description="Persistent orchestrator task queue")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list open tasks")

    p_add = sub.add_parser("add", help="enqueue a task")
    p_add.add_argument("task")
    p_add.add_argument("--priority", type=int, default=5)
    p_add.add_argument("--payload", default=None, help="JSON string")

    sub.add_parser("claim", help="claim next task")

    p_done = sub.add_parser("done", help="complete a task")
    p_done.add_argument("id", type=int)

    p_fail = sub.add_parser("fail", help="fail a task (retry w/ backoff)")
    p_fail.add_argument("id", type=int)
    p_fail.add_argument("--error", default=None)

    sub.add_parser("seed", help="seed backlog items (idempotent)")

    args = ap.parse_args()

    if args.cmd == "list":
        rows = list_pending()
        if not rows:
            print("(queue empty)")
        for r in rows:
            print(f"[{r['id']:>3}] p{r['priority']} {r['state']:<8} retries={r['retries']} {r['task'][:100]}")
    elif args.cmd == "add":
        payload = json.loads(args.payload) if args.payload else None
        tid = enqueue(args.task, args.priority, payload)
        print(f"enqueued id={tid}")
    elif args.cmd == "claim":
        t = claim()
        print(json.dumps(t, indent=2) if t else "(nothing to claim)")
    elif args.cmd == "done":
        complete(args.id)
        print(f"done id={args.id}")
    elif args.cmd == "fail":
        res = fail(args.id, args.error)
        print(f"fail id={args.id} -> {res}")
    elif args.cmd == "seed":
        ids = seed_backlog()
        print(f"seeded ids={ids}")


if __name__ == "__main__":
    sys.exit(_cli())
