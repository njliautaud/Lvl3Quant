#!/usr/bin/env python3
"""
queue_puller.py — Node-side daemon that pulls and executes tasks from the
persistent on-disk task queue (ops/task_queue.jsonl).

Responsibilities:
  * Identify which node it is running on (from --node or hostname mapping).
  * Every POLL_INTERVAL seconds:
      - If no task is currently running on this node:
          atomically claim the highest-priority pending task for this node
          via `task_queue.claim_next(node)` and launch it as a child process.
      - If a child IS running: check liveness, capture exit code on completion,
          mark task done/failed accordingly.
  * Heartbeat file (ops/puller_<node>.heartbeat) updated every poll so a
    monitor can detect a dead puller.
  * Log every claim/launch/finish to ops/logs/puller_<node>.log
  * Self-restart-safe: relies on systemd / cron / pm2 / nohup-supervisor
    to bring it back up if it dies. Does not crash on transient errors.

CLI:
  queue_puller.py --node jupiter            # blocking foreground (use systemd)
  queue_puller.py --node jupiter --once     # claim+launch one task and exit
  queue_puller.py --node jupiter --status   # print current state and exit

NOTE: this daemon is INTENTIONALLY single-task-at-a-time per node. Multi-tenant
GPU nodes that want parallelism should run multiple pullers with different
--worker-id values (queue claim is atomic via flock).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make sibling module importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import task_queue as tq  # noqa: E402

OPS_DIR = Path("/home/jupiter/Lvl3Quant/ops")
LOG_DIR = OPS_DIR / "logs"

POLL_INTERVAL_SEC = 30
HEARTBEAT_FILE_TMPL = "puller_{node}.heartbeat"
PULLER_LOG_TMPL = "puller_{node}.log"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_node(arg: str | None) -> str:
    if arg:
        return arg
    host = socket.gethostname().lower()
    if "jupiter" in host:
        return "jupiter"
    if "neptune" in host:
        return "neptune"
    if "saturn" in host:
        return "saturn"
    if "razer" in host:
        return "razer"
    raise ValueError(f"could not infer node from hostname {host!r}; pass --node")


def _log(node: str, msg: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{_now()}] [{node}] {msg}\n"
    with open(LOG_DIR / PULLER_LOG_TMPL.format(node=node), "a") as f:
        f.write(line)
        f.flush()
    print(line, end="", flush=True)


def _heartbeat(node: str, payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    hb = OPS_DIR / HEARTBEAT_FILE_TMPL.format(node=node)
    payload = {"ts": _now(), "node": node, **payload}
    tmp = hb.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, hb)


def _launch_task(node: str, task: dict) -> tuple[subprocess.Popen, Path]:
    """Launch task['cmd'] as a child shell process, capture stdout+stderr to a log."""
    tid = task["id"]
    log_path = LOG_DIR / f"task_{tid}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w")
    # Pre-write header
    log_f.write(f"# task_id={tid}\n# node={node}\n# kind={task.get('kind')}\n")
    log_f.write(f"# cmd={task.get('cmd')}\n# started_at={_now()}\n\n")
    log_f.flush()
    # Inherit env. Run via /bin/bash -lc so PATH / conda activations work.
    proc = subprocess.Popen(
        ["/bin/bash", "-lc", task["cmd"]],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        cwd=str(Path.home()),
        start_new_session=True,
    )
    _log(node, f"LAUNCHED task {tid} pid={proc.pid} log={log_path}")
    return proc, log_path


def _poll_once(node: str, state: dict) -> None:
    """One iteration of the puller loop. Mutates state in-place."""
    proc = state.get("proc")
    task = state.get("task")

    # Phase 1: if a task is running, check on it.
    if proc is not None and task is not None:
        rc = proc.poll()
        if rc is None:
            _heartbeat(node, {"status": "running", "task_id": task["id"],
                              "pid": proc.pid, "log": str(state.get("log_path"))})
            return
        # Process finished
        log_path = str(state.get("log_path", ""))
        if rc == 0:
            tq.mark_done(task["id"], log_path=log_path)
            _log(node, f"DONE task {task['id']} rc=0 log={log_path}")
        else:
            tq.mark_failed(task["id"], error=f"exit_code={rc}", log_path=log_path)
            _log(node, f"FAIL task {task['id']} rc={rc} log={log_path}")
        state["proc"] = None
        state["task"] = None
        state["log_path"] = None

    # Phase 2: idle — try to claim next.
    try:
        claimed = tq.claim_next(node)
    except Exception as e:
        _log(node, f"claim_next ERROR: {e}")
        _heartbeat(node, {"status": "idle", "error": str(e)})
        return

    if claimed is None:
        _heartbeat(node, {"status": "idle", "task_id": None})
        return

    try:
        proc, log_path = _launch_task(node, claimed)
        state["proc"] = proc
        state["task"] = claimed
        state["log_path"] = log_path
        _heartbeat(node, {"status": "running", "task_id": claimed["id"],
                          "pid": proc.pid, "log": str(log_path)})
    except Exception as e:
        tq.mark_failed(claimed["id"], error=f"launch_failed: {e}")
        _log(node, f"LAUNCH FAILED task {claimed['id']}: {e}")


def run_forever(node: str, poll_interval: int = POLL_INTERVAL_SEC) -> None:
    _log(node, f"queue_puller START interval={poll_interval}s pid={os.getpid()}")
    state: dict = {"proc": None, "task": None, "log_path": None}

    stop = {"v": False}

    def _handle(sig, _frame):
        stop["v"] = True
        _log(node, f"signal {sig} received — will exit after current poll")
    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    while not stop["v"]:
        try:
            _poll_once(node, state)
        except Exception as e:
            _log(node, f"poll ERROR: {e!r}")
        time.sleep(poll_interval)

    # On shutdown: if a child is still running, leave it alone (it will
    # outlive us and be reaped by init). Log a note.
    if state.get("proc") is not None:
        t = state["task"]
        _log(node, f"shutting down with task {t['id']} still running pid={state['proc'].pid}")
    _log(node, "queue_puller STOP")


def status(node: str) -> None:
    hb = OPS_DIR / HEARTBEAT_FILE_TMPL.format(node=node)
    if not hb.exists():
        print(f"no heartbeat file for node={node}")
        return
    print(hb.read_text())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node", default=None,
                   help="node identity (jupiter/neptune/saturn/razer). "
                        "Auto-detected from hostname if omitted.")
    p.add_argument("--once", action="store_true",
                   help="single poll iteration then exit (testing)")
    p.add_argument("--status", action="store_true",
                   help="print current heartbeat and exit")
    p.add_argument("--interval", type=int, default=POLL_INTERVAL_SEC,
                   help="poll interval seconds (default 30)")
    args = p.parse_args(argv)

    node = _resolve_node(args.node)

    if args.status:
        status(node)
        return 0

    if args.once:
        state: dict = {"proc": None, "task": None, "log_path": None}
        _poll_once(node, state)
        # If a process was launched, wait for it (so the test is deterministic)
        if state.get("proc") is not None:
            proc = state["proc"]
            proc.wait()
            # Single follow-up poll to update done/failed status
            _poll_once(node, state)
        return 0

    run_forever(node, poll_interval=args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
