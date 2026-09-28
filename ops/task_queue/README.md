# Persistent Task Queue — HC #422 Rule 10

A simple, durable, on-disk work queue that survives Claude context resets,
PM2 restarts, and machine reboots. Built per the autonomy RCA in
`output/hc422_autonomy_rca.md`.

## Why this exists

Crons fire prompts to Claude. If Claude is busy, compacting, or absent, the
prompt is dropped — work doesn't happen. SESSION_STATE.md is *memory*, not a
*work queue*: it records what was done, not what to do next in a machine-
consumable form. Result: nodes go idle (Jupiter idle since 09:05 ET today,
per HC #422 Rule 9).

This queue closes the gap: any process — a cron, a node-side puller, Claude
itself, or a human — can enqueue a task. Any node-side puller can atomically
claim and execute the next eligible task.

## Layout

```
ops/task_queue/
  README.md              # this file
  schema.md              # task JSON schema
  queue.jsonl            # canonical queue, one task per line
  queue.lock             # fcntl.flock target (zero bytes)
  completed.jsonl        # append-only audit log
  claimed/<id>.json      # per-claim sidecar (claimer + claim_ts)
  logs/                  # task stdout/stderr lands here (by convention)
  enqueue.py             # add a task
  claim.py               # atomically claim the next eligible task
  complete.py            # mark a task done/failed
  list.py                # read-only view
  _lib.py                # shared primitives (locking, IO)
```

## Quick start

Enqueue:

```bash
cd /home/jupiter/Lvl3Quant/ops/task_queue
python enqueue.py \
    --command "python scripts/hc422_jupiter_rules_sweep.py --window 56d" \
    --node-preference jupiter \
    --priority 2 \
    --max-runtime-s 21600 \
    --tag hc422 fifo rules_sweep
```

Inspect:

```bash
python list.py                     # counts + top-20 pending
python list.py --status pending --json
python list.py --node jupiter
```

Claim (typically called by a node-side puller daemon):

```bash
python claim.py --node jupiter
# emits a JSON line with the claimed task; sidecar at claimed/<id>.json
```

Complete:

```bash
python complete.py --id <task_id> --exit-code 0 --log-path /path/to/run.log
# or, on failure:
python complete.py --id <task_id> --exit-code 137 --error "OOM"
```

## Atomicity guarantees

All mutating operations acquire `fcntl.flock(LOCK_EX)` on `queue.lock` and
rewrite `queue.jsonl` via `os.replace`. Reads use `LOCK_SH`. Concurrent
claimers will serialize; exactly one wins per pending task.

`fcntl.flock` is advisory but every script in this scaffold uses it
consistently — do NOT touch `queue.jsonl` from any tool that bypasses the lock.

## Dependencies

Tasks can declare `depends_on: ["<task_id>", ...]`. A dependent task is
not eligible for claim until all of its dependencies reach `status == "done"`.
A failed dependency leaves dependents stuck — operator must either retry/heal
the dependency or remove the `depends_on` reference.

## Integration points

- **Cron**: a 1-minute cron on each node runs `claim.py --node $HOSTNAME`. If
  it gets a task, it spawns it; if not, exits cleanly. Idempotent.
- **Idle detector**: `nvidia-smi` / `mpstat` based watcher pushes a high-priority
  task into the queue when a node sits idle > N minutes.
- **Claude**: when Claude has plans, it appends them as tasks instead of
  writing prose into SESSION_STATE.md. SESSION_STATE.md becomes a log; the
  queue becomes the action plan.

## What this scaffold does NOT include (deferred)

- Reaper for crashed claimers (use sidecar mtime > max_runtime_s + 600s).
- Backoff / retry policy.
- Resource accounting (GPU memory, CPU cores).
- Web dashboard. (Use `list.py --json | jq` for now.)
- The node-side puller daemon itself — see `ops/queue_puller.py` for the
  existing prototype that consumes the older `ops/task_queue.jsonl`; an updated
  variant pointed at this subdirectory is the obvious next step.
