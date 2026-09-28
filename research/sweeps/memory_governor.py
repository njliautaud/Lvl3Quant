#!/usr/bin/env python3
"""
Neptune training memory governor (HC #211).
Runs as a sidecar daemon on Neptune. Polls RSS of train_split_dqn every 30s.
At soft cap (24GB): logs warning, dumps memory profile.
At hard cap (26GB): sends SIGUSR1 (graceful checkpoint) THEN SIGTERM after 60s grace.
Goal: never let kernel OOM-kill the trainer (loses replay buffer + corrupts ckpt).

Watchdog companion (auto-resume on kill) is a separate cron.
"""
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path("/home/nick/Lvl3Quant/output/split_dqn_v1_r22_patched")
STATE_DIR.mkdir(parents=True, exist_ok=True)
GOV_LOG = STATE_DIR / "memory_governor.log"
GOV_STATE = STATE_DIR / "memory_governor_state.json"

POLL_S = 30
SOFT_CAP_GB = 45       # warn + dump profile
HARD_CAP_GB = 50       # graceful checkpoint then kill
GRACE_S = 60           # seconds between SIGUSR1 and SIGTERM
SYS_RAM_CAP_GB = 31    # absolute system cap per HC #200

CMDLINE_PATTERN = "train_split_dqn"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(GOV_LOG, "a") as f:
        f.write(line + "\n")


def find_master_pid() -> int | None:
    """Return the PID of the trainer master (parent of n-workers spawned children)."""
    try:
        out = subprocess.run(
            ["pgrep", "-fa", CMDLINE_PATTERN],
            capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    candidates = []
    for line in out.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        candidates.append(pid)
    if not candidates:
        return None
    # master is the lowest PID with no python parent matching pattern
    pids = set(candidates)
    masters = []
    for pid in candidates:
        try:
            with open(f"/proc/{pid}/status") as f:
                ppid = int([l for l in f if l.startswith("PPid:")][0].split()[1])
        except Exception:
            continue
        if ppid not in pids:
            masters.append(pid)
    return min(masters) if masters else min(candidates)


def proc_rss_mb(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def tree_rss_mb(master_pid: int) -> dict:
    """Sum RSS across master + all descendants."""
    pids = {master_pid}
    # BFS via /proc/*/status PPid
    try:
        all_pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except Exception:
        return {"total_mb": proc_rss_mb(master_pid) or 0, "n": 1}
    parents = {}
    for p in all_pids:
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        parents[p] = int(line.split()[1])
                        break
        except Exception:
            pass
    # find descendants
    changed = True
    while changed:
        changed = False
        for p, ppid in parents.items():
            if ppid in pids and p not in pids:
                pids.add(p); changed = True
    total = 0
    n = 0
    for p in pids:
        rss = proc_rss_mb(p)
        if rss is not None:
            total += rss; n += 1
    return {"total_mb": total, "n": n, "pids": sorted(pids)}


def system_ram_used_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":", 1)
                mem[k.strip()] = int(v.strip().split()[0])
        used = mem["MemTotal"] - mem["MemAvailable"]
        return round(used / 1024 / 1024, 2)
    except Exception:
        return -1.0


def write_state(payload: dict) -> None:
    payload["ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    GOV_STATE.write_text(json.dumps(payload, indent=2, default=str))


def graceful_kill(pid: int, why: str) -> None:
    """Escalating signal ladder: SIGINT (KeyboardInterrupt -> finally blocks) -> SIGTERM -> SIGKILL.
    The last completed-epoch checkpoint (HC #208) is the worst-case loss window.
    """
    log(f"GRACEFUL KILL initiated for pid={pid} reason={why}. Stage 1: SIGINT.")
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        log(f"  pid {pid} already gone before SIGINT"); return
    log(f"  Waiting {GRACE_S}s for KeyboardInterrupt cleanup...")
    deadline = time.time() + GRACE_S
    while time.time() < deadline:
        if proc_rss_mb(pid) is None:
            log(f"  pid {pid} exited within SIGINT grace window"); return
        time.sleep(2)
    log(f"  SIGINT grace expired. Stage 2: SIGTERM.")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(15)
    # HC215_TREE_KILL: nuke the whole tree if SIGTERM didn't take.
    if proc_rss_mb(pid) is not None:
        log(f"  SIGTERM ineffective. Stage 3: SIGKILL master+tree.")
        # Get tree before kill (master PID still alive)
        tree_pids = list(tree_rss_mb(pid).get("pids", [pid]))
        # First try kill process group (setsid'd by launcher)
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
            log(f"  SIGKILL sent to pgid={pgid}")
        except (ProcessLookupError, PermissionError) as _e:
            log(f"  killpg failed: {_e}")
        # Then individually nuke each PID in the tree (catch any reparented orphans)
        for tp in tree_pids:
            try:
                os.kill(tp, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Final verify
        time.sleep(3)
        survivors = [tp for tp in tree_pids if proc_rss_mb(tp) is not None]
        if survivors:
            log(f"  WARNING: {len(survivors)} pids still alive after SIGKILL: {survivors}")
        else:
            log(f"  Tree fully killed ({len(tree_pids)} pids).")


def main():
    log(f"memory_governor started. soft_cap={SOFT_CAP_GB}GB hard_cap={HARD_CAP_GB}GB sys_cap={SYS_RAM_CAP_GB}GB poll={POLL_S}s")
    last_warn = 0.0
    while True:
        master = find_master_pid()
        if master is None:
            write_state({"trainer": "absent", "msg": "no train_split_dqn process found"})
            time.sleep(POLL_S)
            continue
        tree = tree_rss_mb(master)
        rss_gb = round(tree["total_mb"] / 1024, 2)
        sys_used = system_ram_used_gb()
        state = {
            "master_pid": master,
            "n_processes": tree["n"],
            "pids": tree.get("pids"),
            "rss_total_gb": rss_gb,
            "system_ram_used_gb": sys_used,
            "soft_cap_gb": SOFT_CAP_GB,
            "hard_cap_gb": HARD_CAP_GB,
            "system_cap_gb": SYS_RAM_CAP_GB,
            "verdict": "ok",
        }
        # decision tree
        if rss_gb >= HARD_CAP_GB or sys_used >= SYS_RAM_CAP_GB:
            state["verdict"] = "hard_cap_breach"
            log(f"HARD CAP BREACH: train_rss={rss_gb}GB sys_used={sys_used}GB. Triggering graceful kill.")
            write_state(state)
            graceful_kill(master, why=f"train_rss={rss_gb}GB sys_used={sys_used}GB")
            time.sleep(60)  # avoid rapid re-trigger; auto-resume handled by separate watchdog
            continue
        if rss_gb >= SOFT_CAP_GB or sys_used >= (SYS_RAM_CAP_GB - 2):
            state["verdict"] = "soft_cap_warning"
            now = time.time()
            if now - last_warn > 120:
                log(f"SOFT CAP WARNING: train_rss={rss_gb}GB sys_used={sys_used}GB. {HARD_CAP_GB - rss_gb:.1f}GB headroom to hard cap.")
                last_warn = now
        write_state(state)
        time.sleep(POLL_S)


if __name__ == "__main__":
    if "--once" in sys.argv:
        master = find_master_pid()
        if master is None:
            print("no trainer process")
            sys.exit(0)
        tree = tree_rss_mb(master)
        sys_used = system_ram_used_gb()
        write_state({
            "master_pid": master,
            "n_processes": tree["n"],
            "rss_total_gb": round(tree["total_mb"] / 1024, 2),
            "system_ram_used_gb": sys_used,
            "soft_cap_gb": SOFT_CAP_GB,
            "hard_cap_gb": HARD_CAP_GB,
            "system_cap_gb": SYS_RAM_CAP_GB,
            "verdict": "probe",
        })
        print(json.dumps(json.loads(GOV_STATE.read_text()), indent=2))
        sys.exit(0)
    main()
