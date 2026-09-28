"""
Neptune Split DQN heartbeat emitter.
Runs on Neptune in a loop. Scrapes /tmp/split_dqn_v1_r22.log.
Writes structured JSON heartbeat. Pulled by Jupiter cluster_status.
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

LOG = Path("/tmp/split_dqn_v1_r22.log")
OUT_DIR = Path("/home/nick/Lvl3Quant/output/split_dqn_v1_r22_patched")
OUT = OUT_DIR / "heartbeat.json"
PERIOD_S = 30

PROGRESS_RE = re.compile(
    r"Fold (\d+) Ep (\d+): (\d+)/(\d+) files \| trades=(\d+) \| pnl=\$([\-\d.]+) "
    r"\| GPU updates: E=(\d+) C=(\d+) X=(\d+) \| ε=([\d.]+) "
    r"\| buf: E=(\d+) C=(\d+) X=(\d+)"
)
EPOCH_DONE_RE = re.compile(r"Fold (\d+) Ep (\d+) (DONE|complete|finished)", re.I)
SAVE_RE = re.compile(r"Saved.*fold(\d+)_ep(\d+)")
CRASH_RE = re.compile(r"\b(Traceback|CUDA out of memory|OOMKilled|Killed|RuntimeError|MemoryError)\b")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def find_pid() -> int | None:
    try:
        out = subprocess.run(
            ["pgrep", "-f", "train_split_dqn"],
            capture_output=True, text=True, timeout=5)
        for line in out.stdout.split():
            if line.strip().isdigit():
                return int(line.strip())
    except Exception:
        pass
    return None


def proc_stats(pid: int) -> dict:
    out = {"uptime_s": None, "rss_mb": None, "n_threads": None, "n_workers": None}
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        starttime = int(parts[21]) / os.sysconf("SC_CLK_TCK")
        with open("/proc/uptime") as f:
            sysuptime = float(f.read().split()[0])
        out["uptime_s"] = int(sysuptime - starttime)
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_mb"] = int(line.split()[1]) // 1024
                if line.startswith("Threads:"):
                    out["n_threads"] = int(line.split()[1])
        # count python workers under this pid tree
        try:
            children = subprocess.run(
                ["pgrep", "-P", str(pid)],
                capture_output=True, text=True, timeout=5)
            out["n_workers"] = len([x for x in children.stdout.split() if x.strip().isdigit()])
        except Exception:
            pass
    except Exception:
        pass
    return out


def system_stats() -> dict:
    s = {"ram_used_gb": None, "ram_total_gb": None, "ram_pct": None, "load_1m": None}
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":", 1)
                mem[k.strip()] = int(v.strip().split()[0])  # kB
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", 0)
        used_kb = total_kb - avail_kb
        s["ram_total_gb"] = round(total_kb / 1024 / 1024, 2)
        s["ram_used_gb"] = round(used_kb / 1024 / 1024, 2)
        s["ram_pct"] = round(100 * used_kb / total_kb, 1) if total_kb else None
    except Exception:
        pass
    try:
        with open("/proc/loadavg") as f:
            s["load_1m"] = float(f.read().split()[0])
    except Exception:
        pass
    return s


def gpu_stats() -> dict:
    g = {"util_pct": None, "mem_used_mb": None, "mem_total_mb": None}
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        line = out.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        g["util_pct"] = int(parts[0])
        g["mem_used_mb"] = int(parts[1])
        g["mem_total_mb"] = int(parts[2])
    except Exception:
        pass
    return g


def tail_lines(path: Path, n: int = 500) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            return f.read().decode("utf-8", errors="replace").splitlines()[-n:]
    except FileNotFoundError:
        return []


def list_checkpoints() -> list[dict]:
    items = []
    if not OUT_DIR.exists():
        return items
    for p in sorted(OUT_DIR.glob("fold*_ep*")):
        m = re.match(r"fold(\d+)_ep(\d+)$", p.name)
        if m:
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = None
            items.append({
                "fold": int(m.group(1)),
                "epoch": int(m.group(2)),
                "path": str(p),
                "mtime": mtime,
            })
    return items


def parse_log(lines: list[str]) -> dict:
    s = {
        "fold": None, "epoch": None, "files_done": None, "files_total": None,
        "trades": None, "pnl_usd": None,
        "gpu_updates_e": None, "gpu_updates_c": None, "gpu_updates_x": None,
        "epsilon": None,
        "buffer_e": None, "buffer_c": None, "buffer_x": None,
        "last_progress_ts": None, "last_log_line": None,
        "last_error": None, "last_error_ts": None,
    }
    for line in lines:
        m = PROGRESS_RE.search(line)
        if m:
            s["fold"] = int(m.group(1))
            s["epoch"] = int(m.group(2))
            s["files_done"] = int(m.group(3))
            s["files_total"] = int(m.group(4))
            s["trades"] = int(m.group(5))
            s["pnl_usd"] = float(m.group(6))
            s["gpu_updates_e"] = int(m.group(7))
            s["gpu_updates_c"] = int(m.group(8))
            s["gpu_updates_x"] = int(m.group(9))
            s["epsilon"] = float(m.group(10))
            s["buffer_e"] = int(m.group(11))
            s["buffer_c"] = int(m.group(12))
            s["buffer_x"] = int(m.group(13))
            ts_match = TS_RE.match(line)
            if ts_match:
                s["last_progress_ts"] = ts_match.group(1)
            s["last_log_line"] = line.strip()[:300]
        if CRASH_RE.search(line):
            ts_match = TS_RE.match(line)
            s["last_error_ts"] = ts_match.group(1) if ts_match else None
            s["last_error"] = line.strip()[:300]
    if lines:
        s["last_log_line"] = s["last_log_line"] or lines[-1].strip()[:300]
    return s


def write_heartbeat() -> None:
    pid = find_pid()
    alive = pid is not None
    pstats = proc_stats(pid) if alive else {"uptime_s": None, "rss_mb": None, "n_threads": None, "n_workers": None}
    sysstats = system_stats()
    gpu = gpu_stats()
    log_mtime = LOG.stat().st_mtime if LOG.exists() else None
    log_age_s = int(time.time() - log_mtime) if log_mtime else None
    parsed = parse_log(tail_lines(LOG, 500))
    ckpts = list_checkpoints()
    last_ckpt = ckpts[-1] if ckpts else None

    health = "ok"
    reasons = []
    if not alive:
        health = "down"; reasons.append("training_pid_missing")
    if log_age_s is not None and log_age_s > 900:  # 15 min
        health = "stalled"; reasons.append(f"log_stale_{log_age_s}s")
    if sysstats.get("ram_used_gb") and sysstats["ram_used_gb"] > 28:
        health = "memory_warn"; reasons.append(f"ram_{sysstats['ram_used_gb']}gb_over_28gb_cap")
    if parsed.get("last_error_ts"):
        # only flag if error occurred in last 5 min
        try:
            err_t = time.mktime(time.strptime(parsed["last_error_ts"], "%Y-%m-%d %H:%M:%S"))
            if time.time() - err_t < 300:
                health = "error"; reasons.append("recent_traceback_in_log")
        except Exception:
            pass

    hb = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "node": "neptune",
        "service": "split_dqn_v1_r22_patched",
        "pid": pid,
        "alive": alive,
        "uptime_s": pstats.get("uptime_s"),
        "rss_mb": pstats.get("rss_mb"),
        "n_threads": pstats.get("n_threads"),
        "n_workers": pstats.get("n_workers"),
        "log_path": str(LOG),
        "log_mtime_unix": log_mtime,
        "log_age_s": log_age_s,
        "health": health,
        "health_reasons": reasons,
        "system": sysstats,
        "gpu": gpu,
        "checkpoints_count": len(ckpts),
        "latest_checkpoint": last_ckpt,
        **parsed,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(hb, indent=2, default=str))
    os.replace(tmp, OUT)


def main():
    one_shot = "--once" in sys.argv
    while True:
        try:
            write_heartbeat()
        except Exception as e:
            try:
                err = {
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "health": "emitter_error",
                    "error": str(e)[:500],
                }
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(err, indent=2))
            except Exception:
                pass
        if one_shot:
            return
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    main()
