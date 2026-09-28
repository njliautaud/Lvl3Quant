#!/usr/bin/env python3
"""razer_watchdog.py — Persistent watchdog for Razer live trading services.

Manages two critical processes:
  1. mbo_recorder.py (MBO market data recorder)
  2. paper_trading_mamba_v2.py (CNN-Mamba v2 live inference)

Features:
  - Exponential backoff on restarts (30s -> 5min)
  - Staggered starts (recorder first, inference 10s later)
  - Rithmic session conflict prevention (only one reconnect at a time)
  - Process detection via command-line matching
  - Heartbeat file for external monitoring
  - Full logging to logs/watchdog.log

Designed for Windows — uses CREATE_NO_WINDOW, wmi-free process detection.
Run via start_razer_live.bat or directly: python razer_watchdog.py
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import signal
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PYTHON = r"C:\Python311\python.exe"
WORKING_DIR = r"C:\Users\claude\Lvl3Quant\live_trading"
LOG_DIR = Path(WORKING_DIR) / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HEARTBEAT_FILE = LOG_DIR / "watchdog_heartbeat.json"
LOG_FILE = LOG_DIR / "watchdog.log"

# Windows process creation flag — hides console window for child processes
CREATE_NO_WINDOW = 0x08000000

# Health check interval (seconds)
CHECK_INTERVAL = 30

# Backoff settings (seconds)
BACKOFF_INITIAL = 30
BACKOFF_MAX = 300  # 5 minutes
BACKOFF_MULTIPLIER = 2

# Stagger delay between recorder and inference starts (seconds)
STAGGER_DELAY = 10

# QCC daemon for alerting (Jupiter via Tailscale)
QCC_URL = "http://uranus:3456"


def send_alert(message: str, severity: str = "warning"):
    """Send alert to QCC daemon and log file."""
    log.warning("ALERT: %s", message)
    try:
        payload = json.dumps({
            "severity": severity,
            "source": "razer_watchdog",
            "node": "razer",
            "message": message,
        }).encode()
        req = Request(f"{QCC_URL}/api/alerts", data=payload,
                      headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=5)
    except Exception as exc:
        log.warning("Failed to send QCC alert: %s", exc)

# ---------------------------------------------------------------------------
# Service definitions
# ---------------------------------------------------------------------------
SERVICES = {
    "mbo_recorder": {
        "script": "mbo_recorder.py",
        "args": ["--symbol", "ESM6", "--exchange", "CME"],
        "match_token": "mbo_recorder.py",
        "priority": 0,  # lower = started first
    },
    "paper_trading": {
        "script": "paper_trading_mamba_v2.py",
        "args": [
            "--weights", r"C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_10_best.pt",
            "--stats", r"C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_smart_v3_mar\fold_09_feature_stats.npz",
            "--device", "cuda",
        ],
        "match_token": "paper_trading_mamba_v2.py",
        "priority": 1,
    },
}

# Environment variables required by both services
SERVICE_ENV = {
    "PYTHONPATH": r"C:\Users\claude\Lvl3Quant",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
    "PYTHONUNBUFFERED": "1",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("watchdog")
log.setLevel(logging.DEBUG)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")

_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
log.addHandler(_fh)

_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
_sh.setFormatter(_fmt)
log.addHandler(_sh)


# ---------------------------------------------------------------------------
# Process detection (no WMI dependency)
# ---------------------------------------------------------------------------
def _list_processes_wmic() -> list[dict]:
    """Use wmic to list running python processes with command lines."""
    try:
        result = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe' or name='python3.exe' or name='python311.exe'",
             "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
            capture_output=True, text=True, timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        processes = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("Node,"):
                continue
            # CSV format: Node,CommandLine,ProcessId
            parts = line.split(",", 2)
            if len(parts) >= 3:
                try:
                    pid = int(parts[-1].strip())
                    cmdline = ",".join(parts[1:-1]).strip()
                    processes.append({"pid": pid, "cmdline": cmdline})
                except ValueError:
                    continue
        return processes
    except Exception as exc:
        log.warning("wmic process list failed: %s", exc)
        return []


def _list_processes_tasklist() -> list[dict]:
    """Fallback: use tasklist + /V for verbose info."""
    try:
        result = subprocess.run(
            ["tasklist", "/V", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        processes = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.strip('"').split('","')
            if len(parts) >= 2:
                try:
                    pid = int(parts[1].strip())
                    # tasklist doesn't give cmdline, but we can check /V window title
                    processes.append({"pid": pid, "cmdline": line})
                except ValueError:
                    continue
        return processes
    except Exception as exc:
        log.warning("tasklist fallback failed: %s", exc)
        return []


def find_process(match_token: str) -> int | None:
    """Return PID of a running process whose command line contains match_token.
    Returns None if not found. Excludes our own PID."""
    my_pid = os.getpid()
    processes = _list_processes_wmic()
    if not processes:
        processes = _list_processes_tasklist()

    for proc in processes:
        if proc["pid"] == my_pid:
            continue
        if match_token in proc.get("cmdline", ""):
            return proc["pid"]
    return None


def is_process_alive(pid: int) -> bool:
    """Check if a PID is still running on Windows."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
        return str(pid) in result.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Service manager
# ---------------------------------------------------------------------------
class ServiceState:
    """Tracks state for a managed service."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.pid: int | None = None
        self.process: subprocess.Popen | None = None
        self.backoff = BACKOFF_INITIAL
        self.last_restart: float = 0
        self.restart_count = 0
        self.status = "unknown"
        self.last_seen_alive: float = 0

    def reset_backoff(self):
        """Reset backoff after sustained uptime (>5 min)."""
        self.backoff = BACKOFF_INITIAL

    def increase_backoff(self):
        self.backoff = min(self.backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pid": self.pid,
            "status": self.status,
            "backoff_seconds": self.backoff,
            "restart_count": self.restart_count,
            "last_restart": datetime.fromtimestamp(self.last_restart, tz=timezone.utc).isoformat()
            if self.last_restart else None,
            "last_seen_alive": datetime.fromtimestamp(self.last_seen_alive, tz=timezone.utc).isoformat()
            if self.last_seen_alive else None,
        }


class Watchdog:
    """Main watchdog loop managing all services."""

    def __init__(self):
        # Sort services by priority (recorder before inference)
        self.services: list[ServiceState] = sorted(
            [ServiceState(name, cfg) for name, cfg in SERVICES.items()],
            key=lambda s: s.config["priority"],
        )
        self._stop = threading.Event()
        self._restart_lock = threading.Lock()  # Rithmic: one reconnect at a time
        self._start_time = time.time()

    def _build_env(self) -> dict:
        """Build environment for child processes."""
        env = os.environ.copy()
        env.update(SERVICE_ENV)
        return env

    def _build_cmd(self, svc: ServiceState) -> list[str]:
        """Build the full command list for a service."""
        return [PYTHON, svc.config["script"]] + svc.config["args"]

    def _start_service(self, svc: ServiceState) -> bool:
        """Start a service process. Returns True on success."""
        now = time.time()
        elapsed = now - svc.last_restart if svc.last_restart else float("inf")

        # Enforce backoff
        if elapsed < svc.backoff:
            remaining = svc.backoff - elapsed
            log.debug("%s: backoff active, %.0fs remaining", svc.name, remaining)
            return False

        cmd = self._build_cmd(svc)
        log.info("Starting %s: %s", svc.name, " ".join(cmd))

        try:
            # Serialize Rithmic connections — only one service connects at a time
            with self._restart_lock:
                log_file = LOG_DIR / f"{svc.name}_stdout.log"
                with open(log_file, "a", encoding="utf-8") as logf:
                    logf.write(f"\n--- Watchdog start at {datetime.now(timezone.utc).isoformat()} ---\n")
                    logf.flush()

                proc = subprocess.Popen(
                    cmd,
                    cwd=WORKING_DIR,
                    env=self._build_env(),
                    stdout=open(LOG_DIR / f"{svc.name}_stdout.log", "a", encoding="utf-8"),
                    stderr=subprocess.STDOUT,
                    creationflags=CREATE_NO_WINDOW,
                )
                svc.process = proc
                svc.pid = proc.pid
                svc.status = "running"
                svc.last_restart = time.time()
                svc.restart_count += 1
                svc.increase_backoff()
                log.info("%s started with PID %d (next backoff: %ds)",
                         svc.name, svc.pid, svc.backoff)
                return True

        except Exception as exc:
            log.error("Failed to start %s: %s", svc.name, exc)
            svc.status = "error"
            svc.increase_backoff()
            return False

    def _check_service(self, svc: ServiceState) -> bool:
        """Check if a service is alive. Returns True if running."""
        # Method 1: Check our Popen handle
        if svc.process is not None:
            retcode = svc.process.poll()
            if retcode is None:
                svc.status = "running"
                svc.last_seen_alive = time.time()
                # Reset backoff after 5 min sustained uptime
                if svc.last_seen_alive - svc.last_restart > 300:
                    svc.reset_backoff()
                return True
            else:
                log.warning("%s exited with code %s", svc.name, retcode)
                svc.process = None
                svc.pid = None
                svc.status = "stopped"
                return False

        # Method 2: Search by command-line match (process started before watchdog)
        pid = find_process(svc.config["match_token"])
        if pid is not None:
            svc.pid = pid
            svc.status = "running"
            svc.last_seen_alive = time.time()
            log.info("%s found running externally (PID %d)", svc.name, pid)
            return True

        svc.status = "stopped"
        svc.pid = None
        return False

    def _write_heartbeat(self):
        """Write heartbeat JSON for external monitoring."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": int(time.time() - self._start_time),
            "watchdog_pid": os.getpid(),
            "services": {svc.name: svc.to_dict() for svc in self.services},
        }
        try:
            tmp = HEARTBEAT_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(HEARTBEAT_FILE)
        except Exception as exc:
            log.warning("Failed to write heartbeat: %s", exc)

    def run(self):
        """Main watchdog loop."""
        log.info("=" * 60)
        log.info("Razer Watchdog starting (PID %d)", os.getpid())
        log.info("Monitoring %d services, check interval %ds",
                 len(self.services), CHECK_INTERVAL)
        log.info("=" * 60)

        # Initial startup — staggered
        pending_starts: list[ServiceState] = []
        for svc in self.services:
            if not self._check_service(svc):
                pending_starts.append(svc)
            else:
                log.info("%s already running (PID %d)", svc.name, svc.pid)

        for i, svc in enumerate(pending_starts):
            if i > 0:
                log.info("Stagger delay: waiting %ds before starting %s",
                         STAGGER_DELAY, svc.name)
                time.sleep(STAGGER_DELAY)
            self._start_service(svc)

        # Main loop
        while not self._stop.is_set():
            self._stop.wait(CHECK_INTERVAL)
            if self._stop.is_set():
                break

            needs_start: list[ServiceState] = []

            for svc in self.services:
                alive = self._check_service(svc)
                if alive:
                    log.debug("%s: alive (PID %d)", svc.name, svc.pid)
                else:
                    log.warning("%s: DOWN — scheduling restart", svc.name)
                    send_alert(f"🔴 {svc.name} CRASHED on Razer — auto-restarting (attempt #{svc.restart_count + 1})", "critical")
                    needs_start.append(svc)

            # Stagger restarts to avoid Rithmic session conflicts
            for i, svc in enumerate(needs_start):
                if i > 0:
                    log.info("Stagger delay: %ds before restarting %s",
                             STAGGER_DELAY, svc.name)
                    time.sleep(STAGGER_DELAY)
                self._start_service(svc)

            self._write_heartbeat()

        log.info("Watchdog shutting down")

    def stop(self):
        """Signal the watchdog to stop."""
        self._stop.set()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    watchdog = Watchdog()

    def _signal_handler(signum, frame):
        log.info("Received signal %d, shutting down...", signum)
        watchdog.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        watchdog.run()
    except KeyboardInterrupt:
        watchdog.stop()
    except Exception:
        log.exception("Watchdog crashed")
        raise


if __name__ == "__main__":
    main()
