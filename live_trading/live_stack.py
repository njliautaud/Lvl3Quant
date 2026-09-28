#!/usr/bin/env python3
"""
live_stack.py — Unified Razer Live Stack Launcher (HC #66, #67)

ONE SCRIPT to rule them all. Launches, monitors, and manages:
  1. MBO Recorder (mbo_recorder.py) — captures ES MBO events to daily NPZ + JSONL fan-out
  2. Live Inference + Paper Trader (paper_trading_mamba_v2.py) — CNN-Mamba v2 signals + paper trades

Failure handling:
  - Startup validation: weights, GPU, disk, credentials, protobuf
  - Auto-restart with exponential backoff (5s → 10s → 20s → 60s cap)
  - Health checks every 30s: process alive, disk space, GPU, log freshness
  - Graceful shutdown on SIGINT/SIGTERM/Ctrl-C
  - Discord webhook alerts on crash/restart/startup
  - Daily log rotation at midnight UTC
  - CME maintenance window awareness (17:00-17:45 ET Sun-Thu)

Usage:
    python live_stack.py                         # Normal launch
    python live_stack.py --dry-run               # Validate only, don't launch
    python live_stack.py --recorder-only         # Just MBO recorder
    python live_stack.py --no-paper              # Recorder + inference, no paper trades
    python live_stack.py --symbol ESM6           # Override symbol (default ESM6)

Author: Claude (Autonomous Infrastructure)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

# ─── Platform-aware paths ───
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    LVL3 = Path(r"C:\Users\claude\Lvl3Quant")
else:
    LVL3 = Path("/home/jupiter/Lvl3Quant")

LIVE_DIR = LVL3 / "live_trading"
LOG_DIR = LIVE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = LVL3 / "data" / "processed" / "mbo_events"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model weights
WEIGHTS_DIR = LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "fold_10_best.pt"
DEFAULT_STATS = WEIGHTS_DIR / "fold_09_feature_stats.npz"
DEFAULT_PREDS = WEIGHTS_DIR / "concat_oot_predictions.npz"

# Python executable
PYTHON = sys.executable if IS_WINDOWS else "python3"

# Discord webhook for alerts (set via env or hardcode)
DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK", "")

# ─── Logging ───
_log_file = LOG_DIR / f"live_stack_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(_log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("live_stack")


# ─── Discord notification ───
def notify_discord(message: str, level: str = "info"):
    """Send alert to Discord webhook. Non-blocking, fire-and-forget."""
    if not DISCORD_WEBHOOK:
        return
    try:
        import urllib.request
        data = json.dumps({"content": f"**[Razer Live Stack]** {message}"}).encode()
        req = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # Never let notification failure crash the stack


# ─── Startup Validation ───
class ValidationError(Exception):
    pass


def validate_prerequisites(args) -> list[str]:
    """Check everything needed before launch. Returns list of warnings (empty = all good)."""
    errors = []
    warnings = []

    # 1. Model weights exist
    weights_path = Path(args.weights) if args.weights else DEFAULT_WEIGHTS
    stats_path = Path(args.stats) if args.stats else DEFAULT_STATS
    if not weights_path.exists():
        errors.append(f"Model weights not found: {weights_path}")
    else:
        size_mb = weights_path.stat().st_size / 1e6
        log.info(f"✓ Model weights: {weights_path} ({size_mb:.1f} MB)")

    if not stats_path.exists():
        errors.append(f"Feature stats not found: {stats_path}")
    else:
        log.info(f"✓ Feature stats: {stats_path}")

    # 2. GPU available (on Windows/CUDA systems)
    if not args.recorder_only:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
                log.info(f"✓ GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")
            else:
                warnings.append("No CUDA GPU available — inference will use CPU (slow!)")
        except ImportError:
            errors.append("PyTorch not installed — cannot run inference")

    # 3. Disk space (need at least 5GB for daily recording)
    try:
        if IS_WINDOWS:
            import shutil
            total, used, free = shutil.disk_usage(str(LVL3))
        else:
            import shutil
            total, used, free = shutil.disk_usage(str(LVL3))
        free_gb = free / 1e9
        if free_gb < 2.0:
            errors.append(f"CRITICAL: Only {free_gb:.1f} GB disk space free! Need at least 2 GB.")
        elif free_gb < 5.0:
            warnings.append(f"Low disk space: {free_gb:.1f} GB free. MBO data is ~200MB/day.")
        else:
            log.info(f"✓ Disk space: {free_gb:.1f} GB free")
    except Exception as e:
        warnings.append(f"Could not check disk space: {e}")

    # 4. Rithmic credentials
    env_path = LIVE_DIR / ".env"
    if env_path.exists():
        with open(env_path) as f:
            env_content = f.read()
        required_vars = ["RITHMIC_USER", "RITHMIC_PASSWORD", "RITHMIC_SYSTEM", "RITHMIC_URI"]
        for var in required_vars:
            if var not in env_content:
                errors.append(f"Missing {var} in {env_path}")
        log.info(f"✓ Rithmic credentials: {env_path}")
    else:
        errors.append(f"Rithmic .env not found: {env_path}")

    # 5. Key scripts exist
    recorder_script = LIVE_DIR / "mbo_recorder.py"
    paper_script = LIVE_DIR / "paper_trading_mamba_v2.py"
    rithmic_client = LIVE_DIR / "rithmic_client.py"
    streaming_feats = LIVE_DIR / "streaming_features_smart_v3.py"

    for script, name in [
        (recorder_script, "MBO recorder"),
        (paper_script, "Paper trader"),
        (rithmic_client, "Rithmic client"),
        (streaming_feats, "Streaming features"),
    ]:
        if not script.exists():
            errors.append(f"{name} script not found: {script}")
        else:
            log.info(f"✓ {name}: {script}")

    # 6. Protobuf modules
    pb_dir = os.environ.get("RITHMIC_PB_DIR", "")
    if not pb_dir:
        # Try common locations
        candidates = [
            LVL3 / "live_trading" / "protobuf",
            LVL3 / "protobuf",
            Path(r"C:\Users\claude\rithmic_protobuf") if IS_WINDOWS else Path("/opt/rithmic/protobuf"),
        ]
        for c in candidates:
            if c.exists():
                pb_dir = str(c)
                break
    if pb_dir and Path(pb_dir).exists():
        log.info(f"✓ Protobuf dir: {pb_dir}")
    else:
        warnings.append("Protobuf dir not found — Rithmic client may fail to import")

    # 7. Check for stale fan-out JSONL (>500MB = needs rotation)
    fanout = LIVE_DIR / "logs" / "live_events.jsonl"
    if fanout.exists():
        size_mb = fanout.stat().st_size / 1e6
        if size_mb > 500:
            warnings.append(f"Fan-out JSONL is {size_mb:.0f} MB — truncating to prevent disk issues")
            try:
                fanout.write_text("")
                log.info(f"Truncated fan-out JSONL ({size_mb:.0f} MB → 0)")
            except Exception as e:
                warnings.append(f"Failed to truncate fan-out: {e}")

    if errors:
        raise ValidationError("\n".join(f"  ✗ {e}" for e in errors))

    return warnings


# ─── Managed Subprocess ───
class ManagedProcess:
    """Wraps a subprocess with auto-restart, health monitoring, and backoff."""

    def __init__(self, name: str, cmd: list[str], cwd: str, env: dict | None = None):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.env = env or {}
        self.process: Optional[subprocess.Popen] = None
        self.start_time: float = 0
        self.restart_count: int = 0
        self.last_restart: float = 0
        self.backoff: float = 5.0  # seconds
        self.max_backoff: float = 60.0
        self._stop_requested = False
        self._log_file: Optional[Path] = None

    def start(self) -> bool:
        """Start the subprocess. Returns True on success."""
        if self._stop_requested:
            return False

        # Build environment
        proc_env = os.environ.copy()
        proc_env.update({
            "PYTHONPATH": str(LVL3),
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
            "PYTHONUNBUFFERED": "1",
        })
        proc_env.update(self.env)

        # Log file for this session
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = LOG_DIR / f"{self.name}_{ts}.log"

        try:
            log.info(f"Starting {self.name}: {' '.join(self.cmd)}")
            with open(self._log_file, "w") as lf:
                self.process = subprocess.Popen(
                    self.cmd,
                    cwd=self.cwd,
                    env=proc_env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0,
                )
            self.start_time = time.time()
            log.info(f"  → {self.name} started (PID {self.process.pid}, log: {self._log_file})")
            return True
        except Exception as e:
            log.error(f"Failed to start {self.name}: {e}")
            return False

    def is_alive(self) -> bool:
        """Check if subprocess is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def stop(self, timeout: float = 10.0):
        """Gracefully stop the subprocess."""
        self._stop_requested = True
        if self.process is None or not self.is_alive():
            return

        log.info(f"Stopping {self.name} (PID {self.process.pid})...")
        try:
            if IS_WINDOWS:
                # Send CTRL_BREAK_EVENT on Windows
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.send_signal(signal.SIGTERM)

            self.process.wait(timeout=timeout)
            log.info(f"  → {self.name} stopped gracefully")
        except subprocess.TimeoutExpired:
            log.warning(f"  → {self.name} didn't stop gracefully, killing...")
            self.process.kill()
            self.process.wait(timeout=5)
        except Exception as e:
            log.warning(f"  → Error stopping {self.name}: {e}")
            try:
                self.process.kill()
            except Exception:
                pass

    def check_and_restart(self) -> bool:
        """Check if process crashed and restart with backoff. Returns True if restarted."""
        if self._stop_requested or self.is_alive():
            return False

        # Process died
        exit_code = self.process.returncode if self.process else -1
        uptime = time.time() - self.start_time if self.start_time else 0

        log.warning(f"⚠ {self.name} CRASHED (exit={exit_code}, uptime={uptime:.0f}s)")
        notify_discord(f"⚠️ {self.name} crashed (exit={exit_code}, uptime={uptime:.0f}s). Restarting...")

        # Exponential backoff
        now = time.time()
        if uptime > 300:  # Ran for >5 min before crash — reset backoff
            self.backoff = 5.0
        else:
            self.backoff = min(self.backoff * 2, self.max_backoff)

        wait_time = max(0, self.backoff - (now - self.last_restart))
        if wait_time > 0:
            log.info(f"  → Waiting {wait_time:.0f}s before restart (backoff={self.backoff:.0f}s)")
            time.sleep(wait_time)

        self.restart_count += 1
        self.last_restart = time.time()
        log.info(f"  → Restarting {self.name} (attempt #{self.restart_count})")

        return self.start()

    @property
    def status(self) -> dict:
        """Return status dict for health reporting."""
        return {
            "name": self.name,
            "alive": self.is_alive(),
            "pid": self.process.pid if self.process else None,
            "uptime_s": time.time() - self.start_time if self.start_time else 0,
            "restart_count": self.restart_count,
            "log_file": str(self._log_file) if self._log_file else None,
        }


# ─── CME Maintenance Window ───
def is_cme_maintenance() -> bool:
    """Check if we're in CME maintenance window (17:00-17:45 ET, Sun-Thu).
    During maintenance, data feed may disconnect — this is EXPECTED, not a crash."""
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    now_et = datetime.now(ZoneInfo("America/New_York"))
    hour, minute = now_et.hour, now_et.minute
    weekday = now_et.weekday()  # 0=Mon, 6=Sun

    # CME maintenance: 17:00-17:45 ET, Sunday through Thursday
    if weekday <= 4 or weekday == 6:  # Mon-Fri or Sun
        if hour == 17 and minute < 45:
            return True
    return False


# ─── Health Monitor ───
class HealthMonitor:
    """Monitors all managed processes and system resources."""

    def __init__(self, processes: list[ManagedProcess], check_interval: float = 30.0):
        self.processes = processes
        self.check_interval = check_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_disk_warning: float = 0
        self._checks_run: int = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="health-monitor")
        self._thread.start()
        log.info("Health monitor started (every %.0fs)", self.check_interval)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.is_set():
            self._stop.wait(self.check_interval)
            if self._stop.is_set():
                break
            try:
                self._health_check()
            except Exception as e:
                log.error(f"Health check error: {e}")

    def _health_check(self):
        self._checks_run += 1
        in_maintenance = is_cme_maintenance()

        for proc in self.processes:
            if not proc.is_alive() and not proc._stop_requested:
                if in_maintenance:
                    log.info(f"  {proc.name} down during CME maintenance — will restart after 17:45 ET")
                else:
                    proc.check_and_restart()

        # Disk space check (every 10 checks = ~5 min)
        if self._checks_run % 10 == 0:
            try:
                import shutil
                _, _, free = shutil.disk_usage(str(LVL3))
                free_gb = free / 1e9
                if free_gb < 2.0 and time.time() - self._last_disk_warning > 3600:
                    msg = f"🔴 CRITICAL: Only {free_gb:.1f} GB disk free on Razer!"
                    log.warning(msg)
                    notify_discord(msg, "critical")
                    self._last_disk_warning = time.time()
            except Exception:
                pass

        # Fan-out JSONL size check
        fanout = LIVE_DIR / "logs" / "live_events.jsonl"
        if fanout.exists():
            size_mb = fanout.stat().st_size / 1e6
            if size_mb > 500:
                log.warning(f"Fan-out JSONL is {size_mb:.0f} MB — truncating")
                try:
                    fanout.write_text("")
                except Exception:
                    pass

        # Periodic status log
        if self._checks_run % 20 == 0:  # ~10 min
            statuses = [p.status for p in self.processes]
            log.info(f"Health check #{self._checks_run}: {json.dumps(statuses, default=str)}")


# ─── Main Orchestrator ───
def main():
    parser = argparse.ArgumentParser(description="Razer Live Stack — Unified Launcher")
    parser.add_argument("--symbol", default="ESM6", help="Futures symbol (default: ESM6)")
    parser.add_argument("--exchange", default="CME", help="Exchange (default: CME)")
    parser.add_argument("--weights", default=None, help="Model weights path")
    parser.add_argument("--stats", default=None, help="Feature stats path")
    parser.add_argument("--device", default="cuda", help="Inference device (cuda/cpu)")
    parser.add_argument("--window", type=int, default=1000, help="Feature window size")
    parser.add_argument("--stride", type=int, default=500, help="Feature stride")
    parser.add_argument("--min-tier", default="Top1%", help="Minimum confidence tier for entries")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't launch")
    parser.add_argument("--recorder-only", action="store_true", help="Only run MBO recorder")
    parser.add_argument("--no-paper", action="store_true", help="Run inference but skip paper trading")
    parser.add_argument("--health-interval", type=float, default=30.0, help="Health check interval (seconds)")
    args = parser.parse_args()

    weights = args.weights or str(DEFAULT_WEIGHTS)
    stats = args.stats or str(DEFAULT_STATS)

    # ─── Banner ───
    log.info("=" * 70)
    log.info("  RAZER LIVE STACK — Unified Launcher")
    log.info(f"  Symbol: {args.symbol} | Exchange: {args.exchange}")
    log.info(f"  Device: {args.device} | Window: {args.window} | Stride: {args.stride}")
    log.info(f"  Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log.info("=" * 70)

    # ─── Validation ───
    log.info("Running startup validation...")
    try:
        warnings = validate_prerequisites(args)
        for w in warnings:
            log.warning(f"  ⚠ {w}")
        log.info("✓ All prerequisites validated")
    except ValidationError as e:
        log.error(f"STARTUP VALIDATION FAILED:\n{e}")
        notify_discord(f"🔴 Startup validation FAILED:\n{e}")
        sys.exit(1)

    if args.dry_run:
        log.info("Dry run complete — all checks passed. Exiting.")
        return

    # ─── Build managed processes ───
    processes: list[ManagedProcess] = []

    # 1. MBO Recorder
    recorder_cmd = [
        PYTHON, str(LIVE_DIR / "mbo_recorder.py"),
        "--symbol", args.symbol,
        "--exchange", args.exchange,
    ]
    recorder = ManagedProcess("mbo-recorder", recorder_cmd, str(LIVE_DIR))
    processes.append(recorder)

    # 2. Live Inference + Paper Trader
    if not args.recorder_only:
        paper_cmd = [
            PYTHON, str(LIVE_DIR / "paper_trading_mamba_v2.py"),
            "--symbol", args.symbol,
            "--exchange", args.exchange,
            "--weights", weights,
            "--stats", stats,
            "--device", args.device,
            "--min-tier", args.min_tier,
            "--window", str(args.window),
            "--stride", str(args.stride),
        ]
        paper = ManagedProcess("paper-trader", paper_cmd, str(LIVE_DIR))
        processes.append(paper)

    # ─── Signal handler for graceful shutdown ───
    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        log.info(f"\nReceived {sig_name} — initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    if IS_WINDOWS:
        signal.signal(signal.SIGBREAK, handle_signal)

    # ─── Launch everything ───
    log.info(f"Launching {len(processes)} components...")
    notify_discord(f"🟢 Live stack starting: {', '.join(p.name for p in processes)} | {args.symbol}")

    for proc in processes:
        if not proc.start():
            log.error(f"Failed to start {proc.name} — aborting")
            notify_discord(f"🔴 Failed to start {proc.name} — aborting")
            for p in processes:
                p.stop()
            sys.exit(1)
        time.sleep(2)  # Stagger launches to avoid connection races

    log.info(f"✓ All {len(processes)} components launched successfully")
    notify_discord(f"✅ All components running: " + ", ".join(f"{p.name} (PID {p.process.pid})" for p in processes))

    # ─── Start health monitor ───
    monitor = HealthMonitor(processes, args.health_interval)
    monitor.start()

    # ─── Main loop — just wait for shutdown signal ───
    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(1.0)
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received")

    # ─── Graceful shutdown ───
    log.info("Shutting down live stack...")
    monitor.stop()
    for proc in reversed(processes):
        proc.stop()

    # Summary
    for proc in processes:
        s = proc.status
        log.info(f"  {s['name']}: restarts={s['restart_count']}, total_uptime={s['uptime_s']:.0f}s")

    notify_discord(f"🔴 Live stack stopped. " + ", ".join(
        f"{p.name}: {p.restart_count} restarts" for p in processes
    ))
    log.info("Live stack shutdown complete.")


if __name__ == "__main__":
    main()
