#!/usr/bin/env python3
"""
Durable Watchdog — Runs independently of Claude sessions
=========================================================

PM2-managed persistent service that:
1. Monitors Razer paper trader health (SSH heartbeat check)
2. Monitors Neptune training progress (log freshness)
3. Auto-restarts Razer paper trader if dead
4. Logs everything to disk (JSON structured logs)
5. Sends alerts to Discord via bot token (NOT via Claude)
6. Runs 24/7, survives Claude crashes/resets

HC #242: "The system should work fully functional without Claude doing anything."

Managed by PM2: pm2 start durable_watchdog.py --name razer-watchdog --interpreter python3

Author: Claude (Infrastructure Builder)
Date: 2026-05-07
"""

import os
import sys
import json
import time
import signal
import logging
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

# ─── Configuration ────────────────────────────────────────────────────────────

# Paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
LOG_DIR = BASE_DIR / "infra" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

HEARTBEAT_FILE = LOG_DIR / "watchdog_heartbeat.json"
EVENTS_LOG = LOG_DIR / "watchdog_events.jsonl"
STATUS_LOG = LOG_DIR / "cluster_status.json"

# Discord
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_CHANNEL_SYSTEM_STATUS = os.environ.get("DISCORD_SYSTEM_STATUS_CHANNEL_ID", "")  # #systemStatus channel
DISCORD_CHANNEL_GENERAL = os.environ.get("DISCORD_GENERAL_CHANNEL_ID", "") # #general channel

# SSH credentials
RAZER_HOST = "razer"
RAZER_USER = "claude"
RAZER_PASS = os.environ.get("CLUSTER_SSH_PASSWORD", "")

NEPTUNE_HOST = "neptune"
NEPTUNE_USER = "nick"

# Weekend/holiday awareness (HC #492 — don't alert on expected weekend downtime)
def _is_weekend():
    """Paper trader is intentionally off on weekends. Don't alert."""
    from datetime import datetime
    try:
        import zoneinfo
        n = datetime.now(zoneinfo.ZoneInfo("America/New_York"))
    except Exception:
        n = datetime.now()
    return n.weekday() >= 5  # Saturday=5, Sunday=6

# Check intervals (seconds)
RAZER_CHECK_INTERVAL = 120      # Check Razer every 2 minutes
NEPTUNE_CHECK_INTERVAL = 300    # Check Neptune every 5 minutes
HEARTBEAT_INTERVAL = 60         # Write own heartbeat every 60s
DISCORD_ALERT_COOLDOWN = 900    # Don't spam Discord — 15 min between identical alerts

# Razer auto-restart
RAZER_STALE_THRESHOLD = 600     # Consider Razer dead if no new log output in 10 min
RAZER_MAX_RESTART_ATTEMPTS = 3  # Max restarts before alerting human
RAZER_RESTART_COOLDOWN = 300    # 5 min between restart attempts

# ─── Logging Setup ────────────────────────────────────────────────────────────

logger = logging.getLogger("watchdog")
logger.setLevel(logging.INFO)

# Console
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# Rotating file log
fh = RotatingFileHandler(LOG_DIR / "watchdog.log", maxBytes=10_000_000, backupCount=5)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)


# ─── Discord Messaging ───────────────────────────────────────────────────────

def send_discord(channel_id: str, message: str) -> bool:
    """Send a message to Discord via bot API. No Claude dependency."""
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    data = json.dumps({"content": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "TradingBot/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 201):
                return True
    except Exception as e:
        logger.error(f"Discord send failed: {e}")
    return False


def alert_discord(message: str, channel: str = "alerts"):
    """Send alert to Discord with cooldown to prevent spam."""
    channel_id = DISCORD_CHANNEL_SYSTEM_STATUS if channel == "alerts" else DISCORD_CHANNEL_GENERAL
    # Check cooldown
    cooldown_key = hash(message[:50])
    now = time.time()
    if cooldown_key in _alert_cooldowns:
        if now - _alert_cooldowns[cooldown_key] < DISCORD_ALERT_COOLDOWN:
            logger.debug(f"Alert suppressed (cooldown): {message[:50]}")
            return
    _alert_cooldowns[cooldown_key] = now
    send_discord(channel_id, message)

_alert_cooldowns: dict = {}


# ─── SSH Command Execution ───────────────────────────────────────────────────

def ssh_exec(host: str, user: str, command: str, timeout: int = 15, password: str = None) -> tuple:
    """Execute command via SSH. Returns (success, stdout, stderr)."""
    ssh_cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", f"ConnectTimeout={timeout}",
               "-o", "BatchMode=yes", f"{user}@{host}", command]

    # For password auth, use sshpass if available
    if password:
        ssh_cmd = ["sshpass", f"-p{password}"] + ssh_cmd

    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout + 5)
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "SSH timeout"
    except FileNotFoundError:
        # sshpass not installed, try without password (key auth)
        ssh_cmd_nopass = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", f"ConnectTimeout={timeout}",
                          f"{user}@{host}", command]
        try:
            result = subprocess.run(ssh_cmd_nopass, capture_output=True, text=True, timeout=timeout + 5)
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except Exception as e:
            return False, "", str(e)
    except Exception as e:
        return False, "", str(e)


# ─── Event Logging ───────────────────────────────────────────────────────────

def log_event(event_type: str, node: str, details: dict):
    """Append structured event to JSONL log."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "node": node,
        **details,
    }
    try:
        with open(EVENTS_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"Failed to log event: {e}")


# ─── Razer Monitoring ────────────────────────────────────────────────────────

class RazerMonitor:
    """Monitors Razer paper trader health and auto-restarts if dead."""

    def __init__(self):
        self.last_check = 0
        self.last_restart = 0
        self.restart_count = 0
        self.last_status = "unknown"
        self.last_trade_count = 0
        self.last_event_count = 0
        self.consecutive_failures = 0

    def check(self) -> dict:
        """Check Razer paper trader health. Returns status dict."""
        now = time.time()
        if now - self.last_check < RAZER_CHECK_INTERVAL:
            return {"status": self.last_status, "skipped": True}
        self.last_check = now

        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "razer",
            "python_procs": 0,
            "status": "unknown",
            "details": "",
        }

        # Check if python processes are running
        ok, stdout, stderr = ssh_exec(RAZER_HOST, RAZER_USER,
                                       'tasklist /FI "IMAGENAME eq python.exe" /NH 2>nul',
                                       password=RAZER_PASS)

        if not ok:
            status["status"] = "unreachable"
            status["details"] = f"SSH failed: {stderr[:100]}"
            self.consecutive_failures += 1
            log_event("razer_check", "razer", status)

            if self.consecutive_failures >= 3:
                alert_discord(f"**RAZER UNREACHABLE** — SSH failed {self.consecutive_failures}x: {stderr[:80]}")
            self.last_status = "unreachable"
            return status

        self.consecutive_failures = 0

        # Count python processes
        python_lines = [l for l in stdout.split("\n") if "python.exe" in l.lower()]
        status["python_procs"] = len(python_lines)

        if len(python_lines) == 0:
            status["status"] = "dead"
            status["details"] = "No python processes found"
            log_event("razer_dead", "razer", status)
            logger.warning("RAZER: Paper trader DEAD — no python processes!")
            self._handle_dead(status)

        elif len(python_lines) >= 2:
            # Check memory (models loaded = ~1GB+ per process)
            total_mem_kb = 0
            for line in python_lines:
                parts = line.split()
                # Find the memory column (usually the last numeric value with 'K' suffix)
                for part in reversed(parts):
                    cleaned = part.replace(",", "").replace("K", "")
                    if cleaned.isdigit():
                        total_mem_kb += int(cleaned)
                        break

            if total_mem_kb > 500_000:  # > 500MB = models loaded
                status["status"] = "healthy"
                status["details"] = f"{len(python_lines)} procs, {total_mem_kb//1024}MB total"
            else:
                status["status"] = "degraded"
                status["details"] = f"{len(python_lines)} procs but only {total_mem_kb//1024}MB — models may not be loaded"
                log_event("razer_degraded", "razer", status)

        else:
            status["status"] = "degraded"
            status["details"] = f"Only {len(python_lines)} python proc (expected 2)"
            log_event("razer_degraded", "razer", status)

        self.last_status = status["status"]
        log_event("razer_check", "razer", status)
        return status

    def _handle_dead(self, status: dict):
        """Handle dead Razer — attempt auto-restart.
        HC #492: Skip all alerting/restart on weekends — paper trader intentionally off.
        """
        if _is_weekend():
            logger.info("Razer dead but it's the weekend — paper trader expected to be off. Skipping alert/restart.")
            return

        now = time.time()

        if self.restart_count >= RAZER_MAX_RESTART_ATTEMPTS:
            alert_discord(
                f"**RAZER DEAD — AUTO-RESTART FAILED {self.restart_count}x** "
                f"Manual intervention required. Paper trader is not running.",
                channel="general"
            )
            log_event("razer_restart_exhausted", "razer", {"attempts": self.restart_count})
            return

        if now - self.last_restart < RAZER_RESTART_COOLDOWN:
            logger.info(f"Razer restart cooldown ({RAZER_RESTART_COOLDOWN}s), waiting...")
            return

        logger.info(f"Attempting Razer auto-restart (attempt {self.restart_count + 1}/{RAZER_MAX_RESTART_ATTEMPTS})")
        self.last_restart = now
        self.restart_count += 1

        # Try to restart the paper trader
        restart_cmd = (
            'cd C:\\Users\\claude\\Lvl3Quant && '
            'set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python && '
            'start /b python -m alpha_discovery.live.paper_trading_mamba_v2_patched '
            '--config paper_trader_config.json '
            '> logs\\paper_trader_auto_%date:~0,4%%date:~5,2%%date:~8,2%.log 2>&1'
        )

        ok, stdout, stderr = ssh_exec(RAZER_HOST, RAZER_USER, restart_cmd,
                                       timeout=30, password=RAZER_PASS)

        if ok:
            logger.info("Razer restart command sent successfully")
            alert_discord(
                f"**RAZER AUTO-RESTARTED** (attempt {self.restart_count}) — "
                f"Paper trader was dead, sent restart command. Verifying in 2 min...",
                channel="alerts"
            )
            log_event("razer_restart", "razer", {"attempt": self.restart_count, "success": True})
        else:
            logger.error(f"Razer restart failed: {stderr}")
            log_event("razer_restart", "razer", {"attempt": self.restart_count, "success": False, "error": stderr[:200]})


# ─── Neptune Monitoring ──────────────────────────────────────────────────────

class NeptuneMonitor:
    """Monitors Neptune training progress via log freshness."""

    def __init__(self):
        self.last_check = 0
        self.last_status = "unknown"
        self.last_log_line = ""
        self.consecutive_stalls = 0

    def check(self) -> dict:
        """Check Neptune training health."""
        now = time.time()
        if now - self.last_check < NEPTUNE_CHECK_INTERVAL:
            return {"status": self.last_status, "skipped": True}
        self.last_check = now

        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "neptune",
            "status": "unknown",
            "details": "",
        }

        # Check training log freshness
        ok, stdout, stderr = ssh_exec(NEPTUNE_HOST, NEPTUNE_USER,
                                       'tail -3 /tmp/split_dqn_v3_all56.log 2>/dev/null; '
                                       'echo "---MEM---"; free -h | grep Mem')

        if not ok:
            status["status"] = "unreachable"
            status["details"] = f"SSH failed: {stderr[:100]}"
            log_event("neptune_check", "neptune", status)
            self.last_status = "unreachable"
            return status

        lines = stdout.strip().split("\n")
        log_lines = [l for l in lines if "split_dqn" in l or "fifo_rl" in l]
        mem_lines = [l for l in lines if "Mem" in l or "Gi" in l]

        if log_lines:
            latest = log_lines[-1]
            # Check if log is fresh (contains today's date or recent timestamp)
            status["last_log"] = latest[:120]

            # Parse progress if available
            if "Ep " in latest and "trades=" in latest:
                status["status"] = "training"
                status["details"] = latest.split("]")[-1].strip() if "]" in latest else latest[:100]
                self.consecutive_stalls = 0
            elif latest == self.last_log_line:
                self.consecutive_stalls += 1
                if self.consecutive_stalls >= 3:  # 15 min stall
                    status["status"] = "stalled"
                    status["details"] = f"Log unchanged for {self.consecutive_stalls * NEPTUNE_CHECK_INTERVAL}s"
                    alert_discord(f"**NEPTUNE STALLED** — Training log unchanged for {self.consecutive_stalls * NEPTUNE_CHECK_INTERVAL // 60} min")
                else:
                    status["status"] = "training"
                    status["details"] = "Processing (between log entries)"
            else:
                status["status"] = "training"
                status["details"] = "Active (new log entries)"
                self.consecutive_stalls = 0

            self.last_log_line = latest
        else:
            status["status"] = "no_training"
            status["details"] = "No training log found"

        if mem_lines:
            status["memory"] = mem_lines[0].strip()

        self.last_status = status["status"]
        log_event("neptune_check", "neptune", status)
        return status


# ─── Jupiter Self-Monitoring ─────────────────────────────────────────────────

class JupiterMonitor:
    """Monitors local Jupiter processes."""

    def __init__(self):
        self.last_check = 0

    def check(self) -> dict:
        """Check Jupiter local training processes."""
        now = time.time()
        if now - self.last_check < NEPTUNE_CHECK_INTERVAL:
            return {"status": "skipped"}
        self.last_check = now

        status = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": "jupiter",
            "status": "unknown",
            "processes": [],
        }

        # Check for training processes
        try:
            result = subprocess.run(
                ["pgrep", "-af", "train_exit_dqn|train_split_dqn|train_supervised|precompute"],
                capture_output=True, text=True, timeout=5
            )
            if result.stdout.strip():
                procs = result.stdout.strip().split("\n")
                status["processes"] = procs[:5]
                status["status"] = "training"
                status["details"] = f"{len(procs)} training process(es)"
            else:
                status["status"] = "idle"
                status["details"] = "No training processes"
        except Exception as e:
            status["status"] = "error"
            status["details"] = str(e)

        # Check exit DQN log if running
        exit_log = Path("/tmp/exit_dqn_v1.log")
        if exit_log.exists():
            try:
                with open(exit_log, "r") as f:
                    f.seek(max(0, exit_log.stat().st_size - 2000))
                    tail = f.read().strip().split("\n")
                    for line in reversed(tail):
                        if "Fold" in line and "Ep" in line:
                            status["exit_dqn_progress"] = line.strip()[-120:]
                            break
            except Exception:
                pass

        log_event("jupiter_check", "jupiter", status)
        return status


# ─── Status Writer ────────────────────────────────────────────────────────────

def write_status(razer: dict, neptune: dict, jupiter: dict):
    """Write combined cluster status to disk — readable by Claude or user."""
    status = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "watchdog_uptime_s": time.time() - _start_time,
        "nodes": {
            "razer": razer,
            "neptune": neptune,
            "jupiter": jupiter,
        }
    }
    try:
        with open(STATUS_LOG, "w") as f:
            json.dump(status, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to write status: {e}")


# ─── Heartbeat ────────────────────────────────────────────────────────────────

def write_heartbeat():
    """Write own heartbeat so other systems know watchdog is alive."""
    hb = {
        "pid": os.getpid(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_s": time.time() - _start_time,
        "checks_total": _check_count,
    }
    try:
        with open(HEARTBEAT_FILE, "w") as f:
            json.dump(hb, f, indent=2)
    except Exception:
        pass


# ─── Morning Briefing ────────────────────────────────────────────────────────

def send_morning_briefing(razer_status: dict, neptune_status: dict, jupiter_status: dict):
    """Send morning briefing to Discord at 8:23 AM ET."""
    msg = (
        f"**Morning Briefing — {datetime.now().strftime('%a %b %d, %I:%M %p ET')}**\n\n"
        f"**Razer**: {razer_status.get('status', 'unknown')} — {razer_status.get('details', 'N/A')}\n"
        f"**Neptune**: {neptune_status.get('status', 'unknown')} — {neptune_status.get('details', 'N/A')}\n"
        f"**Jupiter**: {jupiter_status.get('status', 'unknown')} — {jupiter_status.get('details', 'N/A')}\n"
        f"\n_Sent by durable watchdog (PID {os.getpid()})_"
    )
    send_discord(DISCORD_CHANNEL_GENERAL, msg)
    log_event("morning_briefing", "all", {"message": msg[:200]})


def send_eod_summary(razer_status: dict, neptune_status: dict, jupiter_status: dict):
    """Send EOD summary to Discord at 3:41 PM ET."""
    msg = (
        f"**EOD Summary — {datetime.now().strftime('%a %b %d, %I:%M %p ET')}**\n\n"
        f"**Razer**: {razer_status.get('status', 'unknown')} — {razer_status.get('details', 'N/A')}\n"
        f"**Neptune**: {neptune_status.get('status', 'unknown')} — {neptune_status.get('details', 'N/A')}\n"
        f"**Jupiter**: {jupiter_status.get('status', 'unknown')} — {jupiter_status.get('details', 'N/A')}\n"
        f"Watchdog uptime: {(time.time() - _start_time) / 3600:.1f} hours, {_check_count} checks\n"
        f"\n_Sent by durable watchdog (PID {os.getpid()})_"
    )
    send_discord(DISCORD_CHANNEL_GENERAL, msg)
    log_event("eod_summary", "all", {"message": msg[:200]})


# ─── Main Loop ────────────────────────────────────────────────────────────────

_start_time = time.time()
_check_count = 0
_last_heartbeat = 0
_morning_sent_today = False
_eod_sent_today = False

def main():
    global _check_count, _last_heartbeat, _morning_sent_today, _eod_sent_today

    logger.info(f"Durable Watchdog starting (PID {os.getpid()})")
    logger.info(f"Log dir: {LOG_DIR}")
    logger.info(f"Razer check interval: {RAZER_CHECK_INTERVAL}s")
    logger.info(f"Neptune check interval: {NEPTUNE_CHECK_INTERVAL}s")

    # Send startup alert
    send_discord(DISCORD_CHANNEL_SYSTEM_STATUS,
                 f"**Durable Watchdog STARTED** (PID {os.getpid()}) — "
                 f"Monitoring Razer every {RAZER_CHECK_INTERVAL}s, Neptune every {NEPTUNE_CHECK_INTERVAL}s. "
                 f"Auto-restart enabled for Razer. Logs: {LOG_DIR}")

    razer = RazerMonitor()
    neptune = NeptuneMonitor()
    jupiter = JupiterMonitor()

    # Graceful shutdown
    def shutdown(sig, frame):
        logger.info("Watchdog shutting down gracefully")
        send_discord(DISCORD_CHANNEL_SYSTEM_STATUS, f"**Watchdog STOPPED** (signal {sig})")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    while True:
        try:
            now = time.time()

            # Run checks
            razer_status = razer.check()
            neptune_status = neptune.check()
            jupiter_status = jupiter.check()
            _check_count += 1

            # Write combined status
            if not razer_status.get("skipped") or not neptune_status.get("skipped"):
                write_status(razer_status, neptune_status, jupiter_status)

            # Write heartbeat
            if now - _last_heartbeat >= HEARTBEAT_INTERVAL:
                write_heartbeat()
                _last_heartbeat = now

            # Scheduled messages (check hour/minute in ET)
            from datetime import timezone as tz
            from zoneinfo import ZoneInfo
            et_now = datetime.now(ZoneInfo("America/New_York"))
            today_date = et_now.strftime("%Y-%m-%d")

            # DISABLED 2026-05-21 per user verbatim 7:15 AM ET:
            # "please remove anythng about a morning briefing or EOD briefing..
            #  I'll ASK YOU when I'm ready..."
            # User will request status explicitly. No scheduled-time briefings.
            # ORIGINAL morning-briefing block at 8:20-8:25 AM ET removed.
            # ORIGINAL EOD-summary block at 3:38-3:43 PM ET removed.
            pass

            # Reset daily flags
            if et_now.hour == 0 and et_now.minute < 2:
                _morning_sent_today = False
                _eod_sent_today = False

            # Sleep — short interval for responsive monitoring
            time.sleep(30)

        except KeyboardInterrupt:
            logger.info("Watchdog stopped by user")
            break
        except Exception as e:
            logger.error(f"Watchdog loop error: {e}", exc_info=True)
            time.sleep(60)  # Back off on errors


if __name__ == "__main__":
    main()
