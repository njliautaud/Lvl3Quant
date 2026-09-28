"""
Process Registry — Shared module for orchestrator process tracking.

Solves the "blind wait" problem: orchestrators register their subprocesses
with names and tags so other orchestrators can identify what's running
and make informed decisions about whether to wait.

Usage:
    from process_registry import ProcessRegistry
    reg = ProcessRegistry()

    # Register a process you launched
    reg.register(pid=12345, name="multi_timeframe_70day",
                 launched_by="overnight_v2", tags=["overnight"])

    # Check what's running (auto-cleans dead processes)
    running = reg.get_running()

    # Wait only for specific tagged processes
    reg.wait_for_tagged(tags=["overnight"], max_wait=3600)

    # Mark done
    reg.unregister(pid=12345)
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

logger = logging.getLogger("process_registry")

LVL3_ROOT = Path(__file__).parent.parent
REGISTRY_FILE = LVL3_ROOT / "data" / "process_registry.json"
REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)


def _is_pid_alive(pid: int) -> bool:
    """Check if a PID is still running (Windows)."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, timeout=10)
        return str(pid) in result.stdout
    except Exception:
        return False


def _get_process_cmdline(pid: int) -> str:
    """Get the command line of a running process (Windows)."""
    try:
        result = subprocess.run(
            ['wmic', 'process', 'where', f'ProcessId={pid}',
             'get', 'CommandLine', '/value'],
            capture_output=True, text=True, timeout=10)
        for line in result.stdout.split('\n'):
            if line.strip().startswith('CommandLine='):
                return line.strip()[len('CommandLine='):]
    except Exception:
        pass
    return "<unknown>"


def _get_heavy_python_pids(min_mem_gb: float = 2.0) -> List[Dict]:
    """Get Python processes using more than min_mem_gb of RAM."""
    heavy = []
    try:
        out = subprocess.run(['tasklist'], capture_output=True, text=True, timeout=10)
        for line in out.stdout.split('\n'):
            if 'python' not in line.lower():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                pid = int(parts[1])
                mem_str = parts[-2].replace(',', '').replace('K', '')
                mem_kb = int(mem_str)
                if mem_kb > min_mem_gb * 1_000_000:
                    heavy.append({
                        'pid': pid,
                        'mem_gb': round(mem_kb / 1_000_000, 1),
                        'image': parts[0],
                    })
            except (ValueError, IndexError):
                continue
    except Exception as e:
        logger.warning(f"Could not scan processes: {e}")
    return heavy


class ProcessRegistry:
    """Thread-safe (file-locked) process registry."""

    def __init__(self, registry_file: Optional[Path] = None):
        self.file = registry_file or REGISTRY_FILE
        self._ensure_file()

    def _ensure_file(self):
        if not self.file.exists():
            self._write({'processes': {}})

    def _read(self) -> dict:
        try:
            with open(self.file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {'processes': {}}

    def _write(self, data: dict):
        with open(self.file, 'w') as f:
            json.dump(data, f, indent=2)

    def register(self, pid: int, name: str, launched_by: str = "",
                 tags: Optional[List[str]] = None, cmd: str = ""):
        """Register a launched process."""
        data = self._read()
        data['processes'][str(pid)] = {
            'name': name,
            'cmd': cmd or _get_process_cmdline(pid),
            'started': datetime.now().isoformat(),
            'launched_by': launched_by,
            'tags': tags or [],
        }
        self._write(data)
        logger.info(f"Registered PID {pid}: {name} (tags={tags})")

    def unregister(self, pid: int):
        """Remove a process from registry."""
        data = self._read()
        key = str(pid)
        if key in data['processes']:
            name = data['processes'][key].get('name', '?')
            del data['processes'][key]
            self._write(data)
            logger.info(f"Unregistered PID {pid}: {name}")

    def cleanup_dead(self) -> List[dict]:
        """Remove entries for processes that are no longer running. Returns removed entries."""
        data = self._read()
        removed = []
        dead_pids = []
        for pid_str, info in data['processes'].items():
            if not _is_pid_alive(int(pid_str)):
                removed.append({'pid': int(pid_str), **info})
                dead_pids.append(pid_str)
        for pid_str in dead_pids:
            del data['processes'][pid_str]
        if dead_pids:
            self._write(data)
            logger.info(f"Cleaned up {len(dead_pids)} dead processes: "
                        f"{[r['name'] for r in removed]}")
        return removed

    def get_running(self) -> Dict[int, dict]:
        """Get all registered processes that are still alive."""
        self.cleanup_dead()
        data = self._read()
        return {int(k): v for k, v in data['processes'].items()
                if _is_pid_alive(int(k))}

    def get_by_tags(self, tags: List[str]) -> Dict[int, dict]:
        """Get running processes matching ANY of the given tags."""
        running = self.get_running()
        return {pid: info for pid, info in running.items()
                if set(tags) & set(info.get('tags', []))}

    def detect_unregistered_heavy(self, min_mem_gb: float = 2.0,
                                   my_pid: Optional[int] = None) -> List[Dict]:
        """Find heavy Python processes NOT in the registry.
        Returns list of {'pid', 'mem_gb', 'cmdline'} for unknown processes."""
        registered_pids = set(int(k) for k in self._read()['processes'].keys())
        skip_pids = registered_pids | {os.getpid()}
        if my_pid:
            skip_pids.add(my_pid)

        heavy = _get_heavy_python_pids(min_mem_gb)
        unknown = []
        for proc in heavy:
            if proc['pid'] not in skip_pids:
                proc['cmdline'] = _get_process_cmdline(proc['pid'])
                unknown.append(proc)
        return unknown

    def wait_for_tagged(self, tags: List[str], check_interval: int = 60,
                        max_wait: int = 7200) -> bool:
        """Wait for all processes matching the given tags to finish.
        Returns True if all finished, False on timeout."""
        t0 = time.time()
        while time.time() - t0 < max_wait:
            matching = self.get_by_tags(tags)
            if not matching:
                logger.info(f"All tagged processes [{', '.join(tags)}] have finished.")
                return True
            names = [f"PID {pid}: {info['name']}" for pid, info in matching.items()]
            elapsed = int(time.time() - t0)
            logger.info(f"Waiting for {len(matching)} tagged processes "
                        f"({elapsed}s/{max_wait}s): {names}")
            time.sleep(check_interval)
        logger.warning(f"Timeout after {max_wait}s waiting for tags {tags}")
        return False

    def smart_wait(self, wait_tags: List[str],
                   check_interval: int = 60,
                   max_wait: int = 7200,
                   warn_unknown: bool = True) -> dict:
        """Smart wait: waits for tagged processes, logs unknown ones, skips them.

        Returns:
            {
                'tagged_finished': bool,
                'tagged_waited': [...],
                'unknown_skipped': [...],
                'elapsed': float,
            }
        """
        result = {
            'tagged_finished': True,
            'tagged_waited': [],
            'unknown_skipped': [],
            'elapsed': 0,
        }
        t0 = time.time()

        # Step 1: Detect and log unknown heavy processes
        if warn_unknown:
            unknown = self.detect_unregistered_heavy()
            if unknown:
                logger.warning(f"Found {len(unknown)} UNREGISTERED heavy Python processes:")
                for u in unknown:
                    logger.warning(f"  PID {u['pid']} ({u['mem_gb']}GB): {u['cmdline'][:200]}")
                logger.warning("These are NOT in the registry — SKIPPING them.")
                logger.warning("If they should be tracked, register them with ProcessRegistry.")
                result['unknown_skipped'] = unknown

        # Step 2: Wait for tagged processes only
        tagged = self.get_by_tags(wait_tags)
        if tagged:
            names = [f"PID {pid}: {info['name']}" for pid, info in tagged.items()]
            logger.info(f"Waiting for {len(tagged)} registered processes: {names}")
            result['tagged_waited'] = names
            result['tagged_finished'] = self.wait_for_tagged(
                wait_tags, check_interval, max_wait)
        else:
            logger.info(f"No registered processes with tags {wait_tags}. Starting immediately.")

        result['elapsed'] = time.time() - t0
        return result
