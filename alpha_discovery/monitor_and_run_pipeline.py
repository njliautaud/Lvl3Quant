"""
Monitor cache rebuild and auto-launch 27-day pipeline when complete.

Polls for PID completion and cache file count, then runs the pipeline.

Usage:
    python alpha_discovery/monitor_and_run_pipeline.py
    python alpha_discovery/monitor_and_run_pipeline.py --pid 12492
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
CACHE_DIR = ROOT / "data" / "processed" / "medium_snapshots_cache"
LOG_FILE = ROOT / "alpha_discovery" / "results" / "cache_rebuild_alldays.log"
PIPELINE_SCRIPT = ROOT / "alpha_discovery" / "run_full_27day_pipeline.py"

EXPECTED_FILES = 27  # 27 .dbn files
MIN_FILES = 20       # minimum to proceed

from alpha_discovery.process_registry import ProcessRegistry
registry = ProcessRegistry()


def is_pid_running(pid):
    """Check if a process is still running."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True, timeout=10
        )
        return str(pid) in result.stdout
    except Exception:
        return False


def count_cache_files():
    """Count completed cache files."""
    return len(list(CACHE_DIR.glob("file_*_snapshots.npz")))


def get_log_tail(n=3):
    """Get last N lines of rebuild log."""
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
        return [l.strip() for l in lines[-n:]]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, default=12492,
                        help='PID of cache rebuild process')
    parser.add_argument('--poll-interval', type=int, default=60,
                        help='Poll interval in seconds')
    parser.add_argument('--skip-feature-cache', action='store_true',
                        help='Pass --skip-feature-cache to pipeline')
    args = parser.parse_args()

    print(f"Monitoring cache rebuild (PID {args.pid})...")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Expected: {EXPECTED_FILES} files, minimum: {MIN_FILES}")
    print(f"Polling every {args.poll_interval}s")
    print()

    while True:
        n_files = count_cache_files()
        pid_alive = is_pid_running(args.pid)
        tail = get_log_tail(2)

        ts = time.strftime('%H:%M:%S')
        print(f"[{ts}] Cache files: {n_files}/{EXPECTED_FILES} | "
              f"PID {args.pid}: {'RUNNING' if pid_alive else 'DONE'}")
        if tail:
            print(f"  Last log: {tail[-1][:100]}")

        if not pid_alive:
            print(f"\nCache rebuild process (PID {args.pid}) has exited.")
            print(f"Total cache files: {n_files}")

            if n_files >= MIN_FILES:
                print(f"\n{'='*60}")
                print(f"LAUNCHING FULL 27-DAY PIPELINE")
                print(f"{'='*60}\n")

                cmd = [sys.executable, str(PIPELINE_SCRIPT),
                       '--skip-feature-cache'] if args.skip_feature_cache else \
                      [sys.executable, str(PIPELINE_SCRIPT)]

                # Run pipeline (this will take a while)
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                )
                registry.register(pid=proc.pid, name="full_27day_pipeline",
                                  launched_by="monitor_and_run", tags=["pipeline"])
                proc.wait()
                registry.unregister(proc.pid)
                print(f"\nPipeline exited with code {proc.returncode}")
            else:
                print(f"WARNING: Only {n_files} files, expected {EXPECTED_FILES}+")
                print("Cache rebuild may have failed. Check log.")

            break

        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
