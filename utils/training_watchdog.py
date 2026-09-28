#!/usr/bin/env python3
"""
Training Watchdog - Kills zombie processes that produce no logs.

BAND-AID SOLUTION for 2026-04-18 zombie failures:
- Both Neptune and Razer had processes running 8+ hours with 0 log output
- No way to diagnose = wasted GPU time
- This watchdog kills ANY process that runs >10min with empty log file

Usage:
    python utils/training_watchdog.py --log-file /path/to/training.log --pid 12345 --max-silent-minutes 10
"""
import argparse
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Kill training process if log file stays empty")
    parser.add_argument("--log-file", required=True, help="Path to log file to monitor")
    parser.add_argument("--pid", type=int, required=True, help="Process ID to monitor")
    parser.add_argument("--max-silent-minutes", type=int, default=10, help="Max minutes with empty log before kill")
    parser.add_argument("--check-interval", type=int, default=30, help="Seconds between checks")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    start_time = time.time()

    print(f"[WATCHDOG] Monitoring PID {args.pid}, log: {log_path}")
    print(f"[WATCHDOG] Will kill if log stays empty for >{args.max_silent_minutes} min")

    while True:
        time.sleep(args.check_interval)

        # Check if process still exists
        try:
            os.kill(args.pid, 0)  # Signal 0 = check existence
        except OSError:
            print(f"[WATCHDOG] Process {args.pid} no longer exists. Exiting.")
            sys.exit(0)

        # Check log file size
        if not log_path.exists():
            elapsed_min = (time.time() - start_time) / 60
            if elapsed_min > args.max_silent_minutes:
                print(f"[WATCHDOG] KILLING PID {args.pid} - No log file after {elapsed_min:.1f} min")
                os.kill(args.pid, 9)  # SIGKILL
                sys.exit(1)
            continue

        log_size = log_path.stat().st_size

        if log_size == 0:
            elapsed_min = (time.time() - start_time) / 60
            if elapsed_min > args.max_silent_minutes:
                print(f"[WATCHDOG] KILLING PID {args.pid} - Empty log for {elapsed_min:.1f} min")
                os.kill(args.pid, 9)
                sys.exit(1)
            print(f"[WATCHDOG] Empty log, {elapsed_min:.1f}/{args.max_silent_minutes} min")
        else:
            print(f"[WATCHDOG] Log active: {log_size} bytes")
            # Reset timer once we see log output
            start_time = time.time()


if __name__ == "__main__":
    main()
