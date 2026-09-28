#!/usr/bin/env python3
"""
Waits for the arch benchmark (run_overnight_pc.py, PID from args or auto-detect)
to complete, then runs deep_benchmark.py with temporal/spatial models.

Usage:
    python run_deep_after_arch.py --wait-pid 34552
    python run_deep_after_arch.py  # auto-detects
"""
import gc
import json
import logging
import os
import psutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [deep_runner] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('deep_runner')

BASE = Path(__file__).parent.parent
FEATURE_CACHE = str(BASE / 'data' / 'processed' / 'mbo_features_cache')
SNAPSHOT_CACHE = str(BASE / 'data' / 'processed' / 'medium_snapshots_cache')
RESULTS_DIR = Path(__file__).parent / 'results'


def find_arch_benchmark_pid():
    """Find the running arch benchmark / overnight_pc process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            if 'run_overnight_pc' in cmd or ('arch_benchmark' in cmd and 'python' in proc.info['name'].lower()):
                mem_gb = proc.info['memory_info'].rss / (1024**3)
                if mem_gb > 1.0:  # Must be using significant memory
                    return proc.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def wait_for_pid(pid, check_interval=30):
    """Wait for a process to complete."""
    logger.info(f"Waiting for PID {pid} to complete...")
    while True:
        try:
            proc = psutil.Process(pid)
            mem_gb = proc.memory_info().rss / (1024**3)
            cpu = proc.cpu_percent(interval=1)
            logger.info(f"  PID {pid} still running (RAM={mem_gb:.1f}GB, CPU={cpu:.0f}%)")
        except psutil.NoSuchProcess:
            logger.info(f"  PID {pid} has completed!")
            return True
        time.sleep(check_interval)


def run_cmd(cmd, label):
    """Run a command and log output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STARTING: {label}")
    logger.info(f"{'='*60}")
    t0 = time.time()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        logger.info(f"  {line}")

    proc.wait()
    elapsed = time.time() - t0
    logger.info(f"COMPLETED: {label} ({elapsed:.0f}s, exit={proc.returncode})")
    return proc.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, default=None,
                        help='PID to wait for before starting')
    parser.add_argument('--no-wait', action='store_true',
                        help='Skip waiting, run immediately')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = RESULTS_DIR / f'deep_benchmark_{timestamp}.log'
    RESULTS_DIR.mkdir(exist_ok=True)

    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(logging.Formatter('%(asctime)s [deep_runner] %(message)s', '%H:%M:%S'))
    logger.addHandler(fh)

    # Also write to a live log
    live_log = RESULTS_DIR / 'deep_benchmark_live.log'
    lh = logging.FileHandler(str(live_log), mode='w')
    lh.setFormatter(logging.Formatter('%(asctime)s [deep_runner] %(message)s', '%H:%M:%S'))
    logger.addHandler(lh)

    logger.info("=" * 60)
    logger.info("DEEP LEARNING BENCHMARK SUITE")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 60)

    # Wait for arch benchmark to finish if needed
    if not args.no_wait:
        pid = args.wait_pid or find_arch_benchmark_pid()
        if pid:
            logger.info(f"Found arch benchmark PID: {pid}")
            wait_for_pid(pid, check_interval=60)
            logger.info("Arch benchmark finished. Starting deep benchmarks...")
            time.sleep(10)  # Brief pause
        else:
            logger.info("No arch benchmark running. Starting immediately.")

    python = sys.executable
    bench = str(Path(__file__).parent / 'deep_benchmark.py')

    # ============================================================
    # PHASE 1: Quick 20-day test of temporal models
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 1: Quick 20-day test of temporal models")
    logger.info("=" * 60)

    # Test each temporal model with top-50 features, window=20
    for arch in ['cnn', 'lstm', 'transformer']:
        rc = run_cmd([
            python, bench,
            '--arch', arch,
            '--n-days', '20',
            '--horizon', 'ret_10s',
            '--window', '20',
            '--top-k', '50',
            '--max-epochs', '15',
            '--subsample-train', '3',
            '--feature-cache', FEATURE_CACHE,
        ], f"20-day {arch.upper()} (window=20, top-50 features)")
        gc.collect()

    # ============================================================
    # PHASE 2: Quick test of spatial CNN (uses raw snapshots)
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 2: Quick 20-day Spatial CNN test")
    logger.info("=" * 60)

    rc = run_cmd([
        python, bench,
        '--arch', 'spatial_cnn',
        '--n-days', '20',
        '--horizon', 'ret_10s',
        '--window', '20',
        '--max-epochs', '15',
        '--subsample-train', '3',
        '--snapshot-cache', SNAPSHOT_CACHE,
        '--feature-cache', FEATURE_CACHE,
    ], "20-day Spatial CNN (raw book data)")
    gc.collect()

    # ============================================================
    # PHASE 3: Extended run of best temporal model (more epochs, more data)
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 3: Extended temporal CNN (more data/epochs)")
    logger.info("=" * 60)

    # CNN with 50 days, 25 epochs (proven fastest and often competitive)
    rc = run_cmd([
        python, bench,
        '--arch', 'cnn',
        '--n-days', '50',
        '--horizon', 'ret_10s',
        '--window', '20',
        '--top-k', '80',
        '--max-epochs', '25',
        '--subsample-train', '2',
        '--feature-cache', FEATURE_CACHE,
    ], "50-day CNN extended (80 features, 25 epochs)")
    gc.collect()

    # Transformer with 50 days
    rc = run_cmd([
        python, bench,
        '--arch', 'transformer',
        '--n-days', '50',
        '--horizon', 'ret_10s',
        '--window', '20',
        '--top-k', '80',
        '--max-epochs', '25',
        '--subsample-train', '2',
        '--feature-cache', FEATURE_CACHE,
    ], "50-day Transformer extended (80 features, 25 epochs)")
    gc.collect()

    # ============================================================
    # PHASE 4: Window size comparison
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 4: Window size comparison (CNN)")
    logger.info("=" * 60)

    for window in [10, 20, 50]:
        rc = run_cmd([
            python, bench,
            '--arch', 'cnn',
            '--n-days', '20',
            '--horizon', 'ret_10s',
            '--window', str(window),
            '--top-k', '50',
            '--max-epochs', '15',
            '--subsample-train', '3',
            '--feature-cache', FEATURE_CACHE,
        ], f"CNN window={window} comparison")
        gc.collect()

    # ============================================================
    # PHASE 5: Multi-horizon deep comparison
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 5: Multi-horizon CNN comparison")
    logger.info("=" * 60)

    for horizon in ['ret_3s', 'ret_5s', 'ret_10s']:
        rc = run_cmd([
            python, bench,
            '--arch', 'cnn',
            '--n-days', '30',
            '--horizon', horizon,
            '--window', '20',
            '--top-k', '50',
            '--max-epochs', '15',
            '--subsample-train', '3',
            '--feature-cache', FEATURE_CACHE,
        ], f"CNN @ {horizon}")
        gc.collect()

    logger.info("\n\n" + "=" * 60)
    logger.info("DEEP LEARNING BENCHMARK COMPLETE")
    logger.info(f"Finished: {datetime.now()}")
    logger.info(f"Results in: {RESULTS_DIR}")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
