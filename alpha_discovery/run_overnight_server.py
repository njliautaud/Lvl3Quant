#!/usr/bin/env python3
"""
Overnight Runner — Server (CPU only, 16 cores, 46GB RAM)
Runs CPU-optimized architecture benchmarks on Jupiter server.

Order of operations:
1. Quick 20-day test of CPU architectures (LightGBM, XGBoost, CatBoost)
2. Full 70-day run of all three
3. Ensemble experiments: stacking different GBM types

Estimated time: ~8-12 hours total (CPU is slower than GPU)
"""
import gc
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [overnight_server] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('overnight_server')

BASE = Path(__file__).parent.parent
FEATURE_CACHE = str(BASE / 'data' / 'processed' / 'mbo_features_cache')
SNAPSHOT_CACHE = str(BASE / 'data' / 'processed' / 'medium_snapshots_cache')
RESULTS_DIR = Path(__file__).parent / 'results'


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

    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        output_lines.append(line)
        logger.info(f"  {line}")

    proc.wait()
    elapsed = time.time() - t0
    logger.info(f"COMPLETED: {label} ({elapsed:.0f}s, exit={proc.returncode})")
    return proc.returncode, output_lines


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = RESULTS_DIR / f'overnight_server_{timestamp}.log'
    RESULTS_DIR.mkdir(exist_ok=True)

    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(logging.Formatter('%(asctime)s [overnight_server] %(message)s', '%H:%M:%S'))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("OVERNIGHT SERVER BENCHMARK SUITE (CPU)")
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Feature cache: {FEATURE_CACHE}")
    logger.info("=" * 60)

    python = sys.executable
    bench = str(Path(__file__).parent / 'arch_benchmark.py')

    # CPU-only architectures
    cpu_archs = ['lgbm', 'xgb', 'catboost']

    # ============================================================
    # PHASE 1: Quick 20-day test
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 1: Quick 20-day test of GBM variants")
    logger.info("=" * 60)

    rc, output = run_cmd([
        python, bench,
        '--arch', ','.join(cpu_archs),
        '--n-days', '20',
        '--horizon', 'ret_10s',
        '--device', 'cpu',
        '--feature-cache', FEATURE_CACHE,
        '--snapshot-cache', SNAPSHOT_CACHE,
    ], "20-day quick GBM comparison")

    # ============================================================
    # PHASE 2: Full 70-day run of all GBM variants
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 2: Full 70-day test of all GBM variants")
    logger.info("=" * 60)

    for arch in cpu_archs:
        rc, output = run_cmd([
            python, bench,
            '--arch', arch,
            '--n-days', '70',
            '--horizon', 'ret_10s',
            '--device', 'cpu',
            '--feature-cache', FEATURE_CACHE,
            '--snapshot-cache', SNAPSHOT_CACHE,
        ], f"70-day {arch.upper()} (CPU)")

        gc.collect()

    # ============================================================
    # PHASE 3: Multi-horizon for best GBM
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 3: Multi-horizon comparison (XGBoost)")
    logger.info("=" * 60)

    for horizon in ['ret_3s', 'ret_5s', 'ret_10s']:
        rc, output = run_cmd([
            python, bench,
            '--arch', 'xgb',
            '--n-days', '70',
            '--horizon', horizon,
            '--device', 'cpu',
            '--feature-cache', FEATURE_CACHE,
            '--snapshot-cache', SNAPSHOT_CACHE,
        ], f"70-day XGBoost @ {horizon}")

        gc.collect()

    logger.info("\n\n" + "=" * 60)
    logger.info(f"OVERNIGHT SERVER BENCHMARK COMPLETE")
    logger.info(f"Finished: {datetime.now()}")
    logger.info(f"Results in: {RESULTS_DIR}")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
