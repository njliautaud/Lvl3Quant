#!/usr/bin/env python3
"""
Overnight Runner — PC (GPU)
Runs all GPU-accelerated architecture benchmarks sequentially.

Order of operations:
1. Quick 20-day test of ALL architectures (identifies promising ones)
2. Full 70-day run of top performers

Estimated time: ~6-10 hours total
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
    format='%(asctime)s [overnight_pc] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('overnight_pc')

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
    log_file = RESULTS_DIR / f'overnight_pc_{timestamp}.log'
    RESULTS_DIR.mkdir(exist_ok=True)

    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(logging.Formatter('%(asctime)s [overnight_pc] %(message)s', '%H:%M:%S'))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("OVERNIGHT PC BENCHMARK SUITE")
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Feature cache: {FEATURE_CACHE}")
    logger.info("=" * 60)

    python = sys.executable
    bench = str(Path(__file__).parent / 'arch_benchmark.py')

    # ============================================================
    # PHASE 1: Quick 20-day test of all architectures
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 1: Quick 20-day test of promising architectures")
    logger.info("=" * 60)

    # TCN excluded (IC=0.03 on quick test)
    rc, output = run_cmd([
        python, bench,
        '--arch', 'lgbm,catboost,mlp,tabnet,xgb',
        '--n-days', '20',
        '--horizon', 'ret_10s',
        '--device', 'gpu',
        '--feature-cache', FEATURE_CACHE,
        '--snapshot-cache', SNAPSHOT_CACHE,
    ], "20-day quick benchmark (lgbm, catboost, mlp, tabnet, xgb)")

    # Parse results to find top performers
    # Look for the JSON results file
    json_files = sorted(RESULTS_DIR.glob('arch_benchmark_*.json'), reverse=True)
    top_archs = []
    if json_files:
        try:
            with open(json_files[0]) as f:
                quick_results = json.load(f)
            results = quick_results.get('results', {})
            # Sort by IC, pick those with IC > 0.05
            sorted_archs = sorted(
                [(k, v.get('ic', -999)) for k, v in results.items() if 'error' not in v],
                key=lambda x: x[1],
                reverse=True,
            )
            top_archs = [name for name, ic in sorted_archs if ic > 0.05]
            logger.info(f"\nTop architectures (IC > 0.05): {top_archs}")
            for name, ic in sorted_archs:
                logger.info(f"  {name}: IC={ic:.4f}")
        except Exception as e:
            logger.warning(f"Could not parse quick results: {e}")
            top_archs = ['lgbm', 'xgb', 'catboost', 'mlp']

    if not top_archs:
        top_archs = ['lgbm', 'xgb', 'catboost', 'mlp']
        logger.info(f"Using default top architectures: {top_archs}")

    # ============================================================
    # PHASE 2: Full 70-day run of top performers
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info(f"PHASE 2: Full 70-day test of top performers: {top_archs}")
    logger.info("=" * 60)

    for arch in top_archs:
        rc, output = run_cmd([
            python, bench,
            '--arch', arch,
            '--n-days', '70',
            '--horizon', 'ret_10s',
            '--device', 'gpu',
            '--feature-cache', FEATURE_CACHE,
            '--snapshot-cache', SNAPSHOT_CACHE,
        ], f"70-day {arch.upper()} benchmark")

        gc.collect()

    # ============================================================
    # PHASE 3: Multi-horizon comparison for best architecture
    # ============================================================
    logger.info("\n\n" + "=" * 60)
    logger.info("PHASE 3: Multi-horizon comparison for best architecture")
    logger.info("=" * 60)

    best_arch = top_archs[0] if top_archs else 'lgbm'
    for horizon in ['ret_3s', 'ret_5s', 'ret_10s']:
        rc, output = run_cmd([
            python, bench,
            '--arch', best_arch,
            '--n-days', '70',
            '--horizon', horizon,
            '--device', 'gpu',
            '--feature-cache', FEATURE_CACHE,
            '--snapshot-cache', SNAPSHOT_CACHE,
        ], f"70-day {best_arch.upper()} @ {horizon}")

        gc.collect()

    logger.info("\n\n" + "=" * 60)
    logger.info(f"OVERNIGHT PC BENCHMARK COMPLETE")
    logger.info(f"Finished: {datetime.now()}")
    logger.info(f"Results in: {RESULTS_DIR}")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
