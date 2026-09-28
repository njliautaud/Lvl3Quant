"""
V2 Alpha Discovery Pipeline — Orchestrates multi-phase alpha search.

Phases:
  1A: Multi-alpha scan with return targets (validates V1 at 100-day scale)
  1B: Multi-alpha scan with MFE targets (captures path-dependent alpha)
  3:  Multi-horizon scanning (3s, 5s, 10s, 30s, 1m — both return and MFE)

Each phase uses pre-computed features from Rust cache (saves ~3h per run).
Results saved to results/ with timestamps.

Usage:
    python alpha_discovery/run_v2_pipeline.py --phase 1b
    python alpha_discovery/run_v2_pipeline.py --phase multi-horizon
    python alpha_discovery/run_v2_pipeline.py --phase all
"""

import os
import sys
import json
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

LVL3_ROOT = Path(__file__).parent.parent
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature cache from Rust expander (290 pre-computed features, ~4 min vs ~3h)
FEATURE_CACHE = str(LVL3_ROOT / "data" / "processed" / "mbo_features_cache")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [v2_pipeline] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(str(RESULTS_DIR / f"v2_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("v2_pipeline")

PYTHON = sys.executable
SCRIPT = str(LVL3_ROOT / "alpha_discovery" / "run_multi_alpha.py")


def run_scan(horizon: str, target_type: str, label: str, use_feature_cache: bool = True):
    """Run a single multi-alpha scan with given parameters."""
    cmd = [
        PYTHON, SCRIPT,
        '--horizon', horizon,
        '--target-type', target_type,
        '--min-train-days', '5',
    ]
    if use_feature_cache and os.path.isdir(FEATURE_CACHE):
        cmd.extend(['--feature-cache', FEATURE_CACHE])

    logger.info(f"{'='*70}")
    logger.info(f"STARTING: {label}")
    logger.info(f"  Horizon: {horizon}, Target: {target_type}")
    logger.info(f"  Feature cache: {'YES' if use_feature_cache else 'NO'}")
    logger.info(f"  Command: {' '.join(cmd)}")
    logger.info(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(LVL3_ROOT),
    )

    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info(f"COMPLETED: {label} [{elapsed:.0f}s = {elapsed/60:.1f}m]")
    else:
        logger.error(f"FAILED: {label} [{elapsed:.0f}s]")
        logger.error(f"  stderr: {result.stderr[-500:] if result.stderr else 'none'}")

    return result.returncode == 0


def phase_1b():
    """Phase 1B: MFE target scan on 5s horizon."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 1B: MFE Target Multi-Alpha Scan")
    logger.info("="*80 + "\n")

    return run_scan('ret_5s', 'mfe_net', 'Phase 1B: ret_5s + mfe_net')


def phase_multi_horizon():
    """Phase 3: Multi-horizon scanning with both return and MFE targets."""
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: Multi-Horizon Scanning")
    logger.info("="*80 + "\n")

    horizons = ['ret_3s', 'ret_5s', 'ret_10s', 'ret_30s', 'ret_1m']
    target_types = ['return', 'mfe_net']

    results = {}
    for hz in horizons:
        for tt in target_types:
            label = f"Phase 3: {hz} + {tt}"
            # Skip ret_5s + return (already done in Phase 1A)
            if hz == 'ret_5s' and tt == 'return':
                logger.info(f"SKIPPING {label} (already done in Phase 1A)")
                continue
            # Skip ret_5s + mfe_net (already done in Phase 1B)
            if hz == 'ret_5s' and tt == 'mfe_net':
                logger.info(f"SKIPPING {label} (already done in Phase 1B)")
                continue

            success = run_scan(hz, tt, label)
            results[f"{hz}_{tt}"] = success

            if not success:
                logger.warning(f"  {label} failed, continuing with next...")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='V2 Alpha Discovery Pipeline')
    parser.add_argument('--phase', type=str, required=True,
                        choices=['1b', 'multi-horizon', 'all'],
                        help='Which phase to run')
    args = parser.parse_args()

    logger.info("V2 Alpha Discovery Pipeline Starting")
    logger.info(f"  Phase: {args.phase}")
    logger.info(f"  Feature cache: {FEATURE_CACHE}")
    logger.info(f"  Results dir: {RESULTS_DIR}")

    t_start = time.time()

    if args.phase == '1b':
        phase_1b()
    elif args.phase == 'multi-horizon':
        phase_multi_horizon()
    elif args.phase == 'all':
        phase_1b()
        phase_multi_horizon()

    total = time.time() - t_start
    logger.info(f"\nV2 Pipeline Complete — Total: {total:.0f}s ({total/3600:.1f}h)")


if __name__ == '__main__':
    main()
