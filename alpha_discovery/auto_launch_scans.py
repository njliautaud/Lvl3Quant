"""
Auto-Launch Scan Pipeline — Monitors cache rebuild and launches scans when complete.

Watches the cache directory for completion (100 .npz files with v3 format),
then automatically launches:
1. Multi-channel alpha scan (290 features, 5 channels + residual + ensemble)
2. Multi-tick classifier (3+ tick event detection, multiple thresholds)
3. Baseline single-model comparison

Usage:
    python alpha_discovery/auto_launch_scans.py [--no-monitor] [--skip-multitick]

    --no-monitor   : Skip cache monitoring, run scans immediately
    --skip-multitick: Skip the multi-tick classifier
"""

import gc
import os
import sys
import json
import time
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "data" / "processed" / "medium_snapshots_cache"
RESULTS_DIR = ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = RESULTS_DIR / f"auto_launch_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode='w'),
    ]
)
logger = logging.getLogger("auto_launch")


def check_cache_ready(min_files: int = 90) -> bool:
    """Check if cache rebuild is complete."""
    if not CACHE_DIR.exists():
        return False
    npz_files = list(CACHE_DIR.glob("????-??-??_snapshots.npz"))
    if len(npz_files) < min_files:
        return False

    # Check newest file has v3 format (96 global features)
    import numpy as np
    newest = max(npz_files, key=lambda f: f.stat().st_mtime)
    try:
        data = np.load(str(newest))
        gf = data['global_features']
        if gf.shape[1] < 96:
            logger.info(f"Cache format too old: {gf.shape[1]} globals (need 96)")
            return False
        return True
    except Exception as e:
        logger.warning(f"Error checking cache: {e}")
        return False


def wait_for_cache(check_interval: int = 60, timeout: int = 36000):
    """Wait for cache rebuild to complete. Check every minute, timeout after 10 hours."""
    start = time.time()
    logger.info(f"Monitoring {CACHE_DIR} for cache completion...")
    logger.info(f"Check interval: {check_interval}s, timeout: {timeout}s")

    while time.time() - start < timeout:
        if check_cache_ready():
            npz_count = len(list(CACHE_DIR.glob("????-??-??_snapshots.npz")))
            logger.info(f"Cache ready! {npz_count} files with v3 format.")
            return True

        npz_count = len(list(CACHE_DIR.glob("????-??-??_snapshots.npz")))
        elapsed = time.time() - start
        logger.info(f"  [{elapsed/60:.0f}m] Cache not ready. {npz_count} files found. "
                     f"Waiting {check_interval}s...")
        time.sleep(check_interval)

    logger.error(f"Timeout after {timeout}s. Cache not ready.")
    return False


def run_multi_alpha(horizon: str = 'ret_5s'):
    """Run the multi-channel alpha scan."""
    logger.info("=" * 70)
    logger.info("PHASE 1: Multi-Channel Alpha Scan")
    logger.info(f"  Horizon: {horizon}, Features: 290 + 14 event = 304")
    logger.info("=" * 70)

    cmd = [
        sys.executable, str(ROOT / "alpha_discovery" / "run_multi_alpha.py"),
        '--horizon', horizon,
        '--no-execution',  # Skip execution backtest for speed
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        logger.info(f"Multi-alpha completed in {(time.time()-t0)/60:.1f}m")
        if result.returncode != 0:
            logger.error(f"Multi-alpha FAILED:\n{result.stderr[-2000:]}")
        else:
            # Find and log the comparison table
            for line in result.stdout.split('\n'):
                if 'IC' in line or 'Channel' in line or '---' in line or 'MULTI-ALPHA' in line:
                    logger.info(f"  {line.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("Multi-alpha TIMEOUT after 2 hours")
        return False
    except Exception as e:
        logger.error(f"Multi-alpha exception: {e}")
        return False


def run_multitick_classifier(horizon: str = '5s'):
    """Run the multi-tick classifier with threshold sweep."""
    logger.info("=" * 70)
    logger.info("PHASE 2: Multi-Tick Classifier")
    logger.info(f"  Horizon: {horizon}, Thresholds: 2,3,4,5 ticks")
    logger.info("=" * 70)

    cmd = [
        sys.executable, str(ROOT / "alpha_discovery" / "multitick_classifier.py"),
        '--horizon', horizon,
        '--thresholds', '2,3,4,5',
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        logger.info(f"Multitick classifier completed in {(time.time()-t0)/60:.1f}m")
        if result.returncode != 0:
            logger.error(f"Multitick FAILED:\n{result.stderr[-2000:]}")
        else:
            for line in result.stdout.split('\n'):
                if any(kw in line for kw in ['RESULTS', 'ticks', 'Win rate', 'Profit', 'Mean net', 'Combined']):
                    logger.info(f"  {line.strip()}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("Multitick TIMEOUT after 2 hours")
        return False
    except Exception as e:
        logger.error(f"Multitick exception: {e}")
        return False


def run_baseline_comparison(horizon: str = 'ret_5s'):
    """Run single-model baseline for comparison."""
    logger.info("=" * 70)
    logger.info("PHASE 3: Single Model Baseline")
    logger.info(f"  Horizon: {horizon}, All 290 features, single LightGBM")
    logger.info("=" * 70)

    # Import and run directly to avoid subprocess overhead
    try:
        import numpy as np
        from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner

        t0 = time.time()
        scanner = MBOAlphaScanner()
        load_info = scanner.load_from_cache()
        logger.info(f"Loaded {load_info['n_days']} days, {load_info['n_snapshots']:,} bars")

        # Compute target
        horizon_bars = {'ret_3s': 30, 'ret_5s': 50, 'ret_10s': 100, 'ret_30s': 300}.get(horizon, 50)
        N = len(scanner.mid_prices)
        future_mid = np.empty(N, dtype=np.float32)
        future_mid[:N - horizon_bars] = scanner.mid_prices[horizon_bars:]
        future_mid[N - horizon_bars:] = np.nan
        target = (future_mid - scanner.mid_prices) / np.maximum(scanner.mid_prices, 1.0)

        result = scanner.walk_forward_evaluate(
            target=target,
            target_name='return',
            horizon_name=horizon,
        )

        elapsed = (time.time() - t0) / 60
        logger.info(f"Baseline completed in {elapsed:.1f}m")
        logger.info(f"  IC={result.get('ic', 0):.4f}, "
                     f"ICIR={result.get('icir', 0):.2f}, "
                     f"t={result.get('tstat', 0):.2f}, "
                     f"HR={result.get('hit_rate', 0):.1%}")

        # Save
        out_file = RESULTS_DIR / f"baseline_single_{horizon}_{timestamp}.json"
        with open(str(out_file), 'w') as f:
            json.dump({k: v for k, v in result.items()
                      if isinstance(v, (int, float, str, bool, list))},
                     f, indent=2, default=str)
        logger.info(f"Saved to {out_file}")

        del scanner
        gc.collect()
        return True

    except Exception as e:
        logger.error(f"Baseline failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Auto-Launch Scan Pipeline")
    parser.add_argument('--no-monitor', action='store_true',
                        help='Skip cache monitoring, run scans immediately')
    parser.add_argument('--skip-multitick', action='store_true',
                        help='Skip multi-tick classifier')
    parser.add_argument('--horizon', default='ret_5s',
                        help='Target horizon (default: ret_5s)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("AUTO-LAUNCH SCAN PIPELINE")
    logger.info(f"  Timestamp: {timestamp}")
    logger.info(f"  Horizon: {args.horizon}")
    logger.info(f"  Cache dir: {CACHE_DIR}")
    logger.info(f"  Log file: {LOG_FILE}")
    logger.info("=" * 70)

    # Step 1: Wait for cache
    if not args.no_monitor:
        if not wait_for_cache():
            logger.error("ABORTED: Cache not ready")
            return
    else:
        if not check_cache_ready():
            logger.error("ABORTED: Cache not ready and --no-monitor specified")
            return

    total_start = time.time()
    results = {}

    # Step 2: Run baseline first (quick reference)
    results['baseline'] = run_baseline_comparison(args.horizon)

    # Step 3: Multi-channel alpha scan
    results['multi_alpha'] = run_multi_alpha(args.horizon)

    # Step 4: Multi-tick classifier
    if not args.skip_multitick:
        horizon_short = args.horizon.replace('ret_', '')
        results['multitick'] = run_multitick_classifier(horizon_short)

    total_elapsed = (time.time() - total_start) / 60
    logger.info("\n" + "=" * 70)
    logger.info(f"ALL SCANS COMPLETE — Total time: {total_elapsed:.1f} minutes")
    logger.info("=" * 70)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {name}: {status}")

    # Save summary
    summary_file = RESULTS_DIR / f"auto_launch_summary_{timestamp}.json"
    with open(str(summary_file), 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'horizon': args.horizon,
            'total_time_minutes': total_elapsed,
            'results': results,
        }, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
