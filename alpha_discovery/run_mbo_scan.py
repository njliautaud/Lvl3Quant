"""
Run MBO Alpha Discovery Scan.

Uses pre-computed NPZ snapshot caches for 1000x faster loading.
Caches computed features to avoid recomputation.

Usage:
    python alpha_discovery/run_mbo_scan.py --n-files 15
    python alpha_discovery/run_mbo_scan.py --n-files 5 --no-cache
"""

import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'scan.log', mode='a'),
    ]
)
logger = logging.getLogger("run_mbo_scan")

# Feature cache path
FEATURE_CACHE = RESULTS_DIR / "feature_cache"
FEATURE_CACHE.mkdir(parents=True, exist_ok=True)


def get_cache_path(n_days) -> Path:
    """Get feature cache file path for given number of days."""
    label = f"{n_days}days" if n_days else "alldays"
    return FEATURE_CACHE / f"features_{label}.npz"


def _feature_names_hash() -> str:
    """Compute a hash of current feature names for cache validation."""
    import hashlib
    from alpha_discovery.mbo_features import get_feature_names
    names_str = ",".join(get_feature_names())
    return hashlib.md5(names_str.encode()).hexdigest()[:12]


def save_feature_cache(scanner: MBOAlphaScanner, n_days, stats: dict):
    """Save computed features to cache for fast reloading."""
    path = get_cache_path(n_days)
    np.savez_compressed(
        str(path),
        features=scanner.features,
        mid_prices=scanner.mid_prices,
        hour_of_day=scanner.hour_of_day,
        time_since_rth=scanner.time_since_rth,
        day_boundaries=np.array(scanner.day_boundaries),
    )
    # FIX: Save feature name hash for cache validation (not just column count).
    # This catches cases where features are reordered or replaced with same count.
    stats['feature_names_hash'] = _feature_names_hash()
    stats['n_features'] = scanner.features.shape[1]
    with open(str(path).replace('.npz', '_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Feature cache saved: {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_feature_cache(scanner: MBOAlphaScanner, n_days) -> dict:
    """Load features from cache. Returns None if cache is missing or stale."""
    from alpha_discovery.mbo_features import TOTAL_FEATURES

    path = get_cache_path(n_days)
    if not path.exists():
        return None

    logger.info(f"Loading cached features from {path}...")
    t0 = time.time()
    data = np.load(str(path), allow_pickle=True)

    cached_features = data['features']

    # Validate feature dimensions match current code
    if cached_features.shape[1] != TOTAL_FEATURES:
        logger.warning(
            f"Feature cache STALE: {cached_features.shape[1]} features "
            f"vs {TOTAL_FEATURES} expected. Recomputing..."
        )
        data.close()
        return None

    # FIX: Also validate feature name hash (catches reorder/rename with same count)
    stats_path = str(path).replace('.npz', '_stats.json')
    if Path(stats_path).exists():
        with open(stats_path) as f:
            stats = json.load(f)
        cached_hash = stats.get('feature_names_hash', '')
        current_hash = _feature_names_hash()
        if cached_hash and cached_hash != current_hash:
            logger.warning(
                f"Feature cache STALE: feature names changed "
                f"(hash {cached_hash} vs {current_hash}). Recomputing..."
            )
            data.close()
            return None
    else:
        stats = {
            'n_snapshots': len(data['mid_prices']),
            'n_days': len(data['day_boundaries']) - 1,
            'n_features': cached_features.shape[1],
        }

    scanner.features = cached_features
    scanner.mid_prices = data['mid_prices']
    scanner.hour_of_day = data['hour_of_day']
    scanner.time_since_rth = data['time_since_rth']
    scanner.day_boundaries = list(data['day_boundaries'])
    scanner.feature_names = __import__('alpha_discovery.mbo_features',
                                        fromlist=['get_feature_names']).get_feature_names()

    logger.info(f"Cache loaded in {time.time() - t0:.1f}s: {scanner.features.shape}")
    return stats


def main():
    parser = argparse.ArgumentParser(description='MBO Alpha Discovery Scan')
    parser.add_argument('--n-days', type=int, default=None,
                        help='Number of trading days to use (default: all)')
    parser.add_argument('--sample-ms', type=int, default=100,
                        help='Snapshot interval in ms (default: 100 for medium cache)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force recompute features (skip feature cache)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("MBO ALPHA DISCOVERY SCAN (cache-accelerated)")
    logger.info(f"  Days: {args.n_days or 'all'}")
    logger.info(f"  Sample interval: {args.sample_ms}ms")
    logger.info(f"  Feature cache: {'disabled' if args.no_cache else 'enabled'}")
    logger.info("=" * 70)

    # NOTE on normalization: LightGBM is tree-based and invariant to
    # monotonic transforms (log, sqrt, normalization). Trees split on
    # rank order, not absolute values. Individual features already use
    # appropriate normalization (z-scores, ratios, log returns) where
    # it helps numerical stability. No global normalization needed.

    t_start = time.time()

    # Create scanner
    scanner = MBOAlphaScanner(
        sample_interval_ms=args.sample_ms,
    )

    # Phase 1: Load data (try feature cache first)
    logger.info("\n--- PHASE 1: Load Data ---")
    stats = None
    if not args.no_cache:
        stats = load_feature_cache(scanner, args.n_days)

    if stats is None:
        logger.info("Computing features from snapshot cache...")
        stats = scanner.load_from_cache(n_days=args.n_days)
        logger.info(f"Data loaded: {stats}")

        # Save feature cache for next run
        save_feature_cache(scanner, args.n_days, stats)
    else:
        logger.info(f"Using cached features: {stats}")

    load_elapsed = time.time() - t_start
    logger.info(f"Phase 1 complete: {load_elapsed:.1f}s")

    # Phase 2: Run full scan
    logger.info("\n--- PHASE 2: Walk-Forward Alpha Scan ---")
    logger.info(f"Testing {len(scanner.horizons)} horizons × {len(scanner.target_types)} targets "
                f"= {len(scanner.horizons) * len(scanner.target_types)} scans")

    results = scanner.run_full_scan()

    # Phase 3: Report
    logger.info("\n--- PHASE 3: Results ---")
    scoreboard = MBOAlphaScanner.format_scoreboard(results)
    logger.info("\n" + scoreboard)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"mbo_scan_{timestamp}.json"

    # Convert results for JSON serialization
    serializable = []
    for r in results:
        sr = dict(r)
        sr['top_features'] = [(n, float(v)) for n, v in r.get('top_features', [])]
        sr['leakage_flags'] = [(n, float(v)) for n, v in r.get('leakage_flags', [])]
        serializable.append(sr)

    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'n_days': args.n_days,
                'sample_ms': args.sample_ms,
            },
            'stats': stats,
            'results': serializable,
            'scoreboard': scoreboard,
            'elapsed_sec': time.time() - t_start,
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Total elapsed: {time.time() - t_start:.0f}s")

    # Print scoreboard for visibility
    print("\n" + "=" * 70)
    print(scoreboard)
    print("=" * 70)


if __name__ == '__main__':
    main()
