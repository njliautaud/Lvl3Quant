"""
Run CLEAN MBO Alpha Scan — with problematic features EXCLUDED.

Removes features that could cause spurious signal:
  - mid, best_bid, best_ask, microprice: absolute price = day identifier
  - hour_norm, minute_norm, time_since_rth, time_to_close: trivial intraday seasonality

This tests whether the MICROSTRUCTURE features alone carry any signal.

Usage:
    python alpha_discovery/run_clean_scan.py
"""

import sys
import json
import time
import logging
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
        logging.FileHandler(RESULTS_DIR / 'clean_scan.log', mode='a'),
    ]
)
logger = logging.getLogger("run_clean_scan")

# Feature cache path
FEATURE_CACHE = RESULTS_DIR / "feature_cache"

# Features to EXCLUDE (leaky/trivial)
EXCLUDE_FEATURES = [
    # Absolute price level — acts as day identifier with only 16 days
    'mid', 'best_bid', 'best_ask', 'microprice',
    # Time-of-day features — trivial intraday seasonality everyone knows
    'hour_norm', 'minute_norm', 'time_since_rth', 'time_to_close',
]


def _feature_names_hash() -> str:
    """Compute a hash of current feature names for cache validation."""
    import hashlib
    from alpha_discovery.mbo_features import get_feature_names
    names_str = ",".join(get_feature_names())
    return hashlib.md5(names_str.encode()).hexdigest()[:12]


def load_feature_cache(scanner: MBOAlphaScanner) -> dict:
    """Load features from cache. Returns None if cache is missing or stale."""
    from alpha_discovery.mbo_features import TOTAL_FEATURES

    path = FEATURE_CACHE / "features_alldays.npz"
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
            f"vs {TOTAL_FEATURES} expected. Will recompute from snapshot cache."
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
                f"(hash {cached_hash} vs {current_hash}). Will recompute."
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
    logger.info("=" * 70)
    logger.info("CLEAN MBO ALPHA SCAN (microstructure features only)")
    logger.info(f"  Excluding: {EXCLUDE_FEATURES}")
    logger.info("=" * 70)

    t_start = time.time()

    # Create scanner
    scanner = MBOAlphaScanner(sample_interval_ms=100)

    # Load from feature cache
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.info("No feature cache found, computing from scratch...")
        stats = scanner.load_from_cache()
    else:
        logger.info(f"Using cached features: {stats}")

    logger.info(f"Total features: {len(scanner.feature_names)}")
    logger.info(f"Excluding: {len(EXCLUDE_FEATURES)} features")
    logger.info(f"Remaining: {len(scanner.feature_names) - len(EXCLUDE_FEATURES)} features")

    # Run scan with exclusions
    logger.info("\n--- Walk-Forward Alpha Scan (CLEAN) ---")
    results = scanner.run_full_scan(exclude_features=EXCLUDE_FEATURES)

    # Report
    logger.info("\n--- Results (CLEAN) ---")
    scoreboard = MBOAlphaScanner.format_scoreboard(results)
    logger.info("\n" + scoreboard)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"clean_scan_{timestamp}.json"

    serializable = []
    for r in results:
        sr = dict(r)
        sr['top_features'] = [(n, float(v)) for n, v in r.get('top_features', [])]
        sr['leakage_flags'] = [(n, float(v)) for n, v in r.get('leakage_flags', [])]
        serializable.append(sr)

    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'excluded_features': EXCLUDE_FEATURES,
            'stats': stats,
            'results': serializable,
            'scoreboard': scoreboard,
            'elapsed_sec': time.time() - t_start,
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Total elapsed: {time.time() - t_start:.0f}s")

    print("\n" + "=" * 70)
    print(scoreboard)
    print("=" * 70)


if __name__ == '__main__':
    main()
