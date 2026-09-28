#!/usr/bin/env python3
"""
Rebuild the per-day feature cache using the latest mbo_features.py (340 features).

Reads raw snapshot NPZ files from rust_full_test/ and computes all 340 features
per day, saving to mbo_features_cache/.

Usage:
    python alpha_discovery/rebuild_feature_cache.py [--workers 4] [--force]
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_features import compute_mbo_features, TOTAL_FEATURES, get_feature_names

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [rebuild_cache] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

SNAPSHOT_DIR = ROOT / "data" / "processed" / "rust_full_test"
CACHE_DIR = ROOT / "data" / "processed" / "mbo_features_cache"
TICK_SIZE = 0.25
DEPTH_LEVELS = 10


def process_one_day(snap_path: Path, out_path: Path) -> dict:
    """Load snapshot, compute 340 features, save to cache."""
    t0 = time.time()

    data = np.load(str(snap_path))
    mid_prices = data['mid_prices'].astype(np.float64)
    global_features = data['global_features'].astype(np.float32)
    node_features = data['node_features'].astype(np.float32)
    N = len(mid_prices)
    data.close()

    # Single day: boundaries = [0, N]
    features = compute_mbo_features(
        mid_prices=mid_prices,
        global_features_raw=global_features,
        node_features_raw=node_features,
        tick_size=TICK_SIZE,
        depth_levels=DEPTH_LEVELS,
        day_boundaries=[0, N],
    )

    np.savez_compressed(str(out_path), mbo_features=features)
    elapsed = time.time() - t0

    return {
        'date': snap_path.name[:10],
        'rows': N,
        'cols': features.shape[1],
        'elapsed': elapsed,
        'size_mb': out_path.stat().st_size / 1e6,
    }


def main():
    parser = argparse.ArgumentParser(description='Rebuild feature cache with 340 features')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default=1, sequential)')
    parser.add_argument('--force', action='store_true',
                        help='Rebuild even if cache file exists with correct feature count')
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    snap_files = sorted(SNAPSHOT_DIR.glob("*_snapshots.npz"))
    if not snap_files:
        log.error(f"No snapshot files found in {SNAPSHOT_DIR}")
        sys.exit(1)

    log.info(f"Found {len(snap_files)} snapshot files in {SNAPSHOT_DIR}")
    log.info(f"Target features: {TOTAL_FEATURES} ({len(get_feature_names())} named)")
    log.info(f"Output: {CACHE_DIR}")
    log.info(f"Workers: {args.workers}")

    # Filter out already-processed files unless --force
    to_process = []
    skipped = 0
    for sf in snap_files:
        date_str = sf.name[:10]
        out_path = CACHE_DIR / f"{date_str}_mbo_features.npz"

        if out_path.exists() and not args.force:
            # Check if existing cache has correct feature count
            try:
                existing = np.load(str(out_path))
                n_cols = existing['mbo_features'].shape[1]
                existing.close()
                if n_cols == TOTAL_FEATURES:
                    skipped += 1
                    continue
                else:
                    log.info(f"  {date_str}: stale ({n_cols} cols, need {TOTAL_FEATURES})")
            except Exception:
                pass  # Corrupted file, rebuild

        to_process.append((sf, out_path))

    if skipped:
        log.info(f"Skipping {skipped} files already at {TOTAL_FEATURES} features")

    if not to_process:
        log.info("All files up to date. Use --force to rebuild all.")
        return

    log.info(f"Processing {len(to_process)} files...")

    t0_total = time.time()
    completed = 0

    if args.workers <= 1:
        # Sequential processing (safer for memory)
        for snap_path, out_path in to_process:
            result = process_one_day(snap_path, out_path)
            completed += 1
            log.info(
                f"  [{completed}/{len(to_process)}] {result['date']}: "
                f"{result['rows']:,} bars, {result['cols']} features, "
                f"{result['size_mb']:.1f} MB [{result['elapsed']:.1f}s]"
            )
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_one_day, sp, op): sp.name[:10]
                for sp, op in to_process
            }
            for future in as_completed(futures):
                date_str = futures[future]
                try:
                    result = future.result()
                    completed += 1
                    log.info(
                        f"  [{completed}/{len(to_process)}] {result['date']}: "
                        f"{result['rows']:,} bars, {result['cols']} features, "
                        f"{result['size_mb']:.1f} MB [{result['elapsed']:.1f}s]"
                    )
                except Exception as e:
                    log.error(f"  FAILED {date_str}: {e}")

    elapsed_total = time.time() - t0_total
    log.info(f"\nDone: {completed}/{len(to_process)} files in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    log.info(f"Average: {elapsed_total/max(completed,1):.1f}s per day")

    # Verify
    cache_files = sorted(CACHE_DIR.glob("*_mbo_features.npz"))
    if cache_files:
        sample = np.load(str(cache_files[0]))
        log.info(f"Verification: {cache_files[0].name} → {sample['mbo_features'].shape}")
        sample.close()


if __name__ == '__main__':
    main()
