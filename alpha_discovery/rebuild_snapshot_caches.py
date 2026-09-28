"""
Rebuild NPZ snapshot caches — ONE FILE PER TRADING DAY.

Output format: YYYY-MM-DD_snapshots.npz (one file per unique calendar day)
Plus cache_manifest.json with full metadata for auditability.

Previous versions saved one cache per source .dbn file, which caused data
overlap because Databento daily files contain multi-day sliding windows
(e.g., Tuesday's file includes Monday+Tuesday RTH sessions). This led to
the SAME trading day appearing 2-4x in walk-forward, contaminating
train/test splits.

This version:
  - Extracts ACTUAL calendar dates from nanosecond UTC timestamps
  - Saves exactly 1 cache file per real trading day
  - Skips weekends, holidays (flat days), and duplicates automatically
  - Creates cache_manifest.json mapping every date to its source file
  - Resume-capable: if interrupted, --resume skips already-processed files
  - Format compatible with Databento streaming API conventions

Usage:
    python alpha_discovery/rebuild_snapshot_caches.py
    python alpha_discovery/rebuild_snapshot_caches.py --dry-run
    python alpha_discovery/rebuild_snapshot_caches.py --resume
"""

import gc
import sys
import time
import json
import logging
import argparse
import traceback
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.ingest import iter_files, load_mbo_events, get_es_instrument_id, get_es_contract_name
from src.data.lob import OrderBook
from src.features.engineering import compute_features

# ============================================================================
# CONFIG
# ============================================================================
SAMPLE_INTERVAL_MS = 100       # 100ms snapshots
DEPTH_LEVELS = 10
MBO_DIR = ROOT / "mbo"
CACHE_DIR = ROOT / "data" / "processed" / "medium_snapshots_cache"

# RTH (Regular Trading Hours) 9:30 AM - 4:00 PM ET
RTH_START_MINUTES = 9 * 60 + 30
RTH_END_MINUTES = 16 * 60

# Minimum thresholds for a valid trading day
MIN_SNAPSHOTS = 100            # At least 100 RTH snapshots (10 seconds of data)
MIN_PRICE_RANGE_PTS = 1.0      # At least 1 point of price movement

# DST transition for 2025 (US)
# DST starts: Mar 9, 2025 -> EDT (UTC-4)
# DST ends:   Nov 2, 2025 -> EST (UTC-5)
DST_END_2025_NS = int(datetime(2025, 11, 2, 6, 0, 0).timestamp() * 1e9)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("rebuild_caches")


# ============================================================================
# TIMESTAMP UTILITIES
# ============================================================================

def _to_ns(value) -> int:
    """Convert various timestamp types to nanoseconds int."""
    if hasattr(value, "value"):
        return int(value.value)
    return int(value)


def _get_et_offset(timestamp_ns: int) -> int:
    """Return UTC offset for Eastern Time: -4 (EDT) or -5 (EST)."""
    if timestamp_ns < DST_END_2025_NS:
        return -4  # EDT (summer)
    return -5  # EST (winter)


def _is_within_rth(timestamp_ns: int) -> bool:
    """Check if a nanosecond UTC timestamp falls within RTH (9:30-16:00 ET)."""
    if timestamp_ns == 0:
        return False
    try:
        dt_utc = datetime.utcfromtimestamp(timestamp_ns / 1e9)
        et_offset = _get_et_offset(timestamp_ns)
        dt_et = dt_utc + timedelta(hours=et_offset)
        time_minutes = dt_et.hour * 60 + dt_et.minute
        return RTH_START_MINUTES <= time_minutes < RTH_END_MINUTES
    except Exception:
        return False


def _ts_to_et_date(timestamp_ns: int) -> str:
    """Convert nanosecond UTC timestamp to Eastern Time calendar date YYYY-MM-DD.

    Since we only call this for RTH snapshots (9:30 AM - 4:00 PM ET),
    the date is always unambiguous — no overnight boundary issues.
    """
    dt_utc = datetime.utcfromtimestamp(timestamp_ns / 1e9)
    et_offset = _get_et_offset(timestamp_ns)
    dt_et = dt_utc + timedelta(hours=et_offset)
    return dt_et.strftime('%Y-%m-%d')


def _is_weekday(date_str: str) -> bool:
    """Check if a YYYY-MM-DD date string is a weekday (Mon-Fri)."""
    return datetime.strptime(date_str, '%Y-%m-%d').weekday() < 5


# ============================================================================
# FILE PROCESSING
# ============================================================================

def process_single_file(file_idx: int, file_path: Path) -> dict:
    """Process a single .dbn file and return per-date snapshot data.

    Replays all events through the order book, takes RTH snapshots at
    100ms intervals, and groups them by calendar date (ET timezone).

    Returns:
        dict mapping date_str (YYYY-MM-DD) to:
            {'node': np.array, 'global': np.array, 'mid': np.array,
             'source_file': str, 'price_range': float}
        Empty dict on error or no valid data.
    """
    # Determine correct instrument_id (handles ESU5 -> ESZ5 rollover)
    instrument_id = get_es_instrument_id(file_path.name)
    contract_name = get_es_contract_name(file_path.name)
    if instrument_id is None:
        logger.warning(f"  Cannot determine instrument_id for {file_path.name}, skipping")
        return {}
    logger.info(f"  Using {contract_name} (instrument_id={instrument_id})")

    book = OrderBook(DEPTH_LEVELS)

    # Group snapshots by calendar date
    date_snapshots = {}  # date_str -> {nodes: [], globals: [], mids: []}

    interval_ns = SAMPLE_INTERVAL_MS * 1_000_000

    t0 = time.time()
    n_events = 0
    rth_count = 0
    non_rth_skipped = 0

    try:
        df = load_mbo_events(
            file_path,
            max_events=None,
            filter_instrument_id=instrument_id,
        )

        if "ts_event" not in df.columns:
            logger.warning(f"  No ts_event column in {file_path.name}, skipping")
            return {}

        df = df.sort_values("ts_event")
        n_events = len(df)
        next_snapshot_ts = None

        # Pre-extract columns as lists for 10-50x speedup over iterrows
        cols = df.columns.tolist()
        ts_events_list = df["ts_event"].tolist()
        actions_list = df["action"].tolist() if "action" in cols else [None] * n_events
        sides_list = df["side"].tolist() if "side" in cols else [None] * n_events
        prices_list = df["price"].tolist() if "price" in cols else [0.0] * n_events
        sizes_list = df["size"].tolist() if "size" in cols else [0.0] * n_events
        order_ids_list = df["order_id"].tolist() if "order_id" in cols else [None] * n_events
        flags_list = df["flags"].tolist() if "flags" in cols else [0] * n_events
        seq_list = df["sequence"].tolist() if "sequence" in cols else [None] * n_events

        del df
        gc.collect()

        t_load = time.time() - t0
        logger.info(f"    Loaded {n_events:,} events in {t_load:.1f}s, starting book replay...")

        for i in range(n_events):
            ts_event = _to_ns(ts_events_list[i])
            if next_snapshot_ts is None:
                next_snapshot_ts = ts_event

            event = {
                "action": actions_list[i],
                "side": sides_list[i],
                "price": prices_list[i],
                "size": sizes_list[i],
                "order_id": order_ids_list[i],
                "ts_event": ts_events_list[i],
                "flags": flags_list[i],
                "sequence": seq_list[i],
            }
            book.update(event)

            while ts_event >= next_snapshot_ts:
                if not _is_within_rth(next_snapshot_ts):
                    non_rth_skipped += 1
                    next_snapshot_ts += interval_ns
                    continue

                snapshot = book.snapshot()
                snapshot.timestamp_ns = next_snapshot_ts

                if snapshot.mid == snapshot.mid:  # Not NaN
                    node_feat, global_feat = compute_features(
                        snapshot,
                        DEPTH_LEVELS,
                        include_order_flow=True,
                        include_microstructure=True,
                        include_temporal=True,
                    )

                    # Group by calendar date (ET timezone)
                    date_str = _ts_to_et_date(next_snapshot_ts)
                    if date_str not in date_snapshots:
                        date_snapshots[date_str] = {
                            'nodes': [], 'globals': [], 'mids': []
                        }
                    date_snapshots[date_str]['nodes'].append(node_feat)
                    date_snapshots[date_str]['globals'].append(global_feat)
                    date_snapshots[date_str]['mids'].append(snapshot.mid)
                    rth_count += 1

                next_snapshot_ts += interval_ns

            # Progress every 1M events
            if i > 0 and i % 1_000_000 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (n_events - i) / rate
                logger.info(
                    f"    ... {i:,}/{n_events:,} events ({i/n_events*100:.0f}%), "
                    f"{rate:.0f} evt/s, ETA {eta:.0f}s, "
                    f"{len(date_snapshots)} dates found so far"
                )

        del ts_events_list, actions_list, sides_list, prices_list
        del sizes_list, order_ids_list, flags_list, seq_list
        gc.collect()

    except Exception as e:
        logger.error(f"  Error processing {file_path.name}: {e}")
        traceback.print_exc()
        return {}

    elapsed = time.time() - t0
    dates_found = sorted(date_snapshots.keys())
    logger.info(
        f"  [{file_idx}] {file_path.name}: {n_events:,} events -> "
        f"{rth_count:,} RTH snapshots across {len(dates_found)} dates "
        f"({non_rth_skipped:,} non-RTH skipped) [{elapsed:.0f}s]"
    )
    if dates_found:
        logger.info(f"    Dates found: {', '.join(dates_found)}")

    # Convert to arrays and validate each date
    result = {}
    for date_str in sorted(date_snapshots.keys()):
        data = date_snapshots[date_str]
        n_snaps = len(data['mids'])

        # Skip tiny segments (< MIN_SNAPSHOTS bars)
        if n_snaps < MIN_SNAPSHOTS:
            logger.info(f"    {date_str}: only {n_snaps} snapshots (need {MIN_SNAPSHOTS}), skipping")
            continue

        mid_arr = np.array(data['mids'], dtype=np.float32)
        price_range = float(mid_arr.max() - mid_arr.min())

        # Skip flat days (weekend/holiday with minimal price movement)
        if price_range < MIN_PRICE_RANGE_PTS:
            logger.info(f"    {date_str}: flat (range={price_range:.2f}pts < {MIN_PRICE_RANGE_PTS}), skipping")
            continue

        # Skip weekend dates (shouldn't happen for RTH, but belt-and-suspenders)
        if not _is_weekday(date_str):
            logger.info(f"    {date_str}: weekend, skipping")
            continue

        result[date_str] = {
            'node': np.array(data['nodes'], dtype=np.float32),
            'global': np.array(data['globals'], dtype=np.float32),
            'mid': mid_arr,
            'source_file': file_path.name,
            'price_range': price_range,
        }
        logger.info(f"    {date_str}: {n_snaps:,} RTH snapshots, range={price_range:.1f}pts")

    del date_snapshots
    gc.collect()

    return result


# ============================================================================
# MULTIPROCESSING WORKER
# ============================================================================

def _process_file_worker(args):
    """Worker function for multiprocessing Pool.

    Takes (file_idx, file_path_str, total_files) tuple.
    Returns (source_filename, date_data_dict) tuple.
    """
    file_idx, file_path_str, total_files = args
    file_path = Path(file_path_str)
    logger.info(f"\n[{file_idx + 1}/{total_files}] Processing {file_path.name}...")
    result = process_single_file(file_idx, file_path)
    return (file_path.name, result)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Rebuild per-date snapshot caches (one file per trading day)'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without doing it')
    parser.add_argument('--resume', action='store_true',
                        help='Skip source files already fully processed (reads manifest)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default 4)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("REBUILD SNAPSHOT CACHES — ONE FILE PER TRADING DAY")
    logger.info(f"  MBO dir: {MBO_DIR}")
    logger.info(f"  Cache dir: {CACHE_DIR}")
    logger.info(f"  Output format: YYYY-MM-DD_snapshots.npz")
    logger.info(f"  Instrument: auto-detect per file (ESU5 -> ESZ5 rollover)")
    logger.info(f"  Interval: {SAMPLE_INTERVAL_MS}ms")
    logger.info(f"  Depth: {DEPTH_LEVELS} levels")
    logger.info(f"  Min snapshots: {MIN_SNAPSHOTS}")
    logger.info(f"  Min price range: {MIN_PRICE_RANGE_PTS} pts")
    logger.info("=" * 70)

    # List source files — prefer .dbn over .dbn.zst, deduplicate by date prefix
    all_files = list(iter_files(str(MBO_DIR)))
    seen_prefixes = set()
    files = []
    for f in sorted(all_files, key=lambda p: (p.name.endswith('.zst'), p.name)):
        date_part = f.name.split('.')[0]  # e.g. glbx-mdp3-20250714
        if date_part not in seen_prefixes:
            seen_prefixes.add(date_part)
            files.append(f)
    logger.info(f"Found {len(files)} unique source files (from {len(all_files)} total)")

    if args.dry_run:
        for i, f in enumerate(files):
            iid = get_es_instrument_id(f.name)
            cname = get_es_contract_name(f.name)
            logger.info(f"  [{i}] {f.name} -> {cname} (id={iid})")
        logger.info(f"\nDRY RUN — {len(files)} files would be processed")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load resume state from manifest if --resume
    saved_dates = set()
    processed_source_files = set()

    if args.resume:
        manifest_path = CACHE_DIR / 'cache_manifest.json'
        if manifest_path.exists():
            try:
                with open(str(manifest_path)) as f:
                    old_manifest = json.load(f)
                saved_dates = set(old_manifest.get('dates', {}).keys())
                processed_source_files = set(old_manifest.get('processed_files', []))
                logger.info(
                    f"Resume mode: {len(saved_dates)} dates cached, "
                    f"{len(processed_source_files)} source files processed"
                )
            except Exception as e:
                logger.warning(f"Could not read manifest for resume: {e}")
                # Fall back to scanning cache directory
                for f in CACHE_DIR.glob("????-??-??_snapshots.npz"):
                    saved_dates.add(f.name[:10])
                logger.info(f"Resume mode (from files): {len(saved_dates)} dates found")
    else:
        # Clean start: remove ALL old cache files
        old_npz = list(CACHE_DIR.glob("*.npz"))
        old_json = list(CACHE_DIR.glob("*.json"))
        if old_npz or old_json:
            logger.info(f"Deleting {len(old_npz)} old .npz and {len(old_json)} old .json files...")
            for f in old_npz + old_json:
                f.unlink()
            logger.info("Old caches deleted")

        # Delete stale feature caches (must be recomputed with new data layout)
        feature_cache_dir = ROOT / "alpha_discovery" / "results" / "feature_cache"
        if feature_cache_dir.exists():
            stale = list(feature_cache_dir.glob("features_*"))
            if stale:
                logger.info(f"Deleting {len(stale)} stale feature caches...")
                for f in stale:
                    f.unlink()

    # ================================================================
    # Process all source files
    # ================================================================
    t_start = time.time()

    manifest = {
        'format_version': 3,
        'description': (
            'Per-trading-day snapshot caches. One file = one calendar day. '
            'Built from Databento MBO data with calendar date extracted from '
            'nanosecond UTC timestamps converted to Eastern Time.'
        ),
        'config': {
            'interval_ms': SAMPLE_INTERVAL_MS,
            'depth_levels': DEPTH_LEVELS,
            'rth_only': True,
            'rth_start': '09:30 ET',
            'rth_end': '16:00 ET',
            'min_snapshots': MIN_SNAPSHOTS,
            'min_price_range_pts': MIN_PRICE_RANGE_PTS,
        },
        'dates': {},
        'processed_files': list(processed_source_files),
    }

    # Carry forward already-saved dates from resume
    if args.resume and saved_dates:
        manifest_path = CACHE_DIR / 'cache_manifest.json'
        if manifest_path.exists():
            try:
                with open(str(manifest_path)) as f:
                    old = json.load(f)
                manifest['dates'] = old.get('dates', {})
            except Exception:
                pass

    total_new = 0
    total_skipped_dup = 0
    total_skipped_source = 0
    total_snapshots = sum(
        v.get('n_snapshots', 0) for v in manifest['dates'].values()
    )

    # Build list of files to process (skip already-processed in resume mode)
    files_to_process = []
    for file_idx, file_path in enumerate(files):
        if file_path.name in processed_source_files:
            total_skipped_source += 1
            logger.info(
                f"  [{file_idx + 1}/{len(files)}] {file_path.name} "
                f"already processed, skipping"
            )
            continue
        files_to_process.append((file_idx, file_path))

    n_workers = min(args.workers, len(files_to_process))
    n_workers = max(1, n_workers)
    logger.info(
        f"\nProcessing {len(files_to_process)} files with "
        f"{n_workers} parallel worker(s)..."
    )

    def _handle_result(source_name, date_data):
        """Save results from one processed file. Called in main process."""
        nonlocal total_new, total_skipped_dup, total_snapshots

        for date_str in sorted(date_data.keys()):
            if date_str in saved_dates:
                total_skipped_dup += 1
                logger.info(f"    {date_str}: already saved from earlier file, skipping duplicate")
                continue

            data = date_data[date_str]

            # Save cache
            cache_path = CACHE_DIR / f"{date_str}_snapshots.npz"
            np.savez_compressed(
                str(cache_path),
                node_features=data['node'],
                global_features=data['global'],
                mid_prices=data['mid'],
            )

            saved_dates.add(date_str)
            total_new += 1
            n_snaps = len(data['mid'])
            total_snapshots += n_snaps

            cache_size = cache_path.stat().st_size / 1e6
            manifest['dates'][date_str] = {
                'source_file': data['source_file'],
                'n_snapshots': n_snaps,
                'price_range': round(data['price_range'], 2),
                'open': round(float(data['mid'][0]), 2),
                'close': round(float(data['mid'][-1]), 2),
                'high': round(float(data['mid'].max()), 2),
                'low': round(float(data['mid'].min()), 2),
                'cache_size_mb': round(cache_size, 1),
            }

            logger.info(
                f"    SAVED {date_str}: {n_snaps:,} snapshots, "
                f"{cache_size:.1f} MB"
            )

        # Mark source file as processed
        processed_source_files.add(source_name)
        manifest['processed_files'] = list(processed_source_files)

        # Save interim manifest
        _save_manifest(manifest, saved_dates, total_snapshots, t_start)

    work_items = [
        (idx, str(fp), len(files)) for idx, fp in files_to_process
    ]

    n_completed = 0
    if n_workers > 1:
        with multiprocessing.Pool(n_workers) as pool:
            for source_name, date_data in pool.imap_unordered(
                _process_file_worker, work_items
            ):
                _handle_result(source_name, date_data)
                n_completed += 1
                logger.info(
                    f"  [{n_completed}/{len(files_to_process)}] files done, "
                    f"{len(saved_dates)} unique dates saved"
                )
                del date_data
                gc.collect()
    else:
        for item in work_items:
            source_name, date_data = _process_file_worker(item)
            _handle_result(source_name, date_data)
            n_completed += 1
            del date_data
            gc.collect()

    # ================================================================
    # Final manifest
    # ================================================================
    elapsed = time.time() - t_start
    _save_manifest(manifest, saved_dates, total_snapshots, t_start, final=True)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("REBUILD COMPLETE")
    logger.info(f"  Trading days saved: {total_new}")
    if total_skipped_dup > 0:
        logger.info(f"  Duplicate day-segments skipped: {total_skipped_dup}")
    if total_skipped_source > 0:
        logger.info(f"  Source files skipped (resume): {total_skipped_source}")
    logger.info(f"  Total unique trading days: {len(saved_dates)}")
    logger.info(f"  Total RTH snapshots: {total_snapshots:,}")

    if saved_dates:
        all_dates = sorted(saved_dates)
        logger.info(f"  Date range: {all_dates[0]} to {all_dates[-1]}")

        # Verify the expected count is reasonable
        expected_trading_days = _count_expected_trading_days(all_dates[0], all_dates[-1])
        logger.info(
            f"  Expected ~{expected_trading_days} trading days in range, "
            f"got {len(saved_dates)} "
            f"({'OK' if abs(len(saved_dates) - expected_trading_days) < 10 else 'CHECK!'})"
        )

    logger.info(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Manifest: {CACHE_DIR / 'cache_manifest.json'}")
    logger.info("=" * 70)

    # Verify one cache file
    if total_new > 0:
        verify_date = sorted(manifest['dates'].keys())[0]
        verify_path = CACHE_DIR / f"{verify_date}_snapshots.npz"
        data = np.load(str(verify_path))
        logger.info(f"\n  Verification ({verify_path.name}):")
        logger.info(f"    node_features: {data['node_features'].shape}")
        logger.info(f"    global_features: {data['global_features'].shape}")
        logger.info(f"    mid_prices: {data['mid_prices'].shape}")
        data.close()

    logger.info("\nDone! Run alpha scan with: python alpha_discovery/run_overnight_alpha.py --skip-cache-rebuild")


def _save_manifest(manifest, saved_dates, total_snapshots, t_start, final=False):
    """Save the cache manifest (called after each file for resume-ability)."""
    if saved_dates:
        all_dates = sorted(saved_dates)
        manifest['date_range'] = {'first': all_dates[0], 'last': all_dates[-1]}
    manifest['total_days'] = len(saved_dates)
    manifest['total_snapshots'] = total_snapshots
    manifest['build_time_sec'] = round(time.time() - t_start)
    if final:
        manifest['completed'] = datetime.now().isoformat()
    else:
        manifest['last_updated'] = datetime.now().isoformat()

    manifest_path = CACHE_DIR / 'cache_manifest.json'
    with open(str(manifest_path), 'w') as f:
        json.dump(manifest, f, indent=2)


def _count_expected_trading_days(first_date: str, last_date: str) -> int:
    """Estimate number of trading days between two dates (rough, excludes holidays)."""
    d1 = datetime.strptime(first_date, '%Y-%m-%d')
    d2 = datetime.strptime(last_date, '%Y-%m-%d')
    total = 0
    current = d1
    while current <= d2:
        if current.weekday() < 5:
            total += 1
        current += timedelta(days=1)
    # Subtract ~8 US market holidays in Jul-Nov period
    return total - 3  # Labor Day, Thanksgiving, + a couple partial days


if __name__ == '__main__':
    main()
