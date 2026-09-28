#!/usr/bin/env python3
"""
Orderflow Feature Engineering — Jupiter
Reads MBO book tensor data and computes:
  1. Volume Profile (POC, Value Area High/Low)
  2. Cumulative Delta (bid vs ask volume imbalance over time)
  3. Large Order Detection (orders > 50 lots)

Checks available data in /home/jupiter/Lvl3Quant/data/

Deployment: /home/jupiter/Lvl3Quant/
"""
import sys, json, time, logging
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import databento as db
    HAS_DATABENTO = True
except ImportError:
    HAS_DATABENTO = False

logging.basicConfig(
    format='%(asctime)s [orderflow] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger('orderflow')

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
BOOK_CACHE_DIR = LVL3_ROOT / "data" / "processed" / "dl_book_cache_oot"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "orderflow_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MBO processing params
RTH_START_NANOS = 9 * 3600 * int(1e9) + 30 * 60 * int(1e9)   # 09:30 ET
RTH_END_NANOS   = 16 * 3600 * int(1e9)                         # 16:00 ET
BAR_NS = 100_000_000  # 100ms bars
LARGE_ORDER_LOTS = 50

# Volume profile params
TICK_SIZE = 0.25      # ES = 0.25 point per tick = $12.50
VALUE_AREA_PCT = 0.70  # 70% of volume defines value area


def compute_volume_profile(prices, volumes, tick_size=TICK_SIZE):
    """
    Compute volume profile: POC, Value Area High, Value Area Low.
    Returns dict with poc_price, vah_price, val_price.
    """
    if len(prices) == 0:
        return {'poc': np.nan, 'vah': np.nan, 'val': np.nan}

    # Round prices to tick
    rounded = np.round(prices / tick_size) * tick_size
    unique_prices = np.unique(rounded)

    # Aggregate volume at each price level
    vol_at_price = {}
    for p, v in zip(rounded, volumes):
        vol_at_price[p] = vol_at_price.get(p, 0) + v

    if not vol_at_price:
        return {'poc': np.nan, 'vah': np.nan, 'val': np.nan}

    sorted_prices = sorted(vol_at_price.keys())
    sorted_vols = [vol_at_price[p] for p in sorted_prices]
    total_vol = sum(sorted_vols)

    # POC = price with highest volume
    poc_idx = np.argmax(sorted_vols)
    poc = sorted_prices[poc_idx]

    # Value Area: expand from POC until 70% of volume captured
    target_vol = total_vol * VALUE_AREA_PCT
    accumulated = sorted_vols[poc_idx]
    lo_idx = poc_idx
    hi_idx = poc_idx

    while accumulated < target_vol:
        # Expand in direction of more volume
        can_go_up = hi_idx < len(sorted_prices) - 1
        can_go_down = lo_idx > 0

        if not can_go_up and not can_go_down:
            break

        vol_up = sorted_vols[hi_idx + 1] if can_go_up else 0
        vol_dn = sorted_vols[lo_idx - 1] if can_go_down else 0

        if vol_up >= vol_dn:
            hi_idx += 1
            accumulated += sorted_vols[hi_idx]
        else:
            lo_idx -= 1
            accumulated += sorted_vols[lo_idx]

    return {
        'poc': poc,
        'vah': sorted_prices[hi_idx],
        'val': sorted_prices[lo_idx],
        'total_volume': total_vol,
        'price_levels': len(sorted_prices),
    }


def compute_cumulative_delta(bid_vols, ask_vols):
    """
    Compute cumulative delta = cumsum(ask_vol - bid_vol).
    Positive delta = more buying pressure.
    Returns: cumulative delta array, rolling 100-bar delta.
    """
    delta = ask_vols - bid_vols
    cum_delta = np.cumsum(delta)
    # Rolling 100-bar delta (10 seconds at 100ms bars)
    roll_delta = np.convolve(delta, np.ones(100), mode='same')
    return cum_delta, roll_delta


def detect_large_orders(sizes, threshold=LARGE_ORDER_LOTS):
    """
    Count and flag bars with large orders (> threshold lots).
    Returns: large_order_count (per bar), large_order_binary (0/1 per bar).
    """
    large_mask = sizes >= threshold
    return large_mask.astype(np.int32), large_mask.astype(np.float32)


def process_mbo_file_pure_numpy(mbo_path: Path, date_str: str):
    """
    Process MBO file using databento Python API.
    Falls back to reading raw bytes if databento not available.
    Returns dict of feature arrays keyed by feature name.
    """
    log.info(f"Processing {mbo_path.name} ...")

    if not HAS_DATABENTO:
        log.error("databento package not installed. Cannot decode MBO files.")
        log.info("Install with: pip install databento")
        return None

    try:
        # Use databento to decode MBO file
        store = db.DBNStore.from_file(str(mbo_path))

        # Filter to RTH and ES/MES instruments
        # MBO records have: ts_event, price, size, side, action, order_id
        bars = []  # (ts_bar, price, size, side, action)

        for rec in store:
            if not hasattr(rec, 'ts_event'):
                continue

            # Filter to RTH
            ts_ns = rec.ts_event
            # Convert to seconds within day (UTC offset ~5h for ET)
            # Using bar bucket
            ts_day_ns = ts_ns % (24 * 3600 * int(1e9))
            if ts_day_ns < RTH_START_NANOS or ts_day_ns >= RTH_END_NANOS:
                continue

            bar_idx = int(ts_day_ns // BAR_NS)
            raw_price = getattr(rec, 'price', 0)
            price = raw_price / 1e9  # Databento fixed-point: divide by 1e9
            size = getattr(rec, 'size', 0)
            side = getattr(rec, 'side', 0)
            action = getattr(rec, 'action', 0)
            # Skip sentinel/invalid prices
            if price > 1e6 or price <= 0:
                continue

            bars.append((bar_idx, price, size, side, action))

        if not bars:
            log.warning(f"No RTH MBO records found in {mbo_path.name}")
            return None

        n_rth_bars = int((RTH_END_NANOS - RTH_START_NANOS) // BAR_NS)
        log.info(f"  {len(bars)} RTH MBO records, {n_rth_bars} 100ms bars")

        # Aggregate per bar
        bar_bid_vol = np.zeros(n_rth_bars)
        bar_ask_vol = np.zeros(n_rth_bars)
        bar_trade_vol = np.zeros(n_rth_bars)
        bar_max_size = np.zeros(n_rth_bars)
        bar_prices = []
        bar_vols = []

        for bar_idx, price, size, side, action in bars:
            if bar_idx >= n_rth_bars:
                continue
            # Normalize action and side to char
            act_char = chr(action) if isinstance(action, int) else (action.decode() if isinstance(action, bytes) else str(action))
            side_char = chr(side) if isinstance(side, int) else (side.decode() if isinstance(side, bytes) else str(side))
            # Trade records (action='T')
            if act_char == 'T':
                bar_trade_vol[bar_idx] += size
                if side_char == 'B':   # Bid aggressor = sell-initiated trade
                    bar_bid_vol[bar_idx] += size
                elif side_char == 'A': # Ask aggressor = buy-initiated trade
                    bar_ask_vol[bar_idx] += size
                bar_max_size[bar_idx] = max(bar_max_size[bar_idx], size)
                if price > 0 and price < 1e6:  # Sanity filter
                    bar_prices.append(price)
                    bar_vols.append(size)

        # All prices/volumes for volume profile
        all_prices = np.array(bar_prices)
        all_vols = np.array(bar_vols)

        # 1. Volume Profile (session-level)
        vp = compute_volume_profile(all_prices, all_vols)
        log.info(f"  Volume Profile: POC={vp['poc']:.2f}, VAH={vp['vah']:.2f}, VAL={vp['val']:.2f}")

        # 2. Cumulative Delta
        cum_delta, roll_delta = compute_cumulative_delta(bar_bid_vol, bar_ask_vol)
        final_cum_delta = float(cum_delta[-1]) if len(cum_delta) > 0 else 0
        log.info(f"  Cumulative Delta: {final_cum_delta:+.0f} (session end)")

        # 3. Large Orders
        large_count, large_binary = detect_large_orders(bar_max_size)
        n_large = int(large_count.sum())
        log.info(f"  Large Orders (>{LARGE_ORDER_LOTS} lots): {n_large} bars ({100*n_large/n_rth_bars:.1f}%)")

        # 4. Rolling 30-min volume profile (for intraday use)
        bars_per_30min = 30 * 60 * 10  # 30 min * 60s * 10 bars/s
        rolling_poc = np.full(n_rth_bars, np.nan)
        for i in range(bars_per_30min, n_rth_bars):
            window_vols = bar_trade_vol[i - bars_per_30min:i]
            if window_vols.sum() > 0:
                rolling_poc[i] = bar_prices[np.argmax(window_vols)] if len(bar_prices) > 0 else np.nan

        features = {
            'date': date_str,
            'n_bars': n_rth_bars,
            'bar_bid_vol': bar_bid_vol,
            'bar_ask_vol': bar_ask_vol,
            'bar_trade_vol': bar_trade_vol,
            'bar_max_size': bar_max_size,
            'cum_delta': cum_delta,
            'roll_delta_10s': roll_delta,
            'large_order_mask': large_binary,
            'large_order_count': large_count,
            'session_poc': float(vp['poc']),
            'session_vah': float(vp['vah']),
            'session_val': float(vp['val']),
            'session_total_vol': float(vp['total_volume']) if 'total_volume' in vp else 0,
            'final_cum_delta': final_cum_delta,
            'n_large_order_bars': n_large,
        }

        return features

    except Exception as e:
        log.error(f"Error processing {mbo_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_book_cache_fallback(date_str: str):
    """
    Fallback: use pre-built book cache tensors if MBO decoding fails.
    These have OHLCV-style data already aggregated.
    """
    date_clean = date_str.replace('-', '')
    # Try to find book cache file for this date
    patterns = [
        BOOK_CACHE_DIR / f"{date_clean}*.npz",
        BOOK_CACHE_DIR / f"{date_str}*.npz",
    ]

    cache_files = []
    for p in patterns:
        cache_files.extend(BOOK_CACHE_DIR.glob(p.name))

    if not cache_files:
        return None

    cf = cache_files[0]
    log.info(f"Using book cache fallback: {cf.name}")

    try:
        d = np.load(cf)
        keys = list(d.keys())
        log.info(f"  Book cache keys: {keys}")

        # Extract what we can from the cached tensor
        features = {'date': date_str, 'source': 'book_cache'}

        # Look for bid/ask volume columns
        if 'book' in d:
            book = d['book']
            log.info(f"  Book tensor shape: {book.shape}")

            n_bars = len(book)
            features['n_bars'] = n_bars

            # Common book tensor layout: [bid_size, ask_size, mid, spread, ...]
            # Try to extract cumulative delta from bid/ask size changes
            if book.shape[1] >= 2:
                bid_sz = book[:, 0]
                ask_sz = book[:, 1]
                # Delta proxy: ask_size - bid_size imbalance
                delta_proxy = ask_sz - bid_sz
                features['cum_delta'] = np.cumsum(delta_proxy)
                features['roll_delta_10s'] = np.convolve(delta_proxy, np.ones(100), mode='same')
                features['final_cum_delta'] = float(features['cum_delta'][-1])

            # Mid price for volume profile approximation
            if book.shape[1] >= 3:
                mid = book[:, 2]
                # Synthetic volume (uniform) for basic POC
                vol_proxy = np.ones(n_bars)
                vp = compute_volume_profile(mid, vol_proxy)
                features['session_poc'] = float(vp['poc'])
                features['session_vah'] = float(vp['vah'])
                features['session_val'] = float(vp['val'])

            # Large order proxy from size spikes
            if book.shape[1] >= 1:
                total_sz = book[:, 0] + (book[:, 1] if book.shape[1] > 1 else 0)
                # Normalize and flag outliers as large orders
                sz_thresh = np.percentile(total_sz[total_sz > 0], 95) if (total_sz > 0).sum() > 0 else 1
                features['large_order_mask'] = (total_sz >= sz_thresh).astype(np.float32)
                features['large_order_count'] = features['large_order_mask'].astype(np.int32)
                features['n_large_order_bars'] = int(features['large_order_mask'].sum())

        return features

    except Exception as e:
        log.error(f"Error loading book cache {cf.name}: {e}")
        return None


def save_features(features: dict, date_str: str):
    """Save feature arrays to NPZ and summary to JSON."""
    date_clean = date_str.replace('-', '')
    npz_path = OUT_DIR / f"{date_clean}_orderflow.npz"
    json_path = OUT_DIR / f"{date_clean}_orderflow_summary.json"

    # Arrays to save
    arrays = {}
    summary = {}

    for k, v in features.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        else:
            summary[k] = v

    if arrays:
        np.savez_compressed(str(npz_path), **arrays)

    summary['npz_file'] = str(npz_path)
    summary['generated_at'] = datetime.now().isoformat()
    json_path.write_text(json.dumps(summary, indent=2))

    log.info(f"Saved features to {npz_path.name} + summary JSON")
    return npz_path


def check_data():
    """Check what data is available."""
    log.info("Checking available data...")
    log.info(f"MBO files: {len(list(MBO_DIR.glob('*.mbo.dbn.zst')))} files")
    log.info(f"Book cache: {len(list(BOOK_CACHE_DIR.glob('*.npz')))} files")
    log.info(f"databento available: {HAS_DATABENTO}")

    # List some dates
    mbo_files = sorted(MBO_DIR.glob("*.mbo.dbn.zst"))
    if mbo_files:
        log.info(f"MBO date range: {mbo_files[0].name[:20]} ... {mbo_files[-1].name[:20]}")


def main():
    log.info("=" * 60)
    log.info("ORDERFLOW FEATURE ENGINEERING")
    log.info("=" * 60)

    check_data()

    # Process recent dates first (most useful for research)
    mbo_files = sorted(MBO_DIR.glob("*.mbo.dbn.zst"))

    if not mbo_files:
        log.error(f"No MBO files found in {MBO_DIR}")
        sys.exit(1)

    # Focus on Sep-Nov 2025 to align with meta-model coverage
    # These files follow pattern glbx-mdp3-YYYYMMDD.mbo.dbn.zst
    target_files = []
    for f in mbo_files:
        stem = f.stem.replace('.mbo.dbn', '')  # strip extension noise
        date_part = stem.split('-')[-1]  # YYYYMMDD
        if len(date_part) == 8 and date_part.isdigit():
            year = int(date_part[:4])
            month = int(date_part[4:6])
            # Sep-Nov 2025
            if year == 2025 and 9 <= month <= 11:
                target_files.append((date_part, f))
            # Also include recent 2026 dates for testing
            elif year == 2026 and month <= 3:
                target_files.append((date_part, f))

    # Sort and take first 30 (enough for research)
    target_files = sorted(target_files)[:30]
    log.info(f"Processing {len(target_files)} dates")

    all_summaries = []
    processed = 0
    failed = 0

    for date_nodash, mbo_path in target_files:
        date_str = f"{date_nodash[:4]}-{date_nodash[4:6]}-{date_nodash[6:8]}"
        out_check = OUT_DIR / f"{date_nodash}_orderflow_summary.json"

        if out_check.exists():
            log.info(f"Skipping {date_str} (already processed)")
            processed += 1
            try:
                summary = json.loads(out_check.read_text())
                all_summaries.append(summary)
            except:
                pass
            continue

        # Try MBO decoding first
        features = None
        if HAS_DATABENTO:
            features = process_mbo_file_pure_numpy(mbo_path, date_str)

        # Fallback to book cache
        if features is None:
            features = process_book_cache_fallback(date_str)

        if features is None:
            log.warning(f"Could not process {date_str} — skipping")
            failed += 1
            continue

        npz_path = save_features(features, date_str)
        all_summaries.append({
            'date': date_str,
            'n_large_order_bars': features.get('n_large_order_bars', 0),
            'final_cum_delta': features.get('final_cum_delta', 0),
            'session_poc': features.get('session_poc', np.nan),
            'session_vah': features.get('session_vah', np.nan),
            'session_val': features.get('session_val', np.nan),
        })
        processed += 1

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETED: {processed} processed, {failed} failed")
    log.info(f"Output dir: {OUT_DIR}")

    # Aggregate summary
    if all_summaries:
        summary_path = OUT_DIR / "all_dates_summary.json"
        summary_path.write_text(json.dumps(all_summaries, indent=2, default=str))

        # Print stats
        deltas = [s['final_cum_delta'] for s in all_summaries if 'final_cum_delta' in s and s['final_cum_delta'] != 0]
        large_bars = [s['n_large_order_bars'] for s in all_summaries if 'n_large_order_bars' in s]

        if deltas:
            log.info(f"\nCumulative Delta stats:")
            log.info(f"  Mean: {np.mean(deltas):+.0f}")
            log.info(f"  Std:  {np.std(deltas):.0f}")
            log.info(f"  Bullish days (delta>0): {100*(np.array(deltas)>0).mean():.0f}%")

        if large_bars:
            log.info(f"\nLarge Orders stats:")
            log.info(f"  Mean large-order bars/day: {np.mean(large_bars):.1f}")
            log.info(f"  Max: {max(large_bars)}")

        log.info(f"\nSummary saved to {summary_path}")

    if not HAS_DATABENTO:
        log.info("\n*** NOTE: databento not installed. Feature quality is limited. ***")
        log.info("*** For full MBO decoding: pip install databento ***")
        log.info("*** The script produced best-effort features from book cache tensors. ***")


if __name__ == "__main__":
    main()
