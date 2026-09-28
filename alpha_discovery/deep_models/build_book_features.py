"""
Build per-event book state features from raw Databento MBO DBN files.

For each MBO event (add/cancel/modify/trade/fill), reconstructs the order book
and computes 30 features:
  - 20 book state: bid/ask price+size for top 5 levels
  -  5 running OF:  cum_delta, rolling_imbalance_100, trade_intensity_100,
                    depth_imbalance_5, spread
  -  5 book deltas: bid_size_change, ask_size_change, mid_change, spread_change, net_of

Output: NPZ per day at data/processed/mbo_book_features/<date>_book_features.npz
  keys: features (N,30), timestamps (N,), labels_1s/5s/10s/30s (N,)

Usage:
  python build_book_features.py --date 20250714          # single day
  python build_book_features.py --workers 8              # all days, parallel
"""

import os
import sys
import time
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

RAW_DIR = Path('/home/jupiter/Lvl3Quant/data/raw/mbo')
MBO_EVENTS_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events')
OUT_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_book_features')
TICK_SIZE = 0.25  # ES tick size in points — prices stored as int64 * 1e-9

N_LEVELS = 5
ROLLING_WINDOW = 100  # events for rolling stats

FEATURE_NAMES = [
    # Book state (20)
    'bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4', 'bid_price_5',
    'ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5',
    'bid_size_1',  'bid_size_2',  'bid_size_3',  'bid_size_4',  'bid_size_5',
    'ask_size_1',  'ask_size_2',  'ask_size_3',  'ask_size_4',  'ask_size_5',
    # Running OF (5)
    'cum_delta',
    'rolling_imbalance_100',
    'trade_intensity_100',
    'depth_imbalance_5',
    'spread',
    # Book deltas (5)
    'bid_size_change',
    'ask_size_change',
    'mid_price_change',
    'spread_change',
    'net_order_flow',
]
assert len(FEATURE_NAMES) == 30


def price_to_ticks(price_int: int, ref_price: float, tick_size: float = TICK_SIZE) -> float:
    """Convert Databento int64 price (1e-9 units) to ticks relative to ref."""
    price_float = price_int * 1e-9
    return (price_float - ref_price) / tick_size


def process_day(dbn_path: Path, npz_path: Optional[Path] = None, out_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Process one day of MBO DBN data → 30-feature per-event NPZ.

    Args:
        dbn_path: path to .mbo.dbn.zst file
        npz_path: path to existing mbo_events NPZ (for labels + timestamps alignment)
        out_dir: output directory (defaults to OUT_DIR module-level constant)

    Returns:
        output path if successful, None on error
    """
    import databento as db

    _out_dir = out_dir if out_dir is not None else OUT_DIR
    date_str = dbn_path.name.split('-')[-1].replace('.mbo.dbn.zst', '')
    out_path = _out_dir / f'{date_str}_book_features.npz'

    if out_path.exists():
        logger.info(f'[{date_str}] Already exists, skipping')
        return out_path

    logger.info(f'[{date_str}] Loading DBN file ({dbn_path.stat().st_size/1e6:.0f} MB)...')
    t0 = time.time()

    # Load corresponding mbo_events NPZ for labels and timestamps
    if npz_path is None:
        npz_path = MBO_EVENTS_DIR / f'{date_str}_mbo_events.npz'
    if not npz_path.exists():
        logger.error(f'[{date_str}] No mbo_events NPZ found at {npz_path}')
        return None

    labels_data = np.load(npz_path, allow_pickle=True)
    ref_timestamps = labels_data['timestamps']  # (N,) int64 nanoseconds
    labels_1s  = labels_data['labels_1s']
    labels_5s  = labels_data['labels_5s']
    labels_10s = labels_data['labels_10s']
    labels_30s = labels_data.get('labels_30s', np.full_like(labels_1s, np.nan))

    logger.info(f'[{date_str}] Labels loaded: {len(ref_timestamps)} events')

    # Book state: {price_int: size} for each side
    bids: dict[int, int] = {}   # price_int → total_size
    asks: dict[int, int] = {}
    orders: dict[int, tuple] = {}  # order_id → (side, price_int, size)

    # Rolling buffers for OF stats
    roll_signed_vol = np.zeros(ROLLING_WINDOW, dtype=np.float32)  # circular
    roll_is_trade   = np.zeros(ROLLING_WINDOW, dtype=np.float32)
    roll_ptr = 0
    cum_delta = 0.0
    cum_add = 0.0
    cum_cancel = 0.0

    # Output arrays — pre-allocate to max possible size
    N_MAX = len(ref_timestamps) * 2  # DBN may have more events (e.g. R records)
    features_buf = np.zeros((N_MAX, 30), dtype=np.float32)
    timestamps_buf = np.zeros(N_MAX, dtype=np.int64)
    n_out = 0

    prev_bid_sz1 = 0.0
    prev_ask_sz1 = 0.0
    prev_mid = np.nan
    prev_spread = np.nan

    def get_top5(book: dict, descending: bool) -> tuple:
        """Return top-5 (price_float, size) tuples from book dict."""
        if not book:
            return [(np.nan, 0.0)] * N_LEVELS
        prices = sorted(book.keys(), reverse=descending)[:N_LEVELS]
        result = [(p * 1e-9, float(book[p])) for p in prices]
        while len(result) < N_LEVELS:
            result.append((np.nan, 0.0))
        return result

    store = db.DBNStore.from_file(str(dbn_path))
    n_records = 0
    ref_price = None  # will be set from first valid trade price

    CHUNK_SIZE = 50000  # rows per DataFrame chunk
    for chunk in store.to_df(count=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            n_records += 1
            action = row['action']
            side   = row['side']
            price  = row['price']    # int64, 1e-9 units
            size   = int(row['size'])
            oid    = int(row['order_id'])
            ts     = row['ts_event']

            # Convert timestamp to nanoseconds int
            ts_ns = int(ts.value) if hasattr(ts, 'value') else int(ts)

            is_bid = (side == 'B')
            is_ask = (side == 'A')
            is_trade = action in ('T', 'F')
            is_add    = (action == 'A')
            is_cancel = (action == 'C')
            is_modify = (action == 'M')

            # Skip non-order records (R = reset/clear)
            if action == 'R':
                continue

            price_int = int(price) if not np.isnan(price) else 0

            # Set ref_price from first trade
            if ref_price is None and is_trade and price_int > 0:
                ref_price = price_int * 1e-9

            # --- Update book state ---
            book = bids if is_bid else asks

            if is_add:
                orders[oid] = (side, price_int, size)
                book[price_int] = book.get(price_int, 0) + size
                cum_add += size

            elif is_cancel:
                if oid in orders:
                    o_side, o_price, o_size = orders.pop(oid)
                    b = bids if o_side == 'B' else asks
                    b[o_price] = max(0, b.get(o_price, 0) - o_size)
                    if b[o_price] == 0:
                        del b[o_price]
                    cum_cancel += o_size

            elif is_modify:
                if oid in orders:
                    o_side, o_price, o_size = orders[oid]
                    b = bids if o_side == 'B' else asks
                    b[o_price] = max(0, b.get(o_price, 0) - o_size)
                    if b[o_price] == 0:
                        del b[o_price]
                    # New price/size
                    orders[oid] = (o_side, price_int, size)
                    b[price_int] = b.get(price_int, 0) + size

            elif is_trade:
                # Trade: remove from book (aggressor hits resting order)
                if is_bid:
                    asks[price_int] = max(0, asks.get(price_int, 0) - size)
                    if asks.get(price_int, 0) == 0:
                        asks.pop(price_int, None)
                    signed = float(size)   # buy aggressor = positive delta
                else:
                    bids[price_int] = max(0, bids.get(price_int, 0) - size)
                    if bids.get(price_int, 0) == 0:
                        bids.pop(price_int, None)
                    signed = -float(size)  # sell aggressor = negative delta

                cum_delta += signed
                roll_signed_vol[roll_ptr] = signed
                roll_is_trade[roll_ptr]   = 1.0
                roll_ptr = (roll_ptr + 1) % ROLLING_WINDOW

            # Rolling update for non-trade events
            if not is_trade:
                roll_signed_vol[roll_ptr] = 0.0
                roll_is_trade[roll_ptr]   = 0.0
                roll_ptr = (roll_ptr + 1) % ROLLING_WINDOW

            # --- Snapshot book state ---
            top_bids = get_top5(bids, descending=True)
            top_asks = get_top5(asks, descending=False)

            bid_prices = [b[0] for b in top_bids]
            ask_prices = [a[0] for a in top_asks]
            bid_sizes  = [b[1] for b in top_bids]
            ask_sizes  = [a[1] for a in top_asks]

            bid1 = bid_prices[0] if not np.isnan(bid_prices[0]) else np.nan
            ask1 = ask_prices[0] if not np.isnan(ask_prices[0]) else np.nan
            mid = (bid1 + ask1) / 2 if (not np.isnan(bid1) and not np.isnan(ask1)) else np.nan
            spread = (ask1 - bid1) if (not np.isnan(bid1) and not np.isnan(ask1)) else np.nan

            # Normalize prices to ticks relative to rolling mid (use last known mid)
            if ref_price is None:
                ref_price = mid if not np.isnan(mid) else 0.0

            bid_p_ticks = [(p - ref_price) / TICK_SIZE if not np.isnan(p) else np.nan for p in bid_prices]
            ask_p_ticks = [(p - ref_price) / TICK_SIZE if not np.isnan(p) else np.nan for p in ask_prices]
            spread_ticks = spread / TICK_SIZE if spread is not None and not np.isnan(spread) else np.nan

            # Rolling OF stats
            roll_imbalance = float(np.sum(roll_signed_vol))  # raw signed vol rolling sum
            roll_trade_count = float(np.sum(roll_is_trade))

            bid_sz_top5 = sum(bid_sizes[:N_LEVELS])
            ask_sz_top5 = sum(ask_sizes[:N_LEVELS])
            depth_imb = (bid_sz_top5 - ask_sz_top5) / max(bid_sz_top5 + ask_sz_top5, 1)

            # Book deltas
            bid_sz1 = bid_sizes[0]
            ask_sz1 = ask_sizes[0]
            d_bid_sz  = bid_sz1 - prev_bid_sz1
            d_ask_sz  = ask_sz1 - prev_ask_sz1
            d_mid     = (mid - prev_mid) / TICK_SIZE if (not np.isnan(mid) and not np.isnan(prev_mid)) else 0.0
            d_spread  = (spread_ticks - prev_spread / TICK_SIZE) if (not np.isnan(spread_ticks) and not np.isnan(prev_spread)) else 0.0 if not np.isnan(spread_ticks) else 0.0
            net_of    = cum_add - cum_cancel

            prev_bid_sz1 = bid_sz1
            prev_ask_sz1 = ask_sz1
            prev_mid     = mid if not np.isnan(mid) else prev_mid
            prev_spread  = spread if spread is not None and not np.isnan(spread) else prev_spread

            # --- Build feature vector ---
            if n_out >= N_MAX:
                logger.warning(f'[{date_str}] Buffer overflow at {n_out}, expanding')
                features_buf  = np.vstack([features_buf,  np.zeros((N_MAX, 30), dtype=np.float32)])
                timestamps_buf = np.concatenate([timestamps_buf, np.zeros(N_MAX, dtype=np.int64)])
                N_MAX *= 2

            row_feats = (
                bid_p_ticks +       # 5
                ask_p_ticks +       # 5
                bid_sizes  +        # 5
                ask_sizes  +        # 5
                [                   # 5 OF
                    cum_delta,
                    roll_imbalance,
                    roll_trade_count,
                    depth_imb,
                    spread_ticks if not np.isnan(spread_ticks) else 0.0,
                ] +
                [                   # 5 deltas
                    d_bid_sz,
                    d_ask_sz,
                    d_mid,
                    d_spread,
                    net_of,
                ]
            )

            features_buf[n_out] = [x if x is not None and not (isinstance(x, float) and np.isnan(x)) else 0.0 for x in row_feats]
            timestamps_buf[n_out] = ts_ns
            n_out += 1

        if n_records % 500000 == 0:
            elapsed = time.time() - t0
            logger.info(f'[{date_str}] {n_records/1e6:.1f}M records processed ({elapsed:.0f}s)...')

    features_buf  = features_buf[:n_out]
    timestamps_buf = timestamps_buf[:n_out]

    logger.info(f'[{date_str}] DBN done: {n_records} records → {n_out} events in {time.time()-t0:.0f}s')

    # Align to mbo_events timestamps (inner join on ts_ns)
    # The DBN events should match mbo_events 1:1 — verify count
    logger.info(f'[{date_str}] DBN events: {n_out}, NPZ events: {len(ref_timestamps)}')

    if abs(n_out - len(ref_timestamps)) / max(len(ref_timestamps), 1) > 0.05:
        logger.warning(f'[{date_str}] Event count mismatch >5% — aligning by timestamp')
        # Align by timestamp: for each ref timestamp, find closest DBN timestamp
        sort_idx = np.argsort(timestamps_buf)
        ts_sorted = timestamps_buf[sort_idx]
        feat_sorted = features_buf[sort_idx]

        aligned_feats = np.zeros((len(ref_timestamps), 30), dtype=np.float32)
        for i, ts in enumerate(ref_timestamps):
            pos = np.searchsorted(ts_sorted, ts)
            pos = min(pos, len(ts_sorted) - 1)
            aligned_feats[i] = feat_sorted[pos]

        features_buf = aligned_feats
    elif n_out != len(ref_timestamps):
        # Minor count diff — truncate/pad to match
        if n_out > len(ref_timestamps):
            features_buf = features_buf[:len(ref_timestamps)]
        else:
            pad = np.zeros((len(ref_timestamps) - n_out, 30), dtype=np.float32)
            features_buf = np.vstack([features_buf, pad])

    _out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features_buf,
        timestamps=ref_timestamps,
        labels_1s=labels_1s,
        labels_5s=labels_5s,
        labels_10s=labels_10s,
        labels_30s=labels_30s,
        feature_names=np.array(FEATURE_NAMES),
    )

    logger.info(f'[{date_str}] Saved: {out_path} ({features_buf.shape}, {out_path.stat().st_size/1e6:.1f} MB)')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Build per-event book state features from MBO DBN files')
    parser.add_argument('--date', type=str, default=None,
                        help='Process single date YYYYMMDD (default: all dates)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Parallel workers (default: 1 for single-day proof of concept)')
    parser.add_argument('--raw-dir', type=str, default=str(RAW_DIR))
    parser.add_argument('--out-dir', type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.date:
        # Single day
        dbn_files = list(raw_dir.glob(f'*{args.date}*.dbn.zst'))
        if not dbn_files:
            logger.error(f'No DBN file found for date {args.date} in {raw_dir}')
            sys.exit(1)
        result = process_day(dbn_files[0], out_dir=out_dir)
        if result:
            logger.info(f'SUCCESS: {result}')
        else:
            logger.error('FAILED')
            sys.exit(1)
    else:
        # All days
        dbn_files = sorted(raw_dir.glob('*.dbn.zst'))
        logger.info(f'Processing {len(dbn_files)} days with {args.workers} workers')

        if args.workers == 1:
            for f in dbn_files:
                process_day(f, out_dir=out_dir)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(process_day, f, None, out_dir): f for f in dbn_files}
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    f = futs[fut]
                    try:
                        result = fut.result()
                        status = 'OK' if result else 'FAILED'
                    except Exception as e:
                        status = f'ERROR: {e}'
                    logger.info(f'[{done}/{len(dbn_files)}] {f.name}: {status}')


if __name__ == '__main__':
    main()
