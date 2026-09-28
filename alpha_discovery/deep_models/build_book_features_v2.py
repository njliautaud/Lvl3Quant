"""
Build per-event book state features from raw Databento MBO DBN files.
v2: uses numpy array extraction instead of iterrows() — ~10x faster.

For each MBO event (add/cancel/modify/trade/fill), reconstructs the order book
and computes 30 features:
  - 20 book state: bid/ask price+size for top 5 levels
  -  5 running OF:  cum_delta, rolling_imbalance_100, trade_intensity_100,
                    depth_imbalance_5, spread_ticks
  -  5 book deltas: bid_size_change, ask_size_change, mid_change, spread_change, net_of

Output: NPZ per day at data/processed/mbo_book_features/<date>_book_features.npz
  keys: features (N,30), timestamps (N,), labels_1s/5s/10s/30s (N,)

Usage:
  python build_book_features_v2.py --date 20250714
  python build_book_features_v2.py --workers 8
"""

import os
import sys
import time
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional

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
TICK_SIZE = 0.25

N_LEVELS = 5
ROLLING_WINDOW = 100
CHUNK_SIZE = 100000

FEATURE_NAMES = [
    'bid_price_1','bid_price_2','bid_price_3','bid_price_4','bid_price_5',
    'ask_price_1','ask_price_2','ask_price_3','ask_price_4','ask_price_5',
    'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5',
    'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5',
    'cum_delta','rolling_imbalance_100','trade_intensity_100',
    'depth_imbalance_5','spread_ticks',
    'bid_size_change','ask_size_change','mid_price_change',
    'spread_change','net_order_flow',
]
assert len(FEATURE_NAMES) == 30


def process_day(dbn_path: Path, npz_path: Optional[Path] = None,
                out_dir: Optional[Path] = None) -> Optional[Path]:
    import databento as db

    _out_dir = out_dir if out_dir is not None else OUT_DIR
    date_str = dbn_path.name.split('-')[-1].replace('.mbo.dbn.zst', '')
    out_path = _out_dir / f'{date_str}_book_features.npz'

    if out_path.exists():
        logger.info(f'[{date_str}] Already exists, skipping')
        return out_path

    logger.info(f'[{date_str}] Loading DBN ({dbn_path.stat().st_size/1e6:.0f} MB)...')
    t0 = time.time()

    if npz_path is None:
        npz_path = MBO_EVENTS_DIR / f'{date_str}_mbo_events.npz'
    if not npz_path.exists():
        logger.error(f'[{date_str}] No mbo_events NPZ at {npz_path}')
        return None

    labels_data = np.load(npz_path, allow_pickle=True)
    ref_timestamps = labels_data['timestamps']
    labels_1s  = labels_data['labels_1s']
    labels_5s  = labels_data['labels_5s']
    labels_10s = labels_data['labels_10s']
    labels_30s = labels_data.get('labels_30s', np.full_like(labels_1s, np.nan))
    N_expected = len(ref_timestamps)
    logger.info(f'[{date_str}] Expected {N_expected} events')

    # Book state: price_int → size (separate dicts for bid/ask)
    bids: dict = {}
    asks: dict = {}
    orders: dict = {}  # order_id → (is_bid: bool, price_int: int, size: int)

    # Rolling buffer for OF stats (circular)
    roll_signed = np.zeros(ROLLING_WINDOW, dtype=np.float64)
    roll_trade  = np.zeros(ROLLING_WINDOW, dtype=np.float64)
    roll_ptr = 0

    cum_delta = 0.0
    cum_add   = 0.0
    cum_cancel = 0.0

    # Output buffer
    N_MAX = N_expected + 100000
    features_buf   = np.zeros((N_MAX, 30), dtype=np.float32)
    timestamps_buf = np.zeros(N_MAX, dtype=np.int64)
    n_out = 0

    prev_bid_sz1 = 0.0
    prev_ask_sz1 = 0.0
    prev_mid     = np.nan
    prev_spread_t = np.nan
    ref_price    = None
    n_records    = 0

    store = db.DBNStore.from_file(str(dbn_path))

    def snapshot_book():
        """Snapshot top-N_LEVELS bid/ask levels. Returns arrays."""
        if bids:
            bp = sorted(bids.keys(), reverse=True)[:N_LEVELS]
            bpv = [p * 1e-9 for p in bp]
            bsv = [float(bids[p]) for p in bp]
        else:
            bpv, bsv = [], []
        while len(bpv) < N_LEVELS:
            bpv.append(np.nan); bsv.append(0.0)

        if asks:
            ap = sorted(asks.keys())[:N_LEVELS]
            apv = [p * 1e-9 for p in ap]
            asv = [float(asks[p]) for p in ap]
        else:
            apv, asv = [], []
        while len(apv) < N_LEVELS:
            apv.append(np.nan); asv.append(0.0)

        return bpv, bsv, apv, asv

    for chunk in store.to_df(count=CHUNK_SIZE):
        # Extract columns as numpy arrays — much faster than iterrows
        actions   = chunk['action'].values        # str array
        sides     = chunk['side'].values          # str array
        prices    = chunk['price'].values         # float64
        sizes     = chunk['size'].values.astype(np.int64)
        order_ids = chunk['order_id'].values.astype(np.int64)
        ts_vals   = chunk['ts_event'].values      # datetime64[ns, UTC]

        # Convert timestamps to int64 nanoseconds
        ts_ns = ts_vals.astype('int64')  # pandas Timestamp → ns since epoch

        n_chunk = len(chunk)
        for i in range(n_chunk):
            n_records += 1
            action = actions[i]
            side   = sides[i]
            price_raw = prices[i]
            size   = int(sizes[i])
            oid    = int(order_ids[i])
            ts     = int(ts_ns[i])

            is_bid   = (side == 'B')
            is_trade = (action == 'T' or action == 'F')
            is_add   = (action == 'A')
            is_cancel = (action == 'C')
            is_modify = (action == 'M')

            if action == 'R':
                continue  # reset record, skip

            price_int = int(price_raw * 1e9) if not np.isnan(price_raw) else 0

            if ref_price is None and is_trade and price_int > 0:
                ref_price = price_raw  # float dollars

            book = bids if is_bid else asks

            # Update book state
            signed_vol = 0.0
            is_tr = 0.0

            if is_add:
                orders[oid] = (is_bid, price_int, size)
                book[price_int] = book.get(price_int, 0) + size
                cum_add += size

            elif is_cancel:
                if oid in orders:
                    o_bid, o_price, o_size = orders.pop(oid)
                    b = bids if o_bid else asks
                    cur = b.get(o_price, 0) - o_size
                    if cur <= 0:
                        b.pop(o_price, None)
                    else:
                        b[o_price] = cur
                    cum_cancel += o_size

            elif is_modify:
                if oid in orders:
                    o_bid, o_price, o_size = orders[oid]
                    b = bids if o_bid else asks
                    cur = b.get(o_price, 0) - o_size
                    if cur <= 0:
                        b.pop(o_price, None)
                    else:
                        b[o_price] = cur
                    orders[oid] = (o_bid, price_int, size)
                    b2 = bids if o_bid else asks
                    b2[price_int] = b2.get(price_int, 0) + size

            elif is_trade:
                # Aggressor crosses the book
                if is_bid:  # buy aggressor hits ask
                    cur = asks.get(price_int, 0) - size
                    if cur <= 0:
                        asks.pop(price_int, None)
                    else:
                        asks[price_int] = cur
                    signed_vol = float(size)
                else:        # sell aggressor hits bid
                    cur = bids.get(price_int, 0) - size
                    if cur <= 0:
                        bids.pop(price_int, None)
                    else:
                        bids[price_int] = cur
                    signed_vol = -float(size)
                cum_delta += signed_vol
                is_tr = 1.0

            # Update rolling buffer
            roll_signed[roll_ptr] = signed_vol
            roll_trade[roll_ptr]  = is_tr
            roll_ptr = (roll_ptr + 1) % ROLLING_WINDOW

            # Snapshot book
            bpv, bsv, apv, asv = snapshot_book()

            bid1 = bpv[0]
            ask1 = apv[0]
            mid = (bid1 + ask1) * 0.5 if (not np.isnan(bid1) and not np.isnan(ask1)) else np.nan

            if ref_price is None:
                ref_price = mid if not np.isnan(mid) else 0.0

            # Normalize prices to ticks relative to ref_price
            def to_ticks(p):
                return (p - ref_price) / TICK_SIZE if not np.isnan(p) else 0.0

            bp_t = [to_ticks(p) for p in bpv]
            ap_t = [to_ticks(p) for p in apv]
            spread_t = (ask1 - bid1) / TICK_SIZE if not np.isnan(mid) else 0.0
            mid_t = to_ticks(mid) if not np.isnan(mid) else 0.0

            # OF stats
            roll_imb = float(np.sum(roll_signed))
            roll_tr  = float(np.sum(roll_trade))
            bid_sz5  = sum(bsv)
            ask_sz5  = sum(asv)
            depth_imb = (bid_sz5 - ask_sz5) / max(bid_sz5 + ask_sz5, 1.0)

            # Deltas
            bid_sz1 = bsv[0]
            ask_sz1 = asv[0]
            d_bid = bid_sz1 - prev_bid_sz1
            d_ask = ask_sz1 - prev_ask_sz1
            d_mid = mid_t - (to_ticks(prev_mid) if not np.isnan(prev_mid) else 0.0)
            d_spr = spread_t - (prev_spread_t if not np.isnan(prev_spread_t) else 0.0)
            net_of = cum_add - cum_cancel

            prev_bid_sz1  = bid_sz1
            prev_ask_sz1  = ask_sz1
            prev_mid      = mid if not np.isnan(mid) else prev_mid
            prev_spread_t = spread_t

            # Assemble feature row
            if n_out >= N_MAX:
                features_buf   = np.vstack([features_buf, np.zeros((N_MAX, 30), dtype=np.float32)])
                timestamps_buf = np.concatenate([timestamps_buf, np.zeros(N_MAX, dtype=np.int64)])
                N_MAX *= 2

            features_buf[n_out, :5]  = bp_t
            features_buf[n_out, 5:10] = ap_t
            features_buf[n_out, 10:15] = bsv
            features_buf[n_out, 15:20] = asv
            features_buf[n_out, 20] = cum_delta
            features_buf[n_out, 21] = roll_imb
            features_buf[n_out, 22] = roll_tr
            features_buf[n_out, 23] = depth_imb
            features_buf[n_out, 24] = spread_t
            features_buf[n_out, 25] = d_bid
            features_buf[n_out, 26] = d_ask
            features_buf[n_out, 27] = d_mid
            features_buf[n_out, 28] = d_spr
            features_buf[n_out, 29] = net_of
            timestamps_buf[n_out]   = ts

            n_out += 1

        elapsed = time.time() - t0
        if n_records % 500000 < CHUNK_SIZE:
            logger.info(f'[{date_str}] {n_records/1e6:.1f}M records, {n_out} events out ({elapsed:.0f}s)')

    features_buf   = features_buf[:n_out]
    timestamps_buf = timestamps_buf[:n_out]

    elapsed = time.time() - t0
    logger.info(f'[{date_str}] Done: {n_records} records → {n_out} events in {elapsed:.0f}s')
    logger.info(f'[{date_str}] Expected {N_expected}, got {n_out} — diff={n_out - N_expected}')

    # Align to mbo_events event count
    if n_out > N_expected:
        features_buf   = features_buf[:N_expected]
        timestamps_buf = timestamps_buf[:N_expected]
    elif n_out < N_expected:
        pad = N_expected - n_out
        features_buf   = np.vstack([features_buf, np.zeros((pad, 30), dtype=np.float32)])
        timestamps_buf = np.concatenate([timestamps_buf, ref_timestamps[n_out:]])

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
    sz_mb = out_path.stat().st_size / 1e6
    logger.info(f'[{date_str}] Saved {out_path} ({features_buf.shape}, {sz_mb:.1f} MB)')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--raw-dir', type=str, default=str(RAW_DIR))
    parser.add_argument('--out-dir', type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if args.date:
        dbn_files = list(raw_dir.glob(f'*{args.date}*.dbn.zst'))
        if not dbn_files:
            logger.error(f'No DBN file for date {args.date} in {raw_dir}')
            sys.exit(1)
        result = process_day(dbn_files[0], out_dir=out_dir)
        sys.exit(0 if result else 1)
    else:
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
                        r = fut.result()
                        status = 'OK' if r else 'FAILED'
                    except Exception as e:
                        status = f'ERROR: {e}'
                    logger.info(f'[{done}/{len(dbn_files)}] {f.name}: {status}')


if __name__ == '__main__':
    main()
