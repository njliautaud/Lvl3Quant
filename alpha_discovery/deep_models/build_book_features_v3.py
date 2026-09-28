"""
Build per-event book state features from raw Databento MBO DBN files.
v3: Fixed price key (float), fixed event alignment (skip R records only), no timestamp search.

30 features per event:
  [0:5]   bid_price_1..5  (ticks relative to session ref)
  [5:10]  ask_price_1..5
  [10:15] bid_size_1..5
  [15:20] ask_size_1..5
  [20]    cum_delta
  [21]    rolling_imbalance_100
  [22]    trade_intensity_100
  [23]    depth_imbalance_5
  [24]    spread_ticks
  [25]    bid_size_change
  [26]    ask_size_change
  [27]    mid_price_change_ticks
  [28]    spread_change_ticks
  [29]    net_order_flow  (cum_add - cum_cancel, in lots)

Usage:
  python build_book_features_v3.py --date 20250714
  python build_book_features_v3.py --workers 8
"""

import sys, time, argparse, logging, json
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

RAW_DIR       = Path('/home/jupiter/Lvl3Quant/data/raw/mbo')
MBO_EVENTS_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events')
OUT_DIR       = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_book_features')
TICK_SIZE     = 0.25
N_LEVELS      = 5
ROLLING_W     = 100
CHUNK_SIZE    = 100_000

FEATURE_NAMES = [
    'bid_price_1','bid_price_2','bid_price_3','bid_price_4','bid_price_5',
    'ask_price_1','ask_price_2','ask_price_3','ask_price_4','ask_price_5',
    'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5',
    'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5',
    'cum_delta','rolling_imbalance_100','trade_intensity_100',
    'depth_imbalance_5','spread_ticks',
    'bid_size_change','ask_size_change','mid_price_change_ticks',
    'spread_change_ticks','net_order_flow',
]
assert len(FEATURE_NAMES) == 30


def process_day(dbn_path: Path, npz_path: Optional[Path] = None,
                out_dir: Optional[Path] = None) -> Optional[Path]:
    import databento as db

    _out_dir = out_dir or OUT_DIR
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
        logger.error(f'[{date_str}] No NPZ at {npz_path}')
        return None

    npz = np.load(npz_path, allow_pickle=True)
    ref_ts   = npz['timestamps']     # (N,) int64 ns — ground truth order
    labels_1s  = npz['labels_1s']
    labels_5s  = npz['labels_5s']
    labels_10s = npz['labels_10s']
    labels_30s = npz.get('labels_30s', np.full_like(labels_1s, np.nan))
    N = len(ref_ts)
    logger.info(f'[{date_str}] NPZ: {N} events')

    # Book state — keyed by price as float (rounded to avoid float drift)
    # Use int representation: price_key = round(price / TICK_SIZE) as int (tick index)
    bids: dict = {}   # tick_idx → size
    asks: dict = {}
    orders: dict = {}  # order_id → (is_bid: bool, tick_idx: int, size: int)

    # Rolling OF buffers
    roll_signed = np.zeros(ROLLING_W, dtype=np.float64)
    roll_trade  = np.zeros(ROLLING_W, dtype=np.float64)
    roll_ptr    = 0

    cum_delta  = 0.0
    cum_add    = 0
    cum_cancel = 0

    # Output
    features_buf = np.zeros((N, 30), dtype=np.float32)
    n_out = 0

    prev_bid1 = 0.0
    prev_ask1 = 0.0
    prev_mid_t  = 0.0
    prev_spr_t  = 0.0
    ref_price = None  # set from first trade, used for price normalization

    # Determine target instrument_id from NPZ metadata
    import json
    meta = json.loads(npz['metadata'].item())
    target_iid = int(meta['instrument_id'])
    logger.info(f'[{date_str}] Filtering to instrument_id={target_iid}')

    n_records = 0
    store = db.DBNStore.from_file(str(dbn_path))

    def price_to_tick_idx(p_float: float) -> int:
        """Convert float dollar price to integer tick index."""
        return int(round(p_float / TICK_SIZE))

    def snap5(book: dict, descending: bool):
        """Return top-5 (tick_idx, size) pairs."""
        if not book:
            return [(0, 0)] * N_LEVELS
        keys = sorted(book.keys(), reverse=descending)[:N_LEVELS]
        result = [(k, book[k]) for k in keys]
        while len(result) < N_LEVELS:
            result.append((0, 0))
        return result

    for chunk in store.to_df(count=CHUNK_SIZE):
        # Filter to target instrument only, then drop R records
        valid = chunk[(chunk['instrument_id'] == target_iid) & (chunk['action'] != 'R')]
        if valid.empty:
            continue

        actions   = valid['action'].values
        sides     = valid['side'].values
        prices    = valid['price'].values        # float dollars
        sizes     = valid['size'].values.astype(np.int32)
        order_ids = valid['order_id'].values.astype(np.int64)

        for i in range(len(valid)):
            if n_out >= N:
                break  # safety: don't exceed NPZ event count

            action = actions[i]
            side   = sides[i]
            price  = prices[i]
            size   = int(sizes[i])
            oid    = int(order_ids[i])
            n_records += 1

            is_bid   = (side == 'B')
            is_trade = (action == 'T' or action == 'F')
            is_add   = (action == 'A')
            is_cancel = (action == 'C')
            is_modify = (action == 'M')

            tk = price_to_tick_idx(price) if not (price != price) else 0  # nan check
            book = bids if is_bid else asks
            signed = 0.0
            is_tr  = 0.0

            if is_add:
                orders[oid] = (is_bid, tk, size)
                book[tk] = book.get(tk, 0) + size
                cum_add += size

            elif is_cancel:
                if oid in orders:
                    o_bid, o_tk, o_sz = orders.pop(oid)
                    b = bids if o_bid else asks
                    cur = b.get(o_tk, 0) - o_sz
                    if cur <= 0: b.pop(o_tk, None)
                    else: b[o_tk] = cur
                    cum_cancel += o_sz

            elif is_modify:
                if oid in orders:
                    o_bid, o_tk, o_sz = orders[oid]
                    b = bids if o_bid else asks
                    cur = b.get(o_tk, 0) - o_sz
                    if cur <= 0: b.pop(o_tk, None)
                    else: b[o_tk] = cur
                    orders[oid] = (o_bid, tk, size)
                    b2 = bids if o_bid else asks
                    b2[tk] = b2.get(tk, 0) + size

            elif is_trade:
                if is_bid:  # buy aggressor hits ask
                    cur = asks.get(tk, 0) - size
                    if cur <= 0: asks.pop(tk, None)
                    else: asks[tk] = cur
                    signed =  float(size)
                else:        # sell aggressor hits bid
                    cur = bids.get(tk, 0) - size
                    if cur <= 0: bids.pop(tk, None)
                    else: bids[tk] = cur
                    signed = -float(size)
                cum_delta += signed
                is_tr = 1.0
                if ref_price is None:
                    ref_price = tk  # tick index of first trade

            roll_signed[roll_ptr] = signed
            roll_trade[roll_ptr]  = is_tr
            roll_ptr = (roll_ptr + 1) % ROLLING_W

            # Snapshot
            top_b = snap5(bids, descending=True)
            top_a = snap5(asks, descending=False)

            bid1_tk = top_b[0][0]
            ask1_tk = top_a[0][0]

            if ref_price is None and bid1_tk > 0 and ask1_tk > 0:
                ref_price = (bid1_tk + ask1_tk) // 2

            ref = ref_price if ref_price else 0

            # Book feature arrays (ticks relative to ref)
            bp_t = [float(b[0] - ref) if b[1] > 0 else 0.0 for b in top_b]
            ap_t = [float(a[0] - ref) if a[1] > 0 else 0.0 for a in top_a]
            bsz  = [float(b[1]) for b in top_b]
            asz  = [float(a[1]) for a in top_a]

            bid1_t = bp_t[0]
            ask1_t = ap_t[0]
            mid_t  = (bid1_t + ask1_t) * 0.5 if (bsz[0] > 0 and asz[0] > 0) else 0.0
            spr_t  = (ask1_t - bid1_t) if (bsz[0] > 0 and asz[0] > 0) else 0.0

            roll_imb = float(np.sum(roll_signed))
            roll_tr  = float(np.sum(roll_trade))
            b5 = sum(bsz); a5 = sum(asz)
            depth_imb = (b5 - a5) / max(b5 + a5, 1.0)

            # Deltas
            d_bid = bsz[0] - prev_bid1
            d_ask = asz[0] - prev_ask1
            d_mid = mid_t   - prev_mid_t
            d_spr = spr_t   - prev_spr_t
            prev_bid1 = bsz[0]; prev_ask1 = asz[0]
            prev_mid_t = mid_t; prev_spr_t = spr_t

            features_buf[n_out, :5]   = bp_t
            features_buf[n_out, 5:10] = ap_t
            features_buf[n_out, 10:15]= bsz
            features_buf[n_out, 15:20]= asz
            features_buf[n_out, 20]   = cum_delta
            features_buf[n_out, 21]   = roll_imb
            features_buf[n_out, 22]   = roll_tr
            features_buf[n_out, 23]   = depth_imb
            features_buf[n_out, 24]   = spr_t
            features_buf[n_out, 25]   = d_bid
            features_buf[n_out, 26]   = d_ask
            features_buf[n_out, 27]   = d_mid
            features_buf[n_out, 28]   = d_spr
            features_buf[n_out, 29]   = float(cum_add - cum_cancel)
            n_out += 1

        if n_records % 500_000 < CHUNK_SIZE:
            logger.info(f'[{date_str}] {n_records/1e6:.1f}M records, {n_out}/{N} events ({time.time()-t0:.0f}s)')

        if n_out >= N:
            break

    elapsed = time.time() - t0
    logger.info(f'[{date_str}] Done: {n_records} DBN records → {n_out}/{N} events in {elapsed:.0f}s')

    if n_out < N:
        logger.warning(f'[{date_str}] Short by {N - n_out} events — padding with zeros')
        # features_buf already zero-padded

    _out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        features=features_buf,
        timestamps=ref_ts,
        labels_1s=labels_1s,
        labels_5s=labels_5s,
        labels_10s=labels_10s,
        labels_30s=labels_30s,
        feature_names=np.array(FEATURE_NAMES),
    )
    logger.info(f'[{date_str}] Saved {out_path} shape={features_buf.shape} '
                f'size={out_path.stat().st_size/1e6:.1f}MB')
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
        files = list(raw_dir.glob(f'*{args.date}*.dbn.zst'))
        if not files:
            logger.error(f'No DBN file for {args.date} in {raw_dir}')
            sys.exit(1)
        result = process_day(files[0], out_dir=out_dir)
        sys.exit(0 if result else 1)
    else:
        files = sorted(raw_dir.glob('*.dbn.zst'))
        logger.info(f'Processing {len(files)} days, {args.workers} workers')
        if args.workers == 1:
            for f in files:
                process_day(f, out_dir=out_dir)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = {ex.submit(process_day, f, None, out_dir): f for f in files}
                done = 0
                for fut in as_completed(futs):
                    done += 1
                    f = futs[fut]
                    try:
                        r = fut.result()
                        status = 'OK' if r else 'FAILED'
                    except Exception as e:
                        status = f'ERROR: {e}'
                    logger.info(f'[{done}/{len(files)}] {f.name}: {status}')


if __name__ == '__main__':
    main()
