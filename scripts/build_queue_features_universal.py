#!/usr/bin/env python3
"""
build_queue_features_universal.py — Universal Queue Feature Extractor
=====================================================================

Like build_queue_augmented_features.py but does NOT need prediction files.
Samples the order book at regular intervals (every 1 second by default)
to build queue microstructure features for ANY date with raw DBN data.

This enables queue features for all 238 DBN dates, not just the 41 with
prediction timestamps.

Output: features_YYYYMMDD.parquet per date, joinable to minute bars by
rounding ts_ns to minute.

Rules:
  - HC #493: reuse canonical OrderBook from fifo_market_replay
  - HC #495: as-of only, backward-looking windows
  - HC #0: no expanding windows
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time as time_mod
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
if str(LVL3_ROOT) not in sys.path:
    sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.deep_models.fifo_market_replay import (
    OrderBook, TICK_RAW,
    A_ADD, A_CANCEL, A_MODIFY, A_TRADE, A_FILL, A_RESET,
    S_BID, S_ASK,
    find_dbn_path,
)

OUT_DIR = LVL3_ROOT / 'output' / 'queue_features_universal'
LOG_DIR = LVL3_ROOT / 'logs'

# Sample interval in nanoseconds (1 second = 1e9 ns)
SAMPLE_INTERVAL_NS = 1_000_000_000  # 1 second

# Rolling windows
WIN_1S_NS  = 1_000_000_000
WIN_5S_NS  = 5_000_000_000
WIN_10S_NS = 10_000_000_000

# RTH boundaries (UTC)
RTH_START_HOUR_UTC = 13  # 9:30 AM ET = 13:30 UTC
RTH_START_MIN_UTC = 30
RTH_END_HOUR_UTC = 20    # 4:00 PM ET = 20:00 UTC

log = logging.getLogger('queue_features_universal')


def _bytes(v) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, np.bytes_):
        return bytes(v)
    return bytes([int(v)])


def _evict_old(buf: Deque, cutoff_ns: int) -> None:
    while buf and buf[0][0] < cutoff_ns:
        buf.popleft()


def _windowed_sum(buf: Deque, since_ns: int) -> int:
    s = 0
    for ts, q in buf:
        if ts >= since_ns:
            s += q
    return s


def _level_qty_and_count(book: OrderBook, side_byte: bytes,
                         price_raw: int) -> Tuple[int, int, float]:
    side_dict = book.bids if side_byte == S_BID else book.asks
    pl = side_dict.get(price_raw)
    if pl is None:
        return 0, 0, 0.0
    qtys = list(pl.orders.values())
    n = len(qtys)
    total = int(sum(qtys))
    p50 = float(np.median(qtys)) if n > 0 else 0.0
    return total, n, p50


def snapshot(signal_ts: int, book: OrderBook,
             add_buf: Dict, cancel_buf: Dict, trade_buf: Dict,
             last_bbo_change_ts: Dict, last_add_ts: Dict,
             last_cancel_ts: Dict) -> Optional[Dict]:
    """Build a single as-of feature row at signal_ts."""
    bb = book.best_bid()
    ba = book.best_ask()
    if bb is None or ba is None:
        return None

    bid_qty, bid_n, bid_p50 = _level_qty_and_count(book, S_BID, bb)
    ask_qty, ask_n, ask_p50 = _level_qty_and_count(book, S_ASK, ba)

    bid_age = max(0.0, (signal_ts - last_bbo_change_ts[S_BID]) / 1e9)
    ask_age = max(0.0, (signal_ts - last_bbo_change_ts[S_ASK]) / 1e9)
    bid_t_add = max(0.0, (signal_ts - last_add_ts[S_BID]) / 1e9) if last_add_ts[S_BID] > 0 else float('nan')
    bid_t_cancel = max(0.0, (signal_ts - last_cancel_ts[S_BID]) / 1e9) if last_cancel_ts[S_BID] > 0 else float('nan')
    ask_t_add = max(0.0, (signal_ts - last_add_ts[S_ASK]) / 1e9) if last_add_ts[S_ASK] > 0 else float('nan')
    ask_t_cancel = max(0.0, (signal_ts - last_cancel_ts[S_ASK]) / 1e9) if last_cancel_ts[S_ASK] > 0 else float('nan')

    s_1s = signal_ts - WIN_1S_NS
    s_5s = signal_ts - WIN_5S_NS

    bid_add_1s = _windowed_sum(add_buf[S_BID], s_1s)
    ask_add_1s = _windowed_sum(add_buf[S_ASK], s_1s)
    bid_cancel_1s = _windowed_sum(cancel_buf[S_BID], s_1s)
    ask_cancel_1s = _windowed_sum(cancel_buf[S_ASK], s_1s)
    bid_trade_1s = _windowed_sum(trade_buf[S_BID], s_1s)
    ask_trade_1s = _windowed_sum(trade_buf[S_ASK], s_1s)

    ofi_1s = (bid_add_1s + ask_cancel_1s + ask_trade_1s) - \
             (ask_add_1s + bid_cancel_1s + bid_trade_1s)

    bid_add_5s = _windowed_sum(add_buf[S_BID], s_5s)
    ask_add_5s = _windowed_sum(add_buf[S_ASK], s_5s)
    bid_cancel_5s = _windowed_sum(cancel_buf[S_BID], s_5s)
    ask_cancel_5s = _windowed_sum(cancel_buf[S_ASK], s_5s)
    bid_trade_5s = _windowed_sum(trade_buf[S_BID], s_5s)
    ask_trade_5s = _windowed_sum(trade_buf[S_ASK], s_5s)
    ofi_5s = (bid_add_5s + ask_cancel_5s + ask_trade_5s) - \
             (ask_add_5s + bid_cancel_5s + bid_trade_5s)

    # 10s OFI from full buffer
    bid_add_10s = sum(q for _, q in add_buf[S_BID])
    ask_add_10s = sum(q for _, q in add_buf[S_ASK])
    bid_cancel_10s = sum(q for _, q in cancel_buf[S_BID])
    ask_cancel_10s = sum(q for _, q in cancel_buf[S_ASK])
    bid_trade_10s = sum(q for _, q in trade_buf[S_BID])
    ask_trade_10s = sum(q for _, q in trade_buf[S_ASK])
    ofi_10s = (bid_add_10s + ask_cancel_10s + ask_trade_10s) - \
              (ask_add_10s + bid_cancel_10s + bid_trade_10s)

    tot = bid_qty + ask_qty
    top_imb = ((bid_qty - ask_qty) / tot) if tot > 0 else 0.0
    if tot > 0:
        microprice = (bb * ask_qty + ba * bid_qty) / tot
        mid = 0.5 * (bb + ba)
        microprice_offset_ticks = (microprice - mid) / TICK_RAW
    else:
        microprice_offset_ticks = 0.0

    # Mid price in actual dollars (for price path reconstruction)
    mid_price = 0.5 * (bb + ba) / 1e9  # Databento uses fixed-point 1e-9

    return {
        'ts_ns': signal_ts,
        'mid_price': mid_price,

        'bid_qty_at_touch': int(bid_qty),
        'bid_n_orders': int(bid_n),
        'bid_q_ahead_p50': float(bid_p50),

        'ask_qty_at_touch': int(ask_qty),
        'ask_n_orders': int(ask_n),
        'ask_q_ahead_p50': float(ask_p50),

        'bid_level_age_s': float(bid_age),
        'ask_level_age_s': float(ask_age),
        'bid_time_since_last_add_s': float(bid_t_add),
        'bid_time_since_last_cancel_s': float(bid_t_cancel),
        'ask_time_since_last_add_s': float(ask_t_add),
        'ask_time_since_last_cancel_s': float(ask_t_cancel),

        'bid_add_rate_1s': float(bid_add_1s),
        'ask_add_rate_1s': float(ask_add_1s),
        'bid_cancel_rate_1s': float(bid_cancel_1s),
        'ask_cancel_rate_1s': float(ask_cancel_1s),
        'bid_trade_rate_1s': float(bid_trade_1s),
        'ask_trade_rate_1s': float(ask_trade_1s),

        'ofi_1s': float(ofi_1s),
        'ofi_5s': float(ofi_5s),
        'ofi_10s': float(ofi_10s),

        'top_imbalance': float(top_imb),
        'microprice_offset_ticks': float(microprice_offset_ticks),
    }


def build_features_for_date(date_str: str, out_dir: Path,
                            sample_interval_ns: int = SAMPLE_INTERVAL_NS) -> Optional[Dict]:
    """Walk MBO events for date, emit features at regular intervals."""
    t0 = time_mod.time()

    try:
        import databento as db
    except ImportError:
        raise RuntimeError("databento not installed")

    dbn_path = find_dbn_path(date_str)
    store = db.DBNStore.from_file(dbn_path)
    recs = store.to_ndarray()

    ids, counts = np.unique(recs['instrument_id'], return_counts=True)
    instr_id = int(ids[np.argmax(counts)])
    recs = recs[recs['instrument_id'] == instr_id]
    log.info(f"  {date_str}: {len(recs):,} MBO events, instr={instr_id}")

    ts_arr = recs['ts_recv'].astype(np.int64)
    action_arr = recs['action']
    side_arr = recs['side']
    price_arr = recs['price'].astype(np.int64)
    size_arr = recs['size'].astype(np.int64)
    oid_arr = recs['order_id'].astype(np.int64)
    n_rec = len(recs)

    # Determine RTH boundaries for this date
    from datetime import datetime, timezone, timedelta
    date_dt = datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
    rth_start_ns = int((date_dt + timedelta(hours=RTH_START_HOUR_UTC, minutes=RTH_START_MIN_UTC)).timestamp() * 1e9)
    rth_end_ns = int((date_dt + timedelta(hours=RTH_END_HOUR_UTC)).timestamp() * 1e9)

    book = OrderBook()

    add_buf = {S_BID: deque(), S_ASK: deque()}
    cancel_buf = {S_BID: deque(), S_ASK: deque()}
    trade_buf = {S_BID: deque(), S_ASK: deque()}

    last_bbo_price = {S_BID: None, S_ASK: None}
    last_bbo_change_ts = {S_BID: 0, S_ASK: 0}
    last_add_ts = {S_BID: 0, S_ASK: 0}
    last_cancel_ts = {S_BID: 0, S_ASK: 0}

    rows: List[Dict] = []
    next_sample_ts = rth_start_ns
    last_log_t = time_mod.time()

    for rec_idx in range(n_rec):
        if rec_idx and rec_idx % 2_000_000 == 0:
            now = time_mod.time()
            log.info(f"  {date_str}: event {rec_idx:,}/{n_rec:,} "
                     f"({len(rows)} samples, {now-last_log_t:.1f}s)")
            last_log_t = now

        ts = int(ts_arr[rec_idx])
        action = _bytes(action_arr[rec_idx])
        side = _bytes(side_arr[rec_idx])
        price = int(price_arr[rec_idx])
        qty = int(size_arr[rec_idx])
        oid = int(oid_arr[rec_idx])

        # Snapshot at regular intervals during RTH
        while next_sample_ts <= ts and next_sample_ts < rth_end_ns:
            if next_sample_ts >= rth_start_ns:
                cutoff = next_sample_ts - WIN_10S_NS
                _evict_old(add_buf[S_BID], cutoff)
                _evict_old(add_buf[S_ASK], cutoff)
                _evict_old(cancel_buf[S_BID], cutoff)
                _evict_old(cancel_buf[S_ASK], cutoff)
                _evict_old(trade_buf[S_BID], cutoff)
                _evict_old(trade_buf[S_ASK], cutoff)

                row = snapshot(next_sample_ts, book,
                               add_buf, cancel_buf, trade_buf,
                               last_bbo_change_ts, last_add_ts, last_cancel_ts)
                if row is not None:
                    rows.append(row)

            next_sample_ts += sample_interval_ns

        # Skip events outside RTH
        if ts < rth_start_ns - WIN_10S_NS or ts > rth_end_ns:
            # Still need to build book for pre-RTH events (within 10s buffer)
            pass

        # Update rolling buffers
        if action == A_ADD and side in (S_BID, S_ASK) and qty > 0:
            add_buf[side].append((ts, qty))
            last_add_ts[side] = ts
        elif action == A_CANCEL and side in (S_BID, S_ASK):
            if qty > 0:
                cancel_buf[side].append((ts, qty))
            last_cancel_ts[side] = ts
        elif action in (A_TRADE, A_FILL) and side in (S_BID, S_ASK) and qty > 0:
            trade_buf[side].append((ts, qty))

        # Apply event to book
        if action == A_RESET:
            book.reset()
            last_bbo_price = {S_BID: None, S_ASK: None}
            last_bbo_change_ts = {S_BID: ts, S_ASK: ts}
        elif action == A_ADD:
            if side in (S_BID, S_ASK) and qty > 0:
                book.add(oid, side, price, qty)
        elif action == A_CANCEL:
            book.cancel(oid)
        elif action == A_MODIFY:
            book.modify(oid, qty, price)
        elif action in (A_TRADE, A_FILL):
            book.trade(side, price, qty)

        # Detect BBO change
        bb = book.best_bid()
        ba = book.best_ask()
        if bb != last_bbo_price[S_BID]:
            last_bbo_price[S_BID] = bb
            last_bbo_change_ts[S_BID] = ts
        if ba != last_bbo_price[S_ASK]:
            last_bbo_price[S_ASK] = ba
            last_bbo_change_ts[S_ASK] = ts

    if not rows:
        log.warning(f"  {date_str}: 0 feature rows emitted")
        return None

    df = pd.DataFrame(rows)
    out_path = out_dir / f'features_{date_str}.parquet'
    df.to_parquet(out_path, index=False)
    dt = time_mod.time() - t0

    stats = {
        'date': date_str,
        'n_samples': int(len(df)),
        'wall_sec': round(dt, 2),
        'mid_price_range': [float(df['mid_price'].min()), float(df['mid_price'].max())],
    }
    log.info(f"  {date_str}: wrote {len(df):,} samples in {dt:.1f}s "
             f"mid=[{stats['mid_price_range'][0]:.2f}, {stats['mid_price_range'][1]:.2f}]")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dates', nargs='*', default=None,
                    help='Specific dates to process (YYYYMMDD)')
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--sample-interval', type=float, default=1.0,
                    help='Sample interval in seconds (default 1.0)')
    ap.add_argument('--smoke', action='store_true',
                    help='Process only first date')
    ap.add_argument('--skip-existing', action='store_true', default=True,
                    help='Skip dates that already have output')
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    from datetime import datetime as dt
    log_path = LOG_DIR / f"queue_features_universal_{dt.now().strftime('%Y%m%d_%H%M')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path, mode='a'),
                  logging.StreamHandler(sys.stdout)],
    )

    sample_interval_ns = int(args.sample_interval * 1e9)

    if args.dates:
        dates = list(args.dates)
    else:
        # Discover all dates with raw DBN files
        from alpha_discovery.deep_models.fifo_market_replay import RAW_MBO_DIRS
        dates = set()
        for d in RAW_MBO_DIRS:
            p = Path(d)
            if p.exists():
                for f in p.glob('glbx-mdp3-*.mbo.dbn.zst'):
                    date_str = f.name.split('-')[-1].replace('.mbo.dbn.zst', '')
                    dates.add(date_str)
        dates = sorted(dates)

    # Skip existing
    if args.skip_existing:
        existing = {f.stem.replace('features_', '')
                    for f in args.out_dir.glob('features_*.parquet')}
        dates = [d for d in dates if d not in existing]

    if args.smoke:
        dates = dates[:1]

    log.info(f"Queue features universal extractor: {len(dates)} date(s)")
    log.info(f"Sample interval: {args.sample_interval}s, Output: {args.out_dir}")
    log.info(f"Log: {log_path}")

    summary = {'started': time_mod.strftime('%Y-%m-%dT%H:%M:%S'),
               'n_dates': len(dates), 'per_date': []}

    for i, d in enumerate(dates):
        log.info(f"[{i+1}/{len(dates)}] Processing {d}...")
        try:
            stats = build_features_for_date(d, args.out_dir, sample_interval_ns)
            if stats:
                summary['per_date'].append(stats)
        except Exception as e:
            log.error(f"  {d}: FAILED: {e}")
            summary['per_date'].append({'date': d, 'error': str(e)})

    summary['finished'] = time_mod.strftime('%Y-%m-%dT%H:%M:%S')
    sum_path = args.out_dir / '_summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\nDone. {len(summary['per_date'])} dates processed.")


if __name__ == '__main__':
    sys.exit(main())
