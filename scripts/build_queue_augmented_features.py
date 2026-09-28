#!/usr/bin/env python3
"""
Queue-Augmented Feature Extractor (HC #498 R4 follow-up)
========================================================

Sibling pipeline to build_mbo_walker_fill_labels.py. Where that script emits
fill-OUTCOMES at horizons, this script emits as-of FEATURES (no horizons,
no walk-forward) at each signal-event timestamp. These features are what
queue-position v2 (and any queue-aware fill predictor) needs as inputs.

Output schema (one row per signal event, joinable to mbo_walker_labels by
(event_id, ts_ns)):

  Identity
    event_id, ts_ns

  Queue at touch — BID side
    bid_qty_at_touch, bid_n_orders, bid_q_ahead_if_join_back,
    bid_q_ahead_p50

  Queue at touch — ASK side
    ask_qty_at_touch, ask_n_orders, ask_q_ahead_if_join_back,
    ask_q_ahead_p50

  Time-at-level
    bid_level_age_s, ask_level_age_s,
    bid_time_since_last_add_s, bid_time_since_last_cancel_s,
    ask_time_since_last_add_s, ask_time_since_last_cancel_s

  Flow rates (rolling backward window ending at signal_ts)
    bid_add_rate_1s, ask_add_rate_1s,
    bid_cancel_rate_1s, ask_cancel_rate_1s,
    bid_trade_rate_1s, ask_trade_rate_1s
    ofi_1s, ofi_5s, ofi_10s

  Top-of-book imbalance
    top_imbalance, microprice_offset_ticks

Rules
-----
- HC #493: reuse canonical OrderBook/PriceLevel from
  alpha_discovery.deep_models.fifo_market_replay. NO engine mods.
- HC #495: as-of only. All features computed from book + rolling state
  AT signal_ts. Backward-looking windows only.
- HC #420: user's own quant research codebase. Authorized.

Output: output/queue_augmented_features/features_YYYYMMDD.parquet (+ _summary.json)
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
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

from alpha_discovery.deep_models.fifo_market_replay import (  # noqa: E402
    OrderBook, TICK_RAW,
    A_ADD, A_CANCEL, A_MODIFY, A_TRADE, A_FILL, A_RESET,
    S_BID, S_ASK,
    find_dbn_path,
)

# Reuse signal-loading helpers from the walker label script (HC #493 — share, don't fork)
from build_mbo_walker_fill_labels import (  # noqa: E402
    PRED_DIRS, MBO_EVENT_DIR,
    load_signal_events_for_date, discover_dates,
)

OUT_DIR = LVL3_ROOT / 'output' / 'queue_augmented_features'
LOG_DIR = LVL3_ROOT / 'logs'

# Rolling-window sizes in ns
WIN_1S_NS  = 1_000_000_000
WIN_5S_NS  = 5_000_000_000
WIN_10S_NS = 10_000_000_000

log = logging.getLogger('queue_augmented_features')


def _bytes(v) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, np.bytes_):
        return bytes(v)
    return bytes([int(v)])


def _evict_old(buf: Deque, cutoff_ns: int) -> None:
    """Drop entries older than cutoff_ns from the left of the deque."""
    while buf and buf[0][0] < cutoff_ns:
        buf.popleft()


def _sum_qty(buf: Deque, since_ns: int) -> int:
    """Sum qty for entries with ts >= since_ns. Buffer is time-ordered."""
    s = 0
    for ts, q in buf:
        if ts >= since_ns:
            s += q
    return s


def _bufs_to_arrays(buf: Deque) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a deque of (ts, qty) to parallel numpy arrays."""
    if not buf:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    arr = np.fromiter(buf, dtype=np.dtype([('t', 'i8'), ('q', 'i8')]),
                      count=len(buf))
    return arr['t'].copy(), arr['q'].copy()


def _windowed_sum(ts_arr: np.ndarray, qty_arr: np.ndarray,
                  since_ns: int) -> int:
    """O(log N) lookup, O(window) sum. ts_arr must be sorted ascending."""
    if ts_arr.size == 0:
        return 0
    idx = int(np.searchsorted(ts_arr, since_ns, side='left'))
    if idx >= ts_arr.size:
        return 0
    return int(qty_arr[idx:].sum())


def _level_qty_and_count(book: OrderBook, side_byte: bytes,
                         price_raw: int) -> Tuple[int, int, float]:
    """Return (total_qty_at_level, n_orders_at_level, p50_qty_per_order).

    p50 is the median per-order qty at this level; if level absent → 0.
    """
    side_dict = book.bids if side_byte == S_BID else book.asks
    pl = side_dict.get(price_raw)
    if pl is None:
        return 0, 0, 0.0
    # canonical PriceLevel exposes orders OrderedDict[oid -> qty]
    qtys = list(pl.orders.values())
    n = len(qtys)
    total = int(sum(qtys))
    p50 = float(np.median(qtys)) if n > 0 else 0.0
    return total, n, p50


def build_features_for_date(date_str: str, out_dir: Path) -> Optional[Dict]:
    """Walk MBO events for date, emit one feature row per signal event."""
    t0 = time_mod.time()
    sig = load_signal_events_for_date(date_str)
    if sig is None:
        log.warning(f"  {date_str}: no prediction or MBO data — SKIP")
        return None
    n_signals = sig['n_signals']
    log.info(f"  {date_str}: loaded {n_signals:,} signal events")

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

    ts_arr     = recs['ts_recv'].astype(np.int64)
    action_arr = recs['action']
    side_arr   = recs['side']
    price_arr  = recs['price'].astype(np.int64)
    size_arr   = recs['size'].astype(np.int64)
    oid_arr    = recs['order_id'].astype(np.int64)
    n_rec = len(recs)

    sig_ts   = sig['ts_ns']
    order = np.argsort(sig_ts, kind='stable')
    sig_ts = sig_ts[order]
    sig_side = sig['side'][order]
    sig_pred = sig['preds'][order]
    n_sig = len(sig_ts)

    book = OrderBook()

    # Rolling event buffers (per side, per action). Each is deque of (ts, qty).
    add_buf    = {S_BID: deque(), S_ASK: deque()}
    cancel_buf = {S_BID: deque(), S_ASK: deque()}
    trade_buf  = {S_BID: deque(), S_ASK: deque()}

    # Time-at-level tracking per side
    last_bbo_price = {S_BID: None, S_ASK: None}
    last_bbo_change_ts = {S_BID: 0, S_ASK: 0}
    last_add_ts    = {S_BID: 0, S_ASK: 0}
    last_cancel_ts = {S_BID: 0, S_ASK: 0}

    rows: List[Dict] = []
    sig_idx = 0
    event_id_ctr = 0
    last_log_t = time_mod.time()
    log.info(f"  {date_str}: walking {n_rec:,} events, {n_sig:,} signals")

    for rec_idx in range(n_rec):
        if rec_idx and rec_idx % 1_000_000 == 0:
            now = time_mod.time()
            log.info(f"  {date_str}: event {rec_idx:,}/{n_rec:,} "
                     f"(signals processed={sig_idx}/{n_sig}, "
                     f"chunk_dt={now-last_log_t:.1f}s)")
            last_log_t = now
        ts = int(ts_arr[rec_idx])
        action = _bytes(action_arr[rec_idx])
        side   = _bytes(side_arr[rec_idx])
        price  = int(price_arr[rec_idx])
        qty    = int(size_arr[rec_idx])
        oid    = int(oid_arr[rec_idx])

        # 1) Snapshot any pending signals (ts <= current event ts) BEFORE
        #    this event mutates the book — so feature values are as-of the
        #    last completed event, which is the as-of-only contract.
        #    Evict stale rolling-window entries ONCE per signal (perf).
        while sig_idx < n_sig and sig_ts[sig_idx] <= ts:
            stime = int(sig_ts[sig_idx])
            cutoff = stime - WIN_10S_NS
            _evict_old(add_buf[S_BID],    cutoff)
            _evict_old(add_buf[S_ASK],    cutoff)
            _evict_old(cancel_buf[S_BID], cutoff)
            _evict_old(cancel_buf[S_ASK], cutoff)
            _evict_old(trade_buf[S_BID],  cutoff)
            _evict_old(trade_buf[S_ASK],  cutoff)
            row = _snapshot(stime, int(sig_side[sig_idx]),
                            sig_pred[sig_idx],
                            book, add_buf, cancel_buf, trade_buf,
                            last_bbo_change_ts, last_add_ts, last_cancel_ts,
                            event_id=event_id_ctr)
            if row is not None:
                rows.append(row)
                event_id_ctr += 1
            sig_idx += 1

        # 2) Update rolling buffers (no per-event eviction — done at signal time)
        if action == A_ADD and side in (S_BID, S_ASK) and qty > 0:
            add_buf[side].append((ts, qty))
            last_add_ts[side] = ts
        elif action == A_CANCEL and side in (S_BID, S_ASK):
            if qty > 0:
                cancel_buf[side].append((ts, qty))
            last_cancel_ts[side] = ts
        elif action in (A_TRADE, A_FILL) and side in (S_BID, S_ASK) and qty > 0:
            trade_buf[side].append((ts, qty))

        # 3) Apply event to canonical book
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

        # 4) Detect BBO change after this event to update level-age clocks
        bb = book.best_bid()
        ba = book.best_ask()
        if bb != last_bbo_price[S_BID]:
            last_bbo_price[S_BID] = bb
            last_bbo_change_ts[S_BID] = ts
        if ba != last_bbo_price[S_ASK]:
            last_bbo_price[S_ASK] = ba
            last_bbo_change_ts[S_ASK] = ts

    # Flush any tail signals after last event — snapshot as-of final state
    while sig_idx < n_sig:
        stime = int(sig_ts[sig_idx])
        row = _snapshot(stime, int(sig_side[sig_idx]),
                        sig_pred[sig_idx],
                        book, add_buf, cancel_buf, trade_buf,
                        last_bbo_change_ts, last_add_ts, last_cancel_ts,
                        event_id=event_id_ctr)
        if row is not None:
            rows.append(row)
            event_id_ctr += 1
        sig_idx += 1

    if not rows:
        log.warning(f"  {date_str}: 0 feature rows emitted")
        return None

    df = pd.DataFrame(rows)
    out_path = out_dir / f'features_{date_str}.parquet'
    df.to_parquet(out_path, index=False)
    dt = time_mod.time() - t0

    # Per-date summary stats
    stats = {
        'date':           date_str,
        'n_signals_input': int(sig['n_signals']),
        'n_feature_rows':  int(len(df)),
        'wall_sec':        round(dt, 2),
    }
    # Sanity ranges on a few key features
    for col in ('bid_qty_at_touch', 'ask_qty_at_touch',
                'top_imbalance', 'ofi_1s', 'bid_level_age_s',
                'microprice_offset_ticks'):
        if col in df.columns:
            stats[f'{col}_mean']   = float(df[col].mean())
            stats[f'{col}_median'] = float(df[col].median())
            stats[f'{col}_nan_pct'] = float(df[col].isna().mean())

    log.info(f"  {date_str}: wrote {len(df):,} feature rows in {dt:.1f}s "
             f"| bid_qty_p50={stats.get('bid_qty_at_touch_median',0):.0f} "
             f"top_imb_mean={stats.get('top_imbalance_mean',0):+.3f}")
    return stats


def _snapshot(signal_ts: int, side_int: int, preds: np.ndarray,
              book: OrderBook,
              add_buf: Dict, cancel_buf: Dict, trade_buf: Dict,
              last_bbo_change_ts: Dict, last_add_ts: Dict,
              last_cancel_ts: Dict, event_id: int) -> Optional[Dict]:
    """Build a single as-of feature row at signal_ts. Returns None if no book."""
    bb = book.best_bid()
    ba = book.best_ask()
    if bb is None or ba is None:
        return None

    # Queue at touch — both sides
    bid_qty, bid_n, bid_p50 = _level_qty_and_count(book, S_BID, bb)
    ask_qty, ask_n, ask_p50 = _level_qty_and_count(book, S_ASK, ba)

    # Time-at-level (seconds since last BBO change on each side)
    bid_age = max(0.0, (signal_ts - last_bbo_change_ts[S_BID]) / 1e9)
    ask_age = max(0.0, (signal_ts - last_bbo_change_ts[S_ASK]) / 1e9)
    bid_t_add    = max(0.0, (signal_ts - last_add_ts[S_BID]) / 1e9) \
                   if last_add_ts[S_BID] > 0 else float('nan')
    bid_t_cancel = max(0.0, (signal_ts - last_cancel_ts[S_BID]) / 1e9) \
                   if last_cancel_ts[S_BID] > 0 else float('nan')
    ask_t_add    = max(0.0, (signal_ts - last_add_ts[S_ASK]) / 1e9) \
                   if last_add_ts[S_ASK] > 0 else float('nan')
    ask_t_cancel = max(0.0, (signal_ts - last_cancel_ts[S_ASK]) / 1e9) \
                   if last_cancel_ts[S_ASK] > 0 else float('nan')

    # Flow rates over rolling windows — vectorize:
    # 1) Convert each deque to (ts, qty) numpy arrays ONCE per signal
    # 2) searchsorted to locate window start, slice-sum
    s_1s  = signal_ts - WIN_1S_NS
    s_5s  = signal_ts - WIN_5S_NS
    s_10s = signal_ts - WIN_10S_NS

    add_bid_t, add_bid_q = _bufs_to_arrays(add_buf[S_BID])
    add_ask_t, add_ask_q = _bufs_to_arrays(add_buf[S_ASK])
    can_bid_t, can_bid_q = _bufs_to_arrays(cancel_buf[S_BID])
    can_ask_t, can_ask_q = _bufs_to_arrays(cancel_buf[S_ASK])
    trd_bid_t, trd_bid_q = _bufs_to_arrays(trade_buf[S_BID])
    trd_ask_t, trd_ask_q = _bufs_to_arrays(trade_buf[S_ASK])

    bid_add_1s    = _windowed_sum(add_bid_t, add_bid_q, s_1s)
    ask_add_1s    = _windowed_sum(add_ask_t, add_ask_q, s_1s)
    bid_cancel_1s = _windowed_sum(can_bid_t, can_bid_q, s_1s)
    ask_cancel_1s = _windowed_sum(can_ask_t, can_ask_q, s_1s)
    bid_trade_1s  = _windowed_sum(trd_bid_t, trd_bid_q, s_1s)
    ask_trade_1s  = _windowed_sum(trd_ask_t, trd_ask_q, s_1s)

    # OFI = (bid_adds + ask_cancels + ask_trades) - (ask_adds + bid_cancels + bid_trades)
    # Positive OFI ⇒ bid pressure; negative ⇒ ask pressure.
    ofi_1s = (bid_add_1s + ask_cancel_1s + ask_trade_1s) - \
             (ask_add_1s + bid_cancel_1s + bid_trade_1s)

    bid_add_5s    = _windowed_sum(add_bid_t, add_bid_q, s_5s)
    ask_add_5s    = _windowed_sum(add_ask_t, add_ask_q, s_5s)
    bid_cancel_5s = _windowed_sum(can_bid_t, can_bid_q, s_5s)
    ask_cancel_5s = _windowed_sum(can_ask_t, can_ask_q, s_5s)
    bid_trade_5s  = _windowed_sum(trd_bid_t, trd_bid_q, s_5s)
    ask_trade_5s  = _windowed_sum(trd_ask_t, trd_ask_q, s_5s)
    ofi_5s = (bid_add_5s + ask_cancel_5s + ask_trade_5s) - \
             (ask_add_5s + bid_cancel_5s + bid_trade_5s)

    # 10s window = whole buffer (already evicted to 10s at signal time)
    bid_add_10s    = int(add_bid_q.sum())
    ask_add_10s    = int(add_ask_q.sum())
    bid_cancel_10s = int(can_bid_q.sum())
    ask_cancel_10s = int(can_ask_q.sum())
    bid_trade_10s  = int(trd_bid_q.sum())
    ask_trade_10s  = int(trd_ask_q.sum())
    ofi_10s = (bid_add_10s + ask_cancel_10s + ask_trade_10s) - \
              (ask_add_10s + bid_cancel_10s + bid_trade_10s)

    # Top-of-book imbalance + microprice
    tot = bid_qty + ask_qty
    top_imb = ((bid_qty - ask_qty) / tot) if tot > 0 else 0.0
    if tot > 0:
        microprice = (bb * ask_qty + ba * bid_qty) / tot  # raw price units
        mid = 0.5 * (bb + ba)
        microprice_offset_ticks = (microprice - mid) / TICK_RAW
    else:
        microprice_offset_ticks = 0.0

    return {
        'event_id':                   event_id,
        'ts_ns':                      signal_ts,
        'side':                       side_int,
        'pred_1s':                    float(preds[0]),
        'pred_5s':                    float(preds[1]),
        'pred_10s':                   float(preds[2]),

        'bid_qty_at_touch':           int(bid_qty),
        'bid_n_orders':               int(bid_n),
        'bid_q_ahead_if_join_back':   int(bid_qty),
        'bid_q_ahead_p50':            float(bid_p50),

        'ask_qty_at_touch':           int(ask_qty),
        'ask_n_orders':               int(ask_n),
        'ask_q_ahead_if_join_back':   int(ask_qty),
        'ask_q_ahead_p50':            float(ask_p50),

        'bid_level_age_s':            float(bid_age),
        'ask_level_age_s':            float(ask_age),
        'bid_time_since_last_add_s':  float(bid_t_add),
        'bid_time_since_last_cancel_s': float(bid_t_cancel),
        'ask_time_since_last_add_s':  float(ask_t_add),
        'ask_time_since_last_cancel_s': float(ask_t_cancel),

        'bid_add_rate_1s':            float(bid_add_1s),
        'ask_add_rate_1s':            float(ask_add_1s),
        'bid_cancel_rate_1s':         float(bid_cancel_1s),
        'ask_cancel_rate_1s':         float(ask_cancel_1s),
        'bid_trade_rate_1s':          float(bid_trade_1s),
        'ask_trade_rate_1s':          float(ask_trade_1s),

        'ofi_1s':                     float(ofi_1s),
        'ofi_5s':                     float(ofi_5s),
        'ofi_10s':                    float(ofi_10s),

        'top_imbalance':              float(top_imb),
        'microprice_offset_ticks':    float(microprice_offset_ticks),
    }


def _worker(args):
    date_str, out_dir, log_path = args
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(process)d] %(message)s',
        handlers=[logging.FileHandler(log_path, mode='a'),
                  logging.StreamHandler(sys.stdout)],
    )
    try:
        return build_features_for_date(date_str, out_dir)
    except Exception as e:
        log.exception(f"[{date_str}] FAILED: {e}")
        return {'date': date_str, 'error': str(e)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dates', nargs='*', default=None)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--log-path', type=Path, default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--n-procs', type=int, default=8)
    # Restrict to the 41 dates that already have walker labels — these are
    # the only dates queue-position v2 can train on (need joinable labels).
    ap.add_argument('--walker-labels-dir', type=Path,
                    default=LVL3_ROOT / 'output' / 'mbo_walker_labels')
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.log_path is None:
        from datetime import datetime
        args.log_path = LOG_DIR / f"queue_augmented_features_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(args.log_path, mode='a'),
                  logging.StreamHandler(sys.stdout)],
    )

    if args.dates:
        dates = list(args.dates)
    else:
        # Intersect discoverable dates with the 41 walker-label dates so we
        # only emit features that are joinable to labels.
        all_dates = set(discover_dates())
        if args.walker_labels_dir.exists():
            label_dates = {f.stem.replace('labels_', '')
                           for f in args.walker_labels_dir.glob('labels_*.parquet')}
            dates = sorted(all_dates & label_dates)
            log.info(f"Intersected with {len(label_dates)} walker-label dates "
                     f"-> {len(dates)} target dates")
        else:
            dates = sorted(all_dates)
    if args.smoke:
        dates = dates[:1]

    log.info(f"Queue-augmented feature extractor: {len(dates)} date(s) -> {args.out_dir}")
    log.info(f"Log: {args.log_path}")

    summary = {'started_utc': time_mod.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                time_mod.gmtime()),
               'n_dates': len(dates),
               'per_date': []}

    if args.n_procs <= 1 or len(dates) == 1:
        for d in dates:
            res = build_features_for_date(d, args.out_dir)
            if res is not None:
                summary['per_date'].append(res)
    else:
        with mp.Pool(args.n_procs) as pool:
            for res in pool.imap_unordered(
                _worker,
                [(d, args.out_dir, str(args.log_path)) for d in dates],
                chunksize=1,
            ):
                if res is not None:
                    summary['per_date'].append(res)

    summary['finished_utc'] = time_mod.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                time_mod.gmtime())
    if summary['per_date']:
        for col in ('bid_qty_at_touch_mean', 'top_imbalance_mean',
                    'ofi_1s_mean', 'bid_level_age_s_mean'):
            vals = [r[col] for r in summary['per_date'] if col in r]
            if vals:
                summary[f'global_{col}'] = float(np.mean(vals))

    sum_path = args.out_dir / '_summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary written to {sum_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
