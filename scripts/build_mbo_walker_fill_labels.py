#!/usr/bin/env python3
"""
MBO-Walker Fill Label Generator (HC #498 follow-up)
====================================================

Why this exists
---------------
- v1 fill-prob AUC=0.86 was self-distillation: the label was a closed-form
  function of the input features.
- v2/v2.1/v2.2 used (mae_1s >= 1.0 tick) — honest but a poor proxy for the
  thing we trade (queue-aware passive fill).
- Existing `surviving_canonical_fifo_fills.parquet` is FILLED-ONLY (100%
  positive rate). Training a fill classifier on it = self-distillation trap.

This script emits an HONEST queue-aware fill label per signal event:
  For each prediction emit time t:
    1. Post a hypothetical passive limit at the touch (best_bid for long,
       best_ask for short signals).
    2. Land at the BACK of the queue at that price level.
    3. Walk the raw MBO event stream forward for h seconds:
       - Trades at our level drain queue front. Our queue_position
         decrements by the consumed quantity ahead of us.
       - Cancels at our level: if the cancel is ahead of us (queued earlier),
         our queue_position decrements by the cancel size. Because we
         actually inserted our sim order into the canonical OrderBook, the
         PriceLevel's qty_ahead_of(sim_oid) method gives this correctly.
       - New limits at our level land BEHIND us (no effect).
       - Modifies are handled by canonical OrderBook.modify().
    4. If our queue position reaches 0 within h seconds -> filled_h = 1.
    5. Otherwise -> filled_h = 0, queue_rank_at_h = remaining/initial.

Crucially, the label is derived strictly from observed MBO event drains
through the canonical FIFO PriceLevel mechanics. NO closed-form-of-features.

Reuse (HC #493): we import OrderBook, PriceLevel, TICK_RAW, S_BID, S_ASK,
A_* action bytes from `alpha_discovery.deep_models.fifo_market_replay`.

Authorization: HC #420 — user's own legitimate quant research codebase.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time as time_mod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Canonical FIFO primitives (HC #493 — reuse, do not reinvent)
# ──────────────────────────────────────────────────────────────────────────────
LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
if str(LVL3_ROOT) not in sys.path:
    sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.deep_models.fifo_market_replay import (  # noqa: E402
    OrderBook, PriceLevel, TICK_RAW,
    A_ADD, A_CANCEL, A_MODIFY, A_TRADE, A_FILL, A_RESET,
    S_BID, S_ASK, S_NONE,
    find_dbn_path,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PRED_DIRS = [
    LVL3_ROOT / 'output' / 'cnn_mamba_v2_bulk_oot_v2',
    LVL3_ROOT / 'output' / 'cnn_mamba_v2_all_oot',
    LVL3_ROOT / 'output' / 'cnn_mamba_v2_bulk_oot',
]
MBO_EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
OUT_DIR       = LVL3_ROOT / 'output' / 'mbo_walker_labels'
LOG_DIR       = LVL3_ROOT / 'logs'

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SIM_OID_BASE = 9_000_000_000_000   # well above any real order_id range

HORIZONS_S = (1.0, 5.0, 10.0)
HORIZON_NS = tuple(int(h * 1e9) for h in HORIZONS_S)

# Gate threshold for what counts as a "signal event" — we emit labels for
# events whose |prediction| crosses this gate at ANY of the three horizons.
# Mirrors the v2.1 trainer convention: anything below this is not a tradeable
# signal so we don't waste cycles labelling it.
SIGNAL_GATE = 0.05

log = logging.getLogger('mbo_walker_labels')


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _bytes(v) -> bytes:
    """Coerce DBN action/side codes to single-byte bytestring."""
    if isinstance(v, bytes):
        return v
    if isinstance(v, np.bytes_):
        return bytes(v)
    return bytes([int(v)])


def find_pred_file(date_str: str) -> Optional[Path]:
    """Search known prediction directories for a date file. Skips _stale_."""
    for d in PRED_DIRS:
        for pat in (f'{date_str}_predictions.npz', f'fold_*_{date_str}*.npz'):
            for f in d.glob(pat):
                if '_stale_' in f.name:
                    continue
                return f
        # Also try fold_XX_oot_predictions where oot_files names the date
        for f in d.glob('fold_*_oot_predictions.npz'):
            try:
                z = np.load(f, allow_pickle=True)
                df = str(z.get('date', ''))
                if df == date_str:
                    return f
            except Exception:
                continue
    return None


def load_signal_events_for_date(date_str: str) -> Optional[Dict]:
    """Load CNN-Mamba prediction npz + matching MBO timestamps.

    Returns dict with:
      ts_ns         : (N,) int64       — signal timestamps
      side          : (N,) int8        — +1 long, -1 short
      strength      : (N,) float32     — max |pred| across horizons
      pred_1s_5s_10s: (N,3) float32    — raw signed predictions
    Or None if data unavailable for this date.
    """
    pred_path = find_pred_file(date_str)
    if pred_path is None:
        return None
    mbo_path = MBO_EVENT_DIR / f'{date_str}_mbo_events.npz'
    if not mbo_path.exists():
        return None

    pd_z = np.load(pred_path, allow_pickle=True)
    preds = pd_z['predictions']                  # (N, 3)
    n_preds = preds.shape[0]
    win = int(pd_z.get('window_size', 1000))
    stride = int(pd_z.get('stride', 250))

    mbo_z = np.load(mbo_path, allow_pickle=True)
    ts_arr = mbo_z['timestamps']
    n_events = len(ts_arr)

    # Map prediction i -> last event in its window
    idx = np.minimum(np.arange(n_preds) * stride + win - 1, n_events - 1)
    pred_ts = ts_arr[idx].astype(np.int64)

    strength = np.abs(preds).max(axis=1)
    # Direction: use the horizon with largest |pred| sign for the signal side.
    h_argmax = np.argmax(np.abs(preds), axis=1)
    signed = preds[np.arange(n_preds), h_argmax]
    side = np.where(signed > 0, 1, -1).astype(np.int8)

    # Gate: drop events below the signal threshold to keep file sizes sane.
    # This matches the v2.1 trainer convention — events below the gate are
    # not tradeable signals.
    mask = strength >= SIGNAL_GATE
    if not mask.any():
        return None

    return {
        'date':          date_str,
        'ts_ns':         pred_ts[mask],
        'side':          side[mask],
        'strength':      strength[mask].astype(np.float32),
        'preds':         preds[mask].astype(np.float32),  # (n_kept, 3)
        'n_signals':     int(mask.sum()),
        'n_predictions': n_preds,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core: build labels for one date
# ──────────────────────────────────────────────────────────────────────────────
def build_labels_for_date(date_str: str, out_dir: Path,
                          horizon_ns_list: Tuple[int, ...] = HORIZON_NS,
                          ) -> Optional[Dict]:
    """Walk MBO events for `date_str`, emit fill labels for each signal event.

    The output parquet has one row per signal event, with queue_rank and
    filled flags at each horizon in horizon_ns_list.
    """
    t0 = time_mod.time()
    sig = load_signal_events_for_date(date_str)
    if sig is None:
        log.warning(f"  {date_str}: no prediction or MBO data — SKIP")
        return None
    n_signals = sig['n_signals']
    log.info(f"  {date_str}: loaded {n_signals:,} signal events "
             f"(of {sig['n_predictions']:,} preds)")

    # Load raw DBN
    try:
        import databento as db
    except ImportError:
        raise RuntimeError("databento not installed")
    dbn_path = find_dbn_path(date_str)
    store = db.DBNStore.from_file(dbn_path)
    recs = store.to_ndarray()
    # Auto-detect instrument
    ids, counts = np.unique(recs['instrument_id'], return_counts=True)
    instr_id = int(ids[np.argmax(counts)])
    recs = recs[recs['instrument_id'] == instr_id]
    log.info(f"  {date_str}: {len(recs):,} MBO events, instr={instr_id}")

    # Pre-extract columns to avoid repeated np structured access
    ts_arr     = recs['ts_recv'].astype(np.int64)
    action_arr = recs['action']
    side_arr   = recs['side']
    price_arr  = recs['price'].astype(np.int64)
    size_arr   = recs['size'].astype(np.int64)
    oid_arr    = recs['order_id'].astype(np.int64)
    n_rec = len(recs)

    # Sort signals by ts (defensive)
    sig_ts   = sig['ts_ns']
    sig_side = sig['side']
    sig_pred = sig['preds']
    order = np.argsort(sig_ts, kind='stable')
    sig_ts   = sig_ts[order]
    sig_side = sig_side[order]
    sig_pred = sig_pred[order]

    book = OrderBook()
    # active_orders maps sim_oid -> dict tracking horizon-checkpoint state.
    # Each active order has:
    #   side_byte:   S_BID or S_ASK
    #   price_raw:   int
    #   initial_qty_ahead: int        (queue depth at submit, in front of us)
    #   ts_ns_submit: int             (when we submitted)
    #   deadline_ns:  int             (signal_ts + max(horizon_ns_list))
    #   horizon_q_rank: dict h_ns -> float or None (filled queue_rank)
    #   filled_h: dict h_ns -> 0/1
    #   time_to_fill: dict h_ns -> float or NaN (seconds from signal)
    #   filled_event_idx: optional rec idx when filled (-1 if not yet)
    active: Dict[int, Dict] = {}
    sim_oid_ctr = SIM_OID_BASE

    # Output rows accumulator — list of dicts, materialized at end.
    rows: List[Dict] = []
    # For each signal that goes pending, store a row template; we'll fill in
    # the horizon outputs as the walker progresses.
    row_idx_by_oid: Dict[int, int] = {}

    max_h_ns = max(horizon_ns_list)

    sig_idx = 0
    n_sig = len(sig_ts)

    # Main MBO walk
    for rec_idx in range(n_rec):
        ts = int(ts_arr[rec_idx])
        action = _bytes(action_arr[rec_idx])
        side   = _bytes(side_arr[rec_idx])
        price  = int(price_arr[rec_idx])
        qty    = int(size_arr[rec_idx])
        oid    = int(oid_arr[rec_idx])

        # 1) Apply event to canonical book
        consumed_oids: List[int] = []
        if action == A_RESET:
            book.reset()
            # Any of our pending sims are gone — treat as cancelled fills at
            # queue_rank = current rank. We'll mark them as not-filled at all
            # remaining horizons. (Rare path; safe fallback.)
            for sim_oid, st in list(active.items()):
                _finalize_unfilled_pending(st, rows, row_idx_by_oid[sim_oid],
                                           horizon_ns_list)
                active.pop(sim_oid)
                row_idx_by_oid.pop(sim_oid, None)
        elif action == A_ADD:
            if side in (S_BID, S_ASK) and qty > 0:
                book.add(oid, side, price, qty)
        elif action == A_CANCEL:
            book.cancel(oid)
        elif action == A_MODIFY:
            book.modify(oid, qty, price)
        elif action in (A_TRADE, A_FILL):
            consumed_oids = book.trade(side, price, qty)

        # 2) Process any signals with ts <= this event ts
        while sig_idx < n_sig and sig_ts[sig_idx] <= ts:
            _submit_signal(
                signal_ts=int(sig_ts[sig_idx]),
                side_int=int(sig_side[sig_idx]),
                preds=sig_pred[sig_idx],
                book=book,
                active=active,
                row_idx_by_oid=row_idx_by_oid,
                rows=rows,
                sim_oid_holder=[sim_oid_ctr],
                max_h_ns=max_h_ns,
                horizon_ns_list=horizon_ns_list,
                cur_ts=ts,
            )
            # bump counter
            sim_oid_ctr = max(active) + 1 if active else sim_oid_ctr + 1
            sig_idx += 1

        # 3) Sweep horizon checkpoints FIRST (before processing fills) so
        # that any horizon that expired strictly before this event's fill
        # gets its queue_rank snapshot from the pre-trade book state.
        # Order matters: book mutation in step 1 already happened. We must
        # snapshot queue_rank with the order STILL in the book — i.e. before
        # _on_filled removes it. Therefore checkpoint -> then fill.
        if active:
            for sim_oid, st in active.items():
                for h_ns in horizon_ns_list:
                    h_key = int(h_ns)
                    if st['filled_h'][h_key] == -1:
                        if ts > st['signal_ts'] + h_ns:
                            # Strict '>' so that if fill happens exactly at
                            # signal_ts+h, _on_filled treats it as filled
                            # within the horizon (inclusive bound).
                            pl = (book.bids if st['side_byte'] == S_BID
                                  else book.asks).get(st['price_raw'])
                            if pl is None:
                                qa = st['initial_qty_ahead']
                                qrank = 1.0 if qa > 0 else 0.0
                            else:
                                qa_now = pl.qty_ahead_of(sim_oid)
                                init = st['initial_qty_ahead']
                                qrank = (qa_now / init) if init > 0 else 0.0
                            st['queue_rank'][h_key] = float(qrank)
                            st['filled_h'][h_key]   = 0
                            st['time_to_fill'][h_key] = float('nan')

        # 4) Now process any fills from this event's trade action
        if consumed_oids:
            for o in consumed_oids:
                if o in active:
                    st = active[o]
                    _on_filled(st, rows, row_idx_by_oid[o],
                               ts_fill=ts, horizon_ns_list=horizon_ns_list)
                    active.pop(o)
                    row_idx_by_oid.pop(o, None)

        # 5) Free any orders for which all horizons are decided
        if active:
            expired_oids = []
            for sim_oid, st in active.items():
                if all(st['filled_h'][int(h)] != -1 for h in horizon_ns_list):
                    expired_oids.append(sim_oid)
            for sim_oid in expired_oids:
                st = active.pop(sim_oid)
                _flush_row(st, rows, row_idx_by_oid[sim_oid], horizon_ns_list)
                row_idx_by_oid.pop(sim_oid, None)
                book.cancel(sim_oid)

    # End-of-day: any remaining active orders -> mark unfilled at their
    # remaining horizons. (Final queue rank from book state.)
    for sim_oid, st in list(active.items()):
        _finalize_unfilled_pending(st, rows, row_idx_by_oid[sim_oid],
                                   horizon_ns_list, book=book,
                                   sim_oid=sim_oid)
        active.pop(sim_oid)
        row_idx_by_oid.pop(sim_oid, None)
        book.cancel(sim_oid)

    # Any signals that arrived after the last event timestamp — give them
    # no_book rows (skip; effectively unlabelled).
    while sig_idx < n_sig:
        sig_idx += 1   # drop tail signals (no book to land in)

    if not rows:
        log.warning(f"  {date_str}: 0 label rows emitted")
        return None

    df = pd.DataFrame(rows)
    out_path = out_dir / f'labels_{date_str}.parquet'
    df.to_parquet(out_path, index=False)
    dt = time_mod.time() - t0

    # Per-date stats
    stats = {
        'date': date_str,
        'n_signals_input': int(sig['n_signals']),
        'n_label_rows':    int(len(df)),
        'wall_sec':        round(dt, 2),
    }
    for h_s in HORIZONS_S:
        h_key = int(h_s * 1e9)
        col_f = f'filled_{int(h_s)}s'
        col_r = f'queue_rank_at_{int(h_s)}s'
        if col_f in df.columns:
            stats[f'fill_rate_{int(h_s)}s'] = float(df[col_f].mean())
        if col_r in df.columns:
            non_filled = df[df[col_f] == 0][col_r].dropna()
            if len(non_filled) > 0:
                stats[f'queue_rank_p50_{int(h_s)}s'] = float(non_filled.median())
                stats[f'queue_rank_p25_{int(h_s)}s'] = float(non_filled.quantile(0.25))
                stats[f'queue_rank_p75_{int(h_s)}s'] = float(non_filled.quantile(0.75))

    log.info(f"  {date_str}: wrote {len(df):,} labels in {dt:.1f}s "
             f"| fill_1s={stats.get('fill_rate_1s',0):.3f} "
             f"fill_5s={stats.get('fill_rate_5s',0):.3f} "
             f"fill_10s={stats.get('fill_rate_10s',0):.3f}")
    return stats


def _submit_signal(signal_ts: int, side_int: int, preds: np.ndarray,
                   book: OrderBook, active: Dict, row_idx_by_oid: Dict,
                   rows: List[Dict], sim_oid_holder: List[int],
                   max_h_ns: int, horizon_ns_list: Tuple[int, ...],
                   cur_ts: int) -> None:
    """Insert our sim limit at the touch on the appropriate side."""
    bb = book.best_bid()
    ba = book.best_ask()
    if bb is None or ba is None:
        return   # no book — skip
    if side_int > 0:
        side_byte = S_BID
        price_raw = bb
    else:
        side_byte = S_ASK
        price_raw = ba

    # Queue depth at touch BEFORE we add ourselves
    qty_ahead = book.qty_at(side_byte, price_raw)

    # Pick a unique sim oid not in use
    if active:
        next_oid = max(max(active.keys()), sim_oid_holder[0]) + 1
    else:
        next_oid = sim_oid_holder[0] + 1
    sim_oid_holder[0] = next_oid
    sim_oid = next_oid

    # Insert at end of queue (canonical PriceLevel uses OrderedDict insertion
    # order, so add() appends).
    book.add(sim_oid, side_byte, price_raw, 1)

    state = {
        'event_id':           len(rows),
        'signal_ts':          signal_ts,
        'side_int':           side_int,
        'side_byte':          side_byte,
        'price_raw':          price_raw,
        'price':              price_raw / TICK_RAW,
        'initial_qty_ahead':  qty_ahead,
        'pred_1s':            float(preds[0]),
        'pred_5s':            float(preds[1]),
        'pred_10s':           float(preds[2]),
        'queue_rank':         {int(h): None for h in horizon_ns_list},
        'filled_h':           {int(h): -1 for h in horizon_ns_list},
        'time_to_fill':       {int(h): float('nan') for h in horizon_ns_list},
    }
    active[sim_oid] = state
    # Reserve row slot — flushed when all horizons decided
    rows.append({})  # placeholder
    row_idx_by_oid[sim_oid] = len(rows) - 1


def _on_filled(st: Dict, rows: List[Dict], row_idx: int,
               ts_fill: int, horizon_ns_list: Tuple[int, ...]) -> None:
    """Fill happened — record fill at every horizon >= time_to_fill."""
    dt_ns = ts_fill - st['signal_ts']
    dt_s = dt_ns / 1e9
    for h_ns in horizon_ns_list:
        h_key = int(h_ns)
        if st['filled_h'][h_key] != -1:
            continue   # already decided by checkpoint sweep
        if dt_ns <= h_ns:
            st['filled_h'][h_key] = 1
            st['queue_rank'][h_key] = 0.0
            st['time_to_fill'][h_key] = float(dt_s)
        else:
            # Fill came after this horizon AND the checkpoint sweep didn't
            # decide it (e.g. signals processed and consumed in the same
            # event tick — boundary case). Mark unfilled with rank=NaN to
            # avoid biasing queue_rank distributions.
            st['filled_h'][h_key] = 0
            st['queue_rank'][h_key] = float('nan')
            st['time_to_fill'][h_key] = float('nan')
    _flush_row(st, rows, row_idx, horizon_ns_list)


def _finalize_unfilled_pending(st: Dict, rows: List[Dict], row_idx: int,
                               horizon_ns_list: Tuple[int, ...],
                               book: Optional[OrderBook] = None,
                               sim_oid: Optional[int] = None) -> None:
    """End-of-day or book-reset cleanup: mark remaining horizons unfilled."""
    for h_ns in horizon_ns_list:
        h_key = int(h_ns)
        if st['filled_h'][h_key] != -1:
            continue
        if book is not None and sim_oid is not None:
            pl = (book.bids if st['side_byte'] == S_BID
                  else book.asks).get(st['price_raw'])
            if pl is None:
                qrank = 1.0 if st['initial_qty_ahead'] > 0 else 0.0
            else:
                qa_now = pl.qty_ahead_of(sim_oid)
                init = st['initial_qty_ahead']
                qrank = (qa_now / init) if init > 0 else 0.0
        else:
            qrank = 1.0
        st['queue_rank'][h_key] = float(qrank)
        st['filled_h'][h_key] = 0
        st['time_to_fill'][h_key] = float('nan')
    _flush_row(st, rows, row_idx, horizon_ns_list)


def _flush_row(st: Dict, rows: List[Dict], row_idx: int,
               horizon_ns_list: Tuple[int, ...]) -> None:
    """Materialize a single row at row_idx."""
    row = {
        'event_id':              st['event_id'],
        'ts_ns':                 int(st['signal_ts']),
        'side':                  int(st['side_int']),
        'price':                 float(st['price']),
        'queue_depth_at_touch':  int(st['initial_qty_ahead']),
        'pred_1s':               float(st['pred_1s']),
        'pred_5s':               float(st['pred_5s']),
        'pred_10s':              float(st['pred_10s']),
    }
    for h_ns in horizon_ns_list:
        h_key = int(h_ns)
        h_s = int(h_ns / 1e9)
        row[f'queue_rank_at_{h_s}s']    = (float(st['queue_rank'][h_key])
                                           if st['queue_rank'][h_key] is not None
                                           else float('nan'))
        row[f'filled_{h_s}s']           = int(max(st['filled_h'][h_key], 0))
        row[f'time_to_fill_s_{h_s}s']   = float(st['time_to_fill'][h_key])
    rows[row_idx] = row


# ──────────────────────────────────────────────────────────────────────────────
# Multi-date driver
# ──────────────────────────────────────────────────────────────────────────────
def discover_dates() -> List[str]:
    """All dates that have BOTH a non-stale prediction and a raw DBN."""
    pred_dates = set()
    for d in PRED_DIRS:
        for f in d.glob('*_predictions.npz'):
            if '_stale_' in f.name:
                continue
            stem = f.name
            date_str = stem.split('_')[0]
            if len(date_str) == 8 and date_str.isdigit():
                pred_dates.add(date_str)
    # Need raw DBN
    dbn_dates = set()
    for d in (LVL3_ROOT / 'data' / 'raw_mbo', LVL3_ROOT / 'data' / 'raw' / 'mbo'):
        if not d.exists():
            continue
        for f in d.glob('glbx-mdp3-*.mbo.dbn.zst'):
            ds = f.name.replace('glbx-mdp3-', '').replace('.mbo.dbn.zst', '')
            dbn_dates.add(ds)
    mbo_dates = set()
    for f in MBO_EVENT_DIR.glob('*_mbo_events.npz'):
        mbo_dates.add(f.name.replace('_mbo_events.npz', ''))
    return sorted(pred_dates & dbn_dates & mbo_dates)


def _worker(args):
    """Multiprocessing worker. Reconfigures logging in child."""
    date_str, out_dir, log_path = args
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(process)d] %(message)s',
        handlers=[logging.FileHandler(log_path, mode='a'),
                  logging.StreamHandler(sys.stdout)],
    )
    try:
        return build_labels_for_date(date_str, out_dir)
    except Exception as e:
        log.exception(f"[{date_str}] FAILED: {e}")
        return {'date': date_str, 'error': str(e)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dates', nargs='*', default=None,
                    help='Specific YYYYMMDD dates; default = all discoverable')
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--log-path', type=Path,
                    default=None,
                    help='Log file path (default: auto-named in logs/)')
    ap.add_argument('--smoke', action='store_true',
                    help='Smoke mode: run only the first discoverable date')
    ap.add_argument('--n-procs', type=int, default=8,
                    help='Multiprocessing workers (default 8)')
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if args.log_path is None:
        from datetime import datetime
        args.log_path = LOG_DIR / f"mbo_walker_labels_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(args.log_path, mode='a'),
                  logging.StreamHandler(sys.stdout)],
    )

    if args.dates:
        dates = list(args.dates)
    else:
        dates = discover_dates()
    if args.smoke:
        dates = dates[:1]

    log.info(f"MBO walker label generator: {len(dates)} date(s) -> {args.out_dir}")
    log.info(f"Log: {args.log_path}")

    summary = {'started_utc': time_mod.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                time_mod.gmtime()),
               'n_dates': len(dates),
               'horizons_s': list(HORIZONS_S),
               'per_date': []}

    if args.n_procs <= 1 or len(dates) == 1:
        for d in dates:
            res = build_labels_for_date(d, args.out_dir)
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
    # Aggregate fill rates
    if summary['per_date']:
        for h_s in HORIZONS_S:
            key = f'fill_rate_{int(h_s)}s'
            vals = [r[key] for r in summary['per_date'] if key in r]
            if vals:
                summary[f'global_{key}_mean'] = float(np.mean(vals))
                summary[f'global_{key}_min']  = float(np.min(vals))
                summary[f'global_{key}_max']  = float(np.max(vals))

        # Sanity check (HC #498 follow-up)
        for h_s in HORIZONS_S:
            key = f'global_fill_rate_{int(h_s)}s_mean'
            if key in summary:
                mean_fr = summary[key]
                if mean_fr > 0.95 or mean_fr < 0.005:
                    log.error(f"SANITY FAIL: fill_rate_{int(h_s)}s mean={mean_fr:.3f} "
                              f"is OUTSIDE [0.005, 0.95] — possible queue-mechanics bug.")
                    summary['sanity_fail'] = True

    sum_path = args.out_dir / '_summary.json'
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary written to {sum_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
