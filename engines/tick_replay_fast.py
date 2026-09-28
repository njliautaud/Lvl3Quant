#!/usr/bin/env python3
"""
Fast Vectorized Tick-Level FIFO Replay Engine for ES Futures
=============================================================

Two-phase design:
  Phase 1 (preprocess): Parse .dbn.zst MBO files once, filter to front-month
      ES + RTH hours, encode as compact numpy arrays, save as .npz.
  Phase 2 (simulate): Numba @njit inner loop over preprocessed arrays.
      No Python dicts, no dataclass overhead -- pure numeric arrays.

Same trade logic as tick_replay_engine.py:
  - Entry: passive limit at BBO (buy at bid for long, sell at ask for short)
  - Fill: back-of-queue FIFO -- requires traded volume to deplete queue ahead
  - TP exit: passive limit (FIFO queue tracking)
  - SL exit: market order (immediate)
  - Time stop: market exit after N seconds
  - Permutation test: random sign flips, p-value output

Databento MBO side conventions:
  - ADD/CANCEL/MODIFY: side = which side of the book (B=bid, A=ask)
  - TRADE (T): side = AGGRESSOR side (A=aggressive sell, B=aggressive buy)
  - FILL (F): side = PASSIVE order side (B=resting bid filled, A=resting ask filled)
  - For our fill logic:
    * Long entry (passive buy at bid): fills on T side=A (agg sell hits bids)
    * Short entry (passive sell at ask): fills on T side=B (agg buy lifts asks)
    * Long TP (passive sell at TP): fills on T side=B (agg buy)
    * Short TP (passive buy at TP): fills on T side=A (agg sell)

Cost model (AMP/Rithmic canonical):
  - Commission: $4.70 RT = 0.376 ticks
  - Passive exit: 0.376 ticks
  - Market exit: 1.376 ticks (commission + 1 tick spread)

Author: Claude (tick_replay_fast)
"""

import numpy as np
import glob
import os
import sys
import time
import json
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import njit, int64, float64, int32, int8, boolean
    from numba import types as nb_types
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("WARNING: numba not found, falling back to pure numpy (slower)")

# =============================================================================
# Constants
# =============================================================================

TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_PASSIVE_EXIT = COMMISSION_RT_TICKS       # 0.376
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS  # 1.376

PRED_STRIDE = 250
PRED_WINDOW = 1500

# Action encoding for numba (int8)
ACT_ADD = 0
ACT_CANCEL = 1
ACT_MODIFY = 2
ACT_TRADE = 3
ACT_FILL = 4
ACT_RESET = 5

# Side encoding (int8)
SIDE_BID = 0   # 'B'
SIDE_ASK = 1   # 'A'
SIDE_NONE = 2  # 'N'

# Trade result columns
N_TRADE_COLS = 14
# Exit reason encoding
EXIT_TP = 0
EXIT_SL = 1
EXIT_TIME = 2
EXIT_EOD = 3

# Default paths
DEFAULT_MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
DEFAULT_PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
DEFAULT_PREPROC_DIR = "/home/jupiter/Lvl3Quant/data/preprocessed_mbo"
DEFAULT_OUTPUT = "/home/jupiter/Lvl3Quant/engines/tick_replay_fast_results.json"


# =============================================================================
# Phase 1: Preprocessing
# =============================================================================

def detect_rth_bounds_utc(date_str: str) -> Tuple[int, int]:
    """
    Detect RTH bounds in UTC seconds-of-day for a given date.
    ES RTH: 9:30-16:00 ET.
    """
    import datetime
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    dt = datetime.date(year, month, day)

    def dst_start(y):
        d = datetime.date(y, 3, 8)
        while d.weekday() != 6:
            d += datetime.timedelta(days=1)
        return d

    def dst_end(y):
        d = datetime.date(y, 11, 1)
        while d.weekday() != 6:
            d += datetime.timedelta(days=1)
        return d

    if dst_start(year) <= dt < dst_end(year):
        utc_open_secs = (9 * 3600 + 30 * 60) + 4 * 3600
        utc_close_secs = 16 * 3600 + 4 * 3600
    else:
        utc_open_secs = (9 * 3600 + 30 * 60) + 5 * 3600
        utc_close_secs = 16 * 3600 + 5 * 3600

    return utc_open_secs, utc_close_secs


def preprocess_one_day(mbo_path: str, output_dir: str, date_str: str = None) -> str:
    """Preprocess a single MBO .dbn.zst file into a compact numpy .npz."""
    import databento as db

    basename = os.path.basename(mbo_path)
    if date_str is None:
        date_str = basename.split('-')[2].split('.')[0]

    out_path = os.path.join(output_dir, f"mbo_{date_str}.npz")
    if os.path.exists(out_path):
        print(f"  {date_str}: already preprocessed, skipping")
        return out_path

    t0 = time.time()
    print(f"  {date_str}: loading {basename}...")

    dbn = db.DBNStore.from_file(mbo_path)
    df = dbn.to_df()
    t_load = time.time() - t0

    # Find front-month ES (most trades, outright contract only)
    es_symbols = [s for s in df['symbol'].unique()
                  if s.startswith('ES') and '-' not in s and len(s) <= 5]
    if not es_symbols:
        print(f"  {date_str}: WARNING - no ES outright contract found")
        return None

    best_sym = max(es_symbols,
                   key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))
    df = df[df['symbol'] == best_sym]
    print(f"  {date_str}: front-month={best_sym}, {len(df)} events")

    # Keep ALL events (do NOT filter RTH here — prediction indices are relative
    # to the full event stream). RTH constraints are applied in simulation.
    ts_ns = df['ts_event'].values.astype('int64')
    utc_open, utc_close = detect_rth_bounds_utc(date_str)
    # Store RTH bounds in the output for the simulation to use
    rth_open_ns = int(utc_open * 1_000_000_000)
    rth_close_ns = int(utc_close * 1_000_000_000)
    print(f"  {date_str}: {len(df)} events (all, no RTH filter)")

    if len(df) == 0:
        print(f"  {date_str}: WARNING - no events")
        return None

    action_map = {'A': ACT_ADD, 'C': ACT_CANCEL, 'M': ACT_MODIFY,
                  'T': ACT_TRADE, 'F': ACT_FILL, 'R': ACT_RESET}
    actions_str = df['action'].values
    actions = np.array([action_map.get(a, ACT_RESET) for a in actions_str], dtype=np.int8)

    side_map = {'B': SIDE_BID, 'A': SIDE_ASK, 'N': SIDE_NONE}
    sides_str = df['side'].values
    sides = np.array([side_map.get(s, SIDE_NONE) for s in sides_str], dtype=np.int8)

    prices = df['price'].values.astype(np.float64)
    sizes = df['size'].values.astype(np.int32)
    order_ids = df['order_id'].values.astype(np.int64)

    n_trades = int(np.sum(actions == ACT_TRADE))

    # Compute RTH bounds as nanosecond-of-day for simulation use
    rth_open_sod_ns = int(utc_open * 1_000_000_000)
    rth_close_sod_ns = int(utc_close * 1_000_000_000)

    np.savez_compressed(out_path,
                        ts_ns=ts_ns,
                        action=actions,
                        side=sides,
                        price=prices,
                        size=sizes,
                        order_id=order_ids,
                        date=date_str,
                        symbol=best_sym,
                        n_trades=n_trades,
                        rth_open_sod_ns=rth_open_sod_ns,
                        rth_close_sod_ns=rth_close_sod_ns)

    elapsed = time.time() - t0
    fsize_mb = os.path.getsize(out_path) / 1e6
    print(f"  {date_str}: saved ({fsize_mb:.1f}MB, "
          f"{len(df)} events, {n_trades} trades, {elapsed:.1f}s)")
    return out_path


def preprocess_all(mbo_dir: str, output_dir: str, max_days: int = None,
                   pred_dir: str = None) -> List[str]:
    """Preprocess all MBO files (or only those matching prediction dates)."""
    os.makedirs(output_dir, exist_ok=True)
    mbo_files = sorted(glob.glob(os.path.join(mbo_dir, 'glbx-mdp3-*.mbo.dbn.zst')))

    if pred_dir:
        pred_dates = set()
        for f in glob.glob(os.path.join(pred_dir, 'oot_*.npz')):
            d = os.path.basename(f).replace('oot_', '').replace('.npz', '')
            pred_dates.add(d)
        mbo_files = [f for f in mbo_files
                     if os.path.basename(f).split('-')[2].split('.')[0] in pred_dates]
        print(f"Filtering to {len(mbo_files)} MBO files matching {len(pred_dates)} prediction dates")

    if max_days:
        mbo_files = mbo_files[:max_days]

    print(f"Preprocessing {len(mbo_files)} MBO files -> {output_dir}")
    outputs = []
    for i, mbo_path in enumerate(mbo_files):
        print(f"\n[{i+1}/{len(mbo_files)}]")
        out = preprocess_one_day(mbo_path, output_dir)
        if out:
            outputs.append(out)

    print(f"\nDone: {len(outputs)} files preprocessed")
    return outputs


# =============================================================================
# Phase 2: Numba-JIT Simulation
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _simulate_day_numba(ts_ns, action, side, price, size, order_id,
                            predictions, pred_event_indices,
                            tp_ticks, sl_ticks, hold_ns, cancel_ns,
                            signal_threshold, max_concurrent):
        """
        Core simulation loop in numba.

        BBO tracking:
        - ADD/CANCEL/MODIFY update BBO from book side
        - TRADE events: side = aggressor.
          T side=A (SIDE_ASK) = aggressive sell -> trade at bid -> bid consumed
          T side=B (SIDE_BID) = aggressive buy -> trade at ask -> ask consumed
        - We anchor BBO from trade prices since ES is 1-tick wide during RTH
        - FILL events: side = passive order side. Used for book depth reduction
          but NOT for our order fill logic (T events handle that).
        """
        n_events = len(ts_ns)
        n_preds = len(predictions)

        best_bid = 0.0
        best_ask = 0.0
        bid_size = 0
        ask_size = 0

        MAX_ORDERS = 16
        pend_active = np.zeros(MAX_ORDERS, dtype=np.int8)
        pend_side = np.zeros(MAX_ORDERS, dtype=np.int8)
        pend_price = np.zeros(MAX_ORDERS, dtype=np.float64)
        pend_signal_ts = np.zeros(MAX_ORDERS, dtype=np.int64)
        pend_signal_str = np.zeros(MAX_ORDERS, dtype=np.float64)
        pend_queue = np.zeros(MAX_ORDERS, dtype=np.float64)
        pend_tp = np.zeros(MAX_ORDERS, dtype=np.float64)
        pend_sl = np.zeros(MAX_ORDERS, dtype=np.float64)
        pend_hold_ns = np.zeros(MAX_ORDERS, dtype=np.int64)
        pend_cancel_ts = np.zeros(MAX_ORDERS, dtype=np.int64)

        pos_active = np.zeros(MAX_ORDERS, dtype=np.int8)
        pos_side = np.zeros(MAX_ORDERS, dtype=np.int8)
        pos_entry_price = np.zeros(MAX_ORDERS, dtype=np.float64)
        pos_fill_ts = np.zeros(MAX_ORDERS, dtype=np.int64)
        pos_signal_ts = np.zeros(MAX_ORDERS, dtype=np.int64)
        pos_signal_str = np.zeros(MAX_ORDERS, dtype=np.float64)
        pos_queue_depth = np.zeros(MAX_ORDERS, dtype=np.int32)
        pos_fill_latency = np.zeros(MAX_ORDERS, dtype=np.int64)
        pos_tp_price = np.zeros(MAX_ORDERS, dtype=np.float64)
        pos_sl_price = np.zeros(MAX_ORDERS, dtype=np.float64)
        pos_deadline = np.zeros(MAX_ORDERS, dtype=np.int64)
        pos_tp_queue = np.zeros(MAX_ORDERS, dtype=np.float64)
        pos_best_price = np.zeros(MAX_ORDERS, dtype=np.float64)
        pos_worst_price = np.zeros(MAX_ORDERS, dtype=np.float64)

        max_trades = n_preds * 2 + 100
        trades_out = np.zeros((max_trades, N_TRADE_COLS), dtype=np.float64)
        n_completed = 0
        pred_idx = 0
        last_trade_price = 0.0
        tick_size = 0.25

        for evt_i in range(n_events):
            act = action[evt_i]
            sd = side[evt_i]
            px = price[evt_i]
            sz = size[evt_i]
            ts = ts_ns[evt_i]

            if px != px or px <= 0:
                if pred_idx < n_preds and evt_i == pred_event_indices[pred_idx]:
                    pred_idx += 1
                continue

            # ----- Update BBO -----
            if act == ACT_ADD:
                if sd == SIDE_BID:
                    if px > best_bid or best_bid == 0.0:
                        best_bid = px
                        bid_size = sz
                    elif abs(px - best_bid) < 0.001:
                        bid_size += sz
                elif sd == SIDE_ASK:
                    if px < best_ask or best_ask == 0.0:
                        best_ask = px
                        ask_size = sz
                    elif abs(px - best_ask) < 0.001:
                        ask_size += sz

            elif act == ACT_CANCEL:
                if sd == SIDE_BID and abs(px - best_bid) < 0.001:
                    bid_size -= sz
                    if bid_size <= 0:
                        best_bid -= tick_size
                        bid_size = 0
                elif sd == SIDE_ASK and abs(px - best_ask) < 0.001:
                    ask_size -= sz
                    if ask_size <= 0:
                        best_ask += tick_size
                        ask_size = 0

            elif act == ACT_MODIFY:
                if sd == SIDE_BID and px > best_bid:
                    best_bid = px
                    bid_size = sz
                elif sd == SIDE_ASK and (px < best_ask or best_ask == 0.0):
                    best_ask = px
                    ask_size = sz

            elif act == ACT_TRADE:
                last_trade_price = px

                # Anchor BBO from trade prices.
                # T side=A: aggressive sell hitting bid at px
                #   bid is at px (possibly consumed), ask is at px + tick
                # T side=B: aggressive buy lifting ask at px
                #   ask is at px (possibly consumed), bid is at px - tick
                if sd == SIDE_ASK:
                    # Aggressive sell hit bid
                    if abs(px - best_bid) < 0.001:
                        bid_size -= sz
                        if bid_size <= 0:
                            bid_size = 0
                    # Anchor: trade at bid level, ask must be above
                    best_bid = px
                    if best_ask < px + tick_size or best_ask == 0.0:
                        best_ask = px + tick_size
                        ask_size = 0
                elif sd == SIDE_BID:
                    # Aggressive buy lifted ask
                    if abs(px - best_ask) < 0.001:
                        ask_size -= sz
                        if ask_size <= 0:
                            ask_size = 0
                    best_ask = px
                    if best_bid > px - tick_size or best_bid == 0.0:
                        best_bid = px - tick_size
                        bid_size = 0

                # ----- Check entry fills (on TRADE events only) -----
                for oi in range(MAX_ORDERS):
                    if pend_active[oi] == 0:
                        continue

                    filled = False
                    if pend_side[oi] == 0:
                        # Long (passive buy at bid): needs T side=A (agg sell)
                        if sd == SIDE_ASK:
                            if abs(px - pend_price[oi]) < 0.001:
                                pend_queue[oi] -= sz
                                if pend_queue[oi] <= 0:
                                    filled = True
                            elif px < pend_price[oi]:
                                filled = True
                    else:
                        # Short (passive sell at ask): needs T side=B (agg buy)
                        if sd == SIDE_BID:
                            if abs(px - pend_price[oi]) < 0.001:
                                pend_queue[oi] -= sz
                                if pend_queue[oi] <= 0:
                                    filled = True
                            elif px > pend_price[oi]:
                                filled = True

                    if filled:
                        pi = -1
                        for j in range(MAX_ORDERS):
                            if pos_active[j] == 0:
                                pi = j
                                break
                        if pi >= 0:
                            pos_active[pi] = 1
                            pos_side[pi] = pend_side[oi]
                            pos_entry_price[pi] = pend_price[oi]
                            pos_fill_ts[pi] = ts
                            pos_signal_ts[pi] = pend_signal_ts[oi]
                            pos_signal_str[pi] = pend_signal_str[oi]
                            pos_queue_depth[pi] = int(max(pend_queue[oi] + sz, 0))
                            pos_fill_latency[pi] = ts - pend_signal_ts[oi]
                            pos_tp_price[pi] = pend_tp[oi]
                            pos_sl_price[pi] = pend_sl[oi]
                            pos_deadline[pi] = ts + pend_hold_ns[oi]
                            if pend_side[oi] == 0:
                                pos_tp_queue[pi] = float(max(ask_size, 0))
                            else:
                                pos_tp_queue[pi] = float(max(bid_size, 0))
                            pos_best_price[pi] = pend_price[oi]
                            pos_worst_price[pi] = pend_price[oi]
                        pend_active[oi] = 0

                # ----- Check exits (on TRADE events only) -----
                for pi in range(MAX_ORDERS):
                    if pos_active[pi] == 0:
                        continue

                    exit_reason = -1
                    exit_price = 0.0
                    cost = COST_PASSIVE_EXIT

                    # Update MFE/MAE
                    if pos_side[pi] == 0:
                        if px > pos_best_price[pi]:
                            pos_best_price[pi] = px
                        if px < pos_worst_price[pi]:
                            pos_worst_price[pi] = px
                    else:
                        if px < pos_best_price[pi]:
                            pos_best_price[pi] = px
                        if px > pos_worst_price[pi]:
                            pos_worst_price[pi] = px

                    # SL (market exit)
                    if pos_side[pi] == 0 and px <= pos_sl_price[pi]:
                        exit_reason = EXIT_SL
                        exit_price = pos_sl_price[pi]
                        cost = COST_MARKET_EXIT
                    elif pos_side[pi] == 1 and px >= pos_sl_price[pi]:
                        exit_reason = EXIT_SL
                        exit_price = pos_sl_price[pi]
                        cost = COST_MARKET_EXIT

                    # TP (passive limit with FIFO queue)
                    if exit_reason < 0:
                        if pos_side[pi] == 0 and sd == SIDE_BID:
                            # Long TP: selling at TP ask; needs agg buy (T sd=B)
                            if abs(px - pos_tp_price[pi]) < 0.001:
                                pos_tp_queue[pi] -= sz
                                if pos_tp_queue[pi] <= 0:
                                    exit_reason = EXIT_TP
                                    exit_price = pos_tp_price[pi]
                                    cost = COST_PASSIVE_EXIT
                            elif px > pos_tp_price[pi]:
                                exit_reason = EXIT_TP
                                exit_price = pos_tp_price[pi]
                                cost = COST_PASSIVE_EXIT
                        elif pos_side[pi] == 1 and sd == SIDE_ASK:
                            # Short TP: buying at TP bid; needs agg sell (T sd=A)
                            if abs(px - pos_tp_price[pi]) < 0.001:
                                pos_tp_queue[pi] -= sz
                                if pos_tp_queue[pi] <= 0:
                                    exit_reason = EXIT_TP
                                    exit_price = pos_tp_price[pi]
                                    cost = COST_PASSIVE_EXIT
                            elif px < pos_tp_price[pi]:
                                exit_reason = EXIT_TP
                                exit_price = pos_tp_price[pi]
                                cost = COST_PASSIVE_EXIT

                    # Time stop
                    if exit_reason < 0 and ts >= pos_deadline[pi]:
                        exit_reason = EXIT_TIME
                        exit_price = px
                        cost = COST_MARKET_EXIT

                    if exit_reason >= 0 and n_completed < max_trades:
                        ep = pos_entry_price[pi]
                        if pos_side[pi] == 0:
                            raw_pnl = (exit_price - ep) / tick_size
                            mfe = (pos_best_price[pi] - ep) / tick_size
                            mae = (ep - pos_worst_price[pi]) / tick_size
                        else:
                            raw_pnl = (ep - exit_price) / tick_size
                            mfe = (ep - pos_best_price[pi]) / tick_size
                            mae = (pos_worst_price[pi] - ep) / tick_size

                        trades_out[n_completed, 0] = float(pos_side[pi])
                        trades_out[n_completed, 1] = ep
                        trades_out[n_completed, 2] = exit_price
                        trades_out[n_completed, 3] = float(pos_fill_ts[pi])
                        trades_out[n_completed, 4] = float(ts)
                        trades_out[n_completed, 5] = float(pos_signal_ts[pi])
                        trades_out[n_completed, 6] = pos_signal_str[pi]
                        trades_out[n_completed, 7] = float(exit_reason)
                        trades_out[n_completed, 8] = float(pos_queue_depth[pi])
                        trades_out[n_completed, 9] = float(pos_fill_latency[pi])
                        trades_out[n_completed, 10] = raw_pnl - cost
                        trades_out[n_completed, 11] = cost
                        trades_out[n_completed, 12] = max(mfe, 0.0)
                        trades_out[n_completed, 13] = max(mae, 0.0)
                        n_completed += 1
                        pos_active[pi] = 0

            elif act == ACT_FILL:
                last_trade_price = px
                # F events: side = passive order side
                # F side=B: resting bid filled (same trade as T side=A)
                # F side=A: resting ask filled (same trade as T side=B)
                # Reduce book depth only (fill checking done on T events)
                if sd == SIDE_BID and abs(px - best_bid) < 0.001:
                    bid_size -= sz
                    if bid_size <= 0:
                        bid_size = 0
                elif sd == SIDE_ASK and abs(px - best_ask) < 0.001:
                    ask_size -= sz
                    if ask_size <= 0:
                        ask_size = 0

            # ----- Pending cancels (every 1000 events) -----
            if evt_i % 1000 == 0:
                for oi in range(MAX_ORDERS):
                    if pend_active[oi] == 1 and ts > pend_cancel_ts[oi]:
                        pend_active[oi] = 0

            # ----- Prediction check -----
            if pred_idx < n_preds and evt_i == pred_event_indices[pred_idx]:
                pred_val = predictions[pred_idx]
                pred_idx += 1

                if best_bid <= 0.0 or best_ask <= 0.0:
                    continue
                if best_ask - best_bid > 2 * tick_size:
                    continue

                total_exp = 0
                for oi in range(MAX_ORDERS):
                    if pend_active[oi]:
                        total_exp += 1
                    if pos_active[oi]:
                        total_exp += 1
                if total_exp >= max_concurrent:
                    continue

                order_side = -1
                entry_price = 0.0
                queue_ahead = 0.0
                tp_price = 0.0
                sl_price = 0.0

                if pred_val > signal_threshold:
                    order_side = 0
                    entry_price = best_bid
                    queue_ahead = float(bid_size)
                    tp_price = entry_price + tp_ticks * tick_size
                    sl_price = entry_price - sl_ticks * tick_size
                elif pred_val < -signal_threshold:
                    order_side = 1
                    entry_price = best_ask
                    queue_ahead = float(ask_size)
                    tp_price = entry_price - tp_ticks * tick_size
                    sl_price = entry_price + sl_ticks * tick_size

                if order_side >= 0:
                    slot = -1
                    for oi in range(MAX_ORDERS):
                        if pend_active[oi] == 0:
                            slot = oi
                            break
                    if slot >= 0:
                        pend_active[slot] = 1
                        pend_side[slot] = int8(order_side)
                        pend_price[slot] = entry_price
                        pend_signal_ts[slot] = ts
                        pend_signal_str[slot] = pred_val
                        pend_queue[slot] = queue_ahead
                        pend_tp[slot] = tp_price
                        pend_sl[slot] = sl_price
                        pend_hold_ns[slot] = hold_ns
                        pend_cancel_ts[slot] = ts + cancel_ns

        # ----- EOD: force close remaining -----
        if last_trade_price > 0:
            final_ts = ts_ns[-1] if n_events > 0 else 0
            for pi in range(MAX_ORDERS):
                if pos_active[pi] == 1 and n_completed < max_trades:
                    ep = pos_entry_price[pi]
                    exit_price = last_trade_price
                    cost = COST_MARKET_EXIT
                    if pos_side[pi] == 0:
                        raw_pnl = (exit_price - ep) / tick_size
                        mfe = (pos_best_price[pi] - ep) / tick_size
                        mae = (ep - pos_worst_price[pi]) / tick_size
                    else:
                        raw_pnl = (ep - exit_price) / tick_size
                        mfe = (ep - pos_best_price[pi]) / tick_size
                        mae = (pos_worst_price[pi] - ep) / tick_size

                    trades_out[n_completed, 0] = float(pos_side[pi])
                    trades_out[n_completed, 1] = ep
                    trades_out[n_completed, 2] = exit_price
                    trades_out[n_completed, 3] = float(pos_fill_ts[pi])
                    trades_out[n_completed, 4] = float(final_ts)
                    trades_out[n_completed, 5] = float(pos_signal_ts[pi])
                    trades_out[n_completed, 6] = pos_signal_str[pi]
                    trades_out[n_completed, 7] = float(EXIT_EOD)
                    trades_out[n_completed, 8] = float(pos_queue_depth[pi])
                    trades_out[n_completed, 9] = float(pos_fill_latency[pi])
                    trades_out[n_completed, 10] = raw_pnl - cost
                    trades_out[n_completed, 11] = cost
                    trades_out[n_completed, 12] = max(mfe, 0.0)
                    trades_out[n_completed, 13] = max(mae, 0.0)
                    n_completed += 1
                    pos_active[pi] = 0

        return trades_out[:n_completed]


# Fallback: pure numpy/python simulation (no numba)
def _simulate_day_python(ts_ns, action, side, price, size, order_id,
                         predictions, pred_event_indices,
                         tp_ticks, sl_ticks, hold_ns, cancel_ns,
                         signal_threshold, max_concurrent):
    """Pure Python fallback -- same logic as numba version but slower."""
    n_events = len(ts_ns)
    n_preds = len(predictions)

    best_bid = 0.0
    best_ask = 0.0
    bid_size = 0
    ask_size = 0
    last_trade_price = 0.0
    tick_size = 0.25

    pending = []
    positions = []
    completed = []
    pred_idx = 0

    for evt_i in range(n_events):
        act = action[evt_i]
        sd = side[evt_i]
        px = price[evt_i]
        sz = int(size[evt_i])
        ts = ts_ns[evt_i]

        if np.isnan(px) or px <= 0:
            if pred_idx < n_preds and evt_i == pred_event_indices[pred_idx]:
                pred_idx += 1
            continue

        # Update BBO
        if act == ACT_ADD:
            if sd == SIDE_BID:
                if px > best_bid or best_bid == 0:
                    best_bid = px; bid_size = sz
                elif abs(px - best_bid) < 0.001:
                    bid_size += sz
            elif sd == SIDE_ASK:
                if px < best_ask or best_ask == 0:
                    best_ask = px; ask_size = sz
                elif abs(px - best_ask) < 0.001:
                    ask_size += sz
        elif act == ACT_CANCEL:
            if sd == SIDE_BID and abs(px - best_bid) < 0.001:
                bid_size -= sz
                if bid_size <= 0:
                    best_bid -= tick_size; bid_size = 0
            elif sd == SIDE_ASK and abs(px - best_ask) < 0.001:
                ask_size -= sz
                if ask_size <= 0:
                    best_ask += tick_size; ask_size = 0
        elif act == ACT_MODIFY:
            if sd == SIDE_BID and px > best_bid:
                best_bid = px; bid_size = sz
            elif sd == SIDE_ASK and (px < best_ask or best_ask == 0):
                best_ask = px; ask_size = sz
        elif act == ACT_TRADE:
            last_trade_price = px
            if sd == SIDE_ASK:
                if abs(px - best_bid) < 0.001:
                    bid_size -= sz
                    if bid_size <= 0: bid_size = 0
                best_bid = px
                if best_ask < px + tick_size or best_ask == 0:
                    best_ask = px + tick_size; ask_size = 0
            elif sd == SIDE_BID:
                if abs(px - best_ask) < 0.001:
                    ask_size -= sz
                    if ask_size <= 0: ask_size = 0
                best_ask = px
                if best_bid > px - tick_size or best_bid == 0:
                    best_bid = px - tick_size; bid_size = 0

            # Check entry fills
            newly_filled = []
            for oi, o in enumerate(pending):
                filled = False
                if o['side'] == 0 and sd == SIDE_ASK:
                    if abs(px - o['price']) < 0.001:
                        o['queue'] -= sz
                        if o['queue'] <= 0: filled = True
                    elif px < o['price']: filled = True
                elif o['side'] == 1 and sd == SIDE_BID:
                    if abs(px - o['price']) < 0.001:
                        o['queue'] -= sz
                        if o['queue'] <= 0: filled = True
                    elif px > o['price']: filled = True
                if filled:
                    newly_filled.append(oi)
                    tpq = float(max(ask_size if o['side']==0 else bid_size, 0))
                    positions.append({
                        'side': o['side'], 'entry': o['price'], 'fill_ts': ts,
                        'sig_ts': o['sig_ts'], 'sig_str': o['sig_str'],
                        'qdepth': int(max(o['queue']+sz, 0)),
                        'fill_lat': ts - o['sig_ts'],
                        'tp': o['tp'], 'sl': o['sl'],
                        'deadline': ts + o['hold_ns'],
                        'tp_queue': tpq,
                        'best': o['price'], 'worst': o['price'],
                    })
            for oi in reversed(newly_filled):
                pending.pop(oi)

            # Check exits
            closed_idx = []
            for pi, p in enumerate(positions):
                exit_reason = -1; exit_price = 0.0; cost = COST_PASSIVE_EXIT
                if p['side'] == 0:
                    p['best'] = max(p['best'], px)
                    p['worst'] = min(p['worst'], px)
                else:
                    p['best'] = min(p['best'], px)
                    p['worst'] = max(p['worst'], px)

                if p['side'] == 0 and px <= p['sl']:
                    exit_reason = EXIT_SL; exit_price = p['sl']; cost = COST_MARKET_EXIT
                elif p['side'] == 1 and px >= p['sl']:
                    exit_reason = EXIT_SL; exit_price = p['sl']; cost = COST_MARKET_EXIT

                if exit_reason < 0:
                    if p['side'] == 0 and sd == SIDE_BID:
                        if abs(px - p['tp']) < 0.001:
                            p['tp_queue'] -= sz
                            if p['tp_queue'] <= 0:
                                exit_reason = EXIT_TP; exit_price = p['tp']
                        elif px > p['tp']:
                            exit_reason = EXIT_TP; exit_price = p['tp']
                    elif p['side'] == 1 and sd == SIDE_ASK:
                        if abs(px - p['tp']) < 0.001:
                            p['tp_queue'] -= sz
                            if p['tp_queue'] <= 0:
                                exit_reason = EXIT_TP; exit_price = p['tp']
                        elif px < p['tp']:
                            exit_reason = EXIT_TP; exit_price = p['tp']

                if exit_reason < 0 and ts >= p['deadline']:
                    exit_reason = EXIT_TIME; exit_price = px; cost = COST_MARKET_EXIT

                if exit_reason >= 0:
                    ep = p['entry']
                    if p['side'] == 0:
                        raw = (exit_price - ep) / tick_size
                        mfe = (p['best'] - ep) / tick_size
                        mae = (ep - p['worst']) / tick_size
                    else:
                        raw = (ep - exit_price) / tick_size
                        mfe = (ep - p['best']) / tick_size
                        mae = (p['worst'] - ep) / tick_size
                    completed.append([
                        float(p['side']), ep, exit_price, float(p['fill_ts']),
                        float(ts), float(p['sig_ts']), p['sig_str'],
                        float(exit_reason), float(p['qdepth']),
                        float(p['fill_lat']), raw - cost, cost,
                        max(mfe, 0), max(mae, 0)
                    ])
                    closed_idx.append(pi)
            for pi in reversed(closed_idx):
                positions.pop(pi)

        elif act == ACT_FILL:
            last_trade_price = px
            if sd == SIDE_BID and abs(px - best_bid) < 0.001:
                bid_size -= sz
                if bid_size <= 0: bid_size = 0
            elif sd == SIDE_ASK and abs(px - best_ask) < 0.001:
                ask_size -= sz
                if ask_size <= 0: ask_size = 0

        # Cancels
        if evt_i % 1000 == 0:
            pending = [o for o in pending if ts <= o['cancel_ts']]

        # Predictions
        if pred_idx < n_preds and evt_i == pred_event_indices[pred_idx]:
            pred_val = predictions[pred_idx]
            pred_idx += 1
            if best_bid <= 0 or best_ask <= 0: continue
            if best_ask - best_bid > 2 * tick_size: continue
            total_exp = len(pending) + len(positions)
            if total_exp >= max_concurrent: continue

            if pred_val > signal_threshold:
                pending.append({'side': 0, 'price': best_bid, 'sig_ts': ts,
                    'sig_str': pred_val, 'queue': float(bid_size),
                    'tp': best_bid + tp_ticks*tick_size,
                    'sl': best_bid - sl_ticks*tick_size,
                    'hold_ns': hold_ns, 'cancel_ts': ts + cancel_ns})
            elif pred_val < -signal_threshold:
                pending.append({'side': 1, 'price': best_ask, 'sig_ts': ts,
                    'sig_str': pred_val, 'queue': float(ask_size),
                    'tp': best_ask - tp_ticks*tick_size,
                    'sl': best_ask + sl_ticks*tick_size,
                    'hold_ns': hold_ns, 'cancel_ts': ts + cancel_ns})

    # EOD
    if last_trade_price > 0:
        final_ts = ts_ns[-1] if n_events > 0 else 0
        for p in positions:
            ep = p['entry']; exit_price = last_trade_price; cost = COST_MARKET_EXIT
            if p['side'] == 0:
                raw = (exit_price - ep) / tick_size
                mfe = (p['best'] - ep) / tick_size
                mae = (ep - p['worst']) / tick_size
            else:
                raw = (ep - exit_price) / tick_size
                mfe = (ep - p['best']) / tick_size
                mae = (p['worst'] - ep) / tick_size
            completed.append([
                float(p['side']), ep, exit_price, float(p['fill_ts']),
                float(final_ts), float(p['sig_ts']), p['sig_str'],
                float(EXIT_EOD), float(p['qdepth']),
                float(p['fill_lat']), raw - cost, cost,
                max(mfe, 0), max(mae, 0)
            ])

    if completed:
        return np.array(completed, dtype=np.float64)
    return np.zeros((0, N_TRADE_COLS), dtype=np.float64)


def simulate_day(preproc_path: str, predictions: np.ndarray,
                 tp_ticks: int = 4, sl_ticks: int = 3,
                 hold_seconds: float = 30.0, cancel_seconds: float = 15.0,
                 signal_threshold: float = 0.3, max_concurrent: int = 1) -> np.ndarray:
    """Run simulation on one preprocessed day."""
    data = np.load(preproc_path)
    ts_ns = data['ts_ns']
    action_arr = data['action']
    side_arr = data['side']
    price_arr = data['price']
    size_arr = data['size']
    order_id_arr = data['order_id']

    n_events = len(ts_ns)

    n_preds = len(predictions)
    pred_event_indices = np.arange(PRED_WINDOW,
                                   PRED_WINDOW + n_preds * PRED_STRIDE,
                                   PRED_STRIDE, dtype=np.int64)
    valid_mask = pred_event_indices < n_events
    pred_event_indices = pred_event_indices[valid_mask]
    predictions = predictions[:len(pred_event_indices)].astype(np.float64)

    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)

    sim_fn = _simulate_day_numba if HAS_NUMBA else _simulate_day_python

    trades = sim_fn(
        ts_ns, action_arr, side_arr, price_arr, size_arr.astype(np.int32), order_id_arr,
        predictions, pred_event_indices,
        tp_ticks, sl_ticks, hold_ns, cancel_ns,
        signal_threshold, max_concurrent
    )

    return trades


# =============================================================================
# Metrics & Reporting
# =============================================================================

EXIT_NAMES = {EXIT_TP: 'tp', EXIT_SL: 'sl', EXIT_TIME: 'time_stop', EXIT_EOD: 'eod'}
SIDE_NAMES = {0: 'long', 1: 'short'}


def compute_metrics(trades_arr: np.ndarray, label: str = "") -> Dict:
    """Compute metrics from trade array (n_trades x N_TRADE_COLS)."""
    if len(trades_arr) == 0:
        return {
            'label': label, 'n_trades': 0, 'net_pnl_ticks': 0, 'net_pnl_dollars': 0,
            'win_rate': 0, 'profit_factor': 0, 'sharpe': 0, 'sortino': 0,
            'max_dd_ticks': 0, 'avg_pnl_ticks': 0, 'avg_mfe': 0, 'avg_mae': 0,
        }

    pnls = trades_arr[:, 10]
    n = len(pnls)
    wins = int(np.sum(pnls > 0))
    losses = n - wins
    wr = wins / n

    gross_profit = float(np.sum(pnls[pnls > 0])) if wins > 0 else 0
    gross_loss = float(abs(np.sum(pnls[pnls <= 0]))) if losses > 0 else 0.001
    pf = gross_profit / gross_loss

    entry_ts = trades_arr[:, 3]
    day_keys = (entry_ts / (24 * 3600 * 1e9)).astype(np.int64)
    unique_days = np.unique(day_keys)
    daily_returns = np.array([float(np.sum(pnls[day_keys == d])) for d in unique_days])

    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0:
            sortino = float(np.mean(daily_returns) / np.std(downside) * np.sqrt(252))
        else:
            sortino = 99.0
    else:
        sharpe = 0.0
        sortino = 0.0

    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    max_dd = float(np.max(peak - cumulative)) if len(cumulative) > 0 else 0

    exit_reasons = {}
    for code, name in EXIT_NAMES.items():
        cnt = int(np.sum(trades_arr[:, 7] == code))
        if cnt > 0:
            exit_reasons[name] = cnt

    n_long = int(np.sum(trades_arr[:, 0] == 0))
    n_short = int(np.sum(trades_arr[:, 0] == 1))

    return {
        'label': label,
        'n_trades': n,
        'net_pnl_ticks': float(np.sum(pnls)),
        'net_pnl_dollars': float(np.sum(pnls) * TICK_VALUE),
        'win_rate': float(wr),
        'profit_factor': float(pf),
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd_ticks': max_dd,
        'avg_pnl_ticks': float(np.mean(pnls)),
        'avg_mfe': float(np.mean(trades_arr[:, 12])),
        'avg_mae': float(np.mean(trades_arr[:, 13])),
        'n_days': len(unique_days),
        'avg_trades_per_day': float(n / max(len(unique_days), 1)),
        'exit_reasons': exit_reasons,
        'long_trades': n_long,
        'short_trades': n_short,
    }


def print_metrics(m: Dict, prefix: str = ""):
    p = f"{prefix}  " if prefix else "  "
    print(f"{p}Trades:   {m['n_trades']} "
          f"({m['long_trades']}L / {m['short_trades']}S)")
    print(f"{p}Net PnL:  {m['net_pnl_ticks']:.1f} ticks "
          f"(${m['net_pnl_dollars']:.0f})")
    print(f"{p}WR:       {m['win_rate']:.1%}")
    print(f"{p}PF:       {m['profit_factor']:.2f}")
    print(f"{p}Sharpe:   {m['sharpe']:.2f}")
    print(f"{p}Sortino:  {m['sortino']:.2f}")
    print(f"{p}Max DD:   {m['max_dd_ticks']:.1f} ticks")
    print(f"{p}Avg MFE:  {m['avg_mfe']:.1f}t, Avg MAE: {m['avg_mae']:.1f}t")
    print(f"{p}Exits:    {m['exit_reasons']}")


# =============================================================================
# Sweep & Permutation Test
# =============================================================================

def load_predictions(pred_dir: str, head: str = 'pred_log_ret_1s') -> Dict[str, np.ndarray]:
    """Load predictions from per-date .npz files.

    Args:
        pred_dir: Directory containing oot_YYYYMMDD.npz files
        head: Prediction head to use. Options:
            - pred_log_ret_1s (default, best IC)
            - pred_log_ret_5s
            - pred_log_ret_10s
            - predictions (legacy format)
    """
    preds = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, 'oot_*.npz'))):
        d = np.load(f, allow_pickle=True)
        date_str = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        if head in d.files:
            preds[date_str] = d[head].astype(np.float64)
        elif head == 'pred_log_ret_1s' and 'predictions' in d.files:
            preds[date_str] = d['predictions'].astype(np.float64)
    return preds


def find_preprocessed(preproc_dir: str) -> Dict[str, str]:
    mapping = {}
    for f in sorted(glob.glob(os.path.join(preproc_dir, 'mbo_*.npz'))):
        date_str = os.path.basename(f).replace('mbo_', '').replace('.npz', '')
        mapping[date_str] = f
    return mapping


def match_dates(preproc_map: Dict[str, str],
                pred_map: Dict[str, np.ndarray]) -> List[Tuple[str, str, np.ndarray]]:
    matched = []
    for date_str in sorted(preproc_map.keys()):
        if date_str in pred_map:
            matched.append((date_str, preproc_map[date_str], pred_map[date_str]))
    return matched


def run_one_config(matched_dates: List[Tuple[str, str, np.ndarray]],
                   tp_ticks: int, sl_ticks: int,
                   hold_seconds: float, cancel_seconds: float,
                   signal_threshold: float, max_concurrent: int,
                   verbose: bool = True) -> np.ndarray:
    all_trades = []
    for date_str, preproc_path, preds in matched_dates:
        t0 = time.time()
        trades = simulate_day(preproc_path, preds,
                              tp_ticks=tp_ticks, sl_ticks=sl_ticks,
                              hold_seconds=hold_seconds,
                              cancel_seconds=cancel_seconds,
                              signal_threshold=signal_threshold,
                              max_concurrent=max_concurrent)
        elapsed = time.time() - t0
        if verbose:
            pnl = float(np.sum(trades[:, 10])) if len(trades) > 0 else 0
            print(f"  {date_str}: {len(trades)} trades, "
                  f"PnL={pnl:.1f}t ({elapsed:.1f}s)")
        all_trades.append(trades)

    if all_trades and any(len(t) > 0 for t in all_trades):
        return np.vstack([t for t in all_trades if len(t) > 0])
    return np.zeros((0, N_TRADE_COLS))


def run_permutation(matched_dates: List[Tuple[str, str, np.ndarray]],
                    tp_ticks: int, sl_ticks: int,
                    hold_seconds: float, cancel_seconds: float,
                    signal_threshold: float, max_concurrent: int,
                    n_perms: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    random_pnls = np.zeros(n_perms)

    for perm_i in range(n_perms):
        perm_trades_list = []
        for date_str, preproc_path, preds in matched_dates:
            signs = rng.choice(np.array([-1.0, 1.0]), size=len(preds))
            random_preds = preds * signs
            trades = simulate_day(preproc_path, random_preds,
                                  tp_ticks=tp_ticks, sl_ticks=sl_ticks,
                                  hold_seconds=hold_seconds,
                                  cancel_seconds=cancel_seconds,
                                  signal_threshold=signal_threshold,
                                  max_concurrent=max_concurrent)
            if len(trades) > 0:
                perm_trades_list.append(trades)

        if perm_trades_list:
            all_perm = np.vstack(perm_trades_list)
            random_pnls[perm_i] = float(np.sum(all_perm[:, 10]))

        if (perm_i + 1) % 25 == 0:
            print(f"    Perm {perm_i+1}/{n_perms}: random PnL = {random_pnls[perm_i]:.1f}t")

    return random_pnls


def run_sweep(preproc_dir: str, pred_dir: str,
              tp_range: range, sl_range: range,
              signal_threshold: float = 0.3,
              hold_seconds: float = 30.0,
              cancel_seconds: float = 15.0,
              max_concurrent: int = 1,
              max_days: int = None,
              n_perms: int = 100,
              output_path: str = None,
              head: str = 'pred_log_ret_1s'):
    print("=" * 70)
    print("FAST TICK-LEVEL FIFO REPLAY ENGINE")
    print("=" * 70)

    print(f"\nLoading predictions from {pred_dir} (head={head})...")
    pred_map = load_predictions(pred_dir, head=head)
    print(f"  {len(pred_map)} prediction dates")

    preproc_map = find_preprocessed(preproc_dir)
    print(f"  {len(preproc_map)} preprocessed MBO dates")

    matched = match_dates(preproc_map, pred_map)
    if max_days:
        matched = matched[:max_days]
    print(f"  {len(matched)} matched dates for simulation")

    if not matched:
        print("ERROR: No matching preprocessed + prediction dates!")
        print("  Run --mode preprocess first.")
        return

    if HAS_NUMBA:
        print("\nWarming up numba JIT (first call compiles)...")
        warmup_date = matched[0]
        t0 = time.time()
        _ = simulate_day(warmup_date[1], warmup_date[2],
                         tp_ticks=list(tp_range)[0],
                         sl_ticks=list(sl_range)[0],
                         hold_seconds=hold_seconds,
                         cancel_seconds=cancel_seconds,
                         signal_threshold=signal_threshold,
                         max_concurrent=max_concurrent)
        print(f"  JIT warmup done ({time.time()-t0:.1f}s)")

    configs = [(tp, sl) for tp in tp_range for sl in sl_range]
    print(f"\nSweeping {len(configs)} TP/SL configs: "
          f"TP={list(tp_range)}, SL={list(sl_range)}")
    print(f"Signal threshold: {signal_threshold}")
    print(f"Hold: {hold_seconds}s, Cancel: {cancel_seconds}s")
    print(f"Permutation tests: {n_perms} per config")
    print()

    results = []
    total_t0 = time.time()

    for ci, (tp, sl) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"CONFIG {ci+1}/{len(configs)}: TP={tp} SL={sl}")
        print(f"{'='*60}")

        config_t0 = time.time()

        model_trades = run_one_config(
            matched, tp, sl, hold_seconds, cancel_seconds,
            signal_threshold, max_concurrent, verbose=True)
        model_metrics = compute_metrics(model_trades, f"TP{tp}_SL{sl}")

        if n_perms > 0 and model_metrics['net_pnl_ticks'] > 0:
            print(f"\n  Running {n_perms} permutation trials (model PnL={model_metrics['net_pnl_ticks']:.0f}t > 0)...")
            random_pnls = run_permutation(
                matched, tp, sl, hold_seconds, cancel_seconds,
                signal_threshold, max_concurrent, n_perms=n_perms)

            p_value = float(np.mean(random_pnls >= model_metrics['net_pnl_ticks']))
            model_metrics['p_value'] = p_value
            model_metrics['random_mean_pnl'] = float(np.mean(random_pnls))
            model_metrics['random_std_pnl'] = float(np.std(random_pnls))
        elif n_perms > 0 and model_metrics['net_pnl_ticks'] <= 0:
            print(f"\n  SKIPPING permutations — model PnL={model_metrics['net_pnl_ticks']:.0f}t <= 0 (already unprofitable)")
            model_metrics['p_value'] = 1.0  # Not significant by definition
            model_metrics['random_mean_pnl'] = 0
            model_metrics['random_std_pnl'] = 0
        else:
            model_metrics['p_value'] = -1
            model_metrics['random_mean_pnl'] = 0
            model_metrics['random_std_pnl'] = 0

        model_metrics['tp_ticks'] = tp
        model_metrics['sl_ticks'] = sl

        config_elapsed = time.time() - config_t0

        sig = ""
        if model_metrics['p_value'] >= 0:
            sig = ("***" if model_metrics['p_value'] < 0.01
                   else "**" if model_metrics['p_value'] < 0.05
                   else "*" if model_metrics['p_value'] < 0.10 else "")

        print(f"\n  RESULT: TP{tp}_SL{sl} ({config_elapsed:.0f}s)")
        print_metrics(model_metrics)
        if model_metrics['p_value'] >= 0:
            print(f"  p-value:  {model_metrics['p_value']:.3f} {sig}")
            print(f"  Random:   mean={model_metrics['random_mean_pnl']:.1f}t, "
                  f"std={model_metrics['random_std_pnl']:.1f}t")

        results.append(model_metrics)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"SWEEP SUMMARY (total: {total_elapsed:.0f}s)")
    print("=" * 70)
    print(f"{'Config':<15} {'Trades':>7} {'PnL(t)':>8} {'PnL($)':>8} "
          f"{'WR':>6} {'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'p-val':>7}")
    print("-" * 85)

    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        sig = ("***" if r.get('p_value',1) < 0.01
               else "**" if r.get('p_value',1) < 0.05 else "")
        print(f"TP{r['tp_ticks']:>2}_SL{r['sl_ticks']:>2}   "
              f"{r['n_trades']:>7} "
              f"{r['net_pnl_ticks']:>8.1f} "
              f"{r['net_pnl_dollars']:>8.0f} "
              f"{r['win_rate']:>5.1%} "
              f"{r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>7.2f} "
              f"{r['sortino']:>8.2f} "
              f"{r.get('p_value',-1):>6.3f}{sig}")

    if output_path is None:
        output_path = DEFAULT_OUTPUT
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    significant = [r for r in results if r.get('p_value', 1) < 0.05]
    if significant:
        print(f"\n*** {len(significant)} configs beat random at p < 0.05 ***")
        for r in sorted(significant, key=lambda x: x['sharpe'], reverse=True):
            print(f"  TP{r['tp_ticks']}_SL{r['sl_ticks']}: "
                  f"Sharpe={r['sharpe']:.2f}, PnL={r['net_pnl_ticks']:.0f}t, "
                  f"p={r['p_value']:.3f}")
    else:
        print("\n*** NO configs beat random at p < 0.05 ***")

    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fast tick-level FIFO replay engine (numba-accelerated)')
    parser.add_argument('--mode', choices=['preprocess', 'test', 'sweep'],
                        required=True,
                        help='preprocess: parse .dbn.zst -> .npz; '
                             'test: single day; sweep: TP/SL grid + permutations')
    parser.add_argument('--mbo-dir', default=DEFAULT_MBO_DIR)
    parser.add_argument('--pred-dir', default=DEFAULT_PRED_DIR)
    parser.add_argument('--preproc-dir', default=DEFAULT_PREPROC_DIR)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--max-days', type=int, default=None)
    parser.add_argument('--pred-only', action='store_true',
                        help='Preprocess only dates with predictions')
    parser.add_argument('--head', type=str, default='pred_log_ret_1s',
                        help='Prediction head to use (pred_log_ret_1s/5s/10s)')

    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--hold', type=float, default=30.0)
    parser.add_argument('--cancel', type=float, default=15.0)
    parser.add_argument('--max-concurrent', type=int, default=1)
    parser.add_argument('--perms', type=int, default=100)

    parser.add_argument('--tp-min', type=int, default=2)
    parser.add_argument('--tp-max', type=int, default=20)
    parser.add_argument('--tp-step', type=int, default=2)
    parser.add_argument('--sl-min', type=int, default=1)
    parser.add_argument('--sl-max', type=int, default=10)
    parser.add_argument('--sl-step', type=int, default=2)

    args = parser.parse_args()

    if args.mode == 'preprocess':
        preprocess_all(
            mbo_dir=args.mbo_dir,
            output_dir=args.preproc_dir,
            max_days=args.max_days,
            pred_dir=args.pred_dir if args.pred_only else None,
        )

    elif args.mode == 'test':
        pred_map = load_predictions(args.pred_dir, head=args.head)
        preproc_map = find_preprocessed(args.preproc_dir)
        matched = match_dates(preproc_map, pred_map)
        if args.max_days:
            matched = matched[:args.max_days]
        if not matched:
            print("ERROR: No matched dates. Run --mode preprocess first.")
            return

        print(f"Testing on {len(matched)} day(s)")
        if HAS_NUMBA:
            print("Warming up numba JIT...")

        all_trades = run_one_config(
            matched, tp_ticks=4, sl_ticks=3,
            hold_seconds=args.hold, cancel_seconds=args.cancel,
            signal_threshold=args.threshold,
            max_concurrent=args.max_concurrent, verbose=True)

        metrics = compute_metrics(all_trades, "test")
        print(f"\nResults:")
        print_metrics(metrics)

        if len(all_trades) > 0:
            print(f"\n  First 5 trades:")
            for i in range(min(5, len(all_trades))):
                t = all_trades[i]
                sd = SIDE_NAMES.get(int(t[0]), '?')
                exit_r = EXIT_NAMES.get(int(t[7]), '?')
                lat_ms = t[9] / 1e6
                print(f"    {sd} @ {t[1]:.2f} -> {t[2]:.2f} "
                      f"= {t[10]:+.2f}t ({exit_r}), "
                      f"fill_lat={lat_ms:.0f}ms")

    elif args.mode == 'sweep':
        run_sweep(
            preproc_dir=args.preproc_dir,
            pred_dir=args.pred_dir,
            tp_range=range(args.tp_min, args.tp_max + 1, args.tp_step),
            sl_range=range(args.sl_min, args.sl_max + 1, args.sl_step),
            signal_threshold=args.threshold,
            hold_seconds=args.hold,
            cancel_seconds=args.cancel,
            max_concurrent=args.max_concurrent,
            max_days=args.max_days,
            n_perms=args.perms,
            output_path=args.output,
            head=args.head,
        )


if __name__ == '__main__':
    main()
