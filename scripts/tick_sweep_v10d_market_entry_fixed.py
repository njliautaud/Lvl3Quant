#!/usr/bin/env python3
"""
Tick-Level Replay v10d: MARKET ENTRY with TRADE-ONLY BBO
=========================================================

v10c FAILED due to BBO corruption from deep-book ADD events (prices 688-688475
in MBO data that includes non-ES levels). Market entry at corrupt BBO = disaster.

FIX: BBO is ONLY derived from TRADE events. ES is always 1-tick wide during RTH.
- After agg sell (T sd=ASK at px): bid=px, ask=px+0.25
- After agg buy (T sd=BID at px): ask=px, bid=px-0.25

Entry: IMMEDIATE at trade-anchored BBO when signal fires.
Exit: TP passive (FIFO on trade events), SL/Time market (at trade-BBO).

HC #659 compliant.
"""

import numpy as np
import os
import sys
import time
import json
from numba import njit, int64, float64, int32, int8

sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')

PREPROC_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate'
ALIGNED_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo/pred_indices_aligned'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v10d_market_fixed'

os.makedirs(OUTPUT_DIR, exist_ok=True)

TICK_SIZE = 0.25
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0

EXIT_TP = 0
EXIT_SL = 1
EXIT_TIME = 2
EXIT_EOD = 3

N_TRADE_COLS = 13


@njit(cache=True)
def simulate_day_market_trade_bbo(ts_ns, action, side, price, size,
                                   predictions, pred_event_indices,
                                   tp_ticks, sl_ticks, hold_ns, cancel_ns,
                                   signal_threshold, max_concurrent):
    """
    Market entry simulation with TRADE-ONLY BBO anchoring.

    BBO state ONLY changes on TRADE events. ADD/CANCEL/MODIFY are processed
    ONLY for queue depth at current levels (for TP fill estimation).

    Entry: immediate at BBO when signal fires (must have valid BBO from recent trade).
    TP: passive limit exit (FIFO queue from trade volume).
    SL: triggers when trade price reaches SL level.
    Time: market exit at current trade-BBO.
    """
    n_events = len(ts_ns)
    n_preds = len(predictions)
    tick = 0.25

    # Trade-anchored BBO
    best_bid = 0.0
    best_ask = 0.0
    bbo_valid = False
    last_trade_ts = int64(0)

    # Queue depth tracking (for TP fills)
    bid_depth = 0
    ask_depth = 0

    # Positions
    MAX_POS = 16
    pos_active = np.zeros(MAX_POS, dtype=int8)
    pos_side = np.zeros(MAX_POS, dtype=int8)
    pos_entry_px = np.zeros(MAX_POS, dtype=float64)
    pos_entry_ts = np.zeros(MAX_POS, dtype=int64)
    pos_tp_px = np.zeros(MAX_POS, dtype=float64)
    pos_sl_px = np.zeros(MAX_POS, dtype=float64)
    pos_hold_end = np.zeros(MAX_POS, dtype=int64)
    pos_signal_str = np.zeros(MAX_POS, dtype=float64)
    pos_entry_event = np.zeros(MAX_POS, dtype=int64)
    pos_tp_queue = np.zeros(MAX_POS, dtype=float64)

    max_trades = n_preds * 2 + 100
    trades = np.zeros((max_trades, N_TRADE_COLS), dtype=float64)
    trade_count = 0
    pred_idx = 0
    n_active = 0

    ACT_TRADE = 3
    SIDE_BID = 0
    SIDE_ASK = 1

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

        # ===== ONLY process BBO/fills/SL on TRADE events =====
        if act == ACT_TRADE:
            last_trade_ts = ts

            if sd == SIDE_ASK:
                # Aggressive sell hit bid at px
                best_bid = px
                best_ask = px + tick
                bbo_valid = True
            elif sd == SIDE_BID:
                # Aggressive buy lifted ask at px
                best_ask = px
                best_bid = px - tick
                bbo_valid = True

            # --- Check TP fills ---
            for pi in range(MAX_POS):
                if pos_active[pi] == 0:
                    continue

                if pos_side[pi] == 1:  # Short: TP = buy at bid (needs agg sell T sd=ASK)
                    if sd == SIDE_ASK:
                        if px <= pos_tp_px[pi]:
                            # TP hit (trade at or below our TP level)
                            pos_tp_queue[pi] -= sz
                            if pos_tp_queue[pi] <= 0 or px < pos_tp_px[pi]:
                                # Filled
                                entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                                exit_cost = COMMISSION_RT_TICKS  # passive
                                pnl = (pos_entry_px[pi] - pos_tp_px[pi]) / tick - entry_cost - exit_cost
                                if trade_count < max_trades:
                                    trades[trade_count, 0] = pos_entry_ts[pi]
                                    trades[trade_count, 1] = ts
                                    trades[trade_count, 2] = pos_entry_px[pi]
                                    trades[trade_count, 3] = pos_tp_px[pi]
                                    trades[trade_count, 4] = 1
                                    trades[trade_count, 5] = pos_signal_str[pi]
                                    trades[trade_count, 6] = pos_entry_event[pi]
                                    trades[trade_count, 7] = evt_i
                                    trades[trade_count, 8] = EXIT_TP
                                    trades[trade_count, 9] = ts - pos_entry_ts[pi]
                                    trades[trade_count, 10] = pnl
                                    trades[trade_count, 11] = entry_cost
                                    trades[trade_count, 12] = exit_cost
                                    trade_count += 1
                                pos_active[pi] = 0
                                n_active -= 1
                                continue

                elif pos_side[pi] == 0:  # Long: TP = sell at ask (needs agg buy T sd=BID)
                    if sd == SIDE_BID:
                        if px >= pos_tp_px[pi]:
                            pos_tp_queue[pi] -= sz
                            if pos_tp_queue[pi] <= 0 or px > pos_tp_px[pi]:
                                entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                                exit_cost = COMMISSION_RT_TICKS
                                pnl = (pos_tp_px[pi] - pos_entry_px[pi]) / tick - entry_cost - exit_cost
                                if trade_count < max_trades:
                                    trades[trade_count, 0] = pos_entry_ts[pi]
                                    trades[trade_count, 1] = ts
                                    trades[trade_count, 2] = pos_entry_px[pi]
                                    trades[trade_count, 3] = pos_tp_px[pi]
                                    trades[trade_count, 4] = 0
                                    trades[trade_count, 5] = pos_signal_str[pi]
                                    trades[trade_count, 6] = pos_entry_event[pi]
                                    trades[trade_count, 7] = evt_i
                                    trades[trade_count, 8] = EXIT_TP
                                    trades[trade_count, 9] = ts - pos_entry_ts[pi]
                                    trades[trade_count, 10] = pnl
                                    trades[trade_count, 11] = entry_cost
                                    trades[trade_count, 12] = exit_cost
                                    trade_count += 1
                                pos_active[pi] = 0
                                n_active -= 1
                                continue

            # --- Check SL (only on trade events, using trade price) ---
            for pi in range(MAX_POS):
                if pos_active[pi] == 0:
                    continue

                if pos_side[pi] == 1:  # Short: SL if trade goes above SL level
                    if px >= pos_sl_px[pi]:
                        entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                        exit_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                        # Exit at ask (= trade_px if agg buy, or trade_px + tick if agg sell)
                        exit_px = best_ask
                        pnl = (pos_entry_px[pi] - exit_px) / tick - entry_cost - exit_cost
                        if trade_count < max_trades:
                            trades[trade_count, 0] = pos_entry_ts[pi]
                            trades[trade_count, 1] = ts
                            trades[trade_count, 2] = pos_entry_px[pi]
                            trades[trade_count, 3] = exit_px
                            trades[trade_count, 4] = 1
                            trades[trade_count, 5] = pos_signal_str[pi]
                            trades[trade_count, 6] = pos_entry_event[pi]
                            trades[trade_count, 7] = evt_i
                            trades[trade_count, 8] = EXIT_SL
                            trades[trade_count, 9] = ts - pos_entry_ts[pi]
                            trades[trade_count, 10] = pnl
                            trades[trade_count, 11] = entry_cost
                            trades[trade_count, 12] = exit_cost
                            trade_count += 1
                        pos_active[pi] = 0
                        n_active -= 1

                elif pos_side[pi] == 0:  # Long: SL if trade goes below SL level
                    if px <= pos_sl_px[pi]:
                        entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                        exit_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                        exit_px = best_bid
                        pnl = (exit_px - pos_entry_px[pi]) / tick - entry_cost - exit_cost
                        if trade_count < max_trades:
                            trades[trade_count, 0] = pos_entry_ts[pi]
                            trades[trade_count, 1] = ts
                            trades[trade_count, 2] = pos_entry_px[pi]
                            trades[trade_count, 3] = exit_px
                            trades[trade_count, 4] = 0
                            trades[trade_count, 5] = pos_signal_str[pi]
                            trades[trade_count, 6] = pos_entry_event[pi]
                            trades[trade_count, 7] = evt_i
                            trades[trade_count, 8] = EXIT_SL
                            trades[trade_count, 9] = ts - pos_entry_ts[pi]
                            trades[trade_count, 10] = pnl
                            trades[trade_count, 11] = entry_cost
                            trades[trade_count, 12] = exit_cost
                            trade_count += 1
                        pos_active[pi] = 0
                        n_active -= 1

        # --- Check time stop (on every event for time precision) ---
        for pi in range(MAX_POS):
            if pos_active[pi] == 0:
                continue
            if ts >= pos_hold_end[pi] and bbo_valid:
                entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                exit_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                if pos_side[pi] == 1:
                    exit_px = best_ask
                    pnl = (pos_entry_px[pi] - exit_px) / tick - entry_cost - exit_cost
                else:
                    exit_px = best_bid
                    pnl = (exit_px - pos_entry_px[pi]) / tick - entry_cost - exit_cost
                if trade_count < max_trades:
                    trades[trade_count, 0] = pos_entry_ts[pi]
                    trades[trade_count, 1] = ts
                    trades[trade_count, 2] = pos_entry_px[pi]
                    trades[trade_count, 3] = exit_px
                    trades[trade_count, 4] = pos_side[pi]
                    trades[trade_count, 5] = pos_signal_str[pi]
                    trades[trade_count, 6] = pos_entry_event[pi]
                    trades[trade_count, 7] = evt_i
                    trades[trade_count, 8] = EXIT_TIME
                    trades[trade_count, 9] = ts - pos_entry_ts[pi]
                    trades[trade_count, 10] = pnl
                    trades[trade_count, 11] = entry_cost
                    trades[trade_count, 12] = exit_cost
                    trade_count += 1
                pos_active[pi] = 0
                n_active -= 1

        # --- Check for new signal ---
        if pred_idx < n_preds and evt_i == pred_event_indices[pred_idx]:
            pred_val = predictions[pred_idx]
            pred_idx += 1

            if not bbo_valid:
                continue
            if n_active >= max_concurrent:
                continue
            # Require recent trade (within 5s) for valid BBO
            if ts - last_trade_ts > 5000000000:
                continue

            enter = False
            if pred_val < -signal_threshold:
                # Short: sell at bid (market order hits bid)
                entry_px = best_bid
                trade_side = int8(1)
                tp_px = entry_px - tp_ticks * tick
                sl_px_val = entry_px + sl_ticks * tick
                tp_queue = float64(100)  # Estimate initial queue
                enter = True
            elif pred_val > signal_threshold:
                # Long: buy at ask (market order lifts ask)
                entry_px = best_ask
                trade_side = int8(0)
                tp_px = entry_px + tp_ticks * tick
                sl_px_val = entry_px - sl_ticks * tick
                tp_queue = float64(100)
                enter = True

            if enter:
                for pi in range(MAX_POS):
                    if pos_active[pi] == 0:
                        pos_active[pi] = 1
                        pos_side[pi] = trade_side
                        pos_entry_px[pi] = entry_px
                        pos_entry_ts[pi] = ts
                        pos_tp_px[pi] = tp_px
                        pos_sl_px[pi] = sl_px_val
                        pos_hold_end[pi] = ts + hold_ns
                        pos_signal_str[pi] = abs(pred_val)
                        pos_entry_event[pi] = evt_i
                        pos_tp_queue[pi] = tp_queue
                        n_active += 1
                        break

    # EOD close remaining
    if n_events > 0 and bbo_valid:
        final_ts = ts_ns[n_events - 1]
        for pi in range(MAX_POS):
            if pos_active[pi] == 0:
                continue
            entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
            exit_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
            if pos_side[pi] == 1:
                exit_px = best_ask
                pnl = (pos_entry_px[pi] - exit_px) / tick - entry_cost - exit_cost
            else:
                exit_px = best_bid
                pnl = (exit_px - pos_entry_px[pi]) / tick - entry_cost - exit_cost
            if trade_count < max_trades:
                trades[trade_count, 0] = pos_entry_ts[pi]
                trades[trade_count, 1] = final_ts
                trades[trade_count, 2] = pos_entry_px[pi]
                trades[trade_count, 3] = exit_px
                trades[trade_count, 4] = pos_side[pi]
                trades[trade_count, 5] = pos_signal_str[pi]
                trades[trade_count, 6] = pos_entry_event[pi]
                trades[trade_count, 7] = n_events - 1
                trades[trade_count, 8] = EXIT_EOD
                trades[trade_count, 9] = final_ts - pos_entry_ts[pi]
                trades[trade_count, 10] = pnl
                trades[trade_count, 11] = entry_cost
                trades[trade_count, 12] = exit_cost
                trade_count += 1
            pos_active[pi] = 0

    return trades[:trade_count]


def load_day_data(date_str):
    """Load MBO + aligned predictions."""
    aligned_path = os.path.join(ALIGNED_DIR, f'aligned_{date_str}.npz')
    pred_path = os.path.join(PRED_DIR, f'oot_{date_str}.npz')
    preproc_path = os.path.join(PREPROC_DIR, f'mbo_{date_str}.npz')

    if not all(os.path.exists(p) for p in [aligned_path, pred_path, preproc_path]):
        return None

    mbo = np.load(preproc_path)
    ts_ns = mbo['ts_ns']
    action = mbo['action']
    side = mbo['side']
    price = mbo['price']
    size = mbo['size'].astype(np.int32)

    aligned = np.load(aligned_path)
    event_indices = aligned['event_indices']
    in_range = aligned['in_mbo_range']

    pred_file = np.load(pred_path)
    if 'pred_log_ret_1s' in pred_file:
        preds = pred_file['pred_log_ret_1s'].astype(np.float64)
    elif 'predictions' in pred_file:
        preds = pred_file['predictions'].astype(np.float64)
    else:
        return None

    valid_preds = preds[in_range]
    valid_indices = event_indices[in_range].astype(np.int64)

    n_events = len(ts_ns)
    bounds_mask = (valid_indices >= 0) & (valid_indices < n_events)
    valid_preds = valid_preds[bounds_mask]
    valid_indices = valid_indices[bounds_mask]

    sort_idx = np.argsort(valid_indices)
    valid_preds = valid_preds[sort_idx]
    valid_indices = valid_indices[sort_idx]

    return {
        'ts_ns': ts_ns, 'action': action, 'side': side,
        'price': price, 'size': size,
        'predictions': valid_preds, 'event_indices': valid_indices,
    }


def run_config(day_data_list, tp_ticks, sl_ticks, hold_seconds, cancel_seconds,
               signal_threshold, max_concurrent, short_only, long_only):
    """Run config across all dates."""
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)
    all_trades = []
    daily_pnls = []

    for dd in day_data_list:
        preds = dd['predictions'].copy()
        indices = dd['event_indices'].copy()

        if short_only:
            preds[preds > 0] = 0.0
        elif long_only:
            preds[preds < 0] = 0.0

        trades = simulate_day_market_trade_bbo(
            dd['ts_ns'], dd['action'], dd['side'], dd['price'], dd['size'],
            preds, indices,
            tp_ticks, sl_ticks, hold_ns, cancel_ns,
            signal_threshold, max_concurrent
        )

        if len(trades) > 0:
            all_trades.append(trades)
            daily_pnls.append(trades[:, 10].sum())
        else:
            daily_pnls.append(0.0)

    return all_trades, np.array(daily_pnls)


def permutation_test(day_data_list, tp_ticks, sl_ticks, hold_seconds, cancel_seconds,
                     signal_threshold, max_concurrent, short_only, long_only,
                     n_perms=100, real_sharpe=0.0):
    """HC #659 R3: Permutation test."""
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)
    random_sharpes = []

    for _ in range(n_perms):
        daily_pnls = []
        for dd in day_data_list:
            preds = dd['predictions'].copy()
            signs = np.random.choice([-1.0, 1.0], size=len(preds))
            preds = preds * signs
            if short_only:
                preds[preds > 0] = 0.0
            elif long_only:
                preds[preds < 0] = 0.0

            trades = simulate_day_market_trade_bbo(
                dd['ts_ns'], dd['action'], dd['side'], dd['price'], dd['size'],
                preds, dd['event_indices'].copy(),
                tp_ticks, sl_ticks, hold_ns, cancel_ns,
                signal_threshold, max_concurrent
            )
            daily_pnls.append(trades[:, 10].sum() if len(trades) > 0 else 0.0)

        daily_arr = np.array(daily_pnls)
        s = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0.0
        random_sharpes.append(s)

    random_sharpes = np.array(random_sharpes)
    return (random_sharpes >= real_sharpe).mean(), random_sharpes.mean(), random_sharpes.std()


def main():
    print("=" * 70, flush=True)
    print("TICK REPLAY v10d — MARKET ENTRY, TRADE-ONLY BBO", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)
    print("Fix: BBO anchored ONLY from TRADE events (not ADD/CANCEL).", flush=True)
    print("SL checked ONLY on trade events (real price, not book noise).", flush=True)
    print("Cost: entry 1.376t (market) + TP exit 0.376t / SL exit 1.376t", flush=True)
    print(flush=True)

    dates = sorted([f[8:16] for f in os.listdir(ALIGNED_DIR) if f.startswith('aligned_')])
    print(f"Loading {len(dates)} dates...", flush=True)

    day_data_list = []
    for date_str in dates:
        dd = load_day_data(date_str)
        if dd is not None:
            if len(dd['predictions']) < 1000:
                continue
            day_data_list.append(dd)

    print(f"Loaded {len(day_data_list)} trading dates", flush=True)
    total_preds = sum(len(dd['predictions']) for dd in day_data_list)
    print(f"Total predictions: {total_preds:,}", flush=True)
    print(flush=True)

    # JIT warmup
    print("Warming up Numba JIT...", flush=True)
    d = day_data_list[0]
    _ = simulate_day_market_trade_bbo(
        d['ts_ns'][:5000], d['action'][:5000], d['side'][:5000],
        d['price'][:5000], d['size'][:5000],
        d['predictions'][:5], d['event_indices'][:5],
        3, 4, int(5e9), int(5e9), 0.8, 4
    )
    print("JIT ready.", flush=True)
    print(flush=True)

    # Configs — need TP ≥ 3 ticks to cover 1.752 min RT cost
    configs = [
        # Short-only (strongest edge from signal decay analysis)
        (3, 3, 5, 5, 0.6, True, False, 4, "S_TP3_SL3_h5_thr0.6"),
        (3, 4, 5, 5, 0.6, True, False, 4, "S_TP3_SL4_h5_thr0.6"),
        (4, 4, 8, 5, 0.6, True, False, 4, "S_TP4_SL4_h8_thr0.6"),
        (4, 6, 10, 8, 0.6, True, False, 4, "S_TP4_SL6_h10_thr0.6"),
        (3, 3, 5, 5, 0.8, True, False, 4, "S_TP3_SL3_h5_thr0.8"),
        (3, 4, 5, 5, 0.8, True, False, 4, "S_TP3_SL4_h5_thr0.8"),
        (4, 4, 8, 5, 0.8, True, False, 4, "S_TP4_SL4_h8_thr0.8"),
        (4, 6, 10, 8, 0.8, True, False, 4, "S_TP4_SL6_h10_thr0.8"),
        (5, 6, 12, 8, 0.8, True, False, 4, "S_TP5_SL6_h12_thr0.8"),
        (3, 3, 5, 5, 1.0, True, False, 4, "S_TP3_SL3_h5_thr1.0"),
        (3, 4, 5, 5, 1.0, True, False, 4, "S_TP3_SL4_h5_thr1.0"),
        (4, 4, 8, 5, 1.0, True, False, 4, "S_TP4_SL4_h8_thr1.0"),
        (4, 6, 10, 8, 1.0, True, False, 4, "S_TP4_SL6_h10_thr1.0"),
        (5, 6, 12, 8, 1.0, True, False, 4, "S_TP5_SL6_h12_thr1.0"),
        # Ultra-selective
        (3, 4, 5, 5, 1.2, True, False, 4, "S_TP3_SL4_h5_thr1.2"),
        (4, 6, 10, 8, 1.2, True, False, 4, "S_TP4_SL6_h10_thr1.2"),
        # Long-only
        (3, 4, 5, 5, 0.8, False, True, 4, "L_TP3_SL4_h5_thr0.8"),
        (4, 6, 10, 8, 0.8, False, True, 4, "L_TP4_SL6_h10_thr0.8"),
        # Both
        (3, 4, 5, 5, 0.8, False, False, 4, "B_TP3_SL4_h5_thr0.8"),
        (4, 6, 10, 8, 1.0, False, False, 4, "B_TP4_SL6_h10_thr1.0"),
    ]

    print(f"Testing {len(configs)} configs", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    results = {}
    promising = []

    for ci, (tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label) in enumerate(configs):
        t0 = time.time()
        all_trades, daily_pnls = run_config(
            day_data_list, tp, sl, hold_s, cancel_s, thr, max_conc, short_only, long_only
        )
        elapsed = time.time() - t0

        if not all_trades:
            print(f"[{ci+1}/{len(configs)}] {label}: NO TRADES [{elapsed:.1f}s]", flush=True)
            results[label] = {'n_trades': 0, 'status': 'no_trades'}
            continue

        trades = np.vstack(all_trades)
        n_trades = len(trades)
        n_per_day = n_trades / len(day_data_list)

        pnl = trades[:, 10]
        total_pnl = pnl.sum()
        winners = pnl > 0
        wr = winners.mean() if len(pnl) > 0 else 0

        gross_win = pnl[winners].sum() if winners.any() else 0
        gross_loss = abs(pnl[~winners].sum()) if (~winners).any() else 1e-9
        pf = gross_win / gross_loss

        sharpe = (daily_pnls.mean() / daily_pnls.std() * np.sqrt(252)) if daily_pnls.std() > 0 else 0
        down = daily_pnls[daily_pnls < 0]
        sortino = (daily_pnls.mean() / down.std() * np.sqrt(252)) if len(down) > 0 and down.std() > 0 else 0

        green = int((daily_pnls > 0).sum())
        red = int((daily_pnls < 0).sum())

        exits = trades[:, 8]
        tp_pct = (exits == EXIT_TP).mean() * 100
        sl_pct = (exits == EXIT_SL).mean() * 100
        time_pct = (exits == EXIT_TIME).mean() * 100

        # Sanity: avg PnL should be bounded by SL
        avg_pnl = pnl.mean()
        max_loss = -(sl + 2.752)  # SL + max cost

        results[label] = {
            'n_trades': int(n_trades), 'per_day': round(float(n_per_day), 1),
            'total_pnl_ticks': round(float(total_pnl), 1),
            'avg_pnl': round(float(avg_pnl), 3),
            'wr': round(float(wr), 4), 'pf': round(float(pf), 3),
            'sharpe': round(float(sharpe), 2), 'sortino': round(float(sortino), 2),
            'green_days': green, 'red_days': red,
            'tp_pct': round(float(tp_pct), 1),
            'sl_pct': round(float(sl_pct), 1),
            'time_pct': round(float(time_pct), 1),
        }

        status = "✓" if sharpe > 0 and pf > 1.0 else "✗"
        print(f"[{ci+1}/{len(configs)}] {status} {label}", flush=True)
        print(f"  {n_trades} trades ({n_per_day:.0f}/day), PnL={total_pnl:+.1f}t, avg={avg_pnl:+.3f}t "
              f"WR={wr:.3f}, PF={pf:.3f}, Sharpe={sharpe:+.2f}", flush=True)
        print(f"  Exits: TP={tp_pct:.0f}% SL={sl_pct:.0f}% Time={time_pct:.0f}% | "
              f"Days: {green}G/{red}R [{elapsed:.1f}s]", flush=True)

        # Sanity check
        if avg_pnl < max_loss:
            print(f"  ⚠️ SANITY FAIL: avg_pnl {avg_pnl:.1f} < max_possible_loss {max_loss:.1f}", flush=True)
        print(flush=True)

        if sharpe > 0.5 and pf > 1.0 and wr > 0.45:
            promising.append((ci, tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label))

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'v10d_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Permutation test on promising
    if promising:
        print(flush=True)
        print("=" * 70, flush=True)
        print(f"PERMUTATION TEST — {len(promising)} configs", flush=True)
        print("=" * 70, flush=True)
        print(flush=True)

        for ci, tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label in promising:
            real_sharpe = results[label]['sharpe']
            print(f"Testing {label} (Sharpe={real_sharpe:+.2f})...", flush=True)
            t0 = time.time()
            p_val, mean_rand, std_rand = permutation_test(
                day_data_list, tp, sl, hold_s, cancel_s, thr, max_conc,
                short_only, long_only, n_perms=100, real_sharpe=real_sharpe
            )
            elapsed = time.time() - t0
            results[label]['perm_p_value'] = round(float(p_val), 4)
            results[label]['perm_random_mean'] = round(float(mean_rand), 2)
            results[label]['perm_random_std'] = round(float(std_rand), 2)
            verdict = "REAL EDGE ✓" if p_val < 0.05 else "ARTIFACT ✗"
            print(f"  p={p_val:.3f} | Random: {mean_rand:+.2f}±{std_rand:.2f} | {verdict} [{elapsed:.0f}s]", flush=True)
            print(flush=True)

    # Final table
    print(flush=True)
    print("=" * 70, flush=True)
    print("FINAL SUMMARY (sorted by Sharpe)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Config':<28} {'N':>6} {'PnL':>8} {'Avg':>7} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'G/R':>5} {'p':>6}", flush=True)
    print("-" * 80, flush=True)
    sorted_r = sorted(
        [(k, v) for k, v in results.items() if v.get('n_trades', 0) > 0],
        key=lambda x: x[1].get('sharpe', -999), reverse=True
    )
    for label, r in sorted_r:
        p_str = f"{r['perm_p_value']:.3f}" if 'perm_p_value' in r else "  —"
        print(f"{label:<28} {r['n_trades']:>6} {r['total_pnl_ticks']:>+8.0f} "
              f"{r['avg_pnl']:>+7.3f} {r['wr']:>6.3f} {r['pf']:>6.3f} "
              f"{r['sharpe']:>+7.2f} {r['green_days']}/{r['red_days']:>2} {p_str:>6}", flush=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}", flush=True)

    validated = [k for k, v in results.items()
                 if v.get('perm_p_value', 1.0) < 0.05 and v.get('sharpe', 0) > 0.5]
    if validated:
        print(f"\n🎯 VALIDATED: {validated}", flush=True)
    else:
        print("\n❌ No validated configs.", flush=True)


if __name__ == "__main__":
    main()
