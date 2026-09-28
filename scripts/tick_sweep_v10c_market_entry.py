#!/usr/bin/env python3
"""
Tick-Level Replay v10c: MARKET ENTRY (immediate fill, pay spread)
=================================================================

v10b proved: FIFO passive entry kills ALL edge. By the time you fill at
back-of-queue, the predicted move already happened (adverse selection).

This test: MARKET ENTRY. You pay 1 tick spread to enter immediately at the
signal. If model's top shorts predict +1.05 ticks MFE (per alignment validation),
and market entry costs 1 tick, top signals might still net-positive.

Entry: IMMEDIATE at signal event timestamp.
  - Short: enter at BID (sell market = hit bid = 1 tick below ask)
  - Long: enter at ASK (buy market = lift ask = 1 tick above bid)
Exit:
  - TP: passive limit at entry ± tp_ticks (join queue, track FIFO)
  - SL: market order at entry ∓ sl_ticks (immediate)
  - Time stop: market exit after hold_seconds

Cost model:
  - Entry: 1.376 ticks (commission + 1 tick spread crossing)
  - TP exit: 0.376 ticks (passive, commission only)
  - SL/Time exit: 1.376 ticks (market)
  So total RT cost: entry 1.376 + exit 0.376-1.376 = 1.752-2.752 ticks

HC #659: tick-level replay, permutation test on positive results.
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
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/tick_replay_v10c_market_entry'

os.makedirs(OUTPUT_DIR, exist_ok=True)

TICK_SIZE = 0.25
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0

# Exit codes
EXIT_TP = 0
EXIT_SL = 1
EXIT_TIME = 2
EXIT_EOD = 3

# Trade columns: [entry_ts, exit_ts, entry_px, exit_px, side, signal_str,
#                 entry_event, exit_event, exit_reason, hold_ns, pnl_ticks,
#                 entry_cost, exit_cost]
N_TRADE_COLS = 13


@njit(cache=True)
def simulate_day_market_entry(ts_ns, action, side, price, size,
                              predictions, pred_event_indices,
                              tp_ticks, sl_ticks, hold_ns, cancel_ns,
                              signal_threshold, max_concurrent):
    """
    Market entry tick simulation.

    Entry: immediate at signal, at current BBO (short = sell at bid, long = buy at ask).
    TP exit: passive limit (wait for fill via FIFO queue).
    SL/Time exit: market order (immediate).
    """
    n_events = len(ts_ns)
    n_preds = len(predictions)

    best_bid = 0.0
    best_ask = 0.0
    bid_size = 0
    ask_size = 0

    # Position tracking
    MAX_POS = 16
    pos_active = np.zeros(MAX_POS, dtype=int8)
    pos_side = np.zeros(MAX_POS, dtype=int8)  # 0=long, 1=short
    pos_entry_px = np.zeros(MAX_POS, dtype=float64)
    pos_entry_ts = np.zeros(MAX_POS, dtype=int64)
    pos_tp_px = np.zeros(MAX_POS, dtype=float64)
    pos_sl_px = np.zeros(MAX_POS, dtype=float64)
    pos_hold_end = np.zeros(MAX_POS, dtype=int64)
    pos_signal_str = np.zeros(MAX_POS, dtype=float64)
    pos_entry_event = np.zeros(MAX_POS, dtype=int64)
    pos_tp_queue = np.zeros(MAX_POS, dtype=float64)  # FIFO queue for TP

    # Results
    max_trades = n_preds + 100
    trades = np.zeros((max_trades, N_TRADE_COLS), dtype=float64)
    trade_count = 0

    pred_idx = 0
    n_active = 0

    ACT_ADD = 0
    ACT_CANCEL = 1
    ACT_MODIFY = 2
    ACT_TRADE = 3
    ACT_FILL = 4

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

        # Update BBO
        if act == ACT_TRADE:
            if sd == SIDE_ASK:  # Aggressive sell hit bid
                best_bid = px
                if best_ask < px + TICK_SIZE or best_ask == 0:
                    best_ask = px + TICK_SIZE
                bid_size = max(bid_size - sz, 0)
            elif sd == SIDE_BID:  # Aggressive buy lifted ask
                best_ask = px
                if best_bid > px - TICK_SIZE or best_bid == 0:
                    best_bid = px - TICK_SIZE
                ask_size = max(ask_size - sz, 0)

            # Check TP fills for existing positions (passive TP via FIFO)
            for pi in range(MAX_POS):
                if pos_active[pi] == 0:
                    continue
                if pos_side[pi] == 1:  # Short position, TP = buy at bid (need agg sell)
                    if sd == SIDE_ASK and abs(px - pos_tp_px[pi]) < 0.001:
                        pos_tp_queue[pi] -= sz
                        if pos_tp_queue[pi] <= 0:
                            # TP filled (passive exit)
                            exit_cost = COMMISSION_RT_TICKS  # passive only commission
                            entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS  # market entry
                            pnl = (pos_entry_px[pi] - pos_tp_px[pi]) / TICK_SIZE - entry_cost - exit_cost
                            if trade_count < max_trades:
                                trades[trade_count, 0] = pos_entry_ts[pi]
                                trades[trade_count, 1] = ts
                                trades[trade_count, 2] = pos_entry_px[pi]
                                trades[trade_count, 3] = pos_tp_px[pi]
                                trades[trade_count, 4] = 1  # short
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
                    elif sd == SIDE_ASK and px < pos_tp_px[pi]:
                        # Trade through TP level -> filled
                        exit_cost = COMMISSION_RT_TICKS
                        entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                        pnl = (pos_entry_px[pi] - pos_tp_px[pi]) / TICK_SIZE - entry_cost - exit_cost
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

                elif pos_side[pi] == 0:  # Long position, TP = sell at ask (need agg buy)
                    if sd == SIDE_BID and abs(px - pos_tp_px[pi]) < 0.001:
                        pos_tp_queue[pi] -= sz
                        if pos_tp_queue[pi] <= 0:
                            exit_cost = COMMISSION_RT_TICKS
                            entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                            pnl = (pos_tp_px[pi] - pos_entry_px[pi]) / TICK_SIZE - entry_cost - exit_cost
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
                    elif sd == SIDE_BID and px > pos_tp_px[pi]:
                        exit_cost = COMMISSION_RT_TICKS
                        entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                        pnl = (pos_tp_px[pi] - pos_entry_px[pi]) / TICK_SIZE - entry_cost - exit_cost
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

        elif act == ACT_ADD:
            if sd == SIDE_BID:
                if px > best_bid or best_bid == 0:
                    best_bid = px
                    bid_size = sz
                elif abs(px - best_bid) < 0.001:
                    bid_size += sz
            elif sd == SIDE_ASK:
                if px < best_ask or best_ask == 0:
                    best_ask = px
                    ask_size = sz
                elif abs(px - best_ask) < 0.001:
                    ask_size += sz
        elif act == ACT_CANCEL:
            if sd == SIDE_BID and abs(px - best_bid) < 0.001:
                bid_size -= sz
                if bid_size <= 0:
                    best_bid -= TICK_SIZE
                    bid_size = 0
            elif sd == SIDE_ASK and abs(px - best_ask) < 0.001:
                ask_size -= sz
                if ask_size <= 0:
                    best_ask += TICK_SIZE
                    ask_size = 0

        # Check SL / Time stop for active positions
        for pi in range(MAX_POS):
            if pos_active[pi] == 0:
                continue

            # Time stop
            if ts >= pos_hold_end[pi]:
                # Market exit at current price
                entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                exit_cost = COMMISSION_RT_TICKS + SPREAD_TICKS  # market exit
                if pos_side[pi] == 1:  # short
                    exit_px = best_ask if best_ask > 0 else pos_entry_px[pi]
                    pnl = (pos_entry_px[pi] - exit_px) / TICK_SIZE - entry_cost - exit_cost
                else:  # long
                    exit_px = best_bid if best_bid > 0 else pos_entry_px[pi]
                    pnl = (exit_px - pos_entry_px[pi]) / TICK_SIZE - entry_cost - exit_cost

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
                continue

            # SL check
            if pos_side[pi] == 1:  # short: SL if ask rises above entry + sl
                sl_level = pos_entry_px[pi] + sl_ticks * TICK_SIZE
                if best_ask > 0 and best_ask >= sl_level:
                    entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                    exit_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                    exit_px = best_ask
                    pnl = (pos_entry_px[pi] - exit_px) / TICK_SIZE - entry_cost - exit_cost
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
            else:  # long: SL if bid falls below entry - sl
                sl_level = pos_entry_px[pi] - sl_ticks * TICK_SIZE
                if best_bid > 0 and best_bid <= sl_level:
                    entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                    exit_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
                    exit_px = best_bid
                    pnl = (exit_px - pos_entry_px[pi]) / TICK_SIZE - entry_cost - exit_cost
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

        # Check for new signal at this event
        if pred_idx < n_preds and evt_i == pred_event_indices[pred_idx]:
            pred_val = predictions[pred_idx]
            pred_idx += 1

            if n_active >= max_concurrent:
                continue
            if best_bid <= 0 or best_ask <= 0:
                continue

            # Determine direction and entry
            enter = False
            if pred_val < -signal_threshold:
                # Short signal: market sell at bid
                entry_px = best_bid
                trade_side = int8(1)
                tp_px = entry_px - tp_ticks * TICK_SIZE
                sl_px = entry_px + sl_ticks * TICK_SIZE
                # For TP: we're buying back at tp_px (passive buy at bid=tp_px)
                # Queue = current ask_size if tp_px is at ask, else estimate
                tp_queue = float64(bid_size) if abs(tp_px - best_bid) < 0.001 else 100.0
                enter = True
            elif pred_val > signal_threshold:
                # Long signal: market buy at ask
                entry_px = best_ask
                trade_side = int8(0)
                tp_px = entry_px + tp_ticks * TICK_SIZE
                sl_px = entry_px - sl_ticks * TICK_SIZE
                tp_queue = float64(ask_size) if abs(tp_px - best_ask) < 0.001 else 100.0
                enter = True

            if enter:
                # Find slot
                for pi in range(MAX_POS):
                    if pos_active[pi] == 0:
                        pos_active[pi] = 1
                        pos_side[pi] = trade_side
                        pos_entry_px[pi] = entry_px
                        pos_entry_ts[pi] = ts
                        pos_tp_px[pi] = tp_px
                        pos_sl_px[pi] = sl_px
                        pos_hold_end[pi] = ts + hold_ns
                        pos_signal_str[pi] = abs(pred_val)
                        pos_entry_event[pi] = evt_i
                        pos_tp_queue[pi] = tp_queue
                        n_active += 1
                        break

    # Close any remaining positions at EOD (market exit)
    if n_events > 0:
        final_ts = ts_ns[n_events - 1]
        for pi in range(MAX_POS):
            if pos_active[pi] == 0:
                continue
            entry_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
            exit_cost = COMMISSION_RT_TICKS + SPREAD_TICKS
            if pos_side[pi] == 1:
                exit_px = best_ask if best_ask > 0 else pos_entry_px[pi]
                pnl = (pos_entry_px[pi] - exit_px) / TICK_SIZE - entry_cost - exit_cost
            else:
                exit_px = best_bid if best_bid > 0 else pos_entry_px[pi]
                pnl = (exit_px - pos_entry_px[pi]) / TICK_SIZE - entry_cost - exit_cost
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
    """Load MBO + aligned predictions for a date."""
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
    order_id = mbo['order_id']

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

    # Sort by index
    sort_idx = np.argsort(valid_indices)
    valid_preds = valid_preds[sort_idx]
    valid_indices = valid_indices[sort_idx]

    return {
        'ts_ns': ts_ns, 'action': action, 'side': side,
        'price': price, 'size': size, 'order_id': order_id,
        'predictions': valid_preds, 'event_indices': valid_indices,
    }


def run_config(day_data_list, tp_ticks, sl_ticks, hold_seconds, cancel_seconds,
               signal_threshold, max_concurrent, short_only, long_only):
    """Run market-entry config across all dates."""
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)

    all_trades = []
    daily_pnls = []

    for dd in day_data_list:
        preds = dd['predictions'].copy()
        indices = dd['event_indices'].copy()

        # Direction filter: zero out unwanted directions
        if short_only:
            preds[preds > 0] = 0.0
        elif long_only:
            preds[preds < 0] = 0.0

        trades = simulate_day_market_entry(
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
    """HC #659 R3: Random direction permutation test."""
    hold_ns = int(hold_seconds * 1e9)
    cancel_ns = int(cancel_seconds * 1e9)
    random_sharpes = []

    for _ in range(n_perms):
        daily_pnls = []
        for dd in day_data_list:
            preds = dd['predictions'].copy()
            # Random sign flip
            signs = np.random.choice([-1.0, 1.0], size=len(preds))
            preds = preds * signs

            if short_only:
                preds[preds > 0] = 0.0
            elif long_only:
                preds[preds < 0] = 0.0

            indices = dd['event_indices'].copy()

            trades = simulate_day_market_entry(
                dd['ts_ns'], dd['action'], dd['side'], dd['price'], dd['size'],
                preds, indices,
                tp_ticks, sl_ticks, hold_ns, cancel_ns,
                signal_threshold, max_concurrent
            )

            if len(trades) > 0:
                daily_pnls.append(trades[:, 10].sum())
            else:
                daily_pnls.append(0.0)

        daily_arr = np.array(daily_pnls)
        s = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0.0
        random_sharpes.append(s)

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= real_sharpe).mean()
    return p_value, random_sharpes.mean(), random_sharpes.std()


def main():
    print("=" * 70)
    print("TICK REPLAY v10c — MARKET ENTRY (immediate fill, pay spread)")
    print("=" * 70)
    print()
    print("Rationale: FIFO passive entry kills all edge (v10b proved).")
    print("Testing: market entry (pay 1 tick spread) + passive TP exit.")
    print("Cost model: entry 1.376t + TP exit 0.376t = 1.752t min RT cost")
    print()

    # Load dates
    dates = sorted([f[8:16] for f in os.listdir(ALIGNED_DIR) if f.startswith('aligned_')])
    print(f"Loading {len(dates)} dates...")

    day_data_list = []
    for date_str in dates:
        dd = load_day_data(date_str)
        if dd is not None:
            # Skip tiny dates (< 1000 predictions)
            if len(dd['predictions']) < 1000:
                continue
            day_data_list.append(dd)

    print(f"Loaded {len(day_data_list)} dates (skipped partial days)")
    total_preds = sum(len(dd['predictions']) for dd in day_data_list)
    print(f"Total predictions: {total_preds:,}")
    print()

    # JIT warmup
    print("Warming up Numba JIT...")
    dummy = day_data_list[0]
    _ = simulate_day_market_entry(
        dummy['ts_ns'][:1000], dummy['action'][:1000], dummy['side'][:1000],
        dummy['price'][:1000], dummy['size'][:1000],
        dummy['predictions'][:10], dummy['event_indices'][:10],
        2, 3, int(5e9), int(5e9), 0.6, 4
    )
    print("JIT ready.\n")

    # Configs: focus on what matters — high selectivity shorts with wider TP
    # Since we're paying 1.752 ticks minimum, we need TP ≥ 3 to have any chance
    configs = [
        # (tp, sl, hold_s, cancel_s, threshold, short_only, long_only, max_conc, label)
        # Short-only, high threshold (most promising)
        (3, 3, 5, 5, 0.8, True, False, 4, "S_TP3_SL3_h5_thr0.8"),
        (3, 4, 5, 5, 0.8, True, False, 4, "S_TP3_SL4_h5_thr0.8"),
        (4, 4, 8, 8, 0.8, True, False, 4, "S_TP4_SL4_h8_thr0.8"),
        (4, 6, 10, 8, 0.8, True, False, 4, "S_TP4_SL6_h10_thr0.8"),
        (5, 6, 12, 10, 0.8, True, False, 4, "S_TP5_SL6_h12_thr0.8"),
        (3, 3, 5, 5, 1.0, True, False, 4, "S_TP3_SL3_h5_thr1.0"),
        (3, 4, 5, 5, 1.0, True, False, 4, "S_TP3_SL4_h5_thr1.0"),
        (4, 4, 8, 8, 1.0, True, False, 4, "S_TP4_SL4_h8_thr1.0"),
        (4, 6, 10, 8, 1.0, True, False, 4, "S_TP4_SL6_h10_thr1.0"),
        (5, 6, 12, 10, 1.0, True, False, 4, "S_TP5_SL6_h12_thr1.0"),
        (6, 8, 15, 10, 1.0, True, False, 4, "S_TP6_SL8_h15_thr1.0"),
        # Ultra-selective (top 0.5%)
        (3, 4, 5, 5, 1.2, True, False, 4, "S_TP3_SL4_h5_thr1.2"),
        (4, 6, 10, 8, 1.2, True, False, 4, "S_TP4_SL6_h10_thr1.2"),
        (5, 6, 12, 10, 1.2, True, False, 4, "S_TP5_SL6_h12_thr1.2"),
        # Wider TP to cover costs
        (4, 3, 8, 8, 0.6, True, False, 4, "S_TP4_SL3_h8_thr0.6"),
        (5, 4, 10, 8, 0.6, True, False, 4, "S_TP5_SL4_h10_thr0.6"),
        # Both directions at ultra-high threshold
        (4, 4, 8, 8, 1.0, False, False, 4, "B_TP4_SL4_h8_thr1.0"),
        (4, 6, 10, 8, 1.0, False, False, 4, "B_TP4_SL6_h10_thr1.0"),
    ]

    print(f"Testing {len(configs)} configs across {len(day_data_list)} dates")
    print("=" * 70)
    print()

    results = {}
    promising = []

    for ci, (tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label) in enumerate(configs):
        t0 = time.time()

        all_trades, daily_pnls = run_config(
            day_data_list, tp, sl, hold_s, cancel_s, thr, max_conc, short_only, long_only
        )

        elapsed = time.time() - t0

        if not all_trades:
            print(f"[{ci+1}/{len(configs)}] {label}: NO TRADES [{elapsed:.1f}s]")
            results[label] = {'n_trades': 0, 'status': 'no_trades'}
            continue

        trades = np.vstack(all_trades)
        n_trades = len(trades)
        n_per_day = n_trades / len(day_data_list)

        pnl = trades[:, 10]
        total_pnl = pnl.sum()
        winners = pnl > 0
        wr = winners.mean()

        gross_win = pnl[winners].sum() if winners.any() else 0
        gross_loss = abs(pnl[~winners].sum()) if (~winners).any() else 1e-9
        pf = gross_win / gross_loss

        sharpe = (daily_pnls.mean() / daily_pnls.std() * np.sqrt(252)) if daily_pnls.std() > 0 else 0
        sortino_denom = daily_pnls[daily_pnls < 0].std() if (daily_pnls < 0).any() else 1e-9
        sortino = daily_pnls.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else 0

        green = (daily_pnls > 0).sum()
        red = (daily_pnls < 0).sum()

        exits = trades[:, 8]
        tp_pct = (exits == EXIT_TP).mean() * 100
        sl_pct = (exits == EXIT_SL).mean() * 100
        time_pct = (exits == EXIT_TIME).mean() * 100

        # Average cost breakdown
        avg_entry_cost = trades[:, 11].mean()
        avg_exit_cost = trades[:, 12].mean()

        results[label] = {
            'n_trades': int(n_trades), 'per_day': round(float(n_per_day), 1),
            'total_pnl_ticks': round(float(total_pnl), 1),
            'wr': round(float(wr), 4), 'pf': round(float(pf), 3),
            'sharpe': round(float(sharpe), 2), 'sortino': round(float(sortino), 2),
            'green_days': int(green), 'red_days': int(red),
            'tp_pct': round(float(tp_pct), 1),
            'sl_pct': round(float(sl_pct), 1),
            'time_pct': round(float(time_pct), 1),
            'avg_entry_cost': round(float(avg_entry_cost), 3),
            'avg_exit_cost': round(float(avg_exit_cost), 3),
        }

        status = "✓" if sharpe > 0 and pf > 1.0 else "✗"
        print(f"[{ci+1}/{len(configs)}] {status} {label}")
        print(f"  {n_trades} trades ({n_per_day:.0f}/day), PnL={total_pnl:+.1f}t, "
              f"WR={wr:.3f}, PF={pf:.3f}, Sharpe={sharpe:+.2f}, Sortino={sortino:+.2f}")
        print(f"  Exits: TP={tp_pct:.0f}% SL={sl_pct:.0f}% Time={time_pct:.0f}% | "
              f"Days: {green}G/{red}R | Costs: entry={avg_entry_cost:.3f} exit={avg_exit_cost:.3f} [{elapsed:.1f}s]")
        print()
        sys.stdout.flush()

        if sharpe > 0.5 and pf > 1.0 and wr > 0.45:
            promising.append((ci, tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label))

    # Save intermediate
    out_path = os.path.join(OUTPUT_DIR, 'v10c_market_entry_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Permutation test
    if promising:
        print()
        print("=" * 70)
        print(f"PERMUTATION TEST — {len(promising)} promising configs (100 trials each)")
        print("=" * 70)
        print()
        sys.stdout.flush()

        for ci, tp, sl, hold_s, cancel_s, thr, short_only, long_only, max_conc, label in promising:
            real_sharpe = results[label]['sharpe']
            print(f"Testing {label} (real Sharpe={real_sharpe:+.2f})...")
            sys.stdout.flush()
            t0 = time.time()

            p_val, mean_rand, std_rand = permutation_test(
                day_data_list, tp, sl, hold_s, cancel_s, thr, max_conc,
                short_only, long_only, n_perms=100, real_sharpe=real_sharpe
            )

            elapsed = time.time() - t0
            results[label]['perm_p_value'] = round(float(p_val), 4)
            results[label]['perm_random_sharpe_mean'] = round(float(mean_rand), 2)
            results[label]['perm_random_sharpe_std'] = round(float(std_rand), 2)

            verdict = "REAL EDGE ✓✓✓" if p_val < 0.05 else "ARTIFACT ✗"
            print(f"  p={p_val:.3f} | Random: {mean_rand:+.2f}±{std_rand:.2f} | {verdict} [{elapsed:.0f}s]")
            print()
            sys.stdout.flush()

    # Final summary
    print()
    print("=" * 70)
    print("FINAL SUMMARY — MARKET ENTRY (sorted by Sharpe)")
    print("=" * 70)
    hdr = f"{'Config':<30} {'Trades':>7} {'PnL':>8} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Sort':>6} {'G/R':>5} {'p-val':>6}"
    print(hdr)
    print("-" * len(hdr))
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v.get('n_trades', 0) > 0],
        key=lambda x: x[1].get('sharpe', -999), reverse=True
    )
    for label, r in sorted_results:
        p_str = f"{r['perm_p_value']:.3f}" if 'perm_p_value' in r else "  —"
        print(f"{label:<30} {r['n_trades']:>7} {r['total_pnl_ticks']:>+8.0f} "
              f"{r['wr']:>6.3f} {r['pf']:>6.3f} {r['sharpe']:>+7.2f} "
              f"{r.get('sortino', 0):>+6.2f} {r['green_days']}/{r['red_days']:>2} {p_str:>6}")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")

    validated = [k for k, v in results.items()
                 if v.get('perm_p_value', 1.0) < 0.05 and v.get('sharpe', 0) > 0.5]
    if validated:
        print(f"\n🎯 VALIDATED CONFIGS (p<0.05, Sharpe>0.5): {validated}")
        val_path = os.path.join(OUTPUT_DIR, 'VALIDATED_CONFIGS.json')
        with open(val_path, 'w') as f:
            json.dump({k: results[k] for k in validated}, f, indent=2)
    else:
        print("\n❌ No configs passed. Model edge may be too small to overcome market entry costs.")
        print("   Next steps: (1) wider TP with longer hold, (2) multi-signal confluence,")
        print("   (3) hybrid entry (passive with short cancel + market fallback)")


if __name__ == "__main__":
    main()
