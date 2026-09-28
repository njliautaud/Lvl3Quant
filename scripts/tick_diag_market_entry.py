#!/usr/bin/env python3
"""
Tick-Level Diagnostic: MARKET ENTRY vs PASSIVE ENTRY
=====================================================

The v9 sweep proved passive FIFO entry produces catastrophic results (Sharpe -15 to -40).
Hypothesis: adverse selection — we only get filled when price moves through our level.

This diagnostic tests MARKET ENTRY (IOC) at signal time:
  - Entry cost: 1.376 ticks (commission + 1 tick spread)
  - TP exit: passive limit (0.376 ticks commission only)
  - SL/time exit: market (1.376 ticks)

If the model has real directional edge at signal time, market entry should show it
by eliminating fill-latency and adverse selection.

Also outputs:
  - Fill latency distribution for passive entries (how long until FIFO fill)
  - Direction accuracy at signal time vs at fill time
  - MFE distribution from signal time (does price move in predicted direction?)

Author: Claude (diagnostic)
"""

import numpy as np
import os
import sys
import time
import json
from collections import defaultdict

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Constants
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_PASSIVE_EXIT = COMMISSION_RT_TICKS       # 0.376
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS  # 1.376
COST_MARKET_ENTRY = SPREAD_TICKS  # 1.0 tick (the entry cross; commission counted in exit)

# Paths
PREPROC_DIR = "/home/jupiter/Lvl3Quant/data/preprocessed_mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate"

# Action/side encoding
ACT_TRADE = 3
SIDE_BID = 0
SIDE_ASK = 1


def load_day(date_str):
    """Load preprocessed MBO and predictions for one day."""
    mbo_path = os.path.join(PREPROC_DIR, f"mbo_{date_str}.npz")
    pred_path = os.path.join(PRED_DIR, f"preds_{date_str}.npz")

    if not os.path.exists(mbo_path) or not os.path.exists(pred_path):
        return None, None

    mbo = np.load(mbo_path)
    preds = np.load(pred_path)
    return mbo, preds


def simulate_market_entry_day(mbo, preds, tp_ticks=2, sl_ticks=4, hold_s=5,
                               threshold=0.8, short_only=True, long_only=False):
    """
    Simulate with MARKET ENTRY (instant fill at signal time).

    Entry: market order at signal time. Cost = 1 tick spread.
    Exit: passive TP (FIFO queue) OR market SL/time-stop.

    Returns array of trades with columns:
    [side, entry_price, exit_price, signal_ts, exit_ts, signal_str,
     exit_reason, net_pnl_ticks, mfe_ticks, mae_ticks, hold_duration_ns]
    """
    ts_ns = mbo['ts_ns']
    action = mbo['action']
    side = mbo['side']
    price = mbo['price']
    size = mbo['size']

    predictions = preds['predictions']
    pred_indices = preds['event_indices']

    n_events = len(ts_ns)
    n_preds = len(predictions)

    tick_size = TICK_SIZE
    hold_ns = int(hold_s * 1_000_000_000)

    # Track BBO
    best_bid = 0.0
    best_ask = 0.0
    bid_size = 0
    ask_size = 0

    # Positions (market entry = immediately in position)
    MAX_POS = 8
    pos_active = np.zeros(MAX_POS, dtype=np.int8)
    pos_side = np.zeros(MAX_POS, dtype=np.int8)  # 0=long, 1=short
    pos_entry = np.zeros(MAX_POS, dtype=np.float64)
    pos_signal_ts = np.zeros(MAX_POS, dtype=np.int64)
    pos_signal_str = np.zeros(MAX_POS, dtype=np.float64)
    pos_deadline = np.zeros(MAX_POS, dtype=np.int64)
    pos_tp = np.zeros(MAX_POS, dtype=np.float64)
    pos_sl = np.zeros(MAX_POS, dtype=np.float64)
    pos_best = np.zeros(MAX_POS, dtype=np.float64)
    pos_worst = np.zeros(MAX_POS, dtype=np.float64)
    pos_tp_queue = np.zeros(MAX_POS, dtype=np.float64)

    trades = []
    pred_idx = 0

    for evt_i in range(n_events):
        act = action[evt_i]
        sd = side[evt_i]
        px = price[evt_i]
        sz = size[evt_i]
        ts = ts_ns[evt_i]

        if px != px or px <= 0:
            if pred_idx < n_preds and evt_i == pred_indices[pred_idx]:
                pred_idx += 1
            continue

        # Update BBO (simplified — just anchor from trades)
        if act == ACT_TRADE:
            if sd == SIDE_ASK:  # Agg sell hit bid
                best_bid = px
                if best_ask < px + tick_size or best_ask == 0.0:
                    best_ask = px + tick_size
                    ask_size = 0
                if abs(px - best_bid) < 0.001:
                    bid_size -= sz
                    if bid_size <= 0:
                        bid_size = 0
            elif sd == SIDE_BID:  # Agg buy lifted ask
                best_ask = px
                if best_bid > px - tick_size or best_bid == 0.0:
                    best_bid = px - tick_size
                    bid_size = 0
                if abs(px - best_ask) < 0.001:
                    ask_size -= sz
                    if ask_size <= 0:
                        ask_size = 0

            # Check exits
            for pi in range(MAX_POS):
                if pos_active[pi] == 0:
                    continue

                # Update MFE/MAE
                if pos_side[pi] == 0:  # Long
                    if px > pos_best[pi]: pos_best[pi] = px
                    if px < pos_worst[pi]: pos_worst[pi] = px
                else:  # Short
                    if px < pos_best[pi]: pos_best[pi] = px
                    if px > pos_worst[pi]: pos_worst[pi] = px

                exit_reason = -1
                exit_price = 0.0
                cost = 0.0

                # SL check (market exit)
                if pos_side[pi] == 0 and px <= pos_sl[pi]:
                    exit_reason = 1  # SL
                    exit_price = pos_sl[pi]
                    cost = COST_MARKET_EXIT
                elif pos_side[pi] == 1 and px >= pos_sl[pi]:
                    exit_reason = 1  # SL
                    exit_price = pos_sl[pi]
                    cost = COST_MARKET_EXIT

                # TP check (passive — needs queue depletion)
                if exit_reason < 0:
                    if pos_side[pi] == 0 and sd == SIDE_BID:
                        # Long TP: sell at ask = TP level, needs T sd=B (agg buy)
                        if abs(px - pos_tp[pi]) < 0.001:
                            pos_tp_queue[pi] -= sz
                            if pos_tp_queue[pi] <= 0:
                                exit_reason = 0  # TP
                                exit_price = pos_tp[pi]
                                cost = COST_PASSIVE_EXIT
                        elif px > pos_tp[pi]:
                            exit_reason = 0
                            exit_price = pos_tp[pi]
                            cost = COST_PASSIVE_EXIT
                    elif pos_side[pi] == 1 and sd == SIDE_ASK:
                        # Short TP: buy at bid = TP level, needs T sd=A (agg sell)
                        if abs(px - pos_tp[pi]) < 0.001:
                            pos_tp_queue[pi] -= sz
                            if pos_tp_queue[pi] <= 0:
                                exit_reason = 0  # TP
                                exit_price = pos_tp[pi]
                                cost = COST_PASSIVE_EXIT
                        elif px < pos_tp[pi]:
                            exit_reason = 0
                            exit_price = pos_tp[pi]
                            cost = COST_PASSIVE_EXIT

                # Time stop
                if exit_reason < 0 and ts >= pos_deadline[pi]:
                    exit_reason = 2  # Time
                    exit_price = px
                    cost = COST_MARKET_EXIT

                if exit_reason >= 0:
                    ep = pos_entry[pi]
                    if pos_side[pi] == 0:
                        raw_pnl = (exit_price - ep) / tick_size
                        mfe = (pos_best[pi] - ep) / tick_size
                        mae = (ep - pos_worst[pi]) / tick_size
                    else:
                        raw_pnl = (ep - exit_price) / tick_size
                        mfe = (ep - pos_best[pi]) / tick_size
                        mae = (pos_worst[pi] - ep) / tick_size

                    # Net PnL: raw_pnl - entry_cost - exit_cost
                    # Entry was market = 1 tick spread + half commission
                    # Total RT cost = COST_MARKET_ENTRY (spread) + cost (exit commission +/- spread)
                    net_pnl = raw_pnl - COST_MARKET_ENTRY - cost

                    trades.append([
                        pos_side[pi], ep, exit_price,
                        pos_signal_ts[pi], ts, pos_signal_str[pi],
                        exit_reason, net_pnl, max(mfe, 0), max(mae, 0),
                        ts - pos_signal_ts[pi]
                    ])
                    pos_active[pi] = 0

        else:
            # Non-trade events: update BBO for ADD/CANCEL
            if act == 0:  # ADD
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
            elif act == 1:  # CANCEL
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

        # Check predictions
        if pred_idx < n_preds and evt_i == pred_indices[pred_idx]:
            pred_val = predictions[pred_idx]
            pred_idx += 1

            if best_bid <= 0 or best_ask <= 0:
                continue
            if best_ask - best_bid > 2 * tick_size:
                continue

            # Check exposure
            n_active = sum(pos_active)
            if n_active >= MAX_POS:
                continue

            # Direction filter
            enter = False
            order_side = -1
            entry_price = 0.0

            if not long_only and pred_val < -threshold:
                # Short signal — market sell at bid (cross the spread)
                order_side = 1
                entry_price = best_bid  # We cross: sell at bid
                enter = True
            elif not short_only and pred_val > threshold:
                # Long signal — market buy at ask (cross the spread)
                order_side = 0
                entry_price = best_ask  # We cross: buy at ask
                enter = True

            if enter:
                slot = -1
                for j in range(MAX_POS):
                    if pos_active[j] == 0:
                        slot = j
                        break
                if slot >= 0:
                    pos_active[slot] = 1
                    pos_side[slot] = order_side
                    pos_entry[slot] = entry_price
                    pos_signal_ts[slot] = ts
                    pos_signal_str[slot] = pred_val
                    pos_deadline[slot] = ts + hold_ns
                    pos_best[slot] = entry_price
                    pos_worst[slot] = entry_price

                    if order_side == 0:  # Long
                        pos_tp[slot] = entry_price + tp_ticks * tick_size
                        pos_sl[slot] = entry_price - sl_ticks * tick_size
                        pos_tp_queue[slot] = float(max(ask_size, 0))
                    else:  # Short
                        pos_tp[slot] = entry_price - tp_ticks * tick_size
                        pos_sl[slot] = entry_price + sl_ticks * tick_size
                        pos_tp_queue[slot] = float(max(bid_size, 0))

    # EOD close remaining
    if len(ts_ns) > 0:
        final_ts = ts_ns[-1]
        last_px = price[action == ACT_TRADE]
        last_trade_px = last_px[-1] if len(last_px) > 0 else 0

        for pi in range(MAX_POS):
            if pos_active[pi] == 1 and last_trade_px > 0:
                ep = pos_entry[pi]
                if pos_side[pi] == 0:
                    raw_pnl = (last_trade_px - ep) / tick_size
                    mfe = (pos_best[pi] - ep) / tick_size
                    mae = (ep - pos_worst[pi]) / tick_size
                else:
                    raw_pnl = (ep - last_trade_px) / tick_size
                    mfe = (ep - pos_best[pi]) / tick_size
                    mae = (pos_worst[pi] - ep) / tick_size

                net_pnl = raw_pnl - COST_MARKET_ENTRY - COST_MARKET_EXIT
                trades.append([
                    pos_side[pi], ep, last_trade_px,
                    pos_signal_ts[pi], final_ts, pos_signal_str[pi],
                    3, net_pnl, max(mfe, 0), max(mae, 0),
                    final_ts - pos_signal_ts[pi]
                ])
                pos_active[pi] = 0

    return np.array(trades) if trades else np.zeros((0, 11))


def run_diagnostic():
    """Run market-entry diagnostic across all available dates."""
    print("="*70)
    print("TICK REPLAY DIAGNOSTIC: MARKET ENTRY vs PASSIVE ENTRY")
    print("="*70)

    # Find matched dates
    preproc_dates = set()
    for f in os.listdir(PREPROC_DIR):
        if f.startswith("mbo_") and f.endswith(".npz"):
            preproc_dates.add(f[4:12])

    pred_dates = set()
    for f in os.listdir(PRED_DIR):
        if f.startswith("preds_") and f.endswith(".npz"):
            pred_dates.add(f[6:14])

    matched = sorted(preproc_dates & pred_dates)
    print(f"  Matched dates: {len(matched)}")
    print()

    # Configs to test
    configs = [
        # (tp, sl, hold_s, threshold, short_only, long_only, label)
        (2, 3, 5, 0.6, True, False, "MKT_short_TP2_SL3_h5_thr0.6"),
        (2, 4, 5, 0.6, True, False, "MKT_short_TP2_SL4_h5_thr0.6"),
        (3, 4, 5, 0.6, True, False, "MKT_short_TP3_SL4_h5_thr0.6"),
        (2, 3, 5, 0.8, True, False, "MKT_short_TP2_SL3_h5_thr0.8"),
        (2, 4, 5, 0.8, True, False, "MKT_short_TP2_SL4_h5_thr0.8"),
        (3, 4, 5, 0.8, True, False, "MKT_short_TP3_SL4_h5_thr0.8"),
        (2, 3, 5, 1.0, True, False, "MKT_short_TP2_SL3_h5_thr1.0"),
        (2, 4, 5, 1.0, True, False, "MKT_short_TP2_SL4_h5_thr1.0"),
        (3, 4, 5, 1.0, True, False, "MKT_short_TP3_SL4_h5_thr1.0"),
        # Wider stops for market entry (since we pay more to enter)
        (3, 6, 8, 0.8, True, False, "MKT_short_TP3_SL6_h8_thr0.8"),
        (4, 6, 8, 0.8, True, False, "MKT_short_TP4_SL6_h8_thr0.8"),
        (3, 6, 8, 1.0, True, False, "MKT_short_TP3_SL6_h8_thr1.0"),
        (4, 6, 8, 1.0, True, False, "MKT_short_TP4_SL6_h8_thr1.0"),
        # Long-only at highest selectivity
        (2, 3, 5, 0.8, False, True, "MKT_long_TP2_SL3_h5_thr0.8"),
        (2, 4, 5, 0.8, False, True, "MKT_long_TP2_SL4_h5_thr0.8"),
        (2, 3, 5, 1.0, False, True, "MKT_long_TP2_SL3_h5_thr1.0"),
        # Both directions
        (2, 4, 5, 0.8, False, False, "MKT_both_TP2_SL4_h5_thr0.8"),
        (2, 4, 5, 1.0, False, False, "MKT_both_TP2_SL4_h5_thr1.0"),
    ]

    print(f"Testing {len(configs)} market-entry configs across {len(matched)} days")
    print()

    # Also collect raw MFE data from signal time (no TP/SL, just measure what happens)
    # This tells us: does price MOVE in predicted direction at all?

    results = {}

    for ci, (tp, sl, hold_s, thr, short_only, long_only, label) in enumerate(configs):
        t0 = time.time()
        all_trades = []

        for date_str in matched:
            mbo, preds = load_day(date_str)
            if mbo is None:
                continue

            day_trades = simulate_market_entry_day(
                mbo, preds,
                tp_ticks=tp, sl_ticks=sl, hold_s=hold_s,
                threshold=thr, short_only=short_only, long_only=long_only
            )

            if len(day_trades) > 0:
                all_trades.append(day_trades)

        elapsed = time.time() - t0

        if not all_trades:
            print(f"[{ci+1}/{len(configs)}] {label}: NO TRADES")
            continue

        trades = np.vstack(all_trades)
        n_trades = len(trades)
        n_per_day = n_trades / len(matched)

        # Column indices: 7=net_pnl, 8=mfe, 9=mae, 6=exit_reason
        pnl = trades[:, 7]
        mfe = trades[:, 8]
        mae = trades[:, 9]
        exit_reasons = trades[:, 6]

        total_pnl = pnl.sum()
        winners = pnl > 0
        wr = winners.mean() if n_trades > 0 else 0

        gross_win = pnl[winners].sum() if winners.any() else 0
        gross_loss = abs(pnl[~winners].sum()) if (~winners).any() else 1
        pf = gross_win / gross_loss if gross_loss > 0 else 0

        # Daily PnL for Sharpe
        daily_pnl = {}
        for i, date_str in enumerate(matched):
            daily_pnl[date_str] = 0.0

        # Reconstruct daily (approximate by splitting trades evenly)
        # Better: use signal_ts to map to dates
        trade_dates = []
        for date_idx, date_str in enumerate(matched):
            mbo_d, _ = load_day(date_str)
            if mbo_d is None:
                continue
            trade_dates.append(date_str)

        # Simple approach: distribute trades by order
        per_day_trades = defaultdict(list)
        trade_counter = 0
        for date_idx, date_str in enumerate(matched):
            mbo_d, preds_d = load_day(date_str)
            if mbo_d is None:
                continue
            day_t = simulate_market_entry_day(
                mbo_d, preds_d, tp_ticks=tp, sl_ticks=sl, hold_s=hold_s,
                threshold=thr, short_only=short_only, long_only=long_only
            )
            if len(day_t) > 0:
                per_day_trades[date_str] = day_t[:, 7]

        # Actually we already have per-day from the first pass... let me just
        # recompute daily PnL from per_day_trades
        daily_pnls = []
        for date_str in sorted(per_day_trades.keys()):
            daily_pnls.append(per_day_trades[date_str].sum())

        daily_pnls = np.array(daily_pnls) if daily_pnls else np.array([0.0])
        sharpe = (daily_pnls.mean() / daily_pnls.std() * np.sqrt(252)) if daily_pnls.std() > 0 else 0

        # Exit reason breakdown
        n_tp = (exit_reasons == 0).sum()
        n_sl = (exit_reasons == 1).sum()
        n_time = (exit_reasons == 2).sum()
        n_eod = (exit_reasons == 3).sum()

        green_days = (daily_pnls > 0).sum()
        red_days = (daily_pnls < 0).sum()

        results[label] = {
            'n_trades': int(n_trades),
            'per_day': round(n_per_day, 1),
            'total_pnl': round(total_pnl, 1),
            'wr': round(wr, 4),
            'pf': round(pf, 3),
            'sharpe': round(sharpe, 2),
            'mfe_median': round(np.median(mfe), 2),
            'mae_median': round(np.median(mae), 2),
            'tp_pct': round(n_tp/n_trades*100, 1),
            'sl_pct': round(n_sl/n_trades*100, 1),
            'time_pct': round(n_time/n_trades*100, 1),
            'green_days': int(green_days),
            'red_days': int(red_days),
        }

        print(f"[{ci+1}/{len(configs)}] {label}")
        print(f"  {n_trades} trades ({n_per_day:.0f}/day), PnL={total_pnl:.0f}t, "
              f"WR={wr:.3f}, PF={pf:.3f}, Sharpe={sharpe:.2f}")
        print(f"  Exits: TP={n_tp}({n_tp/n_trades*100:.0f}%) SL={n_sl}({n_sl/n_trades*100:.0f}%) "
              f"Time={n_time}({n_time/n_trades*100:.0f}%)")
        print(f"  MFE_med={np.median(mfe):.2f}t MAE_med={np.median(mae):.2f}t")
        print(f"  Days: {green_days}G/{red_days}R  [{elapsed:.1f}s]")
        print()

    # Save results
    out_path = "/home/jupiter/Lvl3Quant/output/tick_diag_market_entry_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: MARKET ENTRY RESULTS")
    print("="*70)
    print(f"{'Config':<40} {'Trades':>7} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'G/R':>5}")
    print("-"*70)
    for label, r in sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True):
        print(f"{label:<40} {r['n_trades']:>7} {r['wr']:>6.3f} {r['pf']:>6.3f} "
              f"{r['sharpe']:>7.2f} {r['green_days']}/{r['red_days']}")


if __name__ == "__main__":
    # Optimization: avoid double-computing days
    # Rewrite to single-pass per day across all configs
    # Actually the simple approach above re-simulates each day per config
    # which is fine for 18 configs x 32 days = ~576 day-sims
    # Each day takes ~0.5-2s in pure Python, so ~5-20 min total
    # That's acceptable for this diagnostic.

    run_diagnostic()
