#!/usr/bin/env python3
"""
Execution Simulator v2 (#3)
============================
Simulates realistic order execution on ES futures using MBO event data.

Key features:
- Passive limit orders at bid/ask (FIFO queue-based fills)
- Chase logic: if signal persists and unfilled, upgrade to aggressive
- Cancel timing: if signal decays, cancel stale orders
- High-confidence signal flip exits (not regular noise flips)
- Both LONG and SHORT — optimized separately
- Time-based exits, TP/SL, trailing stops
- Cost: $4.70 RT commission only for passive fills (0.376 ticks)

Directly answers:
- Can we profit with passive limits? With market orders?
- Does chase improve profitability?
- What cancel timing works best?
- Does high-confidence signal flip exit help?
- What's the optimal hold time?

Uses CNN-Mamba v2 predictions + MBO events on 39 OOT dates.
Multi-core for speed (HC #62).
"""

import os
import sys
import numpy as np
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

LVL3 = Path('/home/jupiter/Lvl3Quant')
PRED_DIR = LVL3 / 'output' / 'decay_v4_comprehensive' / 'CNN-Mamba_v2'
MBO_DIR = LVL3 / 'data' / 'processed' / 'mbo_events'
OUT_DIR = LVL3 / 'execution' / 'results' / 'exec_sim_v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ES Constants (HC #52)
TICK_USD = 12.50
COMMISSION_RT = 4.70
COMMISSION_TICKS = COMMISSION_RT / TICK_USD  # 0.376

# Event types
EVT_ADD = 0
EVT_CANCEL = 1
EVT_MODIFY = 2
EVT_TRADE = 3
EVT_FILL = 4
SIDE_BID = 0
SIDE_ASK = 1


@dataclass
class ExecConfig:
    """One execution configuration."""
    name: str
    direction: str  # 'long', 'short'

    # Entry
    confidence_pct: float  # top N% by |signal|
    horizon_mode: str  # '1s', '5s', '10s', 'composite'

    # Order management
    order_type: str  # 'passive', 'chase', 'market'
    chase_delay_ms: int  # ms before upgrading passive to aggressive
    cancel_if_signal_decays: bool
    cancel_decay_threshold: float  # cancel if |signal| drops below this fraction of entry signal

    # Exit rules
    tp_ticks: float
    sl_ticks: float
    max_hold_ms: int
    trailing_stop_activation_ticks: float  # activate trail after this MFE
    trailing_stop_offset_ticks: float  # trail distance

    # Signal-based exits
    signal_flip_exit: bool  # exit on ANY signal flip
    high_conf_flip_exit: bool  # exit only on HIGH confidence reversal
    high_conf_flip_threshold: float  # |signal| threshold for "high confidence"
    signal_decay_exit: bool
    signal_decay_exit_threshold: float  # exit if signal decays below this

    # Anti-churn
    min_interval_ms: int


def load_date(date_str):
    """Load all data for a date."""
    pred_path = PRED_DIR / date_str / 'predictions.npz'
    mbo_path = MBO_DIR / f'{date_str}_mbo_events.npz'

    if not pred_path.exists() or not mbo_path.exists():
        return None

    pred = np.load(pred_path)
    mbo = np.load(mbo_path)

    preds_arr = pred['preds']
    valid_idx = pred['valid_indices']
    timestamps = mbo['timestamps']

    # Filter valid_idx to be within bounds
    max_idx = len(timestamps) - 1
    in_bounds = valid_idx <= max_idx
    valid_idx = valid_idx[in_bounds]
    preds_arr = preds_arr[in_bounds]

    return {
        'date': date_str,
        'preds': preds_arr,  # (N, 3) — [1s, 5s, 10s]
        'valid_idx': valid_idx,
        'events': mbo['events'],  # (M, 6) — [time_delta_log, evt_type, side, price_rel, qty_log, spread]
        'timestamps': timestamps,
        'labels_1s': mbo['labels_1s'],
        'labels_5s': mbo['labels_5s'],
        'labels_10s': mbo['labels_10s'],
    }


def get_signal(preds, horizon_mode):
    """Extract signal based on horizon mode."""
    if horizon_mode == '1s':
        return preds[:, 0]
    elif horizon_mode == '5s':
        return preds[:, 1]
    elif horizon_mode == '10s':
        return preds[:, 2]
    elif horizon_mode == 'composite':
        return preds.mean(axis=1)
    else:
        return preds.mean(axis=1)


def simulate_config_on_date(args):
    """Simulate one config on one date."""
    config, date_str = args
    data = load_date(date_str)
    if data is None:
        return []

    preds = data['preds']
    valid_idx = data['valid_idx']
    timestamps = data['timestamps']
    labels_1s = data['labels_1s']
    labels_5s = data['labels_5s']
    labels_10s = data['labels_10s']
    events = data['events']
    N_preds = len(preds)

    if N_preds < 20:
        return []

    signal = get_signal(preds, config.horizon_mode)
    abs_signal = np.abs(signal)
    threshold = np.percentile(abs_signal, 100 - config.confidence_pct)

    # Direction filter
    if config.direction == 'long':
        dir_mask = signal > 0
    elif config.direction == 'short':
        dir_mask = signal < 0
    else:
        dir_mask = np.ones(N_preds, dtype=bool)

    # Confidence filter
    conf_mask = abs_signal >= threshold
    entry_mask = dir_mask & conf_mask

    # Get entry signal indices
    entry_indices = np.where(entry_mask)[0]

    trades = []
    last_exit_ts = 0

    for entry_idx in entry_indices:
        entry_mbo_idx = valid_idx[entry_idx]
        entry_ts = timestamps[entry_mbo_idx]
        entry_signal = signal[entry_idx]

        # Anti-churn
        if entry_ts - last_exit_ts < config.min_interval_ms * 1_000_000:
            continue

        # Determine entry price level from labels (mid-price based)
        # For passive orders: we're joining the queue at current bid (long) or ask (short)
        # Spread at signal time
        spread = events[entry_mbo_idx, 5] if entry_mbo_idx < len(events) else 1.0
        spread = max(spread, 1.0)  # minimum 1 tick

        # ── FILL SIMULATION ────────────────────────────────────────
        fill_ts = None
        fill_cost_ticks = 0

        if config.order_type == 'market':
            # Market order: instant fill, pay the spread
            fill_ts = entry_ts
            fill_cost_ticks = spread / 2 + COMMISSION_TICKS  # cross half-spread + commission

        elif config.order_type == 'passive':
            # Passive limit: need queue-based fill
            # Approximate: look for trade events at our price level
            # Use label data to estimate fill: if price moved through our level within some window
            # Simple model: fill after some delay proportional to queue depth
            # More realistic: check if labels show price went THROUGH our entry

            # For a long at bid: we get filled when price drops to our level
            # For a short at ask: we get filled when price rises to our level
            # Since we're AT best bid/ask, we need trades to hit our price

            # Conservative estimate: fill if price moves away from us first then back
            # Or fills naturally as market trades at our price
            # Use a simple fill probability model based on events after signal

            fill_window_ns = 10_000_000_000  # 10s max wait for fill
            cancel_ts = entry_ts + config.cancel_if_signal_decays * (config.chase_delay_ms * 1_000_000) if config.cancel_if_signal_decays else entry_ts + fill_window_ns

            # Look at subsequent events for trades at our price level
            # Simplified: check if label at some horizon shows price went through
            # Better approximation: simulate queue consumption from trade events

            # Check volume consumed at our price level
            total_vol = 0.0
            est_queue_ahead = 50  # typical queue depth at best bid/ask in ES
            search_end = min(entry_mbo_idx + 100000, len(events))

            filled = False
            for j in range(entry_mbo_idx + 1, search_end):
                t_j = timestamps[j]

                # Check cancel condition
                if config.cancel_if_signal_decays:
                    # Find nearest prediction to check if signal decayed
                    pred_j = np.searchsorted(valid_idx, j)
                    if pred_j < N_preds:
                        current_signal = signal[pred_j]
                        if abs(current_signal) < abs(entry_signal) * config.cancel_decay_threshold:
                            break  # Signal decayed, cancel order

                # Timeout
                if t_j - entry_ts > fill_window_ns:
                    break

                # Check if this is a trade at our price level
                evt_type = events[j, 1]
                if evt_type in (EVT_TRADE, EVT_FILL):
                    evt_side = events[j, 2]
                    qty = np.exp(events[j, 4]) if events[j, 4] != 0 else 1.0

                    # For long (buying at bid): fills come from sells hitting bid
                    # For short (selling at ask): fills come from buys hitting ask
                    if config.direction == 'long' and evt_side == SIDE_BID:
                        total_vol += qty
                    elif config.direction == 'short' and evt_side == SIDE_ASK:
                        total_vol += qty

                    if total_vol >= est_queue_ahead:
                        fill_ts = t_j
                        fill_cost_ticks = COMMISSION_TICKS  # passive fill, no spread cost
                        filled = True
                        break

            if not filled:
                # Chase upgrade?
                if config.order_type != 'chase':
                    continue  # unfilled passive, skip

        elif config.order_type == 'chase':
            # Try passive first, then upgrade to aggressive after delay
            chase_deadline_ns = entry_ts + config.chase_delay_ms * 1_000_000
            fill_window_ns = 10_000_000_000

            total_vol = 0.0
            est_queue_ahead = 50
            search_end = min(entry_mbo_idx + 100000, len(events))
            filled_passive = False

            for j in range(entry_mbo_idx + 1, search_end):
                t_j = timestamps[j]

                # Signal decay cancel?
                if config.cancel_if_signal_decays:
                    pred_j = np.searchsorted(valid_idx, j)
                    if pred_j < N_preds:
                        current_signal = signal[pred_j]
                        if abs(current_signal) < abs(entry_signal) * config.cancel_decay_threshold:
                            break

                if t_j - entry_ts > fill_window_ns:
                    break

                evt_type = events[j, 1]
                if evt_type in (EVT_TRADE, EVT_FILL):
                    evt_side = events[j, 2]
                    qty = np.exp(events[j, 4]) if events[j, 4] != 0 else 1.0

                    if config.direction == 'long' and evt_side == SIDE_BID:
                        total_vol += qty
                    elif config.direction == 'short' and evt_side == SIDE_ASK:
                        total_vol += qty

                    if total_vol >= est_queue_ahead:
                        fill_ts = t_j
                        fill_cost_ticks = COMMISSION_TICKS
                        filled_passive = True
                        break

                # Chase upgrade: if past deadline and still unfilled, cross spread
                if not filled_passive and t_j >= chase_deadline_ns:
                    # Check signal still valid before chasing
                    pred_j = np.searchsorted(valid_idx, j)
                    if pred_j < N_preds and abs(signal[pred_j]) >= threshold * 0.7:
                        fill_ts = t_j
                        fill_cost_ticks = spread / 2 + COMMISSION_TICKS
                        break
                    else:
                        break  # signal gone, don't chase

            if fill_ts is None:
                continue

        if fill_ts is None:
            continue

        # ── EXIT SIMULATION ────────────────────────────────────────
        # Find fill index in MBO array
        fill_mbo_idx = np.searchsorted(timestamps, fill_ts)
        fill_pred_idx = np.searchsorted(valid_idx, fill_mbo_idx)

        # Track P&L using label data
        # Mid-price at fill: use labels to track forward price movement
        exit_ts = None
        exit_reason = None
        pnl_ticks = 0
        max_favorable = 0  # MFE tracking
        trailing_stop_active = False
        trailing_stop_level = float('-inf') if config.direction == 'long' else float('inf')

        max_exit_idx = min(fill_mbo_idx + 500000, len(timestamps))
        max_hold_ns = config.max_hold_ms * 1_000_000

        # Use label-based P&L tracking
        # labels_Xs[i] = mid(t_i + X) - mid(t_i) in ticks
        # For fill at index f, price at index j relative to fill:
        #   mid(j) - mid(f) ≈ labels_10s[f] - labels_10s[j] + labels_10s[j]
        # Actually: if we know labels at f, we know where mid goes

        # Simpler: track using labels at the fill point
        # labels_10s[fill_idx] tells us: mid(fill_time + 10s) - mid(fill_time)
        # For more granular tracking, we need event-by-event mid reconstruction

        # Use event-by-event approach with labels
        for j in range(fill_mbo_idx + 1, max_exit_idx):
            t_j = timestamps[j]
            hold_time_ns = t_j - fill_ts

            # Get current mid relative to fill mid using labels
            # Approximation: use the difference in same-horizon labels
            # mid(j) - mid(fill) ≈ labels_10s[fill_mbo_idx] - labels_10s[j] (if both within 10s window)
            # This is an approximation; use the longest available horizon

            if j < len(labels_10s) and fill_mbo_idx < len(labels_10s):
                # Current unrealized P&L in ticks
                # labels_10s[f] = mid(f+10s) - mid(f)
                # labels_10s[j] = mid(j+10s) - mid(j)
                # So: mid(j) - mid(f) is NOT directly available from labels
                # We need another approach

                # Better: use labels at fill point directly
                # labels_1s[f] = mid(f+1s) - mid(f) → P&L if we exit at f+1s
                # But we want continuous tracking...

                # For simplicity: use discrete checkpoints from labels
                if hold_time_ns >= 1_000_000_000 and hold_time_ns < 2_000_000_000:
                    unrealized = labels_1s[fill_mbo_idx] if fill_mbo_idx < len(labels_1s) else 0
                elif hold_time_ns >= 5_000_000_000 and hold_time_ns < 6_000_000_000:
                    unrealized = labels_5s[fill_mbo_idx] if fill_mbo_idx < len(labels_5s) else 0
                elif hold_time_ns >= 10_000_000_000:
                    unrealized = labels_10s[fill_mbo_idx] if fill_mbo_idx < len(labels_10s) else 0
                else:
                    # Interpolate between available horizons
                    hold_s = hold_time_ns / 1e9
                    if hold_s < 1:
                        unrealized = labels_1s[fill_mbo_idx] * hold_s if fill_mbo_idx < len(labels_1s) else 0
                    elif hold_s < 5:
                        l1 = labels_1s[fill_mbo_idx] if fill_mbo_idx < len(labels_1s) else 0
                        l5 = labels_5s[fill_mbo_idx] if fill_mbo_idx < len(labels_5s) else 0
                        frac = (hold_s - 1) / 4
                        unrealized = l1 + (l5 - l1) * frac
                    else:
                        l5 = labels_5s[fill_mbo_idx] if fill_mbo_idx < len(labels_5s) else 0
                        l10 = labels_10s[fill_mbo_idx] if fill_mbo_idx < len(labels_10s) else 0
                        frac = (hold_s - 5) / 5
                        unrealized = l5 + (l10 - l5) * frac
            else:
                continue

            # Direction-adjust P&L
            if config.direction == 'short':
                unrealized = -unrealized

            # Update MFE
            max_favorable = max(max_favorable, unrealized)

            # ── Check exit conditions ──────────────────────────────

            # 1. Take profit
            if unrealized >= config.tp_ticks:
                exit_ts = t_j
                exit_reason = 'tp'
                pnl_ticks = config.tp_ticks
                break

            # 2. Stop loss
            if unrealized <= -config.sl_ticks:
                exit_ts = t_j
                exit_reason = 'sl'
                pnl_ticks = -config.sl_ticks
                break

            # 3. Trailing stop
            if config.trailing_stop_activation_ticks > 0 and max_favorable >= config.trailing_stop_activation_ticks:
                trailing_stop_active = True
                trail_level = max_favorable - config.trailing_stop_offset_ticks
                if unrealized <= trail_level:
                    exit_ts = t_j
                    exit_reason = 'trailing_stop'
                    pnl_ticks = trail_level
                    break

            # 4. Max hold
            if hold_time_ns >= max_hold_ns:
                exit_ts = t_j
                exit_reason = 'max_hold'
                pnl_ticks = unrealized
                break

            # 5. Signal-based exits (check every ~1000 events for speed)
            if j % 1000 == 0:
                pred_j = np.searchsorted(valid_idx, j)
                if pred_j < N_preds:
                    current_signal = signal[pred_j]

                    # High confidence signal flip exit
                    if config.high_conf_flip_exit:
                        # Signal flipped AND is strong in opposite direction
                        if config.direction == 'long' and current_signal < -config.high_conf_flip_threshold:
                            exit_ts = t_j
                            exit_reason = 'high_conf_flip'
                            pnl_ticks = unrealized
                            break
                        elif config.direction == 'short' and current_signal > config.high_conf_flip_threshold:
                            exit_ts = t_j
                            exit_reason = 'high_conf_flip'
                            pnl_ticks = unrealized
                            break

                    # Regular signal flip exit
                    elif config.signal_flip_exit:
                        if config.direction == 'long' and current_signal < 0:
                            exit_ts = t_j
                            exit_reason = 'signal_flip'
                            pnl_ticks = unrealized
                            break
                        elif config.direction == 'short' and current_signal > 0:
                            exit_ts = t_j
                            exit_reason = 'signal_flip'
                            pnl_ticks = unrealized
                            break

                    # Signal decay exit
                    if config.signal_decay_exit:
                        if abs(current_signal) < config.signal_decay_exit_threshold:
                            exit_ts = t_j
                            exit_reason = 'signal_decay'
                            pnl_ticks = unrealized
                            break

        if exit_ts is None:
            # EOD exit
            exit_ts = timestamps[min(max_exit_idx - 1, len(timestamps) - 1)]
            exit_reason = 'eod'
            pnl_ticks = unrealized if 'unrealized' in dir() else 0

        # Net P&L
        pnl_ticks_net = pnl_ticks - fill_cost_ticks
        pnl_dollars = pnl_ticks_net * TICK_USD

        trades.append({
            'date': date_str,
            'direction': config.direction,
            'entry_ts': int(entry_ts),
            'fill_ts': int(fill_ts),
            'exit_ts': int(exit_ts),
            'exit_reason': exit_reason,
            'pnl_ticks': float(pnl_ticks),
            'pnl_ticks_net': float(pnl_ticks_net),
            'pnl_dollars': float(pnl_dollars),
            'fill_cost_ticks': float(fill_cost_ticks),
            'mfe_ticks': float(max_favorable),
            'hold_ms': float((exit_ts - fill_ts) / 1_000_000),
            'queue_wait_ms': float((fill_ts - entry_ts) / 1_000_000),
            'signal_strength': float(abs(entry_signal)),
        })

        last_exit_ts = exit_ts

    return trades


def build_configs():
    """Build all execution configurations to test."""
    configs = []
    config_id = 0

    for direction in ['long', 'short']:
        # ── Group 1: Passive Limit Orders ──────────────────────────
        for conf_pct in [3, 5, 10, 20]:
            for hz in ['composite', '10s']:
                for tp, sl in [(4, 3), (6, 4), (8, 5), (10, 6)]:
                    for max_hold in [5000, 10000, 30000]:
                        configs.append(ExecConfig(
                            name=f'passive_{direction}_{conf_pct}pct_{hz}_tp{tp}sl{sl}_hold{max_hold}',
                            direction=direction,
                            confidence_pct=conf_pct,
                            horizon_mode=hz,
                            order_type='passive',
                            chase_delay_ms=0,
                            cancel_if_signal_decays=True,
                            cancel_decay_threshold=0.3,
                            tp_ticks=tp,
                            sl_ticks=sl,
                            max_hold_ms=max_hold,
                            trailing_stop_activation_ticks=0,
                            trailing_stop_offset_ticks=0,
                            signal_flip_exit=False,
                            high_conf_flip_exit=True,
                            high_conf_flip_threshold=1.5,
                            signal_decay_exit=True,
                            signal_decay_exit_threshold=0.3,
                            min_interval_ms=2000,
                        ))
                        config_id += 1

        # ── Group 2: Chase Entry (passive → aggressive if signal persists) ──
        for conf_pct in [3, 5, 10]:
            for chase_delay in [500, 1000, 2000]:
                for tp, sl in [(4, 3), (6, 4), (8, 5)]:
                    configs.append(ExecConfig(
                        name=f'chase_{direction}_{conf_pct}pct_delay{chase_delay}_tp{tp}sl{sl}',
                        direction=direction,
                        confidence_pct=conf_pct,
                        horizon_mode='composite',
                        order_type='chase',
                        chase_delay_ms=chase_delay,
                        cancel_if_signal_decays=True,
                        cancel_decay_threshold=0.3,
                        tp_ticks=tp,
                        sl_ticks=sl,
                        max_hold_ms=10000,
                        trailing_stop_activation_ticks=0,
                        trailing_stop_offset_ticks=0,
                        signal_flip_exit=False,
                        high_conf_flip_exit=True,
                        high_conf_flip_threshold=1.5,
                        signal_decay_exit=True,
                        signal_decay_exit_threshold=0.3,
                        min_interval_ms=2000,
                    ))
                    config_id += 1

        # ── Group 3: Market Orders (instant fill, higher cost) ─────
        for conf_pct in [1, 3, 5]:
            for tp, sl in [(6, 4), (8, 5), (10, 6)]:
                configs.append(ExecConfig(
                    name=f'market_{direction}_{conf_pct}pct_tp{tp}sl{sl}',
                    direction=direction,
                    confidence_pct=conf_pct,
                    horizon_mode='composite',
                    order_type='market',
                    chase_delay_ms=0,
                    cancel_if_signal_decays=False,
                    cancel_decay_threshold=0,
                    tp_ticks=tp,
                    sl_ticks=sl,
                    max_hold_ms=10000,
                    trailing_stop_activation_ticks=0,
                    trailing_stop_offset_ticks=0,
                    signal_flip_exit=False,
                    high_conf_flip_exit=True,
                    high_conf_flip_threshold=1.5,
                    signal_decay_exit=True,
                    signal_decay_exit_threshold=0.3,
                    min_interval_ms=2000,
                ))
                config_id += 1

        # ── Group 4: Signal Flip Exit Comparison ───────────────────
        for conf_pct in [5, 10]:
            # Regular flip vs high-confidence flip vs no flip
            for flip_type, flip_flag, hc_flag, hc_thr in [
                ('no_flip', False, False, 0),
                ('regular_flip', True, False, 0),
                ('hc_flip_1.0', False, True, 1.0),
                ('hc_flip_1.5', False, True, 1.5),
                ('hc_flip_2.0', False, True, 2.0),
            ]:
                configs.append(ExecConfig(
                    name=f'{flip_type}_{direction}_{conf_pct}pct',
                    direction=direction,
                    confidence_pct=conf_pct,
                    horizon_mode='composite',
                    order_type='passive',
                    chase_delay_ms=0,
                    cancel_if_signal_decays=True,
                    cancel_decay_threshold=0.3,
                    tp_ticks=6,
                    sl_ticks=4,
                    max_hold_ms=10000,
                    trailing_stop_activation_ticks=0,
                    trailing_stop_offset_ticks=0,
                    signal_flip_exit=flip_flag,
                    high_conf_flip_exit=hc_flag,
                    high_conf_flip_threshold=hc_thr,
                    signal_decay_exit=True,
                    signal_decay_exit_threshold=0.3,
                    min_interval_ms=2000,
                ))
                config_id += 1

        # ── Group 5: Trailing Stops ────────────────────────────────
        for conf_pct in [5, 10]:
            for trail_act, trail_off in [(3, 2), (4, 2), (5, 3), (6, 3)]:
                configs.append(ExecConfig(
                    name=f'trail_{direction}_{conf_pct}pct_act{trail_act}_off{trail_off}',
                    direction=direction,
                    confidence_pct=conf_pct,
                    horizon_mode='composite',
                    order_type='passive',
                    chase_delay_ms=0,
                    cancel_if_signal_decays=True,
                    cancel_decay_threshold=0.3,
                    tp_ticks=10,  # wide TP to let trail do the work
                    sl_ticks=4,
                    max_hold_ms=15000,
                    trailing_stop_activation_ticks=trail_act,
                    trailing_stop_offset_ticks=trail_off,
                    signal_flip_exit=False,
                    high_conf_flip_exit=True,
                    high_conf_flip_threshold=1.5,
                    signal_decay_exit=False,
                    signal_decay_exit_threshold=0,
                    min_interval_ms=2000,
                ))
                config_id += 1

    print(f"Built {len(configs)} configurations ({len(configs)//2} per side)")
    return configs


def run_config(config):
    """Run one config across all dates."""
    dates = sorted([d.name for d in PRED_DIR.iterdir() if d.is_dir()])

    all_trades = []
    for date_str in dates:
        trades = simulate_config_on_date((config, date_str))
        all_trades.extend(trades)

    if not all_trades:
        return None

    # Compute summary metrics
    pnls = np.array([t['pnl_ticks_net'] for t in all_trades])
    pnl_dollars = np.array([t['pnl_dollars'] for t in all_trades])
    mfes = np.array([t['mfe_ticks'] for t in all_trades])
    holds = np.array([t['hold_ms'] for t in all_trades])
    waits = np.array([t['queue_wait_ms'] for t in all_trades])

    n_trades = len(pnls)
    n_dates = len(set(t['date'] for t in all_trades))
    trades_per_day = n_trades / max(n_dates, 1)

    win_rate = np.mean(pnls > 0) if n_trades > 0 else 0
    total_pnl = np.sum(pnl_dollars)
    avg_pnl = np.mean(pnl_dollars)
    profit_factor = abs(np.sum(pnl_dollars[pnls > 0]) / np.sum(pnl_dollars[pnls < 0])) if np.any(pnls < 0) and np.any(pnls > 0) else 0

    # Daily P&L for Sharpe/Sortino
    daily_pnl = {}
    for t in all_trades:
        daily_pnl.setdefault(t['date'], 0)
        daily_pnl[t['date']] += t['pnl_dollars']

    daily_returns = np.array(list(daily_pnl.values()))
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        neg_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(neg_returns) if len(neg_returns) > 0 else np.std(daily_returns)
        sortino = np.mean(daily_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # Exit reason breakdown
    exit_reasons = {}
    for t in all_trades:
        r = t['exit_reason']
        exit_reasons[r] = exit_reasons.get(r, 0) + 1

    return {
        'config_name': config.name,
        'direction': config.direction,
        'order_type': config.order_type,
        'confidence_pct': config.confidence_pct,
        'n_trades': n_trades,
        'n_dates': n_dates,
        'trades_per_day': round(trades_per_day, 2),
        'win_rate': round(win_rate, 4),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'profit_factor': round(profit_factor, 3),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'avg_mfe_ticks': round(np.mean(mfes), 2),
        'avg_hold_ms': round(np.mean(holds), 0),
        'avg_wait_ms': round(np.mean(waits), 0),
        'exit_reasons': exit_reasons,
    }


def main():
    print("=" * 70)
    print("EXECUTION SIMULATOR v2 — Chase, Cancel, Signal Exits")
    print("Both LONG and SHORT")
    print("=" * 70)

    configs = build_configs()

    n_workers = min(14, os.cpu_count() or 4)
    print(f"Running {len(configs)} configs on {n_workers} cores...")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(run_config, configs))

    results = [r for r in results if r is not None]
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s — {len(results)} configs returned results")

    # ── Sort and display results ───────────────────────────────────
    # Sort by Sortino (risk-adjusted, per user preference)
    results.sort(key=lambda r: r['sortino'], reverse=True)

    # Save full results
    out_path = OUT_DIR / 'exec_sim_v2_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # ── Display Top Results ────────────────────────────────────────
    for direction in ['long', 'short']:
        dir_results = [r for r in results if r['direction'] == direction and r['n_trades'] >= 10]
        dir_results.sort(key=lambda r: r['sortino'], reverse=True)

        print(f"\n{'='*70}")
        print(f"TOP 15 {direction.upper()} CONFIGS (by Sortino, min 10 trades)")
        print(f"{'='*70}")
        print(f"{'Config':>45s} | {'Type':>7s} | {'Trd':>4s} | {'T/D':>4s} | {'WR':>5s} | {'PF':>5s} | {'Sharpe':>6s} | {'Sort':>6s} | {'$Total':>8s} | {'AvgMFE':>6s}")
        print(f"{'-'*45}-+-{'-'*7}-+-{'-'*4}-+-{'-'*4}-+-{'-'*5}-+-{'-'*5}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")

        for r in dir_results[:15]:
            name_short = r['config_name'][-45:]
            print(f"{name_short:>45s} | {r['order_type']:>7s} | {r['n_trades']:>4d} | {r['trades_per_day']:>4.1f} | {r['win_rate']:>4.1%} | {r['profit_factor']:>5.2f} | {r['sharpe']:>6.1f} | {r['sortino']:>6.1f} | ${r['total_pnl']:>7.0f} | {r['avg_mfe_ticks']:>5.1f}t")

    # ── Order Type Comparison ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("ORDER TYPE COMPARISON (avg across configs, min 10 trades)")
    print(f"{'='*70}")

    for direction in ['long', 'short']:
        print(f"\n  {direction.upper()}:")
        for order_type in ['passive', 'chase', 'market']:
            type_results = [r for r in results if r['direction'] == direction and r['order_type'] == order_type and r['n_trades'] >= 10]
            if type_results:
                avg_wr = np.mean([r['win_rate'] for r in type_results])
                avg_sortino = np.mean([r['sortino'] for r in type_results])
                avg_pf = np.mean([r['profit_factor'] for r in type_results])
                profitable = sum(1 for r in type_results if r['total_pnl'] > 0)
                print(f"    {order_type:>8s}: {len(type_results):>3d} configs | WR={avg_wr:.1%} | PF={avg_pf:.2f} | Sortino={avg_sortino:.1f} | {profitable}/{len(type_results)} profitable")

    # ── Signal Flip Exit Comparison ────────────────────────────────
    print(f"\n{'='*70}")
    print("SIGNAL FLIP EXIT COMPARISON")
    print(f"{'='*70}")

    for direction in ['long', 'short']:
        print(f"\n  {direction.upper()}:")
        for flip_type in ['no_flip', 'regular_flip', 'hc_flip_1.0', 'hc_flip_1.5', 'hc_flip_2.0']:
            flip_results = [r for r in results if r['config_name'].startswith(flip_type) and r['direction'] == direction and r['n_trades'] >= 5]
            if flip_results:
                avg_sortino = np.mean([r['sortino'] for r in flip_results])
                avg_wr = np.mean([r['win_rate'] for r in flip_results])
                avg_pf = np.mean([r['profit_factor'] for r in flip_results])
                print(f"    {flip_type:>15s}: WR={avg_wr:.1%} | PF={avg_pf:.2f} | Sortino={avg_sortino:.1f}")

    print(f"\n💾 Full results saved to {out_path}")
    print("\nDONE — Execution Simulator v2 Complete")


if __name__ == '__main__':
    main()
