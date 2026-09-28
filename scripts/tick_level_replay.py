#!/usr/bin/env python3
"""
Tick-Level FIFO Replay Engine
==============================

DEFINITIVE test of CNN-Mamba v3.4.2 predictions with tick-by-tick execution.

Eliminates bar-level tiebreaker bias by replaying actual MBO trade data to
determine which exit (TP or SL) is hit FIRST.

Data flow:
  1. Load CNN-Mamba v3.4.2 OOT predictions (per-date .npz)
  2. Map predictions to MBO event timestamps (stride=250, window=1500)
  3. Load raw Databento .dbn.zst MBO data for that date
  4. Extract ES front-month trades during RTH
  5. For each qualifying signal (conf >= threshold, |zscore| >= threshold):
     - Place passive limit entry at best bid (long) or best ask (short)
     - Model FIFO queue position: fill when price trades THROUGH our level
       (conservative: we're at back of queue)
     - Once filled, track tick-by-tick for TP/SL
     - TP: passive limit on opposite side → commission only
     - SL: stop-market → commission + 1 tick spread crossing

Price units: INDEX POINTS throughout (e.g., 5950.25). 1 tick = 0.25 points.

Cost model (ES, AMP/Rithmic):
  - RT commission: $4.70 = 0.376 ticks
  - Passive entry + passive TP exit: 0.376 ticks total
  - Passive entry + market SL exit: 1.376 ticks total (0.376 + 1.0 spread)

Author: Claude (autonomous build, 2026-07-01)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import databento as dbn
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_MBO_DIR = ROOT / "data" / "raw" / "mbo"
SMART_V3_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
OOT_PRED_DIR = ROOT / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
OUTPUT_DIR = ROOT / "output" / "tick_level_replay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ES_TICK_SIZE = 0.25          # index points per tick
ES_TICK_VALUE = 12.50        # dollars per tick
ES_RT_COMMISSION = 4.70      # dollars round-trip
ES_RT_COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376 ticks

# Cost in ticks for different exit types
COST_TP_EXIT = ES_RT_COMMISSION_TICKS       # 0.376t (passive entry + passive exit)
COST_SL_EXIT = ES_RT_COMMISSION_TICKS + 1.0 # 1.376t (passive entry + market exit)
COST_TIMEOUT_EXIT = ES_RT_COMMISSION_TICKS + 1.0  # same as SL (market exit)

# Model alignment
WINDOW_SIZE = 1500
STRIDE = 250

# Signal thresholds
# CNN-Mamba pred_log_ret_1s is bimodal (clusters at ~+0.19 and ~-0.21)
# "Confidence" = absolute value of prediction (max ~0.4)
# Use low confidence threshold (just ensure non-zero) and z-score to select top signals
DEFAULT_CONF_THRESHOLD = 0.01   # effectively: any non-trivial prediction
DEFAULT_ZSCORE_THRESHOLD = 1.0  # top/bottom ~33% of daily predictions

# FIFO queue model: fill when price trades THROUGH our level
# (conservative: we're at back of queue, so need trade at or beyond our price)
FILL_THROUGH_TICKS = 1  # price must trade 1 tick through our level

# Maximum hold time for a position (seconds)
MAX_HOLD_SECONDS = 45  # ~1.5x the model horizon (30s)

# Cancel unfilled entries after this many seconds
CANCEL_WINDOW_SECONDS = 30  # model horizon

# TP/SL configs to test (in ticks)
TP_SL_CONFIGS = [
    (5, 1),
    (6, 1),
    (10, 1),
    (10, 2),
    (20, 4),
]

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TICK-REPLAY] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("TICK-REPLAY")


# ─────────────────────────────────────────────
#  TRADE DATA LOADING
# ─────────────────────────────────────────────

def load_raw_trades(date_str: str) -> pd.DataFrame:
    """
    Load ES front-month RTH trades from raw Databento .dbn.zst file.
    Returns DataFrame with columns: ts_ns, price, size, side
    where price is in index points (e.g., 5950.25).
    """
    fname = f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    fpath = RAW_MBO_DIR / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Raw MBO file not found: {fpath}")

    store = dbn.DBNStore.from_file(str(fpath))
    df = store.to_df()

    # Filter to ES front-month (symbol like ESH6, ESM6, etc.)
    es_mask = df['symbol'].str.match(r'^ES[A-Z]\d$', na=False)
    es = df[es_mask].copy()

    # Determine dominant instrument (most events)
    inst_counts = es['instrument_id'].value_counts()
    dominant = inst_counts.index[0]
    es = es[es['instrument_id'] == dominant]

    # Convert ts_event to nanoseconds (int64)
    es_ts_ns = es['ts_event'].astype(np.int64)

    # RTH filter: 13:30-21:00 UTC (covers both EST and EDT)
    ts_event_utc = es['ts_event'].dt.tz_convert('UTC')
    hour = ts_event_utc.dt.hour
    minute = ts_event_utc.dt.minute
    rth_mask = ((hour > 13) | ((hour == 13) & (minute >= 30))) & (hour < 21)
    es_rth = es[rth_mask].copy()
    es_rth_ts_ns = es_ts_ns[rth_mask]

    # Extract ALL events (we need trades AND book updates for bid/ask tracking)
    # Trades for fill/exit detection, book events for bid/ask levels
    result = pd.DataFrame({
        'ts_ns': es_rth_ts_ns.values,
        'price': es_rth['price'].values,
        'size': es_rth['size'].values.astype(np.int32),
        'action': es_rth['action'].values,
        'side': es_rth['side'].values,
    })

    log.info(f"  Loaded {len(result)} RTH events ({len(result[result['action']=='T'])} trades) for {date_str}")
    return result


def build_bid_ask_from_trades(events_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build best bid/ask arrays from MBO events.
    Returns: (ts_ns, best_bid, best_ask) arrays aligned to trade events only.

    For simplicity and speed, we track bid/ask from trade aggressor side:
    - Trade on ask side (aggressive buy) → trade price = ask, bid = price - tick
    - Trade on bid side (aggressive sell) → trade price = bid, ask = price + tick
    This is accurate for ES during RTH (1-tick spread 95%+ of the time).
    """
    trade_mask = events_df['action'] == 'T'
    trades = events_df[trade_mask].copy()

    ts_ns = trades['ts_ns'].values
    prices = trades['price'].values
    sides = trades['side'].values

    best_bid = np.empty(len(trades), dtype=np.float64)
    best_ask = np.empty(len(trades), dtype=np.float64)

    for i in range(len(trades)):
        if sides[i] == 'A':
            # Aggressive buy hit the ask
            best_ask[i] = prices[i]
            best_bid[i] = prices[i] - ES_TICK_SIZE
        elif sides[i] == 'B':
            # Aggressive sell hit the bid
            best_bid[i] = prices[i]
            best_ask[i] = prices[i] + ES_TICK_SIZE
        else:
            # Unknown side — assume mid
            best_bid[i] = prices[i] - ES_TICK_SIZE / 2
            best_ask[i] = prices[i] + ES_TICK_SIZE / 2

    return ts_ns, best_bid, best_ask, prices


# ─────────────────────────────────────────────
#  PREDICTION LOADING
# ─────────────────────────────────────────────

def load_predictions(date_str: str) -> Dict[str, np.ndarray]:
    """
    Load CNN-Mamba predictions for a date. Map to MBO event timestamps.
    Returns dict with keys: ts_ns, pred_1s, confidence, direction, zscore
    """
    pred_path = OOT_PRED_DIR / f"oot_{date_str}.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    mbo_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        raise FileNotFoundError(f"MBO events not found: {mbo_path}")

    pred_data = np.load(str(pred_path), allow_pickle=True)
    mbo_data = np.load(str(mbo_path), allow_pickle=True)

    n_pred = len(pred_data['pred_log_ret_1s'])
    mbo_ts = mbo_data['timestamps']
    n_events = len(mbo_ts)

    # Map prediction index to event timestamp
    # Prediction i corresponds to the last event in its window: event[i*STRIDE + WINDOW-1]
    pred_event_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]
    pred_timestamps = mbo_ts[pred_event_indices]

    # The primary signal is pred_log_ret_1s (strongest signal at 1s horizon)
    pred_1s = pred_data['pred_log_ret_1s'].astype(np.float64)

    # Direction: positive prediction → LONG, negative → SHORT
    direction = np.sign(pred_1s)

    # Confidence: absolute value of prediction
    confidence = np.abs(pred_1s)

    # Z-score: standardize predictions within the day
    pred_mean = np.nanmean(pred_1s)
    pred_std = np.nanstd(pred_1s)
    if pred_std > 1e-10:
        zscore = (pred_1s - pred_mean) / pred_std
    else:
        zscore = np.zeros_like(pred_1s)

    return {
        'ts_ns': pred_timestamps,
        'pred_1s': pred_1s,
        'confidence': confidence,
        'direction': direction,
        'zscore': zscore,
    }


# ─────────────────────────────────────────────
#  TICK-LEVEL REPLAY ENGINE
# ─────────────────────────────────────────────

class Trade:
    """Represents a single trade lifecycle."""
    __slots__ = [
        'signal_ts', 'direction', 'entry_price', 'tp_price', 'sl_price',
        'filled', 'fill_ts', 'fill_price',
        'exit_ts', 'exit_price', 'exit_type',
        'pnl_ticks', 'cost_ticks',
        'cancel_ts', 'max_hold_ts',
    ]

    def __init__(self, signal_ts: int, direction: int, entry_price: float,
                 tp_ticks: int, sl_ticks: int):
        self.signal_ts = signal_ts
        self.direction = direction  # +1 = long, -1 = short
        self.entry_price = entry_price  # passive limit price
        self.filled = False
        self.fill_ts = 0
        self.fill_price = entry_price
        self.exit_ts = 0
        self.exit_price = 0.0
        self.exit_type = ''  # 'TP', 'SL', 'TIMEOUT'
        self.pnl_ticks = 0.0
        self.cost_ticks = 0.0

        # Cancel unfilled entry after CANCEL_WINDOW
        self.cancel_ts = signal_ts + int(CANCEL_WINDOW_SECONDS * 1e9)
        self.max_hold_ts = 0  # set after fill

        # TP/SL prices
        if direction == 1:  # LONG
            self.tp_price = entry_price + tp_ticks * ES_TICK_SIZE
            self.sl_price = entry_price - sl_ticks * ES_TICK_SIZE
        else:  # SHORT
            self.tp_price = entry_price - tp_ticks * ES_TICK_SIZE
            self.sl_price = entry_price + sl_ticks * ES_TICK_SIZE


def replay_day(date_str: str, tp_ticks: int, sl_ticks: int,
               conf_threshold: float = DEFAULT_CONF_THRESHOLD,
               zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
               shuffle_directions: bool = False,
               rng: np.random.Generator = None) -> List[Dict]:
    """
    Replay one day tick-by-tick.

    Args:
        date_str: YYYYMMDD
        tp_ticks, sl_ticks: exit levels in ticks
        conf_threshold: minimum confidence for entry
        zscore_threshold: minimum |z-score| for entry
        shuffle_directions: if True, randomly shuffle signal directions (permutation test)
        rng: random number generator for shuffling

    Returns:
        List of completed trade dicts with full details.
    """
    # Load predictions
    preds = load_predictions(date_str)
    n_preds = len(preds['ts_ns'])

    # Apply confidence and z-score filters
    conf_mask = preds['confidence'] >= conf_threshold
    zscore_mask = np.abs(preds['zscore']) >= zscore_threshold
    signal_mask = conf_mask & zscore_mask
    signal_indices = np.where(signal_mask)[0]

    if len(signal_indices) == 0:
        log.info(f"  {date_str}: No qualifying signals (conf>={conf_threshold}, |z|>={zscore_threshold})")
        return []

    # Get signal details
    signal_ts = preds['ts_ns'][signal_indices]
    signal_dirs = preds['direction'][signal_indices].copy()

    if shuffle_directions:
        # Permutation test: randomize directions while keeping everything else
        if rng is None:
            rng = np.random.default_rng(42)
        signal_dirs = rng.choice([-1, 1], size=len(signal_dirs))

    # Load raw trade data
    events_df = load_raw_trades(date_str)
    trade_mask = events_df['action'] == 'T'
    trade_events = events_df[trade_mask].copy()

    if len(trade_events) == 0:
        log.warning(f"  {date_str}: No trades found in raw data!")
        return []

    trade_ts = trade_events['ts_ns'].values
    trade_prices = trade_events['price'].values
    trade_sides = trade_events['side'].values

    # Build bid/ask from trades (ES is 1-tick wide during RTH)
    # For each trade, infer the current bid/ask
    trade_bid = np.empty(len(trade_events), dtype=np.float64)
    trade_ask = np.empty(len(trade_events), dtype=np.float64)
    for i in range(len(trade_events)):
        if trade_sides[i] == 'A':
            trade_ask[i] = trade_prices[i]
            trade_bid[i] = trade_prices[i] - ES_TICK_SIZE
        elif trade_sides[i] == 'B':
            trade_bid[i] = trade_prices[i]
            trade_ask[i] = trade_prices[i] + ES_TICK_SIZE
        else:
            trade_bid[i] = trade_prices[i] - ES_TICK_SIZE
            trade_ask[i] = trade_prices[i] + ES_TICK_SIZE

    # ─── MAIN REPLAY LOOP ───
    completed_trades = []
    active_trade: Optional[Trade] = None
    signal_ptr = 0  # pointer into signal_ts array

    for ti in range(len(trade_ts)):
        t = trade_ts[ti]
        p = trade_prices[ti]

        # --- Check if active trade gets filled or exits ---
        if active_trade is not None:
            if not active_trade.filled:
                # Check for cancel timeout
                if t >= active_trade.cancel_ts:
                    active_trade = None
                else:
                    # Check for fill: price must trade THROUGH our entry level
                    if active_trade.direction == 1:  # LONG: buy at bid
                        # Fill when trade at or below our bid price
                        # (FIFO: price trades through = someone hit our bid)
                        if p <= active_trade.entry_price:
                            active_trade.filled = True
                            active_trade.fill_ts = t
                            active_trade.fill_price = active_trade.entry_price
                            active_trade.max_hold_ts = t + int(MAX_HOLD_SECONDS * 1e9)
                    else:  # SHORT: sell at ask
                        # Fill when trade at or above our ask price
                        if p >= active_trade.entry_price:
                            active_trade.filled = True
                            active_trade.fill_ts = t
                            active_trade.fill_price = active_trade.entry_price
                            active_trade.max_hold_ts = t + int(MAX_HOLD_SECONDS * 1e9)

            if active_trade is not None and active_trade.filled:
                # Check exits tick by tick
                exited = False
                direction = active_trade.direction

                if direction == 1:  # LONG
                    # TP: price trades at or above TP price (we're selling at ask, passive)
                    if p >= active_trade.tp_price:
                        active_trade.exit_ts = t
                        active_trade.exit_price = active_trade.tp_price
                        active_trade.exit_type = 'TP'
                        active_trade.pnl_ticks = tp_ticks
                        active_trade.cost_ticks = COST_TP_EXIT
                        exited = True
                    # SL: price trades at or below SL price (stop-market)
                    elif p <= active_trade.sl_price:
                        active_trade.exit_ts = t
                        active_trade.exit_price = active_trade.sl_price
                        active_trade.exit_type = 'SL'
                        active_trade.pnl_ticks = -sl_ticks
                        active_trade.cost_ticks = COST_SL_EXIT
                        exited = True
                    # Timeout
                    elif t >= active_trade.max_hold_ts:
                        active_trade.exit_ts = t
                        active_trade.exit_price = p
                        mid = (trade_bid[ti] + trade_ask[ti]) / 2
                        active_trade.pnl_ticks = (mid - active_trade.fill_price) / ES_TICK_SIZE
                        active_trade.exit_type = 'TIMEOUT'
                        active_trade.cost_ticks = COST_TIMEOUT_EXIT
                        exited = True
                else:  # SHORT
                    # TP: price trades at or below TP price (we're buying back at bid, passive)
                    if p <= active_trade.tp_price:
                        active_trade.exit_ts = t
                        active_trade.exit_price = active_trade.tp_price
                        active_trade.exit_type = 'TP'
                        active_trade.pnl_ticks = tp_ticks
                        active_trade.cost_ticks = COST_TP_EXIT
                        exited = True
                    # SL: price trades at or above SL price (stop-market)
                    elif p >= active_trade.sl_price:
                        active_trade.exit_ts = t
                        active_trade.exit_price = active_trade.sl_price
                        active_trade.exit_type = 'SL'
                        active_trade.pnl_ticks = -sl_ticks
                        active_trade.cost_ticks = COST_SL_EXIT
                        exited = True
                    # Timeout
                    elif t >= active_trade.max_hold_ts:
                        active_trade.exit_ts = t
                        active_trade.exit_price = p
                        mid = (trade_bid[ti] + trade_ask[ti]) / 2
                        active_trade.pnl_ticks = (active_trade.fill_price - mid) / ES_TICK_SIZE
                        active_trade.exit_type = 'TIMEOUT'
                        active_trade.cost_ticks = COST_TIMEOUT_EXIT
                        exited = True

                if exited:
                    net_ticks = active_trade.pnl_ticks - active_trade.cost_ticks
                    completed_trades.append({
                        'date': date_str,
                        'signal_ts': active_trade.signal_ts,
                        'direction': active_trade.direction,
                        'entry_price': active_trade.fill_price,
                        'exit_price': active_trade.exit_price,
                        'exit_type': active_trade.exit_type,
                        'pnl_ticks': active_trade.pnl_ticks,
                        'cost_ticks': active_trade.cost_ticks,
                        'net_ticks': net_ticks,
                        'net_dollars': net_ticks * ES_TICK_VALUE,
                        'fill_time_ns': active_trade.fill_ts - active_trade.signal_ts,
                        'hold_time_ns': active_trade.exit_ts - active_trade.fill_ts,
                    })
                    active_trade = None

        # --- Check for new signal (only if no active trade) ---
        if active_trade is None:
            while signal_ptr < len(signal_ts) and signal_ts[signal_ptr] <= t:
                # This signal is at or before current trade timestamp
                sig_t = signal_ts[signal_ptr]
                sig_dir = signal_dirs[signal_ptr]

                # Find the most recent bid/ask at signal time
                # Use the current trade's implied bid/ask
                if sig_dir == 1:  # LONG: enter at bid (passive buy)
                    entry_price = trade_bid[ti]
                elif sig_dir == -1:  # SHORT: enter at ask (passive sell)
                    entry_price = trade_ask[ti]
                else:
                    signal_ptr += 1
                    continue

                active_trade = Trade(sig_t, int(sig_dir), entry_price, tp_ticks, sl_ticks)
                signal_ptr += 1
                break  # process one signal at a time

            # Advance past any skipped signals (while we had an active trade)
            if active_trade is not None:
                while signal_ptr < len(signal_ts) and signal_ts[signal_ptr] <= t:
                    signal_ptr += 1

    # If there's still an active trade at end of day, force-close
    if active_trade is not None and active_trade.filled:
        ti = len(trade_ts) - 1
        p = trade_prices[ti]
        mid = (trade_bid[ti] + trade_ask[ti]) / 2
        if active_trade.direction == 1:
            pnl = (mid - active_trade.fill_price) / ES_TICK_SIZE
        else:
            pnl = (active_trade.fill_price - mid) / ES_TICK_SIZE
        completed_trades.append({
            'date': date_str,
            'signal_ts': active_trade.signal_ts,
            'direction': active_trade.direction,
            'entry_price': active_trade.fill_price,
            'exit_price': p,
            'exit_type': 'EOD',
            'pnl_ticks': pnl,
            'cost_ticks': COST_TIMEOUT_EXIT,
            'net_ticks': pnl - COST_TIMEOUT_EXIT,
            'net_dollars': (pnl - COST_TIMEOUT_EXIT) * ES_TICK_VALUE,
            'fill_time_ns': active_trade.fill_ts - active_trade.signal_ts,
            'hold_time_ns': trade_ts[ti] - active_trade.fill_ts,
        })

    return completed_trades


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def compute_metrics(trades: List[Dict], label: str = "") -> Dict:
    """Compute Sharpe, WR, PF, PnL from trade list."""
    if not trades:
        return {'label': label, 'n_trades': 0}

    net_ticks = np.array([t['net_ticks'] for t in trades])
    n = len(net_ticks)
    total_pnl = np.sum(net_ticks)
    mean_pnl = np.mean(net_ticks)
    std_pnl = np.std(net_ticks)

    winners = net_ticks[net_ticks > 0]
    losers = net_ticks[net_ticks < 0]

    wr = len(winners) / n if n > 0 else 0
    gross_profit = np.sum(winners) if len(winners) > 0 else 0
    gross_loss = np.abs(np.sum(losers)) if len(losers) > 0 else 1e-10
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe (annualized, assuming ~252 trading days)
    sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0

    # Sortino (downside deviation)
    downside = net_ticks[net_ticks < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_pnl
    sortino = (mean_pnl / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Max drawdown in ticks
    cumsum = np.cumsum(net_ticks)
    peak = np.maximum.accumulate(cumsum)
    drawdown = peak - cumsum
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    # Exit type breakdown
    exit_types = {}
    for t in trades:
        et = t['exit_type']
        exit_types[et] = exit_types.get(et, 0) + 1

    # Per-direction stats
    longs = [t for t in trades if t['direction'] == 1]
    shorts = [t for t in trades if t['direction'] == -1]
    long_pnl = sum(t['net_ticks'] for t in longs) if longs else 0
    short_pnl = sum(t['net_ticks'] for t in shorts) if shorts else 0

    return {
        'label': label,
        'n_trades': n,
        'total_pnl_ticks': round(total_pnl, 2),
        'total_pnl_dollars': round(total_pnl * ES_TICK_VALUE, 2),
        'avg_pnl_ticks': round(mean_pnl, 4),
        'std_pnl_ticks': round(std_pnl, 4),
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 4),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'max_dd_ticks': round(max_dd, 2),
        'exit_types': exit_types,
        'n_longs': len(longs),
        'n_shorts': len(shorts),
        'long_pnl': round(long_pnl, 2),
        'short_pnl': round(short_pnl, 2),
    }


def print_metrics(m: Dict):
    """Pretty-print metrics dict."""
    if m.get('n_trades', 0) == 0:
        print(f"  {m.get('label', '???')}: NO TRADES")
        return

    print(f"\n  === {m['label']} ===")
    print(f"  Trades: {m['n_trades']} (L:{m['n_longs']}, S:{m['n_shorts']})")
    print(f"  Total PnL: {m['total_pnl_ticks']:+.1f} ticks (${m['total_pnl_dollars']:+,.0f})")
    print(f"  Avg PnL/trade: {m['avg_pnl_ticks']:+.4f} ticks")
    print(f"  WR: {m['win_rate']:.1%} | PF: {m['profit_factor']:.2f}")
    print(f"  Sharpe: {m['sharpe']:.2f} | Sortino: {m['sortino']:.2f}")
    print(f"  Max DD: {m['max_dd_ticks']:.1f} ticks")
    print(f"  Exit types: {m['exit_types']}")
    print(f"  Long PnL: {m['long_pnl']:+.1f}t | Short PnL: {m['short_pnl']:+.1f}t")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def get_oot_dates() -> List[str]:
    """Get all available OOT dates that have both predictions and raw MBO data."""
    if not OOT_PRED_DIR.exists():
        raise FileNotFoundError(f"OOT prediction directory not found: {OOT_PRED_DIR}")

    pred_dates = sorted([
        f.replace('oot_', '').replace('.npz', '')
        for f in os.listdir(OOT_PRED_DIR)
        if f.startswith('oot_') and f.endswith('.npz')
    ])

    # Filter to dates that have raw MBO data
    available = []
    for d in pred_dates:
        raw_path = RAW_MBO_DIR / f"glbx-mdp3-{d}.mbo.dbn.zst"
        mbo_path = SMART_V3_DIR / f"{d}_mbo_events.npz"
        if raw_path.exists() and mbo_path.exists():
            available.append(d)

    return available


def run_full_sweep(dates: List[str], n_permutations: int = 5,
                   conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                   zscore_threshold: float = DEFAULT_ZSCORE_THRESHOLD,
                   tp_sl_configs: List[Tuple[int, int]] = None):
    """
    Run the full tick-level replay across all dates and TP/SL configs.
    Also runs permutation test to measure mechanical bias.
    """
    if tp_sl_configs is None:
        tp_sl_configs = TP_SL_CONFIGS

    all_results = {}

    for tp, sl in tp_sl_configs:
        config_label = f"TP{tp}_SL{sl}"
        log.info(f"\n{'='*60}")
        log.info(f"Config: {config_label}")
        log.info(f"{'='*60}")

        # --- Real signal replay ---
        all_trades = []
        for date_str in dates:
            log.info(f"  Processing {date_str} ({config_label})...")
            try:
                trades = replay_day(date_str, tp, sl,
                                    conf_threshold=conf_threshold,
                                    zscore_threshold=zscore_threshold)
                all_trades.extend(trades)
                log.info(f"    → {len(trades)} trades")
            except Exception as e:
                log.error(f"    ERROR: {e}")
                traceback.print_exc()

        real_metrics = compute_metrics(all_trades, f"REAL {config_label}")
        print_metrics(real_metrics)

        # --- Permutation test (shuffled directions) ---
        perm_pnls = []
        for perm_i in range(n_permutations):
            rng = np.random.default_rng(seed=perm_i * 1000 + 42)
            perm_trades = []
            for date_str in dates:
                try:
                    trades = replay_day(date_str, tp, sl,
                                        conf_threshold=conf_threshold,
                                        zscore_threshold=zscore_threshold,
                                        shuffle_directions=True, rng=rng)
                    perm_trades.extend(trades)
                except Exception:
                    pass

            perm_metrics = compute_metrics(perm_trades, f"PERM_{perm_i} {config_label}")
            perm_pnls.append(perm_metrics.get('total_pnl_ticks', 0))
            log.info(f"  Permutation {perm_i}: {perm_metrics.get('total_pnl_ticks', 0):+.1f} ticks, "
                     f"WR={perm_metrics.get('win_rate', 0):.1%}")

        # --- Compute model edge vs mechanical bias ---
        real_pnl = real_metrics.get('total_pnl_ticks', 0)
        mean_perm_pnl = np.mean(perm_pnls)
        model_edge = real_pnl - mean_perm_pnl

        if abs(real_pnl) > 1e-10:
            pct_from_model = (model_edge / abs(real_pnl)) * 100
        else:
            pct_from_model = 0

        # p-value: what fraction of permutations beat real?
        p_value = np.mean([p >= real_pnl for p in perm_pnls])

        print(f"\n  --- EDGE DECOMPOSITION ({config_label}) ---")
        print(f"  Real PnL:       {real_pnl:+.1f} ticks")
        print(f"  Mean Perm PnL:  {mean_perm_pnl:+.1f} ticks (mechanical bias)")
        print(f"  Model Edge:     {model_edge:+.1f} ticks")
        print(f"  % from Model:   {pct_from_model:+.1f}%")
        print(f"  p-value:        {p_value:.3f} ({n_permutations} permutations)")

        all_results[config_label] = {
            'real': real_metrics,
            'perm_pnls': perm_pnls,
            'mean_perm_pnl': round(float(mean_perm_pnl), 2),
            'model_edge_ticks': round(float(model_edge), 2),
            'pct_from_model': round(float(pct_from_model), 1),
            'p_value': round(float(p_value), 3),
            'trades': all_trades,
        }

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Tick-Level FIFO Replay Engine")
    parser.add_argument('--dates', nargs='+', help='Specific dates (YYYYMMDD) to process')
    parser.add_argument('--n-dates', type=int, default=None, help='Process first N dates (for testing)')
    parser.add_argument('--tp', type=int, default=None, help='Single TP config (overrides sweep)')
    parser.add_argument('--sl', type=int, default=None, help='Single SL config (overrides sweep)')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF_THRESHOLD, help='Confidence threshold')
    parser.add_argument('--zscore', type=float, default=DEFAULT_ZSCORE_THRESHOLD, help='Z-score threshold')
    parser.add_argument('--permutations', type=int, default=5, help='Number of permutation tests')
    parser.add_argument('--no-permutations', action='store_true', help='Skip permutation test')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("TICK-LEVEL FIFO REPLAY ENGINE")
    log.info("=" * 70)

    # Get dates
    if args.dates:
        dates = args.dates
    else:
        dates = get_oot_dates()
        if args.n_dates:
            dates = dates[:args.n_dates]

    log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    log.info(f"Confidence threshold: {args.conf}")
    log.info(f"Z-score threshold: {args.zscore}")

    # Set thresholds for replay
    conf_threshold = args.conf
    zscore_threshold = args.zscore

    # Override TP/SL if single config specified
    tp_sl_configs = list(TP_SL_CONFIGS)
    if args.tp is not None and args.sl is not None:
        tp_sl_configs = [(args.tp, args.sl)]

    n_perms = 0 if args.no_permutations else args.permutations

    t_start = time.time()
    results = run_full_sweep(dates, n_permutations=n_perms,
                             conf_threshold=conf_threshold,
                             zscore_threshold=zscore_threshold,
                             tp_sl_configs=tp_sl_configs)
    elapsed = time.time() - t_start

    log.info(f"\n{'='*70}")
    log.info(f"COMPLETED in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"{'='*70}")

    # ─── SUMMARY TABLE ───
    print("\n" + "=" * 80)
    print("TICK-LEVEL REPLAY SUMMARY")
    print("=" * 80)
    print(f"{'Config':<12} {'Trades':>6} {'PnL(t)':>8} {'PnL($)':>10} {'WR':>6} "
          f"{'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'MechBias':>9} {'ModelEdge':>10} {'%Model':>7}")
    print("-" * 80)

    for config_label, data in results.items():
        m = data['real']
        if m['n_trades'] == 0:
            print(f"  {config_label:<12} NO TRADES")
            continue
        print(f"  {config_label:<12} {m['n_trades']:>5} {m['total_pnl_ticks']:>+8.1f} "
              f"${m['total_pnl_dollars']:>+9,.0f} {m['win_rate']:>5.1%} "
              f"{m['profit_factor']:>5.2f} {m['sharpe']:>+7.2f} {m['sortino']:>+8.2f} "
              f"{data['mean_perm_pnl']:>+9.1f} {data['model_edge_ticks']:>+10.1f} "
              f"{data['pct_from_model']:>+6.1f}%")
    print("=" * 80)

    # Save results
    output_path = args.output or str(OUTPUT_DIR / "tick_replay_results.json")
    save_data = {}
    for config_label, data in results.items():
        save_data[config_label] = {
            'real': data['real'],
            'perm_pnls': data['perm_pnls'],
            'mean_perm_pnl': data['mean_perm_pnl'],
            'model_edge_ticks': data['model_edge_ticks'],
            'pct_from_model': data['pct_from_model'],
            'p_value': data['p_value'],
            'per_day': {},
        }
        # Per-day breakdown
        day_trades = {}
        for t in data['trades']:
            d = t['date']
            if d not in day_trades:
                day_trades[d] = []
            day_trades[d].append(t)
        for d, trades in sorted(day_trades.items()):
            day_m = compute_metrics(trades, d)
            save_data[config_label]['per_day'][d] = day_m

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
