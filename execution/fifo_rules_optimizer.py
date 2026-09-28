#!/usr/bin/env python3
"""
FIFO-Approximate Rules-Based Execution Optimizer
=================================================
Replays processed MBO .npz event data to simulate limit-order fills
with approximate FIFO queue tracking. No raw Databento DBN required.

Designed for Jupiter CPU (16 cores) — runs 14 parallel workers to sweep
rule combinations across 39 OOT dates (Mar 16 → Apr 29, 2026).

Data inputs:
  - MBO events: data/processed/mbo_events/YYYYMMDD_mbo_events.npz
      events (N,6): [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
      timestamps (N,): int64 nanoseconds
      labels_1s/5s/10s/30s: forward mid-price changes in ticks
  - Predictions: output/decay_v4_comprehensive/CNN-Mamba_v2/YYYYMMDD/predictions.npz
      preds (M,3): predictions for 1s, 5s, 10s horizons
      valid_indices (M,): indices into MBO events array

Queue approximation (no order IDs available):
  When placing a virtual limit order at best bid/ask, estimate queue depth
  from Add/Cancel event flow at that price level. Fill when Trade events
  consume enough volume at our price level to reach our position.

Mid-price reconstruction:
  labels_Xs[i] = mid(t_i + X) - mid(t_i).
  For event j at dt < X seconds after fill event i:
    mid(t_j) - mid(t_i) ≈ labels_Xs[i] - labels_Xs[j]
  Use the longest horizon label to minimize approximation error.

ES Futures constants (HC #52):
  Tick = $12.50 (0.25 pts)
  RT Commission = $4.70 = 0.376 ticks
  Passive limit order: commission only (no spread cost)

Usage:
    python fifo_rules_optimizer.py                    # full sweep
    python fifo_rules_optimizer.py --quick             # reduced grid for testing
    python fifo_rules_optimizer.py --workers 8         # custom worker count
    python fifo_rules_optimizer.py --dates 20260316,20260317
"""

import os
import sys
import json
import time
import logging
import argparse
import itertools
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from multiprocessing import Pool, cpu_count
from collections import defaultdict

# ── Paths ───────────────────────────────────────────────────────────────────
LVL3_ROOT = Path('/home/jupiter/Lvl3Quant')
MBO_DIR   = LVL3_ROOT / 'data' / 'processed' / 'mbo_events'
PRED_DIR  = LVL3_ROOT / 'output' / 'decay_v4_comprehensive' / 'CNN-Mamba_v2'
OUT_DIR   = LVL3_ROOT / 'execution' / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ES Constants (HC #52) ──────────────────────────────────────────────────
TICK_USD         = 12.50
COMMISSION_RT    = 4.70
COMMISSION_TICKS = COMMISSION_RT / TICK_USD   # 0.376

# ── Event type / side encodings ────────────────────────────────────────────
EVT_ADD    = 0
EVT_CANCEL = 1
EVT_MODIFY = 2
EVT_TRADE  = 3
EVT_FILL   = 4

SIDE_BID = 0
SIDE_ASK = 1
SIDE_NONE = 2

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger('fifo_rules_optimizer')
log.setLevel(logging.INFO)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class TradeResult:
    """A completed simulated trade."""
    date: str
    signal_idx: int
    direction: str              # 'long' or 'short'
    signal_ts_ns: int
    fill_ts_ns: int
    exit_ts_ns: int
    exit_reason: str            # 'tp', 'sl', 'max_hold', 'signal_decay', 'signal_flip', 'eod'
    pnl_ticks: float            # gross P&L in ticks
    pnl_ticks_net: float        # net P&L after commission
    pnl_dollars: float          # net P&L in dollars
    queue_wait_ns: int          # time from signal to fill
    hold_time_ns: int           # time from fill to exit
    pred_strength: float        # |prediction| at signal
    spread_at_signal: float     # spread in ticks at signal time


@dataclass
class RuleConfig:
    """One rule combination to test."""
    config_id: int
    # Signal generation
    horizon_mode: str           # '1s', '5s', '10s', '2of3', '3of3'
    threshold_pct: float        # top N% by absolute z-score
    direction: str              # 'both', 'long', 'short'
    # Execution
    tp_ticks: float
    sl_ticks: float
    max_hold_ms: int
    cancel_ms: int
    # Signal-based exits
    signal_decay_exit: bool
    decay_threshold: float      # exit if z-score drops below this
    signal_flip_exit: bool      # exit if prediction direction reverses
    # Anti-churn
    min_interval_ms: int        # minimum time between trades


# =============================================================================
# Data loading
# =============================================================================

def load_date_data(date_str: str) -> Optional[dict]:
    """
    Load MBO events and predictions for one date.
    Returns dict with all arrays needed for simulation.
    """
    mbo_path = MBO_DIR / f"{date_str}_mbo_events.npz"
    pred_path = PRED_DIR / date_str / 'predictions.npz'

    if not mbo_path.exists() or not pred_path.exists():
        return None

    try:
        mbo = np.load(mbo_path, allow_pickle=True)
        pred = np.load(pred_path, allow_pickle=True)
    except Exception as e:
        log.warning(f"Failed to load {date_str}: {e}")
        return None

    events = mbo['events']          # (N, 6) float32
    timestamps = mbo['timestamps']  # (N,) int64
    labels_1s = mbo['labels_1s']    # (N,) float32
    labels_5s = mbo['labels_5s']    # (N,) float32
    labels_10s = mbo['labels_10s']  # (N,) float32
    labels_30s = mbo['labels_30s']  # (N,) float32

    preds = pred['preds']           # (M, 3)
    valid_indices = pred['valid_indices']  # (M,) int64

    # Map predictions to timestamps
    safe_idx = np.clip(valid_indices, 0, len(timestamps) - 1)
    pred_ts = timestamps[safe_idx]

    # Compute z-scores for each horizon across this date
    zscores = np.zeros_like(preds)
    for h in range(3):
        col = preds[:, h]
        mu, sigma = col.mean(), col.std()
        if sigma > 1e-8:
            zscores[:, h] = (col - mu) / sigma
        else:
            zscores[:, h] = 0.0

    # Precompute event types and quantities for fast access
    event_types = events[:, 1].astype(np.int8)
    event_sides = events[:, 2].astype(np.int8)
    event_prices = events[:, 3]  # float32
    event_qtys = np.exp(events[:, 4])  # float32 -> actual quantities
    event_spreads = events[:, 5]  # float32

    return {
        'date': date_str,
        'events': events,
        'timestamps': timestamps,
        'event_types': event_types,
        'event_sides': event_sides,
        'event_prices': event_prices,
        'event_qtys': event_qtys,
        'event_spreads': event_spreads,
        'labels_1s': labels_1s,
        'labels_5s': labels_5s,
        'labels_10s': labels_10s,
        'labels_30s': labels_30s,
        'preds': preds,
        'zscores': zscores,
        'valid_indices': safe_idx,
        'pred_ts': pred_ts,
        'n_events': len(events),
        'n_preds': len(preds),
    }


def discover_oot_dates() -> List[str]:
    """Find all dates that have both MBO events and predictions."""
    pred_dates = set()
    if PRED_DIR.exists():
        for d in PRED_DIR.iterdir():
            if d.is_dir() and len(d.name) == 8 and d.name.isdigit():
                if (d / 'predictions.npz').exists():
                    pred_dates.add(d.name)

    mbo_dates = set()
    if MBO_DIR.exists():
        for f in MBO_DIR.glob('*_mbo_events.npz'):
            mbo_dates.add(f.name[:8])

    common = sorted(pred_dates & mbo_dates)
    # Filter to Mar 16 - Apr 29 range
    common = [d for d in common if '20260316' <= d <= '20260429']
    return common


# =============================================================================
# Mid-price reconstruction
# =============================================================================

def get_mid_change(fill_idx: int, event_idx: int,
                   timestamps: np.ndarray,
                   labels_1s: np.ndarray,
                   labels_5s: np.ndarray,
                   labels_10s: np.ndarray,
                   labels_30s: np.ndarray) -> float:
    """
    Estimate mid-price change from fill_idx to event_idx using label differencing.

    The formula: mid(t_j) - mid(t_i) ≈ labels_Xs[i] - labels_Xs[j]
    where X is chosen to be as large as possible (X >> dt for accuracy).

    Use the longest horizon label where both events have valid (non-NaN) values.
    """
    dt_ns = timestamps[event_idx] - timestamps[fill_idx]
    dt_s = dt_ns / 1e9

    # Try labels from longest horizon first (most accurate for short dt)
    # For dt up to ~25s, labels_30s gives best accuracy
    # For dt up to ~8s, labels_10s is also good
    # Fall back to shorter horizons if labels are NaN

    label_pairs = [
        (labels_30s[fill_idx], labels_30s[event_idx], 30.0),
        (labels_10s[fill_idx], labels_10s[event_idx], 10.0),
        (labels_5s[fill_idx],  labels_5s[event_idx],  5.0),
        (labels_1s[fill_idx],  labels_1s[event_idx],  1.0),
    ]

    for fill_label, event_label, horizon_s in label_pairs:
        if np.isnan(fill_label) or np.isnan(event_label):
            continue
        # Only use this horizon if dt < horizon (otherwise approximation breaks)
        if dt_s < horizon_s * 0.9:
            return float(fill_label - event_label)

    # Fallback: if dt >= all horizons, use the longest available anyway
    # (less accurate but better than nothing)
    for fill_label, event_label, horizon_s in label_pairs:
        if not np.isnan(fill_label) and not np.isnan(event_label):
            return float(fill_label - event_label)

    return 0.0


# =============================================================================
# Core simulation engine for one date
# =============================================================================

def simulate_date(data: dict, config: RuleConfig) -> Tuple[List[TradeResult], int]:
    """
    Run fill simulation for one date with one rule config.
    Returns (trade_results, n_signals).
    """
    timestamps = data['timestamps']
    event_types = data['event_types']
    event_sides = data['event_sides']
    event_prices = data['event_prices']
    event_qtys = data['event_qtys']
    event_spreads = data['event_spreads']
    labels_1s = data['labels_1s']
    labels_5s = data['labels_5s']
    labels_10s = data['labels_10s']
    labels_30s = data['labels_30s']
    preds = data['preds']
    zscores = data['zscores']
    valid_indices = data['valid_indices']
    pred_ts = data['pred_ts']
    date_str = data['date']
    n_events = data['n_events']

    # ── Generate signals ──
    signals = generate_signals_for_config(preds, zscores, pred_ts, valid_indices, config)
    n_signals = len(signals)

    if n_signals == 0:
        return [], 0

    # ── Constants ──
    cancel_ns = int(config.cancel_ms * 1_000_000)
    max_hold_ns = int(config.max_hold_ms * 1_000_000)
    min_interval_ns = int(config.min_interval_ms * 1_000_000)

    tp_ticks = config.tp_ticks
    sl_ticks = config.sl_ticks

    results = []
    last_exit_ts = np.int64(0)

    for sig in signals:
        sig_ts = sig['ts_ns']

        # Anti-churn
        if sig_ts < last_exit_ts + min_interval_ns:
            continue

        sig_event_idx = sig['event_idx']

        # Bounds check
        if sig_event_idx >= n_events - 1:
            continue

        spread_at_signal = float(event_spreads[sig_event_idx])

        # Skip illiquid conditions (spread > 4 ticks)
        if spread_at_signal > 4:
            continue

        direction = sig['direction']

        # ── Determine order price and queue tracking parameters ──
        # For long: place bid at best bid = mid - spread/2
        # For short: place ask at best ask = mid + spread/2
        if direction == 'long':
            order_price_rel = -spread_at_signal / 2.0
            order_side = SIDE_BID
            fill_aggressor_side = SIDE_ASK  # sellers hitting our bid
        else:
            order_price_rel = spread_at_signal / 2.0
            order_side = SIDE_ASK
            fill_aggressor_side = SIDE_BID  # buyers lifting our ask

        order_price_rel = round(order_price_rel * 2) / 2  # snap to half-tick

        # ── Estimate initial queue depth ──
        queue_ahead = _estimate_queue_depth_fast(
            event_types, event_sides, event_prices, event_qtys,
            sig_event_idx, order_side, order_price_rel, lookback=1000
        )

        # ── Phase 1: Wait for fill (scan forward for trades at our price) ──
        filled = False
        fill_ts = np.int64(0)
        fill_event_idx = 0
        cumulative_trade_vol = 0.0
        price_tol = 0.3  # half-tick tolerance for price matching

        scan_end = min(sig_event_idx + 5_000_000, n_events)  # cap scan range
        for i in range(sig_event_idx + 1, scan_end):
            ts_i = timestamps[i]

            # Cancel timeout
            if ts_i - sig_ts > cancel_ns:
                break

            etype = event_types[i]
            eside = event_sides[i]
            eprice = event_prices[i]

            # Only care about events at our price level
            if abs(eprice - order_price_rel) > price_tol:
                continue

            if etype == EVT_TRADE or etype == EVT_FILL:
                # Trade at our price — check aggressor side
                if eside == fill_aggressor_side:
                    cumulative_trade_vol += event_qtys[i]
                    if cumulative_trade_vol >= queue_ahead:
                        filled = True
                        fill_ts = ts_i
                        fill_event_idx = i
                        break

            elif etype == EVT_CANCEL:
                # Cancel at our price level reduces queue ahead of us
                # Conservative: assume 50% of cancels are ahead of us
                if queue_ahead > 0:
                    queue_ahead = max(0, queue_ahead - event_qtys[i] * 0.5)

        if not filled:
            continue

        # ── Phase 2: Track position until exit ──
        exit_reason = 'eod'
        exit_ts = timestamps[n_events - 1]
        exit_event_idx = n_events - 1
        exit_mid_change = 0.0

        # For signal decay/flip: find next prediction indices
        fill_pred_idx = sig['pred_idx']
        next_pred_ptr = fill_pred_idx + 1

        # Scan forward from fill to track mid-price and check exits
        # Sample mid-price at trade events (they indicate price discovery moments)
        scan_end = min(fill_event_idx + 5_000_000, n_events)

        for i in range(fill_event_idx + 1, scan_end):
            ts_i = timestamps[i]
            dt_ns = ts_i - fill_ts

            # Max hold check
            if dt_ns > max_hold_ns:
                exit_reason = 'max_hold'
                exit_ts = ts_i
                exit_event_idx = i
                exit_mid_change = get_mid_change(
                    fill_event_idx, i, timestamps,
                    labels_1s, labels_5s, labels_10s, labels_30s
                )
                break

            etype = event_types[i]

            # Only check TP/SL on trade events (price discovery moments)
            if etype == EVT_TRADE:
                mid_change = get_mid_change(
                    fill_event_idx, i, timestamps,
                    labels_1s, labels_5s, labels_10s, labels_30s
                )

                if direction == 'long':
                    favorable = mid_change
                    adverse = -mid_change
                else:
                    favorable = -mid_change
                    adverse = mid_change

                # TP hit
                if favorable >= tp_ticks:
                    exit_reason = 'tp'
                    exit_ts = ts_i
                    exit_event_idx = i
                    exit_mid_change = mid_change
                    break

                # SL hit
                if adverse >= sl_ticks:
                    exit_reason = 'sl'
                    exit_ts = ts_i
                    exit_event_idx = i
                    exit_mid_change = mid_change
                    break

            # Signal decay / flip checks (only at prediction points)
            if (config.signal_decay_exit or config.signal_flip_exit):
                if next_pred_ptr < len(preds) and ts_i >= pred_ts[next_pred_ptr]:
                    current_pred = preds[next_pred_ptr]
                    next_pred_ptr += 1

                    if config.signal_decay_exit:
                        max_abs_pred = float(np.max(np.abs(current_pred)))
                        if max_abs_pred < config.decay_threshold:
                            exit_reason = 'signal_decay'
                            exit_ts = ts_i
                            exit_event_idx = i
                            exit_mid_change = get_mid_change(
                                fill_event_idx, i, timestamps,
                                labels_1s, labels_5s, labels_10s, labels_30s
                            )
                            break

                    if config.signal_flip_exit:
                        avg_pred = float(np.mean(current_pred))
                        if (direction == 'long' and avg_pred < 0) or \
                           (direction == 'short' and avg_pred > 0):
                            exit_reason = 'signal_flip'
                            exit_ts = ts_i
                            exit_event_idx = i
                            exit_mid_change = get_mid_change(
                                fill_event_idx, i, timestamps,
                                labels_1s, labels_5s, labels_10s, labels_30s
                            )
                            break

        # If we hit EOD without explicit exit, compute final mid change
        if exit_reason == 'eod':
            exit_mid_change = get_mid_change(
                fill_event_idx, exit_event_idx, timestamps,
                labels_1s, labels_5s, labels_10s, labels_30s
            )

        # ── Calculate P&L ──
        if direction == 'long':
            pnl_ticks_gross = exit_mid_change
        else:
            pnl_ticks_gross = -exit_mid_change

        # For TP, cap P&L at tp_ticks (we're exiting with limit at exact TP)
        if exit_reason == 'tp':
            pnl_ticks_gross = tp_ticks
        elif exit_reason == 'sl':
            pnl_ticks_gross = -sl_ticks

        # Cost model:
        # COMMISSION_TICKS = 0.376 is RT (both legs combined)
        # Passive entry: no crossing cost
        # TP exit via passive limit: no crossing cost -> total = commission only
        # Other exits via market: +0.5 tick crossing cost
        if exit_reason == 'tp':
            total_cost_ticks = COMMISSION_TICKS  # passive entry + passive exit
        else:
            total_cost_ticks = COMMISSION_TICKS + 0.5  # passive entry + aggressive exit

        pnl_ticks_net = pnl_ticks_gross - total_cost_ticks
        pnl_dollars = pnl_ticks_net * TICK_USD

        queue_wait_ns = int(fill_ts - sig_ts)
        hold_time_ns = int(exit_ts - fill_ts)

        results.append(TradeResult(
            date=date_str,
            signal_idx=sig['pred_idx'],
            direction=direction,
            signal_ts_ns=int(sig_ts),
            fill_ts_ns=int(fill_ts),
            exit_ts_ns=int(exit_ts),
            exit_reason=exit_reason,
            pnl_ticks=float(pnl_ticks_gross),
            pnl_ticks_net=float(pnl_ticks_net),
            pnl_dollars=float(pnl_dollars),
            queue_wait_ns=queue_wait_ns,
            hold_time_ns=hold_time_ns,
            pred_strength=sig['strength'],
            spread_at_signal=spread_at_signal,
        ))
        last_exit_ts = exit_ts

    return results, n_signals


def _estimate_queue_depth_fast(
    event_types: np.ndarray, event_sides: np.ndarray,
    event_prices: np.ndarray, event_qtys: np.ndarray,
    event_idx: int, order_side: int, order_price_rel: float,
    lookback: int = 1000
) -> float:
    """
    Estimate queue depth at a price level by counting recent net Add-Cancel flow.
    """
    start = max(0, event_idx - lookback)

    # Vectorized filtering
    window_types = event_types[start:event_idx + 1]
    window_sides = event_sides[start:event_idx + 1]
    window_prices = event_prices[start:event_idx + 1]
    window_qtys = event_qtys[start:event_idx + 1]

    # Filter: same side AND same price (within tolerance)
    side_match = window_sides == order_side
    price_match = np.abs(window_prices - order_price_rel) < 0.3

    mask = side_match & price_match

    if not mask.any():
        return 1.0

    # Net depth: adds minus cancels minus trades
    matched_types = window_types[mask]
    matched_qtys = window_qtys[mask]

    adds = matched_qtys[matched_types == EVT_ADD].sum()
    cancels = matched_qtys[matched_types == EVT_CANCEL].sum()
    trades = matched_qtys[(matched_types == EVT_TRADE) | (matched_types == EVT_FILL)].sum()

    net = adds - cancels - trades
    return max(1.0, float(net))


# =============================================================================
# Signal generation
# =============================================================================

def generate_signals_for_config(
    preds: np.ndarray,         # (M, 3)
    zscores: np.ndarray,       # (M, 3)
    pred_ts: np.ndarray,       # (M,) int64
    valid_indices: np.ndarray,  # (M,) int64
    config: RuleConfig,
) -> List[dict]:
    """
    Generate trade signals based on horizon mode, threshold, and direction.
    Returns list of signal dicts sorted by timestamp.
    """
    M = len(preds)
    if M == 0:
        return []

    horizon_mode = config.horizon_mode

    if horizon_mode == '1s':
        raw = preds[:, 0]
        z = zscores[:, 0]
    elif horizon_mode == '5s':
        raw = preds[:, 1]
        z = zscores[:, 1]
    elif horizon_mode == '10s':
        raw = preds[:, 2]
        z = zscores[:, 2]
    elif horizon_mode == '2of3':
        signs = np.sign(preds)
        agreement = np.abs(np.sum(signs, axis=1))
        mask_agree = agreement >= 2
        z = np.mean(np.abs(zscores), axis=1) * mask_agree
        raw = np.mean(preds, axis=1) * mask_agree
    elif horizon_mode == '3of3':
        signs = np.sign(preds)
        agreement = np.abs(np.sum(signs, axis=1))
        mask_agree = agreement == 3
        z = np.mean(np.abs(zscores), axis=1) * mask_agree
        raw = np.mean(preds, axis=1) * mask_agree
    else:
        raise ValueError(f"Unknown horizon_mode: {horizon_mode}")

    abs_z = np.abs(z)
    nonzero_mask = abs_z > 1e-8
    if nonzero_mask.sum() == 0:
        return []

    # Threshold: top N% by absolute z-score
    pct = 100.0 - config.threshold_pct
    threshold_val = np.percentile(abs_z[nonzero_mask], max(0, pct))
    signal_mask = abs_z >= threshold_val

    # Direction filter
    if config.direction == 'long':
        signal_mask &= raw > 0
    elif config.direction == 'short':
        signal_mask &= raw < 0

    indices = np.where(signal_mask)[0]

    signals = []
    for idx in indices:
        direction = 'long' if raw[idx] > 0 else 'short'
        signals.append({
            'ts_ns': int(pred_ts[idx]),
            'event_idx': int(valid_indices[idx]),
            'direction': direction,
            'strength': float(abs_z[idx]),
            'pred_idx': int(idx),
        })

    signals.sort(key=lambda s: s['ts_ns'])
    return signals


# =============================================================================
# Metrics computation
# =============================================================================

def compute_metrics(trades: List[TradeResult], n_signals: int, n_dates: int) -> dict:
    """Compute comprehensive metrics from trade results."""
    if not trades:
        return _empty_metrics(n_signals, n_dates)

    n_trades = len(trades)
    pnls = np.array([t.pnl_ticks_net for t in trades])
    pnl_dollars = np.array([t.pnl_dollars for t in trades])

    wins = pnls > 0
    losses = pnls < 0
    n_wins = int(wins.sum())
    n_losses = int(losses.sum())

    total_pnl_ticks = float(pnls.sum())
    total_pnl_dollars = float(pnl_dollars.sum())
    mean_pnl = float(pnls.mean())

    win_rate = n_wins / n_trades

    gross_profit = float(pnls[wins].sum()) if n_wins > 0 else 0.0
    gross_loss = abs(float(pnls[losses].sum())) if n_losses > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else 0.0

    avg_win = float(pnls[wins].mean()) if n_wins > 0 else 0.0
    avg_loss = abs(float(pnls[losses].mean())) if n_losses > 0 else 1e-9
    rr_ratio = avg_win / avg_loss if avg_loss > 1e-10 else 0.0

    # Sharpe / Sortino (per-trade)
    if n_trades > 1 and pnls.std() > 1e-10:
        sharpe = float(pnls.mean() / pnls.std())
        downside = pnls[pnls < 0]
        downside_std = float(downside.std()) if len(downside) > 1 else abs(float(downside.mean())) if len(downside) == 1 else 1e-9
        sortino = float(pnls.mean() / downside_std) if downside_std > 1e-10 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    fill_rate = n_trades / n_signals if n_signals > 0 else 0.0

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t.exit_reason] += 1

    tp_rate = exit_reasons.get('tp', 0) / n_trades
    sl_rate = exit_reasons.get('sl', 0) / n_trades
    max_hold_rate = exit_reasons.get('max_hold', 0) / n_trades
    decay_rate = exit_reasons.get('signal_decay', 0) / n_trades
    flip_rate = exit_reasons.get('signal_flip', 0) / n_trades
    eod_rate = exit_reasons.get('eod', 0) / n_trades

    avg_queue_wait_ms = float(np.mean([t.queue_wait_ns / 1e6 for t in trades]))
    avg_hold_time_ms = float(np.mean([t.hold_time_ns / 1e6 for t in trades]))
    avg_spread = float(np.mean([t.spread_at_signal for t in trades]))

    # Daily P&L breakdown
    daily_pnls = defaultdict(float)
    for t in trades:
        daily_pnls[t.date] += t.pnl_dollars

    daily_pnl_array = np.array(list(daily_pnls.values())) if daily_pnls else np.array([0.0])

    win_days = int((daily_pnl_array > 0).sum())
    loss_days = int((daily_pnl_array < 0).sum())

    # Daily Sharpe/Sortino
    if len(daily_pnl_array) > 1 and daily_pnl_array.std() > 1e-10:
        daily_sharpe = float(daily_pnl_array.mean() / daily_pnl_array.std())
        daily_downside = daily_pnl_array[daily_pnl_array < 0]
        if len(daily_downside) > 1:
            daily_down_std = float(daily_downside.std())
        elif len(daily_downside) == 1:
            daily_down_std = abs(float(daily_downside[0]))
        else:
            daily_down_std = 1e-9
        daily_sortino = float(daily_pnl_array.mean() / daily_down_std) if daily_down_std > 1e-10 else 0.0
    else:
        daily_sharpe = 0.0
        daily_sortino = 0.0

    # Max drawdown (daily)
    cumulative = np.cumsum(daily_pnl_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = float(abs(drawdowns.min())) if len(drawdowns) > 0 else 0.0

    annual_sharpe = daily_sharpe * np.sqrt(252)
    annual_sortino = daily_sortino * np.sqrt(252)

    return {
        'n_dates': n_dates,
        'n_signals': n_signals,
        'n_trades': n_trades,
        'fill_rate': round(fill_rate, 4),
        'total_pnl_ticks': round(total_pnl_ticks, 2),
        'total_pnl_dollars': round(total_pnl_dollars, 2),
        'mean_pnl_per_trade': round(mean_pnl, 4),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 4),
        'rr_ratio': round(rr_ratio, 4),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'annual_sharpe': round(float(annual_sharpe), 4),
        'annual_sortino': round(float(annual_sortino), 4),
        'daily_sharpe': round(daily_sharpe, 4),
        'daily_sortino': round(daily_sortino, 4),
        'tp_rate': round(tp_rate, 4),
        'sl_rate': round(sl_rate, 4),
        'max_hold_rate': round(max_hold_rate, 4),
        'decay_rate': round(decay_rate, 4),
        'flip_rate': round(flip_rate, 4),
        'eod_rate': round(eod_rate, 4),
        'avg_queue_wait_ms': round(avg_queue_wait_ms, 1),
        'avg_hold_time_ms': round(avg_hold_time_ms, 1),
        'avg_spread_at_signal': round(avg_spread, 2),
        'max_drawdown': round(max_drawdown, 2),
        'win_days': win_days,
        'loss_days': loss_days,
        'pct_profitable_days': round(win_days / max(1, win_days + loss_days), 4),
        'best_day': round(float(daily_pnl_array.max()), 2),
        'worst_day': round(float(daily_pnl_array.min()), 2),
        'pnl_per_day': round(float(daily_pnl_array.mean()), 2),
        'trades_per_day': round(n_trades / max(1, n_dates), 1),
    }


def _empty_metrics(n_signals: int, n_dates: int) -> dict:
    return {
        'n_dates': n_dates, 'n_signals': n_signals, 'n_trades': 0,
        'fill_rate': 0, 'total_pnl_ticks': 0, 'total_pnl_dollars': 0,
        'mean_pnl_per_trade': 0, 'win_rate': 0, 'profit_factor': 0,
        'rr_ratio': 0, 'sharpe': 0, 'sortino': 0,
        'annual_sharpe': 0, 'annual_sortino': 0,
        'daily_sharpe': 0, 'daily_sortino': 0,
        'tp_rate': 0, 'sl_rate': 0, 'max_hold_rate': 0,
        'decay_rate': 0, 'flip_rate': 0, 'eod_rate': 0,
        'avg_queue_wait_ms': 0, 'avg_hold_time_ms': 0,
        'avg_spread_at_signal': 0,
        'max_drawdown': 0, 'win_days': 0, 'loss_days': 0,
        'pct_profitable_days': 0, 'best_day': 0, 'worst_day': 0,
        'pnl_per_day': 0, 'trades_per_day': 0,
    }


# =============================================================================
# Grid construction
# =============================================================================

def build_full_grid() -> List[RuleConfig]:
    """Build the comprehensive sweep grid."""
    configs = []
    config_id = 0

    # ── Base grid: all combinations ──
    horizons = ['1s', '5s', '10s', '2of3', '3of3']
    thresholds = [1, 3, 5, 10, 20]
    tp_list = [2, 3, 4, 6, 8, 12]
    sl_list = [2, 3, 4, 6]
    hold_list = [5000, 10000, 30000, 60000]
    cancel_list = [2000, 5000, 10000]
    dir_list = ['both', 'long', 'short']

    for hz, thr, tp, sl, hold, cancel, dirn in itertools.product(
        horizons, thresholds, tp_list, sl_list, hold_list, cancel_list, dir_list
    ):
        # Skip anti-R:R configs where TP <= SL and TP is tiny
        if tp < sl and tp <= 2:
            continue

        configs.append(RuleConfig(
            config_id=config_id,
            horizon_mode=hz, threshold_pct=thr, direction=dirn,
            tp_ticks=tp, sl_ticks=sl, max_hold_ms=hold, cancel_ms=cancel,
            signal_decay_exit=False, decay_threshold=0.3,
            signal_flip_exit=False, min_interval_ms=2000,
        ))
        config_id += 1

    # ── Signal decay/flip variants on representative subset ──
    extra_combos = [
        {'signal_decay_exit': True, 'decay_threshold': 0.2, 'signal_flip_exit': False},
        {'signal_decay_exit': True, 'decay_threshold': 0.3, 'signal_flip_exit': False},
        {'signal_decay_exit': True, 'decay_threshold': 0.5, 'signal_flip_exit': False},
        {'signal_decay_exit': False, 'decay_threshold': 0.3, 'signal_flip_exit': True},
        {'signal_decay_exit': True, 'decay_threshold': 0.3, 'signal_flip_exit': True},
    ]

    for hz in ['1s', '5s', '10s']:
        for thr in [3, 5, 10]:
            for tp in [4, 6, 8]:
                for sl in [2, 3, 4]:
                    if tp < sl and tp <= 2:
                        continue
                    for hold in [10000, 30000]:
                        for extra in extra_combos:
                            configs.append(RuleConfig(
                                config_id=config_id,
                                horizon_mode=hz, threshold_pct=thr, direction='both',
                                tp_ticks=tp, sl_ticks=sl, max_hold_ms=hold, cancel_ms=5000,
                                signal_decay_exit=extra['signal_decay_exit'],
                                decay_threshold=extra['decay_threshold'],
                                signal_flip_exit=extra['signal_flip_exit'],
                                min_interval_ms=2000,
                            ))
                            config_id += 1

    # ── Min-interval variants ──
    for interval in [500, 1000, 5000, 10000]:
        for hz in ['1s', '5s', '10s']:
            for thr in [5, 10]:
                for tp, sl in [(4, 2), (6, 3), (8, 4)]:
                    for hold in [10000, 30000]:
                        configs.append(RuleConfig(
                            config_id=config_id,
                            horizon_mode=hz, threshold_pct=thr, direction='both',
                            tp_ticks=tp, sl_ticks=sl, max_hold_ms=hold, cancel_ms=5000,
                            signal_decay_exit=False, decay_threshold=0.3,
                            signal_flip_exit=False, min_interval_ms=interval,
                        ))
                        config_id += 1

    log.info(f"Built grid with {len(configs)} configurations")
    return configs


def build_quick_grid() -> List[RuleConfig]:
    """Reduced grid for quick testing."""
    configs = []
    config_id = 0

    for hz in ['1s', '5s', '10s', '2of3']:
        for thr in [5, 10, 20]:
            for tp in [4, 6, 8]:
                for sl in [2, 3, 4]:
                    if tp < sl and tp <= 2:
                        continue
                    for hold in [10000, 30000]:
                        configs.append(RuleConfig(
                            config_id=config_id,
                            horizon_mode=hz, threshold_pct=thr, direction='both',
                            tp_ticks=tp, sl_ticks=sl, max_hold_ms=hold, cancel_ms=5000,
                            signal_decay_exit=False, decay_threshold=0.3,
                            signal_flip_exit=False, min_interval_ms=2000,
                        ))
                        config_id += 1

    log.info(f"Built quick grid with {len(configs)} configurations")
    return configs


# =============================================================================
# Worker function for multiprocessing
# =============================================================================

# Global cache for loaded date data (shared via fork)
_GLOBAL_DATE_DATA: Dict[str, dict] = {}


def _worker_run_config(config: RuleConfig) -> Optional[dict]:
    """
    Run one config across all preloaded dates. Called by multiprocessing pool.
    """
    try:
        all_trades = []
        total_signals = 0
        n_dates = 0

        for date_str in sorted(_GLOBAL_DATE_DATA.keys()):
            data = _GLOBAL_DATE_DATA[date_str]
            trades, n_sigs = simulate_date(data, config)
            all_trades.extend(trades)
            total_signals += n_sigs
            n_dates += 1

        metrics = compute_metrics(all_trades, total_signals, n_dates)

        result = {
            'config_id': config.config_id,
            'horizon_mode': config.horizon_mode,
            'threshold_pct': config.threshold_pct,
            'tp_ticks': config.tp_ticks,
            'sl_ticks': config.sl_ticks,
            'max_hold_ms': config.max_hold_ms,
            'cancel_ms': config.cancel_ms,
            'direction': config.direction,
            'signal_decay_exit': config.signal_decay_exit,
            'decay_threshold': config.decay_threshold,
            'signal_flip_exit': config.signal_flip_exit,
            'min_interval_ms': config.min_interval_ms,
        }
        result.update(metrics)
        return result

    except Exception as e:
        log.error(f"Config {config.config_id} failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='FIFO Rules-Based Execution Optimizer')
    parser.add_argument('--quick', action='store_true', help='Use reduced grid for testing')
    parser.add_argument('--workers', type=int, default=14, help='Number of parallel workers')
    parser.add_argument('--dates', type=str, default=None,
                        help='Comma-separated dates (default: all OOT dates)')
    parser.add_argument('--max-configs', type=int, default=None,
                        help='Limit number of configs to test')
    parser.add_argument('--output', type=str, default=None, help='Output JSON path')
    parser.add_argument('--dry-run', action='store_true', help='Show config count only')
    args = parser.parse_args()

    # ── Discover dates ──
    if args.dates:
        dates = args.dates.split(',')
    else:
        dates = discover_oot_dates()

    log.info(f"Found {len(dates)} OOT dates: {dates[0]}..{dates[-1]}")

    # ── Build grid ──
    if args.quick:
        configs = build_quick_grid()
    else:
        configs = build_full_grid()

    if args.max_configs:
        configs = configs[:args.max_configs]

    log.info(f"Grid: {len(configs)} configs x {len(dates)} dates = "
             f"{len(configs) * len(dates)} evaluations")

    if args.dry_run:
        log.info("Dry run — exiting")
        for c in configs[:5]:
            log.info(f"  Sample: {c}")
        return

    # ── Preload all date data ──
    global _GLOBAL_DATE_DATA
    log.info("Loading MBO event data and predictions...")
    t0 = time.time()

    for i, date_str in enumerate(dates):
        data = load_date_data(date_str)
        if data is not None:
            _GLOBAL_DATE_DATA[date_str] = data
            log.info(f"  [{i+1}/{len(dates)}] {date_str}: "
                     f"{data['n_events']:,} events, {data['n_preds']} predictions")
        else:
            log.warning(f"  [{i+1}/{len(dates)}] Skipping {date_str}: missing data")

    load_time = time.time() - t0
    total_events = sum(d['n_events'] for d in _GLOBAL_DATE_DATA.values())
    log.info(f"Loaded {len(_GLOBAL_DATE_DATA)} dates in {load_time:.1f}s ({total_events:,} events)")

    if not _GLOBAL_DATE_DATA:
        log.error("No data loaded — exiting")
        return

    # ── Run sweep ──
    log.info(f"Starting sweep with {args.workers} workers...")
    t0 = time.time()

    results = []
    completed = 0

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_worker_run_config, configs, chunksize=4):
            completed += 1
            if result is not None:
                results.append(result)
            if completed % 100 == 0 or completed == len(configs):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (len(configs) - completed) / rate if rate > 0 else 0
                log.info(f"  Progress: {completed}/{len(configs)} "
                         f"({elapsed:.0f}s, ~{remaining:.0f}s left)")

    sweep_time = time.time() - t0
    log.info(f"Sweep done: {len(results)} results in {sweep_time:.1f}s "
             f"({sweep_time / max(1, len(results)):.2f}s/config)")

    if not results:
        log.error("No results — all configs failed")
        return

    # ── Sort by daily Sortino ──
    results.sort(key=lambda r: r.get('daily_sortino', 0), reverse=True)

    # ── Save JSON ──
    output_path = args.output or str(OUT_DIR / 'fifo_rules_optimizer_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'n_configs': len(configs),
                'n_results': len(results),
                'n_dates': len(_GLOBAL_DATE_DATA),
                'dates': sorted(_GLOBAL_DATE_DATA.keys()),
                'sweep_time_s': round(sweep_time, 1),
                'load_time_s': round(load_time, 1),
                'workers': args.workers,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            },
            'results': results,
        }, f, indent=2, default=str)

    log.info(f"Results saved to {output_path}")

    # ── Print results ──
    print_top_results(results[:30])
    print_best_summary(results)


def print_top_results(results: List[dict]):
    """Print formatted table of top results."""
    print("\n" + "=" * 145)
    print("TOP 30 CONFIGURATIONS BY DAILY SORTINO")
    print("=" * 145)
    header = (f"{'#':>3} {'Hz':>5} {'Thr%':>4} {'Dir':>5} "
              f"{'TP':>3} {'SL':>3} {'Hold':>6} {'Can':>5} "
              f"{'Decay':>5} {'Flip':>4} "
              f"{'Trades':>6} {'FillR':>5} {'WR%':>5} {'PF':>5} "
              f"{'Daily$':>8} {'Tot$':>9} {'Sharpe':>7} {'Sortino':>8} "
              f"{'DSharpe':>8} {'DSortino':>9} {'MDD':>8} {'W/L':>5} "
              f"{'R:R':>5}")
    print(header)
    print("-" * 145)

    for i, r in enumerate(results):
        decay_str = f"{r['decay_threshold']:.1f}" if r.get('signal_decay_exit') else '-'
        flip_str = 'Y' if r.get('signal_flip_exit') else '-'
        dir_str = r['direction'][:1].upper() if r['direction'] != 'both' else 'B'

        line = (
            f"{i+1:>3} "
            f"{r['horizon_mode']:>5} "
            f"{r['threshold_pct']:>4.0f} "
            f"{dir_str:>5} "
            f"{r['tp_ticks']:>3.0f} "
            f"{r['sl_ticks']:>3.0f} "
            f"{r['max_hold_ms']:>6.0f} "
            f"{r['cancel_ms']:>5.0f} "
            f"{decay_str:>5} "
            f"{flip_str:>4} "
            f"{r['n_trades']:>6} "
            f"{r['fill_rate']:>5.2f} "
            f"{r['win_rate']*100:>5.1f} "
            f"{r['profit_factor']:>5.2f} "
            f"{r['pnl_per_day']:>8.1f} "
            f"{r['total_pnl_dollars']:>9.1f} "
            f"{r['sharpe']:>7.3f} "
            f"{r['sortino']:>8.3f} "
            f"{r['daily_sharpe']:>8.3f} "
            f"{r['daily_sortino']:>9.3f} "
            f"{r['max_drawdown']:>8.1f} "
            f"{r['win_days']:>2}/{r['loss_days']:<2} "
            f"{r['rr_ratio']:>5.2f}"
        )
        print(line)

    print("=" * 145)


def print_best_summary(results: List[dict]):
    """Print summary of best configurations."""
    if not results:
        return

    best = results[0]

    print(f"\n{'='*60}")
    print("BEST CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Horizon:        {best['horizon_mode']}")
    print(f"  Threshold:      top {best['threshold_pct']}%")
    print(f"  Direction:      {best['direction']}")
    print(f"  TP/SL:          {best['tp_ticks']}/{best['sl_ticks']} ticks")
    print(f"  Max Hold:       {best['max_hold_ms']}ms")
    print(f"  Cancel:         {best['cancel_ms']}ms")
    print(f"  Signal Decay:   {'Yes (th={})'.format(best['decay_threshold']) if best.get('signal_decay_exit') else 'No'}")
    print(f"  Signal Flip:    {'Yes' if best.get('signal_flip_exit') else 'No'}")
    print(f"  Min Interval:   {best.get('min_interval_ms', 2000)}ms")
    print(f"  ---")
    print(f"  Trades:         {best['n_trades']} ({best['trades_per_day']:.1f}/day)")
    print(f"  Fill Rate:      {best['fill_rate']*100:.1f}%")
    print(f"  Win Rate:       {best['win_rate']*100:.1f}%")
    print(f"  Profit Factor:  {best['profit_factor']:.2f}")
    print(f"  R:R Ratio:      {best['rr_ratio']:.2f}")
    print(f"  ---")
    print(f"  Total P&L:      ${best['total_pnl_dollars']:,.2f}")
    print(f"  Daily P&L:      ${best['pnl_per_day']:,.2f}")
    print(f"  Daily Sharpe:   {best['daily_sharpe']:.3f}")
    print(f"  Daily Sortino:  {best['daily_sortino']:.3f}")
    print(f"  Annual Sharpe:  {best['annual_sharpe']:.3f}")
    print(f"  Annual Sortino: {best['annual_sortino']:.3f}")
    print(f"  Max Drawdown:   ${best['max_drawdown']:,.2f}")
    print(f"  Win/Loss Days:  {best['win_days']}/{best['loss_days']}")
    print(f"  ---")
    print(f"  Exit Reasons:   TP={best['tp_rate']*100:.0f}% SL={best['sl_rate']*100:.0f}% "
          f"MaxHold={best['max_hold_rate']*100:.0f}% EOD={best['eod_rate']*100:.0f}%")
    print(f"  Avg Queue Wait: {best['avg_queue_wait_ms']:.0f}ms")
    print(f"  Avg Hold Time:  {best['avg_hold_time_ms']:.0f}ms")
    print(f"  Avg Spread:     {best['avg_spread_at_signal']:.1f} ticks")

    # Best by each criterion (min 20 trades)
    print(f"\n{'='*60}")
    print("BEST BY DIFFERENT CRITERIA (min 20 trades)")
    print(f"{'='*60}")

    valid = [r for r in results if r['n_trades'] >= 20]
    if not valid:
        valid = results

    for key, label in [
        ('daily_sortino', 'Daily Sortino'),
        ('daily_sharpe', 'Daily Sharpe'),
        ('total_pnl_dollars', 'Total P&L ($)'),
        ('profit_factor', 'Profit Factor'),
        ('win_rate', 'Win Rate'),
    ]:
        best_r = max(valid, key=lambda r: r.get(key, 0))
        print(f"  {label:>18}: {best_r[key]:.3f}  "
              f"[{best_r['horizon_mode']}/{best_r['threshold_pct']}% "
              f"TP={best_r['tp_ticks']} SL={best_r['sl_ticks']} "
              f"Hold={best_r['max_hold_ms']} {best_r['direction']}] "
              f"({best_r['n_trades']} trades, ${best_r['total_pnl_dollars']:,.0f})")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
