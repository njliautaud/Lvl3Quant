#!/usr/bin/env python3
"""
Streaming Trade Manager v4 — Prediction-Driven Exits (HC #507)
==============================================================
The KEY innovation vs v3: NO fixed TP/SL. The model's streaming predictions
drive ALL entry/hold/exit decisions. "Even if price moving against us, if
model predicting tons of buying pressure, helps us trade."

Architecture:
    1. Load per-day CNN-Mamba v2 OOT predictions (pred_1s, pred_5s, pred_10s)
    2. Map predictions to timestamps via smart_v3 events
    3. Interpolate minute-bar mid prices to prediction timestamps
    4. State machine: FLAT -> LONG/SHORT -> FLAT
    5. Entry: pressure_score exceeds threshold (directional signal)
    6. Hold:  streaming predictions STILL favor position (even if P&L negative)
    7. Exit:  pressure REVERSES or FADES for N consecutive predictions

Cost model (HC #512 — spread is NOT a cost):
    Entry: taker at ask (long) or bid (short) = mid +/- 0.5 tick
    Exit:  taker at bid (long) or ask (short) = mid -/+ 0.5 tick
    Commission: $4.70 RT = 0.376 ticks RT (the ONLY cost beyond fill price)
    Spread is embedded in bid/ask fills, NOT added as separate cost.

Data sources (all on Jupiter):
    - output/cnn_mamba_v2_all_oot/*_predictions.npz  (46 dates)
    - output/cnn_mamba_v2_bulk_oot_v2/*_predictions.npz  (48 dates)
    - data/processed/mbo_events_smart_v3/*_mbo_events.npz  (timestamps)
    - data/processed/mbo_minute_bars_v1/*.parquet  (mid prices)

Usage:
    # Single config
    python streaming_trade_manager_v4.py --entry-threshold 0.3 --fade-threshold 0.1

    # Full config sweep
    python streaming_trade_manager_v4.py --sweep

    # Custom data paths
    python streaming_trade_manager_v4.py --predictions-dir output/my_preds \\
        --events-dir data/processed/mbo_events_smart_v3 --sweep
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE = Path("/home/jupiter/Lvl3Quant")

# Default data directories
DEFAULT_PRED_DIRS = [
    BASE / "output" / "cnn_mamba_v2_all_oot",
    BASE / "output" / "cnn_mamba_v2_bulk_oot_v2",
    BASE / "output" / "cnn_mamba_v2_bulk_oot",
]
DEFAULT_EVENTS_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3"
DEFAULT_BARS_DIR = BASE / "data" / "processed" / "mbo_minute_bars_v1"
DEFAULT_OUTPUT_DIR = BASE / "output" / "streaming_trade_manager_v4"

# Cost constants (canonical — ES futures, AMP/Rithmic)
# HC #512: Cost = commission ONLY. Spread is NOT a cost — it's the fill price.
# When you buy at the ask, that IS your entry. When you sell at the bid, that IS your exit.
# The spread is already embedded in gross P&L via bid/ask fills. Don't double-count.
TICK_SIZE = 0.25          # ES tick = 0.25 points
TICK_VALUE = 12.50        # $12.50 per tick
COMMISSION_RT_TICKS = 0.376  # $4.70 RT / $12.50 per tick = 0.376 ticks TOTAL RT (the ONLY cost)

# CNN-Mamba v2 model parameters
WINDOW_SIZE = 3000
STRIDE = 250

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('stm_v4')


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class TradeConfig:
    """Configuration for the streaming trade manager."""
    entry_threshold: float = 1.5       # |z-scored pressure| must exceed this to enter
    fade_threshold: float = 0.3        # Directional pressure below this = fading
    fade_n: int = 8                    # N consecutive fading predictions before exit
    max_hold_s: float = 300.0          # Safety max hold (seconds)
    reversal_threshold: float = -0.5   # Pressure must flip past this to trigger reversal exit
    min_hold_preds: int = 8            # Min predictions to hold (prevent instant exit)
    pressure_weight_1s: float = 1.0    # Weight for 1s prediction in pressure score
    pressure_weight_5s: float = 0.5    # Weight for 5s prediction
    pressure_weight_10s: float = 0.3   # Weight for 10s prediction
    cost_ticks: float = COMMISSION_RT_TICKS  # HC #512: commission only, spread is in fills
    min_trades_day: int = 3            # Min trades per day for day to count
    sides: str = "both"                # "long", "short", or "both"
    cooldown_preds: int = 20           # Min predictions between trades (~5s at RTH density)
    warmup_preds: int = 2000           # Skip first N predictions for z-score warmup
    smooth_window: int = 20            # EMA smoothing window for pressure (reduces noise)
    # HC #510: Prediction Memory
    pred_memory_window: int = 20       # Rolling window of past predictions to track accuracy
    min_pred_accuracy: float = 0.0     # Min rolling accuracy to allow entry (0 = disabled, 0.55 = strict)


@dataclass
class Trade:
    """Record of a single completed trade."""
    date: str
    side: str                    # "LONG" or "SHORT"
    entry_time_ns: int = 0
    exit_time_ns: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_pred_idx: int = 0
    exit_pred_idx: int = 0
    gross_pnl_ticks: float = 0.0
    net_pnl_ticks: float = 0.0
    hold_duration_s: float = 0.0
    n_predictions: int = 0       # Predictions during trade
    max_favorable_ticks: float = 0.0   # MFE
    max_adverse_ticks: float = 0.0     # MAE
    exit_reason: str = ""        # "reversal", "fade", "timeout"
    entry_pressure: float = 0.0
    avg_pressure_during: float = 0.0
    peak_pressure_during: float = 0.0


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def discover_prediction_dates(pred_dirs: List[Path]) -> Dict[str, Path]:
    """Discover all per-day prediction files across directories.

    Later directories take priority for duplicate dates (bulk_oot_v2 may overlap with all_oot).
    """
    date_files = {}
    for pred_dir in pred_dirs:
        if not pred_dir.exists():
            continue
        for f in sorted(pred_dir.glob("*_predictions.npz")):
            base = f.stem.replace("_predictions", "")
            if base.startswith("fold_"):
                continue
            if len(base) == 8 and base.isdigit():
                date_files[base] = f
    return date_files


def load_day_data(date_str: str, pred_file: Path, events_dir: Path,
                  bars_dir: Path) -> Optional[dict]:
    """Load all data for a single day.

    Returns dict with:
        predictions: (N, 3) array [pred_1s, pred_5s, pred_10s]
        pred_timestamps_ns: (N,) array of nanosecond timestamps per prediction
        mid_prices: (N,) array of interpolated mid prices at prediction times
        date: str
    """
    # 1. Load predictions
    try:
        pred_data = np.load(str(pred_file), allow_pickle=True)
        predictions = pred_data['predictions']  # (N, 3)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        stride = int(pred_data.get('stride', STRIDE))
        window = int(pred_data.get('window_size', WINDOW_SIZE))
    except Exception as e:
        log.warning(f"Failed to load predictions for {date_str}: {e}")
        return None

    n_preds = len(predictions)
    if n_preds < 100:
        log.warning(f"Too few predictions for {date_str}: {n_preds}")
        return None

    # 2. Load event timestamps
    events_file = events_dir / f"{date_str}_mbo_events.npz"
    if not events_file.exists():
        log.warning(f"No events file for {date_str}")
        return None

    try:
        events_data = np.load(str(events_file), allow_pickle=True)
        event_timestamps = events_data['timestamps']  # (M,) ns
    except Exception as e:
        log.warning(f"Failed to load events for {date_str}: {e}")
        return None

    # Map each prediction to its timestamp via event index
    # Prediction i uses events [i*stride : i*stride + window]
    # The prediction's reference time = timestamp of event at i*stride + window - 1
    pred_timestamps = np.zeros(n_preds, dtype=np.int64)
    max_event_idx = len(event_timestamps) - 1
    for i in range(n_preds):
        event_idx = min(i * stride + window - 1, max_event_idx)
        pred_timestamps[i] = event_timestamps[event_idx]

    # 3. Load minute bars for mid prices
    bars_file = bars_dir / f"{date_str}.parquet"
    if not bars_file.exists():
        log.warning(f"No minute bars for {date_str}")
        return None

    try:
        import pandas as pd
        bars_df = pd.read_parquet(str(bars_file))
    except Exception as e:
        log.warning(f"Failed to load minute bars for {date_str}: {e}")
        return None

    # Build minute-level mid-price lookup (timestamp_ns -> mid_price)
    bar_timestamps_ns = []
    bar_mids = []
    for _, row in bars_df.iterrows():
        ts = row['ts_minute']
        if hasattr(ts, 'timestamp'):
            ts_ns = int(ts.timestamp() * 1e9)
        else:
            ts_ns = int(ts)
        mid = (row['high'] + row['low']) / 2.0  # Approximate mid from OHLC
        bar_timestamps_ns.append(ts_ns)
        bar_mids.append(mid)

    bar_timestamps_ns = np.array(bar_timestamps_ns, dtype=np.int64)
    bar_mids = np.array(bar_mids, dtype=np.float64)

    # Sort by timestamp
    sort_idx = np.argsort(bar_timestamps_ns)
    bar_timestamps_ns = bar_timestamps_ns[sort_idx]
    bar_mids = bar_mids[sort_idx]

    # Interpolate mid prices to prediction timestamps
    mid_prices = np.interp(
        pred_timestamps.astype(np.float64),
        bar_timestamps_ns.astype(np.float64),
        bar_mids
    )

    # Filter to RTH only (13:30-20:00 UTC for DST, 14:30-21:00 for non-DST)
    d = datetime.strptime(date_str, "%Y%m%d")
    is_dst = 3 <= d.month <= 10  # Approximate DST
    if is_dst:
        rth_open_h, rth_close_h = 13, 20
    else:
        rth_open_h, rth_close_h = 14, 21

    midnight_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    rth_open_ns = int((midnight_utc + timedelta(hours=rth_open_h, minutes=30)).timestamp() * 1e9)
    rth_close_ns = int((midnight_utc + timedelta(hours=rth_close_h)).timestamp() * 1e9)

    rth_mask = (pred_timestamps >= rth_open_ns) & (pred_timestamps <= rth_close_ns)
    rth_indices = np.where(rth_mask)[0]

    if len(rth_indices) < 100:
        log.warning(f"Too few RTH predictions for {date_str}: {len(rth_indices)}")
        return None

    return {
        'predictions': predictions[rth_indices],
        'pred_timestamps_ns': pred_timestamps[rth_indices],
        'mid_prices': mid_prices[rth_indices],
        'date': date_str,
        'n_total_preds': n_preds,
        'n_rth_preds': len(rth_indices),
    }


def load_all_days(pred_dirs: List[Path], events_dir: Path, bars_dir: Path,
                  max_days: int = 0) -> List[dict]:
    """Load all available days with complete data."""
    import pandas as pd  # Ensure available

    date_files = discover_prediction_dates(pred_dirs)
    log.info(f"Discovered {len(date_files)} prediction dates")

    all_days = []
    for date_str in sorted(date_files.keys()):
        day_data = load_day_data(date_str, date_files[date_str], events_dir, bars_dir)
        if day_data is not None:
            all_days.append(day_data)
            if max_days > 0 and len(all_days) >= max_days:
                break

    log.info(f"Loaded {len(all_days)} days with complete data")
    return all_days


# ---------------------------------------------------------------------------
# Pressure Score Computation
# ---------------------------------------------------------------------------
def compute_pressure_scores(predictions: np.ndarray, cfg: TradeConfig) -> np.ndarray:
    """Compute directional pressure score from multi-horizon predictions.

    For CNN-Mamba v2 (3 heads: 1s, 5s, 10s):
        pressure = w1*pred_1s + w5*pred_5s + w10*pred_10s

    For v4 multi-head (ntps, eofi, tia, dir):
        pressure = w_ntps*ntps + w_tia*tia  (configured separately)

    Positive pressure = bullish (enter/hold long)
    Negative pressure = bearish (enter/hold short)
    """
    n_heads = predictions.shape[1] if predictions.ndim == 2 else 1

    if n_heads >= 3:
        # CNN-Mamba v2 format: [pred_1s, pred_5s, pred_10s]
        p1s = predictions[:, 0]
        p5s = predictions[:, 1]
        p10s = predictions[:, 2]
        pressure = (cfg.pressure_weight_1s * p1s
                    + cfg.pressure_weight_5s * p5s
                    + cfg.pressure_weight_10s * p10s)
    elif n_heads == 1:
        pressure = predictions[:, 0] * cfg.pressure_weight_1s
    else:
        # 2 heads — weight by 1s and 5s
        pressure = (cfg.pressure_weight_1s * predictions[:, 0]
                    + cfg.pressure_weight_5s * predictions[:, 1])

    return pressure


def z_score_expanding(arr: np.ndarray, min_samples: int = 200) -> np.ndarray:
    """Walk-forward expanding z-score (no lookahead)."""
    z = np.zeros_like(arr)
    cumsum = 0.0
    cumsq = 0.0
    for i in range(len(arr)):
        cumsum += arr[i]
        cumsq += arr[i] ** 2
        n = i + 1
        if n >= min_samples:
            mean = cumsum / n
            var = cumsq / n - mean ** 2
            std = max(np.sqrt(max(var, 0)), 1e-8)
            z[i] = (arr[i] - mean) / std
    return z


# ---------------------------------------------------------------------------
# Streaming Trade Simulation (State Machine)
# ---------------------------------------------------------------------------
def simulate_day(day_data: dict, cfg: TradeConfig) -> List[Trade]:
    """Simulate one day of streaming prediction-driven trading.

    State machine:
        FLAT  -> LONG  (pressure > entry_threshold AND prediction memory says "hot")
        FLAT  -> SHORT (pressure < -entry_threshold AND prediction memory says "hot")
        LONG  -> FLAT  (reversal / fade / timeout)
        SHORT -> FLAT  (reversal / fade / timeout)

    HC #510 PREDICTION MEMORY: Tracks rolling window of past predictions +
    how price reacted after each. Creates self-calibrating confidence:
        - Model "hot" (recent predictions accurate) → enter aggressively
        - Model "cold" (recent predictions wrong) → stay flat or require higher threshold
    """
    predictions = day_data['predictions']
    timestamps = day_data['pred_timestamps_ns']
    mid_prices = day_data['mid_prices']
    date_str = day_data['date']
    n = len(predictions)

    # Compute pressure scores, z-score, then smooth with EMA
    raw_pressure = compute_pressure_scores(predictions, cfg)
    z_pressure = z_score_expanding(raw_pressure, min_samples=200)

    # EMA smoothing to reduce noise (prevents whipsaw entries/exits)
    if cfg.smooth_window > 1:
        alpha = 2.0 / (cfg.smooth_window + 1.0)
        pressure = np.zeros_like(z_pressure)
        pressure[0] = z_pressure[0]
        for j in range(1, len(z_pressure)):
            pressure[j] = alpha * z_pressure[j] + (1.0 - alpha) * pressure[j - 1]
    else:
        pressure = z_pressure

    # -----------------------------------------------------------------------
    # HC #510: PREDICTION MEMORY — track past predictions + price reactions
    # -----------------------------------------------------------------------
    MEMORY_WINDOW = getattr(cfg, 'pred_memory_window', 20)  # Last N predictions
    REACTION_LOOKBACK = 4  # How many predictions later to check price reaction
    MIN_ACCURACY_FOR_ENTRY = getattr(cfg, 'min_pred_accuracy', 0.0)  # 0 = disabled

    pred_memory_signs = np.zeros(n)       # sign of each prediction (+1/-1/0)
    reaction_correct = np.zeros(n)        # 1 if prediction was correct, 0 if wrong, nan if unknown
    reaction_correct[:] = np.nan
    rolling_accuracy = np.zeros(n)        # rolling hit rate over last MEMORY_WINDOW
    rolling_streak = np.zeros(n, dtype=int)  # consecutive correct predictions
    confidence_multiplier = np.ones(n)    # 1.0 = neutral, >1 = hot, <1 = cold

    # Pre-compute prediction correctness: did price move in the predicted direction?
    for i in range(n - REACTION_LOOKBACK):
        pred_sign = np.sign(pressure[i])
        pred_memory_signs[i] = pred_sign
        if pred_sign == 0 or not np.isfinite(mid_prices[i]) or not np.isfinite(mid_prices[i + REACTION_LOOKBACK]):
            continue
        # Price move from prediction time to REACTION_LOOKBACK predictions later
        price_move = mid_prices[i + REACTION_LOOKBACK] - mid_prices[i]
        price_sign = np.sign(price_move)
        # Correct if predicted direction matches actual price move
        reaction_correct[i] = 1.0 if (pred_sign == price_sign and price_sign != 0) else 0.0

    # Compute rolling accuracy and streak
    correct_count = 0
    total_count = 0
    current_streak = 0
    for i in range(n):
        if np.isfinite(reaction_correct[i]):
            total_count += 1
            correct_count += int(reaction_correct[i])
            if reaction_correct[i] == 1.0:
                current_streak = max(current_streak + 1, 1)
            else:
                current_streak = min(current_streak - 1, -1)

        # Remove old values outside window
        old_idx = i - MEMORY_WINDOW
        if old_idx >= 0 and np.isfinite(reaction_correct[old_idx]):
            total_count -= 1
            correct_count -= int(reaction_correct[old_idx])

        if total_count > 0:
            rolling_accuracy[i] = correct_count / total_count
        else:
            rolling_accuracy[i] = 0.5  # neutral when no data

        rolling_streak[i] = current_streak

        # Confidence multiplier: hot model → boost, cold model → suppress
        # Accuracy 0.5 = coin flip → multiplier 1.0 (neutral)
        # Accuracy 0.6 = good → multiplier ~1.2
        # Accuracy 0.4 = bad → multiplier ~0.8
        # Streak 5+ → additional boost
        acc_mult = 0.5 + rolling_accuracy[i]  # range [0.5, 1.5]
        streak_mult = 1.0 + max(0, current_streak - 3) * 0.05  # slight boost for hot streaks
        streak_mult = max(streak_mult, 1.0 - max(0, -current_streak - 3) * 0.05)  # suppress cold
        confidence_multiplier[i] = acc_mult * streak_mult

    # Apply confidence multiplier to pressure (HC #510: prediction-aware scoring)
    adjusted_pressure = pressure * confidence_multiplier
    # -----------------------------------------------------------------------

    trades: List[Trade] = []
    state = "FLAT"
    entry_idx = 0
    entry_price = 0.0
    entry_pressure = 0.0
    fade_count = 0
    cooldown_until = max(cfg.warmup_preds, 0)  # Skip warmup period
    pressures_during: List[float] = []
    mfe = 0.0
    mae = 0.0

    for i in range(n):
        p = adjusted_pressure[i]  # HC #510: use confidence-adjusted pressure
        p_raw = pressure[i]       # Raw pressure for logging
        mid = mid_prices[i]
        ts = timestamps[i]

        if not np.isfinite(p) or mid <= 0:
            continue

        if state == "FLAT":
            # Check cooldown (includes warmup)
            if i < cooldown_until:
                continue

            # HC #510: Require minimum prediction accuracy before entering
            if MIN_ACCURACY_FOR_ENTRY > 0 and rolling_accuracy[i] < MIN_ACCURACY_FOR_ENTRY:
                continue

            # Entry logic (now using confidence-adjusted pressure)
            can_long = cfg.sides in ("both", "long")
            can_short = cfg.sides in ("both", "short")

            if can_long and p > cfg.entry_threshold:
                state = "LONG"
                entry_idx = i
                entry_price = mid + 0.5 * TICK_SIZE  # Taker at ask
                entry_pressure = p
                fade_count = 0
                pressures_during = [p]
                mfe = 0.0
                mae = 0.0

            elif can_short and p < -cfg.entry_threshold:
                state = "SHORT"
                entry_idx = i
                entry_price = mid - 0.5 * TICK_SIZE  # Taker at bid
                entry_pressure = p
                fade_count = 0
                pressures_during = [p]
                mfe = 0.0
                mae = 0.0

        elif state in ("LONG", "SHORT"):
            is_long = state == "LONG"
            preds_held = i - entry_idx
            hold_s = (ts - timestamps[entry_idx]) / 1e9

            # Track MFE/MAE
            if is_long:
                unrealized = (mid - entry_price) / TICK_SIZE
            else:
                unrealized = (entry_price - mid) / TICK_SIZE

            mfe = max(mfe, unrealized)
            mae = min(mae, unrealized)

            pressures_during.append(p)

            # HC #510: Directional pressure uses confidence-adjusted value
            dir_pressure = p if is_long else -p

            # Check min hold
            if preds_held < cfg.min_hold_preds:
                continue

            # Exit checks (in priority order)
            exit_reason = ""

            # (a) REVERSAL: directional pressure flipped against us
            # dir_pressure < reversal_threshold means model predicts OPPOSITE direction
            # reversal_threshold is negative (e.g., -0.5 means model must predict against us at z=-0.5)
            if dir_pressure < cfg.reversal_threshold:
                exit_reason = "reversal"

            # (b) FADE: directional pressure dropped below threshold for N consecutive predictions
            if not exit_reason:
                if dir_pressure < cfg.fade_threshold:
                    fade_count += 1
                else:
                    fade_count = 0

                if fade_count >= cfg.fade_n:
                    exit_reason = "fade"

            # (c) TIMEOUT: max hold exceeded
            if not exit_reason and hold_s >= cfg.max_hold_s:
                exit_reason = "timeout"

            # Execute exit
            if exit_reason:
                if is_long:
                    exit_price = mid - 0.5 * TICK_SIZE  # Taker at bid
                    gross_ticks = (exit_price - entry_price) / TICK_SIZE
                else:
                    exit_price = mid + 0.5 * TICK_SIZE  # Taker at ask
                    gross_ticks = (entry_price - exit_price) / TICK_SIZE

                net_ticks = gross_ticks - COMMISSION_RT_TICKS

                trade = Trade(
                    date=date_str,
                    side=state,
                    entry_time_ns=timestamps[entry_idx],
                    exit_time_ns=ts,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_pred_idx=entry_idx,
                    exit_pred_idx=i,
                    gross_pnl_ticks=round(gross_ticks, 4),
                    net_pnl_ticks=round(net_ticks, 4),
                    hold_duration_s=round(hold_s, 3),
                    n_predictions=preds_held,
                    max_favorable_ticks=round(mfe, 4),
                    max_adverse_ticks=round(mae, 4),
                    exit_reason=exit_reason,
                    entry_pressure=round(entry_pressure, 4),
                    avg_pressure_during=round(float(np.mean(pressures_during)), 4),
                    peak_pressure_during=round(float(np.max(np.abs(pressures_during))), 4),
                )
                trades.append(trade)

                state = "FLAT"
                cooldown_until = i + cfg.cooldown_preds

    # Force close any open position at end of day
    if state != "FLAT":
        i = n - 1
        mid = mid_prices[i]
        ts = timestamps[i]
        is_long = state == "LONG"
        hold_s = (ts - timestamps[entry_idx]) / 1e9

        if is_long:
            exit_price = mid - 0.5 * TICK_SIZE
            gross_ticks = (exit_price - entry_price) / TICK_SIZE
        else:
            exit_price = mid + 0.5 * TICK_SIZE
            gross_ticks = (entry_price - exit_price) / TICK_SIZE

        net_ticks = gross_ticks - COMMISSION_RT_TICKS

        trade = Trade(
            date=date_str,
            side=state,
            entry_time_ns=timestamps[entry_idx],
            exit_time_ns=ts,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_pred_idx=entry_idx,
            exit_pred_idx=i,
            gross_pnl_ticks=round(gross_ticks, 4),
            net_pnl_ticks=round(net_ticks, 4),
            hold_duration_s=round(hold_s, 3),
            n_predictions=i - entry_idx,
            max_favorable_ticks=round(mfe, 4),
            max_adverse_ticks=round(mae, 4),
            exit_reason="eod_close",
            entry_pressure=round(entry_pressure, 4),
            avg_pressure_during=round(float(np.mean(pressures_during)), 4) if pressures_during else 0.0,
            peak_pressure_during=round(float(np.max(np.abs(pressures_during))), 4) if pressures_during else 0.0,
        )
        trades.append(trade)

    return trades


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def compute_metrics(all_trades: List[Trade], cfg: TradeConfig) -> Optional[dict]:
    """Compute comprehensive metrics from trade results."""
    if not all_trades:
        return None

    n = len(all_trades)
    net_pnls = np.array([t.net_pnl_ticks for t in all_trades])
    gross_pnls = np.array([t.gross_pnl_ticks for t in all_trades])
    holds = np.array([t.hold_duration_s for t in all_trades])

    total_net = float(np.sum(net_pnls))
    total_gross = float(np.sum(gross_pnls))
    wins = int(np.sum(net_pnls > 0))
    losses = int(np.sum(net_pnls <= 0))
    wr = wins / n if n > 0 else 0

    gross_win = float(np.sum(net_pnls[net_pnls > 0]))
    gross_loss = float(np.sum(np.abs(net_pnls[net_pnls < 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    # Daily P&L
    daily_pnl = {}
    daily_trades = {}
    for t in all_trades:
        daily_pnl[t.date] = daily_pnl.get(t.date, 0.0) + t.net_pnl_ticks
        daily_trades[t.date] = daily_trades.get(t.date, 0) + 1

    # Filter days with minimum trades
    valid_days = {d: pnl for d, pnl in daily_pnl.items()
                  if daily_trades.get(d, 0) >= cfg.min_trades_day}

    daily_vals = list(valid_days.values()) if valid_days else list(daily_pnl.values())
    n_days = len(daily_vals)

    if n_days > 1 and np.std(daily_vals) > 0:
        sharpe = float(np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252))
    else:
        sharpe = 0.0

    neg_daily = [v for v in daily_vals if v < 0]
    if neg_daily and np.std(neg_daily) > 0:
        sortino = float(np.mean(daily_vals) / np.std(neg_daily) * np.sqrt(252))
    else:
        sortino = 0.0

    green_days = sum(1 for v in daily_vals if v > 0)
    red_days = sum(1 for v in daily_vals if v <= 0)

    # Regime analysis (by month as proxy for green/red periods)
    month_pnl = {}
    for d, pnl in daily_pnl.items():
        month = d[:6]
        month_pnl.setdefault(month, []).append(pnl)

    month_sharpes = {}
    for m, vals in sorted(month_pnl.items()):
        if len(vals) > 1 and np.std(vals) > 0:
            month_sharpes[m] = float(np.mean(vals) / np.std(vals) * np.sqrt(252))
        else:
            month_sharpes[m] = 0.0

    # Regime asymmetry (max difference between any two months)
    sharpe_vals = list(month_sharpes.values())
    if len(sharpe_vals) >= 2:
        max_s = max(abs(s) for s in sharpe_vals)
        regime_asym = abs(max(sharpe_vals) - min(sharpe_vals)) / max_s if max_s > 0 else 999
    else:
        regime_asym = 0.0

    # Day concentration
    daily_abs = [abs(v) for v in daily_vals]
    total_abs = sum(daily_abs)
    day_conc = max(daily_abs) / total_abs if total_abs > 0 else 1.0

    # Max drawdown (cumulative daily PnL)
    cum_pnl = np.cumsum(daily_vals)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Exit reason distribution
    exit_reasons = {}
    for t in all_trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Side distribution
    longs = [t for t in all_trades if t.side == "LONG"]
    shorts = [t for t in all_trades if t.side == "SHORT"]

    # MFE/MAE
    mfes = np.array([t.max_favorable_ticks for t in all_trades])
    maes = np.array([t.max_adverse_ticks for t in all_trades])

    avg_hold = float(np.mean(holds))
    tpd = n / n_days if n_days > 0 else 0

    return {
        'n_trades': n,
        'n_days': n_days,
        'tpd': round(tpd, 1),
        'total_net_ticks': round(total_net, 1),
        'total_gross_ticks': round(total_gross, 1),
        'avg_net_per_trade': round(total_net / n, 3) if n > 0 else 0,
        'avg_gross_per_trade': round(total_gross / n, 3) if n > 0 else 0,
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 3),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'green_days': green_days,
        'red_days': red_days,
        'max_drawdown_ticks': round(max_dd, 1),
        'avg_hold_s': round(avg_hold, 1),
        'median_hold_s': round(float(np.median(holds)), 1),
        'avg_mfe': round(float(np.mean(mfes)), 2),
        'avg_mae': round(float(np.mean(maes)), 2),
        'n_longs': len(longs),
        'n_shorts': len(shorts),
        'long_net': round(sum(t.net_pnl_ticks for t in longs), 1) if longs else 0,
        'short_net': round(sum(t.net_pnl_ticks for t in shorts), 1) if shorts else 0,
        'exit_reasons': {k: round(v / n * 100, 1) for k, v in exit_reasons.items()},
        'regime_asymmetry': round(regime_asym, 3),
        'regime_pass': regime_asym <= 0.50,
        'day_concentration': round(day_conc, 3),
        'day_conc_pass': day_conc <= 0.70,
        'month_sharpes': {k: round(v, 2) for k, v in month_sharpes.items()},
        'daily_pnl': {k: round(v, 2) for k, v in sorted(daily_pnl.items())},
    }


def format_config_label(cfg: TradeConfig) -> str:
    """Create a short label for a config."""
    return (f"e{cfg.entry_threshold}_f{cfg.fade_threshold}_fn{cfg.fade_n}"
            f"_h{int(cfg.max_hold_s)}s_r{cfg.reversal_threshold}_s{cfg.smooth_window}"
            f"_w{cfg.pressure_weight_1s}/{cfg.pressure_weight_5s}/{cfg.pressure_weight_10s}"
            f"_{cfg.sides}")


# ---------------------------------------------------------------------------
# Fast Simulation for Sweep (returns tuples, not Trade objects)
# ---------------------------------------------------------------------------
def simulate_day_fast(pressure: np.ndarray, timestamps: np.ndarray,
                      mid_prices: np.ndarray, date_str: str,
                      entry_threshold: float, fade_threshold: float,
                      fade_n: int, max_hold_s: float,
                      reversal_threshold: float, min_hold_preds: int,
                      cooldown_preds: int, warmup_preds: int,
                      sides: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fast day simulation returning (net_pnl_array, hold_s_array).

    Optimized for sweep: no Trade objects, minimal allocation.
    Returns arrays of net_pnl_ticks and hold_seconds for completed trades.
    """
    n = len(pressure)
    # Pre-allocate trade results (worst case: N/2 trades)
    max_trades = max(n // (cooldown_preds + min_hold_preds + 1), 100)
    net_pnls = np.empty(max_trades, dtype=np.float64)
    hold_secs = np.empty(max_trades, dtype=np.float64)
    n_trades = 0

    state = 0  # 0=FLAT, 1=LONG, -1=SHORT
    entry_idx = 0
    entry_price = 0.0
    fade_count = 0
    cooldown_until = warmup_preds

    can_long = sides in ("both", "long")
    can_short = sides in ("both", "short")

    for i in range(n):
        p = pressure[i]
        mid = mid_prices[i]

        if not np.isfinite(p) or mid <= 0:
            continue

        if state == 0:
            if i < cooldown_until:
                continue

            if can_long and p > entry_threshold:
                state = 1
                entry_idx = i
                entry_price = mid + 0.5 * TICK_SIZE
                fade_count = 0
            elif can_short and p < -entry_threshold:
                state = -1
                entry_idx = i
                entry_price = mid - 0.5 * TICK_SIZE
                fade_count = 0
        else:
            preds_held = i - entry_idx
            if preds_held < min_hold_preds:
                continue

            hold_s = (timestamps[i] - timestamps[entry_idx]) / 1e9
            dir_p = p if state == 1 else -p

            # Exit checks
            exit_now = False

            # Reversal
            if dir_p < reversal_threshold:
                exit_now = True
            # Fade
            elif dir_p < fade_threshold:
                fade_count += 1
                if fade_count >= fade_n:
                    exit_now = True
            else:
                fade_count = 0
            # Timeout
            if not exit_now and hold_s >= max_hold_s:
                exit_now = True

            if exit_now:
                if state == 1:
                    gross = (mid - 0.5 * TICK_SIZE - entry_price) / TICK_SIZE
                else:
                    gross = (entry_price - mid - 0.5 * TICK_SIZE) / TICK_SIZE

                if n_trades < max_trades:
                    net_pnls[n_trades] = gross - COMMISSION_RT_TICKS
                    hold_secs[n_trades] = hold_s
                    n_trades += 1

                state = 0
                cooldown_until = i + cooldown_preds

    # Force close open position
    if state != 0 and n > 0:
        i = n - 1
        mid = mid_prices[i]
        hold_s = (timestamps[i] - timestamps[entry_idx]) / 1e9
        if state == 1:
            gross = (mid - 0.5 * TICK_SIZE - entry_price) / TICK_SIZE
        else:
            gross = (entry_price - mid - 0.5 * TICK_SIZE) / TICK_SIZE

        if n_trades < max_trades:
            net_pnls[n_trades] = gross - COMMISSION_RT_TICKS
            hold_secs[n_trades] = hold_s
            n_trades += 1

    return net_pnls[:n_trades], hold_secs[:n_trades]


def compute_fast_metrics(day_pnls: Dict[str, np.ndarray], min_trades_day: int) -> Optional[dict]:
    """Compute metrics from per-day PnL arrays (fast version for sweep)."""
    all_nets = []
    daily_pnl = {}
    daily_n = {}

    for date_str, pnls in sorted(day_pnls.items()):
        if len(pnls) == 0:
            continue
        all_nets.extend(pnls.tolist())
        daily_pnl[date_str] = float(np.sum(pnls))
        daily_n[date_str] = len(pnls)

    n = len(all_nets)
    if n < 5:
        return None

    net_arr = np.array(all_nets)
    total_net = float(np.sum(net_arr))
    wins = int(np.sum(net_arr > 0))
    wr = wins / n

    gross_win = float(np.sum(net_arr[net_arr > 0]))
    gross_loss = float(np.sum(np.abs(net_arr[net_arr < 0])))
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    # Daily stats
    valid_days = {d: pnl for d, pnl in daily_pnl.items()
                  if daily_n.get(d, 0) >= min_trades_day}
    daily_vals = list(valid_days.values()) if valid_days else list(daily_pnl.values())
    n_days = len(daily_vals)

    if n_days > 1 and np.std(daily_vals) > 0:
        sharpe = float(np.mean(daily_vals) / np.std(daily_vals) * np.sqrt(252))
    else:
        sharpe = 0.0

    neg_daily = [v for v in daily_vals if v < 0]
    sortino = float(np.mean(daily_vals) / np.std(neg_daily) * np.sqrt(252)) \
        if neg_daily and np.std(neg_daily) > 0 else 0.0

    green_days = sum(1 for v in daily_vals if v > 0)

    # Regime analysis
    month_pnl = {}
    for d, pnl in daily_pnl.items():
        month_pnl.setdefault(d[:6], []).append(pnl)

    month_sharpes = {}
    for m, vals in month_pnl.items():
        if len(vals) > 1 and np.std(vals) > 0:
            month_sharpes[m] = float(np.mean(vals) / np.std(vals) * np.sqrt(252))

    sharpe_vals = list(month_sharpes.values())
    if len(sharpe_vals) >= 2:
        max_s = max(abs(s) for s in sharpe_vals)
        regime_asym = abs(max(sharpe_vals) - min(sharpe_vals)) / max_s if max_s > 0 else 999
    else:
        regime_asym = 0.0

    daily_abs = [abs(v) for v in daily_vals]
    total_abs = sum(daily_abs)
    day_conc = max(daily_abs) / total_abs if total_abs > 0 else 1.0

    tpd = n / n_days if n_days > 0 else 0

    return {
        'n_trades': n,
        'n_days': n_days,
        'tpd': round(tpd, 1),
        'total_net_ticks': round(total_net, 1),
        'avg_net_per_trade': round(total_net / n, 3),
        'avg_gross_per_trade': round((total_net + n * COMMISSION_RT_TICKS) / n, 3),
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 3),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'green_days': green_days,
        'red_days': n_days - green_days,
        'regime_asymmetry': round(regime_asym, 3),
        'regime_pass': regime_asym <= 0.50,
        'day_concentration': round(day_conc, 3),
        'day_conc_pass': day_conc <= 0.70,
        'month_sharpes': {k: round(v, 2) for k, v in month_sharpes.items()},
    }


# ---------------------------------------------------------------------------
# Config Sweep (Optimized)
# ---------------------------------------------------------------------------
def precompute_pressure_arrays(all_days: List[dict],
                               weight_combos: List[Tuple],
                               smooth_windows: List[int]) -> dict:
    """Precompute z-scored + smoothed pressure arrays for all (weights, smooth) combos.

    Returns: {(w5s, w10s, smooth): {date: pressure_array}}
    """
    cache = {}
    for w5s, w10s in weight_combos:
        for sw in smooth_windows:
            key = (w5s, w10s, sw)
            day_pressures = {}
            for day_data in all_days:
                cfg_tmp = TradeConfig(pressure_weight_5s=w5s, pressure_weight_10s=w10s,
                                     smooth_window=sw)
                predictions = day_data['predictions']
                raw_pressure = compute_pressure_scores(predictions, cfg_tmp)
                z_pressure = z_score_expanding(raw_pressure, min_samples=200)

                if sw > 1:
                    alpha = 2.0 / (sw + 1.0)
                    pressure = np.zeros_like(z_pressure)
                    pressure[0] = z_pressure[0]
                    for j in range(1, len(z_pressure)):
                        pressure[j] = alpha * z_pressure[j] + (1.0 - alpha) * pressure[j - 1]
                else:
                    pressure = z_pressure

                day_pressures[day_data['date']] = pressure
            cache[key] = day_pressures
    return cache


def run_sweep(all_days: List[dict], output_dir: Path) -> List[dict]:
    """Run a grid sweep over configuration parameters.

    Optimized: precomputes pressure arrays for each (weights, smooth) combo,
    then sweeps only the state-machine params in the inner loop.
    """
    # Sweep grid
    entry_thresholds = [0.8, 1.0, 1.5, 2.0, 2.5]
    fade_thresholds = [0.0, 0.2, 0.5]
    fade_ns = [4, 8, 16]
    max_hold_ss = [60.0, 120.0, 300.0]
    reversal_thresholds = [-0.3, -0.5, -1.0]
    smooth_windows = [10, 20, 40]
    weight_5s_opts = [0.0, 0.5]
    weight_10s_opts = [0.0, 0.3]
    sides_opts = ['both', 'short']
    # HC #510: Prediction memory sweep
    min_accuracy_opts = [0.0, 0.52, 0.55]  # 0 = disabled, 0.52/0.55 = require model to be "hot"

    weight_combos = list(product(weight_5s_opts, weight_10s_opts))
    state_combos = list(product(
        entry_thresholds, fade_thresholds, fade_ns, max_hold_ss,
        reversal_thresholds, sides_opts
    ))

    total = len(weight_combos) * len(smooth_windows) * len(state_combos)
    log.info(f"Sweep: {total} configs ({len(weight_combos)} weight combos x "
             f"{len(smooth_windows)} smooth x {len(state_combos)} state configs) "
             f"x {len(all_days)} days")

    # Phase 1: Precompute pressure arrays
    log.info("Phase 1: Precomputing pressure arrays...")
    t0 = time.time()
    pressure_cache = precompute_pressure_arrays(all_days, weight_combos, smooth_windows)
    log.info(f"  Precomputed {len(pressure_cache)} pressure variants in {time.time()-t0:.1f}s")

    # Prepare day metadata
    day_timestamps = {d['date']: d['pred_timestamps_ns'] for d in all_days}
    day_midprices = {d['date']: d['mid_prices'] for d in all_days}
    day_dates = [d['date'] for d in all_days]

    # Phase 2: Sweep state-machine parameters
    log.info("Phase 2: Sweeping state-machine parameters...")
    all_results = []
    count = 0
    t1 = time.time()

    for (w5s, w10s), sw in product(weight_combos, smooth_windows):
        pressure_key = (w5s, w10s, sw)
        pressures = pressure_cache[pressure_key]

        for (entry_t, fade_t, fade_n, max_hold, rev_t, sides) in state_combos:
            count += 1

            day_pnls = {}
            for date_str in day_dates:
                pressure = pressures[date_str]
                timestamps = day_timestamps[date_str]
                mid_prices = day_midprices[date_str]

                pnls, _ = simulate_day_fast(
                    pressure, timestamps, mid_prices, date_str,
                    entry_threshold=entry_t, fade_threshold=fade_t,
                    fade_n=fade_n, max_hold_s=max_hold,
                    reversal_threshold=rev_t, min_hold_preds=8,
                    cooldown_preds=20, warmup_preds=2000,
                    sides=sides
                )
                if len(pnls) > 0:
                    day_pnls[date_str] = pnls

            if not day_pnls:
                continue

            metrics = compute_fast_metrics(day_pnls, min_trades_day=3)
            if metrics is None:
                continue

            params = {
                'entry_threshold': entry_t,
                'fade_threshold': fade_t,
                'fade_n': fade_n,
                'max_hold_s': max_hold,
                'reversal_threshold': rev_t,
                'smooth_window': sw,
                'pressure_weight_5s': w5s,
                'pressure_weight_10s': w10s,
                'sides': sides,
            }
            cfg_tmp = TradeConfig(**params)
            metrics['config_label'] = format_config_label(cfg_tmp)
            metrics['config'] = params
            all_results.append(metrics)

            if count % 500 == 0:
                elapsed = time.time() - t1
                rate = count / elapsed
                eta = (total - count) / rate
                log.info(f"  Progress: {count}/{total} ({rate:.0f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t1
    log.info(f"Sweep complete: {len(all_results)} valid configs in {elapsed:.1f}s")
    return all_results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_results(results: List[dict], title: str = "RESULTS"):
    """Print formatted results table."""
    print()
    print("=" * 120)
    print(f" {title}")
    print("=" * 120)

    if not results:
        print("  No results.")
        return

    # Sort by Sharpe
    results_sorted = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)

    # Acceptance test (HC #506 R5)
    accepted = [r for r in results_sorted
                if r.get('total_net_ticks', 0) > 0
                and r.get('profit_factor', 0) >= 1.2
                and r.get('sharpe', 0) >= 0.5
                and r.get('regime_pass', False)
                and r.get('day_conc_pass', False)
                and r.get('tpd', 0) >= 3]

    if accepted:
        print(f"\n  ACCEPTED ({len(accepted)} configs pass all gates):")
        print(f"  {'Config':<55} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
              f"{'Net':>8} {'TPD':>5} {'AvgHold':>8} {'Exits':>20}")
        print("  " + "-" * 118)
        for r in accepted[:30]:
            exits = r.get('exit_reasons', {})
            exit_str = " ".join(f"{k}:{v}%" for k, v in sorted(exits.items()))
            print(f"  {r['config_label']:<55} "
                  f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
                  f"{r['profit_factor']:>6.3f} {r['win_rate']*100:>5.1f}% "
                  f"{r['total_net_ticks']:>7.1f}t {r['tpd']:>5.1f} "
                  f"{r['avg_hold_s']:>7.1f}s {exit_str:>20}")

    # Top 30 by Sharpe regardless of acceptance
    print(f"\n  TOP 30 BY SHARPE (net>0, n_trades>=30):")
    top = [r for r in results_sorted
           if r.get('total_net_ticks', 0) > 0 and r.get('n_trades', 0) >= 30]
    if top:
        print(f"  {'Config':<55} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
              f"{'Net':>8} {'N':>5} {'TPD':>5} {'Hold':>6} {'L/S':>8} {'Regime':>7}")
        print("  " + "-" * 118)
        for r in top[:30]:
            regime_str = "PASS" if r.get('regime_pass', False) else "FAIL"
            print(f"  {r['config_label']:<55} "
                  f"{r['sharpe']:>7.2f} {r['sortino']:>8.2f} "
                  f"{r['profit_factor']:>6.3f} {r['win_rate']*100:>5.1f}% "
                  f"{r['total_net_ticks']:>7.1f}t {r['n_trades']:>5} {r['tpd']:>5.1f} "
                  f"{r['avg_hold_s']:>5.1f}s "
                  f"{r.get('n_longs',0)}/{r.get('n_shorts',0):>5} "
                  f"{regime_str:>7}")

    # Cost analysis
    print(f"\n  COST ANALYSIS:")
    if results_sorted:
        avg_gross = np.mean([r['avg_gross_per_trade'] for r in results_sorted
                             if r.get('avg_gross_per_trade', 0) != 0])
        avg_net = np.mean([r['avg_net_per_trade'] for r in results_sorted
                           if r.get('avg_net_per_trade', 0) != 0])
        print(f"  Avg gross/trade: {avg_gross:.3f} ticks")
        print(f"  Avg net/trade:   {avg_net:.3f} ticks")
        print(f"  Cost impact:     {COMMISSION_RT_TICKS:.3f} ticks/RT")

    # MFE/MAE analysis for profitable configs
    profitable = [r for r in results_sorted if r.get('total_net_ticks', 0) > 0]
    if profitable:
        avg_mfe = np.mean([r.get('avg_mfe', 0) for r in profitable[:10]])
        avg_mae = np.mean([r.get('avg_mae', 0) for r in profitable[:10]])
        print(f"\n  Top-10 profitable configs:")
        print(f"    Avg MFE: {avg_mfe:.2f} ticks")
        print(f"    Avg MAE: {avg_mae:.2f} ticks")
        print(f"    MFE/MAE ratio: {abs(avg_mfe/avg_mae):.2f}" if avg_mae != 0 else "")


def print_detailed_analysis(trades: List[Trade], metrics: dict, cfg: TradeConfig):
    """Print detailed analysis for a single config."""
    print()
    print("=" * 100)
    print(f"  DETAILED ANALYSIS: {format_config_label(cfg)}")
    print("=" * 100)

    if not metrics:
        print("  No results.")
        return

    # Summary
    print(f"\n  Performance Summary:")
    print(f"    Trades: {metrics['n_trades']} ({metrics['tpd']:.1f}/day over {metrics['n_days']} days)")
    print(f"    Net P&L: {metrics['total_net_ticks']:.1f} ticks "
          f"(${metrics['total_net_ticks'] * TICK_VALUE:.0f})")
    print(f"    Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
    print(f"    PF: {metrics['profit_factor']:.3f} | WR: {metrics['win_rate']*100:.1f}%")
    print(f"    Avg net/trade: {metrics['avg_net_per_trade']:.3f} ticks")
    print(f"    Max drawdown: {metrics['max_drawdown_ticks']:.1f} ticks")

    # Hold duration
    print(f"\n  Hold Duration:")
    print(f"    Mean: {metrics['avg_hold_s']:.1f}s | Median: {metrics['median_hold_s']:.1f}s")

    # MFE/MAE
    print(f"\n  MFE/MAE:")
    print(f"    Avg MFE: {metrics['avg_mfe']:.2f} ticks | Avg MAE: {metrics['avg_mae']:.2f} ticks")

    # Side analysis
    print(f"\n  Side Analysis:")
    print(f"    Long: {metrics['n_longs']} trades, {metrics['long_net']:.1f} ticks net")
    print(f"    Short: {metrics['n_shorts']} trades, {metrics['short_net']:.1f} ticks net")

    # Exit reasons
    print(f"\n  Exit Reasons:")
    for reason, pct in sorted(metrics.get('exit_reasons', {}).items()):
        print(f"    {reason}: {pct:.1f}%")

    # Monthly breakdown
    print(f"\n  Monthly Sharpe:")
    for month, s in sorted(metrics.get('month_sharpes', {}).items()):
        print(f"    {month}: {s:.2f}")

    # Regime check
    regime_str = "PASS" if metrics.get('regime_pass') else "FAIL"
    conc_str = "PASS" if metrics.get('day_conc_pass') else "FAIL"
    print(f"\n  Acceptance Gates:")
    print(f"    Regime asymmetry: {metrics.get('regime_asymmetry', 0):.3f} ({regime_str})")
    print(f"    Day concentration: {metrics.get('day_concentration', 0):.3f} ({conc_str})")

    # Daily P&L table
    print(f"\n  Daily P&L:")
    daily = metrics.get('daily_pnl', {})
    for d, pnl in sorted(daily.items()):
        bar = "+" * max(0, int(pnl / 2)) if pnl > 0 else "-" * max(0, int(-pnl / 2))
        print(f"    {d}: {pnl:>8.1f}t  {bar}")

    # Per-trade detail (first 20)
    if trades:
        print(f"\n  Sample Trades (first 20):")
        print(f"    {'Date':<10} {'Side':<6} {'Entry':>10} {'Exit':>10} {'Gross':>7} {'Net':>7} "
              f"{'Hold':>7} {'MFE':>6} {'MAE':>7} {'Reason':<10} {'EntryP':>7}")
        for t in trades[:20]:
            print(f"    {t.date:<10} {t.side:<6} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
                  f"{t.gross_pnl_ticks:>7.2f} {t.net_pnl_ticks:>7.2f} "
                  f"{t.hold_duration_s:>6.1f}s {t.max_favorable_ticks:>6.2f} {t.max_adverse_ticks:>7.2f} "
                  f"{t.exit_reason:<10} {t.entry_pressure:>7.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Streaming Trade Manager v4 — Prediction-Driven Exits")
    parser.add_argument("--predictions-dir", type=str, nargs="+",
                        help="Directories with per-day prediction npz files")
    parser.add_argument("--events-dir", type=str,
                        default=str(DEFAULT_EVENTS_DIR),
                        help="Directory with smart_v3 event npz files")
    parser.add_argument("--bars-dir", type=str,
                        default=str(DEFAULT_BARS_DIR),
                        help="Directory with minute bar parquet files")
    parser.add_argument("--output-dir", type=str,
                        default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for results")

    # Sweep mode
    parser.add_argument("--sweep", action="store_true",
                        help="Run full config sweep")

    # Single config mode
    parser.add_argument("--entry-threshold", type=float, default=1.5)
    parser.add_argument("--fade-threshold", type=float, default=0.3)
    parser.add_argument("--fade-n", type=int, default=8)
    parser.add_argument("--max-hold-s", type=float, default=300.0)
    parser.add_argument("--reversal-threshold", type=float, default=-0.5)
    parser.add_argument("--min-hold-preds", type=int, default=8)
    parser.add_argument("--pressure-weight-1s", type=float, default=1.0)
    parser.add_argument("--pressure-weight-5s", type=float, default=0.5)
    parser.add_argument("--pressure-weight-10s", type=float, default=0.3)
    parser.add_argument("--cost-ticks", type=float, default=COMMISSION_RT_TICKS)
    parser.add_argument("--min-trades-day", type=int, default=3)
    parser.add_argument("--sides", type=str, default="both",
                        choices=["long", "short", "both"])
    parser.add_argument("--cooldown-preds", type=int, default=20)
    parser.add_argument("--warmup-preds", type=int, default=2000)
    parser.add_argument("--smooth-window", type=int, default=20)
    # HC #510: Prediction Memory
    parser.add_argument("--pred-memory-window", type=int, default=20,
                        help="Rolling window for prediction accuracy tracking")
    parser.add_argument("--min-pred-accuracy", type=float, default=0.0,
                        help="Min rolling prediction accuracy to allow entry (0=disabled, 0.55=strict)")

    # Other
    parser.add_argument("--max-days", type=int, default=0,
                        help="Limit number of days (0=all)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Setup paths
    if args.predictions_dir:
        pred_dirs = [Path(d) for d in args.predictions_dir]
    else:
        pred_dirs = DEFAULT_PRED_DIRS

    events_dir = Path(args.events_dir)
    bars_dir = Path(args.bars_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add file logging
    fh = logging.FileHandler(str(output_dir / "stm_v4.log"), mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(fh)

    print("=" * 100)
    print(" STREAMING TRADE MANAGER v4 — Prediction-Driven Exits (HC #507)")
    print(" Key: NO fixed TP/SL. Model predictions drive all exit decisions.")
    print(f" Cost model: {COMMISSION_RT_TICKS:.3f} ticks RT (commission only, spread is in bid/ask fills (HC #512))")
    print("=" * 100)

    # Load data
    log.info("Loading data...")
    all_days = load_all_days(pred_dirs, events_dir, bars_dir, max_days=args.max_days)

    if not all_days:
        log.error("No data loaded. Check paths.")
        sys.exit(1)

    total_preds = sum(d['n_rth_preds'] for d in all_days)
    log.info(f"Loaded {len(all_days)} days, {total_preds:,} RTH predictions")
    log.info(f"Date range: {all_days[0]['date']} to {all_days[-1]['date']}")

    if args.sweep:
        # Full sweep mode
        results = run_sweep(all_days, output_dir)

        # Print results
        print_results(results, "CONFIG SWEEP RESULTS")

        # Save results
        save_path = output_dir / "sweep_results.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Saved {len(results)} results to {save_path}")

        # Run detailed analysis on the best config
        if results:
            best = max(results, key=lambda x: x.get('sharpe', 0))
            best_cfg = TradeConfig(**best['config'])
            log.info(f"Best config: {format_config_label(best_cfg)} (Sharpe={best['sharpe']:.2f})")

            # Re-simulate best config to get per-trade detail
            best_trades = []
            for day_data in all_days:
                best_trades.extend(simulate_day(day_data, best_cfg))

            best_metrics = compute_metrics(best_trades, best_cfg)
            print_detailed_analysis(best_trades, best_metrics, best_cfg)

            # Save per-trade detail
            trades_path = output_dir / "best_trades.json"
            with open(trades_path, 'w') as f:
                json.dump([asdict(t) for t in best_trades], f, indent=2, default=str)
            log.info(f"Saved {len(best_trades)} trades to {trades_path}")

    else:
        # Single config mode
        cfg = TradeConfig(
            entry_threshold=args.entry_threshold,
            fade_threshold=args.fade_threshold,
            fade_n=args.fade_n,
            max_hold_s=args.max_hold_s,
            reversal_threshold=args.reversal_threshold,
            min_hold_preds=args.min_hold_preds,
            pressure_weight_1s=args.pressure_weight_1s,
            pressure_weight_5s=args.pressure_weight_5s,
            pressure_weight_10s=args.pressure_weight_10s,
            cost_ticks=args.cost_ticks,
            min_trades_day=args.min_trades_day,
            sides=args.sides,
            cooldown_preds=args.cooldown_preds,
            warmup_preds=args.warmup_preds,
            smooth_window=args.smooth_window,
            pred_memory_window=args.pred_memory_window,
            min_pred_accuracy=args.min_pred_accuracy,
        )

        log.info(f"Config: {format_config_label(cfg)}")

        # Simulate
        all_trades = []
        for day_data in all_days:
            day_trades = simulate_day(day_data, cfg)
            all_trades.extend(day_trades)
            if args.verbose:
                day_net = sum(t.net_pnl_ticks for t in day_trades)
                log.info(f"  {day_data['date']}: {len(day_trades)} trades, "
                         f"net={day_net:.1f}t")

        log.info(f"Total: {len(all_trades)} trades across {len(all_days)} days")

        # Compute and display metrics
        metrics = compute_metrics(all_trades, cfg)
        print_detailed_analysis(all_trades, metrics, cfg)

        # Save results
        save_path = output_dir / "single_run_results.json"
        with open(save_path, 'w') as f:
            json.dump({
                'config': asdict(cfg),
                'config_label': format_config_label(cfg),
                'metrics': metrics,
                'trades': [asdict(t) for t in all_trades],
            }, f, indent=2, default=str)
        log.info(f"Saved to {save_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
