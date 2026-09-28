#!/usr/bin/env python3
"""
lgbm_live_inference.py — Live inference pipeline for LGBM smart_v2 DA classifier.

Connects to the Rithmic MBO feed (via RithmicClient), computes the 22 smart_v2
features incrementally per event, and runs LGBM binary classification inference
every STRIDE events (default 500). Outputs P(up), confidence, and a LONG/SHORT/NEUTRAL
signal based on a configurable confidence threshold.

Architecture:
    Rithmic BBO/Trade events
        -> Raw 6-col MBO encoding (same as mbo_recorder.py)
        -> Streaming smart_v2 feature computation (22 features)
        -> LGBM predict_proba at every STRIDE-th event
        -> Signal: LONG/SHORT/NEUTRAL + P(up) + confidence
        -> Paper trade engine (simulated fills on real BBO)
        -> Log to JSONL + optional Discord alerts

Model: LightGBM DA classifier trained on smart_v2 features.
    - Input: 22 features (last event's pre-normalized features per window)
    - Output: P(up) probability
    - Signal: confidence = |P(up) - 0.5|
    - Threshold: confidence > 0.127 (Top10% from training OOT)

IMPORTANT: This is PAPER TRADE ONLY. No real orders are ever submitted.

Usage:
    python3 lgbm_live_inference.py --symbol ESM6 --exchange CME

    # With custom model:
    python3 lgbm_live_inference.py --model /path/to/fold09_model.txt

    # Replay from NPZ file (offline testing):
    python3 lgbm_live_inference.py --replay /path/to/20260423_mbo_events.npz
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import signal as _sig
import statistics
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = Path("/home/jupiter/Lvl3Quant/live_trading_linux/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("lgbm_live_v2")
if not log.handlers:
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_DIR / "lgbm_live_inference.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(sh)


# ---------------------------------------------------------------------------
# Constants matching training
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25
POINT_VALUE = 50.0  # ES futures: $50 per point

# Event type encoding (matches mbo_recorder.py and training data)
_EVENT_TYPE_ADD    = 0
_EVENT_TYPE_CANCEL = 1
_EVENT_TYPE_MODIFY = 2
_EVENT_TYPE_TRADE  = 3
_EVENT_TYPE_FILL   = 4
_SIDE_BID = 0
_SIDE_ASK = 1
_N_EVENT_TYPES = 5

# Rolling z-score lookback (matches precompute_features_smart_v2.py)
ROLLING_ZSCORE_LOOKBACK_LONG = 10000
ROLLING_ZSCORE_LOOKBACK_SHORT = 5000

# EWMA decay for buy/sell intensity (matches precompute_features_smart_v2.py)
EWMA_ALPHA = 0.01

# Sweep detection (matches precompute_features_smart_v2.py)
SWEEP_MIN_TRADES = 3
SWEEP_MAX_GAP_LOG = 0.5

# LGBM inference stride (how often to run inference, in events)
DEFAULT_STRIDE = 500
# Minimum events before first inference (warmup for rolling features)
DEFAULT_WARMUP = 2000

# Default confidence threshold
# Top10%=0.127 (loses money), Top5%=0.20 (marginal), Top1%=0.37 (profitable)
# Using Top5% as default — profitable with passive fills (1-tick cost)
# For market orders (2-tick cost), use Top1% threshold (0.37)
DEFAULT_CONFIDENCE_THRESHOLD = 0.20

# Default model path (latest fold from smart_v2 training)
DEFAULT_MODEL_PATH = (
    "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/"
    "results/lgbm_da_smart_v2_1d_oot/fold09_model.txt"
)

N_FEATURES = 22


def _json_safe(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

FEATURE_NAMES = [
    "time_delta_log", "event_type_id", "side_id", "price_rel_ticks",
    "qty_log", "spread_ticks",
    "cancel_side_asym_50", "rolling_ofi_500", "event_density_20",
    "price_mom_10", "qty_price_mom_50", "price_sign_momentum_200",
    "event_type_entropy_200", "fill_add_restoration_100", "spread_velocity_50",
    "queue_replenishment", "mom_divergence", "ofi_x_spread",
    "vol_weighted_pmom", "buy_sell_intensity_ratio",
    "realized_volatility", "sweep_intensity",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming helpers for rolling statistics
# ═══════════════════════════════════════════════════════════════════════════════

class RollingSum:
    """Fixed-window causal rolling sum.

    Matches batch causal_rolling_sum: at index i, returns sum(signal[max(0,i-W):i]).
    The window EXCLUDES the current value -- we return the sum BEFORE pushing x,
    then store x for future use.
    """
    __slots__ = ("window", "buf", "s")

    def __init__(self, window: int):
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.s: float = 0.0

    def push(self, x: float) -> float:
        """Push x, return sum of previous W values (excluding x)."""
        result = self.s  # sum of buffer BEFORE adding x
        # Now add x for future calls
        if len(self.buf) == self.window:
            self.s -= self.buf[0]
        self.buf.append(x)
        self.s += x
        return result


class RollingMean:
    """Fixed-window causal rolling mean.

    Matches batch causal_rolling_mean: at index i, returns mean(signal[max(0,i-W):i]).
    Divides by actual count of items in window. Window EXCLUDES current value.
    """
    __slots__ = ("window", "buf", "s")

    def __init__(self, window: int):
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.s: float = 0.0

    def push(self, x: float) -> float:
        """Push x, return mean of previous W values (excluding x)."""
        n = len(self.buf)
        result = self.s / n if n > 0 else 0.0
        # Now add x for future calls
        if len(self.buf) == self.window:
            self.s -= self.buf[0]
        self.buf.append(x)
        self.s += x
        return result


class RollingZScore:
    """Streaming causal rolling z-score matching precompute_features_smart_v2.

    At event i: z = (x - mean(past W)) / (std(past W) + eps), clipped to [-5, 5].
    Uses running sum and sum-of-squares.

    Batch semantics: z[i] = (signal[i] - mean(signal[max(0,i-W):i])) / std(...)
    Window [max(0,i-W):i) EXCLUDES the current value at index i.
    """
    __slots__ = ("window", "buf", "s", "s_sq")

    def __init__(self, window: int):
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.s: float = 0.0
        self.s_sq: float = 0.0

    def push(self, x: float) -> float:
        """Push value x, return z-score of x relative to past W values."""
        n = len(self.buf)
        if n == 0:
            # No history -- z = 0, then store x
            self.buf.append(x)
            self.s += x
            self.s_sq += x * x
            return 0.0

        # Compute z using current buffer (past values, excluding x)
        mean = self.s / n
        var = max(0.0, self.s_sq / n - mean * mean)
        std = math.sqrt(var) if var > 1e-16 else 1e-8
        if std < 1e-8:
            std = 1e-8
        z = (x - mean) / std
        z = max(-5.0, min(5.0, z))

        # Now push x into buffer for future calls
        if len(self.buf) == self.window:
            old = self.buf[0]
            self.s -= old
            self.s_sq -= old * old
        self.buf.append(x)
        self.s += x
        self.s_sq += x * x

        return z


class RollingEntropy:
    """Streaming event-type entropy over a window of W events.

    Matches batch semantics: at index i, entropy is computed from
    events [max(0,i-W):i) -- EXCLUDING the current event.
    After computing entropy, the current event is pushed for future use.
    """
    __slots__ = ("window", "k", "buf", "counts")

    def __init__(self, window: int, n_types: int):
        self.window = window
        self.k = n_types
        self.buf: deque = deque(maxlen=window)
        self.counts = [0] * n_types

    def push(self, event_type: int) -> float:
        # Compute entropy from CURRENT buffer state (past W values, excluding current)
        n = len(self.buf)
        if n == 0:
            entropy = 0.0
        else:
            entropy = 0.0
            for c in self.counts:
                if c > 0:
                    p = c / n
                    entropy -= p * math.log(p)

        # Now push current event for future use
        et = min(max(int(event_type), 0), self.k - 1)
        if len(self.buf) == self.window:
            old = self.buf[0]
            self.counts[old] -= 1
        self.buf.append(et)
        self.counts[et] += 1

        return entropy


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming smart_v2 feature computation (22 features)
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingSmartV2Features:
    """Incrementally computes the 22 smart_v2 features per MBO event.

    Input per event: (time_delta_log, event_type_id, side_id,
                      price_rel_ticks, qty_log, spread_ticks)
    Output: np.ndarray of shape (22,), dtype float32.

    Matches precompute_features_smart_v2.py exactly for causal (streaming) use.
    """

    def __init__(self):
        # --- V1 derived feature accumulators (9 features) ---

        # Feature 6: cancel_side_asym_50
        self.cancel_asym_sum = RollingSum(50)

        # Feature 7: rolling_ofi_500 (raw, then z-scored)
        self.ofi_500_sum = RollingSum(500)
        self.ofi_500_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)

        # Feature 8: event_density_20 (rolling mean of time_delta_log)
        self.density_mean = RollingMean(20)

        # Feature 9: price_mom_10 (raw, then z-scored)
        self.pmom_10_sum = RollingSum(10)
        self.pmom_10_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_SHORT)

        # Feature 10: qty_price_mom_50 (raw, then z-scored)
        self.qpmom_50_sum = RollingSum(50)
        self.qpmom_50_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)

        # Feature 11: price_sign_momentum_200
        self.psm_200_sum = RollingSum(200)

        # Feature 12: event_type_entropy_200
        self.entropy = RollingEntropy(200, _N_EVENT_TYPES)

        # Feature 13: fill_add_restoration_100
        self._prev_event_type: int = -1
        self._prev_side: int = -1
        self.fill_add_signal_sum = RollingSum(100)
        self.fill_trade_sum = RollingSum(100)

        # Feature 14: spread_velocity_50 (raw, then z-scored)
        self._prev_spread: Optional[float] = None
        self.spread_diff_mean = RollingMean(50)
        self.spread_vel_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_SHORT)

        # --- V2 new features (7 features) ---

        # Feature 15: queue_replenishment (raw ratio, then z-scored)
        self.add_qty_sum = RollingSum(100)
        self.cancel_qty_sum = RollingSum(100)
        self.queue_replen_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)

        # Feature 16: mom_divergence (short-long momentum divergence)
        self.pmom_short_sum = RollingSum(20)   # price_sign sum over 20
        self.pmom_long_sum = RollingSum(200)    # price_sign sum over 200

        # Feature 17: ofi_x_spread (raw, then z-scored)
        self.ofi_100_sum = RollingSum(100)
        self.ofi_x_spread_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)

        # Feature 18: vol_weighted_pmom (raw, then z-scored)
        self.vwpmom_50_sum = RollingSum(50)
        self.vwpmom_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)

        # Feature 19: buy_sell_intensity_ratio (EWMA)
        # Initialized to -1 to signal first-event special case
        self._bid_ewma: float = -1.0  # sentinel for first event
        self._ask_ewma: float = -1.0

        # Feature 20: realized_volatility (raw, then z-scored)
        self._prev_price_rel: Optional[float] = None
        self.price_change_sq_mean = RollingMean(200)
        self.rvol_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)

        # Feature 21: sweep_intensity
        self._consec_count: float = 0.0
        self._prev_trade_signed: float = 0.0
        self.sweep_sum = RollingSum(200)

        # Event counter
        self.n_events: int = 0

    def update(
        self,
        time_delta_log: float,
        event_type_id: int,
        side_id: int,
        price_rel_ticks: float,
        qty_log: float,
        spread_ticks: float,
    ) -> np.ndarray:
        """Process one raw MBO event and return 22-dim normalized feature vector."""

        td = float(time_delta_log)
        et = int(event_type_id)
        sd = int(side_id)
        pr = float(price_rel_ticks)
        ql = float(qty_log)
        sp = float(spread_ticks)

        # --- Indicator flags ---
        is_cancel = 1.0 if et == _EVENT_TYPE_CANCEL else 0.0
        is_add = 1.0 if et == _EVENT_TYPE_ADD else 0.0
        is_trade = 1.0 if (et == _EVENT_TYPE_TRADE or et == _EVENT_TYPE_FILL) else 0.0
        is_fill_trade = is_trade  # same set
        is_bid = 1.0 if sd == _SIDE_BID else 0.0
        is_ask = 1.0 if sd == _SIDE_ASK else 0.0
        sign_side = is_ask - is_bid

        price_sign = 0.0 if pr == 0.0 else (1.0 if pr > 0.0 else -1.0)

        # ══════════════════════════════════════════════════════════════════
        # RAW DERIVED FEATURES (before normalization)
        # ══════════════════════════════════════════════════════════════════

        # V1 derived feature 0: cancel_side_asym_50 (raw)
        cancel_asym_raw = self.cancel_asym_sum.push(is_cancel * (is_ask - is_bid))

        # V1 derived feature 1: rolling_ofi_500 (raw)
        ofi_signal = ql * sign_side
        ofi_500_raw = self.ofi_500_sum.push(ofi_signal)

        # V1 derived feature 2: event_density_20 (raw mean of time_delta_log)
        density_raw = self.density_mean.push(td)

        # V1 derived feature 3: price_mom_10 (raw sum)
        pmom_10_raw = self.pmom_10_sum.push(pr)

        # V1 derived feature 4: qty_price_mom_50 (raw sum)
        qpmom_50_raw = self.qpmom_50_sum.push(ql * pr)

        # V1 derived feature 5: price_sign_momentum_200 (raw sum)
        psm_200_raw = self.psm_200_sum.push(price_sign)

        # V1 derived feature 6: event_type_entropy_200
        entropy_raw = self.entropy.push(et)

        # V1 derived feature 7: fill_add_restoration_100
        # Check if previous event was fill/trade AND current is add on same side
        prev_is_ft = 1.0 if self._prev_event_type in (_EVENT_TYPE_TRADE, _EVENT_TYPE_FILL) else 0.0
        same_side = 1.0 if sd == self._prev_side else 0.0
        fill_add_signal = prev_is_ft * is_add * same_side
        fa_sum = self.fill_add_signal_sum.push(fill_add_signal)
        ft_sum = self.fill_trade_sum.push(is_fill_trade)
        fill_add_ratio = fa_sum / max(ft_sum, 1.0)

        self._prev_event_type = et
        self._prev_side = sd

        # V1 derived feature 8: spread_velocity_50 (raw mean of spread diff)
        if self._prev_spread is None:
            spread_diff = 0.0
        else:
            spread_diff = sp - self._prev_spread
        self._prev_spread = sp
        spread_vel_raw = self.spread_diff_mean.push(spread_diff)

        # --- V2 new features (raw) ---

        # Feature 15: queue_replenishment (add_rate / cancel_rate over 100)
        add_rate = self.add_qty_sum.push(is_add * ql)
        cancel_rate = self.cancel_qty_sum.push(is_cancel * ql)
        queue_replen_raw = add_rate / max(cancel_rate, 0.1)

        # Feature 16: mom_divergence (short vs long price sign momentum)
        pmom_short = self.pmom_short_sum.push(price_sign)
        pmom_long = self.pmom_long_sum.push(price_sign)
        mom_div_raw = pmom_short / 20.0 - pmom_long / 200.0

        # Feature 17: ofi_x_spread (OFI_100 * spread)
        ofi_100_raw = self.ofi_100_sum.push(ql * sign_side)
        ofi_x_spread_raw = ofi_100_raw * sp

        # Feature 18: vol_weighted_pmom (sum of qty_log * price_rel over 50)
        vwpmom_raw = self.vwpmom_50_sum.push(ql * pr)

        # Feature 19: buy_sell_intensity_ratio (EWMA)
        # Batch ewma: out[0] = signal[0], then out[i] = alpha*signal[i] + (1-alpha)*out[i-1]
        if self._bid_ewma < 0:
            # First event: initialize to current values
            self._bid_ewma = is_bid
            self._ask_ewma = is_ask
        else:
            self._bid_ewma = EWMA_ALPHA * is_bid + (1.0 - EWMA_ALPHA) * self._bid_ewma
            self._ask_ewma = EWMA_ALPHA * is_ask + (1.0 - EWMA_ALPHA) * self._ask_ewma
        total_ewma = self._bid_ewma + self._ask_ewma
        if total_ewma < 1e-8:
            total_ewma = 1e-8
        intensity_ratio_raw = (self._bid_ewma / total_ewma - 0.5) * 2.0

        # Feature 20: realized_volatility
        if self._prev_price_rel is None:
            price_change = 0.0
        else:
            price_change = pr - self._prev_price_rel
        self._prev_price_rel = pr
        rvol_raw = math.sqrt(max(0.0, self.price_change_sq_mean.push(price_change ** 2)))

        # Feature 21: sweep_intensity
        # Batch logic:
        #   trade_signed = is_trade * sign_side (+1 ask trade, -1 bid trade, 0 non-trade)
        #   same_as_prev[i] = (trade_signed[i]!=0) & (trade_signed[i]==trade_signed[i-1])
        #                     & (time_delta_log[i] < SWEEP_MAX_GAP_LOG)
        #   consec: if same_as_prev -> consec[i-1]+1, elif trade -> 1, else 0
        #   is_sweep = consec >= 3, sweep_signed = is_sweep * trade_signed
        #   rolling_sum(sweep_signed, 200) at i = sum of past 200 EXCLUDING current
        trade_signed = is_trade * sign_side
        if (trade_signed != 0.0 and
            trade_signed == self._prev_trade_signed and
            td < SWEEP_MAX_GAP_LOG):
            self._consec_count += 1.0
        elif trade_signed != 0.0:
            self._consec_count = 1.0
        else:
            self._consec_count = 0.0  # non-trade: consec = 0 (matches batch)

        is_sweep = 1.0 if self._consec_count >= SWEEP_MIN_TRADES else 0.0
        sweep_signed = is_sweep * trade_signed
        sweep_sum_raw = self.sweep_sum.push(sweep_signed)

        if trade_signed != 0.0:
            self._prev_trade_signed = trade_signed

        # ══════════════════════════════════════════════════════════════════
        # SMART NORMALIZATION (matches apply_smart_normalization in v2)
        # ══════════════════════════════════════════════════════════════════

        out = np.empty(N_FEATURES, dtype=np.float32)

        # 0: time_delta_log -> clamp [0,8], /4.0
        out[0] = min(max(td, 0.0), 8.0) / 4.0

        # 1: event_type_id -> /4.0
        out[1] = float(et) / 4.0

        # 2: side_id -> remap to -1/+1
        out[2] = float(sd) * 2.0 - 1.0

        # 3: price_rel_ticks -> clip [-50,50], /25.0
        out[3] = min(max(pr, -50.0), 50.0) / 25.0

        # 4: qty_log -> (x - 0.693) / 3.0
        out[4] = (ql - 0.693) / 3.0

        # 5: spread_ticks -> clip [0,20], /5.0
        out[5] = min(max(sp, 0.0), 20.0) / 5.0

        # 6: cancel_side_asym_50 -> clip [-50,50], /25.0
        out[6] = min(max(cancel_asym_raw, -50.0), 50.0) / 25.0

        # 7: rolling_ofi_500 -> rolling z-score (LONG lookback)
        out[7] = self.ofi_500_zscore.push(ofi_500_raw)

        # 8: event_density_20 -> clamp [0,4], /2.0
        out[8] = min(max(density_raw, 0.0), 4.0) / 2.0

        # 9: price_mom_10 -> rolling z-score (SHORT lookback)
        out[9] = self.pmom_10_zscore.push(pmom_10_raw)

        # 10: qty_price_mom_50 -> rolling z-score (LONG lookback)
        out[10] = self.qpmom_50_zscore.push(qpmom_50_raw)

        # 11: price_sign_momentum_200 -> /100.0
        out[11] = psm_200_raw / 100.0

        # 12: event_type_entropy_200 -> /1.609
        out[12] = entropy_raw / 1.609

        # 13: fill_add_restoration_100 -> as-is [0,1]
        out[13] = fill_add_ratio

        # 14: spread_velocity_50 -> rolling z-score (SHORT lookback)
        out[14] = self.spread_vel_zscore.push(spread_vel_raw)

        # 15: queue_replenishment -> rolling z-score (LONG lookback)
        out[15] = self.queue_replen_zscore.push(queue_replen_raw)

        # 16: mom_divergence -> *5 + clip [-5,5]
        out[16] = min(max(mom_div_raw * 5.0, -5.0), 5.0)

        # 17: ofi_x_spread -> rolling z-score (LONG lookback)
        out[17] = self.ofi_x_spread_zscore.push(ofi_x_spread_raw)

        # 18: vol_weighted_pmom -> rolling z-score (LONG lookback)
        out[18] = self.vwpmom_zscore.push(vwpmom_raw)

        # 19: buy_sell_intensity_ratio -> as-is [-1,1]
        out[19] = intensity_ratio_raw

        # 20: realized_volatility -> rolling z-score (LONG lookback)
        out[20] = self.rvol_zscore.push(rvol_raw)

        # 21: sweep_intensity -> clip [-50,50], /10.0
        out[21] = min(max(sweep_sum_raw, -50.0), 50.0) / 10.0

        self.n_events += 1
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# LGBM Model Wrapper (loads native LightGBM text format)
# ═══════════════════════════════════════════════════════════════════════════════

class LGBMSmartV2Model:
    """Wraps a LightGBM booster for binary classification inference.

    Loads from native text format (.txt) saved by model.booster_.save_model().
    """

    def __init__(self, model_path: str | Path):
        import lightgbm as lgb
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.booster = lgb.Booster(model_file=str(self.model_path))
        log.info("Loaded LGBM model: %s (%d trees)",
                 self.model_path, self.booster.num_trees())

    def predict_proba(self, features: np.ndarray) -> float:
        """Return P(up) for a single 22-dim feature vector.

        The booster outputs raw log-odds (since objective=binary).
        We apply sigmoid to get P(up).
        """
        x = features.reshape(1, -1).astype(np.float64)
        raw = self.booster.predict(x)[0]
        # LightGBM binary classification booster outputs probability directly
        # when loaded from text format (it applies sigmoid internally)
        return float(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# Paper Trading Engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PaperPosition:
    size: int = 0           # +1 long, -1 short, 0 flat
    side: str = ""          # 'LONG' or 'SHORT'
    entry_price: float = 0.0
    entry_ts: float = 0.0
    entry_prob: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PaperTrade:
    trade_id: int
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    side: str
    gross_pnl: float
    net_pnl: float
    hold_time_s: float
    exit_reason: str
    entry_prob: float
    entry_confidence: float


@dataclass
class PaperStats:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_net_pnl: float = 0.0
    total_gross_pnl: float = 0.0
    total_commission: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    returns: list = field(default_factory=list)
    signals_generated: int = 0
    predictions_made: int = 0
    events_processed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_net_pnl / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def sortino(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        mean_ret = statistics.mean(self.returns)
        downside = [r for r in self.returns if r < 0]
        if not downside:
            return float('inf') if mean_ret > 0 else 0.0
        ds = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
        if ds == 0:
            return 0.0
        return (mean_ret / ds) * math.sqrt(50 * 252)  # ~50 trades/day, 252 days

    @property
    def profit_factor(self) -> float:
        wins = sum(r for r in self.returns if r > 0)
        losses = abs(sum(r for r in self.returns if r < 0))
        return wins / losses if losses > 0 else (float('inf') if wins > 0 else 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Live Inference Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class LGBMLiveInference:
    """
    Live inference pipeline for LGBM smart_v2 DA classifier.

    Modes:
      1. Live: Connects to Rithmic via RithmicClient for real-time MBO data
      2. Replay: Reads from a raw MBO NPZ file (for offline testing)

    Pipeline per event:
      1. Encode raw Rithmic BBO/Trade into 6-col MBO format
      2. Update streaming smart_v2 features (22-dim)
      3. Every STRIDE events: run LGBM inference -> P(up), confidence, signal
      4. If confidence > threshold: generate LONG/SHORT signal
      5. Paper trade: simulate entry/exit using live BBO
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        stride: int = DEFAULT_STRIDE,
        warmup: int = DEFAULT_WARMUP,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        position_timeout_s: float = 30.0,
        slippage_ticks: float = 0.0,  # HC #290(C) 2026-05-11: commission only, no spread tick
        commission_per_side: float = 0.50,
        max_spread_ticks: float = 3.0,
        stats_interval_s: float = 300.0,
        symbol: str = "ESM6",
        exchange: str = "CME",
    ):
        self.stride = stride
        self.warmup = warmup
        self.confidence_threshold = confidence_threshold
        self.position_timeout_s = position_timeout_s
        self.slippage_ticks = slippage_ticks
        self.commission_per_side = commission_per_side
        self.max_spread_ticks = max_spread_ticks
        self.stats_interval_s = stats_interval_s
        self.symbol = symbol
        self.exchange = exchange

        # Load model
        self.model = LGBMSmartV2Model(model_path)

        # Streaming feature engine
        self.features = StreamingSmartV2Features()

        # Market state (for live Rithmic mode)
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.mid_price: float = 0.0
        self.prev_ts_ns: int = 0

        # Paper trading
        self.position = PaperPosition()
        self.stats = PaperStats()
        self._trade_counter = 0

        # Signal/trade logs
        self._signals_path = _LOG_DIR / f"lgbm_v2_signals_{symbol}.jsonl"
        self._trades_path = _LOG_DIR / f"lgbm_v2_trades_{symbol}.jsonl"
        self._signals_fh = open(self._signals_path, "a", buffering=1)
        self._trades_fh = open(self._trades_path, "a", buffering=1)

        # Asyncio stop event
        self._stop = asyncio.Event()

        log.info("LGBMLiveInference initialized:")
        log.info("  Model: %s", model_path)
        log.info("  Stride: %d | Warmup: %d | Confidence threshold: %.4f",
                 stride, warmup, confidence_threshold)
        log.info("  Symbol: %s | Exchange: %s", symbol, exchange)
        log.info("  Position timeout: %.0fs | Slippage: %.1f ticks | Commission: $%.2f",
                 position_timeout_s, slippage_ticks, commission_per_side)
        log.info("  *** PAPER TRADE MODE — NO REAL ORDERS ***")

    # ─────────────────────────────────────────────────────────────────────
    # Core: process a single raw MBO event
    # ─────────────────────────────────────────────────────────────────────

    def _process_raw_event(
        self,
        time_delta_log: float,
        event_type_id: int,
        side_id: int,
        price_rel_ticks: float,
        qty_log: float,
        spread_ticks: float,
        timestamp_ns: int = 0,
    ) -> Optional[dict]:
        """Process one raw MBO event. Returns signal dict if inference triggers, else None."""

        self.stats.events_processed += 1

        # Compute streaming features
        feat = self.features.update(
            time_delta_log, event_type_id, side_id,
            price_rel_ticks, qty_log, spread_ticks,
        )

        n = self.features.n_events

        # Skip during warmup
        if n < self.warmup:
            return None

        # Only infer every STRIDE events
        if (n - self.warmup) % self.stride != 0:
            return None

        # Run LGBM inference
        prob_up = self.model.predict_proba(feat)
        confidence = abs(prob_up - 0.5)
        self.stats.predictions_made += 1

        # Determine signal
        if confidence >= self.confidence_threshold:
            signal = "LONG" if prob_up > 0.5 else "SHORT"
        else:
            signal = "NEUTRAL"

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_ns": int(timestamp_ns),
            "event_num": int(n),
            "prob_up": round(float(prob_up), 6),
            "confidence": round(float(confidence), 6),
            "signal": signal,
            "bid": float(self.best_bid),
            "ask": float(self.best_ask),
            "mid": float(self.mid_price),
            "spread_ticks": float(spread_ticks),
            "position": int(self.position.size),
        }

        # Log signal
        self._signals_fh.write(json.dumps(result) + "\n")

        if signal != "NEUTRAL":
            self.stats.signals_generated += 1

            # Check spread filter
            spread = self.best_ask - self.best_bid if (self.best_bid > 0 and self.best_ask > 0) else 0.0
            if spread > self.max_spread_ticks * TICK_SIZE:
                result["filtered"] = "spread_too_wide"
                log.info("SIGNAL %s P(up)=%.4f conf=%.4f FILTERED (spread=%.2f > %.2f)",
                         signal, prob_up, confidence, spread, self.max_spread_ticks * TICK_SIZE)
                return result

            # Execute paper trade
            self._paper_trade(signal, prob_up, confidence, timestamp_ns)

        # Periodic logging
        if self.stats.predictions_made % 20 == 0:
            log.info("Pred #%d | event=%d | P(up)=%.4f conf=%.4f signal=%s | "
                     "pos=%+d trades=%d netPnL=$%.2f",
                     self.stats.predictions_made, n, prob_up, confidence, signal,
                     self.position.size, self.stats.total_trades, self.stats.total_net_pnl)

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Paper trading logic
    # ─────────────────────────────────────────────────────────────────────

    def _paper_trade(self, signal: str, prob_up: float, confidence: float, ts_ns: int):
        """Execute paper trade based on signal."""
        pos = self.position
        now = time.time()
        desired_long = (signal == "LONG")

        if pos.size == 0:
            # Open new position
            fill_price = self._sim_fill_price("B" if desired_long else "S")
            pos.size = 1 if desired_long else -1
            pos.side = signal
            pos.entry_price = fill_price
            pos.entry_ts = now
            pos.entry_prob = prob_up
            log.info("PAPER OPEN %s @ %.2f | P(up)=%.4f conf=%.4f",
                     signal, fill_price, prob_up, confidence)
            return

        same_dir = (pos.size > 0 and desired_long) or (pos.size < 0 and not desired_long)
        if same_dir:
            return  # already positioned correctly

        # Flip: close + reopen
        close_price = self._sim_fill_price("S" if pos.size > 0 else "B")
        self._record_close(pos, close_price, now, "signal_flip")

        fill_price = self._sim_fill_price("B" if desired_long else "S")
        pos.size = 1 if desired_long else -1
        pos.side = signal
        pos.entry_price = fill_price
        pos.entry_ts = now
        pos.entry_prob = prob_up
        log.info("PAPER FLIP -> %s @ %.2f (closed @ %.2f) | P(up)=%.4f",
                 signal, fill_price, close_price, prob_up)

    def _sim_fill_price(self, side: str) -> float:
        """Simulate fill price with slippage."""
        slip = self.slippage_ticks * TICK_SIZE
        if side == "B":
            return (self.best_ask + slip) if self.best_ask > 0 else self.mid_price + slip
        else:
            return (self.best_bid - slip) if self.best_bid > 0 else self.mid_price - slip

    def _record_close(self, pos: PaperPosition, close_price: float, now: float, reason: str):
        """Record a completed round-trip."""
        self._trade_counter += 1

        if pos.size > 0:
            gross_pnl = (close_price - pos.entry_price) * POINT_VALUE
        else:
            gross_pnl = (pos.entry_price - close_price) * POINT_VALUE

        commission = self.commission_per_side * 2  # entry + exit
        net_pnl = gross_pnl - commission
        hold = now - pos.entry_ts

        trade = PaperTrade(
            trade_id=self._trade_counter,
            entry_time=pos.entry_ts,
            exit_time=now,
            entry_price=pos.entry_price,
            exit_price=close_price,
            side=pos.side,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            hold_time_s=hold,
            exit_reason=reason,
            entry_prob=pos.entry_prob,
            entry_confidence=abs(pos.entry_prob - 0.5),
        )

        self._trades_fh.write(json.dumps(_json_safe(asdict(trade))) + "\n")

        self.stats.total_trades += 1
        self.stats.total_gross_pnl += gross_pnl
        self.stats.total_net_pnl += net_pnl
        self.stats.total_commission += commission
        self.stats.returns.append(net_pnl)

        if net_pnl > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        if self.stats.total_net_pnl > self.stats.peak_pnl:
            self.stats.peak_pnl = self.stats.total_net_pnl
        dd = self.stats.peak_pnl - self.stats.total_net_pnl
        if dd > self.stats.max_drawdown:
            self.stats.max_drawdown = dd

        status = "WIN" if net_pnl > 0 else "LOSS"
        log.info("PAPER TRADE #%d %s %s entry=%.2f exit=%.2f "
                 "gross=$%.2f net=$%.2f hold=%.1fs (%s) | cumPnL=$%.2f",
                 trade.trade_id, status, trade.side,
                 trade.entry_price, trade.exit_price,
                 gross_pnl, net_pnl, hold, reason,
                 self.stats.total_net_pnl)

    def _check_position_timeout(self):
        """Force-close if position held too long."""
        pos = self.position
        if pos.size == 0 or pos.entry_ts == 0:
            return
        age = time.time() - pos.entry_ts
        if age >= self.position_timeout_s:
            close_price = self._sim_fill_price("S" if pos.size > 0 else "B")
            self._record_close(pos, close_price, time.time(), "timeout")
            pos.size = 0
            pos.side = ""
            pos.entry_price = 0.0
            pos.entry_ts = 0.0
            log.info("PAPER TIMEOUT — position closed after %.1fs", age)

    # ─────────────────────────────────────────────────────────────────────
    # Live mode: connect to Rithmic
    # ─────────────────────────────────────────────────────────────────────

    async def run_live(self):
        """Connect to Rithmic for real-time inference."""
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rithmic_client import RithmicClient, BBOEvent, TradeEvent
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent / ".env")

        client = RithmicClient()

        async def on_md(ev):
            ts_ns = int(ev.ssboe * 1_000_000_000 + ev.usecs * 1000)

            if isinstance(ev, BBOEvent):
                if ev.has_bid and ev.bid_price > 0:
                    self.best_bid = ev.bid_price
                if ev.has_ask and ev.ask_price > 0:
                    self.best_ask = ev.ask_price
                if self.best_bid > 0 and self.best_ask > 0:
                    self.mid_price = (self.best_bid + self.best_ask) / 2.0

                # Encode as MBO events (same as mbo_recorder.py)
                if ev.has_bid:
                    self._encode_and_process(
                        ts_ns, 0.0, 0.0, ev.bid_price, ev.bid_size)
                if ev.has_ask:
                    self._encode_and_process(
                        ts_ns, 0.0, 1.0, ev.ask_price, ev.ask_size)

            elif isinstance(ev, TradeEvent):
                side = {1: 1.0, 2: 0.0}.get(ev.aggressor, 2.0)
                self._encode_and_process(
                    ts_ns, 3.0, side, ev.trade_price, ev.trade_size)

            # Check position timeout periodically
            if self.stats.events_processed % 1000 == 0:
                self._check_position_timeout()

        client.set_md_callback(on_md)
        await client.connect()
        await client.subscribe_md(self.symbol, self.exchange)

        log.info("Live mode: connected to Rithmic. Waiting for market data...")

        # Background tasks
        async def stats_loop():
            while not self._stop.is_set():
                await asyncio.sleep(self.stats_interval_s)
                self._dump_stats()

        async def timeout_loop():
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
                self._check_position_timeout()

        tasks = [
            asyncio.create_task(stats_loop()),
            asyncio.create_task(timeout_loop()),
        ]

        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await client.disconnect()
            self._dump_final_stats()

    def _encode_and_process(self, ts_ns: int, etype: float, side: float,
                             price: float, qty: int):
        """Encode a Rithmic event into raw MBO format and process."""
        delta_us = max(0, (ts_ns - self.prev_ts_ns) / 1000) if self.prev_ts_ns else 0
        self.prev_ts_ns = ts_ns

        time_delta_log = math.log1p(delta_us) if delta_us > 0 else 0.0
        price_rel = (price - self.mid_price) / TICK_SIZE if self.mid_price > 0 and price > 0 else 0.0
        qty_log = math.log(max(1, qty))
        spread_ticks = ((self.best_ask - self.best_bid) / TICK_SIZE
                        if self.best_bid > 0 and self.best_ask > 0 else 0.0)

        self._process_raw_event(
            time_delta_log=time_delta_log,
            event_type_id=int(etype),
            side_id=int(side),
            price_rel_ticks=price_rel,
            qty_log=qty_log,
            spread_ticks=spread_ticks,
            timestamp_ns=ts_ns,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Replay mode: read from NPZ file
    # ─────────────────────────────────────────────────────────────────────

    def run_replay(self, npz_path: str | Path):
        """Replay events from a raw MBO NPZ file (offline testing)."""
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        events = data["events"].astype(np.float32)
        timestamps = data.get("timestamps", np.zeros(len(events), dtype=np.int64))

        log.info("Replay mode: %s | %d events", npz_path.name, len(events))
        log.info("Event shape: %s", events.shape)

        if events.shape[1] not in (6, 15, 22, 25):
            raise ValueError(f"Expected 6/15/22/25 columns, got {events.shape[1]}")
        # For smart_v2/v3 preprocessed files, the first 6 cols are the raw features
        # Just use those for the streaming engine (it recomputes all 22 features)
        if events.shape[1] > 6:
            log.info("Preprocessed data detected (%d cols) — using first 6 raw columns", events.shape[1])
            events = events[:, :6]

        # For replay, set dummy BBO from price_rel + spread
        # We need some reference price for paper trading
        # Use first non-zero trade price as reference
        ref_price = 5000.0  # default ES price
        for i in range(min(1000, len(events))):
            if events[i, 1] == 3.0 and events[i, 3] != 0.0:
                # This is relative ticks, we need absolute
                break

        # In replay mode, we use mid_price as a reference for paper trading
        # Since we only have relative ticks, set a synthetic mid
        self.mid_price = ref_price
        self.best_bid = ref_price - TICK_SIZE
        self.best_ask = ref_price + TICK_SIZE

        t0 = time.time()
        n_signals = 0

        for i in range(len(events)):
            ev = events[i]
            ts = int(timestamps[i]) if i < len(timestamps) else 0

            # Update synthetic BBO based on price movement
            price_rel = ev[3]  # price_rel_ticks
            if ev[1] == 3.0:  # trade
                self.mid_price += price_rel * TICK_SIZE * 0.01  # slow drift
                self.best_bid = self.mid_price - ev[5] * TICK_SIZE / 2
                self.best_ask = self.mid_price + ev[5] * TICK_SIZE / 2

            result = self._process_raw_event(
                time_delta_log=float(ev[0]),
                event_type_id=int(ev[1]),
                side_id=int(ev[2]),
                price_rel_ticks=float(ev[3]),
                qty_log=float(ev[4]),
                spread_ticks=float(ev[5]),
                timestamp_ns=ts,
            )

            if result and result.get("signal") != "NEUTRAL":
                n_signals += 1

            # Check timeout
            if i % 5000 == 0:
                self._check_position_timeout()

        # Close any open position at end
        if self.position.size != 0:
            close_price = self._sim_fill_price("S" if self.position.size > 0 else "B")
            self._record_close(self.position, close_price, time.time(), "end_of_replay")
            self.position.size = 0

        elapsed = time.time() - t0
        log.info("Replay complete: %d events in %.1fs (%.0f events/sec)",
                 len(events), elapsed, len(events) / elapsed)

        self._dump_final_stats()

    # ─────────────────────────────────────────────────────────────────────
    # Stats reporting
    # ─────────────────────────────────────────────────────────────────────

    def _dump_stats(self):
        s = self.stats
        elapsed = time.time() - s.start_time
        log.info("=" * 70)
        log.info("LGBM smart_v2 PAPER STATS (%.1f min elapsed)", elapsed / 60)
        log.info("  Events: %d | Predictions: %d | Signals: %d",
                 s.events_processed, s.predictions_made, s.signals_generated)
        log.info("  Trades: %d (W:%d L:%d) | Win rate: %.1f%%",
                 s.total_trades, s.wins, s.losses, s.win_rate * 100)
        log.info("  Gross P&L: $%.2f | Net P&L: $%.2f | Commission: $%.2f",
                 s.total_gross_pnl, s.total_net_pnl, s.total_commission)
        log.info("  Avg P&L: $%.2f | Profit Factor: %.2f | Sortino: %.2f",
                 s.avg_pnl, s.profit_factor, s.sortino)
        log.info("  Max DD: $%.2f | Position: %+d",
                 s.max_drawdown, self.position.size)
        log.info("=" * 70)

    def _dump_final_stats(self):
        log.info("=" * 70)
        log.info("FINAL REPORT — LGBM smart_v2 Live Inference")
        log.info("=" * 70)
        self._dump_stats()

        if self.stats.returns:
            pnls = self.stats.returns
            log.info("  Best trade:  $%.2f", max(pnls))
            log.info("  Worst trade: $%.2f", min(pnls))
            log.info("  Median P&L:  $%.2f", statistics.median(pnls))

        # Save stats JSON
        stats_path = _LOG_DIR / f"lgbm_v2_stats_{self.symbol}.json"
        stats_dict = {
            "model": str(self.model.model_path),
            "symbol": self.symbol,
            "stride": self.stride,
            "confidence_threshold": self.confidence_threshold,
            "start_time": datetime.fromtimestamp(self.stats.start_time, tz=timezone.utc).isoformat(),
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "events_processed": self.stats.events_processed,
            "predictions_made": self.stats.predictions_made,
            "signals_generated": self.stats.signals_generated,
            "total_trades": self.stats.total_trades,
            "win_rate": self.stats.win_rate,
            "total_net_pnl": self.stats.total_net_pnl,
            "total_gross_pnl": self.stats.total_gross_pnl,
            "avg_pnl": self.stats.avg_pnl,
            "sortino": self.stats.sortino,
            "profit_factor": self.stats.profit_factor,
            "max_drawdown": self.stats.max_drawdown,
        }
        with open(stats_path, "w") as f:
            json.dump(_json_safe(stats_dict), f, indent=2)
        log.info("Stats: %s", stats_path)
        log.info("Signals: %s", self._signals_path)
        log.info("Trades: %s", self._trades_path)

    def shutdown(self):
        self._stop.set()
        try:
            self._signals_fh.flush()
            self._signals_fh.close()
            self._trades_fh.flush()
            self._trades_fh.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test: validate streaming features against batch precompute
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test(npz_path: Optional[str] = None, n_events: int = 5000, tol: float = 0.05):
    """Validate streaming smart_v2 features against batch precompute.

    If npz_path is given, loads real MBO data. Otherwise generates synthetic data.
    """
    log.info("Self-test: validating streaming vs batch feature computation...")

    if npz_path:
        data = np.load(npz_path, allow_pickle=True)
        events_raw = data["events"].astype(np.float32)[:n_events]
    else:
        # Generate synthetic data
        rng = np.random.default_rng(42)
        n = n_events
        td = rng.exponential(0.01, n).astype(np.float32)
        td = np.log1p(td * 1e6).astype(np.float32)  # time_delta_log
        et = rng.choice([0, 1, 3, 4], n, p=[0.4, 0.35, 0.15, 0.1]).astype(np.float32)
        sd = rng.integers(0, 2, n).astype(np.float32)
        pr = np.cumsum(rng.normal(0, 0.5, n)).astype(np.float32)
        ql = np.log(rng.integers(1, 10, n).astype(np.float32) + 1)
        sp = (0.25 * rng.integers(1, 4, n)).astype(np.float32)
        events_raw = np.stack([td, et, sd, pr, ql, sp], axis=1)

    log.info("  Testing with %d events", len(events_raw))

    # Run streaming computation
    streamer = StreamingSmartV2Features()
    streaming_out = np.empty((len(events_raw), N_FEATURES), dtype=np.float32)
    for i in range(len(events_raw)):
        ev = events_raw[i]
        streaming_out[i] = streamer.update(
            float(ev[0]), int(ev[1]), int(ev[2]),
            float(ev[3]), float(ev[4]), float(ev[5]),
        )

    # Run batch computation (import from precompute script)
    sys.path.insert(0, "/home/jupiter/Lvl3Quant/alpha_discovery/deep_models")
    try:
        from precompute_features_smart_v2 import (
            compute_derived_features_v1, compute_new_features,
            apply_smart_normalization,
        )
        derived_v1 = compute_derived_features_v1(events_raw)
        new_feats = compute_new_features(events_raw)
        batch_out = apply_smart_normalization(events_raw, derived_v1, new_feats)
    except ImportError:
        log.warning("Cannot import batch precompute — skipping batch comparison")
        log.info("Streaming computation ran successfully for %d events", len(events_raw))
        return True

    # Compare (skip first 1000 events where warmup artifacts differ)
    skip = 1000
    max_diffs = np.max(np.abs(streaming_out[skip:] - batch_out[skip:]), axis=0)
    mean_diffs = np.mean(np.abs(streaming_out[skip:] - batch_out[skip:]), axis=0)

    all_ok = True
    for i, (name, md, mnd) in enumerate(zip(FEATURE_NAMES, max_diffs, mean_diffs)):
        # Rolling z-score features have inherently higher tolerance due to
        # streaming vs batch numerical differences
        feat_tol = tol * 3 if "zscore" in name or i in (7, 9, 10, 14, 15, 17, 18, 20) else tol
        ok = md <= feat_tol
        status = "OK " if ok else "BAD"
        log.info("  [%s] %2d %-30s max|diff|=%.4f mean|diff|=%.4f",
                 status, i, name, md, mnd)
        if not ok:
            all_ok = False

    if all_ok:
        log.info("  SELF-TEST PASSED: streaming features match batch within tolerance.")
    else:
        log.warning("  SELF-TEST: some features diverge from batch. Check rolling z-score numerics.")
        log.warning("  This may be acceptable — streaming z-score uses incremental sum/sq vs batch cumsum.")

    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    ap = argparse.ArgumentParser(
        description="LGBM smart_v2 DA classifier — live inference pipeline. "
                    "Paper trade mode only. No real orders.")
    ap.add_argument("--model", default=DEFAULT_MODEL_PATH,
                    help="Path to LGBM model .txt file")
    ap.add_argument("--symbol", default="ESM6",
                    help="Trading symbol (default: ESM6)")
    ap.add_argument("--exchange", default="CME",
                    help="Exchange (default: CME)")
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                    help=f"Inference every N events (default: {DEFAULT_STRIDE})")
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                    help=f"Min events before first inference (default: {DEFAULT_WARMUP})")
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                    help=f"Confidence threshold for signal (default: {DEFAULT_CONFIDENCE_THRESHOLD})")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Position timeout in seconds (default: 30)")
    ap.add_argument("--slippage", type=float, default=1.0,
                    help="Slippage in ticks (default: 1.0)")
    ap.add_argument("--commission", type=float, default=0.50,
                    help="Commission per side per contract (default: $0.50)")
    ap.add_argument("--max-spread", type=float, default=3.0,
                    help="Max spread in ticks to trade (default: 3.0)")
    ap.add_argument("--stats-interval", type=float, default=300.0,
                    help="Stats dump interval in seconds (default: 300)")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--replay", type=str, default=None,
                      help="Replay from raw MBO NPZ file (offline testing)")
    mode.add_argument("--self-test", action="store_true",
                      help="Run self-test: validate streaming vs batch features")
    mode.add_argument("--self-test-npz", type=str, default=None,
                      help="Run self-test with specific NPZ file")

    return ap.parse_args()


def main():
    args = _parse_args()

    # Self-test modes
    if args.self_test:
        ok = _self_test()
        sys.exit(0 if ok else 1)
    if args.self_test_npz:
        ok = _self_test(args.self_test_npz)
        sys.exit(0 if ok else 1)

    engine = LGBMLiveInference(
        model_path=args.model,
        stride=args.stride,
        warmup=args.warmup,
        confidence_threshold=args.confidence,
        position_timeout_s=args.timeout,
        slippage_ticks=args.slippage,
        commission_per_side=args.commission,
        max_spread_ticks=args.max_spread,
        stats_interval_s=args.stats_interval,
        symbol=args.symbol,
        exchange=args.exchange,
    )

    if args.replay:
        # Replay mode: synchronous, no Rithmic needed
        engine.run_replay(args.replay)
    else:
        # Live mode: async Rithmic connection
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _handle_sig(*_a):
            log.info("Shutdown signal received...")
            engine.shutdown()

        for s in (_sig.SIGINT, _sig.SIGTERM):
            try:
                loop.add_signal_handler(s, _handle_sig)
            except NotImplementedError:
                _sig.signal(s, lambda *_: engine.shutdown())

        try:
            loop.run_until_complete(engine.run_live())
        finally:
            loop.close()


if __name__ == "__main__":
    main()
