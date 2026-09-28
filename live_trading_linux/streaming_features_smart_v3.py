#!/usr/bin/env python3
"""
streaming_features_smart_v3.py — Online feature computation for Mamba v7 live inference.

Matches precompute_features_smart_v3.py EXACTLY — 25 features computed incrementally
from raw MBO events.

CRITICAL CORRECTNESS NOTE:
    The precompute uses causal_rolling_sum(signal, W) which returns, for event i:
        sum(signal[max(0, i-W) : i])  — sum of PREVIOUS W values, NOT including i.
    Similarly, causal_rolling_zscore normalizes signal[i] against stats computed
    from signal[max(0, i-W) : i] — the previous W values, NOT including i.

    The streaming engine must match this exactly: read sum/stats BEFORE pushing
    the current event's signal, then push for future events.

The 25 features (matching smart_v3 precompute):
  RAW (0-5):
    0: time_delta_log     -> clamp [0,8], /4.0
    1: event_type_id      -> /4.0
    2: side_id            -> remap to -1/+1
    3: price_rel_ticks    -> clip [-50,50], /25.0
    4: qty_log            -> (x - 0.693) / 3.0
    5: spread_ticks       -> clip [0,20], /5.0
  DERIVED V1 (6-14):
    6:  cancel_side_asym_50     -> /25.0
    7:  rolling_ofi_500         -> rolling z-score (W=10000)
    8:  event_density_20        -> clamp [0,4], /2.0
    9:  price_mom_10            -> rolling z-score (W=5000)
    10: qty_price_mom_50        -> rolling z-score (W=10000)
    11: price_sign_momentum_200 -> /100.0
    12: event_type_entropy_200  -> /1.609
    13: fill_add_restoration_100 -> as-is [0,1]
    14: spread_velocity_50      -> rolling z-score (W=5000)
  V2 (15-21):
    15: queue_replenishment     -> rolling z-score (W=10000)
    16: mom_divergence          -> *5 + clip [-5,5]
    17: ofi_x_spread            -> rolling z-score (W=10000)
    18: vol_weighted_pmom       -> rolling z-score (W=10000)
    19: buy_sell_intensity_ratio -> as-is [-1,1]
    20: realized_volatility     -> rolling z-score (W=10000)
    21: sweep_intensity         -> clip [-50,50], /10.0
  V3 (22-24):
    22: ofi_short_100           -> rolling z-score (W=5000)
    23: ofi_long_2000           -> rolling z-score (W=10000)
    24: ofi_acceleration        -> rolling z-score (W=5000)

Event type codes: 0=add, 1=cancel, 2=modify, 3=trade, 4=fill
Side codes:       0=bid, 1=ask
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np


# ============================================================
# Constants
# ============================================================
EVENT_TYPE_ADD    = 0
EVENT_TYPE_CANCEL = 1
EVENT_TYPE_MODIFY = 2
EVENT_TYPE_TRADE  = 3
EVENT_TYPE_FILL   = 4
N_EVENT_TYPES     = 5

SIDE_BID = 0
SIDE_ASK = 1

ROLLING_ZSCORE_LOOKBACK_LONG  = 10000
ROLLING_ZSCORE_LOOKBACK_SHORT = 5000
EWMA_ALPHA = 0.01
SWEEP_MIN_TRADES = 3
SWEEP_MAX_GAP_LOG = 0.5

N_FEATURES = 25

FEATURE_NAMES = [
    'time_delta_log', 'event_type_id', 'side_id',
    'price_rel_ticks', 'qty_log', 'spread_ticks',
    'cancel_side_asym_50', 'rolling_ofi_500', 'event_density_20',
    'price_mom_10', 'qty_price_mom_50', 'price_sign_momentum_200',
    'event_type_entropy_200', 'fill_add_restoration_100', 'spread_velocity_50',
    'queue_replenishment', 'mom_divergence', 'ofi_x_spread',
    'vol_weighted_pmom', 'buy_sell_intensity_ratio', 'realized_volatility',
    'sweep_intensity',
    'ofi_short_100', 'ofi_long_2000', 'ofi_acceleration',
]
assert len(FEATURE_NAMES) == N_FEATURES


# ============================================================
# Streaming helpers
# ============================================================

class RollingSum:
    """Fixed-window rolling sum. O(1) per update.

    IMPORTANT: Use sum_then_push() to match causal_rolling_sum behavior:
    returns the sum of PREVIOUS W values, THEN adds the current value.
    """
    __slots__ = ("window", "buf", "s")

    def __init__(self, window: int) -> None:
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.s: float = 0.0

    def push(self, x: float) -> None:
        if len(self.buf) == self.window:
            self.s -= self.buf[0]
        self.buf.append(x)
        self.s += x

    def sum(self) -> float:
        return self.s

    def sum_then_push(self, x: float) -> float:
        """Return current sum (causal, NOT including x), then push x.
        Matches precompute's causal_rolling_sum behavior."""
        causal_sum = self.s
        self.push(x)
        return causal_sum

    def count(self) -> int:
        return len(self.buf)


class RollingMean:
    """Fixed-window rolling mean. Divides by actual count.

    Use mean_then_push() to match causal_rolling_mean behavior.
    """
    __slots__ = ("_rs",)

    def __init__(self, window: int) -> None:
        self._rs = RollingSum(window)

    def push(self, x: float) -> None:
        self._rs.push(x)

    def mean(self) -> float:
        c = self._rs.count()
        if c == 0:
            return 0.0
        return self._rs.s / c

    def mean_then_push(self, x: float) -> float:
        """Return mean of previous W values (causal), then push x."""
        c = self._rs.count()
        if c == 0:
            m = 0.0
        else:
            m = self._rs.s / c
        self._rs.push(x)
        return m


class RollingZScore:
    """Causal rolling z-score matching precompute_features_smart_v3.py.

    push_and_score(x) computes z = (x - mean_prev) / std_prev, then stores x.
    This matches the precompute behavior where stats are from previous W values.
    """
    __slots__ = ("window", "buf", "s", "s_sq")

    def __init__(self, window: int) -> None:
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.s: float = 0.0
        self.s_sq: float = 0.0

    def push_and_score(self, x: float) -> float:
        """Compute z-score of x against previous buffer, THEN push x."""
        # 1. Score against existing buffer (BEFORE adding x)
        n = len(self.buf)
        if n < 2:
            z = 0.0
        else:
            mean = self.s / n
            var = max(self.s_sq / n - mean * mean, 0.0)
            std = max(math.sqrt(var), 1e-8)
            z = (x - mean) / std
            z = max(-5.0, min(5.0, z))

        # 2. Push x into buffer for future computations
        if len(self.buf) == self.window:
            old = self.buf[0]
            self.s -= old
            self.s_sq -= old * old
        self.buf.append(x)
        self.s += x
        self.s_sq += x * x

        return z


class EWMATracker:
    """Exponentially weighted moving average."""
    __slots__ = ("alpha", "value", "initialized")

    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: float = 0.0
        self.initialized: bool = False

    def push(self, x: float) -> float:
        if not self.initialized:
            self.value = x
            self.initialized = True
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class EventTypeEntropyTracker:
    """Rolling entropy of event types over a window.
    Causal: returns entropy of PREVIOUS W events, then pushes current.
    """
    __slots__ = ("window", "buf", "type_counts", "n_types")

    def __init__(self, window: int, n_types: int = N_EVENT_TYPES) -> None:
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.type_counts = [0] * n_types
        self.n_types = n_types

    def push_and_entropy(self, event_type: int) -> float:
        """Return entropy of previous W events (causal), then push current."""
        # 1. Compute entropy from existing buffer (BEFORE adding current)
        total = len(self.buf)
        if total == 0:
            entropy = 0.0
        else:
            entropy = 0.0
            for c in self.type_counts:
                if c > 0:
                    p = c / total
                    entropy -= p * math.log(p)

        # 2. Push current event type
        if len(self.buf) == self.window:
            old = self.buf[0]
            if 0 <= old < self.n_types:
                self.type_counts[old] -= 1
        self.buf.append(event_type)
        if 0 <= event_type < self.n_types:
            self.type_counts[event_type] += 1

        return entropy


class FillAddRestorationTracker:
    """Rolling fill-add restoration ratio.
    Causal: returns ratio from PREVIOUS W events, then pushes current.
    """
    __slots__ = ("window", "fill_add_buf", "fill_trade_buf",
                 "fill_add_sum", "fill_trade_sum",
                 "prev_was_fill_trade", "prev_side")

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.fill_add_buf: deque = deque(maxlen=window)
        self.fill_trade_buf: deque = deque(maxlen=window)
        self.fill_add_sum: float = 0.0
        self.fill_trade_sum: float = 0.0
        self.prev_was_fill_trade: bool = False
        self.prev_side: int = -1

    def push_and_ratio(self, event_type: int, side: int) -> float:
        """Return ratio of previous W events (causal), then push current."""
        is_fill_trade = (event_type == EVENT_TYPE_TRADE or event_type == EVENT_TYPE_FILL)
        is_add = (event_type == EVENT_TYPE_ADD)

        fill_add = 1.0 if (self.prev_was_fill_trade and is_add and side == self.prev_side) else 0.0
        ft_val = 1.0 if is_fill_trade else 0.0

        # 1. Read ratio BEFORE pushing
        denom = max(self.fill_trade_sum, 1.0)
        ratio = self.fill_add_sum / denom

        # 2. Push current signals
        if len(self.fill_add_buf) == self.window:
            self.fill_add_sum -= self.fill_add_buf[0]
        if len(self.fill_trade_buf) == self.window:
            self.fill_trade_sum -= self.fill_trade_buf[0]

        self.fill_add_buf.append(fill_add)
        self.fill_trade_buf.append(ft_val)
        self.fill_add_sum += fill_add
        self.fill_trade_sum += ft_val

        # Update prev state
        self.prev_was_fill_trade = is_fill_trade
        self.prev_side = side

        return ratio


class SweepTracker:
    """Track sweep intensity with causal rolling sum."""
    __slots__ = ("window", "buf", "s", "prev_trade_signed", "consec_count")

    def __init__(self, window: int = 200) -> None:
        self.window = window
        self.buf: deque = deque(maxlen=window)
        self.s: float = 0.0
        self.prev_trade_signed: float = 0.0
        self.consec_count: int = 0

    def push_and_sum(self, event_type: int, side: int, time_delta_log: float) -> float:
        """Return causal sum (previous W), then push current sweep signal."""
        is_trade = (event_type == EVENT_TYPE_TRADE or event_type == EVENT_TYPE_FILL)
        sign_side = 1.0 if side == SIDE_ASK else -1.0
        trade_signed = sign_side if is_trade else 0.0

        same_as_prev = (
            trade_signed != 0.0 and
            trade_signed == self.prev_trade_signed and
            time_delta_log < SWEEP_MAX_GAP_LOG
        )

        if same_as_prev:
            self.consec_count += 1
        elif trade_signed != 0.0:
            self.consec_count = 1
        else:
            self.consec_count = 0

        is_sweep = 1.0 if self.consec_count >= SWEEP_MIN_TRADES else 0.0
        sweep_signed = is_sweep * trade_signed

        # 1. Read causal sum BEFORE pushing
        causal_sum = self.s

        # 2. Push current
        if len(self.buf) == self.window:
            self.s -= self.buf[0]
        self.buf.append(sweep_signed)
        self.s += sweep_signed

        self.prev_trade_signed = trade_signed
        return causal_sum


# ============================================================
# Main streaming engine
# ============================================================

class StreamingFeaturesSmartV3:
    """Incrementally compute the 25 smart_v3 features for Mamba v7 inference.

    Call `update(time_delta_log, event_type_id, side_id,
                 price_rel_ticks, qty_log, spread_ticks)`
    once per MBO event. Returns np.ndarray shape (25,) float32.

    Warm-up: first ~5000-10000 events produce less reliable z-scores.
    """

    MIN_WARMUP = 5000

    def __init__(self) -> None:
        self._n_events: int = 0

        # === Derived V1 rolling trackers ===
        self._cancel_side_asym = RollingSum(50)
        self._ofi_500 = RollingSum(500)
        self._ofi_500_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)
        self._event_density = RollingMean(20)
        self._price_mom_10 = RollingSum(10)
        self._price_mom_10_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_SHORT)
        self._qty_price_mom = RollingSum(50)
        self._qty_price_mom_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)
        self._price_sign_mom = RollingSum(200)
        self._entropy = EventTypeEntropyTracker(200)
        self._fill_add = FillAddRestorationTracker(100)
        self._spread_vel_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_SHORT)
        self._spread_diff_mean = RollingMean(50)
        self._prev_spread: float = 0.0
        self._has_prev_spread: bool = False

        # === V2 rolling trackers ===
        self._add_qty_100 = RollingSum(100)
        self._cancel_qty_100 = RollingSum(100)
        self._queue_replen_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)
        self._pmom_short_20 = RollingSum(20)
        self._pmom_long_200 = RollingSum(200)
        self._ofi_100 = RollingSum(100)
        self._ofi_x_spread_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)
        self._vol_pmom_50 = RollingSum(50)
        self._vol_pmom_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)
        self._bid_ewma = EWMATracker(EWMA_ALPHA)
        self._ask_ewma = EWMATracker(EWMA_ALPHA)
        self._price_change_sq_200 = RollingMean(200)
        self._rvol_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)
        self._prev_price: float = 0.0
        self._has_prev_price: bool = False
        self._sweep = SweepTracker(200)

        # === V3 rolling trackers ===
        self._ofi_short_100 = RollingSum(100)
        self._ofi_short_100_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_SHORT)
        self._ofi_long_2000 = RollingSum(2000)
        self._ofi_long_2000_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_LONG)
        self._ofi_accel_100 = RollingSum(100)
        self._ofi_accel_500 = RollingSum(500)
        self._ofi_accel_zscore = RollingZScore(ROLLING_ZSCORE_LOOKBACK_SHORT)

    def is_warm(self) -> bool:
        return self._n_events >= self.MIN_WARMUP

    @property
    def n_events(self) -> int:
        return self._n_events

    def update(self, time_delta_log: float, event_type_id: int, side_id: int,
               price_rel_ticks: float, qty_log: float, spread_ticks: float) -> np.ndarray:
        """Process one MBO event and return 25-feature vector (float32).

        All rolling sums/means/z-scores use CAUSAL semantics:
        at event i, stats are computed from previous W events (not including i).
        """
        self._n_events += 1
        out = np.zeros(N_FEATURES, dtype=np.float32)

        # Precompute common signals
        is_cancel = 1.0 if event_type_id == EVENT_TYPE_CANCEL else 0.0
        is_add = 1.0 if event_type_id == EVENT_TYPE_ADD else 0.0
        is_trade = 1.0 if (event_type_id == EVENT_TYPE_TRADE or
                           event_type_id == EVENT_TYPE_FILL) else 0.0
        is_ask = 1.0 if side_id == SIDE_ASK else 0.0
        is_bid = 1.0 if side_id == SIDE_BID else 0.0
        sign_side = is_ask - is_bid
        price_sign = 1.0 if price_rel_ticks > 0 else (-1.0 if price_rel_ticks < 0 else 0.0)
        ofi_signal = qty_log * sign_side

        # ================================================================
        # RAW FEATURES (0-5) — smart normalization (no rolling state)
        # ================================================================
        out[0] = min(max(time_delta_log, 0.0), 8.0) / 4.0
        out[1] = event_type_id / 4.0
        out[2] = side_id * 2.0 - 1.0
        out[3] = min(max(price_rel_ticks, -50.0), 50.0) / 25.0
        out[4] = (qty_log - 0.693) / 3.0
        out[5] = min(max(spread_ticks, 0.0), 20.0) / 5.0

        # ================================================================
        # DERIVED V1 FEATURES (6-14) — all use causal (sum before push)
        # ================================================================

        # 6: cancel_side_asym_50 -> /25.0
        cancel_signal = is_cancel * sign_side
        raw_cancel_asym = self._cancel_side_asym.sum_then_push(cancel_signal)
        out[6] = min(max(raw_cancel_asym, -50.0), 50.0) / 25.0

        # 7: rolling_ofi_500 -> rolling z-score (W=10000)
        raw_ofi_500 = self._ofi_500.sum_then_push(ofi_signal)
        out[7] = self._ofi_500_zscore.push_and_score(raw_ofi_500)

        # 8: event_density_20 -> clamp [0,4], /2.0
        raw_density = self._event_density.mean_then_push(time_delta_log)
        out[8] = min(max(raw_density, 0.0), 4.0) / 2.0

        # 9: price_mom_10 -> rolling z-score (W=5000)
        raw_pmom10 = self._price_mom_10.sum_then_push(price_rel_ticks)
        out[9] = self._price_mom_10_zscore.push_and_score(raw_pmom10)

        # 10: qty_price_mom_50 -> rolling z-score (W=10000)
        qty_price = qty_log * price_rel_ticks
        raw_qpmom = self._qty_price_mom.sum_then_push(qty_price)
        out[10] = self._qty_price_mom_zscore.push_and_score(raw_qpmom)

        # 11: price_sign_momentum_200 -> /100.0
        raw_psm = self._price_sign_mom.sum_then_push(price_sign)
        out[11] = raw_psm / 100.0

        # 12: event_type_entropy_200 -> /1.609
        entropy = self._entropy.push_and_entropy(event_type_id)
        out[12] = entropy / 1.609

        # 13: fill_add_restoration_100 -> as-is [0,1]
        out[13] = self._fill_add.push_and_ratio(event_type_id, side_id)

        # 14: spread_velocity_50 -> rolling z-score (W=5000)
        if self._has_prev_spread:
            spread_diff = spread_ticks - self._prev_spread
        else:
            spread_diff = 0.0
        self._prev_spread = spread_ticks
        self._has_prev_spread = True
        raw_spread_vel = self._spread_diff_mean.mean_then_push(spread_diff)
        out[14] = self._spread_vel_zscore.push_and_score(raw_spread_vel)

        # ================================================================
        # V2 FEATURES (15-21)
        # ================================================================

        # 15: queue_replenishment -> rolling z-score (W=10000)
        add_rate = self._add_qty_100.sum_then_push(is_add * qty_log)
        cancel_rate = max(self._cancel_qty_100.sum_then_push(is_cancel * qty_log), 0.1)
        raw_replen = add_rate / cancel_rate
        out[15] = self._queue_replen_zscore.push_and_score(raw_replen)

        # 16: mom_divergence -> *5 + clip [-5,5]
        pmom_short = self._pmom_short_20.sum_then_push(price_sign)
        pmom_long = self._pmom_long_200.sum_then_push(price_sign)
        raw_divergence = pmom_short / 20.0 - pmom_long / 200.0
        out[16] = min(max(raw_divergence * 5.0, -5.0), 5.0)

        # 17: ofi_x_spread -> rolling z-score (W=10000)
        ofi_100_val = self._ofi_100.sum_then_push(ofi_signal)
        raw_ofi_x_spread = ofi_100_val * spread_ticks
        out[17] = self._ofi_x_spread_zscore.push_and_score(raw_ofi_x_spread)

        # 18: vol_weighted_pmom -> rolling z-score (W=10000)
        raw_vol_pmom = self._vol_pmom_50.sum_then_push(qty_log * price_rel_ticks)
        out[18] = self._vol_pmom_zscore.push_and_score(raw_vol_pmom)

        # 19: buy_sell_intensity_ratio -> as-is [-1,1]
        # EWMA is inherently causal — each push returns the EWMA after seeing x,
        # which matches precompute's ewma[0] = signal[0], ewma[i] = alpha*x + (1-alpha)*prev
        bid_ewma = self._bid_ewma.push(is_bid)
        ask_ewma = self._ask_ewma.push(is_ask)
        total_ewma = bid_ewma + ask_ewma
        if total_ewma < 1e-8:
            total_ewma = 1e-8
        out[19] = (bid_ewma / total_ewma - 0.5) * 2.0

        # 20: realized_volatility -> rolling z-score (W=10000)
        if self._has_prev_price:
            price_change = price_rel_ticks - self._prev_price
        else:
            price_change = 0.0
        self._prev_price = price_rel_ticks
        self._has_prev_price = True
        # price_change_sq[0] = 0 (precompute sets price_change[0] = 0)
        # Then causal_rolling_mean of price_change_sq over 200
        # Then sqrt -> rvol
        # Then causal_rolling_zscore of rvol over 10000
        raw_rvol_mean = self._price_change_sq_200.mean_then_push(price_change ** 2)
        rvol = math.sqrt(max(raw_rvol_mean, 0.0))
        out[20] = self._rvol_zscore.push_and_score(rvol)

        # 21: sweep_intensity -> clip [-50,50], /10.0
        raw_sweep = self._sweep.push_and_sum(event_type_id, side_id, time_delta_log)
        out[21] = min(max(raw_sweep, -50.0), 50.0) / 10.0

        # ================================================================
        # V3 FEATURES (22-24)
        # ================================================================

        # 22: ofi_short_100 -> rolling z-score (W=5000)
        raw_ofi_short = self._ofi_short_100.sum_then_push(ofi_signal)
        out[22] = self._ofi_short_100_zscore.push_and_score(raw_ofi_short)

        # 23: ofi_long_2000 -> rolling z-score (W=10000)
        raw_ofi_long = self._ofi_long_2000.sum_then_push(ofi_signal)
        out[23] = self._ofi_long_2000_zscore.push_and_score(raw_ofi_long)

        # 24: ofi_acceleration -> rolling z-score (W=5000)
        # ofi_accel = ofi_100_rate - ofi_500_rate (each is sum/window)
        ofi_accel_100 = self._ofi_accel_100.sum_then_push(ofi_signal)
        ofi_accel_500 = self._ofi_accel_500.sum_then_push(ofi_signal)
        raw_accel = ofi_accel_100 / 100.0 - ofi_accel_500 / 500.0
        out[24] = self._ofi_accel_zscore.push_and_score(raw_accel)

        return out

    def reset(self) -> None:
        """Reset all state. Use when starting a new trading day."""
        self.__init__()


# ============================================================
# Validation: compare streaming vs batch precompute
# ============================================================
def validate_against_batch(npz_path: str, max_events: int = 20000,
                           verbose: bool = True) -> dict:
    """Load a precomputed smart_v3 NPZ and compare streaming output."""
    from pathlib import Path

    data = np.load(npz_path)
    precomputed = data['events']  # (N, 25) float32
    N = min(len(precomputed), max_events)

    # Find raw events file
    raw_dir = Path(npz_path).parent.parent / 'mbo_events'
    raw_file = raw_dir / Path(npz_path).name
    if not raw_file.exists():
        for alt_dir in [
            Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events'),
            Path('/home/nick/Lvl3Quant/data/processed/mbo_events'),
        ]:
            alt_file = alt_dir / Path(npz_path).name
            if alt_file.exists():
                raw_file = alt_file
                break

    if not raw_file.exists():
        raise FileNotFoundError(f"Cannot find raw events file for {npz_path}")

    raw_data = np.load(raw_file)
    raw_events = raw_data['events']

    engine = StreamingFeaturesSmartV3()
    errors = np.zeros((N, N_FEATURES), dtype=np.float64)

    for i in range(N):
        ev = raw_events[i]
        streaming_feats = engine.update(
            time_delta_log=float(ev[0]),
            event_type_id=int(ev[1]),
            side_id=int(ev[2]),
            price_rel_ticks=float(ev[3]),
            qty_log=float(ev[4]),
            spread_ticks=float(ev[5]),
        )
        errors[i] = np.abs(streaming_feats - precomputed[i])

    warmup = min(5000, N // 2)
    errors_post_warmup = errors[warmup:]

    max_abs = float(np.max(errors_post_warmup))
    mean_abs = float(np.mean(errors_post_warmup))
    per_feature_max = [float(np.max(errors_post_warmup[:, j])) for j in range(N_FEATURES)]

    if verbose:
        print(f"\nValidation: {N} events from {Path(npz_path).name}")
        print(f"  Post-warmup ({warmup}+ events):")
        print(f"  Max absolute error:  {max_abs:.6f}")
        print(f"  Mean absolute error: {mean_abs:.6f}")
        print(f"\n  Per-feature max error (post-warmup):")
        for j, name in enumerate(FEATURE_NAMES):
            err = per_feature_max[j]
            status = "OK" if err < 0.01 else ("WARN" if err < 0.1 else "FAIL")
            print(f"    {status:4s} {j:2d} {name:30s}  max_err={err:.6f}")

    return {
        'max_abs_error': max_abs,
        'mean_abs_error': mean_abs,
        'per_feature_max_error': per_feature_max,
        'n_events': N,
        'warmup': warmup,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        result = validate_against_batch(sys.argv[1])
        print(f"\nOverall: max_err={result['max_abs_error']:.6f}, "
              f"mean_err={result['mean_abs_error']:.6f}")
    else:
        engine = StreamingFeaturesSmartV3()
        print(f"StreamingFeaturesSmartV3 -- {N_FEATURES} features")
        import random
        random.seed(42)
        for i in range(100):
            feats = engine.update(
                time_delta_log=random.uniform(0, 5),
                event_type_id=random.randint(0, 4),
                side_id=random.randint(0, 1),
                price_rel_ticks=random.gauss(0, 5),
                qty_log=random.uniform(0, 3),
                spread_ticks=random.uniform(0.25, 2.0),
            )
            if i % 20 == 0:
                print(f"  Event {i}: warm={engine.is_warm()}, "
                      f"feat[0:3]={feats[:3].tolist()}")
        print(f"\n  After {engine.n_events} events: warm={engine.is_warm()}")
        print("  Smoke test passed")
