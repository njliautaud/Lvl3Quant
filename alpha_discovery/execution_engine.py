"""
Smart Execution Engine -- ES Futures MBO Direction Trading
==========================================================

Converts a raw LightGBM directional signal into actionable orders by
layering four filters on top of signal strength:

    1. VolRegimeGate   -- only trade on "active" intraday regimes
    2. PerformanceConditionAnalyzer -- scale size by historical hit-rate
    3. ImbalanceFilter -- align with order-book pressure or skip
    4. ExecutionRouter -- map signal strength -> order type
    5. PassiveToAggressiveEscalator -- manage live order lifecycle

Phase-1 cost context (1 contract ES):
    - TICK_SIZE        = $0.25
    - TICK_VALUE       = $12.50 / tick
    - COMMISSION_RT    = $3.00  (AMP + Rithmic + CME)
    - TOTAL_COST       = $15.50 / round-trip = 1.24 ticks
    - cost_ratio (IC x scale / cost):
        ret_3s  : 0.26  -> passive limit only
        ret_10s : 0.51  -> midpoint / aggressive limit
        ret_30s : 0.91  -> aggressive limit / market
        ret_1m  : 1.31  -> market order viable

All state is updated with past data only (strictly causal).

Usage (backtesting):
    from alpha_discovery.execution_engine import SmartExecutionEngine

    engine = SmartExecutionEngine()
    for i, bar in enumerate(bars):
        decision = engine.process(
            signal_strength=abs(pred[i]),
            signal_direction=int(np.sign(pred[i])),
            horizon='ret_30s',
            bar=bar,           # dict with book fields
            bar_index=i,
        )
        if decision.action == 'enter':
            sim.open_trade(decision)

Usage (live):
    engine = SmartExecutionEngine(mode='live')
    decision = engine.process(...)
    if decision.action == 'enter':
        broker.submit(decision.order_type, decision.size, ...)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

# ============================================================================
# LOGGING (ASCII-only for Windows cp1252 compatibility)
# ============================================================================
log = logging.getLogger("execution_engine")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE          = 0.25           # ES minimum price increment
TICK_VALUE         = 12.50          # Dollar value per tick
ES_POINT_VALUE     = 50.0           # Dollar value per ES point (4 ticks)
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)           # Round-trip commission ($)
COMMISSION_TICKS   = COMMISSION_RT / TICK_VALUE   # 0.24 ticks
HALF_TICK          = TICK_SIZE / 2  # 0.125 -- mid-to-best-bid/ask distance
BARS_PER_SEC       = 10             # 100ms bars

# Total round-trip cost in ticks (half-spread crossing in + half-spread crossing out + commission)
# Crossing market: 0.5 tick (half-spread) + 0.5 tick (half-spread) + 0.24 tick commission = 1.24
# Using limit entry saves 0.5 tick on entry side
TOTAL_COST_TICKS_MARKET = 0.5 + 0.5 + COMMISSION_TICKS   # 1.24 ticks ($15.50)
TOTAL_COST_TICKS_LIMIT  = 0.0 + 0.5 + COMMISSION_TICKS   # 0.74 ticks ($9.25)

# Signal strength thresholds (expressed as multiples of total round-trip cost in ticks)
# Phase 1 signal translates to: expected_ticks = IC * std(mid_return) / tick_size
# Conservative scaling: strong=2x cost, medium=1x cost
STRONG_SIGNAL_MULTIPLIER  = 2.0   # expected move > 2x total cost -> market
MEDIUM_SIGNAL_MULTIPLIER  = 1.0   # expected move > 1x total cost -> aggressive limit / midpoint
WEAK_SIGNAL_MULTIPLIER    = 0.5   # expected move > 0.5x total cost -> passive limit only
# Below 0.5x cost -> skip

# Horizon-to-expected-move mapping (from Phase 1 IC x typical ES vol in ticks)
# These are conservative estimates -- calibrate from live data over time
HORIZON_PARAMS: Dict[str, Dict] = {
    'ret_3s':  {'cost_ratio': 0.26, 'expected_ticks': 0.32, 'bars': 30},
    'ret_10s': {'cost_ratio': 0.51, 'expected_ticks': 0.63, 'bars': 100},
    'ret_30s': {'cost_ratio': 0.91, 'expected_ticks': 1.13, 'bars': 300},
    'ret_1m':  {'cost_ratio': 1.31, 'expected_ticks': 1.62, 'bars': 600},
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BarData:
    """Snapshot of one 100ms bar from the MBO feed.

    All fields match the feature names in mbo_features.py so the engine
    can be wired directly to the feature pipeline.
    """
    mid: float = 0.0
    spread: float = TICK_SIZE
    bid_depth: float = 0.0           # total_bid_vol
    ask_depth: float = 0.0           # total_ask_vol
    vol_imbalance: float = 0.0       # (bid-ask)/(bid+ask)
    ofi_5: float = 0.0               # order-flow imbalance, 5-bar window
    ofi_20: float = 0.0
    ofi_50: float = 0.0
    trade_imbalance: float = 0.0     # signed trade flow
    vpin_50: float = 0.5             # toxicity proxy
    hour_norm: float = 0.5           # fraction through RTH [0,1]
    ret_5: float = 0.0               # 5-bar price momentum
    realized_vol_20: float = 0.0     # 20-bar realized vol in bps
    realized_vol_50: float = 0.0
    spread_ticks: float = 1.0


@dataclass
class ExecutionDecision:
    """Output of SmartExecutionEngine.process().

    action: 'enter' | 'skip' | 'hold'
    order_type: 'market' | 'aggressive_limit' | 'midpoint' | 'limit' | None
    size: number of contracts (0 = skip)
    confidence: composite score [0, 1]
    reasoning: human-readable chain of decisions
    escalation_schedule: list of (bars_from_now, order_type) if passive entry
    """
    action: str = 'skip'
    order_type: Optional[str] = None
    size: int = 0
    confidence: float = 0.0
    reasoning: str = ''
    escalation_schedule: List[Tuple[int, str]] = field(default_factory=list)
    signal_strength: float = 0.0
    signal_direction: int = 0
    horizon: str = ''
    expected_ticks: float = 0.0
    regime_active: bool = False
    imbalance_score: float = 0.0
    condition_score: float = 0.0


# ============================================================================
# 1. EXECUTION ROUTER
# ============================================================================

class ExecutionRouter:
    """Map signal strength to order type based on expected profit vs cost.

    The key insight: a stronger signal means we can afford a more aggressive
    (higher-slippage) execution because the expected profit per trade is larger.

    Thresholds:
        strong  (expected > 2x cost) -> market order (guaranteed fill, pay spread)
        medium  (expected 1-2x cost) -> aggressive limit or midpoint
        weak    (expected 0.5-1x cost) -> passive limit only
        noise   (expected < 0.5x cost) -> skip (no edge after costs)

    Args:
        strong_mult: multiplier on total cost ticks for market routing
        medium_mult: multiplier on total cost ticks for midpoint routing

    Usage:
        router = ExecutionRouter()
        order_type = router.route(signal_strength=0.8, horizon='ret_30s')
        # -> 'aggressive_limit'
    """

    def __init__(self,
                 strong_mult: float = STRONG_SIGNAL_MULTIPLIER,
                 medium_mult: float = MEDIUM_SIGNAL_MULTIPLIER,
                 weak_mult: float = WEAK_SIGNAL_MULTIPLIER):
        self.strong_mult = strong_mult
        self.medium_mult = medium_mult
        self.weak_mult   = weak_mult

    def expected_ticks(self, signal_strength: float, horizon: str) -> float:
        """Estimate expected gross move in ticks for this signal.

        signal_strength is the normalized prediction (z-score or raw model output).
        We scale by the horizon's typical expected ticks from Phase 1 calibration.

        For live use, replace `base_ticks` with a rolling estimate from
        PerformanceConditionAnalyzer.

        Args:
            signal_strength: absolute value of normalized prediction (> 0)
            horizon: 'ret_3s' | 'ret_10s' | 'ret_30s' | 'ret_1m'

        Returns:
            Expected gross move in ES ticks
        """
        params = HORIZON_PARAMS.get(horizon, HORIZON_PARAMS['ret_30s'])
        # Scale by signal strength relative to median signal
        # signal_strength is assumed to be in [0, 1] range (percentile rank)
        base_ticks = params['expected_ticks']
        return float(np.clip(signal_strength * base_ticks * 2.0, 0.0, 20.0))

    def route(self,
              signal_strength: float,
              horizon: str,
              override_expected_ticks: Optional[float] = None) -> Tuple[str, float]:
        """Choose order type based on expected profit vs cost.

        Args:
            signal_strength: normalized prediction strength, [0, 1]
            horizon: trading horizon key
            override_expected_ticks: if provided, use this instead of internal estimate

        Returns:
            (order_type, expected_ticks_estimate)
            order_type: 'market' | 'aggressive_limit' | 'midpoint' | 'limit' | 'skip'
        """
        exp_ticks = (override_expected_ticks
                     if override_expected_ticks is not None
                     else self.expected_ticks(signal_strength, horizon))

        cost = TOTAL_COST_TICKS_MARKET

        if exp_ticks >= self.strong_mult * cost:
            return 'market', exp_ticks
        elif exp_ticks >= self.medium_mult * cost:
            # Between 1x and 2x cost: use aggressive limit (1 tick through mid)
            # or midpoint depending on exact ratio
            ratio = exp_ticks / cost
            if ratio >= 1.5:
                return 'aggressive_limit', exp_ticks
            else:
                return 'midpoint', exp_ticks
        elif exp_ticks >= self.weak_mult * cost:
            return 'limit', exp_ticks
        else:
            return 'skip', exp_ticks

    def size_from_confidence(self,
                              confidence: float,
                              max_contracts: int = 2) -> int:
        """Scale position size based on overall confidence score.

        Args:
            confidence: composite score from SmartExecutionEngine [0, 1]
            max_contracts: maximum contracts per trade

        Returns:
            Number of contracts (0 = skip)
        """
        if confidence < 0.3:
            return 0
        elif confidence < 0.6:
            return 1
        else:
            return min(max_contracts, max(1, int(round(confidence * max_contracts))))


# ============================================================================
# 2. IMBALANCE FILTER
# ============================================================================

class ImbalanceFilter:
    """Check whether order book imbalance agrees with the directional signal.

    Uses bid/ask depth imbalance and order-flow imbalance (OFI) from the
    current bar. If the book strongly disagrees with the signal direction,
    reduce confidence or skip entirely.

    Imbalance convention:
        +1.0 = strongly bid-side (bullish pressure)
        -1.0 = strongly ask-side (bearish pressure)
         0.0 = balanced

    Decision logic:
        - signal UP  + imbalance > +threshold -> boost confidence (+boost)
        - signal UP  + imbalance < -threshold -> reduce confidence (skip if severe)
        - signal DOWN + imbalance < -threshold -> boost confidence
        - signal DOWN + imbalance > +threshold -> reduce confidence

    Args:
        skip_threshold: imbalance score beyond which we skip (default 0.6)
        boost_threshold: imbalance score for confidence boost (default 0.3)

    Usage:
        filt = ImbalanceFilter()
        result = filt.evaluate(bar, signal_direction=+1)
        # result: {'score': 0.45, 'agrees': True, 'confidence_adj': +0.15}
    """

    def __init__(self,
                 skip_threshold: float = 0.6,
                 boost_threshold: float = 0.3,
                 ofi_weight: float = 0.4,
                 depth_weight: float = 0.4,
                 trade_imb_weight: float = 0.2):
        self.skip_threshold  = skip_threshold
        self.boost_threshold = boost_threshold
        self.ofi_weight       = ofi_weight
        self.depth_weight     = depth_weight
        self.trade_imb_weight = trade_imb_weight

    def compute_imbalance_score(self, bar: BarData) -> float:
        """Compute composite imbalance score in [-1, +1].

        Positive = bullish pressure, Negative = bearish pressure.

        Components:
          1. Depth imbalance (vol_imbalance): (bid_vol - ask_vol) / (bid_vol + ask_vol)
          2. OFI (multi-window average): positive = more buy orders
          3. Trade imbalance: signed trade flow

        All inputs are already computed in mbo_features.py -- just read directly.
        """
        # Component 1: Depth imbalance (already in [-1, 1] from feature engineering)
        depth_imb = np.clip(float(bar.vol_imbalance), -1.0, 1.0)

        # Component 2: OFI composite (average over windows, clip to [-1, 1])
        # OFI features are flow imbalance: positive = net buying
        ofi_composite = np.clip(
            (float(bar.ofi_5) + float(bar.ofi_20) + float(bar.ofi_50)) / 3.0,
            -1.0, 1.0
        )

        # Component 3: Trade imbalance (net buy vs sell volume ratio)
        trade_imb = np.clip(float(bar.trade_imbalance), -1.0, 1.0)

        score = (self.depth_weight  * depth_imb
                 + self.ofi_weight  * ofi_composite
                 + self.trade_imb_weight * trade_imb)

        return float(np.clip(score, -1.0, 1.0))

    def signal_agrees_with_imbalance(self,
                                      imbalance_score: float,
                                      signal_direction: int) -> bool:
        """Return True if book pressure aligns with signal direction.

        Args:
            imbalance_score: from compute_imbalance_score(), [-1, +1]
            signal_direction: +1 (long) or -1 (short)
        """
        # Agreement: signal and imbalance same sign, beyond boost threshold
        return float(signal_direction) * imbalance_score >= self.boost_threshold

    def evaluate(self, bar: BarData, signal_direction: int) -> Dict:
        """Full evaluation: score, agreement, and confidence adjustment.

        Args:
            bar: current BarData snapshot
            signal_direction: +1 or -1

        Returns:
            dict with keys: score, agrees, confidence_adj, skip
        """
        score = self.compute_imbalance_score(bar)
        aligned_score = float(signal_direction) * score  # positive = agreement

        agrees = aligned_score >= self.boost_threshold
        skip   = aligned_score <= -self.skip_threshold

        # Confidence adjustment: linear in [-0.3, +0.3]
        confidence_adj = float(np.clip(aligned_score * 0.3, -0.3, 0.3))

        return {
            'score': score,
            'aligned_score': aligned_score,
            'agrees': agrees,
            'skip': skip,
            'confidence_adj': confidence_adj,
        }


# ============================================================================
# 3. PASSIVE-TO-AGGRESSIVE ESCALATOR
# ============================================================================

class PassiveToAggressiveEscalator:
    """State machine for converting a passive limit order into a market order.

    Escalation schedule depends on signal strength:
        Strong signal -> escalate fast (fewer bars per stage)
        Weak signal   -> escalate slow (more time for passive fill)

    State flow:
        INACTIVE -> PASSIVE -> MIDPOINT -> AGGRESSIVE -> MARKET -> DONE

    Usage:
        esc = PassiveToAggressiveEscalator()
        schedule = esc.build_schedule(signal_strength=0.7, horizon='ret_30s')
        # schedule: [(10, 'limit'), (20, 'midpoint'), (30, 'aggressive_limit'), (50, 'market')]

        # Per bar:
        esc.place_initial_order(start_bar_index=i, signal_strength=0.7, horizon='ret_30s')
        current_type = esc.get_current_order_type(bar_index=j)
        # Returns current order type or None if escalation is done/inactive
    """

    STATES = ['limit', 'midpoint', 'aggressive_limit', 'market', 'done', 'inactive']

    def __init__(self,
                 base_passive_bars: int = 30,    # bars at limit stage at median signal
                 base_midpoint_bars: int = 20,   # bars at midpoint stage
                 base_aggressive_bars: int = 15, # bars at aggressive limit stage
                 min_stage_bars: int = 5):        # minimum bars per stage
        self.base_passive_bars    = base_passive_bars
        self.base_midpoint_bars   = base_midpoint_bars
        self.base_aggressive_bars = base_aggressive_bars
        self.min_stage_bars       = min_stage_bars

        self._schedule: List[Tuple[int, str]] = []  # (cutoff_bar, order_type)
        self._start_bar: int = -1
        self._state: str = 'inactive'

    def _signal_to_speed_factor(self, signal_strength: float) -> float:
        """Convert signal strength [0, 1] to escalation speed factor.

        Higher signal -> lower factor -> fewer bars per stage -> faster escalation.
        Factor range: [0.3, 1.5]
          signal=1.0 -> factor=0.3 (escalate 3x faster)
          signal=0.5 -> factor=0.85
          signal=0.1 -> factor=1.5 (escalate slowly)
        """
        # Linear interpolation
        return float(np.clip(1.5 - 1.2 * signal_strength, 0.3, 1.5))

    def build_schedule(self,
                        signal_strength: float,
                        horizon: str) -> List[Tuple[int, str]]:
        """Build escalation schedule: list of (bars_offset, order_type).

        The schedule defines when to upgrade the order type relative to
        the initial placement bar.

        Args:
            signal_strength: normalized [0, 1]
            horizon: used to cap total escalation time to horizon window

        Returns:
            List of (bars_offset_from_start, order_type) tuples.
            First entry is always (0, 'limit').
        """
        speed = self._signal_to_speed_factor(signal_strength)
        horizon_bars = HORIZON_PARAMS.get(horizon, HORIZON_PARAMS['ret_30s'])['bars']

        passive_bars    = max(self.min_stage_bars, int(self.base_passive_bars    * speed))
        midpoint_bars   = max(self.min_stage_bars, int(self.base_midpoint_bars   * speed))
        aggressive_bars = max(self.min_stage_bars, int(self.base_aggressive_bars * speed))

        schedule = [
            (0,                                       'limit'),
            (passive_bars,                             'midpoint'),
            (passive_bars + midpoint_bars,             'aggressive_limit'),
            (passive_bars + midpoint_bars + aggressive_bars, 'market'),
        ]

        # Clip so we don't escalate beyond the signal horizon
        clipped = [(t, ot) for t, ot in schedule if t < horizon_bars]
        if not clipped or clipped[-1][1] != 'market':
            # Force market at horizon - 1 bar to ensure we exit within horizon
            clipped.append((horizon_bars - 1, 'market'))

        return clipped

    def place_initial_order(self,
                             start_bar_index: int,
                             signal_strength: float,
                             horizon: str) -> List[Tuple[int, str]]:
        """Start a new escalation sequence.

        Args:
            start_bar_index: absolute bar index when order is first placed
            signal_strength: [0, 1]
            horizon: trading horizon key

        Returns:
            Escalation schedule with absolute bar indices
        """
        relative_schedule = self.build_schedule(signal_strength, horizon)
        self._start_bar = start_bar_index
        self._schedule  = [(start_bar_index + t, ot) for t, ot in relative_schedule]
        self._state     = 'limit'
        return self._schedule

    def check_escalation(self, bar_index: int) -> Optional[str]:
        """Check if the order should be upgraded at this bar.

        Returns:
            New order type if escalation occurs this bar, else None.
        """
        if self._state in ('done', 'inactive'):
            return None

        new_type = None
        for cutoff, order_type in reversed(self._schedule):
            if bar_index >= cutoff:
                new_type = order_type
                break

        if new_type and new_type != self._state:
            self._state = new_type
            return new_type
        return None

    def get_current_order_type(self, bar_index: int) -> Optional[str]:
        """Return the current order type at this bar index.

        Returns:
            Current order type string, or None if inactive/done.
        """
        if self._state in ('done', 'inactive'):
            return None

        current = 'limit'
        for cutoff, order_type in self._schedule:
            if bar_index >= cutoff:
                current = order_type

        if current == 'market' and self._state != 'done':
            self._state = 'done'

        return current

    def cancel(self):
        """Mark escalation as done (order filled or cancelled externally)."""
        self._state = 'done'

    def reset(self):
        """Reset to inactive state."""
        self._schedule = []
        self._start_bar = -1
        self._state = 'inactive'

    @property
    def is_active(self) -> bool:
        return self._state not in ('done', 'inactive')

    @property
    def state(self) -> str:
        return self._state


# ============================================================================
# 4. VOL REGIME GATE (intraday extension)
# ============================================================================

class VolRegimeGate:
    """Intraday vol regime detection using rolling realized volatility.

    Extends the daily-level vol_regime_gate.py with within-session
    rolling windows so we can detect dead-market microstructure in real time.

    Two signals:
      1. Daily vol gate (inherited from vol_regime_gate.py logic):
         active if today's rolling vol >= daily_threshold_bps
      2. Intraday rolling vol gate:
         active if recent N-bar vol >= intraday_threshold_bps

    A bar is tradeable only when BOTH gates are open.

    Args:
        daily_threshold_bps: vol threshold for daily regime (default from prior research)
        intraday_window: bars for rolling intraday vol (default 500 = 50s)
        intraday_threshold_bps: min intraday vol to trade (default 0.03 bps)
        lookback_daily_bars: how many bars to use for daily vol estimate

    Usage:
        gate = VolRegimeGate()
        gate.update(mid_price=5000.25)  # call each bar
        if gate.is_active_regime():
            ...
    """

    REGIME_DEAD   = 'DEAD'
    REGIME_ACTIVE = 'ACTIVE'
    REGIME_LOW    = 'LOW_VOL'
    REGIME_HIGH   = 'HIGH_VOL'

    def __init__(self,
                 daily_threshold_bps: float = 0.06,
                 intraday_window: int = 500,
                 intraday_threshold_bps: float = 0.02,
                 daily_window: int = 27000):    # ~45 min of 100ms bars
        self.daily_threshold_bps    = daily_threshold_bps
        self.intraday_window        = intraday_window
        self.intraday_threshold_bps = intraday_threshold_bps
        self.daily_window           = daily_window

        # Circular buffers for rolling log returns
        self._intraday_log_rets: deque = deque(maxlen=intraday_window)
        self._daily_log_rets: deque    = deque(maxlen=daily_window)
        self._prev_mid: float          = 0.0

        # Cached regime values
        self._intraday_vol_bps: float = 0.0
        self._daily_vol_bps: float    = 0.0
        self._regime: str             = self.REGIME_DEAD

    def update(self, mid_price: float) -> str:
        """Feed one bar mid price. Returns current regime label.

        Call this once per bar before calling is_active_regime().

        Args:
            mid_price: current mid price (must be > 0)

        Returns:
            Regime label: 'DEAD' | 'LOW_VOL' | 'ACTIVE' | 'HIGH_VOL'
        """
        if self._prev_mid > 0 and mid_price > 0:
            log_ret = float(np.log(mid_price / self._prev_mid))
            self._intraday_log_rets.append(log_ret)
            self._daily_log_rets.append(log_ret)

        self._prev_mid = float(mid_price)
        self._regime   = self._compute_regime()
        return self._regime

    def _compute_regime(self) -> str:
        """Internal: compute regime from current buffers."""
        if len(self._intraday_log_rets) < max(50, self.intraday_window // 5):
            return self.REGIME_DEAD  # not enough data yet

        # Intraday vol
        intra_arr = np.array(self._intraday_log_rets, dtype=np.float32)
        self._intraday_vol_bps = float(np.std(intra_arr)) * 1e4

        # Daily vol
        if len(self._daily_log_rets) >= 100:
            daily_arr = np.array(self._daily_log_rets, dtype=np.float32)
            self._daily_vol_bps = float(np.std(daily_arr)) * 1e4
        else:
            self._daily_vol_bps = self._intraday_vol_bps

        # Gate logic
        daily_ok   = self._daily_vol_bps   >= self.daily_threshold_bps
        intraday_ok = self._intraday_vol_bps >= self.intraday_threshold_bps

        if not daily_ok or not intraday_ok:
            return self.REGIME_DEAD

        # Sub-classify by intraday vol level (for position sizing)
        p33 = self.daily_threshold_bps * 1.5
        p67 = self.daily_threshold_bps * 2.5
        if self._intraday_vol_bps < p33:
            return self.REGIME_LOW
        elif self._intraday_vol_bps > p67:
            return self.REGIME_HIGH
        else:
            return self.REGIME_ACTIVE

    def is_active_regime(self) -> bool:
        """Return True if current regime supports trading.

        LOW_VOL is borderline -- included for now, can be excluded if needed.
        """
        return self._regime in (self.REGIME_ACTIVE, self.REGIME_LOW, self.REGIME_HIGH)

    def get_regime(self) -> str:
        """Return current regime label."""
        return self._regime

    def get_gate_signal(self) -> Dict:
        """Return full regime state for logging/debugging.

        Returns:
            dict with regime, intraday_vol_bps, daily_vol_bps, is_active
        """
        return {
            'regime': self._regime,
            'intraday_vol_bps': round(self._intraday_vol_bps, 4),
            'daily_vol_bps': round(self._daily_vol_bps, 4),
            'is_active': self.is_active_regime(),
            'daily_threshold_bps': self.daily_threshold_bps,
            'intraday_threshold_bps': self.intraday_threshold_bps,
        }

    def reset_day(self):
        """Call at the start of each new trading day to reset daily buffer."""
        self._daily_log_rets.clear()
        self._daily_vol_bps = 0.0


# ============================================================================
# 5. PERFORMANCE CONDITION ANALYZER
# ============================================================================

class PerformanceConditionAnalyzer:
    """Track historical model performance conditioned on market state.

    Bins predictions and outcomes by condition dimensions:
        - hour_of_day  (9 bins: 9am-4pm RTH hours)
        - vol_regime   (3 bins: low/mid/high realized vol)
        - spread_level (3 bins: 1-tick / 2-tick / 3+ tick spread)
        - trend_state  (3 bins: trending up / flat / trending down)

    Uses a rolling window (last N observations) to keep conditions adaptive.

    Causal guarantee: conditions for bar i are computed from bars 0..i-1 only.

    Args:
        window: max observations to keep per condition bin
        min_samples: minimum samples needed before reporting non-neutral score

    Usage:
        analyzer = PerformanceConditionAnalyzer()
        # Per bar (after trade result is known):
        analyzer.update(
            prediction=0.7,
            actual_return=0.002,
            conditions={'hour': 10, 'vol_bps': 0.08, 'spread_ticks': 1.0, 'trend': 0.001}
        )
        # Before trade decision:
        score = analyzer.get_condition_score(
            conditions={'hour': 10, 'vol_bps': 0.08, 'spread_ticks': 1.0, 'trend': 0.001}
        )
        # score in [0, 1]: 0=historically poor, 1=historically excellent
    """

    def __init__(self, window: int = 2000, min_samples: int = 20):
        self.window      = window
        self.min_samples = min_samples

        # Storage: condition_key -> deque of (prediction, actual) pairs
        self._bins: Dict[str, deque] = {}

        # Bin definitions (edges for np.digitize)
        # Hour bins: 9-10, 10-11, ..., 15-16 (7 bins for RTH)
        self._hour_edges = np.arange(9, 17, dtype=float)  # [9,10,11,...,16]
        # Vol bins (bps): <0.04, 0.04-0.08, >0.08
        self._vol_edges  = np.array([0.04, 0.08], dtype=float)
        # Spread bins (ticks): 1, 2, 3+
        self._spread_edges = np.array([1.5, 2.5], dtype=float)
        # Trend bins (5-bar momentum in bps): trending down, flat, trending up
        self._trend_edges = np.array([-0.5, 0.5], dtype=float)  # bps

    def _bin_conditions(self, conditions: Dict) -> str:
        """Convert conditions dict to a hashable bin key string.

        Args:
            conditions: dict with keys 'hour', 'vol_bps', 'spread_ticks', 'trend'

        Returns:
            String key like 'h2_v1_s0_t1'
        """
        h = int(np.digitize(conditions.get('hour', 12.0), self._hour_edges))
        v = int(np.digitize(conditions.get('vol_bps', 0.06), self._vol_edges))
        s = int(np.digitize(conditions.get('spread_ticks', 1.0), self._spread_edges))
        t = int(np.digitize(conditions.get('trend', 0.0), self._trend_edges))
        return f'h{h}_v{v}_s{s}_t{t}'

    def update(self,
               prediction: float,
               actual_return: float,
               conditions: Dict):
        """Record one prediction/outcome pair for a given set of conditions.

        IMPORTANT: Call this AFTER the trade result is known (next bar or after
        hold period ends). Never call with same-bar actual_return.

        Args:
            prediction: raw model output for this bar
            actual_return: realized return over the hold period
            conditions: dict with 'hour', 'vol_bps', 'spread_ticks', 'trend'
        """
        if not (np.isfinite(prediction) and np.isfinite(actual_return)):
            return

        key = self._bin_conditions(conditions)
        if key not in self._bins:
            self._bins[key] = deque(maxlen=self.window)

        # Store signed correctness: prediction and actual same sign = correct
        correct = float(np.sign(prediction) == np.sign(actual_return))
        self._bins[key].append(correct)

    def get_condition_score(self, conditions: Dict) -> float:
        """Return performance score for current market conditions.

        Score = recent hit rate in this condition bin.
        Falls back to adjacent bins and global average if bin is empty.

        Args:
            conditions: dict with 'hour', 'vol_bps', 'spread_ticks', 'trend'

        Returns:
            float in [0, 1]: 0=historically poor, 0.5=neutral, 1=excellent
            Returns 0.5 (neutral) when insufficient data.
        """
        key = self._bin_conditions(conditions)

        if key in self._bins and len(self._bins[key]) >= self.min_samples:
            arr = np.array(self._bins[key], dtype=np.float32)
            hit_rate = float(np.mean(arr))
            # Normalize: 50% hit rate -> 0.5 score, 70% -> 1.0, 30% -> 0.0
            score = np.clip((hit_rate - 0.3) / 0.4, 0.0, 1.0)
            return float(score)

        # Fallback: aggregate across all bins
        all_vals = []
        for d in self._bins.values():
            all_vals.extend(d)

        if len(all_vals) >= self.min_samples:
            global_hr = float(np.mean(all_vals))
            return float(np.clip((global_hr - 0.3) / 0.4, 0.0, 1.0))

        return 0.5  # neutral when no data

    def get_stats(self) -> Dict:
        """Return summary statistics across all condition bins.

        Returns:
            dict with n_bins, total_samples, best_bin, worst_bin, global_hit_rate
        """
        if not self._bins:
            return {'n_bins': 0, 'total_samples': 0, 'global_hit_rate': 0.5}

        bin_stats = []
        all_vals = []
        for key, dq in self._bins.items():
            arr = np.array(dq, dtype=np.float32)
            all_vals.extend(arr)
            if len(arr) >= self.min_samples:
                bin_stats.append({'key': key, 'n': len(arr), 'hit_rate': float(np.mean(arr))})

        bin_stats.sort(key=lambda x: x['hit_rate'], reverse=True)

        return {
            'n_bins': len(self._bins),
            'total_samples': len(all_vals),
            'global_hit_rate': float(np.mean(all_vals)) if all_vals else 0.5,
            'best_bin': bin_stats[0] if bin_stats else None,
            'worst_bin': bin_stats[-1] if bin_stats else None,
            'n_bins_with_data': len(bin_stats),
        }


# ============================================================================
# 6. SMART EXECUTION ENGINE (orchestrator)
# ============================================================================

class SmartExecutionEngine:
    """Orchestrate all execution components into one decision pipeline.

    Decision flow for each bar:
        1. Update VolRegimeGate with current mid price
        2. Check if regime is active -> if not, return skip
        3. Get condition score from PerformanceConditionAnalyzer
        4. Evaluate book imbalance with ImbalanceFilter
        5. Compute composite confidence score
        6. Route to order type via ExecutionRouter
        7. If passive order, build escalation schedule

    The engine is STATELESS between calls except for:
        - VolRegimeGate state (rolling vol buffers)
        - PerformanceConditionAnalyzer state (historical performance)

    Args:
        mode: 'backtest' or 'live' (affects logging verbosity)
        max_contracts: maximum position size
        min_confidence: skip if composite confidence below this
        daily_vol_threshold_bps: pass-through to VolRegimeGate

    Usage:
        engine = SmartExecutionEngine()

        for i in range(len(bars)):
            bar = bars[i]
            decision = engine.process(
                signal_strength=abs(pred[i]),
                signal_direction=int(np.sign(pred[i])),
                horizon='ret_30s',
                bar=bar,
                bar_index=i,
            )
            if decision.action == 'enter':
                # Submit order
                open_trade(decision)
                # Start escalation tracking
                engine.escalator.place_initial_order(i, abs(pred[i]), 'ret_30s')
            elif decision.action == 'skip':
                pass

            # Later bars -- check escalation on open orders
            new_type = engine.escalator.check_escalation(i)
            if new_type:
                replace_order(new_type)

            # After trade closes -- update performance tracker
            if trade_closed:
                engine.record_outcome(
                    prediction=pred[entry_bar],
                    actual_return=realized_return,
                    bar=bars[entry_bar],
                )
    """

    def __init__(self,
                 mode: str = 'backtest',
                 max_contracts: int = 2,
                 min_confidence: float = 0.3,
                 daily_vol_threshold_bps: float = 0.06,
                 intraday_vol_threshold_bps: float = 0.02,
                 imbalance_skip_threshold: float = 0.6,
                 imbalance_boost_threshold: float = 0.3):
        self.mode            = mode
        self.max_contracts   = max_contracts
        self.min_confidence  = min_confidence

        # Sub-components
        self.vol_gate    = VolRegimeGate(
            daily_threshold_bps=daily_vol_threshold_bps,
            intraday_threshold_bps=intraday_vol_threshold_bps,
        )
        self.imb_filter  = ImbalanceFilter(
            skip_threshold=imbalance_skip_threshold,
            boost_threshold=imbalance_boost_threshold,
        )
        self.router      = ExecutionRouter()
        self.escalator   = PassiveToAggressiveEscalator()
        self.perf_analyzer = PerformanceConditionAnalyzer()

        # Running counters for diagnostics
        self._n_processed: int = 0
        self._n_entered: int   = 0
        self._n_skipped_regime: int    = 0
        self._n_skipped_imbalance: int = 0
        self._n_skipped_confidence: int = 0

    def _extract_conditions(self, bar: BarData) -> Dict:
        """Build conditions dict from BarData for PerformanceConditionAnalyzer."""
        return {
            'hour': float(bar.hour_norm * 7.0 + 9.0),    # RTH: 9-16
            'vol_bps': float(bar.realized_vol_50),
            'spread_ticks': float(bar.spread_ticks),
            'trend': float(bar.ret_5 * 1e4),              # 5-bar return in bps
        }

    def process(self,
                signal_strength: float,
                signal_direction: int,
                horizon: str,
                bar: BarData,
                bar_index: int) -> ExecutionDecision:
        """Main entry point: process one bar and return a trading decision.

        Args:
            signal_strength: abs(normalized_prediction), in [0, 1]
            signal_direction: +1 (long) or -1 (short)
            horizon: 'ret_3s' | 'ret_10s' | 'ret_30s' | 'ret_1m'
            bar: current BarData snapshot (all fields causal)
            bar_index: absolute bar counter (for escalation timing)

        Returns:
            ExecutionDecision with action='enter'|'skip'|'hold'
        """
        self._n_processed += 1

        # --- 1. Vol regime gate ---
        regime = self.vol_gate.update(bar.mid)
        regime_active = self.vol_gate.is_active_regime()

        if not regime_active:
            self._n_skipped_regime += 1
            return ExecutionDecision(
                action='skip',
                signal_strength=signal_strength,
                signal_direction=signal_direction,
                horizon=horizon,
                regime_active=False,
                reasoning=f'SKIP: vol regime={regime} (DEAD), no edge',
            )

        # --- 2. Performance condition score ---
        conditions   = self._extract_conditions(bar)
        cond_score   = self.perf_analyzer.get_condition_score(conditions)

        # --- 3. Imbalance filter ---
        imb_result   = self.imb_filter.evaluate(bar, signal_direction)

        if imb_result['skip']:
            self._n_skipped_imbalance += 1
            return ExecutionDecision(
                action='skip',
                signal_strength=signal_strength,
                signal_direction=signal_direction,
                horizon=horizon,
                regime_active=True,
                imbalance_score=imb_result['score'],
                condition_score=cond_score,
                reasoning=(
                    f'SKIP: book strongly disagrees with signal. '
                    f'imbalance_score={imb_result["score"]:.3f}, '
                    f'signal_dir={signal_direction:+d}'
                ),
            )

        # --- 4. Composite confidence score ---
        # Base: signal_strength (model confidence)
        # Adj:  imbalance adjustment (book alignment)
        # Adj:  condition score (historical performance in this regime)
        # Adj:  vol regime bonus (HIGH_VOL has slightly more edge)
        regime_bonus = 0.05 if regime == VolRegimeGate.REGIME_HIGH else 0.0
        confidence = float(np.clip(
            signal_strength * 0.5
            + cond_score     * 0.3
            + imb_result['confidence_adj']
            + regime_bonus,
            0.0, 1.0
        ))

        if confidence < self.min_confidence:
            self._n_skipped_confidence += 1
            return ExecutionDecision(
                action='skip',
                signal_strength=signal_strength,
                signal_direction=signal_direction,
                horizon=horizon,
                confidence=confidence,
                regime_active=True,
                imbalance_score=imb_result['score'],
                condition_score=cond_score,
                reasoning=(
                    f'SKIP: confidence={confidence:.3f} < min={self.min_confidence}. '
                    f'signal={signal_strength:.3f}, cond={cond_score:.3f}, '
                    f'imb_adj={imb_result["confidence_adj"]:+.3f}'
                ),
            )

        # --- 5. Route to order type ---
        order_type, exp_ticks = self.router.route(signal_strength, horizon)

        if order_type == 'skip':
            return ExecutionDecision(
                action='skip',
                signal_strength=signal_strength,
                signal_direction=signal_direction,
                horizon=horizon,
                confidence=confidence,
                regime_active=True,
                imbalance_score=imb_result['score'],
                condition_score=cond_score,
                expected_ticks=exp_ticks,
                reasoning=(
                    f'SKIP: expected_ticks={exp_ticks:.3f} < cost threshold. '
                    f'Horizon {horizon} has insufficient edge at this signal level.'
                ),
            )

        # --- 6. Position sizing ---
        size = self.router.size_from_confidence(confidence, self.max_contracts)
        if size == 0:
            return ExecutionDecision(
                action='skip',
                signal_strength=signal_strength,
                signal_direction=signal_direction,
                horizon=horizon,
                confidence=confidence,
                regime_active=True,
                imbalance_score=imb_result['score'],
                condition_score=cond_score,
                expected_ticks=exp_ticks,
                reasoning=f'SKIP: size rounded to 0 at confidence={confidence:.3f}',
            )

        # --- 7. Escalation schedule (for passive orders) ---
        escalation_schedule = []
        if order_type in ('limit', 'midpoint'):
            escalation_schedule = self.escalator.place_initial_order(
                bar_index, signal_strength, horizon
            )
        else:
            self.escalator.reset()

        self._n_entered += 1

        reasoning = (
            f'ENTER {signal_direction:+d} x{size} {order_type} | '
            f'horizon={horizon} | '
            f'signal={signal_strength:.3f} | '
            f'exp_ticks={exp_ticks:.2f} | '
            f'confidence={confidence:.3f} | '
            f'regime={regime} | '
            f'imbalance={imb_result["score"]:+.3f} ({"agrees" if imb_result["agrees"] else "neutral"}) | '
            f'cond_score={cond_score:.3f}'
        )

        if self.mode == 'live':
            log.info(reasoning)

        return ExecutionDecision(
            action='enter',
            order_type=order_type,
            size=size,
            confidence=confidence,
            reasoning=reasoning,
            escalation_schedule=escalation_schedule,
            signal_strength=signal_strength,
            signal_direction=signal_direction,
            horizon=horizon,
            expected_ticks=exp_ticks,
            regime_active=True,
            imbalance_score=imb_result['score'],
            condition_score=cond_score,
        )

    def record_outcome(self,
                        prediction: float,
                        actual_return: float,
                        bar: BarData):
        """Update the PerformanceConditionAnalyzer after a trade closes.

        MUST be called with the entry bar's BarData, not the exit bar.
        actual_return is the realized return from entry to exit.

        Args:
            prediction: model prediction at trade entry
            actual_return: realized return (signed, same sign as prediction = win)
            bar: BarData snapshot at trade entry bar
        """
        conditions = self._extract_conditions(bar)
        self.perf_analyzer.update(prediction, actual_return, conditions)

    def get_diagnostics(self) -> Dict:
        """Return engine statistics for monitoring.

        Returns:
            dict with entry rate, skip breakdown, vol gate state, perf stats
        """
        entry_rate = (self._n_entered / max(1, self._n_processed))
        skip_regime     = self._n_skipped_regime
        skip_imbalance  = self._n_skipped_imbalance
        skip_confidence = self._n_skipped_confidence

        return {
            'n_processed': self._n_processed,
            'n_entered': self._n_entered,
            'entry_rate_pct': round(entry_rate * 100, 2),
            'skips': {
                'regime': skip_regime,
                'imbalance': skip_imbalance,
                'confidence': skip_confidence,
                'router': (self._n_processed
                           - self._n_entered
                           - skip_regime
                           - skip_imbalance
                           - skip_confidence),
            },
            'vol_gate': self.vol_gate.get_gate_signal(),
            'perf_analyzer': self.perf_analyzer.get_stats(),
        }


# ============================================================================
# 7. BACKTESTING HELPER
# ============================================================================

def simulate_execution(predictions: np.ndarray,
                        mid_prices: np.ndarray,
                        bar_data_arr: Optional[List[BarData]],
                        horizon: str = 'ret_30s',
                        max_contracts: int = 1,
                        min_confidence: float = 0.3,
                        entry_threshold_pct: float = 70.0,
                        verbose: bool = False) -> Dict:
    """Vectorized backtesting loop using SmartExecutionEngine.

    Applies the full engine bar-by-bar on historical data to estimate
    what the entry filter would have done.

    NOTE: This does NOT simulate actual fills or slippage -- it only
    counts which bars the engine would have entered and at what order type.
    For full PnL simulation, combine with run_hybrid_execution_sim.py logic.

    Args:
        predictions: LightGBM raw predictions, shape (N,)
        mid_prices: mid prices, shape (N,)
        bar_data_arr: list of BarData objects (one per bar) or None
            If None, a minimal BarData is constructed from mid_prices only.
        horizon: trading horizon key
        max_contracts: max position size
        min_confidence: min confidence to enter
        entry_threshold_pct: percentile of |signal| required to even consider entry
        verbose: log each decision

    Returns:
        dict with entries, entry_rate, order_type_counts, condition_coverage
    """
    N = len(predictions)
    assert len(mid_prices) == N, "predictions and mid_prices must be same length"

    # Normalize signals to [0, 1] using percentile rank
    valid = np.isfinite(predictions)
    signal_ranks = np.full(N, 0.5, dtype=np.float32)
    if valid.sum() > 10:
        from scipy.stats import rankdata
        signal_ranks[valid] = (rankdata(np.abs(predictions[valid])) /
                                valid.sum()).astype(np.float32)

    # Entry threshold filter (only consider top (100 - pct)% signals)
    strength_threshold = np.percentile(signal_ranks[valid], entry_threshold_pct)

    engine = SmartExecutionEngine(
        mode='backtest',
        max_contracts=max_contracts,
        min_confidence=min_confidence,
    )

    entries: List[Dict] = []
    order_type_counts: Dict[str, int] = {}

    for i in range(N):
        if not valid[i]:
            engine.vol_gate.update(float(mid_prices[i]) if np.isfinite(mid_prices[i]) else 0.0)
            continue

        sig_strength = float(signal_ranks[i])
        sig_dir = int(np.sign(predictions[i]))
        if sig_dir == 0:
            sig_dir = 1

        # Only process if signal is strong enough to even consider
        if sig_strength < strength_threshold / 100.0:
            engine.vol_gate.update(float(mid_prices[i]))
            continue

        # Build bar (from array if provided, else minimal)
        if bar_data_arr is not None and i < len(bar_data_arr):
            bar = bar_data_arr[i]
        else:
            bar = BarData(mid=float(mid_prices[i]))

        decision = engine.process(
            signal_strength=sig_strength,
            signal_direction=sig_dir,
            horizon=horizon,
            bar=bar,
            bar_index=i,
        )

        if decision.action == 'enter':
            entry_rec = {
                'bar_index': i,
                'order_type': decision.order_type,
                'size': decision.size,
                'confidence': decision.confidence,
                'signal_direction': decision.signal_direction,
                'expected_ticks': decision.expected_ticks,
                'regime': engine.vol_gate.get_regime(),
            }
            entries.append(entry_rec)

            ot = decision.order_type or 'unknown'
            order_type_counts[ot] = order_type_counts.get(ot, 0) + 1

            if verbose:
                log.info(f"Bar {i}: {decision.reasoning}")

    total_considered = int((signal_ranks >= strength_threshold / 100.0).sum())
    entry_rate = len(entries) / max(1, total_considered)

    diag = engine.get_diagnostics()

    return {
        'n_bars': N,
        'n_considered': total_considered,
        'n_entries': len(entries),
        'entry_rate_pct': round(entry_rate * 100, 2),
        'order_type_counts': order_type_counts,
        'entries': entries,
        'diagnostics': diag,
        'horizon': horizon,
    }


# ============================================================================
# QUICK SANITY CHECK (run standalone)
# ============================================================================

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        stream=sys.stdout)

    log.info("=== SmartExecutionEngine sanity check ===")

    # Minimal synthetic data
    np.random.seed(42)
    N = 5000
    mid = 5000.0 + np.cumsum(np.random.randn(N) * 0.25)
    preds = np.random.randn(N) * 0.5

    # Build bar list
    bars = []
    for i in range(N):
        bars.append(BarData(
            mid=float(mid[i]),
            spread=0.25,
            vol_imbalance=float(np.random.randn() * 0.2),
            ofi_5=float(np.random.randn() * 0.3),
            ofi_20=float(np.random.randn() * 0.2),
            ofi_50=float(np.random.randn() * 0.15),
            trade_imbalance=float(np.random.randn() * 0.25),
            hour_norm=float((i % 27000) / 27000.0),
            ret_5=float(np.random.randn() * 0.0002),
            realized_vol_50=float(np.abs(np.random.randn()) * 0.05 + 0.04),
            spread_ticks=1.0,
        ))

    result = simulate_execution(
        predictions=preds,
        mid_prices=mid,
        bar_data_arr=bars,
        horizon='ret_30s',
        max_contracts=1,
        min_confidence=0.3,
        entry_threshold_pct=70.0,
        verbose=False,
    )

    log.info(f"Bars processed : {result['n_bars']:,}")
    log.info(f"Bars considered: {result['n_considered']:,}")
    log.info(f"Entries        : {result['n_entries']:,}")
    log.info(f"Entry rate     : {result['entry_rate_pct']:.1f}%")
    log.info(f"Order types    : {result['order_type_counts']}")
    log.info(f"Diagnostics    : {result['diagnostics']}")

    # Test individual components
    log.info("")
    log.info("--- ExecutionRouter ---")
    router = ExecutionRouter()
    for horizon in ['ret_3s', 'ret_10s', 'ret_30s', 'ret_1m']:
        for strength in [0.2, 0.5, 0.8]:
            ot, et = router.route(strength, horizon)
            log.info(f"  {horizon} sig={strength:.1f} -> {ot} (exp={et:.2f} ticks)")

    log.info("")
    log.info("--- PassiveToAggressiveEscalator ---")
    esc = PassiveToAggressiveEscalator()
    sched = esc.build_schedule(signal_strength=0.7, horizon='ret_30s')
    log.info(f"  Schedule (rel bars): {sched}")
    abs_sched = esc.place_initial_order(start_bar_index=1000, signal_strength=0.7, horizon='ret_30s')
    log.info(f"  Schedule (abs bars): {abs_sched}")
    for b in [1000, 1010, 1025, 1045, 1060]:
        ot = esc.get_current_order_type(b)
        log.info(f"  bar={b} -> {ot}")

    log.info("")
    log.info("--- VolRegimeGate ---")
    gate = VolRegimeGate()
    for p in mid[:600]:
        gate.update(p)
    log.info(f"  Gate state: {gate.get_gate_signal()}")

    log.info("")
    log.info("=== All checks passed ===")
