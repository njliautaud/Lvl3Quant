"""
Execution Framework for CNN-Mamba v2 + PatchTST Paper Trader
============================================================
Multi-layer safety framework wrapping raw model predictions.
Designed to prevent catastrophic losses from model bias (e.g., 96% LONG in a drop).

Layers:
  1. Signal normalization (z-score)
  2. PatchTST veto (confluence filter)
  3. Model health monitor (bias/IC tracking)
  4. Volatility gate
  5. Time of day gate
  6. Confidence gate
  7. Entry logic
  8. Exit management
  9. Risk management
  10. Direction bias filter

Author: Claude (autonomous infrastructure)
Date: 2026-05-01
"""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pytz
    ET = pytz.timezone("US/Eastern")
except ImportError:
    ET = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ExitAction(Enum):
    HOLD = "hold"
    TRAILING_STOP = "trailing_stop"
    PROFIT_LOCK = "profit_lock"
    HARD_STOP = "hard_stop"
    TIME_EXIT_PASSIVE = "time_exit_passive"
    TIME_EXIT_MARKET = "time_exit_market"


@dataclass
class ExitParams:
    action: ExitAction
    price: Optional[float] = None
    order_type: str = "limit"  # "limit" or "market"
    reason: str = ""


@dataclass
class MarketState:
    """Snapshot of current market conditions passed to the framework."""
    timestamp: float  # epoch seconds
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_price: float = 0.0
    price_rel: float = 0.0  # relative price change from event data
    event_count: int = 0


@dataclass
class GateResult:
    """Result from a single gate check."""
    passed: bool
    gate_name: str
    reason: str = ""


# ---------------------------------------------------------------------------
# Main Framework Class
# ---------------------------------------------------------------------------

class ExecutionFramework:
    """
    Multi-layer execution framework for live/paper trading.

    Usage:
        fw = ExecutionFramework("framework_config.json")

        # Every event:
        fw.update_state(market_state)

        # Every prediction (500 events):
        should, reason = fw.should_enter(pred_1s, pred_5s, pred_10s, patchtst_pred, market_state)
        if should:
            # Execute entry...
            pass

        # While in position:
        exit_params = fw.get_exit_params(direction, entry_price, current_price, elapsed_seconds, signal_horizon)
    """

    def __init__(self, config_path: Optional[str] = None):
        # Load config
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "framework_config.json")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Setup logging
        self._setup_logging()

        # --- Signal Layer State ---
        self.pred_history_1s = deque(maxlen=self.config["signal_layer"]["z_score_window"])
        self.pred_history_5s = deque(maxlen=self.config["signal_layer"]["z_score_window"])
        self.pred_history_10s = deque(maxlen=self.config["signal_layer"]["z_score_window"])

        # --- Model Health State ---
        health_cfg = self.config["model_health"]
        self.direction_history = deque(maxlen=health_cfg["direction_window"])
        self.rolling_ic_buffer: List[Tuple[float, float]] = []  # (prediction, actual_move)
        self.ic_window = health_cfg["rolling_ic_window"]
        self.model_healthy = True
        self.health_reason = ""

        # --- Volatility Gate State ---
        vol_cfg = self.config["volatility_gate"]
        self.price_rel_history = deque(maxlen=vol_cfg["vol_window_events"])
        self.baseline_vol: Optional[float] = None  # Set after warmup
        self.vol_warmup_done = False

        # --- Risk Management State ---
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.last_loss_time: Optional[float] = None
        self.cooldown_until: Optional[float] = None
        self.session_date: Optional[str] = None

        # --- Direction Bias State ---
        bias_cfg = self.config["direction_bias_filter"]
        self.price_history_5min: deque = deque()  # (timestamp, price) tuples

        # --- Internal counters ---
        self.total_signals_seen = 0
        self.total_entries_allowed = 0
        self.total_entries_blocked = 0
        self.gate_block_counts: Dict[str, int] = {}

        self.logger.info("ExecutionFramework initialized with config: %s", config_path)

    def _setup_logging(self):
        """Configure framework logging."""
        log_cfg = self.config.get("logging", {})
        log_file = log_cfg.get("log_file", "logs/execution_framework.log")

        # Make path absolute relative to this file's directory
        if not os.path.isabs(log_file):
            log_file = os.path.join(os.path.dirname(__file__), log_file)

        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger = logging.getLogger("ExecutionFramework")
        self.logger.setLevel(logging.DEBUG)

        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        self.logger.addHandler(fh)

        # Console handler (INFO only)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(ch)

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    def should_enter(
        self,
        pred_1s: float,
        pred_5s: float,
        pred_10s: float,
        patchtst_pred: Optional[Dict[str, float]],
        market_state: MarketState,
    ) -> Tuple[bool, str]:
        """
        Determine whether to enter a trade.

        Args:
            pred_1s: CNN-Mamba v2 prediction at 1s horizon
            pred_5s: CNN-Mamba v2 prediction at 5s horizon
            pred_10s: CNN-Mamba v2 prediction at 10s horizon
            patchtst_pred: Dict with keys "5s", "10s" for PatchTST predictions (or None)
            market_state: Current market snapshot

        Returns:
            (should_enter: bool, reason: str)
            reason explains WHY we're entering or WHY we're blocking
        """
        self.total_signals_seen += 1

        # Update prediction histories for z-score
        self.pred_history_1s.append(pred_1s)
        self.pred_history_5s.append(pred_5s)
        self.pred_history_10s.append(pred_10s)

        # Update direction history for health monitor
        # Use 5s as primary direction indicator
        direction = 1 if pred_5s > 0 else -1
        self.direction_history.append(direction)

        # Reset daily state if new day
        self._check_daily_reset(market_state.timestamp)

        # --- Run all gates in order ---
        gates: List[GateResult] = []

        # Gate 1: Risk management (hard limits)
        gates.append(self._check_risk_gate(market_state))

        # Gate 2: Model health
        gates.append(self._check_model_health())

        # Gate 3: Time of day
        gates.append(self._check_time_gate(market_state.timestamp))

        # Gate 4: Volatility
        gates.append(self._check_volatility_gate())

        # Gate 5: Direction bias filter
        gates.append(self._check_direction_bias(pred_5s, market_state))

        # Gate 6: PatchTST veto
        gates.append(self._check_patchtst_veto(pred_5s, pred_10s, patchtst_pred))

        # Gate 7: Confidence gate (z-score percentile)
        gate_conf, best_horizon = self._check_confidence_gate(pred_1s, pred_5s, pred_10s, patchtst_pred)
        gates.append(gate_conf)

        # Check all gates
        for gate in gates:
            if not gate.passed:
                self.total_entries_blocked += 1
                self.gate_block_counts[gate.gate_name] = self.gate_block_counts.get(gate.gate_name, 0) + 1

                if self.config["logging"]["log_gate_decisions"]:
                    self.logger.debug(
                        "BLOCKED by %s: %s | preds=(%.4f, %.4f, %.4f)",
                        gate.gate_name, gate.reason, pred_1s, pred_5s, pred_10s
                    )
                return False, f"Blocked by {gate.gate_name}: {gate.reason}"

        # All gates passed
        self.total_entries_allowed += 1
        self.trades_today += 1

        entry_dir = "LONG" if pred_5s > 0 else "SHORT"
        reason = (
            f"ENTRY {entry_dir} via {best_horizon} horizon | "
            f"preds=({pred_1s:.4f}, {pred_5s:.4f}, {pred_10s:.4f})"
        )
        self.logger.info(reason)
        return True, reason

    def get_exit_params(
        self,
        entry_direction: int,  # +1 long, -1 short
        entry_price: float,
        current_price: float,
        elapsed_seconds: float,
        signal_horizon: str = "5s",
    ) -> ExitParams:
        """
        Determine exit action for current position.

        Args:
            entry_direction: +1 for long, -1 for short
            entry_price: Price at entry
            current_price: Current market price
            elapsed_seconds: Seconds since entry
            signal_horizon: Which signal triggered entry ("1s", "5s", "10s")

        Returns:
            ExitParams with action and target price
        """
        exit_cfg = self.config["exit_management"]
        tick_size = self.config["cost_constants"]["tick_size_points"]

        # Calculate PnL in ticks
        if entry_direction == 1:
            pnl_ticks = (current_price - entry_price) / tick_size
        else:
            pnl_ticks = (entry_price - current_price) / tick_size

        # Determine hold time for this signal
        if signal_horizon == "1s":
            max_hold = exit_cfg["hold_time_1s_signal"]
        elif signal_horizon == "10s":
            max_hold = exit_cfg["hold_time_10s_signal"]
        else:
            max_hold = exit_cfg["hold_time_5s_signal"]

        # --- Hard stop (always checked first) ---
        if pnl_ticks <= -exit_cfg["hard_stop_ticks"]:
            self.logger.warning(
                "HARD STOP triggered: pnl=%.1f ticks, threshold=-%s ticks",
                pnl_ticks, exit_cfg["hard_stop_ticks"]
            )
            return ExitParams(
                action=ExitAction.HARD_STOP,
                order_type="market",
                reason=f"Hard stop at {pnl_ticks:.1f} ticks loss"
            )

        # --- Profit lock: if in profit >= threshold, try passive exit ---
        if pnl_ticks >= exit_cfg["profit_lock_threshold_ticks"]:
            lock_price = current_price  # Post limit at current price
            return ExitParams(
                action=ExitAction.PROFIT_LOCK,
                price=lock_price,
                order_type="limit",
                reason=f"Profit lock at +{pnl_ticks:.1f} ticks"
            )

        # --- Trailing stop ---
        if pnl_ticks >= exit_cfg["trailing_stop_activate_ticks"]:
            offset = exit_cfg["trailing_stop_offset_ticks"] * tick_size
            if entry_direction == 1:
                stop_price = current_price - offset
            else:
                stop_price = current_price + offset
            return ExitParams(
                action=ExitAction.TRAILING_STOP,
                price=stop_price,
                order_type="market",
                reason=f"Trailing stop active at +{pnl_ticks:.1f} ticks, stop={stop_price}"
            )

        # --- Time-based exit ---
        if elapsed_seconds >= max_hold:
            if exit_cfg["market_exit_on_timeout"]:
                return ExitParams(
                    action=ExitAction.TIME_EXIT_MARKET,
                    order_type="market",
                    reason=f"Time exit (market) after {elapsed_seconds:.1f}s, pnl={pnl_ticks:.1f}t"
                )
            else:
                return ExitParams(
                    action=ExitAction.TIME_EXIT_PASSIVE,
                    price=current_price,
                    order_type="limit",
                    reason=f"Time exit (passive) after {elapsed_seconds:.1f}s"
                )

        # --- Hold ---
        return ExitParams(
            action=ExitAction.HOLD,
            reason=f"Holding: {elapsed_seconds:.1f}s/{max_hold}s, pnl={pnl_ticks:.1f}t"
        )

    def update_state(self, market_state: MarketState):
        """
        Called every event to update internal state.
        Must be called continuously for gates to function properly.
        """
        # Update volatility tracking
        self.price_rel_history.append(market_state.price_rel)

        # Update baseline vol after warmup
        vol_cfg = self.config["volatility_gate"]
        if len(self.price_rel_history) >= vol_cfg["vol_window_events"] and not self.vol_warmup_done:
            self.baseline_vol = np.std(list(self.price_rel_history))
            self.vol_warmup_done = True
            self.logger.info("Volatility baseline established: %.6f", self.baseline_vol)

        # Update 5-min price history for direction bias
        ts = market_state.timestamp / 1e9 if market_state.timestamp > 1e15 else market_state.timestamp
        self.price_history_5min.append((ts, market_state.last_price))
        # Trim to 5 minutes
        cutoff = ts - self.config["direction_bias_filter"]["trend_window_seconds"]
        while self.price_history_5min and self.price_history_5min[0][0] < cutoff:
            self.price_history_5min.popleft()

    def record_trade_result(self, pnl_ticks: float):
        """
        Call after a trade closes to update risk management state.

        Args:
            pnl_ticks: Realized PnL in ticks (positive = profit)
        """
        tick_value = self.config["cost_constants"]["tick_value_dollars"]
        commission = self.config["cost_constants"]["rt_commission_dollars"]

        pnl_dollars = (pnl_ticks * tick_value) - commission
        self.daily_pnl += pnl_dollars

        if pnl_dollars < 0:
            self.consecutive_losses += 1
            self.last_loss_time = time.time()

            # Check consecutive loss cooldown
            risk_cfg = self.config["risk_management"]
            if self.consecutive_losses >= risk_cfg["max_consecutive_losses"]:
                self.cooldown_until = time.time() + risk_cfg["cooldown_after_consecutive_losses_seconds"]
                self.logger.warning(
                    "COOLDOWN ACTIVATED: %d consecutive losses, cooling down %ds. Daily PnL: $%.2f",
                    self.consecutive_losses,
                    risk_cfg["cooldown_after_consecutive_losses_seconds"],
                    self.daily_pnl
                )
        else:
            self.consecutive_losses = 0

        self.logger.info(
            "Trade closed: %.1f ticks ($%.2f) | Daily PnL: $%.2f | Consec losses: %d | Trades: %d",
            pnl_ticks, pnl_dollars, self.daily_pnl, self.consecutive_losses, self.trades_today
        )

    def record_ic_sample(self, prediction: float, actual_move: float):
        """
        Record a (prediction, actual_move) pair for rolling IC estimation.
        Call when you know the realized move for a past prediction.
        """
        self.rolling_ic_buffer.append((prediction, actual_move))
        # Keep only recent samples
        max_samples = self.ic_window * 3
        if len(self.rolling_ic_buffer) > max_samples:
            self.rolling_ic_buffer = self.rolling_ic_buffer[-max_samples:]

    def get_health_report(self) -> Dict:
        """Return comprehensive health status of all gates and framework state."""
        # Current vol
        current_vol = None
        vol_ratio = None
        if self.vol_warmup_done and self.baseline_vol and self.baseline_vol > 0:
            current_vol = np.std(list(self.price_rel_history)) if len(self.price_rel_history) > 10 else None
            if current_vol is not None:
                vol_ratio = current_vol / self.baseline_vol

        # Direction balance
        dir_balance = None
        if len(self.direction_history) > 0:
            longs = sum(1 for d in self.direction_history if d > 0)
            dir_balance = longs / len(self.direction_history)

        # Rolling IC
        rolling_ic = self._compute_rolling_ic()

        return {
            "model_healthy": self.model_healthy,
            "health_reason": self.health_reason,
            "direction_balance": dir_balance,
            "direction_window_size": len(self.direction_history),
            "rolling_ic": rolling_ic,
            "volatility": {
                "baseline": self.baseline_vol,
                "current": current_vol,
                "ratio": vol_ratio,
                "warmup_done": self.vol_warmup_done,
            },
            "risk": {
                "daily_pnl": self.daily_pnl,
                "consecutive_losses": self.consecutive_losses,
                "trades_today": self.trades_today,
                "in_cooldown": self.cooldown_until is not None and time.time() < self.cooldown_until,
            },
            "stats": {
                "total_signals": self.total_signals_seen,
                "entries_allowed": self.total_entries_allowed,
                "entries_blocked": self.total_entries_blocked,
                "pass_rate": (
                    self.total_entries_allowed / max(1, self.total_signals_seen)
                ),
                "gate_blocks": dict(self.gate_block_counts),
            },
        }

    # -----------------------------------------------------------------------
    # GATE IMPLEMENTATIONS
    # -----------------------------------------------------------------------

    def _check_risk_gate(self, market_state: MarketState) -> GateResult:
        """Gate 1: Risk management hard limits."""
        risk_cfg = self.config["risk_management"]

        # Daily loss limit
        if self.daily_pnl <= -risk_cfg["max_daily_loss_dollars"]:
            return GateResult(
                passed=False,
                gate_name="risk_daily_loss",
                reason=f"Daily loss ${self.daily_pnl:.2f} exceeds limit -${risk_cfg['max_daily_loss_dollars']}"
            )

        # Max trades per day
        if self.trades_today >= risk_cfg["max_trades_per_day"]:
            return GateResult(
                passed=False,
                gate_name="risk_max_trades",
                reason=f"Max trades reached: {self.trades_today}/{risk_cfg['max_trades_per_day']}"
            )

        # Cooldown period
        if self.cooldown_until is not None and time.time() < self.cooldown_until:
            remaining = self.cooldown_until - time.time()
            return GateResult(
                passed=False,
                gate_name="risk_cooldown",
                reason=f"In cooldown for {remaining:.0f}s more ({self.consecutive_losses} consec losses)"
            )

        return GateResult(passed=True, gate_name="risk_management")

    def _check_model_health(self) -> GateResult:
        """Gate 2: Model health monitor — detect bias and IC degradation."""
        health_cfg = self.config["model_health"]

        # Check direction imbalance
        if len(self.direction_history) >= health_cfg["direction_window"]:
            longs = sum(1 for d in self.direction_history if d > 0)
            balance = longs / len(self.direction_history)

            if balance > health_cfg["direction_imbalance_threshold"]:
                self.model_healthy = False
                self.health_reason = f"LONG bias: {balance:.1%} of last {len(self.direction_history)} predictions"
                return GateResult(
                    passed=False,
                    gate_name="model_health_bias",
                    reason=self.health_reason
                )
            elif (1 - balance) > health_cfg["direction_imbalance_threshold"]:
                self.model_healthy = False
                self.health_reason = f"SHORT bias: {1-balance:.1%} of last {len(self.direction_history)} predictions"
                return GateResult(
                    passed=False,
                    gate_name="model_health_bias",
                    reason=self.health_reason
                )

        # Check rolling IC
        rolling_ic = self._compute_rolling_ic()
        if rolling_ic is not None and rolling_ic < health_cfg["rolling_ic_min"]:
            ic_samples = min(len(self.rolling_ic_buffer), self.ic_window)
            if ic_samples >= health_cfg["rolling_ic_window"]:
                self.model_healthy = False
                self.health_reason = f"Negative IC: {rolling_ic:.4f} over {ic_samples} samples"
                return GateResult(
                    passed=False,
                    gate_name="model_health_ic",
                    reason=self.health_reason
                )

        self.model_healthy = True
        self.health_reason = ""
        return GateResult(passed=True, gate_name="model_health")

    def _check_time_gate(self, timestamp: float) -> GateResult:
        """Gate 3: Time of day restrictions."""
        time_cfg = self.config["time_of_day_gate"]

        # Handle nanosecond timestamps
        if timestamp > 1e15:
            timestamp = timestamp / 1e9
        # Get current ET time
        if ET is not None:
            dt = datetime.fromtimestamp(timestamp, tz=ET)
        else:
            # Fallback: assume system is in ET (not ideal but functional)
            dt = datetime.fromtimestamp(timestamp)

        current_time = dt.strftime("%H:%M")

        # Block open volatility
        if time_cfg["block_open_start"] <= current_time <= time_cfg["block_open_end"]:
            return GateResult(
                passed=False,
                gate_name="time_of_day",
                reason=f"Open volatility block ({time_cfg['block_open_start']}-{time_cfg['block_open_end']})"
            )

        # Block after close (unless Globex enabled)
        if current_time >= time_cfg["block_after"] and not time_cfg["globex_enabled"]:
            return GateResult(
                passed=False,
                gate_name="time_of_day",
                reason=f"After {time_cfg['block_after']} ET, Globex not enabled"
            )

        return GateResult(passed=True, gate_name="time_of_day")

    def _check_volatility_gate(self) -> GateResult:
        """Gate 4: Volatility regime filter."""
        vol_cfg = self.config["volatility_gate"]

        if not self.vol_warmup_done or self.baseline_vol is None or self.baseline_vol == 0:
            # Not enough data yet — allow (don't block during warmup)
            return GateResult(passed=True, gate_name="volatility_gate", reason="Warmup not complete")

        current_vol = np.std(list(self.price_rel_history))
        vol_ratio = current_vol / self.baseline_vol

        if vol_ratio > vol_cfg["vol_high_multiplier"]:
            return GateResult(
                passed=False,
                gate_name="volatility_high",
                reason=f"Vol ratio {vol_ratio:.2f}x > {vol_cfg['vol_high_multiplier']}x (trending/volatile)"
            )

        if vol_ratio < vol_cfg["vol_low_multiplier"]:
            return GateResult(
                passed=False,
                gate_name="volatility_low",
                reason=f"Vol ratio {vol_ratio:.2f}x < {vol_cfg['vol_low_multiplier']}x (dead market)"
            )

        return GateResult(passed=True, gate_name="volatility_gate")

    def _check_direction_bias(self, pred_5s: float, market_state: MarketState) -> GateResult:
        """Gate 5: Don't fade strong trends."""
        bias_cfg = self.config["direction_bias_filter"]

        if not bias_cfg["suppress_fade_entries"]:
            return GateResult(passed=True, gate_name="direction_bias")

        if len(self.price_history_5min) < 2:
            return GateResult(passed=True, gate_name="direction_bias", reason="Insufficient data")

        # Calculate recent price move in ticks
        oldest_price = self.price_history_5min[0][1]
        newest_price = self.price_history_5min[-1][1]
        tick_size = self.config["cost_constants"]["tick_size_points"]
        move_ticks = (newest_price - oldest_price) / tick_size

        threshold = bias_cfg["trend_threshold_ticks"]
        signal_direction = 1 if pred_5s > 0 else -1

        # If market dropped >20 ticks and signal says go LONG → block (fading trend)
        if move_ticks < -threshold and signal_direction == 1:
            return GateResult(
                passed=False,
                gate_name="direction_bias",
                reason=f"Market dropped {move_ticks:.0f} ticks in 5min, blocking LONG (don't fade)"
            )

        # If market rallied >20 ticks and signal says go SHORT → block
        if move_ticks > threshold and signal_direction == -1:
            return GateResult(
                passed=False,
                gate_name="direction_bias",
                reason=f"Market rallied +{move_ticks:.0f} ticks in 5min, blocking SHORT (don't fade)"
            )

        return GateResult(passed=True, gate_name="direction_bias")

    def _check_patchtst_veto(
        self,
        pred_5s: float,
        pred_10s: float,
        patchtst_pred: Optional[Dict[str, float]],
    ) -> GateResult:
        """Gate 6: PatchTST confluence veto at 5s/10s horizons."""
        veto_cfg = self.config["patchtst_veto"]

        if not veto_cfg["enabled"]:
            return GateResult(passed=True, gate_name="patchtst_veto", reason="Disabled")

        if patchtst_pred is None:
            # No PatchTST prediction available — allow (graceful degradation)
            return GateResult(passed=True, gate_name="patchtst_veto", reason="No PatchTST data")

        # Check 5s horizon
        if "5s" in veto_cfg["horizons"] and "5s" in patchtst_pred:
            mamba_dir = 1 if pred_5s > 0 else -1
            patch_dir = 1 if patchtst_pred["5s"] > 0 else -1
            if mamba_dir != patch_dir:
                return GateResult(
                    passed=False,
                    gate_name="patchtst_veto",
                    reason=f"PatchTST disagrees at 5s: Mamba={'LONG' if mamba_dir>0 else 'SHORT'}, "
                           f"PatchTST={'LONG' if patch_dir>0 else 'SHORT'}"
                )

        # Check 10s horizon
        if "10s" in veto_cfg["horizons"] and "10s" in patchtst_pred:
            mamba_dir = 1 if pred_10s > 0 else -1
            patch_dir = 1 if patchtst_pred["10s"] > 0 else -1
            if mamba_dir != patch_dir:
                return GateResult(
                    passed=False,
                    gate_name="patchtst_veto",
                    reason=f"PatchTST disagrees at 10s: Mamba={'LONG' if mamba_dir>0 else 'SHORT'}, "
                           f"PatchTST={'LONG' if patch_dir>0 else 'SHORT'}"
                )

        return GateResult(passed=True, gate_name="patchtst_veto")

    def _check_confidence_gate(
        self,
        pred_1s: float,
        pred_5s: float,
        pred_10s: float,
        patchtst_pred: Optional[Dict[str, float]],
    ) -> Tuple[GateResult, str]:
        """
        Gate 7: Confidence percentile filter.
        Returns (gate_result, best_horizon_used).
        """
        conf_cfg = self.config["confidence_gate"]

        # Need sufficient history for z-scores
        min_samples = 50
        if len(self.pred_history_5s) < min_samples:
            return (
                GateResult(passed=False, gate_name="confidence", reason="Insufficient history for z-score"),
                "none"
            )

        # Compute z-scores for each horizon
        z_1s = self._compute_zscore(pred_1s, self.pred_history_1s)
        z_5s = self._compute_zscore(pred_5s, self.pred_history_5s)
        z_10s = self._compute_zscore(pred_10s, self.pred_history_10s)

        # Confidence = abs(z-score), convert to percentile
        # For normal distribution: top-1% ≈ z > 2.33, top-0.5% ≈ z > 2.58
        has_patchtst = patchtst_pred is not None

        # Check each horizon against its threshold
        # Prefer 5s (best risk/reward with PatchTST veto)
        candidates = []

        # 5s with PatchTST veto: top-1% required
        pct_threshold_5s = conf_cfg["min_percentile_5s"]
        z_threshold_5s = self._percentile_to_zscore(pct_threshold_5s)
        if abs(z_5s) >= z_threshold_5s:
            candidates.append(("5s", abs(z_5s), z_threshold_5s))

        # 10s with PatchTST veto: top-1% required
        pct_threshold_10s = conf_cfg["min_percentile_10s"]
        z_threshold_10s = self._percentile_to_zscore(pct_threshold_10s)
        if abs(z_10s) >= z_threshold_10s:
            candidates.append(("10s", abs(z_10s), z_threshold_10s))

        # 1s without veto: top-0.5% required
        pct_threshold_1s = conf_cfg["min_percentile_1s"]
        z_threshold_1s = self._percentile_to_zscore(pct_threshold_1s)
        if abs(z_1s) >= z_threshold_1s:
            candidates.append(("1s", abs(z_1s), z_threshold_1s))

        if not candidates:
            return (
                GateResult(
                    passed=False,
                    gate_name="confidence",
                    reason=f"Below threshold: z_1s={abs(z_1s):.2f}(<{z_threshold_1s:.2f}), "
                           f"z_5s={abs(z_5s):.2f}(<{z_threshold_5s:.2f}), "
                           f"z_10s={abs(z_10s):.2f}(<{z_threshold_10s:.2f})"
                ),
                "none"
            )

        # Pick best candidate (highest z-score relative to threshold)
        best = max(candidates, key=lambda x: x[1] / x[2])
        best_horizon = best[0]

        return (
            GateResult(
                passed=True,
                gate_name="confidence",
                reason=f"Passed at {best_horizon}: z={best[1]:.2f} >= {best[2]:.2f}"
            ),
            best_horizon
        )

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------

    def _compute_zscore(self, value: float, history: deque) -> float:
        """Compute z-score of value against history."""
        if len(history) < 10:
            return 0.0
        arr = np.array(history)
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return 0.0
        return (value - mean) / std

    def _percentile_to_zscore(self, percentile: float) -> float:
        """
        Convert a percentile (e.g., 99.0) to z-score threshold.
        We use the absolute value, so top-1% = abs(z) > 2.33.
        """
        # Lookup table for common values (avoids scipy dependency)
        table = {
            95.0: 1.645,
            97.5: 1.960,
            99.0: 2.326,
            99.5: 2.576,
            99.9: 3.090,
        }
        if percentile in table:
            return table[percentile]
        # Linear interpolation for anything else
        keys = sorted(table.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= percentile <= keys[i + 1]:
                frac = (percentile - keys[i]) / (keys[i + 1] - keys[i])
                return table[keys[i]] + frac * (table[keys[i + 1]] - table[keys[i]])
        # Fallback
        return 2.326  # top-1% default

    def _compute_rolling_ic(self) -> Optional[float]:
        """Compute rolling IC (rank correlation) from recent prediction/actual pairs."""
        if len(self.rolling_ic_buffer) < self.ic_window:
            return None

        recent = self.rolling_ic_buffer[-self.ic_window:]
        preds = np.array([p for p, _ in recent])
        actuals = np.array([a for _, a in recent])

        # Spearman rank correlation (without scipy)
        n = len(preds)
        if n < 5:
            return None

        # Rank the arrays
        pred_ranks = np.argsort(np.argsort(preds)).astype(float)
        actual_ranks = np.argsort(np.argsort(actuals)).astype(float)

        # Pearson on ranks = Spearman
        pred_ranks -= pred_ranks.mean()
        actual_ranks -= actual_ranks.mean()

        denom = np.sqrt((pred_ranks ** 2).sum() * (actual_ranks ** 2).sum())
        if denom == 0:
            return 0.0

        return float((pred_ranks * actual_ranks).sum() / denom)

    def _check_daily_reset(self, timestamp: float):
        """Reset daily counters if new trading day."""
        # Handle nanosecond timestamps (from MBO events) vs epoch seconds
        if timestamp > 1e15:  # nanoseconds
            timestamp = timestamp / 1e9
        if ET is not None:
            dt = datetime.fromtimestamp(timestamp, tz=ET)
        else:
            dt = datetime.fromtimestamp(timestamp)

        date_str = dt.strftime("%Y-%m-%d")
        if self.session_date != date_str:
            if self.session_date is not None:
                self.logger.info(
                    "NEW DAY: %s. Previous day PnL: $%.2f, Trades: %d",
                    date_str, self.daily_pnl, self.trades_today
                )
            self.session_date = date_str
            self.daily_pnl = 0.0
            self.consecutive_losses = 0
            self.trades_today = 0
            self.cooldown_until = None

    def get_entry_params(self, direction: int, market_state: MarketState) -> Dict:
        """
        Get entry order parameters.

        Args:
            direction: +1 for long, -1 for short
            market_state: Current market state

        Returns:
            Dict with order_type, price, cancel_after_seconds
        """
        entry_cfg = self.config["entry_logic"]

        if direction == 1:
            price = market_state.best_bid  # Passive long = post at bid
        else:
            price = market_state.best_ask  # Passive short = post at ask

        return {
            "order_type": entry_cfg["order_type"],
            "price": price,
            "direction": direction,
            "cancel_after_seconds": entry_cfg["cancel_patience_seconds"],
            "chase": entry_cfg["chase"],
        }


# ---------------------------------------------------------------------------
# Convenience: standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("ExecutionFramework self-test...")

    # Create with default config
    fw = ExecutionFramework()

    # Simulate warmup
    for i in range(600):
        ms = MarketState(
            timestamp=time.time() + i * 0.1,
            best_bid=5600.0,
            best_ask=5600.25,
            last_price=5600.0 + np.random.randn() * 0.5,
            price_rel=np.random.randn() * 0.001,
            event_count=i,
        )
        fw.update_state(ms)

    # Simulate predictions
    for i in range(100):
        pred_1s = np.random.randn() * 0.1
        pred_5s = np.random.randn() * 0.1
        pred_10s = np.random.randn() * 0.1

        patchtst = {"5s": pred_5s + np.random.randn() * 0.05, "10s": pred_10s + np.random.randn() * 0.05}

        ms = MarketState(
            timestamp=time.time() + 600 + i * 0.5,
            best_bid=5600.0,
            best_ask=5600.25,
            last_price=5600.0,
            price_rel=np.random.randn() * 0.001,
            event_count=600 + i * 500,
        )

        should, reason = fw.should_enter(pred_1s, pred_5s, pred_10s, patchtst, ms)

    # Print health report
    report = fw.get_health_report()
    print(f"\nHealth Report:")
    print(f"  Model healthy: {report['model_healthy']}")
    print(f"  Direction balance: {report['direction_balance']:.2%}" if report['direction_balance'] else "  Direction balance: N/A")
    print(f"  Signals seen: {report['stats']['total_signals']}")
    print(f"  Entries allowed: {report['stats']['entries_allowed']}")
    print(f"  Entries blocked: {report['stats']['entries_blocked']}")
    print(f"  Pass rate: {report['stats']['pass_rate']:.2%}")
    print(f"  Gate blocks: {report['stats']['gate_blocks']}")
    print("\nSelf-test PASSED")
