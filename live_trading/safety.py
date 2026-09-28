"""
Safety Mechanisms for Live Trading
===================================

Finance-grade robustness with multiple layers of protection.
Every order validated, every failure mode handled.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time as dtime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Order representation"""
    card_name: str
    side: str  # "BUY" or "SELL"
    size: int
    symbol: str
    price: float
    order_type: str  # "MARKET" or "LIMIT"
    timestamp: float


@dataclass
class Position:
    """Position representation"""
    card_name: str
    side: str
    size: int
    entry_price: float
    entry_time: float
    unrealized_pnl: float

    @property
    def hold_time(self) -> float:
        return time.time() - self.entry_time


class ModelHealthCheck:
    """Verify model is producing valid predictions"""

    def __init__(self, max_prediction_value: float = 10.0, max_latency_ms: float = 50.0):
        self.max_prediction_value = max_prediction_value
        self.max_latency_ms = max_latency_ms
        self.recent_predictions = []
        self.max_recent = 100

    def validate_prediction(self, pred) -> Tuple[bool, str]:
        """Validate a prediction before using it"""
        # 1. Prediction in reasonable range?
        if abs(pred.value) > self.max_prediction_value:
            return False, f"Prediction out of range: {pred.value:.2f} (max {self.max_prediction_value})"

        # 2. Model latency acceptable?
        if pred.latency_ms > self.max_latency_ms:
            return False, f"Model latency too high: {pred.latency_ms:.1f}ms (max {self.max_latency_ms})"

        # 3. Not NaN or Inf?
        if not np.isfinite(pred.value):
            return False, "Prediction is NaN or Inf"

        # 4. Track recent predictions
        self.recent_predictions.append(pred.value)
        if len(self.recent_predictions) > self.max_recent:
            self.recent_predictions.pop(0)

        # 5. Model frozen? (all predictions identical)
        if len(self.recent_predictions) >= 10:
            if len(set(self.recent_predictions[-10:])) == 1:
                return False, "Model appears frozen (identical predictions)"

        return True, "OK"


class OrderValidator:
    """Validate every order before submission"""

    def __init__(
        self,
        max_position_per_card: int = 1,
        max_total_position: int = 5,
        max_daily_trades: int = 200,
        max_daily_loss: float = 500.0,
        allowed_symbols: List[str] = None,
        trading_hours_start: dtime = dtime(9, 30),
        trading_hours_end: dtime = dtime(16, 0),
    ):
        self.max_position_per_card = max_position_per_card
        self.max_total_position = max_total_position
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss = max_daily_loss
        self.allowed_symbols = allowed_symbols or ["ES"]
        self.trading_hours_start = trading_hours_start
        self.trading_hours_end = trading_hours_end

        # State
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.positions: Dict[str, Position] = {}
        self.recent_orders: List[Order] = []
        self.last_reset = datetime.now().date()

    def reset_daily_counters(self):
        """Reset daily counters at start of new trading day"""
        today = datetime.now().date()
        if today != self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_reset = today
            logger.info("Daily counters reset")

    def validate_order(self, order: Order, current_positions: Dict[str, Position]) -> Tuple[bool, str]:
        """Validate order before submission"""
        self.reset_daily_counters()
        self.positions = current_positions

        # 1. Symbol whitelist
        if order.symbol not in self.allowed_symbols:
            return False, f"Symbol {order.symbol} not in whitelist"

        # 2. Market hours check
        if not self._in_trading_hours():
            return False, "Outside trading hours"

        # 3. Position limits
        if order.side == "BUY":
            if not self._check_position_limits(order.card_name, order.size):
                return False, "Position limit would be exceeded"

        # 4. Daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit reached ({self.max_daily_trades})"

        # 5. Daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return False, f"Daily loss limit reached (${self.max_daily_loss})"

        # 6. Price sanity check
        if not self._price_reasonable(order.price, order.symbol):
            return False, f"Price unreasonable: {order.price}"

        # 7. Duplicate order detection
        if self._is_duplicate(order):
            return False, "Duplicate order detected (submitted <1s ago)"

        # 8. Size validation
        if order.size <= 0:
            return False, "Invalid order size"

        # Record order
        self.recent_orders.append(order)
        if len(self.recent_orders) > 100:
            self.recent_orders.pop(0)

        return True, "OK"

    def _in_trading_hours(self) -> bool:
        """Check if currently in trading hours"""
        now = datetime.now().time()
        return self.trading_hours_start <= now <= self.trading_hours_end

    def _check_position_limits(self, card_name: str, additional_size: int) -> bool:
        """Check if adding position would exceed limits"""
        # Current position for this card
        current_card_pos = self.positions.get(card_name)
        card_size = current_card_pos.size if current_card_pos else 0

        # Total position across all cards
        total_size = sum(p.size for p in self.positions.values())

        # Check limits
        if card_size + additional_size > self.max_position_per_card:
            return False
        if total_size + additional_size > self.max_total_position:
            return False

        return True

    def _price_reasonable(self, price: float, symbol: str) -> bool:
        """Sanity check on price"""
        # ES futures typically 4000-7000
        if symbol == "ES":
            return 3000 < price < 10000
        # Default: any positive price
        return price > 0

    def _is_duplicate(self, order: Order) -> bool:
        """Check if this order was submitted very recently"""
        if not self.recent_orders:
            return False

        # Check last 10 orders
        recent = self.recent_orders[-10:]
        for prev_order in recent:
            # Same card, side, within 1 second?
            if (prev_order.card_name == order.card_name and
                prev_order.side == order.side and
                prev_order.symbol == order.symbol and
                order.timestamp - prev_order.timestamp < 1.0):
                return True

        return False


class RiskMonitor:
    """Monitor risk in real-time, force-close positions if needed"""

    def __init__(
        self,
        max_position_hold_seconds: int = 3600,
        max_unrealized_loss_per_position: float = 500.0,
        max_account_drawdown_pct: float = 0.02,
        tick_value: float = 12.50,  # ES futures
    ):
        self.max_position_hold_seconds = max_position_hold_seconds
        self.max_unrealized_loss = max_unrealized_loss_per_position
        self.max_account_drawdown_pct = max_account_drawdown_pct
        self.tick_value = tick_value

        self.starting_balance = 10000.0  # Track for drawdown
        self.current_balance = 10000.0

    def check_positions(self, positions: Dict[str, Position]) -> List[Tuple[str, str]]:
        """Check positions and return list of (card_name, reason) to force close"""
        to_close = []

        for card_name, pos in positions.items():
            # 1. Position held too long?
            if pos.hold_time > self.max_position_hold_seconds:
                to_close.append((card_name, f"Max hold time exceeded ({pos.hold_time:.0f}s)"))
                continue

            # 2. Unrealized loss too large?
            unrealized_loss_dollars = abs(pos.unrealized_pnl) * self.tick_value
            if unrealized_loss_dollars > self.max_unrealized_loss:
                to_close.append((card_name, f"Max loss exceeded (${unrealized_loss_dollars:.2f})"))
                continue

        # 3. Account drawdown too large?
        if self._account_drawdown() > self.max_account_drawdown_pct:
            # Close everything
            for card_name in positions.keys():
                to_close.append((card_name, "Account drawdown limit exceeded"))

        return to_close

    def _account_drawdown(self) -> float:
        """Calculate current drawdown as fraction of starting balance"""
        return max(0, (self.starting_balance - self.current_balance) / self.starting_balance)

    def update_balance(self, realized_pnl_ticks: float):
        """Update balance after trade closes"""
        self.current_balance += realized_pnl_ticks * self.tick_value


class CircuitBreaker:
    """Automatic trading halts on anomalous conditions"""

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        max_hourly_trades: int = 50,
        min_win_rate_threshold: float = 0.35,
        max_slippage_ticks: float = 2.0,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.max_hourly_trades = max_hourly_trades
        self.min_win_rate_threshold = min_win_rate_threshold
        self.max_slippage_ticks = max_slippage_ticks

        # State
        self.consecutive_losses = 0
        self.recent_trades: List[dict] = []
        self.halted = False
        self.halt_reason = None

    def record_trade(self, pnl: float, entry_price: float, fill_price: float):
        """Record a completed trade"""
        self.recent_trades.append({
            'timestamp': time.time(),
            'pnl': pnl,
            'slippage': abs(fill_price - entry_price)
        })

        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Trim old trades (keep last 100)
        if len(self.recent_trades) > 100:
            self.recent_trades.pop(0)

    def check_circuit_breakers(self) -> Tuple[bool, Optional[str]]:
        """Check if any circuit breaker should trigger"""
        if self.halted:
            return True, self.halt_reason

        # 1. Too many consecutive losses?
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.halt("Circuit Breaker: Consecutive losses")
            return True, self.halt_reason

        # 2. Trading too fast?
        recent_hour = [t for t in self.recent_trades if time.time() - t['timestamp'] < 3600]
        if len(recent_hour) > self.max_hourly_trades:
            self.halt("Circuit Breaker: Trade rate too high")
            return True, self.halt_reason

        # 3. Win rate collapsed?
        if len(self.recent_trades) >= 20:
            recent_20 = self.recent_trades[-20:]
            wins = sum(1 for t in recent_20 if t['pnl'] > 0)
            win_rate = wins / len(recent_20)
            if win_rate < self.min_win_rate_threshold:
                self.halt(f"Circuit Breaker: Win rate {win_rate:.1%} below threshold")
                return True, self.halt_reason

        # 4. Fill quality degraded?
        if len(recent_hour) >= 10:
            avg_slippage = np.mean([t['slippage'] for t in recent_hour])
            if avg_slippage > self.max_slippage_ticks:
                self.halt(f"Circuit Breaker: Slippage {avg_slippage:.2f} ticks too high")
                return True, self.halt_reason

        return False, None

    def halt(self, reason: str):
        """Halt trading"""
        self.halted = True
        self.halt_reason = reason
        logger.critical(f"CIRCUIT BREAKER TRIGGERED: {reason}")

    def reset(self):
        """Reset circuit breaker (manual action required)"""
        self.halted = False
        self.halt_reason = None
        self.consecutive_losses = 0
        logger.warning("Circuit breaker manually reset")


class KillSwitch:
    """Emergency stop mechanism"""

    def __init__(self, kill_file_path: str = "/tmp/trading_kill_switch"):
        self.kill_file = Path(kill_file_path)
        self.enabled = True

    def check(self) -> Tuple[bool, Optional[str]]:
        """Check if kill switch activated"""
        # Kill file exists?
        if self.kill_file.exists():
            reason = self.kill_file.read_text().strip() if self.kill_file.stat().st_size > 0 else "Kill file detected"
            return True, reason

        return False, None

    def activate(self, reason: str = "Manual activation"):
        """Activate kill switch"""
        self.kill_file.write_text(reason)
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate(self):
        """Deactivate kill switch"""
        if self.kill_file.exists():
            self.kill_file.unlink()
        logger.warning("Kill switch deactivated")


class SafetyManager:
    """Unified safety management"""

    def __init__(self, config: dict):
        self.model_health = ModelHealthCheck(
            max_prediction_value=config.get('max_prediction_value', 10.0),
            max_latency_ms=config.get('max_latency_ms', 50.0),
        )

        self.order_validator = OrderValidator(
            max_position_per_card=config.get('max_card_position', 1),
            max_total_position=config.get('max_total_position', 5),
            max_daily_trades=config.get('max_daily_trades', 200),
            max_daily_loss=config.get('max_daily_loss_dollars', 500.0),
            allowed_symbols=config.get('allowed_symbols', ["ES"]),
        )

        self.risk_monitor = RiskMonitor(
            max_position_hold_seconds=config.get('max_position_hold_seconds', 3600),
            max_unrealized_loss_per_position=config.get('max_unrealized_loss', 500.0),
        )

        self.circuit_breaker = CircuitBreaker(
            max_consecutive_losses=config.get('max_consecutive_losses', 5),
            max_hourly_trades=config.get('max_hourly_trades', 50),
        )

        self.kill_switch = KillSwitch()

        logger.info("Safety manager initialized with all protection layers")

    def validate_prediction(self, pred) -> Tuple[bool, str]:
        """Validate prediction before using"""
        return self.model_health.validate_prediction(pred)

    def validate_order(self, order: Order, positions: Dict[str, Position]) -> Tuple[bool, str]:
        """Validate order before submission"""
        return self.order_validator.validate_order(order, positions)

    def check_positions(self, positions: Dict[str, Position]) -> List[Tuple[str, str]]:
        """Check positions for risk violations"""
        return self.risk_monitor.check_positions(positions)

    def record_trade(self, pnl: float, entry_price: float, fill_price: float):
        """Record completed trade"""
        self.circuit_breaker.record_trade(pnl, entry_price, fill_price)
        self.risk_monitor.update_balance(pnl)

    def should_halt(self) -> Tuple[bool, Optional[str]]:
        """Check if trading should halt"""
        # Kill switch?
        kill_active, kill_reason = self.kill_switch.check()
        if kill_active:
            return True, f"Kill Switch: {kill_reason}"

        # Circuit breakers?
        breaker_active, breaker_reason = self.circuit_breaker.check_circuit_breakers()
        if breaker_active:
            return True, breaker_reason

        return False, None
