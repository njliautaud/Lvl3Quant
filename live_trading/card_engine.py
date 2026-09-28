"""
Trading Card Engine
===================

Executes trading strategies (cards) based on model predictions.

A "card" is a self-contained trading strategy with:
- Entry rules (threshold, tier, filters)
- Position sizing
- Exit rules (TP/SL, time-based, conviction decay)
- Risk parameters

Supports:
- Multiple concurrent cards
- Card prioritization
- Card-level PnL tracking
- Dynamic enable/disable
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, List
from enum import Enum
import numpy as np

from live_trading.inference_engine import Prediction

logger = logging.getLogger(__name__)


class CardState(Enum):
    """Card execution state"""
    INACTIVE = "inactive"  # No position
    ACTIVE = "active"      # Position open
    PENDING = "pending"    # Waiting for fill
    DISABLED = "disabled"  # Card disabled


class Side(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class CardConfig:
    """Trading card configuration"""
    name: str
    model_name: str

    # Entry rules
    threshold: float = 0.0  # Minimum |prediction| to enter
    min_tier: str = "all"   # Tier filter: "all", "top50", "top25", "top10"
    max_position_size: int = 1  # Max contracts

    # Exit rules
    take_profit_ticks: Optional[float] = None
    stop_loss_ticks: Optional[float] = None
    max_hold_seconds: Optional[float] = None
    conviction_decay: bool = False  # Exit when |pred| < threshold

    # Filters
    time_filter: Optional[str] = None  # "morning", "afternoon", "morning_afternoon"
    vol_filter: Optional[float] = None  # Min volatility percentile

    # Execution
    chase_max_ticks: int = 1  # Max ticks to chase
    chase_max_reprices: int = 3  # Max reprice attempts

    # Risk
    max_daily_loss_ticks: Optional[float] = None
    max_daily_trades: Optional[int] = None

    # Metadata
    enabled: bool = True
    description: str = ""


@dataclass
class CardPosition:
    """Current position for a card"""
    side: Side
    size: int
    entry_price: float
    entry_time: float
    entry_prediction: float
    order_id: Optional[str] = None

    # Running metrics
    max_favorable_excursion: float = 0.0  # MFE (ticks)
    max_adverse_excursion: float = 0.0    # MAE (ticks)

    # Exit tracking
    exit_triggered: bool = False
    exit_reason: Optional[str] = None


@dataclass
class CardStats:
    """Card performance statistics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl_ticks: float = 0.0
    total_pnl_dollars: float = 0.0
    max_drawdown_ticks: float = 0.0
    current_drawdown: float = 0.0
    daily_trades: int = 0
    daily_pnl_ticks: float = 0.0
    last_reset_date: str = ""


class TradingCard:
    """
    Single trading strategy implementation.

    Lifecycle:
    1. Receives prediction
    2. Evaluates entry rules
    3. Generates order if triggered
    4. Monitors position
    5. Generates exit order when exit condition met
    """

    def __init__(self, config: CardConfig):
        self.config = config
        self.state = CardState.INACTIVE
        self.position: Optional[CardPosition] = None
        self.stats = CardStats()

        # Callbacks
        self._order_callback: Optional[Callable] = None

    def set_order_callback(self, callback: Callable):
        """Set callback for order generation: callback(card_name, side, size, price_type, limit_price)"""
        self._order_callback = callback

    def on_prediction(self, prediction: Prediction, current_price: float):
        """
        Process prediction and decide action.

        Args:
            prediction: Model prediction
            current_price: Current market mid price
        """
        if not self.config.enabled:
            return

        # If we have a position, check exit conditions
        if self.state == CardState.ACTIVE and self.position is not None:
            self._check_exit_conditions(prediction, current_price)
            return

        # If no position, check entry conditions
        if self.state == CardState.INACTIVE:
            self._check_entry_conditions(prediction, current_price)

    def _check_entry_conditions(self, prediction: Prediction, current_price: float):
        """Evaluate entry rules"""
        # Threshold check
        if abs(prediction.value) < self.config.threshold:
            return

        # Daily limits check
        if self.config.max_daily_trades and self.stats.daily_trades >= self.config.max_daily_trades:
            return

        if self.config.max_daily_loss_ticks and self.stats.daily_pnl_ticks < -abs(self.config.max_daily_loss_ticks):
            return

        # Time filter check
        if self.config.time_filter:
            if not self._check_time_filter():
                return

        # All checks passed — generate entry order
        side = Side.LONG if prediction.value > 0 else Side.SHORT
        size = min(self.config.max_position_size, 1)  # Start with 1 contract

        # Generate order
        self._generate_order(
            side=side,
            size=size,
            entry_price=current_price,
            entry_prediction=prediction.value,
        )

    def _generate_order(self, side: Side, size: int, entry_price: float, entry_prediction: float):
        """Generate and submit order"""
        if self._order_callback is None:
            logger.warning(f"Card {self.config.name}: No order callback set")
            return

        # Create position object
        self.position = CardPosition(
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=time.time(),
            entry_prediction=entry_prediction,
        )

        self.state = CardState.PENDING

        # Submit order via callback
        order_side = "BUY" if side == Side.LONG else "SELL"
        try:
            self._order_callback(
                card_name=self.config.name,
                side=order_side,
                size=size,
                price_type="MARKET",  # TODO: support limit orders with chase logic
                limit_price=None,
            )
            logger.info(f"Card {self.config.name}: Generated {order_side} order for {size} @ {entry_price:.2f}")
        except Exception as e:
            logger.error(f"Card {self.config.name}: Order generation failed: {e}")
            self.state = CardState.INACTIVE
            self.position = None

    def on_fill(self, fill_price: float, fill_size: int):
        """Handle order fill notification"""
        if self.state != CardState.PENDING or self.position is None:
            logger.warning(f"Card {self.config.name}: Unexpected fill received")
            return

        # Update position with actual fill price
        self.position.entry_price = fill_price
        self.state = CardState.ACTIVE
        self.stats.total_trades += 1
        self.stats.daily_trades += 1

        logger.info(f"Card {self.config.name}: Filled {fill_size} @ {fill_price:.2f}")

    def _check_exit_conditions(self, prediction: Prediction, current_price: float):
        """Check if position should be exited"""
        if self.position is None:
            return

        # Update MFE/MAE
        if self.position.side == Side.LONG:
            pnl_ticks = (current_price - self.position.entry_price) / 0.25  # ES tick = 0.25
        else:
            pnl_ticks = (self.position.entry_price - current_price) / 0.25

        self.position.max_favorable_excursion = max(self.position.max_favorable_excursion, pnl_ticks)
        self.position.max_adverse_excursion = min(self.position.max_adverse_excursion, pnl_ticks)

        exit_reason = None

        # Take profit check
        if self.config.take_profit_ticks and pnl_ticks >= self.config.take_profit_ticks:
            exit_reason = "take_profit"

        # Stop loss check
        if self.config.stop_loss_ticks and pnl_ticks <= -abs(self.config.stop_loss_ticks):
            exit_reason = "stop_loss"

        # Max hold time check
        if self.config.max_hold_seconds:
            hold_time = time.time() - self.position.entry_time
            if hold_time >= self.config.max_hold_seconds:
                exit_reason = "max_hold"

        # Conviction decay check
        if self.config.conviction_decay:
            # Exit when prediction sign flips or magnitude below threshold
            same_direction = (
                (self.position.side == Side.LONG and prediction.value > 0) or
                (self.position.side == Side.SHORT and prediction.value < 0)
            )
            if not same_direction or abs(prediction.value) < self.config.threshold:
                exit_reason = "conviction_decay"

        # Generate exit order
        if exit_reason:
            self._exit_position(current_price, exit_reason)

    def _exit_position(self, exit_price: float, reason: str):
        """Close position"""
        if self.position is None:
            return

        # Calculate final PnL
        if self.position.side == Side.LONG:
            pnl_ticks = (exit_price - self.position.entry_price) / 0.25
        else:
            pnl_ticks = (self.position.entry_price - exit_price) / 0.25

        pnl_dollars = pnl_ticks * 12.50  # ES tick value

        # Update stats
        self.stats.total_pnl_ticks += pnl_ticks
        self.stats.total_pnl_dollars += pnl_dollars
        self.stats.daily_pnl_ticks += pnl_ticks

        if pnl_ticks > 0:
            self.stats.winning_trades += 1
        else:
            self.stats.losing_trades += 1

        # Generate exit order
        if self._order_callback:
            exit_side = "SELL" if self.position.side == Side.LONG else "BUY"
            self._order_callback(
                card_name=self.config.name,
                side=exit_side,
                size=self.position.size,
                price_type="MARKET",
                limit_price=None,
            )

        logger.info(
            f"Card {self.config.name}: EXIT {reason} | "
            f"PnL: {pnl_ticks:+.1f}t (${pnl_dollars:+.2f}) | "
            f"MFE: {self.position.max_favorable_excursion:.1f}t | "
            f"MAE: {self.position.max_adverse_excursion:.1f}t"
        )

        # Reset state
        self.position = None
        self.state = CardState.INACTIVE

    def _check_time_filter(self) -> bool:
        """Check if current time passes filter"""
        from datetime import datetime
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        minutes_since_930 = (hour - 9) * 60 + (minute - 30)

        if self.config.time_filter == "morning":
            return 0 <= minutes_since_930 < 120  # 9:30-11:30
        elif self.config.time_filter == "afternoon":
            return 240 <= minutes_since_930 < 390  # 1:30-4:00
        elif self.config.time_filter == "morning_afternoon":
            return (0 <= minutes_since_930 < 120) or (240 <= minutes_since_930 < 390)
        return True

    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.stats.daily_trades = 0
        self.stats.daily_pnl_ticks = 0.0
        from datetime import datetime
        self.stats.last_reset_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Card {self.config.name}: Daily stats reset")


class CardEngine:
    """Manages multiple trading cards"""

    def __init__(self):
        self.cards: Dict[str, TradingCard] = {}

    def add_card(self, config: CardConfig) -> TradingCard:
        """Add new trading card"""
        card = TradingCard(config)
        self.cards[config.name] = card
        logger.info(f"Added card: {config.name} (model={config.model_name})")
        return card

    def remove_card(self, name: str):
        """Remove card"""
        if name in self.cards:
            del self.cards[name]
            logger.info(f"Removed card: {name}")

    def get_card(self, name: str) -> Optional[TradingCard]:
        """Get card by name"""
        return self.cards.get(name)

    def list_cards(self) -> List[str]:
        """List all card names"""
        return list(self.cards.keys())

    def on_prediction(self, prediction: Prediction, current_price: float):
        """Route prediction to relevant cards"""
        for card in self.cards.values():
            if card.config.model_name == prediction.model_name:
                card.on_prediction(prediction, current_price)

    def reset_all_daily_stats(self):
        """Reset daily stats for all cards"""
        for card in self.cards.values():
            card.reset_daily_stats()

    def get_summary(self) -> Dict:
        """Get summary stats for all cards"""
        summary = {}
        for name, card in self.cards.items():
            summary[name] = {
                "state": card.state.value,
                "enabled": card.config.enabled,
                "total_trades": card.stats.total_trades,
                "win_rate": card.stats.winning_trades / max(1, card.stats.total_trades),
                "total_pnl_ticks": card.stats.total_pnl_ticks,
                "total_pnl_dollars": card.stats.total_pnl_dollars,
                "daily_trades": card.stats.daily_trades,
                "daily_pnl_ticks": card.stats.daily_pnl_ticks,
            }
        return summary
