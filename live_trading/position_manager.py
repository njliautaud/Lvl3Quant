"""
Position and PnL Manager
=========================

Tracks:
- Open positions by card
- Real-time mark-to-market PnL
- Closed trade history
- Position limits
- Aggregate exposure
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position"""
    card_name: str
    side: str  # "LONG" or "SHORT"
    size: int
    entry_price: float
    entry_time: float
    current_price: float = 0.0

    @property
    def unrealized_pnl_ticks(self) -> float:
        """Unrealized PnL in ticks"""
        if self.side == "LONG":
            return (self.current_price - self.entry_price) / 0.25
        else:
            return (self.entry_price - self.current_price) / 0.25

    @property
    def unrealized_pnl_dollars(self) -> float:
        """Unrealized PnL in dollars"""
        return self.unrealized_pnl_ticks * 12.50 * self.size


@dataclass
class ClosedTrade:
    """Closed trade record"""
    card_name: str
    side: str
    size: int
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    pnl_ticks: float
    pnl_dollars: float
    exit_reason: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict"""
        return {
            "card_name": self.card_name,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "pnl_ticks": self.pnl_ticks,
            "pnl_dollars": self.pnl_dollars,
            "exit_reason": self.exit_reason,
            "hold_seconds": self.exit_time - self.entry_time,
        }


class PositionManager:
    """
    Central position and PnL manager.

    Responsibilities:
    - Track open positions per card
    - Update mark-to-market PnL
    - Record closed trades
    - Enforce position limits
    - Generate PnL reports
    """

    def __init__(
        self,
        max_total_position: int = 5,  # Max aggregate position size
        max_card_position: int = 1,   # Max position per card
        trade_log_path: Optional[str] = None,
    ):
        self.max_total_position = max_total_position
        self.max_card_position = max_card_position

        # State
        self.positions: Dict[str, Position] = {}  # card_name -> Position
        self.closed_trades: List[ClosedTrade] = []

        # Running totals
        self._total_realized_pnl_ticks = 0.0
        self._total_realized_pnl_dollars = 0.0
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0

        # Trade log
        if trade_log_path is None:
            trade_log_path = "/home/jupiter/Lvl3Quant/live_trading/logs/trades.jsonl"
        self.trade_log_path = Path(trade_log_path)
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)

    def open_position(self, card_name: str, side: str, size: int, entry_price: float) -> bool:
        """
        Open new position for card.

        Args:
            card_name: Card name
            side: "LONG" or "SHORT"
            size: Position size
            entry_price: Entry price

        Returns:
            True if position opened, False if rejected (limits exceeded)
        """
        # Check if card already has position
        if card_name in self.positions:
            logger.warning(f"PositionManager: Card {card_name} already has open position")
            return False

        # Check card position limit
        if size > self.max_card_position:
            logger.warning(f"PositionManager: Card {card_name} size {size} exceeds limit {self.max_card_position}")
            return False

        # Check total position limit
        total_position = sum(p.size for p in self.positions.values())
        if total_position + size > self.max_total_position:
            logger.warning(f"PositionManager: Total position {total_position + size} exceeds limit {self.max_total_position}")
            return False

        # Create position
        position = Position(
            card_name=card_name,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=time.time(),
            current_price=entry_price,
        )

        self.positions[card_name] = position
        logger.info(f"PositionManager: Opened {side} {size} for {card_name} @ {entry_price:.2f}")
        return True

    def close_position(self, card_name: str, exit_price: float, exit_reason: str = "unknown") -> Optional[ClosedTrade]:
        """
        Close position for card.

        Args:
            card_name: Card name
            exit_price: Exit price
            exit_reason: Reason for exit

        Returns:
            ClosedTrade object or None if no position
        """
        position = self.positions.get(card_name)
        if position is None:
            logger.warning(f"PositionManager: No position found for {card_name}")
            return None

        # Calculate PnL
        if position.side == "LONG":
            pnl_ticks = (exit_price - position.entry_price) / 0.25
        else:
            pnl_ticks = (position.entry_price - exit_price) / 0.25

        pnl_dollars = pnl_ticks * 12.50 * position.size

        # Create closed trade record
        trade = ClosedTrade(
            card_name=card_name,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=time.time(),
            pnl_ticks=pnl_ticks,
            pnl_dollars=pnl_dollars,
            exit_reason=exit_reason,
        )

        # Update running totals
        self._total_trades += 1
        self._total_realized_pnl_ticks += pnl_ticks
        self._total_realized_pnl_dollars += pnl_dollars

        if pnl_ticks > 0:
            self._winning_trades += 1
        else:
            self._losing_trades += 1

        # Store trade
        self.closed_trades.append(trade)

        # Write to log
        self._log_trade(trade)

        # Remove position
        del self.positions[card_name]

        logger.info(
            f"PositionManager: Closed {position.side} {position.size} for {card_name} @ {exit_price:.2f} | "
            f"PnL: {pnl_ticks:+.1f}t (${pnl_dollars:+.2f}) | Reason: {exit_reason}"
        )

        return trade

    def update_mark_price(self, mark_price: float):
        """Update mark price for all positions"""
        for position in self.positions.values():
            position.current_price = mark_price

    def get_position(self, card_name: str) -> Optional[Position]:
        """Get position for card"""
        return self.positions.get(card_name)

    def has_position(self, card_name: str) -> bool:
        """Check if card has open position"""
        return card_name in self.positions

    def get_total_exposure(self) -> int:
        """Get total position size across all cards"""
        return sum(p.size for p in self.positions.values())

    def get_total_unrealized_pnl(self) -> tuple:
        """
        Get total unrealized PnL.

        Returns:
            (pnl_ticks, pnl_dollars)
        """
        total_ticks = sum(p.unrealized_pnl_ticks for p in self.positions.values())
        total_dollars = sum(p.unrealized_pnl_dollars for p in self.positions.values())
        return total_ticks, total_dollars

    def get_total_realized_pnl(self) -> tuple:
        """
        Get total realized PnL.

        Returns:
            (pnl_ticks, pnl_dollars)
        """
        return self._total_realized_pnl_ticks, self._total_realized_pnl_dollars

    def get_summary(self) -> dict:
        """Get full PnL summary"""
        unrealized_ticks, unrealized_dollars = self.get_total_unrealized_pnl()

        return {
            "timestamp": time.time(),
            "open_positions": len(self.positions),
            "total_exposure": self.get_total_exposure(),
            "unrealized_pnl_ticks": unrealized_ticks,
            "unrealized_pnl_dollars": unrealized_dollars,
            "realized_pnl_ticks": self._total_realized_pnl_ticks,
            "realized_pnl_dollars": self._total_realized_pnl_dollars,
            "total_pnl_ticks": unrealized_ticks + self._total_realized_pnl_ticks,
            "total_pnl_dollars": unrealized_dollars + self._total_realized_pnl_dollars,
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "losing_trades": self._losing_trades,
            "win_rate": self._winning_trades / max(1, self._total_trades),
        }

    def get_recent_trades(self, n: int = 10) -> List[dict]:
        """Get N most recent closed trades"""
        return [t.to_dict() for t in self.closed_trades[-n:]]

    def _log_trade(self, trade: ClosedTrade):
        """Append trade to JSONL log"""
        try:
            with open(self.trade_log_path, 'a') as f:
                f.write(json.dumps(trade.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def reset_daily(self):
        """Reset daily statistics (keep positions open)"""
        # This is called at start of new trading day
        # Could reset daily counters here if needed
        logger.info("PositionManager: Daily reset")
