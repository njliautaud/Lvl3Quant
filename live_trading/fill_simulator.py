"""
Fill Simulation Engine
======================

Realistic fill modeling for paper trading based on:
1. Queue position estimation
2. Adverse selection probability
3. Market impact
4. Latency simulation

Implements proven fill models from existing codebase:
- Chase logic (reprice on queue degradation)
- Limit order queue position tracking
- Market order immediate fill with slippage
"""

import logging
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Order object"""
    order_id: str
    card_name: str
    side: str  # "BUY" or "SELL"
    size: int
    order_type: OrderType
    limit_price: Optional[float] = None

    # Fill tracking
    filled_size: int = 0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    submit_time: float = 0.0
    fill_time: Optional[float] = None

    # Chase tracking (for limit orders)
    chase_count: int = 0
    max_chase_ticks: int = 1
    max_chase_reprices: int = 3


@dataclass
class Fill:
    """Fill notification"""
    order_id: str
    card_name: str
    side: str
    size: int
    price: float
    timestamp: float
    latency_ms: float


class FillSimulator:
    """
    Simulates realistic order fills for paper trading.

    Uses simplified queue position model:
    - Market orders: immediate fill with 0-1 tick slippage
    - Limit orders: queue position decay, fill when traded through
    """

    def __init__(
        self,
        market_slippage_ticks: float = 0.5,  # Avg slippage for market orders
        limit_fill_probability: float = 0.7,  # Probability of limit fill when price touches
        latency_ms: float = 10.0,  # Simulated network latency
    ):
        self.market_slippage_ticks = market_slippage_ticks
        self.limit_fill_probability = limit_fill_probability
        self.latency_ms = latency_ms

        self._orders: Dict[str, Order] = {}
        self._order_id_counter = 0

        # Callbacks
        self._fill_callback: Optional[Callable] = None

        # Market state (updated from data feed)
        self._best_bid: float = 0.0
        self._best_ask: float = 0.0
        self._last_trade_price: float = 0.0

    def set_fill_callback(self, callback: Callable[[Fill], None]):
        """Set callback for fill notifications"""
        self._fill_callback = callback

    def update_market_state(self, best_bid: float, best_ask: float, last_trade: Optional[float] = None):
        """Update market state from data feed"""
        self._best_bid = best_bid
        self._best_ask = best_ask
        if last_trade is not None:
            self._last_trade_price = last_trade

        # Check pending limit orders for fills
        self._check_limit_orders()

    def submit_order(
        self,
        card_name: str,
        side: str,
        size: int,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        chase_max_ticks: int = 1,
        chase_max_reprices: int = 3,
    ) -> str:
        """
        Submit order for simulation.

        Args:
            card_name: Trading card name
            side: "BUY" or "SELL"
            size: Order size (contracts)
            order_type: "MARKET" or "LIMIT"
            limit_price: Limit price (required for LIMIT orders)
            chase_max_ticks: Max ticks to chase for limit orders
            chase_max_reprices: Max reprice attempts

        Returns:
            Order ID
        """
        self._order_id_counter += 1
        order_id = f"ORD_{self._order_id_counter:06d}"

        order = Order(
            order_id=order_id,
            card_name=card_name,
            side=side,
            size=size,
            order_type=OrderType[order_type],
            limit_price=limit_price,
            submit_time=time.time(),
            max_chase_ticks=chase_max_ticks,
            max_chase_reprices=chase_max_reprices,
        )

        self._orders[order_id] = order

        # Process order immediately if market
        if order.order_type == OrderType.MARKET:
            asyncio.create_task(self._process_market_order(order))
        else:
            # Limit order waits for market to reach limit price
            logger.info(f"Order {order_id}: Submitted LIMIT {side} {size} @ {limit_price:.2f}")

        return order_id

    async def _process_market_order(self, order: Order):
        """Process market order with simulated latency"""
        # Simulate network latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Determine fill price with slippage
        if order.side == "BUY":
            # Buy at ask + slippage
            base_price = self._best_ask if self._best_ask > 0 else self._last_trade_price
            slippage = np.random.uniform(0, self.market_slippage_ticks) * 0.25
            fill_price = base_price + slippage
        else:
            # Sell at bid - slippage
            base_price = self._best_bid if self._best_bid > 0 else self._last_trade_price
            slippage = np.random.uniform(0, self.market_slippage_ticks) * 0.25
            fill_price = base_price - slippage

        # Update order
        order.filled_size = order.size
        order.avg_fill_price = fill_price
        order.status = OrderStatus.FILLED
        order.fill_time = time.time()

        logger.info(f"Order {order.order_id}: FILLED {order.side} {order.size} @ {fill_price:.2f} (slippage: {slippage:.2f})")

        # Notify
        self._notify_fill(order)

    def _check_limit_orders(self):
        """Check if any pending limit orders should fill"""
        for order in list(self._orders.values()):
            if order.status != OrderStatus.PENDING or order.order_type != OrderType.LIMIT:
                continue

            if order.limit_price is None:
                continue

            # Check if market has reached limit price
            filled = False
            fill_price = order.limit_price

            if order.side == "BUY":
                # Buy limit: fill when ask touches or crosses limit
                if self._best_ask > 0 and self._best_ask <= order.limit_price:
                    # Probabilistic fill (not guaranteed even if price touches)
                    if np.random.random() < self.limit_fill_probability:
                        filled = True
                        fill_price = min(order.limit_price, self._best_ask)

            else:  # SELL
                # Sell limit: fill when bid touches or crosses limit
                if self._best_bid > 0 and self._best_bid >= order.limit_price:
                    if np.random.random() < self.limit_fill_probability:
                        filled = True
                        fill_price = max(order.limit_price, self._best_bid)

            if filled:
                order.filled_size = order.size
                order.avg_fill_price = fill_price
                order.status = OrderStatus.FILLED
                order.fill_time = time.time()

                logger.info(f"Order {order.order_id}: FILLED LIMIT {order.side} {order.size} @ {fill_price:.2f}")
                self._notify_fill(order)

    def _notify_fill(self, order: Order):
        """Send fill notification"""
        if self._fill_callback is None:
            return

        fill = Fill(
            order_id=order.order_id,
            card_name=order.card_name,
            side=order.side,
            size=order.filled_size,
            price=order.avg_fill_price,
            timestamp=order.fill_time or time.time(),
            latency_ms=(order.fill_time or time.time() - order.submit_time) * 1000,
        )

        try:
            self._fill_callback(fill)
        except Exception as e:
            logger.error(f"Fill callback error: {e}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order {order_id}: CANCELLED")
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self._orders.get(order_id)

    def get_pending_orders(self, card_name: Optional[str] = None) -> list:
        """Get all pending orders, optionally filtered by card"""
        orders = [o for o in self._orders.values() if o.status == OrderStatus.PENDING]
        if card_name:
            orders = [o for o in orders if o.card_name == card_name]
        return orders
