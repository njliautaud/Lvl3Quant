#!/usr/bin/env python3
"""
Tick-Level FIFO Replay Engine for ES Futures Backtesting
=========================================================

Replays raw Databento MBO events tick-by-tick, simulates passive limit order
fills with honest FIFO queue position tracking, and validates CNN-Mamba
predictions against random permutation baselines.

Key design:
- Entry: passive limit at BBO (bid for longs, ask for shorts)
- Fill: requires enough volume to trade THROUGH our queue position (back of queue)
- TP exit: passive limit (fills when traded through, FIFO queue)
- SL exit: market order (immediate fill at adverse price)
- Time stop: market exit after N seconds
- Permutation test: 100 random-direction trials per config, p < 0.05 required

Cost model (AMP/Rithmic canonical):
- Commission: $4.70 RT = 0.376 ticks
- Passive entry/exit: 0.376 ticks (commission only)
- Market exit (SL/time stop): 1.376 ticks (commission + 1 tick spread crossing)

Author: Claude (tick_replay_engine)
"""

import numpy as np
import glob
import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Constants
# =============================================================================

TICK_SIZE = 0.25          # ES tick size in points
TICK_VALUE = 12.50        # $ per tick
COMMISSION_RT_TICKS = 0.376  # $4.70 / $12.50
SPREAD_TICKS = 1.0        # ES is 1-tick wide during RTH

# Cost per exit type (in ticks, includes commission)
COST_PASSIVE_EXIT = COMMISSION_RT_TICKS      # 0.376 ticks (TP hit)
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS  # 1.376 ticks (SL / time stop)

# Prediction stride (250 MBO events per prediction)
PRED_STRIDE = 250
PRED_WINDOW = 1500  # Window of events before prediction point

# RTH bounds (ET, as nanoseconds offset from midnight)
RTH_OPEN_NS  = 9 * 3600 * 10**9 + 30 * 60 * 10**9   # 09:30 ET
RTH_CLOSE_NS = 16 * 3600 * 10**9                      # 16:00 ET

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Trade:
    """A completed trade."""
    trade_id: int
    side: str          # 'long' or 'short'
    entry_price: float
    exit_price: float
    entry_time_ns: int
    exit_time_ns: int
    signal_time_ns: int
    signal_strength: float
    exit_reason: str   # 'tp', 'sl', 'time_stop', 'eod'
    queue_depth_at_entry: int
    fill_latency_ns: int  # Time from signal to fill
    pnl_ticks: float = 0.0
    pnl_dollars: float = 0.0
    cost_ticks: float = 0.0
    mfe_ticks: float = 0.0
    mae_ticks: float = 0.0


@dataclass
class PendingOrder:
    """A resting limit order awaiting fill."""
    order_id: int
    side: str          # 'long' or 'short'
    price: float       # Limit price
    post_time_ns: int
    signal_time_ns: int
    signal_strength: float
    queue_ahead: float  # Contracts ahead in FIFO queue
    tp_price: float
    sl_price: float
    max_hold_ns: int   # Max hold time in ns from fill
    cancel_time_ns: int  # Cancel if not filled by this time


@dataclass
class OpenPosition:
    """A filled position awaiting exit."""
    order_id: int
    side: str
    entry_price: float
    fill_time_ns: int
    signal_time_ns: int
    signal_strength: float
    queue_depth_at_entry: int
    fill_latency_ns: int
    tp_price: float
    sl_price: float
    exit_deadline_ns: int  # Market exit after this time
    # TP exit order tracking
    tp_queue_ahead: float = 0.0
    tp_posted: bool = False
    # MFE/MAE tracking
    best_price: float = 0.0
    worst_price: float = 0.0


# =============================================================================
# Order Book (simplified BBO tracker)
# =============================================================================

class BBOTracker:
    """
    Track best bid/ask from MBO events.
    We only need BBO + depth at BBO for queue position estimation.
    """
    def __init__(self):
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.bid_size = 0      # Total size at best bid
        self.ask_size = 0      # Total size at best ask
        # Track all price levels for depth
        self.bids: Dict[float, int] = defaultdict(int)  # price -> total size
        self.asks: Dict[float, int] = defaultdict(int)
        self._order_book: Dict[int, Tuple[str, float, int]] = {}  # order_id -> (side, price, size)

    def process_event(self, action: str, side: str, price: float, size: int, order_id: int):
        """Process a single MBO event and update BBO."""
        if np.isnan(price) or price <= 0:
            return

        if action == 'R':
            # Book reset/snapshot
            return

        if side == 'B':
            levels = self.bids
        elif side == 'A':
            levels = self.asks
        else:
            return

        if action == 'A':
            # Add order — but skip crossing orders (aggressive limit orders that
            # will immediately match). In Databento CME MBO, a crossing limit order
            # generates an 'A' event BEFORE the T/F matching events. Adding it to
            # the book temporarily crosses bid/ask, corrupting BBO.
            # Track in _order_book for subsequent T/F cleanup, but don't add to levels.
            if side == 'B' and self.asks and price >= min(self.asks.keys()):
                # Crossing bid — will be matched by subsequent T/F events
                self._order_book[order_id] = (side, price, size)
            elif side == 'A' and self.bids and price <= max(self.bids.keys()):
                # Crossing ask — will be matched by subsequent T/F events
                self._order_book[order_id] = (side, price, size)
            else:
                levels[price] += size
                self._order_book[order_id] = (side, price, size)
        elif action == 'C':
            # Cancel order
            if order_id in self._order_book:
                _, old_price, old_size = self._order_book.pop(order_id)
                old_levels = self.bids if side == 'B' else self.asks
                old_levels[old_price] = max(0, old_levels[old_price] - old_size)
                if old_levels[old_price] == 0:
                    del old_levels[old_price]
            else:
                levels[price] = max(0, levels[price] - size)
                if levels[price] == 0 and price in levels:
                    del levels[price]
        elif action == 'M':
            # Modify order (change size or price)
            if order_id in self._order_book:
                _, old_price, old_size = self._order_book[order_id]
                old_levels = self.bids if side == 'B' else self.asks
                old_levels[old_price] = max(0, old_levels[old_price] - old_size)
                if old_levels[old_price] == 0 and old_price in old_levels:
                    del old_levels[old_price]
            levels[price] += size
            self._order_book[order_id] = (side, price, size)
        elif action == 'T':
            # Trade event = AGGRESSOR order in Databento CME MBO.
            # Each match produces [T, F, F, ...]: T is aggressor, F's are resting fills.
            # The aggressor is NOT a resting order — do NOT reduce price levels here.
            # The F events will handle book reduction for the resting orders.
            # We only need to track the aggressor order_id if it was somehow added
            # (IOC orders aren't added, but limit orders that cross the spread are).
            if order_id in self._order_book:
                _, old_price, old_size = self._order_book[order_id]
                old_levels = self.bids if side == 'B' else self.asks
                old_levels[old_price] = max(0, old_levels.get(old_price, 0) - size)
                if old_levels[old_price] == 0 and old_price in old_levels:
                    del old_levels[old_price]
                remaining = old_size - size
                if remaining <= 0:
                    del self._order_book[order_id]
                else:
                    self._order_book[order_id] = (side, old_price, remaining)
        elif action == 'F':
            # Fill event = RESTING order filled in Databento CME MBO.
            # `side` is the resting order's own side (B=resting bid, A=resting ask).
            # Reduce from the resting order's OWN side (same side, not opposite).
            levels[price] = max(0, levels.get(price, 0) - size)
            if levels[price] == 0 and price in levels:
                del levels[price]
            if order_id in self._order_book:
                del self._order_book[order_id]

        # Update BBO
        if self.bids:
            self.best_bid = max(self.bids.keys())
            self.bid_size = self.bids[self.best_bid]
        else:
            self.best_bid = 0.0
            self.bid_size = 0

        if self.asks:
            self.best_ask = min(self.asks.keys())
            self.ask_size = self.asks[self.best_ask]
        else:
            self.best_ask = 0.0
            self.ask_size = 0

    def get_depth_at_price(self, price: float, side: str) -> int:
        """Get total displayed size at a price level."""
        if side == 'B':
            return self.bids.get(price, 0)
        else:
            return self.asks.get(price, 0)


# =============================================================================
# Fill Simulation Engine
# =============================================================================

class TickReplayEngine:
    """
    Event-by-event FIFO fill simulation engine.

    For each prediction:
    1. If |pred| > threshold, post a passive limit order at BBO
    2. Track queue position (back of queue at post time)
    3. Fill when enough volume trades through our price level
    4. After fill, post TP limit and track SL as market stop
    5. Exit on TP fill, SL breach, time stop, or EOD
    """

    def __init__(self, tp_ticks: int, sl_ticks: int, hold_seconds: float = 30.0,
                 signal_threshold: float = 0.0, cancel_seconds: float = 15.0,
                 max_concurrent: int = 1):
        self.tp_ticks = tp_ticks
        self.sl_ticks = sl_ticks
        self.hold_ns = int(hold_seconds * 1e9)
        self.signal_threshold = signal_threshold
        self.cancel_ns = int(cancel_seconds * 1e9)
        self.max_concurrent = max_concurrent

        self.book = BBOTracker()
        self.pending_orders: List[PendingOrder] = []
        self.open_positions: List[OpenPosition] = []
        self.completed_trades: List[Trade] = []
        self.next_order_id = 0
        self.last_trade_price = 0.0

    def _ticks_to_price(self, ticks: int) -> float:
        return ticks * TICK_SIZE

    def _price_to_ticks(self, price_diff: float) -> float:
        return price_diff / TICK_SIZE

    def _post_entry_order(self, pred: float, ts_ns: int):
        """Post a passive limit entry order at BBO."""
        if self.book.best_bid <= 0 or self.book.best_ask <= 0:
            return
        if self.book.best_ask - self.book.best_bid > 2 * TICK_SIZE:
            return  # Skip wide spreads

        # Determine direction
        if pred > self.signal_threshold:
            side = 'long'
            entry_price = self.book.best_bid  # Passive buy at bid
            queue_ahead = self.book.bid_size   # Back of queue
            tp_price = entry_price + self._ticks_to_price(self.tp_ticks)
            sl_price = entry_price - self._ticks_to_price(self.sl_ticks)
        elif pred < -self.signal_threshold:
            side = 'short'
            entry_price = self.book.best_ask  # Passive sell at ask
            queue_ahead = self.book.ask_size   # Back of queue
            tp_price = entry_price - self._ticks_to_price(self.tp_ticks)
            sl_price = entry_price + self._ticks_to_price(self.sl_ticks)
        else:
            return

        # Check if we already have an order or position
        total_exposure = len(self.pending_orders) + len(self.open_positions)
        if total_exposure >= self.max_concurrent:
            return

        order = PendingOrder(
            order_id=self.next_order_id,
            side=side,
            price=entry_price,
            post_time_ns=ts_ns,
            signal_time_ns=ts_ns,
            signal_strength=pred,
            queue_ahead=float(queue_ahead),
            tp_price=tp_price,
            sl_price=sl_price,
            max_hold_ns=self.hold_ns,
            cancel_time_ns=ts_ns + self.cancel_ns,
        )
        self.next_order_id += 1
        self.pending_orders.append(order)

    def _check_pending_fills(self, trade_price: float, trade_size: int,
                              trade_side: str, ts_ns: int):
        """Check if any pending orders get filled by this trade."""
        filled = []
        for order in self.pending_orders:
            # Check if trade is at our price level
            if order.side == 'long' and trade_side == 'A':
                # Aggressive sell hitting the bid — can fill our buy
                if abs(trade_price - order.price) < 0.001:
                    order.queue_ahead -= trade_size
                    if order.queue_ahead <= 0:
                        filled.append(order)
                elif trade_price < order.price:
                    # Traded through our level — immediate fill
                    filled.append(order)
            elif order.side == 'short' and trade_side == 'B':
                # Aggressive buy hitting the ask — can fill our sell
                if abs(trade_price - order.price) < 0.001:
                    order.queue_ahead -= trade_size
                    if order.queue_ahead <= 0:
                        filled.append(order)
                elif trade_price > order.price:
                    # Traded through our level — immediate fill
                    filled.append(order)

        for order in filled:
            self.pending_orders.remove(order)
            # Determine TP queue depth
            if order.side == 'long':
                tp_queue = self.book.get_depth_at_price(order.tp_price, 'A')
            else:
                tp_queue = self.book.get_depth_at_price(order.tp_price, 'B')

            pos = OpenPosition(
                order_id=order.order_id,
                side=order.side,
                entry_price=order.price,
                fill_time_ns=ts_ns,
                signal_time_ns=order.signal_time_ns,
                signal_strength=order.signal_strength,
                queue_depth_at_entry=int(order.queue_ahead + trade_size),
                fill_latency_ns=ts_ns - order.signal_time_ns,
                tp_price=order.tp_price,
                sl_price=order.sl_price,
                exit_deadline_ns=ts_ns + order.max_hold_ns,
                tp_queue_ahead=float(max(tp_queue, 0)),
                tp_posted=True,
                best_price=order.price,
                worst_price=order.price,
            )
            self.open_positions.append(pos)

    def _check_pending_cancels(self, ts_ns: int):
        """Cancel orders that have been resting too long."""
        cancelled = [o for o in self.pending_orders if ts_ns > o.cancel_time_ns]
        for o in cancelled:
            self.pending_orders.remove(o)

    def _check_exits(self, trade_price: float, trade_size: int,
                     trade_side: str, ts_ns: int):
        """Check if any open positions should exit."""
        closed = []
        for pos in self.open_positions:
            exit_reason = None
            exit_price = None
            cost = COST_PASSIVE_EXIT  # Default: passive exit

            # Update MFE/MAE
            if pos.side == 'long':
                pos.best_price = max(pos.best_price, trade_price)
                pos.worst_price = min(pos.worst_price, trade_price)
            else:
                pos.best_price = min(pos.best_price, trade_price)
                pos.worst_price = max(pos.worst_price, trade_price)

            # 1. Check SL (market exit — immediate, no queue)
            if pos.side == 'long' and trade_price <= pos.sl_price:
                exit_reason = 'sl'
                exit_price = pos.sl_price  # SL price (market at adverse)
                cost = COST_MARKET_EXIT
            elif pos.side == 'short' and trade_price >= pos.sl_price:
                exit_reason = 'sl'
                exit_price = pos.sl_price
                cost = COST_MARKET_EXIT

            # 2. Check TP (passive limit — needs FIFO queue depletion)
            if exit_reason is None and pos.tp_posted:
                if pos.side == 'long' and trade_side == 'B':
                    # Buy-side aggression at our TP ask level
                    if abs(trade_price - pos.tp_price) < 0.001:
                        pos.tp_queue_ahead -= trade_size
                        if pos.tp_queue_ahead <= 0:
                            exit_reason = 'tp'
                            exit_price = pos.tp_price
                            cost = COST_PASSIVE_EXIT
                    elif trade_price > pos.tp_price:
                        exit_reason = 'tp'
                        exit_price = pos.tp_price
                        cost = COST_PASSIVE_EXIT
                elif pos.side == 'short' and trade_side == 'A':
                    # Sell-side aggression at our TP bid level
                    if abs(trade_price - pos.tp_price) < 0.001:
                        pos.tp_queue_ahead -= trade_size
                        if pos.tp_queue_ahead <= 0:
                            exit_reason = 'tp'
                            exit_price = pos.tp_price
                            cost = COST_PASSIVE_EXIT
                    elif trade_price < pos.tp_price:
                        exit_reason = 'tp'
                        exit_price = pos.tp_price
                        cost = COST_PASSIVE_EXIT

            # 3. Check time stop (market exit)
            if exit_reason is None and ts_ns >= pos.exit_deadline_ns:
                exit_reason = 'time_stop'
                exit_price = trade_price  # Market exit at current price
                cost = COST_MARKET_EXIT

            if exit_reason is not None:
                # Compute PnL
                if pos.side == 'long':
                    raw_pnl_ticks = self._price_to_ticks(exit_price - pos.entry_price)
                    mfe = self._price_to_ticks(pos.best_price - pos.entry_price)
                    mae = self._price_to_ticks(pos.entry_price - pos.worst_price)
                else:
                    raw_pnl_ticks = self._price_to_ticks(pos.entry_price - exit_price)
                    mfe = self._price_to_ticks(pos.entry_price - pos.best_price)
                    mae = self._price_to_ticks(pos.worst_price - pos.entry_price)

                net_pnl_ticks = raw_pnl_ticks - cost

                trade = Trade(
                    trade_id=pos.order_id,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    entry_time_ns=pos.fill_time_ns,
                    exit_time_ns=ts_ns,
                    signal_time_ns=pos.signal_time_ns,
                    signal_strength=pos.signal_strength,
                    exit_reason=exit_reason,
                    queue_depth_at_entry=pos.queue_depth_at_entry,
                    fill_latency_ns=pos.fill_latency_ns,
                    pnl_ticks=net_pnl_ticks,
                    pnl_dollars=net_pnl_ticks * TICK_VALUE,
                    cost_ticks=cost,
                    mfe_ticks=max(mfe, 0),
                    mae_ticks=max(mae, 0),
                )
                self.completed_trades.append(trade)
                closed.append(pos)

        for pos in closed:
            self.open_positions.remove(pos)

    def _force_eod_exits(self, ts_ns: int):
        """Force-close all positions at EOD with market exit."""
        for pos in self.open_positions:
            exit_price = self.last_trade_price
            cost = COST_MARKET_EXIT

            if pos.side == 'long':
                raw_pnl = self._price_to_ticks(exit_price - pos.entry_price)
                mfe = self._price_to_ticks(pos.best_price - pos.entry_price)
                mae = self._price_to_ticks(pos.entry_price - pos.worst_price)
            else:
                raw_pnl = self._price_to_ticks(pos.entry_price - exit_price)
                mfe = self._price_to_ticks(pos.entry_price - pos.best_price)
                mae = self._price_to_ticks(pos.worst_price - pos.entry_price)

            trade = Trade(
                trade_id=pos.order_id,
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                entry_time_ns=pos.fill_time_ns,
                exit_time_ns=ts_ns,
                signal_time_ns=pos.signal_time_ns,
                signal_strength=pos.signal_strength,
                exit_reason='eod',
                queue_depth_at_entry=pos.queue_depth_at_entry,
                fill_latency_ns=pos.fill_latency_ns,
                pnl_ticks=raw_pnl - cost,
                pnl_dollars=(raw_pnl - cost) * TICK_VALUE,
                cost_ticks=cost,
                mfe_ticks=max(mfe, 0),
                mae_ticks=max(mae, 0),
            )
            self.completed_trades.append(trade)
        self.open_positions.clear()
        self.pending_orders.clear()

    def run_day(self, mbo_path: str, predictions: np.ndarray,
                pred_indices: Optional[np.ndarray] = None) -> List[Trade]:
        """
        Run simulation for one day.

        Args:
            mbo_path: Path to .dbn.zst MBO file
            predictions: 1D array of model predictions (aligned with MBO events at stride)
            pred_indices: Optional array of MBO event indices for each prediction.
                         If None, uses default stride=250, window=1500.

        Returns:
            List of completed trades
        """
        import databento as db

        # Reset state
        self.book = BBOTracker()
        self.pending_orders = []
        self.open_positions = []
        self.completed_trades = []
        self.next_order_id = 0
        self.last_trade_price = 0.0

        # Load MBO data
        dbn = db.DBNStore.from_file(mbo_path)
        df = dbn.to_df()

        # Find front-month ES contract (largest volume)
        es_symbols = [s for s in df['symbol'].unique()
                      if s.startswith('ES') and '-' not in s and len(s) <= 4]
        if not es_symbols:
            # Try longer symbols like ESH6
            es_symbols = [s for s in df['symbol'].unique()
                          if s.startswith('ES') and '-' not in s]
        if not es_symbols:
            print(f"  WARNING: No ES contract found in {mbo_path}")
            return []

        # Pick the one with most trades
        best_sym = max(es_symbols, key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))
        df = df[df['symbol'] == best_sym].copy()

        # Extract arrays for fast iteration
        ts_event = df['ts_event'].values.astype('int64')  # nanosecond timestamps
        actions = df['action'].values
        sides = df['side'].values
        prices = df['price'].values.astype('float64')
        sizes = df['size'].values.astype('int64')
        order_ids = df['order_id'].values.astype('int64')

        n_events = len(df)

        # Build prediction index mapping
        if pred_indices is None:
            # Default: predictions at stride=250 starting at window=1500
            n_preds = len(predictions)
            pred_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
            # Clamp to valid range
            pred_indices = pred_indices[pred_indices < n_events]
            predictions = predictions[:len(pred_indices)]

        # Create a set of prediction indices for fast lookup
        pred_map = {}
        for i, idx in enumerate(pred_indices):
            if idx < n_events:
                pred_map[int(idx)] = predictions[i]

        # Main event loop
        for evt_idx in range(n_events):
            action = actions[evt_idx]
            side = sides[evt_idx]
            price = prices[evt_idx]
            size = sizes[evt_idx]
            oid = order_ids[evt_idx]
            ts = ts_event[evt_idx]

            # Update order book
            self.book.process_event(action, side, price, size, oid)

            # Process trades
            if action in ('T', 'F') and not np.isnan(price) and price > 0:
                self.last_trade_price = price

                # Check entry fills
                self._check_pending_fills(price, size, side, ts)

                # Check exit conditions
                self._check_exits(price, size, side, ts)

            # Check for cancelled pending orders
            if evt_idx % 1000 == 0:
                self._check_pending_cancels(ts)

            # Check for prediction at this event index
            if evt_idx in pred_map:
                pred = pred_map[evt_idx]
                self._post_entry_order(pred, ts)

        # Force close any remaining positions
        if self.open_positions or self.pending_orders:
            self._force_eod_exits(ts_event[-1])

        return self.completed_trades

    def run_day_from_arrays(self, ts_event, actions, sides, prices, sizes, order_ids,
                            predictions: np.ndarray,
                            pred_indices: Optional[np.ndarray] = None) -> List[Trade]:
        """
        Run simulation using pre-loaded numpy arrays (avoids re-parsing MBO files).

        Args:
            ts_event: int64 array of nanosecond timestamps
            actions: array of action strings ('A','C','M','T','F','R')
            sides: array of side strings ('B','A')
            prices: float64 array of prices
            sizes: int64 array of sizes
            order_ids: int64 array of order IDs
            predictions: 1D array of model predictions
            pred_indices: Optional array of MBO event indices for each prediction

        Returns:
            List of completed trades
        """
        # Reset state
        self.book = BBOTracker()
        self.pending_orders = []
        self.open_positions = []
        self.completed_trades = []
        self.next_order_id = 0
        self.last_trade_price = 0.0

        n_events = len(ts_event)

        # Build prediction index mapping
        if pred_indices is None:
            n_preds = len(predictions)
            pred_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
            pred_indices = pred_indices[pred_indices < n_events]
            predictions = predictions[:len(pred_indices)]

        pred_map = {}
        for i, idx in enumerate(pred_indices):
            if idx < n_events:
                pred_map[int(idx)] = predictions[i]

        # Main event loop (identical to run_day)
        for evt_idx in range(n_events):
            action = actions[evt_idx]
            side = sides[evt_idx]
            price = prices[evt_idx]
            size = sizes[evt_idx]
            oid = order_ids[evt_idx]
            ts = ts_event[evt_idx]

            # Update order book
            self.book.process_event(action, side, price, size, oid)

            # Process trades
            if action in ('T', 'F') and not np.isnan(price) and price > 0:
                self.last_trade_price = price
                self._check_pending_fills(price, size, side, ts)
                self._check_exits(price, size, side, ts)

            # Check for cancelled pending orders
            if evt_idx % 1000 == 0:
                self._check_pending_cancels(ts)

            # Check for prediction at this event index
            if evt_idx in pred_map:
                pred = pred_map[evt_idx]
                self._post_entry_order(pred, ts)

        # Force close any remaining positions
        if self.open_positions or self.pending_orders:
            self._force_eod_exits(ts_event[-1])

        return self.completed_trades


    @staticmethod
    def preload_mbo(mbo_path: str):
        """
        Load and preprocess an MBO file into numpy arrays.
        Returns a dict of arrays suitable for run_day_from_arrays().
        Cache the result to avoid re-parsing.
        """
        import databento as db

        dbn = db.DBNStore.from_file(mbo_path)
        df = dbn.to_df()

        # Find front-month ES
        es_symbols = [s for s in df['symbol'].unique()
                      if s.startswith('ES') and '-' not in s and len(s) <= 4]
        if not es_symbols:
            es_symbols = [s for s in df['symbol'].unique()
                          if s.startswith('ES') and '-' not in s]
        if not es_symbols:
            return None

        best_sym = max(es_symbols,
                       key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))
        df = df[df['symbol'] == best_sym].copy()

        return {
            'ts_event': df['ts_event'].values.astype('int64'),
            'actions': df['action'].values,
            'sides': df['side'].values,
            'prices': df['price'].values.astype('float64'),
            'sizes': df['size'].values.astype('int64'),
            'order_ids': df['order_id'].values.astype('int64'),
            'n_events': len(df),
        }


# =============================================================================
# Analysis & Reporting
# =============================================================================

def compute_metrics(trades: List[Trade], label: str = "") -> Dict:
    """Compute performance metrics from a list of trades."""
    if not trades:
        return {
            'label': label, 'n_trades': 0, 'net_pnl_ticks': 0, 'net_pnl_dollars': 0,
            'win_rate': 0, 'profit_factor': 0, 'sharpe': 0, 'sortino': 0,
            'max_dd_ticks': 0, 'avg_pnl_ticks': 0, 'avg_mfe': 0, 'avg_mae': 0,
        }

    pnls = np.array([t.pnl_ticks for t in trades])
    n = len(pnls)
    wins = np.sum(pnls > 0)
    losses = np.sum(pnls <= 0)
    wr = wins / n if n > 0 else 0

    gross_profit = np.sum(pnls[pnls > 0]) if wins > 0 else 0
    gross_loss = abs(np.sum(pnls[pnls <= 0])) if losses > 0 else 0.001
    pf = gross_profit / gross_loss

    # Daily PnL for Sharpe/Sortino
    daily_pnl = defaultdict(float)
    for t in trades:
        # Group by date (using entry time)
        day_key = t.entry_time_ns // (24 * 3600 * 10**9)
        daily_pnl[day_key] += t.pnl_ticks

    daily_returns = np.array(list(daily_pnl.values()))
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0:
            sortino = np.mean(daily_returns) / np.std(downside) * np.sqrt(252)
        else:
            sortino = 99.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t.exit_reason] += 1

    return {
        'label': label,
        'n_trades': n,
        'net_pnl_ticks': float(np.sum(pnls)),
        'net_pnl_dollars': float(np.sum(pnls) * TICK_VALUE),
        'win_rate': float(wr),
        'profit_factor': float(pf),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd_ticks': float(max_dd),
        'avg_pnl_ticks': float(np.mean(pnls)),
        'avg_mfe': float(np.mean([t.mfe_ticks for t in trades])),
        'avg_mae': float(np.mean([t.mae_ticks for t in trades])),
        'n_days': len(daily_pnl),
        'avg_trades_per_day': float(n / max(len(daily_pnl), 1)),
        'exit_reasons': dict(exit_reasons),
        'long_trades': sum(1 for t in trades if t.side == 'long'),
        'short_trades': sum(1 for t in trades if t.side == 'short'),
    }


def run_permutation_test(engine_factory, mbo_paths: List[str],
                         predictions_by_date: Dict[str, np.ndarray],
                         n_perms: int = 100) -> Tuple[Dict, float]:
    """
    Run permutation test: compare actual model vs random directions.

    Returns:
        (model_metrics, p_value)
    """
    # Run actual model
    all_trades = []
    for mbo_path, date_key in _match_mbo_preds(mbo_paths, predictions_by_date):
        engine = engine_factory()
        preds = predictions_by_date[date_key]
        trades = engine.run_day(mbo_path, preds)
        all_trades.extend(trades)

    model_metrics = compute_metrics(all_trades, "MODEL")
    model_pnl = model_metrics['net_pnl_ticks']

    # Run random permutations
    random_pnls = []
    rng = np.random.RandomState(42)
    for perm_i in range(n_perms):
        perm_trades = []
        for mbo_path, date_key in _match_mbo_preds(mbo_paths, predictions_by_date):
            engine = engine_factory()
            preds = predictions_by_date[date_key]
            # Randomize directions: flip sign randomly
            random_preds = preds * rng.choice([-1, 1], size=len(preds))
            trades = engine.run_day(mbo_path, random_preds)
            perm_trades.extend(trades)

        perm_pnl = sum(t.pnl_ticks for t in perm_trades)
        random_pnls.append(perm_pnl)

        if (perm_i + 1) % 10 == 0:
            print(f"    Permutation {perm_i + 1}/{n_perms}: random PnL = {perm_pnl:.1f} ticks")

    # p-value: fraction of random trials that beat the model
    random_pnls = np.array(random_pnls)
    p_value = float(np.mean(random_pnls >= model_pnl))

    return model_metrics, p_value


def _match_mbo_preds(mbo_paths, predictions_by_date):
    """Match MBO files to prediction dates."""
    matched = []
    for mbo_path in mbo_paths:
        basename = os.path.basename(mbo_path)
        # Extract date from filename: glbx-mdp3-20260223.mbo.dbn.zst
        date8 = basename.split('-')[2].split('.')[0]
        if date8 in predictions_by_date:
            matched.append((mbo_path, date8))
    return sorted(matched, key=lambda x: x[1])


# =============================================================================
# Data Loading
# =============================================================================

def load_predictions(pred_dir: str) -> Dict[str, np.ndarray]:
    """
    Load per-date OOT predictions.
    Returns dict of date_str -> prediction array (pred_log_ret_1s).
    """
    preds = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, 'oot_*.npz'))):
        d = np.load(f, allow_pickle=True)
        date_str = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        # Use pred_log_ret_1s as the primary signal (strongest IC)
        if 'pred_log_ret_1s' in d:
            preds[date_str] = d['pred_log_ret_1s']
        elif 'predictions' in d:
            preds[date_str] = d['predictions']
    return preds


def find_mbo_files(mbo_dir: str) -> List[str]:
    """Find all MBO .dbn.zst files."""
    return sorted(glob.glob(os.path.join(mbo_dir, 'glbx-mdp3-*.mbo.dbn.zst')))


# =============================================================================
# Main Sweep
# =============================================================================

def run_sweep(mbo_dir: str, pred_dir: str,
              tp_range: range, sl_range: range,
              signal_threshold: float = 0.3,
              hold_seconds: float = 30.0,
              cancel_seconds: float = 15.0,
              max_days: int = None,
              n_perms: int = 100,
              output_path: str = None):
    """
    Sweep TP/SL combinations with permutation testing.
    """
    print("=" * 70)
    print("TICK-LEVEL FIFO REPLAY ENGINE")
    print("=" * 70)

    # Load predictions
    print(f"\nLoading predictions from {pred_dir}...")
    predictions = load_predictions(pred_dir)
    print(f"  Loaded {len(predictions)} dates: {sorted(predictions.keys())[:5]}...")

    # Find MBO files
    mbo_files = find_mbo_files(mbo_dir)
    print(f"  Found {len(mbo_files)} MBO files")

    # Match
    matched_dates = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in predictions:
            matched_dates.append((mbo_path, date8))

    if max_days:
        matched_dates = matched_dates[:max_days]

    print(f"  Matched {len(matched_dates)} dates for simulation")

    if not matched_dates:
        print("ERROR: No matching MBO + prediction dates found!")
        return

    # Sweep configs
    configs = []
    for tp in tp_range:
        for sl in sl_range:
            configs.append((tp, sl))

    print(f"\nSweeping {len(configs)} TP/SL configs: "
          f"TP={list(tp_range)}, SL={list(sl_range)}")
    print(f"Signal threshold: {signal_threshold}")
    print(f"Hold: {hold_seconds}s, Cancel: {cancel_seconds}s")
    print(f"Permutation tests: {n_perms} per config")
    print()

    results = []

    for config_idx, (tp, sl) in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"CONFIG {config_idx+1}/{len(configs)}: TP={tp} SL={sl}")
        print(f"{'='*60}")

        # Run model
        all_trades = []
        for mbo_path, date_key in matched_dates:
            t0 = time.time()
            engine = TickReplayEngine(
                tp_ticks=tp, sl_ticks=sl,
                hold_seconds=hold_seconds,
                signal_threshold=signal_threshold,
                cancel_seconds=cancel_seconds,
            )
            preds = predictions[date_key]
            trades = engine.run_day(mbo_path, preds)
            all_trades.extend(trades)
            elapsed = time.time() - t0
            print(f"  {date_key}: {len(trades)} trades, "
                  f"PnL={sum(t.pnl_ticks for t in trades):.1f}t, "
                  f"({elapsed:.1f}s)")

        model_metrics = compute_metrics(all_trades, f"TP{tp}_SL{sl}")

        # Run permutation test
        print(f"\n  Running {n_perms} permutation trials...")
        rng = np.random.RandomState(42)
        random_pnls = []

        for perm_i in range(n_perms):
            perm_trades = []
            for mbo_path, date_key in matched_dates:
                engine = TickReplayEngine(
                    tp_ticks=tp, sl_ticks=sl,
                    hold_seconds=hold_seconds,
                    signal_threshold=signal_threshold,
                    cancel_seconds=cancel_seconds,
                )
                preds = predictions[date_key]
                random_preds = preds * rng.choice([-1, 1], size=len(preds))
                trades = engine.run_day(mbo_path, random_preds)
                perm_trades.extend(trades)

            perm_pnl = sum(t.pnl_ticks for t in perm_trades)
            random_pnls.append(perm_pnl)

            if (perm_i + 1) % 25 == 0:
                print(f"    Perm {perm_i+1}/{n_perms}: "
                      f"random PnL = {perm_pnl:.1f}t")

        random_pnls = np.array(random_pnls)
        p_value = float(np.mean(random_pnls >= model_metrics['net_pnl_ticks']))

        model_metrics['p_value'] = p_value
        model_metrics['random_mean_pnl'] = float(np.mean(random_pnls))
        model_metrics['random_std_pnl'] = float(np.std(random_pnls))
        model_metrics['tp_ticks'] = tp
        model_metrics['sl_ticks'] = sl

        sig = "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""

        print(f"\n  RESULT: TP{tp}_SL{sl}")
        print(f"    Trades:   {model_metrics['n_trades']} "
              f"({model_metrics['long_trades']}L / {model_metrics['short_trades']}S)")
        print(f"    Net PnL:  {model_metrics['net_pnl_ticks']:.1f} ticks "
              f"(${model_metrics['net_pnl_dollars']:.0f})")
        print(f"    WR:       {model_metrics['win_rate']:.1%}")
        print(f"    PF:       {model_metrics['profit_factor']:.2f}")
        print(f"    Sharpe:   {model_metrics['sharpe']:.2f}")
        print(f"    Sortino:  {model_metrics['sortino']:.2f}")
        print(f"    Max DD:   {model_metrics['max_dd_ticks']:.1f} ticks")
        print(f"    Avg MFE:  {model_metrics['avg_mfe']:.1f}t, "
              f"Avg MAE: {model_metrics['avg_mae']:.1f}t")
        print(f"    Exits:    {model_metrics['exit_reasons']}")
        print(f"    p-value:  {p_value:.3f} {sig}")
        print(f"    Random:   mean={np.mean(random_pnls):.1f}t, "
              f"std={np.std(random_pnls):.1f}t")

        results.append(model_metrics)

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Config':<15} {'Trades':>7} {'PnL(t)':>8} {'PnL($)':>8} "
          f"{'WR':>6} {'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'p-val':>7}")
    print("-" * 85)

    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        sig = "***" if r['p_value'] < 0.01 else "**" if r['p_value'] < 0.05 else ""
        print(f"TP{r['tp_ticks']:>2}_SL{r['sl_ticks']:>2}   "
              f"{r['n_trades']:>7} "
              f"{r['net_pnl_ticks']:>8.1f} "
              f"{r['net_pnl_dollars']:>8.0f} "
              f"{r['win_rate']:>5.1%} "
              f"{r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>7.2f} "
              f"{r['sortino']:>8.2f} "
              f"{r['p_value']:>6.3f}{sig}")

    # Save results
    if output_path is None:
        output_path = os.path.join(os.path.dirname(pred_dir),
                                    'tick_replay_results.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Highlight significant configs
    significant = [r for r in results if r['p_value'] < 0.05]
    if significant:
        print(f"\n*** {len(significant)} configs beat random at p < 0.05 ***")
        for r in sorted(significant, key=lambda x: x['sharpe'], reverse=True):
            print(f"  TP{r['tp_ticks']}_SL{r['sl_ticks']}: "
                  f"Sharpe={r['sharpe']:.2f}, PnL={r['net_pnl_ticks']:.0f}t, "
                  f"p={r['p_value']:.3f}")
    else:
        print("\n*** NO configs beat random at p < 0.05 — model has no tradeable edge at these settings ***")

    return results


# =============================================================================
# Quick single-day test mode
# =============================================================================

def run_single_day_test(mbo_path: str, predictions: np.ndarray,
                        tp: int = 4, sl: int = 3,
                        signal_threshold: float = 0.3,
                        hold_seconds: float = 30.0):
    """Quick test on a single day — useful for debugging."""
    print(f"Single-day test: {os.path.basename(mbo_path)}")
    print(f"  Config: TP={tp}, SL={sl}, threshold={signal_threshold}, hold={hold_seconds}s")

    engine = TickReplayEngine(
        tp_ticks=tp, sl_ticks=sl,
        hold_seconds=hold_seconds,
        signal_threshold=signal_threshold,
    )

    t0 = time.time()
    trades = engine.run_day(mbo_path, predictions)
    elapsed = time.time() - t0

    metrics = compute_metrics(trades, f"TP{tp}_SL{sl}")

    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Trades: {metrics['n_trades']}")
    print(f"  Net PnL: {metrics['net_pnl_ticks']:.1f} ticks (${metrics['net_pnl_dollars']:.0f})")
    print(f"  WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
    print(f"  Exits: {metrics['exit_reasons']}")

    if trades:
        print(f"\n  First 5 trades:")
        for t in trades[:5]:
            print(f"    {t.side} @ {t.entry_price:.2f} -> {t.exit_price:.2f} "
                  f"= {t.pnl_ticks:+.2f}t ({t.exit_reason}), "
                  f"fill_lat={t.fill_latency_ns/1e6:.0f}ms")

    return trades, metrics


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
    PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
    OUTPUT = "/home/jupiter/Lvl3Quant/engines/tick_replay_results.json"

    import argparse
    parser = argparse.ArgumentParser(description='Tick-level FIFO replay engine')
    parser.add_argument('--mode', choices=['test', 'sweep'], default='test',
                        help='test = single day quick test; sweep = full TP/SL sweep')
    parser.add_argument('--max-days', type=int, default=None,
                        help='Limit number of days (for testing)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Signal threshold (|pred| must exceed this)')
    parser.add_argument('--hold', type=float, default=30.0,
                        help='Max hold time in seconds')
    parser.add_argument('--cancel', type=float, default=15.0,
                        help='Cancel unfilled orders after N seconds')
    parser.add_argument('--perms', type=int, default=100,
                        help='Number of permutation trials')
    parser.add_argument('--tp-min', type=int, default=2, help='Min TP ticks')
    parser.add_argument('--tp-max', type=int, default=20, help='Max TP ticks')
    parser.add_argument('--sl-min', type=int, default=1, help='Min SL ticks')
    parser.add_argument('--sl-max', type=int, default=10, help='Max SL ticks')
    args = parser.parse_args()

    if args.mode == 'test':
        # Quick single-day test
        predictions = load_predictions(PRED_DIR)
        mbo_files = find_mbo_files(MBO_DIR)

        # Find first matching date
        for mbo_path in mbo_files:
            date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
            if date8 in predictions:
                print(f"Testing on {date8} with {len(predictions[date8])} predictions")
                trades, metrics = run_single_day_test(
                    mbo_path, predictions[date8],
                    tp=4, sl=3,
                    signal_threshold=args.threshold,
                    hold_seconds=args.hold,
                )
                break

    elif args.mode == 'sweep':
        results = run_sweep(
            mbo_dir=MBO_DIR,
            pred_dir=PRED_DIR,
            tp_range=range(args.tp_min, args.tp_max + 1, 2),  # Step by 2 for speed
            sl_range=range(args.sl_min, args.sl_max + 1, 2),
            signal_threshold=args.threshold,
            hold_seconds=args.hold,
            cancel_seconds=args.cancel,
            max_days=args.max_days,
            n_perms=args.perms,
            output_path=OUTPUT,
        )
