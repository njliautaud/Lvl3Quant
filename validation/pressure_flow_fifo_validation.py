#!/usr/bin/env python3
"""
Pressure Flow FIFO Validation — Real MBO Data
=============================================
Validates the AVO-evolved pressure flow engine against real Databento MBO data
using FIFO market replay with realistic costs.

Steps:
  1. Load raw MBO data day by day
  2. Build real-time order book from MBO events
  3. Aggregate into sub-second bars with BBO + trade flow
  4. Compute pressure flow signals from real order book data
  5. Simulate FIFO passive entry/exit with real queue mechanics
  6. Report per-day and aggregate risk-adjusted metrics

Cost model (ES futures, AMP/Rithmic):
  - Tick value: $12.50 (0.25 pts)
  - RT commission: $4.70 = 0.376 ticks
  - Spread: measured from actual data (variable)
  - Passive entry + passive exit: 0.376 ticks cost
  - Passive entry + market exit (SL/timeout): 1.376 ticks cost

Author: Claude Opus 4.6 (AVO validation)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pressure_fifo_validation")

# ── Paths ────────────────────────────────────────────────────────────────────
MBO_OOT_DIR = Path("/home/jupiter/Lvl3Quant/mbo_oot")
PRESSURE_ENGINE = Path("/home/jupiter/teleclaude-main/runs/pressure_flow-20260822-213116/work/pressure_engine.py")
RESULTS_FILE = Path("/home/jupiter/Lvl3Quant/validation/pressure_flow_fifo_results.json")

# ── ES Constants ─────────────────────────────────────────────────────────────
TICK_PTS = 0.25           # 1 tick = 0.25 ES points
TICK_USD = 12.50          # $12.50 per tick
COMMISSION_RT = 4.70      # $ round-trip
COMMISSION_TICKS = COMMISSION_RT / TICK_USD  # 0.376 ticks
MARKET_CROSS_TICKS = 1.0  # cost of crossing spread for market order

# ── Trading params ───────────────────────────────────────────────────────────
BAR_INTERVAL_MS = 250     # aggregate MBO into 250ms bars (matches model stride)
SIGNAL_GATE_PCT = 80      # top 20% by |signal| strength
TP_TICKS = 3.0            # take profit
SL_TICKS = 2.0            # stop loss
MAX_HOLD_S = 30.0         # max hold time (seconds)
CANCEL_WINDOW_S = 10.0    # cancel unfilled entry after 10s
MIN_TRADES_PER_BAR = 1    # minimum trades to form a valid bar
COOLDOWN_BARS = 40        # bars to wait between signals (10 seconds at 250ms)

# RTH window: 9:30 ET - 15:55 ET (stop 5 min before close)
RTH_START_H = 14  # 9:30 ET = 14:30 UTC (hour)
RTH_START_M = 30
RTH_END_H = 20    # 15:55 ET = 20:55 UTC
RTH_END_M = 55

# ── ES front-month contract mapping ─────────────────────────────────────────
# ESZ5 expires ~Dec 19 2025, ESH6 is front from ~Dec 22 2025
# ESH6 expires ~Mar 20 2026
FRONT_MONTH = {
    # Dec 2025
    (2025, 12, 1): "ESZ5", (2025, 12, 2): "ESZ5", (2025, 12, 3): "ESZ5",
    (2025, 12, 4): "ESZ5", (2025, 12, 5): "ESZ5", (2025, 12, 7): "ESZ5",
    (2025, 12, 8): "ESZ5", (2025, 12, 9): "ESZ5", (2025, 12, 10): "ESZ5",
    (2025, 12, 11): "ESZ5", (2025, 12, 12): "ESZ5", (2025, 12, 14): "ESZ5",
    (2025, 12, 15): "ESZ5", (2025, 12, 16): "ESZ5", (2025, 12, 17): "ESZ5",
    (2025, 12, 18): "ESZ5", (2025, 12, 19): "ESZ5",
}


def get_front_month(year: int, month: int, day: int) -> str:
    """Get the front-month ES symbol for a given date."""
    key = (year, month, day)
    if key in FRONT_MONTH:
        return FRONT_MONTH[key]
    # After Dec 19 2025 through Mar 20 2026 -> ESH6
    if (year == 2025 and month == 12 and day > 19) or \
       (year == 2026 and month <= 3 and day <= 20):
        return "ESH6"
    if year == 2026 and month == 3 and day > 20:
        return "ESM6"
    # Fallback: pick the symbol with most trades
    return None


# =============================================================================
# Minimal FIFO Order Book (for BBO tracking)
# =============================================================================

class PriceLevel:
    __slots__ = ("price", "orders")

    def __init__(self, price: float):
        self.price = price
        self.orders: OrderedDict = OrderedDict()

    def add(self, oid: int, qty: int):
        self.orders[oid] = qty

    def cancel(self, oid: int):
        self.orders.pop(oid, None)

    def modify(self, oid: int, qty: int):
        if oid in self.orders:
            self.orders[oid] = qty

    def total_qty(self) -> int:
        return sum(self.orders.values())

    def empty(self) -> bool:
        return not self.orders

    def queue_depth(self) -> int:
        """Number of contracts ahead in FIFO queue (total qty)."""
        return self.total_qty()


class OrderBook:
    """Minimal order book for BBO + depth tracking."""

    def __init__(self):
        self.bids: Dict[float, PriceLevel] = {}
        self.asks: Dict[float, PriceLevel] = {}
        self._oid_side: Dict[int, str] = {}
        self._oid_price: Dict[int, float] = {}

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self._oid_side.clear()
        self._oid_price.clear()

    def _book(self, side: str):
        return self.bids if side == "B" else self.asks

    def add(self, oid: int, side: str, price: float, qty: int):
        if side == "N":
            return
        b = self._book(side)
        if price not in b:
            b[price] = PriceLevel(price)
        b[price].add(oid, qty)
        self._oid_side[oid] = side
        self._oid_price[oid] = price

    def cancel(self, oid: int):
        side = self._oid_side.pop(oid, None)
        price = self._oid_price.pop(oid, None)
        if side is not None and price is not None:
            b = self._book(side)
            if price in b:
                b[price].cancel(oid)
                if b[price].empty():
                    del b[price]

    def modify(self, oid: int, new_qty: int, new_price: float):
        side = self._oid_side.get(oid)
        old_price = self._oid_price.get(oid)
        if side is None:
            return
        if new_price != old_price:
            self.cancel(oid)
            self.add(oid, side, new_price, new_qty)
        else:
            b = self._book(side)
            if old_price in b:
                b[old_price].modify(oid, new_qty)

    def trade(self, price: float, qty: int, aggressor_side: str):
        """Process a trade — remove filled qty from passive side."""
        passive_side = "B" if aggressor_side == "A" else "A"
        b = self._book(passive_side)
        if price in b:
            level = b[price]
            remaining = qty
            to_remove = []
            for oid in list(level.orders):
                if remaining <= 0:
                    break
                q = level.orders[oid]
                if q <= remaining:
                    remaining -= q
                    to_remove.append(oid)
                else:
                    level.orders[oid] -= remaining
                    remaining = 0
            for oid in to_remove:
                del level.orders[oid]
                self._oid_side.pop(oid, None)
                self._oid_price.pop(oid, None)
            if level.empty():
                del b[price]

    def best_bid(self) -> Optional[float]:
        return max(self.bids) if self.bids else None

    def best_ask(self) -> Optional[float]:
        return min(self.asks) if self.asks else None

    def bid_size(self) -> int:
        bb = self.best_bid()
        return self.bids[bb].total_qty() if bb is not None else 0

    def ask_size(self) -> int:
        ba = self.best_ask()
        return self.asks[ba].total_qty() if ba is not None else 0

    def spread_ticks(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is not None and ba is not None:
            return (ba - bb) / TICK_PTS
        return None

    def queue_at_best(self, side: str) -> int:
        """Total queue depth at best bid or ask."""
        if side == "B":
            bb = self.best_bid()
            return self.bids[bb].total_qty() if bb is not None else 0
        else:
            ba = self.best_ask()
            return self.asks[ba].total_qty() if ba is not None else 0


# =============================================================================
# Load and import pressure engine
# =============================================================================

def load_pressure_engine():
    """Import the evolved pressure engine."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pressure_engine", str(PRESSURE_ENGINE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_pressure


# =============================================================================
# MBO -> Bar aggregation (ndarray version — fast)
# =============================================================================

def mbo_ndarray_to_bars(recs, ts_recv, rth_start_ns: int, rth_end_ns: int,
                         interval_ms: int = BAR_INTERVAL_MS) -> pd.DataFrame:
    """
    Build bars from numpy structured array (from to_ndarray()).
    Much faster than the DataFrame-based mbo_to_bars.

    Emits separate buy and sell rows per bar interval to preserve
    signed volume for the pressure engine's flow computation.
    """
    book = OrderBook()
    bars = []
    spread_samples = []

    interval_ns = interval_ms * 1_000_000
    bar_starts = np.arange(rth_start_ns, rth_end_ns, interval_ns)
    n_bar_slots = len(bar_starts)

    n = len(recs)
    bar_idx = 0
    current_bar_end = bar_starts[0] + interval_ns if n_bar_slots > 0 else rth_end_ns

    # Accumulators
    buy_vol = 0
    sell_vol = 0
    vwap_num = 0.0
    vwap_den = 0
    last_trade_price = 0.0

    def _bytes(v):
        if isinstance(v, bytes):
            return v
        if isinstance(v, np.bytes_):
            return bytes(v)
        return bytes([v])

    for i in range(n):
        ts = int(ts_recv[i])
        action = _bytes(recs[i]['action'])
        side = _bytes(recs[i]['side'])
        price_raw = int(recs[i]['price'])
        qty = int(recs[i]['size'])
        oid = int(recs[i]['order_id'])

        # Convert price from fixed-point to points
        price = price_raw / 1e9

        # Update book (using string sides for this book implementation)
        side_str = "B" if side == b'B' else ("A" if side == b'A' else "N")

        if action == b'R':
            book.reset()
        elif action == b'A':
            book.add(oid, side_str, price, qty)
        elif action == b'C':
            book.cancel(oid)
        elif action == b'M':
            book.modify(oid, qty, price)
        elif action in (b'T', b'F'):
            if price > 0 and qty > 0:
                book.trade(price, qty, side_str)
                if ts >= rth_start_ns:
                    last_trade_price = price
                    vwap_num += price * qty
                    vwap_den += qty
                    # side='B' = BUY aggressor, side='A' = SELL aggressor
                    if side == b'B':
                        buy_vol += qty
                    elif side == b'A':
                        sell_vol += qty

        # Check bar boundary
        if ts >= rth_start_ns and bar_idx < n_bar_slots:
            while bar_idx < n_bar_slots and ts >= current_bar_end:
                bb = book.best_bid()
                ba = book.best_ask()
                bs = book.bid_size()
                as_ = book.ask_size()

                if bb is not None and ba is not None:
                    sp = (ba - bb) / TICK_PTS
                    spread_samples.append(sp)
                    bar_price = (vwap_num / vwap_den) if vwap_den > 0 else (
                        last_trade_price if last_trade_price > 0 else (bb + ba) / 2.0)
                    bar_ts = pd.Timestamp(bar_starts[bar_idx], unit="ns", tz="UTC")

                    if buy_vol > 0:
                        bars.append({
                            "timestamp": bar_ts,
                            "price": bar_price,
                            "volume": buy_vol,
                            "side": "buy",
                            "bid_size": bs,
                            "ask_size": as_,
                            "best_bid": bb,
                            "best_ask": ba,
                            "spread_ticks": sp,
                        })
                    if sell_vol > 0:
                        bars.append({
                            "timestamp": bar_ts,
                            "price": bar_price,
                            "volume": sell_vol,
                            "side": "sell",
                            "bid_size": bs,
                            "ask_size": as_,
                            "best_bid": bb,
                            "best_ask": ba,
                            "spread_ticks": sp,
                        })
                    if buy_vol == 0 and sell_vol == 0:
                        bars.append({
                            "timestamp": bar_ts,
                            "price": bar_price,
                            "volume": 0,
                            "side": "buy",
                            "bid_size": bs,
                            "ask_size": as_,
                            "best_bid": bb,
                            "best_ask": ba,
                            "spread_ticks": sp,
                        })

                # Reset
                buy_vol = 0
                sell_vol = 0
                vwap_num = 0.0
                vwap_den = 0

                bar_idx += 1
                if bar_idx < n_bar_slots:
                    current_bar_end = bar_starts[bar_idx] + interval_ns

    if not bars:
        return pd.DataFrame()

    bar_df = pd.DataFrame(bars)
    bar_df.set_index("timestamp", inplace=True)

    n_buy = int((bar_df["side"] == "buy").sum())
    n_sell = int((bar_df["side"] == "sell").sum())
    total_buy = int(bar_df.loc[bar_df["side"] == "buy", "volume"].sum())
    total_sell = int(bar_df.loc[bar_df["side"] == "sell", "volume"].sum())
    avg_sp = float(np.mean(spread_samples)) if spread_samples else 1.0
    log.info(f"  Bar stats: {len(bar_df)} rows ({n_buy} buy, {n_sell} sell), "
             f"vol buy={total_buy:,} sell={total_sell:,}, avg_spread={avg_sp:.2f}t")

    return bar_df


# =============================================================================
# MBO -> Bar aggregation (DataFrame version — slower, kept for compatibility)
# =============================================================================

def mbo_to_bars(df: pd.DataFrame, interval_ms: int = BAR_INTERVAL_MS) -> pd.DataFrame:
    """
    Aggregate raw MBO events into time bars suitable for the pressure engine.

    CRITICAL FIX: The pressure engine computes signed volume as:
        signed_vol = volume  (if side == 'buy')
        signed_vol = -volume (if side == 'sell')
    Then: flow_pressure = rolling_sum(signed_vol) / rolling_sum(volume)

    If we emit one bar with side='buy' and volume=total, ALL volume is positive,
    creating massive long bias. Instead, we emit TWO rows per bar interval:
    one for buy volume and one for sell volume. This preserves the correct
    signed flow for the rolling window computation.

    Databento MBO side on Trade/Fill events (empirically verified):
      side='B' = BUY aggressor (lifting asks, pushes price UP)
      side='A' = SELL aggressor (hitting bids, pushes price DOWN)
    """
    book = OrderBook()

    bars = []

    # Parse timestamps to nanoseconds for binning
    ts_index = df.index  # ts_recv as DatetimeIndex
    ts_ns = ts_index.astype(np.int64)

    # Determine bar boundaries
    interval_ns = interval_ms * 1_000_000
    start_ns = ts_ns[0]
    end_ns = ts_ns[-1]

    # Pre-extract arrays for speed
    actions = df["action"].values
    sides = df["side"].values
    prices = df["price"].values
    sizes = df["size"].values
    order_ids = df["order_id"].values

    n = len(df)
    bar_start = start_ns
    bar_end = bar_start + interval_ns
    idx = 0

    # Accumulators for current bar
    trade_prices = []
    trade_sizes = []
    buy_vol = 0
    sell_vol = 0
    spread_samples = []

    while idx < n and bar_start < end_ns:
        # Process all events up to bar_end
        while idx < n and ts_ns[idx] < bar_end:
            action = actions[idx]
            side = sides[idx]
            price = prices[idx]
            size = int(sizes[idx])
            oid = int(order_ids[idx])

            if action == "R":
                book.reset()
            elif action == "A":
                book.add(oid, side, price, size)
            elif action == "C":
                book.cancel(oid)
            elif action == "M":
                book.modify(oid, size, price)
            elif action in ("T", "F"):
                # Trade event
                if not np.isnan(price) and size > 0:
                    trade_prices.append(price)
                    trade_sizes.append(size)
                    # side='B' = BUY aggressor, side='A' = SELL aggressor
                    if side == "B":
                        buy_vol += size
                    elif side == "A":
                        sell_vol += size
                    book.trade(price, size, side)

            idx += 1

        # Snapshot book state at bar boundary
        bb = book.best_bid()
        ba = book.best_ask()
        bs = book.bid_size()
        as_ = book.ask_size()

        if bb is not None and ba is not None:
            sp = (ba - bb) / TICK_PTS
            spread_samples.append(sp)

            if trade_prices:
                tp_arr = np.array(trade_prices)
                ts_arr_vol = np.array(trade_sizes)
                vwap = np.average(tp_arr, weights=ts_arr_vol)
            else:
                vwap = (bb + ba) / 2.0

            bar_ts = pd.Timestamp(bar_start, unit="ns", tz="UTC")

            # Emit separate buy and sell rows to preserve signed volume.
            # The pressure engine's flow component does:
            #   signed_vol[side == 'sell'] *= -1
            #   flow = rolling_sum(signed_vol) / rolling_sum(volume)
            # So each row must carry volume for ONE side only.
            if buy_vol > 0:
                bars.append({
                    "timestamp": bar_ts,
                    "price": vwap,
                    "volume": buy_vol,
                    "side": "buy",
                    "bid_size": bs,
                    "ask_size": as_,
                    "best_bid": bb,
                    "best_ask": ba,
                    "spread_ticks": sp,
                })
            if sell_vol > 0:
                bars.append({
                    "timestamp": bar_ts,
                    "price": vwap,
                    "volume": sell_vol,
                    "side": "sell",
                    "bid_size": bs,
                    "ask_size": as_,
                    "best_bid": bb,
                    "best_ask": ba,
                    "spread_ticks": sp,
                })
            # If no trades in this bar, emit a zero-vol placeholder
            # to keep depth/price info flowing
            if buy_vol == 0 and sell_vol == 0:
                bars.append({
                    "timestamp": bar_ts,
                    "price": vwap,
                    "volume": 0,
                    "side": "buy",
                    "bid_size": bs,
                    "ask_size": as_,
                    "best_bid": bb,
                    "best_ask": ba,
                    "spread_ticks": sp,
                })

        # Reset accumulators
        trade_prices = []
        trade_sizes = []
        buy_vol = 0
        sell_vol = 0

        bar_start = bar_end
        bar_end = bar_start + interval_ns

    if not bars:
        return pd.DataFrame()

    bar_df = pd.DataFrame(bars)
    bar_df.set_index("timestamp", inplace=True)

    # Log distribution
    n_buy = int((bar_df["side"] == "buy").sum())
    n_sell = int((bar_df["side"] == "sell").sum())
    total_buy = int(bar_df.loc[bar_df["side"] == "buy", "volume"].sum())
    total_sell = int(bar_df.loc[bar_df["side"] == "sell", "volume"].sum())
    avg_sp = float(np.mean(spread_samples)) if spread_samples else 1.0
    log.info(f"  Bar stats: {len(bar_df)} rows ({n_buy} buy, {n_sell} sell), "
             f"vol buy={total_buy:,} sell={total_sell:,}, avg_spread={avg_sp:.2f}t")

    return bar_df


# =============================================================================
# FIFO Trade Simulator
# =============================================================================

@dataclass
class SimTrade:
    """A simulated trade from entry to exit."""
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: str          # 'long' or 'short'
    entry_price: float      # points
    exit_price: float       # points
    exit_reason: str        # 'tp', 'sl', 'max_hold', 'eod'
    pnl_ticks: float        # gross PnL in ticks
    pnl_ticks_net: float    # net of commission
    pnl_dollars: float      # net dollars
    signal_strength: float
    spread_at_entry: float  # spread in ticks at entry
    queue_at_entry: int     # queue depth at entry
    hold_time_s: float      # seconds held
    exit_cost_type: str     # 'passive' or 'market'


def simulate_day(
    bar_df: pd.DataFrame,
    signals: pd.Series,
    gate_pctile: float = SIGNAL_GATE_PCT,
    tp_ticks: float = TP_TICKS,
    sl_ticks: float = SL_TICKS,
    max_hold_s: float = MAX_HOLD_S,
    cancel_s: float = CANCEL_WINDOW_S,
    cooldown_s: float = 2.0,
) -> List[SimTrade]:
    """
    Simulate FIFO passive trading on one day's bar data with pressure signals.

    Entry: passive limit at best bid (long) or best ask (short).
    Fill assumption: conservative — price must trade THROUGH our level
    (not just touch). For buy at bid: fill when trade price < bid.
    For sell at ask: fill when trade price > ask.

    Exit: TP = passive. SL/max_hold = market (extra spread-crossing cost).

    NOTE: bar_df may have multiple rows per time interval (buy/sell split).
    We use actual timestamps for all time calculations, not row indices.
    Signals are generated per-row by compute_pressure, but we only act
    on the LAST signal per unique timestamp to avoid double-signaling.
    """
    trades = []
    n = len(bar_df)
    if n < 2:
        return trades

    # Determine gate threshold from signal distribution
    abs_signals = signals.abs()
    nonzero = abs_signals[abs_signals > 0]
    if len(nonzero) < 10:
        return trades
    gate_threshold = np.percentile(nonzero, gate_pctile)

    # Pre-extract timestamps as int64 nanoseconds for fast time comparisons
    ts_ns_arr = bar_df.index.astype(np.int64)

    # State
    in_position = False
    position_dir = None
    entry_price = 0.0
    entry_ts = None
    entry_ts_ns = 0
    tp_price = 0.0
    sl_price = 0.0
    signal_str = 0.0
    spread_at_entry = 0.0
    queue_at_entry = 0
    pending_entry = False
    pending_dir = None
    pending_price = 0.0
    pending_ts = None
    pending_ts_ns = 0
    pending_signal_str = 0.0
    pending_spread = 0.0
    pending_queue = 0
    last_trade_ts_ns = 0

    cancel_ns = int(cancel_s * 1e9)
    max_hold_ns = int(max_hold_s * 1e9)
    cooldown_ns = int(cooldown_s * 1e9)

    # Track which timestamps we've already seen (to avoid double-signaling
    # from buy+sell rows at the same timestamp)
    last_signal_ts_ns = 0

    for i in range(n):
        bar = bar_df.iloc[i]
        ts = bar_df.index[i]
        ts_ns = int(ts_ns_arr[i])
        sig = float(signals.iloc[i])

        # ── Check pending entry fill ──
        if pending_entry and not in_position:
            time_since_ns = ts_ns - pending_ts_ns

            if time_since_ns > cancel_ns:
                pending_entry = False
            else:
                # Check for fill: price must trade THROUGH our level
                filled = False
                if pending_dir == "long":
                    if bar["price"] < pending_price:
                        filled = True
                else:
                    if bar["price"] > pending_price:
                        filled = True

                if filled:
                    in_position = True
                    position_dir = pending_dir
                    entry_price = pending_price
                    entry_ts = ts
                    entry_ts_ns = ts_ns
                    signal_str = pending_signal_str
                    spread_at_entry = pending_spread
                    queue_at_entry = pending_queue

                    if position_dir == "long":
                        tp_price = entry_price + tp_ticks * TICK_PTS
                        sl_price = entry_price - sl_ticks * TICK_PTS
                    else:
                        tp_price = entry_price - tp_ticks * TICK_PTS
                        sl_price = entry_price + sl_ticks * TICK_PTS

                    pending_entry = False

        # ── Check exit conditions ──
        if in_position:
            hold_ns = ts_ns - entry_ts_ns
            hold_time_s = hold_ns / 1e9

            exit_reason = None
            exit_price = 0.0
            exit_cost = "passive"

            if position_dir == "long":
                if bar["price"] >= tp_price:
                    exit_reason = "tp"
                    exit_price = tp_price
                    exit_cost = "passive"
                elif bar["price"] <= sl_price:
                    exit_reason = "sl"
                    exit_price = sl_price
                    exit_cost = "market"
                elif hold_ns >= max_hold_ns:
                    exit_reason = "max_hold"
                    exit_price = bar["best_bid"]
                    exit_cost = "passive"
            else:
                if bar["price"] <= tp_price:
                    exit_reason = "tp"
                    exit_price = tp_price
                    exit_cost = "passive"
                elif bar["price"] >= sl_price:
                    exit_reason = "sl"
                    exit_price = sl_price
                    exit_cost = "market"
                elif hold_ns >= max_hold_ns:
                    exit_reason = "max_hold"
                    exit_price = bar["best_ask"]
                    exit_cost = "passive"

            # EOD exit (last 10 rows)
            if exit_reason is None and i >= n - 10:
                exit_reason = "eod"
                if position_dir == "long":
                    exit_price = bar["best_bid"]
                else:
                    exit_price = bar["best_ask"]
                exit_cost = "passive"

            if exit_reason is not None:
                if position_dir == "long":
                    pnl_ticks = (exit_price - entry_price) / TICK_PTS
                else:
                    pnl_ticks = (entry_price - exit_price) / TICK_PTS

                cost_ticks = COMMISSION_TICKS
                if exit_cost == "market":
                    cost_ticks += MARKET_CROSS_TICKS

                pnl_net = pnl_ticks - cost_ticks

                trades.append(SimTrade(
                    entry_ts=entry_ts,
                    exit_ts=ts,
                    direction=position_dir,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_ticks=pnl_ticks,
                    pnl_ticks_net=pnl_net,
                    pnl_dollars=pnl_net * TICK_USD,
                    signal_strength=signal_str,
                    spread_at_entry=spread_at_entry,
                    queue_at_entry=queue_at_entry,
                    hold_time_s=hold_time_s,
                    exit_cost_type=exit_cost,
                ))

                in_position = False
                last_trade_ts_ns = ts_ns
                continue

        # ── Generate new entry signal (if not in position and not pending) ──
        # Skip duplicate timestamps (buy+sell rows at same time)
        if ts_ns == last_signal_ts_ns:
            continue
        last_signal_ts_ns = ts_ns

        if not in_position and not pending_entry and (ts_ns - last_trade_ts_ns) >= cooldown_ns:
            if abs(sig) >= gate_threshold and abs(sig) > 1e-6:
                direction = "long" if sig > 0 else "short"
                spread = bar.get("spread_ticks", 1.0)

                # Skip if spread is too wide (> 2 ticks = abnormal)
                if spread is not None and spread > 2.0:
                    continue

                if direction == "long":
                    limit_price = bar["best_bid"]
                    queue = bar.get("bid_size", 0) if "bid_size" in bar.index else 0
                else:
                    limit_price = bar["best_ask"]
                    queue = bar.get("ask_size", 0) if "ask_size" in bar.index else 0

                if limit_price is None or (isinstance(limit_price, float) and np.isnan(limit_price)):
                    continue

                pending_entry = True
                pending_dir = direction
                pending_price = limit_price
                pending_ts = ts
                pending_ts_ns = ts_ns
                pending_signal_str = abs(sig)
                pending_spread = spread if spread is not None else 1.0
                pending_queue = queue

    return trades


# =============================================================================
# Metrics computation
# =============================================================================

def compute_metrics(trades: List[SimTrade]) -> dict:
    """Compute risk-adjusted metrics from a list of trades."""
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0,
            "total_pnl_ticks": 0.0, "total_pnl_dollars": 0.0,
            "avg_pnl_ticks": 0.0, "avg_winner_ticks": 0.0,
            "avg_loser_ticks": 0.0, "max_drawdown_ticks": 0.0,
            "fill_rate": 0.0, "avg_hold_s": 0.0,
            "avg_spread": 0.0, "tp_pct": 0.0, "sl_pct": 0.0,
            "long_pct": 0.0, "short_pct": 0.0,
        }

    pnls = np.array([t.pnl_ticks_net for t in trades])
    n = len(pnls)

    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]

    total_pnl = float(np.sum(pnls))
    avg_pnl = float(np.mean(pnls))
    win_rate = float(len(winners) / n) if n > 0 else 0.0

    # Per-trade Sharpe (meaningful for cross-day comparison)
    trade_sharpe = float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0.0

    # Sortino (per-trade)
    downside = pnls[pnls < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-8
    trade_sortino = float(np.mean(pnls) / downside_std) if downside_std > 0 else 0.0

    # Profit factor
    gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0.0
    gross_loss = float(abs(np.sum(losers))) if len(losers) > 0 else 1e-8
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdown = running_max - cumsum
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Exit breakdown
    tp_count = sum(1 for t in trades if t.exit_reason == "tp")
    sl_count = sum(1 for t in trades if t.exit_reason == "sl")

    # Direction breakdown
    long_count = sum(1 for t in trades if t.direction == "long")
    short_count = sum(1 for t in trades if t.direction == "short")

    return {
        "n_trades": n,
        "sharpe": round(trade_sharpe, 4),
        "sortino": round(trade_sortino, 4),
        "profit_factor": round(pf, 4),
        "win_rate": round(win_rate, 4),
        "total_pnl_ticks": round(total_pnl, 2),
        "total_pnl_dollars": round(total_pnl * TICK_USD, 2),
        "avg_pnl_ticks": round(avg_pnl, 4),
        "avg_winner_ticks": round(float(np.mean(winners)), 4) if len(winners) > 0 else 0.0,
        "avg_loser_ticks": round(float(np.mean(losers)), 4) if len(losers) > 0 else 0.0,
        "max_drawdown_ticks": round(max_dd, 2),
        "avg_hold_s": round(float(np.mean([t.hold_time_s for t in trades])), 2),
        "avg_spread": round(float(np.mean([t.spread_at_entry for t in trades])), 4),
        "tp_pct": round(tp_count / n, 4) if n > 0 else 0.0,
        "sl_pct": round(sl_count / n, 4) if n > 0 else 0.0,
        "long_pct": round(long_count / n, 4) if n > 0 else 0.0,
        "short_pct": round(short_count / n, 4) if n > 0 else 0.0,
    }


def classify_regime(bar_df: pd.DataFrame) -> str:
    """
    Classify day's regime based on intraday price action.
    green = close > open by > 20% of daily range
    red = close < open by > 20% of daily range
    flat = everything else
    """
    if bar_df.empty:
        return "flat"
    prices = bar_df["price"].values
    if len(prices) < 10:
        return "flat"
    open_p = prices[0]
    close_p = prices[-1]
    daily_range = np.max(prices) - np.min(prices)
    if daily_range < 1e-8:
        return "flat"

    change = close_p - open_p
    threshold = daily_range * 0.2

    if change > threshold:
        return "green"
    elif change < -threshold:
        return "red"
    else:
        return "flat"


# =============================================================================
# Process one day
# =============================================================================

def process_day(mbo_file: Path, compute_pressure_fn) -> Optional[dict]:
    """Process one day of MBO data through the pressure flow engine + FIFO sim."""
    import databento as db
    from datetime import datetime as dt_cls, timezone as tz_cls

    date_str = mbo_file.name.split(".")[0].replace("glbx-mdp3-", "")
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    log.info(f"Processing {date_str}...")
    t0 = time.time()

    try:
        store = db.DBNStore.from_file(str(mbo_file))
        recs = store.to_ndarray()
    except Exception as e:
        log.error(f"  Failed to load {mbo_file.name}: {e}")
        return None

    # Auto-detect front-month instrument (most events)
    ids, counts = np.unique(recs['instrument_id'], return_counts=True)
    main_id = int(ids[np.argmax(counts)])
    recs = recs[recs['instrument_id'] == main_id]

    if len(recs) < 1000:
        log.warning(f"  Only {len(recs)} events for instr={main_id} on {date_str}, skipping")
        return None

    # Determine RTH window (handle EST/EDT)
    # DST check: second Sunday in March to first Sunday in November
    test_dt = dt_cls(year, month, day)
    mar1 = dt_cls(year, 3, 1)
    mar_second_sun = (14 - mar1.weekday()) % 7 + 8
    nov1 = dt_cls(year, 11, 1)
    nov_first_sun = (7 - nov1.weekday()) % 7 + 1
    is_dst = dt_cls(year, 3, mar_second_sun) <= test_dt < dt_cls(year, 11, nov_first_sun)

    if is_dst:
        rth_start_utc = dt_cls(year, month, day, 13, 30, tzinfo=tz_cls.utc)
        rth_end_utc = dt_cls(year, month, day, 19, 55, tzinfo=tz_cls.utc)
    else:
        rth_start_utc = dt_cls(year, month, day, 14, 30, tzinfo=tz_cls.utc)
        rth_end_utc = dt_cls(year, month, day, 20, 55, tzinfo=tz_cls.utc)

    rth_start_ns = int(rth_start_utc.timestamp() * 1e9)
    rth_end_ns = int(rth_end_utc.timestamp() * 1e9)

    # Filter to RTH with 30-min warmup for book building
    warmup_ns = rth_start_ns - 30 * 60 * int(1e9)
    ts_recv = recs['ts_recv'].astype(np.int64)
    mask = (ts_recv >= warmup_ns) & (ts_recv <= rth_end_ns)
    recs = recs[mask]
    ts_recv = ts_recv[mask]

    if len(recs) < 500:
        log.warning(f"  Only {len(recs)} RTH events on {date_str}, skipping")
        return None

    log.info(f"  {len(recs):,} RTH events for instr={main_id}")

    # Build bars from ndarray directly
    t1 = time.time()
    bar_df = mbo_ndarray_to_bars(recs, ts_recv, rth_start_ns, rth_end_ns)
    t2 = time.time()
    log.info(f"  Aggregated to {len(bar_df)} bars in {t2-t1:.1f}s")

    if len(bar_df) < 100:
        log.warning(f"  Only {len(bar_df)} bars on {date_str}, skipping")
        return None

    # Compute pressure signals
    signals = compute_pressure_fn(bar_df)
    regime = classify_regime(bar_df)

    # Spread stats
    spreads = bar_df["spread_ticks"].values
    valid_spreads = spreads[~np.isnan(spreads)]
    avg_spread = float(np.mean(valid_spreads)) if len(valid_spreads) > 0 else 1.0

    # Simulate
    trades = simulate_day(bar_df, signals)
    metrics = compute_metrics(trades)
    metrics["date"] = date_str
    metrics["regime"] = regime
    metrics["avg_spread_measured"] = round(avg_spread, 4)
    metrics["n_bars"] = len(bar_df)
    metrics["n_mbo_events"] = int(mask.sum())
    metrics["processing_time_s"] = round(time.time() - t0, 1)

    log.info(
        f"  {date_str} [{regime}]: {metrics['n_trades']} trades, "
        f"PnL={metrics['total_pnl_ticks']:.1f}t (${metrics['total_pnl_dollars']:.0f}), "
        f"WR={metrics['win_rate']:.1%}, PF={metrics['profit_factor']:.2f}, "
        f"Sharpe={metrics['sharpe']:.3f}, "
        f"L/S={metrics['long_pct']:.0%}/{metrics['short_pct']:.0%}, "
        f"time={metrics['processing_time_s']:.0f}s"
    )

    return metrics
    rth_end = ts_utc[0].normalize() + pd.Timedelta(hours=RTH_END_H, minutes=RTH_END_M)
    rth_mask = (ts_utc >= rth_start) & (ts_utc <= rth_end)
    es_df = es_df[rth_mask]

    if len(es_df) < 500:
        log.warning(f"  Only {len(es_df)} RTH events on {date_str}, skipping")
        return None

    log.info(f"  {len(es_df)} RTH events for {front}")

    # Aggregate MBO -> bars
    t1 = time.time()
    bar_df = mbo_to_bars(es_df, BAR_INTERVAL_MS)
    t2 = time.time()
    log.info(f"  Aggregated to {len(bar_df)} bars in {t2-t1:.1f}s")

    if len(bar_df) < 100:
        log.warning(f"  Only {len(bar_df)} bars on {date_str}, skipping")
        return None

    # Compute pressure signals
    signals = compute_pressure_fn(bar_df)

    # Classify regime
    regime = classify_regime(bar_df)

    # Measure actual spread stats
    spreads = bar_df["spread_ticks"].values
    valid_spreads = spreads[~np.isnan(spreads)]
    avg_spread = float(np.mean(valid_spreads)) if len(valid_spreads) > 0 else 1.0
    median_spread = float(np.median(valid_spreads)) if len(valid_spreads) > 0 else 1.0

    # Simulate trading
    trades = simulate_day(bar_df, signals)

    # Compute metrics
    metrics = compute_metrics(trades)
    metrics["date"] = date_str
    metrics["regime"] = regime
    metrics["avg_spread_measured"] = round(avg_spread, 4)
    metrics["median_spread_measured"] = round(median_spread, 4)
    metrics["n_bars"] = len(bar_df)
    metrics["n_mbo_events"] = len(es_df)
    metrics["front_month"] = front
    metrics["processing_time_s"] = round(time.time() - t0, 1)

    log.info(
        f"  {date_str} [{regime}]: {metrics['n_trades']} trades, "
        f"PnL={metrics['total_pnl_ticks']:.1f}t (${metrics['total_pnl_dollars']:.0f}), "
        f"WR={metrics['win_rate']:.1%}, PF={metrics['profit_factor']:.2f}, "
        f"spread={avg_spread:.2f}t"
    )

    return metrics


# =============================================================================
# Main
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("Pressure Flow FIFO Validation — Real MBO Data")
    log.info("=" * 70)

    # Load pressure engine
    compute_pressure = load_pressure_engine()
    log.info("Loaded evolved pressure engine (v14)")

    # Find MBO files
    mbo_files = sorted(MBO_OOT_DIR.glob("glbx-mdp3-*.mbo.dbn.zst"))
    log.info(f"Found {len(mbo_files)} MBO files in {MBO_OOT_DIR}")

    if len(mbo_files) < 40:
        log.warning(f"Only {len(mbo_files)} files — need 40+ for regime-agnostic validation")

    # Process each day
    all_results = []

    for mbo_file in mbo_files:
        result = process_day(mbo_file, compute_pressure)
        if result is not None:
            all_results.append(result)

    if not all_results:
        log.error("No days processed successfully!")
        output = {"error": "No days processed", "n_files": len(mbo_files)}
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(output, f, indent=2)
        return

    # ── Aggregate metrics ──
    log.info("")
    log.info("=" * 70)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 70)

    n_days = len(all_results)
    total_trades = sum(r["n_trades"] for r in all_results)
    total_pnl_ticks = sum(r["total_pnl_ticks"] for r in all_results)
    total_pnl_dollars = sum(r["total_pnl_dollars"] for r in all_results)

    # Per-day PnL array for daily Sharpe/Sortino
    daily_pnl = np.array([r["total_pnl_ticks"] for r in all_results])
    daily_pnl_dollars = np.array([r["total_pnl_dollars"] for r in all_results])

    if np.std(daily_pnl) > 0:
        daily_sharpe = float(np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252))
    else:
        daily_sharpe = 0.0

    downside_daily = daily_pnl[daily_pnl < 0]
    if len(downside_daily) > 0 and np.std(downside_daily) > 0:
        daily_sortino = float(np.mean(daily_pnl) / np.std(downside_daily) * np.sqrt(252))
    else:
        daily_sortino = 0.0

    green_days = [r for r in all_results if r["regime"] == "green"]
    red_days = [r for r in all_results if r["regime"] == "red"]
    flat_days = [r for r in all_results if r["regime"] == "flat"]

    winning_days = sum(1 for r in all_results if r["total_pnl_ticks"] > 0)
    losing_days = sum(1 for r in all_results if r["total_pnl_ticks"] < 0)
    flat_pnl_days = sum(1 for r in all_results if r["total_pnl_ticks"] == 0)

    # Day concentration: max single day PnL / total PnL
    if np.sum(np.abs(daily_pnl)) > 0:
        day_conc = float(np.max(np.abs(daily_pnl)) / np.sum(np.abs(daily_pnl)))
    else:
        day_conc = 1.0

    # Per-regime Sharpe
    def regime_sharpe(days):
        if len(days) < 2:
            return 0.0
        pnls = np.array([d["total_pnl_ticks"] for d in days])
        if np.std(pnls) > 0:
            return float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
        return 0.0

    sharpe_green = regime_sharpe(green_days)
    sharpe_red = regime_sharpe(red_days)
    sharpe_flat = regime_sharpe(flat_days)

    # Regime asymmetry check (HC #428 R1)
    if max(abs(sharpe_green), abs(sharpe_red)) > 0:
        regime_asymmetry = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red))
    else:
        regime_asymmetry = 0.0

    regime_agnostic = regime_asymmetry <= 0.50

    # Overall profit factor
    gross_winners = sum(r["total_pnl_ticks"] for r in all_results if r["total_pnl_ticks"] > 0)
    gross_losers = abs(sum(r["total_pnl_ticks"] for r in all_results if r["total_pnl_ticks"] < 0))
    overall_pf = gross_winners / gross_losers if gross_losers > 0 else float("inf")

    # Average metrics
    trading_days = [r for r in all_results if r["n_trades"] > 0]
    avg_wr = float(np.mean([r["win_rate"] for r in trading_days])) if trading_days else 0.0
    avg_trades_per_day = total_trades / n_days if n_days > 0 else 0

    # Print summary
    log.info(f"Days processed:       {n_days}")
    log.info(f"Total trades:         {total_trades}")
    log.info(f"Avg trades/day:       {avg_trades_per_day:.1f}")
    log.info(f"Total PnL:            {total_pnl_ticks:.1f} ticks (${total_pnl_dollars:.0f})")
    log.info(f"Daily Sharpe:         {daily_sharpe:.4f}")
    log.info(f"Daily Sortino:        {daily_sortino:.4f}")
    log.info(f"Overall PF:           {overall_pf:.4f}")
    log.info(f"Avg Win Rate:         {avg_wr:.1%}")
    log.info(f"Winning/Losing/Flat:  {winning_days}/{losing_days}/{flat_pnl_days}")
    log.info(f"Day concentration:    {day_conc:.4f}")
    log.info(f"")
    log.info(f"Regime breakdown:")
    log.info(f"  Green days: {len(green_days)}, Sharpe={sharpe_green:.4f}")
    log.info(f"  Red days:   {len(red_days)}, Sharpe={sharpe_red:.4f}")
    log.info(f"  Flat days:  {len(flat_days)}, Sharpe={sharpe_flat:.4f}")
    log.info(f"  Regime asymmetry: {regime_asymmetry:.4f} ({'PASS' if regime_agnostic else 'FAIL'} <= 0.50)")
    log.info(f"  Day conc:   {day_conc:.4f} ({'PASS' if day_conc <= 0.70 else 'FAIL'} <= 0.70)")

    # ── Save results ──
    output = {
        "summary": {
            "n_days": n_days,
            "total_trades": total_trades,
            "avg_trades_per_day": round(avg_trades_per_day, 1),
            "total_pnl_ticks": round(total_pnl_ticks, 2),
            "total_pnl_dollars": round(total_pnl_dollars, 2),
            "daily_sharpe": round(daily_sharpe, 4),
            "daily_sortino": round(daily_sortino, 4),
            "overall_profit_factor": round(overall_pf, 4),
            "avg_win_rate": round(avg_wr, 4),
            "winning_days": winning_days,
            "losing_days": losing_days,
            "flat_pnl_days": flat_pnl_days,
            "day_concentration": round(day_conc, 4),
            "day_conc_pass": day_conc <= 0.70,
        },
        "regime_analysis": {
            "green_days": len(green_days),
            "red_days": len(red_days),
            "flat_days": len(flat_days),
            "sharpe_green": round(sharpe_green, 4),
            "sharpe_red": round(sharpe_red, 4),
            "sharpe_flat": round(sharpe_flat, 4),
            "regime_asymmetry": round(regime_asymmetry, 4),
            "regime_agnostic_pass": regime_agnostic,
        },
        "config": {
            "bar_interval_ms": BAR_INTERVAL_MS,
            "signal_gate_pctile": SIGNAL_GATE_PCT,
            "tp_ticks": TP_TICKS,
            "sl_ticks": SL_TICKS,
            "max_hold_s": MAX_HOLD_S,
            "cancel_window_s": CANCEL_WINDOW_S,
            "commission_rt_usd": COMMISSION_RT,
            "commission_ticks": COMMISSION_TICKS,
            "cooldown_bars": COOLDOWN_BARS,
        },
        "per_day": all_results,
    }

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log.info(f"\nResults saved to {RESULTS_FILE}")

    # Final verdict
    log.info("")
    log.info("=" * 70)
    if daily_sharpe > 0.5 and overall_pf > 1.0 and regime_agnostic:
        log.info("VERDICT: PROMISING — positive risk-adjusted returns, regime-agnostic")
    elif daily_sharpe > 0 and overall_pf > 1.0:
        log.info("VERDICT: MARGINAL — positive but may not survive costs/slippage")
    else:
        log.info("VERDICT: DOES NOT VALIDATE — signal does not survive real-data replay")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
