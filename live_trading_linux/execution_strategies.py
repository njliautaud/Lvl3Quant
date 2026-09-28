#!/usr/bin/env python3
"""
execution_strategies.py — Multi-strategy paper trading execution comparison system.

Tests MULTIPLE execution strategies in parallel on the same LGBM smart_v2 signal
stream to determine which strategy maximizes net P&L after costs.

Architecture:
    Signal stream (LGBM smart_v2 predictions)
        -> StrategyRunner (distributes to all strategies)
        -> Strategy 1: Baseline Market Order
        -> Strategy 2: High Confidence Only (Top1%)
        -> Strategy 3: MFE-Informed TP/SL
        -> Strategy 4: Passive Entry (Limit Order)
        -> Strategy 5: Confidence-Scaled Position
        -> Strategy 6: Signal Persistence
        -> Per-strategy: trade log, equity curve, performance stats

Key thresholds (from LGBM smart_v2 OOT analysis):
    Top10%: confidence >= 0.127  (DA=62%, NOT profitable at 2-tick cost)
    Top5%:  confidence >= 0.20   (DA=68%, profitable at 1-tick cost)
    Top1%:  confidence >= 0.37   (DA=73.2%, profitable at 2-tick cost)

NQ Futures: tick=$0.25, point=$20, tick_value=$5.00 (using $12.50 for ES compat)
AMP commission: $4.70 RT = $2.35/side = 0.376 ticks RT

IMPORTANT: PAPER TRADE ONLY. No real orders are ever submitted.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

log = logging.getLogger("exec_strategies")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25
POINT_VALUE = 50.0        # NQ: $20 per point, $5 per tick. ES: $50 per point
TICK_VALUE = 12.50         # $12.50 per tick (NQ tick = 0.25 points)
# AMP Futures: $4.70 round-trip = $2.35 per side
COMMISSION_PER_SIDE = 2.35 # $2.35 per side (AMP $4.70 RT)
# In ticks: $4.70 / $12.50 = 0.376 ticks RT commission

# Confidence thresholds from LGBM smart_v2 OOT concat analysis
CONF_TOP10 = 0.127
CONF_TOP5 = 0.20
CONF_TOP1 = 0.37

# MFE/MAE expectations from OOT analysis (in ticks)
MFE_TOP5 = 2.5   # expected max favorable excursion for Top5%
MFE_TOP1 = 3.85  # expected MFE for Top1%
MAE_TOP5 = 0.8   # expected max adverse excursion for Top5% winners
MAE_TOP1 = 0.5   # expected MAE for Top1% winners


def _json_safe(obj):
    """Convert numpy/special types for JSON serialization."""
    import numpy as np
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
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Tracker (shared across all strategies)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyStats:
    """Comprehensive performance statistics for a single strategy."""
    strategy_name: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    total_commission: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    returns: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)  # cumulative P&L
    trade_log: List[Dict] = field(default_factory=list)
    signals_received: int = 0
    signals_acted: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def avg_pnl_per_trade(self) -> float:
        return self.total_net_pnl / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def avg_pnl_ticks(self) -> float:
        """Average net P&L per trade in ticks."""
        return self.avg_pnl_per_trade / TICK_VALUE if self.total_trades > 0 else 0.0

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
        # Annualize: ~50 trades/day, 252 days/year
        return (mean_ret / ds) * math.sqrt(50 * 252)

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(r for r in self.returns if r > 0)
        gross_losses = abs(sum(r for r in self.returns if r < 0))
        if gross_losses == 0:
            return float('inf') if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    def record_trade(self, trade: Dict):
        """Record a completed round-trip trade."""
        net_pnl = trade["net_pnl"]
        self.total_trades += 1
        self.total_gross_pnl += trade["gross_pnl"]
        self.total_net_pnl += net_pnl
        self.total_commission += trade.get("commission", COMMISSION_PER_SIDE * 2)
        self.returns.append(net_pnl)
        self.equity_curve.append(self.total_net_pnl)
        self.trade_log.append(trade)

        if net_pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        if self.total_net_pnl > self.peak_pnl:
            self.peak_pnl = self.total_net_pnl
        dd = self.peak_pnl - self.total_net_pnl
        if dd > self.max_drawdown:
            self.max_drawdown = dd

    def summary_dict(self) -> Dict:
        """Return summary as a dict for JSON/display."""
        return {
            "strategy": self.strategy_name,
            "trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate * 100, 1),
            "net_pnl": round(self.total_net_pnl, 2),
            "gross_pnl": round(self.total_gross_pnl, 2),
            "commission": round(self.total_commission, 2),
            "avg_pnl_per_trade": round(self.avg_pnl_per_trade, 2),
            "avg_pnl_ticks": round(self.avg_pnl_ticks, 2),
            "sortino": round(self.sortino, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "signals_received": self.signals_received,
            "signals_acted": self.signals_acted,
        }

    def summary_line(self) -> str:
        """One-line summary for comparison table."""
        return (
            f"{self.strategy_name:<28s} | "
            f"Trades:{self.total_trades:>4d} | "
            f"WR:{self.win_rate*100:>5.1f}% | "
            f"Net:${self.total_net_pnl:>+8.2f} | "
            f"Avg:${self.avg_pnl_per_trade:>+6.2f} ({self.avg_pnl_ticks:>+5.2f}t) | "
            f"Sortino:{self.sortino:>6.2f} | "
            f"PF:{self.profit_factor:>5.2f} | "
            f"MaxDD:${self.max_drawdown:>7.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Base Strategy Class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Tracks an open position for a strategy."""
    size: int = 0             # +1 long, -1 short, 0 flat
    side: str = ""            # "LONG" or "SHORT"
    entry_price: float = 0.0
    entry_time: float = 0.0
    entry_prob: float = 0.0
    entry_confidence: float = 0.0
    # For TP/SL strategies
    take_profit: float = 0.0   # price level
    stop_loss: float = 0.0     # price level
    # For limit order strategies
    pending_limit: bool = False
    limit_price: float = 0.0
    limit_side: str = ""
    limit_submit_time: float = 0.0
    limit_timeout: float = 2.0  # seconds


class Strategy(ABC):
    """Base class for execution strategies.

    Each strategy receives the same signal stream and BBO updates, but handles
    entry/exit differently. The strategy tracks its own position and performance.

    Subclasses must implement:
        on_signal(timestamp, prob_up, confidence, bid, ask) -> None
        on_price_update(timestamp, bid, ask) -> None
    """

    def __init__(self, name: str, slippage_ticks: float = 0.0,  # HC #290(C) 2026-05-11: commission only
                 commission_per_side: float = COMMISSION_PER_SIDE):
        self.name = name
        self.slippage_ticks = slippage_ticks
        self.commission_per_side = commission_per_side
        self.position = Position()
        self.stats = StrategyStats(strategy_name=name)
        self._trade_counter = 0

    @abstractmethod
    def on_signal(self, timestamp: float, prob_up: float, confidence: float,
                  bid: float, ask: float) -> None:
        """Called when a new LGBM prediction is available.

        Args:
            timestamp: Unix timestamp (seconds)
            prob_up: P(up) from LGBM, range [0, 1]
            confidence: |P(up) - 0.5|, range [0, 0.5]
            bid: Current best bid price
            ask: Current best ask price
        """
        pass

    @abstractmethod
    def on_price_update(self, timestamp: float, bid: float, ask: float) -> None:
        """Called on every BBO update (for TP/SL monitoring, limit fill checks, etc.).

        Args:
            timestamp: Unix timestamp (seconds)
            bid: Current best bid price
            ask: Current best ask price
        """
        pass

    # ─────── Shared execution helpers ───────

    def _fill_price_market(self, side: str, bid: float, ask: float) -> float:
        """Simulate market order fill with slippage.

        Args:
            side: "LONG" (buy) or "SHORT" (sell)
        """
        slip = self.slippage_ticks * TICK_SIZE
        if side == "LONG":
            return (ask + slip) if ask > 0 else 0.0
        else:
            return (bid - slip) if bid > 0 else 0.0

    def _fill_price_passive(self, side: str, bid: float, ask: float) -> float:
        """Simulate passive (limit) fill — NO slippage, fill at limit price.

        For LONG: post at bid (join the bid queue).
        For SHORT: post at ask (join the ask queue).
        Cost is effectively 0 ticks crossing cost (vs 1-2 for market).
        """
        if side == "LONG":
            return bid if bid > 0 else 0.0
        else:
            return ask if ask > 0 else 0.0

    def _open_position(self, side: str, fill_price: float, timestamp: float,
                       prob_up: float, confidence: float,
                       cost_ticks: float = None):
        """Open a new position."""
        if cost_ticks is None:
            cost_ticks = self.slippage_ticks  # default market order cost

        self.position.size = 1 if side == "LONG" else -1
        self.position.side = side
        self.position.entry_price = fill_price
        self.position.entry_time = timestamp
        self.position.entry_prob = prob_up
        self.position.entry_confidence = confidence
        self.stats.signals_acted += 1

    def _close_position(self, fill_price: float, timestamp: float,
                        reason: str, cost_ticks: float = None):
        """Close current position and record the trade."""
        pos = self.position
        if pos.size == 0:
            return

        if cost_ticks is None:
            cost_ticks = self.slippage_ticks

        self._trade_counter += 1

        if pos.size > 0:  # was long
            gross_pnl = (fill_price - pos.entry_price) * POINT_VALUE
        else:  # was short
            gross_pnl = (pos.entry_price - fill_price) * POINT_VALUE

        commission = self.commission_per_side * 2
        net_pnl = gross_pnl - commission
        hold_time = timestamp - pos.entry_time

        trade = {
            "trade_id": self._trade_counter,
            "strategy": self.name,
            "entry_time": pos.entry_time,
            "exit_time": timestamp,
            "entry_price": pos.entry_price,
            "exit_price": fill_price,
            "side": pos.side,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "commission": commission,
            "hold_time_s": hold_time,
            "exit_reason": reason,
            "entry_prob": pos.entry_prob,
            "entry_confidence": pos.entry_confidence,
        }

        self.stats.record_trade(trade)

        # Reset position
        pos.size = 0
        pos.side = ""
        pos.entry_price = 0.0
        pos.entry_time = 0.0
        pos.take_profit = 0.0
        pos.stop_loss = 0.0
        pos.pending_limit = False

        return trade

    def _set_tp_sl(self, tp_ticks: float, sl_ticks: float):
        """Set take-profit and stop-loss levels based on entry price and side."""
        pos = self.position
        if pos.size == 0:
            return

        tp_offset = tp_ticks * TICK_SIZE
        sl_offset = sl_ticks * TICK_SIZE

        if pos.size > 0:  # long
            pos.take_profit = pos.entry_price + tp_offset
            pos.stop_loss = pos.entry_price - sl_offset
        else:  # short
            pos.take_profit = pos.entry_price - tp_offset
            pos.stop_loss = pos.entry_price + sl_offset

    def _check_tp_sl(self, timestamp: float, bid: float, ask: float) -> bool:
        """Check if TP or SL has been hit. Returns True if position was closed."""
        pos = self.position
        if pos.size == 0 or (pos.take_profit == 0.0 and pos.stop_loss == 0.0):
            return False

        if pos.size > 0:  # long
            # TP: can we sell at or above TP? (bid >= TP)
            if pos.take_profit > 0 and bid >= pos.take_profit:
                self._close_position(pos.take_profit, timestamp, "take_profit",
                                     cost_ticks=0.0)  # exit at TP level
                return True
            # SL: bid drops to SL
            if pos.stop_loss > 0 and bid <= pos.stop_loss:
                # SL fill is at stop price (market order on trigger)
                sl_fill = pos.stop_loss - self.slippage_ticks * TICK_SIZE
                self._close_position(sl_fill, timestamp, "stop_loss")
                return True
        else:  # short
            # TP: can we buy at or below TP? (ask <= TP)
            if pos.take_profit > 0 and ask <= pos.take_profit:
                self._close_position(pos.take_profit, timestamp, "take_profit",
                                     cost_ticks=0.0)
                return True
            # SL: ask rises to SL
            if pos.stop_loss > 0 and ask >= pos.stop_loss:
                sl_fill = pos.stop_loss + self.slippage_ticks * TICK_SIZE
                self._close_position(sl_fill, timestamp, "stop_loss")
                return True

        return False

    def _check_timeout(self, timestamp: float, timeout_s: float,
                       bid: float, ask: float) -> bool:
        """Check if position has timed out. Returns True if closed."""
        pos = self.position
        if pos.size == 0 or pos.entry_time == 0:
            return False

        age = timestamp - pos.entry_time
        if age >= timeout_s:
            fill = self._fill_price_market(
                "SHORT" if pos.size > 0 else "LONG", bid, ask)
            self._close_position(fill, timestamp, "timeout")
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 1: Baseline Market Order
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineMarketOrder(Strategy):
    """Enter with market order when confidence > threshold. Exit on signal flip or timeout.

    This replicates the current paper trading behavior in lgbm_live_inference.py.
    Cost model: 0.376 ticks (commission only — HC #231(A): no spread crossing cost).
    """

    def __init__(self, confidence_threshold: float = CONF_TOP10,
                 timeout_s: float = 30.0, **kwargs):
        super().__init__(
            name=f"Baseline(conf>{confidence_threshold:.3f})",
            **kwargs,
        )
        self.confidence_threshold = confidence_threshold
        self.timeout_s = timeout_s

    def on_signal(self, timestamp: float, prob_up: float, confidence: float,
                  bid: float, ask: float) -> None:
        self.stats.signals_received += 1

        if confidence < self.confidence_threshold:
            return

        desired = "LONG" if prob_up > 0.5 else "SHORT"
        pos = self.position

        if pos.size == 0:
            fill = self._fill_price_market(desired, bid, ask)
            self._open_position(desired, fill, timestamp, prob_up, confidence)
            return

        # Already positioned in same direction
        same_dir = (pos.size > 0 and desired == "LONG") or \
                   (pos.size < 0 and desired == "SHORT")
        if same_dir:
            return

        # Flip: close and reopen
        close_fill = self._fill_price_market(
            "SHORT" if pos.size > 0 else "LONG", bid, ask)
        self._close_position(close_fill, timestamp, "signal_flip")

        fill = self._fill_price_market(desired, bid, ask)
        self._open_position(desired, fill, timestamp, prob_up, confidence)

    def on_price_update(self, timestamp: float, bid: float, ask: float) -> None:
        self._check_timeout(timestamp, self.timeout_s, bid, ask)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 2: High Confidence Only (Top1%)
# ═══════════════════════════════════════════════════════════════════════════════

class HighConfidenceOnly(Strategy):
    """Only trade at Top1% confidence (threshold=0.37). Fewer trades, each should be profitable.

    Same as baseline but with much higher bar. DA=73.2% means +0.83 ticks/trade
    after 2-tick costs.
    """

    def __init__(self, confidence_threshold: float = CONF_TOP1,
                 timeout_s: float = 30.0, **kwargs):
        super().__init__(
            name=f"HighConf(Top1%,conf>{confidence_threshold:.2f})",
            **kwargs,
        )
        self.confidence_threshold = confidence_threshold
        self.timeout_s = timeout_s

    def on_signal(self, timestamp: float, prob_up: float, confidence: float,
                  bid: float, ask: float) -> None:
        self.stats.signals_received += 1

        if confidence < self.confidence_threshold:
            return

        desired = "LONG" if prob_up > 0.5 else "SHORT"
        pos = self.position

        if pos.size == 0:
            fill = self._fill_price_market(desired, bid, ask)
            self._open_position(desired, fill, timestamp, prob_up, confidence)
            return

        same_dir = (pos.size > 0 and desired == "LONG") or \
                   (pos.size < 0 and desired == "SHORT")
        if same_dir:
            return

        close_fill = self._fill_price_market(
            "SHORT" if pos.size > 0 else "LONG", bid, ask)
        self._close_position(close_fill, timestamp, "signal_flip")

        fill = self._fill_price_market(desired, bid, ask)
        self._open_position(desired, fill, timestamp, prob_up, confidence)

    def on_price_update(self, timestamp: float, bid: float, ask: float) -> None:
        self._check_timeout(timestamp, self.timeout_s, bid, ask)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 3: MFE-Informed TP/SL
# ═══════════════════════════════════════════════════════════════════════════════

class MFEInformedTPSL(Strategy):
    """Enter at Top5%+ confidence. Set TP at MFE expectation, SL at MAE expectation.

    From OOT analysis:
      Top5%: MFE=2.5t, MAE=0.8t (winners), MAE=2.17t (losers)
      Top1%: MFE=3.85t, MAE=0.5t

    We use confidence-adaptive TP/SL: interpolate between Top5% and Top1% levels
    based on actual confidence value.

    This should improve P&L by:
      1. Capturing winners before they reverse (TP at MFE)
      2. Cutting losers early (SL at MAE threshold)
      3. Not waiting for signal flip or timeout on every trade
    """

    def __init__(self, min_confidence: float = CONF_TOP5,
                 timeout_s: float = 30.0, **kwargs):
        super().__init__(
            name="MFE-TP/SL(Top5%+)",
            **kwargs,
        )
        self.min_confidence = min_confidence
        self.timeout_s = timeout_s

    def _adaptive_tp_sl(self, confidence: float) -> tuple:
        """Compute TP/SL in ticks based on confidence level.

        Interpolate between Top5% and Top1% MFE/MAE expectations.
        """
        # Normalize confidence to [0, 1] within the Top5%->Top1% range
        t = min(1.0, max(0.0, (confidence - CONF_TOP5) / (CONF_TOP1 - CONF_TOP5)))

        # Interpolate MFE (take profit target)
        tp = MFE_TOP5 + t * (MFE_TOP1 - MFE_TOP5)

        # Interpolate MAE (stop loss) — use winner MAE as SL
        # At Top5% confidence, allow wider stop (losers drawdown more)
        sl_tight = MAE_TOP1   # 0.5 ticks for Top1%
        sl_wide = MAE_TOP5 + 1.0   # 1.8 ticks for Top5% (a bit wider than winner MAE)
        sl = sl_wide + t * (sl_tight - sl_wide)

        return round(tp, 1), round(sl, 1)

    def on_signal(self, timestamp: float, prob_up: float, confidence: float,
                  bid: float, ask: float) -> None:
        self.stats.signals_received += 1

        if confidence < self.min_confidence:
            return

        desired = "LONG" if prob_up > 0.5 else "SHORT"
        pos = self.position

        if pos.size == 0:
            fill = self._fill_price_market(desired, bid, ask)
            self._open_position(desired, fill, timestamp, prob_up, confidence)

            # Set adaptive TP/SL
            tp_ticks, sl_ticks = self._adaptive_tp_sl(confidence)
            self._set_tp_sl(tp_ticks, sl_ticks)
            return

        # If already in position, check for flip
        same_dir = (pos.size > 0 and desired == "LONG") or \
                   (pos.size < 0 and desired == "SHORT")
        if same_dir:
            return

        # Flip: close current, reopen with new TP/SL
        close_fill = self._fill_price_market(
            "SHORT" if pos.size > 0 else "LONG", bid, ask)
        self._close_position(close_fill, timestamp, "signal_flip")

        fill = self._fill_price_market(desired, bid, ask)
        self._open_position(desired, fill, timestamp, prob_up, confidence)
        tp_ticks, sl_ticks = self._adaptive_tp_sl(confidence)
        self._set_tp_sl(tp_ticks, sl_ticks)

    def on_price_update(self, timestamp: float, bid: float, ask: float) -> None:
        if self.position.size == 0:
            return

        # Check TP/SL first (higher priority than timeout)
        if self._check_tp_sl(timestamp, bid, ask):
            return

        # Then check timeout
        self._check_timeout(timestamp, self.timeout_s, bid, ask)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 4: Passive Entry (Limit Order Simulation)
# ═══════════════════════════════════════════════════════════════════════════════

class PassiveEntry(Strategy):
    """Post limit order at bid (LONG) or ask (SHORT) for Top5%+ confidence.

    Instead of crossing the spread (1 tick cost), join the queue:
      - LONG: post limit buy at current bid. Wait up to 2s for fill.
      - SHORT: post limit sell at current ask. Wait up to 2s for fill.

    If filled, effective cost is ~0 ticks crossing + commission only.
    At 1-tick cost (vs 2), Top5% becomes +0.29/trade (from -0.71).

    Fill simulation: We assume fill occurs when price touches our limit level.
    For LONG at bid: filled when ask drops to our bid (aggressive sellers arrive).
    For SHORT at ask: filled when bid rises to our ask (aggressive buyers arrive).
    This is conservative — in practice you may get filled by queue priority alone.

    Exit: market order on signal flip or timeout.
    """

    def __init__(self, min_confidence: float = CONF_TOP5,
                 limit_timeout_s: float = 2.0,
                 position_timeout_s: float = 30.0, **kwargs):
        super().__init__(
            name="Passive(Limit,Top5%+)",
            slippage_ticks=0.0,  # passive fill = no crossing cost
            **kwargs,
        )
        self.min_confidence = min_confidence
        self.limit_timeout_s = limit_timeout_s
        self.position_timeout_s = position_timeout_s
        # For exit, we use market orders with standard slippage
        self._exit_slippage = 1.0

    def on_signal(self, timestamp: float, prob_up: float, confidence: float,
                  bid: float, ask: float) -> None:
        self.stats.signals_received += 1

        if confidence < self.min_confidence:
            return

        desired = "LONG" if prob_up > 0.5 else "SHORT"
        pos = self.position

        # If we have a pending limit and signal flips, cancel the limit
        if pos.pending_limit and pos.limit_side != desired:
            pos.pending_limit = False
            pos.limit_price = 0.0

        if pos.size != 0:
            # Already in position
            same_dir = (pos.size > 0 and desired == "LONG") or \
                       (pos.size < 0 and desired == "SHORT")
            if same_dir:
                return

            # Flip: close at market, then post new limit
            close_fill = bid - self._exit_slippage * TICK_SIZE if pos.size > 0 \
                else ask + self._exit_slippage * TICK_SIZE
            self._close_position(close_fill, timestamp, "signal_flip")

        # Post limit order (don't open position yet — wait for fill)
        if not pos.pending_limit and pos.size == 0:
            pos.pending_limit = True
            pos.limit_side = desired
            pos.limit_submit_time = timestamp
            pos.limit_timeout = self.limit_timeout_s

            if desired == "LONG":
                pos.limit_price = bid  # join the bid
            else:
                pos.limit_price = ask  # join the ask

            # Store signal info for when fill happens
            pos.entry_prob = prob_up
            pos.entry_confidence = confidence

    def on_price_update(self, timestamp: float, bid: float, ask: float) -> None:
        pos = self.position

        # Check pending limit order fill
        if pos.pending_limit:
            # Check limit timeout
            if timestamp - pos.limit_submit_time > pos.limit_timeout:
                pos.pending_limit = False
                pos.limit_price = 0.0
                return

            filled = False
            if pos.limit_side == "LONG":
                # Fill when market sells into our bid: trade at or below our limit
                # Conservative: fill when ask drops to our limit (someone crosses)
                if ask <= pos.limit_price or bid < pos.limit_price:
                    filled = True
            else:  # SHORT
                # Fill when market buys into our ask
                if bid >= pos.limit_price or ask > pos.limit_price:
                    filled = True

            if filled:
                fill_price = pos.limit_price  # filled at our limit price
                self._open_position(
                    pos.limit_side, fill_price, timestamp,
                    pos.entry_prob, pos.entry_confidence,
                    cost_ticks=0.0,  # passive fill
                )
                pos.pending_limit = False
                pos.limit_price = 0.0
                return

        # Check position timeout (exit at market)
        if pos.size != 0 and pos.entry_time > 0:
            age = timestamp - pos.entry_time
            if age >= self.position_timeout_s:
                # Exit with market order (standard slippage)
                if pos.size > 0:
                    fill = bid - self._exit_slippage * TICK_SIZE
                else:
                    fill = ask + self._exit_slippage * TICK_SIZE
                self._close_position(fill, timestamp, "timeout")


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 5: Confidence-Scaled Position
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceScaled(Strategy):
    """Enter at Top10%+, scale "virtual size" by confidence tier.

    Virtual sizing (no real position scaling — just P&L multiplier):
      Top10% (conf 0.127-0.20): size = 1
      Top5%  (conf 0.20-0.37):  size = 2
      Top1%  (conf 0.37+):      size = 3

    This tests the hypothesis that higher confidence deserves more capital.
    Track aggregate P&L to find the optimal weighting scheme.
    """

    def __init__(self, timeout_s: float = 30.0, **kwargs):
        super().__init__(
            name="ConfScaled(Top10%+,1/2/3x)",
            **kwargs,
        )
        self.timeout_s = timeout_s
        self._virtual_size = 1  # multiplier for P&L

    def _confidence_to_size(self, confidence: float) -> int:
        """Map confidence to virtual position size."""
        if confidence >= CONF_TOP1:
            return 3
        elif confidence >= CONF_TOP5:
            return 2
        elif confidence >= CONF_TOP10:
            return 1
        return 0

    def on_signal(self, timestamp: float, prob_up: float, confidence: float,
                  bid: float, ask: float) -> None:
        self.stats.signals_received += 1

        vsize = self._confidence_to_size(confidence)
        if vsize == 0:
            return

        desired = "LONG" if prob_up > 0.5 else "SHORT"
        pos = self.position

        if pos.size == 0:
            fill = self._fill_price_market(desired, bid, ask)
            self._open_position(desired, fill, timestamp, prob_up, confidence)
            self._virtual_size = vsize
            return

        same_dir = (pos.size > 0 and desired == "LONG") or \
                   (pos.size < 0 and desired == "SHORT")
        if same_dir:
            # Update virtual size if confidence changed tier
            self._virtual_size = max(self._virtual_size, vsize)
            return

        # Flip: close and reopen
        close_fill = self._fill_price_market(
            "SHORT" if pos.size > 0 else "LONG", bid, ask)
        self._close_position_scaled(close_fill, timestamp, "signal_flip")

        fill = self._fill_price_market(desired, bid, ask)
        self._open_position(desired, fill, timestamp, prob_up, confidence)
        self._virtual_size = vsize

    def _close_position_scaled(self, fill_price: float, timestamp: float,
                               reason: str):
        """Close with virtual size multiplier applied to P&L."""
        pos = self.position
        if pos.size == 0:
            return

        self._trade_counter += 1

        if pos.size > 0:
            gross_pnl = (fill_price - pos.entry_price) * POINT_VALUE * self._virtual_size
        else:
            gross_pnl = (pos.entry_price - fill_price) * POINT_VALUE * self._virtual_size

        commission = self.commission_per_side * 2 * self._virtual_size
        net_pnl = gross_pnl - commission
        hold_time = timestamp - pos.entry_time

        trade = {
            "trade_id": self._trade_counter,
            "strategy": self.name,
            "entry_time": pos.entry_time,
            "exit_time": timestamp,
            "entry_price": pos.entry_price,
            "exit_price": fill_price,
            "side": pos.side,
            "virtual_size": self._virtual_size,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "commission": commission,
            "hold_time_s": hold_time,
            "exit_reason": reason,
            "entry_prob": pos.entry_prob,
            "entry_confidence": pos.entry_confidence,
        }

        self.stats.record_trade(trade)

        pos.size = 0
        pos.side = ""
        pos.entry_price = 0.0
        pos.entry_time = 0.0

    def on_price_update(self, timestamp: float, bid: float, ask: float) -> None:
        pos = self.position
        if pos.size == 0 or pos.entry_time == 0:
            return

        age = timestamp - pos.entry_time
        if age >= self.timeout_s:
            close_fill = self._fill_price_market(
                "SHORT" if pos.size > 0 else "LONG", bid, ask)
            self._close_position_scaled(close_fill, timestamp, "timeout")


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 6: Signal Persistence
# ═══════════════════════════════════════════════════════════════════════════════

class SignalPersistence(Strategy):
    """Only enter if signal is consistent for N consecutive predictions.

    Requires the SAME direction signal for `required_consistent` consecutive
    predictions (each 500 events apart = ~1000+ events for 2 consecutive).

    This reduces whipsaw trades where the model flip-flops between signals.
    The hypothesis: persistent signals indicate genuine microstructure pressure,
    not noise.
    """

    def __init__(self, min_confidence: float = CONF_TOP5,
                 required_consistent: int = 2,
                 timeout_s: float = 30.0, **kwargs):
        super().__init__(
            name=f"Persist(Top5%+,{required_consistent}x)",
            **kwargs,
        )
        self.min_confidence = min_confidence
        self.required_consistent = required_consistent
        self.timeout_s = timeout_s
        self._signal_history: List[str] = []  # recent signal directions
        self._last_confidence: float = 0.0
        self._last_prob: float = 0.5

    def on_signal(self, timestamp: float, prob_up: float, confidence: float,
                  bid: float, ask: float) -> None:
        self.stats.signals_received += 1

        if confidence < self.min_confidence:
            # Below threshold = NEUTRAL, breaks streak
            self._signal_history.clear()
            return

        desired = "LONG" if prob_up > 0.5 else "SHORT"

        # Track signal consistency
        if self._signal_history and self._signal_history[-1] != desired:
            # Direction changed, reset streak
            self._signal_history.clear()

        self._signal_history.append(desired)
        self._last_confidence = confidence
        self._last_prob = prob_up

        # Only act if we have enough consistent signals
        if len(self._signal_history) < self.required_consistent:
            return

        pos = self.position

        if pos.size == 0:
            fill = self._fill_price_market(desired, bid, ask)
            self._open_position(desired, fill, timestamp, prob_up, confidence)
            self._signal_history.clear()  # reset after entry
            return

        same_dir = (pos.size > 0 and desired == "LONG") or \
                   (pos.size < 0 and desired == "SHORT")
        if same_dir:
            return

        # Flip on persistent contrary signal
        close_fill = self._fill_price_market(
            "SHORT" if pos.size > 0 else "LONG", bid, ask)
        self._close_position(close_fill, timestamp, "signal_flip")

        fill = self._fill_price_market(desired, bid, ask)
        self._open_position(desired, fill, timestamp, prob_up, confidence)
        self._signal_history.clear()

    def on_price_update(self, timestamp: float, bid: float, ask: float) -> None:
        self._check_timeout(timestamp, self.timeout_s, bid, ask)


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy Runner — feeds the same signal stream to all strategies
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyRunner:
    """Orchestrates multiple strategies on the same signal + price stream.

    Usage:
        runner = StrategyRunner([strategy1, strategy2, ...])
        # From live feed or replay:
        runner.on_signal(timestamp, prob_up, confidence, bid, ask)
        runner.on_price_update(timestamp, bid, ask)
        # At end:
        runner.print_comparison()
        runner.save_results("results.json")
    """

    def __init__(self, strategies: List[Strategy]):
        self.strategies = strategies
        self._signal_count = 0
        self._price_update_count = 0
        self._start_time = time.time()
        log.info("StrategyRunner initialized with %d strategies:", len(strategies))
        for s in strategies:
            log.info("  - %s", s.name)

    def on_signal(self, timestamp: float, prob_up: float, confidence: float,
                  bid: float, ask: float) -> None:
        """Distribute a new LGBM signal to all strategies."""
        self._signal_count += 1
        for strategy in self.strategies:
            try:
                strategy.on_signal(timestamp, prob_up, confidence, bid, ask)
            except Exception as e:
                log.exception("Strategy %s error on signal: %s", strategy.name, e)

    def on_price_update(self, timestamp: float, bid: float, ask: float) -> None:
        """Distribute a BBO update to all strategies."""
        self._price_update_count += 1
        for strategy in self.strategies:
            try:
                strategy.on_price_update(timestamp, bid, ask)
            except Exception as e:
                log.exception("Strategy %s error on price update: %s",
                              strategy.name, e)

    def close_all(self, timestamp: float, bid: float, ask: float) -> None:
        """Force-close all open positions at end of session."""
        for strategy in self.strategies:
            if strategy.position.size != 0:
                fill = strategy._fill_price_market(
                    "SHORT" if strategy.position.size > 0 else "LONG",
                    bid, ask,
                )
                strategy._close_position(fill, timestamp, "session_end")

    def print_comparison(self) -> str:
        """Print a comparison table of all strategies. Returns the table string."""
        elapsed = time.time() - self._start_time
        lines = []
        lines.append("=" * 120)
        lines.append(f"EXECUTION STRATEGY COMPARISON — {elapsed/60:.1f} min | "
                     f"{self._signal_count} signals | {self._price_update_count} price updates")
        lines.append("=" * 120)
        lines.append(
            f"{'Strategy':<28s} | {'Trades':>6s} | {'WR':>6s} | "
            f"{'Net P&L':>10s} | {'Avg/Trade':>16s} | "
            f"{'Sortino':>7s} | {'PF':>5s} | {'MaxDD':>9s}"
        )
        lines.append("-" * 120)

        for s in self.strategies:
            lines.append(s.stats.summary_line())

        lines.append("=" * 120)

        # Find best strategy by net P&L
        if any(s.stats.total_trades > 0 for s in self.strategies):
            best = max(self.strategies, key=lambda s: s.stats.total_net_pnl)
            lines.append(f"BEST by Net P&L: {best.name} "
                         f"(${best.stats.total_net_pnl:+.2f})")

            # Best by Sortino (among those with trades)
            traded = [s for s in self.strategies if s.stats.total_trades >= 3]
            if traded:
                best_sortino = max(traded, key=lambda s: s.stats.sortino)
                lines.append(f"BEST by Sortino: {best_sortino.name} "
                             f"({best_sortino.stats.sortino:.2f})")

        lines.append("=" * 120)

        table = "\n".join(lines)
        log.info("\n%s", table)
        return table

    def save_results(self, output_path: str | Path) -> Path:
        """Save full results to JSON including equity curves."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_signals": self._signal_count,
                "total_price_updates": self._price_update_count,
                "elapsed_seconds": time.time() - self._start_time,
                "num_strategies": len(self.strategies),
            },
            "strategies": {},
        }

        for s in self.strategies:
            results["strategies"][s.name] = {
                "summary": s.stats.summary_dict(),
                "equity_curve": s.stats.equity_curve,
                "returns": s.stats.returns,
                "trades": s.stats.trade_log,
            }

        with open(output_path, "w") as f:
            json.dump(_json_safe(results), f, indent=2)

        log.info("Results saved to %s", output_path)
        return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Factory: create the default set of 6 strategies
# ═══════════════════════════════════════════════════════════════════════════════

def create_default_strategies() -> List[Strategy]:
    """Create the standard set of 6 execution strategies for comparison."""
    return [
        BaselineMarketOrder(
            confidence_threshold=CONF_TOP10,
            timeout_s=30.0,
        ),
        HighConfidenceOnly(
            confidence_threshold=CONF_TOP1,
            timeout_s=30.0,
        ),
        MFEInformedTPSL(
            min_confidence=CONF_TOP5,
            timeout_s=30.0,
        ),
        PassiveEntry(
            min_confidence=CONF_TOP5,
            limit_timeout_s=2.0,
            position_timeout_s=30.0,
        ),
        ConfidenceScaled(
            timeout_s=30.0,
        ),
        SignalPersistence(
            min_confidence=CONF_TOP5,
            required_consistent=2,
            timeout_s=30.0,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Replay from signal JSONL file
# ═══════════════════════════════════════════════════════════════════════════════

def replay_from_signals_jsonl(jsonl_path: str | Path,
                               strategies: Optional[List[Strategy]] = None,
                               output_path: Optional[str | Path] = None) -> StrategyRunner:
    """Replay signals from a JSONL log file (from lgbm_live_inference.py).

    Each line should have: timestamp, prob_up, confidence, bid, ask, signal, etc.

    Args:
        jsonl_path: Path to signal JSONL file
        strategies: List of strategies (defaults to create_default_strategies())
        output_path: Path to save results JSON (optional)

    Returns:
        StrategyRunner with completed results
    """
    if strategies is None:
        strategies = create_default_strategies()

    runner = StrategyRunner(strategies)
    jsonl_path = Path(jsonl_path)

    log.info("Replaying signals from %s", jsonl_path)
    n_lines = 0
    last_bid, last_ask = 0.0, 0.0

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            n_lines += 1

            # Extract fields (compatible with lgbm_live_inference.py output)
            prob_up = rec.get("prob_up", 0.5)
            confidence = rec.get("confidence", 0.0)
            bid = rec.get("bid", last_bid)
            ask = rec.get("ask", last_ask)

            # Parse timestamp
            ts_str = rec.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str).timestamp()
                except (ValueError, TypeError):
                    ts = time.time()
            else:
                ts = rec.get("ts", time.time())

            if bid > 0:
                last_bid = bid
            if ask > 0:
                last_ask = ask

            # Send price update first (for TP/SL checks)
            runner.on_price_update(ts, bid, ask)
            # Then send signal
            runner.on_signal(ts, prob_up, confidence, bid, ask)

    # Close all open positions
    runner.close_all(time.time(), last_bid, last_ask)

    log.info("Replay complete: %d signals processed", n_lines)
    runner.print_comparison()

    if output_path:
        runner.save_results(output_path)

    return runner


# ═══════════════════════════════════════════════════════════════════════════════
# CLI for standalone testing
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI for replaying signals through strategies."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    ap = argparse.ArgumentParser(
        description="Execution Strategy Comparison — replay signals through "
                    "multiple strategies and compare performance.")
    ap.add_argument("--signals", required=True,
                    help="Path to signal JSONL file from lgbm_live_inference.py")
    ap.add_argument("--output", default=None,
                    help="Path to save results JSON")
    args = ap.parse_args()

    output = args.output or str(
        Path(args.signals).parent / "strategy_comparison_results.json"
    )

    replay_from_signals_jsonl(args.signals, output_path=output)


if __name__ == "__main__":
    main()
