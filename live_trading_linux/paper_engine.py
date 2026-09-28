#!/usr/bin/env python3
"""
paper_engine.py — Paper Trading Engine with Simulated Fills on Real Rithmic Feed.

This engine connects to Rithmic LIVE for real-time market data but NEVER submits
any orders to the exchange. Instead, it simulates order fills locally using the
actual BBO/trade stream, producing results indistinguishable from live trading
in the logs.

Fill simulation logic:
  * MARKET BUY  → filled at current best_ask + slippage
  * MARKET SELL → filled at current best_bid - slippage
  * Slippage model: configurable fixed ticks (default 0.25 = 1 tick on ES)
  * Fill latency: configurable delay (default 50ms) before fill is "confirmed"
  * Commission: $0.50 per side per contract (AMP micro-rate)

Tracking:
  * Real-time P&L (unrealized + realized)
  * Trade log (JSONL) with full audit trail
  * Position tracking with entry/exit prices
  * Performance stats: win rate, avg P&L, Sortino, max drawdown
  * Periodic stats dump to log + Discord notification capability

IMPORTANT: This engine uses the Rithmic LIVE feed (account XXXXXX) for data
           but will NEVER call submit_order(). Zero risk of accidental execution.

CLI:
    python3 -m live_trading_linux.paper_engine \\
        --model /path/to/model.pkl \\
        --symbol ESM6 --exchange CME \\
        --tier top10 --timeout 30
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import signal as _sig
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List

import numpy as np

from live_trading_linux.lgbm_inference import LGBMInference, TIER_ORDER
from live_trading_linux.rithmic_client import (
    BBOEvent, RithmicClient, TradeEvent,
)
from live_trading_linux.streaming_features import StreamingFeatures


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("paper_engine")
if not log.handlers:
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_DIR / "paper_engine.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(sh)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PaperConfig:
    """Paper trading configuration."""
    symbol: str = "ESM6"
    exchange: str = "CME"
    tick_size: float = 0.25           # ES tick size
    point_value: float = 50.0         # ES $50 per point
    commission_per_side: float = 0.50  # AMP rate per contract per side
    slippage_ticks: float = 0.0       # HC #290(C) 2026-05-11: commission only. Fill price IS the price. No synthetic spread tick regardless of order type.
    fill_delay_ms: float = 50.0       # simulated fill latency
    max_position: int = 1             # max contracts
    position_timeout_s: float = 30.0  # force-close after N seconds
    threshold: float = 0.0            # min |prediction| to trade
    min_tier: str = "top10"           # minimum confidence tier
    max_spread_ticks: float = 3.0     # max spread in ticks to trade (cards: 0.75)
    stats_interval_s: float = 300.0   # dump stats every 5 min


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SimulatedFill:
    """A simulated fill event."""
    fill_id: int
    timestamp: float          # local time
    exchange_ts: float        # exchange time
    symbol: str
    side: str                 # 'B' or 'S'
    qty: int
    fill_price: float         # simulated fill price (with slippage)
    market_bid: float         # BBO at time of fill
    market_ask: float
    slippage: float           # slippage applied
    commission: float         # commission charged
    reason: str               # 'open', 'flip', 'close:timeout', etc.
    signal_pred: float        # model prediction that triggered this
    signal_tier: str


@dataclass
class PaperTrade:
    """A completed round-trip trade."""
    trade_id: int
    entry_time: float
    exit_time: float
    entry_price: float
    exit_price: float
    side: str                 # 'B' (long) or 'S' (short)
    qty: int
    gross_pnl: float          # before commission
    net_pnl: float            # after commission
    commission: float
    hold_time_s: float
    exit_reason: str          # 'signal_flip', 'timeout', 'stop'
    entry_pred: float
    entry_tier: str


@dataclass
class PaperPosition:
    """Current position state."""
    size: int = 0             # +N long, -N short, 0 flat
    side: str = ""
    entry_price: float = 0.0
    entry_ts: float = 0.0
    entry_pred: float = 0.0
    entry_tier: str = ""
    unrealized_pnl: float = 0.0


@dataclass
class PaperStats:
    """Cumulative performance statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_gross_pnl: float = 0.0
    total_net_pnl: float = 0.0
    total_commission: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    current_drawdown: float = 0.0
    returns: list = field(default_factory=list)  # per-trade net returns
    signals_generated: int = 0
    signals_traded: int = 0
    events_processed: int = 0
    start_time: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_net_pnl / self.total_trades

    @property
    def sortino(self) -> float:
        """Sortino ratio (annualized from per-trade returns)."""
        if len(self.returns) < 2:
            return 0.0
        mean_ret = statistics.mean(self.returns)
        downside = [r for r in self.returns if r < 0]
        if not downside:
            return float('inf') if mean_ret > 0 else 0.0
        downside_std = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
        if downside_std == 0:
            return 0.0
        # Rough annualization: assume ~50 trades/day, 252 days/year
        trades_per_year = 50 * 252
        return (mean_ret / downside_std) * math.sqrt(trades_per_year)

    @property
    def profit_factor(self) -> float:
        gross_wins = sum(r for r in self.returns if r > 0)
        gross_losses = abs(sum(r for r in self.returns if r < 0))
        if gross_losses == 0:
            return float('inf') if gross_wins > 0 else 0.0
        return gross_wins / gross_losses


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------
class PaperEngine:
    """
    Paper trading engine: real Rithmic feed, simulated fills.

    NEVER sends orders to the exchange. All fills are computed locally
    from the live BBO stream.
    """

    def __init__(self, config: PaperConfig, model_path: str | Path,
                 calibration_path: Optional[str | Path] = None) -> None:
        self.cfg = config
        self.features = StreamingFeatures()
        self.model = LGBMInference(model_path, calibration_path)

        # Rithmic client — DATA ONLY, never used for orders
        self.client = RithmicClient()
        self.client.set_md_callback(self._on_md)
        # Deliberately NOT setting order callback — we never send orders

        # Market state
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.last_trade_price: float = 0.0
        self.last_event_ts: Optional[float] = None

        # Paper trading state
        self.position = PaperPosition()
        self.stats = PaperStats(start_time=time.time())
        self.trades: List[PaperTrade] = []
        self._fill_counter = 0
        self._trade_counter = 0
        self._stop = asyncio.Event()

        # Logging
        self._trades_log_path = _LOG_DIR / f"paper_trades_{config.symbol}.jsonl"
        self._fills_log_path = _LOG_DIR / f"paper_fills_{config.symbol}.jsonl"
        self._signals_log_path = _LOG_DIR / f"paper_signals_{config.symbol}.jsonl"
        self._trades_fh = open(self._trades_log_path, "a", buffering=1)
        self._fills_fh = open(self._fills_log_path, "a", buffering=1)
        self._signals_fh = open(self._signals_log_path, "a", buffering=1)

        # Tasks
        self._tasks: list[asyncio.Task] = []

        log.info("PaperEngine initialized: %s @ %s | tier=%s threshold=%.3f timeout=%.0fs",
                 config.symbol, config.exchange, config.min_tier,
                 config.threshold, config.position_timeout_s)
        log.info("  slippage=%.2f ticks | commission=$%.2f/side | max_pos=%d",
                 config.slippage_ticks, config.commission_per_side, config.max_position)
        log.info("  *** PAPER MODE — NO REAL ORDERS WILL BE SENT ***")

    # --------------------------------------------------------------- lifecycle
    async def run(self) -> None:
        """Connect to Rithmic for data and run paper trading loop."""
        log.info("Connecting to Rithmic for MARKET DATA ONLY...")
        await self.client.connect()
        log.info("  Account: %s (DATA ONLY — orders disabled)",
                 self.client.account_id)

        await self.client.subscribe_md(self.cfg.symbol, self.cfg.exchange)

        # Background tasks
        self._tasks.append(asyncio.create_task(
            self._position_timeout_loop(), name="paper_timeout"))
        self._tasks.append(asyncio.create_task(
            self._stats_loop(), name="paper_stats"))

        log.info("PaperEngine running. Waiting for market data...")
        try:
            await self._stop.wait()
        finally:
            for t in self._tasks:
                t.cancel()
            for t in self._tasks:
                try:
                    await t
                except Exception:
                    pass
            await self.client.disconnect()
            self._dump_final_stats()
            for fh in (self._trades_fh, self._fills_fh, self._signals_fh):
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
            log.info("PaperEngine stopped.")

    def stop(self) -> None:
        self._stop.set()

    # --------------------------------------------------------------- MD handling
    async def _on_md(self, event) -> None:
        try:
            if isinstance(event, BBOEvent):
                await self._handle_bbo(event)
            elif isinstance(event, TradeEvent):
                await self._handle_trade(event)
        except Exception as e:
            log.exception("_on_md error: %s", e)

    async def _handle_bbo(self, ev: BBOEvent) -> None:
        prev_bid = self.best_bid
        prev_ask = self.best_ask
        if ev.has_bid:
            self.best_bid = ev.bid_price
        if ev.has_ask:
            self.best_ask = ev.ask_price

        # Update unrealized P&L
        if self.position.size != 0:
            mid = (self.best_bid + self.best_ask) / 2 if (self.best_bid and self.best_ask) else self.last_trade_price
            if self.position.size > 0:
                self.position.unrealized_pnl = (mid - self.position.entry_price) * self.cfg.point_value * abs(self.position.size)
            else:
                self.position.unrealized_pnl = (self.position.entry_price - mid) * self.cfg.point_value * abs(self.position.size)

        spread = self.best_ask - self.best_bid if (self.best_bid and self.best_ask) else 0.0
        now_exch = self._exchange_ts(ev.ssboe, ev.usecs)

        if ev.has_bid and ev.bid_price != prev_bid:
            await self._process_event(0, 0, ev.bid_price,
                                       max(1, int(ev.bid_size) if ev.bid_size else 1),
                                       spread, now_exch, ev.recv_ts, ev.symbol)
        if ev.has_ask and ev.ask_price != prev_ask:
            await self._process_event(0, 1, ev.ask_price,
                                       max(1, int(ev.ask_size) if ev.ask_size else 1),
                                       spread, now_exch, ev.recv_ts, ev.symbol)

    async def _handle_trade(self, ev: TradeEvent) -> None:
        self.last_trade_price = ev.trade_price
        spread = self.best_ask - self.best_bid if (self.best_bid and self.best_ask) else 0.0
        if ev.aggressor == 1:
            side = 1
        elif ev.aggressor == 2:
            side = 0
        else:
            side = 1 if (ev.trade_price >= self.best_ask and self.best_ask > 0) else 0
        now_exch = self._exchange_ts(ev.ssboe, ev.usecs)
        await self._process_event(3, side, ev.trade_price, int(ev.trade_size),
                                   spread, now_exch, ev.recv_ts, ev.symbol)

    @staticmethod
    def _exchange_ts(ssboe: int, usecs: int) -> float:
        return float(ssboe) + float(usecs) * 1e-6

    # --------------------------------------------------------------- core pipeline
    async def _process_event(self, event_type, side, price, qty,
                              spread, exch_ts, local_ts, symbol) -> None:
        self.stats.events_processed += 1

        if self.last_event_ts is None:
            time_delta = 0.0
        else:
            time_delta = max(0.0, exch_ts - self.last_event_ts)
        self.last_event_ts = exch_ts

        feat = self.features.update(
            event_type=event_type, side=side, price=float(price),
            qty=float(qty), spread=float(spread), time_delta=float(time_delta),
        )
        tier, pred = self.model.predict_tier(feat)

        # Warmup gate
        if self.features.n_events < self.features.W500:
            return

        # Spread filter (from trading cards)
        if spread > self.cfg.max_spread_ticks * self.cfg.tick_size:
            return

        # Threshold / tier gate
        if abs(pred) < self.cfg.threshold:
            return
        tier_rank = TIER_ORDER.get(tier, 0)
        min_rank = TIER_ORDER.get(self.cfg.min_tier, 2)
        if tier_rank < min_rank:
            return

        self.stats.signals_generated += 1
        desired_side = "B" if pred > 0 else "S"

        # Log signal
        sig_data = {
            "ts": local_ts, "exch_ts": exch_ts, "symbol": symbol,
            "side": desired_side, "pred": float(pred), "tier": tier,
            "bid": self.best_bid, "ask": self.best_ask,
            "pos": self.position.size, "unrealized_pnl": self.position.unrealized_pnl,
            "features_hash": hashlib.sha1(feat.tobytes()).hexdigest()[:12],
        }
        self._signals_fh.write(json.dumps(sig_data) + "\n")

        await self._act_on_signal(desired_side, pred, tier, price, exch_ts, local_ts)

    # --------------------------------------------------------------- simulated execution
    async def _act_on_signal(self, desired_side: str, pred: float, tier: str,
                              ref_price: float, exch_ts: float, local_ts: float) -> None:
        pos = self.position

        if pos.size == 0:
            # Open new position
            fill = self._simulate_fill(desired_side, 1, exch_ts, local_ts, pred, tier, "open")
            pos.size = +1 if desired_side == "B" else -1
            pos.side = desired_side
            pos.entry_price = fill.fill_price
            pos.entry_ts = local_ts
            pos.entry_pred = pred
            pos.entry_tier = tier
            self.stats.signals_traded += 1
            log.info("PAPER OPEN %s 1 @ %.2f (pred=%+.4f tier=%s bid=%.2f ask=%.2f)",
                     desired_side, fill.fill_price, pred, tier, self.best_bid, self.best_ask)
            return

        same_dir = (pos.size > 0 and desired_side == "B") or \
                   (pos.size < 0 and desired_side == "S")
        if same_dir:
            return  # already positioned

        # Flip: close current + open new
        close_side = "S" if pos.size > 0 else "B"
        close_fill = self._simulate_fill(close_side, abs(pos.size), exch_ts, local_ts,
                                          pred, tier, "close:flip")
        self._record_trade(pos, close_fill, "signal_flip")

        open_fill = self._simulate_fill(desired_side, 1, exch_ts, local_ts,
                                         pred, tier, "open:flip")
        pos.size = +1 if desired_side == "B" else -1
        pos.side = desired_side
        pos.entry_price = open_fill.fill_price
        pos.entry_ts = local_ts
        pos.entry_pred = pred
        pos.entry_tier = tier
        self.stats.signals_traded += 1
        log.info("PAPER FLIP → %s 1 @ %.2f (closed @ %.2f | pred=%+.4f tier=%s)",
                 desired_side, open_fill.fill_price, close_fill.fill_price, pred, tier)

    def _simulate_fill(self, side: str, qty: int, exch_ts: float, local_ts: float,
                        pred: float, tier: str, reason: str) -> SimulatedFill:
        """Simulate a market order fill using current BBO + slippage."""
        self._fill_counter += 1
        slip = self.cfg.slippage_ticks * self.cfg.tick_size

        if side == "B":
            # Buy at ask + slippage
            fill_price = self.best_ask + slip if self.best_ask > 0 else self.last_trade_price + slip
        else:
            # Sell at bid - slippage
            fill_price = self.best_bid - slip if self.best_bid > 0 else self.last_trade_price - slip

        commission = self.cfg.commission_per_side * qty

        fill = SimulatedFill(
            fill_id=self._fill_counter,
            timestamp=local_ts,
            exchange_ts=exch_ts,
            symbol=self.cfg.symbol,
            side=side,
            qty=qty,
            fill_price=fill_price,
            market_bid=self.best_bid,
            market_ask=self.best_ask,
            slippage=slip,
            commission=commission,
            reason=reason,
            signal_pred=pred,
            signal_tier=tier,
        )

        self._fills_fh.write(json.dumps(asdict(fill)) + "\n")
        return fill

    def _record_trade(self, pos: PaperPosition, close_fill: SimulatedFill,
                       exit_reason: str) -> None:
        """Record a completed round-trip trade."""
        self._trade_counter += 1

        if pos.size > 0:  # was long
            gross_pnl = (close_fill.fill_price - pos.entry_price) * self.cfg.point_value * abs(pos.size)
        else:  # was short
            gross_pnl = (pos.entry_price - close_fill.fill_price) * self.cfg.point_value * abs(pos.size)

        # Commission: entry + exit
        total_commission = self.cfg.commission_per_side * abs(pos.size) * 2
        net_pnl = gross_pnl - total_commission
        hold_time = close_fill.timestamp - pos.entry_ts

        trade = PaperTrade(
            trade_id=self._trade_counter,
            entry_time=pos.entry_ts,
            exit_time=close_fill.timestamp,
            entry_price=pos.entry_price,
            exit_price=close_fill.fill_price,
            side=pos.side,
            qty=abs(pos.size),
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            commission=total_commission,
            hold_time_s=hold_time,
            exit_reason=exit_reason,
            entry_pred=pos.entry_pred,
            entry_tier=pos.entry_tier,
        )

        self.trades.append(trade)
        self._trades_fh.write(json.dumps(asdict(trade)) + "\n")

        # Update stats
        self.stats.total_trades += 1
        self.stats.total_gross_pnl += gross_pnl
        self.stats.total_net_pnl += net_pnl
        self.stats.total_commission += total_commission
        self.stats.returns.append(net_pnl)

        if net_pnl > 0:
            self.stats.winning_trades += 1
        else:
            self.stats.losing_trades += 1

        # Drawdown tracking
        if self.stats.total_net_pnl > self.stats.peak_pnl:
            self.stats.peak_pnl = self.stats.total_net_pnl
        self.stats.current_drawdown = self.stats.peak_pnl - self.stats.total_net_pnl
        if self.stats.current_drawdown > self.stats.max_drawdown:
            self.stats.max_drawdown = self.stats.current_drawdown

        emoji = "✅" if net_pnl > 0 else "❌"
        log.info("PAPER TRADE #%d %s %s 1x%s entry=%.2f exit=%.2f "
                 "gross=$%.2f net=$%.2f hold=%.1fs (%s) | cumPnL=$%.2f",
                 trade.trade_id, emoji, trade.side, self.cfg.symbol,
                 trade.entry_price, trade.exit_price,
                 gross_pnl, net_pnl, hold_time, exit_reason,
                 self.stats.total_net_pnl)

    # --------------------------------------------------------------- timeout
    async def _close_position_timeout(self) -> None:
        """Force-close position on timeout (simulated)."""
        pos = self.position
        if pos.size == 0:
            return
        close_side = "S" if pos.size > 0 else "B"
        exch_ts = self.last_event_ts or time.time()
        local_ts = time.time()
        close_fill = self._simulate_fill(close_side, abs(pos.size), exch_ts, local_ts,
                                          0.0, "", "close:timeout")
        self._record_trade(pos, close_fill, "timeout")
        log.info("PAPER TIMEOUT — closed %s @ %.2f", close_side, close_fill.fill_price)
        pos.size = 0
        pos.side = ""
        pos.entry_price = 0.0
        pos.entry_ts = 0.0

    async def _position_timeout_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
                pos = self.position
                if pos.size != 0 and pos.entry_ts > 0:
                    age = time.time() - pos.entry_ts
                    if age >= self.cfg.position_timeout_s:
                        await self._close_position_timeout()
        except asyncio.CancelledError:
            pass

    # --------------------------------------------------------------- stats
    async def _stats_loop(self) -> None:
        """Periodically dump performance stats."""
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self.cfg.stats_interval_s)
                self._dump_stats()
        except asyncio.CancelledError:
            pass

    def _dump_stats(self) -> None:
        s = self.stats
        elapsed = time.time() - s.start_time
        elapsed_min = elapsed / 60

        log.info("=" * 70)
        log.info("PAPER TRADING STATS (%.1f min elapsed)", elapsed_min)
        log.info("  Events processed: %d | Signals: %d | Traded: %d",
                 s.events_processed, s.signals_generated, s.signals_traded)
        log.info("  Trades: %d (W:%d L:%d) | Win rate: %.1f%%",
                 s.total_trades, s.winning_trades, s.losing_trades, s.win_rate * 100)
        log.info("  Gross P&L: $%.2f | Net P&L: $%.2f | Commission: $%.2f",
                 s.total_gross_pnl, s.total_net_pnl, s.total_commission)
        log.info("  Avg P&L/trade: $%.2f | Profit Factor: %.2f",
                 s.avg_pnl, s.profit_factor)
        log.info("  Sortino: %.2f | Max Drawdown: $%.2f | Current DD: $%.2f",
                 s.sortino, s.max_drawdown, s.current_drawdown)
        if self.position.size != 0:
            log.info("  POSITION: %+d @ %.2f | Unrealized: $%.2f",
                     self.position.size, self.position.entry_price,
                     self.position.unrealized_pnl)
        else:
            log.info("  POSITION: FLAT")
        log.info("=" * 70)

    def _dump_final_stats(self) -> None:
        """Final stats dump on shutdown."""
        log.info("=" * 70)
        log.info("FINAL PAPER TRADING REPORT")
        log.info("=" * 70)
        self._dump_stats()

        # Per-trade summary
        if self.trades:
            pnls = [t.net_pnl for t in self.trades]
            holds = [t.hold_time_s for t in self.trades]
            log.info("  Best trade:  $%.2f", max(pnls))
            log.info("  Worst trade: $%.2f", min(pnls))
            log.info("  Avg hold:    %.1fs", statistics.mean(holds))
            log.info("  Median hold: %.1fs", statistics.median(holds))

        # Save stats to JSON
        stats_path = _LOG_DIR / f"paper_stats_{self.cfg.symbol}.json"
        stats_dict = {
            "symbol": self.cfg.symbol,
            "start_time": datetime.fromtimestamp(self.stats.start_time, tz=timezone.utc).isoformat(),
            "end_time": datetime.now(tz=timezone.utc).isoformat(),
            "total_trades": self.stats.total_trades,
            "win_rate": self.stats.win_rate,
            "total_net_pnl": self.stats.total_net_pnl,
            "total_gross_pnl": self.stats.total_gross_pnl,
            "total_commission": self.stats.total_commission,
            "avg_pnl": self.stats.avg_pnl,
            "sortino": self.stats.sortino,
            "profit_factor": self.stats.profit_factor,
            "max_drawdown": self.stats.max_drawdown,
            "signals_generated": self.stats.signals_generated,
            "signals_traded": self.stats.signals_traded,
            "events_processed": self.stats.events_processed,
            "config": asdict(self.cfg),
        }
        with open(stats_path, "w") as f:
            json.dump(stats_dict, f, indent=2)
        log.info("Stats saved to %s", stats_path)
        log.info("Trades log: %s", self._trades_log_path)
        log.info("Fills log:  %s", self._fills_log_path)
        log.info("Signals log: %s", self._signals_log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Paper Trading Engine — real Rithmic feed, simulated fills. "
                    "NEVER sends real orders.")
    ap.add_argument("--model", required=True,
                    help="Path to model .pkl (joblib)")
    ap.add_argument("--calibration", default=None,
                    help="Path to calibration JSON")
    ap.add_argument("--symbol", default="ESM6",
                    help="Trading symbol (default: ESM6)")
    ap.add_argument("--exchange", default="CME",
                    help="Exchange (default: CME)")
    ap.add_argument("--tier", default="top10",
                    choices=list(TIER_ORDER.keys()),
                    help="Minimum confidence tier (default: top10)")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Minimum |prediction| threshold")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="Position timeout in seconds (default: 30)")
    ap.add_argument("--slippage", type=float, default=1.0,
                    help="Slippage in ticks (default: 1)")
    ap.add_argument("--commission", type=float, default=0.50,
                    help="Commission per side per contract (default: $0.50)")
    ap.add_argument("--max-spread", type=float, default=3.0,
                    help="Max spread in ticks to trade (default: 3.0, cards recommend 0.75)")
    ap.add_argument("--stats-interval", type=float, default=300.0,
                    help="Stats dump interval in seconds (default: 300)")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    config = PaperConfig(
        symbol=args.symbol,
        exchange=args.exchange,
        min_tier=args.tier,
        threshold=args.threshold,
        position_timeout_s=args.timeout,
        slippage_ticks=args.slippage,
        commission_per_side=args.commission,
        max_spread_ticks=args.max_spread,
        stats_interval_s=args.stats_interval,
    )

    engine = PaperEngine(config, args.model, args.calibration)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_sig(*_a):
        log.info("Shutdown signal received...")
        engine.stop()

    for s in (_sig.SIGINT, _sig.SIGTERM):
        try:
            loop.add_signal_handler(s, _handle_sig)
        except NotImplementedError:
            _sig.signal(s, lambda *_: engine.stop())

    try:
        loop.run_until_complete(engine.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
