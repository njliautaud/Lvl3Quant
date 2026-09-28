#!/usr/bin/env python3
"""
monday_paper_runner.py — Multi-strategy paper trading runner for Monday market open.

Runs 13 execution strategies SIMULTANEOUSLY on the same CNN-Mamba/Mamba signal stream.
Each strategy has independent position state, entry/exit logic, and P&L tracking.

Strategies (Original v1):
    1. midday_z2.5_30s     — 11:00-14:00 ET, z>=2.5, 30s hold, signal-flip exit
    2. open_30min_z2.5_30s — 9:30-10:00 ET, z>=2.5, 30s hold
    3. bracket_wide_z5.0   — All hours, z>=5.0, SL=4 ticks, TP=8 ticks, 60s max
    4. bracket_bal_z5.0    — All hours, z>=5.0, SL=3 ticks, TP=4 ticks, 60s max
    5. baseline_z2.5_30s   — All hours, z>=2.5, 30s hold (control)

New v5 Strategies (backtested +$900-$814/5 days on real fills):
    6. momentum_3_z2.5_30s     — 3 consecutive preds agree, z>=2.5, 30s hold
    7. momentum_2_z3.0_midday  — 2 consecutive + z>=3.0 + midday filter
    8. momentum_3_z5.0_bracket — 3 consecutive + z>=5.0 + SL3/TP4 bracket
    9. agree_all3_z3.0_midday  — All 3 horizons agree + z>=3.0 + midday
   10. agree_all3_z5.0_bracket — All 3 horizons agree + z>=5.0 + SL4/TP8

Signal source: Mamba v7 (or CNN-Mamba v2 fallback) running CPU inference.

Modes:
    --live       Connect to Rithmic for real-time MBO data
    --backtest FILE  Replay recorded MBO NPZ data (for pre-Monday validation)
    --dry-run    Run inference only, log predictions, no position tracking

Safety:
    - PAPER ONLY — no real orders ever
    - Max 1 position per strategy (5 total possible)
    - $500 daily loss limit per strategy
    - Kill switch: touch /tmp/kill_paper_trading
    - Graceful shutdown on SIGINT/SIGTERM

Usage:
    # Live trading Monday:
    python3 monday_paper_runner.py --live

    # Backtest on Friday's data:
    python3 monday_paper_runner.py --backtest /home/jupiter/Lvl3Quant/data/processed/mbo_events/20260424_mbo_events.npz

    # Dry-run (inference only, no positions):
    python3 monday_paper_runner.py --backtest FILE --dry-run

    # PM2 deployment:
    pm2 start monday_paper_runner.py --name paper-5strat --interpreter python3 -- --live

IMPORTANT: PAPER TRADE ONLY. No real orders are ever submitted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import signal as _sig
import statistics
import sys
import time
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — works standalone or as module
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from streaming_features_smart_v3 import StreamingFeaturesSmartV3, N_FEATURES
from mamba_inference import MambaInferenceEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LVL3 = Path("/home/jupiter/Lvl3Quant")

# ES futures
TICK_SIZE = 0.25
POINT_VALUE = 50.0       # $50 per point for ES
TICK_VALUE = 12.50        # $12.50 per tick
COMMISSION_RT = 4.70      # $4.70 round-trip
COMMISSION_PER_SIDE = 2.35

# Model paths (Mamba v7 primary, CNN-Mamba v2 if available)
MAMBA_V7_WEIGHTS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_best.pt"
MAMBA_V7_STATS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/fold_15_feature_stats.npz"
MAMBA_V7_PREDS = LVL3 / "output/mamba_v7_tiny_smart_v3_mar_apr/concat_oot_predictions.npz"

# CNN-Mamba v2 (preferred if exists)
CNN_MAMBA_V2_WEIGHTS = LVL3 / "output/cnn_mamba_v2_smart_v3/fold_best.pt"
CNN_MAMBA_V2_STATS = LVL3 / "output/cnn_mamba_v2_smart_v3/fold_feature_stats.npz"
CNN_MAMBA_V2_PREDS = LVL3 / "output/cnn_mamba_v2_smart_v3/concat_oot_predictions.npz"

# MBO data
MBO_DATA_DIR = LVL3 / "data/processed/mbo_events"

# Safety
KILL_SWITCH_PATH = Path("/tmp/kill_paper_trading")
DAILY_LOSS_LIMIT_PER_STRATEGY = 500.0  # $500

# Inference
WINDOW_SIZE = 1000
STRIDE = 500

# Discord webhook for notifications
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_PAPER_WEBHOOK", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = _SCRIPT_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

log = logging.getLogger("monday_paper")
if not log.handlers:
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_DIR / f"monday_paper_{_SESSION_ID}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(sh)


# ---------------------------------------------------------------------------
# Discord notification helper
# ---------------------------------------------------------------------------
def discord_notify(message: str, urgent: bool = False):
    """Send Discord notification via webhook. Non-blocking, swallows errors."""
    url = DISCORD_WEBHOOK_URL
    if not url:
        return
    try:
        # Try loading from config if env var not set
        webhook_cfg = _SCRIPT_DIR.parent.parent / "teleclaude-main/config/webhooks.json"
        if not url and webhook_cfg.exists():
            cfg = json.loads(webhook_cfg.read_text())
            url = cfg.get("webhooks", {}).get("paper_trading", "")
        if not url:
            return

        payload = json.dumps({"content": message[:2000]}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.debug("Discord notify failed: %s", e)


# ============================================================================
# Time utilities (Eastern Time)
# ============================================================================
def now_et() -> datetime:
    """Current time in US/Eastern."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def time_in_range(start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
    """Check if current ET time is within [start, end)."""
    t = now_et()
    start = t.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = t.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return start <= t < end


def backtest_time_in_range(ts_ns: int, start_h: int, start_m: int,
                           end_h: int, end_m: int) -> bool:
    """Check if a nanosecond timestamp falls within an ET time range."""
    if ts_ns <= 0:
        return True  # No timestamp, allow all (fallback)
    from zoneinfo import ZoneInfo
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=ZoneInfo("America/New_York"))
    start = dt.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end = dt.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    return start <= dt < end


# ============================================================================
# Expanding Z-Score Normalizer (no lookahead)
# ============================================================================
class ExpandingZScore:
    """Expanding-window z-score normalizer for model predictions.

    Maintains running mean/var of all predictions seen so far.
    No lookahead — safe for live/paper trading.
    """

    def __init__(self, min_samples: int = 20):
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0  # sum of squared deviations
        self.min_samples = min_samples

    def update(self, x: float) -> float:
        """Add observation, return z-score. Returns 0 until min_samples reached."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

        if self.n < self.min_samples:
            return 0.0

        variance = self.m2 / self.n
        std = math.sqrt(max(variance, 1e-12))
        return (x - self.mean) / std


# ============================================================================
# Strategy Position
# ============================================================================
@dataclass
class StrategyPosition:
    """Position state for a single strategy."""
    direction: int = 0        # +1 long, -1 short, 0 flat
    entry_price: float = 0.0
    entry_time: float = 0.0   # wall-clock time
    entry_ts_ns: int = 0      # exchange timestamp (ns)
    entry_z: float = 0.0
    entry_pred_1s: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.direction == 0

    def clear(self):
        self.direction = 0
        self.entry_price = 0.0
        self.entry_time = 0.0
        self.entry_ts_ns = 0
        self.entry_z = 0.0
        self.entry_pred_1s = 0.0


# ============================================================================
# Strategy Stats
# ============================================================================
@dataclass
class StrategyStats:
    """Performance statistics for a single strategy."""
    name: str = ""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    commission: float = 0.0
    max_drawdown: float = 0.0
    peak_pnl: float = 0.0
    returns: List[float] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_trades, 1) * 100

    @property
    def avg_pnl(self) -> float:
        return self.net_pnl / max(self.total_trades, 1)

    @property
    def sortino(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        mean_r = statistics.mean(self.returns)
        downside = [r for r in self.returns if r < 0]
        if not downside:
            return float("inf") if mean_r > 0 else 0.0
        ds = statistics.stdev(downside) if len(downside) > 1 else abs(downside[0])
        if ds == 0:
            return 0.0
        return (mean_r / ds) * math.sqrt(50 * 252)

    @property
    def profit_factor(self) -> float:
        wins_total = sum(r for r in self.returns if r > 0)
        losses_total = abs(sum(r for r in self.returns if r < 0))
        if losses_total == 0:
            return float("inf") if wins_total > 0 else 0.0
        return wins_total / losses_total

    def record_trade(self, trade: Dict):
        net = trade["net_pnl"]
        self.total_trades += 1
        self.gross_pnl += trade["gross_pnl"]
        self.net_pnl += net
        self.commission += trade.get("commission", COMMISSION_RT)
        self.returns.append(net)
        self.trades.append(trade)
        if net > 0:
            self.wins += 1
        else:
            self.losses += 1
        if self.net_pnl > self.peak_pnl:
            self.peak_pnl = self.net_pnl
        dd = self.peak_pnl - self.net_pnl
        if dd > self.max_drawdown:
            self.max_drawdown = dd


# ============================================================================
# Base Strategy
# ============================================================================
class BaseStrategy(ABC):
    """Base class for all execution strategies."""

    def __init__(self, name: str, z_threshold: float, max_hold_s: float,
                 use_brackets: bool = False, sl_ticks: float = 0, tp_ticks: float = 0,
                 time_filter: Optional[tuple] = None, signal_flip_exit: bool = True):
        self.name = name
        self.z_threshold = z_threshold
        self.max_hold_s = max_hold_s
        self.use_brackets = use_brackets
        self.sl_ticks = sl_ticks
        self.tp_ticks = tp_ticks
        self.time_filter = time_filter  # (start_h, start_m, end_h, end_m) or None
        self.signal_flip_exit = signal_flip_exit

        self.pos = StrategyPosition()
        self.stats = StrategyStats(name=name)
        self.prev_direction: Optional[int] = None
        self.halted = False  # daily loss limit breaker

    def is_time_allowed(self, ts_ns: int, is_backtest: bool) -> bool:
        """Check if current time is within strategy's trading window."""
        if self.time_filter is None:
            return True
        sh, sm, eh, em = self.time_filter
        if is_backtest:
            return backtest_time_in_range(ts_ns, sh, sm, eh, em)
        return time_in_range(sh, sm, eh, em)

    def check_daily_loss_limit(self) -> bool:
        """Returns True if strategy should be halted."""
        if self.stats.net_pnl <= -DAILY_LOSS_LIMIT_PER_STRATEGY:
            if not self.halted:
                self.halted = True
                log.warning("STRATEGY HALTED: %s hit daily loss limit ($%.2f)",
                            self.name, self.stats.net_pnl)
                discord_notify(
                    f"HALT {self.name}: Daily loss limit hit (${self.stats.net_pnl:.2f})",
                    urgent=True,
                )
            return True
        return False

    def on_signal(
        self,
        pred_1s: float,
        pred_5s: float,
        pred_10s: float,
        z_score: float,
        direction: int,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        wall_time: float,
        ts_ns: int,
        is_backtest: bool,
        dry_run: bool,
    ) -> Optional[Dict]:
        """Process a new prediction signal. Returns trade dict if exit occurred."""
        if dry_run:
            self.prev_direction = direction
            return None

        if self.halted or self.check_daily_loss_limit():
            return None

        trade = None

        # --- CHECK EXIT CONDITIONS (if in position) ---
        if not self.pos.is_flat:
            should_exit, exit_reason = self._check_exit(
                direction, z_score, mid_price, best_bid, best_ask, wall_time, ts_ns
            )
            if should_exit:
                trade = self._exit_position(mid_price, wall_time, ts_ns, exit_reason)

        # --- CHECK ENTRY CONDITIONS (if flat) ---
        if self.pos.is_flat:
            if self._check_entry(direction, z_score, mid_price, wall_time, ts_ns, is_backtest):
                self._enter_position(direction, mid_price, wall_time, ts_ns, z_score, pred_1s)

        self.prev_direction = direction
        return trade

    def on_price_update(
        self,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        wall_time: float,
        ts_ns: int,
    ) -> Optional[Dict]:
        """Check bracket/timeout exits on every price update (for bracket strategies).
        Called more frequently than on_signal (every event vs every stride).
        Returns trade dict if exit occurred.
        """
        if self.pos.is_flat or self.halted:
            return None

        should_exit, reason = self._check_exit(
            self.prev_direction or 0, 0.0,
            mid_price, best_bid, best_ask, wall_time, ts_ns,
        )
        if should_exit:
            return self._exit_position(mid_price, wall_time, ts_ns, reason)
        return None

    def _check_entry(self, direction: int, z_score: float, mid_price: float,
                     wall_time: float, ts_ns: int, is_backtest: bool) -> bool:
        """Check if we should enter a position."""
        if abs(z_score) < self.z_threshold:
            return False
        if not self.is_time_allowed(ts_ns, is_backtest):
            return False
        if mid_price <= 0:
            return False
        return True

    def _check_exit(self, direction: int, z_score: float, mid_price: float,
                    best_bid: float, best_ask: float,
                    wall_time: float, ts_ns: int) -> tuple:
        """Check exit conditions. Returns (should_exit, reason)."""

        hold_time = wall_time - self.pos.entry_time

        # 1. Timeout
        if hold_time >= self.max_hold_s:
            return True, "timeout"

        # 2. Signal flip exit
        if self.signal_flip_exit and self.prev_direction is not None:
            if direction != 0 and direction != self.pos.direction:
                return True, "signal_flip"

        # 3. Bracket exits (SL/TP)
        if self.use_brackets and mid_price > 0:
            if self.pos.direction > 0:  # long
                pnl_ticks = (mid_price - self.pos.entry_price) / TICK_SIZE
            else:  # short
                pnl_ticks = (self.pos.entry_price - mid_price) / TICK_SIZE

            if self.sl_ticks > 0 and pnl_ticks <= -self.sl_ticks:
                return True, "stop_loss"
            if self.tp_ticks > 0 and pnl_ticks >= self.tp_ticks:
                return True, "take_profit"

        return False, ""

    def _enter_position(self, direction: int, mid_price: float, wall_time: float,
                        ts_ns: int, z_score: float, pred_1s: float):
        """Enter a new position."""
        # Slippage: 1 tick adverse for market entry
        slip = TICK_SIZE
        if direction > 0:
            entry_price = mid_price + slip / 2  # buy slightly above mid
        else:
            entry_price = mid_price - slip / 2  # sell slightly below mid

        self.pos.direction = direction
        self.pos.entry_price = entry_price
        self.pos.entry_time = wall_time
        self.pos.entry_ts_ns = ts_ns
        self.pos.entry_z = z_score
        self.pos.entry_pred_1s = pred_1s

        dir_str = "LONG" if direction > 0 else "SHORT"
        log.info("[%s] ENTRY %s @ %.2f | z=%.2f | pred_1s=%.4f",
                 self.name, dir_str, entry_price, z_score, pred_1s)

        discord_notify(
            f"{'LONG' if direction > 0 else 'SHORT'} {self.name}: "
            f"{'LONG' if direction > 0 else 'SHORT'} ES @ {entry_price:.2f}, "
            f"z={z_score:.2f}"
        )

    def _exit_position(self, mid_price: float, wall_time: float,
                       ts_ns: int, reason: str) -> Dict:
        """Exit current position and record trade."""
        # Slippage: 1 tick adverse for market exit
        slip = TICK_SIZE
        if self.pos.direction > 0:
            exit_price = mid_price - slip / 2  # sell slightly below mid
        else:
            exit_price = mid_price + slip / 2  # buy slightly above mid

        # P&L
        if self.pos.direction > 0:
            gross_pnl = (exit_price - self.pos.entry_price) * POINT_VALUE
        else:
            gross_pnl = (self.pos.entry_price - exit_price) * POINT_VALUE

        net_pnl = gross_pnl - COMMISSION_RT
        hold_time = wall_time - self.pos.entry_time
        ticks_pnl = gross_pnl / TICK_VALUE

        trade = {
            "strategy": self.name,
            "direction": "LONG" if self.pos.direction > 0 else "SHORT",
            "entry_price": round(self.pos.entry_price, 2),
            "exit_price": round(exit_price, 2),
            "gross_pnl": round(gross_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "commission": COMMISSION_RT,
            "ticks": round(ticks_pnl, 2),
            "hold_time_s": round(hold_time, 2),
            "reason": reason,
            "entry_z": round(self.pos.entry_z, 3),
            "entry_pred_1s": round(self.pos.entry_pred_1s, 6),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ts_ns": ts_ns,
        }

        self.stats.record_trade(trade)
        self.pos.clear()

        emoji = "+" if net_pnl > 0 else ""
        log.info("[%s] EXIT %s @ %.2f | reason=%s | %s$%.2f (%.1f ticks, %.1fs) | "
                 "cumP&L=$%.2f (%d trades, %.0f%% WR)",
                 self.name, trade["direction"], exit_price, reason,
                 emoji, net_pnl, ticks_pnl, hold_time,
                 self.stats.net_pnl, self.stats.total_trades, self.stats.win_rate)

        result_emoji = "+" if net_pnl > 0 else ""
        discord_notify(
            f"{'CLOSED' if net_pnl >= 0 else 'STOPPED'} {self.name}: "
            f"{trade['direction']} {result_emoji}${net_pnl:.2f} "
            f"({ticks_pnl:.1f} ticks, {hold_time:.1f}s, {reason})"
        )

        return trade


# ============================================================================
# Concrete Strategy Implementations
# ============================================================================

class MiddayZ25Strategy(BaseStrategy):
    """Strategy 1: Midday z2.5, 30s hold, signal-flip exit. 11:00-14:00 ET."""
    def __init__(self):
        super().__init__(
            name="midday_z2.5_30s",
            z_threshold=2.5,
            max_hold_s=30.0,
            time_filter=(11, 0, 14, 0),
            signal_flip_exit=True,
        )


class Open30minZ25Strategy(BaseStrategy):
    """Strategy 2: Open 30min z2.5, 30s hold. 9:30-10:00 ET."""
    def __init__(self):
        super().__init__(
            name="open_30min_z2.5_30s",
            z_threshold=2.5,
            max_hold_s=30.0,
            time_filter=(9, 30, 10, 0),
            signal_flip_exit=True,
        )


class BracketWideZ5Strategy(BaseStrategy):
    """Strategy 3: Ultra-selective z>=5.0, SL=4 ticks, TP=8 ticks, 60s max."""
    def __init__(self):
        super().__init__(
            name="bracket_wide_z5.0",
            z_threshold=5.0,
            max_hold_s=60.0,
            use_brackets=True,
            sl_ticks=4.0,
            tp_ticks=8.0,
            signal_flip_exit=False,  # brackets handle exit
        )


class BracketBalancedZ5Strategy(BaseStrategy):
    """Strategy 4: Ultra-selective z>=5.0, SL=3 ticks, TP=4 ticks, 60s max."""
    def __init__(self):
        super().__init__(
            name="bracket_bal_z5.0",
            z_threshold=5.0,
            max_hold_s=60.0,
            use_brackets=True,
            sl_ticks=3.0,
            tp_ticks=4.0,
            signal_flip_exit=False,
        )


class BaselineZ25Strategy(BaseStrategy):
    """Strategy 5: All hours, z>=2.5, 30s hold (control)."""
    def __init__(self):
        super().__init__(
            name="baseline_z2.5_30s",
            z_threshold=2.5,
            max_hold_s=30.0,
            signal_flip_exit=True,
        )


# ── NEW: Momentum Filter Strategies (v5 discovery) ──────────────────────────

class MomentumFilterStrategy(BaseStrategy):
    """Strategy with momentum filter: require N consecutive predictions
    to agree on direction before entering.

    This was the #1 strategy discovery in backtesting:
    momentum_3_z2.5_30s: +$900/5days, Sortino=72.6
    momentum_2_z3.0_midday: +$814/5days, PF=2.08
    """

    def __init__(self, name: str, z_threshold: float, max_hold_s: float,
                 n_consecutive: int = 3,
                 use_brackets: bool = False, sl_ticks: float = 0, tp_ticks: float = 0,
                 time_filter: Optional[tuple] = None, signal_flip_exit: bool = True):
        super().__init__(
            name=name,
            z_threshold=z_threshold,
            max_hold_s=max_hold_s,
            use_brackets=use_brackets,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            time_filter=time_filter,
            signal_flip_exit=signal_flip_exit,
        )
        self.n_consecutive = n_consecutive
        self.direction_history: deque = deque(maxlen=n_consecutive)

    def _check_entry(self, direction: int, z_score: float, mid_price: float,
                     wall_time: float, ts_ns: int, is_backtest: bool) -> bool:
        """Override to add momentum filter."""
        # Track direction history
        self.direction_history.append(direction)

        # Basic checks first
        if abs(z_score) < self.z_threshold:
            return False
        if not self.is_time_allowed(ts_ns, is_backtest):
            return False
        if mid_price <= 0:
            return False

        # Momentum check: all last N predictions must agree on direction
        if len(self.direction_history) < self.n_consecutive:
            return False
        if direction > 0:
            if not all(d > 0 for d in self.direction_history):
                return False
        elif direction < 0:
            if not all(d < 0 for d in self.direction_history):
                return False
        else:
            return False  # direction=0 means no signal

        return True


class Momentum3Z25Strategy(MomentumFilterStrategy):
    """TOP STRATEGY: 3 consecutive agree, z>=2.5, 30s hold.
    Backtest: +$900/5 days, Sortino=72.6"""
    def __init__(self):
        super().__init__(
            name="momentum_3_z2.5_30s",
            z_threshold=2.5,
            max_hold_s=30.0,
            n_consecutive=3,
            signal_flip_exit=True,
        )


class Momentum2Z30MiddayStrategy(MomentumFilterStrategy):
    """2 consecutive agree, z>=3.0, midday filter, 30s hold.
    Backtest: +$814/5 days, PF=2.08, $24.66/trade avg"""
    def __init__(self):
        super().__init__(
            name="momentum_2_z3.0_midday",
            z_threshold=3.0,
            max_hold_s=30.0,
            n_consecutive=2,
            time_filter=(10, 0, 14, 0),
            signal_flip_exit=True,
        )


class Momentum3Z50BracketStrategy(MomentumFilterStrategy):
    """3 consecutive + ultra-conviction z>=5.0 + bracket SL3/TP4.
    Backtest: +$178/5 days, 83.3% WR, PF=4.67"""
    def __init__(self):
        super().__init__(
            name="momentum_3_z5.0_bracket",
            z_threshold=5.0,
            max_hold_s=60.0,
            n_consecutive=3,
            use_brackets=True,
            sl_ticks=3.0,
            tp_ticks=4.0,
            signal_flip_exit=False,
        )


# ── NEW: Multi-Horizon Agreement Strategy ────────────────────────────────────

class MultiHorizonAgreementStrategy(BaseStrategy):
    """Require all 3 horizons (1s, 5s, 10s) to agree on direction.
    Backtest: agree_all3_z3.0_midday: +$809/5 days, PF=1.46"""

    def __init__(self, name: str, z_threshold: float, max_hold_s: float,
                 time_filter: Optional[tuple] = None,
                 use_brackets: bool = False, sl_ticks: float = 0, tp_ticks: float = 0):
        super().__init__(
            name=name,
            z_threshold=z_threshold,
            max_hold_s=max_hold_s,
            time_filter=time_filter,
            use_brackets=use_brackets,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            signal_flip_exit=not use_brackets,
        )
        # Will be set by the runner before calling on_signal
        self._last_preds = (0.0, 0.0, 0.0)

    def set_multi_horizon_preds(self, pred_1s: float, pred_5s: float, pred_10s: float):
        """Called by runner to pass all 3 horizon predictions."""
        self._last_preds = (pred_1s, pred_5s, pred_10s)

    def _check_entry(self, direction: int, z_score: float, mid_price: float,
                     wall_time: float, ts_ns: int, is_backtest: bool) -> bool:
        """Override: all 3 horizons must agree on direction."""
        if abs(z_score) < self.z_threshold:
            return False
        if not self.is_time_allowed(ts_ns, is_backtest):
            return False
        if mid_price <= 0:
            return False

        p1, p5, p10 = self._last_preds
        # All must be same sign (positive for long, negative for short)
        if direction > 0:
            if not (p1 > 0 and p5 > 0 and p10 > 0):
                return False
        elif direction < 0:
            if not (p1 < 0 and p5 < 0 and p10 < 0):
                return False
        else:
            return False

        return True


class AgreeAll3Z30MiddayStrategy(MultiHorizonAgreementStrategy):
    """All 3 horizons agree, z>=3.0, midday, 30s hold.
    Backtest: +$809/5 days, PF=1.46"""
    def __init__(self):
        super().__init__(
            name="agree_all3_z3.0_midday",
            z_threshold=3.0,
            max_hold_s=30.0,
            time_filter=(10, 0, 14, 0),
        )


class AgreeAll3Z50BracketStrategy(MultiHorizonAgreementStrategy):
    """All 3 horizons agree, z>=5.0, bracket SL4/TP8.
    Backtest: +$458/5 days, 61.5% WR, PF=2.50"""
    def __init__(self):
        super().__init__(
            name="agree_all3_z5.0_bracket",
            z_threshold=5.0,
            max_hold_s=60.0,
            use_brackets=True,
            sl_ticks=4.0,
            tp_ticks=8.0,
        )


# ── NEW: Midday-Optimized Strategies (v6 — time-only exit, NO signal flip) ──

class MiddayZ35_60sTimeStrategy(BaseStrategy):
    """V6 #1: Midday 10-14 ET, z>=3.5, 60s hard timeout, NO signal-flip exit.
    Backtest: +$1,190/5d, Sortino=39.76, PF=2.22, WR=61.5%, 3/5 green days.
    Cross-validated: midday filter is ROBUST across models."""
    def __init__(self):
        super().__init__(
            name="midday_z3.5_60s_time",
            z_threshold=3.5,
            max_hold_s=60.0,
            time_filter=(10, 0, 14, 0),
            signal_flip_exit=False,  # KEY: v6 found signal flip ALWAYS loses
        )


class MiddayZ30_45sTimeStrategy(BaseStrategy):
    """V6 #2: Midday 10-14 ET, z>=3.0, 45s hold, time-only exit.
    Backtest: +$1,015/5d, Sortino=25.36, PF=1.44, WR=57.4%"""
    def __init__(self):
        super().__init__(
            name="midday_z3.0_45s_time",
            z_threshold=3.0,
            max_hold_s=45.0,
            time_filter=(10, 0, 14, 0),
            signal_flip_exit=False,
        )


class MiddayZ30_20sTimeStrategy(BaseStrategy):
    """V6 most consistent: Midday z>=3.0, 20s hold, time-only.
    Backtest: +$1,085/5d, Sortino=15.70, PF=1.81, 4/5 green days (MOST CONSISTENT)."""
    def __init__(self):
        super().__init__(
            name="midday_z3.0_20s_time",
            z_threshold=3.0,
            max_hold_s=20.0,
            time_filter=(10, 0, 14, 0),
            signal_flip_exit=False,
        )


# ============================================================================
# Multi-Strategy Runner
# ============================================================================

class MultiStrategyRunner:
    """
    Orchestrates 5 strategies on a shared signal stream.

    Event flow:
        Raw MBO event
        -> StreamingFeaturesSmartV3 (25 features)
        -> MambaInferenceEngine (window=1000, stride=500)
        -> ExpandingZScore (no lookahead normalization)
        -> All 5 strategies evaluate simultaneously
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        stats_path: Optional[str] = None,
        preds_path: Optional[str] = None,
        symbol: str = "ESM6",
        exchange: str = "CME",
        dry_run: bool = False,
        backtest_mode: bool = False,
    ):
        self.symbol = symbol
        self.exchange = exchange
        self.dry_run = dry_run
        self.is_backtest = backtest_mode

        # Resolve model paths
        w, s, p = self._resolve_model_paths(weights_path, stats_path, preds_path)

        # Feature engine
        self.features = StreamingFeaturesSmartV3()

        # Mamba inference
        log.info("Loading model weights from: %s", w)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        self.engine = MambaInferenceEngine(
            weights_path=str(w),
            stats_path=str(s),
            window_size=WINDOW_SIZE,
            stride=STRIDE,
            device=_device,
        )
        if Path(p).exists():
            self.engine.calibrate_thresholds(str(p))

        # Z-score normalizer for predictions (expanding, no lookahead)
        self.zscore_1s = ExpandingZScore(min_samples=20)
        self.zscore_5s = ExpandingZScore(min_samples=20)
        self.zscore_10s = ExpandingZScore(min_samples=20)

        # Initialize all 13 strategies across 4 generations of backtesting
        self.strategies: List[BaseStrategy] = [
            # -- V6 MIDDAY-OPTIMIZED (cross-validated, MOST ROBUST) --
            MiddayZ35_60sTimeStrategy(),   # #1: $1,190/5d, Sortino=39.8, PF=2.22, WR=61.5%
            MiddayZ30_45sTimeStrategy(),   # #2: $1,015/5d, Sortino=25.4, PF=1.44, WR=57.4%
            MiddayZ30_20sTimeStrategy(),   # Most consistent: $1,085/5d, 4/5 green days

            # -- V5 Momentum filters (strong on v2, experimental on v7) --
            Momentum3Z25Strategy(),        # +$900/5d on v2, Sortino=72.6
            Momentum2Z30MiddayStrategy(),  # +$814/5d, ROBUST on both models

            # -- V1 Original (reference/control) --
            MiddayZ25Strategy(),           # +$488/4d, cross-validated positive
            Open30minZ25Strategy(),        # +$526/4d on v2 only
            BracketWideZ5Strategy(),       # +$198/4d, ultra-selective
            BracketBalancedZ5Strategy(),   # +$130/4d, high WR
            BaselineZ25Strategy(),         # Control strategy

            # -- Multi-horizon agreement --
            AgreeAll3Z30MiddayStrategy(),  # +$809/5d on v2, needs DA filter
            AgreeAll3Z50BracketStrategy(), # +$458/5d, ultra-selective
            Momentum3Z50BracketStrategy(), # +$178/5d, 83.3% WR
        ]

        # Market state
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.mid_price: float = 0.0
        self.prev_ts_ns: int = 0

        # Counters
        self.events_processed: int = 0
        self.predictions_made: int = 0

        # Signal log
        self._signal_log_path = _LOG_DIR / f"monday_signals_{_SESSION_ID}.jsonl"
        self._signal_fh = open(self._signal_log_path, "a", buffering=1)

        # Trade log (all strategies combined)
        self._trade_log_path = _LOG_DIR / f"monday_trades_{_SESSION_ID}.jsonl"
        self._trade_fh = open(self._trade_log_path, "a", buffering=1)

        # Per-second state log
        self._state_log_path = _LOG_DIR / f"monday_state_{_SESSION_ID}.jsonl"
        self._state_fh = open(self._state_log_path, "a", buffering=1)

        # Hourly summary tracking
        self._last_hourly_report = 0.0
        self._last_stats_dump = 0.0

        # Asyncio
        self._stop = asyncio.Event()

        log.info("=" * 72)
        log.info("MONDAY PAPER RUNNER — 13-Strategy Multi-Execution")
        log.info("=" * 72)
        log.info("  Model: %s", w)
        log.info("  Symbol: %s | Exchange: %s", symbol, exchange)
        log.info("  Window: %d | Stride: %d", WINDOW_SIZE, STRIDE)
        log.info("  Mode: %s", "DRY-RUN" if dry_run else ("BACKTEST" if backtest_mode else "LIVE"))
        log.info("  Strategies:")
        for strat in self.strategies:
            log.info("    - %s (z>=%.1f, %ss hold%s%s)",
                     strat.name, strat.z_threshold, strat.max_hold_s,
                     f", SL={strat.sl_ticks}/TP={strat.tp_ticks}" if strat.use_brackets else "",
                     f", {strat.time_filter}" if strat.time_filter else "")
        log.info("  Signal log: %s", self._signal_log_path)
        log.info("  Trade log:  %s", self._trade_log_path)
        log.info("  Kill switch: %s", KILL_SWITCH_PATH)
        log.info("  *** PAPER TRADE ONLY — NO REAL ORDERS ***")
        log.info("=" * 72)

        discord_notify(
            f"Monday Paper Runner STARTED ({symbol})\n"
            f"Mode: {'DRY-RUN' if dry_run else ('BACKTEST' if backtest_mode else 'LIVE')}\n"
            f"Strategies: {', '.join(s.name for s in self.strategies)}"
        )

    @staticmethod
    def _resolve_model_paths(weights, stats, preds):
        """Resolve model paths with CNN-Mamba v2 preferred, Mamba v7 fallback."""
        if weights and Path(weights).exists():
            w = Path(weights)
            s = Path(stats) if stats else w.parent / "fold_feature_stats.npz"
            p = Path(preds) if preds else w.parent / "concat_oot_predictions.npz"
            return str(w), str(s), str(p)

        # Try CNN-Mamba v2 first
        if CNN_MAMBA_V2_WEIGHTS.exists():
            log.info("Using CNN-Mamba v2 model")
            return str(CNN_MAMBA_V2_WEIGHTS), str(CNN_MAMBA_V2_STATS), str(CNN_MAMBA_V2_PREDS)

        # Fallback to Mamba v7
        if MAMBA_V7_WEIGHTS.exists():
            log.info("Using Mamba v7 model (fallback)")
            return str(MAMBA_V7_WEIGHTS), str(MAMBA_V7_STATS), str(MAMBA_V7_PREDS)

        raise FileNotFoundError(
            f"No model weights found. Checked:\n"
            f"  CNN-Mamba v2: {CNN_MAMBA_V2_WEIGHTS}\n"
            f"  Mamba v7: {MAMBA_V7_WEIGHTS}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Kill switch check
    # ──────────────────────────────────────────────────────────────────────
    def _check_kill_switch(self) -> bool:
        if KILL_SWITCH_PATH.exists():
            log.warning("KILL SWITCH ACTIVATED: %s exists. Shutting down.", KILL_SWITCH_PATH)
            discord_notify("KILL SWITCH: Paper trading stopped by /tmp/kill_paper_trading",
                           urgent=True)
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Core: process one raw MBO event
    # ──────────────────────────────────────────────────────────────────────
    def process_raw_event(
        self,
        time_delta_log: float,
        event_type_id: int,
        side_id: int,
        price_rel_ticks: float,
        qty_log: float,
        spread_ticks: float,
        timestamp_ns: int = 0,
    ):
        """Process one raw MBO event through the full pipeline."""
        self.events_processed += 1

        # Kill switch check every 10K events
        if self.events_processed % 10000 == 0 and self._check_kill_switch():
            self._stop.set()
            return

        # Step 1: Compute streaming features
        feat_vec = self.features.update(
            time_delta_log, event_type_id, side_id,
            price_rel_ticks, qty_log, spread_ticks,
        )

        # Step 2: Check bracket/timeout exits on every event (for bracket strategies)
        wall_time = time.time()
        for strat in self.strategies:
            if strat.use_brackets and not strat.pos.is_flat:
                trade = strat.on_price_update(
                    self.mid_price, self.best_bid, self.best_ask,
                    wall_time, timestamp_ns,
                )
                if trade:
                    self._log_trade(trade)

        # Step 3: Feed to Mamba inference
        pred = self.engine.add_event(feat_vec)
        if pred is None:
            return  # Not at stride boundary

        self.predictions_made += 1

        # Step 4: Compute z-scores (expanding normalization)
        pred_1s = pred["pred_1s"]
        pred_5s = pred["pred_5s"]
        pred_10s = pred["pred_10s"]

        z_1s = self.zscore_1s.update(pred_1s)
        z_5s = self.zscore_5s.update(pred_5s)
        z_10s = self.zscore_10s.update(pred_10s)

        # Use max absolute z across horizons as the signal strength
        z_score = max(abs(z_1s), abs(z_5s), abs(z_10s))
        # Direction from 1s prediction (primary horizon)
        direction = 1 if pred_1s > 0 else -1

        # Step 5: Log signal
        signal_record = {
            "n": self.predictions_made,
            "events": self.events_processed,
            "ts_ns": timestamp_ns,
            "t": datetime.now(timezone.utc).isoformat(),
            "pred_1s": round(pred_1s, 6),
            "pred_5s": round(pred_5s, 6),
            "pred_10s": round(pred_10s, 6),
            "z_1s": round(z_1s, 3),
            "z_5s": round(z_5s, 3),
            "z_10s": round(z_10s, 3),
            "z_max": round(z_score, 3),
            "dir": direction,
            "mid": round(self.mid_price, 2),
            "bid": round(self.best_bid, 2),
            "ask": round(self.best_ask, 2),
        }
        self._signal_fh.write(json.dumps(signal_record) + "\n")

        # Step 6: Distribute signal to all strategies
        wall_time = time.time()
        # Pass multi-horizon preds to agreement strategies
        for strat in self.strategies:
            if hasattr(strat, 'set_multi_horizon_preds'):
                strat.set_multi_horizon_preds(pred_1s, pred_5s, pred_10s)
        for strat in self.strategies:
            trade = strat.on_signal(
                pred_1s=pred_1s,
                pred_5s=pred_5s,
                pred_10s=pred_10s,
                z_score=z_score,
                direction=direction,
                mid_price=self.mid_price,
                best_bid=self.best_bid,
                best_ask=self.best_ask,
                wall_time=wall_time,
                ts_ns=timestamp_ns,
                is_backtest=self.is_backtest,
                dry_run=self.dry_run,
            )
            if trade:
                self._log_trade(trade)

        # Periodic logging
        if self.predictions_made % 10 == 0:
            self._log_state(timestamp_ns)

        # Hourly Discord summary (live mode)
        if not self.is_backtest:
            now = time.time()
            if now - self._last_hourly_report >= 3600:
                self._send_hourly_summary()
                self._last_hourly_report = now

        # Periodic stats dump
        now = time.time()
        if now - self._last_stats_dump >= 300:  # every 5 min
            self._dump_stats()
            self._last_stats_dump = now

    # ──────────────────────────────────────────────────────────────────────
    # Encoding: Rithmic events -> 6-col MBO format
    # ──────────────────────────────────────────────────────────────────────
    def _encode_and_process(self, ts_ns: int, etype: float, side: float,
                            price: float, qty: int):
        """Encode raw Rithmic data to 6-col MBO format and process."""
        delta_us = max(0, (ts_ns - self.prev_ts_ns) / 1000) if self.prev_ts_ns else 0
        self.prev_ts_ns = ts_ns
        price_rel = ((price - self.mid_price) / TICK_SIZE
                     if self.mid_price > 0 and price > 0 else 0.0)
        spread = ((self.best_ask - self.best_bid) / TICK_SIZE
                  if self.best_bid > 0 and self.best_ask > 0 else 0.0)

        self.process_raw_event(
            time_delta_log=math.log1p(delta_us) if delta_us > 0 else 0.0,
            event_type_id=int(etype),
            side_id=int(side),
            price_rel_ticks=price_rel,
            qty_log=math.log(max(1, qty)),
            spread_ticks=spread,
            timestamp_ns=ts_ns,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────
    def _log_trade(self, trade: Dict):
        """Log a trade to the trade log file."""
        self._trade_fh.write(json.dumps(trade) + "\n")

    def _log_state(self, ts_ns: int):
        """Log current state of all strategies."""
        state = {
            "t": datetime.now(timezone.utc).isoformat(),
            "ts_ns": ts_ns,
            "events": self.events_processed,
            "preds": self.predictions_made,
            "mid": round(self.mid_price, 2),
            "strategies": {},
        }
        for strat in self.strategies:
            state["strategies"][strat.name] = {
                "pos": strat.pos.direction,
                "entry": round(strat.pos.entry_price, 2) if not strat.pos.is_flat else 0,
                "trades": strat.stats.total_trades,
                "pnl": round(strat.stats.net_pnl, 2),
                "wr": round(strat.stats.win_rate, 1),
                "halted": strat.halted,
            }
        self._state_fh.write(json.dumps(state) + "\n")

    def _dump_stats(self):
        """Dump periodic stats to log."""
        log.info("=" * 72)
        log.info("PERIODIC STATUS | events=%d preds=%d | mid=%.2f",
                 self.events_processed, self.predictions_made, self.mid_price)
        for strat in self.strategies:
            s = strat.stats
            pos_str = ("FLAT" if strat.pos.is_flat else
                       f"{'LONG' if strat.pos.direction > 0 else 'SHORT'} "
                       f"@ {strat.pos.entry_price:.2f}")
            log.info("  %-25s | %s | %d trades WR=%.0f%% P&L=$%.2f%s",
                     strat.name, pos_str, s.total_trades, s.win_rate, s.net_pnl,
                     " HALTED" if strat.halted else "")
        log.info("=" * 72)

    def _send_hourly_summary(self):
        """Send hourly summary to Discord."""
        lines = [f"**Hourly Summary** ({self.symbol})\n"]
        for strat in self.strategies:
            s = strat.stats
            pos_str = ("FLAT" if strat.pos.is_flat else
                       f"{'L' if strat.pos.direction > 0 else 'S'}")
            lines.append(
                f"  {strat.name}: {s.total_trades}T "
                f"WR={s.win_rate:.0f}% P&L=${s.net_pnl:.2f} [{pos_str}]"
            )
        discord_notify("\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────
    # Final report
    # ──────────────────────────────────────────────────────────────────────
    def final_report(self):
        """Generate and save final report."""
        log.info("=" * 72)
        log.info("FINAL REPORT — Monday Paper Runner")
        log.info("=" * 72)
        log.info("  Events: %d | Predictions: %d",
                 self.events_processed, self.predictions_made)

        report = {
            "session": _SESSION_ID,
            "symbol": self.symbol,
            "mode": "dry_run" if self.dry_run else ("backtest" if self.is_backtest else "live"),
            "events_processed": self.events_processed,
            "predictions_made": self.predictions_made,
            "strategies": {},
        }

        discord_lines = [f"**FINAL REPORT** ({self.symbol})\n"]

        for strat in self.strategies:
            s = strat.stats
            log.info("")
            log.info("  Strategy: %s", strat.name)
            log.info("    Trades: %d (W:%d L:%d) | Win Rate: %.1f%%",
                     s.total_trades, s.wins, s.losses, s.win_rate)
            log.info("    Gross P&L: $%.2f | Net P&L: $%.2f | Commission: $%.2f",
                     s.gross_pnl, s.net_pnl, s.commission)
            log.info("    Avg P&L: $%.2f | Sortino: %.2f | PF: %.2f",
                     s.avg_pnl, s.sortino, s.profit_factor)
            log.info("    Max DD: $%.2f | Halted: %s", s.max_drawdown, strat.halted)

            report["strategies"][strat.name] = {
                "total_trades": s.total_trades,
                "wins": s.wins,
                "losses": s.losses,
                "win_rate": round(s.win_rate, 1),
                "gross_pnl": round(s.gross_pnl, 2),
                "net_pnl": round(s.net_pnl, 2),
                "commission": round(s.commission, 2),
                "avg_pnl": round(s.avg_pnl, 2),
                "sortino": round(s.sortino, 2) if not math.isinf(s.sortino) else "inf",
                "profit_factor": round(s.profit_factor, 2) if not math.isinf(s.profit_factor) else "inf",
                "max_drawdown": round(s.max_drawdown, 2),
                "halted": strat.halted,
                "trades": s.trades,
            }

            emoji = "+" if s.net_pnl > 0 else ""
            discord_lines.append(
                f"  {strat.name}: {s.total_trades}T "
                f"WR={s.win_rate:.0f}% {emoji}${s.net_pnl:.2f} "
                f"Sortino={s.sortino:.1f}"
            )

        # Save report JSON
        report_path = _LOG_DIR / f"monday_report_{_SESSION_ID}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        log.info("")
        log.info("  Report: %s", report_path)
        log.info("  Signals: %s", self._signal_log_path)
        log.info("  Trades: %s", self._trade_log_path)
        log.info("  State: %s", self._state_log_path)
        log.info("=" * 72)

        discord_notify("\n".join(discord_lines))

        # Close file handles
        for fh in (self._signal_fh, self._trade_fh, self._state_fh):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

        return report

    # ──────────────────────────────────────────────────────────────────────
    # Close all open positions at end of day
    # ──────────────────────────────────────────────────────────────────────
    def close_all_positions(self, reason: str = "eod"):
        """Force-close all open positions."""
        wall_time = time.time()
        for strat in self.strategies:
            if not strat.pos.is_flat:
                trade = strat._exit_position(
                    self.mid_price, wall_time, 0, reason
                )
                if trade:
                    self._log_trade(trade)

    # ──────────────────────────────────────────────────────────────────────
    # BACKTEST MODE
    # ──────────────────────────────────────────────────────────────────────
    def run_backtest(self, npz_path: str, max_events: int = None):
        """Replay recorded MBO events through all 5 strategies."""
        data = np.load(npz_path)
        events = data["events"]
        timestamps = data.get("timestamps", np.zeros(len(events), dtype=np.int64))

        N = len(events) if max_events is None else min(len(events), max_events)
        log.info("BACKTEST: %s (%d events)", npz_path, N)

        t0 = time.time()
        for i in range(N):
            ev = events[i]
            ts = int(timestamps[i]) if i < len(timestamps) else 0

            # Update market state from event data (approximate mid from price_rel_ticks)
            # In backtest, we use relative ticks as a proxy
            if self.mid_price == 0:
                self.mid_price = 5000.0  # Default ES approximate price
                self.best_bid = self.mid_price - TICK_SIZE / 2
                self.best_ask = self.mid_price + TICK_SIZE / 2
            else:
                # Update mid from price_rel_ticks movement
                self.mid_price += float(ev[3]) * TICK_SIZE * 0.001  # Dampened random walk
                self.best_bid = self.mid_price - float(ev[5]) * TICK_SIZE / 2
                self.best_ask = self.mid_price + float(ev[5]) * TICK_SIZE / 2

            self.process_raw_event(
                time_delta_log=float(ev[0]),
                event_type_id=int(ev[1]),
                side_id=int(ev[2]),
                price_rel_ticks=float(ev[3]),
                qty_log=float(ev[4]),
                spread_ticks=float(ev[5]),
                timestamp_ns=ts,
            )

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                log.info("  [%d/%d] %.0f ev/s | preds=%d",
                         i + 1, N, rate, self.predictions_made)
                for strat in self.strategies:
                    if strat.stats.total_trades > 0:
                        log.info("    %s: %d trades P&L=$%.2f",
                                 strat.name, strat.stats.total_trades, strat.stats.net_pnl)

        # Close all open positions
        self.close_all_positions(reason="backtest_end")

        elapsed = time.time() - t0
        log.info("Backtest complete: %d events in %.1fs (%.0f ev/s)",
                 N, elapsed, N / elapsed)

        return self.final_report()

    # ──────────────────────────────────────────────────────────────────────
    # LIVE MODE
    # ──────────────────────────────────────────────────────────────────────
    async def run_live(self):
        """Connect to Rithmic for live paper trading."""
        from rithmic_client import RithmicClient, BBOEvent, TradeEvent
        from dotenv import load_dotenv
        load_dotenv(_SCRIPT_DIR / ".env")

        client = RithmicClient()

        async def on_md(ev):
            ts_ns = int(ev.ssboe * 1_000_000_000 + ev.usecs * 1000)
            if isinstance(ev, BBOEvent):
                if ev.has_bid and ev.bid_price > 0:
                    self.best_bid = ev.bid_price
                if ev.has_ask and ev.ask_price > 0:
                    self.best_ask = ev.ask_price
                if self.best_bid > 0 and self.best_ask > 0:
                    self.mid_price = (self.best_bid + self.best_ask) / 2.0
                if ev.has_bid:
                    self._encode_and_process(ts_ns, 0.0, 0.0, ev.bid_price, ev.bid_size)
                if ev.has_ask:
                    self._encode_and_process(ts_ns, 0.0, 1.0, ev.ask_price, ev.ask_size)
            elif isinstance(ev, TradeEvent):
                side = {1: 1.0, 2: 0.0}.get(ev.aggressor, 2.0)
                self._encode_and_process(ts_ns, 3.0, side, ev.trade_price, ev.trade_size)

        client.set_md_callback(on_md)

        log.info("Connecting to Rithmic (MARKET DATA ONLY)...")
        await client.connect()
        await client.subscribe_md(self.symbol, self.exchange)
        log.info("LIVE: Connected. Receiving %s on %s.", self.symbol, self.exchange)
        log.info("  Warming up features (need %d events)...", self.features.MIN_WARMUP)

        discord_notify(
            f"LIVE CONNECTED: {self.symbol} on {self.exchange}\n"
            f"Warming up ({self.features.MIN_WARMUP} events needed)"
        )

        # Background tasks
        tasks = []

        async def stats_loop():
            while not self._stop.is_set():
                await asyncio.sleep(300)  # 5 min
                self._dump_stats()

        async def kill_check_loop():
            while not self._stop.is_set():
                await asyncio.sleep(30)
                if self._check_kill_switch():
                    self._stop.set()

        async def eod_check_loop():
            """Close all positions at 16:00 ET."""
            while not self._stop.is_set():
                await asyncio.sleep(60)
                if time_in_range(16, 0, 16, 5):
                    log.info("EOD: Closing all positions")
                    self.close_all_positions(reason="eod")

        tasks.append(asyncio.create_task(stats_loop(), name="stats"))
        tasks.append(asyncio.create_task(kill_check_loop(), name="kill_check"))
        tasks.append(asyncio.create_task(eod_check_loop(), name="eod_check"))

        def handle_signal(*_):
            log.info("Shutdown signal received")
            self._stop.set()

        loop = asyncio.get_event_loop()
        for s in (_sig.SIGINT, _sig.SIGTERM):
            try:
                loop.add_signal_handler(s, handle_signal)
            except NotImplementedError:
                _sig.signal(s, lambda *_: self._stop.set())

        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            self.close_all_positions(reason="shutdown")
            await client.disconnect()
            self.final_report()


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monday Paper Runner — 5-strategy multi-execution paper trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Live trading:
    python3 monday_paper_runner.py --live

    # Backtest on recorded data:
    python3 monday_paper_runner.py --backtest data/processed/mbo_events/20260424_mbo_events.npz

    # Dry-run (inference only):
    python3 monday_paper_runner.py --backtest FILE --dry-run

    # With specific model weights:
    python3 monday_paper_runner.py --live --weights /path/to/model.pt --stats /path/to/stats.npz
        """
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true",
                      help="Connect to Rithmic for live paper trading")
    mode.add_argument("--backtest", type=str, metavar="NPZ_FILE",
                      help="Replay recorded MBO NPZ data")

    parser.add_argument("--dry-run", action="store_true",
                        help="Run inference only, no position tracking")
    parser.add_argument("--symbol", type=str, default="ESM6",
                        help="Trading symbol (default: ESM6)")
    parser.add_argument("--exchange", type=str, default="CME",
                        help="Exchange (default: CME)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Model weights path (auto-detect if not set)")
    parser.add_argument("--stats", type=str, default=None,
                        help="Feature stats path")
    parser.add_argument("--preds", type=str, default=None,
                        help="Historical predictions for calibration")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Max events to process (backtest only)")

    return parser.parse_args()


def main():
    args = parse_args()

    runner = MultiStrategyRunner(
        weights_path=args.weights,
        stats_path=args.stats,
        preds_path=args.preds,
        symbol=args.symbol,
        exchange=args.exchange,
        dry_run=args.dry_run,
        backtest_mode=bool(args.backtest),
    )

    if args.backtest:
        npz_path = args.backtest
        if not Path(npz_path).exists():
            # Try relative to MBO data dir
            alt = MBO_DATA_DIR / npz_path
            if alt.exists():
                npz_path = str(alt)
            else:
                log.error("File not found: %s", npz_path)
                sys.exit(1)

        runner.run_backtest(npz_path, max_events=args.max_events)

    elif args.live:
        asyncio.run(runner.run_live())


if __name__ == "__main__":
    main()
