#!/usr/bin/env python3
"""
trade_journal.py — Unified Performance Tracking for ES Futures Paper Trading
=============================================================================

Provides the TradeJournal class imported by paper_trading_mamba_v2.py and
rl_execution_agent.py on Razer.  One instance per config tracks every trade
with full market-microstructure detail and computes rolling risk-adjusted
metrics after each close.

Config names: "sac_v7", "ppo_v7", "rules_top01pct", "rules_top1pct"

Output (daily-rotated):
    {config}_trades_{YYYYMMDD}.jsonl   — one JSON object per completed trade
    {config}_metrics_{YYYYMMDD}.json   — rolling metrics, overwritten each trade
    {config}_summary_{YYYYMMDD}.json   — periodic snapshot (every N trades / M min)

Thread-safe: all mutations guarded by a threading.Lock.

Dependencies: stdlib only (json, math, statistics, datetime, threading, os, time).

Author: Claude (Infrastructure Builder)
Date:   2026-05-03
"""

from __future__ import annotations

import json
import math
import os
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── ES Futures Constants (AMP / Rithmic) ─────────────────────────────────────

TICK_VALUE: float = 12.50          # 1 tick = $12.50
TICK_SIZE: float = 0.25            # 1 tick = 0.25 index points
COMMISSION_PER_SIDE: float = 2.35  # AMP $4.70 RT / 2
COMMISSION_RT: float = 4.70        # Round-trip commission

# Annualisation: 252 trading days.  Trades-per-day is estimated from the data.
TRADING_DAYS_PER_YEAR: int = 252


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class PendingEntry:
    """State held between record_entry() and record_exit()."""
    entry_price: float = 0.0
    entry_timestamp_utc: str = ""
    entry_timestamp_ns: int = 0
    direction: str = ""              # "LONG" or "SHORT"
    signal_confidence: float = 0.0
    signal_tier: str = ""
    entry_zscore: float = 0.0
    pred_1s: float = 0.0
    pred_5s: float = 0.0
    pred_10s: float = 0.0
    entry_action_type: str = ""      # "limit" or "market"
    spread_at_entry: float = 0.0
    bid_at_entry: float = 0.0
    ask_at_entry: float = 0.0
    signal_timestamp_ns: int = 0     # when the signal fired (for time_to_fill)
    config_name: str = ""


@dataclass
class TradeRecord:
    """Complete record for one round-trip trade, serialised to JSONL."""
    trade_number: int = 0
    config_name: str = ""

    # Prices
    entry_price: float = 0.0
    exit_price: float = 0.0

    # Timestamps
    entry_timestamp_utc: str = ""
    exit_timestamp_utc: str = ""
    entry_timestamp_ns: int = 0
    exit_timestamp_ns: int = 0

    # Direction & hold
    direction: str = ""
    hold_time_s: float = 0.0

    # Signal features at entry
    signal_confidence: float = 0.0
    signal_tier: str = ""
    entry_zscore: float = 0.0
    pred_1s: float = 0.0
    pred_5s: float = 0.0
    pred_10s: float = 0.0

    # P&L
    gross_pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    pnl_ticks: float = 0.0
    commission_usd: float = COMMISSION_RT

    # Excursions
    mfe_ticks: float = 0.0
    mae_ticks: float = 0.0

    # Exit
    exit_reason: str = ""

    # Microstructure
    entry_action_type: str = ""
    spread_at_entry: float = 0.0
    spread_at_exit: float = 0.0
    bid_at_entry: float = 0.0
    ask_at_entry: float = 0.0
    bid_at_exit: float = 0.0
    ask_at_exit: float = 0.0
    time_to_fill_ms: float = 0.0
    slippage_ticks: float = 0.0


# ── Metrics Container ────────────────────────────────────────────────────────

def _empty_metrics() -> Dict[str, Any]:
    """Return a fresh metrics dict with all fields zeroed."""
    return {
        "cumulative_pnl_usd": 0.0,
        "cumulative_pnl_ticks": 0.0,
        "n_trades": 0,
        "n_winners": 0,
        "n_losers": 0,
        "win_rate_pct": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "profit_factor": 0.0,
        "avg_win_usd": 0.0,
        "avg_loss_usd": 0.0,
        "avg_pnl_usd": 0.0,
        "avg_hold_time_s": 0.0,
        "median_hold_time_s": 0.0,
        "max_hold_time_s": 0.0,
        "min_hold_time_s": 0.0,
        "avg_mfe_ticks": 0.0,
        "avg_mae_ticks": 0.0,
        "max_drawdown_usd": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_time_to_fill_ms": 0.0,
        "avg_slippage_ticks": 0.0,
        "avg_spread_at_entry": 0.0,
        "best_trade_usd": 0.0,
        "worst_trade_usd": 0.0,
        "longest_win_streak": 0,
        "longest_loss_streak": 0,
        "current_streak": 0,
        "current_streak_type": "",  # "W" or "L" or ""
        "trades_per_hour": 0.0,
        "long_count": 0,
        "short_count": 0,
        "long_win_rate": 0.0,
        "short_win_rate": 0.0,
        "pnl_by_hour": {},
        "exit_reason_breakdown": {},
        "updated_utc": "",
    }


# ── TradeJournal ─────────────────────────────────────────────────────────────

class TradeJournal:
    """
    Thread-safe performance tracker for a single paper-trading config.

    Usage::

        journal = TradeJournal("sac_v7", log_dir="logs/")

        # When a fill comes in:
        journal.record_entry(
            entry_price=5432.25,
            direction="SHORT",
            entry_timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
            entry_timestamp_ns=market_ts_ns,
            signal_confidence=0.87,
            signal_tier="top_1pct",
            ...
        )

        # When the position closes:
        journal.record_exit(
            exit_price=5431.50,
            exit_timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
            exit_timestamp_ns=market_ts_ns,
            exit_reason="signal_decay",
            mfe_ticks=4.0,
            mae_ticks=1.0,
            ...
        )

        metrics = journal.get_metrics()
        print(journal.get_summary_str())
    """

    def __init__(
        self,
        config_name: str,
        log_dir: str = "logs/",
        summary_interval_trades: int = 25,
        summary_interval_seconds: float = 900.0,  # 15 min
    ) -> None:
        if config_name not in ("sac_v7", "ppo_v7", "rules_top01pct", "rules_top1pct"):
            # Allow arbitrary names but warn — don't hard-crash.
            pass

        self.config_name = config_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.summary_interval_trades = summary_interval_trades
        self.summary_interval_seconds = summary_interval_seconds

        self._lock = threading.Lock()

        # Mutable state
        self._trades: List[TradeRecord] = []
        self._pending: Optional[PendingEntry] = None
        self._trade_counter: int = 0

        # For rolling metric computation
        self._net_pnls: List[float] = []       # net_pnl_usd per trade
        self._gross_wins: float = 0.0
        self._gross_losses: float = 0.0
        self._equity_curve: List[float] = [0.0]  # cumulative P&L points
        self._peak_equity: float = 0.0
        self._max_dd_usd: float = 0.0

        # Streak tracking
        self._current_streak: int = 0
        self._current_streak_type: str = ""   # "W" or "L"
        self._longest_win_streak: int = 0
        self._longest_loss_streak: int = 0

        # Directional stats
        self._long_count: int = 0
        self._long_wins: int = 0
        self._short_count: int = 0
        self._short_wins: int = 0

        # By-hour PnL
        self._pnl_by_hour: Dict[int, float] = defaultdict(float)

        # Exit reason counts
        self._exit_reasons: Dict[str, int] = defaultdict(int)

        # Time tracking
        self._first_trade_time: Optional[float] = None
        self._last_summary_time: float = time.time()
        self._last_summary_trade_count: int = 0

        # Current trading date for file rotation
        self._current_date: str = _today_str()

        # Open file handles (lazy)
        self._jsonl_fh: Optional[Any] = None
        self._jsonl_path: Optional[Path] = None

    # ── File management ──────────────────────────────────────────────────────

    def _ensure_files(self) -> None:
        """Open/rotate JSONL file handle if the date changed."""
        today = _today_str()
        if today != self._current_date or self._jsonl_fh is None:
            self._rotate(today)

    def _rotate(self, new_date: str) -> None:
        """Close old handle, open new one for the given date."""
        if self._jsonl_fh is not None:
            try:
                self._jsonl_fh.close()
            except Exception:
                pass
        self._current_date = new_date
        self._jsonl_path = self.log_dir / f"{self.config_name}_trades_{new_date}.jsonl"
        self._jsonl_fh = open(self._jsonl_path, "a", encoding="utf-8")

    def _write_trade(self, rec: TradeRecord) -> None:
        """Append one trade to the JSONL file."""
        self._ensure_files()
        line = json.dumps(asdict(rec), default=str)
        self._jsonl_fh.write(line + "\n")
        self._jsonl_fh.flush()

    def _write_metrics(self, metrics: Dict[str, Any]) -> None:
        """Overwrite the metrics JSON for today."""
        path = self.log_dir / f"{self.config_name}_metrics_{self._current_date}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        os.replace(tmp, path)

    def _write_summary(self, metrics: Dict[str, Any]) -> None:
        """Write periodic summary snapshot."""
        path = self.log_dir / f"{self.config_name}_summary_{self._current_date}.json"
        snapshot = {
            "snapshot_utc": datetime.now(tz=timezone.utc).isoformat(),
            "config_name": self.config_name,
            **metrics,
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, default=str)
        os.replace(tmp, path)

    # ── Public API ───────────────────────────────────────────────────────────

    def record_entry(
        self,
        entry_price: float,
        direction: str,
        entry_timestamp_utc: str = "",
        entry_timestamp_ns: int = 0,
        signal_confidence: float = 0.0,
        signal_tier: str = "",
        entry_zscore: float = 0.0,
        pred_1s: float = 0.0,
        pred_5s: float = 0.0,
        pred_10s: float = 0.0,
        entry_action_type: str = "",
        spread_at_entry: float = 0.0,
        bid_at_entry: float = 0.0,
        ask_at_entry: float = 0.0,
        signal_timestamp_ns: int = 0,
    ) -> None:
        """
        Record a new position entry.  Stores pending state until record_exit()
        is called.  If there is already a pending entry (no exit recorded),
        it is silently overwritten — the caller is responsible for ensuring
        proper entry/exit pairing.
        """
        direction = direction.upper()
        if direction not in ("LONG", "SHORT"):
            raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")

        if not entry_timestamp_utc:
            entry_timestamp_utc = datetime.now(tz=timezone.utc).isoformat()

        with self._lock:
            self._pending = PendingEntry(
                entry_price=entry_price,
                entry_timestamp_utc=entry_timestamp_utc,
                entry_timestamp_ns=entry_timestamp_ns,
                direction=direction,
                signal_confidence=signal_confidence,
                signal_tier=signal_tier,
                entry_zscore=entry_zscore,
                pred_1s=pred_1s,
                pred_5s=pred_5s,
                pred_10s=pred_10s,
                entry_action_type=entry_action_type,
                spread_at_entry=spread_at_entry,
                bid_at_entry=bid_at_entry,
                ask_at_entry=ask_at_entry,
                signal_timestamp_ns=signal_timestamp_ns,
                config_name=self.config_name,
            )

    def record_exit(
        self,
        exit_price: float,
        exit_timestamp_utc: str = "",
        exit_timestamp_ns: int = 0,
        exit_reason: str = "",
        mfe_ticks: float = 0.0,
        mae_ticks: float = 0.0,
        spread_at_exit: float = 0.0,
        bid_at_exit: float = 0.0,
        ask_at_exit: float = 0.0,
        time_to_fill_ms: float = 0.0,
        slippage_ticks: float = 0.0,
    ) -> Optional[TradeRecord]:
        """
        Complete a pending trade, compute P&L, write to JSONL, update metrics.

        Returns the completed TradeRecord, or None if there was no pending entry.
        """
        if not exit_timestamp_utc:
            exit_timestamp_utc = datetime.now(tz=timezone.utc).isoformat()

        with self._lock:
            if self._pending is None:
                return None

            p = self._pending
            self._pending = None
            self._trade_counter += 1

            # ── P&L computation ──────────────────────────────────────────
            if p.direction == "LONG":
                price_diff = exit_price - p.entry_price
            else:
                price_diff = p.entry_price - exit_price

            pnl_ticks = price_diff / TICK_SIZE
            gross_pnl_usd = pnl_ticks * TICK_VALUE
            net_pnl_usd = gross_pnl_usd - COMMISSION_RT

            # ── Hold time ────────────────────────────────────────────────
            hold_time_s = 0.0
            if p.entry_timestamp_ns and exit_timestamp_ns:
                hold_time_s = (exit_timestamp_ns - p.entry_timestamp_ns) / 1e9
            else:
                # Fallback: parse UTC strings
                try:
                    t_entry = datetime.fromisoformat(p.entry_timestamp_utc)
                    t_exit = datetime.fromisoformat(exit_timestamp_utc)
                    hold_time_s = (t_exit - t_entry).total_seconds()
                except Exception:
                    pass
            if hold_time_s < 0:
                hold_time_s = 0.0

            # ── Build record ─────────────────────────────────────────────
            rec = TradeRecord(
                trade_number=self._trade_counter,
                config_name=self.config_name,
                entry_price=p.entry_price,
                exit_price=exit_price,
                entry_timestamp_utc=p.entry_timestamp_utc,
                exit_timestamp_utc=exit_timestamp_utc,
                entry_timestamp_ns=p.entry_timestamp_ns,
                exit_timestamp_ns=exit_timestamp_ns,
                direction=p.direction,
                hold_time_s=round(hold_time_s, 4),
                signal_confidence=p.signal_confidence,
                signal_tier=p.signal_tier,
                entry_zscore=p.entry_zscore,
                pred_1s=p.pred_1s,
                pred_5s=p.pred_5s,
                pred_10s=p.pred_10s,
                gross_pnl_usd=round(gross_pnl_usd, 2),
                net_pnl_usd=round(net_pnl_usd, 2),
                pnl_ticks=round(pnl_ticks, 4),
                commission_usd=COMMISSION_RT,
                mfe_ticks=mfe_ticks,
                mae_ticks=mae_ticks,
                exit_reason=exit_reason,
                entry_action_type=p.entry_action_type,
                spread_at_entry=p.spread_at_entry,
                spread_at_exit=spread_at_exit,
                bid_at_entry=p.bid_at_entry,
                ask_at_entry=p.ask_at_entry,
                bid_at_exit=bid_at_exit,
                ask_at_exit=ask_at_exit,
                time_to_fill_ms=time_to_fill_ms,
                slippage_ticks=slippage_ticks,
            )

            # ── Update internal state ────────────────────────────────────
            self._trades.append(rec)
            self._net_pnls.append(net_pnl_usd)

            cum_pnl = self._equity_curve[-1] + net_pnl_usd
            self._equity_curve.append(cum_pnl)

            if cum_pnl > self._peak_equity:
                self._peak_equity = cum_pnl
            dd = self._peak_equity - cum_pnl
            if dd > self._max_dd_usd:
                self._max_dd_usd = dd

            is_winner = net_pnl_usd > 0

            if is_winner:
                self._gross_wins += net_pnl_usd
            else:
                self._gross_losses += abs(net_pnl_usd)

            # Streaks
            if is_winner:
                if self._current_streak_type == "W":
                    self._current_streak += 1
                else:
                    self._current_streak_type = "W"
                    self._current_streak = 1
                if self._current_streak > self._longest_win_streak:
                    self._longest_win_streak = self._current_streak
            else:
                if self._current_streak_type == "L":
                    self._current_streak += 1
                else:
                    self._current_streak_type = "L"
                    self._current_streak = 1
                if self._current_streak > self._longest_loss_streak:
                    self._longest_loss_streak = self._current_streak

            # Directional
            if p.direction == "LONG":
                self._long_count += 1
                if is_winner:
                    self._long_wins += 1
            else:
                self._short_count += 1
                if is_winner:
                    self._short_wins += 1

            # By hour
            try:
                hour = datetime.fromisoformat(exit_timestamp_utc).hour
            except Exception:
                hour = datetime.now(tz=timezone.utc).hour
            self._pnl_by_hour[hour] += net_pnl_usd

            # Exit reason
            self._exit_reasons[exit_reason] += 1

            # First trade time
            if self._first_trade_time is None:
                self._first_trade_time = time.time()

            # ── Write files ──────────────────────────────────────────────
            self._write_trade(rec)
            metrics = self._compute_metrics_unlocked()
            self._write_metrics(metrics)

            # Periodic summary
            now = time.time()
            trades_since = self._trade_counter - self._last_summary_trade_count
            time_since = now - self._last_summary_time
            if (trades_since >= self.summary_interval_trades
                    or time_since >= self.summary_interval_seconds):
                self._write_summary(metrics)
                self._last_summary_time = now
                self._last_summary_trade_count = self._trade_counter

            return rec

    def has_pending_entry(self) -> bool:
        """Check if there is an open position awaiting exit."""
        with self._lock:
            return self._pending is not None

    def get_pending_direction(self) -> Optional[str]:
        """Return the direction of the pending entry, or None."""
        with self._lock:
            return self._pending.direction if self._pending else None

    def get_metrics(self) -> Dict[str, Any]:
        """Return current rolling metrics dict (thread-safe copy)."""
        with self._lock:
            return self._compute_metrics_unlocked()

    def get_trade_count(self) -> int:
        """Return number of completed trades."""
        with self._lock:
            return self._trade_counter

    def get_cumulative_pnl(self) -> float:
        """Return cumulative net P&L in USD."""
        with self._lock:
            return self._equity_curve[-1] if self._equity_curve else 0.0

    def get_summary_str(self) -> str:
        """
        Return a formatted multi-line summary string suitable for logging
        or sending to Discord/Telegram.
        """
        m = self.get_metrics()
        n = m["n_trades"]
        if n == 0:
            return f"[{self.config_name}] No trades recorded yet."

        streak_char = m.get("current_streak_type", "")
        streak_str = f"{m['current_streak']}{streak_char}" if streak_char else "0"

        lines = [
            f"=== {self.config_name} Trade Journal ===",
            f"  Trades: {n}  (L:{m['long_count']} S:{m['short_count']})",
            f"  Win Rate: {m['win_rate_pct']:.1f}%  "
            f"(L:{m['long_win_rate']:.1f}% S:{m['short_win_rate']:.1f}%)",
            f"  Cum P&L: ${m['cumulative_pnl_usd']:+.2f}  "
            f"({m['cumulative_pnl_ticks']:+.1f} ticks)",
            f"  Sharpe: {m['sharpe_ratio']:.2f}  "
            f"Sortino: {m['sortino_ratio']:.2f}  "
            f"PF: {m['profit_factor']:.2f}",
            f"  Avg Win: ${m['avg_win_usd']:.2f}  "
            f"Avg Loss: ${m['avg_loss_usd']:.2f}  "
            f"Avg: ${m['avg_pnl_usd']:.2f}",
            f"  Best: ${m['best_trade_usd']:+.2f}  "
            f"Worst: ${m['worst_trade_usd']:+.2f}",
            f"  Max DD: ${m['max_drawdown_usd']:.2f}  "
            f"({m['max_drawdown_pct']:.1f}%)",
            f"  Hold: avg={m['avg_hold_time_s']:.1f}s  "
            f"med={m['median_hold_time_s']:.1f}s  "
            f"max={m['max_hold_time_s']:.1f}s",
            f"  MFE: {m['avg_mfe_ticks']:.2f}t  "
            f"MAE: {m['avg_mae_ticks']:.2f}t  "
            f"Slip: {m['avg_slippage_ticks']:.3f}t",
            f"  Fill: {m['avg_time_to_fill_ms']:.1f}ms  "
            f"Spread@Entry: {m['avg_spread_at_entry']:.3f}",
            f"  Streak: {streak_str}  "
            f"(best W:{m['longest_win_streak']} L:{m['longest_loss_streak']})",
            f"  Rate: {m['trades_per_hour']:.1f} trades/hr",
        ]

        # Top exit reasons
        er = m.get("exit_reason_breakdown", {})
        if er:
            top = sorted(er.items(), key=lambda x: x[1], reverse=True)[:5]
            reasons_str = ", ".join(f"{k}:{v}" for k, v in top)
            lines.append(f"  Exits: {reasons_str}")

        # Hourly P&L (non-zero hours only)
        pnl_h = m.get("pnl_by_hour", {})
        if pnl_h:
            hours_sorted = sorted(pnl_h.items(), key=lambda x: int(x[0]))
            parts = [f"{h}h:${v:+.0f}" for h, v in hours_sorted if abs(v) > 0.01]
            if parts:
                lines.append(f"  Hourly: {' '.join(parts)}")

        return "\n".join(lines)

    def flush(self) -> None:
        """Force-flush the JSONL file handle and write current metrics/summary."""
        with self._lock:
            if self._jsonl_fh is not None:
                self._jsonl_fh.flush()
            if self._trade_counter > 0:
                metrics = self._compute_metrics_unlocked()
                self._write_metrics(metrics)
                self._write_summary(metrics)

    def close(self) -> None:
        """Flush and close file handles.  Safe to call multiple times."""
        with self._lock:
            if self._jsonl_fh is not None:
                try:
                    self._jsonl_fh.flush()
                    self._jsonl_fh.close()
                except Exception:
                    pass
                self._jsonl_fh = None

    # ── Metrics computation (caller must hold self._lock) ────────────────────

    def _compute_metrics_unlocked(self) -> Dict[str, Any]:
        m = _empty_metrics()
        n = len(self._net_pnls)
        m["n_trades"] = n
        m["updated_utc"] = datetime.now(tz=timezone.utc).isoformat()

        if n == 0:
            return m

        pnls = self._net_pnls

        # Cumulative
        m["cumulative_pnl_usd"] = round(self._equity_curve[-1], 2)
        m["cumulative_pnl_ticks"] = round(sum(
            t.pnl_ticks for t in self._trades
        ), 4)

        # Win/loss counts
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        m["n_winners"] = len(winners)
        m["n_losers"] = len(losers)
        m["win_rate_pct"] = round(100.0 * len(winners) / n, 2)

        # Averages
        m["avg_win_usd"] = round(statistics.mean(winners), 2) if winners else 0.0
        m["avg_loss_usd"] = round(statistics.mean(losers), 2) if losers else 0.0
        m["avg_pnl_usd"] = round(statistics.mean(pnls), 2)

        # Best / worst
        m["best_trade_usd"] = round(max(pnls), 2)
        m["worst_trade_usd"] = round(min(pnls), 2)

        # Profit factor
        m["profit_factor"] = round(
            self._gross_wins / self._gross_losses, 4
        ) if self._gross_losses > 0 else (
            float("inf") if self._gross_wins > 0 else 0.0
        )

        # Sharpe ratio (annualised)
        if n >= 2:
            mean_ret = statistics.mean(pnls)
            std_ret = statistics.stdev(pnls)
            if std_ret > 1e-9:
                trades_per_day = self._estimate_trades_per_day()
                ann_factor = math.sqrt(TRADING_DAYS_PER_YEAR * trades_per_day)
                m["sharpe_ratio"] = round(
                    (mean_ret / std_ret) * ann_factor, 4
                )

        # Sortino ratio (annualised, downside deviation)
        if n >= 2:
            mean_ret = statistics.mean(pnls)
            downside = [p for p in pnls if p < 0]
            if len(downside) >= 1:
                downside_sq = [d * d for d in downside]
                downside_dev = math.sqrt(sum(downside_sq) / n)  # full-sample denominator
                if downside_dev > 1e-9:
                    trades_per_day = self._estimate_trades_per_day()
                    ann_factor = math.sqrt(TRADING_DAYS_PER_YEAR * trades_per_day)
                    m["sortino_ratio"] = round(
                        (mean_ret / downside_dev) * ann_factor, 4
                    )

        # Hold times
        hold_times = [t.hold_time_s for t in self._trades]
        m["avg_hold_time_s"] = round(statistics.mean(hold_times), 2)
        m["median_hold_time_s"] = round(statistics.median(hold_times), 2)
        m["max_hold_time_s"] = round(max(hold_times), 2)
        m["min_hold_time_s"] = round(min(hold_times), 2)

        # Excursions
        m["avg_mfe_ticks"] = round(
            statistics.mean(t.mfe_ticks for t in self._trades), 4
        )
        m["avg_mae_ticks"] = round(
            statistics.mean(t.mae_ticks for t in self._trades), 4
        )

        # Drawdown
        m["max_drawdown_usd"] = round(self._max_dd_usd, 2)
        peak = self._peak_equity
        m["max_drawdown_pct"] = round(
            100.0 * self._max_dd_usd / peak, 2
        ) if peak > 0 else 0.0

        # Fill timing & slippage
        m["avg_time_to_fill_ms"] = round(
            statistics.mean(t.time_to_fill_ms for t in self._trades), 2
        )
        m["avg_slippage_ticks"] = round(
            statistics.mean(t.slippage_ticks for t in self._trades), 4
        )
        m["avg_spread_at_entry"] = round(
            statistics.mean(t.spread_at_entry for t in self._trades), 4
        )

        # Streaks
        m["longest_win_streak"] = self._longest_win_streak
        m["longest_loss_streak"] = self._longest_loss_streak
        m["current_streak"] = self._current_streak
        m["current_streak_type"] = self._current_streak_type

        # Trades per hour
        m["trades_per_hour"] = round(self._compute_trades_per_hour(), 2)

        # Directional
        m["long_count"] = self._long_count
        m["short_count"] = self._short_count
        m["long_win_rate"] = round(
            100.0 * self._long_wins / self._long_count, 2
        ) if self._long_count > 0 else 0.0
        m["short_win_rate"] = round(
            100.0 * self._short_wins / self._short_count, 2
        ) if self._short_count > 0 else 0.0

        # By-hour PnL  (str keys for JSON)
        m["pnl_by_hour"] = {
            str(k): round(v, 2) for k, v in sorted(self._pnl_by_hour.items())
        }

        # Exit reason breakdown
        m["exit_reason_breakdown"] = dict(self._exit_reasons)

        return m

    def _estimate_trades_per_day(self) -> float:
        """Estimate average trades per trading day from observed rate."""
        if self._first_trade_time is None or len(self._net_pnls) < 2:
            return max(len(self._net_pnls), 1)
        elapsed_s = time.time() - self._first_trade_time
        if elapsed_s < 60:
            # Not enough time to estimate; assume all trades in one day
            return max(len(self._net_pnls), 1)
        elapsed_days = elapsed_s / 86400.0
        if elapsed_days < 0.01:
            return max(len(self._net_pnls), 1)
        return len(self._net_pnls) / elapsed_days

    def _compute_trades_per_hour(self) -> float:
        """Rolling trades-per-hour from first trade to now."""
        if self._first_trade_time is None:
            return 0.0
        elapsed_h = (time.time() - self._first_trade_time) / 3600.0
        if elapsed_h < 0.001:
            return 0.0
        return len(self._net_pnls) / elapsed_h

    # ── Context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "TradeJournal":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"TradeJournal(config={self.config_name!r}, "
            f"trades={self._trade_counter}, "
            f"pnl=${self._equity_curve[-1]:+.2f})"
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _today_str() -> str:
    """Return today's date as YYYYMMDD in UTC."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d")


# ── Quick self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        j = TradeJournal("sac_v7", log_dir=tmpdir, summary_interval_trades=2)

        # Simulate 5 trades
        trades_data = [
            ("LONG",  5430.00, 5431.00, "signal_decay",  4.0, 1.0),
            ("SHORT", 5432.00, 5431.50, "signal_decay",  3.0, 0.5),
            ("SHORT", 5433.00, 5434.00, "stop_loss",     1.0, 5.0),
            ("LONG",  5430.25, 5430.75, "target_hit",    3.0, 0.5),
            ("SHORT", 5435.00, 5434.00, "signal_decay",  5.0, 1.0),
        ]

        for i, (d, ep, xp, reason, mfe, mae) in enumerate(trades_data):
            j.record_entry(
                entry_price=ep,
                direction=d,
                signal_confidence=0.85 + i * 0.01,
                signal_tier="top_1pct",
                pred_1s=0.15,
                pred_5s=0.10,
                pred_10s=0.07,
                entry_action_type="limit",
                spread_at_entry=0.25,
                bid_at_entry=ep - 0.25,
                ask_at_entry=ep,
            )
            j.record_exit(
                exit_price=xp,
                exit_reason=reason,
                mfe_ticks=mfe,
                mae_ticks=mae,
                spread_at_exit=0.25,
                bid_at_exit=xp - 0.25,
                ask_at_exit=xp,
                time_to_fill_ms=45.0 + i * 10,
                slippage_ticks=0.1 * i,
            )

        print(j.get_summary_str())
        print()
        print(f"Files in {tmpdir}:")
        for f in sorted(os.listdir(tmpdir)):
            size = os.path.getsize(os.path.join(tmpdir, f))
            print(f"  {f}  ({size} bytes)")
        print()
        print("Metrics JSON keys:", sorted(j.get_metrics().keys()))
        print()

        # Verify JSONL readback
        jsonl_files = [f for f in os.listdir(tmpdir) if f.endswith(".jsonl")]
        assert len(jsonl_files) == 1, f"Expected 1 JSONL file, got {jsonl_files}"
        with open(os.path.join(tmpdir, jsonl_files[0])) as fh:
            lines = fh.readlines()
        assert len(lines) == 5, f"Expected 5 trade lines, got {len(lines)}"
        first = json.loads(lines[0])
        assert first["trade_number"] == 1
        assert first["direction"] == "LONG"
        assert first["config_name"] == "sac_v7"
        print("All assertions passed.")

        j.close()
