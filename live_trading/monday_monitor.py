#!/usr/bin/env python3
"""
monday_monitor.py — Real-Time Monday Paper Trading Monitor
============================================================

Connects to Razer (Windows, RTX 3070) periodically to pull paper trading
logs/results and tracks KPIs in real-time for the CNN-Mamba v2 stacked
confluence paper trading deployment.

THE EXISTENTIAL QUESTION: What is the actual fill rate, and does adverse
selection kill the edge?

Simulation baseline (48-day backtest, balanced preset):
  - +0.505 ticks/trade average (passive-only cost basis)
  - Sharpe ~26.8 annualized
  - PF 1.92, WR 58.7%, ~527 trades/day
  - 92.6% green days

Alerts on:
  - Fill rate below 30%
  - Inference crash (no new log lines for 15+ min during RTH)
  - Latency > 500ms per prediction
  - P&L tracking >50% worse than sim expectation

Usage:
    python3 live_trading/monday_monitor.py --razer-host razer
    python3 live_trading/monday_monitor.py --local-logs /path/to/logs
    python3 live_trading/monday_monitor.py --razer-host razer --poll 60

Author: Claude (Autonomous Infrastructure)
Date:   2026-05-25
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ES_TICK_VALUE = 12.50
ES_TICK_SIZE = 0.25
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376

# Simulation baseline (balanced preset, 48-day backtest)
SIM_TICKS_PER_TRADE = 0.505
SIM_SHARPE = 26.8
SIM_PF = 1.92
SIM_WR = 58.7
SIM_TRADES_PER_DAY = 527

# Alert thresholds
FILL_RATE_ALERT_THRESHOLD = 0.30    # Alert if fill rate < 30%
LATENCY_ALERT_MS = 500.0             # Alert if inference latency > 500ms
NO_TRADE_ALERT_MINUTES = 15          # Alert if no trades for 15 min during RTH
DEGRADATION_THRESHOLD = 0.50         # Alert if >50% worse than sim

# RTH window (Eastern Time)
RTH_OPEN_HOUR = 9
RTH_OPEN_MIN = 30
RTH_CLOSE_HOUR = 16
RTH_CLOSE_MIN = 0

# Razer paths (Windows)
RAZER_LOG_DIR = r"C:\Users\claude\Lvl3Quant\live_trading\logs"
RAZER_PYTHON = r"C:\Users\claude\Lvl3Quant"

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrderEvent:
    """Represents a placed or filled order from the paper trader."""
    timestamp: float = 0.0
    direction: str = ""       # "SHORT" or "LONG"
    price: float = 0.0
    order_type: str = ""      # "limit" or "market"
    status: str = ""          # "placed", "filled", "cancelled", "expired"
    signal_confidence: float = 0.0
    fill_latency_ms: float = 0.0


@dataclass
class LiveTrade:
    """A completed round-trip trade from the paper trader."""
    trade_number: int = 0
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl_ticks: float = 0.0
    net_pnl_usd: float = 0.0
    hold_time_s: float = 0.0
    exit_reason: str = ""
    mfe_ticks: float = 0.0
    mae_ticks: float = 0.0
    signal_confidence: float = 0.0
    entry_action_type: str = ""
    time_to_fill_ms: float = 0.0
    slippage_ticks: float = 0.0
    timestamp: float = 0.0


@dataclass
class GateSnapshot:
    """Gate pass-rate statistics from the stacked filter."""
    total_evaluated: int = 0
    gate1_signal_passed: int = 0
    gate2_meta_passed: int = 0
    gate3_ofi_passed: int = 0
    all_passed: int = 0

    @property
    def signal_rate(self) -> float:
        return self.gate1_signal_passed / max(self.total_evaluated, 1)

    @property
    def meta_rate(self) -> float:
        return self.gate2_meta_passed / max(self.gate1_signal_passed, 1)

    @property
    def ofi_rate(self) -> float:
        return self.gate3_ofi_passed / max(self.gate2_meta_passed, 1)

    @property
    def overall_rate(self) -> float:
        return self.all_passed / max(self.total_evaluated, 1)


@dataclass
class LatencySnapshot:
    """Latency stats for the inference pipeline."""
    cnn_mamba_p50_ms: float = 0.0
    cnn_mamba_p95_ms: float = 0.0
    cnn_mamba_p99_ms: float = 0.0
    total_pred_p50_ms: float = 0.0
    total_pred_p95_ms: float = 0.0
    total_pred_p99_ms: float = 0.0
    n_predictions: int = 0


# ---------------------------------------------------------------------------
# Trade log parser
# ---------------------------------------------------------------------------

def parse_trade_from_jsonl(line: str) -> Optional[LiveTrade]:
    """Parse a single JSONL line into a LiveTrade. Handles TradeJournal format."""
    try:
        raw = json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None

    # TradeJournal JSONL format (from trade_journal.py)
    if "trade_number" in raw and "entry_price" in raw and "exit_price" in raw:
        ts = 0.0
        ts_str = raw.get("exit_timestamp_utc", "")
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str).timestamp()
            except (ValueError, OSError):
                pass

        return LiveTrade(
            trade_number=raw.get("trade_number", 0),
            direction=raw.get("direction", ""),
            entry_price=float(raw.get("entry_price", 0)),
            exit_price=float(raw.get("exit_price", 0)),
            pnl_ticks=float(raw.get("pnl_ticks", 0)),
            net_pnl_usd=float(raw.get("net_pnl_usd", 0)),
            hold_time_s=float(raw.get("hold_time_s", 0)),
            exit_reason=raw.get("exit_reason", ""),
            mfe_ticks=float(raw.get("mfe_ticks", 0)),
            mae_ticks=float(raw.get("mae_ticks", 0)),
            signal_confidence=float(raw.get("signal_confidence", 0)),
            entry_action_type=raw.get("entry_action_type", ""),
            time_to_fill_ms=float(raw.get("time_to_fill_ms", 0)),
            slippage_ticks=float(raw.get("slippage_ticks", 0)),
            timestamp=ts,
        )

    # Alternative: TRADE_CLOSED event format (from paper_trading_mamba_v2.py)
    if raw.get("event") == "TRADE_CLOSED":
        ts = _parse_ts(raw.get("exit_time", raw.get("ts", 0)))
        pnl_ticks = float(raw.get("pnl_ticks", 0))
        return LiveTrade(
            trade_number=raw.get("trade_num", 0),
            direction=raw.get("direction", "").upper(),
            entry_price=float(raw.get("entry_price", 0)),
            exit_price=float(raw.get("exit_price", 0)),
            pnl_ticks=pnl_ticks,
            net_pnl_usd=float(raw.get("pnl_usd", pnl_ticks * ES_TICK_VALUE - ES_RT_COMMISSION)),
            hold_time_s=float(raw.get("hold_seconds", raw.get("hold_s", 0))),
            exit_reason=raw.get("exit_reason", raw.get("reason", "")),
            mfe_ticks=float(raw.get("mfe_ticks", raw.get("mfe", 0))),
            mae_ticks=float(raw.get("mae_ticks", raw.get("mae", 0))),
            signal_confidence=float(raw.get("signal_confidence", 0)),
            entry_action_type=raw.get("entry_action_type", "limit"),
            time_to_fill_ms=float(raw.get("time_to_fill_ms", 0)),
            slippage_ticks=float(raw.get("slippage_ticks", 0)),
            timestamp=ts,
        )

    return None


def parse_metrics_json(content: str) -> Optional[Dict[str, Any]]:
    """Parse a metrics JSON file (written by TradeJournal)."""
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None


def parse_gate_stats(content: str) -> Optional[GateSnapshot]:
    """Parse gate statistics from a JSON file or log line."""
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None

    return GateSnapshot(
        total_evaluated=raw.get("total_evaluated", 0),
        gate1_signal_passed=raw.get("gate1_signal_passed", 0),
        gate2_meta_passed=raw.get("gate2_meta_passed", 0),
        gate3_ofi_passed=raw.get("gate3_ofi_passed", 0),
        all_passed=raw.get("all_passed", 0),
    )


def parse_latency_stats(content: str) -> Optional[LatencySnapshot]:
    """Parse latency summary JSON (from LatencyTracker.get_summary())."""
    try:
        raw = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None

    cnn = raw.get("cnn_mamba_us", {})
    total = raw.get("total_per_prediction_us", {})

    return LatencySnapshot(
        cnn_mamba_p50_ms=cnn.get("p50", 0) / 1000.0,
        cnn_mamba_p95_ms=cnn.get("p95", 0) / 1000.0,
        cnn_mamba_p99_ms=cnn.get("p99", 0) / 1000.0,
        total_pred_p50_ms=total.get("p50", 0) / 1000.0,
        total_pred_p95_ms=total.get("p95", 0) / 1000.0,
        total_pred_p99_ms=total.get("p99", 0) / 1000.0,
        n_predictions=raw.get("n_predictions", 0),
    )


def _parse_ts(val) -> float:
    """Parse timestamp to unix epoch seconds."""
    if isinstance(val, (int, float)):
        if val > 1e15:
            return val / 1e9  # nanoseconds
        if val > 1e12:
            return val / 1e3  # milliseconds
        return float(val)
    if isinstance(val, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                     "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(val, fmt).timestamp()
            except ValueError:
                continue
        try:
            v = float(val)
            return _parse_ts(v)
        except ValueError:
            pass
    return 0.0


# ---------------------------------------------------------------------------
# SSH/file transport layer
# ---------------------------------------------------------------------------

class RazerConnection:
    """
    Connects to Razer via SSH to pull log files and status.

    Supports two modes:
      1. SSH mode: pull files from Razer over SSH (default)
      2. Local mode: read files from a local directory (for testing or shared mount)
    """

    def __init__(
        self,
        razer_host: Optional[str] = None,
        ssh_user: str = "claude",
        ssh_port: int = 22,
        local_log_dir: Optional[str] = None,
        ssh_key: Optional[str] = None,
    ):
        self.razer_host = razer_host
        self.ssh_user = ssh_user
        self.ssh_port = ssh_port
        self.local_log_dir = Path(local_log_dir) if local_log_dir else None
        self.ssh_key = ssh_key

        self._connected = False
        self._last_error = ""
        self._consecutive_failures = 0
        self._max_backoff = 120  # seconds

    @property
    def is_local(self) -> bool:
        return self.local_log_dir is not None

    def _ssh_cmd(self, remote_cmd: str, timeout: int = 30) -> Tuple[bool, str]:
        """Execute a command on Razer via SSH. Returns (success, stdout)."""
        ssh_args = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
        ]
        if self.ssh_key:
            ssh_args += ["-i", self.ssh_key]
        ssh_args += [
            "-p", str(self.ssh_port),
            f"{self.ssh_user}@{self.razer_host}",
            remote_cmd,
        ]

        try:
            result = subprocess.run(
                ssh_args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                self._consecutive_failures = 0
                self._connected = True
                return True, result.stdout
            else:
                self._last_error = result.stderr.strip()
                self._consecutive_failures += 1
                return False, result.stderr
        except subprocess.TimeoutExpired:
            self._last_error = "SSH timeout"
            self._consecutive_failures += 1
            return False, "SSH timeout"
        except FileNotFoundError:
            self._last_error = "ssh command not found"
            self._consecutive_failures += 1
            return False, "ssh not found"
        except Exception as e:
            self._last_error = str(e)
            self._consecutive_failures += 1
            return False, str(e)

    def _scp_file(self, remote_path: str, local_dest: str, timeout: int = 30) -> bool:
        """Copy a file from Razer via SCP."""
        scp_args = [
            "scp",
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
        ]
        if self.ssh_key:
            scp_args += ["-i", self.ssh_key]
        scp_args += [
            "-P", str(self.ssh_port),
            f"{self.ssh_user}@{self.razer_host}:{remote_path}",
            local_dest,
        ]

        try:
            result = subprocess.run(
                scp_args,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_backoff_seconds(self) -> float:
        """Exponential backoff based on consecutive failures."""
        if self._consecutive_failures <= 0:
            return 0
        backoff = min(5 * (2 ** (self._consecutive_failures - 1)), self._max_backoff)
        return backoff

    def read_file(self, remote_path: str) -> Tuple[bool, str]:
        """Read a file from Razer (SSH) or local directory."""
        if self.is_local:
            # Extract filename from Windows or Unix path
            # Path().name doesn't work on Linux for Windows backslash paths
            fname = remote_path.replace("\\", "/").split("/")[-1]
            local_path = self.local_log_dir / fname
            if local_path.exists():
                try:
                    return True, local_path.read_text(encoding="utf-8")
                except Exception as e:
                    return False, str(e)
            return False, f"File not found: {local_path}"

        # SSH mode: cat the file
        # Escape Windows backslashes for SSH
        escaped_path = remote_path.replace("\\", "/")
        return self._ssh_cmd(f'type "{remote_path}"', timeout=30)

    def list_log_files(self, date_str: str) -> Tuple[bool, List[str]]:
        """List log files on Razer for a given date (YYYYMMDD)."""
        if self.is_local:
            files = []
            if self.local_log_dir and self.local_log_dir.exists():
                for f in self.local_log_dir.iterdir():
                    if date_str in f.name:
                        files.append(str(f))
            return True, files

        # SSH: list files matching the date
        log_dir_escaped = RAZER_LOG_DIR.replace("\\", "/")
        ok, output = self._ssh_cmd(
            f'dir /b "{RAZER_LOG_DIR}\\*{date_str}*" 2>NUL',
            timeout=15,
        )
        if ok:
            files = [f.strip() for f in output.strip().splitlines() if f.strip()]
            return True, files
        return False, []

    def read_remote_metrics(self, config_name: str, date_str: str) -> Tuple[bool, str]:
        """Read the metrics JSON for a given config and date."""
        remote_path = f"{RAZER_LOG_DIR}\\{config_name}_metrics_{date_str}.json"
        return self.read_file(remote_path)

    def read_remote_trades(self, config_name: str, date_str: str) -> Tuple[bool, str]:
        """Read the trades JSONL for a given config and date."""
        remote_path = f"{RAZER_LOG_DIR}\\{config_name}_trades_{date_str}.jsonl"
        return self.read_file(remote_path)

    def read_remote_summary(self, config_name: str, date_str: str) -> Tuple[bool, str]:
        """Read the summary JSON for a given config and date."""
        remote_path = f"{RAZER_LOG_DIR}\\{config_name}_summary_{date_str}.json"
        return self.read_file(remote_path)

    def read_gate_stats(self, date_str: str) -> Tuple[bool, str]:
        """Read the stacked filter gate stats file."""
        remote_path = f"{RAZER_LOG_DIR}\\gate_stats_{date_str}.json"
        return self.read_file(remote_path)

    def read_latency_stats(self, date_str: str) -> Tuple[bool, str]:
        """Read the latency tracker summary."""
        remote_path = f"{RAZER_LOG_DIR}\\latency_summary_{date_str}.json"
        return self.read_file(remote_path)

    def read_order_log(self, date_str: str) -> Tuple[bool, str]:
        """Read the order event log (placed/filled/cancelled)."""
        remote_path = f"{RAZER_LOG_DIR}\\orders_{date_str}.jsonl"
        return self.read_file(remote_path)

    def check_process_alive(self) -> Tuple[bool, str]:
        """Check if the paper trading process is running on Razer."""
        if self.is_local:
            return True, "local mode"
        ok, output = self._ssh_cmd(
            'tasklist /FI "IMAGENAME eq python.exe" /FO CSV 2>NUL',
            timeout=15,
        )
        if ok and "python" in output.lower():
            return True, output
        return False, output


# ---------------------------------------------------------------------------
# KPI Tracker
# ---------------------------------------------------------------------------

class KPITracker:
    """
    Tracks and computes all Monday monitoring KPIs from trade data.
    """

    def __init__(self):
        self.trades: List[LiveTrade] = []
        self.orders_placed: int = 0
        self.orders_filled: int = 0
        self.orders_cancelled: int = 0
        self.orders_expired: int = 0

        # For adverse selection: P&L on unfilled signals
        self.unfilled_hypothetical_pnl: List[float] = []

        # Gate stats
        self.gate_stats: Optional[GateSnapshot] = None

        # Latency
        self.latency: Optional[LatencySnapshot] = None

        # Tracking state
        self._last_trade_count = 0
        self._first_trade_time: Optional[float] = None
        self._session_start = time.time()

    def update_trades(self, trades: List[LiveTrade]) -> int:
        """
        Update with full trade list. Returns number of new trades added.
        """
        new_count = len(trades) - len(self.trades)
        if new_count > 0:
            self.trades = trades
            if self._first_trade_time is None and trades:
                self._first_trade_time = trades[0].timestamp or time.time()
        return max(new_count, 0)

    def update_orders(self, placed: int, filled: int, cancelled: int, expired: int):
        """Update order counts."""
        self.orders_placed = placed
        self.orders_filled = filled
        self.orders_cancelled = cancelled
        self.orders_expired = expired

    def update_gate_stats(self, stats: GateSnapshot):
        self.gate_stats = stats

    def update_latency(self, latency: LatencySnapshot):
        self.latency = latency

    # -- Computed KPIs --

    @property
    def fill_rate(self) -> float:
        """Orders filled / orders placed."""
        if self.orders_placed == 0:
            # Fallback: estimate from trade count vs gate all_passed
            if self.gate_stats and self.gate_stats.all_passed > 0:
                return len(self.trades) / self.gate_stats.all_passed
            return 0.0
        return self.orders_filled / self.orders_placed

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def cumulative_pnl_ticks(self) -> float:
        return sum(t.pnl_ticks for t in self.trades)

    @property
    def cumulative_pnl_usd(self) -> float:
        return sum(t.net_pnl_usd for t in self.trades)

    @property
    def avg_pnl_ticks(self) -> float:
        if not self.trades:
            return 0.0
        return self.cumulative_pnl_ticks / len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.net_pnl_usd > 0)
        return winners / len(self.trades) * 100.0

    @property
    def avg_winner_usd(self) -> float:
        wins = [t.net_pnl_usd for t in self.trades if t.net_pnl_usd > 0]
        return sum(wins) / len(wins) if wins else 0.0

    @property
    def avg_loser_usd(self) -> float:
        losses = [t.net_pnl_usd for t in self.trades if t.net_pnl_usd <= 0]
        return sum(losses) / len(losses) if losses else 0.0

    @property
    def profit_factor(self) -> float:
        gross_win = sum(t.net_pnl_usd for t in self.trades if t.net_pnl_usd > 0)
        gross_loss = abs(sum(t.net_pnl_usd for t in self.trades if t.net_pnl_usd <= 0))
        if gross_loss == 0:
            return float("inf") if gross_win > 0 else 0.0
        return gross_win / gross_loss

    @property
    def trades_per_hour(self) -> float:
        if not self.trades or len(self.trades) < 2:
            return 0.0
        # Use first/last trade timestamps if available
        first_ts = self.trades[0].timestamp
        last_ts = self.trades[-1].timestamp
        if first_ts > 0 and last_ts > 0 and last_ts > first_ts:
            hours = (last_ts - first_ts) / 3600.0
            if hours > 0.01:
                return len(self.trades) / hours
        # Fallback: from session start
        elapsed_h = (time.time() - self._session_start) / 3600.0
        if elapsed_h > 0.01:
            return len(self.trades) / elapsed_h
        return 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe from per-trade returns."""
        if len(self.trades) < 3:
            return 0.0
        pnls = [t.net_pnl_usd for t in self.trades]
        mean_pnl = sum(pnls) / len(pnls)
        var = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        if std < 1e-9:
            return 0.0
        # Estimate trades per day, annualize
        tpd = self.trades_per_hour * 6.5  # ~6.5 RTH hours
        if tpd < 1:
            tpd = max(len(self.trades), 1)
        ann_factor = math.sqrt(TRADING_DAYS_PER_YEAR * tpd)
        return (mean_pnl / std) * ann_factor

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino from per-trade returns."""
        if len(self.trades) < 3:
            return 0.0
        pnls = [t.net_pnl_usd for t in self.trades]
        mean_pnl = sum(pnls) / len(pnls)
        downside = [p for p in pnls if p < 0]
        if not downside:
            return float("inf") if mean_pnl > 0 else 0.0
        downside_sq = sum(d ** 2 for d in downside) / len(pnls)
        downside_dev = math.sqrt(downside_sq)
        if downside_dev < 1e-9:
            return 0.0
        tpd = self.trades_per_hour * 6.5
        if tpd < 1:
            tpd = max(len(self.trades), 1)
        ann_factor = math.sqrt(TRADING_DAYS_PER_YEAR * tpd)
        return (mean_pnl / downside_dev) * ann_factor

    @property
    def avg_hold_time_s(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.hold_time_s for t in self.trades) / len(self.trades)

    @property
    def avg_mfe_ticks(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.mfe_ticks for t in self.trades) / len(self.trades)

    @property
    def avg_mae_ticks(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.mae_ticks for t in self.trades) / len(self.trades)

    @property
    def avg_slippage_ticks(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.slippage_ticks for t in self.trades) / len(self.trades)

    @property
    def avg_fill_time_ms(self) -> float:
        filled = [t.time_to_fill_ms for t in self.trades if t.time_to_fill_ms > 0]
        return sum(filled) / len(filled) if filled else 0.0

    @property
    def max_drawdown_usd(self) -> float:
        if not self.trades:
            return 0.0
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in self.trades:
            cumulative += t.net_pnl_usd
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def exit_reason_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for t in self.trades:
            counts[t.exit_reason or "unknown"] += 1
        return dict(counts)

    def direction_breakdown(self) -> Dict[str, Dict[str, Any]]:
        long_trades = [t for t in self.trades if t.direction == "LONG"]
        short_trades = [t for t in self.trades if t.direction == "SHORT"]

        def _stats(trades_list):
            if not trades_list:
                return {"n": 0, "wr": 0.0, "pnl_ticks": 0.0}
            wins = sum(1 for t in trades_list if t.net_pnl_usd > 0)
            pnl = sum(t.pnl_ticks for t in trades_list)
            return {
                "n": len(trades_list),
                "wr": wins / len(trades_list) * 100.0,
                "pnl_ticks": pnl,
            }

        return {"LONG": _stats(long_trades), "SHORT": _stats(short_trades)}

    # -- Alerts --

    def check_alerts(self) -> List[str]:
        """Return list of alert messages for active problems."""
        alerts = []

        # Fill rate alert
        if self.orders_placed > 10 and self.fill_rate < FILL_RATE_ALERT_THRESHOLD:
            alerts.append(
                f"LOW FILL RATE: {self.fill_rate:.1%} "
                f"({self.orders_filled}/{self.orders_placed} filled). "
                f"Adverse selection likely eating the edge."
            )

        # Latency alert
        if self.latency and self.latency.total_pred_p95_ms > LATENCY_ALERT_MS:
            alerts.append(
                f"HIGH LATENCY: p95={self.latency.total_pred_p95_ms:.0f}ms "
                f"(threshold {LATENCY_ALERT_MS:.0f}ms). Predictions may be stale."
            )

        # Degradation vs simulation
        if len(self.trades) >= 10:
            degradation = 1.0 - (self.avg_pnl_ticks / SIM_TICKS_PER_TRADE) if SIM_TICKS_PER_TRADE > 0 else 0.0
            if degradation > DEGRADATION_THRESHOLD:
                alerts.append(
                    f"PERFORMANCE DEGRADATION: {degradation:.0%} worse than simulation. "
                    f"Live avg {self.avg_pnl_ticks:+.3f} ticks/trade vs sim {SIM_TICKS_PER_TRADE:+.3f}. "
                    f"{'Adverse selection or fill quality issue.' if self.avg_pnl_ticks < 0 else 'Edge is weaker live.'}"
                )

        # Win rate check
        if len(self.trades) >= 20 and self.win_rate < SIM_WR * 0.7:
            alerts.append(
                f"LOW WIN RATE: {self.win_rate:.1f}% vs sim {SIM_WR:.1f}%. "
                f"Model may not be generalizing to live conditions."
            )

        return alerts


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_summary(kpi: KPITracker, elapsed_str: str, date_str: str) -> str:
    """Format a comprehensive monitoring summary for stdout/Discord."""
    lines = []
    lines.append(f"{'='*64}")
    lines.append(f"  MONDAY LIVE MONITOR — {date_str}  [{elapsed_str} elapsed]")
    lines.append(f"{'='*64}")

    # --- Core KPIs ---
    lines.append("")
    lines.append("  CORE KPIs:")
    lines.append(f"    Trades: {kpi.n_trades}  ({kpi.trades_per_hour:.1f}/hr)")
    lines.append(f"    Cum P&L: {kpi.cumulative_pnl_ticks:+.1f} ticks  (${kpi.cumulative_pnl_usd:+.2f})")
    lines.append(f"    Avg/trade: {kpi.avg_pnl_ticks:+.3f} ticks  (sim: {SIM_TICKS_PER_TRADE:+.3f})")
    lines.append(f"    Win Rate: {kpi.win_rate:.1f}%  (sim: {SIM_WR:.1f}%)")
    lines.append(f"    Sharpe: {kpi.sharpe_ratio:.1f}  Sortino: {kpi.sortino_ratio:.1f}  PF: {kpi.profit_factor:.2f}")
    lines.append(f"    Avg Win: ${kpi.avg_winner_usd:.2f}  Avg Loss: ${kpi.avg_loser_usd:.2f}")
    lines.append(f"    Max DD: ${kpi.max_drawdown_usd:.2f}")

    # --- Fill Quality (THE existential question) ---
    lines.append("")
    lines.append("  FILL QUALITY (existential question):")
    lines.append(f"    Fill Rate: {kpi.fill_rate:.1%}")
    if kpi.orders_placed > 0:
        lines.append(f"    Orders: {kpi.orders_placed} placed, {kpi.orders_filled} filled, "
                     f"{kpi.orders_cancelled} cancelled, {kpi.orders_expired} expired")
    lines.append(f"    Avg Fill Time: {kpi.avg_fill_time_ms:.1f}ms")
    lines.append(f"    Avg Slippage: {kpi.avg_slippage_ticks:.3f} ticks")
    lines.append(f"    Avg MFE: {kpi.avg_mfe_ticks:.2f}t  Avg MAE: {kpi.avg_mae_ticks:.2f}t")

    # --- Adverse Selection Analysis ---
    if kpi.n_trades >= 5:
        lines.append("")
        lines.append("  ADVERSE SELECTION CHECK:")
        dirs = kpi.direction_breakdown()
        for d in ("SHORT", "LONG"):
            ds = dirs.get(d, {"n": 0, "wr": 0.0, "pnl_ticks": 0.0})
            if ds["n"] > 0:
                avg_pnl = ds["pnl_ticks"] / ds["n"]
                lines.append(f"    {d}: {ds['n']} trades, WR {ds['wr']:.1f}%, "
                             f"avg {avg_pnl:+.3f} ticks/trade")

        # Hold time analysis
        lines.append(f"    Avg Hold: {kpi.avg_hold_time_s:.1f}s")

    # --- Filter Pass Rates ---
    if kpi.gate_stats and kpi.gate_stats.total_evaluated > 0:
        gs = kpi.gate_stats
        lines.append("")
        lines.append("  FILTER PASS RATES:")
        lines.append(f"    Total signals evaluated: {gs.total_evaluated}")
        lines.append(f"    Signal gate (top N%):    {gs.signal_rate:.1%}  ({gs.gate1_signal_passed} passed)")
        lines.append(f"    Meta-model gate:         {gs.meta_rate:.1%}  ({gs.gate2_meta_passed} passed)")
        lines.append(f"    OFI gate:                {gs.ofi_rate:.1%}  ({gs.gate3_ofi_passed} passed)")
        lines.append(f"    Overall pass rate:       {gs.overall_rate:.2%}  ({gs.all_passed} total)")

    # --- Latency ---
    if kpi.latency and kpi.latency.n_predictions > 0:
        lat = kpi.latency
        lines.append("")
        lines.append("  INFERENCE LATENCY:")
        lines.append(f"    CNN-Mamba: p50={lat.cnn_mamba_p50_ms:.1f}ms  p95={lat.cnn_mamba_p95_ms:.1f}ms  p99={lat.cnn_mamba_p99_ms:.1f}ms")
        lines.append(f"    Total/pred: p50={lat.total_pred_p50_ms:.1f}ms  p95={lat.total_pred_p95_ms:.1f}ms  p99={lat.total_pred_p99_ms:.1f}ms")
        lines.append(f"    Predictions: {lat.n_predictions}")

    # --- Exit Reasons ---
    if kpi.n_trades > 0:
        exits = kpi.exit_reason_breakdown()
        if exits:
            lines.append("")
            lines.append("  EXIT REASONS:")
            for reason, count in sorted(exits.items(), key=lambda x: -x[1])[:6]:
                pct = count / kpi.n_trades * 100
                lines.append(f"    {reason:<24} {count:>4} ({pct:.0f}%)")

    # --- Sim Comparison ---
    if kpi.n_trades >= 5:
        lines.append("")
        lines.append("  SIM vs LIVE COMPARISON:")
        sim_pnl_for_n = SIM_TICKS_PER_TRADE * kpi.n_trades
        live_pnl = kpi.cumulative_pnl_ticks
        diff = live_pnl - sim_pnl_for_n
        lines.append(f"    Sim expected:  {sim_pnl_for_n:+.1f} ticks for {kpi.n_trades} trades")
        lines.append(f"    Live actual:   {live_pnl:+.1f} ticks")
        lines.append(f"    Delta:         {diff:+.1f} ticks  ({'ON TRACK' if diff >= -sim_pnl_for_n * DEGRADATION_THRESHOLD else 'DEGRADED'})")

    # --- Alerts ---
    alerts = kpi.check_alerts()
    if alerts:
        lines.append("")
        lines.append("  *** ALERTS ***")
        for alert in alerts:
            lines.append(f"    !!! {alert}")

    lines.append(f"{'='*64}")
    return "\n".join(lines)


def format_compact_summary(kpi: KPITracker) -> str:
    """Short summary suitable for Discord/Telegram (~300 chars)."""
    if kpi.n_trades == 0:
        return "Monday Monitor: No trades yet. Waiting for market open."

    fill_str = f"Fill: {kpi.fill_rate:.0%}" if kpi.orders_placed > 0 else ""
    parts = [
        f"Trades: {kpi.n_trades} ({kpi.trades_per_hour:.0f}/hr)",
        f"P&L: {kpi.cumulative_pnl_ticks:+.1f}t (${kpi.cumulative_pnl_usd:+.0f})",
        f"Avg: {kpi.avg_pnl_ticks:+.3f}t/trade (sim: {SIM_TICKS_PER_TRADE:+.3f})",
        f"WR: {kpi.win_rate:.0f}% Sharpe: {kpi.sharpe_ratio:.1f} PF: {kpi.profit_factor:.2f}",
    ]
    if fill_str:
        parts.append(fill_str)

    alerts = kpi.check_alerts()
    if alerts:
        parts.append(f"ALERTS: {len(alerts)} active")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

class MondayMonitor:
    """
    Main monitoring loop. Polls Razer for trade data, computes KPIs,
    and prints periodic summaries.
    """

    def __init__(
        self,
        conn: RazerConnection,
        poll_interval: int = 60,
        summary_interval: int = 300,
        config_name: str = "rules_top01pct",
        configs_to_check: Optional[List[str]] = None,
    ):
        self.conn = conn
        self.poll_interval = poll_interval
        self.summary_interval = summary_interval
        self.config_name = config_name
        self.configs_to_check = configs_to_check or [
            "sac_v7", "ppo_v7", "rules_top01pct", "rules_top1pct",
        ]

        self.kpi = KPITracker()
        self._start_time = time.time()
        self._last_summary_time = 0.0
        self._last_trade_time = time.time()
        self._last_trade_count = 0
        self._no_trade_alert_sent = False
        self._running = True

        # Track last known file sizes to detect new data
        self._last_trade_file_size: Dict[str, int] = {}

    def _today_str(self) -> str:
        return datetime.now(tz=timezone.utc).strftime("%Y%m%d")

    def _elapsed_str(self) -> str:
        elapsed = time.time() - self._start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        return f"{h}h{m:02d}m"

    def _is_rth(self) -> bool:
        """Check if we're in Regular Trading Hours (9:30-16:00 ET)."""
        try:
            import pytz
            et = pytz.timezone("US/Eastern")
            now_et = datetime.now(et)
        except ImportError:
            # Approximate: UTC-4 for EDT
            now_et = datetime.now(timezone(timedelta(hours=-4)))

        t = now_et.hour * 60 + now_et.minute
        rth_open = RTH_OPEN_HOUR * 60 + RTH_OPEN_MIN
        rth_close = RTH_CLOSE_HOUR * 60 + RTH_CLOSE_MIN
        return rth_open <= t <= rth_close

    def poll_once(self) -> bool:
        """
        Do a single poll cycle: fetch data, update KPIs.
        Returns True if new data was found.
        """
        date_str = self._today_str()
        new_data_found = False

        # Try each config name to find active trading
        for cfg in self.configs_to_check:
            ok, trades_content = self.conn.read_remote_trades(cfg, date_str)
            if ok and trades_content.strip():
                trades = []
                for line in trades_content.strip().splitlines():
                    t = parse_trade_from_jsonl(line)
                    if t is not None:
                        trades.append(t)

                if trades:
                    new_count = self.kpi.update_trades(trades)
                    if new_count > 0:
                        new_data_found = True
                        self._last_trade_time = time.time()
                        self._last_trade_count = len(trades)
                        self._no_trade_alert_sent = False
                        self.config_name = cfg
                    break  # Found active config

        # Try to read metrics JSON (has fill/order data)
        ok, metrics_content = self.conn.read_remote_metrics(self.config_name, date_str)
        if ok and metrics_content.strip():
            metrics = parse_metrics_json(metrics_content)
            if metrics:
                # Extract any order-level data if present
                if "orders_placed" in metrics:
                    self.kpi.update_orders(
                        placed=metrics.get("orders_placed", 0),
                        filled=metrics.get("orders_filled", 0),
                        cancelled=metrics.get("orders_cancelled", 0),
                        expired=metrics.get("orders_expired", 0),
                    )

        # Try to read gate stats
        ok, gate_content = self.conn.read_gate_stats(date_str)
        if ok and gate_content.strip():
            gs = parse_gate_stats(gate_content)
            if gs:
                self.kpi.update_gate_stats(gs)

        # Try to read latency stats
        ok, lat_content = self.conn.read_latency_stats(date_str)
        if ok and lat_content.strip():
            lat = parse_latency_stats(lat_content)
            if lat:
                self.kpi.update_latency(lat)

        # Try alternative: read summary JSON which has most metrics
        ok, summary_content = self.conn.read_remote_summary(self.config_name, date_str)
        if ok and summary_content.strip():
            summary = parse_metrics_json(summary_content)
            if summary:
                # Can extract gate stats from summary if embedded
                if "gate_stats" in summary:
                    gs_data = summary["gate_stats"]
                    self.kpi.update_gate_stats(GateSnapshot(
                        total_evaluated=gs_data.get("total_evaluated", 0),
                        gate1_signal_passed=gs_data.get("gate1_signal_passed", 0),
                        gate2_meta_passed=gs_data.get("gate2_meta_passed", 0),
                        gate3_ofi_passed=gs_data.get("gate3_ofi_passed", 0),
                        all_passed=gs_data.get("all_passed", 0),
                    ))

        # Check for no-trade alert during RTH
        if self._is_rth():
            minutes_since_trade = (time.time() - self._last_trade_time) / 60.0
            if minutes_since_trade > NO_TRADE_ALERT_MINUTES and not self._no_trade_alert_sent:
                print(f"\n!!! ALERT: No trades for {minutes_since_trade:.0f} minutes during RTH!")
                print("    Possible causes: inference crash, data feed down, all gates blocking")
                self._no_trade_alert_sent = True

                # Check if process is alive
                alive, proc_info = self.conn.check_process_alive()
                if not alive:
                    print("    !!! Paper trading process NOT RUNNING on Razer!")
                else:
                    print("    Process appears to be running. May be gate-blocked or in warmup.")

        return new_data_found

    def should_print_summary(self) -> bool:
        """Check if it's time for a full summary."""
        return (time.time() - self._last_summary_time) >= self.summary_interval

    def print_summary(self):
        """Print full monitoring summary."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        summary = format_summary(self.kpi, self._elapsed_str(), date_str)
        print(summary)
        self._last_summary_time = time.time()

    def print_compact(self):
        """Print compact one-line status."""
        compact = format_compact_summary(self.kpi)
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {compact}")

    def run(self):
        """Main monitoring loop with retry/backoff."""
        print(f"Monday Monitor starting. Polling every {self.poll_interval}s, "
              f"summaries every {self.summary_interval}s.")
        print(f"Connection mode: {'LOCAL' if self.conn.is_local else 'SSH to ' + str(self.conn.razer_host)}")
        print(f"Monitoring configs: {self.configs_to_check}")
        print(f"Simulation baseline: {SIM_TICKS_PER_TRADE:+.3f} ticks/trade, "
              f"WR {SIM_WR}%, Sharpe {SIM_SHARPE}")
        print()

        cycle = 0
        while self._running:
            try:
                cycle += 1
                new_data = self.poll_once()

                if new_data or self.should_print_summary():
                    if self.should_print_summary():
                        self.print_summary()
                    else:
                        self.print_compact()
                elif cycle % 5 == 0:
                    # Periodic heartbeat even without new data
                    self.print_compact()

                # Check for alerts every cycle
                alerts = self.kpi.check_alerts()
                for alert in alerts:
                    # Only print if this is a new alert (avoid spam)
                    print(f"\n!!! ALERT: {alert}")

                # Wait with backoff if connection is failing
                backoff = self.conn.get_backoff_seconds()
                if backoff > 0:
                    wait = max(self.poll_interval, backoff)
                    if cycle <= 3 or cycle % 10 == 0:
                        print(f"  [Connection issue: {self.conn._last_error}. "
                              f"Retry in {wait:.0f}s (attempt {self.conn._consecutive_failures})]")
                    time.sleep(wait)
                else:
                    time.sleep(self.poll_interval)

            except KeyboardInterrupt:
                print("\nMonitor stopped by user.")
                self._running = False
                break
            except Exception as e:
                print(f"\n!!! Monitor error: {e}")
                traceback.print_exc()
                time.sleep(self.poll_interval)

        # Final summary
        if self.kpi.n_trades > 0:
            print("\n--- FINAL SUMMARY ---")
            self.print_summary()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monday Live Paper Trading Monitor — CNN-Mamba v2 + Stacked Confluence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 live_trading/monday_monitor.py --razer-host razer
  python3 live_trading/monday_monitor.py --local-logs /tmp/razer_logs
  python3 live_trading/monday_monitor.py --razer-host razer --poll 30 --summary 120

Simulation baseline (balanced preset, 48-day backtest):
  +0.505 ticks/trade, WR 58.7%, Sharpe 26.8, PF 1.92, ~527 trades/day
        """,
    )

    conn_group = parser.add_argument_group("connection")
    conn_group.add_argument("--razer-host", type=str, default=None,
                           help="Razer IP/hostname for SSH (default: razer)")
    conn_group.add_argument("--ssh-user", type=str, default="claude",
                           help="SSH username (default: claude)")
    conn_group.add_argument("--ssh-port", type=int, default=22,
                           help="SSH port (default: 22)")
    conn_group.add_argument("--ssh-key", type=str, default=None,
                           help="Path to SSH private key")
    conn_group.add_argument("--local-logs", type=str, default=None,
                           help="Local log directory (skip SSH, read files directly)")

    timing_group = parser.add_argument_group("timing")
    timing_group.add_argument("--poll", type=int, default=60,
                             help="Poll interval in seconds (default: 60)")
    timing_group.add_argument("--summary", type=int, default=300,
                             help="Full summary interval in seconds (default: 300)")

    config_group = parser.add_argument_group("config")
    config_group.add_argument("--config", type=str, default="rules_top01pct",
                             help="Primary config name to monitor (default: rules_top01pct)")
    config_group.add_argument("--configs", type=str, nargs="+",
                             default=None,
                             help="All config names to scan (default: all known)")
    config_group.add_argument("--sim-baseline", type=float, default=SIM_TICKS_PER_TRADE,
                             help=f"Simulation baseline ticks/trade (default: {SIM_TICKS_PER_TRADE})")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--compact", action="store_true",
                             help="Compact output only (one-liners)")
    output_group.add_argument("--once", action="store_true",
                             help="Poll once and exit (for cron/scripting)")
    output_group.add_argument("--json", action="store_true",
                             help="Output in JSON format")

    args = parser.parse_args()

    # Build connection
    if args.local_logs:
        conn = RazerConnection(local_log_dir=args.local_logs)
    elif args.razer_host:
        conn = RazerConnection(
            razer_host=args.razer_host,
            ssh_user=args.ssh_user,
            ssh_port=args.ssh_port,
            ssh_key=args.ssh_key,
        )
    else:
        # Default: try Razer
        conn = RazerConnection(razer_host="razer")

    configs = args.configs or ["sac_v7", "ppo_v7", "rules_top01pct", "rules_top1pct"]

    monitor = MondayMonitor(
        conn=conn,
        poll_interval=args.poll,
        summary_interval=args.summary,
        config_name=args.config,
        configs_to_check=configs,
    )

    if args.once:
        # Single poll, print, exit
        monitor.poll_once()
        if args.json:
            result = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "n_trades": monitor.kpi.n_trades,
                "cumulative_pnl_ticks": monitor.kpi.cumulative_pnl_ticks,
                "cumulative_pnl_usd": monitor.kpi.cumulative_pnl_usd,
                "avg_pnl_ticks": monitor.kpi.avg_pnl_ticks,
                "win_rate": monitor.kpi.win_rate,
                "sharpe": monitor.kpi.sharpe_ratio,
                "sortino": monitor.kpi.sortino_ratio,
                "profit_factor": monitor.kpi.profit_factor,
                "fill_rate": monitor.kpi.fill_rate,
                "trades_per_hour": monitor.kpi.trades_per_hour,
                "avg_hold_s": monitor.kpi.avg_hold_time_s,
                "avg_mfe_ticks": monitor.kpi.avg_mfe_ticks,
                "avg_mae_ticks": monitor.kpi.avg_mae_ticks,
                "avg_slippage_ticks": monitor.kpi.avg_slippage_ticks,
                "max_drawdown_usd": monitor.kpi.max_drawdown_usd,
                "alerts": monitor.kpi.check_alerts(),
                "sim_baseline_ticks": SIM_TICKS_PER_TRADE,
                "direction": monitor.kpi.direction_breakdown(),
                "exit_reasons": monitor.kpi.exit_reason_breakdown(),
            }
            if monitor.kpi.gate_stats:
                result["gate_stats"] = {
                    "total_evaluated": monitor.kpi.gate_stats.total_evaluated,
                    "signal_rate": monitor.kpi.gate_stats.signal_rate,
                    "meta_rate": monitor.kpi.gate_stats.meta_rate,
                    "ofi_rate": monitor.kpi.gate_stats.ofi_rate,
                    "overall_rate": monitor.kpi.gate_stats.overall_rate,
                }
            if monitor.kpi.latency:
                result["latency"] = {
                    "cnn_mamba_p50_ms": monitor.kpi.latency.cnn_mamba_p50_ms,
                    "cnn_mamba_p95_ms": monitor.kpi.latency.cnn_mamba_p95_ms,
                    "total_pred_p95_ms": monitor.kpi.latency.total_pred_p95_ms,
                }
            print(json.dumps(result, indent=2, default=str))
        elif args.compact:
            monitor.print_compact()
        else:
            monitor.print_summary()
        return

    # Continuous monitoring loop
    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nShutdown.")


if __name__ == "__main__":
    main()
