#!/usr/bin/env python3
"""
Paper Trading Performance Report — Unified multi-config comparison.

Reads trade journals from all paper trading configs and produces a
formatted comparison report.

Supported input formats:
  1. TradeJournal JSONL (new): trades_{config}_{date}.jsonl
  2. RL paper trades JSONL (existing): rl_paper_trades_{symbol}_{ts}.jsonl
  3. Legacy signal logs: mamba_v2_signals_*.jsonl, paper_mamba_v2_*.jsonl

Usage:
  python performance_report.py                     # today
  python performance_report.py --date 2026-05-04   # specific date
  python performance_report.py --all               # all available dates
  python performance_report.py --discord            # compact Discord output
  python performance_report.py --logdir /path/to/logs  # custom log dir
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376

CONFIGS = ["sac_v7", "ppo_v7", "rules_top01pct", "rules_top1pct"]
CONFIG_DISPLAY = {
    "sac_v7":          "SAC v7",
    "ppo_v7":          "PPO v7",
    "rules_top01pct":  "Rules Top0.1%",
    "rules_top1pct":   "Rules Top1%",
}

DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "logs"

# Try numpy for Sharpe/Sortino, fallback to pure python
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ══════════════════════════════════════════════════════════════════════════════
# Pure-python math helpers (used when numpy unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: List[float], ddof: int = 1) -> float:
    if len(xs) <= ddof:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - ddof))

def _downside_std(xs: List[float], target: float = 0.0) -> float:
    """Downside deviation (ddof=0, standard for Sortino ratio)."""
    downs = [(x - target) ** 2 for x in xs if x < target]
    if not downs:
        return 0.0
    return math.sqrt(sum(downs) / len(downs))


# ══════════════════════════════════════════════════════════════════════════════
# Trade normalization — convert any format to a canonical dict
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_trade(raw: Dict[str, Any], config_name: str = "") -> Optional[Dict[str, Any]]:
    """
    Return a canonical trade dict or None if the record is not a closed trade.

    Canonical fields:
        ts        : float  (unix epoch seconds)
        dir       : str    ("LONG" or "SHORT")
        entry     : float  (price)
        exit      : float  (price)
        pnl_ticks : float
        pnl_usd   : float
        hold_s    : float  (seconds)
        reason    : str    (exit reason)
        mfe       : float  (ticks, max favorable excursion)
        mae       : float  (ticks, max adverse excursion)
        config    : str
        hour      : int    (hour of entry, 0-23)
    """
    # ── TradeJournal format (new) ────────────────────────────────────────────
    # Has: event="TRADE_CLOSED", config, entry_time, exit_time, direction,
    #       entry_price, exit_price, pnl_ticks, pnl_usd, hold_seconds,
    #       exit_reason, mfe_ticks, mae_ticks
    if raw.get("event") == "TRADE_CLOSED":
        ts = _parse_ts(raw.get("exit_time") or raw.get("ts") or raw.get("timestamp", 0))
        entry_ts = _parse_ts(raw.get("entry_time", ts))
        return {
            "ts":        ts,
            "dir":       raw.get("direction", raw.get("dir", "UNKNOWN")).upper(),
            "entry":     float(raw.get("entry_price", 0)),
            "exit":      float(raw.get("exit_price", 0)),
            "pnl_ticks": float(raw.get("pnl_ticks", 0)),
            "pnl_usd":   float(raw.get("pnl_usd", 0)),
            "hold_s":    float(raw.get("hold_seconds", raw.get("hold_s", 0))),
            "reason":    raw.get("exit_reason", raw.get("reason", "")),
            "mfe":       float(raw.get("mfe_ticks", raw.get("mfe", 0))),
            "mae":       float(raw.get("mae_ticks", raw.get("mae", 0))),
            "config":    raw.get("config", config_name),
            "hour":      _hour_from_ts(entry_ts),
        }

    # ── RL paper trades format (existing) ────────────────────────────────────
    # Has: ts, dir, entry, exit, pnl_ticks, pnl_usd, hold_s, reason, mfe, mae, sortino
    if "pnl_ticks" in raw and "entry" in raw and "exit" in raw:
        ts = _parse_ts(raw.get("ts", 0))
        return {
            "ts":        ts,
            "dir":       raw.get("dir", "UNKNOWN").upper(),
            "entry":     float(raw.get("entry", 0)),
            "exit":      float(raw.get("exit", 0)),
            "pnl_ticks": float(raw.get("pnl_ticks", 0)),
            "pnl_usd":   float(raw.get("pnl_usd", 0)),
            "hold_s":    float(raw.get("hold_s", 0)),
            "reason":    raw.get("reason", ""),
            "mfe":       float(raw.get("mfe", 0)),
            "mae":       float(raw.get("mae", 0)),
            "config":    config_name,
            "hour":      _hour_from_ts(ts),
        }

    # ── Legacy signal log with trade info ────────────────────────────────────
    # mamba_v2_signals / paper_mamba_v2: may have action, pnl, price fields
    if raw.get("action") in ("BUY", "SELL", "CLOSE") and "pnl" in raw:
        ts = _parse_ts(raw.get("ts", raw.get("timestamp", 0)))
        pnl_usd = float(raw.get("pnl", 0))
        return {
            "ts":        ts,
            "dir":       "LONG" if raw.get("action") == "BUY" else "SHORT",
            "entry":     float(raw.get("entry_price", raw.get("price", 0))),
            "exit":      float(raw.get("exit_price", raw.get("price", 0))),
            "pnl_ticks": round(pnl_usd / ES_TICK_VALUE, 4) if ES_TICK_VALUE else 0,
            "pnl_usd":   pnl_usd,
            "hold_s":    float(raw.get("hold_s", raw.get("hold_seconds", 0))),
            "reason":    raw.get("reason", raw.get("exit_reason", "unknown")),
            "mfe":       float(raw.get("mfe", raw.get("mfe_ticks", 0))),
            "mae":       float(raw.get("mae", raw.get("mae_ticks", 0))),
            "config":    config_name,
            "hour":      _hour_from_ts(ts),
        }

    return None


def _parse_ts(val) -> float:
    """Parse timestamp to unix epoch seconds."""
    if isinstance(val, (int, float)):
        # Already epoch. If ms, convert.
        return val / 1000.0 if val > 1e12 else float(val)
    if isinstance(val, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                     "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(val, fmt).timestamp()
            except ValueError:
                continue
        # Try epoch string
        try:
            v = float(val)
            return v / 1000.0 if v > 1e12 else v
        except ValueError:
            pass
    return 0.0


def _hour_from_ts(ts: float) -> int:
    try:
        return datetime.fromtimestamp(ts).hour if ts > 0 else -1
    except (OSError, ValueError):
        return -1


# ══════════════════════════════════════════════════════════════════════════════
# File discovery — find trade files for a given date
# ══════════════════════════════════════════════════════════════════════════════

def _date_str(d: date) -> str:
    return d.strftime("%Y%m%d")

def _date_str_dash(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def discover_trade_files(log_dir: Path, target_date: Optional[date] = None
                         ) -> Dict[str, List[Path]]:
    """
    Return {config_name: [file_paths]} for a given date.

    File naming conventions searched:
      trades_{config}_{YYYYMMDD}*.jsonl          (TradeJournal new format)
      {config}_trades_{YYYYMMDD}*.jsonl
      rl_paper_trades_*_{YYYYMMDD}*.jsonl        (RL existing format)
      paper_mamba_v2_{YYYYMMDD}*.jsonl
      mamba_v2_signals_{YYYYMMDD}*.jsonl
      rl_decisions_{YYYYMMDD}*.jsonl

    Also scans for any .jsonl containing config name and date string.
    """
    if not log_dir.exists():
        return {}

    date_patterns = []
    if target_date:
        date_patterns = [_date_str(target_date), _date_str_dash(target_date)]

    result: Dict[str, List[Path]] = {c: [] for c in CONFIGS}

    for f in sorted(log_dir.iterdir()):
        if not f.name.endswith(".jsonl"):
            continue

        fname = f.name.lower()

        # If date filtering, check date is in filename
        if date_patterns and not any(dp in fname for dp in date_patterns):
            # Also check if file content matches date (for files without date in name)
            # We'll do a quick first-line check
            if not _file_matches_date(f, target_date):
                continue

        # Map file to config
        matched_config = _match_file_to_config(fname)
        if matched_config:
            result[matched_config].append(f)

    return result


def _match_file_to_config(fname: str) -> Optional[str]:
    """Determine which config a filename belongs to."""
    fname_lower = fname.lower()

    # Direct config name match
    for cfg in CONFIGS:
        if cfg in fname_lower:
            return cfg

    # RL paper trades -> check for sac/ppo indicators
    if "rl_paper_trades" in fname_lower or "rl_decisions" in fname_lower:
        if "sac" in fname_lower:
            return "sac_v7"
        if "ppo" in fname_lower:
            return "ppo_v7"
        # Default RL trades to sac_v7 if unspecified
        return "sac_v7"

    # Legacy formats
    if "paper_mamba" in fname_lower or "mamba_v2_signals" in fname_lower:
        return "rules_top1pct"  # Legacy mamba mapped to rules

    return None


def _file_matches_date(f: Path, target_date: Optional[date]) -> bool:
    """Quick check if first trade in file matches the target date."""
    if target_date is None:
        return True
    try:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ts = _parse_ts(rec.get("ts", rec.get("timestamp",
                               rec.get("exit_time", rec.get("entry_time", 0)))))
                if ts > 0:
                    try:
                        d = datetime.fromtimestamp(ts).date()
                        return d == target_date
                    except (OSError, ValueError):
                        pass
                break
    except (json.JSONDecodeError, IOError):
        pass
    return False


def discover_all_dates(log_dir: Path) -> List[date]:
    """Find all dates that have trade data."""
    dates = set()
    if not log_dir.exists():
        return []

    # Extract dates from filenames
    for f in log_dir.iterdir():
        if not f.name.endswith(".jsonl"):
            continue
        # Try YYYYMMDD pattern
        m = re.search(r'(\d{4})(\d{2})(\d{2})', f.name)
        if m:
            try:
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if 2020 <= d.year <= 2030:
                    dates.add(d)
            except ValueError:
                pass

    # Also scan file contents for dates not in filenames
    for f in log_dir.iterdir():
        if not f.name.endswith(".jsonl"):
            continue
        if re.search(r'\d{8}', f.name):
            continue  # Already handled
        try:
            with open(f) as fh:
                line = fh.readline().strip()
                if line:
                    rec = json.loads(line)
                    ts = _parse_ts(rec.get("ts", rec.get("timestamp", 0)))
                    if ts > 0:
                        try:
                            dates.add(datetime.fromtimestamp(ts).date())
                        except (OSError, ValueError):
                            pass
        except (json.JSONDecodeError, IOError):
            pass

    return sorted(dates)


# ══════════════════════════════════════════════════════════════════════════════
# Trade loading
# ══════════════════════════════════════════════════════════════════════════════

def load_trades(files: Dict[str, List[Path]], target_date: Optional[date] = None
                ) -> Dict[str, List[Dict[str, Any]]]:
    """Load and normalize trades from all files, grouped by config."""
    result: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CONFIGS}

    for config, paths in files.items():
        for p in paths:
            try:
                with open(p) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            raw = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        trade = _normalize_trade(raw, config)
                        if trade is None:
                            continue

                        # Date filter on trade timestamp
                        if target_date and trade["ts"] > 0:
                            try:
                                td = datetime.fromtimestamp(trade["ts"]).date()
                                if td != target_date:
                                    continue
                            except (OSError, ValueError):
                                pass

                        result[config].append(trade)
            except IOError:
                continue

    # Sort each config's trades by timestamp
    for cfg in result:
        result[cfg].sort(key=lambda t: t["ts"])

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Metrics computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute performance metrics for a list of trades."""
    n = len(trades)
    if n == 0:
        return _empty_metrics()

    pnls_usd = [t["pnl_usd"] for t in trades]
    pnls_ticks = [t["pnl_ticks"] for t in trades]
    holds = [t["hold_s"] for t in trades]
    mfes = [t["mfe"] for t in trades]
    maes = [t["mae"] for t in trades]

    wins = [p for p in pnls_usd if p > 0]
    losses = [p for p in pnls_usd if p <= 0]

    total_pnl = sum(pnls_usd)
    wr = len(wins) / n * 100.0

    # Sharpe (annualized, assuming ~252 trading days, ~23 trades/day as rough baseline)
    # For intraday: Sharpe = mean / std * sqrt(n_trades_per_year)
    # Simpler: just compute daily-level Sharpe analog = mean / std * sqrt(252)
    # But with a single day of trades, just do trade-level ratio
    mean_pnl = _mean(pnls_usd)
    std_pnl = _std(pnls_usd)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    # Sortino
    ds = _downside_std(pnls_usd)
    sortino = mean_pnl / ds if ds > 0 else 0.0

    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0)

    # Max drawdown
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls_usd:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = cumulative - peak
        if dd < max_dd:
            max_dd = dd

    # Average hold time
    avg_hold = _mean(holds) if holds else 0.0

    # MFE / MAE
    avg_mfe = _mean(mfes) if mfes else 0.0
    avg_mae = _mean(maes) if maes else 0.0

    # Best / worst trade
    best_trade = max(pnls_usd) if pnls_usd else 0.0
    worst_trade = min(pnls_usd) if pnls_usd else 0.0

    # Trades per hour
    if n >= 2 and trades[-1]["ts"] > 0 and trades[0]["ts"] > 0:
        span_hours = (trades[-1]["ts"] - trades[0]["ts"]) / 3600.0
        trades_per_hour = n / span_hours if span_hours > 0 else float(n)
    else:
        trades_per_hour = float(n)

    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for p in pnls_usd:
        if p > 0:
            cur_win += 1
            cur_loss = 0
            max_win_streak = max(max_win_streak, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss_streak = max(max_loss_streak, cur_loss)

    # Long vs Short split
    long_trades = [t for t in trades if t["dir"] == "LONG"]
    short_trades = [t for t in trades if t["dir"] == "SHORT"]

    long_stats = _direction_stats(long_trades)
    short_stats = _direction_stats(short_trades)

    # Hourly PnL
    hourly_pnl: Dict[int, float] = defaultdict(float)
    hourly_count: Dict[int, int] = defaultdict(int)
    for t in trades:
        h = t["hour"]
        if h >= 0:
            hourly_pnl[h] += t["pnl_usd"]
            hourly_count[h] += 1

    # Exit reason breakdown
    reason_counts: Dict[str, int] = defaultdict(int)
    reason_pnl: Dict[str, float] = defaultdict(float)
    for t in trades:
        r = t["reason"] or "unknown"
        reason_counts[r] += 1
        reason_pnl[r] += t["pnl_usd"]

    return {
        "n_trades":         n,
        "wr":               wr,
        "total_pnl":        total_pnl,
        "sharpe":           sharpe,
        "sortino":          sortino,
        "pf":               pf,
        "avg_hold":         avg_hold,
        "max_dd":           max_dd,
        "trades_per_hour":  trades_per_hour,
        "best_trade":       best_trade,
        "worst_trade":      worst_trade,
        "avg_mfe":          avg_mfe,
        "avg_mae":          avg_mae,
        "max_win_streak":   max_win_streak,
        "max_loss_streak":  max_loss_streak,
        "long":             long_stats,
        "short":            short_stats,
        "hourly_pnl":       dict(hourly_pnl),
        "hourly_count":     dict(hourly_count),
        "reason_counts":    dict(reason_counts),
        "reason_pnl":       {k: round(v, 2) for k, v in reason_pnl.items()},
        "mean_pnl":         mean_pnl,
        "std_pnl":          std_pnl,
    }


def _direction_stats(trades: List[Dict]) -> Dict[str, Any]:
    if not trades:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "avg_pnl": 0.0}
    pnls = [t["pnl_usd"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    return {
        "n":       len(trades),
        "wr":      wins / len(trades) * 100.0,
        "pnl":     sum(pnls),
        "avg_pnl": _mean(pnls),
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "n_trades": 0, "wr": 0.0, "total_pnl": 0.0, "sharpe": 0.0,
        "sortino": 0.0, "pf": 0.0, "avg_hold": 0.0, "max_dd": 0.0,
        "trades_per_hour": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
        "avg_mfe": 0.0, "avg_mae": 0.0, "max_win_streak": 0, "max_loss_streak": 0,
        "long": {"n": 0, "wr": 0.0, "pnl": 0.0, "avg_pnl": 0.0},
        "short": {"n": 0, "wr": 0.0, "pnl": 0.0, "avg_pnl": 0.0},
        "hourly_pnl": {}, "hourly_count": {},
        "reason_counts": {}, "reason_pnl": {},
        "mean_pnl": 0.0, "std_pnl": 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report formatting
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_pnl(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}"

def _fmt_hold(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    elif s < 3600:
        return f"{s/60:.1f}m"
    else:
        return f"{s/3600:.1f}h"

def _fmt_pf(v: float) -> str:
    if v == float("inf"):
        return "inf"
    return f"{v:.2f}"

def _fmt_ratio(v: float) -> str:
    return f"{v:.2f}"


def format_summary_table(date_str: str,
                         config_metrics: Dict[str, Dict],
                         combined_metrics: Dict) -> str:
    """Build the main comparison table."""
    lines = []
    lines.append(f"PAPER TRADING PERFORMANCE -- {date_str}")
    lines.append("=" * 80)
    header = f"{'Config':<18} {'Trades':>6} {'WR%':>6} {'PnL($)':>10} {'Sharpe':>7} " \
             f"{'Sortino':>8} {'PF':>6} {'AvgHold':>8} {'MaxDD':>9}"
    lines.append(header)
    lines.append("-" * 80)

    for cfg in CONFIGS:
        m = config_metrics.get(cfg)
        if m is None or m["n_trades"] == 0:
            lines.append(f"{CONFIG_DISPLAY.get(cfg, cfg):<18} {'--':>6} {'--':>6} "
                         f"{'--':>10} {'--':>7} {'--':>8} {'--':>6} {'--':>8} {'--':>9}")
            continue

        lines.append(
            f"{CONFIG_DISPLAY.get(cfg, cfg):<18} {m['n_trades']:>6} {m['wr']:>5.1f}% "
            f"{_fmt_pnl(m['total_pnl']):>10} {_fmt_ratio(m['sharpe']):>7} "
            f"{_fmt_ratio(m['sortino']):>8} {_fmt_pf(m['pf']):>6} "
            f"{_fmt_hold(m['avg_hold']):>8} {_fmt_pnl(m['max_dd']):>9}"
        )

    lines.append("-" * 80)

    cm = combined_metrics
    if cm["n_trades"] > 0:
        lines.append(
            f"{'COMBINED':<18} {cm['n_trades']:>6} {cm['wr']:>5.1f}% "
            f"{_fmt_pnl(cm['total_pnl']):>10} {_fmt_ratio(cm['sharpe']):>7} "
            f"{_fmt_ratio(cm['sortino']):>8} {_fmt_pf(cm['pf']):>6} "
            f"{_fmt_hold(cm['avg_hold']):>8} {_fmt_pnl(cm['max_dd']):>9}"
        )
    else:
        lines.append(f"{'COMBINED':<18}   No trades found")

    lines.append("=" * 80)
    return "\n".join(lines)


def format_config_detail(cfg: str, trades: List[Dict], metrics: Dict) -> str:
    """Build detailed section for a single config."""
    if metrics["n_trades"] == 0:
        return f"\n--- {CONFIG_DISPLAY.get(cfg, cfg)} --- No trades\n"

    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  {CONFIG_DISPLAY.get(cfg, cfg)} -- DETAIL")
    lines.append(f"{'='*80}")

    # Summary line
    lines.append(f"  Trades: {metrics['n_trades']} | WR: {metrics['wr']:.1f}% | "
                 f"PnL: {_fmt_pnl(metrics['total_pnl'])} | "
                 f"Best: {_fmt_pnl(metrics['best_trade'])} | "
                 f"Worst: {_fmt_pnl(metrics['worst_trade'])}")
    lines.append(f"  Avg MFE: {metrics['avg_mfe']:.2f}t | "
                 f"Avg MAE: {metrics['avg_mae']:.2f}t | "
                 f"Trades/hr: {metrics['trades_per_hour']:.1f}")
    lines.append(f"  Win streak: {metrics['max_win_streak']} | "
                 f"Loss streak: {metrics['max_loss_streak']}")

    # Long vs Short
    ls = metrics["long"]
    ss = metrics["short"]
    lines.append(f"\n  Long vs Short:")
    lines.append(f"    LONG:  {ls['n']:>4} trades | WR {ls['wr']:>5.1f}% | "
                 f"PnL {_fmt_pnl(ls['pnl']):>10} | Avg {_fmt_pnl(ls['avg_pnl']):>8}")
    lines.append(f"    SHORT: {ss['n']:>4} trades | WR {ss['wr']:>5.1f}% | "
                 f"PnL {_fmt_pnl(ss['pnl']):>10} | Avg {_fmt_pnl(ss['avg_pnl']):>8}")

    # Exit reason breakdown
    if metrics["reason_counts"]:
        lines.append(f"\n  Exit Reasons:")
        for reason, count in sorted(metrics["reason_counts"].items(),
                                     key=lambda x: -x[1]):
            rpnl = metrics["reason_pnl"].get(reason, 0)
            lines.append(f"    {reason:<24} {count:>4} trades  PnL: {_fmt_pnl(rpnl):>10}")

    # Hourly PnL
    if metrics["hourly_pnl"]:
        lines.append(f"\n  Hourly Breakdown:")
        for hour in sorted(metrics["hourly_pnl"].keys()):
            hpnl = metrics["hourly_pnl"][hour]
            hcount = metrics["hourly_count"].get(hour, 0)
            bar = "+" * max(0, int(hpnl / 5)) if hpnl > 0 else "-" * max(0, int(-hpnl / 5))
            lines.append(f"    {hour:02d}:00  {hcount:>3} trades  {_fmt_pnl(hpnl):>10}  {bar}")

    # Trade-by-trade log
    lines.append(f"\n  Trade Log:")
    lines.append(f"  {'Time':<12} {'Dir':<6} {'Entry':>10} {'Exit':>10} "
                 f"{'PnL($)':>10} {'Hold':>7} {'Reason':<20}")
    lines.append(f"  {'-'*75}")
    for t in trades:
        try:
            ts_str = datetime.fromtimestamp(t["ts"]).strftime("%H:%M:%S") if t["ts"] > 0 else "??:??:??"
        except (OSError, ValueError):
            ts_str = "??:??:??"

        lines.append(
            f"  {ts_str:<12} {t['dir']:<6} {t['entry']:>10.2f} {t['exit']:>10.2f} "
            f"{_fmt_pnl(t['pnl_usd']):>10} {_fmt_hold(t['hold_s']):>7} "
            f"{t['reason']:<20}"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Discord compact format
# ══════════════════════════════════════════════════════════════════════════════

def format_discord(date_str: str,
                   config_metrics: Dict[str, Dict],
                   combined_metrics: Dict) -> str:
    """Compact report for Discord (under 2000 chars)."""
    lines = []
    lines.append(f"**PAPER TRADING -- {date_str}**")
    lines.append("```")

    # Compact table
    lines.append(f"{'Config':<15} {'#':>3} {'WR%':>5} {'PnL':>8} {'SR':>5} {'PF':>5}")
    lines.append("-" * 45)

    for cfg in CONFIGS:
        m = config_metrics.get(cfg)
        if m is None or m["n_trades"] == 0:
            continue
        name = CONFIG_DISPLAY.get(cfg, cfg)[:14]
        lines.append(
            f"{name:<15} {m['n_trades']:>3} {m['wr']:>4.0f}% "
            f"{_fmt_pnl(m['total_pnl']):>8} {m['sharpe']:>5.2f} "
            f"{_fmt_pf(m['pf']):>5}"
        )

    cm = combined_metrics
    if cm["n_trades"] > 0:
        lines.append("-" * 45)
        lines.append(
            f"{'TOTAL':<15} {cm['n_trades']:>3} {cm['wr']:>4.0f}% "
            f"{_fmt_pnl(cm['total_pnl']):>8} {cm['sharpe']:>5.2f} "
            f"{_fmt_pf(cm['pf']):>5}"
        )

    lines.append("```")

    # Best/worst per config (compact)
    notes = []
    for cfg in CONFIGS:
        m = config_metrics.get(cfg)
        if m and m["n_trades"] > 0:
            name = CONFIG_DISPLAY.get(cfg, cfg)
            notes.append(f"{name}: best {_fmt_pnl(m['best_trade'])}, "
                         f"worst {_fmt_pnl(m['worst_trade'])}, "
                         f"maxDD {_fmt_pnl(m['max_dd'])}")

    if notes:
        lines.append("\n".join(notes))

    output = "\n".join(lines)

    # Ensure under 2000 chars for Discord
    if len(output) > 1950:
        output = output[:1947] + "..."

    return output


# ══════════════════════════════════════════════════════════════════════════════
# Main report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(log_dir: Path, target_date: date,
                    discord_mode: bool = False) -> str:
    """Generate a full performance report for a given date."""
    files = discover_trade_files(log_dir, target_date)
    trades = load_trades(files, target_date)

    config_metrics: Dict[str, Dict] = {}
    for cfg in CONFIGS:
        config_metrics[cfg] = compute_metrics(trades[cfg])

    # Combined metrics
    all_trades = []
    for cfg in CONFIGS:
        all_trades.extend(trades[cfg])
    all_trades.sort(key=lambda t: t["ts"])
    combined = compute_metrics(all_trades)

    date_str = target_date.strftime("%Y-%m-%d")

    if discord_mode:
        return format_discord(date_str, config_metrics, combined)

    # Full report
    parts = []
    parts.append(format_summary_table(date_str, config_metrics, combined))

    for cfg in CONFIGS:
        if trades[cfg]:
            parts.append(format_config_detail(cfg, trades[cfg], config_metrics[cfg]))

    if not any(trades[cfg] for cfg in CONFIGS):
        parts.append(f"\nNo trade data found for {date_str} in {log_dir}")

    return "\n".join(parts)


def generate_all_dates_report(log_dir: Path, discord_mode: bool = False) -> str:
    """Generate reports for all available dates."""
    dates = discover_all_dates(log_dir)
    if not dates:
        return f"No trade data found in {log_dir}"

    parts = []
    for d in dates:
        parts.append(generate_report(log_dir, d, discord_mode))
        if not discord_mode:
            parts.append("\n" + "=" * 80 + "\n")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Programmatic API — for use by other scripts
# ══════════════════════════════════════════════════════════════════════════════

def get_report_data(log_dir: Path, target_date: date
                    ) -> Dict[str, Any]:
    """
    Return structured report data (for programmatic consumption).

    Returns:
        {
            "date": "2026-05-04",
            "configs": {config_name: {"trades": [...], "metrics": {...}}},
            "combined": {"trades": [...], "metrics": {...}},
        }
    """
    files = discover_trade_files(log_dir, target_date)
    trades = load_trades(files, target_date)

    config_data = {}
    all_trades = []
    for cfg in CONFIGS:
        m = compute_metrics(trades[cfg])
        config_data[cfg] = {"trades": trades[cfg], "metrics": m}
        all_trades.extend(trades[cfg])

    all_trades.sort(key=lambda t: t["ts"])
    combined = compute_metrics(all_trades)

    return {
        "date": target_date.isoformat(),
        "configs": config_data,
        "combined": {"trades": all_trades, "metrics": combined},
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Paper Trading Performance Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python performance_report.py                     # today
  python performance_report.py --date 2026-05-04   # specific date
  python performance_report.py --all               # all available dates
  python performance_report.py --discord            # Discord-friendly output
  python performance_report.py --json               # JSON output for piping
        """,
    )
    parser.add_argument("--date", "-d", type=str, default=None,
                        help="Target date (YYYY-MM-DD). Default: today")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Report for all available dates")
    parser.add_argument("--discord", action="store_true",
                        help="Compact Discord-friendly output (<2000 chars)")
    parser.add_argument("--json", action="store_true",
                        help="JSON output for programmatic use")
    parser.add_argument("--logdir", "-l", type=str, default=None,
                        help=f"Log directory (default: {DEFAULT_LOG_DIR})")

    args = parser.parse_args()

    log_dir = Path(args.logdir) if args.logdir else DEFAULT_LOG_DIR

    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}", file=sys.stderr)
        print(f"Creating it: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        if args.json:
            dates = discover_all_dates(log_dir)
            results = []
            for d in dates:
                results.append(get_report_data(log_dir, d))
            print(json.dumps(results, indent=2, default=str))
        else:
            print(generate_all_dates_report(log_dir, discord_mode=args.discord))
    else:
        if args.date:
            try:
                target = datetime.strptime(args.date, "%Y-%m-%d").date()
            except ValueError:
                print(f"Invalid date format: {args.date}. Use YYYY-MM-DD.", file=sys.stderr)
                sys.exit(1)
        else:
            target = date.today()

        if args.json:
            data = get_report_data(log_dir, target)
            print(json.dumps(data, indent=2, default=str))
        else:
            print(generate_report(log_dir, target, discord_mode=args.discord))


if __name__ == "__main__":
    main()
