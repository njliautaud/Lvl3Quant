#!/usr/bin/env python3
"""
Monday EOD Analysis — Paper Trader Results vs Simulation Baseline

Reads a CSV of paper trades from Razer's live trading session and produces:
- Performance summary (ticks, dollars, Sharpe, Sortino, WR, PF)
- Breakdowns by side, confidence tier, and hour
- Comparison vs 48-day OOT simulation baseline
- Clean Discord-ready text summary
- Detailed JSON output

Usage:
    python monday_eod_analysis.py --trades-csv /path/to/trades.csv
    python monday_eod_analysis.py --trades-csv trades.csv --output results.json
    python monday_eod_analysis.py --trades-csv trades.csv --discord-only

Expected CSV columns (all optional except pnl_ticks):
    timestamp, side, entry_price, exit_price, pnl_ticks, fill_type,
    confidence_tier, meta_score, ofi_agreement, hold_seconds, attempted
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376  # $4.70 / $12.50

# 48-day OOT simulation baseline
BASELINE = {
    "avg_ticks_per_trade": 0.46,
    "win_rate": 0.55,
    "profit_factor": 1.92,
    "trades_per_day": 527,
}

DEVIATION_THRESHOLD = 0.20  # flag if >20% off baseline

# Annualization: assume ~6.5 trading hours/day, 252 days/year
TRADES_PER_YEAR_APPROX = 527 * 252  # ~132,804


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    variance = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(variance)


def _sharpe(returns: List[float]) -> float:
    """Annualized Sharpe from per-trade tick returns (assumes zero risk-free)."""
    if len(returns) < 2:
        return 0.0
    m = _mean(returns)
    s = _std(returns)
    if s == 0:
        return 0.0
    # sqrt(n_trades_per_year) annualizes per-trade Sharpe
    return (m / s) * math.sqrt(TRADES_PER_YEAR_APPROX)


def _sortino(returns: List[float]) -> float:
    """Annualized Sortino (downside deviation below 0)."""
    if len(returns) < 2:
        return 0.0
    m = _mean(returns)
    neg = [r for r in returns if r < 0]
    if not neg:
        return float("inf")
    downside_std = math.sqrt(sum(r ** 2 for r in neg) / len(neg))
    if downside_std == 0:
        return 0.0
    return (m / downside_std) * math.sqrt(TRADES_PER_YEAR_APPROX)


def _profit_factor(returns: List[float]) -> float:
    gross_win = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _win_rate(returns: List[float]) -> float:
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return wins / len(returns)


def _flag(actual: float, baseline_val: float, label: str) -> Optional[str]:
    """Return a flag string if deviation exceeds threshold."""
    if baseline_val == 0:
        return None
    dev = (actual - baseline_val) / abs(baseline_val)
    if abs(dev) > DEVIATION_THRESHOLD:
        direction = "above" if dev > 0 else "below"
        pct = abs(dev) * 100
        return f"{label}: {actual:.2f} vs baseline {baseline_val:.2f} ({pct:.0f}% {direction})"
    return None


# ── CSV Loading ────────────────────────────────────────────────────────────────

def load_trades(csv_path: Path) -> Tuple[List[Dict], List[str]]:
    """Load trades from CSV. Returns (trades_list, warnings)."""
    warnings: List[str] = []

    if not csv_path.exists():
        return [], [f"File not found: {csv_path}"]

    trades = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [], ["CSV has no header row"]

        required = {"pnl_ticks"}
        available = set(reader.fieldnames)
        missing = required - available
        if missing:
            return [], [f"CSV missing required column(s): {missing}"]

        for i, row in enumerate(reader):
            try:
                trade: Dict[str, Any] = {}

                # Required
                trade["pnl_ticks"] = float(row["pnl_ticks"])

                # Optional with defaults
                trade["side"] = row.get("side", "unknown").lower().strip()
                trade["fill_type"] = row.get("fill_type", "unknown").lower().strip()
                trade["confidence_tier"] = row.get("confidence_tier", "unknown").strip()
                trade["attempted"] = str(row.get("attempted", "1")).strip() in ("1", "true", "True", "yes")
                trade["hold_seconds"] = float(row.get("hold_seconds", 0) or 0)

                # Meta/OFI scores
                try:
                    trade["meta_score"] = float(row.get("meta_score") or 0)
                except (ValueError, TypeError):
                    trade["meta_score"] = None

                try:
                    trade["ofi_agreement"] = str(row.get("ofi_agreement", "")).strip().lower() in ("1", "true", "yes")
                except (ValueError, TypeError):
                    trade["ofi_agreement"] = None

                # Timestamp — try multiple formats
                ts_raw = row.get("timestamp", "")
                trade["timestamp"] = None
                trade["hour"] = None
                if ts_raw:
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
                                "%H:%M:%S", "%m/%d/%Y %H:%M:%S"):
                        try:
                            dt = datetime.strptime(ts_raw, fmt)
                            trade["timestamp"] = dt.isoformat()
                            trade["hour"] = dt.hour
                            break
                        except ValueError:
                            continue

                # Prices
                for col in ("entry_price", "exit_price"):
                    try:
                        trade[col] = float(row.get(col) or 0)
                    except (ValueError, TypeError):
                        trade[col] = None

                trades.append(trade)

            except (ValueError, KeyError) as e:
                warnings.append(f"Row {i+1} skipped: {e}")

    if not trades:
        warnings.append("No valid trades loaded from CSV")

    return trades, warnings


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(trades: List[Dict]) -> Dict[str, Any]:
    """Run full analysis. Returns results dict."""

    # Separate attempted vs filled
    attempted = [t for t in trades if t.get("attempted", True)]
    filled = [t for t in trades if not t.get("attempted", False) or t.get("fill_type", "") != "rejected"]

    # For trades that actually executed, use all loaded trades unless 'attempted' field distinguishes
    # If no 'attempted' column present in CSV, treat all trades as filled
    has_attempted_col = any("attempted" in t for t in trades)
    if has_attempted_col:
        executed = filled
        attempted_count = len(attempted)
    else:
        executed = trades
        attempted_count = len(trades)

    returns = [t["pnl_ticks"] for t in executed]

    # Time span
    hours_active = None
    timestamps = [t["hour"] for t in executed if t.get("hour") is not None]
    if timestamps:
        # Estimate from first/last hour seen
        min_h, max_h = min(timestamps), max(timestamps)
        hours_active = max(max_h - min_h + 1, 1)

    # ── Core metrics ──
    n = len(executed)
    net_ticks = sum(returns)
    net_dollars = net_ticks * ES_TICK_VALUE
    avg_ticks = _mean(returns)
    wr = _win_rate(returns)
    pf = _profit_factor(returns)
    sharpe = _sharpe(returns)
    sortino = _sortino(returns)
    fill_rate = n / attempted_count if attempted_count > 0 else 1.0
    trades_per_hour = n / hours_active if hours_active else None

    # ── Breakdown by side ──
    sides: Dict[str, List[float]] = defaultdict(list)
    for t in executed:
        sides[t["side"]].append(t["pnl_ticks"])

    side_stats = {}
    for side, rets in sides.items():
        side_stats[side] = {
            "count": len(rets),
            "net_ticks": sum(rets),
            "avg_ticks": _mean(rets),
            "win_rate": _win_rate(rets),
            "profit_factor": _profit_factor(rets),
        }

    # ── Breakdown by confidence tier ──
    tiers: Dict[str, List[float]] = defaultdict(list)
    for t in executed:
        tiers[t["confidence_tier"]].append(t["pnl_ticks"])

    tier_stats = {}
    for tier, rets in tiers.items():
        tier_stats[tier] = {
            "count": len(rets),
            "avg_ticks": _mean(rets),
            "win_rate": _win_rate(rets),
            "profit_factor": _profit_factor(rets),
        }

    # ── Breakdown by hour ──
    hours: Dict[int, List[float]] = defaultdict(list)
    for t in executed:
        if t.get("hour") is not None:
            hours[t["hour"]].append(t["pnl_ticks"])

    hour_stats = {}
    for hour, rets in sorted(hours.items()):
        label = f"{hour:02d}:00"
        hour_stats[label] = {
            "count": len(rets),
            "avg_ticks": _mean(rets),
            "net_ticks": sum(rets),
            "win_rate": _win_rate(rets),
        }

    # ── Filter pass rates ──
    # meta_score > 0 = passed meta gate
    meta_passed = sum(1 for t in executed if t.get("meta_score") is not None and t["meta_score"] > 0)
    ofi_passed = sum(1 for t in executed if t.get("ofi_agreement") is True)
    meta_available = sum(1 for t in executed if t.get("meta_score") is not None)
    ofi_available = sum(1 for t in executed if t.get("ofi_agreement") is not None)

    filter_rates = {}
    if meta_available > 0:
        filter_rates["meta_gate_pass_rate"] = meta_passed / meta_available
    if ofi_available > 0:
        filter_rates["ofi_gate_pass_rate"] = ofi_passed / ofi_available

    # ── Baseline comparison ──
    flags = []
    f = _flag(avg_ticks, BASELINE["avg_ticks_per_trade"], "Avg ticks/trade")
    if f:
        flags.append(f)
    f = _flag(wr, BASELINE["win_rate"], "Win rate")
    if f:
        flags.append(f)
    f = _flag(pf, BASELINE["profit_factor"], "Profit factor")
    if f:
        flags.append(f)
    f = _flag(n, BASELINE["trades_per_day"], "Trade count")
    if f:
        flags.append(f)

    return {
        "summary": {
            "total_trades": n,
            "attempted_count": attempted_count,
            "fill_rate": fill_rate,
            "trades_per_hour": trades_per_hour,
            "hours_active": hours_active,
            "net_ticks": net_ticks,
            "net_dollars": net_dollars,
            "avg_ticks_per_trade": avg_ticks,
            "win_rate": wr,
            "profit_factor": pf,
            "sharpe_annualized": sharpe,
            "sortino_annualized": sortino,
        },
        "by_side": side_stats,
        "by_confidence_tier": tier_stats,
        "by_hour": hour_stats,
        "filter_rates": filter_rates,
        "baseline_comparison": {
            "baseline": BASELINE,
            "flags": flags,
        },
    }


# ── Reporting ──────────────────────────────────────────────────────────────────

def format_discord(results: Dict, warnings: List[str]) -> str:
    """Format a plain-English Discord summary (≤15 lines)."""
    s = results["summary"]
    bc = results["baseline_comparison"]

    lines = []
    lines.append("**Monday Paper Trading Results**")

    n = s["total_trades"]
    fill_pct = s["fill_rate"] * 100
    net_d = s["net_dollars"]
    net_t = s["net_ticks"]
    avg_t = s["avg_ticks_per_trade"]
    wr_pct = s["win_rate"] * 100
    pf = s["profit_factor"]
    sharpe = s["sharpe_annualized"]
    sortino = s["sortino_annualized"]

    lines.append(
        f"{n} trades executed ({fill_rate_str(s)}) | "
        f"Net: {net_t:+.1f} ticks (${net_d:+.0f})"
    )
    lines.append(
        f"Avg/trade: {avg_t:+.3f} ticks | "
        f"WR: {wr_pct:.1f}% | PF: {pf:.2f}"
    )
    lines.append(
        f"Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f}"
    )

    # Side breakdown
    by_side = results["by_side"]
    if by_side:
        parts = []
        for side in ("short", "long"):
            if side in by_side:
                st = by_side[side]
                parts.append(
                    f"{side.capitalize()}: {st['count']} trades, "
                    f"{st['avg_ticks']:+.3f}t avg, "
                    f"WR {st['win_rate']*100:.0f}%"
                )
        if parts:
            lines.append(" | ".join(parts))

    # Confidence tiers
    tier_stats = results["by_confidence_tier"]
    if tier_stats and "unknown" not in tier_stats:
        tier_parts = []
        for tier, ts in sorted(tier_stats.items()):
            tier_parts.append(f"{tier}: {ts['avg_ticks']:+.3f}t ({ts['count']})")
        lines.append("Tiers — " + ", ".join(tier_parts))

    # Filter rates
    fr = results["filter_rates"]
    if fr:
        fr_parts = [f"{k.replace('_gate_pass_rate','').upper()} gate: {v*100:.0f}%" for k, v in fr.items()]
        lines.append("Filters — " + ", ".join(fr_parts))

    # Baseline comparison
    flags = bc["flags"]
    if not flags:
        lines.append("vs Baseline: all metrics within 20% of sim — on track")
    else:
        lines.append(f"vs Baseline: {len(flags)} deviation(s) flagged")
        for flag in flags[:3]:  # cap at 3 to stay ≤15 lines
            lines.append(f"  ⚠ {flag}")

    if warnings:
        lines.append(f"Notes: {warnings[0]}" + (f" (+{len(warnings)-1} more)" if len(warnings) > 1 else ""))

    return "\n".join(lines)


def fill_rate_str(s: Dict) -> str:
    attempted = s["attempted_count"]
    filled = s["total_trades"]
    if attempted == filled:
        return "100% fill"
    pct = filled / attempted * 100 if attempted else 0
    return f"{pct:.0f}% fill"


def format_text_report(results: Dict, warnings: List[str]) -> str:
    """Detailed plain-text report."""
    s = results["summary"]
    lines = []
    lines.append("=" * 60)
    lines.append("MONDAY EOD ANALYSIS — PAPER TRADER RESULTS")
    lines.append("=" * 60)
    lines.append("")

    lines.append("OVERVIEW")
    lines.append(f"  Trades executed : {s['total_trades']}")
    lines.append(f"  Attempted       : {s['attempted_count']}")
    lines.append(f"  Fill rate       : {s['fill_rate']*100:.1f}%")
    if s["trades_per_hour"]:
        lines.append(f"  Trades/hour     : {s['trades_per_hour']:.1f}")
    lines.append(f"  Hours active    : {s['hours_active']}")
    lines.append("")

    lines.append("PERFORMANCE")
    lines.append(f"  Net P&L         : {s['net_ticks']:+.2f} ticks (${s['net_dollars']:+.2f})")
    lines.append(f"  Avg/trade       : {s['avg_ticks_per_trade']:+.4f} ticks")
    lines.append(f"  Win rate        : {s['win_rate']*100:.1f}%")
    lines.append(f"  Profit factor   : {s['profit_factor']:.3f}")
    lines.append(f"  Sharpe (ann.)   : {s['sharpe_annualized']:.3f}")
    lines.append(f"  Sortino (ann.)  : {s['sortino_annualized']:.3f}")
    lines.append("")

    lines.append("BY SIDE")
    for side, st in sorted(results["by_side"].items()):
        lines.append(f"  {side.upper():8s}: {st['count']:4d} trades | "
                     f"avg {st['avg_ticks']:+.4f}t | "
                     f"WR {st['win_rate']*100:.1f}% | "
                     f"PF {st['profit_factor']:.2f} | "
                     f"net {st['net_ticks']:+.2f}t")
    lines.append("")

    lines.append("BY CONFIDENCE TIER")
    for tier, ts in sorted(results["by_confidence_tier"].items()):
        lines.append(f"  {tier:15s}: {ts['count']:4d} trades | "
                     f"avg {ts['avg_ticks']:+.4f}t | "
                     f"WR {ts['win_rate']*100:.1f}% | "
                     f"PF {ts['profit_factor']:.2f}")
    lines.append("")

    lines.append("BY HOUR")
    for hour_label, hs in results["by_hour"].items():
        lines.append(f"  {hour_label}: {hs['count']:3d} trades | "
                     f"avg {hs['avg_ticks']:+.4f}t | "
                     f"net {hs['net_ticks']:+.2f}t | "
                     f"WR {hs['win_rate']*100:.1f}%")
    lines.append("")

    fr = results["filter_rates"]
    if fr:
        lines.append("FILTER PASS RATES")
        for k, v in fr.items():
            lines.append(f"  {k}: {v*100:.1f}%")
        lines.append("")

    bc = results["baseline_comparison"]
    lines.append("BASELINE COMPARISON (48-day OOT sim)")
    lines.append(f"  Avg ticks/trade : actual {s['avg_ticks_per_trade']:+.4f} vs baseline {bc['baseline']['avg_ticks_per_trade']:.4f}")
    lines.append(f"  Win rate        : actual {s['win_rate']*100:.1f}% vs baseline {bc['baseline']['win_rate']*100:.1f}%")
    lines.append(f"  Profit factor   : actual {s['profit_factor']:.3f} vs baseline {bc['baseline']['profit_factor']:.3f}")
    lines.append(f"  Trade count     : actual {s['total_trades']} vs baseline {bc['baseline']['trades_per_day']}")
    if bc["flags"]:
        lines.append(f"  FLAGGED ({len(bc['flags'])} items):")
        for flag in bc["flags"]:
            lines.append(f"    ! {flag}")
    else:
        lines.append("  All within 20% of baseline.")
    lines.append("")

    if warnings:
        lines.append("WARNINGS")
        for w in warnings:
            lines.append(f"  {w}")
        lines.append("")

    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monday EOD analysis of paper trader results"
    )
    parser.add_argument(
        "--trades-csv", required=True,
        help="Path to trades CSV file from Razer paper trader"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for JSON output (default: same dir as CSV with .json extension)"
    )
    parser.add_argument(
        "--discord-only", action="store_true",
        help="Print only the Discord-format summary"
    )
    args = parser.parse_args()

    csv_path = Path(args.trades_csv)
    trades, warnings = load_trades(csv_path)

    if not trades:
        print("ERROR: No valid trades found.")
        for w in warnings:
            print(f"  {w}")
        sys.exit(1)

    results = analyze(trades)

    # JSON output
    output_path = Path(args.output) if args.output else csv_path.with_suffix(".json")
    output_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "warnings": warnings,
        "results": results,
    }
    output_path.write_text(json.dumps(output_data, indent=2, default=str))

    if args.discord_only:
        print(format_discord(results, warnings))
    else:
        print(format_text_report(results, warnings))
        print()
        print("--- DISCORD SUMMARY ---")
        print(format_discord(results, warnings))
        print()
        print(f"JSON saved to: {output_path}")


if __name__ == "__main__":
    main()
