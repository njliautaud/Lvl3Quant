#!/usr/bin/env python3
"""
Wheel Strategy Optimization Research — Comprehensive Parameter Sweep.

Backtests the wheel strategy across a grid of parameters to find optimal
configurations for both "balanced" and "scalp" wheel profiles.

Sweep dimensions:
  1. CSP delta: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
  2. CC delta: [0.20, 0.25, 0.30, 0.35, 0.40]
  3. DTE: [7, 14, 21, 30, 45]
  4. Early close: [0.50, 0.65, 0.75, None]
  5. Underlyings: categorized by IV regime (low/med/high)

Uses the existing WheelBacktester with FMP data layer for PIT backtesting.

Output: JSON results + CSV summary with ranked configurations.

Usage:
    python wheel_optimization_sweep.py --months 12 --tickers AAPL,MSFT,AMD
    python wheel_optimization_sweep.py --months 12 --universe balanced
    python wheel_optimization_sweep.py --months 12 --universe all --workers 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Path setup
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/jupiter/teleclaude-main")
sys.path.insert(0, str(REPO_ROOT))

from trading_agents.wheel_strategy.backtester import WheelBacktester

# Try to use FMP data layer for PIT data
try:
    from trading_agents.wheel_strategy.fmp_yf_adapter import (
        patch_backtester_to_use_fmp,
        download_prices,
    )
    HAS_FMP = True
except ImportError:
    HAS_FMP = False

try:
    from trading_agents.wheel_strategy.timing_aware_wheel import (
        TimingGates,
        should_enter_csp,
    )
    from trading_agents.wheel_strategy.fmp_data_layer import FMPDataLayer
    HAS_TIMING = True
except ImportError:
    HAS_TIMING = False

OUT_DIR = SCRIPT_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Universe definitions ─────────────────────────────────────────────────────

# Categorized by typical IV regime and suitability for wheeling
UNIVERSE = {
    "low_iv_stable": {
        "description": "Blue chips, low IV, stable price action — best for conservative wheel",
        "tickers": ["AAPL", "MSFT", "JNJ", "PG", "KO", "PEP", "WMT", "V", "MA", "HD"],
    },
    "medium_iv_growth": {
        "description": "Growth names with moderate IV — balanced risk/reward",
        "tickers": ["AMD", "CRM", "GOOGL", "AMZN", "NVDA", "ADBE", "QCOM", "TXN"],
    },
    "high_iv_premium": {
        "description": "High IV names with rich premiums — aggressive wheel candidates",
        "tickers": ["TSLA", "PLTR", "HOOD", "SOFI", "COIN", "MARA", "NIO", "RIVN"],
    },
    "financials": {
        "description": "Banks and fintech — moderate IV, dividend support",
        "tickers": ["BAC", "JPM", "C", "WFC", "GS", "SCHW"],
    },
    "airlines_energy": {
        "description": "Cyclicals with elevated IV",
        "tickers": ["DAL", "AAL", "XOM", "CVX", "COP"],
    },
    "etfs": {
        "description": "ETFs — liquid, no earnings risk, lower IV",
        "tickers": ["SPY", "QQQ", "IWM", "XLF", "XLE"],
    },
}


def get_universe(name: str) -> list[str]:
    """Get ticker list by universe name."""
    if name == "all":
        tickers = []
        for group in UNIVERSE.values():
            tickers.extend(group["tickers"])
        return sorted(set(tickers))
    elif name == "balanced":
        # Mix of stable + moderate IV for balanced wheel
        return (
            UNIVERSE["low_iv_stable"]["tickers"][:5]
            + UNIVERSE["medium_iv_growth"]["tickers"][:3]
            + UNIVERSE["financials"]["tickers"][:3]
        )
    elif name == "scalp":
        # High IV names for aggressive premium capture
        return (
            UNIVERSE["high_iv_premium"]["tickers"]
            + UNIVERSE["airlines_energy"]["tickers"][:3]
        )
    elif name in UNIVERSE:
        return UNIVERSE[name]["tickers"]
    else:
        return name.split(",")


# ── Sweep parameter grid ─────────────────────────────────────────────────────

SWEEP_GRID = {
    "csp_delta": [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "cc_delta": [0.20, 0.25, 0.30, 0.35, 0.40],
    "target_dte": [7, 14, 21, 30, 45],
    "early_close_pct": [0.50, 0.65, 0.75, 1.0],  # 1.0 = no early close
}

# Reduced grid for quick sweeps
QUICK_GRID = {
    "csp_delta": [0.20, 0.30, 0.40],
    "cc_delta": [0.25, 0.35],
    "target_dte": [7, 14, 30],
    "early_close_pct": [0.50, 0.75],
}


def compute_metrics(equity_curve: list[float]) -> dict[str, float]:
    """Compute risk-adjusted metrics from an equity curve."""
    if not equity_curve or len(equity_curve) < 2:
        return {
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd_pct": 0.0,
            "daily_wr_pct": 0.0,
            "calmar": 0.0,
        }

    arr = np.array(equity_curve, dtype=float)
    total_ret = (arr[-1] / arr[0]) - 1.0
    n_days = len(arr) - 1
    years = n_days / 252.0

    # CAGR
    if years > 0 and arr[-1] > 0 and arr[0] > 0:
        cagr = (arr[-1] / arr[0]) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    # Daily returns
    daily_rets = np.diff(arr) / arr[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]

    if len(daily_rets) < 2:
        return {
            "total_return_pct": total_ret * 100,
            "cagr_pct": cagr * 100,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd_pct": 0.0,
            "daily_wr_pct": 0.0,
            "calmar": 0.0,
        }

    mean_ret = np.mean(daily_rets)
    std_ret = np.std(daily_rets, ddof=1)

    # Sharpe (annualized)
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

    # Sortino (downside deviation)
    down_rets = daily_rets[daily_rets < 0]
    dd_std = np.std(down_rets, ddof=1) if len(down_rets) > 1 else std_ret
    sortino = (mean_ret / dd_std * np.sqrt(252)) if dd_std > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    max_dd = np.min(drawdown) * 100

    # Win rate
    wr = np.mean(daily_rets > 0) * 100

    # Calmar
    calmar = (cagr / abs(max_dd / 100)) if max_dd != 0 else 0.0

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 2),
        "daily_wr_pct": round(wr, 1),
        "calmar": round(calmar, 3),
    }


def run_single_backtest(
    ticker: str,
    months: int,
    csp_delta: float,
    cc_delta: float,
    target_dte: int,
    early_close_pct: float,
    capital: float = 100_000,
) -> dict[str, Any] | None:
    """Run a single backtest configuration and return results."""
    try:
        bt = WheelBacktester(
            ticker=ticker,
            capital=capital,
            csp_delta=csp_delta,
            cc_delta=cc_delta,
            target_dte=target_dte,
            sizing_method="min_one",
            max_position_pct=0.30,
            commission_per_contract=1.00,
            bid_ask_haircut=0.10,
        )

        # Patch to use FMP if available
        if HAS_FMP:
            patch_backtester_to_use_fmp(bt)

        results = bt.run(months=months)

        if not results or not bt.equity_curve:
            return None

        metrics = compute_metrics(bt.equity_curve)

        return {
            "ticker": ticker,
            "csp_delta": csp_delta,
            "cc_delta": cc_delta,
            "target_dte": target_dte,
            "early_close_pct": early_close_pct,
            "trades": len(bt.trades),
            "assignments": bt.assignments,
            "calls_away": bt.calls_away,
            "expirations_otm": bt.expirations_otm,
            "rolls": bt.rolls,
            "premium_collected": round(bt.premium_collected, 2),
            **metrics,
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "csp_delta": csp_delta,
            "cc_delta": cc_delta,
            "target_dte": target_dte,
            "early_close_pct": early_close_pct,
            "error": str(e)[:200],
        }


def run_sweep(
    tickers: list[str],
    months: int = 12,
    grid: dict | None = None,
    workers: int = 1,
    capital: float = 100_000,
) -> list[dict[str, Any]]:
    """Run the full parameter sweep across tickers and grid."""
    if grid is None:
        grid = QUICK_GRID

    # Build all tasks
    tasks = []
    for ticker in tickers:
        for csp_d in grid["csp_delta"]:
            for cc_d in grid["cc_delta"]:
                for dte in grid["target_dte"]:
                    for ec in grid["early_close_pct"]:
                        tasks.append((ticker, months, csp_d, cc_d, dte, ec, capital))

    print(f"Running {len(tasks)} backtest configurations across {len(tickers)} tickers...")
    print(f"Grid: {len(grid['csp_delta'])} CSP deltas x {len(grid['cc_delta'])} CC deltas "
          f"x {len(grid['target_dte'])} DTEs x {len(grid['early_close_pct'])} early-close")

    results = []
    start_time = time.time()

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_single_backtest, *t): t for t in tasks
            }
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    results.append(result)
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {i+1}/{len(tasks)} done ({elapsed:.0f}s)")
    else:
        for i, task in enumerate(tasks):
            result = run_single_backtest(*task)
            if result:
                results.append(result)
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  {i+1}/{len(tasks)} done ({elapsed:.0f}s)")

    elapsed = time.time() - start_time
    print(f"\nCompleted {len(results)} backtests in {elapsed:.1f}s")
    return results


def analyze_results(results: list[dict]) -> dict[str, Any]:
    """Analyze sweep results and find optimal configurations."""
    # Filter out errors
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if not valid:
        return {"error": "No valid results", "errors": errors}

    # Sort by Sharpe ratio (primary), then by CAGR (secondary)
    ranked = sorted(valid, key=lambda x: (x.get("sharpe", 0), x.get("cagr_pct", 0)), reverse=True)

    # Best overall config
    best = ranked[0]

    # Best per dimension analysis
    analysis = {}

    # Best CSP delta
    delta_perf = {}
    for r in valid:
        d = r["csp_delta"]
        if d not in delta_perf:
            delta_perf[d] = []
        delta_perf[d].append(r.get("sharpe", 0))
    analysis["csp_delta_avg_sharpe"] = {
        str(d): round(np.mean(v), 3) for d, v in sorted(delta_perf.items())
    }

    # Best CC delta
    cc_perf = {}
    for r in valid:
        d = r["cc_delta"]
        if d not in cc_perf:
            cc_perf[d] = []
        cc_perf[d].append(r.get("sharpe", 0))
    analysis["cc_delta_avg_sharpe"] = {
        str(d): round(np.mean(v), 3) for d, v in sorted(cc_perf.items())
    }

    # Best DTE
    dte_perf = {}
    for r in valid:
        d = r["target_dte"]
        if d not in dte_perf:
            dte_perf[d] = []
        dte_perf[d].append(r.get("sharpe", 0))
    analysis["dte_avg_sharpe"] = {
        str(d): round(np.mean(v), 3) for d, v in sorted(dte_perf.items())
    }

    # Best early close
    ec_perf = {}
    for r in valid:
        d = r["early_close_pct"]
        if d not in ec_perf:
            ec_perf[d] = []
        ec_perf[d].append(r.get("sharpe", 0))
    analysis["early_close_avg_sharpe"] = {
        str(d): round(np.mean(v), 3) for d, v in sorted(ec_perf.items())
    }

    # Best per ticker
    ticker_best = {}
    for r in valid:
        t = r["ticker"]
        if t not in ticker_best or r.get("sharpe", 0) > ticker_best[t].get("sharpe", 0):
            ticker_best[t] = r
    analysis["best_per_ticker"] = ticker_best

    # Aggregate portfolio metrics (top 10 configs)
    top10 = ranked[:10]

    return {
        "timestamp": datetime.now().isoformat(),
        "total_configs": len(results),
        "valid_configs": len(valid),
        "error_count": len(errors),
        "best_config": best,
        "top_10": top10,
        "dimension_analysis": analysis,
        "errors_sample": errors[:5] if errors else [],
    }


def format_summary(analysis: dict) -> str:
    """Format analysis results as a readable summary."""
    lines = []
    lines.append("=" * 70)
    lines.append("WHEEL STRATEGY OPTIMIZATION — SWEEP RESULTS")
    lines.append("=" * 70)

    if "error" in analysis:
        lines.append(f"ERROR: {analysis['error']}")
        return "\n".join(lines)

    best = analysis["best_config"]
    lines.append(f"\nTotal configs tested: {analysis['total_configs']}")
    lines.append(f"Valid results: {analysis['valid_configs']}")
    lines.append(f"Errors: {analysis['error_count']}")

    lines.append(f"\n{'='*70}")
    lines.append("BEST OVERALL CONFIGURATION")
    lines.append(f"{'='*70}")
    lines.append(f"  Ticker:       {best['ticker']}")
    lines.append(f"  CSP Delta:    {best['csp_delta']}")
    lines.append(f"  CC Delta:     {best['cc_delta']}")
    lines.append(f"  DTE:          {best['target_dte']}")
    lines.append(f"  Early Close:  {best['early_close_pct']}")
    lines.append(f"  ---")
    lines.append(f"  Sharpe:       {best.get('sharpe', 0):.3f}")
    lines.append(f"  Sortino:      {best.get('sortino', 0):.3f}")
    lines.append(f"  CAGR:         {best.get('cagr_pct', 0):.2f}%")
    lines.append(f"  Max DD:       {best.get('max_dd_pct', 0):.2f}%")
    lines.append(f"  Win Rate:     {best.get('daily_wr_pct', 0):.1f}%")
    lines.append(f"  Total Return: {best.get('total_return_pct', 0):.2f}%")
    lines.append(f"  Trades:       {best.get('trades', 0)}")

    dim = analysis.get("dimension_analysis", {})

    lines.append(f"\n{'='*70}")
    lines.append("OPTIMAL PARAMETERS BY DIMENSION (avg Sharpe across all configs)")
    lines.append(f"{'='*70}")

    for key, label in [
        ("csp_delta_avg_sharpe", "CSP Delta"),
        ("cc_delta_avg_sharpe", "CC Delta"),
        ("dte_avg_sharpe", "DTE"),
        ("early_close_avg_sharpe", "Early Close %"),
    ]:
        if key in dim:
            lines.append(f"\n  {label}:")
            for param, sharpe in dim[key].items():
                bar = "#" * max(0, int(sharpe * 10))
                lines.append(f"    {param:>6s}: Sharpe {sharpe:+.3f}  {bar}")

    # Best per ticker
    if "best_per_ticker" in dim:
        lines.append(f"\n{'='*70}")
        lines.append("BEST CONFIG PER TICKER")
        lines.append(f"{'='*70}")
        lines.append(f"  {'Ticker':>6s}  {'CSPd':>5s}  {'CCd':>5s}  {'DTE':>4s}  {'EC%':>5s}  {'Sharpe':>7s}  {'CAGR%':>7s}  {'DD%':>7s}")
        lines.append(f"  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*4}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")
        for t, cfg in sorted(dim["best_per_ticker"].items()):
            lines.append(
                f"  {t:>6s}  {cfg['csp_delta']:>5.2f}  {cfg['cc_delta']:>5.2f}  "
                f"{cfg['target_dte']:>4d}  {cfg['early_close_pct']:>5.2f}  "
                f"{cfg.get('sharpe', 0):>7.3f}  {cfg.get('cagr_pct', 0):>7.2f}  "
                f"{cfg.get('max_dd_pct', 0):>7.2f}"
            )

    # Top 10
    lines.append(f"\n{'='*70}")
    lines.append("TOP 10 CONFIGURATIONS BY SHARPE")
    lines.append(f"{'='*70}")
    lines.append(f"  {'#':>3s}  {'Ticker':>6s}  {'CSPd':>5s}  {'CCd':>5s}  {'DTE':>4s}  {'EC%':>5s}  {'Sharpe':>7s}  {'Sortino':>8s}  {'CAGR%':>7s}  {'DD%':>7s}")
    for i, cfg in enumerate(analysis.get("top_10", []), 1):
        lines.append(
            f"  {i:>3d}  {cfg['ticker']:>6s}  {cfg['csp_delta']:>5.2f}  {cfg['cc_delta']:>5.2f}  "
            f"{cfg['target_dte']:>4d}  {cfg['early_close_pct']:>5.2f}  "
            f"{cfg.get('sharpe', 0):>7.3f}  {cfg.get('sortino', 0):>8.3f}  "
            f"{cfg.get('cagr_pct', 0):>7.2f}  {cfg.get('max_dd_pct', 0):>7.2f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Wheel Strategy Optimization Sweep")
    parser.add_argument("--months", type=int, default=12, help="Backtest window in months")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--universe", type=str, default="balanced",
                        help="Universe: balanced, scalp, all, or a UNIVERSE key")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--capital", type=float, default=100_000, help="Starting capital")
    parser.add_argument("--full-grid", action="store_true", help="Use full grid (slower)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    # Get tickers
    if args.tickers:
        tickers = args.tickers.split(",")
    else:
        tickers = get_universe(args.universe)

    print(f"Universe: {args.universe} ({len(tickers)} tickers)")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Backtest window: {args.months} months")
    print(f"Capital: ${args.capital:,.0f}")

    grid = SWEEP_GRID if args.full_grid else QUICK_GRID

    # Run sweep
    results = run_sweep(
        tickers=tickers,
        months=args.months,
        grid=grid,
        workers=args.workers,
        capital=args.capital,
    )

    # Analyze
    analysis = analyze_results(results)

    # Print summary
    summary = format_summary(analysis)
    print(summary)

    # Save results
    out_file = args.output or str(OUT_DIR / f"sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nResults saved to: {out_file}")

    # Save summary
    summary_file = str(Path(out_file).with_suffix(".txt"))
    with open(summary_file, "w") as f:
        f.write(summary)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
