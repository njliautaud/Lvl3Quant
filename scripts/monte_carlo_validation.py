#!/usr/bin/env python3
"""
Monte Carlo Validation Module for Lvl3Quant Fill Simulation Results.

Produces robustness metrics via bootstrap resampling of trade PnL sequences.
Works as both a standalone CLI tool and an importable library.

Usage:
    # From fill sim results (single file)
    python monte_carlo_validation.py --input fill_sim_results.json --sims 10000

    # From fill sim results (directory of per-date files)
    python monte_carlo_validation.py --input cnn_wf_sim_results/ --sims 10000

    # From a simple PnL file (one PnL value per line)
    python monte_carlo_validation.py --pnl-file trade_pnls.txt --sims 10000

    # As library
    from monte_carlo_validation import run_monte_carlo
    report = run_monte_carlo(pnl_array, n_sims=10000, initial_capital=25000)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Competitor benchmark (for comparison line in report)
# ---------------------------------------------------------------------------
COMPETITOR = {
    "win_rate": 0.83,
    "sortino": 12.0,
    "profit_factor": 1.57,
    "max_dd_pct": 0.12,
    "label": "Competitor (backtested 2000-2026)",
}

# Minimum acceptance thresholds
MIN_THRESHOLDS = {
    "sortino_p5": 2.0,
    "prob_ruin": 0.05,   # must be BELOW this
    "prob_profit": 0.80,  # must be ABOVE this
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_trades_from_fillsim(path: str) -> Tuple[List[dict], List[dict]]:
    """Load trades from one or many fill-sim JSON files.

    Returns:
        (all_trades, per_day_summaries)
    """
    p = Path(path)
    files: List[Path] = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.json"))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    if not files:
        raise ValueError(f"No JSON files found in {path}")

    all_trades: List[dict] = []
    per_day: List[dict] = []

    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)

        trades = data.get("trades", [])
        if not trades:
            continue

        # Extract date from filename (e.g. ..._2025-12-01.json)
        date_str = fp.stem.rsplit("_", 1)[-1] if "_" in fp.stem else fp.stem

        day_pnls = [t["pnl_dollars"] for t in trades]
        wins = sum(1 for p in day_pnls if p > 0)
        day_wr = wins / len(day_pnls) if day_pnls else 0.0

        per_day.append({
            "date": date_str,
            "pnl": round(sum(day_pnls), 2),
            "trades": len(trades),
            "wr": round(day_wr, 4),
            "file": str(fp.name),
        })

        all_trades.extend(trades)

    return all_trades, per_day


def load_pnl_from_file(path: str) -> np.ndarray:
    """Load PnL values from a text file (one value per line)."""
    pnls = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                pnls.append(float(line))
    return np.array(pnls, dtype=np.float64)


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------

def compute_backtest_stats(
    pnl: np.ndarray,
    initial_capital: float,
    trades: Optional[List[dict]] = None,
) -> dict:
    """Compute standard backtest statistics from a PnL array."""
    n = len(pnl)
    if n == 0:
        return {}

    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / n if n else 0.0
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0

    gross_profit = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    total_pnl = float(np.sum(pnl))
    equity = initial_capital + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd_pct = float(np.max(drawdown)) if len(drawdown) else 0.0
    max_dd_dollar = float(np.max(peak - equity)) if len(peak) else 0.0

    # Per-trade Sharpe and Sortino (annualized assuming ~252 trading days)
    mean_pnl = float(np.mean(pnl))
    std_pnl = float(np.std(pnl, ddof=1)) if n > 1 else 1e-9
    downside = pnl[pnl < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9

    # Estimate trades per day from trade metadata if available
    trades_per_day = _estimate_trades_per_day(trades) if trades else max(n / 60, 1)

    ann_factor = np.sqrt(252 * trades_per_day)
    sharpe = (mean_pnl / std_pnl) * ann_factor if std_pnl > 1e-12 else 0.0
    sortino = (mean_pnl / downside_std) * ann_factor if downside_std > 1e-12 else 0.0

    # Average hold time
    avg_hold_ns = 0.0
    if trades:
        holds = [t.get("hold_duration_ns", 0) for t in trades if t.get("hold_duration_ns")]
        if holds:
            avg_hold_ns = np.mean(holds)
    avg_hold_sec = avg_hold_ns / 1e9

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd_pct, 4),
        "max_dd_dollar": round(max_dd_dollar, 2),
        "total_pnl": round(total_pnl, 2),
        "mean_pnl_per_trade": round(mean_pnl, 2),
        "trades_per_day": round(trades_per_day, 2),
        "avg_hold_sec": round(avg_hold_sec, 2),
        "initial_capital": initial_capital,
    }


def _estimate_trades_per_day(trades: List[dict]) -> float:
    """Estimate trades per day from fill timestamps."""
    if not trades:
        return 1.0
    # Use signal_time_ns to detect unique dates
    times = sorted(t.get("signal_time_ns", 0) for t in trades if t.get("signal_time_ns"))
    if len(times) < 2:
        return max(len(trades), 1)
    # Rough: span in nanoseconds -> days
    span_ns = times[-1] - times[0]
    span_days = span_ns / (24 * 3600 * 1e9)
    if span_days < 1:
        return float(len(trades))
    return len(trades) / span_days


# ---------------------------------------------------------------------------
# Monte Carlo engine (pure numpy, vectorized)
# ---------------------------------------------------------------------------

def _mc_simulate(
    pnl: np.ndarray,
    n_sims: int,
    initial_capital: float,
    rng: np.random.Generator,
) -> dict:
    """Run bootstrap Monte Carlo simulations.

    Returns dict with arrays of per-simulation metrics.
    """
    n_trades = len(pnl)
    # Generate all bootstrap indices at once: (n_sims, n_trades)
    indices = rng.integers(0, n_trades, size=(n_sims, n_trades))
    # Resampled PnL matrix
    sim_pnl = pnl[indices]  # (n_sims, n_trades)

    # Cumulative equity curves
    cum_pnl = np.cumsum(sim_pnl, axis=1)  # (n_sims, n_trades)
    equity = initial_capital + cum_pnl      # (n_sims, n_trades)

    # Final equity
    final_equity = equity[:, -1]

    # Max drawdown per simulation
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdown_dollar = running_max - equity
    drawdown_pct = drawdown_dollar / np.where(running_max > 0, running_max, 1.0)
    max_dd_pct = np.max(drawdown_pct, axis=1)
    max_dd_dollar = np.max(drawdown_dollar, axis=1)

    # Per-sim win rate
    win_counts = np.sum(sim_pnl > 0, axis=1)
    win_rates = win_counts / n_trades

    # Per-sim profit factor
    gross_wins = np.sum(np.where(sim_pnl > 0, sim_pnl, 0.0), axis=1)
    gross_losses = np.abs(np.sum(np.where(sim_pnl <= 0, sim_pnl, 0.0), axis=1))
    gross_losses = np.where(gross_losses < 1e-9, 1e-9, gross_losses)
    profit_factors = gross_wins / gross_losses

    # Per-sim Sharpe / Sortino (annualized)
    mean_pnl_per_sim = np.mean(sim_pnl, axis=1)
    std_pnl_per_sim = np.std(sim_pnl, axis=1, ddof=1)
    std_pnl_per_sim = np.where(std_pnl_per_sim < 1e-12, 1e-12, std_pnl_per_sim)

    # Downside std: for each sim, std of negative PnLs
    # Vectorized approximation: use all below-mean as downside
    neg_mask = sim_pnl < 0
    # Replace positives with NaN for per-row std
    neg_only = np.where(neg_mask, sim_pnl, np.nan)
    with np.errstate(all="ignore"):
        downside_std = np.nanstd(neg_only, axis=1, ddof=1)
    downside_std = np.where(np.isnan(downside_std) | (downside_std < 1e-12), 1e-12, downside_std)

    # Estimate annualization factor (same for all sims)
    trades_per_day = max(n_trades / 60, 1)  # rough default
    ann_factor = np.sqrt(252 * trades_per_day)

    sharpes = (mean_pnl_per_sim / std_pnl_per_sim) * ann_factor
    sortinos = (mean_pnl_per_sim / downside_std) * ann_factor

    # Recovery from max DD: index of max DD -> how many trades to recover to peak
    max_dd_idx = np.argmax(drawdown_dollar, axis=1)
    recovery_bars = np.zeros(n_sims, dtype=np.int64)
    for i in range(n_sims):
        dd_idx = max_dd_idx[i]
        peak_val = running_max[i, dd_idx]
        recovered = np.where(equity[i, dd_idx:] >= peak_val)[0]
        recovery_bars[i] = recovered[0] if len(recovered) else (n_trades - dd_idx)

    return {
        "final_equity": final_equity,
        "max_dd_pct": max_dd_pct,
        "max_dd_dollar": max_dd_dollar,
        "win_rates": win_rates,
        "profit_factors": profit_factors,
        "sharpes": sharpes,
        "sortinos": sortinos,
        "recovery_bars": recovery_bars,
        "sim_pnl": sim_pnl,
    }


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def _compute_risk_metrics(
    pnl: np.ndarray,
    mc: dict,
    initial_capital: float,
) -> dict:
    """Compute VaR, CVaR, Kelly, worst-case DD."""
    # VaR / CVaR on per-trade PnL
    sorted_pnl = np.sort(pnl)
    n = len(sorted_pnl)
    var_95_idx = int(n * 0.05)
    var_99_idx = int(n * 0.01)
    var_95 = float(sorted_pnl[var_95_idx]) if var_95_idx < n else float(sorted_pnl[0])
    var_99 = float(sorted_pnl[var_99_idx]) if var_99_idx < n else float(sorted_pnl[0])
    cvar_95 = float(np.mean(sorted_pnl[: max(var_95_idx, 1)]))

    # Kelly criterion: f* = (p * b - q) / b
    # p = win rate, q = 1-p, b = avg_win / avg_loss (absolute)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    p = len(wins) / n if n else 0.0
    q = 1 - p
    avg_w = float(np.mean(wins)) if len(wins) else 0.0
    avg_l = float(np.abs(np.mean(losses))) if len(losses) else 1.0
    b = avg_w / avg_l if avg_l > 0 else 0.0
    kelly = (p * b - q) / b if b > 0 else 0.0
    kelly = max(kelly, 0.0)  # no negative Kelly

    # Worst-case drawdown at 95% confidence (from MC)
    worst_dd_95 = float(np.percentile(mc["max_dd_pct"], 95))

    # Median recovery bars
    median_recovery = float(np.median(mc["recovery_bars"]))

    return {
        "var_95": round(var_95, 2),
        "var_99": round(var_99, 2),
        "cvar_95": round(cvar_95, 2),
        "kelly_fraction": round(kelly, 4),
        "worst_dd_95_pct": round(worst_dd_95, 4),
        "median_recovery_bars": int(median_recovery),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_monte_carlo(
    pnl: Union[np.ndarray, List[float]],
    n_sims: int = 10_000,
    initial_capital: float = 25_000.0,
    risk_per_trade: float = 0.009,
    trades: Optional[List[dict]] = None,
    per_day: Optional[List[dict]] = None,
    seed: int = 42,
) -> dict:
    """Run full Monte Carlo validation and return structured report.

    Args:
        pnl: Array of per-trade PnL values (dollars).
        n_sims: Number of bootstrap simulations.
        initial_capital: Starting account equity.
        risk_per_trade: Fraction of capital risked per trade (for reference).
        trades: Optional raw trade dicts (for hold-time stats).
        per_day: Optional per-day summaries (for regime analysis).
        seed: RNG seed for reproducibility.

    Returns:
        Complete report dict suitable for JSON serialization / dashboard.
    """
    pnl = np.asarray(pnl, dtype=np.float64)
    if len(pnl) == 0:
        raise ValueError("Empty PnL array — nothing to simulate.")

    rng = np.random.default_rng(seed)

    # Section 1: Original backtest stats
    backtest = compute_backtest_stats(pnl, initial_capital, trades)

    # Section 2: Monte Carlo
    mc = _mc_simulate(pnl, n_sims, initial_capital, rng)

    fe = mc["final_equity"]
    mc_summary = {
        "n_simulations": n_sims,
        "final_equity": {
            "median": round(float(np.median(fe)), 2),
            "p5": round(float(np.percentile(fe, 5)), 2),
            "p25": round(float(np.percentile(fe, 25)), 2),
            "p75": round(float(np.percentile(fe, 75)), 2),
            "p95": round(float(np.percentile(fe, 95)), 2),
        },
        "max_drawdown_pct": {
            "median": round(float(np.median(mc["max_dd_pct"])), 4),
            "p95": round(float(np.percentile(mc["max_dd_pct"], 95)), 4),
        },
        "sharpe": {
            "median": round(float(np.median(mc["sharpes"])), 4),
            "p5": round(float(np.percentile(mc["sharpes"], 5)), 4),
        },
        "sortino": {
            "median": round(float(np.median(mc["sortinos"])), 4),
            "p5": round(float(np.percentile(mc["sortinos"], 5)), 4),
        },
        "profit_factor": {
            "median": round(float(np.median(mc["profit_factors"])), 4),
            "p5": round(float(np.percentile(mc["profit_factors"], 5)), 4),
        },
        "win_rate": {
            "median": round(float(np.median(mc["win_rates"])), 4),
            "p5": round(float(np.percentile(mc["win_rates"], 5)), 4),
        },
        "prob_profit": round(float(np.mean(fe > initial_capital)), 4),
        "prob_ruin": round(float(np.mean(mc["max_dd_pct"] > 0.50)), 4),
    }

    # Section 3: Risk metrics
    risk = _compute_risk_metrics(pnl, mc, initial_capital)
    mc_summary.update({
        "var_95": risk["var_95"],
        "var_99": risk["var_99"],
        "cvar_95": risk["cvar_95"],
        "kelly_fraction": risk["kelly_fraction"],
    })

    # Section 4: Regime / per-day
    per_day_out = per_day or []
    regime = {}
    if per_day_out:
        day_pnls = [d["pnl"] for d in per_day_out]
        profitable_days = sum(1 for p in day_pnls if p > 0)
        regime = {
            "total_days": len(per_day_out),
            "profitable_days": profitable_days,
            "consistency_pct": round(profitable_days / len(per_day_out), 4) if per_day_out else 0.0,
            "worst_day": round(min(day_pnls), 2),
            "best_day": round(max(day_pnls), 2),
            "avg_daily_pnl": round(float(np.mean(day_pnls)), 2),
        }

    # Pass/fail verdict
    verdict = _evaluate_thresholds(mc_summary)

    report = {
        "backtest": backtest,
        "monte_carlo": mc_summary,
        "risk": risk,
        "per_day": per_day_out,
        "regime": regime,
        "competitor_benchmark": COMPETITOR,
        "verdict": verdict,
        "config": {
            "n_sims": n_sims,
            "initial_capital": initial_capital,
            "risk_per_trade": risk_per_trade,
            "seed": seed,
        },
    }
    return report


def _evaluate_thresholds(mc: dict) -> dict:
    """Evaluate pass/fail against minimum thresholds."""
    sortino_p5 = mc["sortino"]["p5"]
    prob_ruin = mc["prob_ruin"]
    prob_profit = mc["prob_profit"]

    checks = {
        "sortino_p5": {
            "value": sortino_p5,
            "threshold": MIN_THRESHOLDS["sortino_p5"],
            "pass": sortino_p5 > MIN_THRESHOLDS["sortino_p5"],
            "rule": f"Sortino p5 ({sortino_p5:.2f}) > {MIN_THRESHOLDS['sortino_p5']}",
        },
        "prob_ruin": {
            "value": prob_ruin,
            "threshold": MIN_THRESHOLDS["prob_ruin"],
            "pass": prob_ruin < MIN_THRESHOLDS["prob_ruin"],
            "rule": f"Prob ruin ({prob_ruin:.2%}) < {MIN_THRESHOLDS['prob_ruin']:.0%}",
        },
        "prob_profit": {
            "value": prob_profit,
            "threshold": MIN_THRESHOLDS["prob_profit"],
            "pass": prob_profit > MIN_THRESHOLDS["prob_profit"],
            "rule": f"Prob profit ({prob_profit:.2%}) > {MIN_THRESHOLDS['prob_profit']:.0%}",
        },
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "overall": "PASS" if all_pass else "FAIL",
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_report(report: dict) -> None:
    """Print a human-readable Monte Carlo validation report."""
    bt = report["backtest"]
    mc = report["monte_carlo"]
    risk = report["risk"]
    regime = report.get("regime", {})
    verdict = report["verdict"]
    comp = report["competitor_benchmark"]

    w = 60
    print("=" * w)
    print("  MONTE CARLO VALIDATION REPORT")
    print("=" * w)

    # Section 1
    print(f"\n{'-' * w}")
    print("  SECTION 1: ORIGINAL BACKTEST STATS")
    print(f"{'-' * w}")
    print(f"  Total trades:        {bt['total_trades']}")
    print(f"  Win rate:            {bt['win_rate']:.2%}")
    print(f"  Avg win:            ${bt['avg_win']:>10,.2f}")
    print(f"  Avg loss:           ${bt['avg_loss']:>10,.2f}")
    print(f"  Profit factor:       {bt['profit_factor']:.4f}")
    print(f"  Sharpe (ann):        {bt['sharpe']:.4f}")
    print(f"  Sortino (ann):       {bt['sortino']:.4f}")
    print(f"  Max drawdown:        {bt['max_dd_pct']:.2%}  (${bt['max_dd_dollar']:,.2f})")
    print(f"  Total PnL:          ${bt['total_pnl']:>10,.2f}")
    print(f"  Mean PnL/trade:     ${bt['mean_pnl_per_trade']:>10,.2f}")
    print(f"  Trades/day:          {bt['trades_per_day']:.1f}")
    print(f"  Avg hold time:       {bt['avg_hold_sec']:.1f}s")

    # Section 2
    print(f"\n{'-' * w}")
    print(f"  SECTION 2: MONTE CARLO DISTRIBUTION ({mc['n_simulations']:,} sims)")
    print(f"{'-' * w}")
    fe = mc["final_equity"]
    print(f"  Final equity (median):   ${fe['median']:>12,.2f}")
    print(f"  Final equity (p5):       ${fe['p5']:>12,.2f}")
    print(f"  Final equity (p25):      ${fe['p25']:>12,.2f}")
    print(f"  Final equity (p75):      ${fe['p75']:>12,.2f}")
    print(f"  Final equity (p95):      ${fe['p95']:>12,.2f}")
    print()
    dd = mc["max_drawdown_pct"]
    print(f"  Max DD median:           {dd['median']:.2%}")
    print(f"  Max DD p95 (worst):      {dd['p95']:.2%}")
    print()
    print(f"  Prob of profit:          {mc['prob_profit']:.2%}")
    print(f"  Prob of ruin (>50% DD):  {mc['prob_ruin']:.2%}")
    print()
    print(f"  Sharpe  median:          {mc['sharpe']['median']:.4f}")
    print(f"  Sharpe  p5 (worst):      {mc['sharpe']['p5']:.4f}")
    print(f"  Sortino median:          {mc['sortino']['median']:.4f}")
    print(f"  Sortino p5 (worst):      {mc['sortino']['p5']:.4f}")
    print(f"  PF      median:          {mc['profit_factor']['median']:.4f}")
    print(f"  PF      p5 (worst):      {mc['profit_factor']['p5']:.4f}")
    print(f"  WR      median:          {mc['win_rate']['median']:.2%}")
    print(f"  WR      p5 (worst):      {mc['win_rate']['p5']:.2%}")

    # Section 3
    print(f"\n{'-' * w}")
    print("  SECTION 3: RISK METRICS")
    print(f"{'-' * w}")
    print(f"  VaR (95%):              ${risk['var_95']:>10,.2f}")
    print(f"  VaR (99%):              ${risk['var_99']:>10,.2f}")
    print(f"  CVaR / Exp. Shortfall:  ${risk['cvar_95']:>10,.2f}")
    print(f"  Kelly fraction:          {risk['kelly_fraction']:.4f}")
    print(f"  Worst DD @95% conf:      {risk['worst_dd_95_pct']:.2%}")
    print(f"  Recovery from max DD:    {risk['median_recovery_bars']} trades (median)")

    # Section 4
    if regime:
        print(f"\n{'-' * w}")
        print("  SECTION 4: REGIME ROBUSTNESS")
        print(f"{'-' * w}")
        print(f"  Trading days:            {regime['total_days']}")
        print(f"  Profitable days:         {regime['profitable_days']}")
        print(f"  Consistency:             {regime['consistency_pct']:.2%}")
        print(f"  Worst single day:       ${regime['worst_day']:>10,.2f}")
        print(f"  Best single day:        ${regime['best_day']:>10,.2f}")
        print(f"  Avg daily PnL:          ${regime['avg_daily_pnl']:>10,.2f}")

    # Competitor comparison
    print(f"\n{'-' * w}")
    print(f"  COMPETITOR COMPARISON: {comp['label']}")
    print(f"{'-' * w}")
    _cmp("Win rate", bt["win_rate"], comp["win_rate"], fmt=".2%", higher_better=True)
    _cmp("Sortino", bt["sortino"], comp["sortino"], fmt=".2f", higher_better=True)
    _cmp("Profit factor", bt["profit_factor"], comp["profit_factor"], fmt=".4f", higher_better=True)
    _cmp("Max DD %", bt["max_dd_pct"], comp["max_dd_pct"], fmt=".2%", higher_better=False)

    # Verdict
    print(f"\n{'=' * w}")
    overall = verdict["overall"]
    tag = "*** PASS ***" if overall == "PASS" else "*** FAIL ***"
    print(f"  VERDICT: {tag}")
    print(f"{'=' * w}")
    for name, chk in verdict["checks"].items():
        status = "OK" if chk["pass"] else "FAIL"
        print(f"  [{status:>4}] {chk['rule']}")
    print()


def _cmp(label: str, ours: float, theirs: float, fmt: str, higher_better: bool) -> None:
    """Print a comparison line."""
    ours_s = f"{ours:{fmt}}"
    theirs_s = f"{theirs:{fmt}}"
    if higher_better:
        arrow = ">>>" if ours > theirs else "<<<" if ours < theirs else "==="
    else:
        arrow = ">>>" if ours < theirs else "<<<" if ours > theirs else "==="
    winner = "OURS" if arrow == ">>>" else ("THEIRS" if arrow == "<<<" else "TIE")
    print(f"  {label:<18} Ours: {ours_s:>10}  vs  {theirs_s:<10}  [{winner}]")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo validation for Lvl3Quant fill simulation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to fill_sim JSON file or directory of per-date JSON files.",
    )
    parser.add_argument(
        "--pnl-file",
        help="Path to text file with one PnL value per line.",
    )
    parser.add_argument(
        "--sims", "-n",
        type=int, default=10_000,
        help="Number of Monte Carlo simulations (default: 10000).",
    )
    parser.add_argument(
        "--capital",
        type=float, default=25_000.0,
        help="Initial capital in dollars (default: 25000).",
    )
    parser.add_argument(
        "--risk",
        type=float, default=0.009,
        help="Risk per trade as fraction of capital (default: 0.009 = 0.9%%).",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to write JSON report (optional; also prints to stdout).",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON, suppress human-readable report.",
    )

    args = parser.parse_args()

    if not args.input and not args.pnl_file:
        parser.error("Provide either --input (fill sim JSON) or --pnl-file (PnL text file).")

    trades = None
    per_day = None

    if args.input:
        raw_trades, per_day = load_trades_from_fillsim(args.input)
        if not raw_trades:
            print("ERROR: No trades found in input.", file=sys.stderr)
            sys.exit(1)
        pnl = np.array([t["pnl_dollars"] for t in raw_trades], dtype=np.float64)
        trades = raw_trades
        if not args.json_only:
            print(f"Loaded {len(pnl)} trades from {args.input} ({len(per_day)} days)")
    else:
        pnl = load_pnl_from_file(args.pnl_file)
        if not args.json_only:
            print(f"Loaded {len(pnl)} PnL values from {args.pnl_file}")

    t0 = time.perf_counter()
    report = run_monte_carlo(
        pnl=pnl,
        n_sims=args.sims,
        initial_capital=args.capital,
        risk_per_trade=args.risk,
        trades=trades,
        per_day=per_day,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    if not args.json_only:
        print_report(report)
        print(f"  Completed {args.sims:,} simulations in {elapsed:.2f}s\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        if not args.json_only:
            print(f"  JSON report saved to: {args.output}")

    if args.json_only:
        print(json.dumps(report, indent=2, default=str))

    # Exit code based on verdict
    sys.exit(0 if report["verdict"]["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
