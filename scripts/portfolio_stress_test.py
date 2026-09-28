#!/usr/bin/env python3
"""
Portfolio Tail-Risk Stress Test
================================
Tests the honest 3-strategy portfolio (V5+IC+ETF) under historical stress periods.
Goes beyond annual returns to show week-by-week and month-by-month behavior
during the worst drawdowns.

Output: output/portfolio_stress_test/
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "portfolio_stress_test"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Stress periods to analyze
STRESS_PERIODS = {
    "COVID Crash (Feb-Apr 2020)": ("2020-02-19", "2020-04-30"),
    "COVID Recovery (May-Jul 2020)": ("2020-05-01", "2020-07-31"),
    "2022 Rate Hiking (Jan-Jun 2022)": ("2022-01-03", "2022-06-30"),
    "2022 Bear Market (Jun-Oct 2022)": ("2022-06-01", "2022-10-31"),
    "SVB Crisis (Mar 2023)": ("2023-03-01", "2023-03-31"),
    "Aug 2024 Vol Spike": ("2024-07-25", "2024-08-15"),
    "2025 Tariff Shock (Apr 2025)": ("2025-04-01", "2025-04-30"),
    "Full 2020": ("2020-01-02", "2020-12-31"),
    "Full 2022": ("2022-01-03", "2022-12-30"),
}

LEVERAGE_LEVELS = {
    "Conservative (1x)": ("equity_conservative.parquet", 1.0),
    "Balanced (2x)": ("equity_balanced.parquet", 2.0),
    "Aggressive (3x)": ("equity_aggressive.parquet", 3.0),
}


def load_equity(filename):
    path = ROOT / "output" / "honest_portfolio_optimizer" / filename
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["daily_ret"] = df["equity"].pct_change()
    return df


def analyze_period(eq, start, end, label):
    """Analyze a specific date range."""
    mask = (eq.index >= start) & (eq.index <= end)
    period = eq.loc[mask].copy()

    if len(period) < 2:
        return None

    rets = period["daily_ret"].dropna()
    if len(rets) == 0:
        return None

    equity_start = period["equity"].iloc[0]
    equity_end = period["equity"].iloc[-1]
    total_return = (equity_end / equity_start - 1) * 100

    # Drawdown within period
    running_max = period["equity"].cummax()
    drawdown = (period["equity"] / running_max - 1) * 100
    max_dd = drawdown.min()

    # Worst single day
    worst_day = rets.min() * 100
    worst_day_date = rets.idxmin().strftime("%Y-%m-%d") if len(rets) > 0 else "N/A"

    # Best single day
    best_day = rets.max() * 100

    # Win rate
    wr = (rets > 0).mean() * 100

    # Annualized Sharpe (if enough data)
    if rets.std() > 0:
        sharpe = (rets.mean() - 0.04/252) / rets.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Weekly returns
    weekly = period["equity"].resample("W-FRI").last().pct_change().dropna()
    worst_week = weekly.min() * 100 if len(weekly) > 0 else 0
    best_week = weekly.max() * 100 if len(weekly) > 0 else 0

    # Monthly returns
    monthly = period["equity"].resample("ME").last().pct_change().dropna()
    worst_month = monthly.min() * 100 if len(monthly) > 0 else 0

    # Consecutive losing days
    losing_streak = 0
    max_losing_streak = 0
    for r in rets:
        if r < 0:
            losing_streak += 1
            max_losing_streak = max(max_losing_streak, losing_streak)
        else:
            losing_streak = 0

    return {
        "period": label,
        "days": len(rets),
        "total_return_pct": round(total_return, 2),
        "max_dd_pct": round(max_dd, 2),
        "worst_day_pct": round(worst_day, 2),
        "worst_day_date": worst_day_date,
        "best_day_pct": round(best_day, 2),
        "worst_week_pct": round(worst_week, 2),
        "worst_month_pct": round(worst_month, 2),
        "win_rate_pct": round(wr, 1),
        "sharpe": round(sharpe, 2),
        "max_losing_streak": max_losing_streak,
    }


def monthly_returns_table(eq):
    """Generate a month-by-month returns table."""
    monthly = eq["equity"].resample("ME").last()
    monthly_ret = monthly.pct_change() * 100

    # Pivot to year x month
    df = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "return": monthly_ret.values,
    }).dropna()

    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

    # Add annual total
    annual = eq["equity"].resample("YE").last().pct_change() * 100
    year_map = {y: r for y, r in zip(annual.index.year, annual.values) if not np.isnan(r)}
    pivot["Annual"] = pivot.index.map(lambda y: year_map.get(y, np.nan))

    return pivot


def drawdown_analysis(eq):
    """Find top 5 worst drawdowns."""
    running_max = eq["equity"].cummax()
    dd = (eq["equity"] / running_max - 1) * 100

    # Find drawdown periods
    in_dd = dd < 0
    drawdowns = []
    start = None

    for i, (dt, is_dd) in enumerate(in_dd.items()):
        if is_dd and start is None:
            start = dt
        elif not is_dd and start is not None:
            # End of drawdown
            period_dd = dd.loc[start:dt]
            trough_date = period_dd.idxmin()
            trough_val = period_dd.min()
            duration = (dt - start).days
            recovery_days = (dt - trough_date).days
            drawdowns.append({
                "start": start.strftime("%Y-%m-%d"),
                "trough": trough_date.strftime("%Y-%m-%d"),
                "recovery": dt.strftime("%Y-%m-%d"),
                "max_dd_pct": round(trough_val, 2),
                "duration_days": duration,
                "recovery_days": recovery_days,
            })
            start = None

    # Sort by severity
    drawdowns.sort(key=lambda x: x["max_dd_pct"])
    return drawdowns[:10]


def main():
    print("=" * 80)
    print("PORTFOLIO TAIL-RISK STRESS TEST")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 80)

    all_results = {}

    for lev_label, (filename, lev) in LEVERAGE_LEVELS.items():
        print(f"\n{'─' * 70}")
        print(f"  {lev_label}")
        print(f"{'─' * 70}")

        eq = load_equity(filename)
        results = []

        for period_label, (start, end) in STRESS_PERIODS.items():
            r = analyze_period(eq, start, end, period_label)
            if r:
                results.append(r)

        # Print table
        print(f"\n{'Period':40s} | {'Return':>7} | {'MaxDD':>7} | {'WorstDay':>8} | {'WorstWk':>7} | {'WR':>5} | {'Sharpe':>6}")
        print("-" * 95)
        for r in results:
            print(f"{r['period']:40s} | {r['total_return_pct']:>6.1f}% | {r['max_dd_pct']:>6.1f}% | {r['worst_day_pct']:>7.2f}% | {r['worst_week_pct']:>6.1f}% | {r['win_rate_pct']:>4.0f}% | {r['sharpe']:>6.2f}")

        all_results[lev_label] = results

        # Monthly returns (conservative only to avoid spam)
        if lev == 1.0:
            print(f"\n  Monthly Returns (%):")
            monthly = monthly_returns_table(eq)
            print(monthly.to_string(float_format=lambda x: f"{x:+.1f}" if not np.isnan(x) else ""))

            print(f"\n  Top 10 Drawdowns:")
            dds = drawdown_analysis(eq)
            for i, dd in enumerate(dds):
                print(f"    {i+1}. {dd['max_dd_pct']:+.2f}% | {dd['start']} → {dd['trough']} → {dd['recovery']} | {dd['duration_days']}d total, {dd['recovery_days']}d recovery")

    # Key findings
    print(f"\n{'=' * 80}")
    print("KEY FINDINGS")
    print(f"{'=' * 80}")

    cons = all_results.get("Conservative (1x)", [])
    agg = all_results.get("Aggressive (3x)", [])

    # COVID crash comparison
    covid_cons = next((r for r in cons if "COVID Crash" in r["period"]), None)
    covid_agg = next((r for r in agg if "COVID Crash" in r["period"]), None)

    if covid_cons and covid_agg:
        print(f"\n  COVID Crash (worst stress period):")
        print(f"    1x: {covid_cons['total_return_pct']:+.1f}% return, {covid_cons['max_dd_pct']:.1f}% max DD, worst day {covid_cons['worst_day_pct']:.2f}%")
        print(f"    3x: {covid_agg['total_return_pct']:+.1f}% return, {covid_agg['max_dd_pct']:.1f}% max DD, worst day {covid_agg['worst_day_pct']:.2f}%")

    # 2022 bear market
    bear_cons = next((r for r in cons if "Bear Market" in r["period"]), None)
    bear_agg = next((r for r in agg if "Bear Market" in r["period"]), None)

    if bear_cons and bear_agg:
        print(f"\n  2022 Bear Market:")
        print(f"    1x: {bear_cons['total_return_pct']:+.1f}% return, {bear_cons['max_dd_pct']:.1f}% max DD")
        print(f"    3x: {bear_agg['total_return_pct']:+.1f}% return, {bear_agg['max_dd_pct']:.1f}% max DD")

    # Overall assessment
    worst_cons = min(r["max_dd_pct"] for r in cons) if cons else 0
    worst_agg = min(r["max_dd_pct"] for r in agg) if agg else 0

    print(f"\n  Worst drawdown across all stress periods:")
    print(f"    Conservative (1x): {worst_cons:.1f}%")
    print(f"    Aggressive (3x):   {worst_agg:.1f}%")

    # Any negative-return stress periods?
    neg_cons = [r for r in cons if r["total_return_pct"] < 0]
    print(f"\n  Stress periods with NEGATIVE returns (1x): {len(neg_cons)}/{len(cons)}")
    for r in neg_cons:
        print(f"    {r['period']}: {r['total_return_pct']:+.1f}%")

    if not neg_cons:
        print(f"    NONE — portfolio was profitable in every stress period at 1x")

    # Save
    save_data = {
        "generated": datetime.now().isoformat(),
        "stress_results": all_results,
    }
    with open(OUTPUT / "stress_test_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nSaved to {OUTPUT}/")


if __name__ == "__main__":
    main()
