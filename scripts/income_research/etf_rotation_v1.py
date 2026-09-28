#!/usr/bin/env python3
"""
ETF Dual-Momentum Rotation Backtest v1
=======================================
HC #705: All adversarial checks built in from line 1.

Universe: 13 ETFs with 12+ year history (no survivorship bias — ETFs rebalance internally).
Strategies: 6 monthly rotation variants.
Quality gates: permutation test, regime test, sub-period consistency,
              outlier removal, benchmark comparison, random rotation baseline.

Output: /home/jupiter/Lvl3Quant/output/etf_rotation_v1/backtest_report.json
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLF", "XLE", "XLK", "XLV", "XLU", "TLT", "GLD", "SHY"]
CASH_PROXY = "SHY"
BENCHMARK = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2026-07-01"
INITIAL_CAPITAL = 440.0
N_PERMUTATIONS = 200
N_RANDOM_ROTATIONS = 200
RANDOM_SEED = 42
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/etf_rotation_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Quality gate thresholds
PERM_P_THRESHOLD = 0.05
REGIME_GAP_THRESHOLD = 0.50
OUTLIER_SHARPE_RETENTION = 0.50  # Sharpe after removing top 5% must be > 50% of original
RANDOM_PERCENTILE_THRESHOLD = 80  # strategy must beat 80th pctile of random rotations


# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data():
    """Download adjusted close prices for all ETFs."""
    print(f"Downloading data for {len(UNIVERSE)} ETFs: {', '.join(UNIVERSE)}")
    data = yf.download(UNIVERSE, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Verify all ETFs have data
    for etf in UNIVERSE:
        if etf not in prices.columns:
            print(f"  WARNING: {etf} missing from download")
        else:
            first_valid = prices[etf].first_valid_index()
            last_valid = prices[etf].last_valid_index()
            print(f"  {etf}: {first_valid.strftime('%Y-%m-%d')} to {last_valid.strftime('%Y-%m-%d')} ({prices[etf].notna().sum()} days)")

    prices = prices.ffill().dropna()
    print(f"Clean data: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')} ({len(prices)} trading days)")
    return prices


# ============================================================
# HELPER: Monthly rebalance dates
# ============================================================
def get_monthly_rebalance_dates(prices, lookback_months=6):
    """Get month-end rebalance dates with sufficient lookback."""
    monthly = prices.resample("ME").last()
    # Need lookback_months of history before first trade
    start_idx = max(lookback_months, 12)  # ensure 12m lookback available too
    return monthly.index[start_idx:]


# ============================================================
# HELPER: Compute returns
# ============================================================
def compute_trailing_returns(prices, date, months):
    """Compute trailing total return over N months ending at date."""
    monthly = prices.resample("ME").last()
    idx = monthly.index.get_loc(date)
    if idx < months:
        return pd.Series(np.nan, index=prices.columns)
    start_prices = monthly.iloc[idx - months]
    end_prices = monthly.iloc[idx]
    return (end_prices / start_prices) - 1


def compute_sma(prices, date, window=200):
    """Check if ETFs are above their 200-day SMA as of date."""
    loc = prices.index.get_loc(date, method="ffill") if date not in prices.index else prices.index.get_loc(date)
    if loc < window:
        return pd.Series(False, index=prices.columns)
    subset = prices.iloc[loc - window + 1 : loc + 1]
    sma = subset.mean()
    current = prices.iloc[loc]
    return current > sma


# ============================================================
# STRATEGY DEFINITIONS
# ============================================================
def strategy_relative_momentum_top3(prices, date):
    """Buy top 3 ETFs by 6-month return."""
    rets = compute_trailing_returns(prices, date, 6)
    tradeable = rets.drop(CASH_PROXY, errors="ignore").dropna()
    top3 = tradeable.nlargest(3).index.tolist()
    weights = {etf: 1.0 / 3 for etf in top3}
    return weights


def strategy_relative_momentum_top2(prices, date):
    """Buy top 2 ETFs by 6-month return."""
    rets = compute_trailing_returns(prices, date, 6)
    tradeable = rets.drop(CASH_PROXY, errors="ignore").dropna()
    top2 = tradeable.nlargest(2).index.tolist()
    weights = {etf: 1.0 / 2 for etf in top2}
    return weights


def strategy_absolute_momentum_filter(prices, date):
    """Top 3 by 6m return, but only if > SHY's 6m return. Rest in SHY."""
    rets = compute_trailing_returns(prices, date, 6)
    shy_ret = rets.get(CASH_PROXY, 0)
    tradeable = rets.drop(CASH_PROXY, errors="ignore").dropna()
    top3 = tradeable.nlargest(3)
    # Filter: must beat SHY
    passing = top3[top3 > shy_ret]
    if len(passing) == 0:
        return {CASH_PROXY: 1.0}
    weights = {etf: 1.0 / 3 for etf in passing.index}
    remainder = 1.0 - sum(weights.values())
    if remainder > 0.01:
        weights[CASH_PROXY] = remainder
    return weights


def strategy_dual_momentum_12m(prices, date):
    """Same as absolute momentum but 12-month lookback."""
    rets = compute_trailing_returns(prices, date, 12)
    shy_ret = rets.get(CASH_PROXY, 0)
    tradeable = rets.drop(CASH_PROXY, errors="ignore").dropna()
    top3 = tradeable.nlargest(3)
    passing = top3[top3 > shy_ret]
    if len(passing) == 0:
        return {CASH_PROXY: 1.0}
    weights = {etf: 1.0 / 3 for etf in passing.index}
    remainder = 1.0 - sum(weights.values())
    if remainder > 0.01:
        weights[CASH_PROXY] = remainder
    return weights


def strategy_mean_rev_bottom3(prices, date):
    """Buy bottom 3 by 1-month return (sector mean-reversion)."""
    rets = compute_trailing_returns(prices, date, 1)
    tradeable = rets.drop(CASH_PROXY, errors="ignore").dropna()
    bottom3 = tradeable.nsmallest(3).index.tolist()
    weights = {etf: 1.0 / 3 for etf in bottom3}
    return weights


def strategy_trend_following(prices, date):
    """Top 3 by 6m return, but only if above 200-day SMA. Below → SHY."""
    rets = compute_trailing_returns(prices, date, 6)
    above_sma = compute_sma(prices, date, 200)
    tradeable = rets.drop(CASH_PROXY, errors="ignore").dropna()
    top3 = tradeable.nlargest(3)
    passing = [etf for etf in top3.index if above_sma.get(etf, False)]
    if len(passing) == 0:
        return {CASH_PROXY: 1.0}
    weights = {etf: 1.0 / 3 for etf in passing}
    remainder = 1.0 - sum(weights.values())
    if remainder > 0.01:
        weights[CASH_PROXY] = remainder
    return weights


STRATEGIES = {
    "relative_momentum_top3": strategy_relative_momentum_top3,
    "relative_momentum_top2": strategy_relative_momentum_top2,
    "absolute_momentum_filter": strategy_absolute_momentum_filter,
    "dual_momentum_12m": strategy_dual_momentum_12m,
    "mean_rev_bottom3": strategy_mean_rev_bottom3,
    "trend_following": strategy_trend_following,
}


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(prices, strategy_fn, rebalance_dates=None):
    """
    Run monthly rotation backtest. Returns dict with equity curve and trade-level returns.
    """
    monthly_prices = prices.resample("ME").last()

    if rebalance_dates is None:
        rebalance_dates = get_monthly_rebalance_dates(prices)

    # Track portfolio
    monthly_returns = []
    holdings_history = []

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else None
        if next_date is None:
            break

        # Get weights from strategy
        weights = strategy_fn(prices, date)

        # Compute portfolio return for next month
        if date in monthly_prices.index and next_date in monthly_prices.index:
            date_loc = monthly_prices.index.get_loc(date)
            next_loc = monthly_prices.index.get_loc(next_date)

            port_ret = 0.0
            for etf, w in weights.items():
                if etf in monthly_prices.columns:
                    p0 = monthly_prices[etf].iloc[date_loc]
                    p1 = monthly_prices[etf].iloc[next_loc]
                    if p0 > 0:
                        port_ret += w * (p1 / p0 - 1)

            monthly_returns.append({
                "date": next_date,
                "return": port_ret,
                "holdings": weights,
            })

    if not monthly_returns:
        return None

    # Build equity curve
    equity = [INITIAL_CAPITAL]
    dates = [rebalance_dates[0]]
    returns_series = []

    for mr in monthly_returns:
        equity.append(equity[-1] * (1 + mr["return"]))
        dates.append(mr["date"])
        returns_series.append(mr["return"])

    returns_arr = np.array(returns_series)

    return {
        "equity": equity,
        "dates": [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in dates],
        "monthly_returns": returns_arr,
        "trade_details": monthly_returns,
        "final_equity": equity[-1],
        "total_return_pct": (equity[-1] / equity[0] - 1) * 100,
        "n_months": len(returns_series),
    }


def compute_metrics(returns_arr):
    """Compute risk-adjusted metrics from monthly returns array."""
    if len(returns_arr) == 0 or np.all(returns_arr == 0):
        return {"sharpe": 0, "sortino": 0, "cagr_pct": 0, "max_dd_pct": 0, "win_rate": 0, "profit_factor": 0}

    # Annualized Sharpe (monthly returns * sqrt(12))
    mean_m = np.mean(returns_arr)
    std_m = np.std(returns_arr, ddof=1) if len(returns_arr) > 1 else 1e-9
    sharpe = (mean_m / max(std_m, 1e-9)) * np.sqrt(12)

    # Sortino (downside deviation)
    downside = returns_arr[returns_arr < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_m / max(downside_std, 1e-9)) * np.sqrt(12)

    # CAGR
    n_years = len(returns_arr) / 12
    total_ret = np.prod(1 + returns_arr)
    cagr = (total_ret ** (1 / max(n_years, 0.1)) - 1) * 100

    # Max drawdown
    equity = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = np.min(drawdowns) * 100

    # Win rate
    wins = np.sum(returns_arr > 0)
    win_rate = wins / len(returns_arr) * 100

    # Profit factor
    gross_profit = np.sum(returns_arr[returns_arr > 0])
    gross_loss = abs(np.sum(returns_arr[returns_arr < 0]))
    profit_factor = gross_profit / max(gross_loss, 1e-9)

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 3),
        "mean_monthly_ret_pct": round(mean_m * 100, 3),
        "std_monthly_ret_pct": round(std_m * 100, 3),
        "n_months": len(returns_arr),
    }


# ============================================================
# QUALITY GATE (a): PERMUTATION TEST — RANDOM DATE ENTRIES
# ============================================================
def permutation_test_random_dates(prices, strategy_fn, observed_mean, rebalance_dates, n_perms=N_PERMUTATIONS):
    """
    Random date entry permutation test.
    For each permutation: randomly select entry dates from the available monthly dates,
    compute forward 1-month returns using the strategy, compare mean to observed.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    monthly_prices = prices.resample("ME").last()
    all_dates = list(rebalance_dates)
    n_trades = len(all_dates) - 1  # number of holding periods

    # Pool of all possible monthly dates (with enough lookback)
    all_monthly = list(monthly_prices.index[12:])  # ensure 12m lookback available

    perm_means = []
    for _ in range(n_perms):
        # Randomly sample n_trades entry dates (with replacement)
        random_dates = sorted(rng.choice(all_monthly, size=min(n_trades, len(all_monthly)), replace=True))

        perm_returns = []
        for rd in random_dates:
            # Find next month
            rd_loc = monthly_prices.index.get_loc(rd)
            if rd_loc + 1 >= len(monthly_prices):
                continue
            next_date = monthly_prices.index[rd_loc + 1]

            # Get strategy weights at random date
            weights = strategy_fn(prices, rd)

            # Compute forward return
            port_ret = 0.0
            for etf, w in weights.items():
                if etf in monthly_prices.columns:
                    p0 = monthly_prices[etf].iloc[rd_loc]
                    p1 = monthly_prices[etf].iloc[rd_loc + 1]
                    if p0 > 0:
                        port_ret += w * (p1 / p0 - 1)
            perm_returns.append(port_ret)

        if perm_returns:
            perm_means.append(np.mean(perm_returns))

    perm_means = np.array(perm_means)
    # p-value: fraction of permutations with mean >= observed
    p_value = np.mean(perm_means >= observed_mean)

    return {
        "observed_mean": round(float(observed_mean), 6),
        "perm_mean_avg": round(float(np.mean(perm_means)), 6),
        "perm_mean_std": round(float(np.std(perm_means)), 6),
        "p_value": round(float(p_value), 4),
        "pass": bool(p_value < PERM_P_THRESHOLD),
        "n_permutations": n_perms,
    }


# ============================================================
# QUALITY GATE (b): REGIME TEST — SPY GREEN vs RED MONTHS
# ============================================================
def regime_test(monthly_returns, spy_monthly_returns):
    """
    Stratify strategy returns by SPY green/red months.
    |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
    """
    # Align
    common_dates = set(r["date"] for r in monthly_returns) & set(spy_monthly_returns.index)

    green_rets, red_rets = [], []
    for r in monthly_returns:
        d = r["date"]
        if d in common_dates:
            spy_r = spy_monthly_returns.loc[d]
            if spy_r >= 0:
                green_rets.append(r["return"])
            else:
                red_rets.append(r["return"])

    green_rets = np.array(green_rets)
    red_rets = np.array(red_rets)

    def monthly_sharpe(arr):
        if len(arr) < 2:
            return 0
        return (np.mean(arr) / max(np.std(arr, ddof=1), 1e-9)) * np.sqrt(12)

    sharpe_green = monthly_sharpe(green_rets)
    sharpe_red = monthly_sharpe(red_rets)

    max_abs = max(abs(sharpe_green), abs(sharpe_red), 1e-9)
    gap = abs(sharpe_green - sharpe_red) / max_abs

    return {
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "n_green_months": len(green_rets),
        "n_red_months": len(red_rets),
        "gap": round(gap, 3),
        "pass": bool(gap <= REGIME_GAP_THRESHOLD),
        "mean_green_pct": round(float(np.mean(green_rets) * 100), 3) if len(green_rets) > 0 else 0,
        "mean_red_pct": round(float(np.mean(red_rets) * 100), 3) if len(red_rets) > 0 else 0,
    }


# ============================================================
# QUALITY GATE (c): SUB-PERIOD CONSISTENCY
# ============================================================
def sub_period_test(monthly_returns):
    """2010-2017 vs 2018-2026 both profitable."""
    split = pd.Timestamp("2018-01-01")

    early = [r["return"] for r in monthly_returns if r["date"] < split]
    late = [r["return"] for r in monthly_returns if r["date"] >= split]

    early_arr = np.array(early) if early else np.array([0])
    late_arr = np.array(late) if late else np.array([0])

    early_total = float(np.prod(1 + early_arr) - 1) * 100
    late_total = float(np.prod(1 + late_arr) - 1) * 100

    early_metrics = compute_metrics(early_arr)
    late_metrics = compute_metrics(late_arr)

    return {
        "early_period": "2010-2017",
        "late_period": "2018-2026",
        "early_total_ret_pct": round(early_total, 2),
        "late_total_ret_pct": round(late_total, 2),
        "early_sharpe": early_metrics["sharpe"],
        "late_sharpe": late_metrics["sharpe"],
        "early_n_months": len(early),
        "late_n_months": len(late),
        "pass": bool(early_total > 0 and late_total > 0),
    }


# ============================================================
# QUALITY GATE (d): OUTLIER REMOVAL
# ============================================================
def outlier_removal_test(returns_arr):
    """Remove top 5% of trades. Sharpe must stay > 50% of original."""
    original_metrics = compute_metrics(returns_arr)
    original_sharpe = original_metrics["sharpe"]

    # Remove top 5% returns
    cutoff = np.percentile(returns_arr, 95)
    filtered = returns_arr[returns_arr <= cutoff]

    filtered_metrics = compute_metrics(filtered)
    filtered_sharpe = filtered_metrics["sharpe"]

    retention = filtered_sharpe / max(abs(original_sharpe), 1e-9) if original_sharpe != 0 else 0

    return {
        "original_sharpe": original_sharpe,
        "filtered_sharpe": filtered_sharpe,
        "n_removed": int(len(returns_arr) - len(filtered)),
        "retention_ratio": round(float(retention), 3),
        "pass": bool(retention >= OUTLIER_SHARPE_RETENTION),
    }


# ============================================================
# QUALITY GATE (e): BENCHMARK COMPARISON
# ============================================================
def benchmark_comparison(strategy_metrics, benchmark_returns):
    """Compare strategy Sharpe to SPY buy-and-hold."""
    bench_metrics = compute_metrics(benchmark_returns)

    return {
        "strategy_sharpe": strategy_metrics["sharpe"],
        "benchmark_sharpe": bench_metrics["sharpe"],
        "strategy_cagr": strategy_metrics["cagr_pct"],
        "benchmark_cagr": bench_metrics["cagr_pct"],
        "strategy_max_dd": strategy_metrics["max_dd_pct"],
        "benchmark_max_dd": bench_metrics["max_dd_pct"],
        "beats_benchmark_sharpe": bool(strategy_metrics["sharpe"] > bench_metrics["sharpe"]),
    }


# ============================================================
# RANDOM ROTATION BASELINE
# ============================================================
def random_rotation_baseline(prices, rebalance_dates, n_random=N_RANDOM_ROTATIONS, n_pick=3):
    """
    Run N random rotations: each month randomly pick n_pick ETFs from universe.
    Return distribution of Sharpes for comparison.
    """
    rng = np.random.RandomState(RANDOM_SEED + 1)
    tradeable = [e for e in UNIVERSE if e != CASH_PROXY]
    monthly_prices = prices.resample("ME").last()

    random_sharpes = []

    for _ in range(n_random):
        rets = []
        for i in range(len(rebalance_dates) - 1):
            date = rebalance_dates[i]
            next_date = rebalance_dates[i + 1]

            if date not in monthly_prices.index or next_date not in monthly_prices.index:
                continue

            date_loc = monthly_prices.index.get_loc(date)
            next_loc = monthly_prices.index.get_loc(next_date)

            # Random selection
            picks = rng.choice(tradeable, size=min(n_pick, len(tradeable)), replace=False)
            w = 1.0 / len(picks)

            port_ret = 0.0
            for etf in picks:
                p0 = monthly_prices[etf].iloc[date_loc]
                p1 = monthly_prices[etf].iloc[next_loc]
                if p0 > 0:
                    port_ret += w * (p1 / p0 - 1)
            rets.append(port_ret)

        if rets:
            rets_arr = np.array(rets)
            metrics = compute_metrics(rets_arr)
            random_sharpes.append(metrics["sharpe"])

    return np.array(random_sharpes)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("ETF DUAL-MOMENTUM ROTATION BACKTEST v1")
    print("HC #705: All adversarial checks built-in")
    print("=" * 80)

    # Download data
    prices = download_data()

    # Compute SPY monthly returns for regime test
    spy_monthly = prices["SPY"].resample("ME").last().pct_change().dropna()

    # Compute SPY buy-and-hold monthly returns for benchmark
    spy_bh_monthly = prices["SPY"].resample("ME").last().pct_change().dropna()

    # Get rebalance dates
    rebalance_dates = get_monthly_rebalance_dates(prices)
    print(f"\nRebalance dates: {rebalance_dates[0].strftime('%Y-%m-%d')} to {rebalance_dates[-1].strftime('%Y-%m-%d')} ({len(rebalance_dates)} months)")

    # Align SPY benchmark returns to rebalance period
    spy_bench_returns = spy_bh_monthly.loc[spy_bh_monthly.index.isin(rebalance_dates)].values

    # Compute random rotation baseline
    print(f"\nComputing {N_RANDOM_ROTATIONS} random rotation baselines...")
    random_sharpes = random_rotation_baseline(prices, rebalance_dates)
    print(f"  Random rotation Sharpe: mean={np.mean(random_sharpes):.3f}, std={np.std(random_sharpes):.3f}, "
          f"80th pctile={np.percentile(random_sharpes, RANDOM_PERCENTILE_THRESHOLD):.3f}")

    # Run all strategies
    results = {}
    print("\n" + "=" * 80)
    print("STRATEGY RESULTS")
    print("=" * 80)

    for name, strategy_fn in STRATEGIES.items():
        print(f"\n{'─' * 60}")
        print(f"Strategy: {name}")
        print(f"{'─' * 60}")

        # Run backtest
        bt = run_backtest(prices, strategy_fn, rebalance_dates)
        if bt is None:
            print("  SKIP: No trades generated")
            results[name] = {"status": "no_trades"}
            continue

        metrics = compute_metrics(bt["monthly_returns"])

        print(f"  Final equity: ${bt['final_equity']:.2f} (from ${INITIAL_CAPITAL:.2f})")
        print(f"  Total return: {bt['total_return_pct']:.1f}%")
        print(f"  CAGR: {metrics['cagr_pct']:.2f}%  |  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
        print(f"  Max DD: {metrics['max_dd_pct']:.1f}%  |  WR: {metrics['win_rate']:.1f}%  |  PF: {metrics['profit_factor']:.3f}")

        # ── QUALITY GATE (a): Permutation test ──
        observed_mean = float(np.mean(bt["monthly_returns"]))
        perm_result = permutation_test_random_dates(prices, strategy_fn, observed_mean, rebalance_dates)
        gate_a = "PASS" if perm_result["pass"] else "FAIL"
        print(f"  Gate (a) Permutation: p={perm_result['p_value']:.4f} [{gate_a}]")

        # ── QUALITY GATE (b): Regime test ──
        regime_result = regime_test(bt["trade_details"], spy_monthly)
        gate_b = "PASS" if regime_result["pass"] else "FAIL"
        print(f"  Gate (b) Regime: Sharpe green={regime_result['sharpe_green']:.3f}, red={regime_result['sharpe_red']:.3f}, gap={regime_result['gap']:.3f} [{gate_b}]")

        # ── QUALITY GATE (c): Sub-period consistency ──
        subperiod_result = sub_period_test(bt["trade_details"])
        gate_c = "PASS" if subperiod_result["pass"] else "FAIL"
        print(f"  Gate (c) Sub-period: early={subperiod_result['early_total_ret_pct']:.1f}%, late={subperiod_result['late_total_ret_pct']:.1f}% [{gate_c}]")

        # ── QUALITY GATE (d): Outlier removal ──
        outlier_result = outlier_removal_test(bt["monthly_returns"])
        gate_d = "PASS" if outlier_result["pass"] else "FAIL"
        print(f"  Gate (d) Outlier: original Sharpe={outlier_result['original_sharpe']:.3f}, filtered={outlier_result['filtered_sharpe']:.3f}, retention={outlier_result['retention_ratio']:.3f} [{gate_d}]")

        # ── QUALITY GATE (e): Benchmark comparison ──
        bench_result = benchmark_comparison(metrics, spy_bench_returns)
        gate_e_label = "BEATS" if bench_result["beats_benchmark_sharpe"] else "LAGS"
        print(f"  Gate (e) Benchmark: strategy Sharpe={metrics['sharpe']:.3f} vs SPY={bench_result['benchmark_sharpe']:.3f} [{gate_e_label}]")

        # ── RANDOM ROTATION COMPARISON ──
        pctile = float(np.mean(random_sharpes <= metrics["sharpe"]) * 100)
        random_pass = pctile >= RANDOM_PERCENTILE_THRESHOLD
        random_label = "PASS" if random_pass else "FAIL"
        print(f"  Random rotation: strategy at {pctile:.1f}th percentile of random [{random_label}]")

        # Count gates passed
        gates_passed = sum([
            perm_result["pass"],
            regime_result["pass"],
            subperiod_result["pass"],
            outlier_result["pass"],
            bench_result["beats_benchmark_sharpe"],
            random_pass,
        ])

        verdict = "TRADEABLE" if gates_passed >= 5 else ("MARGINAL" if gates_passed >= 3 else "REJECT")
        print(f"\n  >>> VERDICT: {verdict} ({gates_passed}/6 gates passed)")

        results[name] = {
            "metrics": metrics,
            "final_equity": round(bt["final_equity"], 2),
            "total_return_pct": round(bt["total_return_pct"], 2),
            "n_months": bt["n_months"],
            "quality_gates": {
                "permutation_test": perm_result,
                "regime_test": regime_result,
                "sub_period_test": subperiod_result,
                "outlier_removal_test": outlier_result,
                "benchmark_comparison": bench_result,
                "random_rotation": {
                    "strategy_sharpe": metrics["sharpe"],
                    "random_80th_pctile": round(float(np.percentile(random_sharpes, RANDOM_PERCENTILE_THRESHOLD)), 3),
                    "strategy_percentile": round(pctile, 1),
                    "pass": random_pass,
                },
            },
            "gates_passed": gates_passed,
            "verdict": verdict,
        }

    # ── FINAL SUMMARY ──
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<30} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Gates':>6} {'Verdict':<10}")
    print("─" * 75)

    for name, r in results.items():
        if "metrics" not in r:
            print(f"{name:<30} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>6} {'NO TRADES':<10}")
            continue
        m = r["metrics"]
        print(f"{name:<30} {m['sharpe']:>7.3f} {m['cagr_pct']:>6.2f}% {m['max_dd_pct']:>6.1f}% {r['gates_passed']:>4}/6 {r['verdict']:<10}")

    bench_metrics = compute_metrics(spy_bench_returns)
    print(f"\n{'SPY Buy & Hold':<30} {bench_metrics['sharpe']:>7.3f} {bench_metrics['cagr_pct']:>6.2f}% {bench_metrics['max_dd_pct']:>6.1f}%")
    print(f"{'Random Rotation (avg)':<30} {np.mean(random_sharpes):>7.3f}")
    print(f"{'Random Rotation (80th pct)':<30} {np.percentile(random_sharpes, 80):>7.3f}")

    # Save report
    report = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            "universe": UNIVERSE,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "initial_capital": INITIAL_CAPITAL,
            "n_permutations": N_PERMUTATIONS,
            "n_random_rotations": N_RANDOM_ROTATIONS,
        },
        "benchmark": {
            "spy_buy_and_hold": bench_metrics,
            "random_rotation_mean_sharpe": round(float(np.mean(random_sharpes)), 3),
            "random_rotation_80th_pctile_sharpe": round(float(np.percentile(random_sharpes, 80)), 3),
        },
        "strategies": results,
    }

    report_path = OUTPUT_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    return results


if __name__ == "__main__":
    main()
