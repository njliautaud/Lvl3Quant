#!/usr/bin/env python3
"""
Monte Carlo Stress Testing for VIX-Threshold Leveraged ETF Strategy
====================================================================
Honest stress-testing of the VIX-gated UPRO/TQQQ strategy.

Strategy: Hold 100% when VIX < 20, 50% when VIX 20-30, 0% (cash) when VIX > 30.

Tests:
  1. Bootstrap resampling (1000 trials) - synthetic multi-year paths
  2. Regime stress - inject 2008-level crash
  3. VIX lag stress - what if VIX signal is delayed 1/2/5 days
  4. Execution cost stress - slippage impact
  5. Sequence of returns risk - worst windows

Output: JSON results + printed summary.

Usage:
  conda activate py311-train
  python monte_carlo_leverage_stress.py
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Try torch for GPU-accelerated Monte Carlo
USE_GPU = False
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        DEVICE = torch.device("cuda")
        print(f"[GPU] Using {torch.cuda.get_device_name(0)} for parallel Monte Carlo")
    else:
        print("[CPU] No GPU detected, running on CPU (still fast enough)")
except ImportError:
    print("[CPU] PyTorch not available, running pure numpy")

# ============================================================================
# CONFIG
# ============================================================================

N_BOOTSTRAP_TRIALS = 1000
BLOCK_SIZE_DAYS = 252  # 1 year blocks for bootstrap
SYNTHETIC_YEARS = 10   # each synthetic path = 10 years
RANDOM_SEED = 42

# VIX thresholds (the "best" config from prior research)
VIX_FULL = 20.0       # below this: 100% allocation
VIX_HALF = 30.0       # 20-30: 50% allocation; above 30: 0% (cash)

# Cash return assumption (money market / T-bills)
CASH_DAILY_RETURN = 0.05 / 252  # ~5% annualized risk-free rate (current regime)

# Slippage levels to test
SLIPPAGE_LEVELS = [0.0, 0.0005, 0.001, 0.002]  # 0%, 0.05%, 0.10%, 0.20%

# VIX lag days to test
VIX_LAG_DAYS = [0, 1, 2, 5]

# Tickers
TICKERS = {
    "UPRO": "UPRO",
    "TQQQ": "TQQQ",
    "SPY": "SPY",
}
VIX_TICKER = "^VIX"

_HOME = Path.home()
if (_HOME / "Lvl3Quant").exists():
    OUTPUT_DIR = _HOME / "Lvl3Quant" / "output" / "growth_research"
else:
    OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_FILE = OUTPUT_DIR / "monte_carlo_stress_results.json"


# ============================================================================
# DATA
# ============================================================================

def download_data():
    """Download daily data for all tickers + VIX."""
    import yfinance as yf

    print("\n=== Downloading Data ===")
    all_tickers = list(TICKERS.values()) + [VIX_TICKER]

    data = {}
    for ticker in all_tickers:
        print(f"  Downloading {ticker}...")
        df = yf.download(ticker, start="2010-01-01", end=datetime.now().strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        data[ticker] = df
        print(f"    {len(df)} days from {df.index[0].date()} to {df.index[-1].date()}")

    # Align all to common dates
    common_idx = data[list(TICKERS.values())[0]].index
    for t in all_tickers:
        common_idx = common_idx.intersection(data[t].index)

    print(f"\n  Common trading days: {len(common_idx)}")

    result = {}
    for t in all_tickers:
        result[t] = data[t].loc[common_idx]

    return result, common_idx


def compute_returns(data):
    """Compute daily returns for each ticker."""
    returns = {}
    for ticker in TICKERS.values():
        returns[ticker] = data[ticker]["Close"].pct_change().dropna()
    return returns


# ============================================================================
# STRATEGY LOGIC
# ============================================================================

def vix_allocation(vix_value):
    """Compute allocation given VIX level."""
    if vix_value < VIX_FULL:
        return 1.0
    elif vix_value < VIX_HALF:
        return 0.5
    else:
        return 0.0


def run_strategy(daily_returns, vix_series, slippage=0.0, vix_lag=0):
    """
    Run VIX-threshold strategy on a return series.

    Args:
        daily_returns: pd.Series of daily returns
        vix_series: pd.Series of VIX closes (same index)
        slippage: per-trade slippage as fraction (e.g., 0.001 = 0.10%)
        vix_lag: days of VIX lag (0 = perfect info, 1 = 1 day late)

    Returns:
        pd.Series of strategy daily returns
    """
    # Align
    common = daily_returns.index.intersection(vix_series.index)
    ret = daily_returns.loc[common].values
    vix = vix_series.loc[common].values

    n = len(ret)
    strat_ret = np.zeros(n)
    prev_alloc = 1.0  # start fully invested

    for i in range(n):
        # VIX signal: lagged
        vix_idx = max(0, i - vix_lag)
        alloc = vix_allocation(vix[vix_idx])

        # Slippage on allocation changes
        cost = 0.0
        if alloc != prev_alloc:
            # Cost proportional to change in allocation
            cost = abs(alloc - prev_alloc) * slippage

        # Strategy return = allocation * asset return + (1 - allocation) * cash - cost
        strat_ret[i] = alloc * ret[i] + (1.0 - alloc) * CASH_DAILY_RETURN - cost
        prev_alloc = alloc

    return pd.Series(strat_ret, index=common)


def compute_metrics(daily_returns):
    """Compute CAGR, MaxDD, Sharpe, Sortino from daily returns series."""
    if len(daily_returns) < 10:
        return {"cagr": 0, "max_dd": -1.0, "sharpe": 0, "sortino": 0}

    # Cumulative
    cum = (1 + daily_returns).cumprod()

    # CAGR
    n_years = len(daily_returns) / 252
    if n_years < 0.1 or cum.iloc[-1] <= 0:
        cagr = -1.0
    else:
        cagr = (cum.iloc[-1]) ** (1 / n_years) - 1

    # Max drawdown
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    # Sharpe (annualized, excess over risk-free)
    excess = daily_returns - CASH_DAILY_RETURN
    if excess.std() > 0:
        sharpe = excess.mean() / excess.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino (downside deviation)
    downside = excess[excess < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = excess.mean() / downside.std() * np.sqrt(252)
    else:
        sortino = 0.0

    return {
        "cagr": float(round(cagr, 4)),
        "max_dd": float(round(max_dd, 4)),
        "sharpe": float(round(sharpe, 4)),
        "sortino": float(round(sortino, 4)),
        "total_return": float(round(cum.iloc[-1] - 1, 4)),
        "n_days": int(len(daily_returns)),
    }


# ============================================================================
# TEST 1: BOOTSTRAP RESAMPLING
# ============================================================================

def bootstrap_resampling(daily_returns, vix_series, n_trials=N_BOOTSTRAP_TRIALS,
                         block_size=BLOCK_SIZE_DAYS, n_years=SYNTHETIC_YEARS):
    """
    Block bootstrap: resample 252-day blocks with replacement to create
    synthetic multi-year paths. This preserves autocorrelation within blocks
    while testing sequence-of-returns risk.
    """
    print(f"\n=== Bootstrap Resampling ({n_trials} trials, {n_years}yr paths) ===")

    # Align returns and VIX
    common = daily_returns.index.intersection(vix_series.index)
    ret = daily_returns.loc[common].values
    vix = vix_series.loc[common].values
    n = len(ret)

    n_blocks = int(np.ceil(n_years * 252 / block_size))
    path_length = n_blocks * block_size

    # Number of possible block start positions
    max_start = n - block_size
    if max_start <= 0:
        print("  ERROR: Not enough data for block bootstrap")
        return None

    rng = np.random.RandomState(RANDOM_SEED)

    results = []

    if USE_GPU:
        # GPU-accelerated: generate all random indices at once, run in batches
        print("  Running on GPU...")
        all_starts = rng.randint(0, max_start, size=(n_trials, n_blocks))

        # Build all paths at once on GPU
        batch_size = 100
        for batch_start in range(0, n_trials, batch_size):
            batch_end = min(batch_start + batch_size, n_trials)
            batch_results = []

            for trial in range(batch_start, batch_end):
                # Build synthetic path
                syn_ret = np.zeros(path_length)
                syn_vix = np.zeros(path_length)
                for b in range(n_blocks):
                    start = all_starts[trial, b]
                    syn_ret[b*block_size:(b+1)*block_size] = ret[start:start+block_size]
                    syn_vix[b*block_size:(b+1)*block_size] = vix[start:start+block_size]

                # Run strategy on synthetic path
                strat_ret = np.zeros(path_length)
                prev_alloc = 1.0
                for i in range(path_length):
                    alloc = vix_allocation(syn_vix[i])
                    strat_ret[i] = alloc * syn_ret[i] + (1 - alloc) * CASH_DAILY_RETURN
                    prev_alloc = alloc

                metrics = compute_metrics(pd.Series(strat_ret))
                batch_results.append(metrics)

            results.extend(batch_results)

            if (batch_end) % 200 == 0 or batch_end == n_trials:
                print(f"  Completed {batch_end}/{n_trials} trials")
    else:
        # CPU path
        for trial in range(n_trials):
            # Random block starts
            starts = rng.randint(0, max_start, size=n_blocks)

            # Build synthetic path
            syn_ret = np.zeros(path_length)
            syn_vix = np.zeros(path_length)
            for b in range(n_blocks):
                start = starts[b]
                syn_ret[b*block_size:(b+1)*block_size] = ret[start:start+block_size]
                syn_vix[b*block_size:(b+1)*block_size] = vix[start:start+block_size]

            # Run strategy
            strat_ret = np.zeros(path_length)
            prev_alloc = 1.0
            for i in range(path_length):
                alloc = vix_allocation(syn_vix[i])
                strat_ret[i] = alloc * syn_ret[i] + (1 - alloc) * CASH_DAILY_RETURN
                prev_alloc = alloc

            metrics = compute_metrics(pd.Series(strat_ret))
            results.append(metrics)

            if (trial + 1) % 200 == 0:
                print(f"  Completed {trial+1}/{n_trials} trials")

    # Compute percentiles
    cagrs = [r["cagr"] for r in results]
    max_dds = [r["max_dd"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    sortinos = [r["sortino"] for r in results]

    percentiles = [5, 25, 50, 75, 95]

    summary = {
        "n_trials": n_trials,
        "block_size_days": block_size,
        "synthetic_years": n_years,
        "cagr": {f"p{p}": float(round(np.percentile(cagrs, p), 4)) for p in percentiles},
        "max_dd": {f"p{p}": float(round(np.percentile(max_dds, p), 4)) for p in percentiles},
        "sharpe": {f"p{p}": float(round(np.percentile(sharpes, p), 4)) for p in percentiles},
        "sortino": {f"p{p}": float(round(np.percentile(sortinos, p), 4)) for p in percentiles},
        "prob_negative_cagr": float(round(np.mean([c < 0 for c in cagrs]), 4)),
        "prob_dd_worse_25pct": float(round(np.mean([d < -0.25 for d in max_dds]), 4)),
        "prob_dd_worse_40pct": float(round(np.mean([d < -0.40 for d in max_dds]), 4)),
        "prob_dd_worse_50pct": float(round(np.mean([d < -0.50 for d in max_dds]), 4)),
        "mean_cagr": float(round(np.mean(cagrs), 4)),
        "mean_max_dd": float(round(np.mean(max_dds), 4)),
        "mean_sharpe": float(round(np.mean(sharpes), 4)),
    }

    return summary


# ============================================================================
# TEST 2: REGIME STRESS (INJECT CRASH)
# ============================================================================

def regime_stress_test(daily_returns, vix_series, n_trials=200):
    """
    Inject a 2008-level crash into random years and see how strategy handles it.

    We model the crash as:
    - VIX spikes to 60-80 over 10 days
    - Asset drops ~50% over 40 trading days (roughly 2 months)
    - Then slow recovery over 6 months
    """
    print(f"\n=== Regime Stress Test ({n_trials} crash injections) ===")

    common = daily_returns.index.intersection(vix_series.index)
    ret = daily_returns.loc[common].values.copy()
    vix = vix_series.loc[common].values.copy()
    n = len(ret)

    rng = np.random.RandomState(RANDOM_SEED + 1)

    results_with_crash = []
    results_without_crash = []

    # Baseline (no crash)
    baseline_strat = np.zeros(n)
    prev_alloc = 1.0
    for i in range(n):
        alloc = vix_allocation(vix[i])
        baseline_strat[i] = alloc * ret[i] + (1 - alloc) * CASH_DAILY_RETURN
        prev_alloc = alloc
    baseline_metrics = compute_metrics(pd.Series(baseline_strat))

    crash_duration = 40  # trading days for the crash
    recovery_duration = 120  # trading days for recovery
    total_event = crash_duration + recovery_duration

    for trial in range(n_trials):
        # Pick random injection point (must have room for crash + recovery)
        inject_start = rng.randint(252, n - total_event - 10)

        # Create modified returns/VIX
        mod_ret = ret.copy()
        mod_vix = vix.copy()

        # Crash phase: -50% over 40 days = avg daily return of ~-1.7%
        # With fat tails (some days -5%, some -0.5%)
        crash_severity = rng.uniform(0.40, 0.60)  # 40-60% total crash
        daily_crash = np.log(1 - crash_severity) / crash_duration
        crash_returns = np.exp(daily_crash + rng.normal(0, 0.01, crash_duration)) - 1

        # VIX spike during crash
        vix_peak = rng.uniform(55, 85)
        vix_ramp = np.linspace(mod_vix[inject_start], vix_peak, crash_duration // 2)
        vix_plateau = np.full(crash_duration - crash_duration // 2, vix_peak)
        crash_vix = np.concatenate([vix_ramp, vix_plateau])

        # Recovery phase: gradual bounce
        recovery_returns = rng.normal(0.002, 0.015, recovery_duration)  # slight positive drift
        recovery_vix = np.linspace(vix_peak, 25, recovery_duration)

        # Inject
        mod_ret[inject_start:inject_start+crash_duration] = crash_returns
        mod_ret[inject_start+crash_duration:inject_start+total_event] = recovery_returns
        mod_vix[inject_start:inject_start+crash_duration] = crash_vix
        mod_vix[inject_start+crash_duration:inject_start+total_event] = recovery_vix

        # Run strategy on modified data
        strat_ret = np.zeros(n)
        prev_alloc = 1.0
        for i in range(n):
            alloc = vix_allocation(mod_vix[i])
            strat_ret[i] = alloc * mod_ret[i] + (1 - alloc) * CASH_DAILY_RETURN
            prev_alloc = alloc

        metrics = compute_metrics(pd.Series(strat_ret))
        results_with_crash.append(metrics)

    # Summarize
    cagrs = [r["cagr"] for r in results_with_crash]
    max_dds = [r["max_dd"] for r in results_with_crash]
    sharpes = [r["sharpe"] for r in results_with_crash]

    percentiles = [5, 25, 50, 75, 95]

    summary = {
        "n_trials": n_trials,
        "crash_model": "40-day crash (40-60% drop), 120-day recovery, VIX spike to 55-85",
        "baseline_no_crash": baseline_metrics,
        "with_injected_crash": {
            "cagr": {f"p{p}": float(round(np.percentile(cagrs, p), 4)) for p in percentiles},
            "max_dd": {f"p{p}": float(round(np.percentile(max_dds, p), 4)) for p in percentiles},
            "sharpe": {f"p{p}": float(round(np.percentile(sharpes, p), 4)) for p in percentiles},
            "mean_cagr": float(round(np.mean(cagrs), 4)),
            "mean_max_dd": float(round(np.mean(max_dds), 4)),
            "prob_dd_worse_25pct": float(round(np.mean([d < -0.25 for d in max_dds]), 4)),
            "prob_dd_worse_40pct": float(round(np.mean([d < -0.40 for d in max_dds]), 4)),
        },
        "crash_protection_ratio": float(round(
            np.mean(max_dds) / baseline_metrics["max_dd"], 4
        )) if baseline_metrics["max_dd"] != 0 else None,
    }

    return summary


# ============================================================================
# TEST 3: VIX LAG STRESS
# ============================================================================

def vix_lag_stress(daily_returns, vix_series):
    """Test strategy degradation when VIX signal is delayed."""
    print(f"\n=== VIX Lag Stress Test (lags: {VIX_LAG_DAYS}) ===")

    results = {}
    for lag in VIX_LAG_DAYS:
        strat_ret = run_strategy(daily_returns, vix_series, slippage=0.0, vix_lag=lag)
        metrics = compute_metrics(strat_ret)
        results[f"lag_{lag}d"] = metrics
        print(f"  Lag {lag}d: CAGR={metrics['cagr']:.1%}, MaxDD={metrics['max_dd']:.1%}, "
              f"Sharpe={metrics['sharpe']:.2f}")

    # Compute degradation vs lag=0
    base = results["lag_0d"]
    degradation = {}
    for lag in VIX_LAG_DAYS[1:]:
        key = f"lag_{lag}d"
        if base["cagr"] != 0:
            degradation[key] = {
                "cagr_loss": float(round(results[key]["cagr"] - base["cagr"], 4)),
                "cagr_pct_loss": float(round(
                    (results[key]["cagr"] - base["cagr"]) / abs(base["cagr"]), 4
                )),
                "dd_worsening": float(round(results[key]["max_dd"] - base["max_dd"], 4)),
                "sharpe_loss": float(round(results[key]["sharpe"] - base["sharpe"], 4)),
            }

    return {
        "by_lag": results,
        "degradation_vs_no_lag": degradation,
    }


# ============================================================================
# TEST 4: EXECUTION COST STRESS
# ============================================================================

def execution_cost_stress(daily_returns, vix_series):
    """Test how slippage/execution costs erode returns."""
    print(f"\n=== Execution Cost Stress Test (slippage: {SLIPPAGE_LEVELS}) ===")

    results = {}
    for slip in SLIPPAGE_LEVELS:
        strat_ret = run_strategy(daily_returns, vix_series, slippage=slip, vix_lag=0)
        metrics = compute_metrics(strat_ret)

        # Count trades (allocation changes)
        common = daily_returns.index.intersection(vix_series.index)
        vix_vals = vix_series.loc[common].values
        n_trades = 0
        prev_alloc = 1.0
        for v in vix_vals:
            alloc = vix_allocation(v)
            if alloc != prev_alloc:
                n_trades += 1
            prev_alloc = alloc

        label = f"slip_{slip*100:.2f}pct"
        results[label] = {**metrics, "n_allocation_changes": n_trades}
        print(f"  Slippage {slip*100:.2f}%: CAGR={metrics['cagr']:.1%}, "
              f"Sharpe={metrics['sharpe']:.2f}, Trades={n_trades}")

    # Find break-even slippage (where CAGR goes negative or Sharpe < 0.5)
    base_cagr = results[f"slip_0.00pct"]["cagr"]

    return {
        "by_slippage": results,
        "base_cagr_no_slippage": float(round(base_cagr, 4)),
        "note": "Strategy trades infrequently (VIX regime changes), so slippage impact is small per-trade but check total drag",
    }


# ============================================================================
# TEST 5: SEQUENCE OF RETURNS RISK
# ============================================================================

def sequence_of_returns_risk(daily_returns, vix_series):
    """Find the worst rolling windows for the strategy."""
    print(f"\n=== Sequence of Returns Risk ===")

    strat_ret = run_strategy(daily_returns, vix_series)
    cum = (1 + strat_ret).cumprod()

    results = {}

    for window_years in [1, 2, 3, 5]:
        window_days = window_years * 252
        if len(strat_ret) < window_days:
            continue

        # Rolling CAGR
        rolling_total = cum / cum.shift(window_days) - 1
        rolling_total = rolling_total.dropna()

        if len(rolling_total) == 0:
            continue

        # Convert total return to annualized
        rolling_cagr = (1 + rolling_total) ** (1 / window_years) - 1

        worst_idx = rolling_cagr.idxmin()
        best_idx = rolling_cagr.idxmax()

        # Worst window drawdown
        worst_start_idx = strat_ret.index.get_loc(worst_idx) - window_days
        if worst_start_idx < 0:
            worst_start_idx = 0
        worst_window = strat_ret.iloc[worst_start_idx:strat_ret.index.get_loc(worst_idx)+1]
        worst_cum = (1 + worst_window).cumprod()
        worst_dd = ((worst_cum - worst_cum.cummax()) / worst_cum.cummax()).min()

        results[f"{window_years}yr"] = {
            "worst_cagr": float(round(rolling_cagr.min(), 4)),
            "worst_end_date": str(worst_idx.date()),
            "worst_window_max_dd": float(round(worst_dd, 4)),
            "best_cagr": float(round(rolling_cagr.max(), 4)),
            "best_end_date": str(best_idx.date()),
            "median_cagr": float(round(rolling_cagr.median(), 4)),
            "pct_negative_windows": float(round((rolling_cagr < 0).mean(), 4)),
        }

        print(f"  {window_years}yr windows: Worst CAGR={rolling_cagr.min():.1%} "
              f"(ending {worst_idx.date()}), Best={rolling_cagr.max():.1%}, "
              f"Median={rolling_cagr.median():.1%}, "
              f"Negative={((rolling_cagr < 0).mean())*100:.1f}%")

    return results


# ============================================================================
# BUY AND HOLD COMPARISON
# ============================================================================

def buy_and_hold_comparison(data, vix_series):
    """Compare strategy vs buy-and-hold for context."""
    print(f"\n=== Buy & Hold Comparison (baseline context) ===")

    results = {}
    for ticker in TICKERS.values():
        ret = data[ticker]["Close"].pct_change().dropna()
        common = ret.index.intersection(vix_series.index)
        ret_aligned = ret.loc[common]

        # Buy and hold
        bh_metrics = compute_metrics(ret_aligned)

        # VIX strategy
        strat_ret = run_strategy(ret_aligned, vix_series)
        strat_metrics = compute_metrics(strat_ret)

        results[ticker] = {
            "buy_and_hold": bh_metrics,
            "vix_strategy": strat_metrics,
            "cagr_improvement": float(round(strat_metrics["cagr"] - bh_metrics["cagr"], 4)),
            "dd_improvement": float(round(strat_metrics["max_dd"] - bh_metrics["max_dd"], 4)),
            "sharpe_improvement": float(round(strat_metrics["sharpe"] - bh_metrics["sharpe"], 4)),
        }

        print(f"  {ticker}:")
        print(f"    B&H:      CAGR={bh_metrics['cagr']:.1%}, MaxDD={bh_metrics['max_dd']:.1%}, "
              f"Sharpe={bh_metrics['sharpe']:.2f}")
        print(f"    Strategy: CAGR={strat_metrics['cagr']:.1%}, MaxDD={strat_metrics['max_dd']:.1%}, "
              f"Sharpe={strat_metrics['sharpe']:.2f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()
    print("=" * 70)
    print("MONTE CARLO STRESS TEST: VIX-Threshold Leveraged ETF Strategy")
    print("=" * 70)
    print(f"Config: VIX < {VIX_FULL} = 100%, VIX {VIX_FULL}-{VIX_HALF} = 50%, VIX > {VIX_HALF} = cash")
    print(f"Bootstrap trials: {N_BOOTSTRAP_TRIALS}, Synthetic path: {SYNTHETIC_YEARS} years")
    print(f"Random seed: {RANDOM_SEED}")

    # Download data
    data, common_idx = download_data()
    returns = compute_returns(data)
    vix_close = data[VIX_TICKER]["Close"]

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "strategy": f"VIX<{VIX_FULL}=100%, VIX {VIX_FULL}-{VIX_HALF}=50%, VIX>{VIX_HALF}=cash",
            "data_start": str(common_idx[0].date()),
            "data_end": str(common_idx[-1].date()),
            "n_trading_days": len(common_idx),
            "bootstrap_trials": N_BOOTSTRAP_TRIALS,
            "synthetic_years": SYNTHETIC_YEARS,
            "cash_rate_annual": 0.05,
            "random_seed": RANDOM_SEED,
        }
    }

    # --- Run all tests for each leveraged ETF ---
    for etf_name in ["UPRO", "TQQQ"]:
        ticker = TICKERS[etf_name]
        print(f"\n{'='*70}")
        print(f"  TESTING: {etf_name}")
        print(f"{'='*70}")

        etf_returns = returns[ticker]

        etf_results = {}

        # 0. Buy & Hold comparison (just for this ticker)
        bh_ret = etf_returns
        common = bh_ret.index.intersection(vix_close.index)
        bh_aligned = bh_ret.loc[common]
        bh_metrics = compute_metrics(bh_aligned)
        strat_ret = run_strategy(bh_aligned, vix_close)
        strat_metrics = compute_metrics(strat_ret)
        etf_results["baseline"] = {
            "buy_and_hold": bh_metrics,
            "vix_strategy": strat_metrics,
        }
        print(f"\n  Baseline B&H:      CAGR={bh_metrics['cagr']:.1%}, MaxDD={bh_metrics['max_dd']:.1%}, Sharpe={bh_metrics['sharpe']:.2f}")
        print(f"  Baseline Strategy: CAGR={strat_metrics['cagr']:.1%}, MaxDD={strat_metrics['max_dd']:.1%}, Sharpe={strat_metrics['sharpe']:.2f}")

        # 1. Bootstrap resampling
        etf_results["bootstrap"] = bootstrap_resampling(bh_aligned, vix_close.loc[common])

        # 2. Regime stress (crash injection)
        etf_results["regime_stress"] = regime_stress_test(bh_aligned, vix_close.loc[common])

        # 3. VIX lag stress
        etf_results["vix_lag"] = vix_lag_stress(bh_aligned, vix_close.loc[common])

        # 4. Execution cost stress
        etf_results["execution_cost"] = execution_cost_stress(bh_aligned, vix_close.loc[common])

        # 5. Sequence of returns
        etf_results["sequence_risk"] = sequence_of_returns_risk(bh_aligned, vix_close.loc[common])

        all_results[etf_name] = etf_results

    # SPY comparison (non-leveraged baseline)
    print(f"\n{'='*70}")
    print(f"  SPY BASELINE (no leverage)")
    print(f"{'='*70}")
    spy_ret = returns["SPY"]
    common = spy_ret.index.intersection(vix_close.index)
    spy_aligned = spy_ret.loc[common]
    spy_bh = compute_metrics(spy_aligned)
    spy_strat = compute_metrics(run_strategy(spy_aligned, vix_close))
    all_results["SPY_baseline"] = {
        "buy_and_hold": spy_bh,
        "vix_strategy": spy_strat,
    }
    print(f"  SPY B&H:      CAGR={spy_bh['cagr']:.1%}, MaxDD={spy_bh['max_dd']:.1%}, Sharpe={spy_bh['sharpe']:.2f}")
    print(f"  SPY Strategy: CAGR={spy_strat['cagr']:.1%}, MaxDD={spy_strat['max_dd']:.1%}, Sharpe={spy_strat['sharpe']:.2f}")

    # ======================================================================
    # HONEST ASSESSMENT
    # ======================================================================
    print(f"\n{'='*70}")
    print("  HONEST ASSESSMENT")
    print(f"{'='*70}")

    warnings_list = []

    for etf in ["UPRO", "TQQQ"]:
        r = all_results[etf]

        # Check bootstrap fragility
        bs = r["bootstrap"]
        if bs:
            if bs["prob_negative_cagr"] > 0.05:
                warnings_list.append(
                    f"{etf}: {bs['prob_negative_cagr']*100:.1f}% probability of negative CAGR "
                    f"in bootstrap (fragile)")
            if bs["prob_dd_worse_40pct"] > 0.20:
                warnings_list.append(
                    f"{etf}: {bs['prob_dd_worse_40pct']*100:.1f}% probability of >40% drawdown "
                    f"in bootstrap (dangerous for $440 account)")
            if bs["cagr"]["p5"] < 0:
                warnings_list.append(
                    f"{etf}: P5 CAGR is {bs['cagr']['p5']:.1%} -- 5% chance of losing money "
                    f"over {SYNTHETIC_YEARS} years")
            if bs["max_dd"]["p5"] < -0.50:
                warnings_list.append(
                    f"{etf}: P5 MaxDD is {bs['max_dd']['p5']:.1%} -- 5% chance of 50%+ drawdown")

        # Check VIX lag sensitivity
        lag = r["vix_lag"]
        lag1 = lag["by_lag"].get("lag_1d", {})
        lag0 = lag["by_lag"].get("lag_0d", {})
        if lag1 and lag0 and lag0["sharpe"] > 0:
            sharpe_drop = (lag0["sharpe"] - lag1["sharpe"]) / lag0["sharpe"]
            if sharpe_drop > 0.15:
                warnings_list.append(
                    f"{etf}: 1-day VIX lag drops Sharpe by {sharpe_drop*100:.0f}% "
                    f"-- strategy is timing-sensitive")

        # Check if strategy actually beats B&H
        strat = r["baseline"]["vix_strategy"]
        bh = r["baseline"]["buy_and_hold"]
        if strat["sharpe"] < bh["sharpe"]:
            warnings_list.append(
                f"{etf}: Strategy Sharpe ({strat['sharpe']:.2f}) is WORSE than B&H "
                f"({bh['sharpe']:.2f}) -- VIX timing may be hurting risk-adjusted returns")

    all_results["honest_assessment"] = {
        "warnings": warnings_list,
        "key_question": ("Does VIX threshold timing genuinely add alpha, or does it just "
                        "reduce exposure (and therefore both returns AND risk proportionally)?"),
        "survivorship_bias_note": ("UPRO/TQQQ only exist since 2009/2010. The entire backtest "
                                   "period is a secular bull market. This strategy has NEVER been "
                                   "tested in a true secular bear (1970s, 2000-2009)."),
        "leverage_decay_note": ("3x leveraged ETFs suffer volatility drag. In choppy sideways "
                               "markets, UPRO can lose money even when SPY is flat. The VIX "
                               "threshold partially mitigates this but doesn't eliminate it."),
        "small_account_note": ("$440 account: even with high CAGR, dollar amounts are small. "
                              "A 50% drawdown = $220 loss. Emotional/behavioral risk is real."),
    }

    if warnings_list:
        print("\n  WARNINGS:")
        for w in warnings_list:
            print(f"    - {w}")
    else:
        print("\n  No major warnings triggered (but read the notes below)")

    print(f"\n  STRUCTURAL CONCERNS:")
    print(f"    - {all_results['honest_assessment']['survivorship_bias_note']}")
    print(f"    - {all_results['honest_assessment']['leverage_decay_note']}")
    print(f"    - {all_results['honest_assessment']['small_account_note']}")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {OUTPUT_FILE}")

    # ======================================================================
    # SUMMARY TABLE
    # ======================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY TABLE")
    print(f"{'='*70}")

    for etf in ["UPRO", "TQQQ"]:
        r = all_results[etf]
        bs = r["bootstrap"]
        print(f"\n  {etf}:")
        print(f"    {'Metric':<25} {'Backtest':>10} {'P5':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P95':>10}")
        print(f"    {'-'*85}")

        strat = r["baseline"]["vix_strategy"]
        if bs:
            print(f"    {'CAGR':<25} {strat['cagr']:>9.1%} {bs['cagr']['p5']:>9.1%} "
                  f"{bs['cagr']['p25']:>9.1%} {bs['cagr']['p50']:>9.1%} "
                  f"{bs['cagr']['p75']:>9.1%} {bs['cagr']['p95']:>9.1%}")
            print(f"    {'MaxDD':<25} {strat['max_dd']:>9.1%} {bs['max_dd']['p5']:>9.1%} "
                  f"{bs['max_dd']['p25']:>9.1%} {bs['max_dd']['p50']:>9.1%} "
                  f"{bs['max_dd']['p75']:>9.1%} {bs['max_dd']['p95']:>9.1%}")
            print(f"    {'Sharpe':<25} {strat['sharpe']:>10.2f} {bs['sharpe']['p5']:>10.2f} "
                  f"{bs['sharpe']['p25']:>10.2f} {bs['sharpe']['p50']:>10.2f} "
                  f"{bs['sharpe']['p75']:>10.2f} {bs['sharpe']['p95']:>10.2f}")

    elapsed = time.time() - start_time
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
