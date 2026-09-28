"""
ETF Cross-Sectional Momentum v3 — FULLY VECTORIZED
====================================================
HC #0   : Sliding walk-forward (36-month train, 6-month OOT, sliding)
HC #428 : Regime-agnostic validation (R1 gap check)
HC #694 : Commission-free (Robinhood) — 0 brokerage commissions
HC #705 : Adversarial checks (permutation, sub-period, outlier removal, R1, data sanity)

Key difference from v2: NO day-by-day Python loops.
All returns, rankings, and signals are precomputed as DataFrames.
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUTPUT_DIR / "etf_cross_momentum_v3_results.json"

START = "2007-01-01"
END = "2026-07-16"

TICKERS = [
    "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "HYG", "GLD",
]

# Strategy parameter grid
LOOKBACKS = {"3m": 63, "6m": 126, "12m": 252}
TOP_K = [3, 5]
FILTERS = ["none", "ma200", "dual"]  # dual = only buy if trailing return > 0

# Walk-forward
WF_TRAIN_DAYS = 756   # ~36 months
WF_TEST_DAYS = 126    # ~6 months
REBAL_FREQ = 21       # monthly rebalance

# Adversarial
N_PERMUTATIONS = 200


def download_data() -> pd.DataFrame:
    """Download adjusted close prices for all ETFs."""
    print(f"Downloading {len(TICKERS)} ETFs from {START} to {END}...")
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns (Price, Ticker) for multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = TICKERS[:1]
    prices = prices[TICKERS]  # ensure column order
    prices = prices.dropna(how="all")
    print(f"  Got {len(prices)} trading days, {prices.shape[1]} ETFs")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    # Forward-fill small gaps, drop leading NaNs
    prices = prices.ffill().dropna()
    print(f"  After cleanup: {len(prices)} days with full data")
    return prices


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().iloc[1:]


def compute_trailing_returns(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Vectorized trailing total return over lookback days."""
    return prices / prices.shift(lookback) - 1.0


def compute_ma_filter(prices: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Boolean: price > 200-day MA."""
    return prices > prices.rolling(window).mean()


def build_signal_matrix(
    prices: pd.DataFrame,
    daily_rets: pd.DataFrame,
    lookback: int,
    top_k: int,
    filt: str,
    rebal_freq: int = REBAL_FREQ,
) -> pd.DataFrame:
    """
    Build a fully vectorized signal matrix (weights) for the strategy.
    Returns a DataFrame aligned to daily_rets index with weights per ETF per day.
    """
    n_etfs = prices.shape[1]
    trailing = compute_trailing_returns(prices, lookback)

    # Align to daily_rets index
    trailing = trailing.reindex(daily_rets.index)

    # Determine rebalance dates: every rebal_freq days
    rebal_mask = np.zeros(len(daily_rets), dtype=bool)
    rebal_mask[::rebal_freq] = True
    rebal_dates = daily_rets.index[rebal_mask]

    # At each rebalance date, rank ETFs and pick top K
    # Build weights only at rebalance dates, then forward-fill
    weights_at_rebal = pd.DataFrame(0.0, index=rebal_dates, columns=daily_rets.columns)

    # Vectorized ranking at all rebalance dates at once
    trail_at_rebal = trailing.loc[rebal_dates]

    # Apply filters
    if filt == "ma200":
        ma_filter = compute_ma_filter(prices, 200).reindex(daily_rets.index)
        ma_at_rebal = ma_filter.loc[rebal_dates]
        # Set trailing return to NaN where filter fails (won't be ranked)
        trail_at_rebal = trail_at_rebal.where(ma_at_rebal)
    elif filt == "dual":
        # Only buy if trailing return > 0
        trail_at_rebal = trail_at_rebal.where(trail_at_rebal > 0)

    # Rank across columns (axis=1), ascending=False so highest return = rank 1
    # Use method='first' to break ties deterministically
    ranks = trail_at_rebal.rank(axis=1, ascending=False, method="first")

    # Select top K: weight = 1/K if rank <= K and not NaN
    valid = trail_at_rebal.notna()
    selected = (ranks <= top_k) & valid

    # Count how many are actually selected per row (may be < K if filters exclude)
    n_selected = selected.sum(axis=1).replace(0, np.nan)

    # Equal weight among selected
    weights_at_rebal = selected.astype(float).div(n_selected, axis=0).fillna(0.0)

    # Forward-fill weights to all trading days
    weights_full = weights_at_rebal.reindex(daily_rets.index).ffill().fillna(0.0)

    return weights_full


def compute_strategy_returns(weights: pd.DataFrame, daily_rets: pd.DataFrame) -> pd.Series:
    """Portfolio return = sum of (weight * daily_return) across ETFs."""
    return (weights * daily_rets).sum(axis=1)


def sharpe(returns: pd.Series, ann: float = 252) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 20 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ann))


def sortino(returns: pd.Series, ann: float = 252) -> float:
    """Annualized Sortino ratio."""
    if len(returns) < 20:
        return 0.0
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return float(returns.mean() / downside * np.sqrt(ann))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from cumulative returns."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def profit_factor(returns: pd.Series) -> float:
    """Gross profits / gross losses."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate."""
    if len(returns) < 2:
        return 0.0
    cum = (1 + returns).prod()
    years = len(returns) / 252
    if years <= 0 or cum <= 0:
        return 0.0
    return float(cum ** (1 / years) - 1)


def compute_metrics(returns: pd.Series) -> dict:
    """Compute standard performance metrics."""
    return {
        "sharpe": round(sharpe(returns), 3),
        "sortino": round(sortino(returns), 3),
        "cagr_pct": round(cagr(returns) * 100, 2),
        "max_dd_pct": round(max_drawdown(returns) * 100, 2),
        "profit_factor": round(profit_factor(returns), 3),
        "win_rate_pct": round((returns > 0).mean() * 100, 1),
        "n_days": len(returns),
        "total_return_pct": round(((1 + returns).prod() - 1) * 100, 2),
    }


def walk_forward_backtest(prices: pd.DataFrame, daily_rets: pd.DataFrame) -> dict:
    """
    Sliding walk-forward: 36mo train, 6mo test.
    On train: pick best (lookback, top_k, filter) by Sharpe.
    Apply to test. Concatenate all OOT returns.
    """
    n = len(daily_rets)
    total_window = WF_TRAIN_DAYS + WF_TEST_DAYS

    if n < total_window:
        print(f"  WARNING: Not enough data for walk-forward ({n} < {total_window})")
        return {}

    # Precompute ALL signal matrices for all param combos (the key speedup)
    print("  Precomputing all signal matrices...")
    t0 = time.time()
    all_weights = {}
    for lb_name, lb_days in LOOKBACKS.items():
        for k in TOP_K:
            for f in FILTERS:
                key = (lb_name, k, f)
                all_weights[key] = build_signal_matrix(prices, daily_rets, lb_days, k, f)
    print(f"  Precomputed {len(all_weights)} signal matrices in {time.time()-t0:.1f}s")

    # Precompute all strategy returns for all combos
    all_strat_rets = {}
    for key, w in all_weights.items():
        all_strat_rets[key] = compute_strategy_returns(w, daily_rets)

    # Walk-forward windows
    oot_returns_list = []
    oot_params_chosen = []
    window_starts = list(range(0, n - total_window + 1, WF_TEST_DAYS))

    print(f"  Running {len(window_starts)} walk-forward windows...")
    for start in window_starts:
        train_slice = slice(start, start + WF_TRAIN_DAYS)
        test_slice = slice(start + WF_TRAIN_DAYS, start + total_window)

        # Find best params on training window
        best_sharpe = -999
        best_key = None
        for key, sr in all_strat_rets.items():
            train_rets = sr.iloc[train_slice]
            s = sharpe(train_rets)
            if s > best_sharpe:
                best_sharpe = s
                best_key = key

        # Apply best params to test window
        test_rets = all_strat_rets[best_key].iloc[test_slice]
        oot_returns_list.append(test_rets)
        oot_params_chosen.append(best_key)

    # Concatenate OOT returns
    oot_returns = pd.concat(oot_returns_list)
    # Remove any duplicate indices (overlapping windows shouldn't happen with sliding, but safety)
    oot_returns = oot_returns[~oot_returns.index.duplicated(keep="first")]

    # Compute OOT metrics
    oot_metrics = compute_metrics(oot_returns)

    # Parameter stability: how often each combo was chosen
    from collections import Counter
    param_counts = Counter(oot_params_chosen)
    most_common = param_counts.most_common(3)

    print(f"\n  Walk-Forward OOT Results ({len(oot_returns)} days):")
    print(f"    Sharpe:  {oot_metrics['sharpe']}")
    print(f"    Sortino: {oot_metrics['sortino']}")
    print(f"    CAGR:    {oot_metrics['cagr_pct']}%")
    print(f"    Max DD:  {oot_metrics['max_dd_pct']}%")
    print(f"    PF:      {oot_metrics['profit_factor']}")
    print(f"    WR:      {oot_metrics['win_rate_pct']}%")
    print(f"    Total:   {oot_metrics['total_return_pct']}%")
    print(f"    Most chosen params: {most_common}")

    return {
        "oot_metrics": oot_metrics,
        "param_stability": {str(k): v for k, v in param_counts.items()},
        "most_common_params": [{"params": str(k), "count": v} for k, v in most_common],
        "n_windows": len(window_starts),
        "oot_returns": oot_returns,  # kept for adversarial checks, removed before JSON save
    }


def fixed_param_backtests(prices: pd.DataFrame, daily_rets: pd.DataFrame) -> dict:
    """Run all fixed-parameter backtests (no walk-forward) for comparison."""
    results = {}
    for lb_name, lb_days in LOOKBACKS.items():
        for k in TOP_K:
            for f in FILTERS:
                key = f"{lb_name}_top{k}_{f}"
                weights = build_signal_matrix(prices, daily_rets, lb_days, k, f)
                strat_rets = compute_strategy_returns(weights, daily_rets)
                metrics = compute_metrics(strat_rets)
                results[key] = metrics
    return results


# ---------------------------------------------------------------------------
# ADVERSARIAL CHECKS (HC #705)
# ---------------------------------------------------------------------------

def permutation_test(daily_rets: pd.DataFrame, oot_returns: pd.Series,
                     n_perms: int = N_PERMUTATIONS) -> dict:
    """
    Shuffle which ETFs are selected each rebalance period.
    Compare real Sharpe to distribution of random Sharpes.
    """
    print(f"\n  Permutation test ({n_perms} shuffles)...")
    real_sharpe = sharpe(oot_returns)

    # For speed: use a simple random top-K selection with the most common params
    # We'll randomly permute columns at each rebalance date
    n_etfs = daily_rets.shape[1]
    n_days = len(daily_rets)
    rets_arr = daily_rets.values  # (n_days, n_etfs)

    # Use K=4 (average of 3 and 5) for permutation baseline
    k = 4
    rebal_indices = list(range(0, n_days, REBAL_FREQ))

    perm_sharpes = []
    rng = np.random.default_rng(42)

    for _ in range(n_perms):
        weights = np.zeros((n_days, n_etfs))
        for ri, rb in enumerate(rebal_indices):
            end = rebal_indices[ri + 1] if ri + 1 < len(rebal_indices) else n_days
            chosen = rng.choice(n_etfs, size=min(k, n_etfs), replace=False)
            weights[rb:end, chosen] = 1.0 / k
        perm_rets = (weights * rets_arr).sum(axis=1)
        # Only use the OOT portion
        perm_rets_oot = perm_rets[-len(oot_returns):]
        s = np.mean(perm_rets_oot) / (np.std(perm_rets_oot) + 1e-10) * np.sqrt(252)
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())

    print(f"    Real Sharpe: {real_sharpe:.3f}")
    print(f"    Perm mean:   {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}")
    print(f"    p-value:     {p_value:.3f}")

    return {
        "real_sharpe": round(real_sharpe, 3),
        "perm_mean_sharpe": round(float(perm_sharpes.mean()), 3),
        "perm_std_sharpe": round(float(perm_sharpes.std()), 3),
        "p_value": round(p_value, 3),
        "significant_at_05": p_value < 0.05,
    }


def regime_test(oot_returns: pd.Series, spy_rets: pd.Series) -> dict:
    """
    R1 regime-agnostic test: classify days by SPY direction.
    Sharpe gap must be < 0.50.
    """
    print("\n  Regime test (SPY green/red/flat)...")
    # Align
    common = oot_returns.index.intersection(spy_rets.index)
    oot_c = oot_returns.loc[common]
    spy_c = spy_rets.loc[common]

    green = spy_c > 0.001
    red = spy_c < -0.001
    flat = ~green & ~red

    regimes = {"green": oot_c[green], "red": oot_c[red], "flat": oot_c[flat]}
    regime_sharpes = {}
    for name, rets in regimes.items():
        s = sharpe(rets) if len(rets) > 20 else np.nan
        regime_sharpes[name] = round(s, 3)
        print(f"    {name}: Sharpe={s:.3f}, n={len(rets)}")

    # R1 gap check
    s_green = regime_sharpes.get("green", 0)
    s_red = regime_sharpes.get("red", 0)
    max_abs = max(abs(s_green), abs(s_red), 0.001)
    gap = abs(s_green - s_red) / max_abs
    passed = gap < 0.50

    print(f"    Regime gap: {gap:.3f} ({'PASS' if passed else 'FAIL'} < 0.50)")

    return {
        "regime_sharpes": regime_sharpes,
        "regime_gap": round(gap, 3),
        "r1_passed": passed,
    }


def sub_period_test(oot_returns: pd.Series) -> dict:
    """First half vs second half consistency."""
    print("\n  Sub-period test (first half vs second half)...")
    mid = len(oot_returns) // 2
    first_half = oot_returns.iloc[:mid]
    second_half = oot_returns.iloc[mid:]

    m1 = compute_metrics(first_half)
    m2 = compute_metrics(second_half)

    print(f"    First half:  Sharpe={m1['sharpe']}, CAGR={m1['cagr_pct']}%")
    print(f"    Second half: Sharpe={m2['sharpe']}, CAGR={m2['cagr_pct']}%")

    # Check if both halves are profitable
    both_profitable = m1["cagr_pct"] > 0 and m2["cagr_pct"] > 0

    return {
        "first_half": m1,
        "second_half": m2,
        "both_profitable": both_profitable,
    }


def outlier_removal_test(oot_returns: pd.Series) -> dict:
    """Remove top 5% of daily returns and recompute metrics."""
    print("\n  Outlier removal test (remove top 5% days)...")
    threshold = oot_returns.quantile(0.95)
    filtered = oot_returns[oot_returns <= threshold]
    metrics_full = compute_metrics(oot_returns)
    metrics_filtered = compute_metrics(filtered)

    print(f"    Full:     Sharpe={metrics_full['sharpe']}, CAGR={metrics_full['cagr_pct']}%")
    print(f"    Filtered: Sharpe={metrics_filtered['sharpe']}, CAGR={metrics_filtered['cagr_pct']}%")

    # Strategy should still be profitable after removing best days
    robust = metrics_filtered["sharpe"] > 0

    return {
        "full_metrics": metrics_full,
        "filtered_metrics": metrics_filtered,
        "still_profitable_after_removal": robust,
    }


# ---------------------------------------------------------------------------
# BENCHMARK
# ---------------------------------------------------------------------------

def compute_benchmark(daily_rets: pd.DataFrame, oot_returns: pd.Series) -> dict:
    """SPY buy-and-hold over the same OOT period."""
    spy_oot = daily_rets["SPY"].reindex(oot_returns.index).dropna()
    return compute_metrics(spy_oot)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # 1. Download data
    prices = download_data()
    daily_rets = compute_daily_returns(prices)

    # 2. Fixed-parameter backtests (full sample, for reference)
    print("\n=== Fixed-Parameter Backtests (full sample) ===")
    fixed_results = fixed_param_backtests(prices, daily_rets)
    # Print summary table
    print(f"\n  {'Config':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'PF':>6} {'WR%':>5}")
    print("  " + "-" * 70)
    for key, m in sorted(fixed_results.items(), key=lambda x: -x[1]["sharpe"]):
        print(f"  {key:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr_pct']:>7.2f} {m['max_dd_pct']:>7.2f} {m['profit_factor']:>6.2f} {m['win_rate_pct']:>5.1f}")

    # 3. Walk-forward backtest
    print("\n=== Walk-Forward Backtest ===")
    wf_result = walk_forward_backtest(prices, daily_rets)
    if not wf_result:
        print("Walk-forward failed — not enough data")
        return

    oot_returns = wf_result.pop("oot_returns")

    # 4. Benchmark
    print("\n=== Benchmark (SPY Buy & Hold, OOT period) ===")
    bench = compute_benchmark(daily_rets, oot_returns)
    print(f"  SPY: Sharpe={bench['sharpe']}, CAGR={bench['cagr_pct']}%, MaxDD={bench['max_dd_pct']}%")

    # 5. Adversarial checks (HC #705)
    print("\n=== Adversarial Checks ===")
    spy_rets = daily_rets["SPY"]

    perm_result = permutation_test(daily_rets, oot_returns)
    regime_result = regime_test(oot_returns, spy_rets)
    subperiod_result = sub_period_test(oot_returns)
    outlier_result = outlier_removal_test(oot_returns)

    # 6. Overall verdict
    adversarial_pass = all([
        perm_result["significant_at_05"],
        regime_result["r1_passed"],
        subperiod_result["both_profitable"],
        outlier_result["still_profitable_after_removal"],
    ])

    print(f"\n=== OVERALL VERDICT: {'PASS' if adversarial_pass else 'FAIL'} ===")
    print(f"  Permutation test:  {'PASS' if perm_result['significant_at_05'] else 'FAIL'} (p={perm_result['p_value']})")
    print(f"  Regime R1:         {'PASS' if regime_result['r1_passed'] else 'FAIL'} (gap={regime_result['regime_gap']})")
    print(f"  Sub-period:        {'PASS' if subperiod_result['both_profitable'] else 'FAIL'}")
    print(f"  Outlier removal:   {'PASS' if outlier_result['still_profitable_after_removal'] else 'FAIL'}")

    # 7. Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "universe": TICKERS,
        "data_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
        "n_trading_days": len(daily_rets),
        "fixed_param_backtests": fixed_results,
        "walk_forward": wf_result,
        "benchmark_spy": bench,
        "adversarial": {
            "permutation_test": perm_result,
            "regime_test": regime_result,
            "sub_period_test": subperiod_result,
            "outlier_removal_test": outlier_result,
            "overall_pass": adversarial_pass,
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
