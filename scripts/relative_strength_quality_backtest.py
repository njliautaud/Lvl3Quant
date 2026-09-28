#!/usr/bin/env python3
"""
Relative Strength within Quality Universe Backtest
===================================================
Tests whether picking stocks by relative strength from a quality universe
adds alpha over random/equal-weight selection.

6 Variants + Equal-Weight Benchmark, with permutation testing.
OOT: Jan 2022 – Jul 2026 | Starting Capital: $645
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
QUALITY_UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START_DATE = "2021-07-01"  # extra lookback for momentum calculation
END_DATE = "2026-07-31"
OOT_START = "2022-01-01"
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
N_PERMUTATIONS = 1000
RISK_FREE_RATE = 0.0
RANDOM_SEED = 42

# ── Data Download ──────────────────────────────────────────────────────────
print("Downloading price data...")
tickers = QUALITY_UNIVERSE + ["SPY"]
data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    prices = data["Close"]
else:
    prices = data[["Close"]].copy()
    prices.columns = tickers

# Forward fill then drop any remaining NaN rows at start
prices = prices.ffill().dropna()

print(f"Price data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")

# ── Helper Functions ───────────────────────────────────────────────────────

def monthly_returns(prices_df, columns=None):
    """Get monthly returns for given columns."""
    if columns is None:
        columns = prices_df.columns
    monthly = prices_df[columns].resample("ME").last()
    return monthly.pct_change().dropna()


def rolling_return(prices_df, ticker, months):
    """Calculate rolling N-month return for a ticker at month-end dates."""
    monthly = prices_df[ticker].resample("ME").last()
    return monthly.pct_change(months)


def rolling_vol(prices_df, ticker, months):
    """Calculate rolling N-month daily volatility for a ticker."""
    daily_ret = prices_df[ticker].pct_change()
    monthly_end = prices_df.resample("ME").last().index
    vols = {}
    for date in monthly_end:
        lookback_start = date - pd.DateOffset(months=months)
        mask = (daily_ret.index > lookback_start) & (daily_ret.index <= date)
        window = daily_ret[mask]
        if len(window) >= 20:
            vols[date] = window.std() * np.sqrt(252)
        else:
            vols[date] = np.nan
    return pd.Series(vols)


def spy_regime(prices_df):
    """Classify each date as bull (SPY > 200-SMA) or bear."""
    spy = prices_df["SPY"]
    sma200 = spy.rolling(200).mean()
    regime = pd.Series("bull", index=spy.index)
    regime[spy < sma200] = "bear"
    # Resample to month-end
    monthly_regime = regime.resample("ME").last()
    return monthly_regime


def calculate_metrics(monthly_rets, regime_series):
    """Calculate performance metrics from monthly return series."""
    total_ret = (1 + monthly_rets).prod() - 1
    n_years = len(monthly_rets) / 12
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    monthly_std = monthly_rets.std()
    ann_std = monthly_std * np.sqrt(12)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0

    downside = monthly_rets[monthly_rets < 0].std() * np.sqrt(12)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cumulative = (1 + monthly_rets).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    win_rate = (monthly_rets > 0).mean()

    # Regime-stratified Sharpe
    aligned_regime = regime_series.reindex(monthly_rets.index).ffill()
    bull_rets = monthly_rets[aligned_regime == "bull"]
    bear_rets = monthly_rets[aligned_regime == "bear"]

    def _sharpe(r):
        if len(r) < 3:
            return 0.0
        ann_r = r.mean() * 12
        ann_s = r.std() * np.sqrt(12)
        return ann_r / ann_s if ann_s > 0 else 0.0

    sharpe_bull = _sharpe(bull_rets)
    sharpe_bear = _sharpe(bear_rets)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "n_months": len(monthly_rets),
        "n_bull_months": len(bull_rets),
        "n_bear_months": len(bear_rets),
    }


def simulate_strategy(prices_df, selection_fn, n_hold, oot_start, capital, slippage,
                       weighting_fn=None):
    """
    Simulate a monthly-rebalance strategy.

    selection_fn(date, prices_df) -> list of tickers to hold
    weighting_fn(tickers, date, prices_df) -> dict {ticker: weight} (optional, default equal weight)

    Returns monthly return series.
    """
    stock_cols = [c for c in QUALITY_UNIVERSE if c in prices_df.columns]
    monthly_dates = prices_df[stock_cols].resample("ME").last().index
    oot_dates = monthly_dates[monthly_dates >= pd.Timestamp(oot_start)]

    portfolio_value = capital
    monthly_returns_list = []
    dates_list = []
    prev_weights = {}

    for i, date in enumerate(oot_dates):
        selected = selection_fn(date, prices_df)

        if weighting_fn:
            target_weights = weighting_fn(selected, date, prices_df)
        else:
            if len(selected) > 0:
                w = 1.0 / len(selected)
                target_weights = {t: w for t in selected}
            else:
                target_weights = {}

        # Calculate turnover for slippage
        all_tickers = set(list(prev_weights.keys()) + list(target_weights.keys()))
        turnover = sum(abs(target_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers)
        slippage_cost = turnover * slippage  # each way, but turnover counts both sides

        # Get next month's return
        if i + 1 < len(oot_dates):
            next_date = oot_dates[i + 1]
        else:
            # Last month: use last available date
            next_date = prices_df.index[-1]
            # Resample to get the actual last date
            if next_date <= date:
                break

        # Calculate weighted portfolio return for this month
        port_ret = 0.0
        for ticker, weight in target_weights.items():
            # Get price at rebalance date and next rebalance
            try:
                p0 = prices_df[ticker].asof(date)
                p1 = prices_df[ticker].asof(next_date)
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    stock_ret = (p1 / p0) - 1
                    port_ret += weight * stock_ret
            except Exception:
                pass

        # Cash portion earns 0
        cash_weight = 1.0 - sum(target_weights.values())
        # port_ret already only counts invested portion

        # Subtract slippage
        port_ret -= slippage_cost

        monthly_returns_list.append(port_ret)
        dates_list.append(next_date)
        prev_weights = target_weights.copy()

    return pd.Series(monthly_returns_list, index=dates_list)


# ── Pre-compute momentum/vol signals ──────────────────────────────────────
print("Computing momentum and volatility signals...")

stock_cols = [c for c in QUALITY_UNIVERSE if c in prices.columns]

# 3-month and 6-month returns
ret_3m = {}
ret_6m = {}
vol_3m = {}

for ticker in stock_cols:
    ret_3m[ticker] = rolling_return(prices, ticker, 3)
    ret_6m[ticker] = rolling_return(prices, ticker, 6)
    vol_3m[ticker] = rolling_vol(prices, ticker, 3)

ret_3m_df = pd.DataFrame(ret_3m)
ret_6m_df = pd.DataFrame(ret_6m)
vol_3m_df = pd.DataFrame(vol_3m)

# Universe average for relative strength
avg_3m = ret_3m_df.mean(axis=1)
avg_6m = ret_6m_df.mean(axis=1)

# Relative strength = stock return - universe average
rs_3m = ret_3m_df.sub(avg_3m, axis=0)
rs_6m = ret_6m_df.sub(avg_6m, axis=0)

# Risk-adjusted RS
rs_risk_adj = ret_3m_df.div(vol_3m_df.replace(0, np.nan))

# Regime classification
regime = spy_regime(prices)

# ── Strategy Definitions ──────────────────────────────────────────────────

def make_top_n_selector(rs_df, n, ascending=False):
    """Create selection function that picks top/bottom N by relative strength."""
    def select(date, prices_df):
        nearest = rs_df.index[rs_df.index <= date]
        if len(nearest) == 0:
            return []
        row = rs_df.loc[nearest[-1]].dropna()
        available = [t for t in row.index if t in stock_cols]
        row = row[available]
        ranked = row.sort_values(ascending=ascending)
        return ranked.head(n).index.tolist()
    return select


def variant_e_weighting(selected_tickers, date, prices_df):
    """Overweight top 5 (40%), underweight bottom 15 (60%)."""
    # Get top 5 by 3m RS
    nearest = rs_3m.index[rs_3m.index <= date]
    if len(nearest) == 0:
        return {t: 1/20 for t in stock_cols}
    row = rs_3m.loc[nearest[-1]].dropna()
    available = [t for t in row.index if t in stock_cols]
    row = row[available]
    ranked = row.sort_values(ascending=False)
    top5 = set(ranked.head(5).index.tolist())
    bottom15 = set(ranked.tail(len(ranked) - 5).index.tolist())

    weights = {}
    for t in top5:
        weights[t] = 0.40 / 5  # 8% each
    for t in bottom15:
        weights[t] = 0.60 / len(bottom15)
    return weights


def variant_f_selector(date, prices_df):
    """Dual momentum: top 5 by 3m RS, but only if absolute 3m return > 0."""
    nearest = rs_3m.index[rs_3m.index <= date]
    if len(nearest) == 0:
        return []
    row = rs_3m.loc[nearest[-1]].dropna()
    abs_row = ret_3m_df.loc[nearest[-1]].dropna()
    available = [t for t in row.index if t in stock_cols]
    row = row[available]
    abs_row = abs_row[available]
    ranked = row.sort_values(ascending=False)
    top5 = ranked.head(5).index.tolist()
    # Filter: only hold if absolute return > 0
    return [t for t in top5 if abs_row.get(t, 0) > 0]


# ── Run Strategies ─────────────────────────────────────────────────────────
print("\nRunning strategy simulations...")

strategies = {
    "A_top5_3mRS": {
        "selector": make_top_n_selector(rs_3m, 5),
        "n_hold": 5,
        "desc": "Top 5 by 3-month relative strength, equal weight",
    },
    "B_top3_6mRS": {
        "selector": make_top_n_selector(rs_6m, 3),
        "n_hold": 3,
        "desc": "Top 3 by 6-month relative strength, concentrated",
    },
    "C_top5_riskadj": {
        "selector": make_top_n_selector(rs_risk_adj, 5),
        "n_hold": 5,
        "desc": "Top 5 by risk-adjusted 3m RS (return/vol), equal weight",
    },
    "D_bottom5_contrarian": {
        "selector": make_top_n_selector(rs_3m, 5, ascending=True),
        "n_hold": 5,
        "desc": "Bottom 5 by 3m RS (contrarian/mean-reversion), equal weight",
    },
    "E_overweight_top5": {
        "selector": lambda date, pdf: stock_cols,  # hold all 20
        "n_hold": 20,
        "weighting_fn": variant_e_weighting,
        "desc": "All 20, overweight top 5 (40%) underweight bottom 15 (60%)",
    },
    "F_dual_momentum": {
        "selector": variant_f_selector,
        "n_hold": 5,
        "desc": "Top 5 by 3m RS only if absolute 3m return > 0, else cash",
    },
}

# Benchmark: equal weight all 20
benchmark_selector = lambda date, pdf: stock_cols

print("  Running benchmark (equal-weight all 20)...")
bench_rets = simulate_strategy(prices, benchmark_selector, 20, OOT_START, STARTING_CAPITAL, SLIPPAGE_PCT)
bench_metrics = calculate_metrics(bench_rets, regime)

results = {"benchmark_equal_weight": bench_metrics}
results["benchmark_equal_weight"]["description"] = "Equal-weight all 20 quality stocks, monthly rebalance"
strategy_returns = {"benchmark": bench_rets}

for name, config in strategies.items():
    print(f"  Running {name}...")
    rets = simulate_strategy(
        prices, config["selector"], config["n_hold"], OOT_START,
        STARTING_CAPITAL, SLIPPAGE_PCT,
        weighting_fn=config.get("weighting_fn"),
    )
    metrics = calculate_metrics(rets, regime)
    metrics["description"] = config["desc"]
    results[name] = metrics
    strategy_returns[name] = rets

# ── Permutation Tests ──────────────────────────────────────────────────────
print("\nRunning permutation tests (1000 iterations)...")
rng = np.random.RandomState(RANDOM_SEED)


def random_selector(n, rng_instance):
    """Create a selector that picks n random stocks."""
    def select(date, prices_df):
        available = [t for t in stock_cols if t in prices_df.columns]
        if len(available) < n:
            return available
        chosen = rng_instance.choice(available, size=n, replace=False).tolist()
        return chosen
    return select


perm_results = {}
for name, config in strategies.items():
    n = config["n_hold"]
    if n >= 20:
        # Skip permutation for variant E (holds all 20)
        perm_results[name] = {"p_value": None, "percentile": None, "note": "Holds all 20, permutation N/A"}
        continue

    actual_sharpe = results[name]["sharpe"]
    perm_sharpes = []

    for i in range(N_PERMUTATIONS):
        sel = random_selector(n, np.random.RandomState(RANDOM_SEED + i + 1))
        perm_rets = simulate_strategy(prices, sel, n, OOT_START, STARTING_CAPITAL, SLIPPAGE_PCT)
        if len(perm_rets) > 0:
            perm_metrics = calculate_metrics(perm_rets, regime)
            perm_sharpes.append(perm_metrics["sharpe"])

    perm_sharpes = np.array(perm_sharpes)
    percentile = (perm_sharpes < actual_sharpe).mean() * 100
    p_value = 1.0 - percentile / 100

    perm_results[name] = {
        "actual_sharpe": actual_sharpe,
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
        "perm_median_sharpe": round(float(np.median(perm_sharpes)), 3),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 3),
        "percentile": round(percentile, 1),
        "p_value": round(p_value, 3),
    }
    print(f"  {name}: Sharpe={actual_sharpe:.3f}, perm median={np.median(perm_sharpes):.3f}, "
          f"percentile={percentile:.1f}%, p={p_value:.3f}")

# ── 5-Gate Validation ──────────────────────────────────────────────────────
print("\n5-Gate Validation:")
gate_results = {}

for name in list(strategies.keys()):
    m = results[name]
    perm = perm_results[name]

    gates = {
        "G1_sharpe_gt_0.5": m["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": perm["p_value"] < 0.05 if perm["p_value"] is not None else False,
        "G3_regime_gap_lt_0.5": m["regime_gap"] < 0.5,
        "G4_maxdd_gt_neg50": m["max_drawdown_pct"] > -50,
        "G5_rebalance_events_gte_20": m["n_months"] >= 20,
    }
    gates["passed_all"] = all(gates.values())
    gate_results[name] = gates

    status = "PASS" if gates["passed_all"] else "FAIL"
    failed = [k for k, v in gates.items() if not v and k != "passed_all"]
    print(f"  {name}: {status}" + (f" (failed: {', '.join(failed)})" if failed else ""))

# ── Compile Final Output ──────────────────────────────────────────────────
output = {
    "metadata": {
        "strategy": "Relative Strength within Quality Universe",
        "universe": QUALITY_UNIVERSE,
        "oot_period": f"{OOT_START} to {END_DATE}",
        "starting_capital": STARTING_CAPITAL,
        "slippage_each_way_pct": SLIPPAGE_PCT * 100,
        "rebalance_frequency": "monthly",
        "n_permutations": N_PERMUTATIONS,
        "run_date": datetime.now().isoformat(),
        "price_data_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
    },
    "benchmark": results["benchmark_equal_weight"],
    "variants": {},
    "permutation_tests": perm_results,
    "five_gate_validation": gate_results,
}

for name in strategies:
    output["variants"][name] = results[name]

# Add alpha vs benchmark
for name in strategies:
    alpha = results[name]["sharpe"] - bench_metrics["sharpe"]
    output["variants"][name]["sharpe_alpha_vs_benchmark"] = round(alpha, 3)
    alpha_ret = results[name]["annualized_return_pct"] - bench_metrics["annualized_return_pct"]
    output["variants"][name]["ann_return_alpha_vs_benchmark"] = round(alpha_ret, 2)

# Save
output_path = Path("/home/jupiter/Lvl3Quant/data/relative_strength_quality_results.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# ── Print Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("RELATIVE STRENGTH WITHIN QUALITY UNIVERSE — BACKTEST RESULTS")
print("=" * 100)
print(f"OOT Period: {OOT_START} to {END_DATE} | Starting Capital: ${STARTING_CAPITAL}")
print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% each way | Rebalance: Monthly")
print()

header = f"{'Variant':<30} {'TotRet%':>8} {'AnnRet%':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'WinR%':>6} {'ShBull':>7} {'ShBear':>7} {'RGap':>6} {'Perm-p':>7} {'5-Gate':>7}"
print(header)
print("-" * len(header))

# Benchmark first
b = bench_metrics
print(f"{'BENCHMARK (EW-20)':<30} {b['total_return_pct']:>8.1f} {b['annualized_return_pct']:>8.1f} "
      f"{b['sharpe']:>7.3f} {b['sortino']:>8.3f} {b['max_drawdown_pct']:>7.1f} {b['win_rate_pct']:>6.1f} "
      f"{b['sharpe_bull']:>7.3f} {b['sharpe_bear']:>7.3f} {b['regime_gap']:>6.3f} {'N/A':>7} {'N/A':>7}")

for name in strategies:
    m = results[name]
    p = perm_results[name]
    g = gate_results[name]
    p_str = f"{p['p_value']:.3f}" if p['p_value'] is not None else "N/A"
    g_str = "PASS" if g["passed_all"] else "FAIL"
    print(f"{name:<30} {m['total_return_pct']:>8.1f} {m['annualized_return_pct']:>8.1f} "
          f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_drawdown_pct']:>7.1f} {m['win_rate_pct']:>6.1f} "
          f"{m['sharpe_bull']:>7.3f} {m['sharpe_bear']:>7.3f} {m['regime_gap']:>6.3f} {p_str:>7} {g_str:>7}")

print()
print("Alpha vs Benchmark (Sharpe):")
for name in strategies:
    alpha = output["variants"][name]["sharpe_alpha_vs_benchmark"]
    direction = "+" if alpha > 0 else ""
    print(f"  {name}: {direction}{alpha:.3f}")

print()
print("KEY QUESTION: Does RS selection add alpha over random selection?")
any_significant = False
for name, p in perm_results.items():
    if p["p_value"] is not None and p["p_value"] < 0.05:
        any_significant = True
        print(f"  {name}: YES — p={p['p_value']:.3f}, Sharpe at {p['percentile']:.0f}th percentile vs random")
    elif p["p_value"] is not None:
        print(f"  {name}: NO — p={p['p_value']:.3f}, Sharpe at {p['percentile']:.0f}th percentile vs random")

if not any_significant:
    print("\n  CONCLUSION: NO variant shows statistically significant alpha from RS selection.")
    print("  Consistent with prior finding (entry 1731): stock selection within quality universe")
    print("  does NOT reliably beat random selection. The edge is in the UNIVERSE, not the RANKING.")
else:
    sig_variants = [n for n, p in perm_results.items() if p["p_value"] is not None and p["p_value"] < 0.05]
    passing = [n for n, g in gate_results.items() if g["passed_all"]]
    if passing:
        print(f"\n  VARIANTS PASSING ALL 5 GATES: {', '.join(passing)}")
    else:
        print(f"\n  Significant RS alpha in: {', '.join(sig_variants)}, but no variant passes all 5 gates.")
