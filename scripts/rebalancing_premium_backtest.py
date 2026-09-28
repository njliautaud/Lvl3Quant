#!/usr/bin/env python3
"""
Rebalancing Premium Backtest
============================
Tests whether systematic rebalancing generates excess returns ("volatility harvesting")
across 6 variants vs SPY buy-and-hold.

OOT: Jan 2022 – Jul 2026 | Starting Capital: $645 | Slippage: 0.02% each way
"""

import json
import datetime
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Universe ──────────────────────────────────────────────────────────────────
SECTORS = {
    "Tech": ["AAPL", "MSFT", "AVGO", "GOOGL", "META", "V", "MA"],
    "Healthcare": ["JNJ", "UNH", "LLY", "ABBV", "MRK"],
    "Consumer": ["PG", "KO", "PEP", "HD", "COST", "WMT"],
    "Finance": ["JPM"],
    "Growth": ["AMZN"],
}
ALL_TICKERS = [t for s in SECTORS.values() for t in s]
N_STOCKS = len(ALL_TICKERS)
assert N_STOCKS == 20

START = "2021-12-01"  # fetch extra for vol lookback
END = "2026-07-31"
BT_START = "2022-01-03"
CAPITAL = 645.0
SLIPPAGE_BPS = 0.0002  # 0.02% each way = 0.04% round-trip

# ── Data Download ─────────────────────────────────────────────────────────────
def download_data():
    tickers = ALL_TICKERS + ["SPY", "^VIX"]
    print(f"Downloading {len(tickers)} tickers...")
    df = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df
    # ^VIX ticker
    if "^VIX" in close.columns:
        vix = close["^VIX"].copy()
        close = close.drop(columns=["^VIX"], errors="ignore")
    else:
        vix = pd.Series(20.0, index=close.index)
    close = close.ffill().dropna(how="all")
    vix = vix.ffill().fillna(20.0)
    return close, vix


def compute_returns(close):
    return close.pct_change().fillna(0.0)


# ── Portfolio Engine ──────────────────────────────────────────────────────────
class PortfolioBacktest:
    def __init__(self, close_df, vix_series, capital=CAPITAL):
        self.close = close_df
        self.vix = vix_series
        self.capital = capital
        self.stock_tickers = [t for t in ALL_TICKERS if t in close_df.columns]
        self.dates = close_df.loc[BT_START:].index

    def run_variant(self, variant_name, target_weight_fn, rebalance_schedule_fn):
        """
        target_weight_fn(date, prices_history) -> dict {ticker: weight}
        rebalance_schedule_fn(date, idx, holdings, prices, target_weights, vix_val) -> bool
        """
        holdings = {}  # ticker -> shares (fractional)
        cash = self.capital
        portfolio_values = []
        rebalance_dates = []
        total_turnover_dollars = 0.0

        for i, date in enumerate(self.dates):
            prices = self.close.loc[date]

            # Current portfolio value
            port_val = cash
            for t in self.stock_tickers:
                if t in holdings:
                    port_val += holdings.get(t, 0) * prices[t]

            # Get target weights
            hist = self.close.loc[:date]
            target_w = target_weight_fn(date, hist)

            # Check if we should rebalance
            current_weights = {}
            if port_val > 0:
                for t in self.stock_tickers:
                    current_weights[t] = (holdings.get(t, 0) * prices[t]) / port_val

            vix_val = self.vix.loc[date] if date in self.vix.index else 20.0
            should_rebalance = rebalance_schedule_fn(
                date, i, current_weights, prices, target_w, vix_val
            )

            if should_rebalance or i == 0:
                rebalance_dates.append(date)
                # Execute rebalance
                for t in self.stock_tickers:
                    target_shares = (target_w.get(t, 0) * port_val) / prices[t]
                    current_shares = holdings.get(t, 0)
                    delta_shares = target_shares - current_shares
                    trade_value = abs(delta_shares * prices[t])
                    slippage_cost = trade_value * SLIPPAGE_BPS  # each way
                    cash -= delta_shares * prices[t]
                    cash -= slippage_cost
                    total_turnover_dollars += trade_value
                    holdings[t] = target_shares

                # Recalc after costs
                port_val = cash
                for t in self.stock_tickers:
                    port_val += holdings.get(t, 0) * prices[t]

            portfolio_values.append(port_val)

        series = pd.Series(portfolio_values, index=self.dates)
        avg_val = series.mean()
        turnover_pct = (total_turnover_dollars / avg_val * 100) if avg_val > 0 else 0

        return {
            "series": series,
            "rebalance_dates": rebalance_dates,
            "turnover_pct": turnover_pct,
            "total_turnover_dollars": total_turnover_dollars,
        }


# ── Weight Functions ──────────────────────────────────────────────────────────
def equal_weight_all(date, hist):
    tickers = [t for t in ALL_TICKERS if t in hist.columns]
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def sector_equal_weight(date, hist):
    """25% per sector, equal within sector."""
    weights = {}
    n_sectors = len(SECTORS)
    sector_w = 1.0 / n_sectors
    for sector, tickers in SECTORS.items():
        valid = [t for t in tickers if t in hist.columns]
        if valid:
            w = sector_w / len(valid)
            for t in valid:
                weights[t] = w
    return weights


def risk_parity_weight(date, hist):
    """Weight inversely proportional to 60-day realized vol."""
    tickers = [t for t in ALL_TICKERS if t in hist.columns]
    rets = hist[tickers].pct_change().tail(60)
    vols = rets.std()
    vols = vols.replace(0, vols[vols > 0].min() if (vols > 0).any() else 0.01)
    inv_vol = 1.0 / vols
    inv_vol = inv_vol / inv_vol.sum()
    return {t: inv_vol[t] for t in tickers}


# ── Rebalance Schedule Functions ──────────────────────────────────────────────
def monthly_rebalance(date, idx, current_w, prices, target_w, vix_val):
    if idx == 0:
        return True
    return date.month != pd.Timestamp(date).to_period("M").start_time.month or date.day <= 3


def _make_monthly():
    last_month = [None]
    def fn(date, idx, current_w, prices, target_w, vix_val):
        m = (date.year, date.month)
        if last_month[0] is None or m != last_month[0]:
            last_month[0] = m
            return True
        return False
    return fn


def _make_weekly():
    last_week = [None]
    def fn(date, idx, current_w, prices, target_w, vix_val):
        w = date.isocalendar()[1]
        y = date.year
        key = (y, w)
        if last_week[0] is None or key != last_week[0]:
            last_week[0] = key
            return True
        return False
    return fn


def _make_threshold(threshold=0.05):
    last_month = [None]
    def fn(date, idx, current_w, prices, target_w, vix_val):
        if idx == 0:
            return True
        # Check drift
        for t in target_w:
            curr = current_w.get(t, 0)
            tgt = target_w.get(t, 0)
            if abs(curr - tgt) > threshold:
                return True
        return False
    return fn


def _make_monthly_vix_filter(vix_threshold=30):
    last_month = [None]
    def fn(date, idx, current_w, prices, target_w, vix_val):
        m = (date.year, date.month)
        if last_month[0] is None or m != last_month[0]:
            last_month[0] = m
            if vix_val > vix_threshold:
                return False  # skip rebalance in panic
            return True
        return False
    return fn


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(series, capital, label=""):
    daily_ret = series.pct_change().dropna()
    n_days = len(daily_ret)
    ann_factor = 252

    total_ret = (series.iloc[-1] / series.iloc[0]) - 1
    n_years = n_days / ann_factor
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    mu = daily_ret.mean() * ann_factor
    sigma = daily_ret.std() * np.sqrt(ann_factor)
    sharpe = mu / sigma if sigma > 0 else 0

    downside = daily_ret[daily_ret < 0].std() * np.sqrt(ann_factor)
    sortino = mu / downside if downside > 0 else 0

    # Max drawdown
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    max_dd = dd.min()

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "final_value": round(series.iloc[-1], 2),
        "n_days": n_days,
    }


def regime_gap(series, spy_series):
    """Compute performance gap in green vs red regimes (based on SPY monthly)."""
    port_monthly = series.resample("ME").last().pct_change().dropna()
    spy_monthly = spy_series.resample("ME").last().pct_change().dropna()

    common = port_monthly.index.intersection(spy_monthly.index)
    port_monthly = port_monthly.loc[common]
    spy_monthly = spy_monthly.loc[common]

    green = spy_monthly > 0
    red = spy_monthly <= 0

    if green.sum() > 1 and red.sum() > 1:
        green_sharpe = (port_monthly[green].mean() / port_monthly[green].std()) * np.sqrt(12) if port_monthly[green].std() > 0 else 0
        red_sharpe = (port_monthly[red].mean() / port_monthly[red].std()) * np.sqrt(12) if port_monthly[red].std() > 0 else 0
        gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.001)
    else:
        green_sharpe = red_sharpe = gap = 0

    return {
        "green_months": int(green.sum()),
        "red_months": int(red.sum()),
        "green_sharpe": round(green_sharpe, 3),
        "red_sharpe": round(red_sharpe, 3),
        "regime_gap": round(gap, 3),
    }


# ── Permutation Test ──────────────────────────────────────────────────────────
def permutation_test(close_df, vix_series, variant_fn, n_iter=1000):
    """
    Randomize rebalance dates to test if systematic rebalancing adds value.
    Compare actual final value vs random-date rebalancing.
    """
    bt = PortfolioBacktest(close_df, vix_series)

    # Run actual
    weight_fn, sched_fn = variant_fn()
    actual = bt.run_variant("actual", weight_fn, sched_fn)
    actual_final = actual["series"].iloc[-1]
    n_rebalances = len(actual["rebalance_dates"])

    # Random permutations
    all_dates = bt.dates
    n_dates = len(all_dates)
    random_finals = []

    for _ in range(n_iter):
        # Pick random rebalance dates (same count)
        rand_indices = set(np.random.choice(n_dates, size=min(n_rebalances, n_dates), replace=False))

        def make_random_sched(ri):
            def fn(date, idx, current_w, prices, target_w, vix_val):
                return idx in ri
            return fn

        result = bt.run_variant("random", equal_weight_all, make_random_sched(rand_indices))
        random_finals.append(result["series"].iloc[-1])

    random_finals = np.array(random_finals)
    p_value = np.mean(random_finals >= actual_final)

    return {
        "actual_final": round(actual_final, 2),
        "random_mean": round(random_finals.mean(), 2),
        "random_median": round(np.median(random_finals), 2),
        "random_std": round(random_finals.std(), 2),
        "p_value": round(p_value, 4),
        "n_iterations": n_iter,
        "n_rebalances": n_rebalances,
        "pct_better_than_random": round((1 - p_value) * 100, 1),
    }


# ── 5-Gate Validation ────────────────────────────────────────────────────────
def five_gate_validation(metrics, regime, turnover_pct, perm_result):
    gates = {}

    # Gate 1: Statistical significance (p < 0.05 on permutation)
    gates["G1_statistical_significance"] = {
        "pass": perm_result["p_value"] < 0.05,
        "p_value": perm_result["p_value"],
        "threshold": 0.05,
    }

    # Gate 2: Regime robustness (gap < 0.50)
    gates["G2_regime_robustness"] = {
        "pass": regime["regime_gap"] < 0.50,
        "regime_gap": regime["regime_gap"],
        "threshold": 0.50,
    }

    # Gate 3: Drawdown sanity (max DD > -60%)
    gates["G3_drawdown_sanity"] = {
        "pass": metrics["max_drawdown_pct"] > -60,
        "max_dd": metrics["max_drawdown_pct"],
        "threshold": -60,
    }

    # Gate 4: Sharpe > 0
    gates["G4_positive_sharpe"] = {
        "pass": metrics["sharpe"] > 0,
        "sharpe": metrics["sharpe"],
    }

    # Gate 5: Net-of-costs still positive total return
    gates["G5_net_positive_return"] = {
        "pass": metrics["total_return_pct"] > 0,
        "total_return_pct": metrics["total_return_pct"],
    }

    all_pass = all(g["pass"] for g in gates.values())
    return {"gates": gates, "all_pass": all_pass}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("REBALANCING PREMIUM BACKTEST")
    print(f"Period: Jan 2022 – Jul 2026 | Capital: ${CAPITAL} | Slippage: {SLIPPAGE_BPS*100:.2f}% each way")
    print("=" * 70)

    close, vix = download_data()

    # Filter to backtest period
    close_bt = close.loc[BT_START:]
    missing = [t for t in ALL_TICKERS if t not in close.columns]
    if missing:
        print(f"WARNING: Missing tickers: {missing}")

    bt = PortfolioBacktest(close, vix)

    # SPY benchmark (buy and hold)
    spy_prices = close["SPY"].loc[BT_START:]
    spy_series = spy_prices / spy_prices.iloc[0] * CAPITAL
    spy_metrics = compute_metrics(spy_series, CAPITAL, "SPY")

    # Define variants
    variants = {
        "A: EW Monthly": (equal_weight_all, _make_monthly()),
        "B: EW Weekly": (equal_weight_all, _make_weekly()),
        "C: EW Threshold 5%": (equal_weight_all, _make_threshold(0.05)),
        "D: Sector-EW Monthly": (sector_equal_weight, _make_monthly()),
        "E: Risk-Parity Monthly": (risk_parity_weight, _make_monthly()),
        "F: EW Monthly VIX<30": (equal_weight_all, _make_monthly_vix_filter(30)),
    }

    results = {}
    results["SPY_BuyHold"] = {
        "metrics": spy_metrics,
        "turnover_pct": 0.0,
        "regime": regime_gap(spy_series, spy_series),
        "five_gate": None,
    }

    print(f"\n{'Variant':<25} {'TotRet%':>8} {'AnnRet%':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Turnover%':>10} {'Final$':>8}")
    print("-" * 95)
    print(f"{'SPY Buy&Hold':<25} {spy_metrics['total_return_pct']:>8.1f} {spy_metrics['annualized_return_pct']:>8.1f} {spy_metrics['sharpe']:>7.3f} {spy_metrics['sortino']:>8.3f} {spy_metrics['max_drawdown_pct']:>8.1f} {'0.0':>10} {spy_metrics['final_value']:>8.2f}")

    for name, (wfn, sfn) in variants.items():
        res = bt.run_variant(name, wfn, sfn)
        metrics = compute_metrics(res["series"], CAPITAL, name)
        reg = regime_gap(res["series"], spy_series)

        # Permutation test (use variant A for baseline permutation)
        def make_variant_fn(w=wfn):
            def fn():
                return (w, _make_monthly())
            return fn

        if name == "A: EW Monthly":
            print(f"\nRunning permutation test (1000 iter) for baseline variant A...")
            perm = permutation_test(close, vix, make_variant_fn(wfn), n_iter=1000)
        else:
            perm = {"p_value": None, "actual_final": metrics["final_value"],
                    "random_mean": None, "random_median": None, "random_std": None,
                    "n_iterations": 0, "n_rebalances": len(res["rebalance_dates"]),
                    "pct_better_than_random": None}

        gate = five_gate_validation(metrics, reg, res["turnover_pct"],
                                     {"p_value": perm["p_value"] if perm["p_value"] is not None else 1.0})

        results[name] = {
            "metrics": metrics,
            "turnover_pct": round(res["turnover_pct"], 1),
            "total_turnover_dollars": round(res["total_turnover_dollars"], 2),
            "n_rebalances": len(res["rebalance_dates"]),
            "regime": reg,
            "five_gate": gate,
            "permutation_test": perm,
        }

        g_label = "PASS" if gate["all_pass"] else "FAIL"
        print(f"{name:<25} {metrics['total_return_pct']:>8.1f} {metrics['annualized_return_pct']:>8.1f} {metrics['sharpe']:>7.3f} {metrics['sortino']:>8.3f} {metrics['max_drawdown_pct']:>8.1f} {res['turnover_pct']:>10.1f} {metrics['final_value']:>8.2f}  [{g_label}]")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS")
    print("=" * 70)
    print(f"{'Variant':<25} {'Green#':>7} {'Red#':>6} {'GreenSh':>8} {'RedSh':>7} {'Gap':>6}")
    print("-" * 60)
    for name, data in results.items():
        r = data["regime"]
        print(f"{name:<25} {r['green_months']:>7} {r['red_months']:>6} {r['green_sharpe']:>8.3f} {r['red_sharpe']:>7.3f} {r['regime_gap']:>6.3f}")

    # Permutation test results
    perm_a = results["A: EW Monthly"]["permutation_test"]
    print(f"\n{'='*70}")
    print("PERMUTATION TEST (Variant A: EW Monthly)")
    print(f"{'='*70}")
    print(f"Actual final value:      ${perm_a['actual_final']:.2f}")
    print(f"Random rebalance mean:   ${perm_a['random_mean']:.2f}")
    print(f"Random rebalance median: ${perm_a['random_median']:.2f}")
    print(f"Random rebalance std:    ${perm_a['random_std']:.2f}")
    print(f"P-value:                 {perm_a['p_value']:.4f}")
    print(f"% better than random:    {perm_a['pct_better_than_random']}%")

    # Key question
    print(f"\n{'='*70}")
    print("KEY QUESTION: Does rebalancing premium overcome costs at $645?")
    print(f"{'='*70}")

    a_ret = results["A: EW Monthly"]["metrics"]["total_return_pct"]
    b_ret = results["B: EW Weekly"]["metrics"]["total_return_pct"]
    c_ret = results["C: EW Threshold 5%"]["metrics"]["total_return_pct"]
    spy_ret = spy_metrics["total_return_pct"]

    a_turn = results["A: EW Monthly"]["turnover_pct"]
    b_turn = results["B: EW Weekly"]["turnover_pct"]

    print(f"SPY buy-and-hold:          {spy_ret:>+.1f}%")
    print(f"A (monthly rebal):         {a_ret:>+.1f}% | turnover {a_turn:.0f}%")
    print(f"B (weekly rebal):          {b_ret:>+.1f}% | turnover {b_turn:.0f}%")
    print(f"B-A spread (weekly cost):  {b_ret - a_ret:>+.2f}%")
    print(f"A vs SPY alpha:            {a_ret - spy_ret:>+.2f}%")

    if b_ret > a_ret:
        print("\n>> More frequent rebalancing DOES generate incremental return net of costs.")
    else:
        print("\n>> More frequent rebalancing does NOT overcome additional transaction costs.")

    best_variant = max(
        [(n, d["metrics"]["sharpe"]) for n, d in results.items() if n != "SPY_BuyHold"],
        key=lambda x: x[1]
    )
    print(f"\nBest risk-adjusted variant: {best_variant[0]} (Sharpe {best_variant[1]:.3f})")

    # 5-Gate summary
    print(f"\n{'='*70}")
    print("5-GATE VALIDATION SUMMARY")
    print(f"{'='*70}")
    for name, data in results.items():
        if data["five_gate"] is None:
            continue
        g = data["five_gate"]
        status = "ALL PASS" if g["all_pass"] else "FAIL"
        failed = [k for k, v in g["gates"].items() if not v["pass"]]
        fail_str = f" (failed: {', '.join(failed)})" if failed else ""
        print(f"  {name:<25} {status}{fail_str}")

    # Save results
    output = {
        "metadata": {
            "backtest": "Rebalancing Premium",
            "period": "2022-01-03 to 2026-07-31",
            "capital": CAPITAL,
            "slippage_bps": SLIPPAGE_BPS * 10000,
            "universe": ALL_TICKERS,
            "n_stocks": N_STOCKS,
            "generated": datetime.datetime.now().isoformat(),
        },
        "results": {},
    }

    for name, data in results.items():
        entry = {k: v for k, v in data.items() if k != "series"}
        output["results"][name] = entry

    out_path = Path("/home/jupiter/Lvl3Quant/data/rebalancing_premium_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
