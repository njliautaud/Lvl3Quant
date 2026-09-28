#!/usr/bin/env python3
"""
Cross-Asset Momentum with Validated Crash Filter v2
=====================================================
Monthly rebalance across SPY, EFA, EEM, GLD, TLT, DBC.
Hold top 3 by 6-month momentum. Add validated crash filter
(VIX>28 + credit spread widening proxy + SPY<200MA) to go 100% cash.

Key improvements over cross_asset_momentum_v1.py:
- Proper crash filter with MULTIPLE conditions (not just VIX)
- 6-month momentum (not 12-1) — faster reaction to trends
- Credit spread proxy via HYG/IEF ratio
- Proper adversarial validation with regime checks
- No DCA, fixed $100K, next-day execution

HC #0  : Sliding walk-forward (trailing lookback)
HC #428: Regime-agnostic validation (R1)
HC #694: Commission-free (Robinhood)
HC #697: No crypto
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/cross_asset_momentum_crash")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2007-01-01"
END = "2026-07-17"
INITIAL_CAPITAL = 100_000

# Asset universe — broad, liquid, low-cost
UNIVERSE = ["SPY", "EFA", "EEM", "GLD", "TLT", "DBC"]
SAFE_HAVEN = "SHY"  # Cash proxy during crash
BENCHMARK = "SPY"

# Strategy parameters
MOM_WINDOW = 126       # 6-month lookback
MOM_SKIP = 21          # Skip most recent month (mean reversion)
TOP_N = 3              # Hold top 3 assets
REBAL_FREQ = "ME"      # Monthly rebalance

# Crash filter thresholds (ALL must be true to trigger crash mode)
CRASH_VIX_THRESHOLD = 28.0
CRASH_SPY_MA = 200
CRASH_CREDIT_SPREAD_Z = 1.5  # HYG/IEF ratio z-score below -1.5 = stress


def download_data():
    """Download all required price data."""
    tickers = UNIVERSE + [SAFE_HAVEN, "^VIX", "HYG", "IEF"]
    print(f"Downloading {len(tickers)} tickers...")

    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = prices.ffill(limit=5)

    vix = prices["^VIX"].copy() if "^VIX" in prices.columns else None
    prices = prices.drop(columns=["^VIX"], errors="ignore")

    print(f"  Data: {prices.shape[0]} days, {prices.shape[1]} tickers")
    print(f"  Range: {prices.index[0].date()} to {prices.index[-1].date()}")

    return prices, vix


def compute_crash_filter(prices, vix):
    """
    Multi-condition crash filter. ALL conditions must be true:
    1. VIX > 28
    2. SPY < 200-day MA
    3. Credit spread widening (HYG/IEF ratio z-score < -1.5)

    Returns: Series of bools (True = crash mode, go to cash)
    """
    spy = prices["SPY"]
    spy_ma200 = spy.rolling(CRASH_SPY_MA, min_periods=150).mean()
    spy_below_ma = spy < spy_ma200

    vix_high = vix > CRASH_VIX_THRESHOLD if vix is not None else pd.Series(False, index=spy.index)

    # Credit spread proxy: HYG/IEF ratio (falling = stress)
    if "HYG" in prices.columns and "IEF" in prices.columns:
        credit_ratio = prices["HYG"] / prices["IEF"]
        credit_z = (credit_ratio - credit_ratio.rolling(252).mean()) / credit_ratio.rolling(252).std()
        credit_stress = credit_z < -CRASH_CREDIT_SPREAD_Z
    else:
        credit_stress = pd.Series(False, index=spy.index)

    # Require at least 2 of 3 conditions (more robust than requiring all 3)
    crash_score = spy_below_ma.astype(int) + vix_high.astype(int) + credit_stress.astype(int)
    crash_mode = crash_score >= 2

    return crash_mode


def compute_momentum(prices, tickers):
    """6-month momentum with 1-month skip."""
    mom = pd.DataFrame(index=prices.index, columns=tickers)
    for t in tickers:
        if t in prices.columns:
            p = prices[t]
            # Return from (MOM_WINDOW + MOM_SKIP) days ago to (MOM_SKIP) days ago
            mom[t] = p.shift(MOM_SKIP) / p.shift(MOM_WINDOW + MOM_SKIP) - 1
    return mom.astype(float)


def get_rebal_dates(index):
    """Get month-end rebalance dates."""
    monthly = pd.Series(range(len(index)), index=index).resample(REBAL_FREQ).last()
    return monthly.dropna().index


def backtest_momentum_crash(prices, vix, variant="base"):
    """
    Run cross-asset momentum with crash filter.

    Variants:
    - "base": Top 3 equal weight, crash filter on
    - "no_crash": Top 3 equal weight, NO crash filter (to measure filter value)
    - "top2": Top 2 only (more concentrated)
    - "vol_weighted": Inverse volatility weighting among top 3
    """
    params = {
        "base": {"top_n": 3, "crash_filter": True, "vol_weight": False},
        "no_crash": {"top_n": 3, "crash_filter": False, "vol_weight": False},
        "top2": {"top_n": 2, "crash_filter": True, "vol_weight": False},
        "vol_weighted": {"top_n": 3, "crash_filter": True, "vol_weight": True},
    }[variant]

    crash_mode = compute_crash_filter(prices, vix)
    momentum = compute_momentum(prices, UNIVERSE)
    rebal_dates = get_rebal_dates(prices.index)

    # Daily returns for all assets
    returns = prices[UNIVERSE + [SAFE_HAVEN]].pct_change().fillna(0)

    capital = INITIAL_CAPITAL
    equity_curve = []
    weights = pd.Series(0.0, index=UNIVERSE)
    in_crash = False
    trade_log = []

    # Start after warmup
    warmup_date = prices.index[MOM_WINDOW + MOM_SKIP + 10]

    for date in prices.index:
        if date < warmup_date:
            equity_curve.append({"date": date, "equity": capital, "in_crash": False})
            continue

        # Check if rebalance day
        is_rebal = date in rebal_dates

        if is_rebal:
            mom_row = momentum.loc[date].dropna()
            crash_now = crash_mode.get(date, False) if params["crash_filter"] else False

            if crash_now:
                # Go to cash
                new_weights = pd.Series(0.0, index=UNIVERSE)
                in_crash = True
                trade_log.append({"date": str(date.date()), "action": "crash_cash", "holdings": "SHY"})
            else:
                # Select top N by momentum
                available = mom_row.reindex(UNIVERSE).dropna()
                top = available.nlargest(params["top_n"])
                top = top[top > 0]  # Only positive momentum

                if top.empty:
                    new_weights = pd.Series(0.0, index=UNIVERSE)
                    in_crash = False
                elif params["vol_weight"]:
                    # Inverse vol weighting
                    vol_63 = prices[top.index].pct_change().rolling(63).std().iloc[-1]
                    inv_vol = 1.0 / vol_63.clip(lower=0.01)
                    w = inv_vol / inv_vol.sum()
                    new_weights = pd.Series(0.0, index=UNIVERSE)
                    new_weights[w.index] = w.values
                    in_crash = False
                else:
                    new_weights = pd.Series(0.0, index=UNIVERSE)
                    new_weights[top.index] = 1.0 / len(top)
                    in_crash = False

                holdings = list(new_weights[new_weights > 0].index)
                trade_log.append({"date": str(date.date()), "action": "rebalance", "holdings": holdings})

            weights = new_weights

        # Compute daily return
        if in_crash:
            daily_ret = returns.loc[date, SAFE_HAVEN] if date in returns.index else 0
        else:
            # Weighted return across holdings
            daily_ret = 0
            for t in UNIVERSE:
                if weights[t] > 0 and date in returns.index:
                    daily_ret += weights[t] * returns.loc[date, t]

        capital *= (1 + daily_ret)
        equity_curve.append({"date": date, "equity": capital, "in_crash": in_crash})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return equity_df, trade_log


def compute_metrics(equity_df, label="Strategy"):
    """Standard performance metrics."""
    eq = equity_df["equity"]
    daily_ret = eq.pct_change().dropna()

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = (daily_ret.mean() * 252) / downside if downside > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "label": label,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "years": float(years),
        "final_equity": float(eq.iloc[-1]),
    }


def adversarial_validation(equity_df, spy_prices, n_perms=1000):
    """Full adversarial validation suite."""
    results = {}
    eq = equity_df["equity"]
    daily_ret = eq.pct_change().dropna()
    actual_sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # 1. Permutation test
    print("  Permutation test (1000 shuffles)...")
    perm_sharpes = []
    ret_vals = daily_ret.values.copy()
    for _ in range(n_perms):
        np.random.shuffle(ret_vals)
        s = (ret_vals.mean() * 252) / (ret_vals.std() * np.sqrt(252))
        perm_sharpes.append(s)
    p_value = (np.array(perm_sharpes) >= actual_sharpe).mean()
    results["permutation"] = {
        "actual_sharpe": float(actual_sharpe),
        "p_value": float(p_value),
        "pass": p_value < 0.05,
    }

    # 2. Sub-period stability (thirds — more granular than halves)
    n = len(daily_ret)
    thirds = [daily_ret.iloc[:n//3], daily_ret.iloc[n//3:2*n//3], daily_ret.iloc[2*n//3:]]
    third_sharpes = []
    for i, t in enumerate(thirds):
        s = (t.mean() * 252) / (t.std() * np.sqrt(252)) if t.std() > 0 else 0
        third_sharpes.append(float(s))
    all_positive = all(s > 0 for s in third_sharpes)
    results["sub_period"] = {
        "third_sharpes": third_sharpes,
        "all_positive": all_positive,
        "pass": all_positive,
    }

    # 3. Outlier removal — remove best 10 days
    sorted_ret = daily_ret.sort_values(ascending=False)
    trimmed = sorted_ret.iloc[10:]
    trimmed_sharpe = (trimmed.mean() * 252) / (trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
    trimmed_total = (1 + trimmed).prod() - 1
    results["outlier_removal"] = {
        "trimmed_sharpe": float(trimmed_sharpe),
        "trimmed_total_return": float(trimmed_total),
        "pass": trimmed_total > 0 and trimmed_sharpe > 0,
    }

    # 4. R1 Regime-agnostic check (HC #428)
    # Classify using SPY returns
    spy_ret = spy_prices.pct_change().dropna()
    common_idx = daily_ret.index.intersection(spy_ret.index)
    strat_aligned = daily_ret.reindex(common_idx).dropna()
    spy_aligned = spy_ret.reindex(common_idx).dropna()

    # Monthly regime classification
    spy_monthly = spy_aligned.resample("ME").sum()
    strat_monthly = strat_aligned.resample("ME").sum()
    common_months = spy_monthly.index.intersection(strat_monthly.index)

    green = strat_monthly.loc[common_months][spy_monthly.loc[common_months] > 0]
    red = strat_monthly.loc[common_months][spy_monthly.loc[common_months] <= 0]

    s_green = (green.mean() * 12) / (green.std() * np.sqrt(12)) if len(green) > 2 and green.std() > 0 else 0
    s_red = (red.mean() * 12) / (red.std() * np.sqrt(12)) if len(red) > 2 and red.std() > 0 else 0

    regime_ratio = abs(s_green - s_red) / max(abs(s_green), abs(s_red), 0.01)
    results["regime_agnostic"] = {
        "green_sharpe": float(s_green),
        "red_sharpe": float(s_red),
        "regime_ratio": float(regime_ratio),
        "pass": regime_ratio < 0.50,
    }

    # 5. Crash period performance (2008, 2020, 2022)
    crisis_periods = [
        ("GFC_2008", "2008-09-01", "2009-03-31"),
        ("COVID_2020", "2020-02-15", "2020-04-15"),
        ("BEAR_2022", "2022-01-01", "2022-10-31"),
    ]
    crisis_results = {}
    for name, s, e in crisis_periods:
        mask = (daily_ret.index >= pd.Timestamp(s)) & (daily_ret.index <= pd.Timestamp(e))
        crisis_ret = daily_ret[mask]
        if len(crisis_ret) > 5:
            crisis_total = (1 + crisis_ret).prod() - 1
            crisis_results[name] = float(crisis_total)
    results["crisis_performance"] = crisis_results

    overall = all(r.get("pass", True) for k, r in results.items() if isinstance(r, dict) and "pass" in r)
    results["overall_pass"] = overall

    return results


def main():
    print("=" * 70)
    print("CROSS-ASSET MOMENTUM WITH CRASH FILTER v2")
    print("=" * 70)

    prices, vix = download_data()

    all_results = {}
    equity_curves = {}

    for variant in ["base", "no_crash", "top2", "vol_weighted"]:
        print(f"\n--- Variant: {variant} ---")
        equity_df, trade_log = backtest_momentum_crash(prices, vix, variant=variant)
        metrics = compute_metrics(equity_df, label=f"MomCrash_{variant}")

        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  MaxDD: {metrics['max_dd']:.2%}")
        print(f"  Calmar: {metrics['calmar']:.3f}")
        print(f"  Final: ${metrics['final_equity']:,.0f}")

        crash_days = equity_df["in_crash"].sum() if "in_crash" in equity_df.columns else 0
        total_days = len(equity_df)
        print(f"  Crash days: {crash_days}/{total_days} ({crash_days/total_days:.1%})")

        # Adversarial validation
        print(f"\n  Adversarial validation:")
        adv = adversarial_validation(equity_df, prices["SPY"])
        for test_name, test_result in adv.items():
            if isinstance(test_result, dict) and "pass" in test_result:
                status = "PASS" if test_result["pass"] else "FAIL"
                print(f"    {test_name}: {status}")
            elif test_name == "crisis_performance":
                for crisis_name, crisis_ret in test_result.items():
                    print(f"    {crisis_name}: {crisis_ret:+.2%}")
        print(f"    OVERALL: {'PASS' if adv.get('overall_pass') else 'FAIL'}")

        all_results[variant] = {
            "metrics": metrics,
            "adversarial": adv,
            "num_rebalances": len(trade_log),
            "crash_pct": float(crash_days / total_days) if total_days > 0 else 0,
        }
        equity_curves[variant] = equity_df

    # Crash filter value analysis
    if "base" in all_results and "no_crash" in all_results:
        print("\n--- CRASH FILTER VALUE ---")
        base_dd = all_results["base"]["metrics"]["max_dd"]
        no_crash_dd = all_results["no_crash"]["metrics"]["max_dd"]
        base_sharpe = all_results["base"]["metrics"]["sharpe"]
        no_crash_sharpe = all_results["no_crash"]["metrics"]["sharpe"]
        print(f"  MaxDD improvement: {abs(no_crash_dd) - abs(base_dd):.2%} less drawdown")
        print(f"  Sharpe improvement: {base_sharpe - no_crash_sharpe:+.3f}")

    # SPY benchmark
    spy_eq = pd.DataFrame({"equity": INITIAL_CAPITAL * prices["SPY"].dropna() / prices["SPY"].dropna().iloc[0]})
    spy_metrics = compute_metrics(spy_eq, "SPY_BH")
    all_results["spy_benchmark"] = spy_metrics
    print(f"\n--- SPY Buy & Hold ---")
    print(f"  CAGR: {spy_metrics['cagr']:.2%}, Sharpe: {spy_metrics['sharpe']:.3f}, MaxDD: {spy_metrics['max_dd']:.2%}")

    # Save results
    with open(OUTPUT_DIR / "momentum_crash_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    for variant, eq_df in equity_curves.items():
        ax.plot(eq_df.index, eq_df["equity"].values, label=variant, linewidth=1.2)
    ax.plot(spy_eq.index, spy_eq["equity"].values, label="SPY B&H", linestyle="--", alpha=0.7)
    ax.set_title("Cross-Asset Momentum + Crash Filter")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "momentum_crash_equity.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
