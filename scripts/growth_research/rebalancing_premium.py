#!/usr/bin/env python3
"""
Rebalancing Premium Study
=========================
Quantifies the return added by various rebalancing strategies on a multi-strategy
portfolio: 50% UPRO (VIX-gated), 30% CTA trend-following, 15% ETF reversal, 5% cash.

Tests 7 rebalancing regimes, walk-forward validates across 3 sub-periods,
and runs a 100-shuffle permutation test for statistical significance.
"""

import json, os, warnings, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/rebalancing_premium")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REBAL_COST_BPS = 5  # 5 bps per rebalance trade
TARGET_WEIGHTS = np.array([0.50, 0.30, 0.15, 0.05])  # UPRO, CTA, Reversal, Cash
SLEEVE_NAMES = ["UPRO_VIX", "CTA", "Reversal", "Cash"]

# ── Data Download ────────────────────────────────────────────────────────────

def download_data():
    """Download all required tickers from yfinance."""
    tickers = ["UPRO", "SPY", "^VIX", "GLD", "SLV", "USO", "UNG", "DBA", "COPX",
               "UUP", "TLT", "EEM",
               # Sector ETFs for reversal proxy
               "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
               "XLRE", "XLC", "IWM", "QQQ", "IYT", "IBB", "VNQ", "HYG", "LQD", "AGG", "TIP"]

    print("Downloading data from yfinance...")
    data = yf.download(tickers, start="2013-01-01", end="2026-07-17", auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns (Price, Ticker)
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    close = close.ffill().dropna(how="all")
    print(f"  Data range: {close.index[0].date()} to {close.index[-1].date()} ({len(close)} days)")
    return close


# ── Sleeve Return Series ─────────────────────────────────────────────────────

def compute_upro_vix_gated(close):
    """UPRO returns gated by VIX level. Daily."""
    upro_ret = close["UPRO"].pct_change()
    vix = close["^VIX"]

    # Allocation fraction based on prior-day VIX (no look-ahead)
    vix_prev = vix.shift(1)
    alloc = pd.Series(0.0, index=vix_prev.index)
    alloc[vix_prev < 17] = 1.0
    alloc[(vix_prev >= 17) & (vix_prev < 25)] = 0.30
    alloc[vix_prev >= 25] = 0.0

    gated = upro_ret * alloc
    gated = gated.dropna()
    return gated


def compute_cta_trend(close):
    """CTA trend following: SMA50 on 9 commodity/macro ETFs, equal weight."""
    cta_tickers = ["GLD", "SLV", "USO", "UNG", "DBA", "COPX", "UUP", "TLT", "EEM"]

    daily_rets = []
    for t in cta_tickers:
        if t not in close.columns:
            continue
        price = close[t]
        sma50 = price.rolling(50).mean()
        ret = price.pct_change()
        # Long when price > SMA50 (prior day signal), else flat
        signal = (price.shift(1) > sma50.shift(1)).astype(float)
        daily_rets.append(ret * signal)

    cta = pd.concat(daily_rets, axis=1).mean(axis=1)
    return cta.dropna()


def compute_etf_reversal(close):
    """ETF weekly reversal: buy 5 worst weekly performers from ~21 ETFs.
    Approximated from actual sector ETF data."""
    rev_tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB",
                   "XLRE", "XLC", "IWM", "QQQ", "SPY", "IYT", "IBB", "VNQ", "HYG",
                   "LQD", "AGG", "TIP"]
    available = [t for t in rev_tickers if t in close.columns]

    prices = close[available]
    daily_ret = prices.pct_change()
    weekly_ret = prices.pct_change(5)  # 5-day trailing return

    # Each day: buy the 5 worst trailing-5d performers, hold 1 day
    n_hold = 5
    port_ret = []
    for i in range(6, len(prices)):
        prev_week = weekly_ret.iloc[i - 1]
        valid = prev_week.dropna()
        if len(valid) < n_hold:
            port_ret.append(0.0)
            continue
        losers = valid.nsmallest(n_hold).index
        day_return = daily_ret.iloc[i][losers].mean()
        port_ret.append(day_return if not np.isnan(day_return) else 0.0)

    idx = prices.index[6:]
    return pd.Series(port_ret, index=idx)


def build_sleeve_returns(close):
    """Build aligned daily return series for all 4 sleeves."""
    upro = compute_upro_vix_gated(close)
    cta = compute_cta_trend(close)
    rev = compute_etf_reversal(close)

    # Align all series
    idx = upro.index.intersection(cta.index).intersection(rev.index)
    idx = idx.sort_values()

    returns = pd.DataFrame({
        "UPRO_VIX": upro.reindex(idx).fillna(0),
        "CTA": cta.reindex(idx).fillna(0),
        "Reversal": rev.reindex(idx).fillna(0),
        "Cash": 0.0002  # ~5% annual risk-free
    }, index=idx)

    return returns


# ── Rebalancing Strategies ───────────────────────────────────────────────────

def simulate_portfolio(returns, strategy, params=None):
    """
    Simulate portfolio with given rebalancing strategy.
    Returns dict with equity curve, metrics, rebalance count, turnover.
    """
    n_days = len(returns)
    n_sleeves = 4
    weights = TARGET_WEIGHTS.copy().astype(float)

    equity = np.ones(n_days)
    weight_history = np.zeros((n_days, n_sleeves))
    weight_history[0] = weights.copy()
    rebalance_days = []
    total_turnover = 0.0

    ret_matrix = returns[SLEEVE_NAMES].values
    dates = returns.index

    for i in range(1, n_days):
        # Grow weights by daily returns
        growth = 1.0 + ret_matrix[i]
        new_vals = weights * growth
        port_val = new_vals.sum()

        if port_val <= 0:
            equity[i] = equity[i-1] * 0.01
            weight_history[i] = weights.copy()
            continue

        weights = new_vals / port_val
        equity[i] = equity[i-1] * port_val

        # Check if we should rebalance
        do_rebal = False

        if strategy == "never":
            do_rebal = False

        elif strategy == "monthly":
            # Rebalance on first trading day of each month
            if i > 0 and dates[i].month != dates[i-1].month:
                do_rebal = True

        elif strategy == "quarterly":
            if i > 0 and dates[i].month != dates[i-1].month and dates[i].month in [1, 4, 7, 10]:
                do_rebal = True

        elif strategy == "threshold":
            threshold = params.get("threshold", 0.05)
            drift = np.abs(weights - TARGET_WEIGHTS)
            if drift.max() > threshold:
                do_rebal = True

        elif strategy == "counter_trend":
            # Rebalance when best performer has a 5% drawdown
            if i > 20:
                lookback = min(i, 60)
                sleeve_cum = np.ones(n_sleeves)
                for j in range(max(0, i - lookback), i + 1):
                    sleeve_cum *= (1.0 + ret_matrix[j])
                sleeve_peak = sleeve_cum  # simplified: use cumulative as proxy
                best_sleeve = np.argmax(weights - TARGET_WEIGHTS)  # most overweight
                # Check if overweight sleeve had recent drawdown
                recent_rets = ret_matrix[max(0, i-5):i+1, best_sleeve]
                if len(recent_rets) > 0:
                    cum_recent = np.prod(1.0 + recent_rets) - 1.0
                    if cum_recent < -0.05 and (weights[best_sleeve] - TARGET_WEIGHTS[best_sleeve]) > 0.03:
                        do_rebal = True

        elif strategy == "calendar_threshold":
            threshold = params.get("threshold", 0.075)
            drift = np.abs(weights - TARGET_WEIGHTS)
            # Quarterly OR drift > threshold
            quarterly = (i > 0 and dates[i].month != dates[i-1].month and dates[i].month in [1, 4, 7, 10])
            if quarterly or drift.max() > threshold:
                do_rebal = True

        if do_rebal:
            turnover = np.abs(weights - TARGET_WEIGHTS).sum() / 2.0
            cost = turnover * REBAL_COST_BPS / 10000.0
            equity[i] *= (1.0 - cost)
            total_turnover += turnover
            rebalance_days.append(i)
            weights = TARGET_WEIGHTS.copy()

        weight_history[i] = weights.copy()

    # Compute metrics
    eq_series = pd.Series(equity, index=dates)
    daily_rets = eq_series.pct_change().dropna()

    ann_ret = (equity[-1] / equity[0]) ** (252.0 / n_days) - 1.0
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Sortino
    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0.0

    # CAGR
    years = n_days / 252.0
    cagr = (equity[-1] / equity[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    return {
        "equity": eq_series,
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "ann_vol": ann_vol,
        "n_rebalances": len(rebalance_days),
        "total_turnover": total_turnover,
        "avg_annual_turnover": total_turnover / years if years > 0 else 0,
        "final_equity": equity[-1],
        "weight_history": weight_history,
    }


# ── Run All Strategies ───────────────────────────────────────────────────────

STRATEGIES = {
    "Never Rebalance": ("never", None),
    "Monthly": ("monthly", None),
    "Quarterly": ("quarterly", None),
    "Threshold 5%": ("threshold", {"threshold": 0.05}),
    "Threshold 10%": ("threshold", {"threshold": 0.10}),
    "Counter-Trend": ("counter_trend", None),
    "Calendar+Threshold 7.5%": ("calendar_threshold", {"threshold": 0.075}),
}


def run_all_strategies(returns, label="Full Period"):
    """Run all 7 strategies on the given return series."""
    results = {}
    for name, (strat, params) in STRATEGIES.items():
        res = simulate_portfolio(returns, strat, params)
        results[name] = res
    return results


def print_results_table(results, label):
    """Print formatted results table."""
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    print(f"{'Strategy':<28} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'#Rebal':>7} {'Ann Turn':>9} {'Final $':>9}")
    print(f"{'-'*100}")

    never_cagr = results.get("Never Rebalance", {}).get("cagr", 0)

    for name, r in results.items():
        premium = r["cagr"] - never_cagr
        premium_str = f" (+{premium*100:.2f}%)" if name != "Never Rebalance" else ""
        print(f"{name:<28} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr']*100:>7.2f}% {r['max_dd']*100:>7.1f}% {r['n_rebalances']:>7d} {r['avg_annual_turnover']:>8.2f}x  ${r['final_equity']:>7.2f}{premium_str}")

    return never_cagr


# ── Walk-Forward Sub-Period Validation ───────────────────────────────────────

def walk_forward_validation(returns):
    """Split into 3 sub-periods and validate each."""
    n = len(returns)
    third = n // 3

    periods = [
        ("Period 1 (earliest third)", returns.iloc[:third]),
        ("Period 2 (middle third)", returns.iloc[third:2*third]),
        ("Period 3 (latest third)", returns.iloc[2*third:]),
    ]

    all_period_results = {}
    rebal_wins = {name: 0 for name in STRATEGIES.keys() if name != "Never Rebalance"}

    for plabel, pdata in periods:
        daterange = f"{pdata.index[0].date()} to {pdata.index[-1].date()}"
        full_label = f"{plabel}: {daterange}"
        res = run_all_strategies(pdata, full_label)
        print_results_table(res, full_label)
        all_period_results[plabel] = res

        never_sharpe = res["Never Rebalance"]["sharpe"]
        for name in rebal_wins:
            if res[name]["sharpe"] > never_sharpe:
                rebal_wins[name] += 1

    print(f"\n{'='*100}")
    print("  Walk-Forward Validation Summary: # periods where strategy beats Never-Rebalance (Sharpe)")
    print(f"{'='*100}")
    for name, wins in rebal_wins.items():
        status = "PASS (all 3)" if wins == 3 else f"PARTIAL ({wins}/3)"
        print(f"  {name:<28}: {wins}/3 periods  [{status}]")

    return all_period_results, rebal_wins


# ── Permutation Test ─────────────────────────────────────────────────────────

def permutation_test(returns, n_shuffles=100):
    """
    Shuffle component return series independently to break correlations,
    then test if rebalancing premium is significant.
    """
    print(f"\n{'='*100}")
    print(f"  Permutation Test ({n_shuffles} shuffles)")
    print(f"{'='*100}")

    # Actual rebalancing premium (monthly vs never)
    actual_never = simulate_portfolio(returns, "never")
    actual_monthly = simulate_portfolio(returns, "monthly")
    actual_premium = actual_monthly["cagr"] - actual_never["cagr"]

    # Also test threshold 5%
    actual_thresh = simulate_portfolio(returns, "threshold", {"threshold": 0.05})
    actual_thresh_premium = actual_thresh["cagr"] - actual_never["cagr"]

    print(f"  Actual monthly rebalancing premium: {actual_premium*100:.3f}% CAGR")
    print(f"  Actual threshold-5% rebalancing premium: {actual_thresh_premium*100:.3f}% CAGR")

    shuffle_premiums_monthly = []
    shuffle_premiums_thresh = []

    np.random.seed(42)

    for s in range(n_shuffles):
        if (s + 1) % 20 == 0:
            print(f"    Shuffle {s+1}/{n_shuffles}...")

        # Shuffle each sleeve's returns independently
        shuffled = returns.copy()
        for col in SLEEVE_NAMES[:3]:  # Don't shuffle cash
            vals = shuffled[col].values.copy()
            np.random.shuffle(vals)
            shuffled[col] = vals

        s_never = simulate_portfolio(shuffled, "never")
        s_monthly = simulate_portfolio(shuffled, "monthly")
        s_thresh = simulate_portfolio(shuffled, "threshold", {"threshold": 0.05})

        shuffle_premiums_monthly.append(s_monthly["cagr"] - s_never["cagr"])
        shuffle_premiums_thresh.append(s_thresh["cagr"] - s_never["cagr"])

    # p-value: fraction of shuffled premiums >= actual premium
    p_monthly = np.mean(np.array(shuffle_premiums_monthly) >= actual_premium)
    p_thresh = np.mean(np.array(shuffle_premiums_thresh) >= actual_thresh_premium)

    mean_shuf_m = np.mean(shuffle_premiums_monthly) * 100
    std_shuf_m = np.std(shuffle_premiums_monthly) * 100
    mean_shuf_t = np.mean(shuffle_premiums_thresh) * 100
    std_shuf_t = np.std(shuffle_premiums_thresh) * 100

    print(f"\n  Monthly rebalance premium:")
    print(f"    Actual: {actual_premium*100:.3f}%")
    print(f"    Shuffled mean: {mean_shuf_m:.3f}% +/- {std_shuf_m:.3f}%")
    print(f"    p-value: {p_monthly:.3f} ({'SIGNIFICANT' if p_monthly < 0.05 else 'NOT significant'} at 5%)")

    print(f"\n  Threshold-5% rebalance premium:")
    print(f"    Actual: {actual_thresh_premium*100:.3f}%")
    print(f"    Shuffled mean: {mean_shuf_t:.3f}% +/- {std_shuf_t:.3f}%")
    print(f"    p-value: {p_thresh:.3f} ({'SIGNIFICANT' if p_thresh < 0.05 else 'NOT significant'} at 5%)")

    return {
        "monthly": {
            "actual_premium_pct": round(actual_premium * 100, 4),
            "shuffled_mean_pct": round(mean_shuf_m, 4),
            "shuffled_std_pct": round(std_shuf_m, 4),
            "p_value": round(p_monthly, 4),
            "significant_5pct": bool(p_monthly < 0.05),
        },
        "threshold_5pct": {
            "actual_premium_pct": round(actual_thresh_premium * 100, 4),
            "shuffled_mean_pct": round(mean_shuf_t, 4),
            "shuffled_std_pct": round(std_shuf_t, 4),
            "p_value": round(p_thresh, 4),
            "significant_5pct": bool(p_thresh < 0.05),
        }
    }


# ── Correlation Analysis ────────────────────────────────────────────────────

def correlation_analysis(returns):
    """Analyze inter-sleeve correlations (the driver of rebalancing premium)."""
    print(f"\n{'='*100}")
    print("  Inter-Sleeve Correlation Matrix (daily returns)")
    print(f"{'='*100}")
    corr = returns[SLEEVE_NAMES[:3]].corr()
    print(corr.round(3).to_string())

    # Rolling correlation UPRO vs CTA
    rolling_corr = returns["UPRO_VIX"].rolling(63).corr(returns["CTA"])
    print(f"\n  UPRO-CTA rolling 63d correlation:")
    print(f"    Mean: {rolling_corr.mean():.3f}")
    print(f"    Min:  {rolling_corr.min():.3f}")
    print(f"    Max:  {rolling_corr.max():.3f}")
    print(f"    Negative fraction: {(rolling_corr < 0).mean()*100:.1f}%")

    return corr


# ── Weight Drift Analysis ───────────────────────────────────────────────────

def drift_analysis(results):
    """Analyze how much weights drift under never-rebalance."""
    never = results["Never Rebalance"]
    wh = never["weight_history"]

    print(f"\n{'='*100}")
    print("  Weight Drift Analysis (Never Rebalance)")
    print(f"{'='*100}")

    # Final weights
    final_w = wh[-1]
    print(f"  Target weights: {TARGET_WEIGHTS}")
    print(f"  Final weights:  [{', '.join(f'{w:.3f}' for w in final_w)}]")
    print(f"  Max drift:      {np.abs(final_w - TARGET_WEIGHTS).max()*100:.1f}%")

    # Drift over time (sample at yearly intervals)
    n = len(wh)
    yearly = max(1, n // (n // 252)) if n > 252 else n
    print(f"\n  Weight evolution (yearly snapshots):")
    print(f"  {'Year':>6}  {'UPRO':>8}  {'CTA':>8}  {'Rev':>8}  {'Cash':>8}  {'MaxDrift':>9}")
    for yr in range(0, n, 252):
        idx = min(yr, n - 1)
        w = wh[idx]
        drift = np.abs(w - TARGET_WEIGHTS).max()
        print(f"  {yr//252:>6}  {w[0]:>8.3f}  {w[1]:>8.3f}  {w[2]:>8.3f}  {w[3]:>8.3f}  {drift*100:>8.1f}%")
    # Final
    w = wh[-1]
    drift = np.abs(w - TARGET_WEIGHTS).max()
    print(f"  {'Final':>6}  {w[0]:>8.3f}  {w[1]:>8.3f}  {w[2]:>8.3f}  {w[3]:>8.3f}  {drift*100:>8.1f}%")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("  REBALANCING PREMIUM STUDY")
    print("  Multi-Strategy Portfolio: 50% UPRO(VIX-gated) / 30% CTA / 15% Reversal / 5% Cash")
    print("=" * 100)

    # 1. Download data
    close = download_data()

    # 2. Build sleeve returns
    print("\nBuilding sleeve return series...")
    returns = build_sleeve_returns(close)
    print(f"  Aligned series: {returns.index[0].date()} to {returns.index[-1].date()} ({len(returns)} days)")

    # 3. Sleeve statistics
    print(f"\n  Sleeve annualized stats:")
    for col in SLEEVE_NAMES[:3]:
        r = returns[col]
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        sr = ann / vol if vol > 0 else 0
        print(f"    {col:<12}: CAGR={ann*100:.1f}%  Vol={vol*100:.1f}%  Sharpe={sr:.2f}")

    # 4. Correlation analysis
    corr = correlation_analysis(returns)

    # 5. Run all strategies - full period
    print("\n" + "=" * 100)
    print("  FULL PERIOD RESULTS")
    full_results = run_all_strategies(returns)
    never_cagr = print_results_table(full_results,
        f"Full Period: {returns.index[0].date()} to {returns.index[-1].date()}")

    # 6. Drift analysis
    drift_analysis(full_results)

    # 7. Walk-forward validation
    print("\n" + "=" * 100)
    print("  WALK-FORWARD VALIDATION (3 sub-periods)")
    wf_results, wf_wins = walk_forward_validation(returns)

    # 8. Permutation test
    perm_results = permutation_test(returns, n_shuffles=100)

    # 9. Summary
    print(f"\n{'='*100}")
    print("  FINAL SUMMARY")
    print(f"{'='*100}")

    # Find best strategy
    best_name = max(full_results.keys(), key=lambda k: full_results[k]["sharpe"])
    best = full_results[best_name]
    never = full_results["Never Rebalance"]

    print(f"\n  Best strategy by Sharpe: {best_name}")
    print(f"    Sharpe:  {best['sharpe']:.3f} (vs {never['sharpe']:.3f} never-rebalance)")
    print(f"    CAGR:    {best['cagr']*100:.2f}% (vs {never['cagr']*100:.2f}%)")
    print(f"    MaxDD:   {best['max_dd']*100:.1f}% (vs {never['max_dd']*100:.1f}%)")
    print(f"    Premium: +{(best['cagr'] - never['cagr'])*100:.2f}% CAGR from rebalancing")
    print(f"    Rebalance events: {best['n_rebalances']} over {len(returns)/252:.1f} years")

    # Cost analysis
    gross_premium = best['cagr'] - never['cagr']
    cost_drag = best['n_rebalances'] * REBAL_COST_BPS / 10000 / (len(returns)/252)
    print(f"\n  Cost analysis (5bps per rebalance):")
    print(f"    Annual cost drag: ~{cost_drag*100:.3f}%")
    print(f"    Net premium after costs: +{(gross_premium - cost_drag)*100:.3f}% (already included in numbers above)")

    # Shannon's demon check
    upro_vol = returns["UPRO_VIX"].std() * np.sqrt(252)
    cta_vol = returns["CTA"].std() * np.sqrt(252)
    corr_uc = returns["UPRO_VIX"].corr(returns["CTA"])
    theoretical = 0.5 * (0.5 * 0.3) * (upro_vol**2 + cta_vol**2 - 2 * corr_uc * upro_vol * cta_vol)
    print(f"\n  Shannon's Demon theoretical premium estimate:")
    print(f"    UPRO vol: {upro_vol*100:.1f}%, CTA vol: {cta_vol*100:.1f}%, corr: {corr_uc:.3f}")
    print(f"    Theoretical: ~{theoretical*100:.2f}% annually")
    print(f"    Realized:    ~{gross_premium*100:.2f}% annually")

    # ── Save outputs ─────────────────────────────────────────────────────────

    # JSON summary
    json_out = {
        "study": "Rebalancing Premium",
        "date_run": datetime.now().isoformat(),
        "data_range": f"{returns.index[0].date()} to {returns.index[-1].date()}",
        "n_days": len(returns),
        "target_weights": dict(zip(SLEEVE_NAMES, TARGET_WEIGHTS.tolist())),
        "rebalance_cost_bps": REBAL_COST_BPS,
        "sleeve_correlations": corr.to_dict(),
        "full_period_results": {},
        "best_strategy": best_name,
        "rebalancing_premium_cagr_pct": round((best['cagr'] - never['cagr']) * 100, 4),
        "walk_forward_wins": {k: v for k, v in wf_wins.items()},
        "permutation_test": perm_results,
    }

    for name, r in full_results.items():
        json_out["full_period_results"][name] = {
            "sharpe": round(r["sharpe"], 4),
            "sortino": round(r["sortino"], 4),
            "cagr_pct": round(r["cagr"] * 100, 4),
            "max_dd_pct": round(r["max_dd"] * 100, 2),
            "n_rebalances": r["n_rebalances"],
            "avg_annual_turnover": round(r["avg_annual_turnover"], 4),
            "final_equity": round(r["final_equity"], 4),
        }

    json_path = OUT_DIR / "rebalancing_premium_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\n  Saved JSON: {json_path}")

    # CSV with equity curves
    eq_df = pd.DataFrame({name: r["equity"] for name, r in full_results.items()})
    csv_path = OUT_DIR / "equity_curves.csv"
    eq_df.to_csv(csv_path)
    print(f"  Saved CSV:  {csv_path}")

    # CSV with strategy comparison
    comp_rows = []
    for name, r in full_results.items():
        comp_rows.append({
            "Strategy": name,
            "Sharpe": round(r["sharpe"], 4),
            "Sortino": round(r["sortino"], 4),
            "CAGR_pct": round(r["cagr"] * 100, 4),
            "MaxDD_pct": round(r["max_dd"] * 100, 2),
            "N_Rebalances": r["n_rebalances"],
            "Avg_Annual_Turnover": round(r["avg_annual_turnover"], 4),
            "Final_Equity": round(r["final_equity"], 4),
            "Premium_vs_Never_pct": round((r["cagr"] - never["cagr"]) * 100, 4),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_csv = OUT_DIR / "strategy_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"  Saved CSV:  {comp_csv}")

    print(f"\n{'='*100}")
    print("  STUDY COMPLETE")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
