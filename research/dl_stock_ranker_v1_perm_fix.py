#!/usr/bin/env python3
"""
DL Stock Ranker v1 — Fixed Permutation Test + Regime Analysis
=============================================================
The original permutation test was broken: it shuffled the temporal order of
portfolio returns, which does NOT change Sharpe (Sharpe is order-invariant).
Result: null_std = 0.0, p = 1.0 for all permutations.

CORRECT approach: For each permutation, randomly select 5 stocks (instead of
the model's top-5) in each rebalancing period, compute the portfolio return,
then compute Sharpe across all periods. This tests whether the MODEL'S stock
selection adds value vs random stock picks.

Author: Claude Opus 4.6 | Date: 2026-07-24
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESEARCH_DIR = Path("/home/jupiter/Lvl3Quant/research")
RESULTS_CSV = RESEARCH_DIR / "attention_results.csv"
PRICE_CACHE = RESEARCH_DIR / "price_cache.pkl"
TOP_K = 5
COST_BPS = 10
N_PERMUTATIONS = 200
ANN_FACTOR = 12.0  # monthly returns -> annual

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "GS", "BAC",
    "V", "MA", "UNH", "JNJ", "PG", "KO", "PEP", "MRK", "ABBV", "LLY",
    "HD", "COST", "WMT", "CRM", "AMD", "NFLX", "ADBE", "INTC", "CSCO", "QCOM",
    "XOM", "CVX", "PFE", "TMO", "ABT", "AVGO", "TXN", "MCD", "NKE", "DIS",
    "CMCSA", "T", "VZ", "NEE", "SO", "SHW", "LMT", "RTX", "CAT", "DE",
]


def compute_metrics(returns, ann_factor=12.0):
    """Compute risk-adjusted metrics for monthly returns."""
    if len(returns) < 2:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "cagr": 0,
                "max_dd": 0, "n_periods": len(returns)}

    ret = np.array(returns, dtype=float)
    mean_ret = ret.mean()
    std_ret = ret.std(ddof=1)

    if std_ret < 1e-10:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "cagr": 0,
                "max_dd": 0, "n_periods": len(returns)}

    sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor)

    downside = ret[ret < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(ann_factor) if downside_std > 0 else 0

    gross_profit = ret[ret > 0].sum()
    gross_loss = abs(ret[ret < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    wr = (ret > 0).mean()

    cumret = np.cumprod(1 + ret)
    n_years = len(ret) / ann_factor
    cagr = (cumret[-1] ** (1 / n_years) - 1) if n_years > 0 and cumret[-1] > 0 else 0

    cummax = np.maximum.accumulate(cumret)
    dd = (cumret - cummax) / cummax
    max_dd = dd.min()

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "n_periods": len(returns),
    }


def compute_period_stock_returns(close, results_df, tickers):
    """
    For each rebalancing period, compute the return of every stock.
    Returns: list of dicts, one per period, mapping ticker -> return.
    """
    period_returns = []

    for _, row in results_df.iterrows():
        rebal_date = pd.Timestamp(row["rebal_date"])
        end_date = pd.Timestamp(row["end_date"])

        stock_rets = {}
        for ticker in tickers:
            if ticker not in close.columns:
                continue
            # Get prices at rebal and end dates (find nearest available)
            mask_start = close.index >= rebal_date
            mask_end = close.index <= end_date

            avail_start = close.index[mask_start]
            avail_end = close.index[mask_end]

            if len(avail_start) == 0 or len(avail_end) == 0:
                stock_rets[ticker] = 0.0
                continue

            start_price = close.loc[avail_start[0], ticker]
            end_price = close.loc[avail_end[-1], ticker]

            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                stock_rets[ticker] = (end_price / start_price) - 1
            else:
                stock_rets[ticker] = 0.0

        period_returns.append(stock_rets)

    return period_returns


def run_fixed_permutation_test(results_df, period_stock_returns, n_perm=200, top_k=5, cost_bps=10):
    """
    FIXED permutation test.

    For each permutation:
      - For each period, randomly pick top_k stocks (instead of model's picks)
      - Compute equal-weight portfolio return (minus costs)
      - Compute Sharpe across all periods

    Also handles the asymmetric filter: when filter_active is False, the
    portfolio holds SPY regardless of stock selection. So we only randomize
    stock picks in periods where the filter was active.

    Null hypothesis: model's stock selection is no better than random.
    """
    cost = cost_bps / 10000
    n_periods = len(results_df)

    # Get the observed (model) returns
    observed_rets = results_df["portfolio_ret"].values
    observed_sharpe = compute_metrics(observed_rets)["sharpe"]

    # Identify which periods had active stock selection vs SPY hold
    filter_active = results_df["filter_active"].values.astype(bool) if "filter_active" in results_df.columns else np.ones(n_periods, dtype=bool)
    spy_rets = results_df["spy_ret"].values

    # Available tickers per period
    available_tickers_per_period = []
    for pr in period_stock_returns:
        available = [t for t, r in pr.items() if not np.isnan(r)]
        available_tickers_per_period.append(available)

    rng = np.random.default_rng(seed=42)
    null_sharpes = []

    for perm_i in range(n_perm):
        perm_rets = np.zeros(n_periods)

        for t in range(n_periods):
            if not filter_active[t]:
                # Filter inactive -> hold SPY (same as original)
                perm_rets[t] = spy_rets[t]
            else:
                # Randomly pick top_k stocks
                avail = available_tickers_per_period[t]
                if len(avail) < top_k:
                    picks = avail
                else:
                    picks = rng.choice(avail, size=top_k, replace=False).tolist()

                # Equal-weight return
                pick_rets = [period_stock_returns[t].get(p, 0.0) for p in picks]
                perm_rets[t] = np.mean(pick_rets) - cost

        s = compute_metrics(perm_rets)["sharpe"]
        null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= observed_sharpe).mean()

    return {
        "observed_sharpe": observed_sharpe,
        "null_mean_sharpe": round(null_sharpes.mean(), 4),
        "null_std_sharpe": round(null_sharpes.std(), 4),
        "null_median_sharpe": round(np.median(null_sharpes), 4),
        "null_p5": round(np.percentile(null_sharpes, 5), 4),
        "null_p95": round(np.percentile(null_sharpes, 95), 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_perm,
        "n_active_periods": int(filter_active.sum()),
        "n_total_periods": n_periods,
        "pass": p_value < 0.05,
    }


def run_regime_test(results_df, close, benchmark="SPY"):
    """
    Regime test: compare Sharpe in bull/bear/flat SPY regimes.
    Bull: SPY 63d return > 5%, Bear: < -5%, Flat: in between.
    Also does simple green/red (> 0 / <= 0).
    """
    spy_ret_63d = close[benchmark].pct_change(63)

    bull_rets, bear_rets, flat_rets = [], [], []
    green_rets, red_rets = [], []

    for _, row in results_df.iterrows():
        rebal_date = pd.Timestamp(row["rebal_date"])
        closest = spy_ret_63d.index[spy_ret_63d.index <= rebal_date]
        if len(closest) == 0:
            continue
        regime_val = spy_ret_63d.loc[closest[-1]]
        if np.isnan(regime_val):
            continue

        port_ret = row["portfolio_ret"]

        # Three-way regime
        if regime_val > 0.05:
            bull_rets.append(port_ret)
        elif regime_val < -0.05:
            bear_rets.append(port_ret)
        else:
            flat_rets.append(port_ret)

        # Two-way regime
        if regime_val > 0:
            green_rets.append(port_ret)
        else:
            red_rets.append(port_ret)

    bull_m = compute_metrics(bull_rets) if len(bull_rets) > 2 else {"sharpe": 0}
    bear_m = compute_metrics(bear_rets) if len(bear_rets) > 2 else {"sharpe": 0}
    flat_m = compute_metrics(flat_rets) if len(flat_rets) > 2 else {"sharpe": 0}
    green_m = compute_metrics(green_rets) if len(green_rets) > 2 else {"sharpe": 0}
    red_m = compute_metrics(red_rets) if len(red_rets) > 2 else {"sharpe": 0}

    # HC #428 gap test
    sharpe_bull = bull_m["sharpe"]
    sharpe_bear = bear_m["sharpe"]
    max_s = max(abs(sharpe_bull), abs(sharpe_bear), 1e-10)
    gap_3way = abs(sharpe_bull - sharpe_bear) / max_s

    sharpe_green = green_m["sharpe"]
    sharpe_red = red_m["sharpe"]
    max_s2 = max(abs(sharpe_green), abs(sharpe_red), 1e-10)
    gap_2way = abs(sharpe_green - sharpe_red) / max_s2

    return {
        "three_way": {
            "bull": {"sharpe": sharpe_bull, "n": len(bull_rets), "avg_ret": round(np.mean(bull_rets), 5) if bull_rets else 0},
            "bear": {"sharpe": sharpe_bear, "n": len(bear_rets), "avg_ret": round(np.mean(bear_rets), 5) if bear_rets else 0},
            "flat": {"sharpe": flat_m["sharpe"], "n": len(flat_rets), "avg_ret": round(np.mean(flat_rets), 5) if flat_rets else 0},
            "gap": round(gap_3way, 4),
            "pass": gap_3way < 0.50,
        },
        "two_way": {
            "green": {"sharpe": sharpe_green, "n": len(green_rets), "avg_ret": round(np.mean(green_rets), 5) if green_rets else 0},
            "red": {"sharpe": sharpe_red, "n": len(red_rets), "avg_ret": round(np.mean(red_rets), 5) if red_rets else 0},
            "gap": round(gap_2way, 4),
            "pass": gap_2way < 0.50,
        },
    }


def main():
    print("=" * 70)
    print("DL Stock Ranker v1 — FIXED Permutation Test + Regime Analysis")
    print("=" * 70)

    # Load results
    print("\n[1] Loading attention results...")
    results_df = pd.read_csv(RESULTS_CSV)
    print(f"    Loaded {len(results_df)} periods from attention_results.csv")
    print(f"    Date range: {results_df['rebal_date'].iloc[0]} to {results_df['end_date'].iloc[-1]}")

    # Load price cache
    print("\n[2] Loading price cache...")
    close = pd.read_pickle(PRICE_CACHE)
    print(f"    Price data: {close.shape[0]} days x {close.shape[1]} instruments")

    # Verify we have the right tickers
    available = [t for t in TICKERS if t in close.columns]
    print(f"    Available tickers: {len(available)}/{len(TICKERS)}")

    # Compute per-period stock returns
    print("\n[3] Computing per-period stock returns for all 50 stocks...")
    period_stock_returns = compute_period_stock_returns(close, results_df, TICKERS)

    # Sanity check: verify model returns match
    print("\n[4] Sanity check — verifying model portfolio returns...")
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        selected = row["selected"].split(",")
        filter_active = row.get("filter_active", True)

        if filter_active and filter_active != False:
            pick_rets = [period_stock_returns[i].get(t, 0.0) for t in selected]
            recomputed = np.mean(pick_rets) - COST_BPS / 10000
            orig = row["portfolio_ret"]
            diff = abs(recomputed - orig)
            status = "OK" if diff < 0.001 else f"MISMATCH (diff={diff:.6f})"
            print(f"    Period {i}: orig={orig:.6f}, recomputed={recomputed:.6f} -> {status}")
        else:
            print(f"    Period {i}: filter inactive (holding SPY), skipping verification")

    # -----------------------------------------------------------------------
    # EXPLAIN THE BUG
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BUG DIAGNOSIS")
    print("=" * 70)
    print("""
    The original permutation test (lines 1008-1033 of dl_stock_ranker_v1.py)
    shuffled the TEMPORAL ORDER of portfolio returns:

        shuffled = returns.sample(frac=1, replace=False).reset_index(drop=True)
        null_sharpes.append(compute_metrics(shuffled)["sharpe"])

    This is wrong because Sharpe = mean/std, and both mean and std are
    ORDER-INVARIANT. Shuffling the order of returns does not change the
    Sharpe ratio at all. Result: every permutation produces the exact same
    Sharpe -> null_std = 0.0, p = 1.0.

    CORRECT approach: For each permutation, randomly select 5 stocks
    (instead of the model's top-5) in each period. This tests whether the
    model's STOCK SELECTION skill is real or just luck.
    """)

    # -----------------------------------------------------------------------
    # FIXED PERMUTATION TEST
    # -----------------------------------------------------------------------
    print("=" * 70)
    print(f"FIXED PERMUTATION TEST ({N_PERMUTATIONS} shuffles)")
    print("=" * 70)

    perm = run_fixed_permutation_test(
        results_df, period_stock_returns,
        n_perm=N_PERMUTATIONS, top_k=TOP_K, cost_bps=COST_BPS
    )

    print(f"\n    Observed Sharpe:     {perm['observed_sharpe']:.4f}")
    print(f"    Null mean Sharpe:    {perm['null_mean_sharpe']:.4f}")
    print(f"    Null std Sharpe:     {perm['null_std_sharpe']:.4f}")
    print(f"    Null median Sharpe:  {perm['null_median_sharpe']:.4f}")
    print(f"    Null 5th-95th pctl:  [{perm['null_p5']:.4f}, {perm['null_p95']:.4f}]")
    print(f"    p-value:             {perm['p_value']:.4f}")
    print(f"    Active periods:      {perm['n_active_periods']}/{perm['n_total_periods']}")
    print(f"    RESULT:              {'PASS (p < 0.05) — genuine skill' if perm['pass'] else 'FAIL (p >= 0.05) — no evidence of skill'}")

    # Also run a STRICTER test: only consider active (filter_active=True) periods
    print(f"\n--- Strict test (active periods only) ---")
    active_mask = results_df["filter_active"].values.astype(bool) if "filter_active" in results_df.columns else np.ones(len(results_df), dtype=bool)
    n_active = active_mask.sum()
    if n_active > 5:
        active_results = results_df[active_mask].reset_index(drop=True)
        active_period_rets = [period_stock_returns[i] for i in range(len(results_df)) if active_mask[i]]

        # For active-only test, force all periods to be "active"
        active_results_copy = active_results.copy()
        active_results_copy["filter_active"] = True

        perm_strict = run_fixed_permutation_test(
            active_results_copy, active_period_rets,
            n_perm=N_PERMUTATIONS, top_k=TOP_K, cost_bps=COST_BPS
        )
        print(f"    Active-only observed Sharpe: {perm_strict['observed_sharpe']:.4f}")
        print(f"    Active-only null mean:       {perm_strict['null_mean_sharpe']:.4f}")
        print(f"    Active-only null std:         {perm_strict['null_std_sharpe']:.4f}")
        print(f"    Active-only p-value:         {perm_strict['p_value']:.4f}")
        print(f"    RESULT:                      {'PASS' if perm_strict['pass'] else 'FAIL'}")
    else:
        perm_strict = None
        print(f"    Only {n_active} active periods — too few for standalone test")

    # -----------------------------------------------------------------------
    # REGIME TEST
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS")
    print("=" * 70)

    regime = run_regime_test(results_df, close)

    print("\n  Three-way regime (bull > +5%, bear < -5%, flat):")
    for r_name in ["bull", "bear", "flat"]:
        r = regime["three_way"][r_name]
        print(f"    {r_name:5s}: Sharpe={r['sharpe']:6.3f}, n={r['n']:3d}, avg_ret={r['avg_ret']:+.5f}")
    print(f"    Gap (bull-bear): {regime['three_way']['gap']:.4f} -> {'PASS' if regime['three_way']['pass'] else 'FAIL'} (threshold: 0.50)")

    print("\n  Two-way regime (green > 0%, red <= 0%):")
    for r_name in ["green", "red"]:
        r = regime["two_way"][r_name]
        print(f"    {r_name:5s}: Sharpe={r['sharpe']:6.3f}, n={r['n']:3d}, avg_ret={r['avg_ret']:+.5f}")
    print(f"    Gap (green-red): {regime['two_way']['gap']:.4f} -> {'PASS' if regime['two_way']['pass'] else 'FAIL'} (threshold: 0.50)")

    # -----------------------------------------------------------------------
    # ADDITIONAL: concentration check
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STOCK SELECTION ANALYSIS")
    print("=" * 70)

    # Count how often each stock is selected
    all_selections = []
    for _, row in results_df.iterrows():
        if row.get("filter_active", True):
            all_selections.extend(row["selected"].split(","))

    if all_selections:
        from collections import Counter
        counts = Counter(all_selections)
        total = len(all_selections)
        print(f"\n  Total stock selections across active periods: {total}")
        print(f"  Unique stocks selected: {len(counts)}/{len(TICKERS)}")
        print(f"\n  Top 10 most selected:")
        for ticker, count in counts.most_common(10):
            pct = count / total * 100
            print(f"    {ticker:6s}: {count:3d} times ({pct:5.1f}%)")

        # Herfindahl concentration
        shares = np.array([c / total for c in counts.values()])
        hhi = (shares ** 2).sum()
        print(f"\n  HHI concentration: {hhi:.4f} (1/{len(TICKERS)}={1/len(TICKERS):.4f} = perfectly diversified)")

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    overall_metrics = compute_metrics(results_df["portfolio_ret"].values)

    perm_pass = perm["pass"]
    regime_pass = regime["three_way"]["pass"] and regime["two_way"]["pass"]

    print(f"\n  Attention Model (filtered):")
    print(f"    Sharpe:  {overall_metrics['sharpe']}")
    print(f"    Sortino: {overall_metrics['sortino']}")
    print(f"    PF:      {overall_metrics['pf']}")
    print(f"    WR:      {overall_metrics['wr']}")
    print(f"    CAGR:    {overall_metrics['cagr']:.2%}")
    print(f"    MaxDD:   {overall_metrics['max_dd']:.2%}")

    print(f"\n  Validation:")
    print(f"    Permutation test:  {'PASS' if perm_pass else 'FAIL'} (p={perm['p_value']})")
    print(f"    Regime test (3w):  {'PASS' if regime['three_way']['pass'] else 'FAIL'} (gap={regime['three_way']['gap']})")
    print(f"    Regime test (2w):  {'PASS' if regime['two_way']['pass'] else 'FAIL'} (gap={regime['two_way']['gap']})")

    all_pass = perm_pass and regime_pass
    if all_pass:
        print(f"\n  VERDICT: VALIDATED — The DL Stock Ranker shows genuine stock-selection")
        print(f"  skill beyond random picking, and works across market regimes.")
    else:
        reasons = []
        if not perm_pass:
            reasons.append(f"permutation test failed (p={perm['p_value']}, random picks achieve similar Sharpe)")
        if not regime["three_way"]["pass"]:
            reasons.append(f"regime-dependent (bull-bear gap={regime['three_way']['gap']:.2f})")
        if not regime["two_way"]["pass"]:
            reasons.append(f"regime-dependent (green-red gap={regime['two_way']['gap']:.2f})")
        print(f"\n  VERDICT: NOT FULLY VALIDATED")
        for r in reasons:
            print(f"    - {r}")

    # Save results
    output = {
        "bug_explanation": "Original test shuffled temporal order of returns (order-invariant for Sharpe). Fixed: shuffle stock selection.",
        "fixed_permutation_test": perm,
        "strict_permutation_test": perm_strict,
        "regime_analysis": regime,
        "overall_metrics": overall_metrics,
        "validated": all_pass,
    }

    out_path = RESEARCH_DIR / "dl_stock_ranker_v1_perm_fix_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
