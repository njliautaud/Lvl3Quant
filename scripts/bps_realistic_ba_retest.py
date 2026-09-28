#!/usr/bin/env python3
"""
BPS Realistic Bid-Ask Cost Sensitivity Analysis
================================================
Re-runs the full-stack BPS backtest across multiple BA cost scenarios
based on live validation findings (prior 5% BA was 3x too low).

Live validation showed:
  - Tier 1 (large-cap): 16-18% median real BA
  - Tier 2 (mid-cap): 24-58% median real BA

Scenarios tested:
  1. 5% flat (original, comparison baseline)
  2. 10% flat (optimistic: patient limit orders with good fills)
  3. 15% flat (realistic: live validation median for tier 1)
  4. 20% flat (pessimistic / conservative)
  5. Tiered realistic: tier-1 at 15%, tier-2 at 30%
  6. Tier-1 only at 15% (70 tickers) vs tiered 98 tickers
  7. Break-even analysis: find BA% where Sharpe hits 0

Output: output/bps_realistic_ba/
"""

import sys, json, time, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "scripts"))

# Import everything from the original backtest
from bps_full_stack_backtest import (
    TIER1_TICKERS, TIER2_TICKERS, TIER2_SECTORS,
    STARTING_CAPITAL, DTE_TARGET,
    bs_price, strike_from_delta, trade_cost,
    COST_PER_CONTRACT, SLIPPAGE_FRAC, SLIPPAGE_MIN,
    load_all_data, build_earnings_lookup, has_earnings_within,
    dynamic_aggressive, fixed_d30,
    run_bps, compute_full_metrics, run_permutation_test,
)

OUTPUT = ROOT / "output" / "bps_realistic_ba"
OUTPUT.mkdir(parents=True, exist_ok=True)


def run_scenario(prices, iv, macro, fund, universe, earnings_lookup,
                 ticker_list, ticker_tier_map, ba_costs, label,
                 delta_fn=dynamic_aggressive):
    """Run full-stack backtest with given BA costs and return metrics."""
    result = run_bps(
        prices, iv, macro, fund, universe, earnings_lookup,
        ticker_list=ticker_list,
        ticker_tier_map=ticker_tier_map,
        ba_costs=ba_costs,
        delta_fn=delta_fn,
        label=label,
        profit_take=0.65, margin_cap=0.25,
        vix_scale=True, vix_base=15.0,
        vix_hard_cutoff=30,
        portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
        earnings_filter=True, earnings_buffer_days=7,
    )

    if "error" in result:
        return {"label": label, "error": result["error"]}

    metrics = compute_full_metrics(result["equity_df"], result["trades_df"], label)
    metrics["n_earnings_blocked"] = result.get("n_earnings_blocked", 0)
    metrics["n_vix_blocked_days"] = result.get("n_vix_blocked", 0)
    metrics["n_cb_frozen_days"] = result.get("n_cb_frozen", 0)
    metrics["cb_triggers"] = result.get("cb_triggers", 0)

    return {
        "label": label,
        "metrics": metrics,
        "equity_df": result["equity_df"],
        "trades_df": result["trades_df"],
    }


def find_breakeven_ba(prices, iv, macro, fund, universe, earnings_lookup,
                      ticker_list, ticker_tier_map, delta_fn=dynamic_aggressive):
    """Binary search for the BA% where Sharpe crosses 0."""
    print("\n  Finding break-even BA% (binary search)...")

    def sharpe_at_ba(ba_pct):
        ba_frac = ba_pct / 100.0
        ba_costs = {"tier1": ba_frac, "tier2": ba_frac}
        result = run_bps(
            prices, iv, macro, fund, universe, earnings_lookup,
            ticker_list=ticker_list,
            ticker_tier_map=ticker_tier_map,
            ba_costs=ba_costs,
            delta_fn=delta_fn,
            label=f"breakeven_ba_{ba_pct:.1f}pct",
            profit_take=0.65, margin_cap=0.25,
            vix_scale=True, vix_base=15.0,
            vix_hard_cutoff=30,
            portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
            earnings_filter=True, earnings_buffer_days=7,
        )
        if "error" in result:
            return -1.0
        eq = result["equity_df"].copy().sort_values("date").reset_index(drop=True)
        rets = eq["equity"].pct_change().dropna()
        if rets.std() <= 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(252))

    # First check endpoints
    s_low = sharpe_at_ba(1.0)
    s_high = sharpe_at_ba(50.0)
    print(f"    Sharpe at 1% BA: {s_low:.3f}")
    print(f"    Sharpe at 50% BA: {s_high:.3f}")

    if s_low <= 0:
        print("    Strategy unprofitable even at 1% BA!")
        return 1.0, s_low
    if s_high > 0:
        print("    Strategy still profitable at 50% BA! Break-even > 50%")
        return 50.0, s_high

    # Binary search
    lo, hi = 1.0, 50.0
    for iteration in range(15):  # ~0.003% precision
        mid = (lo + hi) / 2.0
        s_mid = sharpe_at_ba(mid)
        print(f"    Iteration {iteration+1}: BA={mid:.2f}% -> Sharpe={s_mid:.3f}")
        if s_mid > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.1:
            break

    breakeven = (lo + hi) / 2.0
    s_be = sharpe_at_ba(breakeven)
    print(f"    Break-even BA: ~{breakeven:.1f}% (Sharpe={s_be:.3f})")
    return breakeven, s_be


def main():
    t0 = time.time()
    print("=" * 100)
    print("BPS REALISTIC BID-ASK COST SENSITIVITY ANALYSIS")
    print("Prior backtests used 5% BA -- live validation shows 15-18% for tier 1, 24-58% for tier 2")
    print("=" * 100)

    # Load data
    prices, iv, macro, fund, universe, earnings = load_all_data()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    earnings_lookup = build_earnings_lookup(earnings)

    # Build tier maps
    ticker_tier = {}
    for tk in TIER1_TICKERS:
        ticker_tier[tk] = "tier1"
    for tk in TIER2_TICKERS:
        ticker_tier[tk] = "tier2"

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique())
    t1_avail = [t for t in TIER1_TICKERS if t in available]
    t2_avail = [t for t in TIER2_TICKERS if t in available]
    all_avail = t1_avail + t2_avail

    print(f"\n  Available: Tier 1 = {len(t1_avail)}, Tier 2 = {len(t2_avail)}, Total = {len(all_avail)}")

    # ═══════════════════════════════════════════════════════════════
    # SCENARIO DEFINITIONS
    # ═══════════════════════════════════════════════════════════════

    scenarios = [
        {
            "label": "S1_5pct_flat_original",
            "desc": "5% flat BA (original backtest assumption)",
            "tickers": all_avail,
            "ba": {"tier1": 0.05, "tier2": 0.05},
        },
        {
            "label": "S2_10pct_flat_optimistic",
            "desc": "10% flat BA (optimistic: very patient limit orders)",
            "tickers": all_avail,
            "ba": {"tier1": 0.10, "tier2": 0.10},
        },
        {
            "label": "S3_15pct_flat_realistic",
            "desc": "15% flat BA (realistic: live validation tier-1 median)",
            "tickers": all_avail,
            "ba": {"tier1": 0.15, "tier2": 0.15},
        },
        {
            "label": "S4_20pct_flat_pessimistic",
            "desc": "20% flat BA (pessimistic / conservative)",
            "tickers": all_avail,
            "ba": {"tier1": 0.20, "tier2": 0.20},
        },
        {
            "label": "S5_tiered_realistic",
            "desc": "Tiered: tier-1 at 15%, tier-2 at 30% (live validation calibrated)",
            "tickers": all_avail,
            "ba": {"tier1": 0.15, "tier2": 0.30},
        },
        {
            "label": "S6_tier1_only_15pct",
            "desc": "Tier-1 only (70 tickers) at 15% BA -- drop expensive tier-2",
            "tickers": t1_avail,
            "ba": {"tier1": 0.15, "tier2": 0.15},
        },
    ]

    all_results = {}

    for sc in scenarios:
        print("\n" + "=" * 100)
        print(f"SCENARIO: {sc['label']}")
        print(f"  {sc['desc']}")
        print(f"  Tickers: {len(sc['tickers'])}, BA: tier1={sc['ba']['tier1']*100:.0f}%, tier2={sc['ba']['tier2']*100:.0f}%")
        print("=" * 100)

        result = run_scenario(
            prices, iv, macro, fund, universe, earnings_lookup,
            ticker_list=sc["tickers"],
            ticker_tier_map=ticker_tier,
            ba_costs=sc["ba"],
            label=sc["label"],
        )
        all_results[sc["label"]] = result

        # Save equity curve
        if "error" not in result:
            result["equity_df"][["date", "equity"]].to_parquet(
                OUTPUT / f"eq_{sc['label']}.parquet", index=False)
            if not result["trades_df"].empty:
                result["trades_df"].to_parquet(
                    OUTPUT / f"trades_{sc['label']}.parquet", index=False)

    # ═══════════════════════════════════════════════════════════════
    # BREAK-EVEN ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("BREAK-EVEN BA% ANALYSIS")
    print("Finding the BA cost level where Sharpe ratio hits 0")
    print("=" * 100)

    breakeven_ba, breakeven_sharpe = find_breakeven_ba(
        prices, iv, macro, fund, universe, earnings_lookup,
        ticker_list=all_avail, ticker_tier_map=ticker_tier,
    )

    # ═══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 140)
    print("MASTER COMPARISON TABLE -- BA COST SENSITIVITY")
    print("=" * 140)
    header = (f"{'Scenario':<35} {'BA_T1':>6} {'BA_T2':>6} {'#Tkrs':>6} "
              f"{'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} "
              f"{'Calmar':>7} {'PF':>6} {'TradeWR':>8} {'Trades':>7} {'Final$':>12}")
    print(header)
    print("-" * 140)

    for sc in scenarios:
        label = sc["label"]
        r = all_results[label]
        if "error" in r:
            print(f"  {label}: ERROR - {r.get('error', 'unknown')}")
            continue
        m = r["metrics"]
        ba_t1 = sc["ba"]["tier1"] * 100
        ba_t2 = sc["ba"]["tier2"] * 100
        n_tickers = len(sc["tickers"])
        print(f"{label:<35} {ba_t1:>5.0f}% {ba_t2:>5.0f}% {n_tickers:>6d} "
              f"{m['cagr_pct']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd_pct']:>7.1f}% {m['calmar']:>7.2f} {m['profit_factor']:>6.2f} "
              f"{m['trade_wr_pct']:>7.1f}% {m['n_trades']:>7d} ${m['final_equity']:>11,.0f}")

    print(f"\n  BREAK-EVEN BA: ~{breakeven_ba:.1f}% (Sharpe crosses 0 here)")

    # ═══════════════════════════════════════════════════════════════
    # SHARPE DEGRADATION CURVE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("SHARPE DEGRADATION vs BA COST")
    print("=" * 100)

    sharpe_by_ba = {}
    for sc in scenarios:
        label = sc["label"]
        r = all_results[label]
        if "error" not in r:
            # Use tier1 BA as the x-axis value for flat scenarios
            ba_key = sc["ba"]["tier1"] * 100
            sharpe_by_ba[f"{label} ({ba_key:.0f}%)"] = r["metrics"]["sharpe"]

    for name, sharpe in sorted(sharpe_by_ba.items(), key=lambda x: x[1], reverse=True):
        bar_len = max(0, int(sharpe * 20))
        bar = "#" * bar_len
        print(f"  {name:<45} Sharpe={sharpe:>6.2f}  {bar}")

    # ═══════════════════════════════════════════════════════════════
    # DETAILED REPORTS PER SCENARIO
    # ═══════════════════════════════════════════════════════════════
    for sc in scenarios:
        label = sc["label"]
        r = all_results[label]
        if "error" in r:
            continue
        m = r["metrics"]

        print(f"\n{'='*100}")
        print(f"DETAILED: {label}")
        print(f"  {sc['desc']}")
        print(f"{'='*100}")

        print(f"\n  -- Core Metrics --")
        print(f"    CAGR:           {m['cagr_pct']:>8.2f}%")
        print(f"    Sharpe:         {m['sharpe']:>8.2f}")
        print(f"    Sortino:        {m['sortino']:>8.2f}")
        print(f"    Calmar:         {m['calmar']:>8.2f}")
        print(f"    Max Drawdown:   {m['max_dd_pct']:>8.2f}%")
        print(f"    Profit Factor:  {m['profit_factor']:>8.2f}")
        print(f"    Daily WR:       {m['daily_wr_pct']:>8.1f}%")
        print(f"    Final Equity:   ${m['final_equity']:>11,.0f}")

        print(f"\n  -- Trade Stats --")
        print(f"    Trades:         {m['n_trades']:>7d}")
        print(f"    Trade WR:       {m['trade_wr_pct']:>7.1f}%")
        print(f"    Avg Premium:    ${m['avg_premium_collected']:>8.2f}")
        print(f"    Avg Trade PnL:  ${m['avg_trade_pnl']:>8.2f}")
        print(f"    Breach Rate:    {m['breach_rate_pct']:>7.2f}%")

        print(f"\n  -- Worst Drawdown --")
        dd = m["worst_dd_episode"]
        print(f"    Peak:     {dd['peak_date']} (${dd['peak_equity']:>11,.0f})")
        print(f"    Trough:   {dd['trough_date']} (${dd['trough_equity']:>11,.0f})")
        print(f"    Depth:    {dd['max_dd_pct']:.2f}%  (${dd['loss_dollars']:>11,.0f})")
        print(f"    Recovery: {dd['recovery_days']} days")

        print(f"\n  -- Per-Year --")
        print(f"    {'Year':>6} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8}")
        print(f"    {'-'*40}")
        for yr in sorted(m["per_year"].keys()):
            y = m["per_year"][yr]
            print(f"    {yr:>6} {y['return_pct']:>7.1f}% {y['sharpe']:>8.2f} {y['sortino']:>8.2f} {y['max_dd_pct']:>7.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # TIER-1 ONLY vs FULL 98 COMPARISON
    # ═══════════════════════════════════════════════════════════════
    s5 = all_results.get("S5_tiered_realistic", {})
    s6 = all_results.get("S6_tier1_only_15pct", {})
    if "error" not in s5 and "error" not in s6:
        m5 = s5["metrics"]
        m6 = s6["metrics"]
        print("\n" + "=" * 100)
        print("DIVERSIFICATION ANALYSIS: 70 TIER-1 ONLY vs 98 TIERED")
        print("Does adding 28 tier-2 tickers (at 30% BA) help or hurt?")
        print("=" * 100)
        print(f"  {'Metric':<20} {'T1 Only (70@15%)':<20} {'Tiered (98)':<20} {'Delta':<15}")
        print(f"  {'-'*70}")
        for metric, fmt in [
            ("sharpe", ".2f"), ("sortino", ".2f"), ("cagr_pct", ".1f"),
            ("max_dd_pct", ".1f"), ("calmar", ".2f"), ("profit_factor", ".2f"),
            ("trade_wr_pct", ".1f"), ("n_trades", "d"),
        ]:
            v6 = m6[metric]
            v5 = m5[metric]
            delta = v5 - v6
            if fmt == "d":
                print(f"  {metric:<20} {v6:<20d} {v5:<20d} {delta:>+d}")
            else:
                print(f"  {metric:<20} {format(v6, fmt):<20} {format(v5, fmt):<20} {format(delta, '+' + fmt)}")

        print(f"\n  Verdict: ", end="")
        if m5["sharpe"] > m6["sharpe"] and m5["max_dd_pct"] > m6["max_dd_pct"]:
            print("Tiered 98 has BETTER Sharpe AND better MaxDD -- diversification HELPS")
        elif m5["sharpe"] > m6["sharpe"]:
            print("Tiered 98 has better Sharpe but worse MaxDD -- mixed, lean toward full universe")
        elif m6["sharpe"] > m5["sharpe"]:
            print("Tier-1 only has better Sharpe -- tier-2 at 30% BA HURTS, drop them")
        else:
            print("Similar performance -- tier-2 adds little value at 30% BA")

    # ═══════════════════════════════════════════════════════════════
    # PERMUTATION TEST on realistic scenario (S3 = 15% flat)
    # ═══════════════════════════════════════════════════════════════
    s3 = all_results.get("S3_15pct_flat_realistic", {})
    if "error" not in s3 and not s3["trades_df"].empty:
        print("\n" + "=" * 100)
        print("PERMUTATION TEST on S3 (15% flat BA -- realistic scenario)")
        print("=" * 100)
        perm = run_permutation_test(s3["trades_df"], n_permutations=100)
        print(f"  Actual PnL:       ${perm['actual_total_pnl']:>11,.0f}")
        print(f"  Actual Sharpe:    {perm['actual_sharpe']:>8.2f}")
        print(f"  Random PnL mean:  ${perm['random_pnl_mean']:>11,.0f}")
        print(f"  P-value (PnL):    {perm['p_value_pnl']:>8.4f}")
        print(f"  P-value (Sharpe): {perm['p_value_sharpe']:>8.4f}")
        print(f"  >>> VERDICT: {perm['verdict']}")
    else:
        perm = None

    # ═══════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY")
    print("=" * 100)

    s1 = all_results.get("S1_5pct_flat_original", {})
    if "error" not in s1 and "error" not in s3:
        m1 = s1["metrics"]
        m3 = s3["metrics"]
        print(f"\n  Original (5% BA):  Sharpe={m1['sharpe']:.2f}, CAGR={m1['cagr_pct']:.1f}%, MaxDD={m1['max_dd_pct']:.1f}%")
        print(f"  Realistic (15% BA): Sharpe={m3['sharpe']:.2f}, CAGR={m3['cagr_pct']:.1f}%, MaxDD={m3['max_dd_pct']:.1f}%")
        sharpe_haircut = (1 - m3['sharpe'] / m1['sharpe']) * 100 if m1['sharpe'] != 0 else 0
        print(f"  Sharpe haircut:    {sharpe_haircut:.1f}%")
        print(f"  Break-even BA:     ~{breakeven_ba:.1f}%")

        if m3['sharpe'] >= 1.0:
            print(f"\n  CONCLUSION: Strategy SURVIVES realistic costs. Sharpe {m3['sharpe']:.2f} still above 1.0.")
            print(f"  This is a real edge, not a cost artifact.")
        elif m3['sharpe'] >= 0.5:
            print(f"\n  CONCLUSION: Strategy MARGINAL at realistic costs. Sharpe {m3['sharpe']:.2f} -- viable but thin.")
            print(f"  Requires excellent execution (limit orders, patient fills) to be worthwhile.")
        else:
            print(f"\n  CONCLUSION: Strategy FAILS at realistic costs. Sharpe {m3['sharpe']:.2f} -- not tradeable.")
            print(f"  The prior 5% BA result was a cost artifact.")

    # ═══════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ═══════════════════════════════════════════════════════════════
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    save_data = {
        "generated": pd.Timestamp.now().isoformat(),
        "purpose": "Realistic BA cost sensitivity -- live validation showed 5% BA was 3x too low",
        "starting_capital": STARTING_CAPITAL,
        "live_validation_context": {
            "tier1_median_ba_pct": "16-18%",
            "tier2_median_ba_pct": "24-58%",
            "prior_assumption_pct": "5%",
            "underestimate_factor": "3x",
        },
        "breakeven_ba_pct": round(breakeven_ba, 1),
        "scenarios": {},
    }

    for sc in scenarios:
        label = sc["label"]
        r = all_results[label]
        if "error" in r:
            save_data["scenarios"][label] = {"error": r.get("error", "unknown"), "desc": sc["desc"]}
        else:
            save_data["scenarios"][label] = {
                "desc": sc["desc"],
                "ba_costs": sc["ba"],
                "n_tickers": len(sc["tickers"]),
                "metrics": r["metrics"],
            }

    if perm is not None:
        save_data["permutation_test_15pct"] = convert(perm)

    save_data = convert(save_data)

    with open(OUTPUT / "realistic_ba_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*100}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to {OUTPUT}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
