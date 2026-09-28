#!/usr/bin/env python3
"""
Iron Condor Stress Test — Cost Sensitivity Analysis
=====================================================
Tests the two R1-passing configs under progressively worse cost assumptions:
  - Wider bid-ask: 10% (default), 15%, 20%, 25%, 30%
  - Per-contract commission: $0.65/leg (Robinhood) = $2.60 RT per contract (4 legs)
  - Slippage: 0, 1%, 2% additional adverse fill

Purpose: Find the HONEST breakeven cost level where the strategy stops working.
If it only works at 10% BA and dies at 15%, it's fragile. If it survives 25%, it's robust.
"""

import sys, json, time, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

# Import everything from iron_condor_income_v1
sys.path.insert(0, str(Path("/home/jupiter/Lvl3Quant/scripts/income_research")))

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "income_research"

# Re-use the IC script's functions
from iron_condor_income_v1 import (
    load_all_data, build_earnings_lookup,
    compute_metrics, regime_test, trade_stats, permutation_test,
    run_iron_condor
)

STARTING_CAPITAL = 100_000


def main():
    np.random.seed(42)
    t0 = time.time()

    prices, iv, macro, vix_df, spy, earnings, avail = load_all_data()
    earnings_lookup = build_earnings_lookup(earnings)

    print(f"\n{'#'*80}")
    print(f"# IRON CONDOR STRESS TEST — COST SENSITIVITY")
    print(f"{'#'*80}\n")

    # The two R1-passing configs
    base_configs = [
        {
            "name": "IC_7d_VIX",
            "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 7,
            "put_delta": -0.20, "call_delta": 0.20,
            "use_vix_gate": True, "use_spy_trend": False, "use_emergency_close": False,
        },
        {
            "name": "IC_21d_VIX+SPY_1.5x",
            "premium_mult": 1.5, "max_concurrent": 15, "dte_target": 21,
            "put_delta": -0.20, "call_delta": 0.20,
            "use_vix_gate": True, "use_spy_trend": True, "use_emergency_close": False,
        },
    ]

    # Cost stress levels
    ba_levels = [0.10, 0.15, 0.20, 0.25, 0.30]
    commission_per_leg = [0.0, 0.65]  # $0 and $0.65/leg
    slippage_pcts = [0.0, 0.01, 0.02]  # 0%, 1%, 2% adverse fill

    all_results = []

    for bcfg in base_configs:
        name = bcfg["name"]
        print(f"\n{'='*70}")
        print(f"  STRESS TESTING: {name}")
        print(f"{'='*70}")

        for ba in ba_levels:
            for comm in commission_per_leg:
                for slip in slippage_pcts:
                    # Skip redundant combos — only test key combinations
                    if comm > 0 and slip > 0 and ba > 0.15:
                        continue  # Too many combos, focus on key ones

                    label = f"{name}_BA{int(ba*100)}_C{comm:.0f}_S{int(slip*100)}"

                    # Run with adjusted costs
                    # Commission is modeled as additional BA cost per leg
                    # Slippage is modeled as additional BA fraction
                    effective_ba = ba + slip

                    eq, trades, stats = run_iron_condor(
                        prices, iv, macro, vix_df, spy, earnings_lookup, avail,
                        ba_frac=effective_ba,
                        put_delta=bcfg["put_delta"],
                        call_delta=bcfg["call_delta"],
                        long_put_offset=0.05,
                        long_call_offset=0.05,
                        profit_take=0.50,
                        stop_loss_mult=2.0,
                        max_concurrent=bcfg["max_concurrent"],
                        per_name_pct=0.04,
                        earnings_buffer=7,
                        premium_mult=bcfg["premium_mult"],
                        dte_target=bcfg["dte_target"],
                        use_vix_gate=bcfg["use_vix_gate"],
                        use_spy_trend=bcfg["use_spy_trend"],
                        use_emergency_close=bcfg["use_emergency_close"],
                    )

                    # Add per-contract commission to trades
                    if comm > 0 and trades:
                        # 4 legs × 2 (open + close) × $0.65 = $5.20 per contract RT
                        for tr in trades:
                            n_contracts = 1  # approximate
                            comm_cost = 4 * 2 * comm * n_contracts
                            tr["pnl"] -= comm_cost

                        # Recompute equity curve with commission drag
                        # Simple: subtract total commission from final equity
                        total_comm = len(trades) * 4 * 2 * comm
                        if eq:
                            # Distribute commission evenly across days
                            n_days = len(eq)
                            daily_comm = total_comm / n_days
                            cumulative_comm = 0
                            for i in range(len(eq)):
                                cumulative_comm += daily_comm
                                eq[i]["equity"] -= cumulative_comm

                    m = compute_metrics(eq, label)
                    r = regime_test(eq, spy, label)
                    t = trade_stats(trades, label)

                    result = {
                        "label": label, "base": name,
                        "ba": ba, "comm": comm, "slip": slip,
                        "effective_ba": effective_ba,
                    }
                    if "error" not in m:
                        result.update({
                            "sharpe": m["sharpe"], "cagr": m["cagr_pct"],
                            "dd": m["max_dd_pct"], "pf": m["pf"],
                            "r1_gap": r.get("gap", 999),
                            "r1_pass": r.get("pass", False),
                            "green": r.get("green", 0), "red": r.get("red", 0),
                            "trades": t.get("n", 0), "wr": t.get("wr", 0),
                        })
                    else:
                        result["error"] = m.get("error", "unknown")

                    all_results.append(result)

                    if "error" not in result:
                        r1_tag = "✅" if result["r1_pass"] else "❌"
                        print(f"    {r1_tag} BA={ba:.0%} C=${comm:.2f} S={slip:.0%} → "
                              f"Sharpe={result['sharpe']:>5} CAGR={result['cagr']:>5.1f}% "
                              f"DD={result['dd']:>6.1f}% R1gap={result['r1_gap']:.3f}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("STRESS TEST SUMMARY — COST SENSITIVITY")
    print(f"{'='*120}")

    for bcfg in base_configs:
        name = bcfg["name"]
        print(f"\n  {name}:")
        print(f"  {'BA%':>5} {'Comm':>6} {'Slip':>5} {'EffBA':>6} │ {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>7} "
              f"{'PF':>6} {'R1gap':>6} {'R1':>5} │ {'Grn':>5} {'Red':>5}")
        print(f"  {'─'*60}┼{'─'*55}")

        subset = [r for r in all_results if r["base"] == name and "error" not in r]
        for r in sorted(subset, key=lambda x: x["effective_ba"]):
            r1_tag = "PASS" if r["r1_pass"] else "FAIL"
            print(f"  {r['ba']:>4.0%} {r['comm']:>5.2f} {r['slip']:>4.0%} {r['effective_ba']:>5.0%} │ "
                  f"{r['sharpe']:>7} {r['cagr']:>6.1f}% {r['dd']:>6.1f}% "
                  f"{r['pf']:>6.2f} {r['r1_gap']:>6.3f} {r1_tag:>5} │ "
                  f"{r['green']:>5} {r['red']:>5}")

    # Find breakeven BA for each config
    print(f"\n\n{'='*80}")
    print("BREAKEVEN ANALYSIS")
    print(f"{'='*80}")

    for bcfg in base_configs:
        name = bcfg["name"]
        subset = [r for r in all_results if r["base"] == name and "error" not in r
                  and r["comm"] == 0 and r["slip"] == 0]
        subset.sort(key=lambda x: x["ba"])

        print(f"\n  {name}:")
        profitable = [r for r in subset if r["cagr"] > 0]
        r1_passing = [r for r in subset if r["r1_pass"]]
        above_15 = [r for r in subset if r["cagr"] >= 15]

        if profitable:
            max_ba_profitable = max(r["ba"] for r in profitable)
            print(f"    Profitable up to BA={max_ba_profitable:.0%}")
        if r1_passing:
            max_ba_r1 = max(r["ba"] for r in r1_passing)
            print(f"    R1 PASS up to BA={max_ba_r1:.0%}")
        if above_15:
            max_ba_15 = max(r["ba"] for r in above_15)
            print(f"    ≥15% CAGR up to BA={max_ba_15:.0%}")
        else:
            print(f"    ≥15% CAGR: NONE at any BA level")

    # Save
    out = OUTPUT / "iron_condor_stress_test.json"
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (pd.Timestamp, np.bool_)): return str(obj)
        return obj
    with open(out, "w") as f:
        json.dump([{k: clean(v) for k, v in r.items()} for r in all_results], f, indent=2)

    print(f"\nSaved to {out}")
    print(f"Runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
