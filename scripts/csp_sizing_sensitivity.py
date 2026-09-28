#!/usr/bin/env python3
"""
CSP Position Sizing Sensitivity Study
=======================================
HC #661: Exposure / position sizing / leverage = CRITICAL.
HC #662 R2: Stress-test position sizing sensitivity.

Tests V5 (DTE=10, the winner) under varying:
  - margin_cap: {20%, 30%, 40%, 50%, 60%}
  - per_name_pct: {1%, 2%, 3%, 5%}

Goal: find the margin cap that maximizes Sharpe/Calmar without
excessive drawdown. Also identify if we're over/under-leveraged.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jupiter/Lvl3Quant/output/wheel_higher_returns_study")
from higher_returns_study import run_baseline_csp, load_data

OUTPUT = "/home/jupiter/Lvl3Quant/output/csp_sizing_sensitivity"
os.makedirs(OUTPUT, exist_ok=True)


def compute_metrics(eq_df, starting_cash=100_000.0):
    eq = eq_df["equity"]
    rets = eq.pct_change().dropna()
    total_days = len(rets)
    total_years = total_days / 252
    total_ret = (eq.iloc[-1] / starting_cash) - 1
    cagr = (1 + total_ret) ** (1 / max(total_years, 0.01)) - 1

    sharpe = rets.mean() / max(rets.std(), 1e-9) * np.sqrt(252)
    downside = rets[rets < 0].std()
    sortino = rets.mean() / max(downside, 1e-9) * np.sqrt(252)

    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0

    # Worst drawdown duration (days)
    dd = (eq - peak) / peak
    in_dd = dd < 0
    dd_groups = (in_dd != in_dd.shift()).cumsum()
    dd_lengths = in_dd.groupby(dd_groups).sum()
    worst_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

    # Tail risk: worst 5% of daily returns
    worst_5pct = rets.quantile(0.05)
    cvar_5 = rets[rets <= worst_5pct].mean()

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "worst_dd_days": worst_dd_duration,
        "cvar_5_pct": round(cvar_5 * 100, 3),
        "total_ret_pct": round(total_ret * 100, 2),
    }


def main():
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()

    margin_caps = [0.20, 0.30, 0.40, 0.50, 0.60]
    per_name_pcts = [0.01, 0.02, 0.03, 0.05]

    results = {}

    # ── Phase 1: Margin cap sweep (per_name fixed at 3%) ──
    print("\n" + "="*60)
    print("PHASE 1: Margin Cap Sweep (per_name=3%)")
    print("="*60)

    for mc in margin_caps:
        label = f"margin_{int(mc*100)}pct"
        print(f"\n--- Margin Cap = {mc:.0%} ---")
        r = run_baseline_csp(
            prices, iv, macro, fund, universe, earnings,
            dte_override=10,  # V5 DTE
            margin_cap=mc,
            per_name_pct=0.03,
            label=label,
        )
        m = compute_metrics(r["equity_curve"])
        results[label] = m
        print(f"  Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
              f"CAGR={m['cagr_pct']}%, MaxDD={m['max_dd_pct']}%, "
              f"Calmar={m['calmar']}, CVaR5={m['cvar_5_pct']}%")

    # ── Phase 2: Per-name sweep (margin cap fixed at best from Phase 1) ──
    best_mc_label = max(
        [(k, v) for k, v in results.items()],
        key=lambda x: x[1]["calmar"]
    )
    best_mc = float(best_mc_label[0].split("_")[1].replace("pct", "")) / 100
    print(f"\nBest margin cap by Calmar: {best_mc:.0%}")

    print("\n" + "="*60)
    print(f"PHASE 2: Per-Name Sweep (margin_cap={best_mc:.0%})")
    print("="*60)

    for pn in per_name_pcts:
        label = f"perName_{int(pn*100)}pct_mc{int(best_mc*100)}"
        print(f"\n--- Per-Name = {pn:.0%}, Margin Cap = {best_mc:.0%} ---")
        r = run_baseline_csp(
            prices, iv, macro, fund, universe, earnings,
            dte_override=10,
            margin_cap=best_mc,
            per_name_pct=pn,
            label=label,
        )
        m = compute_metrics(r["equity_curve"])
        results[label] = m
        print(f"  Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
              f"CAGR={m['cagr_pct']}%, MaxDD={m['max_dd_pct']}%, "
              f"Calmar={m['calmar']}, CVaR5={m['cvar_5_pct']}%")

    # ── Phase 3: Stress test — worst-case scenarios ──
    print("\n" + "="*60)
    print("PHASE 3: Extreme Configs (stress boundaries)")
    print("="*60)

    stress_configs = [
        ("conservative", 0.20, 0.01),   # Very conservative
        ("moderate", 0.30, 0.02),        # Moderate
        ("current_v5", 0.40, 0.03),      # Current V5 config
        ("aggressive", 0.60, 0.05),      # Aggressive
    ]

    for name, mc, pn in stress_configs:
        label = f"stress_{name}"
        print(f"\n--- {name}: margin={mc:.0%}, per_name={pn:.0%} ---")
        r = run_baseline_csp(
            prices, iv, macro, fund, universe, earnings,
            dte_override=10,
            margin_cap=mc,
            per_name_pct=pn,
            label=label,
        )
        m = compute_metrics(r["equity_curve"])
        results[label] = m
        print(f"  Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
              f"CAGR={m['cagr_pct']}%, MaxDD={m['max_dd_pct']}%, "
              f"Calmar={m['calmar']}, CVaR5={m['cvar_5_pct']}%")

    # ── Summary ──
    print("\n" + "="*60)
    print("SUMMARY: Ranked by Calmar (risk-adjusted)")
    print("="*60)

    ranked = sorted(results.items(), key=lambda x: x[1]["calmar"], reverse=True)
    for i, (name, m) in enumerate(ranked, 1):
        print(f"  {i:2d}. {name:30s} Calmar={m['calmar']:6.3f}  "
              f"Sharpe={m['sharpe']:5.3f}  MaxDD={m['max_dd_pct']:7.2f}%  "
              f"CAGR={m['cagr_pct']:6.2f}%  CVaR5={m['cvar_5_pct']:6.3f}%")

    # ── Optimal recommendation ──
    # Best = highest Calmar with MaxDD > -30% (HC #661 exposure discipline)
    safe = [(n, m) for n, m in ranked if m["max_dd_pct"] > -30]
    if safe:
        best_name, best_m = safe[0]
        print(f"\n  RECOMMENDATION: {best_name}")
        print(f"    Calmar={best_m['calmar']}, Sharpe={best_m['sharpe']}, "
              f"MaxDD={best_m['max_dd_pct']}%, CAGR={best_m['cagr_pct']}%")

        # Compare to current V5 config
        if "stress_current_v5" in results:
            cur = results["stress_current_v5"]
            print(f"\n  vs Current V5: Calmar {cur['calmar']} → {best_m['calmar']} "
                  f"({'+' if best_m['calmar'] > cur['calmar'] else ''}"
                  f"{((best_m['calmar']/cur['calmar'])-1)*100:.0f}%)")

    # ── Save ──
    with open(f"{OUTPUT}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT}/results.json")


if __name__ == "__main__":
    main()
