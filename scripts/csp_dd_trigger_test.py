#!/usr/bin/env python3
"""
CSP Drawdown Trigger Test — Optimized
======================================
Tests whether the fast drawdown trigger (3-day lookback, -5% halt)
that improved BPS Sharpe 4.09→4.26 also helps V4/V5 CSP engines.

Approach: Run baseline ONCE, get equity curve, then analyze whether
trigger-day entries lead to worse outcomes. Permutation test randomizes
trigger day selection.

HC #659: Permutation test mandatory.
HC #662 R4: Research continues.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jupiter/Lvl3Quant/output/wheel_higher_returns_study")
from higher_returns_study import run_baseline_csp, load_data

OUTPUT = "/home/jupiter/Lvl3Quant/output/csp_dd_trigger_test"
os.makedirs(OUTPUT, exist_ok=True)


def compute_metrics(eq_series, starting_cash=100_000.0):
    """Compute risk-adjusted metrics from equity series."""
    rets = eq_series.pct_change().dropna()
    total_days = len(rets)
    total_years = total_days / 252
    total_ret = (eq_series.iloc[-1] / starting_cash) - 1
    cagr = (1 + total_ret) ** (1 / max(total_years, 0.01)) - 1

    std_ret = rets.std()
    sharpe = rets.mean() / max(std_ret, 1e-9) * np.sqrt(252)

    downside = rets[rets < 0].std()
    sortino = rets.mean() / max(downside, 1e-9) * np.sqrt(252)

    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0

    # Win rate on daily returns
    wr = (rets > 0).mean()

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "total_ret_pct": round(total_ret * 100, 2),
        "calmar": round(calmar, 3),
        "daily_wr_pct": round(wr * 100, 1),
    }


def analyze_dd_trigger(eq_df, lookback, threshold, hold_period=14,
                       random_seed=None):
    """
    Analyze impact of a drawdown trigger on CSP equity curve.

    Instead of re-running the full backtest, we measure:
    1. Which days the trigger would fire
    2. Forward returns from those days vs non-trigger days
    3. Estimate if avoiding entries on trigger days improves risk-adjusted returns

    The logic: if forward returns after trigger days are systematically worse,
    then the trigger adds value by avoiding bad entry timing.
    """
    dates = eq_df["date"].values
    equities = eq_df["equity"].values
    n = len(dates)

    # Identify trigger days
    trigger_mask = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        trailing_ret = (equities[i] - equities[i - lookback]) / equities[i - lookback]
        if trailing_ret < threshold:
            trigger_mask[i] = True

    if random_seed is not None:
        # Permutation: keep same NUMBER of trigger days, randomize WHICH ones
        rng = np.random.RandomState(random_seed)
        n_trigger = trigger_mask.sum()
        trigger_mask = np.zeros(n, dtype=bool)
        idx = rng.choice(n, size=n_trigger, replace=False)
        trigger_mask[idx] = True

    n_trigger = trigger_mask.sum()

    # Forward returns analysis
    trigger_fwd = []
    normal_fwd = []

    for i in range(n - hold_period):
        fwd_ret = (equities[i + hold_period] - equities[i]) / equities[i]
        if trigger_mask[i]:
            trigger_fwd.append(fwd_ret)
        else:
            normal_fwd.append(fwd_ret)

    # Simulate the trigger effect on daily returns
    # On trigger days, assume we earn risk-free (~0%) instead of the actual return
    # This approximates halting new entries (existing positions still run,
    # but the incremental P&L from NEW positions opened that day is avoided)
    daily_rets = np.diff(equities) / equities[:-1]

    # Estimate: what fraction of daily P&L comes from positions opened that day?
    # For a 14-DTE CSP portfolio with ~30 positions, each day opens ~2 new ones.
    # So ~2/30 = ~7% of P&L is from same-day entries. On trigger days, we skip those.
    # Conservative estimate: 10% of daily P&L comes from new entries.
    NEW_ENTRY_FRACTION = 0.10

    modified_rets = daily_rets.copy()
    for i in range(len(modified_rets)):
        if i < n - 1 and trigger_mask[i]:
            # Remove the new-entry component of return
            modified_rets[i] *= (1 - NEW_ENTRY_FRACTION)

    # Reconstruct modified equity curve
    modified_eq = np.zeros(n)
    modified_eq[0] = equities[0]
    for i in range(1, n):
        modified_eq[i] = modified_eq[i - 1] * (1 + modified_rets[i - 1])

    base_metrics = compute_metrics(pd.Series(equities))
    modified_metrics = compute_metrics(pd.Series(modified_eq))

    return {
        "n_trigger_days": int(n_trigger),
        "pct_trigger_days": round(n_trigger / n * 100, 1),
        "avg_fwd_ret_trigger": round(np.mean(trigger_fwd) * 100, 3) if trigger_fwd else None,
        "avg_fwd_ret_normal": round(np.mean(normal_fwd) * 100, 3) if normal_fwd else None,
        "avoidance_value_pct": round(
            (np.mean(normal_fwd) - np.mean(trigger_fwd)) * 100, 3
        ) if trigger_fwd and normal_fwd else None,
        "base_sharpe": base_metrics["sharpe"],
        "modified_sharpe": modified_metrics["sharpe"],
        "base_maxdd": base_metrics["max_dd_pct"],
        "modified_maxdd": modified_metrics["max_dd_pct"],
        "sharpe_delta": round(modified_metrics["sharpe"] - base_metrics["sharpe"], 3),
    }


def main():
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()

    # ── A: Run V4 Baseline ONCE ──
    print("\n=== Running V4 Baseline (14 DTE, 30-delta, 65% PT) ===")
    baseline = run_baseline_csp(prices, iv, macro, fund, universe, earnings,
                                label="V4 Baseline")
    eq_df = baseline["equity_curve"]
    base_metrics = compute_metrics(eq_df["equity"])
    print(f"  Sharpe: {base_metrics['sharpe']}, Sortino: {base_metrics['sortino']}")
    print(f"  CAGR: {base_metrics['cagr_pct']}%, MaxDD: {base_metrics['max_dd_pct']}%")
    print(f"  Total return: {base_metrics['total_ret_pct']}% over {len(eq_df)} days")

    # ── B: Test DD trigger configs ──
    configs = [
        (3, -0.05, "3d/-5%"),
        (3, -0.03, "3d/-3%"),
        (5, -0.05, "5d/-5%"),
        (5, -0.03, "5d/-3%"),
        (3, -0.07, "3d/-7%"),
        (5, -0.07, "5d/-7%"),
    ]

    all_results = {"baseline": base_metrics, "configs": {}}

    best_config = None
    best_avoidance = -999

    for lookback, threshold, label in configs:
        print(f"\n=== DD Trigger: {label} ===")
        result = analyze_dd_trigger(eq_df, lookback, threshold)
        all_results["configs"][label] = result

        n_trig = result["n_trigger_days"]
        pct_trig = result["pct_trigger_days"]
        avg_trig = result["avg_fwd_ret_trigger"]
        avg_no = result["avg_fwd_ret_normal"]
        avoidance = result["avoidance_value_pct"]

        print(f"  Trigger fired: {n_trig} days ({pct_trig}%)")
        print(f"  Avg 14d fwd return after trigger: {avg_trig}%")
        print(f"  Avg 14d fwd return normal days:   {avg_no}%")
        print(f"  Avoidance value: {avoidance}% per 14d period")
        print(f"  Sharpe: {result['base_sharpe']} → {result['modified_sharpe']} (Δ{result['sharpe_delta']})")
        print(f"  MaxDD:  {result['base_maxdd']}% → {result['modified_maxdd']}%")

        if avoidance is not None and avoidance > best_avoidance:
            best_avoidance = avoidance
            best_config = (lookback, threshold, label)

    # ── C: Permutation Test on best config ──
    if best_config:
        lb, th, lbl = best_config
        print(f"\n=== Permutation Test: {lbl} (200 random seeds) ===")
        real = analyze_dd_trigger(eq_df, lb, th)
        real_avoidance = real["avoidance_value_pct"]

        random_avoidances = []
        for seed in range(200):
            r = analyze_dd_trigger(eq_df, lb, th, random_seed=seed)
            if r["avoidance_value_pct"] is not None:
                random_avoidances.append(r["avoidance_value_pct"])

        p_value = np.mean([ra >= real_avoidance for ra in random_avoidances])
        print(f"  Real avoidance: {real_avoidance}%")
        print(f"  Random mean:    {np.mean(random_avoidances):.3f}%")
        print(f"  Random std:     {np.std(random_avoidances):.3f}%")
        print(f"  p-value:        {p_value:.3f}")

        if p_value < 0.05:
            print(f"  → PASS: {lbl} trigger timing significantly better than random")
            verdict = "PASS"
        else:
            print(f"  → FAIL: trigger timing NOT significantly better than random")
            verdict = "FAIL"

        all_results["permutation"] = {
            "config": lbl,
            "p_value": round(float(p_value), 4),
            "real_avoidance": real_avoidance,
            "random_mean": round(float(np.mean(random_avoidances)), 3),
            "random_std": round(float(np.std(random_avoidances)), 3),
            "verdict": verdict,
        }

    # ── Save ──
    with open(f"{OUTPUT}/results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT}/results.json")
    print(f"\n{'='*60}")
    print(f"VERDICT: Best config = {best_config[2] if best_config else 'NONE'}")
    if best_config and "permutation" in all_results:
        p = all_results["permutation"]
        print(f"  Permutation: {p['verdict']} (p={p['p_value']})")
        if p["verdict"] == "PASS":
            print(f"  → RECOMMEND adding {best_config[2]} DD trigger to V4/V5 paper engines")
        else:
            print(f"  → DO NOT ADD trigger — avoidance value is not statistically significant")


if __name__ == "__main__":
    main()
