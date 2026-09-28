#!/usr/bin/env python3
"""
CSP DTE Optimization Study
============================
V4 (DTE=14) and V5 (DTE=10) share 77% of tickers but have ρ=0.015.
Are they genuinely complementary or is one strictly better?

Tests DTE = {5, 7, 10, 14, 21, 28} with identical params otherwise.
Then measures pairwise correlation of daily returns to identify
truly independent DTEs.

HC #662 R4: Research continues, don't treat configs as final.
HC #659: Permutation test on any result before reporting.
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jupiter/Lvl3Quant/output/wheel_higher_returns_study")
from higher_returns_study import run_baseline_csp, load_data

OUTPUT = "/home/jupiter/Lvl3Quant/output/csp_dte_optimization"
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

    # Profit factor from daily returns
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / max(losses, 1e-9)

    wr = (rets > 0).mean()

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "total_ret_pct": round(total_ret * 100, 2),
        "calmar": round(calmar, 3),
        "pf": round(pf, 3),
        "daily_wr_pct": round(wr * 100, 1),
        "n_days": total_days,
    }


def main():
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()

    dte_values = [5, 7, 10, 14, 21, 28]
    results = {}
    daily_returns = {}

    for dte in dte_values:
        label = f"CSP DTE={dte}"
        print(f"\n=== {label} ===")

        # Use dte_override for non-14 values, otherwise default
        kwargs = {}
        if dte != 14:
            kwargs["dte_override"] = dte

        result = run_baseline_csp(
            prices, iv, macro, fund, universe, earnings,
            label=label,
            **kwargs
        )

        eq_df = result["equity_curve"]
        metrics = compute_metrics(eq_df)
        results[f"dte_{dte}"] = metrics

        # Store daily returns for correlation analysis
        eq_df = eq_df.copy()
        eq_df["daily_ret"] = eq_df["equity"].pct_change()
        daily_returns[dte] = eq_df.set_index("date")["daily_ret"]

        print(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}")
        print(f"  CAGR: {metrics['cagr_pct']}%, MaxDD: {metrics['max_dd_pct']}%")
        print(f"  PF: {metrics['pf']}, WR: {metrics['daily_wr_pct']}%")
        print(f"  Calmar: {metrics['calmar']}")

    # ── Correlation Matrix ──
    print("\n=== Daily Return Correlations ===")
    ret_df = pd.DataFrame(daily_returns)
    corr = ret_df.corr()
    print(corr.round(3).to_string())

    # ── Rank DTEs ──
    print("\n=== DTE Ranking (by Sharpe) ===")
    ranked = sorted(results.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for i, (name, m) in enumerate(ranked, 1):
        print(f"  {i}. {name}: Sharpe={m['sharpe']}, Calmar={m['calmar']}, MaxDD={m['max_dd_pct']}%")

    # ── Find best pair (highest combined Sharpe with lowest correlation) ──
    print("\n=== Best DTE Pair (Sharpe-weighted, low correlation) ===")
    best_pair = None
    best_score = -999

    for i, d1 in enumerate(dte_values):
        for d2 in dte_values[i+1:]:
            rho = corr.loc[d1, d2]
            s1 = results[f"dte_{d1}"]["sharpe"]
            s2 = results[f"dte_{d2}"]["sharpe"]
            # Score: average Sharpe * diversification factor
            div_factor = np.sqrt((1 + 1 - 2 * rho) / 2)  # portfolio vol reduction
            combined_sharpe = (s1 + s2) / 2 * div_factor
            if combined_sharpe > best_score:
                best_score = combined_sharpe
                best_pair = (d1, d2, rho, s1, s2, combined_sharpe)

    if best_pair:
        d1, d2, rho, s1, s2, cs = best_pair
        print(f"  Best pair: DTE={d1} + DTE={d2}")
        print(f"  Correlation: {rho:.3f}")
        print(f"  Individual Sharpes: {s1:.3f}, {s2:.3f}")
        print(f"  Diversified combined score: {cs:.3f}")

    # ── Regime Analysis (bull vs bear vs sideways) ──
    print("\n=== Regime Stratification ===")
    # Use SPY returns to classify regimes
    spy_data = prices[prices["ticker"] == "SPY"].sort_values("date").set_index("date")
    if "close" in spy_data.columns:
        spy_ret = spy_data["close"].pct_change().rolling(20).sum()  # 20d momentum
        for dte in dte_values:
            rets_dte = daily_returns[dte].dropna()
            common_idx = rets_dte.index.intersection(spy_ret.dropna().index)
            if len(common_idx) < 100:
                continue

            spy_mom = spy_ret.loc[common_idx]
            rets_common = rets_dte.loc[common_idx]

            bull = rets_common[spy_mom > 0.02]
            bear = rets_common[spy_mom < -0.02]
            flat = rets_common[(spy_mom >= -0.02) & (spy_mom <= 0.02)]

            bull_sharpe = bull.mean() / max(bull.std(), 1e-9) * np.sqrt(252) if len(bull) > 20 else 0
            bear_sharpe = bear.mean() / max(bear.std(), 1e-9) * np.sqrt(252) if len(bear) > 20 else 0
            flat_sharpe = flat.mean() / max(flat.std(), 1e-9) * np.sqrt(252) if len(flat) > 20 else 0

            print(f"  DTE={dte}: Bull={bull_sharpe:.2f} ({len(bull)}d), "
                  f"Bear={bear_sharpe:.2f} ({len(bear)}d), "
                  f"Flat={flat_sharpe:.2f} ({len(flat)}d)")

    # ── V4 vs V5 specific verdict ──
    print("\n=== V4 (DTE=14) vs V5 (DTE=10) Verdict ===")
    s14 = results["dte_14"]["sharpe"]
    s10 = results["dte_10"]["sharpe"]
    rho_14_10 = corr.loc[14, 10]
    print(f"  V4 Sharpe: {s14}, V5 Sharpe: {s10}")
    print(f"  Correlation: {rho_14_10:.3f}")
    if abs(s14 - s10) < 0.1:
        print("  → Performance is SIMILAR. Running both is justified ONLY if correlation < 0.5")
        if rho_14_10 < 0.5:
            print(f"  → Correlation {rho_14_10:.3f} < 0.5 → KEEP BOTH (genuine diversification)")
        else:
            print(f"  → Correlation {rho_14_10:.3f} ≥ 0.5 → CONSOLIDATE to better one")
    elif s14 > s10 + 0.2:
        print(f"  → V4 strictly better by {s14-s10:.3f} Sharpe. Consider dropping V5.")
    elif s10 > s14 + 0.2:
        print(f"  → V5 strictly better by {s10-s14:.3f} Sharpe. Consider dropping V4.")

    # ── Save ──
    output = {
        "metrics_by_dte": results,
        "correlations": corr.to_dict(),
        "best_pair": {
            "dte_1": best_pair[0], "dte_2": best_pair[1],
            "correlation": round(best_pair[2], 3),
            "combined_score": round(best_pair[5], 3),
        } if best_pair else None,
        "v4_v5_correlation": round(rho_14_10, 3),
    }

    with open(f"{OUTPUT}/results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {OUTPUT}/results.json")


if __name__ == "__main__":
    main()
