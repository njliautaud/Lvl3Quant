#!/usr/bin/env python3
"""
Wheel Strategy Regime Analysis
================================
HC #662 R2: Regime-specific failure modes.
HC #428 R1: Regime-agnostic validation.

Tests V5 CSP and BPS across market regimes:
  - Bull (SPY 20d mom > +2%)
  - Bear (SPY 20d mom < -2%)
  - Flat/Sideways
  - High vol (VIX > 25)
  - Low vol (VIX < 15)
  - Crash (VIX > 35)

Goal: identify if any strategy ONLY works in one regime.
HC #428 R1 REJECT if |Sharpe_bull - Sharpe_bear| / max > 0.50
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/jupiter/Lvl3Quant/output/wheel_higher_returns_study")
from higher_returns_study import run_baseline_csp, run_bull_put_spread, load_data

OUTPUT = "/home/jupiter/Lvl3Quant/output/wheel_regime_analysis"
os.makedirs(OUTPUT, exist_ok=True)


def regime_sharpe(rets, min_days=20):
    """Compute annualized Sharpe from a return series."""
    if len(rets) < min_days:
        return None
    std = rets.std()
    if std < 1e-9:
        return 0
    return round(rets.mean() / std * np.sqrt(252), 3)


def regime_stats(rets, label=""):
    """Compute stats for a regime slice."""
    if len(rets) < 10:
        return {"n_days": len(rets), "sharpe": None, "sortino": None,
                "daily_wr_pct": None, "avg_daily_ret_pct": None,
                "worst_day_pct": None, "note": "too few days"}

    std = rets.std()
    sharpe = rets.mean() / max(std, 1e-9) * np.sqrt(252)
    downside = rets[rets < 0].std()
    sortino = rets.mean() / max(downside, 1e-9) * np.sqrt(252) if downside > 0 else 0

    wr = (rets > 0).mean()
    max_daily_loss = rets.min()
    avg_ret = rets.mean()

    return {
        "n_days": len(rets),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "daily_wr_pct": round(wr * 100, 1),
        "avg_daily_ret_pct": round(avg_ret * 100, 4),
        "worst_day_pct": round(max_daily_loss * 100, 3),
    }


def main():
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()

    # ── Run V5 CSP ──
    print("\n=== Running V5 CSP (DTE=10) ===")
    csp_result = run_baseline_csp(
        prices, iv, macro, fund, universe, earnings,
        dte_override=10, label="V5 CSP"
    )
    csp_eq = csp_result["equity_curve"].copy()
    csp_eq["daily_ret"] = csp_eq["equity"].pct_change()

    # ── Run BPS ──
    print("\n=== Running BPS $10 Weekly (best_combo config) ===")
    bps_result = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, max_concurrent=30, per_name_pct=0.025,
        margin_cap=0.30, dte_target=7, profit_take=0.50,
        label="BPS $10 Weekly 30%"
    )
    bps_eq = bps_result["equity_curve"].copy()
    bps_eq["daily_ret"] = bps_eq["equity"].pct_change()

    # ── Build regime classifications from SPY + VIX ──
    spy = prices[prices["ticker"] == "SPY"].sort_values("date").copy()
    spy["mom_20d"] = spy["close"].pct_change(20)
    spy_mom = spy.set_index("date")["mom_20d"]

    vix = macro[["date", "vix"]].dropna().set_index("date")["vix"]

    strategies = {
        "V5_CSP": csp_eq.set_index("date"),
        "BPS_10w": bps_eq.set_index("date"),
    }

    results = {}

    for strat_name, eq_data in strategies.items():
        print(f"\n{'='*60}")
        print(f"REGIME ANALYSIS: {strat_name}")
        print(f"{'='*60}")

        rets = eq_data["daily_ret"].dropna()

        # Overall
        overall = regime_stats(rets, "Overall")
        print(f"\n  Overall: Sharpe={overall['sharpe']}, "
              f"Sortino={overall['sortino']}, WR={overall['daily_wr_pct']}%, "
              f"n={overall['n_days']}d")

        # SPY momentum regimes
        common_dates = rets.index.intersection(spy_mom.dropna().index)
        rets_aligned = rets.loc[common_dates]
        mom_aligned = spy_mom.loc[common_dates]

        bull_mask = mom_aligned > 0.02
        bear_mask = mom_aligned < -0.02
        flat_mask = (mom_aligned >= -0.02) & (mom_aligned <= 0.02)

        bull = regime_stats(rets_aligned[bull_mask], "Bull")
        bear = regime_stats(rets_aligned[bear_mask], "Bear")
        flat = regime_stats(rets_aligned[flat_mask], "Flat")

        print(f"\n  By SPY 20d Momentum:")
        print(f"    Bull  (mom>+2%): Sharpe={bull['sharpe']}, "
              f"WR={bull['daily_wr_pct']}%, n={bull['n_days']}d, "
              f"worst={bull['worst_day_pct']}%")
        print(f"    Bear  (mom<-2%): Sharpe={bear['sharpe']}, "
              f"WR={bear['daily_wr_pct']}%, n={bear['n_days']}d, "
              f"worst={bear['worst_day_pct']}%")
        print(f"    Flat  (|mom|<2%): Sharpe={flat['sharpe']}, "
              f"WR={flat['daily_wr_pct']}%, n={flat['n_days']}d, "
              f"worst={flat['worst_day_pct']}%")

        # HC #428 R1 regime-agnostic check
        if bull['sharpe'] is not None and bear['sharpe'] is not None:
            max_s = max(abs(bull['sharpe']), abs(bear['sharpe']))
            if max_s > 0:
                regime_skew = abs(bull['sharpe'] - bear['sharpe']) / max_s
                verdict = "PASS" if regime_skew <= 0.50 else "FAIL"
                print(f"    HC #428 R1 Regime Test: |{bull['sharpe']}-{bear['sharpe']}|/"
                      f"max = {regime_skew:.2f} → {verdict} (threshold: 0.50)")

        # VIX regimes
        common_vix = rets.index.intersection(vix.index)
        rets_vix = rets.loc[common_vix]
        vix_aligned = vix.loc[common_vix]

        low_vol = regime_stats(rets_vix[vix_aligned < 15], "LowVol")
        mid_vol = regime_stats(rets_vix[(vix_aligned >= 15) & (vix_aligned <= 25)], "MidVol")
        high_vol = regime_stats(rets_vix[(vix_aligned > 25) & (vix_aligned <= 35)], "HighVol")
        crash = regime_stats(rets_vix[vix_aligned > 35], "Crash")

        print(f"\n  By VIX Level:")
        print(f"    Low   (VIX<15):  Sharpe={low_vol['sharpe']}, "
              f"n={low_vol['n_days']}d, worst={low_vol['worst_day_pct']}%")
        print(f"    Mid   (15-25):   Sharpe={mid_vol['sharpe']}, "
              f"n={mid_vol['n_days']}d, worst={mid_vol['worst_day_pct']}%")
        print(f"    High  (25-35):   Sharpe={high_vol['sharpe']}, "
              f"n={high_vol['n_days']}d, worst={high_vol['worst_day_pct']}%")
        print(f"    Crash (VIX>35):  Sharpe={crash['sharpe']}, "
              f"n={crash['n_days']}d, worst={crash.get('worst_day_pct', 'N/A')}%")

        # Year-by-year (rolling 252d Sharpe)
        print(f"\n  By Year:")
        for year in sorted(rets.index.year.unique()):
            yr_rets = rets[rets.index.year == year]
            if len(yr_rets) < 20:
                continue
            yr = regime_stats(yr_rets, str(year))
            print(f"    {year}: Sharpe={yr['sharpe']}, WR={yr['daily_wr_pct']}%, "
                  f"n={yr['n_days']}d, worst={yr['worst_day_pct']}%")

        # Store results
        results[strat_name] = {
            "overall": overall,
            "bull": bull,
            "bear": bear,
            "flat": flat,
            "low_vol": low_vol,
            "mid_vol": mid_vol,
            "high_vol": high_vol,
            "crash": crash,
        }

        # Check if HC #428 R1 passes
        if bull['sharpe'] is not None and bear['sharpe'] is not None:
            results[strat_name]["hc428_regime_skew"] = round(regime_skew, 3)
            results[strat_name]["hc428_verdict"] = verdict

    # ── Cross-strategy comparison ──
    print(f"\n{'='*60}")
    print("CROSS-STRATEGY COMPARISON")
    print(f"{'='*60}")

    for regime in ["bull", "bear", "flat", "low_vol", "high_vol", "crash"]:
        csp_s = results["V5_CSP"].get(regime, {}).get("sharpe")
        bps_s = results["BPS_10w"].get(regime, {}).get("sharpe")
        if csp_s is not None and bps_s is not None:
            better = "CSP" if csp_s > bps_s else "BPS"
            print(f"  {regime:10s}: CSP={csp_s:6.3f}  BPS={bps_s:6.3f}  → {better} wins")

    # ── Save ──
    with open(f"{OUTPUT}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT}/results.json")


if __name__ == "__main__":
    main()
