#!/usr/bin/env python3
"""
BPS Expanded Universe Study — HC #660
======================================
Runs the BPS $10 weekly strategy on the full expanded universe (197+ tickers
including high-beta, mid-cap, biotech, semis, energy, REITs).

Compares to the original 230-ticker baseline. Reports by sector bucket and
beta bucket. Runs a permutation test (HC #659) before reporting.

Output: output/bps_expanded_universe/
"""
import sys
import json
import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import (
    load_data, run_bull_put_spread, compute_metrics
)

OUTPUT = ROOT / "output" / "bps_expanded_universe"
OUTPUT.mkdir(parents=True, exist_ok=True)

def compute_regime_metrics(equity_df, macro):
    """Compute Sharpe in bull/bear/correction regimes."""
    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    eq["ret"] = eq["equity"].pct_change()

    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    # SPY SMA50 for regime classification
    spy_sma = macro[["date"]].copy()
    # Use VIX as proxy: VIX > 25 = bear, VIX 18-25 = correction, VIX < 18 = bull
    if "vix" in macro.columns:
        vix_by_date = macro.set_index("date")["vix"].to_dict()
    else:
        vix_by_date = {}

    results = {}
    for regime, vix_range in [("bull", (0, 18)), ("correction", (18, 25)), ("bear", (25, 200))]:
        regime_dates = {d for d, v in vix_by_date.items()
                       if not np.isnan(v) and vix_range[0] <= v < vix_range[1]}
        regime_rets = eq[eq["date"].isin(regime_dates)]["ret"].dropna()
        if len(regime_rets) > 10:
            sharpe = regime_rets.mean() / max(regime_rets.std(), 1e-10) * np.sqrt(252)
        else:
            sharpe = float("nan")
        results[f"{regime}_sharpe"] = round(sharpe, 3)
        results[f"{regime}_days"] = len(regime_rets)

    # HC #428 regime gap
    bull_s = results.get("bull_sharpe", 0)
    bear_s = results.get("bear_sharpe", 0)
    denom = max(abs(bull_s), abs(bear_s), 0.01)
    results["regime_gap"] = round(abs(bull_s - bear_s) / denom, 3)
    results["hc428_pass"] = results["regime_gap"] <= 0.50

    return results


def run_permutation_test(prices, iv, macro, fund, universe, earnings, n_perms=20):
    """HC #659: Run with randomized entry signals to verify edge isn't artifact."""
    print(f"\n=== Permutation Test ({n_perms} shuffles) ===")

    # Real result
    real = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, dte_target=7, max_concurrent=60,
        margin_cap=0.30, per_name_pct=0.05,
        label="BPS $10 Weekly REAL"
    )
    real_sharpe = real["metrics"].get("sharpe", 0)
    real_cagr = real["metrics"].get("cagr_pct", 0)
    print(f"  Real: Sharpe={real_sharpe}, CAGR={real_cagr}%")

    # Permutation: randomize the IV rank ordering (which determines stock selection)
    perm_sharpes = []
    perm_cagrs = []
    for i in range(n_perms):
        # Shuffle IV rank within each date to randomize stock selection
        iv_shuffled = iv.copy()
        for date_val in iv_shuffled["date"].unique():
            mask = iv_shuffled["date"] == date_val
            iv_rk_vals = iv_shuffled.loc[mask, "iv_rank"].values.copy()
            np.random.shuffle(iv_rk_vals)
            iv_shuffled.loc[mask, "iv_rank"] = iv_rk_vals

        perm_result = run_bull_put_spread(
            prices, iv_shuffled, macro, fund, universe, earnings,
            spread_width=10.0, dte_target=7, max_concurrent=60,
            margin_cap=0.30, per_name_pct=0.05,
            label=f"BPS Perm #{i+1}"
        )
        ps = perm_result["metrics"].get("sharpe", 0)
        pc = perm_result["metrics"].get("cagr_pct", 0)
        perm_sharpes.append(ps)
        perm_cagrs.append(pc)
        if (i+1) % 5 == 0:
            print(f"  Perm {i+1}/{n_perms}: Sharpe={ps:.2f}")

    # P-value: fraction of permutations with Sharpe >= real
    p_value = sum(1 for s in perm_sharpes if s >= real_sharpe) / n_perms
    artifact_pct = 100 * np.mean(perm_sharpes) / max(abs(real_sharpe), 0.01)

    perm_results = {
        "real_sharpe": real_sharpe,
        "real_cagr": real_cagr,
        "perm_sharpe_mean": round(float(np.mean(perm_sharpes)), 3),
        "perm_sharpe_std": round(float(np.std(perm_sharpes)), 3),
        "perm_cagr_mean": round(float(np.mean(perm_cagrs)), 1),
        "p_value": round(p_value, 4),
        "artifact_pct": round(artifact_pct, 1),
        "n_perms": n_perms,
        "verdict": "PASS" if p_value < 0.05 else "FAIL (artifact)"
    }
    print(f"\n  Permutation test: p={p_value:.3f}, artifact={artifact_pct:.1f}%")
    print(f"  VERDICT: {perm_results['verdict']}")

    return perm_results, real


def run_sector_breakdown(ledger_df, fund):
    """Break down P&L by sector."""
    if ledger_df.empty:
        return {}
    sector_of = dict(zip(fund["ticker"], fund.get("sector", pd.Series(["Unknown"]*len(fund)))))
    ledger_df = ledger_df.copy()
    ledger_df["sector"] = ledger_df["ticker"].map(sector_of).fillna("Unknown")

    results = {}
    for sector, grp in ledger_df.groupby("sector"):
        total_pnl = grp["pnl"].sum()
        n_trades = len(grp)
        wr = (grp["pnl"] > 0).mean()
        avg_pnl = grp["pnl"].mean()
        results[sector] = {
            "n_trades": n_trades,
            "total_pnl": round(total_pnl, 2),
            "win_rate": round(wr, 3),
            "avg_pnl": round(avg_pnl, 2),
        }
    return dict(sorted(results.items(), key=lambda x: x[1]["total_pnl"], reverse=True))


def main():
    t0 = time.time()
    print("=" * 60)
    print("BPS EXPANDED UNIVERSE STUDY (HC #660)")
    print("=" * 60)

    prices, iv, macro, fund, universe, earnings = load_data()
    n_tickers = prices["ticker"].nunique()
    print(f"\nTotal universe: {n_tickers} tickers")
    print(f"Date range: {prices['date'].min().date()} to {prices['date'].max().date()}")

    # ── CONFIG 1: BPS $10 Weekly, 30% margin (current champion) ──
    print("\n=== Config 1: BPS $10 Weekly, 30% margin cap ===")
    result_30 = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, dte_target=7, max_concurrent=60,
        margin_cap=0.30, per_name_pct=0.05,
        label="BPS $10 Weekly 30% margin"
    )
    m30 = result_30["metrics"]
    print(f"  CAGR: {m30.get('cagr_pct')}%  Sharpe: {m30.get('sharpe')}  "
          f"Sortino: {m30.get('sortino')}  MaxDD: {m30.get('max_dd_pct')}%  "
          f"PF: {m30.get('profit_factor')}  WR: {m30.get('daily_wr')}")
    result_30["equity_curve"].to_parquet(OUTPUT / "eq_bps10_weekly_30pct.parquet")

    # Regime analysis
    regime_30 = compute_regime_metrics(result_30["equity_curve"], macro)
    print(f"  Regime: Bull={regime_30['bull_sharpe']} | Correction={regime_30['correction_sharpe']} "
          f"| Bear={regime_30['bear_sharpe']} | Gap={regime_30['regime_gap']}")

    # Sector breakdown
    sector_30 = run_sector_breakdown(result_30.get("ledger", pd.DataFrame()), fund)

    # ── CONFIG 2: BPS $10 Weekly, 20% margin (conservative) ──
    print("\n=== Config 2: BPS $10 Weekly, 20% margin cap (conservative) ===")
    result_20 = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, dte_target=7, max_concurrent=60,
        margin_cap=0.20, per_name_pct=0.04,
        label="BPS $10 Weekly 20% margin"
    )
    m20 = result_20["metrics"]
    print(f"  CAGR: {m20.get('cagr_pct')}%  Sharpe: {m20.get('sharpe')}  "
          f"Sortino: {m20.get('sortino')}  MaxDD: {m20.get('max_dd_pct')}%")
    result_20["equity_curve"].to_parquet(OUTPUT / "eq_bps10_weekly_20pct.parquet")
    regime_20 = compute_regime_metrics(result_20["equity_curve"], macro)

    # ── CONFIG 3: BPS $10 Weekly, 40% margin (aggressive) ──
    print("\n=== Config 3: BPS $10 Weekly, 40% margin cap (aggressive) ===")
    result_40 = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, dte_target=7, max_concurrent=80,
        margin_cap=0.40, per_name_pct=0.05,
        label="BPS $10 Weekly 40% margin"
    )
    m40 = result_40["metrics"]
    print(f"  CAGR: {m40.get('cagr_pct')}%  Sharpe: {m40.get('sharpe')}  "
          f"Sortino: {m40.get('sortino')}  MaxDD: {m40.get('max_dd_pct')}%")
    result_40["equity_curve"].to_parquet(OUTPUT / "eq_bps10_weekly_40pct.parquet")
    regime_40 = compute_regime_metrics(result_40["equity_curve"], macro)

    # ── CONFIG 4: BPS $5 Weekly (tighter spreads, more positions) ──
    print("\n=== Config 4: BPS $5 Weekly, 30% margin cap ===")
    result_5w = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=5.0, dte_target=7, max_concurrent=80,
        margin_cap=0.30, per_name_pct=0.04,
        label="BPS $5 Weekly 30% margin"
    )
    m5w = result_5w["metrics"]
    print(f"  CAGR: {m5w.get('cagr_pct')}%  Sharpe: {m5w.get('sharpe')}  "
          f"Sortino: {m5w.get('sortino')}  MaxDD: {m5w.get('max_dd_pct')}%")
    result_5w["equity_curve"].to_parquet(OUTPUT / "eq_bps5_weekly_30pct.parquet")
    regime_5w = compute_regime_metrics(result_5w["equity_curve"], macro)

    # ── PERMUTATION TEST on best config (HC #659) ──
    perm_results, _ = run_permutation_test(
        prices, iv, macro, fund, universe, earnings, n_perms=20
    )

    # ── SAVE RESULTS ──
    summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "universe_size": n_tickers,
        "date_range": f"{prices['date'].min().date()} to {prices['date'].max().date()}",
        "configs": {
            "bps10_weekly_30pct": {
                "metrics": m30,
                "regime": regime_30,
                "sector_pnl": sector_30,
            },
            "bps10_weekly_20pct": {
                "metrics": m20,
                "regime": regime_20,
            },
            "bps10_weekly_40pct": {
                "metrics": m40,
                "regime": regime_40,
            },
            "bps5_weekly_30pct": {
                "metrics": m5w,
                "regime": regime_5w,
            },
        },
        "permutation_test": perm_results,
        "comparison_to_original_230": {
            "note": "Original BPS $10 Weekly on 230 tickers: CAGR 154%, Sharpe 4.09, MaxDD -30.4%",
            "original_sharpe": 4.09,
            "original_cagr": 154.0,
        },
    }

    with open(OUTPUT / "expanded_universe_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"Results saved to {OUTPUT}")
    print(f"{'='*60}")

    # Print comparison table
    print("\n=== COMPARISON TABLE ===")
    print(f"{'Config':<30} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'PF':>8}")
    print("-" * 80)
    for name, m in [("BPS10 30% (expanded)", m30),
                     ("BPS10 20% (expanded)", m20),
                     ("BPS10 40% (expanded)", m40),
                     ("BPS5 30% (expanded)", m5w)]:
        print(f"{name:<30} {m.get('cagr_pct','?'):>7}% {m.get('sharpe','?'):>8} "
              f"{m.get('sortino','?'):>8} {m.get('max_dd_pct','?'):>7}% {m.get('profit_factor','?'):>8}")
    print(f"{'BPS10 30% (original 230)':<30} {'154.0':>7}% {'4.09':>8} {'4.65':>8} {'-30.4':>7}% {'?':>8}")

    print(f"\nPermutation test: {perm_results['verdict']} (p={perm_results['p_value']})")


if __name__ == "__main__":
    main()
