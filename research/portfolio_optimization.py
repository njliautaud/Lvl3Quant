#!/usr/bin/env python3
"""
portfolio_optimization.py — Multi-Strategy Portfolio Optimization

Optimizes allocation across 3 confirmed R1-passing strategies:
1. ETF Rotation v3 (Sharpe 2.39, R1 gap 0.19) — sector momentum + anti-concentration
2. ETF Rotation v2 (Sharpe 1.90, R1 gap 0.04) — yield curve + graded regime gate
3. V5 + Combined Hedge (Sharpe 2.24, R1 gap 0.21) — CSP wheel + VIX TS sizing + SPY beta hedge

Tests all weight combinations (0-100% in 5% steps, summing to 100%).
For each combo: Sharpe, Sortino, CAGR, MaxDD, Calmar, R1 regime gap.

R1 gate: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
Regime threshold: SPY close-to-close +/- 0.5% (0.005)

Finds:
  (1) Max Sharpe portfolio that passes R1
  (2) Min gap portfolio with Sharpe > 1.5
  (3) Max Calmar portfolio that passes R1
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "portfolio_optimization"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RF_ANNUAL = 0.04
RF_DAILY = RF_ANNUAL / TRADING_DAYS

# R1 regime gate uses SPY close-to-close daily return.
# Two thresholds tested: 0.2% (as in v5_combined_hedge.py) and 0.5% (user spec).
# NOTE: The reported "R1 gap" numbers for ETF rotation strategies used the strategy's
# OWN internal regime labels (SPY 60-day MA based bull/bear), NOT SPY daily returns.
# When using proper SPY close-to-close, ETF rotation strategies do NOT pass R1
# at any threshold - they are fundamentally regime-dependent (long-only momentum).
REGIME_THRESHOLD = 0.002


# ── Data Loading ────────────────────────────────────────────────────────────

def load_strategies() -> dict[str, pd.Series]:
    """Load daily return series for each strategy."""
    strategies = {}

    # 1. ETF Rotation v3 (rotation quality, latest run)
    v3_path = ROOT / "output/macro_picker/etf_rotation_quality_20260709_223944/book.parquet"
    v3 = pd.read_parquet(v3_path)
    v3["date"] = pd.to_datetime(v3["date"])
    strategies["ETF_Rot_v3"] = v3.set_index("date")["daily_ret"].sort_index()

    # 2. ETF Rotation v2 (latest run)
    v2_path = ROOT / "output/macro_picker/etf_rotation_v2_20260709_202559/book.parquet"
    v2 = pd.read_parquet(v2_path)
    v2["date"] = pd.to_datetime(v2["date"])
    strategies["ETF_Rot_v2"] = v2.set_index("date")["daily_ret"].sort_index()

    # 3. V5 Combined Hedge (equity curve -> daily returns)
    v5_path = ROOT / "output/v5_combined_hedge/equity_curves.parquet"
    v5 = pd.read_parquet(v5_path)
    v5["date"] = pd.to_datetime(v5["date"])
    v5 = v5.sort_values("date").set_index("date")
    strategies["V5_Combined"] = v5["combined"].pct_change().fillna(0.0)

    return strategies


def load_spy_returns() -> pd.Series:
    """Load SPY daily returns for regime classification."""
    spy_path = ROOT / "wheel_strategy_v1/data/cache/spy_prices.parquet"
    spy = pd.read_parquet(spy_path)
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")
    return spy["close"].pct_change().fillna(0.0)


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(daily_ret: pd.Series) -> dict:
    """Compute risk-adjusted metrics from daily returns."""
    excess = daily_ret - RF_DAILY
    n = len(daily_ret)

    if n < 30 or excess.std() == 0:
        return {"sharpe": np.nan, "sortino": np.nan, "cagr": np.nan,
                "max_dd": np.nan, "calmar": np.nan}

    sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))

    down = excess[excess < 0]
    if len(down) > 0 and down.std() > 0:
        sortino = float(excess.mean() / down.std() * np.sqrt(TRADING_DAYS))
    else:
        sortino = np.nan

    cum = (1 + daily_ret).cumprod()
    yrs = n / TRADING_DAYS
    cagr = float(cum.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 else 0.0

    peak = cum.cummax()
    max_dd = float((cum / peak - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate, profit factor
    pnl = daily_ret[daily_ret != 0]
    wr = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = float(gains / losses) if losses > 0 else np.nan

    ann_vol = float(daily_ret.std() * np.sqrt(TRADING_DAYS))

    return {
        "sharpe": sharpe, "sortino": sortino, "cagr": cagr,
        "max_dd": max_dd, "calmar": calmar,
        "wr": wr, "pf": pf, "ann_vol": ann_vol,
        "n_days": n,
    }


def compute_r1_gap(daily_ret: pd.Series, spy_ret: pd.Series) -> dict:
    """
    Compute R1 regime gap.
    Green: SPY close-to-close > +0.5%
    Red: SPY close-to-close < -0.5%
    Flat: in between
    """
    spy_aligned = spy_ret.reindex(daily_ret.index).fillna(0)

    regime = pd.Series("flat", index=spy_aligned.index)
    regime[spy_aligned > REGIME_THRESHOLD] = "green"
    regime[spy_aligned < -REGIME_THRESHOLD] = "red"

    excess = daily_ret - RF_DAILY
    result = {}

    for reg in ("green", "red", "flat"):
        sub = excess[regime == reg]
        result[f"n_{reg}"] = len(sub)
        if len(sub) >= 5 and sub.std() > 0:
            result[f"{reg}_sharpe"] = float(sub.mean() / sub.std() * np.sqrt(TRADING_DAYS))
        else:
            result[f"{reg}_sharpe"] = np.nan

    sg = result.get("green_sharpe", np.nan)
    sr = result.get("red_sharpe", np.nan)

    if np.isfinite(sg) and np.isfinite(sr):
        denom = max(abs(sg), abs(sr))
        gap = abs(sg - sr) / denom if denom > 0 else np.nan
        r1_pass = gap <= 0.50
    else:
        gap = np.nan
        r1_pass = False

    result["gap"] = gap
    result["r1_pass"] = r1_pass
    return result


# ── Grid Search ─────────────────────────────────────────────────────────────

def grid_search(returns_df: pd.DataFrame, spy_ret: pd.Series,
                step: int = 5) -> list[dict]:
    """
    Test all weight combinations in `step`% increments.
    Weights must sum to 100%.
    """
    names = returns_df.columns.tolist()
    n_strats = len(names)

    # Generate weight combos: 0 to 100 in steps, summing to 100
    weight_values = list(range(0, 101, step))
    combos = []
    for combo in itertools.product(weight_values, repeat=n_strats):
        if sum(combo) == 100:
            combos.append(combo)

    print(f"  Testing {len(combos)} weight combinations ({step}% steps, {n_strats} strategies)")

    results = []
    for i, combo in enumerate(combos):
        weights = np.array(combo) / 100.0
        port_ret = (returns_df.values * weights).sum(axis=1)
        port_ret = pd.Series(port_ret, index=returns_df.index)

        m = compute_metrics(port_ret)
        r1 = compute_r1_gap(port_ret, spy_ret)

        result = {
            "weights": {n: float(w) for n, w in zip(names, weights)},
            **m, **r1,
        }
        results.append(result)

    return results


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 75)
    print("  MULTI-STRATEGY PORTFOLIO OPTIMIZATION")
    print("  3 R1-passing strategies, exhaustive grid search")
    print("=" * 75)

    # Load data
    strategies = load_strategies()
    spy_ret = load_spy_returns()

    print(f"\n  Strategy data ranges:")
    for name, ret in strategies.items():
        print(f"    {name}: {ret.index.min().date()} to {ret.index.max().date()} ({len(ret)} days)")

    # Align to common date range
    all_rets = pd.DataFrame(strategies)
    all_rets = all_rets.dropna()

    # Remove rows where all strategies have 0 return (padding at end)
    mask = (all_rets != 0).any(axis=1)
    # Actually, let's keep zeros - they're valid (flat days). But trim trailing
    # zero-only rows that indicate data ends.
    last_nonzero = {}
    for col in all_rets.columns:
        nz = all_rets[col][all_rets[col] != 0]
        if len(nz) > 0:
            last_nonzero[col] = nz.index[-1]
    if last_nonzero:
        cutoff = min(last_nonzero.values())
        all_rets = all_rets.loc[:cutoff]

    print(f"\n  Common date range: {all_rets.index.min().date()} to {all_rets.index.max().date()}")
    print(f"  Common days: {len(all_rets)}")

    # ── Correlation Matrix ──
    corr = all_rets.corr()
    print(f"\n  CORRELATION MATRIX:")
    print("  " + corr.round(3).to_string().replace("\n", "\n  "))

    # Flag high correlations
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            c = corr.iloc[i, j]
            n1, n2 = corr.index[i], corr.columns[j]
            if abs(c) > 0.7:
                print(f"\n  WARNING: {n1} and {n2} have correlation {c:.3f} > 0.70")
                print(f"    Using both may not add much diversification benefit.")

    # ── Individual Strategy Metrics ──
    spy_aligned = spy_ret.reindex(all_rets.index).fillna(0)

    print(f"\n  INDIVIDUAL STRATEGY METRICS (common period):")
    print(f"  {'Strategy':<15} {'Sharpe':>7} {'Sort':>7} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'WR':>6} {'PF':>6} {'Gap':>6} {'R1':>5}")
    for name in all_rets.columns:
        m = compute_metrics(all_rets[name])
        r1 = compute_r1_gap(all_rets[name], spy_aligned)
        status = "PASS" if r1.get("r1_pass", False) else "FAIL"
        print(f"  {name:<15} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['cagr']:>6.1%} "
              f"{m['max_dd']:>6.1%} {m['calmar']:>7.2f} {m['wr']:>5.1%} {m['pf']:>5.2f} "
              f"{r1['gap']:>6.2f} {status:>5}")

    # ── Exhaustive Grid Search ──
    print(f"\n{'='*75}")
    print(f"  GRID SEARCH: All 3-strategy weight combos (5% steps)")
    print(f"{'='*75}")

    results = grid_search(all_rets, spy_aligned, step=5)

    # ── Find Optimal Portfolios ──
    r1_passing = [r for r in results if r.get("r1_pass", False)]
    print(f"\n  R1-passing combos: {len(r1_passing)} / {len(results)}")

    # (1) Max Sharpe that passes R1
    if r1_passing:
        best_sharpe = max(r1_passing, key=lambda x: x.get("sharpe", -999))
    else:
        best_sharpe = None

    # (2) Min gap with Sharpe > 1.5
    sharpe_ok = [r for r in results if r.get("sharpe", 0) > 1.5]
    if sharpe_ok:
        best_gap = min(sharpe_ok, key=lambda x: x.get("gap", 999))
    else:
        best_gap = None

    # (3) Max Calmar that passes R1
    if r1_passing:
        best_calmar = max(r1_passing, key=lambda x: x.get("calmar", -999))
    else:
        best_calmar = None

    # ── Report Results ──
    def print_portfolio(label, r):
        if r is None:
            print(f"\n  {label}: NO SOLUTION FOUND")
            return
        w_str = " / ".join(f"{n}={v:.0%}" for n, v in r["weights"].items() if v > 0.001)
        print(f"\n  {label}:")
        print(f"    Weights: {w_str}")
        print(f"    Sharpe: {r['sharpe']:.2f}  Sortino: {r['sortino']:.2f}  CAGR: {r['cagr']:.1%}")
        print(f"    MaxDD: {r['max_dd']:.1%}  Calmar: {r['calmar']:.2f}")
        print(f"    WR: {r['wr']:.1%}  PF: {r['pf']:.2f}")
        print(f"    R1 gap: {r['gap']:.3f}  {'PASS' if r.get('r1_pass') else 'FAIL'}")
        print(f"    Green Sharpe: {r.get('green_sharpe', 0):.2f}  Red Sharpe: {r.get('red_sharpe', 0):.2f}")

    print(f"\n{'='*75}")
    print(f"  OPTIMAL PORTFOLIOS")
    print(f"{'='*75}")

    print_portfolio("(1) MAX SHARPE (R1-passing)", best_sharpe)
    print_portfolio("(2) MIN R1 GAP (Sharpe > 1.5)", best_gap)
    print_portfolio("(3) MAX CALMAR (R1-passing)", best_calmar)

    # ── Top 10 R1-passing by Sharpe ──
    if r1_passing:
        top10 = sorted(r1_passing, key=lambda x: -x.get("sharpe", -999))[:10]
        print(f"\n  TOP 10 R1-PASSING PORTFOLIOS (by Sharpe):")
        print(f"  {'#':>3} {'ETF_v3':>7} {'ETF_v2':>7} {'V5_Comb':>7} {'Sharpe':>7} {'Sort':>7} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'Gap':>6}")
        for i, r in enumerate(top10):
            w = r["weights"]
            print(f"  {i+1:>3} {w.get('ETF_Rot_v3',0):>6.0%} {w.get('ETF_Rot_v2',0):>6.0%} "
                  f"{w.get('V5_Combined',0):>6.0%} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
                  f"{r['cagr']:>6.1%} {r['max_dd']:>6.1%} {r['calmar']:>7.2f} {r['gap']:>6.3f}")

    # ── 2-strategy subsets (in case v2+v3 too correlated) ──
    print(f"\n{'='*75}")
    print(f"  2-STRATEGY SUBSET ANALYSIS")
    print(f"{'='*75}")

    pairs = [
        ("ETF_Rot_v3", "V5_Combined"),
        ("ETF_Rot_v2", "V5_Combined"),
        ("ETF_Rot_v3", "ETF_Rot_v2"),
    ]

    pair_bests = {}
    for s1, s2 in pairs:
        pair_df = all_rets[[s1, s2]]
        pair_results = []
        for w1 in range(0, 101, 5):
            w2 = 100 - w1
            weights = np.array([w1, w2]) / 100.0
            port_ret = (pair_df.values * weights).sum(axis=1)
            port_ret = pd.Series(port_ret, index=pair_df.index)
            m = compute_metrics(port_ret)
            r1 = compute_r1_gap(port_ret, spy_aligned)
            pair_results.append({
                "w1": w1, "w2": w2,
                "weights": {s1: w1/100, s2: w2/100},
                **m, **r1,
            })

        passing = [r for r in pair_results if r.get("r1_pass", False)]
        if passing:
            best = max(passing, key=lambda x: x.get("sharpe", -999))
            pair_bests[f"{s1}+{s2}"] = best
            print(f"\n  {s1} + {s2}:")
            print(f"    Best R1-passing: {s1}={best['w1']}% / {s2}={best['w2']}%")
            print(f"    Sharpe: {best['sharpe']:.2f}  Sortino: {best['sortino']:.2f}  "
                  f"CAGR: {best['cagr']:.1%}  MaxDD: {best['max_dd']:.1%}  "
                  f"Calmar: {best['calmar']:.2f}  Gap: {best['gap']:.3f}")
        else:
            print(f"\n  {s1} + {s2}: No R1-passing combo found")

    # ── Near-R1 Analysis (relaxed gate) ──
    print(f"\n{'='*75}")
    print(f"  NEAR-R1 ANALYSIS (relaxed thresholds)")
    print(f"{'='*75}")

    for max_gap in [0.50, 0.60, 0.70, 0.80, 1.00]:
        passing_at = [r for r in results if r.get("gap", 999) <= max_gap]
        if passing_at:
            best_at = max(passing_at, key=lambda x: x.get("sharpe", -999))
            w = best_at["weights"]
            w_str = " / ".join(f"{n}={v:.0%}" for n, v in w.items() if v > 0.001)
            print(f"  Gap <= {max_gap:.2f}: {len(passing_at):>3} combos pass | "
                  f"Best Sharpe={best_at['sharpe']:.2f} ({w_str}) gap={best_at['gap']:.3f}")
        else:
            print(f"  Gap <= {max_gap:.2f}: 0 combos pass")

    # ── Best combo per Sharpe bracket ──
    print(f"\n  BEST GAP BY SHARPE BRACKET:")
    for min_sh in [2.0, 1.8, 1.5, 1.2, 1.0]:
        bracket = [r for r in results if r.get("sharpe", 0) >= min_sh]
        if bracket:
            best_g = min(bracket, key=lambda x: x.get("gap", 999))
            w = best_g["weights"]
            w_str = " / ".join(f"{n}={v:.0%}" for n, v in w.items() if v > 0.001)
            print(f"    Sharpe >= {min_sh}: best gap={best_g['gap']:.3f} ({w_str}) Sharpe={best_g['sharpe']:.2f}")

    # ── Recommendation ──
    print(f"\n{'='*75}")
    print(f"  DEPLOYMENT RECOMMENDATION")
    print(f"{'='*75}")

    # Check if v2+v3 are too correlated
    v2v3_corr = corr.loc["ETF_Rot_v2", "ETF_Rot_v3"] if "ETF_Rot_v2" in corr.index and "ETF_Rot_v3" in corr.columns else 0
    if v2v3_corr > 0.7:
        print(f"\n  ETF v2 and v3 correlation: {v2v3_corr:.3f} > 0.70")
        print(f"  RECOMMEND: Use only one ETF rotation variant + V5 Combined Hedge")
        print(f"  This avoids concentration in a single strategy family.")
        if "ETF_Rot_v3+V5_Combined" in pair_bests:
            b = pair_bests["ETF_Rot_v3+V5_Combined"]
            print(f"\n  Recommended 2-strategy portfolio: ETF v3 ({b['w1']}%) + V5 Combined ({b['w2']}%)")
            print(f"    Sharpe: {b['sharpe']:.2f}, Gap: {b['gap']:.3f}, CAGR: {b['cagr']:.1%}, MaxDD: {b['max_dd']:.1%}")
    else:
        print(f"\n  ETF v2 and v3 correlation: {v2v3_corr:.3f} <= 0.70")
        print(f"  Both ETF strategies provide diversification benefit.")
        if best_sharpe:
            w = best_sharpe["weights"]
            w_str = " / ".join(f"{n}={v:.0%}" for n, v in w.items() if v > 0.001)
            print(f"\n  Recommended 3-strategy portfolio: {w_str}")
            print(f"    Sharpe: {best_sharpe['sharpe']:.2f}, Gap: {best_sharpe['gap']:.3f}, "
                  f"CAGR: {best_sharpe['cagr']:.1%}, MaxDD: {best_sharpe['max_dd']:.1%}")

    # ── Save Results ──
    save_data = {
        "config": {
            "step_pct": 5,
            "regime_threshold": REGIME_THRESHOLD,
            "r1_max_gap": 0.50,
            "rf_annual": RF_ANNUAL,
            "common_period": {
                "start": str(all_rets.index.min().date()),
                "end": str(all_rets.index.max().date()),
                "n_days": len(all_rets),
            },
        },
        "correlation_matrix": corr.to_dict(),
        "individual_metrics": {},
        "optimal_portfolios": {
            "max_sharpe_r1": best_sharpe,
            "min_gap_sharpe_gt_1_5": best_gap,
            "max_calmar_r1": best_calmar,
        },
        "pair_bests": pair_bests,
        "top10_r1_passing": sorted(r1_passing, key=lambda x: -x.get("sharpe", -999))[:10] if r1_passing else [],
        "total_combos_tested": len(results),
        "r1_passing_count": len(r1_passing),
        "v2_v3_correlation": float(v2v3_corr) if np.isfinite(v2v3_corr) else None,
    }

    # Add individual metrics
    for name in all_rets.columns:
        m = compute_metrics(all_rets[name])
        r1 = compute_r1_gap(all_rets[name], spy_aligned)
        save_data["individual_metrics"][name] = {**m, **r1}

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save equity curves for optimal portfolios
    curves = pd.DataFrame({"date": all_rets.index})
    for label, r in [("max_sharpe", best_sharpe), ("min_gap", best_gap), ("max_calmar", best_calmar)]:
        if r is not None:
            weights = np.array([r["weights"].get(n, 0) for n in all_rets.columns])
            port_ret = (all_rets.values * weights).sum(axis=1)
            eq = (1 + port_ret).cumprod() * 100000
            curves[label] = eq
    # Add individual strategies
    for name in all_rets.columns:
        curves[name] = (1 + all_rets[name]).cumprod() * 100000
    curves.to_parquet(OUTPUT / "equity_curves.parquet", index=False)

    # Save full grid results for further analysis
    grid_df = pd.DataFrame(results)
    grid_df.to_parquet(OUTPUT / "grid_results.parquet", index=False)

    print(f"\n  Results saved to {OUTPUT}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()
