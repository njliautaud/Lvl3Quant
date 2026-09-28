#!/usr/bin/env python3
"""
etf_v3_hedge_overlay.py — Can a dynamic SPY hedge make ETF Rotation v3 pass R1?

ETF Rotation v3 (rotation-quality) has Sharpe 2.39 but FAILS R1 regime gate
(gap 1.51: Sharpe_green=+8.97, Sharpe_red=-4.61). This is the classic long-only
momentum vulnerability: makes money in up markets, loses in down.

Approach (proven on V5 CSP wheel, gap 1.75 -> 0.21):
  hedged_return_t = etf_return_t - hedge_ratio_t * spy_return_t

Sweep:
  A. Fixed hedge ratios: 0.10 to 0.70
  B. Dynamic: base_ratio when VIX < threshold, stress_ratio when VIX >= threshold
     - VIX thresholds: 15, 18, 20, 22, 25
     - Base/stress combos: (0.10,0.30), (0.15,0.40), (0.20,0.50), (0.25,0.60), (0.30,0.70)

R1 gate:  |Sharpe_green - Sharpe_red| / max(|Sharpe_green|,|Sharpe_red|) <= 0.50
Target:   R1 pass AND Sharpe > 1.0

Usage:
    python3 /home/jupiter/Lvl3Quant/research/etf_v3_hedge_overlay.py
"""
from __future__ import annotations

import json
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "etf_v3_hedge"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

# --------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------

def load_etf_v3_returns() -> pd.Series:
    """Load the ETF rotation quality daily returns (latest run)."""
    book = pd.read_parquet(
        ROOT / "output/macro_picker/etf_rotation_quality_20260709_223944/book.parquet"
    )
    book["date"] = pd.to_datetime(book["date"])
    return book.set_index("date")["daily_ret"].sort_index()


def load_spy_returns() -> pd.Series:
    """Load SPY daily close-to-close returns."""
    px = pd.read_parquet(ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet")
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    return spy.pct_change().dropna()


def load_vix() -> pd.Series:
    """Load VIX daily close."""
    vix = pd.read_parquet(ROOT / "wheel_strategy_v1/data/cache/vix_history.parquet")
    vix["date"] = pd.to_datetime(vix["date"])
    return vix.set_index("date")["close"].sort_index()


# --------------------------------------------------------------------------
# R1 Regime Classification (canonical: SPY close-to-close)
# --------------------------------------------------------------------------

def classify_regime(spy_ret: pd.Series, thr: float = 0.005) -> pd.Series:
    """Green: SPY daily return > +0.5%, Red: < -0.5%, else Flat."""
    labels = pd.Series("flat", index=spy_ret.index, dtype=object)
    labels[spy_ret > thr] = "green"
    labels[spy_ret < -thr] = "red"
    return labels


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(daily_ret: pd.Series, regime_labels: pd.Series) -> dict:
    """Compute Sharpe, Sortino, CAGR, MaxDD, per-regime Sharpe, R1 gap."""
    ret = daily_ret.dropna()
    if len(ret) < 30 or ret.std() == 0:
        return {"error": "insufficient data", "sharpe": float("nan")}

    # Overall
    sharpe = float(ret.mean() / ret.std() * np.sqrt(TRADING_DAYS))

    down = ret[ret < 0]
    sortino = float(
        ret.mean() / down.std() * np.sqrt(TRADING_DAYS)
    ) if len(down) > 0 and down.std() > 0 else 0.0

    equity = (1 + ret).cumprod()
    n_years = len(ret) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    pnl = ret[ret != 0]
    wr = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = float(gains / losses) if losses > 0 else float("inf")

    # Per-regime Sharpe
    aligned = regime_labels.reindex(ret.index).fillna("flat")
    regime_sharpe = {}
    regime_n = {}
    for r in ("green", "red", "flat"):
        sub = ret[aligned == r]
        regime_n[r] = int(len(sub))
        if len(sub) >= 5 and sub.std() > 0:
            regime_sharpe[r] = float(sub.mean() / sub.std() * np.sqrt(TRADING_DAYS))
        else:
            regime_sharpe[r] = float("nan")

    # R1 gap
    sg = regime_sharpe.get("green", float("nan"))
    sr = regime_sharpe.get("red", float("nan"))
    if np.isfinite(sg) and np.isfinite(sr) and max(abs(sg), abs(sr)) > 1e-9:
        r1_gap = abs(sg - sr) / max(abs(sg), abs(sr))
    else:
        r1_gap = float("nan")
    r1_pass = bool(np.isfinite(r1_gap) and r1_gap <= 0.50)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "wr": wr,
        "pf": pf,
        "n_days": len(ret),
        "sharpe_green": sg,
        "sharpe_red": sr,
        "sharpe_flat": regime_sharpe.get("flat", float("nan")),
        "n_green": regime_n.get("green", 0),
        "n_red": regime_n.get("red", 0),
        "n_flat": regime_n.get("flat", 0),
        "r1_gap": r1_gap,
        "r1_pass": r1_pass,
    }


# --------------------------------------------------------------------------
# Hedge Application
# --------------------------------------------------------------------------

def apply_fixed_hedge(etf_ret: pd.Series, spy_ret: pd.Series,
                      hedge_ratio: float) -> pd.Series:
    """hedged = etf - hedge_ratio * spy, daily."""
    common = etf_ret.index.intersection(spy_ret.index)
    return etf_ret.loc[common] - hedge_ratio * spy_ret.loc[common]


def apply_dynamic_hedge(etf_ret: pd.Series, spy_ret: pd.Series,
                        vix: pd.Series, base_ratio: float,
                        stress_ratio: float,
                        vix_threshold: float) -> pd.Series:
    """Dynamic: base_ratio when VIX(t-1) < threshold, stress_ratio otherwise."""
    common = etf_ret.index.intersection(spy_ret.index)
    etf_c = etf_ret.loc[common]
    spy_c = spy_ret.loc[common]

    # Use t-1 VIX (no look-ahead)
    vix_aligned = vix.reindex(common).ffill()
    vix_prev = vix_aligned.shift(1)

    hr = pd.Series(base_ratio, index=common)
    hr[vix_prev >= vix_threshold] = stress_ratio
    # First day: use base (no prior VIX available)
    hr.iloc[0] = base_ratio

    hedged = etf_c - hr * spy_c
    return hedged


def compute_rolling_beta(etf_ret: pd.Series, spy_ret: pd.Series,
                         window: int = 60) -> pd.Series:
    """Rolling beta of ETF returns vs SPY."""
    common = etf_ret.index.intersection(spy_ret.index)
    e = etf_ret.loc[common]
    s = spy_ret.loc[common]
    cov = e.rolling(window, min_periods=30).cov(s)
    var = s.rolling(window, min_periods=30).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)


def apply_beta_hedge(etf_ret: pd.Series, spy_ret: pd.Series,
                     beta: pd.Series, scale: float = 1.0) -> pd.Series:
    """Hedge using rolling beta * scale. Uses t-1 beta."""
    common = etf_ret.index.intersection(spy_ret.index).intersection(beta.index)
    beta_prev = beta.loc[common].shift(1).fillna(0)
    hr = beta_prev * scale
    return etf_ret.loc[common] - hr * spy_ret.loc[common]


# --------------------------------------------------------------------------
# Main Sweep
# --------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ETF ROTATION v3 — DYNAMIC SPY HEDGE OVERLAY RESEARCH")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    etf_ret = load_etf_v3_returns()
    spy_ret = load_spy_returns()
    vix = load_vix()

    # Align to common dates
    common = etf_ret.index.intersection(spy_ret.index)
    etf_ret = etf_ret.loc[common]
    spy_ret = spy_ret.loc[common]
    regime_labels = classify_regime(spy_ret)

    print(f"    ETF v3 returns: {len(etf_ret)} days, "
          f"{etf_ret.index.min().date()} to {etf_ret.index.max().date()}")
    print(f"    VIX data: {len(vix)} days, "
          f"{vix.index.min().date()} to {vix.index.max().date()}")

    # Baseline (unhedged)
    baseline = compute_metrics(etf_ret, regime_labels)
    print(f"\n[2] BASELINE (unhedged):")
    print(f"    Sharpe={baseline['sharpe']:.2f}  Sortino={baseline['sortino']:.2f}  "
          f"CAGR={baseline['cagr']*100:.1f}%  MaxDD={baseline['max_dd']*100:.1f}%")
    print(f"    Sharpe_green={baseline['sharpe_green']:.2f}  "
          f"Sharpe_red={baseline['sharpe_red']:.2f}  "
          f"R1 gap={baseline['r1_gap']:.3f}  R1_PASS={baseline['r1_pass']}")

    # Rolling beta analysis
    print("\n[3] Rolling beta analysis...")
    beta_60d = compute_rolling_beta(etf_ret, spy_ret, 60)
    beta_120d = compute_rolling_beta(etf_ret, spy_ret, 120)
    print(f"    60d beta: mean={beta_60d.mean():.3f}  "
          f"std={beta_60d.std():.3f}  "
          f"min={beta_60d.min():.3f}  max={beta_60d.max():.3f}")
    print(f"    120d beta: mean={beta_120d.mean():.3f}  "
          f"std={beta_120d.std():.3f}")

    # Beta by regime
    beta_aligned = beta_60d.reindex(common).ffill()
    regime_aligned = regime_labels.reindex(common).fillna("flat")
    for r in ("green", "red", "flat"):
        sub_beta = beta_aligned[regime_aligned == r].dropna()
        if len(sub_beta) > 0:
            print(f"    Beta in {r} regime: mean={sub_beta.mean():.3f}  "
                  f"std={sub_beta.std():.3f}")

    all_results = []

    # ----- A. Fixed hedge ratios -----
    print("\n[4] FIXED HEDGE RATIO SWEEP")
    print(f"    {'Ratio':<8} {'Sharpe':<8} {'Sortino':<9} {'CAGR':<8} {'MaxDD':<8} "
          f"{'Sh_G':<8} {'Sh_R':<8} {'R1_gap':<8} {'R1':<5}")

    fixed_ratios = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    for hr in fixed_ratios:
        hedged = apply_fixed_hedge(etf_ret, spy_ret, hr)
        m = compute_metrics(hedged, regime_labels)
        m["config_type"] = "fixed"
        m["hedge_ratio"] = hr
        m["config_label"] = f"fixed_{hr:.2f}"
        all_results.append(m)

        flag = " ***" if m["r1_pass"] and m["sharpe"] > 1.0 else ""
        print(f"    {hr:<8.2f} {m['sharpe']:<8.2f} {m['sortino']:<9.2f} "
              f"{m['cagr']*100:<8.1f} {m['max_dd']*100:<8.1f} "
              f"{m['sharpe_green']:<8.2f} {m['sharpe_red']:<8.2f} "
              f"{m['r1_gap']:<8.3f} {'PASS' if m['r1_pass'] else 'FAIL':<5}{flag}")

    # ----- B. Dynamic hedge (VIX-triggered) -----
    print("\n[5] DYNAMIC HEDGE SWEEP (VIX-triggered)")
    vix_thresholds = [15, 18, 20, 22, 25]
    base_stress_combos = [
        (0.10, 0.30), (0.15, 0.40), (0.20, 0.50),
        (0.25, 0.60), (0.30, 0.70),
    ]

    print(f"    {'Base':<6} {'Stress':<8} {'VIX_thr':<8} {'Sharpe':<8} {'Sortino':<9} "
          f"{'CAGR':<8} {'MaxDD':<8} {'Sh_G':<8} {'Sh_R':<8} {'R1_gap':<8} {'R1':<5}")

    for (base_r, stress_r), vix_thr in product(base_stress_combos, vix_thresholds):
        hedged = apply_dynamic_hedge(etf_ret, spy_ret, vix,
                                     base_r, stress_r, vix_thr)
        m = compute_metrics(hedged, regime_labels)
        m["config_type"] = "dynamic"
        m["base_ratio"] = base_r
        m["stress_ratio"] = stress_r
        m["vix_threshold"] = vix_thr
        m["config_label"] = f"dyn_{base_r:.2f}_{stress_r:.2f}_v{vix_thr}"
        all_results.append(m)

        flag = " ***" if m["r1_pass"] and m["sharpe"] > 1.0 else ""
        print(f"    {base_r:<6.2f} {stress_r:<8.2f} {vix_thr:<8} "
              f"{m['sharpe']:<8.2f} {m['sortino']:<9.2f} "
              f"{m['cagr']*100:<8.1f} {m['max_dd']*100:<8.1f} "
              f"{m['sharpe_green']:<8.2f} {m['sharpe_red']:<8.2f} "
              f"{m['r1_gap']:<8.3f} {'PASS' if m['r1_pass'] else 'FAIL':<5}{flag}")

    # ----- C. Beta-scaled hedge -----
    print("\n[6] BETA-SCALED HEDGE SWEEP")
    beta_scales = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]

    print(f"    {'Scale':<8} {'Sharpe':<8} {'Sortino':<9} {'CAGR':<8} {'MaxDD':<8} "
          f"{'Sh_G':<8} {'Sh_R':<8} {'R1_gap':<8} {'R1':<5}")

    for scale in beta_scales:
        hedged = apply_beta_hedge(etf_ret, spy_ret, beta_60d, scale)
        m = compute_metrics(hedged, regime_labels)
        m["config_type"] = "beta_scaled"
        m["beta_scale"] = scale
        m["config_label"] = f"beta_{scale:.2f}"
        all_results.append(m)

        flag = " ***" if m["r1_pass"] and m["sharpe"] > 1.0 else ""
        print(f"    {scale:<8.2f} {m['sharpe']:<8.2f} {m['sortino']:<9.2f} "
              f"{m['cagr']*100:<8.1f} {m['max_dd']*100:<8.1f} "
              f"{m['sharpe_green']:<8.2f} {m['sharpe_red']:<8.2f} "
              f"{m['r1_gap']:<8.3f} {'PASS' if m['r1_pass'] else 'FAIL':<5}{flag}")

    # ----- D. Dynamic beta-scaled + VIX -----
    print("\n[7] DYNAMIC BETA HEDGE (beta * scale, VIX-triggered boost)")
    dyn_beta_configs = [
        (0.50, 1.00, 18), (0.50, 1.25, 18), (0.50, 1.50, 18),
        (0.75, 1.25, 18), (0.75, 1.50, 18), (0.75, 1.25, 20),
        (0.50, 1.00, 20), (0.50, 1.25, 20), (0.50, 1.50, 20),
        (0.75, 1.00, 20), (0.75, 1.50, 20),
        (0.50, 1.00, 22), (0.75, 1.25, 22),
    ]
    print(f"    {'Base_s':<8} {'Stress_s':<9} {'VIX_thr':<8} {'Sharpe':<8} {'Sortino':<9} "
          f"{'CAGR':<8} {'MaxDD':<8} {'Sh_G':<8} {'Sh_R':<8} {'R1_gap':<8} {'R1':<5}")

    for base_s, stress_s, vix_thr in dyn_beta_configs:
        common_b = etf_ret.index.intersection(spy_ret.index).intersection(beta_60d.index)
        beta_prev = beta_60d.loc[common_b].shift(1).fillna(0)
        vix_aligned = vix.reindex(common_b).ffill()
        vix_prev = vix_aligned.shift(1)

        scale = pd.Series(base_s, index=common_b)
        scale[vix_prev >= vix_thr] = stress_s
        scale.iloc[0] = base_s

        hr = beta_prev * scale
        hedged = etf_ret.loc[common_b] - hr * spy_ret.loc[common_b]
        m = compute_metrics(hedged, regime_labels)
        m["config_type"] = "dynamic_beta"
        m["base_scale"] = base_s
        m["stress_scale"] = stress_s
        m["vix_threshold"] = vix_thr
        m["config_label"] = f"dynbeta_{base_s:.2f}_{stress_s:.2f}_v{vix_thr}"
        all_results.append(m)

        flag = " ***" if m["r1_pass"] and m["sharpe"] > 1.0 else ""
        print(f"    {base_s:<8.2f} {stress_s:<9.2f} {vix_thr:<8} "
              f"{m['sharpe']:<8.2f} {m['sortino']:<9.2f} "
              f"{m['cagr']*100:<8.1f} {m['max_dd']*100:<8.1f} "
              f"{m['sharpe_green']:<8.2f} {m['sharpe_red']:<8.2f} "
              f"{m['r1_gap']:<8.3f} {'PASS' if m['r1_pass'] else 'FAIL':<5}{flag}")

    # ========================================================================
    # Summary: find best configs
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Filter R1-passing configs with Sharpe > 1.0
    passing = [r for r in all_results
                if r.get("r1_pass", False) and r.get("sharpe", 0) > 1.0]

    print(f"\n  Configs tested: {len(all_results)}")
    print(f"  R1-passing with Sharpe > 1.0: {len(passing)}")

    if passing:
        # Sort by Sharpe descending
        passing.sort(key=lambda x: x.get("sharpe", 0), reverse=True)

        print(f"\n  TOP 10 R1-COMPLIANT CONFIGS (by Sharpe):")
        print(f"    {'#':<3} {'Label':<32} {'Sharpe':<8} {'Sortino':<9} "
              f"{'CAGR':<8} {'MaxDD':<8} {'Sh_G':<8} {'Sh_R':<8} {'R1gap':<8}")
        for i, r in enumerate(passing[:10]):
            print(f"    {i+1:<3} {r['config_label']:<32} "
                  f"{r['sharpe']:<8.2f} {r['sortino']:<9.2f} "
                  f"{r['cagr']*100:<8.1f} {r['max_dd']*100:<8.1f} "
                  f"{r['sharpe_green']:<8.2f} {r['sharpe_red']:<8.2f} "
                  f"{r['r1_gap']:<8.3f}")

        best = passing[0]
        print(f"\n  BEST CONFIG: {best['config_label']}")
        print(f"    Sharpe: {best['sharpe']:.2f} (baseline: {baseline['sharpe']:.2f}, "
              f"cost: {baseline['sharpe'] - best['sharpe']:.2f})")
        print(f"    R1 gap: {best['r1_gap']:.3f} (baseline: {baseline['r1_gap']:.3f})")
        print(f"    CAGR: {best['cagr']*100:.1f}% (baseline: {baseline['cagr']*100:.1f}%)")
        print(f"    MaxDD: {best['max_dd']*100:.1f}% (baseline: {baseline['max_dd']*100:.1f}%)")
    else:
        print("\n  NO CONFIG PASSES R1 WITH SHARPE > 1.0")

        # Find closest
        near = [r for r in all_results if r.get("r1_pass", False)]
        if near:
            near.sort(key=lambda x: x.get("sharpe", 0), reverse=True)
            best_near = near[0]
            print(f"  Closest R1-passing config: {best_near['config_label']}")
            print(f"    Sharpe: {best_near['sharpe']:.2f}  R1_gap: {best_near['r1_gap']:.3f}")
        else:
            # Best R1 gap among all
            by_gap = sorted(all_results,
                            key=lambda x: x.get("r1_gap", float("inf")))
            best_gap = by_gap[0]
            print(f"  Closest to R1 pass: {best_gap['config_label']}")
            print(f"    R1 gap: {best_gap['r1_gap']:.3f}  Sharpe: {best_gap['sharpe']:.2f}")

    # Sharpe cost analysis
    print(f"\n  SHARPE COST OF HEDGING:")
    for hr in [0.20, 0.30, 0.40, 0.50]:
        fixed_match = [r for r in all_results
                        if r.get("config_type") == "fixed" and abs(r.get("hedge_ratio", 0) - hr) < 0.01]
        if fixed_match:
            m = fixed_match[0]
            cost = baseline["sharpe"] - m["sharpe"]
            print(f"    HR={hr:.2f}: Sharpe {m['sharpe']:.2f} "
                  f"(cost {cost:+.2f}), R1 gap {m['r1_gap']:.3f} "
                  f"({'PASS' if m['r1_pass'] else 'FAIL'})")

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT / "sweep_results.csv", index=False)

    # Save best config
    summary = {
        "baseline": baseline,
        "n_configs_tested": len(all_results),
        "n_r1_passing": len([r for r in all_results if r.get("r1_pass")]),
        "n_r1_passing_sharpe_gt_1": len(passing),
        "best_config": passing[0] if passing else None,
        "top_5": passing[:5] if passing else [],
        "conclusion": (
            f"YES - ETF v3 can be made R1-compliant. "
            f"Best: {passing[0]['config_label']} with Sharpe {passing[0]['sharpe']:.2f} "
            f"(cost {baseline['sharpe'] - passing[0]['sharpe']:.2f} from {baseline['sharpe']:.2f})"
            if passing else
            "NO - no hedge config achieves R1 pass with Sharpe > 1.0"
        ),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # Save hedged equity curve for best config
    if passing:
        best = passing[0]
        if best["config_type"] == "fixed":
            best_hedged = apply_fixed_hedge(etf_ret, spy_ret, best["hedge_ratio"])
        elif best["config_type"] == "dynamic":
            best_hedged = apply_dynamic_hedge(
                etf_ret, spy_ret, vix,
                best["base_ratio"], best["stress_ratio"], best["vix_threshold"]
            )
        elif best["config_type"] == "beta_scaled":
            best_hedged = apply_beta_hedge(etf_ret, spy_ret, beta_60d, best["beta_scale"])
        elif best["config_type"] == "dynamic_beta":
            common_b = etf_ret.index.intersection(spy_ret.index).intersection(beta_60d.index)
            bp = beta_60d.loc[common_b].shift(1).fillna(0)
            va = vix.reindex(common_b).ffill()
            vp = va.shift(1)
            sc = pd.Series(best["base_scale"], index=common_b)
            sc[vp >= best["vix_threshold"]] = best["stress_scale"]
            sc.iloc[0] = best["base_scale"]
            hr = bp * sc
            best_hedged = etf_ret.loc[common_b] - hr * spy_ret.loc[common_b]
        else:
            best_hedged = etf_ret

        equity_hedged = (1 + best_hedged).cumprod()
        equity_base = (1 + etf_ret).cumprod()
        equity_spy = (1 + spy_ret.loc[etf_ret.index]).cumprod()

        eq_df = pd.DataFrame({
            "date": equity_hedged.index,
            "equity_hedged": equity_hedged.values,
            "equity_base": equity_base.reindex(equity_hedged.index).values,
            "equity_spy": equity_spy.reindex(equity_hedged.index).values,
        })
        eq_df.to_parquet(OUTPUT / "equity_curves.parquet", index=False)

    print(f"\n  Output saved to: {OUTPUT}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    main()
