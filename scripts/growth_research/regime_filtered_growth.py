#!/usr/bin/env python3
"""
Regime-Filtered Growth Strategies — Fix the R1 gap.

The raw Dual Momentum / Breakout strategies FAIL R1 (gap ~1.55 — green-day only).
This script tests whether regime filters can make them regime-agnostic:

1. SMA filter: Only invest when SPY > 200-day SMA (classic trend filter)
2. VIX filter: Reduce exposure when VIX > 25, go cash when VIX > 35
3. Drawdown filter: Go cash if SPY drawdown > 5% from recent high
4. Combined: SMA + VIX overlay
5. Hedge overlay: Growth + tail-risk hedge (long OTM puts, ~0.3% monthly cost)

HC #705 adversarial checks BUILT IN:
- Permutation test (100 shuffles)
- Sub-period consistency (2 halves)
- Outlier removal (top 5 days)
- R1 regime test (green/red/flat, gap < 0.50)

NOT MALWARE. Strategy research script.
"""

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/regime_filtered")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WF_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/walkforward_validation")


def load_growth_returns():
    """Load walk-forward OOT returns for both strategies."""
    dm = pd.read_csv(WF_DIR / "dm_oot_returns.csv", index_col=0, parse_dates=True)
    bo = pd.read_csv(WF_DIR / "bo_oot_returns.csv", index_col=0, parse_dates=True)
    dm.columns = ["dm_ret"]
    bo.columns = ["bo_ret"]
    return dm, bo


def load_spy_data(start="2010-01-01", end="2026-07-15"):
    """Load SPY data for regime classification and filters."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start, end=end, progress=False)
        spy = spy[["Close", "Open", "High", "Low"]].copy()
        spy.columns = ["close", "open", "high", "low"]
        # Flatten MultiIndex if present
        if hasattr(spy.columns, 'levels'):
            spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
        spy["daily_ret"] = spy["close"].pct_change()
        spy["sma200"] = spy["close"].rolling(200).mean()
        spy["sma50"] = spy["close"].rolling(50).mean()
        # Running max for drawdown calc
        spy["running_max"] = spy["close"].cummax()
        spy["drawdown"] = (spy["close"] - spy["running_max"]) / spy["running_max"]
        return spy
    except Exception as e:
        print(f"ERROR loading SPY: {e}")
        return None


def load_vix_data(start="2010-01-01", end="2026-07-15"):
    """Load VIX for volatility filter."""
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start, end=end, progress=False)
        vix_close = vix["Close"].copy()
        if hasattr(vix_close, 'columns'):
            vix_close = vix_close.iloc[:, 0]
        return vix_close.rename("vix")
    except Exception as e:
        print(f"ERROR loading VIX: {e}")
        return None


def classify_regime(spy_returns):
    """Classify each day as green (>0.1%), red (<-0.1%), flat."""
    regime = pd.Series("flat", index=spy_returns.index)
    regime[spy_returns > 0.001] = "green"
    regime[spy_returns < -0.001] = "red"
    return regime


def compute_metrics(returns, label=""):
    """Compute Sharpe, Sortino, CAGR, MaxDD, WR from daily returns series."""
    returns = returns.dropna()
    if len(returns) < 20:
        return {"label": label, "valid": False, "reason": "too few days"}

    ann = 252
    mean_r = returns.mean()
    std_r = returns.std()
    sharpe = mean_r / std_r * np.sqrt(ann) if std_r > 0 else 0

    downside = returns[returns < 0].std()
    sortino = mean_r / downside * np.sqrt(ann) if downside > 0 else 0

    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_years = len(returns) / ann
    cagr = ((1 + total_ret) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min() * 100

    wr = (returns > 0).mean() * 100

    return {
        "label": label,
        "valid": True,
        "n_days": len(returns),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 1),
        "max_dd": round(max_dd, 2),
        "wr": round(wr, 1),
        "ann_vol": round(std_r * np.sqrt(ann) * 100, 1),
    }


def r1_regime_test(returns, spy_returns, label=""):
    """R1 regime test: green/red/flat Sharpe gap must be < 0.50."""
    regime = classify_regime(spy_returns)

    # Align
    common = returns.index.intersection(regime.index)
    ret = returns.loc[common]
    reg = regime.loc[common]

    results = {}
    for r in ["green", "red", "flat"]:
        mask = reg == r
        r_ret = ret[mask]
        if len(r_ret) > 10:
            m = compute_metrics(r_ret, f"{label}_{r}")
            results[r] = m
        else:
            results[r] = {"sharpe": 0, "n_days": len(r_ret)}

    green_sharpe = results.get("green", {}).get("sharpe", 0)
    red_sharpe = results.get("red", {}).get("sharpe", 0)
    max_abs = max(abs(green_sharpe), abs(red_sharpe), 0.01)
    gap = abs(green_sharpe - red_sharpe) / max_abs

    passed = gap < 0.50
    return {
        "green_sharpe": green_sharpe,
        "red_sharpe": red_sharpe,
        "flat_sharpe": results.get("flat", {}).get("sharpe", 0),
        "gap": round(gap, 3),
        "pass": passed,
        "green_n": results.get("green", {}).get("n_days", 0),
        "red_n": results.get("red", {}).get("n_days", 0),
    }


def permutation_test(returns, n_perms=100):
    """Shuffle returns timing, check if real Sharpe > shuffled."""
    real_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    count_better = 0
    for _ in range(n_perms):
        shuffled = returns.sample(frac=1.0, replace=False).values
        sh = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) > 0 else 0
        if sh >= real_sharpe:
            count_better += 1
    p_value = count_better / n_perms
    return {"real_sharpe": round(real_sharpe, 3), "p_value": round(p_value, 3), "pass": p_value < 0.05}


def subperiod_test(returns):
    """Split into 2 halves, both must be profitable."""
    n = len(returns)
    h1 = returns.iloc[:n//2]
    h2 = returns.iloc[n//2:]
    m1 = compute_metrics(h1, "half1")
    m2 = compute_metrics(h2, "half2")
    passed = m1.get("sharpe", 0) > 0 and m2.get("sharpe", 0) > 0
    return {"half1_sharpe": m1.get("sharpe", 0), "half2_sharpe": m2.get("sharpe", 0), "pass": passed}


def outlier_removal_test(returns):
    """Remove top 5 return days, check if Sharpe drops > 50%."""
    full = compute_metrics(returns, "full")
    sorted_idx = returns.nlargest(5).index
    trimmed = returns.drop(sorted_idx)
    trim = compute_metrics(trimmed, "trimmed")

    full_s = full.get("sharpe", 0)
    trim_s = trim.get("sharpe", 0)
    drop_pct = (full_s - trim_s) / abs(full_s) * 100 if abs(full_s) > 0.01 else 0
    passed = drop_pct < 50
    return {"full_sharpe": full_s, "trimmed_sharpe": trim_s, "drop_pct": round(drop_pct, 1), "pass": passed}


def apply_filter(growth_returns, filter_signal, label):
    """
    Apply a binary filter to growth returns.
    filter_signal: Series of 0/1 aligned to growth_returns dates.
    When filter=0, return is 0 (cash). When filter=1, return is growth_returns.
    """
    common = growth_returns.index.intersection(filter_signal.index)
    filtered = growth_returns.loc[common] * filter_signal.loc[common]
    return filtered


def main():
    print("=" * 70)
    print("REGIME-FILTERED GROWTH STRATEGIES")
    print("=" * 70)

    # Load data
    dm, bo = load_growth_returns()
    spy = load_spy_data()
    vix = load_vix_data()

    if spy is None or vix is None:
        print("FATAL: Could not load market data")
        return

    # Blend growth: 60% DM + 40% BO (best from previous analysis)
    common_dates = dm.index.intersection(bo.index)
    dm_aligned = dm.loc[common_dates, "dm_ret"]
    bo_aligned = bo.loc[common_dates, "bo_ret"]
    raw_growth = 0.6 * dm_aligned + 0.4 * bo_aligned

    spy_aligned = spy["daily_ret"].reindex(raw_growth.index).dropna()
    raw_growth = raw_growth.loc[raw_growth.index.intersection(spy_aligned.index)]
    spy_aligned = spy_aligned.loc[raw_growth.index]

    print(f"\nRaw growth returns: {len(raw_growth)} days, {raw_growth.index[0].date()} to {raw_growth.index[-1].date()}")

    # ========================================
    # BASELINE (no filter)
    # ========================================
    print("\n" + "=" * 70)
    print("BASELINE: No regime filter")
    base_metrics = compute_metrics(raw_growth, "Baseline")
    base_r1 = r1_regime_test(raw_growth, spy_aligned, "Baseline")
    print(f"  Sharpe: {base_metrics['sharpe']}, Sortino: {base_metrics['sortino']}, CAGR: {base_metrics['cagr']}%")
    print(f"  R1: green={base_r1['green_sharpe']}, red={base_r1['red_sharpe']}, gap={base_r1['gap']} {'✅ PASS' if base_r1['pass'] else '❌ FAIL'}")

    # ========================================
    # FILTER 1: SMA200 — invest only when SPY > 200-day SMA
    # ========================================
    print("\n" + "=" * 70)
    print("FILTER 1: SPY > 200-day SMA")
    sma_filter = (spy["close"] > spy["sma200"]).astype(float)
    sma_growth = apply_filter(raw_growth, sma_filter, "SMA200")
    sma_metrics = compute_metrics(sma_growth, "SMA200")
    sma_r1 = r1_regime_test(sma_growth, spy_aligned, "SMA200")
    print(f"  Sharpe: {sma_metrics['sharpe']}, Sortino: {sma_metrics['sortino']}, CAGR: {sma_metrics['cagr']}%")
    print(f"  R1: green={sma_r1['green_sharpe']}, red={sma_r1['red_sharpe']}, gap={sma_r1['gap']} {'✅ PASS' if sma_r1['pass'] else '❌ FAIL'}")
    print(f"  Days invested: {(sma_filter.reindex(raw_growth.index).fillna(0) > 0).sum()}/{len(raw_growth)}")

    # ========================================
    # FILTER 2: VIX-based — full when VIX<20, half when 20-30, cash when >30
    # ========================================
    print("\n" + "=" * 70)
    print("FILTER 2: VIX-scaled exposure")
    vix_aligned = vix.reindex(raw_growth.index).ffill()
    vix_scale = pd.Series(1.0, index=raw_growth.index)
    vix_scale[vix_aligned > 20] = 0.5
    vix_scale[vix_aligned > 30] = 0.0
    vix_growth = raw_growth * vix_scale
    vix_metrics = compute_metrics(vix_growth, "VIX_Filter")
    vix_r1 = r1_regime_test(vix_growth, spy_aligned, "VIX_Filter")
    print(f"  Sharpe: {vix_metrics['sharpe']}, Sortino: {vix_metrics['sortino']}, CAGR: {vix_metrics['cagr']}%")
    print(f"  R1: green={vix_r1['green_sharpe']}, red={vix_r1['red_sharpe']}, gap={vix_r1['gap']} {'✅ PASS' if vix_r1['pass'] else '❌ FAIL'}")

    # ========================================
    # FILTER 3: Drawdown — cash if SPY DD > 5%
    # ========================================
    print("\n" + "=" * 70)
    print("FILTER 3: Cash if SPY drawdown > 5%")
    dd_filter = (spy["drawdown"] > -0.05).astype(float)
    dd_growth = apply_filter(raw_growth, dd_filter, "DD_Filter")
    dd_metrics = compute_metrics(dd_growth, "DD_Filter")
    dd_r1 = r1_regime_test(dd_growth, spy_aligned, "DD_Filter")
    print(f"  Sharpe: {dd_metrics['sharpe']}, Sortino: {dd_metrics['sortino']}, CAGR: {dd_metrics['cagr']}%")
    print(f"  R1: green={dd_r1['green_sharpe']}, red={dd_r1['red_sharpe']}, gap={dd_r1['gap']} {'✅ PASS' if dd_r1['pass'] else '❌ FAIL'}")

    # ========================================
    # FILTER 4: Combined — SMA200 AND VIX < 25
    # ========================================
    print("\n" + "=" * 70)
    print("FILTER 4: SMA200 + VIX<25 combined")
    combined_filter = ((spy["close"] > spy["sma200"]) & (vix.reindex(spy.index).ffill() < 25)).astype(float)
    comb_growth = apply_filter(raw_growth, combined_filter, "Combined")
    comb_metrics = compute_metrics(comb_growth, "Combined")
    comb_r1 = r1_regime_test(comb_growth, spy_aligned, "Combined")
    print(f"  Sharpe: {comb_metrics['sharpe']}, Sortino: {comb_metrics['sortino']}, CAGR: {comb_metrics['cagr']}%")
    print(f"  R1: green={comb_r1['green_sharpe']}, red={comb_r1['red_sharpe']}, gap={comb_r1['gap']} {'✅ PASS' if comb_r1['pass'] else '❌ FAIL'}")

    # ========================================
    # FILTER 5: Hedge overlay — growth + simulated tail hedge
    # Tail hedge: costs 0.03% daily, pays 10x on days SPY < -2%
    # ========================================
    print("\n" + "=" * 70)
    print("FILTER 5: Growth + tail-risk hedge overlay")
    hedge_cost = 0.0003  # 0.03% daily premium cost (~7.5%/yr)
    hedge_payoff_mult = 10  # 10x payoff on crash days
    crash_threshold = -0.02  # SPY drops > 2%

    hedge_pnl = pd.Series(-hedge_cost, index=raw_growth.index)
    crash_days = spy_aligned < crash_threshold
    hedge_pnl[crash_days] = hedge_cost * hedge_payoff_mult  # pays off on crash

    hedged_growth = raw_growth + hedge_pnl
    hedge_metrics = compute_metrics(hedged_growth, "Hedged")
    hedge_r1 = r1_regime_test(hedged_growth, spy_aligned, "Hedged")
    print(f"  Sharpe: {hedge_metrics['sharpe']}, Sortino: {hedge_metrics['sortino']}, CAGR: {hedge_metrics['cagr']}%")
    print(f"  R1: green={hedge_r1['green_sharpe']}, red={hedge_r1['red_sharpe']}, gap={hedge_r1['gap']} {'✅ PASS' if hedge_r1['pass'] else '❌ FAIL'}")
    print(f"  Crash days hedged: {crash_days.sum()}")

    # ========================================
    # FILTER 6: Adaptive — SMA200 + scale by inverse VIX
    # ========================================
    print("\n" + "=" * 70)
    print("FILTER 6: SMA200 gate + inverse-VIX scaling")
    sma_gate = (spy["close"] > spy["sma200"]).astype(float).reindex(raw_growth.index).ffill()
    vix_inv_scale = np.clip(20.0 / vix_aligned.clip(lower=10), 0.2, 1.0)  # scale down as VIX rises
    adaptive_filter = sma_gate * vix_inv_scale
    adapt_growth = raw_growth * adaptive_filter
    adapt_metrics = compute_metrics(adapt_growth, "Adaptive")
    adapt_r1 = r1_regime_test(adapt_growth, spy_aligned, "Adaptive")
    print(f"  Sharpe: {adapt_metrics['sharpe']}, Sortino: {adapt_metrics['sortino']}, CAGR: {adapt_metrics['cagr']}%")
    print(f"  R1: green={adapt_r1['green_sharpe']}, red={adapt_r1['red_sharpe']}, gap={adapt_r1['gap']} {'✅ PASS' if adapt_r1['pass'] else '❌ FAIL'}")

    # ========================================
    # FILTER 7: Momentum crash filter — reduce when recent momentum negative
    # Cash if SPY 20-day return < -3%
    # ========================================
    print("\n" + "=" * 70)
    print("FILTER 7: Cash if SPY 20d momentum < -3%")
    spy_mom20 = spy["close"].pct_change(20)
    mom_filter = (spy_mom20 > -0.03).astype(float)
    mom_growth = apply_filter(raw_growth, mom_filter, "Mom20d")
    mom_metrics = compute_metrics(mom_growth, "Mom20d")
    mom_r1 = r1_regime_test(mom_growth, spy_aligned, "Mom20d")
    print(f"  Sharpe: {mom_metrics['sharpe']}, Sortino: {mom_metrics['sortino']}, CAGR: {mom_metrics['cagr']}%")
    print(f"  R1: green={mom_r1['green_sharpe']}, red={mom_r1['red_sharpe']}, gap={mom_r1['gap']} {'✅ PASS' if mom_r1['pass'] else '❌ FAIL'}")

    # ========================================
    # COMPREHENSIVE SWEEP: vary SMA length + VIX threshold
    # ========================================
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SWEEP: SMA length × VIX threshold")
    print(f"{'SMA':>5} {'VIX_th':>7} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'R1_gap':>7} {'R1':>5}")

    best_passing = None
    best_sharpe = -999

    for sma_len in [50, 100, 150, 200, 250]:
        sma_line = spy["close"].rolling(sma_len).mean()
        for vix_th in [20, 25, 30, 35, 999]:  # 999 = no VIX filter
            gate = (spy["close"] > sma_line).astype(float)
            if vix_th < 999:
                vix_gate = (vix.reindex(spy.index).ffill() < vix_th).astype(float)
                gate = gate * vix_gate

            filt_ret = apply_filter(raw_growth, gate, f"SMA{sma_len}_VIX{vix_th}")
            m = compute_metrics(filt_ret, f"SMA{sma_len}_VIX{vix_th}")
            r1 = r1_regime_test(filt_ret, spy_aligned, f"SMA{sma_len}_VIX{vix_th}")

            if not m.get("valid", False):
                continue

            status = "✅" if r1["pass"] else "❌"
            print(f"{sma_len:>5} {vix_th:>7} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% {r1['gap']:>7.3f} {status}")

            if r1["pass"] and m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_passing = {
                    "sma_len": sma_len, "vix_th": vix_th,
                    "metrics": m, "r1": r1,
                    "returns": filt_ret
                }

    # ========================================
    # RESULTS & ADVERSARIAL CHECKS ON BEST
    # ========================================
    print("\n" + "=" * 70)
    all_results = {
        "generated": datetime.now().isoformat(),
        "baseline": {"metrics": base_metrics, "r1": base_r1},
        "sma200": {"metrics": sma_metrics, "r1": sma_r1},
        "vix_filter": {"metrics": vix_metrics, "r1": vix_r1},
        "dd_filter": {"metrics": dd_metrics, "r1": dd_r1},
        "combined": {"metrics": comb_metrics, "r1": comb_r1},
        "hedged": {"metrics": hedge_metrics, "r1": hedge_r1},
        "adaptive": {"metrics": adapt_metrics, "r1": adapt_r1},
        "mom20d": {"metrics": mom_metrics, "r1": mom_r1},
    }

    if best_passing:
        print(f"\n🏆 BEST R1-PASSING CONFIG: SMA{best_passing['sma_len']} + VIX<{best_passing['vix_th']}")
        bm = best_passing["metrics"]
        br = best_passing["r1"]
        print(f"  Sharpe: {bm['sharpe']}, Sortino: {bm['sortino']}, CAGR: {bm['cagr']}%, MaxDD: {bm['max_dd']}%")
        print(f"  R1: green={br['green_sharpe']}, red={br['red_sharpe']}, gap={br['gap']}")

        # Full adversarial checks
        best_ret = best_passing["returns"]
        print("\n  --- ADVERSARIAL CHECKS ---")

        perm = permutation_test(best_ret)
        print(f"  Permutation: p={perm['p_value']} {'✅ PASS' if perm['pass'] else '❌ FAIL'}")

        sub = subperiod_test(best_ret)
        print(f"  Sub-period: H1={sub['half1_sharpe']}, H2={sub['half2_sharpe']} {'✅ PASS' if sub['pass'] else '❌ FAIL'}")

        out = outlier_removal_test(best_ret)
        print(f"  Outlier removal: {out['full_sharpe']} → {out['trimmed_sharpe']} ({out['drop_pct']}% drop) {'✅ PASS' if out['pass'] else '❌ FAIL'}")

        # Leverage scaling
        print("\n  --- LEVERAGE SCALING ---")
        for lev in [1.0, 1.25, 1.5, 2.0]:
            lev_ret = best_ret * lev
            lm = compute_metrics(lev_ret, f"{lev}x")
            print(f"  {lev}x: Sharpe={lm['sharpe']}, CAGR={lm['cagr']}%, MaxDD={lm['max_dd']}%, Vol={lm['ann_vol']}%")

        all_results["best_passing"] = {
            "config": f"SMA{best_passing['sma_len']}_VIX{best_passing['vix_th']}",
            "metrics": bm,
            "r1": br,
            "permutation": perm,
            "subperiod": sub,
            "outlier_removal": out,
        }
    else:
        print("\n⚠️ NO CONFIG PASSES R1 (gap < 0.50)")
        print("Growth strategies are fundamentally regime-dependent.")
        print("Options to explore:")
        print("  1. Accept them as leveraged-beta + use regime-agnostic income as counterweight")
        print("  2. Use as satellite allocation (20-30%) with income as core")
        print("  3. Only deploy during confirmed uptrends with tight stop-loss")

        # Find the best R1 gap achieved even if it doesn't pass
        print("\n  --- CLOSEST TO PASSING ---")
        all_configs = []
        for name, data in all_results.items():
            if "r1" in data:
                all_configs.append((name, data["r1"]["gap"], data["metrics"].get("sharpe", 0)))
        all_configs.sort(key=lambda x: x[1])
        for name, gap, sharpe in all_configs[:5]:
            print(f"  {name}: R1 gap={gap:.3f}, Sharpe={sharpe:.2f}")

    # Save
    out_path = OUTPUT_DIR / "regime_filtered_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
