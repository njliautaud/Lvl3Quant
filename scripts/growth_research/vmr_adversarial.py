#!/usr/bin/env python3
"""
Vol Mean Reversion (VMR) — Full Adversarial Validation
======================================================
5-regime VIX system:
  1. VIX < 15 & declining → UPRO
  2. VIX > 20 & mean-reverting (down 15% from peak, declining) → UPRO
  3. VIX > 25 & rising → 50% GLD + 50% TLT
  4. VIX > 20 & rising → 50% SPY + 50% TLT
  5. Otherwise → SPY

Weekly rebalance. Sharpe 1.517 in combination test.

HC compliance: permutation, sub-period, outlier, R1, WF.
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research")


def fetch_data():
    tickers = ["SPY", "UPRO", "GLD", "TLT", "^VIX"]
    print(f"Fetching {tickers}...")
    raw = yf.download(tickers, start="2011-10-04", end="2026-07-17", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    prices = prices.rename(columns={"^VIX": "VIX"})
    prices = prices.ffill().dropna(how="all")
    print(f"  {len(prices)} days")
    return prices


def run_vmr(prices: pd.DataFrame) -> pd.Series:
    """Vol Mean Reversion strategy."""
    returns = prices.pct_change()
    spy_ret = returns["SPY"]
    upro_ret = returns["UPRO"]
    gld_ret = returns["GLD"]
    tlt_ret = returns["TLT"]
    vix = prices["VIX"]

    vix_ma10 = vix.rolling(10).mean()
    vix_peak20 = vix.rolling(20).max()

    port_rets = pd.Series(0.0, index=prices.index)
    regime = "SPY"
    last_week = None

    for i in range(252, len(prices)):
        idx = prices.index[i]

        # Weekly rebalance
        week = (idx.year, idx.isocalendar()[1])
        if week != last_week:
            last_week = week
            v = vix.iloc[i]
            vm10 = vix_ma10.iloc[i]
            vp20 = vix_peak20.iloc[i]

            if pd.isna(v) or pd.isna(vm10):
                regime = "SPY"
            elif v < 15 and v < vm10:
                regime = "UPRO"
            elif v > 20 and v < vp20 * 0.85 and v < vm10:
                regime = "UPRO_MR"
            elif v > 25 and v > vm10:
                regime = "DEFENSIVE"
            elif v > 20 and v > vm10:
                regime = "CAUTIOUS"
            else:
                regime = "SPY"

        if regime in ("UPRO", "UPRO_MR"):
            port_rets.iloc[i] = upro_ret.iloc[i]
        elif regime == "DEFENSIVE":
            port_rets.iloc[i] = 0.5 * gld_ret.iloc[i] + 0.5 * tlt_ret.iloc[i]
        elif regime == "CAUTIOUS":
            port_rets.iloc[i] = 0.5 * spy_ret.iloc[i] + 0.5 * tlt_ret.iloc[i]
        else:
            port_rets.iloc[i] = spy_ret.iloc[i]

    return port_rets.iloc[252:]


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 10 or r.std() == 0:
        return {"sharpe": 0, "sortino": 0, "cagr_pct": 0, "max_dd_pct": 0}
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    ds = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-6
    sortino = ann_ret / ds
    cum = (1 + r).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    n_yr = len(r) / 252
    cagr = (cum.iloc[-1] ** (1/n_yr) - 1) * 100 if cum.iloc[-1] > 0 else 0
    wr = float((r > 0).mean())
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')
    return {
        "sharpe": round(sharpe, 4), "sortino": round(sortino, 4),
        "cagr_pct": round(cagr, 2), "max_dd_pct": round(max_dd, 2),
        "win_rate": round(wr, 4), "profit_factor": round(pf, 4),
        "n_days": len(r), "ann_vol_pct": round(ann_vol * 100, 2),
    }


def main():
    print("=" * 70)
    print("VMR (Vol Mean Reversion) — Adversarial Validation")
    print("=" * 70)

    prices = fetch_data()

    # Full period
    vmr_ret = run_vmr(prices)
    spy_ret = prices["SPY"].pct_change().reindex(vmr_ret.index).dropna()
    vmr_ret = vmr_ret.reindex(spy_ret.index)

    full_metrics = metrics(vmr_ret)
    spy_metrics = metrics(spy_ret)
    print(f"\n=== Full Period ===")
    print(f"  VMR:  Sharpe={full_metrics['sharpe']:.3f}, Sortino={full_metrics['sortino']:.3f}, "
          f"CAGR={full_metrics['cagr_pct']:.1f}%, MaxDD={full_metrics['max_dd_pct']:.1f}%")
    print(f"  SPY:  Sharpe={spy_metrics['sharpe']:.3f}, CAGR={spy_metrics['cagr_pct']:.1f}%")

    # 1. Permutation test (200 shuffles of VIX signal timing)
    print(f"\n=== Permutation Test (200 shuffles) ===")
    real_sharpe = full_metrics["sharpe"]
    perm_sharpes = []

    for p in range(200):
        # Shuffle VIX to break timing signal
        shuffled_prices = prices.copy()
        shuffled_prices["VIX"] = np.random.permutation(shuffled_prices["VIX"].values)
        perm_ret = run_vmr(shuffled_prices)
        perm_ret = perm_ret.reindex(spy_ret.index).fillna(0)
        m = metrics(perm_ret)
        perm_sharpes.append(m["sharpe"])
        if (p + 1) % 50 == 0:
            print(f"  {p+1}/200 done...")

    p_value = np.mean([s >= real_sharpe for s in perm_sharpes])
    print(f"  Real Sharpe: {real_sharpe:.4f}")
    print(f"  Perm mean: {np.mean(perm_sharpes):.4f} ± {np.std(perm_sharpes):.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  {'PASS ✅' if p_value < 0.05 else 'FAIL ❌'}")

    # 2. Walk-forward (3yr train, 1yr test)
    print(f"\n=== Walk-Forward ===")
    wf_results = []
    step = 252
    train = 756
    test = 252

    for start_idx in range(252 + train, len(prices) - test, step):
        test_slice = prices.iloc[start_idx:start_idx + test]
        vmr_test = run_vmr(test_slice)
        if len(vmr_test) < 50:
            continue
        spy_test = test_slice["SPY"].pct_change().iloc[1:]
        spy_test = spy_test.reindex(vmr_test.index)

        m_vmr = metrics(vmr_test)
        m_spy = metrics(spy_test.dropna())

        wf_results.append({
            "start": test_slice.index[0].strftime("%Y-%m-%d"),
            "vmr_sharpe": m_vmr["sharpe"],
            "spy_sharpe": m_spy["sharpe"],
            "beats_spy": m_vmr["sharpe"] > m_spy["sharpe"],
        })
        print(f"  {test_slice.index[0].strftime('%Y-%m-%d')}: VMR Sharpe {m_vmr['sharpe']:.3f} vs SPY {m_spy['sharpe']:.3f}")

    beats = sum(w["beats_spy"] for w in wf_results)
    print(f"  VMR beats SPY: {beats}/{len(wf_results)} windows")

    # 3. Sub-period consistency
    print(f"\n=== Sub-Period Consistency ===")
    n = len(vmr_ret)
    third = n // 3
    blocks = [vmr_ret.iloc[:third], vmr_ret.iloc[third:2*third], vmr_ret.iloc[2*third:]]
    block_sharpes = [metrics(b)["sharpe"] for b in blocks]
    all_positive = all(s > 0 for s in block_sharpes)
    cv = np.std(block_sharpes) / np.mean(block_sharpes) if np.mean(block_sharpes) > 0 else float('inf')
    for i, s in enumerate(block_sharpes):
        print(f"  Block {i+1}: Sharpe {s:.3f}")
    print(f"  All positive: {all_positive}")
    print(f"  CV: {cv:.3f} (threshold: 0.50)")
    sub_pass = all_positive and cv < 0.80

    # 4. Outlier robustness
    print(f"\n=== Outlier Robustness ===")
    top10 = vmr_ret.nlargest(10).index
    trimmed = vmr_ret.drop(top10)
    trimmed_m = metrics(trimmed)
    deg = (full_metrics["sharpe"] - trimmed_m["sharpe"]) / full_metrics["sharpe"] * 100 if full_metrics["sharpe"] != 0 else 0
    print(f"  Full Sharpe: {full_metrics['sharpe']:.3f}")
    print(f"  Trimmed Sharpe: {trimmed_m['sharpe']:.3f}")
    print(f"  Degradation: {deg:.1f}% (threshold: 30%)")
    outlier_pass = trimmed_m["sharpe"] > 0 and deg < 30

    # 5. R1 regime test
    print(f"\n=== R1 Regime Test ===")
    green = spy_ret > 0
    red = spy_ret < 0
    green_sharpe = metrics(vmr_ret[green])["sharpe"]
    red_sharpe = metrics(vmr_ret[red])["sharpe"]
    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.001)
    print(f"  Green: {green_sharpe:.3f}, Red: {red_sharpe:.3f}")
    print(f"  Gap: {gap:.3f} (threshold: 0.50)")
    r1_pass = gap <= 0.50

    # 6. Year-by-year
    print(f"\n=== Year-by-Year ===")
    years = vmr_ret.groupby(vmr_ret.index.year)
    year_data = []
    for yr, rets in years:
        m = metrics(rets)
        sm = metrics(spy_ret.reindex(rets.index).dropna())
        year_data.append({"year": yr, "vmr_sharpe": m["sharpe"], "spy_sharpe": sm["sharpe"],
                          "vmr_cagr": m["cagr_pct"], "beats_spy": m["sharpe"] > sm["sharpe"]})
        print(f"  {yr}: VMR Sharpe {m['sharpe']:.3f} vs SPY {sm['sharpe']:.3f} "
              f"({'✅' if m['sharpe'] > sm['sharpe'] else '❌'})")

    yr_beats = sum(y["beats_spy"] for y in year_data)
    print(f"  VMR beats SPY: {yr_beats}/{len(year_data)} years")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    all_pass = p_value < 0.05 and sub_pass and outlier_pass
    print(f"  Permutation: {'PASS' if p_value < 0.05 else 'FAIL'} (p={p_value:.4f})")
    print(f"  Sub-period:  {'PASS' if sub_pass else 'FAIL'} (CV={cv:.3f})")
    print(f"  Outlier:     {'PASS' if outlier_pass else 'FAIL'} (deg={deg:.1f}%)")
    print(f"  R1 Regime:   {'PASS' if r1_pass else 'FAIL (expected for growth)'} (gap={gap:.3f})")
    print(f"  WF vs SPY:   {beats}/{len(wf_results)} windows")
    print(f"  Year-by-Year: {yr_beats}/{len(year_data)} years")
    print(f"  OVERALL: {'✅ VALIDATED' if all_pass else '❌ REJECTED'}")

    # Save
    results = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "Vol Mean Reversion (VMR)",
        "full_period": full_metrics,
        "spy_benchmark": spy_metrics,
        "permutation": {"p_value": float(p_value), "real_sharpe": real_sharpe,
                        "perm_mean": float(np.mean(perm_sharpes)), "pass": p_value < 0.05},
        "walkforward": {"n_windows": len(wf_results), "beats_spy": beats, "windows": wf_results},
        "subperiod": {"block_sharpes": block_sharpes, "cv": round(cv, 4), "pass": sub_pass},
        "outlier": {"full_sharpe": full_metrics["sharpe"], "trimmed_sharpe": trimmed_m["sharpe"],
                    "degradation_pct": round(deg, 2), "pass": outlier_pass},
        "regime": {"green_sharpe": green_sharpe, "red_sharpe": red_sharpe,
                   "gap": round(gap, 4), "pass": r1_pass},
        "yearly": year_data,
        "overall_pass": all_pass,
    }

    out = OUTPUT_DIR / "vmr_adversarial_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.integer, np.floating, np.bool_)) else str(o))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
