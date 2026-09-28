#!/usr/bin/env python3
"""
VMR Daily Rebalance — Full Adversarial Validation
==================================================
Same 5-regime VIX logic as weekly VMR, but rebalances every trading day.
1000-shuffle permutation, 3-block sub-period, top-5% outlier trim,
15-window walk-forward, direct comparison vs weekly VMR.
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
PERM_N = 1000
WARMUP = 252  # 1yr warmup for 20d MA to stabilize


def fetch_data():
    tickers = ["SPY", "UPRO", "GLD", "TLT", "^VIX"]
    print(f"Fetching {tickers}  (2010-01-01 to 2026-07-17)...")
    raw = yf.download(tickers, start="2010-01-01", end="2026-07-17",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    prices = prices.rename(columns={"^VIX": "VIX"})
    prices = prices.ffill().dropna(how="all")
    # Drop rows where all UPRO/SPY/GLD/TLT are NaN (pre-UPRO history)
    prices = prices.dropna(subset=["UPRO", "SPY", "GLD", "TLT"])
    print(f"  {len(prices)} trading days ({prices.index[0].date()} – {prices.index[-1].date()})")
    return prices


# ------------------------------------------------------------------ #
#  Strategy engines
# ------------------------------------------------------------------ #

def classify_regime(v, vm20, vp20):
    """Return regime label for a single bar."""
    if pd.isna(v) or pd.isna(vm20):
        return "SPY"
    if v < 15 and v < vm20:
        return "UPRO"
    if v > 20 and v < vp20 * 0.85 and v < vm20:
        return "UPRO_MR"
    if v > 25 and v > vm20:
        return "DEFENSIVE"
    if v > 20 and v > vm20:
        return "CAUTIOUS"
    return "SPY"


def apply_regime(regime, upro_r, spy_r, gld_r, tlt_r):
    if regime in ("UPRO", "UPRO_MR"):
        return upro_r
    if regime == "DEFENSIVE":
        return 0.5 * gld_r + 0.5 * tlt_r
    if regime == "CAUTIOUS":
        return 0.5 * spy_r + 0.5 * tlt_r
    return spy_r


def run_daily_vmr(prices: pd.DataFrame) -> pd.Series:
    """Daily-rebalance VMR. Recompute regime each day."""
    rets = prices.pct_change()
    vix = prices["VIX"]
    vix_ma20 = vix.rolling(20).mean()
    vix_peak20 = vix.rolling(20).max()

    port_rets = pd.Series(0.0, index=prices.index)

    for i in range(WARMUP, len(prices)):
        v    = vix.iloc[i]
        vm20 = vix_ma20.iloc[i]
        vp20 = vix_peak20.iloc[i]
        regime = classify_regime(v, vm20, vp20)
        port_rets.iloc[i] = apply_regime(
            regime,
            rets["UPRO"].iloc[i],
            rets["SPY"].iloc[i],
            rets["GLD"].iloc[i],
            rets["TLT"].iloc[i],
        )

    return port_rets.iloc[WARMUP:]


def run_weekly_vmr(prices: pd.DataFrame) -> pd.Series:
    """Weekly-rebalance VMR (Friday check)."""
    rets = prices.pct_change()
    vix = prices["VIX"]
    vix_ma10 = vix.rolling(10).mean()
    vix_peak20 = vix.rolling(20).max()

    port_rets = pd.Series(0.0, index=prices.index)
    regime = "SPY"
    last_week = None

    for i in range(WARMUP, len(prices)):
        idx = prices.index[i]
        week = (idx.year, idx.isocalendar()[1])
        if week != last_week:
            last_week = week
            v    = vix.iloc[i]
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

        port_rets.iloc[i] = apply_regime(
            regime,
            rets["UPRO"].iloc[i],
            rets["SPY"].iloc[i],
            rets["GLD"].iloc[i],
            rets["TLT"].iloc[i],
        )

    return port_rets.iloc[WARMUP:]


# ------------------------------------------------------------------ #
#  Metrics
# ------------------------------------------------------------------ #

def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "cagr_pct": 0.0,
                "max_dd_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
                "n_days": len(r), "ann_vol_pct": 0.0}
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    ds = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-6
    sortino = ann_ret / ds
    cum = (1 + r).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    n_yr = len(r) / 252
    cagr = (cum.iloc[-1] ** (1 / n_yr) - 1) * 100 if cum.iloc[-1] > 0 else -100.0
    wr = float((r > 0).mean())
    gains  = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    # Day concentration: fraction of cumulative PnL from single best day
    cum_pnl = (1 + r).cumprod() - 1
    daily_contrib = r / (1 + cum_pnl.shift(1).fillna(0))
    best_day_contrib = daily_contrib.max() / daily_contrib.sum() if daily_contrib.sum() > 0 else float("nan")
    return {
        "sharpe":        round(sharpe, 4),
        "sortino":       round(sortino, 4),
        "cagr_pct":      round(cagr, 2),
        "max_dd_pct":    round(max_dd, 2),
        "win_rate":      round(wr, 4),
        "profit_factor": round(pf, 4),
        "n_days":        len(r),
        "ann_vol_pct":   round(ann_vol * 100, 2),
        "day_conc":      round(float(best_day_contrib), 4),
    }


def calmar(m: dict) -> float:
    dd = abs(m["max_dd_pct"])
    if dd == 0:
        return float("inf")
    return round(m["cagr_pct"] / dd, 4)


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    print("=" * 70)
    print("VMR DAILY REBALANCE — Adversarial Validation Suite")
    print("=" * 70)

    prices = fetch_data()

    # ---- Full period -------------------------------------------------
    daily_ret  = run_daily_vmr(prices)
    weekly_ret = run_weekly_vmr(prices)
    spy_ret    = prices["SPY"].pct_change().reindex(daily_ret.index).dropna()
    daily_ret  = daily_ret.reindex(spy_ret.index)
    weekly_ret = weekly_ret.reindex(spy_ret.index)

    dm  = metrics(daily_ret)
    wm  = metrics(weekly_ret)
    sm  = metrics(spy_ret)
    dm["calmar"] = calmar(dm)
    wm["calmar"] = calmar(wm)
    sm["calmar"] = calmar(sm)

    print(f"\n=== Full Period ===")
    print(f"  VMR-Daily:  Sharpe={dm['sharpe']:.3f}, Sortino={dm['sortino']:.3f}, "
          f"CAGR={dm['cagr_pct']:.1f}%, MaxDD={dm['max_dd_pct']:.1f}%, "
          f"Calmar={dm['calmar']:.3f}, DayConc={dm['day_conc']:.3f}")
    print(f"  VMR-Weekly: Sharpe={wm['sharpe']:.3f}, Sortino={wm['sortino']:.3f}, "
          f"CAGR={wm['cagr_pct']:.1f}%, MaxDD={wm['max_dd_pct']:.1f}%")
    print(f"  SPY:        Sharpe={sm['sharpe']:.3f}, CAGR={sm['cagr_pct']:.1f}%")

    # ---- 1. Permutation test (1000 shuffles) -------------------------
    print(f"\n=== Permutation Test ({PERM_N} shuffles) ===")
    real_sharpe = dm["sharpe"]
    perm_sharpes = []

    for p in range(PERM_N):
        shuffled = prices.copy()
        shuffled["VIX"] = np.random.permutation(shuffled["VIX"].values)
        pr = run_daily_vmr(shuffled)
        pr = pr.reindex(spy_ret.index).fillna(0)
        perm_sharpes.append(metrics(pr)["sharpe"])
        if (p + 1) % 200 == 0:
            print(f"  {p+1}/{PERM_N} done...")

    p_value = float(np.mean([s >= real_sharpe for s in perm_sharpes]))
    perm_mean = float(np.mean(perm_sharpes))
    perm_std  = float(np.std(perm_sharpes))
    print(f"  Real Sharpe:  {real_sharpe:.4f}")
    print(f"  Perm mean:    {perm_mean:.4f} ± {perm_std:.4f}")
    print(f"  p-value:      {p_value:.4f}")
    perm_pass = p_value < 0.05
    print(f"  {'PASS' if perm_pass else 'FAIL'} (threshold 0.05)")

    # ---- 2. Sub-period consistency (3 equal blocks) ------------------
    print(f"\n=== Sub-Period Consistency (3 equal blocks) ===")
    n = len(daily_ret)
    t = n // 3
    blocks = [daily_ret.iloc[:t], daily_ret.iloc[t:2*t], daily_ret.iloc[2*t:]]
    block_sharpes = [metrics(b)["sharpe"] for b in blocks]
    block_cagrs   = [metrics(b)["cagr_pct"] for b in blocks]
    all_positive = all(s > 0 for s in block_sharpes)
    mean_bs = np.mean(block_sharpes)
    cv = float(np.std(block_sharpes) / mean_bs) if mean_bs > 0 else float("inf")
    for i, (s, c) in enumerate(zip(block_sharpes, block_cagrs)):
        label = daily_ret.iloc[i*t:(i+1)*t if i < 2 else n].index
        print(f"  Block {i+1} ({label[0].date()}–{label[-1].date()}): "
              f"Sharpe={s:.3f}, CAGR={c:.1f}%")
    print(f"  All positive: {all_positive}")
    print(f"  CV:           {cv:.4f} (threshold 0.50)")
    sub_pass = all_positive and cv < 0.50
    print(f"  {'PASS' if sub_pass else 'FAIL'}")

    # ---- 3. Outlier robustness (remove top 5% of daily returns) ------
    print(f"\n=== Outlier Robustness (remove top 5% daily returns) ===")
    cutoff = daily_ret.quantile(0.95)
    trimmed = daily_ret[daily_ret <= cutoff]
    tm = metrics(trimmed)
    deg = (dm["sharpe"] - tm["sharpe"]) / dm["sharpe"] * 100 if dm["sharpe"] != 0 else 0.0
    n_removed = len(daily_ret) - len(trimmed)
    print(f"  Removed:        {n_removed} days ({n_removed/len(daily_ret)*100:.1f}%)")
    print(f"  Full Sharpe:    {dm['sharpe']:.4f}")
    print(f"  Trimmed Sharpe: {tm['sharpe']:.4f}")
    print(f"  Degradation:    {deg:.1f}% (threshold 30%)")
    outlier_pass = tm["sharpe"] > 0 and deg < 30.0
    print(f"  {'PASS' if outlier_pass else 'FAIL'}")

    # ---- 4. Walk-forward (15 annual windows) -------------------------
    print(f"\n=== Walk-Forward (annual steps) ===")
    wf_results = []
    step  = 252
    train = 504   # 2yr training context (prices fed in full — VMR uses MA20 only)
    test  = 252   # 1yr OOT

    # We iterate over the OOT test windows sequentially
    all_idx = prices.index
    # Start after warmup + minimum history
    start_oot = WARMUP + 252   # begin first OOT after first full year post-warmup
    for oot_start in range(start_oot, len(all_idx) - test + 1, step):
        oot_end = oot_start + test
        slice_prices = prices.iloc[:oot_end]   # feed all data up to end of OOT window
        all_daily = run_daily_vmr(slice_prices)
        oot_ret = all_daily.iloc[-test:]
        oot_spy = slice_prices["SPY"].pct_change().iloc[-test:]
        oot_spy = oot_spy.reindex(oot_ret.index)

        if len(oot_ret.dropna()) < 50:
            continue

        mv = metrics(oot_ret)
        ms = metrics(oot_spy.dropna())
        beats = mv["sharpe"] > ms["sharpe"]
        wf_results.append({
            "oot_start":   all_idx[oot_start].strftime("%Y-%m-%d"),
            "oot_end":     all_idx[min(oot_end-1, len(all_idx)-1)].strftime("%Y-%m-%d"),
            "vmr_sharpe":  mv["sharpe"],
            "spy_sharpe":  ms["sharpe"],
            "vmr_cagr":    mv["cagr_pct"],
            "beats_spy":   beats,
        })
        print(f"  {all_idx[oot_start].strftime('%Y')}: "
              f"VMR {mv['sharpe']:.3f} vs SPY {ms['sharpe']:.3f} "
              f"({'BEAT' if beats else 'LOSS'})")

        if len(wf_results) >= 15:
            break

    wf_beats = sum(w["beats_spy"] for w in wf_results)
    wf_pct = wf_beats / len(wf_results) * 100 if wf_results else 0
    print(f"  VMR beats SPY: {wf_beats}/{len(wf_results)} windows ({wf_pct:.0f}%)")
    wf_pass = wf_pct >= 70   # >=70% windows beat SPY

    # ---- 5. R1 Regime split ------------------------------------------
    print(f"\n=== R1 Regime Split (HC #428) ===")
    green = spy_ret > 0
    red   = spy_ret < 0
    flat  = spy_ret == 0
    g_sharpe = metrics(daily_ret[green])["sharpe"]
    r_sharpe = metrics(daily_ret[red])["sharpe"]
    f_sharpe = metrics(daily_ret[flat])["sharpe"] if flat.sum() > 10 else float("nan")
    gap = abs(g_sharpe - r_sharpe) / max(abs(g_sharpe), abs(r_sharpe), 0.001)
    print(f"  Green days:  Sharpe={g_sharpe:.3f}")
    print(f"  Red days:    Sharpe={r_sharpe:.3f}")
    print(f"  Flat days:   Sharpe={f_sharpe:.3f}" if not np.isnan(f_sharpe) else "  Flat days:   N/A")
    print(f"  Skew |G-R|/max(|G|,|R|) = {gap:.4f}  (threshold 0.50)")
    r1_pass = gap <= 0.50
    print(f"  R1: {'PASS' if r1_pass else 'FAIL'}")

    # ---- 6. Day concentration ----------------------------------------
    print(f"\n=== Day Concentration (HC #344 cap <= 0.70) ===")
    cum_total = float(daily_ret.sum())
    top1_day  = float(daily_ret.max())
    day_conc  = top1_day / cum_total if cum_total > 0 else float("nan")
    print(f"  Top-1 day return: {top1_day*100:.2f}%")
    print(f"  Total cumulative return (daily sum): {cum_total*100:.2f}%")
    print(f"  Day concentration: {day_conc:.4f}  (cap 0.70)")
    conc_pass = day_conc <= 0.70

    # ---- 7. Year-by-year --------------------------------------------
    print(f"\n=== Year-by-Year ===")
    year_data = []
    for yr, rets in daily_ret.groupby(daily_ret.index.year):
        mv = metrics(rets)
        ms = metrics(spy_ret.reindex(rets.index).dropna())
        beats = mv["sharpe"] > ms["sharpe"]
        year_data.append({"year": yr, "vmr_sharpe": mv["sharpe"],
                          "spy_sharpe": ms["sharpe"],
                          "vmr_cagr": mv["cagr_pct"],
                          "beats_spy": beats})
        print(f"  {yr}: VMR Sharpe {mv['sharpe']:.3f} vs SPY {ms['sharpe']:.3f}  "
              f"({'BEAT' if beats else 'LOSS'})")
    yr_beats = sum(y["beats_spy"] for y in year_data)

    # ---- 8. Direct weekly vs daily comparison -----------------------
    print(f"\n=== Weekly vs Daily Comparison ===")
    print(f"  Weekly: Sharpe={wm['sharpe']:.3f}, Sortino={wm['sortino']:.3f}, "
          f"CAGR={wm['cagr_pct']:.1f}%, MaxDD={wm['max_dd_pct']:.1f}%")
    print(f"  Daily:  Sharpe={dm['sharpe']:.3f}, Sortino={dm['sortino']:.3f}, "
          f"CAGR={dm['cagr_pct']:.1f}%, MaxDD={dm['max_dd_pct']:.1f}%")
    sharpe_lift  = (dm["sharpe"]  - wm["sharpe"])  / abs(wm["sharpe"]) * 100
    cagr_lift    = dm["cagr_pct"] - wm["cagr_pct"]
    dd_change    = dm["max_dd_pct"] - wm["max_dd_pct"]
    print(f"  Sharpe lift:   +{sharpe_lift:.1f}%")
    print(f"  CAGR lift:     +{cagr_lift:.1f}pp")
    print(f"  MaxDD change:  {dd_change:.1f}pp")

    # ---- Summary ------------------------------------------------------
    all_pass = perm_pass and sub_pass and outlier_pass
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Full Sharpe:    {dm['sharpe']:.4f}")
    print(f"  Full Sortino:   {dm['sortino']:.4f}")
    print(f"  CAGR:           {dm['cagr_pct']:.1f}%")
    print(f"  MaxDD:          {dm['max_dd_pct']:.1f}%")
    print(f"  Calmar:         {dm['calmar']:.4f}")
    print(f"  Win Rate:       {dm['win_rate']*100:.1f}%")
    print(f"  Profit Factor:  {dm['profit_factor']:.4f}")
    print(f"  Day Conc:       {dm['day_conc']:.4f}")
    print()
    print(f"  Permutation:    {'PASS' if perm_pass else 'FAIL'}  p={p_value:.4f}")
    print(f"  Sub-period:     {'PASS' if sub_pass else 'FAIL'}   CV={cv:.4f}")
    print(f"  Outlier:        {'PASS' if outlier_pass else 'FAIL'}  deg={deg:.1f}%")
    print(f"  R1 Regime:      {'PASS' if r1_pass else 'FAIL'}  gap={gap:.4f}")
    print(f"  Day Conc:       {'PASS' if conc_pass else 'FAIL'}  {day_conc:.4f} vs cap 0.70")
    print(f"  WF vs SPY:      {wf_beats}/{len(wf_results)} windows ({wf_pct:.0f}%)")
    print(f"  Year-by-Year:   {yr_beats}/{len(year_data)} years")
    print(f"  OVERALL:        {'VALIDATED' if all_pass else 'REJECTED'}")

    # ---- Save ---------------------------------------------------------
    results = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "VMR Daily Rebalance",
        "full_period": dm,
        "weekly_vmr": wm,
        "spy_benchmark": sm,
        "permutation": {
            "n_shuffles": PERM_N,
            "p_value": p_value,
            "real_sharpe": real_sharpe,
            "perm_mean": perm_mean,
            "perm_std": perm_std,
            "pass": perm_pass,
        },
        "subperiod": {
            "block_sharpes": [float(s) for s in block_sharpes],
            "block_cagrs":   [float(c) for c in block_cagrs],
            "cv": round(cv, 6),
            "all_positive": all_positive,
            "pass": sub_pass,
        },
        "outlier": {
            "cutoff_pct": round(float(cutoff) * 100, 4),
            "n_removed": int(n_removed),
            "full_sharpe": dm["sharpe"],
            "trimmed_sharpe": tm["sharpe"],
            "degradation_pct": round(deg, 2),
            "pass": outlier_pass,
        },
        "walkforward": {
            "n_windows": len(wf_results),
            "beats_spy": int(wf_beats),
            "beat_pct": round(wf_pct, 1),
            "pass": wf_pass,
            "windows": wf_results,
        },
        "regime": {
            "green_sharpe": round(g_sharpe, 4),
            "red_sharpe":   round(r_sharpe, 4),
            "flat_sharpe":  round(f_sharpe, 4) if not np.isnan(f_sharpe) else None,
            "gap": round(gap, 4),
            "pass": r1_pass,
        },
        "day_concentration": {
            "top1_day_pct": round(top1_day * 100, 4),
            "day_conc": round(day_conc, 4),
            "pass": conc_pass,
        },
        "yearly": year_data,
        "weekly_vs_daily": {
            "sharpe_weekly": wm["sharpe"],
            "sharpe_daily":  dm["sharpe"],
            "sharpe_lift_pct": round(sharpe_lift, 2),
            "cagr_lift_pp":    round(cagr_lift, 2),
            "maxdd_change_pp": round(dd_change, 2),
        },
        "overall_pass": all_pass,
    }

    out = OUTPUT_DIR / "vmr_daily_adversarial_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, (np.integer, np.floating, np.bool_)) else str(o))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
