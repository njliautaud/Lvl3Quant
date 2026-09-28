#!/usr/bin/env python3
"""
v5_vix_ts_sizing.py — VIX Term Structure Position Sizing Overlay for Wheel V5

Applies a CONTINUOUS VIX term structure sizing signal to V5 CSP daily returns:
  - Contango (VIX3M/VIX ratio > 1.10): full size (100%)
  - Mild contango (1.00-1.10): 75%
  - Flat/mild backwardation (0.85-1.00): 50%
  - Deep backwardation (< 0.85): 25% (crisis mode)

Key insight: VIX term structure is a LEADING indicator of regime change,
better than VIX level alone. This avoids the blunt "VIX > 20 = cut" approach
that degraded Sharpe from 1.70 to 1.45.

Evaluates R1 regime gate: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50

Usage:
    python3 /home/jupiter/Lvl3Quant/research/v5_vix_ts_sizing.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "v5_vix_ts_sizing"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE = 0.04
STARTING_CAPITAL = 100_000.0


# ── Sizing schedule (continuous, interpolated) ──────────────────────────────
def vix_ts_sizing(ratio: float) -> float:
    """
    Map VIX term structure ratio to position size multiplier.
    Ratio = VIX3M / VIX (or equivalent lag1 feature).
    >1 = contango (calm), <1 = backwardation (fear).

    Uses linear interpolation between breakpoints for smooth transitions.
    """
    breakpoints = [
        (0.75, 0.25),   # deep backwardation → 25%
        (0.85, 0.25),   # backwardation → 25%
        (0.95, 0.50),   # mild backwardation → 50%
        (1.00, 0.75),   # flat → 75%
        (1.10, 1.00),   # contango → 100%
        (1.30, 1.00),   # deep contango → 100% (no leverage)
    ]
    if np.isnan(ratio):
        return 1.0  # default to full size if no data
    if ratio <= breakpoints[0][0]:
        return breakpoints[0][1]
    if ratio >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for i in range(len(breakpoints) - 1):
        r0, s0 = breakpoints[i]
        r1, s1 = breakpoints[i + 1]
        if r0 <= ratio <= r1:
            frac = (ratio - r0) / (r1 - r0)
            return s0 + frac * (s1 - s0)
    return 1.0


def compute_metrics(daily_returns: pd.Series, label: str = "") -> dict:
    """Compute risk-adjusted metrics from daily return series."""
    n = len(daily_returns)
    if n < 30:
        return {"label": label, "error": "insufficient data"}

    ann_ret = daily_returns.mean() * TRADING_DAYS
    ann_vol = daily_returns.std() * np.sqrt(TRADING_DAYS)

    # Sharpe
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside deviation)
    downside = daily_returns[daily_returns < 0]
    dd_vol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else ann_vol
    sortino = (ann_ret - RISK_FREE) / dd_vol if dd_vol > 0 else 0.0

    # Cumulative equity for MaxDD and CAGR
    equity = (1 + daily_returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # CAGR
    years = n / TRADING_DAYS
    total_return = equity.iloc[-1]
    cagr = total_return ** (1 / years) - 1 if years > 0 else 0.0

    # Profit factor
    wins = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    pf = wins / losses if losses > 0 else float("inf")

    # Win rate
    wr = (daily_returns > 0).mean()

    return {
        "label": label,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "pf": pf,
        "wr": wr,
        "n_days": n,
        "total_return": total_return,
    }


def regime_gap(daily_returns: pd.Series, spy_returns: pd.Series) -> dict:
    """
    R1 regime gate: compute per-regime Sharpe and regime gap.
    Green day = SPY close-to-close > 0, Red day = SPY close-to-close < 0.
    """
    # Align
    common = daily_returns.index.intersection(spy_returns.index)
    dr = daily_returns.loc[common]
    sr = spy_returns.loc[common]

    green_mask = sr > 0
    red_mask = sr < 0
    flat_mask = sr == 0

    green_ret = dr[green_mask]
    red_ret = dr[red_mask]

    def _sharpe(r):
        if len(r) < 10:
            return 0.0
        ann = r.mean() * TRADING_DAYS
        vol = r.std() * np.sqrt(TRADING_DAYS)
        return (ann - RISK_FREE) / vol if vol > 0 else 0.0

    s_green = _sharpe(green_ret)
    s_red = _sharpe(red_ret)

    denom = max(abs(s_green), abs(s_red))
    gap = abs(s_green - s_red) / denom if denom > 0 else 0.0

    return {
        "sharpe_green": s_green,
        "sharpe_red": s_red,
        "regime_gap": gap,
        "r1_pass": gap <= 0.50,
        "n_green": int(green_mask.sum()),
        "n_red": int(red_mask.sum()),
        "n_flat": int(flat_mask.sum()),
    }


def main():
    print("=" * 70)
    print("VIX TERM STRUCTURE POSITION SIZING OVERLAY — WHEEL V5 CSP")
    print("=" * 70)

    # ── 1. Load V5 equity curve and compute daily returns ──
    eq_path = ROOT / "output" / "wheel_v5_research" / "equity_v5_combined.parquet"
    if not eq_path.exists():
        print(f"ERROR: V5 equity curve not found at {eq_path}")
        return

    eq = pd.read_parquet(eq_path)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date").sort_index()
    eq["daily_return"] = eq["equity"].pct_change()
    eq = eq.dropna(subset=["daily_return"])

    print(f"V5 equity curve: {eq.index[0].date()} to {eq.index[-1].date()} ({len(eq)} days)")

    # ── 2. Load regime features (VIX term structure) ──
    rf = pd.read_parquet(ROOT / "data" / "feature_store" / "v2" / "regime_features.parquet")
    rf["date"] = pd.to_datetime(rf["date"])
    rf = rf.set_index("date").sort_index()

    # ── 3. Load SPY for regime classification ──
    spy = pd.read_parquet(ROOT / "data" / "spy_daily.parquet")
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.set_index("date").sort_index()
    spy["spy_return"] = spy["close"].pct_change()
    spy = spy.dropna(subset=["spy_return"])

    # ── 4. Merge VIX TS signal into equity frame ──
    merged = eq[["daily_return"]].join(rf[["regime_vix_ts_lag1"]], how="left")
    # Forward-fill VIX TS for any missing days (weekends/holidays alignment)
    merged["regime_vix_ts_lag1"] = merged["regime_vix_ts_lag1"].ffill()

    valid_ts = merged["regime_vix_ts_lag1"].notna()
    print(f"VIX TS coverage: {valid_ts.sum()}/{len(merged)} days ({valid_ts.mean()*100:.1f}%)")

    # ── 5. Compute sizing multiplier ──
    merged["size_mult"] = merged["regime_vix_ts_lag1"].apply(vix_ts_sizing)
    merged["sized_return"] = merged["daily_return"] * merged["size_mult"]

    # ── 6. Distribution of sizing ──
    print(f"\nSizing Distribution:")
    bins = [0, 0.30, 0.55, 0.80, 1.01]
    labels = ["25% (crisis)", "50% (backwardation)", "75% (flat)", "100% (contango)"]
    merged["size_bin"] = pd.cut(merged["size_mult"], bins=bins, labels=labels, include_lowest=True)
    dist = merged["size_bin"].value_counts().sort_index()
    for lbl, cnt in dist.items():
        pct = cnt / len(merged) * 100
        print(f"  {lbl}: {cnt:>5d} days ({pct:>5.1f}%)")
    print(f"  Average position size: {merged['size_mult'].mean():.1%}")

    # ── 7. Compute metrics for both ──
    baseline_metrics = compute_metrics(merged["daily_return"], "V5 Baseline")
    sized_metrics = compute_metrics(merged["sized_return"], "V5 + VIX TS Sizing")

    # ── 8. Regime analysis ──
    spy_ret = spy["spy_return"]
    baseline_regime = regime_gap(merged["daily_return"], spy_ret)
    sized_regime = regime_gap(merged["sized_return"], spy_ret)

    # ── 9. Print results ──
    print(f"\n{'='*80}")
    print(f"{'METRIC':<25} {'V5 BASELINE':>18} {'V5 + VIX TS':>18} {'DELTA':>12}")
    print(f"{'='*80}")

    for key, fmt in [
        ("cagr", "{:.1%}"),
        ("sharpe", "{:.2f}"),
        ("sortino", "{:.2f}"),
        ("max_dd", "{:.1%}"),
        ("ann_ret", "{:.1%}"),
        ("ann_vol", "{:.1%}"),
        ("pf", "{:.2f}"),
        ("wr", "{:.1%}"),
    ]:
        bv = baseline_metrics[key]
        sv = sized_metrics[key]
        delta = sv - bv
        b_str = fmt.format(bv)
        s_str = fmt.format(sv)
        d_str = fmt.format(delta) if key != "max_dd" else fmt.format(delta)
        print(f"  {key.upper():<23} {b_str:>18} {s_str:>18} {d_str:>12}")

    print(f"\n{'='*80}")
    print(f"R1 REGIME GATE (threshold: gap <= 0.50)")
    print(f"{'='*80}")
    print(f"  {'METRIC':<25} {'V5 BASELINE':>18} {'V5 + VIX TS':>18}")
    print(f"  {'Sharpe (green days)':<25} {baseline_regime['sharpe_green']:>18.2f} {sized_regime['sharpe_green']:>18.2f}")
    print(f"  {'Sharpe (red days)':<25} {baseline_regime['sharpe_red']:>18.2f} {sized_regime['sharpe_red']:>18.2f}")
    print(f"  {'Regime gap':<25} {baseline_regime['regime_gap']:>18.2f} {sized_regime['regime_gap']:>18.2f}")
    print(f"  {'R1 PASS?':<25} {'YES' if baseline_regime['r1_pass'] else 'NO':>18} {'YES' if sized_regime['r1_pass'] else 'NO':>18}")
    print(f"  Green days: {baseline_regime['n_green']}, Red days: {baseline_regime['n_red']}")

    # ── 10. Year-by-year breakdown ──
    print(f"\n{'='*80}")
    print(f"YEAR-BY-YEAR BREAKDOWN")
    print(f"{'='*80}")
    print(f"  {'YEAR':<8} {'BL Sharpe':>10} {'TS Sharpe':>10} {'BL CAGR':>10} {'TS CAGR':>10} {'Avg Size':>10}")
    print(f"  {'-'*58}")

    merged["year"] = merged.index.year
    for year in sorted(merged["year"].unique()):
        yr_data = merged[merged["year"] == year]
        if len(yr_data) < 20:
            continue
        bm = compute_metrics(yr_data["daily_return"], f"BL {year}")
        sm = compute_metrics(yr_data["sized_return"], f"TS {year}")
        avg_sz = yr_data["size_mult"].mean()
        print(f"  {year:<8} {bm['sharpe']:>10.2f} {sm['sharpe']:>10.2f} "
              f"{bm['cagr']:>9.1%} {sm['cagr']:>9.1%} {avg_sz:>9.1%}")

    # ── 11. Stress period analysis ──
    print(f"\n{'='*80}")
    print(f"STRESS PERIOD ANALYSIS (VIX TS < 1.0 = backwardation)")
    print(f"{'='*80}")

    backwardation = merged[merged["regime_vix_ts_lag1"] < 1.0]
    contango = merged[merged["regime_vix_ts_lag1"] >= 1.0]

    if len(backwardation) > 10:
        bw_bl = compute_metrics(backwardation["daily_return"], "BW baseline")
        bw_ts = compute_metrics(backwardation["sized_return"], "BW sized")
        print(f"  Backwardation ({len(backwardation)} days):")
        print(f"    Baseline: Sharpe={bw_bl['sharpe']:.2f}, Ann.Ret={bw_bl['ann_ret']:.1%}, MaxDD={bw_bl['max_dd']:.1%}")
        print(f"    VIX TS:   Sharpe={bw_ts['sharpe']:.2f}, Ann.Ret={bw_ts['ann_ret']:.1%}, MaxDD={bw_ts['max_dd']:.1%}")
        print(f"    Avg size during backwardation: {backwardation['size_mult'].mean():.1%}")

    if len(contango) > 10:
        ct_bl = compute_metrics(contango["daily_return"], "CT baseline")
        ct_ts = compute_metrics(contango["sized_return"], "CT sized")
        print(f"  Contango ({len(contango)} days):")
        print(f"    Baseline: Sharpe={ct_bl['sharpe']:.2f}, Ann.Ret={ct_bl['ann_ret']:.1%}")
        print(f"    VIX TS:   Sharpe={ct_ts['sharpe']:.2f}, Ann.Ret={ct_ts['ann_ret']:.1%}")
        print(f"    Avg size during contango: {contango['size_mult'].mean():.1%}")

    # ── 12. Sensitivity sweep on breakpoints ──
    print(f"\n{'='*80}")
    print(f"SENSITIVITY SWEEP — Varying sizing aggressiveness")
    print(f"{'='*80}")
    print(f"  {'Config':<30} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Gap':>8} {'R1?':>5}")
    print(f"  {'-'*75}")

    configs = {
        "Baseline (no sizing)": lambda r: 1.0,
        "Conservative (25-50-75-100)": vix_ts_sizing,  # current
        "Moderate (50-65-85-100)": lambda r: np.interp(
            r if not np.isnan(r) else 1.1,
            [0.75, 0.85, 0.95, 1.00, 1.10, 1.30],
            [0.50, 0.50, 0.65, 0.85, 1.00, 1.00],
        ),
        "Aggressive (10-30-60-100)": lambda r: np.interp(
            r if not np.isnan(r) else 1.1,
            [0.75, 0.85, 0.95, 1.00, 1.10, 1.30],
            [0.10, 0.10, 0.30, 0.60, 1.00, 1.00],
        ),
        "Light (50-75-90-100)": lambda r: np.interp(
            r if not np.isnan(r) else 1.1,
            [0.75, 0.85, 0.95, 1.00, 1.10, 1.30],
            [0.50, 0.50, 0.75, 0.90, 1.00, 1.00],
        ),
        "Blunt VIX>20 cut 50%": lambda r: 1.0,  # placeholder, handled separately
    }

    for name, sizing_fn in configs.items():
        if name == "Blunt VIX>20 cut 50%":
            # Use VIX level instead of term structure
            vix_col = rf["regime_vix_lag1"] if "regime_vix_lag1" in rf.columns else None
            if vix_col is not None:
                vix_merged = eq[["daily_return"]].join(rf[["regime_vix_lag1"]], how="left")
                vix_merged["regime_vix_lag1"] = vix_merged["regime_vix_lag1"].ffill()
                vix_merged["size"] = vix_merged["regime_vix_lag1"].apply(
                    lambda v: 0.5 if (not np.isnan(v) and v > 20) else 1.0
                )
                ret = vix_merged["daily_return"] * vix_merged["size"]
            else:
                continue
        else:
            ret = merged["daily_return"] * merged["regime_vix_ts_lag1"].apply(sizing_fn)

        m = compute_metrics(ret, name)
        rg = regime_gap(ret, spy_ret)
        r1 = "YES" if rg["r1_pass"] else "NO"
        print(f"  {name:<30} {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
              f"{m['cagr']:>7.1%} {m['max_dd']:>7.1%} {rg['regime_gap']:>8.2f} {r1:>5}")

    # ── 13. Save results ──
    results = {
        "baseline": {**baseline_metrics, **{f"regime_{k}": v for k, v in baseline_regime.items()}},
        "vix_ts_sized": {**sized_metrics, **{f"regime_{k}": v for k, v in sized_regime.items()}},
        "sizing_distribution": {str(k): int(v) for k, v in dist.items()},
        "avg_position_size": float(merged["size_mult"].mean()),
    }

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    clean = json.loads(json.dumps(results, default=convert))
    with open(OUTPUT / "results.json", "w") as f:
        json.dump(clean, f, indent=2)

    # Save sized equity curve
    sized_eq = (1 + merged["sized_return"]).cumprod() * STARTING_CAPITAL
    sized_eq.name = "equity_vix_ts"
    baseline_eq = (1 + merged["daily_return"]).cumprod() * STARTING_CAPITAL
    baseline_eq.name = "equity_baseline"
    out_df = pd.DataFrame({"baseline": baseline_eq, "vix_ts_sized": sized_eq})
    out_df.to_parquet(OUTPUT / "equity_curves.parquet")

    print(f"\nResults saved to {OUTPUT}/")

    # ── SUMMARY ──
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    gap_delta = sized_regime["regime_gap"] - baseline_regime["regime_gap"]
    sharpe_delta = sized_metrics["sharpe"] - baseline_metrics["sharpe"]
    print(f"  VIX TS sizing {'IMPROVES' if sharpe_delta > 0 else 'REDUCES'} Sharpe by {abs(sharpe_delta):.2f}")
    print(f"  Regime gap goes from {baseline_regime['regime_gap']:.2f} to {sized_regime['regime_gap']:.2f} "
          f"({'BETTER' if gap_delta < 0 else 'WORSE'})")
    print(f"  R1 gate: Baseline={'PASS' if baseline_regime['r1_pass'] else 'FAIL'}, "
          f"VIX TS={'PASS' if sized_regime['r1_pass'] else 'FAIL'}")
    print(f"  Average position sizing: {merged['size_mult'].mean():.1%} of full")


if __name__ == "__main__":
    main()
