#!/usr/bin/env python3
"""
ic_combined_hedge.py — Apply VIX Term Structure Sizing + Dynamic Beta Hedge to Iron Condor

The IC strategy has Sharpe 6.4-7.4 but FAILS HC #428 R1 (regime gap 0.77-0.92).
It's structurally short-gamma: great in calm markets, weak in volatile/red.

This applies the SAME combined hedge overlay that fixed V5 CSP:
- VIX TS Sizing: reduce IC position size when term structure inverts
- Dynamic Beta Hedge: short SPY proportional to exposure, more in stress

Sweep: base_hedge 0.05-0.30, stressed_hedge 0.20-0.60, vix_threshold 16-24
Goal: regime gap < 0.50 while maintaining meaningful Sharpe

Usage:
    python3 /home/jupiter/Lvl3Quant/research/ic_combined_hedge.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
WS_ROOT = ROOT / "wheel_strategy_v1"
OUTPUT = ROOT / "output" / "ic_combined_hedge"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE = 0.04
RF_DAILY = RISK_FREE / TRADING_DAYS
STARTING_CAPITAL = 100_000.0


# ── VIX Term Structure Sizing ───────────────────────────────────────────────
def vix_ts_sizing(ratio: float) -> float:
    """
    Map VIX3M/VIX ratio to position size multiplier.
    >1 = contango (calm), <1 = backwardation (fear).
    """
    breakpoints = [
        (0.75, 0.25), (0.85, 0.25), (0.95, 0.50),
        (1.00, 0.75), (1.10, 1.00), (1.30, 1.00),
    ]
    if np.isnan(ratio):
        return 1.0
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


# ── Load Data ────────────────────────────────────────────────────────────────
def load_ic_equity():
    """Load IC best config equity curve."""
    p = ROOT / "output" / "ic_stress_test" / "eq_best_config.parquet"
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_spy_vix():
    """Load SPY prices, VIX, and VIX3M."""
    cache = WS_ROOT / "data" / "cache"

    # SPY
    spy_df = pd.read_parquet(cache / "spy_prices.parquet")
    spy_df["date"] = pd.to_datetime(spy_df["date"])
    spy_close = spy_df.set_index("date")["close"].sort_index().astype(float)

    # VIX & VIX3M from macro
    macro = pd.read_parquet(cache / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.set_index("date").sort_index()
    vix = macro["vix"].astype(float)
    vix3m = macro["vix3m"].astype(float)

    return spy_close, vix, vix3m


# ── Overlays (vectorized for speed) ─────────────────────────────────────────
def apply_ts_sizing_vec(daily_ret: pd.Series, dates: pd.DatetimeIndex,
                        vix: pd.Series, vix3m: pd.Series,
                        start_equity: float) -> tuple[pd.Series, pd.Series]:
    """Apply VIX TS sizing to daily returns. Vectorized where possible."""
    ratio = (vix3m / vix).reindex(dates).shift(1)  # t-1, no look-ahead
    sizes = ratio.apply(lambda r: vix_ts_sizing(r) if not np.isnan(r) else 1.0)
    sized_ret = daily_ret * sizes

    # Build equity curve
    eq = start_equity * (1 + sized_ret).cumprod()
    eq.iloc[0] = start_equity
    return eq, sizes


def apply_combined(daily_ret: pd.Series, dates: pd.DatetimeIndex,
                   spy_close: pd.Series, vix: pd.Series, vix3m: pd.Series,
                   base_hedge: float, stressed_hedge: float,
                   vix_threshold: float, start_equity: float,
                   use_ts_sizing: bool = True) -> pd.Series:
    """Apply TS sizing + dynamic beta hedge combined."""
    # Step 1: TS sizing
    if use_ts_sizing:
        ratio = (vix3m / vix).reindex(dates).shift(1)
        sizes = ratio.apply(lambda r: vix_ts_sizing(r) if not np.isnan(r) else 1.0)
        sized_ret = daily_ret * sizes
    else:
        sized_ret = daily_ret.copy()

    # Step 2: Dynamic beta hedge
    spy_ret = spy_close.reindex(dates).pct_change().fillna(0.0)
    vix_aligned = vix.reindex(dates).shift(1)  # t-1

    # Hedge ratio: base when calm, stressed when VIX >= threshold
    hedge_ratio = pd.Series(base_hedge, index=dates)
    stressed_mask = vix_aligned >= vix_threshold
    hedge_ratio[stressed_mask] = stressed_hedge

    # Borrow cost for short SPY
    daily_borrow = 0.003 / 252.0

    # Hedged return = sized_return - hedge_ratio * spy_return - borrow_cost
    hedged_ret = sized_ret - hedge_ratio * spy_ret - daily_borrow * hedge_ratio

    # Build equity
    eq = start_equity * (1 + hedged_ret).cumprod()
    eq.iloc[0] = start_equity
    return eq


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(eq: pd.Series, spy_close: pd.Series, label: str = "") -> dict:
    """Compute full suite of risk-adjusted metrics + regime analysis."""
    daily_ret = eq.pct_change().fillna(0.0)
    excess = daily_ret - RF_DAILY
    n = len(eq) - 1

    if n < 30 or excess.std() == 0:
        return {"label": label, "error": "insufficient data"}

    sharpe = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))

    down = excess[excess < 0]
    sortino = float(excess.mean() / down.std() * np.sqrt(TRADING_DAYS)) if len(down) > 0 and down.std() > 0 else 0.0

    yrs = n / TRADING_DAYS
    cagr = (float(eq.iloc[-1]) / float(eq.iloc[0])) ** (1.0 / yrs) - 1.0

    peak = eq.cummax()
    max_dd = float((eq / peak - 1.0).min())
    calmar = cagr / abs(max_dd) if max_dd != 0 else float("nan")

    # Win rate / profit factor
    pnl = daily_ret[daily_ret != 0]
    wr = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = float(gains / losses) if losses > 0 else float("inf")

    # Regime split (SPY close-to-close)
    spy_ret_s = spy_close.reindex(eq.index).pct_change()
    labels = pd.Series("flat", index=spy_ret_s.index)
    labels[spy_ret_s > 0.002] = "green"
    labels[spy_ret_s < -0.002] = "red"

    regime = {}
    for reg in ("green", "red", "flat"):
        sub = daily_ret[labels == reg]
        regime[f"n_{reg}"] = int(len(sub))
        if len(sub) >= 5 and sub.std() > 0:
            exc = sub - RF_DAILY
            regime[f"{reg}_sharpe"] = float(exc.mean() / exc.std() * np.sqrt(TRADING_DAYS))
        else:
            regime[f"{reg}_sharpe"] = float("nan")

    sg = regime.get("green_sharpe", float("nan"))
    sr = regime.get("red_sharpe", float("nan"))
    if not (np.isnan(sg) or np.isnan(sr)):
        denom = max(abs(sg), abs(sr))
        regime["regime_gap"] = abs(sg - sr) / denom if denom > 0 else float("nan")
        regime["r1_pass"] = bool(regime["regime_gap"] <= 0.50)
    else:
        regime["regime_gap"] = float("nan")
        regime["r1_pass"] = False

    return {
        "label": label,
        "sharpe": round(sharpe, 4), "sortino": round(sortino, 4),
        "cagr": round(cagr, 4), "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 4), "pf": round(pf, 4), "wr": round(wr, 4),
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1]) / float(eq.iloc[0]) - 1.0,
        "n_days": n,
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in regime.items()},
    }


# ── Permutation Test ─────────────────────────────────────────────────────────
def permutation_test(daily_ret: pd.Series, spy_ret: pd.Series,
                     vix_aligned: pd.Series, vix3m_aligned: pd.Series,
                     base_hedge: float, stressed_hedge: float,
                     vix_threshold: float, observed_sharpe: float,
                     n_perms: int = 200) -> dict:
    """Shuffle regime labels to test if hedge benefit is real."""
    rng = np.random.default_rng(42)
    n = len(daily_ret)

    # Observed: the actual hedged Sharpe
    # Null: randomly assign hedge ratios (breaks VIX-hedge relationship)
    null_sharpes = []

    ratio = (vix3m_aligned / vix_aligned)
    sizes_arr = np.array([vix_ts_sizing(r) if not np.isnan(r) else 1.0 for r in ratio.values])
    spy_arr = spy_ret.values
    ret_arr = daily_ret.values
    daily_borrow = 0.003 / 252.0

    for _ in range(n_perms):
        # Shuffle the VIX values to break temporal relationship
        perm_idx = rng.permutation(n)
        perm_vix = vix_aligned.values[perm_idx]
        perm_sizes = sizes_arr[perm_idx]

        # Apply with shuffled regime
        sized_ret = ret_arr * perm_sizes
        hedge_ratio = np.where(perm_vix >= vix_threshold, stressed_hedge, base_hedge)
        hedged_ret = sized_ret - hedge_ratio * spy_arr - daily_borrow * hedge_ratio

        excess = hedged_ret - RF_DAILY
        if excess.std() > 0:
            s = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))
        else:
            s = 0.0
        null_sharpes.append(s)

    null_sharpes = np.array(null_sharpes)
    p_value = float(np.mean(null_sharpes >= observed_sharpe))

    return {
        "observed_sharpe": round(observed_sharpe, 4),
        "null_mean": round(float(null_sharpes.mean()), 4),
        "null_std": round(float(null_sharpes.std()), 4),
        "null_p5": round(float(np.percentile(null_sharpes, 5)), 4),
        "null_p95": round(float(np.percentile(null_sharpes, 95)), 4),
        "p_value": round(p_value, 4),
        "n_permutations": n_perms,
        "significant_at_5pct": bool(p_value < 0.05),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  IC Strategy: Combined VIX TS Sizing + Dynamic Beta Hedge")
    print("  Goal: Reduce regime gap from 0.77-0.92 to < 0.50 (R1 compliance)")
    print("=" * 80)

    # Load data
    ic_eq_df = load_ic_equity()
    spy_close, vix, vix3m = load_spy_vix()
    print(f"\n  IC equity: {len(ic_eq_df)} days, {ic_eq_df['date'].min().date()} to {ic_eq_df['date'].max().date()}")

    # Align to dates
    dates = pd.DatetimeIndex(ic_eq_df["date"].values)
    ic_eq = pd.Series(ic_eq_df["equity"].values, index=dates)
    daily_ret = ic_eq.pct_change().fillna(0.0)
    start_eq = float(ic_eq.iloc[0])

    # ── Baseline IC (unhedged) ──
    m_base = compute_metrics(ic_eq, spy_close, "IC Unhedged (best config)")
    print(f"\n  BASELINE IC: Sharpe {m_base['sharpe']:.2f}, Gap {m_base.get('regime_gap', 'N/A')}, "
          f"MaxDD {m_base['max_dd']:.1%}, CAGR {m_base['cagr']:.1%}")

    # ── TS Sizing Only ──
    ts_eq, ts_sizes = apply_ts_sizing_vec(daily_ret, dates, vix, vix3m, start_eq)
    m_ts = compute_metrics(ts_eq, spy_close, "IC + TS Sizing Only")
    print(f"  TS SIZED:   Sharpe {m_ts['sharpe']:.2f}, Gap {m_ts.get('regime_gap', 'N/A')}, "
          f"MaxDD {m_ts['max_dd']:.1%}, CAGR {m_ts['cagr']:.1%}")

    # ── Combined Sweep ──
    print(f"\n{'='*80}")
    print(f"  COMBINED SWEEP: base_hedge x stressed_hedge x vix_threshold")
    print(f"{'='*80}")

    base_hedges = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    stressed_hedges = [0.20, 0.30, 0.40, 0.50, 0.60]
    vix_thresholds = [16, 18, 20, 22, 24]

    results = []
    for bh, sh, vt in product(base_hedges, stressed_hedges, vix_thresholds):
        if sh <= bh:
            continue  # stressed must be > base

        eq = apply_combined(daily_ret, dates, spy_close, vix, vix3m,
                           bh, sh, vt, start_eq, use_ts_sizing=True)
        m = compute_metrics(eq, spy_close, f"TS+Hedge(b{bh}/s{sh}/v{vt})")
        results.append({
            "base_hedge": bh, "stressed_hedge": sh, "vix_threshold": vt,
            **m
        })

    # Also test hedge-only (no TS sizing) for comparison
    results_hedge_only = []
    for bh, sh, vt in product([0.10, 0.15, 0.20, 0.25, 0.30],
                               [0.30, 0.40, 0.50, 0.60],
                               [18, 20, 22]):
        if sh <= bh:
            continue
        eq = apply_combined(daily_ret, dates, spy_close, vix, vix3m,
                           bh, sh, vt, start_eq, use_ts_sizing=False)
        m = compute_metrics(eq, spy_close, f"HedgeOnly(b{bh}/s{sh}/v{vt})")
        results_hedge_only.append({
            "base_hedge": bh, "stressed_hedge": sh, "vix_threshold": vt,
            **m
        })

    # Filter R1-passing configs
    passing = [r for r in results if r.get("r1_pass", False) and r["sharpe"] > 0.5]
    passing.sort(key=lambda x: -x["sharpe"])

    passing_ho = [r for r in results_hedge_only if r.get("r1_pass", False) and r["sharpe"] > 0.5]
    passing_ho.sort(key=lambda x: -x["sharpe"])

    print(f"\n  COMBINED (TS + Hedge): {len(passing)}/{len(results)} configs pass R1")
    print(f"  HEDGE-ONLY (no TS):   {len(passing_ho)}/{len(results_hedge_only)} configs pass R1")

    print(f"\n  TOP 15 R1-PASSING COMBINED configs (by Sharpe):")
    print(f"  {'Config':<26} {'Sharpe':>7} {'Sort':>6} {'Gap':>6} {'Green':>7} {'Red':>7} {'MaxDD':>8} {'CAGR':>7}")
    for r in passing[:15]:
        params = f"b={r['base_hedge']:.2f}/s={r['stressed_hedge']:.2f}/v={r['vix_threshold']}"
        print(f"  {params:<26} {r['sharpe']:>7.2f} {r['sortino']:>6.2f} {r['regime_gap']:>6.3f} "
              f"{r['green_sharpe']:>7.2f} {r['red_sharpe']:>7.2f} "
              f"{r['max_dd']:>7.1%} {r['cagr']:>6.1%}")

    if not passing:
        # Show closest to passing
        all_sorted = sorted(results, key=lambda x: x.get("regime_gap", 999))
        print("  NO configs pass R1. Top 10 by smallest gap:")
        for r in all_sorted[:10]:
            params = f"b={r['base_hedge']:.2f}/s={r['stressed_hedge']:.2f}/v={r['vix_threshold']}"
            print(f"  {params:<26} Sharpe={r['sharpe']:.2f} Gap={r.get('regime_gap', 999):.3f} "
                  f"Green={r.get('green_sharpe', 0):.2f} Red={r.get('red_sharpe', 0):.2f}")

    # ── Permutation Test on Best Config ──
    print(f"\n{'='*80}")
    print(f"  PERMUTATION TEST (200 permutations)")
    print(f"{'='*80}")

    best = passing[0] if passing else (passing_ho[0] if passing_ho else None)
    if best:
        spy_ret_aligned = spy_close.reindex(dates).pct_change().fillna(0.0)
        vix_aligned = vix.reindex(dates).shift(1)
        vix3m_aligned = vix3m.reindex(dates).shift(1)

        perm_result = permutation_test(
            daily_ret, spy_ret_aligned, vix_aligned, vix3m_aligned,
            best["base_hedge"], best["stressed_hedge"], best["vix_threshold"],
            best["sharpe"], n_perms=200
        )
        print(f"\n  Best config: b={best['base_hedge']}/s={best['stressed_hedge']}/v={best['vix_threshold']}")
        print(f"  Observed Sharpe: {perm_result['observed_sharpe']:.4f}")
        print(f"  Null distribution: mean={perm_result['null_mean']:.4f}, std={perm_result['null_std']:.4f}")
        print(f"  Null 5th-95th: [{perm_result['null_p5']:.4f}, {perm_result['null_p95']:.4f}]")
        print(f"  p-value: {perm_result['p_value']:.4f}")
        print(f"  Significant at 5%: {perm_result['significant_at_5pct']}")
    else:
        perm_result = {"error": "No passing config to test"}
        print("  No config to test (none pass R1)")

    # ── Final Comparison Table ──
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*80}")

    comparisons = [
        ("IC Unhedged", m_base),
        ("IC + TS Sizing Only", m_ts),
    ]
    if passing:
        comparisons.append((f"IC BEST Combined (b{passing[0]['base_hedge']}/s{passing[0]['stressed_hedge']}/v{passing[0]['vix_threshold']})",
                           passing[0]))
    if passing_ho:
        comparisons.append((f"IC Hedge-Only (b{passing_ho[0]['base_hedge']}/s{passing_ho[0]['stressed_hedge']}/v{passing_ho[0]['vix_threshold']})",
                           passing_ho[0]))

    # Load V5 combined hedge result for comparison
    v5_file = ROOT / "output" / "v5_combined_hedge" / "results.json"
    if v5_file.exists():
        with open(v5_file) as f:
            v5_data = json.load(f)
        v5_best = v5_data.get("best_combined")
        if v5_best:
            comparisons.append(("V5 CSP + Combined Hedge", v5_best))

    # SPY benchmark
    spy_eq = spy_close.reindex(dates).ffill().bfill()
    spy_eq_norm = spy_eq / spy_eq.iloc[0] * start_eq
    m_spy = compute_metrics(spy_eq_norm, spy_close, "SPY (Buy & Hold)")
    comparisons.append(("SPY Buy & Hold", m_spy))

    print(f"\n  {'Strategy':<50} {'Sharpe':>7} {'Sort':>6} {'CAGR':>7} {'MaxDD':>7} {'Gap':>6} {'R1':>5}")
    print(f"  {'-'*50} {'-'*7} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")
    for label, m in comparisons:
        gap = m.get("regime_gap", float("nan"))
        r1 = "PASS" if m.get("r1_pass", False) else "FAIL"
        sharpe = m.get("sharpe", 0)
        sortino = m.get("sortino", 0)
        cagr = m.get("cagr", 0)
        max_dd = m.get("max_dd", 0)
        print(f"  {label:<50} {sharpe:>7.2f} {sortino:>6.2f} {cagr:>6.1%} {max_dd:>6.1%} "
              f"{gap:>6.3f} {r1:>5}")

    # ── Save Results ──
    save_data = {
        "generated": pd.Timestamp.now().isoformat(),
        "ic_baseline": m_base,
        "ic_ts_sizing_only": m_ts,
        "best_combined": passing[0] if passing else None,
        "best_hedge_only": passing_ho[0] if passing_ho else None,
        "permutation_test": perm_result,
        "all_combined_results": results,
        "all_hedge_only_results": results_hedge_only,
        "r1_passing_combined": passing,
        "r1_passing_hedge_only": passing_ho,
        "n_configs_tested": len(results) + len(results_hedge_only),
        "conclusion": "",
    }

    # Write conclusion
    if passing:
        best_c = passing[0]
        save_data["conclusion"] = (
            f"IC CAN be made regime-neutral. Best config: "
            f"base_hedge={best_c['base_hedge']}, stressed_hedge={best_c['stressed_hedge']}, "
            f"vix_threshold={best_c['vix_threshold']}. "
            f"Sharpe {best_c['sharpe']:.2f} (down from {m_base['sharpe']:.2f} unhedged), "
            f"regime gap {best_c['regime_gap']:.3f} (PASSES R1 < 0.50). "
            f"Permutation p-value: {perm_result.get('p_value', 'N/A')}."
        )
    else:
        save_data["conclusion"] = (
            f"IC CANNOT pass R1 with this hedge approach. "
            f"Minimum gap achieved: {min(r.get('regime_gap', 999) for r in results):.3f}. "
            f"The structural short-gamma exposure is too strong for a simple SPY hedge to neutralize."
        )

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save equity curves
    if passing:
        best_c = passing[0]
        best_eq = apply_combined(daily_ret, dates, spy_close, vix, vix3m,
                                best_c["base_hedge"], best_c["stressed_hedge"],
                                best_c["vix_threshold"], start_eq, use_ts_sizing=True)
        curves = pd.DataFrame({
            "date": dates,
            "ic_unhedged": ic_eq.values,
            "ic_ts_sized": ts_eq.values,
            "ic_best_combined": best_eq.values,
            "spy": spy_eq_norm.values,
        })
        curves.to_parquet(OUTPUT / "equity_curves.parquet", index=False)

    print(f"\n  Results saved to {OUTPUT}")
    print(f"\n  CONCLUSION: {save_data['conclusion']}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
