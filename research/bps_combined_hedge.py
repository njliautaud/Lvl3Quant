#!/usr/bin/env python3
"""
bps_combined_hedge.py — VIX Term Structure Sizing + Dynamic SPY Beta Hedge for BPS Standard

Applies the same overlay that worked for:
- V5 CSP: gap 1.76 -> 0.46, Sharpe 2.27
- IC Engine: gap 1.25 -> 0.37, Sharpe 5.68

BPS Standard baseline: Sharpe ~1.55, MaxDD -46%, high CAGR but regime-dependent.
Goal: reduce regime gap below 0.50 while preserving Sharpe > 1.5.

Usage:
    python3 /home/jupiter/Lvl3Quant/research/bps_combined_hedge.py
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
OUTPUT = ROOT / "output" / "bps_combined_hedge"
OUTPUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE = 0.04
RF_DAILY = RISK_FREE / TRADING_DAYS
STARTING_CAPITAL = 100_000.0


# ── VIX Term Structure Sizing ───────────────────────────────────────────────
def vix_ts_sizing(ratio: float) -> float:
    """Map VIX3M/VIX ratio to position size multiplier."""
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
def load_bps_equity():
    """Load BPS Standard equity curve (VIX-gated at 20)."""
    p = ROOT / "output" / "bps_vix_gate" / "eq_vix_20.parquet"
    if not p.exists():
        # Fallback to no-gate baseline
        p = ROOT / "output" / "bps_vix_gate" / "eq_no_gate.parquet"
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

    if "vix3m" in macro.columns:
        vix3m = macro["vix3m"].astype(float)
    else:
        # Approximate VIX3M from lagged VIX
        print("WARNING: VIX3M not in macro cache, using lagged VIX ratio proxy")
        vix3m = vix.rolling(60).mean() * 1.05
    return spy_close, vix, vix3m


# ── Combined Overlay (vectorized) ───────────────────────────────────────────
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

    hedge_ratio = pd.Series(base_hedge, index=dates)
    stressed_mask = vix_aligned >= vix_threshold
    hedge_ratio[stressed_mask] = stressed_hedge

    daily_borrow = 0.003 / 252.0

    hedged_ret = sized_ret - hedge_ratio * spy_ret - daily_borrow * hedge_ratio

    eq = start_equity * (1 + hedged_ret).cumprod()
    eq.iloc[0] = start_equity
    return eq


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(eq: pd.Series, spy_close: pd.Series, label: str = "") -> dict:
    """Compute full risk-adjusted metrics + regime analysis."""
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

    pnl = daily_ret[daily_ret != 0]
    wr = float((pnl > 0).mean()) if len(pnl) > 0 else 0.0
    gains = pnl[pnl > 0].sum()
    losses = abs(pnl[pnl < 0].sum())
    pf = float(gains / losses) if losses > 0 else float("inf")

    # Regime split
    spy_ret_s = spy_close.reindex(eq.index).pct_change()
    labels = pd.Series("flat", index=eq.index)
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
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 4),
        "final_equity": round(float(eq.iloc[-1]), 2),
        "n_days": n,
        **regime,
    }


# ── Permutation Test ─────────────────────────────────────────────────────────
def permutation_test_gap(daily_ret: pd.Series, spy_close: pd.Series,
                         n_perms: int = 200) -> dict:
    """Test if observed regime gap is statistically significant."""
    spy_ret_s = spy_close.reindex(daily_ret.index).pct_change()
    labels = pd.Series("flat", index=daily_ret.index)
    labels[spy_ret_s > 0.002] = "green"
    labels[spy_ret_s < -0.002] = "red"

    def calc_gap(ret, lab):
        gaps = []
        for reg in ("green", "red"):
            sub = ret[lab == reg]
            if len(sub) >= 5 and sub.std() > 0:
                exc = sub - RF_DAILY
                gaps.append(float(exc.mean() / exc.std() * np.sqrt(TRADING_DAYS)))
            else:
                return float("nan")
        denom = max(abs(gaps[0]), abs(gaps[1]))
        return abs(gaps[0] - gaps[1]) / denom if denom > 0 else float("nan")

    observed_gap = calc_gap(daily_ret, labels)

    perm_gaps = []
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        shuffled = labels.copy()
        shuffled[:] = rng.permutation(labels.values)
        g = calc_gap(daily_ret, shuffled)
        if not np.isnan(g):
            perm_gaps.append(g)

    if len(perm_gaps) < 50:
        return {"observed_gap": observed_gap, "p_value": float("nan"), "n_valid_perms": len(perm_gaps)}

    perm_gaps = np.array(perm_gaps)
    p_value = float(np.mean(perm_gaps >= observed_gap))

    return {
        "observed_gap": round(observed_gap, 4),
        "p_value": round(p_value, 4),
        "perm_mean_gap": round(float(perm_gaps.mean()), 4),
        "perm_95th": round(float(np.percentile(perm_gaps, 95)), 4),
        "n_valid_perms": len(perm_gaps),
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading BPS Standard equity curve...")
    bps_df = load_bps_equity()
    print(f"  {len(bps_df)} days, {bps_df['date'].min()} to {bps_df['date'].max()}")

    print("Loading SPY/VIX data...")
    spy_close, vix, vix3m = load_spy_vix()

    # Prepare returns
    eq_series = bps_df.set_index("date")["equity"].sort_index().astype(float)
    dates = eq_series.index
    daily_ret = eq_series.pct_change().fillna(0.0)
    start_eq = float(eq_series.iloc[0])

    # Baseline metrics
    print("\nComputing baseline metrics...")
    baseline = compute_metrics(eq_series, spy_close, "BPS Standard (VIX gate 20)")
    print(f"  Baseline: Sharpe={baseline['sharpe']}, MaxDD={baseline['max_dd']:.1%}, "
          f"Gap={baseline.get('regime_gap','?'):.3f}, R1={'PASS' if baseline.get('r1_pass') else 'FAIL'}")

    # Sweep parameters
    base_hedges = [0.05, 0.10, 0.15, 0.20]
    stressed_hedges = [0.20, 0.30, 0.40, 0.50]
    vix_thresholds = [18, 20, 22, 24]

    print(f"\nSweeping {len(base_hedges)*len(stressed_hedges)*len(vix_thresholds)} combinations...")

    all_results = []
    passing_r1 = []

    for bh, sh, vt in product(base_hedges, stressed_hedges, vix_thresholds):
        if sh <= bh:
            continue  # stressed must be higher than base

        hedged_eq = apply_combined(
            daily_ret, dates, spy_close, vix, vix3m,
            base_hedge=bh, stressed_hedge=sh, vix_threshold=vt,
            start_equity=start_eq, use_ts_sizing=True,
        )

        m = compute_metrics(hedged_eq, spy_close,
                           f"b{bh}/s{sh}/v{vt}")

        result = {
            "base_hr": bh,
            "stress_hr": sh,
            "vix_thresh": vt,
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "regime_gap": m.get("regime_gap", float("nan")),
            "r1_pass": m.get("r1_pass", False),
            "green_sharpe": m.get("green_sharpe", float("nan")),
            "red_sharpe": m.get("red_sharpe", float("nan")),
            "max_dd": m["max_dd"],
            "cagr": m["cagr"],
            "calmar": m["calmar"],
            "pf": m["pf"],
            "wr": m["wr"],
        }
        all_results.append(result)

        if result["r1_pass"]:
            passing_r1.append(result)

    # Sort passing by Sharpe
    passing_r1.sort(key=lambda x: x["sharpe"], reverse=True)
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(passing_r1)} / {len(all_results)} configs pass R1 (gap < 0.50)")
    print(f"{'='*60}")

    if passing_r1:
        best = passing_r1[0]
        print(f"\nBEST R1-PASSING CONFIG:")
        print(f"  Base hedge: {best['base_hr']}")
        print(f"  Stressed hedge: {best['stress_hr']}")
        print(f"  VIX threshold: {best['vix_thresh']}")
        print(f"  Sharpe: {best['sharpe']}")
        print(f"  Sortino: {best['sortino']}")
        print(f"  CAGR: {best['cagr']:.1%}")
        print(f"  MaxDD: {best['max_dd']:.1%}")
        print(f"  Regime gap: {best['regime_gap']:.3f}")
        print(f"  Green Sharpe: {best['green_sharpe']:.3f}")
        print(f"  Red Sharpe: {best['red_sharpe']:.3f}")

        # Run permutation test on best
        print(f"\nRunning permutation test (200 perms) on best config...")
        best_eq = apply_combined(
            daily_ret, dates, spy_close, vix, vix3m,
            base_hedge=best["base_hr"], stressed_hedge=best["stress_hr"],
            vix_threshold=best["vix_thresh"], start_equity=start_eq,
            use_ts_sizing=True,
        )
        best_daily_ret = best_eq.pct_change().fillna(0.0)
        perm_result = permutation_test_gap(best_daily_ret, spy_close, n_perms=200)
        print(f"  Permutation p-value: {perm_result['p_value']}")
        print(f"  Perm mean gap: {perm_result['perm_mean_gap']}")
        print(f"  Perm 95th: {perm_result['perm_95th']}")
    else:
        best = None
        perm_result = None
        print("\nNO configs pass R1. Showing top 5 by Sharpe with gap < 1.0:")
        for r in [x for x in all_results if x.get("regime_gap", 99) < 1.0][:5]:
            print(f"  b{r['base_hr']}/s{r['stress_hr']}/v{r['vix_thresh']}: "
                  f"Sharpe={r['sharpe']}, gap={r['regime_gap']:.3f}")

    # Also test TS sizing only and hedge only
    print("\n--- TS Sizing Only ---")
    ts_only_eq = apply_combined(
        daily_ret, dates, spy_close, vix, vix3m,
        base_hedge=0.0, stressed_hedge=0.0, vix_threshold=20,
        start_equity=start_eq, use_ts_sizing=True,
    )
    ts_only = compute_metrics(ts_only_eq, spy_close, "BPS + TS Sizing Only")
    print(f"  Sharpe={ts_only['sharpe']}, Gap={ts_only.get('regime_gap','?'):.3f}")

    # Save results
    output = {
        "baseline": baseline,
        "ts_sizing_only": ts_only,
        "best_combined": best,
        "permutation_test": perm_result,
        "passing_r1": passing_r1,
        "all_combined_results": all_results[:20],  # top 20 by sharpe
        "summary": {
            "n_configs_tested": len(all_results),
            "n_passing_r1": len(passing_r1),
            "best_sharpe_passing": best["sharpe"] if best else None,
            "best_gap_passing": best["regime_gap"] if best else None,
            "baseline_sharpe": baseline["sharpe"],
            "baseline_gap": baseline.get("regime_gap"),
        }
    }

    # Handle NaN for JSON
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(x) for x in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    output = clean_nan(output)

    with open(OUTPUT / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT / 'results.json'}")

    # Save best equity curve
    if best:
        best_eq_df = pd.DataFrame({"date": dates, "equity": best_eq.values})
        best_eq_df.to_parquet(OUTPUT / "eq_best_combined.parquet", index=False)

    return output


if __name__ == "__main__":
    main()
