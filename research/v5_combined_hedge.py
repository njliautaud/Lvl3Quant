#!/usr/bin/env python3
"""
v5_combined_hedge.py — VIX Term Structure Sizing + Dynamic Beta Hedge Combined

Tests the combination of two protective mechanisms for V5 CSP:
1. VIX TS Sizing: reduces position size when term structure inverts (crisis indicator)
2. Dynamic Beta Hedge: shorts SPY to offset market beta, more in stress

Individually:
  - VIX TS sizing alone: Sharpe 2.53 (up from 2.23), but FAILS R1 (gap 1.43)
  - Dynamic beta hedge alone: Sharpe 0.94, PASSES R1 (gap 0.06)

Hypothesis: combining them should give:
  - Higher Sharpe than hedge alone (TS sizing already reduced vol)
  - A lighter hedge ratio needed (less drag) since positions are already smaller in stress
  - R1 compliance from the hedge

Usage:
    python3 /home/jupiter/Lvl3Quant/research/v5_combined_hedge.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
WS_ROOT = ROOT / "wheel_strategy_v1"
OUTPUT = ROOT / "output" / "v5_combined_hedge"
OUTPUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(WS_ROOT))

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
def load_v5_equity():
    """Load the V5 base equity curve from the wheel engine."""
    # Try multiple possible locations
    candidates = [
        WS_ROOT / "results" / "v5_regime_sized" / "equity_daily.parquet",
        WS_ROOT / "results" / "v5_daily" / "equity_daily.parquet",
        WS_ROOT / "results" / "equity_daily.parquet",
        ROOT / "output" / "v5_vix_ts_sizing" / "equity_curves.parquet",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            if "date" in df.columns and "equity" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").reset_index(drop=True)
            # May be wide format
            if "baseline" in df.columns:
                df = df.reset_index()
                if "date" in df.columns:
                    return pd.DataFrame({
                        "date": pd.to_datetime(df["date"]),
                        "equity": df["baseline"].astype(float),
                    }).sort_values("date").reset_index(drop=True)

    raise FileNotFoundError(
        f"Cannot find V5 equity curve. Tried: {[str(p) for p in candidates]}"
    )


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

    # VIX3M might be in macro or separate
    if "vix3m" in macro.columns:
        vix3m = macro["vix3m"].astype(float)
    else:
        # Try to compute from VIX futures term structure or use a proxy
        # Fallback: approximate VIX3M as smoothed VIX * 1.05 (contango bias)
        # This is crude but will be replaced with real data
        print("WARNING: VIX3M not in macro cache, using lagged VIX ratio proxy")
        vix3m = vix.rolling(60).mean() * 1.05  # rough contango proxy

    return spy_close, vix, vix3m


# ── Overlays ─────────────────────────────────────────────────────────────────
def apply_ts_sizing(base_equity: pd.DataFrame,
                    vix: pd.Series,
                    vix3m: pd.Series) -> pd.DataFrame:
    """Scale daily returns by VIX TS sizing. Uses t-1 data."""
    eq = base_equity.set_index("date")["equity"].sort_index().astype(float)
    dates = eq.index
    daily_ret = eq.pct_change().fillna(0.0)

    # Compute VIX3M/VIX ratio
    ratio = (vix3m / vix).sort_index()

    sized_eq = pd.Series(float(eq.iloc[0]), index=dates)
    sizes = pd.Series(1.0, index=dates)

    for i in range(len(dates)):
        dt = dates[i]
        if i == 0:
            sized_eq.iloc[i] = float(eq.iloc[0])
            continue

        # t-1 ratio (no look-ahead)
        prev_dt = dates[i - 1]
        r_val = float(ratio.get(prev_dt, np.nan))
        sz = vix_ts_sizing(r_val)
        sizes.iloc[i] = sz

        r_base = float(daily_ret.iloc[i])
        sized_eq.iloc[i] = sized_eq.iloc[i - 1] * (1.0 + r_base * sz)

    return pd.DataFrame({"date": dates, "equity": sized_eq.values}), sizes


def apply_dynamic_beta_hedge(base_equity: pd.DataFrame,
                              spy_close: pd.Series,
                              vix: pd.Series,
                              base_hedge: float = 0.20,
                              stressed_hedge: float = 0.70,
                              vix_stress: float = 18.0) -> pd.DataFrame:
    """Dynamic SPY short hedge. Uses t-1 VIX."""
    eq = base_equity.set_index("date")["equity"].sort_index().astype(float)
    dates = eq.index
    daily_ret = eq.pct_change().fillna(0.0)
    spy_ret = spy_close.sort_index().pct_change().fillna(0.0)

    hedged_eq = pd.Series(float(eq.iloc[0]), index=dates)
    daily_borrow_cost = 0.003 / 252.0

    for i in range(len(dates)):
        if i == 0:
            hedged_eq.iloc[i] = float(eq.iloc[0])
            continue

        dt = dates[i]
        prev_dt = dates[i - 1]
        v = float(vix.get(prev_dt, np.nan))

        if np.isnan(v) or v < vix_stress:
            hr = base_hedge
        else:
            hr = stressed_hedge

        r_base = float(daily_ret.iloc[i])
        r_spy = float(spy_ret.get(dt, 0.0))
        r_hedged = r_base - hr * r_spy - daily_borrow_cost * hr
        hedged_eq.iloc[i] = hedged_eq.iloc[i - 1] * (1.0 + r_hedged)

    return pd.DataFrame({"date": dates, "equity": hedged_eq.values})


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(eq_df: pd.DataFrame, spy_close: pd.Series,
                    label: str = "") -> dict:
    eq = eq_df.set_index("date")["equity"].sort_index().astype(float)
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

    # Day concentration
    daily_pnl = eq.diff().dropna()
    total_pnl = daily_pnl.sum()
    day_conc = float(daily_pnl.max() / total_pnl) if total_pnl > 0 else float("nan")

    # Regime split
    spy_ret_s = spy_close.sort_index().pct_change()
    labels = pd.Series("flat", index=spy_ret_s.index)
    labels[spy_ret_s > 0.002] = "green"
    labels[spy_ret_s < -0.002] = "red"
    aligned = labels.reindex(daily_ret.index)

    regime = {}
    for reg in ("green", "red", "flat"):
        sub = daily_ret[aligned == reg]
        regime[f"n_{reg}"] = len(sub)
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
        regime["r1_pass"] = regime["regime_gap"] <= 0.50
    else:
        regime["regime_gap"] = float("nan")
        regime["r1_pass"] = False

    return {
        "label": label,
        "sharpe": sharpe, "sortino": sortino, "cagr": cagr,
        "max_dd": max_dd, "calmar": calmar, "pf": pf, "wr": wr,
        "day_conc": day_conc,
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1]) / float(eq.iloc[0]) - 1.0,
        "n_days": n,
        **regime,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("  V5 CSP: Combined VIX TS Sizing + Dynamic Beta Hedge")
    print("=" * 80)

    # Load data
    base_eq = load_v5_equity()
    spy_close, vix, vix3m = load_spy_vix()
    print(f"  Base equity: {len(base_eq)} days, {base_eq['date'].min().date()} to {base_eq['date'].max().date()}")

    # ── Baseline ──
    m_base = compute_metrics(base_eq, spy_close, "V5 Baseline")
    print(f"\n  BASELINE: Sharpe {m_base['sharpe']:.2f}, Gap {m_base.get('regime_gap', float('nan')):.2f}, "
          f"MaxDD {m_base['max_dd']:.1%}, CAGR {m_base['cagr']:.1%}")

    # ── VIX TS Sizing Only ──
    ts_eq, ts_sizes = apply_ts_sizing(base_eq, vix, vix3m)
    m_ts = compute_metrics(ts_eq, spy_close, "V5 + TS Sizing")
    print(f"  TS SIZED:  Sharpe {m_ts['sharpe']:.2f}, Gap {m_ts.get('regime_gap', float('nan')):.2f}, "
          f"MaxDD {m_ts['max_dd']:.1%}, CAGR {m_ts['cagr']:.1%}")

    # ── Dynamic Beta Hedge Only (best config from prior research) ──
    hedge_eq = apply_dynamic_beta_hedge(base_eq, spy_close, vix, 0.20, 0.70, 18.0)
    m_hedge = compute_metrics(hedge_eq, spy_close, "V5 + Beta Hedge (b0.20/s0.70/v18)")
    print(f"  HEDGED:    Sharpe {m_hedge['sharpe']:.2f}, Gap {m_hedge.get('regime_gap', float('nan')):.2f}, "
          f"MaxDD {m_hedge['max_dd']:.1%}, CAGR {m_hedge['cagr']:.1%}")

    # ── Combined: TS Sizing FIRST, then Beta Hedge ──
    # First reduce position sizes via TS, then hedge the remaining exposure
    print(f"\n{'='*80}")
    print(f"  COMBINED SWEEP: VIX TS Sizing → Dynamic Beta Hedge")
    print(f"{'='*80}")

    results = []

    # Since positions are already smaller under stress from TS sizing,
    # the hedge can be lighter → less CAGR drag
    sweep_configs = [
        # (base_hedge, stressed_hedge, vix_stress)
        (0.05, 0.30, 18), (0.05, 0.40, 18), (0.05, 0.50, 18),
        (0.10, 0.30, 18), (0.10, 0.40, 18), (0.10, 0.50, 18), (0.10, 0.60, 18),
        (0.15, 0.30, 18), (0.15, 0.40, 18), (0.15, 0.50, 18), (0.15, 0.60, 18),
        (0.20, 0.40, 18), (0.20, 0.50, 18), (0.20, 0.60, 18), (0.20, 0.70, 18),
        (0.05, 0.30, 20), (0.05, 0.40, 20), (0.05, 0.50, 20),
        (0.10, 0.30, 20), (0.10, 0.40, 20), (0.10, 0.50, 20), (0.10, 0.60, 20),
        (0.15, 0.30, 20), (0.15, 0.40, 20), (0.15, 0.50, 20), (0.15, 0.60, 20),
        (0.20, 0.40, 20), (0.20, 0.50, 20), (0.20, 0.60, 20), (0.20, 0.70, 20),
        (0.05, 0.30, 22), (0.05, 0.40, 22), (0.10, 0.40, 22), (0.10, 0.50, 22),
        (0.15, 0.40, 22), (0.15, 0.50, 22), (0.20, 0.50, 22), (0.20, 0.60, 22),
    ]

    for base_hr, stress_hr, vix_thresh in sweep_configs:
        # Apply TS sizing first
        ts_sized, _ = apply_ts_sizing(base_eq, vix, vix3m)
        # Then hedge the TS-sized curve
        combined = apply_dynamic_beta_hedge(ts_sized, spy_close, vix,
                                            base_hr, stress_hr, vix_thresh)
        m = compute_metrics(combined, spy_close,
                           f"TS+Hedge(b{base_hr}/s{stress_hr}/v{vix_thresh})")
        gap = m.get("regime_gap", float("nan"))
        results.append({
            "base_hr": base_hr, "stress_hr": stress_hr, "vix_thresh": vix_thresh,
            "sharpe": m["sharpe"], "sortino": m.get("sortino", 0),
            "regime_gap": gap, "r1_pass": m.get("r1_pass", False),
            "green_sharpe": m.get("green_sharpe"), "red_sharpe": m.get("red_sharpe"),
            "max_dd": m.get("max_dd"), "cagr": m.get("cagr"),
            "calmar": m.get("calmar"), "pf": m.get("pf"), "wr": m.get("wr"),
        })

    # Sort by Sharpe among R1-passing configs
    passing = [r for r in results if r.get("r1_pass", False) and r["sharpe"] > 0.3]
    passing.sort(key=lambda x: -x["sharpe"])

    failing = [r for r in results if not r.get("r1_pass", False)]
    failing.sort(key=lambda x: x.get("regime_gap", 999))

    print(f"\n  R1-PASSING configs ({len(passing)}/{len(results)}), sorted by Sharpe:")
    print(f"  {'Config':<28} {'Sharpe':>7} {'Sort':>6} {'Gap':>6} {'Green':>7} {'Red':>7} {'MaxDD':>8} {'CAGR':>7} {'Calmar':>7}")
    for r in passing[:15]:
        params = f"b={r['base_hr']:.2f}/s={r['stress_hr']:.2f}/v={r['vix_thresh']}"
        print(f"  {params:<28} {r['sharpe']:>7.2f} {r['sortino']:>6.2f} {r['regime_gap']:>6.2f} "
              f"{r['green_sharpe']:>7.2f} {r['red_sharpe']:>7.2f} "
              f"{r['max_dd']:>7.1%} {r['cagr']:>6.1%} {r['calmar']:>7.2f}")

    if not passing:
        print("  NO configs pass R1. Showing top 5 by smallest gap:")
        for r in failing[:5]:
            params = f"b={r['base_hr']:.2f}/s={r['stress_hr']:.2f}/v={r['vix_thresh']}"
            print(f"  {params:<28} Sharpe={r['sharpe']:.2f} Gap={r['regime_gap']:.2f} "
                  f"Green={r['green_sharpe']:.2f} Red={r['red_sharpe']:.2f}")

    # ── Compare: Best combined vs individual approaches ──
    print(f"\n{'='*80}")
    print(f"  COMPARISON: Baseline vs TS-Only vs Hedge-Only vs Best Combined")
    print(f"{'='*80}")

    best_combined = passing[0] if passing else None

    rows = [
        ("V5 Baseline", m_base),
        ("V5 + TS Sizing", m_ts),
        ("V5 + Beta Hedge (b0.20/s0.70/v18)", m_hedge),
    ]
    if best_combined:
        # Re-run best to get full metrics
        ts_sized, _ = apply_ts_sizing(base_eq, vix, vix3m)
        best_eq = apply_dynamic_beta_hedge(
            ts_sized, spy_close, vix,
            best_combined["base_hr"], best_combined["stress_hr"], best_combined["vix_thresh"])
        m_best = compute_metrics(best_eq, spy_close,
                                 f"BEST COMBINED (b{best_combined['base_hr']}/s{best_combined['stress_hr']}/v{best_combined['vix_thresh']})")
        rows.append((m_best["label"], m_best))

    print(f"\n  {'Strategy':<45} {'Sharpe':>7} {'Sort':>6} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'Gap':>6} {'R1':>5}")
    for label, m in rows:
        gap = m.get("regime_gap", float("nan"))
        r1 = "PASS" if m.get("r1_pass", False) else "FAIL"
        print(f"  {label:<45} {m['sharpe']:>7.2f} {m.get('sortino',0):>6.2f} "
              f"{m.get('cagr',0):>6.1%} {m.get('max_dd',0):>6.1%} "
              f"{m.get('calmar',0):>7.2f} {gap:>6.2f} {r1:>5}")

    # ── Save results ──
    save_data = {
        "baseline": m_base,
        "ts_sizing_only": m_ts,
        "hedge_only": m_hedge,
        "best_combined": best_combined,
        "all_combined_results": results,
        "passing_r1": passing,
    }
    with open(OUTPUT / "results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save equity curves for best combined
    if best_combined:
        curves = pd.DataFrame({
            "date": base_eq["date"],
        })
        # Align all curves to same dates
        base_aligned = base_eq.set_index("date")["equity"]
        ts_aligned = ts_eq.set_index("date")["equity"]
        hedge_aligned = hedge_eq.set_index("date")["equity"]
        best_aligned = best_eq.set_index("date")["equity"]

        curves["baseline"] = base_aligned.reindex(base_eq["date"].values).values
        curves["ts_sized"] = ts_aligned.reindex(base_eq["date"].values).values
        curves["beta_hedged"] = hedge_aligned.reindex(base_eq["date"].values).values
        curves["combined"] = best_aligned.reindex(base_eq["date"].values).values
        curves.to_parquet(OUTPUT / "equity_curves.parquet", index=False)

    print(f"\n  Results saved to {OUTPUT}")
    print(f"\n{'='*80}")

    # Key finding summary
    if best_combined:
        improvement = best_combined["sharpe"] - m_hedge["sharpe"]
        print(f"\n  KEY FINDING: Combined approach {'IMPROVES' if improvement > 0 else 'does NOT improve'} on hedge-only")
        print(f"    Hedge-only Sharpe: {m_hedge['sharpe']:.2f}")
        print(f"    Combined Sharpe:   {best_combined['sharpe']:.2f} (delta: {improvement:+.2f})")
        print(f"    Combined passes R1: {best_combined.get('r1_pass', False)}")
        if best_combined["sharpe"] > m_hedge["sharpe"] and best_combined.get("r1_pass"):
            print(f"    ✓ DEPLOY CANDIDATE: TS sizing + hedge gives better risk-adjusted returns AND passes R1")
        elif best_combined.get("r1_pass"):
            print(f"    ✓ R1-compliant but hedge-only has better Sharpe — TS sizing redundant with hedge")
        else:
            print(f"    ✗ Does not improve on hedge-only — dynamic beta hedge alone is sufficient")
    else:
        print(f"\n  No combined config passes R1 — unexpected, investigate.")


if __name__ == "__main__":
    main()
