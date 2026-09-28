"""
Validation pack for the regime-overlay ETF rotation leader (hold21_longonly).

What this does (single-process, but each step is vectorised numpy/pandas — no
single-core Python loops over rows, per HC #565 R1):
  1. Per-regime Sharpe split on the overlay's actual book (note: with the
     overlay, bear-regime trades are zero by construction; so this confirms
     the bull-period Sharpe holds up rather than testing the original
     HC #428 R1 ratio which collapses for a cash-gated strategy).
  2. Bull-period bootstrap of per-fold Calmars (B=5000 resamples).
  3. Cost stress: re-run with txn_cost_bps ∈ {5, 10, 20} reading existing
     daily-pnl deltas (txn cost is a linear constant per rebalance day).

Writes a single JSON + markdown report next to the leader run dir.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
_DEFAULT_LEADER = ROOT / "output/macro_picker/etf_rotation_regime_v2_20260608_164726_hold21_longonly"
LEADER = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_LEADER
OUT = LEADER / "validation_pack.json"
MD = LEADER / "validation_pack.md"

TRADING_DAYS = 252


def _load_spy() -> pd.DataFrame:
    candidates = [
        ROOT / "wheel_strategy_v1/data/cache/macro.parquet",
        ROOT / "data/feature_store/sector_etf_flows/daily.parquet",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            if "spy_close" in df.columns:
                d = df[["date", "spy_close"]].dropna().copy()
                d["date"] = pd.to_datetime(d["date"])
                return d.sort_values("date").drop_duplicates("date")
            if "etf" in df.columns and (df["etf"] == "SPY").any():
                d = df[df["etf"] == "SPY"][["date", "close"]].rename(
                    columns={"close": "spy_close"}).copy()
                d["date"] = pd.to_datetime(d["date"])
                return d.sort_values("date").drop_duplicates("date")
        except Exception:
            continue
    # Fallback: derive from book dates if no SPY available.
    return pd.DataFrame()


def regime_split(book: pd.DataFrame, spy: pd.DataFrame) -> dict:
    """Split daily returns by SPY-vs-60d-MA regime."""
    if spy.empty:
        return {"note": "SPY series unavailable; regime split skipped."}
    spy = spy.copy()
    spy["ma60"] = spy["spy_close"].rolling(60, min_periods=20).mean()
    spy["regime"] = np.where(spy["spy_close"] > spy["ma60"], "bull", "bear")
    df = book.merge(spy[["date", "regime"]], on="date", how="left")
    out = {}
    for reg in ["bull", "bear"]:
        s = df[df["regime"] == reg]["daily_ret"].astype(float).dropna()
        if len(s) < 5:
            out[reg] = {"n": int(len(s)), "sharpe": None, "mean_d": None,
                        "wr": None}
            continue
        mu = float(s.mean())
        sd = float(s.std(ddof=1))
        sharpe = (mu / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
        out[reg] = {"n": int(len(s)),
                    "sharpe": float(sharpe),
                    "mean_d": mu,
                    "wr": float((s > 0).mean())}
    # Note: HC #428 R1 ratio (|bull-bear| / max) — for cash-gated strategy
    # bear Sharpe ≈ 0 by construction.
    if out.get("bull", {}).get("sharpe") is not None and out.get("bear", {}).get("sharpe") is not None:
        b, r = out["bull"]["sharpe"], out["bear"]["sharpe"]
        denom = max(abs(b), abs(r), 1e-9)
        out["delta_ratio"] = abs(b - r) / denom
    return out


def per_fold_metrics_from_metrics_json() -> list[dict]:
    with open(LEADER / "metrics.json") as f:
        m = json.load(f)
    return (m.get("per_fold")
            or m.get("per_fold_metrics")
            or m.get("fold_metrics")
            or [])


def bootstrap_calmar(per_fold: list[dict], n_boot: int = 5000,
                     seed: int = 7) -> dict:
    cals = np.array([f.get("calmar") for f in per_fold
                     if f.get("calmar") is not None
                     and np.isfinite(f.get("calmar"))])
    if len(cals) < 3:
        return {"note": "too few folds for bootstrap"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cals), size=(n_boot, len(cals)))
    medians = np.median(cals[idx], axis=1)
    return {
        "n_folds": int(len(cals)),
        "median_observed": float(np.median(cals)),
        "ci_5": float(np.quantile(medians, 0.05)),
        "ci_95": float(np.quantile(medians, 0.95)),
        "p_median_ge_1": float((medians >= 1.0).mean()),
        "p_median_ge_2": float((medians >= 2.0).mean()),
    }


def cost_stress(book: pd.DataFrame, base_bps: float,
                stress_bps_list: list[float],
                n_rebals: int, n_years: float) -> dict:
    """Linearly scale txn cost. base_bps already in the book."""
    daily = book["daily_ret"].astype(float).values
    base_mu = float(np.mean(daily)) if len(daily) else 0.0
    base_sd = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    base_sharpe = base_mu / base_sd * np.sqrt(TRADING_DAYS) if base_sd > 0 else 0.0
    out = {"base_bps": base_bps, "base_sharpe": base_sharpe,
           "n_days": int(len(daily)), "n_rebals": int(n_rebals),
           "n_years": float(n_years)}
    if n_years <= 0:
        return out
    rebal_per_yr = n_rebals / n_years
    # txn cost in bps of book per rebalance day → daily equivalent return drag
    # = (delta_bps / 10000) * rebal_per_yr / TRADING_DAYS (approx)
    drag_at = {}
    for bps in stress_bps_list:
        delta_bps = bps - base_bps
        daily_drag = (delta_bps / 10000.0) * (rebal_per_yr / TRADING_DAYS)
        adj = daily - daily_drag
        mu = float(np.mean(adj))
        sd = float(np.std(adj, ddof=1)) if len(adj) > 1 else 0.0
        sh = mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0
        drag_at[str(bps)] = {"sharpe": sh, "mean_d": mu}
    out["stressed"] = drag_at
    return out


def main():
    if not LEADER.exists():
        print(f"ERR: leader dir not found: {LEADER}", file=sys.stderr)
        sys.exit(2)
    book = pd.read_parquet(LEADER / "book.parquet")
    book["date"] = pd.to_datetime(book["date"])
    spy = _load_spy()

    with open(LEADER / "metrics.json") as f:
        leader_metrics = json.load(f)
    pf = per_fold_metrics_from_metrics_json()
    n_rebals = sum(f.get("n_rebal", 0) or 0 for f in pf)
    n_years = max(len(book) / TRADING_DAYS, 1e-6)

    gate = leader_metrics.get("deploy_gate") or {}
    pooled = leader_metrics.get("pooled") or {}
    results = {
        "leader_dir": str(LEADER),
        "leader_headline_metrics": {
            "median_per_fold_calmar": gate.get("median_per_fold_calmar"),
            "worst_fold_max_dd": gate.get("worst_fold_max_dd"),
            "n_passing_folds": gate.get("n_passing_folds"),
            "n_folds": gate.get("n_folds"),
            "deploy": gate.get("deploy"),
            "pooled_sharpe": pooled.get("sharpe"),
            "pooled_calmar": pooled.get("calmar"),
            "pooled_cagr": pooled.get("cagr"),
            "pooled_max_dd": pooled.get("max_dd"),
            "annual_turnover": leader_metrics.get("annual_turnover_est"),
        },
        "regime_split": regime_split(book, spy),
        "bootstrap_per_fold_calmar": bootstrap_calmar(pf),
        "cost_stress": cost_stress(book, base_bps=5.0,
                                   stress_bps_list=[5, 10, 20],
                                   n_rebals=n_rebals, n_years=n_years),
    }

    OUT.write_text(json.dumps(results, indent=2))

    # Markdown
    md = ["# Validation pack — hold21_longonly + regime overlay",
          "",
          f"Book days: {len(book)} | rebals: {n_rebals} | "
          f"years: {n_years:.2f}",
          ""]
    rs = results["regime_split"]
    md.append("## Regime split (SPY > 60d-MA = bull)")
    if isinstance(rs, dict) and "bull" in rs:
        md.append(f"- Bull: n={rs['bull']['n']}, Sharpe="
                  f"{rs['bull']['sharpe']:.2f}, "
                  f"WR={rs['bull']['wr']*100:.1f}%")
        bear_sh = rs['bear']['sharpe'] if rs['bear']['sharpe'] is not None else float('nan')
        md.append(f"- Bear: n={rs['bear']['n']}, Sharpe={bear_sh:.2f}, "
                  f"WR={(rs['bear']['wr'] or 0)*100:.1f}%")
        if "delta_ratio" in rs:
            md.append(f"- |ΔSharpe|/max = {rs['delta_ratio']:.2f} "
                      f"(note: cash-gated strategy → bear≈0 by construction)")
    else:
        md.append(str(rs))
    md.append("")

    bs = results["bootstrap_per_fold_calmar"]
    md.append("## Bootstrap (B=5000) of per-fold Calmar median")
    if "ci_5" in bs:
        md.append(f"- Folds: {bs['n_folds']}, observed median Calmar: "
                  f"{bs['median_observed']:.2f}")
        md.append(f"- 90% CI: [{bs['ci_5']:.2f}, {bs['ci_95']:.2f}]")
        md.append(f"- P(median Calmar ≥ 1.0) = {bs['p_median_ge_1']*100:.1f}%")
        md.append(f"- P(median Calmar ≥ 2.0) = {bs['p_median_ge_2']*100:.1f}%")
    md.append("")

    cs = results["cost_stress"]
    md.append("## Cost stress (linear txn-cost drag)")
    md.append(f"- Base ({cs['base_bps']:.0f}bps): "
              f"Sharpe {cs['base_sharpe']:.2f} on {cs['n_days']} days")
    if "stressed" in cs:
        for bps, v in cs["stressed"].items():
            md.append(f"- @{bps}bps → Sharpe {v['sharpe']:.2f}")
    md.append("")
    MD.write_text("\n".join(md))

    print(json.dumps(results, indent=2))
    print(f"\nWrote {OUT} and {MD}")


if __name__ == "__main__":
    main()
