#!/usr/bin/env python3
"""
LGBM meta-gate regime stratification (HC #428 R1 + HC #487 R3 follow-up)
========================================================================
Joins the threshold-sweep per-day metrics from `lgbm_meta_gate_v1.py` against
`oot_dates_regime.parquet` (canonical close-to-close regime classifier) and
computes per-regime aggregates for every threshold.

Reads:
  /home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1/threshold_sweep_per_day.csv
  /home/jupiter/Lvl3Quant/data/oot_dates_regime.parquet

Writes:
  /home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1/regime_stratified_sweep.csv
  /home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1/regime_stratified_verdict.md

Decision (HC #428 R1):
  REJECT a threshold if regime_imbalance = |Sharpe_g - Sharpe_r| / max(|Sh_g|,|Sh_r|) > 0.50
  even when mean Sharpe is positive.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_SWEEP_CSV = Path("/home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1/threshold_sweep_per_day.csv")
DEFAULT_REGIME_PARQUET = Path("/home/jupiter/Lvl3Quant/data/oot_dates_regime.parquet")
DEFAULT_OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1")


def _normalize_date(s):
    """Coerce various date forms to 8-char YYYYMMDD."""
    if pd.isna(s):
        return None
    s = str(s)
    if len(s) == 8 and s.isdigit():
        return s
    try:
        return pd.to_datetime(s).strftime("%Y%m%d")
    except Exception:
        # try stripping "oot_" prefix
        s2 = s.replace("oot_", "")
        if len(s2) == 8 and s2.isdigit():
            return s2
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-csv", type=Path, default=DEFAULT_SWEEP_CSV)
    ap.add_argument("--regime-parquet", type=Path, default=DEFAULT_REGIME_PARQUET)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if not args.sweep_csv.exists():
        raise SystemExit(f"missing sweep csv: {args.sweep_csv}")
    if not args.regime_parquet.exists():
        raise SystemExit(f"missing regime parquet: {args.regime_parquet}")

    sweep = pd.read_csv(args.sweep_csv)
    regime = pd.read_parquet(args.regime_parquet)

    # Normalize date columns on both sides
    # Sweep: typical col name is "oot_day" or "date" or contains "oot_YYYYMMDD"
    date_col = None
    for c in ("oot_day", "date", "oot_date", "day"):
        if c in sweep.columns:
            date_col = c
            break
    if date_col is None:
        # fallback: first string-ish column
        for c in sweep.columns:
            if sweep[c].dtype == object:
                date_col = c
                break
    if date_col is None:
        raise SystemExit(f"could not find date column in sweep csv. cols={list(sweep.columns)}")
    sweep["date_norm"] = sweep[date_col].map(_normalize_date)

    regime_date_col = None
    for c in ("date", "oot_date", "trading_day", "day"):
        if c in regime.columns:
            regime_date_col = c
            break
    if regime_date_col is None:
        # If the parquet uses the index for date
        if regime.index.name in ("date", "oot_date", "trading_day"):
            regime = regime.reset_index()
            regime_date_col = regime.columns[0]
    if regime_date_col is None:
        raise SystemExit(f"could not find date column in regime parquet. cols={list(regime.columns)} index={regime.index.name}")
    regime["date_norm"] = regime[regime_date_col].map(_normalize_date)

    # Regime column
    regime_label_col = None
    for c in ("regime", "regime_label", "regime_class", "day_class"):
        if c in regime.columns:
            regime_label_col = c
            break
    if regime_label_col is None:
        raise SystemExit(f"could not find regime label column. cols={list(regime.columns)}")

    # Normalize regime labels to {green,red,flat}
    def _norm_regime(x):
        if pd.isna(x): return None
        s = str(x).lower()
        if s in ("green","up","bull","positive","+1","1"): return "green"
        if s in ("red","down","bear","negative","-1"): return "red"
        if s in ("flat","neutral","0"): return "flat"
        return s

    regime["regime_norm"] = regime[regime_label_col].map(_norm_regime)

    # Join
    keep = regime[["date_norm","regime_norm"]].drop_duplicates(subset=["date_norm"])
    merged = sweep.merge(keep, on="date_norm", how="left")
    n_missing = int(merged["regime_norm"].isna().sum())
    if n_missing > 0:
        print(f"[warn] {n_missing} sweep rows had no regime mapping — left as NaN")

    # Identify metric cols (everything numeric except threshold)
    numeric_cols = [c for c in merged.columns
                    if pd.api.types.is_numeric_dtype(merged[c]) and c not in ("threshold",)]

    # Per-regime aggregates per threshold
    out_rows = []
    for thr, grp_thr in merged.groupby("threshold"):
        # ALL regime (no filter)
        d_all = grp_thr.copy()
        # per-regime
        for regime_name in ("all","green","red","flat"):
            if regime_name == "all":
                d = d_all
            else:
                d = grp_thr[grp_thr["regime_norm"] == regime_name]
            if len(d) == 0:
                out_rows.append({
                    "threshold": thr, "regime": regime_name, "days": 0,
                    "total_trades": 0, "mean_per_day_sharpe": np.nan,
                    "median_per_day_sharpe": np.nan, "total_net_ticks": 0.0,
                    "prof_days": 0, "prof_days_frac": np.nan, "wr_overall": np.nan,
                })
                continue

            sharpe_col = next((c for c in ("per_day_sharpe","sharpe","sharpe_daily","mean_sharpe") if c in d.columns), None)
            trades_col = next((c for c in ("n_trades","trades","count","n") if c in d.columns), None)
            net_col = next((c for c in ("net_ticks","total_net_ticks","net_ticks_total","sum_net_ticks") if c in d.columns), None)
            wr_col = next((c for c in ("wr","win_rate","WR") if c in d.columns), None)

            sharpe_vals = d[sharpe_col].dropna() if sharpe_col else pd.Series([], dtype=float)
            n_trades = int(d[trades_col].sum()) if trades_col else 0
            total_net = float(d[net_col].sum()) if net_col else 0.0
            prof_days = int((d[net_col] > 0).sum()) if net_col else 0
            wr_overall = float(d[wr_col].mean()) if wr_col else np.nan

            out_rows.append({
                "threshold": thr,
                "regime": regime_name,
                "days": len(d),
                "total_trades": n_trades,
                "mean_per_day_sharpe": float(sharpe_vals.mean()) if len(sharpe_vals) else np.nan,
                "median_per_day_sharpe": float(sharpe_vals.median()) if len(sharpe_vals) else np.nan,
                "total_net_ticks": total_net,
                "prof_days": prof_days,
                "prof_days_frac": prof_days / len(d) if len(d) else np.nan,
                "wr_overall": wr_overall,
            })

    out = pd.DataFrame(out_rows)
    out_csv = args.out_dir / "regime_stratified_sweep.csv"
    out.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv} ({len(out)} rows)")

    # Verdict computation
    pivoted = out.pivot(index="threshold", columns="regime", values="mean_per_day_sharpe")
    pivoted_n = out.pivot(index="threshold", columns="regime", values="total_trades")
    pivoted_pd = out.pivot(index="threshold", columns="regime", values="prof_days_frac")

    def _imbalance(g, r):
        if pd.isna(g) or pd.isna(r): return np.nan
        denom = max(abs(g), abs(r))
        if denom == 0: return np.nan
        return abs(g - r) / denom

    pivoted["regime_imbalance"] = [_imbalance(g, r) for g, r in zip(pivoted.get("green", np.nan), pivoted.get("red", np.nan))]

    # Find best threshold by ALL-regime mean Sharpe with HC #428 R1 gate
    candidates = pivoted.copy()
    candidates["passes_imbalance"] = candidates["regime_imbalance"] <= 0.50
    candidates["all_pos"] = candidates.get("all", pd.Series(np.nan, index=candidates.index)) > 0
    deployable = candidates[candidates["passes_imbalance"] & candidates["all_pos"]]
    best_thr = None
    if not deployable.empty:
        best_thr = float(deployable["all"].idxmax())
    elif not candidates.empty and candidates.get("all", pd.Series()).notna().any():
        # No deployable cell — pick least-bad for diagnostic
        best_thr = float(candidates["all"].idxmax())

    # Verdict markdown
    md_lines = ["# LGBM meta-gate v1 — Regime-Stratified Threshold Sweep Verdict\n",
                f"Source sweep: `{args.sweep_csv}`",
                f"Regime source: `{args.regime_parquet}`",
                f"Output: `{out_csv}`\n",
                "## Per-threshold mean per-day Sharpe by regime\n",
                pivoted.round(3).to_markdown(),
                "\n## Trade counts by regime\n",
                pivoted_n.fillna(0).astype(int).to_markdown(),
                "\n## Profitable-days fraction by regime\n",
                pivoted_pd.round(3).to_markdown(),
                "\n## Verdict (HC #428 R1)\n",
                ]
    if best_thr is not None and (pivoted.loc[best_thr].get("all", np.nan) > 0)\
            and (pivoted.loc[best_thr].get("regime_imbalance", 1.0) <= 0.50):
        md_lines.append(f"**ALPHA CELL FOUND** at threshold={best_thr}.")
        md_lines.append(f"- All-regime Sharpe: {pivoted.loc[best_thr].get('all'):.3f}")
        md_lines.append(f"- Green: {pivoted.loc[best_thr].get('green'):.3f}")
        md_lines.append(f"- Red: {pivoted.loc[best_thr].get('red'):.3f}")
        md_lines.append(f"- Regime imbalance: {pivoted.loc[best_thr].get('regime_imbalance'):.3f} (≤ 0.50 gate PASS)")
    else:
        md_lines.append("**REJECT** — no threshold produced ALL-regime Sharpe > 0 with regime imbalance ≤ 0.50.")
        if best_thr is not None:
            md_lines.append(f"- Best (least-bad) threshold: {best_thr}")
            md_lines.append(f"- All-regime Sharpe at best: {pivoted.loc[best_thr].get('all'):.3f}")
            md_lines.append(f"- Regime imbalance: {pivoted.loc[best_thr].get('regime_imbalance'):.3f}")
        md_lines.append("\nConclusion: base v3.4.2 OOT preds carry no positive-edge information about FIFO-net profitability at ANY threshold or regime. The meta-gate cannot manufacture edge from base predictions that lack it. Next path: HC #486 Step 3 (new model with 250ms head + supervised stream-stability auxiliary loss).")

    out_md = args.out_dir / "regime_stratified_verdict.md"
    out_md.write_text("\n".join(md_lines))
    print(f"[ok] wrote {out_md}")


if __name__ == "__main__":
    main()
