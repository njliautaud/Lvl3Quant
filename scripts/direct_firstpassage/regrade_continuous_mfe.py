#!/usr/bin/env python
"""
Regrade per_trade_oos.parquet from continuous MFE regressor v1.

For each hold_npts bucket (proxy for horizon) and each Q-gate:
  - Select top-Q rows by |y_pred_signed_terminal| (direction = sign of pred)
  - Realized P&L = realized_terminal_signed, clipped at -SL (SL ∈ {1, 2})
  - Net = realized_pnl - 1.376 (TAKER round-trip)
  - Metrics: cond_WR, mean_net, PF, Sharpe_daily, asymmetry, day_conc, n_trades/day

GO gates (HC #506 R5): mean_net>0 AND PF≥1.2 AND Sharpe≥0.5
  AND asymmetry≤0.50 AND day_conc≤0.70 AND n_trades/day≥5
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TAKER_COST = 1.376
Q_GATES = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10]
SL_OPTIONS = [1.0, 2.0]


def regime_classify_day(green_threshold=0.0, red_threshold=0.0):
    # Placeholder; in practice classify via ES close-to-close. Here we tag by realized mean by day.
    pass


def metrics_for_subset(sub, sl_ticks):
    """sub has columns: y_pred_signed_terminal, y_true_signed_terminal (already side-aligned), oot_date."""
    if len(sub) == 0:
        return None
    pred = sub["y_pred_signed_terminal"].values
    real = sub["y_true_signed_terminal"].values
    # Direction = sign(pred). Realized signed return in our direction = real * sign(pred).
    direction = np.sign(pred)
    # If direction == 0 (zero pred), skip; trade only on non-zero
    nz = direction != 0
    if nz.sum() == 0:
        return None
    pred, real, direction = pred[nz], real[nz], direction[nz]
    realized_in_dir = real * direction
    # Apply SL: clip at -sl_ticks
    pnl = np.clip(realized_in_dir, -sl_ticks, None)
    net = pnl - TAKER_COST
    wr = float((net > 0).mean())
    mean_net = float(net.mean())
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    pf = float(pos / max(neg, 1e-9))
    # Daily metrics
    sub2 = sub.loc[nz].copy()
    sub2["net"] = net
    daily = sub2.groupby("oot_date")["net"].agg(["sum","count","mean"]).reset_index()
    n_days = len(daily)
    sharpe_daily = float(daily["sum"].mean() / max(daily["sum"].std(ddof=1), 1e-9) * np.sqrt(252)) if n_days > 1 else float("nan")
    n_trades_per_day = float(daily["count"].mean())
    # day_conc = max day's |sum| / total |sum|
    tot_abs = daily["sum"].abs().sum()
    day_conc = float(daily["sum"].abs().max() / max(tot_abs, 1e-9)) if n_days > 0 else float("nan")
    return dict(
        n_trades=int(nz.sum()), n_days=n_days,
        cond_wr=wr, mean_net=mean_net, pf=pf,
        sharpe_daily=sharpe_daily, n_trades_per_day=n_trades_per_day,
        day_conc=day_conc,
        gross_mean=float(realized_in_dir.mean()),
        median_net=float(np.median(net)),
        sl_ticks=sl_ticks,
    )


def asymmetry_check(sub, sl_ticks):
    """Stratify by day-classification (mean realized in_dir per day): green/red/flat. Return asym."""
    if len(sub) == 0:
        return float("nan"), {}
    pred = sub["y_pred_signed_terminal"].values
    real = sub["y_true_signed_terminal"].values
    direction = np.sign(pred)
    nz = direction != 0
    pred, real, direction = pred[nz], real[nz], direction[nz]
    realized = real * direction
    pnl = np.clip(realized, -sl_ticks, None) - TAKER_COST
    s2 = sub.loc[nz].copy()
    s2["pnl"] = pnl
    # Classify days by mean ES move (proxy: per-day sign of mean(real) — but we have no ES close).
    # Use per-day total ES tick move proxy: mean of unsigned real per side... simpler:
    # group days by sign of average (real / side). But we don't have raw side here easily.
    # Use mean of REAL terminal-signed in side-favored direction: we already have y_true_signed_terminal.
    # Day classification by SUM of y_true_signed_terminal per day across ALL trades (regardless of pred dir).
    day_signed = sub.groupby("oot_date")["y_true_signed_terminal"].mean().reset_index()
    day_signed["regime"] = pd.cut(day_signed["y_true_signed_terminal"],
                                  bins=[-np.inf, -0.5, 0.5, np.inf],
                                  labels=["red","flat","green"])
    regime_map = dict(zip(day_signed["oot_date"], day_signed["regime"]))
    s2["regime"] = s2["oot_date"].map(regime_map)
    by_reg = s2.groupby("regime", observed=True)["pnl"].agg(["sum","count","mean"]).reset_index()
    sharpe_by_reg = {}
    for r in ["green","red","flat"]:
        if r in by_reg["regime"].values:
            sub_r = s2[s2["regime"] == r].groupby("oot_date")["pnl"].sum()
            if len(sub_r) > 1:
                sh = float(sub_r.mean() / max(sub_r.std(ddof=1),1e-9) * np.sqrt(252))
            else:
                sh = float("nan")
            sharpe_by_reg[r] = sh
        else:
            sharpe_by_reg[r] = float("nan")
    sg = sharpe_by_reg.get("green", float("nan"))
    sr = sharpe_by_reg.get("red", float("nan"))
    if np.isnan(sg) or np.isnan(sr):
        asym = float("nan")
    else:
        denom = max(abs(sg), abs(sr), 1e-9)
        asym = abs(sg - sr) / denom
    return float(asym), sharpe_by_reg


def go_decision(m, asym):
    """HC #506 R5 acceptance gates."""
    if m is None:
        return False, "no_data"
    reasons = []
    if not (m["mean_net"] > 0): reasons.append(f"mean_net={m['mean_net']:.3f}")
    if not (m["pf"] >= 1.2): reasons.append(f"pf={m['pf']:.2f}")
    if not (m["sharpe_daily"] >= 0.5): reasons.append(f"sharpe={m['sharpe_daily']:.2f}")
    if not (asym <= 0.50): reasons.append(f"asym={asym:.2f}")
    if not (m["day_conc"] <= 0.70): reasons.append(f"day_conc={m['day_conc']:.2f}")
    if not (m["n_trades_per_day"] >= 5): reasons.append(f"n_pd={m['n_trades_per_day']:.1f}")
    return (len(reasons) == 0), ";".join(reasons) if reasons else "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.oos_parquet)
    print(f"loaded {len(df)} rows")
    print(f"cols: {list(df.columns)}")

    # Compute hold_npts quartile buckets as horizon proxy
    df["hold_bucket"] = pd.qcut(df["hold_npts"], q=4, labels=["q1_short","q2","q3","q4_long"], duplicates="drop")
    print("hold_bucket counts:")
    print(df["hold_bucket"].value_counts())

    rows = []
    for bucket in ["ALL","q1_short","q2","q3","q4_long"]:
        sub_b = df if bucket == "ALL" else df[df["hold_bucket"] == bucket]
        n_b = len(sub_b)
        if n_b < 100:
            continue
        # rank by |predicted signed terminal|
        for q in Q_GATES:
            n_keep = max(int(n_b * q), 1)
            top = sub_b.nlargest(n_keep, sub_b["y_pred_signed_terminal"].abs().name) if False else \
                  sub_b.iloc[sub_b["y_pred_signed_terminal"].abs().argsort()[::-1][:n_keep]]
            for sl in SL_OPTIONS:
                m = metrics_for_subset(top, sl)
                if m is None: continue
                asym, sharpe_by_reg = asymmetry_check(top, sl)
                go, reason = go_decision(m, asym)
                rows.append(dict(
                    bucket=bucket, q=q, sl=sl,
                    **m, asym=asym,
                    sharpe_green=sharpe_by_reg.get("green", float("nan")),
                    sharpe_red=sharpe_by_reg.get("red", float("nan")),
                    sharpe_flat=sharpe_by_reg.get("flat", float("nan")),
                    GO=go, reason=reason,
                ))

    cells = pd.DataFrame(rows)
    cells.to_parquet(out / "regrade_cells.parquet", index=False)
    cells.to_csv(out / "regrade_cells.csv", index=False)
    print(f"wrote {len(cells)} cells")

    # Spearman per bucket
    from scipy.stats import spearmanr
    sp_rows = []
    for bucket in ["ALL","q1_short","q2","q3","q4_long"]:
        sub_b = df if bucket == "ALL" else df[df["hold_bucket"] == bucket]
        if len(sub_b) < 100:
            continue
        sp_signed = spearmanr(sub_b["y_pred_signed_terminal"], sub_b["y_true_signed_terminal"]).correlation
        if "y_pred_mfe_magnitude" in sub_b.columns:
            sp_mfe = spearmanr(sub_b["y_pred_mfe_magnitude"], sub_b["y_true_mfe_magnitude"]).correlation
        else:
            sp_mfe = float("nan")
        sp_rows.append({"bucket": bucket, "n": len(sub_b),
                        "spearman_signed_terminal": float(sp_signed),
                        "spearman_mfe_magnitude": float(sp_mfe)})
    sp_df = pd.DataFrame(sp_rows)
    sp_df.to_csv(out / "spearman_by_bucket.csv", index=False)
    print(sp_df)

    # Best cell per bucket
    summary = {"per_bucket_spearman": sp_rows, "best_cells": []}
    for bucket in cells["bucket"].unique():
        cb = cells[cells["bucket"] == bucket]
        if len(cb) == 0: continue
        # rank by sharpe_daily but require n_trades_per_day >= 1
        cb2 = cb[cb["n_trades_per_day"] >= 1]
        if len(cb2) == 0: continue
        best = cb2.sort_values("sharpe_daily", ascending=False).iloc[0]
        summary["best_cells"].append({
            "bucket": bucket, "q": float(best["q"]), "sl": float(best["sl"]),
            "cond_wr": float(best["cond_wr"]), "mean_net": float(best["mean_net"]),
            "pf": float(best["pf"]), "sharpe_daily": float(best["sharpe_daily"]),
            "n_trades_per_day": float(best["n_trades_per_day"]),
            "day_conc": float(best["day_conc"]), "asym": float(best["asym"]),
            "GO": bool(best["GO"]), "reason": str(best["reason"]),
        })

    summary["any_GO"] = bool(cells["GO"].any())
    summary["go_cells"] = int(cells["GO"].sum())
    (out / "regrade_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
