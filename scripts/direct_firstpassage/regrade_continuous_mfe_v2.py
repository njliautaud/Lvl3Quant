#!/usr/bin/env python
"""
Regrade v2 — uses REAL first-passage SL times from per_trade_walks.parquet (sl{N}_dt_ns).

Honest regrade:
  - Direction = sign(y_pred_signed_terminal)  (chosen trade direction in side-favored frame)
  - If dir > 0 (with side): SL hit iff sl{N}_dt_ns > 0 (means side-favored drop hit -N first)
  - If dir < 0 (anti-side): SL hit iff tp{N}_dt_ns > 0 (means side-favored rise hit +N first ==
    -N in OUR direction)
  - If SL hit: realized = -N
  - Else: realized = y_true_signed_terminal * direction
  - Net = realized - 1.376 TAKER

GO gates HC #506 R5.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TAKER_COST = 1.376
Q_GATES = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10]
SL_OPTIONS = [1.0, 2.0]


def realized_pnl(df, sl_k, gate_col):
    sl_col = f"sl{int(sl_k)}_dt_ns"
    tp_col = f"tp{int(sl_k)}_dt_ns"
    dir_ = np.sign(df[gate_col].values)
    nz = dir_ != 0
    # For nz==False (pred==0), treat as no trade — but we already gated by abs ranking so few zeros.
    real_term = df["y_true_signed_terminal"].values
    realized_in_dir = real_term * dir_
    sl_hit_withside = (df[sl_col].fillna(-1).values > 0)
    sl_hit_antiside = (df[tp_col].fillna(-1).values > 0)
    sl_hit = np.where(dir_ > 0, sl_hit_withside, sl_hit_antiside)
    realized = np.where(sl_hit, -sl_k, realized_in_dir)
    return realized, nz


def classify_regime(df):
    """Per-day mean realized signed terminal as proxy for green/red/flat."""
    day_signed = df.groupby("oot_date")["y_true_signed_terminal"].mean().reset_index()
    day_signed["regime"] = pd.cut(day_signed["y_true_signed_terminal"],
                                  bins=[-np.inf, -0.5, 0.5, np.inf],
                                  labels=["red", "flat", "green"])
    return dict(zip(day_signed["oot_date"], day_signed["regime"]))


def metrics(df, sl_k, gate_col, ranking_col):
    realized, nz = realized_pnl(df, sl_k, gate_col)
    if nz.sum() == 0:
        return None, None
    pnl = realized[nz] - TAKER_COST
    sub = df.loc[nz].copy()
    sub["net"] = pnl
    daily = sub.groupby("oot_date")["net"].agg(["sum", "count"]).reset_index()
    n_days = len(daily)
    sh = float(daily["sum"].mean() / max(daily["sum"].std(ddof=1), 1e-9) * np.sqrt(252)) if n_days > 1 else float("nan")
    dc = float(daily["sum"].abs().max() / max(daily["sum"].abs().sum(), 1e-9)) if n_days > 0 else float("nan")
    pos = pnl[pnl > 0].sum(); neg = -pnl[pnl < 0].sum()
    pf = float(pos / max(neg, 1e-9))
    # regime asymmetry
    regime_map = classify_regime(df)
    sub["regime"] = sub["oot_date"].map(regime_map)
    sharpe_reg = {}
    for r in ["green", "red", "flat"]:
        rd = sub[sub["regime"] == r].groupby("oot_date")["net"].sum()
        if len(rd) > 1:
            sharpe_reg[r] = float(rd.mean() / max(rd.std(ddof=1), 1e-9) * np.sqrt(252))
        else:
            sharpe_reg[r] = float("nan")
    sg, sr = sharpe_reg["green"], sharpe_reg["red"]
    if np.isnan(sg) or np.isnan(sr):
        asym = float("nan")
    else:
        asym = abs(sg - sr) / max(abs(sg), abs(sr), 1e-9)
    return dict(
        n_trades=int(nz.sum()), n_days=n_days,
        cond_wr=float((pnl > 0).mean()),
        mean_net=float(pnl.mean()),
        pf=pf, sharpe_daily=sh,
        n_trades_per_day=float(daily["count"].mean()),
        day_conc=dc,
        asym=float(asym),
        sharpe_green=sg, sharpe_red=sr, sharpe_flat=sharpe_reg["flat"],
        sl_ticks=float(sl_k),
        gate_col=gate_col, ranking_col=ranking_col,
    ), sharpe_reg


def go_decision(m):
    if m is None: return False, "no_data"
    r = []
    if not (m["mean_net"] > 0): r.append(f"mean_net={m['mean_net']:.3f}")
    if not (m["pf"] >= 1.2): r.append(f"pf={m['pf']:.2f}")
    if not (m["sharpe_daily"] >= 0.5): r.append(f"sh={m['sharpe_daily']:.2f}")
    if not (np.isfinite(m["asym"]) and m["asym"] <= 0.50): r.append(f"asym={m['asym']:.2f}")
    if not (m["day_conc"] <= 0.70): r.append(f"dc={m['day_conc']:.2f}")
    if not (m["n_trades_per_day"] >= 5): r.append(f"npd={m['n_trades_per_day']:.1f}")
    return (len(r) == 0), ";".join(r) if r else "PASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-parquet", required=True)
    ap.add_argument("--walks-parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.oos_parquet)
    walks = pd.read_parquet(args.walks_parquet)
    keys = ["event_id", "ts_ns", "oot_date", "side"]
    df = df.merge(walks[keys + ["sl1_dt_ns", "sl2_dt_ns", "tp1_dt_ns", "tp2_dt_ns"]], on=keys, how="left")
    print(f"loaded n={len(df)}")

    # hold_bucket
    df["hold_bucket"] = pd.qcut(df["hold_npts"], q=4, labels=["q1_short","q2","q3","q4_long"], duplicates="drop")

    # Two gating strategies:
    #  (A) gate by |y_pred_signed_terminal|, direction = sign(y_pred_signed_terminal)  [the brief's intended use]
    #  (B) gate by y_pred_mfe_magnitude (always top, dir = sign of pred_signed)
    #  (C) gate by y_pred_mfe_magnitude, dir = +1 (always with side)
    rows = []
    for bucket in ["ALL", "q1_short", "q2", "q3", "q4_long"]:
        sub_b = df if bucket == "ALL" else df[df["hold_bucket"] == bucket]
        if len(sub_b) < 100: continue

        for strategy, gate_col, rank_col in [
            ("A_signed", "y_pred_signed_terminal", "y_pred_signed_terminal"),  # gate by |signed|, dir = sign(signed)
            ("B_mfe_dirSigned", "y_pred_signed_terminal", "y_pred_mfe_magnitude"),  # rank by MFE, dir=sign(signed)
            ("C_mfe_dirSide", "side", "y_pred_mfe_magnitude"),  # rank by MFE, dir = side (always +1 in side frame)
        ]:
            for q in Q_GATES:
                n_keep = max(int(q * len(sub_b)), 1)
                # rank by abs(rank_col)
                idx = sub_b[rank_col].abs().argsort().values[::-1][:n_keep]
                top = sub_b.iloc[idx].copy()
                # For strategy C, gate_col is "side" -> dir = sign(side) which is +/-1 always
                if strategy == "C_mfe_dirSide":
                    top["__dir_col__"] = top["side"].astype(float)
                    actual_gate = "__dir_col__"
                else:
                    actual_gate = gate_col
                for sl in SL_OPTIONS:
                    m, _ = metrics(top, sl, actual_gate, rank_col)
                    if m is None: continue
                    go, reason = go_decision(m)
                    rows.append(dict(strategy=strategy, bucket=bucket, q=q, **m, GO=go, reason=reason))

    cells = pd.DataFrame(rows)
    cells.to_parquet(out / "regrade_cells_v2.parquet", index=False)
    cells.to_csv(out / "regrade_cells_v2.csv", index=False)
    print(f"wrote {len(cells)} cells, GO_count={cells['GO'].sum()}")

    from scipy.stats import spearmanr
    sp_rows = []
    for bucket in ["ALL", "q1_short", "q2", "q3", "q4_long"]:
        sub_b = df if bucket == "ALL" else df[df["hold_bucket"] == bucket]
        if len(sub_b) < 100: continue
        sp_signed = float(spearmanr(sub_b["y_pred_signed_terminal"], sub_b["y_true_signed_terminal"]).correlation)
        sp_mfe = float(spearmanr(sub_b["y_pred_mfe_magnitude"], sub_b["y_true_mfe_magnitude"]).correlation)
        sp_rows.append({"bucket": bucket, "n": len(sub_b),
                        "spearman_signed_terminal": sp_signed,
                        "spearman_mfe_magnitude": sp_mfe})
    sp_df = pd.DataFrame(sp_rows)
    sp_df.to_csv(out / "spearman_by_bucket_v2.csv", index=False)
    print(sp_df.to_string())

    summary = {"per_bucket_spearman": sp_rows, "go_count": int(cells["GO"].sum()),
               "any_GO": bool(cells["GO"].any())}
    # best by sharpe per strategy
    best = []
    for strat in cells["strategy"].unique():
        cs = cells[cells["strategy"] == strat]
        cs2 = cs[cs["n_trades_per_day"] >= 1]
        if len(cs2) == 0: continue
        bw = cs2.sort_values("sharpe_daily", ascending=False).iloc[0]
        best.append({"strategy": strat, "bucket": bw["bucket"], "q": float(bw["q"]), "sl": float(bw["sl_ticks"]),
                     "mean_net": float(bw["mean_net"]), "pf": float(bw["pf"]),
                     "sharpe_daily": float(bw["sharpe_daily"]),
                     "cond_wr": float(bw["cond_wr"]),
                     "n_trades_per_day": float(bw["n_trades_per_day"]),
                     "day_conc": float(bw["day_conc"]), "asym": float(bw["asym"]),
                     "GO": bool(bw["GO"]), "reason": str(bw["reason"])})
    summary["best_per_strategy"] = best
    (out / "regrade_summary_v2.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
