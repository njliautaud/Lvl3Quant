#!/usr/bin/env python
"""
Regrade direct first-passage heads under TAKER cost 1.376t.

For each (K,S) head and top-Q percentile gate, compute per-trade payoff:
  TP first (within 15s)   -> +K ticks gross
  SL first (within 15s)   -> -S ticks gross
  Hold cap (neither hit)  -> mid_change_hold_ticks * side (gross)
Net = gross - 1.376 (round-trip taker).

Metrics per cell:
  - cond_WR = P(TP_K before SL_S | TP or SL hit) — model-edge interpretable
  - WR_pos_net = fraction trades with net>0 (including hold-cap exits)
  - mean_ticks_net
  - PF = sum(wins_net) / abs(sum(losses_net))
  - Sharpe_daily (sum of trade net per day, then ratio across days)
  - day_concentration = max( |daily_pnl| ) / sum(|daily_pnl|)
  - n_trades, n_trades_per_day
  - HC #428 R1 regime gate: greens/reds Sharpe asymmetry on each Q-cell

Acceptance for GO:
  - mean_ticks_net > 0
  - PF >= 1.2
  - Sharpe >= 0.5 daily
  - regime asymmetry <= 0.50
  - day_concentration <= 0.70
  - n_trades_per_day >= 5
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

HOLD_CAP_NS = 15_000_000_000
TAKER_COST = 1.376

Q_GRID = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0]  # percent (top of distribution)
HEADS = [(2,1),(3,1),(4,1),(5,1),(3,2),(4,2),(5,2),(5,4)]


def es_regime_per_day(walk_df: pd.DataFrame) -> dict[str, str]:
    """Classify each OOT date as green/red/flat based on net mid-change of all signals that day.
    Simple proxy since we don't have ES close-to-close here: use median mid_change_hold_ticks across all events that day.
    """
    out = {}
    for d, g in walk_df.groupby("oot_date"):
        # Use the actual mid-change normalized by side — gives directional drift
        # For regime classification we want raw market direction; use mean of mid_change_hold_ticks regardless of side, signed by aggregate market direction
        # Simpler: weighted by side, then aggregate (since side-flipped events average market movement)
        mkt_drift = (g["mid_change_hold_ticks"] * g["side"]).mean()  # this is signal direction, not market
        # True market direction: avg mid_change for side=+1 events only
        long_drift = g.loc[g["side"] == 1, "mid_change_hold_ticks"].mean() if (g["side"] == 1).any() else 0.0
        if pd.isna(long_drift): long_drift = 0.0
        if long_drift > 0.5:
            out[d] = "green"
        elif long_drift < -0.5:
            out[d] = "red"
        else:
            out[d] = "flat"
    return out


def compute_trade_net(row, K, S):
    """For a row, compute the trade gross ticks (signed) when this (K,S) head fires."""
    tp = row[f"tp{K}_dt_ns"]
    sl = row[f"sl{S}_dt_ns"]
    tp_hit = (tp > 0) & (tp <= HOLD_CAP_NS)
    sl_hit = (sl > 0) & (sl <= HOLD_CAP_NS)
    if tp_hit and (not sl_hit or tp < sl):
        return float(K)  # +K ticks
    if sl_hit and (not tp_hit or sl < tp):
        return float(-S)  # -S ticks
    # Hold cap exit: mid_change_hold_ticks aligns with our side (trade in side direction)
    return float(row["mid_change_hold_ticks"] * row["side"])


def vectorized_gross_ticks(df: pd.DataFrame, K: int, S: int) -> np.ndarray:
    tp = df[f"tp{K}_dt_ns"].values
    sl = df[f"sl{S}_dt_ns"].values
    tp_hit = (tp > 0) & (tp <= HOLD_CAP_NS)
    sl_hit = (sl > 0) & (sl <= HOLD_CAP_NS)
    tp_first = tp_hit & ((~sl_hit) | (tp < sl))
    sl_first = sl_hit & ((~tp_hit) | (sl < tp))
    holdcap = (~tp_first) & (~sl_first)
    gross = np.zeros(len(df), dtype=np.float64)
    gross[tp_first] = float(K)
    gross[sl_first] = float(-S)
    holdcap_idx = np.where(holdcap)[0]
    gross[holdcap_idx] = df["mid_change_hold_ticks"].values[holdcap_idx] * df["side"].values[holdcap_idx]
    return gross, tp_first, sl_first, holdcap


def cell_metrics(df_cell: pd.DataFrame, K: int, S: int, regime_map: dict[str, str]) -> dict:
    if len(df_cell) == 0:
        return None
    gross, tp_first, sl_first, holdcap = vectorized_gross_ticks(df_cell, K, S)
    net = gross - TAKER_COST
    n = len(df_cell)
    # cond_WR — TP before SL given one hit
    cond_n = int(tp_first.sum() + sl_first.sum())
    cond_wr = float(tp_first.sum() / max(cond_n, 1))
    wr_pos_net = float((net > 0).mean())
    mean_net = float(net.mean())
    wins = net[net > 0].sum()
    losses = net[net < 0].sum()
    pf = float(wins / abs(losses)) if losses < 0 else (float("inf") if wins > 0 else float("nan"))

    # Daily aggregation for Sharpe & day-concentration
    df_w = df_cell.copy()
    df_w["net"] = net
    daily = df_w.groupby("oot_date")["net"].sum()
    days = daily.index.tolist()
    n_days = len(days)
    n_trades_per_day = n / max(n_days, 1)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else float("nan")
    abs_daily = daily.abs()
    day_conc = float(abs_daily.max() / abs_daily.sum()) if abs_daily.sum() > 0 else float("nan")
    pos_days = int((daily > 0).sum())

    # Regime split
    greens = [d for d in days if regime_map.get(d) == "green"]
    reds = [d for d in days if regime_map.get(d) == "red"]
    flats = [d for d in days if regime_map.get(d) == "flat"]
    daily_g = daily.reindex(greens).dropna() if greens else pd.Series(dtype=float)
    daily_r = daily.reindex(reds).dropna() if reds else pd.Series(dtype=float)
    sharpe_g = float(daily_g.mean()/daily_g.std()*np.sqrt(252)) if len(daily_g) > 1 and daily_g.std() > 0 else float("nan")
    sharpe_r = float(daily_r.mean()/daily_r.std()*np.sqrt(252)) if len(daily_r) > 1 and daily_r.std() > 0 else float("nan")
    if not (np.isnan(sharpe_g) or np.isnan(sharpe_r)):
        denom = max(abs(sharpe_g), abs(sharpe_r), 1e-9)
        regime_asym = abs(sharpe_g - sharpe_r) / denom
    else:
        regime_asym = float("nan")

    return dict(
        K=K, S=S,
        n_trades=int(n),
        n_days=int(n_days),
        n_trades_per_day=float(n_trades_per_day),
        cond_wr=cond_wr,
        wr_pos_net=wr_pos_net,
        mean_ticks_net=mean_net,
        pf=pf,
        sharpe_daily=sharpe,
        day_concentration=day_conc,
        pos_days=pos_days,
        sharpe_green=sharpe_g,
        sharpe_red=sharpe_r,
        regime_asym=regime_asym,
        n_green_days=int(len(daily_g)),
        n_red_days=int(len(daily_r)),
        n_flat_days=int(len(flats)),
        n_holdcap=int(holdcap.sum()),
        n_tp_first=int(tp_first.sum()),
        n_sl_first=int(sl_first.sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos-parquet", required=True)
    ap.add_argument("--walk-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading OOS preds...", flush=True)
    oos = pd.read_parquet(args.oos_parquet)
    print(f"  {len(oos)} rows, heads={oos[['K','S']].drop_duplicates().values.tolist()}", flush=True)

    print("loading walk parquet (for tp/sl + mid_change_hold)...", flush=True)
    walk = pd.read_parquet(args.walk_parquet)
    keep_cols = ["event_id", "oot_date", "side", "mid_change_hold_ticks",
                 "tp1_dt_ns","tp2_dt_ns","tp3_dt_ns","tp4_dt_ns","tp5_dt_ns",
                 "sl1_dt_ns","sl2_dt_ns","sl3_dt_ns","sl4_dt_ns","sl5_dt_ns"]
    walk = walk[keep_cols]
    print("  walk rows:", len(walk))

    # Merge: oos has event_id + oot_date + side + y_pred_fp, etc. Match on event_id+oot_date.
    print("merging...", flush=True)
    # oos has its own side, drop walk's side dup
    merged = oos.merge(walk.drop(columns=["side"]), on=["event_id", "oot_date"], how="left", validate="many_to_one")
    print(f"  merged: {len(merged)} rows; nulls in tp1: {merged['tp1_dt_ns'].isna().sum()}", flush=True)

    # Build regime map from the walk for the OOT dates used
    regime_map = es_regime_per_day(walk)
    print(f"  regime map: {regime_map}", flush=True)

    rows = []
    for K, S in HEADS:
        sub = merged[(merged["K"] == K) & (merged["S"] == S)]
        if len(sub) == 0:
            continue
        for q_pct in Q_GRID:
            # Use day-by-day top-Q to avoid temporal leakage in gating? Simpler: global top-Q on OOS preds (still leak-free since preds are OOS).
            thr = sub["y_pred_fp"].quantile(1.0 - q_pct / 100.0)
            cell = sub[sub["y_pred_fp"] >= thr]
            m = cell_metrics(cell, K, S, regime_map)
            if m is None:
                continue
            m["q_pct"] = q_pct
            m["threshold"] = float(thr)
            rows.append(m)
        # Also a baseline gate: ALL trades (Q=100)
        m_all = cell_metrics(sub, K, S, regime_map)
        if m_all is not None:
            m_all["q_pct"] = 100.0
            m_all["threshold"] = float("nan")
            rows.append(m_all)

    cells = pd.DataFrame(rows)
    cells_path = out_dir / "regrade_cells.parquet"
    cells.to_parquet(cells_path, index=False)
    cells.to_csv(out_dir / "regrade_cells.csv", index=False)
    print(f"wrote {cells_path}  rows={len(cells)}", flush=True)

    # Acceptance: derive GO list
    cells_go = cells[
        (cells["mean_ticks_net"] > 0) &
        (cells["pf"] >= 1.2) &
        (cells["sharpe_daily"] >= 0.5) &
        (cells["regime_asym"] <= 0.50) &
        (cells["day_concentration"] <= 0.70) &
        (cells["n_trades_per_day"] >= 5)
    ].copy()
    cells_go = cells_go.sort_values("sharpe_daily", ascending=False)
    print(f"GO cells: {len(cells_go)}")
    if len(cells_go) > 0:
        print(cells_go[["K","S","q_pct","cond_wr","mean_ticks_net","pf","sharpe_daily","regime_asym","day_concentration","n_trades_per_day","n_trades"]].to_string())

    # Per-head best cell (any), regardless of GO gating, for diagnostic
    print("\nPer-head best cell by Sharpe (any criteria):")
    best_per_head = cells.sort_values("sharpe_daily", ascending=False).groupby(["K","S"]).head(1)
    print(best_per_head[["K","S","q_pct","cond_wr","mean_ticks_net","pf","sharpe_daily","regime_asym","n_trades_per_day","n_trades"]].to_string())

    # Write summary JSON
    summary = {
        "n_cells": int(len(cells)),
        "n_go_cells": int(len(cells_go)),
        "go_cells": cells_go.to_dict(orient="records"),
        "best_per_head": best_per_head.to_dict(orient="records"),
        "regime_map": regime_map,
    }
    (out_dir / "regrade_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("done.")


if __name__ == "__main__":
    main()
