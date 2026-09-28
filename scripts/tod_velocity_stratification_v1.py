#!/usr/bin/env python3
"""
tod_velocity_stratification_v1.py
=================================
HC #488 creativity-mandate axis #3: conditional-on-regime profit decomposition.

Stratify v3.4.2 OOT predictions by:
    time-of-day (5 buckets) x trade-tape-velocity-decile x side x horizon x confidence-bucket
and surface (time-of-day x velocity) pockets where the model clears HC #428
deploy gates even though pooled analysis fails.

Inputs
------
- output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_{YYYYMMDD}.npz
    pred_log_ret_{1,5,10,30}s   (signed predicted move in TICKS per HC #486 flag)
    target_log_ret_{1,5,10,30}s (realized move in TICKS per HC #486 flag)
    sample_dates                (per-row date string)
- data/processed/mbo_events/{YYYYMMDD}_mbo_events.npz
    events[:,1] == event_type_id, action_encoding: A=0,C=1,M=2,T=3,F=4
    timestamps  (UTC nanoseconds)

Alignment
---------
n_pred == ceil(n_mbo/250) (stride=250 confirmed by ratio ~250.3 across days).
So prediction row i corresponds to MBO event index i*250 (its timestamp).
Trade-tape velocity at prediction-row i = count of MBO events with
event_type==T in the 5-second window strictly BEFORE that timestamp.
(Causal: t-5s .. t-1ns.)

Buckets
-------
TOD (ET):
    open       09:30-10:30
    mid_am     10:30-12:00
    midday     12:00-14:00
    late_pm    14:00-15:00
    close      15:00-16:00
    (rows outside 09:30-16:00 ET are dropped)
Velocity: per-day decile rank of trade-tape velocity (0..9 within day).
Side: long (pred>0), short (pred<0).
Horizon: 1s, 5s, 10s, 30s.
Confidence-top-bucket: top1%, top5%, top10% by |pred| within (day, side, horizon).

Trade economics (passive limit)
-------------------------------
    net_ticks_per_trade = signed_pnl_ticks - 0.376
where signed_pnl_ticks = +target if long else -target  (target already in ticks).

HC #428 deploy gates
--------------------
    net  > +0.10 ticks
    Sharpe > 0.3
    PF > 1.1
    profitable >= 60% of trading days in cell
    regime_imbalance < 0.50  (|Sh_green - Sh_red| / max(|Sh_green|,|Sh_red|))
    n_trades >= 50 (stat floor)

Outputs (under output/tod_velocity_stratification_v1/)
------------------------------------------------------
    stratified_summary.csv
    winning_cells.txt
    closest_miss.json
    REPORT.md
    .regen_complete.json
"""
import json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------
OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/tod_velocity_stratification_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_RT_COMMISSION_TICKS = 0.376    # passive limit round-trip
STRIDE = 250                      # MBO->pred subsample
VELOCITY_WINDOW_NS = 5_000_000_000  # 5 seconds in ns
EVENT_TYPE_TRADE = 3              # action_encoding 'T' == 3
SKIP_DATES = {"20260308", "20260315"}  # per task spec

HORIZONS = ["1s", "5s", "10s", "30s"]
TOD_BUCKETS = [
    ("open",     9*60+30, 10*60+30),
    ("mid_am",   10*60+30, 12*60),
    ("midday",   12*60,    14*60),
    ("late_pm",  14*60,    15*60),
    ("close",    15*60,    16*60),
]
CONF_BUCKETS = [
    ("top1pc",  0.01),
    ("top5pc",  0.05),
    ("top10pc", 0.10),
]

# HC #428 gates
GATE_NET      = 0.10
GATE_SHARPE   = 0.3
GATE_PF       = 1.1
GATE_PROFDAYS = 0.60
GATE_IMBAL    = 0.50
MIN_TRADES    = 50

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def tod_label(et_minute_of_day: np.ndarray) -> np.ndarray:
    out = np.full(et_minute_of_day.shape, "outside", dtype=object)
    for name, lo, hi in TOD_BUCKETS:
        m = (et_minute_of_day >= lo) & (et_minute_of_day < hi)
        out[m] = name
    return out


def compute_velocity(pred_ts_ns: np.ndarray, mbo_ts_ns: np.ndarray, mbo_type: np.ndarray) -> np.ndarray:
    """
    For each prediction timestamp pred_ts_ns[i], count MBO trade events
    (type==T) with timestamp in [pred_ts_ns[i] - 5s, pred_ts_ns[i] - 1].
    Strictly causal.
    """
    # Filter mbo to trade-only timestamps (sorted) once
    trade_mask = (mbo_type == EVENT_TYPE_TRADE)
    trade_ts = mbo_ts_ns[trade_mask]
    # searchsorted: count = idx_right - idx_left
    # left bound: pred_ts - 5s   (inclusive)
    # right bound: pred_ts - 1ns (exclusive of pred_ts)
    lo = pred_ts_ns - VELOCITY_WINDOW_NS
    # searchsorted with 'left' gives first idx >= value
    left_idx  = np.searchsorted(trade_ts, lo,           side="left")
    right_idx = np.searchsorted(trade_ts, pred_ts_ns,   side="left")
    return (right_idx - left_idx).astype(np.int32)


def day_decile(values: np.ndarray) -> np.ndarray:
    """Decile rank 0..9 within array (per-day call). Ties broken arbitrarily."""
    n = len(values)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    # rank then bucket. argsort twice gives rank.
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    dec = (ranks * 10) // n
    dec[dec == 10] = 9
    return dec.astype(np.int8)


def cell_metrics(net_ticks: np.ndarray, day_ids: np.ndarray) -> dict:
    """All metrics for a cell. net_ticks already includes commission."""
    n = len(net_ticks)
    if n == 0:
        return None
    mean = float(np.mean(net_ticks))
    sd   = float(np.std(net_ticks, ddof=1)) if n > 1 else 0.0
    sharpe = (mean / sd) * np.sqrt(n) if sd > 0 else 0.0
    wins = net_ticks > 0
    wr = float(np.mean(wins))
    gp = float(np.sum(net_ticks[wins])) if wins.any() else 0.0
    gl = float(-np.sum(net_ticks[~wins])) if (~wins).any() else 0.0
    pf = (gp / gl) if gl > 1e-9 else (np.inf if gp > 0 else 0.0)
    # per-day profitability
    days = np.unique(day_ids)
    day_means = np.array([np.mean(net_ticks[day_ids == d]) for d in days])
    prof_days_frac = float(np.mean(day_means > 0)) if len(days) > 0 else 0.0
    return {
        "n_trades":      n,
        "n_days":        int(len(days)),
        "net_ticks":     mean,
        "sharpe":        float(sharpe),
        "wr":            wr,
        "pf":            pf,
        "prof_days":     prof_days_frac,
    }


# -----------------------------------------------------------------------------
# Load + merge per day
# -----------------------------------------------------------------------------
def load_day(date: str) -> pd.DataFrame | None:
    pred_path = OOT_DIR / f"oot_{date}.npz"
    mbo_path  = MBO_DIR / f"{date}_mbo_events.npz"
    if not pred_path.exists() or not mbo_path.exists():
        return None
    pred = np.load(pred_path)
    mbo  = np.load(mbo_path)
    n_pred = pred["pred_log_ret_1s"].shape[0]
    mbo_ts = mbo["timestamps"]
    mbo_ev = mbo["events"][:, 1].astype(np.int8)  # event_type_id
    # Align: stride 250
    expected_n = mbo_ts.shape[0] // STRIDE + (1 if mbo_ts.shape[0] % STRIDE else 0)
    # Take every STRIDE-th MBO event timestamp; align to n_pred length
    pred_ts = mbo_ts[::STRIDE][:n_pred]
    if len(pred_ts) != n_pred:
        print(f"  WARN {date}: pred_ts len {len(pred_ts)} != n_pred {n_pred}; skipping", flush=True)
        return None
    # Velocity
    vel = compute_velocity(pred_ts, mbo_ts, mbo_ev)
    # ET minute-of-day from UTC ns
    ts_et = pd.to_datetime(pred_ts, unit="ns", utc=True).tz_convert("America/New_York")
    minute_of_day = ts_et.hour * 60 + ts_et.minute
    df = pd.DataFrame({
        "date":   date,
        "ts_ns":  pred_ts,
        "min_et": np.asarray(minute_of_day, dtype=np.int32),
        "vel":    vel,
    })
    for h in HORIZONS:
        df[f"pred_{h}"]   = pred[f"pred_log_ret_{h}"]
        df[f"target_{h}"] = pred[f"target_log_ret_{h}"]
        df[f"mask_{h}"]   = pred[f"mask_log_ret_{h}"]
    # TOD label + filter to RTH
    df["tod"] = tod_label(df["min_et"].values)
    df = df[df["tod"] != "outside"].reset_index(drop=True)
    if df.empty:
        return None
    # Velocity decile within-day
    df["vel_dec"] = day_decile(df["vel"].values)
    return df


# -----------------------------------------------------------------------------
# Stratification
# -----------------------------------------------------------------------------
def stratify(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    day_ids = df["date"].astype(str).values
    # day -> int id
    udates = np.unique(day_ids)
    day_id_map = {d: i for i, d in enumerate(udates)}
    day_int = np.array([day_id_map[d] for d in day_ids], dtype=np.int32)

    for horizon in HORIZONS:
        pred_col   = f"pred_{horizon}"
        target_col = f"target_{horizon}"
        mask_col   = f"mask_{horizon}"
        valid = df[mask_col].values > 0.5
        preds   = df[pred_col].values[valid]
        targets = df[target_col].values[valid]
        tod     = df["tod"].values[valid]
        vdec    = df["vel_dec"].values[valid]
        dints   = day_int[valid]
        # For each confidence bucket: compute per-(day, side) threshold on |pred|
        for conf_name, conf_frac in CONF_BUCKETS:
            # threshold per-day per-side
            keep_mask = np.zeros(len(preds), dtype=bool)
            for d_i in np.unique(dints):
                d_sel = (dints == d_i)
                for side_name, side_sel in (("long", preds > 0), ("short", preds < 0)):
                    sel = d_sel & side_sel
                    if sel.sum() < 10:
                        continue
                    thr = np.quantile(np.abs(preds[sel]), 1.0 - conf_frac)
                    keep_mask |= sel & (np.abs(preds) >= thr)
            if keep_mask.sum() == 0:
                continue
            kp = preds[keep_mask]
            kt = targets[keep_mask]
            ktod = tod[keep_mask]
            kvdec = vdec[keep_mask]
            kday = dints[keep_mask]
            for side_name, side_sel in (("long", kp > 0), ("short", kp < 0)):
                if side_sel.sum() == 0:
                    continue
                sign = 1.0 if side_name == "long" else -1.0
                # buckets
                tod_s = ktod[side_sel]
                vd_s  = kvdec[side_sel]
                t_s   = kt[side_sel]
                d_s   = kday[side_sel]
                for tod_name, _, _ in TOD_BUCKETS:
                    tmask = (tod_s == tod_name)
                    if tmask.sum() == 0:
                        continue
                    for vd in range(10):
                        cellmask = tmask & (vd_s == vd)
                        n = cellmask.sum()
                        if n < MIN_TRADES:
                            continue
                        signed_pnl = sign * t_s[cellmask]
                        net = signed_pnl - ES_RT_COMMISSION_TICKS
                        mm = cell_metrics(net, d_s[cellmask])
                        if mm is None:
                            continue
                        rows.append({
                            "horizon": horizon, "conf": conf_name, "side": side_name,
                            "tod": tod_name, "vel_dec": int(vd),
                            **mm,
                        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Regime classification + gates
# -----------------------------------------------------------------------------
def classify_green_red_days(df: pd.DataFrame) -> dict[str, str]:
    """ES close-to-close (we proxy with sum target_log_ret_30s for first horizon, but
    we'll use the cumulative realized 1s tick target as a proxy of intraday drift)."""
    # Per HC #428: green/red/flat by ES close-to-close. We don't have raw ES OHLC
    # readily; proxy with cumulative sum of target_log_ret_1s ticks across the RTH day
    # (since these are tick moves at each event, summing gives net move).
    out = {}
    for d, sub in df.groupby("date"):
        valid = sub["mask_1s"].values > 0.5
        cum = float(np.sum(sub.loc[valid, "target_1s"].values))
        if cum > 5:   out[d] = "green"
        elif cum < -5: out[d] = "red"
        else:          out[d] = "flat"
    return out


def regime_imbalance_for_cell(df_cell_rows: pd.DataFrame) -> float:
    """Placeholder: cell metrics already integrate days; we compute imbalance
    from per-day means stratified by regime label."""
    # Not used at row-level here; we compute imbalance per cell in main pipeline
    # by re-running per-day means. See main().
    return np.nan


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("[1/5] discovering OOT dates", flush=True)
    files = sorted(OOT_DIR.glob("oot_*.npz"))
    all_dates = [f.stem.replace("oot_", "") for f in files]
    dates = [d for d in all_dates if d not in SKIP_DATES]
    print(f"      {len(dates)} valid OOT dates (skipped {sorted(SKIP_DATES)})", flush=True)

    print("[2/5] loading + computing velocity per day", flush=True)
    per_day = []
    for i, d in enumerate(dates):
        try:
            df_d = load_day(d)
            if df_d is None or df_d.empty:
                continue
            per_day.append(df_d)
            if (i + 1) % 5 == 0 or i + 1 == len(dates):
                print(f"  loaded {i+1}/{len(dates)}  ({time.time()-t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  ERROR loading {d}: {e}", flush=True)
            traceback.print_exc()
    if not per_day:
        print("ERROR: no days loaded", flush=True)
        sys.exit(1)
    df = pd.concat(per_day, ignore_index=True)
    print(f"      merged: {len(df):,} prediction rows across {df['date'].nunique()} days", flush=True)

    print("[3/5] classifying green/red regime days (proxy: cumulative 1s ticks)", flush=True)
    regime_map = classify_green_red_days(df)
    n_g = sum(v == "green" for v in regime_map.values())
    n_r = sum(v == "red"   for v in regime_map.values())
    n_f = sum(v == "flat"  for v in regime_map.values())
    print(f"      green={n_g} red={n_r} flat={n_f}", flush=True)

    print("[4/5] stratifying by (horizon x conf x side x tod x vel_dec)", flush=True)
    df["date"] = df["date"].astype(str)
    summary = stratify(df)
    print(f"      {len(summary)} cells (n_trades>={MIN_TRADES}, gated by mask)", flush=True)

    # ------------------------------------------------------------------
    # Compute regime_imbalance per cell:
    #   For each cell config, split its trades by date regime and recompute Sharpe.
    # ------------------------------------------------------------------
    print("      computing per-cell regime_imbalance ...", flush=True)
    # Re-build a row-level table identical to stratify, then group.
    # Cheaper path: replay the per-cell computation.
    # We'll redo it inline.
    day_ids = df["date"].astype(str).values
    udates = np.unique(day_ids)
    day_id_map = {d: i for i, d in enumerate(udates)}
    day_int = np.array([day_id_map[d] for d in day_ids], dtype=np.int32)
    day_regime = np.array([regime_map.get(d, "flat") for d in udates])  # by id

    imbal_list = []
    sh_green_list = []
    sh_red_list   = []
    for _, row in summary.iterrows():
        h = row["horizon"]; conf = row["conf"]; side = row["side"]
        tod_name = row["tod"]; vd = int(row["vel_dec"])
        valid = df[f"mask_{h}"].values > 0.5
        preds   = df[f"pred_{h}"].values[valid]
        targets = df[f"target_{h}"].values[valid]
        tods    = df["tod"].values[valid]
        vdecs   = df["vel_dec"].values[valid]
        dints   = day_int[valid]
        # confidence threshold per-day per-side
        side_sign = 1.0 if side == "long" else -1.0
        if side == "long":
            side_mask = preds > 0
        else:
            side_mask = preds < 0
        conf_frac = {"top1pc":0.01,"top5pc":0.05,"top10pc":0.10}[conf]
        keep = np.zeros(len(preds), dtype=bool)
        for d_i in np.unique(dints):
            sel = (dints == d_i) & side_mask
            if sel.sum() < 10:
                continue
            thr = np.quantile(np.abs(preds[sel]), 1.0 - conf_frac)
            keep |= sel & (np.abs(preds) >= thr)
        cell = keep & (tods == tod_name) & (vdecs == vd)
        if cell.sum() == 0:
            imbal_list.append(np.nan); sh_green_list.append(np.nan); sh_red_list.append(np.nan)
            continue
        signed = side_sign * targets[cell] - ES_RT_COMMISSION_TICKS
        c_days = dints[cell]
        c_regime = np.array([day_regime[di] for di in c_days])
        def sh(arr):
            if len(arr) < 2: return 0.0
            s = np.std(arr, ddof=1)
            return float((np.mean(arr)/s)*np.sqrt(len(arr))) if s>0 else 0.0
        sh_g = sh(signed[c_regime == "green"])
        sh_r = sh(signed[c_regime == "red"])
        denom = max(abs(sh_g), abs(sh_r))
        imb = abs(sh_g - sh_r) / denom if denom > 1e-9 else np.nan
        imbal_list.append(imb); sh_green_list.append(sh_g); sh_red_list.append(sh_r)

    summary["sharpe_green"]      = sh_green_list
    summary["sharpe_red"]        = sh_red_list
    summary["regime_imbalance"]  = imbal_list

    # ------------------------------------------------------------------
    # Apply HC #428 gates
    # ------------------------------------------------------------------
    def gate_row(r):
        fails = []
        if r["net_ticks"] <= GATE_NET:                     fails.append(("net",     r["net_ticks"],     GATE_NET))
        if r["sharpe"]    <= GATE_SHARPE:                  fails.append(("sharpe",  r["sharpe"],        GATE_SHARPE))
        if not np.isfinite(r["pf"]) or r["pf"] <= GATE_PF: fails.append(("pf",      r["pf"],            GATE_PF))
        if r["prof_days"] < GATE_PROFDAYS:                 fails.append(("profdays",r["prof_days"],     GATE_PROFDAYS))
        if not np.isfinite(r["regime_imbalance"]) or r["regime_imbalance"] >= GATE_IMBAL:
            fails.append(("imbalance", r["regime_imbalance"], GATE_IMBAL))
        return fails

    fails_list = [gate_row(r) for _, r in summary.iterrows()]
    summary["n_gates_failed"] = [len(f) for f in fails_list]
    summary["passes_all"]     = summary["n_gates_failed"] == 0

    # Sort by Sharpe (desc) for reporting
    summary = summary.sort_values(["passes_all", "sharpe"], ascending=[False, False]).reset_index(drop=True)

    # Save CSV
    csv_path = OUT_DIR / "stratified_summary.csv"
    summary.to_csv(csv_path, index=False)

    winners = summary[summary["passes_all"]].copy()
    win_path = OUT_DIR / "winning_cells.txt"
    with open(win_path, "w") as fp:
        if winners.empty:
            fp.write("NONE - no cell passes all HC #428 gates\n")
        else:
            fp.write(f"{len(winners)} cells pass HC #428 gates:\n\n")
            cols = ["horizon","conf","side","tod","vel_dec","n_trades","n_days",
                    "net_ticks","sharpe","wr","pf","prof_days","regime_imbalance"]
            fp.write(winners[cols].to_string(index=False) + "\n")

    # Closest miss: cells that pass MIN_TRADES, sorted by # gates failed asc then sharpe desc
    miss = summary[(~summary["passes_all"]) & (summary["n_trades"] >= MIN_TRADES)].copy()
    miss = miss.sort_values(["n_gates_failed", "sharpe"], ascending=[True, False]).head(5)
    closest_records = []
    for i, r in miss.iterrows():
        f = gate_row(r)
        closest_records.append({
            "horizon": r["horizon"], "conf": r["conf"], "side": r["side"],
            "tod": r["tod"], "vel_dec": int(r["vel_dec"]),
            "n_trades": int(r["n_trades"]),
            "net_ticks": float(r["net_ticks"]),
            "sharpe": float(r["sharpe"]), "pf": float(r["pf"]) if np.isfinite(r["pf"]) else None,
            "wr": float(r["wr"]), "prof_days": float(r["prof_days"]),
            "regime_imbalance": (None if not np.isfinite(r["regime_imbalance"]) else float(r["regime_imbalance"])),
            "n_gates_failed": int(r["n_gates_failed"]),
            "failed_gates": [
                {"gate": g, "value": (None if (isinstance(v, float) and not np.isfinite(v)) else float(v)),
                 "threshold": float(thr)} for (g, v, thr) in f
            ],
        })
    (OUT_DIR / "closest_miss.json").write_text(json.dumps(closest_records, indent=2))

    # ----------------------------------------------------------
    # Structural pattern detection
    # ----------------------------------------------------------
    # Top-quartile cells by Sharpe -> dominant TOD / velocity / side
    top_q = summary[summary["n_trades"] >= MIN_TRADES].nlargest(
        max(1, len(summary)//4), "sharpe")
    def top_freq(col):
        if len(top_q) == 0: return {}
        vc = top_q[col].value_counts(normalize=True)
        return {str(k): float(v) for k, v in vc.items()}
    pat_tod  = top_freq("tod")
    pat_side = top_freq("side")
    pat_vd   = top_freq("vel_dec")
    pat_hor  = top_freq("horizon")

    # Velocity-edge correlation: spearman between vel_dec and sharpe per (h,side,conf,tod)
    from scipy.stats import spearmanr
    vel_corrs = []
    for (h, s, c, t), grp in summary.groupby(["horizon", "side", "conf", "tod"]):
        if len(grp) >= 5:
            rho, _ = spearmanr(grp["vel_dec"], grp["sharpe"])
            if np.isfinite(rho):
                vel_corrs.append({"horizon": h, "side": s, "conf": c, "tod": t, "rho": float(rho), "n_cells": int(len(grp))})
    vel_corrs_sorted = sorted(vel_corrs, key=lambda x: -abs(x["rho"]))[:10]

    # Verdict
    accept = len(winners) > 0

    # Report
    report = []
    report.append("# TOD x Velocity Stratification v1 - REPORT\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    report.append(f"Source: v3.4.2 OOT preds, {df['date'].nunique()} days, "
                  f"{len(df):,} RTH prediction rows (skipped {sorted(SKIP_DATES)})\n")
    report.append(f"Regime mix: green={n_g}, red={n_r}, flat={n_f}\n\n")
    report.append(f"## VERDICT: {'ACCEPT' if accept else 'REJECT'}\n")
    if accept:
        report.append(f"{len(winners)} cell(s) pass HC #428 gates. See winning_cells.txt.\n")
        report.append("Top winner:\n")
        w0 = winners.iloc[0]
        report.append(f"  horizon={w0['horizon']} conf={w0['conf']} side={w0['side']} "
                      f"tod={w0['tod']} vel_dec={int(w0['vel_dec'])}\n")
        report.append(f"  n={int(w0['n_trades'])} days={int(w0['n_days'])} "
                      f"net={w0['net_ticks']:+.3f}t Sh={w0['sharpe']:.2f} "
                      f"PF={w0['pf']:.2f} WR={w0['wr']:.1%} "
                      f"profdays={w0['prof_days']:.1%} imbal={w0['regime_imbalance']:.2f}\n\n")
    else:
        report.append("No cell clears all HC #428 deploy gates.\n\n")
        if closest_records:
            cr = closest_records[0]
            report.append("Closest miss:\n")
            report.append(f"  {cr['horizon']} {cr['conf']} {cr['side']} {cr['tod']} vd={cr['vel_dec']}: "
                          f"n={cr['n_trades']} net={cr['net_ticks']:+.3f}t Sh={cr['sharpe']:.2f} "
                          f"fails={cr['n_gates_failed']} -> "
                          f"{[g['gate'] for g in cr['failed_gates']]}\n\n")
    report.append("## Structural patterns (top-quartile by Sharpe)\n")
    report.append(f"- TOD frequency:      {pat_tod}\n")
    report.append(f"- Side frequency:     {pat_side}\n")
    report.append(f"- Velocity decile:    {pat_vd}\n")
    report.append(f"- Horizon frequency:  {pat_hor}\n\n")
    report.append("## Velocity-edge correlation (Spearman rho between vel_dec and Sharpe)\n")
    if vel_corrs_sorted:
        report.append("Top |rho| slices:\n")
        for vc in vel_corrs_sorted:
            report.append(f"  {vc['horizon']:>4s} {vc['side']:>5s} {vc['conf']:>7s} {vc['tod']:>8s}: "
                          f"rho={vc['rho']:+.3f}  (n_cells={vc['n_cells']})\n")
    else:
        report.append("(not enough cells per slice)\n")
    report.append("\n## Interpretation\n")
    # Does pooled analysis hide what stratification found?
    # Pooled = grouping everything (no stratification). We computed cells; if winners exist,
    # stratification revealed pockets pooled analysis missed.
    if accept:
        report.append("- Stratification REVEALED pockets of deployable edge that pooled analysis missed.\n")
        # Check TOD concentration
        winner_tods = winners["tod"].value_counts()
        report.append(f"- Winning cells TOD concentration: {dict(winner_tods)}\n")
        winner_sides = winners["side"].value_counts()
        report.append(f"- Winning cells side concentration: {dict(winner_sides)}\n")
    else:
        report.append("- Even fine-grained TOD x velocity x side x horizon x confidence stratification "
                      "does not produce a deploy-grade pocket.\n")
        report.append("- This is consistent with the prior pooled HC #428 reject and the conformal-wrapper "
                      "finding that short-side calibration is broken: no time/velocity slice rescues it.\n")
    report.append("\n## Recommended next move\n")
    if accept:
        report.append("- Lock surviving cell(s) and run shadow paper-trade for 5 OOT days to confirm.\n")
        report.append("- Re-run HC #485 R5 regen on the locked cell config for canonical record.\n")
    else:
        report.append("- v3.4.2 OOT preds appear unrescuable by stratification alone. Next HC #488 axis: "
                      "feature-attribution probe (which input subset, if any, drives the long-side asymmetry?) "
                      "or move to a different signal version. Stop stratifying v3.4.2 raw preds.\n")
    (OUT_DIR / "REPORT.md").write_text("".join(report))

    # ----------------------------------------------------------
    # HC #485 R5 regen stamp
    # ----------------------------------------------------------
    stamp = {
        "script":         "scripts/tod_velocity_stratification_v1.py",
        "completed_at":   time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_seconds":   round(time.time() - t0, 1),
        "n_pred_rows":    int(len(df)),
        "n_oot_days":     int(df["date"].nunique()),
        "skipped_dates":  sorted(SKIP_DATES),
        "horizons":       HORIZONS,
        "tod_buckets":    [b[0] for b in TOD_BUCKETS],
        "conf_buckets":   [c[0] for c in CONF_BUCKETS],
        "min_trades":     MIN_TRADES,
        "n_cells":        int(len(summary)),
        "n_winners":      int(len(winners)),
        "verdict":        "ACCEPT" if accept else "REJECT",
        "regime_mix":     {"green": n_g, "red": n_r, "flat": n_f},
        "outputs": {
            "summary_csv":   str(csv_path.relative_to(Path("/home/jupiter/Lvl3Quant"))),
            "winning_cells": str(win_path.relative_to(Path("/home/jupiter/Lvl3Quant"))),
            "closest_miss":  str((OUT_DIR / "closest_miss.json").relative_to(Path("/home/jupiter/Lvl3Quant"))),
            "report":        str((OUT_DIR / "REPORT.md").relative_to(Path("/home/jupiter/Lvl3Quant"))),
        },
    }
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(stamp, indent=2))

    print(f"[5/5] Done in {time.time()-t0:.1f}s. Verdict: {'ACCEPT' if accept else 'REJECT'}. "
          f"{len(winners)} winners / {len(summary)} cells.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
