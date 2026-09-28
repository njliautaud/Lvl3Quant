#!/usr/bin/env python3
"""
Regime analysis for K=2 LONG signal — find what distinguishes profitable days
(0223, 0225) from losing day (0226) and silent days (0224, 0227).

Inputs:
  - Predictions: cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz
  - FIFO labels (per OOT day): mbo_events_smart_v3_fifo_labels/<DATE>_fifo_labels.npz
  - Book features (per OOT day, for prices): mbo_book_features/<DATE>_book_features.npz

Outputs:
  - per_day_features.csv
  - regime_gate_results.json
  - <stdout>/log file
"""
from __future__ import annotations
import json
import sys
import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# -------------------- CONSTANTS --------------------
LVL3 = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
FIFO_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
BOOK_DIR = LVL3 / "data/processed/mbo_book_features"
OUT_DIR = LVL3 / "output/v3_3_full_execution_analysis_20260514/regime_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
ES_TICK = 0.25
ES_TICK_VALUE = 12.50
ES_RT_COMM_T = 0.376  # round-trip commission in ticks

# K=2 LONG rule = pred_log_ret_60s BOT 20%  AND  pred_log_ret_5min TOP 20%
# (mirror of the K=2 SHORT rule in the canonical engine)
BAND_FRAC = 0.20

# RTH session bounds in ET (will be converted to nanoseconds)
RTH_OPEN_ET = (9, 30)
RTH_CLOSE_ET = (16, 0)

# Pre-signal windows for microstructure (in seconds)
PRE_WINDOWS_S = [30, 60, 300]  # 30s, 60s, 5min


# -------------------- HELPERS --------------------
def ns_to_et(ts_ns: np.ndarray) -> pd.DatetimeIndex:
    """Convert nanosecond UTC timestamps to ET datetime index."""
    et = pd.to_datetime(ts_ns, utc=True).tz_convert("America/New_York")
    return et


def rth_mask(ts_ns: np.ndarray) -> np.ndarray:
    """RTH = 09:30-16:00 ET."""
    et = ns_to_et(ts_ns)
    hh = et.hour
    mm = et.minute
    after_open = (hh > RTH_OPEN_ET[0]) | ((hh == RTH_OPEN_ET[0]) & (mm >= RTH_OPEN_ET[1]))
    before_close = (hh < RTH_CLOSE_ET[0]) | ((hh == RTH_CLOSE_ET[0]) & (mm < RTH_CLOSE_ET[1]))
    return np.asarray(after_open & before_close)


def time_bucket(ts_ns: np.ndarray) -> np.ndarray:
    """Bucket ET time into 5 strings: open, mid_am, lunch, pm, close."""
    et = ns_to_et(ts_ns)
    hh = np.array(et.hour)
    mm = np.array(et.minute)
    mins = hh * 60 + mm
    out = np.full(len(ts_ns), "unknown", dtype=object)
    out[(mins >= 570) & (mins < 630)] = "open_0930_1030"
    out[(mins >= 630) & (mins < 690)] = "midam_1030_1130"
    out[(mins >= 690) & (mins < 780)] = "lunch_1130_1300"
    out[(mins >= 780) & (mins < 870)] = "pm_1300_1430"
    out[(mins >= 870) & (mins < 960)] = "close_1430_1600"
    return out


def load_predictions(long_mask_from_fifo: np.ndarray):
    """Load model predictions and derive global signal indices per OOT day.

    NOTE: `mask_fifo_tp4sl3_net` in the predictions NPZ is the SHORT-fillable
    universe (matches `tp4sl3_short_filled` from FIFO labels). For K=2 LONG we
    must construct the LONG-fillable mask from the per-day FIFO files.

    `long_mask_from_fifo` is provided by caller: concatenation of
    `tp4sl3_long_filled` from each per-day FIFO file (truncated/aligned to
    predictions length)."""
    d = np.load(PRED_NPZ, allow_pickle=True)
    p60 = d["pred_log_ret_60s"][:].astype(np.float64)
    p5m = d["pred_log_ret_5min"][:].astype(np.float64)
    finite = np.isfinite(p60) & np.isfinite(p5m)
    n = len(p60)
    lm = long_mask_from_fifo[:n] if long_mask_from_fifo is not None else np.ones(n, dtype=bool)
    valid = finite & lm
    return p60, p5m, valid, valid  # both masks = long-fillable universe


def load_fifo_labels(date: str):
    """Load FIFO labels for one day."""
    f = FIFO_DIR / f"{date}_fifo_labels.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return {k: d[k][:] for k in d.files}


ES_PX_REF = 5800.0  # reference price (delta encoding anchor)


def load_book(date: str):
    """Load book features (bid/ask prices + timestamps).

    Bid/ask prices are stored as DELTA TICKS from reference price (ES_PX_REF).
    We convert to absolute price for log-return / drift computations.

    Filters out rows where bid==0 AND ask==0 (pre-session warmup zeros)."""
    f = BOOK_DIR / f"{date}_book_features.npz"
    if not f.exists():
        return None
    d = np.load(f)
    feat = d["features"]
    ts = d["timestamps"]
    bid_dt = feat[:, 0]
    ask_dt = feat[:, 5]
    spread = feat[:, 24]
    # Remove warmup rows where book not yet initialized
    valid = ~((bid_dt == 0) & (ask_dt == 0))
    ts = ts[valid]
    bid_dt = bid_dt[valid]
    ask_dt = ask_dt[valid]
    spread = spread[valid]
    mid_dt = 0.5 * (bid_dt + ask_dt)
    mid_px = ES_PX_REF + mid_dt * ES_TICK
    return {"ts": ts, "bid_dt": bid_dt, "ask_dt": ask_dt, "mid": mid_px,
            "mid_dt": mid_dt, "spread": spread}


# -------------------- PART 1 — Per-day regime features --------------------
def daily_regime_features(date: str, book=None) -> dict:
    """Compute realized vol, drift, range, trend strength using book midprice during RTH."""
    if book is None:
        book = load_book(date)
    if book is None:
        return None
    ts = book["ts"]
    mid_px = book["mid"]      # absolute price
    mid_dt = book["mid_dt"]   # delta from ref in ticks
    spread = book["spread"]

    # Restrict to RTH
    rth = rth_mask(ts)
    if not rth.any():
        return None
    ts_rth = ts[rth]
    mid_px_rth = mid_px[rth]
    mid_dt_rth = mid_dt[rth]
    spread_rth = spread[rth]

    # Sample 1s midprice via last-observation-per-second
    secs = (ts_rth // 1_000_000_000).astype(np.int64)
    df = pd.DataFrame({"sec": secs, "mid": mid_px_rth})
    one_s = df.groupby("sec", sort=True)["mid"].last().values
    if one_s.size < 2:
        realized_vol = 0.0
    else:
        # Drop any non-positive (shouldn't happen, but safe)
        one_s = one_s[one_s > 0]
        lr1 = np.diff(np.log(one_s))
        lr1 = lr1[np.isfinite(lr1)]
        realized_vol = float(np.std(lr1)) if lr1.size > 1 else 0.0

    open_px = float(mid_px_rth[0])
    close_px = float(mid_px_rth[-1])
    high_px = float(np.max(mid_px_rth))
    low_px = float(np.min(mid_px_rth))

    drift_ticks = float((close_px - open_px) / ES_TICK)
    range_ticks = float((high_px - low_px) / ES_TICK)
    trend_strength = drift_ticks / range_ticks if range_ticks > 0 else 0.0

    return {
        "date": date,
        "realized_vol_1s": realized_vol,
        "realized_vol_1s_bps": realized_vol * 1e4,
        "open_px": open_px,
        "close_px": close_px,
        "high_px": high_px,
        "low_px": low_px,
        "drift_ticks": drift_ticks,
        "range_ticks": range_ticks,
        "trend_strength": float(trend_strength),
        "mean_spread_ticks": float(np.nanmean(spread_rth)),
    }


# -------------------- PART 2 / PART 3: K=2 LONG signals --------------------
def assemble_k2_long_signals(p60, p5m, mask60, mask5m, fifo_per_date: dict):
    """
    Build the K=2 LONG signal universe across all 5 OOT days.

    Strategy: the predictions npz contains 241351 samples concatenated across
    OOT days. We map them back to dates using the FIFO label file timestamps.
    Each FIFO label file has window_k and ts_ns for that day's signals; total
    length should equal the predictions length.
    """
    # Concatenate FIFO labels in OOT_DATES order to align with predictions
    all_ts = []
    all_dates = []
    for date in OOT_DATES:
        f = fifo_per_date[date]
        if f is None:
            continue
        n = len(f["ts_ns"])
        all_ts.append(f["ts_ns"])
        all_dates.append(np.array([date] * n))
    ts_all = np.concatenate(all_ts)
    date_all = np.concatenate(all_dates)

    n_total = len(ts_all)
    if n_total != len(p60):
        # Truncate to min — predictions sometimes shorter due to mask/horizon
        n_min = min(n_total, len(p60))
        ts_all = ts_all[:n_min]
        date_all = date_all[:n_min]
        p60 = p60[:n_min]
        p5m = p5m[:n_min]
        mask60 = mask60[:n_min]
        mask5m = mask5m[:n_min]

    valid = mask60 & mask5m
    # K=2 LONG: p60 in BOTTOM 20% (low values) AND p5m in TOP 20% (high values)
    # Use threshold from valid universe.
    p60v = p60[valid]
    p5mv = p5m[valid]
    thr60_bot = float(np.quantile(p60v, BAND_FRAC))      # bottom 20% cutoff
    thr5m_top = float(np.quantile(p5mv, 1 - BAND_FRAC))  # top 20% cutoff

    sig_mask = valid & (p60 <= thr60_bot) & (p5m >= thr5m_top)
    sig_idx = np.where(sig_mask)[0]
    return {
        "ts": ts_all,
        "date": date_all,
        "p60": p60,
        "p5m": p5m,
        "sig_idx": sig_idx,
        "thr60_bot": thr60_bot,
        "thr5m_top": thr5m_top,
        "n_valid": int(valid.sum()),
        "n_signals": int(sig_idx.size),
    }


def pre_signal_microstructure(date: str, sig_ts_ns: np.ndarray, books: dict) -> pd.DataFrame:
    """Compute pre-signal vol/drift/spread for each signal on this day."""
    if date not in books or books[date] is None:
        return None
    b = books[date]
    bk_ts = b["ts"]
    bk_mid = b["mid"]
    bk_spread = b["spread"]

    rows = []
    for ts in sig_ts_ns:
        idx = np.searchsorted(bk_ts, ts, side="right") - 1
        if idx < 1:
            rows.append({})
            continue
        row = {}
        for w in PRE_WINDOWS_S:
            w_ns = w * 1_000_000_000
            start_idx = np.searchsorted(bk_ts, ts - w_ns, side="left")
            if start_idx >= idx:
                row[f"vol_{w}s_bps"] = np.nan
                row[f"drift_{w}s_ticks"] = np.nan
                row[f"mean_spread_{w}s_t"] = np.nan
                continue
            window_mid = bk_mid[start_idx:idx + 1]
            window_spr = bk_spread[start_idx:idx + 1]
            # Sample by 1s to compute vol
            window_ts = bk_ts[start_idx:idx + 1]
            secs = (window_ts // 1_000_000_000).astype(np.int64)
            # last mid per second
            ks, last_idx = np.unique(secs, return_index=False), None
            # Use pandas groupby for clean last-per-second
            tmp = pd.DataFrame({"sec": secs, "mid": window_mid}).groupby("sec")["mid"].last().values
            if tmp.size > 1:
                lr = np.diff(np.log(tmp))
                row[f"vol_{w}s_bps"] = float(np.std(lr) * 1e4)
            else:
                row[f"vol_{w}s_bps"] = 0.0
            # drift
            row[f"drift_{w}s_ticks"] = float((window_mid[-1] - window_mid[0]) / ES_TICK)
            row[f"mean_spread_{w}s_t"] = float(np.nanmean(window_spr))
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------- Per-signal outcomes (K=2 LONG) --------------------
def k2_long_outcomes_per_signal(sig_info: dict, fifo_per_date: dict) -> pd.DataFrame:
    """
    For each K=2 LONG signal, look up its FIFO outcome (long side, tp4sl3).
    Match by index within day.
    """
    sig_idx = sig_info["sig_idx"]
    ts_all = sig_info["ts"]
    date_all = sig_info["date"]

    # Build per-day cumulative offset mapping
    offset_by_date = {}
    cur = 0
    for date in OOT_DATES:
        f = fifo_per_date[date]
        if f is None:
            continue
        offset_by_date[date] = cur
        cur += len(f["ts_ns"])

    rows = []
    for gi in sig_idx:
        date = str(date_all[gi])
        ts = int(ts_all[gi])
        f = fifo_per_date.get(date)
        if f is None:
            continue
        within_day = gi - offset_by_date[date]
        # Use tp4sl3_long_*
        filled = bool(f["tp4sl3_long_filled"][within_day])
        net_t = float(f["tp4sl3_long_net_ticks"][within_day]) if filled else np.nan
        exit_r = str(f["tp4sl3_long_exit_reason"][within_day]) if filled else ""
        hold_ns = int(f["tp4sl3_long_hold_time_ns"][within_day]) if filled else 0
        rows.append({
            "global_idx": int(gi),
            "date": date,
            "ts_ns": ts,
            "p60": float(sig_info["p60"][gi]),
            "p5m": float(sig_info["p5m"][gi]),
            "filled": filled,
            "net_ticks": net_t,
            "exit_reason": exit_r,
            "hold_ns": hold_ns,
        })
    return pd.DataFrame(rows)


# -------------------- Regime gates --------------------
def evaluate_gate(df: pd.DataFrame, gate_name: str, mask: np.ndarray) -> dict:
    """Compute aggregate stats over signals that pass the gate (filled only)."""
    sel = df[mask].copy()
    fills = sel[sel.filled]
    if len(fills) == 0:
        return {
            "name": gate_name,
            "n_signals_passing_gate": int(mask.sum()),
            "n_fills": 0,
            "t_per_fill": 0.0,
            "wr": 0.0,
            "pf": 0.0,
            "total_ticks": 0.0,
            "max_day_concentration": 0.0,
            "per_day_pnl": {},
        }
    total = float(fills.net_ticks.sum())
    n = int(len(fills))
    wr = float((fills.net_ticks > 0).mean())
    wins = fills[fills.net_ticks > 0].net_ticks.sum()
    losses = -fills[fills.net_ticks <= 0].net_ticks.sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    day_pnl = fills.groupby("date").net_ticks.sum().to_dict()
    day_fills = fills.groupby("date").size().to_dict()
    max_day_conc = float(max(day_fills.values()) / n) if n > 0 else 0.0
    return {
        "name": gate_name,
        "n_signals_passing_gate": int(mask.sum()),
        "n_fills": n,
        "t_per_fill": total / n if n else 0.0,
        "wr": wr,
        "pf": pf,
        "total_ticks": total,
        "max_day_concentration": max_day_conc,
        "per_day_pnl": {k: float(v) for k, v in day_pnl.items()},
        "per_day_fills": {k: int(v) for k, v in day_fills.items()},
    }


# -------------------- MAIN --------------------
def main():
    t0 = time.time()
    log = []

    def L(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.append(line)

    L(f"PID {os.getpid()} — K=2 LONG regime analysis")
    L(f"OOT dates: {OOT_DATES}")

    # PART 1: per-day regime features
    L("PART 1 — per-day regime features")
    fifo = {}
    books = {}
    daily = []
    for date in OOT_DATES:
        L(f"  Loading {date}...")
        fifo[date] = load_fifo_labels(date)
        books[date] = load_book(date)
        if books[date] is not None:
            d = daily_regime_features(date, book=books[date])
            d["n_events_fifo"] = len(fifo[date]["ts_ns"]) if fifo[date] else 0
            daily.append(d)
        else:
            L(f"  WARN: no book data for {date}")

    daily_df = pd.DataFrame(daily)

    # PART 2/3: K=2 LONG signal universe + outcomes
    L("PART 2/3 — K=2 LONG signal universe")
    # Build the long-fillable mask by concatenating per-day FIFO long_filled flags
    long_masks = []
    for date in OOT_DATES:
        f = fifo[date]
        if f is None:
            continue
        long_masks.append(f["tp4sl3_long_filled"].astype(bool))
    long_mask_global = np.concatenate(long_masks)
    L(f"  Long-fillable universe size (raw): {long_mask_global.sum()}")
    p60, p5m, mask60, mask5m = load_predictions(long_mask_global)
    L(f"  Predictions length: {len(p60)}  valid_60s: {int(mask60.sum())}  valid_5m: {int(mask5m.sum())}")

    sig_info = assemble_k2_long_signals(p60, p5m, mask60, mask5m, fifo)
    L(f"  thr60_bot={sig_info['thr60_bot']:.6f}  thr5m_top={sig_info['thr5m_top']:.6f}")
    L(f"  K=2 LONG signals: {sig_info['n_signals']} (of {sig_info['n_valid']} valid)")

    # Per-signal outcomes
    sig_df = k2_long_outcomes_per_signal(sig_info, fifo)
    L(f"  Signal-outcomes df rows: {len(sig_df)}")
    L(f"  Per-day signal counts: {sig_df.groupby('date').size().to_dict()}")
    L(f"  Per-day fill counts:  {sig_df[sig_df.filled].groupby('date').size().to_dict()}")
    L(f"  Per-day t/fill:")
    for date, sub in sig_df[sig_df.filled].groupby("date"):
        n = len(sub)
        m = float(sub.net_ticks.mean())
        wr = float((sub.net_ticks > 0).mean())
        L(f"    {date}: n_fill={n}  mean_net_t={m:.3f}  WR={wr:.3f}")

    # Time-of-day bucket
    L("  PART 2 — Intraday bucketing")
    sig_df["bucket"] = time_bucket(sig_df.ts_ns.values)
    fill_df = sig_df[sig_df.filled].copy()
    by_bucket = fill_df.groupby("bucket").agg(
        n=("net_ticks", "size"),
        mean_t=("net_ticks", "mean"),
        wr=("net_ticks", lambda x: (x > 0).mean()),
    ).reset_index()
    L(f"  Fills by bucket:\n{by_bucket.to_string(index=False)}")

    by_day_bucket = (
        fill_df.groupby(["date", "bucket"])
        .agg(n=("net_ticks", "size"), mean_t=("net_ticks", "mean"))
        .reset_index()
    )
    L(f"  Fills by day/bucket:\n{by_day_bucket.to_string(index=False)}")

    # PART 3 — pre-signal microstructure
    L("PART 3 — pre-signal microstructure")
    pre_frames = []
    for date in OOT_DATES:
        day_sigs = sig_df[sig_df.date == date]
        if len(day_sigs) == 0 or books.get(date) is None:
            continue
        sub_pre = pre_signal_microstructure(date, day_sigs.ts_ns.values, books)
        sub_pre.index = day_sigs.index
        pre_frames.append(sub_pre)
    if pre_frames:
        pre_df = pd.concat(pre_frames, axis=0)
        sig_df = pd.concat([sig_df, pre_df], axis=1)
    L(f"  Pre-signal cols added: {[c for c in sig_df.columns if c.startswith(('vol_', 'drift_', 'mean_spread_'))]}")

    # Filled-only stats by day (with pre features)
    fill_df = sig_df[sig_df.filled].copy()
    cmp_cols = [c for c in sig_df.columns if c.startswith(("vol_", "drift_", "mean_spread_"))]
    if cmp_cols:
        L("  Per-day means of pre-signal features (filled only):")
        means_by_day = fill_df.groupby("date")[cmp_cols].mean()
        L("\n" + means_by_day.to_string())

    # Augment daily_df with K=2 LONG outcomes
    k2_by_day = (
        sig_df.groupby("date")
        .agg(n_signals=("filled", "size"), n_fills=("filled", "sum"))
        .reset_index()
    )
    fills_grp = fill_df.groupby("date").agg(
        t_per_fill=("net_ticks", "mean"),
        wr=("net_ticks", lambda x: (x > 0).mean()),
        total_t=("net_ticks", "sum"),
    ).reset_index()
    k2_by_day = k2_by_day.merge(fills_grp, on="date", how="left")
    full_daily = daily_df.merge(k2_by_day, left_on="date", right_on="date", how="left")
    L("Full per-day table:")
    L("\n" + full_daily.to_string(index=False))

    full_daily.to_csv(OUT_DIR / "per_day_features.csv", index=False)
    L(f"Saved per_day_features.csv -> {OUT_DIR / 'per_day_features.csv'}")
    sig_df.to_csv(OUT_DIR / "k2_long_signals_with_regime.csv", index=False)
    L(f"Saved k2_long_signals_with_regime.csv -> {OUT_DIR / 'k2_long_signals_with_regime.csv'}")

    # PART 4 — regime gates
    L("PART 4 — propose regime gates")

    # Build candidate gate masks (over all signals)
    n_sig = len(sig_df)
    ts_arr = sig_df.ts_ns.values
    et = ns_to_et(ts_arr)
    mins_of_day = np.array(et.hour) * 60 + np.array(et.minute)

    # Reference: ALL K=2 LONG signals, no gate
    base = evaluate_gate(sig_df, "BASELINE_K2_LONG_no_gate", np.ones(n_sig, dtype=bool))
    L(f"  BASELINE: n_fill={base['n_fills']}  t/fill={base['t_per_fill']:.3f}  WR={base['wr']:.3f}  PF={base['pf']:.3f}")

    gates = {"BASELINE_K2_LONG_no_gate": base}

    # Gate A: skip opening 30 minutes
    gA = (mins_of_day >= 600)  # >= 10:00 ET
    gates["G_A_no_open_30m"] = evaluate_gate(sig_df, "G_A_no_open_30m", gA)

    # Gate B: avoid choppy / negative-trend days using pre-signal drift_60s_ticks > 0 (must be moving up to take long)
    if "drift_60s_ticks" in sig_df.columns:
        gB = sig_df.drift_60s_ticks.fillna(-1e9).values > 0
        gates["G_B_drift60s_pos"] = evaluate_gate(sig_df, "G_B_drift60s_pos", gB)
        gB2 = sig_df.drift_60s_ticks.fillna(-1e9).values >= 1.0
        gates["G_B2_drift60s_ge1"] = evaluate_gate(sig_df, "G_B2_drift60s_ge1", gB2)
        gB3 = sig_df.drift_60s_ticks.fillna(-1e9).values >= 2.0
        gates["G_B3_drift60s_ge2"] = evaluate_gate(sig_df, "G_B3_drift60s_ge2", gB3)

    if "drift_30s_ticks" in sig_df.columns:
        gB30 = sig_df.drift_30s_ticks.fillna(-1e9).values > 0
        gates["G_B_drift30s_pos"] = evaluate_gate(sig_df, "G_B_drift30s_pos", gB30)
        gB30_2 = sig_df.drift_30s_ticks.fillna(-1e9).values >= 2.0
        gates["G_B2_drift30s_ge2"] = evaluate_gate(sig_df, "G_B2_drift30s_ge2", gB30_2)

    # Gate C: vol regime — require pre-signal vol above some threshold
    if "vol_60s_bps" in sig_df.columns:
        v = sig_df.vol_60s_bps.fillna(0).values
        for thr in [0.5, 1.0, 1.5, 2.0, 3.0]:
            gates[f"G_C_vol60s_ge{thr}bps"] = evaluate_gate(
                sig_df, f"G_C_vol60s_ge{thr}bps", v >= thr
            )

    # Gate D: combined — long-trend day (positive drift) + skip open + min vol
    if "drift_60s_ticks" in sig_df.columns and "vol_60s_bps" in sig_df.columns:
        d60 = sig_df.drift_60s_ticks.fillna(-1e9).values
        v60 = sig_df.vol_60s_bps.fillna(0).values
        gD = (mins_of_day >= 600) & (d60 > 0)
        gates["G_D_no_open_AND_drift60s_pos"] = evaluate_gate(
            sig_df, "G_D_no_open_AND_drift60s_pos", gD
        )
        gD2 = (mins_of_day >= 600) & (d60 >= 1.0)
        gates["G_D2_no_open_AND_drift60s_ge1"] = evaluate_gate(
            sig_df, "G_D2_no_open_AND_drift60s_ge1", gD2
        )
        gD3 = (mins_of_day >= 600) & (d60 >= 2.0)
        gates["G_D3_no_open_AND_drift60s_ge2"] = evaluate_gate(
            sig_df, "G_D3_no_open_AND_drift60s_ge2", gD3
        )

    # Gate E: avoid the 0226 day's characteristic by day-level features
    # Use realized_vol_1s and net day drift to gate at signal time (not lookahead — these are intraday accumulated)
    # We approximate "intraday-so-far drift": price moved since 09:30 by entry time.
    # For each signal, compute (mid_at_signal - mid_at_open) in ticks using book ts.
    L("  Computing intraday-so-far drift at each signal time")
    intra_drift = np.full(n_sig, np.nan)
    intra_drift_pct = np.full(n_sig, np.nan)
    for date, sub in sig_df.groupby("date"):
        b = books.get(date)
        if b is None:
            continue
        ts_b = b["ts"]
        mid_b = b["mid"]
        rth = rth_mask(ts_b)
        if not rth.any():
            continue
        open_idx = np.argmax(rth)  # first True
        open_px = mid_b[open_idx]
        for i, ts in zip(sub.index, sub.ts_ns.values):
            j = np.searchsorted(ts_b, ts, side="right") - 1
            if j < 0:
                continue
            intra_drift[i] = (mid_b[j] - open_px) / ES_TICK
            intra_drift_pct[i] = (mid_b[j] - open_px) / open_px
    sig_df["intraday_drift_ticks"] = intra_drift

    # Gate F: only take long when intraday drift is up
    gF = np.where(np.isnan(intra_drift), False, intra_drift > 0)
    gates["G_F_intraday_drift_pos"] = evaluate_gate(sig_df, "G_F_intraday_drift_pos", gF)
    gF2 = np.where(np.isnan(intra_drift), False, intra_drift >= 5.0)
    gates["G_F2_intraday_drift_ge5"] = evaluate_gate(sig_df, "G_F2_intraday_drift_ge5", gF2)
    gF3 = np.where(np.isnan(intra_drift), False, intra_drift >= 10.0)
    gates["G_F3_intraday_drift_ge10"] = evaluate_gate(sig_df, "G_F3_intraday_drift_ge10", gF3)

    # Composite RECOMMENDED gate — chosen later after inspecting results.
    # We'll pick programmatically the gate maximizing t/fill subject to n_fills >= 20 and WR >= 0.5
    candidates = [(name, g) for name, g in gates.items() if name != "BASELINE_K2_LONG_no_gate"]
    candidates_sorted = sorted(
        candidates,
        key=lambda kv: (kv[1]["n_fills"] >= 20, kv[1]["wr"] >= 0.5, kv[1]["t_per_fill"]),
        reverse=True,
    )
    best_name, best = None, None
    for name, g in candidates_sorted:
        if g["n_fills"] >= 20 and g["wr"] >= 0.5 and g["t_per_fill"] > 0:
            best_name, best = name, g
            break
    if best is None:
        # Relax to: any positive t/fill with n_fills >= 10
        for name, g in candidates_sorted:
            if g["n_fills"] >= 10 and g["t_per_fill"] > 0:
                best_name, best = name, g
                break
    if best is None:
        best_name, best = "G_F_intraday_drift_pos", gates["G_F_intraday_drift_pos"]

    L(f"  RECOMMENDED gate: {best_name}")
    L(f"    n_fills={best['n_fills']}  t/fill={best['t_per_fill']:.3f}  WR={best['wr']:.3f}  PF={best['pf']:.3f}")
    L(f"    per-day fills: {best['per_day_fills']}")
    L(f"    per-day pnl ticks: {best['per_day_pnl']}")

    # Summary JSON
    out_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "oot_dates": OOT_DATES,
        "k2_long_rule": "pred_log_ret_60s BOT 20% AND pred_log_ret_5min TOP 20%",
        "k2_long_thresholds": {
            "thr_pred_log_ret_60s_bot20": sig_info["thr60_bot"],
            "thr_pred_log_ret_5min_top20": sig_info["thr5m_top"],
            "n_valid_universe": sig_info["n_valid"],
            "n_k2_long_signals": sig_info["n_signals"],
        },
        "per_day_features": daily_df.to_dict(orient="records"),
        "k2_long_per_day_outcomes": k2_by_day.fillna(0.0).to_dict(orient="records"),
        "intraday_bucket_summary_all_days": by_bucket.to_dict(orient="records"),
        "intraday_bucket_by_day": by_day_bucket.to_dict(orient="records"),
        "gates_evaluated": gates,
        "recommended_gate": {
            "name": best_name,
            "stats": best,
            "rationale": (
                "Gate maximizing t/fill subject to n_fills >= 20 and WR >= 0.5; "
                "if no candidate meets that, relaxed to n_fills >= 10 with positive t/fill."
            ),
        },
        "notes": [
            "K=2 LONG outcome uses tp4sl3_long_net_ticks from FIFO labels (post-cost label).",
            "Pre-signal vol/drift/spread computed from MBO book features midprice resampled to 1s.",
            "Intraday drift = midprice change in ticks from RTH open through signal time.",
            "ES_RT_COMM_T = 0.376 (commission already baked into FIFO net ticks; do not double-count).",
        ],
        "elapsed_sec": time.time() - t0,
    }

    out_path = OUT_DIR / "regime_gate_results.json"
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else (int(o) if isinstance(o, (np.integer,)) else str(o)))
    L(f"Saved regime_gate_results.json -> {out_path}")
    L(f"Elapsed: {time.time() - t0:.1f}s")

    # Save log
    log_path = Path(os.environ.get("REGIME_LOG", str(OUT_DIR / "regime_analysis.log")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("\n".join(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
