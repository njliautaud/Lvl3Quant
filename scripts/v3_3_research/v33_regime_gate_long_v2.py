#!/usr/bin/env python3
"""
K=2 LONG regime gate analysis (v2) — written fresh, self-contained.

Per HC #372 + task brief:
  K=2 LONG entry rule (uses the K=2 SHORT mask but enters LONG):
    pred_log_ret_60s  in TOP 20% of valid universe  (user: "POS top 20%")
    pred_log_ret_5min in BOT 20% of valid universe  (user: "NEG top 20%")

  P&L source: per-event tp4sl3_long_net_ticks from FIFO label files
              (commission already netted, FIFO replay engine).

Outputs to: output/v3_3_full_execution_analysis_20260514/regime_analysis/
  - per_day_features_v2.csv
  - k2_long_signals_with_regime_v2.csv
  - regime_gate_results_v2.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONSTANTS
# ============================================================================
LVL3 = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
FIFO_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
BOOK_DIR = LVL3 / "data/processed/mbo_book_features"
OUT_DIR = LVL3 / "output/v3_3_full_execution_analysis_20260514/regime_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]

ES_TICK = 0.25
ES_RT_COMM_T = 0.376  # round-trip commission in ticks (already in FIFO net ticks)

BAND_FRAC = 0.20

RTH_OPEN_MIN = 9 * 60 + 30   # 09:30 ET
RTH_CLOSE_MIN = 16 * 60      # 16:00 ET
PRE_WINDOWS_S = [30, 60, 300]

# Book features columns (from raw NPZ): col 0 = bid_price_1 (delta ticks),
# col 5 = ask_price_1 (delta ticks), col 24 = spread (ticks).
COL_BID1 = 0
COL_ASK1 = 5
COL_SPREAD = 24

# ============================================================================
# UTILITIES
# ============================================================================
def log_factory(log_file: Path | None):
    buf = []
    def L(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        buf.append(line)
    def flush():
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text("\n".join(buf) + "\n")
    return L, flush


def ns_to_et_minutes(ts_ns: np.ndarray) -> np.ndarray:
    """Return minute-of-day in ET for each ns timestamp (vectorized)."""
    et = pd.to_datetime(ts_ns, utc=True).tz_convert("America/New_York")
    return np.asarray(et.hour) * 60 + np.asarray(et.minute)


def rth_mask_from_min(min_of_day: np.ndarray) -> np.ndarray:
    return (min_of_day >= RTH_OPEN_MIN) & (min_of_day < RTH_CLOSE_MIN)


def hour_bucket(min_of_day: np.ndarray) -> np.ndarray:
    """5 buckets across RTH."""
    out = np.full(len(min_of_day), "non_rth", dtype=object)
    out[(min_of_day >= 570) & (min_of_day < 630)] = "open_0930_1030"
    out[(min_of_day >= 630) & (min_of_day < 690)] = "midam_1030_1130"
    out[(min_of_day >= 690) & (min_of_day < 780)] = "lunch_1130_1300"
    out[(min_of_day >= 780) & (min_of_day < 870)] = "pm_1300_1430"
    out[(min_of_day >= 870) & (min_of_day < 960)] = "close_1430_1600"
    return out


# ============================================================================
# DATA LOADERS
# ============================================================================
def load_fifo(date: str) -> dict | None:
    f = FIFO_DIR / f"{date}_fifo_labels.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return {k: d[k][:] for k in d.files}


def load_book_rth_only(date: str) -> dict | None:
    """Load book features and filter to in-session rows (delta ticks only,
    drop warmup rows whose bid/ask are still in raw-price form).

    The book NPZ stores bid/ask as DELTA ticks from a reference price.
    During warmup the raw prices appear (e.g. 27600, then 27599..). We
    detect warmup by abs(bid_delta) > 10_000 ticks, which is impossible
    for real ES intraday moves.
    """
    f = BOOK_DIR / f"{date}_book_features.npz"
    if not f.exists():
        return None
    d = np.load(f)
    feat = d["features"]
    ts = d["timestamps"]
    bid = feat[:, COL_BID1].astype(np.float64)
    ask = feat[:, COL_ASK1].astype(np.float64)
    spr = feat[:, COL_SPREAD].astype(np.float64)
    # Drop warmup
    sane = (np.abs(bid) < 10_000) & (np.abs(ask) < 10_000) & (bid != 0) & (ask != 0)
    ts = ts[sane]
    bid = bid[sane]
    ask = ask[sane]
    spr = spr[sane]
    if len(ts) == 0:
        return None
    mid_dt = 0.5 * (bid + ask)  # in ticks (delta)
    # Restrict to RTH
    mod = ns_to_et_minutes(ts)
    rth = rth_mask_from_min(mod)
    return {
        "ts": ts[rth],
        "bid_dt": bid[rth],
        "ask_dt": ask[rth],
        "mid_dt": mid_dt[rth],
        "spread": spr[rth],
        "min_of_day": mod[rth],
    }


# ============================================================================
# PART 1 — per-day regime features
# ============================================================================
def daily_regime_features(date: str, b: dict) -> dict:
    """All in TICKS (delta). open/close are mid_dt; vol from 1s mid log-return."""
    ts = b["ts"]
    mid_dt = b["mid_dt"]
    secs = (ts // 1_000_000_000).astype(np.int64)
    df = pd.DataFrame({"sec": secs, "mid": mid_dt})
    one_s = df.groupby("sec", sort=True)["mid"].last().values
    if one_s.size < 3:
        vol_1s_ticks_std = 0.0
    else:
        diffs = np.diff(one_s)
        diffs = diffs[np.isfinite(diffs)]
        vol_1s_ticks_std = float(np.std(diffs)) if diffs.size > 1 else 0.0

    open_dt = float(mid_dt[0])
    close_dt = float(mid_dt[-1])
    high_dt = float(np.max(mid_dt))
    low_dt = float(np.min(mid_dt))
    drift_ticks = close_dt - open_dt
    range_ticks = high_dt - low_dt
    trend = drift_ticks / range_ticks if range_ticks > 0 else 0.0
    return {
        "date": date,
        "vol_1s_ticks_std": vol_1s_ticks_std,
        "open_dt_ticks": open_dt,
        "close_dt_ticks": close_dt,
        "high_dt_ticks": high_dt,
        "low_dt_ticks": low_dt,
        "drift_ticks": drift_ticks,
        "range_ticks": range_ticks,
        "trend_strength": trend,
        "mean_spread_ticks": float(np.nanmean(b["spread"])),
    }


# ============================================================================
# PART 2/3 — K=2 LONG signal assembly
# ============================================================================
def build_k2_long_signal_universe(fifo_per_date: dict):
    """Concatenate per-day FIFO timestamps in OOT order; align to predictions.

    Returns:
      ts_all, date_all, offset_by_date, n_per_day
    """
    ts_chunks, date_chunks = [], []
    offsets = {}
    cursor = 0
    n_per_day = {}
    for date in OOT_DATES:
        f = fifo_per_date.get(date)
        if f is None:
            continue
        n = len(f["ts_ns"])
        offsets[date] = cursor
        n_per_day[date] = n
        ts_chunks.append(f["ts_ns"])
        date_chunks.append(np.array([date] * n))
        cursor += n
    return np.concatenate(ts_chunks), np.concatenate(date_chunks), offsets, n_per_day


def assemble_k2_signals(p60, p5m, valid_universe_mask):
    """K=2 SHORT signal mask (= K=2 LONG entry mask in this study):
       p60 in TOP 20%, p5m in BOT 20% (computed over valid universe)."""
    valid = valid_universe_mask & np.isfinite(p60) & np.isfinite(p5m)
    p60v = p60[valid]
    p5mv = p5m[valid]
    thr60_top = float(np.quantile(p60v, 1.0 - BAND_FRAC))
    thr5m_bot = float(np.quantile(p5mv, BAND_FRAC))
    sig_mask = valid & (p60 >= thr60_top) & (p5m <= thr5m_bot)
    return sig_mask, thr60_top, thr5m_bot, int(valid.sum())


def attach_long_outcomes(sig_idx, ts_all, date_all, fifo_per_date, offsets, p60, p5m):
    rows = []
    for gi in sig_idx:
        date = str(date_all[gi])
        f = fifo_per_date.get(date)
        if f is None:
            continue
        within = gi - offsets[date]
        if within < 0 or within >= len(f["ts_ns"]):
            continue
        filled = bool(f["tp4sl3_long_filled"][within])
        net_t = float(f["tp4sl3_long_net_ticks"][within]) if filled else np.nan
        exit_r = str(f["tp4sl3_long_exit_reason"][within]) if filled else ""
        hold_ns = int(f["tp4sl3_long_hold_time_ns"][within]) if filled else 0
        rows.append({
            "global_idx": int(gi),
            "date": date,
            "ts_ns": int(ts_all[gi]),
            "p60": float(p60[gi]),
            "p5m": float(p5m[gi]),
            "filled": filled,
            "net_ticks": net_t,
            "exit_reason": exit_r,
            "hold_ns": hold_ns,
        })
    return pd.DataFrame(rows)


def per_signal_pre_micro(date: str, sig_ts_ns: np.ndarray, b: dict) -> pd.DataFrame:
    bk_ts = b["ts"]
    bk_mid = b["mid_dt"]
    bk_spr = b["spread"]
    rows = []
    for ts in sig_ts_ns:
        idx = np.searchsorted(bk_ts, ts, side="right") - 1
        row = {}
        if idx < 1:
            for w in PRE_WINDOWS_S:
                row[f"vol_{w}s_ticks"] = np.nan
                row[f"drift_{w}s_ticks"] = np.nan
                row[f"mean_spread_{w}s_t"] = np.nan
            rows.append(row)
            continue
        for w in PRE_WINDOWS_S:
            w_ns = w * 1_000_000_000
            start = np.searchsorted(bk_ts, ts - w_ns, side="left")
            if start >= idx:
                row[f"vol_{w}s_ticks"] = np.nan
                row[f"drift_{w}s_ticks"] = np.nan
                row[f"mean_spread_{w}s_t"] = np.nan
                continue
            wmid = bk_mid[start:idx + 1]
            wspr = bk_spr[start:idx + 1]
            wts = bk_ts[start:idx + 1]
            secs = (wts // 1_000_000_000).astype(np.int64)
            tmp = pd.DataFrame({"s": secs, "m": wmid}).groupby("s")["m"].last().values
            if tmp.size > 1:
                d = np.diff(tmp)
                row[f"vol_{w}s_ticks"] = float(np.std(d)) if d.size > 1 else 0.0
            else:
                row[f"vol_{w}s_ticks"] = 0.0
            row[f"drift_{w}s_ticks"] = float(wmid[-1] - wmid[0])
            row[f"mean_spread_{w}s_t"] = float(np.nanmean(wspr))
        rows.append(row)
    return pd.DataFrame(rows)


def intraday_drift_at_signals(date: str, sig_ts_ns: np.ndarray, b: dict) -> np.ndarray:
    if b is None or len(b["ts"]) == 0:
        return np.full(len(sig_ts_ns), np.nan)
    open_mid = float(b["mid_dt"][0])
    out = np.full(len(sig_ts_ns), np.nan)
    for i, ts in enumerate(sig_ts_ns):
        j = np.searchsorted(b["ts"], ts, side="right") - 1
        if j < 0:
            continue
        out[i] = float(b["mid_dt"][j]) - open_mid
    return out


# ============================================================================
# GATES
# ============================================================================
def evaluate_gate(df: pd.DataFrame, name: str, mask: np.ndarray) -> dict:
    sel = df[mask].copy()
    fills = sel[sel.filled]
    n = len(fills)
    if n == 0:
        return {
            "name": name, "n_passing": int(mask.sum()),
            "n_fills": 0, "t_per_fill": 0.0, "wr": 0.0, "pf": 0.0,
            "total_ticks": 0.0, "max_day_conc": 0.0,
            "per_day_pnl": {}, "per_day_fills": {},
            "days_with_fills": 0,
        }
    total = float(fills.net_ticks.sum())
    wr = float((fills.net_ticks > 0).mean())
    wins = float(fills[fills.net_ticks > 0].net_ticks.sum())
    losses = -float(fills[fills.net_ticks <= 0].net_ticks.sum())
    pf = float(wins / losses) if losses > 0 else float("inf")
    day_pnl = fills.groupby("date").net_ticks.sum().to_dict()
    day_fills = fills.groupby("date").size().to_dict()
    return {
        "name": name,
        "n_passing": int(mask.sum()),
        "n_fills": int(n),
        "t_per_fill": total / n,
        "wr": wr,
        "pf": pf,
        "total_ticks": total,
        "max_day_conc": float(max(day_fills.values()) / n),
        "per_day_pnl": {k: float(v) for k, v in day_pnl.items()},
        "per_day_fills": {k: int(v) for k, v in day_fills.items()},
        "days_with_fills": int(len(day_fills)),
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    log_path = Path(os.environ.get(
        "REGIME_LOG",
        str(OUT_DIR / f"regime_analysis_v2_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ))
    L, flush = log_factory(log_path)
    t0 = time.time()
    L(f"PID {os.getpid()} | regime_gate_long_v2")
    L(f"OOT: {OOT_DATES}")

    # Load FIFO + book
    L("Loading FIFO labels and book features...")
    fifo, books = {}, {}
    daily_rows = []
    for date in OOT_DATES:
        fifo[date] = load_fifo(date)
        books[date] = load_book_rth_only(date)
        if fifo[date] is None:
            L(f"  WARN no FIFO for {date}")
        if books[date] is None:
            L(f"  WARN no book RTH for {date}")
        else:
            r = daily_regime_features(date, books[date])
            r["n_events_fifo"] = len(fifo[date]["ts_ns"]) if fifo[date] else 0
            daily_rows.append(r)
            L(f"  {date}: drift={r['drift_ticks']:+.1f}t range={r['range_ticks']:.1f}t "
              f"vol1s_std={r['vol_1s_ticks_std']:.3f}t mean_spr={r['mean_spread_ticks']:.2f}t")
    daily_df = pd.DataFrame(daily_rows)

    # Load predictions
    L(f"Loading predictions {PRED_NPZ}")
    D = np.load(PRED_NPZ, allow_pickle=True)
    p60 = D["pred_log_ret_60s"][:].astype(np.float64)
    p5m = D["pred_log_ret_5min"][:].astype(np.float64)
    L(f"  preds n={len(p60)}")

    # Build LONG-fillable mask (concat tp4sl3_long_filled per OOT day)
    long_mask_chunks = []
    for date in OOT_DATES:
        f = fifo.get(date)
        if f is None:
            continue
        long_mask_chunks.append(f["tp4sl3_long_filled"].astype(bool))
    long_mask = np.concatenate(long_mask_chunks)
    L(f"  LONG-fillable universe: {int(long_mask.sum())} / {len(long_mask)}")

    # Align lengths
    n_min = min(len(p60), len(long_mask))
    p60 = p60[:n_min]; p5m = p5m[:n_min]; long_mask = long_mask[:n_min]

    # Assemble K=2 signals — use LONG-fillable universe for quantile thresholds
    sig_mask, thr60_top, thr5m_bot, n_valid = assemble_k2_signals(p60, p5m, long_mask)
    sig_idx = np.where(sig_mask)[0]
    L(f"  K=2 thresholds: p60_top20={thr60_top:.6f}  p5m_bot20={thr5m_bot:.6f}")
    L(f"  K=2 LONG signals (in long-fillable universe): n={len(sig_idx)} / {n_valid}")

    # Also build a variant using ALL valid (not just long-fillable) for compatibility
    # with the user-reported 247-event count
    all_valid = np.isfinite(p60) & np.isfinite(p5m)
    sm_all, thr60_top_all, thr5m_bot_all, n_valid_all = assemble_k2_signals(
        p60, p5m, all_valid)
    L(f"  [compat] K=2 thresholds on all_valid: p60_top20={thr60_top_all:.6f} "
      f"p5m_bot20={thr5m_bot_all:.6f}  n_signals={int(sm_all.sum())}")

    # Build ts_all/date_all
    ts_all, date_all, offsets, n_per_day = build_k2_long_signal_universe(fifo)
    n_align = min(len(ts_all), n_min)
    ts_all = ts_all[:n_align]
    date_all = date_all[:n_align]
    L(f"  alignment ts/date len={n_align}  per-day n: {n_per_day}")

    # Use the ALL-VALID signal set so we don't miss events on no-long-fill days
    sig_idx_all = np.where(sm_all[:n_align])[0]
    L(f"  Using ALL-VALID K=2 signal set for outcome study: n={len(sig_idx_all)}")

    sig_df = attach_long_outcomes(sig_idx_all, ts_all, date_all, fifo, offsets, p60, p5m)
    L(f"  Signal-outcomes rows: {len(sig_df)}")
    if len(sig_df) == 0:
        L("FATAL: no signals — aborting")
        flush(); return 1

    per_day_sig = sig_df.groupby("date").size().to_dict()
    per_day_fill = sig_df[sig_df.filled].groupby("date").size().to_dict()
    L(f"  Per-day signals: {per_day_sig}")
    L(f"  Per-day fills:   {per_day_fill}")
    for date, sub in sig_df[sig_df.filled].groupby("date"):
        n = len(sub)
        m = float(sub.net_ticks.mean())
        wr = float((sub.net_ticks > 0).mean())
        L(f"    {date}: n_fill={n}  mean_net_t={m:+.3f}  WR={wr:.3f}")

    # Time-of-day bucket
    sig_df["min_of_day"] = ns_to_et_minutes(sig_df.ts_ns.values)
    sig_df["bucket"] = hour_bucket(sig_df.min_of_day.values)

    fill_df = sig_df[sig_df.filled].copy()
    by_bucket = (
        fill_df.groupby("bucket")
        .agg(n=("net_ticks", "size"), mean_t=("net_ticks", "mean"),
             wr=("net_ticks", lambda x: (x > 0).mean()))
        .reset_index()
    )
    L("  Fills by bucket:\n" + by_bucket.to_string(index=False))
    by_day_bucket = (
        fill_df.groupby(["date", "bucket"])
        .agg(n=("net_ticks", "size"), mean_t=("net_ticks", "mean"))
        .reset_index()
    )

    # Pre-signal microstructure
    L("Computing pre-signal microstructure...")
    pre_frames = []
    for date in OOT_DATES:
        ds = sig_df[sig_df.date == date]
        if len(ds) == 0 or books.get(date) is None:
            continue
        sub = per_signal_pre_micro(date, ds.ts_ns.values, books[date])
        sub.index = ds.index
        pre_frames.append(sub)
    if pre_frames:
        sig_df = pd.concat([sig_df, pd.concat(pre_frames, axis=0)], axis=1)

    # Intraday drift since RTH open
    intra = np.full(len(sig_df), np.nan)
    for date in OOT_DATES:
        ds = sig_df[sig_df.date == date]
        if len(ds) == 0:
            continue
        b = books.get(date)
        vals = intraday_drift_at_signals(date, ds.ts_ns.values, b)
        intra[ds.index.values] = vals
    sig_df["intraday_drift_ticks"] = intra

    # ---- Per-day means of pre-signal features (filled only) ----
    fill_df = sig_df[sig_df.filled].copy()
    pre_cols = [c for c in sig_df.columns
                if c.startswith(("vol_", "drift_", "mean_spread_", "intraday_"))]
    if pre_cols:
        L("Per-day means (filled-only) of pre-signal features:")
        L("\n" + fill_df.groupby("date")[pre_cols].mean().to_string())

    # Save signals CSV
    sig_csv = OUT_DIR / "k2_long_signals_with_regime_v2.csv"
    sig_df.to_csv(sig_csv, index=False)
    L(f"Saved {sig_csv}")

    # ---- Gate evaluation ----
    L("Evaluating regime gates...")
    n_sig = len(sig_df)
    mod = sig_df.min_of_day.values
    gates = {}

    base = evaluate_gate(sig_df, "BASELINE_no_gate", np.ones(n_sig, dtype=bool))
    gates["BASELINE_no_gate"] = base
    L(f"  BASELINE: nfill={base['n_fills']}  t/f={base['t_per_fill']:+.3f}  "
      f"WR={base['wr']:.3f}  PF={base['pf']:.3f}  days={base['days_with_fills']}")

    # Time-of-day gates
    gates["GT_skip_open30m"] = evaluate_gate(sig_df, "GT_skip_open30m", mod >= 600)
    gates["GT_skip_open60m"] = evaluate_gate(sig_df, "GT_skip_open60m", mod >= 630)
    gates["GT_skip_close30m"] = evaluate_gate(sig_df, "GT_skip_close30m", mod < 930)
    gates["GT_skip_both"] = evaluate_gate(
        sig_df, "GT_skip_both", (mod >= 600) & (mod < 930))

    # Pre-signal drift / vol
    for col in ["drift_30s_ticks", "drift_60s_ticks", "drift_300s_ticks"]:
        if col in sig_df.columns:
            v = sig_df[col].fillna(-1e9).values
            for thr in [0.0, 1.0, 2.0, 3.0]:
                gates[f"GD_{col}_ge{thr:.0f}"] = evaluate_gate(
                    sig_df, f"GD_{col}_ge{thr:.0f}", v >= thr)
            for thr in [-1.0, -2.0, -3.0]:
                gates[f"GD_{col}_le{abs(thr):.0f}"] = evaluate_gate(
                    sig_df, f"GD_{col}_le{abs(thr):.0f}", v <= thr)

    for col in ["vol_30s_ticks", "vol_60s_ticks", "vol_300s_ticks"]:
        if col in sig_df.columns:
            v = sig_df[col].fillna(0).values
            for thr in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
                gates[f"GV_{col}_ge{thr}"] = evaluate_gate(
                    sig_df, f"GV_{col}_ge{thr}", v >= thr)

    # Intraday drift gates (mean-reversion thesis: LONG when day already DOWN a lot)
    intra_v = sig_df.intraday_drift_ticks.fillna(0).values
    for thr in [-20.0, -10.0, -5.0, 0.0, 5.0, 10.0]:
        if thr <= 0:
            gates[f"GI_intra_le{abs(thr):.0f}"] = evaluate_gate(
                sig_df, f"GI_intra_le{abs(thr):.0f}", intra_v <= thr)
        else:
            gates[f"GI_intra_ge{thr:.0f}"] = evaluate_gate(
                sig_df, f"GI_intra_ge{thr:.0f}", intra_v >= thr)

    # Composite gates
    if "drift_60s_ticks" in sig_df.columns:
        d60 = sig_df.drift_60s_ticks.fillna(-1e9).values
        gates["GC_skip_open_drift60_ge0"] = evaluate_gate(
            sig_df, "GC_skip_open_drift60_ge0", (mod >= 600) & (d60 >= 0))
        gates["GC_skip_open_drift60_le0"] = evaluate_gate(
            sig_df, "GC_skip_open_drift60_le0", (mod >= 600) & (d60 <= 0))
        gates["GC_skip_open_drift60_le-1"] = evaluate_gate(
            sig_df, "GC_skip_open_drift60_le-1", (mod >= 600) & (d60 <= -1))
        gates["GC_skip_open_drift60_le-2"] = evaluate_gate(
            sig_df, "GC_skip_open_drift60_le-2", (mod >= 600) & (d60 <= -2))

    # Pick best gate
    L("Top 15 gates by (n_fills>=20 & WR>=0.5 & t/fill):")
    cand_list = [(k, v) for k, v in gates.items() if k != "BASELINE_no_gate"]
    cand_list.sort(
        key=lambda kv: (
            int(kv[1]["n_fills"] >= 20),
            int(kv[1]["wr"] >= 0.5),
            kv[1]["t_per_fill"],
        ),
        reverse=True,
    )
    for k, v in cand_list[:15]:
        L(f"  {k}: nfill={v['n_fills']:3d}  t/f={v['t_per_fill']:+.3f}  "
          f"WR={v['wr']:.3f}  PF={v['pf']:.2f}  "
          f"days={v['days_with_fills']}  per_day_pnl={v['per_day_pnl']}")

    best_name, best = None, None
    for k, v in cand_list:
        if v["n_fills"] >= 20 and v["wr"] >= 0.5 and v["t_per_fill"] > 0:
            best_name, best = k, v
            break
    if best is None:
        for k, v in cand_list:
            if v["n_fills"] >= 10 and v["t_per_fill"] > 0 and v["wr"] >= 0.4:
                best_name, best = k, v; break
    if best is None:
        best_name, best = "BASELINE_no_gate", base
    L(f"RECOMMENDED gate: {best_name}")
    L(f"  per_day_fills: {best['per_day_fills']}")
    L(f"  per_day_pnl ticks: {best['per_day_pnl']}")

    # Save full daily
    daily_df.to_csv(OUT_DIR / "per_day_features_v2.csv", index=False)
    L(f"Saved {OUT_DIR / 'per_day_features_v2.csv'}")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "oot_dates": OOT_DATES,
        "rule": "K=2 LONG entry on K=2 SHORT signal mask: p60 in TOP 20% AND p5m in BOT 20%",
        "thresholds_long_fillable": {
            "thr_p60_top20": thr60_top,
            "thr_p5m_bot20": thr5m_bot,
            "n_valid": n_valid,
        },
        "thresholds_all_valid": {
            "thr_p60_top20": thr60_top_all,
            "thr_p5m_bot20": thr5m_bot_all,
            "n_valid": n_valid_all,
            "n_signals": int(sm_all.sum()),
        },
        "per_day_features": daily_df.to_dict(orient="records"),
        "per_day_signal_counts": per_day_sig,
        "per_day_fill_counts": per_day_fill,
        "bucket_summary": by_bucket.to_dict(orient="records"),
        "day_bucket_summary": by_day_bucket.to_dict(orient="records"),
        "gates": gates,
        "recommended_gate": {"name": best_name, "stats": best,
                             "rationale": "max t/fill subject to nfill>=20, WR>=0.5"},
        "notes": [
            "K=2 LONG uses K=2 SHORT mask but enters LONG (per HC #357 followup).",
            "P&L from tp4sl3_long_net_ticks (passive_at_touch, FIFO replay).",
            "Commission already netted into FIFO net ticks (ES_RT_COMM_T=0.376).",
            "Book rows filtered: drop warmup (|bid|>10000 ticks or bid/ask==0).",
            "Intraday drift = mid_now - mid_RTH_open (ticks).",
        ],
        "elapsed_sec": time.time() - t0,
    }
    js = OUT_DIR / "regime_gate_results_v2.json"
    with open(js, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating)
                  else (int(o) if isinstance(o, np.integer) else str(o)))
    L(f"Saved {js}")
    L(f"DONE in {time.time() - t0:.1f}s")
    flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
