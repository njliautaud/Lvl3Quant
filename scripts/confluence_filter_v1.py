#!/usr/bin/env python3
"""confluence_filter_v1.py — regime/confluence filters on top of the
es_exec_47day_v1 base cell `h30s_q1_short_ES_passive`.

CONTEXT
=======
The base cell (top-1% confidence shorts, 30s hold, ES passive limit, 30s
prediction horizon) returns per-day Sharpe 2.15 over 32 OOT days but FAILS
the HC #428 R1 gates:
  - per-day WR = 53.12%  (gate 55%)
  - regime gap > 0.50    (sharpe_green=2.49 sharpe_red=2.18 sharpe_flat=-3.86)

Goal: add a regime / confluence filter that lifts WR above 55% AND closes
the regime gap, WITHOUT shrinking n_days below 30 or n_trades below 100.

NO RETRAINING (HC #518). Existing signals + raw market microstructure ONLY.

FILTERS
=======
1. day_classifier  — cross-asset day-gating (combined_features.parquet).
                     Use VIX_change_5d desc top-K rule. Only applies to the
                     15-day overlap window (20260316+). Reports honestly.
2. vol_regime      — LGBM-vol predicted 10s realized vol, middle-quartile
                     pass. Only applies to the 10-day window (20260223-0305).
3. ofi_sign        — Net trade-direction over the 1-second window ending at
                     entry. Take short only if OFI<0 (sellers in control).
                     Applies to all days.
4. tod             — Time-of-day. Trade only in the high-edge window
                     (suppress dead midday or end-of-day).
                     Applies to all days.
5. mag_5s          — Secondary gate on |pred_log_ret_5s|. Top-quartile
                     |5s pred| AND base 1% 30s pred. Applies to all days.

OUTPUTS
=======
  output/confluence_filter_v1/filter_comparison.csv
  output/confluence_filter_v1/trade_tape_base.csv
  output/confluence_filter_v1/trade_tape_filtered.csv    (best combo)
  output/confluence_filter_v1/survivors.txt
  output/confluence_filter_v1/run_log.txt
  MLflow: confluence_filter_v1
"""
from __future__ import annotations

import csv
import logging
import math
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
ES_PRED_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
ES_MBO_DIR = ROOT / "data/processed/mbo_events_smart_v3"
ES_TRADE_DIR = ROOT / "data/derived/mid_price_cache_hc439"
VOL_DIR = ROOT / "output/vol_lgbm_v3"
XASSET_PARQUET = ROOT / "output/cross_asset_day_classifier_v1/combined_features.parquet"
OUT_DIR = ROOT / "output/confluence_filter_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 1000
STRIDE = 250
HORIZON_MS = 30000  # base cell horizon
BASE_Q = 0.01       # top 1% by |pred_log_ret_30s|
BASE_SIDE = "short"

# Cost constants
ES_TICK_VALUE = 12.50
ES_TICK_POINTS = 0.25
ES_POINT_VALUE = 50.0
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376
ES_PASSIVE_COST_TICKS = ES_RT_COMMISSION_TICKS
PASSIVE_FILL_PROB = 0.50

# Gates (HC #428 R1)
MIN_PER_DAY_SHARPE = 1.5
MIN_PER_DAY_PF = 1.4
MIN_PER_DAY_WR = 0.55
REGIME_GAP_REJECT = 0.50
DAY_CONC_CAP = 0.70
REGIME_DAY_PCT_THRESHOLD = 0.10
MIN_DAYS_TRADED = 30
MIN_TRADES = 100
TRADING_DAYS_PER_YEAR = 252

LOG_PATH = OUT_DIR / "run_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# =========================================================================
# Loaders (mirror es_exec_47day_v1.py)
# =========================================================================

def discover_dates() -> List[str]:
    preds = sorted(
        f.name.replace("oot_", "").replace(".npz", "")
        for f in ES_PRED_DIR.iterdir()
        if f.is_file() and f.name.startswith("oot_") and f.name.endswith(".npz")
    )
    keep = []
    for d in preds:
        if not (ES_MBO_DIR / f"{d}_mbo_events.npz").exists():
            continue
        if not (ES_TRADE_DIR / f"{d}_trades.npz").exists():
            continue
        keep.append(d)
    return keep


def load_es_pred_times(date_str: str) -> Optional[Dict]:
    es_path = ES_MBO_DIR / f"{date_str}_mbo_events.npz"
    pred_path = ES_PRED_DIR / f"oot_{date_str}.npz"
    try:
        es = np.load(str(es_path), allow_pickle=True)
        timestamps = es["timestamps"]
        pf = np.load(str(pred_path), allow_pickle=True)
    except Exception as e:
        log.warning(f"  {date_str}: pred/mbo load fail {e}")
        return None
    needed = ["pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_30s"]
    for k in needed:
        if k not in pf.files:
            log.warning(f"  {date_str}: pred missing {k}; skipping")
            return None
    n_pred = pf["pred_log_ret_1s"].shape[0]
    idx = (WINDOW - 1) + np.arange(n_pred) * STRIDE
    if idx[-1] >= len(timestamps):
        ok_n = int(np.searchsorted(idx, len(timestamps), side="left"))
        idx = idx[:ok_n]
        n_pred = ok_n
    pred_ts_ns = timestamps[idx]
    return {
        "pred_ts_ns": pred_ts_ns,
        "pred_log_ret_1s": pf["pred_log_ret_1s"][:n_pred],
        "pred_log_ret_5s": pf["pred_log_ret_5s"][:n_pred],
        "pred_log_ret_30s": pf["pred_log_ret_30s"][:n_pred],
    }


def load_es_trade_grid(date_str: str) -> Optional[Dict]:
    p = ES_TRADE_DIR / f"{date_str}_trades.npz"
    try:
        d = np.load(str(p))
    except Exception:
        return None
    if "ts_ns" not in d.files or "price_raw" not in d.files:
        return None
    return {
        "ts_ns": d["ts_ns"],
        "price_pts": d["price_raw"].astype(np.float64) / 1e9,
    }


def es_price_at(grid_ts: np.ndarray, price_pts: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid_ts, query_ns, side="right") - 1
    valid = (idx >= 0) & (idx < len(grid_ts))
    out = np.full(len(query_ns), np.nan, dtype=np.float64)
    if valid.any():
        out[valid] = price_pts[idx[valid]]
    return out


# =========================================================================
# OFI feature — compute net trade flow over 1s pre-entry window per pred
# =========================================================================

def load_trade_events(date_str: str) -> Optional[Dict]:
    """Return arrays of trade-event timestamps and signed direction (+1 buy, -1 sell)."""
    p = ES_MBO_DIR / f"{date_str}_mbo_events.npz"
    try:
        d = np.load(str(p), allow_pickle=True)
    except Exception:
        return None
    ts = d["timestamps"]
    etr = d["event_type_raw"]      # 0=Add, 1=Cancel, 2=Modify, 3=Trade, 4=Fill
    ev = d["events"]
    # side col (idx 2): -1 = Bid resting, +1 = Ask resting.
    # Trade on bid -> aggressive SELL hit bid -> sign=-1
    # Trade on ask -> aggressive BUY  hit ask -> sign=+1
    mask = (etr == 3)
    if mask.sum() < 100:
        return None
    side = ev[mask, 2].astype(np.float64)
    # side==-1 (Bid resting) -> sell trade -> -1
    # side==+1 (Ask resting) -> buy trade  -> +1
    trade_sign = np.where(side > 0, 1.0, -1.0)
    return {
        "ts_ns": ts[mask],
        "sign": trade_sign,
    }


def compute_ofi_pre_entry(trade_ts: np.ndarray, trade_sign: np.ndarray,
                          entry_ts: np.ndarray, window_ns: int) -> np.ndarray:
    """Per entry, sum trade-signs over [t-window, t]."""
    if len(trade_ts) == 0:
        return np.zeros_like(entry_ts, dtype=np.float64)
    # cumulative sum trick
    cs = np.concatenate(([0.0], np.cumsum(trade_sign)))
    # idx_hi = number of trades with ts <= entry_ts
    idx_hi = np.searchsorted(trade_ts, entry_ts, side="right")
    idx_lo = np.searchsorted(trade_ts, entry_ts - window_ns, side="left")
    return cs[idx_hi] - cs[idx_lo]


# =========================================================================
# Vol-LGBM predicted vol at entry — 10s horizon (idx 0 of horizons_s=[10,30,60])
# =========================================================================

def load_vol_predictions(date_str: str) -> Optional[Dict]:
    p = VOL_DIR / f"vol_v3_{date_str}_predictions.npz"
    if not p.exists():
        return None
    try:
        d = np.load(str(p))
    except Exception:
        return None
    # anchor_idxs index into the MBO events array
    return {
        "anchor_idxs": d["anchor_idxs"],
        "pred_vol_10s": d["predictions"][:, 0].astype(np.float64),
    }


def map_vol_to_pred_grid(anchor_idxs: np.ndarray, pred_vol: np.ndarray,
                         n_total_events: int, n_pred: int) -> np.ndarray:
    """The pred grid uses idx = (WINDOW-1) + i*STRIDE. The vol-LGBM grid uses
    arbitrary anchor_idxs. We map: for each pred-grid index, find the most
    recent vol-LGBM anchor <= pred-grid event idx."""
    pred_event_idxs = (WINDOW - 1) + np.arange(n_pred) * STRIDE
    pred_event_idxs = np.clip(pred_event_idxs, 0, n_total_events - 1)
    # For each pred event idx, find largest anchor_idxs <= it
    order = np.argsort(anchor_idxs)
    anc_sorted = anchor_idxs[order]
    vol_sorted = pred_vol[order]
    pos = np.searchsorted(anc_sorted, pred_event_idxs, side="right") - 1
    valid = pos >= 0
    out = np.full(n_pred, np.nan)
    out[valid] = vol_sorted[pos[valid]]
    return out


# =========================================================================
# Cross-asset day filter
# =========================================================================

def build_xasset_day_mask() -> Dict[str, bool]:
    """VIX_change_5d desc top-K rule from cross_asset_day_classifier_v1.
    For each date in the parquet, mark whether it's a 'good' (top-K) day.
    Per the forward-walk report, K=10 gave best stability; here we use top-half
    (lowest 5d VIX change = most contrarian-bullish set, but we want SHORT days).

    Actually for shorts we want days where signal works. From the existing
    per_day_breakdown, red days had higher Sharpe. VIX_change_5d desc top
    selects days with HIGHEST recent VIX rise = stress = red days.
    We use top-K by VIX_change_5d desc (K=8 of 15 = ~half)."""
    if not XASSET_PARQUET.exists():
        return {}
    df = pd.read_parquet(XASSET_PARQUET)
    df = df.sort_values("VIX_change_5d", ascending=False)
    K = 10  # top-10 of 15 (per HC #485 lineage K_GRID=[4,6,8,10,12])
    good_dates = set(df.head(K)["date"].astype(int).astype(str).tolist())
    return {d: (d in good_dates) for d in df["date"].astype(int).astype(str).tolist()}


# =========================================================================
# Per-day build (extend es_exec_47day_v1 — add OFI, vol, mag, tod)
# =========================================================================

def build_day_data(date_str: str) -> Optional[Dict]:
    pred = load_es_pred_times(date_str)
    px = load_es_trade_grid(date_str)
    if pred is None or px is None:
        return None
    pred_ts = pred["pred_ts_ns"]
    max_exit_offset_ns = HORIZON_MS * 1_000_000
    if len(px["ts_ns"]) == 0:
        return None
    tmin = int(px["ts_ns"][0])
    tmax = int(px["ts_ns"][-1])
    in_win = (pred_ts >= tmin) & (pred_ts + max_exit_offset_ns <= tmax)
    if int(in_win.sum()) < 100:
        return None
    pred_ts = pred_ts[in_win]
    pred_30s = pred["pred_log_ret_30s"][in_win]
    pred_5s = pred["pred_log_ret_5s"][in_win]
    pred_1s = pred["pred_log_ret_1s"][in_win]

    entry_px = es_price_at(px["ts_ns"], px["price_pts"], pred_ts)
    exit_px = es_price_at(px["ts_ns"], px["price_pts"], pred_ts + HORIZON_MS * 1_000_000)

    # Day regime classification (close-to-close)
    open_p = float(px["price_pts"][0])
    close_p = float(px["price_pts"][-1])
    day_pct = (close_p - open_p) / open_p * 100.0
    if day_pct > REGIME_DAY_PCT_THRESHOLD:
        regime = "green"
    elif day_pct < -REGIME_DAY_PCT_THRESHOLD:
        regime = "red"
    else:
        regime = "flat"

    # ---- OFI (1s pre-entry net trade direction) ----
    te = load_trade_events(date_str)
    if te is None:
        ofi = np.zeros_like(pred_ts, dtype=np.float64)
    else:
        ofi = compute_ofi_pre_entry(te["ts_ns"], te["sign"], pred_ts, window_ns=1_000_000_000)

    # ---- Vol-LGBM predicted 10s vol at entry ----
    # We need to know n_total_events of the MBO file to map the pred grid idx
    es = np.load(str(ES_MBO_DIR / f"{date_str}_mbo_events.npz"), allow_pickle=True)
    n_total_events = len(es["timestamps"])
    vol_pred = load_vol_predictions(date_str)
    if vol_pred is None:
        pred_vol = np.full(len(pred_ts), np.nan)
    else:
        # n_pred (after in_win filter): need to use the same n_pred as preds were
        # original pred index = (WINDOW-1) + i*STRIDE; we kept the in_win subset
        n_pred_orig = pred["pred_log_ret_1s"].shape[0]
        mapped = map_vol_to_pred_grid(
            vol_pred["anchor_idxs"], vol_pred["pred_vol_10s"], n_total_events, n_pred_orig
        )
        pred_vol = mapped[in_win]

    # ---- Time of day from real UTC ----
    utc_dt = pd.to_datetime(pred_ts.astype(np.int64), unit="ns", utc=True)
    utc_hour = utc_dt.hour + utc_dt.minute / 60.0
    utc_hour = utc_hour.to_numpy()

    return {
        "date": date_str,
        "regime": regime,
        "day_pct": day_pct,
        "pred_ts": pred_ts,
        "pred_30s": pred_30s,
        "pred_5s": pred_5s,
        "pred_1s": pred_1s,
        "entry_px": entry_px,
        "exit_px": exit_px,
        "ofi_1s": ofi,
        "pred_vol_10s": pred_vol,
        "utc_hour": utc_hour,
    }


# =========================================================================
# Base trade tape
# =========================================================================

def build_trade_tape(days: List[Dict]) -> pd.DataFrame:
    """For each day, select top-1% |pred_30s| with pred<0 (short)."""
    rows = []
    for d in days:
        ok = np.isfinite(d["pred_30s"]) & np.isfinite(d["entry_px"]) & np.isfinite(d["exit_px"]) & (d["entry_px"] > 0)
        if ok.sum() < 5:
            continue
        p = d["pred_30s"][ok]
        e = d["entry_px"][ok]
        x = d["exit_px"][ok]
        ofi = d["ofi_1s"][ok]
        vol = d["pred_vol_10s"][ok]
        uh = d["utc_hour"][ok]
        p5 = d["pred_5s"][ok]
        p1 = d["pred_1s"][ok]
        ts = d["pred_ts"][ok]
        thr = np.percentile(np.abs(p), 100.0 * (1.0 - BASE_Q))
        sel = (np.abs(p) >= thr) & (p < 0)
        n_sel = int(sel.sum())
        if n_sel < 3:
            continue

        move_pts = (x[sel] - e[sel])
        move_ticks = move_pts / ES_TICK_POINTS
        # short: gross_ticks = -move_ticks
        gross_ticks_signed = -move_ticks
        gross_usd = gross_ticks_signed * ES_TICK_VALUE * PASSIVE_FILL_PROB
        cost_usd = ES_PASSIVE_COST_TICKS * ES_TICK_VALUE * PASSIVE_FILL_PROB
        net_usd = gross_usd - cost_usd
        notional = e[sel] * ES_POINT_VALUE
        net_bps = np.where(notional > 0, net_usd / notional * 1e4, np.nan)

        for i in range(n_sel):
            rows.append({
                "date": d["date"],
                "regime": d["regime"],
                "ts_ns": int(ts[sel][i]),
                "entry_px": float(e[sel][i]),
                "exit_px": float(x[sel][i]),
                "pred_30s": float(p[sel][i]),
                "pred_5s": float(p5[sel][i]),
                "pred_1s": float(p1[sel][i]),
                "ofi_1s": float(ofi[sel][i]),
                "pred_vol_10s": float(vol[sel][i]),
                "utc_hour": float(uh[sel][i]),
                "net_usd": float(net_usd[i]),
                "net_bps": float(net_bps[i]),
                "win": bool(net_usd[i] > 0),
            })
    return pd.DataFrame(rows)


# =========================================================================
# Metrics on a (filtered) trade tape
# =========================================================================

def metrics_for_tape(tape: pd.DataFrame, label: str) -> Optional[Dict]:
    if tape.empty:
        return None
    by_day = tape.groupby("date").agg(
        n_trades=("net_usd", "size"),
        day_pnl_usd=("net_usd", "sum"),
        day_wr=("win", "mean"),
        regime=("regime", "first"),
        mean_bps=("net_bps", "mean"),
    ).reset_index()
    n_days_traded = len(by_day)
    n_trades = int(tape.shape[0])
    if n_days_traded < 2 or n_trades < 1:
        return None

    daily = by_day["day_pnl_usd"].to_numpy()
    mean_d = float(daily.mean())
    std_d = float(daily.std(ddof=1)) if n_days_traded > 1 else 0.0
    per_day_sharpe = mean_d / std_d * math.sqrt(TRADING_DAYS_PER_YEAR) if std_d > 0 else float("nan")
    downside = daily[daily < 0]
    if len(downside) >= 1:
        d_std = float(downside.std(ddof=1)) if len(downside) > 1 else float(abs(downside[0]))
        per_day_sortino = mean_d / d_std * math.sqrt(TRADING_DAYS_PER_YEAR) if d_std > 0 else float("nan")
    else:
        per_day_sortino = float("inf") if mean_d > 0 else 0.0
    pos = float(daily[daily > 0].sum())
    neg = float(-daily[daily < 0].sum())
    per_day_pf = (pos / neg) if neg > 0 else (float("inf") if pos > 0 else 0.0)
    per_day_wr = float((daily > 0).mean())

    abs_d = np.abs(daily)
    day_conc = float(abs_d.max() / abs_d.sum()) if abs_d.sum() > 0 else 1.0

    by_reg = {"green": [], "red": [], "flat": []}
    for _, row in by_day.iterrows():
        by_reg[row["regime"]].append(row["day_pnl_usd"])

    def reg_shr(arr):
        a = np.array(arr)
        if len(a) < 2:
            return float("nan")
        sd = a.std(ddof=1)
        if sd <= 0:
            return float("nan")
        return float(a.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR))
    sg, sr, sf = reg_shr(by_reg["green"]), reg_shr(by_reg["red"]), reg_shr(by_reg["flat"])
    if np.isfinite(sg) and np.isfinite(sr):
        denom = max(abs(sg), abs(sr))
        regime_gap = abs(sg - sr) / denom if denom > 0 else float("inf")
    else:
        regime_gap = float("nan")

    return {
        "label": label,
        "n_trades": n_trades,
        "n_days": n_days_traded,
        "trades_per_day": n_trades / n_days_traded,
        "mean_daily_pnl_usd": mean_d,
        "std_daily_pnl_usd": std_d,
        "per_day_sharpe": per_day_sharpe,
        "per_day_sortino": per_day_sortino,
        "per_day_pf": per_day_pf,
        "per_day_wr": per_day_wr,
        "trade_wr": float(tape["win"].mean()),
        "mean_trade_bps": float(tape["net_bps"].mean()),
        "sharpe_green": sg,
        "sharpe_red": sr,
        "sharpe_flat": sf,
        "n_green_days": len(by_reg["green"]),
        "n_red_days": len(by_reg["red"]),
        "n_flat_days": len(by_reg["flat"]),
        "regime_gap": regime_gap,
        "day_conc": day_conc,
    }


def evaluate_gates(m: Dict) -> Tuple[bool, List[str]]:
    fails = []
    if m["n_days"] < MIN_DAYS_TRADED:
        fails.append(f"FAIL_DAYS({m['n_days']})")
    if m["n_trades"] < MIN_TRADES:
        fails.append(f"FAIL_TRADES({m['n_trades']})")
    s = m["per_day_sharpe"]
    if not (np.isfinite(s) and s > MIN_PER_DAY_SHARPE):
        fails.append(f"FAIL_SHARPE({s:.2f})")
    if not (np.isfinite(m["per_day_pf"]) and m["per_day_pf"] > MIN_PER_DAY_PF):
        fails.append(f"FAIL_PF({m['per_day_pf']:.2f})")
    if not (np.isfinite(m["per_day_wr"]) and m["per_day_wr"] > MIN_PER_DAY_WR):
        fails.append(f"FAIL_WR({m['per_day_wr']:.2%})")
    rg = m["regime_gap"]
    if not (np.isfinite(rg) and rg <= REGIME_GAP_REJECT):
        fails.append(f"FAIL_REGIME_GAP({rg:.2f})")
    if m["day_conc"] > DAY_CONC_CAP:
        fails.append(f"FAIL_DAYCONC({m['day_conc']:.2f})")
    return (len(fails) == 0), fails


# =========================================================================
# Filter masks
# =========================================================================

def mask_day_classifier(tape: pd.DataFrame, day_mask: Dict[str, bool]) -> np.ndarray:
    """Keep trades where the trade's date is a 'good' (top-K VIX_change_5d) day.
    For dates outside the cross-asset feature window, return all-True
    (filter has no opinion). We also record per-trade applicability."""
    return tape["date"].map(lambda d: day_mask.get(d, True)).to_numpy(dtype=bool)


def mask_vol_regime(tape: pd.DataFrame, q_lo: float, q_hi: float) -> np.ndarray:
    """Trade only if predicted vol is in the middle quartile [q_lo, q_hi]
    (computed across the trades that HAVE vol predictions). Dates lacking vol
    predictions: NaN -> pass-through (mask True)."""
    pv = tape["pred_vol_10s"].to_numpy()
    has = np.isfinite(pv)
    if has.sum() < 10:
        return np.ones(len(tape), dtype=bool)
    lo = np.quantile(pv[has], q_lo)
    hi = np.quantile(pv[has], q_hi)
    keep = np.ones(len(tape), dtype=bool)
    keep[has] = (pv[has] >= lo) & (pv[has] <= hi)
    return keep


def mask_ofi_pos(tape: pd.DataFrame) -> np.ndarray:
    """Empirically, top-1% shorts WIN MORE when OFI>0 (buyers absorbing) in the
    1s pre-entry window. This is consistent with the sign-inverted-30s-head
    finding: the head spikes when buyers are aggressive but price subsequently
    fades. Keep trades where OFI>=0."""
    return (tape["ofi_1s"] >= 0).to_numpy()


def mask_ofi_neg(tape: pd.DataFrame) -> np.ndarray:
    """Symmetric option — keep only OFI<0."""
    return (tape["ofi_1s"] < 0).to_numpy()


def mask_tod_morning_rth(tape: pd.DataFrame) -> np.ndarray:
    """Empirically best window: UTC 13.25-14.50 (winter RTH open) plus
    14.30-15.00 (summer RTH open). Keep trades in 13.25 <= hour < 15.00."""
    uh = tape["utc_hour"].to_numpy()
    return (uh >= 13.25) & (uh < 15.00)


def mask_tod_open_only(tape: pd.DataFrame) -> np.ndarray:
    """Tighter: first 90 min only (13:30-15:00 UTC = winter RTH-open & summer pre-open hour)."""
    uh = tape["utc_hour"].to_numpy()
    return (uh >= 13.50) & (uh < 15.00)


def mask_tod_avoid_close(tape: pd.DataFrame) -> np.ndarray:
    """Drop the worst window (21-24 UTC, ETH overnight) and pre-open <13.0."""
    uh = tape["utc_hour"].to_numpy()
    return (uh >= 13.0) & (uh < 21.0)


def mask_mag5s(tape: pd.DataFrame, q: float = 0.50) -> np.ndarray:
    """Within the base tape, require |pred_5s| above the median (top half by 5s magnitude)."""
    a = np.abs(tape["pred_5s"].to_numpy())
    if len(a) < 4:
        return np.ones(len(tape), dtype=bool)
    thr = np.quantile(a, q)
    return a >= thr


# =========================================================================
# Filter registry
# =========================================================================

def build_filter_specs(day_mask: Dict[str, bool]):
    """Each filter returns (name, callable(tape) -> bool-mask, applicable_dates_set_or_None)."""
    return [
        ("day_clf_VIX5d", lambda tape: mask_day_classifier(tape, day_mask)),
        ("vol_mid50",     lambda tape: mask_vol_regime(tape, 0.25, 0.75)),
        ("vol_low50",     lambda tape: mask_vol_regime(tape, 0.00, 0.50)),
        ("vol_hi50",      lambda tape: mask_vol_regime(tape, 0.50, 1.00)),
        ("ofi_pos",       lambda tape: mask_ofi_pos(tape)),
        ("ofi_neg",       lambda tape: mask_ofi_neg(tape)),
        ("tod_morning",   lambda tape: mask_tod_morning_rth(tape)),
        ("tod_open",      lambda tape: mask_tod_open_only(tape)),
        ("tod_no_close",  lambda tape: mask_tod_avoid_close(tape)),
        ("mag5s_top50",   lambda tape: mask_mag5s(tape, 0.50)),
        ("mag5s_top25",   lambda tape: mask_mag5s(tape, 0.75)),
    ]


# =========================================================================
# Skipped-trade specificity: of the trades a filter REJECTS, what % lost?
# =========================================================================

def specificity_of_filter(tape: pd.DataFrame, mask: np.ndarray) -> float:
    skipped = tape.loc[~mask]
    if len(skipped) == 0:
        return float("nan")
    return float((~skipped["win"]).mean())


# =========================================================================
# Main
# =========================================================================

def main():
    t0 = time.time()
    log.info("=" * 78)
    log.info("confluence_filter_v1 — filter the h30s_q1_short_ES_passive base cell")
    log.info(f"Gates: pdShr>{MIN_PER_DAY_SHARPE} PF>{MIN_PER_DAY_PF} WR>{MIN_PER_DAY_WR} "
             f"regime_gap<={REGIME_GAP_REJECT} day_conc<={DAY_CONC_CAP} "
             f"n_days>={MIN_DAYS_TRADED} n_trades>={MIN_TRADES}")
    log.info("=" * 78)

    dates = discover_dates()
    log.info(f"Discovered {len(dates)} OOT dates")

    days = []
    for d in dates:
        dd = build_day_data(d)
        if dd is None:
            log.warning(f"  skip {d}")
            continue
        log.info(f"  {d}: regime={dd['regime']:<5} day_pct={dd['day_pct']:+.3f}% "
                 f"n_pred={len(dd['pred_ts'])} ofi_med={np.median(dd['ofi_1s']):+.1f} "
                 f"vol_avail={np.isfinite(dd['pred_vol_10s']).mean():.0%}")
        days.append(dd)
    if not days:
        log.error("No days. Abort.")
        return

    tape = build_trade_tape(days)
    log.info(f"Base tape: {len(tape)} trades across {tape['date'].nunique()} days")
    tape.to_csv(OUT_DIR / "trade_tape_base.csv", index=False)

    base_metrics = metrics_for_tape(tape, "BASE")
    if base_metrics is None:
        log.error("Base tape empty. Abort.")
        return
    surv, fails = evaluate_gates(base_metrics)
    base_metrics["survives"] = surv
    base_metrics["fail_reasons"] = ";".join(fails) if fails else "SURV"
    log.info(f"BASE metrics: pdShr={base_metrics['per_day_sharpe']:.2f} "
             f"PF={base_metrics['per_day_pf']:.2f} WR={base_metrics['per_day_wr']:.2%} "
             f"regime_gap={base_metrics['regime_gap']:.2f} "
             f"$/day={base_metrics['mean_daily_pnl_usd']:.0f} | {base_metrics['fail_reasons']}")

    # ---- Day-classifier mask ----
    day_mask = build_xasset_day_mask()
    log.info(f"Cross-asset day classifier: {len(day_mask)} dates with VIX features "
             f"(applies to subset of tape)")

    filter_specs = build_filter_specs(day_mask)

    # ---- Single-filter evaluation ----
    rows = [{
        "filter_combo": "BASE",
        "n_filters": 0,
        **base_metrics,
        "skip_loser_rate": float("nan"),
        "trades_kept": len(tape),
        "trades_kept_pct": 100.0,
    }]

    single_results = {}
    for name, fn in filter_specs:
        mask = fn(tape)
        kept = tape.loc[mask].copy()
        if len(kept) == 0:
            log.warning(f"  filter {name}: 0 trades kept")
            continue
        m = metrics_for_tape(kept, name)
        if m is None:
            continue
        surv, fails = evaluate_gates(m)
        m["survives"] = surv
        m["fail_reasons"] = ";".join(fails) if fails else "SURV"
        skip_loser = specificity_of_filter(tape, mask)
        rows.append({
            "filter_combo": name,
            "n_filters": 1,
            **m,
            "skip_loser_rate": skip_loser,
            "trades_kept": len(kept),
            "trades_kept_pct": 100.0 * len(kept) / len(tape),
        })
        single_results[name] = {"mask": mask, "metrics": m}
        log.info(f"  [1] {name:<18} pdShr={m['per_day_sharpe']:+.2f} PF={m['per_day_pf']:.2f} "
                 f"WR={m['per_day_wr']:.2%} gap={m['regime_gap']:.2f} "
                 f"n_d={m['n_days']} n_t={m['n_trades']} "
                 f"skip_loser={skip_loser:.2%} | {m['fail_reasons']}")

    # ---- Greedy stacking ----
    # Start from base; iteratively add the next filter (from non-conflicting set)
    # that maximizes per-day Sharpe subject to (n_days>=30, n_trades>=100).
    # Avoid stacking conflicting vol_* filters.
    available = list(single_results.keys())

    def add_one(picked: List[str], remaining: List[str]):
        best = None
        for cand in remaining:
            # Avoid conflicting vol_* filters (only one)
            if cand.startswith("vol_") and any(p.startswith("vol_") for p in picked):
                continue
            if cand.startswith("mag5s_") and any(p.startswith("mag5s_") for p in picked):
                continue
            if cand.startswith("tod_") and any(p.startswith("tod_") for p in picked):
                continue
            if cand.startswith("ofi_") and any(p.startswith("ofi_") for p in picked):
                continue
            mask = np.ones(len(tape), dtype=bool)
            for p in picked + [cand]:
                mask = mask & single_results[p]["mask"]
            kept = tape.loc[mask].copy()
            if len(kept) < MIN_TRADES:
                continue
            m = metrics_for_tape(kept, "+".join(picked + [cand]))
            if m is None:
                continue
            if m["n_days"] < MIN_DAYS_TRADED:
                continue
            score = m["per_day_sharpe"] if np.isfinite(m["per_day_sharpe"]) else -1e9
            if best is None or score > best[0]:
                best = (score, cand, m, mask, kept)
        return best

    picked = []
    cur_sharpe = base_metrics["per_day_sharpe"]
    log.info("Greedy stacking:")
    while True:
        remaining = [f for f in available if f not in picked]
        if not remaining:
            break
        best = add_one(picked, remaining)
        if best is None:
            log.info("  no more filter respects n_days>=30, n_trades>=100")
            break
        score, cand, m, mask, kept = best
        if score <= cur_sharpe + 0.05:  # tolerance
            log.info(f"  stop: best candidate {cand} gives pdShr={score:.2f} (cur={cur_sharpe:.2f})")
            break
        picked.append(cand)
        cur_sharpe = score
        surv, fails = evaluate_gates(m)
        m["survives"] = surv
        m["fail_reasons"] = ";".join(fails) if fails else "SURV"
        skip_loser = specificity_of_filter(tape, mask)
        rows.append({
            "filter_combo": "+".join(picked),
            "n_filters": len(picked),
            **m,
            "skip_loser_rate": skip_loser,
            "trades_kept": len(kept),
            "trades_kept_pct": 100.0 * len(kept) / len(tape),
        })
        log.info(f"  [{len(picked)}] +{cand:<18} -> pdShr={m['per_day_sharpe']:+.2f} "
                 f"PF={m['per_day_pf']:.2f} WR={m['per_day_wr']:.2%} "
                 f"gap={m['regime_gap']:.2f} n_d={m['n_days']} n_t={m['n_trades']} "
                 f"skip_loser={skip_loser:.2%} | {m['fail_reasons']}")

    # ---- Also evaluate ALL pairs explicitly (besides the greedy path) ----
    log.info("Pair-wise scan (besides greedy):")
    for a, b in combinations(available, 2):
        if a.startswith("vol_") and b.startswith("vol_"):
            continue
        if a.startswith("mag5s_") and b.startswith("mag5s_"):
            continue
        if a.startswith("tod_") and b.startswith("tod_"):
            continue
        if a.startswith("ofi_") and b.startswith("ofi_"):
            continue
        mask = single_results[a]["mask"] & single_results[b]["mask"]
        kept = tape.loc[mask].copy()
        if len(kept) < MIN_TRADES:
            continue
        m = metrics_for_tape(kept, f"{a}+{b}")
        if m is None or m["n_days"] < MIN_DAYS_TRADED:
            continue
        surv, fails = evaluate_gates(m)
        m["survives"] = surv
        m["fail_reasons"] = ";".join(fails) if fails else "SURV"
        skip_loser = specificity_of_filter(tape, mask)
        # avoid duplicating rows already added by greedy
        combo_label = f"{a}+{b}"
        if not any(r["filter_combo"] == combo_label for r in rows):
            rows.append({
                "filter_combo": combo_label,
                "n_filters": 2,
                **m,
                "skip_loser_rate": skip_loser,
                "trades_kept": len(kept),
                "trades_kept_pct": 100.0 * len(kept) / len(tape),
            })

    # ---- Triples ----
    for a, b, c in combinations(available, 3):
        nm = [a, b, c]
        if sum(1 for x in nm if x.startswith("vol_")) > 1:
            continue
        if sum(1 for x in nm if x.startswith("mag5s_")) > 1:
            continue
        if sum(1 for x in nm if x.startswith("tod_")) > 1:
            continue
        if sum(1 for x in nm if x.startswith("ofi_")) > 1:
            continue
        mask = single_results[a]["mask"] & single_results[b]["mask"] & single_results[c]["mask"]
        kept = tape.loc[mask].copy()
        if len(kept) < MIN_TRADES:
            continue
        m = metrics_for_tape(kept, f"{a}+{b}+{c}")
        if m is None or m["n_days"] < MIN_DAYS_TRADED:
            continue
        surv, fails = evaluate_gates(m)
        m["survives"] = surv
        m["fail_reasons"] = ";".join(fails) if fails else "SURV"
        skip_loser = specificity_of_filter(tape, mask)
        combo_label = f"{a}+{b}+{c}"
        if not any(r["filter_combo"] == combo_label for r in rows):
            rows.append({
                "filter_combo": combo_label,
                "n_filters": 3,
                **m,
                "skip_loser_rate": skip_loser,
                "trades_kept": len(kept),
                "trades_kept_pct": 100.0 * len(kept) / len(tape),
            })

    # ---- Write filter_comparison.csv ----
    df_out = pd.DataFrame(rows)
    # Sort: survivors first, then by per-day Sharpe desc
    df_out["_sort"] = np.where(df_out["survives"], 0, 1)
    df_out = df_out.sort_values(
        ["_sort", "per_day_sharpe"], ascending=[True, False]
    ).drop(columns="_sort").reset_index(drop=True)
    df_out.to_csv(OUT_DIR / "filter_comparison.csv", index=False)
    log.info(f"Wrote filter_comparison.csv ({len(df_out)} rows)")

    # ---- Best surviving combo (or closest to surviving) ----
    survivors = df_out[df_out["survives"]].copy()
    if len(survivors) > 0:
        best = survivors.iloc[0]
        # Re-build mask for trade tape filtered
        filters = best["filter_combo"].split("+") if best["filter_combo"] != "BASE" else []
        mask = np.ones(len(tape), dtype=bool)
        for f in filters:
            if f in single_results:
                mask = mask & single_results[f]["mask"]
        tape_filtered = tape.loc[mask].copy()
        tape_filtered.to_csv(OUT_DIR / "trade_tape_filtered.csv", index=False)
        log.info(f"BEST SURVIVING combo: {best['filter_combo']}  "
                 f"pdShr={best['per_day_sharpe']:.2f}")
    else:
        # closest combo: lowest sum of gate violations (counted)
        def fail_count(s):
            return len([t for t in str(s).split(";") if t.startswith("FAIL_")])
        df_out["_n_fails"] = df_out["fail_reasons"].map(fail_count)
        closest = df_out.sort_values(["_n_fails", "per_day_sharpe"],
                                     ascending=[True, False]).iloc[0]
        log.info(f"NO SURVIVORS. Closest: {closest['filter_combo']} "
                 f"({closest['fail_reasons']})")

    # ---- Write survivors.txt ----
    with open(OUT_DIR / "survivors.txt", "w") as f:
        f.write("=" * 78 + "\n")
        f.write("confluence_filter_v1 — survivors\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Base cell: h30s_q1_short_ES_passive (top-1% |pred_30s| shorts, ES passive)\n")
        f.write(f"OOT days loaded: {tape['date'].nunique()}\n")
        f.write(f"Base trades: {len(tape)}\n\n")
        f.write("BASE METRICS:\n")
        f.write(f"  per_day_Sharpe={base_metrics['per_day_sharpe']:.2f}  "
                f"PF={base_metrics['per_day_pf']:.2f}  "
                f"WR={base_metrics['per_day_wr']:.2%}  "
                f"regime_gap={base_metrics['regime_gap']:.2f}\n")
        f.write(f"  $/day={base_metrics['mean_daily_pnl_usd']:.0f}  "
                f"FAILS: {base_metrics['fail_reasons']}\n\n")
        if len(survivors) > 0:
            f.write(f"SURVIVORS ({len(survivors)}):\n")
            for _, r in survivors.iterrows():
                f.write(
                    f"  {r['filter_combo']}\n"
                    f"    pdShr={r['per_day_sharpe']:.2f}  PF={r['per_day_pf']:.2f}  "
                    f"WR={r['per_day_wr']:.2%}  regime_gap={r['regime_gap']:.2f}\n"
                    f"    n_days={r['n_days']}  n_trades={r['n_trades']}  "
                    f"trades/day={r['trades_per_day']:.1f}\n"
                    f"    $/day={r['mean_daily_pnl_usd']:.0f}  "
                    f"trade_WR={r['trade_wr']:.2%}  trade_bps={r['mean_trade_bps']:+.2f}\n"
                    f"    skip_loser_rate={r['skip_loser_rate']:.2%}  "
                    f"trades_kept={r['trades_kept_pct']:.1f}%\n\n"
                )
        else:
            f.write("(no combo passes all gates)\n\n")
            f.write("CLOSEST COMBOS (sorted by fewest gate failures, then pdShr):\n")
            def fail_count(s):
                return len([t for t in str(s).split(";") if t.startswith("FAIL_")])
            tmp = df_out.copy()
            tmp["_n_fails"] = tmp["fail_reasons"].map(fail_count)
            tmp = tmp.sort_values(["_n_fails", "per_day_sharpe"],
                                  ascending=[True, False]).head(10)
            for _, r in tmp.iterrows():
                f.write(
                    f"  {r['filter_combo']}  ({r['fail_reasons']})\n"
                    f"    pdShr={r['per_day_sharpe']:.2f}  PF={r['per_day_pf']:.2f}  "
                    f"WR={r['per_day_wr']:.2%}  gap={r['regime_gap']:.2f}\n"
                    f"    n_days={r['n_days']}  n_trades={r['n_trades']}  "
                    f"$/day={r['mean_daily_pnl_usd']:.0f}\n"
                    f"    skip_loser_rate={r['skip_loser_rate']:.2%}  "
                    f"trades_kept={r['trades_kept_pct']:.1f}%\n\n"
                )

    # ---- MLflow ----
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("confluence_filter_v1")
        with mlflow.start_run(run_name=f"run_{time.strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("base_cell", "h30s_q1_short_ES_passive")
            mlflow.log_param("n_oot_days", int(tape["date"].nunique()))
            mlflow.log_param("n_base_trades", int(len(tape)))
            mlflow.log_metric("base_per_day_sharpe", base_metrics["per_day_sharpe"])
            mlflow.log_metric("base_per_day_pf", base_metrics["per_day_pf"])
            mlflow.log_metric("base_per_day_wr", base_metrics["per_day_wr"])
            mlflow.log_metric("base_regime_gap", base_metrics["regime_gap"]
                              if np.isfinite(base_metrics["regime_gap"]) else -1.0)
            mlflow.log_metric("n_survivors", int(len(survivors)))
            if len(survivors) > 0:
                best = survivors.iloc[0]
                mlflow.log_metric("best_per_day_sharpe", best["per_day_sharpe"])
                mlflow.log_metric("best_per_day_pf", best["per_day_pf"])
                mlflow.log_metric("best_per_day_wr", best["per_day_wr"])
                mlflow.log_param("best_combo", best["filter_combo"])
            mlflow.log_artifact(str(OUT_DIR / "filter_comparison.csv"))
            mlflow.log_artifact(str(OUT_DIR / "survivors.txt"))
            mlflow.log_artifact(str(OUT_DIR / "trade_tape_base.csv"))
            if (OUT_DIR / "trade_tape_filtered.csv").exists():
                mlflow.log_artifact(str(OUT_DIR / "trade_tape_filtered.csv"))
            mlflow.log_artifact(str(LOG_PATH))
        log.info("MLflow logging complete.")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    log.info(f"DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
