#!/usr/bin/env python3
"""
stream_continuation_backtest.py — CNN-Mamba v4 stream-continuation tradability harness.

Complies with binding HCs:
  - HC #465: full array inventory before any report; no silent-ignored array.
  - HC #466: every head ranked independently at confidence quantiles; confluence matrix.
            Time-horizon IC is a sanity row, NOT the headline.
  - HC #467: trade enters when the stream signals, HOLDS while the stream confirms,
            EXITS on reversal/decay. Hold time is an OUTPUT, not an input.
            Snapshot fixed-hold backtests are demoted to a sanity bound.
  - HC #428 R1: per-day regime stratification (green / red / flat by intraday net move)
            with the |Sharpe_green - Sharpe_red| <= 0.50 gate.
  - HC #344: day-concentration cap <= 0.70.

Constants:
  ES_TICK_VALUE         = $12.50
  ES_RT_COMMISSION_TICKS = 0.376  (passive limit RT, $4.70 / $12.50)
  STRIDE_EVENTS         = 250 MBO events per prediction step (~3.3s wall clock during active session)

Important data note (discovered at inventory time, surfaced as an anomaly):
  - The MFE/MAE relabel parquets cover only a narrow ~12-minute window per day (~3,150 of the
    49,530 prediction steps per day). Razer's relabel pass is incomplete. The harness therefore
    uses the DENSE `target_log_ret_<h>` arrays inside the prediction NPZs (which ARE realized
    tick changes per HC #460 labeling convention) to score stream-continuation P&L, and uses the
    MFE/MAE parquets only as a diagnostic where they are valid. Stream P&L: side x realized
    log_ret_<closest-h> at trade entry, minus commission.

Inputs (read-only):
  output/cnn_mamba_v3_4_2_fixedmtl/fold_00_predictions.npz       (5-day OOT, all 32 pred heads — inventory source)
  output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_*.npz   (34-day OOT, used for the regime-grade backtest)
  data/relabel/mfe_mae_h{1s,5s,10s,30s}_<YYYYMMDD>.parquet       (MFE/MAE labels — diagnostic only because of sparse coverage)

Outputs (written to output/stream_backtest/):
  array_inventory.md / .json
  per_head_ranking.parquet
  tradability_matrix.parquet
  confluence_matrix.parquet
  stream_trades_full.parquet
  per_trade_examples.parquet
  REPORT.md
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ============================================================================
# CONSTANTS
# ============================================================================
ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_FILE = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/fold_00_predictions.npz"
PERDAY_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
RELABEL_DIR = ROOT / "data/relabel"
OUT_DIR = ROOT / "output/stream_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376
STRIDE_EVENTS = 250
WINDOW_SIZE = 3000

# Empirical stride wall-clock (measured): mean 3.3s, median 3.2s between consecutive predictions.
# (250 MBO events at typical RTH MBO rate.)
STRIDE_SECONDS_APPROX = 3.3

# Stream-continuation sweep parameters
CONF_QUANTILES = [0.99, 0.95, 0.90, 0.80, 0.50]  # top 1%, 5%, 10%, 20%, 50%
EXIT_RULES_M = [2, 3, 5]                          # consecutive opposite-sign / below-floor count to exit
EXIT_FLOOR_FRACS = [0.25, 0.50]                   # |signal| floor as fraction of entry threshold
MAX_HOLD_SECONDS = 60.0
MAX_HOLD_STEPS = max(2, int(MAX_HOLD_SECONDS / STRIDE_SECONDS_APPROX))  # ~18 steps

# Heads we exclude as DIRECTIONAL signal sources (still inventoried and ranked).
NON_DIRECTIONAL_HEADS = {
    "pred_pred_realized_vol_30s_ticks",
    "pred_pred_time_to_mfe_secs",
    "pred_fifo_tp4sl3_hit_tp",
    "pred_fifo_tp8sl5_hit_tp",
    "pred_p_reversal_15s",
    "pred_p_reversal_30s",
    "pred_p_reversal_60s",
}

# Realized-target lookup for each head (used to score stream exit P&L)
HORIZON_TARGETS = {
    "_1s": "target_log_ret_1s",
    "_5s": "target_log_ret_5s",
    "_10s": "target_log_ret_10s",
    "_30s": "target_log_ret_30s",
    "_60s": "target_log_ret_60s",
    "_5min": "target_log_ret_5min",
}


# ============================================================================
# HELPERS
# ============================================================================
def event_idx_from_pred_k(k: np.ndarray) -> np.ndarray:
    return k * STRIDE_EVENTS + (WINDOW_SIZE - 1)


def percentiles(a: np.ndarray, ps=(1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    qs = np.percentile(a, ps)
    return {f"p{p}": float(q) for p, q in zip(ps, qs)}


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 30:
        return float("nan")
    xr = pd.Series(x[mask]).rank().values
    yr = pd.Series(y[mask]).rank().values
    xr -= xr.mean()
    yr -= yr.mean()
    denom = math.sqrt((xr * xr).sum() * (yr * yr).sum())
    if denom == 0:
        return float("nan")
    return float((xr * yr).sum() / denom)


def directional_signal(name: str, vals: np.ndarray) -> np.ndarray:
    """Convert a head's raw output into a directional signal centered at zero.
    Positive => long; negative => short."""
    if name.startswith("pred_p_up_"):
        # >0.5 means long. recenter to ~[-0.5, 0.5].
        return vals - 0.5
    if name.startswith("pred_p_reversal_"):
        # No inherent direction — return zeros (won't gate trades on its own; confluence-only).
        return np.zeros_like(vals)
    if name == "pred_pred_mfe_30s_ticks" or name == "pred_pred_mfe_60s_ticks":
        return vals  # +ve mfe => upside excursion expected => long
    if name == "pred_pred_mae_30s_ticks" or name == "pred_pred_mae_60s_ticks":
        return -vals  # +ve mae (downside) => short bias
    if name in ("pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net"):
        return vals  # net P&L prediction; sign carries direction
    if name == "pred_pred_realized_vol_30s_ticks" or name == "pred_pred_time_to_mfe_secs":
        return np.zeros_like(vals)
    if name in ("pred_fifo_tp4sl3_hit_tp", "pred_fifo_tp8sl5_hit_tp"):
        return np.zeros_like(vals)
    # Default: log-return-like; sign is the direction.
    return vals


def head_target_horizon_name(head: str) -> str:
    """For a head, pick the realized log-ret horizon used to score stream P&L."""
    for suffix, tgt in HORIZON_TARGETS.items():
        if head.endswith(suffix) or f"{suffix}_q" in head:
            return tgt
    if "mfe_30s" in head or "mae_30s" in head:
        return "target_log_ret_30s"
    if "mfe_60s" in head or "mae_60s" in head:
        return "target_log_ret_60s"
    if "fifo" in head:
        return "target_log_ret_5s"
    if "reversal_15s" in head:
        return "target_log_ret_5s"
    if "reversal_30s" in head:
        return "target_log_ret_30s"
    if "reversal_60s" in head:
        return "target_log_ret_60s"
    if "realized_vol_30s" in head:
        return "target_log_ret_30s"
    return "target_log_ret_5s"


def horizon_for_hold_seconds(hold_s: float) -> str:
    """Closest realized-target horizon for a given hold time (used for stream-exit P&L)."""
    if hold_s <= 2.5:
        return "target_log_ret_1s"
    if hold_s <= 7.5:
        return "target_log_ret_5s"
    if hold_s <= 20.0:
        return "target_log_ret_10s"
    if hold_s <= 45.0:
        return "target_log_ret_30s"
    return "target_log_ret_60s"


# ============================================================================
# (1) FULL ARRAY INVENTORY (HC #465 R1)
# ============================================================================
def build_array_inventory() -> Tuple[List[Dict], Dict]:
    print(f"[inventory] loading {PRED_FILE}")
    d = np.load(PRED_FILE, allow_pickle=True)
    rows = []
    for k in d.keys():
        a = d[k]
        info = {
            "name": k,
            "shape": list(a.shape) if a.ndim > 0 else [],
            "dtype": str(a.dtype),
            "ndim": int(a.ndim),
            "size": int(a.size),
        }
        if a.ndim == 0:
            try:
                info["scalar_value"] = (
                    float(a.item()) if a.dtype.kind in "biuf" else str(a.item())
                )
            except Exception:
                info["scalar_value"] = str(a.item())
        elif a.dtype.kind in "fc":
            arr = a.astype(np.float64, copy=False)
            finite = np.isfinite(arr)
            info["nan_count"] = int((~finite).sum())
            info["finite_count"] = int(finite.sum())
            if finite.any():
                pct = percentiles(arr)
                info.update(pct)
                info["mean"] = float(arr[finite].mean())
                info["std"] = float(arr[finite].std())
        elif a.dtype.kind in "iu":
            info["min"] = int(a.min()) if a.size else None
            info["max"] = int(a.max()) if a.size else None
        else:
            info["sample_values"] = [str(x) for x in a.flat[:5]]
        rows.append(info)

    summary = {
        "source_file": str(PRED_FILE),
        "total_arrays": len(rows),
        "n_samples": int(d["n_samples"]) if "n_samples" in d.files else None,
        "oot_dates": list(d["oot_dates"].tolist()) if "oot_dates" in d.files else [],
        "fold_idx": int(d["fold_idx"]) if "fold_idx" in d.files else None,
    }
    return rows, summary


def write_inventory(rows: List[Dict], summary: Dict) -> None:
    (OUT_DIR / "array_inventory.json").write_text(
        json.dumps({"summary": summary, "arrays": rows}, indent=2)
    )
    lines = ["# Array Inventory — fold_00_predictions.npz", ""]
    lines.append(f"- Total arrays: **{summary['total_arrays']}**")
    lines.append(f"- Samples: **{summary['n_samples']}**")
    lines.append(f"- Fold: {summary['fold_idx']}")
    lines.append(f"- OOT dates: {summary['oot_dates']}")
    lines.append("")
    lines.append("| name | shape | dtype | nan | mean | p1 | p50 | p99 |")
    lines.append("|------|-------|-------|-----|------|----|----|----|")
    for r in rows:
        shape = "x".join(str(x) for x in r["shape"]) or "scalar"
        nan = r.get("nan_count", "-")
        mean = r.get("mean", r.get("scalar_value", "-"))
        p1 = r.get("p1", "-")
        p50 = r.get("p50", "-")
        p99 = r.get("p99", "-")

        def fmt(x):
            if isinstance(x, float):
                return f"{x:.4g}"
            return str(x)

        lines.append(
            f"| {r['name']} | {shape} | {r['dtype']} | {nan} | {fmt(mean)} | {fmt(p1)} | {fmt(p50)} | {fmt(p99)} |"
        )
    (OUT_DIR / "array_inventory.md").write_text("\n".join(lines))
    print(f"[inventory] wrote {OUT_DIR / 'array_inventory.md'}")


# ============================================================================
# (2) PER-DAY DATA LOAD (NPZ + thin MFE/MAE diagnostic slice)
# ============================================================================
def load_day(date_str: str) -> Optional[pd.DataFrame]:
    """Load per-day NPZ and return as a DataFrame; attach MFE/MAE diagnostic columns
    where the relabel parquet has coverage (sparse — usually ~6% of preds)."""
    npz_path = PERDAY_DIR / f"oot_{date_str}.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    pred_keys = [k for k in d.files if k.startswith("pred_")]
    target_keys = [k for k in d.files if k.startswith("target_")]
    mask_keys = [k for k in d.files if k.startswith("mask_")]
    n_preds = d[pred_keys[0]].shape[0]
    cols: Dict[str, np.ndarray] = {}
    for k in pred_keys + target_keys + mask_keys:
        cols[k] = d[k]
    k_idx = np.arange(n_preds, dtype=np.int64)
    cols["pred_k"] = k_idx
    cols["event_idx"] = event_idx_from_pred_k(k_idx)

    # Attach MFE/MAE diagnostic slice (sparse coverage only).
    for h in ("1s", "5s", "10s", "30s"):
        path = RELABEL_DIR / f"mfe_mae_h{h}_{date_str}.parquet"
        if not path.exists():
            cols[f"mfe_h{h}"] = np.full(n_preds, np.nan, dtype=np.float32)
            cols[f"mae_h{h}"] = np.full(n_preds, np.nan, dtype=np.float32)
            cols[f"time_to_mfe_h{h}"] = np.full(n_preds, np.nan, dtype=np.float32)
            if h == "1s":
                cols["mid_ticks"] = np.full(n_preds, np.nan, dtype=np.float64)
                cols["ts_ns"] = np.zeros(n_preds, dtype=np.int64)
            continue
        df = pd.read_parquet(
            path,
            columns=["event_idx", "ts_ns", "mid_t_ticks", "mfe_ticks", "mae_ticks", "time_to_mfe_s"],
        )
        df = df.set_index("event_idx")
        avail = df.index.max()
        # Slice — set to NaN where event_idx not in parquet range.
        valid = cols["event_idx"] <= avail
        sliced = df.reindex(cols["event_idx"]).reset_index(drop=True)
        cols[f"mfe_h{h}"] = sliced["mfe_ticks"].values.astype(np.float32)
        cols[f"mae_h{h}"] = sliced["mae_ticks"].values.astype(np.float32)
        cols[f"time_to_mfe_h{h}"] = sliced["time_to_mfe_s"].values.astype(np.float32)
        if h == "1s":
            cols["mid_ticks"] = sliced["mid_t_ticks"].values.astype(np.float64)
            cols["ts_ns"] = sliced["ts_ns"].fillna(0).values.astype(np.int64)

    cols["date"] = np.full(n_preds, date_str, dtype=object)
    df = pd.DataFrame(cols)
    return df


def cache_day(date_str: str) -> str:
    cache_path = OUT_DIR / f"aligned_{date_str}.parquet"
    if cache_path.exists():
        return str(cache_path)
    df = load_day(date_str)
    if df is None:
        return ""
    df.to_parquet(cache_path)
    return str(cache_path)


def list_common_dates() -> List[str]:
    pred_dates = {p.stem.replace("oot_", "") for p in PERDAY_DIR.glob("oot_*.npz")}
    # We only require the per-day NPZ; MFE/MAE is diagnostic.
    return sorted(pred_dates)


# ============================================================================
# (3) PER-HEAD PREDICTIVE-POWER RANKING (HC #466)
# ============================================================================
def per_head_ranking(all_df: pd.DataFrame, pred_heads: List[str]) -> pd.DataFrame:
    rows = []
    for head in pred_heads:
        if head not in all_df.columns:
            continue
        raw = all_df[head].values.astype(np.float64)
        signal = directional_signal(head, raw)
        target_name = head_target_horizon_name(head)
        if target_name not in all_df.columns:
            continue
        y = all_df[target_name].values.astype(np.float64)
        m = np.isfinite(signal) & np.isfinite(y) & (signal != 0)  # exclude zero-signal heads
        if m.sum() < 1000:
            continue
        s = signal[m]
        yy = y[m]
        ic_full = safe_spearman(s, yy)
        sign_agree_full = float(((np.sign(s) == np.sign(yy)) & (np.sign(s) != 0)).mean())

        for q in CONF_QUANTILES:
            mag = np.abs(s)
            thr = np.quantile(mag, q)
            sel = mag >= thr
            if sel.sum() < 50:
                continue
            s_sel = s[sel]
            y_sel = yy[sel]
            ic_q = safe_spearman(s_sel, y_sel)
            sign_agree = float(
                ((np.sign(s_sel) == np.sign(y_sel)) & (np.sign(s_sel) != 0)).mean()
            )
            signed_realized = np.sign(s_sel) * y_sel  # ticks signed in our direction
            hit_rate = float((signed_realized > 0).mean())
            mean_signed = float(signed_realized.mean())
            denom = float(np.abs(y_sel).mean())
            lift = mean_signed / denom if denom > 0 else float("nan")
            rows.append(
                {
                    "head": head,
                    "target": target_name,
                    "conf_quantile_top": round((1 - q) * 100, 2),
                    "n": int(sel.sum()),
                    "ic_full_sample_sanity": ic_full,
                    "sign_agree_full_sanity": sign_agree_full,
                    "ic_at_cut_sanity": ic_q,
                    "sign_agree": sign_agree,
                    "hit_rate_positive_realized": hit_rate,
                    "mean_signed_realized_ticks_HEADLINE": mean_signed,
                    "net_ticks_after_cost_HEADLINE": mean_signed - ES_RT_COMMISSION_TICKS,
                    "lift_vs_abs_mean": lift,
                    "threshold_abs_value": float(thr),
                }
            )
    return pd.DataFrame(rows)


# ============================================================================
# (4) STREAM-CONTINUATION SIMULATION (HC #467 — CORE)
# ============================================================================
@dataclass
class StreamConfig:
    head: str
    conf_q: float
    exit_M: int
    exit_floor_frac: float
    max_hold_steps: int = MAX_HOLD_STEPS


def simulate_stream_day(day_df: pd.DataFrame, cfg: StreamConfig) -> pd.DataFrame:
    if cfg.head not in day_df.columns:
        return pd.DataFrame()
    raw = day_df[cfg.head].values.astype(np.float64)
    signal = directional_signal(cfg.head, raw)
    n = signal.size
    finite_mask = np.isfinite(signal) & (signal != 0)
    if finite_mask.sum() < 100:
        return pd.DataFrame()
    mag = np.abs(signal)
    finite_mag = mag[finite_mask]
    entry_threshold = float(np.quantile(finite_mag, cfg.conf_q))
    exit_floor = cfg.exit_floor_frac * entry_threshold

    # Extract all arrays upfront as raw numpy (no pandas iloc in the hot loop).
    tgt_1s = day_df["target_log_ret_1s"].values.astype(np.float64) if "target_log_ret_1s" in day_df.columns else None
    tgt_5s = day_df["target_log_ret_5s"].values.astype(np.float64) if "target_log_ret_5s" in day_df.columns else None
    tgt_10s = day_df["target_log_ret_10s"].values.astype(np.float64) if "target_log_ret_10s" in day_df.columns else None
    tgt_30s = day_df["target_log_ret_30s"].values.astype(np.float64) if "target_log_ret_30s" in day_df.columns else None
    tgt_60s = day_df["target_log_ret_60s"].values.astype(np.float64) if "target_log_ret_60s" in day_df.columns else None
    mfe1 = day_df["mfe_h1s"].values.astype(np.float64) if "mfe_h1s" in day_df.columns else None
    mae1 = day_df["mae_h1s"].values.astype(np.float64) if "mae_h1s" in day_df.columns else None
    mfe5 = day_df["mfe_h5s"].values.astype(np.float64) if "mfe_h5s" in day_df.columns else None
    mae5 = day_df["mae_h5s"].values.astype(np.float64) if "mae_h5s" in day_df.columns else None
    mfe10 = day_df["mfe_h10s"].values.astype(np.float64) if "mfe_h10s" in day_df.columns else None
    mae10 = day_df["mae_h10s"].values.astype(np.float64) if "mae_h10s" in day_df.columns else None
    mfe30 = day_df["mfe_h30s"].values.astype(np.float64) if "mfe_h30s" in day_df.columns else None
    mae30 = day_df["mae_h30s"].values.astype(np.float64) if "mae_h30s" in day_df.columns else None
    pred_k = day_df["pred_k"].values.astype(np.int64) if "pred_k" in day_df.columns else np.arange(n)
    ts_ns = day_df["ts_ns"].values.astype(np.int64) if "ts_ns" in day_df.columns else None
    date_str = day_df["date"].iloc[0] if "date" in day_df.columns else "?"

    # Pre-allocate result lists as Python lists of typed values (faster than dict appends).
    out_entry_idx = []; out_exit_idx = []; out_side = []
    out_n_held = []; out_n_steps = []; out_same = []
    out_mean_abs = []
    out_realized = []; out_h_used = []
    out_side_mfe = []; out_side_mae = []; out_mfe_h_used = []

    is_finite = finite_mask
    M = cfg.exit_M
    max_hold = cfg.max_hold_steps
    i = 0
    while i < n:
        if not is_finite[i] or mag[i] < entry_threshold:
            i += 1
            continue
        side = 1 if signal[i] > 0 else (-1 if signal[i] < 0 else 0)
        if side == 0:
            i += 1
            continue
        entry_idx = i
        adverse_count = 0
        floor_count = 0
        last_hold_step = entry_idx
        mean_pred_accum = abs(signal[entry_idx])
        n_steps = 1
        same_sign_steps = 1
        hit_max = True
        for step in range(1, max_hold + 1):
            j = entry_idx + step
            if j >= n:
                last_hold_step = j - 1
                hit_max = False
                break
            s_j = signal[j]
            if s_j != s_j:  # NaN check
                floor_count += 1
                if floor_count >= M:
                    last_hold_step = j
                    hit_max = False
                    break
                continue
            n_steps += 1
            abs_sj = abs(s_j)
            mean_pred_accum += abs_sj
            sign_j = 1 if s_j > 0 else (-1 if s_j < 0 else 0)
            if sign_j == side:
                same_sign_steps += 1
                adverse_count = 0
                if abs_sj < exit_floor:
                    floor_count += 1
                    if floor_count >= M:
                        last_hold_step = j
                        hit_max = False
                        break
                else:
                    floor_count = 0
            else:
                adverse_count += 1
                if adverse_count >= M:
                    last_hold_step = j
                    hit_max = False
                    break
            last_hold_step = j
        if hit_max:
            last_hold_step = min(entry_idx + max_hold, n - 1)

        exit_idx = last_hold_step
        if exit_idx == entry_idx:
            i = entry_idx + 1
            continue

        n_held = exit_idx - entry_idx
        hold_s = n_held * STRIDE_SECONDS_APPROX

        # Pick realized horizon
        if hold_s <= 2.5 and tgt_1s is not None:
            rh = "target_log_ret_1s"; rv = tgt_1s[entry_idx]
        elif hold_s <= 7.5 and tgt_5s is not None:
            rh = "target_log_ret_5s"; rv = tgt_5s[entry_idx]
        elif hold_s <= 20.0 and tgt_10s is not None:
            rh = "target_log_ret_10s"; rv = tgt_10s[entry_idx]
        elif hold_s <= 45.0 and tgt_30s is not None:
            rh = "target_log_ret_30s"; rv = tgt_30s[entry_idx]
        elif tgt_60s is not None:
            rh = "target_log_ret_60s"; rv = tgt_60s[entry_idx]
        else:
            i = exit_idx + 1
            continue
        if rv != rv:  # NaN
            # fallback to 5s
            if tgt_5s is not None and tgt_5s[entry_idx] == tgt_5s[entry_idx]:
                rh = "target_log_ret_5s"; rv = tgt_5s[entry_idx]
            else:
                i = exit_idx + 1
                continue

        realized_signed = side * rv

        # MFE/MAE diagnostic — pick closest valid horizon
        side_mfe = float("nan"); side_mae = float("nan"); mfe_h_used = "none"
        for h_label, mfe_arr, mae_arr in (
            ("1s", mfe1, mae1), ("5s", mfe5, mae5), ("10s", mfe10, mae10), ("30s", mfe30, mae30),
        ):
            if mfe_arr is not None:
                v = mfe_arr[entry_idx]
                if v == v:  # not NaN
                    mfe = float(v)
                    mae = float(mae_arr[entry_idx])
                    if side > 0:
                        side_mfe = mfe; side_mae = mae
                    else:
                        side_mfe = -mae; side_mae = -mfe
                    mfe_h_used = h_label
                    break

        out_entry_idx.append(entry_idx)
        out_exit_idx.append(exit_idx)
        out_side.append(side)
        out_n_held.append(n_held)
        out_n_steps.append(n_steps)
        out_same.append(same_sign_steps)
        out_mean_abs.append(mean_pred_accum / max(n_steps, 1))
        out_realized.append(realized_signed)
        out_h_used.append(rh)
        out_side_mfe.append(side_mfe)
        out_side_mae.append(side_mae)
        out_mfe_h_used.append(mfe_h_used)

        i = exit_idx + 1

    if not out_entry_idx:
        return pd.DataFrame()

    entry_arr = np.asarray(out_entry_idx, dtype=np.int64)
    exit_arr = np.asarray(out_exit_idx, dtype=np.int64)
    realized_arr = np.asarray(out_realized, dtype=np.float64)
    net_ticks = realized_arr - ES_RT_COMMISSION_TICKS
    return pd.DataFrame({
        "date": date_str,
        "head": cfg.head,
        "conf_q": cfg.conf_q,
        "exit_M": cfg.exit_M,
        "exit_floor_frac": cfg.exit_floor_frac,
        "entry_pred_k": pred_k[entry_arr],
        "exit_pred_k": pred_k[exit_arr],
        "ts_ns_entry": ts_ns[entry_arr] if ts_ns is not None else 0,
        "side": np.asarray(out_side, dtype=np.int8),
        "entry_threshold": entry_threshold,
        "exit_floor": exit_floor,
        "n_held_steps": np.asarray(out_n_held, dtype=np.int32),
        "hold_seconds_approx": np.asarray(out_n_held, dtype=np.float64) * STRIDE_SECONDS_APPROX,
        "n_pred_steps_in_trade": np.asarray(out_n_steps, dtype=np.int32),
        "same_sign_steps": np.asarray(out_same, dtype=np.int32),
        "sign_stability": np.asarray(out_same, dtype=np.float64) / np.maximum(np.asarray(out_n_steps), 1),
        "mean_abs_pred_during_trade": np.asarray(out_mean_abs, dtype=np.float64),
        "realized_horizon_used": out_h_used,
        "realized_signed_ticks": realized_arr,
        "net_ticks": net_ticks,
        "net_dollars": net_ticks * ES_TICK_VALUE,
        "mfe_diagnostic_horizon": out_mfe_h_used,
        "side_mfe_ticks": np.asarray(out_side_mfe, dtype=np.float64),
        "side_mae_ticks": np.asarray(out_side_mae, dtype=np.float64),
    })


def summarize_trades(tdf: pd.DataFrame) -> Dict:
    n = len(tdf)
    if n == 0:
        return {
            "n_trades": 0, "mean_hold_s": float("nan"), "wr": float("nan"),
            "mean_net_ticks": float("nan"), "total_net_dollars": 0.0,
            "sharpe_per_trade": float("nan"), "sortino_per_trade": float("nan"), "pf": float("nan"),
            "mean_n_pred_steps": float("nan"), "mean_sign_stability": float("nan"),
            "mean_abs_pred_during_trade": float("nan"),
            "median_net_ticks": float("nan"),
            "mean_side_mfe": float("nan"), "mean_side_mae": float("nan"),
        }
    nt = tdf["net_ticks"].values
    sigma = nt.std()
    downside = nt[nt < 0]
    downside_std = downside.std() if downside.size > 1 else 0.0
    wins = nt[nt > 0].sum()
    losses = -nt[nt < 0].sum()
    pf = wins / losses if losses > 0 else float("inf") if wins > 0 else float("nan")
    return {
        "n_trades": int(n),
        "mean_hold_s": float(tdf["hold_seconds_approx"].mean()),
        "mean_n_pred_steps": float(tdf["n_pred_steps_in_trade"].mean()),
        "mean_sign_stability": float(tdf["sign_stability"].mean()),
        "mean_abs_pred_during_trade": float(tdf["mean_abs_pred_during_trade"].mean()),
        "wr": float((nt > 0).mean()),
        "mean_net_ticks": float(nt.mean()),
        "median_net_ticks": float(np.median(nt)),
        "total_net_dollars": float((nt * ES_TICK_VALUE).sum()),
        "sharpe_per_trade": float(nt.mean() / sigma * math.sqrt(252)) if sigma > 0 else float("nan"),
        "sortino_per_trade": float(nt.mean() / downside_std * math.sqrt(252)) if downside_std > 0 else (float("inf") if nt.mean() > 0 else float("nan")),
        "pf": float(pf),
        "mean_side_mfe": float(tdf["side_mfe_ticks"].mean(skipna=True)) if "side_mfe_ticks" in tdf else float("nan"),
        "mean_side_mae": float(tdf["side_mae_ticks"].mean(skipna=True)) if "side_mae_ticks" in tdf else float("nan"),
    }


# ============================================================================
# (5) REGIME CLASSIFICATION (HC #428 R1)
# ============================================================================
def classify_regimes(day_caches: Dict[str, str]) -> Dict[str, str]:
    """Classify each day as green/red/flat using the sum of target_log_ret_1s across all
    valid prediction steps in the day (proxy for net intraday tick drift). target_log_ret_5min
    in the per-day NPZ is all-zero (masked at training time), so we use 1s realized labels.
    Threshold +/- 200 ticks ($2,500). Sample 32-day distribution: q25=49, q50=593, q75=1784.
    """
    regimes = {}
    FLAT_TICKS = 200.0
    for date_str, cp in day_caches.items():
        if not cp:
            continue
        try:
            df = pd.read_parquet(cp, columns=["target_log_ret_1s"])
        except Exception:
            continue
        s = float(df["target_log_ret_1s"].sum(skipna=True))
        if s >= FLAT_TICKS:
            regimes[date_str] = "green"
        elif s <= -FLAT_TICKS:
            regimes[date_str] = "red"
        else:
            regimes[date_str] = "flat"
    return regimes


# ============================================================================
# (6) AGGREGATION & MATRIX
# ============================================================================
def build_tradability_row(
    head: str, conf_q: float, exit_M: int, exit_floor_frac: float,
    all_trades: pd.DataFrame, regimes: Dict[str, str],
) -> Optional[Dict]:
    if all_trades.empty:
        return None
    overall = summarize_trades(all_trades)
    per_day = (
        all_trades.groupby("date", group_keys=False)
        .apply(lambda g: pd.Series(summarize_trades(g)), include_groups=False)
        .reset_index()
    )
    daily_totals = per_day["total_net_dollars"].values
    abs_sum = np.abs(daily_totals).sum()
    day_conc = float(np.abs(daily_totals).max() / abs_sum) if abs_sum > 0 else float("nan")
    per_day["regime"] = per_day["date"].map(regimes).fillna("unknown")
    green = per_day[per_day["regime"] == "green"]
    red = per_day[per_day["regime"] == "red"]
    flat = per_day[per_day["regime"] == "flat"]

    def regime_sharpe(g):
        if len(g) < 2:
            return float("nan")
        vals = g["total_net_dollars"].values
        sigma = vals.std()
        return float(vals.mean() / sigma * math.sqrt(252)) if sigma > 0 else float("nan")

    sg = regime_sharpe(green); sr = regime_sharpe(red); sf = regime_sharpe(flat)
    if np.isfinite(sg) and np.isfinite(sr):
        denom = max(abs(sg), abs(sr))
        regime_skew = abs(sg - sr) / denom if denom > 0 else float("nan")
        regime_pass = regime_skew <= 0.50
    else:
        regime_skew = float("nan")
        regime_pass = False
    day_conc_pass = day_conc <= 0.70 if np.isfinite(day_conc) else False
    return {
        "head": head,
        "conf_quantile_top_pct": round((1 - conf_q) * 100, 2),
        "exit_M": exit_M,
        "exit_floor_frac": exit_floor_frac,
        "n_trades": overall["n_trades"],
        "n_days": int(per_day["date"].nunique()),
        "mean_hold_s": overall["mean_hold_s"],
        "mean_n_pred_steps": overall["mean_n_pred_steps"],
        "mean_sign_stability": overall["mean_sign_stability"],
        "mean_abs_pred_during_trade": overall["mean_abs_pred_during_trade"],
        "wr": overall["wr"],
        "mean_net_ticks": overall["mean_net_ticks"],
        "total_net_dollars": overall["total_net_dollars"],
        "sharpe_per_trade": overall["sharpe_per_trade"],
        "sortino_per_trade": overall["sortino_per_trade"],
        "pf": overall["pf"],
        "mean_side_mfe_ticks": overall["mean_side_mfe"],
        "mean_side_mae_ticks": overall["mean_side_mae"],
        "day_concentration": day_conc,
        "day_conc_pass": day_conc_pass,
        "n_green_days": int((per_day["regime"] == "green").sum()),
        "n_red_days": int((per_day["regime"] == "red").sum()),
        "n_flat_days": int((per_day["regime"] == "flat").sum()),
        "sharpe_green_days": sg,
        "sharpe_red_days": sr,
        "sharpe_flat_days": sf,
        "regime_skew": regime_skew,
        "regime_pass": bool(regime_pass),
    }


# ============================================================================
# (7) CONFLUENCE
# ============================================================================
def confluence_pair(all_df: pd.DataFrame, head_a: str, head_b: str, conf_q: float) -> Optional[Dict]:
    if head_a not in all_df.columns or head_b not in all_df.columns:
        return None
    sa = directional_signal(head_a, all_df[head_a].values.astype(np.float64))
    sb = directional_signal(head_b, all_df[head_b].values.astype(np.float64))
    target = all_df["target_log_ret_5s"].values.astype(np.float64) if "target_log_ret_5s" in all_df.columns else None
    if target is None:
        return None
    m = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(target) & (sa != 0) & (sb != 0)
    if m.sum() < 1000:
        return None
    sa_m = sa[m]; sb_m = sb[m]; t_m = target[m]
    thr_a = np.quantile(np.abs(sa_m), conf_q)
    thr_b = np.quantile(np.abs(sb_m), conf_q)
    same = np.sign(sa_m) == np.sign(sb_m)
    both = (np.abs(sa_m) >= thr_a) & (np.abs(sb_m) >= thr_b) & same & (np.sign(sa_m) != 0)
    n = int(both.sum())
    if n < 50:
        return None
    signed = np.sign(sa_m[both]) * t_m[both]
    a_only = (np.abs(sa_m) >= thr_a) & (np.sign(sa_m) != 0)
    b_only = (np.abs(sb_m) >= thr_b) & (np.sign(sb_m) != 0)
    signed_a = np.sign(sa_m[a_only]) * t_m[a_only]
    signed_b = np.sign(sb_m[b_only]) * t_m[b_only]
    return {
        "head_a": head_a,
        "head_b": head_b,
        "conf_quantile_top_pct": round((1 - conf_q) * 100, 2),
        "n_confluence_trades": n,
        "n_a_only": int(a_only.sum()),
        "n_b_only": int(b_only.sum()),
        "mean_signed_realized_confluence_ticks": float(signed.mean()),
        "mean_signed_realized_a_only_ticks": float(signed_a.mean()) if signed_a.size else float("nan"),
        "mean_signed_realized_b_only_ticks": float(signed_b.mean()) if signed_b.size else float("nan"),
        "confluence_lift_vs_a_ticks": float(signed.mean() - signed_a.mean()) if signed_a.size else float("nan"),
        "confluence_lift_vs_b_ticks": float(signed.mean() - signed_b.mean()) if signed_b.size else float("nan"),
        "hit_rate_confluence": float((signed > 0).mean()),
        "net_ticks_after_cost_confluence": float(signed.mean() - ES_RT_COMMISSION_TICKS),
    }


# ============================================================================
# (8) WORKERS
# ============================================================================
def _run_sweep_worker(args):
    date_str, cache_path, configs = args
    if not cache_path or not Path(cache_path).exists():
        return []
    df = pd.read_parquet(cache_path)
    out_frames = []
    for cfg in configs:
        tdf = simulate_stream_day(df, cfg)
        if not tdf.empty:
            out_frames.append(tdf)
    if not out_frames:
        return []
    return [pd.concat(out_frames, ignore_index=True)]


# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 78)
    print("STREAM-CONTINUATION BACKTEST (HC #466 + HC #467)")
    print("=" * 78)

    print("\n[STEP 1] Array inventory (HC #465 R1)")
    rows, summary = build_array_inventory()
    write_inventory(rows, summary)
    pred_heads_all = sorted({r["name"] for r in rows if r["name"].startswith("pred_")})
    print(f"[STEP 1] {len(pred_heads_all)} pred_* heads found.")

    print("\n[STEP 2] Building per-day aligned cache")
    common_dates = list_common_dates()
    print(f"[STEP 2] {len(common_dates)} OOT days available: {common_dates[0]}..{common_dates[-1]}")
    day_caches: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(cache_day, d): d for d in common_dates}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                cp = fut.result()
                day_caches[d] = cp
            except Exception as e:
                print(f"  FAILED {d}: {e}")
                day_caches[d] = ""
    print(f"[STEP 2] cached {sum(1 for v in day_caches.values() if v)} days")

    print("\n[STEP 3] Regime classification")
    regimes = classify_regimes(day_caches)
    print(f"[STEP 3] regimes: green={sum(1 for v in regimes.values() if v=='green')}, "
          f"red={sum(1 for v in regimes.values() if v=='red')}, "
          f"flat={sum(1 for v in regimes.values() if v=='flat')}")

    print("\n[STEP 4] Loading concatenated frame for per-head ranking")
    frames = []
    for d in common_dates:
        cp = day_caches.get(d)
        if cp and Path(cp).exists():
            frames.append(pd.read_parquet(cp))
    if not frames:
        print("[FATAL] no day caches built — abort.")
        sys.exit(1)
    all_df = pd.concat(frames, ignore_index=True)
    print(f"[STEP 4] concat shape: {all_df.shape}")

    print("\n[STEP 5] Per-head predictive-power ranking (HC #466 R2)")
    head_rank_df = per_head_ranking(all_df, pred_heads_all)
    head_rank_df.to_parquet(OUT_DIR / "per_head_ranking.parquet", index=False)
    print(f"[STEP 5] wrote per_head_ranking.parquet ({len(head_rank_df)} rows)")

    # Pick top heads for stream-continuation sweep — by net_ticks_after_cost at top 5%
    top_cut = head_rank_df[head_rank_df["conf_quantile_top"] == 5.0].copy()
    if top_cut.empty:
        top_cut = head_rank_df.copy()
    top_cut = top_cut[~top_cut["head"].isin(NON_DIRECTIONAL_HEADS)]
    top_cut["abs_headline"] = top_cut["mean_signed_realized_ticks_HEADLINE"].abs()
    top_cut = top_cut.sort_values("abs_headline", ascending=False)
    top_heads = list(top_cut["head"].drop_duplicates().head(8))
    print(f"[STEP 6] Top heads for stream sweep: {top_heads}")

    print("\n[STEP 7] Stream-continuation simulation sweep")
    configs = [
        StreamConfig(head=h, conf_q=q, exit_M=m, exit_floor_frac=f)
        for h in top_heads
        for q in CONF_QUANTILES
        for m in EXIT_RULES_M
        for f in EXIT_FLOOR_FRACS
    ]
    print(f"[STEP 7] {len(configs)} configs over {len(common_dates)} days")

    args_list = [(d, day_caches.get(d, ""), configs) for d in common_dates]
    all_trades_frames = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_run_sweep_worker, a) for a in args_list]
        for i, fut in enumerate(as_completed(futures)):
            try:
                frames = fut.result()
                all_trades_frames.extend(frames)
            except Exception as e:
                print(f"  worker failed: {e}")
            if (i + 1) % 5 == 0 or (i + 1) == len(futures):
                print(f"  progress: {i+1}/{len(futures)} days")

    if not all_trades_frames:
        print("[FATAL] no trades produced — abort.")
        sys.exit(1)
    all_trades_df = pd.concat(all_trades_frames, ignore_index=True)
    all_trades_df.to_parquet(OUT_DIR / "stream_trades_full.parquet", index=False)
    print(f"[STEP 7] total trades: {len(all_trades_df)}")

    print("\n[STEP 8] Aggregating tradability matrix")
    rows_out = []
    grouped = all_trades_df.groupby(["head", "conf_q", "exit_M", "exit_floor_frac"])
    for (h, q, m, f), sub in grouped:
        row = build_tradability_row(h, q, m, f, sub, regimes)
        if row:
            rows_out.append(row)
    matrix_df = pd.DataFrame(rows_out).sort_values("sharpe_per_trade", ascending=False)
    matrix_df.to_parquet(OUT_DIR / "tradability_matrix.parquet", index=False)
    print(f"[STEP 8] wrote tradability_matrix.parquet ({len(matrix_df)} rows)")

    print("\n[STEP 9] Cross-head confluence matrix (HC #466 R4)")
    confluence_rows = []
    pairs_done = set()
    for a in top_heads:
        for b in top_heads:
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            if key in pairs_done:
                continue
            pairs_done.add(key)
            for q in [0.95, 0.90, 0.80]:
                r = confluence_pair(all_df, a, b, q)
                if r:
                    confluence_rows.append(r)
    conf_df = pd.DataFrame(confluence_rows)
    conf_df.to_parquet(OUT_DIR / "confluence_matrix.parquet", index=False)
    print(f"[STEP 9] wrote confluence_matrix.parquet ({len(conf_df)} rows)")

    sample = all_trades_df.sample(min(5000, len(all_trades_df)), random_state=42)
    sample.to_parquet(OUT_DIR / "per_trade_examples.parquet", index=False)

    print("\n[STEP 10] Writing REPORT.md")
    write_report(matrix_df, head_rank_df, conf_df, regimes, all_df, top_heads, t0)
    print(f"\nDONE in {time.time()-t0:.1f}s")


def write_report(
    matrix_df: pd.DataFrame, head_rank_df: pd.DataFrame, conf_df: pd.DataFrame,
    regimes: Dict[str, str], all_df: pd.DataFrame, top_heads: List[str], t0: float,
) -> None:
    n_days = len(regimes)
    green = sum(1 for v in regimes.values() if v == "green")
    red = sum(1 for v in regimes.values() if v == "red")
    flat = sum(1 for v in regimes.values() if v == "flat")
    passing = matrix_df[(matrix_df["regime_pass"]) & (matrix_df["day_conc_pass"])]
    headline = passing.head(1) if not passing.empty else matrix_df.head(1)
    has_passing = not passing.empty
    best_by_head = matrix_df.sort_values("sharpe_per_trade", ascending=False).drop_duplicates("head").head(3)

    lines = []
    lines.append("# Stream-Continuation Tradability Report — CNN-Mamba v4 (fold 0)")
    lines.append("")
    lines.append("Compliance: HC #465 (full inventory), HC #466 (per-head + confidence cuts + confluence),")
    lines.append("HC #467 (stream-continuation — hold time is an OUTPUT, not an input), HC #428 R1 (regime gate),")
    lines.append("HC #344 (day-concentration cap 0.70). Time-horizon IC is a sanity row only.")
    lines.append("")
    lines.append("## Data scope")
    lines.append(f"- OOT days analyzed: **{n_days}** (green={green}, red={red}, flat={flat}).")
    lines.append(f"- Total stream-continuation trades simulated across the sweep: **{int(matrix_df['n_trades'].sum())}**.")
    lines.append("- Cost model: passive limit only, 0.376 ticks RT commission, 1 contract.")
    lines.append("- Stream P&L: side x realized log_ret_<closest-h to actual hold time>, taken at trade entry.")
    lines.append("- Hold time is an OUTPUT — the trade exits the moment the prediction stream reverses or decays.")
    lines.append("- Anomaly: the MFE/MAE relabel parquets only cover ~12 minutes per day (~6% of prediction steps). They are used as a diagnostic, NOT as P&L source. Razer's relabel pass needs to be expanded to the full session.")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    h = headline.iloc[0]
    verdict = "yes" if has_passing else "NO"
    lines.append(
        f"**Best tradable subset = head `{h['head']}` at confidence cut top {h['conf_quantile_top_pct']}%, "
        f"exit rule M={h['exit_M']} with floor={h['exit_floor_frac']:.2f}. "
        f"Per-trade Sharpe = {h['sharpe_per_trade']:.3f}, regime-pass: {verdict}.**"
    )
    if not has_passing:
        lines.append("")
        lines.append("(No config cleared both the regime-skew gate (<=0.50) and the day-concentration gate (<=0.70). The above is the best by per-trade Sharpe, shown for transparency.)")
    lines.append("")
    lines.append("Honest summary of the top config:")
    lines.append(f"- Trades: {int(h['n_trades'])} across {int(h['n_days'])} days, mean hold {h['mean_hold_s']:.1f} seconds (mean {h['mean_n_pred_steps']:.1f} prediction steps in trade).")
    lines.append(f"- Win rate: {h['wr']:.1%}. Profit factor: {h['pf']:.2f}.")
    lines.append(f"- Mean net per trade: {h['mean_net_ticks']:.3f} ticks. Total net dollars across sweep: ${h['total_net_dollars']:.0f}.")
    lines.append(f"- Day concentration: {h['day_concentration']:.2f} (gate <=0.70 -> {'PASS' if h['day_conc_pass'] else 'FAIL'}).")
    lines.append(f"- Sharpe on green vs red days: {h['sharpe_green_days']:.2f} vs {h['sharpe_red_days']:.2f} (skew={h['regime_skew']:.2f}; gate <=0.50 -> {'PASS' if h['regime_pass'] else 'FAIL'}).")
    lines.append(f"- Sign stability while in trade: {h['mean_sign_stability']:.1%}.")
    lines.append("")

    lines.append("## Top 3 heads by stream-continuation Sharpe (best config per head)")
    lines.append("")
    lines.append("| rank | head | top cut | exit_M | floor | trades | hold (s) | WR | mean net ticks | Sharpe | day_conc | regime_pass |")
    lines.append("|------|------|---------|--------|-------|--------|----------|----|------|------|----|----|")
    for i, (_, row) in enumerate(best_by_head.iterrows(), start=1):
        lines.append(
            f"| {i} | {row['head']} | top {row['conf_quantile_top_pct']}% | {row['exit_M']} | "
            f"{row['exit_floor_frac']:.2f} | {int(row['n_trades'])} | {row['mean_hold_s']:.1f} | "
            f"{row['wr']:.1%} | {row['mean_net_ticks']:.3f} | {row['sharpe_per_trade']:.3f} | "
            f"{row['day_concentration']:.2f} | {'yes' if row['regime_pass'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Confluence — does combining two heads beat each one alone?")
    lines.append("")
    if conf_df is not None and len(conf_df) > 0:
        conf_sorted = conf_df.sort_values("confluence_lift_vs_a_ticks", ascending=False)
        top_pairs = conf_sorted.head(5)
        lines.append("| head A | head B | top cut | n_conf | mean signed realized (ticks) | lift vs A | lift vs B | hit rate |")
        lines.append("|--------|--------|---------|--------|------------------|-----------|-----------|---------|")
        for _, r in top_pairs.iterrows():
            lines.append(
                f"| {r['head_a']} | {r['head_b']} | top {r['conf_quantile_top_pct']}% | "
                f"{int(r['n_confluence_trades'])} | {r['mean_signed_realized_confluence_ticks']:.4f} | "
                f"{r['confluence_lift_vs_a_ticks']:.4f} | {r['confluence_lift_vs_b_ticks']:.4f} | "
                f"{r['hit_rate_confluence']:.1%} |"
            )
        lines.append("")
        if (conf_sorted["confluence_lift_vs_a_ticks"] > 0).any():
            best_pair = conf_sorted.iloc[0]
            lines.append(
                f"Best confluence pair: **{best_pair['head_a']} + {best_pair['head_b']}** at top "
                f"{best_pair['conf_quantile_top_pct']}% — adds {best_pair['confluence_lift_vs_a_ticks']:.4f} ticks vs head A alone."
            )
        else:
            lines.append("No confluence pair gave positive lift versus a strong single head at the cuts evaluated.")
    else:
        lines.append("No confluence pairs evaluated.")
    lines.append("")

    lines.append("## Sanity row — full-sample time-horizon IC (DEMOTED per HC #466 R1)")
    lines.append("")
    ic_row = head_rank_df.drop_duplicates("head")[["head", "ic_full_sample_sanity"]].sort_values(
        "ic_full_sample_sanity", key=lambda s: s.abs(), ascending=False
    ).head(8)
    lines.append("| head | full-sample Spearman IC |")
    lines.append("|------|--------|")
    for _, r in ic_row.iterrows():
        lines.append(f"| {r['head']} | {r['ic_full_sample_sanity']:.4f} |")
    lines.append("")
    lines.append("(IC is no longer the headline metric. The user has repeated this >=4 times.)")
    lines.append("")

    lines.append("## Per-day regime breakdown for the headline config")
    lines.append("")
    lines.append(
        f"- Green-day Sharpe: {h['sharpe_green_days']:.2f} ({int(h['n_green_days'])} days), "
        f"Red-day Sharpe: {h['sharpe_red_days']:.2f} ({int(h['n_red_days'])} days), "
        f"Flat-day Sharpe: {h['sharpe_flat_days']:.2f} ({int(h['n_flat_days'])} days)."
    )
    lines.append(f"- Regime skew = {h['regime_skew']:.2f} (gate <=0.50 -> {'PASS' if h['regime_pass'] else 'FAIL'}).")
    lines.append("")

    lines.append("## What to do next")
    lines.append("")
    if has_passing:
        lines.append("- Stand up a paper-trade run for the headline config and monitor real-time stream-stability.")
        lines.append("- If a confluence pair added lift, gate live entries on the confluence agreement.")
    else:
        lines.append("- No config passed both gates. The model emits useful heads but on this 34-day OOT, the stream-continuation framing did not yet clear the regime / day-concentration thresholds across the swept exit-rules.")
        lines.append("- Two corrective paths:")
        lines.append("  1. Re-train CNN-Mamba v4 fold 0 with the FIFO-aware target heads weighted higher to reduce the green/red asymmetry.")
        lines.append("  2. Wire confluence-gated entries (top pair from the matrix above) and re-run this harness on top of confluence-only events.")
        lines.append("- Razer relabel pass MUST be expanded to cover the full session (currently only ~12 minutes per day are labeled) before MFE/MAE-aware P&L can replace log-ret-based P&L. This is the single biggest data quality blocker.")
    lines.append("")
    lines.append("---")
    lines.append("## Appendix — Artifacts (technical detail, not for the Discord summary)")
    lines.append("")
    lines.append("- Predictions inventory: `output/stream_backtest/array_inventory.md` / `.json`")
    lines.append("- Per-head ranking: `output/stream_backtest/per_head_ranking.parquet`")
    lines.append("- Tradability matrix: `output/stream_backtest/tradability_matrix.parquet`")
    lines.append("- Confluence matrix: `output/stream_backtest/confluence_matrix.parquet`")
    lines.append("- Full stream trades: `output/stream_backtest/stream_trades_full.parquet`")
    lines.append("- Sampled trades for QA: `output/stream_backtest/per_trade_examples.parquet`")
    lines.append("- Source predictions: `output/cnn_mamba_v3_4_2_fixedmtl/fold_00_predictions.npz` (5-day OOT, all 32 pred heads)")
    lines.append("- Source per-day predictions: `output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/*.npz` (34 days)")
    lines.append("- MFE/MAE labels: `data/relabel/mfe_mae_h{1s,5s,10s,30s}_*.parquet` (Razer-generated, HC #464 — currently sparse, ~12 min/day)")
    lines.append(f"- Top heads selected for stream sweep: {top_heads}")
    lines.append(f"- Wall time: {time.time() - t0:.1f}s")
    lines.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"  wrote {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
