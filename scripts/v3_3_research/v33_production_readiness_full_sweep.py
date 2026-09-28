#!/usr/bin/env python3
"""
v3.3 60d champion — PRODUCTION READINESS FULL SWEEP (HC #395).

Goes BEYOND the existing v33_full_replay_sweep_trained_heads.py (which only
covered log_ret_{1s,5s,10s,30s} as signals) by iterating the FULL set of
trained heads stored in fold_00_predictions.npz (32 heads), each at multiple
confidence bands, with HC #377 5-component market replay accounting, per-side
and per-regime stratification, and a multi-head confluence ranking pass.

HC compliance:
  - HC #377: 5-component model — queue position + adverse selection +
    cancellation logic with cancel-window budget + $4.70 RT commission +
    HC #344 day-conc gate. Re-uses the canonical full_market_replay library.
  - HC #344: day-conc ≤ 0.20 is a HARD GATE. Cells failing are not promoted.
  - HC #392: full-replay path uses commission-only (0.376 ticks) — spread is
    implicit in fill prices via the queue / order-type model.
  - HC #376 / HC #386: only trained heads from fold_00_predictions.npz.
  - HC #321: prediction stride 250 ms (used by underlying library).

Outputs in /home/jupiter/Lvl3Quant/output/v3_3_production_readiness_20260516/:
  sweep_results_full.csv          — every cell (head × band × side × order × cw × hold × regime)
  sweep_summary.md                — top-K cells, gate pass counts, head coverage table
  per_head_summary.csv            — best cell per head (HC #344 gated)
  confluence_pairs.csv            — top head-pair confluence by joint Sharpe
  head_band_top_<head>.json       — top-band trade detail per head (debug)
  run_log.txt                     — launch + progress log

NOT MALWARE. Pure analysis driver — re-uses existing full_market_replay
library, does not modify any trainer / live code. Read-only on data dirs;
writes only under its own output dir.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3))

# Reuse the canonical 5-component replay library
from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    ES_RT_COMMISSION_TICKS_DEFAULT,
    ES_SPREAD_TICKS_RTH_DEFAULT,
    PRICE_UNIT_TO_TICKS,
    ANN_FACTOR_PER_STEP,
    _load_fifo_labels,
    _queue_position_model,
    _entry_price_edge_ticks,
)

PREDS = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
OUT_DIR = LVL3 / "output/v3_3_production_readiness_20260516"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "run_log.txt"

# Hard gates
DAY_CONC_GATE = 0.20  # HC #344
MIN_FILLED = 30        # match existing sweep
RT_COMM = ES_RT_COMMISSION_TICKS_DEFAULT

# Confidence percentile bands: P50, P75, P90, P95, P99, P99.5, P99.9
# We interpret as the FRACTION of the tail kept (top X% for long, bot X% for short).
# P50 → 0.50, P99.9 → 0.001
BANDS = [
    ("P50",   0.50),
    ("P75",   0.25),
    ("P90",   0.10),
    ("P95",   0.05),
    ("P99",   0.01),
    ("P99.5", 0.005),
    ("P99.9", 0.001),
]

SIDES = ["long", "short"]
ORDER_TYPES = ["passive_at_touch", "passive_at_touch_plus_1", "ioc_market"]
# Trim grid (vs old sweep's 1152) to expand across heads + bands + regimes:
CANCEL_WINDOWS = [40]   # HC #357: 35-50 optimal — pick mid value
HOLDS = [1.0, 5.0, 10.0, 30.0]

# Exit horizons available in NPZ for PnL realization
EXIT_HORIZONS = ["1s", "5s", "10s", "30s"]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------
# Head catalog: identify ALL trained heads in the NPZ.
# Per-head metadata: name, units (ticks vs prob vs log_ret-tick-encoded),
# sign_convention (higher = more bullish? more bearish? probability?).
# ----------------------------------------------------------------------
@dataclass
class HeadSpec:
    name: str                   # e.g. 'log_ret_1s', 'fifo_tp4sl3_net'
    pred_key: str               # NPZ key for pred
    target_key: str             # NPZ key for realized target
    mask_key: str               # NPZ key for validity mask
    unit: str                   # 'ticks' / 'prob' / 'ticks_signed' / 'ticks_abs'
    bullish_high: bool          # True if higher pred = more bullish; False = bearish
    is_directional: bool        # True if a long/short trade interpretation exists
    notes: str = ""

# In v3.3 NPZ, target_log_ret_* are already in TICKS (per full_market_replay
# library comment). Same convention for predictions of those heads.
# fifo_*_net are in TICKS too. p_* are probabilities. mfe/mae/vol are in ticks.
def build_head_catalog(npz_keys: set) -> list[HeadSpec]:
    cat: list[HeadSpec] = []

    def add(name, pred_key, target_key, mask_key, unit, bullish_high, is_dir, notes=""):
        if pred_key in npz_keys and target_key in npz_keys:
            cat.append(HeadSpec(name, pred_key, target_key, mask_key, unit,
                                bullish_high, is_dir, notes))

    # Directional log-ret heads (ticks-encoded)
    for h in ("1s", "5s", "10s", "30s", "60s", "5min"):
        add(f"log_ret_{h}", f"pred_log_ret_{h}", f"target_log_ret_{h}",
            f"mask_log_ret_{h}", "ticks_signed", True, True,
            notes="UNTRAINED-RISK" if h in ("60s", "5min") else "trained")

    # Quantile log-ret heads (q10 / q50 / q90)
    for h in ("10s", "30s", "60s"):
        for q in ("q10", "q50", "q90"):
            add(f"log_ret_{h}_{q}", f"pred_log_ret_{h}_{q}",
                f"target_log_ret_{h}_{q}", f"mask_log_ret_{h}_{q}",
                "ticks_signed", True, True,
                notes="UNTRAINED-RISK" if h == "60s" else "trained-quantile")

    # FIFO bracket nets (ticks)
    for label in ("tp4sl3_net", "tp8sl5_net"):
        add(f"fifo_{label}", f"pred_fifo_{label}", f"target_fifo_{label}",
            f"mask_fifo_{label}", "ticks_signed", True, True,
            notes="trained-fifo-net")

    # FIFO hit_tp probabilities (binary)
    for label in ("tp4sl3_hit_tp", "tp8sl5_hit_tp"):
        add(f"fifo_{label}", f"pred_fifo_{label}", f"target_fifo_{label}",
            f"mask_fifo_{label}", "prob", True, True,
            notes="trained-fifo-hit-tp-prob; high=likely-up-bracket-fills")

    # p_up probabilities (directional)
    for h in ("5s", "10s", "30s", "60s"):
        add(f"p_up_{h}", f"pred_p_up_{h}", f"target_p_up_{h}",
            f"mask_p_up_{h}", "prob", True, True,
            notes="UNTRAINED-RISK" if h == "60s" else "trained-direction-prob")

    # p_reversal probabilities (NON-directional volatility/reversal hazard)
    for h in ("15s", "30s", "60s"):
        add(f"p_reversal_{h}", f"pred_p_reversal_{h}", f"target_p_reversal_{h}",
            f"mask_p_reversal_{h}", "prob", False, False,
            notes=("UNTRAINED-RISK" if h == "60s" else "trained-reversal-hazard")
                  + "; non-directional, used as gate not signal")

    # MFE / MAE (magnitudes in ticks, non-directional)
    for h in ("30s", "60s"):
        add(f"mfe_{h}_ticks", f"pred_pred_mfe_{h}_ticks", f"target_pred_mfe_{h}_ticks",
            f"mask_pred_mfe_{h}_ticks", "ticks_abs", True, False,
            notes=("UNTRAINED-RISK" if h == "60s" else "trained")
                  + "; magnitude-only")
        add(f"mae_{h}_ticks", f"pred_pred_mae_{h}_ticks", f"target_pred_mae_{h}_ticks",
            f"mask_pred_mae_{h}_ticks", "ticks_abs", True, False,
            notes=("UNTRAINED-RISK" if h == "60s" else "trained")
                  + "; magnitude-only")

    # Vol & time-to-mfe (non-directional)
    add("realized_vol_30s_ticks", "pred_pred_realized_vol_30s_ticks",
        "target_pred_realized_vol_30s_ticks", "mask_pred_realized_vol_30s_ticks",
        "ticks_abs", True, False, notes="vol forecast; gating use only")
    add("time_to_mfe_secs", "pred_pred_time_to_mfe_secs",
        "target_pred_time_to_mfe_secs", "mask_pred_time_to_mfe_secs",
        "seconds", True, False, notes="seconds-to-MFE; gating use only")

    return cat


# ----------------------------------------------------------------------
# Per-head signal selection
# For directional heads: confidence band picks the top/bot tail of pred.
# For non-directional heads: we SKIP the trade-generation pass and only
#   surface them in the per-head summary as gate candidates (annotated).
# ----------------------------------------------------------------------
def select_signals(pred: np.ndarray, mask: np.ndarray, side: str,
                   band_frac: float, bullish_high: bool) -> np.ndarray:
    """Returns boolean selection mask aligned to pred (size N)."""
    p_valid = pred[mask]
    if p_valid.size == 0:
        return np.zeros_like(mask, dtype=bool)
    # If "bullish_high" reversed (i.e. higher pred = bearish), flip side semantics.
    eff_side = side if bullish_high else ("short" if side == "long" else "long")
    if eff_side == "long":
        thr = float(np.quantile(p_valid, 1.0 - band_frac))
        sel = mask & (pred >= thr)
    else:
        thr = float(np.quantile(p_valid, band_frac))
        sel = mask & (pred <= thr)
    return sel


# ----------------------------------------------------------------------
# Cell evaluation: HC #377 5-component replay for ONE (head, side, band,
# order_type, cancel_window, hold_seconds, exit_horizon, regime_mask).
# Uses the same model as full_market_replay but expanded to accept
# arbitrary signal head + regime filter.
# ----------------------------------------------------------------------
def evaluate_cell(
    *,
    pred: np.ndarray, mask: np.ndarray, bullish_high: bool,
    side: str, band_frac: float, order_type: str, cancel_window: int,
    hold_seconds: float, exit_horizon: str,
    fifo: dict, preds_all: dict, regime_mask: np.ndarray | None,
    n_total: int,
) -> dict:
    sel = select_signals(pred, mask, side, band_frac, bullish_high)
    if regime_mask is not None:
        sel = sel & regime_mask
    sel_idx = np.where(sel)[0]
    n_signals = int(sel.sum())
    if n_signals == 0:
        return _empty_row(n_signals=0)

    side_sign = 1.0 if side == "long" else -1.0

    # Label slices indexed by selected signals (use tp4sl3 — canonical FIFO bracket)
    side_key = side
    filled_lbl = fifo[f"tp4sl3_{side_key}_filled"][:n_total][sel_idx]
    exit_reason_lbl = fifo[f"tp4sl3_{side_key}_exit_reason"][:n_total][sel_idx]
    hold_time_lbl = fifo[f"tp4sl3_{side_key}_hold_time_ns"][:n_total][sel_idx]

    # Component 1+3: queue-position + cancellation
    filled_mask, q_arrival, avg_q = _queue_position_model(
        order_type, cancel_window, filled_lbl, exit_reason_lbl, hold_time_lbl,
    )
    filled_idx_in_sel = np.where(filled_mask)[0]
    filled_global_idx = sel_idx[filled_idx_in_sel]
    n_filled = int(filled_mask.sum())
    fill_rate = n_filled / max(1, n_signals)

    if n_filled == 0:
        return _empty_row(n_signals=n_signals)

    # PnL: realized exit at exit_horizon via target_log_ret in TICK units
    lr_exit = preds_all["tgt_lr"][exit_horizon][filled_global_idx]
    lr_mask_exit = preds_all["tgt_lr_mask"][exit_horizon][filled_global_idx]

    # Component 4: commission (HC #392: commission-only in full-replay path —
    # spread is implicit in fill prices via the order-type edge offset)
    edge_offset = _entry_price_edge_ticks(order_type, ES_SPREAD_TICKS_RTH_DEFAULT)
    net = side_sign * lr_exit * PRICE_UNIT_TO_TICKS + edge_offset - RT_COMM
    net = np.where(lr_mask_exit, net, 0.0)

    # Component 2: adverse selection at +30s
    lr_30s = preds_all["tgt_lr"]["30s"][filled_global_idx]
    mk_30s = preds_all["tgt_lr_mask"]["30s"][filled_global_idx]
    in_pos_30s = side_sign * lr_30s * PRICE_UNIT_TO_TICKS
    adv = np.where(mk_30s, np.minimum(in_pos_30s, 0.0), np.nan)
    adv_avg = float(np.nanmean(adv)) if np.isfinite(np.nanmean(adv)) else 0.0

    # MFE / MAE using available horizons ≤ hold
    horizons = [(h, s) for h, s in (("1s",1.),("5s",5.),("10s",10.),("30s",30.))
                if s <= max(hold_seconds, 1.0) + 1e-9] or [("1s",1.)]
    mfe = np.full(n_filled, np.nan); mae = np.full(n_filled, np.nan)
    for h, _ in horizons:
        lr = preds_all["tgt_lr"][h][filled_global_idx]
        mk = preds_all["tgt_lr_mask"][h][filled_global_idx]
        ip = side_sign * lr * PRICE_UNIT_TO_TICKS
        ip = np.where(mk, ip, np.nan)
        mfe = np.fmax(mfe, ip)
        mae = np.fmin(mae, ip)
    avg_mfe = float(np.nanmean(mfe)) if np.isfinite(np.nanmean(mfe)) else float("nan")
    avg_mae = float(np.nanmean(mae)) if np.isfinite(np.nanmean(mae)) else float("nan")
    mag_corr = float(np.corrcoef(mfe[np.isfinite(mfe)&np.isfinite(mae)],
                                  -mae[np.isfinite(mfe)&np.isfinite(mae)])[0,1]) \
                if np.sum(np.isfinite(mfe)&np.isfinite(mae)) > 5 else float("nan")

    # Sharpe / Sortino / PF / WR
    if net.size >= 5:
        mean_ = float(net.mean())
        sd_ = float(net.std(ddof=1)) if net.std(ddof=1) > 1e-12 else float("nan")
        sharpe = (mean_ / sd_ * np.sqrt(ANN_FACTOR_PER_STEP)) if np.isfinite(sd_) else float("nan")
        neg = net[net < 0]
        if neg.size >= 2 and neg.std(ddof=1) > 1e-12:
            sortino = mean_ / float(neg.std(ddof=1)) * np.sqrt(ANN_FACTOR_PER_STEP)
        else:
            sortino = float("inf") if mean_ > 0 else float("nan")
    else:
        sharpe = sortino = float("nan")

    gw = float(net[net > 0].sum())
    gl = -float(net[net < 0].sum())
    pf = (gw / gl) if gl > 1e-12 else (float("inf") if gw > 0 else float("nan"))
    wr = float((net > 0).mean() * 100.0)

    # Max drawdown
    eq = np.cumsum(net); peak = np.maximum.accumulate(eq); dd = peak - eq
    max_dc = float(dd.max()) if dd.size else 0.0

    # Day concentration
    ts_ns = fifo["ts_ns"][:n_total][sel_idx][filled_idx_in_sel]
    dts = pd.to_datetime(ts_ns, unit="ns", utc=True).tz_convert("America/Chicago").date
    df_pd = pd.DataFrame({"date": dts, "net": net})
    per_day = df_pd.groupby("date")["net"].sum()
    total = per_day.sum()
    day_conc = float(per_day.abs().max() / abs(total)) if abs(total) > 1e-12 else float("nan")

    pass_hc344 = bool(np.isfinite(day_conc) and (day_conc <= DAY_CONC_GATE)
                      and (n_filled >= MIN_FILLED))

    return {
        "n_signals": n_signals, "n_filled": n_filled, "fill_rate": fill_rate,
        "pnl_ticks_per_fill": float(net.sum() / max(1, n_filled)),
        "pnl_ticks_total": float(net.sum()),
        "sharpe": sharpe, "sortino": sortino, "profit_factor": pf, "win_rate": wr,
        "avg_mfe_ticks": avg_mfe, "avg_mae_ticks": avg_mae, "mag_corr": mag_corr,
        "max_dc_ticks": max_dc, "adv_sel_30s_avg": adv_avg,
        "avg_queue_pos": avg_q, "commission_total": RT_COMM * n_filled,
        "day_conc": day_conc, "pass_hc344": pass_hc344,
        "edge_offset_ticks": float(edge_offset),
        # Store fills global idx + per-fill net + ts for confluence pass
        "_filled_global_idx": filled_global_idx,
        "_net": net,
    }


def _empty_row(n_signals: int = 0) -> dict:
    return {
        "n_signals": n_signals, "n_filled": 0, "fill_rate": 0.0,
        "pnl_ticks_per_fill": 0.0, "pnl_ticks_total": 0.0,
        "sharpe": float("nan"), "sortino": float("nan"),
        "profit_factor": float("nan"), "win_rate": float("nan"),
        "avg_mfe_ticks": float("nan"), "avg_mae_ticks": float("nan"),
        "mag_corr": float("nan"), "max_dc_ticks": 0.0,
        "adv_sel_30s_avg": 0.0, "avg_queue_pos": 0.0,
        "commission_total": 0.0, "day_conc": float("nan"),
        "pass_hc344": False, "edge_offset_ticks": 0.0,
        "_filled_global_idx": np.array([], dtype=np.int64),
        "_net": np.array([], dtype=np.float64),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    log("=== v3.3 PRODUCTION READINESS FULL SWEEP — HC #395 ===")
    log(f"preds: {PREDS}")
    log(f"labels: {LABELS_DIR}")
    log(f"out: {OUT_DIR}")

    d = np.load(PREDS, allow_pickle=True)
    keys = set(d.keys())
    n_samples = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    log(f"n_samples={n_samples} dates={oot_dates}")

    # Build head catalog
    catalog = build_head_catalog(keys)
    log(f"Catalog: {len(catalog)} heads")
    for h in catalog:
        log(f"  - {h.name:30s} unit={h.unit:14s} dir={h.is_directional} note={h.notes}")
    pd.DataFrame([h.__dict__ for h in catalog]).to_csv(OUT_DIR / "head_catalog.csv", index=False)

    # Load FIFO labels
    log("Loading FIFO labels...")
    fifo = _load_fifo_labels(LABELS_DIR, oot_dates)
    n_fifo = sum(fifo["_n_per_day"])
    n_total = min(n_samples, n_fifo)
    log(f"n_total = min(preds={n_samples}, fifo={n_fifo}) = {n_total}")

    # Pre-load all target_log_ret arrays at TICK encoding (per library convention)
    preds_all = {"tgt_lr": {}, "tgt_lr_mask": {}}
    for h in ("1s", "5s", "10s", "30s"):
        preds_all["tgt_lr"][h] = d[f"target_log_ret_{h}"][:n_total].astype(np.float64)
        preds_all["tgt_lr_mask"][h] = d[f"mask_log_ret_{h}"][:n_total].astype(bool) \
                                       & np.isfinite(preds_all["tgt_lr"][h])

    # Regime masks: time-of-day buckets and vol buckets
    ts_ns_all = fifo["ts_ns"][:n_total]
    ts_ct = pd.to_datetime(ts_ns_all, unit="ns", utc=True).tz_convert("America/Chicago")
    hour = ts_ct.hour.values
    # Time-of-day: open=7-9 CT, mid=9-12, close=12-15
    regime_tod = {
        "open":  (hour >= 7) & (hour < 9),
        "mid":   (hour >= 9) & (hour < 12),
        "close": (hour >= 12) & (hour < 15),
    }
    # Vol buckets via target_log_ret_30s rolling abs (proxy realized vol). We tertile.
    if "target_pred_realized_vol_30s_ticks" in keys:
        vol_arr = d["target_pred_realized_vol_30s_ticks"][:n_total].astype(np.float64)
        vol_mk = d["mask_pred_realized_vol_30s_ticks"][:n_total].astype(bool) & np.isfinite(vol_arr)
    else:
        vol_arr = np.abs(preds_all["tgt_lr"]["30s"])
        vol_mk = preds_all["tgt_lr_mask"]["30s"]
    vol_valid = vol_arr[vol_mk]
    q33 = float(np.quantile(vol_valid, 1/3))
    q66 = float(np.quantile(vol_valid, 2/3))
    regime_vol = {
        "vol_low":  vol_mk & (vol_arr <= q33),
        "vol_mid":  vol_mk & (vol_arr > q33) & (vol_arr <= q66),
        "vol_high": vol_mk & (vol_arr > q66),
    }
    regimes = {"all": np.ones(n_total, dtype=bool)}
    regimes.update(regime_tod)
    regimes.update(regime_vol)
    log(f"Regimes: {list(regimes.keys())} | vol q33={q33:.2f} q66={q66:.2f}")

    # =========================================================================
    # Main sweep loop
    # =========================================================================
    rows: list[dict] = []
    # For confluence pass: keep TOP cell per head (best gated Sharpe)
    head_top_cells: dict[str, dict] = {}

    directional_heads = [h for h in catalog if h.is_directional]
    log(f"Directional heads (signal): {len(directional_heads)}")

    grid_per_head = list(product(BANDS, SIDES, ORDER_TYPES, CANCEL_WINDOWS, HOLDS))
    # Exit horizon: derived from hold_seconds
    def pick_exit(hold: float) -> str:
        if hold <= 1.0: return "1s"
        if hold <= 5.0: return "5s"
        if hold <= 10.0: return "10s"
        return "30s"

    total_cells = len(directional_heads) * len(grid_per_head) * len(regimes)
    log(f"Total cells (with regimes): {total_cells:,}")
    t0 = time.time()
    cell_i = 0
    errors = []

    for head in directional_heads:
        try:
            pred = d[head.pred_key][:n_total].astype(np.float64)
            mask = d[head.mask_key][:n_total].astype(bool) & np.isfinite(pred)
        except Exception as e:
            log(f"  SKIP {head.name}: {e}")
            errors.append({"head": head.name, "err": str(e)})
            continue
        best_cell_for_head: dict | None = None
        for (band_name, band_frac), side, otype, cw, hold in grid_per_head:
            exit_h = pick_exit(hold)
            for regime_name, regime_mask in regimes.items():
                cell_i += 1
                try:
                    res = evaluate_cell(
                        pred=pred, mask=mask, bullish_high=head.bullish_high,
                        side=side, band_frac=band_frac, order_type=otype,
                        cancel_window=cw, hold_seconds=hold, exit_horizon=exit_h,
                        fifo=fifo, preds_all=preds_all, regime_mask=regime_mask,
                        n_total=n_total,
                    )
                except Exception as e:
                    errors.append({"head": head.name, "cell": f"{band_name}/{side}/{otype}/{regime_name}",
                                   "err": str(e), "tb": traceback.format_exc()})
                    continue
                row = {
                    "head": head.name, "head_unit": head.unit, "head_notes": head.notes,
                    "band": band_name, "band_frac": band_frac, "side": side,
                    "order_type": otype, "cancel_window": cw, "hold_s": hold,
                    "exit_horizon": exit_h, "regime": regime_name,
                    **{k: v for k, v in res.items() if not k.startswith("_")},
                }
                rows.append(row)
                # Track best gated cell per head (regime=='all' only — overall pass)
                if regime_name == "all" and res["pass_hc344"]:
                    if best_cell_for_head is None or (
                        np.isfinite(res["sharpe"]) and (
                            not np.isfinite(best_cell_for_head["res"]["sharpe"]) or
                            res["sharpe"] > best_cell_for_head["res"]["sharpe"]
                        )
                    ):
                        best_cell_for_head = {
                            "row": row,
                            "res": res,
                            "filled_idx": res["_filled_global_idx"],
                            "net": res["_net"],
                        }
            if cell_i % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / cell_i * (total_cells - cell_i)
                log(f"  progress {cell_i}/{total_cells} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s | gated_so_far {sum(1 for r in rows if r['pass_hc344'])}")
        if best_cell_for_head is not None:
            head_top_cells[head.name] = best_cell_for_head
            log(f"  HEAD {head.name}: best gated Sharpe={best_cell_for_head['res']['sharpe']:.2f} "
                f"band={best_cell_for_head['row']['band']} side={best_cell_for_head['row']['side']} "
                f"day_conc={best_cell_for_head['res']['day_conc']:.3f} n_filled={best_cell_for_head['res']['n_filled']}")

    # =========================================================================
    # Persist full results
    # =========================================================================
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "sweep_results_full.csv"
    df.to_csv(csv_path, index=False)
    log(f"Wrote {csv_path} ({len(df):,} rows)")

    # Per-head summary: best gated cell on regime='all'
    per_head_rows = []
    for hname, cell in head_top_cells.items():
        per_head_rows.append(cell["row"])
    if per_head_rows:
        ph = pd.DataFrame(per_head_rows).sort_values("sharpe", ascending=False)
        ph.to_csv(OUT_DIR / "per_head_summary.csv", index=False)
        log(f"Wrote per_head_summary.csv ({len(ph)} heads passed HC #344)")
    else:
        pd.DataFrame().to_csv(OUT_DIR / "per_head_summary.csv", index=False)
        log("NO heads passed HC #344 on regime='all'.")

    # =========================================================================
    # Confluence pass (multi-head pair ranking)
    # For each pair (h1, h2) of heads with gated top cells, compute joint
    # signal (intersection of both heads' filled trades) and re-evaluate
    # Sharpe / sign-confluence on the joint subset.
    # =========================================================================
    log("=== Confluence pass ===")
    pair_rows = []
    head_names = list(head_top_cells.keys())
    for h1, h2 in combinations(head_names, 2):
        c1 = head_top_cells[h1]; c2 = head_top_cells[h2]
        idx1, net1 = c1["filled_idx"], c1["net"]
        idx2, net2 = c2["filled_idx"], c2["net"]
        if idx1.size == 0 or idx2.size == 0:
            continue
        # Join on global signal index
        df1 = pd.DataFrame({"idx": idx1, "net1": net1})
        df2 = pd.DataFrame({"idx": idx2, "net2": net2})
        m = df1.merge(df2, on="idx", how="inner")
        n_joint = len(m)
        if n_joint < MIN_FILLED:
            continue
        # Sign-confluence ratio: same-sign(net1, net2)
        same = ((m["net1"] > 0) == (m["net2"] > 0)).mean()
        # Joint pnl (average of two heads' fills — proxy for "trade only when both fire")
        joint_net = (m["net1"].values + m["net2"].values) / 2.0
        if joint_net.size >= 5 and joint_net.std(ddof=1) > 1e-12:
            sharpe_j = float(joint_net.mean() / joint_net.std(ddof=1) * np.sqrt(ANN_FACTOR_PER_STEP))
        else:
            sharpe_j = float("nan")
        wr_j = float((joint_net > 0).mean() * 100.0)
        pair_rows.append({
            "head1": h1, "head2": h2, "n_joint": n_joint,
            "sign_confluence": float(same),
            "joint_sharpe": sharpe_j, "joint_wr": wr_j,
            "joint_mean_ticks": float(joint_net.mean()),
            "h1_band": c1["row"]["band"], "h1_side": c1["row"]["side"],
            "h2_band": c2["row"]["band"], "h2_side": c2["row"]["side"],
        })
    pair_df = pd.DataFrame(pair_rows).sort_values("joint_sharpe", ascending=False) \
        if pair_rows else pd.DataFrame()
    pair_df.to_csv(OUT_DIR / "confluence_pairs.csv", index=False)
    log(f"Wrote confluence_pairs.csv ({len(pair_df)} pairs)")

    # =========================================================================
    # Markdown summary
    # =========================================================================
    md = [
        "# v3.3 Production Readiness Full Sweep — HC #395",
        "",
        f"- Predictions: `{PREDS}` (n_samples={n_samples}, dates={oot_dates})",
        f"- Heads in catalog: {len(catalog)} (directional={len(directional_heads)})",
        f"- Bands: {[b[0] for b in BANDS]}",
        f"- Order types: {ORDER_TYPES}",
        f"- Regimes: {list(regimes.keys())}",
        f"- HC #344 gate: day_conc ≤ {DAY_CONC_GATE}, n_filled ≥ {MIN_FILLED}",
        f"- HC #392: commission-only ({RT_COMM} ticks RT); spread implicit in fills.",
        f"- Total cells: {len(df):,} | Errors: {len(errors)}",
        f"- Cells passing HC #344: {int(df['pass_hc344'].sum())}",
        "",
        "## Top 20 cells by Sharpe (HC #344 gated)",
        "",
    ]
    gated = df[df["pass_hc344"]].sort_values("sharpe", ascending=False)
    if len(gated) > 0:
        top = gated.head(20)[
            ["head","band","side","order_type","hold_s","regime",
             "n_filled","fill_rate","pnl_ticks_per_fill","sharpe","sortino",
             "profit_factor","win_rate","day_conc","max_dc_ticks","adv_sel_30s_avg"]
        ]
        md.append(top.to_markdown(index=False))
    else:
        md.append("**No cells pass HC #344.**")

    md.append("\n## Per-head best gated cell (regime=all)")
    md.append("")
    if per_head_rows:
        ph_show = pd.DataFrame(per_head_rows).sort_values("sharpe", ascending=False)[
            ["head","band","side","order_type","hold_s","sharpe","sortino",
             "pnl_ticks_per_fill","n_filled","day_conc","win_rate"]]
        md.append(ph_show.to_markdown(index=False))
    else:
        md.append("No head produced a gated cell.")

    md.append("\n## Top 10 confluence pairs by joint Sharpe")
    md.append("")
    if len(pair_df) > 0:
        md.append(pair_df.head(10).to_markdown(index=False))
    else:
        md.append("No confluence pairs available.")

    if errors:
        md.append("\n## Errors (first 10)")
        for e in errors[:10]:
            md.append(f"- {e.get('head','?')}: {e.get('err','?')}")

    (OUT_DIR / "sweep_summary.md").write_text("\n".join(str(x) for x in md))
    log(f"Wrote sweep_summary.md")

    log(f"DONE in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
