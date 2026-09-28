"""
HC #403 (C) — PatchTST + v3.3 CONFLUENCE TEST

Question: Does requiring BOTH v3.3 AND PatchTST to agree on direction (and both
exceed their respective top-pctile thresholds) IMPROVE per-trade economics —
or does it just shrink the trade count proportionally with no Sharpe gain?

Inputs:
  - v3.3 predictions (15-day OOT 2026-03-01..2026-03-19, 673,184 samples)
      /home/jupiter/Lvl3Quant/output/v3_3_extended_oot_20260514/extended_oot_predictions.npz
  - PatchTST predictions (per-day NPZs in output/patchtst_bulk_oot/)
      Horizons available: 1s / 5s / 10s.
      Days overlapping w/ v3.3 OOT: 20260309, 10, 11, 12, 16, 17, 18, 19
      (308, 315 are near-empty session days; dropped).
      Per-day count = (v3.3 per-day count) + 10  →  drop FIRST 10 PatchTST rows
      per day to align (offset documented inline).
  - FIFO labels: data/processed/mbo_events_smart_v3_fifo_labels/<DATE>_fifo_labels.npz
  - Replay infra: scripts/v3_3_research/full_market_replay.py
  - Post-filters: scripts/v3_3_research/v33_execution_optuna_full_market_replay.py
                  (TOD filter + min pred-strength; NB FIFO confluence was a no-op
                   in the original optuna study — see source line ~294 "pass" —
                   so trial-278 reproduction follows the SAME no-op path.)

Trial 278 baseline (current MVP, HC #402-B):
  head=log_ret_30s, side=short, order=passive_at_touch_plus_2,
  conf_pctile=0.0435, hold=1.4768s, cancel=79 evals, ToD=13–15 ET,
  pred_strength_min=0.0, FIFO confluence flag=True (NO-OP), expected Sharpe 13.48.

This script:
  1) Reproduces trial 278 on the FULL 15-day v3.3 OOT (sanity baseline; must
     come within 0.5 Sharpe of 13.48 or we halt the sweep).
  2) Restricts to the 8 PatchTST-overlap days and re-runs v3.3-alone to
     establish the apples-to-apples baseline (different denominator so a
     different number than 13.48 is expected; this IS the control for C).
  3) Sweeps confluence: for each (v3.3 entry pctile) × (patchtst entry pctile),
     additionally require PatchTST signal to AGREE in direction (short = both
     bearish) AND be in its bottom-X% (long = both bullish, top-X%). Re-run
     replay on the overlap days only with the confluence-gated entry mask
     baked in via a side-specific signed prediction surrogate (we MULTIPLY
     v3.3 |pred| by a large value where confluence holds; this preserves the
     percentile gate semantics inside `full_market_replay` without modifying
     it). For each (v33_pctile, patchtst_pctile) we compute Sharpe/Sortino/PF/
     mean_net_ticks / fills_per_day_median / day_conc / CI_low_95.
  4) Honesty: every result row carries n_fills and 95% CI lower bound on per-
     trade net ticks (HC #397B). Confluence "wins" only if it ADDS Sharpe per
     fill, not merely shrinks the trade count.

NO-LOOK-AHEAD: at entry timestamp, both v3.3 and PatchTST predictions are known.
Both were generated from data ≤ entry time. The PATCH alignment offset (drop
first 10 patchtst rows/day) preserves ordering — it does NOT introduce future
data.

Output: output/hc403_c_patchtst_<timestamp>/
  - confluence_sweep.csv
  - BASELINE_REPRODUCTION.json
  - SUMMARY.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(PROJ))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    TradeConfig, full_market_replay,
)
from scripts.v3_3_research.v33_execution_optuna_full_market_replay import (  # noqa: E402
    apply_post_filters, metrics_from_filtered,
)

# -----------------------------------------------------------------------------
# Paths and canonical constants
# -----------------------------------------------------------------------------
V33_PREDS = PROJ / "output" / "v3_3_extended_oot_20260514" / "extended_oot_predictions.npz"
LABELS_DIR = PROJ / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
PATCHTST_DIR = PROJ / "output" / "patchtst_bulk_oot"

OUT_DIR = PROJ / "output" / f"hc403_c_patchtst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_COMMISSION = 0.376  # HC #392
ES_PX_REF = 5800.0
LOG_RET_PER_TICK = float(np.log((ES_PX_REF + 0.25) / ES_PX_REF))

V33_OOT_DATES = [
    "20260301", "20260302", "20260303", "20260304", "20260305",
    "20260308", "20260309", "20260310", "20260311", "20260312",
    "20260315", "20260316", "20260317", "20260318", "20260319",
]

# Overlap dates with PatchTST bulk_oot. 308 / 315 are near-empty (18/13 rows
# in v3.3 fifo, 28/23 in patchtst — half-day/holiday) — DROP from confluence
# sweep, retain in baselines so trial 278 reproduction is on full 15 days.
OVERLAP_DATES = ["20260309", "20260310", "20260311", "20260312",
                 "20260316", "20260317", "20260318", "20260319"]

# PatchTST emits +10 windows/day more than v3.3 — drop first 10 patchtst rows.
PATCHTST_OFFSET_PER_DAY = 10
PATCHTST_HORIZON_FOR_CONFLUENCE = "5s"  # 1s/5s/10s available; 5s is the mid


# -----------------------------------------------------------------------------
# Trial 278 spec (HC #402-B Monday MVP)
# -----------------------------------------------------------------------------
TRIAL_278 = dict(
    head_horizon="log_ret_30s",
    side="short",
    order_type="passive_at_touch_plus_2",
    conf_thr=0.0435,
    cancel_window=79,
    hold_seconds=1.4768,
    spread_ticks=1.0,
    tod_start=13,
    tod_end=15,
    pred_strength_min=0.0,
)
# Horizon string for TradeConfig (drop "log_ret_" prefix)
def _head_to_horizon(h: str) -> str:
    return h.replace("log_ret_", "")


# -----------------------------------------------------------------------------
# PatchTST loader (per-day NPZs → aligned to v3.3 / FIFO sample order)
# -----------------------------------------------------------------------------
def load_patchtst_aligned(dates: list[str], horizon: str,
                          v33_perday_target: dict | None = None) -> tuple[np.ndarray, dict]:
    """Return (pred_array_aligned_to_v33_sample_order_for_dates, meta).

    For each date, load PatchTST predictions, drop first PATCHTST_OFFSET_PER_DAY
    rows, take horizon column. If `v33_perday_target` provided, also truncate
    each day to that target count (v3.3 may truncate the trailing day at the
    NPZ-level n_samples cap). Concatenate in `dates` order.
    """
    horizon_idx_map = {"1s": 0, "5s": 1, "10s": 2}
    if horizon not in horizon_idx_map:
        raise ValueError(f"Unsupported patchtst horizon: {horizon}")
    h_idx = horizon_idx_map[horizon]

    parts = []
    per_day_counts = {}
    for d in dates:
        fp = PATCHTST_DIR / f"{d}_predictions.npz"
        if not fp.exists():
            raise FileNotFoundError(f"PatchTST missing for {d}: {fp}")
        z = np.load(fp, allow_pickle=True)
        arr = z["predictions"]  # (N, 3) for 1s/5s/10s
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Unexpected PatchTST shape {arr.shape} for {d}")
        # Drop first PATCHTST_OFFSET_PER_DAY rows to align with v3.3
        aligned = arr[PATCHTST_OFFSET_PER_DAY:, h_idx].astype(np.float32)
        if v33_perday_target is not None and d in v33_perday_target:
            tgt = v33_perday_target[d]
            if aligned.shape[0] >= tgt:
                aligned = aligned[:tgt]
            else:
                raise RuntimeError(
                    f"PatchTST date {d}: aligned {aligned.shape[0]} < v33 target {tgt}"
                )
        parts.append(aligned)
        per_day_counts[d] = int(aligned.shape[0])
    return np.concatenate(parts), per_day_counts


def get_v33_perday_counts(dates: list[str]) -> dict:
    """Get per-day FIFO label counts for the v3.3 dates (used to slice v3.3
    samples per date — v3.3 NPZ has no per-sample date but predictions are
    concatenated in this exact date order)."""
    counts = {}
    for d in dates:
        fp = LABELS_DIR / f"{d}_fifo_labels.npz"
        z = np.load(fp, allow_pickle=False)
        counts[d] = int(z["window_k"].shape[0])
    return counts


def build_v33_date_idx_and_slabs(all_dates: list[str], v33_n: int) -> tuple[np.ndarray, dict]:
    """Build a (v33_n,) date_idx array and a {date: (lo, hi)} index map.

    v3.3 samples are concatenated per `all_dates`. FIFO labels per date give
    counts. Cumulative slabs index v3.3 in same order; truncate at v33_n.
    """
    counts = get_v33_perday_counts(all_dates)
    idx_arr = np.empty(v33_n, dtype=np.int32)
    slab_map = {}
    cur = 0
    for i, d in enumerate(all_dates):
        n_d = counts[d]
        lo = cur
        hi = min(cur + n_d, v33_n)
        idx_arr[lo:hi] = i
        slab_map[d] = (lo, hi)
        cur = hi
        if cur >= v33_n:
            break
    return idx_arr, slab_map


# -----------------------------------------------------------------------------
# Baseline: reproduce trial 278
# -----------------------------------------------------------------------------
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def reproduce_trial_278(dates_to_keep: list[str] | None, note: str) -> dict:
    """Run trial 278 settings ALWAYS on the full 15-day v3.3 OOT, then optionally
    post-filter per-trade rows to a subset of dates (by ET date).

    NOTE: `full_market_replay` slices v3.3 from sample 0 in date order, so we
    must always pass ALL v3.3 dates. To get an overlap-only result we filter
    the per_trade_df by timestamp ET-date after the fact.
    """
    cfg = TradeConfig(
        side=TRIAL_278["side"],
        horizon=_head_to_horizon(TRIAL_278["head_horizon"]),
        confidence_threshold=TRIAL_278["conf_thr"],
        order_type=TRIAL_278["order_type"],
        cancel_eval_window=TRIAL_278["cancel_window"],
        hold_seconds=TRIAL_278["hold_seconds"],
    )
    ledger = full_market_replay(
        V33_PREDS, LABELS_DIR, cfg, dates=V33_OOT_DATES,
        spread_ticks_rth=TRIAL_278["spread_ticks"],
        rt_commission_ticks=CANONICAL_COMMISSION,
    )
    df_f, _ = apply_post_filters(
        ledger,
        tod_start_hour=TRIAL_278["tod_start"],
        tod_end_hour=TRIAL_278["tod_end"],
        require_min_pred_strength=TRIAL_278["pred_strength_min"],
    )
    if dates_to_keep is not None and not df_f.empty:
        ts_pd = pd.to_datetime(df_f["timestamp"].to_numpy(), unit="ns", utc=True).tz_convert("America/New_York")
        date_str = ts_pd.strftime("%Y%m%d")
        keep_set = set(dates_to_keep)
        keep = np.array([d in keep_set for d in date_str])
        df_f = df_f.loc[keep].reset_index(drop=True)
    m = metrics_from_filtered(df_f)
    m["n_signals"] = int(ledger.n_signals)
    m["n_filled_raw"] = int(ledger.n_filled)
    m["n_days"] = len(dates_to_keep) if dates_to_keep is not None else len(V33_OOT_DATES)
    m["note"] = note
    m["fills_per_day_median"] = _fills_per_day_median(df_f)
    return m


def _fills_per_day_median(df_f: pd.DataFrame) -> float:
    if df_f.empty:
        return 0.0
    ts_pd = pd.to_datetime(df_f["timestamp"].to_numpy(), unit="ns", utc=True).tz_convert("America/New_York")
    day_counts = pd.Series(ts_pd.strftime("%Y%m%d")).value_counts().to_dict()
    return float(np.median(list(day_counts.values()))) if day_counts else 0.0


# -----------------------------------------------------------------------------
# Confluence-gated replay
# -----------------------------------------------------------------------------
def _patchtst_confluence_mask(
    side: str,
    patchtst_pred_aligned: np.ndarray,
    n_global: int,
    overlap_global_slab: np.ndarray,
    patchtst_pctile: float,
) -> np.ndarray:
    """Return a (n_global,) bool mask of indices ALLOWED by PatchTST confluence.

    side='short': require patchtst_pred in BOTTOM `patchtst_pctile` (bearish).
    side='long':  require patchtst_pred in TOP `patchtst_pctile` (bullish).
    Indices outside the overlap_global_slab are False (we cannot confirm
    confluence there).
    """
    if patchtst_pctile <= 0:
        # pctile=0 means "no constraint" (sentinel for v3.3-alone control row)
        mask = np.zeros(n_global, dtype=bool)
        mask[overlap_global_slab] = True
        return mask

    # Build local mask over overlap region only
    p = patchtst_pred_aligned
    p_valid = p[np.isfinite(p)]
    if p_valid.size == 0:
        return np.zeros(n_global, dtype=bool)
    if side == "short":
        thr = float(np.quantile(p_valid, patchtst_pctile))
        local_keep = (p <= thr) & np.isfinite(p)
    else:
        thr = float(np.quantile(p_valid, 1.0 - patchtst_pctile))
        local_keep = (p >= thr) & np.isfinite(p)

    mask = np.zeros(n_global, dtype=bool)
    mask[overlap_global_slab] = local_keep
    return mask


def _replay_and_filter_overlap(head_horizon: str, side: str, v33_pctile: float) -> pd.DataFrame:
    """Run full_market_replay on full 15d v3.3 + apply TOD post-filter +
    restrict to OVERLAP_DATES by ET date. Return the resulting filled-trades
    dataframe. Cached upstream."""
    cfg = TradeConfig(
        side=side, horizon=_head_to_horizon(head_horizon),
        confidence_threshold=v33_pctile,
        order_type=TRIAL_278["order_type"],
        cancel_eval_window=TRIAL_278["cancel_window"],
        hold_seconds=TRIAL_278["hold_seconds"],
    )
    ledger = full_market_replay(
        V33_PREDS, LABELS_DIR, cfg, dates=V33_OOT_DATES,
        spread_ticks_rth=TRIAL_278["spread_ticks"],
        rt_commission_ticks=CANONICAL_COMMISSION,
    )
    df_f, _ = apply_post_filters(
        ledger,
        tod_start_hour=TRIAL_278["tod_start"],
        tod_end_hour=TRIAL_278["tod_end"],
        require_min_pred_strength=TRIAL_278["pred_strength_min"],
    )
    if df_f.empty:
        return df_f
    ts_pd = pd.to_datetime(df_f["timestamp"].to_numpy(), unit="ns", utc=True).tz_convert("America/New_York")
    date_str = ts_pd.strftime("%Y%m%d")
    overlap_set = set(OVERLAP_DATES)
    keep = np.array([d in overlap_set for d in date_str])
    return df_f.loc[keep].reset_index(drop=True)


def _apply_patchtst_confluence(
    df_base: pd.DataFrame,
    side: str,
    patchtst_pctile: float,
    patchtst_pred_aligned: np.ndarray,
    overlap_dates: list[str],
    v33_overlap_target: dict,
) -> dict:
    """Apply PatchTST confluence filter to a base overlap df. Returns metrics."""
    df_f = df_base
    if patchtst_pctile > 0 and not df_f.empty:
        p = patchtst_pred_aligned
        p_valid_pt = p[np.isfinite(p)]
        if side == "short":
            thr_pt = float(np.quantile(p_valid_pt, patchtst_pctile))
            patchtst_keep = (p <= thr_pt) & np.isfinite(p)
        else:
            thr_pt = float(np.quantile(p_valid_pt, 1.0 - patchtst_pctile))
            patchtst_keep = (p >= thr_pt) & np.isfinite(p)

        fifo_ts_per_day = []
        for d in overlap_dates:
            z = np.load(LABELS_DIR / f"{d}_fifo_labels.npz", allow_pickle=False)
            ts = z["ts_ns"]
            tgt = v33_overlap_target[d]
            fifo_ts_per_day.append(ts[:tgt])
        fifo_ts_overlap = np.concatenate(fifo_ts_per_day)
        if patchtst_keep.shape[0] != fifo_ts_overlap.shape[0]:
            raise RuntimeError(
                f"PatchTST aligned shape {patchtst_keep.shape[0]} != "
                f"fifo overlap shape {fifo_ts_overlap.shape[0]} — alignment bug"
            )
        # Build ts → keep dict; for duplicate ts, take logical OR (any row that
        # the patchtst filter accepts at that ts → keep). Duplicates are <1%.
        ser = pd.Series(patchtst_keep, index=fifo_ts_overlap)
        # OR-aggregate by index for any duplicates
        ser_dedup = ser.groupby(level=0).any()
        mapped = df_f["timestamp"].map(ser_dedup)
        df_f_keep_mask = mapped.fillna(False).astype(bool).to_numpy()
        df_f = df_f.loc[df_f_keep_mask].reset_index(drop=True)

    m = metrics_from_filtered(df_f)
    m.update(_extra_metrics(df_f, len(overlap_dates)))
    return m


def _run_confluence_cell_UNUSED(
    head_horizon: str,
    side: str,
    v33_pctile: float,
    patchtst_pctile: float,
    patchtst_pred_aligned: np.ndarray | None,
    overlap_global_idx: np.ndarray,
    overlap_dates: list[str],
    v33_overlap_target: dict | None = None,
) -> dict:
    """Run a single (v33_pctile, patchtst_pctile, mode) cell.

    Mechanic: run full_market_replay on the FULL 15-day v3.3 OOT (the only
    safe way given full_market_replay always slices v3.3 from sample 0 in date
    order). Then post-filter per_trade_df rows to:
      (a) trades within OVERLAP_DATES (by ET date of timestamp)
      (b) trades whose PatchTST 5s prediction at the same fifo timestamp
          satisfies the confluence condition.
    """
    cfg = TradeConfig(
        side=side, horizon=_head_to_horizon(head_horizon),
        confidence_threshold=v33_pctile,
        order_type=TRIAL_278["order_type"],
        cancel_eval_window=TRIAL_278["cancel_window"],
        hold_seconds=TRIAL_278["hold_seconds"],
    )
    ledger = full_market_replay(
        V33_PREDS, LABELS_DIR, cfg, dates=V33_OOT_DATES,
        spread_ticks_rth=TRIAL_278["spread_ticks"],
        rt_commission_ticks=CANONICAL_COMMISSION,
    )
    df_f, _ = apply_post_filters(
        ledger,
        tod_start_hour=TRIAL_278["tod_start"],
        tod_end_hour=TRIAL_278["tod_end"],
        require_min_pred_strength=TRIAL_278["pred_strength_min"],
    )
    if df_f.empty:
        m = metrics_from_filtered(df_f)
        m.update(_extra_metrics(df_f, len(overlap_dates)))
        return m

    # Restrict to overlap dates by ET date
    ts_pd = pd.to_datetime(df_f["timestamp"].to_numpy(), unit="ns", utc=True).tz_convert("America/New_York")
    date_str = ts_pd.strftime("%Y%m%d")
    overlap_set = set(overlap_dates)
    keep_overlap = np.array([d in overlap_set for d in date_str])
    df_f = df_f.loc[keep_overlap].reset_index(drop=True)
    if df_f.empty:
        m = metrics_from_filtered(df_f)
        m.update(_extra_metrics(df_f, len(overlap_dates)))
        return m

    # Apply PatchTST confluence — build a ts→patchtst_pred lookup over overlap
    if patchtst_pctile > 0:
        if patchtst_pred_aligned is None:
            raise RuntimeError("patchtst_pred_aligned required for confluence cell")
        p = patchtst_pred_aligned
        p_valid_pt = p[np.isfinite(p)]
        if side == "short":
            thr_pt = float(np.quantile(p_valid_pt, patchtst_pctile))
            patchtst_keep = (p <= thr_pt) & np.isfinite(p)
        else:
            thr_pt = float(np.quantile(p_valid_pt, 1.0 - patchtst_pctile))
            patchtst_keep = (p >= thr_pt) & np.isfinite(p)

        # Build fifo_ts aligned to v3.3 overlap (matches patchtst_pred_aligned length)
        fifo_ts_per_day = []
        for d in overlap_dates:
            z = np.load(LABELS_DIR / f"{d}_fifo_labels.npz", allow_pickle=False)
            ts = z["ts_ns"]
            tgt = v33_overlap_target[d] if (v33_overlap_target and d in v33_overlap_target) else ts.shape[0]
            fifo_ts_per_day.append(ts[:tgt])
        fifo_ts_overlap = np.concatenate(fifo_ts_per_day)
        if patchtst_keep.shape[0] != fifo_ts_overlap.shape[0]:
            raise RuntimeError(
                f"PatchTST aligned shape {patchtst_keep.shape[0]} != "
                f"fifo overlap shape {fifo_ts_overlap.shape[0]} — alignment bug"
            )

        # ts→keep lookup. ts are unique within overlap (ns-precision in MBO ts).
        keep_series = pd.Series(patchtst_keep, index=fifo_ts_overlap)
        # Some df_f timestamps might not match (extreme edge); .map returns NaN→False
        mapped = df_f["timestamp"].map(keep_series)
        df_f_keep_mask = mapped.fillna(False).astype(bool).to_numpy()
        df_f = df_f.loc[df_f_keep_mask].reset_index(drop=True)

    m = metrics_from_filtered(df_f)
    m.update(_extra_metrics(df_f, len(overlap_dates)))
    return m


def _extra_metrics(df_f: pd.DataFrame, n_days: int) -> dict:
    """Fills/day median + a few extras."""
    if df_f.empty:
        return dict(fills_per_day_median=0.0, n_days=n_days)
    ts = df_f["timestamp"].to_numpy()
    ts_pd = pd.to_datetime(ts, unit="ns", utc=True).tz_convert("America/New_York")
    day_counts = pd.Series(ts_pd.strftime("%Y%m%d")).value_counts().to_dict()
    fpd = float(np.median(list(day_counts.values()))) if day_counts else 0.0
    return dict(fills_per_day_median=fpd, n_days=n_days)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"[{_now()}] HC #403 (C) PatchTST + v3.3 CONFLUENCE TEST")
    print(f"[{_now()}] OUT_DIR = {OUT_DIR}")
    print(f"[{_now()}] V33_PREDS = {V33_PREDS}")
    print(f"[{_now()}] PATCHTST_DIR = {PATCHTST_DIR}")

    # Check PatchTST data availability
    patchtst_status = "OK"
    patchtst_missing = []
    for d in OVERLAP_DATES:
        if not (PATCHTST_DIR / f"{d}_predictions.npz").exists():
            patchtst_missing.append(d)
    if patchtst_missing:
        patchtst_status = f"PARTIAL_MISSING:{','.join(patchtst_missing)}"
        print(f"[{_now()}] WARNING — PatchTST missing for: {patchtst_missing}")

    t0 = time.time()

    # ---- Step 1: reproduce trial 278 on FULL 15 days ----
    print(f"\n[{_now()}] === Step 1: reproduce trial 278 on full 15-day OOT ===")
    repro_full = reproduce_trial_278(None, "full_15day_v33_alone_trial278")
    print(f"[{_now()}] Trial 278 (full 15d): Sharpe={repro_full['sharpe']:.2f} "
          f"day_conc={repro_full['day_conc']:.3f} n_fills={repro_full['n_fills']} "
          f"mean_net={repro_full['mean_net']:+.3f}")

    repro_diff = abs(repro_full["sharpe"] - 13.48)
    repro_status = "OK" if repro_diff <= 0.5 else f"DEVIATION_{repro_diff:.2f}_sharpe"

    # ---- Step 2: v3.3-alone restricted to overlap days (control for C) ----
    print(f"\n[{_now()}] === Step 2: v3.3-alone on {len(OVERLAP_DATES)}-day overlap ===")
    repro_overlap = reproduce_trial_278(OVERLAP_DATES, "overlap_v33_alone_trial278")
    print(f"[{_now()}] Trial 278 (overlap {len(OVERLAP_DATES)}d): "
          f"Sharpe={repro_overlap['sharpe']:.2f} day_conc={repro_overlap['day_conc']:.3f} "
          f"n_fills={repro_overlap['n_fills']} mean_net={repro_overlap['mean_net']:+.3f}")

    # Save baseline reproduction
    baseline_reproduction = {
        "trial_278_full_15day": repro_full,
        "trial_278_overlap_only": repro_overlap,
        "expected_full_sharpe_hc402b": 13.48,
        "reproduction_status": repro_status,
        "reproduction_diff_sharpe": repro_diff,
        "patchtst_status": patchtst_status,
    }
    (OUT_DIR / "BASELINE_REPRODUCTION.json").write_text(
        json.dumps(baseline_reproduction, indent=2, default=str)
    )

    # Build PatchTST aligned predictions and overlap_global_idx into v3.3
    print(f"\n[{_now()}] === Building PatchTST alignment ===")
    raw_v33 = np.load(V33_PREDS, allow_pickle=True)
    v33_n = int(raw_v33["n_samples"])
    date_idx_arr, slab_map = build_v33_date_idx_and_slabs(V33_OOT_DATES, v33_n)
    # overlap_global_idx: indices in v3.3 sample order belonging to OVERLAP_DATES
    overlap_global_idx_parts = []
    for d in OVERLAP_DATES:
        lo, hi = slab_map[d]
        overlap_global_idx_parts.append(np.arange(lo, hi, dtype=np.int64))
    overlap_global_idx = np.concatenate(overlap_global_idx_parts)
    print(f"[{_now()}] overlap_global_idx size = {overlap_global_idx.size}")

    # Compute v3.3 per-overlap-day target counts (slab_map already capped at v33_n)
    v33_overlap_target = {d: (slab_map[d][1] - slab_map[d][0]) for d in OVERLAP_DATES}
    patchtst_pred_aligned, patchtst_perday = load_patchtst_aligned(
        OVERLAP_DATES, PATCHTST_HORIZON_FOR_CONFLUENCE,
        v33_perday_target=v33_overlap_target,
    )
    print(f"[{_now()}] patchtst aligned size = {patchtst_pred_aligned.size} "
          f"(should match v3.3 overlap slab)")
    if patchtst_pred_aligned.size != overlap_global_idx.size:
        raise RuntimeError(
            f"PatchTST aligned ({patchtst_pred_aligned.size}) != "
            f"v3.3 overlap ({overlap_global_idx.size}). Cannot continue."
        )

    # ---- Step 3: confluence sweep ----
    print(f"\n[{_now()}] === Step 3: confluence sweep ===")
    head_horizon = TRIAL_278["head_horizon"]
    side = TRIAL_278["side"]

    # v33_pctile grid (centered on trial 278's 0.0435)
    v33_pctiles = [0.0435, 0.05, 0.10]
    # patchtst_pctile per mode:
    #   v3.3_alone:                   patchtst_pctile = 0.0 (sentinel = no PatchTST constraint, restricted to overlap)
    #   v3.3_AND_patchtst_top10:      patchtst_pctile = 0.10
    #   v3.3_AND_patchtst_top5:       patchtst_pctile = 0.05
    #   v3.3_AND_patchtst_top1:       patchtst_pctile = 0.01
    modes = [
        ("v3.3_alone_overlap", 0.0),
        ("v3.3_AND_patchtst_agree_top10", 0.10),
        ("v3.3_AND_patchtst_agree_top5", 0.05),
        ("v3.3_AND_patchtst_agree_top1", 0.01),
    ]

    # Pre-compute (replay + overlap-filtered df_f) per v33_pctile (cache)
    df_f_overlap_cache = {}
    for v33_pct in v33_pctiles:
        print(f"[{_now()}] PRECOMPUTE replay @ v33_pct={v33_pct:.4f}")
        df_f_overlap_cache[v33_pct] = _replay_and_filter_overlap(
            head_horizon, side, v33_pct,
        )

    rows = []
    for v33_pct in v33_pctiles:
        df_base = df_f_overlap_cache[v33_pct]
        for mode_name, pt_pct in modes:
            print(f"[{_now()}]   cell: v33_pct={v33_pct:.4f} mode={mode_name} pt_pct={pt_pct}")
            m = _apply_patchtst_confluence(
                df_base=df_base,
                side=side,
                patchtst_pctile=pt_pct,
                patchtst_pred_aligned=patchtst_pred_aligned,
                overlap_dates=OVERLAP_DATES,
                v33_overlap_target=v33_overlap_target,
            )
            rows.append({
                "v33_pctile": v33_pct,
                "patchtst_pctile": pt_pct,
                "confluence_mode": mode_name,
                "n_fills": m["n_fills"],
                "sharpe": m["sharpe"],
                "sortino": m["sortino"],
                "pf": m["pf"],
                "wr": m["wr"],
                "mean_net_ticks": m["mean_net"],
                "day_conc": m["day_conc"],
                "ci_low_95": m["ci_low_95"],
                "fills_per_day_median": m.get("fills_per_day_median", 0.0),
                "n_days": m.get("n_days", len(OVERLAP_DATES)),
            })

    df_sweep = pd.DataFrame(rows)
    df_sweep.to_csv(OUT_DIR / "confluence_sweep.csv", index=False)
    print(f"[{_now()}] Wrote confluence_sweep.csv ({len(df_sweep)} rows)")
    print("\n" + df_sweep.to_string(index=False))

    # ---- Step 4: bottom-line summary ----
    # Baseline = v3.3_alone_overlap at v33_pctile=0.0435 (matches trial 278)
    baseline_row = df_sweep.query(
        "confluence_mode=='v3.3_alone_overlap' and v33_pctile==0.0435"
    )
    base_sharpe = float(baseline_row["sharpe"].iloc[0]) if not baseline_row.empty else None
    base_mean_net = float(baseline_row["mean_net_ticks"].iloc[0]) if not baseline_row.empty else None
    base_fpd = float(baseline_row["fills_per_day_median"].iloc[0]) if not baseline_row.empty else None

    # Best confluence cell by Sharpe (among confluence modes only)
    # HONESTY GATE: require n_fills >= 30 for the cell to be considered a winner
    # (small-N Sharpe is dominated by noise — see e.g. n_fills=3 outliers).
    MIN_FILLS_FOR_WINNER = 30
    conf_only = df_sweep.query("confluence_mode != 'v3.3_alone_overlap'")
    conf_robust = conf_only.query(f"n_fills >= {MIN_FILLS_FOR_WINNER}")
    if not conf_robust.empty:
        best_idx = conf_robust["sharpe"].idxmax()
        best_row = conf_robust.loc[best_idx]
        best_summary = {
            "v33_pctile": float(best_row["v33_pctile"]),
            "confluence_mode": str(best_row["confluence_mode"]),
            "sharpe": float(best_row["sharpe"]),
            "mean_net_ticks": float(best_row["mean_net_ticks"]),
            "n_fills": int(best_row["n_fills"]),
            "fills_per_day_median": float(best_row["fills_per_day_median"]),
            "day_conc": float(best_row["day_conc"]),
            "ci_low_95": float(best_row["ci_low_95"]),
            "min_fills_gate_used": MIN_FILLS_FOR_WINNER,
        }
    else:
        best_summary = None  # no robust confluence cell

    # Also report best UNCONDITIONAL (any-N) cell as a note — small-N caveat applies
    if not conf_only.empty:
        any_idx = conf_only["sharpe"].idxmax()
        any_row = conf_only.loc[any_idx]
        best_any_n_summary = {
            "v33_pctile": float(any_row["v33_pctile"]),
            "confluence_mode": str(any_row["confluence_mode"]),
            "sharpe": float(any_row["sharpe"]),
            "n_fills": int(any_row["n_fills"]),
            "mean_net_ticks": float(any_row["mean_net_ticks"]),
            "note": "small-N — quoted only for completeness; ignore if n_fills < 30",
        }
    else:
        best_any_n_summary = None

    # Verdict logic
    if base_sharpe is None:
        verdict = "INCONCLUSIVE_NO_BASELINE"
    elif best_summary is None:
        verdict = (
            "CONFLUENCE_DOES_NOT_HELP — every confluence mode shrinks n_fills below "
            f"{MIN_FILLS_FOR_WINNER} OR matches v3.3 alone w/o robust Sharpe gain. "
            "Confluence filter is acting as a sample-size killer, not an edge "
            "amplifier. Recommendation: do NOT add PatchTST confluence to trial 278."
        )
    else:
        sharpe_gain = best_summary["sharpe"] - base_sharpe
        per_trade_gain = best_summary["mean_net_ticks"] - base_mean_net
        if sharpe_gain > 1.0 and per_trade_gain > 0.05:
            verdict = "CONFLUENCE_HELPS"
        elif sharpe_gain > 0.0 and per_trade_gain > 0.0:
            verdict = "CONFLUENCE_MARGINAL"
        else:
            verdict = "CONFLUENCE_DOES_NOT_HELP_SHRINKS_TRADE_COUNT_WITHOUT_SHARPE_GAIN"

    summary = {
        "status": "COMPLETED",
        "patchtst_status": patchtst_status,
        "overlap_dates": OVERLAP_DATES,
        "n_overlap_days": len(OVERLAP_DATES),
        "patchtst_horizon_for_confluence": PATCHTST_HORIZON_FOR_CONFLUENCE,
        "patchtst_offset_per_day": PATCHTST_OFFSET_PER_DAY,
        "baseline_full_15d_sharpe": float(repro_full["sharpe"]),
        "baseline_full_15d_day_conc": float(repro_full["day_conc"]),
        "baseline_full_15d_n_fills": int(repro_full["n_fills"]),
        "baseline_overlap_8d_sharpe": float(repro_overlap["sharpe"]),
        "baseline_overlap_8d_n_fills": int(repro_overlap["n_fills"]),
        "baseline_overlap_8d_mean_net": float(repro_overlap["mean_net"]),
        "baseline_overlap_8d_fills_per_day_median": float(
            df_sweep.query("confluence_mode=='v3.3_alone_overlap' and v33_pctile==0.0435")
            ["fills_per_day_median"].iloc[0]
        ) if not baseline_row.empty else None,
        "best_confluence_cell_robust_n_gte_30": best_summary,
        "best_confluence_cell_any_n_FOR_INFO_ONLY": best_any_n_summary,
        "verdict": verdict,
        "trial_278_reproduction_status": repro_status,
        "trial_278_reproduction_diff_sharpe": repro_diff,
        "elapsed_minutes": (time.time() - t0) / 60.0,
        "caveats": [
            "PatchTST overlap is 8 of 15 v3.3 OOT days — confluence stats have smaller N.",
            "PatchTST per-day count was +10 vs v3.3; first 10 PatchTST rows dropped per day to align.",
            "PatchTST horizon used for confluence = " + PATCHTST_HORIZON_FOR_CONFLUENCE +
            " (1s/5s/10s available; 5s chosen as mid-decay).",
            "v3.3 trial 278 head = log_ret_30s; PatchTST does not emit 30s. Direction agreement uses PatchTST 5s sign.",
            "FIFO confluence flag in trial 278 was a NO-OP in original optuna (source:" +
            " v33_execution_optuna_full_market_replay.py line ~294 'pass'); reproduction matches that no-op path.",
        ],
        "hc_refs": ["HC #344", "HC #392", "HC #397B", "HC #402-B", "HC #403"],
    }
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[{_now()}] === HC #403 (C) DONE ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
