#!/usr/bin/env python3
"""
MFE/MAE Horizon Analysis for ES Futures Execution Optimization
==============================================================
Comprehensive analysis of Maximum Favorable/Adverse Excursion across multiple
time horizons, confidence bands, and signal directions for CNN-Mamba v2 predictions.

Key outputs:
  - MFE/MAE statistics at 1s, 2s, 5s, 10s, 30s, 60s horizons
  - Breakdown by confidence band: top-50%, 20%, 10%, 5%, 1%, 0.5%
  - Breakdown by direction: long vs short signals
  - Breakdown by prediction horizon: 1s pred vs 5s pred vs 10s pred
  - Optimal TP/SL levels derived from MFE/MAE distributions
  - Profitability at each TP/SL combo after commission

Data sources:
  - CNN-Mamba v2 fold predictions: fold_XX_oot_predictions.npz
  - MBO event data (smart_v3): mbo_events_smart_v3/YYYYMMDD_mbo_events.npz
  - Raw MBO event data: mbo_events/YYYYMMDD_mbo_events.npz (for labels)

Mid-price path reconstruction:
  Uses linear interpolation between stored label horizons (1s, 5s, 10s, 30s)
  to approximate mid-price at any time point within the analysis window.

Cost model (HC #231(A)):
  ES tick = $12.50 (0.25 points)
  Commission RT = $4.70 = 0.376 ticks
  All entries: cost = 0.376 ticks (commission only — no spread crossing cost)

Usage:
    python -u mfe_mae_horizon_analysis.py
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ES_TICK_SIZE = 0.25           # points per tick
ES_TICK_VALUE = 12.50         # dollars per tick
ES_RT_COMMISSION = 4.70       # dollars round-trip
ES_COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376 ticks
ES_SPREAD_TICKS = 0.0         # HC #231(A): no spread cost (was 1.0 — fictitious)

COST_PASSIVE_TICKS = ES_COMMISSION_TICKS                     # 0.376
COST_MARKET_TICKS = ES_COMMISSION_TICKS                      # HC #231(A): commission only

# Prediction directories
PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
MBO_SMART_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
MBO_RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/mfe_mae_horizon_analysis")

# Analysis parameters
ANALYSIS_HORIZONS_SEC = [1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
CONFIDENCE_BANDS = {
    "top_50pct": 0.50,
    "top_20pct": 0.20,
    "top_10pct": 0.10,
    "top_5pct": 0.05,
    "top_1pct": 0.01,
    "top_0.5pct": 0.005,
}
PRED_HORIZON_NAMES = ["1s", "5s", "10s"]

# Label horizons in data
LABEL_HORIZON_SEC = np.array([0.0, 1.0, 5.0, 10.0, 30.0], dtype=np.float64)
LABEL_KEYS = ["labels_1s", "labels_5s", "labels_10s", "labels_30s"]

# Window / stride parameters (must match training config)
WINDOW_SIZE = 3000
DEFAULT_STRIDE = 500

# Max workers for parallel processing
MAX_WORKERS = 12

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MFE/MAE] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("mfe_mae_horizon")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ns_to_sec(ns: int) -> float:
    return ns / 1_000_000_000.0


def extract_date_from_path(p: str) -> Optional[str]:
    """Extract YYYYMMDD from a path string."""
    import re
    m = re.search(r"(\d{8})_mbo_events", p)
    return m.group(1) if m else None


def percentile_stats(arr: np.ndarray) -> Dict:
    """Compute standard percentile statistics for an array."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"mean": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "std": 0, "n": 0}
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p75": float(np.percentile(valid, 75)),
        "p90": float(np.percentile(valid, 90)),
        "p95": float(np.percentile(valid, 95)),
        "std": float(np.std(valid)),
        "n": int(len(valid)),
    }


# ---------------------------------------------------------------------------
# Core: Reconstruct prediction event indices
# ---------------------------------------------------------------------------
def reconstruct_event_indices(
    mbo_path: Path,
    n_preds: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Reconstruct which event indices correspond to each prediction.

    Returns:
        event_indices: (N_preds,) int64 - event index for each prediction
        timestamps: (N_all,) int64 - all event timestamps
        labels: dict with keys labels_1s, labels_5s, labels_10s, labels_30s
    """
    data = np.load(mbo_path, allow_pickle=True)
    timestamps = data["timestamps"]
    n_events = len(timestamps)

    labels = {}
    for key in LABEL_KEYS:
        if key in data:
            labels[key] = data[key].astype(np.float32)

    # Reconstruct using windowing logic from training
    def count_with_stride(stride: int) -> np.ndarray:
        starts = np.arange(0, n_events - WINDOW_SIZE + 1, stride)
        label_idxs = starts + WINDOW_SIZE - 1
        # Valid = non-NaN at all required horizons
        valid = np.ones(len(label_idxs), dtype=bool)
        for h in ["labels_1s", "labels_5s", "labels_10s"]:
            if h in labels:
                valid &= ~np.isnan(labels[h][label_idxs])
        return label_idxs[valid]

    # Try default stride
    indices = count_with_stride(DEFAULT_STRIDE)

    # Auto-detect stride if mismatch
    if abs(len(indices) - n_preds) > 10:
        candidates = [
            WINDOW_SIZE // 6,    # 500
            WINDOW_SIZE // 12,   # 250
            WINDOW_SIZE // 4,    # 750
            WINDOW_SIZE // 8,    # 375
            WINDOW_SIZE // 10,   # 300
            WINDOW_SIZE // 3,    # 1000
            WINDOW_SIZE // 2,    # 1500
        ]
        best_stride = DEFAULT_STRIDE
        best_diff = abs(len(indices) - n_preds)

        for cs in candidates:
            if cs <= 0 or cs == DEFAULT_STRIDE:
                continue
            test = count_with_stride(cs)
            diff = abs(len(test) - n_preds)
            if diff < best_diff:
                best_diff = diff
                best_stride = cs
                indices = test

        if best_stride != DEFAULT_STRIDE:
            log.info(f"  Stride auto-detected: {best_stride} -> {len(indices)} samples (target {n_preds})")

    # Truncate or pad to match exactly
    if len(indices) > n_preds:
        indices = indices[:n_preds]
    elif len(indices) < n_preds:
        log.warning(f"  Index count {len(indices)} < predictions {n_preds}, using available")

    return indices, timestamps, labels


# ---------------------------------------------------------------------------
# Core: Compute MFE/MAE at multiple horizons for a single fold
# ---------------------------------------------------------------------------
def compute_mfe_mae_for_fold(fold_idx: int) -> Optional[Dict]:
    """
    Process one fold: load predictions, match to MBO events, compute MFE/MAE
    at multiple time horizons.

    Returns dict with all results for this fold, or None on failure.
    """
    t0 = time.time()
    pred_file = PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not pred_file.exists():
        log.warning(f"Fold {fold_idx:02d}: prediction file not found")
        return None

    # Load predictions
    pred_data = np.load(pred_file, allow_pickle=True)
    predictions = pred_data["predictions"]  # (N, 3) for 1s/5s/10s
    labels_pred = pred_data["labels"]       # (N, 3) actual labels
    n_preds = predictions.shape[0]

    # Get OOT date
    oot_files = pred_data["oot_files"]
    if len(oot_files) == 0:
        log.warning(f"Fold {fold_idx:02d}: no oot_files")
        return None

    oot_path = str(oot_files[0])
    date_str = extract_date_from_path(oot_path)
    if not date_str:
        log.warning(f"Fold {fold_idx:02d}: could not extract date from {oot_path}")
        return None

    # Find MBO data (try smart_v3 first, then raw)
    mbo_path = MBO_SMART_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        mbo_path = MBO_RAW_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        log.warning(f"Fold {fold_idx:02d}: MBO data not found for {date_str}")
        return None

    log.info(f"Fold {fold_idx:02d}: {date_str}, {n_preds} predictions, loading MBO...")

    # Reconstruct event indices
    try:
        event_indices, timestamps, day_labels = reconstruct_event_indices(mbo_path, n_preds)
    except Exception as e:
        log.error(f"Fold {fold_idx:02d}: index reconstruction failed: {e}")
        return None

    n_matched = min(len(event_indices), n_preds)
    if n_matched < 100:
        log.warning(f"Fold {fold_idx:02d}: only {n_matched} matched events, skipping")
        return None

    # Trim to matched count
    event_indices = event_indices[:n_matched]
    predictions = predictions[:n_matched]
    labels_pred = labels_pred[:n_matched]

    n_events = len(timestamps)

    # Build label value arrays at prediction indices for interpolation
    # For each prediction event k, we have mid-price change at 0s, 1s, 5s, 10s, 30s
    label_at_pred = np.zeros((n_matched, 5), dtype=np.float64)  # (N, 5) horizons
    label_at_pred[:, 0] = 0.0  # at t=0, change = 0
    for hi, key in enumerate(LABEL_KEYS, 1):
        if key in day_labels:
            vals = day_labels[key][event_indices]
            # Replace NaN with previous horizon value
            nan_mask = np.isnan(vals)
            vals[nan_mask] = label_at_pred[nan_mask, hi - 1]
            label_at_pred[:, hi] = vals

    # Compute MFE/MAE at each analysis horizon
    # For each prediction and each horizon H:
    #   MFE_H = max favorable excursion within [0, H] seconds
    #   MAE_H = max adverse excursion within [0, H] seconds
    #
    # We approximate the mid-price path by interpolating between label horizons.
    # For denser path, we also look at all forward events within the window
    # and interpolate their mid-price from the prediction event's labels.

    results = {
        "fold": fold_idx,
        "date": date_str,
        "n_predictions": n_matched,
        "horizons": {},
    }

    # For each prediction horizon (1s, 5s, 10s predictions)
    for pred_h_idx, pred_h_name in enumerate(PRED_HORIZON_NAMES):
        pred_vals = predictions[:, pred_h_idx]
        direction = np.sign(pred_vals).astype(np.float64)
        direction[direction == 0] = 1.0  # neutral -> long

        # Confidence = absolute prediction value
        confidence = np.abs(pred_vals)

        # For each analysis horizon, compute MFE/MAE
        horizon_results = {}
        for analysis_h_sec in ANALYSIS_HORIZONS_SEC:
            # Sample dense path points between 0 and analysis_h_sec
            # Use 20 evenly spaced points + the label horizons for interpolation
            sample_times = np.sort(np.unique(np.concatenate([
                np.linspace(0, analysis_h_sec, 20),
                LABEL_HORIZON_SEC[LABEL_HORIZON_SEC <= analysis_h_sec],
                [analysis_h_sec],
            ])))

            # Interpolate mid-price change at each sample time for all predictions
            # label_at_pred: (N, 5) with horizons at [0, 1, 5, 10, 30] sec
            # For horizons > 30s, extrapolate linearly from 10s->30s slope
            mid_at_samples = np.zeros((n_matched, len(sample_times)), dtype=np.float64)
            for t_idx, t_sec in enumerate(sample_times):
                if t_sec <= 30.0:
                    mid_at_samples[:, t_idx] = np.array([
                        np.interp(t_sec, LABEL_HORIZON_SEC, label_at_pred[i])
                        for i in range(n_matched)
                    ])
                else:
                    # Linear extrapolation from 10s->30s slope
                    slope = (label_at_pred[:, 4] - label_at_pred[:, 3]) / 20.0  # per second
                    mid_at_samples[:, t_idx] = label_at_pred[:, 4] + slope * (t_sec - 30.0)

            # Compute directional path (positive = favorable)
            dir_path = mid_at_samples * direction[:, None]  # (N, T)

            # MFE = max(dir_path) within window (always >= 0)
            mfe = np.maximum(np.nanmax(dir_path, axis=1), 0.0)

            # MAE = max adverse = -min(dir_path) within window (always >= 0)
            mae = np.maximum(-np.nanmin(dir_path, axis=1), 0.0)

            # Store per-prediction results
            h_key = f"{analysis_h_sec:.0f}s" if analysis_h_sec >= 1 else f"{analysis_h_sec*1000:.0f}ms"
            horizon_results[h_key] = {
                "mfe": mfe.astype(np.float32),
                "mae": mae.astype(np.float32),
                "direction": direction.astype(np.float32),
                "confidence": confidence.astype(np.float32),
            }

        results["horizons"][pred_h_name] = horizon_results

    elapsed = time.time() - t0
    log.info(f"Fold {fold_idx:02d}: completed in {elapsed:.1f}s ({n_matched} predictions)")
    return results


# ---------------------------------------------------------------------------
# Vectorized interpolation (faster than per-element np.interp)
# ---------------------------------------------------------------------------
def vectorized_interp(x_new: float, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Vectorized interpolation: for each row in fp (shape N x len(xp)),
    interpolate at x_new.

    Args:
        x_new: scalar query point
        xp: (K,) x-coordinates of data points (sorted)
        fp: (N, K) y-values for each of N samples

    Returns:
        (N,) interpolated values
    """
    # Find bracketing indices
    idx = np.searchsorted(xp, x_new, side='right')
    idx = np.clip(idx, 1, len(xp) - 1)

    x_lo = xp[idx - 1]
    x_hi = xp[idx]
    dx = x_hi - x_lo
    if dx == 0:
        return fp[:, idx - 1]

    t = (x_new - x_lo) / dx
    return fp[:, idx - 1] * (1 - t) + fp[:, idx] * t


def compute_mfe_mae_for_fold_vectorized(fold_idx: int) -> Optional[Dict]:
    """
    Vectorized version of fold processing. Much faster than per-prediction loops.
    """
    t0 = time.time()
    pred_file = PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not pred_file.exists():
        log.warning(f"Fold {fold_idx:02d}: prediction file not found")
        return None

    pred_data = np.load(pred_file, allow_pickle=True)
    predictions = pred_data["predictions"]  # (N, 3)
    n_preds = predictions.shape[0]

    oot_files = pred_data["oot_files"]
    if len(oot_files) == 0:
        return None

    oot_path = str(oot_files[0])
    date_str = extract_date_from_path(oot_path)
    if not date_str:
        return None

    mbo_path = MBO_SMART_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        mbo_path = MBO_RAW_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        log.warning(f"Fold {fold_idx:02d}: MBO not found for {date_str}")
        return None

    log.info(f"Fold {fold_idx:02d}: {date_str}, {n_preds} preds")

    try:
        event_indices, timestamps, day_labels = reconstruct_event_indices(mbo_path, n_preds)
    except Exception as e:
        log.error(f"Fold {fold_idx:02d}: {e}")
        return None

    n_matched = min(len(event_indices), n_preds)
    if n_matched < 100:
        log.warning(f"Fold {fold_idx:02d}: only {n_matched} matched")
        return None

    event_indices = event_indices[:n_matched]
    predictions = predictions[:n_matched]

    # Build label interpolation table: (N, 5) at horizons [0, 1, 5, 10, 30] sec
    label_at_pred = np.zeros((n_matched, 5), dtype=np.float64)
    for hi, key in enumerate(LABEL_KEYS, 1):
        if key in day_labels:
            vals = day_labels[key][event_indices].astype(np.float64)
            nan_mask = np.isnan(vals)
            vals[nan_mask] = label_at_pred[nan_mask, hi - 1]
            label_at_pred[:, hi] = vals

    # Also check if we have labels_30s for 60s extrapolation
    has_30s = "labels_30s" in day_labels

    results = {
        "fold": fold_idx,
        "date": date_str,
        "n_predictions": n_matched,
        "horizons": {},
    }

    for pred_h_idx, pred_h_name in enumerate(PRED_HORIZON_NAMES):
        pred_vals = predictions[:, pred_h_idx].astype(np.float64)
        direction = np.sign(pred_vals)
        direction[direction == 0] = 1.0
        confidence = np.abs(pred_vals)

        horizon_results = {}
        for analysis_h_sec in ANALYSIS_HORIZONS_SEC:
            # Create sample points within [0, analysis_h_sec]
            # Use more points for longer horizons
            n_samples = max(20, int(analysis_h_sec * 4))
            sample_times = np.sort(np.unique(np.concatenate([
                np.linspace(0, min(analysis_h_sec, 30.0), n_samples),
                LABEL_HORIZON_SEC[LABEL_HORIZON_SEC <= analysis_h_sec],
                [min(analysis_h_sec, 30.0)],
            ])))
            if analysis_h_sec > 30.0:
                extra = np.linspace(30.0, analysis_h_sec, max(10, int((analysis_h_sec - 30) * 2)))
                sample_times = np.sort(np.unique(np.concatenate([sample_times, extra])))

            # Vectorized interpolation for all predictions at all sample times
            # (N, T) matrix of mid-price changes
            mid_at_samples = np.zeros((n_matched, len(sample_times)), dtype=np.float64)
            for t_idx, t_sec in enumerate(sample_times):
                if t_sec <= 30.0:
                    mid_at_samples[:, t_idx] = vectorized_interp(
                        t_sec, LABEL_HORIZON_SEC, label_at_pred
                    )
                else:
                    # Extrapolate from 10s->30s slope
                    slope = (label_at_pred[:, 4] - label_at_pred[:, 3]) / 20.0
                    mid_at_samples[:, t_idx] = label_at_pred[:, 4] + slope * (t_sec - 30.0)

            # Directional path
            dir_path = mid_at_samples * direction[:, None]

            # MFE / MAE
            mfe = np.maximum(np.nanmax(dir_path, axis=1), 0.0).astype(np.float32)
            mae = np.maximum(-np.nanmin(dir_path, axis=1), 0.0).astype(np.float32)

            h_key = f"{int(analysis_h_sec)}s"
            horizon_results[h_key] = {
                "mfe": mfe,
                "mae": mae,
                "direction": direction.astype(np.float32),
                "confidence": confidence.astype(np.float32),
            }

        results["horizons"][pred_h_name] = horizon_results

    elapsed = time.time() - t0
    log.info(f"Fold {fold_idx:02d}: done in {elapsed:.1f}s")
    return results


# ---------------------------------------------------------------------------
# Analysis: Aggregate statistics across folds
# ---------------------------------------------------------------------------
def compute_statistics(
    all_mfe: np.ndarray,
    all_mae: np.ndarray,
    all_direction: np.ndarray,
    all_confidence: np.ndarray,
) -> Dict:
    """
    Compute full statistics breakdown by confidence band and direction.

    Returns nested dict with all stats.
    """
    results = {}
    n_total = len(all_mfe)

    # Overall stats
    results["overall"] = {
        "n": n_total,
        "mfe": percentile_stats(all_mfe),
        "mae": percentile_stats(all_mae),
    }

    # By direction
    for dir_name, dir_val in [("long", 1.0), ("short", -1.0)]:
        mask = all_direction == dir_val
        if mask.sum() < 10:
            continue
        results[f"direction_{dir_name}"] = {
            "n": int(mask.sum()),
            "mfe": percentile_stats(all_mfe[mask]),
            "mae": percentile_stats(all_mae[mask]),
        }

    # By confidence band
    for band_name, band_pct in CONFIDENCE_BANDS.items():
        threshold = np.percentile(all_confidence, 100 * (1 - band_pct))
        mask = all_confidence >= threshold
        if mask.sum() < 5:
            continue

        band_results = {
            "n": int(mask.sum()),
            "threshold": float(threshold),
            "mfe": percentile_stats(all_mfe[mask]),
            "mae": percentile_stats(all_mae[mask]),
        }

        # Also break by direction within band
        for dir_name, dir_val in [("long", 1.0), ("short", -1.0)]:
            dir_mask = mask & (all_direction == dir_val)
            if dir_mask.sum() < 5:
                continue
            band_results[f"direction_{dir_name}"] = {
                "n": int(dir_mask.sum()),
                "mfe": percentile_stats(all_mfe[dir_mask]),
                "mae": percentile_stats(all_mae[dir_mask]),
            }

        results[band_name] = band_results

    return results


def compute_optimal_tp_sl(stats: Dict) -> Dict:
    """
    Compute optimal TP/SL from MFE/MAE distribution.

    TP = p50 of MFE (hit rate ~50%)
    SL = p90 of MAE (only breached 10% of the time)
    """
    mfe_stats = stats.get("mfe", {})
    mae_stats = stats.get("mae", {})

    tp = mfe_stats.get("median", 0)    # p50 of MFE
    sl = mae_stats.get("p90", 0)       # p90 of MAE

    return {
        "tp_ticks": tp,
        "sl_ticks": sl,
        "tp_sl_ratio": tp / sl if sl > 0 else float("inf"),
    }


def compute_profitability(
    all_mfe: np.ndarray,
    all_mae: np.ndarray,
    tp_ticks: float,
    sl_ticks: float,
    cost_ticks: float,
) -> Dict:
    """
    Simulate profitability at given TP/SL levels.

    For each prediction:
      - If MFE >= TP before MAE >= SL: WIN (profit = TP - cost)
      - If MAE >= SL before MFE >= TP: LOSS (loss = -SL - cost)
      - If neither: assume exit at mid (use MFE - MAE as proxy), subtract cost

    Note: This is approximate since we don't have exact timing of MFE vs MAE.
    We use the heuristic that if MFE > TP AND MAE < SL, it's a win.
    """
    n = len(all_mfe)
    if n == 0 or tp_ticks <= 0 or sl_ticks <= 0:
        return {"win_rate": 0, "expected_pnl_ticks": 0, "n": 0}

    # Win: MFE reached TP (we assume favorable path before adverse)
    # Loss: MAE reached SL
    # Both: check which was likely hit first (approximate)
    wins = (all_mfe >= tp_ticks) & (all_mae < sl_ticks)
    losses = (all_mae >= sl_ticks) & (all_mfe < tp_ticks)
    both = (all_mfe >= tp_ticks) & (all_mae >= sl_ticks)
    neither = ~wins & ~losses & ~both

    # For "both" cases, assume 50% win (conservative)
    n_wins = wins.sum() + 0.5 * both.sum()
    n_losses = losses.sum() + 0.5 * both.sum()
    n_neither = neither.sum()

    win_rate = n_wins / n if n > 0 else 0

    # P&L calculation
    pnl_wins = n_wins * (tp_ticks - cost_ticks)
    pnl_losses = n_losses * (-sl_ticks - cost_ticks)
    # Neither: average of (MFE - MAE) / 2 as proxy for exit price, minus cost
    if n_neither > 0:
        avg_neither_pnl = float(np.mean(all_mfe[neither] - all_mae[neither]) / 2.0) - cost_ticks
        pnl_neither = n_neither * avg_neither_pnl
    else:
        pnl_neither = 0

    total_pnl = pnl_wins + pnl_losses + pnl_neither
    expected_pnl = total_pnl / n if n > 0 else 0

    return {
        "win_rate": float(win_rate),
        "n_wins": float(n_wins),
        "n_losses": float(n_losses),
        "n_both": int(both.sum()),
        "n_neither": int(n_neither),
        "expected_pnl_ticks": float(expected_pnl),
        "expected_pnl_dollars": float(expected_pnl * ES_TICK_VALUE),
        "tp_ticks": float(tp_ticks),
        "sl_ticks": float(sl_ticks),
        "cost_ticks": float(cost_ticks),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------
def run_analysis():
    """Main entry point: process all folds and aggregate results."""
    log.info("=" * 70)
    log.info("MFE/MAE Horizon Analysis — ES Futures Execution Optimization")
    log.info("=" * 70)
    log.info(f"Prediction dir: {PRED_DIR}")
    log.info(f"MBO dir (smart_v3): {MBO_SMART_DIR}")
    log.info(f"Output dir: {OUTPUT_DIR}")
    log.info(f"Analysis horizons: {ANALYSIS_HORIZONS_SEC}")
    log.info(f"Confidence bands: {list(CONFIDENCE_BANDS.keys())}")
    log.info(f"Cost model: passive={COST_PASSIVE_TICKS:.3f} ticks, market={COST_MARKET_TICKS:.3f} ticks")
    log.info("")

    # Find all fold prediction files
    fold_files = sorted(PRED_DIR.glob("fold_*_oot_predictions.npz"))
    fold_indices = []
    for f in fold_files:
        try:
            idx = int(f.stem.split("_")[1])
            fold_indices.append(idx)
        except (ValueError, IndexError):
            continue

    log.info(f"Found {len(fold_indices)} folds: {fold_indices}")

    # Process folds in parallel
    fold_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(compute_mfe_mae_for_fold_vectorized, idx): idx
            for idx in fold_indices
        }
        for future in as_completed(futures):
            fold_idx = futures[future]
            try:
                result = future.result()
                if result is not None:
                    fold_results.append(result)
            except Exception as e:
                log.error(f"Fold {fold_idx:02d} failed: {e}")
                import traceback
                traceback.print_exc()

    if not fold_results:
        log.error("No folds processed successfully!")
        return

    log.info(f"\nProcessed {len(fold_results)} folds successfully")
    log.info("Aggregating results...")

    # Aggregate across folds
    # Structure: pred_horizon -> analysis_horizon -> concatenated arrays
    aggregated = {}

    for pred_h_name in PRED_HORIZON_NAMES:
        aggregated[pred_h_name] = {}
        for analysis_h_key in [f"{int(h)}s" for h in ANALYSIS_HORIZONS_SEC]:
            all_mfe = []
            all_mae = []
            all_dir = []
            all_conf = []

            for fold_res in fold_results:
                if pred_h_name not in fold_res["horizons"]:
                    continue
                if analysis_h_key not in fold_res["horizons"][pred_h_name]:
                    continue

                h_data = fold_res["horizons"][pred_h_name][analysis_h_key]
                all_mfe.append(h_data["mfe"])
                all_mae.append(h_data["mae"])
                all_dir.append(h_data["direction"])
                all_conf.append(h_data["confidence"])

            if not all_mfe:
                continue

            mfe_cat = np.concatenate(all_mfe)
            mae_cat = np.concatenate(all_mae)
            dir_cat = np.concatenate(all_dir)
            conf_cat = np.concatenate(all_conf)

            # Compute statistics
            stats = compute_statistics(mfe_cat, mae_cat, dir_cat, conf_cat)

            # Compute optimal TP/SL for each confidence band
            profitability = {}
            for band_name in list(CONFIDENCE_BANDS.keys()) + ["overall"]:
                if band_name not in stats:
                    continue

                band_stats = stats[band_name]
                tp_sl = compute_optimal_tp_sl(band_stats)
                tp = tp_sl["tp_ticks"]
                sl = tp_sl["sl_ticks"]

                # Get the subset for profitability calc
                if band_name == "overall":
                    mask = np.ones(len(mfe_cat), dtype=bool)
                else:
                    threshold = band_stats.get("threshold", 0)
                    mask = conf_cat >= threshold

                if mask.sum() < 5 or tp <= 0 or sl <= 0:
                    continue

                # Profitability with passive entry
                prof_passive = compute_profitability(
                    mfe_cat[mask], mae_cat[mask], tp, sl, COST_PASSIVE_TICKS
                )
                # Profitability with market entry
                prof_market = compute_profitability(
                    mfe_cat[mask], mae_cat[mask], tp, sl, COST_MARKET_TICKS
                )

                profitability[band_name] = {
                    "optimal_tp_sl": tp_sl,
                    "passive_entry": prof_passive,
                    "market_entry": prof_market,
                }

                # Also try a tighter TP (p75 MFE) and wider SL (p95 MAE)
                tp_tight = band_stats["mfe"].get("p75", tp)
                sl_wide = band_stats["mae"].get("p95", sl)
                if tp_tight > 0 and sl_wide > 0:
                    profitability[band_name]["tight_tp_wide_sl_passive"] = compute_profitability(
                        mfe_cat[mask], mae_cat[mask], tp_tight, sl_wide, COST_PASSIVE_TICKS
                    )

            stats["profitability"] = profitability
            aggregated[pred_h_name][analysis_h_key] = stats

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
            return str(obj)
        return obj

    output_json = OUTPUT_DIR / "mfe_mae_horizon_results.json"
    with open(output_json, "w") as f:
        json.dump(make_serializable(aggregated), f, indent=2)
    log.info(f"\nResults saved to: {output_json}")

    # Print summary table
    print_summary(aggregated)

    return aggregated


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(aggregated: Dict):
    """Print a formatted summary table of key results."""
    print("\n" + "=" * 90)
    print("MFE/MAE HORIZON ANALYSIS — SUMMARY")
    print("=" * 90)

    for pred_h_name in PRED_HORIZON_NAMES:
        if pred_h_name not in aggregated:
            continue

        print(f"\n{'─' * 90}")
        print(f"PREDICTION HORIZON: {pred_h_name}")
        print(f"{'─' * 90}")

        # Header
        print(f"\n{'Analysis':<10} {'Band':<12} {'Dir':<6} {'N':>7} "
              f"{'MFE_mean':>8} {'MFE_p50':>7} {'MFE_p75':>7} "
              f"{'MAE_mean':>8} {'MAE_p50':>7} {'MAE_p90':>7} "
              f"{'TP/SL':>6}")
        print("-" * 90)

        for analysis_h_key in [f"{int(h)}s" for h in ANALYSIS_HORIZONS_SEC]:
            if analysis_h_key not in aggregated[pred_h_name]:
                continue

            stats = aggregated[pred_h_name][analysis_h_key]

            # Print overall
            ov = stats.get("overall", {})
            if ov:
                mfe_s = ov.get("mfe", {})
                mae_s = ov.get("mae", {})
                tp_sl = mfe_s.get("median", 0) / mae_s.get("p90", 1) if mae_s.get("p90", 0) > 0 else 0
                print(f"{analysis_h_key:<10} {'ALL':<12} {'all':<6} {ov.get('n', 0):>7} "
                      f"{mfe_s.get('mean', 0):>8.2f} {mfe_s.get('median', 0):>7.2f} {mfe_s.get('p75', 0):>7.2f} "
                      f"{mae_s.get('mean', 0):>8.2f} {mae_s.get('median', 0):>7.2f} {mae_s.get('p90', 0):>7.2f} "
                      f"{tp_sl:>6.2f}")

            # Print top confidence bands
            for band_name in ["top_10pct", "top_5pct", "top_1pct"]:
                if band_name not in stats:
                    continue
                band = stats[band_name]
                mfe_s = band.get("mfe", {})
                mae_s = band.get("mae", {})
                tp_sl = mfe_s.get("median", 0) / mae_s.get("p90", 1) if mae_s.get("p90", 0) > 0 else 0

                # Overall for this band
                print(f"{'':<10} {band_name:<12} {'all':<6} {band.get('n', 0):>7} "
                      f"{mfe_s.get('mean', 0):>8.2f} {mfe_s.get('median', 0):>7.2f} {mfe_s.get('p75', 0):>7.2f} "
                      f"{mae_s.get('mean', 0):>8.2f} {mae_s.get('median', 0):>7.2f} {mae_s.get('p90', 0):>7.2f} "
                      f"{tp_sl:>6.2f}")

                # By direction within band
                for dir_name in ["direction_long", "direction_short"]:
                    if dir_name not in band:
                        continue
                    d = band[dir_name]
                    mfe_d = d.get("mfe", {})
                    mae_d = d.get("mae", {})
                    dir_label = dir_name.split("_")[1][:5]
                    print(f"{'':<10} {'':<12} {dir_label:<6} {d.get('n', 0):>7} "
                          f"{mfe_d.get('mean', 0):>8.2f} {mfe_d.get('median', 0):>7.2f} {mfe_d.get('p75', 0):>7.2f} "
                          f"{mae_d.get('mean', 0):>8.2f} {mae_d.get('median', 0):>7.2f} {mae_d.get('p90', 0):>7.2f} "
                          f"{'':>6}")

        # Print profitability summary
        print(f"\n  PROFITABILITY ANALYSIS (pred_horizon={pred_h_name}):")
        print(f"  {'Analysis':<8} {'Band':<12} {'TP':>6} {'SL':>6} {'WR%':>6} "
              f"{'E[PnL] passive':>14} {'E[PnL] market':>14} {'$/trade pass':>12} {'$/trade mkt':>12}")
        print(f"  {'-' * 84}")

        for analysis_h_key in [f"{int(h)}s" for h in ANALYSIS_HORIZONS_SEC]:
            if analysis_h_key not in aggregated[pred_h_name]:
                continue
            stats = aggregated[pred_h_name][analysis_h_key]
            prof = stats.get("profitability", {})

            for band_name in ["top_10pct", "top_5pct", "top_1pct", "top_0.5pct"]:
                if band_name not in prof:
                    continue
                p = prof[band_name]
                tp_sl = p.get("optimal_tp_sl", {})
                passive = p.get("passive_entry", {})
                market = p.get("market_entry", {})

                tp = tp_sl.get("tp_ticks", 0)
                sl = tp_sl.get("sl_ticks", 0)
                wr = passive.get("win_rate", 0) * 100
                pnl_pass = passive.get("expected_pnl_ticks", 0)
                pnl_mkt = market.get("expected_pnl_ticks", 0)
                dollar_pass = passive.get("expected_pnl_dollars", 0)
                dollar_mkt = market.get("expected_pnl_dollars", 0)

                print(f"  {analysis_h_key:<8} {band_name:<12} {tp:>6.2f} {sl:>6.2f} {wr:>6.1f} "
                      f"{pnl_pass:>14.3f} {pnl_mkt:>14.3f} "
                      f"${dollar_pass:>10.2f} ${dollar_mkt:>10.2f}")

    print("\n" + "=" * 90)
    print("COST ASSUMPTIONS:")
    print(f"  Passive entry: {COST_PASSIVE_TICKS:.3f} ticks (commission only)")
    print(f"  Market entry:  {COST_MARKET_TICKS:.3f} ticks (HC #231(A): commission only)")
    print(f"  ES tick value: ${ES_TICK_VALUE:.2f}")
    print(f"  RT commission: ${ES_RT_COMMISSION:.2f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_analysis()
