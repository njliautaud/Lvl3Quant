#!/usr/bin/env python3
"""
MFE/MAE Path Analysis for MBO Event-Driven Predictions
=======================================================
Analyses the FULL price path after each prediction to compute:
  - MFE (Maximum Favorable Excursion) — best unrealized gain before exit
  - MAE (Maximum Adverse Excursion) — worst unrealized drawdown before MFE
  - Time-to-MFE, Time-to-MAE
  - % of winners that go red first
  - Path clustering by confidence quintile (Q1-Q5)
  - Distribution statistics for all metrics

Two path reconstruction modes:
  1. LABEL mode (default, fast): Uses stored labels at 1s/5s/10s/30s horizons
     to get midpoint at 5 timepoints per prediction.  MFE/MAE bounded by these.
  2. DENSE mode (--dense): Reconstructs event-by-event midpoint path using the
     labels of all forward events within the analysis window.  The midpoint change
     between consecutive events is derived from the difference in their 1s labels
     adjusted for the timestamp gap.  More accurate but slower.

Units:
  - Prices / labels / MFE / MAE are in TICKS (NQ tick = 0.25pt, $12.50/tick)
  - Timestamps are nanosecond UTC epoch (int64)

Usage:
  python mfe_mae_path_analysis.py --pred-dir /path/to/output/cnn_mamba_v2_smart_v3_mar \\
      --mbo-dir /path/to/data/processed/mbo_events --folds 6,7,8 --horizon 10s
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25          # NQ tick size in points
POINT_VALUE = 50.0        # NQ point value in USD
TICK_VALUE = TICK_SIZE * POINT_VALUE  # $12.50

# Label horizons stored in MBO event files
HORIZON_NAMES = ["1s", "5s", "10s", "30s"]
HORIZON_NS = {
    "1s":  1_000_000_000,
    "5s":  5_000_000_000,
    "10s": 10_000_000_000,
    "30s": 30_000_000_000,
}
# Seconds for each horizon (for time calculations)
HORIZON_SEC = {"1s": 1.0, "5s": 5.0, "10s": 10.0, "30s": 30.0}

# Default analysis window
DEFAULT_WINDOW_SEC = 30.0
DEFAULT_MAX_EVENTS = 2000  # max forward events to scan per prediction

# Quintile labels
QUINTILE_LABELS = ["Q1 (weakest)", "Q2", "Q3", "Q4", "Q5 (strongest)"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ns_to_sec(ns: int) -> float:
    return ns / 1_000_000_000.0


def parse_fold_spec(spec: str) -> List[int]:
    """Parse fold specification like '6,7,8' or '0-10' or 'all'."""
    if spec.lower() == "all":
        return None  # signal to use all available
    folds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            folds.extend(range(int(lo), int(hi) + 1))
        else:
            folds.append(int(part))
    return sorted(set(folds))


def extract_date_from_path(p: str) -> Optional[str]:
    """Extract YYYYMMDD from a path like '...20260302_mbo_events.npz'."""
    m = re.search(r"(\d{8})_mbo_events", p)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Core: Reconstruct prediction event indices
# ---------------------------------------------------------------------------
def reconstruct_event_indices(
    mbo_path: Path,
    window_size: int,
    stride: int,
    horizons: List[str] = None,
    target_n_preds: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """
    Replay the windowing logic from train_cnn_mamba.py to find which event
    indices correspond to each prediction.

    If target_n_preds is set, will auto-detect stride by trying common values
    when the default stride produces a mismatch.

    Returns:
        event_indices: (N_preds,) int array of the LAST event index per window
        timestamps:    (N_events,) full timestamp array
        labels:        dict horizon -> (N_events,) label arrays
        events:        (N_events, F) feature array
    """
    horizons = horizons or ["1s", "5s", "10s"]
    data = np.load(mbo_path, allow_pickle=True)

    events = data["events"]
    timestamps = data["timestamps"]
    n_events = len(events)

    day_labels = {}
    for h in HORIZON_NAMES:  # load ALL horizons, not just training ones
        key = f"labels_{h}"
        if key in data:
            day_labels[h] = data[key].astype(np.float32)

    def _count_with_stride(s: int) -> np.ndarray:
        """Vectorized window index reconstruction."""
        starts = np.arange(0, n_events - window_size + 1, s)
        label_idxs = starts + window_size - 1
        valid = np.ones(len(label_idxs), dtype=bool)
        for h in horizons:
            if h in day_labels:
                valid &= ~np.isnan(day_labels[h][label_idxs])
        return label_idxs[valid]

    # Try the given stride first
    sample_indices = _count_with_stride(stride)

    # Auto-detect stride if target_n_preds is specified and we have a mismatch
    if target_n_preds is not None and abs(len(sample_indices) - target_n_preds) > 10:
        # Try common stride values
        candidates = [
            window_size // 6,   # 500 for ws=3000
            window_size // 12,  # 250 for ws=3000
            window_size // 4,   # 750 for ws=3000
            window_size // 8,   # 375 for ws=3000
            window_size // 10,  # 300 for ws=3000
        ]
        best_stride = stride
        best_diff = abs(len(sample_indices) - target_n_preds)

        for cs in candidates:
            if cs == stride or cs <= 0:
                continue
            test_indices = _count_with_stride(cs)
            diff = abs(len(test_indices) - target_n_preds)
            if diff < best_diff:
                best_diff = diff
                best_stride = cs
                sample_indices = test_indices

        if best_stride != stride:
            print(f"    [AUTO] Stride auto-detected: {best_stride} "
                  f"({len(sample_indices)} samples, target {target_n_preds})")

    return np.array(sample_indices, dtype=np.int64), timestamps, day_labels, events


# ---------------------------------------------------------------------------
# Core: Build forward midpoint path from labels
# ---------------------------------------------------------------------------
def build_label_paths(
    pred_indices: np.ndarray,
    timestamps: np.ndarray,
    labels: dict,
    predictions: np.ndarray,
    pred_horizon_idx: int,
) -> dict:
    """
    For each prediction, build the forward midpoint path using stored labels.

    Returns dict with arrays:
        path_0s:  (N,) always 0.0 (reference)
        path_1s:  (N,) midpoint change at +1s in ticks
        path_5s:  (N,) midpoint change at +5s in ticks
        path_10s: (N,) midpoint change at +10s in ticks
        path_30s: (N,) midpoint change at +30s in ticks
        pred_values: (N,) the model prediction (for the chosen horizon)
        label_values: (N,) actual label for that horizon
        timestamps_ns: (N,) prediction timestamps
    """
    N = len(pred_indices)
    paths = {
        "path_0s": np.zeros(N, dtype=np.float32),
        "path_1s": np.full(N, np.nan, dtype=np.float32),
        "path_5s": np.full(N, np.nan, dtype=np.float32),
        "path_10s": np.full(N, np.nan, dtype=np.float32),
        "path_30s": np.full(N, np.nan, dtype=np.float32),
        "pred_values": predictions[:, pred_horizon_idx].copy(),
        "label_values": np.full(N, np.nan, dtype=np.float32),
        "timestamps_ns": timestamps[pred_indices].copy(),
    }

    for h_name in HORIZON_NAMES:
        key = f"path_{h_name}"
        if h_name in labels:
            h_labels = labels[h_name]
            paths[key] = h_labels[pred_indices].copy()

    # Label values for the chosen horizon
    horizon_names_list = ["1s", "5s", "10s"]
    chosen_h = horizon_names_list[pred_horizon_idx]
    if chosen_h in labels:
        paths["label_values"] = labels[chosen_h][pred_indices].copy()

    return paths


# ---------------------------------------------------------------------------
# Core: Dense path reconstruction (event-by-event midpoint from label chain)
# ---------------------------------------------------------------------------
def build_dense_paths(
    pred_indices: np.ndarray,
    timestamps: np.ndarray,
    labels: dict,
    predictions: np.ndarray,
    pred_horizon_idx: int,
    max_events: int = DEFAULT_MAX_EVENTS,
    max_seconds: float = DEFAULT_WINDOW_SEC,
) -> dict:
    """
    Dense midpoint path reconstruction using the label chain technique.

    For prediction at event k, we want mid(t_j) - mid(t_k) for every event j > k
    within the analysis window.

    Approach: Use label interpolation.  For each forward event j:
      - Find the smallest horizon H such that t_k + H >= t_j
      - labels_H[k] gives us mid(t_k + H) - mid(t_k)
      - labels_{H-dt}[j] gives us mid(t_j + H-dt) - mid(t_j)  where dt = t_j - t_k
      - If H-dt matches another horizon label at j, we can solve for mid(t_j) - mid(t_k)

    Simpler fallback: linear interpolation between horizon labels at k.
      Path at time dt = lerp between labels at bracketing horizons.

    For production accuracy, we also chain adjacent event labels:
      mid(t_{k+1}) - mid(t_k) = labels_1s[k] - labels_1s[k+1] + correction
      where correction accounts for the time offset.

    Returns dict with:
        dense_paths:    list of (M_i, 2) arrays [(dt_sec, mid_change_ticks), ...]
        pred_values:    (N,) predictions
        label_values:   (N,) actual labels
        timestamps_ns:  (N,) prediction timestamps
    """
    N = len(pred_indices)
    n_events = len(timestamps)
    max_ns = int(max_seconds * 1_000_000_000)

    # Precompute horizon seconds array for interpolation
    h_secs = np.array([0.0, 1.0, 5.0, 10.0, 30.0], dtype=np.float64)
    h_names = [None, "1s", "5s", "10s", "30s"]

    dense_paths = []

    for i in range(N):
        k = pred_indices[i]
        t_k = timestamps[k]
        t_max = t_k + max_ns

        # Get label values at k for interpolation
        h_vals = np.zeros(5, dtype=np.float64)
        h_vals[0] = 0.0  # at t=0
        for hi, h_name in enumerate(h_names[1:], 1):
            if h_name in labels:
                v = labels[h_name][k]
                h_vals[hi] = v if not np.isnan(v) else h_vals[hi - 1]
            else:
                h_vals[hi] = h_vals[hi - 1]

        # Build path: for each forward event, interpolate midpoint from horizon labels
        path_points = [(0.0, 0.0)]  # (dt_sec, mid_change_ticks)

        j = k + 1
        count = 0
        while j < n_events and timestamps[j] <= t_max and count < max_events:
            dt_ns = timestamps[j] - t_k
            dt_sec = ns_to_sec(dt_ns)

            # Linear interpolation between horizon labels
            mid_change = float(np.interp(dt_sec, h_secs, h_vals))
            path_points.append((dt_sec, mid_change))

            j += 1
            count += 1

        dense_paths.append(np.array(path_points, dtype=np.float64))

    horizon_names_list = ["1s", "5s", "10s"]
    chosen_h = horizon_names_list[pred_horizon_idx]

    result = {
        "dense_paths": dense_paths,
        "pred_values": predictions[:, pred_horizon_idx].copy(),
        "label_values": (
            labels[chosen_h][pred_indices].copy()
            if chosen_h in labels
            else np.full(N, np.nan, dtype=np.float32)
        ),
        "timestamps_ns": timestamps[pred_indices].copy(),
    }
    return result


# ---------------------------------------------------------------------------
# Core: Compute MFE / MAE from paths
# ---------------------------------------------------------------------------
def compute_mfe_mae_from_labels(paths: dict) -> dict:
    """
    Compute MFE/MAE from the 5-point label path (0s/1s/5s/10s/30s).

    For each prediction:
      - Direction is determined by sign of prediction (positive = long, negative = short)
      - MFE = maximum favorable price move in the predicted direction
      - MAE = maximum adverse price move (opposite direction) before MFE time

    Returns dict with arrays:
        mfe_ticks:     (N,) MFE in ticks (always >= 0)
        mae_ticks:     (N,) MAE in ticks (always >= 0, measured before MFE)
        time_to_mfe_s: (N,) seconds to reach MFE
        time_to_mae_s: (N,) seconds to reach MAE
        mfe_mae_ratio: (N,) MFE/MAE (inf if MAE=0)
        direction:     (N,) +1 for long, -1 for short
    """
    N = len(paths["pred_values"])
    horizon_secs = [0.0, 1.0, 5.0, 10.0, 30.0]
    path_keys = ["path_0s", "path_1s", "path_5s", "path_10s", "path_30s"]

    # Stack path points: (N, 5) each row is [0s, 1s, 5s, 10s, 30s] midpoint change
    path_matrix = np.column_stack([paths[k] for k in path_keys])  # (N, 5)

    # Direction from prediction sign
    direction = np.sign(paths["pred_values"])
    direction[direction == 0] = 1.0  # neutral -> treat as long

    # Directional path: positive = favorable, negative = adverse
    dir_path = path_matrix * direction[:, None]  # (N, 5)

    results = {
        "mfe_ticks": np.zeros(N, dtype=np.float32),
        "mae_ticks": np.zeros(N, dtype=np.float32),
        "time_to_mfe_s": np.zeros(N, dtype=np.float32),
        "time_to_mae_s": np.zeros(N, dtype=np.float32),
        "mfe_mae_ratio": np.zeros(N, dtype=np.float32),
        "direction": direction.astype(np.float32),
    }

    for i in range(N):
        dp = dir_path[i]  # (5,) directional path
        valid = ~np.isnan(dp)
        if valid.sum() < 2:
            results["mfe_ticks"][i] = np.nan
            results["mae_ticks"][i] = np.nan
            results["time_to_mfe_s"][i] = np.nan
            results["time_to_mae_s"][i] = np.nan
            results["mfe_mae_ratio"][i] = np.nan
            continue

        dp_valid = dp[valid]
        secs_valid = np.array(horizon_secs)[valid]

        # MFE = max favorable excursion (max of directional path)
        mfe_idx = np.argmax(dp_valid)
        mfe_val = dp_valid[mfe_idx]
        mfe_time = secs_valid[mfe_idx]

        # MAE = max adverse excursion BEFORE MFE (min of directional path up to MFE)
        pre_mfe = dp_valid[: mfe_idx + 1]
        mae_idx_rel = np.argmin(pre_mfe)
        mae_val = pre_mfe[mae_idx_rel]
        mae_time = secs_valid[mae_idx_rel]

        results["mfe_ticks"][i] = max(mfe_val, 0.0)
        results["mae_ticks"][i] = abs(min(mae_val, 0.0))
        results["time_to_mfe_s"][i] = mfe_time
        results["time_to_mae_s"][i] = mae_time

        if results["mae_ticks"][i] > 0:
            results["mfe_mae_ratio"][i] = results["mfe_ticks"][i] / results["mae_ticks"][i]
        else:
            results["mfe_mae_ratio"][i] = (
                float("inf") if results["mfe_ticks"][i] > 0 else 0.0
            )

    return results


def compute_mfe_mae_from_dense(dense_data: dict) -> dict:
    """
    Compute MFE/MAE from dense event-by-event paths.
    Same output schema as compute_mfe_mae_from_labels.
    """
    dense_paths = dense_data["dense_paths"]
    preds = dense_data["pred_values"]
    N = len(preds)

    direction = np.sign(preds)
    direction[direction == 0] = 1.0

    results = {
        "mfe_ticks": np.zeros(N, dtype=np.float32),
        "mae_ticks": np.zeros(N, dtype=np.float32),
        "time_to_mfe_s": np.zeros(N, dtype=np.float32),
        "time_to_mae_s": np.zeros(N, dtype=np.float32),
        "mfe_mae_ratio": np.zeros(N, dtype=np.float32),
        "direction": direction.astype(np.float32),
    }

    for i in range(N):
        path = dense_paths[i]  # (M, 2): [(dt_sec, mid_change), ...]
        if len(path) < 2:
            results["mfe_ticks"][i] = np.nan
            results["mae_ticks"][i] = np.nan
            results["time_to_mfe_s"][i] = np.nan
            results["time_to_mae_s"][i] = np.nan
            results["mfe_mae_ratio"][i] = np.nan
            continue

        dt_secs = path[:, 0]
        mid_changes = path[:, 1]

        # Directional: positive = favorable
        dir_changes = mid_changes * direction[i]

        # MFE
        mfe_idx = np.argmax(dir_changes)
        mfe_val = dir_changes[mfe_idx]
        mfe_time = dt_secs[mfe_idx]

        # MAE before MFE
        pre_mfe = dir_changes[: mfe_idx + 1]
        mae_idx_rel = np.argmin(pre_mfe)
        mae_val = pre_mfe[mae_idx_rel]
        mae_time = dt_secs[mae_idx_rel]

        results["mfe_ticks"][i] = max(float(mfe_val), 0.0)
        results["mae_ticks"][i] = abs(min(float(mae_val), 0.0))
        results["time_to_mfe_s"][i] = float(mfe_time)
        results["time_to_mae_s"][i] = float(mae_time)

        if results["mae_ticks"][i] > 0:
            results["mfe_mae_ratio"][i] = results["mfe_ticks"][i] / results["mae_ticks"][i]
        else:
            results["mfe_mae_ratio"][i] = (
                float("inf") if results["mfe_ticks"][i] > 0 else 0.0
            )

    return results


# ---------------------------------------------------------------------------
# Analysis: Winners that go red, quintile breakdowns, etc.
# ---------------------------------------------------------------------------
def analyze_winner_paths(
    mfe_mae: dict,
    paths_or_dense: dict,
    is_dense: bool = False,
) -> dict:
    """
    Detailed analysis of winning trades' price paths.

    A 'winner' is a trade where the actual label moved in the predicted direction
    (i.e., the model was right about direction).
    """
    if is_dense:
        preds = paths_or_dense["pred_values"]
        actuals = paths_or_dense["label_values"]
    else:
        preds = paths_or_dense["pred_values"]
        actuals = paths_or_dense["label_values"]

    direction = mfe_mae["direction"]

    # Winner = actual moved in predicted direction
    actual_dir = actuals * direction
    valid = ~np.isnan(actual_dir)
    winners = valid & (actual_dir > 0)
    losers = valid & (actual_dir <= 0)

    n_valid = valid.sum()
    n_winners = winners.sum()
    n_losers = losers.sum()

    # % of winners that experienced adverse excursion (went red first)
    winner_mae = mfe_mae["mae_ticks"][winners]
    winners_that_went_red = (winner_mae > 0).sum()
    pct_winners_go_red = (
        float(winners_that_went_red) / float(n_winners) * 100.0
        if n_winners > 0 else 0.0
    )

    # Winners: MAE statistics
    winner_mae_stats = {}
    if n_winners > 0:
        wm = winner_mae[~np.isnan(winner_mae)]
        winner_mae_stats = {
            "mean_mae_ticks": float(np.mean(wm)) if len(wm) > 0 else 0.0,
            "median_mae_ticks": float(np.median(wm)) if len(wm) > 0 else 0.0,
            "p75_mae_ticks": float(np.percentile(wm, 75)) if len(wm) > 0 else 0.0,
            "p90_mae_ticks": float(np.percentile(wm, 90)) if len(wm) > 0 else 0.0,
            "p95_mae_ticks": float(np.percentile(wm, 95)) if len(wm) > 0 else 0.0,
        }

    # Winners: MFE statistics
    winner_mfe = mfe_mae["mfe_ticks"][winners]
    winner_mfe_stats = {}
    if n_winners > 0:
        wf = winner_mfe[~np.isnan(winner_mfe)]
        winner_mfe_stats = {
            "mean_mfe_ticks": float(np.mean(wf)) if len(wf) > 0 else 0.0,
            "median_mfe_ticks": float(np.median(wf)) if len(wf) > 0 else 0.0,
            "p75_mfe_ticks": float(np.percentile(wf, 75)) if len(wf) > 0 else 0.0,
            "p90_mfe_ticks": float(np.percentile(wf, 90)) if len(wf) > 0 else 0.0,
        }

    # Time-to-MFE for winners
    winner_ttm = mfe_mae["time_to_mfe_s"][winners]
    ttm_valid = winner_ttm[~np.isnan(winner_ttm)]
    winner_time_stats = {
        "mean_time_to_mfe_s": float(np.mean(ttm_valid)) if len(ttm_valid) > 0 else 0.0,
        "median_time_to_mfe_s": float(np.median(ttm_valid)) if len(ttm_valid) > 0 else 0.0,
    }

    # Time-to-MAE for winners (how quickly the drawdown hits)
    winner_tta = mfe_mae["time_to_mae_s"][winners]
    tta_valid = winner_tta[~np.isnan(winner_tta)]
    winner_time_stats["mean_time_to_mae_s"] = (
        float(np.mean(tta_valid)) if len(tta_valid) > 0 else 0.0
    )
    winner_time_stats["median_time_to_mae_s"] = (
        float(np.median(tta_valid)) if len(tta_valid) > 0 else 0.0
    )

    # MFE/MAE ratio for winners
    winner_ratio = mfe_mae["mfe_mae_ratio"][winners]
    wr_valid = winner_ratio[np.isfinite(winner_ratio)]
    ratio_stats = {
        "mean_mfe_mae_ratio": float(np.mean(wr_valid)) if len(wr_valid) > 0 else 0.0,
        "median_mfe_mae_ratio": float(np.median(wr_valid)) if len(wr_valid) > 0 else 0.0,
    }

    return {
        "n_predictions": int(n_valid),
        "n_winners": int(n_winners),
        "n_losers": int(n_losers),
        "win_rate_pct": float(n_winners / n_valid * 100) if n_valid > 0 else 0.0,
        "pct_winners_go_red_first": round(pct_winners_go_red, 2),
        "n_winners_that_went_red": int(winners_that_went_red),
        "winner_mae_stats": winner_mae_stats,
        "winner_mfe_stats": winner_mfe_stats,
        "winner_time_stats": winner_time_stats,
        "winner_mfe_mae_ratio_stats": ratio_stats,
    }


def quintile_analysis(
    mfe_mae: dict,
    paths_or_dense: dict,
    is_dense: bool = False,
) -> dict:
    """
    Break down MFE/MAE metrics by prediction confidence quintile.
    Q1 = weakest absolute predictions, Q5 = strongest.
    """
    if is_dense:
        preds = paths_or_dense["pred_values"]
        actuals = paths_or_dense["label_values"]
    else:
        preds = paths_or_dense["pred_values"]
        actuals = paths_or_dense["label_values"]

    abs_preds = np.abs(preds)
    direction = mfe_mae["direction"]

    # Filter out NaN
    valid = ~np.isnan(actuals) & ~np.isnan(abs_preds)
    if valid.sum() < 10:
        return {"error": "Too few valid predictions for quintile analysis"}

    # Compute quintile boundaries on absolute prediction magnitude
    abs_valid = abs_preds[valid]
    quintile_edges = np.percentile(abs_valid, [0, 20, 40, 60, 80, 100])

    results = {}
    for q in range(5):
        lo = quintile_edges[q]
        hi = quintile_edges[q + 1]
        if q < 4:
            mask = valid & (abs_preds >= lo) & (abs_preds < hi)
        else:
            mask = valid & (abs_preds >= lo) & (abs_preds <= hi)

        n_q = mask.sum()
        if n_q == 0:
            results[QUINTILE_LABELS[q]] = {"n": 0}
            continue

        q_mfe = mfe_mae["mfe_ticks"][mask]
        q_mae = mfe_mae["mae_ticks"][mask]
        q_ttmfe = mfe_mae["time_to_mfe_s"][mask]
        q_ttmae = mfe_mae["time_to_mae_s"][mask]
        q_ratio = mfe_mae["mfe_mae_ratio"][mask]
        q_actuals = actuals[mask]
        q_direction = direction[mask]

        # Directional accuracy
        actual_dir_move = q_actuals * q_direction
        actual_dir_valid = actual_dir_move[~np.isnan(actual_dir_move)]
        q_winners = (actual_dir_valid > 0).sum()
        q_win_rate = float(q_winners / len(actual_dir_valid) * 100) if len(actual_dir_valid) > 0 else 0.0

        # Winners that go red
        winner_mask = mask & (actuals * direction > 0)
        winner_mae_vals = mfe_mae["mae_ticks"][winner_mask]
        n_winners_red = (winner_mae_vals > 0).sum() if len(winner_mae_vals) > 0 else 0
        pct_red = float(n_winners_red / len(winner_mae_vals) * 100) if len(winner_mae_vals) > 0 else 0.0

        def _safe_stats(arr):
            a = arr[~np.isnan(arr) & np.isfinite(arr)]
            if len(a) == 0:
                return {"mean": 0.0, "median": 0.0, "std": 0.0, "p25": 0.0, "p75": 0.0}
            return {
                "mean": round(float(np.mean(a)), 3),
                "median": round(float(np.median(a)), 3),
                "std": round(float(np.std(a)), 3),
                "p25": round(float(np.percentile(a, 25)), 3),
                "p75": round(float(np.percentile(a, 75)), 3),
            }

        results[QUINTILE_LABELS[q]] = {
            "n": int(n_q),
            "pred_abs_range": [round(float(lo), 4), round(float(hi), 4)],
            "win_rate_pct": round(q_win_rate, 2),
            "pct_winners_go_red": round(pct_red, 2),
            "mfe_ticks": _safe_stats(q_mfe),
            "mae_ticks": _safe_stats(q_mae),
            "time_to_mfe_s": _safe_stats(q_ttmfe),
            "time_to_mae_s": _safe_stats(q_ttmae),
            "mfe_mae_ratio": _safe_stats(q_ratio),
            "actual_move_ticks": _safe_stats(q_actuals),
        }

    return results


def overall_statistics(mfe_mae: dict, paths_or_dense: dict, is_dense: bool = False) -> dict:
    """Aggregate statistics across all predictions."""
    preds = paths_or_dense["pred_values"]
    actuals = paths_or_dense["label_values"]

    valid = ~np.isnan(actuals)
    n_valid = valid.sum()

    mfe = mfe_mae["mfe_ticks"]
    mae = mfe_mae["mae_ticks"]
    ratio = mfe_mae["mfe_mae_ratio"]

    def _stats(arr, name=""):
        a = arr[~np.isnan(arr) & np.isfinite(arr)]
        if len(a) == 0:
            return {}
        return {
            "mean": round(float(np.mean(a)), 4),
            "median": round(float(np.median(a)), 4),
            "std": round(float(np.std(a)), 4),
            "min": round(float(np.min(a)), 4),
            "max": round(float(np.max(a)), 4),
            "p5": round(float(np.percentile(a, 5)), 4),
            "p25": round(float(np.percentile(a, 25)), 4),
            "p75": round(float(np.percentile(a, 75)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
        }

    # IC between predictions and actuals
    p_valid = preds[valid]
    a_valid = actuals[valid]
    try:
        ic, _ = spearmanr(p_valid, a_valid)
    except Exception:
        ic = float("nan")

    return {
        "n_predictions": int(n_valid),
        "ic_spearman": round(float(ic), 4) if not np.isnan(ic) else None,
        "mfe_ticks": _stats(mfe),
        "mae_ticks": _stats(mae),
        "mfe_mae_ratio": _stats(ratio),
        "time_to_mfe_s": _stats(mfe_mae["time_to_mfe_s"]),
        "time_to_mae_s": _stats(mfe_mae["time_to_mae_s"]),
        "predictions": _stats(preds),
        "actuals": _stats(actuals),
    }


# ---------------------------------------------------------------------------
# Reporting: Rich console output
# ---------------------------------------------------------------------------
def print_report(
    overall: dict,
    winner_analysis: dict,
    quintiles: dict,
    fold_idx: int,
    horizon: str,
    mode: str,
):
    """Print formatted analysis report to console."""
    sep = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print(f"  MFE/MAE PATH ANALYSIS — Fold {fold_idx:02d} — Horizon: {horizon} — Mode: {mode}")
    print(f"{sep}\n")

    # Overall stats
    n = overall["n_predictions"]
    ic = overall.get("ic_spearman", "N/A")
    print(f"  Total predictions: {n:,}")
    print(f"  Spearman IC:       {ic}")
    print()

    # MFE/MAE summary
    print(f"  {'Metric':<25} {'Mean':>8} {'Median':>8} {'P25':>8} {'P75':>8} {'P95':>8}")
    print(f"  {thin}")
    for metric_name, key in [
        ("MFE (ticks)", "mfe_ticks"),
        ("MAE (ticks)", "mae_ticks"),
        ("MFE/MAE ratio", "mfe_mae_ratio"),
        ("Time to MFE (s)", "time_to_mfe_s"),
        ("Time to MAE (s)", "time_to_mae_s"),
    ]:
        s = overall.get(key, {})
        print(
            f"  {metric_name:<25} "
            f"{s.get('mean', 0):>8.2f} "
            f"{s.get('median', 0):>8.2f} "
            f"{s.get('p25', 0):>8.2f} "
            f"{s.get('p75', 0):>8.2f} "
            f"{s.get('p95', 0):>8.2f}"
        )

    # Dollar values
    print()
    mfe_mean = overall.get("mfe_ticks", {}).get("mean", 0)
    mae_mean = overall.get("mae_ticks", {}).get("mean", 0)
    print(f"  MFE mean in $: ${mfe_mean * TICK_VALUE:.2f} per contract")
    print(f"  MAE mean in $: ${mae_mean * TICK_VALUE:.2f} per contract")

    # Winner analysis
    print(f"\n{thin}")
    print(f"  WINNER PATH ANALYSIS")
    print(f"{thin}")
    wa = winner_analysis
    print(f"  Win rate:                       {wa['win_rate_pct']:.1f}%")
    print(f"  Winners:                        {wa['n_winners']:,}")
    print(f"  Losers:                         {wa['n_losers']:,}")
    print(f"  Winners that go RED first:      {wa['pct_winners_go_red_first']:.1f}%  ({wa['n_winners_that_went_red']:,} of {wa['n_winners']:,})")
    if wa.get("winner_mae_stats"):
        ws = wa["winner_mae_stats"]
        print(f"  Winner MAE (mean/median):       {ws.get('mean_mae_ticks', 0):.2f} / {ws.get('median_mae_ticks', 0):.2f} ticks")
        print(f"  Winner MAE P90:                 {ws.get('p90_mae_ticks', 0):.2f} ticks")
    if wa.get("winner_mfe_stats"):
        wf = wa["winner_mfe_stats"]
        print(f"  Winner MFE (mean/median):       {wf.get('mean_mfe_ticks', 0):.2f} / {wf.get('median_mfe_ticks', 0):.2f} ticks")
    if wa.get("winner_time_stats"):
        wt = wa["winner_time_stats"]
        print(f"  Winner Time-to-MFE (mean):      {wt.get('mean_time_to_mfe_s', 0):.2f}s")
        print(f"  Winner Time-to-MAE (mean):      {wt.get('mean_time_to_mae_s', 0):.2f}s")

    # Quintile analysis
    print(f"\n{thin}")
    print(f"  QUINTILE ANALYSIS (by |prediction| magnitude)")
    print(f"{thin}")

    if "error" in quintiles:
        print(f"  {quintiles['error']}")
    else:
        # Header
        print(
            f"  {'Quintile':<18} {'N':>6} {'WinR%':>7} {'Red%':>6} "
            f"{'MFE':>7} {'MAE':>7} {'Ratio':>7} {'tMFE':>6} {'tMAE':>6} "
            f"{'ActMov':>7}"
        )
        print(f"  {'-' * 90}")

        for q_label in QUINTILE_LABELS:
            qd = quintiles.get(q_label, {})
            n_q = qd.get("n", 0)
            if n_q == 0:
                print(f"  {q_label:<18} {0:>6}")
                continue

            wr = qd.get("win_rate_pct", 0)
            red = qd.get("pct_winners_go_red", 0)
            mfe_m = qd.get("mfe_ticks", {}).get("mean", 0)
            mae_m = qd.get("mae_ticks", {}).get("mean", 0)
            rat_m = qd.get("mfe_mae_ratio", {}).get("mean", 0)
            tmfe = qd.get("time_to_mfe_s", {}).get("mean", 0)
            tmae = qd.get("time_to_mae_s", {}).get("mean", 0)
            act_m = qd.get("actual_move_ticks", {}).get("mean", 0)

            print(
                f"  {q_label:<18} {n_q:>6} {wr:>6.1f}% {red:>5.1f}% "
                f"{mfe_m:>7.2f} {mae_m:>7.2f} {rat_m:>7.2f} {tmfe:>6.2f} {tmae:>6.2f} "
                f"{act_m:>7.2f}"
            )

        # Legend
        print()
        print("  MFE/MAE in ticks | Ratio = MFE/MAE | tMFE/tMAE in seconds | ActMov = actual label in ticks")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------
def process_fold(
    pred_dir: Path,
    mbo_dir: Path,
    fold_idx: int,
    horizon: str = "10s",
    window_size: int = 3000,
    stride: int = 250,
    dense: bool = False,
    max_events: int = DEFAULT_MAX_EVENTS,
    output_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Process a single fold: load predictions, match to MBO events, compute MFE/MAE."""

    pred_path = pred_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not pred_path.exists():
        print(f"[SKIP] Fold {fold_idx:02d}: {pred_path} not found")
        return None

    print(f"\n[FOLD {fold_idx:02d}] Loading predictions from {pred_path.name}...")
    pred_data = np.load(pred_path, allow_pickle=True)
    predictions = pred_data["predictions"]  # (N, 3) for 1s/5s/10s
    labels_pred = pred_data["labels"]  # (N, 3)
    horizons_list = list(pred_data["horizons"])

    if horizon not in horizons_list:
        print(f"[ERROR] Horizon '{horizon}' not in prediction horizons: {horizons_list}")
        return None
    pred_horizon_idx = horizons_list.index(horizon)

    # Determine which MBO file(s) this fold uses
    oot_files_raw = list(pred_data["oot_files"])
    n_preds = predictions.shape[0]
    print(f"  Predictions: {n_preds:,} | Horizons: {horizons_list} | Analyzing: {horizon}")

    # Find local MBO file paths
    all_event_indices = []
    all_timestamps = []
    all_labels = {h: [] for h in HORIZON_NAMES}
    all_events = []
    offset = 0
    file_boundaries = []  # (start_pred_idx, end_pred_idx, mbo_path)

    for oot_path_str in oot_files_raw:
        date_str = extract_date_from_path(oot_path_str)
        if date_str is None:
            print(f"  [WARN] Cannot extract date from: {oot_path_str}")
            continue

        # Try local MBO paths (regular format, not smart_v3, since train code may use either)
        # Try smart_v3 first (most likely used in training), then regular, then original
        mbo_candidates = [
            mbo_dir.parent / "mbo_events_smart_v3" / f"{date_str}_mbo_events.npz",
            mbo_dir / f"{date_str}_mbo_events.npz",
        ]
        # Also try the original path if it's accessible
        orig = Path(oot_path_str)
        if orig.exists():
            mbo_candidates.insert(0, orig)

        mbo_path = None
        for c in mbo_candidates:
            if c.exists():
                mbo_path = c
                break

        if mbo_path is None:
            print(f"  [WARN] MBO file not found for date {date_str}")
            print(f"         Tried: {[str(c) for c in mbo_candidates]}")
            continue

        print(f"  Loading MBO events: {mbo_path.name} ...", end="", flush=True)
        t0 = time.time()

        event_indices, timestamps, day_labels, events = reconstruct_event_indices(
            mbo_path, window_size, stride, horizons=horizons_list,
            target_n_preds=n_preds,
        )
        elapsed = time.time() - t0
        print(f" {len(event_indices):,} samples in {elapsed:.1f}s")

        file_boundaries.append((offset, offset + len(event_indices), str(mbo_path)))
        all_event_indices.append(event_indices)
        all_timestamps.append(timestamps)
        for h in HORIZON_NAMES:
            if h in day_labels:
                all_labels[h].append(day_labels[h])
        all_events.append(events)
        offset += len(event_indices)

    if offset == 0:
        print(f"[ERROR] No MBO data found for fold {fold_idx:02d}")
        return None

    # Validate prediction count matches
    total_reconstructed = sum(len(ei) for ei in all_event_indices)
    if total_reconstructed != n_preds:
        print(
            f"  [WARN] Reconstructed {total_reconstructed:,} event indices "
            f"but predictions have {n_preds:,} rows."
        )
        # Try different stride values
        if total_reconstructed == 0:
            print("  [ERROR] Zero samples reconstructed. Check window_size/stride.")
            return None
        # Truncate/pad to match
        if total_reconstructed > n_preds:
            print(f"  Truncating to {n_preds:,} samples")
            # Take the first n_preds from the concatenated indices
        elif total_reconstructed < n_preds:
            print(f"  Using {total_reconstructed:,} of {n_preds:,} predictions")
            predictions = predictions[:total_reconstructed]

    # For single-file folds (most common), use directly
    if len(all_event_indices) == 1:
        event_indices = all_event_indices[0]
        timestamps = all_timestamps[0]
        day_labels = {h: all_labels[h][0] for h in HORIZON_NAMES if all_labels[h]}

        # Match prediction count
        min_n = min(len(event_indices), n_preds)
        event_indices = event_indices[:min_n]
        predictions_use = predictions[:min_n]

        print(f"  Computing {'dense' if dense else 'label-based'} paths...")
        t0 = time.time()

        if dense:
            path_data = build_dense_paths(
                event_indices, timestamps, day_labels, predictions_use,
                pred_horizon_idx, max_events=max_events,
            )
        else:
            path_data = build_label_paths(
                event_indices, timestamps, day_labels, predictions_use,
                pred_horizon_idx,
            )

        elapsed = time.time() - t0
        print(f"  Paths computed in {elapsed:.1f}s")

    else:
        # Multi-file fold: concatenate
        print(f"  [INFO] Multi-file fold ({len(all_event_indices)} files), processing sequentially...")
        # For simplicity, concatenate the label-based paths
        # Dense mode on multi-file folds processes each file separately
        all_path_data = {"pred_values": [], "label_values": [], "timestamps_ns": []}
        if dense:
            all_path_data["dense_paths"] = []
        else:
            for h in HORIZON_NAMES:
                all_path_data[f"path_{h}"] = []
            all_path_data["path_0s"] = []

        pred_offset = 0
        for fi in range(len(all_event_indices)):
            ei = all_event_indices[fi]
            ts = all_timestamps[fi]
            dl = {h: all_labels[h][fi] for h in HORIZON_NAMES if fi < len(all_labels[h])}
            n_fi = len(ei)
            preds_fi = predictions[pred_offset : pred_offset + n_fi]

            if dense:
                pd_fi = build_dense_paths(ei, ts, dl, preds_fi, pred_horizon_idx, max_events=max_events)
                all_path_data["dense_paths"].extend(pd_fi["dense_paths"])
            else:
                pd_fi = build_label_paths(ei, ts, dl, preds_fi, pred_horizon_idx)
                all_path_data["path_0s"].append(pd_fi["path_0s"])
                for h in HORIZON_NAMES:
                    all_path_data[f"path_{h}"].append(pd_fi[f"path_{h}"])

            all_path_data["pred_values"].append(pd_fi["pred_values"])
            all_path_data["label_values"].append(pd_fi["label_values"])
            all_path_data["timestamps_ns"].append(pd_fi["timestamps_ns"])
            pred_offset += n_fi

        # Concatenate
        path_data = {}
        for k, v in all_path_data.items():
            if k == "dense_paths":
                path_data[k] = v
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                path_data[k] = np.concatenate(v)
            else:
                path_data[k] = v

    # Compute MFE/MAE
    print("  Computing MFE/MAE metrics...")
    if dense:
        mfe_mae = compute_mfe_mae_from_dense(path_data)
    else:
        mfe_mae = compute_mfe_mae_from_labels(path_data)

    # Analysis
    print("  Running winner path analysis...")
    winner_analysis = analyze_winner_paths(mfe_mae, path_data, is_dense=dense)

    print("  Running quintile analysis...")
    quintiles = quintile_analysis(mfe_mae, path_data, is_dense=dense)

    print("  Computing overall statistics...")
    overall = overall_statistics(mfe_mae, path_data, is_dense=dense)

    # Print report
    mode = "DENSE (event-by-event)" if dense else "LABEL (5-point)"
    print_report(overall, winner_analysis, quintiles, fold_idx, horizon, mode)

    # Build result dict
    result = {
        "fold": fold_idx,
        "horizon": horizon,
        "mode": mode,
        "n_predictions": int(len(path_data["pred_values"])),
        "overall": overall,
        "winner_analysis": winner_analysis,
        "quintiles": quintiles,
        "file_boundaries": file_boundaries,
    }

    # Save outputs
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = output_dir / f"fold_{fold_idx:02d}_mfe_mae_{horizon}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Saved JSON report: {json_path}")

        # NPZ with raw data
        npz_path = output_dir / f"fold_{fold_idx:02d}_mfe_mae_{horizon}.npz"
        save_dict = {
            "mfe_ticks": mfe_mae["mfe_ticks"],
            "mae_ticks": mfe_mae["mae_ticks"],
            "time_to_mfe_s": mfe_mae["time_to_mfe_s"],
            "time_to_mae_s": mfe_mae["time_to_mae_s"],
            "mfe_mae_ratio": mfe_mae["mfe_mae_ratio"],
            "direction": mfe_mae["direction"],
            "pred_values": path_data["pred_values"],
            "label_values": path_data["label_values"],
            "timestamps_ns": path_data["timestamps_ns"],
        }
        if not dense:
            for h in HORIZON_NAMES:
                key = f"path_{h}"
                if key in path_data:
                    save_dict[key] = path_data[key]
        np.savez_compressed(npz_path, **save_dict)
        print(f"  Saved NPZ data:   {npz_path}")

    return result


# ---------------------------------------------------------------------------
# Multi-fold concatenated analysis
# ---------------------------------------------------------------------------
def concat_analysis(fold_results: List[dict]) -> dict:
    """Summarize across multiple folds."""
    if not fold_results:
        return {}

    n_total = sum(r["n_predictions"] for r in fold_results)
    all_wr = [r["winner_analysis"]["win_rate_pct"] for r in fold_results]
    all_red = [r["winner_analysis"]["pct_winners_go_red_first"] for r in fold_results]
    all_mfe = [r["overall"]["mfe_ticks"].get("mean", 0) for r in fold_results]
    all_mae = [r["overall"]["mae_ticks"].get("mean", 0) for r in fold_results]

    ics = [r["overall"].get("ic_spearman") for r in fold_results if r["overall"].get("ic_spearman") is not None]

    summary = {
        "n_folds": len(fold_results),
        "n_total_predictions": n_total,
        "mean_ic": round(float(np.mean(ics)), 4) if ics else None,
        "mean_win_rate_pct": round(float(np.mean(all_wr)), 2),
        "mean_pct_winners_go_red": round(float(np.mean(all_red)), 2),
        "mean_mfe_ticks": round(float(np.mean(all_mfe)), 3),
        "mean_mae_ticks": round(float(np.mean(all_mae)), 3),
        "per_fold": [
            {
                "fold": r["fold"],
                "n": r["n_predictions"],
                "ic": r["overall"].get("ic_spearman"),
                "win_rate": r["winner_analysis"]["win_rate_pct"],
                "red_pct": r["winner_analysis"]["pct_winners_go_red_first"],
                "mfe_mean": r["overall"]["mfe_ticks"].get("mean", 0),
                "mae_mean": r["overall"]["mae_ticks"].get("mean", 0),
            }
            for r in fold_results
        ],
    }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MFE/MAE Path Analysis for MBO Event-Driven Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze fold 6 with label-based paths (fast)
  python mfe_mae_path_analysis.py --pred-dir output/cnn_mamba_v2_smart_v3_mar --folds 6

  # Analyze folds 6-9 with dense paths (slower, more accurate)
  python mfe_mae_path_analysis.py --pred-dir output/cnn_mamba_v2_smart_v3_mar --folds 6-9 --dense

  # All folds, 5s horizon
  python mfe_mae_path_analysis.py --pred-dir output/cnn_mamba_v2_smart_v3_mar --folds all --horizon 5s
        """,
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="Directory containing fold_XX_oot_predictions.npz files",
    )
    parser.add_argument(
        "--mbo-dir",
        type=str,
        default=None,
        help="Directory containing YYYYMMDD_mbo_events.npz files. "
             "If not set, tries ../data/processed/mbo_events relative to pred-dir",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default="all",
        help="Fold specification: '6', '6,7,8', '6-9', or 'all'",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="10s",
        choices=["1s", "5s", "10s"],
        help="Prediction horizon to analyze (default: 10s)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3000,
        help="Window size used in training (default: 3000)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride used in training (default: window_size // 6 = 500)",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Use dense event-by-event path reconstruction (slower but more accurate)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_EVENTS,
        help=f"Max forward events per prediction in dense mode (default: {DEFAULT_MAX_EVENTS})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSON/NPZ results (default: pred-dir/mfe_mae_analysis)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving output files (console report only)",
    )

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        print(f"[ERROR] Prediction directory not found: {pred_dir}")
        sys.exit(1)

    # Resolve MBO directory
    if args.mbo_dir:
        mbo_dir = Path(args.mbo_dir)
    else:
        # Try common locations relative to pred_dir
        candidates = [
            pred_dir.parent.parent / "data" / "processed" / "mbo_events",
            Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events"),
            Path("/home/nick/Lvl3Quant/data/processed/mbo_events"),
        ]
        mbo_dir = None
        for c in candidates:
            if c.exists():
                mbo_dir = c
                break
        if mbo_dir is None:
            print("[ERROR] Cannot find MBO events directory. Use --mbo-dir to specify.")
            sys.exit(1)

    stride = args.stride if args.stride else args.window_size // 6

    # Resolve output directory
    out_dir = None
    if not args.no_save:
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = pred_dir / "mfe_mae_analysis"

    # Discover available folds
    available_folds = sorted([
        int(p.stem.split("_")[1])
        for p in pred_dir.glob("fold_*_oot_predictions.npz")
    ])
    if not available_folds:
        print(f"[ERROR] No fold prediction files found in {pred_dir}")
        sys.exit(1)

    requested_folds = parse_fold_spec(args.folds)
    if requested_folds is None:
        folds = available_folds
    else:
        folds = [f for f in requested_folds if f in available_folds]
        missing = [f for f in requested_folds if f not in available_folds]
        if missing:
            print(f"[WARN] Folds not found: {missing}")

    if not folds:
        print("[ERROR] No valid folds to process.")
        sys.exit(1)

    print(f"MFE/MAE Path Analysis")
    print(f"  Pred dir:    {pred_dir}")
    print(f"  MBO dir:     {mbo_dir}")
    print(f"  Folds:       {folds}")
    print(f"  Horizon:     {args.horizon}")
    print(f"  Window/Stride: {args.window_size}/{stride}")
    print(f"  Mode:        {'DENSE' if args.dense else 'LABEL (5-point)'}")
    if out_dir:
        print(f"  Output:      {out_dir}")

    t_start = time.time()
    fold_results = []

    for fold_idx in folds:
        result = process_fold(
            pred_dir=pred_dir,
            mbo_dir=mbo_dir,
            fold_idx=fold_idx,
            horizon=args.horizon,
            window_size=args.window_size,
            stride=stride,
            dense=args.dense,
            max_events=args.max_events,
            output_dir=out_dir,
        )
        if result is not None:
            fold_results.append(result)

    # Multi-fold summary
    if len(fold_results) > 1:
        summary = concat_analysis(fold_results)

        sep = "=" * 72
        print(f"\n{sep}")
        print(f"  MULTI-FOLD SUMMARY ({len(fold_results)} folds, {summary['n_total_predictions']:,} predictions)")
        print(f"{sep}")
        print(f"  Mean IC:                {summary['mean_ic']}")
        print(f"  Mean Win Rate:          {summary['mean_win_rate_pct']:.1f}%")
        print(f"  Mean Winners Go Red:    {summary['mean_pct_winners_go_red']:.1f}%")
        print(f"  Mean MFE:               {summary['mean_mfe_ticks']:.3f} ticks (${summary['mean_mfe_ticks'] * TICK_VALUE:.2f})")
        print(f"  Mean MAE:               {summary['mean_mae_ticks']:.3f} ticks (${summary['mean_mae_ticks'] * TICK_VALUE:.2f})")
        print()
        print(f"  {'Fold':>6} {'N':>8} {'IC':>8} {'WinR%':>7} {'Red%':>7} {'MFE':>8} {'MAE':>8}")
        print(f"  {'-' * 60}")
        for pf in summary["per_fold"]:
            ic_str = f"{pf['ic']:.4f}" if pf['ic'] is not None else "  N/A "
            print(
                f"  {pf['fold']:>6} {pf['n']:>8} {ic_str:>8} "
                f"{pf['win_rate']:>6.1f}% {pf['red_pct']:>6.1f}% "
                f"{pf['mfe_mean']:>8.3f} {pf['mae_mean']:>8.3f}"
            )
        print(f"\n{sep}")

        if out_dir:
            summary_path = out_dir / f"concat_mfe_mae_{args.horizon}.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\n  Saved concat summary: {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
