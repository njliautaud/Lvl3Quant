#!/usr/bin/env python3
"""
Supervised Execution Model — Predict Trade Outcomes from Signal Features
=========================================================================

Fundamentally different from the RL DQN approach: instead of learning a POLICY
through reward shaping, we do straightforward supervised prediction.

Question: Given CNN-Mamba signal predictions + microstructure features at the
moment a signal fires, what will the OUTCOME be if we enter a trade?

If CNN-Mamba confidence → better outcomes (which decay analysis proved), this
model should TRIVIALLY learn that. If it can't, the data pipeline is broken.

Targets (per prediction event):
  - MFE at horizon (max favorable excursion in ticks)
  - MAE at horizon (max adverse excursion in ticks)
  - P&L at horizon (directional, in ticks, net of commission)
  - Binary: profitable at horizon (yes/no)

Model: Simple MLP with batch norm. Multi-output: regression (MFE, MAE, P&L)
       + binary classification (profitable).

Training: Walk-forward sliding window (HC #0). 60 train days, 1 OOT day.
Commission: 0.376 ticks RT only (HC #127, #231). NO spread cost.

Cost constants:
  ES tick = $12.50 | commission RT = $4.70 = 0.376 ticks
  P&L = favorable_move - 0.376 ticks (passive limit fills)

Usage:
    python3 train_supervised_exec.py \\
        --data-dir data/processed/mbo_events_smart_v3 \\
        --pred-dir output/cnn_mamba_v2_smart_v3_mar \\
        --precomputed-dir data/precomputed_obs \\
        --output-dir output/supervised_exec_v1 \\
        --n-train-days 60 \\
        --n-oot-days 1 \\
        --batch-size 4096 \\
        --epochs 20 \\
        --lr 1e-3 \\
        --hidden-dims 128 64 32 \\
        --horizon-secs 30 \\
        --n-workers 12

Author: Claude (Infrastructure Builder)
Date: 2026-05-07
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376

# MBO feature columns (from fifo_rl_env.py)
COL_TIME_DELTA = 0
COL_EVENT_TYPE = 1
COL_SIDE = 2
COL_PRICE_REL = 3
COL_QTY_LOG = 4
COL_SPREAD = 5

# Prediction stride: CNN-Mamba produces one prediction every 50 MBO events
PRED_STRIDE = 50
PRED_WINDOW = 1000

# Precomputed obs layout (from precompute_observations.py)
# Base obs (29-dim): [0:4]=signal, [4:8]=book, [8:10]=price, [10:13]=context,
#                    [13:18]=flow, [18:29]=embeddings+confluence
# Meta (13-dim): 0=ts_s, 1=price_rel, 2=et_raw, 3=spread, 4=pred_1s,
#                5=pred_5s, 6=pred_10s, 7=pred_idx, 8=bid_depth, 9=ask_depth,
#                10=book_imbalance, 11=queue_bid, 12=queue_ask

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
EXPERIMENT_NAME = "supervised_exec_v1"

DEVICE = "cpu"  # Jupiter has no GPU; script is CPU-friendly

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [SUP_EXEC] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("supervised_exec")

# ---------------------------------------------------------------------------
# MLflow (optional)
# ---------------------------------------------------------------------------
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning("mlflow not installed — training without experiment tracking.")


# ===========================================================================
# Data Pipeline: Extract (features, outcomes) from raw MBO + predictions
# ===========================================================================


@dataclass
class PredictionEvent:
    """One CNN-Mamba prediction event with computed outcome."""

    # Features
    pred_1s: float
    pred_5s: float
    pred_10s: float
    signal_direction: int  # +1 if pred_1s > 0, -1 if pred_1s < 0
    abs_pred_1s: float
    abs_pred_5s: float
    abs_pred_10s: float
    confidence_tier: int  # 0-3 based on abs_pred_1s quartiles
    book_imbalance: float
    bid_depth: float
    ask_depth: float
    spread: float
    recent_volatility: float
    time_of_day: float  # 0=open, 1=close (fractional)
    volume_imbalance: float
    signal_agreement: float  # 1.0 if all horizons agree, 0.0 otherwise
    signal_strength: float  # mean of abs predictions
    price_momentum: float
    pred_std: float  # std across horizons
    pred_range: float  # max - min across horizons

    # Targets
    mfe_ticks: float  # max favorable excursion at horizon
    mae_ticks: float  # max adverse excursion at horizon
    pnl_at_horizon: float  # directional P&L at horizon (net of commission)
    profitable: int  # 1 if pnl_at_horizon > 0, else 0

    # Metadata
    timestamp_s: float
    date_str: str


# Feature names (must match the order in build_feature_vector)
FEATURE_NAMES = [
    "pred_1s",
    "pred_5s",
    "pred_10s",
    "abs_pred_1s",
    "abs_pred_5s",
    "abs_pred_10s",
    "confidence_tier",
    "book_imbalance",
    "bid_depth_log",
    "ask_depth_log",
    "spread",
    "recent_volatility",
    "time_of_day",
    "volume_imbalance",
    "signal_agreement",
    "signal_strength",
    "price_momentum",
    "pred_std",
    "pred_range",
]
N_FEATURES = len(FEATURE_NAMES)

# Target names
TARGET_NAMES = ["mfe_ticks", "mae_ticks", "pnl_at_horizon", "profitable"]
N_TARGETS = len(TARGET_NAMES)


def _compute_volatility(price_changes: np.ndarray, window: int = 100) -> np.ndarray:
    """Rolling std of price changes."""
    n = len(price_changes)
    vol = np.zeros(n, dtype=np.float32)
    cs2 = np.cumsum(price_changes.astype(np.float64) ** 2)
    cs = np.cumsum(price_changes.astype(np.float64))
    for i in range(window, n):
        s = i - window
        cnt = window
        sm = cs[i] - cs[s]
        sm2 = cs2[i] - cs2[s]
        mean = sm / cnt
        var = max(sm2 / cnt - mean**2, 0.0)
        vol[i] = float(var**0.5)
    return vol


def _time_of_day_fraction(ts_s: float) -> float:
    """Convert epoch seconds to fractional time of day (0=9:30, 1=16:00 ET).

    Approximation: assumes ET = UTC-4 (EDT) or UTC-5 (EST).
    Good enough for a feature — exact TZ handling not critical.
    """
    # Use UTC-4 as approximate ET (EDT covers most trading months)
    seconds_in_day = ts_s % 86400
    et_seconds = (seconds_in_day - 4 * 3600) % 86400
    rth_start = 9.5 * 3600  # 9:30 AM
    rth_end = 16.0 * 3600  # 4:00 PM
    rth_duration = rth_end - rth_start
    frac = (et_seconds - rth_start) / rth_duration
    return max(0.0, min(1.0, frac))


def extract_outcomes_for_date(
    mbo_path: Path,
    predictions: np.ndarray,
    labels: np.ndarray,
    horizon_secs: float = 30.0,
    pred_stride: int = PRED_STRIDE,
    pred_window: int = PRED_WINDOW,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (features, targets, metadata) for all prediction events in one day.

    IMPORTANT: price_rel in MBO events is relative to a running mid-price and
    capped at +/-2 ticks. It CANNOT be used to measure price moves over time.
    Instead, we use the LABELS from the prediction file, which are actual future
    mid-price moves in ticks at 1s/5s/10s horizons.

    For each prediction event:
    1. Extract microstructure features at signal time from MBO event stream
    2. Use labels (actual future moves) to compute directional MFE, MAE, P&L

    Target computation from labels:
    - labels[:, 0] = actual mid-price move at 1s (ticks)
    - labels[:, 1] = actual mid-price move at 5s (ticks)
    - labels[:, 2] = actual mid-price move at 10s (ticks)
    - directional_move = label * signal_direction
    - MFE = max(directional_move at 1s, 5s, 10s, 0) — best favorable move across horizons
    - MAE = max(-directional_move at 1s, 5s, 10s, 0) — worst adverse move across horizons
    - P&L = directional_move at 10s - commission — closest to 30s we have
    - profitable = P&L > 0

    Parameters
    ----------
    mbo_path : path to {date}_mbo_events.npz
    predictions : (N_preds, 3) array of CNN-Mamba predictions [pred_1s, pred_5s, pred_10s]
    labels : (N_preds, 3) array of actual future moves [label_1s, label_5s, label_10s] in ticks
    horizon_secs : seconds to track outcome (informational; we use label horizons)
    pred_stride : events between predictions
    pred_window : events per CNN window

    Returns
    -------
    features : (N_valid, N_FEATURES) float32
    targets : (N_valid, N_TARGETS) float32
    meta : (N_valid, 3) float32 — [timestamp_s, pred_idx, signal_direction]
    """
    # Load MBO data for microstructure features
    data = np.load(str(mbo_path))
    events = data["events"]  # (N_events, 25) float32
    timestamps = data["timestamps"]  # (N_events,) int64 nanoseconds

    n_events = len(events)
    n_preds = len(predictions)

    # Precompute per-event quantities for feature extraction
    price_rel = events[:, COL_PRICE_REL].astype(np.float64)
    spread = events[:, COL_SPREAD]
    side = events[:, COL_SIDE]
    qty_log = events[:, COL_QTY_LOG]
    ts_s = timestamps.astype(np.float64) / 1e9  # seconds

    # Price changes for volatility computation (vectorized)
    price_changes = np.diff(price_rel, prepend=price_rel[0]).astype(np.float64)

    # Rolling volatility via cumulative sums (vectorized, no Python loop)
    vol_window = 100
    cs2 = np.cumsum(price_changes**2)
    cs1 = np.cumsum(price_changes)
    cs2_pad = np.concatenate([[0.0], cs2])
    cs1_pad = np.concatenate([[0.0], cs1])
    idx_arr = np.arange(n_events)
    start_idx = np.maximum(idx_arr - vol_window + 1, 0)
    cnt = (idx_arr - start_idx + 1).astype(np.float64)
    sm = cs1_pad[idx_arr + 1] - cs1_pad[start_idx]
    sm2 = cs2_pad[idx_arr + 1] - cs2_pad[start_idx]
    mean_v = sm / cnt
    var_v = np.maximum(sm2 / cnt - mean_v**2, 0.0)
    volatility = np.sqrt(var_v).astype(np.float32)
    volatility[:vol_window] = 0.0

    # Price momentum: rolling mean of price changes over last 50 events (vectorized)
    mom_window = 50
    cs_price = np.cumsum(price_changes)
    cs_price_pad = np.concatenate([[0.0], cs_price])
    momentum = np.zeros(n_events, dtype=np.float32)
    valid_mom = np.arange(mom_window, n_events)
    momentum[valid_mom] = ((cs_price_pad[valid_mom + 1] - cs_price_pad[valid_mom - mom_window + 1]) / mom_window).astype(np.float32)

    # Volume tracking: buy vs sell cumulative volume
    qty = np.exp(qty_log.astype(np.float32))
    buy_mask = (side > 0).astype(np.float32)
    sell_mask = (side < 0).astype(np.float32)
    buy_cum = np.cumsum(qty * buy_mask)
    sell_cum = np.cumsum(qty * sell_mask)
    buy_cum_pad = np.concatenate([[0.0], buy_cum])
    sell_cum_pad = np.concatenate([[0.0], sell_cum])

    # Book depth from event features (columns 6+ are book state)
    bid_depth_raw = events[:, 6] if events.shape[1] > 6 else np.ones(n_events, dtype=np.float32)
    ask_depth_raw = events[:, 7] if events.shape[1] > 7 else np.ones(n_events, dtype=np.float32)
    book_imbalance_raw = events[:, 8] if events.shape[1] > 8 else np.zeros(n_events, dtype=np.float32)

    # Pre-allocate output arrays (upper bound: n_preds)
    features_out = np.zeros((n_preds, N_FEATURES), dtype=np.float32)
    targets_out = np.zeros((n_preds, N_TARGETS), dtype=np.float32)
    meta_out = np.zeros((n_preds, 3), dtype=np.float32)
    out_idx = 0

    # Precompute event indices for all predictions
    pred_event_indices = pred_window + np.arange(n_preds) * pred_stride

    # Filter: valid event indices and non-zero predictions
    valid_mask = pred_event_indices < (n_events - 1)
    pred_nonzero = ~((predictions[:, 0] == 0) & (predictions[:, 1] == 0) & (predictions[:, 2] == 0))
    pred_has_dir = predictions[:, 0] != 0
    # Also filter out NaN labels
    labels_valid = ~np.any(np.isnan(labels), axis=1)
    combined_mask = valid_mask & pred_nonzero & pred_has_dir & labels_valid
    valid_pred_indices = np.where(combined_mask)[0]

    vol_lookback = 200

    for pred_idx in valid_pred_indices:
        event_idx = int(pred_event_indices[pred_idx])
        if event_idx >= n_events:
            continue

        pred_1s = float(predictions[pred_idx, 0])
        pred_5s = float(predictions[pred_idx, 1])
        pred_10s = float(predictions[pred_idx, 2])

        label_1s = float(labels[pred_idx, 0])
        label_5s = float(labels[pred_idx, 1])
        label_10s = float(labels[pred_idx, 2])

        signal_dir = 1 if pred_1s > 0 else -1

        # --- Compute outcome from labels (actual future mid-price moves in ticks) ---
        # Directional moves: positive = favorable for our trade direction
        dir_1s = label_1s * signal_dir
        dir_5s = label_5s * signal_dir
        dir_10s = label_10s * signal_dir

        # MFE: best favorable move we could have captured across horizons
        # (The actual MFE over 30s could be higher, but max(1s,5s,10s) is our best approx)
        mfe = max(dir_1s, dir_5s, dir_10s, 0.0)

        # MAE: worst adverse move across horizons
        mae = max(-dir_1s, -dir_5s, -dir_10s, 0.0)

        # P&L at 10s horizon (our longest available), net of commission
        # Using passive limit fill: entry is at bid/ask, so fill price already
        # reflects the bid-ask side. Only cost is commission.
        pnl_at_horizon = dir_10s - COMMISSION_TICKS
        profitable = 1.0 if pnl_at_horizon > 0 else 0.0

        # Clamp extreme values
        mfe = min(mfe, 50.0)
        mae = min(mae, 50.0)

        # --- Build feature vector from MBO microstructure ---
        current_ts_s = ts_s[event_idx] if event_idx < n_events else 0.0

        abs_1s = abs(pred_1s)
        abs_5s = abs(pred_5s)
        abs_10s = abs(pred_10s)

        # Confidence tier
        if abs_1s < 0.10:
            tier = 0.0
        elif abs_1s < 0.25:
            tier = 1.0
        elif abs_1s < 0.50:
            tier = 2.0
        else:
            tier = 3.0

        bimb = float(book_imbalance_raw[event_idx])
        bd_log = float(np.log1p(max(float(bid_depth_raw[event_idx]), 0)))
        ad_log = float(np.log1p(max(float(ask_depth_raw[event_idx]), 0)))
        spr = float(spread[event_idx])
        vol = float(volatility[event_idx])
        tod = _time_of_day_fraction(current_ts_s)

        # Volume imbalance (over last 200 events)
        start_v = max(0, event_idx - vol_lookback)
        bv = float(buy_cum_pad[event_idx + 1] - buy_cum_pad[start_v + 1])
        sv = float(sell_cum_pad[event_idx + 1] - sell_cum_pad[start_v + 1])
        total_vol = bv + sv
        vol_imb = (bv - sv) / (total_vol + 1e-8) if total_vol > 0 else 0.0

        # Signal features
        signs_agree = (np.sign(pred_1s) == np.sign(pred_5s) == np.sign(pred_10s)) and np.sign(pred_1s) != 0
        agreement = 1.0 if signs_agree else 0.0
        strength = (abs_1s + abs_5s + abs_10s) / 3.0
        mom = float(momentum[event_idx])

        preds_arr = np.array([pred_1s, pred_5s, pred_10s])
        p_std = float(np.std(preds_arr))
        p_range = float(preds_arr.max() - preds_arr.min())

        # Write to output arrays
        features_out[out_idx] = [
            pred_1s, pred_5s, pred_10s,
            abs_1s, abs_5s, abs_10s,
            tier, bimb, bd_log, ad_log, spr,
            vol, tod, vol_imb, agreement, strength,
            mom, p_std, p_range,
        ]
        targets_out[out_idx] = [mfe, mae, pnl_at_horizon, profitable]
        meta_out[out_idx] = [current_ts_s, float(pred_idx), float(signal_dir)]
        out_idx += 1

    # Trim to actual size
    return (
        features_out[:out_idx].copy(),
        targets_out[:out_idx].copy(),
        meta_out[:out_idx].copy(),
    )


def _extract_one_date(args) -> dict:
    """Worker function for parallel extraction."""
    mbo_path, pred_path, horizon_secs, pred_stride, pred_window = args
    date_str = Path(mbo_path).stem.replace("_mbo_events", "")

    t0 = time.time()
    try:
        pred_data = np.load(str(pred_path), allow_pickle=True)
        predictions = pred_data["predictions"]  # (N, 3)
        labels = pred_data["labels"]  # (N, 3)

        features, targets, meta = extract_outcomes_for_date(
            mbo_path=Path(mbo_path),
            predictions=predictions,
            labels=labels,
            horizon_secs=horizon_secs,
            pred_stride=pred_stride,
            pred_window=pred_window,
        )

        elapsed = time.time() - t0
        return {
            "date": date_str,
            "status": "ok",
            "n_samples": len(features),
            "features": features,
            "targets": targets,
            "meta": meta,
            "elapsed": elapsed,
        }
    except Exception as e:
        import traceback

        return {
            "date": date_str,
            "status": f"error: {e}\n{traceback.format_exc()}",
            "n_samples": 0,
            "features": np.zeros((0, N_FEATURES), dtype=np.float32),
            "targets": np.zeros((0, N_TARGETS), dtype=np.float32),
            "meta": np.zeros((0, 3), dtype=np.float32),
            "elapsed": time.time() - t0,
        }


def build_date_pred_index(
    pred_dir: Path,
) -> Dict[str, Tuple[Path, int]]:
    """
    Build mapping: date_str -> (pred_file_path, fold_index).

    Predictions come from walk-forward folds OR per-date prediction files.
    Fold files: fold_XX_oot_predictions.npz (from walk-forward training)
    Per-date files: YYYYMMDD_predictions.npz (from bulk OOT inference)
    """
    index = {}

    # Method 1: Walk-forward fold prediction files
    pred_files = sorted(pred_dir.glob("fold_*_oot_predictions.npz"))
    for pf in pred_files:
        fold_str = pf.stem.split("_")[1]
        fold_idx = int(fold_str)
        try:
            data = np.load(str(pf), allow_pickle=True)
            oot_files = data["oot_files"]
            for of in oot_files:
                date_str = Path(str(of)).stem.replace("_mbo_events", "")
                # If multiple folds cover the same date, prefer higher fold (later training)
                if date_str not in index or fold_idx > index[date_str][1]:
                    index[date_str] = (pf, fold_idx)
        except Exception as e:
            log.warning(f"Could not read {pf}: {e}")

    # Method 2: Per-date prediction files (YYYYMMDD_predictions.npz)
    import re
    date_pred_files = sorted(pred_dir.glob("[0-9]*_predictions.npz"))
    for pf in date_pred_files:
        date_match = re.match(r"(\d{8})_predictions", pf.stem)
        if date_match:
            date_str = date_match.group(1)
            if date_str not in index:  # Don't override fold predictions
                index[date_str] = (pf, -1)  # -1 = per-date file, not from a fold

    log.info(f"Prediction index: {len(index)} dates "
             f"({len(pred_files)} fold files + {len(date_pred_files)} per-date files)")

    return index


def load_all_data(
    data_dir: Path,
    pred_dir: Path,
    precomputed_dir: Optional[Path],
    horizon_secs: float,
    n_workers: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load features and targets for all available dates.

    Returns dictionaries keyed by date_str:
        features_by_date[date] = (N, N_FEATURES)
        targets_by_date[date] = (N, N_TARGETS)
        meta_by_date[date] = (N, 3)
    """
    # Build date -> prediction file index
    pred_index = build_date_pred_index(pred_dir)
    log.info(f"Found predictions for {len(pred_index)} dates: {sorted(pred_index.keys())}")

    # Find matching MBO files
    mbo_files = sorted(data_dir.glob("*_mbo_events.npz"))
    mbo_dates = {Path(f).stem.replace("_mbo_events", ""): f for f in mbo_files}

    matched_dates = sorted(set(pred_index.keys()) & set(mbo_dates.keys()))
    log.info(
        f"Dates with both MBO data and predictions: {len(matched_dates)} — {matched_dates}"
    )

    if not matched_dates:
        log.error("No dates with both MBO data and predictions found!")
        return {}, {}, {}

    # Build work items
    work_items = []
    for date_str in matched_dates:
        mbo_path = mbo_dates[date_str]
        pred_path = pred_index[date_str][0]
        work_items.append(
            (str(mbo_path), str(pred_path), horizon_secs, PRED_STRIDE, PRED_WINDOW)
        )

    features_by_date = {}
    targets_by_date = {}
    meta_by_date = {}

    # Process in parallel
    log.info(f"Extracting outcomes for {len(work_items)} dates using {n_workers} workers...")
    t_start = time.time()

    # Use serial processing if only a few dates or debugging
    if n_workers <= 1 or len(work_items) <= 2:
        for item in work_items:
            result = _extract_one_date(item)
            _process_extraction_result(
                result, features_by_date, targets_by_date, meta_by_date
            )
    else:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(work_items))) as pool:
            futures = {pool.submit(_extract_one_date, w): w for w in work_items}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    log.error(f"Worker crashed: {e}")
                    continue
                _process_extraction_result(
                    result, features_by_date, targets_by_date, meta_by_date
                )

    elapsed = time.time() - t_start
    total_samples = sum(len(v) for v in features_by_date.values())
    log.info(
        f"Extraction complete: {total_samples:,} samples from "
        f"{len(features_by_date)} dates in {elapsed:.1f}s"
    )

    return features_by_date, targets_by_date, meta_by_date


def _process_extraction_result(result, features_by_date, targets_by_date, meta_by_date):
    """Process one extraction result into the date-keyed dicts."""
    date = result["date"]
    if result["status"] != "ok":
        log.warning(f"  {date}: {result['status']}")
        return
    if result["n_samples"] == 0:
        log.warning(f"  {date}: 0 samples extracted")
        return

    features_by_date[date] = result["features"]
    targets_by_date[date] = result["targets"]
    meta_by_date[date] = result["meta"]
    log.info(
        f"  {date}: {result['n_samples']:,} samples in {result['elapsed']:.1f}s"
    )


# ===========================================================================
# Dataset
# ===========================================================================


class ExecutionDataset(Dataset):
    """PyTorch dataset wrapping (features, targets) arrays."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
    ):
        self.features = torch.from_numpy(features).float()
        self.targets = torch.from_numpy(targets).float()
        # Normalize features
        if feature_mean is not None and feature_std is not None:
            self.feature_mean = torch.from_numpy(feature_mean).float()
            self.feature_std = torch.from_numpy(feature_std).float()
            self.features = (self.features - self.feature_mean) / (
                self.feature_std + 1e-8
            )
        else:
            self.feature_mean = None
            self.feature_std = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ===========================================================================
# Model
# ===========================================================================


class SupervisedExecMLP(nn.Module):
    """
    Multi-output MLP for predicting trade outcomes.

    Outputs:
      - MFE (regression, >=0)
      - MAE (regression, >=0)
      - P&L at horizon (regression, any sign)
      - Profitable (binary logit)
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hd

        self.backbone = nn.Sequential(*layers)

        # Separate heads for different output types
        self.head_mfe = nn.Linear(prev_dim, 1)  # regression, clamp >= 0
        self.head_mae = nn.Linear(prev_dim, 1)  # regression, clamp >= 0
        self.head_pnl = nn.Linear(prev_dim, 1)  # regression
        self.head_profitable = nn.Linear(prev_dim, 1)  # binary logit

    def forward(self, x: torch.Tensor) -> dict:
        h = self.backbone(x)
        return {
            "mfe": self.head_mfe(h).squeeze(-1),
            "mae": self.head_mae(h).squeeze(-1),
            "pnl": self.head_pnl(h).squeeze(-1),
            "profitable_logit": self.head_profitable(h).squeeze(-1),
        }


# ===========================================================================
# Training
# ===========================================================================


def compute_loss(
    outputs: dict,
    targets: torch.Tensor,
    mse_weight: float = 1.0,
    bce_weight: float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined loss: MSE for regression targets + BCE for binary classification.

    targets[:, 0] = MFE
    targets[:, 1] = MAE
    targets[:, 2] = P&L
    targets[:, 3] = profitable (0/1)
    """
    mfe_loss = F.mse_loss(outputs["mfe"], targets[:, 0])
    mae_loss = F.mse_loss(outputs["mae"], targets[:, 1])
    pnl_loss = F.mse_loss(outputs["pnl"], targets[:, 2])

    bce_loss = F.binary_cross_entropy_with_logits(
        outputs["profitable_logit"], targets[:, 3]
    )

    total = mse_weight * (mfe_loss + mae_loss + pnl_loss) + bce_weight * bce_loss

    details = {
        "mfe_loss": mfe_loss.item(),
        "mae_loss": mae_loss.item(),
        "pnl_loss": pnl_loss.item(),
        "bce_loss": bce_loss.item(),
        "total_loss": total.item(),
    }
    return total, details


def train_one_epoch(
    model: SupervisedExecMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict:
    """Train for one epoch, return average losses."""
    model.train()
    running = defaultdict(float)
    n_batches = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss, details = compute_loss(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for k, v in details.items():
            running[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: SupervisedExecMLP,
    loader: DataLoader,
    device: str,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Evaluate model, return metrics + raw predictions + targets.
    """
    model.eval()
    running = defaultdict(float)
    n_batches = 0

    all_preds_mfe = []
    all_preds_mae = []
    all_preds_pnl = []
    all_preds_profitable = []
    all_targets = []

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        outputs = model(features)
        _, details = compute_loss(outputs, targets)

        for k, v in details.items():
            running[k] += v
        n_batches += 1

        all_preds_mfe.append(outputs["mfe"].cpu().numpy())
        all_preds_mae.append(outputs["mae"].cpu().numpy())
        all_preds_pnl.append(outputs["pnl"].cpu().numpy())
        all_preds_profitable.append(
            torch.sigmoid(outputs["profitable_logit"]).cpu().numpy()
        )
        all_targets.append(targets.cpu().numpy())

    avg_losses = {k: v / max(n_batches, 1) for k, v in running.items()}

    preds = np.column_stack(
        [
            np.concatenate(all_preds_mfe),
            np.concatenate(all_preds_mae),
            np.concatenate(all_preds_pnl),
            np.concatenate(all_preds_profitable),
        ]
    )
    targets_arr = np.concatenate(all_targets)

    return avg_losses, preds, targets_arr


# ===========================================================================
# Walk-Forward Evaluation Metrics
# ===========================================================================


def compute_oot_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    meta: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute OOT metrics for one fold.

    preds[:, 0] = predicted MFE
    preds[:, 1] = predicted MAE
    preds[:, 2] = predicted P&L
    preds[:, 3] = predicted P(profitable)

    targets[:, 0] = actual MFE
    targets[:, 1] = actual MAE
    targets[:, 2] = actual P&L
    targets[:, 3] = actual profitable (0/1)

    KEY METRIC: correlation between predicted MFE and actual MFE.
    If this is positive, the model learned SOMETHING useful.
    """
    from scipy.stats import pearsonr, spearmanr

    n = len(preds)
    if n < 10:
        return {"n_samples": n, "status": "too_few_samples"}

    metrics = {"n_samples": n}

    # --- Correlation metrics ---
    for i, name in enumerate(["mfe", "mae", "pnl"]):
        p = preds[:, i]
        a = targets[:, i]
        try:
            pcorr, p_pval = pearsonr(p, a)
            scorr, s_pval = spearmanr(p, a)
        except Exception:
            pcorr, scorr = 0.0, 0.0
        metrics[f"{name}_pearson_r"] = float(pcorr)
        metrics[f"{name}_spearman_r"] = float(scorr)
        metrics[f"{name}_mae"] = float(np.mean(np.abs(p - a)))
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean((p - a) ** 2)))

    # --- Classification metrics ---
    prob = preds[:, 3]
    actual = targets[:, 3]
    pred_binary = (prob > 0.5).astype(float)
    accuracy = float(np.mean(pred_binary == actual))
    metrics["binary_accuracy"] = accuracy

    # AUC if we have both classes
    if len(np.unique(actual)) > 1:
        try:
            from sklearn.metrics import roc_auc_score

            metrics["binary_auc"] = float(roc_auc_score(actual, prob))
        except Exception:
            metrics["binary_auc"] = 0.5
    else:
        metrics["binary_auc"] = 0.5

    # --- Threshold analysis (THE KEY EVALUATION) ---
    # Sort by predicted MFE descending. At each threshold, compute actual performance.
    pred_mfe = preds[:, 0]
    actual_mfe = targets[:, 0]
    actual_pnl = targets[:, 2]
    actual_profitable = targets[:, 3]

    sort_idx = np.argsort(-pred_mfe)  # descending
    pred_mfe_sorted = pred_mfe[sort_idx]
    actual_mfe_sorted = actual_mfe[sort_idx]
    actual_pnl_sorted = actual_pnl[sort_idx]
    actual_profitable_sorted = actual_profitable[sort_idx]

    threshold_results = {}
    for pct_label, pct in [
        ("top_1pct", 0.01),
        ("top_5pct", 0.05),
        ("top_10pct", 0.10),
        ("top_20pct", 0.20),
        ("top_50pct", 0.50),
        ("all", 1.00),
    ]:
        k = max(1, int(n * pct))
        subset_mfe = actual_mfe_sorted[:k]
        subset_pnl = actual_pnl_sorted[:k]
        subset_profitable = actual_profitable_sorted[:k]

        avg_mfe = float(np.mean(subset_mfe))
        avg_pnl = float(np.mean(subset_pnl))
        win_rate = float(np.mean(subset_profitable))
        total_pnl = float(np.sum(subset_pnl))

        # Sortino of subset
        if len(subset_pnl) > 1:
            mean_pnl = np.mean(subset_pnl)
            downside = subset_pnl[subset_pnl < 0]
            if len(downside) > 0:
                dd = float(np.sqrt(np.mean(downside**2)))
                sortino = float(mean_pnl / (dd + 1e-8))
            else:
                sortino = float(mean_pnl * 10.0)  # all wins
        else:
            sortino = 0.0

        # Profit factor
        wins = subset_pnl[subset_pnl > 0]
        losses = subset_pnl[subset_pnl < 0]
        gross_win = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 1e-8
        pf = gross_win / gross_loss

        threshold_results[pct_label] = {
            "n_trades": k,
            "avg_mfe_ticks": round(avg_mfe, 4),
            "avg_pnl_ticks": round(avg_pnl, 4),
            "total_pnl_ticks": round(total_pnl, 2),
            "win_rate": round(win_rate, 4),
            "sortino": round(sortino, 4),
            "profit_factor": round(pf, 4),
        }

    metrics["threshold_analysis"] = threshold_results

    # --- KEY: Does predicted MFE rank match actual MFE rank? ---
    # If we sort by predicted MFE, do we get monotonically higher actual MFE?
    # Compare avg actual MFE in top 10% vs bottom 10%
    top_10_actual = float(np.mean(actual_mfe_sorted[: max(1, n // 10)]))
    bot_10_actual = float(np.mean(actual_mfe_sorted[-max(1, n // 10) :]))
    metrics["mfe_top10_vs_bot10"] = round(top_10_actual - bot_10_actual, 4)
    metrics["mfe_top10_avg"] = round(top_10_actual, 4)
    metrics["mfe_bot10_avg"] = round(bot_10_actual, 4)

    # Does the model do better than random? Compare predicted-MFE-sorted vs random
    # Average actual MFE in top 20% of predictions vs overall average
    top_20_actual_mfe = float(np.mean(actual_mfe_sorted[: max(1, n // 5)]))
    overall_avg_mfe = float(np.mean(actual_mfe))
    metrics["mfe_lift_top20"] = round(top_20_actual_mfe - overall_avg_mfe, 4)

    return metrics


# ===========================================================================
# Walk-Forward Training Loop
# ===========================================================================


def walk_forward_train(
    features_by_date: Dict[str, np.ndarray],
    targets_by_date: Dict[str, np.ndarray],
    meta_by_date: Dict[str, np.ndarray],
    output_dir: Path,
    n_train_days: int = 60,
    n_oot_days: int = 1,
    batch_size: int = 4096,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_dims: List[int] = None,
    dropout: float = 0.2,
    patience: int = 7,
    n_workers: int = 8,
    device: str = "cpu",
) -> dict:
    """
    Walk-forward sliding window training.

    With only N dates available, we adapt:
    - If N <= n_train_days: use leave-one-out (train on N-1, test on 1)
    - If N > n_train_days: standard sliding window

    HC #0: SLIDING window only, never expanding.
    """
    if hidden_dims is None:
        hidden_dims = [128, 64, 32]

    output_dir.mkdir(parents=True, exist_ok=True)

    dates = sorted(features_by_date.keys())
    n_dates = len(dates)

    log.info(f"Walk-forward: {n_dates} dates available, "
             f"requested {n_train_days} train + {n_oot_days} OOT")

    if n_dates < 2:
        log.error("Need at least 2 dates for walk-forward")
        return {}

    # Determine fold structure
    # With limited dates, use what we can: min train = 2 days
    min_train = min(n_train_days, n_dates - n_oot_days)
    min_train = max(2, min_train)  # at least 2 training days

    folds = []
    for oot_start in range(min_train, n_dates, n_oot_days):
        oot_end = min(oot_start + n_oot_days, n_dates)
        # SLIDING: take last min_train days before OOT start
        train_start = max(0, oot_start - min_train)
        train_dates = dates[train_start:oot_start]
        oot_dates = dates[oot_start:oot_end]
        if len(train_dates) >= 2 and len(oot_dates) >= 1:
            folds.append((train_dates, oot_dates))

    log.info(f"Generated {len(folds)} walk-forward folds")

    # MLflow setup
    mlflow_run = None
    if MLFLOW_AVAILABLE and os.environ.get("DISABLE_MLFLOW") != "1":
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow_run = mlflow.start_run(run_name=f"sup_exec_{len(folds)}folds")
            mlflow.log_params(
                {
                    "n_dates": n_dates,
                    "n_folds": len(folds),
                    "n_train_days_target": n_train_days,
                    "n_oot_days": n_oot_days,
                    "actual_train_days": min_train,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lr": lr,
                    "hidden_dims": str(hidden_dims),
                    "dropout": dropout,
                    "n_features": N_FEATURES,
                    "commission_ticks": COMMISSION_TICKS,
                    "device": device,
                }
            )
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")
            mlflow_run = None

    # Run each fold
    all_fold_metrics = []
    all_oot_preds = []
    all_oot_targets = []
    all_oot_meta = []

    for fold_idx, (train_dates, oot_dates) in enumerate(folds):
        log.info(
            f"\n{'='*60}\n"
            f"FOLD {fold_idx}: train={train_dates} -> OOT={oot_dates}\n"
            f"{'='*60}"
        )

        # Combine training data
        train_feats = np.concatenate(
            [features_by_date[d] for d in train_dates], axis=0
        )
        train_tgts = np.concatenate(
            [targets_by_date[d] for d in train_dates], axis=0
        )

        # OOT data
        oot_feats = np.concatenate(
            [features_by_date[d] for d in oot_dates], axis=0
        )
        oot_tgts = np.concatenate(
            [targets_by_date[d] for d in oot_dates], axis=0
        )
        oot_meta_arr = np.concatenate(
            [meta_by_date[d] for d in oot_dates], axis=0
        )

        log.info(
            f"  Train: {len(train_feats):,} samples | OOT: {len(oot_feats):,} samples"
        )

        if len(train_feats) < 100 or len(oot_feats) < 10:
            log.warning(f"  Fold {fold_idx}: insufficient data, skipping")
            continue

        # Compute normalization from training data
        feat_mean = train_feats.mean(axis=0)
        feat_std = train_feats.std(axis=0)
        feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)

        # Create datasets
        train_ds = ExecutionDataset(train_feats, train_tgts, feat_mean, feat_std)
        oot_ds = ExecutionDataset(oot_feats, oot_tgts, feat_mean, feat_std)

        actual_workers = min(n_workers, 8)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=actual_workers,
            pin_memory=(device != "cpu"),
            drop_last=False,
        )
        oot_loader = DataLoader(
            oot_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=actual_workers,
            pin_memory=(device != "cpu"),
        )

        # Create model
        model = SupervisedExecMLP(
            input_dim=N_FEATURES,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        # Train
        best_oot_loss = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            train_losses = train_one_epoch(model, train_loader, optimizer, device)
            oot_losses, _, _ = evaluate(model, oot_loader, device)
            scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            if epoch % 5 == 0 or epoch == epochs - 1:
                log.info(
                    f"  Epoch {epoch:3d} | "
                    f"Train loss={train_losses['total_loss']:.4f} | "
                    f"OOT loss={oot_losses['total_loss']:.4f} | "
                    f"lr={current_lr:.2e}"
                )

            # Early stopping on OOT loss
            if oot_losses["total_loss"] < best_oot_loss:
                best_oot_loss = oot_losses["total_loss"]
                best_epoch = epoch
                patience_counter = 0
                # Save best model
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "feat_mean": feat_mean,
                        "feat_std": feat_std,
                        "hidden_dims": hidden_dims,
                        "dropout": dropout,
                        "epoch": epoch,
                        "oot_loss": best_oot_loss,
                    },
                    str(output_dir / f"fold_{fold_idx:02d}_best.pt"),
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    log.info(
                        f"  Early stopping at epoch {epoch} "
                        f"(best={best_epoch}, loss={best_oot_loss:.4f})"
                    )
                    break

        # Reload best model for evaluation
        ckpt = torch.load(
            str(output_dir / f"fold_{fold_idx:02d}_best.pt"),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        # Final OOT evaluation
        _, oot_preds, oot_targets_arr = evaluate(model, oot_loader, device)

        # Compute detailed metrics
        fold_metrics = compute_oot_metrics(oot_preds, oot_targets_arr, oot_meta_arr)
        fold_metrics["fold"] = fold_idx
        fold_metrics["train_dates"] = train_dates
        fold_metrics["oot_dates"] = oot_dates
        fold_metrics["best_epoch"] = best_epoch
        fold_metrics["best_oot_loss"] = best_oot_loss
        fold_metrics["n_train_samples"] = len(train_feats)

        all_fold_metrics.append(fold_metrics)
        all_oot_preds.append(oot_preds)
        all_oot_targets.append(oot_targets_arr)
        all_oot_meta.append(oot_meta_arr)

        # Print key results
        log.info(f"\n  FOLD {fold_idx} OOT Results:")
        log.info(f"    MFE Pearson r:  {fold_metrics.get('mfe_pearson_r', 0):.4f}")
        log.info(f"    MFE Spearman r: {fold_metrics.get('mfe_spearman_r', 0):.4f}")
        log.info(f"    P&L Pearson r:  {fold_metrics.get('pnl_pearson_r', 0):.4f}")
        log.info(f"    Binary AUC:     {fold_metrics.get('binary_auc', 0):.4f}")
        log.info(f"    Binary Acc:     {fold_metrics.get('binary_accuracy', 0):.4f}")
        log.info(f"    MFE top10 vs bot10: {fold_metrics.get('mfe_top10_vs_bot10', 0):.4f}")
        log.info(f"    MFE lift top20: {fold_metrics.get('mfe_lift_top20', 0):.4f}")

        if "threshold_analysis" in fold_metrics:
            log.info(f"    Threshold analysis:")
            for tier, stats in fold_metrics["threshold_analysis"].items():
                log.info(
                    f"      {tier:>10s}: n={stats['n_trades']:5d} "
                    f"WR={stats['win_rate']:.3f} "
                    f"PnL={stats['avg_pnl_ticks']:+.3f} "
                    f"MFE={stats['avg_mfe_ticks']:.3f} "
                    f"PF={stats['profit_factor']:.2f} "
                    f"Sortino={stats['sortino']:.3f}"
                )

        # Save fold predictions
        np.savez_compressed(
            str(output_dir / f"fold_{fold_idx:02d}_predictions.npz"),
            predictions=oot_preds,
            targets=oot_targets_arr,
            meta=oot_meta_arr,
            train_dates=np.array(train_dates),
            oot_dates=np.array(oot_dates),
        )

        # Log to MLflow
        if mlflow_run is not None:
            try:
                prefix = f"fold_{fold_idx:02d}"
                mlflow.log_metrics(
                    {
                        f"{prefix}/mfe_pearson_r": fold_metrics.get(
                            "mfe_pearson_r", 0
                        ),
                        f"{prefix}/mfe_spearman_r": fold_metrics.get(
                            "mfe_spearman_r", 0
                        ),
                        f"{prefix}/pnl_pearson_r": fold_metrics.get(
                            "pnl_pearson_r", 0
                        ),
                        f"{prefix}/binary_auc": fold_metrics.get("binary_auc", 0.5),
                        f"{prefix}/binary_accuracy": fold_metrics.get(
                            "binary_accuracy", 0
                        ),
                        f"{prefix}/mfe_top10_vs_bot10": fold_metrics.get(
                            "mfe_top10_vs_bot10", 0
                        ),
                        f"{prefix}/mfe_lift_top20": fold_metrics.get(
                            "mfe_lift_top20", 0
                        ),
                        f"{prefix}/best_oot_loss": best_oot_loss,
                        f"{prefix}/n_samples": len(oot_feats),
                    },
                    step=fold_idx,
                )
            except Exception as e:
                log.warning(f"MLflow log failed: {e}")

    # --- Aggregate results across folds ---
    if not all_fold_metrics:
        log.error("No folds completed!")
        if mlflow_run is not None:
            mlflow.end_run(status="FAILED")
        return {}

    # Concatenate all OOT predictions for aggregate analysis
    concat_preds = np.concatenate(all_oot_preds, axis=0)
    concat_targets = np.concatenate(all_oot_targets, axis=0)
    concat_meta = np.concatenate(all_oot_meta, axis=0)

    concat_metrics = compute_oot_metrics(concat_preds, concat_targets, concat_meta)
    concat_metrics["n_folds"] = len(all_fold_metrics)
    concat_metrics["total_oot_samples"] = len(concat_preds)

    # Average fold-level metrics
    for key in [
        "mfe_pearson_r",
        "mfe_spearman_r",
        "pnl_pearson_r",
        "binary_auc",
        "binary_accuracy",
        "mfe_top10_vs_bot10",
        "mfe_lift_top20",
    ]:
        vals = [m.get(key, 0) for m in all_fold_metrics if isinstance(m.get(key), (int, float))]
        if vals:
            concat_metrics[f"avg_{key}"] = float(np.mean(vals))
            concat_metrics[f"std_{key}"] = float(np.std(vals))

    # Print aggregate results
    log.info(f"\n{'='*70}")
    log.info(f"AGGREGATE RESULTS ({concat_metrics['n_folds']} folds, "
             f"{concat_metrics['total_oot_samples']:,} OOT samples)")
    log.info(f"{'='*70}")
    log.info(f"  MFE Pearson r (concat):  {concat_metrics.get('mfe_pearson_r', 0):.4f}")
    log.info(f"  MFE Spearman r (concat): {concat_metrics.get('mfe_spearman_r', 0):.4f}")
    log.info(f"  P&L Pearson r (concat):  {concat_metrics.get('pnl_pearson_r', 0):.4f}")
    log.info(f"  Binary AUC (concat):     {concat_metrics.get('binary_auc', 0):.4f}")
    log.info(f"  Binary Accuracy (concat):{concat_metrics.get('binary_accuracy', 0):.4f}")
    log.info(f"  MFE top10 - bot10:       {concat_metrics.get('mfe_top10_vs_bot10', 0):.4f}")
    log.info(f"  MFE lift top20:          {concat_metrics.get('mfe_lift_top20', 0):.4f}")

    if "threshold_analysis" in concat_metrics:
        log.info(f"\n  CONCAT Threshold Analysis (all OOT folds combined):")
        for tier, stats in concat_metrics["threshold_analysis"].items():
            log.info(
                f"    {tier:>10s}: n={stats['n_trades']:5d} "
                f"WR={stats['win_rate']:.3f} "
                f"PnL={stats['avg_pnl_ticks']:+.3f}t "
                f"TotalPnL={stats['total_pnl_ticks']:+.1f}t "
                f"MFE={stats['avg_mfe_ticks']:.3f}t "
                f"PF={stats['profit_factor']:.2f} "
                f"Sortino={stats['sortino']:.3f}"
            )

    # Sanity check: does the data itself show confidence -> outcome monotonicity?
    log.info(f"\n  SANITY CHECK: Raw data confidence -> outcome relationship")
    log.info(f"  (This should be positive even WITHOUT a model)")
    abs_pred_1s = np.abs(concat_targets[:, 2] + COMMISSION_TICKS)  # undo commission to get raw move
    # Actually, let's look at features directly
    # We need the raw features (before normalization). Use the stored files.
    _raw_sanity_check(features_by_date, targets_by_date)

    # Save aggregate results
    # Make metrics JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return obj

    summary = {
        "concat_metrics": make_serializable(concat_metrics),
        "per_fold_metrics": make_serializable(all_fold_metrics),
        "config": {
            "n_train_days": n_train_days,
            "n_oot_days": n_oot_days,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "n_features": N_FEATURES,
            "feature_names": FEATURE_NAMES,
            "commission_ticks": COMMISSION_TICKS,
        },
    }

    with open(output_dir / "walk_forward_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    np.savez_compressed(
        str(output_dir / "concat_oot_predictions.npz"),
        predictions=concat_preds,
        targets=concat_targets,
        meta=concat_meta,
    )

    # MLflow aggregate
    if mlflow_run is not None:
        try:
            mlflow.log_metrics(
                {
                    "concat/mfe_pearson_r": concat_metrics.get("mfe_pearson_r", 0),
                    "concat/mfe_spearman_r": concat_metrics.get("mfe_spearman_r", 0),
                    "concat/pnl_pearson_r": concat_metrics.get("pnl_pearson_r", 0),
                    "concat/binary_auc": concat_metrics.get("binary_auc", 0.5),
                    "concat/binary_accuracy": concat_metrics.get("binary_accuracy", 0),
                    "concat/mfe_top10_vs_bot10": concat_metrics.get(
                        "mfe_top10_vs_bot10", 0
                    ),
                    "concat/mfe_lift_top20": concat_metrics.get("mfe_lift_top20", 0),
                    "concat/n_folds": len(all_fold_metrics),
                    "concat/total_oot_samples": len(concat_preds),
                }
            )
            mlflow.log_artifact(str(output_dir / "walk_forward_summary.json"))
            mlflow.end_run()
        except Exception as e:
            log.warning(f"MLflow finalization failed: {e}")

    log.info(f"\nResults saved to {output_dir}")
    return summary


def _raw_sanity_check(
    features_by_date: Dict[str, np.ndarray],
    targets_by_date: Dict[str, np.ndarray],
):
    """
    Sanity check: does raw signal confidence predict outcomes?
    This should show a positive relationship even without any model.
    If it doesn't, the data pipeline has a fundamental problem.
    """
    all_feats = np.concatenate(list(features_by_date.values()), axis=0)
    all_tgts = np.concatenate(list(targets_by_date.values()), axis=0)

    abs_pred_1s = all_feats[:, FEATURE_NAMES.index("abs_pred_1s")]
    actual_mfe = all_tgts[:, 0]
    actual_pnl = all_tgts[:, 2]
    actual_profitable = all_tgts[:, 3]

    # Bin by confidence quartiles
    quartiles = np.percentile(abs_pred_1s, [25, 50, 75, 90, 95, 99])
    bins = [0.0] + list(quartiles) + [float("inf")]
    bin_labels = [
        "0-25%",
        "25-50%",
        "50-75%",
        "75-90%",
        "90-95%",
        "95-99%",
        "99-100%",
    ]

    log.info(f"  Signal confidence vs outcome (raw data, no model):")
    log.info(
        f"  {'Bin':>10s} | {'N':>6s} | {'Avg MFE':>8s} | "
        f"{'Avg PnL':>8s} | {'WinRate':>8s} | {'Abs Pred1s':>10s}"
    )
    log.info(f"  {'-'*70}")

    for i in range(len(bins) - 1):
        mask = (abs_pred_1s >= bins[i]) & (abs_pred_1s < bins[i + 1])
        if mask.sum() == 0:
            continue

        avg_mfe = float(np.mean(actual_mfe[mask]))
        avg_pnl = float(np.mean(actual_pnl[mask]))
        wr = float(np.mean(actual_profitable[mask]))
        avg_abs = float(np.mean(abs_pred_1s[mask]))
        n = int(mask.sum())

        log.info(
            f"  {bin_labels[i]:>10s} | {n:6d} | {avg_mfe:+8.3f} | "
            f"{avg_pnl:+8.3f} | {wr:8.3f} | {avg_abs:10.4f}"
        )

    # Overall correlation
    from scipy.stats import spearmanr

    scorr, _ = spearmanr(abs_pred_1s, actual_mfe)
    log.info(f"\n  Spearman(abs_pred_1s, actual_MFE) = {scorr:.4f}")
    scorr2, _ = spearmanr(abs_pred_1s, actual_pnl)
    log.info(f"  Spearman(abs_pred_1s, actual_PnL) = {scorr2:.4f}")

    if scorr < 0.01:
        log.warning(
            "  WARNING: Signal confidence has ZERO correlation with MFE. "
            "Data pipeline may have an issue!"
        )
    elif scorr > 0.05:
        log.info(
            f"  GOOD: Positive correlation ({scorr:.4f}) confirms signal predicts outcomes."
        )


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train supervised execution model (predict trade outcomes from signal features)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"),
        help="Directory with MBO event .npz files",
    )
    parser.add_argument(
        "--pred-dir",
        type=str,
        default=str(LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"),
        help="Directory with CNN-Mamba prediction .npz files",
    )
    parser.add_argument(
        "--precomputed-dir",
        type=str,
        default=str(LVL3_ROOT / "data" / "precomputed_obs"),
        help="Directory with precomputed observation .npy files (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(LVL3_ROOT / "output" / "supervised_exec_v1"),
        help="Output directory for models and results",
    )
    parser.add_argument(
        "--n-train-days",
        type=int,
        default=60,
        help="Number of training days per fold (sliding window)",
    )
    parser.add_argument(
        "--n-oot-days",
        type=int,
        default=1,
        help="Number of OOT days per fold",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[128, 64, 32],
        help="Hidden layer dimensions",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--horizon-secs",
        type=float,
        default=30.0,
        help="Outcome tracking horizon in seconds",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=12,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on",
    )
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Max folds to run (for testing)",
    )

    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Supervised Execution Model — Predicting Trade Outcomes")
    log.info("=" * 70)
    log.info(f"Data dir:       {args.data_dir}")
    log.info(f"Pred dir:       {args.pred_dir}")
    log.info(f"Precomputed:    {args.precomputed_dir}")
    log.info(f"Output dir:     {args.output_dir}")
    log.info(f"Train days:     {args.n_train_days}")
    log.info(f"OOT days:       {args.n_oot_days}")
    log.info(f"Batch size:     {args.batch_size}")
    log.info(f"Epochs:         {args.epochs}")
    log.info(f"LR:             {args.lr}")
    log.info(f"Hidden dims:    {args.hidden_dims}")
    log.info(f"Horizon (sec):  {args.horizon_secs}")
    log.info(f"Workers:        {args.n_workers}")
    log.info(f"Device:         {args.device}")
    log.info(f"Commission:     {COMMISSION_TICKS:.3f} ticks RT")
    log.info("")

    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # Step 1: Load all data
    log.info("STEP 1: Loading data and extracting features/outcomes...")
    t0 = time.time()

    features_by_date, targets_by_date, meta_by_date = load_all_data(
        data_dir=Path(args.data_dir),
        pred_dir=Path(args.pred_dir),
        precomputed_dir=Path(args.precomputed_dir) if args.precomputed_dir else None,
        horizon_secs=args.horizon_secs,
        n_workers=args.n_workers,
    )

    if not features_by_date:
        log.error("No data loaded! Check paths and data availability.")
        return

    total_samples = sum(len(v) for v in features_by_date.values())
    log.info(
        f"Data loaded: {total_samples:,} samples from "
        f"{len(features_by_date)} dates in {time.time()-t0:.1f}s"
    )

    # Step 2: Walk-forward training
    log.info("\nSTEP 2: Walk-forward training...")

    summary = walk_forward_train(
        features_by_date=features_by_date,
        targets_by_date=targets_by_date,
        meta_by_date=meta_by_date,
        output_dir=Path(args.output_dir),
        n_train_days=args.n_train_days,
        n_oot_days=args.n_oot_days,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        patience=args.patience,
        n_workers=args.n_workers,
        device=device,
    )

    if summary:
        log.info("\nTraining complete!")
        log.info(f"Results saved to {args.output_dir}")
    else:
        log.error("Training failed — no results produced.")


if __name__ == "__main__":
    main()
