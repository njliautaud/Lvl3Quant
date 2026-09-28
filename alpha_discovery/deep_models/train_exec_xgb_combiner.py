#!/usr/bin/env python3
"""
train_exec_xgb_combiner.py — XGBoost Execution Combiner (3-Model Ensemble)
==========================================================================

Replaces the MLP combiner v1 (IC_10s=0.058, early-stopped epoch 5-7).
XGBoost handles small data (9 days, ~213K samples) far better than NNs.

Combines embeddings + predictions from 3 frozen models:
  1. CNN-Mamba v2  — embeddings(96d) + predictions(3 horizons)
  2. PatchTST       — embeddings(256d) + predictions(3 horizons)
  3. Vol LGBM v3    — predictions(3 dims)

Builds 4 XGBoost models (same structure as smart_exec_v4):
  - Gate:  XGBClassifier  — trade/skip decision (P(|y_10s| > 0.5 ticks))
  - TP:    XGBRegressor   — predicted magnitude when correct
  - SL:    XGBRegressor   — predicted magnitude when wrong
  - Exit:  XGBRegressor   — predicted optimal hold time

Walk-forward: sliding 7-day train, 1-day test (HC #0 compliant).
Gate threshold sweep: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80.

Data alignment: CNN-Mamba is reference (~14K/day). PatchTST (~28K/day)
  downsampled by nearest-neighbor. Vol LGBM aligned by anchor_idxs.

Usage:
  python train_exec_xgb_combiner.py
  python train_exec_xgb_combiner.py --train-days 7

Author: Execution combiner for Lvl3Quant
Date: 2026-04-29
"""

import os
import sys
import gc
import re
import time
import json
import logging
import argparse
import warnings
import socket
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import scipy.stats
import xgboost as xgb
import joblib

# MLflow
try:
    if int(os.environ.get("DISABLE_MLFLOW", 0)):
        raise ImportError("MLflow disabled")
    import mlflow
    MLFLOW_AVAILABLE = True
except (ImportError, ValueError):
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow disabled or not installed")

# ============================================================
# Logging
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

_ts = time.strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"exec_xgb_combiner_{_ts}.log"

logging.root.handlers.clear()
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_file_handler = logging.FileHandler(log_path)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_fmt)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_file_handler)
logging.root.addHandler(_stream_handler)

logger = logging.getLogger(__name__)

# Force flush
for _h in logging.root.handlers:
    _orig_emit = _h.emit
    def _flush_emit(record, _emit=_orig_emit, _handler=_h):
        _emit(record)
        _handler.flush()
    _h.emit = _flush_emit

print(f">>> train_exec_xgb_combiner.py loaded | log: {log_path}", flush=True)

# ============================================================
# Paths & Constants
# ============================================================
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent

CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
VOL_LGBM_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"
DEFAULT_OUTPUT_DIR = LVL3_ROOT / "output" / "exec_xgb_combiner_v1"

TICK = 0.25
TICK_VAL = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0
ROUND_TRIP_COST = 2 * (COMMISSION_TICKS + SPREAD_TICKS * 0.5)  # 1.752 ticks

MLFLOW_EXPERIMENT = "ExecXGB_Combiner_3Model"

OVERLAP_DATES = [
    "20260223", "20260224", "20260225", "20260226", "20260227",
    "20260301", "20260302", "20260303", "20260304",
]

GATE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# XGBoost default params
XGB_PARAMS = {
    "max_depth": 6,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "early_stopping_rounds": 20,
}


def _detect_mlflow_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    local_store = LVL3_ROOT / "mlflow"
    if local_store.exists():
        return f"file://{local_store}"
    try:
        s = socket.create_connection(("localhost", 5000), timeout=2)
        s.close()
        return "http://localhost:5000"
    except Exception:
        pass
    local_store.mkdir(exist_ok=True)
    return f"file://{local_store}"


MLFLOW_TRACKING_URI = _detect_mlflow_uri()


# ============================================================
# Data Discovery & Alignment (reused from MLP combiner)
# ============================================================

def _extract_date(oot_files) -> Optional[str]:
    if oot_files is None or len(oot_files) == 0:
        return None
    fname = str(oot_files[0]) if hasattr(oot_files, '__iter__') else str(oot_files)
    m = re.search(r'(\d{8})', fname)
    return m.group(1) if m else None


def discover_cnn_mamba_folds() -> Dict[str, Tuple[Path, int]]:
    date_map = {}
    for f in sorted(CNN_MAMBA_DIR.glob("fold_*_oot_predictions.npz")):
        idx = int(f.stem.split("_")[1])
        try:
            data = np.load(str(f), allow_pickle=True)
            date = _extract_date(data.get("oot_files", None))
            if date:
                date_map[date] = (f, idx)
        except Exception as e:
            logger.warning(f"CNN-Mamba fold {idx}: {e}")
    return date_map


def discover_patchtst_folds() -> Dict[str, Tuple[Path, int]]:
    date_map = {}
    for f in sorted(PATCHTST_DIR.glob("fold_*_oot_predictions.npz")):
        idx = int(f.stem.split("_")[1])
        try:
            data = np.load(str(f), allow_pickle=True)
            date = _extract_date(data.get("oot_files", None))
            if date:
                date_map[date] = (f, idx)
        except Exception as e:
            logger.warning(f"PatchTST fold {idx}: {e}")
    return date_map


def discover_vol_lgbm_files() -> Dict[str, Path]:
    date_map = {}
    for f in sorted(VOL_LGBM_DIR.glob("vol_v3_*_predictions.npz")):
        m = re.search(r'(\d{8})', f.stem)
        if m:
            date_map[m.group(1)] = f
    for f in sorted(VOL_LGBM_DIR.glob("fold_*_predictions.npz")):
        try:
            data = np.load(str(f), allow_pickle=True)
            date = _extract_date(data.get("oot_files", None))
            if date:
                date_map[date] = f
        except Exception:
            pass
    return date_map


def align_samples(cnn_n: int, ptst_n: int, vol_n: int,
                  vol_anchor_idxs: Optional[np.ndarray] = None,
                  cnn_stride: int = 250, cnn_window: int = 3000,
                  ptst_stride: int = 250, ptst_window: int = 500
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align sample indices across 3 models. CNN-Mamba is reference."""
    cnn_positions = np.arange(cnn_n) * cnn_stride + cnn_window
    ptst_positions = np.arange(ptst_n) * ptst_stride + ptst_window

    ptst_idxs = np.searchsorted(ptst_positions, cnn_positions, side="left")
    ptst_idxs = np.clip(ptst_idxs, 0, ptst_n - 1)

    if vol_anchor_idxs is not None and len(vol_anchor_idxs) > 0:
        vol_idxs = np.searchsorted(vol_anchor_idxs, cnn_positions, side="left")
        vol_idxs = np.clip(vol_idxs, 0, vol_n - 1)
    else:
        vol_idxs = np.arange(min(cnn_n, vol_n))
        if vol_n < cnn_n:
            vol_idxs = np.concatenate([
                vol_idxs, np.full(cnn_n - vol_n, vol_n - 1, dtype=np.int64)
            ])

    cnn_idxs = np.arange(cnn_n)
    n = min(len(cnn_idxs), len(ptst_idxs), len(vol_idxs))
    return cnn_idxs[:n], ptst_idxs[:n], vol_idxs[:n]


def load_date_data(date: str,
                   cnn_folds: Dict, ptst_folds: Dict,
                   vol_files: Dict) -> Optional[Dict]:
    """Load and align predictions from all 3 models for a single date."""
    if date not in cnn_folds:
        logger.warning(f"Date {date}: CNN-Mamba not found")
        return None
    if date not in ptst_folds:
        logger.warning(f"Date {date}: PatchTST not found")
        return None

    cnn_path, cnn_fold_idx = cnn_folds[date]
    ptst_path, ptst_fold_idx = ptst_folds[date]

    try:
        cnn_data = np.load(str(cnn_path), allow_pickle=True)
        ptst_data = np.load(str(ptst_path), allow_pickle=True)
    except Exception as e:
        logger.error(f"Date {date}: load error: {e}")
        return None

    cnn_preds = cnn_data["predictions"].astype(np.float32)
    cnn_labels = cnn_data["labels"].astype(np.float32)
    cnn_embeds = cnn_data["embeddings"].astype(np.float32)

    ptst_preds = ptst_data["predictions"].astype(np.float32)
    ptst_embeds = ptst_data["embeddings"].astype(np.float32)

    vol_available = date in vol_files
    vol_anchor_idxs = None
    if vol_available:
        try:
            vol_data = np.load(str(vol_files[date]), allow_pickle=True)
            vol_preds = vol_data["predictions"].astype(np.float32)
            if "anchor_idxs" in vol_data:
                vol_anchor_idxs = vol_data["anchor_idxs"]
        except Exception as e:
            logger.warning(f"Date {date}: vol load failed: {e}")
            vol_available = False

    if not vol_available:
        vol_preds = np.zeros((len(cnn_preds), 3), dtype=np.float32)
        logger.info(f"Date {date}: vol not available, using zeros")

    cnn_idxs, ptst_idxs, vol_idxs = align_samples(
        cnn_n=len(cnn_preds), ptst_n=len(ptst_preds),
        vol_n=len(vol_preds), vol_anchor_idxs=vol_anchor_idxs,
    )

    n = len(cnn_idxs)
    if n < 100:
        logger.warning(f"Date {date}: only {n} aligned samples, skipping")
        return None

    result = {
        "date": date,
        "cnn_preds": cnn_preds[cnn_idxs],
        "cnn_embeds": cnn_embeds[cnn_idxs],
        "cnn_labels": cnn_labels[cnn_idxs],
        "ptst_preds": ptst_preds[ptst_idxs],
        "ptst_embeds": ptst_embeds[ptst_idxs],
        "vol_preds": vol_preds[vol_idxs],
        "vol_available": vol_available,
        "n_samples": n,
        "cnn_fold_idx": cnn_fold_idx,
        "ptst_fold_idx": ptst_fold_idx,
    }

    logger.info(f"Date {date}: {n} aligned samples "
                f"(CNN={len(cnn_preds)}, PatchTST={len(ptst_preds)}, "
                f"Vol={'%d' % len(vol_preds) if vol_available else 'zeros'})")
    return result


# ============================================================
# Feature Engineering
# ============================================================

def compute_derived_features(cnn_preds: np.ndarray,
                             ptst_preds: np.ndarray,
                             vol_preds: np.ndarray,
                             n: int) -> Tuple[np.ndarray, List[str]]:
    """
    Compute derived context features from multi-model predictions.
    Returns (N, n_derived) array and feature names.
    """
    features = []
    names = []

    # Direction agreement per horizon
    for h, label in enumerate(["1s", "5s", "10s"]):
        cnn_dir = np.sign(cnn_preds[:, h])
        ptst_dir = np.sign(ptst_preds[:, h])
        features.append((cnn_dir == ptst_dir).astype(np.float32))
        names.append(f"cnn_ptst_agree_{label}")

    # All 3 agree on 10s
    cnn_dir_10s = np.sign(cnn_preds[:, 2])
    ptst_dir_10s = np.sign(ptst_preds[:, 2])
    vol_dir_10s = np.sign(vol_preds[:, 2])
    features.append(((cnn_dir_10s == ptst_dir_10s) & (ptst_dir_10s == vol_dir_10s)).astype(np.float32))
    names.append("all_agree_10s")

    # Stacked 10s predictions
    stacked_10s = np.stack([cnn_preds[:, 2], ptst_preds[:, 2], vol_preds[:, 2]], axis=1)

    # Confidence spread
    features.append(stacked_10s.max(axis=1) - stacked_10s.min(axis=1))
    names.append("confidence_spread_10s")

    # Mean abs prediction
    features.append(np.abs(stacked_10s).mean(axis=1))
    names.append("mean_abs_pred_10s")

    # CNN confidence (normalized magnitude)
    cnn_std = max(np.abs(cnn_preds[:, 2]).std(), 1e-8)
    features.append(np.abs(cnn_preds[:, 2]) / cnn_std)
    names.append("cnn_confidence_10s")

    # PatchTST confidence
    ptst_std = max(np.abs(ptst_preds[:, 2]).std(), 1e-8)
    features.append(np.abs(ptst_preds[:, 2]) / ptst_std)
    names.append("ptst_confidence_10s")

    # Prediction dispersion
    features.append(stacked_10s.std(axis=1))
    names.append("pred_dispersion_10s")

    # Agreement score (continuous): cnn * ptst (positive = agree)
    features.append(cnn_preds[:, 2] * ptst_preds[:, 2])
    names.append("agreement_score_10s")

    # Best conviction
    features.append(np.maximum(np.abs(cnn_preds[:, 2]), np.abs(ptst_preds[:, 2])))
    names.append("best_conviction_10s")

    # Conviction difference (which model is more sure)
    features.append(np.abs(cnn_preds[:, 2]) - np.abs(ptst_preds[:, 2]))
    names.append("conv_diff_10s")

    # Vol-weighted conviction
    features.append(vol_preds[:, 2] * np.abs(cnn_preds[:, 2]))
    names.append("vol_weighted_conv_10s")

    # Mean prediction (ensemble)
    features.append(stacked_10s.mean(axis=1))
    names.append("ensemble_mean_10s")

    # Horizon momentum: 10s - 1s prediction (are models predicting acceleration?)
    for model_name, preds in [("cnn", cnn_preds), ("ptst", ptst_preds)]:
        features.append(preds[:, 2] - preds[:, 0])
        names.append(f"{model_name}_horizon_momentum")

    return np.column_stack(features).astype(np.float32), names


def assemble_features(date_data: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    Assemble the full feature vector for each sample.

    Layout:
        [0:96]    CNN-Mamba embeddings (96d)
        [96:99]   CNN-Mamba predictions (3)
        [99:355]  PatchTST embeddings (256d)
        [355:358] PatchTST predictions (3)
        [358:361] Vol LGBM predictions (3)
        [361:...]  Derived features (~16)
    """
    n = date_data["n_samples"]

    derived, derived_names = compute_derived_features(
        date_data["cnn_preds"],
        date_data["ptst_preds"],
        date_data["vol_preds"],
        n,
    )

    # Build feature names
    names = []
    names += [f"cnn_embed_{i}" for i in range(96)]
    names += ["cnn_pred_1s", "cnn_pred_5s", "cnn_pred_10s"]
    names += [f"ptst_embed_{i}" for i in range(256)]
    names += ["ptst_pred_1s", "ptst_pred_5s", "ptst_pred_10s"]
    names += ["vol_pred_10s", "vol_pred_30s", "vol_pred_60s"]
    names += derived_names

    features = np.concatenate([
        date_data["cnn_embeds"],    # (N, 96)
        date_data["cnn_preds"],     # (N, 3)
        date_data["ptst_embeds"],   # (N, 256)
        date_data["ptst_preds"],    # (N, 3)
        date_data["vol_preds"],     # (N, 3)
        derived,                    # (N, ~16)
    ], axis=1)

    return features.astype(np.float32), names


# ============================================================
# Label Construction
# ============================================================

def compute_labels(cnn_labels: np.ndarray,
                   cnn_preds: np.ndarray,
                   ptst_preds: np.ndarray,
                   gate_threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Compute labels for all 4 XGBoost models.

    Returns:
        direction:   (N,)  — sign(y_10s), +1 or -1
        magnitude:   (N,)  — |y_10s| in ticks
        gate:        (N,)  — 1 if |y_10s| > gate_threshold
        tp_label:    (N,)  — magnitude when prediction correct (MFE proxy)
        sl_label:    (N,)  — magnitude when prediction wrong (MAE proxy)
        exit_label:  (N,)  — ratio of 10s to 5s return (acceleration)
        y_10s:       (N,)  — raw y_10s for IC calculation
    """
    y_10s = cnn_labels[:, 2]
    y_5s = cnn_labels[:, 1]

    # Gate label: |y_10s| > threshold (trade is worthwhile)
    gate = (np.abs(y_10s) > gate_threshold).astype(np.float32)

    # Direction from ensemble mean
    ensemble_dir = np.sign(cnn_preds[:, 2] + ptst_preds[:, 2])
    # Where ensemble is zero, use CNN direction
    zero_mask = ensemble_dir == 0
    ensemble_dir[zero_mask] = np.sign(cnn_preds[zero_mask, 2])

    # Directional return = y_10s * sign(prediction)
    directional = y_10s * ensemble_dir

    # TP: how much you gain when correct (MFE proxy)
    tp_label = np.maximum(directional, 0.0)

    # SL: how much you lose when wrong (MAE proxy)
    sl_label = np.abs(np.minimum(directional, 0.0))

    # Exit timing: 10s/5s return ratio as proxy for hold time value
    # If |y_10s| > |y_5s|, holding longer is good; else exit early
    exit_label = np.where(
        np.abs(y_5s) > 0.1,
        np.clip(np.abs(y_10s) / np.abs(y_5s), 0.0, 5.0),
        1.0  # neutral
    )

    n_gate = gate.sum()
    logger.info(f"  Labels: gate_rate={100*n_gate/len(gate):.1f}% ({int(n_gate)}/{len(gate)}), "
                f"mean_tp={tp_label.mean():.3f}, mean_sl={sl_label.mean():.3f}")

    return {
        "gate": gate,
        "tp_label": tp_label.astype(np.float32),
        "sl_label": sl_label.astype(np.float32),
        "exit_label": exit_label.astype(np.float32),
        "direction": cnn_labels,  # (N, 3) for IC calc
        "y_10s": y_10s.astype(np.float32),
        "ensemble_dir": ensemble_dir.astype(np.float32),
    }


# ============================================================
# Metrics
# ============================================================

def compute_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return float("nan")
    return float(scipy.stats.spearmanr(preds[mask], labels[mask]).statistic)


def compute_sortino(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return 0.0
    mean_ret = np.mean(pnl)
    downside = pnl[pnl < 0]
    if len(downside) < 2:
        return float("inf") if mean_ret > 0 else 0.0
    dd_std = np.std(downside)
    if dd_std < 1e-10:
        return float("inf") if mean_ret > 0 else 0.0
    return float(mean_ret / dd_std)


def evaluate_at_threshold(gate_probs: np.ndarray,
                          ensemble_preds_10s: np.ndarray,
                          y_10s: np.ndarray,
                          labels_3h: np.ndarray,
                          threshold: float) -> Dict:
    """Evaluate gated performance at a given gate threshold."""
    mask = gate_probs >= threshold
    n_gated = mask.sum()
    n_total = len(gate_probs)

    if n_gated < 5:
        return {
            "threshold": threshold,
            "n_gated": int(n_gated),
            "coverage": 0.0,
        }

    gated_preds = ensemble_preds_10s[mask]
    gated_y = y_10s[mask]
    gated_labels = labels_3h[mask]

    # Direction accuracy
    pred_dir = np.sign(gated_preds)
    actual_dir = np.sign(gated_y)
    dir_acc = np.mean(pred_dir == actual_dir)

    # IC on gated samples
    ic_10s = compute_ic(gated_preds, gated_y)

    # IC per horizon using labels_3h
    ic_1s = compute_ic(gated_labels[:, 0], gated_labels[:, 0])  # self-IC is 1; we need pred
    # Actually compute IC of ensemble direction * magnitude vs actual
    ic_10s_gated = compute_ic(gated_preds, gated_y)

    # PnL simulation
    entry_dir = np.sign(gated_preds)
    pnl_ticks = entry_dir * gated_y - ROUND_TRIP_COST
    total_pnl = pnl_ticks.sum()
    mean_pnl = total_pnl / n_gated
    win_rate = np.mean(pnl_ticks > 0)
    sortino = compute_sortino(pnl_ticks)

    # Gate accuracy (did gate correctly select tradeable moments?)
    actual_gate = (np.abs(gated_y) > 0.5).astype(float)
    gate_precision = float(actual_gate.mean())

    return {
        "threshold": threshold,
        "n_gated": int(n_gated),
        "coverage": float(n_gated / n_total),
        "dir_accuracy": float(dir_acc),
        "ic_10s": ic_10s_gated,
        "total_pnl_ticks": float(total_pnl),
        "mean_pnl_per_trade": float(mean_pnl),
        "win_rate": float(win_rate),
        "sortino": float(sortino),
        "gate_precision": gate_precision,
    }


# ============================================================
# XGBoost Model Training
# ============================================================

def train_gate_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     feature_names: List[str],
                     params: Dict) -> xgb.XGBClassifier:
    """Train gate classifier with class imbalance handling."""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(pos_count, 1)
    scale_pos = min(scale_pos, 10.0)

    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Find best iteration
    best_iter = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else params["n_estimators"]
    logger.info(f"    Gate: best_iteration={best_iter}, scale_pos_weight={scale_pos:.2f}")

    return model


def train_regressor(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    name: str, params: Dict) -> xgb.XGBRegressor:
    """Train a regression model (TP, SL, or Exit)."""
    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric="rmse",
        tree_method="hist",
        verbosity=0,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else params["n_estimators"]
    logger.info(f"    {name}: best_iteration={best_iter}")

    return model


def get_top_features(model, feature_names: List[str], top_n: int = 15) -> Dict[str, float]:
    """Get top N features by importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return {feature_names[i]: float(importances[i]) for i in indices}


# ============================================================
# Walk-Forward Training
# ============================================================

def train_walk_forward(all_dates: List[Dict], args: argparse.Namespace,
                       output_dir: Path) -> Dict:
    """
    Sliding window walk-forward. HC #0 compliant.

    With 9 dates and train_days=7:
      Fold 0: train on days 0-6, test on day 7
      Fold 1: train on days 1-7, test on day 8
    """
    n_dates = len(all_dates)
    train_days = args.train_days

    if n_dates < train_days + 1:
        logger.error(f"Need {train_days + 1} dates, have {n_dates}")
        return {}

    n_folds = n_dates - train_days
    logger.info(f"\nWalk-forward: {n_folds} folds, {train_days} train days, sliding window")

    # Get feature names from first date
    _, feature_names = assemble_features(all_dates[0])
    input_dim = len(feature_names)
    logger.info(f"Input dimension: {input_dim} ({len(feature_names)} features)")

    # XGBoost params with early stopping
    xgb_params = {
        "n_estimators": args.xgb_estimators,
        "max_depth": args.xgb_depth,
        "learning_rate": args.xgb_lr,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
    }

    # Accumulators for concat metrics
    concat_gate_probs = []
    concat_ensemble_preds = []
    concat_y_10s = []
    concat_labels_3h = []
    concat_tp_preds = []
    concat_sl_preds = []
    concat_exit_preds = []
    concat_tp_labels = []
    concat_sl_labels = []
    concat_exit_labels = []

    fold_results = []

    for fold_idx in range(n_folds):
        train_start = fold_idx
        train_end = fold_idx + train_days
        test_idx = train_end

        train_date_data = all_dates[train_start:train_end]
        test_date_data = all_dates[test_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx} | Train: {train_date_data[0]['date']}-{train_date_data[-1]['date']} "
                    f"({len(train_date_data)} days) | Test: {test_date_data['date']}")
        logger.info(f"{'='*60}")

        # ---- Build training data ----
        train_feat_list = []
        train_label_list = []

        for td in train_date_data:
            feat, _ = assemble_features(td)
            labels = compute_labels(
                td["cnn_labels"], td["cnn_preds"],
                td["ptst_preds"],
                gate_threshold=args.gate_threshold,
            )
            train_feat_list.append(feat)
            train_label_list.append(labels)

        X_train = np.concatenate(train_feat_list, axis=0)
        y_gate_train = np.concatenate([l["gate"] for l in train_label_list])
        y_tp_train = np.concatenate([l["tp_label"] for l in train_label_list])
        y_sl_train = np.concatenate([l["sl_label"] for l in train_label_list])
        y_exit_train = np.concatenate([l["exit_label"] for l in train_label_list])

        # ---- Build test data ----
        X_test, _ = assemble_features(test_date_data)
        test_labels = compute_labels(
            test_date_data["cnn_labels"], test_date_data["cnn_preds"],
            test_date_data["ptst_preds"],
            gate_threshold=args.gate_threshold,
        )
        y_gate_test = test_labels["gate"]
        y_tp_test = test_labels["tp_label"]
        y_sl_test = test_labels["sl_label"]
        y_exit_test = test_labels["exit_label"]
        y_10s_test = test_labels["y_10s"]
        labels_3h_test = test_labels["direction"]
        ensemble_dir_test = test_labels["ensemble_dir"]

        n_train = len(X_train)
        n_test = len(X_test)
        logger.info(f"  Train: {n_train:,} samples | Test: {n_test:,} samples")

        # NaN/Inf cleanup
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=10.0, neginf=-10.0)

        # ---- Train 4 XGBoost models ----
        t0 = time.time()

        logger.info("  Training Gate (XGBClassifier)...")
        gate_model = train_gate_model(X_train, y_gate_train, X_test, y_gate_test,
                                      feature_names, xgb_params)

        logger.info("  Training TP (XGBRegressor)...")
        tp_model = train_regressor(X_train, y_tp_train, X_test, y_tp_test,
                                   "TP", xgb_params)

        logger.info("  Training SL (XGBRegressor)...")
        sl_model = train_regressor(X_train, y_sl_train, X_test, y_sl_test,
                                   "SL", xgb_params)

        logger.info("  Training Exit (XGBRegressor)...")
        exit_model = train_regressor(X_train, y_exit_train, X_test, y_exit_test,
                                     "Exit", xgb_params)

        train_time = time.time() - t0
        logger.info(f"  All 4 models trained in {train_time:.1f}s")

        # ---- OOT Predictions ----
        gate_probs = gate_model.predict_proba(X_test)[:, 1]  # P(trade)
        tp_preds = tp_model.predict(X_test)
        sl_preds = sl_model.predict(X_test)
        exit_preds = exit_model.predict(X_test)

        # Ensemble prediction for direction: use CNN + PatchTST mean
        ensemble_preds_10s = (test_date_data["cnn_preds"][:len(X_test), 2] +
                              test_date_data["ptst_preds"][:len(X_test), 2]) / 2.0

        # ---- Per-fold metrics ----
        # Unfiltered IC
        ic_10s_raw = compute_ic(ensemble_preds_10s, y_10s_test)
        logger.info(f"  Unfiltered IC_10s: {ic_10s_raw:.4f}")

        # Gate accuracy
        gate_pred_binary = (gate_probs > 0.5).astype(float)
        gate_acc = np.mean(gate_pred_binary == y_gate_test)
        gate_pred_pos = gate_pred_binary.sum()
        gate_precision = 0.0
        if gate_pred_pos > 0:
            gate_precision = float(((gate_pred_binary > 0) & (y_gate_test > 0)).sum() / gate_pred_pos)
        logger.info(f"  Gate: acc={gate_acc:.4f}, precision={gate_precision:.4f}, "
                    f"pred_rate={gate_pred_pos/len(gate_probs):.3f}")

        # TP/SL metrics
        tp_mae = np.abs(tp_preds - y_tp_test).mean()
        sl_mae = np.abs(sl_preds - y_sl_test).mean()
        logger.info(f"  TP MAE: {tp_mae:.3f} | SL MAE: {sl_mae:.3f}")

        # Threshold sweep for this fold
        fold_threshold_results = {}
        for thresh in GATE_THRESHOLDS:
            res = evaluate_at_threshold(gate_probs, ensemble_preds_10s, y_10s_test,
                                        labels_3h_test, thresh)
            fold_threshold_results[f"{thresh:.2f}"] = res
            if res["n_gated"] > 5:
                logger.info(f"  Gate@{thresh:.2f}: n={res['n_gated']:,}, "
                            f"cov={res['coverage']:.3f}, dir_acc={res['dir_accuracy']:.3f}, "
                            f"IC={res['ic_10s']:.4f}, PnL={res['total_pnl_ticks']:.1f}, "
                            f"WR={res['win_rate']:.3f}, Sortino={res['sortino']:.3f}")

        # Feature importance (gate model)
        gate_importance = get_top_features(gate_model, feature_names, top_n=15)
        logger.info(f"  Top gate features: {list(gate_importance.keys())[:5]}")

        # Accumulate for concat
        concat_gate_probs.append(gate_probs)
        concat_ensemble_preds.append(ensemble_preds_10s)
        concat_y_10s.append(y_10s_test)
        concat_labels_3h.append(labels_3h_test)
        concat_tp_preds.append(tp_preds)
        concat_sl_preds.append(sl_preds)
        concat_exit_preds.append(exit_preds)
        concat_tp_labels.append(y_tp_test)
        concat_sl_labels.append(y_sl_test)
        concat_exit_labels.append(y_exit_test)

        # Save fold predictions
        fold_pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
        np.savez_compressed(str(fold_pred_path),
            gate_probs=gate_probs,
            ensemble_preds_10s=ensemble_preds_10s,
            tp_preds=tp_preds,
            sl_preds=sl_preds,
            exit_preds=exit_preds,
            y_10s=y_10s_test,
            labels=labels_3h_test,
            gate_labels=y_gate_test,
            date=test_date_data["date"],
        )

        # Save models
        model_dir = output_dir / f"fold_{fold_idx:02d}_models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(gate_model, str(model_dir / "gate.joblib"))
        joblib.dump(tp_model, str(model_dir / "tp.joblib"))
        joblib.dump(sl_model, str(model_dir / "sl.joblib"))
        joblib.dump(exit_model, str(model_dir / "exit.joblib"))

        fold_result = {
            "fold": fold_idx,
            "test_date": test_date_data["date"],
            "n_train": n_train,
            "n_test": n_test,
            "train_time_s": round(train_time, 1),
            "ic_10s_raw": ic_10s_raw,
            "gate_accuracy": float(gate_acc),
            "gate_precision": float(gate_precision),
            "tp_mae": float(tp_mae),
            "sl_mae": float(sl_mae),
            "threshold_results": fold_threshold_results,
            "gate_top_features": gate_importance,
        }
        fold_results.append(fold_result)

        # MLflow per-fold
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    f"fold_{fold_idx}_ic_10s": ic_10s_raw if not np.isnan(ic_10s_raw) else 0,
                    f"fold_{fold_idx}_gate_acc": float(gate_acc),
                    f"fold_{fold_idx}_gate_precision": float(gate_precision),
                    f"fold_{fold_idx}_tp_mae": float(tp_mae),
                    f"fold_{fold_idx}_sl_mae": float(sl_mae),
                })
                # Log best threshold PnL
                for thresh in [0.50, 0.65, 0.75]:
                    key = f"{thresh:.2f}"
                    if key in fold_threshold_results and fold_threshold_results[key].get("n_gated", 0) > 5:
                        mlflow.log_metrics({
                            f"fold_{fold_idx}_pnl_t{int(thresh*100)}": fold_threshold_results[key]["total_pnl_ticks"],
                            f"fold_{fold_idx}_sortino_t{int(thresh*100)}": fold_threshold_results[key]["sortino"],
                        })
                mlflow.log_artifact(str(fold_pred_path))
            except Exception as e:
                logger.warning(f"MLflow logging error: {e}")

        # Cleanup
        del gate_model, tp_model, sl_model, exit_model
        del X_train, X_test
        gc.collect()

    # ============================================================
    # Concat Metrics (primary evaluation)
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"CONCAT METRICS -- {n_folds} folds, {sum(len(g) for g in concat_gate_probs):,} total samples")
    logger.info(f"{'='*60}")

    all_gate_probs = np.concatenate(concat_gate_probs)
    all_ensemble_preds = np.concatenate(concat_ensemble_preds)
    all_y_10s = np.concatenate(concat_y_10s)
    all_labels_3h = np.concatenate(concat_labels_3h)
    all_tp_preds = np.concatenate(concat_tp_preds)
    all_sl_preds = np.concatenate(concat_sl_preds)
    all_exit_preds = np.concatenate(concat_exit_preds)
    all_tp_labels = np.concatenate(concat_tp_labels)
    all_sl_labels = np.concatenate(concat_sl_labels)
    all_exit_labels = np.concatenate(concat_exit_labels)

    # Unfiltered concat IC
    concat_ic_10s = compute_ic(all_ensemble_preds, all_y_10s)
    logger.info(f"  Concat IC_10s (unfiltered): {concat_ic_10s:.4f}")

    # Per-horizon IC
    for h, name in enumerate(["1s", "5s", "10s"]):
        ic_h = compute_ic(all_labels_3h[:, h], all_labels_3h[:, h])  # self
        logger.info(f"  (Reference) labels_{name} range: [{all_labels_3h[:, h].min():.3f}, {all_labels_3h[:, h].max():.3f}]")

    # TP/SL concat metrics
    tp_concat_mae = np.abs(all_tp_preds - all_tp_labels).mean()
    sl_concat_mae = np.abs(all_sl_preds - all_sl_labels).mean()
    tp_ic = compute_ic(all_tp_preds, all_tp_labels)
    sl_ic = compute_ic(all_sl_preds, all_sl_labels)
    logger.info(f"  TP concat: MAE={tp_concat_mae:.3f}, IC={tp_ic:.4f}")
    logger.info(f"  SL concat: MAE={sl_concat_mae:.3f}, IC={sl_ic:.4f}")

    # ---- Threshold sweep on concat ----
    logger.info(f"\n  {'='*50}")
    logger.info(f"  GATE THRESHOLD SWEEP (concat)")
    logger.info(f"  {'='*50}")

    concat_threshold_results = {}
    best_sortino = -999
    best_threshold = 0.5

    for thresh in GATE_THRESHOLDS:
        res = evaluate_at_threshold(all_gate_probs, all_ensemble_preds,
                                    all_y_10s, all_labels_3h, thresh)
        concat_threshold_results[f"{thresh:.2f}"] = res

        if res["n_gated"] > 5:
            logger.info(f"  Gate@{thresh:.2f}: n={res['n_gated']:>6,} | "
                        f"cov={res['coverage']:.3f} | "
                        f"dir_acc={res['dir_accuracy']:.3f} | "
                        f"IC_10s={res['ic_10s']:.4f} | "
                        f"PnL={res['total_pnl_ticks']:>8.1f} | "
                        f"WR={res['win_rate']:.3f} | "
                        f"Sortino={res['sortino']:.3f}")

            if res["sortino"] > best_sortino and res["n_gated"] > 100:
                best_sortino = res["sortino"]
                best_threshold = thresh
        else:
            logger.info(f"  Gate@{thresh:.2f}: n={res['n_gated']:>6} (too few)")

    logger.info(f"\n  BEST threshold: {best_threshold:.2f} "
                f"(Sortino={best_sortino:.3f})")

    # MLflow concat metrics
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics({
                "concat_ic_10s": concat_ic_10s if not np.isnan(concat_ic_10s) else 0,
                "concat_tp_mae": float(tp_concat_mae),
                "concat_sl_mae": float(sl_concat_mae),
                "concat_tp_ic": tp_ic if not np.isnan(tp_ic) else 0,
                "concat_sl_ic": sl_ic if not np.isnan(sl_ic) else 0,
                "best_gate_threshold": best_threshold,
                "best_sortino": best_sortino if best_sortino > -999 else 0,
                "n_total_samples": len(all_gate_probs),
            })
            for thresh in GATE_THRESHOLDS:
                key = f"{thresh:.2f}"
                if key in concat_threshold_results:
                    r = concat_threshold_results[key]
                    t_key = int(thresh * 100)
                    if r.get("n_gated", 0) > 5:
                        mlflow.log_metrics({
                            f"concat_pnl_t{t_key}": r["total_pnl_ticks"],
                            f"concat_sortino_t{t_key}": r["sortino"],
                            f"concat_coverage_t{t_key}": r["coverage"],
                            f"concat_wr_t{t_key}": r["win_rate"],
                            f"concat_ic_t{t_key}": r.get("ic_10s", 0),
                        })
        except Exception as e:
            logger.warning(f"MLflow concat logging error: {e}")

    # Save concat predictions
    concat_path = output_dir / "concat_oot_predictions.npz"
    np.savez_compressed(str(concat_path),
        gate_probs=all_gate_probs,
        ensemble_preds_10s=all_ensemble_preds,
        tp_preds=all_tp_preds,
        sl_preds=all_sl_preds,
        exit_preds=all_exit_preds,
        y_10s=all_y_10s,
        labels=all_labels_3h,
        tp_labels=all_tp_labels,
        sl_labels=all_sl_labels,
        exit_labels=all_exit_labels,
    )
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(concat_path))
        except Exception:
            pass

    results = {
        "n_folds": n_folds,
        "concat_ic_10s": concat_ic_10s,
        "concat_tp_mae": float(tp_concat_mae),
        "concat_sl_mae": float(sl_concat_mae),
        "concat_tp_ic": tp_ic,
        "concat_sl_ic": sl_ic,
        "best_gate_threshold": best_threshold,
        "best_sortino": best_sortino,
        "concat_threshold_results": concat_threshold_results,
        "per_fold": fold_results,
    }

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost Execution Combiner -- 3-model ensemble",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument("--cnn-mamba-dir", type=str, default=None)
    parser.add_argument("--patchtst-dir", type=str, default=None)
    parser.add_argument("--vol-lgbm-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    # Walk-forward
    parser.add_argument("--train-days", type=int, default=7)
    parser.add_argument("--dates", type=str, nargs="*", default=None)

    # XGBoost
    parser.add_argument("--xgb-depth", type=int, default=6)
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)

    # Labels
    parser.add_argument("--gate-threshold", type=float, default=0.5,
                        help="Ticks threshold for gate labels: |y_10s| > this")

    args = parser.parse_args()

    # Override data dirs
    global CNN_MAMBA_DIR, PATCHTST_DIR, VOL_LGBM_DIR
    if args.cnn_mamba_dir:
        CNN_MAMBA_DIR = Path(args.cnn_mamba_dir)
    if args.patchtst_dir:
        PATCHTST_DIR = Path(args.patchtst_dir)
    if args.vol_lgbm_dir:
        VOL_LGBM_DIR = Path(args.vol_lgbm_dir)

    ts = time.strftime("%Y%m%d_%H%M")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"XGBoost Execution Combiner v1 -- 3-Model Ensemble")
    logger.info(f"{'='*60}")
    logger.info(f"Device: CPU (Jupiter)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Data sources:")
    logger.info(f"  CNN-Mamba v2:  {CNN_MAMBA_DIR}")
    logger.info(f"  PatchTST:      {PATCHTST_DIR}")
    logger.info(f"  Vol LGBM v3:   {VOL_LGBM_DIR}")
    logger.info(f"Walk-forward: sliding {args.train_days}-day window")
    logger.info(f"XGBoost: depth={args.xgb_depth}, estimators={args.xgb_estimators}, "
                f"lr={args.xgb_lr}, subsample={args.xgb_subsample}, "
                f"colsample={args.xgb_colsample}")
    logger.info(f"Gate threshold (label): {args.gate_threshold} ticks")
    logger.info(f"Gate sweep: {GATE_THRESHOLDS}")

    # ---- Discover data ----
    logger.info(f"\nDiscovering data...")
    cnn_folds = discover_cnn_mamba_folds()
    ptst_folds = discover_patchtst_folds()
    vol_files = discover_vol_lgbm_files()

    logger.info(f"  CNN-Mamba: {len(cnn_folds)} dates: {sorted(cnn_folds.keys())}")
    logger.info(f"  PatchTST:  {len(ptst_folds)} dates: {sorted(ptst_folds.keys())}")
    logger.info(f"  Vol LGBM:  {len(vol_files)} dates: {sorted(vol_files.keys())}")

    # Determine overlap
    if args.dates:
        dates_to_use = args.dates
    else:
        cnn_dates = set(cnn_folds.keys())
        ptst_dates = set(ptst_folds.keys())
        overlap = sorted(cnn_dates & ptst_dates)
        if not overlap:
            overlap = [d for d in OVERLAP_DATES if d in cnn_folds and d in ptst_folds]
        dates_to_use = overlap

    logger.info(f"\nUsing {len(dates_to_use)} dates: {dates_to_use}")

    if len(dates_to_use) < args.train_days + 1:
        logger.error(f"Need at least {args.train_days + 1} dates, have {len(dates_to_use)}. Aborting.")
        sys.exit(1)

    # ---- Load all dates ----
    logger.info(f"\nLoading per-date aligned data...")
    all_dates = []
    for date in dates_to_use:
        data = load_date_data(date, cnn_folds, ptst_folds, vol_files)
        if data is not None:
            all_dates.append(data)
        else:
            logger.warning(f"Date {date}: skipped")

    if len(all_dates) < args.train_days + 1:
        logger.error(f"Only {len(all_dates)} dates loaded. Need {args.train_days + 1}. Aborting.")
        sys.exit(1)

    total_samples = sum(d["n_samples"] for d in all_dates)
    logger.info(f"Loaded {len(all_dates)} dates, {total_samples:,} total aligned samples")

    # ---- MLflow ----
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(run_name=f"exec_xgb_combiner_{ts}")
        mlflow.log_params({
            "model": "exec_xgb_combiner_3model",
            "n_dates": len(all_dates),
            "train_days": args.train_days,
            "xgb_depth": args.xgb_depth,
            "xgb_estimators": args.xgb_estimators,
            "xgb_lr": args.xgb_lr,
            "xgb_subsample": args.xgb_subsample,
            "xgb_colsample": args.xgb_colsample,
            "gate_label_threshold": args.gate_threshold,
            "total_samples": total_samples,
            "node": socket.gethostname(),
            "architecture": "XGBoost_4model_combiner",
            "models": "Gate_XGBClassifier+TP_XGBRegressor+SL_XGBRegressor+Exit_XGBRegressor",
            "feature_sources": "CNN-Mamba_embeds+preds,PatchTST_embeds+preds,VolLGBM_preds",
            "walk_forward": f"sliding_{args.train_days}d",
            "dates": ",".join([d["date"] for d in all_dates]),
        })

    try:
        results = train_walk_forward(all_dates, args, output_dir)

        # Save results summary
        summary = {
            "model": "exec_xgb_combiner_3model_v1",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dates_used": [d["date"] for d in all_dates],
            "args": vars(args),
            "results": results,
        }
        summary_path = output_dir / "results_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_artifact(str(summary_path))
            except Exception:
                pass

        logger.info(f"\nResults saved to {output_dir}")
        logger.info(f"Summary: {summary_path}")
        logger.info("DONE.")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
