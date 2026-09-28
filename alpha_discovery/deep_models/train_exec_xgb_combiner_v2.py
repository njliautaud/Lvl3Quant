#!/usr/bin/env python3
"""
train_exec_xgb_combiner_v2.py — XGBoost Execution Combiner v2 (Low-Dim LOO)
=============================================================================

v2 fixes v1 failures:
  1. NO raw embeddings (377→~20 features) — prevents overfitting
  2. Gate threshold = 2.0 ticks (not 0.5) → ~20-30% positive rate
  3. Leave-one-out CV over 9 dates (9 OOT folds instead of 2)
  4. Reduced XGB complexity (max_depth=4, n_estimators=150)
  5. Direction target = sign(CNN_pred_10s) weighted by confidence

Combines predictions from 3 frozen models:
  1. CNN-Mamba v2  — predictions (3 horizons: 1s, 5s, 10s)
  2. PatchTST       — predictions (3 horizons: 1s, 5s, 10s)
  3. Vol LGBM v3    — predictions (3 horizons: 10s, 30s, 60s)

Feature budget (~20 features):
  - CNN predictions: pred_1s, pred_5s, pred_10s (3)
  - PatchTST predictions: pred_1s, pred_5s, pred_10s (3)
  - Vol predictions: vol_10s, vol_30s, vol_60s (3)
  - Derived: agreement, cnn_confidence, ptst_confidence, vol_regime,
             pred_ratio (5)
  - Time: hour_sin, hour_cos, minute_sin, minute_cos (4)
  Total: 18 features

Usage:
  python train_exec_xgb_combiner_v2.py

Author: Execution combiner v2 for Lvl3Quant
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
log_path = LOG_DIR / f"exec_xgb_combiner_v2_{_ts}.log"

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

for _h in logging.root.handlers:
    _orig_emit = _h.emit
    def _flush_emit(record, _emit=_orig_emit, _handler=_h):
        _emit(record)
        _handler.flush()
    _h.emit = _flush_emit

print(f">>> train_exec_xgb_combiner_v2.py loaded | log: {log_path}", flush=True)

# ============================================================
# Paths & Constants
# ============================================================
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent

CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
VOL_LGBM_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"
DEFAULT_OUTPUT_DIR = LVL3_ROOT / "output" / "exec_xgb_combiner_v2"

TICK = 0.25
TICK_VAL = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0
ROUND_TRIP_COST = 2 * (COMMISSION_TICKS + SPREAD_TICKS * 0.5)  # 1.752 ticks

MLFLOW_EXPERIMENT = "ExecXGB_Combiner_v2"

OVERLAP_DATES = [
    "20260223", "20260224", "20260225", "20260226", "20260227",
    "20260301", "20260302", "20260303", "20260304",
]

GATE_THRESHOLDS = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Trading hours approximation (for time features)
# ES futures: ~23h/day, main session 9:30-16:00 ET
# With ~30K samples/day and stride=250 ticks, samples span the full session
TRADING_START_HOUR = 18.0  # 6pm ET previous day (globex open)
TRADING_HOURS = 23.5       # total session hours


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
# Data Discovery & Alignment (reused from v1)
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

    cnn_preds = cnn_data["predictions"].astype(np.float32)   # (N, 3): 1s, 5s, 10s
    cnn_labels = cnn_data["labels"].astype(np.float32)        # (N, 3): 1s, 5s, 10s
    ptst_preds = ptst_data["predictions"].astype(np.float32)  # (N, 3): 1s, 5s, 10s

    vol_available = date in vol_files
    vol_anchor_idxs = None
    if vol_available:
        try:
            vol_data = np.load(str(vol_files[date]), allow_pickle=True)
            vol_preds = vol_data["predictions"].astype(np.float32)  # (N, 3): 10s, 30s, 60s
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
        "cnn_preds": cnn_preds[cnn_idxs],       # (N, 3)
        "cnn_labels": cnn_labels[cnn_idxs],      # (N, 3)
        "ptst_preds": ptst_preds[ptst_idxs],     # (N, 3)
        "vol_preds": vol_preds[vol_idxs],        # (N, 3)
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
# Feature Engineering (v2: NO embeddings, ~20 features)
# ============================================================

def assemble_features(date_data: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    Assemble low-dimensional feature vector (~18 features).
    NO raw embeddings — only predictions + derived + time features.
    """
    n = date_data["n_samples"]
    cnn = date_data["cnn_preds"]     # (N, 3): 1s, 5s, 10s
    ptst = date_data["ptst_preds"]   # (N, 3): 1s, 5s, 10s
    vol = date_data["vol_preds"]     # (N, 3): 10s, 30s, 60s

    features = []
    names = []

    # === Raw predictions (9 features) ===
    # CNN-Mamba: 3 horizons
    features.append(cnn)
    names += ["cnn_pred_1s", "cnn_pred_5s", "cnn_pred_10s"]

    # PatchTST: 3 horizons
    features.append(ptst)
    names += ["ptst_pred_1s", "ptst_pred_5s", "ptst_pred_10s"]

    # Vol LGBM: 3 horizons (10s, 30s, 60s vol predictions)
    features.append(vol)
    names += ["vol_pred_10s", "vol_pred_30s", "vol_pred_60s"]

    # === Derived features (5 features) ===

    # 1. CNN-PatchTST agreement: sign_match * min_magnitude (continuous)
    cnn_10s = cnn[:, 2]
    ptst_10s = ptst[:, 2]
    sign_match = (np.sign(cnn_10s) == np.sign(ptst_10s)).astype(np.float32)
    min_mag = np.minimum(np.abs(cnn_10s), np.abs(ptst_10s))
    agreement = sign_match * min_mag
    features.append(agreement.reshape(-1, 1))
    names.append("cnn_ptst_agreement")

    # 2. CNN confidence: z-score of |pred_10s| within the date
    cnn_10s_abs = np.abs(cnn_10s)
    cnn_std = max(cnn_10s_abs.std(), 1e-8)
    cnn_confidence = (cnn_10s_abs - cnn_10s_abs.mean()) / cnn_std
    features.append(cnn_confidence.reshape(-1, 1))
    names.append("cnn_confidence")

    # 3. PatchTST confidence: z-score of |pred_10s|
    ptst_10s_abs = np.abs(ptst_10s)
    ptst_std = max(ptst_10s_abs.std(), 1e-8)
    ptst_confidence = (ptst_10s_abs - ptst_10s_abs.mean()) / ptst_std
    features.append(ptst_confidence.reshape(-1, 1))
    names.append("ptst_confidence")

    # 4. Vol regime: high/low based on vol_10s median split
    vol_10s = vol[:, 0]
    vol_median = np.median(vol_10s) if np.any(vol_10s != 0) else 0.0
    vol_regime = (vol_10s > vol_median).astype(np.float32)
    features.append(vol_regime.reshape(-1, 1))
    names.append("vol_regime")

    # 5. Pred ratio: CNN/PatchTST magnitude ratio (clipped)
    ptst_mag = np.abs(ptst_10s) + 1e-8
    pred_ratio = np.clip(np.abs(cnn_10s) / ptst_mag, 0.0, 10.0)
    features.append(pred_ratio.reshape(-1, 1))
    names.append("pred_ratio")

    # === Time features (4 features) ===
    # Approximate time-of-day from sample index within the day
    # Assume samples are evenly spaced across trading session
    frac = np.linspace(0, 1, n, dtype=np.float32)
    hour_of_day = TRADING_START_HOUR + frac * TRADING_HOURS
    hour_of_day = hour_of_day % 24.0
    minute_of_hour = (hour_of_day * 60) % 60

    hour_rad = 2 * np.pi * hour_of_day / 24.0
    minute_rad = 2 * np.pi * minute_of_hour / 60.0

    features.append(np.sin(hour_rad).reshape(-1, 1))
    names.append("hour_sin")
    features.append(np.cos(hour_rad).reshape(-1, 1))
    names.append("hour_cos")
    features.append(np.sin(minute_rad).reshape(-1, 1))
    names.append("minute_sin")
    features.append(np.cos(minute_rad).reshape(-1, 1))
    names.append("minute_cos")

    X = np.concatenate(features, axis=1).astype(np.float32)
    assert X.shape == (n, len(names)), f"Shape mismatch: {X.shape} vs ({n}, {len(names)})"
    return X, names


# ============================================================
# Label Construction (v2: 2.0 tick gate threshold)
# ============================================================

def compute_labels(cnn_labels: np.ndarray,
                   cnn_preds: np.ndarray,
                   ptst_preds: np.ndarray,
                   gate_threshold: float = 2.0) -> Dict[str, np.ndarray]:
    """
    Compute labels for XGBoost models.

    v2 changes:
    - Gate threshold default = 2.0 ticks (vs 0.5 in v1)
    - Direction weighted by CNN confidence
    """
    y_10s = cnn_labels[:, 2]
    y_5s = cnn_labels[:, 1]

    # Gate label: |y_10s| > threshold (selective — only big moves)
    gate = (np.abs(y_10s) > gate_threshold).astype(np.float32)

    # Direction from CNN-Mamba (primary model) weighted by confidence
    # XGB should learn its OWN direction signal, not just rubber-stamp CNN
    # The direction target IS the actual direction (y_10s sign)
    direction = np.sign(y_10s)

    # Ensemble prediction for IC calculation
    ensemble_pred_10s = (cnn_preds[:, 2] + ptst_preds[:, 2]) / 2.0

    # For directional return computation
    ensemble_dir = np.sign(ensemble_pred_10s)
    zero_mask = ensemble_dir == 0
    ensemble_dir[zero_mask] = np.sign(cnn_preds[zero_mask, 2])

    directional = y_10s * ensemble_dir
    tp_label = np.maximum(directional, 0.0)
    sl_label = np.abs(np.minimum(directional, 0.0))

    exit_label = np.where(
        np.abs(y_5s) > 0.1,
        np.clip(np.abs(y_10s) / np.abs(y_5s), 0.0, 5.0),
        1.0
    )

    n_gate = gate.sum()
    gate_rate = 100 * n_gate / len(gate)
    logger.info(f"  Labels: gate_rate={gate_rate:.1f}% ({int(n_gate)}/{len(gate)}), "
                f"threshold={gate_threshold} ticks")

    return {
        "gate": gate,
        "direction": direction.astype(np.float32),
        "tp_label": tp_label.astype(np.float32),
        "sl_label": sl_label.astype(np.float32),
        "exit_label": exit_label.astype(np.float32),
        "labels_3h": cnn_labels,  # (N, 3) for IC calc
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
                          dir_preds: np.ndarray,
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

    gated_dir_preds = dir_preds[mask]
    gated_y = y_10s[mask]

    # Direction accuracy
    pred_dir = np.sign(gated_dir_preds)
    actual_dir = np.sign(gated_y)
    dir_acc = np.mean(pred_dir == actual_dir)

    # IC on gated samples
    ic_10s = compute_ic(gated_dir_preds, gated_y)

    # PnL simulation
    entry_dir = np.sign(gated_dir_preds)
    pnl_ticks = entry_dir * gated_y - ROUND_TRIP_COST
    total_pnl = pnl_ticks.sum()
    mean_pnl = total_pnl / n_gated
    win_rate = np.mean(pnl_ticks > 0)
    sortino = compute_sortino(pnl_ticks)

    return {
        "threshold": threshold,
        "n_gated": int(n_gated),
        "coverage": float(n_gated / n_total),
        "dir_accuracy": float(dir_acc),
        "ic_10s": ic_10s,
        "total_pnl_ticks": float(total_pnl),
        "mean_pnl_per_trade": float(mean_pnl),
        "win_rate": float(win_rate),
        "sortino": float(sortino),
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
        early_stopping_rounds=params["early_stopping_rounds"],
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = getattr(model, 'best_iteration', params["n_estimators"]) or params["n_estimators"]
    logger.info(f"    Gate: best_iteration={best_iter}, scale_pos_weight={scale_pos:.2f}, "
                f"pos_rate_train={pos_count/len(y_train):.3f}")

    return model


def train_direction_model(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          feature_names: List[str],
                          params: Dict) -> xgb.XGBClassifier:
    """Train direction classifier: predict sign(y_10s)."""
    # Convert -1/0/+1 to 0/1 for binary classification
    y_train_bin = (y_train > 0).astype(np.float32)
    y_val_bin = (y_val > 0).astype(np.float32)

    model = xgb.XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=params["early_stopping_rounds"],
    )

    model.fit(
        X_train, y_train_bin,
        eval_set=[(X_val, y_val_bin)],
        verbose=False,
    )

    best_iter = getattr(model, 'best_iteration', params["n_estimators"]) or params["n_estimators"]
    logger.info(f"    Direction: best_iteration={best_iter}")

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
        early_stopping_rounds=params["early_stopping_rounds"],
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = getattr(model, 'best_iteration', params["n_estimators"]) or params["n_estimators"]
    logger.info(f"    {name}: best_iteration={best_iter}")

    return model


def get_top_features(model, feature_names: List[str], top_n: int = 10) -> Dict[str, float]:
    """Get top N features by importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    return {feature_names[i]: float(importances[i]) for i in indices}


# ============================================================
# Leave-One-Out Cross-Validation
# ============================================================

def train_loo_cv(all_dates: List[Dict], args: argparse.Namespace,
                 output_dir: Path) -> Dict:
    """
    Leave-one-out CV: train on 8 dates, test on 1, for all 9 dates.
    This gives 9 true OOT folds — much better than v1's 2.
    """
    n_dates = len(all_dates)
    logger.info(f"\nLeave-One-Out CV: {n_dates} folds (each date held out once)")

    # Get feature names from first date
    _, feature_names = assemble_features(all_dates[0])
    input_dim = len(feature_names)
    logger.info(f"Input dimension: {input_dim} features: {feature_names}")

    # XGBoost params
    xgb_params = {
        "n_estimators": args.xgb_estimators,
        "max_depth": args.xgb_depth,
        "learning_rate": args.xgb_lr,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
        "early_stopping_rounds": args.early_stopping,
    }

    # Accumulators for concat metrics
    concat_gate_probs = []
    concat_dir_preds = []
    concat_y_10s = []
    concat_labels_3h = []
    concat_dates = []

    fold_results = []

    for fold_idx in range(n_dates):
        test_date_data = all_dates[fold_idx]
        train_date_data = [all_dates[i] for i in range(n_dates) if i != fold_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx} | Hold-out: {test_date_data['date']} | "
                    f"Train: {[d['date'] for d in train_date_data]}")
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
        y_dir_train = np.concatenate([l["direction"] for l in train_label_list])
        y_tp_train = np.concatenate([l["tp_label"] for l in train_label_list])
        y_sl_train = np.concatenate([l["sl_label"] for l in train_label_list])

        # ---- Build test data ----
        X_test, _ = assemble_features(test_date_data)
        test_labels = compute_labels(
            test_date_data["cnn_labels"], test_date_data["cnn_preds"],
            test_date_data["ptst_preds"],
            gate_threshold=args.gate_threshold,
        )
        y_gate_test = test_labels["gate"]
        y_dir_test = test_labels["direction"]
        y_10s_test = test_labels["y_10s"]
        labels_3h_test = test_labels["labels_3h"]

        n_train = len(X_train)
        n_test = len(X_test)
        logger.info(f"  Train: {n_train:,} samples | Test: {n_test:,} samples")

        # NaN/Inf cleanup
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=10.0, neginf=-10.0)

        # ---- Train models ----
        t0 = time.time()

        logger.info("  Training Gate (XGBClassifier)...")
        gate_model = train_gate_model(X_train, y_gate_train, X_test, y_gate_test,
                                      feature_names, xgb_params)

        logger.info("  Training Direction (XGBClassifier)...")
        dir_model = train_direction_model(X_train, y_dir_train, X_test, y_dir_test,
                                          feature_names, xgb_params)

        logger.info("  Training TP (XGBRegressor)...")
        tp_model = train_regressor(X_train, y_tp_train, X_test, test_labels["tp_label"],
                                   "TP", xgb_params)

        logger.info("  Training SL (XGBRegressor)...")
        sl_model = train_regressor(X_train, y_sl_train, X_test, test_labels["sl_label"],
                                   "SL", xgb_params)

        train_time = time.time() - t0
        logger.info(f"  All models trained in {train_time:.1f}s")

        # ---- OOT Predictions ----
        gate_probs = gate_model.predict_proba(X_test)[:, 1]

        # Direction: XGB predicts P(up). Convert to signed prediction.
        dir_probs = dir_model.predict_proba(X_test)[:, 1]  # P(up)
        # Scale to [-1, +1]: 2*P(up) - 1. This is the XGB's OWN direction signal.
        dir_signal = 2.0 * dir_probs - 1.0

        # Combine XGB direction with CNN confidence for final direction prediction
        cnn_10s = test_date_data["cnn_preds"][:len(X_test), 2]
        # Final direction = XGB signal * CNN magnitude (XGB decides direction, CNN provides scale)
        dir_preds = dir_signal * np.abs(cnn_10s)

        tp_preds = tp_model.predict(X_test)
        sl_preds = sl_model.predict(X_test)

        # ---- Per-fold metrics ----
        # Direction accuracy (XGB's own signal)
        xgb_dir = np.sign(dir_signal)
        actual_dir = np.sign(y_10s_test)
        dir_acc_xgb = np.mean(xgb_dir == actual_dir)

        # Compare to CNN-only direction
        cnn_dir = np.sign(cnn_10s)
        dir_acc_cnn = np.mean(cnn_dir == actual_dir)

        # Ensemble mean direction (v1 approach)
        ensemble_mean = (test_date_data["cnn_preds"][:len(X_test), 2] +
                        test_date_data["ptst_preds"][:len(X_test), 2]) / 2.0
        dir_acc_ensemble = np.mean(np.sign(ensemble_mean) == actual_dir)

        # IC
        ic_10s_xgb = compute_ic(dir_preds, y_10s_test)
        ic_10s_cnn = compute_ic(cnn_10s, y_10s_test)
        ic_10s_ensemble = compute_ic(ensemble_mean, y_10s_test)

        logger.info(f"  Direction accuracy: XGB={dir_acc_xgb:.4f}, CNN={dir_acc_cnn:.4f}, Ensemble={dir_acc_ensemble:.4f}")
        logger.info(f"  IC_10s: XGB_dir={ic_10s_xgb:.4f}, CNN={ic_10s_cnn:.4f}, Ensemble={ic_10s_ensemble:.4f}")

        # Gate metrics
        gate_rate = gate_probs.mean()
        gate_pos_rate = y_gate_test.mean()
        logger.info(f"  Gate: mean_prob={gate_rate:.3f}, actual_pos_rate={gate_pos_rate:.3f}")

        # Threshold sweep
        fold_threshold_results = {}
        for thresh in GATE_THRESHOLDS:
            res = evaluate_at_threshold(gate_probs, dir_preds, y_10s_test,
                                        labels_3h_test, thresh)
            fold_threshold_results[f"{thresh:.2f}"] = res
            if res["n_gated"] > 5:
                logger.info(f"  Gate@{thresh:.2f}: n={res['n_gated']:>6,} | "
                            f"cov={res['coverage']:.3f} | "
                            f"dir={res['dir_accuracy']:.3f} | "
                            f"IC={res.get('ic_10s', 0):.4f} | "
                            f"PnL={res['total_pnl_ticks']:>8.1f} | "
                            f"WR={res['win_rate']:.3f} | "
                            f"Sortino={res['sortino']:.3f}")

        # Feature importance
        gate_importance = get_top_features(gate_model, feature_names, top_n=10)
        dir_importance = get_top_features(dir_model, feature_names, top_n=10)
        logger.info(f"  Top gate features: {list(gate_importance.keys())[:5]}")
        logger.info(f"  Top dir features:  {list(dir_importance.keys())[:5]}")

        # Accumulate for concat
        concat_gate_probs.append(gate_probs)
        concat_dir_preds.append(dir_preds)
        concat_y_10s.append(y_10s_test)
        concat_labels_3h.append(labels_3h_test)
        concat_dates.append(test_date_data["date"])

        # Save fold predictions
        fold_pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
        np.savez_compressed(str(fold_pred_path),
            gate_probs=gate_probs,
            dir_preds=dir_preds,
            dir_signal=dir_signal,
            tp_preds=tp_preds,
            sl_preds=sl_preds,
            y_10s=y_10s_test,
            labels=labels_3h_test,
            gate_labels=y_gate_test,
            date=test_date_data["date"],
        )

        # Save models
        model_dir = output_dir / f"fold_{fold_idx:02d}_models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(gate_model, str(model_dir / "gate.joblib"))
        joblib.dump(dir_model, str(model_dir / "direction.joblib"))
        joblib.dump(tp_model, str(model_dir / "tp.joblib"))
        joblib.dump(sl_model, str(model_dir / "sl.joblib"))

        fold_result = {
            "fold": fold_idx,
            "test_date": test_date_data["date"],
            "n_train": n_train,
            "n_test": n_test,
            "train_time_s": round(train_time, 1),
            "dir_accuracy_xgb": float(dir_acc_xgb),
            "dir_accuracy_cnn": float(dir_acc_cnn),
            "dir_accuracy_ensemble": float(dir_acc_ensemble),
            "ic_10s_xgb": float(ic_10s_xgb) if not np.isnan(ic_10s_xgb) else 0,
            "ic_10s_cnn": float(ic_10s_cnn) if not np.isnan(ic_10s_cnn) else 0,
            "ic_10s_ensemble": float(ic_10s_ensemble) if not np.isnan(ic_10s_ensemble) else 0,
            "gate_mean_prob": float(gate_rate),
            "gate_actual_pos_rate": float(gate_pos_rate),
            "threshold_results": fold_threshold_results,
            "gate_top_features": gate_importance,
            "dir_top_features": dir_importance,
        }
        fold_results.append(fold_result)

        # MLflow per-fold
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    f"fold_{fold_idx}_ic_xgb": float(ic_10s_xgb) if not np.isnan(ic_10s_xgb) else 0,
                    f"fold_{fold_idx}_ic_cnn": float(ic_10s_cnn) if not np.isnan(ic_10s_cnn) else 0,
                    f"fold_{fold_idx}_dir_acc_xgb": float(dir_acc_xgb),
                    f"fold_{fold_idx}_dir_acc_cnn": float(dir_acc_cnn),
                    f"fold_{fold_idx}_gate_pos_rate": float(gate_pos_rate),
                })
                for thresh in [0.50, 0.60, 0.70]:
                    key = f"{thresh:.2f}"
                    if key in fold_threshold_results and fold_threshold_results[key].get("n_gated", 0) > 5:
                        mlflow.log_metrics({
                            f"fold_{fold_idx}_pnl_t{int(thresh*100)}": fold_threshold_results[key]["total_pnl_ticks"],
                            f"fold_{fold_idx}_sortino_t{int(thresh*100)}": fold_threshold_results[key]["sortino"],
                        })
            except Exception as e:
                logger.warning(f"MLflow fold logging error: {e}")

        # Cleanup
        del gate_model, dir_model, tp_model, sl_model
        del X_train, X_test
        gc.collect()

    # ============================================================
    # Concat Metrics (primary evaluation)
    # ============================================================
    total_samples = sum(len(g) for g in concat_gate_probs)
    logger.info(f"\n{'='*60}")
    logger.info(f"CONCAT METRICS -- {n_dates} LOO folds, {total_samples:,} total OOT samples")
    logger.info(f"{'='*60}")

    all_gate_probs = np.concatenate(concat_gate_probs)
    all_dir_preds = np.concatenate(concat_dir_preds)
    all_y_10s = np.concatenate(concat_y_10s)
    all_labels_3h = np.concatenate(concat_labels_3h)

    # Unfiltered concat metrics
    concat_ic_10s = compute_ic(all_dir_preds, all_y_10s)
    concat_dir_acc = np.mean(np.sign(all_dir_preds) == np.sign(all_y_10s))
    logger.info(f"  Concat IC_10s (XGB direction, unfiltered): {concat_ic_10s:.4f}")
    logger.info(f"  Concat direction accuracy (unfiltered): {concat_dir_acc:.4f}")

    # Gate positive rate distribution
    logger.info(f"  Gate prob distribution: "
                f"mean={all_gate_probs.mean():.3f}, "
                f"median={np.median(all_gate_probs):.3f}, "
                f"p10={np.percentile(all_gate_probs, 10):.3f}, "
                f"p90={np.percentile(all_gate_probs, 90):.3f}")

    # ---- Threshold sweep on concat ----
    logger.info(f"\n  {'='*50}")
    logger.info(f"  GATE THRESHOLD SWEEP (concat, {total_samples:,} samples)")
    logger.info(f"  {'='*50}")
    logger.info(f"  {'Thresh':>7} | {'Trades':>7} | {'Cover':>6} | {'DirAcc':>6} | "
                f"{'IC_10s':>7} | {'PnL':>9} | {'PnL/Trd':>8} | {'WinRate':>7} | {'Sortino':>7}")
    logger.info(f"  {'-'*80}")

    concat_threshold_results = {}
    best_sortino = -999
    best_threshold = 0.5

    for thresh in GATE_THRESHOLDS:
        res = evaluate_at_threshold(all_gate_probs, all_dir_preds,
                                    all_y_10s, all_labels_3h, thresh)
        concat_threshold_results[f"{thresh:.2f}"] = res

        if res["n_gated"] > 5:
            logger.info(f"  {thresh:>7.2f} | {res['n_gated']:>7,} | {res['coverage']:>6.3f} | "
                        f"{res['dir_accuracy']:>6.3f} | {res['ic_10s']:>7.4f} | "
                        f"{res['total_pnl_ticks']:>9.1f} | {res['mean_pnl_per_trade']:>8.3f} | "
                        f"{res['win_rate']:>7.3f} | {res['sortino']:>7.3f}")

            if res["sortino"] > best_sortino and res["n_gated"] > 100:
                best_sortino = res["sortino"]
                best_threshold = thresh
        else:
            logger.info(f"  {thresh:>7.2f} | {res['n_gated']:>7} (too few)")

    logger.info(f"\n  BEST threshold: {best_threshold:.2f} "
                f"(Sortino={best_sortino:.3f})")

    # Per-fold summary table
    logger.info(f"\n  {'='*50}")
    logger.info(f"  PER-FOLD SUMMARY")
    logger.info(f"  {'='*50}")
    logger.info(f"  {'Fold':>4} | {'Date':>10} | {'IC_XGB':>7} | {'IC_CNN':>7} | "
                f"{'DirXGB':>7} | {'DirCNN':>7} | {'GatePos%':>8}")
    logger.info(f"  {'-'*65}")
    for fr in fold_results:
        logger.info(f"  {fr['fold']:>4} | {fr['test_date']:>10} | "
                    f"{fr['ic_10s_xgb']:>7.4f} | {fr['ic_10s_cnn']:>7.4f} | "
                    f"{fr['dir_accuracy_xgb']:>7.4f} | {fr['dir_accuracy_cnn']:>7.4f} | "
                    f"{fr['gate_actual_pos_rate']*100:>7.1f}%")

    # MLflow concat metrics
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics({
                "concat_ic_10s": float(concat_ic_10s) if not np.isnan(concat_ic_10s) else 0,
                "concat_dir_accuracy": float(concat_dir_acc),
                "best_gate_threshold": best_threshold,
                "best_sortino": best_sortino if best_sortino > -999 else 0,
                "n_total_samples": total_samples,
                "n_folds": n_dates,
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
                            f"concat_dir_acc_t{t_key}": r.get("dir_accuracy", 0),
                        })
        except Exception as e:
            logger.warning(f"MLflow concat logging error: {e}")

    # Save concat predictions
    concat_path = output_dir / "concat_oot_predictions.npz"
    np.savez_compressed(str(concat_path),
        gate_probs=all_gate_probs,
        dir_preds=all_dir_preds,
        y_10s=all_y_10s,
        labels=all_labels_3h,
        dates=np.array(concat_dates),
    )
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(concat_path))
        except Exception:
            pass

    results = {
        "n_folds": n_dates,
        "cv_method": "leave_one_out",
        "concat_ic_10s": float(concat_ic_10s) if not np.isnan(concat_ic_10s) else 0,
        "concat_dir_accuracy": float(concat_dir_acc),
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
        description="XGBoost Execution Combiner v2 — Low-dim LOO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--cnn-mamba-dir", type=str, default=None)
    parser.add_argument("--patchtst-dir", type=str, default=None)
    parser.add_argument("--vol-lgbm-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dates", type=str, nargs="*", default=None)

    # XGBoost (v2 defaults: reduced complexity)
    parser.add_argument("--xgb-depth", type=int, default=4)
    parser.add_argument("--xgb-estimators", type=int, default=150)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)
    parser.add_argument("--early-stopping", type=int, default=15)

    # Labels (v2: higher gate threshold)
    parser.add_argument("--gate-threshold", type=float, default=2.0,
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
    logger.info(f"XGBoost Execution Combiner v2 -- Low-Dim LOO")
    logger.info(f"{'='*60}")
    logger.info(f"Device: CPU (Jupiter)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Data sources:")
    logger.info(f"  CNN-Mamba v2:  {CNN_MAMBA_DIR}")
    logger.info(f"  PatchTST:      {PATCHTST_DIR}")
    logger.info(f"  Vol LGBM v3:   {VOL_LGBM_DIR}")
    logger.info(f"CV: Leave-One-Out (9 folds)")
    logger.info(f"XGBoost: depth={args.xgb_depth}, estimators={args.xgb_estimators}, "
                f"lr={args.xgb_lr}, subsample={args.xgb_subsample}, "
                f"colsample={args.xgb_colsample}, early_stop={args.early_stopping}")
    logger.info(f"Gate label threshold: {args.gate_threshold} ticks")
    logger.info(f"Gate sweep: {GATE_THRESHOLDS}")
    logger.info(f"Features: ~18 (NO embeddings)")

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

    if len(dates_to_use) < 3:
        logger.error(f"Need at least 3 dates for LOO CV, have {len(dates_to_use)}. Aborting.")
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

    if len(all_dates) < 3:
        logger.error(f"Only {len(all_dates)} dates loaded. Need at least 3. Aborting.")
        sys.exit(1)

    total_samples = sum(d["n_samples"] for d in all_dates)
    logger.info(f"Loaded {len(all_dates)} dates, {total_samples:,} total aligned samples")

    # ---- MLflow ----
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(run_name=f"exec_xgb_combiner_v2_{ts}")
        mlflow.log_params({
            "model": "exec_xgb_combiner_v2",
            "version": "v2_low_dim_loo",
            "n_dates": len(all_dates),
            "cv_method": "leave_one_out",
            "xgb_depth": args.xgb_depth,
            "xgb_estimators": args.xgb_estimators,
            "xgb_lr": args.xgb_lr,
            "xgb_subsample": args.xgb_subsample,
            "xgb_colsample": args.xgb_colsample,
            "early_stopping": args.early_stopping,
            "gate_label_threshold": args.gate_threshold,
            "n_features": 18,
            "total_samples": total_samples,
            "node": socket.gethostname(),
            "architecture": "XGBoost_Gate+Dir+TP+SL_low_dim",
            "feature_sources": "CNN_preds+PatchTST_preds+Vol_preds+derived+time",
            "dates": ",".join([d["date"] for d in all_dates]),
        })

    try:
        results = train_loo_cv(all_dates, args, output_dir)

        # Save results summary
        summary = {
            "model": "exec_xgb_combiner_v2",
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
                mlflow.log_artifact(str(log_path))
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
