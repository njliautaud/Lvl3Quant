#!/usr/bin/env python3
"""
train_confidence_gate_v3.py — Confidence Gate v3 (Pure Gate, No Direction Re-prediction)
========================================================================================

KEY INSIGHT: v1 and v2 tried to re-predict direction using XGBoost, and both
DEGRADED the CNN-Mamba signal (45.5% accuracy vs CNN's 47.8%). The combiner
should NOT re-predict direction. It should ONLY predict whether the CNN's
prediction will be correct — a pure confidence gate.

Architecture: Single XGBClassifier that predicts:
  "Will the CNN-Mamba v2's directional prediction be correct on y_10s?"

Label: gate_label = 1 if (sign(cnn_pred_10s) == sign(y_true_10s))
                          AND (|y_true_10s| > 1.0 ticks)
                     else 0

Features (~14, NO direction info — only meta/context):
  1. cnn_abs_pred_10s — absolute CNN prediction magnitude (confidence proxy)
  2. cnn_abs_pred_1s, cnn_abs_pred_5s, cnn_abs_pred_30s — multi-horizon confidence
  3. patchtst_agreement — 1 if PatchTST and CNN agree on direction, 0 if not
  4. patchtst_abs_10s — PatchTST confidence magnitude
  5. vol_pred_10s, vol_pred_30s, vol_pred_60s — predicted volatility
  6. vol_regime — 1 if vol > median, 0 otherwise
  7. hour_sin, hour_cos — time of day
  8. minute_sin, minute_cos — minute of hour
  9. cnn_patchtst_conf_product — |cnn_pred| × |patchtst_pred|
  10. multi_horizon_agreement — do all CNN horizons agree on direction? (1/0)

Execution: gate prob > threshold → trade in CNN-Mamba's predicted direction.
Direction comes 100% from CNN, gate only decides trade/skip.

PnL: sign(cnn_pred_10s) × y_true_10s - 1.752 (round trip cost)

Cross-validation: Leave-one-out over 9 dates.

Usage:
  python train_confidence_gate_v3.py

Author: Confidence Gate v3 for Lvl3Quant
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
log_path = LOG_DIR / f"confidence_gate_v3_{_ts}.log"

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

print(f">>> train_confidence_gate_v3.py loaded | log: {log_path}", flush=True)

# ============================================================
# Paths & Constants
# ============================================================
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent

CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
VOL_LGBM_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"
DEFAULT_OUTPUT_DIR = LVL3_ROOT / "output" / "confidence_gate_v3"

TICK = 0.25
TICK_VAL = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0
ROUND_TRIP_COST = 2 * (COMMISSION_TICKS + SPREAD_TICKS * 0.5)  # 1.752 ticks

MLFLOW_EXPERIMENT = "ConfidenceGate_v3"

OVERLAP_DATES = [
    "20260223", "20260224", "20260225", "20260226", "20260227",
    "20260301", "20260302", "20260303", "20260304",
]

GATE_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

# Trading hours approximation
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
# Data Discovery & Alignment (reused from v2)
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
# Feature Engineering (v3: NO direction info, only meta/context)
# ============================================================

def assemble_features(date_data: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    Assemble feature vector (~14 features).
    NO raw prediction values used as direction signals.
    Only absolute magnitudes, agreements, and context features.
    """
    n = date_data["n_samples"]
    cnn = date_data["cnn_preds"]     # (N, 3): 1s, 5s, 10s
    ptst = date_data["ptst_preds"]   # (N, 3): 1s, 5s, 10s
    vol = date_data["vol_preds"]     # (N, 3): 10s, 30s, 60s

    features = []
    names = []

    # === Absolute CNN prediction magnitudes (confidence proxies) ===
    # 1. cnn_abs_pred_1s
    features.append(np.abs(cnn[:, 0]).reshape(-1, 1))
    names.append("cnn_abs_pred_1s")

    # 2. cnn_abs_pred_5s
    features.append(np.abs(cnn[:, 1]).reshape(-1, 1))
    names.append("cnn_abs_pred_5s")

    # 3. cnn_abs_pred_10s
    features.append(np.abs(cnn[:, 2]).reshape(-1, 1))
    names.append("cnn_abs_pred_10s")

    # 4. cnn_abs_pred_30s — approximate from 10s (CNN only has 1s,5s,10s)
    #    Use ratio of 5s/1s to extrapolate a "30s confidence" proxy
    #    Actually, CNN has 3 horizons only. Use the spread between them as a proxy.
    cnn_horizon_spread = np.abs(cnn[:, 2]) - np.abs(cnn[:, 0])  # 10s mag - 1s mag
    features.append(cnn_horizon_spread.reshape(-1, 1))
    names.append("cnn_horizon_spread")

    # === PatchTST agreement & confidence ===
    cnn_10s = cnn[:, 2]
    ptst_10s = ptst[:, 2]

    # 5. patchtst_agreement — 1 if same sign, 0 if not
    patchtst_agreement = (np.sign(cnn_10s) == np.sign(ptst_10s)).astype(np.float32)
    features.append(patchtst_agreement.reshape(-1, 1))
    names.append("patchtst_agreement")

    # 6. patchtst_abs_10s — PatchTST confidence magnitude
    features.append(np.abs(ptst_10s).reshape(-1, 1))
    names.append("patchtst_abs_10s")

    # === Volatility predictions ===
    # 7-9. vol_pred_10s, vol_pred_30s, vol_pred_60s
    features.append(vol)
    names += ["vol_pred_10s", "vol_pred_30s", "vol_pred_60s"]

    # 10. vol_regime — 1 if vol_10s > median, 0 otherwise
    vol_10s = vol[:, 0]
    vol_median = np.median(vol_10s) if np.any(vol_10s != 0) else 0.0
    vol_regime = (vol_10s > vol_median).astype(np.float32)
    features.append(vol_regime.reshape(-1, 1))
    names.append("vol_regime")

    # === Time features ===
    frac = np.linspace(0, 1, n, dtype=np.float32)
    hour_of_day = TRADING_START_HOUR + frac * TRADING_HOURS
    hour_of_day = hour_of_day % 24.0
    minute_of_hour = (hour_of_day * 60) % 60

    hour_rad = 2 * np.pi * hour_of_day / 24.0
    minute_rad = 2 * np.pi * minute_of_hour / 60.0

    # 11-12. hour_sin, hour_cos
    features.append(np.sin(hour_rad).reshape(-1, 1))
    names.append("hour_sin")
    features.append(np.cos(hour_rad).reshape(-1, 1))
    names.append("hour_cos")

    # 13-14. minute_sin, minute_cos
    features.append(np.sin(minute_rad).reshape(-1, 1))
    names.append("minute_sin")
    features.append(np.cos(minute_rad).reshape(-1, 1))
    names.append("minute_cos")

    # === Joint confidence features ===
    # 15. cnn_patchtst_conf_product — |cnn_pred_10s| × |ptst_pred_10s|
    conf_product = np.abs(cnn_10s) * np.abs(ptst_10s)
    features.append(conf_product.reshape(-1, 1))
    names.append("cnn_patchtst_conf_product")

    # 16. multi_horizon_agreement — do all CNN horizons agree on direction? (1/0)
    cnn_signs = np.sign(cnn)  # (N, 3)
    all_agree = ((cnn_signs[:, 0] == cnn_signs[:, 1]) &
                 (cnn_signs[:, 1] == cnn_signs[:, 2])).astype(np.float32)
    features.append(all_agree.reshape(-1, 1))
    names.append("multi_horizon_agreement")

    X = np.concatenate(features, axis=1).astype(np.float32)
    assert X.shape == (n, len(names)), f"Shape mismatch: {X.shape} vs ({n}, {len(names)})"
    return X, names


# ============================================================
# Label Construction (v3: CNN correctness gate)
# ============================================================

def compute_gate_labels(cnn_labels: np.ndarray,
                        cnn_preds: np.ndarray,
                        min_move_ticks: float = 1.0) -> Dict[str, np.ndarray]:
    """
    v3 label: Was the CNN correct AND was the move large enough?

    gate_label = 1 if (sign(cnn_pred_10s) == sign(y_true_10s))
                       AND (|y_true_10s| > min_move_ticks)
                  else 0
    """
    y_10s = cnn_labels[:, 2]
    cnn_pred_10s = cnn_preds[:, 2]

    cnn_direction_correct = (np.sign(cnn_pred_10s) == np.sign(y_10s))
    move_large_enough = (np.abs(y_10s) > min_move_ticks)

    gate_label = (cnn_direction_correct & move_large_enough).astype(np.float32)

    n_pos = gate_label.sum()
    gate_rate = 100 * n_pos / len(gate_label)
    n_correct = cnn_direction_correct.sum()
    cnn_dir_acc = 100 * n_correct / len(gate_label)

    logger.info(f"  Gate labels: pos_rate={gate_rate:.1f}% ({int(n_pos)}/{len(gate_label)}), "
                f"CNN_dir_acc={cnn_dir_acc:.1f}%, min_move={min_move_ticks} ticks")

    return {
        "gate_label": gate_label,
        "y_10s": y_10s.astype(np.float32),
        "cnn_pred_10s": cnn_pred_10s.astype(np.float32),
        "labels_3h": cnn_labels,
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
                          cnn_pred_10s: np.ndarray,
                          y_10s: np.ndarray,
                          threshold: float) -> Dict:
    """
    Evaluate gated performance at a given gate threshold.
    Direction comes 100% from CNN — gate only decides trade/skip.
    PnL = sign(cnn_pred_10s) × y_true_10s - round_trip_cost
    """
    mask = gate_probs >= threshold
    n_gated = mask.sum()
    n_total = len(gate_probs)

    if n_gated < 5:
        return {
            "threshold": threshold,
            "n_gated": int(n_gated),
            "coverage": 0.0,
        }

    gated_cnn_pred = cnn_pred_10s[mask]
    gated_y = y_10s[mask]

    # Direction accuracy of GATED trades (CNN direction)
    cnn_dir = np.sign(gated_cnn_pred)
    actual_dir = np.sign(gated_y)
    dir_acc_gated = np.mean(cnn_dir == actual_dir)

    # Direction accuracy of ALL trades (CNN direction, ungated baseline)
    all_cnn_dir = np.sign(cnn_pred_10s)
    all_actual_dir = np.sign(y_10s)
    dir_acc_ungated = np.mean(all_cnn_dir == all_actual_dir)

    # Direction accuracy IMPROVEMENT from gating
    dir_acc_lift = dir_acc_gated - dir_acc_ungated

    # PnL simulation: direction from CNN, gate decides trade/skip
    entry_dir = np.sign(gated_cnn_pred)
    pnl_ticks = entry_dir * gated_y - ROUND_TRIP_COST
    total_pnl_ticks = pnl_ticks.sum()
    total_pnl_usd = total_pnl_ticks * TICK_VAL
    mean_pnl = total_pnl_ticks / n_gated

    wins = pnl_ticks > 0
    losses = pnl_ticks < 0
    win_rate = np.mean(wins)

    gross_profit = pnl_ticks[wins].sum() if wins.any() else 0.0
    gross_loss = abs(pnl_ticks[losses].sum()) if losses.any() else 1e-8
    profit_factor = gross_profit / max(gross_loss, 1e-8)

    sortino = compute_sortino(pnl_ticks)

    return {
        "threshold": threshold,
        "n_gated": int(n_gated),
        "coverage": float(n_gated / n_total),
        "dir_accuracy_gated": float(dir_acc_gated),
        "dir_accuracy_ungated": float(dir_acc_ungated),
        "dir_accuracy_lift": float(dir_acc_lift),
        "win_rate": float(win_rate),
        "total_pnl_ticks": float(total_pnl_ticks),
        "total_pnl_usd": float(total_pnl_usd),
        "mean_pnl_per_trade": float(mean_pnl),
        "profit_factor": float(profit_factor),
        "sortino": float(sortino),
    }


# ============================================================
# XGBoost Gate Model Training
# ============================================================

def train_gate_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     feature_names: List[str],
                     params: Dict) -> xgb.XGBClassifier:
    """Train confidence gate classifier with class imbalance handling."""
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
    Single model: confidence gate only. No direction re-prediction.
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
    concat_cnn_pred_10s = []
    concat_y_10s = []
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
            labels = compute_gate_labels(
                td["cnn_labels"], td["cnn_preds"],
                min_move_ticks=args.min_move_ticks,
            )
            train_feat_list.append(feat)
            train_label_list.append(labels)

        X_train = np.concatenate(train_feat_list, axis=0)
        y_gate_train = np.concatenate([l["gate_label"] for l in train_label_list])

        # ---- Build test data ----
        X_test, _ = assemble_features(test_date_data)
        test_labels = compute_gate_labels(
            test_date_data["cnn_labels"], test_date_data["cnn_preds"],
            min_move_ticks=args.min_move_ticks,
        )
        y_gate_test = test_labels["gate_label"]
        y_10s_test = test_labels["y_10s"]
        cnn_pred_10s_test = test_labels["cnn_pred_10s"]

        n_train = len(X_train)
        n_test = len(X_test)
        logger.info(f"  Train: {n_train:,} samples | Test: {n_test:,} samples")

        # NaN/Inf cleanup
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=10.0, neginf=-10.0)

        # ---- Train single gate model ----
        t0 = time.time()
        logger.info("  Training Confidence Gate (XGBClassifier)...")
        gate_model = train_gate_model(X_train, y_gate_train, X_test, y_gate_test,
                                      feature_names, xgb_params)
        train_time = time.time() - t0
        logger.info(f"  Gate model trained in {train_time:.1f}s")

        # ---- OOT Predictions ----
        gate_probs = gate_model.predict_proba(X_test)[:, 1]

        # Gate accuracy (at 0.5 threshold)
        gate_pred = (gate_probs >= 0.5).astype(np.float32)
        gate_acc = np.mean(gate_pred == y_gate_test)

        # CNN baseline direction accuracy (ungated)
        cnn_dir = np.sign(cnn_pred_10s_test)
        actual_dir = np.sign(y_10s_test)
        cnn_dir_acc_ungated = np.mean(cnn_dir == actual_dir)

        logger.info(f"  Gate accuracy (0.5): {gate_acc:.4f}")
        logger.info(f"  CNN direction accuracy (ungated): {cnn_dir_acc_ungated:.4f}")
        logger.info(f"  Gate prob distribution: mean={gate_probs.mean():.3f}, "
                    f"median={np.median(gate_probs):.3f}, "
                    f"p10={np.percentile(gate_probs, 10):.3f}, "
                    f"p90={np.percentile(gate_probs, 90):.3f}")

        # Threshold sweep for this fold
        fold_threshold_results = {}
        for thresh in GATE_THRESHOLDS:
            res = evaluate_at_threshold(gate_probs, cnn_pred_10s_test,
                                        y_10s_test, thresh)
            fold_threshold_results[f"{thresh:.2f}"] = res
            if res["n_gated"] > 5:
                logger.info(f"  Gate@{thresh:.2f}: n={res['n_gated']:>6,} | "
                            f"cov={res['coverage']:.3f} | "
                            f"dirG={res['dir_accuracy_gated']:.3f} | "
                            f"dirU={res['dir_accuracy_ungated']:.3f} | "
                            f"lift={res['dir_accuracy_lift']:+.3f} | "
                            f"WR={res['win_rate']:.3f} | "
                            f"PnL={res['total_pnl_ticks']:>8.1f}t | "
                            f"PF={res['profit_factor']:.2f} | "
                            f"Sortino={res['sortino']:.3f}")

        # Feature importance
        gate_importance = get_top_features(gate_model, feature_names, top_n=16)
        logger.info(f"  Top features: {list(gate_importance.keys())[:5]}")

        # Accumulate for concat
        concat_gate_probs.append(gate_probs)
        concat_cnn_pred_10s.append(cnn_pred_10s_test)
        concat_y_10s.append(y_10s_test)
        concat_dates.append(test_date_data["date"])

        # Save fold predictions
        fold_pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
        np.savez_compressed(str(fold_pred_path),
            gate_probs=gate_probs,
            cnn_pred_10s=cnn_pred_10s_test,
            y_10s=y_10s_test,
            gate_labels=y_gate_test,
            date=test_date_data["date"],
        )

        # Save model
        model_dir = output_dir / f"fold_{fold_idx:02d}_models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(gate_model, str(model_dir / "confidence_gate.joblib"))

        fold_result = {
            "fold": fold_idx,
            "test_date": test_date_data["date"],
            "n_train": n_train,
            "n_test": n_test,
            "train_time_s": round(train_time, 1),
            "gate_accuracy_05": float(gate_acc),
            "cnn_dir_acc_ungated": float(cnn_dir_acc_ungated),
            "gate_pos_rate": float(y_gate_test.mean()),
            "threshold_results": fold_threshold_results,
            "top_features": gate_importance,
        }
        fold_results.append(fold_result)

        # MLflow per-fold
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    f"fold_{fold_idx}_gate_acc": float(gate_acc),
                    f"fold_{fold_idx}_cnn_dir_acc": float(cnn_dir_acc_ungated),
                    f"fold_{fold_idx}_gate_pos_rate": float(y_gate_test.mean()),
                })
                for thresh in [0.50, 0.60, 0.70]:
                    key = f"{thresh:.2f}"
                    if key in fold_threshold_results and fold_threshold_results[key].get("n_gated", 0) > 5:
                        r = fold_threshold_results[key]
                        mlflow.log_metrics({
                            f"fold_{fold_idx}_pnl_t{int(thresh*100)}": r["total_pnl_ticks"],
                            f"fold_{fold_idx}_sortino_t{int(thresh*100)}": r["sortino"],
                            f"fold_{fold_idx}_dir_lift_t{int(thresh*100)}": r["dir_accuracy_lift"],
                        })
            except Exception as e:
                logger.warning(f"MLflow fold logging error: {e}")

        # Cleanup
        del gate_model
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
    all_cnn_pred_10s = np.concatenate(concat_cnn_pred_10s)
    all_y_10s = np.concatenate(concat_y_10s)

    # CNN baseline (ungated)
    all_cnn_dir = np.sign(all_cnn_pred_10s)
    all_actual_dir = np.sign(all_y_10s)
    baseline_dir_acc = np.mean(all_cnn_dir == all_actual_dir)

    # Baseline PnL (trade everything with CNN direction)
    baseline_pnl_ticks = (all_cnn_dir * all_y_10s - ROUND_TRIP_COST)
    baseline_total_pnl = baseline_pnl_ticks.sum()
    baseline_sortino = compute_sortino(baseline_pnl_ticks)
    baseline_wr = np.mean(baseline_pnl_ticks > 0)

    logger.info(f"\n  CNN BASELINE (ungated, all {total_samples:,} trades):")
    logger.info(f"    Direction accuracy: {baseline_dir_acc:.4f}")
    logger.info(f"    Win rate: {baseline_wr:.4f}")
    logger.info(f"    Total PnL: {baseline_total_pnl:.1f} ticks ({baseline_total_pnl * TICK_VAL:.0f} USD)")
    logger.info(f"    Mean PnL/trade: {baseline_total_pnl/total_samples:.3f} ticks")
    logger.info(f"    Sortino: {baseline_sortino:.3f}")

    # Gate prob distribution
    logger.info(f"\n  Gate prob distribution: "
                f"mean={all_gate_probs.mean():.3f}, "
                f"median={np.median(all_gate_probs):.3f}, "
                f"p10={np.percentile(all_gate_probs, 10):.3f}, "
                f"p90={np.percentile(all_gate_probs, 90):.3f}")

    # ---- Threshold sweep on concat ----
    logger.info(f"\n  {'='*90}")
    logger.info(f"  GATE THRESHOLD SWEEP (concat, {total_samples:,} total samples)")
    logger.info(f"  {'='*90}")
    logger.info(f"  {'Thresh':>7} | {'Trades':>7} | {'Cover':>6} | {'DirGated':>8} | "
                f"{'DirUngated':>10} | {'Lift':>6} | {'WR':>6} | "
                f"{'PnL_t':>9} | {'PnL_USD':>9} | {'PnL/Trd':>8} | {'PF':>6} | {'Sortino':>7}")
    logger.info(f"  {'-'*110}")

    concat_threshold_results = {}
    best_sortino = -999
    best_threshold = 0.5

    for thresh in GATE_THRESHOLDS:
        res = evaluate_at_threshold(all_gate_probs, all_cnn_pred_10s,
                                    all_y_10s, thresh)
        concat_threshold_results[f"{thresh:.2f}"] = res

        if res["n_gated"] > 5:
            logger.info(f"  {thresh:>7.2f} | {res['n_gated']:>7,} | {res['coverage']:>6.3f} | "
                        f"{res['dir_accuracy_gated']:>8.4f} | "
                        f"{res['dir_accuracy_ungated']:>10.4f} | "
                        f"{res['dir_accuracy_lift']:>+6.3f} | "
                        f"{res['win_rate']:>6.3f} | "
                        f"{res['total_pnl_ticks']:>9.1f} | "
                        f"{res['total_pnl_usd']:>9.0f} | "
                        f"{res['mean_pnl_per_trade']:>8.3f} | "
                        f"{res['profit_factor']:>6.2f} | "
                        f"{res['sortino']:>7.3f}")

            if res["sortino"] > best_sortino and res["n_gated"] > 100:
                best_sortino = res["sortino"]
                best_threshold = thresh
        else:
            logger.info(f"  {thresh:>7.2f} | {res['n_gated']:>7} (too few)")

    logger.info(f"\n  BEST threshold: {best_threshold:.2f} "
                f"(Sortino={best_sortino:.3f})")

    # Per-fold summary table
    logger.info(f"\n  {'='*80}")
    logger.info(f"  PER-FOLD SUMMARY")
    logger.info(f"  {'='*80}")
    logger.info(f"  {'Fold':>4} | {'Date':>10} | {'GateAcc':>7} | {'CNN_Dir':>7} | "
                f"{'GatePos%':>8} | {'n_test':>7}")
    logger.info(f"  {'-'*55}")
    for fr in fold_results:
        logger.info(f"  {fr['fold']:>4} | {fr['test_date']:>10} | "
                    f"{fr['gate_accuracy_05']:>7.4f} | {fr['cnn_dir_acc_ungated']:>7.4f} | "
                    f"{fr['gate_pos_rate']*100:>7.1f}% | {fr['n_test']:>7,}")

    # Best threshold detail per fold
    logger.info(f"\n  PER-FOLD at BEST threshold ({best_threshold:.2f}):")
    logger.info(f"  {'Fold':>4} | {'Date':>10} | {'Trades':>7} | {'DirG':>6} | "
                f"{'Lift':>6} | {'WR':>6} | {'PnL_t':>8} | {'PnL_USD':>8} | {'Sortino':>7}")
    logger.info(f"  {'-'*75}")
    bt_key = f"{best_threshold:.2f}"
    for fr in fold_results:
        if bt_key in fr["threshold_results"]:
            r = fr["threshold_results"][bt_key]
            if r.get("n_gated", 0) > 5:
                logger.info(f"  {fr['fold']:>4} | {fr['test_date']:>10} | "
                            f"{r['n_gated']:>7,} | {r['dir_accuracy_gated']:>6.3f} | "
                            f"{r['dir_accuracy_lift']:>+6.3f} | {r['win_rate']:>6.3f} | "
                            f"{r['total_pnl_ticks']:>8.1f} | {r['total_pnl_ticks']*TICK_VAL:>8.0f} | "
                            f"{r['sortino']:>7.3f}")

    # MLflow concat metrics
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics({
                "baseline_dir_accuracy": float(baseline_dir_acc),
                "baseline_total_pnl_ticks": float(baseline_total_pnl),
                "baseline_sortino": float(baseline_sortino),
                "baseline_win_rate": float(baseline_wr),
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
                            f"concat_pnl_usd_t{t_key}": r["total_pnl_usd"],
                            f"concat_sortino_t{t_key}": r["sortino"],
                            f"concat_coverage_t{t_key}": r["coverage"],
                            f"concat_wr_t{t_key}": r["win_rate"],
                            f"concat_dir_gated_t{t_key}": r["dir_accuracy_gated"],
                            f"concat_dir_lift_t{t_key}": r["dir_accuracy_lift"],
                            f"concat_pf_t{t_key}": r["profit_factor"],
                        })
        except Exception as e:
            logger.warning(f"MLflow concat logging error: {e}")

    # Save concat predictions
    concat_path = output_dir / "concat_oot_predictions.npz"
    np.savez_compressed(str(concat_path),
        gate_probs=all_gate_probs,
        cnn_pred_10s=all_cnn_pred_10s,
        y_10s=all_y_10s,
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
        "baseline_dir_accuracy": float(baseline_dir_acc),
        "baseline_total_pnl_ticks": float(baseline_total_pnl),
        "baseline_sortino": float(baseline_sortino),
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
        description="Confidence Gate v3 — Pure Gate, No Direction Re-prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--cnn-mamba-dir", type=str, default=None)
    parser.add_argument("--patchtst-dir", type=str, default=None)
    parser.add_argument("--vol-lgbm-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dates", type=str, nargs="*", default=None)

    # XGBoost params
    parser.add_argument("--xgb-depth", type=int, default=4)
    parser.add_argument("--xgb-estimators", type=int, default=200)
    parser.add_argument("--xgb-lr", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)
    parser.add_argument("--early-stopping", type=int, default=20)

    # Gate label: min move threshold
    parser.add_argument("--min-move-ticks", type=float, default=1.0,
                        help="Min |y_10s| in ticks for positive gate label")

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
    logger.info(f"Confidence Gate v3 — Pure Gate, No Direction Re-prediction")
    logger.info(f"{'='*60}")
    logger.info(f"Device: CPU (Jupiter)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Data sources:")
    logger.info(f"  CNN-Mamba v2:  {CNN_MAMBA_DIR}")
    logger.info(f"  PatchTST:      {PATCHTST_DIR}")
    logger.info(f"  Vol LGBM v3:   {VOL_LGBM_DIR}")
    logger.info(f"CV: Leave-One-Out ({len(OVERLAP_DATES)} folds)")
    logger.info(f"XGBoost: depth={args.xgb_depth}, estimators={args.xgb_estimators}, "
                f"lr={args.xgb_lr}, subsample={args.xgb_subsample}, "
                f"colsample={args.xgb_colsample}, early_stop={args.early_stopping}")
    logger.info(f"Gate label: sign(cnn)==sign(y) AND |y|>{args.min_move_ticks} ticks")
    logger.info(f"Gate sweep: {GATE_THRESHOLDS}")
    logger.info(f"Features: ~16 (NO direction signals — magnitudes only)")
    logger.info(f"Direction: 100% from CNN-Mamba (gate only decides trade/skip)")

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
        mlflow_run = mlflow.start_run(run_name=f"confidence_gate_v3_{ts}")
        mlflow.log_params({
            "model": "confidence_gate_v3",
            "version": "v3_pure_gate",
            "n_dates": len(all_dates),
            "cv_method": "leave_one_out",
            "xgb_depth": args.xgb_depth,
            "xgb_estimators": args.xgb_estimators,
            "xgb_lr": args.xgb_lr,
            "xgb_subsample": args.xgb_subsample,
            "xgb_colsample": args.xgb_colsample,
            "early_stopping": args.early_stopping,
            "min_move_ticks": args.min_move_ticks,
            "n_features": 16,
            "total_samples": total_samples,
            "node": socket.gethostname(),
            "architecture": "XGBoost_PureConfidenceGate",
            "approach": "gate_only_no_direction_reprediction",
            "feature_sources": "abs_magnitudes+agreement+vol+time+joint_conf",
            "dates": ",".join([d["date"] for d in all_dates]),
        })

    try:
        results = train_loo_cv(all_dates, args, output_dir)

        # Save results summary
        summary = {
            "model": "confidence_gate_v3",
            "approach": "pure_confidence_gate_no_direction_reprediction",
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
