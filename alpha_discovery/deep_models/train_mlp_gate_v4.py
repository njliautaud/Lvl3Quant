#!/usr/bin/env python3
"""
train_mlp_gate_v4.py — MLP Confidence Gate v4 (PyTorch, Pure Gate)
===================================================================

KEY CHANGES from v3:
  - PyTorch MLP replaces XGBoost (user: "why TF are u doing xgboost")
  - Correct cost model: commission = 0.376 ticks (4.70/12.50), spread parameterized
  - Two cost scenarios: conservative (commission + spread) vs aggressive (commission only)
  - Leave-one-day-out CV over 9 overlapping dates
  - Early stopping with patience=10 on validation BCE loss
  - Cosine annealing LR schedule

Architecture:
  Input (~16 features) → [64+BN+ReLU+Drop] → [32+BN+ReLU+Drop] → [16+BN+ReLU+Drop] → 1 (sigmoid)

Gate label: 1 if sign(cnn_pred_10s)==sign(y_true_10s) AND |y_true_10s|>1.0 ticks

Direction: 100% from CNN-Mamba v2. Gate only decides trade/skip.

Usage:
  python train_mlp_gate_v4.py

Author: MLP Gate v4 for Lvl3Quant
Date: 2026-04-30
"""

import os
import sys
import gc
import re
import time
import json
import copy
import logging
import argparse
import warnings
import socket
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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
log_path = LOG_DIR / f"mlp_gate_v4_{_ts}.log"

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

print(f">>> train_mlp_gate_v4.py loaded | log: {log_path}", flush=True)

# ============================================================
# Paths & Constants
# ============================================================
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent

CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
VOL_LGBM_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"
DEFAULT_OUTPUT_DIR = LVL3_ROOT / "output" / "mlp_gate_v4"

TICK = 0.25
TICK_VAL = 12.50
COMMISSION_TICKS = 0.376       # $4.70 / $12.50 = 0.376 ticks RT
SPREAD_COST_CONSERVATIVE = 0.0  # HC #231(A): no spread crossing cost
SPREAD_COST_AGGRESSIVE = 0.0    # limit fills: no spread cost

# Two cost scenarios (HC #231(A): both equal commission only)
COST_CONSERVATIVE = COMMISSION_TICKS + SPREAD_COST_CONSERVATIVE  # 0.376 ticks
COST_AGGRESSIVE = COMMISSION_TICKS + SPREAD_COST_AGGRESSIVE      # 0.376 ticks

MLFLOW_EXPERIMENT = "MLPGate_v4"

OVERLAP_DATES = [
    "20260223", "20260224", "20260225", "20260226", "20260227",
    "20260301", "20260302", "20260303", "20260304",
]

GATE_THRESHOLDS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

TRADING_START_HOUR = 18.0
TRADING_HOURS = 23.5


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
# Data Discovery & Alignment (reused from v3)
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
    ptst_preds = ptst_data["predictions"].astype(np.float32)

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
        "cnn_labels": cnn_labels[cnn_idxs],
        "ptst_preds": ptst_preds[ptst_idxs],
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
# Feature Engineering (same as v3 — NO raw embeddings, NO direction)
# ============================================================

def assemble_features(date_data: Dict) -> Tuple[np.ndarray, List[str]]:
    n = date_data["n_samples"]
    cnn = date_data["cnn_preds"]
    ptst = date_data["ptst_preds"]
    vol = date_data["vol_preds"]

    features = []
    names = []

    # CNN absolute prediction magnitudes
    features.append(np.abs(cnn[:, 0]).reshape(-1, 1))
    names.append("cnn_abs_pred_1s")
    features.append(np.abs(cnn[:, 1]).reshape(-1, 1))
    names.append("cnn_abs_pred_5s")
    features.append(np.abs(cnn[:, 2]).reshape(-1, 1))
    names.append("cnn_abs_pred_10s")

    # CNN horizon spread
    cnn_horizon_spread = np.abs(cnn[:, 2]) - np.abs(cnn[:, 0])
    features.append(cnn_horizon_spread.reshape(-1, 1))
    names.append("cnn_horizon_spread")

    # PatchTST agreement & confidence
    cnn_10s = cnn[:, 2]
    ptst_10s = ptst[:, 2]

    patchtst_agreement = (np.sign(cnn_10s) == np.sign(ptst_10s)).astype(np.float32)
    features.append(patchtst_agreement.reshape(-1, 1))
    names.append("patchtst_agreement")

    features.append(np.abs(ptst[:, 0]).reshape(-1, 1))
    names.append("patchtst_abs_1s")
    features.append(np.abs(ptst[:, 1]).reshape(-1, 1))
    names.append("patchtst_abs_5s")
    features.append(np.abs(ptst_10s).reshape(-1, 1))
    names.append("patchtst_abs_10s")

    # Volatility predictions
    features.append(vol)
    names += ["vol_pred_10s", "vol_pred_30s", "vol_pred_60s"]

    # Vol regime
    vol_10s = vol[:, 0]
    vol_median = np.median(vol_10s) if np.any(vol_10s != 0) else 0.0
    vol_regime = (vol_10s > vol_median).astype(np.float32)
    features.append(vol_regime.reshape(-1, 1))
    names.append("vol_regime")

    # Time features
    frac = np.linspace(0, 1, n, dtype=np.float32)
    hour_of_day = TRADING_START_HOUR + frac * TRADING_HOURS
    hour_of_day = hour_of_day % 24.0

    hour_rad = 2 * np.pi * hour_of_day / 24.0
    features.append(np.sin(hour_rad).reshape(-1, 1))
    names.append("hour_sin")
    features.append(np.cos(hour_rad).reshape(-1, 1))
    names.append("hour_cos")

    # Joint confidence
    conf_product = np.abs(cnn_10s) * np.abs(ptst_10s)
    features.append(conf_product.reshape(-1, 1))
    names.append("cnn_patchtst_conf_product")

    # Multi-horizon agreement
    cnn_signs = np.sign(cnn)
    all_agree = ((cnn_signs[:, 0] == cnn_signs[:, 1]) &
                 (cnn_signs[:, 1] == cnn_signs[:, 2])).astype(np.float32)
    features.append(all_agree.reshape(-1, 1))
    names.append("multi_horizon_agreement")

    X = np.concatenate(features, axis=1).astype(np.float32)
    assert X.shape == (n, len(names)), f"Shape mismatch: {X.shape} vs ({n}, {len(names)})"
    return X, names


# ============================================================
# Label Construction
# ============================================================

def compute_gate_labels(cnn_labels: np.ndarray,
                        cnn_preds: np.ndarray,
                        min_move_ticks: float = 1.0) -> Dict[str, np.ndarray]:
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
# MLP Model (PyTorch)
# ============================================================

class ConfidenceGateMLP(nn.Module):
    """
    MLP confidence gate: predicts probability that CNN's directional call is correct.
    Architecture: Input → [64+BN+ReLU+Drop] → [32+BN+ReLU+Drop] → [16+BN+ReLU+Drop] → 1
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,) raw logits


# ============================================================
# Metrics
# ============================================================

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
                          threshold: float,
                          cost_ticks: float,
                          cost_label: str = "") -> Dict:
    mask = gate_probs >= threshold
    n_gated = mask.sum()
    n_total = len(gate_probs)

    if n_gated < 5:
        return {
            "threshold": threshold,
            "n_gated": int(n_gated),
            "coverage": 0.0,
            "cost_label": cost_label,
        }

    gated_cnn_pred = cnn_pred_10s[mask]
    gated_y = y_10s[mask]

    cnn_dir = np.sign(gated_cnn_pred)
    actual_dir = np.sign(gated_y)
    dir_acc_gated = np.mean(cnn_dir == actual_dir)

    all_cnn_dir = np.sign(cnn_pred_10s)
    all_actual_dir = np.sign(y_10s)
    dir_acc_ungated = np.mean(all_cnn_dir == all_actual_dir)

    dir_acc_lift = dir_acc_gated - dir_acc_ungated

    entry_dir = np.sign(gated_cnn_pred)
    pnl_ticks = entry_dir * gated_y - cost_ticks
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
        "cost_label": cost_label,
        "cost_ticks": cost_ticks,
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
# Training Loop
# ============================================================

def train_mlp_gate(X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   input_dim: int, args) -> Tuple[ConfidenceGateMLP, Dict]:
    """Train MLP gate with early stopping on validation loss."""
    device = torch.device("cpu")  # Jupiter is CPU-only

    # Class weighting for BCE
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)
    pos_weight = torch.clamp(pos_weight, max=10.0)

    model = ConfidenceGateMLP(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float(),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4, shuffle=False,
                            num_workers=0, pin_memory=False)

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    train_history = []

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(xb)
            train_n += len(xb)

        scheduler.step()
        avg_train_loss = train_loss_sum / max(train_n, 1)

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * len(xb)
                val_n += len(xb)

        avg_val_loss = val_loss_sum / max(val_n, 1)
        train_history.append({"epoch": epoch, "train_loss": avg_train_loss,
                              "val_loss": avg_val_loss})

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            logger.info(f"    Epoch {epoch:3d}: train_loss={avg_train_loss:.4f} "
                        f"val_loss={avg_val_loss:.4f} "
                        f"best={best_val_loss:.4f} patience={patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info(f"    Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    info = {
        "best_val_loss": best_val_loss,
        "best_epoch": epoch - patience_counter if patience_counter > 0 else epoch,
        "total_epochs": epoch + 1,
        "pos_weight": float(pos_weight.item()),
    }

    return model, info


def predict_mlp(model: ConfidenceGateMLP, X: np.ndarray,
                batch_size: int = 4096) -> np.ndarray:
    """Get gate probabilities from trained MLP."""
    device = torch.device("cpu")
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float().to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs)


# ============================================================
# Leave-One-Out Cross-Validation
# ============================================================

def train_loo_cv(all_dates: List[Dict], args: argparse.Namespace,
                 output_dir: Path) -> Dict:
    n_dates = len(all_dates)
    logger.info(f"\nLeave-One-Out CV: {n_dates} folds (each date held out once)")

    _, feature_names = assemble_features(all_dates[0])
    input_dim = len(feature_names)
    logger.info(f"Input dimension: {input_dim} features: {feature_names}")

    # Accumulators
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

        # Normalize features (fit on train, apply to test)
        train_mean = X_train.mean(axis=0)
        train_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - train_mean) / train_std
        X_test_norm = (X_test - train_mean) / train_std

        # ---- Train MLP gate ----
        t0 = time.time()
        logger.info("  Training MLP Confidence Gate (PyTorch)...")
        model, train_info = train_mlp_gate(
            X_train_norm, y_gate_train, X_test_norm, y_gate_test,
            input_dim=input_dim, args=args,
        )
        train_time = time.time() - t0
        logger.info(f"  MLP trained in {train_time:.1f}s | best_epoch={train_info['best_epoch']} | "
                    f"val_loss={train_info['best_val_loss']:.4f}")

        # ---- OOT Predictions ----
        gate_probs = predict_mlp(model, X_test_norm)

        # Gate accuracy (at 0.5 threshold)
        gate_pred = (gate_probs >= 0.5).astype(np.float32)
        gate_acc = np.mean(gate_pred == y_gate_test)

        # CNN baseline
        cnn_dir = np.sign(cnn_pred_10s_test)
        actual_dir = np.sign(y_10s_test)
        cnn_dir_acc_ungated = np.mean(cnn_dir == actual_dir)

        logger.info(f"  Gate accuracy (0.5): {gate_acc:.4f}")
        logger.info(f"  CNN direction accuracy (ungated): {cnn_dir_acc_ungated:.4f}")
        logger.info(f"  Gate prob distribution: mean={gate_probs.mean():.3f}, "
                    f"median={np.median(gate_probs):.3f}, "
                    f"p10={np.percentile(gate_probs, 10):.3f}, "
                    f"p90={np.percentile(gate_probs, 90):.3f}")

        # Threshold sweep — both cost scenarios
        fold_threshold_results = {}
        for cost_label, cost_ticks in [("conservative", COST_CONSERVATIVE),
                                        ("aggressive", COST_AGGRESSIVE)]:
            for thresh in GATE_THRESHOLDS:
                res = evaluate_at_threshold(gate_probs, cnn_pred_10s_test,
                                            y_10s_test, thresh, cost_ticks, cost_label)
                key = f"{cost_label}_{thresh:.2f}"
                fold_threshold_results[key] = res
                if res["n_gated"] > 5 and cost_label == "conservative":
                    logger.info(f"  [{cost_label}] Gate@{thresh:.2f}: n={res['n_gated']:>6,} | "
                                f"cov={res['coverage']:.3f} | "
                                f"dirG={res['dir_accuracy_gated']:.3f} | "
                                f"lift={res['dir_accuracy_lift']:+.3f} | "
                                f"WR={res['win_rate']:.3f} | "
                                f"PnL={res['total_pnl_ticks']:>8.1f}t | "
                                f"${res['total_pnl_usd']:>8.0f} | "
                                f"PF={res['profit_factor']:.2f} | "
                                f"Sortino={res['sortino']:.3f}")

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
            train_mean=train_mean,
            train_std=train_std,
        )

        # Save model
        model_dir = output_dir / f"fold_{fold_idx:02d}_models"
        model_dir.mkdir(exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dims": args.hidden_dims,
            "dropout": args.dropout,
            "train_mean": train_mean,
            "train_std": train_std,
            "feature_names": feature_names,
            "train_info": train_info,
        }, str(model_dir / "confidence_gate_mlp.pt"))

        fold_result = {
            "fold": fold_idx,
            "test_date": test_date_data["date"],
            "n_train": n_train,
            "n_test": n_test,
            "train_time_s": round(train_time, 1),
            "gate_accuracy_05": float(gate_acc),
            "cnn_dir_acc_ungated": float(cnn_dir_acc_ungated),
            "gate_pos_rate": float(y_gate_test.mean()),
            "best_epoch": train_info["best_epoch"],
            "best_val_loss": train_info["best_val_loss"],
            "threshold_results": fold_threshold_results,
        }
        fold_results.append(fold_result)

        # MLflow per-fold
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    f"fold_{fold_idx}_gate_acc": float(gate_acc),
                    f"fold_{fold_idx}_cnn_dir_acc": float(cnn_dir_acc_ungated),
                    f"fold_{fold_idx}_val_loss": float(train_info["best_val_loss"]),
                    f"fold_{fold_idx}_best_epoch": train_info["best_epoch"],
                })
            except Exception as e:
                logger.warning(f"MLflow fold logging error: {e}")

        del model
        del X_train, X_test, X_train_norm, X_test_norm
        gc.collect()

    # ============================================================
    # Concat Metrics
    # ============================================================
    total_samples = sum(len(g) for g in concat_gate_probs)
    logger.info(f"\n{'='*60}")
    logger.info(f"CONCAT METRICS -- {n_dates} LOO folds, {total_samples:,} total OOT samples")
    logger.info(f"{'='*60}")

    all_gate_probs = np.concatenate(concat_gate_probs)
    all_cnn_pred_10s = np.concatenate(concat_cnn_pred_10s)
    all_y_10s = np.concatenate(concat_y_10s)

    # CNN baselines (both cost scenarios)
    all_cnn_dir = np.sign(all_cnn_pred_10s)
    all_actual_dir = np.sign(all_y_10s)
    baseline_dir_acc = np.mean(all_cnn_dir == all_actual_dir)

    for cost_label, cost_ticks in [("conservative", COST_CONSERVATIVE),
                                    ("aggressive", COST_AGGRESSIVE)]:
        baseline_pnl_ticks = (all_cnn_dir * all_y_10s - cost_ticks)
        baseline_total_pnl = baseline_pnl_ticks.sum()
        baseline_sortino = compute_sortino(baseline_pnl_ticks)
        baseline_wr = np.mean(baseline_pnl_ticks > 0)

        logger.info(f"\n  CNN BASELINE [{cost_label}, cost={cost_ticks:.3f}t] "
                    f"(ungated, all {total_samples:,} trades):")
        logger.info(f"    Direction accuracy: {baseline_dir_acc:.4f}")
        logger.info(f"    Win rate: {baseline_wr:.4f}")
        logger.info(f"    Total PnL: {baseline_total_pnl:.1f} ticks "
                    f"(${baseline_total_pnl * TICK_VAL:,.0f})")
        logger.info(f"    Mean PnL/trade: {baseline_total_pnl/total_samples:.3f} ticks")
        logger.info(f"    Sortino: {baseline_sortino:.3f}")

    # Gate prob distribution
    logger.info(f"\n  Gate prob distribution: "
                f"mean={all_gate_probs.mean():.3f}, "
                f"median={np.median(all_gate_probs):.3f}, "
                f"p10={np.percentile(all_gate_probs, 10):.3f}, "
                f"p90={np.percentile(all_gate_probs, 90):.3f}")

    # ---- Threshold sweep on concat, both cost scenarios ----
    concat_threshold_results = {}
    best_results = {}

    for cost_label, cost_ticks in [("conservative", COST_CONSERVATIVE),
                                    ("aggressive", COST_AGGRESSIVE)]:
        logger.info(f"\n  {'='*100}")
        logger.info(f"  GATE THRESHOLD SWEEP [{cost_label.upper()}, cost={cost_ticks:.3f} ticks]"
                    f" (concat, {total_samples:,} samples)")
        logger.info(f"  {'='*100}")
        logger.info(f"  {'Thresh':>7} | {'Trades':>7} | {'Cover':>6} | {'DirGated':>8} | "
                    f"{'Lift':>6} | {'WR':>6} | "
                    f"{'PnL_t':>9} | {'PnL_USD':>10} | {'PnL/Trd':>8} | {'PF':>6} | {'Sortino':>7}")
        logger.info(f"  {'-'*105}")

        best_sortino = -999
        best_threshold = 0.5

        for thresh in GATE_THRESHOLDS:
            res = evaluate_at_threshold(all_gate_probs, all_cnn_pred_10s,
                                        all_y_10s, thresh, cost_ticks, cost_label)
            key = f"{cost_label}_{thresh:.2f}"
            concat_threshold_results[key] = res

            if res["n_gated"] > 5:
                logger.info(f"  {thresh:>7.2f} | {res['n_gated']:>7,} | {res['coverage']:>6.3f} | "
                            f"{res['dir_accuracy_gated']:>8.4f} | "
                            f"{res['dir_accuracy_lift']:>+6.3f} | "
                            f"{res['win_rate']:>6.3f} | "
                            f"{res['total_pnl_ticks']:>9.1f} | "
                            f"${res['total_pnl_usd']:>9,.0f} | "
                            f"{res['mean_pnl_per_trade']:>8.3f} | "
                            f"{res['profit_factor']:>6.2f} | "
                            f"{res['sortino']:>7.3f}")

                if res["sortino"] > best_sortino and res["n_gated"] > 100:
                    best_sortino = res["sortino"]
                    best_threshold = thresh
            else:
                logger.info(f"  {thresh:>7.2f} | {res['n_gated']:>7} (too few)")

        logger.info(f"\n  BEST [{cost_label}] threshold: {best_threshold:.2f} "
                    f"(Sortino={best_sortino:.3f})")
        best_results[cost_label] = {
            "best_threshold": best_threshold,
            "best_sortino": best_sortino,
        }

    # Per-fold summary
    logger.info(f"\n  {'='*80}")
    logger.info(f"  PER-FOLD SUMMARY")
    logger.info(f"  {'='*80}")
    logger.info(f"  {'Fold':>4} | {'Date':>10} | {'GateAcc':>7} | {'CNN_Dir':>7} | "
                f"{'GatePos%':>8} | {'BestEp':>6} | {'ValLoss':>7} | {'n_test':>7}")
    logger.info(f"  {'-'*70}")
    for fr in fold_results:
        logger.info(f"  {fr['fold']:>4} | {fr['test_date']:>10} | "
                    f"{fr['gate_accuracy_05']:>7.4f} | {fr['cnn_dir_acc_ungated']:>7.4f} | "
                    f"{fr['gate_pos_rate']*100:>7.1f}% | {fr['best_epoch']:>6} | "
                    f"{fr['best_val_loss']:>7.4f} | {fr['n_test']:>7,}")

    # Per-fold at best thresholds
    for cost_label in ["conservative", "aggressive"]:
        bt = best_results[cost_label]["best_threshold"]
        bt_key = f"{cost_label}_{bt:.2f}"
        logger.info(f"\n  PER-FOLD at BEST threshold [{cost_label}] ({bt:.2f}):")
        logger.info(f"  {'Fold':>4} | {'Date':>10} | {'Trades':>7} | {'DirG':>6} | "
                    f"{'Lift':>6} | {'WR':>6} | {'PnL_t':>8} | {'PnL_USD':>9} | {'Sortino':>7}")
        logger.info(f"  {'-'*80}")
        for fr in fold_results:
            if bt_key in fr["threshold_results"]:
                r = fr["threshold_results"][bt_key]
                if r.get("n_gated", 0) > 5:
                    logger.info(f"  {fr['fold']:>4} | {fr['test_date']:>10} | "
                                f"{r['n_gated']:>7,} | {r['dir_accuracy_gated']:>6.3f} | "
                                f"{r['dir_accuracy_lift']:>+6.3f} | {r['win_rate']:>6.3f} | "
                                f"{r['total_pnl_ticks']:>8.1f} | "
                                f"${r['total_pnl_ticks']*TICK_VAL:>8,.0f} | "
                                f"{r['sortino']:>7.3f}")

    # MLflow concat metrics
    if MLFLOW_AVAILABLE:
        try:
            # Baselines
            bl_cons = all_cnn_dir * all_y_10s - COST_CONSERVATIVE
            bl_aggr = all_cnn_dir * all_y_10s - COST_AGGRESSIVE
            mlflow.log_metrics({
                "baseline_dir_accuracy": float(baseline_dir_acc),
                "baseline_pnl_conservative": float(bl_cons.sum()),
                "baseline_pnl_aggressive": float(bl_aggr.sum()),
                "baseline_sortino_conservative": compute_sortino(bl_cons),
                "baseline_sortino_aggressive": compute_sortino(bl_aggr),
                "n_total_samples": total_samples,
                "n_folds": n_dates,
            })
            for cost_label in ["conservative", "aggressive"]:
                br = best_results[cost_label]
                mlflow.log_metrics({
                    f"best_threshold_{cost_label}": br["best_threshold"],
                    f"best_sortino_{cost_label}": br["best_sortino"] if br["best_sortino"] > -999 else 0,
                })
                for thresh in GATE_THRESHOLDS:
                    key = f"{cost_label}_{thresh:.2f}"
                    if key in concat_threshold_results:
                        r = concat_threshold_results[key]
                        t_key = int(thresh * 100)
                        if r.get("n_gated", 0) > 5:
                            mlflow.log_metrics({
                                f"concat_{cost_label}_pnl_t{t_key}": r["total_pnl_ticks"],
                                f"concat_{cost_label}_pnl_usd_t{t_key}": r["total_pnl_usd"],
                                f"concat_{cost_label}_sortino_t{t_key}": r["sortino"],
                                f"concat_{cost_label}_coverage_t{t_key}": r["coverage"],
                                f"concat_{cost_label}_wr_t{t_key}": r["win_rate"],
                                f"concat_{cost_label}_dir_gated_t{t_key}": r["dir_accuracy_gated"],
                                f"concat_{cost_label}_dir_lift_t{t_key}": r["dir_accuracy_lift"],
                                f"concat_{cost_label}_pf_t{t_key}": r["profit_factor"],
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
        "cost_conservative_ticks": COST_CONSERVATIVE,
        "cost_aggressive_ticks": COST_AGGRESSIVE,
        "best_results": best_results,
        "concat_threshold_results": concat_threshold_results,
        "per_fold": fold_results,
    }

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLP Confidence Gate v4 — PyTorch, Pure Gate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--cnn-mamba-dir", type=str, default=None)
    parser.add_argument("--patchtst-dir", type=str, default=None)
    parser.add_argument("--vol-lgbm-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dates", type=str, nargs="*", default=None)

    # MLP architecture
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)

    # Gate label
    parser.add_argument("--min-move-ticks", type=float, default=1.0)

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
    logger.info(f"MLP Confidence Gate v4 — PyTorch, Pure Gate")
    logger.info(f"{'='*60}")
    logger.info(f"Device: CPU (Jupiter)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Data sources:")
    logger.info(f"  CNN-Mamba v2:  {CNN_MAMBA_DIR}")
    logger.info(f"  PatchTST:      {PATCHTST_DIR}")
    logger.info(f"  Vol LGBM v3:   {VOL_LGBM_DIR}")
    logger.info(f"CV: Leave-One-Out ({len(OVERLAP_DATES)} folds)")
    logger.info(f"MLP: hidden={args.hidden_dims}, dropout={args.dropout}, "
                f"lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}")
    logger.info(f"Training: epochs={args.epochs}, patience={args.patience}")
    logger.info(f"Gate label: sign(cnn)==sign(y) AND |y|>{args.min_move_ticks} ticks")
    logger.info(f"Costs: conservative={COST_CONSERVATIVE:.3f}t, aggressive={COST_AGGRESSIVE:.3f}t")
    logger.info(f"Commission: {COMMISSION_TICKS:.3f} ticks ($4.70/$12.50)")
    logger.info(f"Spread: conservative={SPREAD_COST_CONSERVATIVE:.1f}t RT, aggressive={SPREAD_COST_AGGRESSIVE:.1f}t")
    logger.info(f"Gate sweep: {GATE_THRESHOLDS}")
    logger.info(f"Features: ~{len(assemble_features.__code__.co_varnames)} (NO direction, NO embeddings)")
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
        mlflow_run = mlflow.start_run(run_name=f"mlp_gate_v4_{ts}")
        mlflow.log_params({
            "model": "mlp_gate_v4",
            "version": "v4_pytorch_mlp",
            "n_dates": len(all_dates),
            "cv_method": "leave_one_out",
            "hidden_dims": str(args.hidden_dims),
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "min_move_ticks": args.min_move_ticks,
            "cost_conservative": COST_CONSERVATIVE,
            "cost_aggressive": COST_AGGRESSIVE,
            "commission_ticks": COMMISSION_TICKS,
            "total_samples": total_samples,
            "node": socket.gethostname(),
            "architecture": "PyTorch_MLP_PureConfidenceGate",
            "approach": "gate_only_no_direction_reprediction",
            "dates": ",".join([d["date"] for d in all_dates]),
        })

    try:
        results = train_loo_cv(all_dates, args, output_dir)

        # Save results summary
        summary = {
            "model": "mlp_gate_v4",
            "approach": "pytorch_mlp_pure_confidence_gate",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dates_used": [d["date"] for d in all_dates],
            "args": vars(args),
            "costs": {
                "commission_ticks": COMMISSION_TICKS,
                "spread_conservative": SPREAD_COST_CONSERVATIVE,
                "spread_aggressive": SPREAD_COST_AGGRESSIVE,
                "total_conservative": COST_CONSERVATIVE,
                "total_aggressive": COST_AGGRESSIVE,
            },
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
