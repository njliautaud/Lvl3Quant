#!/usr/bin/env python3
"""
train_exec_mlp.py — Smart Execution MLP Combiner
=================================================

Combines embeddings + predictions from 3 frozen models to make execution decisions:
  1. CNN-Mamba v2  — embeddings(96d) + predictions(3)
  2. PatchTST       — embeddings(256d) + predictions(3)
  3. Vol LGBM v3    — predictions(3)

Multi-task output heads:
  1. Direction prediction (regression, 3 horizons) — combined alpha signal
  2. Trade gate (binary classification) — should we trade or skip?
  3. Confidence score (regression) — how confident is the ensemble?

Training:
  - Sliding window walk-forward (HC #0 — NEVER expanding)
  - 7 days train, 2 days test from 9 overlapping days
  - Pure PyTorch, CPU training (Jupiter — no GPU)
  - MLflow logging mandatory
  - Saves OOT predictions as .npz

Data alignment:
  - CNN-Mamba and Vol LGBM share ~14K samples/day (window=3000, stride=250)
  - PatchTST has ~28K samples/day (window=500, stride=250) — downsampled by 2x
  - Vol LGBM aligned by anchor_idxs nearest-neighbor matching

Overlapping dates (9 days): 20260223 — 20260304

Usage:
  python train_exec_mlp.py
  python train_exec_mlp.py --train-days 7 --epochs 30 --hidden 256
  python train_exec_mlp.py --cnn-mamba-dir /path/to/cnn --patchtst-dir /path/to/ptst

Author: Execution combiner for Lvl3Quant
Date: 2026-04-29
"""

import os
import sys
import gc
import time
import json
import logging
import argparse
import warnings
import socket
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import scipy.stats

# MLflow
try:
    if int(os.environ.get("DISABLE_MLFLOW", 0)):
        raise ImportError("MLflow disabled via DISABLE_MLFLOW env var")
    import mlflow
    MLFLOW_AVAILABLE = True
except (ImportError, ValueError):
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow disabled or not installed — skipping experiment tracking")

# ============================================================
# Logging
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

_ts = time.strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"exec_mlp_{_ts}.log"

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

# Force flush on every log line
for _h in logging.root.handlers:
    _orig_emit = _h.emit
    def _flush_emit(record, _emit=_orig_emit, _handler=_h):
        _emit(record)
        _handler.flush()
    _h.emit = _flush_emit

print(f">>> train_exec_mlp.py loaded | log: {log_path}", flush=True)

# ============================================================
# Paths & Constants
# ============================================================
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent

CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
VOL_LGBM_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"
DEFAULT_OUTPUT_DIR = LVL3_ROOT / "output" / "exec_mlp_combiner"

TICK = 0.25
TICK_VAL = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0

# Horizons for each model
CNN_HORIZONS = ["1s", "5s", "10s"]
VOL_HORIZONS_S = [10, 30, 60]  # Vol LGBM uses seconds

MLFLOW_EXPERIMENT = "ExecMLP_Combiner"

# Overlapping dates between the 3 models
OVERLAP_DATES = [
    "20260223", "20260224", "20260225", "20260226", "20260227",
    "20260301", "20260302", "20260303", "20260304",
]


def _detect_mlflow_uri() -> str:
    """Auto-detect MLflow tracking URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    # Try local file store first (Jupiter has no MLflow server)
    local_store = LVL3_ROOT / "mlflow"
    if local_store.exists():
        return f"file://{local_store}"
    # Try localhost server
    try:
        s = socket.create_connection(("localhost", 5000), timeout=2)
        s.close()
        return "http://localhost:5000"
    except Exception:
        pass
    # Fallback to local file store (create it)
    local_store.mkdir(exist_ok=True)
    return f"file://{local_store}"


MLFLOW_TRACKING_URI = _detect_mlflow_uri()


# ============================================================
# Data Loading — Per-date alignment across 3 models
# ============================================================

def _extract_date_from_oot_files(oot_files) -> Optional[str]:
    """Extract YYYYMMDD date string from oot_files array."""
    if oot_files is None or len(oot_files) == 0:
        return None
    # oot_files contains filenames like 'mbo_events_20260223.npz'
    fname = str(oot_files[0]) if hasattr(oot_files, '__iter__') else str(oot_files)
    # Try to extract date pattern
    import re
    m = re.search(r'(\d{8})', fname)
    return m.group(1) if m else None


def discover_cnn_mamba_folds() -> Dict[str, Tuple[Path, int]]:
    """Find CNN-Mamba folds and map date -> (path, fold_idx)."""
    date_map = {}
    for f in sorted(CNN_MAMBA_DIR.glob("fold_*_oot_predictions.npz")):
        idx = int(f.stem.split("_")[1])
        try:
            data = np.load(str(f), allow_pickle=True)
            oot_files = data.get("oot_files", None)
            date = _extract_date_from_oot_files(oot_files)
            if date:
                date_map[date] = (f, idx)
            else:
                logger.debug(f"CNN-Mamba fold {idx}: no date extracted from oot_files")
        except Exception as e:
            logger.warning(f"CNN-Mamba fold {idx}: failed to inspect: {e}")
    return date_map


def discover_patchtst_folds() -> Dict[str, Tuple[Path, int]]:
    """Find PatchTST folds and map date -> (path, fold_idx)."""
    date_map = {}
    for f in sorted(PATCHTST_DIR.glob("fold_*_oot_predictions.npz")):
        idx = int(f.stem.split("_")[1])
        try:
            data = np.load(str(f), allow_pickle=True)
            oot_files = data.get("oot_files", None)
            date = _extract_date_from_oot_files(oot_files)
            if date:
                date_map[date] = (f, idx)
            else:
                logger.debug(f"PatchTST fold {idx}: no date extracted from oot_files")
        except Exception as e:
            logger.warning(f"PatchTST fold {idx}: failed to inspect: {e}")
    return date_map


def discover_vol_lgbm_files() -> Dict[str, Path]:
    """Find Vol LGBM per-day prediction files: date -> path."""
    date_map = {}
    for f in sorted(VOL_LGBM_DIR.glob("vol_v3_*_predictions.npz")):
        import re
        m = re.search(r'(\d{8})', f.stem)
        if m:
            date_map[m.group(1)] = f
    # Also try fold-based naming
    for f in sorted(VOL_LGBM_DIR.glob("fold_*_predictions.npz")):
        try:
            data = np.load(str(f), allow_pickle=True)
            oot_files = data.get("oot_files", None)
            date = _extract_date_from_oot_files(oot_files)
            if date:
                date_map[date] = f
        except Exception:
            pass
    return date_map


def align_samples(cnn_n: int, ptst_n: int, vol_n: int,
                  vol_anchor_idxs: Optional[np.ndarray] = None,
                  cnn_stride: int = 250, cnn_window: int = 3000,
                  ptst_stride: int = 250, ptst_window: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute aligned sample indices across the 3 models.

    CNN-Mamba is the reference (~14K samples/day).
    PatchTST has 2x more samples — downsample by nearest-neighbor.
    Vol LGBM uses anchor_idxs for alignment.

    Returns:
        cnn_idxs:  indices into CNN-Mamba arrays
        ptst_idxs: indices into PatchTST arrays
        vol_idxs:  indices into Vol LGBM arrays
    """
    # CNN-Mamba anchor positions in the event stream
    cnn_positions = np.arange(cnn_n) * cnn_stride + cnn_window

    # PatchTST anchor positions
    ptst_positions = np.arange(ptst_n) * ptst_stride + ptst_window

    # For each CNN position, find nearest PatchTST position
    ptst_idxs = np.searchsorted(ptst_positions, cnn_positions, side="left")
    ptst_idxs = np.clip(ptst_idxs, 0, ptst_n - 1)

    # For Vol LGBM, align by anchor_idxs if available
    if vol_anchor_idxs is not None and len(vol_anchor_idxs) > 0:
        # Find nearest vol sample for each CNN position
        vol_idxs = np.searchsorted(vol_anchor_idxs, cnn_positions, side="left")
        vol_idxs = np.clip(vol_idxs, 0, vol_n - 1)
    else:
        # Same count assumption — truncate to min
        vol_idxs = np.arange(min(cnn_n, vol_n))
        # Pad if vol has fewer samples
        if vol_n < cnn_n:
            vol_idxs = np.concatenate([
                vol_idxs,
                np.full(cnn_n - vol_n, vol_n - 1, dtype=np.int64)
            ])

    cnn_idxs = np.arange(cnn_n)

    # Ensure all arrays same length
    n = min(len(cnn_idxs), len(ptst_idxs), len(vol_idxs))
    return cnn_idxs[:n], ptst_idxs[:n], vol_idxs[:n]


def load_date_data(date: str,
                   cnn_folds: Dict[str, Tuple[Path, int]],
                   ptst_folds: Dict[str, Tuple[Path, int]],
                   vol_files: Dict[str, Path]) -> Optional[Dict]:
    """
    Load and align predictions from all 3 models for a single date.

    Returns dict with aligned arrays, or None if data missing.
    """
    if date not in cnn_folds:
        logger.warning(f"Date {date}: CNN-Mamba data not found")
        return None
    if date not in ptst_folds:
        logger.warning(f"Date {date}: PatchTST data not found")
        return None

    cnn_path, cnn_fold_idx = cnn_folds[date]
    ptst_path, ptst_fold_idx = ptst_folds[date]

    try:
        cnn_data = np.load(str(cnn_path), allow_pickle=True)
        ptst_data = np.load(str(ptst_path), allow_pickle=True)
    except Exception as e:
        logger.error(f"Date {date}: load error: {e}")
        return None

    cnn_preds = cnn_data["predictions"].astype(np.float32)    # (N_cnn, 3)
    cnn_labels = cnn_data["labels"].astype(np.float32)         # (N_cnn, 3)
    cnn_embeds = cnn_data["embeddings"].astype(np.float32)     # (N_cnn, 96)

    ptst_preds = ptst_data["predictions"].astype(np.float32)   # (N_ptst, 3)
    ptst_embeds = ptst_data["embeddings"].astype(np.float32)   # (N_ptst, 256)

    # Vol LGBM
    vol_available = date in vol_files
    vol_anchor_idxs = None
    if vol_available:
        try:
            vol_data = np.load(str(vol_files[date]), allow_pickle=True)
            vol_preds = vol_data["predictions"].astype(np.float32)   # (N_vol, 3)
            if "anchor_idxs" in vol_data:
                vol_anchor_idxs = vol_data["anchor_idxs"]
        except Exception as e:
            logger.warning(f"Date {date}: vol LGBM load failed: {e}")
            vol_available = False

    if not vol_available:
        vol_preds = np.zeros((len(cnn_preds), 3), dtype=np.float32)
        logger.info(f"Date {date}: vol LGBM not available, using zeros")

    # Align samples
    cnn_idxs, ptst_idxs, vol_idxs = align_samples(
        cnn_n=len(cnn_preds),
        ptst_n=len(ptst_preds),
        vol_n=len(vol_preds),
        vol_anchor_idxs=vol_anchor_idxs,
    )

    n = len(cnn_idxs)
    if n < 100:
        logger.warning(f"Date {date}: only {n} aligned samples, skipping")
        return None

    result = {
        "date": date,
        "cnn_preds": cnn_preds[cnn_idxs],           # (N, 3)
        "cnn_embeds": cnn_embeds[cnn_idxs],          # (N, 96)
        "cnn_labels": cnn_labels[cnn_idxs],          # (N, 3)
        "ptst_preds": ptst_preds[ptst_idxs],         # (N, 3)
        "ptst_embeds": ptst_embeds[ptst_idxs],       # (N, 256)
        "vol_preds": vol_preds[vol_idxs],            # (N, 3)
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
# Feature Engineering — Derived features from multi-model preds
# ============================================================

def compute_derived_features(cnn_preds: np.ndarray,
                             ptst_preds: np.ndarray,
                             vol_preds: np.ndarray,
                             n: int) -> np.ndarray:
    """
    Compute derived context features from multi-model predictions.

    Returns (N, n_derived) array with:
      0: cnn_ptst_agree_1s   — direction agreement on 1s horizon
      1: cnn_ptst_agree_5s   — direction agreement on 5s horizon
      2: cnn_ptst_agree_10s  — direction agreement on 10s horizon
      3: all_agree_10s       — all 3 models agree on 10s direction
      4: confidence_spread   — max - min prediction across models (10s)
      5: mean_abs_pred_10s   — mean |prediction| across models (10s)
      6: cnn_confidence      — |CNN 10s pred| / std(CNN 10s pred)
      7: ptst_confidence     — |PatchTST 10s pred| / std(PatchTST 10s pred)
      8: vol_regime          — rolling rank of vol 10s pred (percentile)
      9: pred_dispersion     — std of 3 models' 10s predictions
    """
    n_derived = 10
    feat = np.zeros((n, n_derived), dtype=np.float32)

    # Direction agreement per horizon
    for h in range(3):
        cnn_dir = np.sign(cnn_preds[:, h])
        ptst_dir = np.sign(ptst_preds[:, h])
        feat[:, h] = (cnn_dir == ptst_dir).astype(np.float32)

    # All 3 agree on 10s
    cnn_dir_10s = np.sign(cnn_preds[:, 2])
    ptst_dir_10s = np.sign(ptst_preds[:, 2])
    vol_dir_10s = np.sign(vol_preds[:, 2])
    feat[:, 3] = ((cnn_dir_10s == ptst_dir_10s) & (ptst_dir_10s == vol_dir_10s)).astype(np.float32)

    # Confidence spread (max - min pred across models, 10s)
    stacked_10s = np.stack([cnn_preds[:, 2], ptst_preds[:, 2], vol_preds[:, 2]], axis=1)
    feat[:, 4] = stacked_10s.max(axis=1) - stacked_10s.min(axis=1)

    # Mean absolute prediction (10s)
    feat[:, 5] = np.abs(stacked_10s).mean(axis=1)

    # CNN confidence (normalized magnitude)
    cnn_std = max(np.abs(cnn_preds[:, 2]).std(), 1e-8)
    feat[:, 6] = np.abs(cnn_preds[:, 2]) / cnn_std

    # PatchTST confidence
    ptst_std = max(np.abs(ptst_preds[:, 2]).std(), 1e-8)
    feat[:, 7] = np.abs(ptst_preds[:, 2]) / ptst_std

    # Vol regime (rolling percentile rank of vol 10s prediction)
    window = min(500, n // 2)
    if window > 10:
        try:
            import pandas as pd
            feat[:, 8] = pd.Series(vol_preds[:, 2]).rolling(
                window, min_periods=10
            ).rank(pct=True).fillna(0.5).values
        except ImportError:
            # Fallback: simple rank
            feat[:, 8] = scipy.stats.rankdata(vol_preds[:, 2]) / n
    else:
        feat[:, 8] = 0.5

    # Prediction dispersion (std of 3 models' 10s predictions)
    feat[:, 9] = stacked_10s.std(axis=1)

    return feat


N_DERIVED_FEATURES = 10


# ============================================================
# Label Construction — Multi-task labels
# ============================================================

def compute_labels(cnn_labels: np.ndarray,
                   cnn_preds: np.ndarray,
                   ptst_preds: np.ndarray,
                   vol_preds: np.ndarray,
                   gate_threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Compute multi-task labels.

    Args:
        cnn_labels:     (N, 3) ground truth returns at 1s, 5s, 10s
        cnn_preds:      (N, 3) CNN predictions
        ptst_preds:     (N, 3) PatchTST predictions
        vol_preds:      (N, 3) Vol LGBM predictions
        gate_threshold: ticks threshold for trade gate

    Returns dict with:
        direction:   (N, 3) — target returns (regression, 3 horizons)
        trade_gate:  (N,)   — 1 if trade is worthwhile, 0 otherwise
        confidence:  (N,)   — how confident the ensemble should be (0-1)
    """
    n = len(cnn_labels)

    # Direction labels: ground truth returns (what we want to predict)
    direction = cnn_labels.copy()  # (N, 3)

    # Trade gate label: 1 if |actual_return_10s| > threshold AND
    # majority of models predict correct direction
    ret_10s = cnn_labels[:, 2]
    actual_dir = np.sign(ret_10s)

    # Check if models agree with actual direction
    cnn_correct = (np.sign(cnn_preds[:, 2]) == actual_dir)
    ptst_correct = (np.sign(ptst_preds[:, 2]) == actual_dir)
    vol_correct = (np.sign(vol_preds[:, 2]) == actual_dir)
    majority_correct = (cnn_correct.astype(int) + ptst_correct.astype(int) +
                        vol_correct.astype(int)) >= 2

    trade_gate = ((np.abs(ret_10s) > gate_threshold) & majority_correct).astype(np.float32)

    # Confidence label: correlation between prediction direction and actual
    # Use a soft measure: how well do the models' magnitudes predict actual magnitude?
    pred_mean_10s = (cnn_preds[:, 2] + ptst_preds[:, 2] + vol_preds[:, 2]) / 3.0
    # Confidence = 1 when prediction and actual align, 0 when they disagree
    # Using a sigmoid-like scaling of signed agreement
    agreement = pred_mean_10s * ret_10s  # positive = agree
    confidence = 1.0 / (1.0 + np.exp(-agreement * 2.0))  # soft sigmoid
    confidence = confidence.astype(np.float32)

    n_gate = trade_gate.sum()
    logger.info(f"  Labels: gate_rate={100*n_gate/n:.1f}% ({int(n_gate)}/{n}), "
                f"mean_confidence={confidence.mean():.3f}")

    return {
        "direction": direction,
        "trade_gate": trade_gate,
        "confidence": confidence,
    }


# ============================================================
# Dataset
# ============================================================

class ExecMLPDataset(Dataset):
    """Dataset for the execution MLP combiner."""

    def __init__(self, features: np.ndarray,
                 direction_labels: np.ndarray,
                 gate_labels: np.ndarray,
                 confidence_labels: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.direction = torch.from_numpy(direction_labels.astype(np.float32))
        self.gate = torch.from_numpy(gate_labels.astype(np.float32))
        self.confidence = torch.from_numpy(confidence_labels.astype(np.float32))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.direction[idx],
                self.gate[idx], self.confidence[idx])


# ============================================================
# Model Architecture — Multi-task Execution MLP
# ============================================================

class ExecMLPCombiner(nn.Module):
    """
    Multi-task MLP combiner for smart execution.

    Input: concatenation of embeddings + predictions + derived features (~365d)
    Outputs:
        1. direction:   (batch, 3) regression — predicted returns at 3 horizons
        2. trade_gate:  (batch, 1) sigmoid — P(should trade)
        3. confidence:  (batch, 1) sigmoid — confidence score
    """

    def __init__(self, input_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.LayerNorm(hidden // 4),
            nn.GELU(),
        )

        hidden_4 = hidden // 4

        # Head 1: Direction prediction (3 horizons)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_4, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

        # Head 2: Trade gate (binary)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        # Head 3: Confidence score
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"ExecMLPCombiner: input={input_dim}, hidden={hidden}, "
                    f"dropout={dropout}, params={n_params:,}")

    def forward(self, x):
        shared = self.backbone(x)
        direction = self.direction_head(shared)          # (B, 3)
        gate = torch.sigmoid(self.gate_head(shared))     # (B, 1)
        confidence = torch.sigmoid(self.confidence_head(shared))  # (B, 1)
        return direction, gate.squeeze(-1), confidence.squeeze(-1)


# ============================================================
# Metrics
# ============================================================

def compute_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank IC."""
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return float("nan")
    return float(scipy.stats.spearmanr(preds[mask], labels[mask]).statistic)


def compute_sortino(pnl_series: np.ndarray, target: float = 0.0) -> float:
    """Sortino ratio."""
    excess = pnl_series - target
    mean_ret = np.mean(excess)
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("inf") if mean_ret > 0 else 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-10:
        return float("inf") if mean_ret > 0 else 0.0
    return float(mean_ret / downside_std)


def compute_gated_metrics(direction_preds: np.ndarray,
                          gate_preds: np.ndarray,
                          labels: np.ndarray,
                          gate_threshold: float = 0.5) -> Dict:
    """
    Compute metrics for gated (filtered) predictions.

    Only evaluate on samples where gate > threshold.
    """
    gate_mask = gate_preds > gate_threshold
    n_gated = gate_mask.sum()

    if n_gated < 5:
        return {"n_gated": int(n_gated), "gate_rate": 0.0}

    gated_preds = direction_preds[gate_mask]
    gated_labels = labels[gate_mask]

    # Direction accuracy on 10s
    dir_acc_10s = np.mean(np.sign(gated_preds[:, 2]) == np.sign(gated_labels[:, 2]))

    # IC at each horizon
    ics = {}
    for h, name in enumerate(["1s", "5s", "10s"]):
        ics[name] = compute_ic(gated_preds[:, h], gated_labels[:, h])

    # Simulated PnL (enter in predicted direction, 10s horizon)
    entry_dir = np.sign(gated_preds[:, 2])
    pnl_ticks = entry_dir * gated_labels[:, 2] - COMMISSION_TICKS - SPREAD_TICKS * 0.5
    total_pnl = pnl_ticks.sum()
    win_rate = np.mean(pnl_ticks > 0)
    sortino = compute_sortino(pnl_ticks)

    return {
        "n_gated": int(n_gated),
        "gate_rate": float(n_gated / len(gate_preds)),
        "dir_acc_10s": float(dir_acc_10s),
        "ic_1s": ics["1s"],
        "ic_5s": ics["5s"],
        "ic_10s": ics["10s"],
        "total_pnl_ticks": float(total_pnl),
        "mean_pnl_per_trade": float(total_pnl / max(n_gated, 1)),
        "win_rate": float(win_rate),
        "sortino": float(sortino),
    }


# ============================================================
# Feature Assembly
# ============================================================

def assemble_features(date_data: Dict) -> np.ndarray:
    """
    Assemble the full feature vector for each sample.

    Layout (total ~365):
        [0:96]    CNN-Mamba embeddings
        [96:99]   CNN-Mamba predictions (3)
        [99:355]  PatchTST embeddings (256)
        [355:358] PatchTST predictions (3)
        [358:361] Vol LGBM predictions (3)
        [361:371] Derived features (10)
    """
    n = date_data["n_samples"]

    derived = compute_derived_features(
        date_data["cnn_preds"],
        date_data["ptst_preds"],
        date_data["vol_preds"],
        n,
    )

    features = np.concatenate([
        date_data["cnn_embeds"],    # (N, 96)
        date_data["cnn_preds"],     # (N, 3)
        date_data["ptst_embeds"],   # (N, 256)
        date_data["ptst_preds"],    # (N, 3)
        date_data["vol_preds"],     # (N, 3)
        derived,                    # (N, 10)
    ], axis=1)

    return features.astype(np.float32)


# ============================================================
# Walk-Forward Training Loop
# ============================================================

def train_walk_forward(all_dates: List[Dict],
                       args: argparse.Namespace,
                       output_dir: Path) -> Dict:
    """
    Sliding window walk-forward training.

    With 9 dates and default train_days=7:
      Fold 0: train on days 0-6, test on day 7
      Fold 1: train on days 1-7, test on day 8

    HC #0: SLIDING window — oldest day drops when new day added.
    """
    n_dates = len(all_dates)
    train_days = args.train_days
    device = torch.device("cpu")  # Jupiter = CPU only

    if n_dates < train_days + 1:
        logger.error(f"Need at least {train_days + 1} dates, have {n_dates}")
        return {}

    n_folds = n_dates - train_days
    logger.info(f"\nWalk-forward: {n_folds} folds, {train_days} train days each")

    # Determine input dimension from first date
    sample_features = assemble_features(all_dates[0])
    input_dim = sample_features.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    del sample_features

    # Accumulators for concat metrics
    concat_dir_preds = []    # (N, 3)
    concat_gate_preds = []   # (N,)
    concat_conf_preds = []   # (N,)
    concat_labels = []       # (N, 3)
    concat_gate_labels = []  # (N,)

    fold_results = []

    for fold_idx in range(n_folds):
        train_start = fold_idx
        train_end = fold_idx + train_days
        test_idx = train_end

        train_dates = all_dates[train_start:train_end]
        test_date = all_dates[test_idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx} | Train: {train_dates[0]['date']}-{train_dates[-1]['date']} "
                    f"({len(train_dates)} days) | Test: {test_date['date']}")
        logger.info(f"{'='*60}")

        # ---- Build training data ----
        train_feat_list = []
        train_dir_list = []
        train_gate_list = []
        train_conf_list = []

        for td in train_dates:
            feat = assemble_features(td)
            labels = compute_labels(
                td["cnn_labels"], td["cnn_preds"],
                td["ptst_preds"], td["vol_preds"],
                gate_threshold=args.gate_threshold,
            )
            train_feat_list.append(feat)
            train_dir_list.append(labels["direction"])
            train_gate_list.append(labels["trade_gate"])
            train_conf_list.append(labels["confidence"])

        train_features = np.concatenate(train_feat_list, axis=0)
        train_direction = np.concatenate(train_dir_list, axis=0)
        train_gate = np.concatenate(train_gate_list, axis=0)
        train_confidence = np.concatenate(train_conf_list, axis=0)

        # Normalize features (train stats only — no leakage)
        feat_mean = train_features.mean(axis=0)
        feat_std = np.maximum(train_features.std(axis=0), 1e-8)
        train_features = (train_features - feat_mean) / feat_std

        # ---- Build test data ----
        test_features = assemble_features(test_date)
        test_label_dict = compute_labels(
            test_date["cnn_labels"], test_date["cnn_preds"],
            test_date["ptst_preds"], test_date["vol_preds"],
            gate_threshold=args.gate_threshold,
        )
        test_direction = test_label_dict["direction"]
        test_gate = test_label_dict["trade_gate"]
        test_confidence = test_label_dict["confidence"]

        # Normalize test with train stats (no leakage)
        test_features = (test_features - feat_mean) / feat_std

        # ---- DataLoaders ----
        train_ds = ExecMLPDataset(train_features, train_direction, train_gate, train_confidence)
        test_ds = ExecMLPDataset(test_features, test_direction, test_gate, test_confidence)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                                 num_workers=0, pin_memory=False)

        n_train = len(train_ds)
        n_test = len(test_ds)
        logger.info(f"  Train: {n_train} samples | Test: {n_test} samples")

        # ---- Model (fresh per fold) ----
        model = ExecMLPCombiner(
            input_dim=input_dim,
            hidden=args.hidden,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Scheduler: cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )

        # Loss functions
        mse_fn = nn.MSELoss()
        bce_fn = nn.BCELoss()

        # Handle class imbalance in trade gate
        gate_pos_ratio = train_gate.mean()
        gate_weight = 1.0 / max(gate_pos_ratio, 0.01)  # upweight positive class
        gate_weight = min(gate_weight, 10.0)  # cap
        logger.info(f"  Gate pos ratio: {gate_pos_ratio:.3f}, weight: {gate_weight:.2f}")

        # ---- Training loop ----
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            epoch_dir_loss = 0.0
            epoch_gate_loss = 0.0
            epoch_conf_loss = 0.0
            n_batches = 0

            for feat, dir_label, gate_label, conf_label in train_loader:
                feat = feat.to(device)
                dir_label = dir_label.to(device)
                gate_label = gate_label.to(device)
                conf_label = conf_label.to(device)

                optimizer.zero_grad()

                dir_pred, gate_pred, conf_pred = model(feat)

                # Multi-task loss
                loss_dir = mse_fn(dir_pred, dir_label)
                loss_gate = bce_fn(gate_pred, gate_label) * gate_weight
                loss_conf = mse_fn(conf_pred, conf_label)

                # Weighted combination
                loss = (args.loss_weight_dir * loss_dir +
                        args.loss_weight_gate * loss_gate +
                        args.loss_weight_conf * loss_conf)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_dir_loss += loss_dir.item()
                epoch_gate_loss += loss_gate.item()
                epoch_conf_loss += loss_conf.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_dir = epoch_dir_loss / max(n_batches, 1)
            avg_gate = epoch_gate_loss / max(n_batches, 1)
            avg_conf = epoch_conf_loss / max(n_batches, 1)

            # ---- Validation ----
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for feat, dir_label, gate_label, conf_label in test_loader:
                    feat = feat.to(device)
                    dir_label = dir_label.to(device)
                    gate_label = gate_label.to(device)
                    conf_label = conf_label.to(device)

                    dir_pred, gate_pred, conf_pred = model(feat)
                    loss_dir = mse_fn(dir_pred, dir_label)
                    loss_gate = bce_fn(gate_pred, gate_label) * gate_weight
                    loss_conf = mse_fn(conf_pred, conf_label)
                    loss = (args.loss_weight_dir * loss_dir +
                            args.loss_weight_gate * loss_gate +
                            args.loss_weight_conf * loss_conf)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)

            if epoch % 5 == 0 or epoch == args.epochs - 1:
                logger.info(f"  Epoch {epoch:>3d}: loss={avg_loss:.4f} "
                            f"(dir={avg_dir:.4f} gate={avg_gate:.4f} conf={avg_conf:.4f}) "
                            f"| val_loss={avg_val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # ---- OOT Evaluation ----
        model.eval()
        all_dir_preds = []
        all_gate_preds = []
        all_conf_preds = []

        with torch.no_grad():
            for feat, _, _, _ in test_loader:
                feat = feat.to(device)
                dir_pred, gate_pred, conf_pred = model(feat)
                all_dir_preds.append(dir_pred.cpu().numpy())
                all_gate_preds.append(gate_pred.cpu().numpy())
                all_conf_preds.append(conf_pred.cpu().numpy())

        oot_dir = np.concatenate(all_dir_preds, axis=0)      # (N, 3)
        oot_gate = np.concatenate(all_gate_preds, axis=0)     # (N,)
        oot_conf = np.concatenate(all_conf_preds, axis=0)     # (N,)

        # Per-horizon IC (unfiltered)
        ic_results = {}
        for h, name in enumerate(["1s", "5s", "10s"]):
            ic = compute_ic(oot_dir[:, h], test_direction[:, h])
            ic_results[name] = ic
            logger.info(f"  IC_{name}: {ic:.4f}")

        # Gate accuracy
        gate_acc = np.mean((oot_gate > 0.5).astype(float) == test_gate)
        gate_precision = 0.0
        gate_pred_pos = (oot_gate > 0.5).sum()
        if gate_pred_pos > 0:
            gate_precision = float(((oot_gate > 0.5) & (test_gate > 0.5)).sum() / gate_pred_pos)
        logger.info(f"  Gate accuracy: {gate_acc:.4f} | precision: {gate_precision:.4f} "
                    f"| pred_rate: {gate_pred_pos/len(oot_gate):.3f}")

        # Gated metrics (filtered by trade gate)
        gated = compute_gated_metrics(oot_dir, oot_gate, test_direction, gate_threshold=0.5)
        logger.info(f"  Gated metrics: n={gated.get('n_gated', 0)}, "
                    f"IC_10s={gated.get('ic_10s', float('nan')):.4f}, "
                    f"PnL={gated.get('total_pnl_ticks', 0):.1f} ticks, "
                    f"WR={gated.get('win_rate', 0):.3f}, "
                    f"Sortino={gated.get('sortino', 0):.3f}")

        # Also compute gated metrics at higher threshold
        gated_70 = compute_gated_metrics(oot_dir, oot_gate, test_direction, gate_threshold=0.7)
        if gated_70.get("n_gated", 0) > 5:
            logger.info(f"  Gated@0.7: n={gated_70['n_gated']}, "
                        f"IC_10s={gated_70.get('ic_10s', float('nan')):.4f}, "
                        f"PnL={gated_70.get('total_pnl_ticks', 0):.1f} ticks, "
                        f"WR={gated_70.get('win_rate', 0):.3f}")

        # Accumulate for concat metrics
        concat_dir_preds.append(oot_dir)
        concat_gate_preds.append(oot_gate)
        concat_conf_preds.append(oot_conf)
        concat_labels.append(test_direction)
        concat_gate_labels.append(test_gate)

        # Save fold predictions
        fold_pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
        np.savez_compressed(str(fold_pred_path),
            predictions=oot_dir,
            gate_predictions=oot_gate,
            confidence_predictions=oot_conf,
            labels=test_direction,
            gate_labels=test_gate,
            date=test_date["date"],
            horizons=np.array(["1s", "5s", "10s"]),
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        # Save model weights
        model_path = output_dir / f"fold_{fold_idx:02d}_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "fold_idx": fold_idx,
            "test_date": test_date["date"],
            "train_dates": [td["date"] for td in train_dates],
        }, str(model_path))

        fold_result = {
            "fold": fold_idx,
            "test_date": test_date["date"],
            "n_train": n_train,
            "n_test": n_test,
            "ic_1s": ic_results["1s"],
            "ic_5s": ic_results["5s"],
            "ic_10s": ic_results["10s"],
            "gate_accuracy": float(gate_acc),
            "gate_precision": float(gate_precision),
            "gated_metrics": gated,
            "gated_70_metrics": gated_70,
            "best_epoch": args.epochs - patience_counter if patience_counter > 0 else args.epochs,
        }
        fold_results.append(fold_result)

        # MLflow per-fold logging
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                f"fold_{fold_idx}_ic_1s": ic_results["1s"],
                f"fold_{fold_idx}_ic_5s": ic_results["5s"],
                f"fold_{fold_idx}_ic_10s": ic_results["10s"],
                f"fold_{fold_idx}_gate_acc": float(gate_acc),
                f"fold_{fold_idx}_gate_precision": float(gate_precision),
                f"fold_{fold_idx}_gated_pnl": gated.get("total_pnl_ticks", 0),
                f"fold_{fold_idx}_gated_sortino": gated.get("sortino", 0),
            })
            mlflow.log_artifact(str(fold_pred_path))

        # Cleanup
        del model, optimizer, scheduler, train_ds, test_ds
        del train_features, test_features
        gc.collect()

    # ============================================================
    # Concat Metrics (primary evaluation)
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"CONCAT METRICS — {n_folds} folds")
    logger.info(f"{'='*60}")

    all_dir = np.concatenate(concat_dir_preds, axis=0)
    all_gate = np.concatenate(concat_gate_preds, axis=0)
    all_conf = np.concatenate(concat_conf_preds, axis=0)
    all_labels = np.concatenate(concat_labels, axis=0)
    all_gate_labels = np.concatenate(concat_gate_labels, axis=0)

    concat_ics = {}
    for h, name in enumerate(["1s", "5s", "10s"]):
        ic = compute_ic(all_dir[:, h], all_labels[:, h])
        concat_ics[name] = ic
        logger.info(f"  Concat IC_{name}: {ic:.4f}")

    # Concat gate accuracy
    concat_gate_acc = np.mean((all_gate > 0.5).astype(float) == all_gate_labels)
    logger.info(f"  Concat gate accuracy: {concat_gate_acc:.4f}")

    # Concat gated metrics
    concat_gated = compute_gated_metrics(all_dir, all_gate, all_labels, gate_threshold=0.5)
    logger.info(f"  Concat gated: n={concat_gated.get('n_gated', 0)}, "
                f"IC_10s={concat_gated.get('ic_10s', float('nan')):.4f}, "
                f"PnL={concat_gated.get('total_pnl_ticks', 0):.1f} ticks, "
                f"WR={concat_gated.get('win_rate', 0):.3f}, "
                f"Sortino={concat_gated.get('sortino', 0):.3f}")

    concat_gated_70 = compute_gated_metrics(all_dir, all_gate, all_labels, gate_threshold=0.7)
    if concat_gated_70.get("n_gated", 0) > 5:
        logger.info(f"  Concat gated@0.7: n={concat_gated_70['n_gated']}, "
                    f"IC_10s={concat_gated_70.get('ic_10s', float('nan')):.4f}, "
                    f"PnL={concat_gated_70.get('total_pnl_ticks', 0):.1f} ticks, "
                    f"WR={concat_gated_70.get('win_rate', 0):.3f}")

    # MLflow concat metrics
    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            "concat_ic_1s": concat_ics["1s"],
            "concat_ic_5s": concat_ics["5s"],
            "concat_ic_10s": concat_ics["10s"],
            "concat_gate_acc": float(concat_gate_acc),
            "concat_gated_pnl": concat_gated.get("total_pnl_ticks", 0),
            "concat_gated_sortino": concat_gated.get("sortino", 0),
            "concat_gated_win_rate": concat_gated.get("win_rate", 0),
            "concat_gated_ic_10s": concat_gated.get("ic_10s", float("nan")),
            "n_total_samples": len(all_dir),
        })

    # Save concat predictions
    concat_path = output_dir / "concat_oot_predictions.npz"
    np.savez_compressed(str(concat_path),
        predictions=all_dir,
        gate_predictions=all_gate,
        confidence_predictions=all_conf,
        labels=all_labels,
        gate_labels=all_gate_labels,
        horizons=np.array(["1s", "5s", "10s"]),
    )
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(str(concat_path))

    results = {
        "n_folds": n_folds,
        "concat_ic_1s": concat_ics["1s"],
        "concat_ic_5s": concat_ics["5s"],
        "concat_ic_10s": concat_ics["10s"],
        "concat_gate_accuracy": float(concat_gate_acc),
        "concat_gated_metrics": concat_gated,
        "concat_gated_70_metrics": concat_gated_70,
        "per_fold": fold_results,
    }

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Smart Execution MLP Combiner — multi-model ensemble training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    parser.add_argument("--cnn-mamba-dir", type=str, default=None,
                        help="CNN-Mamba v2 predictions directory")
    parser.add_argument("--patchtst-dir", type=str, default=None,
                        help="PatchTST predictions directory")
    parser.add_argument("--vol-lgbm-dir", type=str, default=None,
                        help="Vol LGBM v3 predictions directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for predictions and models")

    # Walk-forward
    parser.add_argument("--train-days", type=int, default=7,
                        help="Number of training days per fold (sliding window)")
    parser.add_argument("--dates", type=str, nargs="*", default=None,
                        help="Explicit list of dates (YYYYMMDD) to use; auto-discovered if omitted")

    # Model architecture
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden layer size")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Max epochs per fold")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")

    # Loss weights
    parser.add_argument("--loss-weight-dir", type=float, default=1.0,
                        help="Weight for direction MSE loss")
    parser.add_argument("--loss-weight-gate", type=float, default=0.5,
                        help="Weight for trade gate BCE loss")
    parser.add_argument("--loss-weight-conf", type=float, default=0.3,
                        help="Weight for confidence MSE loss")

    # Label params
    parser.add_argument("--gate-threshold", type=float, default=0.5,
                        help="Ticks threshold for trade gate labels")

    args = parser.parse_args()

    # Override data dirs
    global CNN_MAMBA_DIR, PATCHTST_DIR, VOL_LGBM_DIR
    if args.cnn_mamba_dir:
        CNN_MAMBA_DIR = Path(args.cnn_mamba_dir)
    if args.patchtst_dir:
        PATCHTST_DIR = Path(args.patchtst_dir)
    if args.vol_lgbm_dir:
        VOL_LGBM_DIR = Path(args.vol_lgbm_dir)

    # Output dir
    ts = time.strftime("%Y%m%d_%H%M")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DEFAULT_OUTPUT_DIR / f"run_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Smart Execution MLP Combiner")
    logger.info(f"{'='*60}")
    logger.info(f"Device: CPU (Jupiter)")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Data sources:")
    logger.info(f"  CNN-Mamba v2:  {CNN_MAMBA_DIR}")
    logger.info(f"  PatchTST:      {PATCHTST_DIR}")
    logger.info(f"  Vol LGBM v3:   {VOL_LGBM_DIR}")
    logger.info(f"Walk-forward: sliding {args.train_days}-day window")
    logger.info(f"Model: hidden={args.hidden}, dropout={args.dropout}")
    logger.info(f"Training: epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")
    logger.info(f"Loss weights: dir={args.loss_weight_dir}, gate={args.loss_weight_gate}, "
                f"conf={args.loss_weight_conf}")
    logger.info(f"Gate threshold: {args.gate_threshold} ticks")

    # ---- Discover available data ----
    logger.info(f"\nDiscovering data...")
    cnn_folds = discover_cnn_mamba_folds()
    ptst_folds = discover_patchtst_folds()
    vol_files = discover_vol_lgbm_files()

    logger.info(f"  CNN-Mamba: {len(cnn_folds)} dates found")
    logger.info(f"  PatchTST:  {len(ptst_folds)} dates found")
    logger.info(f"  Vol LGBM:  {len(vol_files)} dates found")

    if cnn_folds:
        logger.info(f"  CNN-Mamba dates: {sorted(cnn_folds.keys())}")
    if ptst_folds:
        logger.info(f"  PatchTST dates:  {sorted(ptst_folds.keys())}")
    if vol_files:
        logger.info(f"  Vol LGBM dates:  {sorted(vol_files.keys())}")

    # Determine overlapping dates
    if args.dates:
        dates_to_use = args.dates
    else:
        # Auto-discover overlap between CNN-Mamba and PatchTST (both required)
        cnn_dates = set(cnn_folds.keys())
        ptst_dates = set(ptst_folds.keys())
        overlap = sorted(cnn_dates & ptst_dates)

        if not overlap:
            # Fallback: use hardcoded overlap dates that exist
            logger.warning("No auto-discovered overlap. Using hardcoded OVERLAP_DATES.")
            overlap = [d for d in OVERLAP_DATES if d in cnn_folds and d in ptst_folds]

        dates_to_use = overlap

    logger.info(f"\nUsing {len(dates_to_use)} dates: {dates_to_use}")

    if len(dates_to_use) < args.train_days + 1:
        logger.error(f"Need at least {args.train_days + 1} dates for walk-forward, "
                     f"have {len(dates_to_use)}. Aborting.")
        sys.exit(1)

    # ---- Load all dates ----
    logger.info(f"\nLoading per-date aligned data...")
    all_dates = []
    for date in dates_to_use:
        data = load_date_data(date, cnn_folds, ptst_folds, vol_files)
        if data is not None:
            all_dates.append(data)
        else:
            logger.warning(f"Date {date}: skipped (data unavailable)")

    if len(all_dates) < args.train_days + 1:
        logger.error(f"Only {len(all_dates)} dates loaded successfully. "
                     f"Need at least {args.train_days + 1}. Aborting.")
        sys.exit(1)

    total_samples = sum(d["n_samples"] for d in all_dates)
    logger.info(f"Loaded {len(all_dates)} dates, {total_samples:,} total aligned samples")

    # ---- MLflow ----
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"exec_mlp_{ts}"
        )
        mlflow.log_params({
            "model": "exec_mlp_combiner",
            "n_dates": len(all_dates),
            "train_days": args.train_days,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "loss_weight_dir": args.loss_weight_dir,
            "loss_weight_gate": args.loss_weight_gate,
            "loss_weight_conf": args.loss_weight_conf,
            "gate_threshold": args.gate_threshold,
            "total_samples": total_samples,
            "node": socket.gethostname(),
            "device": "cpu",
            "walk_forward": f"sliding_{args.train_days}d",
            "dates": ",".join([d["date"] for d in all_dates]),
        })

    try:
        results = train_walk_forward(all_dates, args, output_dir)

        # Save results summary
        summary = {
            "model": "exec_mlp_combiner",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dates_used": [d["date"] for d in all_dates],
            "args": vars(args),
            "results": results,
        }
        summary_path = output_dir / "results_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(summary_path))

        logger.info(f"\nResults saved to {output_dir}")
        logger.info(f"Summary: {summary_path}")
        logger.info("DONE.")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
