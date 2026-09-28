#!/usr/bin/env python3
"""
Execution Stacker — Multi-Model Trade Execution Layer
======================================================
HC #35/36: Stacking model that combines frozen predictions/embeddings from
CNN Mamba v2, PatchTST, and Vol LGBM v3 to make execution decisions.

Modes:
  mlp_signals     — MLP on output signals (9 + context features)
  mlp_embeddings  — MLP on embeddings (CNN 96d + PatchTST 256d + vol 3d = 355)
  rl_signals      — PPO agent with output signals as state
  rl_embeddings   — PPO agent with embedding state

Data sources (on Neptune at /home/nick/Lvl3Quant/):
  CNN Mamba v2:  output/cnn_mamba_v2_smart_v3_mar/fold_XX_oot_predictions.npz
  PatchTST:      output/patchtst_smart_v3_mar/fold_XX_oot_predictions.npz
  Vol LGBM v3:   output/vol_lgbm_v3/fold_XX_predictions.npz

Training rules:
  - SLIDING window walk-forward (HC #0 — NEVER expanding)
  - num_workers=0 on Neptune (HC #25)
  - pin_memory=True for GPU
  - MLflow logging mandatory
  - <5M params (HC #16/#23)
  - fp16/AMP for speed

Usage:
  python train_exec_stacker.py --mode mlp_signals --n-folds 9
  python train_exec_stacker.py --mode rl_signals --n-folds 9 --reward-fn drawdown_penalized
  python train_exec_stacker.py --mode mlp_embeddings --n-folds 9
  python train_exec_stacker.py --mode rl_embeddings --n-folds 9
"""

import os
import sys
import gc
import time
import logging
import argparse
import warnings
import socket
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import scipy.stats

# MLflow
try:
    if int(os.environ.get("DISABLE_MLFLOW", 0)):
        raise ImportError("MLflow disabled via DISABLE_MLFLOW env var")
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow disabled or not installed — skipping experiment tracking")

# ============================================================
# Logging
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

_ts = time.strftime("%Y%m%d_%H%M%S")
log_path = LOG_DIR / f"exec_stacker_{_ts}.log"
_file_stream = open(log_path, "a", buffering=1)
_file_handler = logging.StreamHandler(_file_stream)
_file_handler.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger()


class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


for _h in logging.root.handlers:
    _h.__class__ = _FlushHandler


# ============================================================
# Paths
# ============================================================
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent

# Data source directories — auto-detect Neptune vs Jupiter
CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
PATCHTST_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"
VOL_LGBM_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"

DEFAULT_OUTPUT_DIR = LVL3_ROOT / "output"

# Constants
TICK = 0.25
TICK_VAL = 12.50
COMMISSION_TICKS = 0.376
SPREAD_TICKS = 1.0
HORIZONS = ["1s", "5s", "10s"]

# MLflow
MLFLOW_EXPERIMENT = "exec_stacker"


def _detect_mlflow_uri() -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    try:
        s = socket.create_connection(("localhost", 5000), timeout=2)
        s.close()
        return "http://localhost:5000"
    except Exception:
        pass
    return "http://uranus:5000"


MLFLOW_TRACKING_URI = _detect_mlflow_uri()


# ============================================================
# Data Loading — Align predictions from 3 models per fold
# ============================================================

def load_fold_data(fold_idx: int, use_embeddings: bool = False) -> Optional[Dict]:
    """
    Load and align predictions from all 3 models for a given fold.

    Returns dict with keys:
        cnn_preds:       (N, 3) CNN Mamba v2 predictions
        patchtst_preds:  (N, 3) PatchTST predictions
        vol_preds:       (N, 3) Vol LGBM predictions
        labels:          (N, 3) ground truth labels
        cnn_embeds:      (N, 96)  — only if use_embeddings
        patchtst_embeds: (N, 256) — only if use_embeddings
        n_samples:       int
    """
    cnn_path = CNN_MAMBA_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    ptst_path = PATCHTST_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"

    # Vol LGBM uses date-based naming (vol_v3_YYYYMMDD_predictions.npz), not fold-based
    # Try fold-based first, then fall back to picking Nth date-sorted file
    vol_path = VOL_LGBM_DIR / f"fold_{fold_idx:02d}_predictions.npz"
    vol_available = vol_path.exists()
    if not vol_available:
        # Find date-based vol files sorted chronologically
        vol_files = sorted(VOL_LGBM_DIR.glob("vol_v3_*_predictions.npz"))
        if fold_idx < len(vol_files):
            vol_path = vol_files[fold_idx]
            vol_available = True
            logger.info(f"Fold {fold_idx:02d}: using date-based vol file {vol_path.name}")

    # Check required files exist (CNN + PatchTST mandatory, vol optional)
    missing = []
    if not cnn_path.exists():
        missing.append(f"CNN Mamba: {cnn_path}")
    if not ptst_path.exists():
        missing.append(f"PatchTST: {ptst_path}")

    if missing:
        logger.warning(f"Fold {fold_idx:02d} missing required data: {missing}")
        return None

    try:
        cnn_data = np.load(str(cnn_path), allow_pickle=True)
        ptst_data = np.load(str(ptst_path), allow_pickle=True)
        vol_data = np.load(str(vol_path), allow_pickle=True) if vol_available else None
    except Exception as e:
        logger.error(f"Fold {fold_idx:02d} load error: {e}")
        return None

    cnn_preds = cnn_data["predictions"]      # (N, 3)
    ptst_preds = ptst_data["predictions"]     # (N, 3)
    vol_preds = vol_data["predictions"] if vol_data is not None else np.zeros_like(cnn_preds)
    if not vol_available:
        logger.info(f"Fold {fold_idx:02d}: vol LGBM not available, using zeros")
    labels = cnn_data["labels"]               # (N, 3) — use CNN's labels as ground truth

    # Align by minimum length (all models predict on same event stream)
    n = min(len(cnn_preds), len(ptst_preds), len(vol_preds), len(labels))
    if n < 100:
        logger.warning(f"Fold {fold_idx:02d}: only {n} aligned samples, skipping")
        return None

    result = {
        "cnn_preds": cnn_preds[:n].astype(np.float32),
        "patchtst_preds": ptst_preds[:n].astype(np.float32),
        "vol_preds": vol_preds[:n].astype(np.float32),
        "labels": labels[:n].astype(np.float32),
        "n_samples": n,
        "fold_idx": fold_idx,
    }

    if use_embeddings:
        cnn_embeds = cnn_data.get("embeddings")
        ptst_embeds = ptst_data.get("embeddings")
        if cnn_embeds is None or ptst_embeds is None:
            logger.warning(f"Fold {fold_idx:02d}: embeddings not available")
            return None
        result["cnn_embeds"] = cnn_embeds[:n].astype(np.float32)
        result["patchtst_embeds"] = ptst_embeds[:n].astype(np.float32)

    logger.info(f"Fold {fold_idx:02d}: loaded {n} aligned samples "
                f"(CNN={len(cnn_preds)}, PatchTST={len(ptst_preds)}, "
                f"Vol={len(vol_preds)})")
    return result


def compute_context_features(cnn_preds: np.ndarray,
                             patchtst_preds: np.ndarray,
                             vol_preds: np.ndarray,
                             n_samples: int) -> np.ndarray:
    """
    Compute context features from multi-model predictions.

    Returns (N, n_context) array with:
      - time_of_day proxy (cyclical position within fold)
      - vol_percentile (rolling percentile of vol prediction)
      - cnn_patchtst_agree (direction agreement on 10s horizon)
      - cnn_confidence (magnitude of CNN 10s prediction)
      - patchtst_confidence (magnitude of PatchTST 10s prediction)
      - vol_surprise (vol pred deviation from rolling mean)
      - pred_dispersion (disagreement between models on 10s)
    """
    n_ctx = 7
    ctx = np.zeros((n_samples, n_ctx), dtype=np.float32)

    # Time-of-day proxy: position within the fold (0 to 1)
    ctx[:, 0] = np.linspace(0, 1, n_samples).astype(np.float32)

    # Vol percentile (rolling rank of vol 10s prediction)
    vol_10s = vol_preds[:, 2]  # 10s horizon
    window = min(500, n_samples // 2)
    if window > 10:
        import pandas as pd
        ctx[:, 1] = pd.Series(vol_10s).rolling(window, min_periods=10).rank(pct=True).fillna(0.5).values
    else:
        ctx[:, 1] = 0.5

    # CNN-PatchTST direction agreement on 10s
    cnn_dir = np.sign(cnn_preds[:, 2])
    ptst_dir = np.sign(patchtst_preds[:, 2])
    ctx[:, 2] = (cnn_dir == ptst_dir).astype(np.float32)

    # CNN confidence (abs magnitude of 10s pred, z-normed)
    cnn_abs = np.abs(cnn_preds[:, 2])
    cnn_std = max(cnn_abs.std(), 1e-8)
    ctx[:, 3] = cnn_abs / cnn_std

    # PatchTST confidence
    ptst_abs = np.abs(patchtst_preds[:, 2])
    ptst_std = max(ptst_abs.std(), 1e-8)
    ctx[:, 4] = ptst_abs / ptst_std

    # Vol surprise (deviation from rolling mean)
    if window > 10:
        vol_rmean = pd.Series(vol_10s).rolling(window, min_periods=10).mean().fillna(vol_10s.mean()).values
        vol_rstd = pd.Series(vol_10s).rolling(window, min_periods=10).std().fillna(1e-8).values
        vol_rstd = np.maximum(vol_rstd, 1e-8)
        ctx[:, 5] = (vol_10s - vol_rmean) / vol_rstd
    else:
        ctx[:, 5] = 0.0

    # Prediction dispersion: std of 3 models' 10s predictions
    stacked = np.stack([cnn_preds[:, 2], patchtst_preds[:, 2], vol_preds[:, 2]], axis=1)
    ctx[:, 6] = stacked.std(axis=1)

    return ctx


# ============================================================
# Execution Labels — MFE/MAE-based action labels
# ============================================================

# Actions for MLP classification
ACTION_ENTER_LONG = 0
ACTION_ENTER_SHORT = 1
ACTION_HOLD = 2
ACTION_EXIT = 3  # Only meaningful in position context
N_MLP_ACTIONS = 4


def compute_execution_labels(labels: np.ndarray,
                             cnn_preds: np.ndarray,
                             patchtst_preds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute execution action labels from future price paths.

    Uses MFE/MAE analysis on labels (future returns at 1s, 5s, 10s):
      - ENTER_LONG if positive expected return AND favorable MFE/MAE ratio
      - ENTER_SHORT if negative expected return AND favorable MFE/MAE ratio
      - HOLD otherwise (no clear edge)

    Also computes expected PnL regression targets.

    Args:
        labels:       (N, 3) future returns at 1s, 5s, 10s
        cnn_preds:    (N, 3) CNN predictions
        patchtst_preds: (N, 3) PatchTST predictions

    Returns:
        action_labels: (N,) int — {0: ENTER_LONG, 1: ENTER_SHORT, 2: HOLD}
        pnl_targets:   (N,) float — expected PnL in ticks (for regression)
    """
    n = len(labels)
    action_labels = np.full(n, ACTION_HOLD, dtype=np.int64)
    pnl_targets = np.zeros(n, dtype=np.float32)

    # Use 10s horizon as primary signal
    ret_10s = labels[:, 2]
    ret_5s = labels[:, 1]
    ret_1s = labels[:, 0]

    # MFE/MAE analysis: look at consistency across horizons
    # Favorable = all horizons agree on direction AND 10s > threshold
    long_consistent = (ret_1s > 0) & (ret_5s > 0) & (ret_10s > 0)
    short_consistent = (ret_1s < 0) & (ret_5s < 0) & (ret_10s < 0)

    # Threshold: need 10s return > 0.5 ticks after costs
    min_return = COMMISSION_TICKS + SPREAD_TICKS * 0.5  # ~0.876 ticks min

    # MFE/MAE ratio: max favorable excursion vs max adverse
    # Approximate: if 5s and 10s returns are increasing, MFE is likely high
    mfe_proxy_long = np.maximum(ret_1s, np.maximum(ret_5s, ret_10s))
    mae_proxy_long = np.minimum(ret_1s, np.minimum(ret_5s, ret_10s))
    mfe_proxy_short = -np.minimum(ret_1s, np.minimum(ret_5s, ret_10s))
    mae_proxy_short = -np.maximum(ret_1s, np.maximum(ret_5s, ret_10s))

    # ENTER_LONG: consistent positive, exceeds cost, good MFE/MAE
    long_mask = long_consistent & (ret_10s > min_return) & (mfe_proxy_long > 2 * np.abs(mae_proxy_long + 1e-8))
    action_labels[long_mask] = ACTION_ENTER_LONG

    # ENTER_SHORT: consistent negative, exceeds cost, good MFE/MAE
    short_mask = short_consistent & (-ret_10s > min_return) & (mfe_proxy_short > 2 * np.abs(mae_proxy_short + 1e-8))
    action_labels[short_mask] = ACTION_ENTER_SHORT

    # PnL target: 10s return minus costs (signed)
    pnl_targets = ret_10s.copy()
    pnl_targets[action_labels == ACTION_ENTER_LONG] = ret_10s[action_labels == ACTION_ENTER_LONG] - min_return
    pnl_targets[action_labels == ACTION_ENTER_SHORT] = -ret_10s[action_labels == ACTION_ENTER_SHORT] - min_return
    pnl_targets[action_labels == ACTION_HOLD] = 0.0

    logger.info(f"  Labels: LONG={long_mask.sum()} ({100*long_mask.mean():.1f}%), "
                f"SHORT={short_mask.sum()} ({100*short_mask.mean():.1f}%), "
                f"HOLD={n - long_mask.sum() - short_mask.sum()} "
                f"({100*(1 - long_mask.mean() - short_mask.mean()):.1f}%)")

    return action_labels, pnl_targets


# ============================================================
# Datasets
# ============================================================

class StackerSignalDataset(Dataset):
    """Dataset for signal-based stacker: 9 prediction features + context."""

    def __init__(self, fold_data: Dict, context_features: np.ndarray,
                 action_labels: np.ndarray, pnl_targets: np.ndarray):
        # Concatenate all 3 models' predictions: (N, 9)
        signals = np.concatenate([
            fold_data["cnn_preds"],
            fold_data["patchtst_preds"],
            fold_data["vol_preds"],
        ], axis=1)

        # Add context features: (N, 9 + n_ctx)
        self.features = np.concatenate([signals, context_features], axis=1).astype(np.float32)
        self.action_labels = action_labels
        self.pnl_targets = pnl_targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.action_labels[idx], dtype=torch.long),
            torch.tensor(self.pnl_targets[idx], dtype=torch.float32),
        )


class StackerEmbeddingDataset(Dataset):
    """Dataset for embedding-based stacker: CNN 96d + PatchTST 256d + vol 3d."""

    def __init__(self, fold_data: Dict,
                 action_labels: np.ndarray, pnl_targets: np.ndarray):
        self.features = np.concatenate([
            fold_data["cnn_embeds"],       # (N, 96)
            fold_data["patchtst_embeds"],   # (N, 256)
            fold_data["vol_preds"],         # (N, 3)
        ], axis=1).astype(np.float32)
        self.action_labels = action_labels
        self.pnl_targets = pnl_targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.action_labels[idx], dtype=torch.long),
            torch.tensor(self.pnl_targets[idx], dtype=torch.float32),
        )


# ============================================================
# MLP Architecture
# ============================================================

class ExecutionMLP(nn.Module):
    """
    Lightweight MLP for execution action classification + PnL regression.
    Dual head: classification (action) + regression (expected PnL).
    """

    def __init__(self, input_dim: int, n_actions: int = 3, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 128),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(128, n_actions)
        self.pnl_head = nn.Linear(128, 1)

        # Count params
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"ExecutionMLP: input_dim={input_dim}, hidden={hidden}, "
                    f"n_actions={n_actions}, params={n_params:,}")

    def forward(self, x):
        shared = self.shared(x)
        action_logits = self.action_head(shared)
        pnl_pred = self.pnl_head(shared).squeeze(-1)
        return action_logits, pnl_pred


# ============================================================
# PPO Agent (Extended from rl_execution_agent.py)
# ============================================================

@dataclass
class StackerTradeState:
    """State visible to RL agent — extends TradeState with multi-model signals."""

    # CNN Mamba signals
    cnn_pred_1s: float = 0.0
    cnn_pred_5s: float = 0.0
    cnn_pred_10s: float = 0.0

    # PatchTST signals
    patchtst_pred_1s: float = 0.0
    patchtst_pred_5s: float = 0.0
    patchtst_pred_10s: float = 0.0

    # PatchTST derived
    patchtst_z_score: float = 0.0         # z-normalized prediction
    patchtst_direction_agree: float = 0.0  # 1 if CNN and PatchTST agree on 10s direction
    patchtst_confidence: float = 0.0       # magnitude of PatchTST 10s prediction

    # Vol LGBM signals
    vol_pred_1s: float = 0.0
    vol_pred_5s: float = 0.0
    vol_pred_10s: float = 0.0
    vol_surprise: float = 0.0             # actual vol vs predicted

    # Context
    vol_percentile: float = 0.5
    time_of_day: float = 0.5
    pred_dispersion: float = 0.0

    # Position state
    in_position: float = 0.0
    position_direction: float = 0.0
    unrealized_pnl: float = 0.0
    hold_time_norm: float = 0.0
    max_favorable: float = 0.0
    max_adverse: float = 0.0
    bars_since_trade_norm: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        features = [
            self.cnn_pred_1s, self.cnn_pred_5s, self.cnn_pred_10s,
            self.patchtst_pred_1s, self.patchtst_pred_5s, self.patchtst_pred_10s,
            self.patchtst_z_score,
            self.patchtst_direction_agree,
            self.patchtst_confidence,
            self.vol_pred_1s, self.vol_pred_5s, self.vol_pred_10s,
            self.vol_surprise,
            self.vol_percentile,
            self.time_of_day,
            self.pred_dispersion,
            self.in_position,
            self.position_direction,
            np.clip(self.unrealized_pnl / 50.0, -1, 1),
            self.hold_time_norm,
            np.clip(self.max_favorable / 30.0, 0, 2),
            np.clip(self.max_adverse / 30.0, -2, 0),
            self.bars_since_trade_norm,
        ]
        return torch.tensor(features, dtype=torch.float32)


STACKER_STATE_DIM = 23  # Must match to_tensor output


@dataclass
class StackerEmbTradeState:
    """State with raw embeddings appended."""

    # Base signals (same as StackerTradeState)
    base_features: np.ndarray = field(default_factory=lambda: np.zeros(23, dtype=np.float32))
    # Raw embeddings: CNN (96) + PatchTST (256) = 352
    cnn_embed: np.ndarray = field(default_factory=lambda: np.zeros(96, dtype=np.float32))
    patchtst_embed: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.float32))

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            np.concatenate([self.base_features, self.cnn_embed, self.patchtst_embed]),
            dtype=torch.float32,
        )


STACKER_EMB_STATE_DIM = 23 + 96 + 256  # 375

# RL Actions
RL_ACTION_SKIP = 0
RL_ACTION_ENTER = 1
RL_ACTION_EXIT = 2
N_RL_ACTIONS = 3


class StackerPolicyNetwork(nn.Module):
    """Actor-Critic for the stacker PPO agent."""

    def __init__(self, state_dim: int, n_actions: int = N_RL_ACTIONS, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(f"StackerPolicyNetwork: state_dim={state_dim}, params={n_params:,}")

    def forward(self, state):
        shared = self.shared(state)
        logits = self.actor(shared)
        value = self.critic(shared)
        return logits, value

    def get_action(self, state, deterministic=False):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        log_prob = torch.log(probs[action] + 1e-8)
        return action.item(), log_prob, value


class RewardShaper:
    """Reward functions (from rl_execution_agent.py)."""

    @staticmethod
    def raw_pnl(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        return pnl_ticks

    @staticmethod
    def drawdown_penalized(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        dd_penalty = max(0, -max_dd_ticks) * 0.05
        return pnl_ticks - dd_penalty

    @staticmethod
    def time_efficient(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        time_factor = max(0.5, 1.0 - hold_bars / 500)
        return pnl_ticks * time_factor

    @staticmethod
    def calmar(pnl_ticks, max_dd_ticks=0, hold_bars=0):
        if max_dd_ticks == 0:
            return pnl_ticks
        return pnl_ticks / (1 + abs(max_dd_ticks) * 0.2)


class StackerTradingEnvironment:
    """
    Trading environment that uses multi-model stacker predictions.
    Steps through aligned prediction arrays (not raw bar data).
    Each step = one prediction point (not one bar).
    """

    def __init__(self, cnn_preds, patchtst_preds, vol_preds, labels,
                 context_features, cnn_embeds=None, patchtst_embeds=None,
                 use_embeddings=False, commission=COMMISSION_TICKS):
        self.cnn_preds = cnn_preds          # (N, 3)
        self.ptst_preds = patchtst_preds    # (N, 3)
        self.vol_preds = vol_preds          # (N, 3)
        self.labels = labels                # (N, 3)
        self.ctx = context_features         # (N, 7)
        self.cnn_embeds = cnn_embeds        # (N, 96) or None
        self.ptst_embeds = patchtst_embeds  # (N, 256) or None
        self.use_embeddings = use_embeddings
        self.n = len(cnn_preds)
        self.commission = commission

        # Precompute PatchTST z-scores
        ptst_10s = patchtst_preds[:, 2]
        import pandas as pd
        window = min(500, self.n // 2)
        if window > 10:
            ptst_mean = pd.Series(ptst_10s).rolling(window, min_periods=10).mean().fillna(0).values
            ptst_std = pd.Series(ptst_10s).rolling(window, min_periods=10).std().fillna(1).values
            ptst_std = np.maximum(ptst_std, 1e-8)
            self.ptst_z = (ptst_10s - ptst_mean) / ptst_std
        else:
            self.ptst_z = np.zeros(self.n, dtype=np.float32)

        self.reset()

    def reset(self):
        self.idx = 0
        self.in_position = False
        self.position_dir = 0
        self.entry_idx = 0
        self.entry_ret = 0.0
        self.max_fav = 0.0
        self.max_adv = 0.0
        self.bars_since_trade = 0
        self.trades = []
        self.total_pnl = 0.0
        self.peak_equity = 0.0
        self.max_dd = 0.0
        return self._get_state()

    def _get_state(self):
        i = min(self.idx, self.n - 1)

        unreal_pnl = 0.0
        if self.in_position:
            # Approximate unrealized PnL from labels progression
            unreal_pnl = self.labels[i, 2] * self.position_dir  # 10s return in direction

        base_state = StackerTradeState(
            cnn_pred_1s=float(self.cnn_preds[i, 0]),
            cnn_pred_5s=float(self.cnn_preds[i, 1]),
            cnn_pred_10s=float(self.cnn_preds[i, 2]),
            patchtst_pred_1s=float(self.ptst_preds[i, 0]),
            patchtst_pred_5s=float(self.ptst_preds[i, 1]),
            patchtst_pred_10s=float(self.ptst_preds[i, 2]),
            patchtst_z_score=float(self.ptst_z[i]),
            patchtst_direction_agree=float(self.ctx[i, 2]),
            patchtst_confidence=float(self.ctx[i, 4]),
            vol_pred_1s=float(self.vol_preds[i, 0]),
            vol_pred_5s=float(self.vol_preds[i, 1]),
            vol_pred_10s=float(self.vol_preds[i, 2]),
            vol_surprise=float(self.ctx[i, 5]),
            vol_percentile=float(self.ctx[i, 1]),
            time_of_day=float(self.ctx[i, 0]),
            pred_dispersion=float(self.ctx[i, 6]),
            in_position=float(self.in_position),
            position_direction=float(self.position_dir),
            unrealized_pnl=unreal_pnl,
            hold_time_norm=np.clip((self.idx - self.entry_idx) / 500.0, 0, 2) if self.in_position else 0.0,
            max_favorable=self.max_fav,
            max_adverse=self.max_adv,
            bars_since_trade_norm=np.clip(self.bars_since_trade / 500.0, 0, 2),
        )

        if self.use_embeddings and self.cnn_embeds is not None:
            emb_state = StackerEmbTradeState(
                base_features=base_state.to_tensor().numpy(),
                cnn_embed=self.cnn_embeds[i],
                patchtst_embed=self.ptst_embeds[i],
            )
            return emb_state

        return base_state

    def step(self, action: int):
        i = self.idx
        reward = 0.0
        info = {}

        # End of predictions
        if i >= self.n - 1:
            if self.in_position:
                net_pnl = self.labels[i, 2] * self.position_dir - self.commission - SPREAD_TICKS * 0.5
                reward = net_pnl
                self.total_pnl += net_pnl * TICK_VAL
                self.trades.append({"pnl_ticks": net_pnl, "type": "forced_exit"})
                self.in_position = False
            return self._get_state(), reward, True, info

        if action == RL_ACTION_ENTER and not self.in_position:
            # Enter in majority-vote direction (CNN + PatchTST + Vol consensus)
            votes = (
                np.sign(self.cnn_preds[i, 2]) +
                np.sign(self.ptst_preds[i, 2]) +
                np.sign(self.vol_preds[i, 2])
            )
            direction = 1 if votes > 0 else -1
            self.in_position = True
            self.position_dir = direction
            self.entry_idx = i
            self.entry_ret = self.labels[i, 2]
            self.max_fav = 0.0
            self.max_adv = 0.0
            self.bars_since_trade = 0
            info["action"] = "enter"

        elif action == RL_ACTION_EXIT and self.in_position:
            # Exit: PnL based on 10s label at entry vs now
            exit_pnl = self.labels[i, 2] * self.position_dir
            net_pnl = exit_pnl - self.commission - SPREAD_TICKS * 0.5
            reward = net_pnl
            self.total_pnl += net_pnl * TICK_VAL
            self.trades.append({
                "pnl_ticks": net_pnl,
                "hold_steps": i - self.entry_idx,
                "max_fav": self.max_fav,
                "max_adv": self.max_adv,
                "type": "rl_exit",
            })
            self.in_position = False
            self.position_dir = 0
            info["action"] = "exit"
            info["pnl"] = net_pnl

        else:
            info["action"] = "hold" if self.in_position else "skip"

        # Update position tracking
        if self.in_position:
            current_pnl = self.labels[i, 2] * self.position_dir
            self.max_fav = max(self.max_fav, current_pnl)
            self.max_adv = min(self.max_adv, current_pnl)

            # Force exit after 500 steps
            if i - self.entry_idx >= 500:
                net_pnl = self.labels[i, 2] * self.position_dir - self.commission - SPREAD_TICKS * 0.5
                reward = net_pnl
                self.total_pnl += net_pnl * TICK_VAL
                self.trades.append({"pnl_ticks": net_pnl, "type": "max_hold_exit"})
                self.in_position = False

        # Drawdown tracking
        self.peak_equity = max(self.peak_equity, self.total_pnl)
        dd = self.peak_equity - self.total_pnl
        self.max_dd = max(self.max_dd, dd)

        self.bars_since_trade += 1
        self.idx += 1

        done = self.idx >= self.n - 1
        return self._get_state(), reward, done, info


class StackerPPOTrainer:
    """PPO training loop for stacker agent."""

    def __init__(self, state_dim: int, device="cuda", lr=3e-4, gamma=0.99,
                 eps_clip=0.2, reward_fn="drawdown_penalized"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.policy = StackerPolicyNetwork(state_dim=state_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

        reward_fns = {
            "raw_pnl": RewardShaper.raw_pnl,
            "drawdown_penalized": RewardShaper.drawdown_penalized,
            "time_efficient": RewardShaper.time_efficient,
            "calmar": RewardShaper.calmar,
        }
        self.reward_fn = reward_fns.get(reward_fn, RewardShaper.drawdown_penalized)
        self.reward_fn_name = reward_fn
        logger.info(f"StackerPPO | Device: {self.device} | Reward: {reward_fn} | "
                    f"State dim: {state_dim}")

    def collect_episode(self, env: StackerTradingEnvironment):
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        state = env.reset()
        done = False

        while not done:
            state_tensor = state.to_tensor().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(state_tensor.squeeze(0))
            next_state, reward, done, info = env.step(action)

            if info.get("action") == "exit":
                trade = env.trades[-1] if env.trades else {}
                reward = self.reward_fn(
                    reward,
                    max_dd_ticks=trade.get("max_adv", 0),
                    hold_bars=trade.get("hold_steps", 0),
                )

            states.append(state_tensor.squeeze(0))
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.squeeze())
            dones.append(done)
            state = next_state

        return states, actions, rewards, log_probs, values, dones

    def compute_returns(self, rewards, values, dones):
        returns = []
        gae = 0
        next_value = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0
                gae = 0
            delta = rewards[i] + self.gamma * next_value - values[i].item()
            gae = delta + self.gamma * 0.95 * gae
            returns.insert(0, gae + values[i].item())
            next_value = values[i].item()
        return torch.tensor(returns, dtype=torch.float32).to(self.device)

    def update(self, states, actions, old_log_probs, returns, values):
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        old_log_probs = torch.stack(old_log_probs).to(self.device)
        advantages = returns - torch.stack(values).detach().to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(4):  # PPO epochs
            logits, new_values = self.policy(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()

    def train_fold(self, train_envs: list, n_epochs: int = 50):
        """Train on multiple environments for one fold."""
        logger.info(f"RL training on {len(train_envs)} environments for {n_epochs} epochs")

        best_avg_pnl = -np.inf
        best_state = None

        for epoch in range(n_epochs):
            epoch_pnl = []
            epoch_trades = []

            for env in train_envs:
                states, actions, rewards, log_probs, values, dones = self.collect_episode(env)
                if len(states) < 10:
                    continue
                returns = self.compute_returns(rewards, values, dones)
                self.update(states, actions, log_probs, returns, values)
                epoch_pnl.append(env.total_pnl)
                epoch_trades.append(len(env.trades))

            avg_pnl = np.mean(epoch_pnl) if epoch_pnl else 0
            avg_trades = np.mean(epoch_trades) if epoch_trades else 0

            if epoch % 10 == 0:
                logger.info(f"  Epoch {epoch:>3d}: Avg P&L ${avg_pnl:>8,.2f} | "
                            f"Avg trades {avg_trades:.1f}")

            if avg_pnl > best_avg_pnl:
                best_avg_pnl = avg_pnl
                best_state = {k: v.clone() for k, v in self.policy.state_dict().items()}

        if best_state is not None:
            self.policy.load_state_dict(best_state)

        return best_avg_pnl

    def evaluate(self, envs: list):
        self.policy.eval()
        results = []
        with torch.no_grad():
            for env in envs:
                state = env.reset()
                done = False
                while not done:
                    state_tensor = state.to_tensor().to(self.device)
                    action, _, _ = self.policy.get_action(state_tensor, deterministic=True)
                    state, _, done, _ = env.step(action)
                results.append({
                    "pnl": env.total_pnl,
                    "trades": len(env.trades),
                    "max_dd": env.max_dd,
                })
        self.policy.train()
        return results


# ============================================================
# Metrics
# ============================================================

def compute_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(preds) & np.isfinite(labels)
    if mask.sum() < 10:
        return float("nan")
    return float(scipy.stats.spearmanr(preds[mask], labels[mask]).statistic)


def compute_sortino(pnl_series: np.ndarray, target=0.0) -> float:
    excess = pnl_series - target
    mean_ret = np.mean(excess)
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("inf") if mean_ret > 0 else 0.0
    downside_std = np.std(downside)
    if downside_std < 1e-10:
        return float("inf") if mean_ret > 0 else 0.0
    return float(mean_ret / downside_std)


def compute_conviction_metrics(preds: np.ndarray, labels: np.ndarray,
                               action_preds: np.ndarray = None) -> Dict:
    """Compute metrics at top-10%, top-5%, top-1% conviction tiers."""
    metrics = {}
    abs_pred = np.abs(preds)

    for pct_name, pct in [("top10", 90), ("top5", 95), ("top1", 99)]:
        thresh = np.percentile(abs_pred, pct)
        mask = abs_pred >= thresh
        n_in = mask.sum()
        if n_in < 5:
            continue

        tier_preds = preds[mask]
        tier_labels = labels[mask]
        ic = compute_ic(tier_preds, tier_labels)
        dir_acc = np.mean(np.sign(tier_preds) == np.sign(tier_labels))

        # Simulated PnL for tier
        entry_dir = np.sign(tier_preds)
        pnl_ticks = entry_dir * tier_labels - COMMISSION_TICKS - SPREAD_TICKS * 0.5
        total_pnl = pnl_ticks.sum()
        win_rate = np.mean(pnl_ticks > 0)
        sortino = compute_sortino(pnl_ticks)

        metrics[pct_name] = {
            "n": int(n_in),
            "ic": float(ic),
            "dir_acc": float(dir_acc),
            "total_pnl_ticks": float(total_pnl),
            "win_rate": float(win_rate),
            "sortino": float(sortino),
        }

    return metrics


# ============================================================
# Training — MLP Mode
# ============================================================

def train_mlp(args, all_folds: List[Dict], output_dir: Path, device: torch.device):
    """Walk-forward MLP training with sliding window."""

    use_embeddings = args.mode == "mlp_embeddings"
    n_folds = len(all_folds)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # Determine input dim from first fold
    first = all_folds[0]
    if use_embeddings:
        input_dim = first["cnn_embeds"].shape[1] + first["patchtst_embeds"].shape[1] + 3
        logger.info(f"Embedding mode: input_dim={input_dim} "
                    f"(CNN {first['cnn_embeds'].shape[1]}d + "
                    f"PatchTST {first['patchtst_embeds'].shape[1]}d + vol 3d)")
    else:
        ctx = compute_context_features(
            first["cnn_preds"], first["patchtst_preds"],
            first["vol_preds"], first["n_samples"])
        input_dim = 9 + ctx.shape[1]  # 9 predictions + context
        logger.info(f"Signal mode: input_dim={input_dim} (9 preds + {ctx.shape[1]} context)")

    # Collect concat predictions for final IC
    concat_preds_10s = []
    concat_labels_10s = []
    fold_results = []

    for fold_idx in range(n_folds):
        fold = all_folds[fold_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx:02d} (original fold {fold['fold_idx']:02d}) | "
                    f"{fold['n_samples']} samples")
        logger.info(f"{'='*60}")

        # Sliding window: train on previous folds, test on current
        # Need at least 1 fold for training
        if fold_idx == 0:
            logger.info(f"  Fold 0: skipping (no training data yet)")
            continue

        # Sliding window: use up to 60 previous folds (equivalent to 60d)
        train_start = max(0, fold_idx - 60)
        train_folds = all_folds[train_start:fold_idx]
        oot_fold = fold

        logger.info(f"  Train folds: {train_start}..{fold_idx-1} ({len(train_folds)} folds), "
                    f"OOT: fold {fold_idx}")

        # Build training data
        train_features_list = []
        train_action_list = []
        train_pnl_list = []

        for tf in train_folds:
            ctx = compute_context_features(
                tf["cnn_preds"], tf["patchtst_preds"],
                tf["vol_preds"], tf["n_samples"])
            action_labels, pnl_targets = compute_execution_labels(
                tf["labels"], tf["cnn_preds"], tf["patchtst_preds"])

            if use_embeddings:
                features = np.concatenate([
                    tf["cnn_embeds"], tf["patchtst_embeds"], tf["vol_preds"]
                ], axis=1)
            else:
                signals = np.concatenate([
                    tf["cnn_preds"], tf["patchtst_preds"], tf["vol_preds"]
                ], axis=1)
                features = np.concatenate([signals, ctx], axis=1)

            train_features_list.append(features)
            train_action_list.append(action_labels)
            train_pnl_list.append(pnl_targets)

        train_features = np.concatenate(train_features_list, axis=0).astype(np.float32)
        train_actions = np.concatenate(train_action_list, axis=0)
        train_pnl = np.concatenate(train_pnl_list, axis=0).astype(np.float32)

        # Normalize features (train stats only — no leakage)
        feat_mean = train_features.mean(axis=0)
        feat_std = np.maximum(train_features.std(axis=0), 1e-8)
        train_features = (train_features - feat_mean) / feat_std

        # OOT data
        oot_ctx = compute_context_features(
            oot_fold["cnn_preds"], oot_fold["patchtst_preds"],
            oot_fold["vol_preds"], oot_fold["n_samples"])
        oot_actions, oot_pnl = compute_execution_labels(
            oot_fold["labels"], oot_fold["cnn_preds"], oot_fold["patchtst_preds"])

        if use_embeddings:
            oot_features = np.concatenate([
                oot_fold["cnn_embeds"], oot_fold["patchtst_embeds"], oot_fold["vol_preds"]
            ], axis=1).astype(np.float32)
        else:
            oot_signals = np.concatenate([
                oot_fold["cnn_preds"], oot_fold["patchtst_preds"], oot_fold["vol_preds"]
            ], axis=1)
            oot_features = np.concatenate([oot_signals, oot_ctx], axis=1).astype(np.float32)

        oot_features = (oot_features - feat_mean) / feat_std

        # DataLoaders
        train_tensor = torch.from_numpy(train_features)
        train_act_tensor = torch.from_numpy(train_actions)
        train_pnl_tensor = torch.from_numpy(train_pnl)
        train_ds = torch.utils.data.TensorDataset(train_tensor, train_act_tensor, train_pnl_tensor)
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,
                                  num_workers=0, pin_memory=True)

        oot_tensor = torch.from_numpy(oot_features)
        oot_act_tensor = torch.from_numpy(oot_actions)
        oot_pnl_tensor = torch.from_numpy(oot_pnl)
        oot_ds = torch.utils.data.TensorDataset(oot_tensor, oot_act_tensor, oot_pnl_tensor)
        oot_loader = DataLoader(oot_ds, batch_size=1024, shuffle=False,
                                num_workers=0, pin_memory=True)

        # Model — fresh per fold (no warm start)
        n_actions = 3  # ENTER_LONG, ENTER_SHORT, HOLD (collapse EXIT into HOLD for entry-only)
        model = ExecutionMLP(input_dim=input_dim, n_actions=n_actions, hidden=256).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        # Class weights (handle imbalance)
        class_counts = np.bincount(train_actions, minlength=n_actions)
        class_weights = 1.0 / (class_counts + 1)
        class_weights = class_weights / class_weights.sum() * n_actions
        ce_weight = torch.tensor(class_weights[:n_actions], dtype=torch.float32).to(device)

        ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weight)
        mse_loss_fn = nn.MSELoss()

        # Train
        epochs = int(os.environ.get("STACKER_EPOCHS", 20))
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_feats, batch_acts, batch_pnl in train_loader:
                batch_feats = batch_feats.to(device)
                batch_acts = batch_acts.to(device)
                batch_pnl = batch_pnl.to(device)

                optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    action_logits, pnl_pred = model(batch_feats)
                    loss_ce = ce_loss_fn(action_logits, batch_acts)
                    loss_pnl = mse_loss_fn(pnl_pred, batch_pnl)
                    loss = loss_ce + 0.1 * loss_pnl

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Validation loss on OOT
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_feats, batch_acts, batch_pnl in oot_loader:
                    batch_feats = batch_feats.to(device)
                    batch_acts = batch_acts.to(device)
                    batch_pnl = batch_pnl.to(device)
                    with autocast(enabled=use_amp):
                        action_logits, pnl_pred = model(batch_feats)
                        loss_ce = ce_loss_fn(action_logits, batch_acts)
                        loss_pnl = mse_loss_fn(pnl_pred, batch_pnl)
                        loss = loss_ce + 0.1 * loss_pnl
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)

            if epoch % 5 == 0:
                logger.info(f"  Epoch {epoch:>2d}: train_loss={avg_loss:.4f}, "
                            f"val_loss={avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        # Load best model
        if best_state:
            model.load_state_dict(best_state)

        # OOT evaluation
        model.eval()
        all_oot_logits = []
        all_oot_pnl_preds = []
        with torch.no_grad():
            for batch_feats, _, _ in oot_loader:
                batch_feats = batch_feats.to(device)
                with autocast(enabled=use_amp):
                    action_logits, pnl_pred = model(batch_feats)
                all_oot_logits.append(action_logits.float().cpu().numpy())
                all_oot_pnl_preds.append(pnl_pred.float().cpu().numpy())

        oot_logits = np.concatenate(all_oot_logits, axis=0)
        oot_pnl_preds = np.concatenate(all_oot_pnl_preds, axis=0)
        oot_action_preds = oot_logits.argmax(axis=1)

        # Action accuracy
        action_acc = np.mean(oot_action_preds == oot_actions)

        # PnL regression IC
        pnl_ic = compute_ic(oot_pnl_preds, oot_pnl)

        # Simulated execution PnL
        labels_10s = oot_fold["labels"][:, 2]
        trade_pnl = []
        for i in range(len(oot_action_preds)):
            act = oot_action_preds[i]
            if act == ACTION_ENTER_LONG:
                pnl = labels_10s[i] - COMMISSION_TICKS - SPREAD_TICKS * 0.5
                trade_pnl.append(pnl)
            elif act == ACTION_ENTER_SHORT:
                pnl = -labels_10s[i] - COMMISSION_TICKS - SPREAD_TICKS * 0.5
                trade_pnl.append(pnl)

        trade_pnl = np.array(trade_pnl) if trade_pnl else np.array([0.0])
        total_pnl_ticks = trade_pnl.sum()
        n_trades = len(trade_pnl)
        win_rate = np.mean(trade_pnl > 0) if n_trades > 0 else 0.0
        sortino = compute_sortino(trade_pnl)

        # IC on 10s predictions (use PnL pred as signal)
        ic_10s = compute_ic(oot_pnl_preds, labels_10s)

        # Conviction tier metrics
        conviction_metrics = compute_conviction_metrics(oot_pnl_preds, labels_10s)

        logger.info(f"  OOT Results:")
        logger.info(f"    Action accuracy: {action_acc:.3f}")
        logger.info(f"    PnL IC: {pnl_ic:.4f} | 10s IC: {ic_10s:.4f}")
        logger.info(f"    Trades: {n_trades} | Total PnL: {total_pnl_ticks:.2f} ticks "
                    f"(${total_pnl_ticks * TICK_VAL:.2f})")
        logger.info(f"    Win rate: {win_rate:.3f} | Sortino: {sortino:.3f}")
        for tier, m in conviction_metrics.items():
            logger.info(f"    {tier}: n={m['n']}, IC={m['ic']:.4f}, "
                        f"win={m['win_rate']:.3f}, sortino={m['sortino']:.3f}")

        # Save weights
        ckpt_path = output_dir / f"fold_{fold_idx:02d}_model.pt"
        torch.save(model.state_dict(), str(ckpt_path))

        # Save predictions
        pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
        np.savez_compressed(str(pred_path),
            action_logits=oot_logits,
            pnl_predictions=oot_pnl_preds,
            action_labels=oot_actions,
            pnl_targets=oot_pnl,
            labels=oot_fold["labels"],
            ic_10s=np.array(ic_10s),
            pnl_ic=np.array(pnl_ic),
            action_accuracy=np.array(action_acc),
        )

        # Collect for concat
        concat_preds_10s.append(oot_pnl_preds)
        concat_labels_10s.append(labels_10s)

        fold_results.append({
            "fold": fold_idx,
            "action_acc": action_acc,
            "pnl_ic": pnl_ic,
            "ic_10s": ic_10s,
            "n_trades": n_trades,
            "total_pnl_ticks": total_pnl_ticks,
            "win_rate": win_rate,
            "sortino": sortino,
            "conviction": conviction_metrics,
        })

        # MLflow per-fold metrics
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                f"action_acc_fold{fold_idx:02d}": action_acc,
                f"pnl_ic_fold{fold_idx:02d}": pnl_ic,
                f"ic_10s_fold{fold_idx:02d}": ic_10s,
                f"n_trades_fold{fold_idx:02d}": n_trades,
                f"pnl_ticks_fold{fold_idx:02d}": total_pnl_ticks,
                f"win_rate_fold{fold_idx:02d}": win_rate,
                f"sortino_fold{fold_idx:02d}": sortino,
            }, step=fold_idx)
            mlflow.log_artifact(str(pred_path), artifact_path=f"fold_{fold_idx:02d}")
            mlflow.log_artifact(str(ckpt_path), artifact_path=f"fold_{fold_idx:02d}")

        # Cleanup
        del model, optimizer, train_ds, oot_ds, train_loader, oot_loader
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Concat IC (primary metric)
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info("CONCAT METRICS (all OOT folds combined)")
    logger.info(f"{'='*60}")

    if concat_preds_10s:
        all_preds = np.concatenate(concat_preds_10s)
        all_labels = np.concatenate(concat_labels_10s)
        concat_ic = compute_ic(all_preds, all_labels)
        concat_conviction = compute_conviction_metrics(all_preds, all_labels)

        logger.info(f"  Concat IC (10s): {concat_ic:.4f}")
        logger.info(f"  Total samples: {len(all_preds)}")
        for tier, m in concat_conviction.items():
            logger.info(f"  {tier}: n={m['n']}, IC={m['ic']:.4f}, "
                        f"win={m['win_rate']:.3f}, sortino={m['sortino']:.3f}, "
                        f"PnL={m['total_pnl_ticks']:.1f} ticks")

        # Save concat
        concat_path = output_dir / "concat_oot_predictions.npz"
        np.savez_compressed(str(concat_path),
            pnl_predictions=all_preds,
            labels_10s=all_labels,
            concat_ic_10s=np.array(concat_ic),
        )

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                "concat_ic_10s": concat_ic,
                "concat_n_samples": len(all_preds),
            })
            for tier, m in concat_conviction.items():
                mlflow.log_metrics({
                    f"concat_{tier}_ic": m["ic"],
                    f"concat_{tier}_win_rate": m["win_rate"],
                    f"concat_{tier}_sortino": m["sortino"],
                    f"concat_{tier}_pnl_ticks": m["total_pnl_ticks"],
                })
            mlflow.log_artifact(str(concat_path), artifact_path="concat")

    # Summary
    if fold_results:
        avg_ic = np.mean([r["ic_10s"] for r in fold_results])
        avg_sortino = np.mean([r["sortino"] for r in fold_results if np.isfinite(r["sortino"])])
        total_trades = sum(r["n_trades"] for r in fold_results)
        total_pnl = sum(r["total_pnl_ticks"] for r in fold_results)
        logger.info(f"\n  Avg fold IC (10s): {avg_ic:.4f}")
        logger.info(f"  Avg fold Sortino: {avg_sortino:.3f}")
        logger.info(f"  Total trades: {total_trades} | Total PnL: {total_pnl:.1f} ticks "
                    f"(${total_pnl * TICK_VAL:.2f})")

    return fold_results


# ============================================================
# Training — RL Mode
# ============================================================

def train_rl(args, all_folds: List[Dict], output_dir: Path, device: torch.device):
    """Walk-forward RL (PPO) training with sliding window."""

    use_embeddings = args.mode == "rl_embeddings"
    state_dim = STACKER_EMB_STATE_DIM if use_embeddings else STACKER_STATE_DIM
    n_folds = len(all_folds)
    reward_fn = args.reward_fn

    concat_pnl_all = []
    fold_results = []

    for fold_idx in range(n_folds):
        fold = all_folds[fold_idx]
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx:02d} (original fold {fold['fold_idx']:02d}) | "
                    f"{fold['n_samples']} samples")
        logger.info(f"{'='*60}")

        if fold_idx == 0:
            logger.info("  Fold 0: skipping (no training data)")
            continue

        # Sliding window training folds
        train_start = max(0, fold_idx - 60)
        train_fold_data = all_folds[train_start:fold_idx]
        oot_fold_data = fold

        logger.info(f"  Train folds: {train_start}..{fold_idx-1}, OOT: fold {fold_idx}")

        # Build training environments
        train_envs = []
        for tf in train_fold_data:
            ctx = compute_context_features(
                tf["cnn_preds"], tf["patchtst_preds"],
                tf["vol_preds"], tf["n_samples"])
            env = StackerTradingEnvironment(
                cnn_preds=tf["cnn_preds"],
                patchtst_preds=tf["patchtst_preds"],
                vol_preds=tf["vol_preds"],
                labels=tf["labels"],
                context_features=ctx,
                cnn_embeds=tf.get("cnn_embeds"),
                patchtst_embeds=tf.get("patchtst_embeds"),
                use_embeddings=use_embeddings,
            )
            train_envs.append(env)

        # OOT environment
        oot_ctx = compute_context_features(
            oot_fold_data["cnn_preds"], oot_fold_data["patchtst_preds"],
            oot_fold_data["vol_preds"], oot_fold_data["n_samples"])
        oot_env = StackerTradingEnvironment(
            cnn_preds=oot_fold_data["cnn_preds"],
            patchtst_preds=oot_fold_data["patchtst_preds"],
            vol_preds=oot_fold_data["vol_preds"],
            labels=oot_fold_data["labels"],
            context_features=oot_ctx,
            cnn_embeds=oot_fold_data.get("cnn_embeds"),
            patchtst_embeds=oot_fold_data.get("patchtst_embeds"),
            use_embeddings=use_embeddings,
        )

        # Train PPO
        n_epochs = int(os.environ.get("STACKER_RL_EPOCHS", 50))
        trainer = StackerPPOTrainer(
            state_dim=state_dim,
            device=str(device),
            lr=3e-4,
            reward_fn=reward_fn,
        )
        best_train_pnl = trainer.train_fold(train_envs, n_epochs=n_epochs)

        # OOT evaluation
        oot_results = trainer.evaluate([oot_env])
        r = oot_results[0]
        n_trades = r["trades"]
        total_pnl = r["pnl"]
        max_dd = r["max_dd"]

        # Sortino from trade PnL
        trade_pnls = np.array([t["pnl_ticks"] for t in oot_env.trades]) if oot_env.trades else np.array([0.0])
        win_rate = np.mean(trade_pnls > 0) if len(trade_pnls) > 0 else 0.0
        sortino = compute_sortino(trade_pnls)

        logger.info(f"  OOT Results:")
        logger.info(f"    P&L: ${total_pnl:,.2f} | Trades: {n_trades} | Max DD: ${max_dd:,.2f}")
        logger.info(f"    Win rate: {win_rate:.3f} | Sortino: {sortino:.3f}")
        logger.info(f"    Best train P&L: ${best_train_pnl:,.2f}")

        # Save model
        ckpt_path = output_dir / f"fold_{fold_idx:02d}_rl_model.pt"
        torch.save(trainer.policy.state_dict(), str(ckpt_path))

        # Save trade log
        trades_path = output_dir / f"fold_{fold_idx:02d}_trades.json"
        with open(str(trades_path), "w") as f:
            json.dump(oot_env.trades, f, indent=2, default=str)

        concat_pnl_all.extend(trade_pnls.tolist())

        fold_results.append({
            "fold": fold_idx,
            "pnl_dollars": total_pnl,
            "n_trades": n_trades,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "sortino": sortino,
        })

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                f"pnl_dollars_fold{fold_idx:02d}": total_pnl,
                f"n_trades_fold{fold_idx:02d}": n_trades,
                f"max_dd_fold{fold_idx:02d}": max_dd,
                f"win_rate_fold{fold_idx:02d}": win_rate,
                f"sortino_fold{fold_idx:02d}": sortino,
            }, step=fold_idx)
            mlflow.log_artifact(str(ckpt_path), artifact_path=f"fold_{fold_idx:02d}")
            mlflow.log_artifact(str(trades_path), artifact_path=f"fold_{fold_idx:02d}")

        del trainer, train_envs, oot_env
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RL TRAINING SUMMARY")
    logger.info(f"{'='*60}")

    if fold_results:
        total_pnl_all = sum(r["pnl_dollars"] for r in fold_results)
        avg_sortino = np.mean([r["sortino"] for r in fold_results if np.isfinite(r["sortino"])])
        total_trades = sum(r["n_trades"] for r in fold_results)
        avg_win = np.mean([r["win_rate"] for r in fold_results])

        logger.info(f"  Total P&L: ${total_pnl_all:,.2f}")
        logger.info(f"  Total trades: {total_trades}")
        logger.info(f"  Avg Sortino: {avg_sortino:.3f}")
        logger.info(f"  Avg win rate: {avg_win:.3f}")

        if concat_pnl_all:
            concat_sortino = compute_sortino(np.array(concat_pnl_all))
            logger.info(f"  Concat Sortino: {concat_sortino:.3f}")

            if MLFLOW_AVAILABLE:
                mlflow.log_metrics({
                    "total_pnl_dollars": total_pnl_all,
                    "total_trades": total_trades,
                    "avg_sortino": avg_sortino,
                    "avg_win_rate": avg_win,
                    "concat_sortino": concat_sortino,
                })

    return fold_results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Execution Stacker Training")
    parser.add_argument("--mode", required=True,
                        choices=["mlp_signals", "mlp_embeddings", "rl_signals", "rl_embeddings"],
                        help="Training mode")
    parser.add_argument("--n-folds", type=int, default=9,
                        help="Number of folds to use (default: 9)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output/exec_stacker_{mode}_{timestamp})")
    parser.add_argument("--reward-fn", type=str, default="drawdown_penalized",
                        choices=["raw_pnl", "drawdown_penalized", "time_efficient", "calmar"],
                        help="Reward function for RL modes")
    parser.add_argument("--cnn-mamba-dir", type=str, default=None,
                        help="Override CNN Mamba v2 predictions directory")
    parser.add_argument("--patchtst-dir", type=str, default=None,
                        help="Override PatchTST predictions directory")
    parser.add_argument("--vol-lgbm-dir", type=str, default=None,
                        help="Override Vol LGBM v3 predictions directory")
    args = parser.parse_args()

    # Override data dirs if specified
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
        output_dir = DEFAULT_OUTPUT_DIR / f"exec_stacker_{args.mode}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    logger.info(f"{'='*60}")
    logger.info(f"Execution Stacker — {args.mode}")
    logger.info(f"{'='*60}")
    logger.info(f"Device: {device} ({gpu_name})")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Data sources:")
    logger.info(f"  CNN Mamba v2:  {CNN_MAMBA_DIR}")
    logger.info(f"  PatchTST:      {PATCHTST_DIR}")
    logger.info(f"  Vol LGBM v3:   {VOL_LGBM_DIR}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"N folds: {args.n_folds}")
    if "rl" in args.mode:
        logger.info(f"Reward fn: {args.reward_fn}")

    use_embeddings = "embeddings" in args.mode

    # Load all folds
    logger.info(f"\nLoading fold data...")
    all_folds = []
    for fold_idx in range(args.n_folds + 1):  # +1 because we skip fold 0 for training
        data = load_fold_data(fold_idx, use_embeddings=use_embeddings)
        if data is not None:
            all_folds.append(data)

    if len(all_folds) < 2:
        logger.error(f"Need at least 2 folds, found {len(all_folds)}. Check data paths.")
        sys.exit(1)

    logger.info(f"Loaded {len(all_folds)} folds with aligned predictions")

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"exec_stacker_{args.mode}_{ts}"
        )
        mlflow.log_params({
            "model": "exec_stacker",
            "mode": args.mode,
            "n_folds": len(all_folds),
            "node": socket.gethostname(),
            "gpu": gpu_name,
            "output_dir": str(output_dir),
            "cnn_mamba_dir": str(CNN_MAMBA_DIR),
            "patchtst_dir": str(PATCHTST_DIR),
            "vol_lgbm_dir": str(VOL_LGBM_DIR),
            "walk_forward": "sliding_60d",
            "mixed_precision": "fp16" if device.type == "cuda" else "none",
            "num_workers": 0,
        })
        if "rl" in args.mode:
            mlflow.log_params({"reward_fn": args.reward_fn})

    try:
        if args.mode.startswith("mlp"):
            results = train_mlp(args, all_folds, output_dir, device)
        else:
            results = train_rl(args, all_folds, output_dir, device)

        # Save results summary
        summary_path = output_dir / "results_summary.json"
        with open(str(summary_path), "w") as f:
            json.dump({
                "mode": args.mode,
                "n_folds": len(all_folds),
                "results": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2, default=str)

        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_artifact(str(summary_path))

        logger.info(f"\nResults saved to {output_dir}")
        logger.info("DONE.")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
