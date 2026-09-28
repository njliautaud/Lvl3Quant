#!/usr/bin/env python3
"""
Smart Execution System v2 — GPU-Trained Adaptive Execution Components

V2 improvements over v1:
  1. Log-space TP/SL predictions (fixes SL collapse at ~1.0 tick)
  2. Focal loss for entry gate with class weighting (fixes inconsistent coverage)
  3. Staged training: 5 epochs TP/SL only, then 15 epochs joint (fixes gradient starvation)
  4. Hidden dim 192 (up from 128)
  5. Model agreement + conviction gradient features
  6. Separate LR groups (TP/SL heads get 2x base LR)
  7. Temperature scaling + calibration loss for gate confidence
  8. Batch size 1024, 20 epochs default

Architecture:
  Component 1: Adaptive TP/SL — predicts log(TP), log(SL), exp() + 0.5 tick floor
  Component 2: Entry Gate — focal loss with positive class weighting
  Component 3: Exit Trigger — hold time + risk/reward regression

Walk-forward sliding window (HC #0). MLflow logging. Mixed precision.
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
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import scipy.stats

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Logging (force=True to override MLflow's logging config)
# ============================================================
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
TICK_VAL = 12.50       # USD per tick (NQ)
COMMISSION = 0.376     # ticks one-way
SPREAD = 0.5           # ticks one-way
ROUND_TRIP_COST = 2 * (COMMISSION + SPREAD)  # 1.752 ticks

ET_OFFSET_HOURS = -5   # EST (Feb-Mar 2026 data)

# Session boundaries in ET fractional hours
SESSION_BOUNDS = [
    ("overnight",   0.0,  2.0),
    ("pre_market",  2.0,  9.5),
    ("rth_open",    9.5,  10.5),
    ("rth_core",    10.5, 15.0),
    ("rth_close",   15.0, 16.0),
    ("post_market", 16.0, 17.0),
    ("maintenance", 17.0, 17.75),  # CME daily maintenance
    ("evening",     17.75, 24.0),
]

SESSION_TO_IDX = {s[0]: i for i, s in enumerate(SESSION_BOUNDS)}

# Gap windows where we should NOT trade
GAP_WINDOWS_ET = [
    (16.95, 17.75),  # CME maintenance ~16:57-17:45 ET
]

# Log-space TP/SL constants
LOG_TPSL_MIN_FLOOR = 0.5   # minimum TP/SL in ticks after exp()
LOG_TPSL_CLIP = 6.0        # clip log targets to prevent exp() overflow


def ts_ns_to_et_hour(ts_ns: int) -> float:
    """Convert nanosecond timestamp to fractional hour in ET."""
    from datetime import datetime, timezone
    utc_sec = ts_ns / 1e9
    et_sec = utc_sec + ET_OFFSET_HOURS * 3600
    dt = datetime.fromtimestamp(et_sec, tz=timezone.utc)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def is_in_gap(et_hour: float) -> bool:
    """Check if time falls in a known gap window."""
    for start, end in GAP_WINDOWS_ET:
        if start <= et_hour < end:
            return True
    return False


def get_session_idx(et_hour: float) -> int:
    """Get session index from ET hour."""
    for name, start, end in SESSION_BOUNDS:
        if start <= et_hour < end:
            return SESSION_TO_IDX[name]
    return SESSION_TO_IDX["overnight"]


# ============================================================
# Feature Engineering
# ============================================================

def build_exec_features(
    cnn_preds: np.ndarray,      # (N, 3) predictions for 1s/5s/10s
    ptst_preds: np.ndarray,     # (N, 3) predictions for 1s/5s/10s
    vol_preds: np.ndarray,      # (N, 3) vol predictions
    timestamps: np.ndarray,     # (N,) int64 nanoseconds
    events: np.ndarray,         # (M, 6) raw MBO events for the day
    anchor_idxs: np.ndarray,    # (N,) indices into events array
    zscore_window: int = 3000,
) -> np.ndarray:
    """
    Build rich feature vectors for the execution system.

    V2 additions:
    - Model agreement feature: cnn_direction * ptst_direction (explicit +1/-1)
    - Conviction gradient: abs(cnn_pred_10s) * abs(ptst_pred_10s)

    Returns: (N, F) float32 feature matrix with F features per prediction point.
    """
    N = len(cnn_preds)
    features = []

    # 1. CNN Mamba v2 prediction z-scores (3 horizons)
    for h in range(3):
        z = _rolling_zscore(cnn_preds[:, h], zscore_window)
        features.append(z)

    # 2. PatchTST prediction z-scores (3 horizons)
    for h in range(3):
        z = _rolling_zscore(ptst_preds[:, h], zscore_window)
        features.append(z)

    # 3. Vol predictions (3 horizons, already meaningful scale)
    for h in range(3):
        features.append(vol_preds[:, h])

    # 4. Model agreement (CNN x PatchTST same sign per horizon)
    for h in range(3):
        agreement = np.sign(cnn_preds[:, h]) * np.sign(ptst_preds[:, h])
        features.append(agreement)

    # 5. Conviction strength = abs(cnn_z) * agreement
    for h in range(3):
        cnn_z = _rolling_zscore(cnn_preds[:, h], zscore_window)
        ptst_z = _rolling_zscore(ptst_preds[:, h], zscore_window)
        conviction = np.abs(cnn_z) * np.sign(cnn_z) * np.sign(ptst_z)
        features.append(conviction)

    # 6. [V2 NEW] Model agreement explicit feature: cnn_direction * ptst_direction
    #    For 10s horizon specifically — the primary trading horizon
    model_agreement_10s = np.sign(cnn_preds[:, 2]) * np.sign(ptst_preds[:, 2])
    features.append(model_agreement_10s)

    # 7. [V2 NEW] Conviction gradient: abs(cnn_pred_10s) * abs(ptst_pred_10s)
    conviction_gradient = np.abs(cnn_preds[:, 2]) * np.abs(ptst_preds[:, 2])
    features.append(conviction_gradient)

    # 8. Time-of-day features (cyclical encoding)
    et_hours = np.array([ts_ns_to_et_hour(t) for t in timestamps])
    tod_sin = np.sin(2 * np.pi * et_hours / 24.0)
    tod_cos = np.cos(2 * np.pi * et_hours / 24.0)
    features.append(tod_sin)
    features.append(tod_cos)

    # 9. Session one-hot (8 sessions)
    session_idxs = np.array([get_session_idx(h) for h in et_hours])
    for s_idx in range(len(SESSION_BOUNDS)):
        features.append((session_idxs == s_idx).astype(np.float32))

    # 10. Gap flag
    gap_flag = np.array([is_in_gap(h) for h in et_hours], dtype=np.float32)
    features.append(gap_flag)

    # 11. Microstructure features from MBO events
    spread_at_anchor = np.zeros(N, dtype=np.float32)
    volume_at_anchor = np.zeros(N, dtype=np.float32)
    momentum_5 = np.zeros(N, dtype=np.float32)
    momentum_20 = np.zeros(N, dtype=np.float32)
    event_rate = np.zeros(N, dtype=np.float32)

    if events is not None and len(events) > 0:
        # events[:, 3] = price_rel_ticks, events[:, 4] = qty_log, events[:, 5] = spread_ticks
        for i, aidx in enumerate(anchor_idxs):
            aidx = int(aidx)
            if aidx < len(events):
                spread_at_anchor[i] = events[aidx, 5] if events.shape[1] > 5 else 1.0

                # Volume in recent window
                start = max(0, aidx - 100)
                volume_at_anchor[i] = np.exp(events[start:aidx+1, 4]).sum() if events.shape[1] > 4 else 0

                # Price momentum (short and medium term)
                if aidx >= 5:
                    momentum_5[i] = events[aidx, 3] - events[aidx-5, 3]
                if aidx >= 20:
                    momentum_20[i] = events[aidx, 3] - events[aidx-20, 3]

                # Event arrival rate (events per second in last 100 events)
                if aidx >= 100 and events.shape[1] > 0:
                    dt_sum = np.exp(events[aidx-99:aidx+1, 0]).sum()  # time_delta_log
                    event_rate[i] = 100.0 / max(dt_sum, 1e-6)

    features.extend([spread_at_anchor, volume_at_anchor, momentum_5, momentum_20, event_rate])

    # 12. Volatility regime (rolling std of cnn predictions)
    vol_regime = np.zeros(N, dtype=np.float32)
    window = min(500, N)
    for i in range(window, N):
        vol_regime[i] = np.std(cnn_preds[i-window:i, 2])  # 10s horizon
    features.append(vol_regime)

    # Stack all features
    feature_matrix = np.column_stack(features).astype(np.float32)

    # Replace NaN/Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

    return feature_matrix


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling z-score using cumulative sums."""
    n = len(arr)
    z = np.zeros(n, dtype=np.float32)
    cumsum = np.cumsum(arr)
    cumsum2 = np.cumsum(arr ** 2)

    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < 30:
            continue
        s = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
        s2 = cumsum2[i] - (cumsum2[start - 1] if start > 0 else 0)
        mean = s / count
        var = s2 / count - mean ** 2
        std = np.sqrt(max(var, 1e-10))
        z[i] = (arr[i] - mean) / std

    return z


# ============================================================
# Training Targets from MFE/MAE Data
# ============================================================

def build_targets(
    labels: np.ndarray,         # (N, 3) true price changes 1s/5s/10s
    mfe_ticks: np.ndarray,      # (N,) max favorable excursion
    mae_ticks: np.ndarray,      # (N,) max adverse excursion
    time_to_mfe: np.ndarray,    # (N,) seconds to MFE
    cnn_preds: np.ndarray,      # (N, 3) for direction
) -> Dict[str, np.ndarray]:
    """
    Build training targets for all execution components.

    V2: TP/SL targets are in LOG SPACE — log(ticks) clipped to [-2, LOG_TPSL_CLIP].
    """
    N = len(labels)
    direction = np.sign(cnn_preds[:, 2])  # use 10s prediction for direction

    # Target 1: Optimal TP in log-space
    optimal_tp_raw = np.maximum(mfe_ticks, LOG_TPSL_MIN_FLOOR)
    optimal_tp_log = np.clip(np.log(optimal_tp_raw), -2.0, LOG_TPSL_CLIP)

    # Target 2: Optimal SL in log-space
    optimal_sl_raw = np.maximum(mae_ticks, LOG_TPSL_MIN_FLOOR)
    optimal_sl_log = np.clip(np.log(optimal_sl_raw), -2.0, LOG_TPSL_CLIP)

    # Target 3: Entry quality (was this a profitable trade after costs?)
    realized_pnl = labels[:, 2] * direction  # directional PnL in ticks
    profitable = (realized_pnl > ROUND_TRIP_COST).astype(np.float32)

    # Target 4: Trade magnitude (for weighting)
    trade_magnitude = np.abs(realized_pnl)

    # Target 5: Optimal hold time (time to MFE = optimal exit time)
    optimal_hold = time_to_mfe.copy()

    # Target 6: Risk-reward ratio
    rr_ratio = np.where(mae_ticks > 0.1, mfe_ticks / mae_ticks, 0.0)

    return {
        "optimal_tp_log": optimal_tp_log.astype(np.float32),
        "optimal_sl_log": optimal_sl_log.astype(np.float32),
        "optimal_tp": optimal_tp_raw.astype(np.float32),   # keep raw for eval metrics
        "optimal_sl": optimal_sl_raw.astype(np.float32),   # keep raw for eval metrics
        "entry_profitable": profitable,
        "realized_pnl": realized_pnl.astype(np.float32),
        "trade_magnitude": trade_magnitude.astype(np.float32),
        "optimal_hold_s": optimal_hold.astype(np.float32),
        "risk_reward": rr_ratio.astype(np.float32),
    }


# ============================================================
# Models
# ============================================================

class AdaptiveTPSL_v2(nn.Module):
    """
    V2: Predicts log(TP) and log(SL), then exp() + floor.

    Fixes v1 SL collapse where Softplus + 1.0 made SL always ~1.0 tick.
    By predicting in log-space, the model has a much easier optimization
    landscape and can produce varied TP/SL widths.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Predict in log-space (unconstrained output)
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.encoder(x)
        log_tp = self.tp_head(h).squeeze(-1)  # unconstrained log-space
        log_sl = self.sl_head(h).squeeze(-1)  # unconstrained log-space
        return log_tp, log_sl

    def predict_ticks(self, x):
        """Get TP/SL in tick-space (for inference)."""
        log_tp, log_sl = self.forward(x)
        tp = torch.exp(log_tp.clamp(-2.0, LOG_TPSL_CLIP)) + LOG_TPSL_MIN_FLOOR
        sl = torch.exp(log_sl.clamp(-2.0, LOG_TPSL_CLIP)) + LOG_TPSL_MIN_FLOOR
        return tp, sl


class EntryGate_v2(nn.Module):
    """
    V2: Entry gate with learnable temperature for calibrated confidence.

    Predicts whether to enter a trade. Temperature scaling allows the
    model to produce confident predictions (near 0 or 1) which fixes
    the zero high-confidence trades problem from v1.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Learnable temperature (initialized to 1.0)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logit = self.net(x).squeeze(-1)
        temperature = torch.exp(self.log_temperature).clamp(0.1, 10.0)
        return logit / temperature  # temperature-scaled logit


class ExitTrigger_v2(nn.Module):
    """
    Predicts optimal hold time and exit urgency.
    Output: (predicted_hold_time_s, risk_reward_ratio)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.hold_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # hold time positive
        )
        self.rr_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # risk/reward positive
        )

    def forward(self, x):
        h = self.encoder(x)
        hold = self.hold_head(h).squeeze(-1)
        rr = self.rr_head(h).squeeze(-1)
        return hold, rr


class SmartExecSystem_v2(nn.Module):
    """
    Complete Smart Execution System v2.

    Multi-task model with shared trunk and task-specific heads.
    Supports staged training: freeze/unfreeze gate and exit heads.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Task-specific heads
        self.tpsl = AdaptiveTPSL_v2(hidden_dim, hidden_dim)
        self.gate = EntryGate_v2(hidden_dim, hidden_dim // 2)
        self.exit = ExitTrigger_v2(hidden_dim, hidden_dim // 2)

    def forward(self, x):
        shared_repr = self.shared(x)

        log_tp, log_sl = self.tpsl(shared_repr)
        gate_logit = self.gate(shared_repr)
        hold_time, rr = self.exit(shared_repr)

        return {
            "log_tp": log_tp,
            "log_sl": log_sl,
            "gate_logit": gate_logit,
            "hold_time": hold_time,
            "risk_reward": rr,
        }

    def freeze_gate_exit(self):
        """Freeze gate and exit heads for staged training (phase 1: TP/SL only)."""
        for param in self.gate.parameters():
            param.requires_grad = False
        for param in self.exit.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for joint training (phase 2)."""
        for param in self.parameters():
            param.requires_grad = True

    def get_param_groups(self, base_lr: float):
        """
        Return parameter groups with different LRs.
        TP/SL heads get 2x base LR since they're harder to train.
        """
        tpsl_params = list(self.tpsl.parameters())
        gate_params = list(self.gate.parameters())
        exit_params = list(self.exit.parameters())
        shared_params = list(self.shared.parameters())

        return [
            {"params": shared_params, "lr": base_lr},
            {"params": tpsl_params, "lr": base_lr * 2.0},  # 2x for TP/SL
            {"params": gate_params, "lr": base_lr},
            {"params": exit_params, "lr": base_lr},
        ]


# ============================================================
# Dataset
# ============================================================

class SmartExecDataset(Dataset):
    """Holds pre-computed features and targets for one fold."""

    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray]):
        self.features = torch.from_numpy(features)
        self.targets = {k: torch.from_numpy(v) for k, v in targets.items()}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], {k: v[idx] for k, v in self.targets.items()}


# ============================================================
# Loss Functions
# ============================================================

class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification."""

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Compute BCE with pos_weight
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none',
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
        )
        # Focal modulation
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * bce

        if sample_weights is not None:
            loss = loss * sample_weights

        return loss.mean()


class SmartExecLoss_v2(nn.Module):
    """
    V2 multi-task loss.

    Changes from v1:
    - TP/SL loss in log-space (Huber on log predictions vs log targets)
    - Focal loss for gate with class weighting
    - Calibration loss term to encourage confident gate predictions
    - Supports staged training (can zero out gate/exit loss)
    """
    def __init__(
        self,
        gate_weight: float = 1.0,
        tpsl_weight: float = 1.0,
        exit_weight: float = 0.5,
        calibration_weight: float = 0.1,
        focal_gamma: float = 2.0,
        pos_class_weight: float = 1.0,
        stage: str = "joint",  # "tpsl_only" or "joint"
    ):
        super().__init__()
        self.gate_weight = gate_weight
        self.tpsl_weight = tpsl_weight
        self.exit_weight = exit_weight
        self.calibration_weight = calibration_weight
        self.stage = stage
        self.huber = nn.HuberLoss(delta=2.0)  # tighter delta for log-space
        self.focal = FocalLoss(gamma=focal_gamma, pos_weight=pos_class_weight)

    def forward(self, outputs, targets):
        # TP/SL regression loss in LOG SPACE
        tp_loss = self.huber(outputs["log_tp"], targets["optimal_tp_log"])
        sl_loss = self.huber(outputs["log_sl"], targets["optimal_sl_log"])
        tpsl_loss = tp_loss + sl_loss

        total = self.tpsl_weight * tpsl_loss

        loss_dict = {
            "tp_loss": tp_loss.item(),
            "sl_loss": sl_loss.item(),
        }

        if self.stage == "joint":
            # Entry gate focal loss with magnitude weighting
            mag_weights = 1.0 + targets["trade_magnitude"] / (targets["trade_magnitude"].mean() + 1e-6)
            gate_loss = self.focal(
                outputs["gate_logit"],
                targets["entry_profitable"],
                sample_weights=mag_weights,
            )

            # Calibration loss: encourage gate predictions away from 0.5
            # ECE-inspired: penalize when confident predictions are wrong
            gate_probs = torch.sigmoid(outputs["gate_logit"])
            confidence = torch.abs(gate_probs - 0.5)
            # Reward high confidence when correct, penalize when wrong
            correct = (gate_probs > 0.5).float() == targets["entry_profitable"]
            calibration_loss = (confidence * (~correct).float()).mean() - \
                               0.5 * (confidence * correct.float()).mean()

            # Exit timing loss
            hold_loss = self.huber(outputs["hold_time"], targets["optimal_hold_s"])
            rr_loss = self.huber(outputs["risk_reward"], targets["risk_reward"])
            exit_loss = hold_loss + 0.5 * rr_loss

            total = total + \
                    self.gate_weight * gate_loss + \
                    self.exit_weight * exit_loss + \
                    self.calibration_weight * calibration_loss

            loss_dict.update({
                "gate_loss": gate_loss.item(),
                "calibration_loss": calibration_loss.item(),
                "hold_loss": hold_loss.item(),
                "rr_loss": rr_loss.item(),
            })
        else:
            # Stage 1: TP/SL only — zero gate/exit losses
            loss_dict.update({
                "gate_loss": 0.0,
                "calibration_loss": 0.0,
                "hold_loss": 0.0,
                "rr_loss": 0.0,
            })

        loss_dict["total_loss"] = total.item()

        return total, loss_dict


# ============================================================
# Data Loading
# ============================================================

def load_fold_data(
    fold_idx: int,
    cnn_dir: Path,
    ptst_dir: Path,
    vol_dir: Path,
    mfe_dir: Path,
    mbo_dir: Path,
    horizon: str = "10s",
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Load and align data from all models for a given fold.
    Returns (features, targets) or (None, None) if data unavailable.
    """
    h_idx = {"1s": 0, "5s": 1, "10s": 2}[horizon]

    # Load CNN Mamba v2 predictions
    cnn_path = cnn_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not cnn_path.exists():
        logger.warning(f"CNN predictions not found: {cnn_path}")
        return None, None
    cnn_data = np.load(cnn_path, allow_pickle=True)
    cnn_preds = cnn_data["predictions"]  # (N, 3)
    labels = cnn_data["labels"]          # (N, 3)
    oot_files = cnn_data.get("oot_files", None)

    N_cnn = len(cnn_preds)
    logger.info(f"  CNN preds: {N_cnn} samples, IC_10s={cnn_data.get('ic_10s', 'N/A')}")

    # Load MFE/MAE data for this fold
    mfe_path = mfe_dir / f"fold_{fold_idx:02d}_mfe_mae_{horizon}.npz"
    if mfe_path.exists():
        mfe_data = np.load(mfe_path, allow_pickle=True)
        mfe_ticks = mfe_data["mfe_ticks"]
        mae_ticks = mfe_data["mae_ticks"]
        time_to_mfe = mfe_data.get("time_to_mfe_s", np.ones(N_cnn) * 5.0)
        logger.info(f"  MFE/MAE: {len(mfe_ticks)} samples, mean MFE={mfe_ticks.mean():.2f}, mean MAE={mae_ticks.mean():.2f}")
    else:
        # Approximate MFE/MAE from labels
        logger.warning(f"  No MFE/MAE data, approximating from labels")
        directional = labels[:, h_idx] * np.sign(cnn_preds[:, h_idx])
        mfe_ticks = np.maximum(directional, 0)
        mae_ticks = np.abs(np.minimum(directional, 0))
        time_to_mfe = np.ones(N_cnn, dtype=np.float32) * 5.0

    # Ensure alignment
    min_n = min(N_cnn, len(mfe_ticks))
    cnn_preds = cnn_preds[:min_n]
    labels = labels[:min_n]
    mfe_ticks = mfe_ticks[:min_n]
    mae_ticks = mae_ticks[:min_n]
    time_to_mfe = time_to_mfe[:min_n]

    # Load PatchTST predictions (may have different N due to stride)
    ptst_preds = np.zeros((min_n, 3), dtype=np.float32)
    ptst_path = ptst_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if ptst_path.exists():
        ptst_data = np.load(ptst_path, allow_pickle=True)
        ptst_raw = ptst_data["predictions"]
        # Align via subsampling/interpolation if lengths differ
        if len(ptst_raw) == min_n:
            ptst_preds = ptst_raw[:min_n]
        elif len(ptst_raw) > min_n:
            indices = np.linspace(0, len(ptst_raw)-1, min_n).astype(int)
            ptst_preds = ptst_raw[indices]
        else:
            for h in range(3):
                ptst_preds[:len(ptst_raw), h] = ptst_raw[:, h]
                if len(ptst_raw) < min_n:
                    ptst_preds[len(ptst_raw):, h] = ptst_raw[-1, h]
        logger.info(f"  PatchTST preds: {len(ptst_raw)} -> aligned to {min_n}")
    else:
        logger.warning(f"  No PatchTST preds for fold {fold_idx}, using zeros")

    # Load vol predictions
    vol_preds = np.zeros((min_n, 3), dtype=np.float32)
    vol_files = sorted(vol_dir.glob("vol_v3_*_predictions.npz"))
    if vol_files and oot_files is not None:
        oot_dates = [str(f).split("_")[0][:8] if isinstance(f, str) else str(f)[:8]
                     for f in oot_files]
        for vf in vol_files:
            vdate = vf.stem.split("_")[2]
            if vdate in oot_dates:
                vdata = np.load(vf, allow_pickle=True)
                vpreds = vdata["predictions"]
                fill_n = min(len(vpreds), min_n)
                vol_preds[:fill_n] = vpreds[:fill_n, :3] if vpreds.shape[1] >= 3 else vpreds[:fill_n]
                logger.info(f"  Vol preds matched for {vdate}: {fill_n} samples")
                break

    # Generate placeholder timestamps if not available
    timestamps = np.arange(min_n, dtype=np.int64) * int(1e8)  # ~100ms spacing placeholder
    anchor_idxs = np.arange(min_n, dtype=np.int64)

    # Try to load actual MBO events for microstructure features
    events = None
    if oot_files is not None and mbo_dir.exists():
        for oot_f in oot_files:
            fname = str(oot_f) if isinstance(oot_f, str) else oot_f
            candidates = list(mbo_dir.glob(f"*{fname[:8]}*")) if len(fname) >= 8 else []
            if candidates:
                try:
                    mbo_data = np.load(candidates[0], allow_pickle=True)
                    events = mbo_data.get("events", None)
                    if "timestamps" in mbo_data:
                        ts_raw = mbo_data["timestamps"]
                        if len(ts_raw) >= min_n:
                            timestamps = ts_raw[:min_n]
                    logger.info(f"  MBO events loaded: {len(events)} events")
                except Exception as e:
                    logger.warning(f"  Failed to load MBO events: {e}")
                break

    # Build features
    logger.info(f"  Building execution features ({min_n} samples)...")
    features = build_exec_features(
        cnn_preds, ptst_preds, vol_preds,
        timestamps, events, anchor_idxs,
    )

    # Build targets
    targets = build_targets(labels, mfe_ticks, mae_ticks, time_to_mfe, cnn_preds)

    return features, targets


# ============================================================
# Training Loop
# ============================================================

def train_one_fold(
    model: SmartExecSystem_v2,
    train_features: np.ndarray,
    train_targets: Dict[str, np.ndarray],
    oot_features: np.ndarray,
    oot_targets: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int = 20,
    tpsl_only_epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 3e-4,
    focal_gamma: float = 2.0,
    pos_class_weight: float = 1.0,
    fold_idx: int = 0,
) -> Dict:
    """
    Train model on one fold with STAGED training:
      Phase 1 (epochs 1-5): TP/SL heads only, gate+exit frozen
      Phase 2 (epochs 6-20): all heads jointly

    Returns OOT evaluation metrics.
    """

    train_ds = SmartExecDataset(train_features, train_targets)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)

    scaler = GradScaler()
    best_val_loss = float('inf')
    best_state = None

    # ---- Phase 1: TP/SL only ----
    logger.info(f"  Phase 1: TP/SL only ({tpsl_only_epochs} epochs)")
    model.freeze_gate_exit()

    # Only optimize shared + tpsl params in phase 1
    phase1_params = [
        {"params": list(model.shared.parameters()), "lr": lr},
        {"params": list(model.tpsl.parameters()), "lr": lr * 2.0},
    ]
    optimizer_p1 = torch.optim.AdamW(phase1_params, weight_decay=1e-4)
    scheduler_p1 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_p1, max_lr=[lr, lr * 2.0],
        total_steps=tpsl_only_epochs * len(train_dl),
        pct_start=0.1, anneal_strategy='cos',
    )
    criterion_p1 = SmartExecLoss_v2(
        tpsl_weight=1.0, stage="tpsl_only",
    )

    for epoch in range(tpsl_only_epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_features, batch_targets in train_dl:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = {k: v.to(device, non_blocking=True) for k, v in batch_targets.items()}

            optimizer_p1.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                outputs = model(batch_features)
                loss, loss_dict = criterion_p1(outputs, batch_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_p1)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer_p1)
            scaler.update()
            scheduler_p1.step()

            epoch_losses.append(loss_dict)

        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        wall_time = time.time() - t0

        logger.info(
            f"  [P1] Epoch {epoch+1}/{tpsl_only_epochs} | "
            f"total={avg_losses['total_loss']:.4f} | "
            f"tp={avg_losses['tp_loss']:.4f} sl={avg_losses['sl_loss']:.4f} | "
            f"{wall_time:.1f}s"
        )

    # ---- Phase 2: Joint training ----
    joint_epochs = epochs - tpsl_only_epochs
    logger.info(f"  Phase 2: Joint training ({joint_epochs} epochs)")
    model.unfreeze_all()

    param_groups = model.get_param_groups(lr)
    optimizer_p2 = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    max_lrs = [lr, lr * 2.0, lr, lr]
    scheduler_p2 = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_p2, max_lr=max_lrs,
        total_steps=joint_epochs * len(train_dl),
        pct_start=0.1, anneal_strategy='cos',
    )
    criterion_p2 = SmartExecLoss_v2(
        gate_weight=1.0,
        tpsl_weight=1.0,
        exit_weight=0.5,
        calibration_weight=0.1,
        focal_gamma=focal_gamma,
        pos_class_weight=pos_class_weight,
        stage="joint",
    )

    for epoch in range(joint_epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_features, batch_targets in train_dl:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = {k: v.to(device, non_blocking=True) for k, v in batch_targets.items()}

            optimizer_p2.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                outputs = model(batch_features)
                loss, loss_dict = criterion_p2(outputs, batch_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer_p2)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer_p2)
            scaler.update()
            scheduler_p2.step()

            epoch_losses.append(loss_dict)

        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        wall_time = time.time() - t0

        global_epoch = tpsl_only_epochs + epoch + 1
        logger.info(
            f"  [P2] Epoch {global_epoch}/{epochs} | "
            f"total={avg_losses['total_loss']:.4f} | "
            f"gate={avg_losses['gate_loss']:.4f} | "
            f"tp={avg_losses['tp_loss']:.4f} sl={avg_losses['sl_loss']:.4f} | "
            f"hold={avg_losses['hold_loss']:.4f} cal={avg_losses['calibration_loss']:.4f} | "
            f"{wall_time:.1f}s"
        )

        # OOT eval every epoch in phase 2
        val_metrics = evaluate_oot(model, oot_features, oot_targets, device)
        val_loss = val_metrics.get("total_loss", float('inf'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            f"    OOT | gate_acc={val_metrics.get('gate_accuracy', 0):.3f} | "
            f"gate_prec={val_metrics.get('gate_precision', 0):.3f} | "
            f"coverage={val_metrics.get('gate_coverage', 0):.3f} | "
            f"tp_mae={val_metrics.get('tp_mae', 0):.2f} | "
            f"sl_mae={val_metrics.get('sl_mae', 0):.2f} | "
            f"sortino_imp={val_metrics.get('sortino_improvement', 0):.3f}"
        )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final OOT evaluation
    final_metrics = evaluate_oot(model, oot_features, oot_targets, device, detailed=True)
    final_metrics["best_val_loss"] = best_val_loss

    return final_metrics


def evaluate_oot(
    model: SmartExecSystem_v2,
    features: np.ndarray,
    targets: Dict[str, np.ndarray],
    device: torch.device,
    detailed: bool = False,
) -> Dict:
    """Evaluate model on OOT data. V2: converts log-space TP/SL back to ticks."""
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(features).to(device)

        with autocast(dtype=torch.float16):
            outputs = model(x)

        # Move to CPU numpy
        log_tp = outputs["log_tp"].float().cpu().numpy()
        log_sl = outputs["log_sl"].float().cpu().numpy()
        # Convert log-space to tick-space
        pred_tp = np.exp(np.clip(log_tp, -2.0, LOG_TPSL_CLIP)) + LOG_TPSL_MIN_FLOOR
        pred_sl = np.exp(np.clip(log_sl, -2.0, LOG_TPSL_CLIP)) + LOG_TPSL_MIN_FLOOR

        gate_logit = outputs["gate_logit"].float().cpu().numpy()
        gate_prob = 1.0 / (1.0 + np.exp(-np.clip(gate_logit, -20, 20)))
        pred_hold = outputs["hold_time"].float().cpu().numpy()
        pred_rr = outputs["risk_reward"].float().cpu().numpy()

    # Gate accuracy
    gate_pred = (gate_prob > 0.5).astype(float)
    gate_true = targets["entry_profitable"]
    gate_accuracy = (gate_pred == gate_true).mean()

    # Gate precision (of trades we'd TAKE, what fraction profitable?)
    take_mask = gate_prob > 0.5
    if take_mask.sum() > 0:
        gate_precision = gate_true[take_mask].mean()
        gate_coverage = take_mask.mean()
    else:
        gate_precision = 0.0
        gate_coverage = 0.0

    # TP/SL accuracy (in tick space, compare to raw targets)
    tp_mae = np.abs(pred_tp - targets["optimal_tp"]).mean()
    sl_mae = np.abs(pred_sl - targets["optimal_sl"]).mean()

    # Simulate: PnL with gate vs without gate
    pnl_all = targets["realized_pnl"]
    pnl_gated = pnl_all[take_mask] if take_mask.sum() > 0 else np.array([0.0])

    # Sortino improvement
    def _sortino(pnl):
        if len(pnl) < 2:
            return 0.0
        downside = pnl[pnl < 0]
        downside_std = np.std(downside) if len(downside) > 1 else 1.0
        return pnl.mean() / (downside_std + 1e-8)

    sortino_baseline = _sortino(pnl_all)
    sortino_gated = _sortino(pnl_gated)

    # Compute total loss for model selection
    criterion = SmartExecLoss_v2(stage="joint")
    x_t = torch.from_numpy(features).to(device)
    t_t = {k: torch.from_numpy(v).to(device) for k, v in targets.items()}
    with torch.no_grad(), autocast(dtype=torch.float16):
        out = model(x_t)
        total_loss, _ = criterion(out, t_t)

    metrics = {
        "gate_accuracy": float(gate_accuracy),
        "gate_precision": float(gate_precision),
        "gate_coverage": float(gate_coverage),
        "tp_mae": float(tp_mae),
        "sl_mae": float(sl_mae),
        "sortino_baseline": float(sortino_baseline),
        "sortino_gated": float(sortino_gated),
        "sortino_improvement": float(sortino_gated - sortino_baseline),
        "pnl_baseline_ticks": float(pnl_all.sum()),
        "pnl_gated_ticks": float(pnl_gated.sum()),
        "trades_baseline": len(pnl_all),
        "trades_gated": int(take_mask.sum()),
        "total_loss": float(total_loss.item()),
    }

    if detailed:
        # Confidence-conditional analysis (HC #13)
        abs_gate = np.abs(gate_prob - 0.5)  # distance from decision boundary
        for pct_name, pct_thresh in [("top50", 0.5), ("top25", 0.75),
                                      ("top10", 0.9), ("top5", 0.95), ("top1", 0.99)]:
            thresh = np.percentile(abs_gate, pct_thresh * 100)
            mask = abs_gate >= thresh
            if mask.sum() > 0:
                gated_pnl = pnl_all[mask & (gate_prob > 0.5)]
                metrics[f"pnl_{pct_name}_ticks"] = float(gated_pnl.sum()) if len(gated_pnl) > 0 else 0.0
                metrics[f"sortino_{pct_name}"] = float(_sortino(gated_pnl)) if len(gated_pnl) > 0 else 0.0
                metrics[f"trades_{pct_name}"] = int((mask & (gate_prob > 0.5)).sum())
                metrics[f"winrate_{pct_name}"] = float((gated_pnl > ROUND_TRIP_COST).mean()) if len(gated_pnl) > 0 else 0.0

        # Adaptive TP/SL analysis
        metrics["pred_tp_mean"] = float(pred_tp.mean())
        metrics["pred_tp_std"] = float(pred_tp.std())
        metrics["pred_sl_mean"] = float(pred_sl.mean())
        metrics["pred_sl_std"] = float(pred_sl.std())
        metrics["actual_mfe_mean"] = float(targets["optimal_tp"].mean())
        metrics["actual_mae_mean"] = float(targets["optimal_sl"].mean())

        # V2: Gate confidence distribution
        metrics["gate_prob_mean"] = float(gate_prob.mean())
        metrics["gate_prob_std"] = float(gate_prob.std())
        metrics["gate_high_conf_count"] = int((abs_gate > 0.4).sum())  # near 0 or 1
        metrics["gate_temperature"] = float(
            model.gate.log_temperature.exp().item()
        ) if hasattr(model, 'gate') and hasattr(model.gate, 'log_temperature') else 1.0

    model.train()
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Smart Execution System v2 Training")
    parser.add_argument("--cnn-dir", type=str, default=None,
                        help="CNN Mamba v2 predictions directory")
    parser.add_argument("--ptst-dir", type=str, default=None,
                        help="PatchTST predictions directory")
    parser.add_argument("--vol-dir", type=str, default=None,
                        help="Vol LGBM v3 predictions directory")
    parser.add_argument("--mbo-dir", type=str, default=None,
                        help="MBO events directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total epochs (phase1 + phase2)")
    parser.add_argument("--tpsl-only-epochs", type=int, default=5,
                        help="Phase 1 epochs (TP/SL only, gate/exit frozen)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size (1024 fits easily in 24GB)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Base learning rate (TP/SL heads get 2x)")
    parser.add_argument("--hidden-dim", type=int, default=192,
                        help="Hidden dimension (up from 128 in v1)")
    parser.add_argument("--max-folds", type=int, default=9,
                        help="Maximum number of folds to use")
    parser.add_argument("--horizon", type=str, default="10s",
                        choices=["1s", "5s", "10s"],
                        help="Target horizon")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma for gate")
    parser.add_argument("--gate-weight", type=float, default=1.0,
                        help="Gate loss weight (v1 was 2.0, now 1.0)")
    parser.add_argument("--version", type=str, default="v2",
                        help="Version tag for MLflow")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader num_workers (Neptune 32GB RAM: max 2)")
    args = parser.parse_args()

    # Detect paths
    hostname = socket.gethostname()
    if "neptune" in hostname.lower() or "nick" in str(Path.home()):
        root = Path("/home/nick/Lvl3Quant")
    else:
        root = Path("/home/jupiter/Lvl3Quant")

    cnn_dir = Path(args.cnn_dir) if args.cnn_dir else root / "output" / "cnn_mamba_v2_smart_v3_mar"
    ptst_dir = Path(args.ptst_dir) if args.ptst_dir else root / "output" / "patchtst_smart_v3_mar"
    vol_dir = Path(args.vol_dir) if args.vol_dir else root / "output" / "vol_lgbm_v3"
    mbo_dir = Path(args.mbo_dir) if args.mbo_dir else root / "data" / "processed" / "mbo_events"
    mfe_dir = cnn_dir / "mfe_mae_analysis"

    # Output dir with timestamp
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = root / "output" / f"smart_exec_v2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("Smart Execution System v2 — Training")
    logger.info("=" * 60)
    logger.info(f"Device:      {device}")
    if device.type == "cuda":
        logger.info(f"GPU:         {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"CNN dir:     {cnn_dir}")
    logger.info(f"PatchTST:    {ptst_dir}")
    logger.info(f"Vol dir:     {vol_dir}")
    logger.info(f"MFE/MAE:     {mfe_dir}")
    logger.info(f"MBO dir:     {mbo_dir}")
    logger.info(f"Output:      {output_dir}")
    logger.info(f"Horizon:     {args.horizon}")
    logger.info(f"Epochs:      {args.epochs} (P1={args.tpsl_only_epochs} TP/SL only + P2={args.epochs - args.tpsl_only_epochs} joint)")
    logger.info(f"Hidden dim:  {args.hidden_dim}")
    logger.info(f"Batch size:  {args.batch_size}")
    logger.info(f"Base LR:     {args.lr} (TP/SL heads: {args.lr * 2})")
    logger.info(f"Gate weight: {args.gate_weight}")
    logger.info(f"Focal gamma: {args.focal_gamma}")
    logger.info(f"Version:     {args.version}")
    logger.info("=" * 60)

    # Discover available folds
    cnn_folds = sorted(cnn_dir.glob("fold_*_oot_predictions.npz"))
    n_folds = min(len(cnn_folds), args.max_folds)
    logger.info(f"Found {len(cnn_folds)} CNN folds, using {n_folds}")

    if n_folds < 2:
        logger.error("Need at least 2 folds (1 train + 1 OOT). Exiting.")
        return

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("file://" + str(root / "mlruns"))
            mlflow.set_experiment("SmartExecSystem_v2")
            mlflow_run = mlflow.start_run(
                run_name=f"SmartExec_v2_{time.strftime('%Y%m%d_%H%M')}"
            )
            mlflow.log_params({
                "model": "SmartExecSystem_v2",
                "version": args.version,
                "hidden_dim": args.hidden_dim,
                "epochs": args.epochs,
                "tpsl_only_epochs": args.tpsl_only_epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "lr_tpsl": args.lr * 2.0,
                "horizon": args.horizon,
                "n_folds": n_folds,
                "node": hostname,
                "device": str(device),
                "focal_gamma": args.focal_gamma,
                "gate_weight": args.gate_weight,
                "tpsl_log_space": True,
                "staged_training": True,
                "components": "AdaptiveTPSL_v2+EntryGate_v2+ExitTrigger_v2",
                "v2_fixes": "log_tpsl,focal_loss,staged_training,temperature_scaling,calibration_loss",
            })
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}")

    # Walk-forward sliding window
    all_fold_metrics = []
    concat_gate_preds = []
    concat_gate_true = []
    concat_pnl = []

    for oot_fold in range(1, n_folds):
        train_folds = list(range(max(0, oot_fold - 5), oot_fold))  # up to 5 recent folds

        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {oot_fold} | Train folds: {train_folds} | OOT fold: {oot_fold}")
        logger.info(f"{'='*60}")

        # Load training data from multiple folds
        train_features_list = []
        train_targets_list = {
            "optimal_tp_log": [], "optimal_sl_log": [],
            "optimal_tp": [], "optimal_sl": [],
            "entry_profitable": [],
            "realized_pnl": [], "trade_magnitude": [], "optimal_hold_s": [],
            "risk_reward": [],
        }

        for tf in train_folds:
            feat, targ = load_fold_data(
                tf, cnn_dir, ptst_dir, vol_dir, mfe_dir, mbo_dir, args.horizon
            )
            if feat is not None:
                train_features_list.append(feat)
                for k in train_targets_list:
                    train_targets_list[k].append(targ[k])

        if not train_features_list:
            logger.warning(f"No training data for fold {oot_fold}, skipping")
            continue

        train_features = np.concatenate(train_features_list)
        train_targets = {k: np.concatenate(v) for k, v in train_targets_list.items()}

        # Load OOT data
        oot_features, oot_targets = load_fold_data(
            oot_fold, cnn_dir, ptst_dir, vol_dir, mfe_dir, mbo_dir, args.horizon
        )
        if oot_features is None:
            logger.warning(f"No OOT data for fold {oot_fold}, skipping")
            continue

        logger.info(f"Train: {len(train_features)} samples, OOT: {len(oot_features)} samples")
        logger.info(f"Feature dim: {train_features.shape[1]}")

        # Compute positive class weight for focal loss
        pos_ratio = train_targets["entry_profitable"].mean()
        neg_ratio = 1.0 - pos_ratio
        pos_class_weight = neg_ratio / max(pos_ratio, 1e-6)
        pos_class_weight = min(pos_class_weight, 10.0)  # cap at 10x
        logger.info(f"Positive class ratio: {pos_ratio:.3f}, weight: {pos_class_weight:.2f}")

        # Normalize features using train stats
        feat_mean = train_features.mean(axis=0)
        feat_std = train_features.std(axis=0) + 1e-8
        train_features = (train_features - feat_mean) / feat_std
        oot_features = (oot_features - feat_mean) / feat_std

        # Build model
        input_dim = train_features.shape[1]
        model = SmartExecSystem_v2(input_dim, args.hidden_dim).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model params: {n_params:,}")

        # Train
        fold_metrics = train_one_fold(
            model, train_features, train_targets,
            oot_features, oot_targets, device,
            epochs=args.epochs,
            tpsl_only_epochs=args.tpsl_only_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            focal_gamma=args.focal_gamma,
            pos_class_weight=pos_class_weight,
            fold_idx=oot_fold,
        )

        all_fold_metrics.append(fold_metrics)

        # Save model and predictions
        torch.save({
            "model_state": model.state_dict(),
            "fold_idx": oot_fold,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "metrics": fold_metrics,
            "version": args.version,
        }, output_dir / f"fold_{oot_fold:02d}_smart_exec_v2.pt")

        # Save OOT predictions as npz for downstream analysis
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(oot_features).to(device)
            with autocast(dtype=torch.float16):
                out = model(x)
            gate_prob = torch.sigmoid(out["gate_logit"]).float().cpu().numpy()
            log_tp_pred = out["log_tp"].float().cpu().numpy()
            log_sl_pred = out["log_sl"].float().cpu().numpy()
            tp_pred = np.exp(np.clip(log_tp_pred, -2.0, LOG_TPSL_CLIP)) + LOG_TPSL_MIN_FLOOR
            sl_pred = np.exp(np.clip(log_sl_pred, -2.0, LOG_TPSL_CLIP)) + LOG_TPSL_MIN_FLOOR

        np.savez_compressed(
            output_dir / f"fold_{oot_fold:02d}_oot_predictions.npz",
            gate_prob=gate_prob,
            tp_ticks=tp_pred,
            sl_ticks=sl_pred,
            hold_time=out["hold_time"].float().cpu().numpy(),
            risk_reward=out["risk_reward"].float().cpu().numpy(),
            entry_profitable=oot_targets["entry_profitable"],
            realized_pnl=oot_targets["realized_pnl"],
        )

        concat_gate_preds.append(gate_prob)
        concat_gate_true.append(oot_targets["entry_profitable"])
        concat_pnl.append(oot_targets["realized_pnl"])

        # MLflow per-fold
        if mlflow_run:
            try:
                for k, v in fold_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"fold{oot_fold:02d}_{k}", v)
            except Exception:
                pass

        # Memory cleanup
        del model, train_features, train_targets, oot_features, oot_targets
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Concat metrics (primary metric — HC #13)
    # ============================================================
    if concat_gate_preds:
        all_gate = np.concatenate(concat_gate_preds)
        all_true = np.concatenate(concat_gate_true)
        all_pnl = np.concatenate(concat_pnl)

        logger.info("\n" + "=" * 60)
        logger.info("CONCAT RESULTS — Smart Execution System v2")
        logger.info("=" * 60)

        # Overall gate performance
        gate_pred_binary = (all_gate > 0.5)
        overall_acc = (gate_pred_binary == all_true).mean()

        # Gated vs ungated PnL
        gated_mask = all_gate > 0.5
        pnl_ungated = all_pnl.sum()
        pnl_gated = all_pnl[gated_mask].sum() if gated_mask.sum() > 0 else 0

        def _sortino(pnl):
            if len(pnl) < 2: return 0.0
            d = pnl[pnl < 0]
            return pnl.mean() / (np.std(d) + 1e-8) if len(d) > 1 else pnl.mean()

        sortino_ungated = _sortino(all_pnl)
        sortino_gated = _sortino(all_pnl[gated_mask]) if gated_mask.sum() > 0 else 0

        logger.info(f"Gate accuracy:    {overall_acc:.3f}")
        logger.info(f"Gate coverage:    {gated_mask.mean():.3f} ({gated_mask.sum()}/{len(all_gate)} trades taken)")
        logger.info(f"PnL ungated:      {pnl_ungated:.1f} ticks (${pnl_ungated * TICK_VAL:.2f})")
        logger.info(f"PnL gated:        {pnl_gated:.1f} ticks (${pnl_gated * TICK_VAL:.2f})")
        logger.info(f"Sortino ungated:  {sortino_ungated:.3f}")
        logger.info(f"Sortino gated:    {sortino_gated:.3f}")

        # Gate confidence distribution
        confidence = np.abs(all_gate - 0.5)
        logger.info(f"\nGate confidence stats:")
        logger.info(f"  Mean distance from 0.5: {confidence.mean():.4f}")
        logger.info(f"  High-conf (>0.4):       {(confidence > 0.4).sum()} ({(confidence > 0.4).mean()*100:.1f}%)")
        logger.info(f"  Very high-conf (>0.45):  {(confidence > 0.45).sum()} ({(confidence > 0.45).mean()*100:.1f}%)")

        # Confidence-conditional (HC #13/#14)
        logger.info(f"\n{'Tier':<10} {'Trades':>8} {'WinRate':>8} {'PnL(t)':>10} {'Sortino':>8}")
        logger.info("-" * 50)

        for name, pct in [("All", 0.0), ("Top50%", 0.5), ("Top25%", 0.75),
                          ("Top10%", 0.9), ("Top5%", 0.95), ("Top1%", 0.99)]:
            if pct > 0:
                thresh = np.percentile(confidence, pct * 100)
                mask = (confidence >= thresh) & (all_gate > 0.5)
            else:
                mask = all_gate > 0.5

            if mask.sum() > 0:
                tier_pnl = all_pnl[mask]
                wr = (tier_pnl > ROUND_TRIP_COST).mean()
                s = _sortino(tier_pnl)
                logger.info(f"{name:<10} {mask.sum():>8d} {wr:>8.3f} {tier_pnl.sum():>10.1f} {s:>8.3f}")
            else:
                logger.info(f"{name:<10} {'0':>8} {'N/A':>8} {'0.0':>10} {'N/A':>8}")

        # Per-fold summary
        logger.info(f"\nPer-fold metrics:")
        for i, fm in enumerate(all_fold_metrics):
            logger.info(
                f"  Fold {i+1}: gate_acc={fm['gate_accuracy']:.3f} "
                f"precision={fm['gate_precision']:.3f} "
                f"coverage={fm['gate_coverage']:.3f} "
                f"sortino_imp={fm['sortino_improvement']:.3f} "
                f"tp_mae={fm['tp_mae']:.2f} sl_mae={fm['sl_mae']:.2f} "
                f"tp_std={fm.get('pred_tp_std', 0):.2f} sl_std={fm.get('pred_sl_std', 0):.2f}"
            )

        # TP/SL prediction diversity check (v2 diagnostic)
        logger.info(f"\nTP/SL prediction diversity (v2 fix check):")
        for i, fm in enumerate(all_fold_metrics):
            tp_std = fm.get('pred_tp_std', 0)
            sl_std = fm.get('pred_sl_std', 0)
            status_tp = "OK" if tp_std > 0.1 else "COLLAPSED"
            status_sl = "OK" if sl_std > 0.1 else "COLLAPSED"
            logger.info(f"  Fold {i+1}: TP std={tp_std:.3f} [{status_tp}] | SL std={sl_std:.3f} [{status_sl}]")

        # Save concat results
        concat_results = {
            "version": args.version,
            "gate_accuracy": float(overall_acc),
            "gate_coverage": float(gated_mask.mean()),
            "pnl_ungated_ticks": float(pnl_ungated),
            "pnl_gated_ticks": float(pnl_gated),
            "sortino_ungated": float(sortino_ungated),
            "sortino_gated": float(sortino_gated),
            "n_folds": len(all_fold_metrics),
            "confidence_mean": float(confidence.mean()),
            "high_conf_pct": float((confidence > 0.4).mean()),
            "per_fold": all_fold_metrics,
            "v2_changes": [
                "log_space_tpsl",
                "focal_loss_gamma_2.0",
                "staged_training_5+15",
                "hidden_dim_192",
                "model_agreement_feature",
                "conviction_gradient_feature",
                "temperature_scaling",
                "calibration_loss",
                "separate_lr_groups",
                "batch_size_1024",
            ],
        }

        with open(output_dir / "concat_results.json", "w") as f:
            json.dump(concat_results, f, indent=2, default=str)

        # MLflow concat metrics
        if mlflow_run:
            try:
                mlflow.log_metrics({
                    "concat_gate_accuracy": float(overall_acc),
                    "concat_gate_coverage": float(gated_mask.mean()),
                    "concat_pnl_gated_ticks": float(pnl_gated),
                    "concat_sortino_gated": float(sortino_gated),
                    "concat_sortino_improvement": float(sortino_gated - sortino_ungated),
                    "concat_confidence_mean": float(confidence.mean()),
                    "concat_high_conf_pct": float((confidence > 0.4).mean()),
                })
            except Exception:
                pass

    if mlflow_run:
        try:
            mlflow.end_run()
        except Exception:
            pass

    logger.info("\n" + "=" * 60)
    logger.info("Smart Execution System v2 training complete.")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
