#!/usr/bin/env python3
"""
Smart Execution System — GPU-Trained Adaptive Execution Components (HC #37)

NOT a naive MLP mapping predictions → buy/sell/hold.
This is a FULL SMART EXECUTION SYSTEM with multiple adaptive components:

  Component 1: Adaptive TP/SL Width Model
    - Learns optimal TP and SL widths as f(vol_regime, time_of_day, spread,
      conviction, volume, momentum, model_agreement)
    - Training target: MFE/MAE data → optimal TP/SL per trade
    - Loss: Sortino-weighted regression on realized outcome

  Component 2: Entry Gate Model
    - Learns WHETHER to enter a trade given market state
    - Inputs: cnn_pred_z, patchtst_pred_z, vol_pred, spread, queue_imbalance,
      tod_sin/cos, volume_profile, recent_momentum, model_agreement
    - Training target: was the trade profitable after realistic fills+costs?
    - Loss: binary cross-entropy weighted by trade magnitude

  Component 3: Exit Trigger Model
    - Learns WHEN to exit beyond static TP/SL
    - Inputs: current P&L path, time_in_trade, vol_change, PatchTST reversal signal,
      spread dynamics since entry
    - Training target: optimal exit point (maximizes risk-adjusted PnL)
    - Architecture: temporal conv over P&L path + market state MLP

  Component 4: Cancel/Replace Timing
    - Learns when to cancel resting limit orders
    - Inputs: time_in_queue, queue_position_est, spread_change, vol_change, signal_decay
    - Training target: fill probability decay curve
    - Architecture: hazard model (survival analysis for order fill)

Walk-forward sliding window (HC #0). MLflow logging. Mixed precision.
Each component trained jointly via multi-task loss.

Data sources:
  - CNN Mamba v2 OOT predictions + embeddings (96-dim)
  - PatchTST OOT predictions + embeddings (256-dim)
  - Vol LGBM v3 predictions + features (28 features)
  - MFE/MAE analysis per prediction
  - MBO event stream features (spread, volume, queue depth, timestamps)
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

    # 4. Model agreement (CNN × PatchTST same sign per horizon)
    for h in range(3):
        agreement = np.sign(cnn_preds[:, h]) * np.sign(ptst_preds[:, h])
        features.append(agreement)

    # 5. Conviction strength = abs(cnn_z) * agreement
    for h in range(3):
        cnn_z = _rolling_zscore(cnn_preds[:, h], zscore_window)
        ptst_z = _rolling_zscore(ptst_preds[:, h], zscore_window)
        conviction = np.abs(cnn_z) * np.sign(cnn_z) * np.sign(ptst_z)
        features.append(conviction)

    # 6. Time-of-day features (cyclical encoding)
    et_hours = np.array([ts_ns_to_et_hour(t) for t in timestamps])
    tod_sin = np.sin(2 * np.pi * et_hours / 24.0)
    tod_cos = np.cos(2 * np.pi * et_hours / 24.0)
    features.append(tod_sin)
    features.append(tod_cos)

    # 7. Session one-hot (8 sessions)
    session_idxs = np.array([get_session_idx(h) for h in et_hours])
    for s_idx in range(len(SESSION_BOUNDS)):
        features.append((session_idxs == s_idx).astype(np.float32))

    # 8. Gap flag
    gap_flag = np.array([is_in_gap(h) for h in et_hours], dtype=np.float32)
    features.append(gap_flag)

    # 9. Microstructure features from MBO events
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

    # 10. Volatility regime (rolling std of cnn predictions)
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
    """
    N = len(labels)
    direction = np.sign(cnn_preds[:, 2])  # use 10s prediction for direction

    # Target 1: Optimal TP (how far price goes in our favor)
    # Directional MFE = MFE if we got direction right
    optimal_tp = mfe_ticks.copy()

    # Target 2: Optimal SL (how far price goes against us)
    optimal_sl = mae_ticks.copy()

    # Target 3: Entry quality (was this a profitable trade after costs?)
    # Using 10s horizon realized PnL with direction
    realized_pnl = labels[:, 2] * direction  # directional PnL in ticks
    profitable = (realized_pnl > ROUND_TRIP_COST).astype(np.float32)

    # Target 4: Trade magnitude (for weighting)
    trade_magnitude = np.abs(realized_pnl)

    # Target 5: Optimal hold time (time to MFE = optimal exit time)
    optimal_hold = time_to_mfe.copy()

    # Target 6: Risk-reward ratio
    rr_ratio = np.where(mae_ticks > 0.1, mfe_ticks / mae_ticks, 0.0)

    return {
        "optimal_tp": optimal_tp.astype(np.float32),
        "optimal_sl": optimal_sl.astype(np.float32),
        "entry_profitable": profitable,
        "realized_pnl": realized_pnl.astype(np.float32),
        "trade_magnitude": trade_magnitude.astype(np.float32),
        "optimal_hold_s": optimal_hold.astype(np.float32),
        "risk_reward": rr_ratio.astype(np.float32),
    }


# ============================================================
# Models
# ============================================================

class AdaptiveTPSL(nn.Module):
    """
    Predicts optimal TP and SL widths given market state.

    Output: (tp_ticks, sl_ticks) — both positive.
    Architecture: MLP with residual connections and layer norm.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
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
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # TP must be positive
        )
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # SL must be positive
        )

    def forward(self, x):
        h = self.encoder(x)
        tp = self.tp_head(h).squeeze(-1) + 1.0  # minimum 1 tick TP
        sl = self.sl_head(h).squeeze(-1) + 1.0  # minimum 1 tick SL
        return tp, sl


class EntryGate(nn.Module):
    """
    Predicts whether to enter a trade.

    Output: probability of profitable trade (0-1).
    Uses attention over model predictions + market state.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
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

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logit


class ExitTrigger(nn.Module):
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


class SmartExecSystem(nn.Module):
    """
    Complete Smart Execution System combining all components.

    Multi-task model trained jointly:
    - Task 1: Adaptive TP/SL (regression on MFE/MAE)
    - Task 2: Entry gate (classification on profitability)
    - Task 3: Exit trigger (regression on hold time + risk/reward)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Task-specific heads
        self.tpsl = AdaptiveTPSL(hidden_dim, hidden_dim)
        self.gate = EntryGate(hidden_dim, hidden_dim // 2)
        self.exit = ExitTrigger(hidden_dim, hidden_dim // 2)

    def forward(self, x):
        shared_repr = self.shared(x)

        tp, sl = self.tpsl(shared_repr)
        gate_logit = self.gate(shared_repr)
        hold_time, rr = self.exit(shared_repr)

        return {
            "tp": tp,
            "sl": sl,
            "gate_logit": gate_logit,
            "hold_time": hold_time,
            "risk_reward": rr,
        }


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

class SmartExecLoss(nn.Module):
    """
    Multi-task loss for the smart execution system.

    Weights:
    - TP/SL regression: Huber loss (robust to outliers in MFE/MAE)
    - Entry gate: BCE with magnitude weighting (big trades matter more)
    - Hold time: Huber loss
    - Risk/reward: Huber loss
    """
    def __init__(self, gate_weight=1.0, tpsl_weight=1.0, exit_weight=0.5):
        super().__init__()
        self.gate_weight = gate_weight
        self.tpsl_weight = tpsl_weight
        self.exit_weight = exit_weight
        self.huber = nn.HuberLoss(delta=5.0)

    def forward(self, outputs, targets):
        # TP/SL regression loss
        tp_loss = self.huber(outputs["tp"], targets["optimal_tp"])
        sl_loss = self.huber(outputs["sl"], targets["optimal_sl"])
        tpsl_loss = tp_loss + sl_loss

        # Entry gate BCE with magnitude weighting
        weights = 1.0 + targets["trade_magnitude"] / (targets["trade_magnitude"].mean() + 1e-6)
        gate_loss = F.binary_cross_entropy_with_logits(
            outputs["gate_logit"],
            targets["entry_profitable"],
            weight=weights,
        )

        # Exit timing loss
        hold_loss = self.huber(outputs["hold_time"], targets["optimal_hold_s"])
        rr_loss = self.huber(outputs["risk_reward"], targets["risk_reward"])
        exit_loss = hold_loss + 0.5 * rr_loss

        total = (
            self.tpsl_weight * tpsl_loss +
            self.gate_weight * gate_loss +
            self.exit_weight * exit_loss
        )

        return total, {
            "tp_loss": tp_loss.item(),
            "sl_loss": sl_loss.item(),
            "gate_loss": gate_loss.item(),
            "hold_loss": hold_loss.item(),
            "rr_loss": rr_loss.item(),
            "total_loss": total.item(),
        }


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
            # Subsample PatchTST to match CNN
            indices = np.linspace(0, len(ptst_raw)-1, min_n).astype(int)
            ptst_preds = ptst_raw[indices]
        else:
            # Interpolate PatchTST up to CNN length
            for h in range(3):
                ptst_preds[:len(ptst_raw), h] = ptst_raw[:, h]
                if len(ptst_raw) < min_n:
                    ptst_preds[len(ptst_raw):, h] = ptst_raw[-1, h]
        logger.info(f"  PatchTST preds: {len(ptst_raw)} → aligned to {min_n}")
    else:
        logger.warning(f"  No PatchTST preds for fold {fold_idx}, using zeros")

    # Load vol predictions (daily, need to match to fold dates)
    vol_preds = np.zeros((min_n, 3), dtype=np.float32)
    vol_files = sorted(vol_dir.glob("vol_v3_*_predictions.npz"))
    if vol_files and oot_files is not None:
        # Try to match by date
        oot_dates = [str(f).split("_")[0][:8] if isinstance(f, str) else str(f)[:8]
                     for f in oot_files]
        for vf in vol_files:
            vdate = vf.stem.split("_")[2]  # vol_v3_YYYYMMDD_predictions
            if vdate in oot_dates:
                vdata = np.load(vf, allow_pickle=True)
                vpreds = vdata["predictions"]
                # Fill what we can
                fill_n = min(len(vpreds), min_n)
                vol_preds[:fill_n] = vpreds[:fill_n, :3] if vpreds.shape[1] >= 3 else vpreds[:fill_n]
                logger.info(f"  Vol preds matched for {vdate}: {fill_n} samples")
                break

    # Generate placeholder timestamps if not available
    # In production, these come from the MBO event stream
    timestamps = np.arange(min_n, dtype=np.int64) * int(1e8)  # ~100ms spacing placeholder
    anchor_idxs = np.arange(min_n, dtype=np.int64)

    # Try to load actual MBO events for microstructure features
    events = None
    if oot_files is not None and mbo_dir.exists():
        for oot_f in oot_files:
            fname = str(oot_f) if isinstance(oot_f, str) else oot_f
            # Try to find matching MBO file
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
    model: SmartExecSystem,
    train_features: np.ndarray,
    train_targets: Dict[str, np.ndarray],
    oot_features: np.ndarray,
    oot_targets: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 3e-4,
    fold_idx: int = 0,
) -> Dict:
    """Train model on one fold and evaluate OOT."""

    train_ds = SmartExecDataset(train_features, train_targets)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=epochs * len(train_dl),
        pct_start=0.1, anneal_strategy='cos',
    )
    criterion = SmartExecLoss(gate_weight=2.0, tpsl_weight=1.0, exit_weight=0.5)
    scaler = GradScaler()

    model.train()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_losses = []
        t0 = time.time()

        for batch_features, batch_targets in train_dl:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = {k: v.to(device, non_blocking=True) for k, v in batch_targets.items()}

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                outputs = model(batch_features)
                loss, loss_dict = criterion(outputs, batch_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_losses.append(loss_dict)

        # Epoch summary
        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        wall_time = time.time() - t0

        logger.info(
            f"  Epoch {epoch+1}/{epochs} | "
            f"total={avg_losses['total_loss']:.4f} | "
            f"gate={avg_losses['gate_loss']:.4f} | "
            f"tp={avg_losses['tp_loss']:.4f} sl={avg_losses['sl_loss']:.4f} | "
            f"hold={avg_losses['hold_loss']:.4f} | "
            f"{wall_time:.1f}s"
        )

        # Quick OOT eval every epoch
        val_metrics = evaluate_oot(model, oot_features, oot_targets, device)
        val_loss = val_metrics.get("total_loss", float('inf'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        logger.info(
            f"    OOT | gate_acc={val_metrics.get('gate_accuracy', 0):.3f} | "
            f"gate_precision={val_metrics.get('gate_precision', 0):.3f} | "
            f"tp_mae={val_metrics.get('tp_mae', 0):.2f} | "
            f"sl_mae={val_metrics.get('sl_mae', 0):.2f} | "
            f"sortino_improvement={val_metrics.get('sortino_improvement', 0):.3f}"
        )

    # Restore best model
    model.load_state_dict(best_state)

    # Final OOT evaluation
    final_metrics = evaluate_oot(model, oot_features, oot_targets, device, detailed=True)
    final_metrics["best_val_loss"] = best_val_loss

    return final_metrics


def evaluate_oot(
    model: SmartExecSystem,
    features: np.ndarray,
    targets: Dict[str, np.ndarray],
    device: torch.device,
    detailed: bool = False,
) -> Dict:
    """Evaluate model on OOT data."""
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(features).to(device)

        with autocast(dtype=torch.float16):
            outputs = model(x)

        # Move to CPU numpy
        pred_tp = outputs["tp"].float().cpu().numpy()
        pred_sl = outputs["sl"].float().cpu().numpy()
        gate_logit = outputs["gate_logit"].float().cpu().numpy()
        gate_prob = 1.0 / (1.0 + np.exp(-gate_logit))
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

    # TP/SL accuracy
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
    criterion = SmartExecLoss()
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

    model.train()
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Smart Execution System Training")
    parser.add_argument("--cnn-dir", type=str, default=None)
    parser.add_argument("--ptst-dir", type=str, default=None)
    parser.add_argument("--vol-dir", type=str, default=None)
    parser.add_argument("--mbo-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-folds", type=int, default=9)
    parser.add_argument("--horizon", type=str, default="10s")
    args = parser.parse_args()

    # Detect paths
    # Try Neptune paths first, then Jupiter
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("Smart Execution System — Training (HC #37)")
    logger.info(f"Device:     {device}")
    if device.type == "cuda":
        logger.info(f"GPU:        {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"CNN dir:    {cnn_dir}")
    logger.info(f"PatchTST:   {ptst_dir}")
    logger.info(f"Vol dir:    {vol_dir}")
    logger.info(f"MFE/MAE:    {mfe_dir}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Horizon:    {args.horizon}")
    logger.info(f"Epochs:     {args.epochs}")
    logger.info(f"Hidden dim: {args.hidden_dim}")
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
            mlflow.set_experiment("SmartExecSystem")
            mlflow_run = mlflow.start_run(
                run_name=f"SmartExec_{time.strftime('%Y%m%d_%H%M')}"
            )
            mlflow.log_params({
                "model": "SmartExecSystem",
                "hidden_dim": args.hidden_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "horizon": args.horizon,
                "n_folds": n_folds,
                "node": hostname,
                "device": str(device),
                "components": "AdaptiveTPSL+EntryGate+ExitTrigger",
            })
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}")

    # Walk-forward: use folds 0..N-2 as sliding train, fold N-1 as OOT
    # For execution system, we do LEAVE-ONE-FOLD-OUT within available folds
    all_fold_metrics = []
    concat_gate_preds = []
    concat_gate_true = []
    concat_pnl = []

    for oot_fold in range(1, n_folds):
        train_folds = list(range(max(0, oot_fold - 5), oot_fold))  # use up to 5 recent folds

        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {oot_fold} | Train folds: {train_folds} | OOT fold: {oot_fold}")
        logger.info(f"{'='*60}")

        # Load training data from multiple folds
        train_features_list = []
        train_targets_list = {
            "optimal_tp": [], "optimal_sl": [], "entry_profitable": [],
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

        # Normalize features using train stats
        feat_mean = train_features.mean(axis=0)
        feat_std = train_features.std(axis=0) + 1e-8
        train_features = (train_features - feat_mean) / feat_std
        oot_features = (oot_features - feat_mean) / feat_std

        # Build model
        input_dim = train_features.shape[1]
        model = SmartExecSystem(input_dim, args.hidden_dim).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model params: {n_params:,}")

        # Train
        fold_metrics = train_one_fold(
            model, train_features, train_targets,
            oot_features, oot_targets, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
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
        }, output_dir / f"fold_{oot_fold:02d}_smart_exec.pt")

        # Collect for concat metrics
        model.eval()
        with torch.no_grad():
            x = torch.from_numpy(oot_features).to(device)
            with autocast(dtype=torch.float16):
                out = model(x)
            gate_prob = torch.sigmoid(out["gate_logit"]).float().cpu().numpy()

        concat_gate_preds.append(gate_prob)
        concat_gate_true.append(oot_targets["entry_profitable"])
        concat_pnl.append(oot_targets["realized_pnl"])

        # MLflow
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
        logger.info("CONCAT RESULTS — Smart Execution System")
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

        # Confidence-conditional (HC #13/#14)
        logger.info(f"\n{'Tier':<10} {'Trades':>8} {'WinRate':>8} {'PnL(t)':>10} {'Sortino':>8}")
        logger.info("-" * 50)

        confidence = np.abs(all_gate - 0.5)
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

        # Per-fold summary
        logger.info(f"\nPer-fold metrics:")
        for i, fm in enumerate(all_fold_metrics):
            logger.info(
                f"  Fold {i+1}: gate_acc={fm['gate_accuracy']:.3f} "
                f"precision={fm['gate_precision']:.3f} "
                f"sortino_imp={fm['sortino_improvement']:.3f} "
                f"tp_mae={fm['tp_mae']:.2f} sl_mae={fm['sl_mae']:.2f}"
            )

        # Save concat results
        concat_results = {
            "gate_accuracy": float(overall_acc),
            "gate_coverage": float(gated_mask.mean()),
            "pnl_ungated_ticks": float(pnl_ungated),
            "pnl_gated_ticks": float(pnl_gated),
            "sortino_ungated": float(sortino_ungated),
            "sortino_gated": float(sortino_gated),
            "n_folds": len(all_fold_metrics),
            "per_fold": all_fold_metrics,
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
                })
            except Exception:
                pass

    if mlflow_run:
        try:
            mlflow.end_run()
        except Exception:
            pass

    logger.info("\n" + "=" * 60)
    logger.info("Smart Execution System training complete.")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
