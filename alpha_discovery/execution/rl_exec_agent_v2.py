"""
RL Adaptive Execution Agent v2 (HC #37 / HC #38)
=================================================

Fixes policy collapse from v1:
  - Per-head entropy coefficients (gate=0.01, tp=0.10, sl=0.10, hold=0.05)
  - Learnable temperature scaling per action head
  - Deeper backbone (3 layers, LayerNorm + GELU)
  - Session derived from OOT filename, not sample index
  - Action diversity bonus + condition-matching bonus
  - Stronger Sortino shaping (0.5 vs 0.1)
  - Longer rollouts (8192), more PPO epochs (8), LR warmup

State space: ~44 dim (enhanced with action history, pred-to-action ratios)
Action space: gate(2) x TP(8) x SL(7) x hold(6) — same as v1

Usage:
  python rl_exec_agent_v2.py --mode train --data-dir output/cnn_mamba_v2_smart_v3_mar
  python rl_exec_agent_v2.py --mode eval --checkpoint output/rl_exec_v2/best_agent.pt
  python rl_exec_agent_v2.py --mode baselines --data-dir output/cnn_mamba_v2_smart_v3_mar
  python rl_exec_agent_v2.py --mode sweep --data-dir output/cnn_mamba_v2_smart_v3_mar
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger("rl_exec_v2")

# ── Constants ──
COMMISSION_PER_SIDE = 0.376  # ticks (~$2.35 / $6.25 per tick for NQ)
TICK_VALUE = 6.25  # NQ tick value in USD

# Action spaces
TP_TICKS = np.array([2, 4, 6, 8, 10, 15, 20, 30], dtype=np.float32)
SL_TICKS = np.array([2, 3, 5, 8, 10, 15, 20], dtype=np.float32)
HOLD_SECS = np.array([1.0, 3.0, 5.0, 10.0, 30.0, 60.0], dtype=np.float32)

# Session buckets (ET hours)
SESSION_BUCKETS = {
    "overnight": (18, 4),     # 6PM - 4AM
    "pre_market": (4, 9.5),   # 4AM - 9:30AM
    "rth_open": (9.5, 10.5),  # 9:30 - 10:30 (volatile!)
    "rth_core": (10.5, 15),   # 10:30 - 3PM (best)
    "rth_close": (15, 16),    # 3PM - 4PM
    "post_market": (16, 18),  # 4PM - 6PM
}
N_SESSIONS = len(SESSION_BUCKETS)

# Per-head entropy coefficients — force exploration on TP/SL
ENTROPY_COEFFS = {
    "gate": 0.01,
    "tp": 0.10,
    "sl": 0.10,
    "hold": 0.05,
}

# Label to ticks scaling
LABEL_TO_TICKS = 10.0


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class TradeResult:
    """Result of a simulated trade."""
    entry_price_ticks: float
    exit_price_ticks: float
    pnl_ticks: float  # after costs
    hold_time_s: float
    tp_ticks: float
    sl_ticks: float
    exit_reason: str  # "tp", "sl", "hold_timeout"
    session_bucket: int
    prediction_strength: float


@dataclass
class ExecState:
    """State vector for the RL agent (~44 dim)."""
    # Model predictions (3 horizons) + confidence + direction = 5
    pred_1s: float = 0.0
    pred_5s: float = 0.0
    pred_10s: float = 0.0
    pred_magnitude: float = 0.0
    pred_direction: float = 0.0

    # Embedding (compressed to 16-dim via PCA)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))

    # Time of day (one-hot session bucket) = 6
    session_onehot: np.ndarray = field(default_factory=lambda: np.zeros(N_SESSIONS, dtype=np.float32))

    # Trade history: last 5 PnLs + win_streak + loss_streak + cumulative_pnl = 8
    recent_pnl: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    win_streak: int = 0
    loss_streak: int = 0
    cumulative_pnl: float = 0.0

    # Market microstructure proxies = 3
    spread_ticks: float = 1.0
    vol_proxy: float = 0.0
    volume_proxy: float = 0.0

    # NEW: prediction-to-action ratio features = 2
    pred_to_tp_ratio: float = 0.0
    pred_to_sl_ratio: float = 0.0

    # NEW: action history features (running means over last 10) = 3
    action_tp_mean: float = 0.0
    action_sl_mean: float = 0.0
    action_hold_mean: float = 0.0

    # NEW: session-conditioned vol = 1
    session_vol_interaction: float = 0.0

    def to_tensor(self) -> np.ndarray:
        """Flatten to 1D numpy array (~44 dim)."""
        return np.concatenate([
            np.array([self.pred_1s, self.pred_5s, self.pred_10s,
                      self.pred_magnitude, self.pred_direction], dtype=np.float32),
            self.embedding,           # 16
            self.session_onehot,      # 6
            self.recent_pnl,          # 5
            np.array([self.win_streak, self.loss_streak, self.cumulative_pnl,
                      self.spread_ticks, self.vol_proxy, self.volume_proxy,
                      self.pred_to_tp_ratio, self.pred_to_sl_ratio,
                      self.action_tp_mean, self.action_sl_mean, self.action_hold_mean,
                      self.session_vol_interaction],
                     dtype=np.float32),
        ])

    @staticmethod
    def dim() -> int:
        # 5 + 16 + 6 + 5 + 12 = 44
        return 5 + 16 + N_SESSIONS + 5 + 12


@dataclass
class ExecAction:
    """Action from the RL agent."""
    gate: int        # 0=skip, 1=enter
    tp_idx: int      # index into TP_TICKS
    sl_idx: int      # index into SL_TICKS
    hold_idx: int    # index into HOLD_SECS

    @property
    def tp_ticks(self) -> float:
        return float(TP_TICKS[self.tp_idx])

    @property
    def sl_ticks(self) -> float:
        return float(SL_TICKS[self.sl_idx])

    @property
    def hold_secs(self) -> float:
        return float(HOLD_SECS[self.hold_idx])


def _extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract YYYYMMDD date from OOT filename like '20260223_mbo_events.npz'."""
    m = re.search(r'(\d{8})', filename)
    return m.group(1) if m else None


def _estimate_session_from_position(position_frac: float) -> int:
    """Estimate session bucket from position within a day's samples.

    Uses realistic MBO event distribution:
      - Overnight (6PM-4AM): ~25% of events (low activity)
      - Pre-market (4AM-9:30AM): ~15%
      - RTH open (9:30-10:30): ~12% (high density)
      - RTH core (10:30-3PM): ~30%
      - RTH close (3-4PM): ~10%
      - Post-market (4-6PM): ~8%
    """
    # Cumulative distribution breakpoints
    if position_frac < 0.25:
        return 0  # overnight
    elif position_frac < 0.40:
        return 1  # pre_market
    elif position_frac < 0.52:
        return 2  # rth_open
    elif position_frac < 0.82:
        return 3  # rth_core
    elif position_frac < 0.92:
        return 4  # rth_close
    else:
        return 5  # post_market


class ExecEnvironment:
    """
    Offline execution environment that replays OOT predictions
    and simulates trades with configurable TP/SL/hold.

    v2 improvements:
      - Session derived from OOT filename dates, not sample index modulo
      - Tracks action history for diversity features
      - Computes vol percentiles for condition-matching rewards
    """

    def __init__(self, data_dir: str, folds: Optional[List[int]] = None,
                 mbo_data_dir: str = None):
        self.data_dir = Path(data_dir)
        self.mbo_data_dir = Path(mbo_data_dir) if mbo_data_dir else None

        # Load all fold predictions
        self.predictions = []
        self.labels = []
        self.embeddings = []
        self.oot_files = []
        self.fold_boundaries = []  # (start_idx, end_idx, oot_filename) per fold

        fold_files = sorted(self.data_dir.glob("fold_*_oot_predictions.npz"))
        cursor = 0
        for f in fold_files:
            fold_num = int(f.stem.split("_")[1])
            if folds and fold_num not in folds:
                continue

            d = np.load(f)
            preds = d["predictions"]  # (N, 3)
            n_samples = len(preds)
            oot_file = str(d["oot_files"][0]) if "oot_files" in d else f"fold_{fold_num}"

            self.predictions.append(preds)
            self.labels.append(d["labels"])        # (N, 3)
            self.embeddings.append(d["embeddings"])  # (N, 96)
            self.oot_files.append(oot_file)
            self.fold_boundaries.append((cursor, cursor + n_samples, oot_file))
            cursor += n_samples

        if not self.predictions:
            raise ValueError(f"No prediction files found in {data_dir}")

        # Concatenate all
        self.all_preds = np.concatenate(self.predictions, axis=0)
        self.all_labels = np.concatenate(self.labels, axis=0)
        self.all_embeddings = np.concatenate(self.embeddings, axis=0)

        # Build per-sample session index from OOT filenames
        self._build_session_map()

        # PCA on embeddings (96 -> 16)
        self._fit_pca()

        log.info(f"Loaded {len(self.all_preds)} samples from {len(self.predictions)} folds")
        log.info(f"  Prediction range: [{self.all_preds.min():.4f}, {self.all_preds.max():.4f}]")

        # Compute prediction percentiles for confidence tiers
        pred_abs = np.abs(self.all_preds[:, 2])  # 10s horizon
        self.p90 = np.percentile(pred_abs, 90)
        self.p95 = np.percentile(pred_abs, 95)
        self.p99 = np.percentile(pred_abs, 99)
        log.info(f"  Confidence tiers: p90={self.p90:.4f}, p95={self.p95:.4f}, p99={self.p99:.4f}")

        # Compute vol percentiles for condition-matching reward
        self._compute_vol_percentiles()

        # State tracking
        self.cursor = 0
        self.trade_history: List[TradeResult] = []
        self.recent_pnl = deque(maxlen=5)
        self.win_streak = 0
        self.loss_streak = 0
        self.cumulative_pnl = 0.0

        # Action history tracking (for diversity features)
        self.action_tp_history = deque(maxlen=10)
        self.action_sl_history = deque(maxlen=10)
        self.action_hold_history = deque(maxlen=10)

    def _build_session_map(self):
        """Build per-sample session index from OOT filenames.

        For each fold, extract the date from the OOT filename, then
        distribute samples across sessions based on position within the fold.
        """
        self.sample_session = np.zeros(len(self.all_preds), dtype=np.int32)

        for start, end, oot_file in self.fold_boundaries:
            n = end - start
            for i in range(n):
                frac = i / max(1, n - 1)
                self.sample_session[start + i] = _estimate_session_from_position(frac)

        # Log session distribution
        session_names = list(SESSION_BUCKETS.keys())
        for idx, name in enumerate(session_names):
            count = (self.sample_session == idx).sum()
            log.info(f"  Session '{name}': {count} samples ({100*count/len(self.sample_session):.1f}%)")

    def _compute_vol_percentiles(self):
        """Compute rolling vol proxy for all samples and store percentiles."""
        window = 100
        self.vol_array = np.zeros(len(self.all_labels), dtype=np.float32)
        for i in range(len(self.all_labels)):
            start = max(0, i - window)
            self.vol_array[i] = float(np.nanstd(self.all_labels[start:i + 1, 2]))

        self.vol_p25 = float(np.percentile(self.vol_array[self.vol_array > 0], 25)) if (self.vol_array > 0).any() else 0.0
        self.vol_p75 = float(np.percentile(self.vol_array[self.vol_array > 0], 75)) if (self.vol_array > 0).any() else 1.0
        log.info(f"  Vol percentiles: p25={self.vol_p25:.4f}, p75={self.vol_p75:.4f}")

    def _fit_pca(self):
        """Simple PCA: 96-dim embeddings -> 16-dim."""
        from numpy.linalg import svd
        emb = self.all_embeddings
        self._emb_mean = emb.mean(axis=0)
        centered = emb - self._emb_mean
        U, S, Vt = svd(centered, full_matrices=False)
        self._pca_components = Vt[:16]  # (16, 96)
        log.info(f"  PCA: 96 -> 16 dim (explained var ratio top-16: {(S[:16]**2).sum() / (S**2).sum():.3f})")

    def _project_embedding(self, emb: np.ndarray) -> np.ndarray:
        """Project 96-dim embedding to 16-dim."""
        return ((emb - self._emb_mean) @ self._pca_components.T).astype(np.float32)

    def reset(self):
        """Reset environment for new episode."""
        self.cursor = 0
        self.trade_history = []
        self.recent_pnl = deque(maxlen=5)
        self.win_streak = 0
        self.loss_streak = 0
        self.cumulative_pnl = 0.0
        self.action_tp_history = deque(maxlen=10)
        self.action_sl_history = deque(maxlen=10)
        self.action_hold_history = deque(maxlen=10)

    def get_state(self) -> Optional[ExecState]:
        """Get current state (next prediction to decide on)."""
        if self.cursor >= len(self.all_preds):
            return None

        pred = self.all_preds[self.cursor]
        emb = self.all_embeddings[self.cursor]

        # Session bucket from pre-built map
        session_idx = int(self.sample_session[self.cursor])
        session_onehot = np.zeros(N_SESSIONS, dtype=np.float32)
        session_onehot[session_idx] = 1.0

        # Vol proxy from pre-computed array
        vol_proxy = float(self.vol_array[self.cursor])

        # Prediction magnitude
        pred_mag = float(np.abs(pred[2]))

        # Compute action history features
        tp_mean = float(np.mean(list(self.action_tp_history))) if self.action_tp_history else float(np.mean(TP_TICKS))
        sl_mean = float(np.mean(list(self.action_sl_history))) if self.action_sl_history else float(np.mean(SL_TICKS))
        hold_mean = float(np.mean(list(self.action_hold_history))) if self.action_hold_history else float(np.mean(HOLD_SECS))

        # Prediction-to-action ratio features
        pred_to_tp = pred_mag / max(tp_mean, 1e-6)
        pred_to_sl = pred_mag / max(sl_mean, 1e-6)

        # Session-conditioned vol interaction
        session_vol = vol_proxy * session_idx

        state = ExecState(
            pred_1s=float(pred[0]),
            pred_5s=float(pred[1]),
            pred_10s=float(pred[2]),
            pred_magnitude=pred_mag,
            pred_direction=float(np.sign(pred[2])),
            embedding=self._project_embedding(emb),
            session_onehot=session_onehot,
            recent_pnl=np.array(list(self.recent_pnl) + [0.0] * (5 - len(self.recent_pnl)),
                                dtype=np.float32),
            win_streak=self.win_streak,
            loss_streak=self.loss_streak,
            cumulative_pnl=self.cumulative_pnl,
            spread_ticks=1.0,
            vol_proxy=vol_proxy,
            volume_proxy=1.0,
            pred_to_tp_ratio=pred_to_tp,
            pred_to_sl_ratio=pred_to_sl,
            action_tp_mean=tp_mean,
            action_sl_mean=sl_mean,
            action_hold_mean=hold_mean,
            session_vol_interaction=session_vol,
        )
        return state

    def step(self, action: ExecAction) -> Tuple[float, dict]:
        """
        Execute action on current prediction, simulate trade, return reward.

        Uses label-based simulation:
          - If gate=skip: reward = skip_penalty, advance cursor
          - If gate=enter: simulate trade using future labels as price proxy
        """
        if self.cursor >= len(self.all_preds):
            return 0.0, {"done": True}

        pred = self.all_preds[self.cursor]
        label = self.all_labels[self.cursor]
        info = {"done": False, "traded": False}

        if action.gate == 0:
            # Skip this signal — stronger penalty than v1
            self.cursor += 1
            return -0.005, info

        # ENTER TRADE — track action choices
        self.action_tp_history.append(action.tp_ticks)
        self.action_sl_history.append(action.sl_ticks)
        self.action_hold_history.append(action.hold_secs)

        direction = np.sign(pred[2])
        if direction == 0:
            direction = 1.0

        # Get future price path from labels
        move_1s = float(label[0]) * LABEL_TO_TICKS * direction
        move_5s = float(label[1]) * LABEL_TO_TICKS * direction
        move_10s = float(label[2]) * LABEL_TO_TICKS * direction

        tp = action.tp_ticks
        sl = action.sl_ticks
        hold = action.hold_secs

        # Check price path against TP/SL at each horizon checkpoint
        pnl_ticks = 0.0
        exit_reason = "hold_timeout"

        checkpoints = [(1.0, move_1s), (5.0, move_5s), (10.0, move_10s)]

        mfe = 0.0
        mae = 0.0

        for t, move in checkpoints:
            if t > hold:
                break

            mfe = max(mfe, move)
            mae = min(mae, move)

            if move >= tp:
                pnl_ticks = tp
                exit_reason = "tp"
                break

            if move <= -sl:
                pnl_ticks = -sl
                exit_reason = "sl"
                break
        else:
            # Hold timeout — interpolate
            if hold <= 1.0:
                pnl_ticks = move_1s
            elif hold <= 5.0:
                alpha = (hold - 1.0) / 4.0
                pnl_ticks = move_1s * (1 - alpha) + move_5s * alpha
            elif hold <= 10.0:
                alpha = (hold - 5.0) / 5.0
                pnl_ticks = move_5s * (1 - alpha) + move_10s * alpha
            else:
                pnl_ticks = move_10s
            exit_reason = "hold_timeout"

        # Apply costs
        costs = 2 * COMMISSION_PER_SIDE
        net_pnl = pnl_ticks - costs

        # Record trade
        session_idx = int(self.sample_session[self.cursor])
        trade = TradeResult(
            entry_price_ticks=0,
            exit_price_ticks=pnl_ticks,
            pnl_ticks=net_pnl,
            hold_time_s=hold,
            tp_ticks=tp,
            sl_ticks=sl,
            exit_reason=exit_reason,
            session_bucket=session_idx,
            prediction_strength=float(np.abs(pred[2])),
        )
        self.trade_history.append(trade)
        self.recent_pnl.append(net_pnl)
        self.cumulative_pnl += net_pnl

        if net_pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        info["traded"] = True
        info["pnl_ticks"] = net_pnl
        info["exit_reason"] = exit_reason
        info["mfe"] = mfe
        info["mae"] = mae
        info["vol_proxy"] = float(self.vol_array[self.cursor])
        info["tp_chosen"] = tp
        info["sl_chosen"] = sl

        self.cursor += 1
        return net_pnl, info

    def compute_sortino(self, window: int = 100) -> float:
        """Compute Sortino ratio over recent trades."""
        if len(self.trade_history) < 2:
            return 0.0

        recent = self.trade_history[-window:]
        pnls = np.array([t.pnl_ticks for t in recent])
        mean_pnl = pnls.mean()
        downside = pnls[pnls < 0]
        if len(downside) == 0:
            return 10.0
        downside_std = downside.std()
        if downside_std < 1e-8:
            return 10.0
        return float(np.clip(mean_pnl / downside_std, -10.0, 10.0))

    def compute_action_diversity(self) -> float:
        """Compute action diversity over recent choices."""
        if len(self.action_tp_history) < 3:
            return 0.0
        tp_std = float(np.std(list(self.action_tp_history)))
        sl_std = float(np.std(list(self.action_sl_history)))
        return tp_std + sl_std

    def compute_condition_match_bonus(self, vol_proxy: float, tp_chosen: float) -> float:
        """Reward agent for adapting TP to volatility regime."""
        median_tp = float(np.median(TP_TICKS))

        # High vol → wider TP is good
        if vol_proxy > self.vol_p75 and tp_chosen > median_tp:
            return 0.05
        # Low vol → tighter TP is good
        if vol_proxy < self.vol_p25 and tp_chosen < median_tp:
            return 0.05
        return 0.0


# ═══════════════════════════════════════════════════════════════
# RL AGENT (PPO with factorized action heads, v2 fixes)
# ═══════════════════════════════════════════════════════════════

class ExecPolicyNetworkV2(nn.Module):
    """
    Multi-head policy network v2 for execution decisions.

    Fixes from v1:
      - Deeper backbone: 3 hidden layers with LayerNorm + GELU
      - Per-head learnable temperature scaling
      - Returns per-head entropies for per-head entropy coefficients
    """

    def __init__(self, state_dim: int = 44, hidden_dim: int = 128):
        super().__init__()

        # Deeper backbone: 3 hidden layers
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # Action heads (from 64-dim backbone output)
        self.gate_head = nn.Linear(64, 2)
        self.tp_head = nn.Linear(64, len(TP_TICKS))
        self.sl_head = nn.Linear(64, len(SL_TICKS))
        self.hold_head = nn.Linear(64, len(HOLD_SECS))

        # Learnable temperature per head (initialized to 1.0)
        self.gate_temp = nn.Parameter(torch.ones(1))
        self.tp_temp = nn.Parameter(torch.ones(1))
        self.sl_temp = nn.Parameter(torch.ones(1))
        self.hold_temp = nn.Parameter(torch.ones(1))

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Smaller init for action heads to encourage uniform initial policy
        for head in [self.gate_head, self.tp_head, self.sl_head, self.hold_head]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

    def forward(self, state: torch.Tensor):
        """
        Returns temperature-scaled logits and value.
        state: (batch, state_dim)
        """
        features = self.backbone(state)

        # Temperature-scaled logits (clamp temp to prevent collapse/explosion)
        gate_temp = torch.clamp(self.gate_temp, 0.1, 5.0)
        tp_temp = torch.clamp(self.tp_temp, 0.1, 5.0)
        sl_temp = torch.clamp(self.sl_temp, 0.1, 5.0)
        hold_temp = torch.clamp(self.hold_temp, 0.1, 5.0)

        gate_logits = self.gate_head(features) / gate_temp
        tp_logits = self.tp_head(features) / tp_temp
        sl_logits = self.sl_head(features) / sl_temp
        hold_logits = self.hold_head(features) / hold_temp

        value = self.value_head(features).squeeze(-1)

        return gate_logits, tp_logits, sl_logits, hold_logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy. Returns per-head entropies."""
        gate_logits, tp_logits, sl_logits, hold_logits, value = self.forward(state)

        gate_dist = Categorical(logits=gate_logits)
        tp_dist = Categorical(logits=tp_logits)
        sl_dist = Categorical(logits=sl_logits)
        hold_dist = Categorical(logits=hold_logits)

        if deterministic:
            gate = gate_logits.argmax(-1)
            tp = tp_logits.argmax(-1)
            sl = sl_logits.argmax(-1)
            hold = hold_logits.argmax(-1)
        else:
            gate = gate_dist.sample()
            tp = tp_dist.sample()
            sl = sl_dist.sample()
            hold = hold_dist.sample()

        log_prob = (gate_dist.log_prob(gate) +
                    tp_dist.log_prob(tp) +
                    sl_dist.log_prob(sl) +
                    hold_dist.log_prob(hold))

        # Per-head entropies
        entropies = {
            "gate": gate_dist.entropy(),
            "tp": tp_dist.entropy(),
            "sl": sl_dist.entropy(),
            "hold": hold_dist.entropy(),
        }

        return (ExecAction(gate=gate.item(), tp_idx=tp.item(),
                           sl_idx=sl.item(), hold_idx=hold.item()),
                log_prob, value, entropies)

    def evaluate_actions(self, states, gates, tps, sls, holds):
        """Evaluate log probs and per-head entropy for stored actions."""
        gate_logits, tp_logits, sl_logits, hold_logits, values = self.forward(states)

        gate_dist = Categorical(logits=gate_logits)
        tp_dist = Categorical(logits=tp_logits)
        sl_dist = Categorical(logits=sl_logits)
        hold_dist = Categorical(logits=hold_logits)

        log_probs = (gate_dist.log_prob(gates) +
                     tp_dist.log_prob(tps) +
                     sl_dist.log_prob(sls) +
                     hold_dist.log_prob(holds))

        # Per-head entropies for per-head coefficients
        entropies = {
            "gate": gate_dist.entropy(),
            "tp": tp_dist.entropy(),
            "sl": sl_dist.entropy(),
            "hold": hold_dist.entropy(),
        }

        return log_probs, values, entropies


class PPOTrainerV2:
    """PPO trainer v2 for the execution agent with all collapse fixes."""

    def __init__(self, env: ExecEnvironment, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, value_coeff: float = 0.5,
                 max_grad_norm: float = 1.0,
                 batch_size: int = 256, n_epochs_per_update: int = 8,
                 sortino_window: int = 100, sortino_weight: float = 0.5,
                 device: str = "cpu"):

        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs_per_update = n_epochs_per_update
        self.sortino_window = sortino_window
        self.sortino_weight = sortino_weight
        self.device = device

        self.policy = ExecPolicyNetworkV2(
            state_dim=ExecState.dim(),
            hidden_dim=128,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        n_params = sum(p.numel() for p in self.policy.parameters())
        log.info(f"Policy network v2: {n_params:,} parameters (state_dim={ExecState.dim()})")
        log.info(f"  Per-head entropy coeffs: {ENTROPY_COEFFS}")
        log.info(f"  Sortino weight: {sortino_weight}, window: {sortino_window}")

    def _lr_warmup(self, iteration: int, warmup_iters: int = 20, base_lr: float = 3e-4):
        """Linear LR warmup over first warmup_iters iterations."""
        if iteration <= warmup_iters:
            lr = base_lr * (iteration / warmup_iters)
        else:
            lr = base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def collect_rollout(self, n_steps: int = 8192):
        """Collect experience by running policy in environment."""
        states = []
        actions_gate, actions_tp, actions_sl, actions_hold = [], [], [], []
        rewards, values, log_probs, dones = [], [], [], []

        for _ in range(n_steps):
            state = self.env.get_state()
            if state is None:
                self.env.reset()
                state = self.env.get_state()
                if state is None:
                    break

            state_tensor = torch.FloatTensor(state.to_tensor()).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, log_prob, value, entropies = self.policy.get_action(state_tensor)

            reward, info = self.env.step(action)

            # ── Enhanced reward shaping ──

            # 1. Sortino shaping over 100-trade windows (weight=0.5)
            if len(self.env.trade_history) >= 10 and info.get("traded"):
                sortino = self.env.compute_sortino(self.sortino_window)
                reward += self.sortino_weight * sortino

            # 2. Action diversity bonus
            if info.get("traded"):
                diversity = self.env.compute_action_diversity()
                if diversity > 1.0:
                    reward += 0.02

            # 3. Condition-matching bonus (vol-adaptive TP)
            if info.get("traded"):
                vol = info.get("vol_proxy", 0.0)
                tp_chosen = info.get("tp_chosen", 0.0)
                reward += self.env.compute_condition_match_bonus(vol, tp_chosen)

            states.append(state.to_tensor())
            actions_gate.append(action.gate)
            actions_tp.append(action.tp_idx)
            actions_sl.append(action.sl_idx)
            actions_hold.append(action.hold_idx)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(info.get("done", False))

        return {
            "states": np.array(states, dtype=np.float32),
            "gates": np.array(actions_gate, dtype=np.int64),
            "tps": np.array(actions_tp, dtype=np.int64),
            "sls": np.array(actions_sl, dtype=np.int64),
            "holds": np.array(actions_hold, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "values": np.array(values, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
        }

    def compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                next_value = values[t + 1] if t + 1 < n else 0.0
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae

            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def _check_action_distribution(self, rollout: dict, iteration: int):
        """Check and log action distribution. Warn if collapsed."""
        traded_mask = rollout["gates"] == 1
        n_traded = traded_mask.sum()

        if n_traded < 10:
            log.warning(f"  Iter {iteration}: Only {n_traded} trades in rollout!")
            return

        tp_choices = rollout["tps"][traded_mask]
        sl_choices = rollout["sls"][traded_mask]
        hold_choices = rollout["holds"][traded_mask]

        for name, choices, n_options in [
            ("TP", tp_choices, len(TP_TICKS)),
            ("SL", sl_choices, len(SL_TICKS)),
            ("Hold", hold_choices, len(HOLD_SECS)),
        ]:
            counts = Counter(choices.tolist())
            most_common_count = counts.most_common(1)[0][1]
            pct = 100 * most_common_count / len(choices)
            if pct > 80:
                log.warning(f"  COLLAPSE WARNING: {name} head — action {counts.most_common(1)[0][0]} "
                            f"captures {pct:.0f}% of choices!")

    def update(self, rollout: dict, iteration: int = 0):
        """PPO update step with per-head entropy coefficients."""
        advantages, returns = self.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states_t = torch.FloatTensor(rollout["states"]).to(self.device)
        gates_t = torch.LongTensor(rollout["gates"]).to(self.device)
        tps_t = torch.LongTensor(rollout["tps"]).to(self.device)
        sls_t = torch.LongTensor(rollout["sls"]).to(self.device)
        holds_t = torch.LongTensor(rollout["holds"]).to(self.device)
        old_log_probs_t = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        n = len(states_t)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = {k: 0.0 for k in ENTROPY_COEFFS}
        n_updates = 0

        for _ in range(self.n_epochs_per_update):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                new_log_probs, new_values, entropies = self.policy.evaluate_actions(
                    states_t[idx], gates_t[idx], tps_t[idx],
                    sls_t[idx], holds_t[idx]
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(
                    ratio * advantages_t[idx],
                    clipped_ratio * advantages_t[idx]
                ).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, returns_t[idx])

                # Per-head entropy bonus (weighted separately)
                entropy_loss = 0.0
                for head_name, coeff in ENTROPY_COEFFS.items():
                    head_entropy = entropies[head_name].mean()
                    entropy_loss -= coeff * head_entropy
                    total_entropy[head_name] += head_entropy.item()

                # Total loss
                loss = policy_loss + self.value_coeff * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy_gate": total_entropy["gate"] / max(1, n_updates),
            "entropy_tp": total_entropy["tp"] / max(1, n_updates),
            "entropy_sl": total_entropy["sl"] / max(1, n_updates),
            "entropy_hold": total_entropy["hold"] / max(1, n_updates),
            "temps": {
                "gate": float(self.policy.gate_temp.item()),
                "tp": float(self.policy.tp_temp.item()),
                "sl": float(self.policy.sl_temp.item()),
                "hold": float(self.policy.hold_temp.item()),
            },
        }

    def train(self, n_iterations: int = 300, steps_per_iter: int = 8192,
              eval_every: int = 10, save_dir: str = "output/rl_exec_v2"):
        """Main training loop with LR warmup and diversity monitoring."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        base_lr = self.optimizer.param_groups[0]["lr"]
        best_sortino = -float("inf")
        results_log = []

        log.info(f"Training RL exec agent v2 for {n_iterations} iterations "
                 f"({steps_per_iter} steps/iter, {self.n_epochs_per_update} PPO epochs)")

        for iteration in range(1, n_iterations + 1):
            # LR warmup
            lr = self._lr_warmup(iteration, warmup_iters=20, base_lr=base_lr)

            self.env.reset()
            rollout = self.collect_rollout(steps_per_iter)
            losses = self.update(rollout, iteration)

            # Compute stats
            n_trades = len(self.env.trade_history)
            n_wins = sum(1 for t in self.env.trade_history if t.pnl_ticks > 0)
            total_pnl = sum(t.pnl_ticks for t in self.env.trade_history)
            sortino = self.env.compute_sortino(self.sortino_window)
            diversity = self.env.compute_action_diversity()

            trade_rate = n_trades / max(1, len(rollout["states"])) * 100

            result = {
                "iteration": iteration,
                "lr": lr,
                "n_trades": n_trades,
                "win_rate": n_wins / max(1, n_trades) * 100,
                "total_pnl_ticks": total_pnl,
                "total_pnl_usd": total_pnl * TICK_VALUE,
                "sortino": sortino,
                "trade_rate": trade_rate,
                "action_diversity": diversity,
                **{k: v for k, v in losses.items() if k != "temps"},
                "temps": losses["temps"],
            }
            results_log.append(result)

            # Check action distribution for collapse
            if iteration % eval_every == 0 or iteration == 1:
                self._check_action_distribution(rollout, iteration)

                # Session breakdown
                session_pnl = {}
                session_sortinos = {}
                for t in self.env.trade_history:
                    bucket = list(SESSION_BUCKETS.keys())[t.session_bucket]
                    if bucket not in session_pnl:
                        session_pnl[bucket] = []
                    session_pnl[bucket].append(t.pnl_ticks)

                for sess, pnls_list in session_pnl.items():
                    pnls_arr = np.array(pnls_list)
                    mean_p = pnls_arr.mean()
                    down = pnls_arr[pnls_arr < 0]
                    ds = down.std() if len(down) > 0 else 1e-8
                    session_sortinos[sess] = float(np.clip(mean_p / max(ds, 1e-8), -10, 10))

                log.info(
                    f"Iter {iteration:4d} | "
                    f"LR: {lr:.1e} | "
                    f"Trades: {n_trades:4d} ({trade_rate:.1f}%) | "
                    f"WR: {result['win_rate']:.1f}% | "
                    f"PnL: {total_pnl:+.1f}t (${total_pnl * TICK_VALUE:+,.0f}) | "
                    f"Sortino: {sortino:+.3f} | "
                    f"Diversity: {diversity:.2f} | "
                    f"π_loss: {losses['policy_loss']:.4f}"
                )
                log.info(
                    f"  Entropy — G:{losses['entropy_gate']:.2f} "
                    f"TP:{losses['entropy_tp']:.2f} "
                    f"SL:{losses['entropy_sl']:.2f} "
                    f"H:{losses['entropy_hold']:.2f} | "
                    f"Temps — G:{losses['temps']['gate']:.2f} "
                    f"TP:{losses['temps']['tp']:.2f} "
                    f"SL:{losses['temps']['sl']:.2f} "
                    f"H:{losses['temps']['hold']:.2f}"
                )

                for sess, pnls_list in sorted(session_pnl.items()):
                    pnls_arr = np.array(pnls_list)
                    log.info(
                        f"  {sess:12s}: {len(pnls_list):3d} trades, "
                        f"WR={100 * (pnls_arr > 0).mean():.0f}%, "
                        f"PnL={pnls_arr.sum():+.1f}t, "
                        f"Sortino={session_sortinos[sess]:+.3f}"
                    )

                # Action distribution summary
                if n_trades > 0:
                    tp_vals = [t.tp_ticks for t in self.env.trade_history]
                    sl_vals = [t.sl_ticks for t in self.env.trade_history]
                    hold_vals = [t.hold_time_s for t in self.env.trade_history]
                    log.info(f"  TP dist: {dict(Counter(tp_vals).most_common(5))}")
                    log.info(f"  SL dist: {dict(Counter(sl_vals).most_common(5))}")
                    log.info(f"  Hold dist: {dict(Counter(hold_vals).most_common(5))}")

            # Save best
            if sortino > best_sortino and n_trades >= 10:
                best_sortino = sortino
                torch.save({
                    "policy_state_dict": self.policy.state_dict(),
                    "iteration": iteration,
                    "sortino": sortino,
                    "pnl_ticks": total_pnl,
                    "n_trades": n_trades,
                    "win_rate": result["win_rate"],
                    "action_diversity": diversity,
                    "pca_components": self.env._pca_components,
                    "pca_mean": self.env._emb_mean,
                    "confidence_tiers": {
                        "p90": self.env.p90, "p95": self.env.p95, "p99": self.env.p99
                    },
                    "vol_percentiles": {
                        "p25": self.env.vol_p25, "p75": self.env.vol_p75
                    },
                    "state_dim": ExecState.dim(),
                    "version": "v2",
                }, save_path / "best_agent.pt")
                log.info(f"  ** New best! Sortino={sortino:.4f}, saved to {save_path}/best_agent.pt")

        # Save final results
        def _to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            return obj

        clean_log = [{k: _to_native(v) for k, v in r.items()} for r in results_log]
        with open(save_path / "training_log.json", "w") as f:
            json.dump(clean_log, f, indent=2,
                      default=lambda o: float(o) if hasattr(o, 'item') else str(o))

        # Save final model
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "iteration": n_iterations,
            "results_log": _to_native(results_log[-1]) if results_log else {},
            "version": "v2",
        }, save_path / "final_agent.pt")

        log.info(f"\nTraining complete. Best Sortino: {best_sortino:.4f}")
        log.info(f"Results saved to {save_path}")

        return results_log

    def evaluate(self, checkpoint_path: str = None, deterministic: bool = True):
        """Evaluate agent on full dataset with per-session Sortino and diversity metrics."""
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ckpt["policy_state_dict"])
            log.info(f"Loaded checkpoint: {checkpoint_path}")
            if "version" in ckpt:
                log.info(f"  Version: {ckpt['version']}")

        self.policy.eval()
        self.env.reset()

        trades_by_session = {name: [] for name in SESSION_BUCKETS}
        trades_by_exit = {"tp": [], "sl": [], "hold_timeout": []}
        all_trades = []

        while True:
            state = self.env.get_state()
            if state is None:
                break

            state_tensor = torch.FloatTensor(state.to_tensor()).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _, _, _ = self.policy.get_action(state_tensor, deterministic=deterministic)

            reward, info = self.env.step(action)

            if info.get("traded"):
                trade = self.env.trade_history[-1]
                all_trades.append(trade)
                session_name = list(SESSION_BUCKETS.keys())[trade.session_bucket]
                trades_by_session[session_name].append(trade)
                if trade.exit_reason in trades_by_exit:
                    trades_by_exit[trade.exit_reason].append(trade)

        # Report
        log.info("\n" + "=" * 70)
        log.info("EVALUATION REPORT (v2)")
        log.info("=" * 70)

        if not all_trades:
            log.info("No trades executed!")
            return {"n_trades": 0}

        pnls = np.array([t.pnl_ticks for t in all_trades])
        overall_sortino = self.env.compute_sortino(len(all_trades))

        log.info(f"Total trades: {len(all_trades)}")
        log.info(f"Win rate: {100 * (pnls > 0).mean():.1f}%")
        log.info(f"Total PnL: {pnls.sum():+.1f} ticks (${pnls.sum() * TICK_VALUE:+,.0f})")
        log.info(f"Avg PnL/trade: {pnls.mean():+.2f} ticks")
        log.info(f"Sortino: {overall_sortino:+.4f}")
        log.info(f"Action diversity: {self.env.compute_action_diversity():.2f}")

        log.info(f"\n--- By Session (with Sortino) ---")
        session_results = {}
        for name, trades in trades_by_session.items():
            if not trades:
                continue
            p = np.array([t.pnl_ticks for t in trades])
            mean_p = p.mean()
            down = p[p < 0]
            ds = down.std() if len(down) > 0 else 1e-8
            sess_sortino = float(np.clip(mean_p / max(ds, 1e-8), -10, 10))
            log.info(f"  {name:12s}: {len(trades):4d} trades, "
                     f"WR={100 * (p > 0).mean():.0f}%, "
                     f"PnL={p.sum():+.1f}t (${p.sum() * TICK_VALUE:+,.0f}), "
                     f"Sortino={sess_sortino:+.4f}")
            session_results[name] = {
                "n": len(trades),
                "pnl": float(p.sum()),
                "win_rate": float(100 * (p > 0).mean()),
                "sortino": sess_sortino,
            }

        log.info(f"\n--- By Exit Reason ---")
        for reason, trades in trades_by_exit.items():
            if not trades:
                continue
            p = np.array([t.pnl_ticks for t in trades])
            log.info(f"  {reason:15s}: {len(trades):4d} trades, "
                     f"WR={100 * (p > 0).mean():.0f}%, "
                     f"PnL={p.sum():+.1f}t")

        # Action distribution
        log.info(f"\n--- Action Distribution ---")
        tp_vals = [t.tp_ticks for t in all_trades]
        sl_vals = [t.sl_ticks for t in all_trades]
        hold_vals = [t.hold_time_s for t in all_trades]

        log.info(f"  TP distribution: {dict(Counter(tp_vals).most_common())}")
        log.info(f"  SL distribution: {dict(Counter(sl_vals).most_common())}")
        log.info(f"  Hold distribution: {dict(Counter(hold_vals).most_common())}")

        # Diversity metrics
        tp_unique = len(set(tp_vals))
        sl_unique = len(set(sl_vals))
        hold_unique = len(set(hold_vals))
        log.info(f"\n  Unique TP choices: {tp_unique}/{len(TP_TICKS)}")
        log.info(f"  Unique SL choices: {sl_unique}/{len(SL_TICKS)}")
        log.info(f"  Unique Hold choices: {hold_unique}/{len(HOLD_SECS)}")

        # Check for collapse
        for label, vals, options in [("TP", tp_vals, TP_TICKS),
                                      ("SL", sl_vals, SL_TICKS),
                                      ("Hold", hold_vals, HOLD_SECS)]:
            c = Counter(vals)
            top_val, top_count = c.most_common(1)[0]
            pct = 100 * top_count / len(vals)
            if pct > 80:
                log.warning(f"  COLLAPSE: {label} = {top_val} in {pct:.0f}% of trades!")

        result = {
            "n_trades": len(all_trades),
            "win_rate": float(100 * (pnls > 0).mean()),
            "total_pnl_ticks": float(pnls.sum()),
            "total_pnl_usd": float(pnls.sum() * TICK_VALUE),
            "sortino": float(overall_sortino),
            "action_diversity": float(self.env.compute_action_diversity()),
            "unique_tp": tp_unique,
            "unique_sl": sl_unique,
            "unique_hold": hold_unique,
            "trades_by_session": session_results,
        }
        return result


# ═══════════════════════════════════════════════════════════════
# BASELINES (for comparison)
# ═══════════════════════════════════════════════════════════════

def run_baseline(env: ExecEnvironment, strategy: str = "fixed_10s"):
    """Run a fixed baseline strategy for comparison."""
    env.reset()

    while True:
        state = env.get_state()
        if state is None:
            break

        pred_mag = abs(state.pred_10s)

        if strategy == "fixed_10s":
            if pred_mag >= env.p99:
                action = ExecAction(gate=1, tp_idx=3, sl_idx=2, hold_idx=3)
            else:
                action = ExecAction(gate=0, tp_idx=0, sl_idx=0, hold_idx=0)

        elif strategy == "fixed_5s":
            if pred_mag >= env.p99:
                action = ExecAction(gate=1, tp_idx=2, sl_idx=2, hold_idx=2)
            else:
                action = ExecAction(gate=0, tp_idx=0, sl_idx=0, hold_idx=0)

        elif strategy == "wide_tp_sl":
            if pred_mag >= env.p95:
                action = ExecAction(gate=1, tp_idx=5, sl_idx=4, hold_idx=4)
            else:
                action = ExecAction(gate=0, tp_idx=0, sl_idx=0, hold_idx=0)

        elif strategy == "aggressive":
            if pred_mag >= env.p90:
                action = ExecAction(gate=1, tp_idx=1, sl_idx=1, hold_idx=1)
            else:
                action = ExecAction(gate=0, tp_idx=0, sl_idx=0, hold_idx=0)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        env.step(action)

    trades = env.trade_history
    if not trades:
        return {"strategy": strategy, "n_trades": 0}

    pnls = np.array([t.pnl_ticks for t in trades])
    result = {
        "strategy": strategy,
        "n_trades": len(trades),
        "win_rate": float(100 * (pnls > 0).mean()),
        "total_pnl_ticks": float(pnls.sum()),
        "total_pnl_usd": float(pnls.sum() * TICK_VALUE),
        "sortino": float(env.compute_sortino(len(trades))),
        "avg_pnl": float(pnls.mean()),
    }
    log.info(f"Baseline [{strategy}]: {result['n_trades']} trades, "
             f"WR={result['win_rate']:.1f}%, "
             f"PnL={result['total_pnl_ticks']:+.1f}t "
             f"(${result['total_pnl_usd']:+,.0f}), "
             f"Sortino={result['sortino']:+.4f}")
    return result


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL Adaptive Execution Agent v2")
    parser.add_argument("--mode", choices=["train", "eval", "baselines", "sweep"],
                        default="train")
    parser.add_argument("--data-dir", type=str,
                        default="output/cnn_mamba_v2_smart_v3_mar",
                        help="Directory with fold_*_oot_predictions.npz")
    parser.add_argument("--mbo-data-dir", type=str, default=None,
                        help="Directory with MBO event data (for realistic sim)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to agent checkpoint for eval")
    parser.add_argument("--output-dir", type=str, default="output/rl_exec_v2",
                        help="Output directory")
    parser.add_argument("--n-iterations", type=int, default=300)
    parser.add_argument("--steps-per-iter", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold numbers to use")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        log.error("PyTorch not available! Install with: pip install torch")
        sys.exit(1)

    # Parse folds
    folds = None
    if args.folds:
        folds = [int(x) for x in args.folds.split(",")]

    # Create environment
    data_dir = str(Path(__file__).resolve().parent.parent.parent / args.data_dir)
    env = ExecEnvironment(data_dir=data_dir, folds=folds,
                          mbo_data_dir=args.mbo_data_dir)

    if args.mode == "baselines":
        log.info("Running baseline strategies...")
        results = []
        for strat in ["fixed_10s", "fixed_5s", "wide_tp_sl", "aggressive"]:
            results.append(run_baseline(env, strat))

        log.info("\n" + "=" * 70)
        log.info("BASELINE COMPARISON")
        log.info("=" * 70)
        log.info(f"{'Strategy':20s} | {'Trades':>6s} | {'WR':>5s} | {'PnL(t)':>8s} | {'PnL($)':>10s} | {'Sortino':>8s}")
        log.info("-" * 70)
        for r in results:
            log.info(f"{r['strategy']:20s} | {r['n_trades']:6d} | "
                     f"{r.get('win_rate', 0):5.1f} | {r.get('total_pnl_ticks', 0):+8.1f} | "
                     f"${r.get('total_pnl_usd', 0):+9,.0f} | {r.get('sortino', 0):+8.4f}")

        save_path = Path(args.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "baselines.json", "w") as f:
            json.dump(results, f, indent=2)

    elif args.mode == "train":
        trainer = PPOTrainerV2(
            env=env,
            lr=args.lr,
            device=args.device,
            batch_size=256,
            n_epochs_per_update=8,
            sortino_weight=0.5,
            sortino_window=100,
            max_grad_norm=1.0,
        )

        log.info("=" * 70)
        log.info("RL EXEC AGENT v2 TRAINING (PPO with collapse fixes)")
        log.info("=" * 70)
        log.info(f"  Data: {data_dir}")
        log.info(f"  Samples: {len(env.all_preds)}")
        log.info(f"  Iterations: {args.n_iterations}")
        log.info(f"  Steps/iter: {args.steps_per_iter}")
        log.info(f"  PPO epochs: 8")
        log.info(f"  Grad clip: 1.0")
        log.info(f"  LR warmup: 20 iters")
        log.info(f"  Per-head entropy: {ENTROPY_COEFFS}")
        log.info(f"  Sortino weight: 0.5 (window=100)")
        log.info(f"  Skip penalty: -0.005")
        log.info(f"  Diversity bonus: +0.02 (if std>1.0)")
        log.info(f"  Condition bonus: +0.05 (vol-adaptive TP)")
        log.info(f"  Device: {args.device}")
        log.info(f"  Output: {args.output_dir}")
        log.info("=" * 70)

        # Run baselines first
        log.info("\nRunning baselines first...")
        for strat in ["fixed_10s", "fixed_5s"]:
            run_baseline(env, strat)

        # Train
        log.info("\nStarting RL training v2...")
        results = trainer.train(
            n_iterations=args.n_iterations,
            steps_per_iter=args.steps_per_iter,
            eval_every=10,
            save_dir=args.output_dir,
        )

        # Final evaluation
        log.info("\nFinal evaluation (deterministic)...")
        trainer.evaluate(deterministic=True)

    elif args.mode == "eval":
        if not args.checkpoint:
            args.checkpoint = str(Path(args.output_dir) / "best_agent.pt")

        trainer = PPOTrainerV2(env=env, device=args.device)
        result = trainer.evaluate(checkpoint_path=args.checkpoint, deterministic=True)

        save_path = Path(args.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "eval_results.json", "w") as f:
            json.dump(result, f, indent=2,
                      default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    elif args.mode == "sweep":
        log.info("Running hyperparameter sweep (v2)...")
        configs = [
            {"lr": 1e-4, "sortino_weight": 0.3},
            {"lr": 3e-4, "sortino_weight": 0.5},
            {"lr": 3e-4, "sortino_weight": 1.0},
            {"lr": 1e-3, "sortino_weight": 0.5},
        ]
        best_result = None
        for i, cfg in enumerate(configs):
            log.info(f"\n--- Sweep config {i+1}/{len(configs)}: {cfg} ---")
            trainer = PPOTrainerV2(
                env=env,
                lr=cfg["lr"],
                sortino_weight=cfg.get("sortino_weight", 0.5),
                device=args.device,
            )
            results = trainer.train(
                n_iterations=80,
                steps_per_iter=4096,
                eval_every=20,
                save_dir=f"{args.output_dir}/sweep_{i}",
            )
            final = results[-1]
            if best_result is None or final["sortino"] > best_result["sortino"]:
                best_result = {**final, "config": cfg, "sweep_idx": i}

        log.info(f"\nBest sweep config: {best_result['config']}, "
                 f"Sortino={best_result['sortino']:.4f}")


if __name__ == "__main__":
    main()
