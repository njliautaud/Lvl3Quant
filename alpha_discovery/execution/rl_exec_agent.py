"""
RL Adaptive Execution Agent (HC #37 / HC #38)
=============================================

Trains a PPO-based RL agent that makes execution decisions:
  - WHEN to enter (gate: trade or skip)
  - HOW MUCH TP/SL to set (adaptive, not static)
  - HOW LONG to hold (adaptive hold duration)
  - WHEN to exit (beyond signal-flip: vol exit, time exit, MAE exit)

Optimized directly for Sortino ratio over rolling windows.

State space:
  - Model predictions (1s/5s/10s) + confidence magnitude
  - 96-dim embedding from CNN Mamba v2
  - Time-of-day bucket (one-hot: 6 sessions)
  - Recent trade history (last 5 trades P&L, win streak, loss streak)
  - Market microstructure: spread, volume proxy, vol proxy

Action space (factorized):
  - Gate: {enter, skip} (2)
  - TP ticks: {2, 4, 6, 8, 10, 15, 20, 30} (8)
  - SL ticks: {2, 3, 5, 8, 10, 15, 20} (7)
  - Hold bucket: {1s, 3s, 5s, 10s, 30s, 60s} (6)

Reward: Per-trade P&L (ticks) with Sortino shaping over 50-trade windows.

Usage:
  python rl_exec_agent.py --mode train --data-dir output/cnn_mamba_v2_smart_v3_mar
  python rl_exec_agent.py --mode eval --checkpoint output/rl_exec/best_agent.pt
  python rl_exec_agent.py --mode sweep --data-dir output/cnn_mamba_v2_smart_v3_mar
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
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
log = logging.getLogger("rl_exec")

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
    exit_reason: str  # "tp", "sl", "hold_timeout", "signal_flip"
    session_bucket: int
    prediction_strength: float


@dataclass
class ExecState:
    """State vector for the RL agent."""
    # Model predictions (3 horizons)
    pred_1s: float = 0.0
    pred_5s: float = 0.0
    pred_10s: float = 0.0
    pred_magnitude: float = 0.0  # |pred| = confidence
    pred_direction: float = 0.0  # sign of pred

    # Embedding (compressed to 16-dim via PCA)
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))

    # Time of day (one-hot session bucket)
    session_onehot: np.ndarray = field(default_factory=lambda: np.zeros(N_SESSIONS, dtype=np.float32))

    # Trade history (last 5 trades)
    recent_pnl: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float32))
    win_streak: int = 0
    loss_streak: int = 0
    cumulative_pnl: float = 0.0

    # Market microstructure proxies
    spread_ticks: float = 1.0
    vol_proxy: float = 0.0  # std of recent labels
    volume_proxy: float = 0.0  # event rate proxy

    def to_tensor(self) -> np.ndarray:
        """Flatten to 1D numpy array."""
        return np.concatenate([
            np.array([self.pred_1s, self.pred_5s, self.pred_10s,
                       self.pred_magnitude, self.pred_direction], dtype=np.float32),
            self.embedding,  # 16-dim
            self.session_onehot,  # 6-dim
            self.recent_pnl,  # 5-dim
            np.array([self.win_streak, self.loss_streak,
                       self.cumulative_pnl,
                       self.spread_ticks, self.vol_proxy, self.volume_proxy],
                      dtype=np.float32),
        ])

    @staticmethod
    def dim() -> int:
        return 5 + 16 + N_SESSIONS + 5 + 6  # = 38


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


class ExecEnvironment:
    """
    Offline execution environment that replays OOT predictions
    and simulates trades with configurable TP/SL/hold.

    Uses MBO event data to simulate realistic price paths after entry.
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

        fold_files = sorted(self.data_dir.glob("fold_*_oot_predictions.npz"))
        for f in fold_files:
            fold_num = int(f.stem.split("_")[1])
            if folds and fold_num not in folds:
                continue

            d = np.load(f)
            self.predictions.append(d["predictions"])  # (N, 3)
            self.labels.append(d["labels"])  # (N, 3)
            self.embeddings.append(d["embeddings"])  # (N, 96)
            self.oot_files.append(str(d["oot_files"][0]) if "oot_files" in d else f"fold_{fold_num}")

        if not self.predictions:
            raise ValueError(f"No prediction files found in {data_dir}")

        # Concatenate all
        self.all_preds = np.concatenate(self.predictions, axis=0)
        self.all_labels = np.concatenate(self.labels, axis=0)
        self.all_embeddings = np.concatenate(self.embeddings, axis=0)

        # PCA on embeddings (96 -> 16)
        self._fit_pca()

        log.info(f"Loaded {len(self.all_preds)} samples from {len(self.predictions)} folds")
        log.info(f"  Prediction range: [{self.all_preds.min():.4f}, {self.all_preds.max():.4f}]")

        # State tracking
        self.cursor = 0
        self.trade_history: List[TradeResult] = []
        self.recent_pnl = deque(maxlen=5)
        self.win_streak = 0
        self.loss_streak = 0
        self.cumulative_pnl = 0.0

        # Compute prediction percentiles for confidence tiers
        pred_abs = np.abs(self.all_preds[:, 2])  # 10s horizon
        self.p90 = np.percentile(pred_abs, 90)
        self.p95 = np.percentile(pred_abs, 95)
        self.p99 = np.percentile(pred_abs, 99)
        log.info(f"  Confidence tiers: p90={self.p90:.4f}, p95={self.p95:.4f}, p99={self.p99:.4f}")

    def _fit_pca(self):
        """Simple PCA: 96-dim embeddings -> 16-dim."""
        from numpy.linalg import svd
        emb = self.all_embeddings
        self._emb_mean = emb.mean(axis=0)
        centered = emb - self._emb_mean
        # Truncated SVD
        U, S, Vt = svd(centered, full_matrices=False)
        self._pca_components = Vt[:16]  # (16, 96)
        log.info(f"  PCA: 96 -> 16 dim (explained var ratio top-16: {(S[:16]**2).sum() / (S**2).sum():.3f})")

    def _project_embedding(self, emb: np.ndarray) -> np.ndarray:
        """Project 96-dim embedding to 16-dim."""
        return ((emb - self._emb_mean) @ self._pca_components.T).astype(np.float32)

    def _estimate_session_bucket(self, sample_idx: int) -> int:
        """Estimate session bucket from sample position within the day.
        Rough heuristic: map sample index to time-of-day based on
        typical event distribution."""
        # Approximate: assume events spread ~18hrs (6PM-12PM next day)
        # This is a rough proxy — in production we'd use actual timestamps
        day_frac = (sample_idx % 30000) / 30000  # normalize within day
        hour_et = 18 + day_frac * 22  # 18:00 ET start, 22hr trading day
        if hour_et >= 24:
            hour_et -= 24

        for i, (name, (start, end)) in enumerate(SESSION_BUCKETS.items()):
            if start <= hour_et < end:
                return i
            if start > end:  # overnight wrap
                if hour_et >= start or hour_et < end:
                    return i
        return 0  # default overnight

    def reset(self):
        """Reset environment for new episode."""
        self.cursor = 0
        self.trade_history = []
        self.recent_pnl = deque(maxlen=5)
        self.win_streak = 0
        self.loss_streak = 0
        self.cumulative_pnl = 0.0

    def get_state(self) -> Optional[ExecState]:
        """Get current state (next prediction to decide on)."""
        if self.cursor >= len(self.all_preds):
            return None

        pred = self.all_preds[self.cursor]
        emb = self.all_embeddings[self.cursor]
        label = self.all_labels[self.cursor]

        # Session bucket
        session_idx = self._estimate_session_bucket(self.cursor)
        session_onehot = np.zeros(N_SESSIONS, dtype=np.float32)
        session_onehot[session_idx] = 1.0

        # Vol proxy: std of labels in local window
        window = slice(max(0, self.cursor - 100), self.cursor + 1)
        vol_proxy = float(np.nanstd(self.all_labels[window, 2]))  # 10s label std

        state = ExecState(
            pred_1s=float(pred[0]),
            pred_5s=float(pred[1]),
            pred_10s=float(pred[2]),
            pred_magnitude=float(np.abs(pred[2])),
            pred_direction=float(np.sign(pred[2])),
            embedding=self._project_embedding(emb),
            session_onehot=session_onehot,
            recent_pnl=np.array(list(self.recent_pnl) + [0.0] * (5 - len(self.recent_pnl)),
                                dtype=np.float32),
            win_streak=self.win_streak,
            loss_streak=self.loss_streak,
            cumulative_pnl=self.cumulative_pnl,
            spread_ticks=1.0,  # default for NQ
            vol_proxy=vol_proxy,
            volume_proxy=1.0,  # placeholder
        )
        return state

    def step(self, action: ExecAction) -> Tuple[float, dict]:
        """
        Execute action on current prediction, simulate trade, return reward.

        Uses label-based simulation:
          - If gate=skip: reward=0, advance cursor
          - If gate=enter: simulate trade using future labels as price proxy
        """
        if self.cursor >= len(self.all_preds):
            return 0.0, {"done": True}

        pred = self.all_preds[self.cursor]
        label = self.all_labels[self.cursor]
        info = {"done": False, "traded": False}

        if action.gate == 0:
            # Skip this signal
            self.cursor += 1
            return -0.001, info  # tiny penalty for skipping (encourage trading)

        # ENTER TRADE
        direction = np.sign(pred[2])  # trade direction from 10s prediction
        if direction == 0:
            direction = 1.0  # default long

        # Simulate using labels as future price move (in z-score units)
        # Convert label to ticks: rough mapping based on typical NQ vol
        # Labels are normalized returns, ~1 std ≈ 8-12 ticks for NQ
        LABEL_TO_TICKS = 10.0  # approximate scaling factor

        # Get future price path (use all 3 horizons as checkpoints)
        move_1s = float(label[0]) * LABEL_TO_TICKS * direction
        move_5s = float(label[1]) * LABEL_TO_TICKS * direction
        move_10s = float(label[2]) * LABEL_TO_TICKS * direction

        # Simulate TP/SL/hold
        tp = action.tp_ticks
        sl = action.sl_ticks
        hold = action.hold_secs

        # Check price path against TP/SL at each horizon checkpoint
        pnl_ticks = 0.0
        exit_reason = "hold_timeout"

        checkpoints = [(1.0, move_1s), (5.0, move_5s), (10.0, move_10s)]

        # Track max favorable and max adverse
        mfe = 0.0  # max favorable excursion
        mae = 0.0  # max adverse excursion

        for t, move in checkpoints:
            if t > hold:
                break

            mfe = max(mfe, move)
            mae = min(mae, move)

            # Check TP hit
            if move >= tp:
                pnl_ticks = tp
                exit_reason = "tp"
                break

            # Check SL hit
            if move <= -sl:
                pnl_ticks = -sl
                exit_reason = "sl"
                break
        else:
            # Hold timeout — use the move at hold time
            # Interpolate between available checkpoints
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
        costs = 2 * COMMISSION_PER_SIDE  # round trip
        net_pnl = pnl_ticks - costs

        # Record trade
        trade = TradeResult(
            entry_price_ticks=0,
            exit_price_ticks=pnl_ticks,
            pnl_ticks=net_pnl,
            hold_time_s=hold,
            tp_ticks=tp,
            sl_ticks=sl,
            exit_reason=exit_reason,
            session_bucket=self._estimate_session_bucket(self.cursor),
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

        self.cursor += 1
        return net_pnl, info

    def compute_sortino(self, window: int = 50) -> float:
        """Compute Sortino ratio over recent trades."""
        if len(self.trade_history) < 2:
            return 0.0

        recent = self.trade_history[-window:]
        pnls = np.array([t.pnl_ticks for t in recent])
        mean_pnl = pnls.mean()
        downside = pnls[pnls < 0]
        if len(downside) == 0:
            return 10.0  # cap
        downside_std = downside.std()
        if downside_std < 1e-8:
            return 10.0
        return float(mean_pnl / downside_std)


# ═══════════════════════════════════════════════════════════════
# RL AGENT (PPO with factorized action heads)
# ═══════════════════════════════════════════════════════════════

class ExecPolicyNetwork(nn.Module):
    """
    Multi-head policy network for execution decisions.
    Shared backbone -> 4 separate heads (gate, TP, SL, hold).
    """

    def __init__(self, state_dim: int = 38, hidden_dim: int = 128):
        super().__init__()

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Action heads
        self.gate_head = nn.Linear(hidden_dim, 2)       # skip/enter
        self.tp_head = nn.Linear(hidden_dim, len(TP_TICKS))    # 8
        self.sl_head = nn.Linear(hidden_dim, len(SL_TICKS))    # 7
        self.hold_head = nn.Linear(hidden_dim, len(HOLD_SECS))  # 6

        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor):
        """
        Returns action logits and value estimate.
        state: (batch, state_dim)
        """
        features = self.backbone(state)

        gate_logits = self.gate_head(features)
        tp_logits = self.tp_head(features)
        sl_logits = self.sl_head(features)
        hold_logits = self.hold_head(features)

        value = self.value_head(features).squeeze(-1)

        return gate_logits, tp_logits, sl_logits, hold_logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy."""
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

        entropy = (gate_dist.entropy() +
                   tp_dist.entropy() +
                   sl_dist.entropy() +
                   hold_dist.entropy())

        return (ExecAction(gate=gate.item(), tp_idx=tp.item(),
                           sl_idx=sl.item(), hold_idx=hold.item()),
                log_prob, value, entropy)

    def evaluate_actions(self, states, gates, tps, sls, holds):
        """Evaluate log probs and entropy for stored actions."""
        gate_logits, tp_logits, sl_logits, hold_logits, values = self.forward(states)

        gate_dist = Categorical(logits=gate_logits)
        tp_dist = Categorical(logits=tp_logits)
        sl_dist = Categorical(logits=sl_logits)
        hold_dist = Categorical(logits=hold_logits)

        log_probs = (gate_dist.log_prob(gates) +
                     tp_dist.log_prob(tps) +
                     sl_dist.log_prob(sls) +
                     hold_dist.log_prob(holds))

        entropy = (gate_dist.entropy() +
                   tp_dist.entropy() +
                   sl_dist.entropy() +
                   hold_dist.entropy())

        return log_probs, values, entropy


class PPOTrainer:
    """PPO trainer for the execution agent."""

    def __init__(self, env: ExecEnvironment, lr: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_eps: float = 0.2, entropy_coeff: float = 0.05,
                 value_coeff: float = 0.5, max_grad_norm: float = 0.5,
                 batch_size: int = 256, n_epochs_per_update: int = 4,
                 sortino_window: int = 50, device: str = "cpu"):

        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs_per_update = n_epochs_per_update
        self.sortino_window = sortino_window
        self.device = device

        self.policy = ExecPolicyNetwork(
            state_dim=ExecState.dim(),
            hidden_dim=128,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        n_params = sum(p.numel() for p in self.policy.parameters())
        log.info(f"Policy network: {n_params:,} parameters")

    def collect_rollout(self, n_steps: int = 2048):
        """Collect experience by running policy in environment."""
        states, actions_gate, actions_tp, actions_sl, actions_hold = [], [], [], [], []
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
                action, log_prob, value, entropy = self.policy.get_action(state_tensor)

            reward, info = self.env.step(action)

            # Sortino shaping: bonus/penalty based on rolling Sortino
            if len(self.env.trade_history) >= 10 and info.get("traded"):
                sortino = self.env.compute_sortino(self.sortino_window)
                reward += 0.1 * sortino  # small Sortino bonus

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
        last_value = 0.0

        for t in reversed(range(n)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                next_value = values[t + 1] if t + 1 < n else last_value
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae

            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: dict):
        """PPO update step."""
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
        total_entropy = 0

        for _ in range(self.n_epochs_per_update):
            # Mini-batch updates
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = indices[start:end]

                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
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

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss +
                        self.value_coeff * value_loss +
                        self.entropy_coeff * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()

        n_updates = self.n_epochs_per_update * math.ceil(n / self.batch_size)
        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def train(self, n_iterations: int = 100, steps_per_iter: int = 2048,
              eval_every: int = 10, save_dir: str = "output/rl_exec"):
        """Main training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_sortino = -float("inf")
        results_log = []

        log.info(f"Training RL exec agent for {n_iterations} iterations "
                 f"({steps_per_iter} steps/iter)")

        for iteration in range(1, n_iterations + 1):
            self.env.reset()
            rollout = self.collect_rollout(steps_per_iter)
            losses = self.update(rollout)

            # Compute stats
            n_trades = sum(1 for t in self.env.trade_history)
            n_wins = sum(1 for t in self.env.trade_history if t.pnl_ticks > 0)
            total_pnl = sum(t.pnl_ticks for t in self.env.trade_history)
            sortino = self.env.compute_sortino(100)

            trade_rate = n_trades / max(1, len(rollout["states"])) * 100

            result = {
                "iteration": iteration,
                "n_trades": n_trades,
                "win_rate": n_wins / max(1, n_trades) * 100,
                "total_pnl_ticks": total_pnl,
                "total_pnl_usd": total_pnl * TICK_VALUE,
                "sortino": sortino,
                "trade_rate": trade_rate,
                **losses,
            }
            results_log.append(result)

            # Session breakdown
            session_pnl = {}
            for t in self.env.trade_history:
                bucket = list(SESSION_BUCKETS.keys())[t.session_bucket]
                if bucket not in session_pnl:
                    session_pnl[bucket] = []
                session_pnl[bucket].append(t.pnl_ticks)

            if iteration % eval_every == 0 or iteration == 1:
                log.info(
                    f"Iter {iteration:4d} | "
                    f"Trades: {n_trades:4d} ({trade_rate:.1f}%) | "
                    f"WR: {result['win_rate']:.1f}% | "
                    f"PnL: {total_pnl:+.1f}t (${total_pnl * TICK_VALUE:+,.0f}) | "
                    f"Sortino: {sortino:+.3f} | "
                    f"π_loss: {losses['policy_loss']:.4f} | "
                    f"H: {losses['entropy']:.2f}"
                )

                # Session breakdown
                for sess, pnls in sorted(session_pnl.items()):
                    pnls_arr = np.array(pnls)
                    log.info(
                        f"  {sess:12s}: {len(pnls):3d} trades, "
                        f"WR={100 * (pnls_arr > 0).mean():.0f}%, "
                        f"PnL={pnls_arr.sum():+.1f}t"
                    )

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
                    "pca_components": self.env._pca_components,
                    "pca_mean": self.env._emb_mean,
                    "confidence_tiers": {
                        "p90": self.env.p90, "p95": self.env.p95, "p99": self.env.p99
                    },
                }, save_path / "best_agent.pt")
                log.info(f"  ** New best! Sortino={sortino:.4f}, saved to {save_path}/best_agent.pt")

        # Save final results (convert numpy types to native Python)
        def _to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        clean_log = [{k: _to_native(v) for k, v in r.items()} for r in results_log]
        with open(save_path / "training_log.json", "w") as f:
            json.dump(clean_log, f, indent=2)

        # Save final model
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "iteration": n_iterations,
            "results_log": results_log[-1],
        }, save_path / "final_agent.pt")

        log.info(f"\nTraining complete. Best Sortino: {best_sortino:.4f}")
        log.info(f"Results saved to {save_path}")

        return results_log

    def evaluate(self, checkpoint_path: str = None, deterministic: bool = True):
        """Evaluate agent on full dataset."""
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ckpt["policy_state_dict"])
            log.info(f"Loaded checkpoint: {checkpoint_path}")

        self.policy.eval()
        self.env.reset()

        trades_by_session = {name: [] for name in SESSION_BUCKETS}
        trades_by_exit = {"tp": [], "sl": [], "hold_timeout": [], "signal_flip": []}
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
        log.info("EVALUATION REPORT")
        log.info("=" * 70)

        if not all_trades:
            log.info("No trades executed!")
            return

        pnls = np.array([t.pnl_ticks for t in all_trades])
        log.info(f"Total trades: {len(all_trades)}")
        log.info(f"Win rate: {100 * (pnls > 0).mean():.1f}%")
        log.info(f"Total PnL: {pnls.sum():+.1f} ticks (${pnls.sum() * TICK_VALUE:+,.0f})")
        log.info(f"Avg PnL/trade: {pnls.mean():+.2f} ticks")
        log.info(f"Sortino: {self.env.compute_sortino(len(all_trades)):+.4f}")

        log.info(f"\n--- By Session ---")
        for name, trades in trades_by_session.items():
            if not trades:
                continue
            p = np.array([t.pnl_ticks for t in trades])
            log.info(f"  {name:12s}: {len(trades):4d} trades, "
                     f"WR={100 * (p > 0).mean():.0f}%, "
                     f"PnL={p.sum():+.1f}t (${p.sum() * TICK_VALUE:+,.0f})")

        log.info(f"\n--- By Exit Reason ---")
        for reason, trades in trades_by_exit.items():
            if not trades:
                continue
            p = np.array([t.pnl_ticks for t in trades])
            log.info(f"  {reason:15s}: {len(trades):4d} trades, "
                     f"WR={100 * (p > 0).mean():.0f}%, "
                     f"PnL={p.sum():+.1f}t")

        # TP/SL distribution
        log.info(f"\n--- Action Distribution ---")
        tp_vals = [t.tp_ticks for t in all_trades]
        sl_vals = [t.sl_ticks for t in all_trades]
        hold_vals = [t.hold_time_s for t in all_trades]

        from collections import Counter
        log.info(f"  TP distribution: {dict(Counter(tp_vals).most_common(5))}")
        log.info(f"  SL distribution: {dict(Counter(sl_vals).most_common(5))}")
        log.info(f"  Hold distribution: {dict(Counter(hold_vals).most_common(5))}")

        return {
            "n_trades": len(all_trades),
            "win_rate": float(100 * (pnls > 0).mean()),
            "total_pnl_ticks": float(pnls.sum()),
            "total_pnl_usd": float(pnls.sum() * TICK_VALUE),
            "sortino": float(self.env.compute_sortino(len(all_trades))),
            "trades_by_session": {
                name: {"n": len(trades), "pnl": sum(t.pnl_ticks for t in trades)}
                for name, trades in trades_by_session.items() if trades
            },
        }


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
            # Always enter Top1%, fixed TP=8, SL=5, hold=10s
            if pred_mag >= env.p99:
                action = ExecAction(gate=1, tp_idx=3, sl_idx=2, hold_idx=3)  # TP=8, SL=5, hold=10s
            else:
                action = ExecAction(gate=0, tp_idx=0, sl_idx=0, hold_idx=0)

        elif strategy == "fixed_5s":
            if pred_mag >= env.p99:
                action = ExecAction(gate=1, tp_idx=2, sl_idx=2, hold_idx=2)  # TP=6, SL=5, hold=5s
            else:
                action = ExecAction(gate=0, tp_idx=0, sl_idx=0, hold_idx=0)

        elif strategy == "wide_tp_sl":
            if pred_mag >= env.p95:
                action = ExecAction(gate=1, tp_idx=5, sl_idx=4, hold_idx=4)  # TP=15, SL=10, hold=30s
            else:
                action = ExecAction(gate=0, tp_idx=0, sl_idx=0, hold_idx=0)

        elif strategy == "aggressive":
            if pred_mag >= env.p90:
                action = ExecAction(gate=1, tp_idx=1, sl_idx=1, hold_idx=1)  # TP=4, SL=3, hold=3s
            else:
                action = ExecAction(gate=0, tp_idx=0, sl_idx=0, hold_idx=0)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        env.step(action)

    # Report
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
    parser = argparse.ArgumentParser(description="RL Adaptive Execution Agent")
    parser.add_argument("--mode", choices=["train", "eval", "baselines", "sweep"],
                        default="train")
    parser.add_argument("--data-dir", type=str,
                        default="output/cnn_mamba_v2_smart_v3_mar",
                        help="Directory with fold_*_oot_predictions.npz")
    parser.add_argument("--mbo-data-dir", type=str, default=None,
                        help="Directory with MBO event data (for realistic sim)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to agent checkpoint for eval")
    parser.add_argument("--output-dir", type=str, default="output/rl_exec",
                        help="Output directory")
    parser.add_argument("--n-iterations", type=int, default=200)
    parser.add_argument("--steps-per-iter", type=int, default=4096)
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
        trainer = PPOTrainer(
            env=env,
            lr=args.lr,
            device=args.device,
            batch_size=256,
            n_epochs_per_update=4,
        )

        log.info("=" * 70)
        log.info("RL EXEC AGENT TRAINING (PPO)")
        log.info("=" * 70)
        log.info(f"  Data: {data_dir}")
        log.info(f"  Samples: {len(env.all_preds)}")
        log.info(f"  Iterations: {args.n_iterations}")
        log.info(f"  Steps/iter: {args.steps_per_iter}")
        log.info(f"  Device: {args.device}")
        log.info(f"  Output: {args.output_dir}")
        log.info("=" * 70)

        # First run baselines for comparison
        log.info("\nRunning baselines first...")
        for strat in ["fixed_10s", "fixed_5s"]:
            run_baseline(env, strat)

        # Train RL agent
        log.info("\nStarting RL training...")
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

        trainer = PPOTrainer(env=env, device=args.device)
        result = trainer.evaluate(checkpoint_path=args.checkpoint, deterministic=True)

        save_path = Path(args.output_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "eval_results.json", "w") as f:
            json.dump(result, f, indent=2, default=lambda o: float(o) if hasattr(o, 'item') else str(o))

    elif args.mode == "sweep":
        # Hyperparameter sweep
        log.info("Running hyperparameter sweep...")
        configs = [
            {"lr": 1e-4, "entropy_coeff": 0.01},
            {"lr": 3e-4, "entropy_coeff": 0.05},
            {"lr": 3e-4, "entropy_coeff": 0.10},
            {"lr": 1e-3, "entropy_coeff": 0.05},
        ]
        best_result = None
        for i, cfg in enumerate(configs):
            log.info(f"\n--- Sweep config {i+1}/{len(configs)}: {cfg} ---")
            trainer = PPOTrainer(
                env=env,
                lr=cfg["lr"],
                entropy_coeff=cfg.get("entropy_coeff", 0.05),
                device=args.device,
            )
            results = trainer.train(
                n_iterations=50,
                steps_per_iter=2048,
                eval_every=25,
                save_dir=f"{args.output_dir}/sweep_{i}",
            )
            final = results[-1]
            if best_result is None or final["sortino"] > best_result["sortino"]:
                best_result = {**final, "config": cfg, "sweep_idx": i}

        log.info(f"\nBest sweep config: {best_result['config']}, "
                 f"Sortino={best_result['sortino']:.4f}")


if __name__ == "__main__":
    main()
