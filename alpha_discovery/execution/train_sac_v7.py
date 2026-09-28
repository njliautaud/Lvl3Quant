#!/usr/bin/env python3
# WARNING: This uses a SIMPLIFIED environment, NOT true FIFO. See fifo_rl_env.py for production.
"""
SAC v7: Discrete Soft Actor-Critic for ES Execution
=====================================================
Off-policy RL with automatic entropy tuning, replay buffer, dual Q-networks.
Reuses v7 TradingEnv and data pipeline (same env, different algorithm).

KEY DIFFERENCES from PPO v7:
  - Off-policy: replay buffer stores transitions, reuses old experience
  - Dual Q-networks with target networks (reduces overestimation bias)
  - Automatic entropy tuning (alpha parameter auto-adjusts exploration)
  - No rollout buffer / GAE — uses 1-step TD targets from replay
  - Potentially more sample-efficient (reuses data multiple times)

Algorithm: Discrete SAC (Christodoulou 2019) — adapted for discrete action spaces.
  - Actor outputs action probabilities (softmax), not continuous actions
  - Q-networks estimate Q(s,a) for each discrete action
  - Entropy bonus encourages exploration of less-visited actions
  - Alpha (temperature) is automatically tuned to target entropy

HC compliance:
  - HC #0:  Sliding walk-forward (train=25d, eval=3d)
  - HC #69: Report Sharpe, Sortino, PF, WR (not raw P&L as primary)
  - HC #89: Costs = 0.376 ticks commission RT only. NO spread cost. P&L = exit - entry - 0.376t.
  - HC #124: Neptune = different RL style (SAC, not PPO)

Usage:
  python train_sac_v7.py --data-dir /path/to/mbo_events_smart_v3 \\
                         --pred-dir /path/to/cnn_mamba_v2_bulk_oot \\
                         --patchtst-dir /path/to/patchtst_bulk_oot \\
                         --output-dir ./sac_v7_output

Author: Claude (Infrastructure Builder)
Date: 2026-05-03
"""

from __future__ import annotations

import argparse
import copy
import gc
import math
import os
import sys
import time
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError:
    print("ERROR: PyTorch not found. pip install torch", file=sys.stderr)
    sys.exit(1)

try:
    import mlflow
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("sac_v7")

# ── Constants (HC canonical — identical to PPO v7) ────────────────────────────
COMMISSION_COST = 0.376   # ticks round-trip commission only (HC #89) — NO spread cost

DECISION_STRIDE = 50      # events per RL step (~5ms market time, closer to event-by-event)
WARMUP_EVENTS   = 10_000
RTH_START_UTC   = 13 * 3600 + 30 * 60
RTH_END_UTC     = 21 * 3600

TP_TICKS        = 6
SL_TICKS        = 6
MAX_HOLD_STEPS  = 100
MAX_WAIT_STEPS  = 3
OBS_DIM         = 20
N_ACTIONS       = 5

IDLE_PENALTY    = -0.002
EOD_NO_TRADE    = -2.0
EOD_TRADE_BONUS = +0.5
REWARD_CLIP     = 5.0
PRED_STRIDE     = 50


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING (identical to PPO v7)
# ══════════════════════════════════════════════════════════════════════════════

def find_valid_dates(pred_dir: str, patchtst_dir: str) -> List[str]:
    pred_path = Path(pred_dir)
    pst_path  = Path(patchtst_dir)

    cnn_dates = set()
    for f in pred_path.glob("*_predictions.npz"):
        stem = f.stem.replace("_predictions", "")
        if len(stem) == 8 and stem.isdigit():
            cnn_dates.add(stem)

    pst_dates = set()
    for f in pst_path.glob("*_predictions.npz"):
        stem = f.stem.replace("_predictions", "")
        if len(stem) == 8 and stem.isdigit():
            pst_dates.add(stem)

    return sorted(cnn_dates & pst_dates)


def load_day(date: str, data_dir: str, pred_dir: str, patchtst_dir: str) -> Optional[dict]:
    data_path    = Path(data_dir)
    cnn_path     = Path(pred_dir)
    pst_path_dir = Path(patchtst_dir)

    ev_file = data_path / f"{date}_mbo_events_smart_v3.npz"
    if not ev_file.exists():
        ev_file = data_path / f"{date}_mbo_events.npz"
    if not ev_file.exists():
        return None

    cnn_file = cnn_path / f"{date}_predictions.npz"
    pst_file = pst_path_dir / f"{date}_predictions.npz"
    if not cnn_file.exists() or not pst_file.exists():
        return None

    try:
        ev_data  = np.load(str(ev_file))
        cnn_data = np.load(str(cnn_file))
        pst_data = np.load(str(pst_file))

        events     = np.nan_to_num(ev_data["events"].astype(np.float32), nan=0.0)
        timestamps = ev_data["timestamps"].astype(np.int64)
        labels_1s  = np.nan_to_num(ev_data["labels_1s"].astype(np.float32), nan=0.0)
        labels_5s  = np.nan_to_num(ev_data["labels_5s"].astype(np.float32), nan=0.0)
        labels_10s = np.nan_to_num(ev_data["labels_10s"].astype(np.float32), nan=0.0)

        cnn_preds = cnn_data["predictions"].astype(np.float32)
        pst_preds = pst_data["predictions"].astype(np.float32)

        return dict(
            events=events, timestamps=timestamps,
            labels_1s=labels_1s, labels_5s=labels_5s, labels_10s=labels_10s,
            cnn_preds=cnn_preds, pst_preds=pst_preds,
        )
    except Exception as e:
        log.warning(f"Failed to load {date}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  TRADING ENVIRONMENT (identical to PPO v7)
# ══════════════════════════════════════════════════════════════════════════════

class TradingEnv:
    """Single-day MBO replay environment. One RL step = DECISION_STRIDE events."""

    def __init__(self, day_data: dict):
        self.events     = day_data["events"]
        self.timestamps = day_data["timestamps"]
        self.labels_10s = day_data["labels_10s"]
        self.cnn_preds  = day_data["cnn_preds"]
        self.pst_preds  = day_data["pst_preds"]
        self.N          = len(self.events)
        self.n_steps    = max(1, (self.N - WARMUP_EVENTS) // DECISION_STRIDE)
        self.M_preds    = len(self.cnn_preds)
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_idx         = 0
        self.event_cursor     = WARMUP_EVENTS
        self.in_position      = False
        self.direction        = 0
        self.entry_label_idx  = 0
        self.hold_steps       = 0
        self.max_favorable    = 0.0
        self.max_adverse      = 0.0
        self.pending_order    = False
        self.pending_dir      = 0
        self.pending_wait     = 0
        self.trade_pnls       = []
        self.trade_count      = 0
        self.win_count        = 0
        return self._get_obs()

    def _get_pred_idx(self, step: int) -> int:
        event_offset = step * DECISION_STRIDE
        pred_idx = event_offset // PRED_STRIDE
        return max(0, min(pred_idx, self.M_preds - 1))

    def _get_tod_normalized(self, event_idx: int) -> Tuple[float, float]:
        ts_ns  = self.timestamps[min(event_idx, self.N - 1)]
        ts_s   = ts_ns / 1e9
        tod_s  = ts_s % 86400
        tod_frac = (tod_s - RTH_START_UTC) / (RTH_END_UTC - RTH_START_UTC)
        tod_frac = max(0.0, min(1.0, tod_frac))
        return math.sin(math.pi * tod_frac), math.cos(math.pi * tod_frac)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        s   = self.step_idx
        cur = min(self.event_cursor, self.N - 1)

        p = self._get_pred_idx(s)
        cnn = self.cnn_preds[p]
        pst = self.pst_preds[p]
        obs[0:3] = cnn * 10.0
        obs[3:6] = pst * 10.0
        conf = np.mean(np.sign(cnn) * np.sign(pst))
        obs[6] = float(conf)

        lo = max(0, cur - DECISION_STRIDE)
        hi = cur
        window_events = self.events[lo:hi]
        if len(window_events) > 0:
            obs[7] = float(np.mean(window_events[:, 5]))
            if window_events.shape[1] > 7:
                bid_d = np.mean(window_events[:, 6])
                ask_d = np.mean(window_events[:, 7])
                denom = abs(bid_d) + abs(ask_d) + 1e-8
                obs[8] = float((bid_d - ask_d) / denom)

        if len(window_events) > 0:
            obs[9] = float(np.mean(self.labels_10s[lo:hi])) * 100.0

        obs[10] = 1.0 if self.in_position else 0.0
        obs[11] = float(self.direction)
        if self.in_position:
            entry_cur = self.entry_label_idx
            unrealized = float(np.sum(self.labels_10s[entry_cur:cur])) * self.direction
            obs[12] = unrealized / 10.0
        obs[13] = float(self.hold_steps) / 100.0

        obs[14] = self.max_favorable / 10.0
        obs[15] = self.max_adverse   / 10.0
        obs[16], obs[17] = self._get_tod_normalized(cur)

        if self.trade_count > 0:
            obs[18] = self.win_count / self.trade_count
        obs[19] = float(self.trade_count) / 50.0

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        reward = 0.0
        done   = False

        prev_cursor = self.event_cursor
        self.event_cursor = min(self.event_cursor + DECISION_STRIDE, self.N)
        cur = self.event_cursor

        if self.step_idx >= self.n_steps - 1:
            done = True
            if self.in_position:
                pnl = self._close_position(prev_cursor, cur, market=True)
                reward += pnl
            if self.trade_count == 0:
                reward += EOD_NO_TRADE
            elif self.trade_count >= 3:
                reward += EOD_TRADE_BONUS
            self.step_idx += 1
            return self._get_obs(), float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP)), done

        if self.pending_order:
            self.pending_wait += 1
            if self.pending_wait > MAX_WAIT_STEPS:
                self.pending_order = False
                self.pending_dir   = 0
                self.pending_wait  = 0
            else:
                window = self.events[prev_cursor:cur]
                trade_mask = (window[:, 1] < 0.1)
                n_trades   = int(np.sum(trade_mask))
                if n_trades > DECISION_STRIDE * 0.01:
                    self.in_position     = True
                    self.direction       = self.pending_dir
                    self.entry_label_idx = cur
                    self.hold_steps      = 0
                    self.max_favorable   = 0.0
                    self.max_adverse     = 0.0
                    self.pending_order   = False
                    self.pending_dir     = 0
                    self.pending_wait    = 0

        if self.in_position:
            self.hold_steps += 1
            cumulative_pnl = float(np.sum(
                self.labels_10s[self.entry_label_idx:cur]
            )) * self.direction

            if cumulative_pnl > self.max_favorable:
                self.max_favorable = cumulative_pnl
            if -cumulative_pnl > self.max_adverse:
                self.max_adverse = -cumulative_pnl

            if cumulative_pnl >= TP_TICKS or cumulative_pnl <= -SL_TICKS:
                pnl = self._close_position(prev_cursor, cur, market=True)
                reward += pnl
            elif self.hold_steps >= MAX_HOLD_STEPS:
                pnl = self._close_position(prev_cursor, cur, market=True)
                reward += pnl

        if not done and not self.in_position and not self.pending_order:
            if action == 1:
                self.pending_order = True
                self.pending_dir   = +1
                self.pending_wait  = 0
            elif action == 2:
                self.pending_order = True
                self.pending_dir   = -1
                self.pending_wait  = 0
            elif action == 0:
                reward += IDLE_PENALTY
        elif not done:
            if action == 3:
                if self.in_position:
                    pnl = self._close_position(prev_cursor, cur, market=True)
                    reward += pnl
                elif self.pending_order:
                    self.pending_order = False
                    self.pending_dir   = 0
                    self.pending_wait  = 0
            elif action == 4:
                if self.pending_order:
                    self.pending_order = False
                    self.pending_dir   = 0
                    self.pending_wait  = 0

        self.step_idx += 1
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        return self._get_obs(), reward, done

    def _close_position(self, ev_lo: int, ev_hi: int, market: bool = True) -> float:
        entry_idx = self.entry_label_idx
        cur       = ev_hi
        gross_pnl = float(np.nansum(self.labels_10s[entry_idx:cur])) * self.direction
        if not np.isfinite(gross_pnl):
            gross_pnl = 0.0
        # Cost = RT commission only (0.376 ticks). No spread cost.
        net_pnl = gross_pnl - COMMISSION_COST
        self.in_position   = False
        self.direction     = 0
        self.hold_steps    = 0
        self.max_favorable = 0.0
        self.max_adverse   = 0.0
        self.trade_count += 1
        self.trade_pnls.append(net_pnl)
        if net_pnl > 0:
            self.win_count += 1
        return net_pnl


# ══════════════════════════════════════════════════════════════════════════════
#  SAC NETWORKS
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Discrete Q-network: maps state → Q-value for each action.
    Two layers, ReLU activation (different from PPO's Tanh — more gradient flow).
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (batch, n_actions)


class PolicyNetwork(nn.Module):
    """
    Discrete policy: maps state → action probabilities (softmax).
    Separate from Q-networks (SAC uses separate policy network).
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_probs, log_action_probs) with numerical stability."""
        logits = self.net(x)
        # Stable softmax + log
        action_probs = F.softmax(logits, dim=-1)
        # Clamp for numerical stability before log
        log_action_probs = torch.log(action_probs.clamp(min=1e-8))
        return action_probs, log_action_probs

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        with torch.no_grad():
            action_probs, _ = self.forward(obs)
            if deterministic:
                return action_probs.argmax(dim=-1).item()
            else:
                dist = Categorical(probs=action_probs)
                return dist.sample().item()


# ══════════════════════════════════════════════════════════════════════════════
#  REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Efficient replay buffer using numpy arrays (not list of tuples).
    Pre-allocates memory for max_size transitions.
    """

    def __init__(self, max_size: int = 500_000, obs_dim: int = OBS_DIM):
        self.max_size = max_size
        self.obs_dim  = obs_dim
        self.ptr      = 0
        self.size     = 0

        # Pre-allocate
        self.obs      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(max_size, dtype=np.int64)
        self.rewards  = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(max_size, dtype=np.float32)

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool):
        idx = self.ptr % self.max_size
        self.obs[idx]      = obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.next_obs[idx] = next_obs
        self.dones[idx]    = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        indices = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs      = torch.tensor(self.obs[indices],      dtype=torch.float32, device=device),
            actions  = torch.tensor(self.actions[indices],  dtype=torch.long,    device=device),
            rewards  = torch.tensor(self.rewards[indices],  dtype=torch.float32, device=device),
            next_obs = torch.tensor(self.next_obs[indices], dtype=torch.float32, device=device),
            dones    = torch.tensor(self.dones[indices],    dtype=torch.float32, device=device),
        )

    def __len__(self):
        return self.size


# ══════════════════════════════════════════════════════════════════════════════
#  SAC AGENT
# ══════════════════════════════════════════════════════════════════════════════

class SACAgent:
    """
    Discrete SAC agent with automatic entropy tuning.

    Key components:
      - Policy network (actor): outputs action probabilities
      - 2 Q-networks (critics): estimate Q(s,a), take minimum (double Q trick)
      - 2 Target Q-networks: slowly updated copies for stable targets
      - Alpha (temperature): auto-tuned to maintain target entropy
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 256,
        lr_policy: float = 3e-4,
        lr_q: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        target_entropy_ratio: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.gamma     = gamma
        self.tau       = tau
        self.n_actions = n_actions
        self.device    = device

        # Networks
        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_dim).to(device)
        self.q1     = QNetwork(obs_dim, n_actions, hidden_dim).to(device)
        self.q2     = QNetwork(obs_dim, n_actions, hidden_dim).to(device)
        self.q1_tgt = copy.deepcopy(self.q1)
        self.q2_tgt = copy.deepcopy(self.q2)

        # Freeze target networks
        for p in self.q1_tgt.parameters():
            p.requires_grad = False
        for p in self.q2_tgt.parameters():
            p.requires_grad = False

        # Optimizers
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.q1_optim     = torch.optim.Adam(self.q1.parameters(), lr=lr_q)
        self.q2_optim     = torch.optim.Adam(self.q2.parameters(), lr=lr_q)

        # Auto entropy tuning
        # Target entropy = -ratio * log(1/|A|) = ratio * log(|A|)
        self.target_entropy = target_entropy_ratio * math.log(n_actions)
        self.log_alpha = torch.tensor(
            math.log(alpha_init), dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def n_params(self) -> int:
        total = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total += sum(p.numel() for p in self.q1.parameters() if p.requires_grad)
        total += sum(p.numel() for p in self.q2.parameters() if p.requires_grad)
        return total

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.policy.act(obs_t, deterministic=deterministic)

    def update(self, batch: dict) -> dict:
        """
        One SAC update step on a batch from replay buffer.
        Returns dict of training metrics.
        """
        obs      = batch["obs"]
        actions  = batch["actions"]
        rewards  = batch["rewards"]
        next_obs = batch["next_obs"]
        dones    = batch["dones"]

        alpha = self.log_alpha.exp().detach()

        # ── Q-network update ─────────────────────────────────────────────
        with torch.no_grad():
            # Next state action probs from policy
            next_probs, next_log_probs = self.policy(next_obs)

            # Target Q values (minimum of dual Q)
            q1_tgt_next = self.q1_tgt(next_obs)  # (batch, n_actions)
            q2_tgt_next = self.q2_tgt(next_obs)
            q_tgt_next  = torch.min(q1_tgt_next, q2_tgt_next)

            # V(s') = sum_a pi(a|s') * (Q(s',a) - alpha * log pi(a|s'))
            v_next = (next_probs * (q_tgt_next - alpha * next_log_probs)).sum(dim=-1)

            # TD target
            q_target = rewards + self.gamma * (1.0 - dones) * v_next

        # Current Q values for taken actions
        q1_vals = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_vals = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(q1_vals, q_target)
        q2_loss = F.mse_loss(q2_vals, q_target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optim.step()

        # ── Policy update ────────────────────────────────────────────────
        action_probs, log_action_probs = self.policy(obs)

        # Q values from current (non-target) networks
        with torch.no_grad():
            q1_curr = self.q1(obs)
            q2_curr = self.q2(obs)
            q_curr  = torch.min(q1_curr, q2_curr)

        # Policy loss: minimize E_a[alpha * log pi(a|s) - Q(s,a)]
        policy_loss = (action_probs * (alpha * log_action_probs - q_curr)).sum(dim=-1).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optim.step()

        # ── Alpha (entropy temperature) update ───────────────────────────
        # Increase alpha if entropy too low, decrease if too high
        with torch.no_grad():
            action_probs_d, log_probs_d = self.policy(obs)
            entropy = -(action_probs_d * log_probs_d).sum(dim=-1).mean()

        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ── Soft update target networks ──────────────────────────────────
        with torch.no_grad():
            for p, p_tgt in zip(self.q1.parameters(), self.q1_tgt.parameters()):
                p_tgt.data.mul_(1 - self.tau).add_(p.data * self.tau)
            for p, p_tgt in zip(self.q2.parameters(), self.q2_tgt.parameters()):
                p_tgt.data.mul_(1 - self.tau).add_(p.data * self.tau)

        return dict(
            q1_loss     = q1_loss.item(),
            q2_loss     = q2_loss.item(),
            policy_loss = policy_loss.item(),
            alpha_loss  = alpha_loss.item(),
            alpha       = self.alpha,
            entropy     = entropy.item(),
            q1_mean     = q1_vals.mean().item(),
        )

    def save(self, path: str, fold: int, eval_metrics: dict, args: dict):
        torch.save(dict(
            policy_state = self.policy.state_dict(),
            q1_state     = self.q1.state_dict(),
            q2_state     = self.q2.state_dict(),
            log_alpha    = self.log_alpha.item(),
            fold         = fold,
            eval_metrics = eval_metrics,
            args         = args,
            obs_dim      = OBS_DIM,
            n_actions    = N_ACTIONS,
        ), path)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS (identical to PPO v7)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(all_pnls: List[float], trade_count: float, n_days: int) -> dict:
    if not all_pnls or trade_count == 0:
        return dict(sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
                    avg_pnl=0.0, trades_per_day=0.0)

    pnls   = np.array(all_pnls, dtype=np.float64)
    n      = len(pnls)
    mu     = pnls.mean()
    std    = pnls.std() + 1e-8
    sharpe = mu / std * math.sqrt(n)

    downside = pnls[pnls < 0]
    dstd     = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
    sortino  = mu / dstd * math.sqrt(n)

    gross_win  = pnls[pnls > 0].sum() if (pnls > 0).any() else 0.0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-8
    pf         = gross_win / gross_loss
    wr         = float((pnls > 0).mean())

    return dict(
        sharpe       = float(sharpe),
        sortino      = float(sortino),
        pf           = float(pf),
        wr           = float(wr),
        avg_pnl      = float(mu),
        trades_per_day = float(trade_count / max(n_days, 1)),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER (SAC version — collects to replay buffer)
# ══════════════════════════════════════════════════════════════════════════════

def run_episode_sac(
    env: TradingEnv,
    agent: SACAgent,
    replay_buffer: Optional[ReplayBuffer] = None,
    deterministic: bool = False,
) -> dict:
    """
    Run one episode, optionally storing transitions in replay buffer.
    Returns episode stats.
    """
    obs  = env.reset()
    done = False
    total_reward = 0.0
    step_count   = 0

    while not done:
        action = agent.act(obs, deterministic=deterministic)
        next_obs, reward, done = env.step(action)

        if replay_buffer is not None:
            replay_buffer.add(obs, action, reward, next_obs, done)

        obs           = next_obs
        total_reward += reward
        step_count   += 1

    return dict(
        total_reward = total_reward,
        trade_count  = env.trade_count,
        trade_pnls   = env.trade_pnls,
        steps        = step_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD TRAINING (SAC version)
# ══════════════════════════════════════════════════════════════════════════════

def train_fold_sac(
    agent: SACAgent,
    replay_buffer: ReplayBuffer,
    train_dates: List[str],
    data_dir: str, pred_dir: str, patchtst_dir: str,
    args: argparse.Namespace,
    fold_idx: int,
) -> List[dict]:
    """
    Train SAC on one fold's train days.

    SAC difference from PPO:
      - Collect experience into replay buffer (not rollout buffer)
      - After each day, do N gradient updates sampling from replay
      - Off-policy: old experience is reused, more sample efficient
    """
    epoch_metrics = []

    for epoch in range(args.epochs):
        all_pnls     = []
        total_trades = 0
        update_metrics = defaultdict(list)

        for date in train_dates:
            day_data = load_day(date, data_dir, pred_dir, patchtst_dir)
            if day_data is None:
                continue

            env = TradingEnv(day_data)
            ep  = run_episode_sac(env, agent, replay_buffer=replay_buffer, deterministic=False)
            all_pnls.extend(ep["trade_pnls"])
            total_trades += ep["trade_count"]

            # Off-policy updates: do multiple gradient steps per day
            if len(replay_buffer) >= args.warmup_steps:
                n_updates = max(1, ep["steps"] // args.update_every)
                for _ in range(n_updates):
                    batch = replay_buffer.sample(args.batch_size, agent.device)
                    metrics = agent.update(batch)
                    for k, v in metrics.items():
                        update_metrics[k].append(v)

            del day_data, env
            gc.collect()

        m = compute_metrics(all_pnls, total_trades, len(train_dates))
        m["epoch"] = epoch
        m["fold"]  = fold_idx

        # Add SAC-specific metrics
        for k in ["q1_loss", "q2_loss", "policy_loss", "alpha", "entropy", "q1_mean"]:
            if k in update_metrics:
                m[f"sac_{k}"] = float(np.mean(update_metrics[k]))

        epoch_metrics.append(m)

        log.info(
            f"  Fold {fold_idx} Epoch {epoch:2d} | "
            f"Sortino={m['sortino']:+.3f}  WR={m['wr']:.1%}  PF={m['pf']:.2f}  "
            f"Trades/day={m['trades_per_day']:.1f}  AvgPnL={m['avg_pnl']:+.3f}t  "
            f"Alpha={m.get('sac_alpha', 0):.3f}  Entropy={m.get('sac_entropy', 0):.2f}"
        )

    return epoch_metrics


def eval_fold_sac(
    agent: SACAgent,
    eval_dates: List[str],
    data_dir: str, pred_dir: str, patchtst_dir: str,
    fold_idx: int,
) -> dict:
    """Evaluate SAC (deterministic policy) on eval days."""
    all_pnls     = []
    total_trades = 0

    for date in eval_dates:
        day_data = load_day(date, data_dir, pred_dir, patchtst_dir)
        if day_data is None:
            continue
        env = TradingEnv(day_data)
        ep  = run_episode_sac(env, agent, replay_buffer=None, deterministic=True)
        all_pnls.extend(ep["trade_pnls"])
        total_trades += ep["trade_count"]
        del day_data, env
        gc.collect()

    m = compute_metrics(all_pnls, total_trades, len(eval_dates))
    m["fold"] = fold_idx
    log.info(
        f"  [EVAL] Fold {fold_idx} | "
        f"Sortino={m['sortino']:+.3f}  Sharpe={m['sharpe']:+.3f}  "
        f"WR={m['wr']:.1%}  PF={m['pf']:.2f}  "
        f"Trades/day={m['trades_per_day']:.1f}  AvgPnL={m['avg_pnl']:+.3f}t"
    )
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC v7: Discrete Soft Actor-Critic Execution RL")
    p.add_argument("--data-dir",     required=True, help="Path to mbo_events_smart_v3/")
    p.add_argument("--pred-dir",     required=True, help="Path to cnn_mamba_v2_bulk_oot/")
    p.add_argument("--patchtst-dir", required=True, help="Path to patchtst_bulk_oot/")
    p.add_argument("--output-dir",   required=True, help="Output directory for checkpoints")
    # SAC-specific
    p.add_argument("--hidden-dim",   type=int,   default=256,    help="Network hidden dim")
    p.add_argument("--epochs",       type=int,   default=12,     help="Epochs per fold")
    p.add_argument("--train-days",   type=int,   default=25,     help="Training days per fold")
    p.add_argument("--eval-days",    type=int,   default=3,      help="Eval days per fold")
    p.add_argument("--lr-policy",    type=float, default=3e-4,   help="Policy learning rate")
    p.add_argument("--lr-q",         type=float, default=3e-4,   help="Q-network learning rate")
    p.add_argument("--lr-alpha",     type=float, default=1e-4,   help="Alpha (entropy) learning rate")
    p.add_argument("--gamma",        type=float, default=0.99,   help="Discount factor")
    p.add_argument("--tau",          type=float, default=0.005,  help="Soft update coefficient")
    p.add_argument("--alpha-init",   type=float, default=0.2,    help="Initial entropy temperature")
    p.add_argument("--target-entropy-ratio", type=float, default=0.5, help="Target entropy as ratio of max")
    p.add_argument("--batch-size",   type=int,   default=512,    help="Batch size for gradient updates")
    p.add_argument("--buffer-size",  type=int,   default=500_000, help="Replay buffer capacity")
    p.add_argument("--warmup-steps", type=int,   default=5000,   help="Min buffer size before updates")
    p.add_argument("--update-every", type=int,   default=4,      help="Steps between gradient updates (ratio)")
    p.add_argument("--device",       type=str,   default="cuda", help="torch device")
    p.add_argument("--no-mlflow",    action="store_true",         help="Disable MLflow logging")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find valid dates ─────────────────────────────────────────────────────
    valid_dates = find_valid_dates(args.pred_dir, args.patchtst_dir)

    # Validate MBO event files exist
    dates_loaded = []
    for date in valid_dates:
        data_path = Path(args.data_dir)
        ev_file = data_path / f"{date}_mbo_events_smart_v3.npz"
        if not ev_file.exists():
            ev_file = data_path / f"{date}_mbo_events.npz"
        if ev_file.exists():
            dates_loaded.append(date)

    if len(dates_loaded) < args.train_days + args.eval_days:
        log.error(f"Only {len(dates_loaded)} dates available. Need {args.train_days + args.eval_days}.")
        sys.exit(1)

    log.info(f"Validated {len(dates_loaded)} dates: {dates_loaded[0]} to {dates_loaded[-1]}")

    # ── Build SAC agent ──────────────────────────────────────────────────────
    agent = SACAgent(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=args.hidden_dim,
        lr_policy=args.lr_policy,
        lr_q=args.lr_q,
        lr_alpha=args.lr_alpha,
        gamma=args.gamma,
        tau=args.tau,
        alpha_init=args.alpha_init,
        target_entropy_ratio=args.target_entropy_ratio,
        device=device,
    )

    n_params = agent.n_params()

    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=args.buffer_size, obs_dim=OBS_DIM)

    # ── Walk-forward folds ───────────────────────────────────────────────────
    n_dates = len(dates_loaded)
    n_folds = max(1, (n_dates - args.train_days) // args.eval_days)

    print(f"\n{'='*60}")
    print("=== SAC v7: Discrete Soft Actor-Critic Execution RL ===")
    print(f"Dates: {len(dates_loaded)} OOT dates ({dates_loaded[0]} to {dates_loaded[-1]})")
    print(f"Folds: {n_folds} (train={args.train_days}d, eval={args.eval_days}d)")
    print(f"Obs dim: {OBS_DIM}, Actions: {N_ACTIONS}, Params: {n_params:,}")
    print(f"Hidden: {args.hidden_dim}, Batch: {args.batch_size}, Buffer: {args.buffer_size:,}")
    print(f"Alpha init: {args.alpha_init}, Target entropy ratio: {args.target_entropy_ratio}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow_active = False
    if MLFLOW_OK and not getattr(args, 'no_mlflow', False):
        try:
            mlflow.set_experiment("SAC_v7_Execution")
            mlflow.start_run(run_name=f"sac_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({k: str(v) for k, v in vars(args).items()})
            mlflow.log_param("n_params", n_params)
            mlflow.log_param("n_dates",  len(dates_loaded))
            mlflow.log_param("n_folds",  n_folds)
            mlflow_active = True
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_sortino    = -np.inf
    best_ckpt_path  = None
    all_eval_metrics = []

    for fold in range(n_folds):
        train_start = fold * args.eval_days
        train_end   = train_start + args.train_days
        eval_start  = train_end
        eval_end    = min(eval_start + args.eval_days, n_dates)

        if eval_end > n_dates:
            break

        train_dates_fold = dates_loaded[train_start:train_end]
        eval_dates_fold  = dates_loaded[eval_start:eval_end]

        log.info(
            f"\nFold {fold}/{n_folds-1} | "
            f"Train: {train_dates_fold[0]}..{train_dates_fold[-1]} ({len(train_dates_fold)}d) | "
            f"Eval: {eval_dates_fold[0]}..{eval_dates_fold[-1]} ({len(eval_dates_fold)}d)"
        )

        # NOTE: SAC replay buffer persists across folds (off-policy advantage!)
        # This means later folds benefit from ALL previous experience.

        epoch_metrics = train_fold_sac(
            agent, replay_buffer, train_dates_fold,
            args.data_dir, args.pred_dir, args.patchtst_dir,
            args, fold_idx=fold,
        )

        eval_metrics = eval_fold_sac(
            agent, eval_dates_fold,
            args.data_dir, args.pred_dir, args.patchtst_dir,
            fold_idx=fold,
        )
        all_eval_metrics.append(eval_metrics)

        # Log to MLflow
        if mlflow_active:
            try:
                prefix = f"fold{fold}/"
                mlflow.log_metrics({
                    prefix + "eval_sortino": eval_metrics["sortino"],
                    prefix + "eval_sharpe":  eval_metrics["sharpe"],
                    prefix + "eval_wr":      eval_metrics["wr"],
                    prefix + "eval_pf":      eval_metrics["pf"],
                    prefix + "eval_tpd":     eval_metrics["trades_per_day"],
                }, step=fold)
                # Log last epoch's SAC metrics
                if epoch_metrics:
                    last = epoch_metrics[-1]
                    for k in ["sac_alpha", "sac_entropy", "sac_q1_loss", "sac_q1_mean"]:
                        if k in last:
                            mlflow.log_metric(prefix + k, last[k], step=fold)
            except Exception:
                pass

        # Save best checkpoint
        if eval_metrics["sortino"] > best_sortino:
            best_sortino   = eval_metrics["sortino"]
            best_ckpt_path = str(out_dir / f"best_sac_fold{fold}.pt")
            agent.save(best_ckpt_path, fold, eval_metrics, vars(args))
            log.info(f"  *** New best checkpoint: Sortino={best_sortino:.3f} → {best_ckpt_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    if all_eval_metrics:
        all_sortino = [m["sortino"]        for m in all_eval_metrics]
        all_sharpe  = [m["sharpe"]         for m in all_eval_metrics]
        all_wr      = [m["wr"]             for m in all_eval_metrics]
        all_tpd     = [m["trades_per_day"] for m in all_eval_metrics]
        all_pf      = [m["pf"]            for m in all_eval_metrics]

        print("\n" + "="*60)
        print("SAC v7 WALK-FORWARD SUMMARY (all eval folds)")
        print(f"  Sortino:       {np.mean(all_sortino):+.3f} ± {np.std(all_sortino):.3f}")
        print(f"  Sharpe:        {np.mean(all_sharpe):+.3f} ± {np.std(all_sharpe):.3f}")
        print(f"  Win Rate:      {np.mean(all_wr):.1%}")
        print(f"  Profit Factor: {np.mean(all_pf):.2f}")
        print(f"  Trades/day:    {np.mean(all_tpd):.1f}")
        print(f"  Best Sortino:  fold {np.argmax(all_sortino)} ({max(all_sortino):.3f})")
        print(f"  Best ckpt:     {best_ckpt_path}")
        print("="*60)

        if mlflow_active:
            try:
                mlflow.log_metrics({
                    "summary/sortino_mean": float(np.mean(all_sortino)),
                    "summary/sharpe_mean":  float(np.mean(all_sharpe)),
                    "summary/wr_mean":      float(np.mean(all_wr)),
                    "summary/pf_mean":      float(np.mean(all_pf)),
                    "summary/tpd_mean":     float(np.mean(all_tpd)),
                })
                if best_ckpt_path:
                    mlflow.log_artifact(best_ckpt_path)
                mlflow.end_run()
            except Exception:
                pass

    log.info("Done.")


if __name__ == "__main__":
    main()
