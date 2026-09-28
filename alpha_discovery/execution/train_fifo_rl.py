#!/usr/bin/env python3
"""
PPO Trainer for FIFO RL Execution Agent
========================================

Trains a PPO agent on the FIFOExecutionEnv. No stable-baselines3 dependency —
vanilla PPO implemented in PyTorch.

Key design choices:
- Walk-forward evaluation: train on N days, eval on next M days (chronological)
- Reward = delta Sortino per trade (defined in env; risk-adjusted, no short bias)
- Best agent saved by eval Sortino ratio
- MLflow logging: experiment "FIFO_RL_Execution"
- Training curve PNG: reward, Sortino, WR, avg P&L, trades/day
- Cost model: passive limit = 0.376 ticks commission only (HC #89)
                market order = 1.376 ticks (commission + 1.0 tick spread)

Usage:
  python train_fifo_rl.py --epochs 50 --lr 3e-4 --data-dir /path/to/mbo_events
  python train_fifo_rl.py --help

HC compliance:
  - HC #0:  sliding walk-forward (no expanding window)
  - HC #69: primary metrics = Sharpe, Sortino, PF, WR (not raw P&L)
  - HC #89: cost = 0.376 ticks commission; no separate spread cost for limits
  - HC #100/#104: risk-adjusted reward; symmetric (no hard-coded short bias)

Author: Claude (Infrastructure Builder)
Date: 2026-05-02
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Torch ──────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not found. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

# ── Matplotlib (optional — graceful fallback) ──────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ── MLflow (optional — warn but continue) ─────────────────────────────────────
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Local env ──────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from fifo_rl_env import (
    FIFOExecutionEnv, OBS_DIM,
    EVENT_DIR as DEFAULT_EVENT_DIR,
    PRED_DIR  as DEFAULT_PRED_DIR,
    TICK_VALUE,
    COMMISSION_RT_TICKS,   # 0.376 — the ONLY cost (HC #231(A))
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("train_fifo_rl")

# ── Constants ──────────────────────────────────────────────────────────────────
N_ACTIONS      = 7         # discrete action space (matches env)
MLFLOW_EXPERIMENT = "FIFO_RL_Execution"

# PPO defaults (overridable via CLI)
DEFAULT_EPOCHS       = 100
DEFAULT_LR           = 3e-4
DEFAULT_ROLLOUT_LEN  = 4096    # steps per PPO update (events processed per update)
DEFAULT_N_PPO_EPOCHS = 4       # PPO update passes per rollout
DEFAULT_MINI_BATCH   = 512
DEFAULT_GAMMA        = 0.99
DEFAULT_LAM          = 0.95    # GAE lambda
DEFAULT_CLIP_EPS     = 0.2
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_VF_COEF      = 0.5
DEFAULT_MAX_GRAD     = 0.5
DEFAULT_TRAIN_DAYS   = 40      # walk-forward train window
DEFAULT_EVAL_DAYS    = 5       # walk-forward eval window


# ══════════════════════════════════════════════════════════════════════════════
# Network Architecture
# ══════════════════════════════════════════════════════════════════════════════

class FIFOActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic for FIFO execution.

    Architecture:
      Input (48) → LayerNorm → 3× (Linear → GELU → LayerNorm) → 256-dim hidden
      Actor head: hidden → 128 → 7 logits
      Critic head: hidden → 128 → 1 value

    Design notes:
    - LayerNorm on input handles the mixed scales in the 48-dim observation
    - GELU instead of ReLU: smoother gradients, avoids dead neurons
    - Orthogonal init with gain=√2 for hidden, gain=0.01 for policy head
      (standard PPO initialization — keeps initial policy close to uniform)
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(obs_dim)

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_actions),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Orthogonal init
        self._init_weights()

    def _init_weights(self):
        for module in self.shared.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        # Actor output: small gain → near-uniform initial policy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        # Critic output: unit gain
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value)."""
        x = self.input_norm(obs)
        x = self.shared(x)
        logits = self.actor(x)
        value  = self.critic(x).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log-prob + entropy.

        Returns
        -------
        action    : (B,) int64
        log_prob  : (B,) float
        entropy   : (B,) float
        value     : (B,) float
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value


# ══════════════════════════════════════════════════════════════════════════════
# Rollout Buffer
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Fixed-size buffer for PPO rollouts.

    Stores: obs, actions, log_probs, rewards, dones, values.
    Computes GAE advantages and returns after collection.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        gamma: float,
        lam: float,
        device: torch.device,
    ):
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.gamma    = gamma
        self.lam      = lam
        self.device   = device

        # Pre-allocate on CPU (filled incrementally)
        self.obs       = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions   = np.zeros((capacity,), dtype=np.int64)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.rewards   = np.zeros((capacity,), dtype=np.float32)
        self.dones     = np.zeros((capacity,), dtype=np.float32)
        self.values    = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0       # write pointer
        self.full = False

    def reset(self):
        self.ptr  = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = float(done)
        self.values[self.ptr]    = value
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True

    def is_ready(self) -> bool:
        return self.full

    def compute_advantages(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and discounted returns.

        GAE(λ): A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
        where δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        Parameters
        ----------
        last_value : V(s_{T+1}) — value estimate at end of rollout

        Returns
        -------
        advantages : (capacity,) float32
        returns    : (capacity,) float32  (advantages + values)
        """
        advantages = np.zeros(self.capacity, dtype=np.float32)
        gae = 0.0
        next_val = last_value

        for t in reversed(range(self.capacity)):
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_val * not_done - self.values[t]
            gae   = delta + self.gamma * self.lam * not_done * gae
            advantages[t] = gae
            next_val = self.values[t]

        returns = advantages + self.values
        return advantages, returns

    def get_batches(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
        batch_size: int,
    ):
        """
        Yield shuffled mini-batches as tensors.

        Yields dicts with keys: obs, actions, old_log_probs, advantages, returns
        """
        indices = np.random.permutation(self.capacity)

        for start in range(0, self.capacity, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs":          torch.tensor(self.obs[idx],       device=self.device),
                "actions":      torch.tensor(self.actions[idx],   device=self.device),
                "old_log_probs":torch.tensor(self.log_probs[idx], device=self.device),
                "advantages":   torch.tensor(advantages[idx],     device=self.device),
                "returns":      torch.tensor(returns[idx],        device=self.device),
            }


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Data Split
# ══════════════════════════════════════════════════════════════════════════════

def build_walk_forward_folds(
    event_files: List[Path],
    train_days: int,
    eval_days: int,
) -> List[Dict]:
    """
    Build sliding walk-forward folds (HC #0: SLIDING, not expanding).

    Each fold:
      train: files[i : i + train_days]
      eval:  files[i + train_days : i + train_days + eval_days]

    Slide by eval_days each step so there's no overlap in eval sets.

    Parameters
    ----------
    event_files : chronologically sorted list of episode files
    train_days  : number of days in training window
    eval_days   : number of OOT eval days per fold

    Returns
    -------
    list of dicts with keys: 'fold', 'train_files', 'eval_files'
    """
    folds = []
    n = len(event_files)
    window = train_days + eval_days

    i = 0
    fold_num = 0
    while i + window <= n:
        train_files = event_files[i : i + train_days]
        eval_files  = event_files[i + train_days : i + train_days + eval_days]
        folds.append({
            "fold": fold_num,
            "train_files": train_files,
            "eval_files":  eval_files,
        })
        i += eval_days          # SLIDING: advance by eval window only
        fold_num += 1

    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_episode_metrics(trade_pnls: List[float], n_steps: int) -> Dict[str, float]:
    """
    Compute Sharpe, Sortino, PF, WR from a list of trade P&Ls in ticks.

    Primary metrics (HC #69): Sortino, Sharpe, PF, WR.
    Secondary: avg_pnl_ticks, total_pnl_usd, trades_per_step.

    Cost model (HC #89): already baked into pnl_ticks by the env.
    """
    if not trade_pnls:
        return {
            "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": 0.0, "avg_pnl_ticks": 0.0, "total_pnl_ticks": 0.0,
            "total_pnl_usd": 0.0, "trades_per_step": 0.0,
        }

    pnls = np.array(trade_pnls, dtype=np.float64)
    n = len(pnls)

    win_rate     = float((pnls > 0).mean())
    mean_pnl     = float(pnls.mean())
    std_pnl      = float(pnls.std()) if n > 1 else 1.0

    # Sharpe (per-trade; scale √n for approximate annualization heuristic)
    sharpe = mean_pnl / (std_pnl + 1e-8) * math.sqrt(max(n, 1))

    # Sortino (downside deviation only, MAR=0)
    downside = pnls[pnls < 0]
    if len(downside) == 0:
        sortino = mean_pnl * 10.0
    else:
        dd = float(np.sqrt(np.mean(downside ** 2)))
        sortino = mean_pnl / (dd + 1e-8)

    # Profit factor
    gross_wins   = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_losses = abs(float(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-8
    pf = gross_wins / gross_losses

    return {
        "n_trades":       n,
        "win_rate":       win_rate,
        "sharpe":         sharpe,
        "sortino":        sortino,
        "profit_factor":  pf,
        "avg_pnl_ticks":  mean_pnl,
        "total_pnl_ticks": float(pnls.sum()),
        "total_pnl_usd":  float(pnls.sum()) * TICK_VALUE,
        "trades_per_step": n / max(n_steps, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PPO Update
# ══════════════════════════════════════════════════════════════════════════════

def ppo_update(
    policy: FIFOActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    n_ppo_epochs: int,
    batch_size: int,
    clip_eps: float,
    entropy_coef: float,
    vf_coef: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    """
    Run PPO update for n_ppo_epochs over the collected rollout.

    Clipped surrogate objective:
      L_CLIP = E[min(r_t A_t, clip(r_t, 1±ε) A_t)]
    Value loss:
      L_VF = MSE(V(s), returns)
    Entropy bonus:
      L_ENT = -H[π(·|s)]

    Total: L = -L_CLIP + vf_coef * L_VF - entropy_coef * L_ENT

    Returns
    -------
    dict of mean losses for logging
    """
    advantages, returns = buffer.compute_advantages(last_value)

    # Normalize advantages (reduces variance, standard PPO practice)
    adv_mean = advantages.mean()
    adv_std  = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    total_policy_loss = 0.0
    total_value_loss  = 0.0
    total_entropy     = 0.0
    n_updates = 0

    policy.train()
    for _ in range(n_ppo_epochs):
        for batch in buffer.get_batches(advantages, returns, batch_size):
            obs_b       = batch["obs"]
            act_b       = batch["actions"]
            old_lp_b    = batch["old_log_probs"]
            adv_b       = batch["advantages"]
            returns_b   = batch["returns"]

            _, new_log_prob, entropy, new_value = policy.get_action_and_value(obs_b, act_b)

            # Probability ratio
            ratio = torch.exp(new_log_prob - old_lp_b)

            # Clipped surrogate
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE)
            value_loss = F.mse_loss(new_value, returns_b)

            # Entropy bonus (maximized → minus sign in total)
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += (-entropy_loss).item()
            n_updates += 1

    n_updates = max(n_updates, 1)
    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss":  total_value_loss  / n_updates,
        "entropy":     total_entropy     / n_updates,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Rollout Collection
# ══════════════════════════════════════════════════════════════════════════════

def _worker_collect_rollout(args):
    """
    HC #144: Worker function for parallel rollout collection.

    Runs in a subprocess — has its own env, policy copy, and collects
    a chunk of rollout data independently. Returns numpy arrays.
    """
    (env_kwargs, policy_state, train_files_str, capacity_per_worker,
     obs_dim, hidden_dim, n_actions, device_str, worker_id) = args

    import torch
    device = torch.device(device_str)

    # Reconstruct env and policy in this process
    env = FIFOExecutionEnv(**env_kwargs)
    policy = FIFOActorCritic(obs_dim, n_actions, hidden_dim)
    policy.load_state_dict(policy_state)
    policy.to(device)
    policy.eval()

    train_files = [Path(f) for f in train_files_str]

    # Collect rollout data into local arrays
    obs_buf = np.zeros((capacity_per_worker, obs_dim), dtype=np.float32)
    act_buf = np.zeros(capacity_per_worker, dtype=np.int64)
    logp_buf = np.zeros(capacity_per_worker, dtype=np.float32)
    rew_buf = np.zeros(capacity_per_worker, dtype=np.float32)
    done_buf = np.zeros(capacity_per_worker, dtype=np.float32)
    val_buf = np.zeros(capacity_per_worker, dtype=np.float32)

    episode_trade_pnls = []
    episode_rewards = []
    n_episodes = 0
    idx = 0
    file_idx = worker_id  # each worker starts on a different file

    obs = env.reset(episode_file=train_files[file_idx % len(train_files)])

    while idx < capacity_per_worker:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(obs_t)

        action_np = int(action.item())
        next_obs, reward, done, info = env.step(action_np)

        obs_buf[idx] = obs
        act_buf[idx] = action_np
        logp_buf[idx] = float(log_prob.item())
        rew_buf[idx] = float(reward)
        done_buf[idx] = float(done)
        val_buf[idx] = float(value.item())
        idx += 1

        obs = next_obs
        if done:
            episode_rewards.append(info.get("episode_pnl_ticks", 0.0))
            for t in env._trade_history:
                episode_trade_pnls.append(t.pnl_ticks)
            n_episodes += 1
            file_idx += len(train_files) // max(1, 4)  # stride across files
            obs = env.reset(episode_file=train_files[file_idx % len(train_files)])

    # Get last value for GAE
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, _, _, last_val = policy.get_action_and_value(obs_t)

    return {
        "obs": obs_buf, "actions": act_buf, "log_probs": logp_buf,
        "rewards": rew_buf, "dones": done_buf, "values": val_buf,
        "last_value": float(last_val.item()),
        "trade_pnls": episode_trade_pnls,
        "episode_rewards": episode_rewards,
        "n_episodes": n_episodes,
        "n_steps": idx,
    }


def collect_rollout(
    env: FIFOExecutionEnv,
    policy: FIFOActorCritic,
    buffer: RolloutBuffer,
    train_files: List[Path],
    device: torch.device,
    n_workers: int = 1,  # HC #144: multiprocess workers (1=single, >1=parallel, TODO: fix fork issues)
) -> Tuple[Dict[str, float], float]:
    """
    Fill the rollout buffer using parallel subprocess workers.

    HC #144: Uses N worker processes, each with its own env + policy copy.
    Each worker collects capacity/N steps independently, then results are merged.
    This achieves true multi-core parallelism (no GIL).

    Parameters
    ----------
    n_workers : number of parallel workers. 0 = auto (min(cpu_count, len(train_files), 8))
    """
    import multiprocessing as mp

    if n_workers <= 0:
        n_workers = min(mp.cpu_count(), len(train_files), 8)
    n_workers = max(1, min(n_workers, len(train_files)))

    # For small rollouts or few files, single-process is faster (no fork overhead)
    if n_workers <= 1 or buffer.capacity < 2048:
        return _collect_rollout_single(env, policy, buffer, train_files, device)

    log.info(f"HC #144: Parallel rollout with {n_workers} workers "
             f"({buffer.capacity} total steps, {buffer.capacity // n_workers} per worker)")

    buffer.reset()
    policy.eval()
    capacity_per_worker = buffer.capacity // n_workers

    # Prepare worker args (must be picklable)
    policy_state = {k: v.cpu() for k, v in policy.state_dict().items()}
    env_kwargs = {
        "event_dir": str(env.event_dir),
        "pred_dir": str(env.pred_dir),
        "patchtst_pred_dir": str(env.patchtst_pred_dir) if env.patchtst_pred_dir else None,
        "pred_window": env.pred_window,
        "pred_stride": env.pred_stride,
        "max_steps_per_episode": env.max_steps,
        "rth_only": env.rth_only,
        "verbose": False,
    }
    train_files_str = [str(f) for f in train_files]
    # Infer hidden_dim from first linear layer in shared network
    hidden_dim = policy.shared[0].out_features

    worker_args = [
        (env_kwargs, policy_state, train_files_str, capacity_per_worker,
         OBS_DIM, hidden_dim, env.action_space.n, str(device), i)
        for i in range(n_workers)
    ]

    # Launch workers
    t0 = time.time()
    # Use fork on Linux (fast), spawn on Windows/macOS (CUDA-safe)
    ctx_method = "fork" if sys.platform == "linux" else "spawn"
    ctx = mp.get_context(ctx_method)
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_worker_collect_rollout, worker_args)
    t1 = time.time()

    # Merge results into the main buffer
    all_trade_pnls = []
    all_episode_rewards = []
    total_episodes = 0
    total_steps = 0

    for r in results:
        n = r["n_steps"]
        for j in range(n):
            buffer.add(
                r["obs"][j], int(r["actions"][j]), float(r["log_probs"][j]),
                float(r["rewards"][j]), bool(r["dones"][j]), float(r["values"][j])
            )
            if buffer.is_ready():
                break
        all_trade_pnls.extend(r["trade_pnls"])
        all_episode_rewards.extend(r["episode_rewards"])
        total_episodes += r["n_episodes"]
        total_steps += n

    # Use last value from the first worker
    last_value_np = results[0]["last_value"]

    log.info(f"Parallel rollout done: {total_steps} steps from {n_workers} workers in {t1-t0:.1f}s")

    rollout_metrics = compute_episode_metrics(all_trade_pnls, total_steps)
    rollout_metrics["n_episodes"] = total_episodes
    rollout_metrics["n_workers"] = n_workers
    rollout_metrics["rollout_time_s"] = t1 - t0
    rollout_metrics["mean_episode_pnl"] = (
        float(np.mean(all_episode_rewards)) if all_episode_rewards else 0.0
    )

    return rollout_metrics, last_value_np


def _collect_rollout_single(
    env: FIFOExecutionEnv,
    policy: FIFOActorCritic,
    buffer: RolloutBuffer,
    train_files: List[Path],
    device: torch.device,
) -> Tuple[Dict[str, float], float]:
    """Original single-env rollout collection (fallback for small rollouts)."""
    buffer.reset()
    policy.eval()

    episode_rewards: List[float] = []
    episode_trade_pnls: List[float] = []
    n_episodes = 0
    n_total_steps = 0

    file_idx = 0
    obs = env.reset(episode_file=train_files[file_idx % len(train_files)])

    while not buffer.is_ready():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(obs_t)

        action_np   = int(action.item())
        log_prob_np = float(log_prob.item())
        value_np    = float(value.item())

        next_obs, reward, done, info = env.step(action_np)

        buffer.add(obs, action_np, log_prob_np, float(reward), done, value_np)
        obs = next_obs
        n_total_steps += 1

        if done:
            episode_rewards.append(info.get("episode_pnl_ticks", 0.0))
            for t in env._trade_history:
                episode_trade_pnls.append(t.pnl_ticks)
            n_episodes += 1
            file_idx += 1
            obs = env.reset(episode_file=train_files[file_idx % len(train_files)])

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, _, _, last_value = policy.get_action_and_value(obs_t)
    last_value_np = float(last_value.item())

    rollout_metrics = compute_episode_metrics(episode_trade_pnls, n_total_steps)
    rollout_metrics["n_episodes"] = n_episodes
    rollout_metrics["mean_episode_pnl"] = (
        float(np.mean(episode_rewards)) if episode_rewards else 0.0
    )

    return rollout_metrics, last_value_np


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(
    env: FIFOExecutionEnv,
    policy: FIFOActorCritic,
    eval_files: List[Path],
    device: torch.device,
) -> Dict[str, float]:
    """
    Run deterministic (greedy) policy over eval files and compute metrics.

    Uses argmax action selection (no sampling) for clean eval.
    Returns risk-adjusted metrics (Sortino, Sharpe, PF, WR) — HC #69.
    """
    policy.eval()
    all_trade_pnls: List[float] = []
    trades_per_day: List[int]   = []
    sortinos: List[float]       = []
    wrs: List[float]            = []
    total_steps = 0

    for ep_file in eval_files:
        try:
            obs = env.reset(episode_file=ep_file)
        except Exception as _e:
            log.warning(f"evaluate_policy: skipping corrupt file {ep_file.name} — {_e}")
            continue
        done = False
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy.forward(obs_t)
                action_np = int(logits.argmax(dim=-1).item())
            obs, _, done, _ = env.step(action_np)
            steps += 1

        metrics = env.get_episode_metrics()
        for t in env._trade_history:
            all_trade_pnls.append(t.pnl_ticks)

        trades_per_day.append(metrics["n_trades"])
        sortinos.append(metrics["sortino"])
        wrs.append(metrics["win_rate"])
        total_steps += steps

    eval_metrics = compute_episode_metrics(all_trade_pnls, total_steps)
    eval_metrics["avg_trades_per_day"] = float(np.mean(trades_per_day)) if trades_per_day else 0.0
    eval_metrics["n_eval_days"]        = len(eval_files)

    return eval_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def save_training_curves(
    history: Dict[str, List],
    output_dir: Path,
    run_name: str,
) -> Optional[Path]:
    """
    Save a 5-panel training curve PNG.

    Panels: total_reward, sortino, win_rate, avg_pnl_ticks, trades_per_day.
    Both train (solid) and eval (dashed) lines where available.

    Returns path to saved PNG, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        log.warning("matplotlib not available — skipping training curve plot")
        return None

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(f"FIFO PPO Training — {run_name}", fontsize=14, fontweight="bold")

    epochs   = history.get("epoch", [])
    metrics  = [
        ("train_total_pnl_ticks", "eval_total_pnl_ticks", "Cumulative P&L (ticks)", "tab:blue"),
        ("train_sortino",         "eval_sortino",          "Sortino Ratio",          "tab:orange"),
        ("train_win_rate",        "eval_win_rate",         "Win Rate",               "tab:green"),
        ("train_avg_pnl_ticks",   "eval_avg_pnl_ticks",   "Avg P&L per Trade (ticks)","tab:red"),
        ("train_n_trades",        "eval_avg_trades_per_day","Trades / Day",           "tab:purple"),
    ]

    for ax, (train_key, eval_key, ylabel, color) in zip(axes, metrics):
        if train_key in history and history[train_key]:
            ax.plot(epochs, history[train_key], color=color, alpha=0.8,
                    linewidth=1.5, label="train")
        if eval_key in history and history[eval_key]:
            ax.plot(history.get("eval_epoch", epochs[:len(history[eval_key])]),
                    history[eval_key], color=color, alpha=1.0,
                    linewidth=2.0, linestyle="--", label="eval", marker="o", markersize=4)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Zero line for P&L and Sortino
        if "pnl" in ylabel.lower() or "sortino" in ylabel.lower():
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    axes[-1].set_xlabel("Epoch", fontsize=10)
    plt.tight_layout()

    plot_path = output_dir / f"training_curves_{run_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Training curves saved → {plot_path}")
    return plot_path


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    global MLFLOW_AVAILABLE
    """
    Main PPO training loop with walk-forward evaluation.

    Flow:
    1. Build walk-forward folds (sliding window, HC #0)
    2. For each fold: collect rollout → PPO update → eval on OOT days
    3. Save best checkpoint by eval Sortino (HC #69)
    4. Log all metrics to MLflow
    5. Save training curve PNG
    """
    # ── Setup ──────────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = datetime.now().strftime("ppo_%Y%m%d_%H%M%S")

    # Add file handler for reliable log capture (Windows Start-Process doesn't capture stderr)
    _log_file = output_dir / f"{run_name}.log"
    _fh = logging.FileHandler(str(_log_file), mode="w")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)
    # Also add to fifo_rl_env logger
    logging.getLogger("fifo_rl_env").addHandler(_fh)

    log.info(f"Run: {run_name}")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Log file: {_log_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    event_dir = Path(args.data_dir)
    if not event_dir.exists():
        log.error(f"Data directory not found: {event_dir}")
        sys.exit(1)

    event_files = sorted(event_dir.glob("20??????_mbo_events.npz"))
    if not event_files:
        log.error(f"No MBO event files found in {event_dir}")
        sys.exit(1)

    log.info(f"Found {len(event_files)} episode files ({event_files[0].stem[:8]}–{event_files[-1].stem[:8]})")

    # ── Walk-forward folds ─────────────────────────────────────────────────────
    folds = build_walk_forward_folds(event_files, args.train_days, args.eval_days)
    if not folds:
        log.error(
            f"Not enough data for walk-forward: need {args.train_days + args.eval_days} days, "
            f"have {len(event_files)}"
        )
        sys.exit(1)
    log.info(f"Walk-forward folds: {len(folds)} (train={args.train_days}d, eval={args.eval_days}d)")

    # ── Environment ────────────────────────────────────────────────────────────
    pred_dir = Path(args.pred_dir) if args.pred_dir else DEFAULT_PRED_DIR
    patchtst_dir = Path(args.patchtst_dir) if args.patchtst_dir else None
    env = FIFOExecutionEnv(
        event_dir=event_dir,
        pred_dir=pred_dir,
        patchtst_pred_dir=patchtst_dir,
        rth_only=True,
        verbose=False,
    )
    log.info(f"Env: obs_dim={OBS_DIM}, n_actions={N_ACTIONS}")

    # ── Policy ─────────────────────────────────────────────────────────────────
    policy = FIFOActorCritic(
        obs_dim=OBS_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=args.hidden_dim,
    ).to(device)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log.info(f"Policy params: {n_params:,}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    # LR schedule: cosine decay over all epochs
    total_epochs = args.epochs * len(folds)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs, 1), eta_min=args.lr * 0.05
    )

    # ── Rollout buffer ─────────────────────────────────────────────────────────
    buffer = RolloutBuffer(
        capacity=args.rollout_len,
        obs_dim=OBS_DIM,
        gamma=args.gamma,
        lam=args.lam,
        device=device,
    )

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            # Quick connectivity check — don't hang on unreachable servers
            import socket
            _s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            _s.settimeout(2.0)
            try:
                _s.connect(("localhost", 5000))
                _s.close()
            except (socket.timeout, ConnectionRefusedError, OSError):
                _s.close()
                raise ConnectionError("MLflow server not reachable at localhost:5000")
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "epochs":      args.epochs,
                "lr":          args.lr,
                "rollout_len": args.rollout_len,
                "n_ppo_epochs":args.n_ppo_epochs,
                "mini_batch":  args.mini_batch,
                "gamma":       args.gamma,
                "lam":         args.lam,
                "clip_eps":    args.clip_eps,
                "entropy_coef":args.entropy_coef,
                "vf_coef":     args.vf_coef,
                "hidden_dim":  args.hidden_dim,
                "train_days":  args.train_days,
                "eval_days":   args.eval_days,
                "n_folds":     len(folds),
                "n_event_files": len(event_files),
                "device":      str(device),
            })
            log.info(f"MLflow run: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e} — continuing without MLflow")
            MLFLOW_AVAILABLE = False
    else:
        log.warning("MLflow not available — metrics logged to console only")

    # ── Training state ─────────────────────────────────────────────────────────
    best_sortino   = -float("inf")
    best_ckpt_path = output_dir / f"best_agent_{run_name}.pt"

    history: Dict[str, List] = {
        "epoch": [], "fold": [], "lr": [],
        "train_total_pnl_ticks": [], "train_sortino": [], "train_win_rate": [],
        "train_avg_pnl_ticks": [], "train_n_trades": [],
        "eval_epoch": [], "eval_sortino": [], "eval_win_rate": [],
        "eval_avg_pnl_ticks": [], "eval_avg_trades_per_day": [],
        "eval_total_pnl_ticks": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
    }

    global_epoch = 0

    # ── Walk-forward loop ──────────────────────────────────────────────────────
    for fold_info in folds:
        fold_num    = fold_info["fold"]
        train_files = fold_info["train_files"]
        eval_files  = fold_info["eval_files"]

        log.info(
            f"\n{'='*60}\n"
            f"Fold {fold_num}: train={len(train_files)}d "
            f"({train_files[0].stem[:8]}–{train_files[-1].stem[:8]}) | "
            f"eval={len(eval_files)}d ({eval_files[0].stem[:8]}–{eval_files[-1].stem[:8]})\n"
            f"{'='*60}"
        )

        # Per-fold training epochs
        for epoch in range(args.epochs):
            t0 = time.time()
            global_epoch += 1

            # Collect rollout on training data (n_workers=1: multiprocess disabled
            # until fork deadlock / spawn overhead is resolved — HC #144 TODO)
            rollout_metrics, last_value = collect_rollout(
                env, policy, buffer, train_files, device, n_workers=1
            )

            # PPO update
            update_losses = ppo_update(
                policy, optimizer, buffer, last_value,
                n_ppo_epochs=args.n_ppo_epochs,
                batch_size=args.mini_batch,
                clip_eps=args.clip_eps,
                entropy_coef=args.entropy_coef,
                vf_coef=args.vf_coef,
                max_grad_norm=args.max_grad_norm,
            )

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            elapsed = time.time() - t0

            # ── Log to history ─────────────────────────────────────────────────
            history["epoch"].append(global_epoch)
            history["fold"].append(fold_num)
            history["lr"].append(current_lr)
            history["train_total_pnl_ticks"].append(rollout_metrics.get("total_pnl_ticks", 0.0))
            history["train_sortino"].append(rollout_metrics.get("sortino", 0.0))
            history["train_win_rate"].append(rollout_metrics.get("win_rate", 0.0))
            history["train_avg_pnl_ticks"].append(rollout_metrics.get("avg_pnl_ticks", 0.0))
            history["train_n_trades"].append(rollout_metrics.get("n_trades", 0))
            history["policy_loss"].append(update_losses["policy_loss"])
            history["value_loss"].append(update_losses["value_loss"])
            history["entropy"].append(update_losses["entropy"])

            # ── Print progress ─────────────────────────────────────────────────
            if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
                log.info(
                    f"  Fold {fold_num} | Epoch {epoch+1}/{args.epochs} (global {global_epoch}) | "
                    f"{elapsed:.1f}s | LR={current_lr:.2e} | "
                    f"Sortino={rollout_metrics.get('sortino', 0.0):.3f} | "
                    f"WR={rollout_metrics.get('win_rate', 0.0):.1%} | "
                    f"AvgPnL={rollout_metrics.get('avg_pnl_ticks', 0.0):.3f}t | "
                    f"Trades={rollout_metrics.get('n_trades', 0)} | "
                    f"π_loss={update_losses['policy_loss']:.4f} | "
                    f"V_loss={update_losses['value_loss']:.4f} | "
                    f"H={update_losses['entropy']:.3f}"
                )

            # ── MLflow per-step ────────────────────────────────────────────────
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        "train/sortino":    rollout_metrics.get("sortino", 0.0),
                        "train/win_rate":   rollout_metrics.get("win_rate", 0.0),
                        "train/avg_pnl":    rollout_metrics.get("avg_pnl_ticks", 0.0),
                        "train/n_trades":   rollout_metrics.get("n_trades", 0),
                        "train/sharpe":     rollout_metrics.get("sharpe", 0.0),
                        "train/pf":         rollout_metrics.get("profit_factor", 0.0),
                        "loss/policy":      update_losses["policy_loss"],
                        "loss/value":       update_losses["value_loss"],
                        "loss/entropy":     update_losses["entropy"],
                        "lr":               current_lr,
                        "fold":             float(fold_num),
                    }, step=global_epoch)
                except Exception:
                    pass  # don't crash training on MLflow errors

        # ── Evaluate on OOT eval days (end of each fold) ─────────────────────
        log.info(f"  Evaluating on {len(eval_files)} OOT days...")
        eval_metrics = evaluate_policy(env, policy, eval_files, device)

        history["eval_epoch"].append(global_epoch)
        history["eval_sortino"].append(eval_metrics.get("sortino", 0.0))
        history["eval_win_rate"].append(eval_metrics.get("win_rate", 0.0))
        history["eval_avg_pnl_ticks"].append(eval_metrics.get("avg_pnl_ticks", 0.0))
        history["eval_avg_trades_per_day"].append(eval_metrics.get("avg_trades_per_day", 0.0))
        history["eval_total_pnl_ticks"].append(eval_metrics.get("total_pnl_ticks", 0.0))

        log.info(
            f"  EVAL Fold {fold_num}: "
            f"Sortino={eval_metrics.get('sortino', 0.0):.3f} | "
            f"Sharpe={eval_metrics.get('sharpe', 0.0):.3f} | "
            f"WR={eval_metrics.get('win_rate', 0.0):.1%} | "
            f"PF={eval_metrics.get('profit_factor', 0.0):.2f} | "
            f"AvgPnL={eval_metrics.get('avg_pnl_ticks', 0.0):.3f}t | "
            f"Trades/day={eval_metrics.get('avg_trades_per_day', 0.0):.1f} | "
            f"TotalPnL={eval_metrics.get('total_pnl_usd', 0.0):.0f}$"
        )

        # ── MLflow eval metrics ────────────────────────────────────────────────
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    "eval/sortino":    eval_metrics.get("sortino", 0.0),
                    "eval/sharpe":     eval_metrics.get("sharpe", 0.0),
                    "eval/win_rate":   eval_metrics.get("win_rate", 0.0),
                    "eval/pf":         eval_metrics.get("profit_factor", 0.0),
                    "eval/avg_pnl":    eval_metrics.get("avg_pnl_ticks", 0.0),
                    "eval/total_pnl_usd": eval_metrics.get("total_pnl_usd", 0.0),
                    "eval/trades_per_day": eval_metrics.get("avg_trades_per_day", 0.0),
                }, step=global_epoch)
            except Exception:
                pass

        # ── Save best checkpoint (by eval Sortino) ─────────────────────────────
        eval_sortino = eval_metrics.get("sortino", 0.0)
        if eval_sortino > best_sortino:
            best_sortino = eval_sortino
            torch.save({
                "fold":        fold_num,
                "epoch":       global_epoch,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "eval_metrics": eval_metrics,
                "best_sortino": best_sortino,
                "args":        vars(args),
                "run_name":    run_name,
            }, best_ckpt_path)
            log.info(f"  *** New best checkpoint: Sortino={best_sortino:.3f} → {best_ckpt_path}")
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
                except Exception:
                    pass

    # ── Final checkpoint ───────────────────────────────────────────────────────
    final_ckpt_path = output_dir / f"final_agent_{run_name}.pt"
    torch.save({
        "fold":        fold_num if folds else -1,
        "epoch":       global_epoch,
        "model_state": policy.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history":     history,
        "run_name":    run_name,
    }, final_ckpt_path)
    log.info(f"Final checkpoint saved → {final_ckpt_path}")

    # ── Save history JSON ──────────────────────────────────────────────────────
    history_path = output_dir / f"training_history_{run_name}.json"
    with open(history_path, "w") as f:
        # Convert numpy types to native Python for JSON serialization
        history_serializable = {
            k: [float(x) if hasattr(x, "item") else x for x in v]
            for k, v in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    log.info(f"Training history saved → {history_path}")

    # ── Save training curves PNG ───────────────────────────────────────────────
    plot_path = save_training_curves(history, output_dir, run_name)
    if plot_path and MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(
        f"\n{'='*60}\n"
        f"Training complete: {run_name}\n"
        f"  Epochs:         {global_epoch}\n"
        f"  Folds:          {len(folds)}\n"
        f"  Best Sortino:   {best_sortino:.3f}\n"
        f"  Best ckpt:      {best_ckpt_path}\n"
        f"  Final ckpt:     {final_ckpt_path}\n"
        f"  Training curves: {plot_path}\n"
        f"{'='*60}"
    )

    if MLFLOW_AVAILABLE and mlflow_run:
        try:
            mlflow.log_metrics({
                "summary/best_eval_sortino": best_sortino,
                "summary/total_epochs":      global_epoch,
                "summary/n_folds":           float(len(folds)),
            }, step=global_epoch)
            mlflow.end_run()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PPO Trainer for FIFO RL Execution Agent\n"
            "\n"
            "Trains a PPO agent on the FIFOExecutionEnv using sliding walk-forward\n"
            "evaluation. Reward = delta Sortino per trade (risk-adjusted, HC #100/#104).\n"
            "Cost model: 0.376 ticks commission for limits, 1.376 ticks for market orders (HC #89).\n"
            "\n"
            "Example:\n"
            "  python train_fifo_rl.py --epochs 50 --lr 3e-4\n"
            "  python train_fifo_rl.py --epochs 100 --train-days 60 --eval-days 10\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_EVENT_DIR),
        help=f"Directory with dated *_mbo_events.npz files (default: {DEFAULT_EVENT_DIR})",
    )
    p.add_argument(
        "--pred-dir",
        type=str,
        default=None,
        help=f"Directory with fold_XX_oot_predictions.npz files (default: {DEFAULT_PRED_DIR})",
    )
    p.add_argument(
        "--patchtst-dir",
        type=str,
        default=None,
        help="Directory with PatchTST fold_XX_oot_predictions.npz for confluence (HC #111)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("/home/jupiter/Lvl3Quant/output/fifo_ppo_rl")),
        help="Directory to save checkpoints, logs, and plots",
    )

    # Walk-forward
    p.add_argument(
        "--train-days", type=int, default=DEFAULT_TRAIN_DAYS,
        help=f"Training window size in days (sliding, HC #0) (default: {DEFAULT_TRAIN_DAYS})",
    )
    p.add_argument(
        "--eval-days", type=int, default=DEFAULT_EVAL_DAYS,
        help=f"OOT eval window per fold in days (default: {DEFAULT_EVAL_DAYS})",
    )

    # Training
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"PPO epochs per walk-forward fold (default: {DEFAULT_EPOCHS})",
    )
    p.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Adam learning rate (default: {DEFAULT_LR})",
    )
    p.add_argument(
        "--rollout-len", type=int, default=DEFAULT_ROLLOUT_LEN,
        help=f"Steps per PPO rollout (default: {DEFAULT_ROLLOUT_LEN})",
    )
    p.add_argument(
        "--n-ppo-epochs", type=int, default=DEFAULT_N_PPO_EPOCHS,
        help=f"PPO update passes per rollout (default: {DEFAULT_N_PPO_EPOCHS})",
    )
    p.add_argument(
        "--mini-batch", type=int, default=DEFAULT_MINI_BATCH,
        help=f"Mini-batch size for PPO (default: {DEFAULT_MINI_BATCH})",
    )

    # PPO hyperparams
    p.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Discount factor (default: {DEFAULT_GAMMA})",
    )
    p.add_argument(
        "--lam", type=float, default=DEFAULT_LAM,
        help=f"GAE lambda (default: {DEFAULT_LAM})",
    )
    p.add_argument(
        "--clip-eps", type=float, default=DEFAULT_CLIP_EPS,
        help=f"PPO clip epsilon (default: {DEFAULT_CLIP_EPS})",
    )
    p.add_argument(
        "--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF,
        help=f"Entropy bonus coefficient (default: {DEFAULT_ENTROPY_COEF})",
    )
    p.add_argument(
        "--vf-coef", type=float, default=DEFAULT_VF_COEF,
        help=f"Value function loss coefficient (default: {DEFAULT_VF_COEF})",
    )
    p.add_argument(
        "--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD,
        help=f"Gradient clipping norm (default: {DEFAULT_MAX_GRAD})",
    )

    # Architecture
    p.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden dimension of actor-critic network (default: 256)",
    )

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("FIFO PPO RL Trainer")
    log.info("=" * 60)
    log.info(f"  data_dir:    {args.data_dir}")
    log.info(f"  pred_dir:    {args.pred_dir or DEFAULT_PRED_DIR}")
    log.info(f"  output_dir:  {args.output_dir}")
    log.info(f"  epochs:      {args.epochs} per fold")
    log.info(f"  lr:          {args.lr}")
    log.info(f"  train_days:  {args.train_days}  (sliding walk-forward)")
    log.info(f"  eval_days:   {args.eval_days}")
    log.info(f"  rollout_len: {args.rollout_len}")
    log.info(f"  hidden_dim:  {args.hidden_dim}")
    log.info(f"  torch:       {torch.__version__}")
    log.info(f"  mlflow:      {'yes' if MLFLOW_AVAILABLE else 'NO — install mlflow'}")
    log.info(f"  matplotlib:  {'yes' if MATPLOTLIB_AVAILABLE else 'NO — install matplotlib'}")
    log.info("=" * 60)

    train(args)
