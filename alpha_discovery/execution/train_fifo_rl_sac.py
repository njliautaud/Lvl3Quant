#!/usr/bin/env python3
"""
SAC Trainer for FIFO RL Execution Agent
========================================

Trains a Soft Actor-Critic (SAC) agent on the FIFOExecutionEnv. Off-policy,
entropy-regularized, and more sample-efficient than PPO for our limited data regime.

Key design choices:
- Discrete SAC: actor outputs categorical distribution over 7 actions;
  Q-networks output per-action Q-values (no continuous action Gaussian needed).
- Twin Q-networks (Q1, Q2): prevents Q-value overestimation (TD3 trick).
- Soft target Q-networks: polyak-averaged copies updated every step.
- Automatic entropy tuning: alpha (temperature) is learned to match target entropy.
- Replay buffer: off-policy experience (500k transitions); enables reuse of past data.
- Walk-forward evaluation: same fold structure as PPO (HC #0: SLIDING only).
- Same eval metrics: Sortino, Sharpe, PF, WR (HC #69, #100).
- Same cost model: passive limit = 0.376 ticks, market = 1.376 ticks (HC #89).
- NO short-side bias (HC #104); NO hardcoded decay penalties (HC #115).

Checkpoint compatibility with rl_execution_agent.py:
  Best checkpoint saved as {"model_state": actor.state_dict(), ...}
  The SAC actor has the EXACT same backbone as FIFOActorCritic in train_fifo_rl.py,
  so checkpoints are drop-in compatible (load actor weights only; no critic needed).

Usage:
  python train_fifo_rl_sac.py --epochs 100 --lr 3e-4 --data-dir /path/to/mbo_events
  python train_fifo_rl_sac.py --help

HC compliance:
  - HC #0:   sliding walk-forward (no expanding window)
  - HC #69:  primary metrics = Sharpe, Sortino, PF, WR (not raw P&L)
  - HC #89:  cost = 0.376 ticks commission for limits; 1.376 for market orders
  - HC #100: risk-adjusted reward
  - HC #104: symmetric reward — no short-side bias
  - HC #105: training curve PNG saved
  - HC #115: no hardcoded decay penalties; env reward used as-is
  - HC #117: SAC implementation requested

Author: Claude (Infrastructure Builder)
Date: 2026-05-03
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

# ── WMIC/headless launch fix ─────────────────────────────────────────────────
# When launched via WMIC on Windows, stdout/stderr may be None or broken.
# Redirect to devnull BEFORE any library imports to prevent hangs.
if sys.stdout is None or (hasattr(sys.stdout, 'fileno') and
        not hasattr(sys.stdout, 'write')):
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None or (hasattr(sys.stderr, 'fileno') and
        not hasattr(sys.stderr, 'write')):
    sys.stderr = open(os.devnull, 'w')

# Also handle the case where write() exists but fails on fileno()
try:
    sys.stdout.fileno()
except (OSError, AttributeError, ValueError):
    sys.stdout = open(os.devnull, 'w')
try:
    sys.stderr.fileno()
except (OSError, AttributeError, ValueError):
    sys.stderr = open(os.devnull, 'w')

import argparse
import json
import logging
import math
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
    COMMISSION_RT_TICKS,        # 0.376 — the ONLY cost (HC #231(A))
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("train_fifo_rl_sac")

# ── Constants ──────────────────────────────────────────────────────────────────
N_ACTIONS         = 7          # discrete action space (matches env)
MLFLOW_EXPERIMENT = "FIFO_RL_SAC_Execution"

# SAC defaults (overridable via CLI)
DEFAULT_EPOCHS           = 100
DEFAULT_LR               = 3e-4
DEFAULT_GAMMA            = 0.99
DEFAULT_TAU              = 0.005      # polyak averaging rate for target nets
DEFAULT_ALPHA_INIT       = 0.2        # initial entropy temperature
DEFAULT_BUFFER_SIZE      = 500_000    # replay buffer capacity
DEFAULT_BATCH_SIZE       = 256
DEFAULT_WARMUP_STEPS     = 1_000      # fill buffer before training starts
DEFAULT_UPDATES_PER_STEP = 1          # gradient updates per env step
DEFAULT_TRAIN_DAYS       = 40         # walk-forward train window
DEFAULT_EVAL_DAYS        = 5          # walk-forward eval window


# ══════════════════════════════════════════════════════════════════════════════
# Network Architectures
# ══════════════════════════════════════════════════════════════════════════════

class SACActorNet(nn.Module):
    """
    SAC Actor for discrete action spaces.

    Outputs a categorical distribution over N_ACTIONS.
    Architecture exactly matches FIFOActorCritic's shared backbone + actor head
    from train_fifo_rl.py — so checkpoints are compatible with rl_execution_agent.py.

    Input (48) → LayerNorm
              → Linear(48, hidden) → GELU → LayerNorm
              → Linear(hidden, hidden) → GELU → LayerNorm
              → Linear(hidden, hidden) → GELU → LayerNorm  [= shared backbone]
              → Linear(hidden, 128) → GELU → Linear(128, 7 logits)  [= actor head]

    The actor in SAC outputs logits → Categorical distribution.
    Entropy is computed analytically from the categorical probs (no reparameterization
    trick needed for discrete SAC — we use the full distribution directly).
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(obs_dim)

        # Shared backbone (must match FIFOActorCritic.shared exactly)
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

        # Actor head (must match FIFOActorCritic.actor exactly)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init — standard for RL, keeps initial policy near-uniform."""
        for module in self.shared.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        # Actor output: small gain → near-uniform initial policy
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        probs     : (B, N_ACTIONS) — action probabilities (softmax)
        log_probs : (B, N_ACTIONS) — log action probabilities (log_softmax, numerically stable)
        """
        x = self.input_norm(obs)
        x = self.shared(x)
        logits = self.actor(x)
        probs     = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)  # more stable than log(softmax)
        return probs, log_probs

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action and compute entropy for SAC objective.

        For discrete SAC, we can compute the EXACT expected value over the full
        distribution (no need for reparameterization sampling):
          E_{a~π}[Q(s,a)] = Σ_a π(a|s) * Q(s,a)
          H[π] = -Σ_a π(a|s) * log π(a|s)

        Returns
        -------
        action      : (B,) sampled action index (for env stepping)
        probs       : (B, N_ACTIONS) full distribution (for SAC update)
        log_probs   : (B, N_ACTIONS) log probs (for SAC update)
        """
        probs, log_probs = self.forward(obs)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, probs, log_probs

    def get_action_greedy(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy (argmax) action — used for evaluation."""
        probs, _ = self.forward(obs)
        return probs.argmax(dim=-1)


class SACQNet(nn.Module):
    """
    SAC Q-Network for discrete actions.

    Outputs Q(s, a) for ALL actions simultaneously — no need to pass action as input.
    This is the standard approach for discrete SAC and avoids N forward passes.

    Architecture mirrors the actor backbone with a separate output head:
      Input (48) → LayerNorm
                → Linear(48, hidden) → GELU → LayerNorm
                → Linear(hidden, hidden) → GELU → LayerNorm
                → Linear(hidden, hidden) → GELU → LayerNorm
                → Linear(hidden, 128) → GELU → Linear(128, N_ACTIONS)

    We instantiate this twice (Q1, Q2) for the twin Q-network trick.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(obs_dim)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_actions),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        # Final layer: unit gain for Q-values
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Returns Q-values for all actions.

        Parameters
        ----------
        obs : (B, obs_dim)

        Returns
        -------
        q_values : (B, N_ACTIONS)
        """
        x = self.input_norm(obs)
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer for off-policy SAC.

    Stores (obs, action, reward, next_obs, done) tuples.
    Pre-allocated as numpy arrays for fast indexing.

    Capacity: 500k transitions by default. At ~2000 env steps/sec and
    ~20k steps/day, we accumulate ~250k transitions per training day,
    so 500k holds about 2 days of transitions before oldest data overwrites.
    """

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.ptr      = 0
        self.size     = 0

        # Pre-allocate (avoids repeated realloc during training)
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((capacity,), dtype=np.int64)
        self.rewards  = np.zeros((capacity,), dtype=np.float32)
        self.dones    = np.zeros((capacity,), dtype=np.float32)

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Add a single transition. Overwrites oldest when full (circular)."""
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device:     torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a random mini-batch as tensors on `device`.

        Returns
        -------
        dict with keys: obs, actions, rewards, next_obs, dones
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.tensor(self.obs[idx],      device=device),
            "actions":  torch.tensor(self.actions[idx],  device=device),
            "rewards":  torch.tensor(self.rewards[idx],  device=device),
            "next_obs": torch.tensor(self.next_obs[idx], device=device),
            "dones":    torch.tensor(self.dones[idx],    device=device),
        }

    def push_bulk(
        self,
        obs_arr:      np.ndarray,   # (N, obs_dim)
        actions_arr:  np.ndarray,   # (N,)
        rewards_arr:  np.ndarray,   # (N,)
        next_obs_arr: np.ndarray,   # (N, obs_dim)
        dones_arr:    np.ndarray,   # (N,)
    ) -> None:
        """Add N transitions at once. Much faster than N individual push() calls."""
        n = len(obs_arr)
        if n == 0:
            return

        # If more transitions than capacity, only keep the last `capacity` ones
        if n > self.capacity:
            offset = n - self.capacity
            obs_arr      = obs_arr[offset:]
            actions_arr  = actions_arr[offset:]
            rewards_arr  = rewards_arr[offset:]
            next_obs_arr = next_obs_arr[offset:]
            dones_arr    = dones_arr[offset:]
            n = self.capacity

        # How many fit before we wrap around?
        space_before_wrap = self.capacity - self.ptr
        if n <= space_before_wrap:
            # No wrap needed
            self.obs[self.ptr:self.ptr + n]      = obs_arr
            self.actions[self.ptr:self.ptr + n]   = actions_arr
            self.rewards[self.ptr:self.ptr + n]   = rewards_arr
            self.next_obs[self.ptr:self.ptr + n]  = next_obs_arr
            self.dones[self.ptr:self.ptr + n]     = dones_arr
        else:
            # Split into two chunks (before and after wrap)
            first = space_before_wrap
            second = n - first
            self.obs[self.ptr:]      = obs_arr[:first]
            self.actions[self.ptr:]  = actions_arr[:first]
            self.rewards[self.ptr:]  = rewards_arr[:first]
            self.next_obs[self.ptr:] = next_obs_arr[:first]
            self.dones[self.ptr:]    = dones_arr[:first]

            self.obs[:second]      = obs_arr[first:]
            self.actions[:second]  = actions_arr[first:]
            self.rewards[:second]  = rewards_arr[first:]
            self.next_obs[:second] = next_obs_arr[first:]
            self.dones[:second]    = dones_arr[first:]

        self.ptr  = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def is_ready(self, warmup: int) -> bool:
        """True once we have enough transitions to start training."""
        return self.size >= warmup


# ══════════════════════════════════════════════════════════════════════════════
# Parallel Episode Collection (multicore SAC)
# ══════════════════════════════════════════════════════════════════════════════

def _worker_init():
    """
    Initializer for each Pool worker process.

    Sets torch to single-threaded to avoid fork+OpenMP deadlocks.
    With fork context, the parent's OpenMP/MKL thread state is inherited
    but the threads themselves don't exist in the child — causing deadlock
    when torch tries to use them. Setting threads=1 avoids this entirely.
    """
    import torch
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _worker_collect_episodes(args_tuple):
    """
    Worker function for parallel SAC experience collection.

    Each worker runs its own FIFOExecutionEnv on a subset of training files,
    collecting transitions into pre-allocated numpy arrays. The actor network
    is copied (state_dict) into each worker — no shared memory needed.

    Designed for Linux fork context (fast process creation, COW memory).
    Each worker is fully independent: own env, own actor copy, own data arrays.

    Parameters (packed as tuple for Pool.map compatibility)
    -------------------------------------------------------
    env_kwargs       : dict to construct FIFOExecutionEnv
    actor_state_dict : CPU state dict for SACActorNet
    file_paths_str   : list of str paths to episode files for this worker
    obs_dim          : int
    n_actions        : int
    hidden_dim       : int
    worker_id        : int (for logging)

    Returns
    -------
    dict with numpy arrays: obs, actions, rewards, next_obs, dones, trade_pnls, n_steps
    """
    import torch
    import numpy as np
    from pathlib import Path

    # Ensure single-threaded torch in worker (belt + suspenders with _worker_init)
    torch.set_num_threads(1)

    env_kwargs, actor_state_dict, file_paths_str, obs_dim, n_actions, hidden_dim, worker_id = args_tuple

    # Reconstruct env in this worker process
    env = FIFOExecutionEnv(**env_kwargs)

    # Reconstruct actor (CPU only — workers don't need GPU)
    actor = SACActorNet(obs_dim, n_actions, hidden_dim)
    actor.load_state_dict(actor_state_dict)
    actor.eval()
    device = torch.device("cpu")

    file_paths = [Path(f) for f in file_paths_str]

    # Pre-allocate generous buffers (we'll trim at the end)
    # Typical: ~20k-50k events per file, so total ~= n_files * 30k
    est_steps = len(file_paths) * 50_000
    obs_buf      = np.zeros((est_steps, obs_dim), dtype=np.float32)
    next_obs_buf = np.zeros((est_steps, obs_dim), dtype=np.float32)
    act_buf      = np.zeros(est_steps, dtype=np.int64)
    rew_buf      = np.zeros(est_steps, dtype=np.float32)
    done_buf     = np.zeros(est_steps, dtype=np.float32)

    trade_pnls = []
    idx = 0

    for fp in file_paths:
        try:
            obs = env.reset(episode_file=fp)
        except Exception as e:
            continue  # skip corrupt files

        done = False
        while not done:
            # Grow buffers if needed (rare — only if estimate was too low)
            if idx >= len(obs_buf):
                new_size = len(obs_buf) * 2
                obs_buf      = np.resize(obs_buf,      (new_size, obs_dim))
                next_obs_buf = np.resize(next_obs_buf,  (new_size, obs_dim))
                act_buf      = np.resize(act_buf,       new_size)
                rew_buf      = np.resize(rew_buf,       new_size)
                done_buf     = np.resize(done_buf,      new_size)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = actor.sample(obs_t)
            action_np = int(action.item())

            next_obs, reward, d, info = env.step(action_np)

            obs_buf[idx]      = obs
            next_obs_buf[idx] = next_obs
            act_buf[idx]      = action_np
            rew_buf[idx]      = float(reward)
            done_buf[idx]     = float(d)
            idx += 1

            obs  = next_obs
            done = d

        # Collect trade PnLs from this episode
        for t in env._trade_history:
            trade_pnls.append(t.pnl_ticks)

    return {
        "obs":      obs_buf[:idx],
        "next_obs": next_obs_buf[:idx],
        "actions":  act_buf[:idx],
        "rewards":  rew_buf[:idx],
        "dones":    done_buf[:idx],
        "trade_pnls": trade_pnls,
        "n_steps":  idx,
        "worker_id": worker_id,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Data Split
# ══════════════════════════════════════════════════════════════════════════════

def build_walk_forward_folds(
    event_files: List[Path],
    train_days:  int,
    eval_days:   int,
) -> List[Dict]:
    """
    Build sliding walk-forward folds (HC #0: SLIDING, not expanding).

    Each fold:
      train : files[i : i + train_days]
      eval  : files[i + train_days : i + train_days + eval_days]

    Slides by eval_days so there is no overlap in eval sets.

    Parameters
    ----------
    event_files : chronologically sorted list of episode files
    train_days  : number of days in training window
    eval_days   : number of OOT eval days per fold

    Returns
    -------
    list of dicts with keys: 'fold', 'train_files', 'eval_files'
    """
    folds  = []
    n      = len(event_files)
    window = train_days + eval_days
    i      = 0
    fold_n = 0

    while i + window <= n:
        folds.append({
            "fold":        fold_n,
            "train_files": event_files[i : i + train_days],
            "eval_files":  event_files[i + train_days : i + train_days + eval_days],
        })
        i      += eval_days  # SLIDING: advance by eval window only (HC #0)
        fold_n += 1

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
    n    = len(pnls)

    win_rate = float((pnls > 0).mean())
    mean_pnl = float(pnls.mean())
    std_pnl  = float(pnls.std()) if n > 1 else 1.0

    # Sharpe (per-trade; √n scaling for approximate annualization)
    sharpe = mean_pnl / (std_pnl + 1e-8) * math.sqrt(max(n, 1))

    # Sortino (downside deviation only, MAR = 0)
    downside = pnls[pnls < 0]
    if len(downside) == 0:
        sortino = mean_pnl * 10.0
    else:
        dd      = float(np.sqrt(np.mean(downside ** 2)))
        sortino = mean_pnl / (dd + 1e-8)

    # Profit factor
    gross_wins   = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_losses = abs(float(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-8
    pf = gross_wins / gross_losses

    return {
        "n_trades":        n,
        "win_rate":        win_rate,
        "sharpe":          sharpe,
        "sortino":         sortino,
        "profit_factor":   pf,
        "avg_pnl_ticks":   mean_pnl,
        "total_pnl_ticks": float(pnls.sum()),
        "total_pnl_usd":   float(pnls.sum()) * TICK_VALUE,
        "trades_per_step": n / max(n_steps, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Soft (Polyak) Target Network Update
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def polyak_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    """
    Soft update target network parameters towards online network.

    θ_target ← τ * θ_online + (1-τ) * θ_target

    At tau=0.005 (default), the target moves 0.5% toward online each step.
    This provides a stable training signal (avoids oscillation from hard copies).
    """
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(1.0 - tau)
        p_target.data.add_(tau * p_online.data)


# ══════════════════════════════════════════════════════════════════════════════
# SAC Update Step
# ══════════════════════════════════════════════════════════════════════════════

def sac_update(
    actor:          SACActorNet,
    q1:             SACQNet,
    q2:             SACQNet,
    q1_target:      SACQNet,
    q2_target:      SACQNet,
    log_alpha:      torch.Tensor,       # scalar log(alpha), requires_grad=True
    target_entropy: float,
    buffer:         ReplayBuffer,
    actor_opt:      torch.optim.Optimizer,
    q1_opt:         torch.optim.Optimizer,
    q2_opt:         torch.optim.Optimizer,
    alpha_opt:      torch.optim.Optimizer,
    batch_size:     int,
    gamma:          float,
    tau:            float,
    device:         torch.device,
) -> Dict[str, float]:
    """
    One SAC gradient update step using a mini-batch from the replay buffer.

    Discrete SAC update (Christodoulou 2019, "Soft Actor-Critic for Discrete Action Settings"):

    1. Q-network update (Bellman backup):
         y = r + γ * (1-done) * Σ_a' π(a'|s') * [min(Q1_target(s',a'), Q2_target(s',a')) - α * log π(a'|s')]
         L_Q = MSE(Q(s,a), y)  for both Q1, Q2

    2. Actor update:
         L_π = E_s [ Σ_a π(a|s) * (α * log π(a|s) - min(Q1(s,a), Q2(s,a))) ]
         Note: this is the FULL expected value over the distribution, not a sample estimate.
         More stable for discrete SAC than single-sample estimates.

    3. Temperature (alpha) update (automatic entropy tuning):
         L_α = E_s [ -α * (Σ_a π(a|s) * log π(a|s) + target_entropy) ]
             = E_s [ α * (H[π(·|s)] - target_entropy) ]
         If entropy > target_entropy: increase alpha (more exploration)
         If entropy < target_entropy: decrease alpha (less exploration)

    4. Soft update target networks: θ_target ← τ*θ + (1-τ)*θ_target

    Returns
    -------
    dict of loss values for logging
    """
    batch = buffer.sample(batch_size, device)
    obs_b      = batch["obs"]        # (B, obs_dim)
    act_b      = batch["actions"]    # (B,) int64
    rew_b      = batch["rewards"]    # (B,)
    next_obs_b = batch["next_obs"]   # (B, obs_dim)
    done_b     = batch["dones"]      # (B,)

    alpha = log_alpha.exp().item()   # current temperature (detached for display/losses)

    # ── 1. Q-network targets ───────────────────────────────────────────────────
    with torch.no_grad():
        # Get next action distribution from CURRENT actor (not target actor)
        next_probs, next_log_probs = actor.forward(next_obs_b)  # (B, N_ACTIONS) each

        # Target Q-values for next state (twin — take min to reduce overestimation)
        next_q1_vals = q1_target(next_obs_b)   # (B, N_ACTIONS)
        next_q2_vals = q2_target(next_obs_b)   # (B, N_ACTIONS)
        next_q_min   = torch.min(next_q1_vals, next_q2_vals)  # (B, N_ACTIONS)

        # Discrete SAC Bellman target:
        # V(s') = Σ_{a'} π(a'|s') * [Q_min(s',a') - α * log π(a'|s')]
        # This is the SOFT value function (entropy-augmented)
        next_v = (next_probs * (next_q_min - log_alpha.exp() * next_log_probs)).sum(dim=-1)  # (B,)

        # Bellman backup: y = r + γ(1-done) * V(s')
        y = rew_b + gamma * (1.0 - done_b) * next_v  # (B,)

    # ── Current Q-values at observed (s, a) ───────────────────────────────────
    # Q1 and Q2 output per-action values; we gather the value at the taken action
    q1_all = q1(obs_b)  # (B, N_ACTIONS)
    q2_all = q2(obs_b)  # (B, N_ACTIONS)
    q1_a   = q1_all.gather(1, act_b.unsqueeze(1)).squeeze(1)  # (B,)
    q2_a   = q2_all.gather(1, act_b.unsqueeze(1)).squeeze(1)  # (B,)

    # Q losses (MSE to Bellman target)
    q1_loss = F.mse_loss(q1_a, y)
    q2_loss = F.mse_loss(q2_a, y)

    # Update Q1
    q1_opt.zero_grad()
    q1_loss.backward()
    nn.utils.clip_grad_norm_(q1.parameters(), 1.0)
    q1_opt.step()

    # Update Q2
    q2_opt.zero_grad()
    q2_loss.backward()
    nn.utils.clip_grad_norm_(q2.parameters(), 1.0)
    q2_opt.step()

    # ── 2. Actor update ────────────────────────────────────────────────────────
    # Recompute actor distribution (fresh forward pass, not cached)
    probs, log_probs = actor.forward(obs_b)  # (B, N_ACTIONS)

    # Recompute Q-values (detach — we only want gradients through the actor)
    with torch.no_grad():
        q1_vals = q1(obs_b)  # (B, N_ACTIONS)
        q2_vals = q2(obs_b)  # (B, N_ACTIONS)
        q_min   = torch.min(q1_vals, q2_vals)  # (B, N_ACTIONS)

    # Actor loss: maximize E[Q - alpha*log_pi] over the full distribution
    # = minimize E[alpha*log_pi - Q]
    # Summing over actions (full expectation, not single-sample Monte Carlo)
    inside_sum = probs * (log_alpha.exp() * log_probs - q_min)  # (B, N_ACTIONS)
    actor_loss = inside_sum.sum(dim=-1).mean()                  # scalar

    actor_opt.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_opt.step()

    # ── 3. Alpha (temperature) update ─────────────────────────────────────────
    # Automatic entropy tuning: adjust alpha so entropy tracks target_entropy.
    # Using the full distribution entropy for a stable gradient signal.
    with torch.no_grad():
        # Current policy entropy H[π(·|s)] = -Σ_a π(a|s) log π(a|s)
        current_entropy = -(probs * log_probs).sum(dim=-1)  # (B,)

    # Alpha loss: if H > target_H, increase alpha; if H < target_H, decrease alpha
    # log_alpha is the trainable parameter (numerically stable)
    alpha_loss = -(log_alpha * (current_entropy - target_entropy).detach()).mean()

    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    # ── 4. Soft update target networks ────────────────────────────────────────
    polyak_update(q1, q1_target, tau)
    polyak_update(q2, q2_target, tau)

    return {
        "q1_loss":      q1_loss.item(),
        "q2_loss":      q2_loss.item(),
        "actor_loss":   actor_loss.item(),
        "alpha_loss":   alpha_loss.item(),
        "alpha":        log_alpha.exp().item(),
        "entropy":      float(current_entropy.mean().item()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(
    env:        FIFOExecutionEnv,
    actor:      SACActorNet,
    eval_files: List[Path],
    device:     torch.device,
) -> Dict[str, float]:
    """
    Run greedy (argmax) policy over eval files and compute metrics.

    Uses argmax action selection (no entropy sampling) for clean OOT evaluation.
    Returns risk-adjusted metrics (Sortino, Sharpe, PF, WR) — HC #69.
    """
    actor.eval()
    all_trade_pnls: List[float] = []
    trades_per_day: List[int]   = []
    total_steps = 0

    for ep_file in eval_files:
        try:
            obs = env.reset(episode_file=ep_file)
        except Exception as _e:
            log.warning(f"evaluate_policy: skipping corrupt file {ep_file.name} — {_e}")
            continue
        done  = False
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_np = int(actor.get_action_greedy(obs_t).item())
            obs, _, done, _ = env.step(action_np)
            steps += 1

        metrics = env.get_episode_metrics()
        for t in env._trade_history:
            all_trade_pnls.append(t.pnl_ticks)
        trades_per_day.append(metrics["n_trades"])
        total_steps += steps

    eval_metrics = compute_episode_metrics(all_trade_pnls, total_steps)
    eval_metrics["avg_trades_per_day"] = float(np.mean(trades_per_day)) if trades_per_day else 0.0
    eval_metrics["n_eval_days"]        = len(eval_files)
    return eval_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def save_training_curves(
    history:    Dict[str, List],
    output_dir: Path,
    run_name:   str,
) -> Optional[Path]:
    """
    Save a 6-panel training curve PNG (HC #105).

    Panels: cumulative P&L, Sortino, win rate, avg P&L per trade, trades/day,
            SAC-specific losses (Q loss, actor loss, alpha).

    Both train (solid) and eval (dashed) lines where available.

    Returns path to saved PNG, or None if matplotlib unavailable.
    """
    if not MATPLOTLIB_AVAILABLE:
        log.warning("matplotlib not available — skipping training curve plot")
        return None

    fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
    fig.suptitle(f"FIFO SAC Training — {run_name}", fontsize=14, fontweight="bold")

    epochs     = history.get("epoch", [])
    eval_epochs= history.get("eval_epoch", [])

    # Panel definitions: (train_key, eval_key, ylabel, color)
    panels = [
        ("train_total_pnl_ticks", "eval_total_pnl_ticks",     "Cumulative P&L (ticks)",        "tab:blue"),
        ("train_sortino",         "eval_sortino",              "Sortino Ratio",                  "tab:orange"),
        ("train_win_rate",        "eval_win_rate",             "Win Rate",                       "tab:green"),
        ("train_avg_pnl_ticks",   "eval_avg_pnl_ticks",       "Avg P&L per Trade (ticks)",       "tab:red"),
        ("train_n_trades",        "eval_avg_trades_per_day",  "Trades / Day",                   "tab:purple"),
        ("q1_loss",               None,                        "SAC Losses (Q1, Actor, Alpha)",   "tab:brown"),
    ]

    for ax, (train_key, eval_key, ylabel, color) in zip(axes, panels):
        if train_key == "q1_loss":
            # Special panel: overlay Q1, actor, alpha losses
            for loss_key, lcolor, label in [
                ("q1_loss", "tab:brown", "Q loss"),
                ("actor_loss", "tab:olive", "Actor loss"),
                ("alpha", "tab:pink", "Alpha"),
            ]:
                if loss_key in history and history[loss_key]:
                    ax.plot(epochs[:len(history[loss_key])], history[loss_key],
                            color=lcolor, alpha=0.8, linewidth=1.2, label=label)
        else:
            if train_key in history and history[train_key]:
                ax.plot(epochs[:len(history[train_key])], history[train_key],
                        color=color, alpha=0.8, linewidth=1.5, label="train")
            if eval_key and eval_key in history and history[eval_key]:
                ax.plot(eval_epochs[:len(history[eval_key])], history[eval_key],
                        color=color, alpha=1.0, linewidth=2.0, linestyle="--",
                        label="eval", marker="o", markersize=4)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        if any(kw in ylabel.lower() for kw in ("pnl", "sortino")):
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
    Main SAC training loop with walk-forward evaluation.

    Flow per epoch (within each fold):
    1. Step through all training files collecting transitions into replay buffer.
    2. After each env step, if buffer is ready: perform `updates_per_step` SAC updates.
    3. After completing training files for the epoch, evaluate on OOT eval days.
    4. Save best checkpoint by eval Sortino (HC #69).
    5. Log all metrics to MLflow.
    6. Save training curve PNG after all folds.

    Key SAC details:
    - Warmup: first `warmup_steps` env steps only collect data (no gradient updates).
    - Temperature alpha is auto-tuned toward target_entropy = -0.98 * log(1/N_ACTIONS).
    - Twin Q-networks Q1, Q2 with separate optimizers.
    - Target networks Q1_target, Q2_target soft-updated via polyak averaging.
    - Replay buffer size: 500k. Old transitions overwritten in circular fashion.
    """
    # ── Setup ──────────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Debug checkpoint file for diagnosing WMIC hangs
    _dbg_ckpt = output_dir / "_train_debug.log"
    with open(str(_dbg_ckpt), "w") as _dc:
        _dc.write("train() entered\n")

    run_name = datetime.now().strftime("sac_%Y%m%d_%H%M%S")

    with open(str(_dbg_ckpt), "a") as _dc:
        _dc.write(f"run_name={run_name}\n")

    # Add file handler (Windows Start-Process compatibility — HC #105)
    _log_file = output_dir / f"{run_name}.log"
    _fh = logging.FileHandler(str(_log_file), mode="w")
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)
    logging.getLogger("fifo_rl_env").addHandler(_fh)

    with open(str(_dbg_ckpt), "a") as _dc:
        _dc.write("FileHandler ready, calling first log.info\n")

    log.info(f"Run: {run_name}")

    def _ckpt(msg):
        with open(str(_dbg_ckpt), "a") as _dc:
            _dc.write(f"{msg}\n")

    _ckpt("first log.info OK")
    log.info(f"Output dir: {output_dir}")
    log.info(f"Log file: {_log_file}")

    _ckpt("pre-device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    _ckpt(f"device={device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    event_dir = Path(args.data_dir)
    if not event_dir.exists():
        log.error(f"Data directory not found: {event_dir}")
        sys.exit(1)

    _ckpt("pre-glob")
    event_files = sorted(event_dir.glob("20??????_mbo_events.npz"))
    _ckpt(f"glob done: {len(event_files)} files")
    if not event_files:
        log.error(f"No MBO event files found in {event_dir}")
        sys.exit(1)

    log.info(
        f"Found {len(event_files)} episode files "
        f"({event_files[0].stem[:8]}–{event_files[-1].stem[:8]})"
    )

    # ── Walk-forward folds ─────────────────────────────────────────────────────
    _ckpt("pre-wf-folds")
    folds = build_walk_forward_folds(event_files, args.train_days, args.eval_days)
    if not folds:
        log.error(
            f"Not enough data for walk-forward: need "
            f"{args.train_days + args.eval_days} days, have {len(event_files)}"
        )
        sys.exit(1)
    log.info(
        f"Walk-forward folds: {len(folds)} "
        f"(train={args.train_days}d, eval={args.eval_days}d)"
    )

    # ── Environment ────────────────────────────────────────────────────────────
    _ckpt(f"pre-env pred_dir={args.pred_dir}")
    pred_dir     = Path(args.pred_dir)     if args.pred_dir     else DEFAULT_PRED_DIR
    patchtst_dir = Path(args.patchtst_dir) if args.patchtst_dir else None
    env = FIFOExecutionEnv(
        event_dir        = event_dir,
        pred_dir         = pred_dir,
        patchtst_pred_dir= patchtst_dir,
        rth_only         = True,
        verbose          = False,
    )
    _ckpt("env created")
    log.info(f"Env: obs_dim={OBS_DIM}, n_actions={N_ACTIONS}")

    # ── Networks ───────────────────────────────────────────────────────────────
    _ckpt("pre-networks")
    hidden_dim = args.hidden_dim

    actor    = SACActorNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q1       = SACQNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q2       = SACQNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)

    # Target networks: copies of Q1, Q2 (NOT updated by gradients — polyak only)
    q1_target = SACQNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q2_target = SACQNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    # Freeze target networks (no gradient flow)
    for p in q1_target.parameters():
        p.requires_grad = False
    for p in q2_target.parameters():
        p.requires_grad = False

    n_actor_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    n_q_params     = sum(p.numel() for p in q1.parameters()    if p.requires_grad)
    log.info(f"Actor params: {n_actor_params:,}  |  Q-net params (each): {n_q_params:,}")

    # ── Optimizers ─────────────────────────────────────────────────────────────
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)
    q1_opt    = torch.optim.Adam(q1.parameters(),    lr=args.lr, eps=1e-5)
    q2_opt    = torch.optim.Adam(q2.parameters(),    lr=args.lr, eps=1e-5)

    # ── Entropy temperature alpha (automatic tuning) ───────────────────────────
    # Target entropy: set to -0.98 * log(1/N_ACTIONS) = 0.98 * log(N_ACTIONS)
    # This means we want to keep ~98% of the max categorical entropy.
    # log(7) ≈ 1.946 → target ≈ 1.907 nats
    target_entropy = -args.target_entropy_ratio * math.log(1.0 / N_ACTIONS)
    log.info(
        f"Target entropy: {target_entropy:.4f} nats "
        f"(= {args.target_entropy_ratio:.2f} * log({N_ACTIONS}) = "
        f"{args.target_entropy_ratio:.2f} * {math.log(N_ACTIONS):.4f})"
    )

    # log_alpha is the learnable parameter (more stable than directly learning alpha)
    log_alpha = torch.tensor(
        math.log(args.alpha_init),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr, eps=1e-5)

    # ── Replay buffer ──────────────────────────────────────────────────────────
    _ckpt("pre-buffer")
    buffer = ReplayBuffer(capacity=args.buffer_size, obs_dim=OBS_DIM)
    _ckpt("buffer created")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    _ckpt("pre-mlflow")
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            _tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(_tracking_uri)
            log.info(f"MLflow tracking URI: {_tracking_uri}")
            # Quick connectivity check (5s timeout) to avoid hanging
            import urllib.request
            urllib.request.urlopen(_tracking_uri, timeout=5)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "algorithm":           "SAC",
                "epochs":              args.epochs,
                "lr":                  args.lr,
                "gamma":               args.gamma,
                "tau":                 args.tau,
                "alpha_init":          args.alpha_init,
                "target_entropy_ratio":args.target_entropy_ratio,
                "target_entropy":      target_entropy,
                "buffer_size":         args.buffer_size,
                "batch_size":          args.batch_size,
                "warmup_steps":        args.warmup_steps,
                "updates_per_step":    args.updates_per_step,
                "hidden_dim":          hidden_dim,
                "train_days":          args.train_days,
                "eval_days":           args.eval_days,
                "n_folds":             len(folds),
                "n_event_files":       len(event_files),
                "device":              str(device),
                "obs_dim":             OBS_DIM,
                "n_actions":           N_ACTIONS,
                "n_workers":           args.n_workers,
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
        "epoch": [], "fold": [], "alpha": [],
        "train_total_pnl_ticks": [], "train_sortino": [], "train_win_rate": [],
        "train_avg_pnl_ticks": [], "train_n_trades": [],
        "eval_epoch": [], "eval_sortino": [], "eval_win_rate": [],
        "eval_avg_pnl_ticks": [], "eval_avg_trades_per_day": [],
        "eval_total_pnl_ticks": [],
        "q1_loss": [], "q2_loss": [], "actor_loss": [], "alpha_loss": [],
        "entropy": [],
    }

    global_epoch    = 0
    total_env_steps = 0
    fold_num        = 0  # will be updated in loop (needed in final ckpt save)

    _ckpt("pre-training-loop")
    # ── Walk-forward loop ──────────────────────────────────────────────────────
    for fold_info in folds:
        fold_num    = fold_info["fold"]
        train_files = fold_info["train_files"]
        eval_files  = fold_info["eval_files"]

        log.info(
            f"\n{'='*60}\n"
            f"Fold {fold_num}: train={len(train_files)}d "
            f"({train_files[0].stem[:8]}–{train_files[-1].stem[:8]}) | "
            f"eval={len(eval_files)}d "
            f"({eval_files[0].stem[:8]}–{eval_files[-1].stem[:8]})\n"
            f"{'='*60}"
        )

        # ── Prepare parallel collection infrastructure ──────────────────
        n_workers = getattr(args, 'n_workers', 1)
        use_parallel = (n_workers > 1 and sys.platform == "linux"
                        and len(train_files) >= n_workers)

        if use_parallel:
            # Prepare env kwargs for worker processes (must be picklable)
            env_kwargs = {
                "event_dir":     str(event_dir),
                "pred_dir":      str(pred_dir),
                "patchtst_pred_dir": str(patchtst_dir) if patchtst_dir else None,
                "pred_window":   env.pred_window,
                "pred_stride":   env.pred_stride,
                "max_steps_per_episode": env.max_steps,
                "rth_only":      env.rth_only,
                "verbose":       False,
            }
            log.info(
                f"Parallel collection enabled: {n_workers} workers on Linux "
                f"(fork context, {len(train_files)} train files)"
            )

        # Per-fold training epochs
        for epoch in range(args.epochs):
            t0 = time.time()
            global_epoch += 1

            actor.train()

            episode_trade_pnls: List[float] = []
            epoch_loss_q1     : List[float] = []
            epoch_loss_q2     : List[float] = []
            epoch_loss_actor  : List[float] = []
            epoch_loss_alpha  : List[float] = []
            epoch_entropy     : List[float] = []

            if use_parallel:
                # ── PARALLEL experience collection across N workers ────────
                # HC #150: Give each worker exactly 1 file for max parallelism
                # and minimal blocking. imap_unordered will auto-schedule.
                actor_state_cpu = {k: v.cpu().clone() for k, v in actor.state_dict().items()}

                worker_args = [
                    (env_kwargs, actor_state_cpu,
                     [str(f)],  # 1 file per job — pool auto-load-balances
                     OBS_DIM, N_ACTIONS, hidden_dim, w)
                    for w, f in enumerate(train_files)
                ]

                t_collect_start = time.time()
                # Set torch to single-threaded BEFORE fork to prevent
                # OpenMP/MKL deadlocks in child processes
                _orig_threads = torch.get_num_threads()
                torch.set_num_threads(1)
                ctx = mp.get_context("fork")

                # HC #150: Use imap_unordered so GPU gradient updates run
                # AS SOON AS each worker finishes — not blocked waiting for all.
                epoch_steps = 0
                results_collected = 0
                n_gpu_updates = 0

                with ctx.Pool(len(worker_args), initializer=_worker_init) as pool:
                    result_iter = pool.imap_unordered(
                        _worker_collect_episodes, worker_args, chunksize=1
                    )
                    for r in result_iter:
                        results_collected += 1
                        n = r["n_steps"]
                        if n > 0:
                            buffer.push_bulk(
                                r["obs"], r["actions"], r["rewards"],
                                r["next_obs"], r["dones"],
                            )
                            episode_trade_pnls.extend(r["trade_pnls"])
                            epoch_steps += n

                        # GPU gradient updates WHILE other workers still run
                        torch.set_num_threads(_orig_threads)
                        if buffer.is_ready(args.warmup_steps):
                            # Do proportional updates per worker result
                            updates_this_batch = min(
                                args.updates_per_step * max(n // 100, 1),
                                2000,  # cap per worker
                            )
                            for _ in range(updates_this_batch):
                                losses = sac_update(
                                    actor=actor, q1=q1, q2=q2,
                                    q1_target=q1_target, q2_target=q2_target,
                                    log_alpha=log_alpha,
                                    target_entropy=target_entropy,
                                    buffer=buffer,
                                    actor_opt=actor_opt, q1_opt=q1_opt,
                                    q2_opt=q2_opt, alpha_opt=alpha_opt,
                                    batch_size=args.batch_size,
                                    gamma=args.gamma, tau=args.tau,
                                    device=device,
                                )
                                epoch_loss_q1.append(losses["q1_loss"])
                                epoch_loss_q2.append(losses["q2_loss"])
                                epoch_loss_actor.append(losses["actor_loss"])
                                epoch_loss_alpha.append(losses["alpha_loss"])
                                epoch_entropy.append(losses["entropy"])
                                n_gpu_updates += 1
                        torch.set_num_threads(1)  # back to 1 for remaining workers

                        if results_collected % 4 == 0 or results_collected == len(worker_args):
                            log.info(
                                f"  Worker {results_collected}/{len(worker_args)} done | "
                                f"+{n:,} steps | buf={buffer.size:,} | "
                                f"GPU updates={n_gpu_updates:,}"
                            )

                torch.set_num_threads(_orig_threads)  # restore for rest of epoch
                t_collect_end = time.time()

                total_env_steps += epoch_steps

                if epoch == 0 or (epoch + 1) % max(1, args.epochs // 10) == 0:
                    log.info(
                        f"  Parallel collect: {epoch_steps:,} steps from "
                        f"{len(worker_args)} workers in "
                        f"{t_collect_end - t_collect_start:.1f}s | "
                        f"GPU updates: {n_gpu_updates:,}"
                    )

                # ── Final GPU gradient updates after all workers done ──────
                if buffer.is_ready(args.warmup_steps):
                    # Do remaining updates to hit target budget
                    target_total = args.updates_per_step * max(epoch_steps // 100, 1)
                    target_total = min(target_total, 10_000)
                    remaining = max(0, target_total - n_gpu_updates)
                    for _ in range(remaining):
                        losses = sac_update(
                            actor=actor, q1=q1, q2=q2,
                            q1_target=q1_target, q2_target=q2_target,
                            log_alpha=log_alpha,
                            target_entropy=target_entropy,
                            buffer=buffer,
                            actor_opt=actor_opt, q1_opt=q1_opt,
                            q2_opt=q2_opt, alpha_opt=alpha_opt,
                            batch_size=args.batch_size,
                            gamma=args.gamma, tau=args.tau,
                            device=device,
                        )
                        epoch_loss_q1.append(losses["q1_loss"])
                        epoch_loss_q2.append(losses["q2_loss"])
                        epoch_loss_actor.append(losses["actor_loss"])
                        epoch_loss_alpha.append(losses["alpha_loss"])
                        epoch_entropy.append(losses["entropy"])
                        n_gpu_updates += 1

            else:
                # ── SEQUENTIAL (single-core) experience collection ─────────
                # Original interleaved collect + train loop
                file_idx = 0
                obs = env.reset(episode_file=train_files[file_idx % len(train_files)])

                files_completed = 0
                while files_completed < len(train_files):
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

                    with torch.no_grad():
                        action, _, _ = actor.sample(obs_t)
                    action_np = int(action.item())

                    next_obs, reward, done, info = env.step(action_np)
                    total_env_steps += 1

                    buffer.push(obs, action_np, float(reward), next_obs, done)
                    obs = next_obs

                    if done:
                        for t in env._trade_history:
                            episode_trade_pnls.append(t.pnl_ticks)
                        files_completed += 1
                        file_idx += 1
                        if file_idx < len(train_files):
                            obs = env.reset(episode_file=train_files[file_idx])
                        else:
                            break

                    # SAC gradient updates (interleaved with env steps)
                    if buffer.is_ready(args.warmup_steps):
                        for _ in range(args.updates_per_step):
                            losses = sac_update(
                                actor        = actor,
                                q1           = q1,
                                q2           = q2,
                                q1_target    = q1_target,
                                q2_target    = q2_target,
                                log_alpha    = log_alpha,
                                target_entropy=target_entropy,
                                buffer       = buffer,
                                actor_opt    = actor_opt,
                                q1_opt       = q1_opt,
                                q2_opt       = q2_opt,
                                alpha_opt    = alpha_opt,
                                batch_size   = args.batch_size,
                                gamma        = args.gamma,
                                tau          = args.tau,
                                device       = device,
                            )
                            epoch_loss_q1.append(losses["q1_loss"])
                            epoch_loss_q2.append(losses["q2_loss"])
                            epoch_loss_actor.append(losses["actor_loss"])
                            epoch_loss_alpha.append(losses["alpha_loss"])
                            epoch_entropy.append(losses["entropy"])

            elapsed = time.time() - t0

            # Aggregate epoch losses
            mean_q1_loss    = float(np.mean(epoch_loss_q1))    if epoch_loss_q1    else 0.0
            mean_q2_loss    = float(np.mean(epoch_loss_q2))    if epoch_loss_q2    else 0.0
            mean_actor_loss = float(np.mean(epoch_loss_actor)) if epoch_loss_actor else 0.0
            mean_alpha_loss = float(np.mean(epoch_loss_alpha)) if epoch_loss_alpha else 0.0
            mean_entropy    = float(np.mean(epoch_entropy))    if epoch_entropy    else 0.0
            current_alpha   = log_alpha.exp().item()

            train_metrics = compute_episode_metrics(
                episode_trade_pnls,
                total_env_steps,
            )

            # ── Log to history ─────────────────────────────────────────────────
            history["epoch"].append(global_epoch)
            history["fold"].append(fold_num)
            history["alpha"].append(current_alpha)
            history["train_total_pnl_ticks"].append(train_metrics.get("total_pnl_ticks", 0.0))
            history["train_sortino"].append(train_metrics.get("sortino", 0.0))
            history["train_win_rate"].append(train_metrics.get("win_rate", 0.0))
            history["train_avg_pnl_ticks"].append(train_metrics.get("avg_pnl_ticks", 0.0))
            history["train_n_trades"].append(train_metrics.get("n_trades", 0))
            history["q1_loss"].append(mean_q1_loss)
            history["q2_loss"].append(mean_q2_loss)
            history["actor_loss"].append(mean_actor_loss)
            history["alpha_loss"].append(mean_alpha_loss)
            history["entropy"].append(mean_entropy)

            # ── Print progress ─────────────────────────────────────────────────
            if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
                log.info(
                    f"  Fold {fold_num} | Epoch {epoch+1}/{args.epochs} "
                    f"(global {global_epoch}) | {elapsed:.1f}s | "
                    f"α={current_alpha:.4f} | H={mean_entropy:.3f} | "
                    f"target_H={target_entropy:.3f} | "
                    f"Sortino={train_metrics.get('sortino', 0.0):.3f} | "
                    f"WR={train_metrics.get('win_rate', 0.0):.1%} | "
                    f"AvgPnL={train_metrics.get('avg_pnl_ticks', 0.0):.3f}t | "
                    f"Trades={train_metrics.get('n_trades', 0)} | "
                    f"Q1_loss={mean_q1_loss:.4f} | "
                    f"Actor_loss={mean_actor_loss:.4f} | "
                    f"buf={buffer.size:,}"
                )

            # ── MLflow per-step ────────────────────────────────────────────────
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        "train/sortino":     train_metrics.get("sortino", 0.0),
                        "train/win_rate":    train_metrics.get("win_rate", 0.0),
                        "train/avg_pnl":     train_metrics.get("avg_pnl_ticks", 0.0),
                        "train/n_trades":    float(train_metrics.get("n_trades", 0)),
                        "train/sharpe":      train_metrics.get("sharpe", 0.0),
                        "train/pf":          train_metrics.get("profit_factor", 0.0),
                        "loss/q1":           mean_q1_loss,
                        "loss/q2":           mean_q2_loss,
                        "loss/actor":        mean_actor_loss,
                        "loss/alpha":        mean_alpha_loss,
                        "sac/alpha":         current_alpha,
                        "sac/entropy":       mean_entropy,
                        "sac/target_entropy":target_entropy,
                        "sac/buffer_size":   float(buffer.size),
                        "fold":              float(fold_num),
                        "total_env_steps":   float(total_env_steps),
                    }, step=global_epoch)
                except Exception:
                    pass  # don't crash training on MLflow errors

        # ── Evaluate on OOT eval days (end of each fold) ─────────────────────
        log.info(f"  Evaluating on {len(eval_files)} OOT days...")
        eval_metrics = evaluate_policy(env, actor, eval_files, device)

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
                    "eval/sortino":        eval_metrics.get("sortino", 0.0),
                    "eval/sharpe":         eval_metrics.get("sharpe", 0.0),
                    "eval/win_rate":       eval_metrics.get("win_rate", 0.0),
                    "eval/pf":             eval_metrics.get("profit_factor", 0.0),
                    "eval/avg_pnl":        eval_metrics.get("avg_pnl_ticks", 0.0),
                    "eval/total_pnl_usd":  eval_metrics.get("total_pnl_usd", 0.0),
                    "eval/trades_per_day": eval_metrics.get("avg_trades_per_day", 0.0),
                }, step=global_epoch)
            except Exception:
                pass

        # ── Save best checkpoint (by eval Sortino, HC #69) ────────────────────
        eval_sortino = eval_metrics.get("sortino", 0.0)
        if eval_sortino > best_sortino:
            best_sortino = eval_sortino

            # Checkpoint format compatible with rl_execution_agent.py:
            # The agent uses FIFOActorCritic with actor weights matching SACActorNet.
            # We save actor.state_dict() under "model_state" so the loading code works.
            # Also include SAC metadata for debugging/comparison.
            torch.save({
                # Required keys for rl_execution_agent._load_checkpoint()
                "model_state":  actor.state_dict(),
                "obs_dim":      OBS_DIM,
                "n_actions":    N_ACTIONS,
                "hidden_dim":   hidden_dim,
                # Metadata
                "fold":         fold_num,
                "epoch":        global_epoch,
                "eval_metrics": eval_metrics,
                "best_sortino": best_sortino,
                "algorithm":    "SAC",
                "args":         vars(args),
                "run_name":     run_name,
                "alpha":        current_alpha,
                "target_entropy": target_entropy,
            }, best_ckpt_path)
            log.info(
                f"  *** New best checkpoint: Sortino={best_sortino:.3f} → {best_ckpt_path}"
            )
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
                except Exception:
                    pass

    # ── Final checkpoint ───────────────────────────────────────────────────────
    final_ckpt_path = output_dir / f"final_agent_{run_name}.pt"
    torch.save({
        "model_state":  actor.state_dict(),
        "obs_dim":      OBS_DIM,
        "n_actions":    N_ACTIONS,
        "hidden_dim":   hidden_dim,
        "fold":         fold_num,
        "epoch":        global_epoch,
        "history":      history,
        "run_name":     run_name,
        "algorithm":    "SAC",
        "alpha":        log_alpha.exp().item(),
        # Also save Q-networks for potential fine-tuning
        "q1_state":     q1.state_dict(),
        "q2_state":     q2.state_dict(),
    }, final_ckpt_path)
    log.info(f"Final checkpoint saved → {final_ckpt_path}")

    # ── Save history JSON ──────────────────────────────────────────────────────
    history_path = output_dir / f"training_history_{run_name}.json"
    with open(history_path, "w") as f:
        history_serializable = {
            k: [float(x) if hasattr(x, "item") else x for x in v]
            for k, v in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    log.info(f"Training history saved → {history_path}")

    # ── Save training curves PNG (HC #105) ────────────────────────────────────
    plot_path = save_training_curves(history, output_dir, run_name)
    if plot_path and MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(plot_path), artifact_path="plots")
        except Exception:
            pass

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(
        f"\n{'='*60}\n"
        f"SAC Training complete: {run_name}\n"
        f"  Algorithm:       SAC (Soft Actor-Critic, discrete)\n"
        f"  Epochs:          {global_epoch}\n"
        f"  Folds:           {len(folds)}\n"
        f"  Total env steps: {total_env_steps:,}\n"
        f"  Buffer size:     {buffer.size:,}\n"
        f"  Final alpha:     {log_alpha.exp().item():.4f}\n"
        f"  Best Sortino:    {best_sortino:.3f}\n"
        f"  Best ckpt:       {best_ckpt_path}\n"
        f"  Final ckpt:      {final_ckpt_path}\n"
        f"  Training curves: {plot_path}\n"
        f"{'='*60}"
    )

    if MLFLOW_AVAILABLE and mlflow_run:
        try:
            mlflow.log_metrics({
                "summary/best_eval_sortino": best_sortino,
                "summary/total_epochs":      float(global_epoch),
                "summary/n_folds":           float(len(folds)),
                "summary/total_env_steps":   float(total_env_steps),
                "summary/final_alpha":       log_alpha.exp().item(),
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
            "SAC Trainer for FIFO RL Execution Agent\n"
            "\n"
            "Trains a Soft Actor-Critic agent on the FIFOExecutionEnv using\n"
            "sliding walk-forward evaluation. Off-policy, entropy-regularized,\n"
            "sample-efficient alternative to PPO (HC #117).\n"
            "\n"
            "Reward: delta rolling Sortino per trade (risk-adjusted, HC #100/#104).\n"
            "Cost model: 0.376 ticks for limits, 1.376 ticks for market orders (HC #89).\n"
            "\n"
            "Example:\n"
            "  python train_fifo_rl_sac.py --epochs 50 --lr 3e-4\n"
            "  python train_fifo_rl_sac.py --epochs 100 --train-days 60 --eval-days 10\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Data ───────────────────────────────────────────────────────────────────
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
        help="Directory with PatchTST fold_XX_oot_predictions.npz for confluence",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("/home/jupiter/Lvl3Quant/output/fifo_sac_rl")),
        help="Directory to save checkpoints, logs, and plots",
    )

    # ── Walk-forward ───────────────────────────────────────────────────────────
    p.add_argument(
        "--train-days", type=int, default=DEFAULT_TRAIN_DAYS,
        help=f"Training window size in days (sliding, HC #0) (default: {DEFAULT_TRAIN_DAYS})",
    )
    p.add_argument(
        "--eval-days", type=int, default=DEFAULT_EVAL_DAYS,
        help=f"OOT eval window per fold in days (default: {DEFAULT_EVAL_DAYS})",
    )

    # ── Training ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"SAC training epochs per walk-forward fold (default: {DEFAULT_EPOCHS})",
    )
    p.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Adam learning rate for all networks (default: {DEFAULT_LR})",
    )

    # ── SAC-specific ───────────────────────────────────────────────────────────
    p.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Discount factor (default: {DEFAULT_GAMMA})",
    )
    p.add_argument(
        "--tau", type=float, default=DEFAULT_TAU,
        help=f"Polyak averaging rate for target networks (default: {DEFAULT_TAU})",
    )
    p.add_argument(
        "--alpha-init", type=float, default=DEFAULT_ALPHA_INIT,
        help=f"Initial entropy temperature alpha (default: {DEFAULT_ALPHA_INIT})",
    )
    p.add_argument(
        "--target-entropy-ratio", type=float, default=0.98,
        help="Target entropy as ratio of max entropy: target = ratio * log(N_ACTIONS) "
             "(default: 0.98)",
    )
    p.add_argument(
        "--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE,
        help=f"Replay buffer capacity (default: {DEFAULT_BUFFER_SIZE:,})",
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Mini-batch size for SAC updates (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS,
        help=f"Env steps before SAC updates begin (default: {DEFAULT_WARMUP_STEPS})",
    )
    p.add_argument(
        "--updates-per-step", type=int, default=DEFAULT_UPDATES_PER_STEP,
        help=f"SAC gradient updates per env step (default: {DEFAULT_UPDATES_PER_STEP})",
    )

    # ── Architecture ───────────────────────────────────────────────────────────
    p.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Hidden dimension for all networks (default: 256)",
    )

    # ── Parallelism ────────────────────────────────────────────────────────────
    p.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers for experience collection. "
             "Each worker runs its own env on a subset of training files. "
             "Requires Linux (uses fork context). Set to 24-28 on Neptune (32 cores). "
             "(default: 1 = sequential single-core)",
    )

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("FIFO SAC RL Trainer  (Soft Actor-Critic, discrete)")
    log.info("=" * 60)
    log.info(f"  data_dir:         {args.data_dir}")
    log.info(f"  pred_dir:         {args.pred_dir or DEFAULT_PRED_DIR}")
    log.info(f"  output_dir:       {args.output_dir}")
    log.info(f"  epochs:           {args.epochs} per fold")
    log.info(f"  lr:               {args.lr}")
    log.info(f"  gamma:            {args.gamma}")
    log.info(f"  tau:              {args.tau}")
    log.info(f"  alpha_init:       {args.alpha_init}")
    log.info(
        f"  target_entropy:   {args.target_entropy_ratio:.2f} × log({N_ACTIONS}) = "
        f"{args.target_entropy_ratio * math.log(N_ACTIONS):.4f} nats"
    )
    log.info(f"  buffer_size:      {args.buffer_size:,}")
    log.info(f"  batch_size:       {args.batch_size}")
    log.info(f"  warmup_steps:     {args.warmup_steps}")
    log.info(f"  updates_per_step: {args.updates_per_step}")
    log.info(f"  train_days:       {args.train_days}  (sliding walk-forward, HC #0)")
    log.info(f"  eval_days:        {args.eval_days}")
    log.info(f"  hidden_dim:       {args.hidden_dim}")
    log.info(f"  n_workers:        {args.n_workers}  ({'parallel' if args.n_workers > 1 else 'sequential'})")
    log.info(f"  torch:            {torch.__version__}")
    log.info(f"  mlflow:           {'yes' if MLFLOW_AVAILABLE else 'NO — install mlflow'}")
    log.info(f"  matplotlib:       {'yes' if MATPLOTLIB_AVAILABLE else 'NO — install matplotlib'}")
    log.info("=" * 60)

    train(args)
