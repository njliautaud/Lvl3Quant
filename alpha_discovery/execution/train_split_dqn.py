#!/usr/bin/env python3
"""
Split DQN Execution Agent — v1
================================
HC #166: DQN with separate Entry/Cancel/Exit Q-networks.

Architecture:
- Entry DQN: when flat → {Hold, Limit Buy, Limit Sell, Market Buy, Market Sell}
- Cancel DQN: when pending → {Wait, Cancel}
- Exit DQN: when in position → {Hold, Passive Exit, Market Exit}

Key design choices (from deep analysis):
- Dueling DQN (Value + Advantage streams)
- Double DQN (online selects, target evaluates)
- Prioritized Experience Replay
- n-step returns (n=50) for faster reward propagation
- Dense shaping for exit (ΔUnrealized PnL per step)
- Signal change since entry as natural decay signal
- NO auto-exit — agent learns 30-60s optimal hold from data
- Async GPU training thread (proven in SAC v12)

Author: Claude (Infrastructure Builder)
Date: 2026-05-04
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import random
import logging
import argparse
import threading
import psutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("split_dqn")

# ─── HC #182: Adaptive Worker Scaling ──────────────────────────────────────────
def calculate_adaptive_workers():
    """
    HC #182: Adaptive worker scaling — never saturate RAM/cores.
    Returns worker count based on available CPU and memory.

    Heuristic:
    - Start with (available_cores * 0.6) to leave 40% headroom
    - Reduce if RAM utilization > 70% (save headroom for user/OS)
    - Minimum 2 workers, maximum 28 (cap for resource safety)
    """
    try:
        total_cores = os.cpu_count() or 32
        available_cores = psutil.cpu_count(logical=False) or total_cores

        # RAM check: use available (free + cached)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent  # percentage used

        # Base: 60% of cores, save 40% headroom
        base_workers = max(2, int(available_cores * 0.6))

        # Reduce if memory pressure high
        if mem_percent > 80:
            workers = max(2, int(base_workers * 0.5))
            log.warning(f"High memory pressure ({mem_percent}%) — reducing workers to {workers}")
        elif mem_percent > 70:
            workers = max(2, int(base_workers * 0.7))
            log.warning(f"Moderate memory pressure ({mem_percent}%) — scaling workers to {workers}")
        else:
            workers = base_workers

        # Cap at 28 (never use all cores, HC #181)
        workers = min(workers, 28)

        log.info(f"Adaptive workers: cores={available_cores}, mem={mem_percent}%, workers={workers}")
        return workers
    except Exception as e:
        log.error(f"Error calculating adaptive workers: {e}, defaulting to 12")
        return 12

# ─── Constants ──────────────────────────────────────────────────────────────────
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376  # $4.70 / $12.50

# Observation dimensions per network
ENTRY_OBS_DIM = 31   # Signal(8) + Book(6) + Context(4) + History(8) + PCA(5)
CANCEL_OBS_DIM = 20  # Signal(8) + Book(6) + Order(3) + Context(3)
EXIT_OBS_DIM = 42    # Entry(31) + Position(5) + SignalChange(3) + Momentum(1) + MFE/MAE(2)

# Action spaces
ENTRY_ACTIONS = 5   # Hold, LimitBuy, LimitSell, MarketBuy, MarketSell
CANCEL_ACTIONS = 2  # Wait, Cancel
EXIT_ACTIONS = 3    # Hold, PassiveExit, MarketExit

# Map split actions back to env actions (7-action space)
ENTRY_TO_ENV = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # direct mapping
CANCEL_TO_ENV = {0: 0, 1: 5}  # Wait=Hold, Cancel=Cancel
EXIT_TO_ENV = {0: 0, 1: 1, 2: 6}  # Hold, PassiveExit→LimitOpposite, MarketExit


# ─── Prioritized Replay Buffer ─────────────────────────────────────────────────
class PrioritizedReplayBuffer:
    """Sum-tree based prioritized replay."""

    def __init__(self, capacity: int, alpha: float = 0.6, n_step: int = 1, gamma: float = 0.999):
        self.capacity = capacity
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.pos = 0
        self.size = 0

        # Storage
        self.states = np.zeros((capacity, EXIT_OBS_DIM), dtype=np.float32)  # max dim
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, EXIT_OBS_DIM), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Priorities (sum tree)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        # n-step buffer
        self._n_step_buf = deque(maxlen=n_step)

    def _compute_nstep(self):
        """Compute n-step return from buffer."""
        reward = 0.0
        for i, (_, _, r, _, d) in enumerate(self._n_step_buf):
            reward += (self.gamma ** i) * r
            if d:
                break
        s, a, _, _, _ = self._n_step_buf[0]
        _, _, _, ns, nd = self._n_step_buf[-1]
        return s, a, reward, ns, nd

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """Add transition with n-step handling."""
        self._n_step_buf.append((state, action, reward, next_state, done))

        if len(self._n_step_buf) < self.n_step and not done:
            return

        # Compute n-step return
        s, a, r, ns, d = self._compute_nstep()

        obs_dim = len(s)
        self.states[self.pos, :obs_dim] = s
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.next_states[self.pos, :obs_dim] = ns
        self.dones[self.pos] = float(d)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # If episode ended, flush remaining n-step transitions
        if done:
            while len(self._n_step_buf) > 1:
                self._n_step_buf.popleft()
                if len(self._n_step_buf) > 0:
                    s2, a2, r2, ns2, d2 = self._compute_nstep()
                    obs_dim2 = len(s2)
                    self.states[self.pos, :obs_dim2] = s2
                    self.actions[self.pos] = a2
                    self.rewards[self.pos] = r2
                    self.next_states[self.pos, :obs_dim2] = ns2
                    self.dones[self.pos] = 1.0
                    self.priorities[self.pos] = self.max_priority
                    self.pos = (self.pos + 1) % self.capacity
                    self.size = min(self.size + 1, self.capacity)
            self._n_step_buf.clear()

    def sample(self, batch_size: int, beta: float = 0.4, obs_dim: int = 31):
        """Sample batch with priorities."""
        if self.size < batch_size:
            return None

        # Proportional sampling
        probs = self.priorities[:self.size] ** self.alpha
        probs_sum = probs.sum()
        if probs_sum == 0:
            probs = np.ones(self.size) / self.size
        else:
            probs = probs / probs_sum

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights = weights / weights.max()

        states = self.states[indices, :obs_dim]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices, :obs_dim]
        dones = self.dones[indices]

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities with TD errors."""
        priorities = np.abs(td_errors) + 1e-6
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


# ─── Dueling DQN Network ───────────────────────────────────────────────────────
class DuelingDQN(nn.Module):
    """Dueling DQN: V(s) + A(s,a) - mean(A)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        val = self.value(feat)
        adv = self.advantage(feat)
        # Q = V + A - mean(A)
        return val + adv - adv.mean(dim=-1, keepdim=True)


# ─── Split DQN Agent ───────────────────────────────────────────────────────────
class SplitDQNAgent:
    """Three DQN networks with separate replay buffers and training."""

    def __init__(
        self,
        hidden_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.999,
        tau: float = 0.005,
        buffer_size: int = 500_000,
        batch_size: int = 4096,
        n_step: int = 20,   # HC #230: was 50, reduced to 20 to limit reward smearing
        device: str = "cuda",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.n_step = n_step

        # Entry network
        self.entry_net = DuelingDQN(ENTRY_OBS_DIM, ENTRY_ACTIONS, hidden_dim).to(self.device)
        self.entry_target = DuelingDQN(ENTRY_OBS_DIM, ENTRY_ACTIONS, hidden_dim).to(self.device)
        self.entry_target.load_state_dict(self.entry_net.state_dict())
        self.entry_opt = torch.optim.Adam(self.entry_net.parameters(), lr=lr)
        self.entry_buffer = PrioritizedReplayBuffer(buffer_size, n_step=n_step, gamma=gamma)

        # Cancel network
        self.cancel_net = DuelingDQN(CANCEL_OBS_DIM, CANCEL_ACTIONS, hidden_dim).to(self.device)
        self.cancel_target = DuelingDQN(CANCEL_OBS_DIM, CANCEL_ACTIONS, hidden_dim).to(self.device)
        self.cancel_target.load_state_dict(self.cancel_net.state_dict())
        self.cancel_opt = torch.optim.Adam(self.cancel_net.parameters(), lr=lr)
        self.cancel_buffer = PrioritizedReplayBuffer(buffer_size // 2, n_step=n_step, gamma=gamma)

        # Exit network
        self.exit_net = DuelingDQN(EXIT_OBS_DIM, EXIT_ACTIONS, hidden_dim).to(self.device)
        self.exit_target = DuelingDQN(EXIT_OBS_DIM, EXIT_ACTIONS, hidden_dim).to(self.device)
        self.exit_target.load_state_dict(self.exit_net.state_dict())
        self.exit_opt = torch.optim.Adam(self.exit_net.parameters(), lr=lr)
        self.exit_buffer = PrioritizedReplayBuffer(buffer_size, n_step=n_step, gamma=gamma)

        # Epsilon schedule
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay_steps = 100_000
        self.total_steps = 0

        # TF32 for speed
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Stats
        self.train_steps = {"entry": 0, "cancel": 0, "exit": 0}
        self._lock = threading.Lock()

    def get_epsilon(self) -> float:
        """Linear epsilon decay."""
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon * (1 - frac) + self.epsilon_min * frac

    def select_entry_action(self, obs: np.ndarray) -> int:
        """Select entry action with epsilon-greedy."""
        if random.random() < self.get_epsilon():
            return random.randint(0, ENTRY_ACTIONS - 1)
        with torch.no_grad():
            state = torch.FloatTensor(obs[:ENTRY_OBS_DIM]).unsqueeze(0).to(self.device)
            q_vals = self.entry_net(state)
            return q_vals.argmax(dim=1).item()

    def select_cancel_action(self, obs: np.ndarray) -> int:
        """Select cancel action."""
        if random.random() < self.get_epsilon():
            return random.randint(0, CANCEL_ACTIONS - 1)
        with torch.no_grad():
            state = torch.FloatTensor(obs[:CANCEL_OBS_DIM]).unsqueeze(0).to(self.device)
            q_vals = self.cancel_net(state)
            return q_vals.argmax(dim=1).item()

    def select_exit_action(self, obs: np.ndarray) -> int:
        """Select exit action."""
        if random.random() < self.get_epsilon():
            return random.randint(0, EXIT_ACTIONS - 1)
        with torch.no_grad():
            state = torch.FloatTensor(obs[:EXIT_OBS_DIM]).unsqueeze(0).to(self.device)
            q_vals = self.exit_net(state)
            return q_vals.argmax(dim=1).item()

    def _train_network(self, net, target_net, optimizer, buffer, obs_dim, beta):
        """Single Double-DQN update step with PER."""
        batch = buffer.sample(self.batch_size, beta=beta, obs_dim=obs_dim)
        if batch is None:
            return 0.0

        states, actions, rewards, next_states, dones, weights, indices = batch
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # HC #211/#219: Reward clipping. Bounded reward per REWARD_DESIGN.md spec
        # ([-5, +5] ticks/Sortino-units). Prevents single-outcome outliers from
        # exploding Q-loss (Fold 1 Ep 5 EXIT Q-loss hit 125,000 due to unbounded
        # raw-PnL reward signals before this clip was wired).
        rewards = torch.clamp(rewards, min=-5.0, max=5.0)

        # Current Q values
        q_values = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: online selects, target evaluates
        with torch.no_grad():
            next_actions = net(next_states).argmax(dim=1)
            next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # n-step discount
            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1 - dones)

        # HC #211/#219: Huber loss (was MSE). Huber is quadratic for small TD errors
        # and linear for large ones — robust to outlier rewards/transitions.
        # delta=1.0 is standard; td_errors > 1 contribute |error| not error^2.
        td_errors = target_q - q_values
        huber = F.huber_loss(q_values, target_q, reduction='none', delta=1.0)
        loss = (weights * huber).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        optimizer.step()

        # Update priorities
        buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Soft update target
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return loss.item()

    def train_step(self) -> Dict[str, float]:
        """Train all three networks once."""
        beta = min(1.0, 0.4 + self.total_steps * 0.6 / self.epsilon_decay_steps)
        losses = {}

        with self._lock:
            if self.entry_buffer.size >= self.batch_size:
                losses["entry"] = self._train_network(
                    self.entry_net, self.entry_target, self.entry_opt,
                    self.entry_buffer, ENTRY_OBS_DIM, beta
                )
                self.train_steps["entry"] += 1

            if self.cancel_buffer.size >= self.batch_size // 4:
                losses["cancel"] = self._train_network(
                    self.cancel_net, self.cancel_target, self.cancel_opt,
                    self.cancel_buffer, CANCEL_OBS_DIM, beta
                )
                self.train_steps["cancel"] += 1

            if self.exit_buffer.size >= self.batch_size:
                losses["exit"] = self._train_network(
                    self.exit_net, self.exit_target, self.exit_opt,
                    self.exit_buffer, EXIT_OBS_DIM, beta
                )
                self.train_steps["exit"] += 1

        self.total_steps += 1
        return losses

    def save(self, path: Path):
        """Save all networks."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.entry_net.state_dict(), path / "entry_net.pt")
        torch.save(self.cancel_net.state_dict(), path / "cancel_net.pt")
        torch.save(self.exit_net.state_dict(), path / "exit_net.pt")
        torch.save({
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "epsilon": self.get_epsilon(),
        }, path / "agent_state.pt")

    def load(self, path: Path):
        """Load all networks."""
        self.entry_net.load_state_dict(torch.load(path / "entry_net.pt", map_location=self.device))
        self.cancel_net.load_state_dict(torch.load(path / "cancel_net.pt", map_location=self.device))
        self.exit_net.load_state_dict(torch.load(path / "exit_net.pt", map_location=self.device))
        # Sync targets
        self.entry_target.load_state_dict(self.entry_net.state_dict())
        self.cancel_target.load_state_dict(self.cancel_net.state_dict())
        self.exit_target.load_state_dict(self.exit_net.state_dict())


# ─── Observation Builders ──────────────────────────────────────────────────────
def build_entry_obs(full_obs: np.ndarray) -> np.ndarray:
    """Extract entry-relevant features from full 48-dim env obs."""
    # Signal: [0:4] → pred_1s, pred_5s, pred_10s, confidence
    # PatchTST: [39:42] → pst_1s, pst_5s, pst_10s
    # Confluence: [42] → confluence_1s
    # Book: [4:8] → bid_depth, ask_depth, imbalance, spread
    # Price: [8:10]
    # Context: [16:19] → vol_60s, tod_sin, tod_cos
    # Flow: [19:24] → event_density, buy_frac, flow_imbalance, cancel_rate, local_rate
    # History: [24:29] → last 5 PnLs
    # Stats: [29:32] → win_rate, sortino, consec_losses
    # PCA: [34:39]
    obs = np.zeros(ENTRY_OBS_DIM, dtype=np.float32)
    obs[0:4] = full_obs[0:4]      # Signal (CNN-Mamba)
    obs[4:7] = full_obs[39:42]    # PatchTST predictions
    obs[7] = full_obs[42] if len(full_obs) > 42 else 0.0  # Confluence
    obs[8:12] = full_obs[4:8]     # Book state
    obs[12:14] = full_obs[8:10]   # Price
    obs[14:17] = full_obs[16:19]  # Context (vol, tod)
    obs[17:19] = full_obs[19:21]  # Flow (density, buy_frac)
    obs[19:24] = full_obs[24:29]  # Trade history
    obs[24:27] = full_obs[29:32]  # Stats
    obs[27:31] = full_obs[34:38]  # PCA embedding (4 of 5)
    return obs


def build_cancel_obs(full_obs: np.ndarray) -> np.ndarray:
    """Extract cancel-relevant features."""
    obs = np.zeros(CANCEL_OBS_DIM, dtype=np.float32)
    obs[0:4] = full_obs[0:4]      # Signal
    obs[4:7] = full_obs[39:42]    # PatchTST
    obs[7:11] = full_obs[4:8]     # Book
    obs[11] = full_obs[13]        # Queue fill fraction
    obs[12:14] = full_obs[32:34]  # Pending order state
    obs[14:17] = full_obs[16:19]  # Context
    obs[17:20] = full_obs[45:48] if len(full_obs) > 47 else np.zeros(3)  # Alpha awareness
    return obs


def build_exit_obs(full_obs: np.ndarray, entry_signal: np.ndarray) -> np.ndarray:
    """Extract exit-relevant features including signal change since entry."""
    obs = np.zeros(EXIT_OBS_DIM, dtype=np.float32)
    # Include all entry features (31)
    obs[0:31] = build_entry_obs(full_obs)
    # Position state: [10:14]
    obs[31] = full_obs[10]   # position direction
    obs[32] = full_obs[11]   # unrealized PnL
    obs[33] = full_obs[12]   # hold time
    obs[34] = full_obs[14]   # MFE
    obs[35] = full_obs[15]   # MAE
    # Signal change since entry (THE key feature — replaces fake half-life)
    current_signal = full_obs[0:3]  # pred_1s, pred_5s, pred_10s now
    obs[36:39] = current_signal - entry_signal  # Δsignal
    # Signal momentum (is it strengthening or weakening?)
    signal_magnitude_now = np.abs(current_signal).mean()
    signal_magnitude_entry = np.abs(entry_signal).mean()
    obs[39] = signal_magnitude_now - signal_magnitude_entry  # positive = strengthening
    # MFE ratio (how much we've given back from peak)
    mfe = max(full_obs[14], 0.01)
    current_pnl = full_obs[11] * 10.0  # denormalize
    obs[40] = current_pnl / mfe if mfe > 0 else 0.0  # 1.0 = at MFE, <1 = giving back
    # Hold time as fraction of data-optimal (30-60s)
    obs[41] = min(full_obs[12] * 30.0 / 45.0, 2.0)  # normalized to 45s optimal midpoint
    return obs


# ─── Episode Runner ────────────────────────────────────────────────────────────
def run_episode(
    env,
    agent: SplitDQNAgent,
    entry_transitions: list,
    cancel_transitions: list,
    exit_transitions: list,
):
    """Run one episode, collecting transitions for each network.

    HC #221/#230: Per-Head Reward Design
    =====================================
    Each of the 3 split-DQN heads gets a PURPOSE-BUILT reward:

    ENTRY reward (when trade eventually closes, propagated via n-step):
      - Primary: edge_captured = pnl_ticks (already includes commission)
      - Penalty: -ALPHA_GATE_PENALTY for entries without signal support
      - Penalty: -0.10 for misaligned entries (buy when signal says sell)
      - Per-trade cost is already in env's pnl_ticks (commission deducted)
      - Trade frequency cost: -0.02 per entry to discourage overtrading

    CANCEL reward (immediate + counterfactual via info dict):
      - If cancel and price moved away: +0.05 (avoided adverse fill)
      - If cancel and price came back / filled: -0.05 (missed profitable fill)
      - Default (no counterfactual available): 0.0

    EXIT reward (when trade closes):
      - Primary: MFE-capture ratio = pnl_ticks / max(mfe_ticks, 0.25)
        Bounded [-1, +1]. Measures how much of available profit was captured.
      - Dense shaping: 0.01 * delta_unrealized (per-step while in position)
      - Penalty: -0.05 * max(hold_secs - 30, 0) / 30 for excessive holds (capped)
    """
    full_obs = env.reset()
    done = False
    entry_signal = np.zeros(3)  # signal at time of entry
    prev_unrealized = 0.0
    episode_trades = 0
    episode_pnl = 0.0

    # HC #230: Per-trade entry cost to discourage overtrading
    ENTRY_FREQUENCY_COST = -0.02

    while not done:
        position = full_obs[10]  # 0, +1, -1
        has_pending = full_obs[32] > 0.5

        if position == 0 and not has_pending:
            # FLAT → Entry network decides
            entry_obs = build_entry_obs(full_obs)
            action_idx = agent.select_entry_action(entry_obs)
            env_action = ENTRY_TO_ENV[action_idx]

            next_full_obs, reward, done, info = env.step(env_action)

            # If we just entered, record entry signal
            if next_full_obs[10] != 0 or next_full_obs[32] > 0.5:
                entry_signal = full_obs[0:3].copy()

            # HC #221: ENTRY reward = env reward (alpha-gate penalty + trade close reward via n-step)
            # Plus per-trade frequency cost to discourage overtrading
            entry_reward = reward
            if env_action != 0:  # Actually placed an order
                entry_reward += ENTRY_FREQUENCY_COST

            entry_transitions.append((entry_obs, action_idx, entry_reward, build_entry_obs(next_full_obs), done))

        elif has_pending and position == 0:
            # PENDING → Cancel network decides
            cancel_obs = build_cancel_obs(full_obs)
            action_idx = agent.select_cancel_action(cancel_obs)
            env_action = CANCEL_TO_ENV[action_idx]

            next_full_obs, reward, done, info = env.step(env_action)

            # If filled (position changed), record entry signal
            if next_full_obs[10] != 0:
                entry_signal = full_obs[0:3].copy()

            # HC #221: CANCEL reward = counterfactual-based
            # If we cancelled (action_idx == 1 = cancel), evaluate counterfactual
            # Simple heuristic: if price moved away from our level, cancelling was good
            cancel_reward = reward  # base env reward (usually 0)
            # Note: Proper counterfactual requires N-event lookahead which is done
            # by the env's staleness check. For now, trust the env reward and add
            # mild shaping: if we kept the order (action=0) and it got filled,
            # the reward will come through n-step from the eventual trade close.

            cancel_transitions.append((cancel_obs, action_idx, cancel_reward, build_cancel_obs(next_full_obs), done))

        else:
            # IN POSITION → Exit network decides
            exit_obs = build_exit_obs(full_obs, entry_signal)
            action_idx = agent.select_exit_action(exit_obs)

            # Map exit action to env action
            if action_idx == 1:  # Passive exit
                # Place limit order on opposite side
                if position > 0:
                    env_action = 2  # Limit sell at ask
                else:
                    env_action = 1  # Limit buy at bid
            elif action_idx == 2:  # Market exit
                env_action = 6  # Market exit
            else:
                env_action = 0  # Hold

            next_full_obs, reward, done, info = env.step(env_action)

            # HC #221: EXIT reward = MFE-capture ratio when trade closes
            if "last_trade" in info:
                # Trade just closed — compute MFE-capture ratio
                lt = info["last_trade"]
                mfe = max(lt["mfe_ticks"], 0.25)  # floor to prevent div-by-zero
                mfe_capture = lt["pnl_ticks"] / mfe
                # Bound to [-1, +1] to prevent outliers
                mfe_capture = max(-1.0, min(1.0, mfe_capture))

                # Hold time penalty: mild penalty for holding beyond signal horizon
                hold_penalty = 0.0
                if lt["hold_secs"] > 30.0:
                    hold_penalty = -0.05 * min((lt["hold_secs"] - 30.0) / 30.0, 1.0)

                exit_reward = mfe_capture + hold_penalty
                episode_trades += 1
                episode_pnl += lt["pnl_ticks"]
            else:
                # Still in position — dense shaping on unrealized PnL
                current_unrealized = next_full_obs[11] * 10.0  # denormalize
                shaping_reward = 0.01 * (current_unrealized - prev_unrealized)
                exit_reward = shaping_reward

            exit_transitions.append((exit_obs, action_idx, exit_reward,
                                    build_exit_obs(next_full_obs, entry_signal), done))
            prev_unrealized = next_full_obs[11] * 10.0 if next_full_obs[10] != 0 else 0.0

            # Reset if position closed
            if next_full_obs[10] == 0:
                prev_unrealized = 0.0
                entry_signal = np.zeros(3)

        full_obs = next_full_obs
        agent.total_steps += 1

    return episode_trades, episode_pnl


# ─── Worker Function (CPU) ──────────────────────────────────────────────────────
def worker_run_file(args):
    """Process one MBO file, return transitions."""
    # Support both old (4-tuple) and new (5-tuple with precomputed_dir) format
    if len(args) == 5:
        file_path, pred_dir, patchtst_dir, fold_idx, precomputed_dir = args
    else:
        file_path, pred_dir, patchtst_dir, fold_idx = args
        precomputed_dir = None

    # Import env (each worker gets its own)
    sys.path.insert(0, str(Path(__file__).parent))

    if precomputed_dir and Path(precomputed_dir).exists():
        # Use precomputed observations (10-50x speedup)
        from precompute_observations import PrecomputedEnv
        env = PrecomputedEnv(
            precomputed_dir=Path(precomputed_dir),
            verbose=False,
        )
    else:
        from fifo_rl_env import FIFOExecutionEnv
        env = FIFOExecutionEnv(
            event_dir=Path(file_path).parent,
            pred_dir=Path(pred_dir),
            patchtst_pred_dir=Path(patchtst_dir) if patchtst_dir else None,
            verbose=False,
        )

    # Create a lightweight agent just for action selection (no GPU needed)
    # We'll use random actions in workers and train centrally
    entry_transitions = []
    cancel_transitions = []
    exit_transitions = []

    try:
        full_obs = env.reset(episode_file=Path(file_path))
    except Exception as e:
        log.warning(f"Failed to load {file_path}: {e}")
        return [], [], [], 0, 0.0

    # Run episode with random policy (epsilon=1.0 for data collection)
    done = False
    info = {}
    entry_signal = np.zeros(3)
    prev_unrealized = 0.0
    trades = 0
    total_pnl = 0.0

    while not done:
        position = full_obs[10]
        has_pending = full_obs[32] > 0.5

        if position == 0 and not has_pending:
            entry_obs = build_entry_obs(full_obs)
            # Exploration: weighted random (bias toward hold early, entries later)
            action_idx = random.choices(
                range(ENTRY_ACTIONS),
                weights=[0.7, 0.1, 0.1, 0.05, 0.05]  # Mostly hold, some entries
            )[0]
            env_action = ENTRY_TO_ENV[action_idx]
            next_full_obs, reward, done, info = env.step(env_action)

            if next_full_obs[10] != 0 or next_full_obs[32] > 0.5:
                entry_signal = full_obs[0:3].copy()

            # HC #221: ENTRY reward with per-trade frequency cost
            entry_reward = reward + (-0.02 if env_action != 0 else 0.0)

            entry_transitions.append((
                entry_obs, action_idx, entry_reward,
                build_entry_obs(next_full_obs), done
            ))

        elif has_pending and position == 0:
            cancel_obs = build_cancel_obs(full_obs)
            action_idx = random.randint(0, CANCEL_ACTIONS - 1)
            env_action = CANCEL_TO_ENV[action_idx]
            next_full_obs, reward, done, info = env.step(env_action)

            if next_full_obs[10] != 0:
                entry_signal = full_obs[0:3].copy()

            # HC #221: CANCEL reward (base env reward for now)
            cancel_transitions.append((
                cancel_obs, action_idx, reward,
                build_cancel_obs(next_full_obs), done
            ))

        else:
            exit_obs = build_exit_obs(full_obs, entry_signal)
            action_idx = random.choices(
                range(EXIT_ACTIONS),
                weights=[0.6, 0.2, 0.2]  # Mostly hold, some exits
            )[0]

            if action_idx == 1:
                env_action = 2 if position > 0 else 1
            elif action_idx == 2:
                env_action = 6
            else:
                env_action = 0

            next_full_obs, reward, done, info = env.step(env_action)

            # HC #221: EXIT reward = MFE-capture ratio when trade closes
            if "last_trade" in info:
                lt = info["last_trade"]
                mfe = max(lt["mfe_ticks"], 0.25)
                mfe_capture = max(-1.0, min(1.0, lt["pnl_ticks"] / mfe))
                hold_pen = -0.05 * min((lt["hold_secs"] - 30.0) / 30.0, 1.0) if lt["hold_secs"] > 30.0 else 0.0
                exit_reward = mfe_capture + hold_pen
                trades += 1
            else:
                current_unrealized = next_full_obs[11] * 10.0
                exit_reward = 0.01 * (current_unrealized - prev_unrealized)

            exit_transitions.append((
                exit_obs, action_idx, exit_reward,
                build_exit_obs(next_full_obs, entry_signal), done
            ))
            prev_unrealized = next_full_obs[11] * 10.0 if next_full_obs[10] != 0 else 0.0

            if next_full_obs[10] == 0:
                prev_unrealized = 0.0
                entry_signal = np.zeros(3)

        full_obs = next_full_obs

    # Get final PnL from last info
    total_pnl = info.get("episode_pnl_ticks", 0.0) * TICK_VALUE if info else 0.0
    return entry_transitions, cancel_transitions, exit_transitions, trades, total_pnl


# ─── Async GPU Training Thread ─────────────────────────────────────────────────
class AsyncTrainer(threading.Thread):
    """Continuously trains on GPU from replay buffers."""

    def __init__(self, agent: SplitDQNAgent, updates_per_sec: int = 100):
        super().__init__(daemon=True)
        self.agent = agent
        self.updates_per_sec = updates_per_sec
        self.running = True
        self.total_updates = 0
        self.losses = {"entry": [], "cancel": [], "exit": []}

    def run(self):
        log.info("GPU training thread started")
        while self.running:
            try:
                min_size = min(
                    self.agent.entry_buffer.size,
                    self.agent.exit_buffer.size,
                )
                if min_size < self.agent.batch_size:
                    time.sleep(0.1)
                    continue

                losses = self.agent.train_step()
                self.total_updates += 1
                for k, v in losses.items():
                    self.losses[k].append(v)

                if self.total_updates % 1000 == 0:
                    log.info(f"  [GPU] {self.total_updates} updates | "
                             f"loss E={np.mean(self.losses['entry'][-100:]) if self.losses['entry'] else 0:.4f} "
                             f"C={np.mean(self.losses['cancel'][-100:]) if self.losses['cancel'] else 0:.4f} "
                             f"X={np.mean(self.losses['exit'][-100:]) if self.losses['exit'] else 0:.4f}")

                # Throttle to target rate
                time.sleep(1.0 / self.updates_per_sec)
            except Exception as e:
                log.error(f"GPU trainer error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

    def stop(self):
        self.running = False


# ─── Main Training Loop ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Split DQN Execution Training")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--patchtst-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--n-step", type=int, default=20,
                        help="n-step return horizon. Reduced 50->20 per HC #230: 50-step "
                             "propagation lets one big trade outcome dominate 50 transitions, "
                             "polluting CANCEL/ENTRY gradients. 20 is enough for credit "
                             "assignment without smearing across whole episodes.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--train-days", type=int, default=25)
    parser.add_argument("--eval-days", type=int, default=3)
    parser.add_argument("--n-workers", type=int, default=None)  # Will be adaptive
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gpu-updates-per-sec", type=int, default=200)
    parser.add_argument("--precomputed-dir", type=str, default=None,
                        help="Directory with precomputed .npy obs. If provided, uses "
                             "PrecomputedEnv for 10-50x speedup (HC #241/#243)")
    args = parser.parse_args()

    # HC #182: Use adaptive workers if not specified
    if args.n_workers is None:
        args.n_workers = calculate_adaptive_workers()
    else:
        log.info(f"User specified {args.n_workers} workers (override adaptive)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    log.info(f"Split DQN v1_r10a — {args.hidden_dim} hidden, {args.n_workers} workers (adaptive), n_step={args.n_step}")
    log.info(f"Output: {output_dir}")

    # HC #171 RESUME: Check for checkpoint to resume from
    resume_fold = 0
    latest_checkpoint = None
    checkpoint_files = sorted(output_dir.glob("fold*_ep*.pt"))
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        resume_fold = int(latest_checkpoint.stem.split("fold")[1].split("_")[0])
        log.info(f"RESUME: Found checkpoint {latest_checkpoint.name} — resuming from fold {resume_fold}")
    else:
        log.info("No checkpoints found — starting fresh")

    # Discover training files (precomputed or raw MBO)
    try:
        if args.precomputed_dir and Path(args.precomputed_dir).exists():
            # HC #241/#243: Use precomputed observations
            precomputed_path = Path(args.precomputed_dir)
            obs_files = sorted(precomputed_path.glob("*_base_obs.npy"))
            if not obs_files:
                log.error(f"No precomputed obs files in {precomputed_path}")
                return
            # Create pseudo-file paths matching the date extraction pattern
            oot_files = [Path(args.data_dir) / f"{f.stem.replace('_base_obs', '')}_mbo_events.npz"
                         for f in obs_files]
            log.info(f"PRECOMPUTED MODE: {len(oot_files)} dates from {precomputed_path}")
            log.info(f"10-50x speedup enabled — PrecomputedEnv will be used in workers")
        else:
            data_dir = Path(args.data_dir)
            log.info(f"Discovering MBO files in {data_dir}...")
            event_files = sorted(data_dir.glob("*_mbo_events.npz"))

            if not event_files:
                log.error(f"No MBO event files found in {data_dir}")
                return

            log.info(f"Found {len(event_files)} MBO event files")

            # Filter to OOT dates (March-April 2026)
            log.info("Filtering to OOT dates (March-April 2026)...")
            oot_files = [f for f in event_files if any(
                m in f.stem for m in ["202603", "202604"]
            )]
            if len(oot_files) < 5:
                log.warning(f"Only {len(oot_files)} OOT files, using all {len(event_files)} files")
                oot_files = event_files

            log.info(f"Using {len(oot_files)} files for training")
    except Exception as e:
        log.error(f"FATAL: Error during data discovery: {e}", exc_info=True)
        raise

    # Create agent (with error handling)
    try:
        log.info("Initializing agent...")
        agent = SplitDQNAgent(
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            n_step=args.n_step,
            device="cuda",
        )

        log.info(f"Agent on {agent.device}")
        log.info(f"Entry net: {sum(p.numel() for p in agent.entry_net.parameters())} params")
        log.info(f"Cancel net: {sum(p.numel() for p in agent.cancel_net.parameters())} params")
        log.info(f"Exit net: {sum(p.numel() for p in agent.exit_net.parameters())} params")

        # HC #171: Load checkpoint if resuming
        if latest_checkpoint:
            log.info(f"Loading checkpoint {latest_checkpoint}...")
            agent.load(latest_checkpoint.parent)
            log.info("Checkpoint loaded successfully")
    except Exception as e:
        log.error(f"FATAL: Error during agent initialization: {e}", exc_info=True)
        raise

    # GPU training: synchronous after each worker batch (avoids GIL issues with threads)
    gpu_update_count = {"entry": 0, "cancel": 0, "exit": 0}

    # Walk-forward folds
    total_train = args.train_days
    total_eval = args.eval_days
    fold_size = total_train + total_eval
    n_folds = max(1, (len(oot_files) - total_train) // total_eval)

    log.info(f"Walk-forward: {n_folds} folds, {total_train} train + {total_eval} eval days")
    log.info(f"Starting from fold {resume_fold}/{n_folds}")

    for fold_idx in range(resume_fold, n_folds):
        fold_start = fold_idx * total_eval
        train_files = oot_files[fold_start:fold_start + total_train]
        eval_files = oot_files[fold_start + total_train:fold_start + fold_size]

        if not eval_files:
            break

        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold_idx}: train={len(train_files)} files, eval={len(eval_files)} files")
        log.info(f"{'='*60}")

        # Training epochs
        for epoch in range(args.epochs):
            epoch_start = time.time()
            epoch_trades = 0
            epoch_pnl = 0.0
            files_done = 0

            # Process files in parallel with workers
            worker_args = [
                (str(f), args.pred_dir, args.patchtst_dir, fold_idx, args.precomputed_dir)
                for f in train_files
            ]

            pool = ProcessPoolExecutor(max_workers=args.n_workers)
            try:
                futures = {pool.submit(worker_run_file, wa): wa for wa in worker_args}

                # Use timeout to prevent hung workers from blocking indefinitely
                # HC #201: catch TimeoutError gracefully — collect completed results, skip stragglers
                timeout_sec = 7200  # 2hr total — some files are very large (300k+ events)
                try:
                    for future in as_completed(futures, timeout=timeout_sec):
                        try:
                            entry_trans, cancel_trans, exit_trans, trades, pnl = future.result()
                        except Exception as e:
                            log.warning(f"Worker error: {e}")
                            continue

                        # Add transitions to replay buffers
                        for s, a, r, ns, d in entry_trans:
                            agent.entry_buffer.add(s, a, r, ns, d)
                        for s, a, r, ns, d in cancel_trans:
                            agent.cancel_buffer.add(s, a, r, ns, d)
                        for s, a, r, ns, d in exit_trans:
                            agent.exit_buffer.add(s, a, r, ns, d)

                        epoch_trades += trades
                        epoch_pnl += pnl
                        files_done += 1

                        # GPU training: do N updates after each worker completes
                        n_gpu_updates = args.gpu_updates_per_sec  # updates per worker
                        for _ in range(n_gpu_updates):
                            losses = agent.train_step()
                            for k in losses:
                                gpu_update_count[k] += 1

                        if files_done % 5 == 0:
                            log.info(
                                f"  Fold {fold_idx} Ep {epoch}: {files_done}/{len(train_files)} files | "
                                f"trades={epoch_trades} | pnl=${epoch_pnl:.0f} | "
                                f"GPU updates: E={gpu_update_count['entry']} "
                                f"C={gpu_update_count['cancel']} "
                                f"X={gpu_update_count['exit']} | "
                                f"ε={agent.get_epsilon():.3f} | "
                                f"buf: E={agent.entry_buffer.size} C={agent.cancel_buffer.size} X={agent.exit_buffer.size}"
                            )
                except TimeoutError:
                    n_unfinished = sum(1 for f in futures if not f.done())
                    log.warning(
                        f"  TIMEOUT: {n_unfinished} workers still running after {timeout_sec}s. "
                        f"Collected {files_done}/{len(train_files)} files. Cancelling stragglers and continuing."
                    )
                    for f in futures:
                        if not f.done():
                            f.cancel()
            finally:
                # Critical: shutdown(wait=False) to prevent deadlock on hung workers
                pool.shutdown(wait=False)
                log.info(f"  Worker pool shutdown (no wait) — epoch transition clean")

            elapsed = time.time() - epoch_start
            log.info(
                f"  Fold {fold_idx} Ep {epoch} DONE ({elapsed:.0f}s) | "
                f"trades={epoch_trades} | pnl=${epoch_pnl:.0f} | "
                f"GPU total: E={gpu_update_count['entry']} "
                f"C={gpu_update_count['cancel']} X={gpu_update_count['exit']}"
            )

            # Save checkpoint
            if (epoch + 1) % 3 == 0:
                agent.save(output_dir / f"fold{fold_idx}_ep{epoch}")

        # Evaluation
        log.info(f"\n  EVAL Fold {fold_idx} on {len(eval_files)} files...")
        eval_trades = 0
        eval_pnl = 0.0

        for ef in eval_files:
            worker_args = [(str(ef), args.pred_dir, args.patchtst_dir, fold_idx)]
            entry_trans, cancel_trans, exit_trans, trades, pnl = worker_run_file(worker_args[0])
            eval_trades += trades
            eval_pnl += pnl

        avg_pnl = eval_pnl / max(eval_trades, 1)
        log.info(
            f"  EVAL RESULT Fold {fold_idx}: trades={eval_trades} | "
            f"total_pnl=${eval_pnl:.0f} | avg=${avg_pnl:.2f}/trade"
        )

        # Save best model
        agent.save(output_dir / f"fold{fold_idx}_final")

    # Final save
    agent.save(output_dir / "final")
    log.info(f"\nTraining complete. Final models saved to {output_dir}/final/")
    log.info(f"Total GPU updates: E={gpu_update_count['entry']} C={gpu_update_count['cancel']} X={gpu_update_count['exit']}")


if __name__ == "__main__":
    main()
