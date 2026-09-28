#!/usr/bin/env python3
"""
Exit-Only DQN — Learns optimal EXIT and CANCEL timing
======================================================

Design principle: Signal models (CNN-Mamba/PatchTST) handle ENTRY timing.
This model ONLY learns:
  - When to EXIT an open position (market exit vs hold)
  - When to CANCEL a pending limit order

Uses precomputed observations for 10-50x speedup over raw MBO replay.

Architecture:
  - Single DQN with 4 actions: HOLD, EXIT_MARKET, CANCEL_PENDING, EXIT_LIMIT
  - Entries triggered automatically by signal model confidence thresholds
  - MFE-capture ratio as primary reward signal
  - Walk-forward validated, FIFO-based

HC refs: #239 (exit optimization priority), #240 (Jupiter exit-only model),
         #241 (use precomputed obs), #221 (MFE-capture ratio)

Author: Claude (Execution Research)
Date: 2026-05-07
"""

from __future__ import annotations

import os
import sys
import math
import time
import json
import signal
import logging
import argparse
import resource
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Setup ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("exit_dqn")

# ─── Constants ────────────────────────────────────────────────────────────────
TICK_VALUE = 12.50
COMMISSION_COST = 0.376  # RT commission in ticks (HC canonical)
MAX_HOLD_SECS = 60.0     # Data-driven: signal edge gone by ~30s, hard cap at 60s (HC #226)
SIGNAL_DECAY_HALFLIFE_S = 5.0  # Slower decay for exit decisions

# Entry thresholds — confidence-tiered (HC #243)
# Agent knows which tier triggered the entry
ENTRY_TIERS = [
    (1.50, 3),   # Top ~1%  → tier 3
    (1.00, 2),   # Top ~5%  → tier 2
    (0.50, 1),   # Top ~10% → tier 1
    (0.30, 0),   # Top ~20% → tier 0 (lowest confidence)
]
ENTRY_COOLDOWN_SECS = 2.0          # Reduced from 5s — more trades for learning

# Exit action space
ACTION_HOLD = 0          # Continue holding position
ACTION_EXIT_MARKET = 1   # Market exit (cross spread)
ACTION_CANCEL = 2        # Cancel pending limit order (if any)
ACTION_EXIT_LIMIT = 3    # Place exit limit at favorable price
N_ACTIONS = 4

# Observation dimensions for exit model
# We use a focused subset + exit-specific features
EXIT_OBS_DIM = 24  # Compact observation for exit decisions

# Training
GAMMA = 0.99
TAU = 0.005         # Soft target update
BATCH_SIZE = 512
BUFFER_SIZE = 200_000
LR = 1e-4
N_STEP = 10         # Multi-step returns
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 10_000   # v3: was 200K but only ~300 steps/epoch — never decayed
HUBER_DELTA = 1.0   # Huber loss (HC #211)
REWARD_CLIP = 3.0   # Clip rewards (HC #219)

# Memory cap
MAX_RAM_GB = 55  # Jupiter has 64GB, leave headroom

# ─── Replay Buffer ────────────────────────────────────────────────────────────

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class NStepReplayBuffer:
    """N-step replay buffer with priority-free uniform sampling."""

    def __init__(self, capacity: int, n_step: int, gamma: float):
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def _compute_n_step(self):
        """Compute n-step return from buffer."""
        reward = 0.0
        for i, trans in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * trans.reward
        first = self.n_step_buffer[0]
        last = self.n_step_buffer[-1]
        return Transition(
            state=first.state,
            action=first.action,
            reward=reward,
            next_state=last.next_state,
            done=last.done,
        )

    def push(self, state, action, reward, next_state, done):
        self.n_step_buffer.append(
            Transition(state, action, reward, next_state, done)
        )
        if len(self.n_step_buffer) == self.n_step or done:
            self.buffer.append(self._compute_n_step())
            if done:
                # Flush remaining partial n-step returns
                while len(self.n_step_buffer) > 1:
                    self.n_step_buffer.popleft()
                    if self.n_step_buffer:
                        self.buffer.append(self._compute_n_step())
                self.n_step_buffer.clear()

    def sample(self, batch_size: int):
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]
        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ─── Exit DQN Network ────────────────────────────────────────────────────────

class ExitDQN(nn.Module):
    """
    Compact DQN for exit/cancel decisions.

    Input: 24-dim exit-focused observation
    Output: Q-values for 4 actions (hold, exit_market, cancel, exit_limit)
    """

    def __init__(self, obs_dim: int = EXIT_OBS_DIM, n_actions: int = N_ACTIONS, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ─── Exit-Only Environment (wraps PrecomputedEnv) ────────────────────────────

class ExitOnlyEnv:
    """
    Environment that auto-enters on signal model predictions and
    only gives the agent control over EXIT/CANCEL decisions.

    Observation (24 dims):
    [0:3]   Signal at entry: pred_1s, pred_5s, pred_10s
    [3]     Signal confidence tier (0-3)
    [4:7]   Current signal: pred_1s, pred_5s, pred_10s
    [7]     Signal decay fraction (entry signal remaining)
    [8:10]  Book state: imbalance, spread_ticks
    [10]    Position direction (+1/-1)
    [11]    Unrealized PnL (ticks, normalized)
    [12]    Hold time (seconds, normalized)
    [13:15] MFE/MAE (ticks, normalized)
    [15]    MFE capture ratio so far
    [16]    Price momentum
    [17:19] Flow: event_density, flow_imbalance
    [19]    Has pending exit limit (0/1)
    [20]    Queue fraction filled (for exit limit)
    [21]    Realized vol
    [22:24] Time of day: sin, cos
    """

    def __init__(self, precomputed_dir: Path, entry_threshold: float = 0.30, min_tier: int = 0):
        self.precomputed_dir = Path(precomputed_dir)
        self.entry_threshold = entry_threshold  # Minimum threshold (lowest tier)
        self.min_tier = min_tier  # Minimum tier to accept (0=all, 2=Top5%+, 3=Top1%+)

        # Discover dates
        obs_files = sorted(self.precomputed_dir.glob("*_base_obs.npy"))
        if not obs_files:
            raise FileNotFoundError(f"No precomputed obs in {self.precomputed_dir}")
        self._dates = [f.stem.replace("_base_obs", "") for f in obs_files]
        log.info(f"ExitOnlyEnv: {len(self._dates)} dates available")

        self._rng = np.random.default_rng()

        # Episode state
        self._base_obs = None
        self._meta = None
        self._n_steps = 0
        self._step_idx = 0
        self._date = ""

        # Position state
        self._position = 0          # 0, +1, -1
        self._entry_price = 0.0
        self._entry_ts = 0.0
        self._entry_pred_1s = 0.0
        self._entry_pred_5s = 0.0
        self._entry_pred_10s = 0.0
        self._entry_tier = 0        # Confidence tier (0-3)
        self._mfe = 0.0
        self._mae = 0.0
        self._last_cooldown_ts = -999.0

        # Pending exit limit
        self._exit_limit_price = 0.0
        self._exit_limit_active = False
        self._exit_limit_queue_pos = 0.0
        self._exit_limit_consumed = 0.0

        # Trade tracking
        self._trades = []
        self._episode_pnl = 0.0

    def reset(self, date: Optional[str] = None) -> np.ndarray:
        """Start episode. Returns obs only when in position (skips to first entry)."""
        if date is None:
            date = self._rng.choice(self._dates)
        if date not in self._dates:
            date = self._rng.choice(self._dates)

        self._date = date
        self._base_obs = np.load(
            str(self.precomputed_dir / f"{date}_base_obs.npy"), mmap_mode='r'
        )
        self._meta = np.load(
            str(self.precomputed_dir / f"{date}_meta.npy"), mmap_mode='r'
        )
        self._n_steps = len(self._base_obs)
        self._step_idx = 0
        self._position = 0
        self._entry_price = 0.0
        self._entry_ts = 0.0
        self._mfe = 0.0
        self._mae = 0.0
        self._exit_limit_active = False
        self._trades = []
        self._episode_pnl = 0.0
        self._last_cooldown_ts = -999.0

        # Advance to first entry
        self._advance_to_entry()
        return self._get_obs()

    def _advance_to_entry(self):
        """Scan forward to next high-confidence signal and auto-enter."""
        while self._step_idx < self._n_steps - 100:  # Leave room for exit
            meta = self._meta[self._step_idx]
            pred_1s = float(meta[4])
            pred_5s = float(meta[5])
            pred_10s = float(meta[6])
            ts_s = float(meta[0])
            price_rel = float(meta[1])
            spread = float(meta[3])

            signal_mag = abs(pred_1s)

            if signal_mag >= self.entry_threshold and (ts_s - self._last_cooldown_ts) >= ENTRY_COOLDOWN_SECS:
                # Determine confidence tier (HC #243: model knows which tier)
                tier = 0
                for threshold, t in ENTRY_TIERS:
                    if signal_mag >= threshold:
                        tier = t
                        break

                # Skip if below minimum tier requirement
                if tier < self.min_tier:
                    self._step_idx += 1
                    continue

                # Auto-enter via limit at bid/ask
                half_spread = max(spread / 2.0, 0.5)
                if pred_1s > 0:
                    self._position = 1
                    self._entry_price = price_rel - half_spread
                else:
                    self._position = -1
                    self._entry_price = price_rel + half_spread

                self._entry_ts = ts_s
                self._entry_pred_1s = pred_1s
                self._entry_pred_5s = pred_5s
                self._entry_pred_10s = pred_10s
                self._entry_tier = tier
                self._mfe = 0.0
                self._mae = 0.0
                self._exit_limit_active = False
                return

            self._step_idx += 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Process exit action.
        Returns (obs, reward, done, info).
        """
        if self._step_idx >= self._n_steps - 1 or self._position == 0:
            return self._get_obs(), 0.0, True, self._get_info()

        meta = self._meta[self._step_idx]
        ts_s = float(meta[0])
        price_rel = float(meta[1])
        spread = float(meta[3])
        half_spread = max(spread / 2.0, 0.5)

        # Update MFE/MAE
        unrealized = (price_rel - self._entry_price) * self._position
        if unrealized > self._mfe:
            self._mfe = unrealized
        if -unrealized > self._mae:
            self._mae = -unrealized

        reward = 0.0
        closed = False

        # ── Check exit limit fill ──
        if self._exit_limit_active:
            if self._position == 1:  # Long, exit limit = sell at ask
                if price_rel >= self._exit_limit_price - 0.3:
                    self._exit_limit_consumed += 1.0
            else:  # Short, exit limit = buy at bid
                if price_rel <= self._exit_limit_price + 0.3:
                    self._exit_limit_consumed += 1.0

            if self._exit_limit_consumed >= self._exit_limit_queue_pos:
                # Limit fill!
                reward = self._close_position(self._exit_limit_price, ts_s, "exit_limit")
                closed = True

        if not closed:
            # ── Apply action ──
            if action == ACTION_HOLD:
                # Per-step shaping: small penalty for holding too long
                hold_secs = ts_s - self._entry_ts
                if hold_secs > MAX_HOLD_SECS * 0.8:
                    reward -= 0.002  # Gentle pressure to exit

            elif action == ACTION_EXIT_MARKET:
                if self._position == 1:
                    exit_price = price_rel - half_spread  # Sell at bid
                else:
                    exit_price = price_rel + half_spread  # Buy at ask
                reward = self._close_position(exit_price, ts_s, "exit_market")
                closed = True

            elif action == ACTION_CANCEL:
                if self._exit_limit_active:
                    self._exit_limit_active = False
                    # Small penalty for canceling — cost of indecision
                    reward = -0.01

            elif action == ACTION_EXIT_LIMIT:
                if not self._exit_limit_active and self._position != 0:
                    # Place exit limit at favorable side
                    if self._position == 1:
                        self._exit_limit_price = price_rel + half_spread  # Sell at ask
                        queue_depth = float(self._meta[self._step_idx][10]) if self._meta.shape[1] > 10 else 20.0
                    else:
                        self._exit_limit_price = price_rel - half_spread  # Buy at bid
                        queue_depth = float(self._meta[self._step_idx][11]) if self._meta.shape[1] > 11 else 20.0
                    self._exit_limit_queue_pos = max(queue_depth, 5.0)
                    self._exit_limit_consumed = 0.0
                    self._exit_limit_active = True

        # ── Force close at max hold ──
        hold_secs = ts_s - self._entry_ts
        if not closed and hold_secs >= MAX_HOLD_SECS:
            if self._position == 1:
                exit_price = price_rel - half_spread
            else:
                exit_price = price_rel + half_spread
            reward = self._close_position(exit_price, ts_s, "time_stop")
            closed = True

        # ── Advance step ──
        self._step_idx += 1

        # ── If closed, advance to next entry ──
        done = False
        if closed:
            self._last_cooldown_ts = ts_s
            if self._step_idx < self._n_steps - 100:
                self._advance_to_entry()
                if self._position == 0:
                    done = True  # No more entries found
            else:
                done = True

        # Check if episode exhausted
        if self._step_idx >= self._n_steps - 1:
            if self._position != 0:
                # Force close EOD
                meta_final = self._meta[min(self._step_idx, self._n_steps - 1)]
                price_final = float(meta_final[1])
                spread_final = float(meta_final[3])
                hs = max(spread_final / 2.0, 0.5)
                ep = price_final - hs if self._position == 1 else price_final + hs
                reward += self._close_position(ep, float(meta_final[0]), "eod")
            done = True

        obs = self._get_obs()
        info = self._get_info()

        return obs, float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP)), done, info

    def _close_position(self, exit_price: float, ts_s: float, reason: str) -> float:
        """Close position and compute confidence-weighted MFE-capture reward."""
        raw_move = (exit_price - self._entry_price) * self._position
        pnl_ticks = raw_move - COMMISSION_COST
        hold_secs = ts_s - self._entry_ts

        # ── MFE-capture ratio (primary reward per HC #221/#239) ──
        mfe_denom = max(self._mfe, 0.5)  # Floor to avoid div-by-zero
        mfe_capture = pnl_ticks / mfe_denom  # Can be negative if loss
        mfe_capture = max(-1.0, min(1.5, mfe_capture))

        # ── Confidence multiplier (HC #234/#243) ──
        # Higher tier = higher reward/penalty multiplier
        # Tier 0 (top 20%): 0.5x, Tier 1 (top 10%): 0.75x, Tier 2 (top 5%): 1.0x, Tier 3 (top 1%): 1.5x
        conf_multipliers = [0.5, 0.75, 1.0, 1.5]
        conf_mult = conf_multipliers[min(self._entry_tier, 3)]

        # ── Reward components ──
        # 1. MFE capture ratio (weight 0.6, confidence-weighted)
        reward = 0.6 * mfe_capture * conf_mult

        # 2. Raw PnL signal (weight 0.2, bounded)
        pnl_norm = max(-2.0, min(2.0, pnl_ticks / 5.0))
        reward += 0.2 * pnl_norm

        # 3. MAE penalty — excessive drawdown before exit (weight 0.1)
        if self._mae > 0 and self._mfe > 0:
            mae_ratio = self._mae / max(self._mfe, 0.5)
            reward -= 0.1 * min(mae_ratio, 2.0)

        # 4. Hold time penalty — decays with time past signal horizon
        if hold_secs > 30.0:
            overage_frac = min((hold_secs - 30.0) / 30.0, 1.0)
            reward -= 0.1 * overage_frac

        # 5. Exit method bonus
        if reason == "exit_limit" and pnl_ticks > 0:
            reward += 0.05  # Bonus for passive exit (no spread crossing)
        elif reason == "time_stop":
            reward -= 0.15 * conf_mult  # Stronger penalty for high-conf time stops

        # Store trade
        self._trades.append({
            "direction": self._position,
            "pnl_ticks": pnl_ticks,
            "pnl_usd": pnl_ticks * TICK_VALUE,
            "hold_secs": hold_secs,
            "mfe_ticks": self._mfe,
            "mae_ticks": self._mae,
            "mfe_capture": float(mfe_capture),
            "exit_reason": reason,
            "entry_signal_1s": self._entry_pred_1s,
        })
        self._episode_pnl += pnl_ticks

        # Reset position
        self._position = 0
        self._exit_limit_active = False

        return reward

    def _get_obs(self) -> np.ndarray:
        """Build 24-dim exit-focused observation."""
        obs = np.zeros(EXIT_OBS_DIM, dtype=np.float32)

        if self._step_idx >= self._n_steps or self._position == 0:
            return obs

        base = self._base_obs[self._step_idx]
        meta = self._meta[self._step_idx]
        ts_s = float(meta[0])
        price_rel = float(meta[1])
        spread = float(meta[3])

        # [0:3] Entry signal memory
        obs[0] = max(-3.0, min(3.0, self._entry_pred_1s))
        obs[1] = max(-3.0, min(3.0, self._entry_pred_5s))
        obs[2] = max(-3.0, min(3.0, self._entry_pred_10s))

        # [3] Confidence tier at entry (from actual tier assignment, not re-derived)
        obs[3] = float(self._entry_tier) / 3.0  # Normalized 0-1

        # [4:7] Current signal (from base obs: indices 0,1,2 = pred_1s, pred_5s, pred_10s)
        obs[4] = float(base[0])
        obs[5] = float(base[1])
        obs[6] = float(base[2])

        # [7] Signal decay fraction
        hold_secs = ts_s - self._entry_ts
        decay = math.exp(-0.693 * hold_secs / SIGNAL_DECAY_HALFLIFE_S)
        obs[7] = max(0.0, min(1.0, decay))

        # [8:10] Book state (from base obs: indices 6=imbalance, 7=spread)
        obs[8] = float(base[6])   # book imbalance
        obs[9] = float(base[7])   # spread ticks

        # [10] Position direction
        obs[10] = float(self._position)

        # [11] Unrealized PnL (normalized)
        unrealized = (price_rel - self._entry_price) * self._position
        obs[11] = max(-3.0, min(3.0, unrealized / 5.0))

        # [12] Hold time (normalized by max hold)
        obs[12] = max(0.0, min(2.0, hold_secs / MAX_HOLD_SECS))

        # [13:15] MFE/MAE (normalized)
        obs[13] = max(0.0, min(3.0, self._mfe / 5.0))
        obs[14] = max(0.0, min(3.0, self._mae / 5.0))

        # [15] Current MFE capture ratio
        if self._mfe > 0.5:
            obs[15] = max(-1.0, min(1.5, unrealized / self._mfe))

        # [16] Price momentum (from base obs: index 9)
        obs[16] = float(base[9])

        # [17:19] Flow features (from base obs: indices 10=event_density, 12=flow_imbalance)
        obs[17] = float(base[10])  # event density
        obs[18] = float(base[12])  # flow imbalance

        # [19] Has pending exit limit
        obs[19] = 1.0 if self._exit_limit_active else 0.0

        # [20] Queue fraction filled
        if self._exit_limit_active and self._exit_limit_queue_pos > 0:
            obs[20] = min(1.0, self._exit_limit_consumed / self._exit_limit_queue_pos)

        # [21] Realized vol (from base obs: index 7 in the base... let me use index 16-range)
        # base indices [16,17,18] in 48-dim = vol, tod_sin, tod_cos → stored at base positions [10,11,12]
        # Actually: BASE_INDICES = [0..9, 16..23, 34..44]
        # So base[10] = obs48[16] = realized_vol, base[11] = obs48[17] = tod_sin, base[12] = obs48[18] = tod_cos
        # Wait, I already used base[10] above for event_density... let me re-check
        # BASE_INDICES = [0,1,2,3,4,5,6,7,8,9, 16,17,18,19,20,21,22,23, 34,...,44]
        # base[0..9] = obs48[0..9], base[10] = obs48[16]=vol, base[11] = obs48[17]=tod_sin
        # base[12] = obs48[18]=tod_cos, base[13] = obs48[19]=event_density
        # base[14] = obs48[20]=buy_vol_frac, base[15] = obs48[21]=flow_imbal
        # base[16] = obs48[22]=cancel_rate, base[17] = obs48[23]=event_rate
        # Correction:
        obs[17] = float(base[13])  # event_density (obs48[19])
        obs[18] = float(base[15])  # flow_imbalance (obs48[21])
        obs[21] = float(base[10])  # realized_vol (obs48[16])

        # [22:24] Time of day
        obs[22] = float(base[11])  # tod_sin (obs48[17])
        obs[23] = float(base[12])  # tod_cos (obs48[18])

        return obs

    def _get_info(self) -> dict:
        n_trades = len(self._trades)
        if n_trades == 0:
            return {"n_trades": 0, "episode_pnl_ticks": 0.0, "date": self._date}

        pnls = [t["pnl_ticks"] for t in self._trades]
        mfe_caps = [t["mfe_capture"] for t in self._trades]
        reasons = [t["exit_reason"] for t in self._trades]

        return {
            "n_trades": n_trades,
            "episode_pnl_ticks": self._episode_pnl,
            "episode_pnl_usd": self._episode_pnl * TICK_VALUE,
            "win_rate": sum(1 for p in pnls if p > 0) / max(n_trades, 1),
            "avg_pnl_ticks": np.mean(pnls),
            "avg_mfe_capture": np.mean(mfe_caps),
            "avg_hold_secs": np.mean([t["hold_secs"] for t in self._trades]),
            "exit_reasons": {r: reasons.count(r) for r in set(reasons)},
            "date": self._date,
        }

    def get_episode_metrics(self) -> dict:
        """Full episode metrics for evaluation."""
        if not self._trades:
            return {"n_trades": 0, "sortino": 0.0, "sharpe": 0.0, "profit_factor": 0.0,
                    "win_rate": 0.0, "avg_mfe_capture": 0.0, "total_pnl_ticks": 0.0}

        pnls = np.array([t["pnl_ticks"] for t in self._trades])
        n = len(pnls)
        mean_pnl = float(pnls.mean())
        std_pnl = float(pnls.std()) if n > 1 else 1.0
        sharpe = mean_pnl / (std_pnl + 1e-8) * math.sqrt(n)
        downside = pnls[pnls < 0]
        dd = float(np.sqrt(np.mean(downside ** 2))) if len(downside) > 0 else 1e-8
        sortino = mean_pnl / (dd + 1e-8)
        wins = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
        losses = abs(float(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-8
        pf = wins / losses

        mfe_caps = [t["mfe_capture"] for t in self._trades]
        holds = [t["hold_secs"] for t in self._trades]
        reasons = [t["exit_reason"] for t in self._trades]

        return {
            "n_trades": n,
            "sortino": sortino,
            "sharpe": sharpe,
            "profit_factor": pf,
            "win_rate": float((pnls > 0).mean()),
            "avg_mfe_capture": float(np.mean(mfe_caps)),
            "avg_pnl_ticks": mean_pnl,
            "total_pnl_ticks": float(pnls.sum()),
            "total_pnl_usd": float(pnls.sum()) * TICK_VALUE,
            "avg_hold_secs": float(np.mean(holds)),
            "avg_mfe": float(np.mean([t["mfe_ticks"] for t in self._trades])),
            "avg_mae": float(np.mean([t["mae_ticks"] for t in self._trades])),
            "exit_reasons": {r: reasons.count(r) for r in set(reasons)},
        }


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_exit_dqn(args):
    """Main training loop with walk-forward validation."""

    precomputed_dir = Path(args.precomputed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Set up logging to file
    fh = logging.FileHandler(output_dir / "training.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(fh)
    logging.getLogger().addHandler(fh)

    log.info(f"Exit-Only DQN Training Starting")
    log.info(f"Config: {json.dumps(config, indent=2)}")

    # Discover dates
    env = ExitOnlyEnv(precomputed_dir, entry_threshold=args.entry_threshold)
    all_dates = sorted(env._dates)
    n_dates = len(all_dates)
    log.info(f"Total dates available: {n_dates}")

    # Walk-forward setup
    train_size = args.train_dates
    eval_size = args.eval_dates
    fold_step = eval_size  # Slide by eval_size dates

    n_folds = max(1, (n_dates - train_size) // fold_step)
    log.info(f"Walk-forward: {n_folds} folds, {train_size} train + {eval_size} eval dates each")

    # Device
    device = torch.device("cpu")  # Jupiter is CPU-only
    log.info(f"Device: {device}")

    # Results tracking
    all_fold_results = []

    for fold in range(n_folds):
        fold_start = fold * fold_step
        train_dates = all_dates[fold_start:fold_start + train_size]
        eval_dates = all_dates[fold_start + train_size:fold_start + train_size + eval_size]

        if not eval_dates:
            log.info(f"Fold {fold}: no eval dates left, stopping")
            break

        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold}/{n_folds-1}: Train on {len(train_dates)} dates [{train_dates[0]}..{train_dates[-1]}], "
                 f"Eval on {len(eval_dates)} dates [{eval_dates[0]}..{eval_dates[-1]}]")

        # Initialize networks
        dropout_rate = getattr(args, 'dropout', 0.0)
        wd = getattr(args, 'weight_decay', 1e-5)
        q_net = ExitDQN(EXIT_OBS_DIM, N_ACTIONS, dropout=dropout_rate).to(device)
        target_net = ExitDQN(EXIT_OBS_DIM, N_ACTIONS, dropout=0.0).to(device)  # No dropout on target
        target_net.load_state_dict(q_net.state_dict())
        optimizer = torch.optim.AdamW(q_net.parameters(), lr=LR, weight_decay=wd)

        replay_buffer = NStepReplayBuffer(BUFFER_SIZE, N_STEP, GAMMA)

        total_steps = 0
        total_trades = 0
        q_losses = []

        for epoch in range(args.epochs):
            epoch_start = time.time()
            epoch_trades = 0
            epoch_pnl = 0.0
            epoch_mfe_capture = []
            epoch_rewards = []

            # Shuffle train dates each epoch
            rng = np.random.default_rng(epoch * 1000 + fold)
            shuffled_dates = rng.permutation(train_dates).tolist()

            for date_idx, date in enumerate(shuffled_dates):
                # RAM check
                import psutil
                ram_gb = psutil.Process().memory_info().rss / 1e9
                if ram_gb > MAX_RAM_GB:
                    log.warning(f"RAM {ram_gb:.1f}GB > {MAX_RAM_GB}GB cap, skipping to eval")
                    break

                train_env = ExitOnlyEnv(precomputed_dir, entry_threshold=args.entry_threshold)
                obs = train_env.reset(date=date)
                done = False
                ep_reward = 0.0

                while not done:
                    # Epsilon-greedy
                    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)
                    if np.random.random() < eps:
                        action = np.random.randint(N_ACTIONS)
                    else:
                        with torch.no_grad():
                            q_vals = q_net(torch.FloatTensor(obs).unsqueeze(0).to(device))
                            action = int(q_vals.argmax(dim=1).item())

                    next_obs, reward, done, info = train_env.step(action)
                    replay_buffer.push(obs, action, reward, next_obs, done)
                    obs = next_obs
                    ep_reward += reward
                    total_steps += 1

                    # Train
                    if len(replay_buffer) >= BATCH_SIZE and total_steps % 4 == 0:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
                        states_t = torch.FloatTensor(states).to(device)
                        actions_t = torch.LongTensor(actions).to(device)
                        rewards_t = torch.FloatTensor(rewards).to(device)
                        next_states_t = torch.FloatTensor(next_states).to(device)
                        dones_t = torch.FloatTensor(dones).to(device)

                        # Double DQN
                        with torch.no_grad():
                            next_actions = q_net(next_states_t).argmax(dim=1)
                            next_q = target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                            target_q = rewards_t + (GAMMA ** N_STEP) * next_q * (1 - dones_t)

                        current_q = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                        loss = F.huber_loss(current_q, target_q, delta=HUBER_DELTA)

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                        optimizer.step()

                        q_losses.append(loss.item())

                        # Soft target update
                        for p, tp in zip(q_net.parameters(), target_net.parameters()):
                            tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

                # Episode stats
                metrics = train_env.get_episode_metrics()
                epoch_trades += metrics["n_trades"]
                epoch_pnl += metrics.get("total_pnl_ticks", 0)
                if metrics["n_trades"] > 0:
                    epoch_mfe_capture.append(metrics["avg_mfe_capture"])
                epoch_rewards.append(ep_reward)
                total_trades += metrics["n_trades"]

                if (date_idx + 1) % 5 == 0 or date_idx == len(shuffled_dates) - 1:
                    avg_loss = np.mean(q_losses[-100:]) if q_losses else 0.0
                    eps_val = max(EPS_END, EPS_START - (EPS_START - EPS_END) * total_steps / EPS_DECAY_STEPS)
                    log.info(f"  Fold {fold} Ep {epoch}: {date_idx+1}/{len(shuffled_dates)} dates | "
                             f"trades={epoch_trades} | pnl={epoch_pnl:.0f}tk | "
                             f"MFE_cap={np.mean(epoch_mfe_capture):.3f} | "
                             f"Q_loss={avg_loss:.4f} | ε={eps_val:.3f} | "
                             f"buf={len(replay_buffer)}")

            # Epoch summary
            epoch_time = time.time() - epoch_start
            log.info(f"  Fold {fold} Ep {epoch} DONE ({epoch_time:.0f}s) | "
                     f"trades={epoch_trades} | pnl=${epoch_pnl * TICK_VALUE:.0f} | "
                     f"avg_MFE_cap={np.mean(epoch_mfe_capture) if epoch_mfe_capture else 0:.3f}")

        # ── Evaluation ──
        log.info(f"\n--- FOLD {fold} EVALUATION on {eval_dates} ---")
        q_net.eval()
        eval_all_trades = []

        eval_thresh = args.eval_threshold if args.eval_threshold is not None else args.entry_threshold
        eval_min_tier = args.eval_min_tier if args.eval_min_tier is not None else 0
        for date in eval_dates:
            eval_env = ExitOnlyEnv(precomputed_dir, entry_threshold=eval_thresh, min_tier=eval_min_tier)
            obs = eval_env.reset(date=date)
            done = False

            while not done:
                with torch.no_grad():
                    q_vals = q_net(torch.FloatTensor(obs).unsqueeze(0).to(device))
                    action = int(q_vals.argmax(dim=1).item())
                obs, reward, done, info = eval_env.step(action)

            metrics = eval_env.get_episode_metrics()
            log.info(f"  Eval {date}: {metrics['n_trades']} trades, "
                     f"Sortino={metrics.get('sortino', 0):.3f}, WR={metrics.get('win_rate', 0):.1%}, "
                     f"PF={metrics.get('profit_factor', 0):.2f}, MFE_cap={metrics.get('avg_mfe_capture', 0):.3f}, "
                     f"P&L=${metrics.get('total_pnl_usd', 0):.0f}, "
                     f"exits={metrics.get('exit_reasons', {})}")
            eval_all_trades.extend(eval_env._trades)

        # Fold summary
        if eval_all_trades:
            eval_pnls = np.array([t["pnl_ticks"] for t in eval_all_trades])
            eval_mfe_caps = [t["mfe_capture"] for t in eval_all_trades]
            eval_n = len(eval_pnls)
            eval_wr = float((eval_pnls > 0).mean())
            eval_mean = float(eval_pnls.mean())
            eval_dd = eval_pnls[eval_pnls < 0]
            eval_sortino = eval_mean / (float(np.sqrt(np.mean(eval_dd**2))) + 1e-8) if len(eval_dd) > 0 else eval_mean * 10
            fold_result = {
                "fold": fold,
                "n_trades": eval_n,
                "win_rate": eval_wr,
                "sortino": eval_sortino,
                "avg_mfe_capture": float(np.mean(eval_mfe_caps)),
                "total_pnl_ticks": float(eval_pnls.sum()),
                "total_pnl_usd": float(eval_pnls.sum()) * TICK_VALUE,
                "avg_pnl_ticks": eval_mean,
                "eval_dates": eval_dates,
            }
            all_fold_results.append(fold_result)
            log.info(f"\n  FOLD {fold} OOT RESULT: {eval_n} trades, WR={eval_wr:.1%}, "
                     f"Sortino={eval_sortino:.3f}, MFE_cap={np.mean(eval_mfe_caps):.3f}, "
                     f"P&L=${float(eval_pnls.sum()) * TICK_VALUE:.0f}")

        # Save checkpoint
        ckpt_dir = output_dir / f"fold{fold}"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({
            "q_net": q_net.state_dict(),
            "target_net": target_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "fold": fold,
            "total_steps": total_steps,
        }, ckpt_dir / "checkpoint.pt")

        q_net.train()

    # ── Final Summary ──
    log.info(f"\n{'='*60}")
    log.info(f"TRAINING COMPLETE — {len(all_fold_results)} folds evaluated")
    if all_fold_results:
        concat_trades = sum(r["n_trades"] for r in all_fold_results)
        concat_pnl = sum(r["total_pnl_ticks"] for r in all_fold_results)
        avg_wr = np.mean([r["win_rate"] for r in all_fold_results])
        avg_sortino = np.mean([r["sortino"] for r in all_fold_results])
        avg_mfe_cap = np.mean([r["avg_mfe_capture"] for r in all_fold_results])

        log.info(f"  Concat: {concat_trades} trades, WR={avg_wr:.1%}, "
                 f"Sortino={avg_sortino:.3f}, MFE_cap={avg_mfe_cap:.3f}, "
                 f"P&L=${concat_pnl * TICK_VALUE:.0f}")

        with open(output_dir / "results.json", "w") as f:
            json.dump(all_fold_results, f, indent=2, default=str)

    log.info("Done.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Exit-Only DQN Training")
    parser.add_argument("--precomputed-dir", type=str, required=True,
                        help="Directory with precomputed obs .npy files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Training epochs per fold")
    parser.add_argument("--train-dates", type=int, default=40,
                        help="Number of training dates per fold")
    parser.add_argument("--eval-dates", type=int, default=5,
                        help="Number of eval dates per fold")
    parser.add_argument("--entry-threshold", type=float, default=0.50,
                        help="Signal confidence threshold for auto-entry (training)")
    parser.add_argument("--eval-threshold", type=float, default=None,
                        help="Entry threshold for eval (default: same as entry-threshold)")
    parser.add_argument("--eval-min-tier", type=int, default=None,
                        help="Minimum confidence tier for eval entries (0-3, e.g. 2=Top5%+)")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate for regularization (e.g. 0.15)")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay")
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=BUFFER_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_exit_dqn(args)
