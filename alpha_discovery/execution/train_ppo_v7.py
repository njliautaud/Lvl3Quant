#!/usr/bin/env python3
# WARNING: This uses a SIMPLIFIED environment, NOT true FIFO. See fifo_rl_env.py for production.
"""
PPO v7: Clean Execution RL Trainer
===================================
Self-contained PPO trainer for ES futures execution using CNN-Mamba + PatchTST signals.

Design philosophy (v5 principles, restored after v6 bugs):
  - DECISION_STRIDE = 50 events per step (~5ms market time)
  - Raw PnL ticks as reward (NOT delta Sortino — avoids negative EV trap)
  - Idle penalty (-0.002/skip when flat) + EOD no-trade penalty (-2.0)
  - Labels used as PnL proxy (future mid-price move, ticks) — simplification that works
  - 20-dim observation (simple and interpretable)
  - 5 flat discrete actions (no market/limit split complexity)

v6 bugs this fixes:
  1. 1-step-per-event → DECISION_STRIDE=50 events per step (closer to event-by-event)
  2. Mid-price drift → labels_10s used directly as PnL proxy
  3. Prediction index broken after RTH skip → pred_idx = step * stride_ratio, clamped
  4. Delta Sortino negative EV → raw PnL ticks
  5. No idle penalty → -0.002 per skip step when flat
  6. Alpha-gate penalty + broken signal → removed alpha gate entirely
  7. Queue fill never triggers → simplified: wait > 3 strides = auto-cancel

HC compliance:
  - HC #0:  Sliding walk-forward (train=25d, eval=3d)
  - HC #69: Report Sharpe, Sortino, PF, WR (not raw P&L as primary)
  - HC #89: Costs = 0.376 ticks commission RT only. NO spread cost. P&L = exit - entry - 0.376t.

Usage:
  python train_ppo_v7.py --data-dir /path/to/mbo_events_smart_v3 \\
                         --pred-dir /path/to/cnn_mamba_v2_bulk_oot \\
                         --patchtst-dir /path/to/patchtst_bulk_oot \\
                         --output-dir ./ppo_v7_output

Author: Claude (Infrastructure Builder)
Date: 2026-05-03
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
log = logging.getLogger("ppo_v7")

# ── Constants (HC canonical) ───────────────────────────────────────────────────
COMMISSION_COST = 0.376   # ticks round-trip commission only (HC #89) — NO spread cost

DECISION_STRIDE = 50      # events per RL step (~5ms market time, closer to event-by-event)
WARMUP_EVENTS   = 10_000  # skip this many events to land in RTH
RTH_START_UTC   = 13 * 3600 + 30 * 60  # 9:30 AM ET = 13:30 UTC (seconds since midnight)
RTH_END_UTC     = 21 * 3600             # 4:00 PM ET = 21:00 UTC

# Environment parameters
TP_TICKS        = 6       # take-profit (ticks)
SL_TICKS        = 6       # stop-loss (ticks)
MAX_HOLD_STEPS  = 100     # ~0.5s @ 50 events/step
MAX_WAIT_STEPS  = 3       # strides before limit order auto-cancels
OBS_DIM         = 20
N_ACTIONS       = 5

# Reward shaping
IDLE_PENALTY    = -0.002  # per skip step when flat
EOD_NO_TRADE    = -2.0    # if 0 trades at EOD
EOD_TRADE_BONUS = +0.5    # if >= 3 trades at EOD
REWARD_CLIP     = 5.0

# Prediction alignment: predictions have stride=50 events → one pred per 50 events
PRED_STRIDE     = 50      # events per prediction (how models were trained)

# ── Observation indices (for documentation) ────────────────────────────────────
# [0:3]   CNN-Mamba preds (1s, 5s, 10s) × 10
# [3:6]   PatchTST preds (1s, 5s, 10s) × 10
# [6]     Confluence (CNN × PatchTST sign agreement)
# [7:9]   Book: spread_ticks, book_imbalance
# [9]     Price momentum: mean(labels_10s over last stride) × 100
# [10:14] Position: in_position, direction, unrealized_pnl/10, hold_steps/100
# [14:16] MFE/MAE: max_favorable/10, max_adverse/10
# [16:18] Time: tod_sin, tod_cos
# [18:20] Stats: rolling_win_rate, trade_count/50


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def find_valid_dates(pred_dir: str, patchtst_dir: str) -> List[str]:
    """
    Find dates that have BOTH CNN-Mamba AND PatchTST prediction files.
    Returns sorted list of date strings 'YYYYMMDD'.
    """
    pred_path = Path(pred_dir)
    pst_path  = Path(patchtst_dir)

    cnn_dates = set()
    for f in pred_path.glob("*_predictions.npz"):
        # Expect YYYYMMDD_predictions.npz
        stem = f.stem.replace("_predictions", "")
        if len(stem) == 8 and stem.isdigit():
            cnn_dates.add(stem)

    pst_dates = set()
    for f in pst_path.glob("*_predictions.npz"):
        stem = f.stem.replace("_predictions", "")
        if len(stem) == 8 and stem.isdigit():
            pst_dates.add(stem)

    valid = sorted(cnn_dates & pst_dates)
    return valid


def load_day(date: str, data_dir: str, pred_dir: str, patchtst_dir: str) -> Optional[dict]:
    """
    Load one day's MBO events + predictions. Returns None if any file is missing.
    Returns dict with keys: events, labels_1s, labels_5s, labels_10s, timestamps,
                            cnn_preds, pst_preds
    """
    data_path    = Path(data_dir)
    cnn_path     = Path(pred_dir)
    pst_path_dir = Path(patchtst_dir)

    # MBO events file
    # Try both naming conventions
    ev_file = data_path / f"{date}_mbo_events_smart_v3.npz"
    if not ev_file.exists():
        ev_file = data_path / f"{date}_mbo_events.npz"
    if not ev_file.exists():
        log.debug(f"Missing MBO events: {ev_file}")
        return None

    # Prediction files
    cnn_file = cnn_path / f"{date}_predictions.npz"
    pst_file = pst_path_dir / f"{date}_predictions.npz"
    if not cnn_file.exists() or not pst_file.exists():
        log.debug(f"Missing predictions for {date}")
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

        # Predictions: (M, 3) for 1s/5s/10s horizons
        cnn_preds = cnn_data["predictions"].astype(np.float32)  # (M, 3)
        pst_preds = pst_data["predictions"].astype(np.float32)  # (M, 3)

        return dict(
            events=events, timestamps=timestamps,
            labels_1s=labels_1s, labels_5s=labels_5s, labels_10s=labels_10s,
            cnn_preds=cnn_preds, pst_preds=pst_preds,
        )
    except Exception as e:
        log.warning(f"Failed to load {date}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  TRADING ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class TradingEnv:
    """
    Single-day MBO replay environment.
    One RL step = DECISION_STRIDE events processed at once.
    Labels used as PnL proxy for mid-price moves.
    """

    def __init__(self, day_data: dict):
        self.events     = day_data["events"]       # (N, 25)
        self.timestamps = day_data["timestamps"]   # (N,) ns
        self.labels_10s = day_data["labels_10s"]  # (N,) ticks
        self.cnn_preds  = day_data["cnn_preds"]   # (M, 3)
        self.pst_preds  = day_data["pst_preds"]   # (M, 3)
        self.N          = len(self.events)

        # Derived constants for this day
        self.n_steps = max(1, (self.N - WARMUP_EVENTS) // DECISION_STRIDE)
        # Pred alignment: step s → pred index = (s * DECISION_STRIDE) // PRED_STRIDE
        self.M_preds = len(self.cnn_preds)

        # State variables reset on episode start
        self.reset()

    def reset(self) -> np.ndarray:
        self.step_idx         = 0
        self.event_cursor     = WARMUP_EVENTS   # start after warmup

        # Position tracking
        self.in_position      = False
        self.direction        = 0               # +1 long, -1 short
        self.entry_label_idx  = 0              # event index at entry
        self.hold_steps       = 0
        self.max_favorable    = 0.0
        self.max_adverse      = 0.0

        # Pending limit order
        self.pending_order    = False
        self.pending_dir      = 0
        self.pending_wait     = 0              # strides waited

        # Episode stats
        self.trade_pnls       = []
        self.trade_count      = 0
        self.win_count        = 0

        return self._get_obs()

    def _get_pred_idx(self, step: int) -> int:
        """Map step index to prediction array index. Clamp to valid range."""
        event_offset = step * DECISION_STRIDE
        pred_idx = event_offset // PRED_STRIDE
        return max(0, min(pred_idx, self.M_preds - 1))

    def _get_tod_normalized(self, event_idx: int) -> Tuple[float, float]:
        """Time of day as sin/cos, normalized over [RTH_START_UTC, RTH_END_UTC]."""
        ts_ns  = self.timestamps[min(event_idx, self.N - 1)]
        ts_s   = ts_ns / 1e9
        tod_s  = ts_s % 86400  # seconds since midnight UTC
        tod_frac = (tod_s - RTH_START_UTC) / (RTH_END_UTC - RTH_START_UTC)
        tod_frac = max(0.0, min(1.0, tod_frac))
        return math.sin(math.pi * tod_frac), math.cos(math.pi * tod_frac)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        s   = self.step_idx
        cur = min(self.event_cursor, self.N - 1)

        # ── Predictions ──────────────────────────────────────────────────────
        p = self._get_pred_idx(s)
        cnn = self.cnn_preds[p]   # (3,) — 1s, 5s, 10s
        pst = self.pst_preds[p]   # (3,)
        obs[0:3] = cnn * 10.0     # scale up
        obs[3:6] = pst * 10.0

        # ── Confluence ───────────────────────────────────────────────────────
        # Positive when both models agree on direction (same sign), negative otherwise
        # Use 10s horizon as anchor
        conf = np.mean(np.sign(cnn) * np.sign(pst))
        obs[6] = float(conf)

        # ── Book state (from events in current window) ────────────────────────
        lo = max(0, cur - DECISION_STRIDE)
        hi = cur
        window_events = self.events[lo:hi]
        if len(window_events) > 0:
            obs[7] = float(np.mean(window_events[:, 5]))            # spread_ticks
            # Book imbalance from cols 6 (bid depth) and 7 (ask depth) if available
            if window_events.shape[1] > 7:
                bid_d = np.mean(window_events[:, 6])
                ask_d = np.mean(window_events[:, 7])
                denom = abs(bid_d) + abs(ask_d) + 1e-8
                obs[8] = float((bid_d - ask_d) / denom)

        # ── Price momentum ───────────────────────────────────────────────────
        if len(window_events) > 0:
            obs[9] = float(np.mean(self.labels_10s[lo:hi])) * 100.0

        # ── Position state ───────────────────────────────────────────────────
        obs[10] = 1.0 if self.in_position else 0.0
        obs[11] = float(self.direction)
        # Unrealized PnL: sum of labels_10s since entry (proxy)
        if self.in_position:
            entry_cur = self.entry_label_idx
            unrealized = float(np.sum(self.labels_10s[entry_cur:cur])) * self.direction
            obs[12] = unrealized / 10.0
        obs[13] = float(self.hold_steps) / 100.0

        # ── MFE/MAE ──────────────────────────────────────────────────────────
        obs[14] = self.max_favorable / 10.0
        obs[15] = self.max_adverse   / 10.0

        # ── Time of day ──────────────────────────────────────────────────────
        obs[16], obs[17] = self._get_tod_normalized(cur)

        # ── Rolling stats ─────────────────────────────────────────────────────
        if self.trade_count > 0:
            obs[18] = self.win_count / self.trade_count
        obs[19] = float(self.trade_count) / 50.0

        # NaN guard — replace any NaN/Inf with 0
        np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Process one DECISION_STRIDE of events and apply action.
        Returns: (obs, reward, done)
        """
        reward = 0.0
        done   = False

        # ── Update event cursor ───────────────────────────────────────────────
        prev_cursor = self.event_cursor
        self.event_cursor = min(self.event_cursor + DECISION_STRIDE, self.N)
        cur = self.event_cursor

        # ── Check if episode is done ──────────────────────────────────────────
        if self.step_idx >= self.n_steps - 1:
            done = True
            # Force close position at EOD
            if self.in_position:
                pnl = self._close_position(prev_cursor, cur, market=True)
                reward += pnl
            # EOD trade count bonus/penalty
            if self.trade_count == 0:
                reward += EOD_NO_TRADE
            elif self.trade_count >= 3:
                reward += EOD_TRADE_BONUS
            self.step_idx += 1
            return self._get_obs(), float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP)), done

        # ── Pending limit order: check if filled or cancel ────────────────────
        if self.pending_order:
            self.pending_wait += 1
            if self.pending_wait > MAX_WAIT_STEPS:
                # Auto-cancel: price moved, order never filled
                self.pending_order = False
                self.pending_dir   = 0
                self.pending_wait  = 0
            else:
                # Simulate fill: count trade events at our price level
                # Simplification: if spread is tight and events occurred, assume fill
                window = self.events[prev_cursor:cur]
                trade_mask = (window[:, 1] < 0.1)  # event_type ~= 0 (trade)
                n_trades   = int(np.sum(trade_mask))
                if n_trades > DECISION_STRIDE * 0.01:  # >1% trade events → fill
                    # Filled! Enter position
                    self.in_position     = True
                    self.direction       = self.pending_dir
                    self.entry_label_idx = cur
                    self.hold_steps      = 0
                    self.max_favorable   = 0.0
                    self.max_adverse     = 0.0
                    self.pending_order   = False
                    self.pending_dir     = 0
                    self.pending_wait    = 0

        # ── TP/SL/MaxHold management for open position ────────────────────────
        if self.in_position:
            self.hold_steps += 1
            # Cumulative PnL since entry (labels proxy)
            cumulative_pnl = float(np.sum(
                self.labels_10s[self.entry_label_idx:cur]
            )) * self.direction

            # Track MFE/MAE
            if cumulative_pnl > self.max_favorable:
                self.max_favorable = cumulative_pnl
            if -cumulative_pnl > self.max_adverse:
                self.max_adverse = -cumulative_pnl

            # Check TP/SL
            if cumulative_pnl >= TP_TICKS or cumulative_pnl <= -SL_TICKS:
                pnl = self._close_position(prev_cursor, cur, market=True)
                reward += pnl
            elif self.hold_steps >= MAX_HOLD_STEPS:
                # Force close: held too long
                pnl = self._close_position(prev_cursor, cur, market=True)
                reward += pnl

        # ── Apply action ─────────────────────────────────────────────────────
        if not done and not self.in_position and not self.pending_order:
            if action == 1:   # ENTER LONG (limit buy at bid)
                self.pending_order = True
                self.pending_dir   = +1
                self.pending_wait  = 0
            elif action == 2:  # ENTER SHORT (limit sell at ask)
                self.pending_order = True
                self.pending_dir   = -1
                self.pending_wait  = 0
            elif action == 0:  # SKIP when flat → idle penalty
                reward += IDLE_PENALTY
        elif not done:
            if action == 3:    # EXIT position (market)
                if self.in_position:
                    pnl = self._close_position(prev_cursor, cur, market=True)
                    reward += pnl
                elif self.pending_order:
                    self.pending_order = False
                    self.pending_dir   = 0
                    self.pending_wait  = 0
            elif action == 4:  # CANCEL pending
                if self.pending_order:
                    self.pending_order = False
                    self.pending_dir   = 0
                    self.pending_wait  = 0

        self.step_idx += 1
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        return self._get_obs(), reward, done

    def _close_position(self, ev_lo: int, ev_hi: int, market: bool = True) -> float:
        """
        Close position and compute PnL.
        Uses accumulated labels as exit price proxy.
        Cost: COMMISSION_COST (0.376 ticks RT) only — NO spread cost per user directive.
        P&L = exit_price - entry_price - 0.376 ticks commission.
        """
        entry_idx = self.entry_label_idx
        cur       = ev_hi

        # PnL = directional sum of labels from entry to now
        gross_pnl = float(np.nansum(self.labels_10s[entry_idx:cur])) * self.direction
        if not np.isfinite(gross_pnl):
            gross_pnl = 0.0

        # Cost = RT commission only (0.376 ticks). No spread cost.
        net_pnl = gross_pnl - COMMISSION_COST

        # Reset position
        self.in_position   = False
        self.direction     = 0
        self.hold_steps    = 0
        self.max_favorable = 0.0
        self.max_adverse   = 0.0

        # Track stats
        self.trade_count += 1
        self.trade_pnls.append(net_pnl)
        if net_pnl > 0:
            self.win_count += 1

        return net_pnl


# ══════════════════════════════════════════════════════════════════════════════
#  ACTOR-CRITIC NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Simple MLP actor-critic. Shared body, separate heads.
    hidden_dim × 2 layers, tanh activation.
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head  = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat   = self.body(x)
        logits = self.policy_head(feat)
        value  = self.value_head(feat).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        logp   = dist.log_prob(action)
        return action, logp, value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist   = Categorical(logits=logits)
        logp   = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, value, entropy


# ══════════════════════════════════════════════════════════════════════════════
#  ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores transitions for one PPO update. Supports batch appending."""

    def __init__(self):
        self.obs:     List[np.ndarray] = []
        self.actions: List[int]        = []
        self.logps:   List[float]      = []
        self.rewards: List[float]      = []
        self.values:  List[float]      = []
        self.dones:   List[bool]       = []

    def add(self, obs, action, logp, reward, value, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.logps.append(logp)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)

    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """GAE-lambda advantage estimation."""
        n       = len(self.rewards)
        rewards = np.array(self.rewards, dtype=np.float32)
        values  = np.array(self.values,  dtype=np.float32)
        dones   = np.array(self.dones,   dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0
        for t in reversed(range(n)):
            next_val  = values[t + 1] if t + 1 < n else 0.0
            next_done = dones[t]
            delta     = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
            gae       = delta + gamma * lam * (1.0 - next_done) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def to_tensors(self, device: torch.device) -> dict:
        advantages, returns = self.compute_advantages()
        # Normalize advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return dict(
            obs     = torch.tensor(np.array(self.obs),     dtype=torch.float32).to(device),
            actions = torch.tensor(self.actions,           dtype=torch.long).to(device),
            logps   = torch.tensor(self.logps,             dtype=torch.float32).to(device),
            returns = torch.tensor(returns,                dtype=torch.float32).to(device),
            adv     = torch.tensor(adv,                    dtype=torch.float32).to(device),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PPO UPDATE
# ══════════════════════════════════════════════════════════════════════════════

def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    mini_batch: int = 2048,
    ppo_epochs: int = 4,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
) -> dict:
    """
    Standard PPO update on collected rollout.
    Returns dict of training metrics.
    """
    data = buffer.to_tensors(device)
    n    = len(buffer)
    metrics = defaultdict(list)

    for _ in range(ppo_epochs):
        indices = torch.randperm(n, device=device)
        for start in range(0, n, mini_batch):
            idx = indices[start:start + mini_batch]
            obs_b     = data["obs"][idx]
            act_b     = data["actions"][idx]
            old_logp  = data["logps"][idx]
            ret_b     = data["returns"][idx]
            adv_b     = data["adv"][idx]

            new_logp, value, entropy = model.evaluate(obs_b, act_b)

            # Policy loss (clipped surrogate)
            ratio    = torch.exp(new_logp - old_logp)
            surr1    = ratio * adv_b
            surr2    = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
            pol_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            val_loss = F.mse_loss(value, ret_b)

            # Entropy bonus (encourages exploration)
            ent_loss = -entropy.mean()

            loss = pol_loss + value_coef * val_loss + entropy_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            metrics["pol_loss"].append(pol_loss.item())
            metrics["val_loss"].append(val_loss.item())
            metrics["entropy"].append(-ent_loss.item())
            metrics["clip_frac"].append(
                ((ratio - 1).abs() > clip_eps).float().mean().item()
            )

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  EPISODE RUNNER + METRICS
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(env: TradingEnv, model: ActorCritic, device: torch.device,
                greedy: bool = False, buffer: Optional[RolloutBuffer] = None) -> dict:
    """
    Run one full trading day episode.
    If greedy=True, uses argmax policy (eval mode).
    If buffer is provided, appends transitions for training.
    Returns episode stats.
    """
    obs  = env.reset()
    done = False
    total_reward = 0.0
    model.eval() if greedy else model.train()

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if greedy:
                logits, value = model(obs_t)
                dist   = Categorical(logits=logits)
                action = logits.argmax(dim=-1)
                logp   = dist.log_prob(action)
            else:
                action, logp, value = model.act(obs_t)

        a    = action.item()
        lp   = logp.item()
        val  = value.item()

        obs_next, reward, done = env.step(a)

        if buffer is not None:
            buffer.add(obs, a, lp, reward, val, done)

        obs           = obs_next
        total_reward += reward

    return dict(
        total_reward = total_reward,
        trade_count  = env.trade_count,
        trade_pnls   = env.trade_pnls,
    )


def compute_metrics(all_pnls: List[float], trade_count: float, n_days: int) -> dict:
    """Compute Sharpe, Sortino, PF, WR from list of trade PnLs."""
    if not all_pnls or trade_count == 0:
        return dict(sharpe=0.0, sortino=0.0, pf=0.0, wr=0.0,
                    avg_pnl=0.0, trades_per_day=0.0)

    pnls   = np.array(all_pnls, dtype=np.float64)
    n      = len(pnls)
    mu     = pnls.mean()
    std    = pnls.std() + 1e-8
    sharpe = mu / std * math.sqrt(n)  # trade-basis Sharpe

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
#  WALK-FORWARD TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_dates: List[str],
    data_dir: str, pred_dir: str, patchtst_dir: str,
    device: torch.device,
    args: argparse.Namespace,
    fold_idx: int,
) -> List[dict]:
    """Train model on one fold's train days (lazy loading). Returns per-epoch metrics."""
    import gc
    epoch_metrics = []

    for epoch in range(args.epochs):
        buffer       = RolloutBuffer()
        all_pnls     = []
        total_trades = 0

        # Collect rollouts across all train days (load one at a time)
        for date in train_dates:
            day_data = load_day(date, data_dir, pred_dir, patchtst_dir)
            if day_data is None:
                continue
            env = TradingEnv(day_data)
            ep  = run_episode(env, model, device, greedy=False, buffer=buffer)
            all_pnls.extend(ep["trade_pnls"])
            total_trades += ep["trade_count"]
            del day_data, env  # free memory
            gc.collect()

            # PPO update once rollout is full enough
            if len(buffer) >= args.rollout_len:
                ppo_stats = ppo_update(
                    model, optimizer, buffer, device,
                    mini_batch=args.mini_batch,
                    entropy_coef=args.entropy_coef,
                )
                buffer = RolloutBuffer()  # reset

        # Flush remaining buffer
        if len(buffer) > 0:
            ppo_stats = ppo_update(
                model, optimizer, buffer, device,
                mini_batch=min(args.mini_batch, len(buffer)),
                entropy_coef=args.entropy_coef,
            )

        scheduler.step()

        m = compute_metrics(all_pnls, total_trades, len(train_dates))
        m["epoch"] = epoch
        m["fold"]  = fold_idx
        m["lr"]    = optimizer.param_groups[0]["lr"]
        epoch_metrics.append(m)

        log.info(
            f"  Fold {fold_idx} Epoch {epoch:2d} | "
            f"Sortino={m['sortino']:+.3f}  Sharpe={m['sharpe']:+.3f}  "
            f"WR={m['wr']:.1%}  PF={m['pf']:.2f}  "
            f"Trades/day={m['trades_per_day']:.1f}  "
            f"AvgPnL={m['avg_pnl']:+.3f}t"
        )

    return epoch_metrics


def eval_fold(
    model: ActorCritic,
    eval_dates: List[str],
    data_dir: str, pred_dir: str, patchtst_dir: str,
    device: torch.device,
    fold_idx: int,
) -> dict:
    """Evaluate model on eval days (greedy policy, lazy loading). Returns eval metrics."""
    import gc
    all_pnls     = []
    total_trades = 0

    for date in eval_dates:
        day_data = load_day(date, data_dir, pred_dir, patchtst_dir)
        if day_data is None:
            continue
        env = TradingEnv(day_data)
        ep  = run_episode(env, model, device, greedy=True, buffer=None)
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
        f"Trades/day={m['trades_per_day']:.1f}  "
        f"AvgPnL={m['avg_pnl']:+.3f}t"
    )
    return m


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO v7: Clean Execution RL Trainer")
    p.add_argument("--data-dir",     required=True, help="Path to mbo_events_smart_v3/")
    p.add_argument("--pred-dir",     required=True, help="Path to cnn_mamba_v2_bulk_oot/")
    p.add_argument("--patchtst-dir", required=True, help="Path to patchtst_bulk_oot/")
    p.add_argument("--output-dir",   required=True, help="Output directory for checkpoints")
    p.add_argument("--hidden-dim",   type=int,   default=128,    help="MLP hidden dim")
    p.add_argument("--epochs",       type=int,   default=15,     help="Epochs per fold")
    p.add_argument("--train-days",   type=int,   default=25,     help="Training days per fold")
    p.add_argument("--eval-days",    type=int,   default=3,      help="Eval days per fold")
    p.add_argument("--lr",           type=float, default=3e-4,   help="Learning rate")
    p.add_argument("--rollout-len",  type=int,   default=65536,  help="Rollout steps before PPO update")
    p.add_argument("--mini-batch",   type=int,   default=2048,   help="PPO mini-batch size")
    p.add_argument("--entropy-coef", type=float, default=0.02,   help="Entropy coefficient")
    p.add_argument("--device",       type=str,   default="cuda", help="torch device")
    p.add_argument("--no-mlflow",    action="store_true",         help="Disable MLflow logging")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if str(device) == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # ── Output dir ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find valid dates ─────────────────────────────────────────────────────
    valid_dates = find_valid_dates(args.pred_dir, args.patchtst_dir)
    if len(valid_dates) < args.train_days + args.eval_days:
        log.error(
            f"Not enough dates: found {len(valid_dates)}, "
            f"need {args.train_days + args.eval_days}"
        )
        sys.exit(1)

    # ── Validate dates exist (lazy loading — DON'T preload, too much RAM) ────
    log.info(f"Validating {len(valid_dates)} OOT dates (lazy loading)...")
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

    # ── Build model ─────────────────────────────────────────────────────────
    model     = ActorCritic(OBS_DIM, N_ACTIONS, args.hidden_dim).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    # ── Walk-forward folds ───────────────────────────────────────────────────
    n_dates  = len(dates_loaded)
    n_folds  = max(1, (n_dates - args.train_days) // args.eval_days)

    print(f"\n{'='*60}")
    print("=== PPO v7: Clean Design (v5 principles + multi-model) ===")
    print(f"Dates: {len(dates_loaded)} OOT dates ({dates_loaded[0]} to {dates_loaded[-1]})")
    print(f"Folds: {n_folds} (train={args.train_days}d, eval={args.eval_days}d)")
    print(f"Obs dim: {OBS_DIM}, Actions: {N_ACTIONS}, Params: {n_params:,}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # ── MLflow setup ─────────────────────────────────────────────────────────
    if MLFLOW_OK and not getattr(args, 'no_mlflow', False):
        try:
            mlflow.set_experiment("PPO_v7_Execution")
            mlflow.start_run(run_name=f"ppo_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params(vars(args))
            mlflow.log_param("n_params", n_params)
            mlflow.log_param("n_dates",  len(dates_loaded))
            mlflow.log_param("n_folds",  n_folds)
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")
            mlflow_active = False
    else:
        mlflow_active = False

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

        # LR scheduler: cosine decay over epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )

        # Train (lazy loading)
        epoch_metrics = train_fold(
            model, optimizer, scheduler, train_dates_fold,
            args.data_dir, args.pred_dir, args.patchtst_dir,
            device, args, fold
        )

        # Eval (lazy loading)
        eval_metrics = eval_fold(
            model, eval_dates_fold,
            args.data_dir, args.pred_dir, args.patchtst_dir,
            device, fold
        )
        all_eval_metrics.append(eval_metrics)

        # Log to MLflow
        if MLFLOW_OK and mlflow_active:
            try:
                prefix = f"fold{fold}/"
                mlflow.log_metrics({
                    prefix + "eval_sortino": eval_metrics["sortino"],
                    prefix + "eval_sharpe":  eval_metrics["sharpe"],
                    prefix + "eval_wr":      eval_metrics["wr"],
                    prefix + "eval_pf":      eval_metrics["pf"],
                    prefix + "eval_tpd":     eval_metrics["trades_per_day"],
                }, step=fold)
            except Exception:
                pass

        # Save best checkpoint by eval Sortino
        if eval_metrics["sortino"] > best_sortino:
            best_sortino   = eval_metrics["sortino"]
            best_ckpt_path = str(out_dir / f"best_model_fold{fold}.pt")
            torch.save(
                dict(
                    model_state  = model.state_dict(),
                    optimizer    = optimizer.state_dict(),
                    fold         = fold,
                    eval_metrics = eval_metrics,
                    args         = vars(args),
                    obs_dim      = OBS_DIM,
                    n_actions    = N_ACTIONS,
                ),
                best_ckpt_path,
            )
            log.info(f"  *** New best checkpoint: Sortino={best_sortino:.3f} → {best_ckpt_path}")

    # ── Final summary ────────────────────────────────────────────────────────
    if all_eval_metrics:
        all_sortino = [m["sortino"]      for m in all_eval_metrics]
        all_sharpe  = [m["sharpe"]       for m in all_eval_metrics]
        all_wr      = [m["wr"]           for m in all_eval_metrics]
        all_tpd     = [m["trades_per_day"] for m in all_eval_metrics]
        all_pf      = [m["pf"]           for m in all_eval_metrics]

        print("\n" + "="*60)
        print("WALK-FORWARD SUMMARY (all eval folds)")
        print(f"  Sortino:     {np.mean(all_sortino):+.3f} ± {np.std(all_sortino):.3f}")
        print(f"  Sharpe:      {np.mean(all_sharpe):+.3f} ± {np.std(all_sharpe):.3f}")
        print(f"  Win Rate:    {np.mean(all_wr):.1%}")
        print(f"  Profit Factor: {np.mean(all_pf):.2f}")
        print(f"  Trades/day:  {np.mean(all_tpd):.1f}")
        print(f"  Best Sortino fold: {np.argmax(all_sortino)} ({max(all_sortino):.3f})")
        print(f"  Best checkpoint: {best_ckpt_path}")
        print("="*60)

        if MLFLOW_OK and mlflow_active:
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
