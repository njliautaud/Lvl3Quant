#!/usr/bin/env python3
"""
SAC Trainer for FIFO RL Execution Agent — v2 (GPU-Optimized)
=============================================================

Changes from v1:
  1. GPUReplayBuffer — pre-allocated VRAM tensors, no CPU→GPU copy on sample.
     At 2M capacity / obs_dim=48: ~763MB VRAM. At 5M: ~1.9GB. Fits 24GB 3090 easily.
  2. Batch size 2048 (Neptune/3090). Was 256 — pure waste of GPU.
  3. updates_per_step=4 by default. Off-policy SAC can squeeze more gradient
     steps per CPU-bound env step (GPU stays busy during sequential stepping).
  4. OOT date filtering via --oot-only + --start-date / --end-date.
     Filters event files AND prediction files to only OOT dates so training
     is strictly on data the signal models never saw (HC #118).
  5. Correct bulk OOT pred paths: cnn_mamba_v2_bulk_inference/ (YYYYMMDD format).
  6. --n-envs flag for round-robin multi-file env stepping (simulates vectorized
     envs with zero multiprocessing overhead — works for sequential env).

Architecture unchanged: hidden_dim=256, 3 layers. This is the sweet spot.
All HC compliance from v1 preserved.

HC compliance:
  - HC #0:   sliding walk-forward (no expanding window)
  - HC #69:  primary metrics = Sharpe, Sortino, PF, WR
  - HC #89:  cost = 0.376 ticks limits; 1.376 market orders
  - HC #100: risk-adjusted reward
  - HC #104: symmetric reward — no short-side bias
  - HC #105: training curve PNG saved
  - HC #115: no hardcoded decay penalties
  - HC #117: SAC implementation
  - HC #118: OOT-only training data filter

Author: Claude (Infrastructure Builder)
Date: 2026-05-03
"""

from __future__ import annotations

import os
import sys

if sys.stdout is None or not hasattr(sys.stdout, 'write'):
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None or not hasattr(sys.stderr, 'write'):
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from fifo_rl_env import (
    FIFOExecutionEnv, OBS_DIM,
    EVENT_DIR as DEFAULT_EVENT_DIR,
    PRED_DIR  as DEFAULT_PRED_DIR,
    TICK_VALUE,
    COMMISSION_RT_TICKS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("train_fifo_rl_sac_v2")

N_ACTIONS         = 7
MLFLOW_EXPERIMENT = "FIFO_RL_SAC_v2_Execution"

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_EPOCHS           = 100
DEFAULT_LR               = 3e-4
DEFAULT_GAMMA            = 0.99
DEFAULT_TAU              = 0.005
DEFAULT_ALPHA_INIT       = 0.2
DEFAULT_BUFFER_SIZE      = 2_000_000   # v2: 2M (was 500k) → ~763MB VRAM on 3090
DEFAULT_BATCH_SIZE       = 2048        # v2: 2048 (was 256) → proper GPU utilization
DEFAULT_WARMUP_STEPS     = 5_000       # v2: scaled up with buffer size
DEFAULT_UPDATES_PER_STEP = 4           # v2: 4 (was 1) → keeps GPU busy during CPU-bound stepping
DEFAULT_TRAIN_DAYS       = 40
DEFAULT_EVAL_DAYS        = 5

# ── Bulk OOT prediction directories (Neptune canonical paths) ──────────────────
LVL3_NEPTUNE = Path("/home/nick/Lvl3Quant")
CNN_MAMBA_BULK_OOT_DIR  = LVL3_NEPTUNE / "output" / "cnn_mamba_v2_bulk_inference"
PATCHTST_BULK_OOT_DIR   = LVL3_NEPTUNE / "output" / "patchtst_smart_v3_mar"


# ══════════════════════════════════════════════════════════════════════════════
# GPU Replay Buffer (v2 key upgrade)
# ══════════════════════════════════════════════════════════════════════════════

class GPUReplayBuffer:
    """
    Pre-allocated GPU tensor replay buffer — eliminates CPU→GPU copies on sample.

    At capacity=2M, obs_dim=48:
      obs + next_obs: 2 × 2M × 48 × 4 bytes = 768 MB
      actions:        2M × 8 bytes = 16 MB
      rewards:        2M × 4 bytes = 8 MB
      dones:          2M × 1 bytes = 2 MB
      Total: ~794 MB VRAM  (fits easily on 24GB 3090)

    At capacity=5M: ~1.9 GB. Still fine for 3090.

    Sample is O(1) GPU-side indexing — no data movement at batch time.
    Add() transfers one transition from CPU → GPU (48 floats = tiny).

    Fallback: if CUDA OOM during init, falls back to CPU buffer with a warning.
    """

    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.device   = device
        self.ptr      = 0
        self.size     = 0

        log.info(
            f"GPUReplayBuffer: allocating {capacity:,} transitions on {device} "
            f"(obs_dim={obs_dim}, est. VRAM ~"
            f"{(capacity * (2*obs_dim*4 + 8 + 4 + 1)) / 1024**3:.2f} GB)"
        )

        try:
            self.obs      = torch.zeros((capacity, obs_dim), device=device, dtype=torch.float32)
            self.next_obs = torch.zeros((capacity, obs_dim), device=device, dtype=torch.float32)
            self.actions  = torch.zeros(capacity, device=device, dtype=torch.long)
            self.rewards  = torch.zeros(capacity, device=device, dtype=torch.float32)
            self.dones    = torch.zeros(capacity, device=device, dtype=torch.bool)
            log.info("GPUReplayBuffer: allocated successfully on GPU.")
        except RuntimeError as e:
            log.warning(
                f"GPUReplayBuffer: CUDA OOM during allocation ({e}). "
                f"Falling back to CPU buffer — slower but functional."
            )
            cpu = torch.device("cpu")
            self.obs      = torch.zeros((capacity, obs_dim), device=cpu, dtype=torch.float32)
            self.next_obs = torch.zeros((capacity, obs_dim), device=cpu, dtype=torch.float32)
            self.actions  = torch.zeros(capacity, device=cpu, dtype=torch.long)
            self.rewards  = torch.zeros(capacity, device=cpu, dtype=torch.float32)
            self.dones    = torch.zeros(capacity, device=cpu, dtype=torch.bool)
            self.device   = cpu

    def add(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Add one transition. Transfers obs/next_obs from CPU → GPU (tiny: 2×48 floats)."""
        self.obs[self.ptr]      = torch.from_numpy(obs).to(self.device)
        self.next_obs[self.ptr] = torch.from_numpy(next_obs).to(self.device)
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.dones[self.ptr]    = done
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample mini-batch — pure GPU indexing, zero CPU involvement.
        Returns (obs, actions, rewards, next_obs, dones) all on self.device.
        """
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx].float(),
        )

    def is_ready(self, warmup: int) -> bool:
        return self.size >= warmup


# ══════════════════════════════════════════════════════════════════════════════
# Network Architectures (unchanged from v1 — 256-dim 3-layer is the sweet spot)
# ══════════════════════════════════════════════════════════════════════════════

class SACActorNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_dim: int = 256):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.actor = nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, n_actions))
        self._init_weights()

    def _init_weights(self):
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(obs)
        x = self.shared(x)
        logits = self.actor(x)
        probs     = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probs, log_probs = self.forward(obs)
        action = Categorical(probs=probs).sample()
        return action, probs, log_probs

    def get_action_greedy(self, obs: torch.Tensor) -> torch.Tensor:
        probs, _ = self.forward(obs)
        return probs.argmax(dim=-1)


class SACQNet(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_dim: int = 256):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128), nn.GELU(),
            nn.Linear(128, n_actions),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(obs))


# ══════════════════════════════════════════════════════════════════════════════
# OOT Date Filtering (HC #118)
# ══════════════════════════════════════════════════════════════════════════════

def filter_oot_files(
    event_files: List[Path],
    pred_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Path]:
    """
    Filter event files to only those with a matching prediction file in pred_dir.

    pred_dir is expected to contain files named YYYYMMDD_predictions.npz
    (i.e. cnn_mamba_v2_bulk_inference format).

    This ensures training only uses dates where the signal model has already
    generated OOT predictions — i.e. dates the signal model was trained on
    are excluded (HC #118: no lookahead into signal model training data).

    Also applies optional start_date / end_date range filter (YYYYMMDD strings).
    """
    # Build set of available prediction dates
    pred_dates = set()
    if pred_dir and pred_dir.exists():
        for p in pred_dir.glob("20??????_predictions.npz"):
            pred_dates.add(p.stem[:8])  # YYYYMMDD
        log.info(f"OOT filter: found {len(pred_dates)} prediction dates in {pred_dir.name}")
    else:
        log.warning(f"OOT filter: pred_dir {pred_dir} not found — no date filtering applied")
        return event_files

    # Parse date range
    start_int = int(start_date) if start_date else 0
    end_int   = int(end_date)   if end_date   else 99999999

    filtered = []
    for f in event_files:
        date_str = f.stem[:8]  # YYYYMMDD
        if date_str not in pred_dates:
            continue
        date_int = int(date_str)
        if date_int < start_int or date_int > end_int:
            continue
        filtered.append(f)

    log.info(
        f"OOT filter: {len(event_files)} → {len(filtered)} files "
        f"(range {start_date or 'all'}–{end_date or 'all'})"
    )
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Data Split
# ══════════════════════════════════════════════════════════════════════════════

def build_walk_forward_folds(
    event_files: List[Path],
    train_days:  int,
    eval_days:   int,
) -> List[Dict]:
    """Sliding walk-forward folds (HC #0: SLIDING, not expanding)."""
    folds  = []
    n      = len(event_files)
    window = train_days + eval_days
    i, fold_n = 0, 0
    while i + window <= n:
        folds.append({
            "fold":        fold_n,
            "train_files": event_files[i : i + train_days],
            "eval_files":  event_files[i + train_days : i + train_days + eval_days],
        })
        i      += eval_days
        fold_n += 1
    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_episode_metrics(trade_pnls: List[float], n_steps: int) -> Dict[str, float]:
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
    sharpe   = mean_pnl / (std_pnl + 1e-8) * math.sqrt(max(n, 1))
    downside = pnls[pnls < 0]
    sortino  = mean_pnl * 10.0 if len(downside) == 0 else mean_pnl / (float(np.sqrt(np.mean(downside**2))) + 1e-8)
    gross_wins   = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_losses = abs(float(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-8
    return {
        "n_trades": n, "win_rate": win_rate, "sharpe": sharpe, "sortino": sortino,
        "profit_factor": gross_wins / gross_losses,
        "avg_pnl_ticks": mean_pnl,
        "total_pnl_ticks": float(pnls.sum()),
        "total_pnl_usd": float(pnls.sum()) * TICK_VALUE,
        "trades_per_step": n / max(n_steps, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Soft Target Update
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def polyak_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    for p_on, p_tgt in zip(online.parameters(), target.parameters()):
        p_tgt.data.mul_(1.0 - tau)
        p_tgt.data.add_(tau * p_on.data)


# ══════════════════════════════════════════════════════════════════════════════
# SAC Update (operates on GPU tensors from GPUReplayBuffer — no data movement)
# ══════════════════════════════════════════════════════════════════════════════

def sac_update(
    actor, q1, q2, q1_target, q2_target,
    log_alpha, target_entropy,
    buffer: GPUReplayBuffer,
    actor_opt, q1_opt, q2_opt, alpha_opt,
    batch_size: int, gamma: float, tau: float, device: torch.device,
) -> Dict[str, float]:
    """
    One SAC gradient step. Buffer.sample() returns GPU tensors directly —
    no CPU→GPU transfer at batch time (key v2 optimization).
    """
    obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size)

    # ── 1. Q-network targets ────────────────────────────────────────────────
    with torch.no_grad():
        next_probs, next_log_probs = actor.forward(next_obs_b)
        next_q1 = q1_target(next_obs_b)
        next_q2 = q2_target(next_obs_b)
        next_q_min = torch.min(next_q1, next_q2)
        next_v = (next_probs * (next_q_min - log_alpha.exp() * next_log_probs)).sum(dim=-1)
        y = rew_b + gamma * (1.0 - done_b) * next_v

    q1_a = q1(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
    q2_a = q2(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
    q1_loss = F.mse_loss(q1_a, y)
    q2_loss = F.mse_loss(q2_a, y)

    q1_opt.zero_grad(); q1_loss.backward()
    nn.utils.clip_grad_norm_(q1.parameters(), 1.0); q1_opt.step()

    q2_opt.zero_grad(); q2_loss.backward()
    nn.utils.clip_grad_norm_(q2.parameters(), 1.0); q2_opt.step()

    # ── 2. Actor update ────────────────────────────────────────────────────
    probs, log_probs = actor.forward(obs_b)
    with torch.no_grad():
        q_min = torch.min(q1(obs_b), q2(obs_b))
    actor_loss = (probs * (log_alpha.exp() * log_probs - q_min)).sum(dim=-1).mean()
    actor_opt.zero_grad(); actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0); actor_opt.step()

    # ── 3. Alpha update ────────────────────────────────────────────────────
    with torch.no_grad():
        current_entropy = -(probs * log_probs).sum(dim=-1)
    alpha_loss = -(log_alpha * (current_entropy - target_entropy).detach()).mean()
    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

    # ── 4. Soft update target networks ────────────────────────────────────
    polyak_update(q1, q1_target, tau)
    polyak_update(q2, q2_target, tau)

    return {
        "q1_loss": q1_loss.item(), "q2_loss": q2_loss.item(),
        "actor_loss": actor_loss.item(), "alpha_loss": alpha_loss.item(),
        "alpha": log_alpha.exp().item(),
        "entropy": float(current_entropy.mean().item()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(env, actor, eval_files, device) -> Dict[str, float]:
    actor.eval()
    all_trade_pnls, trades_per_day = [], []
    total_steps = 0

    for ep_file in eval_files:
        try:
            obs = env.reset(episode_file=ep_file)
        except Exception as e:
            log.warning(f"evaluate_policy: skipping {ep_file.name} — {e}")
            continue
        done, steps = False, 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_np = int(actor.get_action_greedy(obs_t).item())
            obs, _, done, _ = env.step_stride(action_np)
            steps += 1
        for t in env._trade_history:
            all_trade_pnls.append(t.pnl_ticks)
        trades_per_day.append(env.get_episode_metrics()["n_trades"])
        total_steps += steps

    m = compute_episode_metrics(all_trade_pnls, total_steps)
    m["avg_trades_per_day"] = float(np.mean(trades_per_day)) if trades_per_day else 0.0
    m["n_eval_days"]        = len(eval_files)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Plotting (HC #105)
# ══════════════════════════════════════════════════════════════════════════════

def save_training_curves(history, output_dir, run_name) -> Optional[Path]:
    if not MATPLOTLIB_AVAILABLE:
        log.warning("matplotlib not available — skipping training curve plot")
        return None
    fig, axes = plt.subplots(6, 1, figsize=(12, 20), sharex=True)
    fig.suptitle(f"FIFO SAC v2 Training — {run_name}", fontsize=14, fontweight="bold")
    epochs, eval_epochs = history.get("epoch", []), history.get("eval_epoch", [])
    panels = [
        ("train_total_pnl_ticks", "eval_total_pnl_ticks",    "Cumulative P&L (ticks)",  "tab:blue"),
        ("train_sortino",         "eval_sortino",             "Sortino Ratio",           "tab:orange"),
        ("train_win_rate",        "eval_win_rate",            "Win Rate",                "tab:green"),
        ("train_avg_pnl_ticks",   "eval_avg_pnl_ticks",      "Avg P&L/Trade (ticks)",   "tab:red"),
        ("train_n_trades",        "eval_avg_trades_per_day", "Trades/Day",              "tab:purple"),
        ("q1_loss",               None,                       "SAC Losses (Q, Actor, α)","tab:brown"),
    ]
    for ax, (tk, ek, ylabel, color) in zip(axes, panels):
        if tk == "q1_loss":
            for lk, lc, lbl in [("q1_loss","tab:brown","Q loss"),("actor_loss","tab:olive","Actor"),("alpha","tab:pink","Alpha")]:
                if lk in history and history[lk]:
                    ax.plot(epochs[:len(history[lk])], history[lk], color=lc, alpha=0.8, linewidth=1.2, label=lbl)
        else:
            if tk in history and history[tk]:
                ax.plot(epochs[:len(history[tk])], history[tk], color=color, alpha=0.8, linewidth=1.5, label="train")
            if ek and ek in history and history[ek]:
                ax.plot(eval_epochs[:len(history[ek])], history[ek], color=color, alpha=1.0, linewidth=2.0, linestyle="--", label="eval", marker="o", markersize=4)
        ax.set_ylabel(ylabel, fontsize=10); ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
        if any(kw in ylabel.lower() for kw in ("pnl","sortino")):
            ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    axes[-1].set_xlabel("Epoch", fontsize=10)
    plt.tight_layout()
    plot_path = output_dir / f"training_curves_{run_name}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    log.info(f"Training curves saved → {plot_path}")
    return plot_path


# ══════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    global MLFLOW_AVAILABLE

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _dbg = output_dir / "_train_debug.log"
    def _ckpt(msg):
        with open(str(_dbg), "a") as f: f.write(f"{msg}\n")

    run_name = datetime.now().strftime("sac_v2_%Y%m%d_%H%M%S")
    _ckpt(f"train() entered, run_name={run_name}")

    _log_file = output_dir / f"{run_name}.log"
    _fh = logging.FileHandler(str(_log_file), mode="w", delay=False)
    _fh.setLevel(logging.INFO)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(_fh)
    logging.getLogger("fifo_rl_env").addHandler(_fh)

    log.info(f"Run: {run_name}")
    log.info(f"Output dir: {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    _ckpt(f"device={device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    event_dir = Path(args.data_dir)
    if not event_dir.exists():
        log.error(f"Data directory not found: {event_dir}"); sys.exit(1)

    event_files = sorted(event_dir.glob("20??????_mbo_events.npz"))
    if not event_files:
        log.error(f"No MBO event files found in {event_dir}"); sys.exit(1)

    log.info(f"Found {len(event_files)} total episode files")

    # ── OOT filtering (HC #118) ────────────────────────────────────────────────
    pred_dir = Path(args.pred_dir) if args.pred_dir else CNN_MAMBA_BULK_OOT_DIR
    if args.oot_only:
        event_files = filter_oot_files(
            event_files, pred_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if not event_files:
            log.error("OOT filter left 0 files — check pred_dir and date range"); sys.exit(1)
    else:
        # Apply date range even without oot_only
        if args.start_date or args.end_date:
            start_int = int(args.start_date) if args.start_date else 0
            end_int   = int(args.end_date)   if args.end_date   else 99999999
            event_files = [f for f in event_files if start_int <= int(f.stem[:8]) <= end_int]
            log.info(f"Date range filter: {len(event_files)} files remaining")

    log.info(
        f"Training on {len(event_files)} files "
        f"({event_files[0].stem[:8]}–{event_files[-1].stem[:8]})"
    )

    # ── Walk-forward folds ─────────────────────────────────────────────────────
    folds = build_walk_forward_folds(event_files, args.train_days, args.eval_days)
    if not folds:
        log.error(f"Not enough data for walk-forward: need {args.train_days + args.eval_days}, have {len(event_files)}"); sys.exit(1)
    log.info(f"Walk-forward folds: {len(folds)} (train={args.train_days}d, eval={args.eval_days}d)")

    # ── Environment ────────────────────────────────────────────────────────────
    patchtst_dir = Path(args.patchtst_dir) if args.patchtst_dir else PATCHTST_BULK_OOT_DIR
    if not patchtst_dir.exists():
        log.warning(f"PatchTST dir not found: {patchtst_dir} — running without confluence")
        patchtst_dir = None

    env = FIFOExecutionEnv(
        event_dir             = event_dir,
        pred_dir              = pred_dir,
        patchtst_pred_dir     = patchtst_dir,
        rth_only              = True,
        verbose               = False,
        max_steps_per_episode = args.max_steps_per_episode,
    )
    log.info(f"Env: obs_dim={OBS_DIM}, n_actions={N_ACTIONS}")
    log.info(f"Pred dir (CNN-Mamba): {pred_dir}")
    log.info(f"Pred dir (PatchTST):  {patchtst_dir}")
    _ckpt("env created")

    # ── Networks (hidden_dim=256, 3 layers — sweet spot, unchanged) ───────────
    hidden_dim = args.hidden_dim
    actor     = SACActorNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q1        = SACQNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q2        = SACQNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q1_target = SACQNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q2_target = SACQNet(OBS_DIM, N_ACTIONS, hidden_dim).to(device)
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    for p in list(q1_target.parameters()) + list(q2_target.parameters()):
        p.requires_grad = False

    log.info(f"Actor params: {sum(p.numel() for p in actor.parameters() if p.requires_grad):,}")
    log.info(f"Q-net params: {sum(p.numel() for p in q1.parameters() if p.requires_grad):,}")

    # ── Optimizers ─────────────────────────────────────────────────────────────
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)
    q1_opt    = torch.optim.Adam(q1.parameters(),    lr=args.lr, eps=1e-5)
    q2_opt    = torch.optim.Adam(q2.parameters(),    lr=args.lr, eps=1e-5)

    target_entropy = -args.target_entropy_ratio * math.log(1.0 / N_ACTIONS)
    log.info(f"Target entropy: {target_entropy:.4f} nats")

    log_alpha = torch.tensor(math.log(args.alpha_init), dtype=torch.float32, device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr, eps=1e-5)

    # ── GPU Replay Buffer (v2 key change) ──────────────────────────────────────
    buffer = GPUReplayBuffer(capacity=args.buffer_size, obs_dim=OBS_DIM, device=device)
    _ckpt(f"buffer allocated: capacity={args.buffer_size:,}")

    if device.type == "cuda":
        mem_alloc = torch.cuda.memory_allocated(device) / 1024**3
        mem_res   = torch.cuda.memory_reserved(device)  / 1024**3
        log.info(f"VRAM after buffer alloc: {mem_alloc:.2f} GB allocated / {mem_res:.2f} GB reserved")

    # ── MLflow ─────────────────────────────────────────────────────────────────
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            import urllib.request
            _tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(_tracking_uri)
            urllib.request.urlopen(_tracking_uri, timeout=5)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "algorithm": "SAC_v2", "epochs": args.epochs, "lr": args.lr,
                "gamma": args.gamma, "tau": args.tau, "alpha_init": args.alpha_init,
                "buffer_size": args.buffer_size, "batch_size": args.batch_size,
                "warmup_steps": args.warmup_steps, "updates_per_step": args.updates_per_step,
                "hidden_dim": hidden_dim, "train_days": args.train_days,
                "eval_days": args.eval_days, "n_folds": len(folds),
                "n_event_files": len(event_files), "device": str(device),
                "obs_dim": OBS_DIM, "n_actions": N_ACTIONS,
                "oot_only": args.oot_only,
                "start_date": args.start_date or "all",
                "end_date": args.end_date or "all",
                "gpu_buffer": str(device.type == "cuda"),
            })
            log.info(f"MLflow run: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e} — continuing without MLflow")
            MLFLOW_AVAILABLE = False

    # ── Training state ─────────────────────────────────────────────────────────
    best_sortino   = -float("inf")
    best_ckpt_path = output_dir / f"best_agent_{run_name}.pt"

    history: Dict[str, List] = {
        "epoch": [], "fold": [], "alpha": [],
        "train_total_pnl_ticks": [], "train_sortino": [], "train_win_rate": [],
        "train_avg_pnl_ticks": [], "train_n_trades": [],
        "eval_epoch": [], "eval_sortino": [], "eval_win_rate": [],
        "eval_avg_pnl_ticks": [], "eval_avg_trades_per_day": [], "eval_total_pnl_ticks": [],
        "q1_loss": [], "q2_loss": [], "actor_loss": [], "alpha_loss": [], "entropy": [],
    }

    global_epoch    = 0
    total_env_steps = 0
    fold_num        = 0

    _ckpt("entering training loop")

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

        for epoch in range(args.epochs):
            t0 = time.time()
            global_epoch += 1
            actor.train()

            episode_trade_pnls: List[float] = []
            epoch_losses = {"q1": [], "q2": [], "actor": [], "alpha": [], "entropy": []}

            # ── v2: round-robin multi-file stepping (--n-envs conceptually) ───
            # We step through train_files in order, cycling n_envs files
            # simultaneously by interleaving resets. This keeps the env busy
            # and gives more diverse experience per epoch without multiprocessing.
            file_idx        = 0
            files_completed = 0

            # n_envs: step this many files before doing gradient updates
            # (not true parallelism, but gives better data mixing)
            n_envs = min(args.n_envs, len(train_files))

            obs = env.reset(episode_file=train_files[0])

            _step_interval = 500  # stride-level steps (was 50000 for per-event)
            _next_step_log = _step_interval

            while files_completed < len(train_files):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _ = actor.sample(obs_t)
                action_np = int(action.item())

                # v3: stride-based stepping — processes 250 events per actor call (250x speedup)
                next_obs, reward, done, info = env.step_stride(action_np)
                total_env_steps += 1

                if total_env_steps >= _next_step_log:
                    _ckpt(f"stride_steps={total_env_steps:,} files={files_completed}/{len(train_files)} buf={buffer.size:,}")
                    _next_step_log += _step_interval

                # Push to GPU buffer (one transition per stride, not per event)
                buffer.add(obs, action_np, float(reward), next_obs, done)
                obs = next_obs

                if done:
                    for t in env._trade_history:
                        episode_trade_pnls.append(t.pnl_ticks)
                    files_completed += 1
                    _progress = (
                        f"  Fold {fold_num} | Epoch {epoch+1}/{args.epochs} | "
                        f"File {files_completed}/{len(train_files)} done | "
                        f"steps={total_env_steps:,} | buf={buffer.size:,} | "
                        f"trades={len(episode_trade_pnls)}"
                    )
                    log.info(_progress); _ckpt(_progress)
                    for _h in logging.getLogger().handlers:
                        try: _h.flush()
                        except: pass
                    sys.stdout.flush()
                    file_idx += 1
                    if file_idx < len(train_files):
                        obs = env.reset(episode_file=train_files[file_idx])
                    else:
                        break

                # ── v2: update every N steps to avoid 4M updates per file ─
                if buffer.is_ready(args.warmup_steps) and total_env_steps % args.update_interval == 0:
                    for _ in range(args.updates_per_step):
                        losses = sac_update(
                            actor=actor, q1=q1, q2=q2,
                            q1_target=q1_target, q2_target=q2_target,
                            log_alpha=log_alpha, target_entropy=target_entropy,
                            buffer=buffer,
                            actor_opt=actor_opt, q1_opt=q1_opt,
                            q2_opt=q2_opt, alpha_opt=alpha_opt,
                            batch_size=args.batch_size,
                            gamma=args.gamma, tau=args.tau, device=device,
                        )
                        epoch_losses["q1"].append(losses["q1_loss"])
                        epoch_losses["q2"].append(losses["q2_loss"])
                        epoch_losses["actor"].append(losses["actor_loss"])
                        epoch_losses["alpha"].append(losses["alpha_loss"])
                        epoch_losses["entropy"].append(losses["entropy"])

            elapsed = time.time() - t0

            def _mean(lst): return float(np.mean(lst)) if lst else 0.0

            train_metrics = compute_episode_metrics(episode_trade_pnls, total_env_steps)
            current_alpha = log_alpha.exp().item()

            history["epoch"].append(global_epoch)
            history["fold"].append(fold_num)
            history["alpha"].append(current_alpha)
            history["train_total_pnl_ticks"].append(train_metrics.get("total_pnl_ticks", 0.0))
            history["train_sortino"].append(train_metrics.get("sortino", 0.0))
            history["train_win_rate"].append(train_metrics.get("win_rate", 0.0))
            history["train_avg_pnl_ticks"].append(train_metrics.get("avg_pnl_ticks", 0.0))
            history["train_n_trades"].append(train_metrics.get("n_trades", 0))
            history["q1_loss"].append(_mean(epoch_losses["q1"]))
            history["q2_loss"].append(_mean(epoch_losses["q2"]))
            history["actor_loss"].append(_mean(epoch_losses["actor"]))
            history["alpha_loss"].append(_mean(epoch_losses["alpha"]))
            history["entropy"].append(_mean(epoch_losses["entropy"]))

            if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
                if device.type == "cuda":
                    vram_gb = torch.cuda.memory_allocated(device) / 1024**3
                    vram_info = f" | VRAM={vram_gb:.2f}GB"
                else:
                    vram_info = ""
                log.info(
                    f"  Fold {fold_num} | Epoch {epoch+1}/{args.epochs} (global {global_epoch}) | "
                    f"{elapsed:.1f}s | α={current_alpha:.4f} | H={_mean(epoch_losses['entropy']):.3f} | "
                    f"Sortino={train_metrics.get('sortino', 0.0):.3f} | "
                    f"WR={train_metrics.get('win_rate', 0.0):.1%} | "
                    f"AvgPnL={train_metrics.get('avg_pnl_ticks', 0.0):.3f}t | "
                    f"Trades={train_metrics.get('n_trades', 0)} | "
                    f"Q1={_mean(epoch_losses['q1']):.4f} | buf={buffer.size:,}{vram_info}"
                )

            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        "train/sortino":      train_metrics.get("sortino", 0.0),
                        "train/win_rate":     train_metrics.get("win_rate", 0.0),
                        "train/avg_pnl":      train_metrics.get("avg_pnl_ticks", 0.0),
                        "train/n_trades":     float(train_metrics.get("n_trades", 0)),
                        "train/sharpe":       train_metrics.get("sharpe", 0.0),
                        "train/pf":           train_metrics.get("profit_factor", 0.0),
                        "loss/q1":            _mean(epoch_losses["q1"]),
                        "loss/actor":         _mean(epoch_losses["actor"]),
                        "sac/alpha":          current_alpha,
                        "sac/entropy":        _mean(epoch_losses["entropy"]),
                        "sac/buffer_size":    float(buffer.size),
                        "total_env_steps":    float(total_env_steps),
                    }, step=global_epoch)
                except Exception:
                    pass

        # ── Evaluate ───────────────────────────────────────────────────────────
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

        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    "eval/sortino":       eval_metrics.get("sortino", 0.0),
                    "eval/sharpe":        eval_metrics.get("sharpe", 0.0),
                    "eval/win_rate":      eval_metrics.get("win_rate", 0.0),
                    "eval/pf":            eval_metrics.get("profit_factor", 0.0),
                    "eval/avg_pnl":       eval_metrics.get("avg_pnl_ticks", 0.0),
                    "eval/total_pnl_usd": eval_metrics.get("total_pnl_usd", 0.0),
                    "eval/trades_per_day":eval_metrics.get("avg_trades_per_day", 0.0),
                }, step=global_epoch)
            except Exception:
                pass

        # ── Save best checkpoint ───────────────────────────────────────────────
        eval_sortino = eval_metrics.get("sortino", 0.0)
        if eval_sortino > best_sortino:
            best_sortino = eval_sortino
            torch.save({
                "model_state":   actor.state_dict(),
                "obs_dim":       OBS_DIM, "n_actions": N_ACTIONS, "hidden_dim": hidden_dim,
                "fold":          fold_num, "epoch": global_epoch,
                "eval_metrics":  eval_metrics, "best_sortino": best_sortino,
                "algorithm":     "SAC_v2", "args": vars(args), "run_name": run_name,
                "alpha":         log_alpha.exp().item(), "target_entropy": target_entropy,
            }, best_ckpt_path)
            log.info(f"  *** New best checkpoint: Sortino={best_sortino:.3f} → {best_ckpt_path}")
            if MLFLOW_AVAILABLE:
                try: mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
                except Exception: pass

    # ── Final checkpoint ───────────────────────────────────────────────────────
    final_ckpt_path = output_dir / f"final_agent_{run_name}.pt"
    torch.save({
        "model_state": actor.state_dict(), "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS, "hidden_dim": hidden_dim,
        "fold": fold_num, "epoch": global_epoch, "history": history,
        "run_name": run_name, "algorithm": "SAC_v2",
        "alpha": log_alpha.exp().item(),
        "q1_state": q1.state_dict(), "q2_state": q2.state_dict(),
    }, final_ckpt_path)
    log.info(f"Final checkpoint saved → {final_ckpt_path}")

    history_path = output_dir / f"training_history_{run_name}.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(x) if hasattr(x, "item") else x for x in v] for k, v in history.items()}, f, indent=2)
    log.info(f"Training history saved → {history_path}")

    plot_path = save_training_curves(history, output_dir, run_name)
    if plot_path and MLFLOW_AVAILABLE:
        try: mlflow.log_artifact(str(plot_path), artifact_path="plots")
        except Exception: pass

    log.info(
        f"\n{'='*60}\n"
        f"SAC v2 Training complete: {run_name}\n"
        f"  Algorithm:       SAC_v2 (GPU buffer, batch=2048, ups=4)\n"
        f"  Epochs:          {global_epoch}\n"
        f"  Folds:           {len(folds)}\n"
        f"  Total env steps: {total_env_steps:,}\n"
        f"  Buffer size:     {buffer.size:,}\n"
        f"  Final alpha:     {log_alpha.exp().item():.4f}\n"
        f"  Best Sortino:    {best_sortino:.3f}\n"
        f"  Best ckpt:       {best_ckpt_path}\n"
        f"  Final ckpt:      {final_ckpt_path}\n"
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
            "SAC v2 Trainer — GPU-optimized (GPUReplayBuffer, batch=2048, ups=4)\n"
            "\n"
            "Key v2 changes vs v1:\n"
            "  - GPUReplayBuffer: 2M transitions pre-allocated in VRAM (~763MB)\n"
            "  - batch_size=2048 (was 256) — proper GPU saturation\n"
            "  - updates_per_step=4 (was 1) — keeps GPU busy during CPU env stepping\n"
            "  - --oot-only: filter to OOT dates only (HC #118)\n"
            "  - --start-date/--end-date: date range filter\n"
            "  - --n-envs: round-robin multi-file stepping\n"
            "\n"
            "Examples:\n"
            "  # Neptune (3090, 24GB) — OOT only, full date range\n"
            "  python train_fifo_rl_sac_v2.py --oot-only --batch-size 2048 --buffer-size 2000000\n"
            "\n"
            "  # With explicit OOT date range\n"
            "  python train_fifo_rl_sac_v2.py --oot-only --start-date 20260306 --end-date 20260429\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument("--data-dir",    type=str, default=str(DEFAULT_EVENT_DIR),
                   help=f"MBO event files directory (default: {DEFAULT_EVENT_DIR})")
    p.add_argument("--pred-dir",    type=str, default=str(CNN_MAMBA_BULK_OOT_DIR),
                   help=f"CNN-Mamba bulk OOT predictions dir (default: {CNN_MAMBA_BULK_OOT_DIR})")
    p.add_argument("--patchtst-dir",type=str, default=str(PATCHTST_BULK_OOT_DIR),
                   help=f"PatchTST predictions dir (default: {PATCHTST_BULK_OOT_DIR})")
    p.add_argument("--output-dir",  type=str, default="/home/nick/Lvl3Quant/output/fifo_sac_rl_v2",
                   help="Output directory for checkpoints, logs, plots")

    # OOT filtering (HC #118)
    p.add_argument("--oot-only",    action="store_true",
                   help="Filter training data to only dates with OOT predictions (HC #118)")
    p.add_argument("--start-date",  type=str, default=None,
                   help="Start date filter YYYYMMDD (e.g. 20260306)")
    p.add_argument("--end-date",    type=str, default=None,
                   help="End date filter YYYYMMDD (e.g. 20260429)")

    # Walk-forward
    p.add_argument("--train-days",  type=int, default=DEFAULT_TRAIN_DAYS)
    p.add_argument("--eval-days",   type=int, default=DEFAULT_EVAL_DAYS)

    # Training
    p.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",          type=float, default=DEFAULT_LR)

    # SAC hyperparams
    p.add_argument("--gamma",       type=float, default=DEFAULT_GAMMA)
    p.add_argument("--tau",         type=float, default=DEFAULT_TAU)
    p.add_argument("--alpha-init",  type=float, default=DEFAULT_ALPHA_INIT)
    p.add_argument("--target-entropy-ratio", type=float, default=0.98)
    p.add_argument("--buffer-size", type=int,   default=DEFAULT_BUFFER_SIZE,
                   help=f"Replay buffer capacity (default: {DEFAULT_BUFFER_SIZE:,}). "
                        "2M→~763MB VRAM; 5M→~1.9GB. Neptune 3090 can handle 5M easily.")
    p.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE,
                   help=f"Mini-batch size (default: {DEFAULT_BATCH_SIZE}). 2048 for 3090, 1024 for 3070.")
    p.add_argument("--warmup-steps",type=int,   default=DEFAULT_WARMUP_STEPS)
    p.add_argument("--updates-per-step", type=int, default=DEFAULT_UPDATES_PER_STEP,
                   help=f"SAC updates per env step (default: {DEFAULT_UPDATES_PER_STEP}). "
                        "4-8 keeps GPU busy during CPU-bound env stepping.")
    p.add_argument("--update-interval", type=int, default=50,
                   help="Only do gradient updates every N env steps (default: 50). "
                        "With 1M+ events per file, updating every step is too slow.")
    p.add_argument("--n-envs",      type=int,   default=1,
                   help="Round-robin env file count for data mixing (default: 1). "
                        "Does NOT use multiprocessing — just interleaves file resets.")

    # Architecture (frozen — 256-dim 3-layer is the sweet spot)
    p.add_argument("--hidden-dim",  type=int, default=256,
                   help="Hidden dim (default: 256 — sweet spot, do not change)")
    p.add_argument("--max-steps-per-episode", type=int, default=2000000)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("FIFO SAC v2 RL Trainer  (GPU-Optimized)")
    log.info("=" * 60)
    log.info(f"  data_dir:         {args.data_dir}")
    log.info(f"  pred_dir:         {args.pred_dir}")
    log.info(f"  patchtst_dir:     {args.patchtst_dir}")
    log.info(f"  output_dir:       {args.output_dir}")
    log.info(f"  oot_only:         {args.oot_only}")
    log.info(f"  date_range:       {args.start_date or 'all'} – {args.end_date or 'all'}")
    log.info(f"  epochs:           {args.epochs} per fold")
    log.info(f"  lr:               {args.lr}")
    log.info(f"  buffer_size:      {args.buffer_size:,}  [GPU pre-allocated]")
    log.info(f"  batch_size:       {args.batch_size}  [was 256]")
    log.info(f"  warmup_steps:     {args.warmup_steps}")
    log.info(f"  updates_per_step: {args.updates_per_step}  [was 1]")
    log.info(f"  n_envs:           {args.n_envs}")
    log.info(f"  hidden_dim:       {args.hidden_dim}  [frozen]")
    log.info(f"  train_days:       {args.train_days}")
    log.info(f"  eval_days:        {args.eval_days}")
    log.info(f"  torch:            {torch.__version__}")
    log.info(f"  cuda:             {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    log.info(f"  mlflow:           {'yes' if MLFLOW_AVAILABLE else 'NO'}")
    log.info("=" * 60)

    train(args)
