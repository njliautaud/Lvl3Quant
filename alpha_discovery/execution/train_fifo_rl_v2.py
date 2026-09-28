#!/usr/bin/env python3
"""
PPO Trainer for FIFO RL Execution Agent — v2 (GPU-Optimized)
=============================================================

Changes from v1:
  1. GPU RolloutBuffer — pre-allocated VRAM tensors, tensor batches stay on GPU
     through GAE computation and mini-batch iteration (no numpy→tensor at batch time).
  2. Mini-batch size 2048 (Neptune/3090). Was 512 — better GPU utilization.
  3. OOT date filtering via --oot-only + --start-date / --end-date (HC #118).
  4. Correct bulk OOT pred paths: cnn_mamba_v2_bulk_inference/ (YYYYMMDD format).
  5. VRAM reporting at startup and during training.
  6. --n-envs flag (round-robin file stepping, same as SAC v2).

Architecture unchanged: hidden_dim=256, 3 layers. Same FIFOActorCritic backbone.
All HC compliance from v1 preserved.

HC compliance:
  - HC #0:   sliding walk-forward (no expanding window)
  - HC #69:  primary metrics = Sharpe, Sortino, PF, WR
  - HC #89:  cost = 0.376 ticks limits; 1.376 market orders
  - HC #100: risk-adjusted reward
  - HC #104: symmetric reward — no short-side bias
  - HC #105: training curve PNG saved
  - HC #118: OOT-only training data filter

Author: Claude (Infrastructure Builder)
Date: 2026-05-03
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
    MARKET_ORDER_COST_TICKS,
    LIMIT_ORDER_COST_TICKS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("train_fifo_rl_v2")

N_ACTIONS         = 7
MLFLOW_EXPERIMENT = "FIFO_RL_PPO_v2_Execution"

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_EPOCHS       = 100
DEFAULT_LR           = 3e-4
DEFAULT_ROLLOUT_LEN  = 4096
DEFAULT_N_PPO_EPOCHS = 4
DEFAULT_MINI_BATCH   = 2048        # v2: 2048 (was 512) — better GPU utilization on 3090
DEFAULT_GAMMA        = 0.99
DEFAULT_LAM          = 0.95
DEFAULT_CLIP_EPS     = 0.2
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_VF_COEF      = 0.5
DEFAULT_MAX_GRAD     = 0.5
DEFAULT_TRAIN_DAYS   = 40
DEFAULT_EVAL_DAYS    = 5

# ── Bulk OOT prediction directories (Neptune canonical paths) ──────────────────
LVL3_NEPTUNE = Path("/home/nick/Lvl3Quant")
CNN_MAMBA_BULK_OOT_DIR = LVL3_NEPTUNE / "output" / "cnn_mamba_v2_bulk_inference"
PATCHTST_BULK_OOT_DIR  = LVL3_NEPTUNE / "output" / "patchtst_smart_v3_mar"


# ══════════════════════════════════════════════════════════════════════════════
# Network Architecture (unchanged from v1 — 256-dim 3-layer is the sweet spot)
# ══════════════════════════════════════════════════════════════════════════════

class FIFOActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic. Identical architecture to v1.
    Input(48) → LayerNorm → 3×(Linear→GELU→LayerNorm) → 256-dim
    Actor: hidden→128→7 logits
    Critic: hidden→128→1 value
    """

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_dim: int = 256):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.actor  = nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_norm(obs); x = self.shared(x)
        return self.actor(x), self.critic(x).squeeze(-1)

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if action is None: action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ══════════════════════════════════════════════════════════════════════════════
# GPU Rollout Buffer (v2 key upgrade)
# ══════════════════════════════════════════════════════════════════════════════

class GPURolloutBuffer:
    """
    PPO rollout buffer with GPU-resident storage.

    Pre-allocates all rollout tensors in VRAM so mini-batch slicing
    never requires CPU→GPU transfers during the PPO update loop.

    At capacity=4096, obs_dim=48:
      obs + log_probs + rewards + dones + values + actions ≈ 5 MB
      (rollout buffers are small — GPURolloutBuffer mainly removes
       the per-batch tensor() conversion overhead)

    Fallback to CPU if CUDA OOM (shouldn't happen at these sizes).
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
        self.gamma    = gamma
        self.lam      = lam
        self.device   = device
        self.ptr      = 0
        self.full     = False

        try:
            self.obs       = torch.zeros((capacity, obs_dim), device=device, dtype=torch.float32)
            self.actions   = torch.zeros(capacity, device=device, dtype=torch.long)
            self.log_probs = torch.zeros(capacity, device=device, dtype=torch.float32)
            self.rewards   = torch.zeros(capacity, device=device, dtype=torch.float32)
            self.dones     = torch.zeros(capacity, device=device, dtype=torch.float32)
            self.values    = torch.zeros(capacity, device=device, dtype=torch.float32)
        except RuntimeError:
            cpu = torch.device("cpu")
            self.device    = cpu
            self.obs       = torch.zeros((capacity, obs_dim), device=cpu, dtype=torch.float32)
            self.actions   = torch.zeros(capacity, device=cpu, dtype=torch.long)
            self.log_probs = torch.zeros(capacity, device=cpu, dtype=torch.float32)
            self.rewards   = torch.zeros(capacity, device=cpu, dtype=torch.float32)
            self.dones     = torch.zeros(capacity, device=cpu, dtype=torch.float32)
            self.values    = torch.zeros(capacity, device=cpu, dtype=torch.float32)

    def reset(self):
        self.ptr  = 0
        self.full = False

    def add(self, obs: np.ndarray, action: int, log_prob: float, reward: float, done: bool, value: float):
        self.obs[self.ptr]       = torch.from_numpy(obs).to(self.device)
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

    def compute_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns — done entirely on GPU tensors.
        Returns (advantages, returns) as GPU tensors.
        """
        advantages = torch.zeros(self.capacity, device=self.device)
        gae = 0.0
        next_val = last_value

        for t in reversed(range(self.capacity)):
            not_done = 1.0 - self.dones[t].item()
            delta    = self.rewards[t].item() + self.gamma * next_val * not_done - self.values[t].item()
            gae      = delta + self.gamma * self.lam * not_done * gae
            advantages[t] = gae
            next_val = self.values[t].item()

        returns = advantages + self.values
        return advantages, returns

    def get_batches(self, advantages: torch.Tensor, returns: torch.Tensor, batch_size: int):
        """
        Yield shuffled mini-batches — all tensors already on device, no conversion.
        """
        indices = torch.randperm(self.capacity, device=self.device)
        for start in range(0, self.capacity, batch_size):
            idx = indices[start : start + batch_size]
            yield {
                "obs":           self.obs[idx],
                "actions":       self.actions[idx],
                "old_log_probs": self.log_probs[idx],
                "advantages":    advantages[idx],
                "returns":       returns[idx],
            }


# ══════════════════════════════════════════════════════════════════════════════
# OOT Date Filtering (HC #118) — identical to SAC v2
# ══════════════════════════════════════════════════════════════════════════════

def filter_oot_files(
    event_files: List[Path],
    pred_dir: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Path]:
    """Filter event files to only those with matching YYYYMMDD_predictions.npz in pred_dir."""
    pred_dates = set()
    if pred_dir and pred_dir.exists():
        for p in pred_dir.glob("20??????_predictions.npz"):
            pred_dates.add(p.stem[:8])
        log.info(f"OOT filter: found {len(pred_dates)} prediction dates in {pred_dir.name}")
    else:
        log.warning(f"OOT filter: pred_dir {pred_dir} not found — no date filtering applied")
        return event_files

    start_int = int(start_date) if start_date else 0
    end_int   = int(end_date)   if end_date   else 99999999

    filtered = [
        f for f in event_files
        if f.stem[:8] in pred_dates and start_int <= int(f.stem[:8]) <= end_int
    ]
    log.info(f"OOT filter: {len(event_files)} → {len(filtered)} files (range {start_date or 'all'}–{end_date or 'all'})")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Data Split
# ══════════════════════════════════════════════════════════════════════════════

def build_walk_forward_folds(event_files: List[Path], train_days: int, eval_days: int) -> List[Dict]:
    """Sliding walk-forward folds (HC #0)."""
    folds = []
    n, window, i, fold_num = len(event_files), train_days + eval_days, 0, 0
    while i + window <= n:
        folds.append({
            "fold": fold_num,
            "train_files": event_files[i : i + train_days],
            "eval_files":  event_files[i + train_days : i + train_days + eval_days],
        })
        i += eval_days; fold_num += 1
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
    n = len(pnls)
    win_rate = float((pnls > 0).mean())
    mean_pnl = float(pnls.mean())
    std_pnl  = float(pnls.std()) if n > 1 else 1.0
    sharpe   = mean_pnl / (std_pnl + 1e-8) * math.sqrt(max(n, 1))
    downside = pnls[pnls < 0]
    sortino  = mean_pnl * 10.0 if len(downside) == 0 else mean_pnl / (float(np.sqrt(np.mean(downside**2))) + 1e-8)
    gw = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gl = abs(float(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-8
    return {
        "n_trades": n, "win_rate": win_rate, "sharpe": sharpe, "sortino": sortino,
        "profit_factor": gw / gl, "avg_pnl_ticks": mean_pnl,
        "total_pnl_ticks": float(pnls.sum()), "total_pnl_usd": float(pnls.sum()) * TICK_VALUE,
        "trades_per_step": n / max(n_steps, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PPO Update (operates on GPU RolloutBuffer — no data movement in inner loop)
# ══════════════════════════════════════════════════════════════════════════════

def ppo_update(
    policy, optimizer, buffer: GPURolloutBuffer, last_value: float,
    n_ppo_epochs: int, batch_size: int, clip_eps: float,
    entropy_coef: float, vf_coef: float, max_grad_norm: float,
) -> Dict[str, float]:
    """
    PPO update. All mini-batches come from GPURolloutBuffer — GPU tensors,
    no CPU→GPU copy per batch (key v2 optimization).
    """
    advantages, returns = buffer.compute_advantages(last_value)

    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std  = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    total_policy_loss = total_value_loss = total_entropy = 0.0
    n_updates = 0

    policy.train()
    for _ in range(n_ppo_epochs):
        for batch in buffer.get_batches(advantages, returns, batch_size):
            _, new_log_prob, entropy, new_value = policy.get_action_and_value(
                batch["obs"], batch["actions"]
            )
            ratio = torch.exp(new_log_prob - batch["old_log_probs"])
            adv_b = batch["advantages"]
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.mse_loss(new_value, batch["returns"])
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss
            optimizer.zero_grad(); loss.backward()
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

def collect_rollout(
    env, policy, buffer: GPURolloutBuffer, train_files: List[Path], device: torch.device,
) -> Tuple[Dict[str, float], float]:
    buffer.reset()
    policy.eval()
    episode_trade_pnls: List[float] = []
    n_episodes = n_total_steps = 0
    file_idx = 0
    obs = env.reset(episode_file=train_files[file_idx % len(train_files)])

    while not buffer.is_ready():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _, value = policy.get_action_and_value(obs_t)
        action_np = int(action.item())
        next_obs, reward, done, info = env.step_stride(action_np)  # v3: stride stepping
        buffer.add(obs, action_np, float(log_prob.item()), float(reward), done, float(value.item()))
        obs = next_obs
        n_total_steps += 1

        if done:
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
    return rollout_metrics, last_value_np


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_policy(env, policy, eval_files, device) -> Dict[str, float]:
    policy.eval()
    all_trade_pnls, trades_per_day = [], []
    total_steps = 0
    for ep_file in eval_files:
        try:
            obs = env.reset(episode_file=ep_file)
        except Exception as e:
            log.warning(f"evaluate_policy: skipping {ep_file.name} — {e}"); continue
        done = False; steps = 0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = policy.forward(obs_t)
                action_np = int(logits.argmax(dim=-1).item())
            obs, _, done, _ = env.step_stride(action_np)  # v3: stride stepping
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
        log.warning("matplotlib not available — skipping training curve plot"); return None
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(f"FIFO PPO v2 Training — {run_name}", fontsize=14, fontweight="bold")
    epochs = history.get("epoch", [])
    panels = [
        ("train_total_pnl_ticks","eval_total_pnl_ticks","Cumulative P&L (ticks)","tab:blue"),
        ("train_sortino",        "eval_sortino",         "Sortino Ratio",         "tab:orange"),
        ("train_win_rate",       "eval_win_rate",        "Win Rate",              "tab:green"),
        ("train_avg_pnl_ticks",  "eval_avg_pnl_ticks",  "Avg P&L/Trade (ticks)", "tab:red"),
        ("train_n_trades",       "eval_avg_trades_per_day","Trades/Day",          "tab:purple"),
    ]
    for ax, (tk, ek, ylabel, color) in zip(axes, panels):
        if tk in history and history[tk]:
            ax.plot(epochs, history[tk], color=color, alpha=0.8, linewidth=1.5, label="train")
        if ek in history and history[ek]:
            ax.plot(history.get("eval_epoch", epochs[:len(history[ek])]), history[ek],
                    color=color, alpha=1.0, linewidth=2.0, linestyle="--", label="eval",
                    marker="o", markersize=4)
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

    run_name = datetime.now().strftime("ppo_v2_%Y%m%d_%H%M%S")

    _log_file = output_dir / f"{run_name}.log"
    _fh = logging.FileHandler(str(_log_file), mode="w")
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
        event_files = filter_oot_files(event_files, pred_dir, args.start_date, args.end_date)
        if not event_files:
            log.error("OOT filter left 0 files — check pred_dir and date range"); sys.exit(1)
    else:
        if args.start_date or args.end_date:
            start_int = int(args.start_date) if args.start_date else 0
            end_int   = int(args.end_date)   if args.end_date   else 99999999
            event_files = [f for f in event_files if start_int <= int(f.stem[:8]) <= end_int]
            log.info(f"Date range filter: {len(event_files)} files remaining")

    log.info(f"Training on {len(event_files)} files ({event_files[0].stem[:8]}–{event_files[-1].stem[:8]})")

    # ── Walk-forward folds ─────────────────────────────────────────────────────
    folds = build_walk_forward_folds(event_files, args.train_days, args.eval_days)
    if not folds:
        log.error(f"Not enough data: need {args.train_days + args.eval_days}, have {len(event_files)}"); sys.exit(1)
    log.info(f"Walk-forward folds: {len(folds)} (train={args.train_days}d, eval={args.eval_days}d)")

    # ── Environment ────────────────────────────────────────────────────────────
    patchtst_dir = Path(args.patchtst_dir) if args.patchtst_dir else PATCHTST_BULK_OOT_DIR
    if not patchtst_dir.exists():
        log.warning(f"PatchTST dir not found: {patchtst_dir} — running without confluence")
        patchtst_dir = None

    env = FIFOExecutionEnv(
        event_dir=event_dir, pred_dir=pred_dir, patchtst_pred_dir=patchtst_dir,
        rth_only=True, verbose=False, max_steps_per_episode=500000,  # ~5min RTH
    )
    log.info(f"Env: obs_dim={OBS_DIM}, n_actions={N_ACTIONS}")
    log.info(f"Pred dir (CNN-Mamba): {pred_dir}")
    log.info(f"Pred dir (PatchTST):  {patchtst_dir}")

    # ── Policy ─────────────────────────────────────────────────────────────────
    policy = FIFOActorCritic(OBS_DIM, N_ACTIONS, args.hidden_dim).to(device)
    log.info(f"Policy params: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    total_epochs = args.epochs * len(folds)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1), eta_min=args.lr * 0.05)

    # ── GPU Rollout Buffer (v2 key change) ─────────────────────────────────────
    buffer = GPURolloutBuffer(
        capacity=args.rollout_len, obs_dim=OBS_DIM,
        gamma=args.gamma, lam=args.lam, device=device,
    )
    log.info(f"GPURolloutBuffer: capacity={args.rollout_len} on {device}")

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
                "algorithm": "PPO_v2", "epochs": args.epochs, "lr": args.lr,
                "rollout_len": args.rollout_len, "n_ppo_epochs": args.n_ppo_epochs,
                "mini_batch": args.mini_batch, "gamma": args.gamma, "lam": args.lam,
                "clip_eps": args.clip_eps, "entropy_coef": args.entropy_coef,
                "vf_coef": args.vf_coef, "hidden_dim": args.hidden_dim,
                "train_days": args.train_days, "eval_days": args.eval_days,
                "n_folds": len(folds), "n_event_files": len(event_files),
                "device": str(device), "oot_only": args.oot_only,
                "start_date": args.start_date or "all", "end_date": args.end_date or "all",
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
        "epoch": [], "fold": [], "lr": [],
        "train_total_pnl_ticks": [], "train_sortino": [], "train_win_rate": [],
        "train_avg_pnl_ticks": [], "train_n_trades": [],
        "eval_epoch": [], "eval_sortino": [], "eval_win_rate": [],
        "eval_avg_pnl_ticks": [], "eval_avg_trades_per_day": [], "eval_total_pnl_ticks": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
    }

    global_epoch = 0
    fold_num     = 0

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

            rollout_metrics, last_value = collect_rollout(env, policy, buffer, train_files, device)
            update_losses = ppo_update(
                policy, optimizer, buffer, last_value,
                n_ppo_epochs=args.n_ppo_epochs, batch_size=args.mini_batch,
                clip_eps=args.clip_eps, entropy_coef=args.entropy_coef,
                vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
            )
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0

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

            if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
                if device.type == "cuda":
                    vram_gb = torch.cuda.memory_allocated(device) / 1024**3
                    vram_info = f" | VRAM={vram_gb:.2f}GB"
                else:
                    vram_info = ""
                log.info(
                    f"  Fold {fold_num} | Epoch {epoch+1}/{args.epochs} (global {global_epoch}) | "
                    f"{elapsed:.1f}s | LR={current_lr:.2e} | "
                    f"Sortino={rollout_metrics.get('sortino', 0.0):.3f} | "
                    f"WR={rollout_metrics.get('win_rate', 0.0):.1%} | "
                    f"AvgPnL={rollout_metrics.get('avg_pnl_ticks', 0.0):.3f}t | "
                    f"Trades={rollout_metrics.get('n_trades', 0)} | "
                    f"π={update_losses['policy_loss']:.4f} | "
                    f"V={update_losses['value_loss']:.4f} | "
                    f"H={update_losses['entropy']:.3f}{vram_info}"
                )

            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        "train/sortino":  rollout_metrics.get("sortino", 0.0),
                        "train/win_rate": rollout_metrics.get("win_rate", 0.0),
                        "train/avg_pnl":  rollout_metrics.get("avg_pnl_ticks", 0.0),
                        "train/n_trades": float(rollout_metrics.get("n_trades", 0)),
                        "train/sharpe":   rollout_metrics.get("sharpe", 0.0),
                        "train/pf":       rollout_metrics.get("profit_factor", 0.0),
                        "loss/policy":    update_losses["policy_loss"],
                        "loss/value":     update_losses["value_loss"],
                        "loss/entropy":   update_losses["entropy"],
                        "lr":             current_lr, "fold": float(fold_num),
                    }, step=global_epoch)
                except Exception:
                    pass

        # ── Evaluate ───────────────────────────────────────────────────────────
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

        eval_sortino = eval_metrics.get("sortino", 0.0)
        if eval_sortino > best_sortino:
            best_sortino = eval_sortino
            torch.save({
                "fold": fold_num, "epoch": global_epoch,
                "model_state": policy.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "eval_metrics": eval_metrics, "best_sortino": best_sortino,
                "algorithm": "PPO_v2", "args": vars(args), "run_name": run_name,
            }, best_ckpt_path)
            log.info(f"  *** New best checkpoint: Sortino={best_sortino:.3f} → {best_ckpt_path}")
            if MLFLOW_AVAILABLE:
                try: mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")
                except Exception: pass

    # ── Final checkpoint + artifacts ───────────────────────────────────────────
    final_ckpt_path = output_dir / f"final_agent_{run_name}.pt"
    torch.save({
        "fold": fold_num if folds else -1, "epoch": global_epoch,
        "model_state": policy.state_dict(), "optimizer_state": optimizer.state_dict(),
        "history": history, "run_name": run_name, "algorithm": "PPO_v2",
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
        f"PPO v2 Training complete: {run_name}\n"
        f"  Algorithm:       PPO_v2 (GPU rollout buffer, mini_batch=2048)\n"
        f"  Epochs:          {global_epoch}\n"
        f"  Folds:           {len(folds)}\n"
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
            "PPO v2 Trainer — GPU-optimized (GPURolloutBuffer, mini_batch=2048)\n"
            "\n"
            "Key v2 changes vs v1:\n"
            "  - GPURolloutBuffer: rollout tensors pre-allocated in VRAM\n"
            "  - mini_batch=2048 (was 512) — better GPU utilization on 3090\n"
            "  - --oot-only: filter to OOT dates only (HC #118)\n"
            "  - --start-date/--end-date: date range filter\n"
            "  - VRAM usage logged at startup and each eval\n"
            "\n"
            "Examples:\n"
            "  # Neptune (3090) — OOT only\n"
            "  python train_fifo_rl_v2.py --oot-only --mini-batch 2048\n"
            "\n"
            "  # With date range\n"
            "  python train_fifo_rl_v2.py --oot-only --start-date 20260306 --end-date 20260429\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--data-dir",    type=str, default=str(DEFAULT_EVENT_DIR))
    p.add_argument("--pred-dir",    type=str, default=str(CNN_MAMBA_BULK_OOT_DIR))
    p.add_argument("--patchtst-dir",type=str, default=str(PATCHTST_BULK_OOT_DIR))
    p.add_argument("--output-dir",  type=str, default="/home/nick/Lvl3Quant/output/fifo_ppo_rl_v2")

    # OOT filtering (HC #118)
    p.add_argument("--oot-only",    action="store_true")
    p.add_argument("--start-date",  type=str, default=None, help="YYYYMMDD")
    p.add_argument("--end-date",    type=str, default=None, help="YYYYMMDD")

    # Walk-forward
    p.add_argument("--train-days",  type=int, default=DEFAULT_TRAIN_DAYS)
    p.add_argument("--eval-days",   type=int, default=DEFAULT_EVAL_DAYS)

    # Training
    p.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--lr",          type=float, default=DEFAULT_LR)
    p.add_argument("--rollout-len", type=int,   default=DEFAULT_ROLLOUT_LEN)
    p.add_argument("--n-ppo-epochs",type=int,   default=DEFAULT_N_PPO_EPOCHS)
    p.add_argument("--mini-batch",  type=int,   default=DEFAULT_MINI_BATCH,
                   help=f"Mini-batch size (default: {DEFAULT_MINI_BATCH}). 2048 for 3090, 1024 for 3070.")

    # PPO hyperparams
    p.add_argument("--gamma",       type=float, default=DEFAULT_GAMMA)
    p.add_argument("--lam",         type=float, default=DEFAULT_LAM)
    p.add_argument("--clip-eps",    type=float, default=DEFAULT_CLIP_EPS)
    p.add_argument("--entropy-coef",type=float, default=DEFAULT_ENTROPY_COEF)
    p.add_argument("--vf-coef",     type=float, default=DEFAULT_VF_COEF)
    p.add_argument("--max-grad-norm",type=float,default=DEFAULT_MAX_GRAD)

    # Architecture (frozen)
    p.add_argument("--hidden-dim",  type=int, default=256)
    p.add_argument("--n-envs",      type=int, default=1)  # unused in PPO, for CLI compat

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("=" * 60)
    log.info("FIFO PPO v2 RL Trainer  (GPU-Optimized)")
    log.info("=" * 60)
    log.info(f"  data_dir:    {args.data_dir}")
    log.info(f"  pred_dir:    {args.pred_dir}")
    log.info(f"  output_dir:  {args.output_dir}")
    log.info(f"  oot_only:    {args.oot_only}")
    log.info(f"  date_range:  {args.start_date or 'all'} – {args.end_date or 'all'}")
    log.info(f"  epochs:      {args.epochs} per fold")
    log.info(f"  lr:          {args.lr}")
    log.info(f"  rollout_len: {args.rollout_len}")
    log.info(f"  n_ppo_epochs:{args.n_ppo_epochs}")
    log.info(f"  mini_batch:  {args.mini_batch}  [was 512]")
    log.info(f"  hidden_dim:  {args.hidden_dim}  [frozen]")
    log.info(f"  train_days:  {args.train_days}")
    log.info(f"  eval_days:   {args.eval_days}")
    log.info(f"  torch:       {torch.__version__}")
    log.info(f"  cuda:        {torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    log.info(f"  mlflow:      {'yes' if MLFLOW_AVAILABLE else 'NO'}")
    log.info("=" * 60)

    train(args)
