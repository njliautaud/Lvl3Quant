#!/usr/bin/env python3
"""
Live RL Inference Pipeline for FIFO Execution Agent
=====================================================

Loads a trained SAC (or PPO) RL checkpoint and runs inference in real-time,
consuming live CNN-Mamba v2 + PatchTST predictions + book state observations
and outputting discrete execution actions.

This is the bridge between:
  - Razer live host (MBO recorder + CNN-Mamba + PatchTST inference → predictions)
  - Jupiter/Neptune paper trading execution engine

Checkpoint compatibility:
  - SAC checkpoint (train_fifo_rl_sac.py):  {"model_state": actor.state_dict(), ...}
  - PPO checkpoint (rl_execution_agent.py): {"policy_state_dict": policy.state_dict(), ...}
  Both are auto-detected from the checkpoint dict keys.

Observation format (48 dims — OBS_DIM from fifo_rl_env.py):
  [0:4]   Signal: pred_1s, pred_5s, pred_10s, confidence_tier (0-3 → normalized)
  [4:8]   Book: bid_depth_log, ask_depth_log, book_imbalance, spread_ticks
  [8:10]  Price: price_rel_ticks, price_momentum_10
  [10:14] Position: position (0/1/-1), unrealized_pnl, time_in_pos_s, queue_frac_filled
  [14:16] MFE/MAE: max_fav_excursion, max_adv_excursion
  [16:19] Context: realized_vol_60s, tod_sin, tod_cos
  [19:24] Flow: event_density_10s, buy_vol_frac_10s, flow_imbalance, cancel_rate, event_rate
  [24:29] Trade history: last 5 trade PnLs (normalized)
  [29:32] Stats: rolling_win_rate, rolling_sortino, consecutive_loss_count
  [32:34] Order: pending_order (0/1), pending_order_side (0/1/-1)
  [34:39] CNN-Mamba embedding: top-5 PCA components (if available)
  [39:45] PatchTST confluence: pst_1s, pst_5s, pst_10s, confluence_1s, confluence_all, sweep_intensity
  [45:48] Alpha-awareness: signal_remaining_frac, entry_signal_strength, current_alpha_alignment

Action space (7 discrete):
  0: HOLD (do nothing / wait)
  1: BUY_LIMIT (passive limit at bid)
  2: SELL_LIMIT (passive limit at ask)
  3: BUY_MARKET (aggressive cross spread)
  4: SELL_MARKET (aggressive cross spread)
  5: CANCEL (cancel pending order)
  6: MARKET_EXIT (force-close open position)

HC compliance:
  - Cost: 0.376 ticks (commission only — HC #231(A): no spread crossing cost)
  - FIFO queue model (no midpoint) (HC #74)
  - Metrics: Sortino, Sharpe, PF, WR (HC #69)

Usage:
  # Standard inference (load checkpoint, feed obs, get action):
  from live_rl_inference import RLExecutionAgent
  agent = RLExecutionAgent()
  agent.load_checkpoint("/path/to/best_agent.pt")
  action = agent.get_action(observation_vector)

  # Test mode — runs inference on a recorded MBO file:
  python live_rl_inference.py --checkpoint /path/to/best.pt --test-file 20260501_mbo_events.npz

Author: Claude (Infrastructure Builder)
Date: 2026-05-03
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
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
    print("ERROR: PyTorch not found. Install: pip install torch", file=sys.stderr)
    sys.exit(1)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("live_rl_inference")

# ── Constants (HC canonical — DO NOT CHANGE) ──────────────────────────────────
OBS_DIM = 48               # Must match fifo_rl_env.py OBS_DIM exactly
N_ACTIONS = 7              # Matches fifo_rl_env.py action space
TICK_VALUE = 12.50         # $ per ES tick (HC #89)
COMMISSION_RT_TICKS = 0.376   # $4.70 / $12.50
MARKET_ORDER_COST_TICKS = 0.376   # HC #231(A): commission only — no spread cost
LIMIT_ORDER_COST_TICKS = 0.376    # commission only (passive fill)

# ── Action names (for logging) ─────────────────────────────────────────────────
ACTION_NAMES = {
    0: "HOLD",
    1: "BUY_LIMIT",
    2: "SELL_LIMIT",
    3: "BUY_MARKET",
    4: "SELL_MARKET",
    5: "CANCEL",
    6: "MARKET_EXIT",
}


# ══════════════════════════════════════════════════════════════════════════════
# Network Architectures
# Both are kept here for checkpoint compatibility with existing training scripts.
# ══════════════════════════════════════════════════════════════════════════════

class SACActorNet(nn.Module):
    """
    SAC Actor network for discrete actions.

    Architecture matches train_fifo_rl_sac.py:SACActorNet exactly so that
    checkpoints saved by the SAC trainer load here without modification.

    Input (48) → LayerNorm
              → Linear(48, hidden) → GELU → LayerNorm
              → Linear(hidden, hidden) → GELU → LayerNorm
              → Linear(hidden, hidden) → GELU → LayerNorm  [shared backbone]
              → Linear(hidden, 128) → GELU → Linear(128, 7)  [actor head]
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

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

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (probs, log_probs), each shape (B, N_ACTIONS)."""
        x = self.input_norm(obs)
        x = self.shared(x)
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def get_action_greedy(self, obs: torch.Tensor) -> torch.Tensor:
        """Greedy argmax action — used for deterministic evaluation/live trading."""
        probs, _ = self.forward(obs)
        return probs.argmax(dim=-1)

    def get_action_stochastic(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stochastic sample — used for exploration."""
        probs, log_probs = self.forward(obs)
        dist = Categorical(probs=probs)
        action = dist.sample()
        return action, probs


class PPOActorNet(nn.Module):
    """
    PPO Actor-Critic policy for loading rl_execution_agent.py checkpoints.

    Architecture matches rl_execution_agent.py:ExecPolicyV3.
    NOTE: This uses STATE_DIM=54 — different from the SAC OBS_DIM=48.
    The agent auto-detects which network type to use from checkpoint keys.

    If your checkpoint has "policy_state_dict" it's PPO (54-dim obs).
    If it has "model_state" it's SAC (48-dim obs).
    """

    def __init__(self, state_dim: int = 54, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim

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

        self.policy_head = nn.Linear(64, N_ACTIONS)
        self.temperature = nn.Parameter(torch.ones(1))

        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, value)."""
        features = self.backbone(state)
        temp = torch.clamp(self.temperature, 0.1, 5.0)
        logits = self.policy_head(features) / temp
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action_greedy(self, state: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(state)
        return logits.argmax(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint Detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_checkpoint_type(ckpt: dict) -> str:
    """
    Auto-detect checkpoint type from dict keys.

    Returns:
      "sac"  — trained by train_fifo_rl_sac.py (key: "model_state")
      "ppo"  — trained by rl_execution_agent.py (key: "policy_state_dict")
      "unknown" — unrecognized format
    """
    if "model_state" in ckpt:
        return "sac"
    elif "policy_state_dict" in ckpt:
        return "ppo"
    else:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Main Agent Class
# ══════════════════════════════════════════════════════════════════════════════

class RLExecutionAgent:
    """
    Live inference wrapper for SAC/PPO RL execution agents.

    This is the primary interface for the paper trading system.
    It handles:
      - Checkpoint loading (auto-detects SAC vs PPO format)
      - Observation validation and NaN-checking
      - Deterministic (greedy) action selection for live trading
      - Optional stochastic mode for exploration
      - Action masking (e.g., disallow BUY when already long)
      - Inference latency tracking
      - Trade state tracking for obs construction helpers

    Usage:
      agent = RLExecutionAgent(device="cpu")
      agent.load_checkpoint("/path/to/best_agent.pt")
      agent.reset()  # call at start of each trading day

      # Each time a new observation is available (e.g. every pred_stride events):
      obs = build_observation(...)  # 48-dim numpy array
      action = agent.get_action(obs)
      # action is int in {0..6} → ACTION_NAMES[action]
    """

    def __init__(
        self,
        device: Optional[str] = None,
        deterministic: bool = True,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        device : "cuda", "cpu", or None (auto-detect)
        deterministic : if True, use argmax action (greedy). If False, sample.
        verbose : if True, log every action decision
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.deterministic = deterministic
        self.verbose = verbose

        self._net: Optional[nn.Module] = None
        self._checkpoint_path: Optional[str] = None
        self._checkpoint_type: str = "unknown"
        self._obs_dim: int = OBS_DIM
        self._n_actions: int = N_ACTIONS
        self._hidden_dim: int = 256
        self._loaded: bool = False

        # Runtime state (for action masking and decision context)
        self._position: int = 0        # 0=flat, +1=long, -1=short
        self._has_pending_order: bool = False
        self._consecutive_losses: int = 0
        self._episode_trades: int = 0
        self._last_action: int = 0

        # Latency tracking (nanoseconds)
        self._inference_times_ns: deque = deque(maxlen=1000)

        log.info(
            f"RLExecutionAgent initialized | device={self.device} | "
            f"deterministic={deterministic}"
        )

    # ── Checkpoint Loading ─────────────────────────────────────────────────────

    def load_checkpoint(self, path: str) -> None:
        """
        Load a SAC or PPO checkpoint.

        Supports:
          - SAC: {"model_state": ..., "obs_dim": 48, "n_actions": 7, "hidden_dim": 256}
          - PPO: {"policy_state_dict": ..., "state_dim": 54, "n_actions": 7}

        Parameters
        ----------
        path : path to .pt checkpoint file

        Raises
        ------
        FileNotFoundError : if checkpoint file not found
        ValueError        : if checkpoint format is unrecognized
        RuntimeError      : if state dict loading fails (architecture mismatch)
        """
        path = str(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        log.info(f"Loading checkpoint: {path}")
        t0 = time.perf_counter()

        # Load checkpoint (weights_only=False required for numpy arrays embedded in ckpt)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        ckpt_type = detect_checkpoint_type(ckpt)
        log.info(f"  Checkpoint type: {ckpt_type.upper()}")

        if ckpt_type == "sac":
            self._load_sac_checkpoint(ckpt, path)
        elif ckpt_type == "ppo":
            self._load_ppo_checkpoint(ckpt, path)
        else:
            # Try to infer from available keys
            keys = list(ckpt.keys())
            raise ValueError(
                f"Unrecognized checkpoint format. Found keys: {keys}\n"
                f"Expected 'model_state' (SAC) or 'policy_state_dict' (PPO)."
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._checkpoint_path = path
        self._loaded = True

        log.info(
            f"  Checkpoint loaded in {elapsed_ms:.1f}ms | "
            f"obs_dim={self._obs_dim} | n_actions={self._n_actions} | "
            f"device={self.device}"
        )

        # Log checkpoint metadata if available
        if "eval_metrics" in ckpt:
            m = ckpt["eval_metrics"]
            log.info(
                f"  Checkpoint eval metrics: "
                f"Sortino={m.get('sortino', 'N/A'):.3f} | "
                f"WR={m.get('win_rate', 0):.1%} | "
                f"PF={m.get('profit_factor', 0):.2f} | "
                f"AvgPnL={m.get('avg_pnl_ticks', 0):.3f}t"
            )
        if "best_sortino" in ckpt:
            log.info(f"  Best Sortino at checkpoint: {ckpt['best_sortino']:.4f}")
        if "fold" in ckpt:
            log.info(f"  Trained through fold: {ckpt['fold']}")
        if "algorithm" in ckpt:
            log.info(f"  Training algorithm: {ckpt['algorithm']}")

    def _load_sac_checkpoint(self, ckpt: dict, path: str) -> None:
        """Load SAC-format checkpoint (from train_fifo_rl_sac.py)."""
        self._obs_dim = int(ckpt.get("obs_dim", OBS_DIM))
        self._n_actions = int(ckpt.get("n_actions", N_ACTIONS))
        self._hidden_dim = int(ckpt.get("hidden_dim", 256))
        self._checkpoint_type = "sac"

        if self._obs_dim != OBS_DIM:
            log.warning(
                f"  SAC checkpoint obs_dim={self._obs_dim} != expected {OBS_DIM}. "
                f"Make sure live observations match the training format."
            )

        self._net = SACActorNet(
            obs_dim=self._obs_dim,
            n_actions=self._n_actions,
            hidden_dim=self._hidden_dim,
        ).to(self.device)

        try:
            self._net.load_state_dict(ckpt["model_state"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load SAC actor weights from {path}.\n"
                f"This usually means the network architecture changed since training.\n"
                f"Error: {e}"
            )

        self._net.eval()

    def _load_ppo_checkpoint(self, ckpt: dict, path: str) -> None:
        """Load PPO-format checkpoint (from rl_execution_agent.py)."""
        state_dim = int(ckpt.get("state_dim", 54))
        self._obs_dim = state_dim
        self._n_actions = int(ckpt.get("n_actions", N_ACTIONS))
        self._checkpoint_type = "ppo"

        log.info(
            f"  PPO checkpoint detected. state_dim={state_dim}. "
            f"Note: PPO uses 54-dim obs, SAC uses 48-dim obs. "
            f"Ensure observations match checkpoint format."
        )

        self._net = PPOActorNet(
            state_dim=state_dim,
            hidden_dim=128,
        ).to(self.device)

        try:
            self._net.load_state_dict(ckpt["policy_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load PPO policy weights from {path}.\n"
                f"Error: {e}"
            )

        self._net.eval()

        # Store PCA components from PPO checkpoint for embedding projection
        if "pca_mean" in ckpt and "pca_components" in ckpt:
            self._ppo_pca_mean = ckpt["pca_mean"]
            self._ppo_pca_components = ckpt["pca_components"]
            log.info(f"  PCA components loaded from PPO checkpoint.")

        if "confidence_tiers" in ckpt:
            ct = ckpt["confidence_tiers"]
            self._ppo_p90 = ct.get("p90", None)
            self._ppo_p95 = ct.get("p95", None)
            self._ppo_p99 = ct.get("p99", None)
            log.info(
                f"  PPO confidence tiers: p90={self._ppo_p90:.4f}, "
                f"p95={self._ppo_p95:.4f}, p99={self._ppo_p99:.4f}"
            )

    # ── Action Selection ───────────────────────────────────────────────────────

    def get_action(
        self,
        observation: np.ndarray,
        mask_invalid: bool = True,
    ) -> int:
        """
        Get the RL agent's action for the current observation.

        This is the PRIMARY inference method. Called every prediction stride
        (~50 MBO events, sub-second) to decide whether to trade.

        Parameters
        ----------
        observation : np.ndarray of shape (obs_dim,)
                      The full observation vector (48 dims for SAC, 54 for PPO).
                      Must match the format used during training (fifo_rl_env._get_obs()).
        mask_invalid : if True, apply action masking (disallow logically invalid actions,
                       e.g., BUY when already long). Strongly recommended for live use.

        Returns
        -------
        action : int in {0..6}
                 0=HOLD, 1=BUY_LIMIT, 2=SELL_LIMIT, 3=BUY_MARKET,
                 4=SELL_MARKET, 5=CANCEL, 6=MARKET_EXIT

        Raises
        ------
        RuntimeError : if no checkpoint has been loaded yet
        """
        if not self._loaded or self._net is None:
            raise RuntimeError(
                "No checkpoint loaded. Call load_checkpoint() first."
            )

        # ── Input validation ───────────────────────────────────────────────────
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self._obs_dim,):
            raise ValueError(
                f"Observation shape mismatch: got {obs.shape}, "
                f"expected ({self._obs_dim},). "
                f"Check that your observation builder uses OBS_DIM={self._obs_dim}."
            )

        # NaN/Inf check — silently replace with 0 to avoid network failures
        bad_mask = ~np.isfinite(obs)
        if bad_mask.any():
            n_bad = bad_mask.sum()
            log.warning(
                f"get_action: {n_bad} non-finite values in observation at indices "
                f"{np.where(bad_mask)[0].tolist()[:10]}. Replacing with 0."
            )
            obs = obs.copy()
            obs[bad_mask] = 0.0

        # ── Convert to tensor ──────────────────────────────────────────────────
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        # ── Forward pass ───────────────────────────────────────────────────────
        t_start = time.perf_counter_ns()
        with torch.no_grad():
            action = self._forward_greedy(obs_t)

        # ── Action masking ─────────────────────────────────────────────────────
        if mask_invalid:
            action = self._apply_action_mask(action, obs)

        elapsed_ns = time.perf_counter_ns() - t_start
        self._inference_times_ns.append(elapsed_ns)
        self._last_action = action

        if self.verbose:
            log.debug(
                f"obs[0:4]={obs[0:4].round(3)} | "
                f"pos={self._position} | "
                f"pending={self._has_pending_order} | "
                f"action={action}({ACTION_NAMES[action]}) | "
                f"latency={elapsed_ns/1000:.1f}µs"
            )

        return action

    def _forward_greedy(self, obs_t: torch.Tensor) -> int:
        """Run greedy forward pass. Returns int action."""
        if self._checkpoint_type == "sac":
            action_t = self._net.get_action_greedy(obs_t)
        elif self._checkpoint_type == "ppo":
            action_t = self._net.get_action_greedy(obs_t)
        else:
            raise RuntimeError(f"Unknown checkpoint type: {self._checkpoint_type}")
        return int(action_t.item())

    def _apply_action_mask(self, action: int, obs: np.ndarray) -> int:
        """
        Apply logical action masking for live trading safety.

        Invalid action combinations:
          - BUY_LIMIT / BUY_MARKET when already long (position=+1)
          - SELL_LIMIT / SELL_MARKET when already short (position=-1)
          - CANCEL when no pending order
          - MARKET_EXIT when position=0 (nothing to exit)
          - BUY or SELL when pending order already exists (prevent double-entry)

        When the preferred action is invalid, falls back to HOLD (action=0).

        Parameters
        ----------
        action  : the raw RL action (0-6)
        obs     : current observation (used to read position/pending state)

        Returns
        -------
        masked action (0-6)
        """
        # Read position from obs[10] and pending from obs[32]
        # These are set directly by the env and match the env's internal state.
        obs_position = int(round(float(obs[10])))   # 0, +1, -1
        obs_pending  = bool(obs[32] > 0.5)          # 1.0 if pending order exists

        # Also use our tracked state as a safety cross-check
        position = obs_position if obs_position != 0 else self._position

        if action in (1, 3):  # BUY: LIMIT or MARKET
            if position == 1 or obs_pending:
                # Already long or have pending order → can't enter
                return 0  # HOLD

        elif action in (2, 4):  # SELL: LIMIT or MARKET
            if position == -1 or obs_pending:
                # Already short or have pending order → can't enter
                return 0  # HOLD

        elif action == 5:  # CANCEL
            if not obs_pending:
                # No pending order to cancel
                return 0  # HOLD

        elif action == 6:  # MARKET_EXIT
            if position == 0:
                # No position to exit
                return 0  # HOLD

        return action

    def get_action_with_probs(
        self,
        observation: np.ndarray,
        mask_invalid: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """
        Get action AND full probability distribution over all actions.

        Useful for monitoring, logging, and decision confidence tracking.

        Parameters
        ----------
        observation : (obs_dim,) numpy array

        Returns
        -------
        action : int (0-6)
        probs  : np.ndarray (N_ACTIONS,) — probability over each action
        """
        if not self._loaded or self._net is None:
            raise RuntimeError("No checkpoint loaded.")

        obs = np.asarray(observation, dtype=np.float32)
        bad_mask = ~np.isfinite(obs)
        if bad_mask.any():
            obs = obs.copy()
            obs[bad_mask] = 0.0

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            if self._checkpoint_type == "sac":
                probs, _ = self._net.forward(obs_t)
                action = int(probs.argmax(dim=-1).item())
            elif self._checkpoint_type == "ppo":
                logits, _ = self._net.forward(obs_t)
                probs = F.softmax(logits, dim=-1)
                action = int(probs.argmax(dim=-1).item())
            else:
                raise RuntimeError(f"Unknown checkpoint type: {self._checkpoint_type}")

        probs_np = probs.squeeze(0).cpu().numpy()

        if mask_invalid:
            action = self._apply_action_mask(action, obs)

        return action, probs_np

    # ── State Tracking ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Reset agent state for a new trading session/day.

        Call this at the start of each trading day (RTH open) to clear
        intra-day tracking state. Does NOT reload the checkpoint.
        """
        self._position = 0
        self._has_pending_order = False
        self._consecutive_losses = 0
        self._episode_trades = 0
        self._last_action = 0
        log.info("RLExecutionAgent: reset() called — intra-day state cleared.")

    def notify_fill(self, direction: int, fill_type: str = "limit") -> None:
        """
        Notify the agent that an order was filled.

        This updates internal position state for action masking.

        Parameters
        ----------
        direction  : +1 = went long, -1 = went short, 0 = closed position
        fill_type  : "limit" or "market" (for logging only)
        """
        prev = self._position
        if direction in (1, -1) and self._position == 0:
            # Opening a new position
            self._position = direction
            self._has_pending_order = False
        elif direction == 0 or (self._position != 0 and direction != self._position):
            # Closing a position
            self._position = 0
            self._has_pending_order = False
            self._episode_trades += 1

        log.info(
            f"notify_fill: direction={direction} ({fill_type}) | "
            f"position: {prev} → {self._position}"
        )

    def notify_cancel(self) -> None:
        """Notify the agent that the pending order was cancelled."""
        self._has_pending_order = False
        log.debug("notify_cancel: pending order cleared.")

    def notify_order_placed(self, side: int) -> None:
        """Notify the agent that a limit order was placed (not yet filled)."""
        self._has_pending_order = True
        log.debug(f"notify_order_placed: side={side}")

    def notify_trade_result(self, pnl_ticks: float) -> None:
        """
        Notify the agent of a completed trade result.

        Updates consecutive loss tracking (used for position sizing decisions).
        """
        if pnl_ticks < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        log.debug(
            f"notify_trade_result: pnl={pnl_ticks:.3f}t | "
            f"consecutive_losses={self._consecutive_losses}"
        )

    # ── Performance Tracking ───────────────────────────────────────────────────

    @property
    def median_latency_us(self) -> float:
        """Median inference latency in microseconds."""
        if not self._inference_times_ns:
            return 0.0
        times = list(self._inference_times_ns)
        return float(np.median(times)) / 1000.0

    @property
    def p99_latency_us(self) -> float:
        """P99 inference latency in microseconds."""
        if not self._inference_times_ns:
            return 0.0
        return float(np.percentile(list(self._inference_times_ns), 99)) / 1000.0

    def get_latency_stats(self) -> Dict[str, float]:
        """Return inference latency statistics in microseconds."""
        if not self._inference_times_ns:
            return {"n": 0, "median_us": 0.0, "p99_us": 0.0, "mean_us": 0.0}
        times = np.array(list(self._inference_times_ns)) / 1000.0
        return {
            "n": len(times),
            "mean_us": float(times.mean()),
            "median_us": float(np.median(times)),
            "p99_us": float(np.percentile(times, 99)),
            "max_us": float(times.max()),
        }

    def __repr__(self) -> str:
        ckpt_name = Path(self._checkpoint_path).name if self._checkpoint_path else "none"
        return (
            f"RLExecutionAgent("
            f"type={self._checkpoint_type}, "
            f"obs_dim={self._obs_dim}, "
            f"device={self.device}, "
            f"checkpoint={ckpt_name}, "
            f"loaded={self._loaded})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Observation Builder
# ══════════════════════════════════════════════════════════════════════════════

def build_observation(
    # [0:4] Signal
    pred_1s: float = 0.0,
    pred_5s: float = 0.0,
    pred_10s: float = 0.0,
    # [4:8] Book
    bid_size_log: float = 0.0,
    ask_size_log: float = 0.0,
    book_imbalance: float = 0.0,
    spread_ticks: float = 1.0,
    # [8:10] Price
    price_rel_ticks: float = 0.0,
    price_momentum: float = 0.0,
    # [10:14] Position
    position: int = 0,
    unrealized_pnl_ticks: float = 0.0,
    time_in_pos_s: float = 0.0,
    queue_frac_filled: float = 0.0,
    # [14:16] MFE/MAE
    mfe_ticks: float = 0.0,
    mae_ticks: float = 0.0,
    # [16:19] Context
    realized_vol_60s: float = 1.0,
    tod_sin: float = 0.0,
    tod_cos: float = 1.0,
    # [19:24] Flow
    event_density: float = 1.0,
    buy_vol_frac: float = 0.5,
    flow_imbalance: float = 0.0,
    cancel_rate: float = 0.0,
    local_event_rate: float = 1.0,
    # [24:29] Trade history (last 5 PnLs, normalized by 10)
    trade_pnls: Optional[List[float]] = None,
    # [29:32] Rolling stats
    rolling_win_rate: float = 0.5,
    rolling_sortino: float = 0.0,
    consecutive_losses: int = 0,
    # [32:34] Order state
    has_pending_order: bool = False,
    pending_order_side: int = 0,
    # [34:39] CNN-Mamba PCA embedding (5 components)
    embedding_pca: Optional[np.ndarray] = None,
    # [39:45] PatchTST confluence
    pst_1s: float = 0.0,
    pst_5s: float = 0.0,
    pst_10s: float = 0.0,
    confluence_1s: float = 0.0,
    confluence_all: float = 0.0,
    sweep_intensity: float = 0.0,
    # [45:48] Alpha-awareness
    signal_remaining_frac: float = 0.0,
    entry_signal_strength: float = 0.0,
    current_alpha_alignment: float = 0.0,
) -> np.ndarray:
    """
    Build the 48-dim observation vector for the RL agent.

    This helper matches fifo_rl_env.py:_get_obs() exactly.
    Use it to construct observations from live data feeds.

    All scaling/clipping matches the training environment.

    Parameters
    ----------
    (See docstring header for full obs layout)

    Returns
    -------
    obs : np.ndarray of shape (48,), dtype=float32
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    def clamp(x, lo, hi):
        return float(lo if x < lo else (hi if x > hi else x))

    # [0:4] Signal features
    mag = abs(pred_1s)
    confidence_tier = float(min(3, int(mag / 0.25)))
    obs[0] = clamp(pred_1s, -5.0, 5.0)
    obs[1] = clamp(pred_5s, -5.0, 5.0)
    obs[2] = clamp(pred_10s, -5.0, 5.0)
    obs[3] = confidence_tier / 3.0

    # [4:8] Book state
    obs[4] = clamp(bid_size_log / 5.0, -3.0, 3.0)
    obs[5] = clamp(ask_size_log / 5.0, -3.0, 3.0)
    obs[6] = clamp(book_imbalance, -1.0, 1.0)
    obs[7] = clamp(spread_ticks / 4.0, 0.0, 2.0)

    # [8:10] Price
    obs[8] = clamp(price_rel_ticks / 2.0, -1.0, 1.0)
    obs[9] = clamp(price_momentum, -1.0, 1.0)

    # [10:14] Position state
    obs[10] = float(position)
    obs[11] = clamp(unrealized_pnl_ticks / 10.0, -3.0, 3.0) if position != 0 else 0.0
    obs[12] = clamp(time_in_pos_s / 30.0, 0.0, 3.0) if position != 0 else 0.0
    obs[13] = clamp(queue_frac_filled, 0.0, 1.0)

    # [14:16] MFE/MAE
    obs[14] = clamp(mfe_ticks / 10.0, 0.0, 3.0)
    obs[15] = clamp(mae_ticks / 10.0, 0.0, 3.0)

    # [16:19] Market context
    obs[16] = clamp(realized_vol_60s / 5.0, 0.0, 3.0)
    obs[17] = clamp(tod_sin, -1.0, 1.0)
    obs[18] = clamp(tod_cos, -1.0, 1.0)

    # [19:24] Flow features
    obs[19] = clamp(event_density / 3.0, 0.0, 3.0)
    obs[20] = clamp(buy_vol_frac, 0.0, 1.0)
    obs[21] = clamp(flow_imbalance, -1.0, 1.0)
    obs[22] = clamp(cancel_rate / 3.0, 0.0, 3.0)
    obs[23] = clamp(local_event_rate, 0.0, 5.0)
    obs[24] = 0.0  # reserved

    # [24:29] Trade history (last 5 PnLs)
    pnls = trade_pnls or []
    n_hist = len(pnls)
    for i in range(5):
        idx = n_hist - 5 + i
        pnl = pnls[idx] if 0 <= idx < n_hist else 0.0
        obs[24 + i] = clamp(pnl / 10.0, -3.0, 3.0)

    # [29:32] Rolling stats
    obs[29] = clamp(rolling_win_rate, 0.0, 1.0)
    obs[30] = clamp(rolling_sortino / 5.0, -3.0, 3.0)
    obs[31] = clamp(consecutive_losses / 5.0, 0.0, 3.0)

    # [32:34] Order state
    obs[32] = 1.0 if has_pending_order else 0.0
    obs[33] = float(pending_order_side)  # +1, -1, or 0

    # [34:39] CNN-Mamba embedding PCA (5 components)
    if embedding_pca is not None:
        emb = np.asarray(embedding_pca, dtype=np.float32)
        n_comp = min(5, len(emb))
        obs[34:34+n_comp] = np.clip(emb[:n_comp] / 3.0, -3.0, 3.0)

    # [39:45] PatchTST confluence
    obs[39] = clamp(pst_1s, -5.0, 5.0)
    obs[40] = clamp(pst_5s, -5.0, 5.0)
    obs[41] = clamp(pst_10s, -5.0, 5.0)
    obs[42] = clamp(confluence_1s, -3.0, 3.0)
    obs[43] = clamp(confluence_all, -1.0, 1.0)
    obs[44] = clamp(sweep_intensity, 0.0, 3.0)

    # [45:48] Alpha-awareness (HC #114)
    obs[45] = clamp(signal_remaining_frac, 0.0, 1.0)
    obs[46] = clamp(entry_signal_strength, 0.0, 3.0)
    obs[47] = clamp(current_alpha_alignment, -3.0, 3.0)

    return obs


def build_observation_from_dict(d: dict) -> np.ndarray:
    """
    Convenience wrapper: build observation from a dictionary of named fields.

    Useful when the live inference loop has a dict of current state.
    Unknown keys are ignored.

    Parameters
    ----------
    d : dict with any subset of the build_observation() keyword arguments

    Returns
    -------
    obs : np.ndarray (OBS_DIM,)
    """
    return build_observation(**{k: v for k, v in d.items()
                                 if k in build_observation.__code__.co_varnames})


# ══════════════════════════════════════════════════════════════════════════════
# Test Mode: Replay a recorded MBO file through the agent
# ══════════════════════════════════════════════════════════════════════════════

def run_test_mode(
    checkpoint_path: str,
    test_file: str,
    pred_dir: Optional[str] = None,
    patchtst_dir: Optional[str] = None,
    device: str = "cpu",
    max_steps: int = 100_000,
    verbose: bool = False,
) -> Dict:
    """
    Test mode: load checkpoint and run inference on a recorded MBO file.

    This uses the FIFOExecutionEnv to provide properly formatted observations,
    then feeds them to the RLExecutionAgent for action selection.
    Compares the RL actions to a random baseline.

    Parameters
    ----------
    checkpoint_path : path to .pt checkpoint file
    test_file       : path to *_mbo_events.npz file
    pred_dir        : directory with CNN-Mamba fold predictions (optional)
    patchtst_dir    : directory with PatchTST fold predictions (optional)
    device          : "cpu" or "cuda"
    max_steps       : cap on number of env steps (for speed)
    verbose         : if True, log every action

    Returns
    -------
    dict with test results: n_steps, action_distribution, latency_stats,
                            episode_metrics (from env)
    """
    # ── Import env ────────────────────────────────────────────────────────────
    _this_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_this_dir))
    try:
        from fifo_rl_env import (
            FIFOExecutionEnv,
            OBS_DIM as ENV_OBS_DIM,
            EVENT_DIR,
            PRED_DIR,
        )
    except ImportError as e:
        log.error(f"Could not import fifo_rl_env: {e}")
        log.error("Make sure fifo_rl_env.py is in the same directory.")
        raise

    assert ENV_OBS_DIM == OBS_DIM, (
        f"fifo_rl_env.OBS_DIM={ENV_OBS_DIM} does not match live_rl_inference.OBS_DIM={OBS_DIM}. "
        f"Update live_rl_inference.py to match."
    )

    # ── Load agent ────────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("Live RL Inference Test Mode")
    log.info(f"{'='*60}")
    log.info(f"  Checkpoint: {checkpoint_path}")
    log.info(f"  Test file:  {test_file}")
    log.info(f"  Device:     {device}")
    log.info(f"  Max steps:  {max_steps:,}")

    agent = RLExecutionAgent(device=device, deterministic=True, verbose=verbose)
    agent.load_checkpoint(checkpoint_path)
    agent.reset()

    # ── Create env ────────────────────────────────────────────────────────────
    event_dir = Path(test_file).parent
    _pred_dir = Path(pred_dir) if pred_dir else PRED_DIR
    _patchtst_dir = Path(patchtst_dir) if patchtst_dir else None

    log.info(f"  Event dir:   {event_dir}")
    log.info(f"  Pred dir:    {_pred_dir}")
    if _patchtst_dir:
        log.info(f"  PatchTST:    {_patchtst_dir}")

    env = FIFOExecutionEnv(
        event_dir=event_dir,
        pred_dir=_pred_dir,
        patchtst_pred_dir=_patchtst_dir,
        rth_only=True,
        verbose=False,
    )

    # ── Run episode ───────────────────────────────────────────────────────────
    test_file_path = Path(test_file)
    if not test_file_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    log.info(f"\nStarting episode replay: {test_file_path.name}")
    obs = env.reset(episode_file=test_file_path)

    action_counts = [0] * N_ACTIONS
    rewards = []
    step = 0
    done = False
    t0 = time.perf_counter()

    while not done and step < max_steps:
        # Get RL action
        action = agent.get_action(obs, mask_invalid=True)
        action_counts[action] += 1

        # Step env
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        step += 1

        if verbose and step % 10_000 == 0:
            log.info(
                f"  Step {step:,} | "
                f"pos={info['position']} | "
                f"trades={info['episode_trades']} | "
                f"pnl={info['episode_pnl_ticks']:.2f}t"
            )

    elapsed = time.perf_counter() - t0

    # ── Episode metrics ────────────────────────────────────────────────────────
    metrics = env.get_episode_metrics()
    latency_stats = agent.get_latency_stats()

    # ── Print summary ──────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("Test Mode Results")
    log.info(f"{'='*60}")
    log.info(f"  Steps:          {step:,} in {elapsed:.1f}s ({step/elapsed:.0f} steps/s)")
    log.info(f"  Episode done:   {done}")
    log.info(f"  Trades:         {metrics['n_trades']}")
    log.info(f"  Win rate:       {metrics['win_rate']:.1%}")
    log.info(f"  Total PnL:      {metrics['total_pnl_ticks']:+.2f} ticks (${metrics['total_pnl_usd']:+.0f})")
    log.info(f"  Avg PnL/trade:  {metrics['avg_pnl_ticks']:+.3f} ticks")
    log.info(f"  Sortino:        {metrics['sortino']:+.4f}")
    log.info(f"  Sharpe:         {metrics['sharpe']:+.4f}")
    log.info(f"  Profit Factor:  {metrics['profit_factor']:.2f}")
    log.info(f"  Avg hold:       {metrics['avg_hold_secs']:.1f}s")
    log.info(f"")
    log.info(f"  Action distribution:")
    total_actions = sum(action_counts)
    for a, cnt in enumerate(action_counts):
        pct = 100.0 * cnt / max(total_actions, 1)
        bar = "█" * int(pct / 2)
        log.info(f"    {a} {ACTION_NAMES[a]:12s}: {cnt:6,} ({pct:5.1f}%) {bar}")
    log.info(f"")
    log.info(f"  Inference latency:")
    log.info(f"    Mean:   {latency_stats.get('mean_us', 0):.1f} µs")
    log.info(f"    Median: {latency_stats.get('median_us', 0):.1f} µs")
    log.info(f"    P99:    {latency_stats.get('p99_us', 0):.1f} µs")
    log.info(f"{'='*60}")

    return {
        "n_steps": step,
        "elapsed_s": elapsed,
        "steps_per_s": step / elapsed,
        "action_counts": action_counts,
        "action_distribution": {
            ACTION_NAMES[a]: cnt for a, cnt in enumerate(action_counts)
        },
        "episode_metrics": metrics,
        "latency_stats": latency_stats,
        "total_reward": float(sum(rewards)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Live RL Inference Pipeline\n"
            "\n"
            "Test mode: loads a SAC/PPO checkpoint and replays it over a recorded\n"
            "MBO event file to verify the inference pipeline works end-to-end.\n"
            "\n"
            "Examples:\n"
            "  # Test with SAC checkpoint (auto-detect):\n"
            "  python live_rl_inference.py \\\n"
            "      --checkpoint /home/jupiter/Lvl3Quant/output/fifo_sac_rl/best_agent_sac_20260503_120000.pt \\\n"
            "      --test-file /home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3/20260430_mbo_events.npz\n"
            "\n"
            "  # Test with PPO checkpoint:\n"
            "  python live_rl_inference.py \\\n"
            "      --checkpoint /home/nick/Lvl3Quant/output/rl_execution_agent/best_fold_overall.pt \\\n"
            "      --test-file /path/to/20260430_mbo_events.npz \\\n"
            "      --device cpu\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file (SAC or PPO format — auto-detected)",
    )
    p.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to *_mbo_events.npz file for test replay (required for test mode)",
    )
    p.add_argument(
        "--pred-dir",
        type=str,
        default=None,
        help="Directory with CNN-Mamba fold_*_oot_predictions.npz files",
    )
    p.add_argument(
        "--patchtst-dir",
        type=str,
        default=None,
        help="Directory with PatchTST fold_*_oot_predictions.npz files",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device: cpu or cuda (default: cpu for inference)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=200_000,
        help="Max MBO event steps per test episode (default: 200,000)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log every action decision",
    )
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick smoke test: load checkpoint, run 100 random obs, check no crashes",
    )

    return p.parse_args()


def _smoke_test(checkpoint_path: str, device: str = "cpu") -> None:
    """
    Quick smoke test: load checkpoint, run 100 random observations.
    Does NOT require MBO data files. Good for CI/deployment checks.
    """
    log.info(f"\n{'='*60}")
    log.info("Smoke Test: load checkpoint + 100 random observations")
    log.info(f"{'='*60}")

    agent = RLExecutionAgent(device=device, deterministic=True, verbose=False)
    agent.load_checkpoint(checkpoint_path)
    agent.reset()

    rng = np.random.default_rng(42)
    n_ok = 0
    action_counts = [0] * N_ACTIONS

    t0 = time.perf_counter()
    for i in range(100):
        # Random observation (no NaNs)
        obs = rng.standard_normal(agent._obs_dim).astype(np.float32) * 0.1
        # Set position and pending to consistent values
        obs[10] = 0.0  # flat
        obs[32] = 0.0  # no pending

        action = agent.get_action(obs, mask_invalid=True)
        assert 0 <= action < N_ACTIONS, f"Invalid action: {action}"
        action_counts[action] += 1
        n_ok += 1

    elapsed = time.perf_counter() - t0
    lat = agent.get_latency_stats()

    log.info(f"  Passed: {n_ok}/100 observations without error")
    log.info(f"  Total time: {elapsed*1000:.1f}ms ({elapsed*10:.1f}ms/obs)")
    log.info(f"  Median latency: {lat['median_us']:.1f} µs")
    log.info(f"  Action distribution: {dict(zip(ACTION_NAMES.values(), action_counts))}")
    log.info(f"  Agent: {agent}")
    log.info(f"{'='*60}")
    log.info("Smoke test PASSED")


if __name__ == "__main__":
    args = parse_args()

    if args.smoke_test:
        _smoke_test(args.checkpoint, device=args.device)
        sys.exit(0)

    if args.test_file is None:
        # If no test file given, do smoke test as a fallback
        log.info("No --test-file specified. Running smoke test instead.")
        _smoke_test(args.checkpoint, device=args.device)
        sys.exit(0)

    results = run_test_mode(
        checkpoint_path=args.checkpoint,
        test_file=args.test_file,
        pred_dir=args.pred_dir,
        patchtst_dir=args.patchtst_dir,
        device=args.device,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )

    # Exit with non-zero if no trades were generated (likely a bug)
    n_trades = results["episode_metrics"].get("n_trades", 0)
    if n_trades == 0:
        log.warning(
            "WARNING: Zero trades executed during test. "
            "The agent may be stuck in HOLD mode. "
            "Check signal availability and action masking."
        )
        # Don't exit with error — zero trades might be valid (no signal day)
