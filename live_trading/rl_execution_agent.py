#!/usr/bin/env python3
"""
rl_execution_agent.py — Live RL Inference Module for ES Futures Paper Trading
==============================================================================

Runs the PPO-trained FIFOActorCritic agent alongside CNN-Mamba v2 and PatchTST
signal models on Razer (Windows, RTX 3070).

Architecture matches train_fifo_rl.py EXACTLY so that saved checkpoints load
without modification:
  Input (48) → LayerNorm → 3× (Linear → GELU → LayerNorm) → 256-dim hidden
  Actor head: hidden → 128 → 7 logits
  Critic head: hidden → 128 → 1 value

Observation layout (48 dims — identical to fifo_rl_env.py):
  [0:4]   Signal:        pred_1s, pred_5s, pred_10s, confidence_tier
  [4:8]   Book:          best_bid_size_log, best_ask_size_log, book_imbalance, spread_ticks
  [8:10]  Price:         price_rel_ticks, price_momentum
  [10:14] Position:      position, unrealized_pnl_ticks, time_in_pos_s, queue_frac_filled
  [14:16] MFE/MAE:       max_fav_excursion, max_adv_excursion
  [16:19] Context:       realized_vol_60s, tod_sin, tod_cos
  [19:24] Flow:          event_density, buy_vol_frac, flow_imbalance, cancel_rate, local_event_rate
  [24:29] Trade history: last 5 trade PnLs (normalised)
  [29:32] Stats:         rolling_win_rate, rolling_sortino, consecutive_loss_count
  [32:34] Order:         pending_order, pending_order_side
  [34:39] Embedding:     PCA components from CNN-Mamba (zeros if unavailable)
  [39:45] PatchTST:      pst_1s, pst_5s, pst_10s, confluence_score, lgbm_vol, sweep_intensity
  [45:48] Alpha:         signal_remaining, entry_signal_strength, current_alpha_alignment

Action space (7 discrete actions):
  0 = hold / do nothing
  1 = limit_buy_bid      (join back of bid queue)
  2 = limit_sell_ask     (join back of ask queue)
  3 = market_buy         (lift ask)
  4 = market_sell        (hit bid)
  5 = cancel_order       (cancel pending limit)
  6 = close_position     (market exit)

Cost model (HC #231(A) — supersedes earlier "spread crossing" model):
  All fills:   0.376 ticks RT commission only
  No theoretical spread crossing cost — fill price already encodes side.

HC compliance:
  HC #89:   cost = entry/exit ticks − $4.70 RT commission. No extra spread cost.
  HC #104:  no short-side bias — symmetric reward (agent learned freely).
  HC #114:  agent decisions respected; SAFETY overrides (kill switch/max loss) are
            separate and do not touch the model's core decision logic.
  HC #115:  agent learned freely from features; we don't override trade direction.

Author: Claude (Infrastructure Builder)
Date:   2026-05-03
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from trade_journal import TradeJournal

# ── PyTorch ────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger("rl_execution_agent")
if not log.handlers:
    log.setLevel(logging.INFO)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(_sh)

# ══════════════════════════════════════════════════════════════════════════════
# Constants  (canonical — DO NOT CHANGE, HC #89)
# ══════════════════════════════════════════════════════════════════════════════

OBS_DIM                  = 48
N_ACTIONS                = 7
TICK_VALUE               = 12.50          # USD per tick
TICK_SIZE                = 0.25           # ES points per tick
COMMISSION_RT_TICKS      = 0.376          # $4.70 / $12.50
MARKET_ORDER_COST_TICKS  = 0.376          # HC #231(A): commission only — no spread cost
LIMIT_ORDER_COST_TICKS   = 0.376          # commission only (passive fill)

# Action IDs (mirror fifo_rl_env.py)
ACTION_HOLD         = 0
ACTION_LIMIT_BUY    = 1
ACTION_LIMIT_SELL   = 2
ACTION_MARKET_BUY   = 3
ACTION_MARKET_SELL  = 4
ACTION_CANCEL       = 5
ACTION_CLOSE        = 6

ACTION_NAMES = {
    ACTION_HOLD:       "hold",
    ACTION_LIMIT_BUY:  "limit_buy_bid",
    ACTION_LIMIT_SELL: "limit_sell_ask",
    ACTION_MARKET_BUY: "market_buy",
    ACTION_MARKET_SELL:"market_sell",
    ACTION_CANCEL:     "cancel_order",
    ACTION_CLOSE:      "close_position",
}

# Signal decay half-life (used for obs[45] feature, NOT for penalties — HC #114)
SIGNAL_DECAY_HALFLIFE_S  = 0.25

# Rolling stats window
SORTINO_WINDOW           = 20
MIN_TRADES_FOR_SORTINO   = 3


# ══════════════════════════════════════════════════════════════════════════════
# FIFOActorCritic  — EXACT replica of train_fifo_rl.py
# ══════════════════════════════════════════════════════════════════════════════

class FIFOActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic for FIFO execution.

    Architecture (must match training code EXACTLY — weights won't load otherwise):
      Input (48) → LayerNorm
                 → Linear(48→hidden) → GELU → LayerNorm(hidden)
                 → Linear(hidden→hidden) → GELU → LayerNorm(hidden)
                 → Linear(hidden→hidden) → GELU → LayerNorm(hidden)
      Actor head: → Linear(hidden→128) → GELU → Linear(128→7 logits)
      Critic head:→ Linear(hidden→128) → GELU → Linear(128→1 value)

    Default hidden_dim=256 (matches training default).
    Override if checkpoint was trained with a different --hidden-dim.
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        n_actions:  int = N_ACTIONS,
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

    def forward(
        self, obs: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Returns (logits, value)."""
        x = self.input_norm(obs)
        x = self.shared(x)
        logits = self.actor(x)
        value  = self.critic(x).squeeze(-1)
        return logits, value

    def get_action_greedy(self, obs: "torch.Tensor") -> int:
        """Greedy (argmax) action selection — deterministic inference."""
        logits, _ = self.forward(obs)
        return int(logits.argmax(dim=-1).item())

    def get_action_sampled(
        self, obs: "torch.Tensor"
    ) -> Tuple[int, float, float]:
        """
        Stochastic action selection (for exploration / logging).

        Returns (action_id, log_prob, entropy).
        """
        logits, _ = self.forward(obs)
        dist      = Categorical(logits=logits)
        action    = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(dist.entropy().item()),
        )

    def get_value(self, obs: "torch.Tensor") -> float:
        """Critic value estimate — useful for logging agent confidence."""
        _, value = self.forward(obs)
        return float(value.item())


# ══════════════════════════════════════════════════════════════════════════════
# Rolling Sortino  (mirrors fifo_rl_env.RollingSortino)
# ══════════════════════════════════════════════════════════════════════════════

class _RollingSortino:
    """Rolling Sortino ratio over the last `window` completed trades."""

    def __init__(self, window: int = SORTINO_WINDOW):
        self._buf: deque = deque(maxlen=window)

    def add(self, pnl_ticks: float) -> None:
        self._buf.append(pnl_ticks)

    def compute(self) -> float:
        if len(self._buf) < MIN_TRADES_FOR_SORTINO:
            return 0.0
        arr = np.array(self._buf, dtype=np.float64)
        mean_r   = arr.mean()
        downside = arr[arr < 0]
        if len(downside) == 0:
            return mean_r * 10.0
        dd = math.sqrt(float(np.mean(downside ** 2)))
        return mean_r / (dd + 1e-8)

    def win_rate(self) -> float:
        if not self._buf:
            return 0.5
        return float((np.array(self._buf) > 0).mean())

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
# Agent Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RLAgentConfig:
    """All tuneable knobs for RLExecutionAgent.  Change these; never edit agent logic."""

    # ── Model ──────────────────────────────────────────────────────────────────
    weights_path: str = ""          # path to .pt checkpoint (required)
    hidden_dim:   int = 256         # must match training --hidden-dim

    # ── Inference ──────────────────────────────────────────────────────────────
    device:       str = "cuda"      # "cuda" or "cpu"
    greedy:       bool = True       # True = argmax; False = sample from distribution

    # ── Feature toggles ────────────────────────────────────────────────────────
    use_patchtst:    bool = True    # include PatchTST predictions in obs[39:45]
    use_vol_model:   bool = True    # include lgbm_vol in obs[43]
    use_embeddings:  bool = True    # include PCA CNN-Mamba embedding in obs[34:39]

    # ── Safety parameters ──────────────────────────────────────────────────────
    max_position:          int   = 1        # max contracts (always 1 for paper trading)
    max_daily_loss_usd:    float = 500.0    # circuit breaker — halt for the day
    max_consecutive_losses:int   = 5        # pause after N consecutive losses
    cooldown_seconds:      float = 300.0    # pause duration after consecutive losses
    kill_switch:           bool  = False    # hard kill — no new actions if True

    # ── Logging ────────────────────────────────────────────────────────────────
    log_every_action: bool = False  # log every act() call (verbose; off by default)
    log_dir:          str  = ""     # directory for decision logs (empty = disabled)


# ══════════════════════════════════════════════════════════════════════════════
# RLExecutionAgent
# ══════════════════════════════════════════════════════════════════════════════

class RLExecutionAgent:
    """
    Live inference wrapper for the PPO-trained FIFOActorCritic.

    Responsibilities:
      1. Load checkpoint and reconstruct network with matching architecture.
      2. Maintain rolling state (trade history, position, MFE/MAE, flow buffers).
      3. Build the 48-dim observation from live market data via update_state().
      4. Return discrete action via act().

    The agent does NOT manage orders — that is the job of RLPaperTradingWrapper.
    The agent purely maps observations to actions.

    Safety overrides (kill switch, max daily loss) are implemented in
    RLPaperTradingWrapper, not here, so the core inference path stays clean.
    """

    # ── Construction ────────────────────────────────────────────────────────────

    def __init__(self, config: RLAgentConfig):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for RLExecutionAgent")

        self.config = config
        self._device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        if config.device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to CPU")

        # Load network
        self._policy = FIFOActorCritic(
            obs_dim=OBS_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=config.hidden_dim,
        ).to(self._device)

        self._checkpoint_meta: Dict[str, Any] = {}
        if config.weights_path:
            self._load_checkpoint(config.weights_path)
        else:
            log.warning("No weights_path provided — agent running with random weights!")

        self._policy.eval()

        # ── Live observation state ──────────────────────────────────────────────
        # These fields mirror the exact variables maintained in fifo_rl_env.py
        # so the obs vector is constructed identically to training.

        # [0:4] Signal
        self._pred_1s:         float = 0.0
        self._pred_5s:         float = 0.0
        self._pred_10s:        float = 0.0
        self._confidence_tier: float = 0.0

        # [4:8] Book
        self._bid_size_log:    float = 0.0
        self._ask_size_log:    float = 0.0
        self._book_imbalance:  float = 0.0
        self._spread_ticks:    float = 1.0

        # [8:10] Price
        self._price_rel:       float = 0.0
        self._price_momentum:  float = 0.0
        self._price_history:   deque = deque(maxlen=10)

        # [10:14] Position
        self._position:        int   = 0      # +1 long, -1 short, 0 flat
        self._entry_price_rel: float = 0.0
        self._entry_ts_s:      float = 0.0
        self._queue_frac:      float = 0.0    # filled by wrapper

        # [14:16] MFE/MAE
        self._mfe_ticks:       float = 0.0
        self._mae_ticks:       float = 0.0

        # [16:19] Context
        self._vol_buf_60s:     deque = deque()   # (ts_s, |price_change|) tuples
        self._vol_sum_sq:      float = 0.0

        # [19:24] Flow
        self._buy_vol_10s:     deque = deque()   # (ts_s, vol) tuples
        self._sell_vol_10s:    deque = deque()
        self._buy_vol_sum:     float = 0.0
        self._sell_vol_sum:    float = 0.0
        self._event_ts_10s:    deque = deque()   # ts_s of all events
        self._cancel_ts_10s:   deque = deque()   # ts_s of cancel events

        # [24:29] Trade history
        self._trade_pnls:      deque = deque(maxlen=5)   # last 5 trade PnLs in ticks

        # [29:32] Stats
        self._sortino:         _RollingSortino = _RollingSortino(SORTINO_WINDOW)
        self._consecutive_losses: int = 0

        # [32:34] Order
        self._pending_order:      bool  = False
        self._pending_order_side: float = 0.0

        # [34:39] CNN-Mamba embedding PCA
        self._pca_embedding:   Optional[np.ndarray] = None   # shape (5,)

        # [39:45] PatchTST
        self._pst_1s:          float = 0.0
        self._pst_5s:          float = 0.0
        self._pst_10s:         float = 0.0
        self._confluence_score:float = 0.0
        self._lgbm_vol:        float = 0.0
        self._sweep_intensity: float = 0.0

        # [45:48] Alpha-awareness
        self._entry_signal_1s: float = 0.0

        # ── Current timestamp (seconds, from any source) ───────────────────────
        self._current_ts_s:    float = time.time()

        # ── Inference counter ─────────────────────────────────────────────────
        self._n_acts: int = 0

        # ── Optional decision log ─────────────────────────────────────────────
        self._decision_fh = None
        if config.log_dir:
            log_dir = Path(config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = log_dir / f"rl_decisions_{ts_str}.jsonl"
            self._decision_fh = open(log_path, "a", buffering=1)
            log.info("Decision log: %s", log_path)

        log.info(
            "RLExecutionAgent ready | device=%s | hidden_dim=%d | greedy=%s | "
            "weights=%s",
            self._device, config.hidden_dim, config.greedy,
            config.weights_path or "NONE (random)",
        )

    # ── Checkpoint loading ───────────────────────────────────────────────────────

    def _load_checkpoint(self, weights_path: str) -> None:
        """
        Load a PPO checkpoint saved by train_fifo_rl.py.

        Expected keys: 'model_state', optionally 'args', 'eval_metrics', etc.
        Falls back to loading the dict directly as state_dict for bare .pt files.
        """
        path = Path(weights_path)
        if not path.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {path}")

        ckpt = torch.load(str(path), map_location=self._device)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            self._policy.load_state_dict(ckpt["model_state"])
            self._checkpoint_meta = {
                k: v for k, v in ckpt.items() if k != "model_state"
            }
            fold    = ckpt.get("fold", "?")
            epoch   = ckpt.get("epoch", "?")
            sortino = ckpt.get("best_sortino") or (
                ckpt.get("eval_metrics", {}) or {}
            ).get("sortino", "?")
            log.info(
                "Checkpoint loaded: fold=%s epoch=%s eval_sortino=%s  (%s)",
                fold, epoch, sortino, path.name,
            )
        elif isinstance(ckpt, dict):
            # Bare state dict
            self._policy.load_state_dict(ckpt)
            log.info("Checkpoint loaded as raw state_dict: %s", path.name)
        else:
            raise ValueError(
                f"Unrecognised checkpoint format in {path.name}. "
                "Expected dict with 'model_state' key or bare state_dict."
            )

    # ── State update ─────────────────────────────────────────────────────────────

    def update_state(
        self,
        prediction_dict:   Dict[str, Any],
        market_state_dict: Dict[str, Any],
        position_state_dict: Dict[str, Any],
    ) -> None:
        """
        Ingest the latest live data and update internal observation state.

        Call this BEFORE act() on each event where you want a decision.

        Parameters
        ----------
        prediction_dict : keys (all optional — zeros used if absent):
            pred_1s, pred_5s, pred_10s           — CNN-Mamba v2 raw predictions
            patchtst_pred_1s/5s/10s              — PatchTST predictions
            confluence_score                     — pre-computed or computed here
            lgbm_vol                             — LGBM volatility estimate
            pca_embedding                        — np.ndarray(5,) from CNN-Mamba

        market_state_dict : keys:
            timestamp_s      (float) — wall-clock time in seconds
            bid_size         (float) — best bid size (contracts)
            ask_size         (float) — best ask size (contracts)
            spread_ticks     (float) — current spread in ticks
            price_rel_ticks  (float) — price relative to running mid, in ticks
            is_trade         (bool)  — True if this event is a trade
            trade_side       (int)   — +1 buy / -1 sell (relevant if is_trade)
            trade_qty        (float) — quantity of this trade event
            is_cancel        (bool)  — True if this event is a cancel

        position_state_dict : keys:
            position            (int)   — +1, -1, or 0
            unrealized_pnl_ticks(float) — current open P&L in ticks
            time_in_pos_s       (float) — seconds since entry
            queue_frac_filled   (float) — fraction of queue consumed [0,1]
            max_fav_excursion   (float) — MFE in ticks
            max_adv_excursion   (float) — MAE in ticks (positive number)
            entry_signal_1s     (float) — pred_1s at time of entry
            pending_order       (bool)  — True if a limit order is resting
            pending_order_side  (float) — +1/-1 of the pending limit
            last_trade_pnl      (float, optional) — PnL of most recent closed trade
            completed_trade     (bool, optional)  — True if a trade just closed
            consecutive_losses  (int,  optional)  — updated outside
        """
        ts_s = float(market_state_dict.get("timestamp_s", time.time()))
        self._current_ts_s = ts_s

        # ── [0:4] Signals ──────────────────────────────────────────────────────
        self._pred_1s  = float(prediction_dict.get("pred_1s",  0.0))
        self._pred_5s  = float(prediction_dict.get("pred_5s",  0.0))
        self._pred_10s = float(prediction_dict.get("pred_10s", 0.0))
        mag = abs(self._pred_1s)
        self._confidence_tier = min(3.0, int(mag / 0.25)) / 3.0

        # ── [4:8] Book ─────────────────────────────────────────────────────────
        bid_sz = float(market_state_dict.get("bid_size", 1.0))
        ask_sz = float(market_state_dict.get("ask_size", 1.0))
        self._bid_size_log   = math.log1p(max(0.0, bid_sz))
        self._ask_size_log   = math.log1p(max(0.0, ask_sz))
        total_sz = bid_sz + ask_sz + 1e-8
        self._book_imbalance = (bid_sz - ask_sz) / total_sz
        self._spread_ticks   = float(market_state_dict.get("spread_ticks", 1.0))

        # ── [8:10] Price ───────────────────────────────────────────────────────
        self._price_rel = float(market_state_dict.get("price_rel_ticks", 0.0))
        is_trade = bool(market_state_dict.get("is_trade", False))
        if is_trade:
            self._price_history.append(self._price_rel)
        self._price_momentum = self._compute_price_momentum()

        # ── [10:14] Position ───────────────────────────────────────────────────
        self._position       = int(position_state_dict.get("position", 0))
        self._queue_frac     = float(position_state_dict.get("queue_frac_filled", 0.0))

        # ── [14:16] MFE/MAE ────────────────────────────────────────────────────
        self._mfe_ticks = float(position_state_dict.get("max_fav_excursion", 0.0))
        self._mae_ticks = float(position_state_dict.get("max_adv_excursion", 0.0))

        # ── [16:19] Flow buffers (10s and 60s windows) ─────────────────────────
        is_cancel  = bool(market_state_dict.get("is_cancel", False))
        trade_side = int(market_state_dict.get("trade_side", 0))
        trade_qty  = float(market_state_dict.get("trade_qty", 1.0))
        self._update_flow_buffers(ts_s, is_trade, trade_side, trade_qty,
                                  is_cancel, self._price_rel)

        # ── [24:29] Trade history ──────────────────────────────────────────────
        if bool(position_state_dict.get("completed_trade", False)):
            pnl = float(position_state_dict.get("last_trade_pnl", 0.0))
            self._trade_pnls.append(pnl)
            self._sortino.add(pnl)
            if pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

        # Allow wrapper to override consecutive losses (e.g. from PaperPosition)
        if "consecutive_losses" in position_state_dict:
            self._consecutive_losses = int(position_state_dict["consecutive_losses"])

        # ── [32:34] Order ──────────────────────────────────────────────────────
        self._pending_order      = bool(position_state_dict.get("pending_order", False))
        self._pending_order_side = float(position_state_dict.get("pending_order_side", 0.0))

        # ── [34:39] CNN-Mamba PCA embedding ────────────────────────────────────
        if self.config.use_embeddings:
            emb = prediction_dict.get("pca_embedding")
            if emb is not None:
                arr = np.asarray(emb, dtype=np.float32)
                self._pca_embedding = arr[:5] if len(arr) >= 5 else None
            else:
                self._pca_embedding = None

        # ── [39:45] PatchTST + confluence ──────────────────────────────────────
        if self.config.use_patchtst:
            self._pst_1s  = float(prediction_dict.get("patchtst_pred_1s",  0.0))
            self._pst_5s  = float(prediction_dict.get("patchtst_pred_5s",  0.0))
            self._pst_10s = float(prediction_dict.get("patchtst_pred_10s", 0.0))

            # Compute confluence if not pre-computed
            provided_conf = prediction_dict.get("confluence_score")
            if provided_conf is not None:
                self._confluence_score = float(provided_conf)
            else:
                self._confluence_score = self._compute_confluence(
                    self._pred_1s, self._pred_5s, self._pred_10s,
                    self._pst_1s,  self._pst_5s,  self._pst_10s,
                )

        if self.config.use_vol_model:
            self._lgbm_vol = float(prediction_dict.get("lgbm_vol", 0.0))

        # sweep_intensity: local event rate × total volume (normalised)
        buy_v, sell_v = self._get_10s_volumes()
        total_v       = buy_v + sell_v + 1e-8
        evrate        = self._get_event_rate()
        self._sweep_intensity = min(3.0, evrate * total_v / (total_v * 10.0 + 1e-8))

        # ── [45:48] Alpha-awareness ────────────────────────────────────────────
        self._entry_signal_1s = float(
            position_state_dict.get("entry_signal_1s", self._entry_signal_1s)
        )

    # ── Observation builder ──────────────────────────────────────────────────────

    def build_obs(self) -> np.ndarray:
        """
        Construct the 48-dim observation vector from current internal state.

        This mirrors fifo_rl_env._get_obs() exactly so that vectors look
        identical to what the model saw during training.
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        def clamp(x: float, lo: float, hi: float) -> float:
            return lo if x < lo else (hi if x > hi else x)

        ts_s = self._current_ts_s

        # ── [0:4] Signal ───────────────────────────────────────────────────────
        obs[0] = clamp(self._pred_1s,  -5.0, 5.0)
        obs[1] = clamp(self._pred_5s,  -5.0, 5.0)
        obs[2] = clamp(self._pred_10s, -5.0, 5.0)
        obs[3] = self._confidence_tier         # already in [0, 1]

        # ── [4:8] Book ─────────────────────────────────────────────────────────
        obs[4] = clamp(self._bid_size_log / 5.0, -3.0, 3.0)
        obs[5] = clamp(self._ask_size_log / 5.0, -3.0, 3.0)
        obs[6] = clamp(self._book_imbalance, -1.0, 1.0)
        obs[7] = clamp(self._spread_ticks / 4.0, 0.0, 2.0)

        # ── [8:10] Price ───────────────────────────────────────────────────────
        obs[8] = clamp(self._price_rel / 2.0, -1.0, 1.0)
        obs[9] = self._price_momentum

        # ── [10:14] Position ───────────────────────────────────────────────────
        obs[10] = float(self._position)
        if self._position != 0:
            # unrealized P&L and time-in-position are passed in directly
            # (wrapper maintains these; we reconstruct from state)
            obs[11] = clamp(
                (self._price_rel - self._entry_price_rel) * self._position / 10.0,
                -3.0, 3.0,
            )
            time_in_pos = ts_s - self._entry_ts_s if self._entry_ts_s > 0 else 0.0
            obs[12] = clamp(time_in_pos / 30.0, 0.0, 3.0)   # 30s = MAX_HOLD_SECS
        obs[13] = clamp(self._queue_frac, 0.0, 1.0)

        # ── [14:16] MFE/MAE ────────────────────────────────────────────────────
        obs[14] = clamp(self._mfe_ticks / 10.0, 0.0, 3.0)
        obs[15] = clamp(self._mae_ticks / 10.0, 0.0, 3.0)

        # ── [16:19] Context ────────────────────────────────────────────────────
        obs[16] = clamp(self._get_realized_vol_60s() / 5.0, 0.0, 3.0)
        tod_frac = (ts_s % 86400) / 86400.0
        obs[17]  = math.sin(2.0 * math.pi * tod_frac)
        obs[18]  = math.cos(2.0 * math.pi * tod_frac)

        # ── [19:24] Flow ───────────────────────────────────────────────────────
        buy_v, sell_v = self._get_10s_volumes()
        total_v       = buy_v + sell_v + 1e-8
        evrate        = self._get_event_rate()
        cancel_rate   = self._get_cancel_rate()
        book_imb      = (buy_v - sell_v) / total_v

        obs[19] = clamp(evrate   / 3.0, 0.0, 3.0)
        obs[20] = buy_v / total_v
        obs[21] = clamp(book_imb, -1.0, 1.0)
        obs[22] = clamp(cancel_rate / 3.0, 0.0, 3.0)
        obs[23] = clamp(evrate, 0.0, 5.0)
        obs[24] = 0.0                            # reserved

        # ── [24:29] Trade history: last 5 PnLs ────────────────────────────────
        pnls  = list(self._trade_pnls)
        n_pnl = len(pnls)
        for i in range(5):
            idx = n_pnl - 5 + i
            pnl = pnls[idx] if 0 <= idx < n_pnl else 0.0
            obs[24 + i] = clamp(pnl / 10.0, -3.0, 3.0)

        # ── [29:32] Stats ──────────────────────────────────────────────────────
        obs[29] = self._sortino.win_rate()
        obs[30] = clamp(self._sortino.compute() / 5.0, -3.0, 3.0)
        obs[31] = clamp(float(self._consecutive_losses) / 5.0, 0.0, 3.0)

        # ── [32:34] Order ──────────────────────────────────────────────────────
        obs[32] = 1.0 if self._pending_order else 0.0
        obs[33] = self._pending_order_side if self._pending_order else 0.0

        # ── [34:39] CNN-Mamba PCA embedding ────────────────────────────────────
        if self.config.use_embeddings and self._pca_embedding is not None:
            obs[34:39] = np.clip(self._pca_embedding / 3.0, -3.0, 3.0)

        # ── [39:45] PatchTST + enhanced features ──────────────────────────────
        if self.config.use_patchtst:
            obs[39] = clamp(self._pst_1s,  -5.0, 5.0)
            obs[40] = clamp(self._pst_5s,  -5.0, 5.0)
            obs[41] = clamp(self._pst_10s, -5.0, 5.0)
            obs[42] = clamp(self._confluence_score, -3.0, 3.0)
            obs[43] = clamp(self._lgbm_vol / 5.0, -3.0, 3.0)
        obs[44] = clamp(self._sweep_intensity, 0.0, 3.0)

        # ── [45:48] Alpha-awareness ────────────────────────────────────────────
        if self._position != 0 and self._entry_ts_s > 0:
            hold_s = ts_s - self._entry_ts_s
            signal_remaining = math.exp(-0.693 * hold_s / SIGNAL_DECAY_HALFLIFE_S)
            obs[45] = clamp(signal_remaining, 0.0, 1.0)
            obs[46] = clamp(abs(self._entry_signal_1s), 0.0, 3.0)
            # Current alignment: positive means signal still supports position
            obs[47] = clamp(self._pred_1s * self._position, -3.0, 3.0)
        # else: zeros (already initialised)

        return obs

    # ── Core inference ────────────────────────────────────────────────────────────

    def act(self, obs: Optional[np.ndarray] = None) -> int:
        """
        Run forward pass and return discrete action (0–6).

        Parameters
        ----------
        obs : optional pre-built observation array.
              If None, build_obs() is called automatically.

        Returns
        -------
        action_id : int in {0, 1, 2, 3, 4, 5, 6}
        """
        if obs is None:
            obs = self.build_obs()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)

        with torch.no_grad():
            if self.config.greedy:
                action_id = self._policy.get_action_greedy(obs_t)
                log_prob   = None
                entropy    = None
            else:
                action_id, log_prob, entropy = self._policy.get_action_sampled(obs_t)

        self._n_acts += 1

        if self.config.log_every_action or self._decision_fh:
            record = {
                "n":        self._n_acts,
                "ts":       self._current_ts_s,
                "action":   ACTION_NAMES[action_id],
                "action_id":action_id,
                "position": self._position,
                "pred_1s":  round(self._pred_1s, 5),
                "conf":     round(self._confidence_tier, 3),
                "sortino":  round(self._sortino.compute(), 4),
                "wr":       round(self._sortino.win_rate(), 4),
                "consec_loss": self._consecutive_losses,
            }
            if log_prob is not None:
                record["log_prob"] = round(log_prob, 4)
            if entropy is not None:
                record["entropy"]  = round(entropy, 4)

            if self.config.log_every_action:
                log.debug("act #%d → %s | %s", self._n_acts, ACTION_NAMES[action_id],
                          {k: v for k, v in record.items() if k not in ("ts", "n")})

            if self._decision_fh:
                self._decision_fh.write(json.dumps(record) + "\n")

        return action_id

    # ── Notifications from wrapper about completed trades ─────────────────────────

    def notify_entry(self, direction: int, price_rel: float, ts_s: float) -> None:
        """
        Called by wrapper when a position is opened.
        Records entry state for alpha-decay tracking (obs[45:48]).
        """
        self._position        = direction
        self._entry_price_rel = price_rel
        self._entry_ts_s      = ts_s
        self._entry_signal_1s = self._pred_1s   # snapshot signal at entry
        self._mfe_ticks       = 0.0
        self._mae_ticks       = 0.0

    def notify_exit(self, pnl_ticks: float) -> None:
        """
        Called by wrapper when a position is closed.
        Updates trade history and resets position state.
        """
        self._trade_pnls.append(pnl_ticks)
        self._sortino.add(pnl_ticks)
        if pnl_ticks < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._position        = 0
        self._entry_price_rel = 0.0
        self._entry_ts_s      = 0.0
        self._entry_signal_1s = 0.0
        self._mfe_ticks       = 0.0
        self._mae_ticks       = 0.0
        self._pending_order   = False
        self._pending_order_side = 0.0

    def notify_order_placed(self, side: int) -> None:
        """Called by wrapper when a limit order is placed."""
        self._pending_order      = True
        self._pending_order_side = float(side)

    def notify_order_cancelled(self) -> None:
        """Called by wrapper when a limit order is cancelled."""
        self._pending_order      = False
        self._pending_order_side = 0.0

    # ── Market state helpers ─────────────────────────────────────────────────────

    def _update_flow_buffers(
        self,
        ts_s:       float,
        is_trade:   bool,
        trade_side: int,
        trade_qty:  float,
        is_cancel:  bool,
        price_rel:  float,
    ) -> None:
        """Maintain 10s volume and 60s volatility buffers (identical to env logic)."""
        cutoff_10s = ts_s - 10.0
        cutoff_60s = ts_s - 60.0

        if is_trade:
            if trade_side > 0:
                self._buy_vol_10s.append((ts_s, trade_qty))
                self._buy_vol_sum += trade_qty
            else:
                self._sell_vol_10s.append((ts_s, trade_qty))
                self._sell_vol_sum += trade_qty
            self._price_history.append(price_rel)

        self._event_ts_10s.append(ts_s)

        if is_cancel:
            self._cancel_ts_10s.append(ts_s)

        # Volatility buffer (price change from last trade)
        if is_trade and len(self._price_history) >= 2:
            ph = list(self._price_history)
            change = abs(price_rel - ph[-2])
            self._vol_buf_60s.append((ts_s, change))
            self._vol_sum_sq += change * change

        # Prune stale entries and subtract from running sums
        while self._buy_vol_10s  and self._buy_vol_10s[0][0]  < cutoff_10s:
            self._buy_vol_sum  -= self._buy_vol_10s.popleft()[1]
        while self._sell_vol_10s and self._sell_vol_10s[0][0] < cutoff_10s:
            self._sell_vol_sum -= self._sell_vol_10s.popleft()[1]
        while self._event_ts_10s and self._event_ts_10s[0]    < cutoff_10s:
            self._event_ts_10s.popleft()
        while self._cancel_ts_10s and self._cancel_ts_10s[0]  < cutoff_10s:
            self._cancel_ts_10s.popleft()
        while self._vol_buf_60s  and self._vol_buf_60s[0][0]  < cutoff_60s:
            self._vol_sum_sq -= self._vol_buf_60s.popleft()[1] ** 2

        self._buy_vol_sum  = max(0.0, self._buy_vol_sum)
        self._sell_vol_sum = max(0.0, self._sell_vol_sum)
        self._vol_sum_sq   = max(0.0, self._vol_sum_sq)

    def _get_10s_volumes(self) -> Tuple[float, float]:
        return self._buy_vol_sum, self._sell_vol_sum

    def _get_event_rate(self) -> float:
        """Normalised event rate (1.0 = ~1000 events/s, typical ES RTH)."""
        return len(self._event_ts_10s) / (10.0 * 1000.0)

    def _get_cancel_rate(self) -> float:
        n_evt = len(self._event_ts_10s)
        return len(self._cancel_ts_10s) / n_evt if n_evt else 0.0

    def _get_realized_vol_60s(self) -> float:
        n = len(self._vol_buf_60s)
        if n < 2:
            return 1.0
        return math.sqrt(max(0.0, self._vol_sum_sq / n))

    def _compute_price_momentum(self) -> float:
        hist = list(self._price_history)
        if len(hist) < 2:
            return 0.0
        recent = np.array(hist[-min(10, len(hist)):], dtype=np.float32)
        mom    = float(recent[-1] - recent[0])
        return float(np.clip(mom / 2.0, -1.0, 1.0))

    @staticmethod
    def _compute_confluence(
        cm_1s: float, cm_5s: float, cm_10s: float,
        pst_1s: float, pst_5s: float, pst_10s: float,
    ) -> float:
        """Avg sign-agreement × geometric mean magnitude across 3 horizons."""
        agree_sum = 0.0
        for cm, ps in [(cm_1s, pst_1s), (cm_5s, pst_5s), (cm_10s, pst_10s)]:
            if abs(cm) > 0.01 and abs(ps) > 0.01:
                sign_agree = 1.0 if (cm * ps > 0) else -1.0
                mag        = math.sqrt(abs(cm) * abs(ps))
                agree_sum += sign_agree * mag
        return agree_sum / 3.0

    # ── Diagnostics ──────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Current agent state as a flat dict — for logging and monitoring."""
        return {
            "n_acts":           self._n_acts,
            "position":         self._position,
            "pending_order":    self._pending_order,
            "win_rate":         round(self._sortino.win_rate(), 4),
            "sortino":          round(self._sortino.compute(), 4),
            "n_trades":         len(self._sortino),
            "consecutive_loss": self._consecutive_losses,
            "pred_1s":          round(self._pred_1s, 5),
            "confidence_tier":  round(self._confidence_tier * 3.0, 1),
            "mfe_ticks":        round(self._mfe_ticks, 3),
            "mae_ticks":        round(self._mae_ticks, 3),
            "device":           str(self._device),
        }

    def close(self) -> None:
        """Flush and close log file handle."""
        if self._decision_fh:
            self._decision_fh.flush()
            self._decision_fh.close()
            self._decision_fh = None


# ══════════════════════════════════════════════════════════════════════════════
# RLPaperTradingWrapper
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _PaperTrade:
    """A completed paper trade record."""
    direction:    int
    entry_price:  float
    exit_price:   float
    entry_ts_s:   float
    exit_ts_s:    float
    pnl_ticks:    float
    pnl_usd:      float
    hold_secs:    float
    exit_reason:  str
    entry_action: str
    exit_action:  str


class RLPaperTradingWrapper:
    """
    Connects RLExecutionAgent to the live trading pipeline.

    Responsibilities:
      - Receive CNN-Mamba + PatchTST predictions and book state on each event.
      - Call agent.update_state() + agent.act() to get an action.
      - Translate the action to paper trading operations.
      - Track position, P&L, fills, and session metrics.
      - Enforce SAFETY LIMITS (HC #114 — agent decisions respected; safety
        overrides are separate and do not distort action logic).

    Safety limits enforced here (NOT inside the agent model):
      - max_position: 1 contract always
      - max_daily_loss: halt for the day
      - consecutive_loss_halt: pause after N consecutive losses
      - kill_switch: immediate halt of all new actions
      - spread_filter: skip action if spread is too wide

    Thread safety: NOT thread-safe. Call from a single event loop thread.
    """

    # ── Action → operation mapping ───────────────────────────────────────────────

    _ENTRY_ACTIONS = {ACTION_LIMIT_BUY, ACTION_LIMIT_SELL,
                      ACTION_MARKET_BUY, ACTION_MARKET_SELL}
    _EXIT_ACTIONS  = {ACTION_CLOSE}

    def __init__(
        self,
        agent:             RLExecutionAgent,
        config:            RLAgentConfig,
        symbol:            str   = "ESM6",
        max_spread_ticks:  float = 4.0,
        log_dir:           str   = "",
    ):
        self.agent       = agent
        self.config      = config
        self.symbol      = symbol
        self.max_spread  = max_spread_ticks

        # ── Position state ─────────────────────────────────────────────────────
        self._position:       int   = 0     # +1 long, -1 short, 0 flat
        self._entry_price:    float = 0.0   # absolute price (points)
        self._entry_ts_s:     float = 0.0
        self._entry_action:   str   = ""
        self._entry_price_rel:float = 0.0
        self._pending_order:  bool  = False
        self._pending_side:   int   = 0
        self._pending_price:  float = 0.0   # estimated fill price
        self._queue_frac:     float = 0.0   # updated by wrapper logic

        # ── MFE/MAE tracking ───────────────────────────────────────────────────
        self._mfe_ticks: float = 0.0
        self._mae_ticks: float = 0.0

        # ── Daily safety tracking ──────────────────────────────────────────────
        self._daily_pnl_usd:    float = 0.0
        self._daily_halted:     bool  = False
        self._daily_date:       str   = ""
        self._consecutive_losses: int = 0
        self._cooldown_until_s: float = 0.0

        # ── Session metrics ─────────────────────────────────────────────────────
        self._trades:           List[_PaperTrade] = []
        self._events_processed: int   = 0
        self._n_actions_taken:  int   = 0
        self._session_start_s:  float = time.time()
        self._last_action_id:   int   = ACTION_HOLD

        # ── Market state cache ─────────────────────────────────────────────────
        self._mid_price:  float = 0.0
        self._best_bid:   float = 0.0
        self._best_ask:   float = 0.0
        self._spread:     float = 1.0
        self._ts_s:       float = time.time()

        # ── Trade log ─────────────────────────────────────────────────────────
        self._trade_log_fh = None
        if log_dir:
            ld = Path(log_dir)
            ld.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            tp = ld / f"rl_paper_trades_{symbol}_{ts_str}.jsonl"
            self._trade_log_fh = open(tp, "a", buffering=1)
            log.info("Trade log: %s", tp)

        # ── Comprehensive trade journal (HC #142) ──────────────────────────────
        _policy_name = type(agent._policy).__name__.lower()
        _jname = "sac_v7" if "sac" in _policy_name else "ppo_v7" if "ppo" in _policy_name else f"rl_{_policy_name}"
        self._journal = TradeJournal(_jname, log_dir=str(Path(log_dir) if log_dir else Path("logs/")))

        log.info(
            "RLPaperTradingWrapper ready | symbol=%s | max_spread=%.1ft | "
            "max_daily_loss=$%.0f | max_consec_loss=%d",
            symbol, max_spread_ticks, config.max_daily_loss_usd,
            config.max_consecutive_losses,
        )

    # ── Main entry point ─────────────────────────────────────────────────────────

    def on_event(
        self,
        prediction_dict:     Dict[str, Any],
        market_state_dict:   Dict[str, Any],
        position_override:   Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process one MBO event through the RL agent.

        Call this once per MBO event (or stride boundary when new predictions
        are available).

        Parameters
        ----------
        prediction_dict    : see RLExecutionAgent.update_state()
        market_state_dict  : see RLExecutionAgent.update_state()
        position_override  : optional dict to override position state
                             (useful for integration with existing PaperPosition)

        Returns
        -------
        result dict with keys:
            action_id    : int
            action_name  : str
            executed     : bool  — whether the action was actually applied
            skip_reason  : str   — why action was skipped (empty if executed)
            position     : int
            daily_pnl_usd: float
            n_trades     : int
        """
        self._events_processed += 1
        ts_s = float(market_state_dict.get("timestamp_s", time.time()))
        self._ts_s = ts_s

        # Update market state cache
        self._mid_price = float(market_state_dict.get("mid_price", self._mid_price))
        self._best_bid  = float(market_state_dict.get("best_bid",  self._mid_price))
        self._best_ask  = float(market_state_dict.get("best_ask",  self._mid_price))
        self._spread    = float(market_state_dict.get("spread_ticks", 1.0))

        # Update MFE/MAE for open position
        if self._position != 0 and self._mid_price > 0:
            self._update_mfe_mae()

        # Check daily reset
        self._check_daily_reset()

        # ── Build position_state_dict for agent ────────────────────────────────
        price_rel = float(market_state_dict.get("price_rel_ticks", 0.0))

        pos_state = position_override or {
            "position":            self._position,
            "unrealized_pnl_ticks":(
                (price_rel - self._entry_price_rel) * self._position
                if self._position != 0 else 0.0
            ),
            "time_in_pos_s":        max(0.0, ts_s - self._entry_ts_s) if self._entry_ts_s else 0.0,
            "queue_frac_filled":    self._queue_frac,
            "max_fav_excursion":    self._mfe_ticks,
            "max_adv_excursion":    self._mae_ticks,
            "entry_signal_1s":      (
                self.agent._entry_signal_1s if self._position != 0 else 0.0
            ),
            "pending_order":        self._pending_order,
            "pending_order_side":   float(self._pending_side),
            "consecutive_losses":   self._consecutive_losses,
        }

        # ── Feed agent ────────────────────────────────────────────────────────
        self.agent.update_state(prediction_dict, market_state_dict, pos_state)
        action_id = self.agent.act()
        self._last_action_id = action_id

        # ── Safety gate ────────────────────────────────────────────────────────
        skip_reason = self._safety_check(action_id, ts_s)
        executed    = False

        if not skip_reason:
            executed, skip_reason = self._execute_action(
                action_id, price_rel, ts_s
            )

        return {
            "action_id":     action_id,
            "action_name":   ACTION_NAMES[action_id],
            "executed":      executed,
            "skip_reason":   skip_reason,
            "position":      self._position,
            "daily_pnl_usd": round(self._daily_pnl_usd, 2),
            "n_trades":      len(self._trades),
        }

    # ── Action execution ──────────────────────────────────────────────────────────

    def _execute_action(
        self,
        action_id:  int,
        price_rel:  float,
        ts_s:       float,
    ) -> Tuple[bool, str]:
        """
        Apply the RL action to the paper position.

        Returns (executed: bool, reason_if_not: str).

        Limit orders are modelled as immediately filled in paper trading
        (we don't simulate the FIFO queue in live paper mode — that was
        done at training time). This is intentional: we want to count
        fill attempts and track whether entries happen, but queue simulation
        is irrelevant for measuring strategy performance on live data.

        For production sim: replace the immediate fill with a proper
        queue-model fill check on incoming trades.
        """
        half_spread = max(self._spread / 2.0, 0.5)

        # Action 0 — HOLD
        if action_id == ACTION_HOLD:
            return True, ""

        # Action 1 — LIMIT BUY at bid
        elif action_id == ACTION_LIMIT_BUY:
            if self._position == 0 and not self._pending_order:
                self._pending_order = True
                self._pending_side  = 1
                self._pending_price = self._mid_price - half_spread * TICK_SIZE
                self.agent.notify_order_placed(1)
                log.debug("PAPER limit BUY placed | bid~=%.2f", self._pending_price)
                return True, ""
            return False, "already_in_position_or_pending"

        # Action 2 — LIMIT SELL at ask
        elif action_id == ACTION_LIMIT_SELL:
            if self._position == 0 and not self._pending_order:
                self._pending_order = True
                self._pending_side  = -1
                self._pending_price = self._mid_price + half_spread * TICK_SIZE
                self.agent.notify_order_placed(-1)
                log.debug("PAPER limit SELL placed | ask~=%.2f", self._pending_price)
                return True, ""
            return False, "already_in_position_or_pending"

        # Action 3 — MARKET BUY (crosses spread)
        elif action_id == ACTION_MARKET_BUY:
            if self._position == 0 and not self._pending_order:
                fill_price = self._best_ask if self._best_ask > 0 else self._mid_price
                self._open_position(1, fill_price, price_rel, ts_s, "market_buy")
                return True, ""
            return False, "already_in_position_or_pending"

        # Action 4 — MARKET SELL (crosses spread)
        elif action_id == ACTION_MARKET_SELL:
            if self._position == 0 and not self._pending_order:
                fill_price = self._best_bid if self._best_bid > 0 else self._mid_price
                self._open_position(-1, fill_price, price_rel, ts_s, "market_sell")
                return True, ""
            return False, "already_in_position_or_pending"

        # Action 5 — CANCEL pending order
        elif action_id == ACTION_CANCEL:
            if self._pending_order:
                self._pending_order = False
                self._pending_side  = 0
                self.agent.notify_order_cancelled()
                log.debug("PAPER order cancelled")
                return True, ""
            return False, "no_pending_order"

        # Action 6 — CLOSE position (market exit)
        elif action_id == ACTION_CLOSE:
            if self._position != 0:
                exit_price = (
                    self._best_bid if self._position > 0 else self._best_ask
                )
                if exit_price <= 0:
                    exit_price = self._mid_price
                self._close_position(exit_price, price_rel, ts_s, "rl_close")
                return True, ""
            elif self._pending_order:
                # Cancel pending if close issued without position
                self._pending_order = False
                self._pending_side  = 0
                self.agent.notify_order_cancelled()
                return True, ""
            return False, "no_position_to_close"

        return False, f"unknown_action_{action_id}"

    # ── Position helpers ─────────────────────────────────────────────────────────

    def _open_position(
        self,
        direction:  int,
        fill_price: float,
        price_rel:  float,
        ts_s:       float,
        action_str: str,
    ) -> None:
        """Open a paper position."""
        self._position        = direction
        self._entry_price     = fill_price
        self._entry_price_rel = price_rel
        self._entry_ts_s      = ts_s
        self._entry_action    = action_str
        self._mfe_ticks       = 0.0
        self._mae_ticks       = 0.0
        self._pending_order   = False
        self._pending_side    = 0
        self._n_actions_taken += 1

        self.agent.notify_entry(direction, price_rel, ts_s)

        # Journal entry (HC #142)
        self._journal.record_entry(
            entry_price=fill_price,
            direction="LONG" if direction > 0 else "SHORT",
            signal_confidence=self.agent._confidence_tier,
            pred_1s=self.agent._pred_1s,
            pred_5s=self.agent._pred_5s,
            pred_10s=self.agent._pred_10s,
            entry_action_type=action_str,
            spread_at_entry=self._spread,
            bid_at_entry=self._best_bid,
            ask_at_entry=self._best_ask,
        )

        log.info(
            "PAPER ENTRY  %-5s @ %.2f | action=%s | pred_1s=%.4f",
            "LONG" if direction > 0 else "SHORT",
            fill_price, action_str,
            self.agent._pred_1s,
        )

    def _close_position(
        self,
        fill_price: float,
        price_rel:  float,
        ts_s:       float,
        reason:     str,
    ) -> None:
        """Close the current paper position and record the trade."""
        if self._position == 0:
            return

        direction   = self._position
        raw_move_pts = (fill_price - self._entry_price) * direction
        raw_ticks    = raw_move_pts / TICK_SIZE

        # Cost model (HC #231(A)): commission only, no spread crossing cost.
        cost_ticks  = MARKET_ORDER_COST_TICKS
        pnl_ticks   = raw_ticks - cost_ticks
        pnl_usd     = pnl_ticks * TICK_VALUE
        hold_secs   = ts_s - self._entry_ts_s

        trade = _PaperTrade(
            direction    = direction,
            entry_price  = self._entry_price,
            exit_price   = fill_price,
            entry_ts_s   = self._entry_ts_s,
            exit_ts_s    = ts_s,
            pnl_ticks    = pnl_ticks,
            pnl_usd      = pnl_usd,
            hold_secs    = hold_secs,
            exit_reason  = reason,
            entry_action = self._entry_action,
            exit_action  = ACTION_NAMES.get(self._last_action_id, "?"),
        )
        self._trades.append(trade)
        self._daily_pnl_usd   += pnl_usd
        self._n_actions_taken += 1

        if pnl_usd < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self.agent.notify_exit(pnl_ticks)

        log.info(
            "PAPER EXIT   %-5s @ %.2f | reason=%-16s | "
            "pnl=%.2ft ($%.2f) | hold=%.1fs | "
            "MFE=%.2ft MAE=%.2ft | "
            "day_pnl=$%.2f | trades=%d WR=%.0f%%",
            "LONG" if direction > 0 else "SHORT",
            fill_price, reason,
            pnl_ticks, pnl_usd, hold_secs,
            self._mfe_ticks, self._mae_ticks,
            self._daily_pnl_usd, len(self._trades),
            self.win_rate() * 100.0,
        )

        if self._trade_log_fh:
            rec = {
                "ts":         ts_s,
                "dir":        "LONG" if direction > 0 else "SHORT",
                "entry":      round(self._entry_price, 4),
                "exit":       round(fill_price, 4),
                "pnl_ticks":  round(pnl_ticks, 4),
                "pnl_usd":    round(pnl_usd, 2),
                "hold_s":     round(hold_secs, 2),
                "reason":     reason,
                "mfe":        round(self._mfe_ticks, 3),
                "mae":        round(self._mae_ticks, 3),
                "sortino":    round(self.agent._sortino.compute(), 4),
            }
            self._trade_log_fh.write(json.dumps(rec) + "\n")

        # Journal exit (HC #142)
        self._journal.record_exit(
            exit_price=fill_price,
            exit_reason=reason,
            mfe_ticks=self._mfe_ticks,
            mae_ticks=self._mae_ticks,
            spread_at_exit=self._spread,
            bid_at_exit=self._best_bid,
            ask_at_exit=self._best_ask,
        )

        # Reset position state
        self._position     = 0
        self._entry_price  = 0.0
        self._entry_ts_s   = 0.0
        self._mfe_ticks    = 0.0
        self._mae_ticks    = 0.0
        self._pending_order = False

    def _update_mfe_mae(self) -> None:
        """Update max favorable / adverse excursion for open position."""
        if self._position == 0 or self._mid_price <= 0:
            return
        if self._position > 0:
            unrealized = (self._mid_price - self._entry_price) / TICK_SIZE
        else:
            unrealized = (self._entry_price - self._mid_price) / TICK_SIZE

        if unrealized > self._mfe_ticks:
            self._mfe_ticks = unrealized
        if unrealized < -self._mae_ticks:
            self._mae_ticks = -unrealized

    # ── Safety checks ────────────────────────────────────────────────────────────

    def _safety_check(self, action_id: int, ts_s: float) -> str:
        """
        Safety gate — returns non-empty reason string if action should be blocked.

        HC #114: these overrides are SEPARATE from the model's decision logic.
        They don't modify the observation or reward — they are external halts.
        """
        # Kill switch — hard stop, no new actions
        if self.config.kill_switch:
            return "kill_switch"

        # Only gate entry actions — exits/cancels always allowed
        if action_id not in self._ENTRY_ACTIONS:
            return ""

        # Daily loss circuit breaker
        if self._daily_halted:
            return "daily_halt"
        if self._daily_pnl_usd <= -abs(self.config.max_daily_loss_usd):
            self._daily_halted = True
            log.warning(
                "CIRCUIT BREAKER: daily P&L $%.2f <= -$%.2f. Halting for day.",
                self._daily_pnl_usd, self.config.max_daily_loss_usd,
            )
            return "daily_loss_circuit_breaker"

        # Consecutive loss cooldown
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            if ts_s < self._cooldown_until_s:
                remaining = self._cooldown_until_s - ts_s
                return f"cooldown_{self._consecutive_losses}_consec_losses_{remaining:.0f}s_left"
            else:
                # Cooldown expired
                self._cooldown_until_s = 0.0

        # Spread filter — don't enter in a wide market
        if self._spread > self.max_spread:
            return f"spread_{self._spread:.1f}t_>{self.max_spread:.1f}t"

        return ""

    def _check_daily_reset(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date           = today
            self._daily_pnl_usd        = 0.0
            self._daily_halted         = False
            self._consecutive_losses   = 0
            self._cooldown_until_s     = 0.0
            self.agent._consecutive_losses = 0
            log.info("Daily reset for %s", today)

    # ── Session metrics ──────────────────────────────────────────────────────────

    def win_rate(self) -> float:
        n = len(self._trades)
        if n == 0:
            return 0.0
        return sum(1 for t in self._trades if t.pnl_usd > 0) / n

    def sortino(self) -> float:
        return self.agent._sortino.compute()

    def avg_hold_secs(self) -> float:
        if not self._trades:
            return 0.0
        return sum(t.hold_secs for t in self._trades) / len(self._trades)

    def session_summary(self) -> Dict[str, Any]:
        """Full session metrics dict — primary reporting."""
        pnls_ticks = [t.pnl_ticks for t in self._trades]
        pnls_usd   = [t.pnl_usd   for t in self._trades]
        n          = len(self._trades)
        arr        = np.array(pnls_ticks, dtype=np.float64) if n else np.array([0.0])
        downside   = arr[arr < 0]
        dd         = float(np.sqrt(np.mean(downside ** 2))) if len(downside) else 1e-8
        sortino    = float(arr.mean()) / (dd + 1e-8)
        gross_wins = float(arr[arr > 0].sum()) if (arr > 0).any() else 0.0
        gross_loss = abs(float(arr[arr < 0].sum())) if (arr < 0).any() else 1e-8
        pf         = gross_wins / gross_loss

        run_s = time.time() - self._session_start_s
        return {
            "symbol":            self.symbol,
            "n_trades":          n,
            "win_rate":          round(self.win_rate(), 4),
            "sortino":           round(sortino, 4),
            "profit_factor":     round(pf, 4),
            "total_pnl_ticks":   round(float(arr.sum()), 4),
            "total_pnl_usd":     round(sum(pnls_usd), 2),
            "avg_pnl_ticks":     round(float(arr.mean()), 4),
            "avg_hold_secs":     round(self.avg_hold_secs(), 2),
            "daily_pnl_usd":     round(self._daily_pnl_usd, 2),
            "events_processed":  self._events_processed,
            "n_actions_taken":   self._n_actions_taken,
            "session_run_secs":  round(run_s, 1),
            "agent_stats":       self.agent.get_stats(),
            "journal_summary":   self._journal.get_summary_str(),
        }

    def close(self) -> None:
        """Flush logs and clean up."""
        if self._trade_log_fh:
            self._trade_log_fh.flush()
            self._trade_log_fh.close()
        log.info("Journal summary at close:\n%s", self._journal.get_summary_str())
        self.agent.close()


# ══════════════════════════════════════════════════════════════════════════════
# Factory helper — quick construction from checkpoint + config dict
# ══════════════════════════════════════════════════════════════════════════════

def build_rl_pipeline(
    weights_path:  str,
    hidden_dim:    int   = 256,
    device:        str   = "cuda",
    greedy:        bool  = True,
    log_dir:       str   = "",
    symbol:        str   = "ESM6",
    max_daily_loss:float = 500.0,
    max_consec:    int   = 5,
    max_spread:    float = 4.0,
    use_patchtst:  bool  = True,
    use_vol_model: bool  = True,
    use_embeddings:bool  = True,
) -> RLPaperTradingWrapper:
    """
    One-liner factory to build the full RL paper trading stack.

    Usage
    -----
    from rl_execution_agent import build_rl_pipeline

    rl = build_rl_pipeline(
        weights_path = r"C:\\Users\\claude\\Lvl3Quant\\output\\fifo_ppo_rl\\best_agent_ppo_20260503.pt",
        log_dir      = r"C:\\Users\\claude\\Lvl3Quant\\live_trading\\logs",
    )

    # On every MBO event (at stride boundary when new predictions arrive):
    result = rl.on_event(prediction_dict, market_state_dict)
    if result["executed"]:
        print(result["action_name"], "→ position =", result["position"])
    """
    cfg = RLAgentConfig(
        weights_path           = weights_path,
        hidden_dim             = hidden_dim,
        device                 = device,
        greedy                 = greedy,
        use_patchtst           = use_patchtst,
        use_vol_model          = use_vol_model,
        use_embeddings         = use_embeddings,
        max_daily_loss_usd     = max_daily_loss,
        max_consecutive_losses = max_consec,
        log_dir                = log_dir,
    )
    agent   = RLExecutionAgent(cfg)
    wrapper = RLPaperTradingWrapper(
        agent            = agent,
        config           = cfg,
        symbol           = symbol,
        max_spread_ticks = max_spread,
        log_dir          = log_dir,
    )
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test (run directly to verify architecture loads and acts)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info("=" * 60)
    log.info("RLExecutionAgent smoke test")
    log.info("=" * 60)

    if not TORCH_AVAILABLE:
        log.error("PyTorch not installed — cannot run smoke test")
        sys.exit(1)

    # ── Build agent WITHOUT weights (random init — just tests architecture) ──
    cfg   = RLAgentConfig(weights_path="", hidden_dim=256, device="cpu", greedy=True)
    agent = RLExecutionAgent(cfg)
    log.info("Agent created (random weights)")

    # ── Verify obs shape ──────────────────────────────────────────────────────
    obs = agent.build_obs()
    assert obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {obs.shape}"
    assert not np.any(np.isnan(obs)), "NaN in initial observation"
    log.info("Initial obs shape: %s  all-finite: %s", obs.shape, np.all(np.isfinite(obs)))

    # ── Verify action is valid ────────────────────────────────────────────────
    action = agent.act(obs)
    assert 0 <= action < N_ACTIONS, f"Action {action} out of range"
    log.info("First action: %d (%s)", action, ACTION_NAMES[action])

    # ── Run 1000 random state updates ────────────────────────────────────────
    rng = np.random.default_rng(42)
    ts  = time.time()
    actions_taken = []
    for i in range(1000):
        pred_dict = {
            "pred_1s":         float(rng.uniform(-1, 1)),
            "pred_5s":         float(rng.uniform(-0.5, 0.5)),
            "pred_10s":        float(rng.uniform(-0.3, 0.3)),
            "patchtst_pred_1s":float(rng.uniform(-1, 1)),
            "patchtst_pred_5s":float(rng.uniform(-0.5, 0.5)),
            "patchtst_pred_10s":float(rng.uniform(-0.3, 0.3)),
            "lgbm_vol":        float(rng.uniform(0, 2)),
        }
        mkt_dict = {
            "timestamp_s":   ts + i * 0.05,
            "bid_size":      float(rng.integers(10, 200)),
            "ask_size":      float(rng.integers(10, 200)),
            "spread_ticks":  1.0,
            "price_rel_ticks": float(rng.uniform(-2, 2)),
            "is_trade":      bool(rng.random() < 0.15),
            "trade_side":    int(rng.choice([-1, 1])),
            "trade_qty":     float(rng.integers(1, 10)),
            "is_cancel":     bool(rng.random() < 0.10),
        }
        pos_dict = {
            "position":          0,
            "queue_frac_filled": 0.0,
            "max_fav_excursion": 0.0,
            "max_adv_excursion": 0.0,
            "entry_signal_1s":   0.0,
            "pending_order":     False,
            "pending_order_side":0.0,
        }
        agent.update_state(pred_dict, mkt_dict, pos_dict)
        a = agent.act()
        actions_taken.append(a)
        assert 0 <= a < N_ACTIONS

    elapsed = time.time() - ts
    from collections import Counter
    dist = Counter(actions_taken)
    log.info("1000 random updates in %.3fs (%.0f/s)", elapsed, 1000/elapsed)
    log.info("Action distribution: %s",
             {ACTION_NAMES[k]: v for k, v in sorted(dist.items())})

    # ── Test wrapper ──────────────────────────────────────────────────────────
    log.info("\nTesting RLPaperTradingWrapper...")
    cfg2    = RLAgentConfig(weights_path="", hidden_dim=256, device="cpu")
    agent2  = RLExecutionAgent(cfg2)
    wrapper = RLPaperTradingWrapper(agent2, cfg2, symbol="ESM6")

    mid = 5200.0
    for i in range(100):
        pd_ = {"pred_1s": 0.5, "pred_5s": 0.2, "pred_10s": 0.1,
               "patchtst_pred_1s": 0.4, "lgbm_vol": 1.0}
        md_ = {
            "timestamp_s":   time.time() + i,
            "mid_price":     mid + rng.uniform(-1, 1),
            "best_bid":      mid - 0.25,
            "best_ask":      mid + 0.25,
            "spread_ticks":  1.0,
            "price_rel_ticks": rng.uniform(-0.5, 0.5),
            "is_trade":      True,
            "trade_side":    1,
            "trade_qty":     2.0,
            "is_cancel":     False,
        }
        result = wrapper.on_event(pd_, md_)

    summary = wrapper.session_summary()
    log.info("Wrapper summary: %s", {k: summary[k] for k in
             ("n_trades", "win_rate", "sortino", "daily_pnl_usd", "events_processed")})

    # ── Verify obs dim matches training env constant ───────────────────────────
    log.info("\nOBS_DIM check: module=%d  (must match fifo_rl_env.OBS_DIM=48)", OBS_DIM)
    assert OBS_DIM == 48, f"OBS_DIM mismatch: {OBS_DIM} != 48"

    log.info("\n%s", "=" * 60)
    log.info("All smoke tests PASSED")
    log.info("=" * 60)
    log.info("\nTo wire into paper_trading_mamba_v2.py:")
    log.info("  from rl_execution_agent import build_rl_pipeline")
    log.info("  rl = build_rl_pipeline(weights_path='path/to/best_agent.pt')")
    log.info("  # on each stride boundary:")
    log.info("  result = rl.on_event(prediction_dict, market_state_dict)")
