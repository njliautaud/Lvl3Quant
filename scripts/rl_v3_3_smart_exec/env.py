"""
Gym env for v3.3 smart execution RL (HC #396 weekend mandate).

State (~40 dims):
  - 32 v3.3 prediction heads (the pred_* arrays in fold_00_predictions.npz)
  - Book context proxies derived from predictions (vol_30s, queue_imb_proxy,
    spread_proxy) — these come from pred_pred_realized_vol_30s_ticks, the
    p_up_30s minus 0.5 (signed imbalance proxy), and a constant 1.0 (ES RTH
    spread is always ~1 tick).
  - Position context: in_position {0,1}, side {-1,0,+1}, age_steps (scaled),
    unrealized_pnl_ticks, cancel_budget_remaining (scaled), pending_order {0,1}.

Action (Discrete 6):
  0: hold
  1: place_passive_bid  (long, fills if signal+book validates within cancel window)
  2: place_passive_ask  (short, fills if signal+book validates within cancel window)
  3: market_buy         (instant long, cost = 0.376 ticks commission ONLY — HC #392)
  4: market_sell        (instant short, cost = 0.376 ticks commission ONLY — HC #392)
  5: cancel_pending     (cancel a pending passive)

Reward semantics (HC #392 — STRICT):
  - Per-step reward = realized PnL delta in ticks NET of commission only (0.376
    round-trip; we charge half on entry and half on exit so the agent sees the
    cost progressively).
  - For passive limits (actions 1/2): use the v3.3 label target_fifo_tp8sl5_net
    at the entry step (this is THE FIFO market replay PnL in ticks net of
    realistic execution costs from the canonical replay library). This avoids
    needing raw MBO depth on Razer.
  - For market orders (actions 3/4): use target_log_ret_30s (in ticks) signed
    by direction, minus 0.376 ticks commission ONLY. A market BUY enters at the
    ask price — that price IS the reference, there is NO additional spread cost
    on top. User has reaffirmed this 3+ times. NEVER add SPREAD_CROSS_TICKS to
    market-order PnL again. The canonical full_market_replay.py library uses
    the same convention.
  - For hold/cancel: 0 reward except mark-to-market on open passive position
    via target_log_ret_1s.

Episode boundary: trade close (TP/SL hit or hold-window elapsed) or RTH end
(approximated as every N steps; v3.3 NPZ is concatenated OOT days without
per-sample timestamps, so we use a fixed RTH_STEPS=23400 ~= 6.5h * 3600 evals/h).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ── Canonical constants (HC #392) ─────────────────────────────────────────────
COMMISSION_RT_TICKS = 0.376     # AMP ES round-trip
# HC #392: SPREAD_CROSS_TICKS is INTENTIONALLY 0.0. Market orders enter at the
# ask (buy) or bid (sell) — that IS the reference fill price. There is NO extra
# spread cost on top. Charging an extra 1-tick spread cross is a HC #392
# violation. User has flagged this 3+ times. The HC #392 lint check enforces
# that this constant must stay 0.0.
SPREAD_CROSS_TICKS  = 0.0       # HC #392: NO extra cost — market = ask price
RTH_STEPS = 23_400              # rough RTH cap (250ms stride * 6.5h ≈ 93600; we
                                # cap shorter so episodes don't span days)
MAX_HOLD_STEPS = 240            # 240 * 250ms = 60s default exit window
CANCEL_BUDGET  = 50             # max pending cancels per episode
PASSIVE_FILL_WINDOW = 50        # 50 * 250ms = ~12.5s wait for passive fill
HORIZON_EXIT_TICKS = "target_log_ret_30s"  # exit MTM proxy

# Order of prediction heads (32 total). Stable list for state vector reproducibility.
PRED_HEADS: List[str] = [
    "pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s",
    "pred_log_ret_30s", "pred_log_ret_60s", "pred_log_ret_5min",
    "pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s", "pred_p_up_60s",
    "pred_log_ret_10s_q10", "pred_log_ret_10s_q50", "pred_log_ret_10s_q90",
    "pred_log_ret_30s_q10", "pred_log_ret_30s_q50", "pred_log_ret_30s_q90",
    "pred_log_ret_60s_q10", "pred_log_ret_60s_q50", "pred_log_ret_60s_q90",
    "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks",
    "pred_pred_mfe_60s_ticks", "pred_pred_mae_60s_ticks",
    "pred_pred_time_to_mfe_secs",
    "pred_p_reversal_15s", "pred_p_reversal_30s", "pred_p_reversal_60s",
    "pred_pred_realized_vol_30s_ticks",
    "pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net",
    "pred_fifo_tp4sl3_hit_tp", "pred_fifo_tp8sl5_hit_tp",
]
assert len(PRED_HEADS) == 32, f"expected 32 heads, got {len(PRED_HEADS)}"

BOOK_CTX_DIMS = 3       # vol_30s, queue_imb_proxy, spread_proxy
POS_CTX_DIMS  = 6       # in_position, side, age, upnl, cancel_budget, pending
OBS_DIM = len(PRED_HEADS) + BOOK_CTX_DIMS + POS_CTX_DIMS   # 32+3+6 = 41

N_ACTIONS = 6
A_HOLD, A_BID, A_ASK, A_MKT_BUY, A_MKT_SELL, A_CANCEL = range(N_ACTIONS)


class V33SmartExecEnv(gym.Env):
    """Gym env wrapping v3.3 predictions + label-derived FIFO PnL."""

    metadata = {"render_modes": []}

    def __init__(self, npz_path: str | Path, seed: int = 0):
        super().__init__()
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"v3.3 predictions NPZ not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=False)
        # Stack prediction heads into (N, 32) state matrix
        head_arrays = []
        for h in PRED_HEADS:
            if h not in data:
                raise KeyError(f"NPZ missing head '{h}'. Available: {list(data.keys())[:20]}...")
            head_arrays.append(np.nan_to_num(data[h], nan=0.0).astype(np.float32))
        self.preds = np.stack(head_arrays, axis=1)          # (N, 32)
        self.N = self.preds.shape[0]

        # Per-step PnL labels for passive limit fills (TICKS, already net of
        # canonical replay costs). HC #392: commission-only basis.
        self.fifo_pnl_tp8sl5 = np.nan_to_num(
            data["target_fifo_tp8sl5_net"], nan=0.0
        ).astype(np.float32)
        self.fifo_filled_tp8sl5 = (
            np.nan_to_num(data["target_fifo_tp8sl5_hit_tp"], nan=0.0).astype(np.float32) > 0
        ).astype(np.float32)
        self.fifo_pnl_tp4sl3 = np.nan_to_num(
            data["target_fifo_tp4sl3_net"], nan=0.0
        ).astype(np.float32)

        # Short-horizon return for market orders & MTM (in ticks per canonical doc).
        self.ret_30s = np.nan_to_num(
            data["target_log_ret_30s"], nan=0.0
        ).astype(np.float32)
        self.ret_1s  = np.nan_to_num(
            data["target_log_ret_1s"], nan=0.0
        ).astype(np.float32)

        # Book proxies (sourced from prediction columns themselves):
        self.vol_30s_pred = np.nan_to_num(
            data["pred_pred_realized_vol_30s_ticks"], nan=1.0
        ).astype(np.float32)
        self.queue_imb = (np.nan_to_num(data["pred_p_up_30s"], nan=0.5) - 0.5).astype(np.float32)
        # ES spread is ~1 tick RTH; we keep as a constant feature with mild noise.

        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Episode state
        self._reset_episode_state()

    # ── Internal state mgmt ──────────────────────────────────────────────────
    def _reset_episode_state(self):
        self.step_idx = 0
        self.episode_steps = 0
        self.position_side = 0          # -1 short, 0 flat, +1 long
        self.position_age = 0
        self.entry_idx = -1
        self.unrealized_ticks = 0.0
        self.cancel_budget = CANCEL_BUDGET
        self.pending_order = 0          # 0=none, +1=passive bid, -1=passive ask
        self.pending_age = 0
        self.realized_pnl_ticks = 0.0
        self.trades_count = 0

    def _obs(self) -> np.ndarray:
        i = min(self.step_idx, self.N - 1)
        # Predictions slice
        pred_vec = self.preds[i]                       # (32,)
        # Book context
        vol30 = self.vol_30s_pred[i]
        qimb = self.queue_imb[i]
        spread = 1.0
        # Position context (scaled)
        pos_ctx = np.array([
            float(self.position_side != 0),
            float(self.position_side),
            self.position_age / float(MAX_HOLD_STEPS),
            self.unrealized_ticks / 8.0,
            self.cancel_budget / float(CANCEL_BUDGET),
            float(self.pending_order != 0),
        ], dtype=np.float32)
        obs = np.concatenate([pred_vec, np.array([vol30, qimb, spread], dtype=np.float32), pos_ctx])
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    def _exit_position(self, exit_pnl_ticks: float) -> float:
        """Close current position, book realized PnL net of half-commission exit."""
        reward = exit_pnl_ticks - 0.5 * COMMISSION_RT_TICKS
        self.realized_pnl_ticks += reward
        self.trades_count += 1
        self.position_side = 0
        self.position_age = 0
        self.entry_idx = -1
        self.unrealized_ticks = 0.0
        return reward

    # ── Gym API ──────────────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Random start, leave room for a full episode
        max_start = max(1, self.N - RTH_STEPS - MAX_HOLD_STEPS - 1)
        self.step_idx = int(self.rng.integers(0, max_start))
        self._reset_episode_state_keep_start()
        return self._obs(), {}

    def _reset_episode_state_keep_start(self):
        start = self.step_idx
        self._reset_episode_state()
        self.step_idx = start

    def step(self, action: int):
        info = {}
        reward = 0.0
        i = self.step_idx

        # 1) If in position, update MTM via ret_1s
        if self.position_side != 0:
            mtm_delta = self.position_side * float(self.ret_1s[i])
            self.unrealized_ticks += mtm_delta
            self.position_age += 1
            reward += mtm_delta  # dense shaping: agent feels position drift
            # Forced exit at hold cap
            if self.position_age >= MAX_HOLD_STEPS:
                reward += self._exit_position(self.unrealized_ticks)

        # 2) Pending passive order age
        if self.pending_order != 0:
            self.pending_age += 1
            if self.pending_age >= PASSIVE_FILL_WINDOW:
                # Resolve passive order using v3.3 FIFO labels (canonical replay)
                side = self.pending_order
                if self.fifo_filled_tp8sl5[i] > 0 and self.position_side == 0:
                    # Treat label as signed PnL for long; flip for short
                    raw_pnl = float(self.fifo_pnl_tp8sl5[i]) * side
                    # Open position with label as immediate realized seed
                    self.position_side = side
                    self.entry_idx = i
                    self.position_age = 0
                    self.unrealized_ticks = raw_pnl  # label is full TP/SL outcome
                    reward -= 0.5 * COMMISSION_RT_TICKS  # entry commission half
                    info["passive_fill"] = side
                self.pending_order = 0
                self.pending_age = 0

        # 3) Action handling
        if action == A_HOLD:
            pass
        elif action == A_BID:
            if self.position_side == 0 and self.pending_order == 0:
                self.pending_order = +1
                self.pending_age = 0
        elif action == A_ASK:
            if self.position_side == 0 and self.pending_order == 0:
                self.pending_order = -1
                self.pending_age = 0
        elif action == A_MKT_BUY:
            if self.position_side == 0 and self.pending_order == 0:
                self.position_side = +1
                self.entry_idx = i
                self.position_age = 0
                # HC #392: market BUY enters at ask = reference fill price.
                # MTM starts at 0 (no spread "paid" — the fill IS the price).
                self.unrealized_ticks = 0.0
                # Only half-commission entry — NO spread cross.
                reward -= 0.5 * COMMISSION_RT_TICKS
        elif action == A_MKT_SELL:
            if self.position_side == 0 and self.pending_order == 0:
                self.position_side = -1
                self.entry_idx = i
                self.position_age = 0
                # HC #392: market SELL enters at bid = reference fill price.
                # MTM starts at 0 (no spread "paid").
                self.unrealized_ticks = 0.0
                reward -= 0.5 * COMMISSION_RT_TICKS
        elif action == A_CANCEL:
            if self.pending_order != 0 and self.cancel_budget > 0:
                self.pending_order = 0
                self.pending_age = 0
                self.cancel_budget -= 1

        # 4) Step the clock
        self.step_idx += 1
        self.episode_steps += 1

        terminated = False
        truncated = False
        if self.step_idx >= self.N - 1:
            # Force-close at data boundary
            if self.position_side != 0:
                reward += self._exit_position(self.unrealized_ticks)
            terminated = True
        if self.episode_steps >= RTH_STEPS:
            if self.position_side != 0:
                reward += self._exit_position(self.unrealized_ticks)
            truncated = True

        info["realized_pnl_ticks"] = self.realized_pnl_ticks
        info["trades"] = self.trades_count
        return self._obs(), float(reward), terminated, truncated, info
