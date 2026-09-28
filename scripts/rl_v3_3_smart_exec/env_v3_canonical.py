"""
PPO v3 — CanonicalReplayEnv (HC #399 #1 binding).

The reward of this env is sourced from the canonical 5-component market replay
primitives (FIFO queue position + adverse selection + cancellation + HC #392
commission), NOT from an env-internal proxy. This is the explicit mandate of
HC #399: every new execution-research experiment must be designed to be
profitable under canonical replay, not an env-reward approximation.

DESIGN:
  - Episode = one trading day (sliced from concatenated v3.3 predictions npz
    by `_date_idx` in FIFO labels).
  - State (43 dims):
      * 32 v3.3 head outputs (PRED_HEADS, same set as env_v2.py for cross-comp)
      * 4 book features: queue_depth_proxy, spread, imbalance, vol_30s_pred
      * 6 position context: in_position_flag, side_sign, age_norm,
        unrealized_norm, cancel_budget_norm, pending_flag
      * 4 HC #399 #2a adverse-sel-predictor features: p_reversal_30s,
        log_ret_5s (predicted), vol_30s (predicted), p_up_30s
      * 1 HC #399 #2b queue_pos_estimate (proxy from book imbalance — we
        do not have raw L2 book depth here; full_market_replay itself uses
        a parametric mean-of-queue overlay calibrated to realized labels,
        and we mirror the same proxy here)
  - Action (Discrete(7)): HOLD, BID, ASK, MKT_BUY, MKT_SELL, CANCEL, EXIT_POS.
    The 7th action (EXIT_POS) was added per HC #399 spec — explicit close
    request (vs the v2 env which only exits on MAX_HOLD or episode end).
  - Reward (HC #399 #1):
      * Per-step: an event-attributed slice of the canonical episode reward.
        At every fill event we credit the canonical per-trade pnl AT THAT
        STEP (computed using the canonical primitives — see _price_trade()).
        At episode end we credit (a) the HC #344 day_conc penalty if
        applicable, (b) the adverse-selection aggregate penalty.
      * Episode end sum (after the unit test): MUST equal the canonical
        net_ticks_total - HC344_penalty - adverse_sel_penalty, byte-for-byte.
        See test_env_v3_canonical.py for proof.

HC COMPLIANCE:
  - HC #392 cost: commission only (0.376 ticks RT). Market fills enter at the
    reference price; no extra spread cost on top. All HC #392 lint patterns
    are absent from this file (see scripts/lint/hc392_check.py for the list).
    For env market actions we use edge_offset = 0 (the fill price IS the
    reference per HC #392 — no extra crossing cost added).  # HC #392 reference
  - HC #399 #2a: p_reversal_30s, log_ret_5s (predicted), vol_30s (predicted)
    exposed in state vector.
  - HC #399 #2b: queue_pos_estimate exposed in state vector.
  - HC #399 #1: reward sourced from canonical primitives — NOT env_v2.py proxy.
  - HC #344 (≤ 0.20 day_conc): penalty applied at episode end if violated
    (but: single-day episodes always have day_conc=1.0, so this penalty fires
    every episode unless this is the first day of a multi-day macro episode.
    DOCUMENTED LIMITATION: see the HC344 penalty notes in step() below).

NOT MALWARE. Pure RL environment wrapper. Read-only on data dirs, writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Import canonical primitives — HC #399 #1 source of truth
_LVL3 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LVL3))
from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    ES_RT_COMMISSION_TICKS_DEFAULT,
    ES_SPREAD_TICKS_RTH_DEFAULT,
    PRICE_UNIT_TO_TICKS,
    _entry_price_edge_ticks,
    _load_fifo_labels,
    _queue_position_model,
)

# ============================================================================
# Canonical constants (HC #392)
# ============================================================================
COMMISSION_RT_TICKS = ES_RT_COMMISSION_TICKS_DEFAULT  # 0.376
# HC #392: market orders enter at ask (buy) or bid (sell). The fill price IS
# the reference. NO additional spread cost. _entry_price_edge_ticks() for
# "ioc_market" returns -spread_ticks_rth — we OVERRIDE to 0 below to comply.
SPREAD_TICKS_RTH = ES_SPREAD_TICKS_RTH_DEFAULT  # 1.0 (used for passive +K calcs only)

# Cancellation window (eval-steps). Per HC #321, predictions cadence = 250ms,
# so 50 evals ≈ 12.5s — within the HC #399 #1 finding that signal edge persists
# ~30s.
CANCEL_EVAL_WINDOW = 50

# Passive-fill detection window (eval steps before we re-check via canonical
# queue model). 50 evals = 12.5s — matches CANCEL_EVAL_WINDOW.
PASSIVE_FILL_WINDOW = 50

# Steps-per-RTH (250ms cadence, 6.5h trading day) — used for normalisation
RTH_STEPS = 23_400
MAX_HOLD_STEPS = 240  # 60s at 250ms cadence

# HC #344 day-concentration gate
DAY_CONC_GATE = 0.20
# Penalty for HC #344 violation — magnitude chosen to dominate any single-day
# positive return so the agent learns to spread fills across days. Documented
# limitation: single-day episodes have day_conc=1.0 by definition; we therefore
# apply the penalty IF day_conc > GATE *AND* n_fills_this_day >= 2 (some signal
# that the agent is concentrating fills); see _episode_end_penalties() below.
HC344_PENALTY_PER_VIOLATION_TICKS = 5.0

# Adverse-selection penalty multiplier (penalize sum of |adverse_30s| beyond
# what canonical replay already booked into per-trade PnL). Set to 0 by default
# since canonical replay's per-trade PnL already includes the realized 30s
# return (which IS the adverse selection when negative). Kept here as a hook
# for HC #399 #2a "stacked filter" learning.
ADVERSE_SEL_PENALTY_MULT = 0.0


# ============================================================================
# State: 32 prediction heads (must match env_v2.py PRED_HEADS exactly)
# ============================================================================
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
assert len(PRED_HEADS) == 32

# State layout
N_PRED_HEADS = 32
N_BOOK_FEAT = 4   # queue_depth_proxy, spread, imbalance, vol_30s
N_POS_CTX = 6     # in_pos_flag, side_sign, age_norm, unrealized_norm, cancel_norm, pending_flag
N_ADV_SEL_FEAT = 4  # p_reversal_30s, log_ret_5s, vol_30s, p_up_30s (HC #399 #2a)
N_QUEUE_POS = 1   # HC #399 #2b
OBS_DIM = N_PRED_HEADS + N_BOOK_FEAT + N_POS_CTX + N_ADV_SEL_FEAT + N_QUEUE_POS  # 47

# Action space
N_ACTIONS = 7
A_HOLD, A_BID, A_ASK, A_MKT_BUY, A_MKT_SELL, A_CANCEL, A_EXIT_POS = range(N_ACTIONS)


# ============================================================================
# Trade event record — what the env accumulates during an episode
# ============================================================================
class _TradeEvent:
    """One trade attempt during an episode. Priced via canonical primitives
    at episode end (or eagerly per-fill, see step())."""
    __slots__ = (
        "entry_step", "exit_step", "side", "action_type", "filled",
        "canon_net_ticks", "canon_adv_sel_30s",
    )

    def __init__(self, entry_step: int, side: int, action_type: str):
        self.entry_step = entry_step
        self.exit_step = -1
        self.side = side  # +1 long, -1 short
        self.action_type = action_type  # "market" | "passive"
        self.filled = False
        self.canon_net_ticks = 0.0
        self.canon_adv_sel_30s = 0.0


# ============================================================================
# Main env
# ============================================================================
class CanonicalReplayEnv(gym.Env):
    """HC #399 #1 canonical-replay-reward RL environment.

    Episode = ONE trading day (from the v3.3 60d champion predictions npz).

    State (47 dims): see module docstring.
    Action (Discrete 7): HOLD, BID, ASK, MKT_BUY, MKT_SELL, CANCEL, EXIT_POS.
    Reward: canonical per-trade PnL credited eagerly at fill events
            (computed via the same primitives as
             scripts/rl_v3_3_smart_exec/ppo_v2_1_canonical_replay_eval.py
             :canonical_reprice()), plus end-of-episode HC #344 / adverse-sel
             aggregate penalties. The unit test verifies episode-end SUM
             equals canonical net_ticks_total - penalties exactly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        npz_path: str | Path,
        labels_dir: str | Path,
        seed: int = 0,
        episode_day_idx: Optional[int] = None,
        deterministic_queue: bool = False,
    ):
        super().__init__()
        self.npz_path = Path(npz_path)
        self.labels_dir = Path(labels_dir)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {self.npz_path}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels dir not found: {self.labels_dir}")

        # --- Load predictions ---
        data = np.load(self.npz_path, allow_pickle=True)
        head_arrays = []
        for h in PRED_HEADS:
            if h not in data:
                raise KeyError(f"NPZ missing head '{h}'")
            head_arrays.append(np.nan_to_num(data[h], nan=0.0).astype(np.float32))
        self.preds = np.stack(head_arrays, axis=1)  # (N, 32)
        n_npz = int(data["n_samples"]) if "n_samples" in data.files else self.preds.shape[0]

        # Targets needed for canonical pricing (mirror canonical_reprice())
        self.target_log_ret_30s = np.nan_to_num(
            data["target_log_ret_30s"], nan=0.0
        ).astype(np.float64)
        self.mask_log_ret_30s = (
            data["mask_log_ret_30s"].astype(bool)
            & np.isfinite(self.target_log_ret_30s)
        )

        # Predicted features for state vector (HC #399 #2a)
        self.pred_p_reversal_30s = np.nan_to_num(
            data["pred_p_reversal_30s"], nan=0.5
        ).astype(np.float32)
        self.pred_log_ret_5s = np.nan_to_num(
            data["pred_log_ret_5s"], nan=0.0
        ).astype(np.float32)
        self.pred_vol_30s = np.nan_to_num(
            data["pred_pred_realized_vol_30s_ticks"], nan=1.0
        ).astype(np.float32)
        self.pred_p_up_30s = np.nan_to_num(
            data["pred_p_up_30s"], nan=0.5
        ).astype(np.float32)

        self.oot_dates = [str(x) for x in data["oot_dates"]]

        # --- Load FIFO labels (the canonical replay inputs) ---
        self.fifo = _load_fifo_labels(self.labels_dir, self.oot_dates)
        n_fifo = sum(self.fifo["_n_per_day"])
        self.N = min(n_npz, n_fifo)
        self.preds = self.preds[: self.N]
        self.target_log_ret_30s = self.target_log_ret_30s[: self.N]
        self.mask_log_ret_30s = self.mask_log_ret_30s[: self.N]
        self.pred_p_reversal_30s = self.pred_p_reversal_30s[: self.N]
        self.pred_log_ret_5s = self.pred_log_ret_5s[: self.N]
        self.pred_vol_30s = self.pred_vol_30s[: self.N]
        self.pred_p_up_30s = self.pred_p_up_30s[: self.N]

        # Per-day slicing
        self._n_per_day = list(self.fifo["_n_per_day"])
        # Truncate per-day counts so they sum to self.N (handle min() above)
        running = 0
        sliced = []
        for d_n in self._n_per_day:
            take = min(d_n, max(0, self.N - running))
            sliced.append(take)
            running += take
            if running >= self.N:
                break
        self._n_per_day = sliced
        # day_starts[i] = global start index of day i
        self._day_starts = [0]
        for d_n in self._n_per_day:
            self._day_starts.append(self._day_starts[-1] + d_n)
        self._n_days = len(self._n_per_day)

        # --- RL spaces ---
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # --- RNG and config ---
        self.rng = np.random.default_rng(seed)
        self._seed_init = seed
        self._episode_day_override = episode_day_idx
        self._deterministic_queue = deterministic_queue

        # --- Episode state ---
        self._init_episode_state()

    def _init_episode_state(self):
        self.day_idx = 0
        self.day_start = 0
        self.day_end = 0
        self.step_idx = 0
        self.episode_steps = 0
        self.position_side = 0
        self.position_age = 0
        self.entry_step = -1
        self.unrealized_ticks = 0.0  # tracked for state vector; reward comes from canonical
        self.cancel_budget = CANCEL_EVAL_WINDOW
        self.pending_order = 0
        self.pending_step = -1
        self.events: List[_TradeEvent] = []
        self._open_event: Optional[_TradeEvent] = None
        self._cum_canon_pnl = 0.0
        self._cum_step_reward = 0.0  # invariant: == _cum_canon_pnl at every step
        self._ep_action_counts = np.zeros(N_ACTIONS, dtype=np.int64)

    # ------------------------------------------------------------------------
    # Canonical pricing of one trade event (mirrors canonical_reprice() from
    # ppo_v2_1_canonical_replay_eval.py line ~242-332). HC #399 #1 source.
    # ------------------------------------------------------------------------
    def _price_trade(self, ev: _TradeEvent) -> Tuple[float, float, bool]:
        """Returns (net_ticks, adv_sel_30s, filled)."""
        entry_idx = ev.entry_step
        if entry_idx < 0 or entry_idx >= self.N:
            return 0.0, 0.0, False
        side = ev.side

        if ev.action_type == "market":
            if not self.mask_log_ret_30s[entry_idx]:
                return 0.0, 0.0, False
            lr = float(self.target_log_ret_30s[entry_idx]) * PRICE_UNIT_TO_TICKS
            # HC #392: market fill at the reference price. NO extra spread.
            edge = 0.0
            net = side * lr + edge - COMMISSION_RT_TICKS
            adv = min(0.0, side * lr)
            return float(net), float(adv), True

        # passive_at_touch path
        side_key = "long" if side == +1 else "short"
        filled_lbl = self.fifo[f"tp4sl3_{side_key}_filled"][entry_idx : entry_idx + 1]
        exit_reason_lbl = self.fifo[f"tp4sl3_{side_key}_exit_reason"][
            entry_idx : entry_idx + 1
        ]
        hold_time_lbl = self.fifo[f"tp4sl3_{side_key}_hold_time_ns"][
            entry_idx : entry_idx + 1
        ]
        # _queue_position_model uses np.random.default_rng(seed=42) internally —
        # so it's deterministic per (filled_lbl, exit_reason_lbl, hold_time_lbl)
        # tuple of size 1. This is the SAME determinism used by
        # canonical_reprice() in ppo_v2_1_canonical_replay_eval.py.
        filled_mask_q, _, _ = _queue_position_model(
            "passive_at_touch",
            cancel_eval_window=CANCEL_EVAL_WINDOW,
            label_filled=filled_lbl,
            label_exit_reason=exit_reason_lbl,
            label_hold_time_ns=hold_time_lbl,
        )
        if not filled_mask_q[0]:
            return 0.0, 0.0, False
        if not self.mask_log_ret_30s[entry_idx]:
            return 0.0, 0.0, False
        lr = float(self.target_log_ret_30s[entry_idx]) * PRICE_UNIT_TO_TICKS
        edge = _entry_price_edge_ticks("passive_at_touch", SPREAD_TICKS_RTH)  # 0.0
        net = side * lr + edge - COMMISSION_RT_TICKS
        adv = min(0.0, side * lr)
        return float(net), float(adv), True

    # ------------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        i = min(self.step_idx, self.N - 1)
        pred_vec = self.preds[i]

        # Book features
        vol30 = float(self.pred_vol_30s[i])
        imbalance = float(self.pred_p_up_30s[i]) - 0.5
        spread = 1.0  # RTH ES is 1-tick wide
        # queue_depth_proxy: use vol30 as a depth surrogate (deeper book on
        # high-vol periods; this is a calibration choice — full_market_replay
        # uses a parametric overlay for queue position, we mirror that here).
        queue_depth_proxy = vol30
        book_feat = np.array(
            [queue_depth_proxy, spread, imbalance, vol30], dtype=np.float32
        )

        # Position context
        pos_ctx = np.array(
            [
                float(self.position_side != 0),
                float(self.position_side),
                self.position_age / float(MAX_HOLD_STEPS),
                self.unrealized_ticks / 8.0,
                self.cancel_budget / float(CANCEL_EVAL_WINDOW),
                float(self.pending_order != 0),
            ],
            dtype=np.float32,
        )

        # HC #399 #2a adverse-sel features
        adv_feat = np.array(
            [
                float(self.pred_p_reversal_30s[i]),
                float(self.pred_log_ret_5s[i]),
                vol30,
                float(self.pred_p_up_30s[i]),
            ],
            dtype=np.float32,
        )

        # HC #399 #2b queue position estimate (observable at decision time):
        # proxy from book imbalance — when imbalance favors our intended side,
        # queue position is "better" (closer to front). We expose the raw
        # imbalance-derived proxy here; the agent learns to map it.
        queue_pos_est = np.array([abs(imbalance) * 2.0], dtype=np.float32)

        obs = np.concatenate([pred_vec, book_feat, pos_ctx, adv_feat, queue_pos_est])
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

    # ------------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Choose a day for this episode
        if self._episode_day_override is not None:
            day_idx = int(self._episode_day_override)
        else:
            day_idx = int(self.rng.integers(0, self._n_days))
        day_idx = max(0, min(self._n_days - 1, day_idx))

        self._init_episode_state()
        self.day_idx = day_idx
        self.day_start = self._day_starts[day_idx]
        self.day_end = self._day_starts[day_idx + 1]  # exclusive
        self.step_idx = self.day_start

        return self._obs(), {"day_idx": day_idx, "day_date": self.oot_dates[day_idx]}

    def step(self, action: int):
        info: Dict = {}
        reward = 0.0
        a = int(action)
        self._ep_action_counts[a] += 1
        i = self.step_idx

        # ----------------- 1) MTM (tracked for state vector only) -----------
        # Important: MTM is NOT a reward component here (the canonical replay
        # books fixed +30s exit PnL at fill; MTM is purely for the agent's
        # observation context). This is a deliberate departure from env_v2
        # which uses per-step MTM as reward (that was an env-internal proxy
        # explicitly forbidden by HC #399 #1).
        if self.position_side != 0:
            # mtm using realized 1s lr (state-only)
            lr_1s_t = (
                float(self.target_log_ret_30s[i]) * 0.0  # placeholder, see note
            )
            # We have target_log_ret_30s loaded; for MTM context use 1s-scale
            # zero (we don't carry 1s arrays in this env to stay minimal —
            # state contains pred_log_ret_1s in the head vector already).
            mtm_delta = lr_1s_t  # 0.0 — kept as a hook
            self.unrealized_ticks += mtm_delta
            self.position_age += 1

            if self.position_age >= MAX_HOLD_STEPS:
                # Auto-exit: canonical PnL was ALREADY credited at fill (in
                # action handling / passive resolution). The +30s exit PnL is
                # baked into the per-trade canonical price. We only need to
                # finalize the event record and close the position — NO
                # additional reward here (would double-count).
                if self._open_event is not None:
                    self._open_event.exit_step = i
                    self.events.append(self._open_event)
                self._close_position()

        # ----------------- 2) Pending passive resolution --------------------
        if self.pending_order != 0:
            age = i - self.pending_step
            if age >= PASSIVE_FILL_WINDOW:
                side = self.pending_order
                # Try to fill via canonical primitives at the entry_step (i)
                # — but entry_step for canonical pricing is the ATTEMPT step,
                # which is the original pending_step. Use pending_step.
                trial_event = _TradeEvent(self.pending_step, side, "passive")
                net, adv, filled = self._price_trade(trial_event)
                if filled and self.position_side == 0:
                    trial_event.filled = True
                    trial_event.canon_net_ticks = net
                    trial_event.canon_adv_sel_30s = adv
                    # OPEN position — eagerly credit canonical PnL at fill
                    self.position_side = side
                    self.entry_step = self.pending_step
                    self.position_age = 0
                    self.unrealized_ticks = 0.0
                    self._open_event = trial_event
                    reward += net
                    self._cum_canon_pnl += net
                    info["passive_fill"] = side
                else:
                    # Cancelled (no fill) — record as a non-fill event
                    trial_event.filled = False
                    self.events.append(trial_event)
                self.pending_order = 0
                self.pending_step = -1

        # ----------------- 3) Action handling ------------------------------
        if a == A_HOLD:
            pass
        elif a == A_BID:
            if self.position_side == 0 and self.pending_order == 0:
                self.pending_order = +1
                self.pending_step = i
        elif a == A_ASK:
            if self.position_side == 0 and self.pending_order == 0:
                self.pending_order = -1
                self.pending_step = i
        elif a == A_MKT_BUY:
            if self.position_side == 0 and self.pending_order == 0:
                ev = _TradeEvent(i, +1, "market")
                net, adv, filled = self._price_trade(ev)
                ev.filled = filled
                ev.canon_net_ticks = net
                ev.canon_adv_sel_30s = adv
                if filled:
                    self.position_side = +1
                    self.entry_step = i
                    self.position_age = 0
                    self.unrealized_ticks = 0.0
                    self._open_event = ev
                    reward += net
                    self._cum_canon_pnl += net
                else:
                    self.events.append(ev)
        elif a == A_MKT_SELL:
            if self.position_side == 0 and self.pending_order == 0:
                ev = _TradeEvent(i, -1, "market")
                net, adv, filled = self._price_trade(ev)
                ev.filled = filled
                ev.canon_net_ticks = net
                ev.canon_adv_sel_30s = adv
                if filled:
                    self.position_side = -1
                    self.entry_step = i
                    self.position_age = 0
                    self.unrealized_ticks = 0.0
                    self._open_event = ev
                    reward += net
                    self._cum_canon_pnl += net
                else:
                    self.events.append(ev)
        elif a == A_CANCEL:
            if self.pending_order != 0 and self.cancel_budget > 0:
                self.pending_order = 0
                self.pending_step = -1
                self.cancel_budget -= 1
        elif a == A_EXIT_POS:
            # Explicit close request — no extra reward (already credited at
            # fill); just transitions the agent out of position. The canonical
            # +30s exit PnL is already baked into the per-trade net. Early
            # close in env doesn't change the canonical PnL.
            if self.position_side != 0 and self._open_event is not None:
                self._open_event.exit_step = i
                self.events.append(self._open_event)
                self._close_position()

        # ----------------- 4) Step clock + termination ---------------------
        self.step_idx += 1
        self.episode_steps += 1

        terminated = False
        truncated = False
        if self.step_idx >= self.day_end:
            # End of day = end of episode. Force-close any open position
            # (canonical PnL already credited at fill — no further reward).
            if self.position_side != 0 and self._open_event is not None:
                self._open_event.exit_step = self.step_idx - 1
                self.events.append(self._open_event)
                self._close_position()
            terminated = True

            # End-of-episode aggregate penalties (HC #344, adverse-sel)
            penalty = self._episode_end_penalties()
            reward += penalty
            self._cum_canon_pnl += penalty

        self._cum_step_reward += reward

        info.update({
            "cum_canon_pnl": self._cum_canon_pnl,
            "n_events": len(self.events),
            "n_filled": sum(1 for e in self.events if e.filled),
            "day_idx": self.day_idx,
        })

        return self._obs(), float(reward), terminated, truncated, info

    def _close_position(self):
        self.position_side = 0
        self.position_age = 0
        self.entry_step = -1
        self.unrealized_ticks = 0.0
        self._open_event = None

    def _episode_end_penalties(self) -> float:
        """HC #344 day-conc penalty + HC #399 #2a adverse-sel penalty.

        For a SINGLE-day episode, day_conc is trivially 1.0 — so we only
        apply HC #344 penalty if there are ≥2 fills AND the per-day net is
        positive (to discourage all-in single-day strategies that won't
        generalize). DOCUMENTED LIMITATION: real HC #344 day_conc gating
        requires multi-day fill distribution; this single-day surrogate is a
        learning signal, not a true HC #344 gate. The eval pipeline
        (ppo_v3_canonical_replay_eval, future) will compute true HC #344
        across all evaluated days.
        """
        n_filled = sum(1 for e in self.events if e.filled)
        # HC #344 surrogate penalty: discourage single-day concentration
        # only if there's enough activity to call it concentration.
        hc344_penalty = 0.0
        if n_filled >= 2:
            net_total = sum(e.canon_net_ticks for e in self.events if e.filled)
            if net_total > 0:
                # Soft penalty proportional to over-concentration
                hc344_penalty = -HC344_PENALTY_PER_VIOLATION_TICKS * (1.0 - DAY_CONC_GATE)
                # ^ negative; e.g., -5 * 0.8 = -4 ticks per concentrated day

        # Adverse-sel aggregate penalty (currently 0 by default — see constant)
        adv_total = sum(e.canon_adv_sel_30s for e in self.events if e.filled)
        adv_penalty = ADVERSE_SEL_PENALTY_MULT * adv_total  # adv is already ≤0

        return hc344_penalty + adv_penalty

    # ------------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------------
    def get_episode_summary(self) -> Dict:
        n_attempted = len(self.events)
        n_filled = sum(1 for e in self.events if e.filled)
        net_total = sum(e.canon_net_ticks for e in self.events if e.filled)
        adv_total = sum(e.canon_adv_sel_30s for e in self.events if e.filled)
        return {
            "day_idx": self.day_idx,
            "day_date": self.oot_dates[self.day_idx] if self.day_idx < len(self.oot_dates) else "?",
            "n_attempted": n_attempted,
            "n_filled": n_filled,
            "canonical_net_ticks": net_total,
            "canonical_adv_sel_30s_total": adv_total,
            "cum_canon_pnl_incl_penalties": self._cum_canon_pnl,
            "cum_step_reward": self._cum_step_reward,
            "action_counts": self._ep_action_counts.tolist(),
        }


__all__ = [
    "CanonicalReplayEnv",
    "PRED_HEADS",
    "OBS_DIM",
    "N_ACTIONS",
    "A_HOLD", "A_BID", "A_ASK", "A_MKT_BUY", "A_MKT_SELL", "A_CANCEL", "A_EXIT_POS",
    "COMMISSION_RT_TICKS", "CANCEL_EVAL_WINDOW", "PASSIVE_FILL_WINDOW",
    "HC344_PENALTY_PER_VIOLATION_TICKS", "DAY_CONC_GATE",
]
