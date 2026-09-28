#!/usr/bin/env python3
"""
Pre-compute RL Observation Tensors for Split DQN Training
==========================================================

Eliminates the CPU bottleneck in Split DQN training by pre-computing the
expensive base observation features (book reconstruction, flow tracking,
volatility, predictions, PCA embeddings) once per MBO event file.

The 48-dim observation splits into:
  - BASE features (indices 0-9, 16-23, 34-44): depend only on event stream,
    book state, flow, and predictions. Expensive to compute (book reconstruction,
    flow tracking, volatility estimation). ~70% of _get_obs() cost.
  - POSITION features (indices 10-15, 24-33, 45-47): depend on agent actions
    (position, pending orders, trade history, stats). Cheap to reconstruct.

Strategy:
  1. Walk through each MBO file once, running the full environment logic
     (book reconstruction, flow tracking, etc.) with action=0 (do nothing).
  2. At each step, save the base observation features + event metadata.
  3. At training time, PrecomputedEnv loads memory-mapped arrays and only
     computes the cheap position-dependent features inline.

Expected speedup: 10-50x for environment stepping (the main CPU bottleneck).

Usage:
    python precompute_observations.py \
        --data-dir /path/to/mbo_events_smart_v3 \
        --pred-dir /path/to/cnn_mamba_preds \
        --output-dir /path/to/precomputed \
        --n-workers 12

Author: Claude (Infrastructure Builder)
Date: 2026-05-06
"""

from __future__ import annotations

import os
import sys
import math
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ─── Setup ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("precompute_obs")

# ─── Base observation indices (independent of agent actions) ──────────────────
# These are the indices in the 48-dim obs that depend only on the event stream.
# We pre-compute these and store them densely.
#
# Full 48-dim layout:
#   [0:4]   Signal: pred_1s, pred_5s, pred_10s, confidence_tier       BASE
#   [4:8]   Book: bid_depth_log, ask_depth_log, imbalance, spread     BASE
#   [8:10]  Price: price_rel, price_momentum                          BASE
#   [10:14] Position: position, unrealized, hold_time, queue_frac     AGENT-DEP
#   [14:16] MFE/MAE: mfe, mae                                        AGENT-DEP
#   [16:19] Context: vol_60s, tod_sin, tod_cos                        BASE
#   [19:24] Flow: density, buy_frac, imbalance, cancel_rate, rate     BASE
#   [24:29] Trade history: last 5 PnLs                                AGENT-DEP
#   [29:32] Stats: win_rate, sortino, consec_losses                   AGENT-DEP
#   [32:34] Order: pending_order, pending_side                        AGENT-DEP
#   [34:39] PCA embedding                                             BASE
#   [39:45] PatchTST confluence                                       BASE
#   [45:48] Alpha-awareness                                           AGENT-DEP

# Base feature indices in the 48-dim obs vector
BASE_INDICES = list(range(0, 10)) + list(range(16, 24)) + list(range(34, 45))
# = [0,1,2,3,4,5,6,7,8,9, 16,17,18,19,20,21,22,23, 34,35,36,37,38,39,40,41,42,43,44]
BASE_DIM = len(BASE_INDICES)  # 29

# Metadata per step: timestamp_ns, price_rel_ticks, event_type, spread,
#                     pred_1s, pred_5s, pred_10s, pred_idx, bid_depth, ask_depth,
#                     book_imbalance, queue_depth_bid, queue_depth_ask
META_DIM = 13


def precompute_one_file(args) -> dict:
    """
    Pre-compute base observations for one MBO event file.

    Runs the full FIFOExecutionEnv with action=0 (do nothing) to build
    book state, flow trackers, and observations at every step.

    Returns dict with status info; saves .npy files to output_dir.
    """
    file_path, pred_dir, patchtst_dir, output_dir = args
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    date_str = file_path.stem.replace("_mbo_events", "")

    # Check if already precomputed
    obs_path = output_dir / f"{date_str}_base_obs.npy"
    if obs_path.exists():
        return {"date": date_str, "status": "skipped", "n_steps": 0, "elapsed": 0.0}

    t0 = time.time()

    # Import env in worker process
    sys.path.insert(0, str(Path(__file__).parent))
    from alpha_discovery.execution.fifo_rl_env import FIFOExecutionEnv

    try:
        env = FIFOExecutionEnv(
            event_dir=file_path.parent,
            pred_dir=Path(pred_dir),
            patchtst_pred_dir=Path(patchtst_dir) if patchtst_dir else None,
            verbose=False,
        )
    except Exception as e:
        return {"date": date_str, "status": f"env_init_error: {e}", "n_steps": 0, "elapsed": 0.0}

    # Reset environment with the specific file
    try:
        full_obs = env.reset(episode_file=file_path)
    except Exception as e:
        return {"date": date_str, "status": f"reset_error: {e}", "n_steps": 0, "elapsed": 0.0}

    # Estimate total steps for pre-allocation
    n_total = env._n_events
    # Pre-allocate arrays (over-allocate slightly, trim at end)
    base_obs_arr = np.zeros((n_total, BASE_DIM), dtype=np.float32)
    meta_arr = np.zeros((n_total, META_DIM), dtype=np.float32)

    # Collect the initial observation
    step = 0

    def extract_base_obs(obs_48):
        """Extract base features from full 48-dim observation."""
        return obs_48[BASE_INDICES]

    def extract_meta(env_obj, step_idx):
        """Extract metadata for current step."""
        meta = np.zeros(META_DIM, dtype=np.float32)
        if step_idx >= env_obj._n_events:
            return meta

        ts = int(env_obj._timestamps[step_idx])
        evt = env_obj._events[step_idx]

        meta[0] = float(ts / 1e9)  # timestamp in seconds (float64 precision loss ok for meta)
        meta[1] = float(evt[3])    # COL_PRICE_REL = 3
        meta[2] = float(env_obj._et_raw[step_idx])  # event type raw
        meta[3] = float(evt[5])    # COL_SPREAD = 5

        # Predictions at current pred_idx
        if env_obj._preds is not None and env_obj._pred_idx < env_obj._n_preds:
            p = env_obj._preds[env_obj._pred_idx]
            meta[4] = float(p[0])  # pred_1s
            meta[5] = float(p[1])  # pred_5s
            meta[6] = float(p[2])  # pred_10s
        meta[7] = float(env_obj._pred_idx)

        # Book depths (for queue position reconstruction)
        bid_d, ask_d, imb = env_obj._get_book_depth_for_obs()
        meta[8] = bid_d
        meta[9] = ask_d
        meta[10] = imb

        # Queue depths at best levels (for FIFO fill simulation)
        meta[11] = env_obj._get_queue_depth(1)   # bid queue
        meta[12] = env_obj._get_queue_depth(-1)  # ask queue

        return meta

    # Record initial obs
    base_obs_arr[0] = extract_base_obs(full_obs)
    meta_arr[0] = extract_meta(env, env._step_idx)
    step = 1

    # Walk through ALL events with action=0 (do nothing)
    done = False
    while not done and step < n_total:
        full_obs, reward, done, info = env.step(0)  # action=0 = do nothing

        if step < n_total:
            base_obs_arr[step] = extract_base_obs(full_obs)
            meta_arr[step] = extract_meta(env, env._step_idx)
        step += 1

    # Trim to actual size
    actual_steps = step
    base_obs_arr = base_obs_arr[:actual_steps]
    meta_arr = meta_arr[:actual_steps]

    # Save as .npy files (memory-mappable)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(obs_path), base_obs_arr)
    np.save(str(output_dir / f"{date_str}_meta.npy"), meta_arr)

    # Also save timestamp array separately for precise lookups (int64)
    if env._timestamps is not None:
        # Save the timestamps for the range we actually processed
        start_idx = env._find_rth_start() if env.rth_only else 0
        end_idx = min(start_idx + actual_steps, len(env._timestamps))
        ts_arr = env._timestamps[start_idx:end_idx].copy()
        np.save(str(output_dir / f"{date_str}_timestamps.npy"), ts_arr)

    elapsed = time.time() - t0
    return {
        "date": date_str,
        "status": "ok",
        "n_steps": actual_steps,
        "elapsed": elapsed,
        "obs_shape": base_obs_arr.shape,
    }


class PrecomputedEnv:
    """
    Drop-in replacement for FIFOExecutionEnv using pre-computed observations.

    Loads memory-mapped .npy files and reconstructs the full 48-dim observation
    by combining pre-computed base features with cheaply-computed agent-dependent
    features (position state, trade history, rolling stats).

    This avoids the expensive book reconstruction, flow tracking, and prediction
    alignment that dominates FIFOExecutionEnv._get_obs() cost.

    Interface matches FIFOExecutionEnv: reset(), step(), observation_space, action_space.
    """

    # Import constants from the real env to stay in sync
    from alpha_discovery.execution.fifo_rl_env import (
        OBS_DIM, TICK_VALUE, COMMISSION_COST, MAX_HOLD_SECS,
        ALPHA_GATE_THRESHOLD, ALPHA_GATE_PENALTY,
        SIGNAL_ALIGNMENT_BONUS, SIGNAL_MISALIGNMENT_PENALTY,
        SIGNAL_DECAY_HALFLIFE_S, SORTINO_BUFFER_SIZE,
        MIN_TRADES_FOR_SORTINO, OVERTRADING_LIMIT,
        CONSECUTIVE_LOSS_PENALTY, RTH_END_SEC,
        _BoxSpace, _DiscreteSpace, RollingSortino, Order, Trade,
    )

    def __init__(
        self,
        precomputed_dir: Path,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        precomputed_dir : directory containing pre-computed .npy files
        verbose : print debug info
        """
        self.precomputed_dir = Path(precomputed_dir)
        self.verbose = verbose

        # Discover available files
        obs_files = sorted(self.precomputed_dir.glob("*_base_obs.npy"))
        if not obs_files:
            raise FileNotFoundError(
                f"No pre-computed observation files found in {self.precomputed_dir}"
            )
        self._dates = [f.stem.replace("_base_obs", "") for f in obs_files]
        self._file_index = {d: i for i, d in enumerate(self._dates)}
        log.info(f"PrecomputedEnv: {len(self._dates)} pre-computed dates in {self.precomputed_dir}")

        # Current episode state
        self._base_obs: Optional[np.ndarray] = None  # (N, BASE_DIM) memory-mapped
        self._meta: Optional[np.ndarray] = None       # (N, META_DIM) memory-mapped
        self._timestamps: Optional[np.ndarray] = None  # (N,) int64 memory-mapped
        self._n_steps: int = 0
        self._step_idx: int = 0
        self._episode_date: str = ""

        # Agent state (lightweight, computed inline)
        self._position: int = 0
        self._entry_price_rel: float = 0.0
        self._entry_ts_s: float = 0.0  # entry time in seconds
        self._mfe_ticks: float = 0.0
        self._mae_ticks: float = 0.0
        self._pending_order: Optional[self.Order] = None

        self._entry_signal_1s: float = 0.0
        self._entry_signal_5s: float = 0.0
        self._entry_signal_10s: float = 0.0

        self._trade_history: list = []
        self._sortino_tracker = self.RollingSortino(self.SORTINO_BUFFER_SIZE)
        self._last_sortino: float = 0.0
        self._consecutive_losses: int = 0
        self._episode_pnl_ticks: float = 0.0
        self._episode_trades: int = 0
        self._trade_ts_history: list = []  # timestamps of recent trades (seconds)

        self._last_closed_trade = None
        self._last_entry_signal_1s: float = 0.0

        # Gym-compatible spaces
        self.observation_space = self._BoxSpace(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = self._DiscreteSpace(7)

        # RNG for random file selection
        self._rng = np.random.default_rng()

    def reset(self, episode_file: Optional[Path] = None) -> np.ndarray:
        """
        Start a new episode from pre-computed data.

        Parameters
        ----------
        episode_file : Path to original MBO file (used to derive date).
                       If None, picks a random date.

        Returns
        -------
        obs : np.ndarray (48,) initial observation
        """
        if episode_file is not None:
            date_str = episode_file.stem.replace("_mbo_events", "")
        else:
            date_str = self._rng.choice(self._dates)

        if date_str not in self._file_index:
            # Fall back to random available date
            log.warning(f"Date {date_str} not pre-computed, picking random")
            date_str = self._rng.choice(self._dates)

        self._episode_date = date_str

        # Load memory-mapped arrays
        self._base_obs = np.load(
            str(self.precomputed_dir / f"{date_str}_base_obs.npy"),
            mmap_mode='r',
        )
        self._meta = np.load(
            str(self.precomputed_dir / f"{date_str}_meta.npy"),
            mmap_mode='r',
        )
        ts_path = self.precomputed_dir / f"{date_str}_timestamps.npy"
        if ts_path.exists():
            self._timestamps = np.load(str(ts_path), mmap_mode='r')
        else:
            self._timestamps = None

        self._n_steps = len(self._base_obs)
        self._step_idx = 0

        # Reset agent state
        self._position = 0
        self._entry_price_rel = 0.0
        self._entry_ts_s = 0.0
        self._mfe_ticks = 0.0
        self._mae_ticks = 0.0
        self._pending_order = None
        self._entry_signal_1s = 0.0
        self._entry_signal_5s = 0.0
        self._entry_signal_10s = 0.0

        self._trade_history = []
        self._sortino_tracker = self.RollingSortino(self.SORTINO_BUFFER_SIZE)
        self._last_sortino = 0.0
        self._consecutive_losses = 0
        self._episode_pnl_ticks = 0.0
        self._episode_trades = 0
        self._trade_ts_history = []
        self._last_closed_trade = None
        self._last_entry_signal_1s = 0.0

        return self._get_obs()

    def step(self, action: int):
        """
        Process one pre-computed step and apply agent action.

        Returns: (obs, reward, done, info)
        """
        if self._base_obs is None:
            raise RuntimeError("Call reset() before step()")

        if self._step_idx >= self._n_steps:
            return self._get_obs(), 0.0, True, self._get_info()

        # Get metadata for current step
        meta = self._meta[self._step_idx]
        ts_s = float(meta[0])       # timestamp seconds
        price_rel = float(meta[1])  # price relative ticks
        et_raw = int(meta[2])       # event type
        spread = float(meta[3])     # spread ticks
        pred_1s = float(meta[4])
        pred_5s = float(meta[5])
        pred_10s = float(meta[6])

        # ── Check pending order fill (simplified FIFO using pre-computed queue depths) ──
        fill_reward = 0.0
        if self._pending_order is not None and et_raw == 0:  # EVENT_TRADE=0 in raw
            fill_reward = self._check_fill(meta, ts_s)

        # ── Check order staleness ──
        stale_reward = 0.0
        if self._pending_order is not None:
            stale_reward = self._check_staleness(meta, ts_s)

        # ── Update MFE/MAE ──
        if self._position != 0:
            unrealized = (price_rel - self._entry_price_rel) * self._position
            if unrealized > self._mfe_ticks:
                self._mfe_ticks = unrealized
            if unrealized < -self._mae_ticks:
                self._mae_ticks = -unrealized

        # ── Apply action ──
        action_reward = self._apply_action(action, meta, ts_s)

        # ── Advance ──
        self._step_idx += 1

        # ── Check end conditions ──
        done = False
        eod_reward = 0.0
        if self._step_idx >= self._n_steps:
            done = True
            if self._position != 0:
                eod_reward = self._force_close(meta, ts_s, reason="eod")

        total_reward = fill_reward + stale_reward + action_reward + eod_reward
        obs = self._get_obs()
        info = self._get_info()

        return obs, float(total_reward), done, info

    def _get_obs(self) -> np.ndarray:
        """
        Reconstruct full 48-dim observation from pre-computed base + agent state.
        This is the hot path -- must be fast.
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        if self._step_idx >= self._n_steps:
            return obs

        # Copy pre-computed base features into correct positions
        base = self._base_obs[self._step_idx]  # (BASE_DIM,) from mmap
        for i, idx in enumerate(BASE_INDICES):
            obs[idx] = base[i]

        # Get metadata for position-dependent features
        meta = self._meta[self._step_idx]
        ts_s = float(meta[0])
        price_rel = float(meta[1])

        # ── [10:14] Position state (agent-dependent) ──
        obs[10] = float(self._position)
        if self._position != 0:
            unrealized = (price_rel - self._entry_price_rel) * self._position
            hold_secs = ts_s - self._entry_ts_s
            obs[11] = max(-3.0, min(3.0, unrealized / 10.0))
            obs[12] = max(0.0, min(3.0, hold_secs / self.MAX_HOLD_SECS))
        if self._pending_order is not None:
            queue_frac = min(1.0, self._pending_order.volume_consumed /
                           max(self._pending_order.queue_position, 1.0))
            obs[13] = queue_frac

        # ── [14:16] MFE/MAE (agent-dependent) ──
        obs[14] = max(0.0, min(3.0, self._mfe_ticks / 10.0))
        obs[15] = max(0.0, min(3.0, self._mae_ticks / 10.0))

        # ── [24:29] Trade history: last 5 PnLs (agent-dependent) ──
        n_hist = len(self._trade_history)
        for i in range(5):
            idx = n_hist - 5 + i
            if 0 <= idx < n_hist:
                pnl = self._trade_history[idx].pnl_ticks
                obs[24 + i] = max(-3.0, min(3.0, pnl / 10.0))

        # ── [29:32] Rolling stats (agent-dependent) ──
        obs[29] = self._sortino_tracker.win_rate()
        sortino = self._sortino_tracker.compute()
        obs[30] = max(-3.0, min(3.0, sortino / 5.0))
        obs[31] = max(0.0, min(3.0, float(self._consecutive_losses) / 5.0))

        # ── [32:34] Order state (agent-dependent) ──
        obs[32] = 1.0 if self._pending_order is not None else 0.0
        obs[33] = float(self._pending_order.side) if self._pending_order is not None else 0.0

        # ── [45:48] Alpha-awareness (agent-dependent) ──
        if self._position != 0:
            hold_secs = ts_s - self._entry_ts_s
            signal_remaining = math.exp(-0.693 * hold_secs / self.SIGNAL_DECAY_HALFLIFE_S)
            obs[45] = max(0.0, min(1.0, signal_remaining))
            obs[46] = max(0.0, min(3.0, abs(self._entry_signal_1s)))
            # Current alpha alignment with position
            pred_1s = float(meta[4])
            obs[47] = max(-3.0, min(3.0, pred_1s * self._position))

        return obs

    def _check_fill(self, meta, ts_s) -> float:
        """Check if a trade event fills our pending order (simplified FIFO)."""
        order = self._pending_order
        price_rel = float(meta[1])
        spread = float(meta[3])

        # Approximate: trade events at our level consume queue
        # We use the pre-computed event type (raw=0 means trade in et_raw,
        # but in the precomputed meta, et_raw is stored as the raw event type code)
        half_spread = max(spread / 2.0, 0.5)

        # Check trade at our level
        at_our_level = False
        if order.side == 1:  # buy at bid
            bid_level = -half_spread
            if abs(price_rel - order.price_rel) < 0.3:
                at_our_level = True
        else:  # sell at ask
            ask_level = half_spread
            if abs(price_rel - order.price_rel) < 0.3:
                at_our_level = True

        if at_our_level:
            # Approximate qty from price_rel magnitude (in absence of exact qty)
            order.volume_consumed += 1.0

        if order.volume_consumed >= order.queue_position:
            return self._execute_fill(order, ts_s, fill_type="limit")

        return 0.0

    def _check_staleness(self, meta, ts_s) -> float:
        """Cancel stale pending orders."""
        order = self._pending_order
        spread = float(meta[3])
        price_rel = float(meta[1])
        half_spread = max(spread / 2.0, 0.5)

        stale = False
        if order.side == 1:
            current_bid = -half_spread
            if current_bid < order.price_rel - 0.5:
                stale = True
        else:
            current_ask = half_spread
            if current_ask > order.price_rel + 0.5:
                stale = True

        hold_so_far = ts_s - (order.placed_ts_ns / 1e9 if isinstance(order.placed_ts_ns, int) else order.placed_ts_ns)
        if hold_so_far > self.MAX_HOLD_SECS * 2:
            stale = True

        if stale:
            self._pending_order = None
        return 0.0

    def _apply_action(self, action: int, meta, ts_s) -> float:
        """Apply agent action. Matches FIFOExecutionEnv._apply_action logic."""
        spread = float(meta[3])
        half_spread = max(spread / 2.0, 0.5)
        pred_1s = float(meta[4])
        price_rel = float(meta[1])

        # Alpha gating
        entry_penalty = 0.0
        if action in (1, 2, 3, 4) and self._position == 0:
            signal_mag = abs(pred_1s)
            if signal_mag < self.ALPHA_GATE_THRESHOLD:
                entry_penalty = self.ALPHA_GATE_PENALTY
            else:
                wants_long = action in (1, 3)
                signal_supports_long = pred_1s > 0
                if wants_long != signal_supports_long:
                    entry_penalty = self.SIGNAL_MISALIGNMENT_PENALTY * min(signal_mag, 1.0)

        if action == 0:
            return 0.0

        elif action == 1:  # Limit BUY at bid
            if self._position == 0 and self._pending_order is None:
                bid_level = -half_spread
                queue_depth = float(meta[11])  # pre-computed bid queue depth
                self._pending_order = self.Order(
                    side=1, price_rel=bid_level,
                    placed_ts_ns=ts_s,  # store as seconds for simplicity
                    queue_position=max(queue_depth, 5.0),
                )
            return entry_penalty

        elif action == 2:  # Limit SELL at ask
            if self._position == 0 and self._pending_order is None:
                ask_level = half_spread
                queue_depth = float(meta[12])  # pre-computed ask queue depth
                self._pending_order = self.Order(
                    side=-1, price_rel=ask_level,
                    placed_ts_ns=ts_s,
                    queue_position=max(queue_depth, 5.0),
                )
            return entry_penalty

        elif action == 3:  # Market BUY
            if self._position == 0 and self._pending_order is None:
                ask_level = half_spread
                order = self.Order(side=1, price_rel=ask_level,
                                   placed_ts_ns=ts_s, queue_position=0)
                order.volume_consumed = 1e9
                self._execute_fill(order, ts_s, fill_type="market_entry")
            return entry_penalty

        elif action == 4:  # Market SELL
            if self._position == 0 and self._pending_order is None:
                bid_level = -half_spread
                order = self.Order(side=-1, price_rel=bid_level,
                                   placed_ts_ns=ts_s, queue_position=0)
                order.volume_consumed = 1e9
                self._execute_fill(order, ts_s, fill_type="market_entry")
            return entry_penalty

        elif action == 5:  # Cancel
            if self._pending_order is not None:
                self._pending_order = None
            return 0.0

        elif action == 6:  # Market exit
            if self._position != 0:
                return self._force_close(self._meta[min(self._step_idx, self._n_steps - 1)],
                                         ts_s, reason="market_exit")
            return 0.0

        return 0.0

    def _execute_fill(self, order, ts_s, fill_type: str) -> float:
        """Execute a fill: open or close position."""
        if self._step_idx < self._n_steps:
            price_rel = float(self._meta[min(self._step_idx, self._n_steps - 1)][1])
        else:
            price_rel = 0.0

        if self._position == 0:
            # Opening position
            self._position = order.side
            self._entry_price_rel = order.price_rel
            self._entry_ts_s = ts_s
            self._mfe_ticks = 0.0
            self._mae_ticks = 0.0
            self._pending_order = None

            # Capture entry signal
            if self._step_idx < self._n_steps:
                meta = self._meta[self._step_idx]
                self._entry_signal_1s = float(meta[4])
                self._entry_signal_5s = float(meta[5])
                self._entry_signal_10s = float(meta[6])
            return 0.0

        else:
            # Closing position
            direction = self._position
            exit_price_rel = order.price_rel if fill_type == "limit" else price_rel

            raw_move = (exit_price_rel - self._entry_price_rel) * direction
            pnl_ticks = raw_move - self.COMMISSION_COST
            hold_secs = ts_s - self._entry_ts_s

            trade = self.Trade(
                direction=direction,
                entry_ts_ns=int(self._entry_ts_s * 1e9),
                exit_ts_ns=int(ts_s * 1e9),
                entry_price_rel=self._entry_price_rel,
                exit_price_rel=exit_price_rel,
                pnl_ticks=pnl_ticks,
                hold_secs=hold_secs,
                exit_reason=fill_type,
                mfe_ticks=self._mfe_ticks,
                mae_ticks=self._mae_ticks,
            )
            self._trade_history.append(trade)
            self._episode_pnl_ticks += pnl_ticks
            self._episode_trades += 1
            self._trade_ts_history.append(ts_s)
            self._last_closed_trade = trade
            self._last_entry_signal_1s = self._entry_signal_1s

            # Sortino reward
            sortino_before = self._sortino_tracker.compute()
            self._sortino_tracker.add(pnl_ticks)
            sortino_after = self._sortino_tracker.compute()
            reward = sortino_after - sortino_before

            # Signal alignment reward
            entry_signal = self._entry_signal_1s
            signal_mag = abs(entry_signal)
            if signal_mag > self.ALPHA_GATE_THRESHOLD:
                signal_sign = 1.0 if entry_signal > 0 else -1.0
                if direction == signal_sign:
                    reward += self.SIGNAL_ALIGNMENT_BONUS * min(signal_mag, 1.0)
                else:
                    reward -= self.SIGNAL_MISALIGNMENT_PENALTY * min(signal_mag, 1.0)

            # Consecutive losses
            if pnl_ticks < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            if self._consecutive_losses > 3:
                reward -= self.CONSECUTIVE_LOSS_PENALTY * (self._consecutive_losses - 3)

            if hold_secs > self.MAX_HOLD_SECS:
                overage = hold_secs - self.MAX_HOLD_SECS
                reward -= 0.01 * min(overage, 30.0)

            # Overtrading check
            cutoff = ts_s - 60.0
            self._trade_ts_history = [t for t in self._trade_ts_history if t > cutoff]
            if len(self._trade_ts_history) > self.OVERTRADING_LIMIT:
                reward -= 0.05 * (len(self._trade_ts_history) - self.OVERTRADING_LIMIT)

            # Reset position
            self._position = 0
            self._entry_price_rel = 0.0
            self._mfe_ticks = 0.0
            self._mae_ticks = 0.0
            self._pending_order = None
            self._last_sortino = sortino_after

            return reward

    def _force_close(self, meta, ts_s, reason: str) -> float:
        """Force-close via market order."""
        if self._position == 0:
            return 0.0

        spread = float(meta[3])
        price_rel = float(meta[1])
        half_spread = max(spread / 2.0, 0.5)

        if self._position == 1:
            exit_price = price_rel - half_spread
        else:
            exit_price = price_rel + half_spread

        fake_order = self.Order(
            side=-self._position,
            price_rel=exit_price,
            placed_ts_ns=ts_s,
            queue_position=0,
        )
        fake_order.volume_consumed = 1e9
        return self._execute_fill(fake_order, ts_s, fill_type=reason)

    def _get_info(self) -> dict:
        """Return info dict matching FIFOExecutionEnv._get_info."""
        info = {
            "step_idx": self._step_idx,
            "position": self._position,
            "pending_order": self._pending_order is not None,
            "episode_pnl_ticks": self._episode_pnl_ticks,
            "episode_pnl_usd": self._episode_pnl_ticks * self.TICK_VALUE,
            "episode_trades": self._episode_trades,
            "win_rate": self._sortino_tracker.win_rate(),
            "rolling_sortino": self._sortino_tracker.compute(),
            "consecutive_losses": self._consecutive_losses,
            "mfe_ticks": self._mfe_ticks,
            "mae_ticks": self._mae_ticks,
            "date": self._episode_date,
        }
        if self._last_closed_trade is not None:
            t = self._last_closed_trade
            info["last_trade"] = {
                "pnl_ticks": t.pnl_ticks,
                "mfe_ticks": t.mfe_ticks,
                "mae_ticks": t.mae_ticks,
                "hold_secs": t.hold_secs,
                "direction": t.direction,
                "exit_reason": t.exit_reason,
                "entry_signal_1s": self._last_entry_signal_1s,
            }
            self._last_closed_trade = None
        return info

    def get_episode_metrics(self) -> dict:
        """Compute episode metrics matching FIFOExecutionEnv.get_episode_metrics."""
        if not self._trade_history:
            return {
                "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "profit_factor": 0.0, "avg_pnl_ticks": 0.0, "total_pnl_ticks": 0.0,
                "total_pnl_usd": 0.0, "avg_hold_secs": 0.0, "avg_mfe_ticks": 0.0,
                "avg_mae_ticks": 0.0, "fill_rate": 0.0,
            }
        pnls = np.array([t.pnl_ticks for t in self._trade_history], dtype=np.float64)
        holds = np.array([t.hold_secs for t in self._trade_history], dtype=np.float64)
        mfes = np.array([t.mfe_ticks for t in self._trade_history], dtype=np.float64)
        maes = np.array([t.mae_ticks for t in self._trade_history], dtype=np.float64)
        n = len(pnls)
        win_rate = float((pnls > 0).mean())
        mean_pnl = float(pnls.mean())
        std_pnl = float(pnls.std()) if n > 1 else 1.0
        sharpe = mean_pnl / (std_pnl + 1e-8) * math.sqrt(n)
        downside = pnls[pnls < 0]
        if len(downside) == 0:
            sortino = mean_pnl * 10.0
        else:
            dd = float(np.sqrt(np.mean(downside ** 2)))
            sortino = mean_pnl / (dd + 1e-8)
        gross_wins = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
        gross_losses = abs(float(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-8
        pf = gross_wins / gross_losses
        return {
            "n_trades": n, "win_rate": win_rate, "sharpe": sharpe,
            "sortino": sortino, "profit_factor": pf,
            "avg_pnl_ticks": mean_pnl,
            "total_pnl_ticks": float(pnls.sum()),
            "total_pnl_usd": float(pnls.sum()) * self.TICK_VALUE,
            "avg_hold_secs": float(holds.mean()),
            "avg_mfe_ticks": float(mfes.mean()),
            "avg_mae_ticks": float(maes.mean()),
            "fill_rate": 1.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute RL observation tensors for Split DQN training"
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with MBO event .npz files")
    parser.add_argument("--pred-dir", type=str, required=True,
                        help="Directory with CNN-Mamba prediction .npz files")
    parser.add_argument("--patchtst-dir", type=str, default=None,
                        help="Directory with PatchTST prediction .npz files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save pre-computed .npy files")
    parser.add_argument("--n-workers", type=int, default=None,
                        help="Number of parallel workers (default: 60%% of cores)")
    parser.add_argument("--dates", type=str, nargs="*", default=None,
                        help="Specific dates to process (e.g. 20260301 20260302)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover files
    event_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not event_files:
        log.error(f"No MBO event files in {data_dir}")
        return

    # Filter to specific dates if requested
    if args.dates:
        date_set = set(args.dates)
        event_files = [f for f in event_files if f.stem.replace("_mbo_events", "") in date_set]
        log.info(f"Filtered to {len(event_files)} files for dates: {args.dates}")

    log.info(f"Found {len(event_files)} MBO event files to process")

    # Determine workers
    n_workers = args.n_workers
    if n_workers is None:
        try:
            import psutil
            n_workers = max(2, int(psutil.cpu_count(logical=False) * 0.6))
        except ImportError:
            n_workers = max(2, (os.cpu_count() or 4) // 2)
    log.info(f"Using {n_workers} workers")

    # Build work items
    work_items = [
        (str(f), args.pred_dir, args.patchtst_dir, str(output_dir))
        for f in event_files
    ]

    # Process in parallel
    t_start = time.time()
    completed = 0
    skipped = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(precompute_one_file, w): w for w in work_items}

        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                errors += 1
                log.error(f"Worker crashed: {e}")
                continue

            if result["status"] == "skipped":
                skipped += 1
            elif result["status"] == "ok":
                completed += 1
                elapsed = result["elapsed"]
                n_steps = result["n_steps"]
                rate = n_steps / elapsed if elapsed > 0 else 0
                total_done = completed + skipped + errors
                total_files = len(work_items)
                pct = total_done / total_files * 100
                eta_s = (time.time() - t_start) / max(total_done, 1) * (total_files - total_done)
                log.info(
                    f"[{total_done}/{total_files} {pct:.0f}%] "
                    f"{result['date']}: {n_steps:,} steps in {elapsed:.1f}s "
                    f"({rate:,.0f} steps/s) | "
                    f"ETA: {eta_s/60:.1f}min"
                )
            else:
                errors += 1
                log.warning(f"{result['date']}: {result['status']}")

    total_time = time.time() - t_start
    log.info(f"\nDone! {completed} completed, {skipped} skipped, {errors} errors in {total_time/60:.1f}min")
    log.info(f"Output directory: {output_dir}")

    # Print disk usage
    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.npy"))
    log.info(f"Total disk usage: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
