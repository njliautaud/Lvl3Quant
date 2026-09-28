"""
v3.3 smart-execution RL environment (HC #396 #2).

Gym-like (no gymnasium dep) discrete-action env wrapping the v3.3 prediction NPZ.
At each 250 ms timestep, the agent sees the full 32-head model output + book
context + own-position context, and chooses one of 6 actions.

Action space (Discrete 6):
    0: HOLD                — do nothing
    1: PLACE PASSIVE BID   — limit order at best bid (queue model: 0.5 fill prob)
    2: PLACE PASSIVE ASK   — limit order at best ask (queue model: 0.5 fill prob)
    3: MARKET BUY          — fill at ask immediately
    4: MARKET SELL         — fill at bid immediately
    5: CANCEL / CLOSE      — cancel pending limit OR close open position at market

Reward (HC #392 binding):
    Per step: position_sign * realized_log_ret_1s[t] / STEPS_PER_SEC  (in ticks)
    On every round-trip CLOSE: subtract ES_RT_COMMISSION_TICKS = 0.376 ticks.
    NO extra spread cost — passive entries don't pay; market entries are
    implicitly charged via worse fill (we charge half-spread = 0.5 tick on
    market entry vs passive). On exit the same logic applies.

Episode termination:
    - position is fully closed AFTER having been opened, OR
    - max_steps reached (default 200), OR
    - end of OOT day reached.

State normalisation:
    Static portion (heads + book) is z-scored using feat_mean/feat_std cached at
    dataset-build time. Position portion is raw (already in interpretable units).
"""
from __future__ import annotations

import numpy as np

from v33_rl_dataset import (
    V33RLDataset,
    N_HEADS, N_BOOK, N_POS, STATE_DIM,
    STEPS_PER_SEC,
)

# Costs (HC #392) — commission-only
COMMISSION_TICKS_RT = 0.376
# Spread crossing penalty for MARKET orders (entry/exit). This is the realized
# half-spread the trader pays vs the mid; it's economic reality, not extra cost
# on top of HC #392. We charge 0.5 tick on each market leg (= 1 full tick RT).
MARKET_HALF_SPREAD_TICKS = 0.5

# Actions
A_HOLD = 0
A_PASSIVE_BID = 1
A_PASSIVE_ASK = 2
A_MARKET_BUY = 3
A_MARKET_SELL = 4
A_CANCEL_OR_CLOSE = 5
N_ACTIONS = 6

DEFAULT_MAX_STEPS = 200
DEFAULT_PASSIVE_FILL_PROB = 0.5   # queue-position heuristic per HC #357 mean-of-queue
DEFAULT_PASSIVE_TIMEOUT_STEPS = 40  # ~10s @ 250ms stride


class V33SmartExecEnv:
    """Single-agent step env. Designed to be vector-wrapped for PPO rollouts."""

    def __init__(
        self,
        dataset: V33RLDataset,
        eligible_indices: np.ndarray | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        passive_fill_prob: float = DEFAULT_PASSIVE_FILL_PROB,
        passive_timeout_steps: int = DEFAULT_PASSIVE_TIMEOUT_STEPS,
        rng: np.random.Generator | None = None,
    ):
        self.ds = dataset
        self.max_steps = int(max_steps)
        self.passive_fill_prob = float(passive_fill_prob)
        self.passive_timeout_steps = int(passive_timeout_steps)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Index pool from which we sample episode starts
        if eligible_indices is None:
            # Don't start in the last max_steps of any day (avoid bumping into day boundary)
            ok = np.ones(self.ds.n_samples, dtype=bool)
            # Drop the trailing max_steps samples of each day
            for di in range(int(self.ds.day_idx.max()) + 1):
                last = np.where(self.ds.day_idx == di)[0][-self.max_steps:]
                ok[last] = False
            eligible_indices = np.where(ok)[0]
        self.eligible_indices = np.asarray(eligible_indices, dtype=np.int64)

        # State buffers
        self._t = 0
        self._step_in_ep = 0
        self._day_end = 0
        # Position: +1 long, -1 short, 0 flat
        self._position = 0
        self._position_age = 0
        self._position_entry_idx = -1
        self._pos_pnl_ticks = 0.0
        # Pending order: side ∈ {+1=bid (will buy), -1=ask (will sell), 0=none}
        self._pending_side = 0
        self._pending_age = 0
        # Trade accounting (for episode-end metrics)
        self._round_trips = 0
        self._round_trip_pnl: list[float] = []
        self._has_traded = False

    # ------------------------------------------------------------------
    # Required Gym-like API
    # ------------------------------------------------------------------
    @property
    def observation_dim(self) -> int:
        return STATE_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def reset(self) -> np.ndarray:
        # Pick a random eligible start
        idx = int(self.rng.choice(self.eligible_indices))
        self._t = idx
        self._step_in_ep = 0
        # End-of-day boundary for this episode
        day = int(self.ds.day_idx[idx])
        day_last = int(np.where(self.ds.day_idx == day)[0][-1])
        self._day_end = min(idx + self.max_steps, day_last)
        # Reset position/order
        self._position = 0
        self._position_age = 0
        self._position_entry_idx = -1
        self._pos_pnl_ticks = 0.0
        self._pending_side = 0
        self._pending_age = 0
        self._round_trips = 0
        self._round_trip_pnl = []
        self._has_traded = False
        return self._observe()

    def step(self, action: int):
        action = int(action)
        # 1) Process pending passive order: maybe fills at start of this step,
        #    or cancel-timeout.
        passive_filled = False
        if self._pending_side != 0:
            self._pending_age += 1
            # Stochastic fill (queue model): each step has p_fill probability
            if self.rng.random() < self.passive_fill_prob:
                # Fill at touch — NO spread crossing cost
                self._position = self._pending_side
                self._position_age = 0
                self._position_entry_idx = self._t
                self._pos_pnl_ticks = 0.0
                self._pending_side = 0
                self._pending_age = 0
                passive_filled = True
                self._has_traded = True
            elif self._pending_age >= self.passive_timeout_steps:
                # Timeout: cancel
                self._pending_side = 0
                self._pending_age = 0

        # 2) Apply action
        step_reward = 0.0
        if action == A_HOLD:
            pass
        elif action == A_PASSIVE_BID:
            if self._position == 0 and self._pending_side == 0:
                self._pending_side = +1  # will go long if filled
                self._pending_age = 0
        elif action == A_PASSIVE_ASK:
            if self._position == 0 and self._pending_side == 0:
                self._pending_side = -1  # will go short if filled
                self._pending_age = 0
        elif action == A_MARKET_BUY:
            if self._position == 0:
                # Cancel any pending opposite passive
                self._pending_side = 0
                self._pending_age = 0
                # Pay half-spread on market entry (economic spread crossing,
                # NOT extra cost on top of HC #392 commission)
                step_reward -= MARKET_HALF_SPREAD_TICKS
                self._position = +1
                self._position_age = 0
                self._position_entry_idx = self._t
                self._pos_pnl_ticks = 0.0
                self._has_traded = True
        elif action == A_MARKET_SELL:
            if self._position == 0:
                self._pending_side = 0
                self._pending_age = 0
                step_reward -= MARKET_HALF_SPREAD_TICKS
                self._position = -1
                self._position_age = 0
                self._position_entry_idx = self._t
                self._pos_pnl_ticks = 0.0
                self._has_traded = True
        elif action == A_CANCEL_OR_CLOSE:
            if self._pending_side != 0:
                self._pending_side = 0
                self._pending_age = 0
            elif self._position != 0:
                # Market exit: pay half-spread + RT commission, then book PnL
                step_reward -= MARKET_HALF_SPREAD_TICKS
                step_reward -= COMMISSION_TICKS_RT
                # Tick PnL accumulated in self._pos_pnl_ticks stays (already added
                # over the holding period in step #3 below). Just flatten.
                self._round_trips += 1
                self._round_trip_pnl.append(self._pos_pnl_ticks)
                self._position = 0
                self._position_age = 0
                self._position_entry_idx = -1
                self._pos_pnl_ticks = 0.0

        # 3) Hold-PnL: incremental tick PnL for an open position over this 250ms step
        if self._position != 0:
            r1 = float(self.ds.realized_log_ret_1s[self._t])
            # 1s realized return spread over STEPS_PER_SEC=4 quarter-second steps
            delta_ticks = self._position * (r1 / float(STEPS_PER_SEC))
            self._pos_pnl_ticks += delta_ticks
            step_reward += delta_ticks
            self._position_age += 1

        # 4) Advance time
        self._t += 1
        self._step_in_ep += 1

        # 5) Termination
        done = False
        info = {"passive_filled": passive_filled}
        # End of day / max_steps reached
        if self._t >= self._day_end or self._step_in_ep >= self.max_steps:
            # Force close any open position at market (charge same costs)
            if self._position != 0:
                step_reward -= MARKET_HALF_SPREAD_TICKS
                step_reward -= COMMISSION_TICKS_RT
                self._round_trips += 1
                self._round_trip_pnl.append(self._pos_pnl_ticks)
                self._position = 0
                self._pos_pnl_ticks = 0.0
            self._pending_side = 0
            self._pending_age = 0
            done = True
            info["episode_round_trips"] = self._round_trips
            info["episode_round_trip_pnl"] = list(self._round_trip_pnl)
            info["episode_has_traded"] = self._has_traded
        # OR: position closed after having been opened → end episode
        elif self._round_trips > 0 and self._position == 0 and self._pending_side == 0:
            done = True
            info["episode_round_trips"] = self._round_trips
            info["episode_round_trip_pnl"] = list(self._round_trip_pnl)
            info["episode_has_traded"] = self._has_traded

        obs = self._observe() if not done else np.zeros(STATE_DIM, dtype=np.float32)
        return obs, float(step_reward), bool(done), info

    # ------------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        t = min(self._t, self.ds.n_samples - 1)
        heads = self.ds.head_matrix[t]
        book = self.ds.book_matrix[t]
        static = np.concatenate([heads, book])
        static_z = (static - self.ds.feat_mean) / self.ds.feat_std
        # Position context (raw units, hand-scaled)
        pos = np.array([
            float(self._position),                                  # in_position (signed)
            float(self._position_age) / 50.0,                        # scaled age
            float(self._pos_pnl_ticks) / 4.0,                        # scaled pnl
            float(max(0, self.passive_timeout_steps - self._pending_age)) / 50.0 if self._pending_side else 0.0,
            float(self._pending_side),
            float(self._pending_age) / 50.0,
        ], dtype=np.float32)
        return np.concatenate([static_z.astype(np.float32), pos], dtype=np.float32)


class VecEnv:
    """Tiny vectorised wrapper over N independent V33SmartExecEnv instances."""

    def __init__(self, n_envs: int, dataset: V33RLDataset, **env_kwargs):
        self.n_envs = int(n_envs)
        seeds = np.random.SeedSequence(2026).spawn(self.n_envs)
        self.envs = [
            V33SmartExecEnv(dataset, rng=np.random.default_rng(s), **env_kwargs)
            for s in seeds
        ]
        self.observation_dim = self.envs[0].observation_dim
        self.n_actions = self.envs[0].n_actions

    def reset(self) -> np.ndarray:
        return np.stack([e.reset() for e in self.envs], axis=0)

    def step(self, actions: np.ndarray):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for e, a in zip(self.envs, actions):
            obs, rew, done, info = e.step(int(a))
            if done:
                obs = e.reset()
            obs_list.append(obs); rew_list.append(rew); done_list.append(done); info_list.append(info)
        return (
            np.stack(obs_list, axis=0),
            np.asarray(rew_list, dtype=np.float32),
            np.asarray(done_list, dtype=bool),
            info_list,
        )


if __name__ == "__main__":
    # Tiny smoke test (no MLflow, no PPO — just verify the env loop runs)
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
    )
    from v33_rl_dataset import build_dataset
    ds = build_dataset(p)
    env = V33SmartExecEnv(ds, max_steps=100)
    rng = np.random.default_rng(0)
    rewards = []
    for ep in range(5):
        obs = env.reset()
        assert obs.shape == (STATE_DIM,), obs.shape
        total = 0.0
        steps = 0
        while True:
            a = int(rng.integers(0, N_ACTIONS))
            obs, r, done, info = env.step(a)
            total += r; steps += 1
            if done:
                rewards.append(total)
                print(f"ep {ep}: steps={steps} reward={total:.3f} round_trips={info.get('episode_round_trips')} traded={info.get('episode_has_traded')}")
                break
    print(f"mean episode reward (random policy, 5 eps): {np.mean(rewards):.3f}")
