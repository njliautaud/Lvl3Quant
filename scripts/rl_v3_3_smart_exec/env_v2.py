"""
v2: Same as env.py but with sized-reward shaping using sizing calibrator.

Shaping rule (variant B from sized-replay, applied only at entry-time):
    shaped = raw * clip(calib_pred / median(calib_pred), 0.5, 2.0)

Applied to entry-time rewards only (passive fill commission/seed, market order
crossings + commission). Hold MTM and exit rewards keep raw to preserve PPO's
learning signal on close-out timing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

COMMISSION_RT_TICKS = 0.376
# HC #392: market orders enter at ask (buy) or bid (sell) — that price IS the
# reference fill. NO extra spread cost on top. User flagged this 3+ times.
# The HC #392 lint check enforces SPREAD_CROSS_TICKS = 0.0 here.
SPREAD_CROSS_TICKS  = 0.0
RTH_STEPS = 23_400
MAX_HOLD_STEPS = 240
CANCEL_BUDGET  = 50
PASSIVE_FILL_WINDOW = 50

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

BOOK_CTX_DIMS = 3
POS_CTX_DIMS  = 6
OBS_DIM = len(PRED_HEADS) + BOOK_CTX_DIMS + POS_CTX_DIMS

N_ACTIONS = 6
A_HOLD, A_BID, A_ASK, A_MKT_BUY, A_MKT_SELL, A_CANCEL = range(N_ACTIONS)


class SizingMLP(nn.Module):
    """Architecture must match scripts/meta_mlp_v3_3/train_sizing.py SizingMLP."""
    def __init__(self, in_dim: int = 35):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, x):
        z = self.trunk(x).squeeze(-1)
        return self.softplus(z)


def _precompute_calib_preds(
    calib_pt: Path,
    preds: np.ndarray,           # (N, 32)
    vol30: np.ndarray,           # (N,)
    qimb: np.ndarray,            # (N,)
) -> Tuple[np.ndarray, float]:
    """Run the sizing calibrator over the entire NPZ once. Returns (preds_arr, median)."""
    ckpt = torch.load(str(calib_pt), map_location="cpu", weights_only=False)
    feat_mu = np.asarray(ckpt["feat_mu"], dtype=np.float32)
    feat_sd = np.asarray(ckpt["feat_sd"], dtype=np.float32)
    in_dim  = int(ckpt.get("in_dim", 35))
    N = preds.shape[0]
    spread = np.ones(N, dtype=np.float32)
    X = np.concatenate([preds, vol30[:, None], qimb[:, None], spread[:, None]], axis=1)
    assert X.shape[1] == in_dim, f"calib in_dim {in_dim} != built features {X.shape[1]}"
    Xn = (X - feat_mu) / feat_sd

    model = SizingMLP(in_dim=in_dim)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    out = np.empty(N, dtype=np.float32)
    bs = 16384
    with torch.no_grad():
        for i in range(0, N, bs):
            xb = torch.from_numpy(Xn[i:i+bs])
            if use_cuda:
                xb = xb.cuda()
            out[i:i+bs] = model(xb).cpu().numpy().astype(np.float32)
    med = float(np.median(out))
    if med < 1e-6:
        med = 1e-6
    return out, med


class V33SmartExecEnvV2(gym.Env):
    """v2 env with sized-reward shaping. Entry-step rewards multiplied by
    clip(calib[i] / median_calib, 0.5, 2.0)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        npz_path: str | Path,
        seed: int = 0,
        sizing_calibrator: Optional[str | Path] = None,
        shape_clip_lo: float = 0.5,
        shape_clip_hi: float = 2.0,
    ):
        super().__init__()
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"v3.3 predictions NPZ not found: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=False)
        head_arrays = []
        for h in PRED_HEADS:
            if h not in data:
                raise KeyError(f"NPZ missing head '{h}'")
            head_arrays.append(np.nan_to_num(data[h], nan=0.0).astype(np.float32))
        self.preds = np.stack(head_arrays, axis=1)
        self.N = self.preds.shape[0]

        self.fifo_pnl_tp8sl5 = np.nan_to_num(data["target_fifo_tp8sl5_net"], nan=0.0).astype(np.float32)
        self.fifo_filled_tp8sl5 = (
            np.nan_to_num(data["target_fifo_tp8sl5_hit_tp"], nan=0.0).astype(np.float32) > 0
        ).astype(np.float32)
        self.fifo_pnl_tp4sl3 = np.nan_to_num(data["target_fifo_tp4sl3_net"], nan=0.0).astype(np.float32)
        self.ret_30s = np.nan_to_num(data["target_log_ret_30s"], nan=0.0).astype(np.float32)
        self.ret_1s  = np.nan_to_num(data["target_log_ret_1s"], nan=0.0).astype(np.float32)
        self.vol_30s_pred = np.nan_to_num(data["pred_pred_realized_vol_30s_ticks"], nan=1.0).astype(np.float32)
        self.queue_imb = (np.nan_to_num(data["pred_p_up_30s"], nan=0.5) - 0.5).astype(np.float32)

        # ── Reward shaping setup ─────────────────────────────────────────────
        self.use_shaping = sizing_calibrator is not None
        self.shape_clip_lo = float(shape_clip_lo)
        self.shape_clip_hi = float(shape_clip_hi)
        if self.use_shaping:
            calib_preds, med = _precompute_calib_preds(
                Path(sizing_calibrator), self.preds, self.vol_30s_pred, self.queue_imb,
            )
            self.calib_preds = calib_preds
            self.calib_median = med
            # Precompute shaping multipliers for fast lookup
            ratios = self.calib_preds / self.calib_median
            self.shape_mult = np.clip(
                ratios, self.shape_clip_lo, self.shape_clip_hi
            ).astype(np.float32)
            # Telemetry: cumulative stats
            self._shape_used_count = 0
            self._shape_sum = 0.0
            self._shape_sum_sq = 0.0
            self._raw_entry_sum = 0.0
            self._raw_entry_sum_sq = 0.0
            self._shaped_entry_sum = 0.0
            self._shaped_entry_sum_sq = 0.0
        else:
            self.calib_preds = None
            self.calib_median = 0.0
            self.shape_mult = None

        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.step_idx = 0
        self.episode_steps = 0
        self.position_side = 0
        self.position_age = 0
        self.entry_idx = -1
        self.unrealized_ticks = 0.0
        self.cancel_budget = CANCEL_BUDGET
        self.pending_order = 0
        self.pending_age = 0
        self.realized_pnl_ticks = 0.0
        self.trades_count = 0

    def _obs(self) -> np.ndarray:
        i = min(self.step_idx, self.N - 1)
        pred_vec = self.preds[i]
        vol30 = self.vol_30s_pred[i]
        qimb = self.queue_imb[i]
        spread = 1.0
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
        reward = exit_pnl_ticks - 0.5 * COMMISSION_RT_TICKS
        self.realized_pnl_ticks += reward
        self.trades_count += 1
        self.position_side = 0
        self.position_age = 0
        self.entry_idx = -1
        self.unrealized_ticks = 0.0
        return reward

    def _shape_entry(self, raw_entry_reward: float, i: int) -> float:
        """Apply sized-reward shaping at entry step. Tracks telemetry."""
        if not self.use_shaping:
            return raw_entry_reward
        m = float(self.shape_mult[i])
        shaped = raw_entry_reward * m
        # Telemetry
        self._shape_used_count += 1
        self._shape_sum += m
        self._shape_sum_sq += m * m
        self._raw_entry_sum += raw_entry_reward
        self._raw_entry_sum_sq += raw_entry_reward * raw_entry_reward
        self._shaped_entry_sum += shaped
        self._shaped_entry_sum_sq += shaped * shaped
        return shaped

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
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

        # 1) MTM if in position (raw, no shaping on MTM/exit)
        if self.position_side != 0:
            mtm_delta = self.position_side * float(self.ret_1s[i])
            self.unrealized_ticks += mtm_delta
            self.position_age += 1
            reward += mtm_delta
            if self.position_age >= MAX_HOLD_STEPS:
                reward += self._exit_position(self.unrealized_ticks)

        # 2) Pending passive resolution
        if self.pending_order != 0:
            self.pending_age += 1
            if self.pending_age >= PASSIVE_FILL_WINDOW:
                side = self.pending_order
                if self.fifo_filled_tp8sl5[i] > 0 and self.position_side == 0:
                    raw_pnl = float(self.fifo_pnl_tp8sl5[i]) * side
                    self.position_side = side
                    self.entry_idx = i
                    self.position_age = 0
                    self.unrealized_ticks = raw_pnl
                    # Entry-step reward = -half commission. Apply shaping.
                    raw_entry = -0.5 * COMMISSION_RT_TICKS
                    reward += self._shape_entry(raw_entry, i)
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
                # HC #392: market BUY enters at ask = reference fill. MTM starts
                # at 0; only half-commission entry, NO spread cross.
                self.unrealized_ticks = 0.0
                raw_entry = -0.5 * COMMISSION_RT_TICKS
                reward += self._shape_entry(raw_entry, i)
        elif action == A_MKT_SELL:
            if self.position_side == 0 and self.pending_order == 0:
                self.position_side = -1
                self.entry_idx = i
                self.position_age = 0
                # HC #392: market SELL enters at bid = reference fill.
                self.unrealized_ticks = 0.0
                raw_entry = -0.5 * COMMISSION_RT_TICKS
                reward += self._shape_entry(raw_entry, i)
        elif action == A_CANCEL:
            if self.pending_order != 0 and self.cancel_budget > 0:
                self.pending_order = 0
                self.pending_age = 0
                self.cancel_budget -= 1

        # 4) Step clock
        self.step_idx += 1
        self.episode_steps += 1

        terminated = False
        truncated = False
        if self.step_idx >= self.N - 1:
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

    def get_shaping_stats(self) -> Dict[str, float]:
        if not self.use_shaping or self._shape_used_count == 0:
            return {}
        n = self._shape_used_count
        mean_m = self._shape_sum / n
        var_m = max(0.0, self._shape_sum_sq / n - mean_m * mean_m)
        mean_raw = self._raw_entry_sum / n
        var_raw = max(0.0, self._raw_entry_sum_sq / n - mean_raw * mean_raw)
        mean_shaped = self._shaped_entry_sum / n
        var_shaped = max(0.0, self._shaped_entry_sum_sq / n - mean_shaped * mean_shaped)
        return {
            "shape_n": float(n),
            "shape_mult_mean": float(mean_m),
            "shape_mult_std": float(var_m ** 0.5),
            "shape_raw_entry_mean": float(mean_raw),
            "shape_raw_entry_std": float(var_raw ** 0.5),
            "shape_shaped_entry_mean": float(mean_shaped),
            "shape_shaped_entry_std": float(var_shaped ** 0.5),
            "calib_median": float(self.calib_median),
        }
