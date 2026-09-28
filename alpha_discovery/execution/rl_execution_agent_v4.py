#!/usr/bin/env python3
"""
RL Execution Agent v3 (ES Futures — Neptune RTX 3090)
======================================================
Trains a PPO-based RL agent that learns DYNAMIC execution decisions
for ES futures, replacing the static params from Optuna MLP v2.

KEY DIFFERENCES FROM rl_exec_agent_v2.py:
  - 7-action flat discrete space (WAIT/ENTER_LONG/ENTER_SHORT/EXIT_PASSIVE/
    EXIT_MARKET/TIGHTEN_STOP/WIDEN_STOP) — agent explicitly manages position
  - Explicit position tracking: flat/long/short, unrealized PnL, hold time, MFE/MAE
  - FIFO-only: no midpoint. Cost = 0.376 ticks RT (HC #89)
  - PatchTST confluence: both CNN-Mamba + PatchTST signals in state
  - ES constants: $12.50/tick (not NQ $6.25)
  - Walk-forward sliding 60d train → 5d eval (HC #0)
  - ToD gate: no entries 9:25-9:40 ET (open volatility exclusion)
  - Daily loss circuit breaker: halt after -20 ticks
  - MLflow logging mandatory (experiment: "rl_execution_agent")
  - Behavioral cloning warm-start from Optuna MLP best params

ARCHITECTURE:
  - State: ~56 dim (signal + position + market + time + PatchTST confluence)
  - Actions: 7 flat discrete (see ACTION_* constants)
  - PPO with per-head entropy, learnable temperature, GAE
  - Policy: 3-layer MLP + LayerNorm + GELU (128-dim hidden)
  - Critic: 2-layer MLP (shares backbone with policy up to last layer)

USAGE:
  # Train walk-forward (all folds)
  python rl_execution_agent.py --mode train \\
      --cnn-pred-dir /home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar \\
      --ptst-pred-dir /home/nick/Lvl3Quant/output/patchtst_smart_v3_mar \\
      --output-dir /home/nick/Lvl3Quant/output/rl_execution_agent \\
      --device cuda

  # Evaluate best checkpoint
  python rl_execution_agent.py --mode eval \\
      --checkpoint /home/nick/Lvl3Quant/output/rl_execution_agent/best_fold_overall.pt

  # Behavioral cloning only (warm-start from static params)
  python rl_execution_agent.py --mode bc_warmstart \\
      --cnn-pred-dir /home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar

COST CONSTANTS (HC #231(A) — ES AMP/Rithmic):
  ES_TICK_VALUE  = $12.50
  ES_RT_COMMISSION = $4.70 = 0.376 ticks
  ALL order types: 0.376 ticks (commission only — no spread crossing cost)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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
    print("ERROR: PyTorch not available. Install with: pip install torch")
    sys.exit(1)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rl_exec_v3")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (HC #89 — ES futures, AMP/Rithmic)
# ═══════════════════════════════════════════════════════════════════════════════

ES_TICK_VALUE        = 12.50              # $ per tick
ES_RT_COMMISSION     = 4.70              # $ round-trip
ES_COMMISSION_TICKS  = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376 ticks

# Order type costs (HC #89)
COST_PASSIVE_TICKS   = ES_COMMISSION_TICKS           # 0.376t — limit at bid/ask
COST_MARKET_TICKS    = ES_COMMISSION_TICKS           # HC #231(A): commission only — no spread cost

# Label-to-ticks scaling (matches CNN-Mamba output scale)
LABEL_TO_TICKS = 10.0

# ── Action space (7 flat discrete) ─────────────────────────────────────────────
ACTION_WAIT          = 0   # Do nothing
ACTION_ENTER_LONG    = 1   # Enter long, limit at bid (passive)
ACTION_ENTER_SHORT   = 2   # Enter short, limit at ask (passive)
ACTION_EXIT_PASSIVE  = 3   # Exit position, limit (passive, 0.376t cost)
ACTION_EXIT_MARKET   = 4   # Exit position, market (aggressive, 0.376t cost — HC #231(A))
ACTION_TIGHTEN_STOP  = 5   # Move SL closer (reduce max loss)
ACTION_WIDEN_STOP    = 6   # Move SL further (give trade more room)
N_ACTIONS = 7

ACTION_NAMES = {
    0: "WAIT", 1: "ENTER_LONG", 2: "ENTER_SHORT",
    3: "EXIT_PASSIVE", 4: "EXIT_MARKET",
    5: "TIGHTEN_STOP", 6: "WIDEN_STOP",
}

# ── Position states ─────────────────────────────────────────────────────────────
POS_FLAT  = 0
POS_LONG  = 1
POS_SHORT = -1

# ── SL levels (ticks) — managed by TIGHTEN/WIDEN actions ──────────────────────
SL_LEVELS = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0], dtype=np.float32)
SL_DEFAULT_IDX = 3   # 5 ticks default SL
SL_MIN_IDX = 0
SL_MAX_IDX = len(SL_LEVELS) - 1

# ── Max hold (steps, not seconds — since steps are signal-driven) ───────────────
MAX_HOLD_STEPS    = 20    # ~10-100s depending on signal density
MAX_DAILY_LOSS    = -20.0  # ticks — circuit breaker

# ── Walk-forward config ─────────────────────────────────────────────────────────
WF_TRAIN_DAYS = 60
WF_EVAL_DAYS  = 5

# ── ToD gate (no entries during open volatility window) ────────────────────────
TOD_GATE_START_HOUR = 9.0 + 25.0/60.0   # 9:25 ET
TOD_GATE_END_HOUR   = 9.0 + 40.0/60.0   # 9:40 ET

# ── PPO hyperparameters ─────────────────────────────────────────────────────────
PPO_GAMMA          = 0.99
PPO_LAMBDA         = 0.95
PPO_CLIP_EPS       = 0.2
PPO_VALUE_COEFF    = 0.5
PPO_MAX_GRAD_NORM  = 1.0
PPO_BATCH_SIZE     = 256
PPO_EPOCHS         = 8
PPO_ROLLOUT_STEPS  = 4096
PPO_LR             = 3e-4
PPO_WARMUP_ITERS   = 20

# Per-action entropy coefficient (flat action space — higher entropy for WAIT to prevent collapse)
ENTROPY_COEFF_BASE = 0.01
ENTROPY_COEFF_WAIT = 0.005   # slight preference not to collapse to pure WAIT

# ── Reward shaping ──────────────────────────────────────────────────────────────
REWARD_WAIT_PENALTY    = -0.02   # per-step cost for waiting
REWARD_HOLD_PENALTY    = -0.005   # per-step cost for holding position (urgency)
REWARD_SORTINO_WEIGHT  = 0.3      # weight of Sortino shaping bonus
SORTINO_WINDOW         = 50       # trades window for Sortino computation

# ── State dimension ─────────────────────────────────────────────────────────────
# Computed below after STATE_DIM_* definitions
# Signal features: CNN-Mamba (1s/5s/10s) + PatchTST (1s/5s/10s) = 6
# CNN confidence percentiles: p90/p95/p99 flags = 3
# Market features: vol, rvol_fast, flow_imbalance, momentum = 4
# Position features: pos_flat/long/short, unrealized_pnl, hold_frac, mfe, mae = 7
# SL level: normalized sl_idx = 1
# Time features: tod_sin, tod_cos, is_open, is_core, mins_since_open, mins_to_close = 6
# Trade history: last 5 pnls, win_streak, loss_streak, cum_pnl_today = 8
# Signal dynamics: decay_rate, zscore_10s, horizon_agreement = 3
# CNN embedding compressed (PCA 96→16) = 16
# TOTAL = 6+3+4+7+1+6+8+3+16 = 54
STATE_DIM = 54


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_date_from_filename(fname: str) -> Optional[str]:
    """Extract YYYYMMDD from filename."""
    m = re.search(r'(\d{8})', str(fname))
    return m.group(1) if m else None


def hours_et_from_index(position_frac: float) -> float:
    """
    Estimate ET hour from sample position fraction (0-1) within a day.
    Assumes RTH 9:30-16:00 ET (most signal activity during this window).
    """
    rth_start = 9.5
    rth_end = 16.0
    return rth_start + position_frac * (rth_end - rth_start)


def load_fold_predictions(
    cnn_pred_dir: Path,
    ptst_pred_dir: Optional[Path],
    fold_files: List[Path],
) -> Optional[dict]:
    """
    Load predictions for a set of fold files (train window or eval window).

    Returns dict with:
      - predictions: (N, 3) CNN-Mamba preds [1s/5s/10s]
      - labels:      (N, 3) future labels
      - embeddings:  (N, 96) CNN-Mamba embeddings
      - ptst_preds:  (N, 3) or None — PatchTST preds
      - hours_et:    (N,) estimated ET hour per sample
      - fold_indices: list of fold numbers included
      - dates:       list of date strings
    """
    all_preds, all_labels, all_embs, all_ptst = [], [], [], []
    all_hours = []
    fold_indices = []
    dates = []

    for fold_f in fold_files:
        try:
            d = np.load(str(fold_f), allow_pickle=True)
        except Exception as e:
            log.warning(f"Failed to load {fold_f}: {e}")
            continue

        preds = d['predictions'].astype(np.float32)   # (N, 3)
        labels = d['labels'].astype(np.float32)        # (N, 3)
        embs = d['embeddings'].astype(np.float32)      # (N, 96)

        N = len(preds)
        # Estimate ET hours from sample position within fold
        fracs = np.linspace(0, 1, N)
        hours = np.array([hours_et_from_index(f) for f in fracs], dtype=np.float32)

        # Load PatchTST predictions for this fold's date
        ptst = None
        oot_files = d.get('oot_files', None)
        date_str = None
        if oot_files is not None:
            oot_list = oot_files.tolist() if hasattr(oot_files, 'tolist') else [str(oot_files)]
            date_str = extract_date_from_filename(str(oot_list[0]))

        if date_str and ptst_pred_dir and ptst_pred_dir.exists():
            for ptst_f in sorted(ptst_pred_dir.glob("fold_*_oot_predictions.npz")):
                try:
                    pd_raw = np.load(str(ptst_f), allow_pickle=True)
                    ptst_oot = pd_raw.get('oot_files', [''])
                    if hasattr(ptst_oot, 'tolist'):
                        ptst_oot = ptst_oot.tolist()
                    ptst_date = extract_date_from_filename(str(ptst_oot[0]))
                    if ptst_date == date_str:
                        ptst_raw = pd_raw['predictions'].astype(np.float32)
                        if len(ptst_raw) == N:
                            ptst = ptst_raw
                        else:
                            # Interpolate to match CNN length
                            from scipy.interpolate import interp1d
                            x_src = np.linspace(0, 1, len(ptst_raw))
                            x_dst = np.linspace(0, 1, N)
                            ptst = np.zeros((N, ptst_raw.shape[1]), dtype=np.float32)
                            for j in range(ptst_raw.shape[1]):
                                f_i = interp1d(x_src, ptst_raw[:, j],
                                               kind='nearest', fill_value='extrapolate')
                                ptst[:, j] = f_i(x_dst)
                        break
                except Exception:
                    continue

        fold_num = int(re.search(r'fold_(\d+)', fold_f.stem).group(1))
        all_preds.append(preds)
        all_labels.append(labels)
        all_embs.append(embs)
        all_hours.append(hours)
        all_ptst.append(ptst if ptst is not None else np.zeros((N, 3), dtype=np.float32))
        fold_indices.append(fold_num)
        if date_str:
            dates.append(date_str)

    if not all_preds:
        return None

    result = {
        'predictions': np.concatenate(all_preds, axis=0),
        'labels':      np.concatenate(all_labels, axis=0),
        'embeddings':  np.concatenate(all_embs, axis=0),
        'ptst_preds':  np.concatenate(all_ptst, axis=0),
        'hours_et':    np.concatenate(all_hours, axis=0),
        'fold_indices': fold_indices,
        'dates':       dates,
        'n_samples':   sum(len(p) for p in all_preds),
    }
    return result


def fit_pca(embeddings: np.ndarray, n_components: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """PCA: reduce (N, 96) embeddings to (N, 16)."""
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]
    return mean, components


def project_pca(embeddings: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Apply fitted PCA projection."""
    return ((embeddings - mean) @ components.T).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """Current open position."""
    direction: int = POS_FLAT          # POS_FLAT / POS_LONG / POS_SHORT
    entry_step: int = 0                # step index when entered
    entry_pred: float = 0.0           # CNN pred at entry
    unrealized_pnl: float = 0.0       # in ticks (against current label path)
    mfe: float = 0.0                   # max favorable excursion (ticks)
    mae: float = 0.0                   # max adverse excursion (ticks, negative)
    sl_idx: int = SL_DEFAULT_IDX      # current SL level index
    entry_cost_paid: float = 0.0      # cost paid at entry (ticks)

    @property
    def is_flat(self) -> bool:
        return self.direction == POS_FLAT

    @property
    def hold_steps(self) -> int:
        return 0 if self.is_flat else (0)  # filled externally

    @property
    def sl_ticks(self) -> float:
        return float(SL_LEVELS[self.sl_idx])


@dataclass
class Trade:
    """Completed trade record."""
    direction: int
    entry_step: int
    exit_step: int
    pnl_ticks: float          # net after all costs
    gross_pnl_ticks: float    # before costs
    exit_type: str            # "passive", "market", "sl", "max_hold", "eod"
    hold_steps: int
    mfe: float
    mae: float
    entry_hour_et: float


class ExecEnvironment:
    """
    Offline execution environment replaying CNN-Mamba + PatchTST predictions.

    Each step = one prediction event. Agent chooses from 7 actions.
    Position management: agent explicitly enters/exits/adjusts SL.
    """

    def __init__(
        self,
        data: dict,
        pca_mean: Optional[np.ndarray] = None,
        pca_components: Optional[np.ndarray] = None,
    ):
        self.data = data
        self.N = data['n_samples']
        preds = data['predictions']
        labels = data['labels']

        # Confidence percentiles (for state features)
        abs_p10s = np.abs(preds[:, 2])
        self.p90 = float(np.percentile(abs_p10s, 90))
        self.p95 = float(np.percentile(abs_p10s, 95))
        self.p99 = float(np.percentile(abs_p10s, 99))

        # PCA for embeddings
        if pca_mean is None:
            self.pca_mean, self.pca_components = fit_pca(data['embeddings'], 16)
        else:
            self.pca_mean = pca_mean
            self.pca_components = pca_components
        self.emb_pca = project_pca(data['embeddings'], self.pca_mean, self.pca_components)

        # Precompute signal dynamics features
        self._precompute_signal_features()

        # Realized vol proxy (rolling std of 10s labels, window=100)
        self.rvol_fast = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            start = max(0, i - 100)
            self.rvol_fast[i] = float(np.nanstd(labels[start:i+1, 2]))

        # Flow imbalance proxy from signal cross-sectional consistency
        self.flow_imb = np.zeros(self.N, dtype=np.float32)
        sign_agreement = np.sign(preds[:, 0]) * np.sign(preds[:, 2])
        for i in range(self.N):
            start = max(0, i - 50)
            self.flow_imb[i] = float(np.mean(sign_agreement[start:i+1]))

        # Momentum (rolling mean of 10s pred, window=20)
        self.momentum = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            start = max(0, i - 20)
            self.momentum[i] = float(np.mean(preds[start:i+1, 2]))

        self.reset()

    def _precompute_signal_features(self):
        """Precompute signal zscore and decay rate for all samples."""
        preds = self.data['predictions']
        N = self.N
        pred_10s = preds[:, 2]

        # Rolling z-score (window=200)
        self.pred_zscore = np.zeros(N, dtype=np.float32)
        for i in range(N):
            start = max(0, i - 200)
            chunk = pred_10s[start:i+1]
            if len(chunk) >= 10:
                mean = chunk.mean()
                std = chunk.std()
                if std > 1e-8:
                    self.pred_zscore[i] = float(np.clip((pred_10s[i] - mean) / std, -5, 5))

        # Decay rate (slope of |pred| over last 5 samples)
        self.decay_rate = np.zeros(N, dtype=np.float32)
        abs_p = np.abs(pred_10s)
        for i in range(5, N):
            y = abs_p[i-5:i+1].astype(np.float64)
            x = np.arange(6, dtype=np.float64)
            slope = np.polyfit(x, y, 1)[0]
            self.decay_rate[i] = float(np.clip(slope, -0.5, 0.5))

        # Horizon agreement: fraction of horizons with same sign as 10s
        signs = np.sign(preds)
        self.horizon_agreement = np.mean(
            signs == signs[:, 2:3], axis=1
        ).astype(np.float32)

    def reset(self, day_start_idx: int = 0, day_end_idx: Optional[int] = None):
        """Reset for new episode (one trading day or full dataset)."""
        self.cursor = day_start_idx
        self.day_end_idx = day_end_idx if day_end_idx is not None else self.N

        # Position state
        self.position = Position()
        self.current_hold_steps = 0

        # Daily circuit breaker
        self.daily_pnl_ticks = 0.0
        self.daily_halted = False

        # Trade history
        self.trades: List[Trade] = []
        self.recent_pnl: deque = deque(maxlen=5)
        self.win_streak = 0
        self.loss_streak = 0
        self.cum_pnl = 0.0

        # Action history (for logging)
        self.action_counts = Counter()

    def get_state(self) -> Optional[np.ndarray]:
        """Build state vector at current cursor. Returns (STATE_DIM,) or None if done."""
        if self.cursor >= self.day_end_idx or self.daily_halted:
            return None

        i = self.cursor
        preds = self.data['predictions']
        ptst = self.data['ptst_preds']
        hours = self.data['hours_et']

        # ── 1. Signal features (6) ──────────────────────────────────────────────
        cnn_1s  = float(preds[i, 0])
        cnn_5s  = float(preds[i, 1])
        cnn_10s = float(preds[i, 2])
        ptst_1s  = float(ptst[i, 0])
        ptst_5s  = float(ptst[i, 1])
        ptst_10s = float(ptst[i, 2])

        # ── 2. Confidence percentile flags (3) ─────────────────────────────────
        abs_10s = abs(cnn_10s)
        conf_p90 = float(abs_10s >= self.p90)
        conf_p95 = float(abs_10s >= self.p95)
        conf_p99 = float(abs_10s >= self.p99)

        # ── 3. Market features (4) ─────────────────────────────────────────────
        rvol      = float(self.rvol_fast[i])
        flow_imb  = float(self.flow_imb[i])
        momentum  = float(self.momentum[i])
        rvol_norm = float(np.clip(rvol / max(self.rvol_fast.mean(), 1e-8), 0, 5))

        # ── 4. Position features (7) ────────────────────────────────────────────
        pos_flat  = float(self.position.direction == POS_FLAT)
        pos_long  = float(self.position.direction == POS_LONG)
        pos_short = float(self.position.direction == POS_SHORT)
        unrealized = float(np.clip(self.position.unrealized_pnl / 10.0, -2, 2))
        hold_frac  = float(min(self.current_hold_steps / MAX_HOLD_STEPS, 1.0))
        mfe_norm   = float(np.clip(self.position.mfe / 10.0, 0, 2))
        mae_norm   = float(np.clip(self.position.mae / 10.0, -2, 0))

        # ── 5. SL level (1) ────────────────────────────────────────────────────
        sl_norm = float(self.position.sl_idx / SL_MAX_IDX)

        # ── 6. Time features (6) ───────────────────────────────────────────────
        hour = float(hours[i])
        tod_sin = float(np.sin(2 * np.pi * hour / 24.0))
        tod_cos = float(np.cos(2 * np.pi * hour / 24.0))
        is_open = float(9.5 <= hour < 10.5)
        is_core = float(10.5 <= hour < 15.0)
        mins_since_open  = float(np.clip((hour - 9.5) * 60, -60, 420))
        mins_until_close = float(np.clip((16.0 - hour) * 60, -60, 420))

        # Normalize time features
        mins_since_open_norm  = mins_since_open / 420.0
        mins_until_close_norm = mins_until_close / 420.0

        # ── 7. Trade history features (8) ──────────────────────────────────────
        recent = list(self.recent_pnl) + [0.0] * (5 - len(self.recent_pnl))
        recent_pnl_norm = [float(np.clip(p / 5.0, -2, 2)) for p in recent]
        win_streak_norm  = float(min(self.win_streak / 10.0, 1.0))
        loss_streak_norm = float(min(self.loss_streak / 10.0, 1.0))
        cum_pnl_norm     = float(np.clip(self.cum_pnl / 50.0, -2, 2))

        # ── 8. Signal dynamics (3) ─────────────────────────────────────────────
        decay_r   = float(np.clip(self.decay_rate[i] * 10, -2, 2))
        zscore    = float(np.clip(self.pred_zscore[i], -3, 3))
        horiz_agr = float(self.horizon_agreement[i])

        # ── 9. CNN embedding PCA (16) ──────────────────────────────────────────
        emb_pca = self.emb_pca[i].astype(np.float32)

        # Assemble state vector
        state = np.array([
            # Signal (6)
            cnn_1s, cnn_5s, cnn_10s,
            ptst_1s, ptst_5s, ptst_10s,
            # Confidence (3)
            conf_p90, conf_p95, conf_p99,
            # Market (4)
            rvol_norm, flow_imb, momentum, rvol,
            # Position (7)
            pos_flat, pos_long, pos_short,
            unrealized, hold_frac, mfe_norm, mae_norm,
            # SL (1)
            sl_norm,
            # Time (6)
            tod_sin, tod_cos, is_open, is_core,
            mins_since_open_norm, mins_until_close_norm,
            # Trade history (8)
            *recent_pnl_norm,
            win_streak_norm, loss_streak_norm, cum_pnl_norm,
            # Signal dynamics (3)
            decay_r, zscore, horiz_agr,
        ], dtype=np.float32)

        # Append PCA embedding (16)
        state = np.concatenate([state, emb_pca])

        assert len(state) == STATE_DIM, f"State dim mismatch: {len(state)} != {STATE_DIM}"
        return state

    def _simulate_unrealized_pnl(self) -> float:
        """
        Estimate current unrealized PnL using future label as price proxy.
        Uses 1s label (shortest available) for current price estimate.
        """
        if self.position.is_flat or self.cursor >= self.N:
            return 0.0

        label = self.data['labels'][self.cursor]
        # 1s move in direction of position
        move_ticks = float(label[0]) * LABEL_TO_TICKS * self.position.direction
        return move_ticks

    def _is_tod_gated(self) -> bool:
        """Return True if current time is in the ToD gate (no entries allowed)."""
        if self.cursor >= self.N:
            return True
        hour = float(self.data['hours_et'][self.cursor])
        return TOD_GATE_START_HOUR <= hour <= TOD_GATE_END_HOUR

    def step(self, action: int) -> Tuple[float, dict]:
        """
        Execute action, advance cursor, return (reward, info).

        FIFO fill model:
          - ENTER via limit: filled at best bid/ask price (passive cost = 0.376t)
          - EXIT via limit: passive cost = 0.376t
          - EXIT via market: cost = 0.376t (HC #231(A): commission only)
          - SL hit: treated as market exit (0.376t cost)
          - Max hold: treated as market exit (0.376t cost)
        """
        if self.cursor >= self.day_end_idx:
            return 0.0, {"done": True}

        info = {"done": False, "action": action, "traded": False}
        reward = 0.0
        i = self.cursor

        # ── Check daily circuit breaker ────────────────────────────────────────
        if self.daily_pnl_ticks <= MAX_DAILY_LOSS and not self.position.is_flat:
            # Force close position via market
            reward, close_info = self._close_position("eod_circuit_break", market=True)
            self.daily_halted = True
            info.update(close_info)
            self.cursor += 1
            return reward, info

        # ── Update unrealized PnL if in position ───────────────────────────────
        if not self.position.is_flat:
            self.current_hold_steps += 1
            move = self._simulate_unrealized_pnl()
            self.position.unrealized_pnl = move
            self.position.mfe = max(self.position.mfe, move)
            self.position.mae = min(self.position.mae, move)

            # Check SL hit
            if move <= -self.position.sl_ticks:
                reward, close_info = self._close_position("sl", market=True)
                info.update(close_info)
                self.cursor += 1
                return reward, info

            # Check max hold
            if self.current_hold_steps >= MAX_HOLD_STEPS:
                reward, close_info = self._close_position("max_hold", market=True)
                info.update(close_info)
                self.cursor += 1
                return reward, info

        # ── Execute action ─────────────────────────────────────────────────────
        self.action_counts[action] += 1

        if action == ACTION_WAIT:
            # Per-step holding penalty
            if not self.position.is_flat:
                reward = REWARD_HOLD_PENALTY
            else:
                reward = REWARD_WAIT_PENALTY

        elif action in (ACTION_ENTER_LONG, ACTION_ENTER_SHORT):
            if not self.position.is_flat:
                # Already in position — treat as WAIT
                reward = REWARD_HOLD_PENALTY
            elif self._is_tod_gated():
                # ToD gate: no entries during open volatility window
                reward = REWARD_WAIT_PENALTY
            else:
                direction = POS_LONG if action == ACTION_ENTER_LONG else POS_SHORT
                self.position = Position(
                    direction=direction,
                    entry_step=i,
                    entry_pred=float(self.data['predictions'][i, 2]),
                    sl_idx=SL_DEFAULT_IDX,
                    entry_cost_paid=COST_PASSIVE_TICKS,
                )
                self.current_hold_steps = 0
                reward = REWARD_HOLD_PENALTY  # start accumulating hold penalty
                info["traded"] = True
                info["entry_direction"] = direction

        elif action == ACTION_EXIT_PASSIVE:
            if self.position.is_flat:
                reward = REWARD_WAIT_PENALTY
            else:
                reward, close_info = self._close_position("passive", market=False)
                info.update(close_info)

        elif action == ACTION_EXIT_MARKET:
            if self.position.is_flat:
                reward = REWARD_WAIT_PENALTY
            else:
                reward, close_info = self._close_position("market", market=True)
                info.update(close_info)

        elif action == ACTION_TIGHTEN_STOP:
            if self.position.is_flat:
                reward = REWARD_WAIT_PENALTY
            else:
                if self.position.sl_idx > SL_MIN_IDX:
                    self.position.sl_idx -= 1
                reward = 0.0  # neutral

        elif action == ACTION_WIDEN_STOP:
            if self.position.is_flat:
                reward = REWARD_WAIT_PENALTY
            else:
                if self.position.sl_idx < SL_MAX_IDX:
                    self.position.sl_idx += 1
                reward = 0.0  # neutral

        # ── Sortino shaping (on completed trades) ──────────────────────────────
        if len(self.trades) >= 10 and info.get("traded"):
            sortino = self._compute_sortino()
            reward += REWARD_SORTINO_WEIGHT * sortino

        self.cursor += 1
        info["position_flat"] = self.position.is_flat
        info["cum_pnl_ticks"] = self.cum_pnl
        return reward, info

    def _close_position(self, exit_type: str, market: bool) -> Tuple[float, dict]:
        """Close current position, record trade, return (reward, info)."""
        if self.position.is_flat:
            return 0.0, {}

        i = self.cursor
        label = self.data['labels'][i]

        # Gross P&L from labels (price proxy)
        if self.current_hold_steps <= 1:
            gross_ticks = float(label[0]) * LABEL_TO_TICKS * self.position.direction
        elif self.current_hold_steps <= 5:
            alpha = (self.current_hold_steps - 1) / 4.0
            gross_ticks = (
                (1 - alpha) * float(label[0]) * LABEL_TO_TICKS * self.position.direction +
                alpha * float(label[1]) * LABEL_TO_TICKS * self.position.direction
            )
        else:
            alpha = min((self.current_hold_steps - 5) / 5.0, 1.0)
            gross_ticks = (
                (1 - alpha) * float(label[1]) * LABEL_TO_TICKS * self.position.direction +
                alpha * float(label[2]) * LABEL_TO_TICKS * self.position.direction
            )

        # Exit cost
        exit_cost = COST_MARKET_TICKS if market else COST_PASSIVE_TICKS
        total_cost = self.position.entry_cost_paid + exit_cost
        net_ticks = gross_ticks - total_cost

        trade = Trade(
            direction=self.position.direction,
            entry_step=self.position.entry_step,
            exit_step=i,
            pnl_ticks=net_ticks,
            gross_pnl_ticks=gross_ticks,
            exit_type=exit_type,
            hold_steps=self.current_hold_steps,
            mfe=self.position.mfe,
            mae=self.position.mae,
            entry_hour_et=float(self.data['hours_et'][self.position.entry_step])
                          if self.position.entry_step < self.N else 9.5,
        )
        self.trades.append(trade)
        self.recent_pnl.append(net_ticks)
        self.cum_pnl += net_ticks
        self.daily_pnl_ticks += net_ticks

        if net_ticks > 0:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        # Reset position
        self.position = Position()
        self.current_hold_steps = 0

        info = {
            "traded": True,
            "pnl_ticks": net_ticks,
            "gross_ticks": gross_ticks,
            "exit_type": exit_type,
            "hold_steps": trade.hold_steps,
        }
        return net_ticks, info

    def _compute_sortino(self) -> float:
        """Rolling Sortino ratio over recent trades."""
        if len(self.trades) < 2:
            return 0.0
        recent = self.trades[-SORTINO_WINDOW:]
        pnls = np.array([t.pnl_ticks for t in recent])
        mean_pnl = pnls.mean()
        downside = pnls[pnls < 0]
        if len(downside) == 0:
            return min(10.0, mean_pnl * 2)
        down_std = downside.std()
        if down_std < 1e-8:
            return 10.0
        return float(np.clip(mean_pnl / down_std, -10.0, 10.0))

    def get_episode_stats(self) -> dict:
        """Return episode performance statistics."""
        if not self.trades:
            return {"n_trades": 0, "sortino": 0.0, "win_rate": 0.0,
                    "total_pnl_ticks": 0.0, "total_pnl_usd": 0.0,
                    "profit_factor": 0.0, "avg_hold_steps": 0.0}

        pnls = np.array([t.pnl_ticks for t in self.trades])
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]

        sortino = self._compute_sortino()
        pf = float(wins.sum() / max(abs(losses.sum()), 1e-8)) if len(wins) > 0 else 0.0

        # Sharpe (annualized proxy)
        if len(pnls) >= 2 and pnls.std() > 1e-8:
            sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # Breakdown by direction
        long_trades = [t for t in self.trades if t.direction == POS_LONG]
        short_trades = [t for t in self.trades if t.direction == POS_SHORT]

        # Action distribution
        total_actions = sum(self.action_counts.values())
        action_pcts = {
            ACTION_NAMES[a]: round(100 * c / max(total_actions, 1), 1)
            for a, c in self.action_counts.items()
        }

        return {
            "n_trades": len(self.trades),
            "win_rate": float(100 * (pnls > 0).mean()),
            "total_pnl_ticks": float(pnls.sum()),
            "total_pnl_usd": float(pnls.sum() * ES_TICK_VALUE),
            "avg_pnl_ticks": float(pnls.mean()),
            "sortino": float(sortino),
            "sharpe": float(sharpe),
            "profit_factor": float(pf),
            "avg_hold_steps": float(np.mean([t.hold_steps for t in self.trades])),
            "n_long": len(long_trades),
            "n_short": len(short_trades),
            "long_pnl": float(sum(t.pnl_ticks for t in long_trades)),
            "short_pnl": float(sum(t.pnl_ticks for t in short_trades)),
            "exit_types": dict(Counter(t.exit_type for t in self.trades)),
            "action_pcts": action_pcts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

class ExecPolicyV3(nn.Module):
    """
    PPO Actor-Critic for flat 7-action execution policy.

    Architecture:
      - Shared backbone: 3-layer MLP + LayerNorm + GELU
      - Policy head: linear → 7 logits with learnable temperature
      - Value head: 2-layer MLP
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 128):
        super().__init__()

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

        # Policy head
        self.policy_head = nn.Linear(64, N_ACTIONS)
        self.temperature = nn.Parameter(torch.ones(1))

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small init for policy head → near-uniform initial distribution
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward(self, state: torch.Tensor):
        features = self.backbone(state)
        temp = torch.clamp(self.temperature, 0.1, 5.0)
        logits = self.policy_head(features) / temp
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False,
                   action_mask: Optional[torch.Tensor] = None):
        """
        Sample action from policy.

        action_mask: boolean tensor (batch, N_ACTIONS) — True = allowed.
        Returns (action, log_prob, value, entropy).
        """
        logits, value = self.forward(state)

        if action_mask is not None:
            # Mask invalid actions with large negative value
            logits = logits.masked_fill(~action_mask, -1e9)

        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(-1)
        else:
            action = dist.sample()

        return action, dist.log_prob(action), value, dist.entropy()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate log probs, values, entropy for PPO update."""
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


# ═══════════════════════════════════════════════════════════════════════════════
# BEHAVIORAL CLONING WARM-START
# ═══════════════════════════════════════════════════════════════════════════════

def behavioral_cloning_warmstart(
    policy: ExecPolicyV3,
    env: ExecEnvironment,
    n_steps: int = 2000,
    lr: float = 1e-3,
    device: str = "cuda",
) -> None:
    """
    Pre-train policy via behavioral cloning from Optuna MLP v2 best static params.

    The expert policy (Optuna v2 best):
      - Enter only when abs(pred_10s) >= p90 (top 10% confidence)
      - For shorts: prefer entering; for longs: enter if p90
      - Exit via passive limit after ~5 steps (cancel_decay=45 windows → ~5 signal steps)
      - Tighten stop when position is profitable (MFE > 0)

    We generate expert actions from this heuristic and clone them.
    """
    log.info(f"[BC Warm-start] Generating {n_steps} expert demonstrations...")
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    states_buf = []
    actions_buf = []

    env.reset()
    for step_i in range(min(n_steps * 3, env.N)):
        state = env.get_state()
        if state is None:
            break

        i = env.cursor
        pred_10s = float(env.data['predictions'][i, 2])
        abs_pred = abs(pred_10s)
        direction = int(np.sign(pred_10s)) if pred_10s != 0 else 1
        in_position = not env.position.is_flat
        hold = env.current_hold_steps

        # Expert action logic (Optuna v2 best params translated):
        # 1. If flat + high confidence + not ToD-gated → enter short-biased
        # 2. If in position + hold >= 5 → passive exit
        # 3. If in position + profitable + hold >= 3 → passive exit
        # 4. If in position + MFE > 2t → tighten stop
        # 5. Otherwise → wait

        expert_action = ACTION_WAIT

        if not in_position and not env._is_tod_gated():
            if abs_pred >= env.p90:
                if direction < 0:
                    expert_action = ACTION_ENTER_SHORT
                else:
                    expert_action = ACTION_ENTER_LONG
        elif in_position:
            unrealized = env.position.unrealized_pnl
            if hold >= 5:
                expert_action = ACTION_EXIT_PASSIVE
            elif unrealized > 2.0 and hold >= 3:
                expert_action = ACTION_EXIT_PASSIVE
            elif env.position.mfe > 3.0 and env.position.sl_idx > SL_MIN_IDX:
                expert_action = ACTION_TIGHTEN_STOP
            elif hold >= 2 and unrealized < -1.0:
                expert_action = ACTION_EXIT_MARKET
            else:
                expert_action = ACTION_WAIT

        states_buf.append(state)
        actions_buf.append(expert_action)

        # Step with expert action
        env.step(expert_action)

        if len(states_buf) >= n_steps:
            break

    if not states_buf:
        log.warning("[BC Warm-start] No demonstrations generated, skipping.")
        return

    # Train
    states_t = torch.FloatTensor(np.array(states_buf)).to(device)
    actions_t = torch.LongTensor(actions_buf).to(device)

    n_epochs = 20
    bs = 256
    total_loss = 0.0
    n_batches = 0

    for epoch in range(n_epochs):
        perm = torch.randperm(len(states_t))
        for start in range(0, len(states_t), bs):
            idx = perm[start:start + bs]
            logits, _ = policy(states_t[idx])
            loss = criterion(logits, actions_t[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    action_dist = Counter(actions_buf)
    log.info(f"[BC Warm-start] Done. Avg loss: {avg_loss:.4f}")
    log.info(f"  Expert action distribution: {dict(action_dist.most_common())}")


# ═══════════════════════════════════════════════════════════════════════════════
# PPO TRAINER
# ═══════════════════════════════════════════════════════════════════════════════

class PPOTrainerV3:
    """PPO trainer for the flat 7-action execution agent."""

    def __init__(
        self,
        env: ExecEnvironment,
        lr: float = PPO_LR,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_LAMBDA,
        clip_eps: float = PPO_CLIP_EPS,
        value_coeff: float = PPO_VALUE_COEFF,
        entropy_coeff: float = ENTROPY_COEFF_BASE,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        batch_size: int = PPO_BATCH_SIZE,
        n_epochs: int = PPO_EPOCHS,
        device: str = "cuda",
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device

        self.policy = ExecPolicyV3(state_dim=STATE_DIM, hidden_dim=128).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        n_params = sum(p.numel() for p in self.policy.parameters())
        log.info(f"[PPO] Policy v3: {n_params:,} params (state_dim={STATE_DIM}, actions={N_ACTIONS})")

    def _lr_warmup(self, iteration: int, base_lr: float) -> float:
        if iteration <= PPO_WARMUP_ITERS:
            lr = base_lr * (iteration / PPO_WARMUP_ITERS)
        else:
            lr = base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def collect_rollout(self, n_steps: int = PPO_ROLLOUT_STEPS) -> dict:
        """Collect experience buffer."""
        states, actions, rewards = [], [], []
        values, log_probs, dones = [], [], []

        steps_collected = 0
        while steps_collected < n_steps:
            state = self.env.get_state()
            if state is None:
                self.env.reset()
                state = self.env.get_state()
                if state is None:
                    break

            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, lp, value, entropy = self.policy.get_action(state_t)

            reward, info = self.env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(lp.item())
            dones.append(info.get("done", False))

            steps_collected += 1

        return {
            "states":    np.array(states, dtype=np.float32),
            "actions":   np.array(actions, dtype=np.int64),
            "rewards":   np.array(rewards, dtype=np.float32),
            "values":    np.array(values, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "dones":     np.array(dones, dtype=bool),
        }

    def compute_gae(self, rewards, values, dones) -> Tuple[np.ndarray, np.ndarray]:
        """Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                next_val = values[t + 1] if t + 1 < n else 0.0
                delta = rewards[t] + self.gamma * next_val - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout: dict) -> dict:
        """PPO update step."""
        advantages, returns = self.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t  = torch.FloatTensor(rollout["states"]).to(self.device)
        actions_t = torch.LongTensor(rollout["actions"]).to(self.device)
        old_lp_t  = torch.FloatTensor(rollout["log_probs"]).to(self.device)
        adv_t     = torch.FloatTensor(advantages).to(self.device)
        ret_t     = torch.FloatTensor(returns).to(self.device)

        n = len(states_t)
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]

                new_lp, new_val, entropy = self.policy.evaluate_actions(
                    states_t[idx], actions_t[idx]
                )

                ratio = torch.exp(new_lp - old_lp_t[idx])
                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                policy_loss = -torch.min(
                    ratio * adv_t[idx], clipped * adv_t[idx]
                ).mean()

                value_loss = F.mse_loss(new_val, ret_t[idx])
                entropy_loss = -self.entropy_coeff * entropy.mean()

                loss = policy_loss + self.value_coeff * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss":  total_value_loss  / max(n_updates, 1),
            "entropy":     total_entropy     / max(n_updates, 1),
            "temperature": float(self.policy.temperature.item()),
        }

    def train_on_window(
        self,
        n_iterations: int = 200,
        rollout_steps: int = PPO_ROLLOUT_STEPS,
        eval_every: int = 20,
        save_path: Optional[Path] = None,
        fold_label: str = "",
        base_lr: float = PPO_LR,
    ) -> List[dict]:
        """Train PPO on a single walk-forward window."""
        results_log = []
        best_sortino = -float("inf")
        best_state_dict = None

        log.info(f"[PPO{fold_label}] Training {n_iterations} iters × {rollout_steps} steps")

        for iteration in range(1, n_iterations + 1):
            lr = self._lr_warmup(iteration, base_lr)
            self.env.reset()
            rollout = self.collect_rollout(rollout_steps)
            losses = self.update(rollout)

            # Episode stats
            stats = self.env.get_episode_stats()
            n_trades = stats["n_trades"]
            sortino   = stats["sortino"]
            win_rate  = stats.get("win_rate", 0.0)
            total_pnl = stats.get("total_pnl_ticks", 0.0)

            result = {
                "iteration": iteration,
                "lr": lr,
                **stats,
                **losses,
            }
            results_log.append(result)

            if iteration % eval_every == 0 or iteration == 1:
                ac = stats.get("action_pcts", {})
                log.info(
                    f"[PPO{fold_label}] Iter {iteration:4d} | "
                    f"LR: {lr:.1e} | "
                    f"Trades: {n_trades:4d} | "
                    f"WR: {win_rate:.1f}% | "
                    f"PnL: {total_pnl:+.1f}t (${total_pnl * ES_TICK_VALUE:+,.0f}) | "
                    f"Sortino: {sortino:+.3f} | "
                    f"PF: {stats.get('profit_factor', 0):.2f} | "
                    f"π_loss: {losses['policy_loss']:.4f} | "
                    f"Temp: {losses['temperature']:.2f}"
                )
                if ac:
                    log.info(f"  Actions: {ac}")

                # Action collapse check
                wait_pct = ac.get("WAIT", 0)
                if wait_pct > 95:
                    log.warning(f"  [COLLAPSE WARNING] WAIT = {wait_pct:.0f}% of actions!")

            # Save best
            if sortino > best_sortino and n_trades >= 5:
                best_sortino = sortino
                best_state_dict = {k: v.cpu().clone()
                                   for k, v in self.policy.state_dict().items()}
                if save_path:
                    ckpt = {
                        "policy_state_dict": best_state_dict,
                        "iteration": iteration,
                        "sortino": sortino,
                        "win_rate": win_rate,
                        "n_trades": n_trades,
                        "total_pnl_ticks": total_pnl,
                        "state_dim": STATE_DIM,
                        "n_actions": N_ACTIONS,
                        "pca_mean": self.env.pca_mean,
                        "pca_components": self.env.pca_components,
                        "confidence_tiers": {
                            "p90": self.env.p90,
                            "p95": self.env.p95,
                            "p99": self.env.p99,
                        },
                        "version": "v3",
                    }
                    torch.save(ckpt, str(save_path / f"best{fold_label}.pt"))

        # Restore best
        if best_state_dict is not None:
            self.policy.load_state_dict(best_state_dict)

        return results_log

    def evaluate(self, deterministic: bool = True) -> dict:
        """Run deterministic evaluation on full env data."""
        self.policy.eval()
        self.env.reset()

        while True:
            state = self.env.get_state()
            if state is None:
                break
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action, _, _, _ = self.policy.get_action(state_t, deterministic=deterministic)
            self.env.step(action.item())

        stats = self.env.get_episode_stats()
        self.policy.train()
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def run_walkforward(
    cnn_pred_dir: Path,
    ptst_pred_dir: Optional[Path],
    output_dir: Path,
    n_iterations: int = 200,
    rollout_steps: int = PPO_ROLLOUT_STEPS,
    device: str = "cuda",
    bc_warmstart: bool = True,
    n_bc_steps: int = 2000,
    experiment_name: str = "rl_execution_agent",
):
    """
    Walk-forward sliding window training.
      - Sort all available fold files by date
      - For each window: train on 60 days, eval on next 5 days
      - SLIDING: drop oldest day when advancing
      - Log each fold to MLflow
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all fold prediction files
    fold_files = sorted(cnn_pred_dir.glob("fold_*_oot_predictions.npz"))
    if not fold_files:
        raise ValueError(f"No fold prediction files found in {cnn_pred_dir}")

    log.info(f"Found {len(fold_files)} fold files in {cnn_pred_dir}")

    # Sort by fold number (proxy for date order)
    def fold_num(f: Path) -> int:
        m = re.search(r'fold_(\d+)', f.stem)
        return int(m.group(1)) if m else 0

    fold_files = sorted(fold_files, key=fold_num)
    n_folds = len(fold_files)

    # Walk-forward: need at least WF_TRAIN_DAYS + WF_EVAL_DAYS folds
    # Each fold = 1 day in the sliding window
    min_folds = WF_TRAIN_DAYS + WF_EVAL_DAYS
    if n_folds < min_folds:
        log.warning(f"Only {n_folds} folds available (need {min_folds} for full 60d+5d WF). "
                    f"Using all available as train window.")
        train_end = max(n_folds - WF_EVAL_DAYS, 1)
        windows = [(fold_files[:train_end], fold_files[train_end:])]
    else:
        # Sliding windows
        windows = []
        n_windows = n_folds - min_folds + 1
        for w in range(0, n_windows):
            train_folds = fold_files[w:w + WF_TRAIN_DAYS]
            eval_folds  = fold_files[w + WF_TRAIN_DAYS:w + WF_TRAIN_DAYS + WF_EVAL_DAYS]
            windows.append((train_folds, eval_folds))
        log.info(f"Walk-forward: {len(windows)} windows ({WF_TRAIN_DAYS}d train, {WF_EVAL_DAYS}d eval each)")

    # MLflow setup
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(experiment_name)
            mlflow_run = mlflow.start_run(
                run_name=f"rl_exec_v3_wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            mlflow.log_params({
                "model": "rl_execution_agent_v3",
                "n_windows": len(windows),
                "train_days": WF_TRAIN_DAYS,
                "eval_days": WF_EVAL_DAYS,
                "n_actions": N_ACTIONS,
                "state_dim": STATE_DIM,
                "es_tick_value": ES_TICK_VALUE,
                "es_commission_ticks": ES_COMMISSION_TICKS,
                "cost_passive": COST_PASSIVE_TICKS,
                "cost_market": COST_MARKET_TICKS,
                "max_daily_loss_ticks": MAX_DAILY_LOSS,
                "tod_gate": f"{TOD_GATE_START_HOUR:.2f}-{TOD_GATE_END_HOUR:.2f}ET",
                "ppo_lr": PPO_LR,
                "ppo_gamma": PPO_GAMMA,
                "ppo_clip_eps": PPO_CLIP_EPS,
                "ppo_epochs": PPO_EPOCHS,
                "ppo_rollout_steps": rollout_steps,
                "n_iterations": n_iterations,
                "bc_warmstart": bc_warmstart,
                "device": device,
                "window_type": "sliding",
                "label_to_ticks": LABEL_TO_TICKS,
                "sl_levels": SL_LEVELS.tolist(),
                "max_hold_steps": MAX_HOLD_STEPS,
            })
            log.info(f"[MLflow] Run started: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"[MLflow] Setup failed: {e}")

    all_fold_results = []
    best_overall_sortino = -float("inf")
    best_overall_state = None

    try:
        for window_idx, (train_folds, eval_folds) in enumerate(windows):
            fold_label = f"_w{window_idx:02d}"
            train_dates = [extract_date_from_filename(f.stem) for f in train_folds]
            eval_dates  = [extract_date_from_filename(f.stem) for f in eval_folds]
            log.info(f"\n{'='*70}")
            log.info(f"WINDOW {window_idx+1}/{len(windows)}")
            log.info(f"  Train: {len(train_folds)} folds "
                     f"({[d for d in train_dates[:3] if d]}... → "
                     f"{[d for d in train_dates[-1:] if d]})")
            log.info(f"  Eval:  {len(eval_folds)} folds "
                     f"({[d for d in eval_dates if d]})")
            log.info(f"{'='*70}")

            # Load train data
            train_data = load_fold_predictions(cnn_pred_dir, ptst_pred_dir, train_folds)
            if train_data is None:
                log.warning(f"  Window {window_idx}: Failed to load train data, skipping.")
                continue
            log.info(f"  Train: {train_data['n_samples']:,} samples loaded")

            # Build train environment
            train_env = ExecEnvironment(train_data)

            # Create trainer
            trainer = PPOTrainerV3(
                env=train_env,
                lr=PPO_LR,
                device=device,
                batch_size=PPO_BATCH_SIZE,
                n_epochs=PPO_EPOCHS,
            )

            # Behavioral cloning warm-start (window 0 only, or first window)
            if bc_warmstart and window_idx == 0:
                behavioral_cloning_warmstart(
                    trainer.policy, train_env,
                    n_steps=n_bc_steps,
                    lr=1e-3,
                    device=device,
                )

            # Train
            train_results = trainer.train_on_window(
                n_iterations=n_iterations,
                rollout_steps=rollout_steps,
                eval_every=20,
                save_path=output_dir,
                fold_label=fold_label,
                base_lr=PPO_LR,
            )

            # Evaluate on held-out eval folds
            eval_data = load_fold_predictions(cnn_pred_dir, ptst_pred_dir, eval_folds)
            eval_stats = {}
            if eval_data is not None:
                log.info(f"  Eval: {eval_data['n_samples']:,} samples")
                eval_env = ExecEnvironment(
                    eval_data,
                    pca_mean=train_env.pca_mean,
                    pca_components=train_env.pca_components,
                )
                # Use same policy for eval
                trainer.env = eval_env
                eval_stats = trainer.evaluate(deterministic=True)

                log.info(
                    f"\n  EVAL WINDOW {window_idx+1}: "
                    f"Trades={eval_stats['n_trades']} | "
                    f"WR={eval_stats.get('win_rate', 0):.1f}% | "
                    f"PnL={eval_stats.get('total_pnl_ticks', 0):+.1f}t "
                    f"(${eval_stats.get('total_pnl_usd', 0):+,.0f}) | "
                    f"Sortino={eval_stats.get('sortino', 0):+.3f} | "
                    f"PF={eval_stats.get('profit_factor', 0):.2f} | "
                    f"Sharpe={eval_stats.get('sharpe', 0):+.3f}"
                )
                if eval_stats.get("exit_types"):
                    log.info(f"  Exit types: {eval_stats['exit_types']}")
                if eval_stats.get("action_pcts"):
                    log.info(f"  Action pcts: {eval_stats['action_pcts']}")

            # Track best overall
            eval_sortino = eval_stats.get("sortino", train_results[-1]["sortino"] if train_results else 0.0)
            if eval_sortino > best_overall_sortino and eval_stats.get("n_trades", 0) >= 5:
                best_overall_sortino = eval_sortino
                best_overall_state = {
                    k: v.cpu().clone()
                    for k, v in trainer.policy.state_dict().items()
                }
                torch.save({
                    "policy_state_dict": best_overall_state,
                    "window_idx": window_idx,
                    "eval_sortino": eval_sortino,
                    "eval_stats": eval_stats,
                    "state_dim": STATE_DIM,
                    "n_actions": N_ACTIONS,
                    "pca_mean": train_env.pca_mean,
                    "pca_components": train_env.pca_components,
                    "confidence_tiers": {
                        "p90": train_env.p90,
                        "p95": train_env.p95,
                        "p99": train_env.p99,
                    },
                    "version": "v3",
                }, str(output_dir / "best_fold_overall.pt"))
                log.info(f"  ** NEW BEST OVERALL: Sortino={eval_sortino:.4f}, saved.")

            # Log to MLflow
            if MLFLOW_AVAILABLE and mlflow_run:
                try:
                    last = train_results[-1] if train_results else {}
                    mlflow_metrics = {
                        f"w{window_idx:02d}_train_sortino": last.get("sortino", 0),
                        f"w{window_idx:02d}_train_pnl_ticks": last.get("total_pnl_ticks", 0),
                        f"w{window_idx:02d}_train_win_rate": last.get("win_rate", 0),
                        f"w{window_idx:02d}_train_n_trades": last.get("n_trades", 0),
                        f"w{window_idx:02d}_eval_sortino": eval_stats.get("sortino", 0),
                        f"w{window_idx:02d}_eval_pnl_ticks": eval_stats.get("total_pnl_ticks", 0),
                        f"w{window_idx:02d}_eval_win_rate": eval_stats.get("win_rate", 0),
                        f"w{window_idx:02d}_eval_n_trades": eval_stats.get("n_trades", 0),
                        f"w{window_idx:02d}_eval_profit_factor": eval_stats.get("profit_factor", 0),
                        f"w{window_idx:02d}_eval_sharpe": eval_stats.get("sharpe", 0),
                    }
                    mlflow.log_metrics(mlflow_metrics, step=window_idx)
                except Exception as e:
                    log.warning(f"[MLflow] Metrics logging failed: {e}")

            fold_summary = {
                "window_idx": window_idx,
                "train_folds": [fold_num(f) for f in train_folds],
                "eval_folds":  [fold_num(f) for f in eval_folds],
                "train_final": train_results[-1] if train_results else {},
                "eval": eval_stats,
            }
            all_fold_results.append(fold_summary)

            # Save rolling results
            _save_json(all_fold_results, output_dir / "walkforward_results.json")

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            try:
                if all_fold_results:
                    eval_sortinos = [
                        r["eval"].get("sortino", 0)
                        for r in all_fold_results
                        if r["eval"].get("n_trades", 0) >= 5
                    ]
                    if eval_sortinos:
                        mlflow.log_metrics({
                            "best_eval_sortino": max(eval_sortinos),
                            "mean_eval_sortino": float(np.mean(eval_sortinos)),
                            "n_windows_completed": len(all_fold_results),
                        })
                mlflow.end_run()
                log.info("[MLflow] Run ended successfully.")
            except Exception as e:
                log.warning(f"[MLflow] End run failed: {e}")

    log.info(f"\n{'='*70}")
    log.info(f"WALK-FORWARD COMPLETE")
    log.info(f"  Windows completed: {len(all_fold_results)}")
    log.info(f"  Best overall eval Sortino: {best_overall_sortino:.4f}")
    log.info(f"  Results saved to: {output_dir}")
    log.info(f"{'='*70}")

    return all_fold_results


def _save_json(obj, path: Path):
    """Save object as JSON, converting numpy types."""
    def _convert(o):
        if isinstance(o, (np.integer,)):   return int(o)
        if isinstance(o, (np.floating,)):  return float(o)
        if isinstance(o, np.ndarray):      return o.tolist()
        if isinstance(o, dict):            return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, list):            return [_convert(x) for x in o]
        return o

    with open(str(path), "w") as f:
        json.dump(_convert(obj), f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(
    checkpoint_path: str,
    cnn_pred_dir: Path,
    ptst_pred_dir: Optional[Path],
    output_dir: Path,
    device: str = "cuda",
    deterministic: bool = True,
):
    """Load a checkpoint and evaluate on all available fold data."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim = ckpt.get("state_dim", STATE_DIM)
    n_actions  = ckpt.get("n_actions", N_ACTIONS)

    policy = ExecPolicyV3(state_dim=state_dim).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    fold_files = sorted(cnn_pred_dir.glob("fold_*_oot_predictions.npz"))
    data = load_fold_predictions(cnn_pred_dir, ptst_pred_dir, fold_files)
    if data is None:
        log.error("Failed to load evaluation data.")
        return {}

    pca_mean = ckpt.get("pca_mean")
    pca_components = ckpt.get("pca_components")
    env = ExecEnvironment(data, pca_mean=pca_mean, pca_components=pca_components)

    # Override confidence tiers from checkpoint
    if "confidence_tiers" in ckpt:
        ct = ckpt["confidence_tiers"]
        env.p90 = ct["p90"]
        env.p95 = ct["p95"]
        env.p99 = ct["p99"]

    env.reset()
    while True:
        state = env.get_state()
        if state is None:
            break
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _ = policy.get_action(state_t, deterministic=deterministic)
        env.step(action.item())

    stats = env.get_episode_stats()

    log.info("\n" + "="*70)
    log.info("EVALUATION REPORT (RL Execution Agent v3)")
    log.info("="*70)
    log.info(f"  Checkpoint:       {checkpoint_path}")
    log.info(f"  Checkpoint iter:  {ckpt.get('iteration', 'N/A')}")
    log.info(f"  Train Sortino:    {ckpt.get('sortino', 'N/A')}")
    log.info(f"  Total samples:    {data['n_samples']:,}")
    log.info("")
    log.info(f"  Total trades:     {stats['n_trades']}")
    log.info(f"  Win rate:         {stats.get('win_rate', 0):.1f}%")
    log.info(f"  Total PnL:        {stats.get('total_pnl_ticks', 0):+.1f} ticks "
             f"(${stats.get('total_pnl_usd', 0):+,.0f})")
    log.info(f"  Avg PnL/trade:    {stats.get('avg_pnl_ticks', 0):+.2f} ticks")
    log.info(f"  Sortino:          {stats.get('sortino', 0):+.4f}")
    log.info(f"  Sharpe:           {stats.get('sharpe', 0):+.4f}")
    log.info(f"  Profit Factor:    {stats.get('profit_factor', 0):.2f}")
    log.info(f"  Avg hold (steps): {stats.get('avg_hold_steps', 0):.1f}")
    log.info(f"  Long trades:      {stats.get('n_long', 0)} "
             f"(PnL: {stats.get('long_pnl', 0):+.1f}t)")
    log.info(f"  Short trades:     {stats.get('n_short', 0)} "
             f"(PnL: {stats.get('short_pnl', 0):+.1f}t)")
    log.info(f"  Exit types:       {stats.get('exit_types', {})}")
    log.info(f"  Action pcts:      {stats.get('action_pcts', {})}")
    log.info(f"  Cost basis:       {COST_PASSIVE_TICKS:.3f}t passive, "
             f"{COST_MARKET_TICKS:.3f}t market")

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(stats, output_dir / "eval_results.json")
    log.info(f"\n  Results saved to {output_dir / 'eval_results.json'}")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RL Execution Agent v3 — ES Futures (Neptune RTX 3090)"
    )
    parser.add_argument(
        "--mode", choices=["train", "eval", "bc_warmstart"],
        default="train",
        help="train: walk-forward PPO training | eval: evaluate checkpoint | "
             "bc_warmstart: behavioral cloning only"
    )
    parser.add_argument(
        "--cnn-pred-dir",
        type=str,
        default="/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar",
        help="Directory with fold_*_oot_predictions.npz (CNN-Mamba v2)"
    )
    parser.add_argument(
        "--ptst-pred-dir",
        type=str,
        default="/home/nick/Lvl3Quant/output/patchtst_smart_v3_mar",
        help="Directory with PatchTST fold predictions (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/nick/Lvl3Quant/output/rl_execution_agent",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for eval mode"
    )
    parser.add_argument(
        "--n-iterations", type=int, default=200,
        help="PPO iterations per walk-forward window"
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=PPO_ROLLOUT_STEPS,
        help="Steps per rollout collection"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="PyTorch device (cuda/cpu)"
    )
    parser.add_argument(
        "--no-bc-warmstart", action="store_true",
        help="Skip behavioral cloning warm-start"
    )
    parser.add_argument(
        "--bc-steps", type=int, default=2000,
        help="Number of behavioral cloning demonstration steps"
    )
    parser.add_argument(
        "--experiment-name", type=str, default="rl_execution_agent",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic policy in eval mode"
    )
    args = parser.parse_args()

    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    cnn_pred_dir  = Path(args.cnn_pred_dir)
    ptst_pred_dir = Path(args.ptst_pred_dir) if args.ptst_pred_dir else None
    output_dir    = Path(args.output_dir)

    log.info("="*70)
    log.info("RL EXECUTION AGENT v3 (ES Futures)")
    log.info("="*70)
    log.info(f"  Mode:          {args.mode}")
    log.info(f"  CNN pred dir:  {cnn_pred_dir}")
    log.info(f"  PatchTST dir:  {ptst_pred_dir}")
    log.info(f"  Output dir:    {output_dir}")
    log.info(f"  Device:        {args.device}")
    log.info(f"  State dim:     {STATE_DIM}")
    log.info(f"  Actions:       {N_ACTIONS} ({list(ACTION_NAMES.values())})")
    log.info(f"  ES tick value: ${ES_TICK_VALUE}")
    log.info(f"  Cost passive:  {COST_PASSIVE_TICKS:.3f} ticks")
    log.info(f"  Cost market:   {COST_MARKET_TICKS:.3f} ticks")
    log.info(f"  Max daily loss:{MAX_DAILY_LOSS} ticks (circuit breaker)")
    log.info(f"  ToD gate:      {TOD_GATE_START_HOUR:.2f}-{TOD_GATE_END_HOUR:.2f} ET (no entries)")
    log.info(f"  SL levels:     {SL_LEVELS.tolist()} ticks")
    log.info(f"  Max hold:      {MAX_HOLD_STEPS} steps")
    log.info(f"  Walk-forward:  {WF_TRAIN_DAYS}d train / {WF_EVAL_DAYS}d eval (SLIDING)")
    log.info("="*70)

    if args.mode == "train":
        if not cnn_pred_dir.exists():
            log.error(f"CNN pred dir not found: {cnn_pred_dir}")
            sys.exit(1)

        run_walkforward(
            cnn_pred_dir=cnn_pred_dir,
            ptst_pred_dir=ptst_pred_dir,
            output_dir=output_dir,
            n_iterations=args.n_iterations,
            rollout_steps=args.rollout_steps,
            device=args.device,
            bc_warmstart=not args.no_bc_warmstart,
            n_bc_steps=args.bc_steps,
            experiment_name=args.experiment_name,
        )

    elif args.mode == "eval":
        checkpoint = args.checkpoint
        if not checkpoint:
            checkpoint = str(output_dir / "best_fold_overall.pt")
        if not Path(checkpoint).exists():
            log.error(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)

        evaluate_checkpoint(
            checkpoint_path=checkpoint,
            cnn_pred_dir=cnn_pred_dir,
            ptst_pred_dir=ptst_pred_dir,
            output_dir=output_dir,
            device=args.device,
            deterministic=True,
        )

    elif args.mode == "bc_warmstart":
        # Quick BC warm-start and save
        fold_files = sorted(cnn_pred_dir.glob("fold_*_oot_predictions.npz"))[:WF_TRAIN_DAYS]
        data = load_fold_predictions(cnn_pred_dir, ptst_pred_dir, fold_files)
        if data is None:
            log.error("Failed to load data for BC warm-start.")
            sys.exit(1)

        env = ExecEnvironment(data)
        policy = ExecPolicyV3(state_dim=STATE_DIM).to(args.device)

        behavioral_cloning_warmstart(
            policy, env,
            n_steps=args.bc_steps,
            lr=1e-3,
            device=args.device,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": policy.state_dict(),
            "mode": "bc_warmstart",
            "state_dim": STATE_DIM,
            "n_actions": N_ACTIONS,
            "pca_mean": env.pca_mean,
            "pca_components": env.pca_components,
            "version": "v3_bc",
        }, str(output_dir / "bc_warmstart.pt"))
        log.info(f"BC warm-start checkpoint saved to {output_dir / 'bc_warmstart.pt'}")


if __name__ == "__main__":
    main()
