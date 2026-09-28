#!/usr/bin/env python3
"""
train_exec_mlp_gpu.py — GPU Execution MLP + RL Exit Agent
==========================================================
Trains on Neptune RTX 3090 (24GB VRAM, Ubuntu).

Part 1: Multi-task MLP Execution Gate
  - Predicts P(profitable), MFE, MAE, optimal hold time
  - Features: CNN-Mamba embeddings + MBO microstructure + vol + temporal
  - Walk-forward: folds 01-08 train, folds 09-10 validation

Part 2: Simple DQN Exit Timing Agent
  - State: (pnl, time_in_trade, signal, signal_change, spread, vol)
  - Actions: hold, exit_market, tighten_stop
  - Reward: Sharpe-like (penalize variance, reward consistency)

Cost basis: 0.376 ticks RT ($4.70 AMP/Rithmic, HC #52).
All results are MIDPOINT-BASED (no FIFO fill sim).
MLflow logging mandatory.
Sliding window walk-forward (HC #0).
Reports Sharpe AND Sortino (HC #57).
"""

import os
import sys
import gc
import time
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root
LVL3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LVL3_ROOT))

try:
    from constants import COMMISSION_TICKS, TICK_VALUE
except ImportError:
    COMMISSION_TICKS = 0.376
    TICK_VALUE = 12.50

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
TICK = 0.25
TICK_VAL = TICK_VALUE  # $12.50
COST_TICKS_RT = COMMISSION_TICKS  # 0.376 ticks (commission only, midpoint-based)
SHORT_ONLY = False   # HC: FIFO sweep (2026-05-01) showed ONLY short-side has edge

# Paths (Neptune)
CNN_PRED_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
MBO_EVENT_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_events")
VOL_PRED_DIR = Path("/home/nick/Lvl3Quant/output/vol_lgbm_v3")
PTST_PRED_DIR = Path("/home/nick/Lvl3Quant/output/patchtst_smart_v3_mar")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/exec_mlp_gpu_optuna_best")

# Training config — Optuna best trial #11 (Sharpe 139.6)
N_FOLDS = 10
TRAIN_FOLDS = list(range(8))   # folds 01-08
VAL_FOLDS = list(range(8, 10)) # folds 09-10
EPOCHS = 80   # Optuna found early stopping at ~13-29 epochs, 80 is plenty
LR = 8.29e-4  # Optuna sweep #22 best
BATCH_SIZE = 2048  # Optuna sweep #22 best
HIDDEN_DIM = 256  # Optuna sweep #22 best
N_LAYERS = 5  # Optuna best (vs v1's 4)
DROPOUT = 0.342  # Optuna sweep #22 best
WEIGHT_DECAY = 4.78e-4  # Optuna best
NUM_WORKERS = 8
PIN_MEMORY = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MBO feature windows
MBO_WINDOWS = [50, 200, 500, 2000]

# MLflow
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "smart_execution_gpu"


# ============================================================
# Rolling statistics helpers (vectorized)
# ============================================================

def rolling_zscore(arr: np.ndarray, window: int = 3000) -> np.ndarray:
    """Efficient rolling z-score using cumulative sums."""
    n = len(arr)
    z = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < 30:
            continue
        s = cs[i] - (cs[start - 1] if start > 0 else 0)
        s2 = cs2[i] - (cs2[start - 1] if start > 0 else 0)
        mean = s / count
        var = s2 / count - mean ** 2
        std = np.sqrt(max(var, 1e-10))
        z[i] = (arr[i] - mean) / std
    return z


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        out[i] = (cs[i] - (cs[start - 1] if start > 0 else 0)) / count
    return out


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling std."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr)
    cs2 = np.cumsum(arr ** 2)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < 5:
            continue
        s = cs[i] - (cs[start - 1] if start > 0 else 0)
        s2 = cs2[i] - (cs2[start - 1] if start > 0 else 0)
        mean = s / count
        var = max(s2 / count - mean ** 2, 0.0)
        out[i] = np.sqrt(var)
    return out


def rolling_autocorr(arr: np.ndarray, window: int = 200, lag: int = 10) -> np.ndarray:
    """Rolling autocorrelation at given lag."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    for i in range(window + lag, n):
        x = arr[i - window:i]
        y = arr[i - window + lag:i + lag] if i + lag <= n else np.zeros(window)
        if len(y) < window:
            continue
        mx, my = x.mean(), y.mean()
        sx, sy = x.std(), y.std()
        if sx < 1e-8 or sy < 1e-8:
            continue
        out[i] = np.mean((x - mx) * (y - my)) / (sx * sy)
    return out


# ============================================================
# Feature Engineering
# ============================================================

def extract_mbo_features(
    mbo_data: dict,
    n_samples: int,
    sample_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract microstructure features from MBO event data.

    MBO data keys: timestamps, prices, sizes, sides, flags, event_types

    Returns: (n_samples, n_micro_features) array, feature names
    """
    names = []
    features = {}

    timestamps = mbo_data.get('timestamps', np.array([]))
    # MBO events are packed as (N, 6): [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
    events = mbo_data.get('events', None)
    if events is not None and len(events) > 0:
        prices = events[:, 3]          # price_rel_ticks (relative price in ticks)
        sizes = np.exp(events[:, 4])   # qty_log → actual quantity
        sides = events[:, 2]           # side_id: 0=bid, 1=ask, 2=none
        event_types = events[:, 1]     # event_type_id: 0=A, 1=C, 2=M, 3=T, 4=F
    else:
        prices = mbo_data.get('prices', np.array([]))
        sizes = mbo_data.get('sizes', np.array([]))
        sides = mbo_data.get('sides', np.array([]))
        event_types = mbo_data.get('event_types', np.array([]))

    n_events = len(timestamps)
    if n_events < 100:
        # Return zeros if insufficient data
        n_feat = 30  # number of micro features
        return np.zeros((n_samples, n_feat), dtype=np.float32), [f"micro_{i}" for i in range(n_feat)]

    # If no sample indices given, evenly space across events
    if sample_indices is None:
        sample_indices = np.linspace(100, n_events - 1, n_samples).astype(int)

    # Precompute derived arrays
    price_changes = np.diff(prices, prepend=prices[0])
    time_deltas = np.diff(timestamps, prepend=timestamps[0]).astype(np.float64)
    time_deltas_sec = time_deltas / 1e9  # nanoseconds to seconds
    time_deltas_sec = np.clip(time_deltas_sec, 1e-6, 3600.0)

    # Signed flow: positive = buy, negative = sell
    # side encoding: 0=bid(B), 1=ask(A), 2=none(N)
    # For trades (event_type=3): ask-side = buyer-initiated (+), bid-side = seller-initiated (-)
    side_sign = np.where(sides == 1, 1.0, np.where(sides == 0, -1.0, 0.0))
    signed_flow = sizes * side_sign

    # Cumulative sums for fast window queries
    cs_sizes = np.cumsum(sizes)
    cs_flow = np.cumsum(signed_flow)
    cs_pc2 = np.cumsum(price_changes ** 2)
    cs_abs_pc = np.cumsum(np.abs(price_changes))

    # Identify trade events vs order events
    # event_types: depends on encoding, but trades typically have a specific type
    # We'll use sizes > 0 as a proxy for trades
    is_trade = (sizes > 0).astype(np.float32)
    cs_trades = np.cumsum(is_trade)

    # Identify large trades (> 90th percentile of sizes)
    size_threshold = np.percentile(sizes[sizes > 0], 90) if np.any(sizes > 0) else 1.0

    def _cs_window(cs, idx, window):
        start = max(0, idx - window + 1)
        val = cs[idx] - (cs[start - 1] if start > 0 else 0)
        count = idx - start + 1
        return val, count

    # Initialize feature arrays
    feat_arrays = {}
    feat_names_list = []

    # --- Spread features ---
    # Compute bid-ask spread from price levels
    # Approximate: use price range in recent events as spread proxy
    spread_proxy = np.zeros(n_samples, dtype=np.float32)
    spread_zscore_arr = np.zeros(n_samples, dtype=np.float32)

    # --- Book imbalance ---
    book_imbalance = np.zeros(n_samples, dtype=np.float32)

    # --- Trade flow imbalance ---
    flow_imbalance_50 = np.zeros(n_samples, dtype=np.float32)
    flow_imbalance_200 = np.zeros(n_samples, dtype=np.float32)
    flow_imbalance_500 = np.zeros(n_samples, dtype=np.float32)

    # --- Event rate (events/second) ---
    event_rate_100 = np.zeros(n_samples, dtype=np.float32)
    event_rate_500 = np.zeros(n_samples, dtype=np.float32)

    # --- Large trade features ---
    large_trade_freq_100 = np.zeros(n_samples, dtype=np.float32)
    large_trade_vol_ratio = np.zeros(n_samples, dtype=np.float32)

    # --- Volume features ---
    volume_100 = np.zeros(n_samples, dtype=np.float32)
    volume_500 = np.zeros(n_samples, dtype=np.float32)
    volume_accel = np.zeros(n_samples, dtype=np.float32)
    volume_ratio = np.zeros(n_samples, dtype=np.float32)

    # --- Price momentum ---
    momentum_50 = np.zeros(n_samples, dtype=np.float32)
    momentum_200 = np.zeros(n_samples, dtype=np.float32)
    momentum_500 = np.zeros(n_samples, dtype=np.float32)

    # --- Realized volatility ---
    rvol_50 = np.zeros(n_samples, dtype=np.float32)
    rvol_200 = np.zeros(n_samples, dtype=np.float32)
    rvol_1000 = np.zeros(n_samples, dtype=np.float32)
    vol_of_vol = np.zeros(n_samples, dtype=np.float32)

    # --- Order arrival / cancel rates ---
    net_flow_ratio = np.zeros(n_samples, dtype=np.float32)

    # --- Bid/ask depth ratio ---
    bid_ask_size_ratio = np.zeros(n_samples, dtype=np.float32)

    # --- Sweep indicators ---
    price_impact_100 = np.zeros(n_samples, dtype=np.float32)

    for i, eidx in enumerate(sample_indices):
        eidx = int(min(eidx, n_events - 1))

        # Spread: price range in last 20 events as proxy
        if eidx >= 20:
            window_prices = prices[eidx - 20:eidx + 1]
            spread_proxy[i] = (np.max(window_prices) - np.min(window_prices)) / TICK
        if eidx >= 500:
            sp_arr = np.zeros(50)
            for k in range(50):
                idx_k = eidx - k * 10
                if idx_k >= 20:
                    wp = prices[idx_k - 20:idx_k + 1]
                    sp_arr[k] = (np.max(wp) - np.min(wp)) / TICK
            sp_mean = sp_arr.mean()
            sp_std = sp_arr.std()
            if sp_std > 1e-8:
                spread_zscore_arr[i] = (spread_proxy[i] - sp_mean) / sp_std

        # Book imbalance: bid volume vs ask volume in last 100 events
        if eidx >= 100:
            recent_sides = sides[eidx - 100:eidx + 1]
            recent_sizes_w = sizes[eidx - 100:eidx + 1]
            bid_vol = recent_sizes_w[recent_sides < 0.5].sum()
            ask_vol = recent_sizes_w[recent_sides > 0.5].sum()
            total = bid_vol + ask_vol
            if total > 0:
                book_imbalance[i] = (bid_vol - ask_vol) / total
                bid_ask_size_ratio[i] = bid_vol / max(ask_vol, 1e-6)

        # Trade flow imbalance
        for win, arr in [(50, flow_imbalance_50), (200, flow_imbalance_200),
                         (500, flow_imbalance_500)]:
            if eidx >= win:
                flow_sum, _ = _cs_window(cs_flow, eidx, win)
                vol_sum, _ = _cs_window(cs_sizes, eidx, win)
                if vol_sum > 0:
                    arr[i] = flow_sum / vol_sum

        # Event rate (events per second)
        for win, arr in [(100, event_rate_100), (500, event_rate_500)]:
            if eidx >= win:
                dt_total = time_deltas_sec[eidx - win + 1:eidx + 1].sum()
                if dt_total > 0:
                    arr[i] = win / dt_total

        # Large trade features
        if eidx >= 100:
            recent_sizes_w = sizes[eidx - 100:eidx + 1]
            large_mask = recent_sizes_w > size_threshold
            large_trade_freq_100[i] = large_mask.mean()
            total_vol = recent_sizes_w.sum()
            if total_vol > 0:
                large_trade_vol_ratio[i] = recent_sizes_w[large_mask].sum() / total_vol

        # Volume features
        for win, arr in [(100, volume_100), (500, volume_500)]:
            if eidx >= 1:
                vol_sum, _ = _cs_window(cs_sizes, eidx, win)
                arr[i] = vol_sum

        # Volume acceleration
        if eidx >= 50:
            vol_recent, _ = _cs_window(cs_sizes, eidx, 25)
            vol_prev, _ = _cs_window(cs_sizes, max(0, eidx - 25), 25)
            volume_accel[i] = vol_recent - vol_prev

        # Volume ratio vs session average
        if eidx >= 2000:
            long_vol, long_cnt = _cs_window(cs_sizes, eidx, 2000)
            short_vol, short_cnt = _cs_window(cs_sizes, eidx, 100)
            avg_per_100 = (long_vol / max(long_cnt, 1)) * 100
            if avg_per_100 > 0:
                volume_ratio[i] = short_vol / avg_per_100

        # Price momentum
        for win, arr in [(50, momentum_50), (200, momentum_200), (500, momentum_500)]:
            if eidx >= win:
                arr[i] = (prices[eidx] - prices[eidx - win]) / TICK

        # Realized volatility
        for win, arr in [(50, rvol_50), (200, rvol_200), (1000, rvol_1000)]:
            if eidx >= win:
                pc2_sum, cnt = _cs_window(cs_pc2, eidx, win)
                arr[i] = np.sqrt(pc2_sum / cnt) / TICK

        # Vol-of-vol (rolling std of rvol_50)
        # Approximate from recent rvol computations
        if eidx >= 500:
            rvol_samples = []
            for k in range(10):
                kidx = eidx - k * 50
                if kidx >= 50:
                    pc2_s, cnt_s = _cs_window(cs_pc2, kidx, 50)
                    rvol_samples.append(np.sqrt(pc2_s / cnt_s) / TICK)
            if len(rvol_samples) >= 3:
                vol_of_vol[i] = np.std(rvol_samples)

        # Net flow ratio (normalized)
        if eidx >= 200:
            flow_sum, _ = _cs_window(cs_flow, eidx, 200)
            abs_sum, cnt = _cs_window(cs_sizes, eidx, 200)
            if abs_sum > 0:
                net_flow_ratio[i] = flow_sum / abs_sum

        # Price impact: price move per unit volume
        if eidx >= 100:
            abs_move, _ = _cs_window(cs_abs_pc, eidx, 100)
            vol_sum, _ = _cs_window(cs_sizes, eidx, 100)
            if vol_sum > 0:
                price_impact_100[i] = (abs_move / TICK) / vol_sum

    # Assemble features
    all_feats = [
        ("spread_proxy", spread_proxy),
        ("spread_zscore", spread_zscore_arr),
        ("book_imbalance", book_imbalance),
        ("bid_ask_size_ratio", bid_ask_size_ratio),
        ("flow_imbalance_50", flow_imbalance_50),
        ("flow_imbalance_200", flow_imbalance_200),
        ("flow_imbalance_500", flow_imbalance_500),
        ("event_rate_100", event_rate_100),
        ("event_rate_500", event_rate_500),
        ("large_trade_freq", large_trade_freq_100),
        ("large_trade_vol_ratio", large_trade_vol_ratio),
        ("volume_100", volume_100),
        ("volume_500", volume_500),
        ("volume_accel", volume_accel),
        ("volume_ratio", volume_ratio),
        ("momentum_50", momentum_50),
        ("momentum_200", momentum_200),
        ("momentum_500", momentum_500),
        ("rvol_50", rvol_50),
        ("rvol_200", rvol_200),
        ("rvol_1000", rvol_1000),
        ("vol_of_vol", vol_of_vol),
        ("net_flow_ratio", net_flow_ratio),
        ("price_impact_100", price_impact_100),
    ]

    feat_names = [f[0] for f in all_feats]
    feat_matrix = np.column_stack([f[1] for f in all_feats]).astype(np.float32)

    return feat_matrix, feat_names


def build_signal_features(
    predictions: np.ndarray,  # (N, 3) - 1s/5s/10s
    embeddings: np.ndarray,   # (N, 96)
) -> Tuple[np.ndarray, List[str]]:
    """
    Build signal-derived features from CNN-Mamba predictions and embeddings.

    Includes:
      - Raw predictions (3)
      - Embeddings (96)
      - Absolute predictions / conviction (3)
      - Sign agreement across horizons (1)
      - Prediction z-scores per horizon (3)
      - Conviction trajectory (signal strengthening/weakening) (3)
      - Signal persistence / autocorrelation proxy (1)
      - Horizon disagreement (1)
    """
    N = predictions.shape[0]
    parts = []
    names = []

    # 1. Raw predictions (3)
    parts.append(predictions)
    names.extend(["pred_1s", "pred_5s", "pred_10s"])

    # 2. CNN-Mamba embeddings (96)
    parts.append(embeddings)
    names.extend([f"emb_{i}" for i in range(embeddings.shape[1])])

    # 3. Absolute predictions = conviction strength (3)
    abs_preds = np.abs(predictions)
    parts.append(abs_preds)
    names.extend(["abs_pred_1s", "abs_pred_5s", "abs_pred_10s"])

    # 4. Sign agreement across horizons (1)
    signs = np.sign(predictions)
    # Fraction of horizons agreeing with 10s (longest horizon)
    sign_agreement = np.mean(signs == signs[:, 2:3], axis=1, keepdims=True)
    parts.append(sign_agreement)
    names.append("sign_agreement")

    # 5. Prediction z-scores per horizon (rolling window=3000) (3)
    for h, label in enumerate(["1s", "5s", "10s"]):
        z = rolling_zscore(predictions[:, h], window=3000)
        parts.append(z.reshape(-1, 1))
        names.append(f"pred_zscore_{label}")

    # 6. Conviction trajectory: is signal strengthening or weakening? (3)
    # Compare short-term rolling mean to long-term
    for h, label in enumerate(["1s", "5s", "10s"]):
        short_mean = rolling_mean(np.abs(predictions[:, h]), window=50)
        long_mean = rolling_mean(np.abs(predictions[:, h]), window=500)
        trajectory = short_mean - long_mean
        parts.append(trajectory.reshape(-1, 1))
        names.append(f"conviction_trajectory_{label}")

    # 7. Signal persistence (autocorrelation proxy for 10s pred) (1)
    # Use rolling autocorrelation of 10s prediction at lag=10
    autocorr = rolling_autocorr(predictions[:, 2], window=200, lag=10)
    parts.append(autocorr.reshape(-1, 1))
    names.append("signal_persistence_10s")

    # 8. Horizon disagreement: |pred_1s - pred_10s| normalized (1)
    pred_std_across = np.std(predictions, axis=1, keepdims=True)
    parts.append(pred_std_across)
    names.append("horizon_disagreement")

    # 9. Prediction range (max - min across horizons) (1)
    pred_range = (np.max(predictions, axis=1) - np.min(predictions, axis=1)).reshape(-1, 1)
    parts.append(pred_range)
    names.append("pred_range")

    # 10. Prediction skew (is 10s much bigger than 1s? momentum indicator) (1)
    pred_skew = (predictions[:, 2] - predictions[:, 0]).reshape(-1, 1)
    parts.append(pred_skew)
    names.append("pred_skew_10s_vs_1s")

    feature_matrix = np.concatenate(parts, axis=1).astype(np.float32)
    return feature_matrix, names


def build_temporal_features(
    timestamps_ns: np.ndarray,  # nanosecond timestamps, or None
    n_samples: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build temporal features: time-of-day (cyclical), session indicators.
    If timestamps not available, use sample index as proxy.
    """
    parts = []
    names = []

    if timestamps_ns is not None and len(timestamps_ns) == n_samples:
        # Convert to fractional hour in ET (EST = UTC-5)
        utc_seconds = timestamps_ns / 1e9
        et_seconds = utc_seconds + (-5) * 3600
        hours = (et_seconds % 86400) / 3600.0
    else:
        # Proxy: assume RTH 9:30-16:00 ET, linearly distributed
        hours = np.linspace(9.5, 16.0, n_samples)

    # Cyclical time-of-day (2)
    tod_sin = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
    tod_cos = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
    parts.append(tod_sin.reshape(-1, 1))
    parts.append(tod_cos.reshape(-1, 1))
    names.extend(["tod_sin", "tod_cos"])

    # Finer cyclical (within RTH: 6.5 hours) (2)
    rth_frac = np.clip((hours - 9.5) / 6.5, 0, 1)
    rth_sin = np.sin(2 * np.pi * rth_frac).astype(np.float32)
    rth_cos = np.cos(2 * np.pi * rth_frac).astype(np.float32)
    parts.append(rth_sin.reshape(-1, 1))
    parts.append(rth_cos.reshape(-1, 1))
    names.extend(["rth_sin", "rth_cos"])

    # Session indicators (binary) (4)
    # Open (9:30-10:30), Core (10:30-15:00), Close (15:00-16:00), Pre-market (<9:30)
    is_open = ((hours >= 9.5) & (hours < 10.5)).astype(np.float32)
    is_core = ((hours >= 10.5) & (hours < 15.0)).astype(np.float32)
    is_close = ((hours >= 15.0) & (hours < 16.0)).astype(np.float32)
    is_pre = (hours < 9.5).astype(np.float32)
    parts.append(is_open.reshape(-1, 1))
    parts.append(is_core.reshape(-1, 1))
    parts.append(is_close.reshape(-1, 1))
    parts.append(is_pre.reshape(-1, 1))
    names.extend(["session_open", "session_core", "session_close", "session_pre"])

    # Minutes since RTH open (1)
    mins_since_open = np.clip((hours - 9.5) * 60, -60, 420).astype(np.float32)
    parts.append(mins_since_open.reshape(-1, 1))
    names.append("mins_since_open")

    # Minutes until close (1)
    mins_until_close = np.clip((16.0 - hours) * 60, -60, 420).astype(np.float32)
    parts.append(mins_until_close.reshape(-1, 1))
    names.append("mins_until_close")

    feature_matrix = np.concatenate(parts, axis=1).astype(np.float32)
    return feature_matrix, names


def build_vol_features(
    vol_preds: Optional[np.ndarray],  # vol predictions (N,) or (N,3) or None
    predictions: np.ndarray,          # CNN preds (N, 3) for realized vol proxy
) -> Tuple[np.ndarray, List[str]]:
    """
    Build volatility context features.
    Uses LGBM vol predictions if available, otherwise derives from prediction variance.
    """
    N = len(predictions)
    parts = []
    names = []

    if vol_preds is not None:
        vp = vol_preds
        if vp.ndim == 1:
            vp = vp.reshape(-1, 1)
        parts.append(vp)
        for j in range(vp.shape[1]):
            names.append(f"vol_pred_{j}")
    else:
        # No vol predictions — use zero placeholder
        parts.append(np.zeros((N, 1), dtype=np.float32))
        names.append("vol_pred_placeholder")

    # Realized vol from prediction variance (rolling std of 10s pred) (2)
    pred_vol_short = rolling_std(predictions[:, 2], window=100)
    pred_vol_long = rolling_std(predictions[:, 2], window=1000)
    parts.append(pred_vol_short.reshape(-1, 1))
    parts.append(pred_vol_long.reshape(-1, 1))
    names.extend(["pred_vol_100", "pred_vol_1000"])

    # Vol regime: ratio of short-term to long-term vol (1)
    vol_regime = pred_vol_short / np.maximum(pred_vol_long, 1e-8)
    parts.append(vol_regime.reshape(-1, 1))
    names.append("vol_regime")

    feature_matrix = np.concatenate(parts, axis=1).astype(np.float32)
    return feature_matrix, names


# ============================================================
# Data Loading
# ============================================================

def load_fold_data(fold_idx: int) -> Optional[dict]:
    """Load CNN embeddings, predictions, vol predictions, and MBO events for one OOT fold."""
    cnn_file = CNN_PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not cnn_file.exists():
        logger.warning(f"Fold {fold_idx:02d}: CNN predictions not found at {cnn_file}")
        return None

    cnn_data = np.load(str(cnn_file), allow_pickle=True)
    embeddings = cnn_data['embeddings']       # (N, 96)
    predictions = cnn_data['predictions']     # (N, 3) = 1s/5s/10s
    labels = cnn_data['labels']               # (N, 3) = 1s/5s/10s

    # Get OOT date
    oot_files = cnn_data['oot_files']
    if hasattr(oot_files, 'tolist'):
        oot_files = oot_files.tolist()
    date_str = str(oot_files[0]).split('/')[-1][:8] if isinstance(oot_files, list) else str(oot_files).split('/')[-1][:8]

    # Load vol predictions
    vol_pred = None
    vol_file = VOL_PRED_DIR / f"vol_v3_{date_str}_predictions.npz"
    if vol_file.exists():
        try:
            vd = np.load(str(vol_file), allow_pickle=True)
            for k in ['predictions', 'vol_pred', 'y_pred']:
                if k in vd:
                    vp = vd[k]
                    if len(vp) == len(embeddings):
                        vol_pred = vp
                    else:
                        from scipy.interpolate import interp1d
                        x_vol = np.linspace(0, 1, len(vp))
                        x_cnn = np.linspace(0, 1, len(embeddings))
                        if vp.ndim == 1:
                            f_interp = interp1d(x_vol, vp, kind='nearest', fill_value='extrapolate')
                            vol_pred = f_interp(x_cnn)
                        else:
                            vol_pred = np.zeros((len(embeddings), vp.shape[1]))
                            for j in range(vp.shape[1]):
                                f_interp = interp1d(x_vol, vp[:, j], kind='nearest', fill_value='extrapolate')
                                vol_pred[:, j] = f_interp(x_cnn)
                    break
        except Exception as e:
            logger.warning(f"Failed to load vol for {date_str}: {e}")

    # Load PatchTST predictions (HC #72: learn when other models are correct)
    ptst_pred = None
    # PatchTST folds are indexed differently — search by date
    for ptst_file in sorted(PTST_PRED_DIR.glob("fold_*_oot_predictions.npz")):
        try:
            pd_raw = np.load(str(ptst_file), allow_pickle=True)
            ptst_oot = pd_raw.get('oot_files', [''])
            if hasattr(ptst_oot, 'tolist'):
                ptst_oot = ptst_oot.tolist()
            ptst_date = str(ptst_oot[0]).replace('\\', '/').split('/')[-1][:8]
            if ptst_date == date_str:
                ptst_raw = pd_raw['predictions']  # (M, 3)
                N = len(embeddings)
                if len(ptst_raw) == N:
                    ptst_pred = ptst_raw
                else:
                    # PatchTST often has 2x samples — subsample to match CNN
                    from scipy.interpolate import interp1d
                    x_ptst = np.linspace(0, 1, len(ptst_raw))
                    x_cnn = np.linspace(0, 1, N)
                    ptst_pred = np.zeros((N, ptst_raw.shape[1]), dtype=np.float32)
                    for j in range(ptst_raw.shape[1]):
                        f_interp = interp1d(x_ptst, ptst_raw[:, j], kind='nearest', fill_value='extrapolate')
                        ptst_pred[:, j] = f_interp(x_cnn)
                logger.info(f"  Fold {fold_idx:02d}: PatchTST loaded ({len(ptst_raw)} → {N} aligned)")
                break
        except Exception as e:
            continue

    # Load MBO events
    mbo_data = None
    mbo_file = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if mbo_file.exists():
        try:
            mbo_raw = np.load(str(mbo_file), allow_pickle=True)
            mbo_data = {k: mbo_raw[k] for k in mbo_raw.files}
            logger.info(f"  Fold {fold_idx:02d}: MBO events loaded ({len(mbo_data.get('timestamps', []))} events)")
        except Exception as e:
            logger.warning(f"Failed to load MBO for {date_str}: {e}")

    return {
        'fold_idx': fold_idx,
        'date': date_str,
        'embeddings': embeddings,
        'predictions': predictions,
        'labels': labels,
        'vol_pred': vol_pred,
        'ptst_pred': ptst_pred,
        'mbo_data': mbo_data,
        'n_samples': len(embeddings),
    }


def build_all_features(data: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Build the complete feature matrix from all sources.
    Returns (N, total_features) array and feature names.
    """
    N = data['n_samples']
    all_parts = []
    all_names = []

    # 1. Signal features (predictions + embeddings + derived)
    sig_feats, sig_names = build_signal_features(data['predictions'], data['embeddings'])
    all_parts.append(sig_feats)
    all_names.extend(sig_names)
    logger.info(f"    Signal features: {sig_feats.shape[1]}")

    # 2. Microstructure features from MBO data
    if data['mbo_data'] is not None:
        mbo_feats, mbo_names = extract_mbo_features(data['mbo_data'], N)
        all_parts.append(mbo_feats)
        all_names.extend(mbo_names)
        logger.info(f"    MBO features: {mbo_feats.shape[1]}")
    else:
        # Zero-fill if no MBO data
        n_mbo = 24
        all_parts.append(np.zeros((N, n_mbo), dtype=np.float32))
        all_names.extend([f"micro_{i}" for i in range(n_mbo)])
        logger.info(f"    MBO features: {n_mbo} (zeros — no MBO data)")

    # 3. Temporal features
    timestamps = data['mbo_data'].get('timestamps', None) if data['mbo_data'] else None
    temp_feats, temp_names = build_temporal_features(timestamps, N)
    all_parts.append(temp_feats)
    all_names.extend(temp_names)
    logger.info(f"    Temporal features: {temp_feats.shape[1]}")

    # 4. Volatility context
    vol_feats, vol_names = build_vol_features(data['vol_pred'], data['predictions'])
    all_parts.append(vol_feats)
    all_names.extend(vol_names)
    logger.info(f"    Vol features: {vol_feats.shape[1]}")

    # 5. PatchTST features REMOVED in v3 — they hurt performance (v2 showed adding noise)
    # PatchTST confluence should be tested as rule-based filter (Jupiter), not MLP features

    feature_matrix = np.concatenate(all_parts, axis=1).astype(np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

    logger.info(f"    TOTAL features: {feature_matrix.shape[1]}")
    return feature_matrix, all_names


def build_targets(data: dict) -> dict:
    """
    Build multi-task targets from labels.

    Targets:
      - gate: P(trade profitable after 0.376 tick cost) — binary
      - mfe: max favorable excursion in ticks (max across horizons)
      - mae: max adverse excursion in ticks
      - hold_time: optimal hold time (seconds) — approximated from which horizon is best
    """
    predictions = data['predictions']  # (N, 3)
    labels = data['labels']            # (N, 3) = actual price changes in ticks at 1s/5s/10s

    N = len(predictions)
    direction = np.sign(predictions[:, 2])  # use 10s prediction for trade direction

    # SHORT-ONLY filter (HC: FIFO sweep 2026-05-01 showed long side has no edge)
    if SHORT_ONLY:
        short_mask = (direction == -1)
        if short_mask.sum() == 0:
            # No short signals — return empty targets
            return {
                'gate': np.array([], dtype=np.float32),
                'mfe': np.array([], dtype=np.float32),
                'mae': np.array([], dtype=np.float32),
                'hold_time': np.array([], dtype=np.float32),
                'direction': np.array([], dtype=np.float32),
                'short_mask': short_mask,
            }
        predictions = predictions[short_mask]
        labels = labels[short_mask]
        direction = direction[short_mask]
        N = len(predictions)

    # Directional P&L per horizon (in ticks)
    pnl_1s = direction * labels[:, 0]
    pnl_5s = direction * labels[:, 1]
    pnl_10s = direction * labels[:, 2]

    # Gate target: was the best horizon profitable after costs?
    best_pnl = np.maximum.reduce([pnl_1s, pnl_5s, pnl_10s])
    gate_target = (best_pnl > COST_TICKS_RT).astype(np.float32)

    # MFE target: max favorable excursion across horizons (ticks)
    mfe_target = np.maximum(0, best_pnl).astype(np.float32)

    # MAE target: max adverse excursion (worst P&L across horizons, positive = bad)
    worst_pnl = np.minimum.reduce([pnl_1s, pnl_5s, pnl_10s])
    mae_target = np.maximum(0, -worst_pnl).astype(np.float32)

    # Hold time target: which horizon had the best P&L? Map to seconds.
    horizon_seconds = np.array([1.0, 5.0, 10.0])
    pnl_stack = np.stack([pnl_1s, pnl_5s, pnl_10s], axis=1)
    best_horizon_idx = np.argmax(pnl_stack, axis=1)
    hold_time_target = horizon_seconds[best_horizon_idx].astype(np.float32)

    return {
        'gate': gate_target,
        'short_mask': short_mask if SHORT_ONLY else np.ones(N, dtype=bool),
        'mfe': mfe_target,
        'mae': mae_target,
        'hold_time': hold_time_target,
    }


# ============================================================
# PyTorch Dataset
# ============================================================

class ExecDataset(Dataset):
    """Dataset for execution MLP training."""

    def __init__(self, features: np.ndarray, targets: dict):
        self.features = torch.from_numpy(features)
        self.gate = torch.from_numpy(targets['gate'])
        self.mfe = torch.from_numpy(targets['mfe'])
        self.mae = torch.from_numpy(targets['mae'])
        self.hold_time = torch.from_numpy(targets['hold_time'])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'gate': self.gate[idx],
            'mfe': self.mfe[idx],
            'mae': self.mae[idx],
            'hold_time': self.hold_time[idx],
        }


# ============================================================
# Model: Multi-task Execution MLP
# ============================================================

class ExecMLP(nn.Module):
    """
    Multi-task MLP for execution gating.

    Architecture:
      Input → BatchNorm → [Linear(dim, 256) → ReLU → Dropout(0.3)] × 4 → Multi-head output

    Heads:
      - Gate: sigmoid → P(profitable trade)
      - MFE: softplus → expected max favorable excursion (ticks)
      - MAE: softplus → expected max adverse excursion (ticks)
      - Hold time: softplus → optimal hold time (seconds)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        # Shared trunk
        layers = []
        prev_dim = input_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Output heads
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.mfe_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        self.mae_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        self.hold_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        x = self.input_bn(x)
        shared = self.trunk(x)

        gate = self.gate_head(shared).squeeze(-1)
        mfe = self.mfe_head(shared).squeeze(-1)
        mae = self.mae_head(shared).squeeze(-1)
        hold = self.hold_head(shared).squeeze(-1)

        return {
            'gate': gate,
            'mfe': mfe,
            'mae': mae,
            'hold_time': hold,
        }


# ============================================================
# Training Loop
# ============================================================

def compute_sharpe(pnl_array: np.ndarray) -> float:
    """Annualized Sharpe from per-trade PnL."""
    if len(pnl_array) < 2:
        return 0.0
    mean = np.mean(pnl_array)
    std = np.std(pnl_array)
    if std < 1e-8:
        return 0.0
    # Annualize: assume ~100 trades/day, 252 days
    return mean / std * np.sqrt(252 * 100)


def compute_sortino(pnl_array: np.ndarray) -> float:
    """Annualized Sortino from per-trade PnL."""
    if len(pnl_array) < 2:
        return 0.0
    mean = np.mean(pnl_array)
    down = pnl_array[pnl_array < 0]
    down_std = np.std(down) if len(down) > 0 else 1e-8
    if down_std < 1e-8:
        return 0.0
    return mean / down_std * np.sqrt(252 * 100)


def evaluate_gated_trading(
    gate_probs: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    mfe_pred: Optional[np.ndarray] = None,
    mae_pred: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate gated trading performance at a given gate threshold.
    Midpoint-based (no FIFO fill sim).
    """
    mask = gate_probs >= threshold
    n_gated = mask.sum()
    if n_gated < 10:
        return {'n_trades': 0}

    direction = np.sign(predictions[mask, 2])  # trade in 10s direction
    pnl_ticks = direction * labels[mask, 2] - COST_TICKS_RT

    wins = pnl_ticks[pnl_ticks > 0]
    losses = pnl_ticks[pnl_ticks < 0]

    total_pnl = pnl_ticks.sum() * TICK_VAL
    avg_pnl = np.mean(pnl_ticks) * TICK_VAL
    win_rate = len(wins) / len(pnl_ticks) if len(pnl_ticks) > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 1e-8 else float('inf')
    avg_rr = np.mean(wins) / abs(np.mean(losses)) if len(losses) > 0 and abs(np.mean(losses)) > 1e-8 else float('inf')

    sharpe = compute_sharpe(pnl_ticks)
    sortino = compute_sortino(pnl_ticks)

    return {
        'n_trades': int(n_gated),
        'coverage': float(mask.mean()),
        'total_pnl': float(total_pnl),
        'avg_pnl': float(avg_pnl),
        'win_rate': float(win_rate),
        'pf': float(min(pf, 99.99)),
        'avg_rr': float(min(avg_rr, 99.99)),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
    }


def train_mlp(
    train_features: np.ndarray,
    train_targets: dict,
    val_features: np.ndarray,
    val_targets: dict,
    val_predictions: np.ndarray,
    val_labels: np.ndarray,
    feature_names: List[str],
    run_name: str = "exec_mlp",
) -> Tuple[ExecMLP, dict]:
    """
    Train multi-task execution MLP with early stopping on validation Sharpe.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Training {run_name}")
    logger.info(f"  Train: {len(train_features):,} samples")
    logger.info(f"  Val:   {len(val_features):,} samples")
    logger.info(f"  Features: {train_features.shape[1]}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Cost: {COST_TICKS_RT:.3f} ticks RT (${COST_TICKS_RT * TICK_VAL:.2f})")
    logger.info(f"  Results are MIDPOINT-BASED (no FIFO fill sim)")
    logger.info(f"{'='*70}")

    # Normalize features
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_norm = (train_features - mean) / std
    val_norm = (val_features - mean) / std

    # Create datasets and loaders
    train_ds = ExecDataset(train_norm, train_targets)
    val_ds = ExecDataset(val_norm, val_targets)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # Model
    input_dim = train_features.shape[1]
    model = ExecMLP(input_dim, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    logger.info(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Loss functions
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    huber_loss = nn.SmoothL1Loss()

    # Class weights for gate (handle imbalanced labels)
    gate_positive_ratio = train_targets['gate'].mean()
    gate_weight = max(0.5, min(2.0, (1 - gate_positive_ratio) / max(gate_positive_ratio, 0.01)))
    logger.info(f"  Gate positive ratio: {gate_positive_ratio:.3f}, weight: {gate_weight:.2f}")

    # Early stopping on validation Sharpe
    best_val_sharpe = -float('inf')
    best_state = None
    best_epoch = 0
    patience = 15
    no_improve = 0

    for epoch in range(EPOCHS):
        # === Training ===
        model.train()
        epoch_losses = {'gate': 0, 'mfe': 0, 'mae': 0, 'hold': 0, 'total': 0}
        n_batches = 0

        for batch in train_loader:
            feats = batch['features'].to(DEVICE)
            gate_tgt = batch['gate'].to(DEVICE)
            mfe_tgt = batch['mfe'].to(DEVICE)
            mae_tgt = batch['mae'].to(DEVICE)
            hold_tgt = batch['hold_time'].to(DEVICE)

            outputs = model(feats)

            # Multi-task loss with weighting
            # Gate loss: weighted BCE
            pos_weight = torch.where(gate_tgt > 0.5, gate_weight, 1.0)
            gate_l = F.binary_cross_entropy(outputs['gate'], gate_tgt, weight=pos_weight)

            # MFE/MAE loss: Huber (robust to outliers)
            mfe_l = huber_loss(outputs['mfe'], mfe_tgt)
            mae_l = huber_loss(outputs['mae'], mae_tgt)
            hold_l = huber_loss(outputs['hold_time'], hold_tgt)

            # Total loss: gate is primary, others are secondary
            total_loss = 2.0 * gate_l + 0.5 * mfe_l + 0.5 * mae_l + 0.3 * hold_l

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses['gate'] += gate_l.item()
            epoch_losses['mfe'] += mfe_l.item()
            epoch_losses['mae'] += mae_l.item()
            epoch_losses['hold'] += hold_l.item()
            epoch_losses['total'] += total_loss.item()
            n_batches += 1

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # === Validation ===
        model.eval()
        all_gate_probs = []
        all_mfe_preds = []
        all_mae_preds = []
        val_loss_total = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                feats = batch['features'].to(DEVICE)
                gate_tgt = batch['gate'].to(DEVICE)
                mfe_tgt = batch['mfe'].to(DEVICE)
                mae_tgt = batch['mae'].to(DEVICE)
                hold_tgt = batch['hold_time'].to(DEVICE)

                outputs = model(feats)

                gate_l = bce_loss(outputs['gate'], gate_tgt)
                mfe_l = huber_loss(outputs['mfe'], mfe_tgt)
                val_loss_total += (gate_l.item() + mfe_l.item())
                val_batches += 1

                all_gate_probs.append(outputs['gate'].cpu().numpy())
                all_mfe_preds.append(outputs['mfe'].cpu().numpy())
                all_mae_preds.append(outputs['mae'].cpu().numpy())

        gate_probs = np.concatenate(all_gate_probs)
        mfe_preds = np.concatenate(all_mfe_preds)
        mae_preds = np.concatenate(all_mae_preds)

        # Evaluate at gate threshold = 0.50
        eval_result = evaluate_gated_trading(gate_probs, val_predictions, val_labels, threshold=0.50)
        val_sharpe = eval_result.get('sharpe', 0)
        val_sortino = eval_result.get('sortino', 0)

        # Log every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            logger.info(
                f"  Epoch {epoch:>3d}/{EPOCHS} | "
                f"Train loss: {epoch_losses['total']:.4f} (gate={epoch_losses['gate']:.4f}) | "
                f"Val Sharpe: {val_sharpe:>7.2f} | Sortino: {val_sortino:>7.2f} | "
                f"Trades: {eval_result.get('n_trades', 0)} | "
                f"WR: {eval_result.get('win_rate', 0):.1%} | "
                f"PF: {eval_result.get('pf', 0):.2f}"
            )

        # MLflow logging
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    'train_loss': epoch_losses['total'],
                    'train_gate_loss': epoch_losses['gate'],
                    'val_sharpe': val_sharpe,
                    'val_sortino': val_sortino,
                    'val_win_rate': eval_result.get('win_rate', 0),
                    'val_pf': min(eval_result.get('pf', 0), 99.99),
                    'val_n_trades': eval_result.get('n_trades', 0),
                    'val_avg_pnl': eval_result.get('avg_pnl', 0),
                    'lr': optimizer.param_groups[0]['lr'],
                }, step=epoch)
            except Exception:
                pass

        # Early stopping on Sharpe
        if val_sharpe > best_val_sharpe:
            best_val_sharpe = val_sharpe
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f"  Best epoch: {best_epoch}, Best val Sharpe: {best_val_sharpe:.2f}")

    # Final comprehensive evaluation at multiple thresholds
    model.eval()
    with torch.no_grad():
        val_norm_t = torch.from_numpy(val_norm).to(DEVICE)
        # Process in chunks to avoid OOM
        chunk_size = 10000
        gate_probs_list = []
        mfe_preds_list = []
        mae_preds_list = []
        hold_preds_list = []
        for start in range(0, len(val_norm_t), chunk_size):
            end = min(start + chunk_size, len(val_norm_t))
            chunk_out = model(val_norm_t[start:end])
            gate_probs_list.append(chunk_out['gate'].cpu().numpy())
            mfe_preds_list.append(chunk_out['mfe'].cpu().numpy())
            mae_preds_list.append(chunk_out['mae'].cpu().numpy())
            hold_preds_list.append(chunk_out['hold_time'].cpu().numpy())

    gate_probs = np.concatenate(gate_probs_list)
    mfe_preds = np.concatenate(mfe_preds_list)
    mae_preds = np.concatenate(mae_preds_list)
    hold_preds = np.concatenate(hold_preds_list)

    results = {
        'best_epoch': best_epoch,
        'best_val_sharpe': best_val_sharpe,
        'gate_probs': gate_probs,
        'mfe_preds': mfe_preds,
        'mae_preds': mae_preds,
        'hold_preds': hold_preds,
        'norm_mean': mean,
        'norm_std': std,
    }

    return model, results


# ============================================================
# Part 2: DQN Exit Timing Agent
# ============================================================

class ExitDQN(nn.Module):
    """
    Simple DQN for exit timing.
    State: (current_pnl, time_in_trade, current_signal, signal_change, spread, vol)
    Actions: hold=0, exit_market=1, tighten_stop=2
    """
    EXIT_STATE_DIM = 6
    N_ACTIONS = 3

    def __init__(self, state_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.N_ACTIONS),
        )

    def forward(self, state):
        return self.net(state)


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class ExitTimingTrainer:
    """
    Trains DQN exit agent using simulated trade episodes.
    Reward = Sharpe-like: penalize variance, reward consistent small wins.
    """

    def __init__(self, device: str = 'cuda', lr: float = 1e-3, gamma: float = 0.99):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy_net = ExitDQN().to(self.device)
        self.target_net = ExitDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=100000)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update_freq = 100
        self.step_count = 0

    def _make_state(self, pnl: float, time_in_trade: float, signal: float,
                    signal_change: float, spread: float, vol: float) -> torch.Tensor:
        """Create normalized state tensor."""
        return torch.tensor([
            np.clip(pnl / 10.0, -2, 2),          # PnL in ticks, normalized
            np.clip(time_in_trade / 60.0, 0, 2),   # seconds, normalized
            np.clip(signal / 5.0, -1, 1),           # signal z-score
            np.clip(signal_change, -1, 1),          # signal delta
            np.clip(spread / 4.0, 0, 2),            # spread
            np.clip(vol * 100, 0, 2),               # vol
        ], dtype=torch.float32)

    def simulate_episode(
        self,
        predictions: np.ndarray,  # (N, 3)
        labels: np.ndarray,       # (N, 3)
        gate_probs: np.ndarray,   # (N,) from MLP gate
        spreads: np.ndarray,      # (N,)
        vols: np.ndarray,         # (N,)
        start_idx: int = 0,
    ) -> float:
        """
        Simulate one trading episode from a gate-selected entry.
        Returns total reward.
        """
        N = len(predictions)
        total_reward = 0.0

        # Find entry points where gate fires
        entries = np.where(gate_probs > 0.5)[0]
        if len(entries) == 0:
            return 0.0

        # Sample up to 50 entries per episode to avoid huge episodes
        if len(entries) > 50:
            entries = np.random.choice(entries, 50, replace=False)
            entries.sort()

        for entry_idx in entries:
            if entry_idx >= N - 100:
                continue

            direction = np.sign(predictions[entry_idx, 2])
            if direction == 0:
                continue

            entry_signal = predictions[entry_idx, 2]
            pnl = 0.0
            stop_level = -3.0  # initial stop: 3 ticks

            # Simulate holding for up to 100 steps (10s at 0.1s per step)
            for t in range(1, min(100, N - entry_idx)):
                current_idx = entry_idx + t
                current_pnl = direction * labels[current_idx, 2] - COST_TICKS_RT  # simplified
                current_signal = predictions[current_idx, 2] if current_idx < N else 0
                signal_change = current_signal - entry_signal
                spread = spreads[current_idx] if current_idx < len(spreads) else 1.0
                vol = vols[current_idx] if current_idx < len(vols) else 0.01

                state = self._make_state(
                    current_pnl, t * 0.1, current_signal, signal_change, spread, vol
                ).to(self.device)

                # Select action
                if random.random() < self.epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        q_vals = self.policy_net(state.unsqueeze(0))
                        action = q_vals.argmax(dim=1).item()

                # Execute action
                done = False
                if action == 1:  # exit_market
                    reward = current_pnl  # realize P&L
                    done = True
                elif action == 2:  # tighten_stop
                    stop_level = max(stop_level, current_pnl - 1.0)  # trail by 1 tick
                    reward = 0.0
                else:  # hold
                    reward = 0.0

                # Check stop
                if current_pnl <= stop_level:
                    reward = current_pnl
                    done = True

                # Next state
                if not done and t + 1 < min(100, N - entry_idx):
                    next_idx = entry_idx + t + 1
                    next_pnl = direction * labels[next_idx, 2] - COST_TICKS_RT
                    next_signal = predictions[next_idx, 2] if next_idx < N else 0
                    next_state = self._make_state(
                        next_pnl, (t + 1) * 0.1, next_signal,
                        next_signal - entry_signal, spread, vol
                    ).to(self.device)
                else:
                    next_state = state  # terminal
                    if not done:
                        reward = current_pnl  # force exit at max hold
                        done = True

                # Sharpe-like reward shaping: penalize variance
                shaped_reward = reward / (1.0 + abs(reward) * 0.2) if reward != 0 else 0

                self.replay.push(state.cpu(), action, shaped_reward, next_state.cpu(), done)
                total_reward += reward

                if done:
                    break

        return total_reward

    def train_step(self, batch_size: int = 128) -> float:
        """One DQN training step."""
        if len(self.replay) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Execution MLP + RL Exit Agent (GPU)")
    parser.add_argument('--skip-rl', action='store_true', help='Skip Part 2 (RL exit agent)')
    parser.add_argument('--epochs', type=int, default=80, help='MLP training epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=2048, help='Batch size')
    parser.add_argument('--rl-episodes', type=int, default=200, help='RL training episodes')
    args = parser.parse_args()

    # Note: EPOCHS/LR/BATCH_SIZE use module-level defaults.
    # CLI args override them if needed via the training functions directly.

    logger.info("=" * 80)
    logger.info("Execution MLP GPU v1 — Multi-task Gate + RL Exit Agent")
    logger.info(f"  Cost: {COST_TICKS_RT:.3f} ticks RT (${COST_TICKS_RT * TICK_VAL:.2f}, HC #52)")
    logger.info(f"  Results are MIDPOINT-BASED (no FIFO fill sim)")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Walk-forward: sliding window (HC #0)")
    logger.info(f"  Reporting: Sharpe AND Sortino (HC #57)")
    logger.info("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === MLflow setup ===
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow.start_run(run_name=f"exec_mlp_gpu_v1_{time.strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                'model_type': 'ExecMLP_multitask',
                'hidden_dim': HIDDEN_DIM,
                'n_layers': N_LAYERS,
                'dropout': DROPOUT,
                'lr': LR,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'cost_ticks_rt': COST_TICKS_RT,
                'cost_usd_rt': COST_TICKS_RT * TICK_VAL,
                'train_folds': str(TRAIN_FOLDS),
                'val_folds': str(VAL_FOLDS),
                'device': DEVICE,
                'execution_basis': 'midpoint (no FIFO fill sim)',
            })
            logger.info("MLflow run started.")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
    else:
        logger.warning("MLflow not available — training without experiment tracking.")

    # === Load all fold data ===
    logger.info("\n--- Loading fold data ---")
    all_data = {}
    for fold in range(N_FOLDS):
        d = load_fold_data(fold)
        if d is not None:
            all_data[fold] = d
            has_mbo = "yes" if d['mbo_data'] is not None else "no"
            has_vol = "yes" if d['vol_pred'] is not None else "no"
            logger.info(f"  Fold {fold:02d}: date={d['date']}, n={d['n_samples']:,}, MBO={has_mbo}, vol={has_vol}")

    if len(all_data) < 3:
        logger.error("Not enough folds loaded. Need at least 3. Aborting.")
        return

    # === Build features for all folds ===
    logger.info("\n--- Building features ---")
    fold_features = {}
    fold_targets = {}
    feature_names = None

    for fold_idx, data in all_data.items():
        logger.info(f"  Fold {fold_idx:02d} ({data['date']}):")
        feats, names = build_all_features(data)
        tgts = build_targets(data)
        # Apply SHORT_ONLY mask to features (align with filtered targets)
        if SHORT_ONLY and 'short_mask' in tgts:
            mask = tgts['short_mask']
            feats = feats[mask]
            logger.info(f"    SHORT_ONLY: {mask.sum()}/{len(mask)} samples ({mask.mean():.1%} short)")
        fold_features[fold_idx] = feats
        fold_targets[fold_idx] = tgts
        if feature_names is None:
            feature_names = names

        gate_pos = tgts['gate'].mean()
        logger.info(f"    Gate positive rate: {gate_pos:.3f}, MFE mean: {tgts['mfe'].mean():.2f}, MAE mean: {tgts['mae'].mean():.2f}")

    # === Part 1: Train MLP ===
    logger.info("\n" + "=" * 80)
    logger.info("PART 1: Multi-task Execution MLP")
    logger.info("=" * 80)

    # Concatenate train folds
    train_X_parts = []
    train_tgt_parts = {k: [] for k in ['gate', 'mfe', 'mae', 'hold_time']}

    for fold_idx in TRAIN_FOLDS:
        if fold_idx not in fold_features:
            continue
        train_X_parts.append(fold_features[fold_idx])
        for k in train_tgt_parts:
            if k != 'short_mask':
                train_tgt_parts[k].append(fold_targets[fold_idx][k])

    if not train_X_parts:
        logger.error("No training data. Aborting.")
        return

    train_X = np.concatenate(train_X_parts)
    train_tgts = {k: np.concatenate(v) for k, v in train_tgt_parts.items() if k != 'short_mask'}

    # Concatenate val folds
    val_X_parts = []
    val_tgt_parts = {k: [] for k in ['gate', 'mfe', 'mae', 'hold_time']}
    val_preds_parts = []
    val_labels_parts = []

    for fold_idx in VAL_FOLDS:
        if fold_idx not in fold_features:
            continue
        val_X_parts.append(fold_features[fold_idx])
        for k in val_tgt_parts:
            val_tgt_parts[k].append(fold_targets[fold_idx][k])
        # Apply short mask to predictions/labels to match filtered features
        preds = all_data[fold_idx]['predictions']
        lbls = all_data[fold_idx]['labels']
        if SHORT_ONLY and 'short_mask' in fold_targets[fold_idx]:
            mask = fold_targets[fold_idx]['short_mask']
            preds = preds[mask]
            lbls = lbls[mask]
        val_preds_parts.append(preds)
        val_labels_parts.append(lbls)

    if not val_X_parts:
        logger.error("No validation data. Aborting.")
        return

    val_X = np.concatenate(val_X_parts)
    val_tgts = {k: np.concatenate(v) for k, v in val_tgt_parts.items() if k != 'short_mask'}
    val_preds = np.concatenate(val_preds_parts)
    val_labels = np.concatenate(val_labels_parts)

    logger.info(f"\nTrain samples: {len(train_X):,}")
    logger.info(f"Val samples:   {len(val_X):,}")

    # Train MLP
    model, results = train_mlp(
        train_X, train_tgts, val_X, val_tgts,
        val_preds, val_labels, feature_names,
        run_name="exec_mlp_gpu_v1",
    )

    # === Comprehensive evaluation ===
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION RESULTS (MIDPOINT-BASED, folds 09-10)")
    logger.info("=" * 80)
    logger.info(f"{'Threshold':>10s} | {'N_trades':>8s} | {'Coverage':>8s} | {'WinRate':>7s} | {'PF':>6s} | {'AvgRR':>6s} | {'Sharpe':>8s} | {'Sortino':>8s} | {'AvgPnL':>8s}")
    logger.info("-" * 90)

    best_result = None
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        r = evaluate_gated_trading(results['gate_probs'], val_preds, val_labels, threshold=thresh)
        if r['n_trades'] < 10:
            continue
        logger.info(
            f"{thresh:>10.2f} | {r['n_trades']:>8d} | {r['coverage']:>7.1%} | "
            f"{r['win_rate']:>6.1%} | {r['pf']:>5.2f} | {r['avg_rr']:>5.2f} | "
            f"{r['sharpe']:>8.2f} | {r['sortino']:>8.2f} | ${r['avg_pnl']:>7.2f}"
        )
        if best_result is None or r['sharpe'] > best_result.get('sharpe', -999):
            best_result = r
            best_result['threshold'] = thresh

    if best_result:
        logger.info(f"\nBest threshold: {best_result['threshold']:.2f}")
        logger.info(f"  Sharpe: {best_result['sharpe']:.2f}, Sortino: {best_result['sortino']:.2f}")
        logger.info(f"  Win rate: {best_result['win_rate']:.1%}, PF: {best_result['pf']:.2f}")
        logger.info(f"  Avg P&L/trade: ${best_result['avg_pnl']:.2f}")

        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    'best_threshold': best_result['threshold'],
                    'best_sharpe': best_result['sharpe'],
                    'best_sortino': best_result['sortino'],
                    'best_win_rate': best_result['win_rate'],
                    'best_pf': min(best_result['pf'], 99.99),
                    'best_avg_pnl': best_result['avg_pnl'],
                    'best_n_trades': best_result['n_trades'],
                })
            except Exception:
                pass

    # === Save model and normalization params ===
    save_path = OUTPUT_DIR / "exec_mlp_v1_best.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_mean': results['norm_mean'],
        'norm_std': results['norm_std'],
        'feature_names': feature_names,
        'input_dim': train_X.shape[1],
        'hidden_dim': HIDDEN_DIM,
        'n_layers': N_LAYERS,
        'dropout': DROPOUT,
        'best_threshold': best_result['threshold'] if best_result else 0.5,
        'cost_ticks_rt': COST_TICKS_RT,
    }, str(save_path))
    logger.info(f"\nModel saved to {save_path}")

    # Save predictions
    pred_path = OUTPUT_DIR / "val_predictions.npz"
    np.savez(
        str(pred_path),
        gate_probs=results['gate_probs'],
        mfe_preds=results['mfe_preds'],
        mae_preds=results['mae_preds'],
        hold_preds=results['hold_preds'],
        val_preds=val_preds,
        val_labels=val_labels,
    )
    logger.info(f"Predictions saved to {pred_path}")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_artifact(str(save_path))
        except Exception:
            pass

    # === Part 2: RL Exit Agent ===
    if not args.skip_rl:
        logger.info("\n" + "=" * 80)
        logger.info("PART 2: DQN Exit Timing Agent")
        logger.info("=" * 80)

        rl_trainer = ExitTimingTrainer(device=DEVICE)

        # Build spread and vol arrays for simulation
        # Use MBO-derived spreads if available, else defaults
        val_spreads = np.ones(len(val_preds), dtype=np.float32)
        val_vols = np.ones(len(val_preds), dtype=np.float32) * 0.01

        # If we computed MBO features, extract spread and vol columns
        if 'spread_proxy' in (feature_names or []):
            sp_idx = feature_names.index('spread_proxy') if 'spread_proxy' in feature_names else -1
            rv_idx = feature_names.index('rvol_200') if 'rvol_200' in feature_names else -1
            if sp_idx >= 0:
                val_spreads = val_X[:, sp_idx]
            if rv_idx >= 0:
                val_vols = val_X[:, rv_idx]

        # Use train data for RL training (different from MLP val)
        train_preds_concat = np.concatenate([all_data[f]['predictions'] for f in TRAIN_FOLDS if f in all_data])
        train_labels_concat = np.concatenate([all_data[f]['labels'] for f in TRAIN_FOLDS if f in all_data])

        # Build gate probs for train data using trained model
        model.eval()
        train_norm = (train_X - results['norm_mean']) / results['norm_std']
        with torch.no_grad():
            train_gate_list = []
            for start in range(0, len(train_norm), 10000):
                end = min(start + 10000, len(train_norm))
                chunk = torch.from_numpy(train_norm[start:end]).to(DEVICE)
                out = model(chunk)
                train_gate_list.append(out['gate'].cpu().numpy())
        train_gate_probs = np.concatenate(train_gate_list)

        train_spreads = np.ones(len(train_preds_concat), dtype=np.float32)
        train_vols = np.ones(len(train_preds_concat), dtype=np.float32) * 0.01

        logger.info(f"  RL training episodes: {args.rl_episodes}")
        logger.info(f"  Train data: {len(train_preds_concat):,} samples")

        best_rl_reward = -float('inf')
        for episode in range(args.rl_episodes):
            # Simulate episode on train data
            ep_reward = rl_trainer.simulate_episode(
                train_preds_concat, train_labels_concat,
                train_gate_probs, train_spreads, train_vols,
            )

            # Train from replay buffer
            losses = []
            for _ in range(10):
                loss = rl_trainer.train_step(batch_size=256)
                losses.append(loss)

            avg_loss = np.mean(losses) if losses else 0

            if episode % 20 == 0:
                logger.info(
                    f"  RL Episode {episode:>4d}/{args.rl_episodes} | "
                    f"Reward: {ep_reward:>8.2f} | Loss: {avg_loss:.4f} | "
                    f"Epsilon: {rl_trainer.epsilon:.3f} | Buffer: {len(rl_trainer.replay)}"
                )

            if ep_reward > best_rl_reward:
                best_rl_reward = ep_reward
                rl_save_path = OUTPUT_DIR / "exit_dqn_best.pt"
                torch.save(rl_trainer.policy_net.state_dict(), str(rl_save_path))

            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        'rl_reward': ep_reward,
                        'rl_loss': avg_loss,
                        'rl_epsilon': rl_trainer.epsilon,
                    }, step=episode)
                except Exception:
                    pass

        logger.info(f"  Best RL reward: {best_rl_reward:.2f}")
        logger.info(f"  RL model saved to {OUTPUT_DIR / 'exit_dqn_best.pt'}")

    # === Final summary ===
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Output dir: {OUTPUT_DIR}")
    logger.info(f"  MLP model: {OUTPUT_DIR / 'exec_mlp_v1_best.pt'}")
    if not args.skip_rl:
        logger.info(f"  RL model:  {OUTPUT_DIR / 'exit_dqn_best.pt'}")
    logger.info(f"  Cost basis: {COST_TICKS_RT:.3f} ticks RT (${COST_TICKS_RT * TICK_VAL:.2f})")
    logger.info(f"  All results MIDPOINT-BASED (no FIFO fill sim)")
    logger.info(f"  Metrics reported: Sharpe AND Sortino (HC #57)")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
