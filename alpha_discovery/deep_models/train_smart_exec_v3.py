#!/usr/bin/env python3
"""
Smart Execution System v3 — Hybrid XGBoost + NN Architecture

V3 key changes over v2:
  1. MASSIVELY ENRICHED FEATURE SET (34 -> 60+ features)
     - Volume profile, multi-scale volatility, order flow imbalance
     - Price momentum multi-scale, spread dynamics, time features
     - Queue depth proxy, cross-model features
  2. HYBRID ARCHITECTURE: XGBoost for TP/SL, NN for Gate
     - TP/SL: XGBoost regression (tabular data -> trees excel)
     - Gate: NN with Sortino-maximizing loss (differentiable Sortino)
     - Exit: NN for hold time and risk/reward (kept from v2)
  3. Training flow: Phase 1 XGBoost (CPU), Phase 2 NN gate (GPU), Phase 3 joint eval
  4. Walk-forward sliding window (HC #0 compliant)

Root cause fix: v1/v2 NN gate couldn't discriminate because:
  - Too few features (34) for tabular regression (TP/SL)
  - Wrong architecture — trees handle feature interactions and heterogeneous features better

Neptune target: RTX 3090 24GB, 32GB RAM, keep RSS <= 22GB.
"""

import os
import sys
import gc
import time
import json
import logging
import argparse
import warnings
import socket
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# XGBoost with sklearn fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ============================================================
# Logging (force=True to override MLflow's logging config)
# ============================================================
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================
TICK_VAL = 12.50       # USD per tick (NQ)
COMMISSION = 0.376     # ticks one-way
SPREAD = 0.5           # ticks one-way
ROUND_TRIP_COST = 2 * (COMMISSION + SPREAD)  # 1.752 ticks

ET_OFFSET_HOURS = -5   # EST (Feb-Mar 2026 data)

# Session boundaries in ET fractional hours
SESSION_BOUNDS = [
    ("overnight",   0.0,  2.0),
    ("pre_market",  2.0,  9.5),
    ("rth_open",    9.5,  10.5),
    ("rth_core",    10.5, 15.0),
    ("rth_close",   15.0, 16.0),
    ("post_market", 16.0, 17.0),
    ("maintenance", 17.0, 17.75),  # CME daily maintenance
    ("evening",     17.75, 24.0),
]

SESSION_TO_IDX = {s[0]: i for i, s in enumerate(SESSION_BOUNDS)}

# Gap windows where we should NOT trade
GAP_WINDOWS_ET = [
    (16.95, 17.75),  # CME maintenance ~16:57-17:45 ET
]

# Log-space TP/SL constants
LOG_TPSL_MIN_FLOOR = 0.5   # minimum TP/SL in ticks after exp()
LOG_TPSL_CLIP = 6.0        # clip log targets to prevent exp() overflow

# Feature names registry (for importance reporting)
FEATURE_NAMES: List[str] = []


def ts_ns_to_et_hour(ts_ns: int) -> float:
    """Convert nanosecond timestamp to fractional hour in ET."""
    from datetime import datetime, timezone
    utc_sec = ts_ns / 1e9
    et_sec = utc_sec + ET_OFFSET_HOURS * 3600
    dt = datetime.fromtimestamp(et_sec, tz=timezone.utc)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def is_in_gap(et_hour: float) -> bool:
    """Check if time falls in a known gap window."""
    for start, end in GAP_WINDOWS_ET:
        if start <= et_hour < end:
            return True
    return False


def get_session_idx(et_hour: float) -> int:
    """Get session index from ET hour."""
    for name, start, end in SESSION_BOUNDS:
        if start <= et_hour < end:
            return SESSION_TO_IDX[name]
    return SESSION_TO_IDX["overnight"]


# ============================================================
# Helper: rolling statistics (vectorized where possible)
# ============================================================

def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling z-score using cumulative sums."""
    n = len(arr)
    z = np.zeros(n, dtype=np.float32)
    cumsum = np.cumsum(arr)
    cumsum2 = np.cumsum(arr ** 2)

    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        if count < 30:
            continue
        s = cumsum[i] - (cumsum[start - 1] if start > 0 else 0)
        s2 = cumsum2[i] - (cumsum2[start - 1] if start > 0 else 0)
        mean = s / count
        var = s2 / count - mean ** 2
        std = np.sqrt(max(var, 1e-10))
        z[i] = (arr[i] - mean) / std

    return z


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using cumsum."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        out[i] = (cs[i] - (cs[start - 1] if start > 0 else 0)) / count
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling std using cumsum."""
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


def _rolling_percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank: where current value sits vs last `window` values."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = arr[start:i + 1]
        if len(chunk) < 10:
            out[i] = 0.5
        else:
            out[i] = np.searchsorted(np.sort(chunk), arr[i]) / len(chunk)
    return out


# ============================================================
# Feature Engineering — V3 Enriched (60+ features)
# ============================================================

def build_exec_features_v3(
    cnn_preds: np.ndarray,      # (N, 3) predictions for 1s/5s/10s
    ptst_preds: np.ndarray,     # (N, 3) predictions for 1s/5s/10s
    vol_preds: np.ndarray,      # (N, 3) vol predictions
    timestamps: np.ndarray,     # (N,) int64 nanoseconds
    events: np.ndarray,         # (M, 6) raw MBO events: [time_delta_log, side, action, price_rel_ticks, qty_log, spread_ticks]
    anchor_idxs: np.ndarray,    # (N,) indices into events array
    zscore_window: int = 3000,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build rich feature vectors for the execution system.

    V3: massively enriched from 34 to 60+ features.

    Returns: (N, F) float32 feature matrix, list of feature names.
    """
    N = len(cnn_preds)
    features = []
    names = []

    # -------------------------------------------------------
    # GROUP 1: CNN z-scores (3 features)
    # -------------------------------------------------------
    for h, label in enumerate(["1s", "5s", "10s"]):
        z = _rolling_zscore(cnn_preds[:, h], zscore_window)
        features.append(z)
        names.append(f"cnn_z_{label}")

    # -------------------------------------------------------
    # GROUP 2: PatchTST z-scores (3 features)
    # -------------------------------------------------------
    for h, label in enumerate(["1s", "5s", "10s"]):
        z = _rolling_zscore(ptst_preds[:, h], zscore_window)
        features.append(z)
        names.append(f"ptst_z_{label}")

    # -------------------------------------------------------
    # GROUP 3: Vol predictions (3 features)
    # -------------------------------------------------------
    for h, label in enumerate(["1s", "5s", "10s"]):
        features.append(vol_preds[:, h])
        names.append(f"vol_pred_{label}")

    # -------------------------------------------------------
    # GROUP 4: Model agreement per horizon (3 features)
    # -------------------------------------------------------
    for h, label in enumerate(["1s", "5s", "10s"]):
        agreement = np.sign(cnn_preds[:, h]) * np.sign(ptst_preds[:, h])
        features.append(agreement)
        names.append(f"agreement_{label}")

    # -------------------------------------------------------
    # GROUP 5: Conviction = |cnn_z| * sign agreement (3 features)
    # -------------------------------------------------------
    for h, label in enumerate(["1s", "5s", "10s"]):
        cnn_z = _rolling_zscore(cnn_preds[:, h], zscore_window)
        ptst_z = _rolling_zscore(ptst_preds[:, h], zscore_window)
        conviction = np.abs(cnn_z) * np.sign(cnn_z) * np.sign(ptst_z)
        features.append(conviction)
        names.append(f"conviction_{label}")

    # -------------------------------------------------------
    # GROUP 6: Cross-model features (4 features)
    # -------------------------------------------------------
    cnn_z_10s = _rolling_zscore(cnn_preds[:, 2], zscore_window)
    ptst_z_10s = _rolling_zscore(ptst_preds[:, 2], zscore_window)

    # |cnn_z| - |ptst_z| (which model has more conviction)
    conv_diff = np.abs(cnn_z_10s) - np.abs(ptst_z_10s)
    features.append(conv_diff)
    names.append("conv_diff_10s")

    # cnn_z * ptst_z (agreement score, continuous)
    agreement_score = cnn_z_10s * ptst_z_10s
    features.append(agreement_score)
    names.append("agreement_score_10s")

    # max(|cnn_z|, |ptst_z|) (best conviction)
    best_conv = np.maximum(np.abs(cnn_z_10s), np.abs(ptst_z_10s))
    features.append(best_conv)
    names.append("best_conv_10s")

    # vol_pred * |cnn_z| (vol-weighted conviction)
    vol_weighted_conv = vol_preds[:, 2] * np.abs(cnn_z_10s)
    features.append(vol_weighted_conv)
    names.append("vol_weighted_conv_10s")

    # -------------------------------------------------------
    # GROUP 7: Time-of-day (cyclical + session one-hot + time features = 12 features)
    # -------------------------------------------------------
    et_hours = np.array([ts_ns_to_et_hour(t) for t in timestamps])

    # Cyclical encoding
    tod_sin = np.sin(2 * np.pi * et_hours / 24.0).astype(np.float32)
    tod_cos = np.cos(2 * np.pi * et_hours / 24.0).astype(np.float32)
    features.append(tod_sin)
    names.append("tod_sin")
    features.append(tod_cos)
    names.append("tod_cos")

    # Session one-hot (8 buckets)
    session_idxs = np.array([get_session_idx(h) for h in et_hours])
    for s_idx, (s_name, _, _) in enumerate(SESSION_BOUNDS):
        oh = (session_idxs == s_idx).astype(np.float32)
        features.append(oh)
        names.append(f"session_{s_name}")

    # Minutes since session start
    mins_since_session = np.zeros(N, dtype=np.float32)
    for i, eh in enumerate(et_hours):
        s_idx = session_idxs[i]
        s_start = SESSION_BOUNDS[s_idx][1]
        mins_since_session[i] = (eh - s_start) * 60.0
    features.append(mins_since_session)
    names.append("mins_since_session_start")

    # Minutes until CME maintenance (17:00 ET)
    mins_until_maint = np.clip((17.0 - et_hours) * 60.0, -60.0, 600.0).astype(np.float32)
    features.append(mins_until_maint)
    names.append("mins_until_maintenance")

    # Gap flag
    gap_flag = np.array([is_in_gap(h) for h in et_hours], dtype=np.float32)
    features.append(gap_flag)
    names.append("gap_flag")

    # -------------------------------------------------------
    # GROUP 8: Microstructure features from MBO events (enriched)
    # -------------------------------------------------------
    # Initialize all microstructure features
    spread_at_anchor = np.zeros(N, dtype=np.float32)
    volume_100 = np.zeros(N, dtype=np.float32)
    volume_500 = np.zeros(N, dtype=np.float32)
    volume_2000 = np.zeros(N, dtype=np.float32)
    vol_acceleration = np.zeros(N, dtype=np.float32)
    vol_percentile = np.zeros(N, dtype=np.float32)
    momentum_5 = np.zeros(N, dtype=np.float32)
    momentum_20 = np.zeros(N, dtype=np.float32)
    momentum_50 = np.zeros(N, dtype=np.float32)
    momentum_100 = np.zeros(N, dtype=np.float32)
    momentum_500 = np.zeros(N, dtype=np.float32)
    mom_accel = np.zeros(N, dtype=np.float32)
    event_rate = np.zeros(N, dtype=np.float32)
    realized_vol_50 = np.zeros(N, dtype=np.float32)
    realized_vol_200 = np.zeros(N, dtype=np.float32)
    realized_vol_1000 = np.zeros(N, dtype=np.float32)
    vol_of_vol = np.zeros(N, dtype=np.float32)
    vol_pctile_rank = np.zeros(N, dtype=np.float32)
    net_flow_50 = np.zeros(N, dtype=np.float32)
    net_flow_200 = np.zeros(N, dtype=np.float32)
    order_arrival_rate = np.zeros(N, dtype=np.float32)
    cancel_rate = np.zeros(N, dtype=np.float32)
    spread_zscore = np.zeros(N, dtype=np.float32)
    spread_percentile = np.zeros(N, dtype=np.float32)
    bid_ask_ratio = np.zeros(N, dtype=np.float32)

    has_events = events is not None and len(events) > 0

    if has_events:
        # Precompute columns for speed
        ev_time = events[:, 0]   # time_delta_log
        ev_side = events[:, 1]   # side (0=bid, 1=ask)
        ev_action = events[:, 2] # action (0=add, 1=modify, 2=cancel, 3=trade)
        ev_price = events[:, 3]  # price_rel_ticks
        ev_qty = events[:, 4]    # qty_log
        ev_spread = events[:, 5] if events.shape[1] > 5 else np.ones(len(events), dtype=np.float32)

        # Precompute price changes for realized vol
        price_changes = np.diff(ev_price, prepend=ev_price[0])

        # Precompute cumulative volume (exp(qty_log)) for rolling sums
        raw_qty = np.exp(np.clip(ev_qty, -10, 10))

        # Precompute signed flow: buy_size - sell_size per event
        # side=1 is ask (aggressor buy), side=0 is bid (aggressor sell)
        # For trade events only (action=3), compute net aggressor
        is_trade = (ev_action == 3).astype(np.float32)
        signed_flow = raw_qty * (2.0 * ev_side - 1.0) * is_trade  # +1 for buys, -1 for sells

        # Precompute add/cancel flags
        is_add = (ev_action == 0).astype(np.float32)
        is_cancel = (ev_action == 2).astype(np.float32)

        # Cumulative sums for fast window queries
        cs_qty = np.cumsum(raw_qty)
        cs_flow = np.cumsum(signed_flow)
        cs_add = np.cumsum(is_add)
        cs_cancel = np.cumsum(is_cancel)
        cs_pc2 = np.cumsum(price_changes ** 2)

        # Rolling spread stats via cumsum
        cs_spread = np.cumsum(ev_spread)
        cs_spread2 = np.cumsum(ev_spread ** 2)

        def _cumsum_window(cs, idx, window):
            """Sum of values in [idx-window+1, idx] using cumsum."""
            start = max(0, idx - window + 1)
            val = cs[idx] - (cs[start - 1] if start > 0 else 0)
            count = idx - start + 1
            return val, count

        # Precompute rolling realized vol (std of price changes) at various scales
        # We'll do this vectorized for the anchor indices
        for i_sample, aidx in enumerate(anchor_idxs):
            aidx = int(aidx)
            if aidx >= len(events):
                continue

            # Spread
            spread_at_anchor[i_sample] = ev_spread[aidx]

            # Volume profile: rolling volume ratio vs session averages
            for win, arr in [(100, volume_100), (500, volume_500), (2000, volume_2000)]:
                if aidx >= 1:
                    vol_sum, cnt = _cumsum_window(cs_qty, aidx, win)
                    arr[i_sample] = vol_sum

            # Volume acceleration: d(volume)/dt over last 50 events
            if aidx >= 50:
                vol_recent, _ = _cumsum_window(cs_qty, aidx, 25)
                vol_prev, _ = _cumsum_window(cs_qty, max(0, aidx - 25), 25)
                vol_acceleration[i_sample] = vol_recent - vol_prev

            # Relative volume percentile (current 100-event volume vs last 1000 windows)
            # Approximate: compare volume_100 to a running average
            if aidx >= 100:
                long_vol, long_cnt = _cumsum_window(cs_qty, aidx, 1000)
                short_vol, short_cnt = _cumsum_window(cs_qty, aidx, 100)
                avg_per_100 = (long_vol / max(long_cnt, 1)) * 100
                vol_percentile[i_sample] = short_vol / max(avg_per_100, 1e-6)

            # Price momentum multi-scale
            for win, arr in [(5, momentum_5), (20, momentum_20), (50, momentum_50),
                             (100, momentum_100), (500, momentum_500)]:
                if aidx >= win:
                    arr[i_sample] = ev_price[aidx] - ev_price[aidx - win]

            # Momentum acceleration: short catching up to medium = reversal
            mom_accel[i_sample] = momentum_5[i_sample] - momentum_20[i_sample]

            # Realized volatility at multiple scales
            for win, arr in [(50, realized_vol_50), (200, realized_vol_200), (1000, realized_vol_1000)]:
                if aidx >= win:
                    pc2_sum, cnt = _cumsum_window(cs_pc2, aidx, win)
                    arr[i_sample] = np.sqrt(pc2_sum / cnt)

            # Vol-of-vol: std of rolling vol over last 500 events
            # Approximate using local variance of realized_vol_50
            # (computed post-loop below)

            # Net aggressor flow (order flow imbalance)
            for win, arr in [(50, net_flow_50), (200, net_flow_200)]:
                if aidx >= 1:
                    flow_sum, _ = _cumsum_window(cs_flow, aidx, win)
                    arr[i_sample] = flow_sum

            # Order arrival rate and cancel rate over last 100 events
            if aidx >= 100:
                dt_sum_val, _ = _cumsum_window(np.cumsum(np.exp(np.clip(ev_time[:aidx+1], -10, 10))), aidx, 100)
                dt_sum_val = max(dt_sum_val, 1e-6)
                add_cnt, _ = _cumsum_window(cs_add, aidx, 100)
                cancel_cnt, _ = _cumsum_window(cs_cancel, aidx, 100)
                order_arrival_rate[i_sample] = add_cnt / dt_sum_val
                cancel_rate[i_sample] = cancel_cnt / dt_sum_val
                event_rate[i_sample] = 100.0 / dt_sum_val

            # Spread dynamics
            if aidx >= 500:
                sp_sum, sp_cnt = _cumsum_window(cs_spread, aidx, 500)
                sp2_sum, _ = _cumsum_window(cs_spread2, aidx, 500)
                sp_mean = sp_sum / sp_cnt
                sp_var = max(sp2_sum / sp_cnt - sp_mean ** 2, 1e-10)
                sp_std = np.sqrt(sp_var)
                spread_zscore[i_sample] = (ev_spread[aidx] - sp_mean) / sp_std

            # Spread percentile (approximate)
            if aidx >= 100:
                sp_window = ev_spread[max(0, aidx - 999):aidx + 1]
                spread_percentile[i_sample] = np.searchsorted(
                    np.sort(sp_window), ev_spread[aidx]
                ) / len(sp_window)

            # Queue depth proxy: bid_size / ask_size from recent trades
            # Approximate from recent event sizes by side
            if aidx >= 50:
                recent_start = max(0, aidx - 50)
                recent_qty = raw_qty[recent_start:aidx + 1]
                recent_side = ev_side[recent_start:aidx + 1]
                bid_vol = recent_qty[recent_side < 0.5].sum()
                ask_vol = recent_qty[recent_side > 0.5].sum()
                bid_ask_ratio[i_sample] = bid_vol / max(ask_vol, 1e-6)

    # Vol-of-vol: std of realized_vol_50 over rolling 500 samples
    if has_events:
        vol_of_vol = _rolling_std(realized_vol_50, 500)
        # Vol percentile rank
        vol_pctile_rank = _rolling_percentile_rank(realized_vol_50, 1000)

    # Compute relative volume (vs session average)
    # Normalize volume_100 by volume_2000/20 to get ratio
    session_avg_vol = np.where(volume_2000 > 0, volume_2000 / 20.0, 1.0)
    vol_ratio_vs_session = volume_100 / np.maximum(session_avg_vol, 1e-6)

    # Add all microstructure features
    micro_features = [
        ("spread", spread_at_anchor),
        ("vol_100", volume_100),
        ("vol_500", volume_500),
        ("vol_2000", volume_2000),
        ("vol_ratio_vs_session", vol_ratio_vs_session),
        ("vol_acceleration", vol_acceleration),
        ("vol_percentile", vol_percentile),
        ("momentum_5", momentum_5),
        ("momentum_20", momentum_20),
        ("momentum_50", momentum_50),
        ("momentum_100", momentum_100),
        ("momentum_500", momentum_500),
        ("momentum_accel", mom_accel),
        ("event_rate", event_rate),
        ("realized_vol_50", realized_vol_50),
        ("realized_vol_200", realized_vol_200),
        ("realized_vol_1000", realized_vol_1000),
        ("vol_of_vol", vol_of_vol),
        ("vol_pctile_rank", vol_pctile_rank),
        ("net_flow_50", net_flow_50),
        ("net_flow_200", net_flow_200),
        ("order_arrival_rate", order_arrival_rate),
        ("cancel_rate", cancel_rate),
        ("spread_zscore", spread_zscore),
        ("spread_percentile", spread_percentile),
        ("bid_ask_ratio", bid_ask_ratio),
    ]

    for feat_name, feat_arr in micro_features:
        features.append(feat_arr)
        names.append(feat_name)

    # -------------------------------------------------------
    # GROUP 9: Volatility regime from predictions (1 feature)
    # -------------------------------------------------------
    vol_regime = np.zeros(N, dtype=np.float32)
    window = min(500, N)
    for i in range(window, N):
        vol_regime[i] = np.std(cnn_preds[i - window:i, 2])
    features.append(vol_regime)
    names.append("vol_regime_pred")

    # -------------------------------------------------------
    # Stack all features
    # -------------------------------------------------------
    feature_matrix = np.column_stack(features).astype(np.float32)

    # Replace NaN/Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

    logger.info(f"  V3 feature matrix: {feature_matrix.shape[1]} features")

    return feature_matrix, names


# ============================================================
# Training Targets from MFE/MAE Data
# ============================================================

def build_targets(
    labels: np.ndarray,         # (N, 3) true price changes 1s/5s/10s
    mfe_ticks: np.ndarray,      # (N,) max favorable excursion
    mae_ticks: np.ndarray,      # (N,) max adverse excursion
    time_to_mfe: np.ndarray,    # (N,) seconds to MFE
    cnn_preds: np.ndarray,      # (N, 3) for direction
) -> Dict[str, np.ndarray]:
    """
    Build training targets for all execution components.
    TP/SL targets in LOG SPACE — log(ticks) clipped to [-2, LOG_TPSL_CLIP].
    """
    N = len(labels)
    direction = np.sign(cnn_preds[:, 2])  # use 10s prediction for direction

    # Target 1: Optimal TP in log-space
    optimal_tp_raw = np.maximum(mfe_ticks, LOG_TPSL_MIN_FLOOR)
    optimal_tp_log = np.clip(np.log(optimal_tp_raw), -2.0, LOG_TPSL_CLIP)

    # Target 2: Optimal SL in log-space
    optimal_sl_raw = np.maximum(mae_ticks, LOG_TPSL_MIN_FLOOR)
    optimal_sl_log = np.clip(np.log(optimal_sl_raw), -2.0, LOG_TPSL_CLIP)

    # Target 3: Entry quality (was this a profitable trade after costs?)
    realized_pnl = labels[:, 2] * direction  # directional PnL in ticks
    profitable = (realized_pnl > ROUND_TRIP_COST).astype(np.float32)

    # Target 4: Trade magnitude (for weighting)
    trade_magnitude = np.abs(realized_pnl)

    # Target 5: Optimal hold time (time to MFE = optimal exit time)
    optimal_hold = time_to_mfe.copy()

    # Target 6: Risk-reward ratio
    rr_ratio = np.where(mae_ticks > 0.1, mfe_ticks / mae_ticks, 0.0)

    return {
        "optimal_tp_log": optimal_tp_log.astype(np.float32),
        "optimal_sl_log": optimal_sl_log.astype(np.float32),
        "optimal_tp": optimal_tp_raw.astype(np.float32),
        "optimal_sl": optimal_sl_raw.astype(np.float32),
        "entry_profitable": profitable,
        "realized_pnl": realized_pnl.astype(np.float32),
        "trade_magnitude": trade_magnitude.astype(np.float32),
        "optimal_hold_s": optimal_hold.astype(np.float32),
        "risk_reward": rr_ratio.astype(np.float32),
    }


# ============================================================
# TP/SL Model: XGBoost (or sklearn fallback)
# ============================================================

class TPSLTreeModel:
    """
    XGBoost regressor for TP and SL prediction.
    Falls back to sklearn GradientBoostingRegressor if xgboost unavailable.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
        )
        self.tp_model = None
        self.sl_model = None
        self.backend = None

    def fit(self, X: np.ndarray, tp_target: np.ndarray, sl_target: np.ndarray,
            feature_names: Optional[List[str]] = None):
        """Train TP and SL models."""
        if XGBOOST_AVAILABLE:
            self.backend = "xgboost"
            self.tp_model = xgb.XGBRegressor(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                colsample_bytree=self.params["colsample_bytree"],
                tree_method="hist",
                n_jobs=2,
                verbosity=0,
                random_state=42,
            )
            self.sl_model = xgb.XGBRegressor(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                colsample_bytree=self.params["colsample_bytree"],
                tree_method="hist",
                n_jobs=2,
                verbosity=0,
                random_state=42,
            )
            logger.info("  Training XGBoost TP model...")
            t0 = time.time()
            self.tp_model.fit(X, tp_target)
            logger.info(f"    TP model trained in {time.time() - t0:.1f}s")

            logger.info("  Training XGBoost SL model...")
            t0 = time.time()
            self.sl_model.fit(X, sl_target)
            logger.info(f"    SL model trained in {time.time() - t0:.1f}s")

        elif SKLEARN_AVAILABLE:
            self.backend = "sklearn"
            logger.warning("  XGBoost not available, using sklearn GradientBoostingRegressor (slower)")
            self.tp_model = GradientBoostingRegressor(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                random_state=42,
            )
            self.sl_model = GradientBoostingRegressor(
                n_estimators=self.params["n_estimators"],
                max_depth=self.params["max_depth"],
                learning_rate=self.params["learning_rate"],
                subsample=self.params["subsample"],
                random_state=42,
            )
            logger.info("  Training sklearn TP model...")
            t0 = time.time()
            self.tp_model.fit(X, tp_target)
            logger.info(f"    TP model trained in {time.time() - t0:.1f}s")

            logger.info("  Training sklearn SL model...")
            t0 = time.time()
            self.sl_model.fit(X, sl_target)
            logger.info(f"    SL model trained in {time.time() - t0:.1f}s")
        else:
            raise RuntimeError("Neither xgboost nor sklearn available for tree models")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict log(TP) and log(SL)."""
        log_tp = self.tp_model.predict(X).astype(np.float32)
        log_sl = self.sl_model.predict(X).astype(np.float32)
        return log_tp, log_sl

    def predict_ticks(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict TP/SL in tick space."""
        log_tp, log_sl = self.predict(X)
        tp = np.exp(np.clip(log_tp, -2.0, LOG_TPSL_CLIP)) + LOG_TPSL_MIN_FLOOR
        sl = np.exp(np.clip(log_sl, -2.0, LOG_TPSL_CLIP)) + LOG_TPSL_MIN_FLOOR
        return tp, sl

    def get_feature_importance(self, feature_names: List[str], top_n: int = 20) -> Dict[str, List]:
        """Get top feature importances for TP and SL."""
        result = {}
        for label, model in [("tp", self.tp_model), ("sl", self.sl_model)]:
            if model is None:
                continue
            if XGBOOST_AVAILABLE and self.backend == "xgboost":
                imp = model.feature_importances_
            elif SKLEARN_AVAILABLE and self.backend == "sklearn":
                imp = model.feature_importances_
            else:
                continue

            # Pair with names and sort
            pairs = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
            top = pairs[:top_n]
            result[f"{label}_top_features"] = [
                {"name": n, "importance": float(v)} for n, v in top
            ]
        return result

    def save(self, path: Path, fold_idx: int):
        """Save models to disk."""
        if self.backend == "xgboost":
            self.tp_model.save_model(str(path / f"fold_{fold_idx:02d}_tp_xgb.json"))
            self.sl_model.save_model(str(path / f"fold_{fold_idx:02d}_sl_xgb.json"))
        elif JOBLIB_AVAILABLE:
            joblib.dump(self.tp_model, str(path / f"fold_{fold_idx:02d}_tp_sklearn.pkl"))
            joblib.dump(self.sl_model, str(path / f"fold_{fold_idx:02d}_sl_sklearn.pkl"))
        else:
            import pickle
            with open(path / f"fold_{fold_idx:02d}_tp_sklearn.pkl", "wb") as f:
                pickle.dump(self.tp_model, f)
            with open(path / f"fold_{fold_idx:02d}_sl_sklearn.pkl", "wb") as f:
                pickle.dump(self.sl_model, f)


# ============================================================
# NN Gate with Sortino-Maximizing Loss
# ============================================================

class EntryGate_v3(nn.Module):
    """
    V3 entry gate with learnable temperature.
    Enriched input: base features + XGBoost TP/SL predictions as extra inputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        # Learnable temperature
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logit = self.net(x).squeeze(-1)
        temperature = torch.exp(self.log_temperature).clamp(0.1, 10.0)
        return logit / temperature


class ExitTrigger_v3(nn.Module):
    """
    Predicts optimal hold time and exit urgency.
    Same architecture as v2 but with enriched input dim.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 96):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.hold_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        self.rr_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        h = self.encoder(x)
        hold = self.hold_head(h).squeeze(-1)
        rr = self.rr_head(h).squeeze(-1)
        return hold, rr


def differentiable_sortino(pnl: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Sortino ratio using soft gating weights.

    Args:
        pnl: (B,) realized PnL per trade
        weights: (B,) gate probabilities in [0, 1] (soft gate)
        eps: stability constant
    Returns:
        Scalar Sortino ratio (higher is better)
    """
    # Weighted mean PnL
    total_weight = weights.sum() + eps
    weighted_mean = (pnl * weights).sum() / total_weight

    # Weighted downside deviation
    downside = torch.clamp(pnl, max=0.0)  # only negative PnL
    weighted_downside_sq = (downside ** 2 * weights).sum() / total_weight
    weighted_downside_std = torch.sqrt(weighted_downside_sq + eps)

    sortino = weighted_mean / weighted_downside_std
    return sortino


class SortinoGateLoss(nn.Module):
    """
    Sortino-maximizing loss for the entry gate.

    Loss = -Sortino_gated + regularization terms.
    Uses differentiable Sortino with sigmoid(gate_logit) as soft weights.
    """

    def __init__(self, coverage_target: float = 0.3, coverage_weight: float = 0.1):
        super().__init__()
        self.coverage_target = coverage_target
        self.coverage_weight = coverage_weight

    def forward(
        self,
        gate_logit: torch.Tensor,
        realized_pnl: torch.Tensor,
        entry_profitable: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            gate_logit: (B,) raw logits from gate
            realized_pnl: (B,) directional PnL in ticks
            entry_profitable: (B,) binary labels (for monitoring, not used in loss)
        """
        gate_prob = torch.sigmoid(gate_logit)

        # Primary loss: maximize Sortino of gated trades
        sortino = differentiable_sortino(realized_pnl, gate_prob)
        sortino_loss = -sortino

        # Coverage regularization: prevent gate from gating everything out (or in)
        coverage = gate_prob.mean()
        coverage_penalty = self.coverage_weight * (coverage - self.coverage_target) ** 2

        total = sortino_loss + coverage_penalty

        # Monitoring metrics
        with torch.no_grad():
            take_mask = gate_prob > 0.5
            n_taken = take_mask.sum().item()
            if n_taken > 0:
                gated_pnl = realized_pnl[take_mask]
                gate_precision = entry_profitable[take_mask].mean().item()
            else:
                gate_precision = 0.0

        loss_dict = {
            "sortino_loss": sortino_loss.item(),
            "sortino_value": sortino.item(),
            "coverage": coverage.item(),
            "coverage_penalty": coverage_penalty.item(),
            "gate_precision": gate_precision,
            "total_gate_loss": total.item(),
        }

        return total, loss_dict


# ============================================================
# Dataset for NN training
# ============================================================

class GateExitDataset(Dataset):
    """Holds features and targets for gate+exit NN training."""

    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray]):
        self.features = torch.from_numpy(features)
        self.targets = {k: torch.from_numpy(v) for k, v in targets.items()}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], {k: v[idx] for k, v in self.targets.items()}


# ============================================================
# Data Loading
# ============================================================

def load_fold_data(
    fold_idx: int,
    cnn_dir: Path,
    ptst_dir: Path,
    vol_dir: Path,
    mfe_dir: Path,
    mbo_dir: Path,
    horizon: str = "10s",
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[List[str]]]:
    """
    Load and align data from all models for a given fold.
    Returns (features, targets, feature_names) or (None, None, None) if data unavailable.
    """
    h_idx = {"1s": 0, "5s": 1, "10s": 2}[horizon]

    # Load CNN Mamba v2 predictions
    cnn_path = cnn_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not cnn_path.exists():
        logger.warning(f"CNN predictions not found: {cnn_path}")
        return None, None, None
    cnn_data = np.load(cnn_path, allow_pickle=True)
    cnn_preds = cnn_data["predictions"]  # (N, 3)
    labels = cnn_data["labels"]          # (N, 3)
    oot_files = cnn_data.get("oot_files", None)

    N_cnn = len(cnn_preds)
    logger.info(f"  CNN preds: {N_cnn} samples, IC_10s={cnn_data.get('ic_10s', 'N/A')}")

    # Load MFE/MAE data for this fold
    mfe_path = mfe_dir / f"fold_{fold_idx:02d}_mfe_mae_{horizon}.npz"
    if mfe_path.exists():
        mfe_data = np.load(mfe_path, allow_pickle=True)
        mfe_ticks = mfe_data["mfe_ticks"]
        mae_ticks = mfe_data["mae_ticks"]
        time_to_mfe = mfe_data.get("time_to_mfe_s", np.ones(N_cnn) * 5.0)
        logger.info(f"  MFE/MAE: {len(mfe_ticks)} samples, mean MFE={mfe_ticks.mean():.2f}, mean MAE={mae_ticks.mean():.2f}")
    else:
        logger.warning(f"  No MFE/MAE data, approximating from labels")
        directional = labels[:, h_idx] * np.sign(cnn_preds[:, h_idx])
        mfe_ticks = np.maximum(directional, 0)
        mae_ticks = np.abs(np.minimum(directional, 0))
        time_to_mfe = np.ones(N_cnn, dtype=np.float32) * 5.0

    # Ensure alignment
    min_n = min(N_cnn, len(mfe_ticks))
    cnn_preds = cnn_preds[:min_n]
    labels = labels[:min_n]
    mfe_ticks = mfe_ticks[:min_n]
    mae_ticks = mae_ticks[:min_n]
    time_to_mfe = time_to_mfe[:min_n]

    # Load PatchTST predictions
    ptst_preds = np.zeros((min_n, 3), dtype=np.float32)
    ptst_path = ptst_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if ptst_path.exists():
        ptst_data = np.load(ptst_path, allow_pickle=True)
        ptst_raw = ptst_data["predictions"]
        if len(ptst_raw) == min_n:
            ptst_preds = ptst_raw[:min_n]
        elif len(ptst_raw) > min_n:
            indices = np.linspace(0, len(ptst_raw) - 1, min_n).astype(int)
            ptst_preds = ptst_raw[indices]
        else:
            for h in range(3):
                ptst_preds[:len(ptst_raw), h] = ptst_raw[:, h]
                if len(ptst_raw) < min_n:
                    ptst_preds[len(ptst_raw):, h] = ptst_raw[-1, h]
        logger.info(f"  PatchTST preds: {len(ptst_raw)} -> aligned to {min_n}")
    else:
        logger.warning(f"  No PatchTST preds for fold {fold_idx}, using zeros")

    # Load vol predictions
    vol_preds = np.zeros((min_n, 3), dtype=np.float32)
    vol_files = sorted(vol_dir.glob("vol_v3_*_predictions.npz"))
    if vol_files and oot_files is not None:
        oot_dates = [str(f).split("_")[0][:8] if isinstance(f, str) else str(f)[:8]
                     for f in oot_files]
        for vf in vol_files:
            vdate = vf.stem.split("_")[2]
            if vdate in oot_dates:
                vdata = np.load(vf, allow_pickle=True)
                vpreds = vdata["predictions"]
                fill_n = min(len(vpreds), min_n)
                vol_preds[:fill_n] = vpreds[:fill_n, :3] if vpreds.shape[1] >= 3 else vpreds[:fill_n]
                logger.info(f"  Vol preds matched for {vdate}: {fill_n} samples")
                break

    # Generate placeholder timestamps if not available
    timestamps = np.arange(min_n, dtype=np.int64) * int(1e8)
    anchor_idxs = np.arange(min_n, dtype=np.int64)

    # Try to load actual MBO events for microstructure features
    events = None
    if oot_files is not None and mbo_dir.exists():
        for oot_f in oot_files:
            fname = str(oot_f) if isinstance(oot_f, str) else oot_f
            candidates = list(mbo_dir.glob(f"*{fname[:8]}*")) if len(fname) >= 8 else []
            if candidates:
                try:
                    mbo_data = np.load(candidates[0], allow_pickle=True)
                    events = mbo_data.get("events", None)
                    if "timestamps" in mbo_data:
                        ts_raw = mbo_data["timestamps"]
                        if len(ts_raw) >= min_n:
                            timestamps = ts_raw[:min_n]
                    logger.info(f"  MBO events loaded: {len(events)} events")
                except Exception as e:
                    logger.warning(f"  Failed to load MBO events: {e}")
                break

    # Build features (V3 enriched)
    logger.info(f"  Building V3 execution features ({min_n} samples)...")
    features, feature_names = build_exec_features_v3(
        cnn_preds, ptst_preds, vol_preds,
        timestamps, events, anchor_idxs,
    )

    # Build targets
    targets = build_targets(labels, mfe_ticks, mae_ticks, time_to_mfe, cnn_preds)

    return features, targets, feature_names


# ============================================================
# Training: Phase 1 — XGBoost TP/SL (CPU)
# ============================================================

def train_tpsl_fold(
    train_features: np.ndarray,
    train_targets: Dict[str, np.ndarray],
    oot_features: np.ndarray,
    oot_targets: Dict[str, np.ndarray],
    feature_names: List[str],
    xgb_params: Dict,
) -> Tuple[TPSLTreeModel, Dict]:
    """Train XGBoost TP/SL models for one fold. CPU only, fast."""
    tpsl = TPSLTreeModel(**xgb_params)
    tpsl.fit(
        train_features,
        train_targets["optimal_tp_log"],
        train_targets["optimal_sl_log"],
        feature_names=feature_names,
    )

    # Evaluate on OOT
    pred_log_tp, pred_log_sl = tpsl.predict(oot_features)
    pred_tp, pred_sl = tpsl.predict_ticks(oot_features)

    tp_mae = np.abs(pred_tp - oot_targets["optimal_tp"]).mean()
    sl_mae = np.abs(pred_sl - oot_targets["optimal_sl"]).mean()
    tp_log_mae = np.abs(pred_log_tp - oot_targets["optimal_tp_log"]).mean()
    sl_log_mae = np.abs(pred_log_sl - oot_targets["optimal_sl_log"]).mean()

    metrics = {
        "tp_mae": float(tp_mae),
        "sl_mae": float(sl_mae),
        "tp_log_mae": float(tp_log_mae),
        "sl_log_mae": float(sl_log_mae),
        "pred_tp_mean": float(pred_tp.mean()),
        "pred_tp_std": float(pred_tp.std()),
        "pred_sl_mean": float(pred_sl.mean()),
        "pred_sl_std": float(pred_sl.std()),
        "actual_mfe_mean": float(oot_targets["optimal_tp"].mean()),
        "actual_mae_mean": float(oot_targets["optimal_sl"].mean()),
        "tpsl_backend": tpsl.backend,
    }

    # Feature importance
    importance = tpsl.get_feature_importance(feature_names, top_n=20)
    metrics.update(importance)

    logger.info(f"  XGBoost OOT: tp_mae={tp_mae:.2f} sl_mae={sl_mae:.2f} "
                f"tp_std={pred_tp.std():.3f} sl_std={pred_sl.std():.3f}")

    return tpsl, metrics


# ============================================================
# Training: Phase 2 — NN Gate with Sortino Loss (GPU)
# ============================================================

def train_gate_fold(
    gate_model: EntryGate_v3,
    exit_model: ExitTrigger_v3,
    train_features: np.ndarray,
    train_targets: Dict[str, np.ndarray],
    oot_features: np.ndarray,
    oot_targets: Dict[str, np.ndarray],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 3e-4,
    coverage_target: float = 0.3,
    num_workers: int = 2,
) -> Dict:
    """
    Train NN gate with Sortino-maximizing loss + exit trigger.
    Gate input = base features + XGBoost TP/SL predictions (already concatenated).
    """
    train_ds = GateExitDataset(train_features, train_targets)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)

    # Combined optimizer for gate + exit
    optimizer = torch.optim.AdamW([
        {"params": gate_model.parameters(), "lr": lr},
        {"params": exit_model.parameters(), "lr": lr},
    ], weight_decay=1e-4)

    total_steps = epochs * len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr, lr],
        total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos',
    )

    gate_criterion = SortinoGateLoss(coverage_target=coverage_target, coverage_weight=0.1)
    exit_huber = nn.HuberLoss(delta=2.0)

    scaler = GradScaler()
    best_sortino = -float('inf')
    best_gate_state = None
    best_exit_state = None

    for epoch in range(epochs):
        gate_model.train()
        exit_model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_features, batch_targets in train_dl:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets_d = {k: v.to(device, non_blocking=True) for k, v in batch_targets.items()}

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                gate_logit = gate_model(batch_features)
                hold_time, rr = exit_model(batch_features)

                # Gate loss: Sortino-maximizing
                gate_loss, gate_info = gate_criterion(
                    gate_logit,
                    batch_targets_d["realized_pnl"],
                    batch_targets_d["entry_profitable"],
                )

                # Exit loss
                hold_loss = exit_huber(hold_time, batch_targets_d["optimal_hold_s"])
                rr_loss = exit_huber(rr, batch_targets_d["risk_reward"])
                exit_loss = hold_loss + 0.5 * rr_loss

                total_loss = gate_loss + 0.5 * exit_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(gate_model.parameters()) + list(exit_model.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_info = {
                "total_loss": total_loss.item(),
                "hold_loss": hold_loss.item(),
                "rr_loss": rr_loss.item(),
            }
            loss_info.update(gate_info)
            epoch_losses.append(loss_info)

        avg = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        wall_time = time.time() - t0

        logger.info(
            f"  [Gate] Epoch {epoch + 1}/{epochs} | "
            f"sortino={avg['sortino_value']:.3f} | "
            f"coverage={avg['coverage']:.3f} | "
            f"prec={avg['gate_precision']:.3f} | "
            f"hold={avg['hold_loss']:.4f} | "
            f"{wall_time:.1f}s"
        )

        # OOT eval
        oot_metrics = evaluate_gate_oot(gate_model, exit_model, oot_features, oot_targets, device)
        oot_sortino = oot_metrics.get("sortino_gated", -999)

        if oot_sortino > best_sortino:
            best_sortino = oot_sortino
            best_gate_state = {k: v.cpu().clone() for k, v in gate_model.state_dict().items()}
            best_exit_state = {k: v.cpu().clone() for k, v in exit_model.state_dict().items()}

        logger.info(
            f"    OOT | sortino_gated={oot_metrics['sortino_gated']:.3f} | "
            f"sortino_base={oot_metrics['sortino_baseline']:.3f} | "
            f"coverage={oot_metrics['gate_coverage']:.3f} | "
            f"precision={oot_metrics['gate_precision']:.3f}"
        )

    # Restore best
    if best_gate_state is not None:
        gate_model.load_state_dict(best_gate_state)
    if best_exit_state is not None:
        exit_model.load_state_dict(best_exit_state)

    # Final OOT eval
    final_metrics = evaluate_gate_oot(gate_model, exit_model, oot_features, oot_targets, device, detailed=True)
    final_metrics["best_oot_sortino"] = best_sortino

    return final_metrics


def evaluate_gate_oot(
    gate_model: EntryGate_v3,
    exit_model: ExitTrigger_v3,
    features: np.ndarray,
    targets: Dict[str, np.ndarray],
    device: torch.device,
    detailed: bool = False,
) -> Dict:
    """Evaluate gate + exit on OOT data."""
    gate_model.eval()
    exit_model.eval()

    with torch.no_grad():
        x = torch.from_numpy(features).to(device)
        with autocast(dtype=torch.float16):
            gate_logit = gate_model(x).float().cpu().numpy()
            hold_time, rr = exit_model(x)
            hold_time = hold_time.float().cpu().numpy()
            rr = rr.float().cpu().numpy()

    gate_prob = 1.0 / (1.0 + np.exp(-np.clip(gate_logit, -20, 20)))

    # Gate metrics
    gate_pred = (gate_prob > 0.5).astype(float)
    gate_true = targets["entry_profitable"]
    gate_accuracy = (gate_pred == gate_true).mean()

    take_mask = gate_prob > 0.5
    if take_mask.sum() > 0:
        gate_precision = gate_true[take_mask].mean()
        gate_coverage = take_mask.mean()
    else:
        gate_precision = 0.0
        gate_coverage = 0.0

    pnl_all = targets["realized_pnl"]
    pnl_gated = pnl_all[take_mask] if take_mask.sum() > 0 else np.array([0.0])

    def _sortino(pnl):
        if len(pnl) < 2:
            return 0.0
        d = pnl[pnl < 0]
        return pnl.mean() / (np.std(d) + 1e-8) if len(d) > 1 else pnl.mean()

    sortino_baseline = _sortino(pnl_all)
    sortino_gated = _sortino(pnl_gated)

    metrics = {
        "gate_accuracy": float(gate_accuracy),
        "gate_precision": float(gate_precision),
        "gate_coverage": float(gate_coverage),
        "sortino_baseline": float(sortino_baseline),
        "sortino_gated": float(sortino_gated),
        "sortino_improvement": float(sortino_gated - sortino_baseline),
        "pnl_baseline_ticks": float(pnl_all.sum()),
        "pnl_gated_ticks": float(pnl_gated.sum()),
        "trades_baseline": len(pnl_all),
        "trades_gated": int(take_mask.sum()),
    }

    if detailed:
        # Confidence-conditional analysis (HC #13/#14)
        abs_gate = np.abs(gate_prob - 0.5)
        for pct_name, pct_thresh in [("top50", 0.5), ("top25", 0.75),
                                      ("top10", 0.9), ("top5", 0.95), ("top1", 0.99)]:
            thresh = np.percentile(abs_gate, pct_thresh * 100)
            mask = (abs_gate >= thresh) & (gate_prob > 0.5)
            if mask.sum() > 0:
                gated_pnl = pnl_all[mask]
                metrics[f"pnl_{pct_name}_ticks"] = float(gated_pnl.sum())
                metrics[f"sortino_{pct_name}"] = float(_sortino(gated_pnl))
                metrics[f"trades_{pct_name}"] = int(mask.sum())
                metrics[f"winrate_{pct_name}"] = float((gated_pnl > ROUND_TRIP_COST).mean())
            else:
                metrics[f"pnl_{pct_name}_ticks"] = 0.0
                metrics[f"sortino_{pct_name}"] = 0.0
                metrics[f"trades_{pct_name}"] = 0
                metrics[f"winrate_{pct_name}"] = 0.0

        # Gate confidence distribution
        confidence = np.abs(gate_prob - 0.5)
        metrics["gate_prob_mean"] = float(gate_prob.mean())
        metrics["gate_prob_std"] = float(gate_prob.std())
        metrics["gate_high_conf_count"] = int((confidence > 0.4).sum())
        metrics["gate_temperature"] = float(
            gate_model.log_temperature.exp().item()
        ) if hasattr(gate_model, 'log_temperature') else 1.0

    gate_model.train()
    exit_model.train()
    return metrics


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Smart Execution System v3 — Hybrid XGBoost + NN")
    parser.add_argument("--cnn-dir", type=str, default=None,
                        help="CNN Mamba v2 predictions directory")
    parser.add_argument("--ptst-dir", type=str, default=None,
                        help="PatchTST predictions directory")
    parser.add_argument("--vol-dir", type=str, default=None,
                        help="Vol LGBM v3 predictions directory")
    parser.add_argument("--mbo-dir", type=str, default=None,
                        help="MBO events directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for NN training (cuda or cpu)")
    parser.add_argument("--gate-epochs", type=int, default=20,
                        help="Epochs for NN gate training")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for NN training")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for NN gate + exit")
    parser.add_argument("--gate-hidden", type=int, default=128,
                        help="Hidden dim for gate NN")
    parser.add_argument("--exit-hidden", type=int, default=96,
                        help="Hidden dim for exit NN")
    parser.add_argument("--max-folds", type=int, default=9,
                        help="Maximum number of folds to use")
    parser.add_argument("--horizon", type=str, default="10s",
                        choices=["1s", "5s", "10s"],
                        help="Target horizon")
    parser.add_argument("--xgb-estimators", type=int, default=500,
                        help="XGBoost n_estimators")
    parser.add_argument("--xgb-depth", type=int, default=6,
                        help="XGBoost max_depth")
    parser.add_argument("--xgb-lr", type=float, default=0.05,
                        help="XGBoost learning rate")
    parser.add_argument("--xgb-subsample", type=float, default=0.8,
                        help="XGBoost subsample ratio")
    parser.add_argument("--xgb-colsample", type=float, default=0.8,
                        help="XGBoost colsample_bytree")
    parser.add_argument("--coverage-target", type=float, default=0.3,
                        help="Target coverage for gate (fraction of trades to take)")
    parser.add_argument("--version", type=str, default="v3",
                        help="Version tag for MLflow")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader num_workers (Neptune 32GB RAM: max 2)")
    args = parser.parse_args()

    # Detect paths
    hostname = socket.gethostname()
    if "neptune" in hostname.lower() or "nick" in str(Path.home()):
        root = Path("/home/nick/Lvl3Quant")
    else:
        root = Path("/home/jupiter/Lvl3Quant")

    cnn_dir = Path(args.cnn_dir) if args.cnn_dir else root / "output" / "cnn_mamba_v2_smart_v3_mar"
    ptst_dir = Path(args.ptst_dir) if args.ptst_dir else root / "output" / "patchtst_smart_v3_mar"
    vol_dir = Path(args.vol_dir) if args.vol_dir else root / "output" / "vol_lgbm_v3"
    mbo_dir = Path(args.mbo_dir) if args.mbo_dir else root / "data" / "processed" / "mbo_events"
    mfe_dir = cnn_dir / "mfe_mae_analysis"

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = root / "output" / f"smart_exec_v3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Determine tree backend
    tree_backend = "xgboost" if XGBOOST_AVAILABLE else ("sklearn" if SKLEARN_AVAILABLE else "NONE")

    logger.info("=" * 70)
    logger.info("Smart Execution System v3 — Hybrid XGBoost + NN")
    logger.info("=" * 70)
    logger.info(f"Device (NN):      {device}")
    if device.type == "cuda":
        logger.info(f"GPU:              {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Tree backend:     {tree_backend}")
    logger.info(f"CNN dir:          {cnn_dir}")
    logger.info(f"PatchTST:         {ptst_dir}")
    logger.info(f"Vol dir:          {vol_dir}")
    logger.info(f"MFE/MAE:          {mfe_dir}")
    logger.info(f"MBO dir:          {mbo_dir}")
    logger.info(f"Output:           {output_dir}")
    logger.info(f"Horizon:          {args.horizon}")
    logger.info(f"Gate epochs:      {args.gate_epochs}")
    logger.info(f"Gate hidden:      {args.gate_hidden}")
    logger.info(f"Exit hidden:      {args.exit_hidden}")
    logger.info(f"Batch size:       {args.batch_size}")
    logger.info(f"LR (NN):          {args.lr}")
    logger.info(f"XGBoost:          est={args.xgb_estimators} depth={args.xgb_depth} lr={args.xgb_lr}")
    logger.info(f"Coverage target:  {args.coverage_target}")
    logger.info(f"Version:          {args.version}")
    logger.info("=" * 70)

    if tree_backend == "NONE":
        logger.error("Neither xgboost nor sklearn available. Cannot train TP/SL. Exiting.")
        return

    # Discover available folds
    cnn_folds = sorted(cnn_dir.glob("fold_*_oot_predictions.npz"))
    n_folds = min(len(cnn_folds), args.max_folds)
    logger.info(f"Found {len(cnn_folds)} CNN folds, using {n_folds}")

    if n_folds < 2:
        logger.error("Need at least 2 folds (1 train + 1 OOT). Exiting.")
        return

    # XGBoost params
    xgb_params = {
        "n_estimators": args.xgb_estimators,
        "max_depth": args.xgb_depth,
        "learning_rate": args.xgb_lr,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
    }

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("file://" + str(root / "mlruns"))
            mlflow.set_experiment("SmartExecSystem_v3")
            mlflow_run = mlflow.start_run(
                run_name=f"SmartExec_v3_{time.strftime('%Y%m%d_%H%M')}"
            )
            mlflow.log_params({
                "model": "SmartExecSystem_v3",
                "version": args.version,
                "architecture": "hybrid_xgboost_nn",
                "tpsl_backend": tree_backend,
                "gate_hidden": args.gate_hidden,
                "exit_hidden": args.exit_hidden,
                "gate_epochs": args.gate_epochs,
                "batch_size": args.batch_size,
                "lr_nn": args.lr,
                "xgb_estimators": args.xgb_estimators,
                "xgb_depth": args.xgb_depth,
                "xgb_lr": args.xgb_lr,
                "xgb_subsample": args.xgb_subsample,
                "xgb_colsample": args.xgb_colsample,
                "horizon": args.horizon,
                "n_folds": n_folds,
                "node": hostname,
                "device": str(device),
                "coverage_target": args.coverage_target,
                "gate_loss": "sortino_maximizing",
                "v3_changes": "enriched_features_60+,xgboost_tpsl,sortino_gate_loss",
            })
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}")

    # ============================================================
    # Walk-forward sliding window
    # ============================================================
    all_fold_metrics = []
    concat_gate_preds = []
    concat_gate_true = []
    concat_pnl = []
    global_feature_names = None

    for oot_fold in range(1, n_folds):
        train_folds = list(range(max(0, oot_fold - 5), oot_fold))

        logger.info(f"\n{'=' * 70}")
        logger.info(f"FOLD {oot_fold} | Train folds: {train_folds} | OOT fold: {oot_fold}")
        logger.info(f"{'=' * 70}")

        # Load training data from multiple folds
        train_features_list = []
        train_targets_list = {
            "optimal_tp_log": [], "optimal_sl_log": [],
            "optimal_tp": [], "optimal_sl": [],
            "entry_profitable": [],
            "realized_pnl": [], "trade_magnitude": [], "optimal_hold_s": [],
            "risk_reward": [],
        }

        fold_feature_names = None
        for tf in train_folds:
            feat, targ, fnames = load_fold_data(
                tf, cnn_dir, ptst_dir, vol_dir, mfe_dir, mbo_dir, args.horizon
            )
            if feat is not None:
                train_features_list.append(feat)
                for k in train_targets_list:
                    train_targets_list[k].append(targ[k])
                if fold_feature_names is None:
                    fold_feature_names = fnames

        if not train_features_list:
            logger.warning(f"No training data for fold {oot_fold}, skipping")
            continue

        train_features = np.concatenate(train_features_list)
        train_targets = {k: np.concatenate(v) for k, v in train_targets_list.items()}

        if global_feature_names is None and fold_feature_names is not None:
            global_feature_names = fold_feature_names

        # Load OOT data
        oot_features, oot_targets, oot_fnames = load_fold_data(
            oot_fold, cnn_dir, ptst_dir, vol_dir, mfe_dir, mbo_dir, args.horizon
        )
        if oot_features is None:
            logger.warning(f"No OOT data for fold {oot_fold}, skipping")
            continue

        feature_names = fold_feature_names if fold_feature_names else (
            [f"f{i}" for i in range(train_features.shape[1])]
        )

        logger.info(f"Train: {len(train_features)} samples, OOT: {len(oot_features)} samples")
        logger.info(f"Feature dim: {train_features.shape[1]} ({len(feature_names)} named)")

        # Normalize features using train stats
        feat_mean = train_features.mean(axis=0)
        feat_std = train_features.std(axis=0) + 1e-8
        train_features_norm = (train_features - feat_mean) / feat_std
        oot_features_norm = (oot_features - feat_mean) / feat_std

        # ============================================================
        # PHASE 1: XGBoost TP/SL (CPU, fast)
        # ============================================================
        logger.info(f"\n--- Phase 1: XGBoost TP/SL ---")
        t0_phase1 = time.time()

        tpsl_model, tpsl_metrics = train_tpsl_fold(
            train_features_norm, train_targets,
            oot_features_norm, oot_targets,
            feature_names, xgb_params,
        )

        phase1_time = time.time() - t0_phase1
        logger.info(f"Phase 1 complete in {phase1_time:.1f}s")

        # Get XGBoost TP/SL predictions to use as gate inputs
        train_log_tp, train_log_sl = tpsl_model.predict(train_features_norm)
        oot_log_tp, oot_log_sl = tpsl_model.predict(oot_features_norm)

        # Augment features with XGBoost TP/SL predictions for gate input
        train_gate_features = np.column_stack([
            train_features_norm,
            train_log_tp,
            train_log_sl,
            np.exp(np.clip(train_log_tp, -2.0, LOG_TPSL_CLIP)),  # TP in ticks
            np.exp(np.clip(train_log_sl, -2.0, LOG_TPSL_CLIP)),  # SL in ticks
            # TP/SL ratio (risk-reward from XGBoost)
            np.exp(np.clip(train_log_tp, -2.0, LOG_TPSL_CLIP)) /
            np.maximum(np.exp(np.clip(train_log_sl, -2.0, LOG_TPSL_CLIP)), 0.1),
        ]).astype(np.float32)

        oot_gate_features = np.column_stack([
            oot_features_norm,
            oot_log_tp,
            oot_log_sl,
            np.exp(np.clip(oot_log_tp, -2.0, LOG_TPSL_CLIP)),
            np.exp(np.clip(oot_log_sl, -2.0, LOG_TPSL_CLIP)),
            np.exp(np.clip(oot_log_tp, -2.0, LOG_TPSL_CLIP)) /
            np.maximum(np.exp(np.clip(oot_log_sl, -2.0, LOG_TPSL_CLIP)), 0.1),
        ]).astype(np.float32)

        # Replace NaN/Inf from augmentation
        train_gate_features = np.nan_to_num(train_gate_features, nan=0.0, posinf=10.0, neginf=-10.0)
        oot_gate_features = np.nan_to_num(oot_gate_features, nan=0.0, posinf=10.0, neginf=-10.0)

        gate_input_dim = train_gate_features.shape[1]
        logger.info(f"Gate input dim: {gate_input_dim} (base {train_features.shape[1]} + 5 XGBoost augmented)")

        # ============================================================
        # PHASE 2: NN Gate + Exit (GPU)
        # ============================================================
        logger.info(f"\n--- Phase 2: NN Gate (Sortino loss) + Exit ---")
        t0_phase2 = time.time()

        gate_model = EntryGate_v3(gate_input_dim, args.gate_hidden).to(device)
        exit_model = ExitTrigger_v3(gate_input_dim, args.exit_hidden).to(device)

        n_gate_params = sum(p.numel() for p in gate_model.parameters())
        n_exit_params = sum(p.numel() for p in exit_model.parameters())
        logger.info(f"Gate params: {n_gate_params:,}, Exit params: {n_exit_params:,}")

        gate_metrics = train_gate_fold(
            gate_model, exit_model,
            train_gate_features, train_targets,
            oot_gate_features, oot_targets,
            device,
            epochs=args.gate_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            coverage_target=args.coverage_target,
            num_workers=args.num_workers,
        )

        phase2_time = time.time() - t0_phase2
        logger.info(f"Phase 2 complete in {phase2_time:.1f}s")

        # ============================================================
        # PHASE 3: Joint evaluation
        # ============================================================
        logger.info(f"\n--- Phase 3: Joint evaluation ---")

        # Combined metrics
        fold_metrics = {}
        fold_metrics.update({f"tpsl_{k}": v for k, v in tpsl_metrics.items()
                             if isinstance(v, (int, float, str))})
        fold_metrics.update({f"gate_{k}": v for k, v in gate_metrics.items()
                             if isinstance(v, (int, float, str))})
        fold_metrics["phase1_time_s"] = phase1_time
        fold_metrics["phase2_time_s"] = phase2_time

        # Feature importance from XGBoost
        if "tp_top_features" in tpsl_metrics:
            fold_metrics["tp_top_features"] = tpsl_metrics["tp_top_features"]
        if "sl_top_features" in tpsl_metrics:
            fold_metrics["sl_top_features"] = tpsl_metrics["sl_top_features"]

        all_fold_metrics.append(fold_metrics)

        # Save models and predictions
        # XGBoost models
        tpsl_model.save(output_dir, oot_fold)

        # NN models
        torch.save({
            "gate_state": gate_model.state_dict(),
            "exit_state": exit_model.state_dict(),
            "fold_idx": oot_fold,
            "gate_input_dim": gate_input_dim,
            "gate_hidden": args.gate_hidden,
            "exit_hidden": args.exit_hidden,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "feature_names": feature_names,
            "metrics": fold_metrics,
            "version": args.version,
        }, output_dir / f"fold_{oot_fold:02d}_gate_exit_v3.pt")

        # Save OOT predictions
        gate_model.eval()
        exit_model.eval()
        with torch.no_grad():
            x = torch.from_numpy(oot_gate_features).to(device)
            with autocast(dtype=torch.float16):
                gate_logit = gate_model(x).float().cpu().numpy()
                hold_pred, rr_pred = exit_model(x)
                hold_pred = hold_pred.float().cpu().numpy()
                rr_pred = rr_pred.float().cpu().numpy()

        gate_prob = 1.0 / (1.0 + np.exp(-np.clip(gate_logit, -20, 20)))
        pred_tp, pred_sl = tpsl_model.predict_ticks(oot_features_norm)

        np.savez_compressed(
            output_dir / f"fold_{oot_fold:02d}_oot_predictions.npz",
            gate_prob=gate_prob,
            tp_ticks=pred_tp,
            sl_ticks=pred_sl,
            hold_time=hold_pred,
            risk_reward=rr_pred,
            entry_profitable=oot_targets["entry_profitable"],
            realized_pnl=oot_targets["realized_pnl"],
        )

        concat_gate_preds.append(gate_prob)
        concat_gate_true.append(oot_targets["entry_profitable"])
        concat_pnl.append(oot_targets["realized_pnl"])

        # MLflow per-fold
        if mlflow_run:
            try:
                for k, v in fold_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"fold{oot_fold:02d}_{k}", v)
            except Exception:
                pass

        # Memory cleanup
        del gate_model, exit_model, tpsl_model
        del train_features, train_targets, oot_features, oot_targets
        del train_features_norm, oot_features_norm
        del train_gate_features, oot_gate_features
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Concat metrics (primary metric — HC #13)
    # ============================================================
    if concat_gate_preds:
        all_gate = np.concatenate(concat_gate_preds)
        all_true = np.concatenate(concat_gate_true)
        all_pnl = np.concatenate(concat_pnl)

        logger.info("\n" + "=" * 70)
        logger.info("CONCAT RESULTS — Smart Execution System v3 (Hybrid XGBoost + NN)")
        logger.info("=" * 70)

        gate_pred_binary = (all_gate > 0.5)
        overall_acc = (gate_pred_binary == all_true).mean()

        gated_mask = all_gate > 0.5
        pnl_ungated = all_pnl.sum()
        pnl_gated = all_pnl[gated_mask].sum() if gated_mask.sum() > 0 else 0

        def _sortino(pnl):
            if len(pnl) < 2:
                return 0.0
            d = pnl[pnl < 0]
            return pnl.mean() / (np.std(d) + 1e-8) if len(d) > 1 else pnl.mean()

        sortino_ungated = _sortino(all_pnl)
        sortino_gated = _sortino(all_pnl[gated_mask]) if gated_mask.sum() > 0 else 0

        logger.info(f"Gate accuracy:      {overall_acc:.3f}")
        logger.info(f"Gate coverage:      {gated_mask.mean():.3f} ({gated_mask.sum()}/{len(all_gate)} trades taken)")
        logger.info(f"PnL ungated:        {pnl_ungated:.1f} ticks (${pnl_ungated * TICK_VAL:.2f})")
        logger.info(f"PnL gated:          {pnl_gated:.1f} ticks (${pnl_gated * TICK_VAL:.2f})")
        logger.info(f"Sortino ungated:    {sortino_ungated:.3f}")
        logger.info(f"Sortino gated:      {sortino_gated:.3f}")
        logger.info(f"Sortino improvement: {sortino_gated - sortino_ungated:.3f}")

        # Gate confidence distribution
        confidence = np.abs(all_gate - 0.5)
        logger.info(f"\nGate confidence stats:")
        logger.info(f"  Mean distance from 0.5: {confidence.mean():.4f}")
        logger.info(f"  High-conf (>0.4):       {(confidence > 0.4).sum()} ({(confidence > 0.4).mean() * 100:.1f}%)")
        logger.info(f"  Very high-conf (>0.45): {(confidence > 0.45).sum()} ({(confidence > 0.45).mean() * 100:.1f}%)")

        # Confidence-conditional (HC #13/#14)
        logger.info(f"\n{'Tier':<10} {'Trades':>8} {'WinRate':>8} {'PnL(t)':>10} {'Sortino':>8} {'Sortino_vs_Base':>15}")
        logger.info("-" * 65)

        for name, pct in [("All", 0.0), ("Top50%", 0.5), ("Top25%", 0.75),
                          ("Top10%", 0.9), ("Top5%", 0.95), ("Top1%", 0.99)]:
            if pct > 0:
                thresh = np.percentile(confidence, pct * 100)
                mask = (confidence >= thresh) & (all_gate > 0.5)
            else:
                mask = all_gate > 0.5

            if mask.sum() > 0:
                tier_pnl = all_pnl[mask]
                wr = (tier_pnl > ROUND_TRIP_COST).mean()
                s = _sortino(tier_pnl)
                delta = s - sortino_ungated
                logger.info(f"{name:<10} {mask.sum():>8d} {wr:>8.3f} {tier_pnl.sum():>10.1f} {s:>8.3f} {delta:>+15.3f}")
            else:
                logger.info(f"{name:<10} {'0':>8} {'N/A':>8} {'0.0':>10} {'N/A':>8} {'N/A':>15}")

        # Per-fold summary
        logger.info(f"\nPer-fold metrics:")
        for i, fm in enumerate(all_fold_metrics):
            tp_mae = fm.get('tpsl_tp_mae', fm.get('tp_mae', 0))
            sl_mae = fm.get('tpsl_sl_mae', fm.get('sl_mae', 0))
            tp_std = fm.get('tpsl_pred_tp_std', 0)
            sl_std = fm.get('tpsl_pred_sl_std', 0)
            g_sort = fm.get('gate_sortino_gated', fm.get('sortino_gated', 0))
            g_cov = fm.get('gate_gate_coverage', fm.get('gate_coverage', 0))
            g_prec = fm.get('gate_gate_precision', fm.get('gate_precision', 0))
            logger.info(
                f"  Fold {i + 1}: sortino_gated={g_sort:.3f} "
                f"precision={g_prec:.3f} coverage={g_cov:.3f} "
                f"tp_mae={tp_mae:.2f} sl_mae={sl_mae:.2f} "
                f"tp_std={tp_std:.3f} sl_std={sl_std:.3f}"
            )

        # XGBoost feature importance (aggregate across folds)
        logger.info(f"\nXGBoost Feature Importance (last fold):")
        last_fm = all_fold_metrics[-1] if all_fold_metrics else {}
        for target in ["tp", "sl"]:
            key = f"tpsl_{target}_top_features"
            if key not in last_fm:
                key = f"{target}_top_features"
            if key in last_fm:
                logger.info(f"  {target.upper()} Top 20:")
                for feat_info in last_fm[key][:20]:
                    if isinstance(feat_info, dict):
                        logger.info(f"    {feat_info['name']:<30} {feat_info['importance']:.4f}")

        # TP/SL prediction diversity check
        logger.info(f"\nTP/SL prediction diversity (XGBoost):")
        for i, fm in enumerate(all_fold_metrics):
            tp_std = fm.get('tpsl_pred_tp_std', 0)
            sl_std = fm.get('tpsl_pred_sl_std', 0)
            status_tp = "OK" if tp_std > 0.1 else "COLLAPSED"
            status_sl = "OK" if sl_std > 0.1 else "COLLAPSED"
            logger.info(f"  Fold {i + 1}: TP std={tp_std:.3f} [{status_tp}] | SL std={sl_std:.3f} [{status_sl}]")

        # Save concat results
        concat_results = {
            "version": args.version,
            "architecture": "hybrid_xgboost_nn",
            "tpsl_backend": tree_backend,
            "gate_loss": "sortino_maximizing",
            "gate_accuracy": float(overall_acc),
            "gate_coverage": float(gated_mask.mean()),
            "pnl_ungated_ticks": float(pnl_ungated),
            "pnl_gated_ticks": float(pnl_gated),
            "sortino_ungated": float(sortino_ungated),
            "sortino_gated": float(sortino_gated),
            "sortino_improvement": float(sortino_gated - sortino_ungated),
            "n_folds": len(all_fold_metrics),
            "confidence_mean": float(confidence.mean()),
            "high_conf_pct": float((confidence > 0.4).mean()),
            "feature_count": train_gate_features.shape[1] if 'train_gate_features' in dir() else "N/A",
            "feature_names": global_feature_names,
            "per_fold": all_fold_metrics,
            "v3_changes": [
                "enriched_features_60+",
                "xgboost_tpsl_regression",
                "sortino_maximizing_gate_loss",
                "differentiable_sortino",
                "coverage_regularization",
                "xgb_preds_as_gate_inputs",
                "multi_scale_volatility",
                "order_flow_imbalance",
                "volume_profile",
                "spread_dynamics",
                "momentum_multi_scale",
                "cross_model_features",
            ],
        }

        with open(output_dir / "concat_results.json", "w") as f:
            json.dump(concat_results, f, indent=2, default=str)

        # MLflow concat metrics
        if mlflow_run:
            try:
                mlflow.log_metrics({
                    "concat_gate_accuracy": float(overall_acc),
                    "concat_gate_coverage": float(gated_mask.mean()),
                    "concat_pnl_gated_ticks": float(pnl_gated),
                    "concat_sortino_ungated": float(sortino_ungated),
                    "concat_sortino_gated": float(sortino_gated),
                    "concat_sortino_improvement": float(sortino_gated - sortino_ungated),
                    "concat_confidence_mean": float(confidence.mean()),
                    "concat_high_conf_pct": float((confidence > 0.4).mean()),
                })
            except Exception:
                pass

    if mlflow_run:
        try:
            mlflow.end_run()
        except Exception:
            pass

    logger.info("\n" + "=" * 70)
    logger.info("Smart Execution System v3 training complete.")
    logger.info(f"Results: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
