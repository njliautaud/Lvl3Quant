#!/usr/bin/env python3
"""
exec_mlp_per_date_inference.py — Per-date Exec MLP inference on CNN-Mamba decay_v4 predictions
================================================================================================

Loads a trained ExecMLP checkpoint, builds the same feature set it was trained on,
and runs inference on each of the 39 OOS dates from decay_v4_comprehensive/CNN-Mamba_v2/.

Output: per-date .npz files in output/exec_mlp_v1_9day/ matching the format expected
by fifo_market_replay.py:
    fold_XX_oot_predictions.npz with keys:
        - gate_predictions (N,)       — P(profitable) from ExecMLP gate head
        - confidence_predictions (N,) — MFE / (MFE + MAE) ratio
        - predictions (N, 3)          — passthrough CNN-Mamba 1s/5s/10s predictions
        - labels (N, 3)               — passthrough labels
        - date (str)                  — YYYYMMDD
        - horizons (3,)               — ['1s', '5s', '10s']

Runs on CPU (no GPU needed for inference).

Usage:
    python execution/exec_mlp_per_date_inference.py
    python execution/exec_mlp_per_date_inference.py --model-path output/exec_mlp_gpu_optuna_best/exec_mlp_v1_best.pt
    python execution/exec_mlp_per_date_inference.py --output-dir output/exec_mlp_v1_decay39
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# ── Project root ────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LVL3_ROOT))

try:
    from constants import COMMISSION_TICKS, TICK_VALUE
except ImportError:
    COMMISSION_TICKS = 0.376
    TICK_VALUE = 12.50

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────
TICK = 0.25

# ── Default paths ───────────────────────────────────────────────────────────────
CNN_PRED_DIR = LVL3_ROOT / "output" / "decay_v4_comprehensive" / "CNN-Mamba_v2"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events"
VOL_PRED_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"

# Model checkpoint search paths (first found wins)
MODEL_SEARCH_PATHS = [
    LVL3_ROOT / "output" / "exec_mlp_gpu_v1" / "exec_mlp_v1_best.pt",
    LVL3_ROOT / "output" / "exec_mlp_gpu_optuna_best" / "exec_mlp_v1_best.pt",
]

OUTPUT_DIR = LVL3_ROOT / "output" / "exec_mlp_v1_9day"


# =============================================================================
# Model Definition — must match train_exec_mlp_gpu.py exactly
# =============================================================================

class ExecMLP(nn.Module):
    """
    Multi-task MLP for execution gating (mirrors train_exec_mlp_gpu.py).

    Architecture:
      Input -> BatchNorm -> [Linear -> ReLU -> Dropout] x N -> Multi-head output

    Heads: gate (sigmoid), mfe (softplus), mae (softplus), hold_time (softplus)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        prev_dim = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.gate_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid(),
        )
        self.mfe_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus(),
        )
        self.mae_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus(),
        )
        self.hold_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus(),
        )

    def forward(self, x):
        x = self.input_bn(x)
        shared = self.trunk(x)
        return {
            'gate': self.gate_head(shared).squeeze(-1),
            'mfe': self.mfe_head(shared).squeeze(-1),
            'mae': self.mae_head(shared).squeeze(-1),
            'hold_time': self.hold_head(shared).squeeze(-1),
        }


# =============================================================================
# Rolling statistics helpers (identical to training script)
# =============================================================================

def rolling_zscore(arr: np.ndarray, window: int = 3000) -> np.ndarray:
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
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        out[i] = (cs[i] - (cs[start - 1] if start > 0 else 0)) / count
    return out


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
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


# =============================================================================
# Feature Engineering (identical logic to training script)
# =============================================================================

def extract_mbo_features(
    mbo_data: dict,
    n_samples: int,
    sample_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Extract 24 microstructure features from MBO event data."""

    timestamps = mbo_data.get('timestamps', np.array([]))
    events = mbo_data.get('events', None)
    if events is not None and len(events) > 0:
        prices = events[:, 3]
        sizes = np.exp(events[:, 4])
        sides = events[:, 2]
        event_types = events[:, 1]
    else:
        prices = mbo_data.get('prices', np.array([]))
        sizes = mbo_data.get('sizes', np.array([]))
        sides = mbo_data.get('sides', np.array([]))
        event_types = mbo_data.get('event_types', np.array([]))

    n_events = len(timestamps)
    if n_events < 100:
        n_feat = 24
        return np.zeros((n_samples, n_feat), dtype=np.float32), [f"micro_{i}" for i in range(n_feat)]

    if sample_indices is None:
        sample_indices = np.linspace(100, n_events - 1, n_samples).astype(int)

    price_changes = np.diff(prices, prepend=prices[0])
    time_deltas = np.diff(timestamps, prepend=timestamps[0]).astype(np.float64)
    time_deltas_sec = time_deltas / 1e9
    time_deltas_sec = np.clip(time_deltas_sec, 1e-6, 3600.0)

    side_sign = np.where(sides == 1, 1.0, np.where(sides == 0, -1.0, 0.0))
    signed_flow = sizes * side_sign

    cs_sizes = np.cumsum(sizes)
    cs_flow = np.cumsum(signed_flow)
    cs_pc2 = np.cumsum(price_changes ** 2)
    cs_abs_pc = np.cumsum(np.abs(price_changes))

    is_trade = (sizes > 0).astype(np.float32)
    cs_trades = np.cumsum(is_trade)

    size_threshold = np.percentile(sizes[sizes > 0], 90) if np.any(sizes > 0) else 1.0

    def _cs_window(cs, idx, window):
        start = max(0, idx - window + 1)
        val = cs[idx] - (cs[start - 1] if start > 0 else 0)
        count = idx - start + 1
        return val, count

    # Initialize all 24 feature arrays
    spread_proxy = np.zeros(n_samples, dtype=np.float32)
    spread_zscore_arr = np.zeros(n_samples, dtype=np.float32)
    book_imbalance = np.zeros(n_samples, dtype=np.float32)
    bid_ask_size_ratio = np.zeros(n_samples, dtype=np.float32)
    flow_imbalance_50 = np.zeros(n_samples, dtype=np.float32)
    flow_imbalance_200 = np.zeros(n_samples, dtype=np.float32)
    flow_imbalance_500 = np.zeros(n_samples, dtype=np.float32)
    event_rate_100 = np.zeros(n_samples, dtype=np.float32)
    event_rate_500 = np.zeros(n_samples, dtype=np.float32)
    large_trade_freq_100 = np.zeros(n_samples, dtype=np.float32)
    large_trade_vol_ratio = np.zeros(n_samples, dtype=np.float32)
    volume_100 = np.zeros(n_samples, dtype=np.float32)
    volume_500 = np.zeros(n_samples, dtype=np.float32)
    volume_accel = np.zeros(n_samples, dtype=np.float32)
    volume_ratio = np.zeros(n_samples, dtype=np.float32)
    momentum_50 = np.zeros(n_samples, dtype=np.float32)
    momentum_200 = np.zeros(n_samples, dtype=np.float32)
    momentum_500 = np.zeros(n_samples, dtype=np.float32)
    rvol_50 = np.zeros(n_samples, dtype=np.float32)
    rvol_200 = np.zeros(n_samples, dtype=np.float32)
    rvol_1000 = np.zeros(n_samples, dtype=np.float32)
    vol_of_vol = np.zeros(n_samples, dtype=np.float32)
    net_flow_ratio = np.zeros(n_samples, dtype=np.float32)
    price_impact_100 = np.zeros(n_samples, dtype=np.float32)

    for i, eidx in enumerate(sample_indices):
        eidx = int(min(eidx, n_events - 1))

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

        if eidx >= 100:
            recent_sides = sides[eidx - 100:eidx + 1]
            recent_sizes_w = sizes[eidx - 100:eidx + 1]
            bid_vol = recent_sizes_w[recent_sides < 0.5].sum()
            ask_vol = recent_sizes_w[recent_sides > 0.5].sum()
            total = bid_vol + ask_vol
            if total > 0:
                book_imbalance[i] = (bid_vol - ask_vol) / total
                bid_ask_size_ratio[i] = bid_vol / max(ask_vol, 1e-6)

        for win, arr in [(50, flow_imbalance_50), (200, flow_imbalance_200),
                         (500, flow_imbalance_500)]:
            if eidx >= win:
                flow_sum, _ = _cs_window(cs_flow, eidx, win)
                vol_sum, _ = _cs_window(cs_sizes, eidx, win)
                if vol_sum > 0:
                    arr[i] = flow_sum / vol_sum

        for win, arr in [(100, event_rate_100), (500, event_rate_500)]:
            if eidx >= win:
                dt_total = time_deltas_sec[eidx - win + 1:eidx + 1].sum()
                if dt_total > 0:
                    arr[i] = win / dt_total

        if eidx >= 100:
            recent_sizes_w = sizes[eidx - 100:eidx + 1]
            large_mask = recent_sizes_w > size_threshold
            large_trade_freq_100[i] = large_mask.mean()
            total_vol = recent_sizes_w.sum()
            if total_vol > 0:
                large_trade_vol_ratio[i] = recent_sizes_w[large_mask].sum() / total_vol

        for win, arr in [(100, volume_100), (500, volume_500)]:
            if eidx >= 1:
                vol_sum, _ = _cs_window(cs_sizes, eidx, win)
                arr[i] = vol_sum

        if eidx >= 50:
            vol_recent, _ = _cs_window(cs_sizes, eidx, 25)
            vol_prev, _ = _cs_window(cs_sizes, max(0, eidx - 25), 25)
            volume_accel[i] = vol_recent - vol_prev

        if eidx >= 2000:
            long_vol, long_cnt = _cs_window(cs_sizes, eidx, 2000)
            short_vol, short_cnt = _cs_window(cs_sizes, eidx, 100)
            avg_per_100 = (long_vol / max(long_cnt, 1)) * 100
            if avg_per_100 > 0:
                volume_ratio[i] = short_vol / avg_per_100

        for win, arr in [(50, momentum_50), (200, momentum_200), (500, momentum_500)]:
            if eidx >= win:
                arr[i] = (prices[eidx] - prices[eidx - win]) / TICK

        for win, arr in [(50, rvol_50), (200, rvol_200), (1000, rvol_1000)]:
            if eidx >= win:
                pc2_sum, cnt = _cs_window(cs_pc2, eidx, win)
                arr[i] = np.sqrt(pc2_sum / cnt) / TICK

        if eidx >= 500:
            rvol_samples = []
            for k in range(10):
                kidx = eidx - k * 50
                if kidx >= 50:
                    pc2_s, cnt_s = _cs_window(cs_pc2, kidx, 50)
                    rvol_samples.append(np.sqrt(pc2_s / cnt_s) / TICK)
            if len(rvol_samples) >= 3:
                vol_of_vol[i] = np.std(rvol_samples)

        if eidx >= 200:
            flow_sum, _ = _cs_window(cs_flow, eidx, 200)
            abs_sum, cnt = _cs_window(cs_sizes, eidx, 200)
            if abs_sum > 0:
                net_flow_ratio[i] = flow_sum / abs_sum

        if eidx >= 100:
            abs_move, _ = _cs_window(cs_abs_pc, eidx, 100)
            vol_sum, _ = _cs_window(cs_sizes, eidx, 100)
            if vol_sum > 0:
                price_impact_100[i] = (abs_move / TICK) / vol_sum

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
    predictions: np.ndarray,   # (N, 3) — 1s/5s/10s
    embeddings: np.ndarray,    # (N, 96) or (N, D) — CNN-Mamba embeddings
) -> Tuple[np.ndarray, List[str]]:
    """Build 113 signal-derived features (predictions + embeddings + derived)."""
    N = predictions.shape[0]
    parts = []
    names = []

    # 1. Raw predictions (3)
    parts.append(predictions)
    names.extend(["pred_1s", "pred_5s", "pred_10s"])

    # 2. CNN-Mamba embeddings (96)
    parts.append(embeddings)
    names.extend([f"emb_{i}" for i in range(embeddings.shape[1])])

    # 3. Absolute predictions (3)
    abs_preds = np.abs(predictions)
    parts.append(abs_preds)
    names.extend(["abs_pred_1s", "abs_pred_5s", "abs_pred_10s"])

    # 4. Sign agreement (1)
    signs = np.sign(predictions)
    sign_agreement = np.mean(signs == signs[:, 2:3], axis=1, keepdims=True)
    parts.append(sign_agreement)
    names.append("sign_agreement")

    # 5. Prediction z-scores (3)
    for h, label in enumerate(["1s", "5s", "10s"]):
        z = rolling_zscore(predictions[:, h], window=3000)
        parts.append(z.reshape(-1, 1))
        names.append(f"pred_zscore_{label}")

    # 6. Conviction trajectory (3)
    for h, label in enumerate(["1s", "5s", "10s"]):
        short_mean = rolling_mean(np.abs(predictions[:, h]), window=50)
        long_mean = rolling_mean(np.abs(predictions[:, h]), window=500)
        trajectory = short_mean - long_mean
        parts.append(trajectory.reshape(-1, 1))
        names.append(f"conviction_trajectory_{label}")

    # 7. Signal persistence (1)
    autocorr = rolling_autocorr(predictions[:, 2], window=200, lag=10)
    parts.append(autocorr.reshape(-1, 1))
    names.append("signal_persistence_10s")

    # 8. Horizon disagreement (1)
    pred_std_across = np.std(predictions, axis=1, keepdims=True)
    parts.append(pred_std_across)
    names.append("horizon_disagreement")

    # 9. Prediction range (1)
    pred_range = (np.max(predictions, axis=1) - np.min(predictions, axis=1)).reshape(-1, 1)
    parts.append(pred_range)
    names.append("pred_range")

    # 10. Prediction skew (1)
    pred_skew = (predictions[:, 2] - predictions[:, 0]).reshape(-1, 1)
    parts.append(pred_skew)
    names.append("pred_skew_10s_vs_1s")

    feature_matrix = np.concatenate(parts, axis=1).astype(np.float32)
    return feature_matrix, names


def build_temporal_features(
    timestamps_ns: Optional[np.ndarray],
    n_samples: int,
) -> Tuple[np.ndarray, List[str]]:
    """Build 10 temporal features."""
    parts = []
    names = []

    if timestamps_ns is not None and len(timestamps_ns) == n_samples:
        utc_seconds = timestamps_ns / 1e9
        et_seconds = utc_seconds + (-5) * 3600
        hours = (et_seconds % 86400) / 3600.0
    else:
        hours = np.linspace(9.5, 16.0, n_samples)

    tod_sin = np.sin(2 * np.pi * hours / 24.0).astype(np.float32)
    tod_cos = np.cos(2 * np.pi * hours / 24.0).astype(np.float32)
    parts.append(tod_sin.reshape(-1, 1))
    parts.append(tod_cos.reshape(-1, 1))
    names.extend(["tod_sin", "tod_cos"])

    rth_frac = np.clip((hours - 9.5) / 6.5, 0, 1)
    rth_sin = np.sin(2 * np.pi * rth_frac).astype(np.float32)
    rth_cos = np.cos(2 * np.pi * rth_frac).astype(np.float32)
    parts.append(rth_sin.reshape(-1, 1))
    parts.append(rth_cos.reshape(-1, 1))
    names.extend(["rth_sin", "rth_cos"])

    is_open = ((hours >= 9.5) & (hours < 10.5)).astype(np.float32)
    is_core = ((hours >= 10.5) & (hours < 15.0)).astype(np.float32)
    is_close = ((hours >= 15.0) & (hours < 16.0)).astype(np.float32)
    is_pre = (hours < 9.5).astype(np.float32)
    parts.append(is_open.reshape(-1, 1))
    parts.append(is_core.reshape(-1, 1))
    parts.append(is_close.reshape(-1, 1))
    parts.append(is_pre.reshape(-1, 1))
    names.extend(["session_open", "session_core", "session_close", "session_pre"])

    mins_since_open = np.clip((hours - 9.5) * 60, -60, 420).astype(np.float32)
    parts.append(mins_since_open.reshape(-1, 1))
    names.append("mins_since_open")

    mins_until_close = np.clip((16.0 - hours) * 60, -60, 420).astype(np.float32)
    parts.append(mins_until_close.reshape(-1, 1))
    names.append("mins_until_close")

    feature_matrix = np.concatenate(parts, axis=1).astype(np.float32)
    return feature_matrix, names


def build_vol_features(
    vol_preds: Optional[np.ndarray],
    predictions: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Build volatility context features (4-6 depending on vol_preds shape)."""
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
        parts.append(np.zeros((N, 1), dtype=np.float32))
        names.append("vol_pred_placeholder")

    pred_vol_short = rolling_std(predictions[:, 2], window=100)
    pred_vol_long = rolling_std(predictions[:, 2], window=1000)
    parts.append(pred_vol_short.reshape(-1, 1))
    parts.append(pred_vol_long.reshape(-1, 1))
    names.extend(["pred_vol_100", "pred_vol_1000"])

    vol_regime = pred_vol_short / np.maximum(pred_vol_long, 1e-8)
    parts.append(vol_regime.reshape(-1, 1))
    names.append("vol_regime")

    feature_matrix = np.concatenate(parts, axis=1).astype(np.float32)
    return feature_matrix, names


def build_all_features(
    predictions: np.ndarray,       # (N, 3)
    embeddings: np.ndarray,        # (N, 96) or zeros
    mbo_data: Optional[dict],
    timestamps: Optional[np.ndarray],
    vol_preds: Optional[np.ndarray],
    n_samples: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the complete feature matrix from all sources.
    Returns (N, total_features) array and feature names.
    """
    all_parts = []
    all_names = []

    # 1. Signal features
    sig_feats, sig_names = build_signal_features(predictions, embeddings)
    all_parts.append(sig_feats)
    all_names.extend(sig_names)

    # 2. MBO microstructure features
    if mbo_data is not None:
        mbo_feats, mbo_names = extract_mbo_features(mbo_data, n_samples)
        all_parts.append(mbo_feats)
        all_names.extend(mbo_names)
    else:
        n_mbo = 24
        all_parts.append(np.zeros((n_samples, n_mbo), dtype=np.float32))
        all_names.extend([f"micro_{i}" for i in range(n_mbo)])

    # 3. Temporal features
    temp_feats, temp_names = build_temporal_features(timestamps, n_samples)
    all_parts.append(temp_feats)
    all_names.extend(temp_names)

    # 4. Volatility features
    vol_feats, vol_names = build_vol_features(vol_preds, predictions)
    all_parts.append(vol_feats)
    all_names.extend(vol_names)

    feature_matrix = np.concatenate(all_parts, axis=1).astype(np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=10.0, neginf=-10.0)

    return feature_matrix, all_names


# =============================================================================
# Data loading for decay_v4 predictions
# =============================================================================

def load_date_data(date_str: str) -> Optional[dict]:
    """
    Load CNN-Mamba predictions from decay_v4_comprehensive for one date,
    plus MBO events and vol predictions if available.
    """
    pred_file = CNN_PRED_DIR / date_str / "predictions.npz"
    if not pred_file.exists():
        log.warning(f"  {date_str}: predictions.npz not found at {pred_file}")
        return None

    pred_data = np.load(str(pred_file), allow_pickle=True)

    # decay_v4 format: preds (N,3), labels_1s, labels_5s, labels_10s, valid_indices
    predictions = pred_data['preds']              # (N, 3)
    labels_1s = pred_data['labels_1s']             # (N,)
    labels_5s = pred_data['labels_5s']             # (N,)
    labels_10s = pred_data['labels_10s']           # (N,)
    valid_indices = pred_data['valid_indices']      # (N,)

    labels = np.stack([labels_1s, labels_5s, labels_10s], axis=1)  # (N, 3)
    N = len(predictions)

    # Embeddings: decay_v4 doesn't include them — use zeros
    # The model was trained with 96-dim embeddings; without them the gate
    # must rely on the other features. Zeros will go through BatchNorm
    # and produce the mean-field response for those dims.
    embeddings = np.zeros((N, 96), dtype=np.float32)

    # Load MBO events for timestamps and microstructure
    mbo_data = None
    mbo_file = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if mbo_file.exists():
        try:
            mbo_raw = np.load(str(mbo_file), allow_pickle=True)
            mbo_data = {k: mbo_raw[k] for k in mbo_raw.files}
            log.info(f"  {date_str}: MBO events loaded ({len(mbo_data.get('timestamps', []))} events)")
        except Exception as e:
            log.warning(f"  {date_str}: Failed to load MBO: {e}")

    # Extract timestamps at sample positions (for temporal features)
    timestamps = None
    if mbo_data is not None and 'timestamps' in mbo_data:
        all_ts = mbo_data['timestamps']
        n_events = len(all_ts)
        if n_events > 0:
            # Map valid_indices into MBO timestamps
            # valid_indices are positions in the prediction array, map them
            # proportionally into MBO event space
            sample_positions = np.linspace(0, n_events - 1, N).astype(int)
            timestamps = all_ts[sample_positions]

    # Load vol predictions (if available)
    vol_pred = None
    vol_file = VOL_PRED_DIR / f"vol_v3_{date_str}_predictions.npz"
    if vol_file.exists():
        try:
            vd = np.load(str(vol_file), allow_pickle=True)
            for k in ['predictions', 'vol_pred', 'y_pred']:
                if k in vd:
                    vp = vd[k]
                    if len(vp) == N:
                        vol_pred = vp
                    else:
                        from scipy.interpolate import interp1d
                        x_vol = np.linspace(0, 1, len(vp))
                        x_cnn = np.linspace(0, 1, N)
                        if vp.ndim == 1:
                            f_interp = interp1d(x_vol, vp, kind='nearest', fill_value='extrapolate')
                            vol_pred = f_interp(x_cnn).astype(np.float32)
                        else:
                            vol_pred = np.zeros((N, vp.shape[1]), dtype=np.float32)
                            for j in range(vp.shape[1]):
                                f_interp = interp1d(x_vol, vp[:, j], kind='nearest', fill_value='extrapolate')
                                vol_pred[:, j] = f_interp(x_cnn)
                    break
        except Exception as e:
            log.warning(f"  {date_str}: Failed to load vol: {e}")

    return {
        'date': date_str,
        'predictions': predictions,
        'labels': labels,
        'embeddings': embeddings,
        'valid_indices': valid_indices,
        'mbo_data': mbo_data,
        'timestamps': timestamps,
        'vol_pred': vol_pred,
        'n_samples': N,
    }


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_path: str) -> Tuple[ExecMLP, dict]:
    """
    Load ExecMLP from checkpoint.
    Returns (model, checkpoint_dict) with norm_mean/norm_std.
    """
    log.info(f"Loading model from {model_path}")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)

    input_dim = ckpt['input_dim']
    hidden_dim = ckpt.get('hidden_dim', ckpt.get('hidden', 256))
    n_layers = ckpt.get('n_layers', 4)
    dropout = ckpt.get('dropout', 0.3)

    model = ExecMLP(input_dim, hidden_dim, n_layers, dropout)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    log.info(f"  Model loaded: input_dim={input_dim}, hidden={hidden_dim}, "
             f"layers={n_layers}, dropout={dropout}")
    log.info(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    return model, ckpt


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model: ExecMLP,
    features: np.ndarray,   # (N, D)
    norm_mean: np.ndarray,  # (D,)
    norm_std: np.ndarray,   # (D,)
    batch_size: int = 4096,
) -> dict:
    """
    Run ExecMLP inference on normalized features.
    Returns dict with gate, mfe, mae, hold_time arrays.
    """
    # Normalize
    features_norm = (features - norm_mean) / norm_std
    features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=10.0, neginf=-10.0)

    N = len(features_norm)
    gate_all = []
    mfe_all = []
    mae_all = []
    hold_all = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x = torch.from_numpy(features_norm[start:end]).float()
            out = model(x)
            gate_all.append(out['gate'].numpy())
            mfe_all.append(out['mfe'].numpy())
            mae_all.append(out['mae'].numpy())
            hold_all.append(out['hold_time'].numpy())

    return {
        'gate': np.concatenate(gate_all),
        'mfe': np.concatenate(mfe_all),
        'mae': np.concatenate(mae_all),
        'hold_time': np.concatenate(hold_all),
    }


def pad_or_trim_features(features: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Ensure feature matrix has exactly target_dim columns.
    Pads with zeros or trims rightmost columns as needed.
    """
    actual_dim = features.shape[1]
    if actual_dim == target_dim:
        return features
    elif actual_dim < target_dim:
        padding = np.zeros((features.shape[0], target_dim - actual_dim), dtype=np.float32)
        log.warning(f"  Feature dim {actual_dim} < model expects {target_dim}, "
                    f"zero-padding {target_dim - actual_dim} columns")
        return np.concatenate([features, padding], axis=1)
    else:
        log.warning(f"  Feature dim {actual_dim} > model expects {target_dim}, "
                    f"trimming {actual_dim - target_dim} columns")
        return features[:, :target_dim]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exec MLP per-date inference on CNN-Mamba decay_v4 predictions"
    )
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='Path to exec_mlp_v1_best.pt checkpoint. Auto-detected if not specified.'
    )
    parser.add_argument(
        '--pred-dir', type=str,
        default=str(CNN_PRED_DIR),
        help='Directory with per-date CNN-Mamba predictions (YYYYMMDD subdirs)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=str(OUTPUT_DIR),
        help='Output directory for fold_XX_oot_predictions.npz files'
    )
    parser.add_argument(
        '--embedding-dim', type=int, default=96,
        help='Embedding dimension (zeros if not available in predictions)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4096,
        help='Inference batch size'
    )
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("Exec MLP Per-Date Inference")
    log.info("=" * 80)

    # ── Find model checkpoint ───────────────────────────────────────────────
    model_path = args.model_path
    if model_path is None:
        for p in MODEL_SEARCH_PATHS:
            if p.exists():
                model_path = str(p)
                break
        if model_path is None:
            log.error("No model checkpoint found. Searched:")
            for p in MODEL_SEARCH_PATHS:
                log.error(f"  {p}")
            log.error("Specify --model-path explicitly.")
            sys.exit(1)

    model, ckpt = load_model(model_path)
    input_dim = ckpt['input_dim']
    norm_mean = ckpt['norm_mean']
    norm_std = ckpt['norm_std']

    # Ensure norm arrays are 1D
    norm_mean = np.asarray(norm_mean, dtype=np.float32).ravel()
    norm_std = np.asarray(norm_std, dtype=np.float32).ravel()
    # Replace zero stds with 1.0 to avoid division by zero
    norm_std = np.where(norm_std < 1e-8, 1.0, norm_std)

    log.info(f"  Model input_dim: {input_dim}")
    log.info(f"  Norm mean/std shape: {norm_mean.shape}, {norm_std.shape}")
    if 'best_threshold' in ckpt:
        log.info(f"  Best gate threshold: {ckpt['best_threshold']:.3f}")

    # ── Discover OOS dates ──────────────────────────────────────────────────
    pred_dir = Path(args.pred_dir)
    date_dirs = sorted([
        d.name for d in pred_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    ])
    log.info(f"\nFound {len(date_dirs)} OOS dates in {pred_dir}")

    # ── Output directory ────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    # ── Process each date ───────────────────────────────────────────────────
    results_summary = []

    for fold_idx, date_str in enumerate(date_dirs):
        log.info(f"\n{'─'*60}")
        log.info(f"Fold {fold_idx:02d} / Date {date_str}")
        log.info(f"{'─'*60}")

        # Load data
        data = load_date_data(date_str)
        if data is None:
            log.warning(f"  Skipping {date_str}: no data")
            continue

        N = data['n_samples']
        log.info(f"  Samples: {N}")

        # Build features
        features, feat_names = build_all_features(
            predictions=data['predictions'],
            embeddings=data['embeddings'],
            mbo_data=data['mbo_data'],
            timestamps=data['timestamps'],
            vol_preds=data['vol_pred'],
            n_samples=N,
        )
        log.info(f"  Built features: {features.shape[1]} dims")

        # Adjust feature dimensions to match model
        features = pad_or_trim_features(features, input_dim)

        # Also adjust norm_mean/norm_std if needed
        if len(norm_mean) != input_dim:
            log.warning(f"  norm_mean length {len(norm_mean)} != input_dim {input_dim}")

        nm = norm_mean[:input_dim] if len(norm_mean) >= input_dim else np.pad(
            norm_mean, (0, input_dim - len(norm_mean)), constant_values=0.0
        )
        ns = norm_std[:input_dim] if len(norm_std) >= input_dim else np.pad(
            norm_std, (0, input_dim - len(norm_std)), constant_values=1.0
        )

        # Run inference
        outputs = run_inference(model, features, nm, ns, batch_size=args.batch_size)

        gate_probs = outputs['gate']
        mfe_preds = outputs['mfe']
        mae_preds = outputs['mae']

        # Confidence = MFE / (MFE + MAE), clipped to [0, 1]
        confidence = mfe_preds / np.maximum(mfe_preds + mae_preds, 1e-8)
        confidence = np.clip(confidence, 0.0, 1.0)

        # Stats
        gate_rate = (gate_probs > 0.5).mean()
        log.info(f"  Gate > 0.5: {gate_rate:.1%} ({int(gate_rate * N)} / {N})")
        log.info(f"  Gate mean: {gate_probs.mean():.4f}, median: {np.median(gate_probs):.4f}")
        log.info(f"  MFE mean: {mfe_preds.mean():.3f}, MAE mean: {mae_preds.mean():.3f}")
        log.info(f"  Confidence mean: {confidence.mean():.4f}")

        # Save in FIFO replay format
        out_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
        np.savez(
            str(out_path),
            predictions=data['predictions'],           # (N, 3) — CNN-Mamba passthrough
            gate_predictions=gate_probs,                # (N,) — ExecMLP gate P(profitable)
            confidence_predictions=confidence,          # (N,) — MFE/(MFE+MAE)
            labels=data['labels'],                      # (N, 3) — 1s/5s/10s labels
            gate_labels=np.zeros(N, dtype=np.float32),  # placeholder (no ground truth here)
            date=date_str,
            horizons=np.array(['1s', '5s', '10s']),
            # Also include raw MLP outputs for downstream analysis
            mfe_predictions=mfe_preds,
            mae_predictions=mae_preds,
            hold_time_predictions=outputs['hold_time'],
            valid_indices=data['valid_indices'],
        )
        log.info(f"  Saved: {out_path}")

        results_summary.append({
            'fold': fold_idx,
            'date': date_str,
            'n_samples': N,
            'gate_rate_50': float(gate_rate),
            'gate_mean': float(gate_probs.mean()),
            'mfe_mean': float(mfe_preds.mean()),
            'mae_mean': float(mae_preds.mean()),
            'confidence_mean': float(confidence.mean()),
        })

    # ── Save summary ────────────────────────────────────────────────────────
    import json
    summary_path = output_dir / "inference_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'model_path': model_path,
            'input_dim': input_dim,
            'n_dates': len(results_summary),
            'pred_dir': str(pred_dir),
            'output_dir': str(output_dir),
            'embedding_dim': args.embedding_dim,
            'note': 'Embeddings zero-filled (not available in decay_v4 predictions)',
            'per_date': results_summary,
        }, f, indent=2)
    log.info(f"\nSummary saved: {summary_path}")

    # ── Final report ────────────────────────────────────────────────────────
    log.info(f"\n{'='*80}")
    log.info("INFERENCE COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"  Dates processed: {len(results_summary)}")
    log.info(f"  Output dir: {output_dir}")
    log.info(f"  Files: fold_00..fold_{len(results_summary)-1:02d}_oot_predictions.npz")

    if results_summary:
        avg_gate = np.mean([r['gate_rate_50'] for r in results_summary])
        avg_conf = np.mean([r['confidence_mean'] for r in results_summary])
        log.info(f"  Avg gate rate (>0.5): {avg_gate:.1%}")
        log.info(f"  Avg confidence: {avg_conf:.4f}")

    log.info(f"\nNOTE: Embeddings were zero-filled because decay_v4 predictions")
    log.info(f"don't include CNN-Mamba hidden-state embeddings. The gate will rely")
    log.info(f"on predictions, MBO microstructure, temporal, and vol features only.")
    log.info(f"For best results, re-run CNN-Mamba inference saving embeddings.")


if __name__ == "__main__":
    main()
