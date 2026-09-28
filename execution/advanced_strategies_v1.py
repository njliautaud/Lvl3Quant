#!/usr/bin/env python3
"""
Advanced Execution Strategies v1
=================================
Develops and tests MULTIPLE creative execution strategies combining
CNN-Mamba v2 and Mamba v7 signals through real MBO market replay.

Strategies:
  1. Signal-Flip Exit — hold until model flips direction (conditional flip option)
  2. Multi-Horizon Agreement Gate — ALL 3 horizons must agree
  3. Embedding-Based MLP Gate — train tiny MLP on embeddings to filter trades
  4. Volatility-Adaptive Threshold — dynamic thresholds based on rolling vol
  5. Time-of-Day Optimization — per-session confidence thresholds
  6. Momentum Burst Detection — enter only on signal acceleration

All P&L via Rust fill_sim_cli on raw MBO data. No mid-price P&L.

Usage:
    python advanced_strategies_v1.py
    python advanced_strategies_v1.py --model cnn-mamba-v2 --workers 8
    python advanced_strategies_v1.py --model mamba-v7 --workers 8
    python advanced_strategies_v1.py --model both --workers 12
"""

import sys
import gc
import json
import time
import argparse
import subprocess
import logging
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR_V3 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
EVENT_DIR_V2 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v2'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'advanced_v1'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'advanced_v1'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── CNN-Mamba v2 Prediction Directory ──
CNN_MAMBA_V2_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'

# ── Mamba v7 Prediction Directory ──
MAMBA_V7_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'

# ── ES Futures Constants ──────────────────────────────────────────────────
TICK_SIZE_PTS = 0.25
TICK_VALUE = 12.50
POINT_VALUE = 50.00
COMMISSION_RT = 4.70
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.376 ticks

# ── Bar/Timing Constants ──
BARS_PER_SEC = 10
BAR_NS = 100_000_000      # 100ms in nanoseconds
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# ── Model Constants ──
WINDOW = 1000
STRIDE = 500

# ── Timestamp ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──────────────────────────────────────────────────────────────
_log_file = str(RESULTS_DIR / f'advanced_strategies_{_ts}.log')
log = logging.getLogger('advanced_strategies')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# RTH Timestamp Utilities
# ============================================================

def rth_start_ns_for_date(date_str: str) -> int:
    """Compute RTH start timestamp (9:30 AM ET) for date YYYYMMDD."""
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    d = datetime(year, month, day)
    dst_start_2025 = datetime(2025, 3, 9)
    dst_end_2025 = datetime(2025, 11, 2)
    dst_start_2026 = datetime(2026, 3, 8)
    dst_end_2026 = datetime(2026, 11, 1)
    if (dst_start_2025 <= d < dst_end_2025) or (dst_start_2026 <= d < dst_end_2026):
        utc_offset = -4
    else:
        utc_offset = -5
    rth_start_utc_hours = 9.5 - utc_offset
    midnight_utc = datetime(year, month, day, tzinfo=timezone.utc)
    rth_start = midnight_utc + timedelta(hours=rth_start_utc_hours)
    return int(rth_start.timestamp() * 1e9)


# ============================================================
# Prediction Loading
# ============================================================

def load_fold(fold_path: Path) -> Optional[Dict]:
    """Load a single fold prediction file."""
    try:
        data = np.load(str(fold_path), allow_pickle=True)
        preds = data['predictions']
        labels = data['labels']

        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]

        result = {
            'predictions': preds.astype(np.float64),
            'labels': labels.astype(np.float64),
            'date_str': date_str,
            'n_samples': preds.shape[0],
            'fold_path': str(fold_path),
        }

        if 'embeddings' in data:
            result['embeddings'] = data['embeddings'].astype(np.float32)

        return result
    except Exception as e:
        log.warning(f"Failed to load {fold_path}: {e}")
        return None


def discover_folds(pred_dir: Path) -> Dict[str, Dict]:
    """Discover all per-fold prediction files. Returns {date_str: fold_data}."""
    folds = {}
    for f in sorted(pred_dir.glob('fold_*_oot_predictions.npz')):
        if 'concat' in f.name:
            continue
        data = load_fold(f)
        if data:
            folds[data['date_str']] = data
            log.info(f"  Fold: {data['date_str']} -> {f.name} "
                     f"({data['n_samples']} samples"
                     f"{', has embeddings' if 'embeddings' in data else ''})")
    return folds


def load_event_timestamps(date_str: str) -> Optional[np.ndarray]:
    """Load event timestamps for a date."""
    for edir in [EVENT_DIR_V3, EVENT_DIR_V2]:
        candidate = edir / f'{date_str}_mbo_events.npz'
        if candidate.exists():
            try:
                ev_data = np.load(str(candidate), allow_pickle=True)
                return ev_data['timestamps']
            except Exception as e:
                log.warning(f"  Failed to load events {candidate}: {e}")
    return None


# ============================================================
# Core Signal Conversion
# ============================================================

def predictions_to_bar_signal(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Convert per-window predictions to bar-indexed z-scored signal.

    Maps predictions (stride=500, window=1000) to 100ms bar indices,
    then applies expanding walk-forward z-score normalization.
    """
    n_events = len(event_timestamps)
    n_preds = len(predictions)

    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1

    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    elif n_preds > len(label_idxs):
        predictions = predictions[:len(label_idxs)]
        n_preds = len(predictions)

    if n_preds == 0:
        return np.zeros(N_RTH_BARS, dtype=np.float64), running_stats or {}

    pred_timestamps = event_timestamps[label_idxs]
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)
    bar_indices_rth = bar_indices[rth_mask]
    predictions_rth = predictions[rth_mask]

    # Bar-level signal (last prediction wins for overlapping windows)
    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, sig in zip(bar_indices_rth, predictions_rth):
        bar_preds[bi] = sig

    # Expanding z-score (walk-forward, no lookahead)
    if running_stats is None:
        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

    zscore_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    rs = running_stats['sum']
    rsq = running_stats['sq']
    cnt = running_stats['count']

    for i in range(N_RTH_BARS):
        v = bar_preds[i]
        if v == 0.0:
            continue
        rs += v
        rsq += v * v
        cnt += 1
        if cnt >= 50:
            mean = rs / cnt
            var = (rsq / cnt) - mean * mean
            std = max(np.sqrt(max(var, 0)), 1e-8)
            zscore_preds[i] = (v - mean) / std

    running_stats = {'sum': rs, 'sq': rsq, 'count': cnt}
    return zscore_preds, running_stats


def predictions_to_bar_raw(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
) -> np.ndarray:
    """Convert predictions to bar-indexed RAW (un-z-scored) signal.
    Used for strategies that need raw prediction values.
    """
    n_events = len(event_timestamps)
    n_preds = len(predictions)

    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1

    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    elif n_preds > len(label_idxs):
        predictions = predictions[:len(label_idxs)]
        n_preds = len(predictions)

    if n_preds == 0:
        return np.zeros(N_RTH_BARS, dtype=np.float64)

    pred_timestamps = event_timestamps[label_idxs]
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)

    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, sig in zip(bar_indices[rth_mask], predictions[rth_mask]):
        bar_preds[bi] = sig

    return bar_preds


# ============================================================
# Strategy 1: Signal-Flip Exit (PRIORITY)
# ============================================================

def generate_signal_flip_configs() -> List[Dict]:
    """Generate fill_sim_cli configurations for signal-flip exit strategy.

    Enter on high-confidence signal, hold until model flips direction.
    - Instant flip: signal_flip_exit with various thresholds
    - Conditional flip: conviction_exit_bars + conviction_exit_mag
    """
    configs = []

    # 1A: Instant signal-flip exit with different z-score entry thresholds
    for z in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        configs.append({
            'label': f'S1_flip_z{int(z*10):02d}_chase',
            'description': f'Signal-flip exit, chase entry, z>{z}',
            'strategy_group': 'signal_flip',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--hold-ms', '300000',  # 5 min max hold
                '--signal-flip-exit',
                '--quiet',
            ],
        })

    # 1B: Conditional flip — only exit when opposing signal is also strong
    for z in [2.5, 3.0, 3.5, 4.0]:
        for conv_bars, conv_mag in [(30, 1.0), (50, 1.5), (100, 2.0), (50, 2.5)]:
            configs.append({
                'label': f'S1_convflip_z{int(z*10):02d}_b{conv_bars}_m{int(conv_mag*10):02d}',
                'description': f'Conviction flip, z>{z}, {conv_bars}bars@{conv_mag}mag',
                'strategy_group': 'signal_flip',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', '300000',
                    '--conviction-exit-bars', str(conv_bars),
                    '--conviction-exit-mag', str(conv_mag),
                    '--quiet',
                ],
            })

    # 1C: Signal-flip + stop-loss protection (limit downside)
    for z in [2.5, 3.0, 3.5]:
        for sl in [2, 3, 4]:
            configs.append({
                'label': f'S1_flip_sl{sl}_z{int(z*10):02d}',
                'description': f'Signal-flip + SL={sl}t, z>{z}',
                'strategy_group': 'signal_flip',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', '300000',
                    '--signal-flip-exit',
                    '--stop-loss-ticks', str(sl),
                    '--quiet',
                ],
            })

    # 1D: Signal-flip + trailing stop (lock in profits)
    for z in [2.5, 3.0, 3.5]:
        for trail in [2, 3, 4]:
            configs.append({
                'label': f'S1_flip_trail{trail}_z{int(z*10):02d}',
                'description': f'Signal-flip + trail={trail}t, z>{z}',
                'strategy_group': 'signal_flip',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', '300000',
                    '--signal-flip-exit',
                    '--trailing-ticks', str(trail),
                    '--quiet',
                ],
            })

    # 1E: Signal-flip with ratchet stop
    for z in [2.5, 3.0, 3.5]:
        configs.append({
            'label': f'S1_flip_ratchet_z{int(z*10):02d}',
            'description': f'Signal-flip + ratchet stop, z>{z}',
            'strategy_group': 'signal_flip',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--hold-ms', '300000',
                '--signal-flip-exit',
                '--ratchet-stop',
                '--quiet',
            ],
        })

    return configs


# ============================================================
# Strategy 2: Multi-Horizon Agreement Gate
# ============================================================

def generate_multi_horizon_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate signal that requires ALL 3 horizons (1s, 5s, 10s) to agree.

    Signal = sign(consensus) * mean(|z_1s|, |z_5s|, |z_10s|)
    Only non-zero when all 3 horizons have same sign.
    """
    preds = fold_data['predictions']  # (N, 3)
    p_1s = preds[:, 0]
    p_5s = preds[:, 1]
    p_10s = preds[:, 2]

    # Agreement mask: all 3 horizons same sign
    agree_mask = (
        (np.sign(p_1s) == np.sign(p_5s)) &
        (np.sign(p_5s) == np.sign(p_10s)) &
        (p_1s != 0) & (p_5s != 0) & (p_10s != 0)
    )

    # Composite signal: direction * average magnitude
    composite = np.where(
        agree_mask,
        np.sign(p_10s) * (np.abs(p_1s) + np.abs(p_5s) + np.abs(p_10s)) / 3.0,
        0.0,
    )

    n_agree = int(np.sum(agree_mask))
    n_total = len(preds)
    agreement_rate = n_agree / max(n_total, 1)

    bar_signal, stats = predictions_to_bar_signal(
        composite, event_timestamps, date_str, running_stats
    )

    meta = {
        'signal_type': 'multi_horizon_agreement',
        'n_agree': n_agree,
        'n_total': n_total,
        'agreement_rate': round(agreement_rate, 4),
    }

    return bar_signal, stats


def generate_multi_horizon_configs() -> List[Dict]:
    """Generate configs for multi-horizon agreement strategy."""
    configs = []

    # 2A: Basic multi-horizon agreement with various thresholds
    for z in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        for hold_ms in [10000, 30000, 60000]:
            hold_label = {10000: '10s', 30000: '30s', 60000: '60s'}[hold_ms]
            configs.append({
                'label': f'S2_agree_z{int(z*10):02d}_hold{hold_label}',
                'description': f'3-horizon agreement, z>{z}, hold {hold_label}',
                'strategy_group': 'multi_horizon',
                'signal_type': 'multi_horizon',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', str(hold_ms),
                    '--quiet',
                ],
            })

    # 2B: Multi-horizon + signal-flip exit
    for z in [2.0, 2.5, 3.0, 3.5]:
        configs.append({
            'label': f'S2_agree_flip_z{int(z*10):02d}',
            'description': f'3-horizon agreement + flip exit, z>{z}',
            'strategy_group': 'multi_horizon',
            'signal_type': 'multi_horizon',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--hold-ms', '300000',
                '--signal-flip-exit',
                '--quiet',
            ],
        })

    # 2C: Multi-horizon + conviction exit
    for z in [2.5, 3.0]:
        for conv_bars, conv_mag in [(50, 1.5), (100, 2.0)]:
            configs.append({
                'label': f'S2_agree_conv_z{int(z*10):02d}_b{conv_bars}',
                'description': f'3-horizon + conviction exit, z>{z}',
                'strategy_group': 'multi_horizon',
                'signal_type': 'multi_horizon',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', '300000',
                    '--conviction-exit-bars', str(conv_bars),
                    '--conviction-exit-mag', str(conv_mag),
                    '--quiet',
                ],
            })

    return configs


# ============================================================
# Strategy 3: Embedding-Based MLP Gate
# ============================================================

def train_embedding_gate(
    embeddings: np.ndarray,
    labels_10s: np.ndarray,
    predictions_10s: np.ndarray,
) -> Optional[Any]:
    """Train a tiny MLP gate on embeddings to predict trade profitability.

    Binary classification: will the trade be profitable?
    Target: sign(label_10s) == sign(prediction_10s) => profitable trade.

    Uses numpy-only logistic regression (no torch dependency needed).
    Architecture: 96 -> 32 -> 1 with ReLU and sigmoid.
    """
    # Binary target: 1 if prediction direction matches label direction
    pred_signs = np.sign(predictions_10s)
    label_signs = np.sign(labels_10s)
    # Profitable = prediction was correct AND magnitude was large enough
    # to overcome commission (0.376 ticks)
    profitable = ((pred_signs == label_signs) &
                  (np.abs(labels_10s) > 0.001)).astype(np.float64)

    n = len(embeddings)
    if n < 200:
        return None

    # Normalize embeddings
    emb = embeddings.astype(np.float64)
    emb_mean = emb.mean(axis=0)
    emb_std = emb.std(axis=0) + 1e-8
    emb_norm = (emb - emb_mean) / emb_std

    # Simple 2-layer MLP via numpy (96 -> 32 -> 1)
    np.random.seed(42)
    W1 = np.random.randn(96, 32) * 0.1
    b1 = np.zeros(32)
    W2 = np.random.randn(32, 1) * 0.1
    b2 = np.zeros(1)

    lr = 0.01
    batch_size = min(256, n)

    for epoch in range(50):
        # Mini-batch SGD
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            X = emb_norm[idx]
            y = profitable[idx].reshape(-1, 1)

            # Forward
            h = X @ W1 + b1
            h_relu = np.maximum(h, 0)
            logits = h_relu @ W2 + b2
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

            # Binary cross-entropy gradient
            dlogits = probs - y
            dW2 = h_relu.T @ dlogits / len(idx)
            db2 = dlogits.mean(axis=0)

            dh_relu = dlogits @ W2.T
            dh = dh_relu * (h > 0).astype(np.float64)
            dW1 = X.T @ dh / len(idx)
            db1 = dh.mean(axis=0)

            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

    return {
        'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
        'emb_mean': emb_mean, 'emb_std': emb_std,
    }


def apply_embedding_gate(
    gate_model: Dict,
    embeddings: np.ndarray,
    threshold: float = 0.6,
) -> np.ndarray:
    """Apply trained MLP gate to embeddings. Returns mask of approved trades."""
    emb = embeddings.astype(np.float64)
    emb_norm = (emb - gate_model['emb_mean']) / (gate_model['emb_std'] + 1e-8)

    h = emb_norm @ gate_model['W1'] + gate_model['b1']
    h_relu = np.maximum(h, 0)
    logits = h_relu @ gate_model['W2'] + gate_model['b2']
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    return (probs.flatten() >= threshold).astype(bool)


def generate_embedding_gated_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    gate_model: Optional[Dict],
    gate_threshold: float = 0.6,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate signal gated by embedding MLP.

    Uses 10s horizon signal, zeroed out where MLP says trade won't be profitable.
    """
    preds = fold_data['predictions']
    p_10s = preds[:, 2]  # 10s horizon

    if gate_model is not None and 'embeddings' in fold_data:
        gate_mask = apply_embedding_gate(
            gate_model, fold_data['embeddings'], gate_threshold
        )
        gated_signal = np.where(gate_mask, p_10s, 0.0)
        n_gated = int(np.sum(~gate_mask))
        n_passed = int(np.sum(gate_mask))
    else:
        gated_signal = p_10s
        n_gated = 0
        n_passed = len(p_10s)

    bar_signal, stats = predictions_to_bar_signal(
        gated_signal, event_timestamps, date_str, running_stats
    )

    meta = {
        'signal_type': 'embedding_gated',
        'n_gated_out': n_gated,
        'n_passed': n_passed,
        'gate_rate': round(n_passed / max(n_passed + n_gated, 1), 4),
    }

    return bar_signal, stats


def generate_embedding_gate_configs() -> List[Dict]:
    """Generate configs for embedding-gated strategy."""
    configs = []

    for z in [2.0, 2.5, 3.0, 3.5]:
        for hold_ms in [10000, 30000, 60000]:
            hold_label = {10000: '10s', 30000: '30s', 60000: '60s'}[hold_ms]
            configs.append({
                'label': f'S3_embgate_z{int(z*10):02d}_hold{hold_label}',
                'description': f'Embedding gate, z>{z}, hold {hold_label}',
                'strategy_group': 'embedding_gate',
                'signal_type': 'embedding_gated',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', str(hold_ms),
                    '--quiet',
                ],
            })

    # With signal-flip exit
    for z in [2.0, 2.5, 3.0]:
        configs.append({
            'label': f'S3_embgate_flip_z{int(z*10):02d}',
            'description': f'Embedding gate + flip exit, z>{z}',
            'strategy_group': 'embedding_gate',
            'signal_type': 'embedding_gated',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--hold-ms', '300000',
                '--signal-flip-exit',
                '--quiet',
            ],
        })

    return configs


# ============================================================
# Strategy 4: Volatility-Adaptive Threshold
# ============================================================

def generate_vol_adaptive_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    vol_window: int = 500,
    low_vol_z: float = 2.0,
    high_vol_z: float = 3.5,
    vol_percentile: float = 75.0,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate volatility-adaptive signal.

    During high volatility: require stronger signal (higher z threshold)
    During low volatility: trade more aggressively (lower z threshold)

    Vol estimated from rolling std of raw predictions (expanding).
    """
    preds = fold_data['predictions']
    p_10s = preds[:, 2]

    # First get raw bar-level signal
    bar_raw = predictions_to_bar_raw(p_10s, event_timestamps, date_str)

    # Compute rolling volatility from non-zero predictions
    nonzero_mask = bar_raw != 0
    nonzero_vals = bar_raw[nonzero_mask]

    if len(nonzero_vals) < 100:
        # Not enough data, fall back to standard z-score
        bar_signal, stats = predictions_to_bar_signal(
            p_10s, event_timestamps, date_str, running_stats
        )
        return bar_signal, stats

    # Expanding volatility estimate
    vol_threshold = np.percentile(np.abs(nonzero_vals), vol_percentile)

    # Adaptive signal: scale predictions by inverse vol
    # High vol bars -> reduce magnitude, low vol bars -> keep/amplify
    adaptive_signal = np.zeros_like(p_10s)
    running_vol_sum = 0.0
    running_vol_sq = 0.0
    running_vol_cnt = 0

    for i in range(len(p_10s)):
        v = p_10s[i]
        if v == 0:
            continue

        running_vol_sum += abs(v)
        running_vol_sq += v * v
        running_vol_cnt += 1

        if running_vol_cnt >= 50:
            local_vol = np.sqrt(
                running_vol_sq / running_vol_cnt -
                (running_vol_sum / running_vol_cnt) ** 2
            )
            local_vol = max(local_vol, 1e-8)

            # In high vol regime, dampen the signal (makes z-score harder to hit)
            # In low vol regime, amplify slightly
            vol_ratio = local_vol / max(vol_threshold, 1e-8)
            if vol_ratio > 1.0:
                # High vol: dampen by ratio
                adaptive_signal[i] = v / vol_ratio
            else:
                # Low vol: slight amplification (capped at 1.5x)
                adaptive_signal[i] = v * min(1.0 / max(vol_ratio, 0.5), 1.5)
        else:
            adaptive_signal[i] = v

    bar_signal, stats = predictions_to_bar_signal(
        adaptive_signal, event_timestamps, date_str, running_stats
    )

    meta = {
        'signal_type': 'vol_adaptive',
        'vol_threshold': round(float(vol_threshold), 6),
        'low_vol_z': low_vol_z,
        'high_vol_z': high_vol_z,
    }

    return bar_signal, stats


def generate_vol_adaptive_configs() -> List[Dict]:
    """Generate configs for volatility-adaptive strategy."""
    configs = []

    for z in [2.0, 2.5, 3.0, 3.5]:
        for hold_ms in [10000, 30000, 60000]:
            hold_label = {10000: '10s', 30000: '30s', 60000: '60s'}[hold_ms]
            configs.append({
                'label': f'S4_voladapt_z{int(z*10):02d}_hold{hold_label}',
                'description': f'Vol-adaptive, z>{z}, hold {hold_label}',
                'strategy_group': 'vol_adaptive',
                'signal_type': 'vol_adaptive',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', str(hold_ms),
                    '--quiet',
                ],
            })

    # With signal-flip
    for z in [2.5, 3.0, 3.5]:
        configs.append({
            'label': f'S4_voladapt_flip_z{int(z*10):02d}',
            'description': f'Vol-adaptive + flip exit, z>{z}',
            'strategy_group': 'vol_adaptive',
            'signal_type': 'vol_adaptive',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--hold-ms', '300000',
                '--signal-flip-exit',
                '--quiet',
            ],
        })

    # Vol-based exit from fill_sim (fast adverse move detection)
    for z in [2.5, 3.0]:
        for vol_exit_ticks, vol_exit_bars in [(3, 50), (4, 100), (5, 200)]:
            configs.append({
                'label': f'S4_voladapt_volexit_z{int(z*10):02d}_t{vol_exit_ticks}b{vol_exit_bars}',
                'description': f'Vol-adaptive + vol-exit {vol_exit_ticks}t/{vol_exit_bars}bars, z>{z}',
                'strategy_group': 'vol_adaptive',
                'signal_type': 'vol_adaptive',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', '60000',
                    '--vol-exit-ticks', str(vol_exit_ticks),
                    '--vol-exit-bars', str(vol_exit_bars),
                    '--quiet',
                ],
            })

    return configs


# ============================================================
# Strategy 5: Time-of-Day Optimization
# ============================================================

def generate_time_of_day_configs() -> List[Dict]:
    """Generate configs testing different time-of-day windows.

    Sessions:
      - Open rush: 09:30 - 10:00
      - Morning: 10:00 - 11:30
      - Prime: 10:30 - 14:30
      - Midday: 11:30 - 14:00
      - Close: 15:00 - 16:00
      - Full day: no filter
    """
    configs = []

    sessions = [
        ('open', '09:30', '10:00'),
        ('morning', '10:00', '11:30'),
        ('prime', '10:30', '14:30'),
        ('midday', '11:30', '14:00'),
        ('close', '15:00', '16:00'),
        ('am', '09:30', '12:00'),
        ('pm', '12:00', '16:00'),
    ]

    # 5A: Time windows with various thresholds
    for session_name, start, end in sessions:
        for z in [2.0, 2.5, 3.0]:
            for hold_ms in [10000, 30000]:
                hold_label = {10000: '10s', 30000: '30s'}[hold_ms]
                configs.append({
                    'label': f'S5_tod_{session_name}_z{int(z*10):02d}_hold{hold_label}',
                    'description': f'TOD {session_name} ({start}-{end}), z>{z}, hold {hold_label}',
                    'strategy_group': 'time_of_day',
                    'cli_args': [
                        '--chase-entry',
                        '--signal-threshold', str(z),
                        '--hold-ms', str(hold_ms),
                        '--time-window-start', start,
                        '--time-window-end', end,
                        '--quiet',
                    ],
                })

    # 5B: Prime hours (built-in flag) with signal-flip
    for z in [2.0, 2.5, 3.0, 3.5]:
        configs.append({
            'label': f'S5_prime_flip_z{int(z*10):02d}',
            'description': f'Prime hours + flip exit, z>{z}',
            'strategy_group': 'time_of_day',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--hold-ms', '300000',
                '--prime-hours',
                '--signal-flip-exit',
                '--quiet',
            ],
        })

    return configs


# ============================================================
# Strategy 6: Momentum Burst Detection
# ============================================================

def generate_momentum_burst_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    burst_window: int = 3,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate momentum burst signal.

    Enter ONLY when:
    (a) high confidence signal AND
    (b) signal magnitude is INCREASING over last `burst_window` predictions
        in the same direction.

    This catches burst momentum moves where the model sees accelerating
    conviction.
    """
    preds = fold_data['predictions']
    p_10s = preds[:, 2]

    burst_signal = np.zeros_like(p_10s)
    n_bursts = 0

    for i in range(burst_window, len(p_10s)):
        current = p_10s[i]
        if current == 0:
            continue

        # Check last `burst_window` predictions
        window = p_10s[i - burst_window:i + 1]

        # All same sign (sustained direction)
        signs = np.sign(window)
        if not np.all(signs == signs[0]) or signs[0] == 0:
            continue

        # Magnitude increasing (each pred stronger than previous)
        magnitudes = np.abs(window)
        is_increasing = True
        for j in range(1, len(magnitudes)):
            if magnitudes[j] <= magnitudes[j - 1]:
                is_increasing = False
                break

        if is_increasing:
            # Burst detected! Use current magnitude as signal
            burst_signal[i] = current
            n_bursts += 1

    bar_signal, stats = predictions_to_bar_signal(
        burst_signal, event_timestamps, date_str, running_stats
    )

    meta = {
        'signal_type': 'momentum_burst',
        'burst_window': burst_window,
        'n_bursts': n_bursts,
        'n_total_preds': len(p_10s),
        'burst_rate': round(n_bursts / max(len(p_10s), 1), 4),
    }

    return bar_signal, stats


def generate_momentum_burst_configs() -> List[Dict]:
    """Generate configs for momentum burst strategy."""
    configs = []

    # 6A: Different burst windows and thresholds
    for z in [1.5, 2.0, 2.5, 3.0, 3.5]:
        for hold_ms in [10000, 30000, 60000]:
            hold_label = {10000: '10s', 30000: '30s', 60000: '60s'}[hold_ms]
            configs.append({
                'label': f'S6_burst_z{int(z*10):02d}_hold{hold_label}',
                'description': f'Momentum burst (w=3), z>{z}, hold {hold_label}',
                'strategy_group': 'momentum_burst',
                'signal_type': 'momentum_burst',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', str(hold_ms),
                    '--quiet',
                ],
            })

    # 6B: Burst + signal-flip exit
    for z in [2.0, 2.5, 3.0]:
        configs.append({
            'label': f'S6_burst_flip_z{int(z*10):02d}',
            'description': f'Momentum burst + flip exit, z>{z}',
            'strategy_group': 'momentum_burst',
            'signal_type': 'momentum_burst',
            'cli_args': [
                '--chase-entry',
                '--signal-threshold', str(z),
                '--hold-ms', '300000',
                '--signal-flip-exit',
                '--quiet',
            ],
        })

    # 6C: Burst + TP/SL
    for z in [2.0, 2.5, 3.0]:
        for tp, sl in [(3, 2), (4, 2), (5, 3)]:
            configs.append({
                'label': f'S6_burst_tp{tp}sl{sl}_z{int(z*10):02d}',
                'description': f'Momentum burst + TP={tp}t SL={sl}t, z>{z}',
                'strategy_group': 'momentum_burst',
                'signal_type': 'momentum_burst',
                'cli_args': [
                    '--chase-entry',
                    '--signal-threshold', str(z),
                    '--hold-ms', '60000',
                    '--take-profit-ticks', str(tp),
                    '--stop-loss-ticks', str(sl),
                    '--quiet',
                ],
            })

    return configs


# ============================================================
# Fill Simulator Interface
# ============================================================

def run_fill_sim(
    date_str: str,
    pred_file: Path,
    config: Dict,
    out_dir: Path,
) -> Optional[Dict]:
    """Run Rust fill_sim_cli for a single day + config."""
    if not BINARY.exists():
        return None

    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = out_dir / f'{config["label"]}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
    ] + config['cli_args']

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            # Only log non-trivially
            if 'no trades' not in r.stderr.lower():
                log.debug(f"Sim failed {config['label']}/{date_str}: {r.stderr[:200]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return None
    except Exception as e:
        log.debug(f"Error {config['label']}/{date_str}: {e}")
        return None


# ============================================================
# Signal Preparation Pipeline
# ============================================================

def prepare_signals_for_model(
    folds: Dict[str, Dict],
    signal_type: str = 'standard',
    gate_models: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Dict[str, Path]]:
    """Prepare bar-indexed prediction NPZ files for each signal type.

    Signal types:
      - standard: z-scored 10s horizon (baseline, used by strategies 1, 5)
      - multi_horizon: 3-horizon agreement composite
      - embedding_gated: MLP-gated signal
      - vol_adaptive: volatility-adapted signal
      - momentum_burst: burst detection signal

    Returns: {signal_type: {date_str: npz_path}}
    """
    signal_types = ['standard']
    if signal_type == 'all':
        signal_types = ['standard', 'multi_horizon', 'vol_adaptive', 'momentum_burst']
        # Only add embedding_gated if we have embeddings
        if any('embeddings' in f for f in folds.values()):
            signal_types.append('embedding_gated')
    elif signal_type != 'standard':
        signal_types = [signal_type]

    all_files = {}

    for stype in signal_types:
        log.info(f"\n  Generating '{stype}' signals...")
        saved = {}
        running_stats = None

        # Sort dates for walk-forward consistency
        sorted_dates = sorted(folds.keys())

        for date_str in sorted_dates:
            fold_data = folds[date_str]
            cache_file = PRED_CACHE_DIR / f'{stype}_{date_str}.npz'

            # Check MBO file exists
            mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
            if not mbo_file.exists():
                mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
            if not mbo_file.exists():
                continue

            if cache_file.exists():
                saved[date_str] = cache_file
                # Still need to update running stats for walk-forward
                # Load and recompute if standard
                continue

            timestamps = load_event_timestamps(date_str)
            if timestamps is None:
                continue

            if stype == 'standard':
                p_10s = fold_data['predictions'][:, 2]
                bar_signal, running_stats = predictions_to_bar_signal(
                    p_10s, timestamps, date_str, running_stats
                )
            elif stype == 'multi_horizon':
                bar_signal, running_stats = generate_multi_horizon_signal(
                    fold_data, timestamps, date_str, running_stats
                )
            elif stype == 'embedding_gated':
                gate = gate_models.get(date_str) if gate_models else None
                bar_signal, running_stats = generate_embedding_gated_signal(
                    fold_data, timestamps, date_str,
                    gate_model=gate, gate_threshold=0.6,
                    running_stats=running_stats,
                )
            elif stype == 'vol_adaptive':
                bar_signal, running_stats = generate_vol_adaptive_signal(
                    fold_data, timestamps, date_str,
                    running_stats=running_stats,
                )
            elif stype == 'momentum_burst':
                bar_signal, running_stats = generate_momentum_burst_signal(
                    fold_data, timestamps, date_str,
                    burst_window=3, running_stats=running_stats,
                )
            else:
                continue

            np.savez_compressed(str(cache_file), predictions=bar_signal)
            saved[date_str] = cache_file

            n_nonzero = int(np.sum(bar_signal != 0))
            log.info(f"    {date_str}: {n_nonzero} bar signals")

            del timestamps
            gc.collect()

        all_files[stype] = saved
        log.info(f"    => {len(saved)} days prepared for '{stype}'")

    return all_files


def train_embedding_gates_walkforward(
    folds: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Train embedding gates in walk-forward fashion.

    Gate for fold N is trained on fold N-1 embeddings/labels.
    """
    sorted_dates = sorted(folds.keys())
    gates = {}

    for i in range(1, len(sorted_dates)):
        train_date = sorted_dates[i - 1]
        test_date = sorted_dates[i]

        train_fold = folds[train_date]
        if 'embeddings' not in train_fold:
            continue

        log.info(f"  Training embedding gate: {train_date} -> {test_date}")
        gate = train_embedding_gate(
            train_fold['embeddings'],
            train_fold['labels'][:, 2],  # 10s labels
            train_fold['predictions'][:, 2],  # 10s predictions
        )

        if gate is not None:
            gates[test_date] = gate

            # Evaluate on training data
            train_mask = apply_embedding_gate(gate, train_fold['embeddings'], 0.6)
            pass_rate = train_mask.sum() / len(train_mask)
            log.info(f"    Gate pass rate (in-sample): {pass_rate:.1%}")

    return gates


# ============================================================
# Analysis & Reporting
# ============================================================

def aggregate_results(
    results: Dict[str, Dict[str, Dict]],
    configs: List[Dict],
) -> List[Dict]:
    """Aggregate per-day sim results into per-strategy summaries."""
    config_map = {c['label']: c for c in configs}
    summaries = []

    for label, date_results in results.items():
        total_pnl = 0.0
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0
        daily_pnls = []
        all_trade_pnls = []
        dates = []

        for date_str, res in sorted(date_results.items()):
            day_pnl = res.get('total_pnl_dollars', 0)
            total_pnl += day_pnl
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            daily_pnls.append(day_pnl)
            dates.append(date_str)

            if 'trades' in res:
                for trade in res['trades']:
                    pnl = trade.get('pnl_dollars', 0)
                    all_trade_pnls.append(pnl)
                    if pnl > 0:
                        total_wins += 1

        n_days = len(date_results)
        if n_days == 0:
            continue

        win_rate = total_wins / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)
        avg_daily = np.mean(daily_pnls) if daily_pnls else 0
        daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 1e-8

        downside = [min(0, x) for x in daily_pnls]
        downside_std = np.std(downside) if downside else 1e-8
        sortino = (avg_daily / max(downside_std, 1e-8)) * np.sqrt(252)
        sharpe = (avg_daily / max(daily_std, 1e-8)) * np.sqrt(252)

        gross_profit = sum(p for p in all_trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_trade_pnls if p < 0))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_trade_ticks = avg_trade_pnl / TICK_VALUE

        avg_winner = (np.mean([p for p in all_trade_pnls if p > 0])
                      if any(p > 0 for p in all_trade_pnls) else 0)
        avg_loser = (np.mean([p for p in all_trade_pnls if p < 0])
                     if any(p < 0 for p in all_trade_pnls) else 0)

        cum = np.cumsum(daily_pnls) if daily_pnls else np.array([0])
        peak = np.maximum.accumulate(cum)
        max_dd = abs(float((cum - peak).min())) if len(cum) > 0 else 0

        trades_per_day = total_trades / max(n_days, 1)

        cfg = config_map.get(label, {})
        description = cfg.get('description', label)
        strategy_group = cfg.get('strategy_group', 'unknown')

        summaries.append({
            'label': label,
            'description': description,
            'strategy_group': strategy_group,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'trades_per_day': round(trades_per_day, 1),
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'profit_factor': round(profit_factor, 2),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_trade_ticks': round(avg_trade_ticks, 3),
            'avg_winner': round(avg_winner, 2),
            'avg_loser': round(avg_loser, 2),
            'max_dd': round(max_dd, 2),
            'annualized_pnl': round(avg_daily * 252, 0),
            'daily_pnls': daily_pnls,
            'dates': dates,
        })

    summaries.sort(key=lambda x: x['sortino'], reverse=True)
    return summaries


def print_results_table(summaries: List[Dict], title: str = "Strategy Results"):
    """Print formatted strategy comparison table."""
    log.info(f"\n{'=' * 160}")
    log.info(f" {title}")
    log.info(f"{'=' * 160}")

    if not summaries:
        log.info("  No results to display.")
        return

    header = (
        f"{'Strategy':<45} "
        f"{'Total P&L':>10} "
        f"{'Trades':>7} "
        f"{'T/Day':>6} "
        f"{'FillR':>6} "
        f"{'WinR':>6} "
        f"{'AvgTrd':>8} "
        f"{'Ticks':>7} "
        f"{'Sharpe':>7} "
        f"{'Sortino':>8} "
        f"{'PF':>5} "
        f"{'MaxDD':>8} "
        f"{'Ann$':>10}"
    )
    log.info(header)
    log.info("-" * 160)

    for s in summaries:
        pnl_marker = '+' if s['total_pnl'] >= 0 else ''
        line = (
            f"{s['label']:<45} "
            f"{pnl_marker}${s['total_pnl']:>8,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['trades_per_day']:>5.1f} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"{s['avg_trade_ticks']:>6.2f}t "
            f"{s['sharpe']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f} "
            f"${s['annualized_pnl']:>9,.0f}"
        )
        log.info(line)

    n_days = summaries[0]['n_days'] if summaries else 0
    n_profitable = sum(1 for s in summaries if s['total_pnl'] > 0)
    log.info(f"\n  ES Futures | Tick=$12.50 | Commission=$4.70 RT | "
             f"{n_days} OOT days | {n_profitable}/{len(summaries)} profitable")


def print_top_strategies(summaries: List[Dict], top_n: int = 10):
    """Print detailed analysis of top N strategies."""
    log.info(f"\n{'=' * 80}")
    log.info(f"  TOP {top_n} STRATEGIES BY SORTINO")
    log.info(f"{'=' * 80}")

    for rank, s in enumerate(summaries[:top_n]):
        pnl_marker = "PROFITABLE" if s['total_pnl'] > 0 else "LOSS"
        log.info(f"\n  #{rank + 1} [{pnl_marker}] {s['label']}")
        log.info(f"  {s['description']}")
        log.info(f"  Total P&L: ${s['total_pnl']:,.2f} | "
                 f"Ann: ${s['annualized_pnl']:,.0f} | "
                 f"Sortino: {s['sortino']:.2f} | "
                 f"Sharpe: {s['sharpe']:.2f}")
        log.info(f"  Trades: {s['n_trades']} ({s['trades_per_day']:.1f}/day) | "
                 f"Fill: {s['fill_rate']:.1%} | "
                 f"Win: {s['win_rate']:.1%} | "
                 f"PF: {s['profit_factor']:.2f}")
        log.info(f"  Avg trade: ${s['avg_trade_pnl']:.2f} ({s['avg_trade_ticks']:.2f}t) | "
                 f"Avg win: ${s['avg_winner']:.2f} | "
                 f"Avg loss: ${s['avg_loser']:.2f}")
        log.info(f"  Max DD: ${s['max_dd']:,.2f}")

        if s.get('daily_pnls') and s.get('dates'):
            log.info(f"  Daily P&L:")
            for d, pnl in zip(s['dates'], s['daily_pnls']):
                marker = "  " if pnl >= 0 else " *"
                log.info(f"   {marker} {d}: ${pnl:>8,.2f}")


def print_strategy_group_summary(summaries: List[Dict]):
    """Print per-strategy-group summary."""
    groups = defaultdict(list)
    for s in summaries:
        groups[s['strategy_group']].append(s)

    log.info(f"\n{'=' * 100}")
    log.info(f"  STRATEGY GROUP SUMMARY")
    log.info(f"{'=' * 100}")
    log.info(f"{'Group':<20} {'Configs':>8} {'Profitable':>11} "
             f"{'Best P&L':>10} {'Best Sortino':>13} {'Best Label':<40}")
    log.info("-" * 100)

    for group_name in ['signal_flip', 'multi_horizon', 'embedding_gate',
                       'vol_adaptive', 'time_of_day', 'momentum_burst']:
        group = groups.get(group_name, [])
        if not group:
            continue
        n_profitable = sum(1 for s in group if s['total_pnl'] > 0)
        best = max(group, key=lambda x: x['sortino'])
        log.info(f"{group_name:<20} {len(group):>8} {n_profitable:>11} "
                 f"${best['total_pnl']:>9,.0f} {best['sortino']:>13.2f} "
                 f"{best['label']:<40}")


# ============================================================
# Main Orchestrator
# ============================================================

def run_model_sweep(
    model_name: str,
    folds: Dict[str, Dict],
    workers: int = 8,
    clear_cache: bool = False,
) -> Tuple[List[Dict], str]:
    """Run full strategy sweep for one model.

    Returns: (summaries, results_json_path)
    """
    log.info(f"\n{'#' * 80}")
    log.info(f"  MODEL: {model_name}")
    log.info(f"  Dates: {sorted(folds.keys())}")
    log.info(f"  Folds: {len(folds)}")
    log.info(f"{'#' * 80}")

    has_embeddings = any('embeddings' in f for f in folds.values())

    # Clear cache if requested
    if clear_cache:
        import shutil
        if PRED_CACHE_DIR.exists():
            for f in PRED_CACHE_DIR.glob('*.npz'):
                f.unlink()
            log.info("  Cache cleared")

    # ── Train embedding gates (walk-forward) ──
    gate_models = {}
    if has_embeddings:
        log.info("\n  Training embedding gates (walk-forward)...")
        gate_models = train_embedding_gates_walkforward(folds)
        log.info(f"  Trained {len(gate_models)} gates")

    # ── Prepare all signal types ──
    log.info("\n  Preparing signal files...")
    signal_files = prepare_signals_for_model(
        folds, signal_type='all', gate_models=gate_models
    )

    # ── Build all strategy configs ──
    all_configs = []

    # Strategy 1: Signal-Flip (uses standard signal)
    s1_configs = generate_signal_flip_configs()
    for c in s1_configs:
        c['signal_type'] = c.get('signal_type', 'standard')
    all_configs.extend(s1_configs)

    # Strategy 2: Multi-Horizon Agreement
    s2_configs = generate_multi_horizon_configs()
    all_configs.extend(s2_configs)

    # Strategy 3: Embedding Gate (only if we have embeddings)
    if has_embeddings and 'embedding_gated' in signal_files:
        s3_configs = generate_embedding_gate_configs()
        all_configs.extend(s3_configs)

    # Strategy 4: Volatility-Adaptive
    s4_configs = generate_vol_adaptive_configs()
    all_configs.extend(s4_configs)

    # Strategy 5: Time-of-Day
    s5_configs = generate_time_of_day_configs()
    for c in s5_configs:
        c['signal_type'] = c.get('signal_type', 'standard')
    all_configs.extend(s5_configs)

    # Strategy 6: Momentum Burst
    s6_configs = generate_momentum_burst_configs()
    all_configs.extend(s6_configs)

    log.info(f"\n  Total configs to test: {len(all_configs)}")
    for group in ['signal_flip', 'multi_horizon', 'embedding_gate',
                  'vol_adaptive', 'time_of_day', 'momentum_burst']:
        n = sum(1 for c in all_configs if c.get('strategy_group') == group)
        if n > 0:
            log.info(f"    {group}: {n} configs")

    # ── Build job list: (config, date, pred_file) ──
    jobs = []
    for config in all_configs:
        stype = config.get('signal_type', 'standard')
        if stype not in signal_files:
            continue
        for date_str, pred_file in sorted(signal_files[stype].items()):
            jobs.append({
                'config': config,
                'date': date_str,
                'pred_file': pred_file,
            })

    log.info(f"\n  Total sim jobs: {len(jobs)}")

    # ── Run all sims ──
    sim_out = RESULTS_DIR / f'sim_{model_name}_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    results = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['config'], sim_out,
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            job = futures[future]
            try:
                result = future.result()
                if result:
                    label = job['config']['label']
                    if label not in results:
                        results[label] = {}
                    results[label][job['date']] = result
            except Exception as e:
                log.debug(f"Job error: {e}")

            if done % 50 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(jobs) - done) / max(rate, 0.01)
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, "
                         f"~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log.info(f"\n  Sweep done: {done} jobs in {elapsed:.1f}s "
             f"({len(results)} strategies with results)")

    # ── Aggregate ──
    summaries = aggregate_results(results, all_configs)

    # ── Report ──
    # Per-group tables
    for group in ['signal_flip', 'multi_horizon', 'embedding_gate',
                  'vol_adaptive', 'time_of_day', 'momentum_burst']:
        group_summaries = [s for s in summaries if s['strategy_group'] == group]
        if group_summaries:
            print_results_table(
                group_summaries,
                f"{model_name.upper()} -- Strategy {group.replace('_', ' ').title()}"
            )

    # Overall ranking
    print_results_table(
        summaries,
        f"{model_name.upper()} -- ALL STRATEGIES (Sorted by Sortino)"
    )

    # Top strategies detail
    print_top_strategies(summaries, top_n=15)

    # Group summary
    print_strategy_group_summary(summaries)

    # ── Save results ──
    out_file = RESULTS_DIR / f'advanced_results_{model_name}_{_ts}.json'
    save_data = []
    for s in summaries:
        entry = {k: v for k, v in s.items()}
        save_data.append(entry)

    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'model': model_name,
            'instrument': 'ES',
            'tick_value': TICK_VALUE,
            'commission_rt': COMMISSION_RT,
            'n_days': len(folds),
            'dates': sorted(folds.keys()),
            'n_configs': len(all_configs),
            'n_with_results': len(results),
            'strategies': save_data,
        }, f, indent=2, default=str)
    log.info(f"\n  Results saved: {out_file}")

    return summaries, str(out_file)


def main():
    parser = argparse.ArgumentParser(
        description='Advanced Execution Strategies v1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', type=str, default='both',
                        choices=['cnn-mamba-v2', 'mamba-v7', 'both'],
                        help='Model to test (default: both)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel sim workers (default: 8)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear prediction cache before running')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show configs without running')

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("ADVANCED EXECUTION STRATEGIES v1")
    log.info("=" * 80)
    log.info(f"  Instrument:     ES (E-mini S&P 500)")
    log.info(f"  Tick value:     ${TICK_VALUE}")
    log.info(f"  Commission RT:  ${COMMISSION_RT}")
    log.info(f"  Model:          {args.model}")
    log.info(f"  Workers:        {args.workers}")
    log.info(f"  Binary:         {BINARY}")
    log.info(f"  Results:        {RESULTS_DIR}")
    log.info("=" * 80)

    if not BINARY.exists():
        log.error(f"fill_sim_cli binary not found: {BINARY}")
        sys.exit(1)

    all_summaries = {}

    # ── CNN-Mamba v2 (Feb 23-26, 4 days, has embeddings) ──
    if args.model in ['cnn-mamba-v2', 'both']:
        log.info("\n  Discovering CNN-Mamba v2 predictions...")
        cnn_mamba_folds = discover_folds(CNN_MAMBA_V2_DIR)
        if cnn_mamba_folds:
            if args.dry_run:
                log.info(f"  CNN-Mamba v2: {len(cnn_mamba_folds)} folds")
            else:
                summaries, out_file = run_model_sweep(
                    'cnn_mamba_v2', cnn_mamba_folds,
                    workers=args.workers,
                    clear_cache=args.clear_cache,
                )
                all_summaries['cnn_mamba_v2'] = summaries
        else:
            log.warning("  No CNN-Mamba v2 predictions found")

    # ── Mamba v7 (Mar 1-13, 11 days) ──
    if args.model in ['mamba-v7', 'both']:
        log.info("\n  Discovering Mamba v7 predictions...")
        mamba_folds = discover_folds(MAMBA_V7_DIR)
        if mamba_folds:
            if args.dry_run:
                log.info(f"  Mamba v7: {len(mamba_folds)} folds")
            else:
                summaries, out_file = run_model_sweep(
                    'mamba_v7', mamba_folds,
                    workers=args.workers,
                    clear_cache=args.clear_cache,
                )
                all_summaries['mamba_v7'] = summaries
        else:
            log.warning("  No Mamba v7 predictions found")

    if args.dry_run:
        # Show all configs
        all_configs = (
            generate_signal_flip_configs() +
            generate_multi_horizon_configs() +
            generate_embedding_gate_configs() +
            generate_vol_adaptive_configs() +
            generate_time_of_day_configs() +
            generate_momentum_burst_configs()
        )
        log.info(f"\n  Total configs: {len(all_configs)}")
        for group in ['signal_flip', 'multi_horizon', 'embedding_gate',
                      'vol_adaptive', 'time_of_day', 'momentum_burst']:
            group_configs = [c for c in all_configs if c.get('strategy_group') == group]
            log.info(f"\n  {group}: {len(group_configs)} configs")
            for c in group_configs[:5]:
                log.info(f"    {c['label']}: {c['description']}")
            if len(group_configs) > 5:
                log.info(f"    ... and {len(group_configs) - 5} more")
        return

    # ── Cross-Model Comparison ──
    if len(all_summaries) > 1:
        log.info(f"\n{'#' * 80}")
        log.info(f"  CROSS-MODEL COMPARISON")
        log.info(f"{'#' * 80}")

        for model_name, summaries in all_summaries.items():
            if summaries:
                best = summaries[0]
                n_profitable = sum(1 for s in summaries if s['total_pnl'] > 0)
                log.info(f"\n  {model_name}:")
                log.info(f"    {len(summaries)} strategies tested, "
                         f"{n_profitable} profitable")
                log.info(f"    Best: {best['label']} "
                         f"(Sortino={best['sortino']:.2f}, "
                         f"P&L=${best['total_pnl']:,.0f})")

    # ── Final Summary ──
    log.info(f"\n{'=' * 80}")
    log.info(f"  EXECUTION COMPLETE")
    log.info(f"{'=' * 80}")

    any_profitable = False
    for model_name, summaries in all_summaries.items():
        profitable = [s for s in summaries if s['total_pnl'] > 0]
        if profitable:
            any_profitable = True
            log.info(f"\n  {model_name}: {len(profitable)} PROFITABLE strategies found!")
            for s in profitable[:5]:
                log.info(f"    {s['label']}: ${s['total_pnl']:,.2f} "
                         f"({s['trades_per_day']:.1f} T/day, "
                         f"Sortino={s['sortino']:.2f})")

    if not any_profitable:
        log.info("\n  NO profitable strategies found across all models.")
        log.info("  Closest to break-even:")
        for model_name, summaries in all_summaries.items():
            if summaries:
                best = min(summaries, key=lambda x: abs(x['total_pnl']))
                log.info(f"    {model_name}: {best['label']} = "
                         f"${best['total_pnl']:,.2f}")

    log.info(f"\n  Log: {_log_file}")
    log.info(f"{'=' * 80}")


if __name__ == '__main__':
    main()
