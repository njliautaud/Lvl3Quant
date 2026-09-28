#!/usr/bin/env python3
"""
Portfolio Execution v4 — Multi-Strategy Portfolio Tester
=========================================================
Advanced strategy groups for CNN-Mamba v2 OOT predictions on ES futures.

Strategy Groups:
  G1: Portfolio (run multiple strategies simultaneously, max 1 contract)
  G2: Momentum Filter (consecutive signal agreement)
  G3: Multi-Horizon Agreement (1s/5s/10s must agree)
  G4: Signal Strength Combos (weighted/max/min z-scores)
  G5: Embedding-Based Gating (cluster profitability filter)
  G6: Adaptive Hold Time (hold scales with conviction)
  G7: Bracket + Time Filter Combos

Data:
  Predictions: CNN-Mamba v2, 5 OOT folds (Feb 23-27)
  Fill sim: Rust fill_sim_cli with real MBO data (FIFO queue sim)

ES Futures: tick=$12.50, commission=$4.70 RT, point_value=$50

Usage:
    python portfolio_execution_v4.py
    python portfolio_execution_v4.py --groups G2,G3 --workers 8
    python portfolio_execution_v4.py --dry-run
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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
PRED_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'portfolio_v4'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'portfolio_v4'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── ES Futures Constants ──────────────────────────────────────────────────
TICK_VALUE = 12.50
POINT_VALUE = 50.00
COMMISSION_RT = 4.70
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE

# ── Bar/Timing Constants ──
BARS_PER_SEC = 10
BAR_NS = 100_000_000  # 100ms
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# ── Model Constants ──
WINDOW = 1000
STRIDE = 500

# ── Fold Discovery ──
FOLD_FILES = sorted(PRED_DIR.glob('fold_0*_oot_predictions.npz'))

# ── Timestamp ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──
_log_file = str(RESULTS_DIR / f'portfolio_v4_{_ts}.log')
log = logging.getLogger('portfolio_v4')
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
    """Load a CNN-Mamba v2 fold prediction file."""
    try:
        data = np.load(str(fold_path), allow_pickle=True)
        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]
        result = {
            'predictions': data['predictions'].astype(np.float64),  # (N, 3)
            'labels': data['labels'].astype(np.float64),            # (N, 3)
            'date_str': date_str,
            'n_samples': data['predictions'].shape[0],
            'fold_path': str(fold_path),
        }
        if 'embeddings' in data:
            result['embeddings'] = data['embeddings'].astype(np.float64)
        return result
    except Exception as e:
        log.warning(f"Failed to load {fold_path}: {e}")
        return None


def load_event_timestamps(date_str: str) -> Optional[np.ndarray]:
    """Load event timestamps for a date."""
    ev_file = EVENT_DIR / f'{date_str}_mbo_events.npz'
    if not ev_file.exists():
        return None
    try:
        ev_data = np.load(str(ev_file), allow_pickle=True)
        ts = ev_data['timestamps']
        del ev_data
        return ts
    except Exception as e:
        log.warning(f"Failed to load events for {date_str}: {e}")
        return None


# ============================================================
# Signal Generation Core
# ============================================================

def map_predictions_to_bars(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map per-window predictions to bar indices.

    Returns:
        (bar_indices, predictions_rth) — only RTH-valid entries
    """
    n_events = len(event_timestamps)
    n_preds = len(predictions)

    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1

    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    elif n_preds > len(label_idxs):
        predictions = predictions[:len(label_idxs)]

    if len(predictions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    pred_timestamps = event_timestamps[label_idxs]
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)

    return bar_indices[rth_mask], predictions[rth_mask]


def expanding_zscore_bar_signal(
    raw_preds: np.ndarray,
    bar_indices: np.ndarray,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Apply expanding z-score to bar-level signal (no lookahead).

    Returns:
        (bar_signal of shape (N_RTH_BARS,), updated running_stats)
    """
    if running_stats is None:
        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

    # Build raw bar array
    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, val in zip(bar_indices, raw_preds):
        bar_preds[bi] = val

    zscore = np.zeros(N_RTH_BARS, dtype=np.float64)
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
            zscore[i] = (v - mean) / std

    return zscore, {'sum': rs, 'sq': rsq, 'count': cnt}


def multi_horizon_expanding_zscore(
    predictions_3h: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats_list: Optional[List[Dict]] = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """Compute expanding z-scores for all 3 horizons independently.

    Returns:
        ([z_1s, z_5s, z_10s], [stats_1s, stats_5s, stats_10s])
    """
    if running_stats_list is None:
        running_stats_list = [None, None, None]

    z_signals = []
    new_stats = []

    for h in range(3):
        preds_h = predictions_3h[:, h]
        bar_idx, preds_rth = map_predictions_to_bars(preds_h, event_timestamps, date_str)
        z_h, stats_h = expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats_list[h])
        z_signals.append(z_h)
        new_stats.append(stats_h)

    return z_signals, new_stats


# ============================================================
# Strategy Signal Generators
# ============================================================

def generate_base_10s_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Standard 10s horizon z-score signal (baseline for simple strategies)."""
    preds_10s = fold_data['predictions'][:, 2]
    bar_idx, preds_rth = map_predictions_to_bars(preds_10s, event_timestamps, date_str)
    return expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats)


def generate_momentum_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    n_consecutive: int,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Momentum filter: only signal when last N consecutive predictions agree on direction.

    The signal value is the z-score of the current prediction, but zeroed out
    if the last N predictions didn't all agree on direction.
    """
    preds_10s = fold_data['predictions'][:, 2]
    bar_idx, preds_rth = map_predictions_to_bars(preds_10s, event_timestamps, date_str)

    # First get the z-score signal
    z_signal, new_stats = expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats)

    # Build momentum mask: for each non-zero bar, check if last N consecutive
    # non-zero predictions all have the same sign
    nonzero_bars = np.where(z_signal != 0)[0]
    if len(nonzero_bars) < n_consecutive:
        return np.zeros(N_RTH_BARS, dtype=np.float64), new_stats

    momentum_signal = np.zeros(N_RTH_BARS, dtype=np.float64)
    # Track recent non-zero signal signs
    recent_signs = []
    for bar in nonzero_bars:
        sign = 1 if z_signal[bar] > 0 else -1
        recent_signs.append(sign)
        if len(recent_signs) > n_consecutive:
            recent_signs = recent_signs[-n_consecutive:]
        if len(recent_signs) >= n_consecutive:
            if all(s == recent_signs[-1] for s in recent_signs[-n_consecutive:]):
                momentum_signal[bar] = z_signal[bar]

    return momentum_signal, new_stats


def generate_multi_horizon_agreement_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    horizons_required: List[int],  # indices: 0=1s, 1=5s, 2=10s
    running_stats_list: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Multi-horizon agreement: only signal when specified horizons agree on direction.

    Signal magnitude is from the 10s z-score.
    """
    z_signals, new_stats = multi_horizon_expanding_zscore(
        fold_data['predictions'], event_timestamps, date_str, running_stats_list
    )

    # Agreement mask: all specified horizons must have same sign (and be non-zero)
    agreement = np.ones(N_RTH_BARS, dtype=bool)
    for h in horizons_required:
        agreement &= (z_signals[h] != 0)

    # Check sign agreement
    if len(horizons_required) >= 2:
        ref_sign = np.sign(z_signals[horizons_required[0]])
        for h in horizons_required[1:]:
            agreement &= (np.sign(z_signals[h]) == ref_sign)

    # Use 10s z-score as signal magnitude where agreement holds
    result = np.where(agreement, z_signals[2], 0.0)  # 10s = index 2
    return result, new_stats


def generate_weighted_z_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    weights: Tuple[float, float, float],  # (w_1s, w_5s, w_10s)
    running_stats_list: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Weighted z-score: signal = w1*z_1s + w2*z_5s + w3*z_10s."""
    z_signals, new_stats = multi_horizon_expanding_zscore(
        fold_data['predictions'], event_timestamps, date_str, running_stats_list
    )

    weighted = (weights[0] * z_signals[0] +
                weights[1] * z_signals[1] +
                weights[2] * z_signals[2])

    # Zero out bars where no predictions exist (all 3 must be non-zero)
    has_signal = (z_signals[0] != 0) | (z_signals[1] != 0) | (z_signals[2] != 0)
    weighted = np.where(has_signal, weighted, 0.0)

    return weighted, new_stats


def generate_max_z_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats_list: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Max z-score: signal = sign(z_10s) * max(|z_1s|, |z_5s|, |z_10s|)."""
    z_signals, new_stats = multi_horizon_expanding_zscore(
        fold_data['predictions'], event_timestamps, date_str, running_stats_list
    )

    abs_z = np.stack([np.abs(z_signals[0]), np.abs(z_signals[1]), np.abs(z_signals[2])], axis=0)
    max_abs = np.max(abs_z, axis=0)
    # Direction from 10s horizon
    result = np.sign(z_signals[2]) * max_abs
    # Zero out where no signal
    has_signal = (z_signals[2] != 0)
    result = np.where(has_signal, result, 0.0)

    return result, new_stats


def generate_min_z_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats_list: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Min z-score: signal = sign(z_10s) * min(|z_1s|, |z_5s|, |z_10s|).

    Only non-zero when ALL horizons have a signal.
    """
    z_signals, new_stats = multi_horizon_expanding_zscore(
        fold_data['predictions'], event_timestamps, date_str, running_stats_list
    )

    abs_z = np.stack([np.abs(z_signals[0]), np.abs(z_signals[1]), np.abs(z_signals[2])], axis=0)
    min_abs = np.min(abs_z, axis=0)
    # All must be non-zero
    all_nonzero = (z_signals[0] != 0) & (z_signals[1] != 0) & (z_signals[2] != 0)
    result = np.where(all_nonzero, np.sign(z_signals[2]) * min_abs, 0.0)

    return result, new_stats


def generate_embedding_gated_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    cluster_labels: Optional[np.ndarray],
    cluster_profitability: Optional[Dict[int, float]],
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict, Optional[np.ndarray], Optional[Dict[int, float]]]:
    """Embedding-gated signal: only trade in historically profitable clusters.

    First pass (cluster_labels=None): cluster embeddings, record which clusters
    are profitable based on label data (not P&L, to avoid lookahead).
    Second+ pass: gate signal by profitable cluster membership.

    Returns:
        (signal, stats, cluster_labels_for_this_fold, updated_profitability)
    """
    from sklearn.cluster import KMeans

    preds_10s = fold_data['predictions'][:, 2]
    labels_10s = fold_data['labels'][:, 2]
    embeddings = fold_data.get('embeddings')
    bar_idx, preds_rth = map_predictions_to_bars(preds_10s, event_timestamps, date_str)
    z_signal, new_stats = expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats)

    if embeddings is None:
        return z_signal, new_stats, None, cluster_profitability

    n_preds = len(preds_10s)

    # Cluster the embeddings
    n_clusters = 10
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=100)
    cl = km.fit_predict(embeddings[:n_preds])

    # Compute per-cluster profitability from LABELS (not future P&L)
    # "Profitable" = predictions have positive correlation with labels in that cluster
    if cluster_profitability is None:
        cluster_profitability = {}

    # Update profitability with this fold's data (expanding window)
    for c in range(n_clusters):
        mask = cl == c
        if mask.sum() < 10:
            continue
        # Correlation between prediction and label in this cluster
        p = preds_10s[mask]
        l = labels_10s[mask]
        if np.std(p) > 1e-10 and np.std(l) > 1e-10:
            corr = float(np.corrcoef(p, l)[0, 1])
        else:
            corr = 0.0
        # Expanding: average with prior
        if c in cluster_profitability:
            old = cluster_profitability[c]
            cluster_profitability[c] = (old + corr) / 2.0
        else:
            cluster_profitability[c] = corr

    # Gate: only keep signals from profitable clusters (corr > 0)
    # Map predictions to bar-level cluster assignments
    # We need the same bar mapping used for predictions
    n_events = len(event_timestamps)
    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1
    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]

    pred_timestamps = event_timestamps[label_idxs[:len(cl)]]
    rth_start = rth_start_ns_for_date(date_str)
    bi = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bi >= 0) & (bi < N_RTH_BARS)

    # Build bar-level cluster map
    bar_cluster = np.full(N_RTH_BARS, -1, dtype=np.int32)
    for b, c_label, m in zip(bi, cl[:len(bi)], rth_mask):
        if m:
            bar_cluster[b] = c_label

    # Gate signal
    gated = np.zeros(N_RTH_BARS, dtype=np.float64)
    profitable_clusters = {c for c, prof in cluster_profitability.items() if prof > 0}
    for i in range(N_RTH_BARS):
        if z_signal[i] != 0 and bar_cluster[i] in profitable_clusters:
            gated[i] = z_signal[i]

    return gated, new_stats, cl, cluster_profitability


def generate_adaptive_hold_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict, np.ndarray]:
    """Adaptive hold: returns z-score signal AND per-bar hold times.

    Hold time = 10s if |z|<3.0, 30s if 3.0<=|z|<5.0, 60s if |z|>=5.0.
    The actual hold time must be baked into separate prediction files for
    each hold tier. Returns hold_ms per bar.
    """
    preds_10s = fold_data['predictions'][:, 2]
    bar_idx, preds_rth = map_predictions_to_bars(preds_10s, event_timestamps, date_str)
    z_signal, new_stats = expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats)

    hold_ms = np.zeros(N_RTH_BARS, dtype=np.int32)
    for i in range(N_RTH_BARS):
        az = abs(z_signal[i])
        if az > 0:
            if az >= 5.0:
                hold_ms[i] = 60000
            elif az >= 3.0:
                hold_ms[i] = 30000
            else:
                hold_ms[i] = 10000

    return z_signal, new_stats, hold_ms


# ============================================================
# Strategy Specification
# ============================================================

@dataclass
class StrategySpec:
    """Strategy specification with signal generation parameters."""
    label: str
    description: str
    group: str

    # Signal generation type
    signal_type: str = 'base_10s'  # base_10s, momentum, agree, weighted, max_z, min_z, emb_gate, adaptive

    # Signal params
    momentum_n: int = 2
    agree_horizons: List[int] = field(default_factory=lambda: [0, 1, 2])
    z_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)

    # fill_sim_cli params
    signal_threshold: float = 2.5
    hold_ms: int = 30000
    chase_entry: bool = True
    stop_loss_ticks: Optional[int] = None
    take_profit_ticks: Optional[int] = None
    trailing_ticks: Optional[int] = None
    ratchet_stop: bool = False
    time_window_start: str = ""
    time_window_end: str = ""

    def to_cli_args(self) -> List[str]:
        args = []
        if self.chase_entry:
            args.append('--chase-entry')
            args.extend(['--chase-max-ticks', '1'])
            args.extend(['--chase-max-reprices', '3'])

        args.extend(['--signal-threshold', str(self.signal_threshold)])
        args.extend(['--hold-ms', str(self.hold_ms)])

        if self.stop_loss_ticks is not None:
            args.extend(['--stop-loss-ticks', str(self.stop_loss_ticks)])
        if self.take_profit_ticks is not None:
            args.extend(['--take-profit-ticks', str(self.take_profit_ticks)])
        if self.trailing_ticks is not None:
            args.extend(['--trailing-ticks', str(self.trailing_ticks)])
        if self.ratchet_stop:
            args.append('--ratchet-stop')
        if self.time_window_start:
            args.extend(['--time-window-start', self.time_window_start])
            args.extend(['--time-window-end', self.time_window_end])

        args.append('--quiet')
        return args


def build_all_strategies() -> Dict[str, List[StrategySpec]]:
    """Build all strategy groups."""
    strategies = {}

    # ── GROUP 2: Momentum Filter ──
    strategies['G2'] = [
        StrategySpec(
            label='momentum_2_z2.5_30s',
            description='Momentum 2: last 2 consecutive agree, z>2.5, 30s hold',
            group='G2', signal_type='momentum', momentum_n=2,
            signal_threshold=2.5, hold_ms=30000,
        ),
        StrategySpec(
            label='momentum_3_z2.5_30s',
            description='Momentum 3: last 3 consecutive agree, z>2.5, 30s hold',
            group='G2', signal_type='momentum', momentum_n=3,
            signal_threshold=2.5, hold_ms=30000,
        ),
        StrategySpec(
            label='momentum_2_z3.0_30s',
            description='Momentum 2: last 2 consecutive agree, z>3.0, 30s hold',
            group='G2', signal_type='momentum', momentum_n=2,
            signal_threshold=3.0, hold_ms=30000,
        ),
        StrategySpec(
            label='momentum_3_z5.0_60s',
            description='Momentum 3: last 3 agree, z>5.0, 60s hold (high conviction)',
            group='G2', signal_type='momentum', momentum_n=3,
            signal_threshold=5.0, hold_ms=60000,
        ),
    ]

    # ── GROUP 3: Multi-Horizon Agreement ──
    strategies['G3'] = [
        StrategySpec(
            label='agree_1s5s_z2.5_30s',
            description='1s+5s agree on direction, z>2.5, 30s hold',
            group='G3', signal_type='agree', agree_horizons=[0, 1],
            signal_threshold=2.5, hold_ms=30000,
        ),
        StrategySpec(
            label='agree_all3_z2.0_30s',
            description='All 3 horizons agree, z>2.0, 30s hold',
            group='G3', signal_type='agree', agree_horizons=[0, 1, 2],
            signal_threshold=2.0, hold_ms=30000,
        ),
        StrategySpec(
            label='agree_all3_z3.0_60s',
            description='All 3 horizons agree, z>3.0, 60s hold',
            group='G3', signal_type='agree', agree_horizons=[0, 1, 2],
            signal_threshold=3.0, hold_ms=60000,
        ),
    ]

    # ── GROUP 4: Signal Strength Combos ──
    strategies['G4'] = [
        StrategySpec(
            label='weighted_z3.0_30s',
            description='Weighted 0.5*z1s+0.3*z5s+0.2*z10s>3.0, 30s hold',
            group='G4', signal_type='weighted', z_weights=(0.5, 0.3, 0.2),
            signal_threshold=3.0, hold_ms=30000,
        ),
        StrategySpec(
            label='max_z4.0_30s',
            description='Max(z_1s,z_5s,z_10s)>4.0, 30s hold',
            group='G4', signal_type='max_z',
            signal_threshold=4.0, hold_ms=30000,
        ),
        StrategySpec(
            label='min_z2.0_30s',
            description='Min(|z_1s|,|z_5s|,|z_10s|)>2.0, 30s hold',
            group='G4', signal_type='min_z',
            signal_threshold=2.0, hold_ms=30000,
        ),
    ]

    # ── GROUP 5: Embedding-Based Gating ──
    strategies['G5'] = [
        StrategySpec(
            label='emb_cluster_z2.5_30s',
            description='KMeans(k=10) cluster gate, z>2.5, 30s hold',
            group='G5', signal_type='emb_gate',
            signal_threshold=2.5, hold_ms=30000,
        ),
    ]

    # ── GROUP 6: Adaptive Hold Time ──
    # Adaptive hold requires running fill_sim for each hold tier separately,
    # then merging. We approximate by splitting signal into 3 tier files.
    strategies['G6'] = [
        StrategySpec(
            label='adaptive_hold_z2.5_tier_10s',
            description='Adaptive hold: |z|<3.0 tier, hold 10s',
            group='G6', signal_type='adaptive_tier',
            signal_threshold=2.5, hold_ms=10000,
        ),
        StrategySpec(
            label='adaptive_hold_z2.5_tier_30s',
            description='Adaptive hold: 3.0<=|z|<5.0 tier, hold 30s',
            group='G6', signal_type='adaptive_tier',
            signal_threshold=3.0, hold_ms=30000,
        ),
        StrategySpec(
            label='adaptive_hold_z2.5_tier_60s',
            description='Adaptive hold: |z|>=5.0 tier, hold 60s',
            group='G6', signal_type='adaptive_tier',
            signal_threshold=5.0, hold_ms=60000,
        ),
        StrategySpec(
            label='adaptive_hold_z3.0_tier_30s',
            description='Adaptive hold (min z3.0): 3.0<=|z|<5.0 tier, hold 30s',
            group='G6', signal_type='adaptive_tier',
            signal_threshold=3.0, hold_ms=30000,
        ),
        StrategySpec(
            label='adaptive_hold_z3.0_tier_60s',
            description='Adaptive hold (min z3.0): |z|>=5.0 tier, hold 60s',
            group='G6', signal_type='adaptive_tier',
            signal_threshold=5.0, hold_ms=60000,
        ),
    ]

    # ── GROUP 7: Bracket + Time Filter Combos ──
    strategies['G7'] = [
        StrategySpec(
            label='midday_bracket_wide_z3.0',
            description='Midday 11-14, SL=4t TP=8t, z>3.0, 60s hold',
            group='G7', signal_type='base_10s',
            signal_threshold=3.0, hold_ms=60000,
            stop_loss_ticks=4, take_profit_ticks=8,
            time_window_start='11:00', time_window_end='14:00',
        ),
        StrategySpec(
            label='open30_bracket_bal_z3.0',
            description='Open 30min, SL=3t TP=4t, z>3.0, 30s hold',
            group='G7', signal_type='base_10s',
            signal_threshold=3.0, hold_ms=30000,
            stop_loss_ticks=3, take_profit_ticks=4,
            time_window_start='09:30', time_window_end='10:00',
        ),
        StrategySpec(
            label='midday_bracket_wide_z5.0',
            description='Midday 11-14, SL=4t TP=8t, z>5.0, 60s hold',
            group='G7', signal_type='base_10s',
            signal_threshold=5.0, hold_ms=60000,
            stop_loss_ticks=4, take_profit_ticks=8,
            time_window_start='11:00', time_window_end='14:00',
        ),
    ]

    return strategies


# ============================================================
# Signal Preparation Pipeline
# ============================================================

def prepare_signals_for_all_strategies(
    strategies: Dict[str, List[StrategySpec]],
) -> Dict[str, Dict[str, Path]]:
    """Prepare prediction NPZ files for each strategy x date combination.

    Returns: {strategy_label: {date_str: pred_npz_path}}
    """
    log.info("\nPreparing prediction signals...")

    # Discover folds
    folds_data = []
    for fp in FOLD_FILES:
        fd = load_fold(fp)
        if fd:
            folds_data.append(fd)
            log.info(f"  Loaded fold: {fd['date_str']} ({fd['n_samples']} samples)")

    if not folds_data:
        log.error("No fold data loaded!")
        return {}

    # Sort by date for walk-forward
    folds_data.sort(key=lambda x: x['date_str'])
    dates = [f['date_str'] for f in folds_data]

    # Load event timestamps for all dates
    timestamps_map = {}
    for fd in folds_data:
        ts = load_event_timestamps(fd['date_str'])
        if ts is not None:
            timestamps_map[fd['date_str']] = ts
        else:
            log.warning(f"  No event timestamps for {fd['date_str']}")

    # Identify unique signal types needed
    signal_types = set()
    for group_strats in strategies.values():
        for s in group_strats:
            signal_types.add(s.signal_type)

    log.info(f"  Signal types needed: {signal_types}")
    log.info(f"  Dates: {dates}")

    # Generate signals per type x date
    # {signal_cache_key: {date_str: Path}}
    signal_cache = {}

    # ── base_10s signals ──
    if 'base_10s' in signal_types:
        log.info("  Generating base_10s signals...")
        running_stats = None
        for fd in folds_data:
            date_str = fd['date_str']
            if date_str not in timestamps_map:
                continue
            cache_key = f'base_10s_{date_str}'
            cache_path = PRED_CACHE_DIR / f'{cache_key}.npz'

            if not cache_path.exists():
                z_signal, running_stats = generate_base_10s_signal(
                    fd, timestamps_map[date_str], date_str, running_stats
                )
                np.savez_compressed(str(cache_path), predictions=z_signal)
                n_nz = int(np.count_nonzero(z_signal))
                log.info(f"    {date_str}: {n_nz} signals")
            else:
                # Advance running stats from cache
                data = np.load(str(cache_path))
                nz = data['predictions'][data['predictions'] != 0]
                if running_stats is None:
                    running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
                running_stats['sum'] += float(np.sum(nz))
                running_stats['sq'] += float(np.sum(nz ** 2))
                running_stats['count'] += len(nz)
                log.info(f"    {date_str}: cached")

            signal_cache.setdefault('base_10s', {})[date_str] = cache_path

    # ── momentum signals ──
    if 'momentum' in signal_types:
        momentum_ns = set()
        for group_strats in strategies.values():
            for s in group_strats:
                if s.signal_type == 'momentum':
                    momentum_ns.add(s.momentum_n)

        for n_consec in momentum_ns:
            log.info(f"  Generating momentum_{n_consec} signals...")
            running_stats = None
            for fd in folds_data:
                date_str = fd['date_str']
                if date_str not in timestamps_map:
                    continue
                cache_key = f'momentum_{n_consec}_{date_str}'
                cache_path = PRED_CACHE_DIR / f'{cache_key}.npz'

                if not cache_path.exists():
                    z_signal, running_stats = generate_momentum_signal(
                        fd, timestamps_map[date_str], date_str, n_consec, running_stats
                    )
                    np.savez_compressed(str(cache_path), predictions=z_signal)
                    n_nz = int(np.count_nonzero(z_signal))
                    log.info(f"    {date_str}: {n_nz} signals (n_consec={n_consec})")
                else:
                    # Advance stats
                    data = np.load(str(cache_path))
                    nz = data['predictions'][data['predictions'] != 0]
                    if running_stats is None:
                        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
                    running_stats['sum'] += float(np.sum(nz))
                    running_stats['sq'] += float(np.sum(nz ** 2))
                    running_stats['count'] += len(nz)
                    log.info(f"    {date_str}: cached")

                signal_cache.setdefault(f'momentum_{n_consec}', {})[date_str] = cache_path

    # ── multi-horizon agreement signals ──
    if 'agree' in signal_types:
        agree_configs = set()
        for group_strats in strategies.values():
            for s in group_strats:
                if s.signal_type == 'agree':
                    agree_configs.add(tuple(s.agree_horizons))

        for horizons in agree_configs:
            h_str = '_'.join(str(h) for h in horizons)
            log.info(f"  Generating agree_{h_str} signals...")
            running_stats_list = None
            for fd in folds_data:
                date_str = fd['date_str']
                if date_str not in timestamps_map:
                    continue
                cache_key = f'agree_{h_str}_{date_str}'
                cache_path = PRED_CACHE_DIR / f'{cache_key}.npz'

                if not cache_path.exists():
                    z_signal, running_stats_list = generate_multi_horizon_agreement_signal(
                        fd, timestamps_map[date_str], date_str, list(horizons), running_stats_list
                    )
                    np.savez_compressed(str(cache_path), predictions=z_signal)
                    n_nz = int(np.count_nonzero(z_signal))
                    log.info(f"    {date_str}: {n_nz} signals (horizons={horizons})")
                else:
                    log.info(f"    {date_str}: cached")
                    # For walk-forward correctness, we still need to advance stats
                    # but for agree signals with multiple horizons that's complex.
                    # In practice the cache should be cleared between runs.
                    if running_stats_list is None:
                        running_stats_list = [None, None, None]

                signal_cache.setdefault(f'agree_{h_str}', {})[date_str] = cache_path

    # ── weighted z signals ──
    if 'weighted' in signal_types:
        weight_configs = set()
        for group_strats in strategies.values():
            for s in group_strats:
                if s.signal_type == 'weighted':
                    weight_configs.add(s.z_weights)

        for weights in weight_configs:
            w_str = f'{weights[0]}_{weights[1]}_{weights[2]}'
            log.info(f"  Generating weighted_{w_str} signals...")
            running_stats_list = None
            for fd in folds_data:
                date_str = fd['date_str']
                if date_str not in timestamps_map:
                    continue
                cache_key = f'weighted_{w_str}_{date_str}'
                cache_path = PRED_CACHE_DIR / f'{cache_key}.npz'

                if not cache_path.exists():
                    z_signal, running_stats_list = generate_weighted_z_signal(
                        fd, timestamps_map[date_str], date_str, weights, running_stats_list
                    )
                    np.savez_compressed(str(cache_path), predictions=z_signal)
                    n_nz = int(np.count_nonzero(z_signal))
                    log.info(f"    {date_str}: {n_nz} signals")
                else:
                    log.info(f"    {date_str}: cached")
                    if running_stats_list is None:
                        running_stats_list = [None, None, None]

                signal_cache.setdefault(f'weighted_{w_str}', {})[date_str] = cache_path

    # ── max_z signals ──
    if 'max_z' in signal_types:
        log.info("  Generating max_z signals...")
        running_stats_list = None
        for fd in folds_data:
            date_str = fd['date_str']
            if date_str not in timestamps_map:
                continue
            cache_key = f'max_z_{date_str}'
            cache_path = PRED_CACHE_DIR / f'{cache_key}.npz'

            if not cache_path.exists():
                z_signal, running_stats_list = generate_max_z_signal(
                    fd, timestamps_map[date_str], date_str, running_stats_list
                )
                np.savez_compressed(str(cache_path), predictions=z_signal)
                n_nz = int(np.count_nonzero(z_signal))
                log.info(f"    {date_str}: {n_nz} signals")
            else:
                log.info(f"    {date_str}: cached")
                if running_stats_list is None:
                    running_stats_list = [None, None, None]

            signal_cache.setdefault('max_z', {})[date_str] = cache_path

    # ── min_z signals ──
    if 'min_z' in signal_types:
        log.info("  Generating min_z signals...")
        running_stats_list = None
        for fd in folds_data:
            date_str = fd['date_str']
            if date_str not in timestamps_map:
                continue
            cache_key = f'min_z_{date_str}'
            cache_path = PRED_CACHE_DIR / f'{cache_key}.npz'

            if not cache_path.exists():
                z_signal, running_stats_list = generate_min_z_signal(
                    fd, timestamps_map[date_str], date_str, running_stats_list
                )
                np.savez_compressed(str(cache_path), predictions=z_signal)
                n_nz = int(np.count_nonzero(z_signal))
                log.info(f"    {date_str}: {n_nz} signals")
            else:
                log.info(f"    {date_str}: cached")
                if running_stats_list is None:
                    running_stats_list = [None, None, None]

            signal_cache.setdefault('min_z', {})[date_str] = cache_path

    # ── emb_gate signals ──
    if 'emb_gate' in signal_types:
        log.info("  Generating embedding-gated signals...")
        running_stats = None
        cluster_profitability = None
        for fd in folds_data:
            date_str = fd['date_str']
            if date_str not in timestamps_map:
                continue
            cache_key = f'emb_gate_{date_str}'
            cache_path = PRED_CACHE_DIR / f'{cache_key}.npz'

            if not cache_path.exists():
                z_signal, running_stats, _, cluster_profitability = generate_embedding_gated_signal(
                    fd, timestamps_map[date_str], date_str,
                    None, cluster_profitability, running_stats
                )
                np.savez_compressed(str(cache_path), predictions=z_signal)
                n_nz = int(np.count_nonzero(z_signal))
                profitable_c = sum(1 for v in (cluster_profitability or {}).values() if v > 0)
                log.info(f"    {date_str}: {n_nz} signals ({profitable_c}/10 profitable clusters)")
            else:
                log.info(f"    {date_str}: cached")
                if running_stats is None:
                    running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

            signal_cache.setdefault('emb_gate', {})[date_str] = cache_path

    # ── adaptive_tier signals ──
    # For adaptive hold, we split the base signal into tiers:
    # tier_10s: only signals with 2.5 <= |z| < 3.0
    # tier_30s: only signals with 3.0 <= |z| < 5.0
    # tier_60s: only signals with |z| >= 5.0
    if 'adaptive_tier' in signal_types:
        log.info("  Generating adaptive hold tier signals...")
        # Use base_10s signals and split by magnitude
        for fd in folds_data:
            date_str = fd['date_str']
            base_path = signal_cache.get('base_10s', {}).get(date_str)
            if base_path is None:
                continue

            base_data = np.load(str(base_path))
            base_signal = base_data['predictions']

            # Tier 10s: 2.5 <= |z| < 3.0
            tier_10s_path = PRED_CACHE_DIR / f'adaptive_tier_10s_{date_str}.npz'
            if not tier_10s_path.exists():
                mask = (np.abs(base_signal) >= 2.5) & (np.abs(base_signal) < 3.0)
                tier_signal = np.where(mask, base_signal, 0.0)
                np.savez_compressed(str(tier_10s_path), predictions=tier_signal)
                log.info(f"    {date_str} tier_10s: {int(np.count_nonzero(tier_signal))} signals")
            signal_cache.setdefault('adaptive_tier_10s', {})[date_str] = tier_10s_path

            # Tier 30s: 3.0 <= |z| < 5.0
            tier_30s_path = PRED_CACHE_DIR / f'adaptive_tier_30s_{date_str}.npz'
            if not tier_30s_path.exists():
                mask = (np.abs(base_signal) >= 3.0) & (np.abs(base_signal) < 5.0)
                tier_signal = np.where(mask, base_signal, 0.0)
                np.savez_compressed(str(tier_30s_path), predictions=tier_signal)
                log.info(f"    {date_str} tier_30s: {int(np.count_nonzero(tier_signal))} signals")
            signal_cache.setdefault('adaptive_tier_30s', {})[date_str] = tier_30s_path

            # Tier 60s: |z| >= 5.0
            tier_60s_path = PRED_CACHE_DIR / f'adaptive_tier_60s_{date_str}.npz'
            if not tier_60s_path.exists():
                mask = np.abs(base_signal) >= 5.0
                tier_signal = np.where(mask, base_signal, 0.0)
                np.savez_compressed(str(tier_60s_path), predictions=tier_signal)
                log.info(f"    {date_str} tier_60s: {int(np.count_nonzero(tier_signal))} signals")
            signal_cache.setdefault('adaptive_tier_60s', {})[date_str] = tier_60s_path

    # Free timestamps
    del timestamps_map
    gc.collect()

    # ── Map strategies to their prediction files ──
    result = {}
    for group_strats in strategies.values():
        for s in group_strats:
            if s.signal_type == 'base_10s':
                result[s.label] = signal_cache.get('base_10s', {})
            elif s.signal_type == 'momentum':
                result[s.label] = signal_cache.get(f'momentum_{s.momentum_n}', {})
            elif s.signal_type == 'agree':
                h_str = '_'.join(str(h) for h in s.agree_horizons)
                result[s.label] = signal_cache.get(f'agree_{h_str}', {})
            elif s.signal_type == 'weighted':
                w_str = f'{s.z_weights[0]}_{s.z_weights[1]}_{s.z_weights[2]}'
                result[s.label] = signal_cache.get(f'weighted_{w_str}', {})
            elif s.signal_type == 'max_z':
                result[s.label] = signal_cache.get('max_z', {})
            elif s.signal_type == 'min_z':
                result[s.label] = signal_cache.get('min_z', {})
            elif s.signal_type == 'emb_gate':
                result[s.label] = signal_cache.get('emb_gate', {})
            elif s.signal_type == 'adaptive_tier':
                # Map adaptive tiers by hold_ms
                if s.hold_ms == 10000:
                    result[s.label] = signal_cache.get('adaptive_tier_10s', {})
                elif s.hold_ms == 30000:
                    result[s.label] = signal_cache.get('adaptive_tier_30s', {})
                elif s.hold_ms == 60000:
                    result[s.label] = signal_cache.get('adaptive_tier_60s', {})

    log.info(f"\n  Signal preparation complete: {len(result)} strategy-signal mappings")
    return result


# ============================================================
# Fill Simulator Interface
# ============================================================

def run_fill_sim(
    date_str: str,
    pred_file: Path,
    strategy: StrategySpec,
    out_dir: Path,
) -> Optional[Dict]:
    """Run Rust fill_sim_cli for a single day + strategy."""
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = out_dir / f'{strategy.label}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
    ] + strategy.to_cli_args()

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            log.debug(f"Sim failed {strategy.label}/{date_str}: {r.stderr[:200]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        log.warning(f"Timeout: {strategy.label}/{date_str}")
        return None
    except Exception as e:
        log.debug(f"Error {strategy.label}/{date_str}: {e}")
        return None


def run_strategy_sweep(
    strategy_signals: Dict[str, Dict[str, Path]],
    strategy_map: Dict[str, StrategySpec],
    workers: int = 8,
) -> Dict[str, Dict[str, Dict]]:
    """Run all strategies across all days in parallel."""
    sim_out = RESULTS_DIR / f'sim_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for label, date_files in sorted(strategy_signals.items()):
        strategy = strategy_map[label]
        for date_str, pred_file in sorted(date_files.items()):
            jobs.append({
                'date': date_str,
                'pred_file': pred_file,
                'strategy': strategy,
            })

    log.info(f"\nRunning {len(jobs)} sim jobs ({workers} workers)")
    log.info(f"  Strategies: {len(strategy_signals)}")

    results = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['strategy'], sim_out,
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            job = futures[future]
            try:
                result = future.result()
                if result:
                    label = job['strategy'].label
                    if label not in results:
                        results[label] = {}
                    results[label][job['date']] = result
            except Exception as e:
                log.debug(f"Job error: {e}")

            if done % 20 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(jobs) - done) / max(rate, 0.01)
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, "
                         f"~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log.info(f"Sweep done: {done} jobs in {elapsed:.1f}s")
    return results


# ============================================================
# Portfolio Strategy (Post-hoc combination)
# ============================================================

def merge_portfolio_results(
    component_results: Dict[str, Dict[str, Dict]],
    portfolio_label: str,
    portfolio_desc: str,
    max_contracts: int = 1,
) -> Dict[str, Dict]:
    """Merge multiple strategy results into a portfolio with position limits.

    Logic: process all trades from all components chronologically.
    If a position is already open, skip new entries from other strategies.
    """
    merged_by_date = {}

    # Get all dates
    all_dates = set()
    for label, date_results in component_results.items():
        all_dates.update(date_results.keys())

    for date_str in sorted(all_dates):
        # Collect all trades from all components for this date
        all_trades = []
        for label, date_results in component_results.items():
            if date_str not in date_results:
                continue
            res = date_results[date_str]
            for trade in res.get('trades', []):
                t = dict(trade)
                t['source_strategy'] = label
                all_trades.append(t)

        if not all_trades:
            merged_by_date[date_str] = {
                'total_pnl_dollars': 0.0,
                'total_trades': 0,
                'total_signals': 0,
                'total_filled': 0,
                'trades': [],
            }
            continue

        # Sort by fill_time_ns (entry time)
        all_trades.sort(key=lambda t: t.get('fill_time_ns', t.get('post_time_ns', 0)))

        # Filter: only allow one position at a time
        accepted_trades = []
        position_exit_ns = 0  # when current position exits

        for trade in all_trades:
            entry_ns = trade.get('fill_time_ns', 0)
            exit_ns = trade.get('exit_time_ns', 0)

            # Can we enter? Only if no current position
            if entry_ns >= position_exit_ns:
                accepted_trades.append(trade)
                position_exit_ns = exit_ns

        total_pnl = sum(t.get('pnl_dollars', 0) for t in accepted_trades)
        total_signals = sum(
            r.get('total_signals', 0)
            for label, dr in component_results.items()
            if date_str in dr
            for r in [dr[date_str]]
        )

        merged_by_date[date_str] = {
            'total_pnl_dollars': total_pnl,
            'total_trades': len(accepted_trades),
            'total_signals': total_signals,
            'total_filled': len(accepted_trades),
            'trades': accepted_trades,
        }

    return merged_by_date


# ============================================================
# Analysis & Reporting
# ============================================================

def aggregate_results(
    results: Dict[str, Dict[str, Dict]],
    strategy_map: Dict[str, StrategySpec] = None,
) -> List[Dict]:
    """Aggregate per-day sim results into per-strategy summaries."""
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
        long_pnls = []
        short_pnls = []

        for date_str, res in sorted(date_results.items()):
            dates.append(date_str)
            day_pnl = res.get('total_pnl_dollars', 0)
            total_pnl += day_pnl
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            daily_pnls.append(day_pnl)

            if 'trades' in res:
                for trade in res['trades']:
                    pnl = trade.get('pnl_dollars', 0)
                    all_trade_pnls.append(pnl)
                    if pnl > 0:
                        total_wins += 1
                    sig = trade.get('signal_strength', trade.get('entry_signal', 0))
                    if sig > 0:
                        long_pnls.append(pnl)
                    else:
                        short_pnls.append(pnl)

        n_days = len(date_results)
        if n_days == 0:
            continue

        win_rate = total_wins / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)
        avg_daily = np.mean(daily_pnls) if daily_pnls else 0

        # Sortino
        if len(daily_pnls) > 1:
            downside = [min(0, x) for x in daily_pnls]
            downside_std = np.std(downside)
            sortino = (avg_daily / max(downside_std, 1e-8)) * np.sqrt(252)
        else:
            sortino = 0.0

        gross_profit = sum(p for p in all_trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_trade_pnls if p < 0))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0

        # Max drawdown
        cum = np.cumsum(daily_pnls) if daily_pnls else np.array([0])
        peak = np.maximum.accumulate(cum)
        max_dd = abs(float((cum - peak).min())) if len(cum) > 0 else 0

        group = ''
        description = label
        if strategy_map and label in strategy_map:
            group = strategy_map[label].group
            description = strategy_map[label].description

        summaries.append({
            'label': label,
            'description': description,
            'group': group,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'trades_per_day': round(total_trades / max(n_days, 1), 1),
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sortino': round(sortino, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_trade_ticks': round(avg_trade_pnl / TICK_VALUE, 3),
            'max_dd': round(max_dd, 2),
            'long_trades': len(long_pnls),
            'long_pnl': round(sum(long_pnls), 2),
            'short_trades': len(short_pnls),
            'short_pnl': round(sum(short_pnls), 2),
            'daily_pnls': {d: round(p, 2) for d, p in zip(dates, daily_pnls)},
        })

    summaries.sort(key=lambda x: x['sortino'], reverse=True)
    return summaries


def print_results_table(summaries: List[Dict], title: str = "Results"):
    """Print formatted results table."""
    log.info(f"\n{'=' * 160}")
    log.info(f" {title}")
    log.info(f"{'=' * 160}")

    if not summaries:
        log.info("  No results.")
        return

    header = (
        f"{'Strategy':<35} "
        f"{'Group':<6} "
        f"{'Total P&L':>10} "
        f"{'Trades':>7} "
        f"{'T/Day':>6} "
        f"{'FillR':>6} "
        f"{'WinR':>6} "
        f"{'AvgTrd':>8} "
        f"{'Sortino':>8} "
        f"{'PF':>5} "
        f"{'MaxDD':>8} "
        f"{'Long$':>8} "
        f"{'Short$':>8}"
    )
    log.info(header)
    log.info("-" * 160)

    for s in summaries:
        pnl_marker = '+' if s['total_pnl'] > 0 else ' '
        line = (
            f"{s['label']:<35} "
            f"{s['group']:<6} "
            f"{pnl_marker}${abs(s['total_pnl']):>8,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['trades_per_day']:>5.1f} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f} "
            f"${s['long_pnl']:>7,.0f} "
            f"${s['short_pnl']:>7,.0f}"
        )
        log.info(line)

    n_days = summaries[0]['n_days'] if summaries else 0
    log.info(f"\n  ES Futures | Tick=$12.50 | Commission=$4.70 RT | {n_days} OOT days")


def print_strategy_detail(summaries: List[Dict], top_n: int = 5):
    """Print detailed analysis of top strategies."""
    profitable = [s for s in summaries if s['total_pnl'] > 0]

    log.info(f"\n{'=' * 80}")
    log.info(f"  PROFITABLE STRATEGIES: {len(profitable)} / {len(summaries)}")
    log.info(f"{'=' * 80}")

    show = profitable if profitable else summaries[:top_n]
    label = "PROFITABLE" if profitable else "TOP BY SORTINO (all negative)"

    for rank, s in enumerate(show[:top_n]):
        log.info(f"\n  #{rank + 1} {s['label']}")
        log.info(f"  {s['description']}")
        log.info(f"  Group:         {s['group']}")
        log.info(f"  Total P&L:     ${s['total_pnl']:,.2f}")
        log.info(f"  Trades:        {s['n_trades']} ({s['trades_per_day']:.1f}/day)")
        log.info(f"  Fill rate:     {s['fill_rate']:.1%}")
        log.info(f"  Win rate:      {s['win_rate']:.1%}")
        log.info(f"  Avg trade:     ${s['avg_trade_pnl']:.2f} ({s['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Sortino:       {s['sortino']:.2f}")
        log.info(f"  Profit factor: {s['profit_factor']:.2f}")
        log.info(f"  Max DD:        ${s['max_dd']:,.2f}")
        log.info(f"  Long:          {s['long_trades']} trades -> ${s['long_pnl']:,.2f}")
        log.info(f"  Short:         {s['short_trades']} trades -> ${s['short_pnl']:,.2f}")
        log.info(f"  Daily P&L:")
        for date, pnl in sorted(s['daily_pnls'].items()):
            marker = '+' if pnl >= 0 else '-'
            log.info(f"    {date}: {marker}${abs(pnl):,.2f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Portfolio Execution v4 — Multi-Strategy Portfolio Tester',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--groups', type=str, default='all',
                        help='Strategy groups: G1,G2,G3,G4,G5,G6,G7 or all')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--clear-cache', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--top-n', type=int, default=10)

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("PORTFOLIO EXECUTION v4 — Multi-Strategy Portfolio Tester")
    log.info("=" * 80)
    log.info(f"  Model:       CNN-Mamba v2 (5 folds, Feb 23-27)")
    log.info(f"  Instrument:  ES (tick=$12.50, commission=$4.70 RT)")
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  Groups:      {args.groups}")
    log.info("=" * 80)

    # Clear cache
    if args.clear_cache:
        import shutil
        if PRED_CACHE_DIR.exists():
            shutil.rmtree(PRED_CACHE_DIR)
            PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("  Cache cleared")

    # Build strategies
    all_strategy_groups = build_all_strategies()

    # Filter groups
    if args.groups == 'all':
        groups_to_run = list(all_strategy_groups.keys())
    else:
        groups_to_run = [g.strip() for g in args.groups.split(',')]

    selected_strategies = {}
    for g in groups_to_run:
        if g in all_strategy_groups:
            selected_strategies[g] = all_strategy_groups[g]
        else:
            log.warning(f"Unknown group: {g}")

    # Flatten strategy map
    strategy_map = {}
    for group_strats in selected_strategies.values():
        for s in group_strats:
            strategy_map[s.label] = s

    total_strats = len(strategy_map)
    log.info(f"\nSelected {total_strats} strategies across {len(selected_strategies)} groups")

    if args.dry_run:
        for group, strats in selected_strategies.items():
            log.info(f"\n  {group}: {len(strats)} strategies")
            for s in strats:
                log.info(f"    {s.label:<35} {s.description}")
                log.info(f"      signal={s.signal_type}, z>{s.signal_threshold}, "
                         f"hold={s.hold_ms}ms, "
                         f"SL={s.stop_loss_ticks}, TP={s.take_profit_ticks}")
        log.info(f"\nDry run complete. {total_strats} strategies.")
        return

    # Check binary
    if not BINARY.exists():
        log.error(f"fill_sim_cli not found: {BINARY}")
        sys.exit(1)

    # ── Phase 1: Prepare signals ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 1: Signal Preparation")
    log.info(f"{'=' * 60}")

    strategy_signals = prepare_signals_for_all_strategies(selected_strategies)

    if not strategy_signals:
        log.error("No signals prepared.")
        sys.exit(1)

    # ── Phase 2: Fill simulation sweep ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 2: Fill Simulation Sweep")
    log.info(f"{'=' * 60}")

    sim_results = run_strategy_sweep(strategy_signals, strategy_map, workers=args.workers)

    # ── Phase 3: Portfolio assembly (G1) ──
    if 'G1' in groups_to_run or args.groups == 'all':
        log.info(f"\n{'=' * 60}")
        log.info(f"  PHASE 3: Portfolio Assembly")
        log.info(f"{'=' * 60}")

        # Define portfolio components using strategies from other groups
        # portfolio_3strat: open30 + midday + bracket_wide
        # We need component strategies that exist in sim_results
        # Use base_10s strategies with appropriate CLI params

        # First, ensure we have the necessary component strategies
        # We create and run them if they don't exist
        portfolio_components = {
            'open30_z2.5_30s': StrategySpec(
                label='_port_open30_z2.5_30s',
                description='Portfolio component: Open30, z>2.5, 30s',
                group='G1', signal_type='base_10s',
                signal_threshold=2.5, hold_ms=30000,
                time_window_start='09:30', time_window_end='10:00',
            ),
            'midday_z2.5_30s': StrategySpec(
                label='_port_midday_z2.5_30s',
                description='Portfolio component: Midday, z>2.5, 30s',
                group='G1', signal_type='base_10s',
                signal_threshold=2.5, hold_ms=30000,
                time_window_start='11:00', time_window_end='14:00',
            ),
            'bracket_wide_z5.0_60s': StrategySpec(
                label='_port_bracket_wide_z5.0_60s',
                description='Portfolio component: Bracket SL4/TP8, z>5.0, 60s',
                group='G1', signal_type='base_10s',
                signal_threshold=5.0, hold_ms=60000,
                stop_loss_ticks=4, take_profit_ticks=8,
            ),
            'bracket_z5.0_60s': StrategySpec(
                label='_port_bracket_z5.0_60s',
                description='Portfolio component: Bracket SL3/TP5, z>5.0, 60s',
                group='G1', signal_type='base_10s',
                signal_threshold=5.0, hold_ms=60000,
                stop_loss_ticks=3, take_profit_ticks=5,
            ),
        }

        # Run component strategies through fill_sim
        component_signals = {}
        component_map = {}
        base_dates = strategy_signals.get(next(
            (k for k in strategy_signals if strategy_signals[k]), ''), {})
        # Find base_10s dates from any existing strategy
        for label, dates in strategy_signals.items():
            if dates:
                base_dates = dates
                break

        for comp_name, comp_spec in portfolio_components.items():
            component_signals[comp_spec.label] = base_dates  # Use base_10s signals
            component_map[comp_spec.label] = comp_spec

        log.info(f"  Running {len(portfolio_components)} portfolio components...")
        comp_results = run_strategy_sweep(component_signals, component_map, workers=args.workers)

        # Assemble portfolios
        # portfolio_3strat: open30 + midday + bracket_wide
        p3_components = {
            k: v for k, v in comp_results.items()
            if k in ['_port_open30_z2.5_30s', '_port_midday_z2.5_30s', '_port_bracket_wide_z5.0_60s']
        }
        if p3_components:
            p3_merged = merge_portfolio_results(
                p3_components, 'portfolio_3strat',
                'Portfolio: open30+midday+bracket_wide (max 1 contract)',
            )
            sim_results['portfolio_3strat'] = p3_merged
            strategy_map['portfolio_3strat'] = StrategySpec(
                label='portfolio_3strat',
                description='Portfolio: open30+midday+bracket_wide (max 1 contract)',
                group='G1', signal_type='portfolio',
            )
            log.info(f"  Assembled portfolio_3strat: {sum(r.get('total_trades', 0) for r in p3_merged.values())} trades")

        # portfolio_2strat: midday + bracket
        p2_components = {
            k: v for k, v in comp_results.items()
            if k in ['_port_midday_z2.5_30s', '_port_bracket_z5.0_60s']
        }
        if p2_components:
            p2_merged = merge_portfolio_results(
                p2_components, 'portfolio_2strat',
                'Portfolio: midday+bracket (max 1 contract)',
            )
            sim_results['portfolio_2strat'] = p2_merged
            strategy_map['portfolio_2strat'] = StrategySpec(
                label='portfolio_2strat',
                description='Portfolio: midday+bracket (max 1 contract)',
                group='G1', signal_type='portfolio',
            )
            log.info(f"  Assembled portfolio_2strat: {sum(r.get('total_trades', 0) for r in p2_merged.values())} trades")

    # ── Phase 4: Aggregate and report ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 4: Analysis & Reporting")
    log.info(f"{'=' * 60}")

    # Also merge adaptive hold tiers into combined results
    adaptive_labels_z25 = ['adaptive_hold_z2.5_tier_10s', 'adaptive_hold_z2.5_tier_30s', 'adaptive_hold_z2.5_tier_60s']
    adaptive_labels_z30 = ['adaptive_hold_z3.0_tier_30s', 'adaptive_hold_z3.0_tier_60s']

    for combined_label, tier_labels, desc in [
        ('adaptive_hold_z2.5', adaptive_labels_z25, 'Adaptive hold: z>=2.5, hold=f(|z|)'),
        ('adaptive_hold_z3.0', adaptive_labels_z30, 'Adaptive hold: z>=3.0, hold=f(|z|)'),
    ]:
        tier_results = {k: v for k, v in sim_results.items() if k in tier_labels}
        if tier_results:
            merged = merge_portfolio_results(tier_results, combined_label, desc)
            sim_results[combined_label] = merged
            strategy_map[combined_label] = StrategySpec(
                label=combined_label, description=desc, group='G6',
                signal_type='adaptive_combined',
            )

    summaries = aggregate_results(sim_results, strategy_map)

    # Print by group
    for group in sorted(set(s['group'] for s in summaries if s['group'])):
        group_summaries = [s for s in summaries if s['group'] == group]
        if group_summaries:
            print_results_table(group_summaries, f"Group {group}")

    # Overall ranking
    print_results_table(summaries, "ALL STRATEGIES — Sorted by Sortino")

    # Top detail
    print_strategy_detail(summaries, top_n=args.top_n)

    # ── Save results ──
    out_file = RESULTS_DIR / f'portfolio_v4_results_{_ts}.json'
    save_data = {
        'timestamp': _ts,
        'instrument': 'ES',
        'tick_value': TICK_VALUE,
        'commission_rt': COMMISSION_RT,
        'n_strategies_tested': len(summaries),
        'n_profitable': len([s for s in summaries if s['total_pnl'] > 0]),
        'strategies': summaries,
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    log.info(f"\n{'=' * 80}")
    log.info(f"  PORTFOLIO EXECUTION v4 COMPLETE")
    log.info(f"{'=' * 80}")
    log.info(f"  Strategies tested:  {len(summaries)}")
    log.info(f"  Profitable:         {len([s for s in summaries if s['total_pnl'] > 0])}")
    if summaries:
        best = summaries[0]
        log.info(f"  Best strategy:      {best['label']} "
                 f"(Sortino={best['sortino']:.2f}, P&L=${best['total_pnl']:,.0f})")
    log.info(f"  Results saved:      {out_file}")
    log.info(f"  Log file:           {_log_file}")
    log.info(f"{'=' * 80}")


if __name__ == '__main__':
    main()
