#!/usr/bin/env python3
"""
Advanced Execution Strategies v2 — CNN-Mamba v2 Multi-Strategy Engine
=====================================================================
Production-grade execution strategy tester for ES futures.

Uses CNN-Mamba v2 predictions (4 OOT folds: Feb 23-26) with 96-dim embeddings.

Strategies implemented:
  1. Signal-Flip Exit: enter on high confidence, exit ONLY on model flip
  2. Multi-Horizon Agreement Gate: all 3 horizons must agree
  3. Confidence-Adaptive Hold: z-score scales max hold time
  4. Embedding MLP Gate: meta-learner on embeddings predicts trade quality
  5. Volatility-Regime Conditional: adapt parameters to vol regime
  6. Spread-Aware Entry: only enter on tight 1-tick spread

Two-phase evaluation:
  Phase 1: Quick label-based P&L screen (all strategies)
  Phase 2: Rust fill_sim_cli market replay (top strategies from Phase 1)

ES Futures: tick=$12.50, commission=$4.70 RT, point_value=$50

Usage:
    python advanced_strategies_v2.py
    python advanced_strategies_v2.py --skip-fillsim
    python advanced_strategies_v2.py --phase1-only
"""

import sys
import os
import gc
import json
import time
import argparse
import subprocess
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
CNNMAMBA_V2_DIR = LVL3_ROOT / 'output' / 'cnn_mamba_v2_smart_v3_mar'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'advanced_v2'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'advanced_v2'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── ES Futures Constants ──
TICK_SIZE_PTS = 0.25
TICK_VALUE = 12.50
POINT_VALUE = 50.00
COMMISSION_RT = 4.70
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.376 ticks
SLIPPAGE_TICKS = 0.5  # average slippage
TOTAL_COST_TICKS = COMMISSION_TICKS + SLIPPAGE_TICKS  # 0.876 ticks

# ── Model Constants ──
WINDOW = 1000
STRIDE = 500

# ── Bar/RTH Constants ──
BARS_PER_SEC = 10
BAR_NS = 100_000_000  # 100ms
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# ── Timestamp ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──
log = logging.getLogger('adv_strategies_v2')
log.setLevel(logging.INFO)
_log_file = str(RESULTS_DIR / f'adv_v2_{_ts}.log')
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ============================================================
# Data Loading
# ============================================================

def load_all_folds() -> List[Dict]:
    """Load all CNN-Mamba v2 fold predictions.

    Returns list of dicts with keys:
        predictions (N,3), labels (N,3), embeddings (N,96),
        date_str, fold_idx, n_samples
    """
    folds = []
    for fold_file in sorted(CNNMAMBA_V2_DIR.glob('fold_*_oot_predictions.npz')):
        data = np.load(str(fold_file), allow_pickle=True)
        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]
        fold_idx = int(fold_file.stem.split('_')[1])

        fold = {
            'predictions': data['predictions'].astype(np.float64),  # (N,3) for 1s/5s/10s
            'labels': data['labels'].astype(np.float64),            # (N,3)
            'embeddings': data['embeddings'].astype(np.float64),    # (N,96)
            'date_str': date_str,
            'fold_idx': fold_idx,
            'n_samples': data['predictions'].shape[0],
            'ic_1s': float(data['ic_1s']),
            'ic_5s': float(data['ic_5s']),
            'ic_10s': float(data['ic_10s']),
            'fold_path': str(fold_file),
        }
        folds.append(fold)
        log.info(f"  Fold {fold_idx}: {date_str}, N={fold['n_samples']}, "
                 f"IC_10s={fold['ic_10s']:.4f}, embeddings={fold['embeddings'].shape}")
    return folds


def compute_expanding_zscore(raw_preds: np.ndarray, min_samples: int = 100) -> np.ndarray:
    """Expanding walk-forward z-score with no lookahead.

    Args:
        raw_preds: (N,) or (N,H) raw predictions
        min_samples: minimum samples before z-scoring (return 0 before that)

    Returns: z-scored predictions, same shape
    """
    if raw_preds.ndim == 1:
        raw_preds = raw_preds.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    N, H = raw_preds.shape
    z = np.zeros_like(raw_preds)

    for h in range(H):
        cumsum = 0.0
        cumsq = 0.0
        count = 0
        for i in range(N):
            v = raw_preds[i, h]
            cumsum += v
            cumsq += v * v
            count += 1
            if count >= min_samples:
                mean = cumsum / count
                var = (cumsq / count) - mean * mean
                std = max(np.sqrt(max(var, 0)), 1e-8)
                z[i, h] = (v - mean) / std

    return z.squeeze() if squeeze else z


def compute_expanding_zscore_carryover(
    raw_preds: np.ndarray,
    running_stats: Optional[Dict] = None,
    min_samples: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """Expanding z-score with carryover state between folds (walk-forward).

    For multi-fold evaluation: fold 0 seeds the stats, fold 1+ carry forward.
    """
    if raw_preds.ndim == 1:
        raw_preds = raw_preds.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    N, H = raw_preds.shape
    z = np.zeros_like(raw_preds)

    if running_stats is None:
        running_stats = {
            'sums': np.zeros(H),
            'sqs': np.zeros(H),
            'counts': np.zeros(H, dtype=np.int64),
        }

    sums = running_stats['sums'].copy()
    sqs = running_stats['sqs'].copy()
    counts = running_stats['counts'].copy()

    for h in range(H):
        for i in range(N):
            v = raw_preds[i, h]
            sums[h] += v
            sqs[h] += v * v
            counts[h] += 1
            if counts[h] >= min_samples:
                mean = sums[h] / counts[h]
                var = (sqs[h] / counts[h]) - mean * mean
                std = max(np.sqrt(max(var, 0)), 1e-8)
                z[i, h] = (v - mean) / std

    new_stats = {
        'sums': sums,
        'sqs': sqs,
        'counts': counts,
    }
    return (z.squeeze() if squeeze else z), new_stats


# ============================================================
# Label-Based P&L Engine (Phase 1 — Quick Screen)
# ============================================================

def label_based_pnl(
    entry_indices: np.ndarray,
    directions: np.ndarray,
    labels: np.ndarray,
    horizon_col: int = 2,
    cost_ticks: float = TOTAL_COST_TICKS,
) -> Dict:
    """Compute theoretical P&L from labels (no fill sim needed).

    Entry at event index i with direction d:
      raw_pnl_ticks = labels[i, horizon_col] * d  (label is in tick-units)
      net_pnl_ticks = raw_pnl_ticks - cost_ticks

    Args:
        entry_indices: (K,) indices into predictions/labels
        directions: (K,) +1 for long, -1 for short
        labels: (N, 3) label array
        horizon_col: which horizon to use for outcome (0=1s, 1=5s, 2=10s)
        cost_ticks: round-trip cost in ticks

    Returns: dict with trade-level and summary statistics
    """
    if len(entry_indices) == 0:
        return {
            'n_trades': 0, 'total_pnl_ticks': 0, 'total_pnl_dollars': 0,
            'avg_pnl_ticks': 0, 'win_rate': 0, 'profit_factor': 0,
            'avg_winner_ticks': 0, 'avg_loser_ticks': 0,
            'trades': [],
        }

    outcomes = labels[entry_indices, horizon_col]
    raw_pnl = outcomes * directions  # positive = correct direction
    net_pnl = raw_pnl - cost_ticks

    winners = net_pnl[net_pnl > 0]
    losers = net_pnl[net_pnl <= 0]

    gross_profit = float(winners.sum()) if len(winners) > 0 else 0
    gross_loss = abs(float(losers.sum())) if len(losers) > 0 else 0.001

    # Per-trade records
    trades = []
    for k in range(len(entry_indices)):
        trades.append({
            'idx': int(entry_indices[k]),
            'direction': int(directions[k]),
            'raw_pnl_ticks': float(raw_pnl[k]),
            'net_pnl_ticks': float(net_pnl[k]),
            'net_pnl_dollars': float(net_pnl[k] * TICK_VALUE),
        })

    # Long/short breakdown
    long_mask = directions > 0
    short_mask = directions < 0
    long_pnl = float(net_pnl[long_mask].sum()) if long_mask.any() else 0
    short_pnl = float(net_pnl[short_mask].sum()) if short_mask.any() else 0
    long_wr = float((net_pnl[long_mask] > 0).mean()) if long_mask.any() else 0
    short_wr = float((net_pnl[short_mask] > 0).mean()) if short_mask.any() else 0

    return {
        'n_trades': len(entry_indices),
        'total_pnl_ticks': float(net_pnl.sum()),
        'total_pnl_dollars': float(net_pnl.sum() * TICK_VALUE),
        'avg_pnl_ticks': float(net_pnl.mean()),
        'avg_pnl_dollars': float(net_pnl.mean() * TICK_VALUE),
        'win_rate': float((net_pnl > 0).mean()),
        'profit_factor': round(gross_profit / gross_loss, 3),
        'avg_winner_ticks': float(winners.mean()) if len(winners) > 0 else 0,
        'avg_loser_ticks': float(losers.mean()) if len(losers) > 0 else 0,
        'max_winner_ticks': float(winners.max()) if len(winners) > 0 else 0,
        'max_loser_ticks': float(losers.min()) if len(losers) > 0 else 0,
        'long_n': int(long_mask.sum()),
        'long_pnl_ticks': long_pnl,
        'long_win_rate': long_wr,
        'short_n': int(short_mask.sum()),
        'short_pnl_ticks': short_pnl,
        'short_win_rate': short_wr,
        'trades': trades,
    }


def multi_horizon_label_pnl(
    entry_indices: np.ndarray,
    directions: np.ndarray,
    labels: np.ndarray,
    cost_ticks: float = TOTAL_COST_TICKS,
) -> Dict:
    """Compute P&L at all 3 horizons (1s, 5s, 10s)."""
    results = {}
    for h, name in enumerate(['1s', '5s', '10s']):
        results[name] = label_based_pnl(
            entry_indices, directions, labels,
            horizon_col=h, cost_ticks=cost_ticks,
        )
    return results


# ============================================================
# Strategy 1: Signal-Flip Exit
# ============================================================

def strategy_signal_flip(
    z_preds: np.ndarray,
    labels: np.ndarray,
    threshold: float = 2.0,
    flip_mode: str = 'immediate',
    flip_threshold: float = 0.0,
    flip_consecutive: int = 1,
    max_hold_samples: int = 100,
) -> Dict:
    """Signal-Flip Exit Strategy.

    Enter on high-confidence signal. Exit ONLY when model flips direction.

    Variants:
        immediate: exit on any opposite-sign prediction
        conditional: exit only if opposing signal > flip_threshold
        slow: exit after N consecutive opposing signals

    Uses labels to compute outcome at the actual exit point.

    Args:
        z_preds: (N,3) z-scored predictions
        labels: (N,3) labels (tick units)
        threshold: entry z-score threshold (on 10s horizon)
        flip_mode: 'immediate', 'conditional', 'slow'
        flip_threshold: z-score threshold for conditional flip
        flip_consecutive: N for slow flip
        max_hold_samples: maximum hold before forced exit (in samples)

    Returns: dict with trades and summary stats
    """
    N = len(z_preds)
    z_10s = z_preds[:, 2]  # 10s horizon for entry conviction

    trades = []
    i = 0
    while i < N - 1:
        z = z_10s[i]
        if abs(z) < threshold:
            i += 1
            continue

        # Entry
        direction = 1 if z > 0 else -1
        entry_idx = i
        exit_idx = i + 1
        consec_opposing = 0

        # Scan forward for exit
        for j in range(i + 1, min(i + max_hold_samples, N)):
            z_j = z_10s[j]
            opposing = (z_j * direction) < 0  # signal flipped

            if flip_mode == 'immediate':
                if opposing:
                    exit_idx = j
                    break
            elif flip_mode == 'conditional':
                if opposing and abs(z_j) >= flip_threshold:
                    exit_idx = j
                    break
            elif flip_mode == 'slow':
                if opposing:
                    consec_opposing += 1
                    if consec_opposing >= flip_consecutive:
                        exit_idx = j
                        break
                else:
                    consec_opposing = 0
            exit_idx = j
        else:
            exit_idx = min(i + max_hold_samples, N - 1)

        # Compute P&L from label difference between entry and exit
        # Each label represents price change from that point
        # So: entry at i, hold H samples -> use label at horizon that matches
        hold_samples = exit_idx - entry_idx

        # Map hold_samples to the best horizon (1s~10, 5s~50, 10s~100 samples approx)
        # Since stride=500 events/sample: 1 sample ~ 5s of events, very approximate
        # Use 10s label as the standard outcome for this hold period
        # But for variable hold: compute cumulative label if possible
        # Simplified: use the 10s label at entry as the baseline outcome
        if hold_samples <= 2:
            hz = 0  # 1s
        elif hold_samples <= 10:
            hz = 1  # 5s
        else:
            hz = 2  # 10s

        raw_pnl_ticks = float(labels[entry_idx, hz] * direction)
        net_pnl_ticks = raw_pnl_ticks - TOTAL_COST_TICKS

        trades.append({
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'direction': direction,
            'entry_z': float(z),
            'hold_samples': hold_samples,
            'horizon_used': ['1s', '5s', '10s'][hz],
            'raw_pnl_ticks': raw_pnl_ticks,
            'net_pnl_ticks': net_pnl_ticks,
            'net_pnl_dollars': net_pnl_ticks * TICK_VALUE,
        })

        # Skip ahead past exit
        i = exit_idx + 1

    # Summarize
    if not trades:
        return {'strategy': 'signal_flip', 'flip_mode': flip_mode,
                'threshold': threshold, 'n_trades': 0,
                'total_pnl_ticks': 0, 'total_pnl_dollars': 0, 'trades': []}

    pnls = np.array([t['net_pnl_ticks'] for t in trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    gp = float(winners.sum()) if len(winners) > 0 else 0
    gl = abs(float(losers.sum())) if len(losers) > 0 else 0.001

    long_trades = [t for t in trades if t['direction'] > 0]
    short_trades = [t for t in trades if t['direction'] < 0]

    return {
        'strategy': 'signal_flip',
        'flip_mode': flip_mode,
        'threshold': threshold,
        'flip_threshold': flip_threshold,
        'flip_consecutive': flip_consecutive,
        'n_trades': len(trades),
        'total_pnl_ticks': float(pnls.sum()),
        'total_pnl_dollars': float(pnls.sum() * TICK_VALUE),
        'avg_pnl_ticks': float(pnls.mean()),
        'win_rate': float((pnls > 0).mean()),
        'profit_factor': round(gp / gl, 3),
        'avg_hold_samples': float(np.mean([t['hold_samples'] for t in trades])),
        'long_n': len(long_trades),
        'long_pnl': sum(t['net_pnl_ticks'] for t in long_trades),
        'short_n': len(short_trades),
        'short_pnl': sum(t['net_pnl_ticks'] for t in short_trades),
        'trades': trades,
    }


# ============================================================
# Strategy 2: Multi-Horizon Agreement Gate
# ============================================================

def strategy_horizon_agreement(
    z_preds: np.ndarray,
    labels: np.ndarray,
    threshold: float = 2.0,
    require_all: bool = True,
    weight_1s: float = 0.3,
    weight_5s: float = 0.3,
    weight_10s: float = 0.4,
) -> Dict:
    """Multi-Horizon Agreement Gate.

    Only enter when ALL 3 horizons agree on direction AND exceed threshold.
    Weight: 1s for timing, 10s for conviction.

    Args:
        z_preds: (N,3) z-scored predictions [1s, 5s, 10s]
        labels: (N,3) labels
        threshold: minimum |z| on EACH horizon
        require_all: True = all 3 must agree, False = 2 of 3
    """
    N = len(z_preds)
    z_1s, z_5s, z_10s = z_preds[:, 0], z_preds[:, 1], z_preds[:, 2]

    signs = np.sign(z_preds)  # (N, 3)
    magnitudes = np.abs(z_preds)  # (N, 3)

    if require_all:
        # All 3 agree on direction
        agreement = (signs[:, 0] == signs[:, 1]) & (signs[:, 1] == signs[:, 2])
        # All 3 exceed threshold
        above_thresh = (magnitudes[:, 0] >= threshold) & \
                       (magnitudes[:, 1] >= threshold) & \
                       (magnitudes[:, 2] >= threshold)
        entry_mask = agreement & above_thresh & (signs[:, 0] != 0)
    else:
        # 2 of 3 agree (majority vote)
        sum_signs = signs.sum(axis=1)
        agreement = np.abs(sum_signs) >= 2
        # Average magnitude > threshold
        avg_mag = magnitudes.mean(axis=1)
        above_thresh = avg_mag >= threshold
        entry_mask = agreement & above_thresh

    entry_indices = np.where(entry_mask)[0]
    if len(entry_indices) == 0:
        return {'strategy': 'horizon_agreement', 'threshold': threshold,
                'require_all': require_all, 'n_trades': 0,
                'total_pnl_ticks': 0, 'total_pnl_dollars': 0}

    # Direction from weighted combination
    weighted_z = (z_1s * weight_1s + z_5s * weight_5s + z_10s * weight_10s)
    directions = np.sign(weighted_z[entry_indices])

    # Filter out zero directions
    valid = directions != 0
    entry_indices = entry_indices[valid]
    directions = directions[valid]

    result = multi_horizon_label_pnl(entry_indices, directions, labels)

    return {
        'strategy': 'horizon_agreement',
        'threshold': threshold,
        'require_all': require_all,
        'n_signals': int(entry_mask.sum()),
        **{f'{hz}_pnl_ticks': result[hz]['total_pnl_ticks'] for hz in ['1s', '5s', '10s']},
        **{f'{hz}_win_rate': result[hz]['win_rate'] for hz in ['1s', '5s', '10s']},
        **{f'{hz}_avg_pnl': result[hz]['avg_pnl_ticks'] for hz in ['1s', '5s', '10s']},
        'n_trades': result['10s']['n_trades'],
        'total_pnl_ticks': result['10s']['total_pnl_ticks'],
        'total_pnl_dollars': result['10s']['total_pnl_dollars'],
        'avg_pnl_ticks': result['10s']['avg_pnl_ticks'],
        'win_rate': result['10s']['win_rate'],
        'profit_factor': result['10s']['profit_factor'],
        'long_n': result['10s']['long_n'],
        'long_pnl': result['10s']['long_pnl_ticks'],
        'short_n': result['10s']['short_n'],
        'short_pnl': result['10s']['short_pnl_ticks'],
    }


# ============================================================
# Strategy 3: Confidence-Adaptive Hold
# ============================================================

def strategy_adaptive_hold(
    z_preds: np.ndarray,
    labels: np.ndarray,
    min_z: float = 2.0,
) -> Dict:
    """Confidence-Adaptive Hold.

    Higher confidence = use longer horizon label.
    z=2.0 -> 1s label, z=3.0 -> 5s label, z>=4.0 -> 10s label
    Exit early if signal flips.

    This naturally allocates more risk to higher-confidence trades.
    """
    N = len(z_preds)
    z_10s = z_preds[:, 2]

    trades = []
    i = 0
    while i < N - 1:
        z = z_10s[i]
        if abs(z) < min_z:
            i += 1
            continue

        direction = 1 if z > 0 else -1
        abs_z = abs(z)

        # Map confidence to horizon
        if abs_z >= 4.0:
            hz = 2  # 10s
            hz_name = '10s'
        elif abs_z >= 3.0:
            hz = 1  # 5s
            hz_name = '5s'
        else:
            hz = 0  # 1s
            hz_name = '1s'

        # Check signal flip (early exit)
        # Look 1-2 samples ahead for opposing signal
        exit_early = False
        if i + 1 < N:
            z_next = z_10s[i + 1]
            if z_next * direction < 0 and abs(z_next) > 1.0:
                # Signal flipped next sample, use 1s horizon instead
                hz = 0
                hz_name = '1s_early'
                exit_early = True

        raw_pnl = float(labels[i, hz] * direction)
        net_pnl = raw_pnl - TOTAL_COST_TICKS

        trades.append({
            'idx': i,
            'direction': direction,
            'z': float(z),
            'horizon': hz_name,
            'exit_early': exit_early,
            'raw_pnl_ticks': raw_pnl,
            'net_pnl_ticks': net_pnl,
        })

        # Skip based on hold horizon
        skip = max(1, [2, 10, 20][hz])
        i += skip

    if not trades:
        return {'strategy': 'adaptive_hold', 'n_trades': 0,
                'total_pnl_ticks': 0, 'total_pnl_dollars': 0}

    pnls = np.array([t['net_pnl_ticks'] for t in trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    gp = float(winners.sum()) if len(winners) > 0 else 0
    gl = abs(float(losers.sum())) if len(losers) > 0 else 0.001

    # Breakdown by horizon used
    hz_breakdown = {}
    for hz_name in ['1s', '5s', '10s', '1s_early']:
        hz_trades = [t for t in trades if t['horizon'] == hz_name]
        if hz_trades:
            hz_pnls = np.array([t['net_pnl_ticks'] for t in hz_trades])
            hz_breakdown[hz_name] = {
                'n': len(hz_trades),
                'avg_pnl': float(hz_pnls.mean()),
                'total_pnl': float(hz_pnls.sum()),
                'win_rate': float((hz_pnls > 0).mean()),
            }

    return {
        'strategy': 'adaptive_hold',
        'min_z': min_z,
        'n_trades': len(trades),
        'total_pnl_ticks': float(pnls.sum()),
        'total_pnl_dollars': float(pnls.sum() * TICK_VALUE),
        'avg_pnl_ticks': float(pnls.mean()),
        'win_rate': float((pnls > 0).mean()),
        'profit_factor': round(gp / gl, 3),
        'horizon_breakdown': hz_breakdown,
        'long_n': sum(1 for t in trades if t['direction'] > 0),
        'long_pnl': sum(t['net_pnl_ticks'] for t in trades if t['direction'] > 0),
        'short_n': sum(1 for t in trades if t['direction'] < 0),
        'short_pnl': sum(t['net_pnl_ticks'] for t in trades if t['direction'] < 0),
    }


# ============================================================
# Strategy 4: Embedding MLP Gate (Meta-Learner)
# ============================================================

def strategy_embedding_mlp_gate(
    folds: List[Dict],
    z_threshold: float = 2.0,
    mlp_threshold: float = 0.5,
    hidden_dim: int = 32,
    n_epochs: int = 50,
    lr: float = 0.001,
) -> Dict:
    """Embedding-Based MLP Gate.

    Train a tiny MLP on CNN-Mamba v2's 96-dim embeddings to predict
    "will this trade be profitable?"

    Walk-forward: train on fold N, predict on fold N+1.
    If MLP says "bad trade" -> skip. If "good trade" -> enter.

    This is a meta-learner that learns WHEN the model is trustworthy.

    Uses simple numpy-based logistic regression + 1 hidden layer
    (no torch dependency for inference speed).
    """
    log.info("\n  Training Embedding MLP Gate (walk-forward)...")

    # We need at least 2 folds for walk-forward
    if len(folds) < 2:
        return {'strategy': 'embedding_mlp', 'n_trades': 0,
                'total_pnl_ticks': 0, 'error': 'need >= 2 folds'}

    all_trades = []
    mlp_stats = []

    for test_fold_idx in range(1, len(folds)):
        train_fold = folds[test_fold_idx - 1]
        test_fold = folds[test_fold_idx]

        # Prepare training data
        train_emb = train_fold['embeddings']  # (N_train, 96)
        train_preds = train_fold['predictions']
        train_labels = train_fold['labels']

        # Z-score the predictions for this fold
        train_z = compute_expanding_zscore(train_preds, min_samples=50)

        # Identify trade candidates in training fold
        z_10s = train_z[:, 2] if train_z.ndim == 2 else train_z
        trade_mask = np.abs(z_10s) >= z_threshold
        trade_indices = np.where(trade_mask)[0]

        if len(trade_indices) < 20:
            log.info(f"    Fold {test_fold_idx}: too few training trades ({len(trade_indices)}), skip MLP")
            continue

        # Create labels: 1 if trade was profitable at 10s horizon
        train_directions = np.sign(z_10s[trade_indices])
        trade_outcomes = train_labels[trade_indices, 2] * train_directions
        trade_profitable = (trade_outcomes > TOTAL_COST_TICKS).astype(np.float64)

        # Training features: embeddings of trade candidates
        X_train = train_emb[trade_indices]
        y_train = trade_profitable

        # Normalize features
        X_mean = X_train.mean(axis=0)
        X_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - X_mean) / X_std

        # Train simple logistic regression with 1 hidden layer via numpy
        # Architecture: 96 -> hidden_dim -> 1
        np.random.seed(42 + test_fold_idx)
        W1 = np.random.randn(96, hidden_dim) * 0.1
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, 1) * 0.1
        b2 = np.zeros(1)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

        def relu(x):
            return np.maximum(0, x)

        def relu_deriv(x):
            return (x > 0).astype(np.float64)

        best_loss = float('inf')
        for epoch in range(n_epochs):
            # Forward
            h = relu(X_train_norm @ W1 + b1)  # (N, hidden)
            logits = (h @ W2 + b2).squeeze()   # (N,)
            probs = sigmoid(logits)

            # Binary cross-entropy loss
            eps = 1e-7
            loss = -np.mean(y_train * np.log(probs + eps) +
                           (1 - y_train) * np.log(1 - probs + eps))

            # Backward
            dlogits = (probs - y_train) / len(y_train)  # (N,)
            dW2 = h.T @ dlogits.reshape(-1, 1)  # (hidden, 1)
            db2 = dlogits.sum().reshape(1)
            dh = dlogits.reshape(-1, 1) @ W2.T  # (N, hidden)
            dh = dh * relu_deriv(X_train_norm @ W1 + b1)
            dW1 = X_train_norm.T @ dh  # (96, hidden)
            db1 = dh.sum(axis=0)

            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

            if loss < best_loss:
                best_loss = loss

        # Evaluate on training data
        h_train = relu(X_train_norm @ W1 + b1)
        train_probs = sigmoid((h_train @ W2 + b2).squeeze())
        train_acc = float(((train_probs > 0.5) == y_train).mean())
        train_pos_rate = float(y_train.mean())

        # Now predict on test fold
        test_emb = test_fold['embeddings']
        test_preds = test_fold['predictions']
        test_labels = test_fold['labels']
        test_z = compute_expanding_zscore(test_preds, min_samples=50)

        z_10s_test = test_z[:, 2] if test_z.ndim == 2 else test_z
        test_trade_mask = np.abs(z_10s_test) >= z_threshold
        test_trade_indices = np.where(test_trade_mask)[0]

        if len(test_trade_indices) == 0:
            continue

        X_test = test_emb[test_trade_indices]
        X_test_norm = (X_test - X_mean) / X_std

        h_test = relu(X_test_norm @ W1 + b1)
        test_probs = sigmoid((h_test @ W2 + b2).squeeze())

        # MLP gate: only take trades where MLP says "good" (prob > mlp_threshold)
        mlp_pass = test_probs > mlp_threshold
        gated_indices = test_trade_indices[mlp_pass]
        gated_directions = np.sign(z_10s_test[gated_indices])

        # Also compute unfiltered for comparison
        all_directions = np.sign(z_10s_test[test_trade_indices])
        all_outcomes = test_labels[test_trade_indices, 2] * all_directions - TOTAL_COST_TICKS
        gated_outcomes = test_labels[gated_indices, 2] * gated_directions - TOTAL_COST_TICKS

        fold_stats = {
            'test_fold': test_fold_idx,
            'date': test_fold['date_str'],
            'train_acc': round(train_acc, 4),
            'train_pos_rate': round(train_pos_rate, 4),
            'train_loss': round(best_loss, 4),
            'n_candidates': len(test_trade_indices),
            'n_gated': len(gated_indices),
            'filter_rate': round(1 - len(gated_indices) / max(len(test_trade_indices), 1), 4),
            'unfiltered_pnl': float(all_outcomes.sum()),
            'gated_pnl': float(gated_outcomes.sum()) if len(gated_outcomes) > 0 else 0,
            'unfiltered_wr': float((all_outcomes > 0).mean()),
            'gated_wr': float((gated_outcomes > 0).mean()) if len(gated_outcomes) > 0 else 0,
        }
        mlp_stats.append(fold_stats)
        log.info(f"    Fold {test_fold_idx} ({test_fold['date_str']}): "
                 f"{fold_stats['n_candidates']} candidates -> {fold_stats['n_gated']} gated "
                 f"(filter {fold_stats['filter_rate']:.1%}), "
                 f"unfiltered={fold_stats['unfiltered_pnl']:.1f}t, "
                 f"gated={fold_stats['gated_pnl']:.1f}t")

        # Collect trades for overall stats
        for k, idx in enumerate(gated_indices):
            d = int(gated_directions[k]) if np.isscalar(gated_directions) == False else int(gated_directions)
            pnl = float(gated_outcomes[k]) if len(gated_outcomes) > 0 else 0
            all_trades.append({
                'fold': test_fold_idx,
                'date': test_fold['date_str'],
                'idx': int(idx),
                'direction': d,
                'net_pnl_ticks': pnl,
                'mlp_prob': float(test_probs[mlp_pass][k]) if mlp_pass.any() else 0,
            })

    # Aggregate
    if not all_trades:
        return {'strategy': 'embedding_mlp', 'n_trades': 0,
                'total_pnl_ticks': 0, 'total_pnl_dollars': 0,
                'mlp_stats': mlp_stats}

    pnls = np.array([t['net_pnl_ticks'] for t in all_trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    gp = float(winners.sum()) if len(winners) > 0 else 0
    gl = abs(float(losers.sum())) if len(losers) > 0 else 0.001

    return {
        'strategy': 'embedding_mlp',
        'z_threshold': z_threshold,
        'mlp_threshold': mlp_threshold,
        'hidden_dim': hidden_dim,
        'n_trades': len(all_trades),
        'total_pnl_ticks': float(pnls.sum()),
        'total_pnl_dollars': float(pnls.sum() * TICK_VALUE),
        'avg_pnl_ticks': float(pnls.mean()),
        'win_rate': float((pnls > 0).mean()),
        'profit_factor': round(gp / gl, 3),
        'mlp_stats': mlp_stats,
        'long_n': sum(1 for t in all_trades if t['direction'] > 0),
        'long_pnl': sum(t['net_pnl_ticks'] for t in all_trades if t['direction'] > 0),
        'short_n': sum(1 for t in all_trades if t['direction'] < 0),
        'short_pnl': sum(t['net_pnl_ticks'] for t in all_trades if t['direction'] < 0),
    }


# ============================================================
# Strategy 5: Volatility-Regime Conditional
# ============================================================

def strategy_vol_regime(
    z_preds: np.ndarray,
    labels: np.ndarray,
    raw_preds: np.ndarray,
    vol_lookback: int = 100,
) -> Dict:
    """Volatility-Regime Conditional Strategy.

    Compute realized volatility from recent labels.
    High vol: tighter threshold (z>=3), use 1s horizon (shorter hold)
    Low vol: looser threshold (z>=1.5), use 10s horizon (longer hold)
    Medium: standard (z>=2), use 5s horizon

    Also includes time-of-day conditioning:
    First 20% of samples (open): higher threshold
    Last 20% of samples (close): higher threshold
    Middle 60% (midday): standard
    """
    N = len(z_preds)
    z_10s = z_preds[:, 2]
    label_10s = labels[:, 2]

    # Compute rolling realized vol from labels
    rolling_vol = np.zeros(N)
    for i in range(vol_lookback, N):
        window = label_10s[i - vol_lookback:i]
        rolling_vol[i] = np.std(window)

    # Vol regime boundaries (percentile-based from the data)
    vol_nonzero = rolling_vol[rolling_vol > 0]
    if len(vol_nonzero) < 10:
        return {'strategy': 'vol_regime', 'n_trades': 0,
                'total_pnl_ticks': 0, 'error': 'insufficient vol data'}

    vol_p33 = np.percentile(vol_nonzero, 33)
    vol_p67 = np.percentile(vol_nonzero, 67)

    # Time-of-day conditioning (by sample index)
    open_end = int(N * 0.15)
    close_start = int(N * 0.85)

    trades = []
    i = vol_lookback
    while i < N:
        z = z_10s[i]
        vol = rolling_vol[i]

        if vol == 0:
            i += 1
            continue

        # Determine regime
        if vol < vol_p33:
            regime = 'low_vol'
            threshold = 1.5
            hz = 2  # 10s (longer hold in calm)
        elif vol > vol_p67:
            regime = 'high_vol'
            threshold = 3.0
            hz = 0  # 1s (shorter hold in vol)
        else:
            regime = 'medium_vol'
            threshold = 2.0
            hz = 1  # 5s

        # Time-of-day adjustment
        if i < open_end or i >= close_start:
            threshold += 0.5  # More selective at open/close
            tod = 'open' if i < open_end else 'close'
        else:
            tod = 'midday'

        if abs(z) < threshold:
            i += 1
            continue

        direction = 1 if z > 0 else -1
        raw_pnl = float(labels[i, hz] * direction)
        net_pnl = raw_pnl - TOTAL_COST_TICKS

        trades.append({
            'idx': i,
            'direction': direction,
            'z': float(z),
            'regime': regime,
            'time_of_day': tod,
            'vol': float(vol),
            'threshold_used': threshold,
            'horizon': ['1s', '5s', '10s'][hz],
            'raw_pnl_ticks': raw_pnl,
            'net_pnl_ticks': net_pnl,
        })

        # Skip based on horizon
        skip = max(1, [2, 10, 20][hz])
        i += skip

    if not trades:
        return {'strategy': 'vol_regime', 'n_trades': 0,
                'total_pnl_ticks': 0, 'total_pnl_dollars': 0}

    pnls = np.array([t['net_pnl_ticks'] for t in trades])
    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    gp = float(winners.sum()) if len(winners) > 0 else 0
    gl = abs(float(losers.sum())) if len(losers) > 0 else 0.001

    # Regime breakdown
    regime_stats = {}
    for regime in ['low_vol', 'medium_vol', 'high_vol']:
        rt = [t for t in trades if t['regime'] == regime]
        if rt:
            rp = np.array([t['net_pnl_ticks'] for t in rt])
            regime_stats[regime] = {
                'n': len(rt),
                'avg_pnl': float(rp.mean()),
                'total_pnl': float(rp.sum()),
                'win_rate': float((rp > 0).mean()),
            }

    # TOD breakdown
    tod_stats = {}
    for tod in ['open', 'midday', 'close']:
        tt = [t for t in trades if t['time_of_day'] == tod]
        if tt:
            tp = np.array([t['net_pnl_ticks'] for t in tt])
            tod_stats[tod] = {
                'n': len(tt),
                'avg_pnl': float(tp.mean()),
                'total_pnl': float(tp.sum()),
                'win_rate': float((tp > 0).mean()),
            }

    return {
        'strategy': 'vol_regime',
        'n_trades': len(trades),
        'total_pnl_ticks': float(pnls.sum()),
        'total_pnl_dollars': float(pnls.sum() * TICK_VALUE),
        'avg_pnl_ticks': float(pnls.mean()),
        'win_rate': float((pnls > 0).mean()),
        'profit_factor': round(gp / gl, 3),
        'vol_p33': float(vol_p33),
        'vol_p67': float(vol_p67),
        'regime_breakdown': regime_stats,
        'tod_breakdown': tod_stats,
        'long_n': sum(1 for t in trades if t['direction'] > 0),
        'long_pnl': sum(t['net_pnl_ticks'] for t in trades if t['direction'] > 0),
        'short_n': sum(1 for t in trades if t['direction'] < 0),
        'short_pnl': sum(t['net_pnl_ticks'] for t in trades if t['direction'] < 0),
    }


# ============================================================
# Strategy 6: Confidence Tier Analysis (Baseline)
# ============================================================

def strategy_confidence_tiers(
    z_preds: np.ndarray,
    labels: np.ndarray,
) -> Dict:
    """Analyze performance at different confidence tiers.

    Reports at: All signals, Top 50%, 25%, 10%, 5%, 1%, 0.5%
    For each tier: P&L at 1s/5s/10s, win rate, avg pnl, long vs short.
    """
    z_10s = z_preds[:, 2]
    abs_z = np.abs(z_10s)
    nonzero_mask = abs_z > 0
    abs_z_nz = abs_z[nonzero_mask]

    if len(abs_z_nz) == 0:
        return {'strategy': 'confidence_tiers', 'tiers': []}

    tiers = []
    tier_specs = [
        ('Top100%', 0), ('Top50%', 50), ('Top25%', 75),
        ('Top10%', 90), ('Top5%', 95), ('Top1%', 99), ('Top0.5%', 99.5),
    ]

    for name, pct in tier_specs:
        if pct > 0:
            thresh = np.percentile(abs_z_nz, pct)
        else:
            thresh = 0

        mask = abs_z >= max(thresh, 0.001)
        indices = np.where(mask)[0]
        directions = np.sign(z_10s[indices])
        valid = directions != 0
        indices = indices[valid]
        directions = directions[valid]

        if len(indices) == 0:
            continue

        result = multi_horizon_label_pnl(indices, directions, labels)

        tier = {
            'tier': name,
            'threshold': float(thresh),
            'n_trades': len(indices),
        }
        for hz in ['1s', '5s', '10s']:
            tier[f'{hz}_total_pnl'] = result[hz]['total_pnl_ticks']
            tier[f'{hz}_avg_pnl'] = result[hz]['avg_pnl_ticks']
            tier[f'{hz}_win_rate'] = result[hz]['win_rate']
            tier[f'{hz}_pf'] = result[hz]['profit_factor']
            tier[f'{hz}_long_n'] = result[hz]['long_n']
            tier[f'{hz}_long_pnl'] = result[hz]['long_pnl_ticks']
            tier[f'{hz}_short_n'] = result[hz]['short_n']
            tier[f'{hz}_short_pnl'] = result[hz]['short_pnl_ticks']

        tiers.append(tier)

    return {'strategy': 'confidence_tiers', 'tiers': tiers}


# ============================================================
# RTH Utilities (for fill_sim bar signal generation)
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


def predictions_to_bar_signal(
    signal: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Convert per-window predictions to bar-indexed z-scored signal for fill_sim_cli.

    Maps predictions (stride=500, window=1000) to 100ms bar indices,
    then applies expanding walk-forward z-score.
    """
    n_events = len(event_timestamps)
    n_preds = len(signal)

    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1

    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    elif n_preds > len(label_idxs):
        signal = signal[:len(label_idxs)]
        n_preds = len(signal)

    if n_preds == 0:
        return np.zeros(N_RTH_BARS, dtype=np.float64), running_stats or {}

    pred_timestamps = event_timestamps[label_idxs]
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)

    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, sig in zip(bar_indices[rth_mask], signal[rth_mask]):
        bar_preds[bi] = sig

    # Expanding z-score
    if running_stats is None:
        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

    zscore = np.zeros(N_RTH_BARS, dtype=np.float64)
    rs, rsq, cnt = running_stats['sum'], running_stats['sq'], running_stats['count']

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

    new_stats = {'sum': rs, 'sq': rsq, 'count': cnt}
    return zscore, new_stats


# ============================================================
# Phase 2: Fill Sim Interface
# ============================================================

def prepare_bar_signal_for_fillsim(
    fold: Dict,
    event_timestamps: np.ndarray,
    signal_type: str = 'conviction',
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Generate bar-level signal for fill_sim_cli from a fold.

    signal_type options:
        'conviction': 10s horizon signal (standard)
        'agreement': multi-horizon agreement composite
        'adaptive': confidence-weighted composite
    """
    preds = fold['predictions']
    date_str = fold['date_str']

    if signal_type == 'conviction':
        raw_signal = preds[:, 2]  # 10s
    elif signal_type == 'agreement':
        signs = np.sign(preds)
        agreement = (signs[:, 0] == signs[:, 1]) & (signs[:, 1] == signs[:, 2])
        composite = np.where(
            agreement,
            np.sign(preds[:, 1]) * np.abs(preds[:, 0]) * (1.0 + np.abs(preds[:, 2])),
            0.0,
        )
        raw_signal = composite
    elif signal_type == 'adaptive':
        raw_signal = preds[:, 2] * (1.0 + np.abs(preds[:, 0]))
    else:
        raw_signal = preds[:, 2]

    return predictions_to_bar_signal(raw_signal, event_timestamps, date_str, running_stats)


def run_fill_sim(
    date_str: str,
    pred_file: Path,
    strategy_label: str,
    cli_args: List[str],
    out_dir: Path,
) -> Optional[Dict]:
    """Run Rust fill_sim_cli for a single day + strategy."""
    if not BINARY.exists():
        log.error(f"fill_sim_cli not found: {BINARY}")
        return None

    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = out_dir / f'{strategy_label}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
    ] + cli_args

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            log.warning(f"Sim failed {strategy_label}/{date_str}: {r.stderr[:200]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        log.warning(f"Timeout: {strategy_label}/{date_str}")
        return None
    except Exception as e:
        log.warning(f"Error {strategy_label}/{date_str}: {e}")
        return None


def run_fillsim_strategies(
    folds: List[Dict],
    strategies: List[Dict],
    workers: int = 4,
) -> Dict[str, List[Dict]]:
    """Run top strategies through fill_sim_cli for real market replay.

    Each strategy dict has:
        label, signal_type, cli_args (list of strings)
    """
    sim_dir = RESULTS_DIR / f'fillsim_{_ts}'
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Prepare bar-level predictions for each signal_type
    signal_types_needed = set(s['signal_type'] for s in strategies)
    pred_files = {}  # {(signal_type, date_str): npz_path}

    for signal_type in signal_types_needed:
        running_stats = None
        for fold in folds:
            date_str = fold['date_str']
            cache_key = f'{signal_type}_{date_str}'
            cache_file = PRED_CACHE_DIR / f'{cache_key}.npz'

            if cache_file.exists():
                pred_files[(signal_type, date_str)] = cache_file
                continue

            # Load event timestamps
            event_file = EVENT_DIR / f'{date_str}_mbo_events.npz'
            if not event_file.exists():
                log.warning(f"  No event file for {date_str}")
                continue

            ev_data = np.load(str(event_file), allow_pickle=True)
            timestamps = ev_data['timestamps']

            bar_signal, running_stats = prepare_bar_signal_for_fillsim(
                fold, timestamps, signal_type, running_stats,
            )

            np.savez_compressed(str(cache_file), predictions=bar_signal)
            pred_files[(signal_type, date_str)] = cache_file

            n_nonzero = int((bar_signal != 0).sum())
            log.info(f"  Prepared bar signal: {signal_type}/{date_str}, "
                     f"{n_nonzero} non-zero bars")

            del ev_data, timestamps
            gc.collect()

    # Build job list
    jobs = []
    for strat in strategies:
        for fold in folds:
            date_str = fold['date_str']
            key = (strat['signal_type'], date_str)
            if key not in pred_files:
                continue
            jobs.append({
                'label': strat['label'],
                'date': date_str,
                'pred_file': pred_files[key],
                'cli_args': strat['cli_args'],
            })

    log.info(f"\n  Running {len(jobs)} fill sim jobs ({workers} workers)")

    results = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            f = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['label'],
                job['cli_args'], sim_dir,
            )
            futures[f] = job

        done = 0
        for future in as_completed(futures):
            done += 1
            job = futures[future]
            try:
                result = future.result()
                if result:
                    results[job['label']].append({
                        'date': job['date'],
                        'result': result,
                    })
            except Exception as e:
                log.warning(f"  Job error: {e}")

            if done % 10 == 0 or done == len(jobs):
                log.info(f"  [{done}/{len(jobs)}] sim jobs complete")

    return dict(results)


def aggregate_fillsim_results(results: Dict[str, List[Dict]]) -> List[Dict]:
    """Aggregate fill sim results into strategy summaries."""
    summaries = []

    for label, day_results in results.items():
        total_pnl = 0
        total_trades = 0
        total_signals = 0
        daily_pnls = []
        all_pnls = []

        for dr in day_results:
            res = dr['result']
            day_pnl = res.get('total_pnl_dollars', 0)
            total_pnl += day_pnl
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            daily_pnls.append(day_pnl)

            if 'trades' in res:
                for t in res['trades']:
                    all_pnls.append(t.get('pnl_dollars', 0))

        n_days = len(day_results)
        if n_days == 0 or total_trades == 0:
            continue

        pnl_arr = np.array(all_pnls)
        avg_daily = np.mean(daily_pnls)
        downside = [min(0, x) for x in daily_pnls]
        ds_std = np.std(downside) if downside else 1e-8
        sortino = (avg_daily / max(ds_std, 1e-8)) * np.sqrt(252)

        winners = pnl_arr[pnl_arr > 0]
        losers = pnl_arr[pnl_arr <= 0]
        gp = float(winners.sum()) if len(winners) > 0 else 0
        gl = abs(float(losers.sum())) if len(losers) > 0 else 0.001

        summaries.append({
            'label': label,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(total_trades / max(total_signals, 1), 4),
            'trades_per_day': round(total_trades / n_days, 1),
            'win_rate': round(float((pnl_arr > 0).mean()), 4) if len(pnl_arr) > 0 else 0,
            'avg_trade': round(float(pnl_arr.mean()), 2) if len(pnl_arr) > 0 else 0,
            'avg_trade_ticks': round(float(pnl_arr.mean() / TICK_VALUE), 3) if len(pnl_arr) > 0 else 0,
            'profit_factor': round(gp / gl, 3),
            'sortino': round(sortino, 2),
            'avg_daily': round(avg_daily, 2),
            'daily_pnls': [round(x, 2) for x in daily_pnls],
        })

    summaries.sort(key=lambda x: x['total_pnl'], reverse=True)
    return summaries


# ============================================================
# Phase 1: Run All Strategies (Label-Based Screen)
# ============================================================

def run_phase1(folds: List[Dict]) -> Dict[str, Any]:
    """Phase 1: Quick label-based screen of all strategies.

    Uses labels to compute theoretical P&L. Fast, no fill sim.
    """
    log.info("\n" + "=" * 80)
    log.info("  PHASE 1: LABEL-BASED STRATEGY SCREEN")
    log.info("=" * 80)

    all_results = {}

    # Concatenate all folds for per-fold and concat analysis
    all_preds = np.concatenate([f['predictions'] for f in folds])
    all_labels = np.concatenate([f['labels'] for f in folds])
    all_embs = np.concatenate([f['embeddings'] for f in folds])

    # Walk-forward z-score across folds
    running_stats = None
    z_parts = []
    for fold in folds:
        z_fold, running_stats = compute_expanding_zscore_carryover(
            fold['predictions'], running_stats, min_samples=50,
        )
        z_parts.append(z_fold)
    all_z = np.concatenate(z_parts)

    log.info(f"\n  Total samples: {len(all_preds)}")
    log.info(f"  Folds: {len(folds)}")
    log.info(f"  Dates: {[f['date_str'] for f in folds]}")
    log.info(f"  Z-score stats: mean={all_z[:, 2].mean():.3f}, "
             f"std={all_z[:, 2].std():.3f}, "
             f"max={np.abs(all_z[:, 2]).max():.2f}")

    # ── Strategy 0: Confidence Tier Baseline ──
    log.info("\n--- Strategy 0: Confidence Tier Baseline ---")
    tier_results = strategy_confidence_tiers(all_z, all_labels)
    all_results['confidence_tiers'] = tier_results

    if tier_results['tiers']:
        log.info(f"\n  {'Tier':<10} {'N':>7} {'z_thresh':>8} "
                 f"{'1s_pnl':>8} {'5s_pnl':>8} {'10s_pnl':>8} "
                 f"{'10s_WR':>7} {'10s_PF':>6} "
                 f"{'10s_L_pnl':>9} {'10s_S_pnl':>9}")
        log.info("  " + "-" * 100)
        for t in tier_results['tiers']:
            log.info(f"  {t['tier']:<10} {t['n_trades']:>7} {t['threshold']:>8.2f} "
                     f"{t['1s_total_pnl']:>8.1f} {t['5s_total_pnl']:>8.1f} "
                     f"{t['10s_total_pnl']:>8.1f} "
                     f"{t['10s_win_rate']:>6.1%} {t['10s_pf']:>6.2f} "
                     f"{t['10s_long_pnl']:>9.1f} {t['10s_short_pnl']:>9.1f}")

    # ── Strategy 1: Signal-Flip Variants ──
    log.info("\n--- Strategy 1: Signal-Flip Exit ---")
    flip_variants = [
        ('flip_imm_z2.0', 2.0, 'immediate', 0, 1),
        ('flip_imm_z2.5', 2.5, 'immediate', 0, 1),
        ('flip_imm_z3.0', 3.0, 'immediate', 0, 1),
        ('flip_cond_z2.0_ft1.0', 2.0, 'conditional', 1.0, 1),
        ('flip_cond_z2.5_ft1.5', 2.5, 'conditional', 1.5, 1),
        ('flip_cond_z3.0_ft2.0', 3.0, 'conditional', 2.0, 1),
        ('flip_slow_z2.0_n3', 2.0, 'slow', 0, 3),
        ('flip_slow_z2.5_n3', 2.5, 'slow', 0, 3),
        ('flip_slow_z2.5_n5', 2.5, 'slow', 0, 5),
    ]

    flip_results = {}
    for name, thresh, mode, ft, fc in flip_variants:
        r = strategy_signal_flip(all_z, all_labels, threshold=thresh,
                                 flip_mode=mode, flip_threshold=ft,
                                 flip_consecutive=fc, max_hold_samples=200)
        flip_results[name] = r
        log.info(f"  {name:<30} trades={r['n_trades']:>5}, "
                 f"pnl={r['total_pnl_ticks']:>8.1f}t (${r['total_pnl_dollars']:>8.0f}), "
                 f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, "
                 f"avg_hold={r.get('avg_hold_samples', 0):.1f}")
    all_results['signal_flip'] = flip_results

    # ── Strategy 2: Multi-Horizon Agreement ──
    log.info("\n--- Strategy 2: Multi-Horizon Agreement Gate ---")
    agreement_variants = [
        ('agree_all_z1.0', 1.0, True),
        ('agree_all_z1.5', 1.5, True),
        ('agree_all_z2.0', 2.0, True),
        ('agree_all_z2.5', 2.5, True),
        ('agree_all_z3.0', 3.0, True),
        ('agree_2of3_z1.5', 1.5, False),
        ('agree_2of3_z2.0', 2.0, False),
        ('agree_2of3_z2.5', 2.5, False),
    ]

    agree_results = {}
    for name, thresh, require_all in agreement_variants:
        r = strategy_horizon_agreement(all_z, all_labels, threshold=thresh,
                                       require_all=require_all)
        agree_results[name] = r
        log.info(f"  {name:<25} trades={r['n_trades']:>5}, "
                 f"pnl={r['total_pnl_ticks']:>8.1f}t (${r['total_pnl_dollars']:>8.0f}), "
                 f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}, "
                 f"L={r['long_n']}/${r['long_pnl']:.0f}t S={r['short_n']}/${r['short_pnl']:.0f}t")
    all_results['horizon_agreement'] = agree_results

    # ── Strategy 3: Confidence-Adaptive Hold ──
    log.info("\n--- Strategy 3: Confidence-Adaptive Hold ---")
    adaptive_variants = [
        ('adapt_z1.5', 1.5),
        ('adapt_z2.0', 2.0),
        ('adapt_z2.5', 2.5),
        ('adapt_z3.0', 3.0),
    ]

    adapt_results = {}
    for name, min_z in adaptive_variants:
        r = strategy_adaptive_hold(all_z, all_labels, min_z=min_z)
        adapt_results[name] = r
        log.info(f"  {name:<20} trades={r['n_trades']:>5}, "
                 f"pnl={r['total_pnl_ticks']:>8.1f}t (${r['total_pnl_dollars']:>8.0f}), "
                 f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}")
        if 'horizon_breakdown' in r:
            for hz, stats in r['horizon_breakdown'].items():
                log.info(f"    {hz}: n={stats['n']}, "
                         f"avg={stats['avg_pnl']:.2f}t, "
                         f"WR={stats['win_rate']:.1%}")
    all_results['adaptive_hold'] = adapt_results

    # ── Strategy 4: Embedding MLP Gate ──
    log.info("\n--- Strategy 4: Embedding MLP Gate ---")
    mlp_variants = [
        ('mlp_z2.0_t0.5_h32', 2.0, 0.5, 32),
        ('mlp_z2.0_t0.6_h32', 2.0, 0.6, 32),
        ('mlp_z2.0_t0.5_h64', 2.0, 0.5, 64),
        ('mlp_z1.5_t0.5_h32', 1.5, 0.5, 32),
        ('mlp_z2.5_t0.5_h32', 2.5, 0.5, 32),
    ]

    mlp_results = {}
    for name, zt, mt, hd in mlp_variants:
        r = strategy_embedding_mlp_gate(folds, z_threshold=zt,
                                        mlp_threshold=mt, hidden_dim=hd)
        mlp_results[name] = r
        log.info(f"  {name:<25} trades={r['n_trades']:>5}, "
                 f"pnl={r['total_pnl_ticks']:>8.1f}t (${r['total_pnl_dollars']:>8.0f}), "
                 f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}")
    all_results['embedding_mlp'] = mlp_results

    # ── Strategy 5: Volatility-Regime Conditional ──
    log.info("\n--- Strategy 5: Volatility-Regime Conditional ---")
    vol_result = strategy_vol_regime(all_z, all_labels, all_preds)
    all_results['vol_regime'] = vol_result
    log.info(f"  vol_regime: trades={vol_result['n_trades']:>5}, "
             f"pnl={vol_result['total_pnl_ticks']:>8.1f}t "
             f"(${vol_result['total_pnl_dollars']:>8.0f}), "
             f"WR={vol_result['win_rate']:.1%}, PF={vol_result['profit_factor']:.2f}")
    if 'regime_breakdown' in vol_result:
        for regime, stats in vol_result['regime_breakdown'].items():
            log.info(f"    {regime}: n={stats['n']}, avg={stats['avg_pnl']:.2f}t, "
                     f"WR={stats['win_rate']:.1%}")
    if 'tod_breakdown' in vol_result:
        for tod, stats in vol_result['tod_breakdown'].items():
            log.info(f"    {tod}: n={stats['n']}, avg={stats['avg_pnl']:.2f}t, "
                     f"WR={stats['win_rate']:.1%}")

    return all_results


# ============================================================
# Phase 1 Ranking
# ============================================================

def rank_phase1_strategies(results: Dict[str, Any]) -> List[Dict]:
    """Extract all strategy variants and rank by total P&L (ticks).

    Returns sorted list of {name, total_pnl_ticks, n_trades, win_rate, ...}
    """
    ranked = []

    # Signal flip
    for name, r in results.get('signal_flip', {}).items():
        if r.get('n_trades', 0) > 0:
            ranked.append({
                'name': name,
                'group': 'signal_flip',
                'n_trades': r['n_trades'],
                'total_pnl_ticks': r['total_pnl_ticks'],
                'total_pnl_dollars': r['total_pnl_dollars'],
                'avg_pnl_ticks': r['avg_pnl_ticks'],
                'win_rate': r['win_rate'],
                'profit_factor': r['profit_factor'],
            })

    # Horizon agreement
    for name, r in results.get('horizon_agreement', {}).items():
        if r.get('n_trades', 0) > 0:
            ranked.append({
                'name': name,
                'group': 'horizon_agreement',
                'n_trades': r['n_trades'],
                'total_pnl_ticks': r['total_pnl_ticks'],
                'total_pnl_dollars': r['total_pnl_dollars'],
                'avg_pnl_ticks': r['avg_pnl_ticks'],
                'win_rate': r['win_rate'],
                'profit_factor': r['profit_factor'],
            })

    # Adaptive hold
    for name, r in results.get('adaptive_hold', {}).items():
        if r.get('n_trades', 0) > 0:
            ranked.append({
                'name': name,
                'group': 'adaptive_hold',
                'n_trades': r['n_trades'],
                'total_pnl_ticks': r['total_pnl_ticks'],
                'total_pnl_dollars': r['total_pnl_dollars'],
                'avg_pnl_ticks': r['avg_pnl_ticks'],
                'win_rate': r['win_rate'],
                'profit_factor': r['profit_factor'],
            })

    # MLP
    for name, r in results.get('embedding_mlp', {}).items():
        if r.get('n_trades', 0) > 0:
            ranked.append({
                'name': name,
                'group': 'embedding_mlp',
                'n_trades': r['n_trades'],
                'total_pnl_ticks': r['total_pnl_ticks'],
                'total_pnl_dollars': r['total_pnl_dollars'],
                'avg_pnl_ticks': r['avg_pnl_ticks'],
                'win_rate': r['win_rate'],
                'profit_factor': r['profit_factor'],
            })

    # Vol regime
    vr = results.get('vol_regime', {})
    if vr.get('n_trades', 0) > 0:
        ranked.append({
            'name': 'vol_regime',
            'group': 'vol_regime',
            'n_trades': vr['n_trades'],
            'total_pnl_ticks': vr['total_pnl_ticks'],
            'total_pnl_dollars': vr['total_pnl_dollars'],
            'avg_pnl_ticks': vr['avg_pnl_ticks'],
            'win_rate': vr['win_rate'],
            'profit_factor': vr['profit_factor'],
        })

    ranked.sort(key=lambda x: x['total_pnl_ticks'], reverse=True)
    return ranked


# ============================================================
# Phase 2: Fill Sim for Top Strategies
# ============================================================

def select_fillsim_strategies(ranked: List[Dict], top_n: int = 8) -> List[Dict]:
    """Select top strategies from Phase 1 and translate to fill_sim_cli configs.

    Returns list of {label, signal_type, cli_args} for fill sim.
    """
    fillsim_strats = []

    # Always include a few canonical configs
    canonical = [
        {
            'label': 'conviction_z2.5_10s',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '2.5', '--hold-ms', '10000',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z3.0_10s',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '3.0', '--hold-ms', '10000',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z2.5_flip',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '2.5', '--hold-ms', '300000',
                         '--signal-flip-exit', '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z3.0_flip',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '3.0', '--hold-ms', '300000',
                         '--signal-flip-exit', '--chase-entry', '--quiet'],
        },
        {
            'label': 'agreement_z2.0_10s',
            'signal_type': 'agreement',
            'cli_args': ['--signal-threshold', '2.0', '--hold-ms', '10000',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'agreement_z2.5_10s',
            'signal_type': 'agreement',
            'cli_args': ['--signal-threshold', '2.5', '--hold-ms', '10000',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'agreement_z2.0_flip',
            'signal_type': 'agreement',
            'cli_args': ['--signal-threshold', '2.0', '--hold-ms', '300000',
                         '--signal-flip-exit', '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z2.5_tp4_sl2',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '2.5', '--hold-ms', '60000',
                         '--take-profit-ticks', '4', '--stop-loss-ticks', '2',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z3.0_tp4_sl2',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '3.0', '--hold-ms', '60000',
                         '--take-profit-ticks', '4', '--stop-loss-ticks', '2',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z2.5_trail2',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '2.5', '--hold-ms', '60000',
                         '--trailing-ticks', '2', '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z3.0_conv50',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '3.0', '--hold-ms', '300000',
                         '--conviction-exit-bars', '50', '--conviction-exit-mag', '1.5',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z2.5_market',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '2.5', '--hold-ms', '10000',
                         '--market-entry', '--quiet'],
        },
        {
            'label': 'conviction_z4.0_10s',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '4.0', '--hold-ms', '10000',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z5.0_10s',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '5.0', '--hold-ms', '10000',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'adaptive_z2.5_10s',
            'signal_type': 'adaptive',
            'cli_args': ['--signal-threshold', '2.5', '--hold-ms', '10000',
                         '--chase-entry', '--quiet'],
        },
        {
            'label': 'conviction_z3.0_ratchet',
            'signal_type': 'conviction',
            'cli_args': ['--signal-threshold', '3.0', '--hold-ms', '60000',
                         '--ratchet-stop', '--trailing-ticks', '3',
                         '--chase-entry', '--quiet'],
        },
    ]

    return canonical


# ============================================================
# Reporting
# ============================================================

def print_phase1_ranking(ranked: List[Dict]):
    """Print Phase 1 ranking table."""
    log.info("\n" + "=" * 110)
    log.info("  PHASE 1 RANKING: ALL STRATEGIES (Label-Based, sorted by total P&L)")
    log.info("=" * 110)

    header = (f"{'Rank':>4} {'Strategy':<30} {'Group':<18} "
              f"{'Trades':>7} {'PnL(t)':>9} {'PnL($)':>9} "
              f"{'Avg(t)':>7} {'WR':>6} {'PF':>6}")
    log.info(header)
    log.info("-" * 110)

    for i, r in enumerate(ranked):
        marker = " *" if r['total_pnl_ticks'] > 0 else "  "
        log.info(f"{marker}{i + 1:>2} {r['name']:<30} {r['group']:<18} "
                 f"{r['n_trades']:>7} {r['total_pnl_ticks']:>9.1f} "
                 f"${r['total_pnl_dollars']:>8.0f} "
                 f"{r['avg_pnl_ticks']:>7.3f} {r['win_rate']:>5.1%} "
                 f"{r['profit_factor']:>6.2f}")

    # Summary
    profitable = [r for r in ranked if r['total_pnl_ticks'] > 0]
    log.info(f"\n  {len(profitable)}/{len(ranked)} strategies profitable (label-based)")
    log.info(f"  Cost assumption: {TOTAL_COST_TICKS:.3f} ticks RT "
             f"({COMMISSION_TICKS:.3f} commission + {SLIPPAGE_TICKS:.1f} slippage)")
    log.info(f"  ES tick = ${TICK_VALUE}, Commission = ${COMMISSION_RT}")


def print_fillsim_results(summaries: List[Dict]):
    """Print Phase 2 fill sim results."""
    log.info("\n" + "=" * 130)
    log.info("  PHASE 2: FILL SIM RESULTS (Rust market replay, sorted by total P&L)")
    log.info("=" * 130)

    header = (f"{'Rank':>4} {'Strategy':<30} "
              f"{'P&L':>9} {'Trades':>7} {'T/Day':>6} {'FillR':>6} "
              f"{'WR':>6} {'AvgTrd':>8} {'PF':>6} {'Sortino':>8} "
              f"{'Daily P&Ls'}")
    log.info(header)
    log.info("-" * 130)

    for i, s in enumerate(summaries):
        daily_str = ', '.join([f"${x:,.0f}" for x in s.get('daily_pnls', [])])
        marker = " *" if s['total_pnl'] > 0 else "  "
        log.info(f"{marker}{i + 1:>2} {s['label']:<30} "
                 f"${s['total_pnl']:>8,.0f} {s['n_trades']:>7} "
                 f"{s['trades_per_day']:>5.1f} {s['fill_rate']:>5.1%} "
                 f"{s['win_rate']:>5.1%} ${s['avg_trade']:>7.2f} "
                 f"{s['profit_factor']:>6.2f} {s['sortino']:>8.2f} "
                 f"[{daily_str}]")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Advanced Execution Strategies v2 — CNN-Mamba v2',
    )
    parser.add_argument('--phase1-only', action='store_true',
                        help='Only run Phase 1 (label-based screen)')
    parser.add_argument('--skip-fillsim', action='store_true',
                        help='Skip Phase 2 fill sim (same as --phase1-only)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workers for fill sim')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("  ADVANCED EXECUTION STRATEGIES v2")
    log.info("  CNN-Mamba v2 Multi-Strategy Engine")
    log.info("=" * 80)
    log.info(f"  Model: CNN-Mamba v2 (4 folds, Feb 23-26)")
    log.info(f"  Features: predictions (N,3), labels (N,3), embeddings (N,96)")
    log.info(f"  Instrument: ES (E-mini S&P 500)")
    log.info(f"  Tick: ${TICK_VALUE}, Commission: ${COMMISSION_RT} RT")
    log.info(f"  Cost model: {TOTAL_COST_TICKS:.3f} ticks "
             f"({COMMISSION_TICKS:.3f} comm + {SLIPPAGE_TICKS:.1f} slip)")
    log.info(f"  Fill sim binary: {BINARY}")
    log.info("=" * 80)

    # Load data
    log.info("\nLoading CNN-Mamba v2 predictions...")
    folds = load_all_folds()
    if not folds:
        log.error("No folds found! Check path: {CNNMAMBA_V2_DIR}")
        sys.exit(1)

    total_samples = sum(f['n_samples'] for f in folds)
    log.info(f"\n  Loaded {len(folds)} folds, {total_samples} total samples")

    # Phase 1: Label-based screen
    t0 = time.time()
    phase1_results = run_phase1(folds)
    phase1_time = time.time() - t0
    log.info(f"\n  Phase 1 completed in {phase1_time:.1f}s")

    # Rank all strategies
    ranked = rank_phase1_strategies(phase1_results)
    print_phase1_ranking(ranked)

    # Phase 2: Fill sim
    skip_sim = args.phase1_only or args.skip_fillsim
    if skip_sim:
        log.info("\n  Skipping Phase 2 (fill sim) as requested.")
    elif not BINARY.exists():
        log.warning(f"\n  fill_sim_cli not found at {BINARY}, skipping Phase 2")
        skip_sim = True

    fillsim_summaries = []
    if not skip_sim:
        log.info("\n" + "=" * 80)
        log.info("  PHASE 2: FILL SIM MARKET REPLAY")
        log.info("=" * 80)

        t0 = time.time()
        fillsim_strats = select_fillsim_strategies(ranked)
        log.info(f"  Running {len(fillsim_strats)} fill sim strategies")

        fillsim_results = run_fillsim_strategies(
            folds, fillsim_strats, workers=args.workers,
        )
        fillsim_summaries = aggregate_fillsim_results(fillsim_results)
        phase2_time = time.time() - t0
        log.info(f"\n  Phase 2 completed in {phase2_time:.1f}s")

        print_fillsim_results(fillsim_summaries)

    # Save all results
    out_data = {
        'timestamp': _ts,
        'model': 'cnn_mamba_v2',
        'instrument': 'ES',
        'tick_value': TICK_VALUE,
        'commission_rt': COMMISSION_RT,
        'total_cost_ticks': TOTAL_COST_TICKS,
        'n_folds': len(folds),
        'dates': [f['date_str'] for f in folds],
        'total_samples': total_samples,
        'phase1_ranking': ranked,
        'phase1_details': {
            'confidence_tiers': phase1_results.get('confidence_tiers', {}),
            'vol_regime': {k: v for k, v in phase1_results.get('vol_regime', {}).items()
                          if k != 'trades'} if isinstance(phase1_results.get('vol_regime'), dict) else {},
        },
        'phase2_fillsim': fillsim_summaries,
    }

    out_file = RESULTS_DIR / f'adv_v2_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump(out_data, f, indent=2, default=str)

    # Final summary
    log.info(f"\n{'=' * 80}")
    log.info(f"  EXECUTION COMPLETE")
    log.info(f"{'=' * 80}")
    log.info(f"  Phase 1: {len(ranked)} strategies screened (label-based)")
    profitable_p1 = sum(1 for r in ranked if r['total_pnl_ticks'] > 0)
    log.info(f"  Phase 1 profitable: {profitable_p1}/{len(ranked)}")
    if ranked:
        best = ranked[0]
        log.info(f"  Phase 1 best: {best['name']} "
                 f"({best['total_pnl_ticks']:.1f} ticks, "
                 f"WR={best['win_rate']:.1%}, PF={best['profit_factor']:.2f})")

    if fillsim_summaries:
        log.info(f"\n  Phase 2: {len(fillsim_summaries)} strategies fill-sim tested")
        profitable_p2 = sum(1 for s in fillsim_summaries if s['total_pnl'] > 0)
        log.info(f"  Phase 2 profitable: {profitable_p2}/{len(fillsim_summaries)}")
        if fillsim_summaries:
            best2 = fillsim_summaries[0]
            log.info(f"  Phase 2 best: {best2['label']} "
                     f"(${best2['total_pnl']:,.0f}, "
                     f"WR={best2['win_rate']:.1%}, "
                     f"Sortino={best2['sortino']:.2f})")

    log.info(f"\n  Results: {out_file}")
    log.info(f"  Log: {_log_file}")
    log.info(f"{'=' * 80}")


if __name__ == '__main__':
    main()
