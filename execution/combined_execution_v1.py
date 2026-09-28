#!/usr/bin/env python3
"""
Combined CNN-Mamba + PatchTST Execution System v1
====================================================
Production-grade combined model execution framework for ES futures.

Architecture:
  - CNN-Mamba (or Mamba v7 stand-in): magnitude predictor, multi-horizon (1s/5s/10s)
  - PatchTST (or LGBM DA stand-in): directional accuracy filter
  - Combined signal: Mamba magnitude x DA direction agreement
  - Market replay via Rust fill_sim_cli on raw MBO data (NO mid-price PnL)

Signal Combination Logic:
  1. Mamba provides magnitude predictions at 3 horizons (1s, 5s, 10s)
  2. DA model provides directional accuracy filter
  3. Entry: Mamba top X% confidence AND DA agrees on direction
  4. 1s horizon for entry timing, 5s/10s for trade conviction
  5. Confidence tiers: Top 50%, 25%, 10%, 5%, 1%, 0.5%

Exit Logic:
  - Signal-flip: exit when model flips direction
  - Conditional signal-flip: only exit if opposing signal is also high-confidence
  - Time-based max hold (configurable: 5s, 10s, 30s, 60s)
  - Stop-loss / take-profit in ticks

Walk-Forward Evaluation:
  - OOT predictions from each fold (no lookahead)
  - Per-day, per-fold, and concat evaluation
  - Full metrics: trades/day, fill rate, win rate, avg $/trade, Sortino, PF, max DD
  - Long vs short breakdown

ES Futures Constants:
  - Tick size: $12.50 per tick (0.25 points)
  - Commission: $4.70 RT = 0.376 ticks
  - Point value: $50/point

Usage:
    python combined_execution_v1.py
    python combined_execution_v1.py --mode mamba-only --workers 8
    python combined_execution_v1.py --mode combined --da-source lgbm
    python combined_execution_v1.py --dry-run
"""

import sys
import gc
import json
import time
import argparse
import subprocess
import logging
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
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
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'combined_v1'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'combined_v1'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Mamba v7 Prediction Directory ──
MAMBA_V7_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'

# ── DA Model Directories ──
LGBM_DA_DIR = LVL3_ROOT / 'output' / 'lgbm_da_smart_v3_1d_oot'
PATCHTST_DIR = LVL3_ROOT / 'output' / 'patchtst_smart_v3_mar'

# ── ES Futures Constants ──────────────────────────────────────────────────
TICK_SIZE_PTS = 0.25       # ES tick = 0.25 points
TICK_VALUE = 12.50         # $12.50 per tick for ES
POINT_VALUE = 50.00        # $50 per point for ES
COMMISSION_RT = 4.70       # Round-trip commission per contract
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.376 ticks

# ── Bar/Timing Constants ──
BARS_PER_SEC = 10
BAR_NS = 100_000_000      # 100ms in nanoseconds
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# ── Mamba Model Constants ──
MAMBA_WINDOW = 1000
MAMBA_STRIDE = 500

# ── Timestamp ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──────────────────────────────────────────────────────────────
_log_file = str(RESULTS_DIR / f'combined_exec_{_ts}.log')
log = logging.getLogger('combined_execution')
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
    # DST: EDT (UTC-4) Mar second Sun - Nov first Sun
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

def load_mamba_fold(fold_path: Path) -> Optional[Dict]:
    """Load a single Mamba v7 fold prediction file.

    Returns dict with keys: predictions (N,3), labels (N,3),
    date_str, n_samples, embeddings (N,96).
    """
    try:
        data = np.load(str(fold_path), allow_pickle=True)
        preds = data['predictions']   # (N, 3) for 1s/5s/10s
        labels = data['labels']       # (N, 3)

        # Extract date from oot_files path
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
            result['embeddings'] = data['embeddings']

        return result
    except Exception as e:
        log.warning(f"Failed to load {fold_path}: {e}")
        return None


def load_mamba_concat(concat_path: Path) -> Optional[Dict]:
    """Load concatenated Mamba predictions (all folds).

    The concat file has flat arrays: preds_1s, preds_5s, preds_10s.
    """
    try:
        data = np.load(str(concat_path), allow_pickle=True)
        n = data['preds_1s'].shape[0]
        preds = np.stack([
            data['preds_1s'].astype(np.float64),
            data['preds_5s'].astype(np.float64),
            data['preds_10s'].astype(np.float64),
        ], axis=1)  # (N, 3)
        labels = np.stack([
            data['labels_1s'].astype(np.float64),
            data['labels_5s'].astype(np.float64),
            data['labels_10s'].astype(np.float64),
        ], axis=1)  # (N, 3)

        ic_1s = float(data['concat_ic_1s']) if 'concat_ic_1s' in data else 0.0
        ic_5s = float(data['concat_ic_5s']) if 'concat_ic_5s' in data else 0.0
        ic_10s = float(data['concat_ic_10s']) if 'concat_ic_10s' in data else 0.0

        return {
            'predictions': preds,
            'labels': labels,
            'n_samples': n,
            'ic_1s': ic_1s,
            'ic_5s': ic_5s,
            'ic_10s': ic_10s,
        }
    except Exception as e:
        log.warning(f"Failed to load concat {concat_path}: {e}")
        return None


def discover_mamba_folds() -> Dict[str, Path]:
    """Discover all Mamba v7 per-fold prediction files.

    Returns: {date_str: fold_path}
    """
    folds = {}
    for f in sorted(MAMBA_V7_DIR.glob('fold_*_oot_predictions.npz')):
        if 'concat' in f.name:
            continue
        data = load_mamba_fold(f)
        if data:
            folds[data['date_str']] = f
            log.info(f"  Mamba fold: {data['date_str']} -> {f.name} "
                     f"({data['n_samples']} samples)")
    return folds


def discover_da_predictions(da_source: str) -> Dict[str, Path]:
    """Discover DA model prediction files.

    Args:
        da_source: 'lgbm', 'patchtst', or 'none'

    Returns: {date_str: path}
    """
    if da_source == 'none':
        return {}

    da_dirs = {
        'lgbm': [LGBM_DA_DIR],
        'patchtst': [PATCHTST_DIR],
    }

    dirs = da_dirs.get(da_source, [])
    files = {}

    for d in dirs:
        if not d.exists():
            log.info(f"  DA dir not found: {d}")
            continue
        # Try per-fold files
        for f in sorted(d.glob('fold_*_oot_predictions.npz')):
            try:
                data = np.load(str(f), allow_pickle=True)
                if 'oot_files' in data:
                    oot_path = str(data['oot_files'][0])
                    basename = oot_path.replace('\\', '/').split('/')[-1]
                    date_str = basename.split('_')[0]
                    files[date_str] = f
            except Exception:
                continue
        # Try per-date files
        for f in sorted(d.glob('*_predictions.npz')):
            try:
                basename = f.stem
                parts = basename.split('_')
                if len(parts[0]) == 8 and parts[0].isdigit():
                    files[parts[0]] = f
            except Exception:
                continue

    if files:
        log.info(f"  Found {len(files)} DA prediction files from {da_source}")
    else:
        log.info(f"  No DA prediction files found for {da_source}")

    return files


# ============================================================
# Signal Generation
# ============================================================

def convert_predictions_to_bar_signal(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
    window: int = MAMBA_WINDOW,
    stride: int = MAMBA_STRIDE,
    expanding_zscore: bool = True,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Convert per-window predictions to bar-indexed z-scored signal.

    Maps Mamba predictions (stride=500, window=1000) to 100ms bar indices,
    then applies expanding walk-forward z-score normalization.

    Args:
        predictions: (N,) raw model predictions for one horizon
        event_timestamps: (M,) nanosecond timestamps of all events in the day
        date_str: YYYYMMDD
        window: sliding window size used during training
        stride: window stride used during training
        expanding_zscore: apply walk-forward expanding z-score
        running_stats: carry-over stats from previous days

    Returns:
        (bar_signal of shape (N_RTH_BARS,), updated running_stats)
    """
    n_events = len(event_timestamps)
    n_preds = len(predictions)

    # Window-based predictions: each prediction corresponds to the END of a window
    # prediction[j] was computed from events[j*stride : j*stride + window]
    # The "label event" is at index j*stride + window - 1
    starts = np.arange(0, n_events - window + 1, stride, dtype=np.int64)
    label_idxs = starts + window - 1

    # Trim to match prediction count
    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    elif n_preds > len(label_idxs):
        predictions = predictions[:len(label_idxs)]
        n_preds = len(predictions)

    if n_preds == 0:
        return np.zeros(N_RTH_BARS, dtype=np.float64), running_stats or {}

    pred_timestamps = event_timestamps[label_idxs]

    # Map to RTH bar indices (100ms bars from 9:30 AM ET)
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)
    bar_indices_rth = bar_indices[rth_mask]
    predictions_rth = predictions[rth_mask]

    # Create bar-level signal (last prediction wins for overlapping windows)
    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, sig in zip(bar_indices_rth, predictions_rth):
        bar_preds[bi] = sig

    if not expanding_zscore:
        return bar_preds, running_stats or {}

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


class CombinedSignalGenerator:
    """Generates combined CNN-Mamba + DA signals for the fill simulator.

    Modes:
      - mamba-only: pure Mamba multi-horizon signal
      - combined: Mamba magnitude x DA directional agreement
      - multi-horizon: composite of 1s/5s/10s with directional agreement gate
    """

    def __init__(
        self,
        mode: str = 'mamba-only',
        da_source: str = 'none',
        entry_horizon: str = '1s',
        direction_horizon: str = '5s',
        conviction_horizon: str = '10s',
        require_all_horizons_agree: bool = True,
    ):
        self.mode = mode
        self.da_source = da_source
        self.entry_horizon = entry_horizon
        self.direction_horizon = direction_horizon
        self.conviction_horizon = conviction_horizon
        self.require_all_horizons_agree = require_all_horizons_agree

        # Horizon column indices
        self._hz_map = {'1s': 0, '5s': 1, '10s': 2}

        # Walk-forward running stats (per-horizon, carried across days)
        self._running_stats = {
            'entry': None,
            'direction': None,
            'conviction': None,
            'combined': None,
        }

    def generate_bar_signal(
        self,
        mamba_fold: Dict,
        da_fold: Optional[Dict],
        event_timestamps: np.ndarray,
        date_str: str,
    ) -> Tuple[np.ndarray, Dict]:
        """Generate bar-level signal for one day.

        Returns:
            (bar_signal of shape (N_RTH_BARS,), metadata dict)
        """
        preds = mamba_fold['predictions']  # (N, 3)
        n_preds = preds.shape[0]

        entry_col = self._hz_map[self.entry_horizon]
        dir_col = self._hz_map[self.direction_horizon]
        conv_col = self._hz_map[self.conviction_horizon]

        p_entry = preds[:, entry_col]
        p_dir = preds[:, dir_col]
        p_conv = preds[:, conv_col]

        meta = {
            'date': date_str,
            'n_mamba_preds': n_preds,
            'mode': self.mode,
        }

        if self.mode == 'mamba-only':
            # Single horizon signal (conviction horizon = 10s by default)
            signal = p_conv
            bar_signal, self._running_stats['conviction'] = convert_predictions_to_bar_signal(
                signal, event_timestamps, date_str,
                expanding_zscore=True,
                running_stats=self._running_stats['conviction'],
            )
            meta['signal_type'] = f'mamba_{self.conviction_horizon}'

        elif self.mode == 'multi-horizon':
            # Multi-horizon composite:
            # All horizons must agree on direction, then:
            # signal = sign(direction) * |entry_timing| * (1 + |conviction|)
            if self.require_all_horizons_agree:
                agree_mask = (
                    (np.sign(p_entry) == np.sign(p_dir)) &
                    (np.sign(p_dir) == np.sign(p_conv))
                )
            else:
                # Only require direction and conviction to agree
                agree_mask = np.sign(p_dir) == np.sign(p_conv)

            composite = np.where(
                agree_mask,
                np.sign(p_dir) * np.abs(p_entry) * (1.0 + np.abs(p_conv)),
                0.0,
            )

            bar_signal, self._running_stats['combined'] = convert_predictions_to_bar_signal(
                composite, event_timestamps, date_str,
                expanding_zscore=True,
                running_stats=self._running_stats['combined'],
            )
            n_agree = int(np.sum(agree_mask))
            meta['signal_type'] = 'multi_horizon_composite'
            meta['n_horizons_agree'] = n_agree
            meta['agreement_rate'] = round(n_agree / max(n_preds, 1), 4)

        elif self.mode == 'combined':
            # Combined with DA model
            if da_fold is None:
                # Fall back to multi-horizon if no DA available
                log.warning(f"  {date_str}: No DA predictions, falling back to multi-horizon")
                return self._fallback_multi_horizon(
                    preds, event_timestamps, date_str, meta
                )

            # Align Mamba and DA predictions
            da_signal = self._extract_da_signal(da_fold)
            mamba_signal, da_aligned = self._align_mamba_da(
                p_conv, da_signal, n_preds, da_fold
            )

            if mamba_signal is None or len(mamba_signal) == 0:
                log.warning(f"  {date_str}: Alignment failed, falling back")
                return self._fallback_multi_horizon(
                    preds, event_timestamps, date_str, meta
                )

            # Combined: Mamba magnitude where DA agrees on direction
            mamba_dir = np.sign(mamba_signal)
            da_dir = np.sign(da_aligned)
            agreement = mamba_dir == da_dir

            # Confluence signal: keep Mamba magnitude only when DA agrees
            combined = np.where(agreement, mamba_signal, 0.0)

            # Need to create a padded version matching original mamba prediction count
            # for bar conversion (the alignment may have different length)
            padded = np.zeros(n_preds, dtype=np.float64)
            n_combined = min(len(combined), n_preds)
            padded[:n_combined] = combined[:n_combined]

            bar_signal, self._running_stats['combined'] = convert_predictions_to_bar_signal(
                padded, event_timestamps, date_str,
                expanding_zscore=True,
                running_stats=self._running_stats['combined'],
            )

            n_agree = int(np.sum(agreement))
            meta['signal_type'] = 'mamba_x_da_confluence'
            meta['da_source'] = self.da_source
            meta['n_da_aligned'] = len(da_aligned)
            meta['n_agreement'] = n_agree
            meta['agreement_rate'] = round(n_agree / max(len(da_aligned), 1), 4)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Compute signal stats
        nonzero = bar_signal[bar_signal != 0]
        if len(nonzero) > 0:
            meta['n_bar_signals'] = int(len(nonzero))
            meta['signal_max'] = round(float(np.max(bar_signal)), 3)
            meta['signal_min'] = round(float(np.min(bar_signal)), 3)
            meta['signal_mean_abs'] = round(float(np.mean(np.abs(nonzero))), 3)
            meta['signal_std'] = round(float(np.std(nonzero)), 3)
        else:
            meta['n_bar_signals'] = 0

        return bar_signal, meta

    def _fallback_multi_horizon(
        self, preds, event_timestamps, date_str, meta
    ) -> Tuple[np.ndarray, Dict]:
        """Fallback to multi-horizon when combined mode cannot be used."""
        dir_col = self._hz_map[self.direction_horizon]
        conv_col = self._hz_map[self.conviction_horizon]
        entry_col = self._hz_map[self.entry_horizon]

        agree = (
            (np.sign(preds[:, entry_col]) == np.sign(preds[:, dir_col])) &
            (np.sign(preds[:, dir_col]) == np.sign(preds[:, conv_col]))
        )
        composite = np.where(
            agree,
            np.sign(preds[:, dir_col]) * np.abs(preds[:, entry_col]) * (1.0 + np.abs(preds[:, conv_col])),
            0.0,
        )
        bar_signal, self._running_stats['combined'] = convert_predictions_to_bar_signal(
            composite, event_timestamps, date_str,
            expanding_zscore=True,
            running_stats=self._running_stats['combined'],
        )
        meta['signal_type'] = 'multi_horizon_fallback'
        nonzero = bar_signal[bar_signal != 0]
        meta['n_bar_signals'] = int(len(nonzero)) if len(nonzero) > 0 else 0
        return bar_signal, meta

    def _extract_da_signal(self, da_fold: Dict) -> np.ndarray:
        """Extract continuous directional signal from DA model predictions."""
        if 'predictions' in da_fold:
            preds = da_fold['predictions']
            if preds.ndim == 2:
                # Multi-horizon: use 10s column
                return preds[:, 2].astype(np.float64)
            return preds.astype(np.float64)
        if 'probs' in da_fold:
            # Binary classifier: prob - 0.5 (centered)
            return (da_fold['probs'] - 0.5).astype(np.float64)
        if 'preds_10s' in da_fold:
            return da_fold['preds_10s'].astype(np.float64)
        raise ValueError("Cannot extract DA signal from fold data")

    def _align_mamba_da(
        self,
        mamba_signal: np.ndarray,
        da_signal: np.ndarray,
        n_mamba: int,
        da_fold: Dict,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Align Mamba and DA predictions.

        Both models operate on the same event stream but may have different
        window/stride. For simplest case, if they have same stride=500,
        they align 1:1. Otherwise, use nearest-neighbor by index.
        """
        n_da = len(da_signal)

        if n_da == n_mamba:
            # Perfect 1:1 alignment
            return mamba_signal, da_signal

        # If DA has different stride (e.g., stride=250 vs Mamba stride=500),
        # subsample DA to match Mamba indices
        # Mamba sample j: event at j*500 + 1000 - 1
        # DA sample i (stride=250): event at i*250 + window - 1
        # Alignment: j*500 = i*250 => i = 2*j
        if n_da > n_mamba:
            ratio = n_da / n_mamba
            if 1.8 < ratio < 2.2:
                # stride ratio ~2 — subsample DA
                da_indices = np.minimum(
                    np.arange(n_mamba) * 2,
                    n_da - 1
                ).astype(int)
                return mamba_signal, da_signal[da_indices]

        # Generic: truncate to shorter
        n_min = min(n_mamba, n_da)
        return mamba_signal[:n_min], da_signal[:n_min]


# ============================================================
# Execution Strategy Definitions (ES-specific)
# ============================================================

@dataclass
class ExecutionStrategy:
    """Execution strategy specification for fill_sim_cli."""
    label: str
    description: str

    # Entry mode
    market_entry: bool = False
    chase_entry: bool = False
    chase_max_ticks: int = 1
    chase_max_reprices: int = 3
    chase_force_cross: bool = False
    chase_interval_ms: int = 100

    # Signal filtering
    signal_threshold: float = 2.0

    # Exit rules
    hold_ms: int = 10000
    signal_flip_exit: bool = False
    take_profit_ticks: Optional[int] = None
    stop_loss_ticks: Optional[int] = None
    trailing_ticks: Optional[int] = None
    ratchet_stop: bool = False
    conviction_exit_bars: int = 0
    conviction_exit_mag: float = 0.0

    # Latency
    latency_ms: int = 0

    # Time filters
    prime_hours: bool = False
    time_window_start: str = ""
    time_window_end: str = ""

    def to_cli_args(self) -> List[str]:
        """Convert to fill_sim_cli command-line arguments."""
        args = []

        if self.market_entry:
            args.append('--market-entry')
        elif self.chase_entry:
            args.append('--chase-entry')
            args.extend(['--chase-max-ticks', str(self.chase_max_ticks)])
            args.extend(['--chase-max-reprices', str(self.chase_max_reprices)])
            args.extend(['--chase-interval-ms', str(self.chase_interval_ms)])
            if self.chase_force_cross:
                args.append('--chase-force-cross')

        args.extend(['--signal-threshold', str(self.signal_threshold)])
        args.extend(['--hold-ms', str(self.hold_ms)])

        if self.signal_flip_exit:
            args.append('--signal-flip-exit')
        if self.take_profit_ticks is not None:
            args.extend(['--take-profit-ticks', str(self.take_profit_ticks)])
        if self.stop_loss_ticks is not None:
            args.extend(['--stop-loss-ticks', str(self.stop_loss_ticks)])
        if self.trailing_ticks is not None:
            args.extend(['--trailing-ticks', str(self.trailing_ticks)])
        if self.ratchet_stop:
            args.append('--ratchet-stop')
        if self.conviction_exit_bars > 0:
            args.extend(['--conviction-exit-bars', str(self.conviction_exit_bars)])
            args.extend(['--conviction-exit-mag', str(self.conviction_exit_mag)])

        if self.latency_ms > 0:
            args.extend(['--latency-ms', str(self.latency_ms)])

        if self.prime_hours:
            args.append('--prime-hours')
        if self.time_window_start:
            args.extend(['--time-window-start', self.time_window_start])
            args.extend(['--time-window-end', self.time_window_end])

        args.append('--quiet')
        return args


def build_combined_strategy_suite() -> Dict[str, List[ExecutionStrategy]]:
    """Build strategy suite tuned for combined CNN-Mamba + PatchTST signals on ES."""
    strategies = {}

    # ── GROUP A: Confidence-Gated Entry (the main sweep) ──
    # The combined signal should have higher signal-to-noise,
    # so we test a wide range of thresholds
    strategies['confidence_sweep'] = []
    for z in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        strategies['confidence_sweep'].append(ExecutionStrategy(
            label=f'chase_z{int(z*10):02d}_10s',
            description=f'Chase 1x3, z>{z}, hold 10s',
            chase_entry=True, signal_threshold=z, hold_ms=10000,
        ))

    # ── GROUP B: Hold Time Sweep ──
    strategies['hold_time'] = []
    for hold_ms, hold_label in [(1000, '1s'), (5000, '5s'), (10000, '10s'),
                                 (30000, '30s'), (60000, '60s')]:
        strategies['hold_time'].append(ExecutionStrategy(
            label=f'chase_z25_hold{hold_label}',
            description=f'Chase 1x3, z>2.5, hold {hold_label}',
            chase_entry=True, signal_threshold=2.5, hold_ms=hold_ms,
        ))

    # ── GROUP C: Exit Strategies ──
    strategies['exit_strategies'] = [
        # Signal-flip exit (dynamic hold based on model)
        ExecutionStrategy(
            label='flip_z25_5min',
            description='Chase 1x3, z>2.5, signal-flip exit, max 5min',
            chase_entry=True, signal_threshold=2.5,
            hold_ms=300000, signal_flip_exit=True,
        ),
        ExecutionStrategy(
            label='flip_z30_5min',
            description='Chase 1x3, z>3.0, signal-flip exit, max 5min',
            chase_entry=True, signal_threshold=3.0,
            hold_ms=300000, signal_flip_exit=True,
        ),
        # Conviction exit (delayed flip — only exit if strong opposing signal)
        ExecutionStrategy(
            label='conviction_z25_50bars',
            description='Chase 1x3, z>2.5, conviction exit 50 bars (5s)',
            chase_entry=True, signal_threshold=2.5,
            hold_ms=300000,
            conviction_exit_bars=50, conviction_exit_mag=1.5,
        ),
        ExecutionStrategy(
            label='conviction_z30_100bars',
            description='Chase 1x3, z>3.0, conviction exit 100 bars (10s)',
            chase_entry=True, signal_threshold=3.0,
            hold_ms=300000,
            conviction_exit_bars=100, conviction_exit_mag=2.0,
        ),
        # TP/SL combinations
        ExecutionStrategy(
            label='tpsl_tp3_sl2_z25',
            description='Chase 1x3, TP=3t SL=2t, z>2.5, hold 60s',
            chase_entry=True, signal_threshold=2.5,
            hold_ms=60000, take_profit_ticks=3, stop_loss_ticks=2,
        ),
        ExecutionStrategy(
            label='tpsl_tp4_sl2_z30',
            description='Chase 1x3, TP=4t SL=2t, z>3.0, hold 60s',
            chase_entry=True, signal_threshold=3.0,
            hold_ms=60000, take_profit_ticks=4, stop_loss_ticks=2,
        ),
        ExecutionStrategy(
            label='tpsl_tp5_sl3_z30',
            description='Chase 1x3, TP=5t SL=3t, z>3.0, hold 60s',
            chase_entry=True, signal_threshold=3.0,
            hold_ms=60000, take_profit_ticks=5, stop_loss_ticks=3,
        ),
        # Trailing stop
        ExecutionStrategy(
            label='trail_2t_z25_60s',
            description='Chase 1x3, trailing 2 ticks, z>2.5, hold 60s',
            chase_entry=True, signal_threshold=2.5,
            hold_ms=60000, trailing_ticks=2,
        ),
        # Signal-flip + TP/SL combo (best of both worlds)
        ExecutionStrategy(
            label='flip_tp4_sl2_z25',
            description='Chase 1x3, flip exit + TP=4t SL=2t, z>2.5',
            chase_entry=True, signal_threshold=2.5,
            hold_ms=300000, signal_flip_exit=True,
            take_profit_ticks=4, stop_loss_ticks=2,
        ),
    ]

    # ── GROUP D: Entry Modes ──
    strategies['entry_modes'] = [
        ExecutionStrategy(
            label='passive_z25_10s',
            description='Passive limit, z>2.5, hold 10s',
            signal_threshold=2.5, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='chase_z25_10s_entry',
            description='Chase 1x3, z>2.5, hold 10s',
            chase_entry=True, signal_threshold=2.5, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='chase_force_z25_10s',
            description='Chase 1x3 force-cross, z>2.5, hold 10s',
            chase_entry=True, chase_force_cross=True,
            signal_threshold=2.5, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='market_z30_10s',
            description='Market order, z>3.0, hold 10s',
            market_entry=True, signal_threshold=3.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='market_z40_10s',
            description='Market order, z>4.0, hold 10s',
            market_entry=True, signal_threshold=4.0, hold_ms=10000,
        ),
    ]

    # ── GROUP E: Ultra-Selective (deployment candidates) ──
    strategies['ultra_selective'] = [
        ExecutionStrategy(
            label='ultra_chase_z35_10s',
            description='Chase 1x3, z>3.5, hold 10s',
            chase_entry=True, signal_threshold=3.5, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='ultra_chase_z40_10s',
            description='Chase 1x3, z>4.0, hold 10s',
            chase_entry=True, signal_threshold=4.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='ultra_chase_z50_10s',
            description='Chase 1x3, z>5.0, hold 10s',
            chase_entry=True, signal_threshold=5.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='ultra_flip_z35_5min',
            description='Chase 1x3, z>3.5, signal-flip, max 5min',
            chase_entry=True, signal_threshold=3.5,
            hold_ms=300000, signal_flip_exit=True,
        ),
        ExecutionStrategy(
            label='ultra_flip_z40_5min',
            description='Chase 1x3, z>4.0, signal-flip, max 5min',
            chase_entry=True, signal_threshold=4.0,
            hold_ms=300000, signal_flip_exit=True,
        ),
        ExecutionStrategy(
            label='ultra_market_z40_10s',
            description='Market order, z>4.0, hold 10s',
            market_entry=True, signal_threshold=4.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='ultra_market_z50_10s',
            description='Market order, z>5.0, hold 10s',
            market_entry=True, signal_threshold=5.0, hold_ms=10000,
        ),
        # Ultra with TP/SL
        ExecutionStrategy(
            label='ultra_tpsl_tp4_sl2_z35',
            description='Chase 1x3, TP=4t SL=2t, z>3.5, hold 60s',
            chase_entry=True, signal_threshold=3.5,
            hold_ms=60000, take_profit_ticks=4, stop_loss_ticks=2,
        ),
        ExecutionStrategy(
            label='ultra_tpsl_tp5_sl2_z40',
            description='Chase 1x3, TP=5t SL=2t, z>4.0, hold 60s',
            chase_entry=True, signal_threshold=4.0,
            hold_ms=60000, take_profit_ticks=5, stop_loss_ticks=2,
        ),
    ]

    # ── GROUP F: Time-of-Day ──
    strategies['time_of_day'] = [
        ExecutionStrategy(
            label='chase_z25_10s_prime',
            description='Chase 1x3, z>2.5, hold 10s, prime hours only',
            chase_entry=True, signal_threshold=2.5,
            hold_ms=10000, prime_hours=True,
        ),
        ExecutionStrategy(
            label='chase_z25_10s_open',
            description='Chase 1x3, z>2.5, hold 10s, first 30min',
            chase_entry=True, signal_threshold=2.5,
            hold_ms=10000,
            time_window_start='09:30', time_window_end='10:00',
        ),
        ExecutionStrategy(
            label='chase_z25_10s_close',
            description='Chase 1x3, z>2.5, hold 10s, last hour',
            chase_entry=True, signal_threshold=2.5,
            hold_ms=10000,
            time_window_start='15:00', time_window_end='16:00',
        ),
    ]

    return strategies


# ============================================================
# Fill Simulator Interface
# ============================================================

def run_fill_sim(
    date_str: str,
    pred_file: Path,
    strategy: ExecutionStrategy,
    mbo_dir: Path = MBO_DIR,
    out_dir: Path = RESULTS_DIR,
) -> Optional[Dict]:
    """Run Rust fill_sim_cli for a single day + strategy.

    Returns parsed JSON result dict or None on failure.
    """
    if not BINARY.exists():
        log.error(f"fill_sim_cli binary not found: {BINARY}")
        return None

    mbo_file = mbo_dir / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = mbo_dir / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        log.warning(f"  No MBO file for {date_str}")
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
            log.warning(f"Sim failed {strategy.label}/{date_str}: {r.stderr[:300]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        log.warning(f"Timeout: {strategy.label}/{date_str}")
        return None
    except Exception as e:
        log.warning(f"Error {strategy.label}/{date_str}: {e}")
        return None


def run_strategy_sweep(
    pred_files: Dict[str, Path],
    strategies: List[ExecutionStrategy],
    workers: int = 8,
) -> Dict[str, Dict[str, Dict]]:
    """Run all strategies across all days in parallel.

    Returns: {strategy_label: {date_str: result_dict}}
    """
    # Create sub-directory for this run
    sim_out = RESULTS_DIR / f'sim_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for date_str, pred_file in sorted(pred_files.items()):
        for strategy in strategies:
            jobs.append({
                'date': date_str,
                'pred_file': pred_file,
                'strategy': strategy,
            })

    log.info(f"\nRunning {len(jobs)} sim jobs ({workers} workers)")
    log.info(f"  Days: {len(pred_files)}")
    log.info(f"  Strategies: {len(strategies)}")

    results = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_fill_sim,
                job['date'], job['pred_file'], job['strategy'],
                MBO_DIR, sim_out,
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
                log.warning(f"Job error: {e}")

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
# Analysis & Reporting (ES-specific)
# ============================================================

def compute_confidence_tiers(
    results_by_date: Dict[str, Dict],
) -> List[Dict]:
    """Analyze performance at different signal confidence tiers.

    Tiers: All, Top50%, Top25%, Top10%, Top5%, Top1%, Top0.5%
    """
    all_trades = []
    for date_str, res in results_by_date.items():
        if 'trades' not in res:
            continue
        for trade in res['trades']:
            tc = dict(trade)
            tc['date'] = date_str
            sig = abs(trade.get('signal_strength', trade.get('entry_signal',
                      trade.get('signal', 0))))
            tc['signal_mag'] = sig
            all_trades.append(tc)

    if not all_trades:
        return []

    signal_mags = np.array([t['signal_mag'] for t in all_trades])
    pnls = np.array([t.get('pnl_dollars', 0) for t in all_trades])

    tiers = []
    tier_specs = [
        ('All', 0), ('Top50%', 50), ('Top25%', 75),
        ('Top10%', 90), ('Top5%', 95), ('Top1%', 99), ('Top0.5%', 99.5),
    ]

    for tier_name, pct in tier_specs:
        if pct > 0:
            thresh = np.percentile(signal_mags, pct)
            mask = signal_mags >= thresh
        else:
            mask = np.ones(len(all_trades), dtype=bool)
            thresh = 0

        tier_pnls = pnls[mask]
        if len(tier_pnls) == 0:
            continue

        tier_trades = [t for t, m in zip(all_trades, mask) if m]

        # Long/short breakdown
        long_pnls = []
        short_pnls = []
        for t in tier_trades:
            pnl_val = t.get('pnl_dollars', 0)
            sig_val = t.get('signal_strength', t.get('entry_signal',
                           t.get('signal', 0)))
            if sig_val > 0:
                long_pnls.append(pnl_val)
            else:
                short_pnls.append(pnl_val)

        # Daily P&L for Sortino
        daily_pnl = defaultdict(float)
        for t in tier_trades:
            daily_pnl[t.get('date', 'unknown')] += t.get('pnl_dollars', 0)
        daily_vals = list(daily_pnl.values())

        n_days = len(daily_vals)
        if n_days > 1:
            avg_daily = np.mean(daily_vals)
            downside = [min(0, x) for x in daily_vals]
            downside_std = np.std(downside)
            sortino = (avg_daily / max(downside_std, 1e-8)) * np.sqrt(252)
        else:
            sortino = 0.0
            avg_daily = daily_vals[0] if daily_vals else 0

        gross_profit = sum(p for p in tier_pnls if p > 0)
        gross_loss = abs(sum(p for p in tier_pnls if p < 0))

        # MFE/MAE
        mfes = [t.get('mfe_ticks', 0) for t in tier_trades if 'mfe_ticks' in t]
        maes = [t.get('mae_ticks', 0) for t in tier_trades if 'mae_ticks' in t]

        mfe_mae = {}
        if mfes:
            mfe_mae = {
                'mfe_mean': round(float(np.mean(mfes)), 2),
                'mfe_median': round(float(np.median(mfes)), 2),
                'mae_mean': round(float(np.mean(maes)), 2) if maes else 0,
                'mae_median': round(float(np.median(maes)), 2) if maes else 0,
                'mfe_mae_ratio': round(
                    float(np.mean(mfes)) / max(float(np.mean(maes)), 0.01), 2
                ) if maes else 0,
            }

        # Max drawdown from daily P&L
        cum = np.cumsum(daily_vals) if daily_vals else np.array([0])
        peak = np.maximum.accumulate(cum)
        max_dd = abs(float((cum - peak).min())) if len(cum) > 0 else 0

        tiers.append({
            'tier': tier_name,
            'threshold': round(float(thresh), 3),
            'n_trades': len(tier_pnls),
            'n_days': n_days,
            'trades_per_day': round(len(tier_pnls) / max(n_days, 1), 1),
            'total_pnl': round(float(tier_pnls.sum()), 2),
            'avg_daily_pnl': round(float(avg_daily), 2),
            'avg_trade_pnl': round(float(tier_pnls.mean()), 2),
            'avg_trade_ticks': round(float(tier_pnls.mean() / TICK_VALUE), 3),
            'win_rate': round(float(np.mean(tier_pnls > 0)), 4),
            'profit_factor': round(gross_profit / max(gross_loss, 0.01), 2),
            'sortino': round(sortino, 2),
            'max_dd': round(max_dd, 2),
            'mfe_mae': mfe_mae,
            'long': {
                'n_trades': len(long_pnls),
                'total_pnl': round(sum(long_pnls), 2),
                'avg_pnl': round(np.mean(long_pnls), 2) if long_pnls else 0,
                'win_rate': round(
                    float(np.mean(np.array(long_pnls) > 0)), 4
                ) if long_pnls else 0,
            },
            'short': {
                'n_trades': len(short_pnls),
                'total_pnl': round(sum(short_pnls), 2),
                'avg_pnl': round(np.mean(short_pnls), 2) if short_pnls else 0,
                'win_rate': round(
                    float(np.mean(np.array(short_pnls) > 0)), 4
                ) if short_pnls else 0,
            },
        })

    return tiers


def aggregate_strategy_results(
    results: Dict[str, Dict[str, Dict]],
    strategy_map: Dict[str, ExecutionStrategy],
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
        long_pnls = []
        short_pnls = []
        fill_times = []

        for date_str, res in sorted(date_results.items()):
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
                    sig = trade.get('signal_strength', trade.get('entry_signal',
                                   trade.get('signal', 0)))
                    if sig > 0:
                        long_pnls.append(pnl)
                    else:
                        short_pnls.append(pnl)
                    if 'time_to_fill_ms' in trade:
                        fill_times.append(trade['time_to_fill_ms'])

        n_days = len(date_results)
        if n_days == 0:
            continue

        win_rate = total_wins / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)
        avg_daily = np.mean(daily_pnls) if daily_pnls else 0
        daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 1e-8
        sharpe = (avg_daily / max(daily_std, 1e-8)) * np.sqrt(252)

        downside = [min(0, x) for x in daily_pnls]
        downside_std = np.std(downside) if downside else 1e-8
        sortino = (avg_daily / max(downside_std, 1e-8)) * np.sqrt(252)

        gross_profit = sum(p for p in all_trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_trade_pnls if p < 0))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_trade_ticks = avg_trade_pnl / TICK_VALUE

        avg_winner = (np.mean([p for p in all_trade_pnls if p > 0])
                      if any(p > 0 for p in all_trade_pnls) else 0)
        avg_loser = (np.mean([p for p in all_trade_pnls if p < 0])
                     if any(p < 0 for p in all_trade_pnls) else 0)

        # Max drawdown
        cum = np.cumsum(daily_pnls) if daily_pnls else np.array([0])
        peak = np.maximum.accumulate(cum)
        max_dd = abs(float((cum - peak).min())) if len(cum) > 0 else 0

        trades_per_day = total_trades / max(n_days, 1)
        avg_fill_time = np.mean(fill_times) if fill_times else 0

        # Confidence tiers
        tiers = compute_confidence_tiers(date_results)

        strategy = strategy_map.get(label)
        description = strategy.description if strategy else label

        summaries.append({
            'label': label,
            'description': description,
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
            'avg_fill_time_ms': round(avg_fill_time, 1),
            'commission_per_trade': COMMISSION_RT,
            'long_trades': len(long_pnls),
            'long_pnl': round(sum(long_pnls), 2),
            'short_trades': len(short_pnls),
            'short_pnl': round(sum(short_pnls), 2),
            'confidence_tiers': tiers,
            'daily_pnls': daily_pnls,
        })

    summaries.sort(key=lambda x: x['sortino'], reverse=True)
    return summaries


# ============================================================
# Reporting
# ============================================================

def print_comparison_table(summaries: List[Dict], title: str = "Strategy Comparison"):
    """Print formatted strategy comparison table."""
    log.info(f"\n{'=' * 150}")
    log.info(f" {title}")
    log.info(f"{'=' * 150}")

    if not summaries:
        log.info("  No results to display.")
        return

    header = (
        f"{'Strategy':<35} "
        f"{'Total P&L':>10} "
        f"{'Trades':>7} "
        f"{'T/Day':>6} "
        f"{'FillR':>6} "
        f"{'WinR':>6} "
        f"{'AvgTrd':>8} "
        f"{'Sharpe':>7} "
        f"{'Sortino':>8} "
        f"{'PF':>5} "
        f"{'MaxDD':>8} "
        f"{'Long$':>8} "
        f"{'Short$':>8} "
        f"{'FillMs':>7}"
    )
    log.info(header)
    log.info("-" * 150)

    for s in summaries:
        line = (
            f"{s['label']:<35} "
            f"${s['total_pnl']:>9,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['trades_per_day']:>5.1f} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"{s['sharpe']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f} "
            f"${s['long_pnl']:>7,.0f} "
            f"${s['short_pnl']:>7,.0f} "
            f"{s['avg_fill_time_ms']:>6.0f}ms"
        )
        log.info(line)

    n_days = summaries[0]['n_days'] if summaries else 0
    log.info(f"\n  ES Futures | Tick=$12.50 | Commission=$4.70 RT | {n_days} OOT days")


def print_strategy_detail(summaries: List[Dict], top_n: int = 5):
    """Print detailed analysis of top N strategies."""
    for rank, s in enumerate(summaries[:top_n]):
        log.info(f"\n{'=' * 80}")
        log.info(f"  #{rank + 1} STRATEGY: {s['label']}")
        log.info(f"  {s['description']}")
        log.info(f"{'=' * 80}")
        log.info(f"  Total P&L:         ${s['total_pnl']:,.2f}")
        log.info(f"  Annualized:        ${s['annualized_pnl']:,.0f}")
        log.info(f"  Days:              {s['n_days']}")
        log.info(f"  Total trades:      {s['n_trades']}")
        log.info(f"  Trades/day:        {s['trades_per_day']:.1f}")
        log.info(f"  Fill rate:         {s['fill_rate']:.1%}")
        log.info(f"  Win rate:          {s['win_rate']:.1%}")
        log.info(f"  Avg trade P&L:     ${s['avg_trade_pnl']:.2f} ({s['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Avg winner:        ${s['avg_winner']:.2f}")
        log.info(f"  Avg loser:         ${s['avg_loser']:.2f}")
        log.info(f"  Sharpe:            {s['sharpe']:.2f}")
        log.info(f"  Sortino:           {s['sortino']:.2f}")
        log.info(f"  Profit factor:     {s['profit_factor']:.2f}")
        log.info(f"  Max drawdown:      ${s['max_dd']:,.2f}")
        log.info(f"  Commission/trade:  ${s['commission_per_trade']:.2f}")
        log.info(f"  Long trades:       {s['long_trades']} -> ${s['long_pnl']:,.2f}")
        log.info(f"  Short trades:      {s['short_trades']} -> ${s['short_pnl']:,.2f}")

        if s.get('confidence_tiers'):
            log.info(f"\n  Confidence Tier Analysis:")
            log.info(f"  {'Tier':<10} {'Trades':>7} {'T/Day':>6} "
                     f"{'P&L':>10} {'AvgTrd':>10} "
                     f"{'WinR':>6} {'PF':>5} {'Sortino':>8} "
                     f"{'Long$':>8} {'Short$':>8}")
            log.info(f"  {'-' * 90}")
            for tier in s['confidence_tiers']:
                log.info(
                    f"  {tier['tier']:<10} "
                    f"{tier['n_trades']:>7} "
                    f"{tier['trades_per_day']:>5.1f} "
                    f"${tier['total_pnl']:>9,.0f} "
                    f"${tier['avg_trade_pnl']:>9.2f} "
                    f"{tier['win_rate']:>5.1%} "
                    f"{tier['profit_factor']:>5.2f} "
                    f"{tier['sortino']:>8.2f} "
                    f"${tier['long']['total_pnl']:>7,.0f} "
                    f"${tier['short']['total_pnl']:>7,.0f}"
                )

        # Per-day P&L
        if s.get('daily_pnls'):
            log.info(f"\n  Daily P&L:")
            for i, (d, pnl) in enumerate(
                zip(sorted(s.get('_dates', [f'day{j}' for j in range(len(s['daily_pnls']))])),
                    s['daily_pnls'])
            ):
                marker = "  " if pnl >= 0 else " *"
                log.info(f"   {marker} {d}: ${pnl:>8,.2f}")


def save_results_json(
    summaries: List[Dict],
    signal_meta: List[Dict],
    mode: str,
    pred_files: Dict[str, Path],
) -> Path:
    """Save comprehensive results to JSON."""
    out_data = []
    for s in summaries:
        entry = {k: v for k, v in s.items() if k != 'daily_pnls'}
        entry['daily_pnl_list'] = s['daily_pnls']
        out_data.append(entry)

    out_file = RESULTS_DIR / f'combined_exec_results_{mode}_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'mode': mode,
            'instrument': 'ES',
            'tick_value': TICK_VALUE,
            'commission_rt': COMMISSION_RT,
            'n_days': len(pred_files),
            'dates': sorted(pred_files.keys()),
            'signal_generation_meta': signal_meta,
            'strategies': out_data,
        }, f, indent=2, default=str)
    log.info(f"\nResults saved: {out_file}")
    return out_file


# ============================================================
# Prediction Preparation Pipeline
# ============================================================

def prepare_combined_predictions(
    mode: str = 'mamba-only',
    da_source: str = 'none',
    max_days: Optional[int] = None,
    entry_horizon: str = '1s',
    direction_horizon: str = '5s',
    conviction_horizon: str = '10s',
) -> Tuple[Dict[str, Path], List[Dict]]:
    """Prepare bar-indexed prediction files for fill_sim_cli.

    Pipeline:
      1. Discover Mamba v7 fold predictions
      2. Discover DA model predictions (if combined mode)
      3. For each OOT day:
         a. Load Mamba predictions (N, 3)
         b. Load event timestamps
         c. Generate combined bar signal
         d. Save as .npz for fill_sim_cli

    Returns:
        ({date_str: pred_npz_path}, [signal_generation_metadata])
    """
    log.info(f"\n{'=' * 60}")
    log.info(f"  Preparing Combined Predictions")
    log.info(f"  Mode: {mode} | DA: {da_source}")
    log.info(f"  Horizons: entry={entry_horizon}, dir={direction_horizon}, "
             f"conv={conviction_horizon}")
    log.info(f"{'=' * 60}")

    # Step 1: Discover predictions
    mamba_folds = discover_mamba_folds()
    if not mamba_folds:
        log.error("No Mamba v7 fold predictions found!")
        return {}, []

    da_folds = discover_da_predictions(da_source) if mode == 'combined' else {}

    # Step 2: Initialize signal generator
    sig_gen = CombinedSignalGenerator(
        mode=mode,
        da_source=da_source,
        entry_horizon=entry_horizon,
        direction_horizon=direction_horizon,
        conviction_horizon=conviction_horizon,
    )

    # Step 3: Process each day
    saved_files = {}
    signal_meta = []

    sorted_dates = sorted(mamba_folds.keys())
    if max_days:
        sorted_dates = sorted_dates[:max_days]

    for date_str in sorted_dates:
        fold_path = mamba_folds[date_str]

        # Check MBO file exists
        mbo_zst = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            log.warning(f"  {date_str}: no MBO file, skipping")
            continue

        # Check cache
        cache_key = f'{mode}_{da_source}_{entry_horizon}_{direction_horizon}_{conviction_horizon}'
        cache_file = PRED_CACHE_DIR / f'{cache_key}_{date_str}.npz'
        if cache_file.exists():
            saved_files[date_str] = cache_file
            signal_meta.append({'date': date_str, 'cached': True})
            log.info(f"  {date_str}: cached")
            continue

        # Load Mamba fold
        mamba_data = load_mamba_fold(fold_path)
        if mamba_data is None:
            continue

        # Load event timestamps
        event_file = None
        for edir in [EVENT_DIR_V3, EVENT_DIR_V2]:
            candidate = edir / f'{date_str}_mbo_events.npz'
            if candidate.exists():
                event_file = candidate
                break

        if event_file is None:
            log.warning(f"  {date_str}: no event file, skipping")
            continue

        try:
            ev_data = np.load(str(event_file), allow_pickle=True)
            timestamps = ev_data['timestamps']
        except Exception as e:
            log.warning(f"  {date_str}: failed to load events: {e}")
            continue

        # Load DA fold if needed
        da_data = None
        if mode == 'combined' and date_str in da_folds:
            try:
                da_data_raw = np.load(str(da_folds[date_str]), allow_pickle=True)
                da_data = {k: da_data_raw[k] for k in da_data_raw.keys()}
            except Exception as e:
                log.warning(f"  {date_str}: failed to load DA: {e}")

        # Generate combined signal
        bar_signal, meta = sig_gen.generate_bar_signal(
            mamba_data, da_data, timestamps, date_str,
        )

        # Save for fill_sim_cli
        np.savez_compressed(str(cache_file), predictions=bar_signal)
        saved_files[date_str] = cache_file
        signal_meta.append(meta)

        n_signals = meta.get('n_bar_signals', 0)
        sig_type = meta.get('signal_type', 'unknown')
        log.info(f"  {date_str}: {n_signals} bar signals ({sig_type})")

        del ev_data, timestamps
        gc.collect()

    log.info(f"\n  Prepared {len(saved_files)} prediction files for fill_sim_cli")
    return saved_files, signal_meta


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combined CNN-Mamba + PatchTST Execution System v1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ES Futures: tick=$12.50, commission=$4.70 RT, point_value=$50

Modes:
  mamba-only      Pure Mamba v7 signal (conviction horizon)
  multi-horizon   Mamba multi-horizon composite (1s timing, 5s direction, 10s conviction)
  combined        Mamba magnitude x DA directional agreement (requires DA predictions)

Examples:
    # Mamba-only with all strategies
    python combined_execution_v1.py --mode mamba-only

    # Multi-horizon composite
    python combined_execution_v1.py --mode multi-horizon

    # Combined with LGBM DA filter
    python combined_execution_v1.py --mode combined --da-source lgbm

    # Quick test: 3 days, confidence sweep only
    python combined_execution_v1.py --max-days 3 --groups confidence_sweep

    # Dry run to see strategies
    python combined_execution_v1.py --dry-run
        """,
    )
    parser.add_argument('--mode', type=str, default='mamba-only',
                        choices=['mamba-only', 'multi-horizon', 'combined'],
                        help='Signal generation mode (default: mamba-only)')
    parser.add_argument('--da-source', type=str, default='none',
                        choices=['none', 'lgbm', 'patchtst'],
                        help='DA model source for combined mode (default: none)')
    parser.add_argument('--entry-horizon', type=str, default='1s',
                        choices=['1s', '5s', '10s'],
                        help='Horizon for entry timing precision (default: 1s)')
    parser.add_argument('--direction-horizon', type=str, default='5s',
                        choices=['1s', '5s', '10s'],
                        help='Horizon for trade direction (default: 5s)')
    parser.add_argument('--conviction-horizon', type=str, default='10s',
                        choices=['1s', '5s', '10s'],
                        help='Horizon for position conviction (default: 10s)')
    parser.add_argument('--groups', type=str, default='all',
                        help='Strategy groups (comma-separated). '
                             'Options: confidence_sweep, hold_time, exit_strategies, '
                             'entry_modes, ultra_selective, time_of_day')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel sim workers (default: 8)')
    parser.add_argument('--max-days', type=int, default=None,
                        help='Limit number of OOT days')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show strategies without running sims')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear prediction cache before running')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top strategies to show in detail')

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("COMBINED CNN-MAMBA + PatchTST EXECUTION SYSTEM v1")
    log.info("=" * 80)
    log.info(f"  Instrument:        ES (E-mini S&P 500)")
    log.info(f"  Tick value:        ${TICK_VALUE}")
    log.info(f"  Commission RT:     ${COMMISSION_RT}")
    log.info(f"  Mode:              {args.mode}")
    log.info(f"  DA source:         {args.da_source}")
    log.info(f"  Entry horizon:     {args.entry_horizon}")
    log.info(f"  Direction horizon: {args.direction_horizon}")
    log.info(f"  Conviction horizon:{args.conviction_horizon}")
    log.info(f"  Binary:            {BINARY}")
    log.info(f"  Workers:           {args.workers}")
    log.info(f"  Max days:          {args.max_days or 'all'}")
    log.info("=" * 80)

    # Clear cache if requested
    if args.clear_cache:
        import shutil
        if PRED_CACHE_DIR.exists():
            shutil.rmtree(PRED_CACHE_DIR)
            PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            log.info("  Prediction cache cleared")

    # Build strategy suite
    all_strategies = build_combined_strategy_suite()

    # Filter groups
    if args.groups == 'all':
        groups_to_run = list(all_strategies.keys())
    else:
        groups_to_run = [g.strip() for g in args.groups.split(',')]

    strategies_flat = []
    strategy_map = {}
    for group in groups_to_run:
        if group not in all_strategies:
            log.warning(f"Unknown strategy group: {group}")
            continue
        for s in all_strategies[group]:
            if s.label not in strategy_map:  # Deduplicate
                strategies_flat.append(s)
                strategy_map[s.label] = s

    log.info(f"\nStrategy groups: {groups_to_run}")
    log.info(f"Total strategies: {len(strategies_flat)}")

    if args.dry_run:
        log.info("\n--- DRY RUN: Strategy List ---")
        for group in groups_to_run:
            if group not in all_strategies:
                continue
            log.info(f"\n  Group: {group}")
            for s in all_strategies[group]:
                cli = ' '.join(s.to_cli_args())
                log.info(f"    {s.label:<35} {s.description}")
                log.info(f"      CLI: {cli}")
        log.info(f"\nDry run complete. {len(strategies_flat)} strategies.")
        return

    # Check binary
    if not BINARY.exists():
        log.error(f"fill_sim_cli binary not found: {BINARY}")
        log.error("Build: cd rust_cache_builder && cargo build --release --bin fill_sim_cli")
        sys.exit(1)

    # Prepare predictions
    pred_files, signal_meta = prepare_combined_predictions(
        mode=args.mode,
        da_source=args.da_source,
        max_days=args.max_days,
        entry_horizon=args.entry_horizon,
        direction_horizon=args.direction_horizon,
        conviction_horizon=args.conviction_horizon,
    )

    if not pred_files:
        log.error("No prediction files prepared. Check data paths.")
        sys.exit(1)

    # Run strategy sweep
    results = run_strategy_sweep(
        pred_files, strategies_flat, workers=args.workers,
    )

    if not results:
        log.error("No sim results. Check binary and MBO files.")
        sys.exit(1)

    # Aggregate and report
    summaries = aggregate_strategy_results(results, strategy_map)

    # Print group-by-group comparison
    for group in groups_to_run:
        if group not in all_strategies:
            continue
        group_labels = {s.label for s in all_strategies[group]}
        group_summaries = [s for s in summaries if s['label'] in group_labels]
        if group_summaries:
            print_comparison_table(
                group_summaries,
                f"{args.mode.upper()} -- {group.replace('_', ' ').title()}"
            )

    # Print overall ranking
    print_comparison_table(
        summaries,
        f"{args.mode.upper()} -- ALL STRATEGIES (Sorted by Sortino)"
    )

    # Print top strategies detail
    print_strategy_detail(summaries, top_n=args.top_n)

    # Save results
    out_file = save_results_json(summaries, signal_meta, args.mode, pred_files)

    # Final summary
    log.info(f"\n{'=' * 80}")
    log.info(f"  EXECUTION COMPLETE")
    log.info(f"{'=' * 80}")
    log.info(f"  Mode:          {args.mode}")
    log.info(f"  Days tested:   {len(pred_files)}")
    log.info(f"  Strategies:    {len(strategies_flat)}")
    if summaries:
        best = summaries[0]
        log.info(f"  Best strategy: {best['label']} "
                 f"(Sortino={best['sortino']:.2f}, "
                 f"P&L=${best['total_pnl']:,.0f}, "
                 f"WinR={best['win_rate']:.1%})")
    log.info(f"  Results:       {out_file}")
    log.info(f"  Log:           {_log_file}")
    log.info(f"{'=' * 80}")


if __name__ == '__main__':
    main()
