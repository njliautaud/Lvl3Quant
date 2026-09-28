#!/usr/bin/env python3
"""
Execution Strategy Tester — Market Replay Framework
====================================================
Comprehensive strategy comparison using the Rust MBO fill simulator.

Orchestrates the fill_sim_cli binary across multiple execution strategies,
prediction sources (LGBM, Mamba, CNN), and signal thresholds to find the
optimal execution parameters for Monday market open.

Architecture:
  1. Load model predictions (.npz) and convert to bar-indexed z-scored signals
  2. Run Rust fill_sim_cli (event-by-event FIFO queue sim on raw MBO data)
  3. Aggregate results across days and strategies
  4. Output comprehensive comparison table with all key metrics

Models supported:
  - LGBM: directional, DA=68.8% at Top1%. Predictions = confidence scores.
  - Mamba: multi-horizon (1s/5s/10s) magnitude predictions. IC_10s=0.07.
  - CNN1D: concat IC_10s=0.132 (champion). Per-event directional signals.

Usage:
    python execution_strategy_tester.py
    python execution_strategy_tester.py --model lgbm --workers 8
    python execution_strategy_tester.py --model mamba --strategies chase,midprice
    python execution_strategy_tester.py --model all --dry-run
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

import numpy as np

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR_V2 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v2'
EVENT_DIR_V3 = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
EVENT_DIR_RAW = LVL3_ROOT / 'data' / 'processed' / 'mbo_events'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── NQ Futures Constants ──
TICK_SIZE = 0.25       # NQ tick size in points
TICK_VALUE = 5.00      # $5 per tick for NQ (not ES which is $12.50)
POINT_VALUE = 20.00    # $20 per point for NQ
COMMISSION_RT = 4.12   # Round-trip commission per contract
BARS_PER_SEC = 10
BAR_NS = 100_000_000  # 100ms in nanoseconds
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# ── Logging ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = str(RESULTS_DIR / f'exec_strategy_test_{_ts}.log')

log = logging.getLogger('exec_strategy_tester')
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
# Model Prediction Loaders
# ============================================================

def rth_start_ns_for_date(date_str: str) -> int:
    """Compute RTH start timestamp (9:30 AM ET) for date YYYYMMDD."""
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    d = datetime(year, month, day)
    # Simple DST: EDT (UTC-4) Mar-Nov, EST (UTC-5) Nov-Mar
    # For 2025-2026 season
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


class PredictionLoader:
    """Load and convert model predictions to bar-indexed z-scored signals."""

    @staticmethod
    def find_prediction_files(model: str) -> Dict[str, Path]:
        """Find all prediction NPZ files for a model.

        Returns: {date_str: path} mapping
        """
        pred_dirs = {
            'cnn': [
                LVL3_ROOT / 'output' / 'cnn_s76_smart_v3',
                LVL3_ROOT / 'output' / 'cnn_s76_smart_v2',
            ],
            'mamba': [
                LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr',
                LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3',
            ],
            'lgbm': [
                LVL3_ROOT / 'output' / 'lgbm_smart_v3',
                LVL3_ROOT / 'output' / 'lgbm_smart_v2',
            ],
            'patchtst': [
                LVL3_ROOT / 'output' / 'patchtst_sliding60d_smart_v2',
            ],
        }

        files = {}
        dirs = pred_dirs.get(model, [])

        for d in dirs:
            if not d.exists():
                continue
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

        # Also check for concat predictions
        concat_patterns = [
            LVL3_ROOT / 'data' / f'{model}*concat*predictions.npz',
            LVL3_ROOT / 'data' / f'cnn_s76_concat_oot_predictions.npz',
        ]
        for pattern in concat_patterns:
            for f in LVL3_ROOT.glob(pattern.name):
                if f.exists():
                    log.info(f"  Found concat file: {f.name}")

        return files

    @staticmethod
    def load_cnn_predictions(pred_file: Path, horizon: str = '10s') -> Tuple[np.ndarray, str]:
        """Load CNN/LGBM predictions from fold file.

        Returns (predictions_array, date_str)
        """
        data = np.load(str(pred_file), allow_pickle=True)

        # Multi-horizon format: preds_1s, preds_5s, preds_10s
        horizon_key = f'preds_{horizon}'
        if horizon_key in data:
            preds = data[horizon_key]
        elif 'predictions' in data:
            # Single array or (N, 3) format
            preds = data['predictions']
            if preds.ndim == 2:
                col_map = {'1s': 0, '5s': 1, '10s': 2}
                preds = preds[:, col_map.get(horizon, 2)]
        else:
            raise ValueError(f"No predictions found in {pred_file}")

        # Extract date
        if 'oot_files' in data:
            oot_path = str(data['oot_files'][0])
            basename = oot_path.replace('\\', '/').split('/')[-1]
            date_str = basename.split('_')[0]
        else:
            date_str = pred_file.stem.split('_')[0]

        return preds.astype(np.float64), date_str

    @staticmethod
    def load_mamba_predictions(pred_file: Path) -> Tuple[Dict[str, np.ndarray], str]:
        """Load Mamba multi-horizon predictions.

        Returns ({horizon: predictions}, date_str)
        """
        data = np.load(str(pred_file), allow_pickle=True)
        preds = data['predictions']  # (N, 3) for 1s/5s/10s

        horizons = {}
        if preds.ndim == 2 and preds.shape[1] == 3:
            horizons['1s'] = preds[:, 0].astype(np.float64)
            horizons['5s'] = preds[:, 1].astype(np.float64)
            horizons['10s'] = preds[:, 2].astype(np.float64)
        else:
            horizons['10s'] = preds.astype(np.float64)

        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]

        return horizons, date_str

    @staticmethod
    def convert_to_bar_signal(
        predictions: np.ndarray,
        event_timestamps: np.ndarray,
        date_str: str,
        window: int = 500,
        stride: int = 250,
        expanding_zscore: bool = True,
        running_stats: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Convert per-event/window predictions to bar-indexed z-scored signal.

        Args:
            predictions: (N,) raw model predictions
            event_timestamps: (N_events,) nanosecond timestamps
            date_str: YYYYMMDD
            window: sliding window size (0 = per-event predictions)
            stride: window stride
            expanding_zscore: use walk-forward expanding z-score
            running_stats: dict with 'sum', 'sq', 'count' for expanding z-score

        Returns:
            (bar_signal of shape (N_RTH_BARS,), updated running_stats)
        """
        n_events = len(event_timestamps)
        n_preds = len(predictions)

        if window > 0:
            # Window-based predictions: map to event timestamps
            starts = np.arange(0, n_events - window + 1, stride, dtype=np.int64)
            label_idxs = starts + window - 1
            # Trim to match prediction count
            if len(label_idxs) > n_preds:
                label_idxs = label_idxs[:n_preds]
            elif n_preds > len(label_idxs):
                predictions = predictions[:len(label_idxs)]
            pred_timestamps = event_timestamps[label_idxs]
        else:
            # Per-event predictions
            if n_preds > n_events:
                predictions = predictions[:n_events]
            elif n_events > n_preds:
                event_timestamps = event_timestamps[:n_preds]
            pred_timestamps = event_timestamps[:len(predictions)]

        # Map to RTH bar indices
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
            return bar_preds.astype(np.float64), running_stats or {}

        # Apply expanding z-score (walk-forward, no lookahead)
        if running_stats is None:
            running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

        zscore_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
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
                zscore_preds[i] = (v - mean) / std

        running_stats = {'sum': rs, 'sq': rsq, 'count': cnt}
        return zscore_preds, running_stats


# ============================================================
# Execution Strategy Definitions
# ============================================================

@dataclass
class ExecutionStrategy:
    """Complete specification of an execution strategy for the fill simulator."""
    label: str
    description: str

    # Entry mode
    market_entry: bool = False
    mid_price_entry: bool = False
    chase_entry: bool = False
    chase_max_ticks: int = 2
    chase_max_reprices: int = 5
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

    # Latency model
    latency_ms: int = 0
    exit_slippage: float = 0.0

    # Time filters
    prime_hours: bool = False
    time_window_start: str = ""
    time_window_end: str = ""

    # Queue filters
    max_wait_bars: Optional[int] = None

    def to_cli_args(self) -> List[str]:
        """Convert to fill_sim_cli command-line arguments."""
        args = []

        # Entry mode
        if self.market_entry:
            args.append('--market-entry')
        elif self.mid_price_entry:
            args.append('--mid-price-entry')
        elif self.chase_entry:
            args.append('--chase-entry')
            args.extend(['--chase-max-ticks', str(self.chase_max_ticks)])
            args.extend(['--chase-max-reprices', str(self.chase_max_reprices)])
            args.extend(['--chase-interval-ms', str(self.chase_interval_ms)])
            if self.chase_force_cross:
                args.append('--chase-force-cross')

        # Signal threshold
        args.extend(['--signal-threshold', str(self.signal_threshold)])

        # Hold time
        args.extend(['--hold-ms', str(self.hold_ms)])

        # Exit rules
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

        # Latency
        if self.latency_ms > 0:
            args.extend(['--latency-ms', str(self.latency_ms)])
        if self.exit_slippage > 0:
            args.extend(['--exit-slippage', str(self.exit_slippage)])

        # Time filters
        if self.prime_hours:
            args.append('--prime-hours')
        if self.time_window_start:
            args.extend(['--time-window-start', self.time_window_start])
            args.extend(['--time-window-end', self.time_window_end])

        # Queue / wait
        if self.max_wait_bars is not None:
            args.extend(['--max-wait-bars', str(self.max_wait_bars)])

        args.append('--quiet')
        return args


def build_strategy_suite() -> Dict[str, List[ExecutionStrategy]]:
    """Build the comprehensive strategy test suite.

    Returns dict of {group_name: [strategies]}
    """
    strategies = {}

    # ── GROUP A: Entry Mode Comparison (same exit, vary entry) ──
    strategies['entry_modes'] = [
        # A1: Passive limit at best bid/ask (default, no flags = passive limit)
        ExecutionStrategy(
            label='passive_z20_10s',
            description='Passive limit at BBO, z>2.0, hold 10s',
            signal_threshold=2.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='passive_z25_10s',
            description='Passive limit at BBO, z>2.5, hold 10s',
            signal_threshold=2.5, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='passive_z30_10s',
            description='Passive limit at BBO, z>3.0, hold 10s',
            signal_threshold=3.0, hold_ms=10000,
        ),

        # A2: Mid-price limit
        ExecutionStrategy(
            label='midprice_z20_10s',
            description='Mid-price limit, z>2.0, hold 10s',
            mid_price_entry=True,
            signal_threshold=2.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='midprice_z25_10s',
            description='Mid-price limit, z>2.5, hold 10s',
            mid_price_entry=True,
            signal_threshold=2.5, hold_ms=10000,
        ),

        # A3: Chase entry (passive then reprice)
        ExecutionStrategy(
            label='chase1x3_z20_10s',
            description='Chase 1 tick, 3 reprices, z>2.0, hold 10s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='chase2x5_z20_10s',
            description='Chase 2 ticks, 5 reprices, z>2.0, hold 10s',
            chase_entry=True, chase_max_ticks=2, chase_max_reprices=5,
            signal_threshold=2.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='chase1x3_force_z20_10s',
            description='Chase 1x3 then force cross, z>2.0, hold 10s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            chase_force_cross=True,
            signal_threshold=2.0, hold_ms=10000,
        ),

        # A4: Market entry (baseline cost)
        ExecutionStrategy(
            label='market_z20_10s',
            description='Market order, z>2.0, hold 10s',
            market_entry=True,
            signal_threshold=2.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='market_z30_10s',
            description='Market order, z>3.0, hold 10s',
            market_entry=True,
            signal_threshold=3.0, hold_ms=10000,
        ),
    ]

    # ── GROUP B: Confidence-Gated Entry (vary threshold) ──
    strategies['confidence_gates'] = [
        ExecutionStrategy(
            label=f'chase_z{int(z*10):02d}_10s',
            description=f'Chase 1x3, z>{z}, hold 10s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=z, hold_ms=10000,
        )
        for z in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    ]

    # ── GROUP C: Exit Strategy Comparison ──
    strategies['exit_strategies'] = [
        # C1: Time-based exit (vary holding period)
        ExecutionStrategy(
            label='chase_z20_1s',
            description='Chase 1x3, z>2.0, hold 1s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=1000,
        ),
        ExecutionStrategy(
            label='chase_z20_5s',
            description='Chase 1x3, z>2.0, hold 5s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=5000,
        ),
        ExecutionStrategy(
            label='chase_z20_10s_exit',
            description='Chase 1x3, z>2.0, hold 10s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='chase_z20_30s',
            description='Chase 1x3, z>2.0, hold 30s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=30000,
        ),
        ExecutionStrategy(
            label='chase_z20_60s',
            description='Chase 1x3, z>2.0, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=60000,
        ),

        # C2: Signal-flip exit
        ExecutionStrategy(
            label='flip_z20_5min',
            description='Chase 1x3, z>2.0, signal-flip exit, max 5min',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=300000, signal_flip_exit=True,
        ),
        ExecutionStrategy(
            label='flip_z25_5min',
            description='Chase 1x3, z>2.5, signal-flip exit, max 5min',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.5, hold_ms=300000, signal_flip_exit=True,
        ),

        # C3: Conviction exit (delayed signal-flip)
        ExecutionStrategy(
            label='conviction_z20_50bars',
            description='Chase 1x3, z>2.0, conviction exit 50 bars (5s)',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=300000,
            conviction_exit_bars=50, conviction_exit_mag=1.0,
        ),

        # C4: TP/SL combinations
        ExecutionStrategy(
            label='tpsl_tp3_sl2_z20',
            description='Chase 1x3, TP=3t, SL=2t, z>2.0, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=60000,
            take_profit_ticks=3, stop_loss_ticks=2,
        ),
        ExecutionStrategy(
            label='tpsl_tp4_sl2_z20',
            description='Chase 1x3, TP=4t, SL=2t, z>2.0, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=60000,
            take_profit_ticks=4, stop_loss_ticks=2,
        ),
        ExecutionStrategy(
            label='tpsl_tp3_sl1_z25',
            description='Chase 1x3, TP=3t, SL=1t, z>2.5, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.5, hold_ms=60000,
            take_profit_ticks=3, stop_loss_ticks=1,
        ),
        ExecutionStrategy(
            label='tpsl_tp5_sl2_z25',
            description='Chase 1x3, TP=5t, SL=2t, z>2.5, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.5, hold_ms=60000,
            take_profit_ticks=5, stop_loss_ticks=2,
        ),

        # C5: Trailing stop
        ExecutionStrategy(
            label='trail_2t_z20_60s',
            description='Chase 1x3, trailing 2 ticks, z>2.0, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=60000,
            trailing_ticks=2,
        ),
        ExecutionStrategy(
            label='trail_3t_z20_60s',
            description='Chase 1x3, trailing 3 ticks, z>2.0, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=60000,
            trailing_ticks=3,
        ),

        # C6: Ratchet stop (adaptive)
        ExecutionStrategy(
            label='ratchet_z20_60s',
            description='Chase 1x3, ratchet stop, z>2.0, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=60000,
            ratchet_stop=True,
        ),

        # C7: Combined: signal-flip + TP/SL
        ExecutionStrategy(
            label='flip_tp4_sl2_z20',
            description='Chase 1x3, flip exit, TP=4t, SL=2t, z>2.0',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=300000, signal_flip_exit=True,
            take_profit_ticks=4, stop_loss_ticks=2,
        ),
        ExecutionStrategy(
            label='flip_tp3_sl1_z25',
            description='Chase 1x3, flip exit, TP=3t, SL=1t, z>2.5',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.5, hold_ms=300000, signal_flip_exit=True,
            take_profit_ticks=3, stop_loss_ticks=1,
        ),
    ]

    # ── GROUP D: Latency Sensitivity ──
    strategies['latency'] = [
        ExecutionStrategy(
            label=f'chase_z20_10s_lat{lat}',
            description=f'Chase 1x3, z>2.0, hold 10s, latency={lat}ms',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=10000,
            latency_ms=lat,
        )
        for lat in [0, 5, 10, 20, 50]
    ]

    # ── GROUP E: Time-of-Day Filters ──
    strategies['time_of_day'] = [
        ExecutionStrategy(
            label='chase_z20_10s_prime',
            description='Chase 1x3, z>2.0, hold 10s, prime hours only',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=10000,
            prime_hours=True,
        ),
        ExecutionStrategy(
            label='chase_z20_10s_open',
            description='Chase 1x3, z>2.0, hold 10s, first 30min',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=10000,
            time_window_start='09:30', time_window_end='10:00',
        ),
        ExecutionStrategy(
            label='chase_z20_10s_close',
            description='Chase 1x3, z>2.0, hold 10s, last hour',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=2.0, hold_ms=10000,
            time_window_start='15:00', time_window_end='16:00',
        ),
    ]

    # ── GROUP F: Ultra-Selective (Monday deployment candidates) ──
    strategies['ultra_selective'] = [
        ExecutionStrategy(
            label='ultra_chase_z35_10s',
            description='Chase 1x3, z>3.5, hold 10s — ultra selective',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=3.5, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='ultra_chase_z40_10s',
            description='Chase 1x3, z>4.0, hold 10s — ultra selective',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=4.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='ultra_chase_z50_10s',
            description='Chase 1x3, z>5.0, hold 10s — ultra selective',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=5.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='ultra_flip_z35_5min',
            description='Chase 1x3, z>3.5, signal-flip, max 5min',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=3.5, hold_ms=300000, signal_flip_exit=True,
        ),
        ExecutionStrategy(
            label='ultra_flip_z40_5min',
            description='Chase 1x3, z>4.0, signal-flip, max 5min',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=4.0, hold_ms=300000, signal_flip_exit=True,
        ),
        # Market entry ultra-selective (guaranteed fills on best signals)
        ExecutionStrategy(
            label='ultra_market_z40_10s',
            description='Market order, z>4.0, hold 10s',
            market_entry=True,
            signal_threshold=4.0, hold_ms=10000,
        ),
        ExecutionStrategy(
            label='ultra_market_z50_10s',
            description='Market order, z>5.0, hold 10s',
            market_entry=True,
            signal_threshold=5.0, hold_ms=10000,
        ),
        # Ultra-selective with TP/SL
        ExecutionStrategy(
            label='ultra_tpsl_tp4_sl2_z35',
            description='Chase 1x3, TP=4t SL=2t, z>3.5, hold 60s',
            chase_entry=True, chase_max_ticks=1, chase_max_reprices=3,
            signal_threshold=3.5, hold_ms=60000,
            take_profit_ticks=4, stop_loss_ticks=2,
        ),
    ]

    # ── GROUP G: Passive-then-Aggressive (wait then cross) ──
    strategies['passive_then_aggressive'] = [
        # Wait 1s passive, then force cross
        ExecutionStrategy(
            label='wait10_force_z20_10s',
            description='Wait 10 bars (1s), then force cross, z>2.0, hold 10s',
            chase_entry=True, chase_max_ticks=0, chase_max_reprices=0,
            chase_force_cross=True, chase_interval_ms=1000,
            signal_threshold=2.0, hold_ms=10000,
            max_wait_bars=10,
        ),
        ExecutionStrategy(
            label='wait30_force_z20_10s',
            description='Wait 30 bars (3s), then force cross, z>2.0, hold 10s',
            chase_entry=True, chase_max_ticks=0, chase_max_reprices=0,
            chase_force_cross=True, chase_interval_ms=3000,
            signal_threshold=2.0, hold_ms=10000,
            max_wait_bars=30,
        ),
        ExecutionStrategy(
            label='wait50_force_z25_10s',
            description='Wait 50 bars (5s), then force cross, z>2.5, hold 10s',
            chase_entry=True, chase_max_ticks=0, chase_max_reprices=0,
            chase_force_cross=True, chase_interval_ms=5000,
            signal_threshold=2.5, hold_ms=10000,
            max_wait_bars=50,
        ),
    ]

    return strategies


# ============================================================
# Fill Sim Runner
# ============================================================

def run_single_sim(
    date_str: str,
    pred_file: Path,
    strategy: ExecutionStrategy,
    mbo_dir: Path = MBO_DIR,
    sim_out_dir: Path = RESULTS_DIR,
) -> Optional[Dict]:
    """Run one Rust fill_sim job for a single day + strategy.

    Returns parsed JSON result or None on failure.
    """
    mbo_file = mbo_dir / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = mbo_dir / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = sim_out_dir / f'{strategy.label}_{date_str}.json'

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
    if not BINARY.exists():
        log.error(f"fill_sim_cli binary not found: {BINARY}")
        log.error("Build it: cd rust_cache_builder && cargo build --release --bin fill_sim_cli")
        return {}

    # Create sim output directory for this run
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
                run_single_sim,
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
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, ~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log.info(f"Sweep done: {done} jobs in {elapsed:.1f}s")
    return results


# ============================================================
# Analysis & Reporting
# ============================================================

def compute_mfe_mae(trades: List[Dict]) -> Dict:
    """Compute MFE/MAE statistics from trade list."""
    mfes = [t.get('mfe_ticks', 0) for t in trades if 'mfe_ticks' in t]
    maes = [t.get('mae_ticks', 0) for t in trades if 'mae_ticks' in t]
    if not mfes:
        return {}
    return {
        'mfe_mean': round(float(np.mean(mfes)), 2),
        'mfe_median': round(float(np.median(mfes)), 2),
        'mfe_p90': round(float(np.percentile(mfes, 90)), 2),
        'mae_mean': round(float(np.mean(maes)), 2),
        'mae_median': round(float(np.median(maes)), 2),
        'mae_p90': round(float(np.percentile(maes, 90)), 2),
        'mfe_mae_ratio': round(float(np.mean(mfes)) / max(float(np.mean(maes)), 0.01), 2),
    }


def compute_confidence_tiers(results_by_date: Dict[str, Dict]) -> List[Dict]:
    """Analyze performance at different signal confidence tiers."""
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
    for tier_name, pct in [('All', 0), ('Top50%', 50), ('Top25%', 75),
                           ('Top10%', 90), ('Top5%', 95), ('Top1%', 99)]:
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

        # Daily P&L for Sortino
        daily_pnl = {}
        for t in tier_trades:
            d = t.get('date', 'unknown')
            daily_pnl[d] = daily_pnl.get(d, 0) + t.get('pnl_dollars', 0)
        daily_vals = list(daily_pnl.values())

        if len(daily_vals) > 1:
            avg_daily = np.mean(daily_vals)
            downside = [min(0, x) for x in daily_vals]
            downside_std = np.std(downside) if downside else 1e-8
            sortino = (avg_daily / max(downside_std, 1e-8)) * np.sqrt(252)
        else:
            sortino = 0.0

        gross_profit = sum(p for p in tier_pnls if p > 0)
        gross_loss = abs(sum(p for p in tier_pnls if p < 0))

        tiers.append({
            'tier': tier_name,
            'threshold': round(float(thresh), 3),
            'n_trades': len(tier_pnls),
            'total_pnl': round(float(tier_pnls.sum()), 2),
            'avg_trade_pnl': round(float(tier_pnls.mean()), 2),
            'avg_trade_ticks': round(float(tier_pnls.mean() / TICK_VALUE), 3),
            'win_rate': round(float(np.mean(tier_pnls > 0)), 4),
            'profit_factor': round(gross_profit / max(gross_loss, 0.01), 2),
            'sortino': round(sortino, 2),
            'mfe_mae': compute_mfe_mae(tier_trades),
        })

    return tiers


def aggregate_strategy_results(
    results: Dict[str, Dict[str, Dict]],
    strategies: Dict[str, ExecutionStrategy],
) -> List[Dict]:
    """Aggregate per-day sim results into per-strategy summaries.

    Returns sorted list of strategy summary dicts.
    """
    summaries = []

    for label, date_results in results.items():
        total_pnl = 0.0
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0
        daily_pnls = []
        all_trade_pnls = []
        fill_times = []
        queue_positions = []

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
                    if 'time_to_fill_ms' in trade:
                        fill_times.append(trade['time_to_fill_ms'])
                    if 'queue_position_at_post' in trade:
                        queue_positions.append(trade['queue_position_at_post'])

        n_days = len(date_results)
        if n_days == 0:
            continue

        # Core metrics
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
        avg_winner = np.mean([p for p in all_trade_pnls if p > 0]) if any(p > 0 for p in all_trade_pnls) else 0
        avg_loser = np.mean([p for p in all_trade_pnls if p < 0]) if any(p < 0 for p in all_trade_pnls) else 0

        # Max drawdown
        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = abs((cum - peak).min()) if len(cum) > 0 else 0

        # Fill time stats
        avg_fill_time = np.mean(fill_times) if fill_times else 0
        median_fill_time = np.median(fill_times) if fill_times else 0

        # Trades per day
        trades_per_day = total_trades / max(n_days, 1)

        # Confidence tiers
        tiers = compute_confidence_tiers(date_results)

        strategy = strategies.get(label)
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
            'sharpe_daily': round(sharpe, 3),
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
            'median_fill_time_ms': round(median_fill_time, 1),
            'confidence_tiers': tiers,
            'daily_pnls': daily_pnls,
        })

    summaries.sort(key=lambda x: x['sortino'], reverse=True)
    return summaries


def print_comparison_table(summaries: List[Dict], title: str = "Strategy Comparison"):
    """Print formatted strategy comparison table."""
    log.info(f"\n{'=' * 140}")
    log.info(f" {title}")
    log.info(f"{'=' * 140}")

    if not summaries:
        log.info("  No results to display.")
        return

    n_days = summaries[0]['n_days'] if summaries else 0

    # Header
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
        f"{'FillMs':>7}"
    )
    log.info(header)
    log.info("-" * 140)

    for s in summaries:
        line = (
            f"{s['label']:<35} "
            f"${s['total_pnl']:>9,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['trades_per_day']:>5.1f} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"{s['sharpe_daily']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f} "
            f"{s['avg_fill_time_ms']:>6.0f}ms"
        )
        log.info(line)

    log.info(f"\n  Based on {n_days} OOT trading days.")


def print_top_strategies_detail(summaries: List[Dict], top_n: int = 5):
    """Print detailed analysis of top N strategies."""
    for rank, s in enumerate(summaries[:top_n]):
        log.info(f"\n{'=' * 80}")
        log.info(f"  #{rank + 1} STRATEGY: {s['label']}")
        log.info(f"  Description: {s['description']}")
        log.info(f"{'=' * 80}")
        log.info(f"  Total P&L:         ${s['total_pnl']:,.2f}")
        log.info(f"  Annualized:        ${s['annualized_pnl']:,.0f}")
        log.info(f"  Days:              {s['n_days']}")
        log.info(f"  Total trades:      {s['n_trades']}")
        log.info(f"  Trades/day:        {s['trades_per_day']:.1f}")
        log.info(f"  Signals sent:      {s['n_signals']}")
        log.info(f"  Fill rate:         {s['fill_rate']:.1%}")
        log.info(f"  Win rate:          {s['win_rate']:.1%}")
        log.info(f"  Avg daily P&L:     ${s['avg_daily_pnl']:.2f}")
        log.info(f"  Avg trade P&L:     ${s['avg_trade_pnl']:.2f} ({s['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Avg winner:        ${s['avg_winner']:.2f}")
        log.info(f"  Avg loser:         ${s['avg_loser']:.2f}")
        log.info(f"  Sharpe (daily):    {s['sharpe_daily']:.2f}")
        log.info(f"  Sortino:           {s['sortino']:.2f}")
        log.info(f"  Profit factor:     {s['profit_factor']:.2f}")
        log.info(f"  Max drawdown:      ${s['max_dd']:,.2f}")
        log.info(f"  Avg fill time:     {s['avg_fill_time_ms']:.0f}ms")
        log.info(f"  Median fill time:  {s['median_fill_time_ms']:.0f}ms")

        if s.get('confidence_tiers'):
            log.info(f"\n  Confidence Tier Analysis:")
            log.info(f"  {'Tier':<10} {'Trades':>7} {'P&L':>10} {'AvgTrd':>10} "
                     f"{'WinR':>6} {'PF':>5} {'Sortino':>8}")
            log.info(f"  {'-' * 65}")
            for tier in s['confidence_tiers']:
                log.info(
                    f"  {tier['tier']:<10} "
                    f"{tier['n_trades']:>7} "
                    f"${tier['total_pnl']:>9,.0f} "
                    f"${tier['avg_trade_pnl']:>9.2f} "
                    f"{tier['win_rate']:>5.1%} "
                    f"{tier['profit_factor']:>5.2f} "
                    f"{tier['sortino']:>8.2f}"
                )

            # MFE/MAE for top tier
            for tier in s['confidence_tiers']:
                if tier.get('mfe_mae') and tier['tier'] in ('All', 'Top10%'):
                    mm = tier['mfe_mae']
                    log.info(f"\n  MFE/MAE ({tier['tier']}):")
                    log.info(f"    MFE: mean={mm.get('mfe_mean', 0):.1f}t, "
                             f"median={mm.get('mfe_median', 0):.1f}t, "
                             f"p90={mm.get('mfe_p90', 0):.1f}t")
                    log.info(f"    MAE: mean={mm.get('mae_mean', 0):.1f}t, "
                             f"median={mm.get('mae_median', 0):.1f}t, "
                             f"p90={mm.get('mae_p90', 0):.1f}t")
                    log.info(f"    MFE/MAE ratio: {mm.get('mfe_mae_ratio', 0):.2f}")


def print_entry_vs_exit_matrix(summaries: List[Dict]):
    """Print a matrix of entry mode vs exit mode performance."""
    log.info(f"\n{'=' * 100}")
    log.info("  ENTRY MODE vs EXIT MODE MATRIX (Sortino)")
    log.info(f"{'=' * 100}")

    # Classify each strategy
    entry_modes = {}
    for s in summaries:
        label = s['label']
        if 'market' in label:
            entry = 'Market'
        elif 'midprice' in label:
            entry = 'MidPrice'
        elif 'chase' in label or 'force' in label or 'wait' in label:
            entry = 'Chase'
        elif 'passive' in label:
            entry = 'Passive'
        else:
            entry = 'Other'

        if entry not in entry_modes:
            entry_modes[entry] = []
        entry_modes[entry].append(s)

    for entry, strats in sorted(entry_modes.items()):
        strats.sort(key=lambda x: x['sortino'], reverse=True)
        log.info(f"\n  {entry} Entry:")
        for s in strats[:5]:
            log.info(f"    {s['label']:<35} Sortino={s['sortino']:>6.2f}  "
                     f"P&L=${s['total_pnl']:>8,.0f}  "
                     f"FillR={s['fill_rate']:>5.1%}  "
                     f"Trades={s['n_trades']}")


def save_results_json(
    summaries: List[Dict],
    model: str,
    pred_files: Dict[str, Path],
) -> Path:
    """Save comprehensive results to JSON."""
    out_data = []
    for s in summaries:
        entry = {k: v for k, v in s.items() if k != 'daily_pnls'}
        entry['daily_pnl_list'] = s['daily_pnls']
        out_data.append(entry)

    out_file = RESULTS_DIR / f'exec_strategy_results_{model}_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'model': model,
            'n_days': len(pred_files),
            'dates': sorted(pred_files.keys()),
            'tick_value': TICK_VALUE,
            'commission_rt': COMMISSION_RT,
            'strategies': out_data,
        }, f, indent=2, default=str)
    log.info(f"\nResults saved: {out_file}")
    return out_file


# ============================================================
# Prediction Preparation Pipeline
# ============================================================

def prepare_predictions_for_model(
    model: str,
    horizon: str = '10s',
    max_days: Optional[int] = None,
) -> Dict[str, Path]:
    """Prepare bar-indexed z-scored prediction files for fill_sim_cli.

    Args:
        model: 'cnn', 'mamba', 'lgbm', 'patchtst'
        horizon: prediction horizon to use
        max_days: limit number of days (for testing)

    Returns: {date_str: pred_npz_path}
    """
    log.info(f"\nPreparing predictions for model={model}, horizon={horizon}")

    pred_files = PredictionLoader.find_prediction_files(model)
    if not pred_files:
        log.warning(f"No prediction files found for model={model}")
        return {}

    log.info(f"  Found {len(pred_files)} fold prediction files")

    # Determine event data directory based on model
    if model in ('mamba',):
        event_dir = EVENT_DIR_V3
        window, stride = 500, 250
    else:
        event_dir = EVENT_DIR_V2
        window, stride = 500, 250

    saved_files = {}
    running_stats = None

    sorted_files = sorted(pred_files.items())
    if max_days:
        sorted_files = sorted_files[:max_days]

    for date_str, pred_file in sorted_files:
        # Check if MBO file exists
        mbo_zst = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            log.warning(f"  {date_str}: no MBO file, skipping")
            continue

        # Check cache
        cache_file = PRED_CACHE_DIR / f'{model}_{horizon}_{date_str}.npz'
        if cache_file.exists():
            saved_files[date_str] = cache_file
            continue

        # Load event timestamps
        event_file = event_dir / f'{date_str}_mbo_events.npz'
        if not event_file.exists():
            # Try other event dirs
            for alt_dir in [EVENT_DIR_V3, EVENT_DIR_V2, EVENT_DIR_RAW]:
                alt_file = alt_dir / f'{date_str}_mbo_events.npz'
                if alt_file.exists():
                    event_file = alt_file
                    break
            else:
                log.warning(f"  {date_str}: no event file, skipping")
                continue

        try:
            ev_data = np.load(str(event_file), allow_pickle=True)
            timestamps = ev_data['timestamps']

            # Load predictions
            if model == 'mamba':
                horizons_dict, _ = PredictionLoader.load_mamba_predictions(pred_file)
                preds = horizons_dict.get(horizon, horizons_dict.get('10s'))
            else:
                preds, _ = PredictionLoader.load_cnn_predictions(pred_file, horizon)

            # Convert to bar signal
            bar_signal, running_stats = PredictionLoader.convert_to_bar_signal(
                preds, timestamps, date_str,
                window=window, stride=stride,
                expanding_zscore=True,
                running_stats=running_stats,
            )

            # Save
            np.savez_compressed(str(cache_file), predictions=bar_signal)
            saved_files[date_str] = cache_file

            n_nonzero = np.count_nonzero(bar_signal)
            z_abs = np.abs(bar_signal[bar_signal != 0])
            log.info(f"  {date_str}: {n_nonzero} signals, "
                     f"max_z={bar_signal.max():.2f}, min_z={bar_signal.min():.2f}, "
                     f"mean|z|={z_abs.mean():.2f}" if len(z_abs) > 0 else
                     f"  {date_str}: 0 signals")

            del ev_data, timestamps
            gc.collect()

        except Exception as e:
            log.warning(f"  {date_str}: error: {e}")
            continue

    log.info(f"  Prepared {len(saved_files)} prediction files")
    return saved_files


# ============================================================
# Multi-Horizon Strategy (Mamba-specific)
# ============================================================

def prepare_multi_horizon_signal(
    model: str = 'mamba',
    timing_horizon: str = '1s',
    direction_horizon: str = '5s',
    conviction_horizon: str = '10s',
    max_days: Optional[int] = None,
) -> Dict[str, Path]:
    """Prepare multi-horizon composite signal for Mamba.

    Strategy: Use 1s for timing, 5s for direction, 10s for conviction.
    Composite signal = sign(5s) * |1s| * (1 + |10s|)
    Only trigger when all horizons agree on direction.

    Returns: {date_str: pred_npz_path}
    """
    log.info(f"\nPreparing multi-horizon signal: "
             f"timing={timing_horizon}, dir={direction_horizon}, "
             f"conviction={conviction_horizon}")

    pred_files = PredictionLoader.find_prediction_files(model)
    if not pred_files:
        return {}

    event_dir = EVENT_DIR_V3
    saved_files = {}
    running_stats = None

    sorted_files = sorted(pred_files.items())
    if max_days:
        sorted_files = sorted_files[:max_days]

    for date_str, pred_file in sorted_files:
        mbo_zst = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            continue

        cache_file = PRED_CACHE_DIR / f'{model}_multihorizon_{date_str}.npz'
        if cache_file.exists():
            saved_files[date_str] = cache_file
            continue

        event_file = event_dir / f'{date_str}_mbo_events.npz'
        if not event_file.exists():
            continue

        try:
            ev_data = np.load(str(event_file), allow_pickle=True)
            timestamps = ev_data['timestamps']
            horizons_dict, _ = PredictionLoader.load_mamba_predictions(pred_file)

            n_events = len(timestamps)
            n_preds = min(len(v) for v in horizons_dict.values())

            # Get all horizon predictions
            p_timing = horizons_dict.get(timing_horizon, horizons_dict['1s'])[:n_preds]
            p_dir = horizons_dict.get(direction_horizon, horizons_dict['5s'])[:n_preds]
            p_conv = horizons_dict.get(conviction_horizon, horizons_dict['10s'])[:n_preds]

            # Composite: all horizons must agree on direction
            agree_mask = (np.sign(p_timing) == np.sign(p_dir)) & \
                         (np.sign(p_dir) == np.sign(p_conv))

            # Composite signal: sign * |timing| * (1 + |conviction|)
            composite = np.where(
                agree_mask,
                np.sign(p_dir) * np.abs(p_timing) * (1.0 + np.abs(p_conv)),
                0.0
            )

            # Convert to bar signal
            bar_signal, running_stats = PredictionLoader.convert_to_bar_signal(
                composite, timestamps, date_str,
                window=500, stride=250,
                expanding_zscore=True,
                running_stats=running_stats,
            )

            np.savez_compressed(str(cache_file), predictions=bar_signal)
            saved_files[date_str] = cache_file

            n_nonzero = np.count_nonzero(bar_signal)
            n_agree = np.sum(agree_mask)
            log.info(f"  {date_str}: {n_agree}/{n_preds} agree, "
                     f"{n_nonzero} bar signals")

            del ev_data
            gc.collect()

        except Exception as e:
            log.warning(f"  {date_str}: error: {e}")
            continue

    log.info(f"  Prepared {len(saved_files)} multi-horizon files")
    return saved_files


# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Execution Strategy Tester — Market Replay Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all strategies with CNN model (default)
    python execution_strategy_tester.py

    # Run with Mamba multi-horizon predictions
    python execution_strategy_tester.py --model mamba

    # Run specific strategy groups
    python execution_strategy_tester.py --groups entry_modes,exit_strategies

    # Dry run: show strategies without running
    python execution_strategy_tester.py --dry-run

    # Run with latency sensitivity analysis
    python execution_strategy_tester.py --groups latency --workers 12

    # Quick test on 3 days
    python execution_strategy_tester.py --max-days 3 --groups entry_modes
        """,
    )
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mamba', 'lgbm', 'patchtst', 'all'],
                        help='Model to use (default: cnn)')
    parser.add_argument('--horizon', type=str, default='10s',
                        choices=['1s', '5s', '10s'],
                        help='Prediction horizon (default: 10s)')
    parser.add_argument('--groups', type=str, default='all',
                        help='Strategy groups to run (comma-separated). '
                             'Options: entry_modes, confidence_gates, exit_strategies, '
                             'latency, time_of_day, ultra_selective, passive_then_aggressive')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel sim workers (default: 8)')
    parser.add_argument('--max-days', type=int, default=None,
                        help='Limit number of OOT days (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show strategies without running sims')
    parser.add_argument('--multi-horizon', action='store_true',
                        help='Use Mamba multi-horizon composite signal')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Use cached prediction files')

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("EXECUTION STRATEGY TESTER — Market Replay Framework")
    log.info("=" * 80)
    log.info(f"  Model:         {args.model}")
    log.info(f"  Horizon:       {args.horizon}")
    log.info(f"  Binary:        {BINARY}")
    log.info(f"  MBO dir:       {MBO_DIR}")
    log.info(f"  Workers:       {args.workers}")
    log.info(f"  Max days:      {args.max_days or 'all'}")
    log.info(f"  Multi-horizon: {args.multi_horizon}")
    log.info(f"  Tick value:    ${TICK_VALUE} (NQ)")
    log.info(f"  Commission RT: ${COMMISSION_RT}")
    log.info("=" * 80)

    # Build strategy suite
    all_strategies = build_strategy_suite()

    # Filter strategy groups
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
                cli_args = ' '.join(s.to_cli_args())
                log.info(f"    {s.label:<35} {s.description}")
                log.info(f"      CLI: {cli_args}")
        log.info(f"\nDry run complete. {len(strategies_flat)} strategies would be tested.")
        return

    # Check binary
    if not BINARY.exists():
        log.error(f"fill_sim_cli binary not found: {BINARY}")
        log.error("Build: cd rust_cache_builder && cargo build --release --bin fill_sim_cli")
        sys.exit(1)

    # Determine models to run
    models_to_run = [args.model] if args.model != 'all' else ['cnn', 'mamba']

    for model in models_to_run:
        log.info(f"\n{'#' * 80}")
        log.info(f"  MODEL: {model.upper()}")
        log.info(f"{'#' * 80}")

        # Prepare predictions
        if args.multi_horizon and model == 'mamba':
            pred_files = prepare_multi_horizon_signal(
                model=model, max_days=args.max_days,
            )
        else:
            pred_files = prepare_predictions_for_model(
                model=model, horizon=args.horizon, max_days=args.max_days,
            )

        if not pred_files:
            log.warning(f"No prediction files for {model}. Skipping.")
            continue

        # Run strategy sweep
        results = run_strategy_sweep(
            pred_files, strategies_flat, workers=args.workers,
        )

        if not results:
            log.error(f"No sim results for {model}. Check binary and MBO files.")
            continue

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
                    f"{model.upper()} — {group.replace('_', ' ').title()}"
                )

        # Print overall ranking
        print_comparison_table(summaries, f"{model.upper()} — ALL STRATEGIES (Sorted by Sortino)")

        # Print top strategies detail
        print_top_strategies_detail(summaries, top_n=5)

        # Print entry vs exit matrix
        print_entry_vs_exit_matrix(summaries)

        # Save results
        save_results_json(summaries, model, pred_files)

    log.info(f"\nDone. Log: {_log_file}")


if __name__ == '__main__':
    main()
