#!/usr/bin/env python3
"""
Cross-Validate v7 — Mamba v7 Robustness Check
================================================
Tests the TOP strategies discovered on CNN-Mamba v2 (Feb 23-27)
against Mamba v7 predictions (Mar 1-13) on different dates.

If strategies are profitable on BOTH models + date ranges, signal is likely real.
If only profitable on v2, we're likely overfitting.

Strategies tested (8 total):
  1. momentum_3_z2.5_30s          — #1 strategy from v4/v5
  2. momentum_2_z3.0_midday       — momentum + time filter
  3. agree_all3_z3.0_midday       — multi-horizon + time filter
  4. agree_all3_z5.0_bracket_wide — high conviction + bracket
  5. midday_z2.5_30s              — time filter only (control)
  6. open30_z2.5_30s              — open session only
  7. bracket_wide_z5.0            — high conviction bracket
  8. baseline_z2.5_30s            — no filter (control)

Data:
  Predictions: Mamba v7, 11 folds (Mar 1-13), skip fold_05 (<1000 events)
  Fill sim: Rust fill_sim_cli with real MBO data

Usage:
    python cross_validate_v7.py
    python cross_validate_v7.py --workers 8
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
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT = Path(__file__).resolve().parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v3'
PRED_DIR = LVL3_ROOT / 'output' / 'mamba_v7_tiny_smart_v3_mar_apr'
RESULTS_DIR = LVL3_ROOT / 'execution' / 'results' / 'cross_validate_v7'
PRED_CACHE_DIR = LVL3_ROOT / 'execution' / 'pred_cache' / 'cross_validate_v7'
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

# ── Minimum events threshold ──
MIN_EVENTS = 1000

# ── Fold Discovery ──
FOLD_FILES = sorted(PRED_DIR.glob('fold_*_oot_predictions.npz'))

# ── Timestamp ──
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──
_log_file = str(RESULTS_DIR / f'cross_validate_v7_{_ts}.log')
log = logging.getLogger('cross_validate_v7')
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
    """Load a Mamba v7 fold prediction file."""
    try:
        data = np.load(str(fold_path), allow_pickle=True)
        oot_path = str(data['oot_files'][0])
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]
        n_samples = data['predictions'].shape[0]

        if n_samples < MIN_EVENTS:
            log.info(f"  Skipping {fold_path.name}: only {n_samples} events (< {MIN_EVENTS})")
            return None

        result = {
            'predictions': data['predictions'].astype(np.float64),  # (N, 3)
            'labels': data['labels'].astype(np.float64),            # (N, 3)
            'date_str': date_str,
            'n_samples': n_samples,
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
# Signal Generation Core (identical to v4/v5)
# ============================================================

def map_predictions_to_bars(
    predictions: np.ndarray,
    event_timestamps: np.ndarray,
    date_str: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map per-window predictions to bar indices. Returns (bar_indices, predictions_rth)."""
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
    """Apply expanding z-score to bar-level signal (no lookahead)."""
    if running_stats is None:
        running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}

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
    """Compute expanding z-scores for all 3 horizons independently."""
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
# Signal Generators
# ============================================================

def generate_base_10s_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """Standard 10s horizon z-score signal (baseline)."""
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
    """Momentum filter: only signal when last N consecutive predictions agree on direction."""
    preds_10s = fold_data['predictions'][:, 2]
    bar_idx, preds_rth = map_predictions_to_bars(preds_10s, event_timestamps, date_str)
    z_signal, new_stats = expanding_zscore_bar_signal(preds_rth, bar_idx, running_stats)

    nonzero_bars = np.where(z_signal != 0)[0]
    if len(nonzero_bars) < n_consecutive:
        return np.zeros(N_RTH_BARS, dtype=np.float64), new_stats

    momentum_signal = np.zeros(N_RTH_BARS, dtype=np.float64)
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


def generate_agree_all3_signal(
    fold_data: Dict,
    event_timestamps: np.ndarray,
    date_str: str,
    running_stats_list: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """All 3 horizons agree on direction. Signal magnitude from 10s z-score."""
    z_signals, new_stats = multi_horizon_expanding_zscore(
        fold_data['predictions'], event_timestamps, date_str, running_stats_list
    )

    agreement = (z_signals[0] != 0) & (z_signals[1] != 0) & (z_signals[2] != 0)
    ref_sign = np.sign(z_signals[0])
    agreement &= (np.sign(z_signals[1]) == ref_sign)
    agreement &= (np.sign(z_signals[2]) == ref_sign)

    result = np.where(agreement, z_signals[2], 0.0)
    return result, new_stats


# ============================================================
# Strategy Specification
# ============================================================

@dataclass
class StrategySpec:
    """Strategy specification."""
    label: str
    description: str
    signal_type: str  # 'base_10s', 'momentum_2', 'momentum_3', 'agree_all3'
    z_threshold: float
    hold_ms: int = 30000
    stop_loss_ticks: Optional[int] = None
    take_profit_ticks: Optional[int] = None
    time_window_start: str = ""
    time_window_end: str = ""

    def to_cli_args(self) -> List[str]:
        args = [
            '--chase-entry',
            '--chase-max-ticks', '1',
            '--chase-max-reprices', '3',
            '--signal-threshold', str(self.z_threshold),
            '--hold-ms', str(self.hold_ms),
        ]
        if self.stop_loss_ticks is not None:
            args.extend(['--stop-loss-ticks', str(self.stop_loss_ticks)])
        if self.take_profit_ticks is not None:
            args.extend(['--take-profit-ticks', str(self.take_profit_ticks)])
        if self.time_window_start:
            args.extend(['--time-window-start', self.time_window_start])
            args.extend(['--time-window-end', self.time_window_end])
        args.append('--quiet')
        return args


def build_strategies() -> List[StrategySpec]:
    """Build the 8 top strategies to cross-validate."""
    return [
        # #1 from v4/v5: momentum 3 consecutive, z>2.5, 30s hold
        StrategySpec(
            label='momentum_3_z2.5_30s',
            description='Momentum 3 consecutive agree, z>2.5, 30s hold',
            signal_type='momentum_3',
            z_threshold=2.5, hold_ms=30000,
        ),
        # Momentum 2 + midday filter
        StrategySpec(
            label='momentum_2_z3.0_midday',
            description='Momentum 2 consecutive, z>3.0, midday 10:00-14:00',
            signal_type='momentum_2',
            z_threshold=3.0, hold_ms=30000,
            time_window_start='10:00', time_window_end='14:00',
        ),
        # All 3 horizons agree + midday
        StrategySpec(
            label='agree_all3_z3.0_midday',
            description='All 3 horizons agree, z>3.0, midday 10:00-14:00',
            signal_type='agree_all3',
            z_threshold=3.0, hold_ms=30000,
            time_window_start='10:00', time_window_end='14:00',
        ),
        # All 3 agree + high conviction bracket
        StrategySpec(
            label='agree_all3_z5.0_bracket_wide',
            description='All 3 agree, z>5.0, SL4/TP8, 60s hold',
            signal_type='agree_all3',
            z_threshold=5.0, hold_ms=60000,
            stop_loss_ticks=4, take_profit_ticks=8,
        ),
        # Midday time filter only (control)
        StrategySpec(
            label='midday_z2.5_30s',
            description='Base 10s, z>2.5, midday 10:00-14:00, 30s hold',
            signal_type='base_10s',
            z_threshold=2.5, hold_ms=30000,
            time_window_start='10:00', time_window_end='14:00',
        ),
        # Open 30min session
        StrategySpec(
            label='open30_z2.5_30s',
            description='Base 10s, z>2.5, open 09:30-10:00, 30s hold',
            signal_type='base_10s',
            z_threshold=2.5, hold_ms=30000,
            time_window_start='09:30', time_window_end='10:00',
        ),
        # High conviction bracket (no time filter)
        StrategySpec(
            label='bracket_wide_z5.0',
            description='Base 10s, z>5.0, SL4/TP8, 60s hold',
            signal_type='base_10s',
            z_threshold=5.0, hold_ms=60000,
            stop_loss_ticks=4, take_profit_ticks=8,
        ),
        # Baseline control (no filters)
        StrategySpec(
            label='baseline_z2.5_30s',
            description='Base 10s, z>2.5, 30s hold (no filter — control)',
            signal_type='base_10s',
            z_threshold=2.5, hold_ms=30000,
        ),
    ]


# ============================================================
# Signal Preparation
# ============================================================

def prepare_all_signals() -> Tuple[Dict[str, Dict[str, Path]], List[str]]:
    """Prepare prediction NPZ files for each signal type x date.

    Returns: (signal_cache {signal_type: {date: path}}, valid_dates)
    """
    log.info("\nPreparing prediction signals from Mamba v7...")

    folds_data = []
    for fp in FOLD_FILES:
        fd = load_fold(fp)
        if fd:
            folds_data.append(fd)
            log.info(f"  Loaded fold: {fd['date_str']} ({fd['n_samples']} samples) from {fp.name}")

    if not folds_data:
        log.error("No fold data loaded!")
        return {}, []

    folds_data.sort(key=lambda x: x['date_str'])
    valid_dates = [f['date_str'] for f in folds_data]
    log.info(f"  Valid dates ({len(valid_dates)}): {valid_dates}")

    timestamps_map = {}
    for fd in folds_data:
        ts = load_event_timestamps(fd['date_str'])
        if ts is not None:
            timestamps_map[fd['date_str']] = ts
        else:
            log.warning(f"  No event timestamps for {fd['date_str']}")

    signal_cache = {}

    # ── base_10s signals ──
    log.info("  Generating base_10s signals...")
    running_stats = None
    for fd in folds_data:
        date_str = fd['date_str']
        if date_str not in timestamps_map:
            continue
        cache_path = PRED_CACHE_DIR / f'base_10s_{date_str}.npz'
        if not cache_path.exists():
            z_signal, running_stats = generate_base_10s_signal(
                fd, timestamps_map[date_str], date_str, running_stats
            )
            np.savez_compressed(str(cache_path), predictions=z_signal)
            n_nz = int(np.count_nonzero(z_signal))
            log.info(f"    {date_str}: {n_nz} signals")
        else:
            data = np.load(str(cache_path))
            nz = data['predictions'][data['predictions'] != 0]
            if running_stats is None:
                running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
            running_stats['sum'] += float(np.sum(nz))
            running_stats['sq'] += float(np.sum(nz ** 2))
            running_stats['count'] += len(nz)
            log.info(f"    {date_str}: cached ({len(nz)} nonzero)")
        signal_cache.setdefault('base_10s', {})[date_str] = cache_path

    # ── momentum_2 signals ──
    log.info("  Generating momentum_2 signals...")
    running_stats = None
    for fd in folds_data:
        date_str = fd['date_str']
        if date_str not in timestamps_map:
            continue
        cache_path = PRED_CACHE_DIR / f'momentum_2_{date_str}.npz'
        if not cache_path.exists():
            z_signal, running_stats = generate_momentum_signal(
                fd, timestamps_map[date_str], date_str, 2, running_stats
            )
            np.savez_compressed(str(cache_path), predictions=z_signal)
            n_nz = int(np.count_nonzero(z_signal))
            log.info(f"    {date_str}: {n_nz} signals")
        else:
            data = np.load(str(cache_path))
            nz = data['predictions'][data['predictions'] != 0]
            if running_stats is None:
                running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
            running_stats['sum'] += float(np.sum(nz))
            running_stats['sq'] += float(np.sum(nz ** 2))
            running_stats['count'] += len(nz)
            log.info(f"    {date_str}: cached ({len(nz)} nonzero)")
        signal_cache.setdefault('momentum_2', {})[date_str] = cache_path

    # ── momentum_3 signals ──
    log.info("  Generating momentum_3 signals...")
    running_stats = None
    for fd in folds_data:
        date_str = fd['date_str']
        if date_str not in timestamps_map:
            continue
        cache_path = PRED_CACHE_DIR / f'momentum_3_{date_str}.npz'
        if not cache_path.exists():
            z_signal, running_stats = generate_momentum_signal(
                fd, timestamps_map[date_str], date_str, 3, running_stats
            )
            np.savez_compressed(str(cache_path), predictions=z_signal)
            n_nz = int(np.count_nonzero(z_signal))
            log.info(f"    {date_str}: {n_nz} signals")
        else:
            data = np.load(str(cache_path))
            nz = data['predictions'][data['predictions'] != 0]
            if running_stats is None:
                running_stats = {'sum': 0.0, 'sq': 0.0, 'count': 0}
            running_stats['sum'] += float(np.sum(nz))
            running_stats['sq'] += float(np.sum(nz ** 2))
            running_stats['count'] += len(nz)
            log.info(f"    {date_str}: cached ({len(nz)} nonzero)")
        signal_cache.setdefault('momentum_3', {})[date_str] = cache_path

    # ── agree_all3 signals ──
    log.info("  Generating agree_all3 signals...")
    running_stats_list = None
    for fd in folds_data:
        date_str = fd['date_str']
        if date_str not in timestamps_map:
            continue
        cache_path = PRED_CACHE_DIR / f'agree_all3_{date_str}.npz'
        if not cache_path.exists():
            z_signal, running_stats_list = generate_agree_all3_signal(
                fd, timestamps_map[date_str], date_str, running_stats_list
            )
            np.savez_compressed(str(cache_path), predictions=z_signal)
            n_nz = int(np.count_nonzero(z_signal))
            log.info(f"    {date_str}: {n_nz} signals")
        else:
            log.info(f"    {date_str}: cached")
            if running_stats_list is None:
                running_stats_list = [None, None, None]
        signal_cache.setdefault('agree_all3', {})[date_str] = cache_path

    del timestamps_map
    gc.collect()

    log.info(f"  Signal types prepared: {list(signal_cache.keys())}")
    return signal_cache, valid_dates


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
    strategies: List[StrategySpec],
    signal_cache: Dict[str, Dict[str, Path]],
    workers: int = 8,
) -> Dict[str, Dict[str, Dict]]:
    """Run all strategies across all days in parallel."""
    sim_out = RESULTS_DIR / f'sim_{_ts}'
    sim_out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for strat in strategies:
        date_files = signal_cache.get(strat.signal_type, {})
        for date_str, pred_file in sorted(date_files.items()):
            jobs.append({
                'date': date_str,
                'pred_file': pred_file,
                'strategy': strat,
            })

    log.info(f"\nRunning {len(jobs)} sim jobs ({workers} workers)")

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

            if done % 10 == 0 or done == len(jobs):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(jobs) - done) / max(rate, 0.01)
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, "
                         f"~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log.info(f"Sweep done: {done} jobs in {elapsed:.1f}s")
    return results


# ============================================================
# Analysis & Reporting
# ============================================================

def aggregate_results(
    results: Dict[str, Dict[str, Dict]],
    strategy_map: Dict[str, StrategySpec],
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
        profitable_days = 0

        for date_str, res in sorted(date_results.items()):
            dates.append(date_str)
            day_pnl = res.get('total_pnl_dollars', 0)
            total_pnl += day_pnl
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            daily_pnls.append(day_pnl)
            if day_pnl > 0:
                profitable_days += 1

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

        strat = strategy_map.get(label)

        summaries.append({
            'label': label,
            'description': strat.description if strat else '',
            'signal_type': strat.signal_type if strat else '',
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
            'profitable_days': profitable_days,
            'profitable_day_rate': round(profitable_days / max(n_days, 1), 4),
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
    log.info(f"\n{'=' * 170}")
    log.info(f" {title}")
    log.info(f"{'=' * 170}")

    if not summaries:
        log.info("  No results.")
        return

    header = (
        f"{'Strategy':<35} "
        f"{'Total P&L':>10} "
        f"{'Trades':>7} "
        f"{'T/Day':>6} "
        f"{'FillR':>6} "
        f"{'WinR':>6} "
        f"{'AvgTrd':>8} "
        f"{'Sortino':>8} "
        f"{'PF':>5} "
        f"{'MaxDD':>8} "
        f"{'ProfD':>6} "
        f"{'Long$':>8} "
        f"{'Short$':>8}"
    )
    log.info(header)
    log.info("-" * 170)

    for s in summaries:
        pnl_marker = '+' if s['total_pnl'] > 0 else ' '
        line = (
            f"{s['label']:<35} "
            f"{pnl_marker}${abs(s['total_pnl']):>8,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['trades_per_day']:>5.1f} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f} "
            f"{s['profitable_days']}/{s['n_days']:>2} "
            f"${s['long_pnl']:>7,.0f} "
            f"${s['short_pnl']:>7,.0f}"
        )
        log.info(line)

    n_days = summaries[0]['n_days'] if summaries else 0
    log.info(f"\n  Mamba v7 | ES Futures | Tick=$12.50 | Commission=$4.70 RT | {n_days} OOT days (Mar 2-13)")


def print_strategy_detail(summaries: List[Dict]):
    """Print detailed analysis of each strategy."""
    log.info(f"\n{'=' * 100}")
    log.info(f"  DETAILED STRATEGY RESULTS")
    log.info(f"{'=' * 100}")

    for rank, s in enumerate(summaries):
        profitable_marker = "PROFITABLE" if s['total_pnl'] > 0 else "UNPROFITABLE"
        log.info(f"\n  #{rank + 1} {s['label']} [{profitable_marker}]")
        log.info(f"  {s['description']}")
        log.info(f"  Total P&L:       ${s['total_pnl']:,.2f}")
        log.info(f"  Trades:          {s['n_trades']} ({s['trades_per_day']:.1f}/day)")
        log.info(f"  Fill rate:       {s['fill_rate']:.1%}")
        log.info(f"  Win rate:        {s['win_rate']:.1%}")
        log.info(f"  Avg trade:       ${s['avg_trade_pnl']:.2f} ({s['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Sortino:         {s['sortino']:.2f}")
        log.info(f"  Profit factor:   {s['profit_factor']:.2f}")
        log.info(f"  Max DD:          ${s['max_dd']:,.2f}")
        log.info(f"  Profitable days: {s['profitable_days']}/{s['n_days']} ({s['profitable_day_rate']:.0%})")
        log.info(f"  Long:            {s['long_trades']} trades -> ${s['long_pnl']:,.2f}")
        log.info(f"  Short:           {s['short_trades']} trades -> ${s['short_pnl']:,.2f}")
        log.info(f"  Daily P&L:")
        for date, pnl in sorted(s['daily_pnls'].items()):
            marker = '+' if pnl >= 0 else '-'
            log.info(f"    {date}: {marker}${abs(pnl):,.2f}")


def print_robustness_verdict(summaries: List[Dict]):
    """Print final robustness verdict."""
    log.info(f"\n{'=' * 100}")
    log.info(f"  ROBUSTNESS VERDICT — Mamba v7 Cross-Validation")
    log.info(f"{'=' * 100}")

    n_profitable = len([s for s in summaries if s['total_pnl'] > 0])
    n_total = len(summaries)

    log.info(f"\n  Profitable strategies: {n_profitable}/{n_total}")
    log.info(f"")

    # Key strategy verdicts
    key_strategies = ['momentum_3_z2.5_30s', 'baseline_z2.5_30s']
    for label in key_strategies:
        s = next((x for x in summaries if x['label'] == label), None)
        if s:
            status = "CONFIRMED" if s['total_pnl'] > 0 else "FAILED"
            log.info(f"  {label}: {status}")
            log.info(f"    P&L=${s['total_pnl']:,.2f}, Sortino={s['sortino']:.2f}, "
                     f"WR={s['win_rate']:.1%}, PF={s['profit_factor']:.2f}")

    log.info(f"")

    # Overall verdict
    momentum_3 = next((x for x in summaries if x['label'] == 'momentum_3_z2.5_30s'), None)
    if momentum_3 and momentum_3['total_pnl'] > 0:
        log.info(f"  VERDICT: momentum_3_z2.5_30s is ROBUST across models and dates.")
        log.info(f"           Signal is likely REAL, not overfit to CNN-Mamba v2.")
    elif momentum_3:
        log.info(f"  VERDICT: momentum_3_z2.5_30s FAILED on Mamba v7.")
        log.info(f"           Signal may be model-specific or date-dependent.")
        log.info(f"           Exercise CAUTION before live deployment.")
    else:
        log.info(f"  VERDICT: Could not test momentum_3_z2.5_30s. Check data.")

    # Compare all strategies
    log.info(f"\n  Strategy-by-strategy:")
    for s in summaries:
        status = "+" if s['total_pnl'] > 0 else "-"
        log.info(f"    [{status}] {s['label']:<35} P&L=${s['total_pnl']:>8,.2f}  "
                 f"Sortino={s['sortino']:>6.2f}  PF={s['profit_factor']:>5.2f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Cross-Validate v7 — Mamba v7 Robustness Check',
    )
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--clear-cache', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("CROSS-VALIDATE v7 — Mamba v7 Robustness Check")
    log.info("=" * 80)
    log.info(f"  Model:       Mamba v7 (tiny_smart_v3_mar_apr)")
    log.info(f"  Dates:       Mar 2-13, 2026 (10 trading days)")
    log.info(f"  Instrument:  ES (tick=$12.50, commission=$4.70 RT)")
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  Purpose:     Cross-validate top strategies from CNN-Mamba v2")
    log.info("=" * 80)

    if args.clear_cache:
        import shutil
        if PRED_CACHE_DIR.exists():
            shutil.rmtree(PRED_CACHE_DIR)
            PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("  Cache cleared")

    # Build strategies
    strategies = build_strategies()
    strategy_map = {s.label: s for s in strategies}

    log.info(f"\n  Strategies to test: {len(strategies)}")
    for s in strategies:
        log.info(f"    {s.label:<35} {s.description}")

    if args.dry_run:
        log.info(f"\nDry run: {len(strategies)} strategies.")
        return

    if not BINARY.exists():
        log.error(f"fill_sim_cli not found: {BINARY}")
        sys.exit(1)

    # ── Phase 1: Prepare signals ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 1: Signal Preparation (Mamba v7)")
    log.info(f"{'=' * 60}")

    signal_cache, valid_dates = prepare_all_signals()
    if not signal_cache:
        log.error("No signals prepared.")
        sys.exit(1)

    log.info(f"\n  Valid dates: {valid_dates}")
    log.info(f"  Signal types: {list(signal_cache.keys())}")

    # ── Phase 2: Fill simulation sweep ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 2: Fill Simulation Sweep ({len(strategies)} strategies x {len(valid_dates)} days)")
    log.info(f"{'=' * 60}")

    sim_results = run_strategy_sweep(strategies, signal_cache, workers=args.workers)

    # ── Phase 3: Aggregate and report ──
    log.info(f"\n{'=' * 60}")
    log.info(f"  PHASE 3: Analysis & Reporting")
    log.info(f"{'=' * 60}")

    summaries = aggregate_results(sim_results, strategy_map)

    # Results table
    print_results_table(summaries, "MAMBA v7 CROSS-VALIDATION — Sorted by Sortino")

    # Detailed per-strategy
    print_strategy_detail(summaries)

    # Robustness verdict
    print_robustness_verdict(summaries)

    # ── Save results ──
    out_file = RESULTS_DIR / f'cross_validate_v7_results_{_ts}.json'
    save_data = {
        'timestamp': _ts,
        'model': 'Mamba v7 (tiny_smart_v3_mar_apr)',
        'dates': valid_dates,
        'instrument': 'ES',
        'tick_value': TICK_VALUE,
        'commission_rt': COMMISSION_RT,
        'purpose': 'Cross-validate top strategies from CNN-Mamba v2 on different model + dates',
        'n_strategies_tested': len(summaries),
        'n_profitable': len([s for s in summaries if s['total_pnl'] > 0]),
        'strategies': summaries,
    }
    with open(out_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    log.info(f"\n{'=' * 80}")
    log.info(f"  CROSS-VALIDATE v7 COMPLETE")
    log.info(f"{'=' * 80}")
    log.info(f"  Strategies tested:  {len(summaries)}")
    log.info(f"  Profitable:         {len([s for s in summaries if s['total_pnl'] > 0])}")
    if summaries:
        best = summaries[0]
        log.info(f"  Best (Sortino):     {best['label']} "
                 f"(Sortino={best['sortino']:.2f}, P&L=${best['total_pnl']:,.0f})")
    log.info(f"  Results saved:      {out_file}")
    log.info(f"  Log file:           {_log_file}")
    log.info(f"{'=' * 80}")


if __name__ == '__main__':
    main()
