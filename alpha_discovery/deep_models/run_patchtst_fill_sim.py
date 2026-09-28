#!/usr/bin/env python3
"""
PatchTST Predictions -> Rust MBO Fill Simulator
================================================
Converts PatchTST walk-forward OOT predictions to the format expected
by the Rust fill_sim_cli binary, then runs a sweep of execution configs.

PatchTST predictions:
  - Raw regression values (not probabilities)
  - Shape: (n_windows, 3) for horizons [1s, 5s, 10s]
  - Column index 2 = 10s prediction
  - Window: 500 events, stride: 250 events

Fill sim expects:
  - Per-day NPZ with 'predictions' key
  - Shape: (N_RTH_BARS,) = (234000,)
  - Z-scored directional signal indexed by 100ms RTH bar

Usage:
    python run_patchtst_fill_sim.py
    python run_patchtst_fill_sim.py --workers 8
"""

import sys
import gc
import json
import time
import argparse
import subprocess
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PATCHTST_DIR = LVL3_ROOT / 'output' / 'patchtst_sliding60d_smart_v2'
EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v2'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'patchtst_fill_sim_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'patchtst_fill_sim_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PatchTST config (must match train_event_patchtst.py) ──
WINDOW = 500
STRIDE = 250
HORIZONS = ['1s', '5s', '10s']
PRED_COL_10S = 2  # Column index for 10s predictions

TICK_VALUE = 12.50
BARS_PER_SEC = 10
BAR_NS = 100_000_000  # 100ms in nanoseconds
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# DST boundary: Nov 2, 2025 06:00 UTC
DST_END_2025_NS = 1_762_056_000_000_000_000

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = str(RESULTS_DIR / f'patchtst_fill_sim_{_ts}.log')

log = logging.getLogger('patchtst_fill_sim')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Folds to use (skip folds with tiny sample counts) ──
USABLE_FOLDS = [0, 1, 2, 3, 5]  # Fold 4 = Sunday (315 samples), Fold 6 = partial (601 samples)

# ── Execution configs ──
EXECUTION_CONFIGS = [
    # GROUP 1: Chase entry, fixed hold
    {'threshold': 1.5, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z15_10s'},
    {'threshold': 2.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z20_10s'},
    {'threshold': 2.5, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z25_10s'},

    # GROUP 2: Signal-flip exit
    {'threshold': 1.5, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z15_5min'},
    {'threshold': 2.0, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z20_5min'},
    {'threshold': 2.5, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z25_5min'},
    {'threshold': 3.0, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z30_5min'},

    # GROUP 3: Market entry
    {'threshold': 2.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_z20_10s'},
    {'threshold': 2.5, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_z25_10s'},
    {'threshold': 3.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_z30_10s'},
    # Market + signal-flip
    {'threshold': 2.0, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_flip_z20_5min'},
    {'threshold': 2.5, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_flip_z25_5min'},

    # GROUP 4: TP/SL configs
    {'threshold': 2.0, 'hold_ms': 60000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': 3, 'stop_loss_ticks': 2, 'market_entry': False,
     'label': 'chase_tp3_sl2_z20_60s'},
    {'threshold': 2.0, 'hold_ms': 60000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': 4, 'stop_loss_ticks': 2, 'market_entry': False,
     'label': 'chase_tp4_sl2_z20_60s'},
    {'threshold': 2.5, 'hold_ms': 60000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': 3, 'stop_loss_ticks': 1, 'market_entry': False,
     'label': 'chase_tp3_sl1_z25_60s'},
    {'threshold': 2.5, 'hold_ms': 60000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': 5, 'stop_loss_ticks': 2, 'market_entry': False,
     'label': 'chase_tp5_sl2_z25_60s'},

    # GROUP 5: Signal-flip + TP/SL
    {'threshold': 2.0, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': 4, 'stop_loss_ticks': 2, 'market_entry': False,
     'label': 'flip_tp4_sl2_z20'},
    {'threshold': 2.5, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': 3, 'stop_loss_ticks': 1, 'market_entry': False,
     'label': 'flip_tp3_sl1_z25'},

    # GROUP 6: With latency
    {'threshold': 2.0, 'hold_ms': 10000, 'latency_ms': 10, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z20_10s_lat10'},
    {'threshold': 2.0, 'hold_ms': 300000, 'latency_ms': 10, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z20_5min_lat10'},

    # GROUP 7: ULTRA-SELECTIVE
    {'threshold': 3.5, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z35_10s_ultra'},
    {'threshold': 4.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z40_10s_ultra'},
    {'threshold': 5.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z50_10s_ultra'},
    {'threshold': 3.5, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z35_5min_ultra'},
    {'threshold': 4.0, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z40_5min_ultra'},
    {'threshold': 3.5, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_z35_10s_ultra'},
    {'threshold': 4.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_z40_10s_ultra'},
]

# Convert to tuple format for sweep
CHASE_CONFIGS = [
    (c['threshold'], c['hold_ms'], c['latency_ms'], c['chase_max_ticks'], c['chase_max_reprices'],
     c['chase_force_cross'], c['chase_interval_ms'], c['label'])
    for c in EXECUTION_CONFIGS
]
_EXTRA_FLAGS = {c['label']: c for c in EXECUTION_CONFIGS}


def rth_start_ns_for_date(date_str):
    """Compute RTH start timestamp (9:30 AM ET) for date YYYYMMDD."""
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    d = datetime(year, month, day)
    dst_end = datetime(2025, 11, 2)
    utc_offset = -4 if d < dst_end else -5
    rth_start_utc_hours = 9.5 - utc_offset
    midnight_utc = datetime(year, month, day, tzinfo=timezone.utc)
    rth_start = midnight_utc + timedelta(hours=rth_start_utc_hours)
    return int(rth_start.timestamp() * 1e9)


def prepare_patchtst_predictions():
    """Convert PatchTST predictions to per-day bar-indexed z-scored NPZ files.

    For each fold:
      1. Load PatchTST predictions (raw regression, column 2 = 10s)
      2. Load event file to get timestamps
      3. Reconstruct window->event->timestamp->bar mapping
      4. Apply expanding z-score across folds (walk-forward)
      5. Save as NPZ with 'predictions' key (234000 bars)
    """
    saved_files = {}
    fold_stats = []

    # Collect fold data
    fold_data = {}
    for fold_idx in USABLE_FOLDS:
        pred_file = PATCHTST_DIR / f'fold_{fold_idx:02d}_oot_predictions.npz'
        if not pred_file.exists():
            log.warning(f"Fold {fold_idx}: prediction file not found, skipping")
            continue

        pred_npz = np.load(str(pred_file), allow_pickle=True)
        predictions = pred_npz['predictions'][:, PRED_COL_10S]  # 10s predictions
        labels = pred_npz['labels'][:, PRED_COL_10S]
        ic_10s = float(pred_npz['ic_10s'])
        oot_files = pred_npz['oot_files']

        # Extract date from oot_files path
        oot_path = str(oot_files[0])
        # Handle both forward and back slashes
        basename = oot_path.replace('\\', '/').split('/')[-1]
        date_str = basename.split('_')[0]

        # Check MBO file
        mbo_zst = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            log.warning(f"Fold {fold_idx} ({date_str}): no MBO file, skipping")
            continue

        # Load event file for timestamps
        event_file = EVENT_DIR / f'{date_str}_mbo_events.npz'
        if not event_file.exists():
            log.warning(f"Fold {fold_idx} ({date_str}): no event file, skipping")
            continue

        ev_data = np.load(str(event_file), allow_pickle=True)
        timestamps = ev_data['timestamps']
        n_events = len(timestamps)

        # Reconstruct window indices (same as PatchTST training)
        starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
        label_idxs = starts + WINDOW - 1

        # Filter valid windows (all horizons non-NaN)
        labels_1s = ev_data['labels_1s'].astype(np.float32)
        labels_5s = ev_data['labels_5s'].astype(np.float32)
        labels_10s = ev_data['labels_10s'].astype(np.float32)
        valid = (~np.isnan(labels_1s[label_idxs]) &
                 ~np.isnan(labels_5s[label_idxs]) &
                 ~np.isnan(labels_10s[label_idxs]))
        valid_label_idxs = label_idxs[valid]

        n_preds = len(predictions)
        n_valid = len(valid_label_idxs)
        if n_preds != n_valid:
            log.warning(f"Fold {fold_idx} ({date_str}): count mismatch "
                       f"({n_preds} preds vs {n_valid} valid windows), using min")
            n_use = min(n_preds, n_valid)
            predictions = predictions[:n_use]
            valid_label_idxs = valid_label_idxs[:n_use]

        # Get timestamps for each prediction window
        pred_timestamps = timestamps[valid_label_idxs]

        # Compute RTH bar indices
        rth_start = rth_start_ns_for_date(date_str)
        bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)

        # Filter to RTH
        rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)
        bar_indices_rth = bar_indices[rth_mask]
        predictions_rth = predictions[rth_mask]

        if len(bar_indices_rth) == 0:
            log.warning(f"Fold {fold_idx} ({date_str}): no RTH predictions, skipping")
            continue

        fold_data[fold_idx] = {
            'date': date_str,
            'bar_indices': bar_indices_rth,
            'raw_signals': predictions_rth,
            'ic_10s': ic_10s,
            'n_preds': n_preds,
            'n_rth': len(bar_indices_rth),
        }

        fold_stats.append({
            'fold': fold_idx,
            'date': date_str,
            'n_preds': n_preds,
            'n_rth': len(bar_indices_rth),
            'ic_10s': ic_10s,
            'pred_mean': float(np.mean(predictions)),
            'pred_std': float(np.std(predictions)),
        })

        log.info(f"  Fold {fold_idx} ({date_str}): {n_preds} predictions, "
                f"{len(bar_indices_rth)} RTH, IC_10s={ic_10s:.4f}")

        del ev_data, timestamps
        gc.collect()

    log.info(f"\nLoaded {len(fold_data)} folds")

    # Apply expanding z-score across folds (walk-forward order)
    running_sum = 0.0
    running_sq = 0.0
    count = 0

    for fold_idx in sorted(fold_data.keys()):
        fd = fold_data[fold_idx]
        date_str = fd['date']
        bar_indices = fd['bar_indices']
        raw_signals = fd['raw_signals']

        # Create bar-level prediction array
        bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)

        # For overlapping windows at same bar, use last (most recent) prediction
        for bi, sig in zip(bar_indices, raw_signals):
            bar_preds[bi] = sig

        # Apply expanding z-score
        zscore_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
        for i in range(N_RTH_BARS):
            v = bar_preds[i]
            if v == 0.0:
                continue
            running_sum += v
            running_sq += v * v
            count += 1
            if count >= 50:
                mean = running_sum / count
                var = (running_sq / count) - mean * mean
                std = max(np.sqrt(max(var, 0)), 1e-8)
                zscore_preds[i] = (v - mean) / std

        # Save
        out_file = PRED_OUT_DIR / f'{date_str}_patchtst.npz'
        np.savez_compressed(str(out_file), predictions=zscore_preds)
        saved_files[date_str] = out_file

        n_nonzero = np.count_nonzero(zscore_preds)
        n_pos = np.sum(zscore_preds > 0)
        n_neg = np.sum(zscore_preds < 0)
        z_abs = np.abs(zscore_preds[zscore_preds != 0])
        log.info(f"  fold{fold_idx:02d} ({date_str}): {n_nonzero} signals "
                f"({n_pos} long, {n_neg} short), "
                f"max={zscore_preds.max():.2f}, min={zscore_preds.min():.2f}, "
                f"mean|z|={z_abs.mean():.2f}")

    log.info(f"\nGenerated {len(saved_files)} prediction files in {PRED_OUT_DIR}")
    return saved_files, fold_stats


def run_single_sim(date_str, pred_file, config_label, threshold, hold_ms,
                   latency_ms, chase_max_ticks, chase_max_reprices,
                   chase_force_cross, chase_interval_ms):
    """Run one Rust fill_sim job."""
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = SIM_OUT_DIR / f'{config_label}_{date_str}.json'

    extra = _EXTRA_FLAGS.get(config_label, {})

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--hold-ms', str(hold_ms),
        '--signal-threshold', str(threshold),
        '--latency-ms', str(latency_ms),
        '--quiet',
    ]

    if extra.get('market_entry', False):
        cmd.append('--market-entry')
    else:
        cmd.append('--chase-entry')
        cmd.extend(['--chase-max-ticks', str(chase_max_ticks)])
        cmd.extend(['--chase-max-reprices', str(chase_max_reprices)])
        cmd.extend(['--chase-interval-ms', str(chase_interval_ms)])
        if chase_force_cross:
            cmd.append('--chase-force-cross')

    if extra.get('signal_flip_exit', False):
        cmd.append('--signal-flip-exit')

    if extra.get('take_profit_ticks') is not None:
        cmd.extend(['--take-profit-ticks', str(extra['take_profit_ticks'])])
    if extra.get('stop_loss_ticks') is not None:
        cmd.extend(['--stop-loss-ticks', str(extra['stop_loss_ticks'])])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            log.warning(f"Sim failed {config_label}/{date_str}: {r.stderr[:300]}")
            return None
        if not out_file.exists():
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        log.warning(f"Timeout: {config_label}/{date_str}")
        return None
    except Exception as e:
        log.warning(f"Error {config_label}/{date_str}: {e}")
        return None


def run_sweep(saved_files, chase_configs, workers=6):
    """Run all sim jobs in parallel."""
    if not BINARY.exists():
        log.error(f"Binary not found: {BINARY}")
        return {}

    jobs = []
    for date_str, pred_file in saved_files.items():
        for (threshold, hold_ms, latency_ms, chase_max_ticks, chase_max_reprices,
             chase_force_cross, chase_interval_ms, base_label) in chase_configs:
            jobs.append({
                'date': date_str,
                'pred_file': pred_file,
                'config_label': base_label,
                'threshold': threshold,
                'hold_ms': hold_ms,
                'latency_ms': latency_ms,
                'chase_max_ticks': chase_max_ticks,
                'chase_max_reprices': chase_max_reprices,
                'chase_force_cross': chase_force_cross,
                'chase_interval_ms': chase_interval_ms,
            })

    log.info(f"\nRunning {len(jobs)} sim jobs ({workers} workers)")
    log.info(f"  Dates: {len(saved_files)}")
    log.info(f"  Configs: {len(chase_configs)}")

    results = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_single_sim,
                job['date'], job['pred_file'], job['config_label'],
                job['threshold'], job['hold_ms'], job['latency_ms'],
                job['chase_max_ticks'], job['chase_max_reprices'],
                job['chase_force_cross'], job['chase_interval_ms']
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            job = futures[future]
            try:
                result = future.result()
                if result:
                    cl = job['config_label']
                    if cl not in results:
                        results[cl] = {}
                    results[cl][job['date']] = result
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


def compute_mfe_mae(trades):
    """Compute MFE/MAE statistics."""
    if not trades:
        return {}
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


def compute_confidence_tiers(results_by_date):
    """Analyze performance at different confidence tiers.
    Tiers: All, 50%, 25%, 10%, 5%, 1%, 0.5%, 0.1%
    """
    all_trades = []
    for date_str, res in results_by_date.items():
        if 'trades' not in res:
            continue
        for trade in res['trades']:
            trade_copy = dict(trade)
            trade_copy['date'] = date_str
            sig = abs(trade.get('signal_strength', trade.get('entry_signal', trade.get('signal', 0))))
            trade_copy['signal_mag'] = sig
            all_trades.append(trade_copy)

    if not all_trades:
        return []

    signal_mags = np.array([t['signal_mag'] for t in all_trades])
    pnls = np.array([t.get('pnl_dollars', 0) for t in all_trades])

    tiers = []
    for tier_name, pct in [('All', 0), ('Top50%', 50), ('Top25%', 75), ('Top10%', 90),
                            ('Top5%', 95), ('Top1%', 99), ('Top0.5%', 99.5), ('Top0.1%', 99.9)]:
        if pct > 0:
            thresh = np.percentile(signal_mags, pct)
            mask = signal_mags >= thresh
        else:
            mask = np.ones(len(all_trades), dtype=bool)
            thresh = 0

        tier_trades = [t for t, m in zip(all_trades, mask) if m]
        tier_pnls = pnls[mask]

        if len(tier_pnls) == 0:
            continue

        long_pnls = [t.get('pnl_dollars', 0) for t in tier_trades
                     if str(t.get('side', t.get('direction', ''))).upper() in ('BUY', 'LONG', '1')]
        short_pnls = [t.get('pnl_dollars', 0) for t in tier_trades
                      if str(t.get('side', t.get('direction', ''))).upper() in ('SELL', 'SHORT', '-1')]

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
        profit_factor = gross_profit / max(gross_loss, 0.01)

        tier_info = {
            'tier': tier_name,
            'threshold': round(float(thresh), 3),
            'n_trades': len(tier_pnls),
            'total_pnl': round(float(tier_pnls.sum()), 2),
            'avg_trade_pnl': round(float(tier_pnls.mean()), 2),
            'avg_trade_ticks': round(float(tier_pnls.mean() / TICK_VALUE), 3),
            'win_rate': round(float(np.mean(tier_pnls > 0)), 4),
            'profit_factor': round(profit_factor, 2),
            'sortino': round(sortino, 2),
            'n_long': len(long_pnls),
            'n_short': len(short_pnls),
            'long_pnl': round(sum(long_pnls), 2) if long_pnls else 0,
            'short_pnl': round(sum(short_pnls), 2) if short_pnls else 0,
            'long_avg': round(float(np.mean(long_pnls)), 2) if long_pnls else 0,
            'short_avg': round(float(np.mean(short_pnls)), 2) if short_pnls else 0,
            'mfe_mae': compute_mfe_mae(tier_trades),
        }
        tiers.append(tier_info)

    return tiers


def aggregate_and_report(results, fold_stats=None):
    """Aggregate per-day results into per-config summaries."""
    log.info("\n" + "=" * 95)
    log.info("PatchTST → RUST MBO FILL SIM RESULTS (MARKET REPLAY)")
    log.info("Walk-forward OOT predictions, expanding z-score normalization")
    log.info("=" * 95)

    summary = []

    for config_label, date_results in sorted(results.items()):
        total_pnl = 0.0
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0
        daily_pnls = []
        all_trade_pnls = []

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

        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = abs((cum - peak).min()) if len(cum) > 0 else 0

        tiers = compute_confidence_tiers(date_results)

        summary.append({
            'config': config_label,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe_daily': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'profit_factor': round(profit_factor, 2),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_trade_ticks': round(avg_trade_ticks, 3),
            'max_dd': round(max_dd, 2),
            'annualized_pnl': round(avg_daily * 252, 0),
            'confidence_tiers': tiers,
            'daily_pnls': daily_pnls,
        })

    summary.sort(key=lambda x: x['total_pnl'], reverse=True)

    if not summary:
        log.warning("No configs produced trades!")
        return summary

    # Print summary table
    log.info(f"\nAll configs by total P&L ({summary[0]['n_days']} OOT days):")
    log.info(f"{'Config':<45} {'P&L':>10} {'Trades':>7} {'FillR':>6} {'WinR':>6} "
             f"{'Sharpe':>7} {'Sortino':>8} {'PF':>5} {'MaxDD':>8}")
    log.info("-" * 120)
    for s in summary:
        log.info(
            f"{s['config']:<45} "
            f"${s['total_pnl']:>9,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"{s['sharpe_daily']:>7.2f} "
            f"{s['sortino']:>8.2f} "
            f"{s['profit_factor']:>5.2f} "
            f"${s['max_dd']:>7,.0f}"
        )

    # Top 5 configs detail
    for rank, best in enumerate(summary[:5]):
        log.info(f"\n{'='*70}")
        log.info(f"#{rank+1} CONFIG: {best['config']}")
        log.info(f"  Total P&L:       ${best['total_pnl']:,.2f}")
        log.info(f"  Days:            {best['n_days']}")
        log.info(f"  Total trades:    {best['n_trades']}")
        log.info(f"  Signals sent:    {best['n_signals']}")
        log.info(f"  Fill rate:       {best['fill_rate']:.1%}")
        log.info(f"  Win rate:        {best['win_rate']:.1%}")
        log.info(f"  Sharpe (daily):  {best['sharpe_daily']:.2f}")
        log.info(f"  Sortino:         {best['sortino']:.2f}")
        log.info(f"  Profit factor:   {best['profit_factor']:.2f}")
        log.info(f"  Avg trade P&L:   ${best['avg_trade_pnl']:.2f} ({best['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Max drawdown:    ${best['max_dd']:,.2f}")
        log.info(f"  Annualized:      ${best['annualized_pnl']:,.0f}")

        if best.get('confidence_tiers'):
            log.info(f"\n  Confidence Tier Analysis:")
            log.info(f"  {'Tier':<10} {'Trades':>7} {'P&L':>10} {'AvgTrade':>10} {'WinR':>6} "
                     f"{'PF':>5} {'Sortino':>8} {'Long':>6} {'Short':>6}")
            log.info(f"  {'-'*80}")
            for tier in best['confidence_tiers']:
                log.info(
                    f"  {tier['tier']:<10} "
                    f"{tier['n_trades']:>7} "
                    f"${tier['total_pnl']:>9,.0f} "
                    f"${tier['avg_trade_pnl']:>9.2f} "
                    f"{tier['win_rate']:>5.1%} "
                    f"{tier['profit_factor']:>5.2f} "
                    f"{tier['sortino']:>8.2f} "
                    f"{tier['n_long']:>6} "
                    f"{tier['n_short']:>6}"
                )

            for tier in best['confidence_tiers']:
                if tier.get('mfe_mae'):
                    mm = tier['mfe_mae']
                    log.info(f"\n  MFE/MAE ({tier['tier']}):")
                    log.info(f"    MFE: mean={mm.get('mfe_mean',0):.1f}t, "
                            f"median={mm.get('mfe_median',0):.1f}t, "
                            f"p90={mm.get('mfe_p90',0):.1f}t")
                    log.info(f"    MAE: mean={mm.get('mae_mean',0):.1f}t, "
                            f"median={mm.get('mae_median',0):.1f}t, "
                            f"p90={mm.get('mae_p90',0):.1f}t")
                    log.info(f"    MFE/MAE ratio: {mm.get('mfe_mae_ratio',0):.2f}")

    # Save results JSON
    out_summary = []
    for s in summary:
        entry = {k: v for k, v in s.items() if k != 'daily_pnls'}
        entry['daily_pnl_list'] = s['daily_pnls']
        out_summary.append(entry)

    out_file = RESULTS_DIR / 'patchtst_fill_sim_results.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'model': 'patchtst_sliding60d_smart_v2',
            'pred_dir': str(PATCHTST_DIR),
            'n_folds': len(fold_stats) if fold_stats else 0,
            'fold_stats': fold_stats or [],
            'configs': out_summary,
        }, f, indent=2, default=str)
    log.info(f"\nResults saved: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='PatchTST -> Rust MBO Fill Sim')
    parser.add_argument('--workers', type=int, default=10,
                        help='Parallel sim workers (default: 10)')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip prediction file generation (use existing)')
    parser.add_argument('--configs', type=str, default='all',
                        help='Config filter: "all", "best", or comma-separated labels')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("PatchTST -> Rust MBO Fill Simulator")
    log.info(f"  Binary:      {BINARY}")
    log.info(f"  Predictions: {PATCHTST_DIR}")
    log.info(f"  Event data:  {EVENT_DIR}")
    log.info(f"  MBO data:    {MBO_DIR}")
    log.info(f"  Pred out:    {PRED_OUT_DIR}")
    log.info(f"  Sim out:     {SIM_OUT_DIR}")
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  Usable folds: {USABLE_FOLDS}")
    log.info("=" * 70)

    if not BINARY.exists():
        log.error(f"fill_sim_cli not found: {BINARY}")
        sys.exit(1)

    # Step 1: Prepare predictions
    if args.skip_gen:
        log.info("Loading existing prediction files...")
        saved_files = {}
        for f in PRED_OUT_DIR.glob('*_patchtst.npz'):
            date_str = f.stem.split('_')[0]
            saved_files[date_str] = f
        log.info(f"Found {len(saved_files)} existing prediction files")
        fold_stats = []
    else:
        saved_files, fold_stats = prepare_patchtst_predictions()

    if not saved_files:
        log.error("No prediction files found/generated. Exiting.")
        sys.exit(1)

    # Step 2: Select configs
    configs = CHASE_CONFIGS
    if args.configs == 'best':
        configs = [CHASE_CONFIGS[0]]
    elif args.configs != 'all':
        labels = args.configs.split(',')
        configs = [c for c in CHASE_CONFIGS if c[-1] in labels]

    # Step 3: Run fill_sim sweep
    results = run_sweep(saved_files, configs, workers=args.workers)

    if not results:
        log.error("No sim results. Check binary and MBO files.")
        sys.exit(1)

    # Step 4: Aggregate and report
    summary = aggregate_and_report(results, fold_stats)
    log.info(f"\nDone. Log: {_log_file}")

    return summary


if __name__ == '__main__':
    main()
