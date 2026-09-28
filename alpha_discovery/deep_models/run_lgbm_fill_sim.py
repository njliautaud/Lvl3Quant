#!/usr/bin/env python3
"""
LGBM DA Smart-v2 Predictions → Rust MBO Fill Simulator
========================================================
Market replay fill simulation for LGBM binary classifier (smart_v2) predictions.

Loads per-fold 1d OOT predictions from lgbm_da_smart_v2_1d_oot/,
maps each fold to its OOT date, converts P(up) probabilities to
directional signals, and feeds them through the Rust fill_sim_cli
binary against raw MBO data for realistic market replay simulation.

Key design:
  - Each fold = 1 OOT day (LGBM_TEST_DAYS=1)
  - Predictions are per-window (WINDOW=1000, STRIDE=500 events)
  - Maps window predictions → 100ms RTH bar indices via event timestamps
  - Applies expanding z-score normalization (no look-ahead)
  - Confidence tier analysis: All/50%/25%/10%/5%
  - MFE/MAE analysis per tier
  - Long vs short breakdown

Usage:
    python run_lgbm_fill_sim.py
    python run_lgbm_fill_sim.py --workers 8
    python run_lgbm_fill_sim.py --threshold 2.0 --hold-ms 1800000
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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'lgbm_da_smart_v2_1d_oot'
EVENT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_events_smart_v2'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'lgbm_fill_sim_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'lgbm_fill_sim_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── LGBM walk-forward config (must match train_lgbm_da_classifier.py) ──
WINDOW = 1000       # events per window
STRIDE = 500        # stride between windows
TRAIN_DAYS = 60     # expanding window minimum
TEST_DAYS = 1       # 1d OOT

TICK_VALUE = 12.50
BARS_PER_SEC = 10
BAR_NS = 100_000_000  # 100ms in nanoseconds
RTH_HOURS = 6.5
N_RTH_BARS = int(RTH_HOURS * 3600 * BARS_PER_SEC)  # 234000

# DST boundary: Nov 2, 2025 06:00 UTC
DST_END_2025_NS = 1_762_056_000_000_000_000

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = str(RESULTS_DIR / f'lgbm_fill_sim_{_ts}.log')

log = logging.getLogger('lgbm_fill_sim')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── Sim configs to sweep ──
# Format: (threshold, hold_ms, latency_ms, chase_max_ticks, chase_max_reprices,
#           chase_force_cross, chase_interval_ms, label)
# Extended config format: dict-based for clarity
# Each config is a dict with keys matching Rust CLI flags
EXECUTION_CONFIGS = [
    # ── GROUP 1: Baseline chase entries (passive limit orders) ──
    # Hold = 10s (match prediction horizon)
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
    # Hold = 30s
    {'threshold': 1.5, 'hold_ms': 30000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z15_30s'},
    {'threshold': 2.0, 'hold_ms': 30000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z20_30s'},

    # ── GROUP 2: Signal-flip exit (exit when signal reverses — ADAPTIVE hold) ──
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

    # ── GROUP 3: Market entry (pays spread, guaranteed fills) ──
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

    # ── GROUP 4: MFE-informed TP/SL (from prior analysis: MFE=4.8t winners, MAE=0.6t) ──
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

    # ── GROUP 5: Signal-flip + TP/SL (best of both) ──
    {'threshold': 2.0, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': 4, 'stop_loss_ticks': 2, 'market_entry': False,
     'label': 'flip_tp4_sl2_z20'},
    {'threshold': 2.5, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': 3, 'stop_loss_ticks': 1, 'market_entry': False,
     'label': 'flip_tp3_sl1_z25'},

    # ── GROUP 6: With realistic latency (10ms) ──
    {'threshold': 2.0, 'hold_ms': 10000, 'latency_ms': 10, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'chase_z20_10s_lat10'},
    {'threshold': 2.0, 'hold_ms': 300000, 'latency_ms': 10, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z20_5min_lat10'},

    # ── GROUP 7: ULTRA-SELECTIVE (top ~1%/0.5%/0.1% — extreme tail signals only) ──
    # Chase entry — ultra-high conviction only
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
    # Signal-flip exit — ultra-selective
    {'threshold': 3.5, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z35_5min_ultra'},
    {'threshold': 4.0, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 1, 'chase_max_reprices': 3,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'label': 'flip_z40_5min_ultra'},
    # Market entry — ultra-selective (pays spread but instant fill on extreme signals)
    {'threshold': 3.5, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_z35_10s_ultra'},
    {'threshold': 4.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': True,
     'label': 'market_z40_10s_ultra'},

    # ── GROUP 8: MID-PRICE ENTRY (split the spread — post at midpoint) ──
    # Saves ~0.5 ticks vs market orders. Fill depends on price crossing mid.
    {'threshold': 2.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'mid_price_entry': True, 'label': 'mid_z20_10s'},
    {'threshold': 2.5, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'mid_price_entry': True, 'label': 'mid_z25_10s'},
    {'threshold': 3.0, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'mid_price_entry': True, 'label': 'mid_z30_10s'},
    {'threshold': 3.5, 'hold_ms': 10000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'mid_price_entry': True, 'label': 'mid_z35_10s_ultra'},
    # Mid-price with signal-flip exit
    {'threshold': 2.0, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'mid_price_entry': True, 'label': 'mid_flip_z20_5min'},
    {'threshold': 2.5, 'hold_ms': 300000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': True,
     'take_profit_ticks': None, 'stop_loss_ticks': None, 'market_entry': False,
     'mid_price_entry': True, 'label': 'mid_flip_z25_5min'},
    # Mid-price with TP/SL
    {'threshold': 2.5, 'hold_ms': 60000, 'latency_ms': 0, 'chase_max_ticks': 0, 'chase_max_reprices': 0,
     'chase_force_cross': False, 'chase_interval_ms': 100, 'signal_flip_exit': False,
     'take_profit_ticks': 3, 'stop_loss_ticks': 1, 'market_entry': False,
     'mid_price_entry': True, 'label': 'mid_tp3_sl1_z25_60s'},
]

# Convert dict configs to tuple format for backward compatibility
CHASE_CONFIGS = [
    (c['threshold'], c['hold_ms'], c['latency_ms'], c['chase_max_ticks'], c['chase_max_reprices'],
     c['chase_force_cross'], c['chase_interval_ms'], c['label'])
    for c in EXECUTION_CONFIGS
]

# Store extra flags per config label for use in run_single_sim
_EXTRA_FLAGS = {c['label']: c for c in EXECUTION_CONFIGS}


def et_offset_hours(ts_ns):
    """UTC offset for Eastern Time: -4 (EDT) or -5 (EST)."""
    return -4 if ts_ns < DST_END_2025_NS else -5


def is_within_rth(ts_ns):
    """Check if nanosecond UTC timestamp is within RTH (9:30-16:00 ET, weekdays)."""
    if ts_ns == 0:
        return False
    ts_sec = ts_ns // 1_000_000_000
    offset = et_offset_hours(ts_ns)
    et_sec = ts_sec + offset * 3600

    days = et_sec // 86400 if et_sec >= 0 else (et_sec - 86399) // 86400
    dow = (days + 4) % 7  # 0=Sun, 6=Sat
    if dow == 0 or dow == 6:
        return False

    secs_in_day = et_sec % 86400
    if secs_in_day < 0:
        secs_in_day += 86400
    minutes = secs_in_day // 60
    return 570 <= minutes < 960  # 9:30 AM = 570 min, 4:00 PM = 960 min


def rth_start_ns_for_date(date_str):
    """Compute the RTH start timestamp (9:30 AM ET) for a given date YYYYMMDD."""
    from datetime import datetime as dt, timezone, timedelta
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Determine ET offset for this date
    # Simple: dates before Nov 2025 are EDT (-4), after are EST (-5)
    d = dt(year, month, day)
    # DST_END_2025 is Nov 2, 2025
    dst_end = dt(2025, 11, 2)
    utc_offset = -4 if d < dst_end else -5

    # 9:30 AM ET in UTC
    rth_start_utc_hours = 9.5 - utc_offset  # 9:30 AM ET → UTC
    midnight_utc = dt(year, month, day, tzinfo=timezone.utc)
    rth_start = midnight_utc + timedelta(hours=rth_start_utc_hours)
    return int(rth_start.timestamp() * 1e9)


def map_fold_to_date():
    """Reconstruct fold-to-date mapping using the data directory file list.

    Replicates the walk-forward logic from train_lgbm_da_classifier.py:
        - files = sorted(data_dir.glob("*.npz"))
        - Valid files have non-NaN labels_10s
        - Folds: min_train = max(TRAIN_DAYS, 30) = 60
        - test_start = 60, TEST_DAYS = 1
        - fold N → files[60 + N]
    """
    all_files = sorted(EVENT_DIR.glob('*.npz'))
    if not all_files:
        log.error(f"No event files in {EVENT_DIR}")
        return {}

    # Filter valid files (must have non-NaN labels_10s) — same as training script
    valid_files = []
    for f in all_files:
        try:
            d = np.load(f, allow_pickle=True)
            if 'labels_10s' in d and not np.all(np.isnan(d['labels_10s'])):
                valid_files.append(f)
        except Exception:
            pass

    log.info(f"Event files: {len(all_files)} total, {len(valid_files)} valid")

    min_train = max(TRAIN_DAYS, 30)
    fold_dates = {}
    test_start = min_train
    fold_idx = 0

    while test_start + TEST_DAYS <= len(valid_files):
        test_file = valid_files[test_start]
        # Extract date from filename: YYYYMMDD_mbo_events.npz
        date_str = test_file.stem.split('_')[0]
        fold_dates[fold_idx] = date_str
        test_start += TEST_DAYS
        fold_idx += 1

    log.info(f"Fold-to-date mapping: {len(fold_dates)} folds")
    if fold_dates:
        first_fold = min(fold_dates.keys())
        last_fold = max(fold_dates.keys())
        log.info(f"  fold{first_fold:02d} → {fold_dates[first_fold]}, "
                 f"fold{last_fold:02d} → {fold_dates[last_fold]}")

    return fold_dates


def zscore_expanding(arr):
    """Expanding-window z-score (no look-ahead). Requires >=50 non-NaN samples."""
    result = np.full_like(arr, np.nan, dtype=np.float64)
    running_sum = 0.0
    running_sq = 0.0
    count = 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v) or v == 0.0:
            result[i] = 0.0
            continue
        running_sum += v
        running_sq += v * v
        count += 1
        if count >= 50:
            mean = running_sum / count
            var = (running_sq / count) - mean * mean
            std = max(np.sqrt(max(var, 0)), 1e-8)
            result[i] = (v - mean) / std
    return result


def prepare_predictions(fold_dates):
    """Load LGBM predictions and create per-day, per-bar NPZ files for Rust sim.

    For each fold:
      1. Load fold predictions (P(up) probabilities)
      2. Load corresponding event file to get timestamps
      3. Map window indices → event indices → timestamps → 100ms RTH bar indices
      4. Convert P(up) to directional signal: signal = P(up) - 0.5
      5. Apply expanding z-score across all predictions seen so far (walk-forward)
      6. Save as NPZ with 'predictions' key indexed by RTH bar
    """
    # Discover available fold prediction files
    pred_files = sorted(PRED_DIR.glob('fold*_preds.npz'))
    if not pred_files:
        log.error(f"No prediction files in {PRED_DIR}")
        return {}, []

    log.info(f"Found {len(pred_files)} fold prediction files")

    # Collect all raw signals across folds for expanding z-score
    # We process folds in order to maintain walk-forward discipline
    all_raw_signals = []  # (date, bar_idx, raw_signal)
    fold_bar_signals = {}  # fold_idx -> list of (bar_idx, raw_signal)

    saved_files = {}
    skipped = []
    fold_stats = []

    for pf in pred_files:
        # Parse fold index from filename
        fold_idx = int(pf.stem.replace('fold', '').replace('_preds', ''))
        if fold_idx not in fold_dates:
            log.warning(f"fold{fold_idx:02d}: no date mapping, skipping")
            skipped.append(fold_idx)
            continue

        date_str = fold_dates[fold_idx]
        date_dash = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

        # Check MBO file exists
        mbo_zst = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            log.warning(f"fold{fold_idx:02d} ({date_str}): no MBO file, skipping")
            skipped.append(fold_idx)
            continue

        # Load predictions
        pred_data = np.load(str(pf), allow_pickle=True)
        probs = pred_data['probs']  # P(up) probabilities
        confidence = pred_data['confidence']  # |P(up) - 0.5|
        n_preds = len(probs)

        # Load event file to get timestamps
        event_file = EVENT_DIR / f'{date_str}_mbo_events.npz'
        if not event_file.exists():
            log.warning(f"fold{fold_idx:02d} ({date_str}): no event file, skipping")
            skipped.append(fold_idx)
            continue

        ev_data = np.load(str(event_file), allow_pickle=True)
        timestamps = ev_data['timestamps']
        n_events = len(timestamps)

        # Reconstruct window → event index mapping
        # build_xy uses: starts = arange(0, n_ev - window + 1, stride)
        # label_idx = start + window - 1 (last event in window)
        # Then filters out NaN and zero labels
        starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
        label_idxs = starts + WINDOW - 1

        # Filter same way as training: valid = not NaN AND not zero
        labels = ev_data['labels_10s'].astype(np.float32)
        valid = (label_idxs < len(labels)) & ~np.isnan(labels[label_idxs]) & (labels[label_idxs] != 0)
        valid_label_idxs = label_idxs[valid]

        if len(valid_label_idxs) != n_preds:
            log.warning(f"fold{fold_idx:02d} ({date_str}): pred count mismatch "
                       f"({n_preds} vs {len(valid_label_idxs)} valid windows), "
                       f"using min")
            n_use = min(n_preds, len(valid_label_idxs))
            probs = probs[:n_use]
            confidence = confidence[:n_use]
            valid_label_idxs = valid_label_idxs[:n_use]

        # Map each prediction to its 100ms RTH bar index
        # Get timestamp of the last event in each window
        pred_timestamps = timestamps[valid_label_idxs]

        # Compute RTH start for this date
        rth_start = rth_start_ns_for_date(date_str)

        # Convert: raw directional signal = P(up) - 0.5
        # Positive = bullish, negative = bearish
        raw_signal = probs - 0.5

        # Map to RTH bar indices
        bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)

        # Filter to RTH bars only (0 to N_RTH_BARS-1)
        rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)
        bar_indices = bar_indices[rth_mask]
        raw_signal_rth = raw_signal[rth_mask]

        if len(bar_indices) == 0:
            log.warning(f"fold{fold_idx:02d} ({date_str}): no RTH predictions, skipping")
            skipped.append(fold_idx)
            continue

        # Store for expanding z-score (across folds, walk-forward order)
        fold_bar_signals[fold_idx] = (date_str, bar_indices, raw_signal_rth)
        all_raw_signals.extend(raw_signal_rth.tolist())

        fold_stats.append({
            'fold': fold_idx,
            'date': date_str,
            'n_preds': n_preds,
            'n_rth': len(bar_indices),
            'mean_prob': float(np.mean(probs)),
            'mean_confidence': float(np.mean(confidence)),
        })

        del ev_data, pred_data, timestamps, labels
        gc.collect()

    log.info(f"Processed {len(fold_bar_signals)} folds, skipped {len(skipped)}")

    # Now apply expanding z-score across ALL folds in order (walk-forward)
    # This means fold N's z-score uses statistics from folds 0..N
    running_sum = 0.0
    running_sq = 0.0
    count = 0

    for fold_idx in sorted(fold_bar_signals.keys()):
        date_str, bar_indices, raw_signal = fold_bar_signals[fold_idx]

        # Create full-day bar-level prediction array
        bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)

        # For bars with multiple predictions (overlapping windows), use the last one
        # (most recent window ending at that bar)
        for bi, sig in zip(bar_indices, raw_signal):
            bar_preds[bi] = sig

        # Apply expanding z-score to this day's predictions
        # using all historical signals seen so far
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

        # Save prediction file
        out_file = PRED_OUT_DIR / f'{date_str}_lgbm_da.npz'
        np.savez_compressed(str(out_file), predictions=zscore_preds)
        saved_files[date_str] = out_file

        n_nonzero = np.count_nonzero(zscore_preds)
        n_pos = np.sum(zscore_preds > 0)
        n_neg = np.sum(zscore_preds < 0)
        log.info(f"  fold{fold_idx:02d} ({date_str}): {n_nonzero} signals "
                f"({n_pos} long, {n_neg} short), "
                f"max={zscore_preds.max():.2f}, min={zscore_preds.min():.2f}")

    log.info(f"\nGenerated {len(saved_files)} prediction files")
    return saved_files, fold_stats


def run_single_sim(date_str, pred_file, config_label, threshold, hold_ms,
                   latency_ms, chase_max_ticks, chase_max_reprices,
                   chase_force_cross, chase_interval_ms):
    """Run one Rust fill_sim job and return parsed JSON result."""
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{date_str}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = SIM_OUT_DIR / f'{config_label}_{date_str}.json'

    # Get extra flags for this config
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

    # Entry mode: market vs mid-price vs chase
    if extra.get('market_entry', False):
        cmd.append('--market-entry')
    elif extra.get('mid_price_entry', False):
        cmd.append('--mid-price-entry')
    else:
        cmd.append('--chase-entry')
        cmd.extend(['--chase-max-ticks', str(chase_max_ticks)])
        cmd.extend(['--chase-max-reprices', str(chase_max_reprices)])
        cmd.extend(['--chase-interval-ms', str(chase_interval_ms)])
        if chase_force_cross:
            cmd.append('--chase-force-cross')

    # Signal-flip exit
    if extra.get('signal_flip_exit', False):
        cmd.append('--signal-flip-exit')

    # Take profit / stop loss
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
    """Compute MFE/MAE statistics from trade list."""
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


def compute_confidence_tiers(results_by_date, all_zscore_magnitudes=None):
    """Analyze performance at different confidence tiers.

    Tiers: All, Top 50%, Top 25%, Top 10%, Top 5%
    For each tier, compute P&L, Sortino, win rate, MFE/MAE.
    """
    # Collect all trades with their signal magnitude
    all_trades = []
    for date_str, res in results_by_date.items():
        if 'trades' not in res:
            continue
        for trade in res['trades']:
            trade_copy = dict(trade)
            trade_copy['date'] = date_str
            # Signal magnitude = abs(prediction at entry)
            # Rust uses 'signal_strength' field
            sig = abs(trade.get('signal_strength', trade.get('entry_signal', trade.get('signal', 0))))
            trade_copy['signal_mag'] = sig
            all_trades.append(trade_copy)

    if not all_trades:
        return []

    signal_mags = np.array([t['signal_mag'] for t in all_trades])
    pnls = np.array([t.get('pnl_dollars', 0) for t in all_trades])

    tiers = []
    for tier_name, pct in [('All', 0), ('Top50%', 50), ('Top25%', 75), ('Top10%', 90), ('Top5%', 95),
                               ('Top1%', 99), ('Top0.5%', 99.5), ('Top0.1%', 99.9)]:
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

        # Long/short breakdown (Rust uses side: "BUY"/"SELL")
        long_pnls = [t.get('pnl_dollars', 0) for t in tier_trades
                     if str(t.get('side', t.get('direction', ''))).upper() in ('BUY', 'LONG', '1')]
        short_pnls = [t.get('pnl_dollars', 0) for t in tier_trades
                      if str(t.get('side', t.get('direction', ''))).upper() in ('SELL', 'SHORT', '-1')]

        # Sortino (daily aggregation)
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

        # Profit factor
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
    """Aggregate per-day results into per-config summaries with confidence tier analysis."""
    log.info("\n" + "=" * 95)
    log.info("LGBM DA Smart-v2 → RUST MBO FILL SIM RESULTS (MARKET REPLAY)")
    log.info("Walk-forward 1d OOT predictions, expanding z-score normalization")
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

        # Sortino
        downside = [min(0, x) for x in daily_pnls]
        downside_std = np.std(downside) if downside else 1e-8
        sortino = (avg_daily / max(downside_std, 1e-8)) * np.sqrt(252)

        # Profit factor
        gross_profit = sum(p for p in all_trade_pnls if p > 0)
        gross_loss = abs(sum(p for p in all_trade_pnls if p < 0))
        profit_factor = gross_profit / max(gross_loss, 0.01)

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_trade_ticks = avg_trade_pnl / TICK_VALUE

        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = abs((cum - peak).min()) if len(cum) > 0 else 0

        # Confidence tier analysis
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

    # Best config detail
    if summary:
        best = summary[0]
        log.info(f"\n{'='*70}")
        log.info(f"BEST CONFIG: {best['config']}")
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

        # Confidence tier breakdown
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

            # MFE/MAE for best tier with trades
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

            # Long vs short breakdown
            log.info(f"\n  Long vs Short Breakdown:")
            for tier in best['confidence_tiers']:
                if tier['n_long'] > 0 or tier['n_short'] > 0:
                    log.info(f"  {tier['tier']:<10}: "
                            f"Long={tier['n_long']} (${tier['long_pnl']:,.0f}, "
                            f"avg=${tier['long_avg']:.2f}) | "
                            f"Short={tier['n_short']} (${tier['short_pnl']:,.0f}, "
                            f"avg=${tier['short_avg']:.2f})")

    # Save results JSON
    out_summary = []
    for s in summary:
        entry = {k: v for k, v in s.items() if k != 'daily_pnls'}
        entry['daily_pnl_list'] = s['daily_pnls']
        out_summary.append(entry)

    out_file = RESULTS_DIR / 'lgbm_fill_sim_results.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'model': 'lgbm_da_smart_v2_1d_oot',
            'pred_dir': str(PRED_DIR),
            'n_folds': len(fold_stats) if fold_stats else 0,
            'fold_stats': fold_stats or [],
            'configs': out_summary,
        }, f, indent=2, default=str)
    log.info(f"\nResults saved: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='LGBM DA Smart-v2 → Rust MBO Fill Sim')
    parser.add_argument('--workers', type=int, default=10,
                        help='Parallel sim workers (default: 10)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override signal threshold for all configs')
    parser.add_argument('--hold-ms', type=int, default=None,
                        help='Override hold time in ms for all configs')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip prediction file generation (use existing)')
    parser.add_argument('--configs', type=str, default='all',
                        help='Config filter: "all", "best", or comma-separated labels')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("LGBM DA Smart-v2 → Rust MBO Fill Simulator")
    log.info(f"  Binary:      {BINARY}")
    log.info(f"  Predictions: {PRED_DIR}")
    log.info(f"  Event data:  {EVENT_DIR}")
    log.info(f"  MBO data:    {MBO_DIR}")
    log.info(f"  Pred out:    {PRED_OUT_DIR}")
    log.info(f"  Sim out:     {SIM_OUT_DIR}")
    log.info(f"  Workers:     {args.workers}")
    log.info("=" * 70)

    # Verify prerequisites
    if not BINARY.exists():
        log.error(f"fill_sim_cli not found: {BINARY}")
        sys.exit(1)
    if not PRED_DIR.exists():
        log.error(f"Prediction directory not found: {PRED_DIR}")
        sys.exit(1)
    if not MBO_DIR.exists():
        log.error(f"MBO directory not found: {MBO_DIR}")
        sys.exit(1)

    # Step 1: Map folds to dates
    fold_dates = map_fold_to_date()
    if not fold_dates:
        log.error("Could not determine fold-to-date mapping")
        sys.exit(1)

    # Step 2: Prepare per-day prediction files
    fold_stats = []
    if args.skip_gen:
        log.info("Loading existing prediction files...")
        saved_files = {}
        for f in PRED_OUT_DIR.glob('*_lgbm_da.npz'):
            date_str = f.stem.split('_')[0]
            saved_files[date_str] = f
        log.info(f"Found {len(saved_files)} existing prediction files")
    else:
        saved_files, fold_stats = prepare_predictions(fold_dates)

    if not saved_files:
        log.error("No prediction files found/generated. Exiting.")
        sys.exit(1)

    # Step 3: Select configs
    configs = CHASE_CONFIGS
    if args.configs == 'best':
        configs = [CHASE_CONFIGS[0]]
    elif args.configs != 'all':
        labels = args.configs.split(',')
        configs = [c for c in CHASE_CONFIGS if c[-1] in labels]

    # Override threshold/hold if specified
    if args.threshold is not None or args.hold_ms is not None:
        new_configs = []
        for c in configs:
            c_list = list(c)
            if args.threshold is not None:
                c_list[0] = args.threshold
            if args.hold_ms is not None:
                c_list[1] = args.hold_ms
            new_configs.append(tuple(c_list))
        configs = new_configs

    # Step 4: Run fill_sim sweep
    results = run_sweep(saved_files, configs, workers=args.workers)

    if not results:
        log.error("No sim results. Check binary and MBO files.")
        sys.exit(1)

    # Step 5: Aggregate and save
    summary = aggregate_and_report(results, fold_stats)
    log.info(f"\nDone. Log: {_log_file}")

    return summary


if __name__ == '__main__':
    main()
