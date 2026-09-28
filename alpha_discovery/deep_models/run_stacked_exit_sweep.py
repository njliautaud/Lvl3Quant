#!/usr/bin/env python3
"""
Stacked Exit Sweep — Multiple Exit Layers Active Simultaneously
================================================================
Tests STACKED exits: each trade has 4 simultaneous exit layers, whichever fires
first exits the trade:

  1. Dynamic signal exit (encoded in prediction file → Rust sees SignalFlip)
  2. Take-profit (Rust sim --take-profit-ticks)
  3. Trailing stop (Rust sim --trailing-ticks)
  4. Max hold timeout (60 min safety, --hold-ms 3600000)

ENTRY METHODS (winners from matrix sweep):
  - smooth_entry: rolling mean z-score (50-bar) > entry_threshold
  - ema_entry: EMA z-score (span 5000) > entry_threshold
  - book_confirmed: expanding z-score > threshold AND book imbalance confirms
  - momentum_entry: z-score acceleration (50-bar diff) > accel_threshold

DYNAMIC EXIT METHODS (encoded in prediction signal):
  - smooth_exit: rolling mean drops below exit_threshold
  - book_exit: book imbalance flips against position
  - momentum_exit: z-score acceleration reverses sign
  - predstd_exit: pred_std exceeds 0.10 (model confused)

TP LAYER:  none, 5, 8, 10, 15, 20 ticks
SL LAYER:  none, 10, 15, 20, 25 ticks (trailing)
VOL:       0 (none), 50, 70
CONV:      1.5, 2.0, 2.5 (z-score pairs) | 0.05, 0.10, 0.15 (raw pairs)
EXIT THR:  0.0, 0.5 (for smooth/ema/raw exits)

NEW (normalization sensitivity):
  - raw entry:     passes raw WF predictions to threshold directly (no z-score)
  - rolling entry: fixed 5000-bar rolling z-score instead of expanding
  - vol=0:         no vol gate applied (baseline)

Total: 8 pairs x 6 TP x 5 SL x 3 vol x 3 conv x ~2 exit_thr = ~4,300 combos x N days

Usage:
    python alpha_discovery/deep_models/run_stacked_exit_sweep.py --workers 24
    python alpha_discovery/deep_models/run_stacked_exit_sweep.py --workers 24 --skip-gen
    python alpha_discovery/deep_models/run_stacked_exit_sweep.py --skip-sim
    python alpha_discovery/deep_models/run_stacked_exit_sweep.py --upload-saturn
"""

import os
import sys
import gc
import json
import time
import bisect
import logging
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
BOOK_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_stacked_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_stacked_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50
MAX_HOLD_MS = 3600000   # 60 min safety net

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('stacked_exit_sweep')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'stacked_exit_sweep_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ==============================================================================
# SIGNAL PROCESSING (all causal, no look-ahead)
# ==============================================================================

def compute_trailing_vol(mid, window=3000):
    """Trailing realized vol (std of 1s returns) over window bars."""
    n = len(mid)
    ret_1s = np.zeros(n)
    ret_1s[10:] = (mid[10:] - mid[:-10]) / np.maximum(mid[:-10], 1e-10) * 10000
    vol = np.full(n, np.nan)
    cs = np.cumsum(ret_1s)
    cs2 = np.cumsum(ret_1s ** 2)
    idx = np.arange(window, n)
    if len(idx) > 0:
        s = cs[idx] - cs[idx - window]
        s2 = cs2[idx] - cs2[idx - window]
        m = s / window
        vol[window:] = np.sqrt(np.maximum(s2 / window - m * m, 0))
    return vol


def compute_expanding_vol_percentile(vol, pct):
    """Expanding percentile threshold for vol gating."""
    import pandas as pd
    return pd.Series(vol).expanding(min_periods=100).quantile(pct / 100.0).values


def zscore_expanding(arr):
    """Expanding-window z-score. Requires >=50 non-NaN samples."""
    n = len(arr)
    result = np.full(n, 0.0, dtype=np.float64)
    valid = ~np.isnan(arr)
    vals = np.where(valid, arr, 0.0)
    cs = np.cumsum(vals)
    cs2 = np.cumsum(vals ** 2)
    cc = np.cumsum(valid.astype(np.float64))
    mask = cc >= 50
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return result
    counts = cc[idx]
    means = cs[idx] / counts
    vars_ = cs2[idx] / counts - means * means
    stds = np.maximum(np.sqrt(np.maximum(vars_, 0)), 1e-8)
    result[idx] = (vals[idx] - means) / stds
    result[~valid] = 0.0
    return result


def zscore_ema(arr, span=5000):
    """EMA-based z-score. More responsive to recent data."""
    result = np.full_like(arr, 0.0, dtype=np.float64)
    alpha = 2.0 / (span + 1)
    ema_mean = 0.0
    ema_var = 0.0
    c = 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v) or v == 0.0:
            continue
        c += 1
        if c == 1:
            ema_mean = v
            ema_var = 0.0
            continue
        ema_mean = alpha * v + (1 - alpha) * ema_mean
        ema_var = alpha * (v - ema_mean) ** 2 + (1 - alpha) * ema_var
        if c >= 50:
            std = max(np.sqrt(ema_var), 1e-8)
            result[i] = (v - ema_mean) / std
    return result


def zscore_rolling(arr, window=5000):
    """Fixed-window rolling z-score. More stable than expanding."""
    n = len(arr)
    result = np.full(n, 0.0, dtype=np.float64)
    vals = np.nan_to_num(arr, nan=0.0)
    cs = np.cumsum(vals)
    cs2 = np.cumsum(vals ** 2)
    for i in range(window, n):
        s = cs[i] - cs[i - window]
        s2 = cs2[i] - cs2[i - window]
        m = s / window
        std = max(np.sqrt(max(s2 / window - m * m, 0)), 1e-8)
        result[i] = (vals[i] - m) / std
    return result


def rolling_mean_smooth(z_scores, window=50):
    """Causal rolling mean of z-scores."""
    import pandas as pd
    return pd.Series(z_scores).rolling(window, min_periods=1).mean().values


def compute_momentum(z_scores, diff_bars=50):
    """Z-score acceleration: z[i] - z[i-diff_bars]. Causal."""
    n = len(z_scores)
    accel = np.zeros(n, dtype=np.float64)
    for i in range(diff_bars, n):
        if z_scores[i] != 0.0 and z_scores[i - diff_bars] != 0.0:
            accel[i] = z_scores[i] - z_scores[i - diff_bars]
    return accel


def compute_pred_std(preds_aligned, window=3000):
    """Rolling std of raw predictions. Causal."""
    n = len(preds_aligned)
    pred_std = np.full(n, np.nan, dtype=np.float64)
    p = np.nan_to_num(preds_aligned, nan=0.0)
    cs = np.cumsum(p)
    cs2 = np.cumsum(p ** 2)
    idx = np.arange(window, n)
    if len(idx) > 0:
        s = cs[idx] - cs[idx - window]
        s2 = cs2[idx] - cs2[idx - window]
        m = s / window
        pred_std[window:] = np.sqrt(np.maximum(s2 / window - m * m, 0))
    return pred_std


def load_book_imbalance(date, n_bars):
    """Load book tensors, compute L1-L5 depth imbalance."""
    book_file = BOOK_DIR / f'{date}_book_tensors.npz'
    if not book_file.exists():
        return None
    bt = np.load(str(book_file))
    book = bt['book_tensors']  # (n_bars, 20, 4)
    if book.shape[0] != n_bars:
        if book.shape[0] > n_bars:
            book = book[:n_bars]
        else:
            pad = np.zeros((n_bars - book.shape[0], 20, 4), dtype=book.dtype)
            book = np.concatenate([book, pad], axis=0)
    bid_depth = book[:, :5, 1].sum(axis=1)
    ask_depth = book[:, 10:15, 1].sum(axis=1)
    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)
    del bt, book
    return imbalance


def time_mask(n_bars):
    """Skip first 30 min and last 15 min of RTH session."""
    secs = np.arange(n_bars) / BARS_PER_SEC
    mins = secs / 60.0
    return (mins >= 30) & (mins < 375)


# ==============================================================================
# HYSTERESIS ENGINE: entry fires -> signal active -> exit fires -> signal zero
# ==============================================================================

def _apply_hysteresis_python(z_score, entry_long, entry_short, exit_long, exit_short):
    """Hysteresis entry/exit logic (pure Python fallback)."""
    n = len(z_score)
    output = np.zeros(n, dtype=np.float64)
    in_long = False
    in_short = False
    for i in range(n):
        if in_long:
            if exit_long[i]:
                in_long = False
            else:
                output[i] = z_score[i]
        elif in_short:
            if exit_short[i]:
                in_short = False
            else:
                output[i] = z_score[i]
        if not in_long and not in_short:
            if entry_long[i]:
                in_long = True
                output[i] = z_score[i]
            elif entry_short[i]:
                in_short = True
                output[i] = z_score[i]
    return output


if HAS_NUMBA:
    @njit(cache=True)
    def apply_hysteresis(z_score, entry_long, entry_short, exit_long, exit_short):
        n = len(z_score)
        output = np.zeros(n, dtype=np.float64)
        in_long = False
        in_short = False
        for i in range(n):
            if in_long:
                if exit_long[i]:
                    in_long = False
                else:
                    output[i] = z_score[i]
            elif in_short:
                if exit_short[i]:
                    in_short = False
                else:
                    output[i] = z_score[i]
            if not in_long and not in_short:
                if entry_long[i]:
                    in_long = True
                    output[i] = z_score[i]
                elif entry_short[i]:
                    in_short = True
                    output[i] = z_score[i]
        return output
else:
    apply_hysteresis = _apply_hysteresis_python


# ==============================================================================
# ENTRY CONDITIONS
# ==============================================================================

def entry_smooth(smooth_z, threshold):
    """Rolling mean z-score exceeds entry_threshold."""
    return (smooth_z > threshold), (smooth_z < -threshold)


def entry_ema(ema_z, threshold):
    """EMA z-score exceeds entry_threshold."""
    return (ema_z > threshold), (ema_z < -threshold)


def entry_book(z_score, imbalance, threshold):
    """Expanding z-score > threshold AND book imbalance confirms."""
    if imbalance is None:
        return np.zeros(len(z_score), dtype=bool), np.zeros(len(z_score), dtype=bool)
    return (z_score > threshold) & (imbalance > 0), (z_score < -threshold) & (imbalance < 0)


def entry_momentum(momentum, threshold):
    """Z-score acceleration exceeds accel_threshold."""
    return (momentum > threshold), (momentum < -threshold)


def entry_raw(preds_aligned, threshold):
    """Raw prediction exceeds threshold (no z-score normalization)."""
    return (preds_aligned > threshold), (preds_aligned < -threshold)


# ==============================================================================
# EXIT CONDITIONS
# ==============================================================================

def exit_smooth(smooth_z, threshold):
    """Rolling mean z-score drops below exit_threshold."""
    return smooth_z < threshold, smooth_z > -threshold


def exit_book(imbalance):
    """Book imbalance flips against position direction."""
    if imbalance is None:
        return None, None
    return imbalance < 0, imbalance > 0


def exit_momentum(momentum):
    """Z-score acceleration reverses sign."""
    return momentum < 0, momentum > 0


def exit_predstd(pred_std, max_std=0.10):
    """Pred_std exceeds max_std (model confused)."""
    confused = pred_std > max_std
    return confused, confused


def exit_raw(preds_aligned, threshold):
    """Raw prediction drops below threshold."""
    return preds_aligned < threshold, preds_aligned > -threshold


# ==============================================================================
# STACKED EXIT COMBO DEFINITIONS
# ==============================================================================

# The 5 winning (entry, dynamic_exit) pairs from the matrix sweep
# Plus new raw/rolling variants for normalization sensitivity testing
ENTRY_EXIT_PAIRS = [
    # (entry_type, exit_type, entry_param_key, exit_param_key, base_label)
    ('smooth',   'smooth_exit',   None, 'exit_thr',   'smooth_smoothExit'),   # Sharpe 4.62
    ('smooth',   'book_exit',     None, None,          'smooth_bookExit'),     # Sharpe 3.70
    ('momentum', 'ema_exit',      None, 'exit_thr',    'mom_emaExit'),         # Sharpe 3.79
    ('book',     'predstd_exit',  None, None,          'book_predstdExit'),    # Sharpe 3.48
    ('ema',      'book_exit',     None, None,          'ema_bookExit'),        # Sharpe 2.88
    ('raw',      'raw_exit',      None, 'exit_thr',    'raw_rawExit'),         # Raw pred, no z-score
    ('raw',      'smooth_exit',   None, 'exit_thr',    'raw_smoothExit'),      # Raw entry, smooth exit
    ('rolling',  'smooth_exit',   None, 'exit_thr',    'rolling_smoothExit'),  # Rolling z-score entry
]

# TP ticks (None = no TP, Rust sim ignores)
TP_VALUES = [None, 5, 8, 10, 15, 20]

# Trailing stop ticks (None = no trailing stop)
SL_VALUES = [None, 10, 15, 20, 25]

# Vol gates (0 = no vol filtering)
VOL_GATES = [0, 50, 70]

# Entry thresholds (conviction z-score threshold)
ENTRY_THRESHOLDS = [1.5, 2.0, 2.5]

# Raw entry thresholds (different scale from z-score — raw predictions are ~0.0-0.3 range)
RAW_ENTRY_THRESHOLDS = [0.05, 0.10, 0.15]

# Exit thresholds (for smooth_exit and ema_exit only)
EXIT_THRESHOLDS = [0.0, 0.5]

# Momentum entry uses accel threshold (different scale)
MOMENTUM_ENTRY_THRESHOLD = 0.3  # fixed from matrix sweep winner


def build_sim_configs():
    """
    Build full list of simulation configs.
    Each config = (entry_exit_pair_idx, entry_thr, exit_thr, vol_gate, tp, sl, label)
    """
    configs = []

    for pair_idx, (entry_type, exit_type, _, exit_param_key, base_label) in enumerate(ENTRY_EXIT_PAIRS):
        # Determine which exit thresholds to sweep
        if exit_param_key == 'exit_thr':
            exit_thrs = EXIT_THRESHOLDS
        else:
            exit_thrs = [None]  # book_exit, predstd_exit don't have a threshold param

        # Raw entry types use RAW_ENTRY_THRESHOLDS (different scale from z-score)
        if entry_type in ('raw',):
            entry_thr_list = RAW_ENTRY_THRESHOLDS
        else:
            entry_thr_list = ENTRY_THRESHOLDS

        for entry_thr in entry_thr_list:
            # Momentum entry uses fixed accel threshold
            if entry_type == 'momentum':
                actual_entry_thr = MOMENTUM_ENTRY_THRESHOLD
                if entry_thr != ENTRY_THRESHOLDS[0]:
                    continue  # Only use fixed threshold once for momentum
            else:
                actual_entry_thr = entry_thr

            for exit_thr in exit_thrs:
                for vg in VOL_GATES:
                    for tp in TP_VALUES:
                        for sl in SL_VALUES:
                            # Build label
                            parts = [base_label, f'conv{entry_thr}']
                            if exit_thr is not None:
                                parts.append(f'ethr{exit_thr}')
                            parts.append(f'vol{vg}')
                            if tp is not None:
                                parts.append(f'tp{tp}')
                            else:
                                parts.append('tpN')
                            if sl is not None:
                                parts.append(f'sl{sl}')
                            else:
                                parts.append('slN')
                            label = '_'.join(parts)

                            configs.append({
                                'pair_idx': pair_idx,
                                'entry_type': entry_type,
                                'exit_type': exit_type,
                                'entry_thr': actual_entry_thr,
                                'exit_thr': exit_thr,
                                'vol_gate': vg,
                                'tp': tp,
                                'sl': sl,
                                'label': label,
                            })

    log.info(f"Built {len(configs)} stacked exit configs")
    return configs


def group_configs_by_pred_key(configs):
    """
    Group configs that share the same prediction file.
    Prediction file is determined by (pair_idx, entry_thr, exit_thr, vol_gate).
    Different TP/SL values use the SAME prediction file but different Rust sim flags.
    """
    groups = defaultdict(list)
    for cfg in configs:
        pred_key = (cfg['pair_idx'], cfg['entry_thr'], cfg['exit_thr'], cfg['vol_gate'])
        groups[pred_key].append(cfg)
    return groups


# ==============================================================================
# PREDICTION FILE GENERATION
# ==============================================================================

def generate_predictions_for_date(date, wf_data, config_groups):
    """
    Generate prediction files for one date, one per unique (entry, exit, vol) combo.
    Returns dict of pred_key -> filepath.
    """
    preds_raw = wf_data[f'{date}_preds'].astype(np.float64)
    mid = wf_data[f'{date}_mid'].astype(np.float64)
    n_bars = len(mid)
    if n_bars < 5000:
        return {}

    # Check MBO exists
    nodash = date.replace('-', '')
    if not (MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst').exists() and \
       not (MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn').exists():
        return {}

    # ── Pre-compute all derived signals (once per date) ──
    aligned = np.zeros(n_bars, dtype=np.float64)
    end = min(n_bars, len(preds_raw) + CNN_OFFSET)
    aligned[CNN_OFFSET:end] = preds_raw[:end - CNN_OFFSET]

    z_score = zscore_expanding(aligned)
    rolling_z = zscore_rolling(aligned, window=5000)
    ema_z = zscore_ema(aligned, span=5000)
    momentum = compute_momentum(z_score, diff_bars=50)
    smooth_z = rolling_mean_smooth(z_score, window=50)
    rolling_smooth_z = rolling_mean_smooth(rolling_z, window=50)
    pred_std = compute_pred_std(aligned, window=3000)
    imbalance = load_book_imbalance(date, n_bars)

    vol = compute_trailing_vol(mid)
    vol_pct = {}
    for vg in VOL_GATES:
        if vg > 0:  # vg=0 means no filtering; skip expensive percentile computation
            vol_pct[vg] = compute_expanding_vol_percentile(vol, vg)

    tmask = time_mask(n_bars)

    saved = {}

    for pred_key, cfgs in config_groups.items():
        pair_idx, entry_thr, exit_thr, vg = pred_key
        entry_type, exit_type = ENTRY_EXIT_PAIRS[pair_idx][0], ENTRY_EXIT_PAIRS[pair_idx][1]

        # ── Compute entry conditions ──
        if entry_type == 'smooth':
            entry_long, entry_short = entry_smooth(smooth_z, entry_thr)
        elif entry_type == 'ema':
            entry_long, entry_short = entry_ema(ema_z, entry_thr)
        elif entry_type == 'book':
            entry_long, entry_short = entry_book(z_score, imbalance, entry_thr)
        elif entry_type == 'momentum':
            entry_long, entry_short = entry_momentum(momentum, entry_thr)
        elif entry_type == 'raw':
            # Raw predictions directly; entry_thr on raw scale (RAW_ENTRY_THRESHOLDS)
            entry_long, entry_short = entry_raw(aligned, entry_thr)
        elif entry_type == 'rolling':
            # Rolling z-score entry (fixed 5000-bar window, smoothed)
            entry_long, entry_short = entry_smooth(rolling_smooth_z, entry_thr)
        else:
            continue

        # ── Compute exit conditions ──
        if exit_type == 'smooth_exit':
            thr = exit_thr if exit_thr is not None else 0.0
            exit_long, exit_short = exit_smooth(smooth_z, thr)
        elif exit_type == 'book_exit':
            el, es = exit_book(imbalance)
            if el is None:
                continue  # No book data for this date
            exit_long, exit_short = el, es
        elif exit_type == 'ema_exit':
            thr = exit_thr if exit_thr is not None else 0.0
            # EMA exit: long exits when ema_z < threshold, short exits when ema_z > -threshold
            exit_long = ema_z < thr
            exit_short = ema_z > -thr
        elif exit_type == 'momentum_exit':
            exit_long, exit_short = exit_momentum(momentum)
        elif exit_type == 'predstd_exit':
            exit_long, exit_short = exit_predstd(pred_std, 0.10)
        elif exit_type == 'raw_exit':
            # Exit when raw prediction drops below entry threshold (signal fades)
            thr = exit_thr if exit_thr is not None else 0.0
            exit_long, exit_short = exit_raw(aligned, thr)
        else:
            continue

        # Ensure arrays are correct length
        for arr_name in ['exit_long', 'exit_short']:
            arr = locals()[arr_name]
            if len(arr) != n_bars:
                locals()[arr_name] = np.broadcast_to(arr, n_bars).copy()

        # ── Choose the z-score carrier for the hysteresis signal value ──
        # Raw/rolling entry types carry the signal on their own z-score
        if entry_type in ('raw',):
            signal_carrier = rolling_z  # use rolling z as neutral carrier
        elif entry_type == 'rolling':
            signal_carrier = rolling_z
        else:
            signal_carrier = z_score

        # ── Apply hysteresis: signal active between entry and exit ──
        signal = apply_hysteresis(
            signal_carrier,
            entry_long.astype(np.bool_),
            entry_short.astype(np.bool_),
            exit_long.astype(np.bool_),
            exit_short.astype(np.bool_),
        )

        # ── Apply vol gate ──
        if vg > 0 and vg in vol_pct:
            vt = vol_pct[vg]
            vol_mask = np.isnan(vol) | (vol < vt)
            signal[vol_mask] = 0.0

        # ── Apply time mask ──
        signal[~tmask] = 0.0

        signal = np.nan_to_num(signal, nan=0.0).astype(np.float32)

        # Build a compact key for the filename
        ethr_str = f'_ethr{exit_thr}' if exit_thr is not None else ''
        fname = f'{date}_{ENTRY_EXIT_PAIRS[pair_idx][4]}_conv{entry_thr}{ethr_str}_vol{vg}.npz'
        fpath = PRED_OUT_DIR / fname
        np.savez_compressed(str(fpath), predictions=signal)
        saved[pred_key] = str(fpath)

    # Cleanup
    del mid, preds_raw, aligned, z_score, rolling_z, ema_z, momentum, smooth_z, rolling_smooth_z, pred_std, vol
    if imbalance is not None:
        del imbalance
    gc.collect()

    return saved


def generate_all_predictions(wf_data, dates, config_groups):
    """Generate prediction files for all dates."""
    all_saved = {}  # (pred_key, date) -> filepath
    for di, date in enumerate(dates):
        day_saved = generate_predictions_for_date(date, wf_data, config_groups)
        for pred_key, fpath in day_saved.items():
            all_saved[(pred_key, date)] = fpath
        if (di + 1) % 5 == 0 or di == 0 or di == len(dates) - 1:
            log.info(f"  Generated {di+1}/{len(dates)} days ({len(all_saved)} files total)")
    log.info(f"Total prediction files: {len(all_saved)}")
    return all_saved


# ==============================================================================
# SIMULATION
# ==============================================================================

def run_single_sim(mbo_file, pred_file, output_file, tp_ticks, sl_ticks):
    """
    Run one Rust fill_sim job with stacked exits.
    Signal threshold = 0.1 (pre-filtered by hysteresis in prediction file).
    Chase 1t/3r. Hold 60min safety. Signal-flip-exit enabled.
    """
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(output_file),
        '--hold-ms', str(MAX_HOLD_MS),
        '--signal-threshold', '0.1',
        '--latency-ms', '0',
        '--chase-entry',
        '--chase-max-ticks', '1',
        '--chase-max-reprices', '3',
        '--signal-flip-exit',  # CRITICAL: dynamic exit via signal->0
        '--quiet',
    ]

    if tp_ticks is not None:
        cmd += ['--take-profit-ticks', str(tp_ticks)]

    if sl_ticks is not None:
        cmd += ['--trailing-ticks', str(sl_ticks)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        pass
    return None


def run_sweep(pred_files, configs, config_groups, workers=24):
    """
    Run all sim jobs in parallel.
    Each config uses its corresponding prediction file + specific TP/SL Rust flags.
    """
    jobs = []  # (type, label, date, data)

    # Index pred_files by pred_key for O(1) lookup
    pred_by_key = defaultdict(dict)  # pred_key -> {date: path}
    for (pk, date), pred_path in pred_files.items():
        pred_by_key[pk][date] = pred_path

    # Pre-check MBO files
    mbo_cache = {}
    for date_files in pred_by_key.values():
        for date in date_files:
            if date in mbo_cache:
                continue
            nodash = date.replace('-', '')
            mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
            if not mbo_file.exists():
                mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
            mbo_cache[date] = str(mbo_file) if mbo_file.exists() else None

    for cfg in configs:
        pred_key = (cfg['pair_idx'], cfg['entry_thr'], cfg['exit_thr'], cfg['vol_gate'])
        date_preds = pred_by_key.get(pred_key, {})

        for date, pred_path in date_preds.items():
            mbo_str = mbo_cache.get(date)
            if mbo_str is None:
                continue

            out_file = SIM_OUT_DIR / f'{cfg["label"]}_{date}.json'

            if out_file.exists():
                jobs.append(('cached', cfg['label'], date, str(out_file)))
            else:
                jobs.append(('run', cfg['label'], date, {
                    'mbo': mbo_str,
                    'pred': pred_path,
                    'out': str(out_file),
                    'tp': cfg['tp'],
                    'sl': cfg['sl'],
                }))

    # Load cached results
    results = defaultdict(dict)
    cached_count = 0
    for job_type, label, date, data in jobs:
        if job_type == 'cached':
            try:
                with open(data) as f:
                    res = json.load(f)
                results[label][date] = res
                cached_count += 1
            except:
                pass

    to_run = [(label, date, params) for jt, label, date, params in jobs if jt == 'run']

    log.info(f"Total sim jobs: {len(jobs)} ({cached_count} cached, {len(to_run)} to run)")
    log.info(f"Workers: {workers}")

    if not to_run:
        log.info("All jobs cached.")
        return dict(results)

    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for label, date, params in to_run:
            f = executor.submit(
                run_single_sim,
                params['mbo'], params['pred'], params['out'],
                params['tp'], params['sl']
            )
            futures[f] = (label, date)

        for future in as_completed(futures):
            label, date = futures[future]
            completed += 1
            try:
                res = future.result()
                if res:
                    results[label][date] = res
            except Exception as e:
                log.warning(f"Job error {label}/{date}: {e}")

            if completed % 200 == 0 or completed == len(to_run):
                el = time.time() - t0
                rate = completed / el if el > 0 else 0
                eta = (len(to_run) - completed) / rate / 60 if rate > 0 else 0
                log.info(f"  [{completed}/{len(to_run)}] {rate:.1f} jobs/s, ETA {eta:.1f}min")

    log.info(f"Simulation done: {completed} jobs in {time.time()-t0:.0f}s")
    return dict(results)


# ==============================================================================
# AGGREGATION & EXIT TYPE ANALYSIS
# ==============================================================================

def analyze_exit_distribution(date_results):
    """
    Analyze per-trade exit reasons across all days for one config.
    Returns dict with exit type counts and avg hold times.
    """
    exit_counts = defaultdict(int)
    exit_hold_times = defaultdict(list)
    exit_pnls = defaultdict(list)
    total_trades = 0

    for date_str, res in date_results.items():
        if 'trades' not in res:
            continue
        for trade in res['trades']:
            reason = trade.get('exit_reason', 'Unknown')
            hold_ms = trade.get('hold_time_ms', 0)
            pnl = trade.get('pnl_dollars', 0)
            exit_counts[reason] += 1
            exit_hold_times[reason].append(hold_ms)
            exit_pnls[reason].append(pnl)
            total_trades += 1

    if total_trades == 0:
        return None

    analysis = {
        'total_trades': total_trades,
        'exit_types': {},
    }

    for reason in sorted(exit_counts.keys()):
        count = exit_counts[reason]
        holds = exit_hold_times[reason]
        pnls = exit_pnls[reason]
        analysis['exit_types'][reason] = {
            'count': count,
            'pct': round(count / total_trades * 100, 1),
            'avg_hold_ms': round(np.mean(holds), 0) if holds else 0,
            'avg_hold_min': round(np.mean(holds) / 60000, 2) if holds else 0,
            'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
            'total_pnl': round(sum(pnls), 2),
            'win_rate': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1) if pnls else 0,
        }

    return analysis


def parse_stacked_label(label):
    """Parse label like 'smooth_smoothExit_conv1.5_ethr0.0_vol50_tp5_sl10' into components."""
    parts = label.split('_')
    result = {
        'entry_exit_pair': '',
        'conv': '',
        'exit_thr': None,
        'vol': '',
        'tp': 'none',
        'sl': 'none',
    }

    # Find key parts
    pair_parts = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p.startswith('conv'):
            result['conv'] = p
            i += 1
        elif p.startswith('ethr'):
            result['exit_thr'] = p
            i += 1
        elif p.startswith('vol'):
            result['vol'] = p
            i += 1
        elif p.startswith('tp'):
            result['tp'] = p
            i += 1
        elif p.startswith('sl'):
            result['sl'] = p
            i += 1
        else:
            pair_parts.append(p)
            i += 1
    result['entry_exit_pair'] = '_'.join(pair_parts)
    return result


def aggregate_and_report(results):
    """Aggregate per-day results, analyze exit distributions, print rankings."""
    summaries = []

    for config_label, date_results in results.items():
        if not date_results:
            continue

        daily_pnls = []
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0
        all_trade_pnls = []

        for date_str, res in sorted(date_results.items()):
            day_pnl = res.get('total_pnl_dollars', 0)
            daily_pnls.append(day_pnl)
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            if 'trades' in res:
                for trade in res['trades']:
                    pnl = trade.get('pnl_dollars', 0)
                    all_trade_pnls.append(pnl)
                    if pnl > 0:
                        total_wins += 1

        n_days = len(daily_pnls)
        if n_days == 0 or total_trades == 0:
            continue

        total_pnl = sum(daily_pnls)
        avg_daily = np.mean(daily_pnls)
        std_daily = np.std(daily_pnls) if n_days > 1 else 1e-8
        sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
        win_rate = total_wins / total_trades
        fill_rate = total_filled / total_signals if total_signals > 0 else 0

        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = abs((cum - peak).min()) if len(cum) > 0 else 0

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_trade_ticks = avg_trade_pnl / TICK_VALUE

        # Exit distribution analysis
        exit_analysis = analyze_exit_distribution(date_results)

        parsed = parse_stacked_label(config_label)

        summaries.append({
            'config': config_label,
            'pair': parsed['entry_exit_pair'],
            'conv': parsed['conv'],
            'exit_thr': parsed.get('exit_thr', ''),
            'vol': parsed['vol'],
            'tp_label': parsed['tp'],
            'sl_label': parsed['sl'],
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe': round(sharpe, 3),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_trade_ticks': round(avg_trade_ticks, 3),
            'max_dd': round(max_dd, 2),
            'annualized': round(avg_daily * 252, 0),
            'exit_analysis': exit_analysis,
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)

    if not summaries:
        log.warning("No configs produced trades!")
        return summaries

    # ──────────────────────────────────────────────────────────────────────
    # 1) TOP 30 BY SHARPE
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 160)
    log.info("STACKED EXIT SWEEP — TOP 30 BY SHARPE (Rust MBO Fill Sim)")
    log.info("4 exit layers: Dynamic Signal + TP + Trailing Stop + 60min Safety")
    log.info("=" * 160)
    log.info(f"{'#':>3} {'Pair':<22} {'Conv':<7} {'Vol':<5} {'TP':<5} {'SL':<5} "
             f"{'Sharpe':>7} {'P&L':>10} {'Trades':>6} {'Fill%':>6} "
             f"{'WR%':>5} {'MaxDD':>8} {'Annual':>10}")
    log.info("-" * 160)

    for i, s in enumerate(summaries[:30]):
        log.info(
            f"{i+1:>3} {s['pair']:<22} {s['conv']:<7} {s['vol']:<5} "
            f"{s['tp_label']:<5} {s['sl_label']:<5} "
            f"{s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} {s['n_trades']:>6} "
            f"{s['fill_rate']*100:>5.1f}% {s['win_rate']*100:>4.1f}% "
            f"${s['max_dd']:>7,.0f} ${s['annualized']:>9,.0f}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 2) EXIT TYPE DISTRIBUTION for top 30
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 160)
    log.info("EXIT TYPE DISTRIBUTION — Top 30 configs")
    log.info("SignalFlip = dynamic exit (encoded in prediction). TP/Trailing/Timeout = Rust sim layers.")
    log.info("=" * 160)
    log.info(f"{'#':>3} {'Config':<50} {'SignalFlip%':>10} {'TP%':>6} {'Trail%':>7} "
             f"{'Timeout%':>8} {'Other%':>7} {'AvgHold(min)':>13}")
    log.info("-" * 160)

    for i, s in enumerate(summaries[:30]):
        ea = s.get('exit_analysis')
        if ea is None:
            continue
        et = ea['exit_types']
        total = ea['total_trades']

        sf_pct = et.get('SignalFlip', {}).get('pct', 0)
        tp_pct = et.get('TakeProfit', {}).get('pct', 0)
        tr_pct = et.get('TrailingStop', {}).get('pct', 0)
        ht_pct = et.get('HoldTimeout', {}).get('pct', 0)
        other_pct = 100 - sf_pct - tp_pct - tr_pct - ht_pct

        # Average hold time across all exit types
        all_holds = []
        for etype, edata in et.items():
            all_holds.extend([edata['avg_hold_min']] * edata['count'])
        avg_hold = np.mean(all_holds) if all_holds else 0

        short_label = s['config'][:50]
        log.info(
            f"{i+1:>3} {short_label:<50} {sf_pct:>9.1f}% {tp_pct:>5.1f}% "
            f"{tr_pct:>6.1f}% {ht_pct:>7.1f}% {other_pct:>6.1f}% "
            f"{avg_hold:>12.1f}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # 3) AVERAGE HOLD TIME BY EXIT TYPE (top 30)
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 120)
    log.info("AVERAGE HOLD TIME BY EXIT TYPE (across top 30 configs)")
    log.info("=" * 120)

    global_exit_holds = defaultdict(list)
    global_exit_pnls = defaultdict(list)
    for s in summaries[:30]:
        ea = s.get('exit_analysis')
        if ea is None:
            continue
        for etype, edata in ea['exit_types'].items():
            global_exit_holds[etype].append(edata['avg_hold_min'])
            global_exit_pnls[etype].append(edata['avg_pnl'])

    log.info(f"{'Exit Type':<20} {'Avg Hold (min)':>15} {'Avg P&L/Trade':>15} {'Occurrences':>12}")
    log.info("-" * 65)
    for etype in ['SignalFlip', 'TakeProfit', 'TrailingStop', 'HoldTimeout', 'MarketExit']:
        if etype in global_exit_holds:
            log.info(
                f"{etype:<20} {np.mean(global_exit_holds[etype]):>14.2f} "
                f"${np.mean(global_exit_pnls[etype]):>13.2f} "
                f"{len(global_exit_holds[etype]):>12}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # 4) BEST PER ENTRY-EXIT PAIR (with best TP/SL combo)
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 120)
    log.info("BEST CONFIG PER ENTRY-EXIT PAIR")
    log.info("=" * 120)

    pair_best = {}
    for s in summaries:
        pair = s['pair']
        if pair not in pair_best or s['sharpe'] > pair_best[pair]['sharpe']:
            pair_best[pair] = s

    for pair, s in sorted(pair_best.items(), key=lambda x: x[1]['sharpe'], reverse=True):
        ea = s.get('exit_analysis', {})
        et = ea.get('exit_types', {}) if ea else {}
        sf_pct = et.get('SignalFlip', {}).get('pct', 0)
        tp_pct = et.get('TakeProfit', {}).get('pct', 0)
        tr_pct = et.get('TrailingStop', {}).get('pct', 0)

        log.info(f"\n  {pair}:")
        log.info(f"    Best: {s['conv']}, {s['vol']}, {s['tp_label']}, {s['sl_label']}")
        log.info(f"    Sharpe: {s['sharpe']:.2f}, P&L: ${s['total_pnl']:,.2f}, "
                 f"Trades: {s['n_trades']}, Fill: {s['fill_rate']:.1%}, "
                 f"WR: {s['win_rate']:.1%}, Annual: ${s['annualized']:,.0f}")
        log.info(f"    Exit mix: Signal {sf_pct:.0f}% | TP {tp_pct:.0f}% | Trail {tr_pct:.0f}%")

    # ──────────────────────────────────────────────────────────────────────
    # 5) TP vs SL HEATMAP (best Sharpe per TP x SL combo, averaged across pairs)
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 100)
    log.info("TP x SL HEATMAP — Average Sharpe across all pairs+convictions")
    log.info("=" * 100)

    tp_sl_sharpes = defaultdict(list)
    for s in summaries:
        tp_sl_sharpes[(s['tp_label'], s['sl_label'])].append(s['sharpe'])

    tp_labels = sorted(set(s['tp_label'] for s in summaries),
                       key=lambda x: -1 if x == 'tpN' else int(x.replace('tp', '')))
    sl_labels = sorted(set(s['sl_label'] for s in summaries),
                       key=lambda x: -1 if x == 'slN' else int(x.replace('sl', '')))

    tp_sl_label = 'TP \\ SL'
    header = f"{tp_sl_label:<8}"
    for sl in sl_labels:
        header += f" {sl:>8}"
    log.info(header)
    log.info("-" * (8 + 9 * len(sl_labels)))

    for tp in tp_labels:
        row = f"{tp:<8}"
        for sl in sl_labels:
            key = (tp, sl)
            if key in tp_sl_sharpes:
                avg_s = np.mean(tp_sl_sharpes[key])
                row += f" {avg_s:>8.2f}"
            else:
                row += f" {'---':>8}"
        log.info(row)

    # ──────────────────────────────────────────────────────────────────────
    # 6) REFERENCE
    # ──────────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 100)
    log.info("REFERENCE — Baseline (no stacked exits):")
    log.info("  IS best (vol70/conv2.5/1t/3r/30min):    Sharpe 3.28, +$15,479/74d, 130 trades")
    log.info("  OOT static:                             Sharpe 1.58, +$4,082/68d, 102 trades")
    log.info("  Matrix best (smooth+smooth_exit):       Sharpe 4.62 (hold only, no TP/SL)")
    log.info("=" * 100)

    # ──────────────────────────────────────────────────────────────────────
    # SAVE RESULTS
    # ──────────────────────────────────────────────────────────────────────
    out_file = RESULTS_DIR / f'stacked_exit_sweep_results_{_ts}.json'

    # Strip exit_analysis from JSON (too verbose), keep summary stats
    clean_summaries = []
    for s in summaries:
        cs = {k: v for k, v in s.items() if k != 'exit_analysis'}
        ea = s.get('exit_analysis')
        if ea:
            et = ea['exit_types']
            cs['exit_pct_signal_flip'] = et.get('SignalFlip', {}).get('pct', 0)
            cs['exit_pct_take_profit'] = et.get('TakeProfit', {}).get('pct', 0)
            cs['exit_pct_trailing_stop'] = et.get('TrailingStop', {}).get('pct', 0)
            cs['exit_pct_hold_timeout'] = et.get('HoldTimeout', {}).get('pct', 0)
            cs['avg_hold_min_signal_flip'] = et.get('SignalFlip', {}).get('avg_hold_min', 0)
            cs['avg_hold_min_take_profit'] = et.get('TakeProfit', {}).get('avg_hold_min', 0)
            cs['avg_hold_min_trailing_stop'] = et.get('TrailingStop', {}).get('avg_hold_min', 0)
        clean_summaries.append(cs)

    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'n_configs': len(summaries),
            'n_profitable': sum(1 for s in summaries if s['total_pnl'] > 0),
            'tp_values': [str(t) for t in TP_VALUES],
            'sl_values': [str(t) for t in SL_VALUES],
            'entry_exit_pairs': [p[4] for p in ENTRY_EXIT_PAIRS],
            'summaries': clean_summaries,
        }, f, indent=2)
    log.info(f"\nResults saved: {out_file}")

    return summaries


# ==============================================================================
# SATURN UPLOAD & REMOTE SWEEP
# ==============================================================================

def upload_to_saturn(pred_files):
    """Upload prediction NPZs to Saturn for parallel sweep."""
    import paramiko

    SATURN_HOST = 'jupiter'  # Via Jupiter gateway
    JUPITER_HOST = 'jupiter'
    JUPITER_USER = 'jupiter'
    JUPITER_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
    SATURN_USER = 'saturn'
    SATURN_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
    REMOTE_DIR = '/home/saturn/Lvl3Quant/data/processed/cnn_wf_stacked_predictions'

    log.info(f"Uploading {len(pred_files)} prediction files via Jupiter->Saturn...")

    try:
        # Connect to Jupiter first
        ssh_jup = paramiko.SSHClient()
        ssh_jup.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_jup.connect(JUPITER_HOST, username=JUPITER_USER, password=JUPITER_PW, timeout=10)

        # Tunnel to Saturn
        transport = ssh_jup.get_transport()
        channel = transport.open_channel('direct-tcpip', ('saturn', 22), ('127.0.0.1', 0))

        ssh_sat = paramiko.SSHClient()
        ssh_sat.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_sat.connect('saturn', username=SATURN_USER, password=SATURN_PW, sock=channel, timeout=10)

        ssh_sat.exec_command(f'mkdir -p {REMOTE_DIR}')
        time.sleep(1)

        sftp = ssh_sat.open_sftp()
        uploaded = 0
        for (pred_key, date), local_path in pred_files.items():
            remote_path = f'{REMOTE_DIR}/{Path(local_path).name}'
            try:
                sftp.put(local_path, remote_path)
                uploaded += 1
                if uploaded % 100 == 0:
                    log.info(f"  Uploaded {uploaded}/{len(pred_files)}")
            except Exception as e:
                log.warning(f"  Upload failed {Path(local_path).name}: {e}")

        sftp.close()
        ssh_sat.close()
        ssh_jup.close()
        log.info(f"Uploaded {uploaded} files to Saturn:{REMOTE_DIR}")
        return True
    except Exception as e:
        log.error(f"Saturn upload failed: {e}")
        return False


def launch_saturn_sweep():
    """Launch stacked exit sweep on Saturn with 40 workers."""
    import paramiko

    JUPITER_HOST = 'jupiter'
    JUPITER_USER = 'jupiter'
    JUPITER_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
    SATURN_USER = 'saturn'
    SATURN_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")

    remote_script = '''#!/usr/bin/env python3
"""Remote stacked exit sweep on Saturn."""
import sys, json, time, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

LVL3_ROOT = Path("/home/saturn/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# TP x SL grid
TP_VALUES = [None, 5, 8, 10, 15, 20]
SL_VALUES = [None, 10, 15, 20, 25]

def run_sim(mbo, pred, out, tp, sl):
    cmd = [str(BINARY), "--mbo-file", str(mbo), "--predictions", str(pred),
           "--output", str(out), "--hold-ms", "3600000", "--signal-threshold", "0.1",
           "--latency-ms", "0", "--quiet", "--chase-entry",
           "--chase-max-ticks", "1", "--chase-max-reprices", "3",
           "--signal-flip-exit"]
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    if sl is not None:
        cmd += ["--trailing-ticks", str(sl)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(out).exists():
            with open(out) as f:
                return json.load(f)
    except:
        pass
    return None

pred_files = sorted(PRED_DIR.glob("*.npz"))
print(f"Found {len(pred_files)} prediction files")

jobs = []
for pf in pred_files:
    stem = pf.stem
    date = stem[:10]
    combo_label = stem[11:]
    nodash = date.replace("-", "")
    mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
    if not mbo.exists():
        mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn"
    if not mbo.exists():
        continue
    for tp in TP_VALUES:
        for sl in SL_VALUES:
            tp_str = f"tp{tp}" if tp is not None else "tpN"
            sl_str = f"sl{sl}" if sl is not None else "slN"
            full_label = f"{combo_label}_{tp_str}_{sl_str}"
            out_file = OUT_DIR / f"{full_label}_{date}.json"
            if out_file.exists():
                continue
            jobs.append((str(mbo), str(pf), str(out_file), tp, sl))

print(f"Jobs to run: {len(jobs)}")
done = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=40) as executor:
    futures = {}
    for mbo, pred, out, tp, sl in jobs:
        f = executor.submit(run_sim, mbo, pred, out, tp, sl)
        futures[f] = out

    for future in as_completed(futures):
        done += 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(jobs) - done) / rate / 60
            print(f"  [{done}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.1f}min")

print(f"Done: {done} jobs in {time.time()-t0:.0f}s")
'''

    try:
        ssh_jup = paramiko.SSHClient()
        ssh_jup.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_jup.connect(JUPITER_HOST, username=JUPITER_USER, password=JUPITER_PW, timeout=10)

        transport = ssh_jup.get_transport()
        channel = transport.open_channel('direct-tcpip', ('saturn', 22), ('127.0.0.1', 0))

        ssh_sat = paramiko.SSHClient()
        ssh_sat.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_sat.connect('saturn', username=SATURN_USER, password=SATURN_PW, sock=channel, timeout=10)

        script_path = '/home/saturn/Lvl3Quant/run_stacked_sweep_remote.py'
        sftp = ssh_sat.open_sftp()
        with sftp.open(script_path, 'w') as f:
            f.write(remote_script)
        sftp.close()

        cmd = f'cd /home/saturn/Lvl3Quant && nohup python3 {script_path} > stacked_sweep_remote.log 2>&1 &'
        ssh_sat.exec_command(cmd)
        log.info(f"Launched Saturn sweep: {cmd}")
        log.info(f"Monitor: ssh saturn 'tail -f /home/saturn/Lvl3Quant/stacked_sweep_remote.log'")

        ssh_sat.close()
        ssh_jup.close()
        return True
    except Exception as e:
        log.error(f"Saturn launch failed: {e}")
        return False


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Stacked Exit Sweep — 4-layer exit system')
    parser.add_argument('--workers', type=int, default=24, help='Parallel sim workers')
    parser.add_argument('--skip-gen', action='store_true', help='Skip prediction file generation')
    parser.add_argument('--skip-sim', action='store_true', help='Skip simulation (aggregate only)')
    parser.add_argument('--upload-saturn', action='store_true', help='Upload preds + launch on Saturn')
    args = parser.parse_args()

    log.info("=" * 80)
    log.info("STACKED EXIT SWEEP — 4-Layer Exit System")
    log.info("  Layer 1: Dynamic signal exit (SignalFlip — encoded in prediction)")
    log.info("  Layer 2: Take-profit (Rust sim --take-profit-ticks)")
    log.info("  Layer 3: Trailing stop (Rust sim --trailing-ticks)")
    log.info("  Layer 4: Max hold timeout (60 min safety)")
    log.info("=" * 80)
    log.info(f"  Binary:      {BINARY}")
    log.info(f"  Predictions: {PRED_FILE}")
    log.info(f"  MBO data:    {MBO_DIR}")
    log.info(f"  Book data:   {BOOK_DIR}")
    log.info(f"  Pred out:    {PRED_OUT_DIR}")
    log.info(f"  Sim out:     {SIM_OUT_DIR}")
    log.info(f"  Workers:     {args.workers}")
    log.info("=" * 80)

    # Verify prerequisites
    if not BINARY.exists():
        log.error(f"fill_sim_cli.exe not found: {BINARY}")
        sys.exit(1)
    if not PRED_FILE.exists():
        log.error(f"WF predictions file not found: {PRED_FILE}")
        sys.exit(1)
    if not MBO_DIR.exists():
        log.error(f"MBO directory not found: {MBO_DIR}")
        sys.exit(1)

    # Build config list
    configs = build_sim_configs()
    config_groups = group_configs_by_pred_key(configs)
    log.info(f"Unique prediction files needed: {len(config_groups)}")
    log.info(f"Total sim configs (pred x TP x SL): {len(configs)}")

    # Step 1: Generate prediction files
    pred_files = {}  # (pred_key, date) -> filepath

    if args.skip_gen or args.skip_sim:
        log.info("Loading existing prediction files...")
        for f in PRED_OUT_DIR.glob('*.npz'):
            stem = f.stem
            date = stem[:10]
            # Reconstruct pred_key from filename — we'll match by label
            pred_files[('file', date)] = str(f)
        log.info(f"Found {len(pred_files)} existing prediction files")

        # If skip-gen but not skip-sim, we need to rebuild the mapping
        if not args.skip_sim and pred_files:
            # Rebuild proper mapping from files
            proper_pred_files = {}
            for f in PRED_OUT_DIR.glob('*.npz'):
                stem = f.stem
                date = stem[:10]
                rest = stem[11:]  # combo label
                proper_pred_files[('file:' + rest, date)] = str(f)
            pred_files = proper_pred_files
    else:
        log.info("Loading WF predictions...")
        wf_data = np.load(str(PRED_FILE), allow_pickle=True)
        dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
        log.info(f"WF dates: {len(dates)} ({dates[0]} to {dates[-1]})")

        pred_files = generate_all_predictions(wf_data, dates, config_groups)
        wf_data.close()

    if not pred_files:
        log.error("No prediction files found/generated. Exiting.")
        sys.exit(1)

    # Upload to Saturn if requested
    if args.upload_saturn:
        if upload_to_saturn(pred_files):
            launch_saturn_sweep()

    # Step 2: Run sim sweep locally
    if not args.skip_sim:
        # Build job list: each config maps to its prediction files via pred_key
        # For skip-gen mode, we need a different approach — match by filename
        if args.skip_gen:
            results = run_sweep_from_files(configs, args.workers)
        else:
            results = run_sweep(pred_files, configs, config_groups, workers=args.workers)

        if not results:
            log.error("No sim results. Check binary and MBO files.")
            sys.exit(1)
    else:
        # Aggregate from existing result files
        results = load_existing_results()

    # Step 3: Aggregate and report
    summaries = aggregate_and_report(results)

    log.info(f"\nLog: {RESULTS_DIR / f'stacked_exit_sweep_{_ts}.log'}")
    log.info("DONE.")

    return summaries


def run_sweep_from_files(configs, workers):
    """
    Run sweep when --skip-gen is used.
    Match prediction files by scanning PRED_OUT_DIR and building jobs for each TP/SL combo.
    """
    pred_dir_files = {}
    for f in PRED_OUT_DIR.glob('*.npz'):
        stem = f.stem
        date = stem[:10]
        combo_label = stem[11:]
        pred_dir_files.setdefault(combo_label, {})[date] = str(f)

    log.info(f"Found {len(pred_dir_files)} unique prediction combos in {PRED_OUT_DIR}")

    jobs = []
    for combo_label, date_files in pred_dir_files.items():
        for date, pred_path in date_files.items():
            nodash = date.replace('-', '')
            mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
            if not mbo_file.exists():
                mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
            if not mbo_file.exists():
                continue

            for tp in TP_VALUES:
                for sl in SL_VALUES:
                    tp_str = f'tp{tp}' if tp is not None else 'tpN'
                    sl_str = f'sl{sl}' if sl is not None else 'slN'
                    full_label = f'{combo_label}_{tp_str}_{sl_str}'
                    out_file = SIM_OUT_DIR / f'{full_label}_{date}.json'

                    if out_file.exists():
                        jobs.append(('cached', full_label, date, str(out_file)))
                    else:
                        jobs.append(('run', full_label, date, {
                            'mbo': str(mbo_file),
                            'pred': pred_path,
                            'out': str(out_file),
                            'tp': tp,
                            'sl': sl,
                        }))

    # Load cached
    results = defaultdict(dict)
    cached_count = 0
    for jt, label, date, data in jobs:
        if jt == 'cached':
            try:
                with open(data) as f:
                    results[label][date] = json.load(f)
                cached_count += 1
            except:
                pass

    to_run = [(label, date, params) for jt, label, date, params in jobs if jt == 'run']
    log.info(f"Total jobs: {len(jobs)} ({cached_count} cached, {len(to_run)} to run)")

    if not to_run:
        log.info("All cached.")
        return dict(results)

    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for label, date, params in to_run:
            f = executor.submit(
                run_single_sim,
                params['mbo'], params['pred'], params['out'],
                params['tp'], params['sl']
            )
            futures[f] = (label, date)

        for future in as_completed(futures):
            label, date = futures[future]
            completed += 1
            try:
                res = future.result()
                if res:
                    results[label][date] = res
            except Exception as e:
                log.warning(f"Job error {label}/{date}: {e}")

            if completed % 200 == 0 or completed == len(to_run):
                el = time.time() - t0
                rate = completed / el if el > 0 else 0
                eta = (len(to_run) - completed) / rate / 60 if rate > 0 else 0
                log.info(f"  [{completed}/{len(to_run)}] {rate:.1f} jobs/s, ETA {eta:.1f}min")

    log.info(f"Done: {completed} jobs in {time.time()-t0:.0f}s")
    return dict(results)


def load_existing_results():
    """Load all result JSON files from SIM_OUT_DIR for aggregation."""
    results = defaultdict(dict)
    result_files = list(SIM_OUT_DIR.glob('*.json'))
    log.info(f"Loading {len(result_files)} existing result files from {SIM_OUT_DIR}...")

    loaded = 0
    for f in result_files:
        stem = f.stem
        # Label is everything before the last _YYYY-MM-DD
        # e.g., smooth_smoothExit_conv1.5_ethr0.0_vol50_tp5_sl10_2025-12-01
        parts = stem.rsplit('_', 3)  # Split off date (YYYY-MM-DD = 3 parts with hyphens)
        # Actually dates are like 2025-12-01, so we need to find where the date starts
        # The date is the last 10 chars of stem
        if len(stem) >= 10:
            potential_date = stem[-10:]
            if len(potential_date) == 10 and potential_date[4] == '-' and potential_date[7] == '-':
                date = potential_date
                label = stem[:-11]  # strip _YYYY-MM-DD
                try:
                    with open(f) as fh:
                        results[label][date] = json.load(fh)
                    loaded += 1
                except:
                    pass

    log.info(f"Loaded {loaded} result files into {len(results)} configs")
    return dict(results)


if __name__ == '__main__':
    main()
