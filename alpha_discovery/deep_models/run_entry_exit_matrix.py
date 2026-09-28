#!/usr/bin/env python3
"""
Entry x Exit Matrix Sweep
===========================
Comprehensive sweep testing every entry signal paired with every exit signal.
Each combination encodes entry+exit logic into prediction files using hysteresis:
  - Signal activates (= z-score value) when entry condition fires
  - Signal deactivates (= 0) when exit condition fires while active
  - Re-entry requires entry condition to fire again

ENTRY METHODS:
  1. ema_entry:      EMA z-score > entry_threshold
  2. momentum_entry: z-score acceleration (50-bar diff) > accel_threshold
  3. book_entry:     expanding z-score > entry_threshold AND book imbalance confirms direction
  4. smooth_entry:   rolling mean z-score (50-bar) > entry_threshold

EXIT METHODS:
  1. ema_exit:       EMA z-score drops below exit_threshold
  2. momentum_exit:  z-score acceleration reverses sign
  3. book_exit:      book imbalance flips against position direction
  4. predstd_exit:   pred_std exceeds max_std (model confused)
  5. smooth_exit:    rolling mean z-score drops below exit_threshold
  6. tp_exit:        let Rust sim handle with --take-profit-ticks

All signal processing is CAUSAL (expanding z-score, trailing vol, backward-only rolling).

Usage:
    python alpha_discovery/deep_models/run_entry_exit_matrix.py --workers 24
    python alpha_discovery/deep_models/run_entry_exit_matrix.py --workers 24 --skip-gen
    python alpha_discovery/deep_models/run_entry_exit_matrix.py --skip-sim  # aggregate only
    python alpha_discovery/deep_models/run_entry_exit_matrix.py --upload-jupiter
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
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_matrix_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_matrix_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50
MAX_HOLD_MS = 3600000   # 60 min safety net

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('entry_exit_matrix')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'entry_exit_matrix_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING (all causal, no look-ahead)
# ══════════════════════════════════════════════════════════════════════════════

def compute_trailing_vol(mid, window=3000):
    """Trailing realized vol (std of 1s returns) over window bars. No look-ahead."""
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
    """Expanding percentile threshold for vol gating. No look-ahead."""
    import pandas as pd
    s = pd.Series(vol)
    return s.expanding(min_periods=100).quantile(pct / 100.0).values


def zscore_expanding(arr):
    """Expanding-window z-score (no look-ahead). Requires >=50 non-NaN samples."""
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
    """EMA-based z-score. More responsive to recent data than expanding."""
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
    """
    Load book tensors, compute L1-L5 depth imbalance.
    book shape: (n_bars, 20, 4) where 0-9=bid, 10-19=ask, features=[price_rel, depth, num_orders, queue_age]
    imbalance = (bid_depth_L1_to_L5 - ask_depth_L1_to_L5) / (bid + ask + eps)
    """
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
    """Skip first 30 min and last 15 min of RTH session (6.5 hrs = 390 min)."""
    secs = np.arange(n_bars) / BARS_PER_SEC
    mins = secs / 60.0
    return (mins >= 30) & (mins < 375)


# ══════════════════════════════════════════════════════════════════════════════
# HYSTERESIS ENGINE: entry fires -> signal active -> exit fires -> signal zero
# ══════════════════════════════════════════════════════════════════════════════

def _apply_hysteresis_python(z_score, entry_cond_long, entry_cond_short,
                              exit_cond_long, exit_cond_short):
    """
    Hysteresis entry/exit logic (pure Python).

    entry_cond_long[i]:  True when entry condition for long fires at bar i
    entry_cond_short[i]: True when entry condition for short fires at bar i
    exit_cond_long[i]:   True when exit condition for long fires at bar i
    exit_cond_short[i]:  True when exit condition for short fires at bar i

    Output: signal = z_score when active, 0 when not.
    """
    n = len(z_score)
    output = np.zeros(n, dtype=np.float64)
    in_long = False
    in_short = False

    for i in range(n):
        if in_long:
            if exit_cond_long[i]:
                in_long = False
            else:
                output[i] = z_score[i]
        elif in_short:
            if exit_cond_short[i]:
                in_short = False
            else:
                output[i] = z_score[i]

        if not in_long and not in_short:
            if entry_cond_long[i]:
                in_long = True
                output[i] = z_score[i]
            elif entry_cond_short[i]:
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


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY CONDITIONS — return (entry_long, entry_short) boolean arrays
# ══════════════════════════════════════════════════════════════════════════════

def entry_ema(ema_z, threshold):
    """EMA z-score exceeds entry_threshold."""
    return (ema_z > threshold), (ema_z < -threshold)


def entry_momentum(momentum, threshold):
    """Z-score acceleration exceeds accel_threshold."""
    return (momentum > threshold), (momentum < -threshold)


def entry_book(z_score, imbalance, threshold):
    """Expanding z-score exceeds threshold AND book imbalance confirms direction."""
    if imbalance is None:
        return np.zeros(len(z_score), dtype=bool), np.zeros(len(z_score), dtype=bool)
    entry_long = (z_score > threshold) & (imbalance > 0)
    entry_short = (z_score < -threshold) & (imbalance < 0)
    return entry_long, entry_short


def entry_smooth(smooth_z, threshold):
    """Rolling mean z-score exceeds entry_threshold."""
    return (smooth_z > threshold), (smooth_z < -threshold)


# ══════════════════════════════════════════════════════════════════════════════
# EXIT CONDITIONS — return (exit_long, exit_short) boolean arrays
# exit_long[i] = True means "exit long at bar i"
# exit_short[i] = True means "exit short at bar i"
# ══════════════════════════════════════════════════════════════════════════════

def exit_ema(ema_z, threshold):
    """EMA z-score drops below exit_threshold (conviction decaying)."""
    # Long exits when EMA z drops below threshold (was positive, now fading)
    exit_long = ema_z < threshold
    # Short exits when EMA z rises above -threshold (was negative, now fading)
    exit_short = ema_z > -threshold
    return exit_long, exit_short


def exit_momentum(momentum):
    """Z-score acceleration reverses sign (model decelerating)."""
    # Long: exit when momentum turns negative
    exit_long = momentum < 0
    # Short: exit when momentum turns positive
    exit_short = momentum > 0
    return exit_long, exit_short


def exit_book(imbalance):
    """Book imbalance flips against position direction."""
    if imbalance is None:
        n = 1  # will be broadcast
        return np.ones(n, dtype=bool), np.ones(n, dtype=bool)
    # Long: exit when more asks than bids (imbalance < 0)
    exit_long = imbalance < 0
    # Short: exit when more bids than asks (imbalance > 0)
    exit_short = imbalance > 0
    return exit_long, exit_short


def exit_predstd(pred_std, max_std):
    """Pred_std exceeds max_std (model confused)."""
    confused = pred_std > max_std
    # Confused = exit regardless of direction
    return confused, confused


def exit_smooth(smooth_z, threshold):
    """Rolling mean z-score drops below exit_threshold."""
    exit_long = smooth_z < threshold
    exit_short = smooth_z > -threshold
    return exit_long, exit_short


# ══════════════════════════════════════════════════════════════════════════════
# MATRIX COMBO DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def build_combo_list():
    """
    Build the selected entry x exit combos (not full cartesian).
    Returns list of dicts with entry/exit config.
    """
    combos = []

    # ── EMA entry × various exits ──
    for conv in [2.0, 2.5]:
        # × ema_exit
        for exit_thr in [0.0, 0.5, 1.0]:
            for vg in [50, 70]:
                combos.append({
                    'entry': 'ema', 'entry_params': {'threshold': conv},
                    'exit': 'ema_exit', 'exit_params': {'threshold': exit_thr},
                    'vol_gate': vg,
                    'label': f'ema{conv}_X_emaExit{exit_thr}_vol{vg}',
                })
        # × book_exit
        for vg in [50, 70]:
            combos.append({
                'entry': 'ema', 'entry_params': {'threshold': conv},
                'exit': 'book_exit', 'exit_params': {},
                'vol_gate': vg,
                'label': f'ema{conv}_X_bookExit_vol{vg}',
            })
        # × momentum_exit
        for vg in [50, 70]:
            combos.append({
                'entry': 'ema', 'entry_params': {'threshold': conv},
                'exit': 'momentum_exit', 'exit_params': {},
                'vol_gate': vg,
                'label': f'ema{conv}_X_momExit_vol{vg}',
            })
        # × predstd_exit
        for max_std in [0.10, 0.15]:
            for vg in [50, 70]:
                combos.append({
                    'entry': 'ema', 'entry_params': {'threshold': conv},
                    'exit': 'predstd_exit', 'exit_params': {'max_std': max_std},
                    'vol_gate': vg,
                    'label': f'ema{conv}_X_predstd{max_std}_vol{vg}',
                })

    # ── Momentum entry × various exits ──
    for accel_thr in [0.3, 0.5]:
        # × ema_exit
        for exit_thr in [0.0, 0.5]:
            for vg in [50, 70]:
                combos.append({
                    'entry': 'momentum', 'entry_params': {'threshold': accel_thr},
                    'exit': 'ema_exit', 'exit_params': {'threshold': exit_thr},
                    'vol_gate': vg,
                    'label': f'mom{accel_thr}_X_emaExit{exit_thr}_vol{vg}',
                })
        # × book_exit
        for vg in [50, 70]:
            combos.append({
                'entry': 'momentum', 'entry_params': {'threshold': accel_thr},
                'exit': 'book_exit', 'exit_params': {},
                'vol_gate': vg,
                'label': f'mom{accel_thr}_X_bookExit_vol{vg}',
            })
        # × smooth_exit
        for exit_thr in [0.0, 0.5]:
            for vg in [50, 70]:
                combos.append({
                    'entry': 'momentum', 'entry_params': {'threshold': accel_thr},
                    'exit': 'smooth_exit', 'exit_params': {'threshold': exit_thr},
                    'vol_gate': vg,
                    'label': f'mom{accel_thr}_X_smoothExit{exit_thr}_vol{vg}',
                })

    # ── Book entry × various exits ──
    for conv in [2.0, 2.5]:
        # × ema_exit
        for exit_thr in [0.0, 0.5]:
            for vg in [50, 70]:
                combos.append({
                    'entry': 'book', 'entry_params': {'threshold': conv},
                    'exit': 'ema_exit', 'exit_params': {'threshold': exit_thr},
                    'vol_gate': vg,
                    'label': f'book{conv}_X_emaExit{exit_thr}_vol{vg}',
                })
        # × momentum_exit
        for vg in [50, 70]:
            combos.append({
                'entry': 'book', 'entry_params': {'threshold': conv},
                'exit': 'momentum_exit', 'exit_params': {},
                'vol_gate': vg,
                'label': f'book{conv}_X_momExit_vol{vg}',
            })
        # × predstd_exit
        for vg in [50, 70]:
            combos.append({
                'entry': 'book', 'entry_params': {'threshold': conv},
                'exit': 'predstd_exit', 'exit_params': {'max_std': 0.10},
                'vol_gate': vg,
                'label': f'book{conv}_X_predstd0.1_vol{vg}',
            })

    # ── Smooth entry × various exits ──
    for conv in [1.5, 2.0]:
        # × smooth_exit
        for exit_thr in [0.0, 0.5]:
            for vg in [50, 70]:
                combos.append({
                    'entry': 'smooth', 'entry_params': {'threshold': conv},
                    'exit': 'smooth_exit', 'exit_params': {'threshold': exit_thr},
                    'vol_gate': vg,
                    'label': f'smooth{conv}_X_smoothExit{exit_thr}_vol{vg}',
                })
        # × book_exit
        for vg in [50, 70]:
            combos.append({
                'entry': 'smooth', 'entry_params': {'threshold': conv},
                'exit': 'book_exit', 'exit_params': {},
                'vol_gate': vg,
                'label': f'smooth{conv}_X_bookExit_vol{vg}',
            })

    log.info(f"Built {len(combos)} entry x exit combos")
    return combos


# Sim variants: (tp_ticks, label_suffix)
SIM_VARIANTS = [
    (None,  'hold'),      # hold-only (60min safety net, exit encoded in signal)
    (5,     'tp5'),       # + take-profit 5 ticks
    (8,     'tp8'),       # + take-profit 8 ticks
]


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION FILE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_predictions_for_date(date, wf_data, combos):
    """
    Generate all entry x exit prediction files for one date.
    Returns dict of (date, combo_label) -> filepath.
    """
    preds_raw = wf_data[f'{date}_preds'].astype(np.float64)
    mid = wf_data[f'{date}_mid'].astype(np.float64)
    n_bars = len(mid)
    if n_bars < 5000:
        return {}

    # Check MBO exists
    nodash = date.replace('-', '')
    mbo_zst = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
    mbo_dbn = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
    if not mbo_zst.exists() and not mbo_dbn.exists():
        return {}

    # ── Pre-compute all derived signals (once per date) ──

    # 1. Align CNN offset
    aligned = np.zeros(n_bars, dtype=np.float64)
    end = min(n_bars, len(preds_raw) + CNN_OFFSET)
    aligned[CNN_OFFSET:end] = preds_raw[:end - CNN_OFFSET]

    # 2. Expanding z-score (base signal)
    z_score = zscore_expanding(aligned)

    # 3. EMA z-score (span 5000)
    ema_z = zscore_ema(aligned, span=5000)

    # 4. Momentum (50-bar diff of expanding z-score)
    momentum = compute_momentum(z_score, diff_bars=50)

    # 5. Smoothed z-score (50-bar rolling mean)
    smooth_z = rolling_mean_smooth(z_score, window=50)

    # 6. Pred_std (rolling std of raw predictions, window 3000)
    pred_std = compute_pred_std(aligned, window=3000)

    # 7. Book imbalance
    imbalance = load_book_imbalance(date, n_bars)

    # 8. Vol
    vol = compute_trailing_vol(mid)
    vol_pct_50 = compute_expanding_vol_percentile(vol, 50)
    vol_pct_70 = compute_expanding_vol_percentile(vol, 70)
    vol_thresholds = {50: vol_pct_50, 70: vol_pct_70}

    # 9. Time mask
    tmask = time_mask(n_bars)

    saved = {}

    for combo in combos:
        entry_type = combo['entry']
        exit_type = combo['exit']
        entry_params = combo['entry_params']
        exit_params = combo['exit_params']
        vg = combo['vol_gate']
        label = combo['label']

        # ── Compute entry conditions ──
        if entry_type == 'ema':
            entry_long, entry_short = entry_ema(ema_z, entry_params['threshold'])
        elif entry_type == 'momentum':
            entry_long, entry_short = entry_momentum(momentum, entry_params['threshold'])
        elif entry_type == 'book':
            entry_long, entry_short = entry_book(z_score, imbalance, entry_params['threshold'])
        elif entry_type == 'smooth':
            entry_long, entry_short = entry_smooth(smooth_z, entry_params['threshold'])
        else:
            continue

        # ── Compute exit conditions ──
        if exit_type == 'ema_exit':
            exit_long, exit_short = exit_ema(ema_z, exit_params['threshold'])
        elif exit_type == 'momentum_exit':
            exit_long, exit_short = exit_momentum(momentum)
        elif exit_type == 'book_exit':
            el, es = exit_book(imbalance)
            # If no book data, skip this combo for this date
            if imbalance is None:
                continue
            exit_long, exit_short = el, es
        elif exit_type == 'predstd_exit':
            exit_long, exit_short = exit_predstd(pred_std, exit_params['max_std'])
        elif exit_type == 'smooth_exit':
            exit_long, exit_short = exit_smooth(smooth_z, exit_params['threshold'])
        elif exit_type == 'tp_exit':
            # No exit logic in signal — let Rust sim handle it
            exit_long = np.zeros(n_bars, dtype=bool)
            exit_short = np.zeros(n_bars, dtype=bool)
        else:
            continue

        # Ensure arrays are correct length
        if len(exit_long) != n_bars:
            exit_long = np.broadcast_to(exit_long, n_bars).copy()
        if len(exit_short) != n_bars:
            exit_short = np.broadcast_to(exit_short, n_bars).copy()

        # ── Apply hysteresis ──
        signal = apply_hysteresis(
            z_score,
            entry_long.astype(np.bool_),
            entry_short.astype(np.bool_),
            exit_long.astype(np.bool_),
            exit_short.astype(np.bool_),
        )

        # ── Apply vol gate ──
        if vg > 0 and vg in vol_thresholds:
            vt = vol_thresholds[vg]
            vol_mask = np.isnan(vol) | (vol < vt)
            signal[vol_mask] = 0.0

        # ── Apply time mask ──
        signal[~tmask] = 0.0

        signal = np.nan_to_num(signal, nan=0.0).astype(np.float32)

        # Save
        fpath = PRED_OUT_DIR / f'{date}_{label}.npz'
        np.savez_compressed(str(fpath), predictions=signal)
        saved[(date, label)] = str(fpath)

    # Cleanup
    del mid, preds_raw, aligned, z_score, ema_z, momentum, smooth_z, pred_std, vol
    if imbalance is not None:
        del imbalance
    gc.collect()

    return saved


def generate_all_predictions(wf_data, dates, combos):
    """Generate prediction files for all dates and combos."""
    all_saved = {}
    for di, date in enumerate(dates):
        day_saved = generate_predictions_for_date(date, wf_data, combos)
        all_saved.update(day_saved)
        if (di + 1) % 5 == 0 or di == 0 or di == len(dates) - 1:
            log.info(f"  Generated {di+1}/{len(dates)} days ({len(all_saved)} files total)")
    log.info(f"Total prediction files: {len(all_saved)}")
    return all_saved


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_single_sim(mbo_file, pred_file, output_file, tp_ticks):
    """Run one Rust fill_sim job. Signal threshold = 0.1 (pre-filtered)."""
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
        '--quiet',
    ]
    if tp_ticks is not None:
        cmd += ['--take-profit-ticks', str(tp_ticks)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        pass
    return None


def run_sweep(saved_files, workers=24):
    """Run all sim jobs (3 variants per combo) in parallel."""
    jobs = []  # (mbo, pred, out, tp, combo_label, sim_label, date)

    for (date, combo_label), pred_file in saved_files.items():
        nodash = date.replace('-', '')
        mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        if not mbo_file.exists():
            mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
        if not mbo_file.exists():
            continue

        for tp_ticks, sim_suffix in SIM_VARIANTS:
            full_label = f'{combo_label}_{sim_suffix}'
            out_file = SIM_OUT_DIR / f'{full_label}_{date}.json'
            if out_file.exists():
                # Load cached
                try:
                    with open(out_file) as f:
                        existing = json.load(f)
                    jobs.append(('cached', full_label, date, existing))
                    continue
                except:
                    pass
            jobs.append(('run', full_label, date, {
                'mbo': str(mbo_file), 'pred': pred_file,
                'out': str(out_file), 'tp': tp_ticks,
            }))

    cached = [(l, d, r) for t, l, d, r in jobs if t == 'cached']
    to_run = [(l, d, p) for t, l, d, p in jobs if t == 'run']

    log.info(f"Total sim jobs: {len(jobs)} ({len(cached)} cached, {len(to_run)} to run)")
    log.info(f"Workers: {workers}")

    results = defaultdict(dict)
    for label, date, res in cached:
        results[label][date] = res

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
                params['mbo'], params['pred'], params['out'], params['tp']
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

            if completed % 100 == 0 or completed == len(to_run):
                el = time.time() - t0
                rate = completed / el if el > 0 else 0
                eta = (len(to_run) - completed) / rate / 60 if rate > 0 else 0
                log.info(f"  [{completed}/{len(to_run)}] {rate:.1f} jobs/s, ETA {eta:.1f}min")

    log.info(f"Simulation done: {completed} jobs in {time.time()-t0:.0f}s")
    return dict(results)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION & MATRIX REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def parse_combo_label(config_label):
    """
    Parse combo label like 'ema2.0_X_emaExit0.5_vol50_hold' into components.
    Returns (entry_method, exit_method, vol_gate, sim_variant).
    """
    # Last part is sim variant (hold, tp5, tp8)
    parts = config_label.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in ('hold', 'tp5', 'tp8'):
        base_label = parts[0]
        sim_var = parts[1]
    else:
        base_label = config_label
        sim_var = 'hold'

    # Split on _X_ to get entry and exit+vol
    x_parts = base_label.split('_X_')
    if len(x_parts) == 2:
        entry_part = x_parts[0]
        rest = x_parts[1]
        # Extract vol gate (last _volNN)
        vol_idx = rest.rfind('_vol')
        if vol_idx >= 0:
            exit_part = rest[:vol_idx]
            vol_part = rest[vol_idx+1:]
        else:
            exit_part = rest
            vol_part = 'vol0'
    else:
        entry_part = base_label
        exit_part = 'unknown'
        vol_part = 'vol0'

    return entry_part, exit_part, vol_part, sim_var


def aggregate_and_report(results):
    """Aggregate per-day results, print entry x exit matrix."""
    summaries = []

    for config_label, date_results in results.items():
        if not date_results:
            continue

        daily_pnls = []
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0

        for date_str, res in sorted(date_results.items()):
            day_pnl = res.get('total_pnl_dollars', 0)
            daily_pnls.append(day_pnl)
            total_trades += res.get('total_trades', 0)
            total_signals += res.get('total_signals', 0)
            total_filled += res.get('total_filled', 0)
            if 'trades' in res:
                for trade in res['trades']:
                    if trade.get('pnl_dollars', 0) > 0:
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

        entry_part, exit_part, vol_part, sim_var = parse_combo_label(config_label)

        summaries.append({
            'config': config_label,
            'entry': entry_part,
            'exit': exit_part,
            'vol': vol_part,
            'sim_var': sim_var,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe': round(sharpe, 3),
            'avg_daily_pnl': round(avg_daily, 2),
            'max_dd': round(max_dd, 2),
            'annualized': round(avg_daily * 252, 0),
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)

    # ── Top 40 by Sharpe ──
    log.info("\n" + "=" * 140)
    log.info("ENTRY x EXIT MATRIX SWEEP — TOP 40 BY SHARPE (Rust MBO Fill Sim)")
    log.info("=" * 140)
    log.info(f"{'#':>3} {'Entry':<18} {'Exit':<20} {'Vol':<5} {'Sim':<5} "
             f"{'Sharpe':>7} {'P&L':>10} {'Trades':>6} {'Fill%':>6} "
             f"{'WR%':>5} {'MaxDD':>8} {'Annual':>10}")
    log.info("-" * 140)

    for i, s in enumerate(summaries[:40]):
        log.info(
            f"{i+1:>3} {s['entry']:<18} {s['exit']:<20} {s['vol']:<5} {s['sim_var']:<5} "
            f"{s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} {s['n_trades']:>6} "
            f"{s['fill_rate']*100:>5.1f}% {s['win_rate']*100:>4.1f}% "
            f"${s['max_dd']:>7,.0f} ${s['annualized']:>9,.0f}"
        )

    # ── MATRIX VIEW: best Sharpe per (entry, exit) pair ──
    log.info("\n" + "=" * 120)
    log.info("MATRIX VIEW — Best Sharpe per Entry x Exit (across vol gates, best sim variant)")
    log.info("=" * 120)

    # Collect unique entries and exits
    entry_set = sorted(set(s['entry'] for s in summaries))
    exit_set = sorted(set(s['exit'] for s in summaries))

    # Build matrix: best sharpe for each (entry, exit)
    matrix = {}
    matrix_pnl = {}
    matrix_trades = {}
    matrix_wr = {}
    matrix_sim = {}
    for s in summaries:
        key = (s['entry'], s['exit'])
        if key not in matrix or s['sharpe'] > matrix[key]:
            matrix[key] = s['sharpe']
            matrix_pnl[key] = s['total_pnl']
            matrix_trades[key] = s['n_trades']
            matrix_wr[key] = s['win_rate']
            matrix_sim[key] = f"{s['vol']}_{s['sim_var']}"

    # Print matrix header
    col_width = 14
    entry_exit_label = "Entry \\ Exit"
    header = f"{entry_exit_label:<18}"
    for ex in exit_set:
        header += f" {ex:>{col_width}}"
    log.info(header)
    log.info("-" * (18 + (col_width + 1) * len(exit_set)))

    for en in entry_set:
        row = f"{en:<18}"
        for ex in exit_set:
            key = (en, ex)
            if key in matrix:
                val = f"{matrix[key]:+.2f}"
            else:
                val = "---"
            row += f" {val:>{col_width}}"
        log.info(row)

    # P&L matrix
    log.info("\nP&L Matrix (total $ over OOT period):")
    header = f"{entry_exit_label:<18}"
    for ex in exit_set:
        header += f" {ex:>{col_width}}"
    log.info(header)
    log.info("-" * (18 + (col_width + 1) * len(exit_set)))

    for en in entry_set:
        row = f"{en:<18}"
        for ex in exit_set:
            key = (en, ex)
            if key in matrix_pnl:
                val = f"${matrix_pnl[key]:,.0f}"
            else:
                val = "---"
            row += f" {val:>{col_width}}"
        log.info(row)

    # Trades matrix
    log.info("\nTrades Matrix:")
    header = f"{entry_exit_label:<18}"
    for ex in exit_set:
        header += f" {ex:>{col_width}}"
    log.info(header)
    log.info("-" * (18 + (col_width + 1) * len(exit_set)))

    for en in entry_set:
        row = f"{en:<18}"
        for ex in exit_set:
            key = (en, ex)
            if key in matrix_trades:
                val = f"{matrix_trades[key]}"
            else:
                val = "—"
            row += f" {val:>{col_width}}"
        log.info(row)

    # ── Best per entry method ──
    log.info("\n" + "=" * 100)
    log.info("BEST CONFIG PER ENTRY METHOD")
    log.info("=" * 100)
    for en in entry_set:
        en_results = [s for s in summaries if s['entry'] == en]
        if en_results:
            best = en_results[0]
            log.info(f"\n  {en}:")
            log.info(f"    Best exit: {best['exit']} ({best['vol']}, {best['sim_var']})")
            log.info(f"    Sharpe: {best['sharpe']:.2f}, P&L: ${best['total_pnl']:,.2f}, "
                     f"Trades: {best['n_trades']}, Fill: {best['fill_rate']:.1%}, "
                     f"WR: {best['win_rate']:.1%}, Annual: ${best['annualized']:,.0f}")

    # ── Best per exit method ──
    log.info("\n" + "=" * 100)
    log.info("BEST CONFIG PER EXIT METHOD")
    log.info("=" * 100)
    for ex in exit_set:
        ex_results = [s for s in summaries if s['exit'] == ex]
        if ex_results:
            best = ex_results[0]
            log.info(f"\n  {ex}:")
            log.info(f"    Best entry: {best['entry']} ({best['vol']}, {best['sim_var']})")
            log.info(f"    Sharpe: {best['sharpe']:.2f}, P&L: ${best['total_pnl']:,.2f}, "
                     f"Trades: {best['n_trades']}, Fill: {best['fill_rate']:.1%}, "
                     f"WR: {best['win_rate']:.1%}, Annual: ${best['annualized']:,.0f}")

    # ── Sim variant comparison ──
    log.info("\n" + "=" * 100)
    log.info("SIM VARIANT COMPARISON (hold vs TP5 vs TP8)")
    log.info("=" * 100)
    for sv in ['hold', 'tp5', 'tp8']:
        sv_results = [s for s in summaries if s['sim_var'] == sv]
        if sv_results:
            best = sv_results[0]
            avg_sharpe = np.mean([s['sharpe'] for s in sv_results])
            pct_positive = np.mean([s['total_pnl'] > 0 for s in sv_results]) * 100
            log.info(f"\n  {sv}: {len(sv_results)} configs, avg Sharpe {avg_sharpe:.2f}, "
                     f"{pct_positive:.0f}% profitable")
            log.info(f"    Best: {best['config']} -> Sharpe {best['sharpe']:.2f}, "
                     f"${best['total_pnl']:,.0f}")

    # ── Reference ──
    log.info("\n" + "=" * 100)
    log.info("REFERENCE — Baseline:")
    log.info("  IS best (vol70/conv2.5/1t/3r/30min): Sharpe 3.28, +$15,479/74d, 130 trades")
    log.info("  OOT static:                          Sharpe 1.58, +$4,082/68d, 102 trades")
    log.info("=" * 100)

    # Save JSON
    out_file = RESULTS_DIR / f'entry_exit_matrix_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'n_configs': len(summaries),
            'matrix_sharpe': {f'{k[0]}|{k[1]}': v for k, v in matrix.items()},
            'matrix_pnl': {f'{k[0]}|{k[1]}': v for k, v in matrix_pnl.items()},
            'summaries': summaries,
        }, f, indent=2)
    log.info(f"\nResults saved: {out_file}")

    return summaries


# ══════════════════════════════════════════════════════════════════════════════
# JUPITER UPLOAD & REMOTE SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_jupiter(saved_files):
    """Upload prediction NPZs to Jupiter for parallel sweep."""
    import paramiko

    JUPITER_HOST = 'jupiter'
    JUPITER_USER = 'jupiter'
    JUPITER_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
    REMOTE_DIR = '/home/jupiter/Lvl3Quant/data/processed/cnn_wf_matrix_predictions'

    log.info(f"Uploading {len(saved_files)} prediction files to Jupiter...")

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(JUPITER_HOST, username=JUPITER_USER, password=JUPITER_PW, timeout=10)
        ssh.exec_command(f'mkdir -p {REMOTE_DIR}')
        time.sleep(1)

        sftp = ssh.open_sftp()
        uploaded = 0
        for (date, label), local_path in saved_files.items():
            remote_path = f'{REMOTE_DIR}/{Path(local_path).name}'
            try:
                sftp.put(local_path, remote_path)
                uploaded += 1
                if uploaded % 100 == 0:
                    log.info(f"  Uploaded {uploaded}/{len(saved_files)}")
            except Exception as e:
                log.warning(f"  Upload failed {Path(local_path).name}: {e}")

        sftp.close()
        ssh.close()
        log.info(f"Uploaded {uploaded} files to Jupiter:{REMOTE_DIR}")
        return True
    except Exception as e:
        log.error(f"Jupiter upload failed: {e}")
        return False


def launch_jupiter_sweep():
    """Launch sweep on Jupiter with 14 workers."""
    import paramiko

    JUPITER_HOST = 'jupiter'
    JUPITER_USER = 'jupiter'
    JUPITER_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")

    remote_script = '''#!/usr/bin/env python3
"""Remote matrix sweep on Jupiter."""
import sys, json, time, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_matrix_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_matrix_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_VARIANTS = [
    (None,  "hold"),
    (5,     "tp5"),
    (8,     "tp8"),
]

def run_sim(mbo, pred, out, tp):
    cmd = [str(BINARY), "--mbo-file", str(mbo), "--predictions", str(pred),
           "--output", str(out), "--hold-ms", "3600000", "--signal-threshold", "0.1",
           "--latency-ms", "0", "--quiet", "--chase-entry",
           "--chase-max-ticks", "1", "--chase-max-reprices", "3"]
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
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
    for tp, sim_suffix in SIM_VARIANTS:
        full_label = f"{combo_label}_{sim_suffix}"
        out_file = OUT_DIR / f"{full_label}_{date}.json"
        if out_file.exists():
            continue
        jobs.append((str(mbo), str(pf), str(out_file), tp))

print(f"Jobs to run: {len(jobs)}")
completed = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=14) as executor:
    futures = {executor.submit(run_sim, *j): j for j in jobs}
    for f in as_completed(futures):
        completed += 1
        if completed % 50 == 0:
            el = time.time() - t0
            rate = completed / el if el > 0 else 0
            print(f"  [{completed}/{len(jobs)}] {rate:.1f}/s")

print(f"Done: {completed} jobs in {time.time()-t0:.0f}s")
'''

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(JUPITER_HOST, username=JUPITER_USER, password=JUPITER_PW, timeout=10)

        sftp = ssh.open_sftp()
        remote_script_path = '/home/jupiter/Lvl3Quant/run_jupiter_matrix_sweep.py'
        with sftp.open(remote_script_path, 'w') as f:
            f.write(remote_script)
        sftp.close()

        cmd = f'cd /home/jupiter/Lvl3Quant && nohup python3 {remote_script_path} > matrix_sweep_jupiter.log 2>&1 &'
        ssh.exec_command(cmd)
        time.sleep(2)
        log.info(f"Launched Jupiter sweep: {cmd}")
        log.info(f"Monitor: ssh jupiter@jupiter 'tail -f /home/jupiter/Lvl3Quant/matrix_sweep_jupiter.log'")
        ssh.close()
        return True
    except Exception as e:
        log.error(f"Jupiter launch failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Entry x Exit Matrix Sweep')
    parser.add_argument('--workers', type=int, default=24,
                        help='Parallel sim workers (default: 24)')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip prediction generation, use existing files')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, only aggregate existing results')
    parser.add_argument('--upload-jupiter', action='store_true',
                        help='Upload predictions to Jupiter and launch remote sweep')
    args = parser.parse_args()

    combos = build_combo_list()

    log.info("=" * 90)
    log.info("ENTRY x EXIT MATRIX SWEEP")
    log.info(f"  Combos:      {len(combos)}")
    log.info(f"  Sim variants: {len(SIM_VARIANTS)} (hold, tp5, tp8)")
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  Binary:      {BINARY}")
    log.info(f"  Predictions: {PRED_FILE}")
    log.info(f"  MBO:         {MBO_DIR}")
    log.info(f"  Book:        {BOOK_DIR}")
    log.info(f"  Pred out:    {PRED_OUT_DIR}")
    log.info(f"  Sim out:     {SIM_OUT_DIR}")
    log.info("=" * 90)

    if not BINARY.exists():
        log.error(f"Binary not found: {BINARY}")
        sys.exit(1)
    if not PRED_FILE.exists():
        log.error(f"WF predictions not found: {PRED_FILE}")
        sys.exit(1)

    # Step 1: Generate prediction files
    if args.skip_gen or args.skip_sim:
        log.info("Loading existing prediction files...")
        saved = {}
        for f in PRED_OUT_DIR.glob('*.npz'):
            stem = f.stem
            date = stem[:10]
            label = stem[11:]
            saved[(date, label)] = str(f)
        log.info(f"Found {len(saved)} existing prediction files")
    else:
        log.info("Loading WF predictions...")
        wf_data = np.load(str(PRED_FILE), allow_pickle=True)
        dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
        log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
        log.info(f"Expected prediction files: ~{len(combos) * len(dates)}")

        saved = generate_all_predictions(wf_data, dates, combos)
        wf_data.close()

    if not saved:
        log.error("No prediction files. Exiting.")
        sys.exit(1)

    # Step 1b: Upload to Jupiter if requested
    if args.upload_jupiter:
        upload_to_jupiter(saved)
        launch_jupiter_sweep()

    # Step 2: Run local sim sweep
    if not args.skip_sim:
        results = run_sweep(saved, workers=args.workers)
    else:
        log.info("Loading existing sim results...")
        results = defaultdict(dict)
        for f in SIM_OUT_DIR.glob('*.json'):
            try:
                with open(f) as fh:
                    res = json.load(fh)
                stem = f.stem
                # Parse: {combo_label}_{sim_suffix}_{date}.json
                # Last 10 chars = date
                date = stem[-10:]
                label = stem[:-11]
                results[label][date] = res
            except:
                continue
        results = dict(results)
        log.info(f"Loaded {sum(len(v) for v in results.values())} results "
                 f"across {len(results)} configs")

    if not results:
        log.error("No results. Exiting.")
        sys.exit(1)

    # Step 3: Aggregate and report
    summaries = aggregate_and_report(results)

    log.info(f"\nLog: {RESULTS_DIR / f'entry_exit_matrix_{_ts}.log'}")
    log.info("DONE.")
    return summaries


if __name__ == '__main__':
    main()
