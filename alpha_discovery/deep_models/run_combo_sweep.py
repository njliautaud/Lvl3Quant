#!/usr/bin/env python3
"""
Combination Strategy Sweep — Stacking Best Approaches
======================================================
Tests 5 combo strategies that layer EMA z-score, book imbalance,
pred_std uncertainty, momentum, and take-profit exits.

COMBO 1: EMA span5000 + Book Confirmation
COMBO 2: EMA + Book + TP5
COMBO 3: Momentum on EMA signal + TP
COMBO 4: EMA + pred_std filter
COMBO 5: Triple filter (EMA + book + pred_std)

All signals use causal/backward-only processing — NO leakage.
Pipeline: raw preds → CNN_OFFSET align → expanding z-score → EMA(span=5000)
  → then apply combo-specific filters.

Sim: hold 30min, chase 1t/3r. Each combo also tested with TP5 where not
already specified.

Usage:
    python alpha_discovery/deep_models/run_combo_sweep.py --workers 24
    python alpha_discovery/deep_models/run_combo_sweep.py --workers 24 --combos 1,2,3,4,5
    python alpha_discovery/deep_models/run_combo_sweep.py --skip-gen --skip-sim  # aggregate only
"""

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

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
BOOK_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_combo_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_combo_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('combo_sweep')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'combo_sweep_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ══════════════════════════════════════════════════════════════════════════════
# SHARED SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def compute_trailing_vol(mid, window=3000):
    """Trailing realized vol (std of 1s returns) over window bars. No look-ahead."""
    ret_1s = np.zeros(len(mid))
    ret_1s[10:] = (mid[10:] - mid[:-10]) / mid[:-10] * 10000
    vol = np.full(len(mid), np.nan)
    cs = np.cumsum(ret_1s)
    cs2 = np.cumsum(ret_1s ** 2)
    for i in range(window, len(mid)):
        s = cs[i] - cs[i - window]
        s2 = cs2[i] - cs2[i - window]
        m = s / window
        vol[i] = np.sqrt(max(s2 / window - m * m, 0))
    return vol


def precompute_vol_percentiles(vol, pcts):
    """Expanding-window vol percentile thresholds (no look-ahead)."""
    n = len(vol)
    result = {p: np.full(n, -np.inf) for p in pcts}
    sv = []
    for i in range(n):
        if not np.isnan(vol[i]):
            bisect.insort(sv, vol[i])
        if len(sv) >= 100:
            for p in pcts:
                result[p][i] = sv[min(int(len(sv) * p / 100), len(sv) - 1)]
    return result


def zscore_expanding(arr):
    """Expanding-window z-score (no look-ahead). Requires >=50 non-NaN samples."""
    result = np.full_like(arr, np.nan, dtype=np.float64)
    rs, rsq, c = 0.0, 0.0, 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v):
            continue
        rs += v
        rsq += v * v
        c += 1
        if c >= 50:
            m = rs / c
            result[i] = (v - m) / max(np.sqrt(rsq / c - m * m), 1e-8)
    return result


def zscore_ema(arr, span=5000):
    """EMA-based z-score — more responsive to recent data than expanding.
    Causal: only uses past data. No look-ahead."""
    result = np.full_like(arr, np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1)
    ema_mean = 0.0
    ema_var = 0.0
    c = 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v):
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


def time_mask_morning_afternoon(n_bars):
    """Time-of-day mask: first 2hrs + last 1.5hrs of RTH."""
    secs = np.arange(n_bars) / BARS_PER_SEC
    mins = secs / 60.0
    return (mins < 120) | ((mins >= 240) & (mins < 330))


def apply_vol_gate(signal, vol, vol_thresholds, vg):
    """Zero out signal bars where vol < expanding Nth percentile."""
    sig = signal.copy()
    if vg > 0 and vg in vol_thresholds:
        thresh = vol_thresholds[vg]
        for i in range(len(sig)):
            if np.isnan(vol[i]) or vol[i] < thresh[i]:
                sig[i] = 0.0
    return sig


def load_book_imbalance(date, n_bars):
    """
    Load book tensors, compute L1-L5 depth imbalance.
    book shape: (n_bars, 20, 4) where 0-9=bid, 10-19=ask,
    features=[price_rel, depth, num_orders, queue_age]
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

    # L1-L5 bid depth (levels 0-4, feature 1=depth)
    bid_depth = book[:, :5, 1].sum(axis=1)
    # L1-L5 ask depth (levels 10-14, feature 1=depth)
    ask_depth = book[:, 10:15, 1].sum(axis=1)

    imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)

    del bt, book
    return imbalance


def compute_pred_std(preds_aligned, n_bars, window=3000):
    """Rolling std of raw (aligned) predictions. No look-ahead."""
    pred_std = np.full(n_bars, np.nan, dtype=np.float64)
    clean = np.nan_to_num(preds_aligned, nan=0.0)
    cs = np.cumsum(clean)
    cs2 = np.cumsum(clean ** 2)
    for i in range(window, n_bars):
        s = cs[i] - cs[i - window]
        s2 = cs2[i] - cs2[i - window]
        m = s / window
        var = s2 / window - m * m
        pred_std[i] = np.sqrt(max(var, 0))
    return pred_std


# ══════════════════════════════════════════════════════════════════════════════
# COMBO 1: EMA span5000 + Book Confirmation
# ══════════════════════════════════════════════════════════════════════════════

def generate_combo1(ema_z, imbalance, vol, vol_thresholds, time_mask, n_bars,
                    conv_threshold, vg):
    """
    Signal = ema_zscore WHEN (ema_z > 0 AND imbalance > 0)
                           OR (ema_z < 0 AND imbalance < 0)
             ELSE 0
    Then apply conv threshold on |signal|.
    """
    signal = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_bars):
        z = ema_z[i]
        if np.isnan(z):
            continue
        imb = imbalance[i]
        # Book confirms direction
        if (z > 0 and imb > 0) or (z < 0 and imb < 0):
            if abs(z) >= conv_threshold:
                signal[i] = z

    signal = apply_vol_gate(signal, vol, vol_thresholds, vg)
    signal[~time_mask] = 0.0
    return np.nan_to_num(signal, nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# COMBO 2: EMA + Book + TP5  (same signal as combo 1, sim uses TP)
# ══════════════════════════════════════════════════════════════════════════════
# Signal generation is identical to combo 1.
# The TP5 is applied at the sim level, not the signal level.


# ══════════════════════════════════════════════════════════════════════════════
# COMBO 3: Momentum on EMA signal + TP
# ══════════════════════════════════════════════════════════════════════════════

def generate_combo3(ema_z, n_bars, accel_threshold, vol, vol_thresholds,
                    time_mask, vg, accel_window=50):
    """
    acceleration = ema_z[i] - ema_z[i-50]
    Signal = acceleration WHEN abs(acceleration) > threshold ELSE 0
    """
    signal = np.zeros(n_bars, dtype=np.float64)
    for i in range(accel_window, n_bars):
        if np.isnan(ema_z[i]) or np.isnan(ema_z[i - accel_window]):
            continue
        accel = ema_z[i] - ema_z[i - accel_window]
        if abs(accel) > accel_threshold:
            signal[i] = accel

    signal = apply_vol_gate(signal, vol, vol_thresholds, vg)
    signal[~time_mask] = 0.0
    return np.nan_to_num(signal, nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# COMBO 4: EMA + pred_std filter
# ══════════════════════════════════════════════════════════════════════════════

def generate_combo4(ema_z, pred_std, vol, vol_thresholds, time_mask, n_bars,
                    conv_threshold, max_std, vg):
    """
    Signal = ema_zscore WHEN pred_std < max_std ELSE 0
    Also apply conv threshold on |ema_z|.
    """
    signal = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_bars):
        z = ema_z[i]
        if np.isnan(z) or np.isnan(pred_std[i]):
            continue
        if abs(z) >= conv_threshold and pred_std[i] < max_std:
            signal[i] = z

    signal = apply_vol_gate(signal, vol, vol_thresholds, vg)
    signal[~time_mask] = 0.0
    return np.nan_to_num(signal, nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# COMBO 5: Triple filter (EMA + book + pred_std)
# ══════════════════════════════════════════════════════════════════════════════

def generate_combo5(ema_z, imbalance, pred_std, vol, vol_thresholds, time_mask,
                    n_bars, conv_threshold, max_std, vg):
    """
    Signal = ema_zscore WHEN (book confirms AND pred_std < max_std) ELSE 0
    Book confirms = (ema_z > 0 AND imbalance > 0) OR (ema_z < 0 AND imbalance < 0)
    """
    signal = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_bars):
        z = ema_z[i]
        if np.isnan(z) or np.isnan(pred_std[i]):
            continue
        imb = imbalance[i]
        book_confirms = (z > 0 and imb > 0) or (z < 0 and imb < 0)
        if book_confirms and pred_std[i] < max_std and abs(z) >= conv_threshold:
            signal[i] = z

    signal = apply_vol_gate(signal, vol, vol_thresholds, vg)
    signal[~time_mask] = 0.0
    return np.nan_to_num(signal, nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION FILE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_config_list(combos_to_run):
    """Build list of all (combo_id, params, label) tuples."""
    configs = []

    # COMBO 1: EMA + Book
    if 1 in combos_to_run:
        for conv in [1.5, 2.0, 2.5]:
            for vg in [50, 70]:
                label = f'combo1_ema_book_conv{conv}_vol{vg}'
                configs.append((1, {'conv': conv, 'vg': vg}, label))

    # COMBO 2: EMA + Book + TP5 (signal is same as combo 1, TP applied in sim)
    if 2 in combos_to_run:
        for conv in [2.0, 2.5]:
            for vg in [50, 70]:
                label = f'combo2_ema_book_tp5_conv{conv}_vol{vg}'
                configs.append((2, {'conv': conv, 'vg': vg}, label))

    # COMBO 3: Momentum + TP
    if 3 in combos_to_run:
        for accel_thr in [0.3, 0.5, 0.75]:
            for vg in [0, 50, 70]:
                label = f'combo3_momentum_athr{accel_thr}_vol{vg}'
                configs.append((3, {'accel_thr': accel_thr, 'vg': vg}, label))

    # COMBO 4: EMA + pred_std
    if 4 in combos_to_run:
        for max_std in [0.08, 0.10, 0.15]:
            for conv in [1.5, 2.0, 2.5]:
                for vg in [50, 70]:
                    label = f'combo4_ema_predstd{max_std}_conv{conv}_vol{vg}'
                    configs.append((4, {'max_std': max_std, 'conv': conv, 'vg': vg}, label))

    # COMBO 5: Triple filter
    if 5 in combos_to_run:
        for conv in [1.5, 2.0, 2.5]:
            for vg in [50, 70]:
                # Without TP
                label = f'combo5_triple_conv{conv}_vol{vg}'
                configs.append((5, {'conv': conv, 'vg': vg, 'tp': False}, label))
                # With TP5
                label_tp = f'combo5_triple_tp5_conv{conv}_vol{vg}'
                configs.append((5, {'conv': conv, 'vg': vg, 'tp': True}, label_tp))

    return configs


def generate_all_predictions(wf_data, dates, combos_to_run):
    """Generate per-day NPZ prediction files for all combos."""
    configs = build_config_list(combos_to_run)
    log.info(f"Total signal configs: {len(configs)}")
    log.info(f"Dates: {len(dates)}, total prediction files: ~{len(configs) * len(dates)}")

    # For combos 2 and 5 with tp=True, the signal file is the same as the
    # non-TP variant — the TP is applied at the sim level.  We still need
    # separate labels so the sim layer can differentiate.

    saved = {}  # (date, label) -> filepath

    for di, date in enumerate(dates):
        preds_raw = wf_data[f'{date}_preds'].astype(np.float64)
        mid = wf_data[f'{date}_mid'].astype(np.float64)
        n_bars = len(mid)
        if n_bars < 5000:
            log.warning(f"Skipping {date}: only {n_bars} bars")
            continue

        # Check MBO file exists
        nodash = date.replace('-', '')
        mbo_zst = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            log.warning(f"No MBO for {date}, skipping")
            continue

        # ── Step 1: Align CNN offset ──
        aligned = np.zeros(n_bars, dtype=np.float64)
        end = min(n_bars, len(preds_raw) + CNN_OFFSET)
        aligned[CNN_OFFSET:end] = preds_raw[:end - CNN_OFFSET]

        # ── Step 2: Expanding z-score first ──
        exp_z = zscore_expanding(aligned)

        # ── Step 3: EMA z-score (span 5000) on top of expanding z-score ──
        ema_z = zscore_ema(exp_z, span=5000)

        # ── Shared features (computed once per day) ──
        vol = compute_trailing_vol(mid)
        all_vgs = set()
        for _, params, _ in configs:
            all_vgs.add(params.get('vg', 0))
        vol_thresholds = precompute_vol_percentiles(vol, tuple(v for v in all_vgs if v > 0))
        tmask = time_mask_morning_afternoon(n_bars)

        # Lazy-load book imbalance and pred_std (only if needed)
        imbalance = None
        pred_std = None

        needs_book = any(c in combos_to_run for c in [1, 2, 5])
        needs_pred_std = any(c in combos_to_run for c in [4, 5])

        if needs_book:
            imbalance = load_book_imbalance(date, n_bars)
            if imbalance is None:
                log.warning(f"  No book tensors for {date}, skipping book-dependent combos")

        if needs_pred_std:
            pred_std = compute_pred_std(aligned, n_bars, window=3000)

        # ── Generate each config ──
        for combo_id, params, label in configs:
            vg = params.get('vg', 0)

            if combo_id == 1:
                if imbalance is None:
                    continue
                sig = generate_combo1(ema_z, imbalance, vol, vol_thresholds,
                                      tmask, n_bars, params['conv'], vg)

            elif combo_id == 2:
                # Same signal as combo 1 (TP applied in sim)
                if imbalance is None:
                    continue
                sig = generate_combo1(ema_z, imbalance, vol, vol_thresholds,
                                      tmask, n_bars, params['conv'], vg)

            elif combo_id == 3:
                sig = generate_combo3(ema_z, n_bars, params['accel_thr'],
                                      vol, vol_thresholds, tmask, vg)

            elif combo_id == 4:
                if pred_std is None:
                    continue
                sig = generate_combo4(ema_z, pred_std, vol, vol_thresholds,
                                      tmask, n_bars, params['conv'],
                                      params['max_std'], vg)

            elif combo_id == 5:
                if imbalance is None or pred_std is None:
                    continue
                sig = generate_combo5(ema_z, imbalance, pred_std, vol,
                                      vol_thresholds, tmask, n_bars,
                                      params['conv'], 0.10, vg)
            else:
                continue

            fpath = PRED_OUT_DIR / f'{date}_{label}.npz'
            np.savez_compressed(str(fpath), predictions=sig.astype(np.float32))
            saved[(date, label)] = str(fpath)

        if (di + 1) % 5 == 0 or di == 0 or di == len(dates) - 1:
            log.info(f"  Generated {di+1}/{len(dates)} days ({len(saved)} files so far)")

        del mid, preds_raw, aligned, exp_z, ema_z, vol, vol_thresholds
        if imbalance is not None:
            del imbalance
        if pred_std is not None:
            del pred_std
        gc.collect()

    log.info(f"Total prediction files generated: {len(saved)}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def get_sim_configs_for_label(label):
    """
    Return list of (hold_ms, chase_t, chase_r, tp_ticks, sim_suffix) for a label.
    - Combo 2 labels already have TP5 baked in → sim with TP5
    - Combo 3 has TP variants → sim with TP5, TP8, TP10 plus plain 30min
    - Combo 5 with tp5 in label → sim with TP5
    - Everything else: 30min hold + also 30min with TP5
    """
    sim_cfgs = []

    if 'combo2_' in label:
        # Combo 2: always TP5
        sim_cfgs.append((1800000, 1, 3, 5, 'hold30m_tp5'))
    elif 'combo3_' in label:
        # Combo 3: plain + TP5, TP8, TP10
        sim_cfgs.append((1800000, 1, 3, None, 'hold30m'))
        sim_cfgs.append((1800000, 1, 3, 5, 'hold30m_tp5'))
        sim_cfgs.append((1800000, 1, 3, 8, 'hold30m_tp8'))
        sim_cfgs.append((1800000, 1, 3, 10, 'hold30m_tp10'))
    elif 'combo5_triple_tp5_' in label:
        # Combo 5 TP variant
        sim_cfgs.append((1800000, 1, 3, 5, 'hold30m_tp5'))
    else:
        # Combos 1, 4, 5 (non-tp): plain + TP5
        sim_cfgs.append((1800000, 1, 3, None, 'hold30m'))
        sim_cfgs.append((1800000, 1, 3, 5, 'hold30m_tp5'))

    return sim_cfgs


def run_single_sim(mbo_file, pred_file, output_file, hold_ms, chase_max_ticks,
                   chase_max_reprices, tp_ticks):
    """Run one Rust fill_sim job."""
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(output_file),
        '--hold-ms', str(hold_ms),
        '--signal-threshold', '0',   # threshold already baked into predictions
        '--latency-ms', '0',
        '--quiet',
        '--chase-entry',
        '--chase-max-ticks', str(chase_max_ticks),
        '--chase-max-reprices', str(chase_max_reprices),
    ]
    if tp_ticks is not None:
        cmd += ['--take-profit-ticks', str(tp_ticks)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return None
        if Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        log.warning(f"Sim error: {e}")
    return None


def run_sweep(saved_files, workers=24):
    """Run all sim jobs in parallel."""
    jobs = []
    for (date, signal_label), pred_file in saved_files.items():
        nodash = date.replace('-', '')
        mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        if not mbo_file.exists():
            mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
        if not mbo_file.exists():
            continue

        for hold_ms, chase_t, chase_r, tp, sim_suffix in get_sim_configs_for_label(signal_label):
            out_label = f'{signal_label}_{sim_suffix}'
            out_file = SIM_OUT_DIR / f'{out_label}_{date}.json'

            if out_file.exists():
                # Load cached result
                try:
                    with open(out_file) as f:
                        existing = json.load(f)
                    jobs.append(('cached', out_label, date, existing))
                    continue
                except Exception:
                    pass

            jobs.append(('run', out_label, date, {
                'mbo': str(mbo_file),
                'pred': pred_file,
                'out': str(out_file),
                'hold_ms': hold_ms,
                'chase_t': chase_t,
                'chase_r': chase_r,
                'tp': tp,
            }))

    # Separate cached vs to-run
    cached = [(label, date, res) for typ, label, date, res in jobs if typ == 'cached']
    to_run = [(label, date, params) for typ, label, date, params in jobs if typ == 'run']

    log.info(f"Total sim jobs: {len(jobs)} ({len(cached)} cached, {len(to_run)} to run)")
    log.info(f"Workers: {workers}")

    results = defaultdict(dict)

    # Load cached
    for label, date, res in cached:
        results[label][date] = res

    if not to_run:
        log.info("All jobs cached, skipping simulation.")
        return dict(results)

    # Run remaining
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for label, date, params in to_run:
            f = executor.submit(
                run_single_sim,
                params['mbo'], params['pred'], params['out'],
                params['hold_ms'], params['chase_t'], params['chase_r'], params['tp']
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

            if completed % 50 == 0 or completed == len(to_run):
                el = time.time() - t0
                rate = completed / el if el > 0 else 0
                eta = (len(to_run) - completed) / rate / 60 if rate > 0 else 0
                log.info(f"  [{completed}/{len(to_run)}] {rate:.1f} jobs/s, ETA {eta:.1f}min")

    log.info(f"Simulation done: {completed} jobs in {time.time()-t0:.0f}s")
    return dict(results)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION & REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_and_report(results):
    """Aggregate per-day results into per-config summaries, print top 20."""
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
        win_rate = total_wins / total_trades if total_trades > 0 else 0
        fill_rate = total_filled / total_signals if total_signals > 0 else 0
        avg_trade = np.mean(all_trade_pnls) if all_trade_pnls else 0

        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = abs((cum - peak).min()) if len(cum) > 0 else 0

        pct_profitable = sum(1 for p in daily_pnls if p > 0) / max(n_days, 1)

        # Parse combo number from label
        combo = config_label.split('_')[0]  # 'combo1', 'combo2', etc.

        summaries.append({
            'config': config_label,
            'combo': combo,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'pct_profitable_days': round(pct_profitable, 4),
            'sharpe': round(sharpe, 3),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade, 2),
            'avg_trade_ticks': round(avg_trade / TICK_VALUE, 2),
            'max_dd': round(max_dd, 2),
            'annualized': round(avg_daily * 252, 0),
        })

    # Sort by Sharpe, filter to configs with 20+ days
    summaries_filtered = [s for s in summaries if s['n_days'] >= 20]
    summaries_filtered.sort(key=lambda x: x['sharpe'], reverse=True)

    # Also keep full list sorted by Sharpe
    summaries.sort(key=lambda x: x['sharpe'], reverse=True)

    # ── Print top 20 (20+ days) ──
    log.info("\n" + "=" * 140)
    log.info("COMBINATION STRATEGY SWEEP — TOP 20 BY SHARPE (20+ days, Rust MBO Fill Sim)")
    log.info("=" * 140)
    log.info(f"{'#':>3} {'Combo':>7} {'Config':<65} {'Sharpe':>7} {'P&L':>10} {'Days':>5} "
             f"{'Trades':>6} {'Fill%':>6} {'WR%':>5} {'Prof%':>5} {'AvgTrd':>8} "
             f"{'MaxDD':>8} {'Annual':>10}")
    log.info("-" * 140)

    for i, s in enumerate(summaries_filtered[:20]):
        log.info(
            f"{i+1:>3} {s['combo']:>7} {s['config']:<65} "
            f"{s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} {s['n_days']:>5} "
            f"{s['n_trades']:>6} {s['fill_rate']*100:>5.1f}% {s['win_rate']*100:>4.1f}% "
            f"{s['pct_profitable_days']*100:>4.0f}% ${s['avg_trade_pnl']:>7.2f} "
            f"${s['max_dd']:>7,.0f} ${s['annualized']:>9,.0f}"
        )

    # ── Best per combo ──
    log.info("\n" + "=" * 110)
    log.info("BEST CONFIG PER COMBO (20+ days)")
    log.info("=" * 110)
    for combo_num in range(1, 6):
        combo_key = f'combo{combo_num}'
        combo_results = [s for s in summaries_filtered if s['combo'] == combo_key]
        if combo_results:
            best = combo_results[0]
            log.info(f"\n  COMBO {combo_num}: {best['config']}")
            log.info(f"    Sharpe: {best['sharpe']:.2f}, P&L: ${best['total_pnl']:,.2f}, "
                     f"Trades: {best['n_trades']}, Fill: {best['fill_rate']:.1%}, "
                     f"WR: {best['win_rate']:.1%}, Days: {best['n_days']}, "
                     f"Annual: ${best['annualized']:,.0f}")
        else:
            log.info(f"\n  COMBO {combo_num}: No results with 20+ days")

    # ── Reference baseline ──
    log.info("\n" + "=" * 110)
    log.info("REFERENCE — Baseline (vol70/conv2.5/1t/3r/30min):")
    log.info("  IS:         Sharpe 3.28, +$15,479/74d, 130 trades, 8.6% fill, 53.8% WR")
    log.info("  OOT static: Sharpe 1.58, +$4,082/68d, 102 trades, 9.6% fill")
    log.info("=" * 110)

    # ── All results table (including <20 days) ──
    if len(summaries) > len(summaries_filtered):
        short_configs = [s for s in summaries if s['n_days'] < 20]
        if short_configs:
            log.info(f"\nNote: {len(short_configs)} configs had <20 days (excluded from top 20).")

    # ── Save results ──
    out_file = RESULTS_DIR / f'combo_sweep_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'description': 'Combination Strategy Sweep (EMA+Book+PredStd+Momentum+TP)',
            'n_configs_total': len(summaries),
            'n_configs_20plus_days': len(summaries_filtered),
            'top20': summaries_filtered[:20],
            'all_summaries': summaries,
        }, f, indent=2)
    log.info(f"\nResults saved: {out_file}")

    return summaries_filtered


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Combination Strategy Sweep')
    parser.add_argument('--workers', type=int, default=24,
                        help='Parallel sim workers (default: 24)')
    parser.add_argument('--combos', type=str, default='1,2,3,4,5',
                        help='Which combos to run (default: 1,2,3,4,5)')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip prediction generation, use existing files')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, only aggregate existing results')
    args = parser.parse_args()

    combos = [int(x) for x in args.combos.split(',')]

    log.info("=" * 80)
    log.info("COMBINATION STRATEGY SWEEP — 5 Stacked Approaches")
    log.info("=" * 80)
    log.info(f"  Combos:    {combos}")
    log.info(f"  Workers:   {args.workers}")
    log.info(f"  Binary:    {BINARY}")
    log.info(f"  Preds:     {PRED_FILE}")
    log.info(f"  MBO:       {MBO_DIR}")
    log.info(f"  Book:      {BOOK_DIR}")
    log.info(f"  Pred out:  {PRED_OUT_DIR}")
    log.info(f"  Sim out:   {SIM_OUT_DIR}")
    log.info("")
    log.info("  COMBO 1: EMA(5000) + Book Confirmation      | conv=[1.5,2.0,2.5] vol=[50,70]")
    log.info("  COMBO 2: EMA + Book + TP5                   | conv=[2.0,2.5] vol=[50,70]")
    log.info("  COMBO 3: Momentum(accel) on EMA + TP        | athr=[0.3,0.5,0.75] vol=[0,50,70]")
    log.info("  COMBO 4: EMA + pred_std filter              | std=[.08,.10,.15] conv=[1.5,2.0,2.5] vol=[50,70]")
    log.info("  COMBO 5: Triple (EMA+book+pred_std<0.10)    | conv=[1.5,2.0,2.5] vol=[50,70] +/- TP5")
    log.info("")

    configs = build_config_list(combos)
    log.info(f"  Signal configs: {len(configs)}")
    log.info("=" * 80)

    if not BINARY.exists():
        log.error(f"Binary not found: {BINARY}")
        sys.exit(1)
    if not PRED_FILE.exists():
        log.error(f"Predictions not found: {PRED_FILE}")
        sys.exit(1)

    # ── Step 1: Generate prediction files ──
    if args.skip_gen or args.skip_sim:
        log.info("Loading existing prediction files...")
        saved = {}
        for f in PRED_OUT_DIR.glob('*.npz'):
            stem = f.stem
            date = stem[:10]
            label = stem[11:]
            # Filter by requested combos
            combo_match = False
            for c in combos:
                if f'combo{c}_' in label:
                    combo_match = True
                    break
            if combo_match:
                saved[(date, label)] = str(f)
        log.info(f"Found {len(saved)} existing prediction files")
    else:
        log.info("Loading WF predictions...")
        wf_data = np.load(str(PRED_FILE), allow_pickle=True)
        dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
        log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
        saved = generate_all_predictions(wf_data, dates, combos)
        wf_data.close()

    if not saved:
        log.error("No prediction files. Exiting.")
        sys.exit(1)

    # ── Step 2: Run sim sweep ──
    if not args.skip_sim:
        results = run_sweep(saved, workers=args.workers)
    else:
        # Load from existing result files
        log.info("Loading existing sim results...")
        results = defaultdict(dict)
        for f in SIM_OUT_DIR.glob('*.json'):
            try:
                with open(f) as fh:
                    res = json.load(fh)
                stem = f.stem
                # Parse: {signal_label}_{sim_suffix}_{date}.json
                # Date is always last 10 chars
                date = stem[-10:]
                label = stem[:-11]
                # Filter by combos
                combo_match = False
                for c in combos:
                    if f'combo{c}_' in label:
                        combo_match = True
                        break
                if combo_match:
                    results[label][date] = res
            except Exception:
                continue
        results = dict(results)
        log.info(f"Loaded {sum(len(v) for v in results.values())} existing results "
                 f"across {len(results)} configs")

    if not results:
        log.error("No results. Exiting.")
        sys.exit(1)

    # ── Step 3: Aggregate and report ──
    summaries = aggregate_and_report(results)

    log.info(f"\nLog: {RESULTS_DIR / f'combo_sweep_{_ts}.log'}")
    log.info("DONE.")
    return summaries


if __name__ == '__main__':
    main()
