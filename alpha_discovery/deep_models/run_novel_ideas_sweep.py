#!/usr/bin/env python3
"""
Novel Signal Processing Sweep — 3 New Ideas
=============================================
IDEA 1: Z-score MOMENTUM (rate of change of z-score as signal)
IDEA 2: Book-conditioned entry (zero z-score when order book doesn't confirm)
IDEA 3: Pred_std uncertainty filter (zero signal when model predictions are noisy)

Each idea generates per-day NPZ prediction files, then runs the Rust MBO fill sim.
Tests with both 30min hold and TP5 exit.

Usage:
    python alpha_discovery/deep_models/run_novel_ideas_sweep.py --workers 24
    python alpha_discovery/deep_models/run_novel_ideas_sweep.py --workers 24 --ideas 1,2,3
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

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
BOOK_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_novel_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_novel_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('novel_sweep')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'novel_sweep_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ══════════════════════════════════════════════════════════════════════════════
# SHARED SIGNAL PROCESSING (same as run_wf_fill_sim.py)
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


# ══════════════════════════════════════════════════════════════════════════════
# IDEA 1: Z-SCORE MOMENTUM (rate of change of z-score)
# ══════════════════════════════════════════════════════════════════════════════

def generate_idea1_momentum(preds_aligned, mid, z_score, vol, vol_thresholds,
                            time_mask, n_bars, accel_window, accel_threshold, vg):
    """
    Signal = z_acceleration WHEN abs(z_acceleration) > threshold ELSE 0
    z_acceleration = z_score[i] - z_score[i - N]
    """
    # Compute z-score acceleration
    z_accel = np.zeros(n_bars, dtype=np.float64)
    for i in range(accel_window, n_bars):
        if not np.isnan(z_score[i]) and not np.isnan(z_score[i - accel_window]):
            z_accel[i] = z_score[i] - z_score[i - accel_window]

    # Apply acceleration threshold: only keep when abs(accel) > threshold
    signal = np.where(np.abs(z_accel) > accel_threshold, z_accel, 0.0)

    # Apply vol gate
    signal = apply_vol_gate(signal, vol, vol_thresholds, vg)

    # Apply time mask
    signal[~time_mask] = 0.0

    return np.nan_to_num(signal, nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# IDEA 2: BOOK-CONDITIONED ENTRY
# ══════════════════════════════════════════════════════════════════════════════

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
        # Resize if needed (truncate or pad)
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


def generate_idea2_book_conditioned(z_score, imbalance, vol, vol_thresholds,
                                     time_mask, n_bars, imb_threshold, conv_threshold, vg):
    """
    Signal = z_score WHEN (z_score > conv AND imbalance > imb_threshold)
                       OR (z_score < -conv AND imbalance < -imb_threshold)
             ELSE 0
    """
    signal = np.zeros(n_bars, dtype=np.float64)

    for i in range(n_bars):
        z = z_score[i]
        if np.isnan(z):
            continue
        imb = imbalance[i]
        # Long: z-score positive AND book confirms (more bids)
        if z > conv_threshold and imb > imb_threshold:
            signal[i] = z
        # Short: z-score negative AND book confirms (more asks)
        elif z < -conv_threshold and imb < -imb_threshold:
            signal[i] = z

    # Apply vol gate
    signal = apply_vol_gate(signal, vol, vol_thresholds, vg)

    # Apply time mask
    signal[~time_mask] = 0.0

    return np.nan_to_num(signal, nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# IDEA 3: PRED_STD UNCERTAINTY FILTER
# ══════════════════════════════════════════════════════════════════════════════

def generate_idea3_pred_std_filter(preds_aligned, z_score, vol, vol_thresholds,
                                    time_mask, n_bars, max_std, conv_threshold, vg):
    """
    Signal = z_score WHEN pred_std < max_std ELSE 0
    pred_std = rolling std of raw predictions (window 3000)
    We apply conv_threshold on top: only keep |z_score| > conv_threshold.
    """
    # Compute rolling std of raw predictions (window 3000)
    window = 3000
    pred_std = np.full(n_bars, np.nan, dtype=np.float64)
    cs = np.cumsum(np.nan_to_num(preds_aligned, nan=0.0))
    cs2 = np.cumsum(np.nan_to_num(preds_aligned, nan=0.0) ** 2)
    for i in range(window, n_bars):
        s = cs[i] - cs[i - window]
        s2 = cs2[i] - cs2[i - window]
        m = s / window
        var = s2 / window - m * m
        pred_std[i] = np.sqrt(max(var, 0))

    # Signal = z_score when |z_score| > conv_threshold AND pred_std < max_std
    signal = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_bars):
        z = z_score[i]
        if np.isnan(z) or np.isnan(pred_std[i]):
            continue
        if abs(z) > conv_threshold and pred_std[i] < max_std:
            signal[i] = z

    # Apply vol gate
    signal = apply_vol_gate(signal, vol, vol_thresholds, vg)

    # Apply time mask
    signal[~time_mask] = 0.0

    return np.nan_to_num(signal, nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION FILE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_predictions(wf_data, dates, ideas_to_run):
    """Generate per-day NPZ prediction files for all 3 ideas."""
    saved = {}  # (date, idea_label) -> filepath
    total_configs = 0

    # Pre-define all configs
    # IDEA 1: Z-score momentum
    idea1_configs = []
    if 1 in ideas_to_run:
        for accel_window in [10, 50, 100]:
            for accel_threshold in [0.5, 1.0, 1.5]:
                for vg in [0, 50, 70]:
                    label = f'idea1_momentum_w{accel_window}_thr{accel_threshold}_vol{vg}'
                    idea1_configs.append((accel_window, accel_threshold, vg, label))

    # IDEA 2: Book-conditioned entry
    idea2_configs = []
    if 2 in ideas_to_run:
        for imb_thr in [0.0, 0.05, 0.10, 0.15]:
            for conv_thr in [1.5, 2.0, 2.5]:
                for vg in [50, 70]:
                    label = f'idea2_book_imb{imb_thr}_conv{conv_thr}_vol{vg}'
                    idea2_configs.append((imb_thr, conv_thr, vg, label))

    # IDEA 3: Pred_std uncertainty filter
    idea3_configs = []
    if 3 in ideas_to_run:
        for max_std in [0.05, 0.08, 0.10, 0.15]:
            for conv_thr in [1.5, 2.0, 2.5]:
                for vg in [50, 70]:
                    label = f'idea3_predstd_max{max_std}_conv{conv_thr}_vol{vg}'
                    idea3_configs.append((max_std, conv_thr, vg, label))

    total_configs = len(idea1_configs) + len(idea2_configs) + len(idea3_configs)
    log.info(f"Signal configs: Idea1={len(idea1_configs)}, Idea2={len(idea2_configs)}, "
             f"Idea3={len(idea3_configs)}, Total={total_configs}")
    log.info(f"Dates: {len(dates)}, total prediction files: ~{total_configs * len(dates)}")

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

        # Align CNN offset
        aligned = np.zeros(n_bars, dtype=np.float64)
        end = min(n_bars, len(preds_raw) + CNN_OFFSET)
        aligned[CNN_OFFSET:end] = preds_raw[:end - CNN_OFFSET]

        # Base z-score (used by all ideas)
        z_score = zscore_expanding(aligned)

        # Vol processing
        vol = compute_trailing_vol(mid)
        all_vgs = set()
        for cfg in idea1_configs:
            all_vgs.add(cfg[2])
        for cfg in idea2_configs:
            all_vgs.add(cfg[2])
        for cfg in idea3_configs:
            all_vgs.add(cfg[2])
        vol_thresholds = precompute_vol_percentiles(vol, tuple(v for v in all_vgs if v > 0))

        # Time mask
        tmask = time_mask_morning_afternoon(n_bars)

        # ── IDEA 1: Z-score momentum ──
        for accel_window, accel_threshold, vg, label in idea1_configs:
            sig = generate_idea1_momentum(
                aligned, mid, z_score, vol, vol_thresholds, tmask,
                n_bars, accel_window, accel_threshold, vg
            )
            fpath = PRED_OUT_DIR / f'{date}_{label}.npz'
            np.savez_compressed(str(fpath), predictions=sig.astype(np.float32))
            saved[(date, label)] = str(fpath)

        # ── IDEA 2: Book-conditioned entry ──
        if idea2_configs:
            imbalance = load_book_imbalance(date, n_bars)
            if imbalance is not None:
                for imb_thr, conv_thr, vg, label in idea2_configs:
                    sig = generate_idea2_book_conditioned(
                        z_score, imbalance, vol, vol_thresholds, tmask,
                        n_bars, imb_thr, conv_thr, vg
                    )
                    fpath = PRED_OUT_DIR / f'{date}_{label}.npz'
                    np.savez_compressed(str(fpath), predictions=sig.astype(np.float32))
                    saved[(date, label)] = str(fpath)
            else:
                log.warning(f"  No book tensors for {date}, skipping Idea 2")

        # ── IDEA 3: Pred_std uncertainty filter ──
        for max_std, conv_thr, vg, label in idea3_configs:
            sig = generate_idea3_pred_std_filter(
                aligned, z_score, vol, vol_thresholds, tmask,
                n_bars, max_std, conv_thr, vg
            )
            fpath = PRED_OUT_DIR / f'{date}_{label}.npz'
            np.savez_compressed(str(fpath), predictions=sig.astype(np.float32))
            saved[(date, label)] = str(fpath)

        if (di + 1) % 5 == 0 or di == 0 or di == len(dates) - 1:
            log.info(f"  Generated {di+1}/{len(dates)} days ({len(saved)} files so far)")

        del mid, preds_raw, aligned, z_score, vol, vol_thresholds
        gc.collect()

    log.info(f"Total prediction files generated: {len(saved)}")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

# Sim configs: (hold_ms, chase_max_ticks, chase_max_reprices, tp_ticks, label_suffix)
SIM_CONFIGS = [
    # Standard 30min hold with chase
    (1800000, 1, 3, None, 'hold30m_chase'),
    # TP5 exit (our best exit method)
    (1800000, 1, 3, 5, 'hold30m_tp5_chase'),
    # TP8 for comparison
    (1800000, 1, 3, 8, 'hold30m_tp8_chase'),
    # TP10 for comparison
    (1800000, 1, 3, 10, 'hold30m_tp10_chase'),
]


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
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
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
    for (date, idea_label), pred_file in saved_files.items():
        nodash = date.replace('-', '')
        mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        if not mbo_file.exists():
            mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
        if not mbo_file.exists():
            continue

        for hold_ms, chase_t, chase_r, tp, sim_label in SIM_CONFIGS:
            out_label = f'{idea_label}_{sim_label}'
            out_file = SIM_OUT_DIR / f'{out_label}_{date}.json'
            if out_file.exists():
                # Load existing result
                try:
                    with open(out_file) as f:
                        existing = json.load(f)
                    jobs.append(('cached', out_label, date, existing))
                    continue
                except:
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

            if completed % 100 == 0 or completed == len(to_run):
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
    """Aggregate per-day results into per-config summaries, print top 30."""
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

        # Parse idea number from label
        idea = config_label.split('_')[0]  # 'idea1', 'idea2', 'idea3'

        summaries.append({
            'config': config_label,
            'idea': idea,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe': round(sharpe, 3),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade, 2),
            'avg_trade_ticks': round(avg_trade / TICK_VALUE, 2),
            'max_dd': round(max_dd, 2),
            'annualized': round(avg_daily * 252, 0),
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)

    # Print top 30
    log.info("\n" + "=" * 130)
    log.info("NOVEL IDEAS SWEEP — TOP 30 BY SHARPE (Rust MBO Fill Sim, Real Fills)")
    log.info("=" * 130)
    log.info(f"{'#':>3} {'Idea':>6} {'Config':<70} {'Sharpe':>7} {'P&L':>10} {'Trades':>6} "
             f"{'Fill%':>6} {'WR%':>5} {'AvgTrd':>8} {'MaxDD':>8} {'Annual':>10}")
    log.info("-" * 130)

    for i, s in enumerate(summaries[:30]):
        log.info(
            f"{i+1:>3} {s['idea']:>6} {s['config']:<70} "
            f"{s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} {s['n_trades']:>6} "
            f"{s['fill_rate']*100:>5.1f}% {s['win_rate']*100:>4.1f}% "
            f"${s['avg_trade_pnl']:>7.2f} ${s['max_dd']:>7,.0f} ${s['annualized']:>9,.0f}"
        )

    # Print best per idea
    log.info("\n" + "=" * 100)
    log.info("BEST CONFIG PER IDEA")
    log.info("=" * 100)
    for idea_num in ['idea1', 'idea2', 'idea3']:
        idea_results = [s for s in summaries if s['idea'] == idea_num]
        if idea_results:
            best = idea_results[0]
            log.info(f"\n  {idea_num.upper()}: {best['config']}")
            log.info(f"    Sharpe: {best['sharpe']:.2f}, P&L: ${best['total_pnl']:,.2f}, "
                     f"Trades: {best['n_trades']}, Fill: {best['fill_rate']:.1%}, "
                     f"WR: {best['win_rate']:.1%}, Annual: ${best['annualized']:,.0f}")
        else:
            log.info(f"\n  {idea_num.upper()}: No results")

    # Compare to baseline
    log.info("\n" + "=" * 100)
    log.info("REFERENCE — Baseline (IS best vol70/conv2.5/1t/3r/30min):")
    log.info("  IS: Sharpe 3.28, +$15,479/74d, 130 trades, 8.6% fill, 53.8% WR")
    log.info("  OOT static: Sharpe 1.58, +$4,082/68d, 102 trades, 9.6% fill")
    log.info("=" * 100)

    # Save results
    out_file = RESULTS_DIR / f'novel_sweep_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'n_configs': len(summaries),
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
    REMOTE_DIR = '/home/jupiter/Lvl3Quant/data/processed/cnn_wf_novel_predictions'

    log.info(f"Uploading {len(saved_files)} prediction files to Jupiter...")

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(JUPITER_HOST, username=JUPITER_USER, password=JUPITER_PW, timeout=10)

        # Create remote directory
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

    # Jupiter sweep script (runs remotely)
    remote_script = '''#!/usr/bin/env python3
"""Remote sweep on Jupiter for novel ideas predictions."""
import sys, json, time, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_novel_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_novel_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_CONFIGS = [
    (1800000, 1, 3, None, "hold30m_chase"),
    (1800000, 1, 3, 5, "hold30m_tp5_chase"),
    (1800000, 1, 3, 8, "hold30m_tp8_chase"),
    (1800000, 1, 3, 10, "hold30m_tp10_chase"),
]

def run_sim(mbo, pred, out, hold_ms, chase_t, chase_r, tp):
    cmd = [str(BINARY), "--mbo-file", str(mbo), "--predictions", str(pred),
           "--output", str(out), "--hold-ms", str(hold_ms), "--signal-threshold", "0",
           "--latency-ms", "0", "--quiet", "--chase-entry",
           "--chase-max-ticks", str(chase_t), "--chase-max-reprices", str(chase_r)]
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

# Build jobs
pred_files = sorted(PRED_DIR.glob("*.npz"))
print(f"Found {len(pred_files)} prediction files")

jobs = []
for pf in pred_files:
    stem = pf.stem  # e.g. 2025-12-01_idea1_momentum_w10_thr0.5_vol50
    date = stem[:10]
    nodash = date.replace("-", "")
    mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
    if not mbo.exists():
        mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn"
    if not mbo.exists():
        continue
    for hold_ms, chase_t, chase_r, tp, sim_label in SIM_CONFIGS:
        out_label = f"{stem[11:]}_{sim_label}"
        out_file = OUT_DIR / f"{out_label}_{date}.json"
        if out_file.exists():
            continue
        jobs.append((str(mbo), str(pf), str(out_file), hold_ms, chase_t, chase_r, tp))

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

        # Write script
        sftp = ssh.open_sftp()
        remote_script_path = '/home/jupiter/Lvl3Quant/run_jupiter_novel_sweep.py'
        with sftp.open(remote_script_path, 'w') as f:
            f.write(remote_script)
        sftp.close()

        # Launch in background with nohup
        cmd = f'cd /home/jupiter/Lvl3Quant && nohup python3 {remote_script_path} > novel_sweep_jupiter.log 2>&1 &'
        stdin, stdout, stderr = ssh.exec_command(cmd)
        time.sleep(2)
        log.info(f"Launched Jupiter sweep: {cmd}")
        log.info(f"Monitor: ssh jupiter@jupiter 'tail -f /home/jupiter/Lvl3Quant/novel_sweep_jupiter.log'")
        ssh.close()
        return True
    except Exception as e:
        log.error(f"Jupiter launch failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Novel Signal Processing Sweep')
    parser.add_argument('--workers', type=int, default=24,
                        help='Parallel sim workers (default: 24)')
    parser.add_argument('--ideas', type=str, default='1,2,3',
                        help='Which ideas to run (default: 1,2,3)')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip prediction generation, use existing files')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, only aggregate existing results')
    parser.add_argument('--upload-jupiter', action='store_true',
                        help='Upload predictions to Jupiter and launch remote sweep')
    args = parser.parse_args()

    ideas = [int(x) for x in args.ideas.split(',')]

    log.info("=" * 80)
    log.info("NOVEL SIGNAL PROCESSING SWEEP — 3 Ideas")
    log.info(f"  Ideas:     {ideas}")
    log.info(f"  Workers:   {args.workers}")
    log.info(f"  Binary:    {BINARY}")
    log.info(f"  Preds:     {PRED_FILE}")
    log.info(f"  MBO:       {MBO_DIR}")
    log.info(f"  Pred out:  {PRED_OUT_DIR}")
    log.info(f"  Sim out:   {SIM_OUT_DIR}")
    log.info("=" * 80)

    if not BINARY.exists():
        log.error(f"Binary not found: {BINARY}")
        sys.exit(1)
    if not PRED_FILE.exists():
        log.error(f"Predictions not found: {PRED_FILE}")
        sys.exit(1)

    # Step 1: Generate prediction files
    if args.skip_gen or args.skip_sim:
        log.info("Loading existing prediction files...")
        saved = {}
        for f in PRED_OUT_DIR.glob('*.npz'):
            stem = f.stem
            date = stem[:10]
            label = stem[11:]
            # Filter by idea
            idea_match = False
            for idea_num in ideas:
                if f'idea{idea_num}' in label:
                    idea_match = True
                    break
            if idea_match:
                saved[(date, label)] = str(f)
        log.info(f"Found {len(saved)} existing prediction files")
    else:
        log.info("Loading WF predictions...")
        wf_data = np.load(str(PRED_FILE), allow_pickle=True)
        dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
        log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")

        saved = generate_all_predictions(wf_data, dates, ideas)
        wf_data.close()

    if not saved:
        log.error("No prediction files. Exiting.")
        sys.exit(1)

    # Step 1b: Upload to Jupiter if requested
    if args.upload_jupiter:
        upload_to_jupiter(saved)
        launch_jupiter_sweep()

    # Step 2: Run local sweep
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
                # Parse: {idea_label}_{sim_label}_{date}.json
                # Last part is date
                date = stem[-10:]
                label = stem[:-11]
                results[label][date] = res
            except:
                continue
        results = dict(results)
        log.info(f"Loaded {sum(len(v) for v in results.values())} existing results across {len(results)} configs")

    if not results:
        log.error("No results. Exiting.")
        sys.exit(1)

    # Step 3: Aggregate and report
    summaries = aggregate_and_report(results)

    log.info(f"\nLog: {RESULTS_DIR / f'novel_sweep_{_ts}.log'}")
    log.info("DONE.")
    return summaries


if __name__ == '__main__':
    main()
