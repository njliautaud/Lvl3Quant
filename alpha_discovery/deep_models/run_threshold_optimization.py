#!/usr/bin/env python3
"""
Threshold Optimization Sweep — Fine-grained smooth×smooth
==========================================================
Fine-grained grid search on the WINNING entry×exit pair (smooth_entry × smooth_exit)
from the entry×exit matrix sweep.

Sweeps:
  - Smooth window for ENTRY: 10, 20, 30, 50, 75, 100, 150, 200 bars
  - Smooth window for EXIT:  10, 20, 30, 50, 75, 100, 150, 200 bars
  - Entry threshold:  1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5
  - Exit threshold:   -0.25, 0.0, 0.25, 0.5, 0.75, 1.0 (must be < entry threshold)
  - Vol gates:        50, 60, 70
  - Take-profit:      none, 3, 5, 8, 10, 15, 20 ticks

Strategy:
  PASS 1: Fix exit_window = entry_window → 8×7×6×3×7 = 7,056 combos
  PASS 2: Top 5 entry configs get full exit_window sweep → 5×8×1×1×1×1 = 40 more

Signal logic (hysteresis, causal):
  1. Expanding z-score from raw CNN predictions (CNN_OFFSET=19)
  2. entry_smooth = rolling mean of z-score (entry_window)
  3. exit_smooth  = rolling mean of z-score (exit_window)
  4. Signal activates when entry_smooth > entry_threshold (or < -entry_threshold)
  5. Signal stays active while exit_smooth > exit_threshold (or < -exit_threshold)
  6. Signal deactivates when exit_smooth drops below exit_threshold
  7. Apply vol gate + time mask

Execution:
  Hold 60min safety, chase 1t/3r
  Neptune: 24 workers
  Saturn (via Jupiter hop): 40 workers

Usage:
    python alpha_discovery/deep_models/run_threshold_optimization.py --workers 24
    python alpha_discovery/deep_models/run_threshold_optimization.py --workers 24 --skip-gen
    python alpha_discovery/deep_models/run_threshold_optimization.py --skip-sim
    python alpha_discovery/deep_models/run_threshold_optimization.py --upload-saturn
    python alpha_discovery/deep_models/run_threshold_optimization.py --pass2 --top-n 5
"""

import os
import sys
import gc
import json
import time
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
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_threshold_opt_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_threshold_opt_results'
for d in [PRED_OUT_DIR, SIM_OUT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50
MAX_HOLD_MS = 3600000   # 60 min safety net

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('threshold_opt')
log.setLevel(logging.INFO)
for h in [logging.FileHandler(str(RESULTS_DIR / f'threshold_opt_{_ts}.log'), mode='w', encoding='utf-8'),
          logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    log.addHandler(h)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── Sweep parameter grids ──
ENTRY_WINDOWS = [10, 20, 30, 50, 75, 100, 150, 200]
EXIT_WINDOWS  = [10, 20, 30, 50, 75, 100, 150, 200]
ENTRY_THRESHOLDS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
EXIT_THRESHOLDS  = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
VOL_GATES = [50, 60, 70]
TP_TICKS_LIST = [None, 3, 5, 8, 10, 15, 20]

# Sim variants: (tp_ticks, label_suffix)
SIM_VARIANTS = [(tp, f'tp{tp}' if tp is not None else 'hold') for tp in TP_TICKS_LIST]


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


def rolling_mean_smooth(z_scores, window=50):
    """Causal rolling mean of z-scores."""
    import pandas as pd
    return pd.Series(z_scores).rolling(window, min_periods=1).mean().values


def time_mask(n_bars):
    """Skip first 30 min and last 15 min of RTH session (6.5 hrs = 390 min)."""
    secs = np.arange(n_bars) / BARS_PER_SEC
    mins = secs / 60.0
    return (mins >= 30) & (mins < 375)


# ══════════════════════════════════════════════════════════════════════════════
# HYSTERESIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _apply_hysteresis_python(z_score, entry_long, entry_short, exit_long, exit_short):
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


# ══════════════════════════════════════════════════════════════════════════════
# COMBO BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_pass1_combos():
    """
    PASS 1: exit_window = entry_window (same smooth for both).
    8 windows × 7 entry_thr × 6 exit_thr × 3 vol_gates = 1,008 prediction combos
    Each gets 7 TP variants in sim = 7,056 total sim jobs per day.
    But exit_thr must be < entry_thr, so actual count is less.
    """
    combos = []
    for ew in ENTRY_WINDOWS:
        for entry_thr in ENTRY_THRESHOLDS:
            for exit_thr in EXIT_THRESHOLDS:
                if exit_thr >= entry_thr:
                    continue  # exit must be < entry
                for vg in VOL_GATES:
                    combos.append({
                        'entry_window': ew,
                        'exit_window': ew,  # same as entry for pass 1
                        'entry_threshold': entry_thr,
                        'exit_threshold': exit_thr,
                        'vol_gate': vg,
                        'label': f'ew{ew}_xw{ew}_et{entry_thr}_xt{exit_thr}_vol{vg}',
                    })
    log.info(f"Pass 1: {len(combos)} prediction combos "
             f"(× {len(SIM_VARIANTS)} TP variants = {len(combos) * len(SIM_VARIANTS)} sim jobs/day)")
    return combos


def build_pass2_combos(top_configs):
    """
    PASS 2: For each top config from pass 1, sweep all exit_windows
    that differ from the entry_window.
    """
    combos = []
    for cfg in top_configs:
        ew = cfg['entry_window']
        entry_thr = cfg['entry_threshold']
        exit_thr = cfg['exit_threshold']
        vg = cfg['vol_gate']
        for xw in EXIT_WINDOWS:
            if xw == ew:
                continue  # already tested in pass 1
            combos.append({
                'entry_window': ew,
                'exit_window': xw,
                'entry_threshold': entry_thr,
                'exit_threshold': exit_thr,
                'vol_gate': vg,
                'label': f'ew{ew}_xw{xw}_et{entry_thr}_xt{exit_thr}_vol{vg}',
            })
    log.info(f"Pass 2: {len(combos)} additional combos "
             f"(× {len(SIM_VARIANTS)} TP variants = {len(combos) * len(SIM_VARIANTS)} sim jobs/day)")
    return combos


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION FILE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_predictions_for_date(date, wf_data, combos):
    """Generate all smooth×smooth prediction files for one date."""
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

    # ── Pre-compute base signals (once per date) ──

    # 1. Align CNN offset
    aligned = np.zeros(n_bars, dtype=np.float64)
    end = min(n_bars, len(preds_raw) + CNN_OFFSET)
    aligned[CNN_OFFSET:end] = preds_raw[:end - CNN_OFFSET]

    # 2. Expanding z-score (base signal)
    z_score = zscore_expanding(aligned)

    # 3. Pre-compute all smoothed z-scores for each window (cache across combos)
    smooth_cache = {}
    all_windows = set()
    for combo in combos:
        all_windows.add(combo['entry_window'])
        all_windows.add(combo['exit_window'])
    for w in all_windows:
        smooth_cache[w] = rolling_mean_smooth(z_score, window=w)

    # 4. Vol
    vol = compute_trailing_vol(mid)
    vol_pct = {}
    for vg in set(c['vol_gate'] for c in combos):
        vol_pct[vg] = compute_expanding_vol_percentile(vol, vg)

    # 5. Time mask
    tmask = time_mask(n_bars)

    saved = {}

    for combo in combos:
        ew = combo['entry_window']
        xw = combo['exit_window']
        entry_thr = combo['entry_threshold']
        exit_thr = combo['exit_threshold']
        vg = combo['vol_gate']
        label = combo['label']

        # Get smoothed signals
        entry_smooth = smooth_cache[ew]
        exit_smooth = smooth_cache[xw]

        # Entry conditions: entry_smooth exceeds entry_threshold
        entry_long = entry_smooth > entry_thr
        entry_short = entry_smooth < -entry_thr

        # Exit conditions: exit_smooth drops below exit_threshold
        exit_long = exit_smooth < exit_thr
        exit_short = exit_smooth > -exit_thr

        # Apply hysteresis
        signal = apply_hysteresis(
            z_score,
            entry_long.astype(np.bool_),
            entry_short.astype(np.bool_),
            exit_long.astype(np.bool_),
            exit_short.astype(np.bool_),
        )

        # Vol gate
        if vg > 0 and vg in vol_pct:
            vt = vol_pct[vg]
            vol_mask = np.isnan(vol) | (vol < vt)
            signal[vol_mask] = 0.0

        # Time mask
        signal[~tmask] = 0.0

        signal = np.nan_to_num(signal, nan=0.0).astype(np.float32)

        # Save
        fpath = PRED_OUT_DIR / f'{date}_{label}.npz'
        np.savez_compressed(str(fpath), predictions=signal)
        saved[(date, label)] = str(fpath)

    # Cleanup
    del mid, preds_raw, aligned, z_score, vol, smooth_cache
    gc.collect()

    return saved


def generate_all_predictions(wf_data, dates, combos):
    """Generate prediction files for all dates and combos."""
    all_saved = {}
    for di, date in enumerate(dates):
        day_saved = generate_predictions_for_date(date, wf_data, combos)
        all_saved.update(day_saved)
        if (di + 1) % 5 == 0 or di == 0 or di == len(dates) - 1:
            log.info(f"  Generated {di+1}/{len(dates)} days "
                     f"({len(all_saved)} files total, {len(day_saved)} this day)")
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
    except Exception:
        pass
    return None


def run_sweep(saved_files, workers=24):
    """Run all sim jobs (7 TP variants per combo) in parallel."""
    jobs = []

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
                try:
                    with open(out_file) as f:
                        existing = json.load(f)
                    jobs.append(('cached', full_label, date, existing))
                    continue
                except Exception:
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

            if completed % 200 == 0 or completed == len(to_run):
                el = time.time() - t0
                rate = completed / el if el > 0 else 0
                eta = (len(to_run) - completed) / rate / 60 if rate > 0 else 0
                log.info(f"  [{completed}/{len(to_run)}] {rate:.1f} jobs/s, ETA {eta:.1f}min")

    log.info(f"Simulation done: {completed} jobs in {time.time()-t0:.0f}s")
    return dict(results)


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATION & REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def parse_config_label(config_label):
    """
    Parse label like 'ew50_xw50_et2.0_xt0.5_vol70_tp8' into components.
    Returns dict of parsed values.
    """
    parts = config_label.split('_')
    result = {}
    for p in parts:
        if p.startswith('ew'):
            result['entry_window'] = int(p[2:])
        elif p.startswith('xw'):
            result['exit_window'] = int(p[2:])
        elif p.startswith('et'):
            result['entry_threshold'] = float(p[2:])
        elif p.startswith('xt'):
            result['exit_threshold'] = float(p[2:])
        elif p.startswith('vol'):
            result['vol_gate'] = int(p[3:])
        elif p.startswith('tp'):
            result['tp'] = int(p[2:])
        elif p == 'hold':
            result['tp'] = None
    return result


def aggregate_and_report(results):
    """Aggregate per-day results, print ranked report."""
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

        parsed = parse_config_label(config_label)

        summaries.append({
            'config': config_label,
            'entry_window': parsed.get('entry_window', 0),
            'exit_window': parsed.get('exit_window', 0),
            'entry_threshold': parsed.get('entry_threshold', 0),
            'exit_threshold': parsed.get('exit_threshold', 0),
            'vol_gate': parsed.get('vol_gate', 0),
            'tp': parsed.get('tp'),
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

    # ── Top 50 by Sharpe ──
    log.info("\n" + "=" * 160)
    log.info("THRESHOLD OPTIMIZATION SWEEP — TOP 50 BY SHARPE (Rust MBO Fill Sim)")
    log.info("=" * 160)
    log.info(f"{'#':>3} {'EW':>4} {'XW':>4} {'EThr':>5} {'XThr':>5} {'Vol':>4} {'TP':>4} "
             f"{'Sharpe':>7} {'P&L':>10} {'Trades':>6} {'Fill%':>6} "
             f"{'WR%':>5} {'MaxDD':>8} {'Annual':>10}")
    log.info("-" * 160)

    for i, s in enumerate(summaries[:50]):
        tp_str = str(s['tp']) if s['tp'] is not None else '---'
        log.info(
            f"{i+1:>3} {s['entry_window']:>4} {s['exit_window']:>4} "
            f"{s['entry_threshold']:>5.2f} {s['exit_threshold']:>5.2f} "
            f"{s['vol_gate']:>4} {tp_str:>4} "
            f"{s['sharpe']:>7.2f} ${s['total_pnl']:>9,.0f} {s['n_trades']:>6} "
            f"{s['fill_rate']*100:>5.1f}% {s['win_rate']*100:>4.1f}% "
            f"${s['max_dd']:>7,.0f} ${s['annualized']:>9,.0f}"
        )

    # ── Parameter stability analysis ──
    log.info("\n" + "=" * 120)
    log.info("PARAMETER STABILITY ANALYSIS")
    log.info("=" * 120)

    # Group by each parameter, show avg Sharpe
    for param_name, param_key in [
        ('Entry Window', 'entry_window'),
        ('Exit Window', 'exit_window'),
        ('Entry Threshold', 'entry_threshold'),
        ('Exit Threshold', 'exit_threshold'),
        ('Vol Gate', 'vol_gate'),
        ('Take Profit', 'tp'),
    ]:
        param_groups = defaultdict(list)
        for s in summaries:
            param_groups[s[param_key]].append(s['sharpe'])

        log.info(f"\n  {param_name}:")
        for val in sorted(param_groups.keys(), key=lambda x: (x is None, x)):
            sharpes = param_groups[val]
            avg_s = np.mean(sharpes)
            pct_pos = np.mean([s > 0 for s in sharpes]) * 100
            max_s = max(sharpes)
            val_str = str(val) if val is not None else 'none'
            log.info(f"    {val_str:>6}: avg Sharpe {avg_s:+.2f}, "
                     f"max {max_s:.2f}, {pct_pos:.0f}% positive ({len(sharpes)} configs)")

    # ── Nearby-config stability for top 10 ──
    log.info("\n" + "=" * 120)
    log.info("NEIGHBOR STABILITY — Top 10 configs (nearby combos should also profit)")
    log.info("=" * 120)

    config_map = {s['config']: s for s in summaries}
    for i, s in enumerate(summaries[:10]):
        neighbors = []
        for other in summaries:
            if other['config'] == s['config']:
                continue
            # "Neighbor" = differs by at most 1 parameter step
            diffs = 0
            if other['entry_window'] != s['entry_window']:
                diffs += 1
            if other['exit_window'] != s['exit_window']:
                diffs += 1
            if abs(other['entry_threshold'] - s['entry_threshold']) > 0.26:
                diffs += 1
            if abs(other['exit_threshold'] - s['exit_threshold']) > 0.26:
                diffs += 1
            if other['vol_gate'] != s['vol_gate']:
                diffs += 1
            if other['tp'] != s['tp']:
                diffs += 1
            if diffs <= 1:
                neighbors.append(other)

        if neighbors:
            neighbor_sharpes = [n['sharpe'] for n in neighbors]
            avg_neighbor = np.mean(neighbor_sharpes)
            pct_neighbor_pos = np.mean([ns > 0 for ns in neighbor_sharpes]) * 100
            log.info(f"  #{i+1} {s['config']} -> Sharpe {s['sharpe']:.2f}")
            log.info(f"       {len(neighbors)} neighbors: avg Sharpe {avg_neighbor:.2f}, "
                     f"{pct_neighbor_pos:.0f}% profitable")
        else:
            log.info(f"  #{i+1} {s['config']} -> Sharpe {s['sharpe']:.2f} (no neighbors found)")

    # ── Reference ──
    log.info("\n" + "=" * 100)
    log.info("REFERENCE — Baseline:")
    log.info("  IS best (vol70/conv2.5/1t/3r/30min): Sharpe 3.28, +$15,479/74d, 130 trades")
    log.info("  OOT static:                          Sharpe 1.58, +$4,082/68d, 102 trades")
    log.info("  OOT best chase:                      Sharpe 2.33, +$3,264/68d, 14 trades")
    log.info("=" * 100)

    # Save JSON
    out_file = RESULTS_DIR / f'threshold_opt_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'n_configs': len(summaries),
            'top_50': summaries[:50],
            'all_summaries': summaries,
        }, f, indent=2)
    log.info(f"\nResults saved: {out_file}")

    return summaries


def get_top_configs(summaries, top_n=5):
    """Extract top N unique (entry_window, entry_thr, exit_thr, vol_gate) configs."""
    seen = set()
    top = []
    for s in summaries:
        key = (s['entry_window'], s['entry_threshold'], s['exit_threshold'], s['vol_gate'])
        if key not in seen:
            seen.add(key)
            top.append({
                'entry_window': s['entry_window'],
                'entry_threshold': s['entry_threshold'],
                'exit_threshold': s['exit_threshold'],
                'vol_gate': s['vol_gate'],
            })
            if len(top) >= top_n:
                break
    return top


# ══════════════════════════════════════════════════════════════════════════════
# SATURN UPLOAD (via Jupiter hop)
# ══════════════════════════════════════════════════════════════════════════════

def upload_to_saturn(saved_files):
    """Upload prediction NPZs to Saturn via Jupiter hop."""
    import paramiko

    JUPITER_HOST = 'jupiter'
    JUPITER_USER = 'jupiter'
    JUPITER_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
    SATURN_USER = 'saturn'
    SATURN_HOST = 'saturn'
    SATURN_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
    REMOTE_PRED_DIR = '/home/saturn/Lvl3Quant/data/processed/cnn_wf_threshold_opt_predictions'
    REMOTE_RESULT_DIR = '/home/saturn/Lvl3Quant/data/processed/cnn_wf_threshold_opt_results'
    JUPITER_STAGING = '/home/jupiter/threshold_opt_staging'

    log.info(f"Uploading {len(saved_files)} prediction files to Saturn via Jupiter...")

    try:
        # Step 1: Upload to Jupiter staging
        ssh_jup = paramiko.SSHClient()
        ssh_jup.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_jup.connect(JUPITER_HOST, username=JUPITER_USER, password=JUPITER_PW, timeout=10)
        ssh_jup.exec_command(f'mkdir -p {JUPITER_STAGING}')
        time.sleep(1)

        sftp_jup = ssh_jup.open_sftp()
        uploaded = 0
        for (date, label), local_path in saved_files.items():
            remote_path = f'{JUPITER_STAGING}/{Path(local_path).name}'
            try:
                sftp_jup.put(local_path, remote_path)
                uploaded += 1
                if uploaded % 200 == 0:
                    log.info(f"  Uploaded to Jupiter: {uploaded}/{len(saved_files)}")
            except Exception as e:
                log.warning(f"  Upload failed {Path(local_path).name}: {e}")
        sftp_jup.close()
        log.info(f"Uploaded {uploaded} files to Jupiter staging")

        # Step 2: rsync from Jupiter to Saturn
        rsync_cmd = (
            f'rsync -az --progress {JUPITER_STAGING}/ '
            f'{SATURN_USER}@{SATURN_HOST}:{REMOTE_PRED_DIR}/'
        )
        log.info(f"Rsyncing to Saturn: {rsync_cmd}")

        # Need to use sshpass for Saturn password in rsync
        full_cmd = (
            f'sshpass -p "{SATURN_PW}" {rsync_cmd} && '
            f'sshpass -p "{SATURN_PW}" ssh {SATURN_USER}@{SATURN_HOST} '
            f'"mkdir -p {REMOTE_RESULT_DIR}"'
        )
        stdin, stdout, stderr = ssh_jup.exec_command(full_cmd, timeout=600)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            log.info("Rsync to Saturn completed successfully")
        else:
            err = stderr.read().decode()
            log.warning(f"Rsync exit code {exit_status}: {err}")
            # Fallback: try with SSH key or expect
            log.info("Trying rsync without sshpass...")
            fallback_cmd = (
                f'rsync -az -e "ssh -o StrictHostKeyChecking=no" '
                f'{JUPITER_STAGING}/ {SATURN_USER}@{SATURN_HOST}:{REMOTE_PRED_DIR}/'
            )
            stdin2, stdout2, stderr2 = ssh_jup.exec_command(fallback_cmd, timeout=600)
            exit2 = stdout2.channel.recv_exit_status()
            if exit2 == 0:
                log.info("Rsync fallback succeeded")
            else:
                log.error(f"Rsync fallback also failed: {stderr2.read().decode()}")

        # Cleanup staging
        ssh_jup.exec_command(f'rm -rf {JUPITER_STAGING}')
        ssh_jup.close()
        return True
    except Exception as e:
        log.error(f"Saturn upload failed: {e}")
        return False


def launch_saturn_sweep():
    """Launch sweep on Saturn (via Jupiter hop) with 40 workers."""
    import paramiko

    JUPITER_HOST = 'jupiter'
    JUPITER_USER = 'jupiter'
    JUPITER_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")
    SATURN_USER = 'saturn'
    SATURN_HOST = 'saturn'
    SATURN_PW = os.environ.get("CLUSTER_SSH_PASSWORD", "")

    remote_script = '''#!/usr/bin/env python3
"""Remote threshold optimization sweep on Saturn."""
import sys, json, time, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path("/home/saturn/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_threshold_opt_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_threshold_opt_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TP_VARIANTS = [
    (None,  "hold"),
    (3,     "tp3"),
    (5,     "tp5"),
    (8,     "tp8"),
    (10,    "tp10"),
    (15,    "tp15"),
    (20,    "tp20"),
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
    for tp, sim_suffix in TP_VARIANTS:
        full_label = f"{combo_label}_{sim_suffix}"
        out_file = OUT_DIR / f"{full_label}_{date}.json"
        if out_file.exists():
            continue
        jobs.append((str(mbo), str(pf), str(out_file), tp))

print(f"Jobs to run: {len(jobs)}")
completed = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=40) as executor:
    futures = {executor.submit(run_sim, *j): j for j in jobs}
    for f in as_completed(futures):
        completed += 1
        if completed % 100 == 0 or completed == len(jobs):
            el = time.time() - t0
            rate = completed / el if el > 0 else 0
            eta = (len(jobs) - completed) / rate / 60 if rate > 0 else 0
            print(f"  [{completed}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.1f}min")

print(f"Done: {completed} jobs in {time.time()-t0:.0f}s")
'''

    try:
        ssh_jup = paramiko.SSHClient()
        ssh_jup.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_jup.connect(JUPITER_HOST, username=JUPITER_USER, password=JUPITER_PW, timeout=10)

        # Write script to Jupiter, then scp to Saturn
        jup_script = '/home/jupiter/run_saturn_threshold_sweep.py'
        sftp = ssh_jup.open_sftp()
        with sftp.open(jup_script, 'w') as f:
            f.write(remote_script)
        sftp.close()

        # Copy script to Saturn and launch
        copy_cmd = (
            f'sshpass -p "{SATURN_PW}" scp {jup_script} '
            f'{SATURN_USER}@{SATURN_HOST}:/home/saturn/Lvl3Quant/run_threshold_sweep.py'
        )
        launch_cmd = (
            f'sshpass -p "{SATURN_PW}" ssh {SATURN_USER}@{SATURN_HOST} '
            f'"cd /home/saturn/Lvl3Quant && '
            f'nohup python3 run_threshold_sweep.py > threshold_sweep_saturn.log 2>&1 &"'
        )

        log.info("Copying sweep script to Saturn...")
        stdin, stdout, stderr = ssh_jup.exec_command(copy_cmd, timeout=30)
        stdout.channel.recv_exit_status()

        log.info("Launching sweep on Saturn with 40 workers...")
        stdin, stdout, stderr = ssh_jup.exec_command(launch_cmd, timeout=30)
        stdout.channel.recv_exit_status()
        time.sleep(2)

        log.info("Saturn sweep launched!")
        log.info("Monitor: ssh jupiter -> ssh saturn 'tail -f /home/saturn/Lvl3Quant/threshold_sweep_saturn.log'")

        ssh_jup.close()
        return True
    except Exception as e:
        log.error(f"Saturn launch failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Threshold Optimization Sweep (smooth x smooth)')
    parser.add_argument('--workers', type=int, default=24,
                        help='Parallel sim workers (default: 24)')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip prediction generation, use existing files')
    parser.add_argument('--skip-sim', action='store_true',
                        help='Skip simulation, only aggregate existing results')
    parser.add_argument('--upload-saturn', action='store_true',
                        help='Upload predictions to Saturn (via Jupiter) and launch remote sweep')
    parser.add_argument('--pass2', action='store_true',
                        help='Run pass 2: sweep exit windows for top configs from pass 1')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top configs for pass 2 (default: 5)')
    parser.add_argument('--pass1-results', type=str, default=None,
                        help='Path to pass 1 results JSON for pass 2 (auto-detected if not given)')
    args = parser.parse_args()

    # Determine which combos to build
    if args.pass2:
        # Load pass 1 results
        if args.pass1_results:
            p1_file = Path(args.pass1_results)
        else:
            # Find latest pass 1 results
            p1_files = sorted(RESULTS_DIR.glob('threshold_opt_results_*.json'))
            if not p1_files:
                log.error("No pass 1 results found. Run pass 1 first.")
                sys.exit(1)
            p1_file = p1_files[-1]

        log.info(f"Loading pass 1 results: {p1_file}")
        with open(p1_file) as f:
            p1_data = json.load(f)

        top_cfgs = get_top_configs(p1_data.get('all_summaries', p1_data.get('top_50', [])),
                                   top_n=args.top_n)
        if not top_cfgs:
            log.error("No top configs extracted from pass 1 results.")
            sys.exit(1)

        log.info(f"Top {len(top_cfgs)} configs from pass 1:")
        for cfg in top_cfgs:
            log.info(f"  ew={cfg['entry_window']}, et={cfg['entry_threshold']}, "
                     f"xt={cfg['exit_threshold']}, vol={cfg['vol_gate']}")

        combos = build_pass2_combos(top_cfgs)
    else:
        combos = build_pass1_combos()

    log.info("=" * 100)
    log.info("THRESHOLD OPTIMIZATION SWEEP — Smooth Entry x Smooth Exit")
    log.info(f"  Pass:        {'2 (exit window sweep)' if args.pass2 else '1 (entry_window = exit_window)'}")
    log.info(f"  Combos:      {len(combos)} prediction configs")
    log.info(f"  TP variants: {len(SIM_VARIANTS)} ({', '.join(s[1] for s in SIM_VARIANTS)})")
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  Binary:      {BINARY}")
    log.info(f"  Predictions: {PRED_FILE}")
    log.info(f"  MBO:         {MBO_DIR}")
    log.info(f"  Pred out:    {PRED_OUT_DIR}")
    log.info(f"  Sim out:     {SIM_OUT_DIR}")
    log.info("=" * 100)

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

    # Step 1b: Upload to Saturn if requested
    if args.upload_saturn:
        upload_to_saturn(saved)
        launch_saturn_sweep()

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
                # Last 10 chars = date
                date = stem[-10:]
                label = stem[:-11]
                results[label][date] = res
            except Exception:
                continue
        results = dict(results)
        log.info(f"Loaded {sum(len(v) for v in results.values())} results "
                 f"across {len(results)} configs")

    if not results:
        log.error("No results. Exiting.")
        sys.exit(1)

    # Step 3: Aggregate and report
    summaries = aggregate_and_report(results)

    log.info(f"\nLog: {RESULTS_DIR / f'threshold_opt_{_ts}.log'}")
    log.info("DONE.")
    return summaries


if __name__ == '__main__':
    main()
