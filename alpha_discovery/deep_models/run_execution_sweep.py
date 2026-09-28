#!/usr/bin/env python3
"""
Execution Optimization Sweep — Signal Reversal + Stops + Conviction Filters
============================================================================
Tests the signal-flip-exit flag plus take-profit, trailing stops, and
conviction thresholds on the WF OOT predictions using the Rust MBO fill sim.

This is the REAL validation — not Python approximation.

Usage:
    python alpha_discovery/deep_models/run_execution_sweep.py
    python alpha_discovery/deep_models/run_execution_sweep.py --workers 8
"""

import sys
import json
import time
import bisect
import logging
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_exec_sweep_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_exec_sweep_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19
BARS_PER_SEC = 10
TICK_VALUE = 12.50

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('exec_sweep')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'exec_sweep_{_ts}.log'), mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── SWEEP CONFIGS ──
# Each config: (threshold, hold_ms, latency_ms, chase_ticks, chase_reprices,
#               signal_flip_exit, trailing_ticks, take_profit_ticks, label)
EXEC_CONFIGS = []

# Baseline (no flip, no stops)
for conv in [1.5, 2.0, 2.5]:
    EXEC_CONFIGS.append((conv, 1800000, 0, 1, 3, False, None, None,
                         f'baseline_conv{int(conv*10)}_30min'))

# Signal-flip-exit (THE BIG ONE)
for conv in [1.5, 2.0, 2.5, 3.0]:
    EXEC_CONFIGS.append((conv, 1800000, 0, 1, 3, True, None, None,
                         f'flip_conv{int(conv*10)}_30min'))

# Signal-flip + trailing stop
for conv in [2.0, 2.5]:
    for trail in [15, 20, 25]:
        EXEC_CONFIGS.append((conv, 1800000, 0, 1, 3, True, trail, None,
                             f'flip_trail{trail}_conv{int(conv*10)}_30min'))

# Signal-flip + take-profit
for conv in [2.0, 2.5]:
    for tp in [10, 15, 20]:
        EXEC_CONFIGS.append((conv, 1800000, 0, 1, 3, True, None, tp,
                             f'flip_tp{tp}_conv{int(conv*10)}_30min'))

# Signal-flip + trailing stop + take-profit combo
for conv in [2.0, 2.5]:
    for trail, tp in [(20, 15), (15, 10), (25, 20)]:
        EXEC_CONFIGS.append((conv, 1800000, 0, 1, 3, True, trail, tp,
                             f'flip_trail{trail}_tp{tp}_conv{int(conv*10)}_30min'))

# No flip, but with stops (to compare)
for conv in [2.0, 2.5]:
    for trail in [15, 20, 25]:
        EXEC_CONFIGS.append((conv, 1800000, 0, 1, 3, False, trail, None,
                             f'noflip_trail{trail}_conv{int(conv*10)}_30min'))

# Passive entry (no chase) with flip
for conv in [2.0, 2.5]:
    EXEC_CONFIGS.append((conv, 1800000, 0, 0, 0, True, None, None,
                         f'flip_passive_conv{int(conv*10)}_30min'))

# 2t/5r chase with flip
for conv in [2.0, 2.5]:
    EXEC_CONFIGS.append((conv, 1800000, 0, 2, 5, True, None, None,
                         f'flip_chase2t5r_conv{int(conv*10)}_30min'))

VOL_GATES = [50, 70, 80]
TIME_FILTERS = ['morning_afternoon']


# ── Signal Processing (same as run_wf_fill_sim.py) ──

def compute_trailing_vol(mid, window=3000):
    ret_1s = np.zeros(len(mid))
    ret_1s[10:] = (mid[10:] - mid[:-10]) / mid[:-10] * 10000
    vol = np.full(len(mid), np.nan)
    cumsum = np.cumsum(ret_1s)
    cumsum2 = np.cumsum(ret_1s ** 2)
    for i in range(window, len(mid)):
        s = cumsum[i] - cumsum[i - window]
        s2 = cumsum2[i] - cumsum2[i - window]
        mean = s / window
        var = s2 / window - mean ** 2
        vol[i] = np.sqrt(max(var, 0))
    return vol


def precompute_vol_percentiles(vol_pred, percentiles=(50, 60, 70, 80, 90)):
    n = len(vol_pred)
    result = {p: np.full(n, -np.inf) for p in percentiles}
    sorted_vals = []
    for i in range(n):
        if not np.isnan(vol_pred[i]):
            bisect.insort(sorted_vals, vol_pred[i])
        if len(sorted_vals) >= 100:
            for p in percentiles:
                idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
                result[p][i] = sorted_vals[idx]
    return result


def zscore_expanding(arr):
    result = np.full_like(arr, np.nan, dtype=np.float64)
    running_sum = running_sq = 0.0
    count = 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v): continue
        running_sum += v
        running_sq += v * v
        count += 1
        if count >= 50:
            mean = running_sum / count
            var = (running_sq / count) - mean * mean
            std = max(np.sqrt(var), 1e-8)
            result[i] = (v - mean) / std
    return result


def compute_time_features(n_bars):
    seconds = np.arange(n_bars) / BARS_PER_SEC
    minutes = seconds / 60.0
    return (minutes >= 30) & (minutes < 360)  # Skip first 30min, last 30min


# ── Prediction Preparation ──

def prepare_predictions(pred_file, vol_gates, time_filters):
    log.info(f"Loading predictions from {pred_file}...")
    wf_data = np.load(str(pred_file), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
    log.info(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    saved_files = {}
    for date in dates:
        preds_raw = wf_data[f'{date}_preds']
        mid = wf_data[f'{date}_mid']
        n = len(preds_raw)
        if n < 5000: continue

        # Align CNN offset
        cp_aligned = np.zeros(n, dtype=np.float64)
        end_idx = min(n, len(preds_raw) + CNN_OFFSET)
        cp_aligned[CNN_OFFSET:end_idx] = preds_raw[:end_idx - CNN_OFFSET]

        # Expanding z-score
        signal = zscore_expanding(cp_aligned)

        # Time mask
        time_mask = compute_time_features(n)

        # Vol
        vol_pred = compute_trailing_vol(mid)
        vol_thresholds = precompute_vol_percentiles(vol_pred, tuple(vol_gates))

        for vg in vol_gates:
            for tf in time_filters:
                sig = signal.copy()
                # Apply vol gate
                for i in range(len(sig)):
                    if np.isnan(sig[i]): continue
                    if np.isnan(vol_pred[i]) or vol_pred[i] < vol_thresholds[vg][i]:
                        sig[i] = 0.0
                # Apply time filter
                sig[~time_mask] = 0.0
                # Replace NaN with 0
                sig = np.nan_to_num(sig, nan=0.0)

                fname = f'{date}_vol{vg}_{tf}.npz'
                fpath = PRED_OUT_DIR / fname
                np.savez_compressed(str(fpath), predictions=sig.astype(np.float32))
                saved_files[(date, vg, tf)] = str(fpath)

    log.info(f"Saved {len(saved_files)} prediction files")
    return saved_files


# ── Simulation Runner ──

def run_single_sim(mbo_file, pred_file, output_file, config):
    """Run a single fill_sim_cli invocation."""
    threshold, hold_ms, lat_ms, chase_t, chase_r, flip_exit, trail, tp, label = config

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(output_file),
        '--hold-ms', str(hold_ms),
        '--signal-threshold', str(threshold),
        '--latency-ms', str(lat_ms),
        '--quiet',
    ]

    if chase_t > 0:
        cmd += ['--chase-entry',
                '--chase-max-ticks', str(chase_t),
                '--chase-max-reprices', str(chase_r)]

    if flip_exit:
        cmd.append('--signal-flip-exit')

    if trail is not None:
        cmd += ['--trailing-ticks', str(trail)]

    if tp is not None:
        cmd += ['--take-profit-ticks', str(tp)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        if Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        log.error(f"Sim failed: {label} — {e}")
    return None


def run_sweep(saved_files, configs, workers=6):
    """Run the full sweep across all dates x configs."""
    jobs = []
    for (date, vg, tf), pred_file in saved_files.items():
        # Find MBO file
        date_compact = date.replace('-', '')
        mbo_candidates = list(MBO_DIR.glob(f'*{date_compact}*.dbn.zst'))
        if not mbo_candidates:
            mbo_candidates = list(MBO_DIR.glob(f'*{date_compact}*.dbn'))
        if not mbo_candidates:
            continue
        mbo_file = mbo_candidates[0]

        for config in configs:
            label = config[-1]
            out_file = SIM_OUT_DIR / f'vol{vg}_{tf}_{label}_{date}.json'
            jobs.append((mbo_file, pred_file, str(out_file), config, date, vg, tf))

    log.info(f"Total jobs: {len(jobs)} ({len(saved_files)} dates x {len(configs)} configs)")
    log.info(f"Running with {workers} workers...")

    completed = 0
    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for mbo, pred, out, cfg, date, vg, tf in jobs:
            f = executor.submit(run_single_sim, mbo, pred, out, cfg)
            futures[f] = (cfg[-1], date, vg, tf)

        for future in as_completed(futures):
            label, date, vg, tf = futures[future]
            completed += 1
            res = future.result()
            if res:
                results.append({
                    'config': label,
                    'date': date,
                    'vol_gate': vg,
                    'time_filter': tf,
                    **{k: v for k, v in res.items() if k != 'trades'},  # Skip per-trade detail for summary
                    'n_trades_detail': len(res.get('trades', [])),
                })

            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (len(jobs) - completed) / rate if rate > 0 else 0
                log.info(f"  Progress: {completed}/{len(jobs)} ({rate:.1f}/s, ETA {eta/60:.1f}min)")

    log.info(f"Completed {completed} jobs in {time.time()-t0:.0f}s")
    return results


def aggregate_results(results):
    """Aggregate per-date results into per-config summaries."""
    from collections import defaultdict
    config_data = defaultdict(list)
    for r in results:
        key = (r['config'], r['vol_gate'], r['time_filter'])
        config_data[key].append(r)

    summaries = []
    for (config, vg, tf), day_results in sorted(config_data.items()):
        total_pnl = sum(r.get('total_pnl_dollars', 0) for r in day_results)
        total_trades = sum(r.get('total_trades', 0) for r in day_results)
        total_signals = sum(r.get('total_signals', 0) for r in day_results)
        total_filled = sum(r.get('total_filled', 0) for r in day_results)
        n_days = len(day_results)
        daily_pnls = [r.get('total_pnl_dollars', 0) for r in day_results]

        avg_daily = np.mean(daily_pnls) if daily_pnls else 0
        std_daily = np.std(daily_pnls) if len(daily_pnls) > 1 else 1
        sharpe = avg_daily / std_daily * np.sqrt(252) if std_daily > 0 else 0
        win_rate = sum(r.get('win_rate', 0) * r.get('total_trades', 0)
                      for r in day_results) / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)

        summaries.append({
            'config': config,
            'vol_gate': vg,
            'time_filter': tf,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe_daily': round(sharpe, 3),
            'avg_daily_pnl': round(avg_daily, 2),
            'annualized_pnl': round(avg_daily * 252, 0),
        })

    summaries.sort(key=lambda x: x['sharpe_daily'], reverse=True)
    return summaries


# ── Main ──

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--vol-gates', type=str, default='50,70,80')
    args = parser.parse_args()

    vol_gates = [int(x) for x in args.vol_gates.split(',')]

    log.info("=" * 70)
    log.info("EXECUTION OPTIMIZATION SWEEP — Rust MBO Fill Sim")
    log.info(f"Configs: {len(EXEC_CONFIGS)}")
    log.info(f"Vol gates: {vol_gates}")
    log.info(f"Workers: {args.workers}")
    log.info("=" * 70)

    # Step 1: Prepare predictions
    saved = prepare_predictions(PRED_FILE, vol_gates, TIME_FILTERS)

    # Step 2: Run sweep
    results = run_sweep(saved, EXEC_CONFIGS, workers=args.workers)

    # Step 3: Aggregate
    summaries = aggregate_results(results)

    # Step 4: Print ranked results
    log.info("\n" + "=" * 70)
    log.info("RANKED RESULTS (by Sharpe)")
    log.info("=" * 70)
    for i, s in enumerate(summaries[:30]):
        flip = "FLIP" if "flip" in s['config'] else "HOLD"
        log.info(f"  #{i+1:>2d} [{flip:>4s}] vol{s['vol_gate']} {s['config']:>45s} | "
                f"Sharpe {s['sharpe_daily']:>6.2f} | P&L ${s['total_pnl']:>10,.2f} | "
                f"{s['n_trades']:>3d}t | WR {s['win_rate']*100:>5.1f}% | Fill {s['fill_rate']*100:.1f}%")

    # Save
    out_path = RESULTS_DIR / f'exec_sweep_results_{_ts}.json'
    with open(out_path, 'w') as f:
        json.dump({'timestamp': _ts, 'n_configs': len(EXEC_CONFIGS),
                  'vol_gates': vol_gates, 'summaries': summaries}, f, indent=2)
    log.info(f"\nResults saved to {out_path}")
    log.info("DONE.")
