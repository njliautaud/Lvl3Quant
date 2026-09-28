#!/usr/bin/env python3
"""
CNN OOT Predictions → Rust MBO Fill Simulator
==============================================
Standalone OOT validation. Does NOT depend on IS feature cache.
Computes vol/time features from CNN prediction mid prices.

Usage:
    python alpha_discovery/cnn_oot_sim_validation.py
    python alpha_discovery/cnn_oot_sim_validation.py --pred-file path/to/oot_predictions_incremental.npz
    python alpha_discovery/cnn_oot_sim_validation.py --skip-gen --workers 8
"""

import bisect
import gc
import sys
import json
import time
import logging
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

LVL3_ROOT = Path(__file__).resolve().parent.parent

# ── Paths ──
DEFAULT_PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_predictions_incremental.npz'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'  # OOT MBO data location
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_oot_sim_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_oot_sim_results'
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19  # window_size - 1
BARS_PER_SEC = 10
RTH_HOURS = 6.5
TICK_VALUE = 12.50

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = str(RESULTS_DIR / f'cnn_oot_sim_{_ts}.log')

log = logging.getLogger('cnn_oot_sim')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)

import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Sweep configs ──
SIM_CONFIGS = [
    # (threshold, hold_ms, latency_ms, trailing_ticks, label)
    (1.5, 1800000, 0, 0, 'conv15_30min_lat0'),
    (2.0, 1800000, 0, 0, 'conv20_30min_lat0'),
    (2.5, 1800000, 0, 0, 'conv25_30min_lat0'),
    (1.5, 1800000, 10, 0, 'conv15_30min_lat10'),
    (2.0, 1800000, 10, 0, 'conv20_30min_lat10'),
    (2.5, 1800000, 10, 0, 'conv25_30min_lat10'),
    (1.5, 1800000, 25, 0, 'conv15_30min_lat25'),
    (2.5, 1800000, 25, 0, 'conv25_30min_lat25'),
    (1.5, 1800000, 50, 0, 'conv15_30min_lat50'),
    (2.5, 1800000, 50, 0, 'conv25_30min_lat50'),
    (1.5, 1800000, 75, 0, 'conv15_30min_lat75'),
    (2.5, 1800000, 75, 0, 'conv25_30min_lat75'),
    (1.5, 1800000, 100, 0, 'conv15_30min_lat100'),
    (2.5, 1800000, 100, 0, 'conv25_30min_lat100'),
    (1.5, 1800000, 150, 0, 'conv15_30min_lat150'),
    (2.5, 1800000, 150, 0, 'conv25_30min_lat150'),
    (1.0, 1800000, 0, 0, 'conv10_30min_lat0'),
    (1.0, 600000, 0, 0, 'conv10_10min_lat0'),
    (1.5, 600000, 0, 0, 'conv15_10min_lat0'),
    (2.0, 600000, 0, 0, 'conv20_10min_lat0'),
]

VOL_GATES = [50, 70, 80]
TIME_FILTERS = ['morning_afternoon']


def compute_trailing_vol(mid, window=3000):
    """Trailing realized vol (std of 1s returns) over window bars."""
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
    """Expanding-window vol percentile thresholds (no look-ahead)."""
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


def compute_time_features(n_bars):
    """Time-of-day features for a trading day (100ms bars)."""
    seconds = np.arange(n_bars) / BARS_PER_SEC
    minutes = seconds / 60.0
    first_30 = minutes < 30
    last_30 = minutes > (RTH_HOURS * 60 - 30)
    morning_afternoon = (minutes < 120) | ((minutes >= 240) & (minutes < 330))
    return minutes, first_30, last_30, morning_afternoon


def zscore_expanding(arr):
    """Expanding-window z-score (no look-ahead)."""
    result = np.full_like(arr, np.nan)
    running_sum = 0.0
    running_sq = 0.0
    count = 0
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        running_sum += arr[i]
        running_sq += arr[i] ** 2
        count += 1
        if count >= 50:
            mean = running_sum / count
            var = (running_sq / count) - mean ** 2
            std = max(np.sqrt(var), 1e-8)
            result[i] = (arr[i] - mean) / std
    return result


def compute_ic_per_day(pred_data, horizon=100):
    """Compute IC (rank correlation of predictions vs future returns) per day."""
    log.info("\n--- Per-Day IC Analysis ---")
    ics = []
    for key in sorted(pred_data.keys()):
        if not key.endswith('_preds'):
            continue
        date = key.replace('_preds', '')
        preds = pred_data[key]
        mid = pred_data.get(f'{date}_mid')
        if mid is None:
            continue

        n = len(mid)
        # Future return at horizon bars
        fwd_ret = np.full(n, np.nan)
        for i in range(n - horizon):
            if mid[i] > 0:
                fwd_ret[i] = (mid[i + horizon] - mid[i]) / mid[i]

        # Only use bars with valid predictions and returns
        valid = ~np.isnan(preds) & ~np.isnan(fwd_ret) & (preds != 0)
        if valid.sum() < 100:
            log.info(f"  {date}: too few valid bars ({valid.sum()})")
            continue

        ic, pval = stats.spearmanr(preds[valid], fwd_ret[valid])
        ics.append({'date': date, 'ic': ic, 'pval': pval, 'n_valid': int(valid.sum())})
        log.info(f"  {date}: IC={ic:+.4f} (p={pval:.4f}, n={valid.sum():,})")

    if ics:
        ic_vals = [x['ic'] for x in ics]
        mean_ic = np.mean(ic_vals)
        median_ic = np.median(ic_vals)
        std_ic = np.std(ic_vals)
        pct_pos = np.mean([x > 0 for x in ic_vals]) * 100
        t_stat = mean_ic / (std_ic / np.sqrt(len(ic_vals))) if std_ic > 0 else 0

        log.info(f"\n  OOT IC Summary ({len(ics)} days):")
        log.info(f"    Mean IC:      {mean_ic:+.4f}")
        log.info(f"    Median IC:    {median_ic:+.4f}")
        log.info(f"    IC Std:       {std_ic:.4f}")
        log.info(f"    t-stat:       {t_stat:.2f}")
        log.info(f"    Pct positive: {pct_pos:.1f}%")
        log.info(f"    Range:        [{min(ic_vals):.4f}, {max(ic_vals):.4f}]")

    return ics


def prepare_oot_predictions(pred_file):
    """Load OOT CNN predictions, apply de-biasing, save per-day NPZ for Rust sim."""
    log.info(f"Loading OOT predictions from {pred_file}...")
    cnn_data = np.load(str(pred_file), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in cnn_data.files if k.endswith('_preds')))
    log.info(f"OOT dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    # Compute IC first
    pred_dict = {k: cnn_data[k] for k in cnn_data.files}
    ics = compute_ic_per_day(pred_dict, horizon=100)

    saved_files = {}

    for di, date in enumerate(dates):
        nodash = date.replace('-', '')
        mbo_zst = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            log.warning(f"No MBO file for {date}, skipping sim prep")
            continue

        preds = cnn_data[f'{date}_preds'].astype(np.float64)
        mid = cnn_data[f'{date}_mid'].astype(np.float64)
        n_bars = len(mid)

        # Align predictions (zero-pad front if needed)
        cp_aligned = np.full(n_bars, 0.0)
        end_idx = min(CNN_OFFSET + len(preds), n_bars)
        cp_aligned[CNN_OFFSET:end_idx] = preds[:end_idx - CNN_OFFSET]

        # Expanding z-score
        signal = zscore_expanding(cp_aligned)

        # Compute vol from mid prices
        vol_pred = compute_trailing_vol(mid)
        vol_pct_thresholds = precompute_vol_percentiles(vol_pred)

        # Time features
        minutes, first_30, last_30, morning_afternoon = compute_time_features(n_bars)

        for vol_gate in VOL_GATES:
            for tf in TIME_FILTERS:
                filtered_signal = signal.copy()

                # Vol gate
                if vol_gate > 0:
                    available = sorted(vol_pct_thresholds.keys())
                    closest = min(available, key=lambda x: abs(x - vol_gate))
                    vol_threshold = vol_pct_thresholds[closest]
                    for i in range(len(filtered_signal)):
                        if np.isnan(vol_pred[i]) or vol_pred[i] < vol_threshold[i]:
                            filtered_signal[i] = 0.0

                # Time filter
                if tf == 'morning_afternoon':
                    time_ok = morning_afternoon
                else:
                    time_ok = np.ones(n_bars, dtype=bool)
                filtered_signal[~time_ok] = 0.0

                filtered_signal = np.nan_to_num(filtered_signal, nan=0.0)

                key = (date, vol_gate, tf)
                out_file = PRED_OUT_DIR / f'{date}_vol{vol_gate}_{tf}.npz'
                np.savez_compressed(str(out_file), predictions=filtered_signal)
                saved_files[key] = out_file

        if (di + 1) % 10 == 0 or di == 0:
            log.info(f"  Prepared {di+1}/{len(dates)} days")

        del mid, preds, signal, vol_pred, vol_pct_thresholds
        gc.collect()

    cnn_data.close()
    log.info(f"Generated {len(saved_files)} prediction files for sim")
    return saved_files, ics


def run_single_sim(date_str, pred_file, config_label, threshold, hold_ms,
                   latency_ms, trailing_ticks):
    """Run a single Rust sim job."""
    nodash = date_str.replace('-', '')
    mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = SIM_OUT_DIR / f'{config_label}_{date_str}.json'

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
    if trailing_ticks > 0:
        cmd.extend(['--trailing-ticks', str(trailing_ticks)])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            log.warning(f"Sim failed for {config_label}/{date_str}: {r.stderr[:200]}")
            return None
        with open(out_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        log.warning(f"Sim timed out for {config_label}/{date_str}")
        return None
    except Exception as e:
        log.warning(f"Sim error for {config_label}/{date_str}: {e}")
        return None


def run_sweep(saved_files, workers=6):
    """Run Rust sim sweep across all configs and dates."""
    if not BINARY.exists():
        log.error(f"Rust binary not found: {BINARY}")
        return {}

    jobs = []
    for (date, vol_gate, tf), pred_file in saved_files.items():
        for threshold, hold_ms, latency_ms, trailing_ticks, base_label in SIM_CONFIGS:
            config_label = f'vol{vol_gate}_{tf}_{base_label}'
            jobs.append({
                'date': date,
                'pred_file': pred_file,
                'config_label': config_label,
                'threshold': threshold,
                'hold_ms': hold_ms,
                'latency_ms': latency_ms,
                'trailing_ticks': trailing_ticks,
            })

    log.info(f"Running {len(jobs)} sim jobs with {workers} workers")

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
                job['trailing_ticks']
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
                log.warning(f"Job failed: {e}")

            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(jobs) - done) / max(rate, 0.01)
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, ~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log.info(f"Sim sweep complete: {done} jobs in {elapsed:.1f}s")
    return results


def aggregate_and_report(results):
    """Aggregate results and print comprehensive report."""
    log.info("\n" + "=" * 80)
    log.info("CNN OOT → RUST MBO SIM RESULTS (REAL FILLS)")
    log.info("=" * 80)

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
            day_trades = res.get('total_trades', 0)
            day_signals = res.get('total_signals', 0)
            day_filled = res.get('total_filled', 0)

            total_pnl += day_pnl
            total_trades += day_trades
            total_signals += day_signals
            total_filled += day_filled
            daily_pnls.append(day_pnl)

            if 'trades' in res:
                for trade in res['trades']:
                    pnl = trade.get('pnl_dollars', 0)
                    all_trade_pnls.append(pnl)
                    if pnl > 0:
                        total_wins += 1

        n_days = len(date_results)
        if n_days == 0 or total_trades == 0:
            continue

        win_rate = total_wins / total_trades
        fill_rate = total_filled / total_signals if total_signals > 0 else 0
        avg_daily_pnl = np.mean(daily_pnls)
        daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 1e-8
        sharpe_daily = (avg_daily_pnl / daily_std) * np.sqrt(252) if daily_std > 0 else 0

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_trade_ticks = avg_trade_pnl / TICK_VALUE

        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        max_dd = abs(dd.min()) if len(dd) > 0 else 0

        summary.append({
            'config': config_label,
            'total_pnl': total_pnl,
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': fill_rate,
            'win_rate': win_rate,
            'sharpe_daily': sharpe_daily,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_trade_ticks': avg_trade_ticks,
            'max_dd': max_dd,
            'avg_daily_pnl': avg_daily_pnl,
        })

    summary.sort(key=lambda x: x['total_pnl'], reverse=True)

    log.info(f"\nTop configs by total P&L ({summary[0]['n_days']} OOT days):")
    log.info(f"{'Config':<45} {'P&L':>10} {'Trades':>7} {'FillR':>6} {'WinR':>6} {'Sharpe':>7} {'AvgT$':>8} {'MaxDD':>8}")
    log.info("-" * 110)

    for s in summary[:20]:
        log.info(
            f"{s['config']:<45} "
            f"${s['total_pnl']:>9,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"{s['sharpe_daily']:>7.2f} "
            f"${s['avg_trade_pnl']:>7.2f} "
            f"${s['max_dd']:>7,.0f}"
        )

    if len(summary) > 5:
        log.info(f"\nBottom 5 configs:")
        for s in summary[-5:]:
            log.info(
                f"{s['config']:<45} "
                f"${s['total_pnl']:>9,.0f} "
                f"{s['n_trades']:>7} "
                f"{s['fill_rate']:>5.1%} "
                f"{s['win_rate']:>5.1%} "
                f"{s['sharpe_daily']:>7.2f}"
            )

    if summary:
        best = summary[0]
        log.info(f"\n{'='*60}")
        log.info(f"BEST OOT CONFIG: {best['config']}")
        log.info(f"  Total P&L:       ${best['total_pnl']:,.2f}")
        log.info(f"  Days:            {best['n_days']}")
        log.info(f"  Total trades:    {best['n_trades']}")
        log.info(f"  Signals sent:    {best['n_signals']}")
        log.info(f"  Fill rate:       {best['fill_rate']:.1%}")
        log.info(f"  Win rate:        {best['win_rate']:.1%}")
        log.info(f"  Sharpe (daily):  {best['sharpe_daily']:.2f}")
        log.info(f"  Avg trade P&L:   ${best['avg_trade_pnl']:.2f} ({best['avg_trade_ticks']:.2f} ticks)")
        log.info(f"  Max drawdown:    ${best['max_dd']:,.2f}")
        log.info(f"  Avg daily P&L:   ${best['avg_daily_pnl']:.2f}")
        log.info(f"  Annualized:      ${best['avg_daily_pnl'] * 252:,.0f}")

    out_file = RESULTS_DIR / f'cnn_oot_sim_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nFull results saved: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='CNN OOT -> Rust MBO Sim Validation')
    parser.add_argument('--pred-file', type=str, default=str(DEFAULT_PRED_FILE))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--skip-gen', action='store_true')
    parser.add_argument('--ic-only', action='store_true', help='Only compute IC, no sim')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("CNN OOT -> Rust MBO Fill Simulator Validation")
    log.info(f"  Binary:      {BINARY}")
    log.info(f"  Predictions: {args.pred_file}")
    log.info(f"  MBO data:    {MBO_DIR}")
    log.info(f"  Workers:     {args.workers}")
    log.info("=" * 60)

    if args.ic_only:
        pred_data = np.load(args.pred_file, allow_pickle=True)
        pred_dict = {k: pred_data[k] for k in pred_data.files}
        ics = compute_ic_per_day(pred_dict, horizon=100)
        pred_data.close()

        out_file = RESULTS_DIR / f'cnn_oot_ic_{_ts}.json'
        with open(out_file, 'w') as f:
            json.dump(ics, f, indent=2, default=str)
        log.info(f"IC results saved: {out_file}")
        return

    if args.skip_gen:
        log.info("Loading existing prediction files...")
        saved_files = {}
        for f in PRED_OUT_DIR.glob('*.npz'):
            # Filename: 2025-12-01_vol50_morning_afternoon.npz
            stem = f.stem
            # Split on first _ after date (date has dashes)
            date = stem[:10]  # 2025-12-01
            rest = stem[11:]  # vol50_morning_afternoon
            rest_parts = rest.split('_', 1)
            vol_str = rest_parts[0]  # vol50
            vol_gate = int(vol_str.replace('vol', ''))
            tf = rest_parts[1] if len(rest_parts) > 1 else 'all'
            saved_files[(date, vol_gate, tf)] = f
        log.info(f"Found {len(saved_files)} existing prediction files")
        ics = []
    else:
        saved_files, ics = prepare_oot_predictions(args.pred_file)

    if not saved_files:
        log.error("No prediction files generated/found. Exiting.")
        return

    results = run_sweep(saved_files, workers=args.workers)

    if results:
        summary = aggregate_and_report(results)
    else:
        log.error("No sim results. Check MBO files and binary.")


if __name__ == '__main__':
    main()
