#!/usr/bin/env python3
"""
GNN Predictions -> Rust MBO Fill Simulator
==========================================
Run GNN walk-forward OOS predictions through the REAL Rust MBO event-by-event
fill simulator. Identical pipeline to cnn_rust_sim_validation.py but adapted
for GNN prediction format.

GNN train_gnn.py saves: gnn_oos_preds_{timestamp}.npz
  Keys: {date}_preds, {date}_targets (per-fold OOS predictions)

Pipeline:
  1. Load GNN walk-forward OOS predictions
  2. Apply de-biasing: expanding z-score + vol gate + time filter
  3. Save per-day NPZ files in Rust sim format
  4. Run fill_sim_cli.exe against each day's MBO .dbn.zst
  5. Aggregate results and report

Usage:
    python alpha_discovery/gnn_rust_sim_validation.py
    python alpha_discovery/gnn_rust_sim_validation.py --pred-file results/gnn_oos_preds_XXXX.npz
    python alpha_discovery/gnn_rust_sim_validation.py --workers 8
    python alpha_discovery/gnn_rust_sim_validation.py --skip-gen  # skip pred gen, just run sim
"""

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

LVL3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LVL3_ROOT))
sys.path.insert(0, str(LVL3_ROOT / 'alpha_discovery'))

from high_conviction_strategy import (
    load_all_dates, load_mbo_day, compute_trailing_vol,
    _precompute_vol_percentiles, compute_time_features,
    HOLD_PERIODS, TICK
)

# -- Paths --
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'mbo'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'gnn_sim_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'gnn_sim_results'
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

# GNN uses same 99-bar offset as CNN (book model warmup)
GNN_OFFSET = 99

TICK_VALUE = 12.50

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = str(RESULTS_DIR / f'gnn_rust_sim_{_ts}.log')

log = logging.getLogger('gnn_rust_sim')
log.setLevel(logging.INFO)

# File handler with UTF-8
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)

# Console handler
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)

# Force stdout to UTF-8
import io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -- Sweep configs (same as CNN for fair comparison) --
SIM_CONFIGS = [
    # (threshold, hold_ms, latency_ms, trailing_ticks, label)
    (0.5, 1800000, 0, 0, 'conv05_30min_lat0'),
    (0.5, 1800000, 10, 0, 'conv05_30min_lat10'),
    (1.0, 1800000, 0, 0, 'conv10_30min_lat0'),
    (1.0, 1800000, 10, 0, 'conv10_30min_lat10'),
    (1.5, 1800000, 0, 0, 'conv15_30min_lat0'),
    (1.5, 1800000, 10, 0, 'conv15_30min_lat10'),
    (2.0, 1800000, 0, 0, 'conv20_30min_lat0'),
    (2.0, 1800000, 10, 0, 'conv20_30min_lat10'),

    # With trailing stops
    (1.0, 1800000, 10, 8, 'conv10_30min_lat10_trail8'),
    (1.5, 1800000, 10, 8, 'conv15_30min_lat10_trail8'),
    (1.5, 1800000, 10, 12, 'conv15_30min_lat10_trail12'),

    # Shorter holds
    (1.0, 600000, 10, 0, 'conv10_10min_lat10'),
    (1.5, 600000, 10, 0, 'conv15_10min_lat10'),
    (1.0, 300000, 10, 0, 'conv10_5min_lat10'),
    (1.5, 300000, 10, 0, 'conv15_5min_lat10'),
]

VOL_GATES = [50, 70, 80]
TIME_FILTERS = ['morning_afternoon']


def zscore_expanding(arr):
    """Expanding-window z-score (no look-ahead). Matches debiased_sweep."""
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


def find_gnn_pred_file(results_dir):
    """Find the most recent GNN OOS predictions file."""
    candidates = sorted(
        results_dir.glob('gnn_oos_preds_*.npz'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return None


def prepare_gnn_predictions(pred_file):
    """Load GNN predictions, apply de-biasing, save per-day NPZ for Rust sim."""
    log.info(f"Loading GNN predictions from: {pred_file}")
    if not pred_file.exists():
        log.error(f"GNN predictions not found: {pred_file}")
        return {}

    gnn_data = np.load(str(pred_file), allow_pickle=True)
    all_keys = list(gnn_data.keys())
    log.info(f"GNN NPZ keys: {len(all_keys)} total")
    log.info(f"  Sample keys: {all_keys[:10]}")

    # Extract dates from prediction keys (format: {date}_preds)
    dates = sorted(set(
        k.rsplit('_', 1)[0] for k in all_keys if k.endswith('_preds')
    ))
    log.info(f"GNN dates with predictions: {len(dates)} ({dates[0] if dates else '?'} to {dates[-1] if dates else '?'})")

    if not dates:
        log.error("No prediction keys found in NPZ. Expected format: {date}_preds")
        log.error(f"  Found keys: {all_keys[:20]}")
        return {}

    # Skip first 20 days (param tuning, consistent with CNN)
    PARAM_TUNE_DAYS = 20
    oos_dates = dates[PARAM_TUNE_DAYS:]
    log.info(f"OOS dates (after {PARAM_TUNE_DAYS} tune days): {len(oos_dates)}")

    saved_files = {}

    for date in oos_dates:
        # Check MBO file exists
        nodash = date.replace('-', '')
        mbo_zst = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            log.warning(f"No MBO file for {date}, skipping")
            continue

        # Load MBO mid/spread for vol computation
        mbo = load_mbo_day(date)
        if mbo is None:
            log.warning(f"Could not load MBO day {date}")
            continue
        mid, spread = mbo
        n_bars = len(mid)

        # Load GNN predictions
        pred_key = f'{date}_preds'
        if pred_key not in gnn_data:
            log.warning(f"No predictions for {date}")
            continue
        gp = gnn_data[pred_key].astype(np.float64)

        # Align with offset: predictions start at bar GNN_OFFSET
        gp_aligned = np.full(n_bars, 0.0)
        end_idx = min(GNN_OFFSET + len(gp), n_bars)
        gp_aligned[GNN_OFFSET:end_idx] = gp[:end_idx - GNN_OFFSET]

        # Apply expanding z-score (no look-ahead)
        signal = zscore_expanding(gp_aligned)

        # Compute vol for gating
        vol_pred = compute_trailing_vol(mid)
        vol_pct_thresholds = _precompute_vol_percentiles(vol_pred)

        # Time features
        minutes, first_30, last_30, power_hour, lunch_dead, morning_afternoon = \
            compute_time_features(n_bars)

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
                elif tf == 'edges':
                    time_ok = first_30 | last_30
                elif tf == 'no_lunch':
                    time_ok = ~lunch_dead
                else:
                    time_ok = np.ones(n_bars, dtype=bool)

                filtered_signal[~time_ok] = 0.0
                filtered_signal = np.nan_to_num(filtered_signal, nan=0.0)

                key = (date, vol_gate, tf)
                out_file = PRED_OUT_DIR / f'{date}_vol{vol_gate}_{tf}.npz'
                np.savez_compressed(str(out_file), predictions=filtered_signal)
                saved_files[key] = out_file

        del mid, spread, mbo, signal, vol_pred, vol_pct_thresholds
        gc.collect()

    log.info(f"Generated {len(saved_files)} prediction files")
    return saved_files


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
        log.error("Build with: cd rust_cache_builder && cargo build --release")
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
                'vol_gate': vol_gate,
                'tf': tf,
            })

    log.info(f"Running {len(jobs)} sim jobs with {workers} workers")
    log.info(f"  Configs: {len(SIM_CONFIGS)}")
    log.info(f"  Vol gates: {VOL_GATES}")
    log.info(f"  Time filters: {TIME_FILTERS}")
    log.info(f"  Dates: {len(set(j['date'] for j in jobs))}")

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

            if done % 50 == 0:
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
    log.info("GNN -> RUST MBO SIM RESULTS (REAL FILLS)")
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

        win_rate = total_wins / total_trades if total_trades > 0 else 0
        fill_rate = total_filled / total_signals if total_signals > 0 else 0
        avg_daily_pnl = np.mean(daily_pnls) if daily_pnls else 0
        daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 1e-8
        sharpe_daily = (avg_daily_pnl / daily_std) * np.sqrt(252) if daily_std > 0 else 0

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_trade_ticks = avg_trade_pnl / TICK_VALUE if TICK_VALUE > 0 else 0

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

    log.info(f"\nTop 20 configs by total P&L ({summary[0]['n_days'] if summary else 0} OOS days):")
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
        log.info(f"BEST CONFIG: {best['config']}")
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

    # Compare with CNN results if available
    cnn_results = sorted(RESULTS_DIR.glob('cnn_rust_sim_results_*.json'), reverse=True)
    if cnn_results:
        try:
            with open(cnn_results[0]) as f:
                cnn_summary = json.load(f)
            if cnn_summary and summary:
                cnn_best = cnn_summary[0]
                gnn_best = summary[0]
                log.info(f"\n{'='*60}")
                log.info("GNN vs CNN COMPARISON (best config each):")
                log.info(f"  {'Metric':<20s} {'GNN':>12s} {'CNN':>12s}")
                log.info(f"  {'-'*44}")
                log.info(f"  {'Total P&L':<20s} ${gnn_best['total_pnl']:>10,.0f} ${cnn_best['total_pnl']:>10,.0f}")
                log.info(f"  {'Sharpe':<20s} {gnn_best['sharpe_daily']:>12.2f} {cnn_best['sharpe_daily']:>12.2f}")
                log.info(f"  {'Trades':<20s} {gnn_best['n_trades']:>12d} {cnn_best['n_trades']:>12d}")
                log.info(f"  {'Fill Rate':<20s} {gnn_best['fill_rate']:>11.1%} {cnn_best['fill_rate']:>11.1%}")
                log.info(f"  {'Win Rate':<20s} {gnn_best['win_rate']:>11.1%} {cnn_best['win_rate']:>11.1%}")
                log.info(f"  {'Avg Trade $':<20s} ${gnn_best['avg_trade_pnl']:>10.2f} ${cnn_best['avg_trade_pnl']:>10.2f}")
        except Exception as e:
            log.warning(f"Could not load CNN results for comparison: {e}")

    out_file = RESULTS_DIR / f'gnn_rust_sim_results_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nFull results saved: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='GNN -> Rust MBO Sim Validation')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--pred-file', type=str, default=None,
                        help='Path to GNN OOS predictions NPZ (auto-detects if not specified)')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip prediction generation, use existing files')
    args = parser.parse_args()

    # Find GNN predictions
    gnn_results_dir = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
    if args.pred_file:
        pred_file = Path(args.pred_file)
    else:
        pred_file = find_gnn_pred_file(gnn_results_dir)
        if pred_file is None:
            log.error(f"No GNN prediction files found in {gnn_results_dir}")
            log.error("Run GNN walk-forward training first: python train_gnn.py --folds 10")
            return

    log.info("=" * 60)
    log.info("GNN -> Rust MBO Fill Simulator Validation")
    log.info(f"  Binary: {BINARY}")
    log.info(f"  GNN predictions: {pred_file}")
    log.info(f"  MBO data dir: {MBO_DIR}")
    log.info(f"  Workers: {args.workers}")
    log.info("=" * 60)

    if args.skip_gen:
        log.info("Loading existing prediction files...")
        saved_files = {}
        for f in PRED_OUT_DIR.glob('*.npz'):
            parts = f.stem.split('_')
            date = f'{parts[0]}-{parts[1]}-{parts[2]}'
            vol_str = parts[3]
            vol_gate = int(vol_str.replace('vol', ''))
            tf = '_'.join(parts[4:])
            saved_files[(date, vol_gate, tf)] = f
        log.info(f"Found {len(saved_files)} existing prediction files")
    else:
        saved_files = prepare_gnn_predictions(pred_file)

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
