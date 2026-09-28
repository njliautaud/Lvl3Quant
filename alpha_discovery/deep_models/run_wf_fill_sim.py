#!/usr/bin/env python3
"""
Walk-Forward Predictions → Rust MBO Fill Simulator
====================================================
Runs the best IS config (chase-entry, 1t/3r, conv2.5, 30min) on the
walk-forward OOT predictions to get early P&L estimates.

Source: oot_wf_predictions_incremental.npz (22 days, Dec 2025 - Jan 2026)
Config: vol70, morning_afternoon filter, chase 1t/3r, threshold 2.5, hold 30min

Replicates the exact pipeline from cnn_oot_sim_validation.py:
  - Expanding z-score (no look-ahead)
  - Trailing vol gate (expanding percentile)
  - Time filter zeroed in Python (not --prime-hours flag)
  - Per-day NPZ with 'predictions' key

Usage:
    python alpha_discovery/deep_models/run_wf_fill_sim.py
    python alpha_discovery/deep_models/run_wf_fill_sim.py --workers 8
    python alpha_discovery/deep_models/run_wf_fill_sim.py --vol-gates 50,70,80
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
PRED_FILE = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_sim_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_sim_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19          # window_size - 1; aligns predictions with bar index
BARS_PER_SEC = 10        # 100ms bars
RTH_HOURS = 6.5
TICK_VALUE = 12.50

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = str(RESULTS_DIR / f'wf_fill_sim_{_ts}.log')

log = logging.getLogger('wf_fill_sim')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(_log_file, mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── Best IS config + sweep of a few alternatives for comparison ──
# Format: (threshold, hold_ms, latency_ms, chase_max_ticks, chase_max_reprices,
#          chase_force_cross, chase_interval_ms, label)
CHASE_CONFIGS = [
    # Best IS config (vol70/conv2.5/1t/3r/30min → IS Sharpe 3.28, +$15.5K/74d)
    (2.5, 1800000, 0, 1, 3, False, 100, 'chase_1t_3r_conv25_30min'),
    # Second-best IS config
    (2.5, 1800000, 0, 1, 3, False, 100, 'chase_1t_3r_conv25_30min_lat0'),  # same, explicit lat0
    # Other strong IS configs for comparison
    (2.0, 1800000, 0, 1, 3, False, 100, 'chase_1t_3r_conv20_30min'),
    (1.5, 1800000, 0, 1, 3, False, 100, 'chase_1t_3r_conv15_30min'),
    (2.5, 1800000, 0, 2, 5, False, 100, 'chase_2t_5r_conv25_30min'),
    (2.0, 1800000, 0, 2, 5, False, 100, 'chase_2t_5r_conv20_30min'),
    (2.5, 1800000, 10, 1, 3, False, 100, 'chase_1t_3r_conv25_30min_lat10'),
]

VOL_GATES = [50, 70, 80]   # overridable via --vol-gates
TIME_FILTERS = ['morning_afternoon']


# ── Signal processing (replicates cnn_oot_sim_validation.py) ──

def compute_trailing_vol(mid, window=3000):
    """Trailing realized vol (std of 1s returns) over window bars. No look-ahead."""
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
    """Time-of-day mask for a full RTH trading day (100ms bars)."""
    seconds = np.arange(n_bars) / BARS_PER_SEC
    minutes = seconds / 60.0
    morning_afternoon = (minutes < 120) | ((minutes >= 240) & (minutes < 330))
    return morning_afternoon


def zscore_expanding(arr):
    """Expanding-window z-score (no look-ahead). Requires >=50 non-NaN samples."""
    result = np.full_like(arr, np.nan, dtype=np.float64)
    running_sum = 0.0
    running_sq = 0.0
    count = 0
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v):
            continue
        running_sum += v
        running_sq += v * v
        count += 1
        if count >= 50:
            mean = running_sum / count
            var = (running_sq / count) - mean * mean
            std = max(np.sqrt(var), 1e-8)
            result[i] = (v - mean) / std
    return result


def compute_ic_per_day(pred_data, horizon=100):
    """Compute per-day IC (rank corr of raw predictions vs future returns)."""
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
        fwd_ret = np.full(n, np.nan)
        for i in range(n - horizon):
            if mid[i] > 0:
                fwd_ret[i] = (mid[i + horizon] - mid[i]) / mid[i]
        valid = ~np.isnan(preds) & ~np.isnan(fwd_ret) & (preds != 0)
        if valid.sum() < 100:
            continue
        ic, pval = stats.spearmanr(preds[valid], fwd_ret[valid])
        ics.append({'date': date, 'ic': float(ic), 'pval': float(pval), 'n_valid': int(valid.sum())})
    return ics


# ── Prediction file preparation ──

def prepare_wf_predictions(pred_file, vol_gates=None, time_filters=None):
    """
    Load WF predictions, apply de-biasing (z-score + vol gate + time filter),
    save per-day NPZ files for the Rust sim.
    """
    if vol_gates is None:
        vol_gates = VOL_GATES
    if time_filters is None:
        time_filters = TIME_FILTERS

    log.info(f"Loading WF predictions from {pred_file}...")
    wf_data = np.load(str(pred_file), allow_pickle=True)
    dates = sorted(set(k.rsplit('_', 1)[0] for k in wf_data.files if k.endswith('_preds')))
    log.info(f"WF dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    # Compute IC for reference
    pred_dict = {k: wf_data[k] for k in wf_data.files}
    log.info("\n--- Per-Day IC (raw predictions vs 10s forward returns) ---")
    ics = compute_ic_per_day(pred_dict, horizon=100)
    for ic_entry in ics:
        log.info(f"  {ic_entry['date']}: IC={ic_entry['ic']:+.4f} (p={ic_entry['pval']:.4f}, n={ic_entry['n_valid']:,})")
    if ics:
        ic_vals = [x['ic'] for x in ics]
        mean_ic = np.mean(ic_vals)
        std_ic = np.std(ic_vals)
        t_stat = mean_ic / (std_ic / np.sqrt(len(ic_vals))) if std_ic > 0 else 0
        pct_pos = np.mean([x > 0 for x in ic_vals]) * 100
        log.info(f"\n  IC Summary: mean={mean_ic:+.4f}, t={t_stat:.2f}, {pct_pos:.0f}% positive")

    saved_files = {}
    skipped = []

    for di, date in enumerate(dates):
        nodash = date.replace('-', '')
        mbo_zst = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
        mbo_dbn = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
        if not mbo_zst.exists() and not mbo_dbn.exists():
            log.warning(f"No MBO file for {date}, skipping")
            skipped.append(date)
            continue

        preds = wf_data[f'{date}_preds'].astype(np.float64)
        mid = wf_data[f'{date}_mid'].astype(np.float64)
        n_bars = len(mid)

        # Align predictions with CNN offset (zero-pad front)
        cp_aligned = np.zeros(n_bars, dtype=np.float64)
        end_idx = min(CNN_OFFSET + len(preds), n_bars)
        cp_aligned[CNN_OFFSET:end_idx] = preds[:end_idx - CNN_OFFSET]

        # Expanding z-score → conviction score (this is what signal-threshold filters on)
        signal = zscore_expanding(cp_aligned)

        # Compute trailing vol from mid prices (for vol gate)
        vol_pred = compute_trailing_vol(mid)
        vol_pct_thresholds = precompute_vol_percentiles(vol_pred)

        # Time filter mask
        morning_afternoon_mask = compute_time_features(n_bars)

        for vg in vol_gates:
            for tf in time_filters:
                filtered_signal = signal.copy()

                # Vol gate: zero out bars where vol < Nth percentile (expanding)
                if vg > 0:
                    available = sorted(vol_pct_thresholds.keys())
                    closest = min(available, key=lambda x: abs(x - vg))
                    vol_threshold = vol_pct_thresholds[closest]
                    for i in range(len(filtered_signal)):
                        if np.isnan(vol_pred[i]) or vol_pred[i] < vol_threshold[i]:
                            filtered_signal[i] = 0.0

                # Time filter: zero out bars outside window
                if tf == 'morning_afternoon':
                    filtered_signal[~morning_afternoon_mask] = 0.0

                filtered_signal = np.nan_to_num(filtered_signal, nan=0.0)

                out_file = PRED_OUT_DIR / f'{date}_vol{vg}_{tf}.npz'
                np.savez_compressed(str(out_file), predictions=filtered_signal)
                saved_files[(date, vg, tf)] = out_file

        if (di + 1) % 5 == 0 or di == 0:
            log.info(f"  Prepared {di+1}/{len(dates)} days")

        del mid, preds, signal, vol_pred, vol_pct_thresholds
        gc.collect()

    wf_data.close()
    log.info(f"\nGenerated {len(saved_files)} prediction files ({len(skipped)} dates skipped)")
    return saved_files, ics


# ── Simulation ──

def run_single_sim(date_str, pred_file, config_label, threshold, hold_ms,
                   latency_ms, chase_max_ticks, chase_max_reprices,
                   chase_force_cross, chase_interval_ms):
    """Run one Rust fill_sim job and return parsed JSON result."""
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
        '--chase-entry',
        '--chase-max-ticks', str(chase_max_ticks),
        '--chase-max-reprices', str(chase_max_reprices),
        '--chase-interval-ms', str(chase_interval_ms),
        '--quiet',
    ]
    if chase_force_cross:
        cmd.append('--chase-force-cross')

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


def run_sweep(saved_files, chase_configs, vol_gates, workers=6):
    """Run all sim jobs in parallel."""
    if not BINARY.exists():
        log.error(f"Binary not found: {BINARY}")
        return {}

    jobs = []
    for (date, vg, tf), pred_file in saved_files.items():
        if vg not in vol_gates:
            continue
        for (threshold, hold_ms, latency_ms, chase_max_ticks, chase_max_reprices,
             chase_force_cross, chase_interval_ms, base_label) in chase_configs:
            config_label = f'vol{vg}_{tf}_{base_label}'
            jobs.append({
                'date': date,
                'pred_file': pred_file,
                'config_label': config_label,
                'threshold': threshold,
                'hold_ms': hold_ms,
                'latency_ms': latency_ms,
                'chase_max_ticks': chase_max_ticks,
                'chase_max_reprices': chase_max_reprices,
                'chase_force_cross': chase_force_cross,
                'chase_interval_ms': chase_interval_ms,
            })

    # Deduplicate jobs (same config_label can appear if configs have same label)
    seen = set()
    unique_jobs = []
    for j in jobs:
        key = (j['date'], j['config_label'])
        if key not in seen:
            seen.add(key)
            unique_jobs.append(j)
    jobs = unique_jobs

    log.info(f"\nRunning {len(jobs)} sim jobs ({workers} workers)")
    log.info(f"  Dates: {len(set(j['date'] for j in jobs))}")
    log.info(f"  Configs: {len(set(j['config_label'] for j in jobs))}")

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


# ── Aggregation ──

def aggregate_and_report(results, ics=None):
    """Aggregate per-day results into per-config summaries."""
    log.info("\n" + "=" * 95)
    log.info("WALK-FORWARD OOT → RUST MBO FILL SIM RESULTS (REAL FILLS)")
    log.info("Predictions from walk-forward retrained models (Dec 2025 - Jan 2026)")
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
        if n_days == 0 or total_trades == 0:
            continue

        win_rate = total_wins / total_trades
        fill_rate = total_filled / total_signals if total_signals > 0 else 0
        avg_daily = np.mean(daily_pnls)
        daily_std = np.std(daily_pnls) if len(daily_pnls) > 1 else 1e-8
        sharpe = (avg_daily / daily_std) * np.sqrt(252) if daily_std > 0 else 0

        avg_trade_pnl = np.mean(all_trade_pnls) if all_trade_pnls else 0
        avg_trade_ticks = avg_trade_pnl / TICK_VALUE

        cum = np.cumsum(daily_pnls)
        peak = np.maximum.accumulate(cum)
        max_dd = abs((cum - peak).min()) if len(cum) > 0 else 0

        summary.append({
            'config': config_label,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe_daily': round(sharpe, 3),
            'avg_daily_pnl': round(avg_daily, 2),
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'avg_trade_ticks': round(avg_trade_ticks, 3),
            'max_dd': round(max_dd, 2),
            'annualized_pnl': round(avg_daily * 252, 0),
            'daily_pnls': daily_pnls,
        })

    summary.sort(key=lambda x: x['total_pnl'], reverse=True)

    if not summary:
        log.warning("No configs produced trades!")
        return summary

    log.info(f"\nAll configs by total P&L ({summary[0]['n_days']} WF-OOT days):")
    log.info(f"{'Config':<55} {'P&L':>10} {'Trades':>7} {'FillR':>6} {'WinR':>6} {'Sharpe':>7} {'MaxDD':>8}")
    log.info("-" * 110)
    for s in summary:
        log.info(
            f"{s['config']:<55} "
            f"${s['total_pnl']:>9,.0f} "
            f"{s['n_trades']:>7} "
            f"{s['fill_rate']:>5.1%} "
            f"{s['win_rate']:>5.1%} "
            f"{s['sharpe_daily']:>7.2f} "
            f"${s['max_dd']:>7,.0f}"
        )

    if summary:
        best = summary[0]
        log.info(f"\n{'='*65}")
        log.info(f"BEST WF-OOT CONFIG: {best['config']}")
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
        log.info(f"  Annualized:      ${best['annualized_pnl']:,.0f}")

    # Compare best WF config vs best IS result
    log.info(f"\n{'='*65}")
    log.info("COMPARISON — IS Best vs WF-OOT Best:")
    log.info(f"  IS Best (vol70/conv2.5/1t/3r/30min):   Sharpe 3.28, +$15,479/74d, 130 trades, 8.6% fill")
    if summary:
        b = summary[0]
        log.info(f"  WF-OOT Best ({b['config'][:35]}): Sharpe {b['sharpe_daily']:.2f}, ${b['total_pnl']:,.0f}/{b['n_days']}d, {b['n_trades']} trades, {b['fill_rate']:.1%} fill")

    # Save clean results (without raw daily_pnls for readability)
    out_summary = [{k: v for k, v in s.items() if k != 'daily_pnls'} for s in summary]
    out_file = RESULTS_DIR / 'wf_fill_sim_results.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'pred_file': str(PRED_FILE),
            'n_dates': summary[0]['n_days'] if summary else 0,
            'ic_summary': {
                'mean': round(np.mean([x['ic'] for x in ics]), 4) if ics else None,
                'pct_positive': round(np.mean([x['ic'] > 0 for x in ics]) * 100, 1) if ics else None,
                'n_days': len(ics),
            } if ics else {},
            'configs': out_summary,
        }, f, indent=2, default=str)
    log.info(f"\nResults saved: {out_file}")

    return summary


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description='WF Predictions → Rust MBO Fill Sim')
    parser.add_argument('--workers', type=int, default=6,
                        help='Parallel sim workers (default: 6)')
    parser.add_argument('--vol-gates', type=str, default=None,
                        help='Comma-separated vol gates, e.g. "50,70,80" (default: all 3)')
    parser.add_argument('--skip-gen', action='store_true',
                        help='Skip prediction file generation (use existing files in PRED_OUT_DIR)')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to model training config JSON with fold boundaries (for OOT date verification)')
    parser.add_argument('--no-verify-oot', action='store_true',
                        help='Skip OOT date verification (NOT recommended — leakage risk)')
    args = parser.parse_args()

    vol_gates = VOL_GATES
    if args.vol_gates:
        vol_gates = [int(x) for x in args.vol_gates.split(',')]

    log.info("=" * 65)
    log.info("Walk-Forward OOT → Rust MBO Fill Simulator")
    log.info(f"  Binary:      {BINARY}")
    log.info(f"  Predictions: {PRED_FILE}")
    log.info(f"  MBO data:    {MBO_DIR}")
    log.info(f"  Pred out:    {PRED_OUT_DIR}")
    log.info(f"  Sim out:     {SIM_OUT_DIR}")
    log.info(f"  Workers:     {args.workers}")
    log.info(f"  Vol gates:   {vol_gates}")
    log.info("=" * 65)

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

    # OOT Date Verification — MANDATORY leakage guard
    # Different researchers train vs test; date ranges can silently disconnect.
    if not args.no_verify_oot:
        log.info("=" * 65)
        log.info("OOT DATE VERIFICATION (--no-verify-oot to skip)")
        wf_data_check = np.load(str(PRED_FILE), allow_pickle=True)
        test_dates = sorted(set(
            k.rsplit('_', 1)[0] for k in wf_data_check.files if k.endswith('_preds')
        ))
        log.info(f"  Fill sim test dates: {test_dates[0]} → {test_dates[-1]} ({len(test_dates)} days)")

        # Try to load training boundary from model config
        train_last_date = None
        config_path = Path(args.model_config) if args.model_config else None

        # Auto-detect: look for training.log or config.json next to pred file
        if config_path is None:
            candidates = [
                PRED_FILE.parent / 'training.log',
                PRED_FILE.parent / 'config.json',
                PRED_FILE.parent / 'fold_config.json',
            ]
            for c in candidates:
                if c.exists():
                    config_path = c
                    break

        if config_path and config_path.exists():
            try:
                if config_path.suffix == '.json':
                    cfg = json.loads(config_path.read_text())
                    # Support both flat and nested fold config formats
                    if 'train_end_date' in cfg:
                        train_last_date = cfg['train_end_date']
                    elif 'folds' in cfg:
                        all_train = [f.get('train_end') for f in cfg['folds'] if f.get('train_end')]
                        if all_train:
                            train_last_date = max(all_train)
                else:
                    # Parse training.log — look for last IS date in fold headers
                    log_text = config_path.read_text(errors='replace')
                    import re
                    # Matches "Train: N days (YYYYMMDD...→YYYYMMDD_mbo_events...)"
                    matches = re.findall(r'Train:.*?(\d{8})_mbo_events', log_text)
                    if matches:
                        last_yyyymmdd = max(matches)
                        train_last_date = f"{last_yyyymmdd[:4]}-{last_yyyymmdd[4:6]}-{last_yyyymmdd[6:8]}"
            except Exception as e:
                log.warning(f"  Could not parse model config {config_path}: {e}")

        if train_last_date:
            log.info(f"  Model IS training ended: {train_last_date}")
            overlapping = [d for d in test_dates if d <= train_last_date]
            if overlapping:
                log.error("=" * 65)
                log.error("LEAKAGE DETECTED — ABORTING FILL SIM")
                log.error(f"  Training ended:  {train_last_date}")
                log.error(f"  Overlapping test dates ({len(overlapping)}): {overlapping}")
                log.error("  These dates were IN the training set. Results would be invalid.")
                log.error("  Fix: use test dates strictly AFTER training end date.")
                log.error("=" * 65)
                sys.exit(1)
            else:
                log.info(f"  OOT VERIFIED: all test dates are strictly after {train_last_date}")
        else:
            log.warning("  Could not determine training end date from model config.")
            log.warning("  Pass --model-config <path/to/training.log or config.json> to enable full check.")
            log.warning("  Proceeding WITHOUT date verification — manual check required.")

        log.info("=" * 65)

    # Step 1: Prepare per-day NPZ prediction files
    ics = []
    if args.skip_gen:
        log.info("Loading existing prediction files...")
        saved_files = {}
        for f in PRED_OUT_DIR.glob('*.npz'):
            stem = f.stem
            date = stem[:10]
            rest = stem[11:]
            rest_parts = rest.split('_', 1)
            vol_str = rest_parts[0]
            tf = rest_parts[1] if len(rest_parts) > 1 else 'all'
            try:
                vg = int(vol_str.replace('vol', ''))
            except ValueError:
                continue
            if vg in vol_gates:
                saved_files[(date, vg, tf)] = f
        log.info(f"Found {len(saved_files)} existing prediction files")
    else:
        saved_files, ics = prepare_wf_predictions(PRED_FILE, vol_gates, TIME_FILTERS)

    if not saved_files:
        log.error("No prediction files found/generated. Exiting.")
        sys.exit(1)

    # Step 2: Run fill_sim sweep
    results = run_sweep(saved_files, CHASE_CONFIGS, vol_gates, workers=args.workers)

    if not results:
        log.error("No sim results. Check binary and MBO files.")
        sys.exit(1)

    # Step 3: Aggregate and save
    summary = aggregate_and_report(results, ics)
    log.info(f"\nDone. Log: {_log_file}")

    return summary


if __name__ == '__main__':
    main()
