#!/usr/bin/env python3
"""
Latency Validation Sweep — Test top configs at realistic retail latencies
=========================================================================
Tests the TOP 10 norm-sweep configs at 0/100/200/300/500ms latency to see
if the edge survives real-world retail conditions.

Uses EXISTING prediction files from cnn_wf_norm_sweep_predictions/.
No regeneration needed — just runs Rust fill_sim with --latency-ms.

Usage:
    python alpha_discovery/deep_models/run_latency_validation.py
    python alpha_discovery/deep_models/run_latency_validation.py --workers 24
"""

import sys
import json
import math
import time
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
PRED_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_norm_sweep_predictions'
RESULTS_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_latency_validation'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Logging ──
log = logging.getLogger('latency_val')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'latency_validation_{_ts}.log'), mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── TOP 10 CONFIGS ──
# Each: (pred_pattern, conv_threshold, hold_ms, take_profit_ticks, label)
# pred_pattern = filename pattern in cnn_wf_norm_sweep_predictions (DATE_ prefix added)
# conv = signal-threshold for fill_sim
# hold_ms = hold time in ms
# tp = take-profit ticks (None if not used)

TOP_CONFIGS = [
    ('ema_zscore_span5000_vol70',    2.5, 1800000, None, 'ema5000_vol70_conv25_hold30m'),
    ('smooth10_expanding_vol70',     2.0, 1800000, 5,    'smooth10_vol70_conv20_hold30m_tp5'),
    ('rolling_zscore_w5000_vol70',   2.5, 1800000, None, 'roll5000_vol70_conv25_hold30m'),
    ('ema_zscore_span3000_vol70',    2.5, 1800000, None, 'ema3000_vol70_conv25_hold30m'),
    ('smooth10_expanding_vol70',     2.5, 1800000, 5,    'smooth10_vol70_conv25_hold30m_tp5'),
    ('expanding_zscore_vol80',       2.5, 1800000, 5,    'expanding_vol80_conv25_hold30m_tp5'),
    ('expanding_zscore_vol50',       2.5, 1800000, None, 'expanding_vol50_conv25_hold30m'),
    ('smooth100_expanding_vol0',     2.5, 1800000, 10,   'smooth100_vol0_conv25_hold30m_tp10'),
    ('smooth50_expanding_vol70',     2.5, 1800000, 10,   'smooth50_vol70_conv25_hold30m_tp10'),
    ('ema_zscore_span5000_vol70',    2.5, 1200000, None, 'ema5000_vol70_conv25_hold20m'),
]

LATENCIES_MS = [0, 100, 200, 300, 500]

# Chase params (from best known config: 1t/3r)
CHASE_MAX_TICKS = 1
CHASE_MAX_REPRICES = 3


def find_dates():
    """Find dates that have BOTH prediction files and MBO data."""
    # Get all dates from predictions
    pred_dates = set()
    for f in PRED_DIR.glob('*.npz'):
        parts = f.stem.split('_', 1)
        if len(parts) == 2 and len(parts[0]) == 10:  # YYYY-MM-DD
            pred_dates.add(parts[0])

    # Get MBO dates
    mbo_dates = {}
    for f in MBO_DIR.glob('*.dbn.zst'):
        # Extract YYYYMMDD from filename like glbx-mdp3-20251201.mbo.dbn.zst
        name = f.stem.split('.')[0]  # glbx-mdp3-20251201
        date_compact = name.split('-')[-1]  # 20251201
        if len(date_compact) == 8:
            date_iso = f'{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:8]}'
            mbo_dates[date_iso] = f

    # Intersection
    common = sorted(pred_dates & set(mbo_dates.keys()))
    log.info(f"Prediction dates: {len(pred_dates)}, MBO dates: {len(mbo_dates)}, Common: {len(common)}")
    return common, mbo_dates


def run_single_sim(mbo_file, pred_file, output_file, conv, hold_ms, latency_ms, tp_ticks):
    """Run a single fill_sim_cli invocation."""
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(output_file),
        '--hold-ms', str(hold_ms),
        '--signal-threshold', str(conv),
        '--latency-ms', str(latency_ms),
        '--chase-entry',
        '--chase-max-ticks', str(CHASE_MAX_TICKS),
        '--chase-max-reprices', str(CHASE_MAX_REPRICES),
        '--quiet',
    ]

    if tp_ticks is not None:
        cmd += ['--take-profit-ticks', str(tp_ticks)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return None
        if Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        log.error(f"Sim failed: {e}")
    return None


def run_sweep(dates, mbo_dates, workers=24):
    """Run latency sweep across all dates x configs x latencies."""
    jobs = []

    for date in dates:
        mbo_file = mbo_dates[date]

        for pred_pattern, conv, hold_ms, tp, label in TOP_CONFIGS:
            pred_file = PRED_DIR / f'{date}_{pred_pattern}.npz'
            if not pred_file.exists():
                continue

            for lat_ms in LATENCIES_MS:
                out_file = RESULTS_DIR / f'{label}_lat{lat_ms}ms_{date}.json'
                jobs.append({
                    'mbo_file': str(mbo_file),
                    'pred_file': str(pred_file),
                    'output_file': str(out_file),
                    'conv': conv,
                    'hold_ms': hold_ms,
                    'latency_ms': lat_ms,
                    'tp': tp,
                    'label': label,
                    'date': date,
                    'lat_ms': lat_ms,
                })

    log.info(f"Total jobs: {len(jobs)} ({len(dates)} dates x {len(TOP_CONFIGS)} configs x {len(LATENCIES_MS)} latencies)")
    log.info(f"Running with {workers} workers...")

    completed = 0
    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            f = executor.submit(
                run_single_sim,
                job['mbo_file'], job['pred_file'], job['output_file'],
                job['conv'], job['hold_ms'], job['latency_ms'], job['tp']
            )
            futures[f] = job

        for future in as_completed(futures):
            job = futures[future]
            completed += 1
            res = future.result()
            if res:
                results.append({
                    'config': job['label'],
                    'date': job['date'],
                    'latency_ms': job['lat_ms'],
                    'total_pnl_dollars': res.get('total_pnl_dollars', 0),
                    'total_trades': res.get('total_trades', 0),
                    'total_signals': res.get('total_signals', 0),
                    'total_filled': res.get('total_filled', 0),
                    'win_rate': res.get('win_rate', 0),
                })

            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - completed) / rate if rate > 0 else 0
                log.info(f"  Progress: {completed}/{len(jobs)} ({rate:.1f}/s, ETA {eta/60:.1f}min)")

    elapsed = time.time() - t0
    log.info(f"Completed {completed} jobs in {elapsed:.0f}s ({completed/elapsed:.1f}/s)")
    return results


def aggregate_and_report(results):
    """Aggregate per-date results into per-config-per-latency summaries, print comparison table."""

    # Group by (config, latency)
    groups = defaultdict(list)
    for r in results:
        groups[(r['config'], r['latency_ms'])].append(r)

    summaries = []
    for (config, lat_ms), day_results in sorted(groups.items()):
        daily_pnls = [r['total_pnl_dollars'] for r in day_results]
        n_days = len(daily_pnls)
        total_pnl = sum(daily_pnls)
        total_trades = sum(r['total_trades'] for r in day_results)
        total_signals = sum(r['total_signals'] for r in day_results)
        total_filled = sum(r['total_filled'] for r in day_results)

        mean_daily = total_pnl / n_days if n_days > 0 else 0
        if n_days > 1:
            variance = sum((x - mean_daily)**2 for x in daily_pnls) / (n_days - 1)
            std_daily = math.sqrt(variance) if variance > 0 else 0
        else:
            std_daily = 0

        sharpe = (mean_daily / std_daily * math.sqrt(252)) if std_daily > 0 else 0
        wr = sum(r['win_rate'] * r['total_trades'] for r in day_results) / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)

        summaries.append({
            'config': config,
            'latency_ms': lat_ms,
            'n_days': n_days,
            'total_pnl': round(total_pnl, 2),
            'annualized_pnl': round(mean_daily * 252, 0),
            'sharpe': round(sharpe, 3),
            'total_trades': total_trades,
            'win_rate': round(wr, 4),
            'fill_rate': round(fill_rate, 4),
            'trades_per_day': round(total_trades / max(n_days, 1), 1),
        })

    # Build comparison table: config -> {latency: sharpe}
    config_lat_sharpe = defaultdict(dict)
    config_lat_pnl = defaultdict(dict)
    config_lat_trades = defaultdict(dict)
    config_lat_wr = defaultdict(dict)
    config_lat_fill = defaultdict(dict)
    for s in summaries:
        config_lat_sharpe[s['config']][s['latency_ms']] = s['sharpe']
        config_lat_pnl[s['config']][s['latency_ms']] = s['total_pnl']
        config_lat_trades[s['config']][s['latency_ms']] = s['total_trades']
        config_lat_wr[s['config']][s['latency_ms']] = s['win_rate']
        config_lat_fill[s['config']][s['latency_ms']] = s['fill_rate']

    # Get config order (by 0ms Sharpe descending)
    config_order = sorted(config_lat_sharpe.keys(),
                          key=lambda c: config_lat_sharpe[c].get(0, 0), reverse=True)

    # ── SHARPE TABLE ──
    log.info('')
    log.info('=' * 120)
    log.info('LATENCY VALIDATION — SHARPE COMPARISON')
    log.info('=' * 120)

    header = f'{"Config":<45}'
    for lat in LATENCIES_MS:
        header += f' {lat:>6}ms'
    header += '  | Drop 0->200ms | Drop 0->300ms | FLAG'
    log.info(header)
    log.info('-' * 120)

    flagged = []
    for config in config_order:
        row = f'{config:<45}'
        s0 = config_lat_sharpe[config].get(0, 0)
        for lat in LATENCIES_MS:
            s = config_lat_sharpe[config].get(lat, 0)
            row += f' {s:>7.2f}'

        s200 = config_lat_sharpe[config].get(200, 0)
        s300 = config_lat_sharpe[config].get(300, 0)
        drop200 = (1 - s200 / s0) * 100 if s0 != 0 else 0
        drop300 = (1 - s300 / s0) * 100 if s0 != 0 else 0

        flag = ''
        if abs(drop200) > 50 or (s0 > 0 and s200 <= 0):
            flag = '*** >50% DROP ***'
            flagged.append(config)

        row += f'  | {drop200:>+10.1f}%    | {drop300:>+10.1f}%    | {flag}'
        log.info(row)

    # ── P&L TABLE ──
    log.info('')
    log.info('=' * 120)
    log.info('LATENCY VALIDATION — TOTAL P&L ($)')
    log.info('=' * 120)
    header = f'{"Config":<45}'
    for lat in LATENCIES_MS:
        header += f' {lat:>8}ms'
    log.info(header)
    log.info('-' * 120)
    for config in config_order:
        row = f'{config:<45}'
        for lat in LATENCIES_MS:
            pnl = config_lat_pnl[config].get(lat, 0)
            row += f' {pnl:>9,.0f}'
        log.info(row)

    # ── TRADES TABLE ──
    log.info('')
    log.info('=' * 120)
    log.info('LATENCY VALIDATION — TOTAL TRADES / WIN RATE / FILL RATE')
    log.info('=' * 120)
    header = f'{"Config":<45}'
    for lat in LATENCIES_MS:
        header += f'  {lat:>3}ms(T/WR/F%)'
    log.info(header)
    log.info('-' * 120)
    for config in config_order:
        row = f'{config:<45}'
        for lat in LATENCIES_MS:
            t = config_lat_trades[config].get(lat, 0)
            wr = config_lat_wr[config].get(lat, 0) * 100
            fr = config_lat_fill[config].get(lat, 0) * 100
            row += f'  {t:>4}/{wr:>4.1f}/{fr:>4.1f}'
        log.info(row)

    # ── SUMMARY ──
    log.info('')
    log.info('=' * 120)
    log.info('SUMMARY')
    log.info('=' * 120)

    surviving = [c for c in config_order
                 if c not in flagged and config_lat_sharpe[c].get(200, 0) > 0.5]
    log.info(f'Configs tested: {len(config_order)}')
    log.info(f'Flagged (>50% Sharpe drop at 200ms): {len(flagged)}')
    log.info(f'Surviving (Sharpe > 0.5 at 200ms, <50% drop): {len(surviving)}')

    if surviving:
        log.info('\nSURVIVORS (viable at retail latency):')
        for c in surviving:
            s0 = config_lat_sharpe[c].get(0, 0)
            s200 = config_lat_sharpe[c].get(200, 0)
            s300 = config_lat_sharpe[c].get(300, 0)
            log.info(f'  {c:<45} Sharpe: {s0:.2f} -> {s200:.2f} (200ms) -> {s300:.2f} (300ms)')

    if flagged:
        log.info('\nFLAGGED (edge destroyed by latency):')
        for c in flagged:
            s0 = config_lat_sharpe[c].get(0, 0)
            s200 = config_lat_sharpe[c].get(200, 0)
            log.info(f'  {c:<45} Sharpe: {s0:.2f} -> {s200:.2f} (200ms)')

    return summaries, flagged, surviving


# ── Main ──

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latency validation sweep for top norm-sweep configs')
    parser.add_argument('--workers', type=int, default=24, help='Parallel workers (default: 24)')
    args = parser.parse_args()

    log.info('=' * 70)
    log.info('LATENCY VALIDATION SWEEP — Rust MBO Fill Sim')
    log.info(f'Configs: {len(TOP_CONFIGS)}')
    log.info(f'Latencies: {LATENCIES_MS}')
    log.info(f'Workers: {args.workers}')
    log.info(f'Chase: {CHASE_MAX_TICKS}t/{CHASE_MAX_REPRICES}r')
    log.info('=' * 70)

    # Find dates
    dates, mbo_dates = find_dates()
    if not dates:
        log.error('No matching dates found!')
        sys.exit(1)

    log.info(f'Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)')

    # Run sweep
    results = run_sweep(dates, mbo_dates, workers=args.workers)

    # Aggregate and report
    summaries, flagged, surviving = aggregate_and_report(results)

    # Save raw results
    out_path = RESULTS_DIR / f'latency_validation_{_ts}.json'
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': _ts,
            'n_configs': len(TOP_CONFIGS),
            'latencies': LATENCIES_MS,
            'n_dates': len(dates),
            'dates': dates,
            'chase': f'{CHASE_MAX_TICKS}t/{CHASE_MAX_REPRICES}r',
            'summaries': summaries,
            'flagged_configs': flagged,
            'surviving_configs': surviving,
        }, f, indent=2)

    log.info(f'\nResults saved to {out_path}')
    log.info('DONE.')
