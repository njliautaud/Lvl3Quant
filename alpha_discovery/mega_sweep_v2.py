#!/usr/bin/env python3
"""
Mega Sweep V2 — Comprehensive fill_sim parameter search
========================================================
Runs on servers (Jupiter/Saturn) with all available parameters.
Designed for maximum coverage: hold times, stops, targets, entry modes,
time filters, signal-flip exits, and more.

Usage:
    python alpha_discovery/mega_sweep_v2.py --workers 36 --batch all
    python alpha_discovery/mega_sweep_v2.py --workers 36 --batch exits   # just exit variants
    python alpha_discovery/mega_sweep_v2.py --workers 36 --batch holds   # just hold time variants
"""

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

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
if sys.platform == 'win32':
    BINARY = BINARY.with_suffix('.exe')
MBO_DIR = ROOT / 'data' / 'raw' / 'mbo'
PRED_DIR = ROOT / 'data' / 'processed' / 'cnn_oot_sim_predictions'
RESULTS_DIR = ROOT / 'alpha_discovery' / 'results' / 'oot'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SIM_OUT_DIR = ROOT / 'data' / 'processed' / 'mega_sweep_v2_results'
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('mega_sweep_v2')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'mega_sweep_v2_{_ts}.log'), mode='w')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)

# ── Parameter grids ──

# Vol gates
VOL_GATES = [50, 60, 70, 80]

# Hold times in ms
HOLD_TIMES = {
    '3min': 180_000,
    '5min': 300_000,
    '10min': 600_000,
    '30min': 1_800_000,
}

# Conviction thresholds (signal z-score)
CONVICTIONS = [1.0, 1.5, 2.0, 2.5]

# Latency in ms
LATENCIES = [0, 25, 50, 100, 150]

# Trailing stops in ticks (0 = none)
TRAILING_STOPS = [0, 2, 4, 6, 8]

# Take profit in ticks (0 = none)
TAKE_PROFITS = [0, 8, 12, 16, 20]

# Entry modes
ENTRY_MODES = {
    'passive': {},
    'chase_1t3r': {'chase': True, 'chase_max_ticks': 1, 'chase_max_reprices': 3},
    'chase_2t5r': {'chase': True, 'chase_max_ticks': 2, 'chase_max_reprices': 5},
}

# Boolean flags
SIGNAL_FLIP_OPTIONS = [False, True]
PRIME_HOURS_OPTIONS = [False, True]


def generate_configs(batch='all'):
    """Generate config list based on batch selection."""
    configs = []

    if batch in ('all', 'core'):
        # Core grid: vol × hold × conviction × latency × entry (passive + chase)
        # No stops/targets, no signal flip, no prime hours
        for vol in VOL_GATES:
            for hold_name, hold_ms in HOLD_TIMES.items():
                for conv in CONVICTIONS:
                    for lat in LATENCIES:
                        for entry_name, entry_params in ENTRY_MODES.items():
                            label = f'v{vol}_{entry_name}_c{conv}_{hold_name}_lat{lat}'
                            configs.append({
                                'label': label,
                                'vol_gate': vol,
                                'hold_ms': hold_ms,
                                'conviction': conv,
                                'latency_ms': lat,
                                'trailing_ticks': 0,
                                'take_profit_ticks': 0,
                                'signal_flip': False,
                                'prime_hours': False,
                                **entry_params,
                            })

    if batch in ('all', 'exits'):
        # Exit variants: trailing stops and take profits
        # Use best vol gates and convictions from prior results
        for vol in [70, 80]:
            for hold_name, hold_ms in HOLD_TIMES.items():
                for conv in [1.5, 2.5]:
                    for trail in TRAILING_STOPS:
                        for tp in TAKE_PROFITS:
                            if trail == 0 and tp == 0:
                                continue  # already covered in core
                            label = f'v{vol}_passive_c{conv}_{hold_name}_lat0_trail{trail}_tp{tp}'
                            configs.append({
                                'label': label,
                                'vol_gate': vol,
                                'hold_ms': hold_ms,
                                'conviction': conv,
                                'latency_ms': 0,
                                'trailing_ticks': trail,
                                'take_profit_ticks': tp,
                                'signal_flip': False,
                                'prime_hours': False,
                            })

    if batch in ('all', 'signal_flip'):
        # Signal-flip exit (exit when signal reverses)
        for vol in [70, 80]:
            for hold_name, hold_ms in HOLD_TIMES.items():
                for conv in [1.5, 2.0, 2.5]:
                    for entry_name, entry_params in [('passive', {}), ('chase_1t3r', {'chase': True, 'chase_max_ticks': 1, 'chase_max_reprices': 3})]:
                        label = f'v{vol}_{entry_name}_c{conv}_{hold_name}_lat0_sigflip'
                        configs.append({
                            'label': label,
                            'vol_gate': vol,
                            'hold_ms': hold_ms,
                            'conviction': conv,
                            'latency_ms': 0,
                            'trailing_ticks': 0,
                            'take_profit_ticks': 0,
                            'signal_flip': True,
                            'prime_hours': False,
                            **entry_params,
                        })

    if batch in ('all', 'prime_hours'):
        # Prime hours only (10:30 AM - 2:30 PM ET)
        for vol in [70, 80]:
            for hold_name, hold_ms in HOLD_TIMES.items():
                for conv in [1.5, 2.0, 2.5]:
                    for entry_name, entry_params in [('passive', {}), ('chase_1t3r', {'chase': True, 'chase_max_ticks': 1, 'chase_max_reprices': 3})]:
                        label = f'v{vol}_{entry_name}_c{conv}_{hold_name}_lat0_prime'
                        configs.append({
                            'label': label,
                            'vol_gate': vol,
                            'hold_ms': hold_ms,
                            'conviction': conv,
                            'latency_ms': 0,
                            'trailing_ticks': 0,
                            'take_profit_ticks': 0,
                            'signal_flip': False,
                            'prime_hours': True,
                            **entry_params,
                        })

    if batch in ('all', 'holds'):
        # Just the new hold times (3min, 5min) with best other params
        for vol in VOL_GATES:
            for hold_name in ['3min', '5min']:
                hold_ms = HOLD_TIMES[hold_name]
                for conv in CONVICTIONS:
                    for entry_name, entry_params in ENTRY_MODES.items():
                        label = f'v{vol}_{entry_name}_c{conv}_{hold_name}_lat0'
                        configs.append({
                            'label': label,
                            'vol_gate': vol,
                            'hold_ms': hold_ms,
                            'conviction': conv,
                            'latency_ms': 0,
                            'trailing_ticks': 0,
                            'take_profit_ticks': 0,
                            'signal_flip': False,
                            'prime_hours': False,
                            **entry_params,
                        })

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = c['label']
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def discover_prediction_files():
    """Find all prediction NPZ files in PRED_DIR."""
    files = {}
    for f in sorted(PRED_DIR.glob('*.npz')):
        # Format: 2025-12-01_vol70_morning_afternoon.npz
        parts = f.stem.split('_', 1)
        if len(parts) >= 2:
            date = parts[0]
            rest = parts[1]
            # Extract vol gate from filename
            for vg in VOL_GATES:
                if f'vol{vg}' in rest:
                    files[(date, vg)] = f
                    break
    return files


def run_single_sim(date_str, pred_file, config):
    """Run a single Rust sim job with full config."""
    nodash = date_str.replace('-', '')
    mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
    if not mbo_file.exists():
        mbo_file = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
    if not mbo_file.exists():
        return None

    out_file = SIM_OUT_DIR / f'{config["label"]}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--hold-ms', str(config['hold_ms']),
        '--signal-threshold', str(config['conviction']),
        '--latency-ms', str(config['latency_ms']),
        '--quiet',
    ]

    if config.get('trailing_ticks', 0) > 0:
        cmd.extend(['--trailing-ticks', str(config['trailing_ticks'])])
    if config.get('take_profit_ticks', 0) > 0:
        cmd.extend(['--take-profit-ticks', str(config['take_profit_ticks'])])
    if config.get('signal_flip', False):
        cmd.append('--signal-flip-exit')
    if config.get('prime_hours', False):
        cmd.append('--prime-hours')
    if config.get('chase', False):
        cmd.append('--chase-entry')
        cmd.extend(['--chase-max-ticks', str(config.get('chase_max_ticks', 2))])
        cmd.extend(['--chase-max-reprices', str(config.get('chase_max_reprices', 5))])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return None
        with open(out_file) as f:
            return json.load(f)
    except Exception:
        return None


def run_sweep(pred_files, configs, workers=36):
    """Run full sweep."""
    jobs = []
    for config in configs:
        vg = config['vol_gate']
        for (date, vol), pred_file in pred_files.items():
            if vol == vg:
                jobs.append({
                    'date': date,
                    'pred_file': pred_file,
                    'config': config,
                })

    log.info(f"Running {len(jobs)} sim jobs ({len(configs)} configs × {len(set(d for d,_ in pred_files))} days) with {workers} workers")

    results = {}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            future = executor.submit(
                run_single_sim,
                job['date'], job['pred_file'], job['config']
            )
            futures[future] = job

        for future in as_completed(futures):
            done += 1
            job = futures[future]
            try:
                result = future.result()
                if result:
                    label = job['config']['label']
                    if label not in results:
                        results[label] = {'config': job['config'], 'days': {}}
                    results[label]['days'][job['date']] = result
            except Exception:
                pass

            if done % 200 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(jobs) - done) / max(rate, 0.01)
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f} jobs/s, ~{remaining:.0f}s remaining")

    elapsed = time.time() - t0
    log.info(f"Sweep complete: {done} jobs in {elapsed:.1f}s")
    return results


def aggregate(results):
    """Aggregate and rank all configs."""
    summary = []

    for label, data in results.items():
        config = data['config']
        days = data['days']

        total_pnl = 0
        total_trades = 0
        total_signals = 0
        total_wins = 0
        daily_pnls = []

        for date, res in sorted(days.items()):
            day_pnl = res.get('total_pnl_dollars', 0)
            day_trades = res.get('total_trades', 0)
            total_pnl += day_pnl
            total_trades += day_trades
            total_signals += res.get('total_signals', 0)
            daily_pnls.append(day_pnl)

            if 'trades' in res:
                for t in res['trades']:
                    if t.get('pnl_dollars', 0) > 0:
                        total_wins += 1

        n_days = len(days)
        if n_days == 0 or total_trades == 0:
            continue

        daily_arr = np.array(daily_pnls)
        mean_daily = np.mean(daily_arr)
        std_daily = np.std(daily_arr, ddof=1) if len(daily_arr) > 1 else 1e-8
        sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 1e-8 else 0

        fill_rate = total_trades / total_signals if total_signals > 0 else 0
        win_rate = total_wins / total_trades if total_trades > 0 else 0
        avg_trade = total_pnl / total_trades
        max_dd = compute_max_dd(daily_pnls)

        summary.append({
            'label': label,
            'total_pnl': round(total_pnl, 2),
            'n_days': n_days,
            'n_trades': total_trades,
            'n_signals': total_signals,
            'fill_rate': round(fill_rate, 4),
            'win_rate': round(win_rate, 4),
            'sharpe': round(sharpe, 2),
            'avg_trade_pnl': round(avg_trade, 2),
            'avg_trade_ticks': round(avg_trade / TICK_VALUE, 2),
            'max_dd': round(max_dd, 2),
            'avg_daily_pnl': round(mean_daily, 2),
            'config': config,
        })

    summary.sort(key=lambda x: x['total_pnl'], reverse=True)
    return summary


def compute_max_dd(daily_pnls):
    """Max drawdown from daily P&L series."""
    cumul = np.cumsum(daily_pnls)
    peak = np.maximum.accumulate(cumul)
    dd = peak - cumul
    return float(np.max(dd)) if len(dd) > 0 else 0


def report(summary):
    """Print results."""
    log.info("\n" + "=" * 120)
    log.info("MEGA SWEEP V2 RESULTS")
    log.info("=" * 120)

    # Top 30
    log.info(f"\nTop 30 configs by total P&L:")
    log.info(f"{'Config':<65} {'P&L':>9} {'Trades':>7} {'FillR':>6} {'WR':>6} {'Sharpe':>7} {'AvgT$':>8} {'MaxDD':>8}")
    log.info("-" * 120)
    for s in summary[:30]:
        log.info(f"{s['label']:<65} ${s['total_pnl']:>8,.0f} {s['n_trades']:>6} {s['fill_rate']:>5.1%} {s['win_rate']:>5.1%} {s['sharpe']:>7.2f} ${s['avg_trade_pnl']:>7.2f} ${s['max_dd']:>7,.0f}")

    # Bottom 5
    log.info(f"\nBottom 5:")
    for s in summary[-5:]:
        log.info(f"{s['label']:<65} ${s['total_pnl']:>8,.0f} {s['n_trades']:>6} {s['sharpe']:>7.2f}")

    # Best by category
    categories = {}
    for s in summary:
        label = s['label']
        if 'trail' in label or 'tp' in label:
            cat = 'exits'
        elif 'sigflip' in label:
            cat = 'signal_flip'
        elif 'prime' in label:
            cat = 'prime_hours'
        elif 'chase' in label:
            cat = 'chase'
        else:
            cat = 'passive'
        if cat not in categories:
            categories[cat] = s

    log.info(f"\nBest by category:")
    for cat, s in categories.items():
        log.info(f"  {cat:<15}: {s['label']:<55} Sharpe={s['sharpe']:>6.2f} P&L=${s['total_pnl']:>8,.0f}")

    # Save
    out_file = RESULTS_DIR / f'mega_sweep_v2_summary_{_ts}.json'
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nFull results saved: {out_file}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=36)
    parser.add_argument('--batch', default='all', choices=['all', 'core', 'exits', 'signal_flip', 'prime_hours', 'holds'])
    parser.add_argument('--dry-run', action='store_true', help='Just count configs, dont run')
    args = parser.parse_args()

    configs = generate_configs(args.batch)
    log.info(f"Generated {len(configs)} configs for batch '{args.batch}'")

    if args.dry_run:
        for c in configs[:20]:
            log.info(f"  {c['label']}")
        log.info(f"  ... and {len(configs) - 20} more")
        return

    pred_files = discover_prediction_files()
    n_dates = len(set(d for d, _ in pred_files))
    log.info(f"Found {len(pred_files)} prediction files ({n_dates} dates)")

    if not pred_files:
        log.error("No prediction files found! Run cnn_oot_sim_validation.py first.")
        return

    results = run_sweep(pred_files, configs, workers=args.workers)
    summary = aggregate(results)
    report(summary)


if __name__ == '__main__':
    main()
