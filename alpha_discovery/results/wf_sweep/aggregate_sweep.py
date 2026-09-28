#!/usr/bin/env python3
"""
aggregate_sweep.py -- Aggregate walk-forward fill_sim sweep results
====================================================================
Reads all per-date JSON result files in the wf_sweep directory, aggregates
by config (stripping the trailing date), computes portfolio-level Sharpe,
ranks by Sharpe, shows parameter sensitivity, and saves summary JSON.

Can be run while the sweep is still in progress for interim results.

Usage:
    python aggregate_sweep.py
    python aggregate_sweep.py --min_days 5   # require at least N days
    python aggregate_sweep.py --top 30       # show top N configs
    python aggregate_sweep.py --out custom_summary.json
"""

import os
import re
import sys
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
# wf_sweep/ dir is the same directory this script lives in
SWEEP_DIR = SCRIPT_DIR

# ── Filename regex ─────────────────────────────────────────────────────────
# Format: wf_v{vol}_c{conv}_h{hold}m_{mode}_ct{ct}r{cr}_lat{lat}_{YYYY-MM-DD}.json
# Example: wf_v70_c2.5_h30m_chase_ct1r3_lat0_2025-12-01.json
#          wf_v50_c1.5_h5m_passive_ct0r0_lat0_2025-12-01.json
FILENAME_RE = re.compile(
    r'^wf_v(\d+)_c([\d.]+)_h(\d+)m_(chase|passive)_ct(\d+)r(\d+)_lat(\d+)_(\d{4}-\d{2}-\d{2})\.json$'
)


def parse_config_from_filename(filename: str):
    """Returns (config_key, params_dict, date) or None if no match."""
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    vol, conv, hold, mode, ct, cr, lat, date = m.groups()
    params = {
        'vol_gate': int(vol),
        'conv': float(conv),
        'hold_min': int(hold),
        'mode': mode,
        'chase_ticks': int(ct),
        'chase_reprices': int(cr),
        'latency_ms': int(lat),
    }
    config_key = f'wf_v{vol}_c{conv}_h{hold}m_{mode}_ct{ct}r{cr}_lat{lat}'
    return config_key, params, date


def std_dev(values):
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_results(sweep_dir: Path, min_days: int = 1):
    """
    Read all per-date JSON files, aggregate by config key.
    Returns list of result dicts sorted by portfolio Sharpe (descending).
    """
    configs = defaultdict(lambda: {
        'days': 0,
        'params': None,
        'total_pnl': 0.0,
        'total_trades': 0,
        'total_signals': 0,
        'total_filled': 0,
        'total_cancelled': 0,
        'wins': 0,
        'daily_pnls': [],
        'fill_rates': [],
        'avg_queue_positions': [],
        'avg_fill_latencies_ms': [],
        'dates': [],
    })

    n_read = 0
    n_skipped = 0
    n_parse_err = 0

    for f in sweep_dir.iterdir():
        if not f.name.endswith('.json'):
            continue
        if f.name.startswith('aggregate_summary') or f.name.startswith('wf_sweep_summary'):
            continue

        parsed = parse_config_from_filename(f.name)
        if parsed is None:
            n_skipped += 1
            continue

        config_key, params, date = parsed

        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:
            n_parse_err += 1
            continue

        s = configs[config_key]
        if s['params'] is None:
            s['params'] = params

        pnl = data.get('total_pnl_dollars', 0) or 0
        trades = data.get('total_trades', 0) or 0
        signals = data.get('total_signals', 0) or 0
        filled = data.get('total_filled', 0) or 0
        cancelled = data.get('total_cancelled', 0) or 0
        win_rate = data.get('win_rate', 0) or 0
        fill_rate = data.get('fill_rate', 0) or 0
        avg_qp = data.get('avg_queue_position') or 0
        avg_lat = data.get('avg_fill_latency_ms') or 0

        s['days'] += 1
        s['total_pnl'] += pnl
        s['total_trades'] += trades
        s['total_signals'] += signals
        s['total_filled'] += filled
        s['total_cancelled'] += cancelled
        s['wins'] += int(round(trades * win_rate))
        s['daily_pnls'].append(pnl)
        s['fill_rates'].append(fill_rate)
        if avg_qp > 0:
            s['avg_queue_positions'].append(avg_qp)
        if avg_lat > 0:
            s['avg_fill_latencies_ms'].append(avg_lat)
        s['dates'].append(date)
        n_read += 1

    print(f'Read {n_read} JSON files | Skipped {n_skipped} non-matching | Parse errors: {n_parse_err}')
    print(f'Unique configs found: {len(configs)}')

    # Compute portfolio-level Sharpe and rank
    results = []
    for ck, s in configs.items():
        if s['days'] < min_days:
            continue

        daily = s['daily_pnls']
        m = mean(daily)
        sd = std_dev(daily)
        # Annualized Sharpe (252 trading days)
        sharpe = (m / (sd + 1e-8)) * math.sqrt(252) if sd > 0 else 0.0

        wr = s['wins'] / max(s['total_trades'], 1)
        fr = mean(s['fill_rates']) if s['fill_rates'] else 0
        avg_qp = mean(s['avg_queue_positions']) if s['avg_queue_positions'] else 0
        avg_lat = mean(s['avg_fill_latencies_ms']) if s['avg_fill_latencies_ms'] else 0

        results.append({
            'config': ck,
            'params': s['params'],
            'days': s['days'],
            'total_pnl': round(s['total_pnl'], 2),
            'annualized_pnl': round(m * 252, 0),
            'daily_pnl_mean': round(m, 2),
            'daily_pnl_std': round(sd, 2),
            'sharpe': round(sharpe, 3),
            'trades': s['total_trades'],
            'trades_per_day': round(s['total_trades'] / s['days'], 1),
            'signals': s['total_signals'],
            'win_rate': round(wr, 4),
            'fill_rate': round(fr, 4),
            'avg_queue_position': round(avg_qp, 1),
            'avg_fill_latency_ms': round(avg_lat, 1),
            'dates_covered': sorted(s['dates']),
        })

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    return results


def param_sensitivity(results, param_name):
    """Group results by a param level and show mean Sharpe per level."""
    by_level = defaultdict(list)
    for r in results:
        val = r['params'].get(param_name)
        if val is not None:
            by_level[val].append(r['sharpe'])
    # Sort by level value
    sorted_levels = sorted(by_level.items(), key=lambda x: x[0])
    return [(level, round(mean(sharpes), 3), len(sharpes))
            for level, sharpes in sorted_levels]


def print_leaderboard(results, top_n=20):
    print()
    print('=' * 130)
    print(f'LEADERBOARD — Top {top_n} Configs by Portfolio Sharpe (across {results[0]["days"] if results else 0} days each)')
    print('=' * 130)
    hdr = (f'{"#":>3} {"Config":<65} {"Days":>4} {"P&L":>10} {"Ann.P&L":>10} '
           f'{"Sharpe":>7} {"Trades":>7} {"T/Day":>5} {"WR":>6} {"FillR":>6} {"AvgQ":>5}')
    print(hdr)
    print('-' * 130)
    for i, r in enumerate(results[:top_n]):
        print(
            f'{i+1:>3} {r["config"]:<65} {r["days"]:>4} '
            f'${r["total_pnl"]:>9,.0f} ${r["annualized_pnl"]:>9,.0f} '
            f'{r["sharpe"]:>7.3f} {r["trades"]:>7} {r["trades_per_day"]:>5.1f} '
            f'{r["win_rate"]:>5.1%} {r["fill_rate"]:>5.1%} {r["avg_queue_position"]:>5.1f}'
        )
    print()


def print_sensitivity(results):
    print('=' * 80)
    print('PARAMETER SENSITIVITY — Mean Sharpe by Parameter Level')
    print('=' * 80)

    params = [
        ('vol_gate', 'Vol Gate (pct)'),
        ('conv', 'Signal Conv Threshold'),
        ('hold_min', 'Hold Time (minutes)'),
        ('chase_ticks', 'Chase Max Ticks'),
        ('chase_reprices', 'Chase Max Reprices'),
        ('latency_ms', 'Latency (ms)'),
        ('mode', 'Mode (passive/chase)'),
    ]

    for param_key, param_label in params:
        sens = param_sensitivity(results, param_key)
        print(f'\n  {param_label}:')
        for level, avg_sharpe, n_configs in sens:
            bar_len = max(0, int((avg_sharpe + 3) * 5))
            bar = '█' * min(bar_len, 40)
            print(f'    {str(level):>8}  Sharpe={avg_sharpe:>7.3f}  ({n_configs} configs)  {bar}')
    print()


def main():
    parser = argparse.ArgumentParser(description='Aggregate wf_sweep fill_sim results')
    parser.add_argument('--dir', type=str, default=str(SWEEP_DIR),
                        help='Directory containing JSON result files')
    parser.add_argument('--min_days', type=int, default=1,
                        help='Minimum days covered to include a config (default: 1)')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top configs to display (default: 20)')
    parser.add_argument('--out', type=str, default='aggregate_summary.json',
                        help='Output JSON summary filename (default: aggregate_summary.json)')
    args = parser.parse_args()

    sweep_dir = Path(args.dir)
    if not sweep_dir.exists():
        print(f'ERROR: Directory not found: {sweep_dir}')
        sys.exit(1)

    print(f'Aggregating results from: {sweep_dir}')
    print(f'Min days filter: {args.min_days}')
    print()

    results = aggregate_results(sweep_dir, min_days=args.min_days)

    if not results:
        print('No results found (check --min_days or directory path).')
        sys.exit(0)

    print(f'Total configs with >= {args.min_days} day(s): {len(results)}')

    print_leaderboard(results, top_n=args.top)
    print_sensitivity(results)

    # Save full summary (top 500 + metadata)
    out_path = sweep_dir / args.out
    summary = {
        'generated_at': __import__('datetime').datetime.now().isoformat(),
        'sweep_dir': str(sweep_dir),
        'total_json_files_read': sum(r['days'] for r in results),
        'total_configs': len(results),
        'min_days_filter': args.min_days,
        'top_configs': results[:500],
    }
    # Strip dates_covered from saved output to keep file size manageable
    for r in summary['top_configs']:
        r.pop('dates_covered', None)

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to: {out_path}')
    print(f'  Total configs: {len(results)}')
    print(f'  Files read: {summary["total_json_files_read"]}')
    if results:
        best = results[0]
        print(f'  Best config: {best["config"]}')
        print(f'    Sharpe: {best["sharpe"]:.3f} | P&L: ${best["total_pnl"]:,.0f} | '
              f'Ann: ${best["annualized_pnl"]:,.0f} | Days: {best["days"]}')


if __name__ == '__main__':
    main()
