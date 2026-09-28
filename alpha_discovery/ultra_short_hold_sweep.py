"""
Ultra-Short Hold Sweep — Test 1s and 2s holds
==============================================
The signal sweep showed IC=0.195 at 1 second but ALL longer holds (5-60s)
lose money in MBO sim. This script generates signals and tests whether
sub-3-second holds can capture the edge before it decays.

Key hypothesis: the IC is real but decays within 1-2 seconds. With a limit
order, you need to get filled AND exit within that window. This tests
the absolute fastest execution the MBO sim supports.

Runs on Jupiter/Saturn (needs Rust fill_sim_cli).

Usage:
    python alpha_discovery/ultra_short_hold_sweep.py --workers 8 --max-days 50
"""

import sys
import os
import json
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent.parent
SNAP_DIR = ROOT / 'data' / 'processed' / 'medium_snapshots_cache'
SIGNAL_DIR = ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
MBO_DIR = ROOT / 'mbo'
BINARY = ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'

# If on server, binary might be in workspace root
if not BINARY.exists():
    BINARY = ROOT / 'fill_sim_cli'

# Feature indices
IDX_MICROPRICE = 3
IDX_PRESSURE_IMBALANCE = 20

# Ultra-short hold parameters
HOLD_OPTIONS = [1000, 1500, 2000, 3000]  # 1s, 1.5s, 2s, 3s
TRAILING_OPTIONS = [2, 3, 4]  # Very tight trailing stops
THRESHOLD_OPTIONS = [0, 0.05, 0.1, 0.2, 0.3, 0.5]

SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_microprice_signal(date_str):
    """Generate microprice deviation signal for one day."""
    snap_file = SNAP_DIR / f'{date_str}_snapshots.npz'
    if not snap_file.exists():
        return None

    data = np.load(str(snap_file))
    gf = data['global_features']
    mid = data['mid_prices']

    # Simple microprice deviation — the strongest individual signal
    signal = (gf[:, IDX_MICROPRICE] - mid).astype(np.float64)

    out_path = SIGNAL_DIR / f'ush_microprice_{date_str}.npz'
    np.savez_compressed(str(out_path), predictions=signal, mid_prices=mid)
    return str(out_path)


def generate_combo_signal(date_str):
    """Generate combined microprice + pressure signal."""
    snap_file = SNAP_DIR / f'{date_str}_snapshots.npz'
    if not snap_file.exists():
        return None

    data = np.load(str(snap_file))
    gf = data['global_features']
    mid = data['mid_prices']

    microprice_dev = gf[:, IDX_MICROPRICE] - mid
    pressure = gf[:, IDX_PRESSURE_IMBALANCE]

    # Normalize
    def norm(x):
        s = np.std(x)
        return (x - np.mean(x)) / max(s, 1e-8) if s > 1e-8 else x * 0

    signal = (0.6 * norm(microprice_dev) + 0.4 * norm(pressure)).astype(np.float64)

    out_path = SIGNAL_DIR / f'ush_combo_{date_str}.npz'
    np.savez_compressed(str(out_path), predictions=signal, mid_prices=mid)
    return str(out_path)


def run_single_sim(mbo_file, pred_file, hold_ms, trailing, threshold, out_file):
    """Run a single Rust MBO sim."""
    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--signal-threshold', str(threshold),
        '--hold-ms', str(hold_ms),
        '--trailing-ticks', str(trailing),
        '--quiet',
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and Path(out_file).exists():
            with open(out_file) as f:
                return json.load(f)
    except:
        pass
    return None


def run_combo(args_tuple):
    """Worker function for parallel execution."""
    signal_type, date_str, hold_ms, trailing, threshold, pred_file = args_tuple

    mbo_date = date_str.replace('-', '')
    mbo_file = MBO_DIR / f'glbx-mdp3-{mbo_date}.mbo.dbn.zst'
    if not mbo_file.exists():
        return None

    combo_id = f'{signal_type}_t{int(threshold*100):02d}_h{hold_ms}ms_tr{trailing}'
    out_file = f'/tmp/ush_{combo_id}_{date_str}.json'

    result = run_single_sim(mbo_file, pred_file, hold_ms, trailing, threshold, out_file)
    if result is None:
        return None

    pnl = result.get('total_pnl_dollars', result.get('pnl_dollars', 0))
    trades = result.get('total_trades', result.get('n_trades', 0))
    fills = result.get('total_filled', result.get('n_fills', 0))
    signals = result.get('total_signals', result.get('n_orders_posted', 0))

    # Save result
    save_result = {
        'combo_id': combo_id,
        'signal_type': signal_type,
        'date': date_str,
        'hold_ms': hold_ms,
        'trailing_ticks': trailing,
        'threshold': threshold,
        'total_pnl_dollars': pnl,
        'total_trades': trades,
        'total_filled': fills,
        'total_signals': signals,
        'win_rate': result.get('win_rate', 0),
    }

    result_file = RESULTS_DIR / f'ush_{combo_id}_{date_str}.json'
    with open(result_file, 'w') as f:
        json.dump(save_result, f, indent=2)

    return save_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-days', type=int, default=50)
    parser.add_argument('--signal', type=str, default='microprice,combo')
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: Rust binary not found at {BINARY}")
        sys.exit(1)

    # Get available dates
    snap_dates = sorted(f.stem.replace('_snapshots', '')
                        for f in SNAP_DIR.glob('*_snapshots.npz'))[:args.max_days]
    print(f"Available dates: {len(snap_dates)}")

    signal_types = args.signal.split(',')

    # Step 1: Generate signal files
    print("\nStep 1: Generating signal files...")
    pred_files = {}  # (signal_type, date) -> pred_file_path

    for sig_type in signal_types:
        for date_str in snap_dates:
            if sig_type == 'microprice':
                path = generate_microprice_signal(date_str)
            elif sig_type == 'combo':
                path = generate_combo_signal(date_str)
            else:
                continue
            if path:
                pred_files[(sig_type, date_str)] = path

    print(f"  Generated {len(pred_files)} signal files")

    # Step 2: Build job list
    jobs = []
    for sig_type in signal_types:
        for date_str in snap_dates:
            key = (sig_type, date_str)
            if key not in pred_files:
                continue
            pred_file = pred_files[key]
            for hold in HOLD_OPTIONS:
                for trail in TRAILING_OPTIONS:
                    for thresh in THRESHOLD_OPTIONS:
                        jobs.append((sig_type, date_str, hold, trail, thresh, pred_file))

    n_combos = len(HOLD_OPTIONS) * len(TRAILING_OPTIONS) * len(THRESHOLD_OPTIONS) * len(signal_types)
    print(f"\nStep 2: Running MBO sim — {len(jobs)} jobs "
          f"({len(signal_types)} signals x {n_combos // len(signal_types)} combos x {len(snap_dates)} days)")
    print(f"  Workers: {args.workers}")

    # Step 3: Run in parallel
    completed = 0
    errors = 0
    results_by_combo = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_combo, job): job for job in jobs}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                combo_id = result['combo_id']
                if combo_id not in results_by_combo:
                    results_by_combo[combo_id] = []
                results_by_combo[combo_id].append(result)
            else:
                errors += 1

            if completed % 200 == 0:
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 1)
                remaining = (len(jobs) - completed) / max(rate, 0.01)
                print(f"  [{completed}/{len(jobs)}] {elapsed:.0f}s elapsed, "
                      f"~{remaining:.0f}s remaining, {errors} errors")

    # Step 4: Analyze results
    print(f"\n{'='*60}")
    print(f"ULTRA-SHORT HOLD RESULTS ({completed} jobs, {errors} errors)")
    print(f"{'='*60}")

    ranked = []
    for combo_id, combo_results in results_by_combo.items():
        if len(combo_results) < 3:
            continue
        pnls = [r['total_pnl_dollars'] for r in combo_results]
        trades = [r['total_trades'] for r in combo_results]
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls)
        profitable_days = sum(1 for p in pnls if p > 0)
        win_rate = profitable_days / len(pnls)

        ranked.append({
            'combo_id': combo_id,
            'days': len(combo_results),
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'profitable_days': profitable_days,
            'win_rate': win_rate,
            'avg_trades': np.mean(trades),
            'std_pnl': np.std(pnls),
        })

    ranked.sort(key=lambda x: x['avg_pnl'], reverse=True)

    print(f"\nTOP 15 by avg daily PnL:")
    for r in ranked[:15]:
        sharpe = r['avg_pnl'] / max(r['std_pnl'], 0.01)
        print(f"  {r['combo_id']:50s}  avg=${r['avg_pnl']:>+8,.2f}/day  "
              f"WR={r['win_rate']:.0%}  trades={r['avg_trades']:.0f}/day  "
              f"sharpe={sharpe:.2f}  days={r['days']}")

    profitable_combos = [r for r in ranked if r['avg_pnl'] > 0]
    print(f"\nProfitable combos: {len(profitable_combos)}/{len(ranked)}")
    if profitable_combos:
        best = profitable_combos[0]
        print(f"BEST: {best['combo_id']} → ${best['avg_pnl']:+.2f}/day, "
              f"{best['win_rate']:.0%} WR, {best['avg_trades']:.0f} trades/day")

    # Save summary
    ts = time.strftime('%Y%m%d_%H%M%S')
    summary_file = RESULTS_DIR / f'ush_summary_{ts}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'experiment': 'ultra_short_hold',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'total_jobs': len(jobs),
            'completed': completed,
            'errors': errors,
            'combos_tested': len(ranked),
            'profitable_combos': len(profitable_combos),
            'ranked': ranked[:50],
        }, f, indent=2)
    print(f"\nSaved: {summary_file}")
