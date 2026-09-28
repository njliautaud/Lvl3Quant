"""
Run ALL microstructure signal strategies through REAL MBO fill sim.
==================================================================
Generates signal prediction arrays from pre-computed features,
feeds them to the Rust MBO fill simulator, collects results.

Each strategy runs over 30+ consecutive days = live-style walk-forward.

Usage (on Jupiter):
    python3 alpha_discovery/run_all_signal_strategies.py --workers 12
    python3 alpha_discovery/run_all_signal_strategies.py --workers 12 --strategies microprice_dev,pressure_imbalance
"""

import json
import subprocess
import tempfile
import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
BINARY = ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR = ROOT / 'mbo'
SNAP_DIR = ROOT / 'data' / 'processed' / 'medium_snapshots_cache'
SIGNAL_DIR = ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SIGNAL DEFINITIONS
# ============================================================================
# Each signal: extract from global_features (96 cols from snapshots cache)
# Feature map (from src/features/engineering.py):
#   Base (0-9):   [mid, spread, imbalance, microprice, total_bid, total_ask, avg_bid_sz, avg_ask_sz, best_bid, best_ask]
#   Flow (10-17): [trade_imbalance, buy_vol, sell_vol, add_count, cancel_count, trade_count, cancel_to_add, trade_to_add]
#   Micro(18-25): [bid_pressure, ask_pressure, pressure_imbalance, depth_concentration, bid_slope, ask_slope, spread_ticks, depth_ratio]

def signal_imbalance(gf, mid):
    """Order book size imbalance. IC ~0.056 @ 1s."""
    return gf[:, 2]

def signal_microprice_dev(gf, mid):
    """Microprice deviation from mid. IC ~0.195 @ 1s, 92% consistency."""
    return gf[:, 3] - mid

def signal_trade_imbalance(gf, mid):
    """Trade flow imbalance. IC ~0.002 (weak)."""
    return gf[:, 10]

def signal_pressure_imbalance(gf, mid):
    """Distance-weighted book pressure imbalance. IC ~0.135 @ 1s, 86% consistency."""
    return gf[:, 20]

def signal_bid_slope(gf, mid):
    """Bid depth slope (negative = bullish). IC ~-0.083 @ 1s, 88% consistency."""
    return -gf[:, 22]  # Negate so positive = bullish

def signal_ask_slope(gf, mid):
    """Ask depth slope. IC ~0.084 @ 1s, 96% consistency."""
    return gf[:, 23]

def signal_composite(gf, mid):
    """Weighted composite of top signals. IC should be higher than any individual."""
    microprice = gf[:, 3] - mid
    pressure = gf[:, 20]
    ask_sl = gf[:, 23]
    bid_sl = -gf[:, 22]
    imb = gf[:, 2]

    # Normalize each to zero mean, unit std
    def norm(x):
        s = np.std(x)
        return (x - np.mean(x)) / max(s, 1e-8) if s > 1e-8 else x * 0

    # Weight by IC strength
    return (0.35 * norm(microprice) +
            0.25 * norm(pressure) +
            0.15 * norm(ask_sl) +
            0.15 * norm(bid_sl) +
            0.10 * norm(imb))

STRATEGIES = {
    'microprice_dev':      signal_microprice_dev,
    'pressure_imbalance':  signal_pressure_imbalance,
    'ask_slope':           signal_ask_slope,
    'bid_slope':           signal_bid_slope,
    'imbalance':           signal_imbalance,
    'composite':           signal_composite,
}

# ============================================================================
# PARAMETER GRID
# ============================================================================
PARAM_GRID = {
    'thresholds': [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    'hold_ms':    [5000, 10000, 30000, 60000],
    'trailing':   [4, 8, 12],
    'signal_flip': [False, True],   # Exit on signal flip
    'prime_hours': [False],         # Restrict to 10:30-14:30 ET (test separately)
}


def generate_signal_file(date_str: str, strategy_name: str, signal_fn) -> Path:
    """Generate a predictions NPZ from pre-computed features for one day."""
    out_path = SIGNAL_DIR / f'{strategy_name}_{date_str}.npz'
    if out_path.exists():
        return out_path

    snap_file = SNAP_DIR / f'{date_str}_snapshots.npz'
    if not snap_file.exists():
        return None

    data = np.load(str(snap_file))
    gf = data['global_features']
    mid = data['mid_prices']

    signal = signal_fn(gf, mid).astype(np.float64)

    # Save as predictions NPZ (compatible with Rust binary)
    np.savez_compressed(str(out_path), predictions=signal, mid_prices=mid)
    return out_path


def run_one_combo(date_str: str, strategy_name: str, thresh: float,
                  hold_ms: int, trail: int, signal_flip: bool = False,
                  prime_hours: bool = False) -> dict:
    """Run one day/strategy/param combo through the Rust MBO sim."""
    date_nodash = date_str.replace('-', '')
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_nodash}.mbo.dbn.zst'
    pred_file = SIGNAL_DIR / f'{strategy_name}_{date_str}.npz'

    flip_tag = '_flip' if signal_flip else ''
    prime_tag = '_prime' if prime_hours else ''
    combo_id = f'{strategy_name}_t{int(thresh*100)}_h{hold_ms//1000}s_tr{trail}{flip_tag}{prime_tag}'
    out_file = RESULTS_DIR / f'sig_{combo_id}_{date_str}.json'

    if not mbo_file.exists():
        return {'date': date_str, 'combo_id': combo_id, 'error': f'No MBO file: {date_nodash}'}
    if not pred_file or not pred_file.exists():
        return {'date': date_str, 'combo_id': combo_id, 'error': 'No signal file'}

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--signal-threshold', str(thresh),
        '--hold-ms', str(hold_ms),
        '--trailing-ticks', str(trail),
        '--quiet',
    ]
    if signal_flip:
        cmd.append('--signal-flip-exit')
    if prime_hours:
        cmd.append('--prime-hours')

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return {'date': date_str, 'combo_id': combo_id, 'error': result.stderr[:200]}

        if out_file.exists():
            with open(out_file) as f:
                data = json.load(f)
            return {'date': date_str, 'combo_id': combo_id, 'data': data}
        else:
            return {'date': date_str, 'combo_id': combo_id, 'error': 'No output file'}
    except subprocess.TimeoutExpired:
        return {'date': date_str, 'combo_id': combo_id, 'error': 'TIMEOUT'}
    except Exception as e:
        return {'date': date_str, 'combo_id': combo_id, 'error': str(e)}


def aggregate_results(results_by_combo: dict, all_dates: list) -> list:
    """Aggregate per-day results into strategy summaries."""
    summaries = []

    for combo_id, day_results in results_by_combo.items():
        daily_pnls = []
        total_trades = 0
        total_signals = 0
        total_fills = 0

        for d in all_dates:
            dr = day_results.get(d, {})
            pnl = dr.get('total_pnl_dollars', dr.get('pnl_dollars', 0))
            daily_pnls.append(pnl)
            total_trades += dr.get('total_trades', dr.get('n_trades', 0))
            total_signals += dr.get('total_signals', dr.get('n_orders_posted', 0))
            total_fills += dr.get('total_filled', dr.get('n_fills', 0))

        dp = np.array(daily_pnls)
        total_pnl = float(dp.sum())
        profitable_days = int((dp > 0).sum())
        std = float(np.std(dp)) if len(dp) > 1 else 1.0
        sharpe = float(np.mean(dp) / max(std, 0.01) * np.sqrt(252)) if std > 0 else 0.0
        avg_daily = float(np.mean(dp))
        max_dd = float(np.min(np.minimum.accumulate(np.cumsum(dp)) - np.cumsum(dp)))

        # Parse strategy name and params from combo_id
        parts = combo_id.split('_')

        summaries.append({
            'combo_id': combo_id,
            'total_pnl': round(total_pnl, 2),
            'avg_daily_pnl': round(avg_daily, 2),
            'sharpe': round(sharpe, 2),
            'profitable_days': profitable_days,
            'total_days': len(all_dates),
            'win_rate': round(profitable_days / max(len(all_dates), 1), 4),
            'total_trades': total_trades,
            'total_signals': total_signals,
            'total_fills': total_fills,
            'fill_rate': round(total_fills / max(total_signals, 1), 4),
            'max_drawdown': round(max_dd, 2),
            'daily_pnls': [round(float(p), 2) for p in daily_pnls],
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-days', type=int, default=100)
    parser.add_argument('--strategies', type=str, default=None,
                        help='Comma-separated strategy names (default: all)')
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: Rust binary not found: {BINARY}")
        print("Build with: cd rust_cache_builder && cargo build --release")
        exit(1)

    # Discover available dates (must have both MBO + snapshots)
    snap_dates = set()
    for f in SNAP_DIR.glob('*_snapshots.npz'):
        date_str = f.stem.replace('_snapshots', '')
        snap_dates.add(date_str)

    mbo_dates = set()
    for f in MBO_DIR.glob('*.zst'):
        # Parse date from filename: glbx-mdp3-YYYYMMDD.mbo.dbn.zst
        name = f.stem  # glbx-mdp3-YYYYMMDD.mbo.dbn
        parts = name.split('-')
        if len(parts) >= 3:
            d = parts[2][:8]
            if len(d) == 8:
                date_str = f'{d[:4]}-{d[4:6]}-{d[6:8]}'
                mbo_dates.add(date_str)

    available_dates = sorted(snap_dates & mbo_dates)[:args.max_days]
    print(f"Available dates: {len(available_dates)} (snapshots: {len(snap_dates)}, MBO: {len(mbo_dates)})")

    if not available_dates:
        print("ERROR: No dates with both snapshots and MBO data!")
        exit(1)

    # Select strategies
    if args.strategies:
        strat_names = args.strategies.split(',')
    else:
        strat_names = list(STRATEGIES.keys())

    print(f"Strategies: {strat_names}")
    print(f"Parameter grid: {len(PARAM_GRID['thresholds'])} thresh x {len(PARAM_GRID['hold_ms'])} hold x {len(PARAM_GRID['trailing'])} trail = {len(PARAM_GRID['thresholds'])*len(PARAM_GRID['hold_ms'])*len(PARAM_GRID['trailing'])} combos per strategy")
    print(f"Workers: {args.workers}")
    print()

    # Step 1: Generate signal prediction files
    print("=== Step 1: Generating signal prediction files ===")
    for sname in strat_names:
        fn = STRATEGIES[sname]
        generated = 0
        for d in available_dates:
            result = generate_signal_file(d, sname, fn)
            if result:
                generated += 1
        print(f"  {sname}: {generated}/{len(available_dates)} signal files ready")

    # Step 2: Build job list
    print("\n=== Step 2: Running MBO fill sim ===")
    jobs = []
    for sname in strat_names:
        for thresh in PARAM_GRID['thresholds']:
            for hold in PARAM_GRID['hold_ms']:
                for trail in PARAM_GRID['trailing']:
                    for flip in PARAM_GRID['signal_flip']:
                        for prime in PARAM_GRID['prime_hours']:
                            for d in available_dates:
                                jobs.append((d, sname, thresh, hold, trail, flip, prime))

    total_jobs = len(jobs)
    combos_per_strat = (len(PARAM_GRID['thresholds']) * len(PARAM_GRID['hold_ms']) *
                        len(PARAM_GRID['trailing']) * len(PARAM_GRID['signal_flip']) *
                        len(PARAM_GRID['prime_hours']))
    print(f"Total jobs: {total_jobs} ({len(strat_names)} strategies x {combos_per_strat} combos x {len(available_dates)} days)")

    # Step 3: Run all jobs
    t_start = time.time()
    results_by_combo = defaultdict(dict)
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one_combo, *job): job for job in jobs}

        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                combo_id = result.get('combo_id', 'unknown')
                date_str = result.get('date', 'unknown')

                if 'error' in result:
                    errors += 1
                    if errors <= 10:
                        print(f"  ERROR [{combo_id}] {date_str}: {result['error'][:80]}")
                elif 'data' in result:
                    results_by_combo[combo_id][date_str] = result['data']
            except Exception as e:
                errors += 1

            if done % 200 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (total_jobs - done) / max(rate, 0.1)
                print(f"  [{done}/{total_jobs}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {errors} errors")

    elapsed = time.time() - t_start
    print(f"\nCompleted {done} jobs in {elapsed:.0f}s ({done/max(elapsed,1):.1f} jobs/sec), {errors} errors")

    # Step 4: Aggregate and save
    print("\n=== Step 3: Aggregating results ===")
    summaries = aggregate_results(results_by_combo, available_dates)

    output = {
        'experiment': 'microstructure_signal_strategies',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'meta': {
            'strategies': strat_names,
            'dates': available_dates,
            'n_days': len(available_dates),
            'n_combos_per_strategy': combos_per_strat,
            'total_jobs': total_jobs,
            'elapsed_secs': round(elapsed, 1),
            'errors': errors,
        },
        'results': summaries,
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'signal_strategies_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Print results
    print(f"\n{'='*90}")
    print(f"MICROSTRUCTURE SIGNAL STRATEGY RESULTS — {len(available_dates)} days, REAL MBO fills")
    print(f"{'='*90}")

    profitable = [s for s in summaries if s['total_pnl'] > 0]
    print(f"\nProfitable combos: {len(profitable)}/{len(summaries)}")

    print(f"\nTOP 15 (by Sharpe):")
    print(f"  {'#':>3s}  {'Combo':>40s}  {'Sharpe':>7s}  {'PnL':>10s}  {'Trades':>7s}  {'WR':>6s}  {'Fills':>6s}  {'Days+':>5s}")
    print(f"  {'-'*85}")
    for i, s in enumerate(summaries[:15]):
        marker = ' <<<' if s['total_pnl'] > 0 else ''
        print(f"  {i+1:3d}  {s['combo_id']:>40s}  {s['sharpe']:>+7.2f}  "
              f"${s['total_pnl']:>+9,.2f}  {s['total_trades']:>7d}  "
              f"{s['win_rate']:>5.1%}  {s['fill_rate']:>5.1%}  "
              f"{s['profitable_days']:>3d}/{s['total_days']}{marker}")

    if profitable:
        print(f"\n{'='*90}")
        print("ALL PROFITABLE STRATEGIES:")
        print(f"{'='*90}")
        for s in profitable:
            print(f"  {s['combo_id']:>40s}  Sharpe={s['sharpe']:>+.2f}  "
                  f"PnL=${s['total_pnl']:>+,.2f}  "
                  f"Trades={s['total_trades']}  WR={s['win_rate']:.1%}  "
                  f"Days+={s['profitable_days']}/{s['total_days']}")

    print(f"\nSaved: {out_file}")
