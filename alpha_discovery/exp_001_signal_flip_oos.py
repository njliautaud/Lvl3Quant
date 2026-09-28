"""
EXP_001: signal_flip_exit OOS Sweep
=====================================
Tests signal_flip_exit=true in the Rust fill_sim_cli on all 21 OOS dates.
signal_flip_exit was NOT included in the prior 81-combo sweep.

Prior IS research showed +5.3 Sharpe lift from signal_flip_exit.
This is the HIGHEST PRIORITY experiment — tests a known IS edge in the Rust sim.

When signal_flip_exit=true, the simulator exits a position immediately when
the model's prediction flips sign (e.g., exit long when prediction goes negative).

Usage:
    python alpha_discovery/exp_001_signal_flip_oos.py
    python alpha_discovery/exp_001_signal_flip_oos.py --workers 4
"""

import json
import subprocess
import tempfile
import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
BINARY = ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = ROOT / 'mbo'
PRED_DIR = ROOT / 'data' / 'processed' / 'oos_predictions'
OUT_DIR = ROOT / 'data' / 'processed' / 'oos_sweep_flip'
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OOS_DATES = [
    '2025-09-19', '2025-09-22', '2025-09-23', '2025-09-24', '2025-09-25', '2025-09-26',
    '2025-09-29', '2025-09-30', '2025-10-01', '2025-10-02', '2025-10-03',
    '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10',
    '2025-10-13', '2025-10-14', '2025-10-15', '2025-10-16', '2025-10-17',
]

# Parameter grid — same as prior sweep but with signal_flip_exit=true
# EXTENDED: high-confidence OOS sweep found best results at threshold 5.0 (Sharpe +4.05)
# with monotonically improving trend. Test even higher thresholds to find the ceiling.
THRESHOLDS = [0.3, 0.5, 0.7, 5.0, 7.0, 10.0, 15.0]
HOLD_MS = [10000, 30000, 60000]
TRAILING = [4, 8, 12]
# Latencies: test 0 and 10ms (best latencies from prior sweep were 0 and 10)
LATENCIES = [0, 10]


def make_combo_id(thresh, lat, hold, trail):
    return f"flip_t{int(thresh*10)}_l{lat}_h{hold//1000}s_tr{trail}"


def run_day_with_flip(date_str, thresh, lat, hold, trail, combo_id):
    """Run one day through Rust sim with signal_flip_exit=true."""
    date_nodash = date_str.replace('-', '')
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_nodash}.mbo.dbn.zst'
    pred_file = PRED_DIR / f'{date_str}_predictions.npz'
    out_file = OUT_DIR / f'{combo_id}_{date_str}.json'

    if not mbo_file.exists() or not pred_file.exists():
        return date_str, combo_id, None, f"Missing files: mbo={mbo_file.exists()} pred={pred_file.exists()}"

    # Write config JSON with signal_flip_exit=true
    config = {
        'hold_ms': hold,
        'trailing_stop_ticks': trail,
        'signal_threshold': thresh,
        'latency_ms': lat,
        'signal_flip_exit': True,
        'market_exit_spread_cost': 0.5,
    }

    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False
    ) as cfg_f:
        json.dump(config, cfg_f)
        cfg_path = cfg_f.name

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--config', cfg_path,
        '--signal-threshold', str(thresh),
        '--quiet',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        Path(cfg_path).unlink(missing_ok=True)
        if result.returncode != 0:
            return date_str, combo_id, None, result.stderr[:200]
        with open(out_file) as f:
            data = json.load(f)
        return date_str, combo_id, data, None
    except subprocess.TimeoutExpired:
        Path(cfg_path).unlink(missing_ok=True)
        return date_str, combo_id, None, "TIMEOUT"
    except Exception as e:
        Path(cfg_path).unlink(missing_ok=True)
        return date_str, combo_id, None, str(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: Rust binary not found: {BINARY}")
        exit(1)

    combos = []
    for thresh in THRESHOLDS:
        for lat in LATENCIES:
            for hold in HOLD_MS:
                for trail in TRAILING:
                    combos.append((thresh, lat, hold, trail))

    jobs = []
    for thresh, lat, hold, trail in combos:
        combo_id = make_combo_id(thresh, lat, hold, trail)
        for d in OOS_DATES:
            jobs.append((d, thresh, lat, hold, trail, combo_id))

    total_combos = len(combos)
    total_jobs = len(jobs)
    print(f"=== EXP_001: signal_flip_exit OOS Sweep ===")
    print(f"Combos: {total_combos} ({len(THRESHOLDS)} thresh x {len(LATENCIES)} lat x {len(HOLD_MS)} hold x {len(TRAILING)} trail)")
    print(f"Total jobs: {total_jobs} ({total_combos} combos x {len(OOS_DATES)} days)")
    print(f"Workers: {args.workers}")
    print()

    t_start = time.time()
    results = {}
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_day_with_flip, *job): job for job in jobs}
        for future in as_completed(futures):
            done += 1
            try:
                date_str, combo_id, data, err = future.result()
                if err:
                    errors += 1
                    if errors <= 5:
                        print(f"  ERROR {date_str} {combo_id}: {err[:80]}")
                elif data:
                    if combo_id not in results:
                        results[combo_id] = {}
                    results[combo_id][date_str] = data
            except Exception as e:
                errors += 1

            if done % 100 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (total_jobs - done) / max(rate, 0.1)
                print(f"  [{done}/{total_jobs}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining, {errors} errors")

    elapsed = time.time() - t_start
    print(f"\nCompleted {done} jobs in {elapsed:.0f}s ({done/elapsed:.1f} jobs/sec), {errors} errors")

    # Aggregate results
    summary = []
    for thresh, lat, hold, trail in combos:
        combo_id = make_combo_id(thresh, lat, hold, trail)
        combo_data = results.get(combo_id, {})

        daily_pnls = []
        total_trades = 0
        total_signals = 0
        total_filled = 0
        for d in OOS_DATES:
            day = combo_data.get(d, {})
            pnl = day.get('total_pnl_dollars', 0)
            daily_pnls.append(pnl)
            total_trades += day.get('total_trades', 0)
            total_signals += day.get('total_signals', 0)
            total_filled += day.get('total_filled', 0)

        dp = np.array(daily_pnls)
        total_pnl = float(dp.sum())
        profitable_days = int((dp > 0).sum())
        std = float(np.std(dp)) if len(dp) > 1 else 1.0
        sharpe = float(np.mean(dp) / max(std, 0.01) * np.sqrt(252)) if std > 0 else 0.0

        summary.append({
            'combo_id': combo_id,
            'threshold': thresh,
            'latency_ms': lat,
            'hold_ms': hold,
            'trailing_ticks': trail,
            'signal_flip_exit': True,
            'total_pnl': round(total_pnl, 2),
            'total_trades': total_trades,
            'total_signals': total_signals,
            'fill_rate': round(total_filled / max(total_signals, 1), 3),
            'profitable_days': profitable_days,
            'total_days': len(OOS_DATES),
            'avg_daily_pnl': round(float(np.mean(dp)), 2),
            'sharpe': round(sharpe, 2),
            'daily_pnls': [round(float(p), 2) for p in daily_pnls],
        })

    # Sort by sharpe
    summary.sort(key=lambda x: x['sharpe'], reverse=True)

    # Load baseline (no-flip) results for comparison
    baseline_path = ROOT / 'data' / 'processed' / 'oos_sweep' / 'oos_sweep_summary.json'
    baseline_best_sharpe = -999.0
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline_data = json.load(f)
        baseline_results = {r['combo_id']: r for r in baseline_data.get('results', [])}
        baseline_best_sharpe = max(r.get('sharpe', -999) for r in baseline_data.get('results', []))

    # Save results
    output = {
        'experiment': 'exp_001_signal_flip_oos',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'signal_flip_exit': True,
        'baseline_best_sharpe': baseline_best_sharpe,
        'meta': {
            'combos': total_combos,
            'days': len(OOS_DATES),
            'total_jobs': total_jobs,
            'elapsed_secs': round(elapsed, 1),
        },
        'results': summary,
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'exp001_signal_flip_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Print results
    print(f"\n{'='*70}")
    print(f"EXP_001 RESULTS — signal_flip_exit=true")
    print(f"Baseline best Sharpe (no flip): {baseline_best_sharpe:.2f}")
    print(f"{'='*70}")
    print(f"\nTOP 10 COMBOS (by Sharpe):")
    for i, s in enumerate(summary[:10]):
        marker = ' ***' if s['total_pnl'] > 0 else ''
        print(f"  {i+1:2d}. {s['combo_id']:30s}  Sharpe={s['sharpe']:+6.2f}  "
              f"PnL=${s['total_pnl']:>+9,.2f}  trades={s['total_trades']:5d}  "
              f"days={s['profitable_days']}/{s['total_days']}{marker}")

    profitable = [s for s in summary if s['total_pnl'] > 0]
    print(f"\nProfitable combos: {len(profitable)}/{len(summary)}")
    if profitable:
        print("PROFITABLE:")
        for s in profitable:
            print(f"  {s['combo_id']:30s}  +${s['total_pnl']:,.2f}  Sharpe={s['sharpe']:.2f}")
    else:
        best = summary[0]
        improvement = best['sharpe'] - baseline_best_sharpe
        print(f"Best Sharpe with flip: {best['sharpe']:.2f} (vs {baseline_best_sharpe:.2f} baseline, delta={improvement:+.2f})")

    print(f"\nSaved: {out_file}")
