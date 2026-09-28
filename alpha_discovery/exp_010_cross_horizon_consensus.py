"""
EXP_010: Cross-Horizon Consensus Signal OOS Sweep
===================================================
The oos_predictions/ files contain 3s, 10s, and 30s predictions.
Prior sweep ONLY used predictions_30s. Tests consensus signals:

1. STRICT CONSENSUS: sign(30s)==sign(10s)==sign(3s), use abs(30s) as magnitude
2. 2/3 MAJORITY: at least 2 horizons agree, use abs(30s) as magnitude
3. WEIGHTED AVG: 0.5*30s + 0.3*10s + 0.2*3s (IC-weighted)
4. PRODUCT: 30s * |10s| * |3s| (high only when all horizons are strong)

When 3 independent models at different horizons all predict the same direction,
the signal should be lower frequency but higher quality.

Usage:
    python alpha_discovery/exp_010_cross_horizon_consensus.py
    python alpha_discovery/exp_010_cross_horizon_consensus.py --workers 4
"""

import json
import subprocess
import time
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
BINARY = ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = ROOT / 'mbo'
PRED_DIR = ROOT / 'data' / 'processed' / 'oos_predictions'
CONSENSUS_DIR = ROOT / 'data' / 'processed' / 'oos_predictions_consensus'
OUT_DIR = ROOT / 'data' / 'processed' / 'oos_sweep_consensus'
CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'

OOS_DATES = [
    '2025-09-19', '2025-09-22', '2025-09-23', '2025-09-24', '2025-09-25', '2025-09-26',
    '2025-09-29', '2025-09-30', '2025-10-01', '2025-10-02', '2025-10-03',
    '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10',
    '2025-10-13', '2025-10-14', '2025-10-15', '2025-10-16', '2025-10-17',
]

SIGNAL_TYPES = ['strict_3of3', 'majority_2of3', 'weighted_avg', 'product']
THRESHOLDS = [0.3, 0.5, 0.7]
HOLD_MS = [10000, 30000, 60000]
TRAILING = [4, 8, 12]
LATENCIES = [0, 10]


def create_consensus_predictions(date_str):
    """Create consensus signals from multi-horizon predictions."""
    pred_file = PRED_DIR / f'{date_str}_predictions.npz'
    if not pred_file.exists():
        return False

    data = np.load(pred_file)
    p30 = data['predictions_30s'].astype(np.float64)
    p10 = data['predictions_10s'].astype(np.float64)
    p3 = data['predictions_3s'].astype(np.float64)

    # 1. Strict 3-of-3 consensus
    sign30 = np.sign(p30)
    sign10 = np.sign(p10)
    sign3 = np.sign(p3)
    strict = p30 * ((sign30 == sign10) & (sign10 == sign3)).astype(float)

    # 2. 2-of-3 majority
    votes = sign30 + sign10 + sign3  # -3 to +3
    majority = p30 * (np.abs(votes) >= 2).astype(float)

    # 3. IC-weighted average (IC30~0.14, IC10~0.11, IC3~0.08 roughly)
    weighted = 0.5 * p30 + 0.3 * p10 + 0.2 * p3

    # 4. Product signal (high only when ALL are strong)
    # Normalize each to unit scale first
    def safe_normalize(arr):
        std = np.std(arr)
        if std < 1e-8:
            return arr
        return arr / std

    product = np.sign(p30) * np.abs(safe_normalize(p30) * safe_normalize(p10) * safe_normalize(p3)) ** (1/3)

    # Save all consensus types
    out_file = CONSENSUS_DIR / f'{date_str}_consensus.npz'
    np.savez_compressed(
        str(out_file),
        strict_3of3=strict,
        majority_2of3=majority,
        weighted_avg=weighted,
        product=product,
        predictions_30s=p30,  # Keep original for reference
    )

    # Also save separate NPZ files for each signal type (Rust sim expects 'predictions' key)
    for signal_name, signal_arr in [
        ('strict_3of3', strict),
        ('majority_2of3', majority),
        ('weighted_avg', weighted),
        ('product', product),
    ]:
        single_out = CONSENSUS_DIR / f'{date_str}_{signal_name}.npz'
        np.savez_compressed(str(single_out), predictions=signal_arr)

    return True


def run_day(date_str, signal_type, thresh, lat, hold, trail):
    """Run one day with a consensus signal type."""
    pred_file = CONSENSUS_DIR / f'{date_str}_{signal_type}.npz'
    if not pred_file.exists():
        return None

    date_nodash = date_str.replace('-', '')
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_nodash}.mbo.dbn.zst'
    if not mbo_file.exists():
        return None

    combo_id = f"{signal_type}_t{int(thresh*10)}_l{lat}_h{hold//1000}s_tr{trail}"
    out_file = OUT_DIR / f'{combo_id}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--hold-ms', str(hold),
        '--trailing-ticks', str(trail),
        '--signal-threshold', str(thresh),
        '--latency-ms', str(lat),
        '--quiet',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return None
        with open(out_file) as f:
            return (date_str, combo_id, signal_type, json.load(f))
    except Exception:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: Rust binary not found: {BINARY}")
        exit(1)

    # Pre-create consensus predictions
    print("Creating consensus prediction files...")
    for date_str in OOS_DATES:
        ok = create_consensus_predictions(date_str)
        if not ok:
            print(f"  WARNING: Could not create consensus for {date_str}")
    print("  Done.")

    # Build jobs
    jobs = []
    for signal_type in SIGNAL_TYPES:
        for thresh in THRESHOLDS:
            for lat in LATENCIES:
                for hold in HOLD_MS:
                    for trail in TRAILING:
                        for date_str in OOS_DATES:
                            jobs.append((date_str, signal_type, thresh, lat, hold, trail))

    print(f"\n=== EXP_010: Cross-Horizon Consensus OOS Sweep ===")
    print(f"Signal types: {SIGNAL_TYPES}")
    print(f"Total jobs: {len(jobs)}")

    t_start = time.time()
    results = {}
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_day, *job): job for job in jobs}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                if result:
                    date_str, combo_id, signal_type, data = result
                    if combo_id not in results:
                        results[combo_id] = {}
                    results[combo_id][date_str] = data
            except Exception:
                pass

            if done % 300 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (len(jobs) - done) / max(rate, 0.1)
                print(f"  [{done}/{len(jobs)}] {elapsed:.0f}s, ~{eta:.0f}s remaining")

    elapsed = time.time() - t_start
    print(f"\nCompleted {done} jobs in {elapsed:.0f}s")

    # Aggregate by signal type
    all_results = []
    for signal_type in SIGNAL_TYPES:
        for thresh in THRESHOLDS:
            for lat in LATENCIES:
                for hold in HOLD_MS:
                    for trail in TRAILING:
                        combo_id = f"{signal_type}_t{int(thresh*10)}_l{lat}_h{hold//1000}s_tr{trail}"
                        combo_data = results.get(combo_id, {})

                        daily_pnls = [combo_data.get(d, {}).get('total_pnl_dollars', 0) for d in OOS_DATES]
                        daily_trades = [combo_data.get(d, {}).get('total_trades', 0) for d in OOS_DATES]
                        dp = np.array(daily_pnls)
                        std = float(np.std(dp)) if len(dp) > 1 else 1.0
                        sharpe = float(np.mean(dp) / max(std, 0.01) * np.sqrt(252))

                        all_results.append({
                            'combo_id': combo_id,
                            'signal_type': signal_type,
                            'threshold': thresh,
                            'hold_ms': hold,
                            'trailing_ticks': trail,
                            'latency_ms': lat,
                            'total_pnl': round(float(dp.sum()), 2),
                            'avg_daily_trades': round(float(np.mean(daily_trades)), 1),
                            'profitable_days': int((dp > 0).sum()),
                            'sharpe': round(sharpe, 2),
                        })

    all_results.sort(key=lambda x: x['sharpe'], reverse=True)

    # Per-signal-type best
    print(f"\n{'='*70}")
    print(f"EXP_010 RESULTS — Cross-Horizon Consensus")
    print(f"{'='*70}")
    for signal_type in SIGNAL_TYPES:
        type_results = [r for r in all_results if r['signal_type'] == signal_type]
        best = type_results[0] if type_results else None
        profitable = [r for r in type_results if r['total_pnl'] > 0]
        if best:
            print(f"\n{signal_type:20s}: best_sharpe={best['sharpe']:+.2f}  "
                  f"pnl=${best['total_pnl']:>+9,.2f}  "
                  f"trades/day={best['avg_daily_trades']:.1f}  "
                  f"profitable={len(profitable)}/{len(type_results)}")
            print(f"  Best combo: {best['combo_id']}")

    # Overall top 10
    print(f"\nOVERALL TOP 10:")
    for i, r in enumerate(all_results[:10]):
        marker = ' ***' if r['total_pnl'] > 0 else ''
        print(f"  {i+1:2d}. {r['combo_id']:40s}  Sharpe={r['sharpe']:+.2f}  "
              f"PnL=${r['total_pnl']:>+9,.2f}{marker}")

    # Save
    output = {
        'experiment': 'exp_010_cross_horizon_consensus',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'signal_types': SIGNAL_TYPES,
        'results': all_results,
    }
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'exp010_consensus_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")
