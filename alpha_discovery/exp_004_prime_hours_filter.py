"""
EXP_004: Prime Hours Filter OOS Sweep
=======================================
Prior IS research showed 10:30-14:30 window gives +2.7 Sharpe lift.
This experiment pre-zeroes predictions outside that window before
feeding to fill_sim_cli — no model retraining needed.

RTH = 9:30-16:00 ET = 6.5h = 390 min = 23,400 bars @ 100ms
Prime: 10:30-14:30 = bars 600-2400 (60min offset to 240min offset from open)
Also tests: 10:30-13:00 (bars 600-2100) and 10:00-15:00 (bars 300-3300)

Usage:
    python alpha_discovery/exp_004_prime_hours_filter.py
    python alpha_discovery/exp_004_prime_hours_filter.py --workers 6
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
FILTERED_DIR = ROOT / 'data' / 'processed' / 'oos_predictions_filtered'
OUT_DIR = ROOT / 'data' / 'processed' / 'oos_sweep_filtered'
FILTERED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OOS_DATES = [
    '2025-09-19', '2025-09-22', '2025-09-23', '2025-09-24', '2025-09-25', '2025-09-26',
    '2025-09-29', '2025-09-30', '2025-10-01', '2025-10-02', '2025-10-03',
    '2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10',
    '2025-10-13', '2025-10-14', '2025-10-15', '2025-10-16', '2025-10-17',
]

# RTH bar count (100ms bars in 6.5h RTH session)
BARS_PER_SEC = 10
RTH_BARS = 234000  # observed from NPZ files (6.5h * 3600 * 10 = 234000)

# Time windows to test (start_bar, end_bar from RTH open)
# 1 bar = 100ms; 600 bars = 60 seconds = 1 minute
TIME_WINDOWS = {
    'full':         (0, RTH_BARS),           # No filter (baseline)
    'prime_10_14':  (600 * 60, 600 * 240),   # 10:30-14:30 (60-240 min)
    'prime_10_13':  (600 * 60, 600 * 210),   # 10:30-13:30
    'prime_1030_1400': (600 * 60, 600 * 270), # 10:30-15:00 (more relaxed)
    'midday_only':  (600 * 90, 600 * 210),   # 11:00-14:00
}

# Sweep params (use top params from prior sweep)
THRESHOLDS = [0.5, 0.7]
HOLD_MS = [30000, 60000]
TRAILING = [8, 12]
LATENCIES = [0, 10]


def create_filtered_predictions(date_str, window_name, start_bar, end_bar):
    """Load predictions, zero out outside window, save to filtered dir."""
    pred_file = PRED_DIR / f'{date_str}_predictions.npz'
    out_file = FILTERED_DIR / f'{date_str}_{window_name}_predictions.npz'

    if not pred_file.exists():
        return None

    if window_name == 'full':
        return pred_file  # No filtering needed

    if out_file.exists():
        return out_file  # Already created

    data = np.load(pred_file)
    preds = data['predictions'].copy()

    # Zero out predictions outside the window
    mask = np.ones(len(preds), dtype=bool)
    mask[:start_bar] = False
    mask[end_bar:] = False
    preds[~mask] = 0.0

    # Save with same format
    save_dict = {'predictions': preds}
    for key in data.keys():
        if key != 'predictions':
            save_dict[key] = data[key]

    np.savez_compressed(str(out_file), **save_dict)
    return out_file


def run_day(date_str, window_name, thresh, lat, hold, trail):
    """Run one day through Rust sim with filtered predictions."""
    start_bar, end_bar = TIME_WINDOWS[window_name]
    filtered_pred = create_filtered_predictions(date_str, window_name, start_bar, end_bar)

    if filtered_pred is None:
        return None

    date_nodash = date_str.replace('-', '')
    mbo_file = MBO_DIR / f'glbx-mdp3-{date_nodash}.mbo.dbn.zst'
    if not mbo_file.exists():
        return None

    combo_id = f"{window_name}_t{int(thresh*10)}_l{lat}_h{hold//1000}s_tr{trail}"
    out_file = OUT_DIR / f'{combo_id}_{date_str}.json'

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(filtered_pred),
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
            return (date_str, combo_id, json.load(f))
    except Exception:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    if not BINARY.exists():
        print(f"ERROR: Rust binary not found: {BINARY}")
        exit(1)

    # Pre-create filtered prediction files
    print("Pre-creating filtered prediction files...")
    for date_str in OOS_DATES:
        for window_name, (start, end) in TIME_WINDOWS.items():
            create_filtered_predictions(date_str, window_name, start, end)
    print("  Done.")

    # Build all jobs
    jobs = []
    for window_name in TIME_WINDOWS:
        for thresh in THRESHOLDS:
            for lat in LATENCIES:
                for hold in HOLD_MS:
                    for trail in TRAILING:
                        for date_str in OOS_DATES:
                            jobs.append((date_str, window_name, thresh, lat, hold, trail))

    print(f"=== EXP_004: Prime Hours Filter OOS Sweep ===")
    print(f"Windows: {list(TIME_WINDOWS.keys())}")
    print(f"Total jobs: {len(jobs)}")
    print()

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
                    date_str, combo_id, data = result
                    if combo_id not in results:
                        results[combo_id] = {}
                    results[combo_id][date_str] = data
            except Exception:
                pass

            if done % 200 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (len(jobs) - done) / max(rate, 0.1)
                print(f"  [{done}/{len(jobs)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - t_start
    print(f"\nCompleted {done} jobs in {elapsed:.0f}s")

    # Aggregate by window type
    window_summaries = {w: [] for w in TIME_WINDOWS}

    for window_name in TIME_WINDOWS:
        for thresh in THRESHOLDS:
            for lat in LATENCIES:
                for hold in HOLD_MS:
                    for trail in TRAILING:
                        combo_id = f"{window_name}_t{int(thresh*10)}_l{lat}_h{hold//1000}s_tr{trail}"
                        combo_data = results.get(combo_id, {})

                        daily_pnls = [combo_data.get(d, {}).get('total_pnl_dollars', 0) for d in OOS_DATES]
                        dp = np.array(daily_pnls)
                        total_pnl = float(dp.sum())
                        std = float(np.std(dp)) if len(dp) > 1 else 1.0
                        sharpe = float(np.mean(dp) / max(std, 0.01) * np.sqrt(252))
                        total_trades = sum(combo_data.get(d, {}).get('total_trades', 0) for d in OOS_DATES)

                        window_summaries[window_name].append({
                            'combo_id': combo_id,
                            'threshold': thresh,
                            'latency_ms': lat,
                            'hold_ms': hold,
                            'trailing_ticks': trail,
                            'total_pnl': round(total_pnl, 2),
                            'total_trades': total_trades,
                            'profitable_days': int((dp > 0).sum()),
                            'sharpe': round(sharpe, 2),
                        })

        window_summaries[window_name].sort(key=lambda x: x['sharpe'], reverse=True)

    # Save results
    output = {
        'experiment': 'exp_004_prime_hours_filter',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'window_definitions': {k: list(v) for k, v in TIME_WINDOWS.items()},
        'results_by_window': window_summaries,
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'exp004_prime_hours_{ts}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"EXP_004 RESULTS — Prime Hours Filter")
    print(f"{'='*70}")
    for window_name, combos in window_summaries.items():
        best = combos[0] if combos else None
        profitable = [c for c in combos if c['total_pnl'] > 0]
        if best:
            print(f"\n{window_name:20s}: best_sharpe={best['sharpe']:+.2f}  "
                  f"best_pnl=${best['total_pnl']:>+9,.2f}  "
                  f"profitable={len(profitable)}/{len(combos)}  "
                  f"({best['combo_id']})")

    print(f"\nSaved: {out_file}")
