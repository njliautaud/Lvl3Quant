#!/usr/bin/env python3
"""
mae_mfe_sweep.py -- MAE/MFE Data Collection + Stop Loss Sweep
=============================================================
Phase 1: Baseline (no SL) with TP15 + chase to collect per-trade MAE/MFE data
Phase 2: Fixed SL sweep with [5,8,10,15,20,25] ticks

Uses WF predictions (cnn_wf_sim_predictions) and OOT MBO data.
Two strategies: conv1.5_vol50 and conv2.5_vol70
"""

import os
import sys
import json
import time
import logging
import subprocess
import statistics
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT   = Path('/home/jupiter/Lvl3Quant')
BINARY      = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR     = LVL3_ROOT / 'mbo_oot'
PRED_DIR    = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_sim_predictions'
OUT_DIR     = LVL3_ROOT / 'data' / 'processed' / 'mae_mfe_sweep_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = OUT_DIR / f'mae_mfe_sweep_{ts}.log'

logging.basicConfig(
    format='%(asctime)s [mae_mfe] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(str(log_file), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('mae_mfe')

# Two strategies
STRATEGIES = [
    {'name': 'conv15_vol50', 'threshold': 1.5, 'vol': 'vol50'},
    {'name': 'conv25_vol70', 'threshold': 2.5, 'vol': 'vol70'},
]

# Common params (matching best known config)
BASE_PARAMS = {
    'hold_ms': 3600000,      # 60 min hold (long enough to capture full MAE/MFE)
    'max_wait_bars': 20,
    'chase_entry': True,
    'chase_max_ticks': 1,
    'chase_max_reprices': 3,
    'take_profit_ticks': 15,
}

# SL sweep values (Phase 2)
SL_TICKS_LIST = [5, 8, 10, 15, 20, 25]

WORKERS = 12


def find_date_pairs():
    """Find matching (prediction_file, mbo_file) pairs for each strategy."""
    pairs = {}
    for strat in STRATEGIES:
        vol = strat['vol']
        strat_pairs = []
        for f in sorted(PRED_DIR.glob(f'*_{vol}_morning_afternoon.npz')):
            date_str = f.stem.split('_')[0]  # YYYY-MM-DD
            nodash = date_str.replace('-', '')
            mbo = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
            if not mbo.exists():
                mbo = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn'
            if mbo.exists():
                strat_pairs.append((date_str, str(f), str(mbo)))
        pairs[strat['name']] = strat_pairs
    return pairs


def run_sim(pred_file, mbo_file, output_file, threshold, extra_args=None):
    """Run a single fill_sim_cli job."""
    cmd = [
        str(BINARY),
        '--mbo-file', mbo_file,
        '--predictions', pred_file,
        '--output', output_file,
        '--signal-threshold', str(threshold),
        '--hold-ms', str(BASE_PARAMS['hold_ms']),
        '--max-wait-bars', str(BASE_PARAMS['max_wait_bars']),
        '--chase-entry',
        '--chase-max-ticks', str(BASE_PARAMS['chase_max_ticks']),
        '--chase-max-reprices', str(BASE_PARAMS['chase_max_reprices']),
        '--take-profit-ticks', str(BASE_PARAMS['take_profit_ticks']),
        '--quiet',
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            log.warning(f"FAIL: {os.path.basename(output_file)}: {r.stderr[:200]}")
            return None
        with open(output_file) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        log.warning(f"TIMEOUT: {os.path.basename(output_file)}")
        return None
    except Exception as e:
        log.warning(f"ERROR: {os.path.basename(output_file)}: {e}")
        return None


def run_phase(phase_name, jobs, workers=WORKERS):
    """Run a batch of sim jobs in parallel."""
    log.info(f"=== {phase_name}: {len(jobs)} jobs, {workers} workers ===")
    results = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for job in jobs:
            fut = executor.submit(run_sim, **job)
            futures[fut] = job['output_file']

        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            ofile = futures[fut]
            if result:
                results.append(result)
                if done % 10 == 0 or done == len(jobs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(jobs) - done) / rate if rate > 0 else 0
                    log.info(f"  [{done}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    log.info(f"  {phase_name} complete: {len(results)}/{len(jobs)} succeeded in {elapsed:.0f}s")
    return results


def analyze_mae_mfe(results, label):
    """Analyze MAE/MFE from baseline results."""
    all_trades = []
    for r in results:
        for t in r.get('trades', []):
            all_trades.append(t)

    if not all_trades:
        log.info(f"  {label}: No trades to analyze")
        return

    maes = [t.get('mae_ticks', 0) for t in all_trades]
    mfes = [t.get('mfe_ticks', 0) for t in all_trades]
    pnls = [t.get('pnl_ticks', 0) for t in all_trades]

    log.info(f"\n=== {label} MAE/MFE Analysis ({len(all_trades)} trades) ===")
    log.info(f"  MAE: mean={statistics.mean(maes):.2f}t, median={statistics.median(maes):.2f}t, "
             f"p25={sorted(maes)[len(maes)//4]:.2f}t, p75={sorted(maes)[3*len(maes)//4]:.2f}t, "
             f"max={max(maes):.2f}t")
    log.info(f"  MFE: mean={statistics.mean(mfes):.2f}t, median={statistics.median(mfes):.2f}t, "
             f"p25={sorted(mfes)[len(mfes)//4]:.2f}t, p75={sorted(mfes)[3*len(mfes)//4]:.2f}t, "
             f"max={max(mfes):.2f}t")
    log.info(f"  PnL: mean={statistics.mean(pnls):.2f}t, total={sum(pnls):.2f}t, "
             f"win_rate={sum(1 for p in pnls if p > 0)/len(pnls)*100:.1f}%")

    # MAE percentiles for SL selection
    sorted_maes = sorted(maes)
    for pct in [50, 75, 90, 95, 99]:
        idx = int(len(sorted_maes) * pct / 100)
        log.info(f"  MAE p{pct}: {sorted_maes[min(idx, len(sorted_maes)-1)]:.2f}t")

    # Capture ratio
    capture_ratios = [pnl / mfe if mfe > 0 else 0 for pnl, mfe in zip(pnls, mfes)]
    if capture_ratios:
        log.info(f"  Capture ratio: mean={statistics.mean(capture_ratios):.3f}, "
                 f"median={statistics.median(capture_ratios):.3f}")


def main():
    log.info("MAE/MFE Data Collection + Stop Loss Sweep")
    log.info(f"Binary: {BINARY}")
    log.info(f"Output: {OUT_DIR}")

    pairs = find_date_pairs()
    for sname, sp in pairs.items():
        log.info(f"  {sname}: {len(sp)} date pairs")

    # ========== PHASE 1: BASELINE (no SL, collect MAE/MFE) ==========
    baseline_jobs = []
    for strat in STRATEGIES:
        for date_str, pred_file, mbo_file in pairs[strat['name']]:
            out = str(OUT_DIR / f"baseline_{strat['name']}_{date_str}.json")
            baseline_jobs.append({
                'pred_file': pred_file,
                'mbo_file': mbo_file,
                'output_file': out,
                'threshold': strat['threshold'],
            })

    baseline_results = run_phase("PHASE 1: BASELINE (MAE/MFE collection)", baseline_jobs)

    # Analyze MAE/MFE per strategy
    for strat in STRATEGIES:
        strat_results = [r for r in baseline_results
                        if strat['vol'] in r.get('predictions_file', '')]
        analyze_mae_mfe(strat_results, strat['name'])

    # ========== PHASE 2: STOP LOSS SWEEP ==========
    sl_jobs = []
    for strat in STRATEGIES:
        for sl_ticks in SL_TICKS_LIST:
            for date_str, pred_file, mbo_file in pairs[strat['name']]:
                out = str(OUT_DIR / f"sl{sl_ticks}_{strat['name']}_{date_str}.json")
                sl_jobs.append({
                    'pred_file': pred_file,
                    'mbo_file': mbo_file,
                    'output_file': out,
                    'threshold': strat['threshold'],
                    'extra_args': ['--stop-loss-ticks', str(sl_ticks)],
                })

    sl_results = run_phase("PHASE 2: STOP LOSS SWEEP", sl_jobs)

    # Analyze SL results per strategy + SL level
    for strat in STRATEGIES:
        log.info(f"\n=== {strat['name']} SL Sweep Summary ===")
        for sl_ticks in SL_TICKS_LIST:
            filtered = []
            for r in sl_results:
                config = r.get('config', {})
                sl_val = config.get('stop_loss_ticks')
                pred = r.get('predictions_file', '')
                if sl_val == sl_ticks and strat['vol'] in pred:
                    filtered.append(r)

            if not filtered:
                continue
            total_pnl = sum(r.get('total_pnl_dollars', 0) for r in filtered)
            total_trades = sum(r.get('total_trades', 0) for r in filtered)
            avg_wr = sum(r.get('win_rate', 0) for r in filtered) / len(filtered) if filtered else 0
            log.info(f"  SL={sl_ticks}t: PnL=${total_pnl:.0f}, trades={total_trades}, "
                     f"avg_WR={avg_wr*100:.1f}%, days={len(filtered)}")

    # Save summary
    summary = {
        'timestamp': ts,
        'baseline_count': len(baseline_results),
        'sl_count': len(sl_results),
        'strategies': [s['name'] for s in STRATEGIES],
        'sl_ticks_tested': SL_TICKS_LIST,
    }
    with open(OUT_DIR / f'mae_mfe_summary_{ts}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nAll done! Results in {OUT_DIR}")
    log.info(f"Log: {log_file}")


if __name__ == '__main__':
    main()
