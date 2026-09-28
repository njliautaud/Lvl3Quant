#!/usr/bin/env python3
"""
Hold Time + Take-Profit + Conviction Mega Sweep
=================================================
Tests ALL hold times (5-30 min) with ALL take-profit levels and conviction filters.
Uses the existing WF prediction files from the exec sweep.

Usage:
    python alpha_discovery/deep_models/run_hold_time_sweep.py --workers 6
"""

import sys, json, time, logging, argparse, subprocess, glob
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path(__file__).resolve().parent.parent.parent
BINARY = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
MBO_DIR = LVL3_ROOT / 'data' / 'raw' / 'mbo'
PRED_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_exec_sweep_predictions'
SIM_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_hold_sweep_results'
RESULTS_DIR = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
SIM_OUT_DIR.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log = logging.getLogger('hold_sweep')
log.setLevel(logging.INFO)
_fh = logging.FileHandler(str(RESULTS_DIR / f'hold_sweep_{_ts}.log'), mode='w', encoding='utf-8')
_fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_fh)
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
log.addHandler(_ch)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── SWEEP GRID ──
# (conv_threshold, hold_ms, take_profit_ticks, trailing_ticks, chase, label)
CONFIGS = []

# Pure hold time sweep (no stops, no TP)
for conv in [2.0, 2.5, 3.0]:
    for hold_min in [5, 10, 15, 20, 25, 30]:
        hold_ms = hold_min * 60 * 1000
        # Passive
        CONFIGS.append((conv, hold_ms, None, None, False, f'hold{hold_min}m_conv{int(conv*10)}_passive'))
        # Chase 1t/3r
        CONFIGS.append((conv, hold_ms, None, None, True, f'hold{hold_min}m_conv{int(conv*10)}_chase'))

# Hold + take-profit combos (best hold times with TP)
for conv in [2.0, 2.5]:
    for hold_min in [10, 15, 20, 30]:
        hold_ms = hold_min * 60 * 1000
        for tp in [5, 8, 10, 15, 20]:
            CONFIGS.append((conv, hold_ms, tp, None, True,
                           f'hold{hold_min}m_tp{tp}_conv{int(conv*10)}_chase'))

# Hold + trailing stop combos
for conv in [2.0, 2.5]:
    for hold_min in [15, 20, 30]:
        hold_ms = hold_min * 60 * 1000
        for trail in [15, 20, 25]:
            CONFIGS.append((conv, hold_ms, None, trail, True,
                           f'hold{hold_min}m_trail{trail}_conv{int(conv*10)}_chase'))

# Hold + TP + trailing combo
for conv in [2.5]:
    for hold_min in [15, 20, 30]:
        hold_ms = hold_min * 60 * 1000
        for tp, trail in [(10, 20), (15, 25), (8, 15)]:
            CONFIGS.append((conv, hold_ms, tp, trail, True,
                           f'hold{hold_min}m_tp{tp}_trail{trail}_conv{int(conv*10)}_chase'))

VOL_GATES = [50, 70, 80]

log.info(f"Total configs: {len(CONFIGS)}")


def run_sim(mbo_file, pred_file, output_file, config):
    conv, hold_ms, tp, trail, chase, label = config
    cmd = [str(BINARY), '--mbo-file', str(mbo_file), '--predictions', str(pred_file),
           '--output', str(output_file), '--hold-ms', str(hold_ms),
           '--signal-threshold', str(conv), '--latency-ms', '0', '--quiet']
    if chase:
        cmd += ['--chase-entry', '--chase-max-ticks', '1', '--chase-max-reprices', '3']
    if tp is not None:
        cmd += ['--take-profit-ticks', str(tp)]
    if trail is not None:
        cmd += ['--trailing-ticks', str(trail)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and Path(output_file).exists():
            with open(output_file) as f:
                return json.load(f)
    except Exception as e:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()

    # Find all prediction files
    pred_files = {}
    for f in glob.glob(str(PRED_DIR / '*.npz')):
        base = Path(f).stem  # e.g. 2025-12-01_vol50_morning_afternoon
        parts = base.split('_')
        date = parts[0]
        vg = int(parts[1].replace('vol', ''))
        pred_files[(date, vg)] = f

    log.info(f"Prediction files: {len(pred_files)}")
    log.info(f"Configs: {len(CONFIGS)}")

    # Build job list
    jobs = []
    for (date, vg), pred_file in pred_files.items():
        date_compact = date.replace('-', '')
        mbo_candidates = list(MBO_DIR.glob(f'*{date_compact}*.dbn.zst'))
        if not mbo_candidates:
            continue
        mbo_file = mbo_candidates[0]

        for config in CONFIGS:
            label = config[-1]
            out_file = SIM_OUT_DIR / f'vol{vg}_{label}_{date}.json'
            if out_file.exists():
                continue  # Skip already completed
            jobs.append((str(mbo_file), pred_file, str(out_file), config, date, vg))

    log.info(f"Jobs to run: {len(jobs)} (skipped {len(pred_files) * len(CONFIGS) - len(jobs)} existing)")

    # Run
    completed = 0
    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for mbo, pred, out, cfg, date, vg in jobs:
            f = executor.submit(run_sim, mbo, pred, out, cfg)
            futures[f] = (cfg[-1], date, vg)

        for future in as_completed(futures):
            label, date, vg = futures[future]
            completed += 1
            res = future.result()
            if res:
                results.append({'config': label, 'date': date, 'vol_gate': vg,
                               'pnl': res.get('total_pnl_dollars', 0),
                               'trades': res.get('total_trades', 0),
                               'signals': res.get('total_signals', 0),
                               'filled': res.get('total_filled', 0),
                               'win_rate': res.get('win_rate', 0)})
            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - completed) / rate / 60 if rate > 0 else 0
                log.info(f"  Progress: {completed}/{len(jobs)} ({rate:.1f}/s, ETA {eta:.1f}min)")

    log.info(f"Completed {completed} jobs in {time.time()-t0:.0f}s")

    # Also load any pre-existing results
    for f in glob.glob(str(SIM_OUT_DIR / '*.json')):
        try:
            with open(f) as fh:
                data = json.load(fh)
            base = Path(f).stem
            parts = base.split('_')
            date = parts[-1] if len(parts[-1]) == 10 else None
            if not date: continue
            vg_str = parts[0]
            label = '_'.join(parts[1:-1])
            results.append({'config': label, 'date': date, 'vol_gate': vg_str,
                           'pnl': data.get('total_pnl_dollars', 0),
                           'trades': data.get('total_trades', 0),
                           'signals': data.get('total_signals', 0),
                           'filled': data.get('total_filled', 0),
                           'win_rate': data.get('win_rate', 0)})
        except:
            continue

    # Aggregate
    from collections import defaultdict
    config_data = defaultdict(list)
    for r in results:
        key = (r['config'], r['vol_gate'])
        config_data[key].append(r)

    summaries = []
    for (label, vg), days in config_data.items():
        total_pnl = sum(d['pnl'] for d in days)
        total_trades = sum(d['trades'] for d in days)
        n_days = len(days)
        daily_pnls = [d['pnl'] for d in days]
        avg = np.mean(daily_pnls)
        std = np.std(daily_pnls) if len(daily_pnls) > 1 else 1
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0
        wr = sum(d['win_rate'] * d['trades'] for d in days) / max(total_trades, 1)
        fill = sum(d['filled'] for d in days) / max(sum(d['signals'] for d in days), 1)
        summaries.append({
            'label': label, 'vg': vg, 'pnl': round(total_pnl, 2),
            'trades': total_trades, 'sharpe': round(sharpe, 3),
            'wr': round(wr, 4), 'fill': round(fill, 4),
            'annual': round(avg * 252, 0), 'n_days': n_days,
        })

    summaries.sort(key=lambda x: x['sharpe'], reverse=True)

    log.info(f"\n{'='*90}")
    log.info("HOLD TIME SWEEP RESULTS — Ranked by Sharpe")
    log.info(f"{'='*90}")
    for i, s in enumerate(summaries[:50]):
        log.info(f"#{i+1:>3} {s['vg']:>5} {s['label']:<50} Sharpe {s['sharpe']:>7.2f} "
                f"P&L ${s['pnl']:>10,.2f} {s['trades']:>5}t WR {s['wr']*100:>5.1f}% "
                f"Fill {s['fill']*100:>5.1f}% Annual ${s['annual']:>9,.0f}")

    out_path = RESULTS_DIR / f'hold_sweep_results_{_ts}.json'
    with open(out_path, 'w') as f:
        json.dump({'timestamp': _ts, 'n_configs': len(CONFIGS), 'summaries': summaries}, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
