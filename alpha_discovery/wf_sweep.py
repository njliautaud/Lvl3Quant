#!/usr/bin/env python3
"""
wf_sweep.py -- Walk-Forward OOT Fill Sim Sweep
===============================================
Generates vol-gated per-day prediction NPZ files from walk-forward predictions
(oot_wf_predictions_incremental.npz), then runs comprehensive fill_sim_cli sweep.

Handles both:
  1. Static OOT predictions (static_oot) - already vol-gated in cnn_oot_sim_predictions/
  2. Walk-forward predictions (wf) - new per-date preds, need vol-gating

Sweep parameters:
  vol_gate:       [50, 60, 70, 80, 90]
  signal_threshold (conv): [1.5, 2.0, 2.5, 3.0]
  hold_ms:        [300000, 600000, 900000, 1200000, 1800000, 2700000, 3600000]
  chase_ticks:    [0, 1, 2, 3] (0=passive, >0=chase)
  chase_reprices: [0, 1, 3, 5]
  latency_ms:     [0, 5, 10, 25, 50, 100]

Total combos: 5 × 4 × 7 × 4 × 4 × 6 = 13,440 per date
WF dates: 20 -> 268,800 jobs
Runtime estimate: ~2-3 seconds per job on 12 workers -> ~12.5 hrs

Usage:
  python wf_sweep.py
  python wf_sweep.py --mode wf_only --workers 12
  python wf_sweep.py --mode static_only --workers 12
  python wf_sweep.py --mode both --workers 12
"""

import os
import sys
import json
import time
import bisect
import logging
import argparse
import subprocess
import threading
import queue
import gc
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
LVL3_ROOT   = Path('/home/jupiter/Lvl3Quant')
BINARY      = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR     = LVL3_ROOT / 'mbo_oot'
BOOK_DIR    = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
PRED_DIR    = LVL3_ROOT / 'data' / 'processed' / 'cnn_oot_sim_predictions'  # static
WF_NPZ      = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'
WF_PRED_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_sim_predictions'  # generated from WF
OUT_DIR     = LVL3_ROOT / 'alpha_discovery' / 'results' / 'wf_sweep'

TICK_VALUE = 12.50
BARS_PER_SEC = 10

# ── Sweep Grid ─────────────────────────────────────────────────────────────
VOL_GATES        = [50, 60, 70, 80, 90]
CONV_THRESHOLDS  = [1.5, 2.0, 2.5, 3.0]
HOLD_MS_LIST     = [300000, 600000, 900000, 1200000, 1800000, 2700000, 3600000]
CHASE_TICKS      = [0, 1, 2, 3]
CHASE_REPRICES   = [0, 1, 3, 5]
LATENCY_MS_LIST  = [0, 5, 10, 25, 50, 100]

# ── Logging ────────────────────────────────────────────────────────────────
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR.mkdir(parents=True, exist_ok=True)
log_file = OUT_DIR / f'wf_sweep_{ts}.log'

logging.basicConfig(
    format='%(asctime)s [wf_sweep] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(str(log_file), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('wf_sweep')


# ── Signal generation from raw predictions ─────────────────────────────────

def zscore_expanding(arr):
    result = np.full_like(arr, np.nan, dtype=np.float64)
    s, s2, n = 0.0, 0.0, 0
    for i in range(len(arr)):
        v = float(arr[i])
        if not (v == v):  # nan check
            continue
        s += v; s2 += v * v; n += 1
        if n >= 50:
            mean = s / n
            var = max(s2 / n - mean * mean, 0.0)
            result[i] = (v - mean) / max(var ** 0.5, 1e-8)
    return result


def compute_trailing_vol(mid, window=3000):
    mid = np.asarray(mid, dtype=np.float64)
    ret = np.zeros(len(mid))
    ret[10:] = (mid[10:] - mid[:-10]) / np.maximum(mid[:-10], 1e-8) * 10000
    vol = np.full(len(mid), np.nan)
    cs = np.cumsum(ret)
    cs2 = np.cumsum(ret ** 2)
    for i in range(window, len(mid)):
        s = cs[i] - cs[i - window]
        s2_ = cs2[i] - cs2[i - window]
        mean = s / window
        var = max(s2_ / window - mean * mean, 0.0)
        vol[i] = var ** 0.5
    return vol


def precompute_vol_percentiles(vol, percentiles=(50, 60, 70, 80, 90)):
    n = len(vol)
    result = {p: np.full(n, -np.inf) for p in percentiles}
    sorted_vals = []
    for i in range(n):
        if vol[i] == vol[i]:  # not nan
            bisect.insort(sorted_vals, vol[i])
        if len(sorted_vals) >= 100:
            for p in percentiles:
                idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
                result[p][i] = sorted_vals[idx]
    return result


def compute_morning_afternoon_mask(n_bars):
    seconds = np.arange(n_bars) / BARS_PER_SEC
    minutes = seconds / 60.0
    return (minutes < 120) | ((minutes >= 240) & (minutes < 330))


def generate_wf_sim_predictions(wf_npz_path, book_dir, out_dir, vol_gates=None):
    """Generate per-date vol-gated NPZ files from walk-forward predictions.
    Returns dict of (date, vol_gate) -> Path
    """
    if vol_gates is None:
        vol_gates = VOL_GATES

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f'Loading WF predictions from {wf_npz_path}...')
    wf_data = np.load(str(wf_npz_path), allow_pickle=True)
    pred_keys = [k for k in wf_data.keys() if k.endswith('_preds')]
    dates = sorted([k.replace('_preds', '') for k in pred_keys])
    log.info(f'  Found {len(dates)} WF dates: {dates[0]} to {dates[-1]}')

    saved = {}
    n_skipped = 0

    for date in dates:
        preds_key = f'{date}_preds'
        mid_key = f'{date}_mid'

        cp = wf_data[preds_key].astype(np.float64)
        n_bars = len(cp)

        # Get mid prices
        if mid_key in wf_data:
            mid = wf_data[mid_key].astype(np.float64)
        else:
            # Try book cache
            book_file = Path(book_dir) / f'{date}_book_tensors.npz'
            if book_file.exists():
                bd = np.load(str(book_file))
                mid = bd['mid_prices'].astype(np.float64)
            else:
                log.warning(f'  No mid for {date}, skipping')
                n_skipped += 1
                continue

        if len(mid) != n_bars:
            min_len = min(len(mid), n_bars)
            mid = mid[:min_len]
            cp = cp[:min_len]
            n_bars = min_len

        # Z-score
        zscored = zscore_expanding(cp)

        # Vol gate
        vol = compute_trailing_vol(mid)
        vol_pct = precompute_vol_percentiles(vol, percentiles=tuple(vol_gates))

        # Time mask
        morning_afternoon = compute_morning_afternoon_mask(n_bars)

        for vg in vol_gates:
            out_file = out_dir / f'{date}_vol{vg}_morning_afternoon.npz'
            if out_file.exists():
                saved[(date, vg)] = out_file
                continue

            vol_threshold = vol_pct[vg]
            # Gate: only trade when vol is high enough AND in morning/afternoon
            gated = zscored.copy()
            above_vol_gate = (vol >= vol_threshold) & ~np.isnan(vol) & ~np.isnan(zscored)
            mask = above_vol_gate & morning_afternoon
            gated[~mask] = 0.0
            gated[np.isnan(gated)] = 0.0

            np.savez_compressed(str(out_file), predictions=gated.astype(np.float32))
            saved[(date, vg)] = out_file

        log.info(f'  {date}: generated {len(vol_gates)} files')
        del cp, mid, zscored, vol, vol_pct, morning_afternoon
        gc.collect()

    log.info(f'Generated {len(saved)} WF prediction files ({n_skipped} skipped)')
    return saved


# ── Task generation ────────────────────────────────────────────────────────

def build_tasks(pred_files_by_key, mbo_dir, out_dir):
    """Build task list for sweep. pred_files_by_key: {(date, vol_gate): Path}"""
    # Index MBO files by date
    mbo_by_date = {}
    for ext in ['*.mbo.dbn.zst', '*.mbo.dbn']:
        for f in Path(mbo_dir).glob(ext):
            stem = f.name.split('.')[0]
            nodash = stem.split('-')[-1]
            if len(nodash) == 8:
                d = f'{nodash[:4]}-{nodash[4:6]}-{nodash[6:8]}'
                if d not in mbo_by_date:
                    mbo_by_date[d] = f

    tasks = []
    for (date, vg), pred_file in sorted(pred_files_by_key.items()):
        if date not in mbo_by_date:
            continue
        mbo_file = mbo_by_date[date]

        for conv, hold, ct, cr, lat in product(
            CONV_THRESHOLDS, HOLD_MS_LIST, CHASE_TICKS, CHASE_REPRICES, LATENCY_MS_LIST
        ):
            # Skip invalid chase combos: if chase_ticks=0, chase_reprices must be 0
            if ct == 0 and cr > 0:
                continue
            # If chase_reprices=0, chase_ticks can only be 0 (passive)
            if cr == 0 and ct > 0:
                continue

            hold_min = hold // 60000
            mode = 'passive' if ct == 0 else 'chase'
            label = f'wf_v{vg}_c{conv}_h{hold_min}m_{mode}_ct{ct}r{cr}_lat{lat}_{date}'
            out_file = Path(out_dir) / f'{label}.json'

            if out_file.exists():
                continue

            tasks.append({
                'label': label,
                'date': date,
                'vol_gate': vg,
                'conv': conv,
                'hold_ms': hold,
                'chase_ticks': ct,
                'chase_reprices': cr,
                'latency_ms': lat,
                'mbo_file': str(mbo_file),
                'pred_file': str(pred_file),
                'out_file': str(out_file),
            })

    return tasks


def run_task(task, binary):
    cmd = [
        str(binary),
        '--mbo-file', task['mbo_file'],
        '--predictions', task['pred_file'],
        '--output', task['out_file'],
        '--signal-threshold', str(task['conv']),
        '--hold-ms', str(task['hold_ms']),
        '--latency-ms', str(task['latency_ms']),
        '--quiet',
    ]
    if task['chase_ticks'] > 0:
        cmd += ['--chase-entry',
                '--chase-max-ticks', str(task['chase_ticks']),
                '--chase-max-reprices', str(task['chase_reprices'])]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and os.path.exists(task['out_file']):
            with open(task['out_file']) as f:
                return json.load(f)
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


# ── Aggregation ────────────────────────────────────────────────────────────

def aggregate_results(out_dir):
    """Aggregate all per-job JSON files into a summary by config."""
    configs = {}

    for f in Path(out_dir).glob('*.json'):
        if f.name.startswith('wf_sweep_summary'):
            continue
        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:
            continue

        # Parse label from filename
        name = f.stem
        # Strip trailing date portion _YYYY-MM-DD
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and len(parts[1]) == 10 and parts[1][4] == '-':
            config_key = parts[0]
        else:
            config_key = name

        if config_key not in configs:
            configs[config_key] = {
                'config_key': config_key,
                'days': 0,
                'total_pnl': 0.0,
                'total_trades': 0,
                'total_signals': 0,
                'total_filled': 0,
                'wins': 0,
                'daily_pnls': [],
                'sharpes': [],
                'fill_rates': [],
            }
        s = configs[config_key]
        s['days'] += 1
        pnl = data.get('total_pnl_dollars', 0) or 0
        trades = data.get('total_trades', 0) or 0
        signals = data.get('total_signals', 0) or 0
        filled = data.get('total_filled', 0) or 0
        win_rate = data.get('win_rate', 0) or 0
        fill_rate = data.get('fill_rate', 0) or 0
        sharpe = data.get('sharpe_per_trade', 0) or 0

        s['total_pnl'] += pnl
        s['total_trades'] += trades
        s['total_signals'] += signals
        s['total_filled'] += filled
        s['wins'] += int(trades * win_rate)
        s['daily_pnls'].append(pnl)
        s['fill_rates'].append(fill_rate)
        s['sharpes'].append(sharpe)

    # Compute portfolio-level Sharpe
    results = []
    for ck, s in configs.items():
        if s['days'] < 5:  # need at least 5 days
            continue
        daily = np.array(s['daily_pnls'])
        sharpe = (daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252)
        wr = s['wins'] / max(s['total_trades'], 1)
        fr = np.mean(s['fill_rates']) if s['fill_rates'] else 0
        results.append({
            'config': ck,
            'days': s['days'],
            'total_pnl': round(s['total_pnl'], 2),
            'trades': s['total_trades'],
            'win_rate': round(wr, 4),
            'fill_rate': round(fr, 4),
            'sharpe': round(float(sharpe), 3),
            'daily_pnl_mean': round(float(daily.mean()), 2),
            'daily_pnl_std': round(float(daily.std()), 2),
            'annualized_pnl': round(float(daily.mean() * 252), 0),
        })

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['wf_only', 'static_only', 'both'], default='both')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--gen_only', action='store_true', help='Only generate prediction files, no sweep')
    parser.add_argument('--static_vol_gates', nargs='+', type=int, default=[50, 60, 70, 80, 90],
                        help='Vol gates to use for static predictions (60/90 will be generated if missing)')
    args = parser.parse_args()

    log.info('=' * 80)
    log.info('WF Fill Sim Sweep')
    log.info('=' * 80)
    log.info(f'  Mode:         {args.mode}')
    log.info(f'  Workers:      {args.workers}')
    log.info(f'  Binary:       {BINARY}')
    log.info(f'  MBO dir:      {MBO_DIR}')
    log.info(f'  Output:       {OUT_DIR}')
    log.info('')

    all_pred_files = {}

    # ── Static predictions ─────────────────────────────────────────────────
    if args.mode in ('static_only', 'both'):
        log.info('Discovering static OOT prediction files...')
        static_files = {}
        for f in PRED_DIR.glob('*.npz'):
            parts = f.stem.split('_', 2)
            if len(parts) >= 2:
                date = parts[0]
                vol_str = parts[1]
                try:
                    vg = int(vol_str.replace('vol', ''))
                    if vg in args.static_vol_gates:
                        static_files[(date, vg)] = f
                except ValueError:
                    pass
        log.info(f'  Found {len(static_files)} static prediction files')
        all_pred_files.update(static_files)

        # Generate vol60/vol90 from static OOT predictions if missing
        missing_vols = [vg for vg in [60, 90] if vg in args.static_vol_gates
                        and not any(k[1] == vg for k in static_files)]
        if missing_vols:
            log.info(f'  Generating missing vol gates {missing_vols} for static predictions...')
            static_npz = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'oos_predictions_book_oot_20260311_092055.npz'
            if static_npz.exists():
                new_files = generate_wf_sim_predictions(
                    str(static_npz), str(BOOK_DIR), str(PRED_DIR), vol_gates=missing_vols
                )
                all_pred_files.update(new_files)
            else:
                log.warning(f'  Static OOT NPZ not found: {static_npz}')

    # ── WF predictions ─────────────────────────────────────────────────────
    if args.mode in ('wf_only', 'both'):
        log.info('Generating WF sim predictions...')
        if not WF_NPZ.exists():
            log.error(f'WF NPZ not found: {WF_NPZ}')
            sys.exit(1)
        wf_files = generate_wf_sim_predictions(str(WF_NPZ), str(BOOK_DIR), str(WF_PRED_DIR))
        all_pred_files.update({(f'wf_{d}', vg): p for (d, vg), p in wf_files.items()})
        # Also add with original date keys for MBO matching
        for (d, vg), p in wf_files.items():
            all_pred_files[(d, vg)] = p

        log.info(f'  WF prediction files: {len(wf_files)}')

    if args.gen_only:
        log.info('gen_only mode, exiting after prediction generation')
        return

    # ── Build tasks ────────────────────────────────────────────────────────
    log.info('Building task list...')
    tasks = build_tasks(all_pred_files, str(MBO_DIR), str(OUT_DIR))
    total = len(tasks)

    if total == 0:
        log.info('All tasks already complete!')
        return

    # Count total possible for reference
    n_dates = len(set(k[0] for k in all_pred_files))
    n_vgs = len(set(k[1] for k in all_pred_files))
    # Valid combos: passive (ct=0,cr=0) + chase (ct>0,cr>0)
    n_valid_chase = len([1 for ct, cr in product(CHASE_TICKS, CHASE_REPRICES) if ct > 0 and cr > 0])
    n_passive = 1
    n_combos = (n_passive + n_valid_chase) * len(CONV_THRESHOLDS) * len(HOLD_MS_LIST) * len(LATENCY_MS_LIST)
    n_total = n_combos * n_dates * n_vgs
    already_done = n_total - total

    log.info(f'  Total possible: {n_total:,} ({n_dates} dates × {n_vgs} vol gates × {n_combos} param combos)')
    log.info(f'  Already done:   {already_done:,}')
    log.info(f'  Remaining:      {total:,}')
    log.info(f'  Workers:        {args.workers}')
    if total > 0:
        secs_per_job = 2.5
        est_hours = total * secs_per_job / args.workers / 3600
        log.info(f'  Est. time:      {est_hours:.1f} hours')

    # ── Run sweep ──────────────────────────────────────────────────────────
    task_q = queue.Queue()
    for t in tasks:
        task_q.put(t)

    results_lock = threading.Lock()
    progress = {'done': 0, 'failed': 0, 'total': total}
    start_time = time.time()
    last_report = [time.time()]
    REPORT_INTERVAL = 300  # 5 min

    def worker():
        while True:
            try:
                t = task_q.get(timeout=3)
            except queue.Empty:
                break
            result = run_task(t, str(BINARY))
            with results_lock:
                if result is not None:
                    progress['done'] += 1
                else:
                    progress['failed'] += 1
                done = progress['done'] + progress['failed']
                now = time.time()
                if now - last_report[0] >= REPORT_INTERVAL:
                    elapsed = now - start_time
                    rate = done / elapsed * 60
                    remain = (total - done) / (done / elapsed) if done > 0 else 0
                    log.info(f'  Progress: {done:,}/{total:,} ({done/total:.1%}) | '
                             f'{rate:.0f}/min | ETA: {remain/3600:.1f}h')
                    last_report[0] = now

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    log.info(f'Sweep complete: {progress["done"]:,} done, {progress["failed"]:,} failed in {elapsed/3600:.1f}h')

    # ── Aggregate results ──────────────────────────────────────────────────
    log.info('Aggregating results...')
    summary = aggregate_results(str(OUT_DIR))

    summary_file = OUT_DIR / f'wf_sweep_summary_{ts}.json'
    with open(str(summary_file), 'w') as f:
        json.dump(summary[:200], f, indent=2)
    log.info(f'Summary saved: {summary_file}')

    # Print leaderboard
    log.info('\n' + '=' * 120)
    log.info('LEADERBOARD — Top 30 Configs by Sharpe')
    log.info('=' * 120)
    log.info(f'{"#":>3} {"Config":<70} {"P&L":>10} {"Sharpe":>7} {"Trades":>7} {"WR":>6} {"FillR":>6} {"Ann P&L":>10}')
    log.info('-' * 120)
    for i, r in enumerate(summary[:30]):
        log.info(f'{i+1:>3} {r["config"]:<70} ${r["total_pnl"]:>9,.0f} {r["sharpe"]:>7.2f} '
                 f'{r["trades"]:>7} {r["win_rate"]:>5.1%} {r["fill_rate"]:>5.1%} ${r["annualized_pnl"]:>9,.0f}')


if __name__ == '__main__':
    main()
