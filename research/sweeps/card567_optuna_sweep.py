#!/usr/bin/env python3
"""
card567_optuna_sweep.py -- Optuna-style grid sweep for Cards 5-7.

Signal types:
  Card 5: raw_rawExit with ethr0.5 vol0
  Card 6: raw_rawExit with vol70
  Card 7: smooth_smoothExit

The conviction and vol_gate parameters are BAKED into prediction filenames.
We sweep runtime parameters (SL, TP, signal_threshold, hold_ms, flip_exit)
across all matching prediction files.

Split for deployment:
  Jupiter (14 workers): Card 7 (smooth_smoothExit)
  Saturn  (40 workers): Cards 5+6 (raw_rawExit)

Optimized grid (~500 combos per pred file) for ~24h completion.

Usage:
  python3 card567_optuna_sweep.py --workers 14 --card-filter card7
  python3 card567_optuna_sweep.py --workers 40 --card-filter card56
  python3 card567_optuna_sweep.py --dry-run
  python3 card567_optuna_sweep.py --aggregate-only
"""
import os
import sys
import json
import time
import math
import logging
import subprocess
import threading
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ── Auto-detect paths ────────────────────────────────────────────────────────
HOME = Path.home()
for candidate in [HOME / 'Lvl3Quant', Path('/home/jupiter/Lvl3Quant'), Path('/home/saturn/Lvl3Quant')]:
    if candidate.exists():
        LVL3_ROOT = candidate
        break
else:
    LVL3_ROOT = HOME / 'Lvl3Quant'

BINARY    = LVL3_ROOT / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO_DIR   = LVL3_ROOT / 'data' / 'raw' / 'mbo'
PRED_DIR  = LVL3_ROOT / 'data' / 'processed' / 'cnn_wf_stacked_predictions'
OUT_DIR   = LVL3_ROOT / 'data' / 'processed' / 'card567_optuna_sweep'

# ── Parameter Grid ───────────────────────────────────────────────────────────
# Runtime parameters -- optimized grid for tractability
# SL: key levels covering none, tight, medium, wide
SL_TICKS        = [None, 8, 10, 12, 15, 18, 20, 25, 30]        # 9 values
# TP: key levels covering none, tight, medium, wide
TP_TICKS        = [None, 10, 15, 20, 25, 30, 40, 50]            # 8 values
# Signal threshold: from very permissive to very strict
SIGNAL_THRS     = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]              # 6 values
# Hold time: 30min, 1hr, 2hr
HOLD_MS_OPTIONS = [1800000, 3600000, 7200000]                    # 3 values
# Signal flip exit: test both
FLIP_EXIT       = [True, False]                                   # 2 values

# Total combos: 9 * 8 * 6 * 3 * 2 = 2,592 per pred file

# Chase entry — proven best
CHASE_MAX_TICKS    = 1
CHASE_MAX_REPRICES = 3
LATENCY_MS         = 0

# OOT date range
OOT_START = '2025-12-01'
OOT_END   = '2026-03-08'

MAX_WORKERS = 40

# ── Logging ──────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = OUT_DIR / f'card567_sweep_{ts}.log'

logging.basicConfig(
    format='%(asctime)s [c567] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(str(log_file), mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('c567')


def find_mbo_files():
    """Build index: date_str -> Path for MBO files in OOT range."""
    mbo_index = {}
    for f in sorted(MBO_DIR.glob('glbx-mdp3-*.mbo.dbn.zst')):
        parts = f.name.split('-')
        if len(parts) >= 3:
            raw_date = parts[2].split('.')[0]
            if len(raw_date) == 8 and raw_date.isdigit():
                date_str = f'{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}'
                if OOT_START <= date_str <= OOT_END:
                    mbo_index[date_str] = f
    return mbo_index


def find_predictions():
    """Build index of prediction files grouped by card type."""
    cards = {'card5': {}, 'card6': {}, 'card7': {}}
    if not PRED_DIR.exists():
        log.warning(f'Prediction dir not found: {PRED_DIR}')
        return cards

    for f in sorted(PRED_DIR.glob('*.npz')):
        stem = f.stem
        if len(stem) < 12:
            continue
        date_str = stem[:10]
        if date_str < OOT_START or date_str > OOT_END:
            continue
        pattern = stem[11:]

        if 'smooth_smoothExit' in pattern:
            # Card 7: all smooth_smoothExit variants (conv1.5, conv2.0)
            # Skip conv2.5 — too aggressive, rarely productive
            if 'conv2.5' not in pattern:
                cards['card7'][(date_str, pattern)] = f
        elif 'raw_rawExit' in pattern:
            # Card 5: ethr0.5 AND vol0
            if 'ethr0.5' in pattern and pattern.endswith('vol0'):
                cards['card5'][(date_str, pattern)] = f
            # Card 6: vol70 (any ethr)
            elif 'vol70' in pattern:
                cards['card6'][(date_str, pattern)] = f

    return cards


def build_tasks(mbo_index, card_preds, card_name):
    """Build task list for a single card type."""
    tasks = []
    skipped = 0

    patterns = sorted(set(pat for (_, pat) in card_preds.keys()))
    log.info(f'  {card_name}: {len(patterns)} unique pred patterns, {len(card_preds)} total files')
    for p in patterns:
        dates_count = sum(1 for (d, pat) in card_preds if pat == p)
        log.info(f'    {p}: {dates_count} dates')

    for (date_str, pred_pattern), pred_file in sorted(card_preds.items()):
        if date_str not in mbo_index:
            continue
        mbo_file = mbo_index[date_str]

        for sig_thr in SIGNAL_THRS:
            for tp in TP_TICKS:
                for sl in SL_TICKS:
                    for hold_ms in HOLD_MS_OPTIONS:
                        for flip in FLIP_EXIT:
                            tp_l = f'tp{tp}' if tp is not None else 'tpN'
                            sl_l = f'sl{sl}' if sl is not None else 'slN'
                            sig_l = f's{sig_thr}'.replace('.', 'p')
                            hold_l = f'h{hold_ms // 60000}m'
                            flip_l = 'fY' if flip else 'fN'

                            label = f'{card_name}_{pred_pattern}_{sig_l}_{tp_l}_{sl_l}_{hold_l}_{flip_l}_{date_str}'
                            out_file = OUT_DIR / f'{label}.json'

                            if out_file.exists():
                                skipped += 1
                                continue

                            tasks.append({
                                'label': label,
                                'card': card_name,
                                'strategy': pred_pattern,
                                'mbo_file': str(mbo_file),
                                'pred_file': str(pred_file),
                                'out_file': str(out_file),
                                'signal_threshold': sig_thr,
                                'hold_ms': hold_ms,
                                'tp_ticks': tp,
                                'sl_ticks': sl,
                                'flip_exit': flip,
                            })

    log.info(f'  {card_name}: {len(tasks):,} tasks to run, {skipped:,} skipped')
    return tasks


def run_task(task):
    """Execute one fill_sim_cli job."""
    cmd = [
        str(BINARY),
        '--mbo-file',           task['mbo_file'],
        '--predictions',        task['pred_file'],
        '--output',             task['out_file'],
        '--signal-threshold',   str(task['signal_threshold']),
        '--hold-ms',            str(task['hold_ms']),
        '--latency-ms',         str(LATENCY_MS),
        '--chase-entry',
        '--chase-max-ticks',    str(CHASE_MAX_TICKS),
        '--chase-max-reprices', str(CHASE_MAX_REPRICES),
        '--max-wait-bars',      '50',
        '--quiet',
    ]
    if task['tp_ticks'] is not None:
        cmd += ['--take-profit-ticks', str(task['tp_ticks'])]
    if task['sl_ticks'] is not None:
        cmd += ['--stop-loss-ticks', str(task['sl_ticks'])]
    if task['flip_exit']:
        cmd += ['--signal-flip-exit']

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.returncode == 0 and os.path.exists(task['out_file'])
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def aggregate_results():
    """Aggregate all output JSON files and rank by Sharpe ratio."""
    log.info('')
    log.info('=' * 80)
    log.info('AGGREGATING RESULTS')
    log.info('=' * 80)

    config_agg = defaultdict(lambda: {
        'daily_pnls': [],
        'total_trades': 0,
        'total_wins': 0,
        'total_pnl_dollars': 0.0,
        'trade_pnls': [],
        'days': 0,
        'card': '',
    })

    file_count = 0
    parse_errors = 0

    for f in sorted(OUT_DIR.glob('card*.json')):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            parse_errors += 1
            continue

        file_count += 1
        stem = f.stem

        # Strip date suffix (_YYYY-MM-DD) to get config key
        if len(stem) > 11 and stem[-4:].isdigit() and stem[-7] == '-' and stem[-10] == '-':
            config_key = stem[:-11]
        else:
            config_key = stem

        card = stem.split('_')[0]
        agg = config_agg[config_key]
        agg['card'] = card

        day_pnl = d.get('total_pnl_dollars', d.get('total_pnl_ticks', 0))
        agg['daily_pnls'].append(day_pnl)
        agg['total_pnl_dollars'] += day_pnl
        agg['total_trades'] += d.get('total_trades', d.get('num_trades', 0))
        agg['days'] += 1

        trades = d.get('trades', [])
        for t in trades:
            pnl = t.get('pnl_dollars', t.get('pnl_ticks', 0))
            agg['trade_pnls'].append(pnl)
            if pnl > 0:
                agg['total_wins'] += 1

    log.info(f'Parsed {file_count:,} output files ({parse_errors} errors)')

    results = []
    for config_key, agg in config_agg.items():
        daily_pnls = agg['daily_pnls']
        n_days = len(daily_pnls)
        if n_days < 2:
            continue

        total_pnl = sum(daily_pnls)
        avg_daily = total_pnl / n_days
        std_daily = math.sqrt(sum((p - avg_daily) ** 2 for p in daily_pnls) / (n_days - 1)) if n_days > 1 else 1e-9
        daily_sharpe = (avg_daily / std_daily * math.sqrt(252)) if std_daily > 0 else 0

        total_trades = agg['total_trades']
        total_wins = agg['total_wins']
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(p for p in agg['trade_pnls'] if p > 0)
        gross_loss = abs(sum(p for p in agg['trade_pnls'] if p < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        wins_list = [p for p in agg['trade_pnls'] if p > 0]
        losses_list = [p for p in agg['trade_pnls'] if p < 0]
        avg_win = sum(wins_list) / len(wins_list) if wins_list else 0
        avg_loss = sum(losses_list) / len(losses_list) if losses_list else 0

        pos_days = sum(1 for p in daily_pnls if p > 0)
        neg_days = sum(1 for p in daily_pnls if p < 0)

        results.append({
            'config': config_key,
            'card': agg['card'],
            'daily_sharpe': round(daily_sharpe, 3),
            'total_pnl': round(total_pnl, 2),
            'avg_daily_pnl': round(avg_daily, 2),
            'total_trades': total_trades,
            'win_rate': round(win_rate, 1),
            'profit_factor': round(min(profit_factor, 999.0), 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'days': n_days,
            'pos_days': pos_days,
            'neg_days': neg_days,
        })

    results.sort(key=lambda x: x['daily_sharpe'], reverse=True)

    summary_file = OUT_DIR / 'card567_sweep_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f'Saved summary: {summary_file} ({len(results):,} configs)')

    # Print top 20 overall
    log.info('')
    log.info('TOP 20 CONFIGS BY DAILY SHARPE (all cards):')
    log.info(f'{"#":<4} {"Card":<6} {"Sharpe":>8} {"TotPnL":>10} {"AvgDay":>9} {"Trades":>7} '
             f'{"WR%":>6} {"PF":>6} {"AvgW":>8} {"AvgL":>8} {"Days":>5} {"W/L":>5}  Config')
    log.info('-' * 150)
    for i, r in enumerate(results[:20], 1):
        log.info(
            f'{i:<4} {r["card"]:<6} {r["daily_sharpe"]:>8.3f} {r["total_pnl"]:>10.2f} '
            f'{r["avg_daily_pnl"]:>9.2f} {r["total_trades"]:>7} {r["win_rate"]:>6.1f} '
            f'{r["profit_factor"]:>6.2f} {r["avg_win"]:>8.2f} {r["avg_loss"]:>8.2f} '
            f'{r["days"]:>5} {r["pos_days"]}/{r["neg_days"]:>2}  {r["config"]}'
        )

    # Top 10 per card
    for card in ['card5', 'card6', 'card7']:
        card_results = [r for r in results if r['card'] == card]
        if not card_results:
            continue
        log.info('')
        log.info(f'=== TOP 10: {card.upper()} ===')
        for i, r in enumerate(card_results[:10], 1):
            log.info(
                f'  {i:>2}. Sharpe={r["daily_sharpe"]:>7.3f}  PnL=${r["total_pnl"]:>9.2f}  '
                f'Avg/day=${r["avg_daily_pnl"]:>8.2f}  Trades={r["total_trades"]:>5}  '
                f'WR={r["win_rate"]:>5.1f}%  PF={r["profit_factor"]:>5.2f}  '
                f'Days={r["days"]}({r["pos_days"]}+/{r["neg_days"]}-)  | {r["config"]}'
            )

    return results


def main():
    parser = argparse.ArgumentParser(description='Card 5-7 Optuna-style parameter sweep')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Parallel workers (default {MAX_WORKERS})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Count tasks only, do not run')
    parser.add_argument('--card-filter', type=str, default=None,
                        choices=['card5', 'card6', 'card7', 'card56', 'card567'],
                        help='Run only specific cards (default: all)')
    parser.add_argument('--aggregate-only', action='store_true',
                        help='Only aggregate existing results, no simulation')
    args = parser.parse_args()

    if args.aggregate_only:
        aggregate_results()
        return

    log.info('=' * 80)
    log.info('CARD 5-7 OPTUNA-STYLE PARAMETER SWEEP')
    log.info('=' * 80)
    log.info(f'  Binary:      {BINARY}')
    log.info(f'  MBO dir:     {MBO_DIR}')
    log.info(f'  Pred dir:    {PRED_DIR}')
    log.info(f'  Output:      {OUT_DIR}')
    log.info(f'  Workers:     {args.workers}')
    log.info(f'  Card filter: {args.card_filter or "ALL"}')
    log.info(f'  OOT range:   {OOT_START} to {OOT_END}')
    log.info('')
    log.info('Runtime parameter grid:')
    log.info(f'  SL ticks:      {SL_TICKS}  ({len(SL_TICKS)} values)')
    log.info(f'  TP ticks:      {TP_TICKS}  ({len(TP_TICKS)} values)')
    log.info(f'  Signal thr:    {SIGNAL_THRS}  ({len(SIGNAL_THRS)} values)')
    log.info(f'  Hold time:     {[h//60000 for h in HOLD_MS_OPTIONS]}min  ({len(HOLD_MS_OPTIONS)} values)')
    log.info(f'  Flip exit:     {FLIP_EXIT}  ({len(FLIP_EXIT)} values)')
    combos = len(SL_TICKS) * len(TP_TICKS) * len(SIGNAL_THRS) * len(HOLD_MS_OPTIONS) * len(FLIP_EXIT)
    log.info(f'  Combos/file:   {combos:,}')
    log.info('')

    # Discover files
    log.info('Discovering data files...')
    mbo_index = find_mbo_files()
    card_preds = find_predictions()

    log.info(f'  MBO files (OOT):   {len(mbo_index)} dates')
    if mbo_index:
        log.info(f'    Range: {min(mbo_index)} to {max(mbo_index)}')
    for card_name in ['card5', 'card6', 'card7']:
        n_files = len(card_preds[card_name])
        n_pats = len(set(pat for (_, pat) in card_preds[card_name].keys()))
        log.info(f'  {card_name} preds:   {n_files} files ({n_pats} patterns)')
    log.info('')

    card_map = {
        'card5': ['card5'],
        'card6': ['card6'],
        'card7': ['card7'],
        'card56': ['card5', 'card6'],
        'card567': ['card5', 'card6', 'card7'],
        None: ['card5', 'card6', 'card7'],
    }
    active_cards = card_map.get(args.card_filter, ['card5', 'card6', 'card7'])

    all_tasks = []
    for card_name in active_cards:
        tasks = build_tasks(mbo_index, card_preds[card_name], card_name)
        all_tasks.extend(tasks)

    total = len(all_tasks)
    if total == 0:
        log.info('No tasks to run! All complete or no matching prediction files.')
        aggregate_results()
        return

    # Estimate based on measured rates: ~14s/task on Jupiter, ~37s on Saturn
    # Use conservative 20s average
    est_sec_per_task = 20.0
    est_hours = total * est_sec_per_task / args.workers / 3600
    log.info(f'Total tasks: {total:,}')
    log.info(f'Est. time:   {est_hours:.1f} hours at {args.workers} workers (~{est_sec_per_task:.0f}s/task)')

    for card in active_cards:
        cnt = sum(1 for t in all_tasks if t['card'] == card)
        card_hours = cnt * est_sec_per_task / args.workers / 3600
        log.info(f'  {card}: {cnt:,} tasks (~{card_hours:.1f}h)')

    if args.dry_run:
        log.info('')
        log.info('DRY RUN -- not executing.')
        return

    # ── Execute sweep ────────────────────────────────────────────────────────
    done = 0
    failed = 0
    lock = threading.Lock()
    start = time.time()
    last_rpt = [time.time()]
    REPORT_INTERVAL = 300

    checkpoint_file = OUT_DIR / 'sweep_progress.json'

    def run_and_track(task):
        nonlocal done, failed
        ok = run_task(task)
        with lock:
            if ok:
                done += 1
            else:
                failed += 1
            total_done = done + failed
            now = time.time()
            if now - last_rpt[0] >= REPORT_INTERVAL:
                elapsed = now - start
                rate = total_done / elapsed * 60 if elapsed > 0 else 0
                remain = (total - total_done) / (total_done / elapsed) if total_done > 0 else 0
                pct = total_done / total * 100
                log.info(
                    f'Progress: {total_done:,}/{total:,} ({pct:.1f}%) '
                    f'| {rate:.0f}/min | OK={done:,} FAIL={failed:,} '
                    f'| ETA: {remain/3600:.1f}h'
                )
                try:
                    with open(checkpoint_file, 'w') as cp:
                        json.dump({
                            'total': total,
                            'done': done,
                            'failed': failed,
                            'elapsed_sec': round(elapsed, 1),
                            'rate_per_min': round(rate, 1),
                            'eta_hours': round(remain / 3600, 2),
                            'pct': round(pct, 1),
                            'timestamp': datetime.now().isoformat(),
                        }, cp)
                except Exception:
                    pass
                last_rpt[0] = now
        return ok

    log.info('')
    log.info(f'Starting sweep with {args.workers} workers at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}...')

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_and_track, t): t for t in all_tasks}
        for f in as_completed(futures):
            pass

    elapsed = time.time() - start

    log.info('')
    log.info('=' * 80)
    log.info('SWEEP COMPLETE')
    log.info('=' * 80)
    log.info(f'  Done:     {done:,}')
    log.info(f'  Failed:   {failed:,}')
    log.info(f'  Time:     {elapsed/3600:.2f}h ({elapsed:.0f}s)')
    if elapsed > 0:
        log.info(f'  Rate:     {(done + failed) / elapsed * 60:.0f}/min')

    try:
        with open(checkpoint_file, 'w') as cp:
            json.dump({
                'status': 'COMPLETE',
                'total': total,
                'done': done,
                'failed': failed,
                'elapsed_hours': round(elapsed / 3600, 2),
                'timestamp': datetime.now().isoformat(),
            }, cp)
    except Exception:
        pass

    aggregate_results()


if __name__ == '__main__':
    main()
