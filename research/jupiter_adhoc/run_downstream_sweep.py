#!/usr/bin/env python3
"""
run_downstream_sweep.py -- Full downstream test: queue filter × SL × TP × latency.

Tests all exit conditions GIVEN the best queue filter (max_wait_bars).
This answers: "once we filter to fast fills only, do SL/TP/trailing work?"

Run: python3 run_downstream_sweep.py --workers 40 --best-wait-bars 30
"""
import os, sys, json, time, logging, subprocess, threading
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3 = Path('/home/jupiter/Lvl3Quant')
BIN = LVL3 / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli'
MBO = LVL3 / 'data' / 'raw' / 'mbo'
PRED = LVL3 / 'data' / 'processed' / 'cnn_wf_stacked_predictions'
OUT = LVL3 / 'data' / 'processed' / 'downstream_sweep_results'

STRATEGIES = ['book_predstdExit_conv1.5_vol50', 'book_predstdExit_conv2.5_vol70']

# Downstream exit parameters to test WITH the queue filter
TP_VALUES = [None, 5, 10, 15, 20, 30]
SL_VALUES = [None, 10, 15, 20, 25, 30, 40, 50]
HOLD_MS_VALUES = [1800000, 3600000]  # 30m, 60m timeout
LATENCY_MS = [0, 50]

OUT.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    format='%(asctime)s [downstream] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S', level=logging.INFO,
    handlers=[logging.FileHandler(str(OUT / f'downstream_{ts}.log'), mode='w'),
              logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger('downstream')

def find_files():
    mbo = {}
    for f in sorted(MBO.glob('glbx-mdp3-*.mbo.dbn.zst')):
        p = f.name.split('-')
        if len(p) >= 3:
            rd = p[2].split('.')[0]
            if len(rd) == 8 and rd.isdigit():
                mbo[f'{rd[:4]}-{rd[4:6]}-{rd[6:8]}'] = f
    pred = {}
    for f in sorted(PRED.glob('*.npz')):
        s = f.stem
        if len(s) >= 12: pred[(s[:10], s[11:])] = f
    return mbo, pred

def build_tasks(wait_bars):
    mbo, pred = find_files()
    tasks = []
    skipped = 0
    for strat in STRATEGIES:
        dates = [(d, pred[(d,strat)]) for d in mbo if (d,strat) in pred]
        log.info(f'  {strat}: {len(dates)} dates')
        for d, pf in sorted(dates):
            for tp in TP_VALUES:
                for sl in SL_VALUES:
                    for hold in HOLD_MS_VALUES:
                        for lat in LATENCY_MS:
                            tp_l = f'_tp{tp}' if tp else '_tpN'
                            sl_l = f'_sl{sl}' if sl else '_slN'
                            h_l = f'_h{hold//60000}m'
                            label = f'ds_{strat}_wb{wait_bars}{tp_l}{sl_l}{h_l}_lat{lat}_{d}'
                            out_f = OUT / f'{label}.json'
                            if out_f.exists():
                                skipped += 1
                                continue
                            tasks.append({
                                'label': label, 'strat': strat, 'date': d,
                                'mbo': str(mbo[d]), 'pred': str(pf), 'out': str(out_f),
                                'wait_bars': wait_bars, 'tp': tp, 'sl': sl,
                                'hold': hold, 'lat': lat
                            })
    log.info(f'Skip: {skipped} | Tasks: {len(tasks):,}')
    return tasks

def run_task(t):
    cmd = [str(BIN), '--mbo-file', t['mbo'], '--predictions', t['pred'],
           '--output', t['out'], '--signal-threshold', '0.1',
           '--hold-ms', str(t['hold']), '--max-wait-bars', str(t['wait_bars']),
           '--latency-ms', str(t['lat']),
           '--chase-entry', '--chase-max-ticks', '1', '--chase-max-reprices', '3', '--quiet']
    if t['tp']: cmd += ['--take-profit-ticks', str(t['tp'])]
    if t['sl']: cmd += ['--trailing-ticks', str(t['sl'])]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except: return False

def aggregate():
    configs = {}
    for f in OUT.glob('ds_*.json'):
        try:
            d = json.load(open(f))
            cfg = f.stem[:-11]
            if cfg not in configs:
                configs[cfg] = {'pnls':[], 'trades':[], 'wrs':[], 'fills':[], 'signals':[]}
            configs[cfg]['pnls'].append(d.get('total_pnl_dollars', 0))
            configs[cfg]['trades'].append(d.get('total_trades', 0))
            configs[cfg]['fills'].append(d.get('total_filled', 0))
            configs[cfg]['signals'].append(d.get('total_signals', 0))
            if d.get('total_trades', 0) > 0:
                configs[cfg]['wrs'].append(d.get('win_rate', 0))
        except: pass

    ranked = []
    for cfg, data in configs.items():
        n = len(data['pnls'])
        pnl = sum(data['pnls'])
        trades = sum(data['trades'])
        signals = sum(data['signals'])
        daily = np.array(data['pnls'])
        sharpe = float(np.mean(daily) / np.std(daily, ddof=1) * np.sqrt(252)) if len(daily) > 1 and np.std(daily) > 0 else 0
        cum = np.cumsum(daily)
        peak = np.maximum.accumulate(cum)
        mdd = float(np.max(peak - cum)) if len(cum) > 0 else 0
        avg_wr = float(np.mean(data['wrs'])) if data['wrs'] else 0
        win_days = sum(1 for p in data['pnls'] if p > 0)
        lose_days = sum(1 for p in data['pnls'] if p < 0)
        # Median daily PnL
        med_daily = float(np.median(daily))
        fill_rate = sum(data['fills']) / sum(data['signals']) * 100 if sum(data['signals']) > 0 else 0
        ranked.append({
            'cfg': cfg, 'sharpe': round(sharpe,2), 'pnl': round(pnl,0),
            'mdd': round(mdd,0), 'trades': trades, 'days': n,
            'avg_d': round(pnl/n,0) if n else 0, 'med_d': round(med_daily,0),
            'avg_wr': round(avg_wr*100,1), 'win_d': win_days, 'lose_d': lose_days,
            'pdd': round(pnl/mdd,1) if mdd > 0 else 0, 'fill_pct': round(fill_rate,1)
        })

    ranked.sort(key=lambda x: x['sharpe'], reverse=True)
    log.info('\n' + '=' * 120)
    log.info('TOP 25 CONFIGS — queue filtered + downstream exits:')
    log.info(f'{"#":>3} {"SHRP":>6} {"PnL":>9} {"MDD":>8} {"P/DD":>5} {"TR":>5} {"WR%":>5} {"W/L":>5} {"$/d":>7} {"med":>6} {"fill%":>5}  CONFIG')
    log.info('-' * 120)
    for i, r in enumerate(ranked[:25], 1):
        wl = f"{r['win_d']}/{r['lose_d']}"
        log.info(f"{i:3d} {r['sharpe']:6.2f} ${r['pnl']:>8,.0f} ${r['mdd']:>7,.0f} {r['pdd']:>4.1f}x {r['trades']:>5} {r['avg_wr']:>5.1f} {wl:>5} ${r['avg_d']:>6,.0f} ${r['med_d']:>5,.0f} {r['fill_pct']:>5.1f}  {r['cfg']}")

    with open(OUT / 'downstream_ranked.json', 'w') as f:
        json.dump(ranked[:50], f, indent=2)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=40)
    p.add_argument('--best-wait-bars', type=int, required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()

    log.info(f'DOWNSTREAM SWEEP — queue filter wb={a.best_wait_bars} × SL × TP × latency')
    log.info(f'SL: {SL_VALUES} | TP: {TP_VALUES} | Hold: {[h//60000 for h in HOLD_MS_VALUES]}m | Lat: {LATENCY_MS}ms')

    tasks = build_tasks(a.best_wait_bars)
    total = len(tasks)
    if not total: log.info('All done!'); aggregate(); return
    est = total * 36.0 / a.workers / 3600
    log.info(f'Tasks: {total:,} | Est: {est:.1f}h at {a.workers} workers')
    if a.dry_run: return

    done = failed = 0
    lock = threading.Lock()
    start = time.time()
    last_rpt = [time.time()]

    def track(task):
        nonlocal done, failed
        ok = run_task(task)
        with lock:
            if ok: done += 1
            else: failed += 1
            td = done + failed
            now = time.time()
            if now - last_rpt[0] >= 300:
                el = now - start
                rate = td / el * 60 if el > 0 else 0
                rem = (total - td) / (td / el) if td > 0 else 0
                log.info(f'Progress: {td}/{total} ({td/total*100:.1f}%) | {rate:.0f}/min | OK={done} FAIL={failed} | ETA: {rem/3600:.1f}h')
                last_rpt[0] = now
        return ok

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for _ in as_completed({ex.submit(track, t): t for t in tasks}): pass

    elapsed = time.time() - start
    log.info(f'COMPLETE: {done:,} OK, {failed:,} failed, {elapsed/3600:.2f}h')
    aggregate()

if __name__ == '__main__':
    main()
