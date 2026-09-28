#!/usr/bin/env python3
"""
Test variations of the #1 Book Confirmed strategy to find the optimal config.
Sweep: TP (3,5,8,10,12,15), vol (50,60,70,80), conv (2.0,2.5,3.0)
All with slN (no stop loss) — confirmed dominant parameter.
"""
import json, glob, os, sys, subprocess, logging, time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

LVL3 = Path(__file__).resolve().parent.parent.parent
PRED_DIR = LVL3 / 'data' / 'processed' / 'cnn_wf_stacked_predictions'
MBO_DIR = LVL3 / 'data' / 'raw' / 'mbo'
BINARY = LVL3 / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
OUT_DIR = LVL3 / 'data' / 'processed' / 'book_confirmed_sweep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger('book_sweep')
log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
log.addHandler(h)

# Sweep parameters — variations around the winning Book Confirmed config
TP_VALUES = [3, 5, 8, 10, 12, 15, 20, None]
CONV_VALUES = [1.5, 2.0, 2.5]  # conv threshold (in prediction file name)
VOL_VALUES = [50, 70]  # vol filter (in prediction file name)

HOLD_MS = 3600000
THRESHOLD = 0.1
CHASE_TICKS = 1
CHASE_REPRICES = 3


def run_sim(pred_file, mbo_file, date, tp, conv, vol):
    tp_str = f"tp{tp}" if tp else "tpN"
    out_file = OUT_DIR / f"book_predstdExit_conv{conv}_vol{vol}_{tp_str}_slN_{date}.json"

    if out_file.exists():
        return str(out_file), True

    cmd = [
        str(BINARY),
        '--mbo-file', str(mbo_file),
        '--predictions', str(pred_file),
        '--output', str(out_file),
        '--hold-ms', str(HOLD_MS),
        '--signal-threshold', str(THRESHOLD),
        '--latency-ms', '0',
        '--quiet',
        '--chase-entry',
        '--chase-max-ticks', str(CHASE_TICKS),
        '--chase-max-reprices', str(CHASE_REPRICES),
    ]
    if tp:
        cmd.extend(['--take-profit-ticks', str(tp)])

    try:
        subprocess.run(cmd, timeout=300, capture_output=True)
        return str(out_file), out_file.exists()
    except:
        return str(out_file), False


def main():
    log.info("=" * 60)
    log.info("BOOK CONFIRMED VARIANT SWEEP")
    log.info(f"TP: {TP_VALUES} | Conv: {CONV_VALUES} | Vol: {VOL_VALUES}")
    log.info("=" * 60)

    jobs = []
    for conv in CONV_VALUES:
        for vol in VOL_VALUES:
            pred_pattern = f"book_predstdExit_conv{conv}_vol{vol}"
            pred_files = sorted(PRED_DIR.glob(f"*_{pred_pattern}.npz"))
            for pf in pred_files:
                date = pf.stem[:10]
                nodash = date.replace('-', '')
                mbo = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
                if not mbo.exists():
                    continue
                for tp in TP_VALUES:
                    jobs.append((str(pf), str(mbo), date, tp, conv, vol))

    log.info(f"Total jobs: {len(jobs)}")

    done = 0
    failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(run_sim, *j): j for j in jobs}
        for fut in as_completed(futures):
            _, success = fut.result()
            done += 1
            if not success: failed += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(jobs) - done) / max(rate, 0.01) / 60
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.0f}min, {failed} failed")

    log.info(f"\nDone: {done} jobs, {failed} failed")

    # Aggregate
    log.info("\n" + "=" * 60)
    log.info("RESULTS BY CONFIG")
    log.info("=" * 60)

    configs = defaultdict(lambda: {"pnls": [], "trades": 0, "wins": 0, "gp": 0, "gl": 0})

    for f in OUT_DIR.glob("*.json"):
        try:
            base = f.stem
            config = "_".join(base.split("_")[:-1])
            with open(f) as fh:
                d = json.load(fh)
            c = configs[config]
            pnl = d.get('total_pnl_dollars', 0)
            t = d.get('total_trades', 0)
            c["pnls"].append(pnl)
            c["trades"] += t
            if t > 0: c["wins"] += round(t * d.get('win_rate', 0))
            if pnl > 0: c["gp"] += pnl
            else: c["gl"] += abs(pnl)
        except:
            pass

    scored = []
    for name, c in configs.items():
        days = len(c["pnls"])
        if days < 20: continue
        total = sum(c["pnls"])
        mean = total / days
        std = np.std(c["pnls"])
        sharpe = mean / max(std, 1) * (252 ** 0.5)
        wr = c["wins"] / max(c["trades"], 1) * 100
        pf = c["gp"] / max(c["gl"], 1)
        prof_days = sum(1 for p in c["pnls"] if p > 0)

        cum = 0; peak = 0; max_dd = 0
        for p in c["pnls"]:
            cum += p
            if cum > peak: peak = cum
            dd = peak - cum
            if dd > max_dd: max_dd = dd

        scored.append({
            "name": name, "pnl": total, "trades": c["trades"], "days": days,
            "sharpe": sharpe, "wr": wr, "pf": pf, "max_dd": max_dd,
            "prof_days": prof_days, "ppt": total / max(c["trades"], 1)
        })

    scored.sort(key=lambda x: x["sharpe"], reverse=True)

    log.info(f"\n{'CONFIG':<55} {'SHRP':>5} {'PnL':>8} {'TRAD':>5} {'WR%':>5} {'PF':>6} {'MDD':>7} {'$/tr':>6}")
    log.info("-" * 100)
    for r in scored[:25]:
        pf_s = f"{r['pf']:.1f}" if r['pf'] < 100 else "INF"
        log.info(f"{r['name'][:55]:<55} {r['sharpe']:>5.2f} ${r['pnl']:>7.0f} {r['trades']:>5} {r['wr']:>4.0f}% {pf_s:>6} ${r['max_dd']:>6.0f} ${r['ppt']:>5.0f}")


if __name__ == '__main__':
    main()
