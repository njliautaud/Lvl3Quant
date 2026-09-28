#!/usr/bin/env python3
"""Test variations of the Momentum Sniper strategy on all 54 OOT dates."""
import json, glob, os, sys, subprocess, logging, time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

LVL3 = Path(__file__).resolve().parent.parent.parent
PRED_DIR = LVL3 / 'data' / 'processed' / 'cnn_wf_stacked_predictions'
MBO_DIR = LVL3 / 'data' / 'raw' / 'mbo'
BINARY = LVL3 / 'rust_cache_builder' / 'target' / 'release' / 'fill_sim_cli.exe'
OUT_DIR = LVL3 / 'data' / 'processed' / 'momentum_variant_sweep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger('mom_sweep')
log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout)
h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
log.addHandler(h)

# Test momentum + ema_bookExit + smooth_smoothExit variants
CONFIGS = [
    # Momentum entry with various exits and params
    {"pred": "mom_emaExit_conv0.3_ethr0.0_vol70", "tp": [5, 8, 10, 15, 20, None], "sl": [None, 20, 25]},
    {"pred": "mom_emaExit_conv0.3_ethr0.5_vol70", "tp": [5, 8, 10, 15, 20, None], "sl": [None, 20, 25]},
    {"pred": "mom_emaExit_conv0.3_ethr0.0_vol50", "tp": [5, 10, 15, 20], "sl": [None, 25]},
    {"pred": "mom_emaExit_conv0.3_ethr0.5_vol50", "tp": [5, 10, 15, 20], "sl": [None, 25]},
    # EMA book exit
    {"pred": "ema_bookExit_conv1.5_vol70", "tp": [5, 8, 10, 15, 20], "sl": [None, 20, 25]},
    {"pred": "ema_bookExit_conv1.5_vol50", "tp": [5, 8, 10, 15], "sl": [None, 25]},
    {"pred": "ema_bookExit_conv2.0_vol70", "tp": [5, 8, 10, 15], "sl": [None, 25]},
    # Smooth smooth exit
    {"pred": "smooth_smoothExit_conv1.5_ethr0.0_vol70", "tp": [5, 8, 10, 15, 20, None], "sl": [None]},
    {"pred": "smooth_smoothExit_conv2.0_ethr0.0_vol70", "tp": [5, 8, 10, 15, 20, None], "sl": [None]},
    {"pred": "smooth_smoothExit_conv1.5_ethr0.5_vol70", "tp": [5, 8, 10, 15, 20, None], "sl": [None]},
]

HOLD_MS = 3600000
THRESHOLD = 0.1


def run_sim(pred_file, mbo_file, date, tp, sl, pred_name):
    tp_str = f"tp{tp}" if tp else "tpN"
    sl_str = f"sl{sl}" if sl else "slN"
    out_file = OUT_DIR / f"{pred_name}_{tp_str}_{sl_str}_{date}.json"
    if out_file.exists():
        return str(out_file), True
    cmd = [
        str(BINARY), '--mbo-file', str(mbo_file), '--predictions', str(pred_file),
        '--output', str(out_file), '--hold-ms', str(HOLD_MS),
        '--signal-threshold', str(THRESHOLD), '--latency-ms', '0', '--quiet',
        '--chase-entry', '--chase-max-ticks', '1', '--chase-max-reprices', '3',
    ]
    if tp: cmd.extend(['--take-profit-ticks', str(tp)])
    if sl: cmd.extend(['--trailing-ticks', str(sl)])
    try:
        subprocess.run(cmd, timeout=300, capture_output=True)
        return str(out_file), out_file.exists()
    except:
        return str(out_file), False


def main():
    log.info("MOMENTUM + EMA + SMOOTH VARIANT SWEEP")
    jobs = []
    for cfg in CONFIGS:
        pred_files = sorted(PRED_DIR.glob(f"*_{cfg['pred']}.npz"))
        for pf in pred_files:
            date = pf.stem[:10]
            nodash = date.replace('-', '')
            mbo = MBO_DIR / f'glbx-mdp3-{nodash}.mbo.dbn.zst'
            if not mbo.exists(): continue
            for tp in cfg['tp']:
                for sl in cfg['sl']:
                    jobs.append((str(pf), str(mbo), date, tp, sl, cfg['pred']))

    log.info(f"Total jobs: {len(jobs)}")
    done = 0; failed = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(run_sim, *j): j for j in jobs}
        for fut in as_completed(futures):
            _, success = fut.result()
            done += 1
            if not success: failed += 1
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                log.info(f"  [{done}/{len(jobs)}] {rate:.1f}/s, {failed} failed")

    log.info(f"Done: {done}, {failed} failed")

    # Aggregate
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
        if days < 20 or c["trades"] < 20: continue
        total = sum(c["pnls"])
        std = np.std(c["pnls"])
        sharpe = (total/days) / max(std, 1) * (252**0.5)
        wr = c["wins"] / max(c["trades"], 1) * 100
        pf = c["gp"] / max(c["gl"], 1)
        cum = 0; peak = 0; mdd = 0
        for p in c["pnls"]:
            cum += p
            if cum > peak: peak = cum
            if peak - cum > mdd: mdd = peak - cum
        scored.append({"name": name, "pnl": total, "trades": c["trades"], "days": days,
                       "sharpe": sharpe, "wr": wr, "pf": pf, "mdd": mdd, "ppt": total/max(c["trades"],1)})

    scored.sort(key=lambda x: x["sharpe"], reverse=True)
    log.info(f"\n{'CONFIG':<60} {'SHRP':>5} {'PnL':>8} {'TRAD':>5} {'WR%':>5} {'PF':>6} {'MDD':>7} {'$/tr':>6}")
    log.info("-" * 105)
    for r in scored[:25]:
        pf_s = f"{r['pf']:.1f}" if r['pf'] < 100 else "INF"
        log.info(f"{r['name'][:60]:<60} {r['sharpe']:>5.2f} ${r['pnl']:>7.0f} {r['trades']:>5} {r['wr']:>4.0f}% {pf_s:>6} ${r['mdd']:>6.0f} ${r['ppt']:>5.0f}")


if __name__ == '__main__':
    main()
