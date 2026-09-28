#!/usr/bin/env python3
"""exec_queue_v2 -- Variance Reduction. Goal: MC p5 Sortino > 1.0"""
import numpy as np, subprocess, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

FILL_SIM = Path("/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli")
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
SINGLE_PREDS = Path("/home/jupiter/Lvl3Quant/fill_sim_test/single_preds")
BASE_OUT = Path("/home/jupiter/Lvl3Quant/results/exec_queue_v2")
BASE_OUT.mkdir(parents=True, exist_ok=True)
WORKERS = 14
N_MC = 1000

matched = []
for mbo in sorted(MBO_DIR.glob("*.dbn.zst")):
    date8 = mbo.name.split("-")[2].split(".")[0]
    pred = SINGLE_PREDS / (date8 + ".npz")
    if pred.exists():
        matched.append((mbo, date8, pred))
print("Matched %d date/MBO pairs" % len(matched))

EXPERIMENTS = {
    "exp1_gate20_tp13_sl40":  dict(tp=13,sl=40,hold=50000,thresh=0.20,prime=False),
    "exp2_gate35_tp13_sl40":  dict(tp=13,sl=40,hold=50000,thresh=0.35,prime=False),
    "exp3_sl20_tp13":         dict(tp=13,sl=20,hold=50000,thresh=0.05,prime=False),
    "exp4_sl20_prime":        dict(tp=13,sl=20,hold=50000,thresh=0.05,prime=True),
    "exp5_gate20_sl20":       dict(tp=13,sl=20,hold=50000,thresh=0.20,prime=False),
    "exp6_prime_only":        dict(tp=13,sl=40,hold=50000,thresh=0.05,prime=True),
    "exp7_gate30_prime":      dict(tp=13,sl=40,hold=50000,thresh=0.30,prime=True),
    "exp8_tp10_sl20":         dict(tp=10,sl=20,hold=50000,thresh=0.05,prime=False),
    "exp9_tp16_prime":        dict(tp=16,sl=40,hold=50000,thresh=0.05,prime=True),
    "exp10_tp13_sl30_g15p":   dict(tp=13,sl=30,hold=50000,thresh=0.15,prime=True),
}

def run_one(exp_name, exp, mbo, date8, pred):
    out_dir = BASE_OUT / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (date8 + ".json")
    if out_file.exists():
        return (exp_name, date8, True)
    cmd = [str(FILL_SIM),
        "--mbo-file", str(mbo), "--predictions", str(pred),
        "--output", str(out_file), "--hold-ms", str(exp["hold"]),
        "--signal-threshold", str(exp["thresh"]),
        "--take-profit-ticks", str(exp["tp"]),
        "--stop-loss-ticks", str(exp["sl"]),
        "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5", "--quiet"
    ]
    if exp.get("prime"):
        cmd.append("--prime-hours")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (exp_name, date8, r.returncode == 0)
    except:
        return (exp_name, date8, False)

tasks = [(en,ev,mbo,d8,pred) for en,ev in EXPERIMENTS.items() for mbo,d8,pred in matched]
print("Total tasks: %d" % len(tasks))
done = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = {ex.submit(run_one,*t):t for t in tasks}
    for fut in as_completed(futs):
        done += 1
        if done % 100 == 0:
            print("  %d/%d (%d%%)" % (done,len(tasks),done*100//len(tasks)))

print("\nAggregating...")
summary = {}
for exp_name in EXPERIMENTS:
    daily_pnl = []
    for f in sorted((BASE_OUT/exp_name).glob("*.json")):
        try:
            daily_pnl.append(json.load(open(f)).get("total_pnl_dollars",0.0))
        except: pass
    if not daily_pnl: continue
    arr = np.array(daily_pnl); n = len(arr); mean = float(arr.mean())
    neg = arr[arr<0]; dstd = float(np.sqrt(np.mean(neg**2))) if len(neg)>0 else 1e-9
    mc_s = []
    for _ in range(N_MC):
        samp = np.random.choice(arr,size=n,replace=True)
        ng = samp[samp<0]; ds = float(np.sqrt(np.mean(ng**2))) if len(ng)>0 else 1e-9
        mc_s.append(float(samp.mean())/ds)
    mc = np.array(mc_s)
    summary[exp_name] = dict(n_days=n,total_pnl=round(float(arr.sum()),2),
        sortino=round(mean/dstd,4),mc_p5=round(float(np.percentile(mc,5)),4),
        mc_median=round(float(np.median(mc)),4),mc_p95=round(float(np.percentile(mc,95)),4),
        mc_prob_profit=round(float((mc>0).mean()),4))

sorted_e = sorted(summary.items(),key=lambda x:x[1]["mc_p5"],reverse=True)
print("\n%-35s %8s %8s %8s %10s" % ("Exp","Sortino","MC_p5","MC_med","PnL"))
print("-"*70)
for en,s in sorted_e:
    print("%-35s %8.3f %8.3f %8.3f %10.0f" % (en,s["sortino"],s["mc_p5"],s["mc_median"],s["total_pnl"]))

json.dump(summary,open(BASE_OUT/"exec_queue_v2_results.json","w"),indent=2)
print("\nDone.")
