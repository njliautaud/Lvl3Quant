#!/usr/bin/env python3
"""GNN OOT Sweep - runs fill_sim across all GNN prediction dates.
Configs: TP3/TP8/TP15 x SL_none/trail25 x conv thresholds.
"""
import subprocess, glob, os, sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BINARY = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/gnn_sim_results"
os.makedirs(OUT_DIR, exist_ok=True)

WORKERS = 14

# Build date -> mbo mapping
mbo_map = {}
for f in glob.glob(os.path.join(MBO_DIR, "*.mbo.dbn.zst")):
    bn = os.path.basename(f)
    d = bn.replace("glbx-mdp3-", "").replace(".mbo.dbn.zst", "")
    if len(d) == 8:
        date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        mbo_map[date] = f

# Find GNN prediction files
gnn_preds = {}
for f in glob.glob(os.path.join(PRED_DIR, "*_gnn_predictions.npz")):
    bn = os.path.basename(f)
    date = bn[:10]  # YYYY-MM-DD
    gnn_preds[date] = f

avail_dates = sorted(set(gnn_preds.keys()) & set(mbo_map.keys()))
print(f"GNN pred dates: {len(gnn_preds)}, MBO dates: {len(mbo_map)}, Overlap: {len(avail_dates)}")
print(f"Date range: {avail_dates[0]} to {avail_dates[-1]}")

# Sweep configs:
# - Take profit: 3, 8, 15 ticks
# - Stop loss: none, trailing 25 ticks
# - Signal threshold: 1.5, 2.0, 2.5
# - Hold: 30min (1800000ms), 10min (600000ms)
# - Chase: 1t/3r (best from CNN)
# - Latency: 0ms, 50ms
configs = []
for tp in [3, 8, 15]:
    for sl_type, sl_val in [("none", 0), ("trail25", 25)]:
        for conv in [1.5, 2.0, 2.5]:
            for hold_min in [10, 30]:
                for lat in [0, 50]:
                    label = f"gnn_tp{tp}_{sl_type}_conv{conv}_hold{hold_min}m_lat{lat}"
                    configs.append({
                        "label": label,
                        "tp": tp,
                        "sl_type": sl_type,
                        "sl_val": sl_val,
                        "conv": conv,
                        "hold_ms": hold_min * 60000,
                        "lat": lat,
                        "chase_t": 1,
                        "chase_r": 3,
                    })

print(f"Configs: {len(configs)}")
print(f"Total jobs: {len(configs) * len(avail_dates)}")

# Build job list
jobs = []
for date in avail_dates:
    mbo = mbo_map[date]
    pred = gnn_preds[date]
    for cfg in configs:
        out_file = os.path.join(OUT_DIR, f"{cfg['label']}_{date}.json")
        if os.path.exists(out_file):
            continue
        jobs.append({
            "date": date,
            "mbo": mbo,
            "pred": pred,
            "out": out_file,
            **cfg,
        })

print(f"Jobs to run (skipping existing): {len(jobs)}")
if not jobs:
    print("All jobs already done!")
    sys.exit(0)


def run_job(job):
    cmd = [
        BINARY,
        "--mbo-file", job["mbo"],
        "--predictions", job["pred"],
        "--output", job["out"],
        "--hold-ms", str(job["hold_ms"]),
        "--signal-threshold", str(job["conv"]),
        "--latency-ms", str(job["lat"]),
        "--chase-entry",
        "--chase-max-ticks", str(job["chase_t"]),
        "--chase-max-reprices", str(job["chase_r"]),
        "--chase-interval-ms", "100",
        "--quiet",
    ]
    if job["tp"] > 0:
        cmd.extend(["--take-profit-ticks", str(job["tp"])])
    if job["sl_type"] == "trail25":
        cmd.extend(["--trailing-ticks", str(job["sl_val"])])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return {"status": "error", "label": job["label"], "date": job["date"], "err": r.stderr[:200]}
        if os.path.exists(job["out"]):
            with open(job["out"]) as f:
                result = json.load(f)
            return {"status": "ok", "label": job["label"], "date": job["date"],
                    "pnl": result.get("total_pnl_dollars", 0),
                    "trades": result.get("total_trades", 0)}
        return {"status": "no_output", "label": job["label"], "date": job["date"]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "label": job["label"], "date": job["date"]}
    except Exception as e:
        return {"status": "exception", "label": job["label"], "date": job["date"], "err": str(e)}


print(f"\nStarting sweep with {WORKERS} workers...")
t0 = time.time()
done = 0
errors = 0
total_pnl = 0.0

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(run_job, j): j for j in jobs}
    for future in as_completed(futures):
        done += 1
        try:
            result = future.result()
            if result["status"] == "ok":
                total_pnl += result.get("pnl", 0)
            else:
                errors += 1
        except Exception as e:
            errors += 1

        if done % 50 == 0 or done == len(jobs):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (len(jobs) - done) / max(rate, 0.01)
            print(f"  [{done}/{len(jobs)}] {rate:.1f}/s, ~{remaining/60:.1f}min left, {errors} errors, running_pnl=${total_pnl:,.0f}")

elapsed = time.time() - t0
print(f"\nDone: {done} jobs in {elapsed:.0f}s ({errors} errors)")

# Aggregate results
print("\n" + "="*100)
print("AGGREGATING GNN SWEEP RESULTS")
print("="*100)

config_results = {}
for f in glob.glob(os.path.join(OUT_DIR, "gnn_*.json")):
    bn = os.path.basename(f).replace(".json", "")
    # Split: last part is date (YYYY-MM-DD)
    parts = bn.rsplit("_", 3)
    date = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
    label = bn[:-(len(date)+1)]

    try:
        with open(f) as fh:
            data = json.load(fh)
    except:
        continue

    if label not in config_results:
        config_results[label] = {"pnls": [], "trades": 0, "wins": 0, "dates": []}

    day_pnl = data.get("total_pnl_dollars", 0)
    day_trades = data.get("total_trades", 0)
    config_results[label]["pnls"].append(day_pnl)
    config_results[label]["trades"] += day_trades
    config_results[label]["dates"].append(date)

    if "trades" in data:
        for t in data["trades"]:
            if t.get("pnl_dollars", 0) > 0:
                config_results[label]["wins"] += 1

import numpy as np

summary = []
for label, cr in config_results.items():
    pnls = cr["pnls"]
    if not pnls or cr["trades"] == 0:
        continue
    total = sum(pnls)
    n_days = len(pnls)
    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
    wr = cr["wins"] / cr["trades"] if cr["trades"] > 0 else 0

    summary.append({
        "config": label,
        "total_pnl": round(total, 2),
        "n_days": n_days,
        "trades": cr["trades"],
        "trades_per_day": round(cr["trades"] / n_days, 1),
        "win_rate": round(wr, 4),
        "sharpe": round(sharpe, 2),
        "avg_daily": round(avg, 2),
        "annualized": round(avg * 252, 0),
    })

summary.sort(key=lambda x: x["sharpe"], reverse=True)

print(f"\n{'Config':<60} {'P&L':>10} {'Days':>5} {'Trades':>7} {'WR':>6} {'Sharpe':>7} {'Annual':>10}")
print("-" * 115)
for s in summary[:30]:
    print(f"{s['config']:<60} ${s['total_pnl']:>9,.0f} {s['n_days']:>5} {s['trades']:>7} {s['win_rate']:>5.1%} {s['sharpe']:>7.2f} ${s['annualized']:>9,.0f}")

# Save summary
with open(os.path.join(OUT_DIR, "gnn_sweep_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {OUT_DIR}/gnn_sweep_summary.json")

# Also save top 10 for quick reference
print(f"\n{'='*80}")
print("TOP 10 GNN CONFIGS BY SHARPE:")
print(f"{'='*80}")
for i, s in enumerate(summary[:10]):
    print(f"  {i+1}. {s['config']}")
    print(f"     Sharpe={s['sharpe']:.2f}, P&L=${s['total_pnl']:,.0f}/{s['n_days']}d, {s['trades']} trades, WR={s['win_rate']:.1%}, Annual=${s['annualized']:,.0f}")
