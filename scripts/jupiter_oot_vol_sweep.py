#!/usr/bin/env python3
"""
jupiter_oot_vol_sweep.py - Run fill sim sweep on OOT vol-filtered predictions.

Tests vol50/70/80 filtered predictions on 68 OOT dates with best configs.
Mirrors what Saturn runs, but on Jupiter (which has more MBO data: 101 files).
Also tests the NEW WF backup predictions for comparison.

Key question: Does vol filtering (only trading when vol > threshold percentile)
improve or hurt performance?
"""

import os
import json
import glob
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_oot_sim_predictions")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/oot_vol_sweep_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 12

# Best configs from v2 sweep (confirmed positive OOT)
CONFIGS = {
    "A_tp13_sl40_t050": "--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 --prime-hours --signal-threshold 0.5",
    "A_tp13_sl40_lat20": "--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 --prime-hours --signal-threshold 0.5 --latency-ms 20",
    "A5_tp10_sl40_t100": "--take-profit-ticks 10 --stop-loss-ticks 40 --hold-ms 1800000 --prime-hours --signal-threshold 1.0",
    # New: try dynamic vol-aware configs
    "A_tp13_sl40_t075_lat10": "--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000 --prime-hours --signal-threshold 0.75 --latency-ms 10",
}

VOL_LEVELS = ["vol50", "vol70", "vol80"]
SESSION = "morning_afternoon"


def get_tasks():
    pred_files = sorted(PRED_DIR.glob("*_%s.npz" % SESSION))
    vol_date_map = {}
    for pf in pred_files:
        name = pf.stem
        parts = name.split("_")
        date = parts[0]
        vol = parts[1]
        vol_date_map[(date, vol)] = pf

    dates = sorted(set(d for d, v in vol_date_map.keys()))
    print("Available dates: %d (%s -> %s)" % (len(dates), dates[0], dates[-1]))

    tasks = []
    for date in dates:
        date8 = date.replace("-", "")
        mbo_file = MBO_DIR / ("glbx-mdp3-%s.mbo.dbn.zst" % date8)
        if not mbo_file.exists():
            continue

        for vol in VOL_LEVELS:
            pred_file = vol_date_map.get((date, vol))
            if pred_file is None:
                continue

            for config_name, config_flags in CONFIGS.items():
                out_file = OUT_DIR / ("%s_%s_%s.json" % (date8, vol, config_name))
                if out_file.exists():
                    continue
                tasks.append((date, vol, config_name, config_flags, mbo_file, pred_file, out_file))

    return tasks


def run_one(task):
    date, vol, config_name, config_flags, mbo_file, pred_file, out_file = task
    cmd = [FILL_SIM, "--mbo-file", str(mbo_file), "--predictions", str(pred_file),
           "--output", str(out_file)] + config_flags.split()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if out_file.exists():
            with open(out_file) as f:
                d = json.load(f)
            return {
                "date": date, "vol": vol, "config": config_name,
                "pnl": d.get("total_pnl_dollars", 0),
                "trades": d.get("total_trades", 0),
                "win_rate": d.get("win_rate", 0),
                "ok": True
            }
        return {"date": date, "vol": vol, "config": config_name, "ok": False,
                "error": "no output: " + result.stderr[:200]}
    except Exception as e:
        return {"date": date, "vol": vol, "config": config_name, "ok": False, "error": str(e)}


def aggregate():
    summary = defaultdict(lambda: {"pnls": [], "trades": 0})
    for f in sorted(OUT_DIR.glob("*.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)
            name = f.stem
            parts = name.split("_", 2)
            if len(parts) < 3:
                continue
            date8, vol, config = parts[0], parts[1], parts[2]
            key = "%s_%s" % (vol, config)
            summary[key]["pnls"].append(d.get("total_pnl_dollars", 0))
            summary[key]["trades"] += d.get("total_trades", 0)
        except:
            pass

    print("\n=== JUPITER OOT VOL SWEEP RESULTS ===")
    print("%-10s %-35s %8s %8s %6s %6s" % ("VOL", "CONFIG", "TOTAL", "AVG/DAY", "DATES", "TRADES"))
    print("-" * 80)

    rows = []
    for key, data in summary.items():
        pnls = data["pnls"]
        if not pnls:
            continue
        total = sum(pnls)
        avg = np.mean(pnls)
        neg = [p for p in pnls if p < 0]
        sortino = np.mean(pnls) / np.std(neg) if neg else 999
        wins = sum(1 for p in pnls if p > 0)
        vol = key.split("_")[0]
        config = "_".join(key.split("_")[1:])
        rows.append((total, vol, config, avg, sortino, wins, len(pnls), data["trades"]))

    rows.sort(key=lambda x: -x[0])
    for total, vol, config, avg, sortino, wins, n, trades in rows:
        print("%-10s %-35s %8.0f %8.0f %6d/%d %6d" % (vol, config, total, avg, wins, n, trades))

    # Save JSON summary
    summary_out = {}
    for key, data in summary.items():
        pnls = data["pnls"]
        if not pnls:
            continue
        neg = [p for p in pnls if p < 0]
        sortino = float(np.mean(pnls) / np.std(neg)) if neg else 0.0
        summary_out[key] = {
            "total_pnl": sum(pnls),
            "avg_daily_pnl": float(np.mean(pnls)),
            "sortino": sortino,
            "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            "n_days": len(pnls),
            "n_trades": data["trades"]
        }

    out_path = str(OUT_DIR / "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print("\nSummary saved:", out_path)
    return summary_out


def main():
    tasks = get_tasks()
    print("Tasks to run: %d" % len(tasks))

    if not tasks:
        print("All done. Aggregating...")
        aggregate()
        return

    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for future in as_completed(futures):
            r = future.result()
            done += 1
            if not r.get("ok"):
                failed += 1
                if failed <= 3:
                    print("FAIL: %s %s %s - %s" % (r["date"], r["vol"], r["config"], r.get("error", "?")))
            elif done % 100 == 0:
                print("[%d/%d] %s %s %s PnL=%.0f" % (done, len(tasks), r["date"], r["vol"], r["config"], r["pnl"]))

    print("Done: %d/%d (failed: %d)" % (done - failed, len(tasks), failed))
    aggregate()


if __name__ == "__main__":
    main()
