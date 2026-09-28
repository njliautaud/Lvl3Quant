#!/usr/bin/env python3
"""
wider_cnn_fill_sim.py - Run fill_sim on 11 wider CNN WF prediction dates.

Uses the wider CNN predictions from hybrid_vs_wider_preds/wider/ directory.
Runs with 2 workers (nice 19) to avoid interfering with existing sweeps.
Tests our best card configs against the wider CNN model predictions.
"""

import os
import sys
import json
import time
import subprocess
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/hybrid_vs_wider_preds/wider")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/wider_cnn_fill_sim")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 2  # Low to avoid overloading (already 54 fill_sim procs running)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("wider_cnn_sim")

# Dates with both predictions AND MBO data
DATES = [
    "2025-07-22", "2025-07-23", "2025-07-24", "2025-07-25",
    "2025-07-28", "2025-07-29", "2025-07-30", "2025-07-31",
    "2025-08-01", "2025-08-04", "2025-08-05",
]

# Card configs - our best optimized settings from 371-sweep
# Note: wider CNN predictions are a single model, so we test multiple card param sets
CONFIGS = [
    # Card 1 style: TP13, 2h hold (our best standard CNN card1)
    {"name": "wider_tp13_h2h", "tp": 13, "hold_ms": 7200000, "sig": 0.1, "latency": 10},
    # Card 1 optimized: TP8, 1h, MAE 20t/300s
    {"name": "wider_tp8_h1h_mae20", "tp": 8, "hold_ms": 3600000, "sig": 0.1, "latency": 10, "mae_ticks": 20, "mae_secs": 300},
    # Card 2 style: TP15, 2h
    {"name": "wider_tp15_h2h", "tp": 15, "hold_ms": 7200000, "sig": 0.1, "latency": 10},
    # Card 4 style: TP20, 2h, sig 0.3
    {"name": "wider_tp20_h2h_sig03", "tp": 20, "hold_ms": 7200000, "sig": 0.3, "latency": 10},
    # Card 7 style: TP25, 2h
    {"name": "wider_tp25_h2h", "tp": 25, "hold_ms": 7200000, "sig": 0.1, "latency": 10},
    # HFT style: TP4, chase entry, 120s hold
    {"name": "wider_hft_tp4_chase", "tp": 4, "hold_ms": 120000, "sig": 0.1, "latency": 10, "chase": True},
    # Wide TP: TP30, 2h, sig 0.5 (wider CNN may predict bigger moves)
    {"name": "wider_tp30_h2h_sig05", "tp": 30, "hold_ms": 7200000, "sig": 0.5, "latency": 10},
    # No TP, just hold: pure alpha test
    {"name": "wider_notp_h2h", "tp": None, "hold_ms": 7200000, "sig": 0.1, "latency": 10},
    # Scalp: TP6, 5min hold
    {"name": "wider_scalp_tp6_5m", "tp": 6, "hold_ms": 300000, "sig": 0.3, "latency": 10},
    # MAE exit only: no TP, MAE 25t/600s
    {"name": "wider_mae25_h2h", "tp": None, "hold_ms": 7200000, "sig": 0.1, "latency": 10, "mae_ticks": 25, "mae_secs": 600},
]


def run_one(args):
    """Run fill_sim for one date + config combo."""
    date, cfg = args
    date_nodash = date.replace("-", "")
    mbo_file = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
    pred_file = PRED_DIR / f"{date}_preds.npz"
    out_file = OUTPUT_DIR / f"{date}_{cfg['name']}.json"

    if not mbo_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no MBO file"}
    if not pred_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no pred file"}
    if out_file.exists():
        try:
            with open(out_file) as f:
                data = json.load(f)
                data["date"] = date
                data["config"] = cfg["name"]
                return data
        except:
            pass

    cmd = [
        "nice", "-n", "19",
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(cfg["sig"]),
        "--hold-ms", str(cfg["hold_ms"]),
        "--latency-ms", str(cfg.get("latency", 10)),
    ]

    if cfg.get("tp"):
        cmd += ["--take-profit-ticks", str(cfg["tp"])]
    if cfg.get("mae_ticks"):
        cmd += ["--vol-exit-ticks", str(cfg["mae_ticks"])]
    if cfg.get("mae_secs"):
        cmd += ["--vol-exit-bars", str(cfg["mae_secs"] * 10)]  # 100ms bars
    if cfg.get("chase"):
        cmd += ["--chase-entry", "--chase-max-ticks", "1", "--chase-max-reprices", "3"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            return {"date": date, "config": cfg["name"], "error": result.stderr[:500]}
        if out_file.exists():
            with open(out_file) as f:
                data = json.load(f)
                data["date"] = date
                data["config"] = cfg["name"]
                return data
        return {"date": date, "config": cfg["name"], "error": "no output file"}
    except subprocess.TimeoutExpired:
        return {"date": date, "config": cfg["name"], "error": "timeout"}
    except Exception as e:
        return {"date": date, "config": cfg["name"], "error": str(e)}


def main():
    tasks = []
    for date in DATES:
        for cfg in CONFIGS:
            tasks.append((date, cfg))

    total = len(tasks)
    log.info(f"Starting wider CNN fill_sim: {total} tasks ({len(DATES)} dates x {len(CONFIGS)} configs)")
    log.info(f"Workers: {WORKERS}, Output: {OUTPUT_DIR}")

    results = []
    completed = 0
    start = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, (d, c)): (d, c["name"]) for d, c in tasks}
        for future in as_completed(futures):
            completed += 1
            date, cname = futures[future]
            try:
                r = future.result()
                results.append(r)
                trades = r.get("total_trades", r.get("num_trades", "?"))
                pnl = r.get("total_pnl", r.get("net_pnl", "?"))
                sharpe = r.get("sharpe_ratio", r.get("sharpe", "?"))
                err = r.get("error", "")
                if err:
                    log.warning(f"  [{completed}/{total}] {date} {cname}: ERROR {err}")
                else:
                    log.info(f"  [{completed}/{total}] {date} {cname}: trades={trades} pnl=${pnl} sharpe={sharpe}")
            except Exception as e:
                log.error(f"  [{completed}/{total}] {date} {cname}: EXCEPTION {e}")
                results.append({"date": date, "config": cname, "error": str(e)})

            if completed % 10 == 0:
                elapsed = time.time() - start
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                log.info(f"  Progress: {completed}/{total} ({rate:.1f}/s, ETA {eta:.0f}s)")

    # Aggregate results by config
    log.info("\n" + "=" * 80)
    log.info("WIDER CNN FILL SIM RESULTS SUMMARY")
    log.info("=" * 80)

    by_config = defaultdict(list)
    for r in results:
        if not r.get("error"):
            by_config[r.get("config", "unknown")].append(r)

    summary = {}
    for cname, runs in sorted(by_config.items()):
        pnls = []
        trades_total = 0
        fills = 0
        signals = 0
        for r in runs:
            p = r.get("total_pnl", r.get("net_pnl", 0))
            if isinstance(p, (int, float)):
                pnls.append(p)
            t = r.get("total_trades", r.get("num_trades", 0))
            if isinstance(t, (int, float)):
                trades_total += t
            f = r.get("total_fills", r.get("fills", 0))
            if isinstance(f, (int, float)):
                fills += f
            s = r.get("total_signals", r.get("signals", 0))
            if isinstance(s, (int, float)):
                signals += s

        pnls_arr = np.array(pnls) if pnls else np.array([0])
        avg_pnl = pnls_arr.mean()
        std_pnl = pnls_arr.std() if len(pnls_arr) > 1 else 0
        sharpe = (avg_pnl / std_pnl * (252 ** 0.5)) if std_pnl > 0 else 0
        total_pnl = pnls_arr.sum()
        fill_rate = fills / signals * 100 if signals > 0 else 0

        summary[cname] = {
            "total_pnl": float(total_pnl),
            "avg_daily_pnl": float(avg_pnl),
            "sharpe_annual": float(sharpe),
            "total_trades": int(trades_total),
            "fill_rate_pct": float(fill_rate),
            "days": len(runs),
        }

        log.info(f"\n{cname}:")
        log.info(f"  Total PnL: ${total_pnl:,.0f} | Avg Daily: ${avg_pnl:,.0f} | Sharpe: {sharpe:.2f}")
        log.info(f"  Trades: {trades_total} | Fill Rate: {fill_rate:.1f}% | Days: {len(runs)}")

    # Save summary
    summary_file = OUTPUT_DIR / "wider_cnn_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"summary": summary, "raw_results": results, "timestamp": datetime.now().isoformat()}, f, indent=2, default=str)

    log.info(f"\nSummary saved to {summary_file}")
    log.info(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
