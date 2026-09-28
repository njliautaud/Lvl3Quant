#!/usr/bin/env python3
"""
meta_gated_fillsim.py
Meta-model gated fill sim — only trade when CNN z-score > threshold.

The fill_sim_cli --signal-threshold flag maps directly to |prediction| > threshold.
Since the wider CNN predictions have std ~0.47, z-score thresholds:
  z=2.0 → |pred| > 0.94
  z=2.5 → |pred| > 1.17
  z=3.0 → |pred| > 1.40

Tests 6 z-score x TP/SL x hold-time combos vs the unfiltered baseline.
Runs in parallel (4 workers), logs to /tmp/meta_gated_fillsim.log.
Results → /home/jupiter/Lvl3Quant/results/meta_gated_fillsim/
"""

import os
import json
import time
import logging
import subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR   = Path("/home/jupiter/Lvl3Quant/mbo_oot")
PRED_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/per_day_oos")
OOS_NPZ   = Path("/home/jupiter/Lvl3Quant/data/processed/wider_cnn_preds/oos_predictions_wider_cnn_oot_20260311_092055.npz")
OUT_DIR   = Path("/home/jupiter/Lvl3Quant/results/meta_gated_fillsim")
WORKERS   = 4

OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("/tmp/meta_gated_fillsim.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("meta_gated")

# CNN predictions std ~0.47 → z-score threshold = z * std
# We pass the raw |pred| threshold directly to --signal-threshold
# z=2.0 → 0.94, z=2.5 → 1.17, z=3.0 → 1.40

# std measured from the OOS npz on 2026-03-28 = 0.4668
PRED_STD = 0.4668

CONFIGS = [
    # Baseline: no z-gate (reproduces fill sim v2)
    {"name": "baseline_tp13_h2h",       "sig": 0.1,  "tp": 13, "sl": 0,  "hold_ms": 7200000,  "prime": False, "chase": False},
    # Z=2.0 gate variants
    {"name": "z20_tp10_sl20_h30s",      "sig": 2.0 * PRED_STD, "tp": 10, "sl": 20, "hold_ms": 30000,   "prime": False, "chase": False},
    {"name": "z20_tp13_sl20_h1m",       "sig": 2.0 * PRED_STD, "tp": 13, "sl": 20, "hold_ms": 60000,   "prime": False, "chase": False},
    {"name": "z20_tp13_sl20_h5m",       "sig": 2.0 * PRED_STD, "tp": 13, "sl": 20, "hold_ms": 300000,  "prime": False, "chase": False},
    {"name": "z20_tp13_sl20_h2h",       "sig": 2.0 * PRED_STD, "tp": 13, "sl": 20, "hold_ms": 7200000, "prime": False, "chase": False},
    # Z=2.5 gate variants
    {"name": "z25_tp10_sl20_h30s",      "sig": 2.5 * PRED_STD, "tp": 10, "sl": 20, "hold_ms": 30000,   "prime": False, "chase": False},
    {"name": "z25_tp13_sl40_h5m",       "sig": 2.5 * PRED_STD, "tp": 13, "sl": 40, "hold_ms": 300000,  "prime": False, "chase": False},
    {"name": "z25_tp13_sl20_h2h",       "sig": 2.5 * PRED_STD, "tp": 13, "sl": 20, "hold_ms": 7200000, "prime": False, "chase": False},
    # Z=3.0 gate variants (high conviction only)
    {"name": "z30_tp10_sl20_h30s",      "sig": 3.0 * PRED_STD, "tp": 10, "sl": 20, "hold_ms": 30000,   "prime": False, "chase": False},
    {"name": "z30_tp13_sl20_h2h",       "sig": 3.0 * PRED_STD, "tp": 13, "sl": 20, "hold_ms": 7200000, "prime": False, "chase": False},
    # Z=2.0 + prime hours (known +2.7 Sharpe lift)
    {"name": "z20_tp13_sl20_prime_h2h", "sig": 2.0 * PRED_STD, "tp": 13, "sl": 20, "hold_ms": 7200000, "prime": True,  "chase": False},
    {"name": "z25_tp13_sl20_prime_h5m", "sig": 2.5 * PRED_STD, "tp": 13, "sl": 20, "hold_ms": 300000,  "prime": True,  "chase": False},
    # Z=2.0 + signal flip exit
    {"name": "z20_tp13_sigflip_h2h",    "sig": 2.0 * PRED_STD, "tp": 13, "sl": 0,  "hold_ms": 7200000, "prime": False, "chase": False, "sigflip": True},
]


def ensure_per_day_files():
    """Extract per-day preds from consolidated OOS npz if not already done."""
    existing = set(f.stem.replace("_preds", "") for f in PRED_DIR.glob("*_preds.npz"))
    d = np.load(str(OOS_NPZ))
    keys = list(d.keys())
    dates = sorted(set(k.replace("_preds", "").replace("_mid", "") for k in keys))
    missing = [dt for dt in dates if dt not in existing]
    if missing:
        log.info(f"Creating {len(missing)} missing per-day pred files...")
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        for date in missing:
            if f"{date}_preds" not in d:
                continue
            preds = d[f"{date}_preds"].astype(np.float32)
            out = PRED_DIR / f"{date}_preds.npz"
            np.savez(str(out), predictions=preds)
    log.info(f"Per-day preds ready: {len(list(PRED_DIR.glob('*_preds.npz')))} files")


def run_one(args):
    date, cfg = args
    date_nodash = date.replace("-", "")
    mbo_file  = MBO_DIR / f"glbx-mdp3-{date_nodash}.mbo.dbn.zst"
    pred_file = PRED_DIR / f"{date}_preds.npz"
    out_file  = OUT_DIR / f"{date}_{cfg['name']}.json"

    if not mbo_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no_mbo"}
    if not pred_file.exists():
        return {"date": date, "config": cfg["name"], "error": "no_pred"}
    if out_file.exists():
        try:
            data = json.loads(out_file.read_text())
            data.update({"date": date, "config": cfg["name"]})
            return data
        except Exception:
            pass

    cmd = [
        FILL_SIM,
        "--mbo-file",    str(mbo_file),
        "--predictions", str(pred_file),
        "--output",      str(out_file),
        "--signal-threshold", str(round(cfg["sig"], 4)),
        "--hold-ms",     str(cfg["hold_ms"]),
        "--latency-ms",  "10",
    ]
    if cfg.get("tp", 0) > 0:
        cmd += ["--take-profit-ticks", str(cfg["tp"])]
    if cfg.get("sl", 0) > 0:
        cmd += ["--stop-loss-ticks", str(cfg["sl"])]
    if cfg.get("prime"):
        cmd += ["--prime-hours"]
    if cfg.get("chase"):
        cmd += ["--chase-entry"]
    if cfg.get("sigflip"):
        cmd += ["--signal-flip-exit"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return {"date": date, "config": cfg["name"], "error": result.stderr[:200]}
        data = json.loads(out_file.read_text())
        data.update({"date": date, "config": cfg["name"]})
        return data
    except subprocess.TimeoutExpired:
        return {"date": date, "config": cfg["name"], "error": "timeout"}
    except Exception as e:
        return {"date": date, "config": cfg["name"], "error": str(e)}


def summarize(results):
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_config[r["config"]].append(r)

    summary = {}
    for cfg_name, days in by_config.items():
        pnls = [d.get("total_pnl", 0) for d in days]
        trades = [d.get("total_trades", 0) for d in days]
        wins = [d.get("wins", 0) for d in days]
        total_trades = sum(trades)
        total_wins = sum(wins)
        total_pnl = sum(pnls)
        if total_trades > 0:
            wr = total_wins / total_trades
        else:
            wr = 0.0
        # Sortino-like: mean daily pnl / std of negative daily pnls
        arr = np.array(pnls)
        neg = arr[arr < 0]
        downside_std = np.std(neg) if len(neg) > 1 else (np.std(arr) + 1e-8)
        sortino = (np.mean(arr) / (downside_std + 1e-8)) * np.sqrt(252)
        summary[cfg_name] = {
            "n_days": len(days),
            "total_trades": total_trades,
            "total_pnl_ticks": round(total_pnl, 2),
            "win_rate": round(wr, 3),
            "sortino": round(float(sortino), 3),
            "avg_daily_pnl": round(float(np.mean(arr)), 2),
            "trades_per_day": round(total_trades / max(len(days), 1), 1),
        }

    # Sort by sortino
    ranked = sorted(summary.items(), key=lambda x: x[1]["sortino"], reverse=True)
    return summary, ranked


def main():
    log.info("=== META-GATED FILL SIM STARTING ===")
    log.info(f"FILL_SIM: {FILL_SIM}")
    log.info(f"Configs: {len(CONFIGS)}")

    ensure_per_day_files()

    # Build work items: all dates × all configs
    dates = sorted(set(
        f.name.replace("_preds.npz", "")
        for f in PRED_DIR.glob("*_preds.npz")
    ))
    log.info(f"Dates available: {len(dates)}")

    work = [(date, cfg) for date in dates for cfg in CONFIGS]
    log.info(f"Total tasks: {len(work)}")

    all_results = []
    completed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, w): w for w in work}
        for fut in as_completed(futures):
            res = fut.result()
            all_results.append(res)
            completed += 1
            if completed % 50 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = (len(work) - completed) / max(rate, 0.001)
                log.info(f"Progress: {completed}/{len(work)} | {rate:.1f}/s | ETA {remaining/60:.1f}m")

    # Save all raw results
    raw_out = OUT_DIR / "all_results_raw.json"
    raw_out.write_text(json.dumps(all_results, indent=2))
    log.info(f"Raw results saved to {raw_out}")

    # Summarize
    summary, ranked = summarize(all_results)
    summary_out = OUT_DIR / "summary.json"
    summary_out.write_text(json.dumps({"summary": summary, "ranked_by_sortino": ranked}, indent=2))

    log.info("=== RESULTS RANKED BY SORTINO ===")
    for cfg_name, stats in ranked[:10]:
        log.info(
            f"  {cfg_name}: Sortino={stats['sortino']:.3f} | "
            f"WR={stats['win_rate']:.1%} | "
            f"PnL={stats['total_pnl_ticks']:.0f}t | "
            f"Trades/day={stats['trades_per_day']:.0f} | "
            f"Days={stats['n_days']}"
        )

    log.info("=== META-GATED FILL SIM COMPLETE ===")
    elapsed = time.time() - t0
    log.info(f"Total time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
