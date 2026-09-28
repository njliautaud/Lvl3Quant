#!/usr/bin/env python3
"""
targeted_fillsim_sweep.py -- EXECUTION RESEARCH: Targeted fill sim re-run
===========================================================================
Goal: Find Sortino > 4.0 by testing the right parameter space with:
  - Correct --signal-threshold flag (was the bug in prior 884-config sweep)
  - C1 (conv1.5_vol50) and C4 (conv2.0_vol70) predictions
  - Time-of-day filter via --prime-hours (12:00-14:00 ET is prime window)
  - Multiple TP/SL combinations at multiple conviction thresholds

Known results to beat:
  - C1 TP13/SL20 z>2.0: Sortino=1.691 (tp_sl sweep)
  - A_tp13_sl40 t=0.5 (C5 stacked): Sortino=2.726 (exec_queue_v1)
  - C7 z>2.5 TP8 hold600s: Sortino=2.768 (multi_card_conviction)
  - C3 baseline TP10 hold3600s: Sortino=2.05 (conviction_refined)

Target: Sortino > 4.0 via combination of:
  1. Right TP/SL ratio (TP>=SL)
  2. Time-of-day filtering (prime_hours)
  3. Conviction threshold tuning
"""
import json, logging, subprocess, datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
import numpy as np

FILL_SIM = Path("/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli")
MBO_DIR  = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUT_DIR  = Path("/home/jupiter/Lvl3Quant/data/processed/targeted_fillsim_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("/home/jupiter/Lvl3Quant/logs/targeted_fillsim.log")

OOT_START = date(2025, 12, 1)
OOT_END   = date(2026, 3, 8)
WORKERS   = 14

CARDS = {
    "c1": {"pred_suffix": "book_predstdExit_conv1.5_vol50"},
    "c4": {"pred_suffix": "book_predstdExit_conv2.0_vol70"},
}

Z_THRESHOLDS = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5]

TP_SL_CONFIGS = [
    (13, 20), (13, 15), (13, 10),
    (20, 20), (20, 15), (20, 10),
    (15, 15), (15, 10),
    (13, 40), (10, 40), (8, 40),
    (8, 8), (10, 10), (8, 20),
    (4, 2), (3, 2), (2, 2),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE)),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("targeted_fillsim")


def discover_dates():
    dates = []
    d = OOT_START
    while d <= OOT_END:
        mbo = MBO_DIR / ("glbx-mdp3-" + d.strftime("%Y%m%d") + ".mbo.dbn.zst")
        if mbo.exists():
            dates.append(d)
        d += datetime.timedelta(days=1)
    return dates


def run_fill_sim(card_id, card_def, tp, sl, z_thresh, date_obj):
    date_iso = date_obj.isoformat()
    date_num = date_obj.strftime("%Y%m%d")
    mbo_file  = MBO_DIR  / ("glbx-mdp3-" + date_num + ".mbo.dbn.zst")
    pred_file = PRED_DIR / (date_iso + "_" + card_def["pred_suffix"] + ".npz")
    if not mbo_file.exists() or not pred_file.exists():
        return None
    z_str = str(z_thresh).replace(".", "p")
    label = "tp" + str(tp) + "_sl" + str(sl) + "_z" + z_str
    card_out_dir = OUT_DIR / card_id / label
    card_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = card_out_dir / (date_iso + ".json")
    if out_path.exists():
        try:
            with open(out_path) as f:
                return json.load(f)
        except Exception:
            pass
    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_path),
        "--signal-threshold", str(z_thresh),
        "--take-profit-ticks", str(tp),
        "--stop-loss-ticks", str(sl),
        "--hold-ms", "7200000",
        "--latency-ms", "20",
        "--prime-hours",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and out_path.exists():
            with open(out_path) as f:
                return json.load(f)
    except Exception as e:
        log.warning("Error " + card_id + " TP" + str(tp) + "/SL" + str(sl) + " z" + str(z_thresh) + " " + date_iso + ": " + str(e))
    return None


def sortino_calc(daily_pnls, mar=0.0):
    if len(daily_pnls) < 5:
        return 0.0
    arr = np.array(daily_pnls, dtype=float)
    excess = arr - mar
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float(np.mean(excess)) / 1e-6 * (252 ** 0.5)
    ds = float(np.sqrt(np.mean(downside ** 2)))
    if ds == 0:
        return 0.0
    return float(np.mean(excess)) / ds * (252 ** 0.5)


def main():
    dates = discover_dates()
    log.info("=" * 70)
    log.info("TARGETED FILL SIM SWEEP -- Execution Research Lead")
    log.info("Cards: " + str(list(CARDS.keys())) + ", TP/SL: " + str(len(TP_SL_CONFIGS)) + " combos")
    log.info("Z thresholds: " + str(Z_THRESHOLDS))
    log.info("OOT: " + str(len(dates)) + " days (" + str(dates[0]) + " to " + str(dates[-1]) + ")")
    total = len(CARDS) * len(TP_SL_CONFIGS) * len(Z_THRESHOLDS) * len(dates)
    log.info("Total jobs: " + str(total))
    log.info("=" * 70)

    jobs = []
    for card_id, card_def in CARDS.items():
        for (tp, sl) in TP_SL_CONFIGS:
            for z in Z_THRESHOLDS:
                for d in dates:
                    jobs.append((card_id, card_def, tp, sl, z, d))

    results_by_key = defaultdict(list)
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_fill_sim, *j): j for j in jobs}
        for fut in as_completed(futures):
            card_id, card_def, tp, sl, z, d = futures[fut]
            try:
                r = fut.result()
                if r is not None:
                    r["date"] = d.isoformat()
                    results_by_key[(card_id, tp, sl, z)].append(r)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                log.warning("Failed " + card_id + " TP" + str(tp) + "/SL" + str(sl) + " z" + str(z) + " " + str(d) + ": " + str(e))
            done += 1
            if done % 500 == 0:
                pct = int(done / len(jobs) * 100)
                log.info("  " + str(done) + "/" + str(len(jobs)) + " (" + str(pct) + "%) " + str(errors) + " errors")

    log.info("Jobs done. " + str(errors) + " errors.")

    all_results = []
    for (card_id, tp, sl, z), rlist in results_by_key.items():
        if len(rlist) < 10:
            continue
        daily_pnls = [r.get("total_pnl_dollars", 0) for r in rlist]
        trades = sum(r.get("total_trades", 0) for r in rlist)
        if trades < 20:
            continue
        s = sortino_calc(daily_pnls)
        total_pnl = sum(daily_pnls)
        n_pos = sum(1 for p in daily_pnls if p > 0)
        all_results.append({
            "card": card_id, "tp": tp, "sl": sl, "z": z,
            "sortino": s, "total_pnl": total_pnl, "trades": trades,
            "n_days": len(rlist), "n_pos_days": n_pos,
            "pct_pos": n_pos / len(rlist),
            "avg_pnl_day": total_pnl / len(rlist),
        })

    all_results.sort(key=lambda x: x["sortino"], reverse=True)

    log.info("")
    log.info("=" * 90)
    log.info("TOP 20 RESULTS (sorted by Sortino)")
    log.info("Rank  Card  TP    SL    Z      Sortino     PnL    Trd   PctPos  Days")
    log.info("-" * 90)
    for i, r in enumerate(all_results[:20], 1):
        log.info("  " + str(i).ljust(3) + " " + r["card"].ljust(5) + " " +
                 str(r["tp"]).ljust(5) + " " + str(r["sl"]).ljust(5) + " " + str(r["z"]).ljust(6) +
                 " " + "{:.3f}".format(r["sortino"]).rjust(8) +
                 " $" + "{:>9,.0f}".format(r["total_pnl"]) +
                 " " + str(r["trades"]).rjust(6) +
                 " " + "{:.1%}".format(r["pct_pos"]).rjust(7) +
                 " " + str(r["n_days"]).rjust(4) + "d")

    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "n_configs": len(all_results),
        "n_positive": sum(1 for r in all_results if r["sortino"] > 0),
        "best": all_results[:20] if all_results else [],
        "best_sortino": all_results[0]["sortino"] if all_results else 0,
        "best_config": (r["card"] + "_tp" + str(r["tp"]) + "_sl" + str(r["sl"]) + "_z" + str(r["z"])) if all_results else "none",
    }
    r = all_results[0] if all_results else {}
    out_file = OUT_DIR / "targeted_sweep_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    log.info("")
    log.info("BEST: " + summary["best_config"] + " Sortino=" + "{:.3f}".format(summary["best_sortino"]))
    log.info("Positive configs: " + str(summary["n_positive"]) + "/" + str(summary["n_configs"]))
    log.info("Saved: " + str(out_file))
    log.info("[targeted_fillsim] DONE")


if __name__ == "__main__":
    main()
