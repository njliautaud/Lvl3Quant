#!/usr/bin/env python3
"""
TRACK 2: Queue Position Targeting Sweep
========================================
Tests min_queue_pos from 5 to 30 (step 5) on fill_sim.
Hypothesis: entering only when you can reach queue position <= N
reduces adverse selection and improves fill quality.

Queue position targeting means we submit a limit order only when the
current queue depth at our price level is <= min_queue_pos — so we
have a reasonable chance of being near front of queue when our price
gets hit.

Runs fill_sim_cli for each config across all OOT dates.
Saves per-config Sortino, WR, WLR, fill rates to JSON.
"""

import json
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("queue_pos_sweep")

BASE      = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
PRED_FILE = BASE / "alpha_discovery" / "deep_models" / "results" / "oot_wf_predictions_incremental.npz"
MBO_DIR   = BASE / "data" / "raw" / "mbo"
FILL_SIM  = BASE / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
OUT_DIR   = BASE / "data" / "processed" / "queue_pos_sweep_results"
OUT_FILE  = Path(__file__).parent / "queue_position_sweep_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_START  = date(2025, 12, 1)
OOT_END    = date(2026, 3, 6)
WORKERS    = 12
SIGNAL_THRESH = 0.05

# Queue position configs: min_queue_pos 5→30, step 5
# baseline (no queue filter) also included
QUEUE_POS_CONFIGS = [None, 5, 10, 15, 20, 25, 30]  # None = no filter

# Base fill params (from best known config at z>2.0)
BASE_TP_TICKS = 12
BASE_SL_TICKS = 30
BASE_HOLD_MS  = 300000  # 5 minutes max hold


def load_predictions():
    """Load OOT predictions keyed by date string."""
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Predictions not found: {PRED_FILE}")
    d = np.load(PRED_FILE, allow_pickle=True)
    if 'dates' in d and 'predictions' in d:
        dates_arr = d['dates']
        preds_arr = d['predictions']
        return {str(dates_arr[i]): preds_arr[i] for i in range(len(dates_arr))}
    # Fallback: keys ending in _preds
    result = {}
    for key in d.keys():
        if key.endswith('_preds'):
            dt = key.replace('_preds', '')
            result[dt] = d[key]
    log.info(f"Loaded predictions for {len(result)} dates")
    return result


def run_one(mbo_path, pred_npz_path, out_json, queue_pos, config_name):
    """Run fill_sim_cli for one date and one queue position config."""
    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_path),
        "--predictions", str(pred_npz_path),
        "--output", str(out_json),
        "--signal-threshold", str(SIGNAL_THRESH),
        "--take-profit-ticks", str(BASE_TP_TICKS),
        "--stop-loss-ticks", str(BASE_SL_TICKS),
        "--hold-ms", str(BASE_HOLD_MS),
        "--chase-entry",
        "--latency-ms", "5",
        "--prime-hours",
    ]
    if queue_pos is not None:
        cmd += ["--min-queue-pos", str(queue_pos)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if out_json.exists():
            with open(out_json) as f:
                return json.load(f)
        log.warning(f"No output for {config_name}: {r.stderr[:200]}")
        return None
    except subprocess.TimeoutExpired:
        log.error(f"Timeout: {config_name}")
        return None
    except Exception as e:
        log.error(f"Error {config_name}: {e}")
        return None


def compute_sortino(results_list):
    """Compute Sortino ratio from list of per-date result dicts."""
    daily_pnls = [r.get('total_pnl_dollars', 0) for r in results_list]
    if len(daily_pnls) < 5:
        return 0.0
    arr = np.array(daily_pnls, dtype=float)
    mean_ret = arr.mean()
    downside = arr[arr < 0]
    if len(downside) < 2:
        return float('inf') if mean_ret > 0 else 0.0
    downside_std = float(np.std(downside))
    if downside_std < 1e-6:
        return float('inf') if mean_ret > 0 else 0.0
    return float(mean_ret / downside_std * np.sqrt(252))


def main():
    log.info("Queue Position Targeting Sweep")
    log.info(f"Configs: {QUEUE_POS_CONFIGS}")
    log.info(f"Workers: {WORKERS}")

    preds_by_date = load_predictions()

    # Match MBO files to dates with predictions
    mbo_files = sorted(MBO_DIR.glob("*.mbo.dbn.zst"))
    matched = []
    for mbo in mbo_files:
        fname  = mbo.name  # e.g. glbx-mdp3-20251201.mbo.dbn.zst
        parts  = fname.split("-")
        if len(parts) < 3:
            continue
        date8  = parts[2].split(".")[0]  # YYYYMMDD
        date_s = f"{date8[:4]}-{date8[4:6]}-{date8[6:]}"
        d      = date(int(date8[:4]), int(date8[4:6]), int(date8[6:]))
        if OOT_START <= d <= OOT_END and date_s in preds_by_date:
            matched.append((mbo, date_s, preds_by_date[date_s]))
    log.info(f"Matched {len(matched)} dates")
    if not matched:
        log.error("No matched dates! Check prediction file format.")
        return

    all_results = {}

    for qp in QUEUE_POS_CONFIGS:
        cfg_name = "baseline_no_filter" if qp is None else f"queue_pos_lte_{qp}"
        log.info(f"\n--- Config: {cfg_name} ---")

        tasks = []
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {}
            for mbo, date_s, preds in matched:
                date8     = date_s.replace("-", "")
                pred_npz  = OUT_DIR / f"pred_{date8}.npz"
                out_json  = OUT_DIR / f"{cfg_name}_{date8}.json"

                # Save predictions for this date
                if not pred_npz.exists():
                    np.savez(str(pred_npz), predictions=preds)

                fut = pool.submit(run_one, mbo, pred_npz, out_json, qp, f"{cfg_name}_{date8}")
                futures[fut] = date_s

            date_results = []
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    date_results.append(r)

        if date_results:
            total_pnl    = sum(r.get('total_pnl_dollars', 0) for r in date_results)
            total_trades = sum(r.get('total_trades', 0) for r in date_results)
            total_filled = sum(r.get('total_filled', 0) for r in date_results)
            sortino      = compute_sortino(date_results)
            fill_rate    = total_filled / max(total_trades, 1)
            wins         = sum(1 for r in date_results if r.get('win_rate', 0) > 0.5)

            all_results[cfg_name] = {
                'queue_pos_filter': qp,
                'total_pnl_dollars': total_pnl,
                'total_trades':      total_trades,
                'total_filled':      total_filled,
                'fill_rate':         fill_rate,
                'sortino':           sortino,
                'n_dates':           len(date_results),
            }
            log.info(f"  PnL=${total_pnl:.0f}  trades={total_trades}  fills={total_filled}  "
                     f"fill_rate={fill_rate:.1%}  Sortino={sortino:.2f}  dates={len(date_results)}")
        else:
            all_results[cfg_name] = {'error': 'no_results'}
            log.warning(f"  No results for {cfg_name}")

    # Print summary table
    print("\n" + "="*70)
    print("QUEUE POSITION TARGETING SWEEP — RESULTS")
    print(f"{'Config':<30} {'PnL':>8} {'Trades':>8} {'Fill%':>7} {'Sortino':>9}")
    print("-"*70)
    for cfg, r in all_results.items():
        if 'error' not in r:
            print(f"{cfg:<30} ${r['total_pnl_dollars']:>7.0f} {r['total_trades']:>8d} "
                  f"{r['fill_rate']:>6.1%} {r['sortino']:>9.2f}")
    print("="*70)

    # Save results
    with open(OUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results saved: {OUT_FILE}")


if __name__ == '__main__':
    main()
