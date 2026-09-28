#!/usr/bin/env python3
"""
TRACK 2: Signal Strength × TP/SL Interaction Sweep
====================================================
Different TP/SL parameters per z-score bin.

Hypothesis: weak signals (z=1.0-1.5) need tight TP/SL to be profitable,
while strong signals (z>2.5) can hold wider TP for larger captures.

Z-score bins:
  - weak:   1.0 ≤ |z| < 1.5
  - medium: 1.5 ≤ |z| < 2.0
  - strong: 2.0 ≤ |z| < 2.5
  - ultra:  |z| ≥ 2.5

TP/SL grid per bin (3×3 = 9 configs per bin = 36 configs total):
  TP: 6, 10, 14 ticks
  SL: 15, 25, 40 ticks

Each config runs across all OOT dates. Results show optimal TP/SL per z-bin.
Decision: combine best TP/SL per bin into a dynamic TP/SL strategy.
"""

import json
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sig_tpsl_sweep")

BASE      = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
PRED_FILE = BASE / "alpha_discovery" / "deep_models" / "results" / "oot_wf_predictions_incremental.npz"
MBO_DIR   = BASE / "data" / "raw" / "mbo"
FILL_SIM  = BASE / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
OUT_DIR   = BASE / "data" / "processed" / "sig_tpsl_sweep_results"
OUT_FILE  = Path(__file__).parent / "signal_strength_tpsl_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_START = date(2025, 12, 1)
OOT_END   = date(2026, 3, 6)
WORKERS   = 12
HOLD_MS   = 300000  # 5 min max

# Z-score bins → signal threshold for fill_sim_cli
Z_BINS = {
    'weak_z1p0_1p5':   (1.0, 1.5),
    'medium_z1p5_2p0': (1.5, 2.0),
    'strong_z2p0_2p5': (2.0, 2.5),
    'ultra_z2p5plus':  (2.5, 10.0),
}

TP_OPTIONS = [6, 10, 14]
SL_OPTIONS = [15, 25, 40]


def load_predictions():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Missing: {PRED_FILE}")
    d = np.load(PRED_FILE, allow_pickle=True)
    if 'dates' in d and 'predictions' in d:
        return {str(d['dates'][i]): d['predictions'][i] for i in range(len(d['dates']))}
    return {k.replace('_preds', ''): d[k] for k in d.keys() if k.endswith('_preds')}


def run_one(mbo_path, pred_npz, out_json, z_min, z_max, tp, sl, cfg_name):
    """
    Run fill_sim_cli with z-score band filtering.
    --signal-threshold z_min: only trade signals with |z| >= z_min
    --signal-threshold-max z_max: don't trade signals with |z| >= z_max (if supported)
    Falls back to single threshold if max not supported.
    """
    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_path),
        "--predictions", str(pred_npz),
        "--output", str(out_json),
        "--signal-threshold", str(z_min),
        "--take-profit-ticks", str(tp),
        "--stop-loss-ticks", str(sl),
        "--hold-ms", str(HOLD_MS),
        "--chase-entry",
        "--latency-ms", "5",
        "--prime-hours",
    ]
    # Try to add upper bound threshold if supported
    if z_max < 9.0:
        cmd += ["--signal-threshold-max", str(z_max)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if out_json.exists():
            with open(out_json) as f:
                return json.load(f)
        return None
    except Exception as e:
        log.debug(f"{cfg_name}: {e}")
        return None


def compute_sortino(results_list):
    daily_pnls = np.array([r.get('total_pnl_dollars', 0) for r in results_list], dtype=float)
    if len(daily_pnls) < 5:
        return 0.0
    mean_ret = daily_pnls.mean()
    downside = daily_pnls[daily_pnls < 0]
    if len(downside) < 2:
        return float('inf') if mean_ret > 0 else 0.0
    dstd = float(np.std(downside))
    return float(mean_ret / dstd * np.sqrt(252)) if dstd > 1e-6 else (float('inf') if mean_ret > 0 else 0.0)


def main():
    log.info("Signal Strength × TP/SL Interaction Sweep")
    log.info(f"Z-bins: {list(Z_BINS.keys())}")
    log.info(f"TP grid: {TP_OPTIONS}, SL grid: {SL_OPTIONS}")
    log.info(f"Total configs: {len(Z_BINS) * len(TP_OPTIONS) * len(SL_OPTIONS)}")

    preds_by_date = load_predictions()

    mbo_files = sorted(MBO_DIR.glob("*.mbo.dbn.zst"))
    matched = []
    for mbo in mbo_files:
        parts = mbo.name.split("-")
        if len(parts) < 3:
            continue
        date8  = parts[2].split(".")[0]
        date_s = f"{date8[:4]}-{date8[4:6]}-{date8[6:]}"
        d      = date(int(date8[:4]), int(date8[4:6]), int(date8[6:]))
        if OOT_START <= d <= OOT_END and date_s in preds_by_date:
            matched.append((mbo, date_s, preds_by_date[date_s]))
    log.info(f"Matched {len(matched)} dates")

    all_results = {}
    best_per_bin = {}

    for bin_name, (z_min, z_max) in Z_BINS.items():
        log.info(f"\n{'='*50}")
        log.info(f"Z-bin: {bin_name} ({z_min} ≤ |z| < {z_max})")
        bin_results = {}

        for tp, sl in product(TP_OPTIONS, SL_OPTIONS):
            cfg_name = f"{bin_name}_tp{tp}_sl{sl}"

            with ProcessPoolExecutor(max_workers=WORKERS) as pool:
                futures = {}
                for mbo, date_s, preds in matched:
                    date8    = date_s.replace("-", "")
                    pred_npz = OUT_DIR / f"pred_{date8}.npz"
                    out_json = OUT_DIR / f"{cfg_name}_{date8}.json"
                    if not pred_npz.exists():
                        np.savez(str(pred_npz), predictions=preds)
                    fut = pool.submit(run_one, mbo, pred_npz, out_json,
                                      z_min, z_max, tp, sl, f"{cfg_name}_{date8}")
                    futures[fut] = date_s

                date_results = [fut.result() for fut in as_completed(futures)
                                if fut.result() is not None]

            if date_results:
                total_pnl    = sum(r.get('total_pnl_dollars', 0) for r in date_results)
                total_trades = sum(r.get('total_trades', 0) for r in date_results)
                total_filled = sum(r.get('total_filled', 0) for r in date_results)
                sortino      = compute_sortino(date_results)
                fill_rate    = total_filled / max(total_trades, 1)
                pnl_per_fill = total_pnl / max(total_filled, 1)

                bin_results[cfg_name] = {
                    'z_bin': bin_name,
                    'z_min': z_min, 'z_max': z_max,
                    'tp_ticks': tp, 'sl_ticks': sl,
                    'total_pnl': total_pnl,
                    'total_trades': total_trades,
                    'total_filled': total_filled,
                    'fill_rate': fill_rate,
                    'sortino': sortino,
                    'pnl_per_fill': pnl_per_fill,
                    'n_dates': len(date_results),
                }
                log.info(f"  TP={tp} SL={sl}: PnL=${total_pnl:.0f}  "
                         f"fills={total_filled}  fill%={fill_rate:.1%}  Sortino={sortino:.2f}")

        all_results[bin_name] = bin_results

        # Find best config per bin (by Sortino)
        if bin_results:
            best_cfg = max(bin_results, key=lambda k: bin_results[k].get('sortino', 0))
            best     = bin_results[best_cfg]
            best_per_bin[bin_name] = {
                'best_config': best_cfg,
                'tp_ticks': best['tp_ticks'],
                'sl_ticks': best['sl_ticks'],
                'sortino': best['sortino'],
                'pnl_per_fill': best['pnl_per_fill'],
            }
            log.info(f"  BEST for {bin_name}: TP={best['tp_ticks']} SL={best['sl_ticks']} "
                     f"Sortino={best['sortino']:.2f}")

    # Summary
    print("\n" + "="*70)
    print("SIGNAL STRENGTH × TP/SL — BEST CONFIG PER Z-BIN")
    print(f"{'Z-Bin':<25} {'TP':>5} {'SL':>5} {'Sortino':>9} {'PnL/Fill':>10}")
    print("-"*70)
    for bn, b in best_per_bin.items():
        print(f"{bn:<25} {b['tp_ticks']:>5} {b['sl_ticks']:>5} "
              f"{b['sortino']:>9.2f} ${b['pnl_per_fill']:>9.2f}")
    print("="*70)
    print("\nNext step: combine into dynamic TP/SL strategy (see Track 2 queue)")

    with open(OUT_FILE, 'w') as f:
        json.dump({'by_bin': all_results, 'best_per_bin': best_per_bin}, f, indent=2)
    log.info(f"Results: {OUT_FILE}")

    # Save best-per-bin as a standalone config for deployment
    deploy_config = {bin_name: {'tp': v['tp_ticks'], 'sl': v['sl_ticks']}
                     for bin_name, v in best_per_bin.items()}
    deploy_path = Path(__file__).parent / "dynamic_tpsl_config.json"
    with open(deploy_path, 'w') as f:
        json.dump(deploy_config, f, indent=2)
    log.info(f"Dynamic TP/SL config: {deploy_path}")


if __name__ == '__main__':
    main()
