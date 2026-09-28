#!/usr/bin/env python3
"""
TRACK 2: Time-of-Day Fill Sim Segmentation
===========================================
Runs separate fill sims for each market session:
  - Open:    09:30 - 11:00 ET (90 min, high volatility)
  - Midday:  11:00 - 13:00 ET (120 min, lower vol, wider spreads)
  - Afternoon: 13:00 - 15:00 ET (120 min, trend continuation)
  - Close:   15:00 - 16:00 ET (60 min, high vol, closing imbalances)

Hypothesis from adverse_selection analysis: toxic fills cluster in the open
and close. If midday or afternoon has superior Sortino, we should gate by time.

Output: per-session Sortino, WR, fill rate, PnL — sorted by Sortino descending.
Decision rule: deploy only sessions where Sortino > 1.5 after fills.
"""

import json
import logging
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tod_fillsim")

BASE      = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
PRED_FILE = BASE / "alpha_discovery" / "deep_models" / "results" / "oot_wf_predictions_incremental.npz"
MBO_DIR   = BASE / "data" / "raw" / "mbo"
FILL_SIM  = BASE / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
OUT_DIR   = BASE / "data" / "processed" / "tod_fillsim_results"
OUT_FILE  = Path(__file__).parent / "time_of_day_fillsim_results.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_START  = date(2025, 12, 1)
OOT_END    = date(2026, 3, 6)
WORKERS    = 12
SIGNAL_THRESH = 0.05

# ET session windows as (start_hour_ET*100 + min, end_hour_ET*100 + min)
# fill_sim_cli uses --time-window-start / --time-window-end in HHMM format
SESSIONS = {
    "open_0930_1100":       ("0930", "1100"),
    "midday_1100_1300":     ("1100", "1300"),
    "afternoon_1300_1500":  ("1300", "1500"),
    "close_1500_1600":      ("1500", "1600"),
    "full_day_baseline":    (None, None),       # no time filter
}

BASE_TP_TICKS = 12
BASE_SL_TICKS = 30
BASE_HOLD_MS  = 300000


def load_predictions():
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Missing: {PRED_FILE}")
    d = np.load(PRED_FILE, allow_pickle=True)
    if 'dates' in d and 'predictions' in d:
        return {str(d['dates'][i]): d['predictions'][i] for i in range(len(d['dates']))}
    return {k.replace('_preds', ''): d[k] for k in d.keys() if k.endswith('_preds')}


def run_one(mbo_path, pred_npz, out_json, session_start, session_end, cfg_name):
    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_path),
        "--predictions", str(pred_npz),
        "--output", str(out_json),
        "--signal-threshold", str(SIGNAL_THRESH),
        "--take-profit-ticks", str(BASE_TP_TICKS),
        "--stop-loss-ticks", str(BASE_SL_TICKS),
        "--hold-ms", str(BASE_HOLD_MS),
        "--chase-entry",
        "--latency-ms", "5",
        "--prime-hours",
    ]
    if session_start is not None:
        cmd += ["--time-window-start", session_start, "--time-window-end", session_end]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if out_json.exists():
            with open(out_json) as f:
                return json.load(f)
        return None
    except Exception as e:
        log.error(f"{cfg_name}: {e}")
        return None


def compute_sortino(results_list):
    daily_pnls = np.array([r.get('total_pnl_dollars', 0) for r in results_list], dtype=float)
    if len(daily_pnls) < 5:
        return 0.0
    mean_ret  = daily_pnls.mean()
    downside  = daily_pnls[daily_pnls < 0]
    if len(downside) < 2:
        return float('inf') if mean_ret > 0 else 0.0
    dstd = float(np.std(downside))
    return float(mean_ret / dstd * np.sqrt(252)) if dstd > 1e-6 else (float('inf') if mean_ret > 0 else 0.0)


def main():
    log.info("Time-of-Day Fill Sim Segmentation")
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

    for session_name, (t_start, t_end) in SESSIONS.items():
        log.info(f"\n--- Session: {session_name} ({t_start or 'all'}-{t_end or 'day'}) ---")

        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {}
            for mbo, date_s, preds in matched:
                date8     = date_s.replace("-", "")
                pred_npz  = OUT_DIR / f"pred_{date8}.npz"
                out_json  = OUT_DIR / f"{session_name}_{date8}.json"
                if not pred_npz.exists():
                    np.savez(str(pred_npz), predictions=preds)
                fut = pool.submit(run_one, mbo, pred_npz, out_json,
                                  t_start, t_end, f"{session_name}_{date8}")
                futures[fut] = date_s

            date_results = [fut.result() for fut in as_completed(futures)
                            if fut.result() is not None]

        if date_results:
            total_pnl    = sum(r.get('total_pnl_dollars', 0) for r in date_results)
            total_trades = sum(r.get('total_trades', 0) for r in date_results)
            total_filled = sum(r.get('total_filled', 0) for r in date_results)
            sortino      = compute_sortino(date_results)
            fill_rate    = total_filled / max(total_trades, 1)

            all_results[session_name] = {
                'session': session_name,
                'time_window': f"{t_start or 'all'}-{t_end or 'day'}",
                'total_pnl_dollars': total_pnl,
                'total_trades':      total_trades,
                'total_filled':      total_filled,
                'fill_rate':         fill_rate,
                'sortino':           sortino,
                'n_dates':           len(date_results),
                'pnl_per_trade':     total_pnl / max(total_filled, 1),
            }
            log.info(f"  PnL=${total_pnl:.0f}  trades={total_trades}  "
                     f"fill%={fill_rate:.1%}  Sortino={sortino:.2f}")
        else:
            all_results[session_name] = {'error': 'no_results'}

    # Print sorted summary
    print("\n" + "="*80)
    print("TIME-OF-DAY FILL SIM RESULTS — sorted by Sortino")
    print(f"{'Session':<30} {'PnL':>8} {'Trades':>8} {'Fill%':>7} {'Sortino':>9} {'PnL/Fill':>10}")
    print("-"*80)
    sorted_results = sorted(
        [(k, v) for k, v in all_results.items() if 'error' not in v],
        key=lambda x: x[1].get('sortino', 0), reverse=True
    )
    for cfg, r in sorted_results:
        print(f"{cfg:<30} ${r['total_pnl_dollars']:>7.0f} {r['total_trades']:>8d} "
              f"{r['fill_rate']:>6.1%} {r['sortino']:>9.2f} ${r['pnl_per_trade']:>9.2f}")
    print("="*80)
    print(f"\nDecision rule: Sessions with Sortino > 1.5 → candidate for time-gating")

    with open(OUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results: {OUT_FILE}")


if __name__ == '__main__':
    main()
