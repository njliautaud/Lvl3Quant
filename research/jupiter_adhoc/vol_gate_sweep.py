#!/usr/bin/env python3
"""Vol gate comparison: Cards 3 & 4 across all 68 OOT days on Jupiter."""

import subprocess
import json
import os
import sys
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
OUT_DIR = "/tmp/vol_gate_sweep"

DATES = [
    "2025-12-01","2025-12-02","2025-12-03","2025-12-04","2025-12-05",
    "2025-12-08","2025-12-09","2025-12-10","2025-12-11","2025-12-12",
    "2025-12-15","2025-12-16","2025-12-17","2025-12-18","2025-12-19",
    "2025-12-22","2025-12-23","2025-12-24","2025-12-26","2025-12-29",
    "2025-12-30","2025-12-31",
    "2026-01-02","2026-01-05","2026-01-06","2026-01-07","2026-01-08",
    "2026-01-09","2026-01-12","2026-01-13","2026-01-14","2026-01-15",
    "2026-01-16","2026-01-19","2026-01-20","2026-01-21","2026-01-22",
    "2026-01-23","2026-01-26","2026-01-27","2026-01-28","2026-01-29",
    "2026-01-30",
    "2026-02-02","2026-02-03","2026-02-04","2026-02-05","2026-02-06",
    "2026-02-09","2026-02-10","2026-02-11","2026-02-12","2026-02-13",
    "2026-02-16","2026-02-17","2026-02-18","2026-02-19","2026-02-20",
    "2026-02-23","2026-02-24","2026-02-25","2026-02-26","2026-02-27",
    "2026-03-02","2026-03-03","2026-03-04","2026-03-05","2026-03-06",
]

CONFIGS = {
    "Card3_vol0": {
        "pred_pattern": "raw_smoothExit_conv0.05_ethr0.0_vol0",
        "tp": 15,
    },
    "Card3_vol50": {
        "pred_pattern": "raw_smoothExit_conv0.05_ethr0.0_vol50",
        "tp": 15,
    },
    "Card3_vol70": {
        "pred_pattern": "raw_smoothExit_conv0.05_ethr0.0_vol70",
        "tp": 15,
    },
    "Card4_vol0": {
        "pred_pattern": "book_predstdExit_conv2.0_vol0",
        "tp": 20,
    },
    "Card4_vol50": {
        "pred_pattern": "book_predstdExit_conv2.0_vol50",
        "tp": 20,
    },
    "Card4_vol70": {
        "pred_pattern": "book_predstdExit_conv2.0_vol70",
        "tp": 20,
    },
}

def run_one(args):
    config_name, date, pred_pattern, tp = args
    date_nodash = date.replace("-", "")
    mbo_file = f"{MBO_DIR}/glbx-mdp3-{date_nodash}.mbo.dbn.zst"
    pred_file = f"{PRED_DIR}/{date}_{pred_pattern}.npz"
    out_file = f"{OUT_DIR}/{config_name}_{date}.json"

    if not os.path.exists(pred_file):
        return config_name, date, None, f"pred missing: {pred_file}"
    if not os.path.exists(mbo_file):
        return config_name, date, None, f"mbo missing: {mbo_file}"

    cmd = [
        FILL_SIM,
        "--mbo-file", mbo_file,
        "--predictions", pred_file,
        "--signal-threshold", "0.5",
        "--hold-ms", "3600000",
        "--take-profit-ticks", str(tp),
        "--quiet",
        "--output", out_file,
    ]

    try:
        subprocess.run(cmd, capture_output=True, timeout=300)
        with open(out_file) as f:
            result = json.load(f)
        return config_name, date, result, None
    except Exception as e:
        return config_name, date, None, str(e)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build job list
    jobs = []
    for config_name, cfg in CONFIGS.items():
        for date in DATES:
            jobs.append((config_name, date, cfg["pred_pattern"], cfg["tp"]))

    print(f"Running {len(jobs)} jobs with 14 workers...")
    sys.stdout.flush()

    results = {}
    errors = []
    done = 0

    with ProcessPoolExecutor(max_workers=14) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            config_name, date, result, err = fut.result()
            done += 1
            if err:
                errors.append(f"{config_name} {date}: {err}")
            else:
                if config_name not in results:
                    results[config_name] = []
                results[config_name].append(result)
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} complete")
                sys.stdout.flush()

    print(f"\nAll {done} jobs complete. Errors: {len(errors)}")
    if errors:
        print("\nFirst 10 errors:")
        for e in errors[:10]:
            print(f"  {e}")

    # Aggregate
    print("\n" + "="*100)
    print(f"{'Config':<16} {'Days':>5} {'Trades':>7} {'TotalPnL':>10} {'PnL/Trade':>10} {'WinRate':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("="*100)

    summary = {}
    for config_name in ["Card3_vol0", "Card3_vol50", "Card3_vol70",
                         "Card4_vol0", "Card4_vol50", "Card4_vol70"]:
        day_results = results.get(config_name, [])
        if not day_results:
            print(f"{config_name:<16} NO DATA")
            continue

        total_pnl = 0
        total_trades = 0
        total_wins = 0
        daily_pnls = []
        all_fill_rates = []

        for r in day_results:
            pnl = r.get("total_pnl_dollars", 0)
            trades = r.get("total_trades", 0)
            wr = r.get("win_rate", 0)  # fraction 0-1
            wins = round(wr * trades)
            fr = r.get("fill_rate", 0)
            total_pnl += pnl
            total_trades += trades
            total_wins += wins
            daily_pnls.append(pnl)
            if trades > 0:
                all_fill_rates.append(fr)

        wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Daily Sharpe
        if len(daily_pnls) > 1:
            mean_d = sum(daily_pnls) / len(daily_pnls)
            var_d = sum((x - mean_d)**2 for x in daily_pnls) / (len(daily_pnls) - 1)
            std_d = math.sqrt(var_d) if var_d > 0 else 1e-9
            sharpe = (mean_d / std_d) * math.sqrt(252)
        else:
            sharpe = 0

        # Max drawdown
        cum = 0
        peak = 0
        max_dd = 0
        for dp in daily_pnls:
            cum += dp
            if cum > peak:
                peak = cum
            dd = peak - cum
            if dd > max_dd:
                max_dd = dd

        print(f"{config_name:<16} {len(day_results):>5} {total_trades:>7} {total_pnl:>10.2f} {avg_pnl:>10.2f} {wr:>7.1f}% {sharpe:>8.2f} {max_dd:>8.2f}")

        summary[config_name] = {
            "days": len(day_results),
            "trades": total_trades,
            "total_pnl": round(total_pnl, 2),
            "pnl_per_trade": round(avg_pnl, 2),
            "win_rate": round(wr, 1),
            "sharpe": round(sharpe, 2),
            "max_dd": round(max_dd, 2),
        }

    print("="*100)

    # Card-level comparison
    for card in ["Card3", "Card4"]:
        print(f"\n--- {card} Vol Gate Comparison ---")
        variants = [f"{card}_vol0", f"{card}_vol50", f"{card}_vol70"]
        for v in variants:
            s = summary.get(v)
            if s:
                print(f"  {v}: PnL={s['total_pnl']:>10.2f}  Sharpe={s['sharpe']:>6.2f}  Trades={s['trades']:>5}  WR={s['win_rate']:>5.1f}%  PnL/trade={s['pnl_per_trade']:>7.2f}  MaxDD={s['max_dd']:>8.2f}")

    # Save JSON
    with open(f"{OUT_DIR}/vol_gate_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {OUT_DIR}/vol_gate_summary.json")


if __name__ == "__main__":
    main()
