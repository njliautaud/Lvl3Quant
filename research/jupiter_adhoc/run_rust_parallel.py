#!/usr/bin/env python3
"""
Parallel Rust fill_sim_cli runner for Jupiter.
Runs all 27 days x 4 configs x 3 thresholds in parallel (max 8 workers).

Run on Jupiter: python3 /home/jupiter/run_rust_parallel.py
"""
import subprocess
import os
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BINARY = "/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/lvl3quant/data/mbo")
CONFIG_DIR = Path("/home/jupiter/lvl3quant/production/configs")
THRESH_BASE = Path("/home/jupiter/lvl3quant/data/processed/rust_predictions_thresh")
OUT_BASE = Path("/home/jupiter/lvl3quant/production/results/rust_sim")
LOG_DIR = OUT_BASE / "logs"

# Parameter configs
CONFIGS = ["h15000_t8_s8", "h15000_t6_s8", "h20000_t8_s7", "h30000_t4_s7"]

# Thresholds (matching Saturn sweep)
THRESHOLDS = ["0.7", "0.5", "0.9"]

# All 27 MBO dates
DATES = [
    "2025-07-14", "2025-07-15", "2025-07-16", "2025-07-17", "2025-07-18",
    "2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23", "2025-07-24",
    "2025-07-25", "2025-07-27", "2025-07-28", "2025-07-29", "2025-07-30",
    "2025-07-31", "2025-08-01", "2025-08-03", "2025-08-04", "2025-08-05",
    "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-10", "2025-08-11",
    "2025-08-12", "2025-08-13",
]

OUT_BASE.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_one(args):
    """Run one fill_sim_cli job. Returns (job_id, success, summary)."""
    cfg, thresh, date_str = args
    thresh_dir = f"thresh_{thresh}"
    date_raw = date_str.replace("-", "")

    mbo_file = MBO_DIR / f"glbx-mdp3-{date_raw}.mbo.dbn"
    pred_file = THRESH_BASE / thresh_dir / f"{date_str}_predictions.npz"
    cfg_file = CONFIG_DIR / f"{cfg}.json"
    out_dir = OUT_BASE / f"{cfg}_t{thresh.replace('.', '')}"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{date_str}.json"
    log_file = LOG_DIR / f"{cfg}_t{thresh.replace('.', '')}_{date_str}.log"

    job_id = f"{cfg}_t{thresh}_{date_str}"

    # Skip if already done
    if out_file.exists():
        try:
            with open(out_file) as f:
                d = json.load(f)
            return (job_id, True, f"SKIP (already done, PnL=${d['summary']['total_pnl_dollars']:.1f})")
        except:
            pass

    # Check files exist
    if not mbo_file.exists():
        return (job_id, False, f"SKIP (no MBO file: {mbo_file.name})")
    if not pred_file.exists():
        return (job_id, False, f"SKIP (no pred file)")

    start_time = time.time()
    cmd = [
        BINARY,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--config", str(cfg_file),
        "--quiet",
    ]

    with open(log_file, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=300)

    elapsed = time.time() - start_time

    if result.returncode != 0:
        return (job_id, False, f"FAILED (exit={result.returncode}, {elapsed:.0f}s)")

    # Read summary
    try:
        with open(out_file) as f:
            d = json.load(f)
        s = d["summary"]
        summary_str = (
            f"PnL=${s['total_pnl_dollars']:.1f}, "
            f"trades={s['total_trades']}, "
            f"fill={s['fill_rate']*100:.0f}%, "
            f"wr={s['win_rate']*100:.0f}%, "
            f"sharpe={s['sharpe_per_trade']:.3f}, "
            f"elapsed={elapsed:.0f}s"
        )
        return (job_id, True, summary_str)
    except Exception as e:
        return (job_id, False, f"PARSE ERROR: {e}")


def main():
    # Build job list
    jobs = []
    for cfg in CONFIGS:
        for thresh in THRESHOLDS:
            for date_str in DATES:
                jobs.append((cfg, thresh, date_str))

    total = len(jobs)
    print(f"Total jobs: {total}")
    print(f"Configs: {CONFIGS}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Dates: {len(DATES)}")
    print()

    # Run with max 8 parallel workers
    MAX_WORKERS = 8
    completed = 0
    failed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}

        for future in as_completed(futures):
            job_id, success, msg = future.result()
            completed += 1
            if "SKIP" in msg:
                skipped += 1
            elif not success:
                failed += 1
                print(f"[{completed}/{total}] FAIL  {job_id}: {msg}", flush=True)
            else:
                print(f"[{completed}/{total}] OK    {job_id}: {msg}", flush=True)

    print()
    print(f"Done. {completed} total, {failed} failed, {skipped} skipped")
    print()

    # Aggregate results per config+threshold
    print("=" * 60)
    print("AGGREGATE RESULTS BY CONFIG + THRESHOLD")
    print("=" * 60)

    for cfg in CONFIGS:
        for thresh in THRESHOLDS:
            out_dir = OUT_BASE / f"{cfg}_t{thresh.replace('.', '')}"
            json_files = sorted(out_dir.glob("*.json"))

            if not json_files:
                print(f"\n{cfg} thresh={thresh}: NO RESULTS")
                continue

            total_pnl = 0.0
            total_trades = 0
            total_filled = 0
            total_posted = 0
            days_ok = 0

            for jf in json_files:
                try:
                    with open(jf) as f:
                        d = json.load(f)
                    s = d["summary"]
                    total_pnl += s["total_pnl_dollars"]
                    total_trades += s["total_trades"]
                    total_filled += s["total_filled"]
                    total_posted += s["total_posted"]
                    days_ok += 1
                except:
                    pass

            avg_pnl = total_pnl / days_ok if days_ok else 0
            fill_rate = total_filled / total_posted if total_posted else 0

            print(f"\n{cfg} thresh={thresh}: {days_ok}/{len(DATES)} days")
            print(f"  Total PnL:    ${total_pnl:+,.2f}")
            print(f"  Avg PnL/day:  ${avg_pnl:+,.2f}")
            print(f"  Total trades: {total_trades}")
            print(f"  Fill rate:    {fill_rate*100:.1f}%")


if __name__ == "__main__":
    main()
