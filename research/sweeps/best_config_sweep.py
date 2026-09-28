#!/usr/bin/env python3
"""Best Config Sweep - tests top trailing SL + TP combos on conv2.5_vol70 predictions.
14 parallel workers. Outputs to /home/jupiter/Lvl3Quant/data/processed/best_config_sweep/
"""
import subprocess, os, json, glob, itertools, time, sys
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Paths
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/best_config_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

# Config grid
TRAILING_STOPS = [15, 20, 25]
TAKE_PROFITS = [3, 5]
WAIT_BARS = [30, 50]
SIGNAL_THR = 0.1
LATENCY_MS = 50
HOLD_MS = 3600000  # 60min
CHASE_MAX_TICKS = 1
CHASE_MAX_REPRICES = 3

# Find matching dates (prediction + MBO both exist)
def find_dates():
    pred_files = glob.glob(os.path.join(PRED_DIR, "*_book_predstdExit_conv2.5_vol70.npz"))
    dates = []
    for pf in sorted(pred_files):
        basename = os.path.basename(pf)
        date_str = basename.split("_book_")[0]  # e.g. "2025-12-01"
        mbo_date = date_str.replace("-", "")
        mbo_file = os.path.join(MBO_DIR, f"glbx-mdp3-{mbo_date}.mbo.dbn.zst")
        if os.path.exists(mbo_file):
            dates.append((date_str, pf, mbo_file))
    return dates

def run_sim(args):
    date_str, pred_file, mbo_file, trail, tp, wb = args
    config_name = f"trail{trail}_tp{tp}_wb{wb}"
    out_file = os.path.join(OUT_DIR, f"{date_str}_{config_name}.json")

    # Skip if already done
    if os.path.exists(out_file):
        try:
            with open(out_file) as f:
                data = json.load(f)
            if "total_pnl_ticks" in data or "summary" in data:
                return {"date": date_str, "config": config_name, "status": "skip"}
        except:
            pass

    cmd = [
        FILL_SIM,
        "--mbo-file", mbo_file,
        "--predictions", pred_file,
        "--output", out_file,
        "--signal-threshold", str(SIGNAL_THR),
        "--hold-ms", str(HOLD_MS),
        "--trailing-ticks", str(trail),
        "--take-profit-ticks", str(tp),
        "--max-wait-bars", str(wb),
        "--latency-ms", str(LATENCY_MS),
        "--chase-entry",
        "--chase-max-ticks", str(CHASE_MAX_TICKS),
        "--chase-max-reprices", str(CHASE_MAX_REPRICES),
        "--quiet",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0 and os.path.exists(out_file):
            return {"date": date_str, "config": config_name, "status": "ok"}
        else:
            return {"date": date_str, "config": config_name, "status": "error",
                    "stderr": result.stderr[:200] if result.stderr else "no output"}
    except subprocess.TimeoutExpired:
        return {"date": date_str, "config": config_name, "status": "timeout"}
    except Exception as e:
        return {"date": date_str, "config": config_name, "status": "error", "stderr": str(e)[:200]}

def main():
    dates = find_dates()
    print(f"Found {len(dates)} dates with both predictions and MBO data")

    # Build task list
    tasks = []
    for date_str, pred_file, mbo_file in dates:
        for trail in TRAILING_STOPS:
            for tp in TAKE_PROFITS:
                for wb in WAIT_BARS:
                    tasks.append((date_str, pred_file, mbo_file, trail, tp, wb))

    print(f"Total tasks: {len(tasks)} ({len(dates)} dates x {len(TRAILING_STOPS)*len(TAKE_PROFITS)*len(WAIT_BARS)} configs)")
    print(f"Configs: trail={TRAILING_STOPS}, tp={TAKE_PROFITS}, wb={WAIT_BARS}")
    print(f"Fixed: sig={SIGNAL_THR}, lat={LATENCY_MS}ms, hold={HOLD_MS}ms, chase=1t/3r")
    print(f"Workers: 14")
    print(f"Output: {OUT_DIR}")
    print(f"Started: {datetime.now().isoformat()}")
    sys.stdout.flush()

    completed = 0
    errors = 0
    skipped = 0
    start_time = time.time()

    with Pool(14) as pool:
        for result in pool.imap_unordered(run_sim, tasks):
            if result["status"] == "skip":
                skipped += 1
            elif result["status"] == "ok":
                completed += 1
            else:
                errors += 1
                print(f"  ERROR: {result['date']} {result['config']}: {result.get('stderr', 'unknown')}")

            total_done = completed + skipped + errors
            if total_done % 50 == 0 or total_done == len(tasks):
                elapsed = time.time() - start_time
                rate = total_done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - total_done) / rate / 60 if rate > 0 else 0
                print(f"  Progress: {total_done}/{len(tasks)} ({completed} new, {skipped} skip, {errors} err) - {rate:.1f}/s - ETA {eta:.1f}min")
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min. New: {completed}, Skipped: {skipped}, Errors: {errors}")

    # Quick summary of results
    print("\n--- RESULTS SUMMARY ---")
    config_results = {}
    for f in glob.glob(os.path.join(OUT_DIR, "*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            basename = os.path.basename(f)
            parts = basename.replace(".json", "").split("_", 1)
            config_name = parts[1] if len(parts) > 1 else basename
            # Extract config from filename
            for trail in TRAILING_STOPS:
                for tp in TAKE_PROFITS:
                    for wb in WAIT_BARS:
                        cn = f"trail{trail}_tp{tp}_wb{wb}"
                        if cn in basename:
                            if cn not in config_results:
                                config_results[cn] = {"pnl": 0, "trades": 0, "days": 0}
                            pnl = data.get("total_pnl_ticks", 0)
                            trades = data.get("total_trades", 0)
                            config_results[cn]["pnl"] += pnl
                            config_results[cn]["trades"] += trades
                            config_results[cn]["days"] += 1
                            break
        except:
            continue

    for cn in sorted(config_results, key=lambda x: config_results[x]["pnl"], reverse=True):
        r = config_results[cn]
        avg_pnl = r["pnl"] / r["days"] if r["days"] > 0 else 0
        avg_trades = r["trades"] / r["days"] if r["days"] > 0 else 0
        pnl_dollars = r["pnl"] * 12.50
        print(f"  {cn}: {r['pnl']:.1f} ticks (${pnl_dollars:.0f}) | {r['trades']} trades over {r['days']} days | avg {avg_pnl:.1f}t/day {avg_trades:.1f}trades/day")

if __name__ == "__main__":
    main()
