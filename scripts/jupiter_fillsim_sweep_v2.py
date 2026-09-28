#!/usr/bin/env python3
"""Fill sim TP/SL sweep on Jupiter — v2: threshold=0.3, all dates, parallel by config.

Changes from v1:
- signal_threshold: 1.0 → 0.3 (v1 had only 0.5% of bars signaling; 0.3 = top ~13%)
- matched[:10] → all matched dates (v1 only ran first 10 dates)
- 6 parallel workers, one per config (launch with: python3 script.py <config_name>)
"""
import numpy as np, subprocess, json, os, glob, sys

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR  = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/predictions"
OUT_DIR  = "/home/jupiter/Lvl3Quant/fill_sim_test/results_v2"
SINGLE_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/single_preds"
SIGNAL_THRESHOLD = 0.3  # |pred| > 0.3 → top ~13% confidence (0.5% at thresh=1.0 was too low)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SINGLE_DIR, exist_ok=True)

# Load all predictions
all_dates = {}
for npz_path in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
    d = np.load(npz_path)
    for key in d.keys():
        if key.endswith("_preds"):
            date_dash = key.replace("_preds", "")
            date8 = date_dash.replace("-", "")
            if date8 not in all_dates:  # first file wins (most recent run)
                all_dates[date8] = d[key]
print("Predictions for %d dates" % len(all_dates))

# Match to MBO files
mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, "*.dbn.zst")))
matched = []
for mbo in mbo_files:
    fname = os.path.basename(mbo)
    date8 = fname.split("-")[2].split(".")[0]
    if date8 in all_dates:
        matched.append((mbo, date8))
print("Matched %d dates with MBO files" % len(matched))

# Write single-date prediction files
for mbo, date8 in matched:
    pred_path = os.path.join(SINGLE_DIR, "%s.npz" % date8)
    if not os.path.exists(pred_path):
        np.savez(pred_path, predictions=all_dates[date8])

configs = {
    "A_tp13_sl40": "--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000",
    "B_tp12_sl20": "--take-profit-ticks 12 --stop-loss-ticks 20 --hold-ms 1800000",
    "C_tp10_sl15": "--take-profit-ticks 10 --stop-loss-ticks 15 --hold-ms 1200000",
    "D_tp16_sl20": "--take-profit-ticks 16 --stop-loss-ticks 20 --hold-ms 1800000",
    "E_trail8_tp12": "--take-profit-ticks 12 --trailing-ticks 8 --hold-ms 1800000",
    "F_trail5_sl20": "--take-profit-ticks 12 --trailing-ticks 5 --stop-loss-ticks 20 --hold-ms 1200000",
}

# If run with a config arg, only process that config
target_configs = sys.argv[1:] if len(sys.argv) > 1 else list(configs.keys())
print("Running configs: %s" % target_configs)

results = {}
for cname in target_configs:
    if cname not in configs:
        print("Unknown config: %s" % cname)
        continue
    flags = configs[cname]
    print("\n=== %s (threshold=%.2f, dates=%d) ===" % (cname, SIGNAL_THRESHOLD, len(matched)))
    cr = []
    for mbo, date8 in matched:
        pred_path = os.path.join(SINGLE_DIR, "%s.npz" % date8)
        out_path  = os.path.join(OUT_DIR, "%s_%s.json" % (cname, date8))
        cmd = "%s --mbo-file %s --predictions %s --output %s --signal-threshold %.2f --prime-hours %s" % (
            FILL_SIM, mbo, pred_path, out_path, SIGNAL_THRESHOLD, flags)
        try:
            subprocess.run(cmd.split(), capture_output=True, text=True, timeout=180)
            if os.path.exists(out_path):
                with open(out_path) as f:
                    data = json.load(f)
                pnl    = data.get("total_pnl_dollars", 0)
                trades = data.get("total_trades", 0)
                fills  = data.get("total_filled", 0)
                wr     = data.get("win_rate", 0)
                print("  %s: PnL=$%.0f trades=%d fills=%d wr=%.1f%%" % (
                    date8, pnl, trades, fills, wr*100))
                cr.append(data)
            else:
                print("  %s: no output" % date8)
        except Exception as e:
            print("  %s: ERROR %s" % (date8, e))

    if cr:
        tp  = sum(r.get("total_pnl_dollars", 0) for r in cr)
        tt  = sum(r.get("total_trades", 0) for r in cr)
        tf  = sum(r.get("total_filled", 0) for r in cr)
        wins = sum(1 for r in cr for t in r.get("trades", []) if t.get("pnl_dollars", 0) > 0)
        total_trades = sum(len(r.get("trades", [])) for r in cr)
        wr_overall = wins / total_trades if total_trades else 0
        avg_pnl = tp / tt if tt else 0
        results[cname] = {
            "pnl": tp, "trades": tt, "fills": tf, "dates": len(cr),
            "win_rate": wr_overall, "avg_per_trade": avg_pnl,
            "signal_threshold": SIGNAL_THRESHOLD
        }
        print("  TOTAL %s: PnL=$%.0f trades=%d wr=%.1f%% avg/trade=$%.1f" % (
            cname, tp, tt, wr_overall*100, avg_pnl))

# Write per-config summary
summary_path = os.path.join(OUT_DIR, "summary_%s.json" % "_".join(target_configs))
with open(summary_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nDone! Results: %s" % summary_path)
