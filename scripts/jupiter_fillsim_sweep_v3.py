#!/usr/bin/env python3
"""Fill sim sweep v3: higher thresholds (0.5, 0.7) to filter to top-confidence signals.

v2 finding: threshold=0.3 unprofitable across all configs (too many signals, commission drag).
v3 tests: threshold=0.5 (~top 5%) and 0.7 (~top 2%) with refined TP/SL configs.
Config additions: wider TP (G, H) since commission is the kill — need bigger wins.

Run: python3 script.py <config_name> <threshold>
e.g.: python3 script.py B_tp12_sl20 0.5
"""
import numpy as np, subprocess, json, os, glob, sys

FILL_SIM   = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR    = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR   = "/home/jupiter/Lvl3Quant/fill_sim_test/predictions"
SINGLE_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/single_preds"
BASE_OUT   = "/home/jupiter/Lvl3Quant/fill_sim_test"

SIGNAL_THRESHOLD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
os.makedirs(SINGLE_DIR, exist_ok=True)

# Load predictions
all_dates = {}
for npz_path in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
    d = np.load(npz_path)
    for key in d.keys():
        if key.endswith("_preds"):
            date8 = key.replace("_preds","").replace("-","")
            if date8 not in all_dates:
                all_dates[date8] = d[key]
print("Predictions: %d dates" % len(all_dates))

# Match MBO
matched = []
for mbo in sorted(glob.glob(os.path.join(MBO_DIR, "*.dbn.zst"))):
    date8 = os.path.basename(mbo).split("-")[2].split(".")[0]
    if date8 in all_dates:
        matched.append((mbo, date8))
print("Matched: %d dates, threshold=%.2f" % (len(matched), SIGNAL_THRESHOLD))

# Write single-date pred files
for mbo, date8 in matched:
    p = os.path.join(SINGLE_DIR, "%s.npz" % date8)
    if not os.path.exists(p):
        np.savez(p, predictions=all_dates[date8])

configs = {
    # v2 configs (keep for comparison at new thresholds)
    "A_tp13_sl40": "--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000",
    "B_tp12_sl20": "--take-profit-ticks 12 --stop-loss-ticks 20 --hold-ms 1800000",
    "D_tp16_sl20": "--take-profit-ticks 16 --stop-loss-ticks 20 --hold-ms 1800000",
    # New wider-TP configs (bigger reward needed to clear commission)
    "G_tp20_sl20": "--take-profit-ticks 20 --stop-loss-ticks 20 --hold-ms 1800000",
    "H_tp25_sl20": "--take-profit-ticks 25 --stop-loss-ticks 20 --hold-ms 1800000",
    "I_tp20_sl30": "--take-profit-ticks 20 --stop-loss-ticks 30 --hold-ms 1800000",
}

thr_tag = ("%.0f" % (SIGNAL_THRESHOLD*10)).replace(".","")
target_cfg = sys.argv[1] if len(sys.argv) > 1 else list(configs.keys())[0]
if target_cfg not in configs:
    print("Unknown config: %s. Options: %s" % (target_cfg, list(configs.keys())))
    sys.exit(1)

OUT_DIR = os.path.join(BASE_OUT, "results_v3_t%s" % thr_tag)
os.makedirs(OUT_DIR, exist_ok=True)

flags = configs[target_cfg]
print("\n=== %s (threshold=%.2f, dates=%d) ===" % (target_cfg, SIGNAL_THRESHOLD, len(matched)))

cr = []
for mbo, date8 in matched:
    pred_path = os.path.join(SINGLE_DIR, "%s.npz" % date8)
    out_path  = os.path.join(OUT_DIR, "%s_%s.json" % (target_cfg, date8))
    cmd = "%s --mbo-file %s --predictions %s --output %s --signal-threshold %.2f --prime-hours %s" % (
        FILL_SIM, mbo, pred_path, out_path, SIGNAL_THRESHOLD, flags)
    try:
        subprocess.run(cmd.split(), capture_output=True, text=True, timeout=180)
        if os.path.exists(out_path):
            d = json.load(open(out_path))
            pnl   = d.get("total_pnl_dollars", 0)
            trades = d.get("total_trades", 0)
            wr     = d.get("win_rate", 0)
            print("  %s: PnL=$%.0f trades=%d wr=%.1f%%" % (date8, pnl, trades, wr*100))
            cr.append(d)
    except Exception as e:
        print("  %s: ERROR %s" % (date8, e))

if cr:
    tp = sum(r.get("total_pnl_dollars",0) for r in cr)
    tt = sum(r.get("total_trades",0) for r in cr)
    wins = sum(1 for r in cr for t in r.get("trades",[]) if t.get("pnl_dollars",0)>0)
    trecs = sum(len(r.get("trades",[])) for r in cr)
    wr_ov = wins/trecs if trecs else 0
    avg   = tp/tt if tt else 0
    result = {"pnl": tp, "trades": tt, "dates": len(cr),
              "win_rate": wr_ov, "avg_per_trade": avg, "threshold": SIGNAL_THRESHOLD}
    print("\n  TOTAL %s@%.2f: PnL=$%.0f trades=%d wr=%.1f%% avg/trade=$%.2f" % (
        target_cfg, SIGNAL_THRESHOLD, tp, tt, wr_ov*100, avg))
    summary_path = os.path.join(OUT_DIR, "summary_%s.json" % target_cfg)
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Done! %s" % summary_path)
