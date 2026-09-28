#!/usr/bin/env python3
"""Fill sim TP/SL sweep on Jupiter — tests 6 exit configs on real MBO data."""
import numpy as np, subprocess, json, os, glob

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/results"
SINGLE_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/single_preds"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SINGLE_DIR, exist_ok=True)

all_dates = {}
for npz_path in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
    d = np.load(npz_path)
    for key in d.keys():
        if key.endswith("_preds"):
            date_dash = key.replace("_preds", "")
            date8 = date_dash.replace("-", "")
            all_dates[date8] = d[key]
print("Predictions for %d dates" % len(all_dates))
print("Sample dates: %s" % list(all_dates.keys())[:5])

mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, "*.dbn.zst")))
matched = []
for mbo in mbo_files:
    fname = os.path.basename(mbo)
    date8 = fname.split("-")[2].split(".")[0]
    if date8 in all_dates:
        matched.append((mbo, date8))
print("Matched %d dates" % len(matched))

for mbo, date8 in matched:
    pred = all_dates[date8]
    np.savez(os.path.join(SINGLE_DIR, "%s.npz" % date8), predictions=pred)

configs = {
    "A_tp13_sl40": "--take-profit-ticks 13 --stop-loss-ticks 40 --hold-ms 1800000",
    "B_tp12_sl20": "--take-profit-ticks 12 --stop-loss-ticks 20 --hold-ms 1800000",
    "C_tp10_sl15": "--take-profit-ticks 10 --stop-loss-ticks 15 --hold-ms 1200000",
    "D_tp16_sl20": "--take-profit-ticks 16 --stop-loss-ticks 20 --hold-ms 1800000",
    "E_trail8_tp12": "--take-profit-ticks 12 --trailing-ticks 8 --hold-ms 1800000",
    "F_trail5_sl20": "--take-profit-ticks 12 --trailing-ticks 5 --stop-loss-ticks 20 --hold-ms 1200000",
}

results = {}
for cname, flags in configs.items():
    print("")
    print("=== %s ===" % cname)
    cr = []
    for mbo, date8 in matched[:10]:
        pred_path = os.path.join(SINGLE_DIR, "%s.npz" % date8)
        out_path = os.path.join(OUT_DIR, "%s_%s.json" % (cname, date8))
        cmd = "%s --mbo-file %s --predictions %s --output %s --signal-threshold 1.0 --prime-hours %s" % (
            FILL_SIM, mbo, pred_path, out_path, flags)
        try:
            subprocess.run(cmd.split(), capture_output=True, text=True, timeout=120)
            if os.path.exists(out_path):
                with open(out_path) as f:
                    data = json.load(f)
                pnl = data.get("total_pnl_dollars", 0)
                trades = data.get("total_trades", 0)
                fills = data.get("total_filled", 0)
                print("  %s: PnL=$%.0f trades=%d fills=%d" % (date8, pnl, trades, fills))
                cr.append(data)
            else:
                print("  %s: no output" % date8)
        except Exception as e:
            print("  %s: ERROR %s" % (date8, e))

    if cr:
        tp = sum(r.get("total_pnl_dollars", 0) for r in cr)
        tt = sum(r.get("total_trades", 0) for r in cr)
        tf = sum(r.get("total_filled", 0) for r in cr)
        results[cname] = {"pnl": tp, "trades": tt, "fills": tf, "dates": len(cr)}
        print("  TOTAL: PnL=$%.0f trades=%d fills=%d dates=%d" % (tp, tt, tf, len(cr)))

print("")
print("=== GRAND COMPARISON ===")
for n, r in sorted(results.items(), key=lambda x: x[1]["pnl"], reverse=True):
    avg = r["pnl"] / r["dates"] if r["dates"] else 0
    print("  %-25s PnL=$%8.0f  trades=%4d  fills=%4d  avg/day=$%6.0f" % (
        n, r["pnl"], r["trades"], r["fills"], avg))

with open(os.path.join(OUT_DIR, "sweep_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print("Done! Results in %s" % OUT_DIR)
