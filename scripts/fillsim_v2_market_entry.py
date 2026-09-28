#!/usr/bin/env python3
"""Fill sim v2 — market entry to test true edge without fill rate noise.
Tests fold 01/02 strong signals (Jul-Aug 2025) with multiple thresholds.
Separates LONGS and SHORTS. Uses --market-entry for fill certainty."""
import numpy as np, subprocess, json, os, glob

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/results_v2"
SINGLE_DIR = "/home/jupiter/Lvl3Quant/fill_sim_test/single_preds_v2"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SINGLE_DIR, exist_ok=True)

# Load predictions per date
all_dates = {}
for npz_path in sorted(glob.glob(os.path.join(PRED_DIR, "*.npz"))):
    d = np.load(npz_path)
    for key in d.keys():
        if key.endswith("_preds"):
            date_dash = key.replace("_preds", "")
            date8 = date_dash.replace("-", "")
            all_dates[date8] = d[key]
print(f"Predictions for {len(all_dates)} dates")

# Match to MBO files — fold 01/02 = Jul-Oct 2025
mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, "*.dbn.zst")))
matched = []
for mbo in mbo_files:
    fname = os.path.basename(mbo)
    date8 = fname.split("-")[2].split(".")[0]
    if date8 in all_dates:
        # Focus on fold 01 (Jul-Aug 2025) first
        if "20250701" <= date8 <= "20251001":
            matched.append((mbo, date8))
print(f"Fold01-02 dates matched: {len(matched)}")

# Direction-split predictions: write long-only and short-only npz files
for mbo, date8 in matched:
    pred = all_dates[date8].copy()
    # Long-only: keep positives, zero out negatives
    long_pred = np.where(pred > 0, pred, 0.0)
    # Short-only: keep negatives (as-is, fill_sim reads sign for direction)
    short_pred = np.where(pred < 0, pred, 0.0)
    np.savez(os.path.join(SINGLE_DIR, f"{date8}_long.npz"), predictions=long_pred)
    np.savez(os.path.join(SINGLE_DIR, f"{date8}_short.npz"), predictions=short_pred)
    np.savez(os.path.join(SINGLE_DIR, f"{date8}_all.npz"), predictions=pred)

# Configs: market entry + various thresholds
# threshold=0 means ALL signals; threshold=0.3 = top ~20%
configs = {
    "MKT_t0_tp12_sl20_longs":   ("_long.npz",  0.0,  "--market-entry --take-profit-ticks 12 --stop-loss-ticks 20 --hold-ms 1800000 --prime-hours"),
    "MKT_t03_tp12_sl20_longs":  ("_long.npz",  0.3,  "--market-entry --take-profit-ticks 12 --stop-loss-ticks 20 --hold-ms 1800000 --prime-hours"),
    "MKT_t0_tp12_sl20_shorts":  ("_short.npz", 0.3,  "--market-entry --take-profit-ticks 12 --stop-loss-ticks 20 --hold-ms 1800000 --prime-hours"),
    "MKT_t0_tp10_sl15_all":     ("_all.npz",   0.3,  "--market-entry --take-profit-ticks 10 --stop-loss-ticks 15 --hold-ms 1200000 --prime-hours"),
    "LMT_t03_tp12_sl20_longs":  ("_long.npz",  0.3,  "--take-profit-ticks 12 --stop-loss-ticks 20 --hold-ms 1800000 --prime-hours"),
    "LMT_t1_tp12_sl20_all":     ("_all.npz",   1.0,  "--take-profit-ticks 12 --stop-loss-ticks 20 --hold-ms 1800000 --prime-hours"),
}

results = {}
for cname, (suffix, threshold, flags) in configs.items():
    print(f"\n=== {cname} (threshold={threshold}) ===")
    cr = []
    for mbo, date8 in matched[:20]:  # 20 dates for speed
        pred_path = os.path.join(SINGLE_DIR, f"{date8}{suffix}")
        if not os.path.exists(pred_path):
            continue
        out_path = os.path.join(OUT_DIR, f"{cname}_{date8}.json")
        cmd = f"{FILL_SIM} --mbo-file {mbo} --predictions {pred_path} --output {out_path} --signal-threshold {threshold} {flags}"
        try:
            subprocess.run(cmd.split(), capture_output=True, text=True, timeout=120)
            if os.path.exists(out_path):
                with open(out_path) as f:
                    data = json.load(f)
                pnl = data.get("total_pnl_dollars", 0)
                trades = data.get("total_trades", 0)
                fills = data.get("total_filled", 0)
                fill_rate = fills/max(trades,1)*100
                if trades > 0:
                    print(f"  {date8}: PnL=${pnl:.0f} trades={trades} fills={fills} rate={fill_rate:.0f}%")
                cr.append(data)
        except Exception as e:
            print(f"  {date8}: ERROR {e}")

    if cr:
        tp = sum(r.get("total_pnl_dollars", 0) for r in cr)
        tt = sum(r.get("total_trades", 0) for r in cr)
        tf = sum(r.get("total_filled", 0) for r in cr)
        fr = tf/max(tt,1)*100
        results[cname] = {"pnl": tp, "trades": tt, "fills": tf, "fill_rate": fr, "dates": len(cr)}
        print(f"  TOTAL: PnL=${tp:.0f} trades={tt} fills={tf} fillrate={fr:.1f}%")

print("\n=== GRAND COMPARISON ===")
for n, r in sorted(results.items(), key=lambda x: x[1]["pnl"], reverse=True):
    print(f"  {n:<35} PnL=${r['pnl']:>8.0f}  fills={r['fills']:>5}  rate={r['fill_rate']:>5.1f}%  dates={r['dates']}")

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDone! Results in {OUT_DIR}")
