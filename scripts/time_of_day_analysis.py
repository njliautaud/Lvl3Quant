#!/usr/bin/env python3
"""
Time-of-Day Analysis - C1 z>2.0 OOT Results
Splits existing fill_sim results by hour to find when edge is strongest.

Data source: z20 conviction sweep results (58 OOT days).
Each file has 'trades' array with signal_time_ns, pnl_dollars, queue_position.

Also runs targeted fill_sim with prime-hours filter to confirm.
"""
import json, os, glob, numpy as np, subprocess
from datetime import datetime
from collections import defaultdict

Z20_DIR = "/home/jupiter/Lvl3Quant/data/processed/conviction_threshold_sweep/z20"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/time_of_day_analysis"
FILL_SIM = "/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
PRED_SUFFIX = "book_predstdExit_conv1.5_vol50"
os.makedirs(OUT_DIR, exist_ok=True)

# Load z20 results
z20_files = sorted(glob.glob(os.path.join(Z20_DIR, "*.json")))
print("Found %d z20 result files" % len(z20_files))

if not z20_files:
    print("ERROR: No z20 files. Run conviction threshold sweep first.")
    import sys; sys.exit(1)

# Analyze trades by hour
hour_stats = defaultdict(list)
all_trades_with_hour = []

for fpath in z20_files:
    date_str = os.path.basename(fpath).replace(".json", "")
    try:
        with open(fpath) as f:
            data = json.load(f)
        for trade in data.get("trades", []):
            sig_ns = trade.get("signal_time_ns", 0)
            pnl = trade.get("pnl_dollars", 0)
            if sig_ns > 0:
                # Convert to ET hour (UTC-5 EST approximation)
                sig_sec = sig_ns / 1e9
                hour_utc = (sig_sec % 86400) / 3600
                hour_et = (hour_utc - 5) % 24
                hour_et_int = int(hour_et)
                hour_stats[hour_et_int].append(pnl)
                all_trades_with_hour.append({
                    "date": date_str,
                    "hour_et": hour_et,
                    "hour_et_int": hour_et_int,
                    "pnl": pnl,
                    "side": trade.get("side"),
                    "exit_reason": trade.get("exit_reason"),
                    "queue_position": trade.get("queue_position_at_post"),
                    "signal_strength": trade.get("signal_strength"),
                })
    except Exception as e:
        print("Error reading %s: %s" % (fpath, str(e)))

print("Total trades analyzed: %d" % len(all_trades_with_hour))

# Hour analysis - RTH only (9-16 ET)
print("\n" + "="*60)
print("TIME-OF-DAY ANALYSIS (C1, z>2.0, TP13/SL20 on Neptune)")
print("="*60)
print("%10s | %7s | %10s | %6s | %8s" % ("Hour (ET)", "Trades", "Total PnL", "WR%", "Avg PnL"))
print("-" * 55)

hour_summary = {}
for hour in sorted(hour_stats.keys()):
    if 8 <= hour <= 16:
        pnls = hour_stats[hour]
        n = len(pnls)
        total = sum(pnls)
        avg = total / n if n > 0 else 0
        wr = 100.0 * sum(1 for p in pnls if p > 0) / n if n > 0 else 0
        time_str = "%02d:00-%02d:00" % (hour, (hour+1) % 24)
        marker = " *EDGE" if avg > 0 else ""
        print("%10s | %7d | $%9.0f | %5.1f%% | $%7.2f%s" % (time_str, n, total, wr, avg, marker))
        hour_summary[hour] = {"n": n, "total_pnl": round(total, 2), "win_rate": round(wr, 1), "avg_pnl": round(avg, 2)}

# Best 2-hour windows
print("\n" + "="*60)
print("BEST 2-HOUR WINDOWS")
print("="*60)
best_windows = []
for start in range(9, 15):
    for end in range(start+1, 17):
        window_pnls = []
        for h in range(start, end):
            window_pnls.extend(hour_stats.get(h, []))
        if window_pnls:
            total = sum(window_pnls)
            wr = 100.0 * sum(1 for p in window_pnls if p > 0) / len(window_pnls)
            avg = total / len(window_pnls)
            mean_d = float(np.mean(window_pnls))
            down = float(np.std([min(p, 0) for p in window_pnls]))
            sortino_proxy = mean_d / down if down > 0 else 0
            best_windows.append({
                "window": "%02d:00-%02d:00" % (start, end),
                "n": len(window_pnls), "total": total, "wr": wr,
                "avg": avg, "sortino_proxy": sortino_proxy
            })

best_windows.sort(key=lambda x: -x["sortino_proxy"])
print("Top 5 time windows by risk-adjusted edge:")
for w in best_windows[:5]:
    print("  %s: n=%d pnl=$%.0f WR=%.1f%% avg=$%.2f sortino_proxy=%.3f" % (
        w["window"], w["n"], w["total"], w["wr"], w["avg"], w["sortino_proxy"]))

# Run fill_sim with prime-hours filter vs all-hours
print("\n" + "="*60)
print("FILL SIM: prime-hours vs all-hours (z>2.0, TP13, 1hr hold)")
print("="*60)

pred_files = {
    os.path.basename(f).split("_" + PRED_SUFFIX)[0]: f
    for f in glob.glob(os.path.join(PRED_DIR, "*_" + PRED_SUFFIX + ".npz"))
}
mbo_keyed = {}
for f in glob.glob(os.path.join(MBO_DIR, "*.dbn.zst")):
    raw = os.path.basename(f).replace("glbx-mdp3-", "").replace(".mbo.dbn.zst", "")
    date_str = raw[:4] + "-" + raw[4:6] + "-" + raw[6:8]
    mbo_keyed[date_str] = f

matched_dates = sorted(set(pred_files.keys()) & set(mbo_keyed.keys()))


def compute_sortino(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    mean = float(np.mean(daily_pnls))
    downside = float(np.std([min(p, 0.0) for p in daily_pnls]))
    return mean / downside if downside > 0 else 0.0


def run_sweep(label, extra_flags, z_thresh=2.0, tp=13.0, hold_ms=3600000):
    out_dir = os.path.join(OUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    day_results = []
    for date_str in matched_dates:
        out_file = os.path.join(out_dir, date_str + ".json")
        if os.path.exists(out_file):
            try:
                with open(out_file) as f:
                    day_results.append(json.load(f))
                continue
            except Exception:
                pass
        cmd = [
            FILL_SIM,
            "--mbo-file", mbo_keyed[date_str],
            "--predictions", pred_files[date_str],
            "--output", out_file,
            "--signal-threshold", str(z_thresh),
            "--take-profit-ticks", str(tp),
            "--hold-ms", str(hold_ms),
        ] + extra_flags
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if os.path.exists(out_file):
                with open(out_file) as f:
                    day_results.append(json.load(f))
        except Exception as e:
            pass
    if day_results:
        daily_pnls = [r.get("total_pnl_dollars", 0) for r in day_results]
        total = sum(daily_pnls)
        trades = sum(r.get("total_trades", 0) for r in day_results)
        n = len(day_results)
        sortino = compute_sortino(daily_pnls)
        all_t = []
        for r in day_results:
            all_t.extend(r.get("trades", []))
        wr = 100.0 * sum(1 for t in all_t if t.get("pnl_dollars", 0) > 0) / len(all_t) if all_t else 0
        print("\n  %s: Sortino=%.3f PnL=$%.0f trades=%d (%.1f/day) WR=%.1f%%" % (
            label, sortino, total, trades, float(trades)/n if n > 0 else 0, wr))
        return {"label": label, "sortino": round(sortino, 3), "pnl": round(total, 2),
                "trades": trades, "wr": round(wr, 1), "n_days": n}
    return None

sweep_results = []
r = run_sweep("all_hours_z20", [])
if r: sweep_results.append(r)
r = run_sweep("prime_hours_z20", ["--prime-hours"])
if r: sweep_results.append(r)
r = run_sweep("all_hours_z25", [], z_thresh=2.5)
if r: sweep_results.append(r)
r = run_sweep("prime_hours_z25", ["--prime-hours"], z_thresh=2.5)
if r: sweep_results.append(r)
# Test shorter TP during prime hours
r = run_sweep("prime_hours_z20_tp8", ["--prime-hours"], tp=8.0)
if r: sweep_results.append(r)
r = run_sweep("prime_hours_z20_tp20", ["--prime-hours"], tp=20.0)
if r: sweep_results.append(r)

# Save results
summary = {
    "timestamp": datetime.now().isoformat(),
    "hourly_breakdown": {str(k): v for k, v in hour_summary.items()},
    "best_windows": best_windows[:10],
    "fill_sim_sweep": sweep_results,
    "total_trades_analyzed": len(all_trades_with_hour),
}
out_file = os.path.join(OUT_DIR, "time_of_day_summary.json")
with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)

print("\n\n" + "="*60)
print("TIME-OF-DAY SUMMARY")
print("="*60)
for r in sweep_results:
    print("  %s: Sortino=%.3f PnL=$%.0f WR=%.1f%%" % (r["label"], r["sortino"], r["pnl"], r["wr"]))
print("\nSaved to: %s" % out_file)
print("DONE")
