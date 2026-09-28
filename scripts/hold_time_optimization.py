#!/usr/bin/env python3
"""
Hold Time Optimization - C1 (book_predstdExit_conv1.5_vol50)
z>2.0, TP13, limit entry.
Tests: hold_ms = [10000, 30000, 60000, 300000, 600000, 1800000, 3600000]

Key question: Does the edge decay or strengthen at shorter holds?
Current C1 (Neptune confirmed): hold_ms=3600000 (1 hour), Sortino=2.046.

Jupiter fill_sim_cli flags: --take-profit-ticks, --trailing-ticks, --hold-ms
No --stop-loss-ticks (not supported in Jupiter version).
"""
import subprocess, json, os, glob, numpy as np
from datetime import datetime

FILL_SIM = "/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/hold_time_optimization"
os.makedirs(OUT_DIR, exist_ok=True)

PRED_SUFFIX = "book_predstdExit_conv1.5_vol50"

# Build file indexes
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
print("C1 hold time sweep: %d OOT days" % len(matched_dates))
print("Config: z>2.0, TP13, limit entry (no stop-loss in Jupiter CLI)")

# Hold times to test (ms)
HOLD_CONFIGS = [
    (10000,   "10s"),
    (30000,   "30s"),
    (60000,   "1min"),
    (300000,  "5min"),
    (600000,  "10min"),
    (1800000, "30min"),
    (3600000, "60min_baseline"),
]


def compute_sortino(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    mean = float(np.mean(daily_pnls))
    downside = float(np.std([min(p, 0.0) for p in daily_pnls]))
    return mean / downside if downside > 0 else 0.0


all_results = []

for hold_ms, hold_label in HOLD_CONFIGS:
    out_dir = os.path.join(OUT_DIR, hold_label)
    os.makedirs(out_dir, exist_ok=True)

    print("\n=== hold=%s (%dms) ===" % (hold_label, hold_ms))
    day_results = []

    for date_str in matched_dates:
        pred_file = pred_files[date_str]
        mbo_file = mbo_keyed[date_str]
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
            "--mbo-file", mbo_file,
            "--predictions", pred_file,
            "--output", out_file,
            "--signal-threshold", "2.0",
            "--take-profit-ticks", "13.0",
            "--hold-ms", str(hold_ms),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if os.path.exists(out_file):
                with open(out_file) as f:
                    data = json.load(f)
                day_results.append(data)
            elif result.stderr:
                print("  %s: ERR %s" % (date_str, result.stderr[:80]))
        except subprocess.TimeoutExpired:
            print("  %s: TIMEOUT" % date_str)
        except Exception as e:
            print("  %s: ERROR %s" % (date_str, str(e)))

    if day_results:
        daily_pnls = [r.get("total_pnl_dollars", 0) for r in day_results]
        total_pnl = sum(daily_pnls)
        total_trades = sum(r.get("total_trades", 0) for r in day_results)
        n_days = len(day_results)
        sortino = compute_sortino(daily_pnls)
        trades_per_day = float(total_trades) / n_days if n_days > 0 else 0
        pos_days = sum(1 for p in daily_pnls if p > 0)

        all_t = []
        for r in day_results:
            all_t.extend(r.get("trades", []))
        wr = 100.0 * sum(1 for t in all_t if t.get("pnl_dollars", 0) > 0) / len(all_t) if all_t else 0
        avg_hold_sec = float(np.mean([t.get("hold_duration_ns", 0) / 1e9 for t in all_t])) if all_t else 0

        exit_reasons = {}
        for t in all_t:
            r = t.get("exit_reason", "Unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        print("  days=%d trades=%d (%.1f/day) pnl=$%.0f sortino=%.3f WR=%.1f%% pos=%d/%d" % (
            n_days, total_trades, trades_per_day, total_pnl, sortino, wr, pos_days, n_days))
        print("  avg_actual_hold=%.0fs exits=%s" % (avg_hold_sec, exit_reasons))

        all_results.append({
            "hold_ms": hold_ms,
            "hold_label": hold_label,
            "n_days": n_days,
            "total_trades": total_trades,
            "trades_per_day": round(trades_per_day, 1),
            "total_pnl": round(total_pnl, 2),
            "sortino": round(sortino, 3),
            "win_rate": round(wr, 1),
            "pos_days": pos_days,
            "avg_actual_hold_sec": round(avg_hold_sec, 1),
            "exit_reasons": exit_reasons,
        })
    else:
        print("  No results for hold=%s" % hold_label)

# Save summary
summary_file = os.path.join(OUT_DIR, "hold_time_summary.json")
summary = {
    "timestamp": datetime.now().isoformat(),
    "card": "C1",
    "pred_suffix": PRED_SUFFIX,
    "config": {"z_threshold": 2.0, "tp_ticks": 13.0, "note": "no stop-loss in Jupiter CLI"},
    "results": all_results,
}
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print("\n\n" + "="*60)
print("HOLD TIME OPTIMIZATION - FINAL RESULTS")
print("="*60)
print("%15s | %8s | %10s | %8s | %6s | %8s" % ("Hold", "Sortino", "Trades/Day", "PnL", "WR%", "AvgHold"))
print("-" * 70)
for r in all_results:
    best_sortino = max(x["sortino"] for x in all_results) if all_results else 0
    marker = " <<BEST" if r["sortino"] == best_sortino else ""
    print("%15s | %8.3f | %10.1f | $%7.0f | %5.1f%% | %6.0fs%s" % (
        r["hold_label"], r["sortino"], r["trades_per_day"],
        r["total_pnl"], r["win_rate"], r["avg_actual_hold_sec"], marker))

print("\nSaved to: %s" % summary_file)
print("DONE")
