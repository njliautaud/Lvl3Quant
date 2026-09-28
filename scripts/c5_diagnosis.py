#!/usr/bin/env python3
"""
C5 Diagnosis - Compare paper trade PnL vs fill_sim predictions.

C5 uses variant: book_predstdExit_conv2.0_vol70
Paper shows +$720 across 108 trades (50% WR, avg win=$38, avg loss=$-24.72)
Fill sim tested TP13/SL20 configs - wrong for C5 (no stop-loss in Jupiter CLI).

This script runs fill_sim with configs matching C5 paper behavior:
- Small TP (1-4 ticks), 120s hold timeout
- Also tests longer holds and higher conviction thresholds
- Jupiter CLI: only --take-profit-ticks, --trailing-ticks, --hold-ms available
"""
import subprocess, json, os, glob, numpy as np
from datetime import datetime

FILL_SIM = "/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/c5_diagnosis"
os.makedirs(OUT_DIR, exist_ok=True)

PRED_SUFFIX = "book_predstdExit_conv2.0_vol70"

# Build prediction file index
pred_files = {
    os.path.basename(f).split("_" + PRED_SUFFIX)[0]: f
    for f in glob.glob(os.path.join(PRED_DIR, "*_" + PRED_SUFFIX + ".npz"))
}
# Build MBO file index
mbo_keyed = {}
for f in glob.glob(os.path.join(MBO_DIR, "*.dbn.zst")):
    raw = os.path.basename(f).replace("glbx-mdp3-", "").replace(".mbo.dbn.zst", "")
    date_str = raw[:4] + "-" + raw[4:6] + "-" + raw[6:8]
    mbo_keyed[date_str] = f

matched_dates = sorted(set(pred_files.keys()) & set(mbo_keyed.keys()))
print("C5 Diagnosis | pred_suffix=%s" % PRED_SUFFIX)
print("Matched dates: %d" % len(matched_dates))

# Configs: (label, tp_ticks, trailing_ticks, hold_ms, z_thresh)
# Note: no stop-loss in Jupiter CLI - use trailing stop or hold timeout
configs = [
    # Paper-like: tiny TP, 120s hold (no stop loss = rely on hold timeout)
    ("c5_tp1_hold120s_z05", 1.0, None, 120000, 0.5),
    ("c5_tp2_hold120s_z05", 2.0, None, 120000, 0.5),
    ("c5_tp3_hold120s_z05", 3.0, None, 120000, 0.5),
    # Higher conviction filter
    ("c5_tp3_hold120s_z20", 3.0, None, 120000, 2.0),
    ("c5_tp5_hold300s_z20", 5.0, None, 300000, 2.0),
    ("c5_tp8_hold600s_z20", 8.0, None, 600000, 2.0),
    # Trailing stop instead of fixed TP
    ("c5_trail3_hold120s_z05", None, 3.0, 120000, 0.5),
    ("c5_trail5_hold300s_z20", None, 5.0, 300000, 2.0),
    # Same as C1 optimal for comparison
    ("c5_tp13_hold3600s_z20", 13.0, None, 3600000, 2.0),
    # Prime hours filter with z>2.0
    ("c5_tp5_hold300s_z20_prime", 5.0, None, 300000, 2.0),
]

def compute_sortino(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    mean = float(np.mean(daily_pnls))
    downside = float(np.std([min(p, 0.0) for p in daily_pnls]))
    return mean / downside if downside > 0 else 0.0

all_results = {}

for cfg in configs:
    label, tp, trail, hold_ms, z_thresh = cfg
    use_prime = label.endswith("_prime")
    print("\n=== %s ===" % label)

    config_out_dir = os.path.join(OUT_DIR, label)
    os.makedirs(config_out_dir, exist_ok=True)
    day_results = []

    for date_str in matched_dates:
        pred_file = pred_files[date_str]
        mbo_file = mbo_keyed[date_str]
        out_file = os.path.join(config_out_dir, date_str + ".json")

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
            "--signal-threshold", str(z_thresh),
            "--hold-ms", str(hold_ms),
        ]
        if tp is not None:
            cmd += ["--take-profit-ticks", str(tp)]
        if trail is not None:
            cmd += ["--trailing-ticks", str(trail)]
        if use_prime:
            cmd += ["--prime-hours"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if os.path.exists(out_file):
                with open(out_file) as f:
                    data = json.load(f)
                day_results.append(data)
                trades = data.get("total_trades", 0)
                pnl = data.get("total_pnl_dollars", 0)
            else:
                print("  %s: no output | stderr: %s" % (date_str, result.stderr[:100]))
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

        all_trades_flat = []
        for r in day_results:
            all_trades_flat.extend(r.get("trades", []))
        wr = 100.0 * sum(1 for t in all_trades_flat if t.get("pnl_dollars", 0) > 0) / len(all_trades_flat) if all_trades_flat else 0

        print("  SUMMARY: days=%d trades=%d (%.1f/day) pnl=$%.0f sortino=%.3f WR=%.1f%% pos=%d/%d" % (
            n_days, total_trades, trades_per_day, total_pnl, sortino, wr, pos_days, n_days))
        all_results[label] = {
            "label": label, "tp_ticks": tp, "trailing_ticks": trail,
            "hold_ms": hold_ms, "z_threshold": z_thresh, "prime_hours": use_prime,
            "n_days": n_days, "total_trades": total_trades,
            "trades_per_day": round(trades_per_day, 1),
            "total_pnl": round(total_pnl, 2), "sortino": round(sortino, 3),
            "win_rate": round(wr, 1), "pos_days": pos_days,
        }
    else:
        print("  No results for %s" % label)

# Save summary
summary_file = os.path.join(OUT_DIR, "c5_diagnosis_summary.json")
summary = {
    "timestamp": datetime.now().isoformat(),
    "pred_suffix": PRED_SUFFIX,
    "paper_baseline": {
        "total_trades": 108, "total_pnl": 720.73, "win_rate": 50.0,
        "avg_win": 38.07, "avg_loss": -24.72,
        "exit_breakdown": {"HOLD_TIMEOUT": 67, "TAKE_PROFIT": 41},
        "hold_ms": 120000, "note": "Live paper engine, no stop-loss, hold_timeout=120s"
    },
    "fill_sim_results": all_results,
    "note": "Jupiter fill_sim_cli has no --stop-loss-ticks. Only TP, trailing, hold-ms."
}
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("C5 DIAGNOSIS SUMMARY")
print("="*60)
print("Paper baseline: 108 trades, $720.73 PnL, 50% WR, 120s hold timeout")
print("\nFill sim results (sorted by Sortino):")
for label, r in sorted(all_results.items(), key=lambda x: -x[1]["sortino"]):
    print("  %s: Sortino=%.3f PnL=$%.0f trades=%d (%.1f/day)" % (
        label, r["sortino"], r["total_pnl"], r["total_trades"], r["trades_per_day"]))

print("\nResults saved to: %s" % summary_file)
print("DONE")
