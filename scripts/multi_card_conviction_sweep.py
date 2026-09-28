#!/usr/bin/env python3
"""
Multi-Card Conviction Threshold Sweep - C4, C5, C7
Tests z=[0.1, 1.0, 2.0, 2.5, 3.0] for each card.

C1 already confirmed: z>2.0 is optimal (Sortino 2.046, limit entry).
Now testing whether same threshold works for C4, C5, C7.

Card -> prediction suffix mapping (from paper engine logs):
  C4: book_predstdExit_conv2.0_vol70 (uses TP20 in paper)
  C5: book_predstdExit_conv2.0_vol70 (same model, small TP in paper)
  C7: book_predstdExit_conv2.0_vol70 (same model, TP13-ish in paper)

Jupiter fill_sim_cli supports: --take-profit-ticks, --trailing-ticks, --hold-ms
No --stop-loss-ticks available.
"""
import subprocess, json, os, glob, numpy as np
from datetime import datetime

FILL_SIM = "/home/jupiter/lvl3quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/multi_card_conviction_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

# Card configurations - testing multiple TP configs per card
# Note: C4/C5/C7 all use the same underlying prediction model
CARD_CONFIGS = {
    "C4_tp20_hold1200s": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "tp": 20.0, "trail": None, "hold_ms": 1200000,
        "desc": "C4 paper-like: TP20, 20min hold",
    },
    "C5_tp3_hold120s": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "tp": 3.0, "trail": None, "hold_ms": 120000,
        "desc": "C5 paper-like: TP3, 120s hold",
    },
    "C5_tp13_hold3600s": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "tp": 13.0, "trail": None, "hold_ms": 3600000,
        "desc": "C5 with C1-optimal: TP13, 1hr hold",
    },
    "C7_tp13_hold3600s": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "tp": 13.0, "trail": None, "hold_ms": 3600000,
        "desc": "C7 with C1-optimal: TP13, 1hr hold",
    },
    "C7_tp8_hold600s": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "tp": 8.0, "trail": None, "hold_ms": 600000,
        "desc": "C7 medium: TP8, 10min hold",
    },
}

Z_THRESHOLDS = [0.1, 1.0, 2.0, 2.5, 3.0]


def compute_sortino(daily_pnls):
    if len(daily_pnls) < 2:
        return 0.0
    mean = float(np.mean(daily_pnls))
    downside = float(np.std([min(p, 0.0) for p in daily_pnls]))
    return mean / downside if downside > 0 else 0.0


def build_pred_index(pred_suffix):
    return {
        os.path.basename(f).split("_" + pred_suffix)[0]: f
        for f in glob.glob(os.path.join(PRED_DIR, "*_" + pred_suffix + ".npz"))
    }


def build_mbo_index():
    idx = {}
    for f in glob.glob(os.path.join(MBO_DIR, "*.dbn.zst")):
        raw = os.path.basename(f).replace("glbx-mdp3-", "").replace(".mbo.dbn.zst", "")
        date_str = raw[:4] + "-" + raw[4:6] + "-" + raw[6:8]
        idx[date_str] = f
    return idx


mbo_keyed = build_mbo_index()
all_summary = {}

for card_id, card_cfg in CARD_CONFIGS.items():
    pred_suffix = card_cfg["pred_suffix"]
    tp = card_cfg["tp"]
    trail = card_cfg.get("trail")
    hold_ms = card_cfg["hold_ms"]

    pred_files = build_pred_index(pred_suffix)
    matched_dates = sorted(set(pred_files.keys()) & set(mbo_keyed.keys()))

    print("\n" + "="*60)
    print("Card: %s | %s" % (card_id, card_cfg["desc"]))
    print("Matched dates: %d" % len(matched_dates))

    card_results = {}
    card_out_dir = os.path.join(OUT_DIR, card_id)
    os.makedirs(card_out_dir, exist_ok=True)

    for z_thresh in Z_THRESHOLDS:
        z_label = "z" + str(z_thresh).replace(".", "")
        z_out_dir = os.path.join(card_out_dir, z_label)
        os.makedirs(z_out_dir, exist_ok=True)

        day_results = []
        print("\n  z>%.1f:" % z_thresh)

        for date_str in matched_dates:
            pred_file = pred_files[date_str]
            mbo_file = mbo_keyed[date_str]
            out_file = os.path.join(z_out_dir, date_str + ".json")

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

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if os.path.exists(out_file):
                    with open(out_file) as f:
                        data = json.load(f)
                    day_results.append(data)
                elif result.stderr:
                    print("    %s: ERR %s" % (date_str, result.stderr[:80]))
            except subprocess.TimeoutExpired:
                print("    %s: TIMEOUT" % date_str)
            except Exception as e:
                print("    %s: ERROR %s" % (date_str, str(e)))

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

            print("    days=%d trades=%d (%.1f/day) pnl=$%.0f sortino=%.3f WR=%.1f%% pos=%d/%d" % (
                n_days, total_trades, trades_per_day, total_pnl, sortino, wr, pos_days, n_days))

            card_results[z_label] = {
                "z_threshold": z_thresh,
                "n_days": n_days, "total_trades": total_trades,
                "trades_per_day": round(trades_per_day, 1),
                "total_pnl": round(total_pnl, 2), "sortino": round(sortino, 3),
                "win_rate": round(wr, 1), "pos_days": pos_days,
            }
        else:
            print("    No results")

    all_summary[card_id] = {"config": card_cfg, "results": card_results}

# Save full summary
summary_file = os.path.join(OUT_DIR, "multi_card_conviction_summary.json")
summary = {
    "timestamp": datetime.now().isoformat(),
    "z_thresholds": Z_THRESHOLDS,
    "cards": all_summary,
    "c1_baseline": {"z_threshold": 2.0, "sortino": 2.046, "note": "Already tested on Neptune"},
}
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print("\n\n" + "="*60)
print("MULTI-CARD CONVICTION SWEEP - FINAL RESULTS")
print("="*60)
for card_id, card_data in all_summary.items():
    if not card_data["results"]:
        print("\n%s: NO RESULTS" % card_id)
        continue
    best_z = max(card_data["results"].items(), key=lambda x: x[1]["sortino"])
    print("\n%s (%s):" % (card_id, card_data["config"]["desc"]))
    for z_label, r in sorted(card_data["results"].items(), key=lambda x: x[1]["z_threshold"]):
        star = " <<BEST" if z_label == best_z[0] else ""
        print("  z>%.1f: Sortino=%.3f PnL=$%.0f trades=%.1f/day%s" % (
            r["z_threshold"], r["sortino"], r["total_pnl"], r["trades_per_day"], star))

print("\nSaved to: %s" % summary_file)
print("DONE")
