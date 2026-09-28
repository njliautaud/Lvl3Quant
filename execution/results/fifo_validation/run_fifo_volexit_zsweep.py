#!/usr/bin/env python3
"""
FIFO Sim — Vol-Exit Production Config — Z Threshold Sweep
==========================================================
Takes the winning config from sim_chase_tight_exits_20260427_001619
(chase + SL15 + vol_exit 5/5b + 60s + prime hours) and sweeps z thresholds
to find the sweet spot of trade volume vs WR vs PnL.
"""

import numpy as np
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

LVL3 = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3 / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3 / "data" / "raw" / "mbo"
RESULTS_DIR = LVL3 / "execution" / "results" / "fifo_validation"
PRED_CACHE_BASE = RESULTS_DIR / "pred_cache"

MODELS = {
    "cnn_mamba_v2": PRED_CACHE_BASE / "cnn_mamba_v2",
    "mamba_v7":     PRED_CACHE_BASE / "mamba_v7",
}

DATES = ["20260302", "20260303", "20260304", "20260305"]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

BASE_ARGS = [
    "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
    "--stop-loss-ticks", "15",
    "--vol-exit-ticks", "5", "--vol-exit-bars", "5",
    "--hold-ms", "60000", "--latency-ms", "5",
    "--conviction-exit-bars", "20", "--conviction-exit-mag", "0.5",
    "--prime-hours", "--quiet",
]


def run_sim(model, date, pred_file, threshold, out_dir):
    mbo_file = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"
    if not mbo_file.exists():
        return None
    tag = f"{model}_z{threshold:.1f}_{date}"
    out_file = out_dir / f"{tag}.json"
    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(threshold),
    ] + BASE_ARGS
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return None
        if out_file.exists():
            with open(out_file) as f:
                return json.load(f)
    except Exception:
        return None


def aggregate(date_results):
    if not date_results:
        return None
    total_pnl = sum(r.get("total_pnl_dollars", 0) for r in date_results.values())
    total_trades = sum(r.get("total_trades", 0) for r in date_results.values())
    total_signals = sum(r.get("total_signals", 0) for r in date_results.values())
    total_filled = sum(r.get("total_filled", 0) for r in date_results.values())
    all_trades = [t for r in date_results.values() for t in r.get("trades", [])]
    if not all_trades:
        return {"total_pnl": total_pnl, "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                "fill_rate": total_filled/max(total_signals,1), "n_signals": total_signals}
    pnls = np.array([t.get("pnl_dollars", 0) for t in all_trades])
    return {
        "total_pnl": round(float(total_pnl), 2),
        "total_trades": total_trades,
        "n_signals": total_signals,
        "fill_rate": round(total_filled / max(total_signals, 1), 3),
        "win_rate": round(float((pnls > 0).mean()), 3),
        "profit_factor": round(float(pnls[pnls > 0].sum() / max(abs(pnls[pnls <= 0].sum()), 1)), 3),
        "avg_winner": round(float(pnls[pnls > 0].mean()) if np.any(pnls > 0) else 0, 2),
        "avg_loser": round(float(pnls[pnls <= 0].mean()) if np.any(pnls <= 0) else 0, 2),
        "pnl_per_trade": round(float(pnls.mean()), 2),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_out = RESULTS_DIR / f"sim_volexit_zsweep_{ts}"
    sim_out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {sim_out}")
    print(f"Sweeping z = {THRESHOLDS}")

    summary = {}
    for model, pred_dir in MODELS.items():
        for z in THRESHOLDS:
            key = f"{model}_z{z:.1f}"
            print(f"\n{key}")
            date_results = {}
            for date in DATES:
                pred_file = pred_dir / f"{date}.npz"
                if not pred_file.exists():
                    continue
                r = run_sim(model, date, pred_file, z, sim_out)
                if r:
                    date_results[date] = r
            agg = aggregate(date_results)
            if agg:
                summary[key] = agg
                print(f"  n={agg['total_trades']:>4} sig={agg['n_signals']:>5} fillR={agg['fill_rate']:.1%} WR={agg['win_rate']:.1%} PF={agg['profit_factor']:.2f} PnL=${agg['total_pnl']:>8.2f} pnl/t=${agg['pnl_per_trade']:>5.2f}")

    summary_path = sim_out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"args": BASE_ARGS, "thresholds": THRESHOLDS, "summary": summary}, f, indent=2)

    print("\n" + "=" * 110)
    print(f"{'Config':<25} {'N':>5} {'Sig':>5} {'Fill%':>6} {'WR':>6} {'PF':>6} {'PnL':>10} {'$/t':>7} {'Win$':>7} {'Loss$':>8}")
    print("=" * 110)
    for key, s in summary.items():
        print(f"{key:<25} {s['total_trades']:>5} {s['n_signals']:>5} {s['fill_rate']:>5.1%} {s['win_rate']:>5.1%} {s['profit_factor']:>6.2f} ${s['total_pnl']:>8.2f} ${s['pnl_per_trade']:>5.2f} ${s.get('avg_winner',0):>5.2f} ${s.get('avg_loser',0):>6.2f}")
    print("=" * 110)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
