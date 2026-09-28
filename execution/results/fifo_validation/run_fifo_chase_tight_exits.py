#!/usr/bin/env python3
"""
FIFO Sim — Chase Entry + TIGHT EXITS + Latency Filter
======================================================
Builds on the time-of-day decomposition finding:
  - StopLoss hits at 20 ticks = 0% WR, -$261 avg → kills profitability
  - HoldTimeout exits = 50-56% WR, +$15-35 avg → the alpha is real

Tests three hypotheses:
  H1: Tighter SL (10t) + MAE-exit (8t @ 30s) cuts losses without sacrificing too many wins
  H2: Adding ratchet stop locks in MFE before drift-back
  H3: Vol-exit (5t in 5 bars) catches fast adverse moves before SL hit

All on the proven CHASE entry config (don't break what works).

Configs tested (cnn_mamba_v2 + mamba_v7 at z3):
  base    — original chase + SL20 + 60s hold (baseline)
  tight   — chase + SL10 + 60s hold (just tighter SL)
  mae     — chase + SL15 + MAE-exit 8t @ 30s
  ratchet — chase + SL15 + ratchet stop
  vol     — chase + SL15 + vol-exit 5t in 5 bars
  combo   — chase + SL12 + MAE-exit 8t @ 30s + vol-exit 6t in 5 bars
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
THRESHOLD = 3.0  # Best from HANDOFF — z3 chase was the only profitable config

CONFIGS = {
    "base": [
        "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
        "--stop-loss-ticks", "20", "--hold-ms", "60000", "--latency-ms", "5",
        "--conviction-exit-bars", "20", "--conviction-exit-mag", "0.5",
        "--prime-hours", "--quiet",
    ],
    "tight_sl10": [
        "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
        "--stop-loss-ticks", "10", "--hold-ms", "60000", "--latency-ms", "5",
        "--conviction-exit-bars", "20", "--conviction-exit-mag", "0.5",
        "--prime-hours", "--quiet",
    ],
    "mae_exit": [
        "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
        "--stop-loss-ticks", "15", "--hold-ms", "60000", "--latency-ms", "5",
        "--mae-exit-ticks", "8", "--mae-exit-hold-sec", "30",
        "--conviction-exit-bars", "20", "--conviction-exit-mag", "0.5",
        "--prime-hours", "--quiet",
    ],
    "ratchet": [
        "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
        "--stop-loss-ticks", "15", "--hold-ms", "60000", "--latency-ms", "5",
        "--ratchet-stop",
        "--conviction-exit-bars", "20", "--conviction-exit-mag", "0.5",
        "--prime-hours", "--quiet",
    ],
    "vol_exit": [
        "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
        "--stop-loss-ticks", "15", "--hold-ms", "60000", "--latency-ms", "5",
        "--vol-exit-ticks", "5", "--vol-exit-bars", "5",
        "--conviction-exit-bars", "20", "--conviction-exit-mag", "0.5",
        "--prime-hours", "--quiet",
    ],
    "combo": [
        "--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
        "--stop-loss-ticks", "12", "--hold-ms", "60000", "--latency-ms", "5",
        "--mae-exit-ticks", "8", "--mae-exit-hold-sec", "30",
        "--vol-exit-ticks", "6", "--vol-exit-bars", "5",
        "--conviction-exit-bars", "20", "--conviction-exit-mag", "0.5",
        "--prime-hours", "--quiet",
    ],
}


def run_sim(model, date, pred_file, config_name, args, out_dir):
    mbo_file = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"
    if not mbo_file.exists():
        return None
    tag = f"{model}_{config_name}_{date}"
    out_file = out_dir / f"{tag}.json"
    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(THRESHOLD),
    ] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"  FAIL {tag}: {r.stderr[:200]}")
            return None
        if out_file.exists():
            with open(out_file) as f:
                return json.load(f)
    except Exception as e:
        print(f"  ERROR {tag}: {e}")
    return None


def aggregate(date_results):
    total_pnl = 0.0
    total_trades = 0
    total_signals = 0
    total_filled = 0
    all_trades = []
    for date, r in date_results.items():
        total_pnl += r.get("total_pnl_dollars", 0)
        total_trades += r.get("total_trades", 0)
        total_signals += r.get("total_signals", 0)
        total_filled += r.get("total_filled", 0)
        all_trades.extend(r.get("trades", []))
    if not all_trades:
        return None
    pnls = np.array([t.get("pnl_dollars", 0) for t in all_trades])
    mfes = np.array([t.get("mfe_ticks", 0) for t in all_trades])
    maes = np.array([abs(t.get("mae_ticks", 0)) for t in all_trades])
    return {
        "total_pnl": round(float(total_pnl), 2),
        "total_trades": total_trades,
        "total_signals": total_signals,
        "fill_rate": round(total_filled / max(total_signals, 1), 3),
        "win_rate": round(float((pnls > 0).mean()), 3),
        "profit_factor": round(float(pnls[pnls > 0].sum() / max(abs(pnls[pnls <= 0].sum()), 1)), 3),
        "mfe_mean": round(float(mfes.mean()), 2),
        "mae_mean": round(float(maes.mean()), 2),
        "avg_winner": round(float(pnls[pnls > 0].mean()) if np.any(pnls > 0) else 0, 2),
        "avg_loser": round(float(pnls[pnls <= 0].mean()) if np.any(pnls <= 0) else 0, 2),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_out = RESULTS_DIR / f"sim_chase_tight_exits_{ts}"
    sim_out.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FIFO Sim — Chase + Tight Exits — z3 only — Mar 2-5")
    print(f"Configs: {list(CONFIGS.keys())}")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Output: {sim_out}")
    print("=" * 90)

    summary = {}
    for model, pred_dir in MODELS.items():
        for cfg_name, args in CONFIGS.items():
            key = f"{model}_{cfg_name}"
            print(f"\n=== {key} ===")
            date_results = {}
            for date in DATES:
                pred_file = pred_dir / f"{date}.npz"
                if not pred_file.exists():
                    print(f"  {date}: missing pred")
                    continue
                t0 = time.time()
                r = run_sim(model, date, pred_file, cfg_name, args, sim_out)
                el = time.time() - t0
                if r:
                    date_results[date] = r
                    pnl = r.get("total_pnl_dollars", 0)
                    n = r.get("total_trades", 0)
                    sig = r.get("total_signals", 0)
                    fillR = r.get("total_filled", 0) / max(sig, 1)
                    wr = r.get("win_rate", 0)
                    print(f"  {date}: PnL=${pnl:>8.2f}  n={n:>4}  sig={sig:>4}  fillR={fillR:.1%}  WR={wr:.1%}  ({el:.1f}s)")
            agg = aggregate(date_results)
            if agg:
                summary[key] = agg

    summary_path = sim_out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"configs": {k: v for k, v in CONFIGS.items()}, "summary": summary}, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'Config':<30} {'N':>5} {'Fill%':>6} {'WR':>6} {'PF':>6} {'PnL':>10} {'Win$':>7} {'Loss$':>8} {'MFE':>5} {'MAE':>5}")
    print("=" * 100)
    for key, s in summary.items():
        print(f"{key:<30} {s['total_trades']:>5} {s['fill_rate']:>5.1%} {s['win_rate']:>5.1%} {s['profit_factor']:>6.2f} ${s['total_pnl']:>8.2f} ${s['avg_winner']:>5.2f} ${s['avg_loser']:>6.2f} {s['mfe_mean']:>4.1f}t {s['mae_mean']:>4.1f}t")
    print("=" * 100)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
