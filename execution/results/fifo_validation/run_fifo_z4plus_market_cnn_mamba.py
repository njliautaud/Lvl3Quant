#!/usr/bin/env python3
"""
FIFO Fill Sim — CNN-Mamba v2 — Z4+ MARKET ENTRY Test
======================================================
Tests the HANDOFF hypothesis: market entry at z4+ on CNN-Mamba may absorb
the spread because 63% WR and 13t MFE were observed at that confidence tier
under the chase config (which only filled 30% of high-confidence signals).

Key differences from run_fifo_mfe_mae.py:
  - --market-entry (cross the spread, 100% fill) instead of --chase-entry
  - Thresholds: 3.5, 4.0, 4.5, 5.0 (vs 1.0, 2.0, 3.0)
  - Tighter SL: 10 ticks (vs 20)
  - Exit slippage: 0.2 ticks (realistic 150ms exit latency)
  - CNN-Mamba v2 only (the model with the best z4+ MFE profile)
  - Same March 2-5 OOT dates for direct apples-to-apples comparison

Hypothesis to validate:
  Market entry pays full spread (~0.5-1 tick at typical times) but achieves
  100% fill rate. The chase config achieved only ~30% fill at z3+ but had
  great MFE/MAE. If the alpha at z4+ is ~13t MFE vs 7t MAE, eating ~1t
  of spread should still leave ~5t of expected edge per trade.

Output: /home/jupiter/Lvl3Quant/execution/results/fifo_validation/sim_z4plus_market_<ts>/
"""

import numpy as np
import subprocess
import json
import sys
import os
import gc
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# ── Paths ──
LVL3 = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3 / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3 / "data" / "raw" / "mbo"
EVENT_DIR = LVL3 / "data" / "processed" / "mbo_events_smart_v3"
RESULTS_DIR = LVL3 / "execution" / "results" / "fifo_validation"
PRED_CACHE = RESULTS_DIR / "pred_cache_z4plus"

MODELS = {
    "cnn_mamba_v2": {
        "pred_dir": LVL3 / "output" / "cnn_mamba_v2_smart_v3_mar",
        "folds": ["fold_06", "fold_07", "fold_08", "fold_09"],  # Mar 2-5
    },
}

# ── Constants ──
WINDOW = 1000
STRIDE = 500
HORIZONS = ["1s", "5s", "10s"]
BAR_NS = 100_000_000
N_RTH_BARS = 234000
TICK_VALUE = 12.50
COMMISSION_RT = 4.70

# ── March dates ──
MARCH_DATES = ["20260302", "20260303", "20260304", "20260305"]

# ── Strategy configs ──
# Test high-confidence tier ONLY — z4+ is where chase showed strongest edge
THRESHOLDS = [3.5, 4.0, 4.5, 5.0]

# MARKET entry — cross the spread for guaranteed fill at high conviction
# Tighter stops since high confidence should imply higher quality
# Exit slippage models 150ms exit latency at typical ES vol
BASE_CLI_ARGS = [
    "--market-entry",
    "--stop-loss-ticks", "10",
    "--hold-ms", "60000",
    "--latency-ms", "5",
    "--exit-slippage", "0.2",
    "--conviction-exit-bars", "20",
    "--conviction-exit-mag", "0.5",
    "--prime-hours",
    "--quiet",
]


# ============================================================
# RTH Utilities
# ============================================================

def rth_start_ns_for_date(date_str):
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    d = datetime(year, month, day)
    dst_start_2026 = datetime(2026, 3, 8)
    dst_end_2026 = datetime(2026, 11, 1)
    if dst_start_2026 <= d < dst_end_2026:
        utc_offset = -4
    else:
        utc_offset = -5
    rth_start_utc_hours = 9.5 - utc_offset
    midnight_utc = datetime(year, month, day, tzinfo=timezone.utc)
    rth_start = midnight_utc + timedelta(hours=rth_start_utc_hours)
    return int(rth_start.timestamp() * 1e9)


def map_predictions_to_bars(predictions, event_timestamps, date_str, horizon_idx=2):
    """Map event-level predictions to RTH bar indices. horizon_idx=2 is 10s."""
    n_events = len(event_timestamps)
    n_preds = len(predictions)

    starts = np.arange(0, n_events - WINDOW + 1, STRIDE, dtype=np.int64)
    label_idxs = starts + WINDOW - 1

    if len(label_idxs) > n_preds:
        label_idxs = label_idxs[:n_preds]
    preds_h = predictions[:len(label_idxs), horizon_idx] if predictions.ndim == 2 else predictions[:len(label_idxs)]

    pred_timestamps = event_timestamps[label_idxs]
    rth_start = rth_start_ns_for_date(date_str)
    bar_indices = ((pred_timestamps - rth_start) // BAR_NS).astype(np.int64)
    rth_mask = (bar_indices >= 0) & (bar_indices < N_RTH_BARS)

    return bar_indices[rth_mask], preds_h[rth_mask]


def expanding_zscore(bar_preds_raw, bar_indices, running_stats=None):
    """Compute expanding z-score signal on 234k bar grid."""
    if running_stats is None:
        running_stats = {"sum": 0.0, "sq": 0.0, "count": 0}

    bar_preds = np.zeros(N_RTH_BARS, dtype=np.float64)
    for bi, val in zip(bar_indices, bar_preds_raw):
        bar_preds[bi] = val

    zscore = np.zeros(N_RTH_BARS, dtype=np.float64)
    rs = running_stats["sum"]
    rsq = running_stats["sq"]
    cnt = running_stats["count"]

    for i in range(N_RTH_BARS):
        v = bar_preds[i]
        if v == 0.0:
            continue
        rs += v
        rsq += v * v
        cnt += 1
        if cnt >= 50:
            mean = rs / cnt
            var = (rsq / cnt) - mean * mean
            std = max(np.sqrt(max(var, 0)), 1e-8)
            zscore[i] = (v - mean) / std

    return zscore, {"sum": rs, "sq": rsq, "count": cnt}


def prepare_predictions(model_name, model_cfg):
    """Convert event predictions to bar-level z-score signals."""
    pred_dir = model_cfg["pred_dir"]
    cache_dir = PRED_CACHE / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    running_stats = None
    date_files = {}

    for date_str, fold in zip(MARCH_DATES, model_cfg["folds"]):
        cache_path = cache_dir / f"{date_str}.npz"
        if cache_path.exists():
            print(f"  {model_name}/{date_str}: cached")
            date_files[date_str] = cache_path
            continue

        ev_path = EVENT_DIR / f"{date_str}.npz"
        pred_path = pred_dir / f"{fold}_oot_predictions.npz"
        if not ev_path.exists() or not pred_path.exists():
            print(f"  {model_name}/{date_str}: missing files (ev={ev_path.exists()} pred={pred_path.exists()})")
            continue

        ev_data = np.load(ev_path)
        timestamps = ev_data["ts_event"] if "ts_event" in ev_data.files else ev_data["timestamps"]
        pred_data = np.load(pred_path)
        predictions = pred_data["predictions"] if "predictions" in pred_data.files else pred_data["pred"]

        bar_indices, bar_preds_raw = map_predictions_to_bars(predictions, timestamps, date_str)
        zscore, running_stats = expanding_zscore(bar_preds_raw, bar_indices, running_stats)

        np.savez_compressed(cache_path, predictions=zscore.astype(np.float32))
        n_nz = int(np.sum(zscore != 0))
        print(f"  {model_name}/{date_str}: {n_nz} active bars ({n_nz/N_RTH_BARS*100:.1f}%)")
        date_files[date_str] = cache_path

        del ev_data, timestamps, predictions
        gc.collect()

    return date_files


def run_sim(model_name, date_str, pred_file, threshold, out_dir):
    """Run fill_sim_cli for one model/date/threshold combo."""
    mbo_file = MBO_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not mbo_file.exists():
        print(f"  No MBO file for {date_str}")
        return None

    tag = f"{model_name}_z{threshold:.1f}_{date_str}"
    out_file = out_dir / f"{tag}.json"

    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(threshold),
    ] + BASE_CLI_ARGS

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


def compute_mfe_mae(trades):
    if not trades:
        return {}
    mfes = np.array([t.get("mfe_ticks", 0) for t in trades])
    maes = np.array([abs(t.get("mae_ticks", 0)) for t in trades])
    pnls = np.array([t.get("pnl_dollars", 0) for t in trades])
    ratios = np.array([m/max(a, 0.01) for m, a in zip(mfes, maes)])

    return {
        "n_trades": len(trades),
        "mfe_mean": float(np.mean(mfes)),
        "mfe_median": float(np.median(mfes)),
        "mfe_p75": float(np.percentile(mfes, 75)),
        "mae_mean": float(np.mean(maes)),
        "mae_median": float(np.median(maes)),
        "mae_p75": float(np.percentile(maes, 75)),
        "mfe_mae_ratio_mean": float(np.mean(ratios)),
        "mfe_mae_ratio_median": float(np.median(ratios)),
        "win_rate": float(np.sum(pnls > 0) / len(pnls)),
        "avg_winner": float(np.mean(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0,
        "avg_loser": float(np.mean(pnls[pnls <= 0])) if np.any(pnls <= 0) else 0.0,
        "profit_factor": float(np.sum(pnls[pnls > 0]) / max(abs(np.sum(pnls[pnls <= 0])), 1.0)),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_out = RESULTS_DIR / f"sim_z4plus_market_{ts}"
    sim_out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FIFO Fill Sim — CNN-Mamba v2 — Z4+ MARKET ENTRY (Mar 2-5)")
    print("=" * 80)
    print(f"  Configs: MARKET entry + SL10 + 60s hold + 5ms latency + 0.2t exit slip")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Output: {sim_out}")
    print()

    # Phase 1
    print("Phase 1: Converting event predictions to bar-level z-scores...")
    all_pred_files = {}
    for model_name, model_cfg in MODELS.items():
        print(f"\n  Model: {model_name}")
        date_files = prepare_predictions(model_name, model_cfg)
        all_pred_files[model_name] = date_files
        print(f"  -> {len(date_files)} dates ready")

    # Phase 2
    print("\n\nPhase 2: Running fill sim with MARKET ENTRY...")
    all_results = {}

    for model_name, date_files in all_pred_files.items():
        for threshold in THRESHOLDS:
            key = f"{model_name}_z{threshold:.1f}_market"
            all_results[key] = {}
            print(f"\n  === {key} ===")
            for date_str, pred_file in sorted(date_files.items()):
                t0 = time.time()
                result = run_sim(model_name, date_str, pred_file, threshold, sim_out)
                elapsed = time.time() - t0
                if result:
                    all_results[key][date_str] = result
                    pnl = result.get("total_pnl_dollars", 0)
                    trades = result.get("total_trades", 0)
                    signals = result.get("total_signals", 0)
                    filled = result.get("total_filled", 0)
                    fill_rate = filled / max(signals, 1)
                    wr = result.get("win_rate", 0)
                    print(f"    {date_str}: PnL=${pnl:>8.2f}  trades={trades:>4}  signals={signals:>5}  fillR={fill_rate:.1%}  WR={wr:.1%}  ({elapsed:.1f}s)")
                else:
                    print(f"    {date_str}: NO RESULT ({elapsed:.1f}s)")

    # Phase 3
    print("\n\nPhase 3: Aggregating + MFE/MAE...")
    summary = {}

    for key, date_results in all_results.items():
        if not date_results:
            continue

        total_pnl = 0.0
        total_trades = 0
        total_signals = 0
        total_filled = 0
        all_trades = []

        for date_str, r in date_results.items():
            total_pnl += r.get("total_pnl_dollars", 0)
            total_trades += r.get("total_trades", 0)
            total_signals += r.get("total_signals", 0)
            total_filled += r.get("total_filled", 0)
            all_trades.extend(r.get("trades", []))

        mfe_mae = compute_mfe_mae(all_trades)
        summary[key] = {
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "total_signals": total_signals,
            "total_filled": total_filled,
            "fill_rate": total_filled / max(total_signals, 1),
            "win_rate": mfe_mae.get("win_rate", 0),
            "profit_factor": mfe_mae.get("profit_factor", 0),
            "mfe_mean": mfe_mae.get("mfe_mean", 0),
            "mae_mean": mfe_mae.get("mae_mean", 0),
            "mfe_mae_ratio": mfe_mae.get("mfe_mae_ratio_mean", 0),
            "avg_winner": mfe_mae.get("avg_winner", 0),
            "avg_loser": mfe_mae.get("avg_loser", 0),
        }

    # Save
    summary_path = sim_out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "config": "market_entry_z4plus",
            "model": "cnn_mamba_v2",
            "thresholds": THRESHOLDS,
            "base_args": BASE_CLI_ARGS,
            "dates": MARCH_DATES,
            "summary": summary,
        }, f, indent=2)

    # Print table
    print("\n" + "=" * 100)
    print(f"{'Config':<35} {'Trades':>7} {'Fill%':>6} {'WR':>6} {'PF':>6} {'PnL':>10} {'MFE':>6} {'MAE':>6} {'M/M':>6}")
    print("=" * 100)
    for key, s in summary.items():
        print(f"{key:<35} {s['total_trades']:>7} {s['fill_rate']:>5.1%} {s['win_rate']:>5.1%} {s['profit_factor']:>6.2f} ${s['total_pnl']:>8.2f} {s['mfe_mean']:>5.1f}t {s['mae_mean']:>5.1f}t {s['mfe_mae_ratio']:>5.2f}")
    print("=" * 100)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
