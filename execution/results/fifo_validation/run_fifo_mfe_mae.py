#!/usr/bin/env python3
"""
FIFO Fill Simulator — Mamba v7 + CNN-Mamba v2 on March 2-5
==========================================================
Runs the Rust fill_sim_cli with chase entry + conviction exit configs
at three threshold levels (z=1, z=2, z=3), then computes MFE/MAE analysis.

Configs:
  Chase entry (2 tick, 5 reprice) + conviction exit (20 bars, mag=0.5)
  + SL 20 ticks + hold 60s + latency 5ms
  Thresholds: 1, 2, 3
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
PRED_CACHE = RESULTS_DIR / "pred_cache"
PRED_CACHE.mkdir(parents=True, exist_ok=True)

# ── Models ──
MODELS = {
    "mamba_v7": {
        "pred_dir": LVL3 / "output" / "mamba_v7_tiny_smart_v3_mar_apr",
        "folds": ["fold_06", "fold_07", "fold_08", "fold_09"],  # Mar 2-5
    },
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
THRESHOLDS = [1.0, 2.0, 3.0]
BASE_CLI_ARGS = [
    "--chase-entry",
    "--chase-max-ticks", "2",
    "--chase-max-reprices", "5",
    "--stop-loss-ticks", "20",
    "--hold-ms", "60000",
    "--latency-ms", "5",
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


# ============================================================
# Prediction Preparation
# ============================================================

def prepare_predictions(model_name, model_cfg):
    """Convert event predictions to bar-level z-score signals."""
    pred_dir = model_cfg["pred_dir"]
    cache_dir = PRED_CACHE / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    running_stats = None
    date_files = {}

    for fold_name in model_cfg["folds"]:
        fold_file = pred_dir / f"{fold_name}_oot_predictions.npz"
        if not fold_file.exists():
            print(f"  WARNING: {fold_file} not found")
            continue

        data = np.load(str(fold_file), allow_pickle=True)
        oot_path = str(data["oot_files"][0])
        basename = oot_path.replace("\\", "/").split("/")[-1]
        date_str = basename.split("_")[0]

        if date_str not in MARCH_DATES:
            print(f"  Skipping {fold_name} ({date_str}) - not in March 2-5")
            continue

        cache_path = cache_dir / f"{date_str}.npz"
        if cache_path.exists():
            print(f"  {model_name}/{date_str}: cached")
            # Update running stats from cached
            cached = np.load(str(cache_path))
            nz = cached["predictions"][cached["predictions"] != 0]
            if running_stats is None:
                running_stats = {"sum": 0.0, "sq": 0.0, "count": 0}
            running_stats["sum"] += float(np.sum(nz))
            running_stats["sq"] += float(np.sum(nz ** 2))
            running_stats["count"] += len(nz)
            date_files[date_str] = cache_path
            continue

        # Load event timestamps
        ev_file = EVENT_DIR / f"{date_str}_mbo_events.npz"
        if not ev_file.exists():
            print(f"  WARNING: Event file not found for {date_str}")
            continue

        ev_data = np.load(str(ev_file), allow_pickle=True)
        timestamps = ev_data["timestamps"]

        predictions = data["predictions"].astype(np.float64)
        bar_idx, preds_rth = map_predictions_to_bars(predictions, timestamps, date_str)
        zscore, running_stats = expanding_zscore(preds_rth, bar_idx, running_stats)

        np.savez_compressed(str(cache_path), predictions=zscore.astype(np.float32))
        n_nz = int(np.count_nonzero(zscore))
        print(f"  {model_name}/{date_str}: {n_nz} active bars ({n_nz/N_RTH_BARS*100:.1f}%)")
        date_files[date_str] = cache_path

        del ev_data, timestamps, predictions
        gc.collect()

    return date_files


# ============================================================
# Fill Sim Execution
# ============================================================

def run_sim(model_name, date_str, pred_file, threshold, out_dir):
    """Run fill_sim_cli for one model/date/threshold combo."""
    mbo_file = MBO_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not mbo_file.exists():
        print(f"  No MBO file for {date_str}")
        return None

    tag = f"{model_name}_z{threshold:.0f}_{date_str}"
    out_file = out_dir / f"{tag}.json"

    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(threshold),
    ] + BASE_CLI_ARGS

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  FAIL {tag}: {r.stderr[:200]}")
            return None
        if out_file.exists():
            with open(out_file) as f:
                return json.load(f)
    except Exception as e:
        print(f"  ERROR {tag}: {e}")
    return None


# ============================================================
# MFE/MAE Analysis
# ============================================================

def compute_mfe_mae(trades):
    """Compute MFE/MAE statistics from trade list."""
    if not trades:
        return {}

    mfes = []
    maes = []
    mfe_mae_ratios = []

    for t in trades:
        mfe = t.get("mfe_ticks", 0)
        mae = t.get("mae_ticks", 0)
        mfes.append(mfe)
        maes.append(abs(mae))
        if abs(mae) > 0:
            mfe_mae_ratios.append(mfe / abs(mae))

    mfes = np.array(mfes)
    maes = np.array(maes)
    ratios = np.array(mfe_mae_ratios) if mfe_mae_ratios else np.array([0])

    result = {
        "n_trades": len(trades),
        "mfe_mean": float(np.mean(mfes)),
        "mfe_median": float(np.median(mfes)),
        "mfe_p25": float(np.percentile(mfes, 25)),
        "mfe_p75": float(np.percentile(mfes, 75)),
        "mae_mean": float(np.mean(maes)),
        "mae_median": float(np.median(maes)),
        "mae_p25": float(np.percentile(maes, 25)),
        "mae_p75": float(np.percentile(maes, 75)),
        "mfe_mae_ratio_mean": float(np.mean(ratios)),
        "mfe_mae_ratio_median": float(np.median(ratios)),
    }

    # Confidence tier analysis (by z-score)
    for z_min, z_label in [(1.0, "z1+"), (2.0, "z2+"), (3.0, "z3+"), (4.0, "z4+")]:
        tier_trades = [t for t in trades if abs(t.get("signal_strength", t.get("entry_signal", 0))) >= z_min]
        if tier_trades:
            tier_mfes = [t.get("mfe_ticks", 0) for t in tier_trades]
            tier_maes = [abs(t.get("mae_ticks", 0)) for t in tier_trades]
            tier_ratios = [m / max(a, 0.01) for m, a in zip(tier_mfes, tier_maes)]
            result[f"{z_label}_n"] = len(tier_trades)
            result[f"{z_label}_mfe"] = float(np.mean(tier_mfes))
            result[f"{z_label}_mae"] = float(np.mean(tier_maes))
            result[f"{z_label}_ratio"] = float(np.mean(tier_ratios))
            tier_pnls = [t.get("pnl_dollars", 0) for t in tier_trades]
            result[f"{z_label}_winrate"] = sum(1 for p in tier_pnls if p > 0) / len(tier_pnls)

    return result


# ============================================================
# Main
# ============================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sim_out = RESULTS_DIR / f"sim_{ts}"
    sim_out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FIFO Fill Sim — Mamba v7 + CNN-Mamba v2 — March 2-5, 2026")
    print("=" * 80)
    print(f"  Configs: Chase(2tick/5reprice) + Conviction(20bar/0.5mag) + SL20 + 60s hold + 5ms latency")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Output: {sim_out}")
    print()

    # Phase 1: Prepare predictions
    print("Phase 1: Converting event predictions to bar-level z-scores...")
    all_pred_files = {}
    for model_name, model_cfg in MODELS.items():
        print(f"\n  Model: {model_name}")
        date_files = prepare_predictions(model_name, model_cfg)
        all_pred_files[model_name] = date_files
        print(f"  -> {len(date_files)} dates ready")

    # Phase 2: Run fill sim
    print("\n\nPhase 2: Running fill sim...")
    all_results = {}  # (model, threshold) -> {date: result}

    for model_name, date_files in all_pred_files.items():
        for threshold in THRESHOLDS:
            key = f"{model_name}_z{threshold:.0f}"
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

    # Phase 3: Aggregate + MFE/MAE
    print("\n\nPhase 3: Aggregating results + MFE/MAE analysis...")
    summary = {}

    for key, date_results in all_results.items():
        if not date_results:
            continue

        total_pnl = 0
        total_trades = 0
        total_signals = 0
        total_filled = 0
        total_wins = 0
        all_trades = []
        daily_pnls = []

        for date_str, res in sorted(date_results.items()):
            day_pnl = res.get("total_pnl_dollars", 0)
            total_pnl += day_pnl
            total_trades += res.get("total_trades", 0)
            total_signals += res.get("total_signals", 0)
            total_filled += res.get("total_filled", 0)
            daily_pnls.append(day_pnl)

            if "trades" in res:
                for t in res["trades"]:
                    if t.get("pnl_dollars", 0) > 0:
                        total_wins += 1
                    all_trades.append(t)

        n_days = len(date_results)
        win_rate = total_wins / max(total_trades, 1)
        fill_rate = total_filled / max(total_signals, 1)
        avg_daily = np.mean(daily_pnls) if daily_pnls else 0

        # Profit factor
        gross_profit = sum(t.get("pnl_dollars", 0) for t in all_trades if t.get("pnl_dollars", 0) > 0)
        gross_loss = abs(sum(t.get("pnl_dollars", 0) for t in all_trades if t.get("pnl_dollars", 0) < 0))
        pf = gross_profit / max(gross_loss, 0.01)

        # Sortino
        if len(daily_pnls) > 1:
            downside = [min(0, x) for x in daily_pnls]
            ds_std = np.std(downside)
            sortino = (avg_daily / max(ds_std, 1e-8)) * np.sqrt(252)
        else:
            sortino = 0.0

        # MFE/MAE
        mfe_mae = compute_mfe_mae(all_trades)

        summary[key] = {
            "total_pnl": round(total_pnl, 2),
            "n_days": n_days,
            "n_trades": total_trades,
            "n_signals": total_signals,
            "fill_rate": round(fill_rate, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(pf, 2),
            "sortino": round(sortino, 2),
            "avg_daily_pnl": round(avg_daily, 2),
            "avg_trade_pnl": round(total_pnl / max(total_trades, 1), 2),
            "daily_pnls": {d: round(p, 2) for d, p in zip(sorted(date_results.keys()), daily_pnls)},
            "mfe_mae": mfe_mae,
        }

    # Phase 4: Print results
    print("\n" + "=" * 120)
    print("  FIFO FILL SIM RESULTS — March 2-5, 2026")
    print("  Config: Chase(2/5) + Conviction(20bar/0.5) + SL20 + 60s + 5ms latency")
    print("=" * 120)

    header = f"{'Strategy':<30} {'PnL':>10} {'Trades':>7} {'Signals':>8} {'FillR':>6} {'WinR':>6} {'PF':>5} {'Sortino':>8} {'AvgTrd':>8}"
    print(header)
    print("-" * 120)

    for key in sorted(summary.keys()):
        s = summary[key]
        marker = "+" if s["total_pnl"] > 0 else " "
        print(f"{key:<30} {marker}${abs(s['total_pnl']):>8,.0f} {s['n_trades']:>7} {s['n_signals']:>8} {s['fill_rate']:>5.1%} {s['win_rate']:>5.1%} {s['profit_factor']:>5.2f} {s['sortino']:>8.2f} ${s['avg_trade_pnl']:>7.2f}")

    # Per-day breakdown
    print(f"\n  PER-DAY BREAKDOWN:")
    for key in sorted(summary.keys()):
        s = summary[key]
        daily = s["daily_pnls"]
        days_str = "  ".join(f"{d}:${p:>+8,.0f}" for d, p in sorted(daily.items()))
        print(f"  {key:<30} {days_str}")

    # MFE/MAE table
    print(f"\n  MFE/MAE ANALYSIS (ticks):")
    print(f"{'Strategy':<30} {'MFE_avg':>8} {'MFE_med':>8} {'MAE_avg':>8} {'MAE_med':>8} {'Ratio':>8}")
    print("-" * 80)
    for key in sorted(summary.keys()):
        mm = summary[key].get("mfe_mae", {})
        if not mm:
            continue
        print(f"{key:<30} {mm.get('mfe_mean',0):>8.2f} {mm.get('mfe_median',0):>8.2f} {mm.get('mae_mean',0):>8.2f} {mm.get('mae_median',0):>8.2f} {mm.get('mfe_mae_ratio_mean',0):>8.2f}")

    # Confidence tier MFE/MAE
    print(f"\n  MFE/MAE BY CONFIDENCE TIER:")
    print(f"{'Strategy':<30} {'Tier':<6} {'N':>6} {'MFE':>8} {'MAE':>8} {'Ratio':>8} {'WinR':>6}")
    print("-" * 80)
    for key in sorted(summary.keys()):
        mm = summary[key].get("mfe_mae", {})
        for tier in ["z1+", "z2+", "z3+", "z4+"]:
            n = mm.get(f"{tier}_n", 0)
            if n > 0:
                print(f"{key:<30} {tier:<6} {n:>6} {mm[f'{tier}_mfe']:>8.2f} {mm[f'{tier}_mae']:>8.2f} {mm[f'{tier}_ratio']:>8.2f} {mm[f'{tier}_winrate']:>5.1%}")

    # Save results
    out_file = RESULTS_DIR / f"fifo_results_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_file}")

    return summary


if __name__ == "__main__":
    main()
