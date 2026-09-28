#!/usr/bin/env python3
"""
Execution Strategy Sweep — REAL Market Replay Fill Sim
======================================================
Tests multiple execution strategies against LGBM smart_v2 and PatchTST
predictions using the Rust event-by-event FIFO fill simulator.

This is NOT a midprice sim. It replays actual MBO events, maintains real
order book queues, and models fill probability based on queue depth.

Strategies tested:
  1. Passive Limit (resting at BBO, wait for fill)
  2. Chase Entry (cancel-replace up to N ticks if BBO moves)
  3. Market Entry (cross spread, 100% fill, pay spread cost)
  4. Passive + Signal Flip Exit (exit when signal reverses)
  5. Chase + Trailing Stop (chase entry, trail 3-5 ticks)
  6. Market + Tight TP/SL (market entry, TP=3t, SL=2t)
  7. Prime Hours Only (10:30-2:30 PM)
  8. High Confidence Only (top percentile signals)
  9. MAE Exit (cut losers after N seconds underwater)
 10. Ratchet Stop (lock in profit at MFE thresholds)

Usage:
    python3 scripts/execution_strategy_sweep.py \
        --model patchtst \
        --model-dir output/patchtst_sliding60d_smart_v2 \
        --window 1000 --stride 500 \
        --folds 0-6

    python3 scripts/execution_strategy_sweep.py \
        --model lgbm \
        --lgbm-dir alpha_discovery/deep_models/results/lgbm_da_smart_v2_1d_oot \
        --folds 0-8
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/jupiter/Lvl3Quant")
DATA_DIR    = BASE_DIR / "data/processed/mbo_events_smart_v2"
RAW_MBO_DIR = BASE_DIR / "data/raw/mbo"
FILL_SIM    = BASE_DIR / "rust_cache_builder/target/release/fill_sim_cli"
OUTPUT_BASE = BASE_DIR / "data/processed/fillsim_strategies"

# ── RTH constants ────────────────────────────────────────────────────────────
BAR_NS        = 100_000_000
RTH_START_MIN = 9 * 60 + 30
RTH_END_MIN   = 16 * 60
RTH_BARS      = int((RTH_END_MIN - RTH_START_MIN) * 60 * 10)  # 234,000

DST_END_2025_NS   = 1_762_056_000_000_000_000
DST_START_2026_NS = 1_773_562_800_000_000_000


def et_offset_hours(ts_ns: int) -> int:
    if ts_ns >= DST_START_2026_NS:
        return -4
    if ts_ns < DST_END_2025_NS:
        return -4
    return -5


def rth_start_ns_for_day(ts_ns: int) -> int:
    ts_sec = ts_ns // 1_000_000_000
    offset = et_offset_hours(ts_ns)
    et_sec = ts_sec + offset * 3600
    day_start_et = (et_sec // 86400) * 86400
    rth_start_et = day_start_et + RTH_START_MIN * 60
    return (rth_start_et - offset * 3600) * 1_000_000_000


def ts_to_rth_bar(ts_ns: int) -> int:
    ts_sec = ts_ns // 1_000_000_000
    et_sec = ts_sec + et_offset_hours(ts_ns) * 3600
    secs_in_day = et_sec % 86400
    mins_in_day = secs_in_day / 60.0
    if mins_in_day < RTH_START_MIN or mins_in_day >= RTH_END_MIN:
        return -1
    elapsed_ns = ts_ns - rth_start_ns_for_day(ts_ns)
    return int(elapsed_ns // BAR_NS)


def extract_date(fpath: str) -> str:
    import re
    m = re.search(r'(\d{8})_mbo_events', str(fpath))
    if m:
        return m.group(1)
    fname = str(fpath).replace("\\", "/").split("/")[-1]
    return fname.split("_")[0]


def find_raw_mbo(date_str: str) -> Optional[Path]:
    for ext in [".mbo.dbn.zst", ".mbo.dbn"]:
        p = RAW_MBO_DIR / f"glbx-mdp3-{date_str}{ext}"
        if p.exists():
            return p
    return None


# ── Execution Strategy Definitions ──────────────────────────────────────────
STRATEGIES = {
    "passive_hold10s": {
        "desc": "Passive limit entry, 10s hold, no TP/SL",
        "args": ["--hold-ms", "10000", "--latency-ms", "10"],
    },
    "passive_hold30s": {
        "desc": "Passive limit entry, 30s hold",
        "args": ["--hold-ms", "30000", "--latency-ms", "10"],
    },
    "chase_hold10s": {
        "desc": "Chase entry (2t/5 reprices), 10s hold",
        "args": ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
                 "--hold-ms", "10000", "--latency-ms", "10"],
    },
    "chase_tpsl": {
        "desc": "Chase entry, TP=4t SL=2t, 30s max hold",
        "args": ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
                 "--take-profit-ticks", "4", "--stop-loss-ticks", "2",
                 "--hold-ms", "30000", "--latency-ms", "10"],
    },
    "chase_trail3": {
        "desc": "Chase entry, trailing stop 3t, 60s max hold",
        "args": ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
                 "--trailing-ticks", "3", "--hold-ms", "60000", "--latency-ms", "10"],
    },
    "market_hold10s": {
        "desc": "Market entry (100% fill, pay spread), 10s hold",
        "args": ["--market-entry", "--hold-ms", "10000", "--latency-ms", "5"],
    },
    "market_tpsl_tight": {
        "desc": "Market entry, TP=3t SL=2t",
        "args": ["--market-entry", "--take-profit-ticks", "3", "--stop-loss-ticks", "2",
                 "--hold-ms", "30000", "--latency-ms", "5"],
    },
    "market_tpsl_wide": {
        "desc": "Market entry, TP=6t SL=3t, 60s hold",
        "args": ["--market-entry", "--take-profit-ticks", "6", "--stop-loss-ticks", "3",
                 "--hold-ms", "60000", "--latency-ms", "5"],
    },
    "chase_signal_flip": {
        "desc": "Chase entry + exit on signal flip",
        "args": ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
                 "--signal-flip-exit", "--hold-ms", "60000", "--latency-ms", "10"],
    },
    "chase_prime_hours": {
        "desc": "Chase entry, prime hours only (10:30-2:30)",
        "args": ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
                 "--hold-ms", "10000", "--latency-ms", "10", "--prime-hours"],
    },
    "market_prime_tpsl": {
        "desc": "Market entry, prime hours, TP=4t SL=2t",
        "args": ["--market-entry", "--take-profit-ticks", "4", "--stop-loss-ticks", "2",
                 "--hold-ms", "30000", "--latency-ms", "5", "--prime-hours"],
    },
    "chase_mae_exit": {
        "desc": "Chase entry + MAE exit (cut losers after 10t/600s)",
        "args": ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
                 "--mae-exit-ticks", "10", "--mae-exit-hold-sec", "600",
                 "--hold-ms", "1800000", "--latency-ms", "10"],
    },
    "chase_ratchet": {
        "desc": "Chase entry + ratcheting stop (lock in MFE profit)",
        "args": ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
                 "--ratchet-stop", "--hold-ms", "60000", "--latency-ms", "10"],
    },
    "chase_conviction": {
        "desc": "Chase entry + conviction exit (10 bars opposing signal)",
        "args": ["--chase-entry", "--chase-max-ticks", "2", "--chase-max-reprices", "5",
                 "--conviction-exit-bars", "100",
                 "--hold-ms", "60000", "--latency-ms", "10"],
    },
}

# Signal threshold tiers to test
THRESHOLD_TIERS = {
    "all_signals":   0.0,
    "moderate":      0.15,
    "strong":        0.30,
    "very_strong":   0.50,
}


# ── Prediction Converters ───────────────────────────────────────────────────

def convert_patchtst_predictions(model_dir: Path, folds: List[int],
                                  window: int, stride: int, horizon_idx: int = 2):
    """Convert PatchTST fold predictions to per-day bar-indexed NPZs.
    horizon_idx: 0=1s, 1=5s, 2=10s (default)"""
    all_bars = []

    for fold_idx in folds:
        pred_file = model_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
        if not pred_file.exists():
            print(f"  Fold {fold_idx}: not found, skipping")
            continue

        f = np.load(pred_file, allow_pickle=True)
        preds = f["predictions"]  # (N, 3)
        oot_files = f["oot_files"]

        sample_offset = 0
        for oot_path in oot_files:
            date_str = extract_date(str(oot_path))
            local_file = DATA_DIR / f"{date_str}_mbo_events.npz"
            if not local_file.exists():
                print(f"    WARNING: {date_str} data file not found, skipping")
                continue

            d = np.load(local_file, allow_pickle=True)
            timestamps = d["timestamps"]
            n_events = len(timestamps)

            if n_events < window:
                continue

            starts = np.arange(0, n_events - window + 1, stride, dtype=np.int64)
            end_indices = starts + window - 1
            n_windows = len(end_indices)

            file_preds = preds[sample_offset:sample_offset + n_windows, horizon_idx]

            for end_idx, pred_val in zip(end_indices, file_preds):
                if np.isnan(pred_val):
                    continue
                bar_idx = ts_to_rth_bar(int(timestamps[end_idx]))
                if 0 <= bar_idx < RTH_BARS:
                    all_bars.append((date_str, bar_idx, float(pred_val)))

            sample_offset += n_windows

        print(f"  Fold {fold_idx}: {len(all_bars)} total bars so far")

    return all_bars


def convert_lgbm_predictions(lgbm_dir: Path, folds: List[int],
                              window: int, stride: int):
    """Convert LGBM DA classifier predictions to per-day bar-indexed NPZs.
    Uses confidence * direction as the signal (positive = buy, negative = sell)."""
    all_bars = []

    # LGBM predictions are per-fold, each fold covers TEST_DAYS days
    # We need to reconstruct which events map to which fold
    # Get the full file list to compute fold boundaries
    files = sorted(DATA_DIR.glob("*.npz"))
    valid_files = []
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            if "labels_10s" in d and not np.all(np.isnan(d["labels_10s"])):
                valid_files.append(f)
        except Exception:
            print(f"    WARNING: Bad file {f.name}, skipping")
            continue

    print(f"  Found {len(valid_files)} valid data files")

    # Detect test_days from prediction file sizes
    # For 1d OOT: each fold is 1 day
    # Check first fold to determine
    fold0 = np.load(lgbm_dir / "fold00_preds.npz", allow_pickle=True)
    n_preds_f0 = len(fold0["preds"])

    # Determine fold boundaries (sliding window, TRAIN_DAYS=60)
    train_days = 60
    # Detect test_days: check if fold files have metadata
    # For 1d OOT: folds slide by 1 day
    # For 15d OOT: folds slide by 15 days

    # Heuristic: count predictions in fold 0 vs fold 1 to infer test_days
    fold1_file = lgbm_dir / "fold01_preds.npz"
    if fold1_file.exists():
        fold1 = np.load(fold1_file, allow_pickle=True)
        # If both have similar small counts, likely 1d OOT
        if n_preds_f0 < 50000:  # 1 day typically has ~15-30k windows
            test_days = 1
        else:
            test_days = 15
    else:
        test_days = 1

    print(f"  Detected test_days={test_days}")

    min_train = max(train_days, 30)
    test_start = min_train

    for fold_idx in folds:
        pred_file = lgbm_dir / f"fold{fold_idx:02d}_preds.npz"
        if not pred_file.exists():
            print(f"  Fold {fold_idx}: not found, skipping")
            continue

        f = np.load(pred_file, allow_pickle=True)
        preds_dir = f["preds"]      # 1=up, 0=down
        confidence = f["confidence"]  # |P(up) - 0.5|

        # Signal = direction * confidence (positive = buy, negative = sell)
        signal = np.where(preds_dir == 1, confidence, -confidence).astype(np.float64)

        # Which days does this fold cover?
        fold_test_start = min_train + fold_idx * test_days
        fold_test_end = fold_test_start + test_days

        if fold_test_end > len(valid_files):
            print(f"  Fold {fold_idx}: exceeds file count, skipping")
            continue

        fold_files = valid_files[fold_test_start:fold_test_end]

        # Reconstruct windows per file
        sample_offset = 0
        for data_file in fold_files:
            date_str = extract_date(data_file.name)
            d = np.load(data_file, allow_pickle=True)
            timestamps = d["timestamps"]
            n_events = len(timestamps)

            if n_events < window:
                continue

            starts = np.arange(0, n_events - window + 1, stride, dtype=np.int64)
            end_indices = starts + window - 1

            # Filter out windows with NaN labels (same as training)
            labels = d.get("labels_10s", None)
            if labels is not None:
                valid_mask = []
                for si in starts:
                    ei = si + window - 1
                    if ei < len(labels) and not np.isnan(labels[ei]):
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                valid_mask = np.array(valid_mask)
                end_indices = end_indices[valid_mask]
                n_valid = valid_mask.sum()
            else:
                n_valid = len(end_indices)

            if sample_offset + n_valid > len(signal):
                n_valid = min(n_valid, len(signal) - sample_offset)

            file_signal = signal[sample_offset:sample_offset + n_valid]

            for i, (end_idx, sig_val) in enumerate(zip(end_indices[:n_valid], file_signal)):
                bar_idx = ts_to_rth_bar(int(timestamps[end_idx]))
                if 0 <= bar_idx < RTH_BARS:
                    all_bars.append((date_str, bar_idx, float(sig_val)))

            sample_offset += n_valid

        print(f"  Fold {fold_idx}: {sample_offset} windows, {len(all_bars)} total bars")

    return all_bars


def write_per_day_preds(all_bars, output_dir: Path) -> List[Tuple[str, int]]:
    """Write per-day bar-indexed prediction NPZs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    by_date = defaultdict(list)
    for date_str, bar_idx, pred_val in all_bars:
        by_date[date_str].append((bar_idx, pred_val))

    written = []
    for date_str in sorted(by_date.keys()):
        bars = by_date[date_str]
        bar_preds = np.zeros(RTH_BARS, dtype=np.float64)
        bar_counts = np.zeros(RTH_BARS, dtype=np.int32)
        for bar_idx, pred_val in bars:
            bar_preds[bar_idx] += pred_val
            bar_counts[bar_idx] += 1

        mask = bar_counts > 0
        bar_preds[mask] /= bar_counts[mask]

        out_path = output_dir / f"{date_str}_preds.npz"
        np.savez_compressed(out_path, predictions=bar_preds)
        written.append((date_str, mask.sum()))

    return written


def run_strategy(date_str: str, pred_path: Path, strategy_name: str,
                 strategy_args: List[str], threshold: float,
                 sim_dir: Path) -> Optional[Dict]:
    """Run fill_sim_cli for one day with one strategy."""
    raw_mbo = find_raw_mbo(date_str)
    if raw_mbo is None:
        return None

    out_path = sim_dir / f"{date_str}_{strategy_name}.json"

    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(raw_mbo),
        "--predictions", str(pred_path),
        "--output", str(out_path),
        "--signal-threshold", str(threshold),
    ] + strategy_args

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return None
        if out_path.exists():
            with open(out_path) as f:
                return json.load(f)
    except (subprocess.TimeoutExpired, Exception):
        pass

    return None


def compute_metrics(day_results: List[Dict]) -> Dict:
    """Compute aggregate metrics from daily results."""
    if not day_results:
        return {}

    pnls = [d.get("pnl", 0) for d in day_results]
    trades = sum(d.get("trades", 0) for d in day_results)
    signals = sum(d.get("signals", 0) for d in day_results)
    n_days = len(pnls)

    metrics = {
        "n_days": n_days,
        "total_pnl": sum(pnls),
        "avg_daily_pnl": np.mean(pnls),
        "total_trades": trades,
        "total_signals": signals,
        "fill_rate": trades / max(signals, 1),
        "positive_days": sum(1 for p in pnls if p > 0),
        "win_pct": sum(1 for p in pnls if p > 0) / max(n_days, 1),
        "max_daily_pnl": max(pnls),
        "min_daily_pnl": min(pnls),
        "max_drawdown": min(np.minimum.accumulate(np.cumsum(pnls))),
    }

    if n_days > 1 and np.std(pnls) > 0:
        metrics["sharpe"] = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        neg_pnls = [p for p in pnls if p < 0]
        if neg_pnls and np.std(neg_pnls) > 0:
            metrics["sortino"] = np.mean(pnls) / np.std(neg_pnls) * np.sqrt(252)
        else:
            metrics["sortino"] = float('inf') if np.mean(pnls) > 0 else 0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Execution Strategy Sweep — Real Market Replay")
    parser.add_argument("--model", required=True, choices=["patchtst", "lgbm"],
                       help="Model type")
    parser.add_argument("--model-dir", default=None,
                       help="Model results dir (for PatchTST)")
    parser.add_argument("--lgbm-dir", default=None,
                       help="LGBM results dir")
    parser.add_argument("--window", type=int, default=1000, help="Event window size")
    parser.add_argument("--stride", type=int, default=500, help="Window stride")
    parser.add_argument("--folds", default="0-6", help="Fold range (e.g. '0-6' or '0,2,4')")
    parser.add_argument("--horizon", default="10s", choices=["1s", "5s", "10s"])
    parser.add_argument("--strategies", default="all",
                       help="Comma-separated strategies or 'all'")
    parser.add_argument("--thresholds", default="moderate",
                       help="Comma-separated threshold tiers or single tier")
    parser.add_argument("--skip-convert", action="store_true",
                       help="Skip prediction conversion (use existing per-day files)")
    args = parser.parse_args()

    # Parse folds
    if "-" in args.folds:
        lo, hi = args.folds.split("-")
        fold_list = list(range(int(lo), int(hi) + 1))
    else:
        fold_list = [int(x) for x in args.folds.split(",")]

    # Parse strategies
    if args.strategies == "all":
        strat_names = list(STRATEGIES.keys())
    else:
        strat_names = [s.strip() for s in args.strategies.split(",")]

    # Parse thresholds
    if args.thresholds in THRESHOLD_TIERS:
        thresholds = {args.thresholds: THRESHOLD_TIERS[args.thresholds]}
    elif args.thresholds == "all":
        thresholds = THRESHOLD_TIERS
    else:
        thresholds = {}
        for t in args.thresholds.split(","):
            t = t.strip()
            if t in THRESHOLD_TIERS:
                thresholds[t] = THRESHOLD_TIERS[t]
            else:
                thresholds[f"custom_{t}"] = float(t)

    horizon_map = {"1s": 0, "5s": 1, "10s": 2}
    horizon_idx = horizon_map[args.horizon]

    output_dir = OUTPUT_BASE / f"{args.model}_{args.horizon}"
    pred_dir = output_dir / "per_day_preds"
    sim_dir = output_dir / "sim_results"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"EXECUTION STRATEGY SWEEP — REAL MARKET REPLAY")
    print(f"{'='*70}")
    print(f"Model:      {args.model}")
    print(f"Horizon:    {args.horizon}")
    print(f"Folds:      {fold_list}")
    print(f"Strategies: {len(strat_names)}")
    print(f"Thresholds: {thresholds}")
    print(f"Output:     {output_dir}")
    print()

    # ── Step 1: Convert predictions ─────────────────────────────────────────
    if not args.skip_convert:
        print("Step 1: Converting predictions to per-day bar-indexed format...")

        if args.model == "patchtst":
            model_dir = BASE_DIR / (args.model_dir or "output/patchtst_sliding60d_smart_v2")
            all_bars = convert_patchtst_predictions(
                model_dir, fold_list, args.window, args.stride, horizon_idx)
        elif args.model == "lgbm":
            lgbm_dir = BASE_DIR / (args.lgbm_dir or "alpha_discovery/deep_models/results/lgbm_da_smart_v2_1d_oot")
            all_bars = convert_lgbm_predictions(
                lgbm_dir, fold_list, args.window, args.stride)

        if not all_bars:
            print("ERROR: No predictions converted!")
            sys.exit(1)

        print(f"\nTotal: {len(all_bars)} bar predictions")
        written = write_per_day_preds(all_bars, pred_dir)
        print(f"Written: {len(written)} day files\n")
    else:
        written = [(f.stem.replace("_preds", ""), 0)
                   for f in sorted(pred_dir.glob("*_preds.npz"))]
        print(f"Using existing {len(written)} day files\n")

    # ── Step 2: Check which days have raw MBO data ──────────────────────────
    sim_dates = []
    for date_str, n_bars in written:
        if find_raw_mbo(date_str) is not None:
            sim_dates.append(date_str)
    print(f"Days with both predictions + raw MBO: {len(sim_dates)}")
    if not sim_dates:
        print("ERROR: No overlapping dates!")
        sys.exit(1)

    # ── Step 3: Run all strategy × threshold combinations ───────────────────
    all_results = {}
    total_combos = len(strat_names) * len(thresholds)
    combo_idx = 0

    for thresh_name, thresh_val in thresholds.items():
        for strat_name in strat_names:
            combo_idx += 1
            strat = STRATEGIES[strat_name]
            combo_key = f"{strat_name}__{thresh_name}"

            print(f"\n[{combo_idx}/{total_combos}] {strat_name} | threshold={thresh_name}({thresh_val})")
            print(f"  {strat['desc']}")

            day_results = []
            for date_str in sim_dates:
                pred_path = pred_dir / f"{date_str}_preds.npz"
                result = run_strategy(
                    date_str, pred_path, strat_name,
                    strat["args"], thresh_val, sim_dir)

                if result:
                    # Handle both top-level and summary sub-object
                    s = result.get("summary", result)
                    day_results.append({
                        "date": date_str,
                        "pnl": s.get("total_pnl_dollars", s.get("total_pnl", 0)),
                        "trades": s.get("total_trades", s.get("trades", 0)),
                        "signals": s.get("total_signals", 0),
                        "win_rate": s.get("win_rate", 0),
                    })

            metrics = compute_metrics(day_results)
            all_results[combo_key] = {
                "strategy": strat_name,
                "threshold": thresh_name,
                "threshold_val": thresh_val,
                "desc": strat["desc"],
                "metrics": metrics,
                "daily": day_results,
            }

            if metrics:
                print(f"  -> P&L: ${metrics.get('total_pnl', 0):+,.0f} | "
                      f"Trades: {metrics.get('total_trades', 0)} | "
                      f"Sharpe: {metrics.get('sharpe', 0):.2f} | "
                      f"Sortino: {metrics.get('sortino', 0):.2f} | "
                      f"Win%: {metrics.get('win_pct', 0)*100:.0f}%")
            else:
                print(f"  -> No results (no fills?)")

    # ── Step 4: Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"STRATEGY SWEEP RESULTS — {args.model.upper()} {args.horizon}")
    print(f"{'='*70}")
    print(f"{'Strategy':<30} {'Thresh':<10} {'P&L':>10} {'Trades':>8} {'Sharpe':>8} {'Sortino':>8} {'Win%':>6}")
    print("-" * 90)

    # Sort by Sharpe
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["metrics"].get("sharpe", -999),
        reverse=True
    )

    for combo_key, data in sorted_results:
        m = data["metrics"]
        if not m:
            continue
        print(f"{data['strategy']:<30} {data['threshold']:<10} "
              f"${m.get('total_pnl', 0):>+9,.0f} {m.get('total_trades', 0):>8} "
              f"{m.get('sharpe', 0):>8.2f} {m.get('sortino', 0):>8.2f} "
              f"{m.get('win_pct', 0)*100:>5.0f}%")

    # Save full results
    results_path = output_dir / "sweep_results.json"
    # Convert for JSON serialization
    serializable = {}
    for k, v in all_results.items():
        m = v["metrics"]
        serializable[k] = {
            "strategy": v["strategy"],
            "threshold": v["threshold"],
            "desc": v["desc"],
            "metrics": {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                       for kk, vv in m.items()} if m else {},
            "n_days": len(v["daily"]),
        }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    # Top 3 recommendation
    if sorted_results:
        print(f"\n{'='*70}")
        print("TOP 3 STRATEGIES FOR LIVE TESTING:")
        print(f"{'='*70}")
        for i, (combo_key, data) in enumerate(sorted_results[:3]):
            m = data["metrics"]
            print(f"\n  #{i+1}: {data['strategy']} (threshold: {data['threshold']})")
            print(f"      {data['desc']}")
            print(f"      P&L: ${m.get('total_pnl', 0):+,.0f} | "
                  f"Sharpe: {m.get('sharpe', 0):.2f} | "
                  f"Sortino: {m.get('sortino', 0):.2f} | "
                  f"Trades/day: {m.get('total_trades', 0)/max(m.get('n_days', 1), 1):.0f}")


if __name__ == "__main__":
    main()
