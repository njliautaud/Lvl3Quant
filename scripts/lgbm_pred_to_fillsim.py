#!/usr/bin/env python3
"""
LGBM v2 Walk-Forward Predictions → Rust Fill Simulator Bridge
==============================================================
Reconstructs fold boundaries from the sorted event NPZ file list (same
sliding-window logic as train_vol_lgbm_v2.py), maps each prediction back
to its source event timestamp, converts to RTH 100ms bar indices, writes
per-day prediction NPZs, then runs fill_sim_cli on every day that has
both a prediction file and a raw MBO file.

Usage:
    python3 scripts/lgbm_pred_to_fillsim.py [--folds 0-6] [--window 500] [--stride 250]
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
RESULTS_DIR = Path("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/vol_lgbm_v2")
RAW_MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
FILL_SIM    = Path("/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli")
OUTPUT_DIR  = Path("/home/jupiter/Lvl3Quant/alpha_discovery/deep_models/results/vol_lgbm_v2/fillsim")

# ── RTH constants (must match rust_cache_builder/src/rth.rs) ─────────────────
BAR_NS         = 100_000_000            # 100ms per bar
RTH_START_MIN  = 9 * 60 + 30           # 9:30 AM ET
RTH_END_MIN    = 16 * 60               # 4:00 PM ET
RTH_BARS       = int((RTH_END_MIN - RTH_START_MIN) * 60 * 10)  # 234,000
DST_END_2025_NS = 1_762_056_000_000_000_000


def et_offset_hours(ts_ns: int) -> int:
    """EDT (-4) before DST end, EST (-5) after."""
    return -4 if ts_ns < DST_END_2025_NS else -5


def rth_start_ns_for_day(ts_ns: int) -> int:
    """RTH start timestamp (ns) for the trading day containing ts_ns."""
    ts_sec = ts_ns // 1_000_000_000
    offset = et_offset_hours(ts_ns)
    et_sec = ts_sec + offset * 3600
    day_start_et = (et_sec // 86400) * 86400
    rth_start_et = day_start_et + RTH_START_MIN * 60
    return (rth_start_et - offset * 3600) * 1_000_000_000


def ts_to_rth_bar(ts_ns: int) -> int:
    """Convert nanosecond UTC timestamp to RTH bar index. Returns -1 if outside RTH."""
    ts_sec = ts_ns // 1_000_000_000
    et_sec = ts_sec + et_offset_hours(ts_ns) * 3600
    secs_in_day = et_sec % 86400
    mins_in_day = secs_in_day / 60.0
    if mins_in_day < RTH_START_MIN or mins_in_day >= RTH_END_MIN:
        return -1
    elapsed_ns = ts_ns - rth_start_ns_for_day(ts_ns)
    return int(elapsed_ns // BAR_NS)


def extract_date(fname: str) -> str:
    """Extract YYYYMMDD from '20250714_mbo_events.npz'."""
    return fname.split("_")[0]


def get_valid_files():
    """Return sorted list of event NPZ files that have non-all-NaN labels_10s."""
    files = sorted(DATA_DIR.glob("*.npz"))
    valid = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "labels_10s" in d and not np.all(np.isnan(d["labels_10s"])):
            valid.append(f)
    return valid


def compute_fold_boundaries(n_files, train_days, test_days, n_folds):
    """Replicate the sliding-window fold logic from train_vol_lgbm_v2.py."""
    folds = []
    min_train = max(train_days, 30)
    test_start = min_train
    while test_start + test_days <= n_files and len(folds) < n_folds:
        train_start = max(0, test_start - train_days)
        test_idx = list(range(test_start, test_start + test_days))
        folds.append((len(folds), train_start, test_idx))
        test_start += test_days
    return folds


def rebuild_sample_index_for_fold(files, window, stride):
    """Rebuild the per-sample mapping: list of (file_idx_in_fold, event_end_idx).

    Mirrors build_xy() logic: iterate files, iterate windows, filter NaN labels_10s.
    """
    index = []
    for fi, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        ev = d["features"] if "features" in d else d["events"]
        labs = d["labels_10s"].astype(np.float32)
        n_ev = len(ev)
        if n_ev < window:
            continue
        starts = np.arange(0, n_ev - window + 1, stride, dtype=np.int32)
        label_idxs = starts + window - 1
        valid = (label_idxs < len(labs)) & ~np.isnan(labs[label_idxs])
        for li in label_idxs[valid]:
            index.append((fi, int(li)))
    return index


def main():
    parser = argparse.ArgumentParser(description="LGBM v2 preds → fill sim bridge")
    parser.add_argument("--window", type=int, default=500, help="Event window size")
    parser.add_argument("--stride", type=int, default=250, help="Window stride")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument("--n-folds", type=int, default=9)
    parser.add_argument("--folds", default="0-6", help="Fold range to process, e.g. '0-6'")
    parser.add_argument("--signal-threshold", type=float, default=0.5)
    parser.add_argument("--hold-ms", type=int, default=10000)
    parser.add_argument("--latency-ms", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "per_day_preds"
    sim_dir = output_dir / "sim_results"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Parse fold range
    if "-" in args.folds:
        lo, hi = args.folds.split("-")
        fold_range = range(int(lo), int(hi) + 1)
    else:
        fold_range = [int(x) for x in args.folds.split(",")]

    print(f"Config: W={args.window} S={args.stride} TRAIN={args.train_days} TEST={args.test_days}")
    print(f"Processing folds: {list(fold_range)}")

    # Load valid files
    print("Loading valid event NPZ file list...")
    valid_files = get_valid_files()
    print(f"  {len(valid_files)} valid files: {valid_files[0].name} .. {valid_files[-1].name}")

    # Compute fold boundaries
    folds = compute_fold_boundaries(
        len(valid_files), args.train_days, args.test_days, args.n_folds
    )
    print(f"  {len(folds)} folds computed")

    # ── Step 1: Convert predictions to per-day bar-indexed NPZs ──────────────
    all_day_preds = defaultdict(list)  # date_str -> [(bar_idx, pred_value)]

    for fold_idx, train_start, test_idx in folds:
        if fold_idx not in fold_range:
            continue

        pred_path = RESULTS_DIR / f"fold{fold_idx:02d}_preds.npz"
        if not pred_path.exists():
            print(f"  Fold {fold_idx}: {pred_path.name} not found, skipping")
            continue

        pred_data = np.load(pred_path)
        preds = pred_data["preds"]
        oot_files = [valid_files[i] for i in test_idx]

        print(f"\n  Fold {fold_idx:02d}: {len(preds)} predictions, "
              f"test days {oot_files[0].name}..{oot_files[-1].name}")

        # Rebuild sample index
        sample_index = rebuild_sample_index_for_fold(
            oot_files, args.window, args.stride
        )
        if len(sample_index) != len(preds):
            print(f"    WARNING: sample_index={len(sample_index)} != preds={len(preds)}, SKIPPING")
            continue
        print(f"    Sample index rebuilt: {len(sample_index)} entries (matched)")

        # Load timestamps per OOT file (lazy cache)
        ts_cache = {}
        for pred_i, (fi, event_idx) in enumerate(sample_index):
            if fi not in ts_cache:
                day_data = np.load(oot_files[fi], allow_pickle=True)
                ts_cache[fi] = day_data["timestamps"]

            ts_ns = int(ts_cache[fi][event_idx])
            bar_idx = ts_to_rth_bar(ts_ns)
            if bar_idx < 0 or bar_idx >= RTH_BARS:
                continue

            pred_val = float(preds[pred_i])
            if not np.isnan(pred_val):
                date_str = extract_date(oot_files[fi].name)
                all_day_preds[date_str].append((bar_idx, pred_val))

        ts_cache.clear()

    # Write per-day NPZs
    print(f"\n{'='*60}")
    print(f"Writing per-day prediction NPZs...")
    days_written = []
    for date_str in sorted(all_day_preds.keys()):
        pairs = all_day_preds[date_str]
        bar_sums = np.zeros(RTH_BARS, dtype=np.float64)
        bar_counts = np.zeros(RTH_BARS, dtype=np.int32)
        for bar_idx, pred_val in pairs:
            bar_sums[bar_idx] += pred_val
            bar_counts[bar_idx] += 1
        mask = bar_counts > 0
        preds_out = np.zeros(RTH_BARS, dtype=np.float32)
        preds_out[mask] = (bar_sums[mask] / bar_counts[mask]).astype(np.float32)

        formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        out_path = pred_dir / f"{formatted}_preds.npz"
        np.savez_compressed(str(out_path), predictions=preds_out)
        n_active = int(mask.sum())
        print(f"  {formatted}: {n_active} active bars ({n_active/RTH_BARS*100:.1f}%) "
              f"from {len(pairs)} predictions")
        days_written.append((date_str, formatted, out_path))

    print(f"\n{len(days_written)} per-day files written to {pred_dir}")

    # ── Step 2: Run fill sim for each day ────────────────────────────────────
    print(f"\n{'='*60}")
    print("Running fill simulator...")

    if not FILL_SIM.exists():
        print(f"ERROR: fill_sim_cli not found at {FILL_SIM}", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for date_str, formatted, pred_path in days_written:
        mbo_path = RAW_MBO_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
        if not mbo_path.exists():
            print(f"  {formatted}: no raw MBO file, skipping")
            continue

        result_path = sim_dir / f"{formatted}_result.json"
        cmd = [
            str(FILL_SIM),
            "--mbo-file", str(mbo_path),
            "--predictions", str(pred_path),
            "--output", str(result_path),
            "--signal-threshold", str(args.signal_threshold),
            "--hold-ms", str(args.hold_ms),
            "--chase-entry",
            "--chase-max-ticks", "2",
            "--latency-ms", str(args.latency_ms),
            "--prime-hours",
        ]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                print(f"  {formatted}: fill_sim FAILED (rc={r.returncode})")
                if r.stderr:
                    print(f"    stderr: {r.stderr[:200]}")
                continue
        except subprocess.TimeoutExpired:
            print(f"  {formatted}: fill_sim TIMEOUT (300s)")
            continue

        if not result_path.exists():
            print(f"  {formatted}: no result file produced")
            continue

        with open(result_path) as f:
            res = json.load(f)

        pnl = res.get("total_pnl_dollars", res.get("total_pnl", 0))
        trades = res.get("total_trades", 0)
        wr = res.get("win_rate", 0) * 100  # stored as fraction
        print(f"  {formatted}: P&L=${pnl:+.2f}  trades={trades}  WR={wr:.1f}%")
        all_results.append(res)

    # ── Step 3: Aggregate summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGGREGATE FILL SIM SUMMARY")
    print(f"{'='*60}")

    if not all_results:
        print("No fill sim results to aggregate.")
        return

    total_pnl = 0
    total_trades = 0
    total_wins = 0
    daily_pnls = []

    for res in all_results:
        pnl = res.get("total_pnl_dollars", res.get("total_pnl", 0))
        trades = res.get("total_trades", 0)
        wr = res.get("win_rate", 0)
        wins = int(round(wr * trades))
        total_pnl += pnl
        total_trades += trades
        total_wins += wins
        daily_pnls.append(pnl)

    daily_pnls = np.array(daily_pnls)
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    # Sortino ratio (annualized, assuming 252 trading days)
    if len(daily_pnls) > 1:
        mean_daily = daily_pnls.mean()
        downside = daily_pnls[daily_pnls < 0]
        downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-8
        sortino = (mean_daily / downside_std) * np.sqrt(252) if downside_std > 1e-8 else float("inf")
    else:
        sortino = float("nan")

    # Max drawdown
    cumulative = np.cumsum(daily_pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0

    # Sharpe for reference
    if len(daily_pnls) > 1 and daily_pnls.std() > 1e-8:
        sharpe = (daily_pnls.mean() / daily_pnls.std()) * np.sqrt(252)
    else:
        sharpe = float("nan")

    print(f"  Days simulated:  {len(all_results)}")
    print(f"  Total P&L:       ${total_pnl:+,.2f}")
    print(f"  Total trades:    {total_trades:,}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Avg daily P&L:   ${daily_pnls.mean():+,.2f}")
    print(f"  Daily P&L std:   ${daily_pnls.std():,.2f}")
    print(f"  Sharpe (ann):    {sharpe:.2f}")
    print(f"  Sortino (ann):   {sortino:.2f}")
    print(f"  Max drawdown:    ${max_dd:+,.2f}")
    print(f"  Best day:        ${daily_pnls.max():+,.2f}")
    print(f"  Worst day:       ${daily_pnls.min():+,.2f}")


if __name__ == "__main__":
    main()
