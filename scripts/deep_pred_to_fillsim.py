#!/usr/bin/env python3
"""
Deep Model (CNN/Mamba) Predictions -> Rust Fill Simulator Bridge
================================================================
Converts walk-forward OOT prediction NPZs from deep models (CNN1D, Mamba)
into per-day bar-indexed predictions, then runs fill_sim_cli.

Key differences from lgbm_pred_to_fillsim.py:
- Predictions are (N, 3) with horizons [1s, 5s, 10s] instead of 1D
- Each fold has exactly 1 OOT file (test_days=1)
- oot_files key tells us which source file was used
- We use the 10s predictions for fill sim signal (configurable)

Usage:
    python3 scripts/deep_pred_to_fillsim.py \
        --model-dir output/mamba_v4_sliding60d_raw6_v2 \
        --model-name mamba_v4 \
        --horizon 10s \
        --window 1000 --stride 500

    python3 scripts/deep_pred_to_fillsim.py \
        --model-dir alpha_discovery/deep_models/results/cnn1d_256ch_8L_sliding \
        --model-name cnn1d_256ch \
        --horizon 10s \
        --window 500 --stride 250
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/home/jupiter/Lvl3Quant")
DATA_DIR    = BASE_DIR / "data/processed/mbo_events"
RAW_MBO_DIR = BASE_DIR / "data/raw/mbo"
FILL_SIM    = BASE_DIR / "rust_cache_builder/target/release/fill_sim_cli"

# ── RTH constants ────────────────────────────────────────────────────────────
BAR_NS        = 100_000_000           # 100ms per bar
RTH_START_MIN = 9 * 60 + 30          # 9:30 AM ET
RTH_END_MIN   = 16 * 60              # 4:00 PM ET
RTH_BARS      = int((RTH_END_MIN - RTH_START_MIN) * 60 * 10)  # 234,000

# DST transitions (approximate — add more as needed)
DST_END_2025_NS  = 1_762_056_000_000_000_000  # Nov 2 2025
DST_START_2026_NS = 1_773_562_800_000_000_000  # Mar 8 2026


def et_offset_hours(ts_ns: int) -> int:
    """EDT (-4) during daylight saving, EST (-5) during standard."""
    if ts_ns >= DST_START_2026_NS:
        return -4  # EDT after Mar 2026
    if ts_ns < DST_END_2025_NS:
        return -4  # EDT before Nov 2025
    return -5      # EST Nov 2025 - Mar 2026


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


def extract_date_from_path(fpath: str) -> str:
    """Extract YYYYMMDD from paths like '.../20260302_mbo_events.npz'.
    Handles both Unix and Windows paths (backslashes)."""
    import re
    # Find YYYYMMDD pattern in the path
    m = re.search(r'(\d{8})_mbo_events', str(fpath))
    if m:
        return m.group(1)
    # Fallback: take last path component
    fname = str(fpath).replace("\\", "/").split("/")[-1]
    return fname.split("_")[0]


def find_raw_mbo_for_date(date_str: str) -> Path | None:
    """Find raw MBO .dbn or .dbn.zst file for a given date."""
    for ext in [".mbo.dbn.zst", ".mbo.dbn"]:
        p = RAW_MBO_DIR / f"glbx-mdp3-{date_str}{ext}"
        if p.exists():
            return p
    # Also check YYYYMMDD format directly
    for f in RAW_MBO_DIR.glob(f"*{date_str}*"):
        if f.suffix in ['.zst', '.dbn']:
            return f
    return None


def load_timestamps_for_file(oot_path: str) -> np.ndarray:
    """Load event timestamps from the OOT source file.

    Handles path remapping: predictions may reference remote paths
    (Neptune/Razer) but we load from Jupiter's local data dir.
    """
    date_str = extract_date_from_path(oot_path)
    # Try local Jupiter paths
    local_path = DATA_DIR / f"{date_str}_mbo_events.npz"
    if local_path.exists():
        d = np.load(local_path, allow_pickle=True)
        return d["timestamps"]

    # Try feat15 directory
    feat15_path = BASE_DIR / f"data/processed/mbo_events_feat15/{date_str}_mbo_events.npz"
    if feat15_path.exists():
        d = np.load(feat15_path, allow_pickle=True)
        return d["timestamps"]

    raise FileNotFoundError(f"Cannot find event file for date {date_str}")


def convert_fold_to_bars(pred_file: Path, window: int, stride: int, horizon_idx: int):
    """Convert a fold's predictions to (date_str, bar_idx, pred_value) tuples."""
    f = np.load(pred_file, allow_pickle=True)
    preds = f["predictions"]  # (N, 3) for [1s, 5s, 10s]
    oot_files = f["oot_files"]

    results = []
    sample_offset = 0

    for oot_path in oot_files:
        oot_path = str(oot_path)
        date_str = extract_date_from_path(oot_path)

        # Load timestamps
        try:
            timestamps = load_timestamps_for_file(oot_path)
        except FileNotFoundError as e:
            print(f"    WARNING: {e}, skipping")
            continue

        n_events = len(timestamps)
        if n_events < window:
            print(f"    WARNING: {date_str} has {n_events} events < window {window}, skipping")
            continue

        # Reconstruct window end indices (same logic as training)
        starts = np.arange(0, n_events - window + 1, stride, dtype=np.int64)
        end_indices = starts + window - 1
        n_windows = len(end_indices)

        # Extract predictions for this file's windows
        file_preds = preds[sample_offset:sample_offset + n_windows, horizon_idx]

        for i, (end_idx, pred_val) in enumerate(zip(end_indices, file_preds)):
            if np.isnan(pred_val):
                continue
            ts_ns = int(timestamps[end_idx])
            bar_idx = ts_to_rth_bar(ts_ns)
            if 0 <= bar_idx < RTH_BARS:
                results.append((date_str, bar_idx, float(pred_val)))

        sample_offset += n_windows

    # Check alignment
    if sample_offset != len(preds):
        print(f"    WARNING: Processed {sample_offset} windows but have {len(preds)} predictions")
        # Try without strict alignment (some windows may have been filtered during training)

    return results


def write_per_day_preds(all_bars, output_dir: Path):
    """Write per-day bar-indexed prediction NPZs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by date
    by_date = defaultdict(list)
    for date_str, bar_idx, pred_val in all_bars:
        by_date[date_str].append((bar_idx, pred_val))

    written = []
    for date_str in sorted(by_date.keys()):
        bars = by_date[date_str]
        # Aggregate: average predictions for same bar
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
        print(f"    {date_str}: {mask.sum()} bars with predictions")

    return written


def run_fill_sim(date_str, pred_path, sim_dir, args):
    """Run Rust fill_sim_cli for one day."""
    raw_mbo = find_raw_mbo_for_date(date_str)
    if raw_mbo is None:
        print(f"    {date_str}: No raw MBO file found, skipping sim")
        return None

    out_path = sim_dir / f"{date_str}_result.json"

    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(raw_mbo),
        "--predictions", str(pred_path),
        "--output", str(out_path),
        "--signal-threshold", str(args.signal_threshold),
        "--hold-ms", str(args.hold_ms),
        "--latency-ms", str(args.latency_ms),
        "--chase-entry",
        "--chase-max-ticks", "2",
        "--chase-max-reprices", "5",
        "--prime-hours",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"    {date_str}: fill_sim FAILED: {result.stderr[:200]}")
            return None

        if out_path.exists():
            with open(out_path) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        print(f"    {date_str}: fill_sim TIMEOUT")
    except Exception as e:
        print(f"    {date_str}: fill_sim ERROR: {e}")

    return None


def main():
    parser = argparse.ArgumentParser(description="Deep model preds -> fill sim bridge")
    parser.add_argument("--model-dir", required=True, help="Path to model results dir (relative to Lvl3Quant)")
    parser.add_argument("--model-name", required=True, help="Short name for output dirs")
    parser.add_argument("--horizon", default="10s", choices=["1s", "5s", "10s"], help="Which horizon to use for signals")
    parser.add_argument("--window", type=int, required=True, help="Event window size used in training")
    parser.add_argument("--stride", type=int, required=True, help="Window stride used in training")
    parser.add_argument("--folds", default="0-1", help="Fold range, e.g. '0-1' or '0,1,2'")
    parser.add_argument("--signal-threshold", type=float, default=0.3, help="Z-score threshold for signals")
    parser.add_argument("--hold-ms", type=int, default=10000, help="Hold time in ms")
    parser.add_argument("--latency-ms", type=int, default=10, help="Order submission latency")
    parser.add_argument("--skip-sim", action="store_true", help="Only convert, don't run fill sim")
    args = parser.parse_args()

    horizon_map = {"1s": 0, "5s": 1, "10s": 2}
    horizon_idx = horizon_map[args.horizon]

    model_dir = BASE_DIR / args.model_dir
    output_dir = BASE_DIR / f"data/processed/fillsim_{args.model_name}"
    pred_dir = output_dir / "per_day_preds"
    sim_dir = output_dir / "sim_results"
    pred_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Parse fold range
    if "-" in args.folds:
        lo, hi = args.folds.split("-")
        fold_range = list(range(int(lo), int(hi) + 1))
    else:
        fold_range = [int(x) for x in args.folds.split(",")]

    print(f"Model: {args.model_name}")
    print(f"Dir:   {model_dir}")
    print(f"Horizon: {args.horizon} (column {horizon_idx})")
    print(f"Window: {args.window}, Stride: {args.stride}")
    print(f"Folds: {fold_range}")
    print(f"Output: {output_dir}")
    print()

    # ── Step 1: Convert all fold predictions to bar-indexed format ────────────
    all_bars = []
    for fold_idx in fold_range:
        pred_file = model_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
        if not pred_file.exists():
            print(f"Fold {fold_idx}: {pred_file.name} not found, skipping")
            continue

        print(f"Fold {fold_idx}: Converting...")
        bars = convert_fold_to_bars(pred_file, args.window, args.stride, horizon_idx)
        print(f"  -> {len(bars)} bar predictions")
        all_bars.extend(bars)

    if not all_bars:
        print("ERROR: No predictions converted. Check paths and fold numbers.")
        sys.exit(1)

    print(f"\nTotal: {len(all_bars)} bar predictions across all folds")

    # ── Step 2: Write per-day NPZs ───────────────────────────────────────────
    print("\nWriting per-day prediction files...")
    written = write_per_day_preds(all_bars, pred_dir)

    if args.skip_sim:
        print("\n--skip-sim specified, stopping after conversion.")
        return

    # ── Step 3: Run fill sim ─────────────────────────────────────────────────
    if not FILL_SIM.exists():
        print(f"\nERROR: fill_sim_cli not found at {FILL_SIM}")
        print("Run: cd rust_cache_builder && cargo build --release --bin fill_sim_cli")
        sys.exit(1)

    print(f"\nRunning fill sim (threshold={args.signal_threshold}, hold={args.hold_ms}ms, latency={args.latency_ms}ms)...")

    total_pnl = 0.0
    total_trades = 0
    total_signals = 0
    day_results = []

    for date_str, n_bars in written:
        pred_path = pred_dir / f"{date_str}_preds.npz"
        print(f"\n  Simulating {date_str} ({n_bars} signal bars)...")
        result = run_fill_sim(date_str, pred_path, sim_dir, args)

        if result and "summary" in result:
            s = result["summary"]
            pnl = s.get("total_pnl_dollars", s.get("total_pnl", 0))
            trades = s.get("total_trades", s.get("trades", 0))
            signals = s.get("total_signals", 0)
            win_rate = s.get("win_rate", 0) * 100 if s.get("win_rate", 0) <= 1 else s.get("win_rate", 0)
            fill_rate = trades / max(signals, 1) * 100

            total_pnl += pnl
            total_trades += trades
            total_signals += signals

            day_results.append({
                "date": date_str, "pnl": pnl, "trades": trades,
                "signals": signals, "win_rate": win_rate
            })

            print(f"    P&L: ${pnl:+.0f} | Trades: {trades} | Signals: {signals} | "
                  f"Fill: {fill_rate:.0f}% | Win: {win_rate:.1f}%")

    # ── Summary ──────────────────────────────────────────────────────────────
    n_days = len(day_results)
    print(f"\n{'='*60}")
    print(f"FILL SIM SUMMARY: {args.model_name}")
    print(f"{'='*60}")
    print(f"Days simulated:  {n_days}")
    print(f"Total P&L:       ${total_pnl:+,.0f}")
    print(f"Avg daily P&L:   ${total_pnl/max(n_days,1):+,.0f}")
    print(f"Total trades:    {total_trades}")
    print(f"Total signals:   {total_signals}")
    print(f"Fill rate:       {total_trades/max(total_signals,1)*100:.1f}%")

    if day_results:
        pnls = [d["pnl"] for d in day_results]
        pos_days = sum(1 for p in pnls if p > 0)
        print(f"Positive days:   {pos_days}/{n_days}")
        if len(pnls) > 1:
            sharpe = np.mean(pnls) / max(np.std(pnls), 0.01) * np.sqrt(252)
            neg_pnls = [p for p in pnls if p < 0]
            sortino = np.mean(pnls) / max(np.std(neg_pnls) if neg_pnls else 0.01, 0.01) * np.sqrt(252)
            print(f"Sharpe:          {sharpe:.2f}")
            print(f"Sortino:         {sortino:.2f}")

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model_name,
            "horizon": args.horizon,
            "config": {
                "signal_threshold": args.signal_threshold,
                "hold_ms": args.hold_ms,
                "latency_ms": args.latency_ms,
                "window": args.window,
                "stride": args.stride,
            },
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "total_signals": total_signals,
            "n_days": n_days,
            "day_results": day_results,
        }, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
