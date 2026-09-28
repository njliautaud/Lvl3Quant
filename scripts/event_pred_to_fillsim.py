#!/usr/bin/env python3
"""
Event-Driven Model Predictions → Rust Fill Simulator Bridge
============================================================
Converts per-fold event prediction NPZs (from train_event_cnn_1d.py,
train_event_mamba.py, etc.) into per-day bar-indexed NPZs that the
Rust fill_sim_cli can consume.

The fill sim indexes predictions by rth_bar_index (100ms bars during
RTH 9:30-16:00 ET). This script:
  1. Loads fold prediction NPZs from a results directory
  2. Rebuilds the sample_index to map each prediction → (day, event_idx)
  3. Reads timestamps from source event NPZs
  4. Converts event timestamps → RTH bar indices
  5. Writes per-day NPZs with 'predictions' key (one per 100ms bar)

Usage:
    python3 scripts/event_pred_to_fillsim.py --results-dir <path> --output-dir <path>
    python3 scripts/event_pred_to_fillsim.py --results-dir <path> --output-dir <path> \
        --horizon 10s --window-size 500 --stride 250
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# RTH constants (must match rust_cache_builder/src/rth.rs)
BAR_NS = 100_000_000           # 100ms per bar
RTH_START_MIN = 9 * 60 + 30    # 9:30 AM ET
RTH_END_MIN = 16 * 60          # 4:00 PM ET
RTH_BARS = int((RTH_END_MIN - RTH_START_MIN) * 60 * 10)  # 234,000
DST_END_2025_NS = 1_762_056_000_000_000_000


def et_offset_hours(ts_ns: int) -> int:
    return -4 if ts_ns < DST_END_2025_NS else -5


def ts_to_rth_bar(ts_ns: int) -> int:
    """Convert nanosecond UTC timestamp to RTH bar index (0-based). Returns -1 if outside RTH."""
    ts_sec = ts_ns // 1_000_000_000
    et_sec = ts_sec + et_offset_hours(ts_ns) * 3600
    secs_in_day = et_sec % 86400
    mins_in_day = secs_in_day / 60.0
    if mins_in_day < RTH_START_MIN or mins_in_day >= RTH_END_MIN:
        return -1
    elapsed_ns_from_rth_start = ts_ns - rth_start_ns_for_day(ts_ns)
    return int(elapsed_ns_from_rth_start // BAR_NS)


def rth_start_ns_for_day(ts_ns: int) -> int:
    """Get the RTH start timestamp (ns) for the trading day containing ts_ns."""
    ts_sec = ts_ns // 1_000_000_000
    offset = et_offset_hours(ts_ns)
    et_sec = ts_sec + offset * 3600
    day_start_et = (et_sec // 86400) * 86400
    rth_start_et = day_start_et + RTH_START_MIN * 60
    return (rth_start_et - offset * 3600) * 1_000_000_000


def extract_date_from_filename(fname: str) -> str:
    """Extract YYYYMMDD date from event NPZ filename like '20250714_mbo_events.npz'."""
    return fname.split("_")[0]


def rebuild_sample_index(npz_files, window_size, stride, horizons):
    """Rebuild the sample_index exactly as MboEventDataset._build_index does.

    Returns list of (day_idx, event_end_idx) for each valid sample.
    """
    index = []
    for day_idx, f in enumerate(npz_files):
        data = np.load(f, allow_pickle=True)
        n_events = len(data["events"])
        day_labels = {h: data[f"labels_{h}"] for h in horizons}
        for start in range(0, n_events - window_size + 1, stride):
            end = start + window_size
            label_idx = end - 1
            if all(not np.isnan(day_labels[h][label_idx]) for h in horizons):
                index.append((day_idx, label_idx))
        del data
    return index


def main():
    parser = argparse.ArgumentParser(description="Event predictions → fill sim per-day NPZ")
    parser.add_argument("--results-dir", required=True, help="Directory with fold_XX_oot_predictions.npz files")
    parser.add_argument("--output-dir", required=True, help="Output directory for per-day NPZ files")
    parser.add_argument("--horizon", default="10s", help="Which horizon column to use (default: 10s)")
    parser.add_argument("--window-size", type=int, default=500, help="Event window size (default: 500)")
    parser.add_argument("--stride", type=int, default=None, help="Stride (default: window_size // 2)")
    parser.add_argument("--horizons", default="1s,5s,10s", help="Comma-separated horizons for label validation")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_size = args.window_size
    stride = args.stride or window_size // 2
    horizons = args.horizons.split(",")
    horizon_idx = horizons.index(args.horizon)

    # Find fold prediction files
    fold_files = sorted(results_dir.glob("fold_*_oot_predictions.npz"))
    if not fold_files:
        print(f"ERROR: No fold_*_oot_predictions.npz found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(fold_files)} fold prediction files in {results_dir}")

    total_days = 0
    for fi, fold_file in enumerate(fold_files):
        data = np.load(fold_file, allow_pickle=True)
        predictions = data["predictions"]  # (N, n_horizons)
        oot_files_raw = data["oot_files"]

        # Resolve OOT file paths
        oot_files = [Path(str(f)) for f in oot_files_raw]
        missing = [f for f in oot_files if not f.exists()]
        if missing:
            print(f"  WARNING: {len(missing)} OOT files missing for fold {fi}, skipping")
            continue

        print(f"  Fold {fi:02d}: {len(predictions)} predictions, {len(oot_files)} OOT days")

        # Rebuild sample index to map prediction → (day_idx, event_idx)
        sample_index = rebuild_sample_index(oot_files, window_size, stride, horizons)
        if len(sample_index) != len(predictions):
            print(f"  WARNING: sample_index ({len(sample_index)}) != predictions ({len(predictions)}), skipping fold")
            continue

        # Group predictions by day and map to bar indices
        day_preds = defaultdict(list)  # day_idx -> [(bar_idx, pred_value)]
        ts_cache = {}  # day_idx -> timestamps array

        for pred_i, (day_idx, event_idx) in enumerate(sample_index):
            if day_idx not in ts_cache:
                day_data = np.load(oot_files[day_idx], allow_pickle=True)
                ts_cache[day_idx] = day_data["timestamps"]
                del day_data

            ts_ns = int(ts_cache[day_idx][event_idx])
            bar_idx = ts_to_rth_bar(ts_ns)
            if bar_idx < 0 or bar_idx >= RTH_BARS:
                continue

            pred_val = predictions[pred_i, horizon_idx] if predictions.ndim == 2 else predictions[pred_i]
            if not np.isnan(pred_val):
                day_preds[day_idx].append((bar_idx, float(pred_val)))

        # Write per-day NPZ files
        for day_idx, bar_pred_pairs in sorted(day_preds.items()):
            date_str = extract_date_from_filename(oot_files[day_idx].name)
            formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            # Aggregate: average predictions landing on same bar
            bar_sums = np.zeros(RTH_BARS, dtype=np.float64)
            bar_counts = np.zeros(RTH_BARS, dtype=np.int32)
            for bar_idx, pred_val in bar_pred_pairs:
                bar_sums[bar_idx] += pred_val
                bar_counts[bar_idx] += 1
            mask = bar_counts > 0
            preds_out = np.zeros(RTH_BARS, dtype=np.float32)
            preds_out[mask] = (bar_sums[mask] / bar_counts[mask]).astype(np.float32)

            out_path = output_dir / f"{formatted}_preds.npz"
            np.savez_compressed(str(out_path), predictions=preds_out)
            n_active = int(mask.sum())
            total_days += 1
            print(f"    {formatted}: {n_active} active bars ({n_active/RTH_BARS*100:.1f}%) from {len(bar_pred_pairs)} predictions")

        # Free cached timestamps
        ts_cache.clear()
        del data

    print(f"\nDone: {total_days} per-day NPZ files written to {output_dir}")
    print(f"Ready for: fill_sim_cli --mbo-file <day.mbo.dbn.zst> --predictions <day_preds.npz> --output <out.json>")


if __name__ == "__main__":
    main()
