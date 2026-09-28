#!/usr/bin/env python3
"""
prebatch_event_tensors.py — Pre-process raw MBO event NPZ files into ready-to-train PyTorch tensor batches.

Runs on Saturn (CPU-only, 32GB RAM, 48 cores). Output is rsynced to GPU nodes
to eliminate the data loading bottleneck during training.

Usage:
    python prebatch_event_tensors.py --window_size 200 --num_workers 32
    python prebatch_event_tensors.py --window_size 500 --input_dir /path/to/npz --output_dir /path/to/out

Input:  NPZ files with events(N,6), labels_1s/5s/10s/30s(N,), timestamps(N,)
Output: .npz files per day + manifest.json (numpy format — no PyTorch dependency)
       GPU nodes convert to tensors on load via torch.from_numpy()
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = "/home/saturn/Lvl3Quant/data/processed/mbo_events"
DEFAULT_OUTPUT_BASE = "/home/saturn/Lvl3Quant/data"


def process_single_file(args):
    """Process one NPZ file into a .pt file of windowed tensors.

    Args is a tuple: (npz_path, output_path, window_size)
    Returns dict with file metadata or None on failure.
    """
    npz_path, output_path, window_size = args
    fname = os.path.basename(npz_path)
    date_str = fname.replace("_mbo_events.npz", "")

    try:
        data = np.load(npz_path)

        events = data["events"]  # (N, 6) float32
        labels_1s = data["labels_1s"]  # (N,) float32
        labels_5s = data["labels_5s"]
        labels_10s = data["labels_10s"]
        labels_30s = data["labels_30s"]
        timestamps = data["timestamps"]  # (N,) int64

        n_events = events.shape[0]

        if n_events < window_size:
            logger.warning(
                f"  {fname}: only {n_events} events, need {window_size}. Skipping."
            )
            return None

        # Non-overlapping windows (stride = window_size)
        num_windows = n_events // window_size
        usable = num_windows * window_size

        # Reshape into (num_windows, window_size, 6) — zero-copy via reshape
        X = events[:usable].reshape(num_windows, window_size, 6)

        # For labels, take the LAST event's label in each window (prediction target)
        label_indices = np.arange(window_size - 1, usable, window_size)
        lbl_1s = labels_1s[label_indices]
        lbl_5s = labels_5s[label_indices]
        lbl_10s = labels_10s[label_indices]
        lbl_30s = labels_30s[label_indices]

        # Timestamp of the last event in each window
        ts = timestamps[label_indices]

        # Save as numpy NPZ (no PyTorch dependency on Saturn)
        # GPU nodes convert to tensors on load: torch.from_numpy(d['X'])
        np.savez_compressed(
            output_path,
            X=X,                # (num_windows, W, 6) float32
            labels_1s=lbl_1s,   # (num_windows,) float32
            labels_5s=lbl_5s,
            labels_10s=lbl_10s,
            labels_30s=lbl_30s,
            timestamps=ts,      # (num_windows,) int64
        )

        # Explicitly free memory
        del data, events, labels_1s, labels_5s, labels_10s, labels_30s, timestamps
        del X, lbl_1s, lbl_5s, lbl_10s, lbl_30s, ts

        logger.info(
            f"  {fname} -> {num_windows} windows, "
            f"discarded {n_events - usable} trailing events"
        )

        return {
            "date": date_str,
            "source_file": fname,
            "output_file": os.path.basename(output_path),
            "num_events": int(n_events),
            "num_windows": int(num_windows),
            "discarded_events": int(n_events - usable),
        }

    except Exception as e:
        logger.error(f"  FAILED {fname}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pre-batch MBO event NPZ files into PyTorch tensor batches"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=200,
        choices=[200, 500, 1000],
        help="Window size for batching events (default: 200)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of parallel workers (default: 32)",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory with NPZ files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {DEFAULT_OUTPUT_BASE}/prebatched_w{window_size}/)",
    )
    args = parser.parse_args()

    window_size = args.window_size
    input_dir = Path(args.input_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(DEFAULT_OUTPUT_BASE) / f"prebatched_w{window_size}"

    # Validate input
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    npz_files = sorted(input_dir.glob("*_mbo_events.npz"))
    if not npz_files:
        logger.error(f"No *_mbo_events.npz files found in {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"{'=' * 60}")
    logger.info(f"Prebatch Event Tensors")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Input dir:    {input_dir}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info(f"  Window size:  {window_size}")
    logger.info(f"  NPZ files:    {len(npz_files)}")
    logger.info(f"  Workers:      {args.num_workers}")
    logger.info(f"{'=' * 60}")

    # Build task list
    tasks = []
    for npz_path in npz_files:
        date_str = npz_path.stem.replace("_mbo_events", "")
        out_name = f"{date_str}_w{window_size}.npz"
        out_path = str(output_dir / out_name)
        tasks.append((str(npz_path), out_path, window_size))

    # Process with multiprocessing pool
    t0 = time.time()
    results = []

    if args.num_workers <= 1:
        # Sequential mode for debugging
        for task in tasks:
            result = process_single_file(task)
            if result:
                results.append(result)
    else:
        with Pool(processes=args.num_workers) as pool:
            for result in pool.imap_unordered(process_single_file, tasks):
                if result:
                    results.append(result)

    elapsed = time.time() - t0

    # Sort results by date
    results.sort(key=lambda x: x["date"])

    # Compute stats
    total_windows = sum(r["num_windows"] for r in results)
    total_events = sum(r["num_events"] for r in results)
    total_discarded = sum(r["discarded_events"] for r in results)

    # Write manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "window_size": window_size,
        "stride": window_size,  # non-overlapping
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_files": len(results),
        "total_windows": total_windows,
        "total_events": total_events,
        "total_discarded_events": total_discarded,
        "processing_time_seconds": round(elapsed, 1),
        "files": results,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    logger.info(f"{'=' * 60}")
    logger.info(f"COMPLETE")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Files processed:  {len(results)}/{len(npz_files)}")
    logger.info(f"  Total windows:    {total_windows:,}")
    logger.info(f"  Total events:     {total_events:,}")
    logger.info(f"  Discarded events: {total_discarded:,}")
    logger.info(f"  Time elapsed:     {elapsed:.1f}s")
    logger.info(f"  Manifest:         {manifest_path}")
    logger.info(f"{'=' * 60}")

    # Estimate output size
    # Each window: (W * 6 * 4) bytes for X + (4 * 4) for labels + 8 for timestamp
    bytes_per_window = window_size * 6 * 4 + 4 * 4 + 8
    est_size_gb = (total_windows * bytes_per_window) / (1024**3)
    logger.info(f"  Est. total size:  ~{est_size_gb:.1f} GB")
    logger.info(f"")
    logger.info(f"Next step: rsync {output_dir}/ to GPU nodes")


if __name__ == "__main__":
    main()
