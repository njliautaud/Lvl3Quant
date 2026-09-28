"""
Pre-compute windowed tensors from MBO event NPZ files.

Eliminates the 30+ minute data loading bottleneck by:
1. Computing global feature stats (mean/std) across all files
2. Extracting all valid windows, normalizing, and saving as .pt files
3. Saving metadata and stats for instant training startup

Usage:
    python precompute_tensors.py                          # CNN defaults (w=500, s=250)
    python precompute_tensors.py --window-size 5000 --stride 2500  # Mamba config
    python precompute_tensors.py --max-days 10            # Quick test on 10 files
    python precompute_tensors.py --derived-features       # Include 5 derived features
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events"
)
DEFAULT_OUTPUT_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_tensors"
)

HORIZONS = ["1s", "5s", "10s"]
N_RAW_FEATURES = 6
N_DERIVED_FEATURES = 5

FEATURE_NAMES_RAW = [
    "time_delta_log", "event_type_id", "side_id",
    "price_rel_ticks", "qty_log", "spread_ticks",
]
FEATURE_NAMES_DERIVED = [
    "trade_intensity", "signed_volume", "price_accel",
    "spread_change", "qty_change",
]


def compute_derived_features(events: np.ndarray) -> np.ndarray:
    """Compute 5 orderflow derivatives from raw 6 features. No look-ahead."""
    n = len(events)
    derived = np.zeros((n, 5), dtype=np.float32)
    derived[:, 0] = np.exp(-events[:, 0])           # trade intensity
    derived[:, 1] = events[:, 2] * events[:, 4]     # signed volume
    derived[1:, 2] = np.diff(events[:, 3])           # price acceleration
    derived[1:, 3] = np.diff(events[:, 5])           # spread change
    derived[1:, 4] = np.diff(events[:, 4])           # qty change
    return derived


def load_npz(f: Path):
    """Load NPZ with retry on PermissionError (Windows file locking)."""
    for attempt in range(5):
        try:
            return np.load(f, allow_pickle=True)
        except PermissionError:
            if attempt < 4:
                logger.warning(f"PermissionError on {f.name}, retry {attempt+1}/5...")
                time.sleep(2)
            else:
                raise


def compute_global_stats(npz_files, use_derived):
    """Single-pass Welford-style mean/std across all files. One file in RAM at a time."""
    n_feat = N_RAW_FEATURES + (N_DERIVED_FEATURES if use_derived else 0)
    total_sum = np.zeros(n_feat, dtype=np.float64)
    total_sq = np.zeros(n_feat, dtype=np.float64)
    total_count = 0

    for i, f in enumerate(npz_files):
        data = load_npz(f)
        events = data["events"].astype(np.float64)
        if use_derived:
            derived = compute_derived_features(events.astype(np.float32)).astype(np.float64)
            events = np.concatenate([events, derived], axis=1)
        total_sum += events.sum(axis=0)
        total_sq += (events ** 2).sum(axis=0)
        total_count += len(events)
        del data, events
        if (i + 1) % 20 == 0:
            logger.info(f"  Stats pass: {i+1}/{len(npz_files)} files...")

    mean = (total_sum / total_count).astype(np.float32)
    var = (total_sq / total_count) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std


def process_file(f, mean, std, window_size, stride, use_derived):
    """Extract all valid normalized windows and labels from one NPZ file."""
    data = load_npz(f)
    events = data["events"].astype(np.float32)

    if use_derived:
        derived = compute_derived_features(events)
        events = np.concatenate([events, derived], axis=1)

    # Load labels
    labels = {}
    for h in HORIZONS:
        key = f"labels_{h}"
        if key in data:
            labels[h] = data[key].astype(np.float32)
    del data

    # Normalize
    events = (events - mean) / (std + 1e-8)

    n_events = len(events)
    windows = []
    label_arrays = {h: [] for h in HORIZONS}

    for start in range(0, n_events - window_size + 1, stride):
        end = start + window_size
        label_idx = end - 1

        # Check all horizon labels are valid (not NaN)
        all_valid = all(
            h in labels and not np.isnan(labels[h][label_idx])
            for h in HORIZONS
        )
        if not all_valid:
            continue

        windows.append(events[start:end])
        for h in HORIZONS:
            label_arrays[h].append(labels[h][label_idx])

    del events, labels

    if not windows:
        return None

    result = {
        "events": torch.from_numpy(np.stack(windows)),  # (N_windows, W, F)
    }
    for h in HORIZONS:
        result[f"labels_{h}"] = torch.tensor(label_arrays[h], dtype=torch.float32)

    return result


def main():
    parser = argparse.ArgumentParser(description="Pre-compute windowed tensors from MBO event NPZ files")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=None, help="Default: window_size // 2")
    parser.add_argument("--max-days", type=int, default=None, help="Limit number of files for testing")
    parser.add_argument("--derived-features", action="store_true", help="Include 5 derived orderflow features")
    args = parser.parse_args()

    stride = args.stride or args.window_size // 2
    use_derived = args.derived_features
    n_feat = N_RAW_FEATURES + (N_DERIVED_FEATURES if use_derived else 0)
    feature_names = FEATURE_NAMES_RAW + (FEATURE_NAMES_DERIVED if use_derived else [])

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if args.max_days:
        npz_files = npz_files[:args.max_days]

    if not npz_files:
        logger.error(f"No NPZ files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} NPZ files in {data_dir}")
    logger.info(f"Config: window={args.window_size}, stride={stride}, "
                f"derived={use_derived}, features={n_feat}")

    # --- Pass 1: compute global stats ---
    t0 = time.time()
    logger.info("Pass 1: Computing global feature statistics...")
    mean, std = compute_global_stats(npz_files, use_derived)
    logger.info(f"  Stats computed in {time.time()-t0:.1f}s")
    logger.info(f"  Mean: {mean}")
    logger.info(f"  Std:  {std}")

    stats = {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}
    torch.save(stats, output_dir / "stats.pt")
    logger.info(f"  Saved stats.pt")

    # --- Pass 2: extract windows and save per-file tensors ---
    logger.info("Pass 2: Extracting and saving windowed tensors...")
    t1 = time.time()
    total_windows = 0
    files_saved = 0

    for i, f in enumerate(npz_files):
        result = process_file(f, mean, std, args.window_size, stride, use_derived)
        if result is None:
            logger.warning(f"  [{i+1}/{len(npz_files)}] {f.name}: 0 valid windows, skipped")
            continue

        n_win = result["events"].shape[0]
        total_windows += n_win

        out_name = f.stem + ".pt"
        torch.save(result, output_dir / out_name)
        files_saved += 1

        if (i + 1) % 10 == 0 or (i + 1) == len(npz_files):
            logger.info(f"  [{i+1}/{len(npz_files)}] {f.name}: {n_win} windows | "
                        f"cumulative: {total_windows} windows")
        del result

    elapsed = time.time() - t1
    logger.info(f"Pass 2 complete in {elapsed:.1f}s")

    # --- Save metadata ---
    metadata = {
        "n_files": files_saved,
        "n_files_total": len(npz_files),
        "n_windows": total_windows,
        "window_size": args.window_size,
        "stride": stride,
        "n_features": n_feat,
        "derived_features": use_derived,
        "feature_names": feature_names,
        "horizons": HORIZONS,
        "source_dir": str(data_dir),
    }
    with open(output_dir / "metadata.json", "w") as fp:
        json.dump(metadata, fp, indent=2)

    total_elapsed = time.time() - t0
    logger.info(f"Done. {files_saved} files, {total_windows} windows saved to {output_dir}")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"Metadata saved to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
