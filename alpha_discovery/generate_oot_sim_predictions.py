#!/usr/bin/env python3
"""
Generate OOT sim prediction files from CNN OOT raw predictions.

Takes the raw CNN predictions (from train_oot_predictions.py) and applies
the same de-biasing pipeline as cnn_rust_sim_validation.py:
  1. Expanding z-score (no look-ahead)
  2. Vol gate (trailing vol percentile filter)
  3. Time filter (morning + afternoon only)

Outputs per-day .npz files in the format expected by the Rust fill simulator:
  {date}_vol{gate}_morning_afternoon.npz with 'predictions' key.

Usage:
    python alpha_discovery/generate_oot_sim_predictions.py
    python alpha_discovery/generate_oot_sim_predictions.py --pred-file results/oos_predictions_book_oot_TIMESTAMP.npz
"""

import argparse
import bisect
import gc
import logging
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

LVL3_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LVL3_ROOT))

logging.basicConfig(
    format='%(asctime)s [oot_sim_gen] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger('oot_sim_gen')

OOT_BOOK_DIR = LVL3_ROOT / 'data' / 'processed' / 'dl_book_cache_oot'
PRED_OUT_DIR = LVL3_ROOT / 'data' / 'processed' / 'cnn_oot_predictions'
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)

CNN_OFFSET = 19  # CNN window = 20 bars, so first valid prediction at bar 19

BARS_PER_SEC = 10  # 100ms bars
RTH_HOURS = 6.5

VOL_GATES = [50, 70, 80]
TIME_FILTERS = ['morning_afternoon']


def zscore_expanding(arr):
    """Expanding-window z-score (no look-ahead)."""
    result = np.full_like(arr, np.nan)
    running_sum = 0.0
    running_sq = 0.0
    count = 0
    for i in range(len(arr)):
        if np.isnan(arr[i]):
            continue
        running_sum += arr[i]
        running_sq += arr[i] ** 2
        count += 1
        if count >= 50:
            mean = running_sum / count
            var = (running_sq / count) - mean ** 2
            std = max(np.sqrt(var), 1e-8)
            result[i] = (arr[i] - mean) / std
    return result


def compute_trailing_vol(mid, window=3000):
    """Compute trailing realized vol (std of 1s returns) over window bars."""
    ret_1s = np.zeros(len(mid))
    ret_1s[10:] = (mid[10:] - mid[:-10]) / mid[:-10] * 10000  # in bps
    vol = np.full(len(mid), np.nan)
    cumsum = np.cumsum(ret_1s)
    cumsum2 = np.cumsum(ret_1s ** 2)
    for i in range(window, len(mid)):
        s = cumsum[i] - cumsum[i - window]
        s2 = cumsum2[i] - cumsum2[i - window]
        mean = s / window
        var = s2 / window - mean ** 2
        vol[i] = np.sqrt(max(var, 0))
    return vol


def precompute_vol_percentiles(vol_pred, percentiles=(50, 60, 70, 80, 90)):
    """Expanding-window vol percentile thresholds (no look-ahead)."""
    n = len(vol_pred)
    result = {p: np.full(n, -np.inf) for p in percentiles}
    sorted_vals = []
    for i in range(n):
        if not np.isnan(vol_pred[i]):
            bisect.insort(sorted_vals, vol_pred[i])
        if len(sorted_vals) >= 100:
            for p in percentiles:
                idx = min(int(len(sorted_vals) * p / 100), len(sorted_vals) - 1)
                result[p][i] = sorted_vals[idx]
    return result


def compute_time_features(n_bars):
    """Compute time-of-day boolean masks."""
    seconds = np.arange(n_bars) / BARS_PER_SEC
    minutes = seconds / 60.0
    morning_afternoon = (minutes < 120) | ((minutes >= 240) & (minutes < 330))
    return morning_afternoon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-file', type=str, default=None,
                        help='Path to CNN OOT predictions .npz file')
    parser.add_argument('--book-dir', type=str, default=str(OOT_BOOK_DIR),
                        help='Directory with OOT book tensor .npz files')
    parser.add_argument('--output-dir', type=str, default=str(PRED_OUT_DIR))
    args = parser.parse_args()

    # Auto-find latest prediction file if not specified
    pred_file = args.pred_file
    if pred_file is None:
        results_dir = LVL3_ROOT / 'alpha_discovery' / 'deep_models' / 'results'
        candidates = sorted(results_dir.glob('oos_predictions_book_oot_*.npz'))
        if not candidates:
            log.error("No OOT prediction files found. Run train_oot_predictions.py first.")
            return
        pred_file = str(candidates[-1])
        log.info(f"Auto-selected: {pred_file}")

    pred_file = Path(pred_file)
    if not pred_file.exists():
        log.error(f"Prediction file not found: {pred_file}")
        return

    book_dir = Path(args.book_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    log.info(f"Loading CNN OOT predictions from {pred_file.name}...")
    cnn_data = np.load(str(pred_file), allow_pickle=True)
    pred_keys = [k for k in cnn_data.keys() if k.endswith('_preds')]
    dates = sorted([k.replace('_preds', '') for k in pred_keys])
    log.info(f"  Found {len(dates)} OOT dates: {dates[0]} to {dates[-1]}")

    saved_files = {}
    n_skipped = 0

    for i, date in enumerate(dates):
        # Load raw CNN predictions
        preds_key = f'{date}_preds'
        mid_key = f'{date}_mid'

        if preds_key not in cnn_data:
            log.warning(f"  No predictions for {date}")
            n_skipped += 1
            continue

        cp = cnn_data[preds_key].astype(np.float64)

        # Load mid prices from book tensor cache (for vol computation)
        book_file = book_dir / f'{date}_book_tensors.npz'
        if mid_key in cnn_data:
            mid = cnn_data[mid_key].astype(np.float64)
        elif book_file.exists():
            book_npz = np.load(str(book_file))
            mid = book_npz['mid_prices'].astype(np.float64)
        else:
            log.warning(f"  No mid prices for {date}, skipping")
            n_skipped += 1
            continue

        n_bars = len(mid)

        # Predictions are already aligned (one per bar, CNN_OFFSET bars have 0 predictions)
        # But let's verify length
        if len(cp) != n_bars:
            log.warning(f"  {date}: pred length {len(cp)} != n_bars {n_bars}, padding")
            cp_aligned = np.zeros(n_bars, dtype=np.float64)
            cp_aligned[:min(len(cp), n_bars)] = cp[:min(len(cp), n_bars)]
            cp = cp_aligned

        # Apply expanding z-score
        signal = zscore_expanding(cp)

        # Compute vol for gating
        vol_pred = compute_trailing_vol(mid)
        vol_pct_thresholds = precompute_vol_percentiles(vol_pred)

        # Time features
        morning_afternoon = compute_time_features(n_bars)

        for vol_gate in VOL_GATES:
            for tf in TIME_FILTERS:
                filtered_signal = signal.copy()

                # Vol gate
                if vol_gate > 0:
                    available = sorted(vol_pct_thresholds.keys())
                    closest = min(available, key=lambda x: abs(x - vol_gate))
                    vol_threshold = vol_pct_thresholds[closest]
                    for j in range(len(filtered_signal)):
                        if np.isnan(vol_pred[j]) or vol_pred[j] < vol_threshold[j]:
                            filtered_signal[j] = 0.0

                # Time filter
                if tf == 'morning_afternoon':
                    filtered_signal[~morning_afternoon] = 0.0

                # Replace NaN with 0
                filtered_signal = np.nan_to_num(filtered_signal, nan=0.0)

                # Save
                key = (date, vol_gate, tf)
                out_file = output_dir / f'{date}_vol{vol_gate}_{tf}.npz'
                np.savez_compressed(str(out_file), predictions=filtered_signal)
                saved_files[key] = out_file

        if (i + 1) % 10 == 0 or i == 0 or i == len(dates) - 1:
            non_zero = np.count_nonzero(signal[~np.isnan(signal)])
            log.info(f"  [{i+1}/{len(dates)}] {date}: {n_bars} bars, "
                    f"non-zero signals: {non_zero}")

        del mid, signal, vol_pred, vol_pct_thresholds
        gc.collect()

    log.info(f"\nGenerated {len(saved_files)} prediction files in {output_dir}")
    log.info(f"  Skipped: {n_skipped} dates")
    log.info(f"  Vol gates: {VOL_GATES}")
    log.info(f"  Time filters: {TIME_FILTERS}")
    log.info(f"  Files per date: {len(VOL_GATES) * len(TIME_FILTERS)}")

    # Summary stats
    sample_file = list(saved_files.values())[0] if saved_files else None
    if sample_file:
        d = np.load(str(sample_file))
        p = d['predictions']
        log.info(f"\n  Sample file: {sample_file.name}")
        log.info(f"    Shape: {p.shape}")
        log.info(f"    Non-zero: {np.count_nonzero(p)}")
        log.info(f"    Range: [{p.min():.3f}, {p.max():.3f}]")
        log.info(f"    Mean (non-zero): {p[p != 0].mean():.3f}" if np.any(p != 0) else "    All zeros")


if __name__ == '__main__':
    main()
