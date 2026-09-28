#!/usr/bin/env python3
"""
Create RL prediction file (oot_wf_predictions_incremental.npz)
================================================================
Converts walk-forward fold prediction files into the format expected by rl_v3_3.py.

Input:  fold_XX_oot_predictions.npz files (from cnn_mamba_v2_smart_v3_mar training)
        Each contains: predictions (N,3), labels (N,3), oot_files (path with date), embeddings (N,96)

Output: oot_wf_predictions_incremental.npz
        Keys: {date}_preds (N,3) and {date}_mid (N,) for each date

Mid price reconstruction:
    labels[:,0] = 1s forward return in ticks (0.25 pts each).
    We reconstruct a synthetic mid price path by treating each event's 1s forward
    label as the approximate inter-event price change. The RL agent uses returns and
    momentum from this series, so the exact absolute level doesn't matter - only the
    relative shape. We use a base price of 5800.0 (typical ES ~Feb-Mar 2026).

    Note: The smart_v3 MBO event files have price_rel_ticks (col 3) clipped to [-2,2],
    making them unsuitable for raw mid price reconstruction. Labels are the reliable source.

Auto-detects Jupiter vs Neptune paths based on hostname.

Usage:
    python create_rl_prediction_file.py
    python create_rl_prediction_file.py --folds 0 1 2 3 4 5 6 7 8 9
    python create_rl_prediction_file.py --output /custom/path/output.npz
    python create_rl_prediction_file.py --base-price 5850.0
"""
import argparse
import re
import socket
import sys
from pathlib import Path

import numpy as np

# ES tick size
TICK_SIZE = 0.25
# Realistic ES futures base price for Feb-Mar 2026
DEFAULT_BASE_PRICE = 5800.0


def get_paths():
    """Auto-detect paths based on hostname."""
    hostname = socket.gethostname().lower()

    if 'neptune' in hostname or hostname == 'nick-desktop':
        lvl3_root = Path('/home/nick/Lvl3Quant')
    else:
        # Jupiter or any other host
        lvl3_root = Path('/home/jupiter/Lvl3Quant')

    paths = {
        'lvl3_root': lvl3_root,
        'fold_dir': lvl3_root / 'output' / 'cnn_mamba_v2_smart_v3_mar',
        'output_dir': lvl3_root / 'alpha_discovery' / 'deep_models' / 'results',
    }
    return paths


def extract_date_from_oot_files(oot_files):
    """Extract YYYYMMDD date string from oot_files path."""
    for f in oot_files:
        match = re.search(r'(\d{8})_mbo_events', str(f))
        if match:
            return match.group(1)
    return None


def reconstruct_mid_from_labels(labels, base_price=DEFAULT_BASE_PRICE):
    """
    Reconstruct synthetic mid prices from labels.

    labels[:,0] = 1s forward return in ticks.
    We treat this as the approximate inter-event price change:
        mid[i+1] = mid[i] + labels[i, 0] * TICK_SIZE

    With ~25k prediction events over ~6.5 trading hours, average inter-event
    spacing is ~0.94s, so labels_1s (1 second horizon) is a reasonable
    approximation for the step-to-step price change.

    The RL agent computes returns (np.diff(mids)/mids) and pct_change from this
    series, so the shape of the path matters more than the absolute level.
    """
    n = len(labels)
    forward_ticks = labels[:, 0].astype(np.float64)

    # Cumulative sum of tick changes gives price path
    mid_prices = base_price + np.cumsum(forward_ticks) * TICK_SIZE
    # Shift so mid_prices[0] = base_price (cumsum starts from first change)
    mid_prices = np.insert(mid_prices[:-1], 0, base_price)

    return mid_prices.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Create RL prediction file from walk-forward folds')
    parser.add_argument('--folds', type=int, nargs='+', default=list(range(100)),
                        help='Fold indices to include (default: 0-99, missing folds skipped)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: auto-detect)')
    parser.add_argument('--base-price', type=float, default=DEFAULT_BASE_PRICE,
                        help=f'Base ES price for mid reconstruction (default: {DEFAULT_BASE_PRICE})')
    parser.add_argument('--min-events', type=int, default=500,
                        help='Minimum prediction count to include a fold (default: 500)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without writing')
    args = parser.parse_args()

    paths = get_paths()
    fold_dir = paths['fold_dir']

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = paths['output_dir'] / 'oot_wf_predictions_incremental.npz'

    print(f"Fold directory: {fold_dir}")
    print(f"Output path:    {output_path}")
    print(f"Base price:     {args.base_price}")
    print()

    if not fold_dir.exists():
        print(f"ERROR: Fold directory not found: {fold_dir}")
        sys.exit(1)

    # Collect all fold predictions
    all_data = {}
    dates_processed = []
    skipped = []

    for fold_idx in args.folds:
        fold_file = fold_dir / f'fold_{fold_idx:02d}_oot_predictions.npz'
        if not fold_file.exists():
            continue

        fold = np.load(str(fold_file), allow_pickle=True)
        preds = fold['predictions']  # (N, 3)
        labels = fold['labels']      # (N, 3)
        oot_files = fold['oot_files']

        date = extract_date_from_oot_files(oot_files)
        if date is None:
            print(f"  fold_{fold_idx:02d}: could not extract date from oot_files={oot_files}, skipping")
            skipped.append(fold_idx)
            continue

        n = len(preds)

        if n < args.min_events:
            print(f"  fold_{fold_idx:02d}: date={date}, N={n:,} (below min {args.min_events}, skipping)")
            skipped.append(fold_idx)
            continue

        # Reconstruct mid prices from labels
        mid = reconstruct_mid_from_labels(labels, base_price=args.base_price)

        # Sanity checks
        assert len(mid) == n, f"Mid length {len(mid)} != preds length {n}"
        mid_range = mid.max() - mid.min()
        mid_mean = mid.mean()

        print(f"  fold_{fold_idx:02d}: date={date}, N={n:,}, "
              f"mid=[{mid.min():.2f}, {mid.max():.2f}], range={mid_range:.2f} pts")

        if not args.dry_run:
            all_data[f'{date}_preds'] = preds.astype(np.float32)
            all_data[f'{date}_mid'] = mid.astype(np.float32)
        dates_processed.append(date)

    print(f"\nProcessed: {len(dates_processed)} dates")
    if skipped:
        print(f"Skipped:   {len(skipped)} folds")
    print(f"Dates:     {sorted(dates_processed)}")

    if args.dry_run:
        print("\n[DRY RUN] No file written.")
        return

    if not all_data:
        print("ERROR: No data collected. Nothing to save.")
        sys.exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    np.savez_compressed(str(output_path), **all_data)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path} ({file_size_mb:.1f} MB)")
    print(f"Keys ({len(all_data)}): {sorted(all_data.keys())}")

    # Verify by loading back
    verify = np.load(str(output_path), allow_pickle=True)
    verify_dates = sorted(set(k.rsplit('_', 1)[0] for k in verify.files if k.endswith('_preds')))
    print(f"\nVerification: {len(verify_dates)} dates loaded successfully")
    for d in verify_dates:
        p = verify[f'{d}_preds']
        m = verify[f'{d}_mid']
        print(f"  {d}: preds={p.shape}, mid={m.shape}, "
              f"mid_range=[{m.min():.2f}, {m.max():.2f}]")


if __name__ == '__main__':
    main()
