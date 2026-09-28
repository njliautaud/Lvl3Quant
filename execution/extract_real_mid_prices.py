#!/usr/bin/env python3
"""
Extract REAL mid prices from raw .dbn.zst MBO files for RL training.
=================================================================
Replaces synthetic mid prices (cumsum of labels) with actual ES futures
mid prices computed from BBO tracking through the order book.

For each OOS date in the fold predictions:
1. Load raw .dbn.zst file
2. Run BBO tracking to get real mid prices for all events
3. Subsample to match prediction count (stride alignment)
4. Save to oot_wf_predictions_incremental.npz

Usage:
    python extract_real_mid_prices.py
    python extract_real_mid_prices.py --output /path/to/output.npz
"""
import argparse
import re
import socket
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

TICK_SIZE = 0.25

# ES contract rollover
ES_CONTRACTS = [
    ("2025-09-19", 14160),       # ESU5
    ("2025-12-19", 294973),      # ESZ5
    ("2026-03-20", 42140878),    # ESH6
    ("2026-06-19", None),        # ESM6
]


def get_paths():
    hostname = socket.gethostname().lower()
    if 'neptune' in hostname or hostname == 'nick-desktop':
        return Path('/home/nick/Lvl3Quant')
    return Path('/home/jupiter/Lvl3Quant')


def get_instrument_id(date_str):
    d = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    for cutoff, iid in ES_CONTRACTS:
        if d < cutoff:
            return iid
    return ES_CONTRACTS[-1][1]


def auto_detect_front_month(df, date_str):
    """Auto-detect front-month ES contract by most trades."""
    if "instrument_id" not in df.columns:
        return get_instrument_id(date_str)
    trades = df[df["action"] == "T"] if "action" in df.columns else df
    if len(trades) > 0:
        iid = trades["instrument_id"].value_counts().idxmax()
        return int(iid)
    iid = df["instrument_id"].value_counts().idxmax()
    return int(iid)


def extract_mid_from_dbn(raw_path, date_str, n_preds):
    """Extract real mid prices from a .dbn.zst file, subsampled to n_preds."""
    import databento as db

    print(f"  Loading {raw_path.name}...")
    store = db.DBNStore.from_file(str(raw_path))
    df = store.to_df()

    if df.empty:
        print(f"  ERROR: Empty dataframe for {date_str}")
        return None

    # Get front-month instrument
    iid = get_instrument_id(date_str)
    if iid is None:
        iid = auto_detect_front_month(df, date_str)

    if iid is not None and "instrument_id" in df.columns:
        df = df[df["instrument_id"] == iid]
        print(f"  Filtered to instrument_id={iid}: {len(df):,} events")

    if len(df) == 0:
        print(f"  ERROR: No events for instrument {iid} on {date_str}")
        return None

    # Extract prices, actions, sides
    prices = df["price"].values.astype(np.float64)
    actions = df["action"].values
    sides = df["side"].values
    n = len(df)

    print(f"  Running BBO tracking on {n:,} events...")

    # Track BBO to get mid prices
    mid_arr = np.full(n, np.nan, dtype=np.float64)
    bb, ba = np.nan, np.nan

    for i in range(n):
        p, act, side = prices[i], actions[i], sides[i]
        if not np.isnan(p):
            if act == "T" or act == "F":
                if side == "A":
                    bb = p
                elif side == "B":
                    ba = p
            elif act == "A":
                if side == "B" and (np.isnan(bb) or p > bb):
                    bb = p
                elif side == "A" and (np.isnan(ba) or p < ba):
                    ba = p
            elif act == "C":
                if side == "B" and not np.isnan(bb) and p >= bb:
                    bb = p - TICK_SIZE
                elif side == "A" and not np.isnan(ba) and p <= ba:
                    ba = p + TICK_SIZE

        if not np.isnan(bb) and not np.isnan(ba):
            mid_arr[i] = (bb + ba) / 2.0
        elif i > 0:
            mid_arr[i] = mid_arr[i - 1]

    # Forward-fill NaNs
    for i in range(1, n):
        if np.isnan(mid_arr[i]):
            mid_arr[i] = mid_arr[i - 1]

    # Count valid mid prices
    valid = np.sum(~np.isnan(mid_arr))
    print(f"  Mid prices: {valid:,}/{n:,} valid, range=[{np.nanmin(mid_arr):.2f}, {np.nanmax(mid_arr):.2f}]")

    if valid < n_preds:
        print(f"  WARNING: fewer valid mids ({valid}) than predictions ({n_preds})")

    # Subsample to match prediction count
    # The CNN-Mamba uses sliding windows with stride over events.
    # Predictions correspond to evenly-spaced events through the day.
    indices = np.linspace(0, n - 1, n_preds, dtype=int)
    mid_subsampled = mid_arr[indices].astype(np.float32)

    # Verify no NaNs in output
    nan_count = np.sum(np.isnan(mid_subsampled))
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN mid prices after subsampling, forward-filling...")
        for i in range(1, len(mid_subsampled)):
            if np.isnan(mid_subsampled[i]):
                mid_subsampled[i] = mid_subsampled[i - 1]
        # Fill any remaining leading NaNs with first valid
        first_valid = mid_subsampled[~np.isnan(mid_subsampled)]
        if len(first_valid) > 0:
            mid_subsampled[np.isnan(mid_subsampled)] = first_valid[0]

    return mid_subsampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--raw-dir', type=str, default=None)
    parser.add_argument('--fold-dir', type=str, default=None)
    args = parser.parse_args()

    lvl3 = get_paths()
    fold_dir = Path(args.fold_dir) if args.fold_dir else lvl3 / 'output' / 'cnn_mamba_v2_smart_v3_mar'
    raw_dir = Path(args.raw_dir) if args.raw_dir else lvl3 / 'data' / 'raw' / 'mbo'
    output_path = Path(args.output) if args.output else lvl3 / 'alpha_discovery' / 'deep_models' / 'results' / 'oot_wf_predictions_incremental.npz'

    print(f"Fold dir: {fold_dir}")
    print(f"Raw dir:  {raw_dir}")
    print(f"Output:   {output_path}")
    print()

    # Load existing predictions (to get preds and date info)
    import glob
    all_data = {}
    dates_info = []

    for fold_file in sorted(fold_dir.glob('fold_*_oot_predictions.npz')):
        fold = np.load(str(fold_file), allow_pickle=True)
        preds = fold['predictions']
        oot_files = fold['oot_files']

        # Extract date
        date = None
        for f in oot_files:
            m = re.search(r'(\d{8})_mbo_events', str(f))
            if m:
                date = m.group(1)
                break
        if date is None:
            continue

        n = len(preds)
        if n < 500:
            continue

        # Extract 10s horizon (column 2) as 1D preds for RL
        if preds.ndim == 2:
            preds_1d = preds[:, 2].astype(np.float32)  # 10s horizon
        else:
            preds_1d = preds.astype(np.float32)

        all_data[f'{date}_preds'] = preds_1d
        dates_info.append((date, n))
        print(f"  Fold {fold_file.stem}: date={date}, N={n:,}")

    print(f"\nFound {len(dates_info)} OOS dates with predictions")
    print()

    # Now extract real mid prices from .dbn.zst files
    for date, n_preds in dates_info:
        raw_file = raw_dir / f'glbx-mdp3-{date}.mbo.dbn.zst'
        if not raw_file.exists():
            print(f"  {date}: raw file not found at {raw_file}, using synthetic mid")
            # Fallback to synthetic
            labels_key = f'{date}_preds'
            mid = 5800.0 + np.cumsum(np.random.randn(n_preds) * 0.1) * TICK_SIZE
            all_data[f'{date}_mid'] = mid.astype(np.float32)
            continue

        mid = extract_mid_from_dbn(raw_file, date, n_preds)
        if mid is not None:
            all_data[f'{date}_mid'] = mid
            print(f"  {date}: REAL mid prices extracted, range=[{mid.min():.2f}, {mid.max():.2f}]")
        else:
            print(f"  {date}: extraction failed, skipping")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **all_data)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    print(f"Keys: {sorted(all_data.keys())}")

    # Verify
    verify = np.load(str(output_path), allow_pickle=True)
    for date, n_preds in dates_info:
        p = verify[f'{date}_preds']
        m = verify[f'{date}_mid']
        print(f"  {date}: preds={p.shape} [{p.min():.3f}, {p.max():.3f}], "
              f"mid={m.shape} [{m.min():.2f}, {m.max():.2f}]")


if __name__ == '__main__':
    main()
