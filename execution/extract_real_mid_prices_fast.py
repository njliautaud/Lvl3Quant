#!/usr/bin/env python3
"""
Extract REAL mid prices from raw .dbn.zst MBO files for RL training.
=================================================================
FAST version using numba JIT for BBO tracking.

For each OOS date in the fold predictions:
1. Load raw .dbn.zst file
2. Run JIT-compiled BBO tracking to get real mid prices
3. Subsample to match prediction count
4. Save to oot_wf_predictions_incremental.npz
"""
import argparse
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
    print("Using numba JIT for BBO tracking (FAST)")
except ImportError:
    HAS_NUMBA = False
    print("WARNING: numba not available, using pure numpy (slower)")

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


if HAS_NUMBA:
    @njit(cache=True)
    def track_bbo_numba(prices, actions_int, sides_int, n):
        """
        Numba-accelerated BBO tracking.
        actions_int: 0=A(add), 1=C(cancel), 2=M(modify), 3=T(trade), 4=F(fill)
        sides_int: 0=B(bid), 1=A(ask), 2=N(none)
        """
        mid_arr = np.full(n, np.nan, dtype=np.float64)
        bb = np.nan
        ba = np.nan
        tick = 0.25

        for i in range(n):
            p = prices[i]
            act = actions_int[i]
            side = sides_int[i]

            if not np.isnan(p) and p > 0:
                # Trade or Fill
                if act == 3 or act == 4:
                    if side == 1:  # Ask side traded = buyer hit ask, so best bid updated
                        bb = p
                    elif side == 0:  # Bid side traded = seller hit bid, so best ask updated
                        ba = p
                # Add order
                elif act == 0:
                    if side == 0 and (np.isnan(bb) or p > bb):
                        bb = p
                    elif side == 1 and (np.isnan(ba) or p < ba):
                        ba = p
                # Cancel order
                elif act == 1:
                    if side == 0 and not np.isnan(bb) and p >= bb:
                        bb = p - tick
                    elif side == 1 and not np.isnan(ba) and p <= ba:
                        ba = p + tick

            if not np.isnan(bb) and not np.isnan(ba) and ba > bb:
                mid_arr[i] = (bb + ba) / 2.0
            elif i > 0:
                mid_arr[i] = mid_arr[i - 1]

        # Forward-fill NaNs
        for i in range(1, n):
            if np.isnan(mid_arr[i]):
                mid_arr[i] = mid_arr[i - 1]

        return mid_arr


def track_bbo_numpy(prices, actions_int, sides_int, n):
    """Pure numpy fallback (still uses a loop but with less overhead)."""
    mid_arr = np.full(n, np.nan, dtype=np.float64)
    bb = np.nan
    ba = np.nan
    tick = 0.25

    for i in range(n):
        p = prices[i]
        act = actions_int[i]
        side = sides_int[i]

        if not np.isnan(p) and p > 0:
            if act == 3 or act == 4:
                if side == 1:
                    bb = p
                elif side == 0:
                    ba = p
            elif act == 0:
                if side == 0 and (np.isnan(bb) or p > bb):
                    bb = p
                elif side == 1 and (np.isnan(ba) or p < ba):
                    ba = p
            elif act == 1:
                if side == 0 and not np.isnan(bb) and p >= bb:
                    bb = p - tick
                elif side == 1 and not np.isnan(ba) and p <= ba:
                    ba = p + tick

        if not np.isnan(bb) and not np.isnan(ba) and ba > bb:
            mid_arr[i] = (bb + ba) / 2.0
        elif i > 0:
            mid_arr[i] = mid_arr[i - 1]

    for i in range(1, n):
        if np.isnan(mid_arr[i]):
            mid_arr[i] = mid_arr[i - 1]

    return mid_arr


ACTION_MAP = {"A": 0, "C": 1, "M": 2, "T": 3, "F": 4,
              "R": 5, "trade": 3, "fill": 4}
SIDE_MAP = {"B": 0, "A": 1, "N": 2, "None": 2}


def extract_mid_from_dbn(raw_path, date_str, n_preds):
    """Extract real mid prices from a .dbn.zst file, subsampled to n_preds."""
    import databento as db

    t0 = time.time()
    print(f"  Loading {raw_path.name}...", flush=True)
    store = db.DBNStore.from_file(str(raw_path))
    df = store.to_df()
    t1 = time.time()
    print(f"  Loaded {len(df):,} events in {t1-t0:.1f}s", flush=True)

    if df.empty:
        print(f"  ERROR: Empty dataframe for {date_str}", flush=True)
        return None

    # Get front-month instrument
    iid = get_instrument_id(date_str)
    if iid is None:
        # Auto-detect
        if "instrument_id" in df.columns:
            trades = df[df["action"] == "T"] if "action" in df.columns else df
            if len(trades) > 0:
                iid = int(trades["instrument_id"].value_counts().idxmax())
            else:
                iid = int(df["instrument_id"].value_counts().idxmax())
            print(f"  Auto-detected instrument_id={iid}", flush=True)

    if iid is not None and "instrument_id" in df.columns:
        df = df[df["instrument_id"] == iid]
        print(f"  Filtered to instrument {iid}: {len(df):,} events", flush=True)

    if len(df) == 0:
        print(f"  ERROR: No events for instrument {iid} on {date_str}", flush=True)
        return None

    # Convert to numeric arrays
    prices = df["price"].values.astype(np.float64)
    n = len(prices)

    # Map actions and sides to integers
    actions_raw = df["action"].values
    sides_raw = df["side"].values

    actions_int = np.zeros(n, dtype=np.int32)
    sides_int = np.full(n, 2, dtype=np.int32)  # default None

    for i in range(n):
        a = str(actions_raw[i])
        s = str(sides_raw[i])
        actions_int[i] = ACTION_MAP.get(a, 2)
        sides_int[i] = SIDE_MAP.get(s, 2)

    t2 = time.time()
    print(f"  Preprocessed in {t2-t1:.1f}s, running BBO tracking on {n:,} events...", flush=True)

    # Run BBO tracking
    if HAS_NUMBA:
        mid_arr = track_bbo_numba(prices, actions_int, sides_int, n)
    else:
        mid_arr = track_bbo_numpy(prices, actions_int, sides_int, n)

    t3 = time.time()
    print(f"  BBO tracking done in {t3-t2:.1f}s", flush=True)

    # Count valid mid prices
    valid = np.sum(~np.isnan(mid_arr))
    if valid == 0:
        print(f"  ERROR: No valid mid prices", flush=True)
        return None

    print(f"  Mid prices: {valid:,}/{n:,} valid, "
          f"range=[{np.nanmin(mid_arr):.2f}, {np.nanmax(mid_arr):.2f}]", flush=True)

    # Subsample to match prediction count
    indices = np.linspace(0, n - 1, n_preds, dtype=int)
    mid_subsampled = mid_arr[indices].astype(np.float32)

    # Fix any NaNs
    nan_count = np.sum(np.isnan(mid_subsampled))
    if nan_count > 0:
        print(f"  Fixing {nan_count} NaN mid prices...", flush=True)
        for i in range(1, len(mid_subsampled)):
            if np.isnan(mid_subsampled[i]):
                mid_subsampled[i] = mid_subsampled[i - 1]
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

    print(f"Fold dir: {fold_dir}", flush=True)
    print(f"Raw dir:  {raw_dir}", flush=True)
    print(f"Output:   {output_path}", flush=True)
    print(flush=True)

    # Load fold predictions
    all_data = {}
    dates_info = []

    for fold_file in sorted(fold_dir.glob('fold_*_oot_predictions.npz')):
        fold = np.load(str(fold_file), allow_pickle=True)
        preds = fold['predictions']
        oot_files = fold['oot_files']

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

        # 10s horizon for RL
        if preds.ndim == 2:
            preds_1d = preds[:, 2].astype(np.float32)
        else:
            preds_1d = preds.astype(np.float32)

        all_data[f'{date}_preds'] = preds_1d
        dates_info.append((date, n))
        print(f"  Fold {fold_file.stem}: date={date}, N={n:,}", flush=True)

    print(f"\nFound {len(dates_info)} OOS dates", flush=True)
    print(flush=True)

    # Extract real mid prices
    t_total = time.time()
    success = 0
    for i, (date, n_preds) in enumerate(dates_info):
        print(f"\n[{i+1}/{len(dates_info)}] Processing {date}...", flush=True)
        raw_file = raw_dir / f'glbx-mdp3-{date}.mbo.dbn.zst'
        if not raw_file.exists():
            print(f"  Raw file not found: {raw_file}", flush=True)
            # Use simple synthetic mid as fallback
            mid = np.full(n_preds, 5800.0, dtype=np.float32)
            all_data[f'{date}_mid'] = mid
            continue

        try:
            mid = extract_mid_from_dbn(raw_file, date, n_preds)
            if mid is not None:
                all_data[f'{date}_mid'] = mid
                success += 1
                print(f"  OK: range=[{mid.min():.2f}, {mid.max():.2f}]", flush=True)
            else:
                all_data[f'{date}_mid'] = np.full(n_preds, 5800.0, dtype=np.float32)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            all_data[f'{date}_mid'] = np.full(n_preds, 5800.0, dtype=np.float32)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"Extracted real mid prices for {success}/{len(dates_info)} dates in {elapsed:.0f}s", flush=True)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **all_data)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)", flush=True)
    print(f"Keys: {sorted(all_data.keys())}", flush=True)

    # Verify
    print("\nVerification:", flush=True)
    verify = np.load(str(output_path), allow_pickle=True)
    for date, n_preds in dates_info:
        p = verify[f'{date}_preds']
        m = verify[f'{date}_mid']
        print(f"  {date}: preds={p.shape} [{p.min():.3f}, {p.max():.3f}], "
              f"mid={m.shape} [{m.min():.2f}, {m.max():.2f}]", flush=True)


if __name__ == '__main__':
    main()
