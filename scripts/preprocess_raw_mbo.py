#!/usr/bin/env python3
"""
Preprocess raw Databento MBO .dbn.zst files into NumPy .npz format.
Matches the format used by tick_replay_engine and v21 sweep.

Target: January 2026 dates that have predictions but lack preprocessed MBO.
"""

import databento as db
import numpy as np
import os
import sys
from pathlib import Path

RAW_DIR = Path('/home/jupiter/Lvl3Quant/data/raw/mbo')
OUT_DIR = Path('/home/jupiter/Lvl3Quant/data/preprocessed_mbo')
PRED_DIR = Path('/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds')

# Action mapping (Databento MBO)
ACTION_MAP = {
    'A': 0, 'Add': 0,
    'C': 1, 'Cancel': 1,
    'M': 2, 'Modify': 2,
    'T': 3, 'Trade': 3,
    'F': 4, 'Fill': 4,
    'R': 5, 'Clear': 5,
}

# Side mapping (Databento MBO)
SIDE_MAP = {
    'B': 0, 'Bid': 0, 'A': 1, 'Ask': 1, 'N': 2, 'None': 2,
}

# Find dates needing preprocessing
pred_dates = set()
for f in os.listdir(PRED_DIR):
    if f.startswith('oot_') and f.endswith('.npz'):
        pred_dates.add(f.replace('oot_', '').replace('.npz', ''))

existing_dates = set()
for f in os.listdir(OUT_DIR):
    if f.startswith('mbo_') and f.endswith('.npz'):
        existing_dates.add(f.replace('mbo_', '').replace('.npz', ''))

missing_dates = sorted(pred_dates - existing_dates)
print(f"Prediction dates: {len(pred_dates)}")
print(f"Already preprocessed: {len(existing_dates)}")
print(f"Missing (need preprocessing): {len(missing_dates)}")
print(f"Dates: {missing_dates}")

# Find raw files for missing dates
to_process = []
for date in missing_dates:
    raw_file = RAW_DIR / f'glbx-mdp3-{date}.mbo.dbn.zst'
    if raw_file.exists():
        to_process.append((date, raw_file))
    else:
        print(f"  {date}: NO raw file found")

print(f"\nWill process: {len(to_process)} files")

for date, raw_path in to_process:
    print(f"\nProcessing {date}...")
    out_path = OUT_DIR / f'mbo_{date}.npz'

    try:
        # Read with databento
        store = db.DBNStore.from_file(str(raw_path))
        df = store.to_df()

        print(f"  Raw records: {len(df)}")

        # Filter to ES only (front month)
        if 'instrument_id' in df.columns:
            # Get the most common instrument (front month ES)
            top_instr = df['instrument_id'].value_counts().index[0]
            df = df[df['instrument_id'] == top_instr]
            print(f"  After ES filter: {len(df)}")

        # Extract fields
        ts_ns = df.index.astype(np.int64).values if hasattr(df.index, 'astype') else df['ts_event'].values

        # Action encoding
        if 'action' in df.columns:
            actions = df['action'].map(ACTION_MAP).fillna(0).astype(np.int8).values
        else:
            actions = np.zeros(len(df), dtype=np.int8)

        # Side encoding
        if 'side' in df.columns:
            side_vals = df['side'].astype(str)
            sides = side_vals.map(SIDE_MAP).fillna(2).astype(np.int8).values
        else:
            sides = np.zeros(len(df), dtype=np.int8) + 2

        # Price (convert from fixed-point if needed)
        if 'price' in df.columns:
            prices = df['price'].values.astype(np.float64)
            # Databento stores prices as fixed-point integers (price * 1e9)
            if prices.mean() > 1e6:
                prices = prices / 1e9
        else:
            prices = np.zeros(len(df), dtype=np.float64)

        # Size
        sizes = df['size'].values.astype(np.int32) if 'size' in df.columns else np.ones(len(df), dtype=np.int32)

        # Order ID
        order_ids = df['order_id'].values.astype(np.int64) if 'order_id' in df.columns else np.zeros(len(df), dtype=np.int64)

        # Count trades
        n_trades = int((actions == 3).sum() + (actions == 4).sum())

        print(f"  Events: {len(ts_ns)}, Trades: {n_trades}")
        print(f"  Price range: {prices[prices > 0].min():.2f} - {prices[prices > 0].max():.2f}")
        print(f"  Side dist: bid={int((sides==0).sum())}, ask={int((sides==1).sum())}, other={int((sides==2).sum())}")

        # Save
        np.savez_compressed(
            out_path,
            ts_ns=ts_ns,
            action=actions,
            side=sides,
            price=prices,
            size=sizes,
            order_id=order_ids,
            date=np.array(date),
            symbol=np.array('ES'),
            n_trades=np.array(n_trades),
        )

        print(f"  Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\nDone. Total preprocessed: {len(os.listdir(OUT_DIR))} files")
