#!/usr/bin/env python3
"""
Generate 10s calibration JSON by running the LGBM model on MBO event data.
Uses StreamingFeatures to compute the exact 21-dim features the model expects.
"""
import numpy as np
import json
import joblib
from pathlib import Path
from live_trading_linux.streaming_features import StreamingFeatures

ROOT = Path('/home/jupiter/Lvl3Quant')
MODEL_PATH = ROOT / 'models/lgbm_60_5_fold35/labels_10s_lgbm.pkl'
DATA_DIR = ROOT / 'data/processed/mbo_events_feat'  # Has raw 15-dim events
OUT_PATH = ROOT / 'live_trading_linux/models/labels_10s_calibration.json'

print(f"Loading model: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Use last 5 days of data for calibration
files = sorted(DATA_DIR.glob('*_mbo_events.npz'))
cal_files = files[-5:]
print(f"Using {len(cal_files)} files: {[f.stem[:8] for f in cal_files]}")

all_preds = []

for fp in cal_files:
    print(f"  Processing {fp.name}...")
    d = np.load(fp, allow_pickle=True)
    events = d['events']  # (N, 15) or (N, 6)

    # Extract the raw 6 event fields the streaming features expect:
    # time_delta, event_type, side, price, qty, spread
    # From the 15-dim format, first 6 columns are the raw event fields
    raw = events[:, :6]

    sf = StreamingFeatures()
    feats = []

    # Process events through streaming features (subsample for speed)
    step = max(1, len(raw) // 50000)  # ~50K samples per file
    for i in range(0, len(raw), step):
        row = raw[i]
        time_delta = float(row[0])
        event_type = int(row[1])
        side = int(row[2])
        price = float(row[3])
        qty = float(row[4])
        spread = float(row[5])

        vec = sf.update(
            time_delta=time_delta,
            event_type=event_type,
            side=side,
            price=price,
            qty=qty,
            spread=spread,
        )
        if vec is not None and len(feats) > 500:  # Skip warm-up
            feats.append(vec)

    if feats:
        X = np.array(feats, dtype=np.float32)
        # Filter out NaN/Inf
        valid = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid]
        if len(X) > 0:
            preds = model.predict(X)
            all_preds.append(preds)
            print(f"    Got {len(preds)} predictions, mean={np.mean(np.abs(preds)):.4f}")

if all_preds:
    all_preds = np.concatenate(all_preds)
    print(f"\nTotal predictions: {len(all_preds)}")

    # Build calibration
    abs_p = np.abs(all_preds)
    cal = {
        "p50": float(np.percentile(abs_p, 50)),
        "p75": float(np.percentile(abs_p, 75)),
        "p90": float(np.percentile(abs_p, 90)),
        "n": int(len(all_preds)),
        "pred_mean": float(np.mean(all_preds)),
        "pred_std": float(np.std(all_preds)),
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(cal, f, indent=2)

    print(f"\nCalibration saved to {OUT_PATH}")
    print(json.dumps(cal, indent=2))
else:
    print("ERROR: No predictions generated!")
