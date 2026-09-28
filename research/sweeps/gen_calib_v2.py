#!/usr/bin/env python3
"""Generate 10s calibration from batch event data."""
import numpy as np
import json
import joblib
from pathlib import Path

ROOT = Path('/home/jupiter/Lvl3Quant')
MODEL_PATH = ROOT / 'models/lgbm_60_5_fold35/labels_10s_lgbm.pkl'
OUT_PATH = ROOT / 'live_trading_linux/models/labels_10s_calibration.json'

# The model expects 21 features in this order:
# time_delta, event_type, side, price, qty, spread,
# rolling_ofi_100, cancel_side_asym_100, event_density_50,
# price_mom_20, qty_price_mom_20, cum_delta, roll_delta_500,
# ofi_short_20, cancel_rate_100, trade_rate_50, add_side_asym_100,
# spread_velocity_50, qty_add_imbalance_100, price_sign_mom_200,
# fill_recovery_20

# The mbo_events_feat files have 15-dim events.
# The mbo_events_feat20 files have 20-dim events with names:
# time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log,
# spread_ticks, cancel_side_asym_50, rolling_ofi_500, event_density_20,
# price_mom_10, qty_price_mom_50, price_sign_mom_200,
# event_type_entropy_200, fill_add_restore_100, spread_velocity_50,
# price_sign_sq, fade_side_price, restore_x_price, rolling_rvol_500,
# rolling_vwap_dist_200

# These don't map 1:1 to the 21 expected features.
# PRACTICAL APPROACH: create calibration from the 1s calibration scaled down.
# The 1s model had pred_std=0.609. The 10s model will have different scale.
# Let's just run the model on SOME data and measure.

# Actually, let's use Saturn's data format. Load the raw 6-dim events
# from mbo_events_feat and compute derived features matching training.

# Or simplest: proportional scaling from 1s calibration
# From fold_meta: IC_1s=0.122, IC_10s=0.041 → ratio ~0.34
# But IC isn't directly proportional to prediction scale.
# Let's estimate: if 1s pred_std=0.609, 10s pred_std ≈ 0.609 * (0.041/0.122) ≈ 0.205
# But this is just an estimate.

# Better: run model on arbitrary 21-dim data and measure prediction distribution
print("Loading model...")
model = joblib.load(MODEL_PATH)

# Generate synthetic features from actual event data to measure pred distribution
DATA_DIR = ROOT / 'data/processed/mbo_events_feat'
files = sorted(DATA_DIR.glob('*.npz'))[-3:]

all_preds = []
for fp in files:
    print(f"  {fp.name}...")
    d = np.load(fp, allow_pickle=True)
    events = d['events']  # (N, 15)

    # Extract base 6 features: time_delta, event_type, side, price, qty, spread
    raw = events[:, :6].astype(np.float32)
    N = len(raw)

    # Compute the 15 derived features using numpy (matching training compute_derived)
    # Features 7-21 are rolling statistics. Approximate with simple windowed ops.

    time_delta = raw[:, 0]
    event_type = raw[:, 1]
    side = raw[:, 2]
    price = raw[:, 3]
    qty = raw[:, 4]
    spread = raw[:, 5]

    # Simple causal rolling computations
    def rolling_mean(x, k):
        """Causal rolling mean matching np.convolve(x, ones(k)/k, 'full')[:N]"""
        cs = np.cumsum(x)
        out = np.zeros_like(x)
        out[:k] = cs[:k] / k
        out[k:] = (cs[k:] - cs[:-k]) / k
        return out

    # Signed OFI: +qty for bid adds, -qty for ask adds
    ofi = np.where(side == 0, qty, -qty)  # bid=0, ask=1

    feat_21 = np.zeros((N, 21), dtype=np.float32)
    feat_21[:, 0] = time_delta
    feat_21[:, 1] = event_type
    feat_21[:, 2] = side
    feat_21[:, 3] = price
    feat_21[:, 4] = qty
    feat_21[:, 5] = spread
    feat_21[:, 6] = rolling_mean(ofi, 100)                    # rolling_ofi_100
    feat_21[:, 7] = rolling_mean((event_type == 1).astype(float) * (2*side - 1), 100)  # cancel_side_asym_100
    feat_21[:, 8] = rolling_mean(np.ones(N), 50)              # event_density_50
    feat_21[:, 9] = rolling_mean(np.diff(np.concatenate([[0], price])), 20)  # price_mom_20
    feat_21[:, 10] = rolling_mean(np.diff(np.concatenate([[0], price])) * qty, 20)  # qty_price_mom_20
    feat_21[:, 11] = np.cumsum(ofi) / np.arange(1, N+1)       # cum_delta (normalized)
    feat_21[:, 12] = rolling_mean(ofi, 500)                    # roll_delta_500
    feat_21[:, 13] = rolling_mean(ofi, 20)                     # ofi_short_20
    feat_21[:, 14] = rolling_mean((event_type == 1).astype(float), 100)  # cancel_rate_100
    feat_21[:, 15] = rolling_mean((event_type == 3).astype(float), 50)   # trade_rate_50
    feat_21[:, 16] = rolling_mean((event_type == 0).astype(float) * (2*side - 1), 100)  # add_side_asym_100
    feat_21[:, 17] = rolling_mean(np.diff(np.concatenate([[0], spread])), 50)  # spread_velocity_50
    feat_21[:, 18] = rolling_mean((event_type == 0).astype(float) * qty * (2*side - 1), 100)  # qty_add_imbalance_100
    feat_21[:, 19] = rolling_mean(np.sign(np.diff(np.concatenate([[0], price]))), 200)  # price_sign_mom_200
    feat_21[:, 20] = rolling_mean((event_type == 3).astype(float) * qty, 20)  # fill_recovery_20

    # Filter NaN/Inf
    valid = ~(np.isnan(feat_21).any(axis=1) | np.isinf(feat_21).any(axis=1))
    feat_21 = feat_21[valid]

    # Subsample for speed
    idx = np.random.choice(len(feat_21), min(100000, len(feat_21)), replace=False)
    feat_21 = feat_21[idx]

    preds = model.predict(feat_21)
    all_preds.append(preds)
    print(f"    {len(preds)} preds, |mean|={np.mean(np.abs(preds)):.4f}, std={np.std(preds):.4f}")

all_preds = np.concatenate(all_preds)
print(f"\nTotal: {len(all_preds)} predictions")

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

print(f"\nCalibration saved: {OUT_PATH}")
print(json.dumps(cal, indent=2))
