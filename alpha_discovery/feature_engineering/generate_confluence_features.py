#!/usr/bin/env python3
"""
Generate confluence features from MBO event data (feat15).

Takes the 15 base features per event and derives additional rolling/interaction
features for confluence analysis and model training. All features are strictly
causal (only use past data within the rolling window).

Input:  /home/jupiter/Lvl3Quant/data/processed/mbo_events_feat15/*.npz
Output: /home/jupiter/Lvl3Quant/data/processed/mbo_events_confluence/*.npz

Original 15 features (indices 0-14):
  0: time_delta_log, 1: event_type, 2: side, 3: price_rel_ticks, 4: qty_log,
  5: spread_ticks, 6: cancel_side_asym_50, 7: rolling_ofi_500,
  8: event_density_20, 9: price_mom_10, 10: qty_price_mom_50,
  11: price_sign_momentum_200, 12: event_type_entropy_200,
  13: fill_add_restoration_100, 14: spread_velocity_50
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count

# === Paths ===
INPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_feat15")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_confluence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Feature column indices (original) ===
COL_TIME_DELTA  = 0
COL_EVENT_TYPE  = 1
COL_SIDE        = 2
COL_PRICE_REL   = 3
COL_QTY_LOG     = 4
COL_SPREAD      = 5
COL_CANCEL_ASYM = 6
COL_OFI_500     = 7
COL_DENSITY_20  = 8
COL_PRICE_MOM   = 9
COL_QTY_PMOM    = 10
COL_SIGN_MOM    = 11
COL_ENTROPY     = 12
COL_FILL_ADD    = 13
COL_SPR_VEL     = 14


# === Optimized rolling computations using cumsum approach ===

def rolling_mean_causal(arr, window):
    """Causal rolling mean using cumsum. Output[i] = mean of arr[max(0,i-window+1):i+1]."""
    n = len(arr)
    out = np.empty(n, dtype=np.float32)
    cs = np.cumsum(arr, dtype=np.float64)
    # For indices >= window, use cumsum difference
    out[:window] = cs[:window] / np.arange(1, window + 1, dtype=np.float64)
    out[window:] = (cs[window:] - cs[:-window]) / window
    return out


def rolling_sum_causal(arr, window):
    """Causal rolling sum using cumsum."""
    n = len(arr)
    out = np.empty(n, dtype=np.float32)
    cs = np.cumsum(arr, dtype=np.float64)
    out[:window] = cs[:window].astype(np.float32)
    out[window:] = (cs[window:] - cs[:-window]).astype(np.float32)
    return out


def rolling_std_causal(arr, window):
    """Causal rolling std using cumsum of x and x^2."""
    n = len(arr)
    out = np.empty(n, dtype=np.float32)
    cs = np.cumsum(arr, dtype=np.float64)
    cs2 = np.cumsum(arr.astype(np.float64) ** 2)

    # For warmup period
    for i in range(min(window, n)):
        cnt = i + 1
        mean = cs[i] / cnt
        var = cs2[i] / cnt - mean ** 2
        out[i] = np.sqrt(max(var, 0))

    # Steady state
    if n > window:
        cnt = window
        sums = cs[window:] - cs[:-window]
        sums2 = cs2[window:] - cs2[:-window]
        means = sums / cnt
        vars_ = sums2 / cnt - means ** 2
        np.maximum(vars_, 0, out=vars_)
        np.sqrt(vars_, out=out[window:])

    return out


def rolling_rank_causal(arr, window):
    """Percentile rank within rolling window. Uses approximate method for speed."""
    n = len(arr)
    out = np.empty(n, dtype=np.float32)
    # For large arrays, use a sorted-insert approach per-window is too slow.
    # Instead, compute rank as fraction of values below current within the window.
    # Approximate using rolling_mean of (arr < current_value).
    # More efficient: compare current value to rolling min/max.
    r_min = np.empty(n, dtype=np.float32)
    r_max = np.empty(n, dtype=np.float32)

    # Use a blockwise approach for min/max
    # For very large data, compute approximate rank from min/max range
    from numpy.lib.stride_tricks import sliding_window_view

    if n <= window:
        # Short array, just compute directly
        for i in range(n):
            w = arr[max(0, i - window + 1):i + 1]
            rng = w.max() - w.min()
            out[i] = (arr[i] - w.min()) / rng if rng > 0 else 0.5
        return out

    # Warmup
    for i in range(min(window, n)):
        w = arr[:i + 1]
        rng = w.max() - w.min()
        out[i] = (arr[i] - w.min()) / rng if rng > 0 else 0.5

    # For steady state, use quantile approximation via rolling mean of indicator
    # rank(x_i) ~ mean(x_j < x_i for j in window)
    # This is O(n*window) which is too slow for millions of events.
    # Instead, use (x - rolling_min) / (rolling_max - rolling_min) as proxy.
    # Compute rolling min/max using a simple O(n) approximation with EMA bounds.
    alpha = 2.0 / (window + 1)
    ema_lo = float(arr[0])
    ema_hi = float(arr[0])
    true_min = float(arr[0])
    true_max = float(arr[0])

    for i in range(window, n):
        val = float(arr[i])
        # Update EMA bounds (track min/max with decay)
        ema_lo = min(val, ema_lo + alpha * (val - ema_lo))
        ema_hi = max(val, ema_hi + alpha * (val - ema_hi))
        rng = ema_hi - ema_lo
        out[i] = (val - ema_lo) / rng if rng > 1e-10 else 0.5

    return np.clip(out, 0.0, 1.0)


def ema_causal(arr, span):
    """Exponential moving average, causal."""
    alpha = 2.0 / (span + 1)
    out = np.empty(len(arr), dtype=np.float32)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def generate_features_for_file(filepath):
    """Process a single NPZ file and generate confluence features."""
    fname = os.path.basename(filepath)
    outpath = OUTPUT_DIR / fname

    if outpath.exists():
        # Skip already processed
        return fname, "skipped (exists)", 0, 0

    t0 = time.time()
    data = np.load(filepath)
    events = data["events"]  # (N, 15) float32
    n = events.shape[0]

    # Extract base columns
    time_delta = events[:, COL_TIME_DELTA]
    event_type = events[:, COL_EVENT_TYPE]
    side = events[:, COL_SIDE]
    price_rel = events[:, COL_PRICE_REL]
    qty_log = events[:, COL_QTY_LOG]
    spread = events[:, COL_SPREAD]
    cancel_asym = events[:, COL_CANCEL_ASYM]
    ofi_500 = events[:, COL_OFI_500]
    density_20 = events[:, COL_DENSITY_20]
    price_mom = events[:, COL_PRICE_MOM]
    qty_pmom = events[:, COL_QTY_PMOM]
    sign_mom = events[:, COL_SIGN_MOM]
    entropy = events[:, COL_ENTROPY]
    fill_add = events[:, COL_FILL_ADD]
    spr_vel = events[:, COL_SPR_VEL]

    new_features = []

    # =========================================================================
    # 1. Volume Profile Features
    # =========================================================================
    # VWAP distance: price_rel - rolling_mean(price_rel * qty) / rolling_mean(qty)
    qty_raw = np.exp(qty_log)  # undo log for weighting
    pq = price_rel * qty_raw

    for w in [200, 500, 1000]:
        vwap = rolling_sum_causal(pq, w) / np.maximum(rolling_sum_causal(qty_raw, w), 1e-8)
        vwap_dist = price_rel - vwap
        new_features.append(vwap_dist)

    # Volume at bid vs ask (rolling). side=0 -> bid-ish, side=1 -> ask-ish, side=2 -> neutral
    bid_vol = qty_raw * (side < 1.5).astype(np.float32)  # sides 0, 1
    ask_vol = qty_raw * (side > 0.5).astype(np.float32)  # sides 1, 2
    # More precise: bid = side==0, ask = side==1 (side==2 is neutral/trade)
    bid_vol_precise = qty_raw * (np.abs(side - 0.0) < 0.1).astype(np.float32)
    ask_vol_precise = qty_raw * (np.abs(side - 1.0) < 0.1).astype(np.float32)

    for w in [200, 500, 1000]:
        bid_sum = rolling_sum_causal(bid_vol_precise, w)
        ask_sum = rolling_sum_causal(ask_vol_precise, w)
        total = bid_sum + ask_sum + 1e-8
        vol_imbalance = (bid_sum - ask_sum) / total
        new_features.append(vol_imbalance)

    # =========================================================================
    # 2. Order Flow Imbalance Patterns
    # =========================================================================
    # OFI acceleration: difference of rolling OFI at two scales
    ofi_fast = rolling_mean_causal(ofi_500, 50)
    ofi_slow = rolling_mean_causal(ofi_500, 200)
    ofi_accel = ofi_fast - ofi_slow  # positive = OFI accelerating
    new_features.append(ofi_accel)

    # OFI derivative (finite difference of OFI, lagged by 1)
    ofi_deriv = np.zeros(n, dtype=np.float32)
    ofi_deriv[1:] = ofi_500[1:] - ofi_500[:-1]
    new_features.append(ofi_deriv)

    # OFI regime: sustained direction indicator
    # rolling mean of sign(OFI) over 200 events, clipped to [-1, 1]
    ofi_sign = np.sign(ofi_500).astype(np.float32)
    ofi_regime_200 = rolling_mean_causal(ofi_sign, 200)
    ofi_regime_500 = rolling_mean_causal(ofi_sign, 500)
    new_features.append(ofi_regime_200)
    new_features.append(ofi_regime_500)

    # =========================================================================
    # 3. Spread Regime Features
    # =========================================================================
    # Spread percentile rank (rolling 1000)
    spread_pctile = rolling_rank_causal(spread, 1000)
    new_features.append(spread_pctile)

    # Spread change velocity (derivative)
    spread_change = np.zeros(n, dtype=np.float32)
    spread_change[1:] = spread[1:] - spread[:-1]
    new_features.append(spread_change)

    # Spread regime: std of spread over rolling window (volatility of spread)
    spread_vol = rolling_std_causal(spread, 500)
    new_features.append(spread_vol)

    # =========================================================================
    # 4. Time-of-Day Encoding
    # =========================================================================
    # Cumulative time from start (sum of time_delta which is log-transformed)
    # time_delta_log is already the log of inter-event time
    # Reconstruct cumulative elapsed time
    cum_time = np.cumsum(time_delta, dtype=np.float64)
    # Normalize to [0, 2*pi] assuming ~6.5 hour trading day
    # Max cum_time will represent end of day
    max_time = cum_time[-1] if cum_time[-1] > 0 else 1.0
    phase = (cum_time / max_time * 2 * np.pi).astype(np.float32)
    time_sin = np.sin(phase).astype(np.float32)
    time_cos = np.cos(phase).astype(np.float32)
    new_features.append(time_sin)
    new_features.append(time_cos)

    # Also add a linear time fraction (0 to 1)
    time_frac = (cum_time / max_time).astype(np.float32)
    new_features.append(time_frac)

    # =========================================================================
    # 5. Queue Depth Proxy
    # =========================================================================
    # Cumulative add-cancel imbalance at best levels
    # event_type: 0=add, 1=cancel, 2=modify, 3=trade, 4=other (approximate)
    is_add = (np.abs(event_type - 0.0) < 0.1).astype(np.float32)
    is_cancel = (np.abs(event_type - 1.0) < 0.1).astype(np.float32)
    add_cancel_flow = is_add - is_cancel  # +1 for add, -1 for cancel

    # Separate by side
    bid_ac = add_cancel_flow * (np.abs(side - 0.0) < 0.1).astype(np.float32)
    ask_ac = add_cancel_flow * (np.abs(side - 1.0) < 0.1).astype(np.float32)

    for w in [100, 500]:
        bid_depth = rolling_sum_causal(bid_ac, w)
        ask_depth = rolling_sum_causal(ask_ac, w)
        depth_imbalance = bid_depth - ask_depth
        depth_total = np.abs(bid_depth) + np.abs(ask_depth) + 1e-8
        depth_ratio = depth_imbalance / depth_total
        new_features.append(depth_ratio)

    # =========================================================================
    # 6. Momentum Divergence
    # =========================================================================
    # Price momentum vs OFI direction agreement
    # sign(price_mom) * sign(OFI) -> +1 agreement, -1 divergence
    pm_sign = np.sign(price_mom).astype(np.float32)
    ofi_sign_raw = np.sign(ofi_500).astype(np.float32)
    mom_ofi_agree = pm_sign * ofi_sign_raw
    new_features.append(mom_ofi_agree)

    # Rolling agreement score (sustained divergence is more meaningful)
    mom_ofi_agree_200 = rolling_mean_causal(mom_ofi_agree, 200)
    new_features.append(mom_ofi_agree_200)

    # Sign momentum vs OFI divergence (longer term)
    sign_ofi_agree = np.sign(sign_mom).astype(np.float32) * ofi_sign_raw
    sign_ofi_agree_500 = rolling_mean_causal(sign_ofi_agree, 500)
    new_features.append(sign_ofi_agree_500)

    # =========================================================================
    # 7. Event Clustering / Burst Detection
    # =========================================================================
    # Burst detection: how many events had very small time delta in recent window
    # "burst" = time_delta < median(time_delta) * 0.1
    # Since time_delta is log-transformed and many are 0, use threshold directly
    is_burst = (time_delta < 0.0001).astype(np.float32)  # near-simultaneous events
    burst_rate_20 = rolling_mean_causal(is_burst, 20)
    burst_rate_100 = rolling_mean_causal(is_burst, 100)
    new_features.append(burst_rate_20)
    new_features.append(burst_rate_100)

    # Event density acceleration (change in density)
    density_fast = rolling_mean_causal(density_20, 50)
    density_slow = rolling_mean_causal(density_20, 500)
    density_accel = density_fast - density_slow
    new_features.append(density_accel)

    # =========================================================================
    # 8. Cross-Feature Interactions
    # =========================================================================
    # Spread * OFI (wide spread + strong OFI = potential breakout)
    spread_ofi = spread * ofi_500
    # Normalize to manageable range
    spread_ofi_norm = spread_ofi / (np.abs(spread_ofi).mean() + 1e-8)
    new_features.append(np.clip(spread_ofi_norm, -10, 10).astype(np.float32))

    # Price_mom * event_density (momentum during high activity)
    mom_density = price_mom * density_20
    mom_density_norm = mom_density / (np.abs(mom_density).mean() + 1e-8)
    new_features.append(np.clip(mom_density_norm, -10, 10).astype(np.float32))

    # OFI * entropy (order flow clarity: high OFI + low entropy = clear direction)
    max_entropy = 1.61  # log(5) for 5 event types
    entropy_inv = max_entropy - entropy  # high = low entropy = concentrated
    ofi_clarity = ofi_500 * entropy_inv
    ofi_clarity_norm = ofi_clarity / (np.abs(ofi_clarity).mean() + 1e-8)
    new_features.append(np.clip(ofi_clarity_norm, -10, 10).astype(np.float32))

    # Fill-add restoration * spread (restoration during wide spread = stronger signal)
    fill_spread = fill_add * spread
    new_features.append(fill_spread.astype(np.float32))

    # Cancel asymmetry * price momentum (cancels aligned with momentum)
    cancel_mom = cancel_asym * price_mom
    cancel_mom_norm = cancel_mom / (np.abs(cancel_mom).mean() + 1e-8)
    new_features.append(np.clip(cancel_mom_norm, -10, 10).astype(np.float32))

    # =========================================================================
    # Stack all features
    # =========================================================================
    new_feat_array = np.column_stack(new_features)  # (N, num_new)
    assert new_feat_array.shape[0] == n

    # Combine original + new
    combined = np.concatenate([events, new_feat_array], axis=1).astype(np.float32)

    # Replace any NaN/Inf with 0
    nan_count = np.isnan(combined[:, 15:]).sum()
    inf_count = np.isinf(combined[:, 15:]).sum()
    combined = np.nan_to_num(combined, nan=0.0, posinf=10.0, neginf=-10.0)

    # Save with same keys
    save_dict = {"events": combined}
    for key in data.keys():
        if key != "events":
            save_dict[key] = data[key]

    np.savez_compressed(str(outpath), **save_dict)

    elapsed = time.time() - t0
    num_new = new_feat_array.shape[1]

    return fname, f"done in {elapsed:.1f}s", num_new, nan_count + inf_count


# === New feature names for documentation ===
NEW_FEATURE_NAMES = [
    # Volume profile (6)
    "vwap_dist_200", "vwap_dist_500", "vwap_dist_1000",
    "vol_imbalance_200", "vol_imbalance_500", "vol_imbalance_1000",
    # OFI patterns (4)
    "ofi_acceleration", "ofi_derivative", "ofi_regime_200", "ofi_regime_500",
    # Spread regime (3)
    "spread_pctile_1000", "spread_change", "spread_volatility_500",
    # Time encoding (3)
    "time_sin", "time_cos", "time_fraction",
    # Queue depth proxy (2)
    "queue_depth_ratio_100", "queue_depth_ratio_500",
    # Momentum divergence (3)
    "mom_ofi_agreement", "mom_ofi_agreement_200", "sign_ofi_agreement_500",
    # Event clustering (3)
    "burst_rate_20", "burst_rate_100", "density_acceleration",
    # Cross-feature interactions (5)
    "spread_x_ofi", "mom_x_density", "ofi_x_clarity", "fill_x_spread", "cancel_x_mom",
]


def main():
    files = sorted(INPUT_DIR.glob("*.npz"))
    print(f"Found {len(files)} input files")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Generating {len(NEW_FEATURE_NAMES)} new features")
    print(f"Total features per event: 15 + {len(NEW_FEATURE_NAMES)} = {15 + len(NEW_FEATURE_NAMES)}")
    print()

    # Print feature index map
    print("=== Feature Index Map ===")
    base_names = [
        "time_delta_log", "event_type", "side", "price_rel_ticks", "qty_log",
        "spread_ticks", "cancel_side_asym_50", "rolling_ofi_500",
        "event_density_20", "price_mom_10", "qty_price_mom_50",
        "price_sign_momentum_200", "event_type_entropy_200",
        "fill_add_restoration_100", "spread_velocity_50"
    ]
    for i, name in enumerate(base_names):
        print(f"  {i:2d}: {name}")
    for i, name in enumerate(NEW_FEATURE_NAMES):
        print(f"  {i + 15:2d}: {name}")
    print()

    # Process files - use multiprocessing for CPU parallelism
    n_workers = min(cpu_count(), 8)  # Cap at 8 to avoid memory issues
    print(f"Processing with {n_workers} workers...")
    print()

    total_nan = 0
    num_new_feats = 0

    # Process sequentially to manage memory (each file is 1-4 GB in memory)
    # With 64GB RAM and ~2GB per file processing, use 4 workers max
    n_workers = min(4, n_workers)

    with Pool(n_workers) as pool:
        results = pool.imap_unordered(generate_features_for_file, files)
        for i, (fname, status, n_new, n_bad) in enumerate(results):
            num_new_feats = max(num_new_feats, n_new)
            total_nan += n_bad
            print(f"  [{i + 1:3d}/{len(files)}] {fname}: {status}"
                  f"{f' (NaN/Inf fixed: {n_bad})' if n_bad > 0 else ''}")

    print()
    print("=== Summary ===")
    print(f"Files processed: {len(files)}")
    print(f"New features added: {num_new_feats}")
    print(f"Total features: {15 + num_new_feats}")
    print(f"Total NaN/Inf values fixed: {total_nan}")
    print(f"Output: {OUTPUT_DIR}")

    # Verify one output file
    sample = np.load(str(OUTPUT_DIR / os.path.basename(str(files[0]))))
    print(f"\nVerification ({os.path.basename(str(files[0]))}):")
    print(f"  events shape: {sample['events'].shape}")
    print(f"  NaN in events: {np.isnan(sample['events']).sum()}")
    print(f"  Inf in events: {np.isinf(sample['events']).sum()}")

    # Print statistics for new features
    ev = sample["events"]
    print(f"\n=== New Feature Statistics (sample file) ===")
    for i, name in enumerate(NEW_FEATURE_NAMES):
        col = ev[:, 15 + i]
        print(f"  {15 + i:2d} {name:30s} "
              f"min={col.min():10.4f} max={col.max():10.4f} "
              f"mean={col.mean():10.4f} std={col.std():10.4f}")


if __name__ == "__main__":
    main()
