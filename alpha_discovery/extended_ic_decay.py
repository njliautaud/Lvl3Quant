#!/usr/bin/env python3
"""
Extended IC Decay — Test top slow-decay features at LONG horizons (60s-600s).

Focus on features with negative alpha from full_ic_decay.py:
  total_depth_log, total_ask_vol, ask_pressure, bid_pressure,
  ask_L2/L3/L4/L5_orders, rvol_10/20/50, event_int_50

Also test COMPOSITE signals (simple average of top features).
"""

import gc
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXCLUDE_FEATURES = [0, 3, 8, 9]

# Extended horizons
HORIZONS = {
    '30s':   3000,
    '60s':   6000,
    '120s':  12000,
    '300s':  30000,
    '600s':  60000,
}

# Empirical move sizes (extend via sqrt scaling from known values)
# Known: 1s=1.73, 10s=5.61, 30s=9.91, 60s=13.81
MOVE_SIZES = {
    '30s':  9.91,
    '60s':  13.81,
    '120s': 19.53,  # 13.81 * sqrt(2)
    '300s': 30.88,  # 13.81 * sqrt(300/60)
    '600s': 43.67,  # 13.81 * sqrt(600/60)
}

COST_TICKS = 1.24


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-days', type=int, default=50)
    args = parser.parse_args()

    data_dir = ROOT / 'data' / 'processed' / 'mbo_features_cache'
    files = sorted(data_dir.glob('*_mbo_features.npz'))[:args.n_days]

    # Get feature names
    feature_names = None
    try:
        from alpha_discovery.mbo_features import get_feature_names
        all_names = get_feature_names()
        keep = [i for i in range(len(all_names)) if i not in EXCLUDE_FEATURES]
        feature_names = [all_names[i] for i in keep]
    except Exception:
        feature_names = None

    # Load data
    print(f"Extended IC Decay Analysis - {len(files)} days")
    print(f"Horizons: {list(HORIZONS.keys())}")
    print("Loading data...")
    t0 = time.time()

    day_features = []
    day_mids = []
    for i, fpath in enumerate(files):
        data = np.load(str(fpath))
        raw = data['mbo_features']
        mid = raw[:, 0].copy()
        mask = np.isnan(mid)
        if mask.any():
            first_valid = np.argmax(~mask)
            mid[:first_valid] = mid[first_valid]
        keep_cols = [j for j in range(raw.shape[1]) if j not in EXCLUDE_FEATURES]
        features = raw[:, keep_cols]
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        day_features.append(features)
        day_mids.append(mid)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}]")
        del raw; gc.collect()

    n_features = day_features[0].shape[1]
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(n_features)]

    # Build feature name -> index lookup
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Target features (top slow-decay from full_ic_decay.py)
    TARGET_FEATURES = [
        'total_depth_log', 'total_ask_vol', 'mean_ask_size', 'ask_pressure',
        'bid_pressure', 'ask_L2_orders', 'ask_L3_orders', 'ask_L4_orders',
        'ask_L5_orders', 'rvol_10', 'rvol_20', 'rvol_50', 'event_int_50',
        'depletion_dir_20', 'queue_depletion_asymmetry_20', 'ask_L1_orders',
        'ret_5', 'book_refresh', 'total_bid_vol', 'mean_bid_size',
    ]

    available_targets = [f for f in TARGET_FEATURES if f in name_to_idx]
    print(f"Testing {len(available_targets)} target features")
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Also compute actual move sizes from data
    print("\nComputing actual move sizes...")
    actual_moves = {h: [] for h in HORIZONS}
    for day_idx in range(len(files)):
        mid = day_mids[day_idx]
        N = len(mid)
        avg_price = np.nanmean(mid[mid > 0])
        tick_frac = 0.25 / avg_price

        for h_name, h_bars in HORIZONS.items():
            if N > h_bars + 100:
                fwd_ret = np.abs(mid[h_bars:N] - mid[:N-h_bars]) / np.where(
                    mid[:N-h_bars] > 0, mid[:N-h_bars], 1.0)
                # Convert to ticks
                abs_ret_ticks = fwd_ret / tick_frac
                actual_moves[h_name].append(np.nanmean(abs_ret_ticks))

    print("Actual move sizes (ticks):")
    for h_name in HORIZONS:
        if actual_moves[h_name]:
            avg = np.mean(actual_moves[h_name])
            MOVE_SIZES[h_name] = avg
            print(f"  {h_name:>5s}: {avg:.2f} ticks (breakeven IC = {COST_TICKS/avg:.4f})")

    # Compute IC for each target feature × horizon
    print("\nComputing IC matrix...")
    t0 = time.time()

    # Aggregate IC across days
    ic_per_day = {f: {h: [] for h in HORIZONS} for f in available_targets}
    ic_per_day['composite_top5'] = {h: [] for h in HORIZONS}
    ic_per_day['composite_top10'] = {h: [] for h in HORIZONS}

    for day_idx in range(len(files)):
        features = day_features[day_idx]
        mid = day_mids[day_idx]
        N = len(mid)

        for h_name, h_bars in HORIZONS.items():
            if N <= h_bars + 100:
                continue

            # Forward return
            fwd_ret = (mid[h_bars:N] - mid[:N-h_bars]) / np.where(
                mid[:N-h_bars] > 0, mid[:N-h_bars], 1.0)

            # Subsample to avoid autocorrelation
            spacing = h_bars  # space by one horizon length
            indices = np.arange(0, min(N - h_bars, len(fwd_ret)), spacing)
            if len(indices) < 20:
                continue

            y = fwd_ret[indices]
            valid_y = np.isfinite(y)

            # Individual features
            for f_name in available_targets:
                f_idx = name_to_idx[f_name]
                x = features[indices, f_idx]
                mask = valid_y & np.isfinite(x)
                if mask.sum() < 20:
                    continue
                ic, _ = spearmanr(x[mask], y[mask])
                if np.isfinite(ic):
                    ic_per_day[f_name][h_name].append(ic)

            # Composite signals (z-score and average)
            top5_names = ['total_depth_log', 'total_ask_vol', 'ask_pressure',
                          'bid_pressure', 'ask_L3_orders']
            top10_names = top5_names + ['ask_L2_orders', 'ask_L4_orders',
                                         'ask_L5_orders', 'rvol_10', 'event_int_50']

            for comp_name, comp_features in [('composite_top5', top5_names),
                                              ('composite_top10', top10_names)]:
                avail = [f for f in comp_features if f in name_to_idx]
                if len(avail) < 2:
                    continue
                # Z-score each feature, then average
                signals = []
                for f_name in avail:
                    f_idx = name_to_idx[f_name]
                    x = features[indices, f_idx].astype(np.float64)
                    std = np.std(x)
                    if std > 0:
                        signals.append((x - np.mean(x)) / std)
                if signals:
                    composite = np.mean(signals, axis=0)
                    mask = valid_y & np.isfinite(composite)
                    if mask.sum() >= 20:
                        ic, _ = spearmanr(composite[mask], y[mask])
                        if np.isfinite(ic):
                            ic_per_day[comp_name][h_name].append(ic)

        if (day_idx + 1) % 10 == 0:
            print(f"  [{day_idx+1}/{len(files)}] ({time.time()-t0:.0f}s)")

    print(f"Done ({time.time()-t0:.1f}s)")

    # Results
    print(f"\n{'='*100}")
    print(f"EXTENDED IC DECAY — Slow-Decay Features at Long Horizons")
    print(f"{'='*100}")

    header = f"{'Feature':30s}"
    for h in HORIZONS:
        header += f" {'IC@'+h:>9s}"
    header += f" {'Best E[ret]':>10s} {'Gap':>5s}"
    print(header)
    print("-" * 100)

    all_features = available_targets + ['composite_top5', 'composite_top10']
    best_overall = None
    best_eret = 0

    for f_name in all_features:
        row = f"{f_name:30s}"
        best_eret_this = 0
        best_h_this = ''
        for h_name in HORIZONS:
            daily_ics = ic_per_day[f_name][h_name]
            if daily_ics:
                mean_ic = np.mean(daily_ics)
                row += f" {mean_ic:+9.4f}"
                # Expected return
                avg_move = MOVE_SIZES.get(h_name, 0)
                eret = abs(mean_ic) * avg_move
                if eret > best_eret_this:
                    best_eret_this = eret
                    best_h_this = h_name
            else:
                row += f" {'N/A':>9s}"

        if best_eret_this > 0:
            gap = COST_TICKS / best_eret_this
            row += f" {best_eret_this:>8.2f}t {gap:>5.1f}x"
            if best_eret_this > best_eret:
                best_eret = best_eret_this
                best_overall = (f_name, best_h_this, best_eret_this)
        print(row)

    # Detailed profitability analysis
    print(f"\n{'='*100}")
    print(f"PROFITABILITY CHECK — Features within 2x of breakeven")
    print(f"{'='*100}")

    profitable_found = False
    for f_name in all_features:
        for h_name in HORIZONS:
            daily_ics = ic_per_day[f_name][h_name]
            if not daily_ics:
                continue
            mean_ic = abs(np.mean(daily_ics))
            std_ic = np.std(daily_ics) if len(daily_ics) > 1 else 0
            n = len(daily_ics)
            t_stat = mean_ic / (std_ic / np.sqrt(n)) if std_ic > 0 else 0
            avg_move = MOVE_SIZES.get(h_name, 0)
            eret = mean_ic * avg_move
            pct_pos = np.mean(np.array(daily_ics) > 0) * 100 if daily_ics else 0

            if eret > COST_TICKS * 0.5:
                gap = COST_TICKS / eret
                status = "PROFITABLE!" if eret >= COST_TICKS else f"gap={gap:.2f}x"
                if eret >= COST_TICKS:
                    profitable_found = True
                print(f"  {f_name:30s} @ {h_name:>5s}: IC={mean_ic:.4f} t={t_stat:.1f} "
                      f"pct+={pct_pos:.0f}% E[ret]={eret:.2f}t [{status}]")

    if best_overall:
        print(f"\n  BEST: {best_overall[0]} @ {best_overall[1]} — E[ret]={best_overall[2]:.2f} ticks")
        gap = COST_TICKS / best_overall[2]
        if gap <= 1.0:
            print(f"  >>> POTENTIALLY PROFITABLE! Need walk-forward validation.")
        else:
            print(f"  >>> Gap = {gap:.2f}x. Need IC improvement of {gap:.1f}x to break even.")

    # Save results
    results = {
        'horizons': list(HORIZONS.keys()),
        'actual_move_sizes': MOVE_SIZES,
        'feature_ics': {},
    }
    for f_name in all_features:
        results['feature_ics'][f_name] = {}
        for h_name in HORIZONS:
            daily_ics = ic_per_day[f_name][h_name]
            if daily_ics:
                results['feature_ics'][f_name][h_name] = {
                    'mean_ic': float(np.mean(daily_ics)),
                    'std_ic': float(np.std(daily_ics)),
                    'n_days': len(daily_ics),
                    'pct_positive': float(np.mean(np.array(daily_ics) > 0) * 100),
                }

    out_path = ROOT / 'alpha_discovery' / 'results' / 'extended_ic_decay.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
