#!/usr/bin/env python3
"""
Full IC Decay Analysis — Test ALL 336 features across multiple horizons.

Goal: Find features whose IC decays SLOWER than sqrt(horizon), making longer holds profitable.

Key metric: "decay_alpha" where IC(h) ~ IC(1s) * h^(-alpha)
- alpha < 0.5: IC decays slower than moves grow → LONGER HOLDS HELP
- alpha = 0.5: IC decays at same rate → no benefit from longer holds
- alpha > 0.5: IC decays faster → shorter horizons better

Features with alpha < 0.3 are prime candidates for profitable longer-hold strategies.
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

EXCLUDE_FEATURES = [0, 3, 8, 9]  # mid, microprice, best_bid, best_ask

# Horizons to test
HORIZONS = {
    '1s':  100,
    '5s':  500,
    '10s': 1000,
    '30s': 3000,
    '60s': 6000,
}

COST_TICKS = 0.376  # Commission only: $4.70 RT / $12.50 per tick. Spread is variable.
TICK_SIZE = 0.25


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-days', type=int, default=50)
    parser.add_argument('--subsample', type=int, default=100)
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

    print(f"Full IC Decay Analysis - {len(files)} days, {len(HORIZONS)} horizons")
    print(f"Testing all features for slow-decay candidates")

    # Pre-load all days
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

        keep = [j for j in range(raw.shape[1]) if j not in EXCLUDE_FEATURES]
        features = raw[:, keep]
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        day_features.append(features)
        day_mids.append(mid)
        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}]")
        del raw; gc.collect()

    n_features = day_features[0].shape[1]
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(n_features)]
    print(f"Loaded {len(files)} days, {n_features} features ({time.time()-t0:.1f}s)")

    # Compute IC for each feature × horizon
    print("\nComputing IC matrix...")
    t0 = time.time()

    # ic_matrix[feature_idx][horizon_name] = list of daily ICs
    ic_matrix = [[[] for _ in HORIZONS] for _ in range(n_features)]

    for day_idx in range(len(files)):
        features = day_features[day_idx]
        mid = day_mids[day_idx]
        N = len(mid)

        for h_idx, (h_name, h_bars) in enumerate(HORIZONS.items()):
            if N <= h_bars + 100:
                continue

            # Forward return
            fwd_ret = (mid[h_bars:N] - mid[:N-h_bars]) / np.where(mid[:N-h_bars] > 0, mid[:N-h_bars], 1.0)

            # Subsample (avoid autocorrelation)
            spacing = max(args.subsample, h_bars)
            indices = np.arange(0, N - h_bars, spacing)
            if len(indices) < 30:
                continue

            y = fwd_ret[indices - (indices >= len(fwd_ret)).astype(int)]
            # Clamp indices
            indices = indices[indices < len(fwd_ret)]
            if len(indices) < 30:
                continue
            y = fwd_ret[indices]
            valid_y = np.isfinite(y)

            for f_idx in range(n_features):
                x = features[indices, f_idx]
                mask = valid_y & np.isfinite(x)
                if mask.sum() < 30:
                    continue
                ic, _ = spearmanr(x[mask], y[mask])
                if np.isfinite(ic):
                    ic_matrix[f_idx][h_idx].append(ic)

        if (day_idx + 1) % 10 == 0:
            print(f"  [{day_idx+1}/{len(files)}] ({time.time()-t0:.0f}s)")

    print(f"IC computation done ({time.time()-t0:.1f}s)")

    # Analyze results
    horizon_names = list(HORIZONS.keys())
    horizon_seconds = [h / 100 for h in HORIZONS.values()]

    # Compute mean IC and decay alpha for each feature
    feature_stats = []
    for f_idx in range(n_features):
        ics = {}
        for h_idx, h_name in enumerate(horizon_names):
            daily_ics = ic_matrix[f_idx][h_idx]
            if daily_ics:
                ics[h_name] = np.mean(daily_ics)
            else:
                ics[h_name] = 0.0

        ic_1s = abs(ics.get('1s', 0.0))
        if ic_1s < 0.01:  # Skip features with negligible 1s IC
            continue

        # Compute decay alpha: fit log(IC) vs log(horizon)
        log_h = []
        log_ic = []
        for h_idx, h_name in enumerate(horizon_names):
            ic_val = abs(ics.get(h_name, 0.0))
            if ic_val > 0.001:
                log_h.append(np.log(horizon_seconds[h_idx]))
                log_ic.append(np.log(ic_val))

        alpha = np.nan
        if len(log_h) >= 3:
            # Linear fit: log(IC) = -alpha * log(h) + const
            coeffs = np.polyfit(log_h, log_ic, 1)
            alpha = -coeffs[0]

        # Profit potential at each horizon
        profit_pots = {}
        for h_name in horizon_names:
            ic_val = abs(ics.get(h_name, 0.0))
            h_sec = HORIZONS[h_name] / 100
            profit_pots[h_name] = ic_val * np.sqrt(h_sec)

        # Best horizon (highest profit potential)
        best_h = max(profit_pots, key=profit_pots.get)

        feature_stats.append({
            'name': feature_names[f_idx],
            'idx': f_idx,
            'ic_1s': ic_1s,
            'ics': ics,
            'alpha': alpha,
            'profit_pots': profit_pots,
            'best_horizon': best_h,
            'best_profit_pot': profit_pots[best_h],
        })

    # Sort by decay alpha (slowest decay first)
    feature_stats.sort(key=lambda x: x['alpha'] if np.isfinite(x['alpha']) else 999)

    print(f"\n{'='*100}")
    print(f"TOP 20 SLOWEST-DECAYING FEATURES (alpha < 0.5 means longer holds help)")
    print(f"{'='*100}")
    header = f"{'Feature':35s} {'alpha':>6s} {'IC@1s':>7s}"
    for h in horizon_names[1:]:
        header += f" {'IC@'+h:>8s}"
    header += f" {'Best H':>7s} {'ProfPot':>8s}"
    print(header)
    print("-" * 100)

    for fs in feature_stats[:20]:
        row = f"{fs['name']:35s} {fs['alpha']:+6.3f} {fs['ic_1s']:+7.4f}"
        for h in horizon_names[1:]:
            row += f" {abs(fs['ics'].get(h, 0)):+8.4f}"
        row += f" {fs['best_horizon']:>7s} {fs['best_profit_pot']:+8.4f}"
        print(row)

    print(f"\n{'='*100}")
    print(f"TOP 20 BY PROFIT POTENTIAL (IC * sqrt(horizon))")
    print(f"{'='*100}")
    feature_stats_by_pp = sorted(feature_stats, key=lambda x: x['best_profit_pot'], reverse=True)

    print(header)
    print("-" * 100)
    for fs in feature_stats_by_pp[:20]:
        row = f"{fs['name']:35s} {fs['alpha']:+6.3f} {fs['ic_1s']:+7.4f}"
        for h in horizon_names[1:]:
            row += f" {abs(fs['ics'].get(h, 0)):+8.4f}"
        row += f" {fs['best_horizon']:>7s} {fs['best_profit_pot']:+8.4f}"
        print(row)

    # Check if ANY feature at ANY horizon could be profitable
    print(f"\n{'='*100}")
    print(f"PROFITABILITY CHECK — Can ANY feature at ANY horizon overcome costs?")
    print(f"{'='*100}")

    # Empirical move sizes
    move_sizes = {'1s': 1.73, '5s': 3.96, '10s': 5.61, '30s': 9.91, '60s': 13.81}

    any_profitable = False
    for fs in feature_stats_by_pp[:30]:
        for h in horizon_names:
            ic_val = abs(fs['ics'].get(h, 0))
            avg_move = move_sizes.get(h, 0)
            expected_ret = ic_val * avg_move
            if expected_ret > COST_TICKS * 0.5:  # Within 2x of breakeven
                print(f"  {fs['name']:35s} @ {h:>4s}: IC={ic_val:.4f}, E[ret]={expected_ret:.2f}t "
                      f"(need {COST_TICKS:.2f}t, gap={COST_TICKS/expected_ret:.1f}x)")
                if expected_ret >= COST_TICKS:
                    print(f"    >>> PROFITABLE!")
                    any_profitable = True

    if not any_profitable:
        print(f"\n  NO feature at any horizon reaches breakeven.")
        print(f"  Closest approach is listed above (within 2x of breakeven).")

    # Save results
    out = {
        'n_days': len(files),
        'n_features_tested': len(feature_stats),
        'horizons': horizon_names,
        'top_slow_decay': [{
            'name': fs['name'],
            'alpha': float(fs['alpha']) if np.isfinite(fs['alpha']) else None,
            'ic_1s': float(fs['ic_1s']),
            'ics': {h: float(fs['ics'].get(h, 0)) for h in horizon_names},
        } for fs in feature_stats[:30]],
        'top_profit_potential': [{
            'name': fs['name'],
            'best_horizon': fs['best_horizon'],
            'best_profit_pot': float(fs['best_profit_pot']),
        } for fs in feature_stats_by_pp[:30]],
    }
    out_path = ROOT / 'alpha_discovery' / 'results' / 'full_ic_decay.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
