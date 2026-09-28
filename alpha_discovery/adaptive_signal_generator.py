"""
Adaptive Signal Generator: Self-adjusting thresholds + regime detection
=======================================================================
Instead of fixed thresholds, generates signals using:
1. Rolling percentile thresholds (95th pctile of last N bars)
2. Regime-conditional signals (different behavior in vol/calm regimes)
3. Signal clustering (only trade in favorable market states)

Outputs prediction NPZ files compatible with the Rust MBO sim.

Usage:
    python alpha_discovery/adaptive_signal_generator.py --method percentile
    python alpha_discovery/adaptive_signal_generator.py --method regime
    python alpha_discovery/adaptive_signal_generator.py --method cluster
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
SNAP_DIR = ROOT / 'data' / 'processed' / 'medium_snapshots_cache'
SIGNAL_DIR = ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature indices (from engineering.py)
IDX_MID = 0
IDX_SPREAD = 1
IDX_IMBALANCE = 2
IDX_MICROPRICE = 3
IDX_TRADE_IMBALANCE = 10
IDX_PRESSURE_IMBALANCE = 20
IDX_BID_SLOPE = 22
IDX_ASK_SLOPE = 23
IDX_SPREAD_TICKS = 24


def generate_percentile_signal(gf: np.ndarray, mid: np.ndarray,
                                lookback: int = 5000,
                                percentile: float = 90.0) -> np.ndarray:
    """
    Adaptive threshold using rolling percentile.
    Signal = microprice_dev, but only when it's in the extreme tail
    of recent history. Self-adjusts to current market conditions.
    """
    n = len(gf)
    microprice_dev = gf[:, IDX_MICROPRICE] - mid
    pressure = gf[:, IDX_PRESSURE_IMBALANCE]

    # Combine signals
    combined = 0.6 * microprice_dev / max(np.std(microprice_dev[:1000]), 1e-8) + \
               0.4 * pressure / max(np.std(pressure[:1000]), 1e-8)

    signal = np.zeros(n)

    for i in range(lookback, n):
        window = combined[i - lookback:i]
        upper = np.percentile(window, percentile)
        lower = np.percentile(window, 100 - percentile)

        if combined[i] > upper:
            # Strong bullish — magnitude proportional to how extreme
            signal[i] = (combined[i] - upper) / max(abs(upper), 1e-8)
        elif combined[i] < lower:
            # Strong bearish
            signal[i] = (combined[i] - lower) / max(abs(lower), 1e-8)
        # else: signal stays 0 (no trade)

    return signal


def generate_regime_signal(gf: np.ndarray, mid: np.ndarray,
                           vol_lookback: int = 3000,
                           calm_mult: float = 1.0,
                           volatile_mult: float = 0.5) -> np.ndarray:
    """
    Regime-conditional: detect volatility regime and adjust signal strength.
    In calm markets: stronger signals (mean reversion works)
    In volatile markets: weaker signals (momentum dominates, harder to trade)
    """
    n = len(gf)
    microprice_dev = gf[:, IDX_MICROPRICE] - mid
    pressure = gf[:, IDX_PRESSURE_IMBALANCE]

    # Compute rolling volatility
    returns = np.diff(mid, prepend=mid[0]) / np.maximum(mid, 1.0)
    vol = np.zeros(n)
    for i in range(vol_lookback, n):
        vol[i] = np.std(returns[i - vol_lookback:i])

    # Median vol as regime boundary
    median_vol = np.median(vol[vol_lookback:])

    # Combined signal with regime adjustment
    base_signal = (0.6 * microprice_dev / max(np.std(microprice_dev[:2000]), 1e-8) +
                   0.4 * pressure / max(np.std(pressure[:2000]), 1e-8))

    signal = np.zeros(n)
    for i in range(vol_lookback, n):
        if vol[i] < median_vol:
            signal[i] = base_signal[i] * calm_mult
        else:
            signal[i] = base_signal[i] * volatile_mult

    return signal


def generate_cluster_signal(gf: np.ndarray, mid: np.ndarray,
                            n_clusters: int = 5,
                            lookback: int = 2000) -> np.ndarray:
    """
    Signal clustering: cluster recent market states, only trade
    in clusters that historically had positive forward returns.
    """
    from sklearn.cluster import MiniBatchKMeans

    n = len(gf)

    # Use key features for clustering
    features = np.column_stack([
        gf[:, IDX_IMBALANCE],
        gf[:, IDX_PRESSURE_IMBALANCE],
        gf[:, IDX_SPREAD],
        gf[:, IDX_TRADE_IMBALANCE],
        gf[:, IDX_BID_SLOPE],
        gf[:, IDX_ASK_SLOPE],
    ])

    # Directional signal
    microprice_dev = gf[:, IDX_MICROPRICE] - mid

    # Forward returns (10s = 100 bars)
    fwd_ret = np.zeros(n)
    fwd_ret[:n-100] = mid[100:] - mid[:n-100]

    signal = np.zeros(n)

    # Walk-forward clustering
    for start in range(lookback, n, lookback):
        end = min(start + lookback, n)
        train_start = max(0, start - lookback)

        # Train clustering on recent data
        train_feats = features[train_start:start]
        mask = np.all(np.isfinite(train_feats), axis=1)
        if mask.sum() < 100:
            continue

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_feats[mask])

        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        km.fit(train_scaled)
        train_labels = km.labels_

        # Compute avg forward return per cluster (from training period)
        train_rets = fwd_ret[train_start:start][mask]
        cluster_returns = {}
        for c in range(n_clusters):
            c_mask = train_labels == c
            if c_mask.sum() > 10:
                cluster_returns[c] = float(np.mean(train_rets[c_mask]))

        # Apply to test period
        test_feats = features[start:end]
        test_mask = np.all(np.isfinite(test_feats), axis=1)
        if test_mask.sum() == 0:
            continue

        test_scaled = scaler.transform(test_feats[test_mask])
        test_labels = km.predict(test_scaled)

        # Only trade in clusters with positive historical returns
        j = 0
        for i_rel in range(end - start):
            i_abs = start + i_rel
            if not test_mask[i_rel]:
                continue
            cluster = test_labels[j]
            j += 1
            cluster_ret = cluster_returns.get(cluster, 0)
            if cluster_ret > 0:
                signal[i_abs] = microprice_dev[i_abs] * 2.0  # Amplify in good clusters
            elif cluster_ret < -0.01:
                signal[i_abs] = 0.0  # Suppress in bad clusters
            else:
                signal[i_abs] = microprice_dev[i_abs] * 0.5  # Dampen in neutral

    return signal


METHODS = {
    'percentile': generate_percentile_signal,
    'regime': generate_regime_signal,
    'cluster': generate_cluster_signal,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=list(METHODS.keys()), default='percentile')
    parser.add_argument('--max-days', type=int, default=50)
    args = parser.parse_args()

    snap_files = sorted(SNAP_DIR.glob('*_snapshots.npz'))[:args.max_days]
    print(f"Generating {args.method} signals for {len(snap_files)} days")

    signal_fn = METHODS[args.method]
    generated = 0
    all_stats = []

    for f in snap_files:
        date_str = f.stem.replace('_snapshots', '')
        data = np.load(str(f))
        gf = data['global_features']
        mid = data['mid_prices']

        signal = signal_fn(gf, mid).astype(np.float64)

        # Stats
        nonzero = np.abs(signal) > 0.01
        stats = {
            'date': date_str,
            'nonzero_pct': float(nonzero.mean()),
            'signal_std': float(np.std(signal)),
            'signal_max': float(np.max(np.abs(signal))),
        }
        all_stats.append(stats)

        # Check predictive power
        fwd_ret = np.zeros(len(mid))
        fwd_ret[:len(mid)-100] = mid[100:] - mid[:len(mid)-100]
        mask = np.isfinite(signal) & np.isfinite(fwd_ret) & (np.abs(signal) > 0.01)
        if mask.sum() > 50:
            s, r = signal[mask], fwd_ret[mask]
            s_dm = s - s.mean()
            r_dm = r - r.mean()
            denom = np.sqrt((s_dm**2).sum() * (r_dm**2).sum())
            ic = float((s_dm * r_dm).sum() / max(denom, 1e-12))
            stats['ic_10s'] = ic
        else:
            stats['ic_10s'] = 0.0

        # Save
        out = SIGNAL_DIR / f'{args.method}_{date_str}.npz'
        np.savez_compressed(str(out), predictions=signal, mid_prices=mid)
        generated += 1

        if generated % 10 == 0:
            recent_ics = [s['ic_10s'] for s in all_stats[-10:]]
            print(f"  [{generated}/{len(snap_files)}] "
                  f"mean IC={np.mean(recent_ics):+.4f}, "
                  f"nonzero={stats['nonzero_pct']:.1%}")

    # Summary
    ics = [s['ic_10s'] for s in all_stats]
    print(f"\n{'='*60}")
    print(f"{args.method.upper()} SIGNAL SUMMARY ({generated} days)")
    print(f"{'='*60}")
    print(f"Mean IC @ 10s: {np.mean(ics):+.4f}")
    print(f"IC std: {np.std(ics):.4f}")
    print(f"Positive IC days: {(np.array(ics) > 0).sum()}/{len(ics)} "
          f"({(np.array(ics) > 0).mean()*100:.0f}%)")
    print(f"t-stat: {np.mean(ics) / max(np.std(ics)/np.sqrt(len(ics)), 1e-6):.2f}")
    print(f"Avg nonzero signals: {np.mean([s['nonzero_pct'] for s in all_stats])*100:.1f}%")

    # Save summary
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'adaptive_{args.method}_{ts}.json'
    with open(out_file, 'w') as f2:
        json.dump({
            'experiment': f'adaptive_{args.method}',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'stats': all_stats,
            'summary': {
                'mean_ic': float(np.mean(ics)),
                'std_ic': float(np.std(ics)),
                'positive_pct': float((np.array(ics) > 0).mean()),
                't_stat': float(np.mean(ics) / max(np.std(ics)/np.sqrt(len(ics)), 1e-6)),
            }
        }, f2, indent=2)
    print(f"Saved: {out_file}")
    print(f"Signal files: {SIGNAL_DIR}/{args.method}_*.npz")
