"""
Meta-Learner: Walk-Forward Ridge Regression on Microstructure Signals
=====================================================================
Instead of testing signals individually with static thresholds, this trains
a simple linear model that learns the optimal COMBINATION of signals.

Walk-forward protocol:
  - Train on days 1..N, predict day N+1
  - Retrain on days 1..N+1, predict day N+2
  - etc.

The model learns which signals matter and outputs a combined prediction
that we then feed through the Rust MBO sim.

This runs on PC CPU (no GPU needed). Uses 340 MBO features or 96 global features.

Usage:
    python alpha_discovery/meta_learner_walkforward.py --mode global --train-days 20
    python alpha_discovery/meta_learner_walkforward.py --mode mbo --train-days 30
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
SNAP_DIR = ROOT / 'data' / 'processed' / 'medium_snapshots_cache'
MBO_FEAT_DIR = ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIGNAL_DIR = ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR = ROOT / 'alpha_discovery' / 'results'
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Horizons to test (in 100ms bars)
HORIZONS = {
    '3s': 30,
    '10s': 100,
    '30s': 300,
}


def load_day_features(date_str: str, mode: str = 'global'):
    """Load features and mid prices for one day."""
    if mode == 'global':
        f = SNAP_DIR / f'{date_str}_snapshots.npz'
        if not f.exists():
            return None, None
        data = np.load(str(f))
        return data['global_features'], data['mid_prices']
    else:
        f = MBO_FEAT_DIR / f'{date_str}_features.npz'
        if not f.exists():
            # Try alternate naming
            candidates = list(MBO_FEAT_DIR.glob(f'*{date_str}*.npz'))
            if not candidates:
                return None, None
            f = candidates[0]
        data = np.load(str(f))
        feats = data.get('mbo_features', data.get('features'))
        # Need mid prices from snapshots
        sf = SNAP_DIR / f'{date_str}_snapshots.npz'
        if sf.exists():
            mid = np.load(str(sf))['mid_prices']
        else:
            mid = None
        return feats, mid


def compute_forward_returns(mid_prices: np.ndarray, horizon_bars: int) -> np.ndarray:
    """Compute forward returns at given horizon."""
    n = len(mid_prices)
    ret = np.full(n, np.nan)
    ret[:n - horizon_bars] = (mid_prices[horizon_bars:] - mid_prices[:n - horizon_bars])
    return ret  # In price points (not pct)


def run_walkforward(dates: list, mode: str, train_days: int, horizon_name: str,
                    alpha: float = 1.0):
    """Run walk-forward training and generate predictions."""
    horizon_bars = HORIZONS[horizon_name]

    # Load all data
    print(f"Loading {len(dates)} days of {mode} features...")
    all_features = []
    all_returns = []
    all_mids = []
    valid_dates = []

    for d in dates:
        feats, mid = load_day_features(d, mode)
        if feats is None or mid is None:
            continue
        fwd_ret = compute_forward_returns(mid, horizon_bars)
        all_features.append(feats)
        all_returns.append(fwd_ret)
        all_mids.append(mid)
        valid_dates.append(d)

    n_days = len(valid_dates)
    print(f"Loaded {n_days} days, {mode} features shape: {all_features[0].shape}")

    if n_days < train_days + 5:
        print(f"ERROR: Need at least {train_days + 5} days, only have {n_days}")
        return None

    # Walk-forward
    predictions_by_day = {}
    ics_by_day = {}

    for test_idx in range(train_days, n_days):
        test_date = valid_dates[test_idx]
        train_start = max(0, test_idx - train_days)

        # Build training set
        X_train_parts = []
        y_train_parts = []
        for i in range(train_start, test_idx):
            feats = all_features[i]
            rets = all_returns[i]
            mask = np.isfinite(rets) & np.all(np.isfinite(feats), axis=1)
            X_train_parts.append(feats[mask])
            y_train_parts.append(rets[mask])

        X_train = np.vstack(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Downsample if too large (>500K samples causes OOM on SVD)
        if len(y_train) > 500000:
            idx = np.random.RandomState(42).choice(len(y_train), 500000, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train Ridge regression (cholesky solver avoids SVD OOM)
        model = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky')
        model.fit(X_train_scaled, y_train)

        # Predict on test day
        X_test = all_features[test_idx]
        X_test_scaled = scaler.transform(X_test)
        preds = model.predict(X_test_scaled)

        # Compute IC on test day
        test_rets = all_returns[test_idx]
        mask = np.isfinite(test_rets) & np.isfinite(preds)
        if mask.sum() > 100:
            p, r = preds[mask], test_rets[mask]
            p_dm = p - p.mean()
            r_dm = r - r.mean()
            denom = np.sqrt((p_dm**2).sum() * (r_dm**2).sum())
            ic = float((p_dm * r_dm).sum() / max(denom, 1e-12))
        else:
            ic = 0.0

        predictions_by_day[test_date] = preds
        ics_by_day[test_date] = ic

        # Save prediction file for MBO sim
        out = SIGNAL_DIR / f'meta_{mode}_{horizon_name}_{test_date}.npz'
        np.savez_compressed(str(out), predictions=preds.astype(np.float64),
                           mid_prices=all_mids[test_idx])

        if (test_idx - train_days) % 5 == 0:
            recent_ics = [ics_by_day[valid_dates[j]]
                         for j in range(max(train_days, test_idx-5), test_idx+1)]
            print(f"  Day {test_idx - train_days + 1}/{n_days - train_days}: "
                  f"{test_date} IC={ic:+.4f} (recent avg: {np.mean(recent_ics):+.4f})")

    # Summary
    test_dates = [valid_dates[i] for i in range(train_days, n_days)]
    ics = [ics_by_day[d] for d in test_dates]

    return {
        'mode': mode,
        'horizon': horizon_name,
        'alpha': alpha,
        'train_days': train_days,
        'test_dates': test_dates,
        'ics': ics,
        'mean_ic': float(np.mean(ics)),
        'std_ic': float(np.std(ics)),
        'positive_pct': float((np.array(ics) > 0).mean()),
        't_stat': float(np.mean(ics) / max(np.std(ics) / np.sqrt(len(ics)), 1e-6)),
        'prediction_files': [str(SIGNAL_DIR / f'meta_{mode}_{horizon_name}_{d}.npz')
                            for d in test_dates],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['global', 'mbo'], default='global')
    parser.add_argument('--train-days', type=int, default=20)
    parser.add_argument('--alpha', type=float, default=1.0, help='Ridge regularization')
    parser.add_argument('--horizons', type=str, default='3s,10s,30s')
    args = parser.parse_args()

    # Find available dates
    snap_dates = sorted(f.stem.replace('_snapshots', '')
                        for f in SNAP_DIR.glob('*_snapshots.npz'))
    print(f"Available dates: {len(snap_dates)}")

    horizons = args.horizons.split(',')
    all_results = []

    for hz in horizons:
        print(f"\n{'='*60}")
        print(f"META-LEARNER: {args.mode} features, {hz} horizon, "
              f"alpha={args.alpha}, train={args.train_days} days")
        print(f"{'='*60}")

        result = run_walkforward(snap_dates, args.mode, args.train_days, hz, args.alpha)
        if result:
            all_results.append(result)
            print(f"\n  Mean IC: {result['mean_ic']:+.4f}")
            print(f"  IC std: {result['std_ic']:.4f}")
            print(f"  Positive IC days: {result['positive_pct']:.1%}")
            print(f"  t-stat: {result['t_stat']:.2f}")
            print(f"  Prediction files: {len(result['prediction_files'])}")

    # Save results
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'meta_learner_{args.mode}_{ts}.json'
    output = {
        'experiment': 'meta_learner_walkforward',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'config': {'mode': args.mode, 'train_days': args.train_days, 'alpha': args.alpha},
        'results': all_results,
    }
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON: Meta-Learner vs Raw Signals")
    print(f"{'='*60}")
    print("Raw signal ICs (from test_raw_signals.py):")
    print("  microprice_dev:      IC=0.074 @ 10s (92% consistency)")
    print("  pressure_imbalance:  IC=0.047 @ 10s (86% consistency)")
    print("  ask_slope:           IC=0.044 @ 10s (96% consistency)")
    print()
    for r in all_results:
        print(f"Meta-learner ({r['mode']}, {r['horizon']}): "
              f"IC={r['mean_ic']:+.4f} ({r['positive_pct']:.0%} positive, "
              f"t={r['t_stat']:.2f})")
