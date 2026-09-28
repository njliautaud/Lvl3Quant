#!/usr/bin/env python3
"""
Intraday Vol Prediction — Can MBO features predict HOURLY realized vol?

KEY INSIGHT: Our 1s rvol prediction (IC=0.67) doesn't aggregate to daily.
But 0DTE options only need intraday prediction (1-4 hours).

This script tests:
1. Can current book state predict next 1hr / 2hr / 4hr realized vol?
2. Walk-forward LightGBM with 10-second sampling
3. Compare to naive (trailing rvol) baseline
4. Estimate economic value for 0DTE straddle trading

The critical question: does IC stay high enough at hourly horizons
to beat the options market's intraday IV pricing?

Usage:
  python intraday_vol_prediction.py
"""

import gc
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    format='%(asctime)s [intra_vol] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
log = logging.getLogger('intra_vol')

MBO_DIR = ROOT_DIR / 'data' / 'processed' / 'mbo_features_cache'
RESULTS_DIR = ROOT_DIR / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Price feature indices to exclude
EXCLUDE_FEATURES = [0, 3, 8, 9]  # mid, microprice, best_bid, best_ask

# MBO bars are 100ms each (234K bars/day ÷ 23400 sec/day = 10 bars/sec)
BARS_PER_SECOND = 10

# Horizons to test (in 100ms bars)
HORIZONS = {
    '30min': 1800 * BARS_PER_SECOND,    # 18,000 bars
    '1hr':   3600 * BARS_PER_SECOND,    # 36,000 bars
    '2hr':   7200 * BARS_PER_SECOND,    # 72,000 bars
    '4hr':   14400 * BARS_PER_SECOND,   # 144,000 bars
}

# Sampling interval: every 10 seconds = 100 bars
SAMPLE_INTERVAL = 100


def load_day(fpath):
    """Load features and mid prices."""
    data = np.load(str(fpath))
    raw = data['mbo_features']
    mid = raw[:, 0].copy()

    # Forward-fill NaN
    mask = np.isnan(mid)
    if mask.any():
        first_valid = np.argmax(~mask)
        if first_valid > 0:
            mid[:first_valid] = mid[first_valid]
        for i in range(1, len(mid)):
            if np.isnan(mid[i]):
                mid[i] = mid[i - 1]

    # Features (exclude price levels)
    keep = [i for i in range(raw.shape[1]) if i not in EXCLUDE_FEATURES]
    features = raw[:, keep]
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    del raw
    return features, mid


def compute_forward_rvol(mid, horizon_bars):
    """Compute forward realized vol for each bar, as annualized pct.
    Uses 1-second (100-bar) returns summed over the horizon window.
    """
    N = len(mid)
    step = BARS_PER_SECOND  # 1 second = 10 bars at 100ms/bar
    # 1s prices
    prices = mid[::step]
    n_prices = len(prices)
    returns = np.diff(np.log(prices))
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    horizon_seconds = horizon_bars // BARS_PER_SECOND
    n_ret = len(returns)

    if n_ret < horizon_seconds:
        return np.full(N, np.nan)

    # Compute rolling rvol using cumsum trick
    r2 = returns ** 2
    cumsum_r2 = np.concatenate(([0.0], np.cumsum(r2)))

    # For each 1s index, compute rvol over next horizon_seconds
    n_valid = n_ret - horizon_seconds + 1
    sum_r2 = cumsum_r2[horizon_seconds:horizon_seconds + n_valid] - cumsum_r2[:n_valid]
    # Annualize: sqrt(sum_r2 / horizon_seconds) * sqrt(seconds_per_year)
    seconds_per_year = 23400 * 252
    rvol = np.sqrt(sum_r2 / horizon_seconds) * np.sqrt(seconds_per_year) * 100

    # Map back to 10ms bar indices (each 1s price maps to bar index * 100)
    fwd_rvol = np.full(N, np.nan)
    for i in range(min(n_valid, N // step)):
        bar_idx = i * step
        if bar_idx < N:
            fwd_rvol[bar_idx] = rvol[i]

    return fwd_rvol


def compute_trailing_rvol(mid, lookback_bars):
    """Compute trailing realized vol (naive baseline predictor)."""
    N = len(mid)
    step = BARS_PER_SECOND  # 10 bars = 1 second
    prices = mid[::step]
    returns = np.diff(np.log(prices))
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    lookback_seconds = lookback_bars // BARS_PER_SECOND
    n_ret = len(returns)

    if n_ret < lookback_seconds:
        return np.full(N, np.nan)

    r2 = returns ** 2
    cumsum_r2 = np.concatenate(([0.0], np.cumsum(r2)))

    # Vectorized: compute rolling sum of r2 over lookback window
    n_valid = n_ret - lookback_seconds + 1
    sum_r2 = cumsum_r2[lookback_seconds:lookback_seconds + n_valid] - cumsum_r2[:n_valid]
    seconds_per_year = 23400 * 252
    rvol = np.sqrt(sum_r2 / lookback_seconds) * np.sqrt(seconds_per_year) * 100

    trail_rvol = np.full(N, np.nan)
    for i in range(n_valid):
        bar_idx = (i + lookback_seconds) * step  # trailing: vol computed up to this point
        if bar_idx < N:
            trail_rvol[bar_idx] = rvol[i]

    return trail_rvol


def main():
    log.info("=" * 70)
    log.info("INTRADAY VOL PREDICTION — Hourly Horizons for 0DTE Options")
    log.info("=" * 70)

    files = sorted(MBO_DIR.glob('*_mbo_features.npz'))
    if not files:
        log.error(f"No files in {MBO_DIR}")
        return

    n_days = len(files)
    log.info(f"Found {n_days} MBO days")
    log.info(f"Horizons: {list(HORIZONS.keys())}")
    log.info(f"Sample interval: {SAMPLE_INTERVAL} bars = {SAMPLE_INTERVAL/100:.0f}s")

    results = {}

    for hz_name, hz_bars in HORIZONS.items():
        log.info(f"\n{'='*70}")
        log.info(f"HORIZON: {hz_name} ({hz_bars//BARS_PER_SECOND}s = {hz_bars//BARS_PER_SECOND/3600:.1f}hr)")
        log.info(f"{'='*70}")

        # Phase 1: Compute raw feature-target IC (no model, just correlation)
        log.info(f"\n  Phase 1: Raw feature correlations ({hz_name})")

        all_ics_per_feat = None
        n_feat = None
        day_ics_naive = []  # trailing rvol → forward rvol

        for i, fpath in enumerate(files):
            date_str = fpath.stem.replace('_mbo_features', '')
            try:
                features, mid = load_day(fpath)
                fwd_vol = compute_forward_rvol(mid, hz_bars)
                trail_vol = compute_trailing_rvol(mid, hz_bars)  # Same window as lookback

                # Sample at 10s intervals
                indices = np.arange(0, len(features), SAMPLE_INTERVAL)
                X = features[indices]
                y_fwd = fwd_vol[indices]
                y_trail = trail_vol[indices]

                # Valid mask
                valid = (np.isfinite(y_fwd) & np.all(np.isfinite(X), axis=1)
                         & (y_fwd > 0))
                if valid.sum() < 50:
                    continue

                X_v = X[valid]
                y_v = y_fwd[valid]

                # IC per feature
                if n_feat is None:
                    n_feat = X_v.shape[1]
                    all_ics_per_feat = []

                feat_ics = []
                for j in range(n_feat):
                    if np.std(X_v[:, j]) > 0:
                        ic, _ = spearmanr(X_v[:, j], y_v)
                        feat_ics.append(ic if np.isfinite(ic) else 0.0)
                    else:
                        feat_ics.append(0.0)
                all_ics_per_feat.append(feat_ics)

                # Naive baseline IC (trailing rvol → forward rvol)
                valid_both = valid & np.isfinite(y_trail)
                if valid_both.sum() > 50:
                    ic_naive, _ = spearmanr(y_trail[valid_both],
                                            y_fwd[valid_both])
                    if np.isfinite(ic_naive):
                        day_ics_naive.append(ic_naive)

                del features, mid, fwd_vol, trail_vol
                gc.collect()

            except Exception as e:
                if i < 3:
                    log.warning(f"    Error {date_str}: {e}")
                    traceback.print_exc()
                continue

        # Summarize raw ICs
        if all_ics_per_feat:
            ic_matrix = np.array(all_ics_per_feat)  # (n_days, n_features)
            mean_ics = np.mean(ic_matrix, axis=0)
            top5_idx = np.argsort(np.abs(mean_ics))[-5:][::-1]

            log.info(f"  Top 5 features by mean |IC| with {hz_name} forward rvol:")
            for rank, j in enumerate(top5_idx):
                m = mean_ics[j]
                std = np.std(ic_matrix[:, j])
                t = m / (std / np.sqrt(len(ic_matrix))) if std > 0 else 0
                pct = np.mean(ic_matrix[:, j] > 0) * 100
                log.info(f"    {rank+1}. feat_{j:3d}  mean_IC={m:+.4f}  t={t:+.1f}  pct+={pct:.0f}%")

            # Naive baseline
            if day_ics_naive:
                naive_mean = np.mean(day_ics_naive)
                naive_t = naive_mean / (np.std(day_ics_naive) / np.sqrt(len(day_ics_naive)))
                log.info(f"\n  NAIVE BASELINE (trailing rvol → forward rvol):")
                log.info(f"    Mean IC: {naive_mean:+.4f}  t={naive_t:.1f}  n={len(day_ics_naive)}")

        # Phase 2: Walk-forward LightGBM
        log.info(f"\n  Phase 2: Walk-forward LightGBM ({hz_name})")

        try:
            import lightgbm as lgb
        except ImportError:
            log.error("  LightGBM not installed")
            continue

        lgbm_params = {
            'objective': 'regression',
            'metric': 'mse',
            'learning_rate': 0.05,
            'num_leaves': 63,
            'max_depth': 6,
            'min_child_samples': 200,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'n_jobs': 8,
            'seed': 42,
        }

        MIN_TRAIN_DAYS = 15

        # Pre-load all days
        day_data = []
        for fpath in files:
            date_str = fpath.stem.replace('_mbo_features', '')
            try:
                features, mid = load_day(fpath)
                fwd_vol = compute_forward_rvol(mid, hz_bars)

                indices = np.arange(0, len(features), SAMPLE_INTERVAL)
                X = features[indices]
                y = fwd_vol[indices]

                valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & (y > 0)
                X = X[valid].astype(np.float32)
                y = y[valid].astype(np.float32)

                if len(y) > 50:
                    day_data.append({'date': date_str, 'X': X, 'y': y})

                del features, mid, fwd_vol
                gc.collect()
            except:
                continue

        log.info(f"  Loaded {len(day_data)} valid days")

        # Walk-forward
        all_ics = []
        all_naive_ics = []

        for test_idx in range(MIN_TRAIN_DAYS, len(day_data)):
            train_days = day_data[:test_idx]
            test_day = day_data[test_idx]

            # Subsample training data if too large
            X_parts = [d['X'] for d in train_days]
            y_parts = [d['y'] for d in train_days]
            X_train = np.vstack(X_parts)
            y_train = np.concatenate(y_parts)

            # Cap training size
            if len(X_train) > 200_000:
                step = len(X_train) // 200_000
                X_train = X_train[::step]
                y_train = y_train[::step]

            X_test = test_day['X']
            y_test = test_day['y']

            if len(y_test) < 20:
                continue

            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(lgbm_params, dtrain, num_boost_round=100)
            preds = model.predict(X_test)

            ic, _ = spearmanr(preds, y_test)
            if np.isfinite(ic):
                all_ics.append(ic)

            # Naive: predict y_test with mean of y_train[-1] (last day's mean)
            naive_pred = np.mean(train_days[-1]['y'])
            # Actually, better naive: use trailing mean of y_test itself
            # But that's cheating. Use rolling mean from last training day.
            ic_naive, _ = spearmanr(np.full_like(y_test, naive_pred), y_test)
            # Note: constant prediction → IC=0 by definition

            if (test_idx - MIN_TRAIN_DAYS) % 10 == 0:
                log.info(f"    [{test_idx+1}/{len(day_data)}] {test_day['date']}  "
                        f"IC={ic:+.4f}  running_mean={np.mean(all_ics):+.4f}")

            del X_train, y_train, dtrain, model
            gc.collect()

        if all_ics:
            ics = np.array(all_ics)
            mean_ic = np.nanmean(ics)
            std_ic = np.nanstd(ics)
            t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else 0
            pct_pos = np.nanmean(ics > 0) * 100

            log.info(f"\n  {'='*60}")
            log.info(f"  {hz_name} WALK-FORWARD RESULTS:")
            log.info(f"  {'='*60}")
            log.info(f"    N folds: {len(ics)}")
            log.info(f"    Mean IC: {mean_ic:+.4f}")
            log.info(f"    Std IC:  {std_ic:.4f}")
            log.info(f"    t-stat:  {t_stat:.2f}")
            log.info(f"    Pct positive: {pct_pos:.1f}%")
            log.info(f"    Min IC: {np.nanmin(ics):+.4f}")
            log.info(f"    Max IC: {np.nanmax(ics):+.4f}")

            results[hz_name] = {
                'n_folds': len(ics),
                'mean_ic': float(mean_ic),
                'std_ic': float(std_ic),
                't_stat': float(t_stat),
                'pct_positive': float(pct_pos),
                'fold_ics': ics.tolist(),
            }

            # Economic assessment for 0DTE
            log.info(f"\n    0DTE ECONOMICS ({hz_name}):")
            # SPY 0DTE ATM straddle cost ~ 0.3-0.5% of spot for few hours to expiry
            # Bid-ask ~ $0.02-0.05 per leg = $0.04-0.10 per straddle
            spy_price = 575  # approximate
            straddle_pct = {'30min': 0.001, '1hr': 0.002, '2hr': 0.003, '4hr': 0.005}
            straddle_cost = spy_price * straddle_pct.get(hz_name, 0.003) * 100  # per contract
            spread_cost = 5  # $0.05 per contract, very tight on SPY
            total_cost = spread_cost * 2  # entry + exit

            # Edge from vol prediction
            mean_rvol = 27.0  # annualized
            vol_of_vol = 13.0
            # IC → edge in vol points
            edge_vol = mean_ic * vol_of_vol * 0.3  # very conservative
            # Convert to straddle P&L: edge as fraction of straddle price
            vega_0dte = straddle_cost * 0.3  # rough vega as % of straddle
            edge_dollar = edge_vol * vega_0dte / 100

            log.info(f"      Straddle cost: ${straddle_cost:.0f}")
            log.info(f"      Spread cost RT: ${total_cost:.0f}")
            log.info(f"      Predicted edge: {edge_vol:.2f} vol pts → ${edge_dollar:.0f}/trade")
            log.info(f"      Edge vs cost: {'VIABLE' if edge_dollar > total_cost else 'NOT VIABLE'}")

        # Clear day data
        del day_data
        gc.collect()

    # Summary across all horizons
    log.info(f"\n{'='*70}")
    log.info(f"SUMMARY: Intraday Vol Prediction IC by Horizon")
    log.info(f"{'='*70}")
    log.info(f"  {'Horizon':<10}  {'Mean IC':>8}  {'t-stat':>7}  {'Pct+':>5}  {'N':>4}")
    log.info(f"  {'-'*45}")
    for hz_name in HORIZONS:
        if hz_name in results:
            r = results[hz_name]
            log.info(f"  {hz_name:<10}  {r['mean_ic']:+.4f}   {r['t_stat']:+.1f}    "
                    f"{r['pct_positive']:.0f}%   {r['n_folds']:>4}")

    # Compare to known results
    log.info(f"\n  Comparison (from previous work):")
    log.info(f"  NOTE: MBO bars are 100ms. Previous '1s' labels were actually 10s!")
    log.info(f"  10s (labeled 1s)   IC=+0.674   t=52.8    100%%   85")
    log.info(f"  100s (labeled 10s) IC=NaN (numerical instability)")
    log.info(f"  300s (labeled 30s) IC=NaN (numerical instability)")
    log.info(f"  daily              IC=+0.423  (LightGBM walk-forward)")

    log.info(f"\n  KEY QUESTION: Does IC decay gradually from 0.67 (10s actual) to 0.42 (daily)?")
    log.info(f"  Or does it collapse at some intermediate horizon?")
    log.info(f"  If IC > 0.3 at 1-4hr, 0DTE options may be viable.")

    # Save
    out_path = RESULTS_DIR / f'intraday_vol_pred_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
