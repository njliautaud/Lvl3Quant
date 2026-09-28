"""
Paranoid Leakage Investigation — ES Futures 3s Return Model

Five-test battery to rule out data leakage in the LightGBM IC=0.079-0.114 result.

Tests:
1. Shuffled-target baseline  — permutation test (10 runs, within-day shuffle)
2. Lagged-feature test        — shift features K bars forward (K=10,30,50,100)
3. Cross-day gap test         — train day D, predict day D+2 (skip a full day)
4. Feature autocorrelation    — are top features predictive because they're stale proxies?
5. Retrodiction test          — can the model predict PAST returns? (ultimate leakage check)

Usage:
    python alpha_discovery/run_leakage_audit.py
    python alpha_discovery/run_leakage_audit.py --tests shuffle lag cross_day autocorr retro
    python alpha_discovery/run_leakage_audit.py --n-perms 5 --fast
"""

import sys
import gc
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, ttest_1samp
from typing import Dict, List, Optional, Tuple

# ============================================================================
# PATH SETUP
# ============================================================================
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache, compute_return_targets,
    EXCLUDE_FEATURES_DIRECTION,
)
from alpha_discovery.run_model_refinement import walk_forward_evaluate

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'leakage_audit.log', mode='a'),
    ]
)
logger = logging.getLogger("leakage_audit")

# ============================================================================
# TOP FEATURES (from prior scan — the 9 most important)
# ============================================================================
TOP_9_FEATURES = [
    'depth_ratio_l1',
    'ask_L1_orders',
    'bid_L1_conc',
    'ask_L1_conc',
    'bid_L1_orders',
    'ofi_5',
    'total_bid_vol',
    'bid_pressure',
    'ret_100',
]

# Faster LightGBM params for permutation tests (fewer trees, still meaningful)
FAST_LGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 100,
    'verbose': -1,
    'n_jobs': -1,
}

# Standard params for single-run tests
STANDARD_LGBM_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 100,
    'verbose': -1,
    'n_jobs': -1,
}


# ============================================================================
# HELPER: safe JSON serializer
# ============================================================================

def _json_safe(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _strip_arrays(d: dict) -> dict:
    """Remove large numpy arrays from result dict before saving."""
    return {
        k: v for k, v in d.items()
        if k not in ('predictions', 'actuals', 'pred_indices')
    }


# ============================================================================
# TEST 1: SHUFFLED-TARGET BASELINE (Permutation Test)
# ============================================================================

def run_shuffled_target_test(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    hour_of_day: np.ndarray,
    n_perms: int = 10,
    use_fast_params: bool = True,
) -> dict:
    """
    Permutation test: shuffle target WITHIN each day, retrain, measure IC.

    If the real model's IC > 99th percentile of shuffled ICs, the signal is
    not purely structural (i.e., doesn't come from label leakage or feature
    co-location artifacts).

    Within-day shuffle preserves:
    - Day-boundary effects (no overnight leakage)
    - Distribution of the target within each day
    - Total number of non-NaN samples

    What it catches:
    - Structural leakage: model exploits feature-target co-location
    - Autocorrelation artifacts: model uses feature stickiness as proxy for target
    - Spurious correlation from any non-temporal source

    Expected result if signal is real: shuffled IC distribution centered near 0.
    Red flag: shuffled IC > 0.02 consistently.
    """
    logger.info("\n" + "=" * 75)
    logger.info("TEST 1: SHUFFLED-TARGET BASELINE (Permutation Test)")
    logger.info(f"  Running {n_perms} permutations, shuffling within-day")
    logger.info("=" * 75)

    params = FAST_LGBM_PARAMS if use_fast_params else STANDARD_LGBM_PARAMS
    n_days = len(day_boundaries) - 1
    N = len(target)

    perm_ics = []
    perm_fold_ics_list = []

    # First: get baseline IC on the real target
    logger.info("  Running baseline (unshuffled) for reference...")
    t0 = time.time()
    baseline_res = walk_forward_evaluate(
        features=features,
        target=target,
        day_boundaries=day_boundaries,
        feature_names=feature_names,
        model_type='lgbm',
        hour_of_day=hour_of_day,
        lgbm_params=params,
    )
    baseline_elapsed = time.time() - t0

    if 'error' in baseline_res:
        logger.error(f"  Baseline failed: {baseline_res['error']}")
        return {'error': baseline_res['error'], 'test': 'shuffled_target'}

    baseline_ic = baseline_res['ic']
    logger.info(
        f"  Baseline IC={baseline_ic:.4f} t={baseline_res['tstat']:.2f} "
        f"({baseline_elapsed:.0f}s)"
    )

    # Now run permutations
    rng = np.random.RandomState(42)

    for perm_idx in range(n_perms):
        t0_perm = time.time()

        # Create within-day shuffled target
        target_shuffled = target.copy()
        for d in range(n_days):
            d_start = day_boundaries[d]
            d_end = day_boundaries[d + 1]
            day_slice = target_shuffled[d_start:d_end]

            # Only shuffle the finite values (NaNs stay NaN at same positions)
            finite_mask = np.isfinite(day_slice)
            finite_vals = day_slice[finite_mask].copy()
            rng.shuffle(finite_vals)
            day_slice[finite_mask] = finite_vals
            target_shuffled[d_start:d_end] = day_slice

        perm_res = walk_forward_evaluate(
            features=features,
            target=target_shuffled,
            day_boundaries=day_boundaries,
            feature_names=feature_names,
            model_type='lgbm',
            hour_of_day=hour_of_day,
            lgbm_params=params,
        )
        elapsed_perm = time.time() - t0_perm

        if 'error' not in perm_res:
            ic_perm = perm_res['ic']
            perm_ics.append(float(ic_perm))
            perm_fold_ics_list.append(perm_res['fold_ics'])
            logger.info(
                f"  Perm {perm_idx+1:02d}/{n_perms}: IC={ic_perm:+.5f} "
                f"t={perm_res['tstat']:+.2f}  ({elapsed_perm:.0f}s)"
            )
        else:
            logger.warning(f"  Perm {perm_idx+1:02d}: FAILED — {perm_res['error']}")

        gc.collect()

    if not perm_ics:
        return {'error': 'All permutations failed', 'test': 'shuffled_target'}

    perm_arr = np.array(perm_ics)

    # p-value: fraction of permutations with IC >= baseline
    pvalue_empirical = float(np.mean(perm_arr >= baseline_ic))
    pvalue_twotail = float(np.mean(np.abs(perm_arr) >= abs(baseline_ic)))

    # Percentile rank of baseline IC in permutation distribution
    pct_rank = float(np.mean(perm_arr < baseline_ic)) * 100

    # Summary stats of permutation distribution
    perm_mean = float(np.mean(perm_arr))
    perm_std = float(np.std(perm_arr))
    perm_p95 = float(np.percentile(perm_arr, 95))
    perm_p99 = float(np.percentile(perm_arr, 99))
    perm_max = float(np.max(perm_arr))

    # Signal-to-noise: how many std devs is real IC above perm distribution?
    snr = (baseline_ic - perm_mean) / perm_std if perm_std > 0 else 0.0

    # Verdict
    if pvalue_empirical < 0.01:
        verdict = "CLEAN: Real IC is in top 1% of permutation distribution. No evidence of structural leakage."
    elif pvalue_empirical < 0.05:
        verdict = "LIKELY CLEAN: Real IC in top 5%. Marginal statistical significance."
    elif pvalue_empirical < 0.10:
        verdict = "SUSPICIOUS: Real IC in top 10% of permutations. Investigate further."
    else:
        verdict = "RED FLAG: Shuffled targets produce similar IC to real. STRUCTURAL LEAKAGE LIKELY."

    # Check if permuted ICs are themselves suspiciously positive
    if perm_mean > 0.005:
        verdict += f"\n  *** ADDITIONAL RED FLAG: Mean permuted IC = {perm_mean:.4f} > 0. Features may be contemporaneously correlated with target. ***"

    logger.info("\n  --- SHUFFLED-TARGET RESULTS ---")
    logger.info(f"  Real IC:          {baseline_ic:.5f}")
    logger.info(f"  Perm distribution: mean={perm_mean:.5f} std={perm_std:.5f} max={perm_max:.5f}")
    logger.info(f"  Perm p95/p99:     {perm_p95:.5f} / {perm_p99:.5f}")
    logger.info(f"  p-value (1-tail): {pvalue_empirical:.4f}")
    logger.info(f"  p-value (2-tail): {pvalue_twotail:.4f}")
    logger.info(f"  SNR (IC / perm_std): {snr:.2f}")
    logger.info(f"  Percentile rank:  {pct_rank:.1f}%")
    logger.info(f"  VERDICT: {verdict}")

    return {
        'test': 'shuffled_target',
        'n_perms': n_perms,
        'baseline_ic': baseline_ic,
        'baseline_tstat': baseline_res['tstat'],
        'baseline_fold_ics': baseline_res['fold_ics'],
        'perm_ics': perm_ics,
        'perm_mean': perm_mean,
        'perm_std': perm_std,
        'perm_p95': perm_p95,
        'perm_p99': perm_p99,
        'perm_max': perm_max,
        'pvalue_empirical': pvalue_empirical,
        'pvalue_twotail': pvalue_twotail,
        'pct_rank': pct_rank,
        'snr': snr,
        'verdict': verdict,
    }


# ============================================================================
# TEST 2: LAGGED-FEATURE TEST
# ============================================================================

def run_lagged_feature_test(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    hour_of_day: np.ndarray,
    lags: List[int] = None,
) -> dict:
    """
    Shift features forward by K bars so that bar i sees features from bar i-K.

    If bar i uses features from K bars ago, the model must learn truly predictive
    patterns (features from the past predicting the future). If IC holds up at
    large K, the features contain forward-looking information or are so
    autocorrelated that K bars is still "contemporaneous" for practical purposes.

    At 100ms intervals:
    - K=10  = 1 second ago
    - K=30  = 3 seconds ago (= the full prediction horizon)
    - K=50  = 5 seconds ago
    - K=100 = 10 seconds ago

    Expected: IC should DEGRADE as K increases (older features = less predictive).
    Red flag: IC does NOT degrade at K=30 (the return horizon). This means the
    model may be using features that are proxies for the same 3-second window.

    Note: We NaN-fill the first K rows of each day (lagged features cross day
    boundary — set to NaN to avoid overnight contamination).
    """
    if lags is None:
        lags = [10, 30, 50, 100]

    logger.info("\n" + "=" * 75)
    logger.info("TEST 2: LAGGED-FEATURE TEST")
    logger.info(f"  Testing lags K={lags} (at 100ms = {[k/10 for k in lags]}s ago)")
    logger.info("=" * 75)

    results_by_lag = {}
    n_days = len(day_boundaries) - 1
    N = len(target)

    # First: baseline (K=0, unlagged)
    logger.info("  Running baseline (K=0, no lag)...")
    t0 = time.time()
    baseline_res = walk_forward_evaluate(
        features=features,
        target=target,
        day_boundaries=day_boundaries,
        feature_names=feature_names,
        model_type='lgbm',
        hour_of_day=hour_of_day,
        lgbm_params=STANDARD_LGBM_PARAMS,
    )
    baseline_elapsed = time.time() - t0

    if 'error' in baseline_res:
        logger.error(f"  Baseline failed: {baseline_res['error']}")
        return {'error': baseline_res['error'], 'test': 'lagged_feature'}

    baseline_ic = baseline_res['ic']
    logger.info(
        f"  K=  0: IC={baseline_ic:.5f} t={baseline_res['tstat']:.2f} "
        f"({baseline_elapsed:.0f}s)  [BASELINE]"
    )
    results_by_lag[0] = {
        'lag_bars': 0,
        'lag_sec': 0.0,
        'ic': baseline_ic,
        'tstat': baseline_res['tstat'],
        'fold_ics': baseline_res['fold_ics'],
        'ic_retention_pct': 100.0,
    }

    # Test each lag
    for K in lags:
        t0 = time.time()
        K_sec = K / 10.0  # at 100ms intervals

        # Create lagged features array: bar i sees features from bar i-K
        # Implementation: roll the entire feature array forward by K
        # features_lagged[i] = features[i-K] for i >= K
        # features_lagged[i] = NaN for i < K (or crosses day boundary)
        features_lagged = np.empty_like(features)
        features_lagged[:] = np.nan

        # For each day, apply lag within-day only
        for d in range(n_days):
            d_start = day_boundaries[d]
            d_end = day_boundaries[d + 1]
            day_len = d_end - d_start

            if day_len <= K:
                # Day is shorter than lag — all NaN
                continue

            # Source rows: features[d_start : d_end - K]
            # Destination rows: features_lagged[d_start + K : d_end]
            # So bar (d_start + K + j) sees features[d_start + j]
            n_valid = day_len - K
            features_lagged[d_start + K:d_end] = features[d_start:d_start + n_valid]

        lag_res = walk_forward_evaluate(
            features=features_lagged,
            target=target,
            day_boundaries=day_boundaries,
            feature_names=feature_names,
            model_type='lgbm',
            hour_of_day=hour_of_day,
            lgbm_params=STANDARD_LGBM_PARAMS,
        )
        elapsed = time.time() - t0

        if 'error' not in lag_res:
            ic_lag = lag_res['ic']
            retention = (ic_lag / baseline_ic * 100) if abs(baseline_ic) > 1e-9 else 0.0
            logger.info(
                f"  K={K:3d} ({K_sec:.1f}s): IC={ic_lag:.5f} t={lag_res['tstat']:.2f} "
                f"retention={retention:+.1f}%  ({elapsed:.0f}s)"
            )
            results_by_lag[K] = {
                'lag_bars': int(K),
                'lag_sec': float(K_sec),
                'ic': float(ic_lag),
                'tstat': float(lag_res['tstat']),
                'fold_ics': lag_res['fold_ics'],
                'ic_retention_pct': float(retention),
            }
        else:
            logger.warning(f"  K={K:3d}: FAILED — {lag_res['error']}")
            results_by_lag[K] = {'error': lag_res['error'], 'lag_bars': K}

        # Free lagged array
        del features_lagged
        gc.collect()

    # Compute decay profile
    lag_ics = [(K, results_by_lag[K].get('ic', np.nan)) for K in sorted(results_by_lag.keys())]
    lag_ics_arr = np.array([(k, ic) for k, ic in lag_ics if np.isfinite(ic)])

    # Verdict based on retention at K=30 (the horizon itself)
    k30_result = results_by_lag.get(30, {})
    k30_ic = k30_result.get('ic', None)
    k30_retention = k30_result.get('ic_retention_pct', None)

    if k30_ic is None:
        verdict = "INCONCLUSIVE: K=30 test failed."
    elif abs(k30_ic) < 0.01:
        verdict = "CLEAN: Features from 3s ago have near-zero IC. Signal is genuinely contemporaneous (not past-proxying)."
    elif k30_retention is not None and k30_retention > 80:
        verdict = f"RED FLAG: K=30 retention = {k30_retention:.1f}%. Features from 3s AGO predict almost as well as current features. Possible temporal leakage or features that are proxies for the full 3s window."
    elif k30_retention is not None and k30_retention > 50:
        verdict = f"SUSPICIOUS: K=30 retention = {k30_retention:.1f}%. Signal is persistent. May be real autocorrelation or leakage — investigate feature autocorrelation."
    else:
        verdict = f"LIKELY CLEAN: IC decays substantially. K=30 retains {k30_retention:.1f}% of baseline IC."

    logger.info(f"\n  VERDICT: {verdict}")

    return {
        'test': 'lagged_feature',
        'lags_tested': sorted(results_by_lag.keys()),
        'results_by_lag': results_by_lag,
        'baseline_ic': baseline_ic,
        'k30_retention_pct': k30_retention,
        'verdict': verdict,
    }


# ============================================================================
# TEST 3: CROSS-DAY GAP TEST
# ============================================================================

def run_cross_day_gap_test(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    hour_of_day: np.ndarray,
) -> dict:
    """
    Train on day D, predict day D+2 (skip a full calendar day).

    Standard walk-forward already has a 1-day purge gap (train ends on day D-1,
    predict day D). This test enforces a 2-day gap: train ends on day D-1,
    skip day D entirely, predict day D+1.

    If IC survives a full 1-extra-day gap, the signal is more robust and less
    likely to be from transient feature-target co-location.

    Expected: IC decreases somewhat (less data, slight regime shift).
    Red flag: IC disappears entirely (signal was purely from recent proximity
    of train/test data, not from generalizable patterns).
    """
    logger.info("\n" + "=" * 75)
    logger.info("TEST 3: CROSS-DAY GAP TEST")
    logger.info("  Compare standard 1-day gap vs 2-day gap (skip one full day)")
    logger.info("=" * 75)

    import lightgbm as lgb
    from scipy.stats import ttest_1samp as _ttest

    n_days = len(day_boundaries) - 1
    min_train_days = 3

    # ---- Standard 1-day gap (baseline) ----
    logger.info("\n  [A] Standard 1-day gap walk-forward...")
    t0 = time.time()
    std_res = walk_forward_evaluate(
        features=features,
        target=target,
        day_boundaries=day_boundaries,
        feature_names=feature_names,
        model_type='lgbm',
        hour_of_day=hour_of_day,
        lgbm_params=STANDARD_LGBM_PARAMS,
    )
    std_elapsed = time.time() - t0

    if 'error' in std_res:
        logger.error(f"  Standard evaluation failed: {std_res['error']}")
        return {'error': std_res['error'], 'test': 'cross_day_gap'}

    logger.info(
        f"  Standard (1-day gap): IC={std_res['ic']:.5f} t={std_res['tstat']:.2f} "
        f"ICIR={std_res['icir']:.2f} ({std_elapsed:.0f}s)"
    )

    # ---- 2-day gap: train on [0, D-1], skip D, predict D+1 ----
    logger.info("\n  [B] 2-day gap walk-forward (skip 1 extra day)...")
    t0 = time.time()

    all_preds_2day = []
    all_actuals_2day = []
    fold_ics_2day = []

    for test_day in range(min_train_days + 1, n_days):
        # Train on days [0, test_day - 2] (skip test_day - 1)
        train_start = day_boundaries[0]
        train_end = day_boundaries[test_day - 1]  # stop 2 days before test

        # Predict test_day (which is 2 days after train end)
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_train = features[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features[test_start:test_end]
        y_test = target[test_start:test_end]

        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 50:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        try:
            split = int(len(X_tr) * 0.8)
            model = lgb.LGBMRegressor(**STANDARD_LGBM_PARAMS)
            model.fit(
                X_tr[:split], y_tr[:split],
                eval_set=[(X_tr[split:], y_tr[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict(X_te)

            all_preds_2day.append(preds)
            all_actuals_2day.append(y_te)

            if len(preds) > 10:
                ic_fold = float(spearmanr(preds, y_te)[0])
                if np.isfinite(ic_fold):
                    fold_ics_2day.append(ic_fold)

            del model
            gc.collect()

        except Exception as e:
            logger.warning(f"  Day {test_day} (2-day gap) failed: {e}")
            continue

    two_day_elapsed = time.time() - t0

    if not all_preds_2day:
        logger.error("  2-day gap: No valid predictions")
        return {
            'test': 'cross_day_gap',
            'error': '2-day gap produced no predictions',
            'standard_ic': std_res['ic'],
        }

    preds_2d = np.concatenate(all_preds_2day)
    acts_2d = np.concatenate(all_actuals_2day)
    valid_2d = np.isfinite(preds_2d) & np.isfinite(acts_2d)
    p2d, a2d = preds_2d[valid_2d], acts_2d[valid_2d]

    ic_2day_pooled = float(spearmanr(p2d, a2d)[0])

    if len(fold_ics_2day) > 2:
        ic_2day = float(np.mean(fold_ics_2day))
        ic_2day_std = float(np.std(fold_ics_2day))
        tstat_2day = ic_2day / ic_2day_std * np.sqrt(len(fold_ics_2day)) if ic_2day_std > 0 else 0.0
    else:
        ic_2day = ic_2day_pooled
        ic_2day_std = 0.0
        tstat_2day = 0.0

    retention_2day = (ic_2day / std_res['ic'] * 100) if abs(std_res['ic']) > 1e-9 else 0.0

    logger.info(
        f"  2-day gap: IC={ic_2day:.5f} (pooled={ic_2day_pooled:.5f}) "
        f"t={tstat_2day:.2f} n_folds={len(fold_ics_2day)} "
        f"retention={retention_2day:+.1f}%  ({two_day_elapsed:.0f}s)"
    )

    # Verdict
    if ic_2day > 0.03 and retention_2day > 50:
        verdict = f"STRONG SIGNAL: IC={ic_2day:.4f} survives 2-day gap with {retention_2day:.1f}% retention. Signal is generalizable and not from data proximity."
    elif ic_2day > 0.01 and retention_2day > 25:
        verdict = f"MODERATE SIGNAL: IC={ic_2day:.4f} at {retention_2day:.1f}% retention. Some signal survives the additional gap."
    elif ic_2day > 0:
        verdict = f"WEAK SIGNAL: IC={ic_2day:.4f} barely positive with {retention_2day:.1f}% retention. Signal may not generalize well."
    else:
        verdict = f"RED FLAG: IC disappears at 2-day gap (IC={ic_2day:.4f}). Signal may depend heavily on data proximity."

    logger.info(f"  VERDICT: {verdict}")

    return {
        'test': 'cross_day_gap',
        'standard_gap': {
            'ic': std_res['ic'],
            'tstat': std_res['tstat'],
            'icir': std_res['icir'],
            'fold_ics': std_res['fold_ics'],
            'n_folds': std_res['n_folds'],
        },
        'two_day_gap': {
            'ic': ic_2day,
            'ic_pooled': ic_2day_pooled,
            'ic_std': ic_2day_std,
            'tstat': tstat_2day,
            'fold_ics': fold_ics_2day,
            'n_folds': len(fold_ics_2day),
            'n_preds': int(len(p2d)),
        },
        'ic_retention_2day_pct': retention_2day,
        'verdict': verdict,
    }


# ============================================================================
# TEST 4: FEATURE AUTOCORRELATION ANALYSIS
# ============================================================================

def run_feature_autocorrelation_test(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    top_features: List[str],
    lags: List[int] = None,
) -> dict:
    """
    For each top feature, compute autocorrelation at specified lags.

    High feature autocorrelation explains WHY the model can still predict at
    lagged K (Test 2): if feat[i] ≈ feat[i-30], then using features from
    30 bars ago is almost the same as using current features.

    This tells us whether the IC survival at lagged K is benign (real signal
    from autocorrelated features) or malicious (features that encode the
    current return because they're computed over the same window).

    Also compute:
    - Cross-correlation between each feature and the TARGET at various lags
    - This reveals if the feature is "explaining" current price because it
      was computed using the same price data

    Key diagnostic:
    - cross_corr(feature[i], target[i]) vs cross_corr(feature[i-K], target[i])
    - If cross_corr is similar at K=0 and K=30, the feature autocorrelation
      explains the IC survival at lag 30 — this is benign.
    - If cross_corr drops sharply at K=1 but feature autocorr is high,
      the feature IS predictive only contemporaneously — genuine alpha.
    """
    if lags is None:
        lags = [0, 1, 5, 10, 30, 50, 100]

    logger.info("\n" + "=" * 75)
    logger.info("TEST 4: FEATURE AUTOCORRELATION ANALYSIS")
    logger.info(f"  Analyzing {len(top_features)} top features at lags {lags}")
    logger.info("=" * 75)

    n_days = len(day_boundaries) - 1
    N = len(target)

    # Build index from feature name to column index
    feat_idx_map = {name: idx for idx, name in enumerate(feature_names)}

    results_by_feature = {}

    for feat_name in top_features:
        if feat_name not in feat_idx_map:
            logger.warning(f"  Feature '{feat_name}' not in feature set — skipping")
            continue

        col_idx = feat_idx_map[feat_name]
        feat_vals = features[:, col_idx].astype(np.float64)

        # Autocorrelation at each lag (within-day)
        # We compute across the full dataset but NaN-fill day boundaries
        autocorrs = {}
        cross_corrs_feat_target = {}

        for lag in lags:
            # Build (feat, lagged_feat) pairs within each day
            feat_current = []
            feat_lagged = []
            target_current = []

            for d in range(n_days):
                d_start = day_boundaries[d]
                d_end = day_boundaries[d + 1]
                day_len = d_end - d_start

                if day_len <= lag + 1:
                    continue

                f_curr = feat_vals[d_start + lag:d_end]
                f_lag = feat_vals[d_start:d_end - lag]
                t_curr = target[d_start + lag:d_end].astype(np.float64)

                feat_current.append(f_curr)
                feat_lagged.append(f_lag)
                target_current.append(t_curr)

            if not feat_current:
                autocorrs[lag] = np.nan
                cross_corrs_feat_target[lag] = np.nan
                continue

            fc = np.concatenate(feat_current)
            fl = np.concatenate(feat_lagged)
            tc = np.concatenate(target_current)

            # Remove NaNs from both
            valid_ac = np.isfinite(fc) & np.isfinite(fl)
            valid_cc = np.isfinite(fc) & np.isfinite(tc)

            if valid_ac.sum() > 100:
                try:
                    ac_val = float(spearmanr(fc[valid_ac], fl[valid_ac])[0])
                    autocorrs[lag] = ac_val if np.isfinite(ac_val) else np.nan
                except Exception:
                    autocorrs[lag] = np.nan
            else:
                autocorrs[lag] = np.nan

            if valid_cc.sum() > 100:
                try:
                    cc_val = float(spearmanr(fc[valid_cc], tc[valid_cc])[0])
                    cross_corrs_feat_target[lag] = cc_val if np.isfinite(cc_val) else np.nan
                except Exception:
                    cross_corrs_feat_target[lag] = np.nan
            else:
                cross_corrs_feat_target[lag] = np.nan

        # Log results for this feature
        ac_str = "  ".join(
            f"lag{k}={autocorrs.get(k, np.nan):+.3f}" for k in lags
        )
        cc_str = "  ".join(
            f"lag{k}={cross_corrs_feat_target.get(k, np.nan):+.4f}" for k in lags
        )
        logger.info(f"\n  Feature: {feat_name}")
        logger.info(f"    AutoCorr:  {ac_str}")
        logger.info(f"    CrossCorr: {cc_str}")

        # Assess: is IC retention at lag30 explained by feature autocorr?
        ac_at_30 = autocorrs.get(30, np.nan)
        cc_at_0 = cross_corrs_feat_target.get(0, np.nan)
        cc_at_30 = cross_corrs_feat_target.get(30, np.nan)

        if np.isfinite(ac_at_30) and np.isfinite(cc_at_0) and np.isfinite(cc_at_30):
            cc_decay = (cc_at_30 / cc_at_0) if abs(cc_at_0) > 1e-6 else np.nan
            if np.isfinite(cc_decay):
                if abs(ac_at_30) > 0.9:
                    interpretation = f"VERY HIGH AUTOCORR at lag30 ({ac_at_30:.3f}). Feature is nearly constant at 3s scale — IC at lag30 will be similar to lag0."
                elif abs(ac_at_30) > 0.7:
                    interpretation = f"HIGH AUTOCORR at lag30 ({ac_at_30:.3f}). Feature changes slowly — explains IC retention at lag30."
                elif abs(ac_at_30) > 0.3:
                    interpretation = f"MODERATE AUTOCORR at lag30 ({ac_at_30:.3f}). Some persistence."
                else:
                    interpretation = f"LOW AUTOCORR at lag30 ({ac_at_30:.3f}). Feature changes rapidly — IC at lag30 reflects genuine predictive decay."
            else:
                interpretation = "Cannot determine (cc_at_0 near zero)"
        else:
            interpretation = "Insufficient data"

        logger.info(f"    Interpretation: {interpretation}")

        results_by_feature[feat_name] = {
            'autocorr_by_lag': {k: float(v) if np.isfinite(v) else None for k, v in autocorrs.items()},
            'crosscorr_feat_target_by_lag': {k: float(v) if np.isfinite(v) else None for k, v in cross_corrs_feat_target.items()},
            'ac_at_lag30': float(ac_at_30) if np.isfinite(ac_at_30) else None,
            'cc_at_lag0': float(cc_at_0) if np.isfinite(cc_at_0) else None,
            'cc_at_lag30': float(cc_at_30) if np.isfinite(cc_at_30) else None,
            'interpretation': interpretation,
        }

    # Aggregate verdict
    high_ac_features = [
        f for f, res in results_by_feature.items()
        if res.get('ac_at_lag30') is not None and abs(res['ac_at_lag30']) > 0.7
    ]
    low_cc_decay = [
        f for f, res in results_by_feature.items()
        if (res.get('cc_at_lag0') is not None and res.get('cc_at_lag30') is not None
            and abs(res['cc_at_lag0']) > 1e-4
            and abs(res.get('cc_at_lag30', 0) / res['cc_at_lag0']) > 0.5)
    ]

    if high_ac_features:
        overall_verdict = (
            f"HIGH AUTOCORRELATION features: {high_ac_features}. "
            f"IC survival at lag30 for these features is EXPECTED from autocorrelation, not leakage. "
            f"Check if these features are computed over windows >= 3s."
        )
    else:
        overall_verdict = (
            f"Features have moderate/low autocorrelation at lag30. "
            f"IC survival at lag30 (Test 2) is surprising if features decay quickly — investigate."
        )

    logger.info(f"\n  OVERALL VERDICT: {overall_verdict}")

    return {
        'test': 'feature_autocorrelation',
        'lags_tested': lags,
        'top_features_analyzed': top_features,
        'results_by_feature': results_by_feature,
        'high_autocorr_features': high_ac_features,
        'verdict': overall_verdict,
    }


# ============================================================================
# TEST 5: RETRODICTION TEST (Ultimate Leakage Check)
# ============================================================================

def run_retrodiction_test(
    features: np.ndarray,
    target_forward: np.ndarray,    # ret_3s: price change in next 3s
    mid_prices: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    hour_of_day: np.ndarray,
    horizon_bars: int = 30,        # 3s at 100ms = 30 bars
) -> dict:
    """
    Retrodiction test: can the model predict PAST returns?

    Construct a backward-looking target:
        target_backward[i] = log(mid[i]) - log(mid[i - horizon_bars])
        (= the log return from 3 seconds AGO to now)

    Then use features at bar i to predict this PAST return.

    If the model achieves high IC predicting backwards, this is a strong
    indicator of contemporaneous contamination:
    - Features at bar i contain information about what happened at bar i-30
    - This could be because features are computed over rolling windows that
      include the past 3 seconds
    - A model that predicts both past AND future equally well is NOT predictive;
      it is contemporaneously correlated

    Expected result if no leakage:
    - IC (forward) >> IC (backward)
    - IC backward should be near zero (the past is already priced in)

    Red flag: IC backward ≈ IC forward. The model is learning from
    contemporaneous contamination, not genuine prediction.

    Note: We intentionally do NOT NaN-fill day boundaries for the backward
    target at day openings (bar i - horizon_bars crosses into previous day).
    We DO NaN-fill those to be conservative — we don't want the backward
    target to use overnight returns.
    """
    logger.info("\n" + "=" * 75)
    logger.info("TEST 5: RETRODICTION TEST (Ultimate Leakage Check)")
    logger.info(f"  Horizon: {horizon_bars} bars = {horizon_bars * 0.1:.1f}s")
    logger.info("  Prediction: features[i] → log_return from i-30 to i (PAST)")
    logger.info("=" * 75)

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # ---- Construct backward target ----
    # target_backward[i] = log(mid[i]) - log(mid[i - horizon_bars])
    log_mid = np.log(np.maximum(mid_prices.astype(np.float64), 1.0))
    target_backward = np.full(N, np.nan, dtype=np.float32)

    for d in range(n_days):
        d_start = day_boundaries[d]
        d_end = day_boundaries[d + 1]
        day_len = d_end - d_start

        if day_len <= horizon_bars:
            continue

        # Valid bars: [d_start + horizon_bars, d_end)
        # At bar i: backward_return = log_mid[i] - log_mid[i - horizon_bars]
        src_future = log_mid[d_start + horizon_bars:d_end]
        src_past = log_mid[d_start:d_end - horizon_bars]

        target_backward[d_start + horizon_bars:d_end] = (src_future - src_past).astype(np.float32)

    valid_bwd = np.isfinite(target_backward)
    valid_fwd = np.isfinite(target_forward)
    logger.info(
        f"  Backward target: {valid_bwd.sum():,} valid bars "
        f"(forward: {valid_fwd.sum():,})"
    )

    # ---- Evaluate forward (standard) ----
    logger.info("\n  [A] Forward prediction (standard IC check)...")
    t0 = time.time()
    fwd_res = walk_forward_evaluate(
        features=features,
        target=target_forward,
        day_boundaries=day_boundaries,
        feature_names=feature_names,
        model_type='lgbm',
        hour_of_day=hour_of_day,
        lgbm_params=STANDARD_LGBM_PARAMS,
    )
    fwd_elapsed = time.time() - t0

    if 'error' in fwd_res:
        logger.error(f"  Forward evaluation failed: {fwd_res['error']}")
        return {'error': fwd_res['error'], 'test': 'retrodiction'}

    logger.info(
        f"  Forward IC={fwd_res['ic']:.5f} t={fwd_res['tstat']:.2f} "
        f"ICIR={fwd_res['icir']:.2f} ({fwd_elapsed:.0f}s)"
    )

    # ---- Evaluate backward (retrodiction) ----
    logger.info("\n  [B] Backward prediction (retrodiction — LEAKAGE CHECK)...")
    t0 = time.time()
    bwd_res = walk_forward_evaluate(
        features=features,
        target=target_backward,
        day_boundaries=day_boundaries,
        feature_names=feature_names,
        model_type='lgbm',
        hour_of_day=hour_of_day,
        lgbm_params=STANDARD_LGBM_PARAMS,
    )
    bwd_elapsed = time.time() - t0

    if 'error' in bwd_res:
        logger.error(f"  Backward evaluation failed: {bwd_res['error']}")
        return {
            'test': 'retrodiction',
            'forward': {
                'ic': fwd_res['ic'],
                'tstat': fwd_res['tstat'],
                'fold_ics': fwd_res['fold_ics'],
            },
            'backward': {'error': bwd_res['error']},
        }

    logger.info(
        f"  Backward IC={bwd_res['ic']:.5f} t={bwd_res['tstat']:.2f} "
        f"ICIR={bwd_res['icir']:.2f} ({bwd_elapsed:.0f}s)"
    )

    # ---- Compare ----
    fwd_ic = fwd_res['ic']
    bwd_ic = bwd_res['ic']
    ratio = bwd_ic / fwd_ic if abs(fwd_ic) > 1e-9 else np.nan

    logger.info(f"\n  Forward IC:  {fwd_ic:+.5f}")
    logger.info(f"  Backward IC: {bwd_ic:+.5f}")
    logger.info(f"  Ratio (bwd/fwd): {ratio:+.3f}" if np.isfinite(ratio) else "  Ratio: N/A")

    # Per-fold comparison
    fwd_fold_arr = np.array(fwd_res['fold_ics'])
    bwd_fold_arr = np.array(bwd_res['fold_ics'])
    min_folds = min(len(fwd_fold_arr), len(bwd_fold_arr))

    fold_comparison = None
    if min_folds >= 2:
        fold_ratio_arr = bwd_fold_arr[:min_folds] / (fwd_fold_arr[:min_folds] + 1e-9)
        fold_comparison = {
            'mean_ratio': float(np.mean(fold_ratio_arr)),
            'std_ratio': float(np.std(fold_ratio_arr)),
            'fwd_fold_ics': fwd_res['fold_ics'],
            'bwd_fold_ics': bwd_res['fold_ics'][:min_folds],
        }

        fold_ratio_str = " ".join(
            f"d{i}:{fold_ratio_arr[i]:+.2f}" for i in range(min_folds)
        )
        logger.info(f"  Fold ratios (bwd/fwd): [{fold_ratio_str}]")

    # Verdict
    if np.isfinite(ratio):
        if abs(ratio) > 0.8:
            verdict = (
                f"CRITICAL RED FLAG: Backward IC ({bwd_ic:.4f}) is {ratio:.1%} of forward IC ({fwd_ic:.4f}). "
                f"The model predicts the past almost as well as the future. "
                f"This is STRONG EVIDENCE of contemporaneous feature contamination. "
                f"Top features almost certainly encode current bar information."
            )
        elif abs(ratio) > 0.5:
            verdict = (
                f"RED FLAG: Backward IC ({bwd_ic:.4f}) is {ratio:.1%} of forward IC ({fwd_ic:.4f}). "
                f"Significant backward predictability. Features may be computed over rolling windows "
                f"that partially overlap with the target horizon."
            )
        elif abs(ratio) > 0.25:
            verdict = (
                f"MODERATE CONCERN: Backward IC ({bwd_ic:.4f}) is {ratio:.1%} of forward IC ({fwd_ic:.4f}). "
                f"Some backward predictability. This is common with high-autocorrelation order book features. "
                f"Investigate which features drive backward IC."
            )
        elif bwd_ic > 0.01:
            verdict = (
                f"MINOR CONCERN: Backward IC={bwd_ic:.4f} is positive but small vs forward IC={fwd_ic:.4f}. "
                f"Likely from feature autocorrelation. Not a major red flag."
            )
        else:
            verdict = (
                f"CLEAN: Backward IC ({bwd_ic:.4f}) is near zero vs forward IC ({fwd_ic:.4f}). "
                f"The model cannot predict the past — signal is genuinely forward-looking."
            )
    else:
        verdict = "INCONCLUSIVE: Cannot compute ratio (forward IC near zero)."

    logger.info(f"\n  VERDICT: {verdict}")

    return {
        'test': 'retrodiction',
        'horizon_bars': horizon_bars,
        'horizon_sec': horizon_bars * 0.1,
        'forward': {
            'ic': float(fwd_ic),
            'tstat': float(fwd_res['tstat']),
            'icir': float(fwd_res['icir']),
            'fold_ics': fwd_res['fold_ics'],
            'n_folds': fwd_res['n_folds'],
        },
        'backward': {
            'ic': float(bwd_ic),
            'tstat': float(bwd_res['tstat']),
            'icir': float(bwd_res['icir']),
            'fold_ics': bwd_res['fold_ics'],
            'n_folds': bwd_res['n_folds'],
        },
        'bwd_fwd_ratio': float(ratio) if np.isfinite(ratio) else None,
        'fold_comparison': fold_comparison,
        'verdict': verdict,
    }


# ============================================================================
# MASTER SUMMARY
# ============================================================================

def generate_leakage_summary(all_results: dict) -> str:
    """Generate a comprehensive leakage audit summary."""
    lines = [
        "",
        "=" * 80,
        "PARANOID LEAKAGE AUDIT — MASTER SUMMARY",
        "=" * 80,
        "",
    ]

    # ---- Test 1: Shuffled target ----
    shuffle_res = all_results.get('shuffled_target', {})
    if 'error' not in shuffle_res and shuffle_res:
        lines.append("TEST 1: SHUFFLED-TARGET BASELINE (Permutation Test)")
        lines.append(f"  Real IC:       {shuffle_res.get('baseline_ic', 'N/A'):.5f}")
        lines.append(f"  Perm IC mean:  {shuffle_res.get('perm_mean', 'N/A'):.5f}")
        lines.append(f"  Perm IC std:   {shuffle_res.get('perm_std', 'N/A'):.5f}")
        lines.append(f"  p-value:       {shuffle_res.get('pvalue_empirical', 'N/A'):.4f}")
        lines.append(f"  SNR:           {shuffle_res.get('snr', 'N/A'):.2f}")
        lines.append(f"  Result: {shuffle_res.get('verdict', 'N/A')[:100]}")
        lines.append("")
    else:
        lines.append("TEST 1: SHUFFLED-TARGET — SKIPPED or FAILED")
        lines.append("")

    # ---- Test 2: Lagged features ----
    lag_res = all_results.get('lagged_feature', {})
    if 'error' not in lag_res and lag_res:
        lines.append("TEST 2: LAGGED-FEATURE TEST")
        lines.append(f"  Baseline IC:   {lag_res.get('baseline_ic', 'N/A'):.5f}")
        by_lag = lag_res.get('results_by_lag', {})
        for k in sorted(by_lag.keys()):
            r = by_lag[k]
            if 'error' not in r:
                lines.append(
                    f"  K={k:3d} ({k*0.1:.1f}s): IC={r.get('ic', float('nan')):.5f}  "
                    f"retention={r.get('ic_retention_pct', 0):+.1f}%"
                )
        lines.append(f"  Result: {lag_res.get('verdict', 'N/A')[:100]}")
        lines.append("")
    else:
        lines.append("TEST 2: LAGGED-FEATURE — SKIPPED or FAILED")
        lines.append("")

    # ---- Test 3: Cross-day gap ----
    gap_res = all_results.get('cross_day_gap', {})
    if 'error' not in gap_res and gap_res:
        std = gap_res.get('standard_gap', {})
        two = gap_res.get('two_day_gap', {})
        lines.append("TEST 3: CROSS-DAY GAP TEST")
        lines.append(f"  Standard (1-day gap): IC={std.get('ic', float('nan')):.5f}")
        lines.append(
            f"  Two-day gap:          IC={two.get('ic', float('nan')):.5f}  "
            f"retention={gap_res.get('ic_retention_2day_pct', 0):+.1f}%"
        )
        lines.append(f"  Result: {gap_res.get('verdict', 'N/A')[:100]}")
        lines.append("")
    else:
        lines.append("TEST 3: CROSS-DAY GAP — SKIPPED or FAILED")
        lines.append("")

    # ---- Test 4: Autocorrelation ----
    ac_res = all_results.get('feature_autocorrelation', {})
    if 'error' not in ac_res and ac_res:
        lines.append("TEST 4: FEATURE AUTOCORRELATION ANALYSIS")
        lines.append(f"  High-autocorr features (lag30 > 0.7): {ac_res.get('high_autocorr_features', [])}")
        rbf = ac_res.get('results_by_feature', {})
        for feat, fres in list(rbf.items())[:5]:  # top 5 only in summary
            ac30 = fres.get('ac_at_lag30')
            cc0 = fres.get('cc_at_lag0')
            ac30_str = f"{ac30:.3f}" if ac30 is not None else "N/A"
            cc0_str = f"{cc0:.4f}" if cc0 is not None else "N/A"
            lines.append(f"  {feat:<35s}: autocorr@30={ac30_str}  crosscorr@0={cc0_str}")
        lines.append(f"  Result: {ac_res.get('verdict', 'N/A')[:100]}")
        lines.append("")
    else:
        lines.append("TEST 4: FEATURE AUTOCORRELATION — SKIPPED or FAILED")
        lines.append("")

    # ---- Test 5: Retrodiction ----
    retro_res = all_results.get('retrodiction', {})
    if 'error' not in retro_res and retro_res:
        fwd = retro_res.get('forward', {})
        bwd = retro_res.get('backward', {})
        ratio = retro_res.get('bwd_fwd_ratio')
        lines.append("TEST 5: RETRODICTION TEST (Ultimate Leakage Check)")
        lines.append(f"  Forward IC:  {fwd.get('ic', float('nan')):+.5f}  t={fwd.get('tstat', 0):.2f}")
        lines.append(f"  Backward IC: {bwd.get('ic', float('nan')):+.5f}  t={bwd.get('tstat', 0):.2f}")
        lines.append(f"  Bwd/Fwd ratio: {ratio:.3f}" if ratio is not None else "  Ratio: N/A")
        lines.append(f"  Result: {retro_res.get('verdict', 'N/A')[:100]}")
        lines.append("")
    else:
        lines.append("TEST 5: RETRODICTION — SKIPPED or FAILED")
        lines.append("")

    # ---- Overall verdict ----
    lines.append("=" * 80)
    lines.append("OVERALL LEAKAGE ASSESSMENT")
    lines.append("=" * 80)

    red_flags = []
    warnings = []

    # Check shuffled test
    if 'pvalue_empirical' in shuffle_res:
        pval = shuffle_res['pvalue_empirical']
        perm_mean = shuffle_res.get('perm_mean', 0)
        if pval > 0.05:
            red_flags.append(f"Shuffled target p-value={pval:.3f} (signal not above permutation noise)")
        if perm_mean > 0.005:
            red_flags.append(f"Permuted IC mean={perm_mean:.4f} > 0 (structural leakage indicator)")

    # Check lag test
    if 'results_by_lag' in lag_res:
        k30 = lag_res['results_by_lag'].get(30, {})
        retention = k30.get('ic_retention_pct', 0)
        if retention is not None and retention > 80:
            red_flags.append(f"K=30 lag retention={retention:.1f}% (features are 3s-window proxies)")
        elif retention is not None and retention > 50:
            warnings.append(f"K=30 lag retention={retention:.1f}% (moderate, investigate autocorr)")

    # Check cross-day gap
    if 'ic_retention_2day_pct' in gap_res:
        ret_2day = gap_res['ic_retention_2day_pct']
        if ret_2day < 10:
            red_flags.append(f"2-day gap IC retention={ret_2day:.1f}% (signal vanishes with small gap increase)")
        elif ret_2day < 25:
            warnings.append(f"2-day gap IC retention={ret_2day:.1f}% (signal weakens significantly)")

    # Check retrodiction
    if 'bwd_fwd_ratio' in retro_res and retro_res['bwd_fwd_ratio'] is not None:
        ratio = retro_res['bwd_fwd_ratio']
        if abs(ratio) > 0.8:
            red_flags.append(f"Retrodiction ratio={ratio:.2f} (model predicts past as well as future!)")
        elif abs(ratio) > 0.5:
            red_flags.append(f"Retrodiction ratio={ratio:.2f} (significant backward predictability)")
        elif abs(ratio) > 0.25:
            warnings.append(f"Retrodiction ratio={ratio:.2f} (moderate backward predictability)")

    if not red_flags and not warnings:
        lines.append("\nRESULT: NO LEAKAGE DETECTED")
        lines.append("  All 5 tests pass. The IC=0.079-0.114 signal appears genuine.")
        lines.append("  Proceed with confidence to live trading evaluation.")
    elif not red_flags and warnings:
        lines.append("\nRESULT: MINOR CONCERNS — SIGNAL LIKELY GENUINE")
        lines.append("  Warnings (not blocking):")
        for w in warnings:
            lines.append(f"    - {w}")
        lines.append("  Recommendation: Note warnings but proceed with caution.")
    elif len(red_flags) <= 2:
        lines.append("\nRESULT: LEAKAGE CONCERNS — INVESTIGATE BEFORE TRADING")
        lines.append("  Red flags:")
        for rf in red_flags:
            lines.append(f"    - {rf}")
        if warnings:
            lines.append("  Warnings:")
            for w in warnings:
                lines.append(f"    - {w}")
        lines.append("  Recommendation: Identify and fix the leakage source before proceeding.")
    else:
        lines.append("\nRESULT: LIKELY LEAKED — DO NOT TRADE")
        lines.append("  Multiple red flags detected:")
        for rf in red_flags:
            lines.append(f"    *** {rf}")
        lines.append("  Recommendation: Fundamental issue with feature construction. Re-examine.")

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Paranoid leakage audit for ES futures LightGBM model'
    )
    parser.add_argument(
        '--tests', nargs='+',
        default=['shuffle', 'lag', 'cross_day', 'autocorr', 'retro'],
        choices=['shuffle', 'lag', 'cross_day', 'autocorr', 'retro'],
        help='Which tests to run'
    )
    parser.add_argument(
        '--n-perms', type=int, default=10,
        help='Number of permutations for shuffled-target test'
    )
    parser.add_argument(
        '--fast', action='store_true',
        help='Use faster LightGBM params (fewer trees, faster but less precise)'
    )
    parser.add_argument(
        '--lags', type=int, nargs='+', default=[10, 30, 50, 100],
        help='Lag values for lagged-feature test (in bars at 100ms)'
    )
    parser.add_argument(
        '--top-n-features', type=int, default=9,
        help='Number of top features to analyze in autocorrelation test'
    )
    args = parser.parse_args()

    t0_total = time.time()

    logger.info("=" * 80)
    logger.info("PARANOID LEAKAGE AUDIT — ES Futures 3s Return LightGBM Model")
    logger.info(f"  Tests: {args.tests}")
    logger.info(f"  Permutations: {args.n_perms}")
    logger.info(f"  Fast mode: {args.fast}")
    logger.info(f"  Lags (bars): {args.lags}")
    logger.info("=" * 80)

    # =========================================================
    # Load data
    # =========================================================
    logger.info("\nLoading data...")
    scanner = MBOAlphaScanner()
    stats = load_feature_cache(scanner)

    if stats is None:
        logger.error(
            "No feature cache found. Run run_return_multihorizon.py first to build cache."
        )
        sys.exit(1)

    logger.info(
        f"Data loaded: {scanner.features.shape[0]:,} snapshots, "
        f"{len(scanner.day_boundaries)-1} days, "
        f"{scanner.features.shape[1]} raw features"
    )

    # Compute 3s return target
    logger.info("Computing 3s return target...")
    targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        horizons_sec={'3s': 3},
        include_flow_target=False,
    )
    target = targets['ret_3s']

    valid_count = int(np.isfinite(target).sum())
    logger.info(f"Target ret_3s: {valid_count:,} valid bars")

    # Build clean feature set (exclude vol proxies, price, time)
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    logger.info(
        f"Features: {features_clean.shape[1]} clean "
        f"(excluded {(~keep_mask).sum()} vol/price/time features)"
    )

    # Select top features that exist in clean set
    top_features_in_clean = [fn for fn in TOP_9_FEATURES if fn in names_clean]
    extra_needed = args.top_n_features - len(top_features_in_clean)
    if extra_needed > 0:
        for fn in names_clean:
            if fn not in top_features_in_clean:
                top_features_in_clean.append(fn)
            if len(top_features_in_clean) >= args.top_n_features:
                break
    top_features_in_clean = top_features_in_clean[:args.top_n_features]

    logger.info(f"Top features for analysis: {top_features_in_clean}")

    # =========================================================
    # Run selected tests
    # =========================================================
    all_results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'tests_run': args.tests,
        'data_stats': {
            'n_snapshots': int(scanner.features.shape[0]),
            'n_days': int(len(scanner.day_boundaries) - 1),
            'n_features_raw': int(scanner.features.shape[1]),
            'n_features_clean': int(features_clean.shape[1]),
            'n_excluded': int((~keep_mask).sum()),
            'target_valid_count': valid_count,
        },
        'config': {
            'n_perms': args.n_perms,
            'fast_mode': args.fast,
            'lags': args.lags,
            'top_features': top_features_in_clean,
        },
    }

    # ---- TEST 1: Shuffled target ----
    if 'shuffle' in args.tests:
        logger.info("\n" + "#" * 80)
        logger.info("# STARTING TEST 1: SHUFFLED-TARGET BASELINE")
        logger.info("#" * 80)
        t0 = time.time()
        shuffle_res = run_shuffled_target_test(
            features=features_clean,
            target=target,
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            hour_of_day=scanner.hour_of_day,
            n_perms=args.n_perms,
            use_fast_params=args.fast,
        )
        all_results['shuffled_target'] = shuffle_res
        logger.info(f"  Test 1 complete in {time.time() - t0:.0f}s")
        gc.collect()

    # ---- TEST 2: Lagged features ----
    if 'lag' in args.tests:
        logger.info("\n" + "#" * 80)
        logger.info("# STARTING TEST 2: LAGGED-FEATURE TEST")
        logger.info("#" * 80)
        t0 = time.time()
        lag_res = run_lagged_feature_test(
            features=features_clean,
            target=target,
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            hour_of_day=scanner.hour_of_day,
            lags=args.lags,
        )
        all_results['lagged_feature'] = lag_res
        logger.info(f"  Test 2 complete in {time.time() - t0:.0f}s")
        gc.collect()

    # ---- TEST 3: Cross-day gap ----
    if 'cross_day' in args.tests:
        logger.info("\n" + "#" * 80)
        logger.info("# STARTING TEST 3: CROSS-DAY GAP TEST")
        logger.info("#" * 80)
        t0 = time.time()
        gap_res = run_cross_day_gap_test(
            features=features_clean,
            target=target,
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            hour_of_day=scanner.hour_of_day,
        )
        all_results['cross_day_gap'] = gap_res
        logger.info(f"  Test 3 complete in {time.time() - t0:.0f}s")
        gc.collect()

    # ---- TEST 4: Feature autocorrelation ----
    if 'autocorr' in args.tests:
        logger.info("\n" + "#" * 80)
        logger.info("# STARTING TEST 4: FEATURE AUTOCORRELATION ANALYSIS")
        logger.info("#" * 80)
        t0 = time.time()
        ac_res = run_feature_autocorrelation_test(
            features=features_clean,
            target=target,
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            top_features=top_features_in_clean,
            lags=[0, 1, 5, 10, 30, 50, 100],
        )
        all_results['feature_autocorrelation'] = ac_res
        logger.info(f"  Test 4 complete in {time.time() - t0:.0f}s")
        gc.collect()

    # ---- TEST 5: Retrodiction ----
    if 'retro' in args.tests:
        logger.info("\n" + "#" * 80)
        logger.info("# STARTING TEST 5: RETRODICTION TEST")
        logger.info("#" * 80)
        t0 = time.time()
        retro_res = run_retrodiction_test(
            features=features_clean,
            target_forward=target,
            mid_prices=scanner.mid_prices,
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            hour_of_day=scanner.hour_of_day,
            horizon_bars=30,  # 3s at 100ms
        )
        all_results['retrodiction'] = retro_res
        logger.info(f"  Test 5 complete in {time.time() - t0:.0f}s")
        gc.collect()

    # =========================================================
    # Summary
    # =========================================================
    elapsed_total = time.time() - t0_total
    all_results['elapsed_sec'] = elapsed_total

    summary = generate_leakage_summary(all_results)
    all_results['summary'] = summary

    logger.info(summary)

    # =========================================================
    # Save results
    # =========================================================
    timestamp = all_results['timestamp']
    out_path = RESULTS_DIR / f"leakage_audit_{timestamp}.json"

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=_json_safe)

    logger.info(f"\nResults saved to: {out_path}")
    logger.info(f"Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    return all_results


if __name__ == '__main__':
    main()
