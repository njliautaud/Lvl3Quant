"""
Forward vs Backward Decomposition — Separating Predictive from Retrodictive Features

Leakage audit found: Backward IC=0.389 vs Forward IC=0.114 (ratio=3.43)
This script dissects WHICH features drive the forward signal and which are
contaminated by backward (retrodictive) information.

Experiments:
1. Per-feature forward/backward IC ratio (top-30 features)
   - Train single-feature models on both fwd target (ret_3s) and bwd target (ret_3s_backward)
   - Ratio > 1.0 -> forward-dominant (genuine prediction)
   - Ratio < 1.0 -> backward-dominant (retrodiction / autocorr artifact)

2. Forward-only model
   - Use only features with fwd/bwd ratio > 0.5
   - Compare IC to full 129-feature model

3. Flow-only model
   - Use only dynamic flow features (OFI, trade imbalance, VPIN, etc.)
   - These change rapidly and should have less backward contamination

4. Orthogonalized features
   - Regress out backward target from each feature (per-day linear regression)
   - Train on residuals
   - If IC survives -> forward signal is truly independent of backward info

5. Lead-lag decomposition
   - For top-9 features, compute Spearman IC at offsets -50 to +50
   - Reveals where the peak predictive relationship is:
     - offset < 0 -> feature leads price (genuine prediction)
     - offset = 0 -> contemporaneous (same event, not useful)
     - offset > 0 -> feature lags price (backward contamination)

Usage:
    python alpha_discovery/run_forward_decomp.py
    python alpha_discovery/run_forward_decomp.py --experiments 1 2 3
    python alpha_discovery/run_forward_decomp.py --top-n 20
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
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache, compute_return_targets, EXCLUDE_FEATURES_DIRECTION,
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
        logging.FileHandler(RESULTS_DIR / 'forward_decomp.log', mode='a', encoding='utf-8'),
    ]
)
logger = logging.getLogger("forward_decomp")

# ============================================================================
# CONSTANTS
# ============================================================================

# LightGBM params for single-feature models (faster, less trees needed)
FAST_LGBM_PARAMS = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 1.0,  # single feature, so 1.0
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 100,
    'verbose': -1,
    'n_jobs': -1,
}

# Standard params for multi-feature models
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

# Top 30 features to decompose (ranked by prior full-model importance)
TOP_30_FEATURES = [
    'depth_ratio_l1',
    'ask_L1_orders',
    'bid_L1_conc',
    'ask_L1_conc',
    'bid_L1_orders',
    'total_bid_vol',
    'bid_L5_conc',
    'ret_100',
    'ofi_5',
    'ask_L4_orders',
    'depth_ratio_l3',
    'ofi_50',
    'depth_ratio_l5',
    'bid_pressure',
    'ask_L5_orders',
    'vol_regime',
    'depth_concentration',
    'bid_L5_orders',
    'ofi_20',
    'total_ask_vol',
    'ask_L5_conc',
    'ask_L2_orders',
    'bid_L4_conc',
    'bid_L2_conc',
    'ask_L3_orders',
    'bid_slope',
    'ask_slope',
    'bid_L3_conc',
    'bid_L4_orders',
    'bid_L2_orders',
]

# Top 9 features for lead-lag decomposition (experiment 5)
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

# Flow/dynamic feature set (experiment 3)
FLOW_FEATURES = [
    # Order flow imbalance
    'ofi_5', 'ofi_10', 'ofi_20',
    # Trade imbalance
    'trade_imb_5', 'trade_imb_10', 'trade_imb_20',
    # Net flow
    'net_flow_5', 'net_flow_10', 'net_flow_20',
    # Aggressive imbalance
    'aggr_imb_5', 'aggr_imb_10', 'aggr_imb_20',
    # Cancel ratio
    'cancel_ratio_5',
    # VPIN
    'vpin_50', 'vpin_100', 'vpin_200',
]


# ============================================================================
# BACKWARD TARGET CONSTRUCTION
# ============================================================================

def build_backward_target(scanner: MBOAlphaScanner) -> np.ndarray:
    """
    Backward target: return from bar (i-30) to bar i (the past 3s return).

    This is the RETRODICTION target. A model that predicts this is learning
    to look backward, not forward. Any feature with high backward IC but
    low forward IC is a retrodiction artifact (backward contamination).

    30 bars at 100ms/bar = 3 seconds backward.
    """
    log_mid = np.log(np.maximum(scanner.mid_prices, 1.0))
    N = len(log_mid)
    target_backward = np.full(N, np.nan, dtype=np.float32)

    # Backward return: log(mid[i]) - log(mid[i-30])
    target_backward[30:] = (log_mid[30:] - log_mid[:-30]).astype(np.float32)

    # NaN out day boundaries (first 30 bars of each day)
    n_days = len(scanner.day_boundaries) - 1
    for d in range(n_days):
        start = scanner.day_boundaries[d]
        end = min(start + 30, scanner.day_boundaries[d + 1])
        target_backward[start:end] = np.nan

    valid = np.isfinite(target_backward).sum()
    logger.info(f"Backward target: {valid:,} valid bars (3s backward return)")
    return target_backward


# ============================================================================
# EXPERIMENT 1: PER-FEATURE FORWARD vs BACKWARD IC
# ============================================================================

def compute_single_feature_ic(
    feature_col: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    min_train_days: int = 3,
    lgbm_params: dict = None,
) -> float:
    """
    Train a single-feature LightGBM model walk-forward and return mean IC.
    Uses the same walk_forward_evaluate infrastructure for consistency.
    """
    features_2d = feature_col.reshape(-1, 1)
    params = lgbm_params or FAST_LGBM_PARAMS

    res = walk_forward_evaluate(
        features=features_2d,
        target=target,
        day_boundaries=day_boundaries,
        feature_names=['feat'],
        model_type='lgbm',
        min_train_days=min_train_days,
        lgbm_params=params,
    )

    if 'error' in res:
        return np.nan

    return res['ic']


def run_experiment1_fwd_bwd_per_feature(
    scanner: MBOAlphaScanner,
    target_fwd: np.ndarray,
    target_bwd: np.ndarray,
    features_clean: np.ndarray,
    names_clean: List[str],
    top_n: int = 30,
) -> List[dict]:
    """
    For each of the top N features, compute:
    - fwd_IC: IC predicting forward ret_3s
    - bwd_IC: IC predicting backward ret_3s (retrodiction)
    - ratio: fwd_IC / bwd_IC
    - classification: forward-dominant (ratio > 1) or backward-dominant
    """
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: Per-feature forward vs backward IC")
    logger.info("=" * 70)

    # Select top-N features that are available
    selected = []
    for fn in TOP_30_FEATURES:
        if fn in names_clean and len(selected) < top_n:
            selected.append(fn)

    # Fill up if we don't have enough from the ranked list
    if len(selected) < top_n:
        for fn in names_clean:
            if fn not in selected and len(selected) < top_n:
                selected.append(fn)

    logger.info(f"  Testing {len(selected)} features")
    logger.info(f"  {'Feature':<30s} {'FwdIC':>8s} {'BwdIC':>8s} {'Ratio':>8s} {'Class':<20s}")
    logger.info(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    results = []

    for i, fn in enumerate(selected):
        if fn not in names_clean:
            logger.warning(f"  [{i+1}/{len(selected)}] {fn}: not in feature set, skipping")
            continue

        feat_idx = names_clean.index(fn)
        feat_col = features_clean[:, feat_idx].astype(np.float64)

        # Replace NaN with 0 for model training (features shouldn't have NaN,
        # but just in case)
        feat_col_clean = np.where(np.isfinite(feat_col), feat_col, 0.0)

        t0 = time.time()

        fwd_ic = compute_single_feature_ic(
            feature_col=feat_col_clean,
            target=target_fwd,
            day_boundaries=scanner.day_boundaries,
            lgbm_params=FAST_LGBM_PARAMS,
        )

        bwd_ic = compute_single_feature_ic(
            feature_col=feat_col_clean,
            target=target_bwd,
            day_boundaries=scanner.day_boundaries,
            lgbm_params=FAST_LGBM_PARAMS,
        )

        elapsed = time.time() - t0

        # Compute ratio (use absolute values since ICs can be negative)
        fwd_abs = abs(fwd_ic) if np.isfinite(fwd_ic) else 0.0
        bwd_abs = abs(bwd_ic) if np.isfinite(bwd_ic) else 0.0

        if bwd_abs > 1e-6:
            ratio = fwd_abs / bwd_abs
        elif fwd_abs > 1e-6:
            ratio = np.inf
        else:
            ratio = 1.0  # Both near zero

        # Classify
        if ratio > 1.5:
            classification = "FORWARD-DOMINANT"
        elif ratio > 0.7:
            classification = "balanced"
        elif ratio > 0.3:
            classification = "backward-leaning"
        else:
            classification = "BACKWARD-DOMINANT"

        rec = {
            'feature': fn,
            'fwd_ic': float(fwd_ic) if np.isfinite(fwd_ic) else None,
            'bwd_ic': float(bwd_ic) if np.isfinite(bwd_ic) else None,
            'fwd_ic_abs': fwd_abs,
            'bwd_ic_abs': bwd_abs,
            'ratio': float(ratio) if np.isfinite(ratio) else 99.0,
            'classification': classification,
        }
        results.append(rec)

        ratio_str = f"{ratio:.3f}" if np.isfinite(ratio) else "  inf"
        fwd_str = f"{fwd_ic:+.4f}" if np.isfinite(fwd_ic) else "   nan"
        bwd_str = f"{bwd_ic:+.4f}" if np.isfinite(bwd_ic) else "   nan"
        logger.info(
            f"  [{i+1:2d}/{len(selected)}] {fn:<30s} {fwd_str:>8s} {bwd_str:>8s} "
            f"{ratio_str:>8s} {classification:<20s} ({elapsed:.1f}s)"
        )

        gc.collect()

    # Summary
    results_sorted = sorted(results, key=lambda x: x['ratio'], reverse=True)
    fwd_dominant = [r for r in results if r['classification'] == 'FORWARD-DOMINANT']
    bwd_dominant = [r for r in results if r['classification'] == 'BACKWARD-DOMINANT']

    logger.info("")
    logger.info(f"  Forward-dominant features ({len(fwd_dominant)}):")
    for r in fwd_dominant:
        logger.info(f"    {r['feature']:<30s} ratio={r['ratio']:.3f} fwd_IC={r['fwd_ic_abs']:.4f}")

    logger.info(f"  Backward-dominant features ({len(bwd_dominant)}):")
    for r in bwd_dominant:
        logger.info(f"    {r['feature']:<30s} ratio={r['ratio']:.3f} bwd_IC={r['bwd_ic_abs']:.4f}")

    return results_sorted


# ============================================================================
# EXPERIMENT 2: FORWARD-ONLY MODEL
# ============================================================================

def run_experiment2_forward_only_model(
    scanner: MBOAlphaScanner,
    target_fwd: np.ndarray,
    features_clean: np.ndarray,
    names_clean: List[str],
    exp1_results: List[dict],
    ratio_threshold: float = 0.5,
) -> dict:
    """
    Build a model using only features where fwd_IC/bwd_IC ratio > threshold.
    Compare IC to the full 129-feature model baseline.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT 2: Forward-only model (ratio > {ratio_threshold})")
    logger.info("=" * 70)

    # Select features passing the threshold
    fwd_features = [r['feature'] for r in exp1_results if r['ratio'] >= ratio_threshold]
    logger.info(f"  Features with ratio >= {ratio_threshold}: {len(fwd_features)}")
    for fn in fwd_features:
        r = next((x for x in exp1_results if x['feature'] == fn), None)
        if r:
            logger.info(f"    {fn:<30s} ratio={r['ratio']:.3f}")

    if len(fwd_features) == 0:
        logger.warning("  No features pass the threshold. Skipping.")
        return {'error': 'No features pass threshold', 'threshold': ratio_threshold}

    # Build feature subset
    fwd_feat_idx = [names_clean.index(fn) for fn in fwd_features if fn in names_clean]
    fwd_feat_idx = [i for i in fwd_feat_idx if i < features_clean.shape[1]]
    feat_subset = features_clean[:, fwd_feat_idx]
    names_subset = [names_clean[i] for i in fwd_feat_idx]

    logger.info(f"  Training forward-only model on {len(names_subset)} features...")

    res = walk_forward_evaluate(
        features=feat_subset,
        target=target_fwd,
        day_boundaries=scanner.day_boundaries,
        feature_names=names_subset,
        model_type='lgbm',
        lgbm_params=STANDARD_LGBM_PARAMS,
    )

    if 'error' in res:
        logger.warning(f"  Forward-only model failed: {res['error']}")
        return res

    logger.info(f"  Forward-only model: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                f"ICIR={res['icir']:.2f} FoldC={res['fold_con']:.0%}")
    logger.info(f"  Fold ICs: {[f'{x:+.4f}' for x in res['fold_ics']]}")

    return {
        'n_features': len(names_subset),
        'features': names_subset,
        'ratio_threshold': ratio_threshold,
        'ic': res['ic'],
        'ic_std': res['ic_std'],
        'icir': res['icir'],
        'tstat': res['tstat'],
        'pvalue': res['pvalue'],
        'fold_ics': res['fold_ics'],
        'fold_con': res['fold_con'],
        'n_folds': res['n_folds'],
        'n_preds': res['n_preds'],
        'top_features': res.get('top_features', []),
    }


# ============================================================================
# EXPERIMENT 3: FLOW-ONLY MODEL
# ============================================================================

def run_experiment3_flow_only_model(
    scanner: MBOAlphaScanner,
    target_fwd: np.ndarray,
    features_clean: np.ndarray,
    names_clean: List[str],
) -> dict:
    """
    Train using only dynamic flow features (OFI, trade imbalance, VPIN, etc.).
    These features change rapidly and should have less backward contamination.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: Flow-only model")
    logger.info("=" * 70)

    available_flow = [fn for fn in FLOW_FEATURES if fn in names_clean]
    unavailable = [fn for fn in FLOW_FEATURES if fn not in names_clean]

    logger.info(f"  Flow features available: {len(available_flow)}/{len(FLOW_FEATURES)}")
    if available_flow:
        logger.info(f"    Available: {', '.join(available_flow)}")
    if unavailable:
        logger.info(f"    Not in dataset: {', '.join(unavailable)}")

    if len(available_flow) == 0:
        logger.warning("  No flow features found in dataset. Skipping.")
        return {'error': 'No flow features available'}

    flow_feat_idx = [names_clean.index(fn) for fn in available_flow]
    feat_subset = features_clean[:, flow_feat_idx]

    logger.info(f"  Training flow-only model on {len(available_flow)} features...")

    # Use full LGBM params but allow all feature usage (not subsampling columns)
    flow_params = dict(STANDARD_LGBM_PARAMS)
    flow_params['colsample_bytree'] = 1.0  # Use all flow features always

    res = walk_forward_evaluate(
        features=feat_subset,
        target=target_fwd,
        day_boundaries=scanner.day_boundaries,
        feature_names=available_flow,
        model_type='lgbm',
        lgbm_params=flow_params,
    )

    if 'error' in res:
        logger.warning(f"  Flow-only model failed: {res['error']}")
        return res

    logger.info(f"  Flow-only model: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                f"ICIR={res['icir']:.2f} FoldC={res['fold_con']:.0%}")
    logger.info(f"  Fold ICs: {[f'{x:+.4f}' for x in res['fold_ics']]}")

    # Feature importance within flow model
    if res.get('top_features'):
        logger.info("  Top flow features by importance:")
        for fname, fimp in res['top_features']:
            logger.info(f"    {fname:<25s}: {fimp:.0f}")

    return {
        'n_features': len(available_flow),
        'features': available_flow,
        'ic': res['ic'],
        'ic_std': res['ic_std'],
        'icir': res['icir'],
        'tstat': res['tstat'],
        'pvalue': res['pvalue'],
        'fold_ics': res['fold_ics'],
        'fold_con': res['fold_con'],
        'n_folds': res['n_folds'],
        'n_preds': res['n_preds'],
        'top_features': res.get('top_features', []),
    }


# ============================================================================
# EXPERIMENT 4: ORTHOGONALIZED FEATURES
# ============================================================================

def orthogonalize_features(
    features: np.ndarray,
    target_bwd: np.ndarray,
    day_boundaries: list,
) -> np.ndarray:
    """
    For each feature and each day, regress out the backward target.
    Residual = feature - beta * backward_target

    This removes the component of the feature that is linearly explained
    by the backward return. What remains is the part of the feature that
    is INDEPENDENT of past price history.

    Per-day regression is important: prevents the backward target from
    absorbing signal that varies across days at different overall levels.
    """
    n_bars, n_feats = features.shape
    residuals = features.copy().astype(np.float64)
    n_days = len(day_boundaries) - 1

    for d in range(n_days):
        start = day_boundaries[d]
        end = day_boundaries[d + 1]

        y = target_bwd[start:end]
        X = features[start:end]

        # Valid mask: need both feature and backward target to be finite
        valid = np.isfinite(y)

        if valid.sum() < 50:
            continue  # Not enough data to fit regression

        y_day = y[valid]

        # For each feature: fit y_bwd = alpha + beta * feat, subtract beta * feat
        for j in range(n_feats):
            x_j = X[valid, j]
            x_j_full = X[:, j]

            x_valid = np.isfinite(x_j) & np.isfinite(y_day)
            if x_valid.sum() < 20:
                continue

            x_fit = x_j[x_valid]
            y_fit = y_day[x_valid]

            # OLS: beta = cov(x, y) / var(x)
            x_demeaned = x_fit - np.mean(x_fit)
            var_x = np.mean(x_demeaned ** 2)
            if var_x < 1e-12:
                continue

            cov_xy = np.mean(x_demeaned * (y_fit - np.mean(y_fit)))
            beta = cov_xy / var_x

            # Residualize: subtract beta * x from ALL bars in this day
            # (not just the valid ones) to preserve alignment
            x_full_day = x_j_full.astype(np.float64)
            finite_mask = np.isfinite(x_full_day)
            residuals[start:end][finite_mask, j] -= beta * x_full_day[finite_mask]

    return residuals


def run_experiment4_orthogonalized(
    scanner: MBOAlphaScanner,
    target_fwd: np.ndarray,
    target_bwd: np.ndarray,
    features_clean: np.ndarray,
    names_clean: List[str],
    top_n: int = 30,
) -> dict:
    """
    Regress out backward target from each feature, then train model on residuals.
    If forward IC survives orthogonalization -> signal is truly independent of
    backward information (genuine forward prediction).
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: Orthogonalized features (backward target removed)")
    logger.info("=" * 70)

    # Use top-N features from ranked list that are available
    selected = []
    for fn in TOP_30_FEATURES:
        if fn in names_clean and len(selected) < top_n:
            selected.append(fn)
    if len(selected) < top_n:
        for fn in names_clean:
            if fn not in selected and len(selected) < top_n:
                selected.append(fn)

    feat_idx = [names_clean.index(fn) for fn in selected if fn in names_clean]
    feat_subset = features_clean[:, feat_idx]
    names_subset = [names_clean[i] for i in feat_idx]

    logger.info(f"  Orthogonalizing {len(names_subset)} features against backward target...")
    t0 = time.time()

    orth_features = orthogonalize_features(
        features=feat_subset,
        target_bwd=target_bwd,
        day_boundaries=scanner.day_boundaries,
    )

    logger.info(f"  Orthogonalization done in {time.time() - t0:.1f}s")

    # Train model on orthogonalized features
    logger.info(f"  Training on orthogonalized features...")

    # Original model (for comparison)
    res_orig = walk_forward_evaluate(
        features=feat_subset.astype(np.float64),
        target=target_fwd,
        day_boundaries=scanner.day_boundaries,
        feature_names=names_subset,
        model_type='lgbm',
        lgbm_params=STANDARD_LGBM_PARAMS,
    )

    # Orthogonalized model
    res_orth = walk_forward_evaluate(
        features=orth_features,
        target=target_fwd,
        day_boundaries=scanner.day_boundaries,
        feature_names=names_subset,
        model_type='lgbm',
        lgbm_params=STANDARD_LGBM_PARAMS,
    )

    if 'error' in res_orig:
        logger.warning(f"  Original model failed: {res_orig['error']}")
    else:
        logger.info(f"  Original (non-orth):   IC={res_orig['ic']:.4f} "
                    f"t={res_orig['tstat']:.2f} ICIR={res_orig['icir']:.2f}")

    if 'error' in res_orth:
        logger.warning(f"  Orthogonalized model failed: {res_orth['error']}")
    else:
        logger.info(f"  Orthogonalized model:  IC={res_orth['ic']:.4f} "
                    f"t={res_orth['tstat']:.2f} ICIR={res_orth['icir']:.2f}")

    if 'error' not in res_orig and 'error' not in res_orth:
        retained_pct = res_orth['ic'] / res_orig['ic'] * 100 if abs(res_orig['ic']) > 1e-6 else 0.0
        logger.info(f"  IC retained after orthogonalization: {retained_pct:.1f}%")
        if retained_pct > 70:
            logger.info("  VERDICT: Forward IC is GENUINE — survives backward removal")
        elif retained_pct > 30:
            logger.info("  VERDICT: Partial forward signal — mixed genuine + backward leakage")
        else:
            logger.info("  VERDICT: Forward IC largely EXPLAINED by backward target")

    def safe_res(res):
        if 'error' in res:
            return res
        return {k: v for k, v in res.items() if k not in ('predictions', 'actuals', 'pred_indices')}

    return {
        'n_features': len(names_subset),
        'features': names_subset,
        'original': safe_res(res_orig),
        'orthogonalized': safe_res(res_orth),
        'ic_retained_pct': (
            res_orth['ic'] / res_orig['ic'] * 100
            if 'error' not in res_orig and 'error' not in res_orth
            and abs(res_orig['ic']) > 1e-6
            else None
        ),
    }


# ============================================================================
# EXPERIMENT 5: LEAD-LAG DECOMPOSITION
# ============================================================================

def compute_lead_lag_profile(
    feature_col: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    offsets: np.ndarray,
    min_samples: int = 1000,
) -> np.ndarray:
    """
    For each offset in [-max_lag, +max_lag], compute Spearman IC between
    feature[i] and target[i + offset].

    offset < 0: feature is AHEAD of price (feature[i] predicts target[i - |offset|])
                -> feature LEADS price (genuine forward prediction)
    offset = 0: contemporaneous
    offset > 0: feature LAGS price (target[i - offset] explains feature[i])
                -> backward contamination

    Day boundaries: never cross days for alignment.
    """
    n = len(feature_col)
    ics = np.full(len(offsets), np.nan)

    # Precompute valid mask for feature and target
    feat_valid = np.isfinite(feature_col)
    tgt_valid = np.isfinite(target)

    for k, offset in enumerate(offsets):
        offset = int(offset)

        # feature[i] vs target[i + offset]
        # Valid range: max(0, -offset) .. min(n, n - offset)
        if offset >= 0:
            feat_slice = feature_col[:n - offset] if offset > 0 else feature_col
            tgt_slice = target[offset:] if offset > 0 else target
            valid_range_start = 0
            valid_range_end = n - offset
        else:
            abs_offset = abs(offset)
            feat_slice = feature_col[abs_offset:]
            tgt_slice = target[:n - abs_offset]
            valid_range_start = abs_offset
            valid_range_end = n

        if len(feat_slice) == 0 or len(tgt_slice) == 0:
            continue

        # Mask out cross-day pairs
        day_cross_mask = np.zeros(len(feat_slice), dtype=bool)
        for d in range(len(day_boundaries) - 1):
            day_start = day_boundaries[d]
            day_end = day_boundaries[d + 1]

            if offset >= 0:
                # feature[i], target[i+offset]: cross-day if day boundary falls between i and i+offset
                # i is in [day_start, day_end), i+offset >= day_end -> cross-day
                # Translated: slice index = i, so slice indices in [day_start, day_end)
                # Cross if i + offset >= day_end -> i >= day_end - offset
                cross_start = max(0, day_end - offset - valid_range_start)
                cross_end = min(len(feat_slice), day_end - valid_range_start)
                if cross_start < cross_end:
                    day_cross_mask[cross_start:cross_end] = True
            else:
                abs_off = abs(offset)
                # feature[i+abs_off], target[i]: i in [valid_range_start, valid_range_end)
                # feature index = i + abs_off, if feature crosses day boundary
                cross_start = max(0, day_end - abs_off - valid_range_start)
                cross_end = min(len(feat_slice), day_end - valid_range_start)
                if cross_start < cross_end:
                    day_cross_mask[cross_start:cross_end] = True

        # Build combined valid mask
        if offset >= 0:
            f_valid_slice = feat_valid[:n - offset] if offset > 0 else feat_valid
            t_valid_slice = tgt_valid[offset:] if offset > 0 else tgt_valid
        else:
            abs_offset = abs(offset)
            f_valid_slice = feat_valid[abs_offset:]
            t_valid_slice = tgt_valid[:n - abs_offset]

        combined_valid = f_valid_slice & t_valid_slice & ~day_cross_mask

        if combined_valid.sum() < min_samples:
            continue

        f = feat_slice[combined_valid]
        t = tgt_slice[combined_valid]

        try:
            ic, _ = spearmanr(f, t)
            if np.isfinite(ic):
                ics[k] = ic
        except Exception:
            pass

    return ics


def run_experiment5_lead_lag(
    scanner: MBOAlphaScanner,
    target_fwd: np.ndarray,
    features_clean: np.ndarray,
    names_clean: List[str],
    max_lag: int = 50,
    save_plots: bool = True,
) -> dict:
    """
    For each of the top-9 features, compute IC at offsets from -max_lag to +max_lag.

    Interpretation:
    - Peak at offset < 0: feature LEADS price -> genuine prediction
    - Peak at offset = 0: contemporaneous -> reacting to same event
    - Peak at offset > 0: feature LAGS price -> backward contamination

    Note: offset=0 = IC(feature[i], target[i]) = the current forward IC
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"EXPERIMENT 5: Lead-lag decomposition (offset: -{max_lag} to +{max_lag})")
    logger.info("=" * 70)

    offsets = np.arange(-max_lag, max_lag + 1)
    results = {}

    for fn in TOP_9_FEATURES:
        if fn not in names_clean:
            logger.warning(f"  {fn}: not in feature set, skipping")
            continue

        feat_idx = names_clean.index(fn)
        feat_col = features_clean[:, feat_idx].astype(np.float64)

        t0 = time.time()
        ics = compute_lead_lag_profile(
            feature_col=feat_col,
            target=target_fwd,
            day_boundaries=scanner.day_boundaries,
            offsets=offsets,
        )

        elapsed = time.time() - t0

        # Find peak
        valid_mask = np.isfinite(ics)
        if valid_mask.sum() == 0:
            logger.warning(f"  {fn}: no valid ICs computed")
            results[fn] = {'error': 'No valid ICs'}
            continue

        ic_at_0 = ics[max_lag] if np.isfinite(ics[max_lag]) else np.nan
        peak_idx = np.nanargmax(np.abs(ics))
        peak_offset = offsets[peak_idx]
        peak_ic = ics[peak_idx]

        # IC in three zones
        neg_zone = offsets < 0
        pos_zone = offsets > 0
        zero_zone = offsets == 0

        ic_neg_mean = float(np.nanmean(ics[neg_zone])) if neg_zone.sum() > 0 else 0.0
        ic_pos_mean = float(np.nanmean(ics[pos_zone])) if pos_zone.sum() > 0 else 0.0
        ic_at_zero = float(ics[max_lag]) if np.isfinite(ics[max_lag]) else 0.0

        # Asymmetry: if IC[neg] > IC[pos] -> feature leads price (good)
        lead_lag_ratio = ic_neg_mean / ic_pos_mean if abs(ic_pos_mean) > 1e-6 else np.inf

        # Classification
        if peak_offset < -5:
            direction = "LEADS price (genuine prediction)"
        elif peak_offset < 0:
            direction = "weakly leads price"
        elif peak_offset == 0:
            direction = "contemporaneous (same event)"
        elif peak_offset <= 5:
            direction = "weakly lags price"
        else:
            direction = "LAGS price (backward contamination)"

        logger.info(f"  {fn}:")
        logger.info(f"    Peak IC={peak_ic:+.4f} at offset={peak_offset:+d} bars "
                    f"({peak_offset * 0.1:.1f}s) -- {direction}")
        logger.info(f"    IC at 0 (forward, offset=0): {ic_at_zero:+.4f}")
        logger.info(f"    Mean IC negative offsets (feature leads): {ic_neg_mean:+.4f}")
        logger.info(f"    Mean IC positive offsets (feature lags):  {ic_pos_mean:+.4f}")
        logger.info(f"    Lead/lag asymmetry ratio: {lead_lag_ratio:.3f}")
        logger.info(f"    Elapsed: {elapsed:.1f}s")

        # ASCII profile (condensed)
        profile_str = _format_lead_lag_ascii(ics, offsets, max_lag)
        logger.info(f"    Profile: {profile_str}")

        results[fn] = {
            'offsets': offsets.tolist(),
            'ics': [float(x) if np.isfinite(x) else None for x in ics],
            'peak_offset': int(peak_offset),
            'peak_ic': float(peak_ic) if np.isfinite(peak_ic) else None,
            'ic_at_zero': float(ic_at_zero) if np.isfinite(ic_at_zero) else None,
            'ic_neg_mean': ic_neg_mean,
            'ic_pos_mean': ic_pos_mean,
            'lead_lag_ratio': float(lead_lag_ratio) if np.isfinite(lead_lag_ratio) else None,
            'direction': direction,
        }

        gc.collect()

    # Save lead-lag plots if matplotlib is available
    if save_plots:
        _save_lead_lag_plots(results, offsets, RESULTS_DIR)

    return results


def _format_lead_lag_ascii(ics: np.ndarray, offsets: np.ndarray, max_lag: int) -> str:
    """
    Compact ASCII representation of the IC profile across offsets.
    Sample every ~5 steps for readability.
    """
    step = max(1, max_lag // 10)
    sample_indices = range(0, len(offsets), step)
    parts = []
    for i in sample_indices:
        ic = ics[i]
        offset = offsets[i]
        if np.isfinite(ic):
            parts.append(f"[{offset:+d}:{ic:+.3f}]")
    return " ".join(parts[:20])  # Limit length


def _save_lead_lag_plots(results: dict, offsets: np.ndarray, out_dir: Path) -> None:
    """Save lead-lag IC profile plots as PNG files."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("  matplotlib not available, skipping plots")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_dir = out_dir / 'lead_lag_plots'
    plot_dir.mkdir(exist_ok=True)

    # Individual plots
    for fn, res in results.items():
        if 'error' in res:
            continue
        ics = np.array([x if x is not None else np.nan for x in res['ics']])
        offs = np.array(res['offsets']) * 0.1  # Convert to seconds

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(offs, ics, 'b-', linewidth=1.5, label='IC(feature[i], target[i+offset])')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='r', linewidth=1.0, linestyle='--', label='offset=0 (forward IC)')

        peak_sec = res['peak_offset'] * 0.1
        peak_ic = res['peak_ic']
        if peak_ic is not None:
            ax.axvline(peak_sec, color='g', linewidth=1.0, linestyle=':', label=f'Peak at {peak_sec:.1f}s')
            ax.scatter([peak_sec], [peak_ic], color='g', s=60, zorder=5)

        ax.set_xlabel('Offset (seconds, negative = feature leads, positive = feature lags)')
        ax.set_ylabel('Spearman IC')
        ax.set_title(f'Lead-Lag Profile: {fn}\n{res["direction"]}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        outpath = plot_dir / f'leadlag_{fn}_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(outpath, dpi=100)
        plt.close(fig)

    # Combined comparison plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    feature_list = [fn for fn in results if 'error' not in results[fn]]
    for ax_i, fn in enumerate(feature_list[:9]):
        row, col = ax_i // 3, ax_i % 3
        ax = axes[row][col]
        res = results[fn]
        ics = np.array([x if x is not None else np.nan for x in res['ics']])
        offs = np.array(res['offsets']) * 0.1
        ax.plot(offs, ics, 'b-', linewidth=1.2)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='r', linewidth=0.8, linestyle='--')
        peak_sec = res['peak_offset'] * 0.1
        if res['peak_ic'] is not None:
            ax.axvline(peak_sec, color='g', linewidth=0.8, linestyle=':')
        ax.set_title(f'{fn}\npeak@{peak_sec:.1f}s', fontsize=9)
        ax.set_xlabel('offset (s)', fontsize=8)
        ax.set_ylabel('IC', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax_i in range(len(feature_list), 9):
        row, col = ax_i // 3, ax_i % 3
        axes[row][col].set_visible(False)

    plt.suptitle('Lead-Lag IC Profiles (Negative offset = Feature Leads Price)', fontsize=12)
    plt.tight_layout()
    combined_path = plot_dir / f'leadlag_combined_{timestamp}.png'
    plt.savefig(combined_path, dpi=100)
    plt.close(fig)
    logger.info(f"  Lead-lag plots saved to: {plot_dir}")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def format_summary(
    exp1: List[dict],
    exp2: dict,
    exp3: dict,
    exp4: dict,
    exp5: dict,
    baseline_ic: float = 0.114,
    backward_ic: float = 0.389,
) -> str:
    """Format a comprehensive ASCII summary of all decomposition results."""
    lines = [
        "",
        "=" * 75,
        "FORWARD vs BACKWARD DECOMPOSITION SUMMARY",
        "=" * 75,
        "",
        f"BASELINE: Forward IC={baseline_ic:.3f}, Backward IC={backward_ic:.3f}, "
        f"Ratio={backward_ic/baseline_ic:.2f}x",
        "",
    ]

    # Exp 1
    lines.append("EXP 1: Per-feature fwd/bwd IC ratio")
    lines.append("-" * 60)
    if exp1:
        lines.append(f"  {'Feature':<30s} {'FwdIC':>8s} {'BwdIC':>8s} {'Ratio':>8s} {'Class':<25s}")
        lines.append(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*25}")
        for r in exp1:
            fwd_s = f"{r['fwd_ic_abs']:.4f}" if r['fwd_ic'] is not None else "   nan"
            bwd_s = f"{r['bwd_ic_abs']:.4f}" if r['bwd_ic'] is not None else "   nan"
            lines.append(
                f"  {r['feature']:<30s} {fwd_s:>8s} {bwd_s:>8s} {r['ratio']:>8.3f} {r['classification']:<25s}"
            )

    lines.append("")

    # Exp 2
    lines.append("EXP 2: Forward-only model")
    lines.append("-" * 60)
    if 'error' not in exp2:
        lines.append(f"  Features used: {exp2['n_features']}")
        lines.append(f"  IC={exp2['ic']:.4f} t={exp2['tstat']:.2f} ICIR={exp2['icir']:.2f}")
        lines.append(f"  vs baseline IC={baseline_ic:.4f} -> "
                     f"{'BETTER' if exp2['ic'] > baseline_ic else 'WORSE'}")
        lines.append(f"  Features: {', '.join(exp2.get('features', [])[:10])}")
    else:
        lines.append(f"  Error: {exp2.get('error', 'unknown')}")

    lines.append("")

    # Exp 3
    lines.append("EXP 3: Flow-only model")
    lines.append("-" * 60)
    if 'error' not in exp3:
        lines.append(f"  Features used: {exp3['n_features']} ({', '.join(exp3.get('features', []))})")
        lines.append(f"  IC={exp3['ic']:.4f} t={exp3['tstat']:.2f} ICIR={exp3['icir']:.2f}")
        lines.append(f"  vs baseline IC={baseline_ic:.4f} -> "
                     f"{'BETTER' if exp3['ic'] > baseline_ic else 'WORSE'}")
        if exp3.get('top_features'):
            top3 = [f[0] for f in exp3['top_features'][:3]]
            lines.append(f"  Most important: {', '.join(top3)}")
    else:
        lines.append(f"  Error: {exp3.get('error', 'unknown')}")

    lines.append("")

    # Exp 4
    lines.append("EXP 4: Orthogonalized features")
    lines.append("-" * 60)
    if 'error' not in exp4:
        orig = exp4.get('original', {})
        orth = exp4.get('orthogonalized', {})
        if 'error' not in orig and 'error' not in orth:
            lines.append(f"  Original IC={orig['ic']:.4f}  Orthogonalized IC={orth['ic']:.4f}")
            retained = exp4.get('ic_retained_pct')
            if retained is not None:
                lines.append(f"  IC retained after backward removal: {retained:.1f}%")
                if retained > 70:
                    lines.append("  VERDICT: Forward signal is GENUINE (independent of backward info)")
                elif retained > 30:
                    lines.append("  VERDICT: MIXED - partial genuine signal + backward leakage")
                else:
                    lines.append("  VERDICT: Most IC is BACKWARD CONTAMINATION")
        else:
            lines.append(f"  Error in orig or orth: {orig.get('error')}, {orth.get('error')}")
    else:
        lines.append(f"  Error: {exp4.get('error', 'unknown')}")

    lines.append("")

    # Exp 5
    lines.append("EXP 5: Lead-lag decomposition")
    lines.append("-" * 60)
    if exp5:
        lines.append(f"  {'Feature':<30s} {'PeakOff':>8s} {'PeakIC':>8s} {'Direction'}")
        lines.append(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*40}")
        for fn, res in exp5.items():
            if 'error' in res:
                lines.append(f"  {fn:<30s}  Error: {res['error']}")
                continue
            peak_s = f"{res['peak_offset']:+d}"
            peak_ic = f"{res['peak_ic']:+.4f}" if res['peak_ic'] is not None else "  nan"
            lines.append(f"  {fn:<30s} {peak_s:>8s} {peak_ic:>8s} {res['direction']}")
    else:
        lines.append("  Not run")

    lines.append("")
    lines.append("=" * 75)
    lines.append("CONCLUSIONS:")
    lines.append("")

    # Auto-generate conclusions
    n_fwd = sum(1 for r in exp1 if r['classification'] == 'FORWARD-DOMINANT')
    n_bwd = sum(1 for r in exp1 if r['classification'] == 'BACKWARD-DOMINANT')
    lines.append(f"  1. Feature decomposition: {n_fwd} forward-dominant, {n_bwd} backward-dominant")

    if 'error' not in exp3 and exp3.get('ic', 0) > 0.05:
        lines.append(f"  2. Flow features alone achieve IC={exp3['ic']:.3f} -> meaningful flow signal")
    elif 'error' not in exp3:
        lines.append(f"  2. Flow features alone: IC={exp3.get('ic', 0):.3f} -> weak standalone signal")

    if 'error' not in exp4:
        retained = exp4.get('ic_retained_pct')
        if retained is not None:
            lines.append(f"  3. Orthogonalization retains {retained:.0f}% of IC -> "
                         f"{'genuine forward signal' if retained > 50 else 'mostly backward contamination'}")

    leads = [fn for fn, res in exp5.items() if 'error' not in res and res.get('peak_offset', 0) < -3]
    lags = [fn for fn, res in exp5.items() if 'error' not in res and res.get('peak_offset', 0) > 3]
    if leads:
        lines.append(f"  4. Features that LEAD price (genuine): {', '.join(leads)}")
    if lags:
        lines.append(f"  5. Features that LAG price (backward): {', '.join(lags)}")

    lines.append("=" * 75)

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Forward vs backward feature decomposition')
    parser.add_argument('--experiments', nargs='+', type=int,
                        default=[1, 2, 3, 4, 5],
                        help='Which experiments to run (1-5)')
    parser.add_argument('--top-n', type=int, default=30,
                        help='Number of features to analyze in exp 1 and 4')
    parser.add_argument('--max-lag', type=int, default=50,
                        help='Max lag/lead offset in bars for exp 5 (default=50 = 5s)')
    parser.add_argument('--ratio-threshold', type=float, default=0.5,
                        help='Fwd/bwd ratio threshold for exp 2 (default=0.5)')
    parser.add_argument('--baseline-fwd-ic', type=float, default=0.114,
                        help='Known baseline forward IC for reference')
    parser.add_argument('--baseline-bwd-ic', type=float, default=0.389,
                        help='Known baseline backward IC for reference')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip saving matplotlib plots')
    args = parser.parse_args()

    t0_total = time.time()

    logger.info("=" * 75)
    logger.info("FORWARD vs BACKWARD DECOMPOSITION")
    logger.info(f"  Experiments: {args.experiments}")
    logger.info(f"  Top-N features: {args.top_n}")
    logger.info(f"  Max lag: {args.max_lag} bars ({args.max_lag * 0.1:.1f}s)")
    logger.info(f"  Ratio threshold (exp2): {args.ratio_threshold}")
    logger.info(f"  Known baseline: fwd IC={args.baseline_fwd_ic}, bwd IC={args.baseline_bwd_ic}")
    logger.info("=" * 75)

    # ============================================================
    # DATA LOADING
    # ============================================================
    logger.info("\nLoading data...")
    scanner = MBOAlphaScanner()
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.error("No feature cache found. Run return_multihorizon first.")
        sys.exit(1)

    logger.info(f"Data: {scanner.features.shape[0]:,} snapshots, "
                f"{len(scanner.day_boundaries)-1} days, "
                f"{scanner.features.shape[1]} features")

    # Build clean feature set (exclude vol proxies, price, time)
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]
    logger.info(f"Features after exclusion: {len(names_clean)} "
                f"(excluded {keep_mask.size - keep_mask.sum()})")

    # ============================================================
    # TARGETS
    # ============================================================
    logger.info("\nComputing targets...")

    # Forward target (ret_3s)
    targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        horizons_sec={'3s': 3},
        include_flow_target=False,
    )
    target_fwd = targets['ret_3s']
    fwd_valid = np.isfinite(target_fwd).sum()
    logger.info(f"Forward target ret_3s: {fwd_valid:,} valid bars")

    # Backward target (ret_3s_backward)
    target_bwd = build_backward_target(scanner)

    # ============================================================
    # EXPERIMENT 1
    # ============================================================
    exp1_results = []
    if 1 in args.experiments:
        logger.info("\n[EXP 1] Per-feature forward vs backward IC ratio")
        exp1_results = run_experiment1_fwd_bwd_per_feature(
            scanner=scanner,
            target_fwd=target_fwd,
            target_bwd=target_bwd,
            features_clean=features_clean,
            names_clean=names_clean,
            top_n=args.top_n,
        )

    # ============================================================
    # EXPERIMENT 2
    # ============================================================
    exp2_results = {}
    if 2 in args.experiments:
        logger.info("\n[EXP 2] Forward-only model")
        if not exp1_results:
            logger.warning("Exp 2 requires exp 1 results. Run exp 1 first or provide results.")
            exp2_results = {'error': 'Exp 1 not run'}
        else:
            exp2_results = run_experiment2_forward_only_model(
                scanner=scanner,
                target_fwd=target_fwd,
                features_clean=features_clean,
                names_clean=names_clean,
                exp1_results=exp1_results,
                ratio_threshold=args.ratio_threshold,
            )

    # ============================================================
    # EXPERIMENT 3
    # ============================================================
    exp3_results = {}
    if 3 in args.experiments:
        logger.info("\n[EXP 3] Flow-only model")
        exp3_results = run_experiment3_flow_only_model(
            scanner=scanner,
            target_fwd=target_fwd,
            features_clean=features_clean,
            names_clean=names_clean,
        )

    # ============================================================
    # EXPERIMENT 4
    # ============================================================
    exp4_results = {}
    if 4 in args.experiments:
        logger.info("\n[EXP 4] Orthogonalized features")
        exp4_results = run_experiment4_orthogonalized(
            scanner=scanner,
            target_fwd=target_fwd,
            target_bwd=target_bwd,
            features_clean=features_clean,
            names_clean=names_clean,
            top_n=args.top_n,
        )

    # ============================================================
    # EXPERIMENT 5
    # ============================================================
    exp5_results = {}
    if 5 in args.experiments:
        logger.info("\n[EXP 5] Lead-lag decomposition")
        exp5_results = run_experiment5_lead_lag(
            scanner=scanner,
            target_fwd=target_fwd,
            features_clean=features_clean,
            names_clean=names_clean,
            max_lag=args.max_lag,
            save_plots=not args.no_plots,
        )

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - t0_total
    logger.info(f"\nAll experiments complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    summary = format_summary(
        exp1=exp1_results,
        exp2=exp2_results,
        exp3=exp3_results,
        exp4=exp4_results,
        exp5=exp5_results,
        baseline_ic=args.baseline_fwd_ic,
        backward_ic=args.baseline_bwd_ic,
    )
    logger.info(summary)
    print("\n" + summary)

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'forward_decomp_{timestamp}.json'

    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, np.ndarray):
            return [_make_serializable(x) for x in obj.tolist()]
        elif isinstance(obj, float) and not np.isfinite(obj):
            return None
        else:
            return obj

    results_out = {
        'timestamp': timestamp,
        'elapsed_sec': elapsed,
        'config': {
            'experiments': args.experiments,
            'top_n': args.top_n,
            'max_lag': args.max_lag,
            'ratio_threshold': args.ratio_threshold,
            'baseline_fwd_ic': args.baseline_fwd_ic,
            'baseline_bwd_ic': args.baseline_bwd_ic,
        },
        'data_info': {
            'n_snapshots': int(scanner.features.shape[0]),
            'n_days': int(len(scanner.day_boundaries) - 1),
            'n_features_clean': int(len(names_clean)),
        },
        'exp1_per_feature': _make_serializable(exp1_results),
        'exp2_forward_only': _make_serializable(exp2_results),
        'exp3_flow_only': _make_serializable(exp3_results),
        'exp4_orthogonalized': _make_serializable(exp4_results),
        'exp5_lead_lag': _make_serializable(exp5_results),
        'summary': summary,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results_out, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {out_path}")
    logger.info(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
