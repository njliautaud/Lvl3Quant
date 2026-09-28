"""
Anti-Dominance V2: Extended Experiments
=======================================

Building on V1 results which showed:
- Ch4 (Deep Book) is 97% redundant with Ch1 (IC drops 0.1350 → 0.0043)
- Ch6 (Queue Dynamics) is 90% redundant with Ch1 (IC drops 0.1206 → 0.0121)

V2 extends with:
1. Ch3 TOXICITY anti-dominance — VPIN/cancel-chain signal is mechanistically different
   from L1 imbalance, most likely to have true independent alpha
2. ASYMMETRIC training — train Ch4 ONLY on bars where Ch1 confidence is LOW
   (Ch1 gets confused → that's where other signals should shine)
3. DISAGREEMENT training — train on bars where Ch1 was WRONG in OOS
4. OPTIMAL WEIGHTED ensemble (Ch1 + w*Ch3_res + w*Ch4_res)

CPU-optimized for Jupiter server (no GPU required).

Usage:
    python alpha_discovery/anti_dominance_v2.py \
        --feature-cache /home/jupiter/lvl3quant/data/processed/mbo_features_cache \
        --n-days 70 --horizon ret_10s
"""

import gc
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import minimize

# Add project root
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner
from alpha_discovery.event_detector import (
    EventDetectionPipeline,
    get_event_feature_names,
    N_EVENT_FEATURES,
)
from alpha_discovery.multi_channel_alpha import (
    CH1_L1_IMBALANCE,
    CH3_TOXICITY,
    CH4_DEEP_BOOK,
    CH6_QUEUE_DYNAMICS,
    CH5_REGIME_VOL,
    CH10_SIZE_CLASS,
    CHANNEL_DEFINITIONS,
)
from alpha_discovery.run_mfe_scan import compute_mfe_targets

# Results directory
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"anti_dominance_v2_{timestamp}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("anti_dominance_v2")

HORIZONS = {
    'ret_3s': 30, 'ret_5s': 50, 'ret_10s': 100,
    'ret_30s': 300, 'ret_1m': 600,
}

# CPU-optimized LightGBM params
LGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 100,
    'max_bin': 255,           # CPU-optimized
    'verbose': -1,
    'n_jobs': -1,
    'device': 'cpu',
    'objective': 'regression',
    'metric': 'rmse',
}


def get_col_indices(feature_names, channel_features):
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    indices = []
    found = []
    for f in channel_features:
        if f in name_to_idx:
            indices.append(name_to_idx[f])
            found.append(f)
    return np.array(indices, dtype=int), found


def train_channel_walk_forward(features, target, col_indices, day_boundaries,
                                min_train_days, channel_name, lgb_params,
                                sample_mask=None):
    """
    Walk-forward LightGBM training for one channel.

    Args:
        sample_mask: Optional boolean array. If given, only train on these bars
                     but STILL predict on ALL test bars (for fair IC comparison).

    Returns:
        (oos_predictions, fold_ics, fold_preds_list, fold_actuals_list)
    """
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    oos_preds = np.full(N, np.nan, dtype=np.float32)
    fold_ics = []
    fold_preds_list = []
    fold_actuals_list = []

    logger.info(f"  [{channel_name}] Walk-forward ({n_days - min_train_days} folds)...")

    for test_day in range(min_train_days, n_days):
        train_start = day_boundaries[0]
        train_end = day_boundaries[test_day]
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_tr = features[train_start:train_end][:, col_indices]
        y_tr = target[train_start:train_end]
        X_te = features[test_start:test_end][:, col_indices]
        y_te = target[test_start:test_end]

        # Valid mask for training
        tr_valid = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)

        # Apply sample_mask to training data if provided
        if sample_mask is not None:
            tr_sample = sample_mask[train_start:train_end]
            tr_valid = tr_valid & tr_sample

        te_valid = np.isfinite(y_te) & np.all(np.isfinite(X_te), axis=1)

        if tr_valid.sum() < 500 or te_valid.sum() < 50:
            continue

        X_tr_v = X_tr[tr_valid]
        y_tr_v = y_tr[tr_valid]
        X_te_v = X_te[te_valid]
        y_te_v = y_te[te_valid]

        split = int(len(X_tr_v) * 0.8)
        if split < 200:
            continue

        try:
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(
                X_tr_v[:split], y_tr_v[:split],
                eval_set=[(X_tr_v[split:], y_tr_v[split:])],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
        except Exception as e:
            logger.warning(f"  [{channel_name}] fold {test_day} failed: {e}")
            continue

        preds = model.predict(X_te_v).astype(np.float32)
        te_indices = np.where(te_valid)[0] + test_start
        oos_preds[te_indices] = preds

        if len(preds) > 10:
            try:
                ic = float(spearmanr(preds, y_te_v)[0])
                if np.isfinite(ic):
                    fold_ics.append(ic)
                    fold_preds_list.append(preds)
                    fold_actuals_list.append(y_te_v)
            except Exception:
                pass

        del model
        gc.collect()

    return oos_preds, fold_ics, fold_preds_list, fold_actuals_list


def compute_channel_ic(fold_ics, fold_preds_list, fold_actuals_list, channel_name):
    if not fold_ics:
        return {'name': channel_name, 'ic': np.nan, 'ic_mean': np.nan, 'ic_std': np.nan,
                'icir': np.nan, 'hit_rate': np.nan, 'n_folds': 0}

    ics = np.array(fold_ics)
    ic_mean = float(ics.mean())
    ic_std = float(ics.std()) if len(ics) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 0 else 0.0

    all_preds = np.concatenate(fold_preds_list)
    all_actuals = np.concatenate(fold_actuals_list)
    valid = np.isfinite(all_preds) & np.isfinite(all_actuals)
    p, a = all_preds[valid], all_actuals[valid]

    overall_ic = float(spearmanr(p, a)[0]) if len(p) > 50 else ic_mean
    hr = float((np.sign(p) == np.sign(a)).mean()) if len(p) > 0 else 0.5

    return {
        'name': channel_name,
        'ic': overall_ic,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'hit_rate': hr,
        'n_folds': len(fold_ics),
        'n_predictions': int(len(p)),
    }


def measure_prediction_correlation(preds_a, preds_b):
    valid = np.isfinite(preds_a) & np.isfinite(preds_b)
    if valid.sum() < 100:
        return np.nan
    return float(np.corrcoef(preds_a[valid], preds_b[valid])[0, 1])


def run_experiment(args):
    logger.info("=" * 70)
    logger.info("ANTI-DOMINANCE V2 — EXTENDED EXPERIMENTS")
    logger.info(f"Horizon: {args.horizon} | N-days: {args.n_days}")
    logger.info("=" * 70)

    t_start = time.time()

    # ================================================================
    # PHASE 1: Load feature cache
    # ================================================================
    logger.info("\n[PHASE 1] Loading feature cache...")
    scanner = MBOAlphaScanner()
    load_info = scanner.load_precomputed_features(
        feature_cache_dir=args.feature_cache,
        n_days=args.n_days,
        extra_cols=N_EVENT_FEATURES,
    )
    logger.info(f"Loaded {load_info['n_days']} days, "
                f"{load_info['n_snapshots']:,} snapshots, "
                f"{load_info['n_features']} features")

    # ================================================================
    # PHASE 2: Event detection
    # ================================================================
    logger.info("\n[PHASE 2] Event detection...")
    n_base = load_info['n_features']
    pipeline = EventDetectionPipeline()
    event_features = pipeline.detect_all(
        scanner.features[:, :n_base],
        scanner.feature_names,
        scanner.day_boundaries,
    )
    augmented_names = list(scanner.feature_names) + get_event_feature_names()
    n_event = event_features.shape[1]

    if load_info.get('extra_cols', 0) >= n_event:
        scanner.features[:, n_base:n_base + n_event] = event_features
        del event_features
        gc.collect()
        features = scanner.features
    else:
        features = np.concatenate([scanner.features, event_features], axis=1)
        del event_features
        gc.collect()
    logger.info(f"Features: {features.shape}")

    # ================================================================
    # PHASE 3: Compute MFE target
    # ================================================================
    logger.info(f"\n[PHASE 3] Computing MFE target ({args.horizon})...")
    horizon_bars = HORIZONS[args.horizon]
    hz_name = args.horizon.replace('ret_', '')
    hz_sec_map = {'3s': 3, '5s': 5, '10s': 10, '30s': 30, '1m': 60}
    hz_sec = hz_sec_map.get(hz_name, 10)

    mfe_targets = compute_mfe_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec={hz_name: hz_sec},
        tick_size=0.25,
    )
    target_mfe = mfe_targets[f'mfe_net_{hz_name}']
    del mfe_targets
    gc.collect()

    n_valid = int(np.isfinite(target_mfe).sum())
    logger.info(f"MFE target: {n_valid:,} valid, std={np.nanstd(target_mfe):.4f}")

    # ================================================================
    # PHASE 4: Train Ch1 (Dominant Signal)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: Ch1 BASELINE + Ch3 TOXICITY ANTI-DOMINANCE")
    logger.info("=" * 70)

    logger.info("\n[PHASE 4] Training Ch1 (L1 Imbalance)...")
    ch1_cols, ch1_found = get_col_indices(augmented_names, CH1_L1_IMBALANCE)
    logger.info(f"  Ch1: {len(ch1_found)}/{len(CH1_L1_IMBALANCE)} features")

    t0 = time.time()
    ch1_oos_preds, ch1_fold_ics, ch1_fold_preds, ch1_fold_actuals = train_channel_walk_forward(
        features, target_mfe, ch1_cols, scanner.day_boundaries,
        args.min_train_days, 'Ch1_L1', LGB_PARAMS,
    )
    ch1_stats = compute_channel_ic(ch1_fold_ics, ch1_fold_preds, ch1_fold_actuals, 'Ch1')
    logger.info(f"  Ch1: IC={ch1_stats['ic']:.4f} ICIR={ch1_stats['icir']:.2f} "
                f"HR={ch1_stats['hit_rate']:.1%} ({time.time()-t0:.0f}s)")

    # Compute residuals
    residual_target = np.full_like(target_mfe, np.nan)
    has_pred = np.isfinite(ch1_oos_preds) & np.isfinite(target_mfe)
    residual_target[has_pred] = target_mfe[has_pred] - ch1_oos_preds[has_pred]
    logger.info(f"  Residual: {int(has_pred.sum()):,} valid, "
                f"std={np.nanstd(residual_target):.4f} (original: {np.nanstd(target_mfe):.4f})")

    del ch1_fold_preds, ch1_fold_actuals
    gc.collect()

    # ================================================================
    # PHASE 5: Ch3 Toxicity — RAW and RESIDUAL
    # ================================================================
    logger.info("\n[PHASE 5] Ch3 Toxicity — RAW target...")
    ch3_cols, ch3_found = get_col_indices(augmented_names, CH3_TOXICITY)
    logger.info(f"  Ch3: {len(ch3_found)}/{len(CH3_TOXICITY)} features")

    t0 = time.time()
    ch3_raw_preds, ch3_raw_ics, ch3_raw_fp, ch3_raw_fa = train_channel_walk_forward(
        features, target_mfe, ch3_cols, scanner.day_boundaries,
        args.min_train_days, 'Ch3_Tox_RAW', LGB_PARAMS,
    )
    ch3_raw_stats = compute_channel_ic(ch3_raw_ics, ch3_raw_fp, ch3_raw_fa, 'Ch3_raw')
    logger.info(f"  Ch3 RAW: IC={ch3_raw_stats['ic']:.4f} ICIR={ch3_raw_stats['icir']:.2f} "
                f"HR={ch3_raw_stats['hit_rate']:.1%} ({time.time()-t0:.0f}s)")

    ch1_ch3_corr_before = measure_prediction_correlation(ch1_oos_preds, ch3_raw_preds)
    logger.info(f"  Ch1-Ch3 correlation (BEFORE): {ch1_ch3_corr_before:.4f}")

    del ch3_raw_fp, ch3_raw_fa
    gc.collect()

    logger.info("\n  Ch3 Toxicity — RESIDUAL target (anti-dominance)...")
    t0 = time.time()
    ch3_res_preds, ch3_res_ics, ch3_res_fp, ch3_res_fa = train_channel_walk_forward(
        features, residual_target, ch3_cols, scanner.day_boundaries,
        args.min_train_days, 'Ch3_Tox_RESIDUAL', LGB_PARAMS,
    )
    ch3_res_stats = compute_channel_ic(ch3_res_ics, ch3_res_fp, ch3_res_fa, 'Ch3_residual')
    logger.info(f"  Ch3 RESIDUAL: IC={ch3_res_stats['ic']:.4f} ICIR={ch3_res_stats['icir']:.2f} "
                f"HR={ch3_res_stats['hit_rate']:.1%} ({time.time()-t0:.0f}s)")

    ch1_ch3_corr_after = measure_prediction_correlation(ch1_oos_preds, ch3_res_preds)
    logger.info(f"  Ch1-Ch3 correlation (AFTER): {ch1_ch3_corr_after:.4f}")

    ch3_ic_drop = 1.0 - (ch3_res_stats['ic'] / max(ch3_raw_stats['ic'], 1e-6))
    logger.info(f"\n  >>> Ch3 IC drop: {ch3_ic_drop:.1%} (Ch4 was 97%, Ch6 was 90%)")
    logger.info(f"  >>> If <50%: Ch3 HAS independent alpha from Ch1!")

    del ch3_res_fp, ch3_res_fa
    gc.collect()

    # ================================================================
    # PHASE 6: Ch4 Deep Book — RAW and RESIDUAL (for comparison with V1)
    # ================================================================
    logger.info("\n[PHASE 6] Ch4 Deep Book — RAW + RESIDUAL...")
    ch4_cols, ch4_found = get_col_indices(augmented_names, CH4_DEEP_BOOK)
    logger.info(f"  Ch4: {len(ch4_found)}/{len(CH4_DEEP_BOOK)} features")

    t0 = time.time()
    ch4_raw_preds, ch4_raw_ics, ch4_raw_fp, ch4_raw_fa = train_channel_walk_forward(
        features, target_mfe, ch4_cols, scanner.day_boundaries,
        args.min_train_days, 'Ch4_Book_RAW', LGB_PARAMS,
    )
    ch4_raw_stats = compute_channel_ic(ch4_raw_ics, ch4_raw_fp, ch4_raw_fa, 'Ch4_raw')
    logger.info(f"  Ch4 RAW: IC={ch4_raw_stats['ic']:.4f} ({time.time()-t0:.0f}s)")
    del ch4_raw_fp, ch4_raw_fa; gc.collect()

    t0 = time.time()
    ch4_res_preds, ch4_res_ics, ch4_res_fp, ch4_res_fa = train_channel_walk_forward(
        features, residual_target, ch4_cols, scanner.day_boundaries,
        args.min_train_days, 'Ch4_Book_RESIDUAL', LGB_PARAMS,
    )
    ch4_res_stats = compute_channel_ic(ch4_res_ics, ch4_res_fp, ch4_res_fa, 'Ch4_residual')
    logger.info(f"  Ch4 RESIDUAL: IC={ch4_res_stats['ic']:.4f} ({time.time()-t0:.0f}s)")
    del ch4_res_fp, ch4_res_fa; gc.collect()

    # ================================================================
    # EXPERIMENT 2: ASYMMETRIC TRAINING
    # Train Ch3/Ch4 ONLY on bars where Ch1 prediction is LOW confidence
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: ASYMMETRIC TRAINING (Ch1 low-confidence bars)")
    logger.info("=" * 70)

    # Define "low confidence" as |Ch1 prediction| < median(|Ch1 prediction|)
    abs_ch1 = np.abs(ch1_oos_preds)
    ch1_median_abs = np.nanmedian(abs_ch1)
    low_conf_mask = abs_ch1 < ch1_median_abs  # ~50% of bars
    n_low = int((low_conf_mask & has_pred).sum())
    n_total = int(has_pred.sum())
    logger.info(f"  Ch1 median |prediction|: {ch1_median_abs:.6f}")
    logger.info(f"  Low-confidence bars: {n_low:,} / {n_total:,} ({100*n_low/max(n_total,1):.1f}%)")

    # Train Ch3 asymmetric (only on low-confidence bars)
    logger.info("\n  Ch3 Toxicity — ASYMMETRIC (train only on Ch1 low-conf bars)...")
    t0 = time.time()
    ch3_asym_preds, ch3_asym_ics, ch3_asym_fp, ch3_asym_fa = train_channel_walk_forward(
        features, target_mfe, ch3_cols, scanner.day_boundaries,
        args.min_train_days, 'Ch3_ASYM', LGB_PARAMS,
        sample_mask=low_conf_mask,
    )
    ch3_asym_stats = compute_channel_ic(ch3_asym_ics, ch3_asym_fp, ch3_asym_fa, 'Ch3_asym')
    logger.info(f"  Ch3 ASYMMETRIC: IC={ch3_asym_stats['ic']:.4f} ICIR={ch3_asym_stats['icir']:.2f} "
                f"({time.time()-t0:.0f}s)")
    del ch3_asym_fp, ch3_asym_fa; gc.collect()

    # Train Ch4 asymmetric
    logger.info("\n  Ch4 Deep Book — ASYMMETRIC...")
    t0 = time.time()
    ch4_asym_preds, ch4_asym_ics, ch4_asym_fp, ch4_asym_fa = train_channel_walk_forward(
        features, target_mfe, ch4_cols, scanner.day_boundaries,
        args.min_train_days, 'Ch4_ASYM', LGB_PARAMS,
        sample_mask=low_conf_mask,
    )
    ch4_asym_stats = compute_channel_ic(ch4_asym_ics, ch4_asym_fp, ch4_asym_fa, 'Ch4_asym')
    logger.info(f"  Ch4 ASYMMETRIC: IC={ch4_asym_stats['ic']:.4f} ({time.time()-t0:.0f}s)")
    del ch4_asym_fp, ch4_asym_fa; gc.collect()

    # ================================================================
    # EXPERIMENT 3: DISAGREEMENT TRAINING
    # Train on bars where Ch1 was WRONG (sign of prediction != sign of actual)
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: DISAGREEMENT TRAINING (Ch1 wrong-prediction bars)")
    logger.info("=" * 70)

    ch1_wrong = has_pred & (np.sign(ch1_oos_preds) != np.sign(target_mfe))
    n_wrong = int(ch1_wrong.sum())
    logger.info(f"  Ch1 wrong predictions: {n_wrong:,} / {n_total:,} ({100*n_wrong/max(n_total,1):.1f}%)")

    logger.info("\n  Ch3 Toxicity — DISAGREEMENT...")
    t0 = time.time()
    ch3_disagree_preds, ch3_d_ics, ch3_d_fp, ch3_d_fa = train_channel_walk_forward(
        features, target_mfe, ch3_cols, scanner.day_boundaries,
        args.min_train_days, 'Ch3_DISAGREE', LGB_PARAMS,
        sample_mask=ch1_wrong,
    )
    ch3_disagree_stats = compute_channel_ic(ch3_d_ics, ch3_d_fp, ch3_d_fa, 'Ch3_disagree')
    logger.info(f"  Ch3 DISAGREE: IC={ch3_disagree_stats['ic']:.4f} ({time.time()-t0:.0f}s)")
    del ch3_d_fp, ch3_d_fa; gc.collect()

    # ================================================================
    # EXPERIMENT 4: ALL-CHANNEL ANTI-DOMINANCE (Ch3+Ch4+Ch5+Ch10)
    # Train each remaining channel on residuals, then combine
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 4: MULTI-CHANNEL RESIDUAL ENSEMBLE")
    logger.info("=" * 70)

    # Get Ch5 and Ch10
    ch5_cols, ch5_found = get_col_indices(augmented_names, CH5_REGIME_VOL)
    ch10_cols, ch10_found = get_col_indices(augmented_names, CH10_SIZE_CLASS)
    logger.info(f"  Ch5: {len(ch5_found)} features, Ch10: {len(ch10_found)} features")

    residual_preds = {}
    residual_stats = {}

    # Train each channel on residual
    for ch_name, ch_cols, ch_feats in [
        ('Ch3_Tox', ch3_cols, ch3_found),
        ('Ch4_Book', ch4_cols, ch4_found),
        ('Ch5_Regime', ch5_cols, ch5_found),
        ('Ch10_Size', ch10_cols, ch10_found),
    ]:
        if len(ch_cols) == 0:
            logger.info(f"  {ch_name}: No features, skipping")
            continue
        logger.info(f"\n  {ch_name} on RESIDUAL ({len(ch_cols)} features)...")
        t0 = time.time()
        preds, ics, fp, fa = train_channel_walk_forward(
            features, residual_target, ch_cols, scanner.day_boundaries,
            args.min_train_days, f'{ch_name}_RES', LGB_PARAMS,
        )
        stats = compute_channel_ic(ics, fp, fa, ch_name)
        logger.info(f"  {ch_name}: IC={stats['ic']:.4f} ICIR={stats['icir']:.2f} ({time.time()-t0:.0f}s)")
        residual_preds[ch_name] = preds
        residual_stats[ch_name] = stats
        del fp, fa; gc.collect()

    # ================================================================
    # PHASE: OPTIMAL WEIGHTED ENSEMBLE
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMAL WEIGHTED ENSEMBLE")
    logger.info("=" * 70)

    # Collect all prediction arrays
    all_channel_preds = {'Ch1': ch1_oos_preds}
    for ch_name, preds in residual_preds.items():
        all_channel_preds[f'{ch_name}_res'] = preds
    # Also include raw Ch3 (since it may have independent alpha)
    all_channel_preds['Ch3_raw'] = ch3_raw_preds

    # Find positions where ALL channels have predictions
    valid_mask = np.isfinite(target_mfe)
    for preds in all_channel_preds.values():
        valid_mask = valid_mask & np.isfinite(preds)
    n_valid_ensemble = int(valid_mask.sum())
    logger.info(f"  Common valid positions: {n_valid_ensemble:,}")

    if n_valid_ensemble > 1000:
        t_ens = target_mfe[valid_mask]
        pred_matrix = np.column_stack([
            all_channel_preds[k][valid_mask] for k in all_channel_preds
        ])
        ch_names = list(all_channel_preds.keys())

        # Try different ensemble strategies
        # Strategy A: Ch1 + Ch3_raw (simple)
        simple_ens = 0.5 * all_channel_preds['Ch1'][valid_mask] + 0.5 * all_channel_preds['Ch3_raw'][valid_mask]
        ic_simple = float(spearmanr(simple_ens, t_ens)[0])
        logger.info(f"  Simple (0.5*Ch1 + 0.5*Ch3_raw): IC={ic_simple:.4f}")

        # Strategy B: Ch1 + all residuals (equal weight)
        res_preds_valid = [all_channel_preds[k][valid_mask] for k in ch_names if '_res' in k]
        if res_preds_valid:
            res_sum = np.zeros(n_valid_ensemble, dtype=np.float32)
            for rp in res_preds_valid:
                res_sum += rp
            additive_ens = all_channel_preds['Ch1'][valid_mask] + res_sum / len(res_preds_valid)
            ic_additive = float(spearmanr(additive_ens, t_ens)[0])
            logger.info(f"  Additive (Ch1 + mean(residuals)): IC={ic_additive:.4f}")

        # Strategy C: Optimal weights via IC maximization
        # Use first 50% for fitting weights, last 50% for evaluation
        n_half = n_valid_ensemble // 2

        def neg_ic(weights):
            combined = pred_matrix[:n_half] @ weights
            try:
                return -float(spearmanr(combined, t_ens[:n_half])[0])
            except:
                return 0

        # Start with equal weights
        w0 = np.ones(len(ch_names)) / len(ch_names)
        result = minimize(neg_ic, w0, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-4})
        opt_weights = result.x

        # Normalize
        opt_weights = opt_weights / np.sum(np.abs(opt_weights))

        # Evaluate on held-out half
        opt_ens_holdout = pred_matrix[n_half:] @ opt_weights
        ic_opt_holdout = float(spearmanr(opt_ens_holdout, t_ens[n_half:])[0])

        logger.info(f"\n  Optimal weights (fit on first half, eval on second):")
        for name, w in zip(ch_names, opt_weights):
            logger.info(f"    {name}: {w:.4f}")
        logger.info(f"  Optimal ensemble IC (holdout): {ic_opt_holdout:.4f}")

        # Ch1 alone IC on holdout for comparison
        ic_ch1_holdout = float(spearmanr(
            all_channel_preds['Ch1'][valid_mask][n_half:], t_ens[n_half:]
        )[0])
        logger.info(f"  Ch1 alone IC (holdout): {ic_ch1_holdout:.4f}")
        logger.info(f"  Ensemble vs Ch1 alone: {ic_opt_holdout - ic_ch1_holdout:+.4f}")

    # ================================================================
    # FINAL REPORT
    # ================================================================
    total_elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("ANTI-DOMINANCE V2 — FINAL REPORT")
    logger.info("=" * 70)
    logger.info(f"Horizon: {args.horizon} | N-days: {args.n_days} | Time: {total_elapsed:.0f}s")

    logger.info("\n--- EXPERIMENT 1: Channel Anti-Dominance ---")
    logger.info(f"  {'Channel':<30} {'IC(raw)':<10} {'IC(residual)':<13} {'Drop%':<8} {'Corr w/Ch1'}")
    logger.info(f"  {'-'*30} {'-'*10} {'-'*13} {'-'*8} {'-'*10}")
    logger.info(f"  {'Ch1 (dominant)':<30} {ch1_stats['ic']:<10.4f} {'N/A':<13} {'N/A':<8} {'1.000'}")

    for ch_name, raw_s, res_s, corr_b, corr_a in [
        ('Ch3 Toxicity', ch3_raw_stats, ch3_res_stats, ch1_ch3_corr_before, ch1_ch3_corr_after),
        ('Ch4 Deep Book', ch4_raw_stats, ch4_res_stats,
         measure_prediction_correlation(ch1_oos_preds, ch4_raw_preds),
         measure_prediction_correlation(ch1_oos_preds, ch4_res_preds)),
    ]:
        drop = 1.0 - (res_s['ic'] / max(raw_s['ic'], 1e-6)) if not np.isnan(res_s['ic']) else np.nan
        logger.info(f"  {ch_name:<30} {raw_s['ic']:<10.4f} {res_s['ic']:<13.4f} "
                    f"{drop:<8.1%} {corr_b:.4f}→{corr_a:.4f}")

    logger.info(f"\n--- EXPERIMENT 2: Asymmetric Training ---")
    logger.info(f"  Ch3 normal IC={ch3_raw_stats['ic']:.4f} → asymmetric IC={ch3_asym_stats['ic']:.4f}")
    logger.info(f"  Ch4 normal IC={ch4_raw_stats['ic']:.4f} → asymmetric IC={ch4_asym_stats['ic']:.4f}")

    logger.info(f"\n--- EXPERIMENT 3: Disagreement Training ---")
    logger.info(f"  Ch3 trained on Ch1-wrong bars: IC={ch3_disagree_stats['ic']:.4f}")
    logger.info(f"  (Baseline Ch3 raw: IC={ch3_raw_stats['ic']:.4f})")

    logger.info(f"\n--- EXPERIMENT 4: Multi-Channel Residual ---")
    for ch_name, stats in residual_stats.items():
        logger.info(f"  {ch_name} residual: IC={stats['ic']:.4f}")

    logger.info(f"\n--- KEY CONCLUSIONS ---")
    if ch3_res_stats['ic'] > 0.02:
        logger.info("  [POSITIVE] Ch3 Toxicity has TRUE independent alpha from Ch1!")
        logger.info(f"  Ch3 residual IC={ch3_res_stats['ic']:.4f} — significant orthogonal signal")
    elif ch3_res_stats['ic'] > 0.005:
        logger.info("  [MODERATE] Ch3 has weak but positive residual signal")
    else:
        logger.info("  [NEGATIVE] Ch3 is also redundant with Ch1 — all channels measure same thing")

    logger.info(f"\n  Log: {_log_file}")
    logger.info("=" * 70)

    # Save JSON results
    json_file = RESULTS_DIR / f"anti_dominance_v2_{timestamp}.json"
    results = {
        'timestamp': timestamp,
        'horizon': args.horizon,
        'n_days': args.n_days,
        'ch1': ch1_stats,
        'ch3_raw': ch3_raw_stats,
        'ch3_residual': ch3_res_stats,
        'ch3_asymmetric': ch3_asym_stats,
        'ch3_disagree': ch3_disagree_stats,
        'ch4_raw': ch4_raw_stats,
        'ch4_residual': ch4_res_stats,
        'ch4_asymmetric': ch4_asym_stats,
        'residual_channels': {k: v for k, v in residual_stats.items()},
        'decorrelation': {
            'ch1_ch3_before': ch1_ch3_corr_before,
            'ch1_ch3_after': ch1_ch3_corr_after,
        },
    }
    # Convert numpy types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    with open(json_file, 'w') as f:
        json.dump(clean_for_json(results), f, indent=2)
    logger.info(f"Results saved: {json_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Anti-dominance V2 extended experiments')
    parser.add_argument('--feature-cache',
                        default='/home/jupiter/lvl3quant/data/processed/mbo_features_cache',
                        help='Path to feature cache')
    parser.add_argument('--n-days', type=int, default=70)
    parser.add_argument('--horizon', default='ret_10s', choices=list(HORIZONS.keys()))
    parser.add_argument('--min-train-days', type=int, default=5)
    args = parser.parse_args()

    try:
        results = run_experiment(args)
        print("\nExperiment complete.")
        return results
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        raise


if __name__ == '__main__':
    main()
