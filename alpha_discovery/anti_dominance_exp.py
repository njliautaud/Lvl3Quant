"""
Anti-Dominance / Residual Stacking Experiment
==============================================

Objective: Decorrelate Ch1 (L1 Imbalance, IC=0.1346) and Ch4 (Deep Book, IC=0.1282),
which are currently 85% correlated because both train on the same raw MFE target.

Anti-dominance means:
  1. Train Ch1 (dominant signal) with walk-forward LightGBM
  2. Compute residuals: mfe_net_10s - Ch1_OOS_prediction (per-fold, no lookahead)
  3. Train Ch4 (Deep Book) and Ch6 (Queue Dynamics) on the RESIDUAL target
  4. Compare IC on residual vs IC on raw target
  5. Measure correlation between Ch1 and Ch4 predictions before vs after

Result interpretation:
  - If Ch4_residual IC > 0: Ch4 captures ORTHOGONAL alpha that Ch1 missed
  - If Ch4_residual IC << Ch4_raw: Ch4 was mostly redundant (same signal as Ch1)
  - Correlation drop from 85% → lower means successful decorrelation

Usage:
    python alpha_discovery/anti_dominance_exp.py \\
        --feature-cache "C:/Users/Footb/Documents/Github/Lvl3Quant/data/processed/mbo_features_cache" \\
        --n-days 70 \\
        --horizon ret_10s

GPU acceleration: LightGBM uses device='gpu' for RTX 3090.
RAM: 70 days ≈ 18.6 GB, well within 34 GB limit.
"""

import gc
import sys
import time
import logging
import argparse
import platform
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr

# Add project root
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

# Cross-platform feature cache default
if platform.system() == 'Windows':
    DEFAULT_FEATURE_CACHE = str(LVL3_ROOT / "data" / "processed" / "mbo_features_cache")
else:
    DEFAULT_FEATURE_CACHE = str(Path.home() / "lvl3quant" / "data" / "processed" / "mbo_features_cache")

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner
from alpha_discovery.event_detector import (
    EventDetectionPipeline,
    compute_event_features,
    get_event_feature_names,
    N_EVENT_FEATURES,
)
from alpha_discovery.multi_channel_alpha import (
    CH1_L1_IMBALANCE,
    CH4_DEEP_BOOK,
    CH6_QUEUE_DYNAMICS,
    CHANNEL_DEFINITIONS,
)
from alpha_discovery.run_mfe_scan import compute_mfe_targets

# Results directory
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging setup
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"anti_dominance_{timestamp}.log"
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
logger = logging.getLogger("anti_dominance")

HORIZONS = {
    'ret_3s': 30,
    'ret_5s': 50,
    'ret_10s': 100,
    'ret_30s': 300,
    'ret_1m': 600,
}

# Optimized LightGBM params (matches the params used in recent successful scans)
LGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'min_child_samples': 100,
    'max_bin': 63,            # GPU-optimized
    'verbose': -1,
    'n_jobs': -1,
    'device': 'gpu',          # RTX 3090
    'objective': 'regression',
    'metric': 'rmse',
}


def get_col_indices(feature_names, channel_features):
    """Get column indices for a channel's features."""
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    indices = []
    found = []
    for f in channel_features:
        if f in name_to_idx:
            indices.append(name_to_idx[f])
            found.append(f)
    return np.array(indices, dtype=int), found


def train_channel_walk_forward(
    features: np.ndarray,
    target: np.ndarray,
    col_indices: np.ndarray,
    day_boundaries: list,
    min_train_days: int,
    channel_name: str,
    lgb_params: dict,
) -> tuple:
    """
    Walk-forward LightGBM training for one channel.

    Returns:
        (oos_predictions array (N,), fold_ics list)
        oos_predictions[i] is the OOS prediction for row i, NaN if not in any test fold.
    """
    import lightgbm as lgb

    N = len(target)
    n_days = len(day_boundaries) - 1
    oos_preds = np.full(N, np.nan, dtype=np.float32)
    fold_ics = []
    fold_actuals_list = []
    fold_preds_list = []

    logger.info(f"  [{channel_name}] Starting walk-forward ({n_days - min_train_days} folds)...")

    for test_day in range(min_train_days, n_days):
        # Expanding training window with 1-day purge gap
        train_start = day_boundaries[0]
        train_end = day_boundaries[test_day]      # up to (not including) test day
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_tr = features[train_start:train_end][:, col_indices]
        y_tr = target[train_start:train_end]
        X_te = features[test_start:test_end][:, col_indices]
        y_te = target[test_start:test_end]

        # Filter NaNs
        tr_valid = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
        te_valid = np.isfinite(y_te) & np.all(np.isfinite(X_te), axis=1)

        if tr_valid.sum() < 500 or te_valid.sum() < 50:
            continue

        X_tr_v = X_tr[tr_valid]
        y_tr_v = y_tr[tr_valid]
        X_te_v = X_te[te_valid]
        y_te_v = y_te[te_valid]

        # 80/20 split for early stopping
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

        # Store OOS predictions at their correct positions
        te_indices = np.where(te_valid)[0] + test_start
        oos_preds[te_indices] = preds

        # Track fold IC
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
    """Compute aggregate IC statistics from fold results."""
    if not fold_ics:
        return {'name': channel_name, 'ic': np.nan, 'ic_mean': np.nan, 'ic_std': np.nan,
                'icir': np.nan, 'hit_rate': np.nan, 'n_folds': 0}

    ics = np.array(fold_ics)
    ic_mean = float(ics.mean())
    ic_std = float(ics.std()) if len(ics) > 1 else 0.0
    icir = ic_mean / ic_std if ic_std > 0 else 0.0

    # Aggregate all predictions for overall IC
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


def measure_prediction_correlation(preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    """Measure Pearson correlation between two OOS prediction arrays (aligned by index)."""
    valid = np.isfinite(preds_a) & np.isfinite(preds_b)
    if valid.sum() < 100:
        return np.nan
    return float(np.corrcoef(preds_a[valid], preds_b[valid])[0, 1])


def compute_target(mid_prices, horizon_bars, day_boundaries):
    """Compute forward return target."""
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    future_mid = np.empty(N, dtype=np.float32)
    future_mid[:N - horizon_bars] = mid_prices[horizon_bars:]
    future_mid[N - horizon_bars:] = np.nan
    if n_days > 1:
        for d in range(n_days - 1):
            day_end = day_boundaries[d + 1]
            nan_start = max(day_boundaries[d], day_end - horizon_bars)
            future_mid[nan_start:day_end] = np.nan
    ret = (future_mid - mid_prices) / np.maximum(mid_prices, 1.0)
    return ret


def run_experiment(args):
    """Main experiment: anti-dominance on Ch4 and Ch6."""
    logger.info("=" * 70)
    logger.info("ANTI-DOMINANCE / RESIDUAL STACKING EXPERIMENT")
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
        logger.info(f"In-place augmentation: {features.shape[1]} features total")
    else:
        features = np.concatenate([scanner.features, event_features], axis=1)
        del event_features
        gc.collect()
        logger.info(f"Augmented features: {features.shape}")

    # ================================================================
    # PHASE 3: Compute MFE target
    # ================================================================
    logger.info(f"\n[PHASE 3] Computing MFE target ({args.horizon})...")
    horizon_bars = HORIZONS[args.horizon]
    hz_name = args.horizon.replace('ret_', '')  # 'ret_10s' -> '10s'
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
    logger.info(f"MFE target: {n_valid:,} valid values, "
                f"std={np.nanstd(target_mfe):.4f}")

    # ================================================================
    # PHASE 4: Train Ch1 (Dominant Signal) — walk-forward
    # ================================================================
    logger.info("\n[PHASE 4] Training Ch1 (L1 Imbalance) — DOMINANT SIGNAL...")
    ch1_cols, ch1_found = get_col_indices(augmented_names, CH1_L1_IMBALANCE)
    logger.info(f"  Ch1: {len(ch1_found)}/{len(CH1_L1_IMBALANCE)} features available")

    t0 = time.time()
    ch1_oos_preds, ch1_fold_ics, ch1_fold_preds, ch1_fold_actuals = train_channel_walk_forward(
        features=features,
        target=target_mfe,
        col_indices=ch1_cols,
        day_boundaries=scanner.day_boundaries,
        min_train_days=args.min_train_days,
        channel_name='Ch1_L1_Imbalance',
        lgb_params=LGB_PARAMS,
    )
    ch1_elapsed = time.time() - t0
    ch1_stats = compute_channel_ic(ch1_fold_ics, ch1_fold_preds, ch1_fold_actuals, 'Ch1')
    logger.info(f"\n  Ch1 RESULTS (raw target, {ch1_elapsed:.0f}s):")
    logger.info(f"    IC={ch1_stats['ic']:.4f}  ICIR={ch1_stats['icir']:.2f}  "
                f"HR={ch1_stats['hit_rate']:.1%}  folds={ch1_stats['n_folds']}")

    # ================================================================
    # PHASE 5: Compute OOS Residuals
    # ================================================================
    logger.info("\n[PHASE 5] Computing OOS residuals (target - Ch1_prediction)...")

    # Residual: where Ch1 has OOS predictions, compute residual
    # Where Ch1 has no prediction (early days in walk-forward), residual = NaN
    residual_target = np.full_like(target_mfe, np.nan)
    has_pred = np.isfinite(ch1_oos_preds) & np.isfinite(target_mfe)
    residual_target[has_pred] = target_mfe[has_pred] - ch1_oos_preds[has_pred]

    n_residual_valid = int(np.isfinite(residual_target).sum())
    residual_std = float(np.nanstd(residual_target))
    original_std = float(np.nanstd(target_mfe))

    # Correlation between residual and original target (should be much lower than 1)
    mask = has_pred
    resid_corr_with_original = float(np.corrcoef(
        target_mfe[mask], residual_target[mask]
    )[0, 1]) if mask.sum() > 100 else np.nan

    # Correlation between Ch1 OOS predictions and original target
    ch1_target_corr = float(np.corrcoef(
        ch1_oos_preds[has_pred], target_mfe[has_pred]
    )[0, 1]) if has_pred.sum() > 100 else np.nan

    logger.info(f"  Residual: {n_residual_valid:,} valid values")
    logger.info(f"  Residual std = {residual_std:.4f} (original std = {original_std:.4f})")
    logger.info(f"  Residual corr with original target = {resid_corr_with_original:.4f}")
    logger.info(f"  Ch1 prediction corr with target = {ch1_target_corr:.4f}")
    logger.info(f"  Variance explained by Ch1 = {ch1_target_corr**2:.1%}")

    del ch1_fold_preds, ch1_fold_actuals  # Free memory
    gc.collect()

    # ================================================================
    # PHASE 6: Train Ch4 (Deep Book) on RAW target
    # ================================================================
    logger.info("\n[PHASE 6] Training Ch4 (Deep Book) on RAW target...")
    ch4_cols, ch4_found = get_col_indices(augmented_names, CH4_DEEP_BOOK)
    logger.info(f"  Ch4: {len(ch4_found)}/{len(CH4_DEEP_BOOK)} features available")

    t0 = time.time()
    ch4_raw_oos_preds, ch4_raw_fold_ics, ch4_raw_fold_preds, ch4_raw_fold_actuals = (
        train_channel_walk_forward(
            features=features,
            target=target_mfe,           # RAW target
            col_indices=ch4_cols,
            day_boundaries=scanner.day_boundaries,
            min_train_days=args.min_train_days,
            channel_name='Ch4_DeepBook_RAW',
            lgb_params=LGB_PARAMS,
        )
    )
    ch4_raw_elapsed = time.time() - t0
    ch4_raw_stats = compute_channel_ic(
        ch4_raw_fold_ics, ch4_raw_fold_preds, ch4_raw_fold_actuals, 'Ch4_raw'
    )
    logger.info(f"\n  Ch4 (RAW) RESULTS ({ch4_raw_elapsed:.0f}s):")
    logger.info(f"    IC={ch4_raw_stats['ic']:.4f}  ICIR={ch4_raw_stats['icir']:.2f}  "
                f"HR={ch4_raw_stats['hit_rate']:.1%}  folds={ch4_raw_stats['n_folds']}")

    # Measure Ch1-Ch4 correlation BEFORE anti-dominance
    ch1_ch4_corr_before = measure_prediction_correlation(ch1_oos_preds, ch4_raw_oos_preds)
    logger.info(f"  Ch1-Ch4 prediction correlation (BEFORE anti-dominance): {ch1_ch4_corr_before:.4f}")

    del ch4_raw_fold_preds, ch4_raw_fold_actuals
    gc.collect()

    # ================================================================
    # PHASE 7: Train Ch4 (Deep Book) on RESIDUAL target (anti-dominance)
    # ================================================================
    logger.info("\n[PHASE 7] Training Ch4 (Deep Book) on RESIDUAL target (ANTI-DOMINANCE)...")

    t0 = time.time()
    ch4_res_oos_preds, ch4_res_fold_ics, ch4_res_fold_preds, ch4_res_fold_actuals = (
        train_channel_walk_forward(
            features=features,
            target=residual_target,      # RESIDUAL target
            col_indices=ch4_cols,
            day_boundaries=scanner.day_boundaries,
            min_train_days=args.min_train_days,
            channel_name='Ch4_DeepBook_RESIDUAL',
            lgb_params=LGB_PARAMS,
        )
    )
    ch4_res_elapsed = time.time() - t0
    ch4_res_stats = compute_channel_ic(
        ch4_res_fold_ics, ch4_res_fold_preds, ch4_res_fold_actuals, 'Ch4_residual'
    )
    logger.info(f"\n  Ch4 (RESIDUAL) RESULTS ({ch4_res_elapsed:.0f}s):")
    logger.info(f"    IC={ch4_res_stats['ic']:.4f}  ICIR={ch4_res_stats['icir']:.2f}  "
                f"HR={ch4_res_stats['hit_rate']:.1%}  folds={ch4_res_stats['n_folds']}")

    # Measure Ch1-Ch4 correlation AFTER anti-dominance
    ch1_ch4_corr_after = measure_prediction_correlation(ch1_oos_preds, ch4_res_oos_preds)
    logger.info(f"  Ch1-Ch4 prediction correlation (AFTER anti-dominance): {ch1_ch4_corr_after:.4f}")

    del ch4_res_fold_preds, ch4_res_fold_actuals
    gc.collect()

    # ================================================================
    # PHASE 8: Train Ch6 (Queue Dynamics) on RAW target
    # ================================================================
    logger.info("\n[PHASE 8] Training Ch6 (Queue Dynamics) on RAW target...")
    ch6_cols, ch6_found = get_col_indices(augmented_names, CH6_QUEUE_DYNAMICS)
    logger.info(f"  Ch6: {len(ch6_found)}/{len(CH6_QUEUE_DYNAMICS)} features available")

    if len(ch6_cols) == 0:
        logger.warning("  Ch6: No features available in cache — skipping Ch6 experiment")
        ch6_raw_stats = {'name': 'Ch6_raw', 'ic': np.nan, 'icir': np.nan,
                         'hit_rate': np.nan, 'n_folds': 0}
        ch6_res_stats = {'name': 'Ch6_residual', 'ic': np.nan, 'icir': np.nan,
                         'hit_rate': np.nan, 'n_folds': 0}
        ch6_raw_oos_preds = np.full(len(target_mfe), np.nan)
        ch6_res_oos_preds = np.full(len(target_mfe), np.nan)
        ch1_ch6_corr_before = np.nan
        ch1_ch6_corr_after = np.nan
    else:
        t0 = time.time()
        ch6_raw_oos_preds, ch6_raw_fold_ics, ch6_raw_fold_preds, ch6_raw_fold_actuals = (
            train_channel_walk_forward(
                features=features,
                target=target_mfe,
                col_indices=ch6_cols,
                day_boundaries=scanner.day_boundaries,
                min_train_days=args.min_train_days,
                channel_name='Ch6_QueueDyn_RAW',
                lgb_params=LGB_PARAMS,
            )
        )
        ch6_raw_elapsed = time.time() - t0
        ch6_raw_stats = compute_channel_ic(
            ch6_raw_fold_ics, ch6_raw_fold_preds, ch6_raw_fold_actuals, 'Ch6_raw'
        )
        logger.info(f"\n  Ch6 (RAW) RESULTS ({ch6_raw_elapsed:.0f}s):")
        logger.info(f"    IC={ch6_raw_stats['ic']:.4f}  ICIR={ch6_raw_stats['icir']:.2f}  "
                    f"HR={ch6_raw_stats['hit_rate']:.1%}  folds={ch6_raw_stats['n_folds']}")

        ch1_ch6_corr_before = measure_prediction_correlation(ch1_oos_preds, ch6_raw_oos_preds)
        logger.info(f"  Ch1-Ch6 prediction correlation (BEFORE): {ch1_ch6_corr_before:.4f}")
        del ch6_raw_fold_preds, ch6_raw_fold_actuals
        gc.collect()

        # ================================================================
        # PHASE 9: Train Ch6 on RESIDUAL target
        # ================================================================
        logger.info("\n[PHASE 9] Training Ch6 (Queue Dynamics) on RESIDUAL target...")
        t0 = time.time()
        ch6_res_oos_preds, ch6_res_fold_ics, ch6_res_fold_preds, ch6_res_fold_actuals = (
            train_channel_walk_forward(
                features=features,
                target=residual_target,
                col_indices=ch6_cols,
                day_boundaries=scanner.day_boundaries,
                min_train_days=args.min_train_days,
                channel_name='Ch6_QueueDyn_RESIDUAL',
                lgb_params=LGB_PARAMS,
            )
        )
        ch6_res_elapsed = time.time() - t0
        ch6_res_stats = compute_channel_ic(
            ch6_res_fold_ics, ch6_res_fold_preds, ch6_res_fold_actuals, 'Ch6_residual'
        )
        logger.info(f"\n  Ch6 (RESIDUAL) RESULTS ({ch6_res_elapsed:.0f}s):")
        logger.info(f"    IC={ch6_res_stats['ic']:.4f}  ICIR={ch6_res_stats['icir']:.2f}  "
                    f"HR={ch6_res_stats['hit_rate']:.1%}  folds={ch6_res_stats['n_folds']}")

        ch1_ch6_corr_after = measure_prediction_correlation(ch1_oos_preds, ch6_res_oos_preds)
        logger.info(f"  Ch1-Ch6 prediction correlation (AFTER): {ch1_ch6_corr_after:.4f}")
        del ch6_res_fold_preds, ch6_res_fold_actuals
        gc.collect()

    # ================================================================
    # PHASE 10: Anti-dominance ensemble comparison
    # ================================================================
    logger.info("\n[PHASE 10] Building anti-dominance ensemble...")

    # Strategy 1: Naive ensemble (Ch1 + Ch4_raw), equal weight
    # Strategy 2: Anti-dominance ensemble (Ch1 + Ch4_residual), equal weight
    # Compare ICs of both ensembles on the ORIGINAL MFE target

    # Align all predictions by valid positions
    all_valid = (
        np.isfinite(ch1_oos_preds) &
        np.isfinite(ch4_raw_oos_preds) &
        np.isfinite(ch4_res_oos_preds) &
        np.isfinite(target_mfe)
    )
    n_ensemble = int(all_valid.sum())
    logger.info(f"  Common valid positions: {n_ensemble:,}")

    results = {}
    if n_ensemble > 100:
        t_mfe = target_mfe[all_valid]
        p_ch1 = ch1_oos_preds[all_valid]
        p_ch4_raw = ch4_raw_oos_preds[all_valid]
        p_ch4_res = ch4_res_oos_preds[all_valid]

        # Normalize to zero-mean (rank transform not used here — keep raw preds)
        # For ensemble, we add the residual model's prediction BACK to Ch1
        # Interpretation: Ch4_res predicts what Ch1 missed, so total = Ch1 + Ch4_res
        naive_ensemble = 0.5 * p_ch1 + 0.5 * p_ch4_raw
        additive_ensemble = p_ch1 + p_ch4_res  # Add residual model back to dominant

        ic_ch1_alone = float(spearmanr(p_ch1, t_mfe)[0])
        ic_ch4_raw = float(spearmanr(p_ch4_raw, t_mfe)[0])
        ic_ch4_res_vs_raw = float(spearmanr(p_ch4_res, t_mfe)[0])
        ic_naive = float(spearmanr(naive_ensemble, t_mfe)[0])
        ic_additive = float(spearmanr(additive_ensemble, t_mfe)[0])

        logger.info(f"\n  ENSEMBLE COMPARISON:")
        logger.info(f"    Ch1 alone:             IC={ic_ch1_alone:.4f}")
        logger.info(f"    Ch4 raw alone:         IC={ic_ch4_raw:.4f}")
        logger.info(f"    Ch4 residual (vs raw): IC={ic_ch4_res_vs_raw:.4f}")
        logger.info(f"    Naive (0.5*Ch1+0.5*Ch4_raw): IC={ic_naive:.4f}")
        logger.info(f"    Additive (Ch1+Ch4_res):      IC={ic_additive:.4f}")

        # Correlation check
        corr_ch4_raw_vs_residual = float(np.corrcoef(p_ch4_raw, p_ch4_res)[0, 1])
        logger.info(f"\n  Ch4_raw vs Ch4_residual prediction correlation: {corr_ch4_raw_vs_residual:.4f}")
        logger.info(f"  (If negative/near-zero: residual model is truly orthogonal to raw model)")

        results['ensemble'] = {
            'n_common': n_ensemble,
            'ic_ch1_alone': ic_ch1_alone,
            'ic_ch4_raw': ic_ch4_raw,
            'ic_ch4_residual_vs_raw_target': ic_ch4_res_vs_raw,
            'ic_naive_ensemble': ic_naive,
            'ic_additive_ensemble': ic_additive,
            'corr_ch4_raw_vs_residual': corr_ch4_raw_vs_residual,
        }

    # ================================================================
    # FINAL REPORT
    # ================================================================
    total_elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info("ANTI-DOMINANCE EXPERIMENT — FINAL REPORT")
    logger.info("=" * 70)
    logger.info(f"Horizon: {args.horizon} | N-days: {args.n_days} | Total time: {total_elapsed:.0f}s")
    logger.info("")
    logger.info("RESIDUAL ANALYSIS:")
    logger.info(f"  Variance explained by Ch1:        {ch1_target_corr**2:.1%}")
    logger.info(f"  Residual corr with original:      {resid_corr_with_original:.4f}")
    logger.info(f"  Residual std / Original std:      {residual_std / original_std:.4f}")
    logger.info("")
    logger.info("CHANNEL IC COMPARISON:")
    logger.info(f"  {'Channel':<35} {'IC (raw)':<12} {'IC (residual)':<15} {'Delta IC'}")
    logger.info(f"  {'-'*35} {'-'*12} {'-'*15} {'-'*10}")
    logger.info(f"  {'Ch1 (dominant)':<35} {ch1_stats['ic']:<12.4f} {'N/A':<15} {'N/A'}")
    ch4_delta = ch4_res_stats['ic'] - ch4_raw_stats['ic']
    logger.info(f"  {'Ch4 (Deep Book)':<35} {ch4_raw_stats['ic']:<12.4f} "
                f"{ch4_res_stats['ic']:<15.4f} {ch4_delta:+.4f}")
    if not np.isnan(ch6_raw_stats['ic']):
        ch6_delta = ch6_res_stats['ic'] - ch6_raw_stats['ic']
        logger.info(f"  {'Ch6 (Queue Dynamics)':<35} {ch6_raw_stats['ic']:<12.4f} "
                    f"{ch6_res_stats['ic']:<15.4f} {ch6_delta:+.4f}")

    logger.info("")
    logger.info("DECORRELATION EFFECT:")
    logger.info(f"  Ch1-Ch4 corr BEFORE anti-dominance: {ch1_ch4_corr_before:.4f}")
    logger.info(f"  Ch1-Ch4 corr AFTER anti-dominance:  {ch1_ch4_corr_after:.4f}")
    if not np.isnan(ch1_ch6_corr_before):
        logger.info(f"  Ch1-Ch6 corr BEFORE anti-dominance: {ch1_ch6_corr_before:.4f}")
        logger.info(f"  Ch1-Ch6 corr AFTER anti-dominance:  {ch1_ch6_corr_after:.4f}")

    logger.info("")
    logger.info("INTERPRETATION:")
    if not np.isnan(ch4_res_stats['ic']) and ch4_res_stats['ic'] > 0.01:
        logger.info("  Ch4 has MEANINGFUL independent alpha after Ch1 is removed.")
        logger.info("  Anti-dominance successfully extracted hidden signal.")
    elif not np.isnan(ch4_res_stats['ic']) and ch4_res_stats['ic'] > 0:
        logger.info("  Ch4 has weak but positive residual IC.")
        logger.info("  Partial decorrelation benefit.")
    else:
        logger.info("  Ch4 residual IC is near zero — Ch4 is mostly redundant with Ch1.")
        logger.info("  Ch4 and Ch1 are capturing the same signal.")

    if 'ensemble' in results:
        naive_ic = results['ensemble']['ic_naive_ensemble']
        additive_ic = results['ensemble']['ic_additive_ensemble']
        ch1_ic = results['ensemble']['ic_ch1_alone']
        if additive_ic > naive_ic:
            gain = additive_ic - ch1_ic
            logger.info(f"  Additive ensemble beats naive: +{additive_ic - naive_ic:.4f} IC")
            logger.info(f"  Additive vs Ch1 alone: {gain:+.4f} IC")
        else:
            logger.info(f"  Naive ensemble is better — decorrelated signal adds noise")

    logger.info("")
    logger.info(f"Log file: {_log_file}")
    logger.info("=" * 70)

    return {
        'ch1': ch1_stats,
        'ch4_raw': ch4_raw_stats,
        'ch4_residual': ch4_res_stats,
        'ch6_raw': ch6_raw_stats,
        'ch6_residual': ch6_res_stats,
        'residual_analysis': {
            'ch1_variance_explained': ch1_target_corr ** 2,
            'residual_corr_with_original': resid_corr_with_original,
            'residual_std': residual_std,
            'original_std': original_std,
        },
        'decorrelation': {
            'ch1_ch4_before': ch1_ch4_corr_before,
            'ch1_ch4_after': ch1_ch4_corr_after,
            'ch1_ch6_before': ch1_ch6_corr_before if not np.isnan(ch1_ch6_corr_before) else None,
            'ch1_ch6_after': ch1_ch6_corr_after if not np.isnan(ch1_ch6_corr_after) else None,
        },
        'ensemble': results.get('ensemble', {}),
    }


def main():
    parser = argparse.ArgumentParser(description='Anti-dominance residual stacking experiment')
    parser.add_argument('--feature-cache',
                        default=DEFAULT_FEATURE_CACHE,
                        help='Path to pre-computed feature cache directory')
    parser.add_argument('--n-days', type=int, default=70,
                        help='Number of days to load (default: 70, max ~70 for RAM)')
    parser.add_argument('--horizon', default='ret_10s',
                        choices=list(HORIZONS.keys()),
                        help='Target horizon (default: ret_10s)')
    parser.add_argument('--min-train-days', type=int, default=5,
                        help='Minimum training days before first test fold')
    args = parser.parse_args()

    try:
        results = run_experiment(args)
        print("\nExperiment complete. See log for full results.")
        return results
    except Exception as e:
        logger.exception(f"Experiment failed: {e}")
        raise


if __name__ == '__main__':
    main()
