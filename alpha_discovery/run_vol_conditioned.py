"""
Vol-Conditioned Direction Strategy

Key insight from return scan: direction prediction IC is BETTER in low vol.
- ret_1s low_vol IC=0.088 vs high_vol IC=0.047
- ret_3s low_vol IC=0.094 vs high_vol IC=0.038 vs mid_vol IC=-0.068!

This script tests:
1. Vol-gated strategy: only trade when vol is below threshold
2. Vol-weighted sizing: reduce size in high vol
3. Combined vol + direction model
4. Ensemble of ret_1s + ret_3s

Usage:
    python alpha_discovery/run_vol_conditioned.py
"""

import sys
import gc
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, ttest_1samp
from typing import Dict, List

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache, compute_return_targets,
    EXCLUDE_FEATURES_DIRECTION,
)
from alpha_discovery.run_model_refinement import walk_forward_evaluate, simulate_pnl

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'vol_conditioned.log', mode='a'),
    ]
)
logger = logging.getLogger("vol_conditioned")

# ============================================================================
# VOL-CONDITIONED EVALUATION
# ============================================================================

def compute_vol_regimes(
    scanner: MBOAlphaScanner,
    vol_feature_name: str = 'rvol_20',
) -> np.ndarray:
    """
    Compute volatility regime labels (0=low, 1=mid, 2=high).
    Uses PAST vol only (no leakage).
    """
    # Find rvol_20 in features
    try:
        vol_idx = scanner.feature_names.index(vol_feature_name)
    except ValueError:
        # Try alternatives
        for alt in ['rvol_50', 'rvol_10', 'event_int_20']:
            try:
                vol_idx = scanner.feature_names.index(alt)
                vol_feature_name = alt
                break
            except ValueError:
                continue
        else:
            logger.warning("No vol feature found, using feature 0 as proxy")
            vol_idx = 0

    logger.info(f"Using {vol_feature_name} (idx={vol_idx}) for vol regime")
    vol_raw = scanner.features[:, vol_idx]

    # Compute expanding percentiles (causal: only past data)
    n = len(vol_raw)
    regime = np.full(n, 0, dtype=np.int32)  # default low

    # Per-day expanding quantile
    n_days = len(scanner.day_boundaries) - 1
    for d in range(n_days):
        day_start = scanner.day_boundaries[d]
        day_end = scanner.day_boundaries[d + 1]

        # Use all data up to (but not including) this day for quantiles
        if d > 0:
            hist_end = scanner.day_boundaries[d]
            hist_vol = vol_raw[:hist_end]
            q33 = np.nanpercentile(hist_vol, 33)
            q67 = np.nanpercentile(hist_vol, 67)
        else:
            # First day: use intra-day expanding
            q33 = np.nanpercentile(vol_raw[day_start:day_end], 33)
            q67 = np.nanpercentile(vol_raw[day_start:day_end], 67)

        day_vol = vol_raw[day_start:day_end]
        day_regime = np.where(day_vol > q67, 2,
                     np.where(day_vol > q33, 1, 0))
        regime[day_start:day_end] = day_regime

    counts = [int(np.sum(regime == i)) for i in range(3)]
    logger.info(f"Vol regimes: low={counts[0]:,} mid={counts[1]:,} high={counts[2]:,}")
    return regime


def walk_forward_vol_gated(
    scanner: MBOAlphaScanner,
    target: np.ndarray,
    target_name: str,
    vol_regime: np.ndarray,
    exclude_features: List[str],
    allowed_regimes: List[int] = [0],  # default: only trade in low vol
    min_train_days: int = 3,
) -> dict:
    """
    Walk-forward evaluation where we train on ALL data but only
    TEST (trade) in specified vol regimes.

    This simulates: "train the model on everything, but only trade
    when conditions are favorable."
    """
    import lightgbm as lgb

    n_days = len(scanner.day_boundaries) - 1
    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_use = scanner.features[:, keep_mask]
    names_use = [fn for fn in scanner.feature_names if fn not in exclude_features]

    lgbm_params = {
        'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'min_child_samples': 100, 'verbose': -1, 'n_jobs': -1,
    }

    all_preds = []
    all_actuals = []
    fold_ics = []

    for test_day in range(min_train_days, n_days):
        train_start = scanner.day_boundaries[0]
        train_end = scanner.day_boundaries[test_day]
        test_start = scanner.day_boundaries[test_day]
        test_end = scanner.day_boundaries[test_day + 1]

        X_train = features_use[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features_use[test_start:test_end]
        y_test = target[test_start:test_end]
        vol_test = vol_regime[test_start:test_end]

        # Train on ALL valid data
        train_valid = np.isfinite(y_train)
        # Test only in allowed regimes
        test_valid = np.isfinite(y_test) & np.isin(vol_test, allowed_regimes)

        if train_valid.sum() < 500 or test_valid.sum() < 50:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(
                X_tr[:split], y_tr[:split],
                eval_set=[(X_tr[split:], y_tr[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict(X_te)
        except Exception as e:
            continue

        all_preds.append(preds)
        all_actuals.append(y_te)

        if len(preds) > 10:
            try:
                ic_fold = spearmanr(preds, y_te)[0]
                if np.isfinite(ic_fold):
                    fold_ics.append(float(ic_fold))
            except:
                pass

        del model
        gc.collect()

    if not all_preds:
        return {'error': 'No predictions'}

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]

    ic = float(spearmanr(p, a)[0])

    if len(fold_ics) > 2:
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
        try:
            _, pvalue = ttest_1samp(fold_ics, 0)
            pvalue = float(pvalue)
        except:
            pvalue = 1.0
        fold_con = float(np.mean([1 for x in fold_ics if x > 0]))
    else:
        ic_mean = ic
        ic_std, icir, tstat, pvalue = 0.0, 0.0, 0.0, 1.0
        fold_con = float(np.mean([1 for x in fold_ics if x > 0])) if fold_ics else 0.0

    return {
        'ic': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'tstat': tstat,
        'pvalue': pvalue,
        'fold_ics': fold_ics,
        'fold_con': fold_con,
        'n_folds': len(fold_ics),
        'n_preds': len(p),
        'predictions': predictions,
        'actuals': actuals,
        'allowed_regimes': allowed_regimes,
        'pct_bars_used': float(len(p)) / float(np.isfinite(actuals).sum()) if np.isfinite(actuals).sum() > 0 else 0,
    }


def ensemble_predictions(
    scanner: MBOAlphaScanner,
    targets: Dict[str, np.ndarray],
    target_names: List[str],
    exclude_features: List[str],
) -> dict:
    """
    Ensemble multiple horizon predictions.
    Train separate models for each horizon, combine predictions
    via rank-averaging.
    """
    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in exclude_features]

    # Get predictions for each target
    per_target_preds = {}
    for tn in target_names:
        logger.info(f"  Getting predictions for {tn}...")
        res = walk_forward_evaluate(
            features=features_clean,
            target=targets[tn],
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            model_type='lgbm',
            hour_of_day=scanner.hour_of_day,
        )
        if 'error' not in res:
            per_target_preds[tn] = res
            logger.info(f"    IC={res['ic']:.4f}, n={res['n_preds']}")

    if len(per_target_preds) < 2:
        return {'error': 'Need at least 2 targets for ensemble'}

    # Align predictions: all must have same length
    # Since walk-forward produces predictions for same test periods,
    # they should align. Use the shortest.
    min_len = min(len(r['predictions']) for r in per_target_preds.values())

    # Rank-average
    from scipy.stats import rankdata

    ranked_preds = []
    for tn, res in per_target_preds.items():
        p = res['predictions'][:min_len]
        valid = np.isfinite(p)
        ranks = np.full(min_len, np.nan)
        ranks[valid] = rankdata(p[valid]) / valid.sum()
        ranked_preds.append(ranks)

    # Ensemble = mean of ranks
    ensemble = np.nanmean(ranked_preds, axis=0)

    # Evaluate against the primary target (first in list)
    primary_target = list(per_target_preds.keys())[0]
    actuals = per_target_preds[primary_target]['actuals'][:min_len]

    valid = np.isfinite(ensemble) & np.isfinite(actuals)
    p, a = ensemble[valid], actuals[valid]

    if len(p) < 50:
        return {'error': 'Too few valid ensemble predictions'}

    ic = float(spearmanr(p, a)[0])

    return {
        'ensemble_ic': ic,
        'n_preds': len(p),
        'per_target': {
            tn: {'ic': res['ic'], 'n_preds': res['n_preds']}
            for tn, res in per_target_preds.items()
        },
        'primary_target': primary_target,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()

    logger.info("=" * 75)
    logger.info("VOL-CONDITIONED DIRECTION STRATEGY")
    logger.info("=" * 75)

    # Load data
    scanner = MBOAlphaScanner()
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.error("No feature cache. Run return_multihorizon first.")
        sys.exit(1)

    logger.info(f"Data: {scanner.features.shape[0]:,} snapshots, "
                f"{len(scanner.day_boundaries)-1} days")

    # Compute targets
    targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        horizons_sec={'1s': 1, '3s': 3, '5s': 5},
        include_flow_target=False,
    )

    # Compute vol regimes
    vol_regime = compute_vol_regimes(scanner)

    all_results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }

    # ============================================================
    # Experiment 1: Vol-gated trading
    # ============================================================
    logger.info("\n" + "=" * 75)
    logger.info("EXPERIMENT 1: Vol-Gated Trading")
    logger.info("=" * 75)

    vol_gated = {}
    for target_name in ['ret_1s', 'ret_3s']:
        vol_gated[target_name] = {}
        for regime_set, label in [
            ([0], 'low_vol_only'),
            ([0, 1], 'low_mid_vol'),
            ([0, 1, 2], 'all_regimes'),
            ([2], 'high_vol_only'),
        ]:
            logger.info(f"\n  {target_name} / {label}:")
            res = walk_forward_vol_gated(
                scanner, targets[target_name], target_name,
                vol_regime, EXCLUDE_FEATURES_DIRECTION,
                allowed_regimes=regime_set,
            )

            if 'error' not in res:
                logger.info(
                    f"    IC={res['ic']:.4f} t={res['tstat']:.2f} "
                    f"ICIR={res['icir']:.2f} FoldC={res['fold_con']:.0%} "
                    f"bars_used={res['pct_bars_used']:.0%} "
                    f"n={res['n_preds']:,}"
                )
                # PnL simulation for this regime
                pnl = simulate_pnl(
                    res['predictions'], res['actuals'],
                    target_horizon_sec=1.0 if '1s' in target_name else 3.0,
                    threshold_quantile=0.7,
                )
                if 'error' not in pnl:
                    logger.info(
                        f"    PnL: net=${pnl['daily_net_pnl']:+.2f}/day "
                        f"gross=${pnl['daily_gross_pnl']:+.2f} "
                        f"cost=${pnl['daily_cost']:.2f} "
                        f"Sharpe={pnl['daily_sharpe']:.2f}"
                    )

                vol_gated[target_name][label] = {
                    k: v for k, v in res.items()
                    if k not in ('predictions', 'actuals')
                }
                vol_gated[target_name][label]['pnl'] = {
                    k: v for k, v in pnl.items() if k != 'daily_chunks'
                } if 'error' not in pnl else pnl
            else:
                logger.warning(f"    {res['error']}")
                vol_gated[target_name][label] = res

    all_results['vol_gated'] = vol_gated

    # ============================================================
    # Experiment 2: Ensemble of horizons
    # ============================================================
    logger.info("\n" + "=" * 75)
    logger.info("EXPERIMENT 2: Ensemble of ret_1s + ret_3s")
    logger.info("=" * 75)

    ens_result = ensemble_predictions(
        scanner, targets, ['ret_1s', 'ret_3s'],
        EXCLUDE_FEATURES_DIRECTION,
    )
    if 'error' not in ens_result:
        logger.info(f"  Ensemble IC={ens_result['ensemble_ic']:.4f}")
        for tn, info in ens_result['per_target'].items():
            logger.info(f"    {tn} alone: IC={info['ic']:.4f}")
    else:
        logger.warning(f"  Ensemble failed: {ens_result['error']}")

    all_results['ensemble'] = ens_result

    # ============================================================
    # Experiment 3: Train ONLY on low-vol bars
    # ============================================================
    logger.info("\n" + "=" * 75)
    logger.info("EXPERIMENT 3: Train and Test ONLY in low vol")
    logger.info("=" * 75)

    low_vol_only = {}
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    for target_name in ['ret_1s', 'ret_3s']:
        logger.info(f"\n  {target_name}:")

        # Mask: only low vol bars
        target_masked = targets[target_name].copy()
        target_masked[vol_regime != 0] = np.nan  # NaN out non-low-vol bars

        res = walk_forward_evaluate(
            features=features_clean,
            target=target_masked,
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            model_type='lgbm',
            hour_of_day=scanner.hour_of_day,
        )

        if 'error' not in res:
            logger.info(
                f"    IC={res['ic']:.4f} t={res['tstat']:.2f} "
                f"ICIR={res['icir']:.2f} n={res['n_preds']:,}"
            )
            low_vol_only[target_name] = {
                k: v for k, v in res.items()
                if k not in ('predictions', 'actuals')
            }
        else:
            logger.warning(f"    {res['error']}")
            low_vol_only[target_name] = res

    all_results['low_vol_train_test'] = low_vol_only

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t0
    all_results['elapsed_sec'] = elapsed

    logger.info("\n" + "=" * 75)
    logger.info("SUMMARY")
    logger.info("=" * 75)

    logger.info("\n--- Vol-Gated Results ---")
    for tgt in ['ret_1s', 'ret_3s']:
        if tgt in vol_gated:
            logger.info(f"\n  {tgt}:")
            for label, res in vol_gated[tgt].items():
                if isinstance(res, dict) and 'error' not in res:
                    pnl_str = ""
                    if 'pnl' in res and isinstance(res['pnl'], dict) and 'error' not in res['pnl']:
                        pnl_str = f" PnL=${res['pnl']['daily_net_pnl']:+.2f}/day"
                    logger.info(
                        f"    {label:>16s}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                        f"bars={res['pct_bars_used']:.0%}{pnl_str}"
                    )

    if 'error' not in ens_result:
        logger.info(f"\n--- Ensemble ---")
        logger.info(f"  ret_1s+ret_3s ensemble IC={ens_result['ensemble_ic']:.4f}")

    logger.info(f"\n--- Low-Vol Only (train + test) ---")
    for tgt, res in low_vol_only.items():
        if 'error' not in res:
            logger.info(f"  {tgt}: IC={res['ic']:.4f} t={res['tstat']:.2f}")

    # Save
    out_path = RESULTS_DIR / f"vol_conditioned_{all_results['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {out_path}")
    logger.info(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
