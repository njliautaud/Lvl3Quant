"""
LightGBM Residual Volatility Scan

Question: Can LightGBM predict the RESIDUAL volatility beyond naive persistence?

Naive baseline: past 5s realized vol is the best predictor of future 5s vol (IC=0.77).
Residual:       future_vol - naive_pred = the part persistence MISSES.

This script:
1. Loads cached features (from tonight's scan)
2. Computes past_vol (rolling std of log returns, window=50, BACKWARD looking)
3. Computes future_vol (5s volatility target from compute_targets)
4. Computes residual = future_vol - past_vol
5. Excludes leaky features (same as clean scan)
6. Runs walk-forward LightGBM on the residual
7. Evaluates: naive IC, LightGBM residual IC, combined IC
8. Also tests: direction prediction conditioned on high-vol regime
9. Reports all results to Discord

Usage:
    python alpha_discovery/run_residual_scan.py
"""

import sys
import gc
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr, ttest_1samp
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import TOTAL_FEATURES, get_feature_names

# ============================================================================
# Setup logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'residual_scan.log', mode='a'),
    ]
)
logger = logging.getLogger("run_residual_scan")

# Feature cache path (built during tonight's scan)
FEATURE_CACHE = RESULTS_DIR / "feature_cache"

# Same leaky features to exclude as clean scan
EXCLUDE_FEATURES = [
    'mid', 'best_bid', 'best_ask', 'microprice',
    'hour_norm', 'minute_norm', 'time_since_rth', 'time_to_close',
]

# 5s horizon = 50 bars at 100ms
HORIZON_STEPS = 50        # 5s forward for vol target
PAST_VOL_WINDOW = 50      # 5s backward for naive baseline (same window)
SAMPLE_INTERVAL_MS = 100  # 100ms bars


# ============================================================================
# Discord helper
# ============================================================================

def send_discord(msg: str):
    """Send progress update to Discord."""
    try:
        import subprocess
        # Use Node.js MCP discord tool by writing a small bridge script
        # Fall back to just logging if discord not reachable
        logger.info(f"[DISCORD] {msg}")
    except Exception:
        logger.info(f"[DISCORD FAIL] {msg}")


# ============================================================================
# Feature cache loader (same logic as run_clean_scan.py)
# ============================================================================

def _feature_names_hash() -> str:
    import hashlib
    names_str = ",".join(get_feature_names())
    return hashlib.md5(names_str.encode()).hexdigest()[:12]


def load_feature_cache(scanner: MBOAlphaScanner) -> dict:
    path = FEATURE_CACHE / "features_alldays.npz"
    if not path.exists():
        return None

    logger.info(f"Loading cached features from {path}...")
    t0 = time.time()
    data = np.load(str(path), allow_pickle=True)
    cached_features = data['features']

    if cached_features.shape[1] != TOTAL_FEATURES:
        logger.warning(f"Feature cache STALE: {cached_features.shape[1]} vs {TOTAL_FEATURES}")
        data.close()
        return None

    stats_path = str(path).replace('.npz', '_stats.json')
    stats = {}
    if Path(stats_path).exists():
        with open(stats_path) as f:
            stats = json.load(f)
        cached_hash = stats.get('feature_names_hash', '')
        current_hash = _feature_names_hash()
        if cached_hash and cached_hash != current_hash:
            logger.warning(f"Feature cache STALE: hash mismatch {cached_hash} vs {current_hash}")
            data.close()
            return None

    scanner.features = cached_features
    scanner.mid_prices = data['mid_prices']
    scanner.hour_of_day = data['hour_of_day']
    scanner.time_since_rth = data['time_since_rth']
    scanner.day_boundaries = list(data['day_boundaries'])
    scanner.feature_names = get_feature_names()

    logger.info(f"Cache loaded in {time.time() - t0:.1f}s: {scanner.features.shape}")
    return stats if stats else {'n_snapshots': len(data['mid_prices'])}


# ============================================================================
# Past volatility computation (BACKWARD-LOOKING ONLY)
# ============================================================================

def compute_past_vol(mid_prices: np.ndarray, day_boundaries: list,
                     window: int = 50) -> np.ndarray:
    """
    Compute backward-looking realized volatility over `window` bars.
    Resets at each day boundary to avoid overnight leakage.

    Returns array of same length as mid_prices with NaN for the first
    `window` bars of each day (insufficient history).
    """
    N = len(mid_prices)
    past_vol = np.full(N, np.nan, dtype=np.float64)
    log_mid = np.log(np.maximum(mid_prices, 1.0))
    log_ret = np.diff(log_mid, prepend=log_mid[0])  # ret[i] = log_mid[i] - log_mid[i-1]

    n_days = len(day_boundaries) - 1
    for d in range(n_days):
        start = day_boundaries[d]
        end = day_boundaries[d + 1]
        day_ret = log_ret[start:end].copy()
        day_len = end - start

        # Vectorized rolling std using cumsum trick
        lr64 = day_ret.astype(np.float64)
        cs = np.cumsum(lr64)
        cs2 = np.cumsum(lr64 ** 2)
        # Prepend 0 for windowing
        cs_pad = np.concatenate([[0.0], cs])
        cs2_pad = np.concatenate([[0.0], cs2])

        for i in range(day_len):
            if i < window:
                # Not enough history
                past_vol[start + i] = np.nan
            else:
                # Window: bars [i-window+1 .. i]
                w_start = i - window + 1
                s1 = cs_pad[i + 1] - cs_pad[w_start]
                s2 = cs2_pad[i + 1] - cs2_pad[w_start]
                var = s2 / window - (s1 / window) ** 2
                past_vol[start + i] = np.sqrt(max(var, 0.0))

    return past_vol.astype(np.float32)


def compute_past_vol_fast(mid_prices: np.ndarray, day_boundaries: list,
                           window: int = 50) -> np.ndarray:
    """
    Faster vectorized version using cumsum within each day.
    """
    N = len(mid_prices)
    past_vol = np.full(N, np.nan, dtype=np.float64)
    log_mid = np.log(np.maximum(mid_prices.astype(np.float64), 1.0))
    log_ret = np.diff(log_mid, prepend=log_mid[0])

    n_days = len(day_boundaries) - 1
    for d in range(n_days):
        start = day_boundaries[d]
        end = day_boundaries[d + 1]
        day_len = end - start

        if day_len < window:
            continue

        lr = log_ret[start:end]
        cs_pad = np.concatenate([[0.0], np.cumsum(lr)])
        cs2_pad = np.concatenate([[0.0], np.cumsum(lr ** 2)])

        # For each bar i >= window-1, rolling window ends at i (inclusive), starts at i-window+1
        n_valid = day_len - window + 1
        idx = np.arange(window, day_len + 1)        # end indices (exclusive in cs_pad sense)
        idx_start = idx - window                     # start indices

        s1 = cs_pad[idx] - cs_pad[idx_start]
        s2 = cs2_pad[idx] - cs2_pad[idx_start]
        var = s2 / window - (s1 / window) ** 2
        var = np.maximum(var, 0.0)

        past_vol[start + window - 1:end] = np.sqrt(var)

    return past_vol.astype(np.float32)


# ============================================================================
# Walk-forward evaluation for residual
# ============================================================================

def walk_forward_residual(
    scanner: MBOAlphaScanner,
    future_vol: np.ndarray,
    past_vol: np.ndarray,
    exclude_features: list,
    min_train_days: int = 3,
) -> dict:
    """
    Walk-forward evaluation of LightGBM on vol residuals.

    For each test fold:
      1. Compute naive baseline IC (past_vol vs future_vol)
      2. Train LightGBM on residual (future_vol - past_vol) using train data
      3. Predict residual on test data
      4. Combine: combined_pred = past_vol + alpha * lgbm_residual_pred
         where alpha is calibrated on (last 20% of training data)
      5. Evaluate naive IC, lgbm residual IC, combined IC

    Returns comprehensive dict with per-fold and aggregate stats.
    """
    N = len(future_vol)
    n_days = len(scanner.day_boundaries) - 1

    # Build feature mask
    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_use = scanner.features[:, keep_mask]
    feature_names_use = [fn for fn in scanner.feature_names if fn not in exclude_features]
    n_features = len(feature_names_use)
    logger.info(f"Using {n_features} features (excluded {sum(~keep_mask)})")

    # Residual target
    residual = future_vol - past_vol  # what persistence misses
    # IMPORTANT: residual has NaN wherever future_vol or past_vol is NaN

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 500,
        'max_depth': 5,           # Shallower than before to avoid overfitting
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 200, # Larger for better generalization
        'verbose': -1,
        'n_jobs': -1,
    }

    fold_results = []
    all_naive_preds = []
    all_lgbm_residuals = []
    all_combined_preds = []
    all_future_vols = []
    feature_importance = np.zeros(n_features)

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1  # 1-day purge gap
        train_start = scanner.day_boundaries[0]
        train_end = scanner.day_boundaries[train_end_day + 1]

        test_start = scanner.day_boundaries[test_day]
        test_end = scanner.day_boundaries[test_day + 1]

        # Training data
        X_train = features_use[train_start:train_end]
        resid_train = residual[train_start:train_end]
        past_vol_train = past_vol[train_start:train_end]
        future_vol_train = future_vol[train_start:train_end]

        # Test data
        X_test = features_use[test_start:test_end]
        resid_test = residual[test_start:test_end]
        past_vol_test = past_vol[test_start:test_end]
        future_vol_test = future_vol[test_start:test_end]

        # Valid masks: need both past_vol AND future_vol to be finite
        train_valid = np.isfinite(resid_train) & np.isfinite(past_vol_train)
        test_valid = np.isfinite(resid_test) & np.isfinite(past_vol_test) & np.isfinite(future_vol_test)

        if train_valid.sum() < 500 or test_valid.sum() < 100:
            logger.warning(f"Day {test_day}: insufficient data (train={train_valid.sum()}, test={test_valid.sum()})")
            continue

        X_tr = X_train[train_valid]
        y_tr = resid_train[train_valid]
        X_te = X_test[test_valid]
        y_te_resid = resid_test[test_valid]
        past_vol_te = past_vol_test[test_valid]
        future_vol_te = future_vol_test[test_valid]

        # Calibration: use last 20% of training data to find alpha
        # alpha * lgbm_residual_pred is the optimal blend
        calib_split = int(len(X_tr) * 0.8)
        X_calib = X_tr[calib_split:]
        y_calib_resid = y_tr[calib_split:]
        past_vol_calib_idx = np.where(train_valid)[0][calib_split:]
        past_vol_calib = past_vol_train[train_valid][calib_split:]
        future_vol_calib = future_vol_train[train_valid][calib_split:]

        # Train model
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[:calib_split], y_tr[:calib_split],
                eval_set=[(X_calib, y_calib_resid)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception as e:
            logger.warning(f"  Training failed day {test_day}: {e}")
            continue

        # Predict residual on calibration set to find alpha
        calib_preds = model.predict(X_calib)
        # combined_calib = past_vol_calib + alpha * calib_preds
        # Find alpha that minimizes MSE on calibration vs future_vol_calib
        # d/d_alpha MSE = 0 => alpha = cov(calib_preds, y_calib_resid) / var(calib_preds)
        cov_cp = np.cov(calib_preds, y_calib_resid)
        if cov_cp[0, 0] > 1e-15:
            alpha = float(cov_cp[0, 1] / cov_cp[0, 0])
            # Clip alpha to reasonable range
            alpha = float(np.clip(alpha, 0.0, 2.0))
        else:
            alpha = 1.0

        logger.info(f"  Day {test_day}: calibrated alpha={alpha:.3f} "
                    f"(train={train_valid.sum():,}, test={test_valid.sum():,})")

        # Predict residual on test set
        lgbm_resid_preds = model.predict(X_te)
        combined_preds = past_vol_te + alpha * lgbm_resid_preds

        # Compute ICs for this fold
        try:
            ic_naive = float(spearmanr(past_vol_te, future_vol_te)[0])
            ic_lgbm_resid = float(spearmanr(lgbm_resid_preds, y_te_resid)[0])
            ic_lgbm_abs = float(spearmanr(lgbm_resid_preds, future_vol_te)[0])
            ic_combined = float(spearmanr(combined_preds, future_vol_te)[0])
        except Exception as e:
            logger.warning(f"  IC computation failed day {test_day}: {e}")
            continue

        fold_results.append({
            'day': test_day,
            'n_test': int(test_valid.sum()),
            'n_train': int(train_valid.sum()),
            'alpha': alpha,
            'ic_naive': ic_naive,
            'ic_lgbm_residual': ic_lgbm_resid,
            'ic_lgbm_absolute': ic_lgbm_abs,
            'ic_combined': ic_combined,
            'best_iteration': model.best_iteration_ if hasattr(model, 'best_iteration_') else None,
            'residual_std': float(np.std(y_te_resid)),
            'naive_pred_std': float(np.std(past_vol_te)),
            'lgbm_pred_std': float(np.std(lgbm_resid_preds)),
        })

        all_naive_preds.append(past_vol_te)
        all_lgbm_residuals.append(lgbm_resid_preds)
        all_combined_preds.append(combined_preds)
        all_future_vols.append(future_vol_te)

        if hasattr(model, 'feature_importances_'):
            feature_importance += model.feature_importances_

        del model
        gc.collect()

    if not fold_results:
        return {'error': 'No valid folds', 'fold_results': []}

    # Aggregate across all folds
    naive_all = np.concatenate(all_naive_preds)
    lgbm_resid_all = np.concatenate(all_lgbm_residuals)
    combined_all = np.concatenate(all_combined_preds)
    future_all = np.concatenate(all_future_vols)

    valid = np.isfinite(naive_all) & np.isfinite(combined_all) & np.isfinite(future_all)
    n_valid_all = naive_all[valid], lgbm_resid_all[valid], combined_all[valid], future_all[valid]
    naive_v, lgbm_v, combined_v, future_v = n_valid_all

    ic_naive_overall = float(spearmanr(naive_v, future_v)[0])
    ic_lgbm_resid_overall = float(spearmanr(lgbm_v, future_v - naive_v)[0])
    ic_lgbm_abs_overall = float(spearmanr(lgbm_v, future_v)[0])
    ic_combined_overall = float(spearmanr(combined_v, future_v)[0])

    # ICIR for combined
    combined_ics = [f['ic_combined'] for f in fold_results if np.isfinite(f['ic_combined'])]
    naive_ics = [f['ic_naive'] for f in fold_results if np.isfinite(f['ic_naive'])]
    lgbm_resid_ics = [f['ic_lgbm_residual'] for f in fold_results if np.isfinite(f['ic_lgbm_residual'])]

    def icir_stats(ics):
        if len(ics) < 2:
            return 0.0, 0.0, 1.0
        m, s = np.mean(ics), np.std(ics)
        icir = m / s if s > 0 else 0.0
        tstat = m / s * np.sqrt(len(ics)) if s > 0 else 0.0
        try:
            _, pval = ttest_1samp(ics, 0)
        except Exception:
            pval = 1.0
        return float(icir), float(tstat), float(pval)

    icir_naive, tstat_naive, pval_naive = icir_stats(naive_ics)
    icir_lgbm, tstat_lgbm, pval_lgbm = icir_stats(lgbm_resid_ics)
    icir_combined, tstat_combined, pval_combined = icir_stats(combined_ics)

    # Top features
    top_feat_idx = np.argsort(feature_importance)[::-1][:20]
    top_features = [(feature_names_use[i], float(feature_importance[i]))
                    for i in top_feat_idx if feature_importance[i] > 0]

    return {
        'n_folds': len(fold_results),
        'n_predictions': int(valid.sum()),
        'fold_results': fold_results,
        # Overall ICs
        'ic_naive': ic_naive_overall,
        'ic_lgbm_residual_vs_residual': ic_lgbm_resid_overall,
        'ic_lgbm_absolute': ic_lgbm_abs_overall,
        'ic_combined': ic_combined_overall,
        # ICIR stats
        'icir_naive': icir_naive,
        'tstat_naive': tstat_naive,
        'pval_naive': pval_naive,
        'icir_lgbm_resid': icir_lgbm,
        'tstat_lgbm_resid': tstat_lgbm,
        'pval_lgbm_resid': pval_lgbm,
        'icir_combined': icir_combined,
        'tstat_combined': tstat_combined,
        'pval_combined': pval_combined,
        # Feature importance
        'top_features': top_features,
    }


# ============================================================================
# Direction prediction in HIGH-VOL regime
# ============================================================================

def walk_forward_direction_highvol(
    scanner: MBOAlphaScanner,
    future_vol: np.ndarray,
    past_vol: np.ndarray,
    future_return: np.ndarray,
    exclude_features: list,
    min_train_days: int = 3,
) -> dict:
    """
    Test whether order flow features predict DIRECTION when we KNOW vol is about to be high.

    High-vol regime = bars where naive vol prediction (past_vol) is above median.
    Within those bars, run walk-forward LightGBM on the RETURN target.

    This tests the two-stage hypothesis:
      1. Use past_vol to identify high-vol bars (IC=0.77, we know this works)
      2. In those bars, can LightGBM predict direction (positive/negative return)?
    """
    N = len(future_return)
    n_days = len(scanner.day_boundaries) - 1

    # Build feature mask
    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_use = scanner.features[:, keep_mask]
    feature_names_use = [fn for fn in scanner.feature_names if fn not in exclude_features]

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 200,
        'verbose': -1,
        'n_jobs': -1,
    }

    fold_results = []
    all_preds_highvol = []
    all_actuals_highvol = []
    all_preds_lowvol = []
    all_actuals_lowvol = []
    feature_importance = np.zeros(len(feature_names_use))

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1
        train_start = scanner.day_boundaries[0]
        train_end = scanner.day_boundaries[train_end_day + 1]
        test_start = scanner.day_boundaries[test_day]
        test_end = scanner.day_boundaries[test_day + 1]

        # Training data
        X_train = features_use[train_start:train_end]
        ret_train = future_return[train_start:train_end]
        past_vol_train = past_vol[train_start:train_end]

        # Test data
        X_test = features_use[test_start:test_end]
        ret_test = future_return[test_start:test_end]
        past_vol_test = past_vol[test_start:test_end]

        # Compute vol median on TRAINING data only (no lookahead)
        train_valid_all = np.isfinite(ret_train) & np.isfinite(past_vol_train)
        if train_valid_all.sum() < 500:
            continue
        vol_median = float(np.nanmedian(past_vol_train[train_valid_all]))

        # Train on ALL bars (not just high-vol) to avoid selection bias in training
        X_tr = X_train[train_valid_all]
        y_tr = ret_train[train_valid_all]
        if len(X_tr) < 500:
            continue

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[:split], y_tr[:split],
                eval_set=[(X_tr[split:], y_tr[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception as e:
            logger.warning(f"  High-vol direction day {test_day} failed: {e}")
            continue

        # Predict on test set
        test_valid_all = np.isfinite(ret_test) & np.isfinite(past_vol_test)
        if test_valid_all.sum() < 50:
            del model; continue

        X_te = X_test[test_valid_all]
        ret_te = ret_test[test_valid_all]
        pv_te = past_vol_test[test_valid_all]
        preds = model.predict(X_te)

        # Regime split on TEST set (using training median — no lookahead)
        high_vol_mask = pv_te > vol_median
        low_vol_mask = ~high_vol_mask

        def safe_ic(p, a):
            if len(p) < 10:
                return np.nan
            try:
                return float(spearmanr(p, a)[0])
            except Exception:
                return np.nan

        ic_all = safe_ic(preds, ret_te)
        ic_high = safe_ic(preds[high_vol_mask], ret_te[high_vol_mask])
        ic_low = safe_ic(preds[low_vol_mask], ret_te[low_vol_mask])

        fold_results.append({
            'day': test_day,
            'vol_median': vol_median,
            'n_total': int(test_valid_all.sum()),
            'n_highvol': int(high_vol_mask.sum()),
            'n_lowvol': int(low_vol_mask.sum()),
            'ic_all': ic_all,
            'ic_highvol': ic_high,
            'ic_lowvol': ic_low,
        })

        if high_vol_mask.sum() > 10:
            all_preds_highvol.append(preds[high_vol_mask])
            all_actuals_highvol.append(ret_te[high_vol_mask])
        if low_vol_mask.sum() > 10:
            all_preds_lowvol.append(preds[low_vol_mask])
            all_actuals_lowvol.append(ret_te[low_vol_mask])

        if hasattr(model, 'feature_importances_'):
            feature_importance += model.feature_importances_

        del model
        gc.collect()

    if not fold_results:
        return {'error': 'No valid folds', 'fold_results': []}

    # Aggregate
    def agg_ic(preds_list, actuals_list):
        if not preds_list:
            return np.nan
        p = np.concatenate(preds_list)
        a = np.concatenate(actuals_list)
        valid = np.isfinite(p) & np.isfinite(a)
        if valid.sum() < 50:
            return np.nan
        try:
            return float(spearmanr(p[valid], a[valid])[0])
        except Exception:
            return np.nan

    ic_highvol_overall = agg_ic(all_preds_highvol, all_actuals_highvol)
    ic_lowvol_overall = agg_ic(all_preds_lowvol, all_actuals_lowvol)

    # ICIR for high-vol direction
    hv_ics = [f['ic_highvol'] for f in fold_results if np.isfinite(f.get('ic_highvol', np.nan))]
    icir_hv, tstat_hv, pval_hv = 0.0, 0.0, 1.0
    if len(hv_ics) >= 2:
        m, s = np.mean(hv_ics), np.std(hv_ics)
        if s > 0:
            icir_hv = m / s
            tstat_hv = m / s * np.sqrt(len(hv_ics))
            try:
                _, pval_hv = ttest_1samp(hv_ics, 0)
            except Exception:
                pass

    top_feat_idx = np.argsort(feature_importance)[::-1][:15]
    top_features = [(feature_names_use[i], float(feature_importance[i]))
                    for i in top_feat_idx if feature_importance[i] > 0]

    return {
        'n_folds': len(fold_results),
        'fold_results': fold_results,
        'ic_direction_all_bars': float(agg_ic(
            all_preds_highvol + all_preds_lowvol,
            all_actuals_highvol + all_actuals_lowvol
        )),
        'ic_direction_highvol': ic_highvol_overall,
        'ic_direction_lowvol': ic_lowvol_overall,
        'icir_highvol': float(icir_hv),
        'tstat_highvol': float(tstat_hv),
        'pval_highvol': float(pval_hv),
        'top_features': top_features,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("LightGBM RESIDUAL VOLATILITY SCAN")
    logger.info("Question: Can we beat naive vol persistence?")
    logger.info(f"Excluding: {EXCLUDE_FEATURES}")
    logger.info("=" * 70)

    # Send initial Discord update
    try:
        from lib.discord import send_message as discord_send
        discord_send("**[Residual Scan] Starting LightGBM residual volatility analysis**\nLoading feature cache...")
    except Exception:
        logger.info("[Discord] Starting LightGBM residual volatility analysis")

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.info("No feature cache, loading from snapshot cache...")
        try:
            from lib.discord import send_message as discord_send
            discord_send("[Residual Scan] No feature cache found, loading from snapshot cache (this takes ~5 min)...")
        except Exception:
            pass
        stats = scanner.load_from_cache()

    n_days = len(scanner.day_boundaries) - 1
    N = len(scanner.mid_prices)
    logger.info(f"Loaded: {N:,} snapshots, {n_days} days")

    # ----------------------------------------------------------------
    # Compute targets
    # ----------------------------------------------------------------
    logger.info("Computing targets...")
    all_targets = scanner.compute_targets()
    future_vol_5s = all_targets['5s']['volatility']
    future_return_5s = all_targets['5s']['return']

    logger.info(f"Future vol: {np.isfinite(future_vol_5s).sum():,} valid bars "
                f"(mean={np.nanmean(future_vol_5s):.6f}, std={np.nanstd(future_vol_5s):.6f})")

    # ----------------------------------------------------------------
    # Compute backward-looking past vol (naive predictor)
    # ----------------------------------------------------------------
    logger.info(f"Computing backward past_vol (window={PAST_VOL_WINDOW} bars)...")
    t0 = time.time()
    past_vol = compute_past_vol_fast(scanner.mid_prices, scanner.day_boundaries, window=PAST_VOL_WINDOW)
    logger.info(f"Past vol computed in {time.time() - t0:.1f}s: "
                f"{np.isfinite(past_vol).sum():,} valid bars "
                f"(mean={np.nanmean(past_vol):.6f}, std={np.nanstd(past_vol):.6f})")

    # Sanity check: naive persistence IC should be ~0.77
    valid_both = np.isfinite(past_vol) & np.isfinite(future_vol_5s)
    naive_ic_check = float(spearmanr(past_vol[valid_both], future_vol_5s[valid_both])[0])
    logger.info(f"Naive persistence IC (FULL DATA, NOT WALK-FORWARD): {naive_ic_check:.4f}")
    logger.info(f"  (This is an IC=0.77 sanity check — expect similar value)")

    # Residual
    residual = future_vol_5s - past_vol
    valid_resid = np.isfinite(residual)
    logger.info(f"Residual: {valid_resid.sum():,} valid bars "
                f"(mean={np.nanmean(residual):.6f}, std={np.nanstd(residual):.6f})")

    try:
        from lib.discord import send_message as discord_send
        discord_send(
            f"**[Residual Scan] Data loaded. Starting walk-forward evaluation...**\n"
            f"• {N:,} snapshots, {n_days} days\n"
            f"• Naive persistence IC (full data): {naive_ic_check:.4f}\n"
            f"• Running walk-forward LightGBM on vol residual..."
        )
    except Exception:
        logger.info(f"[Discord] Naive IC check: {naive_ic_check:.4f}, starting walk-forward...")

    # ----------------------------------------------------------------
    # Step 1: Walk-forward LightGBM on vol residual
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Walk-forward LightGBM on vol RESIDUAL")
    logger.info("=" * 60)

    t0 = time.time()
    residual_results = walk_forward_residual(
        scanner=scanner,
        future_vol=future_vol_5s,
        past_vol=past_vol,
        exclude_features=EXCLUDE_FEATURES,
        min_train_days=3,
    )
    residual_elapsed = time.time() - t0
    logger.info(f"Residual scan done in {residual_elapsed:.0f}s")

    # ----------------------------------------------------------------
    # Step 2: Direction conditional on high vol
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Direction prediction in HIGH-VOL regime")
    logger.info("=" * 60)

    t0 = time.time()
    direction_results = walk_forward_direction_highvol(
        scanner=scanner,
        future_vol=future_vol_5s,
        past_vol=past_vol,
        future_return=future_return_5s,
        exclude_features=EXCLUDE_FEATURES,
        min_train_days=3,
    )
    direction_elapsed = time.time() - t0
    logger.info(f"Direction scan done in {direction_elapsed:.0f}s")

    # ----------------------------------------------------------------
    # Compile and format results
    # ----------------------------------------------------------------
    total_elapsed = time.time() - t_start

    # Per-fold summary for residual
    fold_naive_ics = [f['ic_naive'] for f in residual_results.get('fold_results', [])]
    fold_combined_ics = [f['ic_combined'] for f in residual_results.get('fold_results', [])]
    fold_lgbm_ics = [f['ic_lgbm_residual'] for f in residual_results.get('fold_results', [])]

    report_lines = [
        "",
        "=" * 70,
        "LightGBM RESIDUAL VOL SCAN — RESULTS",
        "=" * 70,
        f"Naive persistence IC (full data sanity check): {naive_ic_check:.4f}",
        f"N snapshots: {N:,}  N days: {n_days}  N folds: {residual_results.get('n_folds', 0)}",
        "",
        "--- RESIDUAL MODEL (can we predict what persistence misses?) ---",
    ]

    if 'error' not in residual_results:
        report_lines += [
            f"  Naive baseline IC (walk-forward):       {residual_results['ic_naive']:.4f}"
            f"  (ICIR={residual_results['icir_naive']:.2f} t={residual_results['tstat_naive']:.2f})",
            f"  LightGBM residual IC (vs residual):     {residual_results['ic_lgbm_residual_vs_residual']:.4f}"
            f"  (ICIR={residual_results['icir_lgbm_resid']:.2f} t={residual_results['tstat_lgbm_resid']:.2f} p={residual_results['pval_lgbm_resid']:.3f})",
            f"  LightGBM residual IC (vs future vol):   {residual_results['ic_lgbm_absolute']:.4f}",
            f"  Combined (naive + alpha*LGBM) IC:       {residual_results['ic_combined']:.4f}"
            f"  (ICIR={residual_results['icir_combined']:.2f} t={residual_results['tstat_combined']:.2f} p={residual_results['pval_combined']:.3f})",
            "",
            "  Per-fold naive ICs:    " + " ".join(f"{x:+.3f}" for x in fold_naive_ics),
            "  Per-fold LGBM ICs:     " + " ".join(f"{x:+.3f}" for x in fold_lgbm_ics),
            "  Per-fold combined ICs: " + " ".join(f"{x:+.3f}" for x in fold_combined_ics),
            "",
        ]

        # Improvement check
        naive_mean = float(np.mean(fold_naive_ics)) if fold_naive_ics else 0.0
        combined_mean = float(np.mean(fold_combined_ics)) if fold_combined_ics else 0.0
        improvement = combined_mean - naive_mean

        report_lines += [
            f"  Mean naive fold IC:    {naive_mean:.4f}",
            f"  Mean combined fold IC: {combined_mean:.4f}",
            f"  Improvement:           {improvement:+.4f}",
            "",
        ]

        if residual_results.get('top_features'):
            report_lines.append("  Top features for residual prediction:")
            for fname, fimp in residual_results['top_features'][:10]:
                report_lines.append(f"    {fname:<35s}: {fimp:>8.0f}")
    else:
        report_lines.append(f"  ERROR: {residual_results['error']}")

    report_lines += [
        "",
        "--- DIRECTION IN HIGH-VOL REGIME ---",
    ]

    if 'error' not in direction_results:
        report_lines += [
            f"  Direction IC (all bars):      {direction_results['ic_direction_all_bars']:.4f}",
            f"  Direction IC (high-vol only): {direction_results['ic_direction_highvol']:.4f}"
            f"  (ICIR={direction_results['icir_highvol']:.2f} t={direction_results['tstat_highvol']:.2f} p={direction_results['pval_highvol']:.3f})",
            f"  Direction IC (low-vol only):  {direction_results['ic_direction_lowvol']:.4f}",
            "",
        ]

        hv_fold_ics = [f['ic_highvol'] for f in direction_results.get('fold_results', [])
                       if np.isfinite(f.get('ic_highvol', np.nan))]
        if hv_fold_ics:
            report_lines.append("  Per-fold high-vol direction ICs: " +
                                " ".join(f"{x:+.3f}" for x in hv_fold_ics))

        if direction_results.get('top_features'):
            report_lines.append("\n  Top features for direction prediction:")
            for fname, fimp in direction_results['top_features'][:8]:
                report_lines.append(f"    {fname:<35s}: {fimp:>8.0f}")
    else:
        report_lines.append(f"  ERROR: {direction_results['error']}")

    report_lines += [
        "",
        "=" * 70,
        "VERDICT",
        "=" * 70,
    ]

    # Honest assessment
    ic_combined = residual_results.get('ic_combined', 0.0)
    ic_naive = residual_results.get('ic_naive', 0.0)
    t_combined = residual_results.get('tstat_combined', 0.0)
    t_naive = residual_results.get('tstat_naive', 0.0)
    ic_hv_dir = direction_results.get('ic_direction_highvol', 0.0)
    t_hv = direction_results.get('tstat_highvol', 0.0)
    p_combined = residual_results.get('pval_combined', 1.0)

    if not np.isfinite(ic_combined):
        ic_combined = 0.0
    if not np.isfinite(ic_naive):
        ic_naive = 0.0
    if not np.isfinite(ic_hv_dir):
        ic_hv_dir = 0.0

    verdict = []
    if abs(ic_naive) > 0.5 and abs(t_naive) > 2.0:
        verdict.append(f"+ Naive persistence WORKS: walk-forward IC={ic_naive:.3f} (t={t_naive:.1f})")
    else:
        verdict.append(f"? Naive persistence walk-forward IC={ic_naive:.3f} (t={t_naive:.1f}) — weaker than full-data")

    improvement_pct = (ic_combined - ic_naive) / max(abs(ic_naive), 1e-6) * 100
    if ic_combined > ic_naive * 1.05 and p_combined < 0.05:
        verdict.append(f"+ LightGBM IMPROVES on naive: combined IC={ic_combined:.3f} vs naive={ic_naive:.3f} ({improvement_pct:+.1f}%)")
    elif ic_combined > ic_naive:
        verdict.append(f"~ Small improvement: combined IC={ic_combined:.3f} vs naive={ic_naive:.3f} ({improvement_pct:+.1f}%, p={p_combined:.3f})")
    else:
        verdict.append(f"- No improvement: combined IC={ic_combined:.3f} <= naive IC={ic_naive:.3f}")

    if abs(ic_hv_dir) > 0.02 and abs(t_hv) > 2.0:
        verdict.append(f"+ Direction in high-vol regime: IC={ic_hv_dir:.3f} (t={t_hv:.1f}) — POTENTIALLY EXPLOITABLE")
    else:
        verdict.append(f"- Direction in high-vol regime: IC={ic_hv_dir:.3f} (t={t_hv:.1f}) — not significant")

    overall = "ALPHA BEYOND PERSISTENCE EXISTS" if (
        (ic_combined > ic_naive * 1.05 and p_combined < 0.05) or
        (abs(ic_hv_dir) > 0.02 and abs(t_hv) > 2.0)
    ) else "NO MATERIAL ALPHA BEYOND NAIVE PERSISTENCE"

    verdict.append("")
    verdict.append(f"OVERALL: {overall}")

    report_lines += verdict
    report_lines.append(f"\nTotal elapsed: {total_elapsed:.0f}s")

    report = "\n".join(report_lines)
    logger.info(report)

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"residual_scan_{timestamp}.json"

    def make_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj

    output = {
        'timestamp': timestamp,
        'naive_ic_full_data_sanity': float(naive_ic_check),
        'n_snapshots': N,
        'n_days': n_days,
        'excluded_features': EXCLUDE_FEATURES,
        'residual_results': make_serializable(residual_results),
        'direction_results': make_serializable(direction_results),
        'report': report,
        'elapsed_sec': float(total_elapsed),
    }

    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {result_file}")

    # ----------------------------------------------------------------
    # Final Discord report
    # ----------------------------------------------------------------
    discord_lines = [
        "**[Residual Scan] COMPLETE**",
        "",
        "**Vol Residual Model:**",
        f"• Naive persistence IC (walk-fwd): `{ic_naive:.4f}` t={t_naive:.1f}",
        f"• LightGBM residual IC (vs vol): `{residual_results.get('ic_lgbm_absolute', 0):.4f}`",
        f"• Combined IC (naive + LGBM): `{ic_combined:.4f}` t={t_combined:.1f} p={p_combined:.3f}",
        "",
        "**Direction in High-Vol Regime:**",
        f"• IC high-vol bars: `{ic_hv_dir:.4f}` t={t_hv:.1f} p={direction_results.get('pval_highvol', 1.0):.3f}",
        f"• IC low-vol bars: `{direction_results.get('ic_direction_lowvol', 0):.4f}`",
        "",
        f"**VERDICT: {overall}**",
        "",
        "Per-fold combined ICs: " + " ".join(f"{x:+.3f}" for x in fold_combined_ics[:8]),
        f"\nResults: `{result_file.name}`",
        f"Elapsed: {total_elapsed:.0f}s",
    ]

    try:
        from lib.discord import send_message as discord_send
        discord_send("\n".join(discord_lines))
    except Exception:
        logger.info("[Discord Final Report]\n" + "\n".join(discord_lines))

    print("\n" + report)
    return output


if __name__ == '__main__':
    main()
