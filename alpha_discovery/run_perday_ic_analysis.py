"""
Per-Day IC Stability Analysis — Lvl3Quant MBO Alpha Project
============================================================

PURPOSE: Diagnose WHY IC is so variable depending on which cache files are loaded.
  - 14 files (27 days): IC=0.026
  - 16 files (31 days): IC=0.120
  - 19 files (34 days): IC=0.079

APPROACH:
1. Load all available cache files, tracking which day came from which file
2. Compute ret_3s target
3. Run walk-forward LightGBM (expanding window, 1-day purge gap)
4. For each test day: per-day IC, bar count, ret_3s stats, vol, file provenance
5. Cumulative IC progression as files are added 1..N

OUTPUT:
  - Detailed table: day | bars | IC | IC_pval | Vol(bps) | mean_ret | file_idx
  - Cumulative IC table: how IC changes as N files are loaded
  - JSON: results/perday_ic_analysis_*.json
"""

import gc
import sys
import json
import time
import logging
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
from alpha_discovery.mbo_features import (
    compute_mbo_features, get_feature_names, TOTAL_FEATURES,
    COL_BUY_VOL, COL_SELL_VOL,
)

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'perday_ic_analysis.log', mode='a'),
    ]
)
logger = logging.getLogger("perday_ic")

# Column indices in global_features (45 cols)
COL_TIME_SINCE_RTH = 28
COL_HOUR_NORM = 26

# LightGBM params matching other scripts (as specified in task)
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 500,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
}

# Features to exclude for direction prediction (same as run_return_multihorizon.py)
EXCLUDE_FEATURES = [
    'mid', 'best_bid', 'best_ask', 'microprice',
    'hour_norm', 'minute_norm', 'time_since_rth', 'time_to_close',
    'rvol_10', 'rvol_20', 'rvol_50',
    'vov_10', 'vov_20', 'vov_50',
    'tick_count',
    'event_int_5', 'event_int_20', 'event_int_50',
    'tick_density_5', 'tick_density_20',
]


# ============================================================================
# CUSTOM CACHE LOADER WITH FILE PROVENANCE TRACKING
# ============================================================================

def load_all_files_with_provenance(cache_dir: Path) -> dict:
    """
    Load all cache files tracking which day came from which file index.

    Returns:
        {
          'global_features': np.ndarray (N, 45),
          'mid_prices': np.ndarray (N,),
          'day_boundaries': list of int,  # [0, end_day0, end_day1, ...]
          'day_file_idx': list of int,    # file index for each day
          'day_file_name': list of str,   # filename for each day
          'n_days': int,
          'n_files': int,
          'cache_files': list of Path,
        }
    """
    cache_files = sorted(cache_dir.glob("file_*_snapshots.npz"))
    if not cache_files:
        raise ValueError(f"No cache files in {cache_dir}")

    logger.info(f"Found {len(cache_files)} cache files")

    all_global = []
    all_nodes = []
    all_mids = []
    day_boundaries = [0]
    day_file_idx = []
    day_file_name = []
    flat_days_skipped = 0

    for fi, fpath in enumerate(cache_files):
        t0 = time.time()
        data = np.load(str(fpath), allow_pickle=True)
        gf = data['global_features']
        nf = data['node_features']
        mp = data['mid_prices']

        # Detect day transitions within this file
        rth_col = gf[:, COL_TIME_SINCE_RTH]
        diffs = np.diff(rth_col)
        day_transitions = np.where(diffs < -0.3)[0]
        raw_day_starts = np.concatenate([[0], day_transitions + 1, [len(gf)]])
        n_file_days = len(raw_day_starts) - 1

        file_days_loaded = 0
        for d in range(n_file_days):
            start = raw_day_starts[d]
            end = raw_day_starts[d + 1]

            gf_day = gf[start:end]
            nf_day = nf[start:end]
            mp_day = mp[start:end]

            # RTH filter
            rth_day = gf_day[:, COL_TIME_SINCE_RTH]
            rth_mask = (rth_day >= 0.0) & (rth_day <= 1.0)

            gf_rth = gf_day[rth_mask]
            nf_rth = nf_day[rth_mask]
            mp_rth = mp_day[rth_mask]

            if len(mp_rth) < 100:
                logger.warning(f"  File {fi} Day {d}: only {len(mp_rth)} RTH bars, skipping")
                continue

            # FLAT day filter
            price_range = mp_rth.max() - mp_rth.min()
            if price_range < 1.0:
                flat_days_skipped += 1
                logger.info(f"  File {fi} Day {d}: FLAT (range={price_range:.2f}pts), skipping")
                continue

            all_global.append(gf_rth)
            all_nodes.append(nf_rth)
            all_mids.append(mp_rth)
            day_boundaries.append(day_boundaries[-1] + len(mp_rth))
            day_file_idx.append(fi)
            day_file_name.append(fpath.name)
            file_days_loaded += 1

        elapsed = time.time() - t0
        logger.info(
            f"  [{fi+1}/{len(cache_files)}] {fpath.name}: "
            f"{n_file_days} raw days, {file_days_loaded} real days loaded "
            f"[{elapsed:.1f}s]"
        )
        data.close()

    if flat_days_skipped > 0:
        logger.info(f"Filtered {flat_days_skipped} FLAT days")

    if not all_global:
        raise ValueError("No valid day segments found")

    n_days = len(day_boundaries) - 1
    total_bars = sum(len(g) for g in all_global)
    logger.info(f"Loaded {n_days} trading days from {len(cache_files)} files, "
                f"{total_bars:,} total RTH bars")

    # Concatenate
    global_feats_raw = np.concatenate(all_global)
    node_feats_raw = np.concatenate(all_nodes)
    mid_prices = np.concatenate(all_mids)

    del all_nodes
    gc.collect()

    return {
        'global_features_raw': global_feats_raw,
        'node_features_raw': node_feats_raw,
        'mid_prices': mid_prices,
        'day_boundaries': day_boundaries,
        'day_file_idx': day_file_idx,
        'day_file_name': day_file_name,
        'n_days': n_days,
        'n_files': len(cache_files),
        'cache_files': cache_files,
    }


# ============================================================================
# COMPUTE ret_3s TARGET
# ============================================================================

def compute_ret3s(mid_prices: np.ndarray, day_boundaries: list,
                  sample_interval_ms: int = 100) -> np.ndarray:
    """
    Compute 3-second log return target with NaN-fill at day boundaries.
    """
    N = len(mid_prices)
    steps = int(3.0 * 1000 / sample_interval_ms)  # 3s at 100ms = 30 steps
    n_days = len(day_boundaries) - 1

    log_mid = np.log(np.maximum(mid_prices, 1.0))

    future_log = np.empty(N, dtype=np.float32)
    future_log[:N - steps] = log_mid[steps:]
    future_log[N - steps:] = np.nan

    ret = (future_log - log_mid).astype(np.float32)

    # NaN-fill bars whose forward window crosses a day boundary
    if n_days > 1:
        for d in range(n_days - 1):
            day_end = day_boundaries[d + 1]
            nan_start = max(day_boundaries[d], day_end - steps)
            ret[nan_start:day_end] = np.nan

    valid = np.isfinite(ret).sum()
    logger.info(f"ret_3s: steps={steps}, valid={valid:,}/{N:,}")
    return ret


# ============================================================================
# WALK-FORWARD WITH PER-DAY IC TRACKING
# ============================================================================

def walk_forward_perday(
    features: np.ndarray,
    feature_names: list,
    target: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: list,
    day_file_idx: list,
    day_file_name: list,
    min_train_days: int = 3,
) -> dict:
    """
    Walk-forward LightGBM with per-day IC decomposition.

    For each test day:
      - Train on expanding window (days 0..test_day-2 with 1-day purge gap)
      - Predict test day
      - Compute: per-day IC (Spearman), bar count, mean/std ret_3s, vol, file_idx

    Returns dict with per-day results and aggregate metrics.
    """
    import lightgbm as lgb

    n_days = len(day_boundaries) - 1
    logger.info(f"Walk-forward: {n_days} days, min_train={min_train_days}")

    # Build feature mask (exclude contaminating features)
    keep_mask = np.array([fn not in EXCLUDE_FEATURES for fn in feature_names])
    features_use = features[:, keep_mask]
    feature_names_use = [fn for fn in feature_names if fn not in EXCLUDE_FEATURES]
    n_feat = len(feature_names_use)
    logger.info(f"Using {n_feat} features (excluded {(~keep_mask).sum()} vol/price/time features)")

    per_day_results = []
    all_preds = []
    all_actuals = []
    fold_ics = []

    for test_day in range(min_train_days, n_days):
        # Expanding window train: days 0..(test_day-2), then 1-day purge gap
        train_end_day = test_day - 1  # exclude test_day-1 as purge gap
        train_start = day_boundaries[0]
        train_end = day_boundaries[train_end_day]  # NOT train_end_day+1 — skip purge day

        # Wait — standard walk-forward: train on [0, test_day-1), purge test_day-1
        # So train = days 0..test_day-2 (inclusive)
        # i.e., train_end = day_boundaries[test_day - 1]
        train_end = day_boundaries[test_day - 1]

        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_train = features_use[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features_use[test_start:test_end]
        y_test = target[test_start:test_end]
        mid_test = mid_prices[test_start:test_end]

        # Remove NaN targets
        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        n_train = train_valid.sum()
        n_test = test_valid.sum()

        day_info = {
            'day_idx': test_day,
            'file_idx': day_file_idx[test_day],
            'file_name': day_file_name[test_day],
            'n_bars_total': int(test_end - test_start),
            'n_bars_valid': int(n_test),
            'n_train_bars': int(n_train),
            'ic': None,
            'ic_pval': None,
            'mean_ret': None,
            'std_ret': None,
            'vol_bps': None,
            'skipped': False,
            'skip_reason': None,
        }

        # Per-day stats (always computed, regardless of model)
        if n_test > 0:
            y_te_all = y_test[test_valid]
            mid_te_all = mid_test[test_valid]
            day_info['mean_ret'] = float(np.nanmean(y_te_all))
            day_info['std_ret'] = float(np.nanstd(y_te_all))

            # Volatility: std of log mid returns in ticks (bps-equivalent)
            log_mids = np.log(np.maximum(mid_te_all, 1.0))
            log_rets = np.diff(log_mids)
            if len(log_rets) > 0:
                day_info['vol_bps'] = float(np.std(log_rets) * 10000)  # in basis points
            else:
                day_info['vol_bps'] = 0.0

        if n_train < LGBM_PARAMS['min_child_samples']:
            day_info['skipped'] = True
            day_info['skip_reason'] = f'train too small ({n_train})'
            per_day_results.append(day_info)
            logger.warning(f"  Day {test_day}: SKIP — train={n_train} bars")
            continue

        if n_test < 50:
            day_info['skipped'] = True
            day_info['skip_reason'] = f'test too small ({n_test})'
            per_day_results.append(day_info)
            logger.warning(f"  Day {test_day}: SKIP — test={n_test} bars")
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        # Train with 80/20 internal split for early stopping
        split = int(len(X_tr) * 0.8)
        if split < 100:
            split = len(X_tr)  # no early stopping for tiny sets

        try:
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            if split < len(X_tr):
                model.fit(
                    X_tr[:split], y_tr[:split],
                    eval_set=[(X_tr[split:], y_tr[split:])],
                    callbacks=[lgb.early_stopping(30, verbose=False)],
                )
            else:
                model.fit(X_tr, y_tr)
        except Exception as e:
            day_info['skipped'] = True
            day_info['skip_reason'] = f'train failed: {e}'
            per_day_results.append(day_info)
            logger.warning(f"  Day {test_day}: TRAIN FAILED — {e}")
            continue

        preds = model.predict(X_te)

        # Per-day IC
        try:
            ic_val, ic_pval = spearmanr(preds, y_te)
            if not np.isfinite(ic_val):
                ic_val, ic_pval = 0.0, 1.0
        except Exception:
            ic_val, ic_pval = 0.0, 1.0

        day_info['ic'] = float(ic_val)
        day_info['ic_pval'] = float(ic_pval)
        day_info['best_iteration'] = int(model.best_iteration_) if hasattr(model, 'best_iteration_') and model.best_iteration_ else LGBM_PARAMS['n_estimators']

        per_day_results.append(day_info)
        fold_ics.append(float(ic_val))
        all_preds.append(preds)
        all_actuals.append(y_te)

        sign = '+' if ic_val >= 0 else ''
        logger.info(
            f"  Day {test_day:3d} (file={day_file_idx[test_day]:2d}): "
            f"IC={sign}{ic_val:.4f} p={ic_pval:.3f} "
            f"n_test={n_test:,} vol={day_info['vol_bps']:.2f}bps "
            f"train={n_train:,}"
        )

        del model
        gc.collect()

    # Aggregate overall IC
    aggregate = {}
    if all_preds:
        p_all = np.concatenate(all_preds)
        a_all = np.concatenate(all_actuals)
        valid = np.isfinite(p_all) & np.isfinite(a_all)
        p_v, a_v = p_all[valid], a_all[valid]

        if len(p_v) > 10:
            ic_overall = float(spearmanr(p_v, a_v)[0])
        else:
            ic_overall = 0.0

        if len(fold_ics) > 2:
            ic_mean = float(np.mean(fold_ics))
            ic_std = float(np.std(fold_ics))
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            tstat = icir * np.sqrt(len(fold_ics))
            try:
                _, pvalue = ttest_1samp(fold_ics, 0)
                pvalue = float(pvalue)
            except Exception:
                pvalue = 1.0
        else:
            ic_mean = float(np.mean(fold_ics)) if fold_ics else 0.0
            ic_std, icir, tstat, pvalue = 0.0, 0.0, 0.0, 1.0

        aggregate = {
            'ic_overall': ic_overall,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'tstat': tstat,
            'pvalue': pvalue,
            'n_folds': len(fold_ics),
            'n_predictions': int(len(p_v)),
            'fold_ics': fold_ics,
            'n_positive_folds': sum(1 for x in fold_ics if x > 0),
        }

    return {
        'per_day': per_day_results,
        'aggregate': aggregate,
    }


# ============================================================================
# CUMULATIVE IC PROGRESSION (vary how many files we load)
# ============================================================================

def compute_cumulative_ic_progression(
    cache_dir: Path,
    min_days_for_test: int = 4,  # need at least min_train + 1 test day
) -> list:
    """
    For N = 1..max_files:
      Load files 1..N, compute features, run walk-forward on ret_3s,
      report overall IC, ICIR, t-stat, n_days.

    This directly replicates the condition that caused:
      14 files (27 days): IC=0.026
      16 files (31 days): IC=0.120
      19 files (34 days): IC=0.079
    """
    cache_files = sorted(cache_dir.glob("file_*_snapshots.npz"))
    n_total = len(cache_files)
    logger.info(f"\nCumulative IC progression: will test {n_total} file subsets")

    results = []

    for n_files in range(1, n_total + 1):
        files_subset = cache_files[:n_files]
        logger.info(f"\n{'='*50}")
        logger.info(f"Files 1..{n_files}: loading {n_files} files")

        t0 = time.time()

        # Load this subset
        all_global = []
        all_nodes = []
        all_mids = []
        day_boundaries = [0]
        day_file_idx_sub = []
        flat_skipped = 0

        for fi, fpath in enumerate(files_subset):
            data = np.load(str(fpath), allow_pickle=True)
            gf = data['global_features']
            nf = data['node_features']
            mp = data['mid_prices']

            rth_col = gf[:, COL_TIME_SINCE_RTH]
            diffs = np.diff(rth_col)
            day_transitions = np.where(diffs < -0.3)[0]
            raw_day_starts = np.concatenate([[0], day_transitions + 1, [len(gf)]])
            n_file_days = len(raw_day_starts) - 1

            for d in range(n_file_days):
                start = raw_day_starts[d]
                end = raw_day_starts[d + 1]

                gf_day = gf[start:end]
                nf_day = nf[start:end]
                mp_day = mp[start:end]

                rth_day = gf_day[:, COL_TIME_SINCE_RTH]
                rth_mask = (rth_day >= 0.0) & (rth_day <= 1.0)

                gf_rth = gf_day[rth_mask]
                nf_rth = nf_day[rth_mask]
                mp_rth = mp_day[rth_mask]

                if len(mp_rth) < 100:
                    continue

                price_range = mp_rth.max() - mp_rth.min()
                if price_range < 1.0:
                    flat_skipped += 1
                    continue

                all_global.append(gf_rth)
                all_nodes.append(nf_rth)
                all_mids.append(mp_rth)
                day_boundaries.append(day_boundaries[-1] + len(mp_rth))
                day_file_idx_sub.append(fi)

            data.close()

        if not all_global:
            logger.warning(f"  No valid days for {n_files} files, skipping")
            continue

        n_days_sub = len(day_boundaries) - 1
        total_bars = sum(len(g) for g in all_global)

        if n_days_sub < min_days_for_test:
            logger.info(f"  Only {n_days_sub} days (need {min_days_for_test}), skipping IC calc")
            results.append({
                'n_files': n_files,
                'n_days': n_days_sub,
                'n_bars': total_bars,
                'ic': None,
                'icir': None,
                'tstat': None,
                'n_folds': 0,
                'skipped': True,
                'reason': f'too few days ({n_days_sub})',
            })
            continue

        # Concatenate
        global_feats_raw = np.concatenate(all_global)
        node_feats_raw = np.concatenate(all_nodes)
        mid_prices_sub = np.concatenate(all_mids)

        del all_nodes
        gc.collect()

        # Compute features
        logger.info(f"  Computing features for {n_days_sub} days, {total_bars:,} bars...")
        try:
            features_sub = compute_mbo_features(
                mid_prices=mid_prices_sub,
                global_features_raw=global_feats_raw,
                node_features_raw=node_feats_raw,
                tick_size=0.25,
                depth_levels=10,
                day_boundaries=day_boundaries,
            )
        except Exception as e:
            logger.error(f"  Feature computation failed: {e}")
            results.append({
                'n_files': n_files,
                'n_days': n_days_sub,
                'n_bars': total_bars,
                'ic': None,
                'icir': None,
                'tstat': None,
                'n_folds': 0,
                'skipped': True,
                'reason': f'feature error: {e}',
            })
            continue

        feature_names_sub = get_feature_names()

        # Compute ret_3s target
        target_sub = compute_ret3s(mid_prices_sub, day_boundaries)

        # Feature mask
        keep_mask = np.array([fn not in EXCLUDE_FEATURES for fn in feature_names_sub])
        features_use_sub = features_sub[:, keep_mask]
        feature_names_use = [fn for fn in feature_names_sub if fn not in EXCLUDE_FEATURES]
        import lightgbm as lgb

        # Walk-forward
        fold_ics_sub = []
        all_preds_sub = []
        all_actuals_sub = []
        min_train_days = 3

        for test_day in range(min_train_days, n_days_sub):
            train_end = day_boundaries[test_day - 1]
            test_start = day_boundaries[test_day]
            test_end = day_boundaries[test_day + 1]

            X_tr_full = features_use_sub[:train_end]
            y_tr_full = target_sub[:train_end]
            X_te = features_use_sub[test_start:test_end]
            y_te = target_sub[test_start:test_end]

            train_valid = np.isfinite(y_tr_full)
            test_valid = np.isfinite(y_te)

            if train_valid.sum() < LGBM_PARAMS['min_child_samples'] or test_valid.sum() < 50:
                continue

            X_tr = X_tr_full[train_valid]
            y_tr = y_tr_full[train_valid]
            X_te_v = X_te[test_valid]
            y_te_v = y_te[test_valid]

            split = int(len(X_tr) * 0.8)
            try:
                model = lgb.LGBMRegressor(**LGBM_PARAMS)
                if split < len(X_tr) and split >= 50:
                    model.fit(
                        X_tr[:split], y_tr[:split],
                        eval_set=[(X_tr[split:], y_tr[split:])],
                        callbacks=[lgb.early_stopping(30, verbose=False)],
                    )
                else:
                    model.fit(X_tr, y_tr)
            except Exception:
                continue

            preds = model.predict(X_te_v)
            try:
                ic_fold = float(spearmanr(preds, y_te_v)[0])
                if np.isfinite(ic_fold):
                    fold_ics_sub.append(ic_fold)
                    all_preds_sub.append(preds)
                    all_actuals_sub.append(y_te_v)
            except Exception:
                pass

            del model
            gc.collect()

        elapsed = time.time() - t0

        if fold_ics_sub:
            p_all = np.concatenate(all_preds_sub)
            a_all = np.concatenate(all_actuals_sub)
            valid = np.isfinite(p_all) & np.isfinite(a_all)
            ic_overall = float(spearmanr(p_all[valid], a_all[valid])[0]) if valid.sum() > 10 else 0.0

            ic_mean = float(np.mean(fold_ics_sub))
            ic_std = float(np.std(fold_ics_sub))
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            tstat = icir * np.sqrt(len(fold_ics_sub))
        else:
            ic_overall, ic_mean, ic_std, icir, tstat = 0.0, 0.0, 0.0, 0.0, 0.0

        logger.info(
            f"  Files 1..{n_files}: {n_days_sub} days, {total_bars:,} bars, "
            f"IC={ic_overall:.4f} (fold_mean={ic_mean:.4f}) "
            f"ICIR={icir:.2f} t={tstat:.2f} "
            f"folds={len(fold_ics_sub)} [{elapsed:.0f}s]"
        )

        results.append({
            'n_files': n_files,
            'n_days': n_days_sub,
            'n_bars': int(total_bars),
            'ic': float(ic_overall),
            'ic_mean': float(ic_mean),
            'ic_std': float(ic_std),
            'icir': float(icir),
            'tstat': float(tstat),
            'n_folds': len(fold_ics_sub),
            'fold_ics': [float(x) for x in fold_ics_sub],
            'elapsed_sec': elapsed,
            'skipped': False,
        })

        del global_feats_raw, node_feats_raw, features_sub, mid_prices_sub, target_sub
        gc.collect()

    return results


# ============================================================================
# FORMATTING
# ============================================================================

def format_perday_table(per_day_results: list) -> str:
    """Format per-day IC results as ASCII table."""
    lines = [
        "",
        "PER-DAY IC STABILITY ANALYSIS",
        "=" * 95,
        f"{'Day':>4s} | {'File':>4s} | {'Bars':>7s} | {'IC':>7s} | {'p-val':>6s} | "
        f"{'Vol(bps)':>8s} | {'Mean_ret':>9s} | {'Std_ret':>8s} | {'File Name'}",
        "-" * 95,
    ]

    for r in per_day_results:
        day_idx = r['day_idx']
        file_idx = r['file_idx']
        n_bars = r['n_bars_valid']
        fname = r.get('file_name', '?')

        if r.get('skipped'):
            lines.append(
                f"{day_idx:>4d} | {file_idx:>4d} | {n_bars:>7,d} | "
                f"{'SKIP':>7s} | {'---':>6s} | {'---':>8s} | {'---':>9s} | {'---':>8s} | "
                f"{fname} ({r.get('skip_reason', '')})"
            )
            continue

        ic = r.get('ic', 0.0) or 0.0
        pval = r.get('ic_pval', 1.0) or 1.0
        vol = r.get('vol_bps', 0.0) or 0.0
        mean_ret = r.get('mean_ret', 0.0) or 0.0
        std_ret = r.get('std_ret', 0.0) or 0.0

        ic_str = f"{ic:+.4f}"
        pval_str = f"{pval:.4f}"

        # Highlight significant days
        sig = " ***" if pval < 0.05 else ("  **" if pval < 0.10 else "")

        lines.append(
            f"{day_idx:>4d} | {file_idx:>4d} | {n_bars:>7,d} | "
            f"{ic_str:>7s} | {pval_str:>6s} | "
            f"{vol:>8.3f} | {mean_ret:>+9.6f} | {std_ret:>8.6f} | "
            f"{fname}{sig}"
        )

    lines.append("=" * 95)

    # Summary stats
    valid_days = [r for r in per_day_results if not r.get('skipped') and r.get('ic') is not None]
    if valid_days:
        ics = [r['ic'] for r in valid_days]
        n_pos = sum(1 for ic in ics if ic > 0)
        n_sig = sum(1 for r in valid_days if r.get('ic_pval', 1.0) < 0.05)
        lines.append(f"\nSUMMARY: {len(valid_days)} days evaluated")
        lines.append(f"  IC range: [{min(ics):.4f}, {max(ics):.4f}]")
        lines.append(f"  Mean IC: {np.mean(ics):.4f}, Std IC: {np.std(ics):.4f}")
        lines.append(f"  Positive IC: {n_pos}/{len(valid_days)} ({n_pos/len(valid_days):.0%})")
        lines.append(f"  Significant (p<0.05): {n_sig}/{len(valid_days)}")

        # Best and worst days
        best = max(valid_days, key=lambda r: r['ic'])
        worst = min(valid_days, key=lambda r: r['ic'])
        lines.append(f"  Best day: idx={best['day_idx']} (file={best['file_idx']}) IC={best['ic']:+.4f}")
        lines.append(f"  Worst day: idx={worst['day_idx']} (file={worst['file_idx']}) IC={worst['ic']:+.4f}")

        # Sort by IC to find drivers
        sorted_days = sorted(valid_days, key=lambda r: r['ic'], reverse=True)
        lines.append(f"\nTOP 5 BEST DAYS (highest IC):")
        for r in sorted_days[:5]:
            lines.append(
                f"  Day {r['day_idx']} (file={r['file_idx']}): IC={r['ic']:+.4f} "
                f"vol={r.get('vol_bps', 0):.2f}bps bars={r['n_bars_valid']:,}"
            )
        lines.append(f"\nBOTTOM 5 WORST DAYS (lowest IC):")
        for r in sorted_days[-5:]:
            lines.append(
                f"  Day {r['day_idx']} (file={r['file_idx']}): IC={r['ic']:+.4f} "
                f"vol={r.get('vol_bps', 0):.2f}bps bars={r['n_bars_valid']:,}"
            )

    return "\n".join(lines)


def format_cumulative_table(cumulative_results: list) -> str:
    """Format cumulative IC progression as ASCII table."""
    lines = [
        "",
        "CUMULATIVE IC PROGRESSION (Adding Files One By One)",
        "=" * 75,
        f"{'N_files':>7s} | {'N_days':>6s} | {'N_bars':>9s} | "
        f"{'IC':>7s} | {'ICIR':>6s} | {'t':>6s} | {'N_folds':>7s}",
        "-" * 75,
    ]

    prev_ic = None
    for r in cumulative_results:
        n_files = r['n_files']
        n_days = r['n_days']
        n_bars = r['n_bars']

        if r.get('skipped'):
            lines.append(
                f"{n_files:>7d} | {n_days:>6d} | {n_bars:>9,d} | "
                f"{'---':>7s} | {'---':>6s} | {'---':>6s} | "
                f"{'SKIP':>7s} ({r.get('reason', '')})"
            )
            continue

        ic = r.get('ic', 0.0) or 0.0
        icir = r.get('icir', 0.0) or 0.0
        tstat = r.get('tstat', 0.0) or 0.0
        n_folds = r.get('n_folds', 0)

        # Delta IC from previous
        delta = ""
        if prev_ic is not None:
            d = ic - prev_ic
            delta = f" ({d:+.4f})"
        prev_ic = ic

        lines.append(
            f"{n_files:>7d} | {n_days:>6d} | {n_bars:>9,d} | "
            f"{ic:>+7.4f}{delta:<12s} | {icir:>6.2f} | {tstat:>6.2f} | {n_folds:>7d}"
        )

    lines.append("=" * 75)

    # Find key transitions
    valid = [r for r in cumulative_results if not r.get('skipped') and r.get('ic') is not None]
    if valid:
        ics = [r['ic'] for r in valid]
        max_ic = max(ics)
        min_ic = min(ics)
        best_n = valid[ics.index(max_ic)]['n_files']
        worst_n = valid[ics.index(min_ic)]['n_files']
        lines.append(f"\nKey transitions:")
        lines.append(f"  Peak IC={max_ic:.4f} at {best_n} files")
        lines.append(f"  Min  IC={min_ic:.4f} at {worst_n} files")

        # Find biggest single-file jumps
        jumps = []
        for i in range(1, len(valid)):
            delta = valid[i]['ic'] - valid[i-1]['ic']
            jumps.append((valid[i]['n_files'], delta, valid[i-1]['ic'], valid[i]['ic']))

        jumps.sort(key=lambda x: abs(x[1]), reverse=True)
        lines.append(f"\n  Biggest IC jumps (adding one file):")
        for n_files, delta, ic_before, ic_after in jumps[:5]:
            sign = '+' if delta >= 0 else ''
            lines.append(
                f"    File #{n_files}: {ic_before:.4f} -> {ic_after:.4f} "
                f"({sign}{delta:.4f})"
            )

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Per-day IC stability analysis')
    parser.add_argument('--skip-cumulative', action='store_true',
                        help='Skip cumulative IC progression (much faster)')
    parser.add_argument('--cumulative-only', action='store_true',
                        help='Only run cumulative IC, skip per-day analysis')
    parser.add_argument('--min-train-days', type=int, default=3,
                        help='Min training days before first test day')
    args = parser.parse_args()

    logger.info("=" * 75)
    logger.info("PER-DAY IC STABILITY ANALYSIS")
    logger.info(f"  Target: ret_3s (3-second log return)")
    logger.info(f"  Model: LightGBM walk-forward, 1-day purge gap")
    logger.info(f"  Excluded: {len(EXCLUDE_FEATURES)} vol/price/time features")
    logger.info("=" * 75)

    t_start = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"perday_ic_analysis_{timestamp}.json"

    cache_dir = ROOT / "data" / "processed" / "medium_snapshots_cache"

    output = {
        'timestamp': timestamp,
        'per_day_analysis': None,
        'cumulative_analysis': None,
    }

    # ----------------------------------------------------------------
    # PART 1: Per-day IC analysis (all files, per-day breakdown)
    # ----------------------------------------------------------------
    if not args.cumulative_only:
        # send_to_discord is not available inside the script — logging is used instead
        def send_to_discord(msg):
            pass

        logger.info("\n" + "=" * 60)
        logger.info("PART 1: Loading all cache files...")
        logger.info("=" * 60)

        raw_data = load_all_files_with_provenance(cache_dir)

        n_days = raw_data['n_days']
        n_bars = len(raw_data['mid_prices'])
        logger.info(f"Loaded: {n_days} days, {n_bars:,} bars from {raw_data['n_files']} files")

        # Compute full feature set
        logger.info(f"\nComputing {TOTAL_FEATURES} features...")
        features_all = compute_mbo_features(
            mid_prices=raw_data['mid_prices'],
            global_features_raw=raw_data['global_features_raw'],
            node_features_raw=raw_data['node_features_raw'],
            tick_size=0.25,
            depth_levels=10,
            day_boundaries=raw_data['day_boundaries'],
        )
        feature_names_all = get_feature_names()
        logger.info(f"Features shape: {features_all.shape}")

        # Free raw node features (large)
        del raw_data['node_features_raw'], raw_data['global_features_raw']
        gc.collect()

        # Compute ret_3s target
        logger.info("\nComputing ret_3s target...")
        target_all = compute_ret3s(
            raw_data['mid_prices'],
            raw_data['day_boundaries'],
        )

        logger.info(f"\nRunning walk-forward per-day IC analysis ({n_days} days)...")

        wf_results = walk_forward_perday(
            features=features_all,
            feature_names=feature_names_all,
            target=target_all,
            mid_prices=raw_data['mid_prices'],
            day_boundaries=raw_data['day_boundaries'],
            day_file_idx=raw_data['day_file_idx'],
            day_file_name=raw_data['day_file_name'],
            min_train_days=args.min_train_days,
        )

        per_day_table = format_perday_table(wf_results['per_day'])
        logger.info(per_day_table)
        print(per_day_table)

        agg = wf_results['aggregate']
        if agg:
            logger.info(f"\nOVERALL (all {n_days} days):")
            logger.info(f"  IC={agg['ic_overall']:.4f} (fold_mean={agg['ic_mean']:.4f} ± {agg['ic_std']:.4f})")
            logger.info(f"  ICIR={agg['icir']:.2f} t={agg['tstat']:.2f} p={agg['pvalue']:.4f}")
            logger.info(f"  Folds: {agg['n_folds']} ({agg['n_positive_folds']} positive)")

        output['per_day_analysis'] = {
            'n_days': n_days,
            'n_files': raw_data['n_files'],
            'n_bars': n_bars,
            'per_day': wf_results['per_day'],
            'aggregate': agg,
            'table': per_day_table,
        }

        del features_all, target_all
        gc.collect()

        elapsed_part1 = time.time() - t_start
        logger.info(f"\nPart 1 complete in {elapsed_part1:.0f}s ({elapsed_part1/60:.1f} min)")

    # ----------------------------------------------------------------
    # PART 2: Cumulative IC progression (1..N files)
    # ----------------------------------------------------------------
    if not args.skip_cumulative:
        logger.info("\n" + "=" * 60)
        logger.info("PART 2: Cumulative IC progression (1..N files)")
        logger.info("=" * 60)
        logger.info("This will take a while (recomputes features for each N)")

        cumulative_results = compute_cumulative_ic_progression(
            cache_dir=cache_dir,
            min_days_for_test=args.min_train_days + 1,
        )

        cumulative_table = format_cumulative_table(cumulative_results)
        logger.info(cumulative_table)
        print(cumulative_table)

        output['cumulative_analysis'] = {
            'results': cumulative_results,
            'table': cumulative_table,
        }

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    elapsed_total = time.time() - t_start

    output['elapsed_sec'] = elapsed_total
    output['excluded_features'] = EXCLUDE_FEATURES
    output['lgbm_params'] = LGBM_PARAMS

    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_file}")
    logger.info(f"Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    # Print final summary
    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETE")
    print(f"Results: {results_file}")

    if output.get('per_day_analysis') and output['per_day_analysis'].get('aggregate'):
        agg = output['per_day_analysis']['aggregate']
        print(f"\nOVERALL IC (all files): {agg['ic_overall']:.4f}")
        print(f"  fold_mean={agg['ic_mean']:.4f} ± {agg['ic_std']:.4f}")
        print(f"  ICIR={agg['icir']:.2f} t={agg['tstat']:.2f} p={agg['pvalue']:.4f}")
        print(f"  {agg['n_folds']} folds, {agg['n_positive_folds']} positive")

    if output.get('cumulative_analysis'):
        cr = output['cumulative_analysis']['results']
        valid_cr = [r for r in cr if not r.get('skipped') and r.get('ic') is not None]
        if valid_cr:
            print("\nCUMULATIVE PROGRESSION SUMMARY:")
            # Show key milestones
            milestones = [14, 16, 19, len(cr)]
            for m in milestones:
                r = next((x for x in valid_cr if x['n_files'] == m), None)
                if r:
                    print(f"  {m} files ({r['n_days']} days): IC={r['ic']:.4f} t={r['tstat']:.2f}")

    print("=" * 75)
    return output


if __name__ == '__main__':
    main()
