"""
Low-Frequency Resampling Test — Can we trade the direction signal profitably?

KEY QUESTION: At 100ms, IC=0.08-0.11 but costs ($12.50/trade) dominate because
ES barely moves in 1-10 seconds. Does aggregating to lower frequency bars
(1s, 5s, 30s, 1min, 5min) preserve or improve the signal enough to be tradable?

APPROACH:
1. Resample 100ms features to longer timeframes using numpy reshape
2. Compute direction targets at the new frequency (e.g., 1s bars predict ret_5s)
3. Walk-forward LightGBM on each resampled frequency
4. Compute realistic PnL at each frequency
5. Report IC, trades/day, gross PnL, net PnL, Sharpe

DATA:
- scanner.features: (3168000, 149) at 100ms
- scanner.mid_prices: (3168000,)
- scanner.day_boundaries: [0, 198000, 396000, ...]  (198000 bars/day = 5.5 RTH hrs)
- 100ms * 198000 = 19,800,000ms = 19,800s = 330min = 5.5 hours exactly

Usage:
    python alpha_discovery/run_low_freq_test.py
    python alpha_discovery/run_low_freq_test.py --fast
    python alpha_discovery/run_low_freq_test.py --freqs 1s 5s 1min
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
    load_feature_cache, compute_return_targets,
    EXCLUDE_FEATURES_DIRECTION,
)
from alpha_discovery.run_model_refinement import walk_forward_evaluate
from alpha_discovery.run_realistic_pnl import simulate_realistic_trading

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'low_freq_test.log', mode='a'),
    ]
)
logger = logging.getLogger("low_freq_test")

# ============================================================================
# CONSTANTS
# ============================================================================
BARS_PER_DAY_100MS = 198000          # exactly 5.5 hours * 3600s * 10 bars/s
BARS_PER_SEC = 10                     # 100ms -> 10 bars per second
ES_POINT_VALUE = 50.0                 # $50 per point
TICK_VALUE = 12.50                    # $12.50 per tick
RT_COST = TICK_VALUE                  # $12.50 round-trip (1 tick)

# Features to use as FLOW features (mean over window) vs SNAPSHOT features (last value)
# Flow features: quantities that accumulate or average over time
FLOW_FEATURE_KEYWORDS = [
    'ofi', 'trade_imb', 'net_flow', 'vpin', 'cancel_to_add', 'trade_to_add',
    'aggr_imb', 'aggr_buy', 'aggr_sell', 'buy_vol', 'sell_vol', 'buy_volume',
    'sell_volume', 'cancel_bid', 'cancel_ask', 'cancel_side',
    'modify_to_add', 'fleeting', 'event_int', 'tick_density', 'tick_count',
    'absorption', 'spoof', 'iceberg', 'price_impact', 'kyle_lambda',
    'adverse_sel', 'toxicity', 'vol_direction', 'flow_during',
    'aggr_momentum', 'mid_ret', 'mid_accel', 'ret_vel',
]


def is_flow_feature(feature_name: str) -> bool:
    """Return True if feature should be averaged (not snapshot) over the window."""
    fn_lower = feature_name.lower()
    return any(kw in fn_lower for kw in FLOW_FEATURE_KEYWORDS)


# ============================================================================
# FREQUENCY CONFIGURATION
# ============================================================================

# Each frequency config specifies:
# - resample_factor: how many 100ms bars per resampled bar
# - targets: list of (target_name, horizon_bars_at_resample_freq)
#   where horizon_bars_at_resample_freq is the number of RESAMPLED bars ahead to predict
FREQ_CONFIGS = {
    '1s': {
        'resample_factor': 10,          # 100ms * 10 = 1s
        'bars_per_day': 19800,          # 198000 / 10
        'targets': [
            ('ret_5s',  5),             # 5 bars ahead at 1s = 5s
            ('ret_30s', 30),            # 30 bars ahead at 1s = 30s
            ('ret_1min', 60),           # 60 bars ahead at 1s = 1min
        ],
        'pnl_entry_pct': 80,
        'min_hold_bars': 5,             # 5s minimum hold
        'max_hold_bars': 60,            # 60s maximum hold
    },
    '5s': {
        'resample_factor': 50,          # 100ms * 50 = 5s
        'bars_per_day': 3960,           # 198000 / 50
        'targets': [
            ('ret_30s', 6),             # 6 bars ahead at 5s = 30s
            ('ret_1min', 12),           # 12 bars ahead at 5s = 1min
            ('ret_5min', 60),           # 60 bars ahead at 5s = 5min
        ],
        'pnl_entry_pct': 80,
        'min_hold_bars': 6,             # 30s minimum hold
        'max_hold_bars': 60,            # 5min maximum hold
    },
    '30s': {
        'resample_factor': 300,         # 100ms * 300 = 30s
        'bars_per_day': 660,            # 198000 / 300
        'targets': [
            ('ret_5min', 10),           # 10 bars ahead at 30s = 5min
            ('ret_15min', 30),          # 30 bars ahead at 30s = 15min
        ],
        'pnl_entry_pct': 80,
        'min_hold_bars': 10,            # 5min minimum hold
        'max_hold_bars': 30,            # 15min maximum hold
    },
    '1min': {
        'resample_factor': 600,         # 100ms * 600 = 60s = 1min
        'bars_per_day': 330,            # 198000 / 600
        'targets': [
            ('ret_5min', 5),            # 5 bars ahead at 1min = 5min
            ('ret_15min', 15),          # 15 bars ahead at 1min = 15min
            ('ret_30min', 30),          # 30 bars ahead at 1min = 30min
        ],
        'pnl_entry_pct': 80,
        'min_hold_bars': 5,             # 5min minimum hold
        'max_hold_bars': 30,            # 30min maximum hold
    },
    '5min': {
        'resample_factor': 3000,        # 100ms * 3000 = 300s = 5min
        'bars_per_day': 66,             # 198000 / 3000
        'targets': [
            ('ret_15min', 3),           # 3 bars ahead at 5min = 15min
            ('ret_30min', 6),           # 6 bars ahead at 5min = 30min
        ],
        'pnl_entry_pct': 80,
        'min_hold_bars': 3,             # 15min minimum hold
        'max_hold_bars': 6,             # 30min maximum hold
    },
}


# ============================================================================
# RESAMPLING
# ============================================================================

def resample_features_and_prices(
    features: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    resample_factor: int,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Resample 100ms features to a lower frequency using day-aligned windows.

    Each day must have exactly BARS_PER_DAY_100MS bars. We resample each day
    independently so day boundaries remain clean multiples of resample_factor.

    For each window of `resample_factor` bars:
    - SNAPSHOT features: take the LAST value (book state at window end)
    - FLOW features: take the MEAN (average activity over the window)

    Returns:
        features_rs: (n_resampled, n_features) float32
        mid_prices_rs: (n_resampled,) float32
        day_boundaries_rs: list of int (resampled bar indices)
    """
    n_days = len(day_boundaries) - 1
    n_features = features.shape[1]

    # Determine which features are flow vs snapshot
    flow_mask = np.array([is_flow_feature(fn) for fn in feature_names], dtype=bool)
    n_flow = flow_mask.sum()
    n_snap = (~flow_mask).sum()
    logger.info(f"  Resampling: {n_snap} snapshot features (last), {n_flow} flow features (mean)")

    # Verify day sizes are divisible by resample_factor
    bars_per_day_rs = BARS_PER_DAY_100MS // resample_factor
    remainder = BARS_PER_DAY_100MS % resample_factor
    if remainder != 0:
        raise ValueError(
            f"BARS_PER_DAY_100MS={BARS_PER_DAY_100MS} is not divisible by "
            f"resample_factor={resample_factor} (remainder={remainder})"
        )

    # Pre-allocate output arrays
    total_rs = n_days * bars_per_day_rs
    features_rs = np.empty((total_rs, n_features), dtype=np.float32)
    mid_prices_rs = np.empty(total_rs, dtype=np.float32)

    for d in range(n_days):
        day_start = day_boundaries[d]
        day_end = day_boundaries[d + 1]
        day_len = day_end - day_start

        # Handle days that might be slightly shorter than expected
        n_complete = day_len // resample_factor
        usable = n_complete * resample_factor

        out_start = d * bars_per_day_rs
        out_end = out_start + n_complete

        if n_complete == 0:
            continue

        # Reshape: (n_complete, resample_factor, n_features)
        feat_day = features[day_start:day_start + usable]  # (usable, n_features)
        feat_reshaped = feat_day.reshape(n_complete, resample_factor, n_features)

        # Snapshot features: last bar of each window
        feat_snap = feat_reshaped[:, -1, :]  # (n_complete, n_features)

        # Flow features: mean over window
        feat_flow = feat_reshaped.mean(axis=1)  # (n_complete, n_features)

        # Combine: use snapshot for non-flow, mean for flow
        features_rs[out_start:out_end] = np.where(
            flow_mask[np.newaxis, :],
            feat_flow,
            feat_snap,
        )

        # Mid prices: take the last price of each window (closing price)
        price_day = mid_prices[day_start:day_start + usable]
        features_rs[out_start:out_end, :]  # already written above
        mid_prices_rs[out_start:out_end] = price_day[resample_factor - 1::resample_factor][:n_complete]

        # Zero-fill any unused slots (if n_complete < bars_per_day_rs due to short days)
        if n_complete < bars_per_day_rs:
            # Fill remaining with NaN to avoid contamination
            features_rs[out_end:out_start + bars_per_day_rs] = np.nan
            mid_prices_rs[out_end:out_start + bars_per_day_rs] = np.nan

    # Build resampled day boundaries
    day_boundaries_rs = [d * bars_per_day_rs for d in range(n_days + 1)]

    logger.info(f"  Resampled: {total_rs:,} bars ({n_days} days x {bars_per_day_rs} bars/day)")
    return features_rs, mid_prices_rs, day_boundaries_rs


def compute_targets_at_freq(
    mid_prices_rs: np.ndarray,
    day_boundaries_rs: list,
    target_specs: List[Tuple[str, int]],
) -> Dict[str, np.ndarray]:
    """
    Compute forward return targets at the resampled frequency.

    target_specs: list of (target_name, horizon_bars)
    where horizon_bars is the number of RESAMPLED bars ahead to predict.

    Returns dict of target_name -> np.ndarray of log returns (with NaN at boundaries).
    """
    N = len(mid_prices_rs)
    n_days = len(day_boundaries_rs) - 1
    log_mid = np.log(np.maximum(mid_prices_rs, 1.0))

    targets = {}
    for target_name, horizon_bars in target_specs:
        if horizon_bars >= N:
            logger.warning(f"  Horizon {horizon_bars} bars >= N={N}, skipping {target_name}")
            continue

        ret = np.empty(N, dtype=np.float32)
        ret[:N - horizon_bars] = (log_mid[horizon_bars:] - log_mid[:N - horizon_bars]).astype(np.float32)
        ret[N - horizon_bars:] = np.nan

        # NaN-fill bars within `horizon_bars` of each day boundary (no overnight leakage)
        if n_days > 1:
            for d in range(n_days - 1):
                day_end = day_boundaries_rs[d + 1]
                nan_start = max(day_boundaries_rs[d], day_end - horizon_bars)
                ret[nan_start:day_end] = np.nan

        valid_count = np.isfinite(ret).sum()
        logger.info(f"  target {target_name}: horizon={horizon_bars} bars, valid={valid_count:,}")
        targets[target_name] = ret

    return targets


# ============================================================================
# WALK-FORWARD (LightGBM) AT RESAMPLED FREQUENCY
# ============================================================================

def walk_forward_lgbm_resample(
    features_rs: np.ndarray,
    target_rs: np.ndarray,
    day_boundaries_rs: list,
    feature_names: List[str],
    min_train_days: int = 3,
) -> dict:
    """
    Walk-forward LightGBM on resampled data.

    Same parameters as baseline 100ms model:
    - 500 trees, depth=6, lr=0.03
    - Same feature exclusion list (EXCLUDE_FEATURES_DIRECTION)
    - Expanding window, 1-day purge gap, test on next day

    Returns dict with IC, fold_ics, predictions array aligned to resampled bars, etc.
    """
    import lightgbm as lgb

    n_days = len(day_boundaries_rs) - 1
    if n_days < min_train_days + 1:
        return {'error': f'Need {min_train_days + 1} days, have {n_days}'}

    # Feature exclusion
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in feature_names], dtype=bool)
    X = features_rs[:, keep_mask]
    names_kept = [fn for fn in feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]
    n_features_used = len(names_kept)

    logger.info(f"    Features: {n_features_used} used, {(~keep_mask).sum()} excluded")

    lgbm_params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': max(20, min(100, day_boundaries_rs[1] // 50)),
        'verbose': -1,
        'n_jobs': -1,
        'objective': 'regression',
        'metric': 'rmse',
    }

    all_preds = []
    all_actuals = []
    all_indices = []
    fold_ics = []
    feature_importance = np.zeros(n_features_used)

    for test_day in range(min_train_days, n_days):
        train_start = day_boundaries_rs[0]
        train_end = day_boundaries_rs[test_day]      # 1-day purge: skip day test_day-1
        test_start = day_boundaries_rs[test_day]
        test_end = day_boundaries_rs[test_day + 1]

        X_train = X[train_start:train_end]
        y_train = target_rs[train_start:train_end]
        X_test = X[test_start:test_end]
        y_test = target_rs[test_start:test_end]

        train_valid = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)
        test_valid = np.isfinite(y_test) & np.all(np.isfinite(X_test), axis=1)

        # At low frequencies, fewer bars per day — relax thresholds
        min_train = max(50, len(X_train) // 10)
        min_test = max(10, len(X_test) // 5)

        if train_valid.sum() < min_train or test_valid.sum() < min_test:
            logger.debug(f"    Day {test_day}: train={train_valid.sum()}, test={test_valid.sum()} — skipping")
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
            if hasattr(model, 'feature_importances_'):
                feature_importance += model.feature_importances_
        except Exception as e:
            logger.warning(f"    Day {test_day} training failed: {e}")
            continue

        all_preds.append(preds)
        all_actuals.append(y_te)

        bar_indices = np.arange(test_start, test_end)[test_valid]
        all_indices.append(bar_indices)

        if len(preds) > 5:
            try:
                ic_fold = spearmanr(preds, y_te)[0]
                if np.isfinite(ic_fold):
                    fold_ics.append(float(ic_fold))
            except Exception:
                pass

        del model
        gc.collect()

    if not all_preds:
        return {'error': 'No valid folds'}

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    pred_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int64)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]
    pi = pred_indices[valid]

    if len(p) < 20:
        return {'error': f'Too few valid predictions: {len(p)}'}

    ic = float(spearmanr(p, a)[0])

    if len(fold_ics) > 2:
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
    else:
        ic_mean, ic_std, icir, tstat = ic, 0.0, 0.0, 0.0

    n_positive = sum(1 for x in fold_ics if x > 0)
    fold_consistency = n_positive / len(fold_ics) if fold_ics else 0.0

    # Top features
    top_k = min(10, n_features_used)
    top_idx = np.argsort(feature_importance)[-top_k:][::-1]
    top_features = [(names_kept[i], float(feature_importance[i])) for i in top_idx if feature_importance[i] > 0]

    return {
        'ic': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'tstat': tstat,
        'fold_ics': [float(x) for x in fold_ics],
        'fold_consistency': fold_consistency,
        'n_folds': len(fold_ics),
        'n_preds': int(len(p)),
        'predictions': predictions,      # raw (may include NaN)
        'actuals': actuals,
        'pred_indices': pi,              # bar indices in the RESAMPLED space
        'top_features': top_features,
    }


# ============================================================================
# PNL SIMULATION AT RESAMPLED FREQUENCY
# ============================================================================

def simulate_pnl_at_freq(
    predictions_rs: np.ndarray,       # predictions in resampled bar space
    pred_indices: np.ndarray,          # which resampled bars have predictions
    mid_prices_rs: np.ndarray,         # resampled mid prices
    day_boundaries_rs: list,
    n_bars_total: int,
    entry_pct: float = 80.0,
    min_hold_bars: int = 5,
    max_hold_bars: int = 60,
) -> dict:
    """
    Simulate trading at the resampled frequency.

    Builds a full prediction array aligned to resampled bar indices,
    then calls simulate_realistic_trading with $12.50 round-trip cost.
    """
    n_rs = len(mid_prices_rs)

    # Build full-length predictions array aligned to resampled bars
    full_pred = np.full(n_rs, np.nan, dtype=np.float64)
    valid_preds = np.isfinite(predictions_rs)
    valid_indices = pred_indices[valid_preds]

    # Clamp indices to valid range
    in_range = valid_indices < n_rs
    full_pred[valid_indices[in_range]] = predictions_rs[valid_preds][in_range]

    n_filled = np.isfinite(full_pred).sum()
    logger.info(f"    Prediction coverage: {n_filled:,}/{n_rs:,} resampled bars ({n_filled/n_rs:.1%})")

    # Run simulation
    result = simulate_realistic_trading(
        predictions=full_pred,
        mid_prices=mid_prices_rs,
        day_boundaries=day_boundaries_rs,
        entry_threshold_pct=entry_pct,
        min_hold_bars=min_hold_bars,
        max_hold_bars=max_hold_bars,
        exit_signal_reversal=True,
        round_trip_ticks=1.0,   # $12.50 RT cost (1 tick)
        contracts=1,
    )
    return result


# ============================================================================
# MAIN TEST LOOP
# ============================================================================

def run_low_freq_test(
    scanner: MBOAlphaScanner,
    freqs_to_test: List[str],
    min_train_days: int = 3,
) -> List[dict]:
    """
    For each frequency in freqs_to_test:
    1. Resample features and prices
    2. Compute targets at resampled frequency
    3. Walk-forward LightGBM
    4. Simulate PnL
    5. Collect results
    """
    all_results = []

    for freq_name in freqs_to_test:
        if freq_name not in FREQ_CONFIGS:
            logger.warning(f"Unknown frequency: {freq_name}, skipping")
            continue

        cfg = FREQ_CONFIGS[freq_name]
        resample_factor = cfg['resample_factor']
        target_specs = cfg['targets']

        logger.info(f"\n{'='*70}")
        logger.info(f"FREQUENCY: {freq_name} (resample_factor={resample_factor}x)")
        logger.info(f"  Targets: {[t[0] for t in target_specs]}")
        logger.info(f"  Expected: {cfg['bars_per_day']} bars/day")
        logger.info(f"{'='*70}")

        t_resample = time.time()

        # Step 1: Resample
        try:
            features_rs, mid_prices_rs, day_boundaries_rs = resample_features_and_prices(
                features=scanner.features,
                mid_prices=scanner.mid_prices,
                day_boundaries=scanner.day_boundaries,
                feature_names=scanner.feature_names,
                resample_factor=resample_factor,
            )
        except Exception as e:
            logger.error(f"Resampling failed for {freq_name}: {e}")
            all_results.append({
                'freq': freq_name,
                'error': f'Resampling failed: {e}',
            })
            continue

        resample_time = time.time() - t_resample
        logger.info(f"  Resampled in {resample_time:.1f}s: {features_rs.shape}, {len(day_boundaries_rs)-1} days")

        # Step 2: Compute targets at this frequency
        logger.info(f"  Computing targets...")
        targets_rs = compute_targets_at_freq(
            mid_prices_rs=mid_prices_rs,
            day_boundaries_rs=day_boundaries_rs,
            target_specs=target_specs,
        )

        if not targets_rs:
            logger.warning(f"  No valid targets for {freq_name}")
            all_results.append({'freq': freq_name, 'error': 'No valid targets'})
            continue

        # Step 3: Walk-forward LightGBM for each target
        for target_name, target_arr in targets_rs.items():
            logger.info(f"\n  --- Target: {target_name} at {freq_name} ---")
            t_wf = time.time()

            wf_result = walk_forward_lgbm_resample(
                features_rs=features_rs,
                target_rs=target_arr,
                day_boundaries_rs=day_boundaries_rs,
                feature_names=scanner.feature_names,
                min_train_days=min_train_days,
            )

            wf_time = time.time() - t_wf

            if 'error' in wf_result:
                logger.warning(f"  Walk-forward failed: {wf_result['error']}")
                all_results.append({
                    'freq': freq_name,
                    'target': target_name,
                    'error': wf_result['error'],
                    'wf_elapsed_sec': wf_time,
                })
                continue

            logger.info(
                f"  IC={wf_result['ic']:.4f} ICIR={wf_result['icir']:.2f} "
                f"t={wf_result['tstat']:.2f} FoldC={wf_result['fold_consistency']:.0%} "
                f"({wf_result['n_folds']} folds, {wf_time:.0f}s)"
            )
            if wf_result.get('fold_ics'):
                ics_str = " ".join(f"{x:+.3f}" for x in wf_result['fold_ics'])
                logger.info(f"  Fold ICs: [{ics_str}]")
            if wf_result.get('top_features'):
                top3 = ", ".join(f"{n}" for n, v in wf_result['top_features'][:5])
                logger.info(f"  Top features: {top3}")

            # Step 4: PnL simulation
            logger.info(f"  Simulating PnL...")
            t_pnl = time.time()
            try:
                pnl = simulate_pnl_at_freq(
                    predictions_rs=wf_result['predictions'],
                    pred_indices=wf_result['pred_indices'],
                    mid_prices_rs=mid_prices_rs,
                    day_boundaries_rs=day_boundaries_rs,
                    n_bars_total=len(mid_prices_rs),
                    entry_pct=cfg['pnl_entry_pct'],
                    min_hold_bars=cfg['min_hold_bars'],
                    max_hold_bars=cfg['max_hold_bars'],
                )
            except Exception as e:
                logger.warning(f"  PnL simulation error: {e}")
                pnl = {'error': str(e)}

            pnl_time = time.time() - t_pnl

            if 'error' in pnl:
                logger.warning(f"  PnL failed: {pnl['error']}")
            else:
                logger.info(
                    f"  Trades: {pnl['n_trades']:,} ({pnl['trades_per_day']:.1f}/day), "
                    f"WinRate={pnl['win_rate']:.1%}, AvgHold={pnl['avg_hold_bars']:.1f} bars"
                )
                logger.info(
                    f"  Gross=${pnl['total_gross_pnl']:+,.0f}  "
                    f"Cost=${pnl['total_cost']:,.0f}  "
                    f"Net=${pnl['total_net_pnl']:+,.0f}"
                )
                logger.info(
                    f"  Daily net: ${pnl['daily_net_pnl']:+,.0f}  "
                    f"Sharpe={pnl['daily_sharpe']:.2f}  "
                    f"WinDays={pnl['pct_win_days']:.0%}"
                )
                logger.info(
                    f"  MaxDD=${pnl['max_drawdown']:,.0f}  "
                    f"PF={pnl['profit_factor']:.2f}"
                )

            # Compute gross IC signal (how much pnl before costs)
            gross_vs_cost_verdict = 'N/A'
            if 'error' not in pnl:
                if pnl['total_gross_pnl'] > pnl['total_cost']:
                    gross_vs_cost_verdict = 'GROSS > COST (potentially tradable)'
                else:
                    gross_vs_cost_verdict = f'GROSS < COST (costs dominate by ${pnl["total_cost"] - pnl["total_gross_pnl"]:,.0f})'
            logger.info(f"  VERDICT: {gross_vs_cost_verdict}")

            result = {
                'freq': freq_name,
                'resample_factor': resample_factor,
                'target': target_name,
                # IC metrics
                'ic': wf_result['ic'],
                'ic_std': wf_result['ic_std'],
                'icir': wf_result['icir'],
                'tstat': wf_result['tstat'],
                'fold_consistency': wf_result['fold_consistency'],
                'n_folds': wf_result['n_folds'],
                'fold_ics': wf_result['fold_ics'],
                'n_preds': wf_result['n_preds'],
                'top_features': wf_result.get('top_features', []),
                # PnL metrics
                'pnl': {k: v for k, v in pnl.items() if k != 'daily_pnl_values'} if 'error' not in pnl else pnl,
                'pnl_daily_values': pnl.get('daily_pnl_values', []) if 'error' not in pnl else [],
                # Meta
                'gross_vs_cost_verdict': gross_vs_cost_verdict,
                'wf_elapsed_sec': wf_time,
                'pnl_elapsed_sec': pnl_time,
            }
            all_results.append(result)

        # Free resampled data before next frequency
        del features_rs, mid_prices_rs
        gc.collect()
        logger.info(f"\n  Done with {freq_name}")

    return all_results


# ============================================================================
# REPORTING
# ============================================================================

def format_scoreboard(results: List[dict]) -> str:
    """Format all results as an ASCII table sorted by freq then target."""
    lines = [
        "",
        "LOW-FREQUENCY RESAMPLING TEST — SCOREBOARD",
        "=" * 130,
        f"{'Freq':<6s} {'Target':<12s} {'IC':>7s} {'ICIR':>6s} {'t':>6s} "
        f"{'FoldC%':>7s} {'Folds':>5s} "
        f"{'Trd/d':>6s} {'Gross$':>10s} {'Cost$':>9s} {'Net$':>10s} "
        f"{'Sharpe':>7s} {'WinD%':>6s} {'PF':>5s} {'Verdict'}",
        "-" * 130,
    ]

    freq_order = ['1s', '5s', '30s', '1min', '5min']
    sorted_results = sorted(
        results,
        key=lambda r: (freq_order.index(r['freq']) if r['freq'] in freq_order else 99,
                       r.get('target', ''))
    )

    for r in sorted_results:
        if 'error' in r and 'ic' not in r:
            lines.append(f"{r['freq']:<6s} {r.get('target', '?'):<12s}  ERROR: {r['error']}")
            continue

        pnl = r.get('pnl', {})
        if 'error' in pnl:
            pnl_str = f"{'N/A':>6s} {'N/A':>10s} {'N/A':>9s} {'N/A':>10s} {'N/A':>7s} {'N/A':>6s} {'N/A':>5s}"
        else:
            tpd = pnl.get('trades_per_day', 0)
            gross = pnl.get('total_gross_pnl', 0)
            cost = pnl.get('total_cost', 0)
            net = pnl.get('total_net_pnl', 0)
            sharpe = pnl.get('daily_sharpe', 0)
            wind = pnl.get('pct_win_days', 0)
            pf = pnl.get('profit_factor', 0)
            pnl_str = (
                f"{tpd:>6.1f} ${gross:>+9,.0f} ${cost:>8,.0f} ${net:>+9,.0f} "
                f"{sharpe:>7.2f} {wind:>5.0%} {pf:>5.2f}"
            )

        fc_pct = f"{r.get('fold_consistency', 0):.0%}"
        verdict = r.get('gross_vs_cost_verdict', 'N/A')
        # Shorten verdict for table
        if 'GROSS > COST' in verdict:
            verdict_short = 'TRADABLE'
        elif 'GROSS < COST' in verdict:
            verdict_short = 'UNTRADEBL'
        else:
            verdict_short = verdict[:10]

        lines.append(
            f"{r['freq']:<6s} {r.get('target', '?'):<12s} "
            f"{r.get('ic', 0):>7.4f} {r.get('icir', 0):>6.2f} {r.get('tstat', 0):>6.2f} "
            f"{fc_pct:>7s} {r.get('n_folds', 0):>5d} "
            f"{pnl_str} {verdict_short}"
        )

    lines.append("=" * 130)
    return "\n".join(lines)


def format_ic_progression(results: List[dict]) -> str:
    """Show how IC changes across frequencies for the same prediction horizon."""
    lines = [
        "",
        "IC PROGRESSION ACROSS FREQUENCIES",
        "=" * 80,
        "How does the direction signal evolve as we sample less frequently?",
        "-" * 80,
    ]

    # Group by approximate target horizon
    horizon_groups = {}
    for r in results:
        if 'error' in r and 'ic' not in r:
            continue
        target = r.get('target', '')
        freq = r.get('freq', '')
        ic = r.get('ic', 0.0)
        tstat = r.get('tstat', 0.0)
        pnl = r.get('pnl', {})
        net = pnl.get('total_net_pnl', None) if 'error' not in pnl else None
        key = target
        if key not in horizon_groups:
            horizon_groups[key] = []
        horizon_groups[key].append((freq, ic, tstat, net))

    for target, entries in sorted(horizon_groups.items()):
        lines.append(f"\nTarget: {target}")
        lines.append(f"  {'Freq':<8s} {'IC':>8s} {'t-stat':>8s} {'Net PnL':>12s}")
        for freq, ic, tstat, net in entries:
            net_str = f"${net:+,.0f}" if net is not None else "N/A"
            lines.append(f"  {freq:<8s} {ic:>8.4f} {tstat:>8.2f} {net_str:>12s}")

    return "\n".join(lines)


def format_tradability_analysis(results: List[dict]) -> str:
    """Show at which frequency the signal becomes tradable."""
    lines = [
        "",
        "TRADABILITY ANALYSIS",
        "=" * 80,
        "At what frequency does gross PnL exceed transaction costs?",
        f"Cost per trade: ${RT_COST:.2f} (1 tick RT)",
        "-" * 80,
    ]

    freq_order = ['1s', '5s', '30s', '1min', '5min']

    for freq in freq_order:
        freq_results = [r for r in results if r.get('freq') == freq and 'error' not in r.get('pnl', {'error': 'x'})]
        if not freq_results:
            continue

        lines.append(f"\n{freq} bars:")
        for r in freq_results:
            pnl = r['pnl']
            target = r.get('target', '?')
            ic = r.get('ic', 0)
            gross = pnl.get('total_gross_pnl', 0)
            cost = pnl.get('total_cost', 0)
            net = pnl.get('total_net_pnl', 0)
            tpd = pnl.get('trades_per_day', 0)
            sharpe = pnl.get('daily_sharpe', 0)
            n_days = pnl.get('n_trading_days', 0)

            tradable = gross > cost
            symbol = ">>>" if tradable else "---"

            lines.append(
                f"  {symbol} {target:<12s}: IC={ic:.4f}, "
                f"{tpd:.1f} trades/day, "
                f"Gross=${gross:+,.0f} vs Cost=${cost:,.0f} -> Net=${net:+,.0f}, "
                f"Sharpe={sharpe:.2f} ({n_days}d)"
            )

    lines.append("")
    lines.append("KEY: >>> = Gross > Cost (signal survives costs)")
    lines.append("     --- = Gross < Cost (costs dominate)")

    return "\n".join(lines)


def format_discord_summary(results: List[dict], elapsed: float) -> str:
    """Concise Discord message with key findings."""
    tradable = [r for r in results
                if 'error' not in r.get('pnl', {'error': 'x'})
                and r.get('pnl', {}).get('total_gross_pnl', 0) > r.get('pnl', {}).get('total_cost', 0)]

    lines = [
        "**LOW-FREQUENCY RESAMPLING TEST COMPLETE**",
        f"Elapsed: {elapsed/60:.1f}min | Configs tested: {len(results)}",
        "",
        "**KEY FINDINGS:**",
    ]

    if tradable:
        lines.append(f"**{len(tradable)} frequency/target combos where Gross > Cost:**")
        for r in sorted(tradable, key=lambda x: x.get('pnl', {}).get('daily_sharpe', 0), reverse=True):
            pnl = r['pnl']
            lines.append(
                f"- `{r['freq']} / {r['target']}`: IC={r['ic']:.4f}, "
                f"Trades/day={pnl['trades_per_day']:.1f}, "
                f"Net=${pnl['total_net_pnl']:+,.0f}, "
                f"Sharpe={pnl['daily_sharpe']:.2f}"
            )
    else:
        lines.append("No frequency/target combo where Gross PnL > Transaction Costs")
        # Show best by IC
        valid = [r for r in results if 'ic' in r and 'error' not in r]
        if valid:
            best = max(valid, key=lambda x: x.get('ic', 0))
            lines.append(f"Best IC: `{best['freq']} / {best['target']}` IC={best['ic']:.4f}, t={best['tstat']:.2f}")

    lines.append("")
    lines.append("**IC by Frequency (best target per freq):**")
    lines.append("```")
    freq_order = ['1s', '5s', '30s', '1min', '5min']
    for freq in freq_order:
        freq_r = [r for r in results if r.get('freq') == freq and 'ic' in r]
        if freq_r:
            best = max(freq_r, key=lambda x: x.get('ic', 0))
            pnl = best.get('pnl', {})
            net_str = f"${pnl.get('total_net_pnl', 0):+,.0f}" if 'error' not in pnl else "N/A"
            lines.append(
                f"{freq:<6s} IC={best['ic']:.4f} t={best['tstat']:.2f} "
                f"Net={net_str}"
            )
    lines.append("```")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Low-frequency resampling test for ES direction strategy')
    parser.add_argument(
        '--freqs', nargs='+',
        default=['1s', '5s', '30s', '1min', '5min'],
        help='Frequencies to test (subset of: 1s 5s 30s 1min 5min)'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: only 1s and 5min, fewer train days required')
    parser.add_argument('--min-train-days', type=int, default=3,
                        help='Minimum training days before first test fold')
    args = parser.parse_args()

    if args.fast:
        freqs = ['1s', '5min']
        min_train_days = 3
        logger.info("FAST MODE: testing only 1s and 5min")
    else:
        freqs = args.freqs
        min_train_days = args.min_train_days

    logger.info("=" * 75)
    logger.info("LOW-FREQUENCY RESAMPLING TEST")
    logger.info(f"  Frequencies: {freqs}")
    logger.info(f"  Min train days: {min_train_days}")
    logger.info(f"  Excluded features: {len(EXCLUDE_FEATURES_DIRECTION)}")
    logger.info(f"  ES RT cost: ${RT_COST:.2f} (1 tick)")
    logger.info(f"  Base: {BARS_PER_DAY_100MS:,} bars/day at 100ms")
    logger.info("=" * 75)

    t_start = time.time()

    # Load feature cache
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.error("No feature cache found. Run the base scan first to build the cache.")
        sys.exit(1)

    n_bars = scanner.features.shape[0]
    n_days = len(scanner.day_boundaries) - 1
    bars_per_day_actual = n_bars // n_days

    logger.info(f"Data: {n_bars:,} bars, {n_days} days, {bars_per_day_actual:,} bars/day")
    logger.info(f"Features: {len(scanner.feature_names)}")
    logger.info(f"Day boundaries: {scanner.day_boundaries[:5]} ...")

    # Verify day size
    if bars_per_day_actual != BARS_PER_DAY_100MS:
        logger.warning(
            f"Expected {BARS_PER_DAY_100MS} bars/day, got {bars_per_day_actual}. "
            f"Resampling will use actual day boundaries from scanner."
        )

    # Validate that all requested frequencies divide evenly into actual bars per day
    for freq in freqs:
        cfg = FREQ_CONFIGS[freq]
        rf = cfg['resample_factor']
        # Check each day
        problems = []
        for d in range(n_days):
            day_len = scanner.day_boundaries[d + 1] - scanner.day_boundaries[d]
            if day_len % rf != 0:
                problems.append(f"Day {d}: {day_len} bars, not divisible by {rf}")
        if problems:
            logger.warning(
                f"Frequency {freq} (resample_factor={rf}) has non-divisible days: "
                f"{problems[:3]}{'...' if len(problems)>3 else ''}. "
                f"Will use floor division (drop trailing bars)."
            )

    # Run the test
    results = run_low_freq_test(
        scanner=scanner,
        freqs_to_test=freqs,
        min_train_days=min_train_days,
    )

    elapsed = time.time() - t_start

    # Format and print reports
    scoreboard = format_scoreboard(results)
    ic_progression = format_ic_progression(results)
    tradability = format_tradability_analysis(results)
    discord_msg = format_discord_summary(results, elapsed)

    logger.info("\n" + scoreboard)
    logger.info("\n" + ic_progression)
    logger.info("\n" + tradability)

    print("\n" + "=" * 75)
    print(scoreboard)
    print(ic_progression)
    print(tradability)
    print("=" * 75)
    print("\n--- DISCORD SUMMARY ---")
    print(discord_msg)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f"low_freq_test_{timestamp}.json"

    serializable_results = []
    for r in results:
        sr = dict(r)
        sr['top_features'] = [(n, float(v)) for n, v in r.get('top_features', [])]
        # Convert numpy types in pnl
        if 'pnl' in sr and isinstance(sr['pnl'], dict):
            sr['pnl'] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                         for k, v in sr['pnl'].items()}
        serializable_results.append(sr)

    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'freqs_tested': freqs,
            'min_train_days': min_train_days,
            'n_bars_total': n_bars,
            'n_days': n_days,
            'bars_per_day_100ms': BARS_PER_DAY_100MS,
            'rt_cost_usd': RT_COST,
            'results': serializable_results,
            'scoreboard': scoreboard,
            'ic_progression': ic_progression,
            'tradability': tradability,
            'elapsed_sec': elapsed,
        }, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {out_path}")
    logger.info(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    return results


if __name__ == '__main__':
    main()
