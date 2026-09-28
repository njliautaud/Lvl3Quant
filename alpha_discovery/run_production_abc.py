"""
Production Model ABC — Clean Features + Multi-Horizon + Vol Gate + Smart Execution
===================================================================================

Implements all three optimization paths simultaneously:

  Option A: MULTI-HORIZON (Lower Frequency)
    Instead of 100ms trading (edge < costs), test 3s, 5s, 10s, 30s targets.
    Same features, longer-horizon targets → fewer trades, larger moves per trade.

  Option B: VOL-GATED TRADING
    Only trade during high-volatility periods (IC=0.184 on active days vs ~0 on dead).
    Intraday vol gate: realized vol of past 200 bars > rolling median.
    Day-level vol gate: only trade on active days (intraday range > threshold).

  Option C: SMART EXECUTION
    Strong signals (top 5%) → market order (guaranteed fill, pay spread)
    Medium signals (top 10-20%) → limit order (earn spread if filled)
    Combines limit entry + hybrid exit (stop/TP/limit exit/timeout)

All paths use the CLEAN 18-feature set (forward-dominant, no retrodiction).

Usage:
    python alpha_discovery/run_production_abc.py
    python alpha_discovery/run_production_abc.py --fast
    python alpha_discovery/run_production_abc.py --horizons 3s 5s
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

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache, compute_return_targets, EXCLUDE_FEATURES_DIRECTION,
)

# ============================================================================
# LOGGING (ASCII-safe for Windows)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / 'production_abc.log'),
                            mode='a', encoding='utf-8'),
    ]
)
log = logging.getLogger('production_abc')

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE       = 0.25
TICK_VALUE      = 12.50
ES_POINT_VALUE  = 50
BARS_PER_SEC    = 10       # 100ms bars
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)     # Round-trip (AMP + Rithmic + CME)
HALF_TICK       = TICK_SIZE / 2  # 0.125
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks

# ============================================================================
# CLEAN FEATURE SET (from forward decomposition analysis)
# ============================================================================
# These 18 features have fwd/bwd IC ratio >= 0.5 (forward-dominant).
# Orthogonalization proved removing backward info retains 106.3% of IC.
CLEAN_FEATURES = [
    'ofi_5',              # Order Flow Imbalance 5-bar (strongest dynamic feature)
    'total_ask_vol',      # Total ask volume (book state)
    'bid_L5_conc',        # Bid level 5 concentration
    'ask_L1_conc',        # Ask level 1 concentration (strongest single IC=0.091)
    'ask_L4_orders',      # Ask level 4 order count
    'ofi_20',             # OFI 20-bar
    'vol_regime',         # Volatility regime (rvol_20/rvol_100)
    'depth_ratio_l1',     # L1 depth ratio (bid/(bid+ask) at L1)
    'depth_concentration',# Overall depth concentration
    'bid_L1_orders',      # Bid L1 order count
    'bid_L3_conc',        # Bid L3 concentration
    'total_bid_vol',      # Total bid volume
    'ask_L1_orders',      # Ask L1 order count
    'bid_L1_conc',        # Bid L1 concentration
    'ofi_50',             # OFI 50-bar
    'bid_L2_conc',        # Bid L2 concentration
    'ask_slope',          # Ask side slope (price sensitivity)
    'bid_pressure',       # Bid pressure (volume-weighted)
]

# Backward-dominant features to DROP (retrodiction contaminated)
BACKWARD_FEATURES_DROP = [
    'ret_100',            # Backward IC=0.532, Forward IC=0.009 (biggest offender)
    'ask_L2_orders', 'bid_L2_orders', 'bid_L4_orders',
    'ask_L3_orders', 'bid_L5_orders', 'ask_L5_orders',
    'depth_ratio_l3', 'depth_ratio_l5',
    'ask_L5_conc', 'bid_L4_conc', 'bid_slope',
]

# Combined exclusion: standard direction exclusions + backward-dominant
ALL_EXCLUDE = list(set(EXCLUDE_FEATURES_DIRECTION + BACKWARD_FEATURES_DROP))


# ============================================================================
# VOL GATE (Option B)
# ============================================================================
def compute_vol_gate(
    mid_prices: np.ndarray,
    day_boundaries: list,
    lookback_bars: int = 200,
    intraday_quantile: float = 0.60,
) -> np.ndarray:
    """
    Compute bar-level vol gate mask.

    Returns boolean array: True = high-vol (trade), False = low-vol (skip).

    Two layers:
    1. Intraday: realized vol of past `lookback_bars` > rolling median
    2. Day-level: day range must exceed minimum threshold
    """
    N = len(mid_prices)
    log_mid = np.log(np.maximum(mid_prices, 1.0))
    log_ret = np.diff(log_mid, prepend=log_mid[0])

    # Compute rolling realized vol
    # Using cumsum trick for rolling std
    lr64 = log_ret.astype(np.float64)
    cs = np.cumsum(lr64)
    cs2 = np.cumsum(lr64 ** 2)
    cs_pad = np.concatenate([[0.0], cs])
    cs2_pad = np.concatenate([[0.0], cs2])

    rvol = np.full(N, np.nan, dtype=np.float64)
    w = lookback_bars
    n_valid = N - w
    if n_valid > 0:
        s = cs_pad[w:w + n_valid] - cs_pad[:n_valid]
        s2 = cs2_pad[w:w + n_valid] - cs2_pad[:n_valid]
        var = s2 / w - (s / w) ** 2
        var = np.maximum(var, 0.0)
        rvol[w:] = np.sqrt(var)

    # Compute rolling median of rvol (using expanding quantile for speed)
    # Approximate: use per-day median as threshold
    n_days = len(day_boundaries) - 1
    gate = np.zeros(N, dtype=bool)

    for d in range(n_days):
        ds, de = day_boundaries[d], day_boundaries[d + 1]
        day_rvol = rvol[ds:de]
        valid_rvol = day_rvol[np.isfinite(day_rvol)]

        if len(valid_rvol) < 100:
            continue

        # Day-level gate: skip dead days (range < 0.06% ~ 4 ticks on $5700)
        day_range = mid_prices[ds:de].max() - mid_prices[ds:de].min()
        day_range_pct = day_range / np.mean(mid_prices[ds:de]) * 100
        if day_range_pct < 0.06:
            log.info(f"  Day {d}: DEAD (range={day_range_pct:.3f}%), skipping entirely")
            continue

        # Intraday gate: only trade when local vol > quantile threshold
        thresh = np.percentile(valid_rvol, intraday_quantile * 100)
        day_gate = day_rvol > thresh
        day_gate[:w] = False  # Warmup period
        gate[ds:de] = day_gate

    pct_active = gate.sum() / N * 100
    log.info(f"  Vol gate: {gate.sum():,}/{N:,} bars active ({pct_active:.1f}%)")
    return gate


# ============================================================================
# WALK-FORWARD WITH CLEAN FEATURES (Option A + B)
# ============================================================================
def walk_forward_clean(
    scanner: MBOAlphaScanner,
    target: np.ndarray,
    target_name: str,
    vol_gate: np.ndarray = None,
    min_train_days: int = 3,
    subsample_every: int = 1,
    lgbm_params: dict = None,
) -> dict:
    """
    Walk-forward LightGBM using ONLY clean features.

    Args:
        scanner: loaded MBOAlphaScanner with features
        target: prediction target array
        target_name: label (e.g., 'ret_5s')
        vol_gate: boolean mask, True = active bars (Option B)
        min_train_days: minimum training days before first prediction
        subsample_every: predict every Nth bar (reduces trade frequency)
        lgbm_params: LightGBM parameters override

    Returns dict with predictions, IC, fold metrics, feature importances.
    """
    import lightgbm as lgb

    feature_names = scanner.feature_names
    # Build keep mask: start with all features, then exclude
    keep_mask = np.array([fn not in ALL_EXCLUDE for fn in feature_names])

    # Further restrict to ONLY clean features if they exist in the feature set
    if CLEAN_FEATURES:
        clean_set = set(CLEAN_FEATURES)
        for i, fn in enumerate(feature_names):
            if keep_mask[i] and fn not in clean_set:
                keep_mask[i] = False

    features_use = scanner.features[:, keep_mask]
    feature_names_use = [fn for fn in feature_names if fn in set(CLEAN_FEATURES)]

    # Validate we actually have the clean features
    found = set(feature_names_use)
    missing = set(CLEAN_FEATURES) - found
    if missing:
        log.warning(f"  Missing clean features (will use what's available): {missing}")
    log.info(f"  Using {len(feature_names_use)} clean features: {feature_names_use[:5]}...")

    if lgbm_params is None:
        lgbm_params = {
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
            'objective': 'regression',
            'metric': 'rmse',
        }

    n_days = len(scanner.day_boundaries) - 1
    all_preds = []
    all_actuals = []
    all_indices = []
    fold_ics = []
    fold_details = []
    feature_importance = np.zeros(features_use.shape[1])

    for test_day in range(min_train_days, n_days):
        # Expanding window: train on all days up to test_day - 1 (1-day purge)
        train_end_day = test_day - 1
        train_start = scanner.day_boundaries[0]
        train_end = scanner.day_boundaries[train_end_day + 1]
        test_start = scanner.day_boundaries[test_day]
        test_end = scanner.day_boundaries[test_day + 1]

        X_train = features_use[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features_use[test_start:test_end]
        y_test = target[test_start:test_end]

        # Valid targets (not NaN)
        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        # Apply vol gate to training data too (train on active periods)
        if vol_gate is not None:
            train_valid &= vol_gate[train_start:train_end]
            test_valid &= vol_gate[test_start:test_end]

        if train_valid.sum() < 500 or test_valid.sum() < 50:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        # Subsample test predictions (Option A: lower frequency)
        if subsample_every > 1:
            idx = np.arange(0, len(X_te), subsample_every)
            X_te = X_te[idx]
            y_te = y_te[idx]
            # Adjust indices for the subsampled test set
            test_bar_indices = np.where(test_valid[test_start - test_start:test_end - test_start])[0]
            if subsample_every > 1 and len(test_bar_indices) > 0:
                test_bar_indices = test_bar_indices[::subsample_every]

        # 80/20 internal split for early stopping
        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**lgbm_params)
            model.fit(
                X_tr[:split], y_tr[:split],
                eval_set=[(X_tr[split:], y_tr[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception as e:
            log.warning(f"  Training failed day {test_day}: {e}")
            continue

        preds = model.predict(X_te)

        # Store predictions with global bar indices
        test_bars_global = np.where(test_valid)[0] + test_start
        if subsample_every > 1:
            test_bars_global = test_bars_global[::subsample_every]
        n_match = min(len(preds), len(test_bars_global))

        all_preds.append(preds[:n_match])
        all_actuals.append(y_te[:n_match])
        all_indices.append(test_bars_global[:n_match])

        # Per-fold IC
        if len(preds) > 10:
            try:
                ic_fold = float(spearmanr(preds[:n_match], y_te[:n_match])[0])
                if np.isfinite(ic_fold):
                    fold_ics.append(ic_fold)
                    fold_details.append({
                        'day': test_day,
                        'ic': ic_fold,
                        'n_samples': n_match,
                        'n_train': int(train_valid.sum()),
                        'hit_rate': float((np.sign(preds[:n_match]) == np.sign(y_te[:n_match])).mean()),
                    })
            except Exception:
                pass

        if hasattr(model, 'feature_importances_'):
            feature_importance += model.feature_importances_

        del model
        gc.collect()

    if not all_preds:
        return {'error': 'No valid predictions', 'target': target_name}

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    indices = np.concatenate(all_indices)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]

    if len(p) < 50:
        return {'error': f'Too few predictions: {len(p)}', 'target': target_name}

    # Compute metrics
    ic = float(spearmanr(p, a)[0])
    hr = float((np.sign(p) == np.sign(a)).mean())

    if len(fold_ics) > 2:
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
        try:
            _, pvalue = ttest_1samp(fold_ics, 0)
            pvalue = float(pvalue)
        except Exception:
            pvalue = 1.0
        n_positive_folds = sum(1 for x in fold_ics if x > 0)
        fold_consistency = n_positive_folds / len(fold_ics) if fold_ics else 0
    else:
        ic_mean, ic_std = ic, 0.0
        icir, tstat, pvalue = 0.0, 0.0, 1.0
        fold_consistency = 0.0

    # Top features
    top_idx = np.argsort(feature_importance)[::-1][:10]
    top_features = [(feature_names_use[i], float(feature_importance[i]))
                    for i in top_idx if i < len(feature_names_use)
                    and feature_importance[i] > 0]

    return {
        'target': target_name,
        'n_clean_features': len(feature_names_use),
        'ic': ic,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'tstat': tstat,
        'pvalue': pvalue,
        'hit_rate': hr,
        'n_predictions': len(p),
        'n_folds': len(fold_ics),
        'fold_consistency': fold_consistency,
        'fold_ics': [float(x) for x in fold_ics],
        'fold_details': fold_details,
        'top_features': top_features,
        'predictions': predictions[valid],
        'pred_indices': indices[valid].astype(np.int64),
        'subsample_every': subsample_every,
    }


# ============================================================================
# SMART EXECUTION SIMULATOR (Option C)
# ============================================================================
def simulate_smart_execution(
    mid_prices: np.ndarray,
    day_boundaries: list,
    predictions: np.ndarray,
    pred_indices: np.ndarray,
    vol_gate: np.ndarray = None,
    # Signal thresholds for smart routing
    strong_quantile: float = 0.95,   # Top 5% → market order
    medium_quantile: float = 0.80,   # Top 20% → limit order
    # Timing
    latency_bars: int = 1,
    fill_horizon_bars: int = 50,     # 5s for limit entry fill
    hold_horizon_bars: int = 100,    # 10s max hold
    # Exit
    stop_loss_ticks: float = 3.0,
    take_profit_ticks: float = 3.0,
    label: str = '',
) -> dict:
    """
    Smart execution: routes strong signals to market orders, medium to limit orders.

    Strong signal (top strong_quantile): Market order entry.
      - Pays half-spread (entry_edge = -0.5 ticks)
      - But guaranteed fill + directional edge

    Medium signal (top medium_quantile, below strong): Limit order entry.
      - Earns half-spread (entry_edge = +0.5 ticks)
      - But may not fill (adverse selection risk)

    Exit for all: hybrid (limit exit + stop loss + take profit + timeout)
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    best_bid = mid_prices - HALF_TICK
    best_ask = mid_prices + HALF_TICK

    # Map predictions to full time axis
    pred_signal = np.full(N, np.nan)
    ok = (pred_indices >= 0) & (pred_indices < N)
    pred_signal[pred_indices[ok]] = predictions[ok]

    # Compute signal thresholds
    valid_p = predictions[np.isfinite(predictions)]
    if len(valid_p) == 0:
        return {'error': 'No valid predictions'}

    strong_thresh_pos = float(np.percentile(valid_p, strong_quantile * 100))
    strong_thresh_neg = float(np.percentile(valid_p, (1 - strong_quantile) * 100))
    medium_thresh_pos = float(np.percentile(valid_p, medium_quantile * 100))
    medium_thresh_neg = float(np.percentile(valid_p, (1 - medium_quantile) * 100))

    trades = []
    n_market_posted = 0
    n_limit_posted = 0
    n_limit_not_filled = 0
    day_pnls = {}

    for d in range(n_days):
        ds, de = day_boundaries[d], day_boundaries[d + 1]
        min_needed = latency_bars + fill_horizon_bars + hold_horizon_bars + 5
        if de - ds < min_needed:
            continue

        m = mid_prices[ds:de]
        ps = pred_signal[ds:de]
        L = len(m)
        last_exit = -1
        day_pnls[d] = []
        max_start = L - latency_bars - fill_horizon_bars - hold_horizon_bars - 3

        # Vol gate for this day
        if vol_gate is not None:
            day_gate = vol_gate[ds:de]
        else:
            day_gate = np.ones(L, dtype=bool)

        i = 0
        while i < max_start:
            if i <= last_exit:
                i += 1
                continue

            sig = ps[i]
            if not np.isfinite(sig) or not day_gate[i]:
                i += 1
                continue

            # Classify signal strength and direction
            is_strong = False
            if sig > strong_thresh_pos:
                direction = 1
                is_strong = True
            elif sig < strong_thresh_neg:
                direction = -1
                is_strong = True
            elif sig > medium_thresh_pos:
                direction = 1
            elif sig < medium_thresh_neg:
                direction = -1
            else:
                i += 1
                continue

            post_bar = i + latency_bars
            if post_bar >= max_start:
                i += 1
                continue

            if not (np.isfinite(m[post_bar])):
                i += 1
                continue

            # ---- ENTRY ROUTING ----
            if is_strong:
                # MARKET ORDER: immediate fill at mid + half_tick cost
                n_market_posted += 1
                fill_bar = post_bar
                fill_mid = m[fill_bar]
                entry_edge = -0.5  # Market order PAYS half-spread
                entry_type = 'market'
            else:
                # LIMIT ORDER: post at best bid/ask, wait for fill
                n_limit_posted += 1
                if direction == 1:
                    entry_lim = m[post_bar] - HALF_TICK  # buy at bid
                else:
                    entry_lim = m[post_bar] + HALF_TICK  # sell at ask

                fill_end = min(post_bar + fill_horizon_bars + 1, L)
                future_m = m[post_bar + 1:fill_end]
                if direction == 1:
                    touch = np.where(future_m <= entry_lim)[0]
                else:
                    touch = np.where(future_m >= entry_lim)[0]

                if len(touch) == 0:
                    n_limit_not_filled += 1
                    i += 1
                    continue

                fill_bar = post_bar + touch[0] + 1
                fill_mid = m[fill_bar]
                entry_edge = 0.5  # Limit order EARNS half-spread
                entry_type = 'limit'

            # ---- EXIT (hybrid for all) ----
            exit_limit_price = (fill_mid + HALF_TICK) if direction == 1 else (fill_mid - HALF_TICK)
            exit_type = 'timeout'
            exit_bar = fill_bar
            max_hold = min(fill_bar + hold_horizon_bars, L - 1)

            for j in range(fill_bar + 1, max_hold + 1):
                curr_mid = m[j]
                unrealized_ticks = (curr_mid - fill_mid) / TICK_SIZE * direction

                # Limit exit check
                if direction == 1 and curr_mid >= exit_limit_price:
                    exit_type = 'limit_exit'
                    exit_bar = j
                    break
                elif direction == -1 and curr_mid <= exit_limit_price:
                    exit_type = 'limit_exit'
                    exit_bar = j
                    break

                # Take profit
                if take_profit_ticks and unrealized_ticks >= take_profit_ticks:
                    exit_type = 'take_profit'
                    exit_bar = j
                    break

                # Stop loss
                if stop_loss_ticks and unrealized_ticks <= -stop_loss_ticks:
                    exit_type = 'stop_loss'
                    exit_bar = j
                    break
            else:
                exit_type = 'timeout'
                exit_bar = max_hold

            exit_mid = m[exit_bar]

            # PnL calculation
            dir_pnl_ticks = (exit_mid - fill_mid) / TICK_SIZE * direction
            exit_edge = 0.5 if exit_type == 'limit_exit' else -0.5
            net_ticks = entry_edge + dir_pnl_ticks + exit_edge - COMMISSION_TICKS
            net_dollars = net_ticks * TICK_VALUE

            trades.append({
                'day': int(d),
                'direction': int(direction),
                'signal': float(sig),
                'is_strong': bool(is_strong),
                'entry_type': entry_type,
                'exit_type': exit_type,
                'fill_bar': int(fill_bar),
                'exit_bar': int(exit_bar),
                'bars_held': int(exit_bar - fill_bar),
                'entry_edge_ticks': float(entry_edge),
                'dir_pnl_ticks': float(dir_pnl_ticks),
                'exit_edge_ticks': float(exit_edge),
                'net_ticks': float(net_ticks),
                'net_dollars': float(net_dollars),
            })
            day_pnls[d].append(net_dollars)
            last_exit = exit_bar
            i = exit_bar + 1

    if not trades:
        return {'error': 'No trades generated', 'label': label}

    # Aggregate metrics
    pnl = np.array([t['net_ticks'] for t in trades])
    pnl_dollars = pnl * TICK_VALUE

    # Breakdown by entry type
    market_trades = [t for t in trades if t['entry_type'] == 'market']
    limit_trades = [t for t in trades if t['entry_type'] == 'limit']

    def _trade_stats(trade_list, name):
        if not trade_list:
            return {'name': name, 'n': 0}
        p = np.array([t['net_ticks'] for t in trade_list])
        return {
            'name': name,
            'n': len(trade_list),
            'mean_pnl_ticks': float(np.mean(p)),
            'mean_pnl_dollars': float(np.mean(p) * TICK_VALUE),
            'total_pnl_dollars': float(np.sum(p) * TICK_VALUE),
            'win_rate': float((p > 0).mean()),
            'mean_dir_pnl': float(np.mean([t['dir_pnl_ticks'] for t in trade_list])),
        }

    market_stats = _trade_stats(market_trades, 'market_entry')
    limit_stats = _trade_stats(limit_trades, 'limit_entry')

    # Exit type breakdown
    exit_counts = {}
    for t in trades:
        exit_counts[t['exit_type']] = exit_counts.get(t['exit_type'], 0) + 1

    # Day-by-day PnL
    day_pnl_totals = []
    for d_idx in sorted(day_pnls.keys()):
        if day_pnls[d_idx]:
            day_pnl_totals.append(float(sum(day_pnls[d_idx])))

    n_pos_days = sum(1 for x in day_pnl_totals if x > 0)
    n_neg_days = sum(1 for x in day_pnl_totals if x < 0)

    # Sharpe
    avg_hold_sec = float(np.mean([t['bars_held'] for t in trades])) / BARS_PER_SEC
    if avg_hold_sec > 0 and np.std(pnl) > 0:
        trades_per_year = 252 * 6.5 * 3600 / max(avg_hold_sec, 1.0)
        sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # Drawdown
    cum_pnl = np.cumsum(pnl_dollars)
    max_dd = float((np.maximum.accumulate(cum_pnl) - cum_pnl).max()) if len(cum_pnl) > 0 else 0.0

    limit_fill_rate = len(limit_trades) / n_limit_posted if n_limit_posted > 0 else 0.0

    return {
        'label': label,
        'config': {
            'strong_quantile': strong_quantile,
            'medium_quantile': medium_quantile,
            'stop_loss_ticks': stop_loss_ticks,
            'take_profit_ticks': take_profit_ticks,
            'hold_horizon_sec': hold_horizon_bars / BARS_PER_SEC,
            'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
        },
        'n_total_trades': len(trades),
        'n_market_entries': len(market_trades),
        'n_limit_entries': len(limit_trades),
        'n_limit_posted': n_limit_posted,
        'n_limit_not_filled': n_limit_not_filled,
        'limit_fill_rate': limit_fill_rate,
        'mean_pnl_ticks': float(np.mean(pnl)),
        'mean_pnl_dollars': float(np.mean(pnl_dollars)),
        'total_pnl_dollars': float(np.sum(pnl_dollars)),
        'win_rate': float((pnl > 0).mean()),
        'sharpe': sharpe,
        'max_drawdown_dollars': max_dd,
        'trades_per_day': len(trades) / max(n_days, 1),
        'market_entry_stats': market_stats,
        'limit_entry_stats': limit_stats,
        'exit_breakdown': {k: v for k, v in exit_counts.items()},
        'day_by_day_pnl': day_pnl_totals,
        'n_positive_days': n_pos_days,
        'n_negative_days': n_neg_days,
        'day_pnl_mean': float(np.mean(day_pnl_totals)) if day_pnl_totals else 0.0,
    }


# ============================================================================
# JSON HELPER
# ============================================================================
def to_safe(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [to_safe(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_safe(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Production Model ABC')
    parser.add_argument('--fast', action='store_true', help='Quick mode (fewer configs)')
    parser.add_argument('--horizons', nargs='+', default=None,
                        help='Specific horizons to test (e.g., 3s 5s 10s)')
    parser.add_argument('--no-vol-gate', action='store_true',
                        help='Disable vol gating (Option B)')
    parser.add_argument('--no-smart-exec', action='store_true',
                        help='Disable smart execution (Option C)')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("PRODUCTION MODEL ABC — CLEAN FEATURES + MULTI-HORIZON + VOL GATE + SMART EXEC")
    log.info(f"  Mode: {'FAST' if args.fast else 'FULL'}")
    log.info(f"  Clean features: {len(CLEAN_FEATURES)}")
    log.info(f"  Excluded features: {len(ALL_EXCLUDE)}")
    log.info("=" * 70)
    t_start = time.time()

    # ---- Load data ----
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        log.info("No feature cache -- loading from snapshot caches...")
        stats = scanner.load_from_cache()

    N = len(scanner.mid_prices)
    n_days = len(scanner.day_boundaries) - 1
    mid_prices = scanner.mid_prices.copy()
    log.info(f"Data loaded: {N:,} bars, {n_days} days, {len(scanner.feature_names)} features")
    log.info(f"  Mid price: ${np.nanmean(mid_prices):.2f}")

    # ---- Option B: Compute vol gate ----
    if not args.no_vol_gate:
        log.info("\n--- OPTION B: Computing vol gate ---")
        vol_gate = compute_vol_gate(
            mid_prices=mid_prices,
            day_boundaries=scanner.day_boundaries,
            lookback_bars=200,
            intraday_quantile=0.60,
        )
    else:
        vol_gate = None
        log.info("Vol gate: DISABLED")

    # ---- Option A: Multi-horizon targets ----
    if args.horizons:
        horizon_map = {h: int(h.replace('s', '')) for h in args.horizons}
    else:
        horizon_map = {
            '3s': 3,
            '5s': 5,
            '10s': 10,
            '30s': 30,
        }

    log.info(f"\n--- OPTION A: Testing horizons: {list(horizon_map.keys())} ---")
    targets = compute_return_targets(
        mid_prices=mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec=horizon_map,
        include_flow_target=False,
    )

    # ---- Walk-forward for each horizon (with and without vol gate) ----
    all_results = {}

    for hz_name in horizon_map.keys():
        target_key = f'ret_{hz_name}'
        if target_key not in targets:
            log.warning(f"  Target {target_key} not found, skipping")
            continue

        target = targets[target_key]
        hz_sec = horizon_map[hz_name]

        # Subsample: for longer horizons, don't predict every 100ms
        # Predict every hz_sec*BARS_PER_SEC bars (once per horizon period)
        subsample = max(1, hz_sec * BARS_PER_SEC // 2)  # predict 2x per horizon

        log.info(f"\n{'='*60}")
        log.info(f"HORIZON: {hz_name} (subsample every {subsample} bars)")
        log.info(f"{'='*60}")

        # Run WITHOUT vol gate (baseline)
        log.info(f"  [{hz_name}] Running WITHOUT vol gate (baseline)...")
        result_no_gate = walk_forward_clean(
            scanner=scanner,
            target=target,
            target_name=f'{target_key}_no_gate',
            vol_gate=None,
            min_train_days=3,
            subsample_every=subsample,
        )
        if 'error' not in result_no_gate:
            log.info(f"  [{hz_name}] NO GATE: IC={result_no_gate['ic']:.4f} "
                     f"ICIR={result_no_gate['icir']:.2f} t={result_no_gate['tstat']:.2f} "
                     f"HR={result_no_gate['hit_rate']:.1%} "
                     f"n={result_no_gate['n_predictions']:,}")
        else:
            log.warning(f"  [{hz_name}] NO GATE: {result_no_gate['error']}")

        all_results[f'{hz_name}_no_gate'] = result_no_gate

        # Run WITH vol gate (Option B)
        if vol_gate is not None:
            log.info(f"  [{hz_name}] Running WITH vol gate...")
            result_vol_gate = walk_forward_clean(
                scanner=scanner,
                target=target,
                target_name=f'{target_key}_vol_gate',
                vol_gate=vol_gate,
                min_train_days=3,
                subsample_every=subsample,
            )
            if 'error' not in result_vol_gate:
                log.info(f"  [{hz_name}] VOL GATE: IC={result_vol_gate['ic']:.4f} "
                         f"ICIR={result_vol_gate['icir']:.2f} t={result_vol_gate['tstat']:.2f} "
                         f"HR={result_vol_gate['hit_rate']:.1%} "
                         f"n={result_vol_gate['n_predictions']:,}")
            else:
                log.warning(f"  [{hz_name}] VOL GATE: {result_vol_gate['error']}")

            all_results[f'{hz_name}_vol_gate'] = result_vol_gate

        gc.collect()

    # ---- Option C: Smart execution on best horizon ----
    execution_results = {}

    if not args.no_smart_exec:
        log.info(f"\n{'='*60}")
        log.info("OPTION C: SMART EXECUTION SIMULATION")
        log.info(f"{'='*60}")

        # Find the best horizon (highest IC from vol-gated results)
        best_hz = None
        best_ic = -1
        best_result = None

        for key, result in all_results.items():
            if 'error' in result:
                continue
            if result['ic'] > best_ic:
                best_ic = result['ic']
                best_hz = key
                best_result = result

        if best_result is not None and 'predictions' in best_result:
            log.info(f"  Best horizon: {best_hz} (IC={best_ic:.4f})")

            # Grid of execution configs
            if args.fast:
                configs = [
                    {'strong_quantile': 0.95, 'medium_quantile': 0.80,
                     'stop_loss_ticks': 3.0, 'take_profit_ticks': 3.0,
                     'hold_horizon_bars': 100},
                    {'strong_quantile': 0.90, 'medium_quantile': 0.70,
                     'stop_loss_ticks': 2.0, 'take_profit_ticks': 3.0,
                     'hold_horizon_bars': 100},
                ]
            else:
                configs = [
                    # Aggressive (more market orders)
                    {'strong_quantile': 0.90, 'medium_quantile': 0.70,
                     'stop_loss_ticks': 2.0, 'take_profit_ticks': 2.0,
                     'hold_horizon_bars': 50},
                    {'strong_quantile': 0.90, 'medium_quantile': 0.70,
                     'stop_loss_ticks': 3.0, 'take_profit_ticks': 3.0,
                     'hold_horizon_bars': 100},
                    # Balanced
                    {'strong_quantile': 0.95, 'medium_quantile': 0.80,
                     'stop_loss_ticks': 3.0, 'take_profit_ticks': 3.0,
                     'hold_horizon_bars': 100},
                    {'strong_quantile': 0.95, 'medium_quantile': 0.80,
                     'stop_loss_ticks': 3.0, 'take_profit_ticks': 5.0,
                     'hold_horizon_bars': 200},
                    # Conservative (mostly limit orders)
                    {'strong_quantile': 0.98, 'medium_quantile': 0.85,
                     'stop_loss_ticks': 2.0, 'take_profit_ticks': 3.0,
                     'hold_horizon_bars': 100},
                    {'strong_quantile': 0.98, 'medium_quantile': 0.85,
                     'stop_loss_ticks': 3.0, 'take_profit_ticks': 5.0,
                     'hold_horizon_bars': 300},
                    # Market-only baseline
                    {'strong_quantile': 0.80, 'medium_quantile': 0.80,
                     'stop_loss_ticks': 3.0, 'take_profit_ticks': 3.0,
                     'hold_horizon_bars': 100},
                    # Limit-heavy
                    {'strong_quantile': 0.99, 'medium_quantile': 0.80,
                     'stop_loss_ticks': 2.0, 'take_profit_ticks': 2.0,
                     'hold_horizon_bars': 50},
                ]

            for ci, cfg in enumerate(configs):
                lbl = (f"sq={cfg['strong_quantile']:.0%}_mq={cfg['medium_quantile']:.0%}_"
                       f"sl={cfg['stop_loss_ticks']}_tp={cfg['take_profit_ticks']}_"
                       f"hold={cfg['hold_horizon_bars']/BARS_PER_SEC:.0f}s")

                log.info(f"  [{ci+1}/{len(configs)}] Testing: {lbl}")

                exec_result = simulate_smart_execution(
                    mid_prices=mid_prices,
                    day_boundaries=scanner.day_boundaries,
                    predictions=best_result['predictions'],
                    pred_indices=best_result['pred_indices'],
                    vol_gate=vol_gate,
                    label=lbl,
                    **cfg,
                )

                if 'error' not in exec_result:
                    log.info(f"    n={exec_result['n_total_trades']} "
                             f"(mkt={exec_result['n_market_entries']}, "
                             f"lmt={exec_result['n_limit_entries']}) "
                             f"mean={exec_result['mean_pnl_ticks']:+.4f}t "
                             f"total=${exec_result['total_pnl_dollars']:+.2f} "
                             f"win={exec_result['win_rate']:.1%} "
                             f"Sharpe={exec_result['sharpe']:.2f}")
                else:
                    log.warning(f"    ERROR: {exec_result['error']}")

                execution_results[lbl] = exec_result
        else:
            log.warning("  No valid predictions for execution simulation")

    # ---- Format Summary ----
    log.info(f"\n{'='*70}")
    log.info("PRODUCTION MODEL ABC — SUMMARY")
    log.info(f"{'='*70}")

    log.info(f"\nClean features used: {len(CLEAN_FEATURES)}")
    log.info(f"Total features excluded: {len(ALL_EXCLUDE)}")

    log.info(f"\n--- OPTION A: Multi-Horizon IC Comparison ---")
    log.info(f"{'Horizon':<25s} {'IC':>7s} {'ICIR':>6s} {'t-stat':>7s} "
             f"{'HR':>6s} {'Folds':>5s} {'Consist':>7s} {'nPreds':>8s}")
    log.info("-" * 75)

    for key in sorted(all_results.keys()):
        r = all_results[key]
        if 'error' in r:
            log.info(f"{key:<25s}  ERROR: {r['error']}")
            continue
        log.info(f"{key:<25s} {r['ic']:>7.4f} {r['icir']:>6.2f} {r['tstat']:>7.2f} "
                 f"{r['hit_rate']:>6.1%} {r['n_folds']:>5d} "
                 f"{r.get('fold_consistency', 0):>7.0%} {r['n_predictions']:>8,d}")

    if execution_results:
        log.info(f"\n--- OPTION C: Smart Execution Results ---")
        log.info(f"{'Config':<45s} {'nTrades':>7s} {'Mean_t':>8s} {'Total$':>10s} "
                 f"{'Win%':>6s} {'Sharpe':>7s} {'MktFrac':>8s}")
        log.info("-" * 95)

        sorted_exec = sorted(execution_results.items(),
                             key=lambda x: x[1].get('sharpe', 0)
                             if 'error' not in x[1] else -999,
                             reverse=True)
        for key, r in sorted_exec:
            if 'error' in r:
                log.info(f"{key:<45s}  ERROR: {r['error']}")
                continue
            mkt_frac = r['n_market_entries'] / max(r['n_total_trades'], 1)
            log.info(f"{key:<45s} {r['n_total_trades']:>7d} "
                     f"{r['mean_pnl_ticks']:>+8.4f}t "
                     f"${r['total_pnl_dollars']:>+9.2f} "
                     f"{r['win_rate']:>6.1%} {r['sharpe']:>7.2f} "
                     f"{mkt_frac:>8.1%}")

    # ---- Verdict ----
    log.info(f"\n{'='*70}")
    log.info("VERDICT")
    log.info(f"{'='*70}")

    # Find best overall
    best_exec_sharpe = -999
    best_exec_key = None
    for key, r in execution_results.items():
        if 'error' not in r and r.get('sharpe', 0) > best_exec_sharpe:
            best_exec_sharpe = r['sharpe']
            best_exec_key = key

    if best_exec_key:
        best = execution_results[best_exec_key]
        log.info(f"  Best execution config: {best_exec_key}")
        log.info(f"  Sharpe: {best['sharpe']:.2f}")
        log.info(f"  Mean PnL: {best['mean_pnl_ticks']:+.4f} ticks (${best['mean_pnl_dollars']:+.2f})")
        log.info(f"  Total PnL: ${best['total_pnl_dollars']:+.2f}")
        log.info(f"  Win rate: {best['win_rate']:.1%}")
        log.info(f"  Trades/day: {best.get('trades_per_day', 0):.1f}")
        log.info(f"  Market entries: {best['n_market_entries']} "
                 f"({best['market_entry_stats'].get('mean_pnl_ticks', 0):+.3f}t each)")
        log.info(f"  Limit entries: {best['n_limit_entries']} "
                 f"({best['limit_entry_stats'].get('mean_pnl_ticks', 0):+.3f}t each)")
        log.info(f"  Limit fill rate: {best['limit_fill_rate']:.1%}")
        log.info(f"  Pos/Neg days: {best['n_positive_days']}/{best['n_negative_days']}")

        profitable = best['mean_pnl_ticks'] > 0
        if profitable:
            log.info(f"\n  >>> PROFITABLE after costs! Ready for paper trading. <<<")
        else:
            log.info(f"\n  >>> NOT profitable. Need stronger signal or lower costs. <<<")

    # ---- Save results ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'production_abc_{timestamp}.json'

    # Strip numpy arrays from results before saving
    save_results = {}
    for key, r in all_results.items():
        sr = {k: v for k, v in r.items() if k not in ('predictions', 'pred_indices')}
        save_results[key] = to_safe(sr)

    save_exec = {}
    for key, r in execution_results.items():
        save_exec[key] = to_safe(r)

    output = {
        'timestamp': timestamp,
        'mode': 'fast' if args.fast else 'full',
        'clean_features': CLEAN_FEATURES,
        'n_clean_features': len(CLEAN_FEATURES),
        'excluded_features': ALL_EXCLUDE,
        'data': {
            'n_bars': int(N),
            'n_days': int(n_days),
            'mid_price_mean': float(np.nanmean(mid_prices)),
        },
        'vol_gate_active': vol_gate is not None,
        'vol_gate_pct': float(vol_gate.mean() * 100) if vol_gate is not None else 100.0,
        'horizons_tested': list(horizon_map.keys()),
        'ic_results': save_results,
        'execution_results': save_exec,
        'best_execution_config': best_exec_key,
        'best_execution_sharpe': best_exec_sharpe if best_exec_key else None,
        'total_elapsed_sec': time.time() - t_start,
    }

    with open(str(out_file), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved: {out_file}")
    log.info(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")

    return output


if __name__ == '__main__':
    main()
