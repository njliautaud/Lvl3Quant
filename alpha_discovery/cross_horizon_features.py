#!/usr/bin/env python3
"""
Cross-Horizon Feature Engineering.

User insight: "a small timeframe event is what TRIGGERS a longer timeframe
continuation or reversal." This script builds features that capture exactly
that — short-term microstructure triggers predicting longer-term moves.

Architecture:
  Input: Raw 340-feature MBO snapshots (10ms bars)
  Output: Cross-horizon feature matrix (~150-200 features) subsampled at
          the target horizon frequency.

Feature Categories:
  1. TRIGGER DETECTION — When did a strong microstructure event fire?
     - Order imbalance spikes (>2σ, >3σ)
     - Volume/trade burst events
     - Spread expansion events
     - Large cancel/modify events
     - Queue depletion events (depth drops >50%)

  2. TRIGGER CONTEXT — What was the market state when the trigger fired?
     - Trend alignment (is trigger with or against trend?)
     - Volatility regime (calm/normal/volatile)
     - Liquidity state (thick/thin book)
     - Time of day (open/mid/close effects)

  3. TRIGGER PERSISTENCE — How long ago? How strong? Still active?
     - Time since last trigger event (in bars)
     - Cumulative trigger count in recent window
     - Trigger decay (exponentially weighted)

  4. MULTI-SCALE FLOW — Order flow aggregated at multiple scales
     - 1s, 5s, 30s, 1min, 5min rolling OFI
     - Cross-scale divergence (short flow vs long flow)
     - Flow acceleration (rate of change of flow)

  5. REGIME FEATURES — Market state classification
     - Realized vol ratio (5s vol / 1min vol)
     - Trend strength (rolling return / rolling vol)
     - Book replenishment rate
     - Trade arrival rate regime

Usage:
  python cross_horizon_features.py --horizon 5m --n-days 100 --workers 12
  python cross_horizon_features.py --horizon 15m --n-days 50
"""

import argparse
import gc
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    format='%(asctime)s [cross_hz] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('cross_hz')

# ---------------------------------------------------------------------------
# Feature indices from the 340-feature MBO engine
# ---------------------------------------------------------------------------
# Price levels (EXCLUDE from model, but USE for feature derivation)
IDX_MID = 0
IDX_MICROPRICE = 3
IDX_BEST_BID = 8
IDX_BEST_ASK = 9

# Order flow features
IDX_BID_SIZE_L1 = 10    # L1 bid size
IDX_ASK_SIZE_L1 = 11    # L1 ask size
IDX_BID_ORDERS_L1 = 12  # L1 bid order count
IDX_ASK_ORDERS_L1 = 13  # L1 ask order count
IDX_SPREAD = 4           # spread
IDX_IMBALANCE = 5        # L1 order imbalance

# Trade/volume features
IDX_TRADE_VOL = 14       # total trade volume
IDX_BUY_VOL = 15         # aggressive buy volume
IDX_SELL_VOL = 16        # aggressive sell volume
IDX_TRADE_COUNT = 17     # trade count

# Depth features
IDX_TOTAL_DEPTH_BID = 18  # total bid depth (L1-L10)
IDX_TOTAL_DEPTH_ASK = 19  # total ask depth (L1-L10)

# MBO event features (adds, cancels, modifies)
IDX_BID_ADDS = 30
IDX_ASK_ADDS = 31
IDX_BID_CANCELS = 32
IDX_ASK_CANCELS = 33

# Time features
IDX_TIME_SINCE_RTH = 28  # seconds since RTH open
IDX_MINUTE_NORM = 27     # normalized minute
IDX_TIME_TO_CLOSE = 29   # seconds to close

# Key derived features
IDX_MAX_CHAIN_LOG = 63    # max order chain (top LightGBM feature)
IDX_TOTAL_DEPTH_LOG = 81  # total depth log
IDX_DEPTH_RATIO_Z = 82    # depth ratio z-score

# Horizon configs
HORIZON_CONFIGS = {
    '1m':  {'bars': 6000,   'label': '1min',  'subsample': 100},
    '5m':  {'bars': 30000,  'label': '5min',  'subsample': 600},
    '15m': {'bars': 90000,  'label': '15min', 'subsample': 1800},
    '30m': {'bars': 180000, 'label': '30min', 'subsample': 3600},
}

# Multi-scale rolling windows (in 10ms bars)
SCALE_WINDOWS = {
    '100ms': 10,
    '500ms': 50,
    '1s':    100,
    '5s':    500,
    '30s':   3000,
    '1min':  6000,
    '5min':  30000,
}


# ---------------------------------------------------------------------------
# Feature computation functions
# ---------------------------------------------------------------------------

def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using cumsum. NaN-safe."""
    clean = np.nan_to_num(arr, nan=0.0)
    cs = np.cumsum(clean)
    rs = np.empty_like(clean)
    rs[:window] = cs[:window] / np.arange(1, window + 1)  # partial windows
    rs[window:] = (cs[window:] - cs[:-window]) / window
    return rs


def rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation. NaN-safe."""
    clean = np.nan_to_num(arr, nan=0.0)
    mean = rolling_mean(clean, window)
    mean_sq = rolling_mean(clean ** 2, window)
    var = mean_sq - mean ** 2
    var = np.clip(var, 0, None)
    result = np.sqrt(var)
    return result


def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling sum. NaN-safe."""
    clean = np.nan_to_num(arr, nan=0.0)
    cs = np.cumsum(clean)
    rs = np.empty_like(clean)
    rs[:window] = cs[:window]  # partial windows
    rs[window:] = cs[window:] - cs[:-window]
    return rs


def rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling max (using stride tricks for speed)."""
    n = len(arr)
    result = np.full(n, np.nan)
    # Use a sliding window approach
    for i in range(window, n):
        result[i] = np.max(arr[i-window:i])
    return result


def ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    result = np.zeros_like(arr, dtype=np.float64)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result


def z_score(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score."""
    mean = rolling_mean(arr, window)
    std = rolling_std(arr, window)
    std = np.where(std < 1e-10, 1e-10, std)
    return (arr - mean) / std


def compute_cross_horizon_features(
    raw_features: np.ndarray,  # (N, 340)
    horizon_bars: int,
    subsample_step: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute cross-horizon features from raw 340-feature MBO snapshots.

    Returns:
        features: (M, F) array where M = N // subsample_step, F = num features
        names: list of feature names
    """
    N = raw_features.shape[0]

    # Extract key raw series (fill NaN with 0 to prevent cumsum propagation)
    def _extract(col_idx):
        arr = raw_features[:, col_idx].astype(np.float64)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    mid = raw_features[:, IDX_MID].astype(np.float64)
    # Forward-fill mid price NaN (day boundary) so returns aren't NaN
    mask = np.isnan(mid)
    if mask.any():
        # Find first valid value and fill backward
        first_valid = np.argmax(~mask)
        mid[:first_valid] = mid[first_valid]

    microprice = _extract(IDX_MICROPRICE)
    spread = _extract(IDX_SPREAD)
    imbalance = _extract(IDX_IMBALANCE)
    bid_size = _extract(IDX_BID_SIZE_L1)
    ask_size = _extract(IDX_ASK_SIZE_L1)
    trade_vol = _extract(IDX_TRADE_VOL)
    buy_vol = _extract(IDX_BUY_VOL)
    sell_vol = _extract(IDX_SELL_VOL)
    trade_count = _extract(IDX_TRADE_COUNT)
    total_depth_bid = _extract(IDX_TOTAL_DEPTH_BID)
    total_depth_ask = _extract(IDX_TOTAL_DEPTH_ASK)
    time_since_rth = _extract(IDX_TIME_SINCE_RTH)

    # Derived base series
    returns = np.diff(mid, prepend=mid[0]) / np.where(mid > 0, mid, 1.0)
    ofi = buy_vol - sell_vol  # order flow imbalance
    depth_ratio = (total_depth_bid - total_depth_ask) / np.clip(total_depth_bid + total_depth_ask, 1, None)
    microprice_dev = microprice - mid  # microprice deviation from mid

    features = []
    names = []

    # =======================================================================
    # CATEGORY 1: MULTI-SCALE ORDER FLOW
    # The KEY insight: aggregate OFI at multiple timescales
    # =======================================================================
    for label, window in [('1s', 100), ('5s', 500), ('30s', 3000), ('1min', 6000), ('5min', 30000)]:
        if window > N // 2:
            continue

        # Rolling OFI (order flow imbalance)
        ofi_roll = rolling_sum(ofi, window)
        features.append(ofi_roll)
        names.append(f'ofi_{label}')

        # Rolling trade imbalance (buy - sell as fraction)
        buy_roll = rolling_sum(buy_vol, window)
        sell_roll = rolling_sum(sell_vol, window)
        total_vol = buy_roll + sell_roll
        trade_imb = np.where(total_vol > 0, (buy_roll - sell_roll) / total_vol, 0)
        features.append(trade_imb)
        names.append(f'trade_imbalance_{label}')

        # Rolling return
        ret_roll = rolling_sum(returns, window)
        features.append(ret_roll)
        names.append(f'return_{label}')

        # Rolling volatility
        vol_roll = rolling_std(returns, window) * np.sqrt(window)
        features.append(vol_roll)
        names.append(f'volatility_{label}')

    # =======================================================================
    # CATEGORY 2: CROSS-SCALE DIVERGENCE
    # When short-term flow disagrees with long-term flow → potential reversal
    # When they agree → potential continuation
    # =======================================================================
    for short_label, short_w, long_label, long_w in [
        ('1s', 100, '1min', 6000),
        ('1s', 100, '5min', 30000),
        ('5s', 500, '5min', 30000),
        ('30s', 3000, '5min', 30000),
    ]:
        if long_w > N // 2:
            continue

        short_ofi = rolling_sum(ofi, short_w)
        long_ofi = rolling_sum(ofi, long_w)

        # Normalize to comparable scale
        short_z = z_score(short_ofi, long_w)
        long_z = z_score(long_ofi, long_w)

        # Divergence: short vs long
        features.append(short_z - long_z)
        names.append(f'ofi_divergence_{short_label}_vs_{long_label}')

        # Agreement: both same direction and strong
        features.append(short_z * long_z)
        names.append(f'ofi_agreement_{short_label}_vs_{long_label}')

        # Short-term return vs long-term flow (mean-reversion signal)
        short_ret = rolling_sum(returns, short_w)
        ret_z = z_score(short_ret, long_w)
        features.append(ret_z - long_z)
        names.append(f'ret_vs_flow_{short_label}_vs_{long_label}')

    # =======================================================================
    # CATEGORY 3: TRIGGER DETECTION
    # Detect when microstructure events exceed thresholds
    # =======================================================================

    # Imbalance spike detection
    imb_z_short = z_score(imbalance, 500)   # 5-second z-score
    imb_z_long = z_score(imbalance, 6000)   # 1-minute z-score
    features.append(imb_z_short)
    names.append('imbalance_z_5s')
    features.append(imb_z_long)
    names.append('imbalance_z_1min')

    # Imbalance spike triggers (binary-ish, but use tanh for smoothness)
    imb_trigger_2sig = np.tanh(np.clip(np.abs(imb_z_short) - 2, 0, None))
    imb_trigger_3sig = np.tanh(np.clip(np.abs(imb_z_short) - 3, 0, None))
    features.append(imb_trigger_2sig * np.sign(imb_z_short))
    names.append('imbalance_trigger_2sig')
    features.append(imb_trigger_3sig * np.sign(imb_z_short))
    names.append('imbalance_trigger_3sig')

    # Volume burst detection
    vol_z = z_score(trade_vol, 3000)  # 30-second z-score of volume
    vol_burst = np.tanh(np.clip(vol_z - 2, 0, None))
    features.append(vol_z)
    names.append('volume_z_30s')
    features.append(vol_burst)
    names.append('volume_burst_trigger')

    # Spread expansion (market stress signal)
    spread_z = z_score(spread, 6000)  # 1-min z-score
    spread_trigger = np.tanh(np.clip(spread_z - 1.5, 0, None))
    features.append(spread_z)
    names.append('spread_z_1min')
    features.append(spread_trigger)
    names.append('spread_expansion_trigger')

    # Depth depletion (one side getting thin)
    depth_total = total_depth_bid + total_depth_ask
    depth_z = z_score(depth_total, 6000)
    depth_depletion = np.tanh(np.clip(-depth_z - 1.5, 0, None))
    features.append(depth_z)
    names.append('depth_z_1min')
    features.append(depth_depletion)
    names.append('depth_depletion_trigger')

    # Microprice deviation trigger (strong directional pressure)
    mpd_z = z_score(microprice_dev, 3000)
    mpd_trigger = np.tanh(np.clip(np.abs(mpd_z) - 2, 0, None))
    features.append(mpd_z)
    names.append('microprice_dev_z_30s')
    features.append(mpd_trigger * np.sign(mpd_z))
    names.append('microprice_dev_trigger')

    # =======================================================================
    # CATEGORY 4: TRIGGER PERSISTENCE & DECAY
    # How recent was the last trigger? How many in the recent window?
    # =======================================================================

    # Count triggers in rolling windows
    for trigger_name, trigger_arr in [
        ('imb_2sig', (np.abs(imb_z_short) > 2).astype(np.float64)),
        ('vol_burst', (vol_z > 2).astype(np.float64)),
        ('spread_exp', (spread_z > 1.5).astype(np.float64)),
    ]:
        for window_label, window in [('1min', 6000), ('5min', 30000)]:
            if window > N // 2:
                continue
            count = rolling_sum(trigger_arr, window)
            features.append(count)
            names.append(f'{trigger_name}_count_{window_label}')

        # Exponential decay since last trigger (vectorized)
        trigger_indices = np.where(trigger_arr > 0)[0]
        if len(trigger_indices) > 0:
            # For each bar, find distance to most recent trigger
            # Use searchsorted for O(N log K) instead of O(N) python loop
            insert_pos = np.searchsorted(trigger_indices, np.arange(N), side='right')
            insert_pos = np.clip(insert_pos - 1, 0, len(trigger_indices) - 1)
            nearest_trigger = trigger_indices[insert_pos]
            # Only count triggers that are BEFORE current bar
            dist = np.arange(N) - nearest_trigger
            dist = np.where(dist >= 0, dist, 10000)
            decay = np.exp(-dist / 3000.0)
        else:
            decay = np.zeros(N, dtype=np.float64)
        features.append(decay)
        names.append(f'{trigger_name}_decay')

    # =======================================================================
    # CATEGORY 5: TRIGGER + CONTEXT INTERACTIONS
    # Trigger strength × market state = directional prediction
    # =======================================================================

    # Trend state (which direction is the market moving?)
    trend_1min = rolling_sum(returns, 6000) if N > 12000 else np.zeros(N)
    trend_5min = rolling_sum(returns, 30000) if N > 60000 else np.zeros(N)

    features.append(trend_1min)
    names.append('trend_1min')
    if N > 60000:
        features.append(trend_5min)
        names.append('trend_5min')

    # Trigger × Trend (is trigger WITH or AGAINST the trend?)
    if N > 12000:
        features.append(imb_z_short * np.sign(trend_1min))
        names.append('imbalance_with_trend_1min')
        features.append(mpd_z * np.sign(trend_1min))
        names.append('microprice_with_trend_1min')

    if N > 60000:
        features.append(imb_z_short * np.sign(trend_5min))
        names.append('imbalance_with_trend_5min')
        features.append(mpd_z * np.sign(trend_5min))
        names.append('microprice_with_trend_5min')

    # Trigger × Volatility regime
    vol_1min = rolling_std(returns, 6000) if N > 12000 else np.ones(N) * 1e-6
    vol_regime = z_score(vol_1min, 30000) if N > 60000 else np.zeros(N)

    if N > 60000:
        features.append(vol_regime)
        names.append('vol_regime_z')
        features.append(imb_z_short * vol_regime)
        names.append('imbalance_x_vol_regime')
        features.append(mpd_z * vol_regime)
        names.append('microprice_x_vol_regime')

    # Trigger × Liquidity state
    liquidity_z = z_score(depth_total, 30000) if N > 60000 else np.zeros(N)
    if N > 60000:
        features.append(liquidity_z)
        names.append('liquidity_regime_z')
        features.append(imb_z_short * liquidity_z)
        names.append('imbalance_x_liquidity')

    # =======================================================================
    # CATEGORY 6: FLOW MOMENTUM & ACCELERATION
    # Rate of change of order flow — flow speeding up or slowing down
    # =======================================================================
    ofi_1s = rolling_sum(ofi, 100)
    ofi_5s = rolling_sum(ofi, 500)
    ofi_30s = rolling_sum(ofi, 3000) if N > 6000 else np.zeros(N)

    # Flow acceleration (diff of flow)
    flow_accel_1s = np.diff(ofi_1s, prepend=0)
    flow_accel_5s = np.diff(ofi_5s, prepend=0)
    features.append(flow_accel_1s)
    names.append('flow_acceleration_1s')
    features.append(flow_accel_5s)
    names.append('flow_acceleration_5s')

    # Flow momentum (is flow strengthening or weakening?)
    if N > 6000:
        flow_momentum = ofi_1s - ofi_30s / 30  # short flow vs avg
        features.append(flow_momentum)
        names.append('flow_momentum_1s_vs_30s')

    # =======================================================================
    # CATEGORY 7: BOOK REPLENISHMENT & RESILIENCE
    # How quickly does the book rebuild after taking liquidity?
    # =======================================================================
    depth_change = np.diff(depth_total, prepend=depth_total[0])

    # Book replenishment rate (positive = rebuilding)
    replenish_1s = rolling_sum(np.clip(depth_change, 0, None), 100)
    depletion_1s = rolling_sum(np.clip(-depth_change, 0, None), 100)
    resilience = np.where(depletion_1s > 0, replenish_1s / np.clip(depletion_1s, 1, None), 1.0)
    features.append(resilience)
    names.append('book_resilience_1s')

    replenish_30s = rolling_sum(np.clip(depth_change, 0, None), 3000) if N > 6000 else np.zeros(N)
    depletion_30s = rolling_sum(np.clip(-depth_change, 0, None), 3000) if N > 6000 else np.zeros(N)
    if N > 6000:
        resilience_30s = np.where(depletion_30s > 0, replenish_30s / np.clip(depletion_30s, 1, None), 1.0)
        features.append(resilience_30s)
        names.append('book_resilience_30s')

    # =======================================================================
    # CATEGORY 8: TRADE INTENSITY REGIME
    # Cluster density of trades — quiet vs active periods
    # =======================================================================
    trade_rate_1s = rolling_sum(trade_count, 100)
    trade_rate_1min = rolling_sum(trade_count, 6000) if N > 12000 else np.zeros(N)

    features.append(trade_rate_1s)
    names.append('trade_rate_1s')
    if N > 12000:
        features.append(trade_rate_1min)
        names.append('trade_rate_1min')

        # Trade intensity regime (relative to recent average)
        intensity_z = z_score(trade_rate_1s, 6000)
        features.append(intensity_z)
        names.append('trade_intensity_z')

    # =======================================================================
    # CATEGORY 9: TIME-OF-DAY INTERACTION
    # Microstructure signals interact differently at different times
    # =======================================================================
    # Normalize time to [0, 1] across RTH
    rth_duration = 23400.0  # 6.5 hours in seconds
    time_frac = np.clip(time_since_rth / rth_duration, 0, 1)

    # Opening (first 30 min)
    is_opening = (time_frac < 30/390).astype(np.float64)
    # Closing (last 30 min)
    is_closing = (time_frac > 360/390).astype(np.float64)
    # Midday
    is_midday = ((time_frac > 120/390) & (time_frac < 240/390)).astype(np.float64)

    features.append(is_opening)
    names.append('is_opening_30min')
    features.append(is_closing)
    names.append('is_closing_30min')
    features.append(is_midday)
    names.append('is_midday')

    # Trigger × time interactions
    features.append(imb_z_short * is_opening)
    names.append('imbalance_trigger_opening')
    features.append(imb_z_short * is_closing)
    names.append('imbalance_trigger_closing')
    features.append(vol_z * is_opening)
    names.append('volume_burst_opening')

    # =======================================================================
    # CATEGORY 10: CUMULATIVE FEATURES (useful for longer horizons)
    # Cumulative sum since market open — tracks intraday drift
    # =======================================================================
    cum_ofi = np.cumsum(ofi)
    cum_ret = np.cumsum(returns)
    cum_vol = np.cumsum(trade_vol)

    features.append(z_score(cum_ofi, 30000) if N > 60000 else np.zeros(N))
    names.append('cumulative_ofi_z')
    features.append(z_score(cum_ret, 30000) if N > 60000 else np.zeros(N))
    names.append('cumulative_return_z')

    # =======================================================================
    # SUBSAMPLE to target horizon frequency
    # =======================================================================
    feature_matrix = np.column_stack(features)

    # Subsample indices
    indices = np.arange(subsample_step - 1, N, subsample_step)

    # For each subsample point, take the feature values
    subsampled = feature_matrix[indices]

    # Also compute the forward return for target
    fwd_return = np.full(N, np.nan)
    fwd_return[:N-horizon_bars] = (mid[horizon_bars:] - mid[:N-horizon_bars]) / mid[:N-horizon_bars]
    target = fwd_return[indices]

    return subsampled, target, names


def load_day(fpath: Path) -> np.ndarray:
    """Load raw 340-feature MBO snapshot for one day."""
    data = np.load(str(fpath))
    return data['mbo_features']


def main():
    parser = argparse.ArgumentParser(description="Cross-horizon feature engineering + LightGBM")
    parser.add_argument('--horizon', type=str, default='5m', choices=list(HORIZON_CONFIGS.keys()))
    parser.add_argument('--n-days', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--data-dir', type=str,
                        default=str(ROOT_DIR / 'data' / 'processed' / 'mbo_features_cache'))
    parser.add_argument('--output-dir', type=str,
                        default=str(ROOT_DIR / 'alpha_discovery' / 'results'))
    args = parser.parse_args()

    config = HORIZON_CONFIGS[args.horizon]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f'cross_horizon_{args.horizon}_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [cross_hz] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    # Find MBO feature files
    files = sorted(data_dir.glob('*_mbo_features.npz'))
    if not files:
        files = sorted(data_dir.glob('mbo_features_*.npz'))
    if args.n_days > 0:
        files = files[:args.n_days]

    logger.info(f"Found {len(files)} days of MBO features")
    logger.info(f"Horizon: {args.horizon} ({config['bars']} bars = {config['bars']/100:.0f}s)")
    logger.info(f"Subsample step: {config['subsample']} bars")
    logger.info(f"Workers: {args.workers}")

    # Walk-forward training
    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed. pip install lightgbm")
        sys.exit(1)

    lgbm_params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 6,
        'min_child_samples': 100,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': args.workers,
        'seed': 42,
    }

    MIN_TRAIN_DAYS = 15
    all_ics = []
    feature_names = None
    feature_importance_total = None

    logger.info("\n" + "=" * 70)
    logger.info(f"WALK-FORWARD: Cross-Horizon LightGBM ({args.horizon})")
    logger.info("=" * 70)

    # Pre-compute features per day
    logger.info("Pre-computing cross-horizon features per day...")
    day_data = []
    t0 = time.time()

    for i, fpath in enumerate(files):
        # Handle both naming conventions: YYYY-MM-DD_mbo_features.npz and mbo_features_YYYY-MM-DD.npz
        stem = fpath.stem
        date_str = stem.replace('_mbo_features', '').replace('mbo_features_', '')
        try:
            raw = load_day(fpath)
            X, y, names = compute_cross_horizon_features(
                raw, config['bars'], config['subsample']
            )

            # Remove rows where target is NaN
            valid = ~np.isnan(y) & np.all(np.isfinite(X), axis=1)
            X_valid = X[valid]
            y_valid = y[valid]

            if feature_names is None:
                feature_names = names
                logger.info(f"Feature count: {len(names)}")

            day_data.append({
                'date': date_str,
                'X': X_valid.astype(np.float32),
                'y': y_valid.astype(np.float32),
                'n_samples': len(y_valid),
            })

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"  [{i+1}/{len(files)}] {date_str}: {len(y_valid)} samples, {X_valid.shape[1]} features")

            del raw
            gc.collect()

        except Exception as e:
            logger.warning(f"  Error processing {date_str}: {e}")
            continue

    elapsed = time.time() - t0
    total_samples = sum(d['n_samples'] for d in day_data)
    logger.info(f"Pre-computed {len(day_data)} days, {total_samples:,} total samples ({elapsed:.1f}s)")

    # Walk-forward loop
    for test_idx in range(MIN_TRAIN_DAYS, len(day_data)):
        train_days = day_data[:test_idx]
        test_day = day_data[test_idx]

        # Assemble training data
        X_train = np.vstack([d['X'] for d in train_days])
        y_train = np.concatenate([d['y'] for d in train_days])
        X_test = test_day['X']
        y_test = test_day['y']

        if len(y_test) < 10:
            continue

        # Replace NaN/inf in training data
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Train LightGBM
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=200)

        # Predict
        preds = model.predict(X_test)

        # IC
        ic, _ = spearmanr(preds, y_test)
        all_ics.append(ic)

        # Feature importance (cumulative)
        imp = model.feature_importance(importance_type='gain')
        if feature_importance_total is None:
            feature_importance_total = imp.astype(np.float64)
        else:
            feature_importance_total += imp

        # Top 3 features
        top3_idx = np.argsort(imp)[-3:][::-1]
        top3 = [feature_names[j] for j in top3_idx]

        if (test_idx - MIN_TRAIN_DAYS) % 5 == 0 or test_idx == len(day_data) - 1:
            mean_ic = np.mean(all_ics)
            logger.info(f"  [{test_idx+1}/{len(day_data)}] {test_day['date']}  "
                       f"IC={ic:+.4f}  mean_IC={mean_ic:+.4f}  "
                       f"train={len(y_train):,}  top3={top3}")

        del X_train, y_train, dtrain, model
        gc.collect()

    # Summary
    if all_ics:
        ics = np.array(all_ics)
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else 0
        pct_pos = np.mean(ics > 0) * 100

        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: Cross-Horizon LightGBM ({args.horizon})")
        logger.info(f"{'='*70}")
        logger.info(f"  Days predicted: {len(ics)}")
        logger.info(f"  Mean IC: {mean_ic:+.4f}")
        logger.info(f"  Std IC:  {std_ic:.4f}")
        logger.info(f"  t-stat:  {t_stat:.2f}")
        logger.info(f"  Pct positive: {pct_pos:.1f}%")
        logger.info(f"  Min IC: {np.min(ics):+.4f}")
        logger.info(f"  Max IC: {np.max(ics):+.4f}")

        # Top 10 features by cumulative importance
        if feature_importance_total is not None:
            top10_idx = np.argsort(feature_importance_total)[-10:][::-1]
            logger.info(f"\n  Top 10 features (cumulative gain):")
            for rank, j in enumerate(top10_idx, 1):
                logger.info(f"    {rank:2d}. {feature_names[j]:40s}  gain={feature_importance_total[j]:.0f}")

        # Compare to raw feature baseline
        logger.info(f"\n  Raw feature IC baseline @ {args.horizon}: ~0.10")
        if mean_ic > 0.12:
            logger.info(f"  >>> CROSS-HORIZON FEATURES ADD VALUE! (+{mean_ic - 0.10:.4f})")
        elif mean_ic > 0.08:
            logger.info(f"  >>> Marginal improvement. Need more features or better triggers.")
        else:
            logger.info(f"  >>> No improvement over baseline.")

        # Save results
        results = {
            'horizon': args.horizon,
            'n_days': len(day_data),
            'n_folds': len(ics),
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            't_stat': float(t_stat),
            'pct_positive': float(pct_pos),
            'fold_ics': ics.tolist(),
            'feature_names': feature_names,
            'feature_importance': feature_importance_total.tolist() if feature_importance_total is not None else [],
        }

        import json
        results_path = output_dir / f'cross_horizon_{args.horizon}_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n  Results saved: {results_path}")

    logger.info(f"\nDone.")


if __name__ == '__main__':
    main()
