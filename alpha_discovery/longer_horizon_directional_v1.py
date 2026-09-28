#!/usr/bin/env python3
"""
Longer-Horizon Directional Model v1 (HC #637)
==============================================

Strategic thesis: "News is already priced in. Someone's already positioned.
Use MBO microstructure to detect WHERE large players are positioned,
get on the right side AHEAD of the move."

This script:
  1. Aggregates 1-minute MBO bars → multi-hour feature bars
  2. Builds rich microstructure features (flow imbalance momentum,
     sweep proxies, absorption, volume profile)
  3. Adds macro regime context (VIX, trend, volatility regime)
  4. Labels: forward N-hour return direction (long/short/flat)
  5. Trains LightGBM with sliding walk-forward validation
  6. Reports per-regime, per-horizon results

Horizons: 1h, 2h, 4h, EOD (remaining session)

Author: Claude (HC #637 research)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
REGIME_DIR = ROOT / "data" / "feature_store" / "v1"
OUTPUT_DIR = ROOT / "output" / "longer_horizon_v1"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s [LH-v1] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "longer_horizon_v1.log")),
    ],
)
log = logging.getLogger('LH-v1')

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
# For longer-horizon trades, assume market entry + market exit
COST_RT_TICKS = 2 * 1.0 + ES_RT_COMMISSION_TICKS  # 2 spread crossings + commission = 2.376 ticks
# But actually: 1 spread crossing entry + 1 spread crossing exit + RT commission
# = 1.0 + 1.0 + 0.376 = 2.376 ticks total
# This is CONSERVATIVE — passive entries would be 1.376

# Minimum edge required (in ticks) to justify a trade at each horizon
MIN_EDGE_TICKS = {
    '1h': 3.0,    # Need 3+ ticks of directional move to be worth it
    '2h': 4.0,
    '4h': 5.0,
    'eod': 6.0,
}


# ═════════════════════════════════════════════
#  STEP 1: LOAD AND AGGREGATE MINUTE BARS
# ═════════════════════════════════════════════

def load_all_minute_bars(min_date: str = "20250714") -> pd.DataFrame:
    """Load all minute bar parquets into a single DataFrame."""
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        if f.stem < min_date:
            continue
        try:
            df = pd.read_parquet(f)
            df['date'] = f.stem
            frames.append(df)
        except Exception as e:
            log.warning(f"Skip {f.stem}: {e}")

    if not frames:
        raise RuntimeError(f"No minute bar files found in {MINUTE_BAR_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    combined['ts_minute'] = pd.to_datetime(combined['ts_minute'], utc=True)
    combined = combined.sort_values('ts_minute').reset_index(drop=True)
    log.info(f"Loaded {len(combined)} minute bars across {len(frames)} days "
             f"({frames[0]['date'].iloc[0]} → {frames[-1]['date'].iloc[0]})")
    return combined


def compute_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 1-minute bars to hourly bars with rich microstructure features.

    Features capture the POSITIONING and FLOW patterns that indicate
    where informed players are building positions.
    """
    # Group by date + hour
    df = df.copy()
    df['hour'] = df['ts_minute'].dt.hour
    df['date_str'] = df['date']

    # Compute per-minute derived features BEFORE grouping
    df['return_1m'] = df.groupby('date_str')['close'].pct_change()
    df['log_volume'] = np.log1p(df['volume'])
    df['abs_ofi'] = df['ofi_1min'].abs()

    # Identify "sweep" proxies: minutes with extreme signed volume
    sv_std = df.groupby('date_str')['signed_volume'].transform('std')
    sv_std = sv_std.replace(0, 1)
    df['sv_zscore'] = df['signed_volume'] / sv_std

    # Volume-weighted price (VWAP deviation from close)
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['close'].clip(lower=1)

    hourly_records = []

    for (date_str, hour), group in df.groupby(['date_str', 'hour']):
        if len(group) < 5:  # need at least 5 minutes
            continue

        close_arr = group['close'].values
        volume_arr = group['volume'].values
        ofi_arr = group['ofi_1min'].values
        sv_arr = group['signed_volume'].values
        ret_arr = group['return_1m'].fillna(0).values
        spread_arr = group['spread_mean'].values
        tc_arr = group['trade_count'].values

        rec = {
            'date': date_str,
            'hour': hour,
            'ts': group['ts_minute'].iloc[0],

            # ── PRICE ACTION ──
            'open': close_arr[0],
            'high': close_arr.max(),
            'low': close_arr.min(),
            'close': close_arr[-1],
            'return_1h': (close_arr[-1] / close_arr[0] - 1) if close_arr[0] > 0 else 0,
            'range_ticks': (close_arr.max() - close_arr.min()) / 0.25,  # ES tick = 0.25
            'close_position': (close_arr[-1] - close_arr.min()) / max(close_arr.max() - close_arr.min(), 0.25),

            # ── VOLUME PROFILE ──
            'total_volume': volume_arr.sum(),
            'avg_volume': volume_arr.mean(),
            'volume_trend': np.polyfit(np.arange(len(volume_arr)), volume_arr, 1)[0] if len(volume_arr) > 1 else 0,
            'volume_concentration': (volume_arr.max() / max(volume_arr.mean(), 1)),  # spiky = informed

            # ── ORDER FLOW IMBALANCE (KEY: Positioning Signal) ──
            'ofi_sum': ofi_arr.sum(),                    # Net flow direction
            'ofi_mean': ofi_arr.mean(),
            'ofi_trend': np.polyfit(np.arange(len(ofi_arr)), ofi_arr, 1)[0] if len(ofi_arr) > 1 else 0,  # Accelerating flow
            'ofi_consistency': np.mean(np.sign(ofi_arr) == np.sign(ofi_arr.sum())) if ofi_arr.sum() != 0 else 0.5,
            'ofi_late_vs_early': (ofi_arr[len(ofi_arr)//2:].sum() - ofi_arr[:len(ofi_arr)//2].sum()),  # Late-hour flow shift

            # ── SIGNED VOLUME (Aggressive Side) ──
            'signed_volume_sum': sv_arr.sum(),
            'signed_volume_ratio': sv_arr.sum() / max(volume_arr.sum(), 1),  # fraction of volume that's directional
            'buy_volume_fraction': np.sum(sv_arr[sv_arr > 0]) / max(volume_arr.sum(), 1),
            'sell_volume_fraction': -np.sum(sv_arr[sv_arr < 0]) / max(volume_arr.sum(), 1),

            # ── SWEEP PROXY (Large Directional Bursts) ──
            'sweep_minutes': np.sum(np.abs(group['sv_zscore'].values) > 2),  # minutes with extreme flow
            'max_sweep_intensity': np.abs(group['sv_zscore'].values).max(),
            'sweep_direction': np.sign(sv_arr[np.abs(group['sv_zscore'].values).argmax()]) if len(sv_arr) > 0 else 0,

            # ── SPREAD & LIQUIDITY ──
            'spread_mean': spread_arr.mean(),
            'spread_max': spread_arr.max(),
            'spread_trend': np.polyfit(np.arange(len(spread_arr)), spread_arr, 1)[0] if len(spread_arr) > 1 else 0,

            # ── TRADE INTENSITY ──
            'trade_count_sum': tc_arr.sum(),
            'trade_count_trend': np.polyfit(np.arange(len(tc_arr)), tc_arr, 1)[0] if len(tc_arr) > 1 else 0,

            # ── VOLATILITY STRUCTURE ──
            'realized_vol': np.std(ret_arr) * np.sqrt(60) if len(ret_arr) > 1 else 0,  # annualized hourly
            'vol_of_vol': np.std(np.abs(ret_arr)) if len(ret_arr) > 1 else 0,
            'up_vol': np.std(ret_arr[ret_arr > 0]) if np.sum(ret_arr > 0) > 1 else 0,
            'down_vol': np.std(ret_arr[ret_arr < 0]) if np.sum(ret_arr < 0) > 1 else 0,
            'vol_asymmetry': 0,  # filled below

            # ── VWAP BEHAVIOR ──
            'vwap_dev_mean': group['vwap_dev'].mean(),
            'vwap_dev_trend': np.polyfit(np.arange(len(group)), group['vwap_dev'].values, 1)[0] if len(group) > 1 else 0,

            # ── VOL REGIME ──
            'vol_regime_mode': group['vol_regime'].mode().iloc[0] if len(group) > 0 else 'med',
        }

        # Vol asymmetry (down vol > up vol = bearish structure)
        if rec['up_vol'] > 0 and rec['down_vol'] > 0:
            rec['vol_asymmetry'] = rec['down_vol'] / rec['up_vol'] - 1
        elif rec['down_vol'] > 0:
            rec['vol_asymmetry'] = 1.0
        elif rec['up_vol'] > 0:
            rec['vol_asymmetry'] = -1.0

        hourly_records.append(rec)

    result = pd.DataFrame(hourly_records)

    # Encode vol_regime as numeric
    regime_map = {'low': 0, 'med': 1, 'high': 2}
    result['vol_regime_num'] = result['vol_regime_mode'].map(regime_map).fillna(1)

    log.info(f"Computed {len(result)} hourly bars with {len(result.columns)} features")
    return result


def add_rolling_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-hour rolling features that capture positioning buildup
    across hours — the key signal for "where are they positioned?"
    """
    df = df.sort_values('ts').reset_index(drop=True)

    # Multi-hour OFI momentum (cumulative positioning over 2h, 4h, full day)
    for window in [2, 4, 6]:
        label = f'{window}h'
        df[f'ofi_sum_{label}'] = df['ofi_sum'].rolling(window, min_periods=1).sum()
        df[f'ofi_trend_{label}'] = df['ofi_trend'].rolling(window, min_periods=1).mean()
        df[f'sv_sum_{label}'] = df['signed_volume_sum'].rolling(window, min_periods=1).sum()

        # Volume profile change
        df[f'volume_ma_{label}'] = df['total_volume'].rolling(window, min_periods=1).mean()
        df[f'volume_vs_ma_{label}'] = df['total_volume'] / df[f'volume_ma_{label}'].clip(lower=1)

        # Volatility trend
        df[f'vol_trend_{label}'] = df['realized_vol'].rolling(window, min_periods=1).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        )

    # Cross-day features (previous day's closing characteristics)
    # Group by date and take last hour of each day
    df['prev_day_ofi'] = df.groupby('date')['ofi_sum'].transform('sum').shift(1)
    df['prev_day_return'] = df.groupby('date')['return_1h'].transform('sum').shift(1)
    df['prev_day_sv'] = df.groupby('date')['signed_volume_sum'].transform('sum').shift(1)

    # Intraday time features (positioning changes through the day)
    df['hours_since_open'] = df.groupby('date').cumcount()
    df['is_first_hour'] = (df['hours_since_open'] == 0).astype(int)
    df['is_last_hour'] = df.groupby('date')['hours_since_open'].transform('max') == df['hours_since_open']
    df['is_last_hour'] = df['is_last_hour'].astype(int)

    # Cumulative intraday flow (building position throughout the day)
    df['intraday_cum_ofi'] = df.groupby('date')['ofi_sum'].cumsum()
    df['intraday_cum_sv'] = df.groupby('date')['signed_volume_sum'].cumsum()
    df['intraday_cum_return'] = df.groupby('date')['return_1h'].cumsum()

    # Flow reversal detection: did OFI flip sign from previous hour?
    df['ofi_sign_change'] = (np.sign(df['ofi_sum']) != np.sign(df['ofi_sum'].shift(1))).astype(int)

    # Absorption proxy: high volume + small price change = absorbed flow
    df['absorption_score'] = df['total_volume'] / (df['range_ticks'].clip(lower=1) * 100)

    log.info(f"Added rolling context: {len(df.columns)} total features")
    return df


# ═════════════════════════════════════════════
#  STEP 2: FORWARD LABELS
# ═════════════════════════════════════════════

def add_forward_labels(df: pd.DataFrame, horizons: Dict[str, int]) -> pd.DataFrame:
    """
    Add forward-looking return labels at multiple horizons.

    horizons: {'1h': 1, '2h': 2, '4h': 4, 'eod': -1}
    Values are number of hours ahead. -1 = remaining session (end of day).
    """
    df = df.sort_values('ts').reset_index(drop=True)

    for label, h in horizons.items():
        if h > 0 and label != '4h':
            # Simple N-hour forward return (within same day)
            df[f'fwd_return_{label}'] = df.groupby('date')['close'].shift(-h) / df['close'] - 1
            df[f'fwd_ticks_{label}'] = (df.groupby('date')['close'].shift(-h) - df['close']) / 0.25
        elif label == '4h':
            # 4h can span into next day — use absolute position not groupby
            # Sort by time, shift by 4 bars regardless of day boundary
            df_sorted = df.sort_values('ts')
            fwd_close = df_sorted['close'].shift(-h).values
            df[f'fwd_return_{label}'] = fwd_close / df['close'].values - 1
            df[f'fwd_ticks_{label}'] = (fwd_close - df['close'].values) / 0.25
            # Null out overnight gaps (if >16h between bars, it's a gap)
            ts_diff = df_sorted['ts'].diff(-h).abs().dt.total_seconds()
            df.loc[ts_diff > 16 * 3600, f'fwd_return_{label}'] = np.nan
            df.loc[ts_diff > 16 * 3600, f'fwd_ticks_{label}'] = np.nan
        else:
            # EOD: return from current close to day's last close
            day_close = df.groupby('date')['close'].transform('last')
            df[f'fwd_return_{label}'] = day_close / df['close'] - 1
            df[f'fwd_ticks_{label}'] = (day_close - df['close']) / 0.25

        # Direction labels: +1 (bullish), -1 (bearish), 0 (flat/noise)
        min_ticks = MIN_EDGE_TICKS.get(label, 3.0)
        fwd = df[f'fwd_ticks_{label}']
        df[f'direction_{label}'] = 0
        df.loc[fwd > min_ticks, f'direction_{label}'] = 1
        df.loc[fwd < -min_ticks, f'direction_{label}'] = -1

        # Also store binary (up/down, excluding flat for now)
        df[f'up_{label}'] = (fwd > 0).astype(int)

    return df


# ═════════════════════════════════════════════
#  STEP 3: MACRO REGIME FEATURES
# ═════════════════════════════════════════════

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add macro/regime features from feature store if available."""
    regime_path = REGIME_DIR / "regime_features.parquet"
    if not regime_path.exists():
        log.warning("No regime features found — using price-derived regime only")
        # Build simple regime from price data
        df_daily = df.groupby('date').agg(
            day_close=('close', 'last'),
            day_vol=('realized_vol', 'mean'),
        ).reset_index()

        df_daily['ma_20d'] = df_daily['day_close'].rolling(20, min_periods=5).mean()
        df_daily['ma_50d'] = df_daily['day_close'].rolling(50, min_periods=10).mean()
        df_daily['above_ma20'] = (df_daily['day_close'] > df_daily['ma_20d']).astype(int)
        df_daily['above_ma50'] = (df_daily['day_close'] > df_daily['ma_50d']).astype(int)
        df_daily['trend_20d'] = df_daily['day_close'].pct_change(20)
        df_daily['vol_20d'] = df_daily['day_vol'].rolling(20, min_periods=5).mean()

        # Merge back
        df = df.merge(
            df_daily[['date', 'above_ma20', 'above_ma50', 'trend_20d', 'vol_20d']],
            on='date', how='left'
        )
    else:
        log.info("Loading macro regime features")
        regime_df = pd.read_parquet(regime_path)
        if 'date' in regime_df.columns:
            # Coerce date types to match (string YYYYMMDD in hourly_df)
            regime_df['date'] = regime_df['date'].astype(str).str.replace('-', '')

            # Drop any existing regime columns to prevent _x/_y suffixes
            regime_cols = [c for c in regime_df.columns if c != 'date']
            existing = [c for c in regime_cols if c in df.columns]
            if existing:
                df = df.drop(columns=existing)

            df = df.merge(regime_df, on='date', how='left')
            log.info(f"Merged {len(regime_cols)} regime features")

            # Build derived regime columns for stratification
            if 'regime_vix_lag1' in df.columns:
                vix = df.groupby('date')['regime_vix_lag1'].first()
                # Bull/bear from price trend: use daily close change
                day_close = df.groupby('date')['close'].last()
                day_ma20 = day_close.rolling(20, min_periods=5).mean()
                above_ma = (day_close > day_ma20).astype(int).to_dict()
                df['above_ma20'] = df['date'].map(above_ma)

                # Also add VIX regime
                vix_dict = vix.to_dict()
                df['vix_regime'] = df['date'].map(vix_dict)
                df['high_vix'] = (df['vix_regime'] > 20).astype(int)
        else:
            log.warning("Regime features have no date column — skipping")

    return df


# ═════════════════════════════════════════════
#  STEP 4: WALK-FORWARD LIGHTGBM TRAINING
# ═════════════════════════════════════════════

def get_feature_columns(df: pd.DataFrame, clean_mode: bool = False) -> List[str]:
    """Get all feature columns (exclude labels, metadata).

    clean_mode=True: remove price-level features (open/high/low/close)
    that could act as look-ahead proxies. Keep only microstructure + regime.
    """
    exclude_prefixes = ('fwd_', 'direction_', 'up_', 'date', 'ts', 'vol_regime_mode')
    # Price-level features that trivially correlate with future prices
    # (we want microstructure FLOW features, not price continuation)
    price_level_cols = {'open', 'high', 'low', 'close', 'intraday_cum_return'}
    cols = [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)]
    if clean_mode:
        cols = [c for c in cols if c not in price_level_cols]
    return cols
    return [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)]


def train_walk_forward(
    df: pd.DataFrame,
    target_col: str,
    train_days: int = 60,
    test_days: int = 1,
    horizons_to_train: List[str] = None,
    clean_mode: bool = False,
) -> Dict:
    """
    Sliding walk-forward training with LightGBM.

    HC #0: SLIDING window only, NEVER expanding.
    HC #428 R1: Report per-regime results.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.error("LightGBM not installed. Install with: pip install lightgbm")
        return {}

    feature_cols = get_feature_columns(df, clean_mode=clean_mode)
    log.info(f"Training with {len(feature_cols)} features (clean={clean_mode}), target={target_col}")

    # Get unique dates
    dates = sorted(df['date'].unique())
    log.info(f"Date range: {dates[0]} → {dates[-1]} ({len(dates)} days)")

    if len(dates) < train_days + test_days + 5:
        log.error(f"Not enough days ({len(dates)}) for {train_days}+{test_days} walk-forward")
        return {}

    # LightGBM params (conservative, avoid overfitting)
    params = {
        'objective': 'regression',  # predict forward ticks, not classification
        'metric': 'mse',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'n_jobs': 8,
        'seed': 42,
    }

    all_preds = []
    all_actuals = []
    all_dates = []
    fold_results = []
    importances = np.zeros(len(feature_cols))

    start_idx = train_days  # first test fold starts after train_days

    for fold_start in range(start_idx, len(dates) - test_days + 1, test_days):
        train_dates = dates[fold_start - train_days : fold_start]
        test_dates = dates[fold_start : fold_start + test_days]

        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'].isin(test_dates)

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values

        # Drop NaN targets
        train_valid = ~np.isnan(y_train)
        test_valid = ~np.isnan(y_test)

        if train_valid.sum() < 50 or test_valid.sum() < 3:
            continue

        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]

        # Replace NaN/inf in features
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_test)

        # Track feature importance
        importances += model.feature_importance(importance_type='gain')

        # Record
        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_dates.extend([test_dates[0]] * len(preds))

        # Per-fold stats
        ic = np.corrcoef(preds, y_test)[0, 1] if len(preds) > 5 else 0
        fold_results.append({
            'test_date': test_dates[0],
            'n_samples': len(preds),
            'ic': ic,
            'pred_mean': preds.mean(),
            'actual_mean': y_test.mean(),
        })

        if len(fold_results) % 20 == 0:
            recent_ic = np.mean([f['ic'] for f in fold_results[-20:]])
            log.info(f"Fold {len(fold_results)}: test={test_dates[0]}, "
                     f"IC={ic:.3f}, recent_20_IC={recent_ic:.3f}")

    if not all_preds:
        log.error("No valid folds produced")
        return {}

    # ── Aggregate Results ──
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    all_dates = np.array(all_dates)

    # Overall IC
    overall_ic = np.corrcoef(all_preds, all_actuals)[0, 1]
    rank_ic = stats.spearmanr(all_preds, all_actuals)[0]

    # Per-day IC
    daily_ics = []
    for d in np.unique(all_dates):
        mask = all_dates == d
        if mask.sum() > 3:
            ic = np.corrcoef(all_preds[mask], all_actuals[mask])[0, 1]
            daily_ics.append({'date': d, 'ic': ic})

    daily_ic_df = pd.DataFrame(daily_ics)

    # Directional accuracy
    pred_direction = np.sign(all_preds)
    actual_direction = np.sign(all_actuals)
    directional_accuracy = np.mean(pred_direction == actual_direction)

    # Top/bottom quantile performance (the money signal)
    q_thresholds = [0.1, 0.2, 0.3]
    quantile_results = {}
    for q in q_thresholds:
        top_mask = all_preds >= np.quantile(all_preds, 1 - q)
        bottom_mask = all_preds <= np.quantile(all_preds, q)

        top_actual = all_actuals[top_mask].mean()
        bottom_actual = all_actuals[bottom_mask].mean()
        long_short = top_actual - bottom_actual

        quantile_results[f'top_{int(q*100)}pct'] = {
            'mean_actual_ticks': top_actual,
            'count': top_mask.sum(),
            'win_rate': np.mean(all_actuals[top_mask] > 0),
        }
        quantile_results[f'bottom_{int(q*100)}pct'] = {
            'mean_actual_ticks': bottom_actual,
            'count': bottom_mask.sum(),
            'win_rate': np.mean(all_actuals[bottom_mask] < 0),
        }
        quantile_results[f'long_short_{int(q*100)}pct'] = long_short

    # Feature importance (top 20)
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances,
    }).sort_values('importance', ascending=False)

    # Regime-stratified analysis
    regime_results = {}
    if 'above_ma20' in df.columns:
        for d in np.unique(all_dates):
            mask = all_dates == d
            date_df = df[df['date'] == d]
            if len(date_df) > 0 and 'above_ma20' in date_df.columns:
                # Get regime for this day
                regime_val = date_df['above_ma20'].iloc[0]
                regime_key = 'bull' if regime_val == 1 else 'bear'
                if regime_key not in regime_results:
                    regime_results[regime_key] = {'preds': [], 'actuals': []}
                regime_results[regime_key]['preds'].extend(all_preds[mask])
                regime_results[regime_key]['actuals'].extend(all_actuals[mask])

    for key in regime_results:
        p = np.array(regime_results[key]['preds'])
        a = np.array(regime_results[key]['actuals'])
        if len(p) > 10:
            regime_results[key] = {
                'ic': np.corrcoef(p, a)[0, 1],
                'dir_acc': np.mean(np.sign(p) == np.sign(a)),
                'n': len(p),
            }

    results = {
        'target': target_col,
        'n_folds': len(fold_results),
        'n_predictions': len(all_preds),
        'overall_ic': overall_ic,
        'rank_ic': rank_ic,
        'directional_accuracy': directional_accuracy,
        'daily_ic_mean': daily_ic_df['ic'].mean() if len(daily_ic_df) > 0 else 0,
        'daily_ic_std': daily_ic_df['ic'].std() if len(daily_ic_df) > 0 else 0,
        'daily_ic_sharpe': (daily_ic_df['ic'].mean() / max(daily_ic_df['ic'].std(), 1e-6)) if len(daily_ic_df) > 0 else 0,
        'quantile_results': quantile_results,
        'regime_results': regime_results,
        'top_features': fi_df.head(20).to_dict('records'),
        'fold_results': fold_results,
        # Raw arrays for trading simulation
        'all_preds': all_preds,
        'all_actuals': all_actuals,
        'all_dates': all_dates,
    }

    return results


# ═════════════════════════════════════════════
#  STEP 5: TRADING SIMULATION
# ═════════════════════════════════════════════

def simulate_trades(
    preds: np.ndarray,
    actuals: np.ndarray,
    confidence_threshold: float = 0.7,  # top/bottom 30%
    cost_ticks: float = COST_RT_TICKS,
) -> Dict:
    """
    Simulate trades based on prediction confidence.
    Only trade when prediction exceeds confidence threshold quantile.
    """
    if len(preds) == 0:
        return {}

    upper = np.quantile(preds, 1 - confidence_threshold)
    lower = np.quantile(preds, confidence_threshold)

    # Go long when prediction is strongly positive, short when strongly negative
    trades = []
    for i in range(len(preds)):
        if preds[i] >= upper:
            pnl_ticks = actuals[i] - cost_ticks
            trades.append({'direction': 'long', 'pnl_ticks': pnl_ticks, 'raw_ticks': actuals[i]})
        elif preds[i] <= lower:
            pnl_ticks = -actuals[i] - cost_ticks
            trades.append({'direction': 'short', 'pnl_ticks': pnl_ticks, 'raw_ticks': -actuals[i]})

    if not trades:
        return {}

    trade_df = pd.DataFrame(trades)
    pnl = trade_df['pnl_ticks'].values
    cum_pnl = np.cumsum(pnl)

    # Risk-adjusted metrics
    sharpe = pnl.mean() / max(pnl.std(), 1e-6) * np.sqrt(252)  # annualized
    sortino_denom = np.sqrt(np.mean(np.minimum(pnl, 0) ** 2))
    sortino = pnl.mean() / max(sortino_denom, 1e-6) * np.sqrt(252)

    win_rate = np.mean(pnl > 0)
    pf = np.sum(pnl[pnl > 0]) / max(-np.sum(pnl[pnl < 0]), 1e-6)
    max_dd = np.min(cum_pnl - np.maximum.accumulate(cum_pnl))

    # Long vs short breakdown
    long_trades = trade_df[trade_df['direction'] == 'long']
    short_trades = trade_df[trade_df['direction'] == 'short']

    return {
        'n_trades': len(trades),
        'total_pnl_ticks': pnl.sum(),
        'total_pnl_dollars': pnl.sum() * ES_TICK_VALUE,
        'avg_pnl_ticks': pnl.mean(),
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': win_rate,
        'profit_factor': pf,
        'max_dd_ticks': max_dd,
        'max_dd_dollars': max_dd * ES_TICK_VALUE,
        'long_trades': len(long_trades),
        'long_wr': long_trades['pnl_ticks'].apply(lambda x: x > 0).mean() if len(long_trades) > 0 else 0,
        'long_avg': long_trades['pnl_ticks'].mean() if len(long_trades) > 0 else 0,
        'short_trades': len(short_trades),
        'short_wr': short_trades['pnl_ticks'].apply(lambda x: x > 0).mean() if len(short_trades) > 0 else 0,
        'short_avg': short_trades['pnl_ticks'].mean() if len(short_trades) > 0 else 0,
    }


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Longer-Horizon Directional Model v1')
    parser.add_argument('--train-days', type=int, default=60, help='Training window (days)')
    parser.add_argument('--test-days', type=int, default=1, help='Test window (days)')
    parser.add_argument('--horizons', nargs='+', default=['1h', '2h', '4h', 'eod'],
                        help='Horizons to test')
    parser.add_argument('--mlflow', action='store_true', help='Log to MLflow')
    parser.add_argument('--clean', action='store_true',
                        help='Clean mode: exclude price-level features, isolate microstructure signal')
    parser.add_argument('--experiment-name', default='longer_horizon_v1')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("LONGER-HORIZON DIRECTIONAL MODEL v1 (HC #637)")
    log.info("=" * 60)

    # MLflow setup
    mlflow = None
    if args.mlflow:
        try:
            import mlflow as _mlflow
            mlflow = _mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment(args.experiment_name)
            suffix = '_clean' if args.clean else ''
            mlflow.start_run(run_name=f"lh_v1{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_params({
                'train_days': args.train_days,
                'test_days': args.test_days,
                'horizons': ','.join(args.horizons),
                'model': 'lightgbm',
                'cost_rt_ticks': COST_RT_TICKS,
                'clean_mode': args.clean,
            })
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")
            mlflow = None

    # ── Load data ──
    log.info("Step 1: Loading minute bars...")
    minute_df = load_all_minute_bars()

    # ── Compute hourly features ──
    log.info("Step 2: Computing hourly features...")
    hourly_df = compute_hourly_features(minute_df)
    del minute_df
    gc.collect()

    # ── Add rolling context ──
    log.info("Step 3: Adding multi-hour rolling context...")
    hourly_df = add_rolling_context(hourly_df)

    # ── Add macro features ──
    log.info("Step 4: Adding macro regime features...")
    hourly_df = add_macro_features(hourly_df)

    # ── Add forward labels ──
    horizon_map = {'1h': 1, '2h': 2, '4h': 4, 'eod': -1}
    horizons_to_use = {h: horizon_map[h] for h in args.horizons if h in horizon_map}
    log.info(f"Step 5: Computing forward labels for {list(horizons_to_use.keys())}...")
    hourly_df = add_forward_labels(hourly_df, horizons_to_use)

    # Save processed dataset
    hourly_df.to_parquet(OUTPUT_DIR / "hourly_features_labeled.parquet", index=False)
    log.info(f"Saved processed dataset: {len(hourly_df)} rows, {len(hourly_df.columns)} cols")

    # ── Train per horizon ──
    all_results = {}
    for horizon_label in args.horizons:
        target = f'fwd_ticks_{horizon_label}'
        if target not in hourly_df.columns:
            continue

        log.info(f"\n{'='*60}")
        log.info(f"TRAINING: {horizon_label} horizon (target={target})")
        log.info(f"{'='*60}")

        results = train_walk_forward(
            hourly_df,
            target_col=target,
            train_days=args.train_days,
            test_days=args.test_days,
            clean_mode=args.clean,
        )

        if not results:
            continue

        all_results[horizon_label] = results

        # ── TRADING SIMULATION ──
        sim_preds = np.array(results['all_preds'])
        sim_actuals = np.array(results['all_actuals'])
        log.info(f"\n── {horizon_label} TRADING SIMULATION ──")
        for conf_label, conf_thresh in [('top_10pct', 0.10), ('top_20pct', 0.20), ('top_30pct', 0.30)]:
            sim = simulate_trades(sim_preds, sim_actuals, confidence_threshold=conf_thresh)
            if sim:
                log.info(f"  {conf_label} filter:")
                log.info(f"    Trades: {sim['n_trades']} | Sharpe: {sim['sharpe']:.2f} | "
                         f"Sortino: {sim['sortino']:.2f} | WR: {sim['win_rate']:.1%} | "
                         f"PF: {sim['profit_factor']:.2f}")
                log.info(f"    Total PnL: {sim['total_pnl_ticks']:.0f} ticks (${sim['total_pnl_dollars']:,.0f}) | "
                         f"Avg: {sim['avg_pnl_ticks']:.1f} tk/trade | MaxDD: {sim['max_dd_ticks']:.0f} ticks")
                log.info(f"    Long: {sim['long_trades']} trades WR={sim['long_wr']:.1%} avg={sim['long_avg']:.1f}tk | "
                         f"Short: {sim['short_trades']} trades WR={sim['short_wr']:.1%} avg={sim['short_avg']:.1f}tk")
                results[f'sim_{conf_label}'] = sim
                if mlflow:
                    mlflow.log_metrics({
                        f'{horizon_label}_{conf_label}_sharpe': sim['sharpe'],
                        f'{horizon_label}_{conf_label}_sortino': sim['sortino'],
                        f'{horizon_label}_{conf_label}_wr': sim['win_rate'],
                        f'{horizon_label}_{conf_label}_pf': sim['profit_factor'],
                        f'{horizon_label}_{conf_label}_trades': sim['n_trades'],
                        f'{horizon_label}_{conf_label}_pnl_ticks': sim['total_pnl_ticks'],
                    })

        log.info(f"\n── {horizon_label} SIGNAL QUALITY ──")
        log.info(f"  Overall IC:     {results['overall_ic']:.4f}")
        log.info(f"  Rank IC:        {results['rank_ic']:.4f}")
        log.info(f"  Daily IC mean:  {results['daily_ic_mean']:.4f} ± {results['daily_ic_std']:.4f}")
        log.info(f"  IC Sharpe:      {results['daily_ic_sharpe']:.2f}")
        log.info(f"  Dir Accuracy:   {results['directional_accuracy']:.1%}")
        log.info(f"  Folds:          {results['n_folds']}")

        # Quantile breakdown
        for key, val in results['quantile_results'].items():
            if isinstance(val, dict):
                log.info(f"  {key}: mean={val['mean_actual_ticks']:.2f}tk, "
                         f"WR={val['win_rate']:.1%}, n={val['count']}")
            else:
                log.info(f"  {key}: {val:.2f}tk spread")

        # Regime breakdown
        for key, val in results['regime_results'].items():
            if isinstance(val, dict) and 'ic' in val:
                log.info(f"  Regime {key}: IC={val['ic']:.4f}, "
                         f"DirAcc={val['dir_acc']:.1%}, n={val['n']}")

        # Top features
        log.info(f"  Top 10 features:")
        for i, feat in enumerate(results['top_features'][:10]):
            log.info(f"    {i+1}. {feat['feature']}: {feat['importance']:.0f}")

        if mlflow:
            mlflow.log_metrics({
                f'{horizon_label}_ic': results['overall_ic'],
                f'{horizon_label}_rank_ic': results['rank_ic'],
                f'{horizon_label}_daily_ic_mean': results['daily_ic_mean'],
                f'{horizon_label}_ic_sharpe': results['daily_ic_sharpe'],
                f'{horizon_label}_dir_acc': results['directional_accuracy'],
            })

    # ── Save summary ──
    summary = {
        'run_time': datetime.now().isoformat(),
        'config': {
            'train_days': args.train_days,
            'test_days': args.test_days,
            'horizons': args.horizons,
            'cost_rt_ticks': COST_RT_TICKS,
        },
        'results': {},
    }
    for h, r in all_results.items():
        summary['results'][h] = {
            'overall_ic': r['overall_ic'],
            'rank_ic': r['rank_ic'],
            'daily_ic_mean': r['daily_ic_mean'],
            'daily_ic_sharpe': r['daily_ic_sharpe'],
            'directional_accuracy': r['directional_accuracy'],
            'quantile_results': r['quantile_results'],
            'regime_results': r['regime_results'],
        }

    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\nSaved summary to {OUTPUT_DIR / 'summary.json'}")

    if mlflow:
        mlflow.log_artifact(str(OUTPUT_DIR / "summary.json"))
        mlflow.end_run()

    # ── Final Report ──
    log.info("\n" + "=" * 60)
    log.info("FINAL SUMMARY — LONGER-HORIZON DIRECTIONAL MODEL v1")
    log.info("=" * 60)
    for h in args.horizons:
        if h in all_results:
            r = all_results[h]
            log.info(f"  {h}: IC={r['overall_ic']:.4f}, RankIC={r['rank_ic']:.4f}, "
                     f"DirAcc={r['directional_accuracy']:.1%}, ICsharpe={r['daily_ic_sharpe']:.2f}")
        else:
            log.info(f"  {h}: NO RESULTS")

    return all_results


if __name__ == '__main__':
    main()
