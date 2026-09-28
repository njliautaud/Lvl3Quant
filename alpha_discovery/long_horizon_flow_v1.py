#!/usr/bin/env python3
"""
long_horizon_flow_v1.py — Daily/multi-hour directional model using accumulated orderflow.

Uses 197 days of minute bars to build session-level + multi-day rolling features,
then predicts forward returns at 1h / half-day / 1d / 2d / 3d horizons.

Walk-forward: 40-day sliding train, 5-day OOT, 5-day slide.
Model: LightGBM regression.
Trading sim: Daily position, Intraday position, Multi-day trend following.

HC #0: SLIDING window only. NO expanding.
HC #428: Regime-agnostic validation, MFE-within-horizon checks.
"""

import os, sys, json, logging, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──
DATA_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_minute_bars_v1")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/long_horizon_flow_v1")
LOG_FILE = Path("/home/nick/Lvl3Quant/logs/long_horizon_flow_v1.log")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376  # $4.70 / $12.50
SLIPPAGE_TICKS = 1.0  # market order slippage each side
COST_RT_TICKS = 2 * (SLIPPAGE_TICKS + COMMISSION_RT_TICKS / 2)  # = 2.376

# Walk-forward params
TRAIN_DAYS = 40
OOT_DAYS = 5
SLIDE_DAYS = 5

# Session split: AM = first 225 min (9:30-1:15 ET), PM = last 225 min
AM_MINUTES = 225  # first half of ~450 min session
FIRST_HOUR_MINUTES = 60
FIRST_HALF_MINUTES = 195  # 9:30 - 12:45 ET (3.25 hrs)


def load_all_minute_bars():
    """Load all parquet files and return sorted by date."""
    files = sorted(DATA_DIR.glob("*.parquet"))
    log.info(f"Found {len(files)} daily files in {DATA_DIR}")
    
    all_data = {}
    for f in files:
        date_str = f.stem  # e.g. '20250714'
        df = pd.read_parquet(f)
        df = df.sort_values('ts_minute').reset_index(drop=True)
        all_data[date_str] = df
    
    log.info(f"Loaded {len(all_data)} days: {min(all_data.keys())} to {max(all_data.keys())}")
    return all_data


def compute_session_features(day_df):
    """Compute session-level features from minute bars for ONE day."""
    n = len(day_df)
    am_end = min(AM_MINUTES, n)
    
    am_df = day_df.iloc[:am_end]
    pm_df = day_df.iloc[am_end:]
    
    o = day_df['open'].iloc[0]
    c = day_df['close'].iloc[-1]
    h = day_df['high'].max()
    l = day_df['low'].min()
    
    # VWAP for the session
    if 'vwap' in day_df.columns and day_df['volume'].sum() > 0:
        session_vwap = (day_df['vwap'] * day_df['volume']).sum() / day_df['volume'].sum()
    else:
        session_vwap = c
    
    # Volume concentration: max 30-min volume / avg 30-min volume
    vol_30min = day_df['volume'].rolling(30, min_periods=1).sum()
    avg_30min_vol = day_df['volume'].sum() / max(1, n / 30)
    vol_concentration = vol_30min.max() / max(1, avg_30min_vol)
    
    # First 2 hours return (momentum_am)
    first_2h_end = min(120, n)
    momentum_am = (day_df['close'].iloc[first_2h_end - 1] - o) / TICK_SIZE if first_2h_end > 0 else 0
    
    # Last 2 hours return (momentum_pm)
    last_2h_start = max(0, n - 120)
    momentum_pm = (c - day_df['open'].iloc[last_2h_start]) / TICK_SIZE if last_2h_start < n else 0
    
    feats = {
        'session_ofi': day_df['ofi_1min'].sum(),
        'session_signed_volume': day_df['signed_volume'].sum(),
        'session_volume': day_df['volume'].sum(),
        'am_ofi': am_df['ofi_1min'].sum() if len(am_df) > 0 else 0,
        'pm_ofi': pm_df['ofi_1min'].sum() if len(pm_df) > 0 else 0,
        'ofi_trend': (pm_df['ofi_1min'].sum() - am_df['ofi_1min'].sum()) if len(pm_df) > 0 and len(am_df) > 0 else 0,
        'vwap_close_deviation': (c - session_vwap) / TICK_SIZE,
        'price_range_ticks': (h - l) / TICK_SIZE,
        'close_vs_open_ticks': (c - o) / TICK_SIZE,
        'volume_concentration': vol_concentration,
        'spread_mean': day_df['spread_mean'].mean(),
        'high_close_pct': (c - l) / max(TICK_SIZE, h - l),  # where in range did we close
        'momentum_am': momentum_am,
        'momentum_pm': momentum_pm,
        'open': o,
        'close': c,
        'high': h,
        'low': l,
        'session_vwap': session_vwap,
        'trade_count': day_df['trade_count'].sum(),
    }
    
    return feats


def build_daily_dataframe(all_data):
    """Build a daily-level DataFrame with session features."""
    records = []
    dates = sorted(all_data.keys())
    
    for date_str in dates:
        day_df = all_data[date_str]
        if len(day_df) < 60:  # skip very short sessions
            log.warning(f"Skipping {date_str}: only {len(day_df)} bars")
            continue
        feats = compute_session_features(day_df)
        feats['date'] = date_str
        records.append(feats)
    
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)
    log.info(f"Built daily DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df


def add_rolling_features(df):
    """Add multi-day rolling features. CRITICAL: only use data up to row i (no leakage)."""
    
    # Rolling OFI sums
    df['ofi_3d'] = df['session_ofi'].rolling(3, min_periods=3).sum()
    df['ofi_5d'] = df['session_ofi'].rolling(5, min_periods=5).sum()
    df['ofi_10d'] = df['session_ofi'].rolling(10, min_periods=10).sum()
    
    # Rolling signed volume
    df['signed_vol_3d'] = df['session_signed_volume'].rolling(3, min_periods=3).sum()
    df['signed_vol_5d'] = df['session_signed_volume'].rolling(5, min_periods=5).sum()
    
    # Cumulative returns in ticks
    df['return_3d'] = df['close_vs_open_ticks'].rolling(3, min_periods=3).sum()
    df['return_5d'] = df['close_vs_open_ticks'].rolling(5, min_periods=5).sum()
    df['return_10d'] = df['close_vs_open_ticks'].rolling(10, min_periods=10).sum()
    
    # Close-to-close returns (more accurate for multi-day)
    df['cc_return_ticks'] = df['close'].diff() / TICK_SIZE
    df['cc_return_3d'] = df['cc_return_ticks'].rolling(3, min_periods=3).sum()
    df['cc_return_5d'] = df['cc_return_ticks'].rolling(5, min_periods=5).sum()
    
    # Volatility regime: 5-day rolling std of close-to-close returns
    df['vol_regime_5d'] = df['cc_return_ticks'].rolling(5, min_periods=5).std()
    
    # OFI direction streak
    ofi_sign = np.sign(df['session_ofi'])
    streaks = []
    streak = 0
    for s in ofi_sign:
        if s == 0:
            streak = 0
        elif len(streaks) == 0:
            streak = s
        elif np.sign(streak) == s:
            streak += s
        else:
            streak = s
        streaks.append(streak)
    df['ofi_direction_streak'] = streaks
    
    # OFI vs price divergence: OFI positive but price negative (or vice versa)
    # Use 3-day sums for both
    df['ofi_vs_price_divergence'] = np.sign(df['ofi_3d']) * np.sign(df['cc_return_3d'])
    # -1 = divergence, +1 = agreement, 0 = neutral
    
    # AM/PM consistency over 3 days
    am_sign_3d = np.sign(df['am_ofi']).rolling(3, min_periods=3).sum()
    pm_sign_3d = np.sign(df['pm_ofi']).rolling(3, min_periods=3).sum()
    df['am_pm_consistency_3d'] = am_sign_3d * pm_sign_3d  # high positive = consistent
    
    # Range expansion: is daily range expanding over 3 days?
    df['range_expansion_3d'] = df['price_range_ticks'].rolling(3, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / max(1, x.iloc[0]) if len(x) == 3 else 0, raw=False
    )
    
    # Volume trend: 5-day slope of volume
    df['volume_trend_5d'] = df['session_volume'].rolling(5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=False
    )
    
    # Normalized OFI (OFI per unit volume — intensity)
    df['ofi_intensity'] = df['session_ofi'] / df['session_volume'].clip(lower=1)
    df['ofi_intensity_3d'] = df['ofi_intensity'].rolling(3, min_periods=3).mean()
    
    # OFI acceleration: change in 3d OFI
    df['ofi_accel_3d'] = df['ofi_3d'].diff()
    
    log.info(f"Added rolling features. Total columns: {len(df.columns)}")
    return df


def add_forward_labels(df):
    """Add forward return labels. CRITICAL: fwd_return_1d uses NEXT day's close."""
    
    # Close-to-close forward returns
    df['fwd_return_1d'] = df['cc_return_ticks'].shift(-1)
    df['fwd_return_2d'] = df['cc_return_ticks'].shift(-1).rolling(2, min_periods=2).sum().shift(-1)
    # Actually: fwd_return_2d = (close[t+2] - close[t]) / tick
    # Let's compute properly
    df['fwd_return_2d'] = (df['close'].shift(-2) - df['close']) / TICK_SIZE
    df['fwd_return_3d'] = (df['close'].shift(-3) - df['close']) / TICK_SIZE
    
    # For intraday labels, we need next-day minute data — store separately
    # fwd_return_1h: next day's first hour return (open to close of first 60 min)
    # fwd_return_halfday: next day's first half return
    # These will be computed separately using minute data
    
    log.info(f"Added forward labels. Non-null 1d labels: {df['fwd_return_1d'].notna().sum()}")
    return df


def add_intraday_forward_labels(df, all_data):
    """Add intraday forward labels using minute bars from NEXT day."""
    dates_list = sorted(all_data.keys())
    date_to_idx = {d: i for i, d in enumerate(dates_list)}
    
    fwd_1h = []
    fwd_halfday = []
    
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y%m%d')
        idx = date_to_idx.get(date_str)
        
        if idx is None or idx + 1 >= len(dates_list):
            fwd_1h.append(np.nan)
            fwd_halfday.append(np.nan)
            continue
        
        next_date = dates_list[idx + 1]
        next_df = all_data[next_date]
        
        if len(next_df) < FIRST_HOUR_MINUTES:
            fwd_1h.append(np.nan)
            fwd_halfday.append(np.nan)
            continue
        
        next_open = next_df['open'].iloc[0]
        
        # First hour return
        h1_close = next_df['close'].iloc[min(FIRST_HOUR_MINUTES - 1, len(next_df) - 1)]
        fwd_1h.append((h1_close - next_open) / TICK_SIZE)
        
        # First half-day return
        hd_end = min(FIRST_HALF_MINUTES, len(next_df))
        hd_close = next_df['close'].iloc[hd_end - 1]
        fwd_halfday.append((hd_close - next_open) / TICK_SIZE)
    
    df['fwd_return_1h'] = fwd_1h
    df['fwd_return_halfday'] = fwd_halfday
    
    log.info(f"Added intraday labels. 1h non-null: {pd.Series(fwd_1h).notna().sum()}, halfday: {pd.Series(fwd_halfday).notna().sum()}")
    return df


def get_feature_cols():
    """Return list of feature column names."""
    return [
        'session_ofi', 'session_signed_volume', 'session_volume',
        'am_ofi', 'pm_ofi', 'ofi_trend',
        'vwap_close_deviation', 'price_range_ticks', 'close_vs_open_ticks',
        'volume_concentration', 'spread_mean', 'high_close_pct',
        'momentum_am', 'momentum_pm', 'trade_count',
        # Rolling features
        'ofi_3d', 'ofi_5d', 'ofi_10d',
        'signed_vol_3d', 'signed_vol_5d',
        'return_3d', 'return_5d', 'return_10d',
        'cc_return_3d', 'cc_return_5d',
        'vol_regime_5d',
        'ofi_direction_streak', 'ofi_vs_price_divergence',
        'am_pm_consistency_3d', 'range_expansion_3d', 'volume_trend_5d',
        'ofi_intensity', 'ofi_intensity_3d', 'ofi_accel_3d',
    ]


LABEL_COLS = ['fwd_return_1d', 'fwd_return_2d', 'fwd_return_3d', 'fwd_return_1h', 'fwd_return_halfday']


def walk_forward_train(df):
    """Sliding walk-forward with LightGBM. HC #0: SLIDING only."""
    feature_cols = get_feature_cols()
    results = {label: [] for label in LABEL_COLS}
    fold_details = []
    
    n = len(df)
    fold_id = 0
    
    # LightGBM params — tuned for small datasets
    lgb_params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 16,
        'max_depth': 4,
        'min_child_samples': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    
    start_idx = 0
    while start_idx + TRAIN_DAYS + OOT_DAYS <= n:
        train_end = start_idx + TRAIN_DAYS
        oot_end = min(train_end + OOT_DAYS, n)
        
        train_df = df.iloc[start_idx:train_end]
        oot_df = df.iloc[train_end:oot_end]
        
        fold_id += 1
        train_dates = f"{train_df['date'].iloc[0].strftime('%Y-%m-%d')} to {train_df['date'].iloc[-1].strftime('%Y-%m-%d')}"
        oot_dates = f"{oot_df['date'].iloc[0].strftime('%Y-%m-%d')} to {oot_df['date'].iloc[-1].strftime('%Y-%m-%d')}"
        
        for label in LABEL_COLS:
            # Get valid training rows (non-null label AND non-null features)
            train_mask = train_df[label].notna() & train_df[feature_cols].notna().all(axis=1)
            oot_mask = oot_df[label].notna() & oot_df[feature_cols].notna().all(axis=1)
            
            X_train = train_df.loc[train_mask, feature_cols].values
            y_train = train_df.loc[train_mask, label].values
            X_oot = oot_df.loc[oot_mask, feature_cols].values
            y_oot = oot_df.loc[oot_mask, label].values
            
            if len(X_train) < 15 or len(X_oot) < 1:
                continue
            
            # Train
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(lgb_params, dtrain, num_boost_round=200)
            
            # Predict OOT
            preds = model.predict(X_oot)
            
            for i, (pred, actual) in enumerate(zip(preds, y_oot)):
                oot_idx = oot_df.index[oot_mask][i]
                results[label].append({
                    'fold': fold_id,
                    'date': df.loc[oot_idx, 'date'],
                    'pred': pred,
                    'actual': actual,
                    'close': df.loc[oot_idx, 'close'],
                    'cc_return_ticks': df.loc[oot_idx, 'cc_return_ticks'] if 'cc_return_ticks' in df.columns else 0,
                })
        
        if fold_id % 5 == 0:
            log.info(f"Fold {fold_id}: train {train_dates}, OOT {oot_dates}")
        
        start_idx += SLIDE_DAYS
    
    log.info(f"Walk-forward complete: {fold_id} folds")
    
    # Also train a final model on last TRAIN_DAYS for feature importance
    final_train = df.iloc[-TRAIN_DAYS - OOT_DAYS:-OOT_DAYS]
    mask = final_train['fwd_return_1d'].notna() & final_train[feature_cols].notna().all(axis=1)
    if mask.sum() > 10:
        dtrain = lgb.Dataset(final_train.loc[mask, feature_cols].values, label=final_train.loc[mask, 'fwd_return_1d'].values)
        final_model = lgb.train(lgb_params, dtrain, num_boost_round=200)
        importance = dict(zip(feature_cols, final_model.feature_importance(importance_type='gain')))
    else:
        importance = {}
    
    return results, importance


def compute_ic_metrics(results):
    """Compute IC and IC Sharpe for each label."""
    ic_report = {}
    
    for label, recs in results.items():
        if len(recs) < 10:
            ic_report[label] = {'ic': np.nan, 'ic_sharpe': np.nan, 'n': len(recs)}
            continue
        
        res_df = pd.DataFrame(recs)
        
        # Overall IC (rank correlation)
        ic_overall = stats.spearmanr(res_df['pred'], res_df['actual'])[0]
        
        # Per-fold IC for IC Sharpe
        fold_ics = []
        for fold_id in res_df['fold'].unique():
            fold_df = res_df[res_df['fold'] == fold_id]
            if len(fold_df) >= 3:
                ic, _ = stats.spearmanr(fold_df['pred'], fold_df['actual'])
                if not np.isnan(ic):
                    fold_ics.append(ic)
        
        ic_mean = np.mean(fold_ics) if fold_ics else np.nan
        ic_std = np.std(fold_ics, ddof=1) if len(fold_ics) > 1 else np.nan
        ic_sharpe = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
        
        # Directional accuracy
        correct_dir = ((res_df['pred'] > 0) & (res_df['actual'] > 0)) | ((res_df['pred'] < 0) & (res_df['actual'] < 0))
        dir_acc = correct_dir.mean()
        
        ic_report[label] = {
            'ic_overall': round(ic_overall, 4),
            'ic_mean_fold': round(ic_mean, 4) if not np.isnan(ic_mean) else None,
            'ic_sharpe': round(ic_sharpe, 3) if not np.isnan(ic_sharpe) else None,
            'dir_accuracy': round(dir_acc, 4),
            'n_predictions': len(res_df),
            'n_folds': len(fold_ics),
            'pred_std': round(res_df['pred'].std(), 2),
            'actual_std': round(res_df['actual'].std(), 2),
        }
    
    return ic_report


def classify_regime(cc_return_ticks):
    """Classify day as green/red/flat based on close-to-close return."""
    if cc_return_ticks > 4:  # > 1 point up
        return 'green'
    elif cc_return_ticks < -4:
        return 'red'
    return 'flat'


def simulate_strategy_a(results, label='fwd_return_1d'):
    """Strategy A: Daily position, hold overnight.
    Enter at close, exit next day close. Cost = 2.376 ticks RT."""
    if label not in results or len(results[label]) < 10:
        return None
    
    res_df = pd.DataFrame(results[label]).sort_values('date').reset_index(drop=True)
    
    # Threshold sweep
    pred_std = res_df['pred'].std()
    thresholds = [0, 0.5 * pred_std, 1.0 * pred_std, 2.0 * pred_std]
    
    all_results = {}
    for thresh in thresholds:
        trades = []
        for _, row in res_df.iterrows():
            if row['pred'] > thresh:
                direction = 1  # long
            elif row['pred'] < -thresh:
                direction = -1  # short
            else:
                continue
            
            pnl_ticks = direction * row['actual'] - COST_RT_TICKS
            regime = classify_regime(row['cc_return_ticks'])
            
            trades.append({
                'date': row['date'],
                'direction': direction,
                'pred': row['pred'],
                'actual': row['actual'],
                'pnl_ticks': pnl_ticks,
                'pnl_dollars': pnl_ticks * TICK_VALUE,
                'regime': regime,
            })
        
        if len(trades) < 5:
            continue
        
        tdf = pd.DataFrame(trades)
        metrics = compute_trading_metrics(tdf, f"StratA_thresh={thresh:.1f}")
        metrics['threshold'] = round(thresh, 2)
        metrics['threshold_mult'] = round(thresh / pred_std, 1) if pred_std > 0 else 0
        all_results[f"thresh_{thresh:.1f}"] = metrics
    
    return all_results


def simulate_strategy_b(results, all_data, daily_df, label='fwd_return_1h'):
    """Strategy B: Intraday position, no overnight risk.
    Enter at open via limit, exit at 15:50 via market."""
    if label not in results or len(results[label]) < 10:
        return None
    
    res_df = pd.DataFrame(results[label]).sort_values('date').reset_index(drop=True)
    pred_std = res_df['pred'].std()
    thresholds = [0, 0.5 * pred_std, 1.0 * pred_std, 2.0 * pred_std]
    
    # For intraday, we need the full-day return for each predicted day
    # The pred is for next day's first hour. We use that to decide direction,
    # but hold all day (open to 15:50 close).
    # We need actual full-session returns from minute data.
    
    # Build a map of date -> full session return
    dates_list = sorted(all_data.keys())
    date_returns = {}
    for date_str in dates_list:
        day_df = all_data[date_str]
        if len(day_df) < 60:
            continue
        session_open = day_df['open'].iloc[0]
        # Exit 10 min before close
        exit_idx = max(0, len(day_df) - 11)  # ~15:50 ET
        session_exit = day_df['close'].iloc[exit_idx]
        date_returns[date_str] = (session_exit - session_open) / TICK_SIZE
    
    all_results = {}
    for thresh in thresholds:
        trades = []
        for _, row in res_df.iterrows():
            if row['pred'] > thresh:
                direction = 1
            elif row['pred'] < -thresh:
                direction = -1
            else:
                continue
            
            # The predicted day is the NEXT day after the feature date
            # row['date'] is the feature date, so next trading day...
            feat_date_str = row['date'].strftime('%Y%m%d')
            feat_idx = dates_list.index(feat_date_str) if feat_date_str in dates_list else -1
            if feat_idx < 0 or feat_idx + 1 >= len(dates_list):
                continue
            
            next_date_str = dates_list[feat_idx + 1]
            if next_date_str not in date_returns:
                continue
            
            full_day_return = date_returns[next_date_str]
            regime_day = daily_df[daily_df['date'] == row['date']]
            regime = classify_regime(row.get('cc_return_ticks', 0))
            
            # Intraday: passive limit entry (~0.376/2 ticks) + market exit (~1.188 ticks)
            intra_cost = COST_RT_TICKS  # keep same for consistency
            pnl_ticks = direction * full_day_return - intra_cost
            
            trades.append({
                'date': pd.Timestamp(next_date_str),
                'direction': direction,
                'pred': row['pred'],
                'actual_1h': row['actual'],
                'actual_day': full_day_return,
                'pnl_ticks': pnl_ticks,
                'pnl_dollars': pnl_ticks * TICK_VALUE,
                'regime': regime,
            })
        
        if len(trades) < 5:
            continue
        
        tdf = pd.DataFrame(trades)
        metrics = compute_trading_metrics(tdf, f"StratB_thresh={thresh:.1f}")
        metrics['threshold'] = round(thresh, 2)
        all_results[f"thresh_{thresh:.1f}"] = metrics
    
    return all_results


def simulate_strategy_c(daily_df, results, label='fwd_return_1d'):
    """Strategy C: Multi-day trend following.
    Enter when 3d + 5d OFI agree AND model predicts continuation.
    Hold until OFI reverses or signal flips. 1-3 day holds."""
    if label not in results or len(results[label]) < 10:
        return None
    
    res_df = pd.DataFrame(results[label]).sort_values('date').reset_index(drop=True)
    pred_map = {row['date']: row['pred'] for _, row in res_df.iterrows()}
    
    # Merge daily features with predictions
    merged = daily_df.copy()
    merged['pred'] = merged['date'].map(pred_map)
    merged = merged.dropna(subset=['pred', 'ofi_3d', 'ofi_5d', 'fwd_return_1d'])
    
    pred_std = merged['pred'].std()
    
    trades = []
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_date = None
    entry_price = None
    hold_days = 0
    max_hold = 3
    
    for i, row in merged.iterrows():
        ofi_3d_sign = np.sign(row['ofi_3d'])
        ofi_5d_sign = np.sign(row['ofi_5d'])
        pred_sign = np.sign(row['pred'])
        
        if position == 0:
            # Entry condition: 3d and 5d OFI agree AND model agrees
            if ofi_3d_sign == ofi_5d_sign and ofi_3d_sign == pred_sign and ofi_3d_sign != 0:
                if abs(row['pred']) > 0.5 * pred_std:
                    position = int(ofi_3d_sign)
                    entry_date = row['date']
                    entry_price = row['close']
                    hold_days = 0
        else:
            hold_days += 1
            # Exit conditions: OFI reverses, or max hold, or signal flips
            ofi_reversed = (np.sign(row['session_ofi']) != position)
            signal_flipped = (np.sign(row['pred']) == -position)
            max_hold_reached = (hold_days >= max_hold)
            
            if ofi_reversed or signal_flipped or max_hold_reached:
                exit_price = row['close']
                pnl_ticks = position * (exit_price - entry_price) / TICK_SIZE - COST_RT_TICKS
                regime = classify_regime(row['cc_return_ticks'])
                
                trades.append({
                    'date': entry_date,
                    'exit_date': row['date'],
                    'direction': position,
                    'hold_days': hold_days,
                    'pnl_ticks': pnl_ticks,
                    'pnl_dollars': pnl_ticks * TICK_VALUE,
                    'regime': regime,
                    'exit_reason': 'ofi_reverse' if ofi_reversed else ('signal_flip' if signal_flipped else 'max_hold'),
                })
                position = 0
    
    if len(trades) < 3:
        return None
    
    tdf = pd.DataFrame(trades)
    metrics = compute_trading_metrics(tdf, "StratC_multiday")
    metrics['avg_hold_days'] = round(tdf['hold_days'].mean(), 1)
    metrics['exit_reasons'] = tdf['exit_reason'].value_counts().to_dict() if 'exit_reason' in tdf.columns else {}
    
    return {'multiday': metrics}


def compute_trading_metrics(tdf, name):
    """Compute Sharpe, Sortino, WR, PF, Calmar, per-regime metrics."""
    pnl = tdf['pnl_ticks'].values
    n_trades = len(pnl)
    
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    
    wr = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    
    # Daily aggregation for Sharpe/Sortino
    if 'date' in tdf.columns:
        daily_pnl = tdf.groupby(tdf['date'].dt.date)['pnl_ticks'].sum()
    else:
        daily_pnl = pd.Series(pnl)
    
    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std(ddof=1) if len(daily_pnl) > 1 else np.nan
    
    # Annualize: ~252 trading days
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily and std_daily > 0 else np.nan
    
    # Sortino: only downside deviation
    downside = daily_pnl[daily_pnl < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (mean_daily / downside_std * np.sqrt(252)) if downside_std and downside_std > 0 else np.nan
    
    # Max drawdown
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd_ticks = dd.max() if len(dd) > 0 else 0
    
    # Calmar: annualized return / max drawdown
    total_return = cum_pnl[-1] if len(cum_pnl) > 0 else 0
    n_days = (tdf['date'].max() - tdf['date'].min()).days if 'date' in tdf.columns and n_trades > 1 else 252
    ann_return = total_return * (252 / max(1, n_days)) if n_days > 0 else 0
    calmar = ann_return / max_dd_ticks if max_dd_ticks > 0 else np.nan
    
    # Per-regime analysis (HC #428)
    regime_metrics = {}
    if 'regime' in tdf.columns:
        for regime in ['green', 'red', 'flat']:
            rdf = tdf[tdf['regime'] == regime]
            if len(rdf) >= 3:
                r_pnl = rdf['pnl_ticks'].values
                r_mean = r_pnl.mean()
                r_std = r_pnl.std(ddof=1) if len(r_pnl) > 1 else np.nan
                r_sharpe = (r_mean / r_std * np.sqrt(252)) if r_std and r_std > 0 else np.nan
                regime_metrics[regime] = {
                    'n': len(rdf),
                    'sharpe': round(r_sharpe, 2) if not np.isnan(r_sharpe) else None,
                    'wr': round(len(r_pnl[r_pnl > 0]) / len(r_pnl), 3),
                    'mean_pnl': round(r_mean, 2),
                }
    
    # Regime gap check (HC #428)
    regime_sharpes = {k: v['sharpe'] for k, v in regime_metrics.items() if v.get('sharpe') is not None}
    regime_gap = None
    if len(regime_sharpes) >= 2:
        vals = list(regime_sharpes.values())
        max_s = max(abs(v) for v in vals)
        if max_s > 0:
            regime_gap = round(abs(max(vals) - min(vals)) / max_s, 3)
    
    # Long vs Short
    long_trades = tdf[tdf['direction'] == 1]['pnl_ticks']
    short_trades = tdf[tdf['direction'] == -1]['pnl_ticks']
    long_sharpe = (long_trades.mean() / long_trades.std(ddof=1) * np.sqrt(252)) if len(long_trades) > 3 and long_trades.std() > 0 else np.nan
    short_sharpe = (short_trades.mean() / short_trades.std(ddof=1) * np.sqrt(252)) if len(short_trades) > 3 and short_trades.std() > 0 else np.nan
    
    # Trades per month
    if 'date' in tdf.columns and n_trades > 1:
        span_months = max(1, (tdf['date'].max() - tdf['date'].min()).days / 30)
        trades_per_month = n_trades / span_months
    else:
        trades_per_month = n_trades
    
    return {
        'name': name,
        'n_trades': n_trades,
        'wr': round(wr, 3),
        'pf': round(pf, 2) if pf != float('inf') else 'inf',
        'sharpe': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'sortino': round(sortino, 2) if not np.isnan(sortino) else None,
        'calmar': round(calmar, 2) if not np.isnan(calmar) else None,
        'avg_win_ticks': round(avg_win, 1),
        'avg_loss_ticks': round(avg_loss, 1),
        'total_pnl_ticks': round(sum(pnl), 1),
        'total_pnl_dollars': round(sum(pnl) * TICK_VALUE, 0),
        'max_dd_ticks': round(max_dd_ticks, 1),
        'max_dd_dollars': round(max_dd_ticks * TICK_VALUE, 0),
        'trades_per_month': round(trades_per_month, 1),
        'long_sharpe': round(long_sharpe, 2) if not np.isnan(long_sharpe) else None,
        'short_sharpe': round(short_sharpe, 2) if not np.isnan(short_sharpe) else None,
        'n_long': len(long_trades),
        'n_short': len(short_trades),
        'regime': regime_metrics,
        'regime_gap': regime_gap,
    }


def main():
    log.info("=" * 70)
    log.info("LONG HORIZON FLOW v1 — Daily directional model from accumulated OFI")
    log.info("=" * 70)
    t0 = time.time()
    
    # ── 1. Load data ──
    all_data = load_all_minute_bars()
    
    # ── 2. Build daily DataFrame ──
    daily_df = build_daily_dataframe(all_data)
    
    # ── 3. Add rolling features (using only past data per row — rolling handles this) ──
    daily_df = add_rolling_features(daily_df)
    
    # ── 4. Add forward labels ──
    daily_df = add_forward_labels(daily_df)
    daily_df = add_intraday_forward_labels(daily_df, all_data)
    
    # ── 5. Data summary ──
    log.info("\n=== DATA SUMMARY ===")
    log.info(f"Total days: {len(daily_df)}")
    log.info(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    log.info(f"Features: {len(get_feature_cols())}")
    
    # Label distributions
    for label in LABEL_COLS:
        vals = daily_df[label].dropna()
        if len(vals) > 0:
            log.info(f"  {label}: n={len(vals)}, mean={vals.mean():.1f}, std={vals.std():.1f}, "
                     f"min={vals.min():.0f}, max={vals.max():.0f}")
    
    # Daily return stats
    cc = daily_df['cc_return_ticks'].dropna()
    log.info(f"\nClose-to-close returns: mean={cc.mean():.1f}, std={cc.std():.1f}, "
             f"range=[{cc.min():.0f}, {cc.max():.0f}] ticks")
    
    # ── 6. Walk-forward training ──
    log.info("\n=== WALK-FORWARD TRAINING ===")
    log.info(f"Train window: {TRAIN_DAYS}d, OOT: {OOT_DAYS}d, Slide: {SLIDE_DAYS}d")
    
    results, feature_importance = walk_forward_train(daily_df)
    
    # ── 7. IC Analysis ──
    log.info("\n=== IC ANALYSIS ===")
    ic_report = compute_ic_metrics(results)
    for label, metrics in ic_report.items():
        log.info(f"  {label}: IC={metrics.get('ic_overall', 'N/A')}, "
                 f"IC_Sharpe={metrics.get('ic_sharpe', 'N/A')}, "
                 f"DirAcc={metrics.get('dir_accuracy', 'N/A')}, "
                 f"n={metrics.get('n_predictions', 0)}")
    
    # ── 8. Feature importance ──
    log.info("\n=== FEATURE IMPORTANCE (top 15) ===")
    if feature_importance:
        sorted_imp = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:15]:
            log.info(f"  {name}: {imp:.1f}")
    
    # ── 9. Trading simulations ──
    log.info("\n=== STRATEGY A: Daily Position (Overnight Hold) ===")
    strat_a = simulate_strategy_a(results, 'fwd_return_1d')
    if strat_a:
        for key, metrics in strat_a.items():
            log.info(f"\n  {metrics['name']}:")
            log.info(f"    Trades: {metrics['n_trades']} ({metrics['trades_per_month']}/mo), "
                     f"WR: {metrics['wr']}, PF: {metrics['pf']}")
            log.info(f"    Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
                     f"Calmar: {metrics['calmar']}")
            log.info(f"    Total P&L: {metrics['total_pnl_ticks']} ticks (${metrics['total_pnl_dollars']})")
            log.info(f"    Avg Win: {metrics['avg_win_ticks']}t, Avg Loss: {metrics['avg_loss_ticks']}t")
            log.info(f"    Max DD: {metrics['max_dd_ticks']} ticks (${metrics['max_dd_dollars']})")
            log.info(f"    Long Sharpe: {metrics['long_sharpe']}, Short Sharpe: {metrics['short_sharpe']}")
            log.info(f"    Regime: {metrics['regime']}")
            if metrics['regime_gap'] is not None:
                status = "PASS" if metrics['regime_gap'] <= 0.50 else "FAIL (>0.50)"
                log.info(f"    Regime Gap: {metrics['regime_gap']} — {status}")
    
    log.info("\n=== STRATEGY B: Intraday Position (No Overnight) ===")
    strat_b = simulate_strategy_b(results, all_data, daily_df, 'fwd_return_1h')
    if strat_b:
        for key, metrics in strat_b.items():
            log.info(f"\n  {metrics['name']}:")
            log.info(f"    Trades: {metrics['n_trades']} ({metrics['trades_per_month']}/mo), "
                     f"WR: {metrics['wr']}, PF: {metrics['pf']}")
            log.info(f"    Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}")
            log.info(f"    Total P&L: {metrics['total_pnl_ticks']} ticks (${metrics['total_pnl_dollars']})")
            log.info(f"    Regime: {metrics['regime']}")
    
    # Also try Strategy B with halfday predictions
    log.info("\n=== STRATEGY B (halfday label): Intraday Position ===")
    strat_b2 = simulate_strategy_b(results, all_data, daily_df, 'fwd_return_halfday')
    if strat_b2:
        for key, metrics in strat_b2.items():
            log.info(f"\n  {metrics['name']}:")
            log.info(f"    Trades: {metrics['n_trades']}, WR: {metrics['wr']}, PF: {metrics['pf']}")
            log.info(f"    Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}")
            log.info(f"    Total P&L: {metrics['total_pnl_ticks']} ticks (${metrics['total_pnl_dollars']})")
    
    log.info("\n=== STRATEGY C: Multi-Day Trend Following ===")
    strat_c = simulate_strategy_c(daily_df, results, 'fwd_return_1d')
    if strat_c:
        for key, metrics in strat_c.items():
            log.info(f"\n  {metrics['name']}:")
            log.info(f"    Trades: {metrics['n_trades']} ({metrics['trades_per_month']}/mo), "
                     f"WR: {metrics['wr']}, PF: {metrics['pf']}")
            log.info(f"    Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}")
            log.info(f"    Total P&L: {metrics['total_pnl_ticks']} ticks (${metrics['total_pnl_dollars']})")
            log.info(f"    Avg Hold: {metrics.get('avg_hold_days', 'N/A')} days")
            log.info(f"    Exit Reasons: {metrics.get('exit_reasons', {})}")
            log.info(f"    Regime: {metrics['regime']}")
    
    # Also try 2d and 3d labels for Strategy A
    log.info("\n=== STRATEGY A with 2d/3d labels ===")
    for lbl in ['fwd_return_2d', 'fwd_return_3d']:
        strat_ax = simulate_strategy_a(results, lbl)
        if strat_ax:
            # Just show best threshold
            best = max(strat_ax.values(), key=lambda x: x.get('sharpe') or -999)
            log.info(f"\n  Best {lbl}: {best['name']}")
            log.info(f"    Trades: {best['n_trades']}, WR: {best['wr']}, PF: {best['pf']}")
            log.info(f"    Sharpe: {best['sharpe']}, Sortino: {best['sortino']}")
            log.info(f"    Total P&L: {best['total_pnl_ticks']} ticks (${best['total_pnl_dollars']})")
    
    # ── 10. Save results ──
    elapsed = time.time() - t0
    log.info(f"\n=== COMPLETE ({elapsed:.0f}s) ===")
    
    # Save comprehensive results
    output = {
        'run_time': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data': {
            'n_days': len(daily_df),
            'date_range': [daily_df['date'].min().isoformat(), daily_df['date'].max().isoformat()],
            'n_features': len(get_feature_cols()),
        },
        'walk_forward': {
            'train_days': TRAIN_DAYS,
            'oot_days': OOT_DAYS,
            'slide_days': SLIDE_DAYS,
        },
        'ic_report': ic_report,
        'feature_importance_top15': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]) if feature_importance else {},
        'strategy_a': strat_a,
        'strategy_b_1h': strat_b,
        'strategy_b_halfday': strat_b2,
        'strategy_c': strat_c,
        'cost_assumptions': {
            'commission_rt_ticks': COMMISSION_RT_TICKS,
            'slippage_per_side_ticks': SLIPPAGE_TICKS,
            'total_cost_rt_ticks': COST_RT_TICKS,
        },
    }
    
    results_path = OUTPUT_DIR / 'results_v1.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")
    
    # Save daily DataFrame for inspection
    daily_df.to_parquet(OUTPUT_DIR / 'daily_features.parquet', index=False)
    log.info(f"Daily features saved to {OUTPUT_DIR / 'daily_features.parquet'}")
    
    # Save OOT predictions for each label
    for label, recs in results.items():
        if recs:
            pred_df = pd.DataFrame(recs)
            pred_df.to_parquet(OUTPUT_DIR / f'predictions_{label}.parquet', index=False)
    
    log.info("All outputs saved. Done.")


if __name__ == '__main__':
    main()
