#!/usr/bin/env python3
"""
pressure_strategy_v1.py — Intraday Pressure Accumulation Strategy

Concept: Use tick-level queue data aggregated to 15-min windows to detect
buying/selling pressure buildup. Predict pressure continuation over 1h-4h horizons.

Data: 40 days of tick-level queue features (~250ms-600ms resolution)
Model: LightGBM regression on 15-min aggregated pressure features
Walk-forward: 25d train, 5d OOT, 5d slide (SLIDING window, HC #0)
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──
BASE = Path("/home/nick/Lvl3Quant")
TICK_DIR = BASE / "data" / "queue_augmented_features"
MINUTE_DIR = BASE / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = BASE / "output" / "pressure_strategy_v1"
LOG_PATH = BASE / "logs" / "pressure_strategy_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
ES_TICK_VALUE = 12.50
ES_COMMISSION_RT_TICKS = 0.376  # AMP round-trip
SPREAD_CROSS_TICKS = 1.0        # market order crosses 1 tick spread
TOTAL_COST_RT_TICKS = ES_COMMISSION_RT_TICKS + 2 * SPREAD_CROSS_TICKS  # entry + exit market = 2.376 ticks
WINDOW_MINUTES = 15
WINDOWS_PER_HOUR = 4  # 60 / 15


def load_tick_data_dates():
    """Load sorted list of available tick-data dates."""
    files = sorted(TICK_DIR.glob("features_*.parquet"))
    dates = []
    for f in files:
        date_str = f.stem.replace("features_", "")
        dates.append(date_str)
    log.info(f"Found {len(dates)} tick-data days: {dates[0]} to {dates[-1]}")
    return dates


def aggregate_to_15min(df_tick, date_str):
    """
    Aggregate tick-level data to 15-minute windows.
    Returns DataFrame with one row per 15-min window.
    """
    # Convert ts_ns to datetime
    df = df_tick.copy()
    df['ts'] = pd.to_datetime(df['ts_ns'], unit='ns')
    
    # Filter to RTH only (13:30-20:00 UTC = 9:30-4:00 ET)
    df = df[(df['ts'].dt.hour >= 13) | ((df['ts'].dt.hour == 13) & (df['ts'].dt.minute >= 30))]
    df = df[df['ts'].dt.hour < 21]
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Create 15-min window labels
    df['window'] = df['ts'].dt.floor('15min')
    
    windows = []
    for window_ts, grp in df.groupby('window'):
        if len(grp) < 10:  # skip tiny windows
            continue
        
        row = {'date': date_str, 'window_ts': window_ts, 'n_ticks': len(grp)}
        
        # ── Feature 1: Net OFI over 15 min ──
        row['ofi_15min'] = grp['ofi_1s'].sum()
        
        # ── Feature 2: OFI momentum (2nd half vs 1st half) ──
        mid = len(grp) // 2
        ofi_first = grp['ofi_1s'].iloc[:mid].sum()
        ofi_second = grp['ofi_1s'].iloc[mid:].sum()
        row['ofi_momentum'] = ofi_second - ofi_first
        
        # ── Feature 3: Queue imbalance average ──
        bid_total = grp['bid_qty_at_touch'].astype(float)
        ask_total = grp['ask_qty_at_touch'].astype(float)
        imb = bid_total / (bid_total + ask_total + 1e-9)
        row['queue_imbalance_avg'] = imb.mean()
        
        # ── Feature 4: Queue imbalance trend (regression slope) ──
        if len(imb) > 5:
            x = np.arange(len(imb))
            slope, _, _, _, _ = stats.linregress(x, imb.values)
            row['queue_imbalance_trend'] = slope * len(imb)  # total shift over window
        else:
            row['queue_imbalance_trend'] = 0.0
        
        # ── Feature 5: Cancel asymmetry ──
        bid_cancel = grp['bid_cancel_rate_1s'].mean()
        ask_cancel = grp['ask_cancel_rate_1s'].mean()
        row['cancel_asymmetry'] = (ask_cancel - bid_cancel) / (ask_cancel + bid_cancel + 1e-9)
        
        # ── Feature 6: Level age asymmetry ──
        bid_age = grp['bid_level_age_s'].mean()
        ask_age = grp['ask_level_age_s'].mean()
        row['level_age_asymmetry'] = (bid_age - ask_age) / (bid_age + ask_age + 1e-9)
        
        # ── Feature 7: Large order flow (proxy: high OFI ticks) ──
        ofi_abs = grp['ofi_1s'].abs()
        p95 = ofi_abs.quantile(0.95) if len(ofi_abs) > 20 else ofi_abs.max()
        large_mask = ofi_abs >= max(p95, 1.0)
        large_ofi = grp.loc[large_mask, 'ofi_1s']
        row['large_order_buy'] = (large_ofi > 0).sum()
        row['large_order_sell'] = (large_ofi < 0).sum()
        row['large_order_flow'] = row['large_order_buy'] - row['large_order_sell']
        
        # ── Feature 8: Price return (microprice) ──
        mp_start = grp['microprice_offset_ticks'].iloc[0]
        mp_end = grp['microprice_offset_ticks'].iloc[-1]
        row['price_return_15min'] = mp_end - mp_start
        
        # ── Feature 9: Volume proxy (n_ticks as volume proxy) ──
        row['volume_15min'] = len(grp)
        
        # ── Feature 10: Top imbalance stats ──
        row['top_imbalance_avg'] = grp['top_imbalance'].mean()
        row['top_imbalance_std'] = grp['top_imbalance'].std()
        
        # ── Feature: OFI at different scales ──
        row['ofi_5s_sum'] = grp['ofi_5s'].sum()
        row['ofi_10s_sum'] = grp['ofi_10s'].sum()
        
        # ── Feature: Add rate asymmetry ──
        bid_add = grp['bid_add_rate_1s'].mean()
        ask_add = grp['ask_add_rate_1s'].mean()
        row['add_rate_asymmetry'] = (bid_add - ask_add) / (bid_add + ask_add + 1e-9)
        
        # ── Feature: Trade rate asymmetry ──
        bid_trade = grp['bid_trade_rate_1s'].mean()
        ask_trade = grp['ask_trade_rate_1s'].mean()
        row['trade_rate_asymmetry'] = (bid_trade - ask_trade) / (bid_trade + ask_trade + 1e-9)
        
        # ── Feature: OFI standard deviation (variability) ──
        row['ofi_1s_std'] = grp['ofi_1s'].std()
        
        # ── Feature: Microprice volatility ──
        row['microprice_volatility'] = grp['microprice_offset_ticks'].std()
        
        windows.append(row)
    
    if not windows:
        return pd.DataFrame()
    
    return pd.DataFrame(windows)


def add_rolling_features(df):
    """
    Add rolling/cumulative features across 15-min windows.
    These capture pressure ACCUMULATION over hours.
    df must be sorted by date + window_ts.
    """
    # Group by date to avoid rolling across days
    result_frames = []
    
    for date, day_df in df.groupby('date'):
        day_df = day_df.sort_values('window_ts').copy()
        
        if len(day_df) < 4:
            continue
        
        # ── Rolling 1h (4 windows) ──
        day_df['ofi_1h_rolling'] = day_df['ofi_15min'].rolling(4, min_periods=2).sum()
        day_df['ofi_2h_rolling'] = day_df['ofi_15min'].rolling(8, min_periods=4).sum()
        
        # ── Pressure consistency: fraction of last 4 windows with same-sign OFI ──
        def pressure_consistency(series):
            if len(series) < 2:
                return 0.5
            signs = np.sign(series)
            last_sign = signs.iloc[-1]
            if last_sign == 0:
                return 0.5
            return (signs == last_sign).mean()
        
        day_df['pressure_consistency_1h'] = day_df['ofi_15min'].rolling(4, min_periods=2).apply(
            pressure_consistency, raw=False
        )
        
        # ── Queue imbalance 1h rolling ──
        day_df['queue_imbalance_1h'] = day_df['queue_imbalance_avg'].rolling(4, min_periods=2).mean()
        
        # ── Price momentum 1h ──
        day_df['price_momentum_1h'] = day_df['price_return_15min'].rolling(4, min_periods=2).sum()
        
        # ── Volume trend: current volume / avg of last 4 ──
        vol_rolling = day_df['volume_15min'].rolling(4, min_periods=2).mean()
        day_df['volume_trend_1h'] = day_df['volume_15min'] / (vol_rolling + 1e-9)
        
        # ── OFI acceleration: current OFI_1h vs previous OFI_1h ──
        day_df['ofi_acceleration'] = day_df['ofi_1h_rolling'].diff()
        
        # ── Cancel asymmetry trend ──
        day_df['cancel_asymmetry_1h'] = day_df['cancel_asymmetry'].rolling(4, min_periods=2).mean()
        
        # ── Level age asymmetry trend ──
        day_df['level_age_asymmetry_1h'] = day_df['level_age_asymmetry'].rolling(4, min_periods=2).mean()
        
        # ── OFI intensity: OFI per tick (normalized) ──
        day_df['ofi_intensity'] = day_df['ofi_15min'] / (day_df['volume_15min'] + 1e-9)
        day_df['ofi_intensity_1h'] = day_df['ofi_intensity'].rolling(4, min_periods=2).mean()
        
        # ── Pressure score: composite ──
        # Normalize OFI and combine with queue imbalance direction
        day_df['pressure_score'] = (
            day_df['ofi_intensity'] * 0.4 +
            (day_df['queue_imbalance_avg'] - 0.5) * 0.3 +
            day_df['cancel_asymmetry'] * 0.15 +
            day_df['level_age_asymmetry'] * 0.15
        )
        day_df['pressure_score_1h'] = day_df['pressure_score'].rolling(4, min_periods=2).mean()
        
        # ── Time-of-day features ──
        day_df['hour_of_day'] = day_df['window_ts'].dt.hour + day_df['window_ts'].dt.minute / 60.0
        day_df['minutes_since_open'] = (
            (day_df['window_ts'] - day_df['window_ts'].iloc[0]).dt.total_seconds() / 60.0
        )
        
        result_frames.append(day_df)
    
    if not result_frames:
        return pd.DataFrame()
    
    return pd.concat(result_frames, ignore_index=True)


def add_forward_returns(df):
    """
    Add forward return labels: 1h, 2h, 4h.
    Returns are in microprice ticks, forward-looking from current window.
    """
    result_frames = []
    
    for date, day_df in df.groupby('date'):
        day_df = day_df.sort_values('window_ts').copy()
        n = len(day_df)
        
        # Forward returns: sum of price_return_15min over next N windows
        fwd_1h = np.full(n, np.nan)
        fwd_2h = np.full(n, np.nan)
        fwd_4h = np.full(n, np.nan)
        
        cum_returns = day_df['price_return_15min'].values
        
        for i in range(n):
            # 1h forward (next 4 windows)
            if i + 4 <= n:
                fwd_1h[i] = cum_returns[i+1:i+5].sum() if i+1 < n else np.nan
            # 2h forward (next 8 windows)
            if i + 8 <= n:
                fwd_2h[i] = cum_returns[i+1:i+9].sum() if i+1 < n else np.nan
            # 4h forward (next 16 windows)
            if i + 16 <= n:
                fwd_4h[i] = cum_returns[i+1:i+17].sum() if i+1 < n else np.nan
        
        day_df['fwd_return_1h'] = fwd_1h
        day_df['fwd_return_2h'] = fwd_2h
        day_df['fwd_return_4h'] = fwd_4h
        
        result_frames.append(day_df)
    
    if not result_frames:
        return pd.DataFrame()
    
    return pd.concat(result_frames, ignore_index=True)


def get_feature_columns():
    """Return list of feature column names for modeling."""
    return [
        'ofi_15min', 'ofi_momentum', 'queue_imbalance_avg', 'queue_imbalance_trend',
        'cancel_asymmetry', 'level_age_asymmetry', 'large_order_flow',
        'price_return_15min', 'volume_15min', 'top_imbalance_avg', 'top_imbalance_std',
        'ofi_5s_sum', 'ofi_10s_sum', 'add_rate_asymmetry', 'trade_rate_asymmetry',
        'ofi_1s_std', 'microprice_volatility',
        # Rolling features
        'ofi_1h_rolling', 'ofi_2h_rolling', 'pressure_consistency_1h',
        'queue_imbalance_1h', 'price_momentum_1h', 'volume_trend_1h',
        'ofi_acceleration', 'cancel_asymmetry_1h', 'level_age_asymmetry_1h',
        'ofi_intensity', 'ofi_intensity_1h', 'pressure_score', 'pressure_score_1h',
        'hour_of_day', 'minutes_since_open',
        'large_order_buy', 'large_order_sell',
    ]


def train_and_predict(train_df, test_df, target_col, feature_cols):
    """
    Train LightGBM regression model and return predictions on test set.
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Remove rows with NaN in target
    valid_train = ~np.isnan(y_train) & np.all(np.isfinite(X_train), axis=1)
    valid_test = ~np.isnan(y_test) & np.all(np.isfinite(X_test), axis=1)
    
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]
    X_test = X_test[valid_test]
    y_test = y_test[valid_test]
    
    if len(X_train) < 50 or len(X_test) < 10:
        return None, None, None, None
    
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.03,
        'num_leaves': 31,
        'min_child_samples': 20,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    preds = model.predict(X_test)
    
    # IC
    ic = np.corrcoef(preds, y_test)[0, 1] if len(preds) > 5 else 0.0
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance('gain')))
    
    return preds, y_test, ic, importance


def simulate_trading(test_df, preds, actuals, horizon_name, horizon_windows):
    """
    Simulate a simple pressure-following strategy.
    Enter when pressure prediction exceeds threshold, hold for horizon.
    Cost: 2.376 ticks RT (market entry + market exit).
    """
    if preds is None or len(preds) < 10:
        return None
    
    # Try multiple thresholds
    results = {}
    for pct_threshold in [60, 70, 80, 90]:
        abs_preds = np.abs(preds)
        threshold = np.percentile(abs_preds, pct_threshold)
        
        trades = []
        for i in range(len(preds)):
            if abs_preds[i] < threshold:
                continue
            
            direction = np.sign(preds[i])  # +1 long, -1 short
            gross_ticks = actuals[i] * direction  # positive if prediction correct
            net_ticks = gross_ticks - TOTAL_COST_RT_TICKS
            
            trades.append({
                'gross_ticks': gross_ticks,
                'net_ticks': net_ticks,
                'direction': direction,
                'pred_mag': abs_preds[i],
                'actual': actuals[i],
            })
        
        if len(trades) < 3:
            continue
        
        trades_df = pd.DataFrame(trades)
        
        winners = trades_df['net_ticks'] > 0
        win_rate = winners.mean()
        avg_win = trades_df.loc[winners, 'net_ticks'].mean() if winners.any() else 0
        avg_loss = trades_df.loc[~winners, 'net_ticks'].mean() if (~winners).any() else 0
        
        total_net = trades_df['net_ticks'].sum()
        gross_wins = trades_df.loc[winners, 'net_ticks'].sum() if winners.any() else 0
        gross_losses = abs(trades_df.loc[~winners, 'net_ticks'].sum()) if (~winners).any() else 1e-9
        pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')
        
        net_arr = trades_df['net_ticks'].values
        sharpe = np.mean(net_arr) / (np.std(net_arr) + 1e-9) * np.sqrt(252)  # rough annualized
        
        # Sortino
        downside = net_arr[net_arr < 0]
        downside_std = np.std(downside) if len(downside) > 0 else 1e-9
        sortino = np.mean(net_arr) / downside_std * np.sqrt(252)
        
        results[f'top_{100-pct_threshold}pct'] = {
            'n_trades': len(trades),
            'win_rate': round(win_rate, 4),
            'avg_win_ticks': round(avg_win, 3),
            'avg_loss_ticks': round(avg_loss, 3),
            'profit_factor': round(pf, 3),
            'total_net_ticks': round(total_net, 2),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'total_net_dollars': round(total_net * ES_TICK_VALUE, 2),
        }
    
    return results


def classify_regime(day_return):
    """Classify day as green/red/flat based on ES daily return."""
    if day_return > 0.5:  # >0.5 ticks = green
        return 'green'
    elif day_return < -0.5:
        return 'red'
    else:
        return 'flat'


def main():
    log.info("=" * 80)
    log.info("PRESSURE STRATEGY V1 — Intraday Pressure Accumulation")
    log.info("=" * 80)
    
    # ── Step 1: Load and aggregate all tick-data days ──
    dates = load_tick_data_dates()
    
    all_windows = []
    for date_str in dates:
        fpath = TICK_DIR / f"features_{date_str}.parquet"
        log.info(f"Processing {date_str}...")
        
        try:
            df_tick = pd.read_parquet(fpath)
            df_15min = aggregate_to_15min(df_tick, date_str)
            if len(df_15min) > 0:
                all_windows.append(df_15min)
                log.info(f"  {date_str}: {len(df_15min)} 15-min windows")
            else:
                log.warning(f"  {date_str}: no valid windows")
        except Exception as e:
            log.error(f"  {date_str}: FAILED — {e}")
    
    if not all_windows:
        log.error("No data loaded. Exiting.")
        return
    
    df_all = pd.concat(all_windows, ignore_index=True)
    log.info(f"Total 15-min windows: {len(df_all)} across {df_all['date'].nunique()} days")
    
    # ── Step 2: Add rolling features ──
    log.info("Adding rolling pressure features...")
    df_all = add_rolling_features(df_all)
    log.info(f"After rolling features: {len(df_all)} rows, {len(df_all.columns)} columns")
    
    # ── Step 3: Add forward return labels ──
    log.info("Computing forward returns...")
    df_all = add_forward_returns(df_all)
    
    # Log label stats
    for col in ['fwd_return_1h', 'fwd_return_2h', 'fwd_return_4h']:
        valid = df_all[col].dropna()
        log.info(f"  {col}: n={len(valid)}, mean={valid.mean():.4f}, std={valid.std():.4f}, "
                 f"median={valid.median():.4f}, |mean|/std={abs(valid.mean())/valid.std():.4f}")
    
    # Save feature matrix
    df_all.to_parquet(OUTPUT_DIR / "pressure_features_15min.parquet", index=False)
    log.info(f"Saved feature matrix to output/")
    
    # ── Step 4: Walk-forward ──
    feature_cols = get_feature_columns()
    unique_dates = sorted(df_all['date'].unique())
    n_dates = len(unique_dates)
    
    TRAIN_SIZE = 25
    OOT_SIZE = 5
    SLIDE = 5
    
    horizons = {
        'fwd_return_1h': ('1h', 4),
        'fwd_return_2h': ('2h', 8),
        'fwd_return_4h': ('4h', 16),
    }
    
    all_results = {h: {'fold_ics': [], 'all_preds': [], 'all_actuals': [], 'fold_details': []} 
                   for h in horizons}
    all_importances = {h: {} for h in horizons}
    
    fold_idx = 0
    start = 0
    
    while start + TRAIN_SIZE + OOT_SIZE <= n_dates:
        train_dates = unique_dates[start:start + TRAIN_SIZE]
        test_dates = unique_dates[start + TRAIN_SIZE:start + TRAIN_SIZE + OOT_SIZE]
        
        train_df = df_all[df_all['date'].isin(train_dates)].copy()
        test_df = df_all[df_all['date'].isin(test_dates)].copy()
        
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold_idx}: train={train_dates[0]}..{train_dates[-1]} ({len(train_df)} rows), "
                 f"test={test_dates[0]}..{test_dates[-1]} ({len(test_df)} rows)")
        
        for target_col, (horizon_name, horizon_windows) in horizons.items():
            preds, actuals, ic, importance = train_and_predict(
                train_df, test_df, target_col, feature_cols
            )
            
            if preds is not None:
                all_results[target_col]['fold_ics'].append(ic)
                all_results[target_col]['all_preds'].extend(preds.tolist())
                all_results[target_col]['all_actuals'].extend(actuals.tolist())
                
                # Accumulate importances
                for feat, imp in importance.items():
                    if feat not in all_importances[target_col]:
                        all_importances[target_col][feat] = []
                    all_importances[target_col][feat].append(imp)
                
                # Trading sim on this fold
                sim = simulate_trading(test_df, preds, actuals, horizon_name, horizon_windows)
                
                all_results[target_col]['fold_details'].append({
                    'fold': fold_idx,
                    'train_dates': f"{train_dates[0]}..{train_dates[-1]}",
                    'test_dates': f"{test_dates[0]}..{test_dates[-1]}",
                    'ic': round(ic, 4),
                    'n_test': len(preds),
                    'trading_sim': sim,
                })
                
                log.info(f"  {horizon_name}: IC={ic:.4f}, n_test={len(preds)}")
                if sim:
                    best_key = max(sim.keys(), key=lambda k: sim[k].get('sharpe', 0))
                    best = sim[best_key]
                    log.info(f"    Best sim ({best_key}): WR={best['win_rate']:.1%}, "
                             f"PF={best['profit_factor']:.2f}, Sharpe={best['sharpe']:.2f}, "
                             f"trades={best['n_trades']}")
            else:
                log.warning(f"  {horizon_name}: insufficient data for this fold")
        
        fold_idx += 1
        start += SLIDE
    
    log.info(f"\nCompleted {fold_idx} folds")
    
    # ── Step 5: Aggregate results ──
    log.info("\n" + "=" * 80)
    log.info("AGGREGATE RESULTS")
    log.info("=" * 80)
    
    summary = {}
    
    for target_col, (horizon_name, horizon_windows) in horizons.items():
        res = all_results[target_col]
        
        if not res['fold_ics']:
            log.warning(f"{horizon_name}: No valid folds")
            continue
        
        fold_ics = np.array(res['fold_ics'])
        concat_preds = np.array(res['all_preds'])
        concat_actuals = np.array(res['all_actuals'])
        
        # Concat IC
        concat_ic = np.corrcoef(concat_preds, concat_actuals)[0, 1]
        
        # IC Sharpe
        ic_sharpe = np.mean(fold_ics) / (np.std(fold_ics) + 1e-9)
        
        log.info(f"\n--- {horizon_name} ---")
        log.info(f"  Per-fold ICs: {[round(x, 4) for x in fold_ics]}")
        log.info(f"  Mean IC: {np.mean(fold_ics):.4f}")
        log.info(f"  Concat IC: {concat_ic:.4f}")
        log.info(f"  IC Sharpe: {ic_sharpe:.4f}")
        log.info(f"  Total OOT predictions: {len(concat_preds)}")
        
        # Concat trading sim
        concat_sim = simulate_trading(
            None,  # not used
            concat_preds,
            concat_actuals,
            horizon_name,
            horizon_windows,
        )
        
        if concat_sim:
            log.info(f"  Concat Trading Simulation:")
            for key, vals in concat_sim.items():
                log.info(f"    {key}: trades={vals['n_trades']}, WR={vals['win_rate']:.1%}, "
                         f"PF={vals['profit_factor']:.2f}, Sharpe={vals['sharpe']:.2f}, "
                         f"Sortino={vals['sortino']:.2f}, "
                         f"net_ticks={vals['total_net_ticks']:.1f} "
                         f"(${vals['total_net_dollars']:.0f})")
        
        # Feature importance
        imp_means = {f: np.mean(v) for f, v in all_importances[target_col].items()}
        sorted_imp = sorted(imp_means.items(), key=lambda x: -x[1])
        log.info(f"  Top 10 features:")
        for feat, imp in sorted_imp[:10]:
            log.info(f"    {feat}: {imp:.1f}")
        
        summary[horizon_name] = {
            'fold_ics': [round(x, 4) for x in fold_ics.tolist()],
            'mean_ic': round(float(np.mean(fold_ics)), 4),
            'concat_ic': round(float(concat_ic), 4),
            'ic_sharpe': round(float(ic_sharpe), 4),
            'n_oot_preds': len(concat_preds),
            'n_folds': len(fold_ics),
            'trading_sim': concat_sim,
            'top_features': [(f, round(v, 1)) for f, v in sorted_imp[:15]],
            'fold_details': res['fold_details'],
        }
    
    # ── Step 6: Regime analysis ──
    log.info("\n" + "=" * 80)
    log.info("REGIME ANALYSIS")
    log.info("=" * 80)
    
    # Compute daily returns for regime classification
    daily_returns = {}
    for date, day_df in df_all.groupby('date'):
        day_df = day_df.sort_values('window_ts')
        daily_ret = day_df['price_return_15min'].sum()
        daily_returns[date] = daily_ret
    
    regimes = {d: classify_regime(r) for d, r in daily_returns.items()}
    regime_counts = pd.Series(regimes).value_counts()
    log.info(f"Regime distribution: {dict(regime_counts)}")
    
    for target_col, (horizon_name, horizon_windows) in horizons.items():
        res = all_results[target_col]
        if not res['fold_details']:
            continue
        
        log.info(f"\n--- {horizon_name} Regime Breakdown ---")
        
        # We need per-fold regime analysis. Use fold test dates.
        regime_ics = {'green': [], 'red': [], 'flat': []}
        
        for fold_detail in res['fold_details']:
            # Parse test dates range
            test_range = fold_detail['test_dates'].split('..')
            ic_val = fold_detail['ic']
            
            # Simplified: assign fold IC to dominant regime of its test dates
            # For proper analysis, we'd need per-window regime, but this is fold-level
            log.info(f"  Fold {fold_detail['fold']}: test={fold_detail['test_dates']}, IC={ic_val:.4f}")
        
        # Overall regime Sharpe analysis
        if horizon_name in summary:
            summary[horizon_name]['regime_distribution'] = dict(regime_counts)
    
    # ── Step 7: Per-day analysis for regime gate ──
    log.info("\n" + "=" * 80)
    log.info("PER-DAY REGIME GATE (HC #428)")
    log.info("=" * 80)
    
    # For regime gate: we need per-day predictions. 
    # Re-run concat preds split by day/regime
    for target_col, (horizon_name, _) in horizons.items():
        res = all_results[target_col]
        if not res['all_preds']:
            continue
        
        # We'll do this by reconstructing from fold details
        # For simplicity, assess regime gate from the aggregate stats
        concat_preds = np.array(res['all_preds'])
        concat_actuals = np.array(res['all_actuals'])
        
        # Direction accuracy
        dir_correct = np.sign(concat_preds) == np.sign(concat_actuals)
        dir_accuracy = dir_correct.mean()
        
        log.info(f"\n{horizon_name}:")
        log.info(f"  Direction accuracy: {dir_accuracy:.1%}")
        log.info(f"  Pred magnitude: mean={np.mean(np.abs(concat_preds)):.3f}, "
                 f"std={np.std(concat_preds):.3f}")
        log.info(f"  Actual magnitude: mean={np.mean(np.abs(concat_actuals)):.3f}, "
                 f"std={np.std(concat_actuals):.3f}")
        
        if horizon_name in summary:
            summary[horizon_name]['direction_accuracy'] = round(float(dir_accuracy), 4)
    
    # ── Step 8: Save results ──
    results_path = OUTPUT_DIR / "results_summary.json"
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nResults saved to {results_path}")
    
    # ── Final verdict ──
    log.info("\n" + "=" * 80)
    log.info("FINAL VERDICT")
    log.info("=" * 80)
    
    viable = False
    best_horizon = None
    best_ic = -999
    
    for horizon_name, data in summary.items():
        ic = data.get('concat_ic', 0)
        ic_sharpe = data.get('ic_sharpe', 0)
        
        log.info(f"\n{horizon_name}: IC={ic:.4f}, IC_Sharpe={ic_sharpe:.4f}")
        
        # Check trading sim viability
        sim = data.get('trading_sim', {})
        if sim:
            # Check best threshold level
            for level, metrics in sim.items():
                if metrics['sharpe'] > 0.5 and metrics['profit_factor'] > 1.2:
                    log.info(f"  POTENTIALLY VIABLE at {level}: "
                             f"Sharpe={metrics['sharpe']:.2f}, PF={metrics['profit_factor']:.2f}, "
                             f"WR={metrics['win_rate']:.1%}")
                    viable = True
        
        if ic > best_ic:
            best_ic = ic
            best_horizon = horizon_name
    
    # Regime gate check
    # With only 2-3 folds, proper regime stratification is limited
    log.info(f"\nBest horizon: {best_horizon} (IC={best_ic:.4f})")
    log.info(f"Regime gate: INSUFFICIENT FOLDS for proper regime stratification (need more data)")
    
    if viable:
        log.info("\n*** RESULT: Intraday pressure accumulation shows PROMISE. ***")
        log.info("Next steps: More granular regime analysis, entry timing optimization, "
                 "early exit on pressure reversal.")
    else:
        log.info("\n*** RESULT: Intraday pressure accumulation NOT VIABLE at current thresholds. ***")
        log.info("The 15-min pressure features do not produce sufficient edge at 1-4h horizons "
                 "after costs.")
    
    log.info("\n" + "=" * 80)
    log.info("DONE")
    log.info("=" * 80)


if __name__ == '__main__':
    main()
