#!/usr/bin/env python3
"""
halfday_refined_v1.py — Refined half-day intraday strategy.

Builds on long_horizon_flow_v1 findings:
  - halfday label had best raw IC (0.090) but terrible IC_Sharpe (0.015)
  - Adding overnight gap, opening flow, prior session features to stabilize

Walk-forward: 40d sliding train, 5d OOT (HC #0: SLIDING only).
Model: LightGBM regression.
Trading: Enter 9:35 ET passive, exit 12:30 ET (halfday) or 15:50 ET (session).
Cost: 2.376 ticks RT (market+market) — trivial at this scale.

HC #428: Regime-agnostic validation, MFE-within-horizon.
"""

import os, sys, json, logging, warnings, time, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──
DATA_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_minute_bars_v1")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/halfday_refined_v1")
LOG_FILE = Path("/home/nick/Lvl3Quant/logs/halfday_refined_v1.log")

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
COMMISSION_RT_TICKS = 0.376   # $4.70 / $12.50
COST_RT_TICKS = 2.376         # market entry + market exit (1 tick spread each side + commission)

# Walk-forward params
TRAIN_DAYS = 40
OOT_DAYS = 5
SLIDE_DAYS = 5

# Session timing (minute offsets from 9:30 ET = bar 0)
FIRST_15MIN = 15
FIRST_30MIN = 30
FIRST_HOUR = 60
HALFDAY_END = 180      # 9:30 + 180 min = 12:30 ET
AFTERNOON_START = 180   # 12:30 ET
SESSION_EXIT = 380      # 15:50 ET (= 9:30 + 380 min)
ENTRY_OFFSET = 5        # enter at 9:35 ET (5 min after open)


def load_all_minute_bars():
    """Load all parquet files and return sorted by date."""
    files = sorted(DATA_DIR.glob("*.parquet"))
    log.info(f"Found {len(files)} daily files in {DATA_DIR}")
    
    all_data = {}
    for f in files:
        date_str = f.stem
        df = pd.read_parquet(f)
        df = df.sort_values('ts_minute').reset_index(drop=True)
        all_data[date_str] = df
    
    log.info(f"Loaded {len(all_data)} days: {min(all_data.keys())} to {max(all_data.keys())}")
    return all_data


def compute_session_features(day_df, prev_day_df=None, all_closes=None, day_idx=None):
    """Compute session-level features including NEW overnight/opening features."""
    n = len(day_df)
    am_end = min(225, n)  # first ~3.75 hrs
    
    am_df = day_df.iloc[:am_end]
    pm_df = day_df.iloc[am_end:]
    
    o = day_df['open'].iloc[0]
    c = day_df['close'].iloc[-1]
    h = day_df['high'].max()
    l = day_df['low'].min()
    
    # VWAP
    if 'vwap' in day_df.columns and day_df['volume'].sum() > 0:
        session_vwap = (day_df['vwap'] * day_df['volume']).sum() / day_df['volume'].sum()
    else:
        session_vwap = c
    
    vol_30min = day_df['volume'].rolling(30, min_periods=1).sum()
    avg_30min_vol = day_df['volume'].sum() / max(1, n / 30)
    vol_concentration = vol_30min.max() / max(1, avg_30min_vol)
    
    first_2h_end = min(120, n)
    momentum_am = (day_df['close'].iloc[first_2h_end - 1] - o) / TICK_SIZE if first_2h_end > 0 else 0
    
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
        'high_close_pct': (c - l) / max(TICK_SIZE, h - l),
        'momentum_am': momentum_am,
        'momentum_pm': momentum_pm,
        'open': o,
        'close': c,
        'high': h,
        'low': l,
        'session_vwap': session_vwap,
        'trade_count': day_df['trade_count'].sum(),
    }
    
    # ────── NEW FEATURES ──────
    
    # 1. Overnight gap: difference between today's open and yesterday's close
    if prev_day_df is not None and len(prev_day_df) > 0:
        prev_close = prev_day_df['close'].iloc[-1]
        feats['overnight_gap_ticks'] = (o - prev_close) / TICK_SIZE
    else:
        feats['overnight_gap_ticks'] = 0.0
    
    # 2. First 15 minutes return (opening momentum)
    f15_end = min(FIRST_15MIN, n)
    feats['first_15min_return'] = (day_df['close'].iloc[f15_end - 1] - o) / TICK_SIZE
    
    # 3. First 30 minutes OFI
    f30_end = min(FIRST_30MIN, n)
    feats['first_30min_ofi'] = day_df['ofi_1min'].iloc[:f30_end].sum()
    
    # 4. First 30 min volume vs 20-day average (will be computed later with rolling)
    feats['first_30min_volume'] = day_df['volume'].iloc[:f30_end].sum()
    
    # 5. Previous session last hour OFI
    if prev_day_df is not None and len(prev_day_df) >= 60:
        feats['prev_session_last_hour_ofi'] = prev_day_df['ofi_1min'].iloc[-60:].sum()
    else:
        feats['prev_session_last_hour_ofi'] = 0.0
    
    # 6. Previous day range percentile (computed later with rolling)
    feats['prev_day_range_ticks'] = (prev_day_df['high'].max() - prev_day_df['low'].min()) / TICK_SIZE if prev_day_df is not None and len(prev_day_df) > 0 else 0.0
    
    # 7. Days since N-day high/low (computed later with rolling)
    # Store close for rolling computation
    
    return feats


def build_daily_dataframe(all_data):
    """Build daily-level DataFrame with session + new features."""
    records = []
    dates = sorted(all_data.keys())
    
    all_closes = []
    for i, date_str in enumerate(dates):
        day_df = all_data[date_str]
        if len(day_df) < 60:
            log.warning(f"Skipping {date_str}: only {len(day_df)} bars")
            continue
        
        prev_day_df = all_data[dates[i - 1]] if i > 0 else None
        feats = compute_session_features(day_df, prev_day_df, all_closes, i)
        feats['date'] = date_str
        records.append(feats)
        all_closes.append(feats['close'])
    
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)
    log.info(f"Built daily DataFrame: {len(df)} rows, {len(df.columns)} columns")
    return df


def add_rolling_features(df):
    """Add multi-day rolling features + new gap/range features."""
    
    # ── Original rolling features ──
    df['ofi_3d'] = df['session_ofi'].rolling(3, min_periods=3).sum()
    df['ofi_5d'] = df['session_ofi'].rolling(5, min_periods=5).sum()
    df['ofi_10d'] = df['session_ofi'].rolling(10, min_periods=10).sum()
    
    df['signed_vol_3d'] = df['session_signed_volume'].rolling(3, min_periods=3).sum()
    df['signed_vol_5d'] = df['session_signed_volume'].rolling(5, min_periods=5).sum()
    
    df['return_3d'] = df['close_vs_open_ticks'].rolling(3, min_periods=3).sum()
    df['return_5d'] = df['close_vs_open_ticks'].rolling(5, min_periods=5).sum()
    df['return_10d'] = df['close_vs_open_ticks'].rolling(10, min_periods=10).sum()
    
    df['cc_return_ticks'] = df['close'].diff() / TICK_SIZE
    df['cc_return_3d'] = df['cc_return_ticks'].rolling(3, min_periods=3).sum()
    df['cc_return_5d'] = df['cc_return_ticks'].rolling(5, min_periods=5).sum()
    
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
    
    df['ofi_vs_price_divergence'] = np.sign(df['ofi_3d']) * np.sign(df['cc_return_3d'])
    
    am_sign_3d = np.sign(df['am_ofi']).rolling(3, min_periods=3).sum()
    pm_sign_3d = np.sign(df['pm_ofi']).rolling(3, min_periods=3).sum()
    df['am_pm_consistency_3d'] = am_sign_3d * pm_sign_3d
    
    df['range_expansion_3d'] = df['price_range_ticks'].rolling(3, min_periods=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / max(1, x.iloc[0]) if len(x) == 3 else 0, raw=False
    )
    
    df['volume_trend_5d'] = df['session_volume'].rolling(5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=False
    )
    
    df['ofi_intensity'] = df['session_ofi'] / df['session_volume'].clip(lower=1)
    df['ofi_intensity_3d'] = df['ofi_intensity'].rolling(3, min_periods=3).mean()
    df['ofi_accel_3d'] = df['ofi_3d'].diff()
    
    # ── NEW rolling features ──
    
    # First 30 min volume vs 20-day rolling average
    avg_30min_vol_20d = df['first_30min_volume'].rolling(20, min_periods=5).mean()
    df['first_30min_volume_vs_avg'] = df['first_30min_volume'] / avg_30min_vol_20d.clip(lower=1)
    
    # Previous day range percentile (rank over last 20 days)
    df['prev_day_range_percentile'] = df['prev_day_range_ticks'].rolling(20, min_periods=5).apply(
        lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100 if len(x) > 1 else 0.5, raw=False
    )
    
    # Days since 20-day high / low
    days_since_high = []
    days_since_low = []
    for i in range(len(df)):
        lookback = min(i + 1, 20)
        window = df['close'].iloc[max(0, i - lookback + 1):i + 1]
        if len(window) < 2:
            days_since_high.append(0)
            days_since_low.append(0)
            continue
        high_idx = window.idxmax()
        low_idx = window.idxmin()
        days_since_high.append(i - high_idx)
        days_since_low.append(i - low_idx)
    df['days_since_high'] = days_since_high
    df['days_since_low'] = days_since_low
    
    # Overnight gap rolling stats
    df['overnight_gap_abs'] = df['overnight_gap_ticks'].abs()
    df['overnight_gap_percentile'] = df['overnight_gap_abs'].rolling(20, min_periods=5).apply(
        lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) / 100 if len(x) > 1 else 0.5, raw=False
    )
    
    # Opening flow consistency: does first-15min direction match OFI?
    df['opening_flow_agreement'] = np.sign(df['first_15min_return']) * np.sign(df['first_30min_ofi'])
    
    log.info(f"Added rolling features. Total columns: {len(df.columns)}")
    return df


def add_intraday_labels(df, all_data):
    """Add intraday forward labels: halfday, afternoon, full session.
    
    Labels are computed from the NEXT trading day's minute bars.
    Features from day T predict day T+1's intraday returns.
    """
    dates_list = sorted(all_data.keys())
    date_to_idx = {d: i for i, d in enumerate(dates_list)}
    
    fwd_halfday = []
    fwd_afternoon = []
    fwd_session = []
    
    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y%m%d')
        idx = date_to_idx.get(date_str)
        
        if idx is None or idx + 1 >= len(dates_list):
            fwd_halfday.append(np.nan)
            fwd_afternoon.append(np.nan)
            fwd_session.append(np.nan)
            continue
        
        next_date = dates_list[idx + 1]
        next_df = all_data[next_date]
        
        if len(next_df) < HALFDAY_END:
            fwd_halfday.append(np.nan)
            fwd_afternoon.append(np.nan)
            fwd_session.append(np.nan)
            continue
        
        # Entry at 9:35 ET (bar index ENTRY_OFFSET)
        entry_price = next_df['close'].iloc[min(ENTRY_OFFSET, len(next_df) - 1)]
        
        # Halfday exit at 12:30 ET
        hd_end = min(HALFDAY_END, len(next_df))
        hd_exit = next_df['close'].iloc[hd_end - 1]
        fwd_halfday.append((hd_exit - entry_price) / TICK_SIZE)
        
        # Afternoon return: 12:30 to 15:50
        if len(next_df) >= SESSION_EXIT:
            sess_exit = next_df['close'].iloc[SESSION_EXIT - 1]
            fwd_afternoon.append((sess_exit - hd_exit) / TICK_SIZE)
            fwd_session.append((sess_exit - entry_price) / TICK_SIZE)
        else:
            last_bar = next_df['close'].iloc[-1]
            fwd_afternoon.append((last_bar - hd_exit) / TICK_SIZE)
            fwd_session.append((last_bar - entry_price) / TICK_SIZE)
    
    df['fwd_return_halfday'] = fwd_halfday
    df['fwd_return_afternoon'] = fwd_afternoon
    df['fwd_return_session'] = fwd_session
    
    log.info(f"Added intraday labels. Halfday non-null: {pd.Series(fwd_halfday).notna().sum()}, "
             f"Afternoon: {pd.Series(fwd_afternoon).notna().sum()}, Session: {pd.Series(fwd_session).notna().sum()}")
    return df


def get_feature_cols():
    """Return list of feature column names (original + new)."""
    return [
        # Original session features
        'session_ofi', 'session_signed_volume', 'session_volume',
        'am_ofi', 'pm_ofi', 'ofi_trend',
        'vwap_close_deviation', 'price_range_ticks', 'close_vs_open_ticks',
        'volume_concentration', 'spread_mean', 'high_close_pct',
        'momentum_am', 'momentum_pm', 'trade_count',
        # Original rolling features
        'ofi_3d', 'ofi_5d', 'ofi_10d',
        'signed_vol_3d', 'signed_vol_5d',
        'return_3d', 'return_5d', 'return_10d',
        'cc_return_3d', 'cc_return_5d',
        'vol_regime_5d',
        'ofi_direction_streak', 'ofi_vs_price_divergence',
        'am_pm_consistency_3d', 'range_expansion_3d', 'volume_trend_5d',
        'ofi_intensity', 'ofi_intensity_3d', 'ofi_accel_3d',
        # NEW features
        'overnight_gap_ticks',
        'first_15min_return',
        'first_30min_ofi',
        'first_30min_volume_vs_avg',
        'prev_session_last_hour_ofi',
        'prev_day_range_percentile',
        'days_since_high', 'days_since_low',
        'overnight_gap_percentile',
        'opening_flow_agreement',
    ]


LABEL_COLS = ['fwd_return_halfday', 'fwd_return_afternoon', 'fwd_return_session']


def classify_regime(cc_return_ticks):
    """Classify day as green/red/flat based on ES close-to-close."""
    if cc_return_ticks > 4:
        return 'green'
    elif cc_return_ticks < -4:
        return 'red'
    return 'flat'


def walk_forward_train(df):
    """Sliding walk-forward with LightGBM. HC #0: SLIDING only."""
    feature_cols = get_feature_cols()
    results = {label: [] for label in LABEL_COLS}
    
    n = len(df)
    fold_id = 0
    
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
        
        for label in LABEL_COLS:
            train_mask = train_df[label].notna() & train_df[feature_cols].notna().all(axis=1)
            oot_mask = oot_df[label].notna() & oot_df[feature_cols].notna().all(axis=1)
            
            X_train = train_df.loc[train_mask, feature_cols].values
            y_train = train_df.loc[train_mask, label].values
            X_oot = oot_df.loc[oot_mask, feature_cols].values
            y_oot = oot_df.loc[oot_mask, label].values
            
            if len(X_train) < 15 or len(X_oot) < 1:
                continue
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(lgb_params, dtrain, num_boost_round=200)
            
            preds = model.predict(X_oot)
            
            for i, (pred, actual) in enumerate(zip(preds, y_oot)):
                oot_idx = oot_df.index[oot_mask][i]
                results[label].append({
                    'fold': fold_id,
                    'date': df.loc[oot_idx, 'date'],
                    'pred': float(pred),
                    'actual': float(actual),
                    'close': float(df.loc[oot_idx, 'close']),
                    'cc_return_ticks': float(df.loc[oot_idx, 'cc_return_ticks']) if 'cc_return_ticks' in df.columns and pd.notna(df.loc[oot_idx, 'cc_return_ticks']) else 0,
                    'overnight_gap': float(df.loc[oot_idx, 'overnight_gap_ticks']) if 'overnight_gap_ticks' in df.columns else 0,
                })
        
        if fold_id % 5 == 0:
            log.info(f"Fold {fold_id}: train {train_df['date'].iloc[0].strftime('%Y-%m-%d')} to {train_df['date'].iloc[-1].strftime('%Y-%m-%d')}, "
                     f"OOT {oot_df['date'].iloc[0].strftime('%Y-%m-%d')} to {oot_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        
        start_idx += SLIDE_DAYS
    
    log.info(f"Walk-forward complete: {fold_id} folds")
    
    # Feature importance from final model
    final_train = df.iloc[-TRAIN_DAYS - OOT_DAYS:-OOT_DAYS]
    importance = {}
    for label in LABEL_COLS:
        mask = final_train[label].notna() & final_train[feature_cols].notna().all(axis=1)
        if mask.sum() > 10:
            dtrain = lgb.Dataset(final_train.loc[mask, feature_cols].values, label=final_train.loc[mask, label].values)
            final_model = lgb.train(lgb_params, dtrain, num_boost_round=200)
            importance[label] = dict(zip(feature_cols, [float(x) for x in final_model.feature_importance(importance_type='gain')]))
    
    return results, importance


def compute_ic_metrics(results):
    """Compute IC, IC Sharpe, directional accuracy."""
    ic_report = {}
    
    for label, recs in results.items():
        if len(recs) < 10:
            ic_report[label] = {'ic': np.nan, 'ic_sharpe': np.nan, 'n': len(recs)}
            continue
        
        res_df = pd.DataFrame(recs)
        
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
        
        correct_dir = ((res_df['pred'] > 0) & (res_df['actual'] > 0)) | ((res_df['pred'] < 0) & (res_df['actual'] < 0))
        dir_acc = correct_dir.mean()
        
        ic_report[label] = {
            'ic_overall': round(float(ic_overall), 4),
            'ic_mean_fold': round(float(ic_mean), 4) if not np.isnan(ic_mean) else None,
            'ic_sharpe': round(float(ic_sharpe), 3) if not np.isnan(ic_sharpe) else None,
            'dir_accuracy': round(float(dir_acc), 4),
            'n_predictions': len(res_df),
            'n_folds': len(fold_ics),
            'pred_std': round(float(res_df['pred'].std()), 2),
            'actual_std': round(float(res_df['actual'].std()), 2),
        }
    
    return ic_report


def compute_trading_metrics(tdf, name):
    """Compute Sharpe, Sortino, WR, PF, per-regime metrics."""
    pnl = tdf['pnl_ticks'].values
    n_trades = len(pnl)
    
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    
    wr = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')
    
    # Daily PnL for Sharpe/Sortino
    if 'trade_date' in tdf.columns:
        daily_pnl = tdf.groupby('trade_date')['pnl_ticks'].sum()
    elif 'date' in tdf.columns:
        daily_pnl = tdf.groupby(tdf['date'].dt.date)['pnl_ticks'].sum()
    else:
        daily_pnl = pd.Series(pnl)
    
    mean_daily = daily_pnl.mean()
    std_daily = daily_pnl.std(ddof=1) if len(daily_pnl) > 1 else np.nan
    
    sharpe = float(mean_daily / std_daily * np.sqrt(252)) if std_daily and std_daily > 0 else np.nan
    
    downside = daily_pnl[daily_pnl < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = float(mean_daily / downside_std * np.sqrt(252)) if downside_std and downside_std > 0 else np.nan
    
    # Max drawdown
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd_ticks = float(dd.max()) if len(dd) > 0 else 0
    
    total_pnl = float(cum_pnl[-1]) if len(cum_pnl) > 0 else 0
    
    # Per-regime breakdown
    regime_metrics = {}
    if 'regime' in tdf.columns:
        for regime in ['green', 'red', 'flat']:
            r_df = tdf[tdf['regime'] == regime]
            if len(r_df) >= 3:
                r_pnl = r_df['pnl_ticks'].values
                r_wins = r_pnl[r_pnl > 0]
                r_losses = r_pnl[r_pnl < 0]
                r_wr = len(r_wins) / len(r_pnl)
                r_sharpe = np.nan
                if len(r_pnl) > 1:
                    r_std = r_pnl.std(ddof=1)
                    if r_std > 0:
                        r_sharpe = float(r_pnl.mean() / r_std * np.sqrt(252))
                regime_metrics[regime] = {
                    'n': len(r_pnl),
                    'total_pnl': round(float(r_pnl.sum()), 1),
                    'wr': round(float(r_wr), 3),
                    'sharpe': round(r_sharpe, 2) if not np.isnan(r_sharpe) else None,
                }
    
    return {
        'name': name,
        'n_trades': n_trades,
        'total_pnl_ticks': round(total_pnl, 1),
        'total_pnl_dollars': round(total_pnl * TICK_VALUE, 0),
        'sharpe': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'sortino': round(sortino, 2) if not np.isnan(sortino) else None,
        'win_rate': round(float(wr), 3),
        'profit_factor': round(pf, 2) if pf != float('inf') else None,
        'avg_win': round(avg_win, 1),
        'avg_loss': round(avg_loss, 1),
        'max_dd_ticks': round(max_dd_ticks, 1),
        'regime': regime_metrics,
    }


def simulate_intraday(results, all_data, df, label, exit_type='halfday'):
    """Simulate intraday strategy.
    
    Enter at 9:35 ET (passive limit), exit at 12:30 (halfday) or 15:50 (session).
    Cost: 2.376 ticks RT.
    
    Threshold sweep: [0.25x, 0.5x, 0.75x, 1.0x, 1.5x] of prediction std.
    """
    if label not in results or len(results[label]) < 10:
        return None
    
    res_df = pd.DataFrame(results[label]).sort_values('date').reset_index(drop=True)
    pred_std = res_df['pred'].std()
    
    if pred_std == 0:
        return None
    
    threshold_mults = [0.25, 0.5, 0.75, 1.0, 1.5]
    dates_list = sorted(all_data.keys())
    
    # Pre-compute actual intraday returns for each next-day
    date_returns = {}
    for date_str in dates_list:
        day_df = all_data[date_str]
        if len(day_df) < HALFDAY_END:
            continue
        entry_price = day_df['close'].iloc[min(ENTRY_OFFSET, len(day_df) - 1)]
        hd_exit = day_df['close'].iloc[min(HALFDAY_END - 1, len(day_df) - 1)]
        sess_exit = day_df['close'].iloc[min(SESSION_EXIT - 1, len(day_df) - 1)]
        date_returns[date_str] = {
            'halfday': (hd_exit - entry_price) / TICK_SIZE,
            'session': (sess_exit - entry_price) / TICK_SIZE,
        }
    
    all_results = {}
    for mult in threshold_mults:
        thresh = mult * pred_std
        trades = []
        
        for _, row in res_df.iterrows():
            if abs(row['pred']) <= thresh:
                continue
            
            direction = 1 if row['pred'] > 0 else -1
            
            # Feature date -> next trading day
            feat_date_str = row['date'].strftime('%Y%m%d')
            if feat_date_str not in dates_list:
                # Find closest
                feat_date_str_candidates = [d for d in dates_list if d == feat_date_str]
                if not feat_date_str_candidates:
                    continue
            
            feat_idx = dates_list.index(feat_date_str) if feat_date_str in dates_list else -1
            if feat_idx < 0 or feat_idx + 1 >= len(dates_list):
                continue
            
            next_date_str = dates_list[feat_idx + 1]
            if next_date_str not in date_returns:
                continue
            
            actual_return = date_returns[next_date_str][exit_type]
            regime = classify_regime(row.get('cc_return_ticks', 0))
            
            pnl_ticks = direction * actual_return - COST_RT_TICKS
            
            trades.append({
                'date': pd.Timestamp(next_date_str),
                'direction': direction,
                'pred': row['pred'],
                'actual': actual_return,
                'pnl_ticks': pnl_ticks,
                'pnl_dollars': pnl_ticks * TICK_VALUE,
                'regime': regime,
                'overnight_gap': row.get('overnight_gap', 0),
            })
        
        if len(trades) < 5:
            continue
        
        tdf = pd.DataFrame(trades)
        metrics = compute_trading_metrics(tdf, f"Intra_{exit_type}_thresh={mult}x")
        metrics['threshold_mult'] = mult
        metrics['threshold_ticks'] = round(float(thresh), 1)
        all_results[f"thresh_{mult}x"] = metrics
    
    return all_results


def analyze_overnight_gap(results, label='fwd_return_halfday'):
    """Analyze overnight gap as a predictor and filter.
    
    Key questions:
    1. After big gap down, does session tend to mean-revert?
    2. After big gap up, does momentum continue?
    3. Can we use gap as a filter or additional feature?
    """
    if label not in results or len(results[label]) < 10:
        return None
    
    res_df = pd.DataFrame(results[label]).sort_values('date').reset_index(drop=True)
    
    gaps = res_df['overnight_gap'].values
    actuals = res_df['actual'].values
    preds = res_df['pred'].values
    
    analysis = {}
    
    # Gap-return correlation
    if len(gaps) > 10 and np.std(gaps) > 0:
        gap_return_corr = stats.spearmanr(gaps, actuals)[0]
        gap_pred_corr = stats.spearmanr(gaps, preds)[0]
        analysis['gap_return_correlation'] = round(float(gap_return_corr), 4)
        analysis['gap_pred_correlation'] = round(float(gap_pred_corr), 4)
    
    # Stratify by gap size
    gap_abs = np.abs(gaps)
    gap_median = np.median(gap_abs)
    
    # Big gap down: gap < -median
    big_gap_down = res_df[gaps < -gap_median]
    big_gap_up = res_df[gaps > gap_median]
    small_gap = res_df[gap_abs <= gap_median]
    
    for name, subset in [('big_gap_down', big_gap_down), ('big_gap_up', big_gap_up), ('small_gap', small_gap)]:
        if len(subset) < 5:
            continue
        avg_return = subset['actual'].mean()
        pct_positive = (subset['actual'] > 0).mean()
        
        # Does model work better in gap conditions?
        if len(subset) >= 5 and subset['pred'].std() > 0 and subset['actual'].std() > 0:
            ic = stats.spearmanr(subset['pred'], subset['actual'])[0]
        else:
            ic = np.nan
        
        analysis[name] = {
            'n': len(subset),
            'avg_return_ticks': round(float(avg_return), 1),
            'pct_positive': round(float(pct_positive), 3),
            'model_ic': round(float(ic), 4) if not np.isnan(ic) else None,
        }
    
    # Gap as mean-reversion signal: big gap down -> positive return = mean-revert
    if len(big_gap_down) >= 5:
        gap_down_positive = (big_gap_down['actual'] > 0).mean()
        analysis['gap_down_mean_revert_rate'] = round(float(gap_down_positive), 3)
    
    if len(big_gap_up) >= 5:
        gap_up_positive = (big_gap_up['actual'] > 0).mean()
        analysis['gap_up_momentum_rate'] = round(float(gap_up_positive), 3)
    
    log.info(f"Overnight gap analysis: {json.dumps(analysis, indent=2, default=str)}")
    return analysis


def regime_gate(strat_metrics):
    """HC #428: Check regime-agnostic gate.
    |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
    """
    regime = strat_metrics.get('regime', {})
    green = regime.get('green', {})
    red = regime.get('red', {})
    
    s_green = green.get('sharpe')
    s_red = red.get('sharpe')
    
    if s_green is None or s_red is None:
        return {'pass': False, 'reason': 'insufficient regime data', 'gap': None}
    
    denom = max(abs(s_green), abs(s_red))
    if denom == 0:
        return {'pass': False, 'reason': 'zero sharpe in both regimes', 'gap': None}
    
    gap = abs(s_green - s_red) / denom
    passed = gap <= 0.50
    
    return {
        'pass': passed,
        'sharpe_green': s_green,
        'sharpe_red': s_red,
        'regime_gap': round(float(gap), 3),
        'threshold': 0.50,
    }


def run_feature_ablation(df, label='fwd_return_halfday'):
    """Quick ablation: train with original features only vs original + new features.
    Shows whether new features improve IC."""
    
    original_feats = [
        'session_ofi', 'session_signed_volume', 'session_volume',
        'am_ofi', 'pm_ofi', 'ofi_trend',
        'vwap_close_deviation', 'price_range_ticks', 'close_vs_open_ticks',
        'volume_concentration', 'spread_mean', 'high_close_pct',
        'momentum_am', 'momentum_pm', 'trade_count',
        'ofi_3d', 'ofi_5d', 'ofi_10d',
        'signed_vol_3d', 'signed_vol_5d',
        'return_3d', 'return_5d', 'return_10d',
        'cc_return_3d', 'cc_return_5d',
        'vol_regime_5d',
        'ofi_direction_streak', 'ofi_vs_price_divergence',
        'am_pm_consistency_3d', 'range_expansion_3d', 'volume_trend_5d',
        'ofi_intensity', 'ofi_intensity_3d', 'ofi_accel_3d',
    ]
    
    new_feats = [
        'overnight_gap_ticks',
        'first_15min_return',
        'first_30min_ofi',
        'first_30min_volume_vs_avg',
        'prev_session_last_hour_ofi',
        'prev_day_range_percentile',
        'days_since_high', 'days_since_low',
        'overnight_gap_percentile',
        'opening_flow_agreement',
    ]
    
    all_feats = original_feats + new_feats
    
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
    
    ablation_results = {}
    
    for feat_name, feat_list in [('original_only', original_feats), ('original_plus_new', all_feats)]:
        fold_ics = []
        n = len(df)
        start_idx = 0
        
        while start_idx + TRAIN_DAYS + OOT_DAYS <= n:
            train_end = start_idx + TRAIN_DAYS
            oot_end = min(train_end + OOT_DAYS, n)
            
            train_df = df.iloc[start_idx:train_end]
            oot_df = df.iloc[train_end:oot_end]
            
            train_mask = train_df[label].notna() & train_df[feat_list].notna().all(axis=1)
            oot_mask = oot_df[label].notna() & oot_df[feat_list].notna().all(axis=1)
            
            X_train = train_df.loc[train_mask, feat_list].values
            y_train = train_df.loc[train_mask, label].values
            X_oot = oot_df.loc[oot_mask, feat_list].values
            y_oot = oot_df.loc[oot_mask, label].values
            
            if len(X_train) < 15 or len(X_oot) < 3:
                start_idx += SLIDE_DAYS
                continue
            
            dtrain = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(lgb_params, dtrain, num_boost_round=200)
            preds = model.predict(X_oot)
            
            ic, _ = stats.spearmanr(preds, y_oot)
            if not np.isnan(ic):
                fold_ics.append(ic)
            
            start_idx += SLIDE_DAYS
        
        if fold_ics:
            ic_mean = np.mean(fold_ics)
            ic_std = np.std(fold_ics, ddof=1) if len(fold_ics) > 1 else np.nan
            ic_sharpe = ic_mean / ic_std if ic_std > 0 else np.nan
            ablation_results[feat_name] = {
                'ic_mean': round(float(ic_mean), 4),
                'ic_std': round(float(ic_std), 4) if not np.isnan(ic_std) else None,
                'ic_sharpe': round(float(ic_sharpe), 3) if not np.isnan(ic_sharpe) else None,
                'n_folds': len(fold_ics),
            }
    
    log.info(f"Feature ablation: {json.dumps(ablation_results, indent=2)}")
    return ablation_results


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("halfday_refined_v1 — Refined half-day intraday strategy")
    log.info("=" * 70)
    
    # ── Load data ──
    all_data = load_all_minute_bars()
    
    # ── Build features ──
    log.info("Building daily features (original + new overnight/opening features)...")
    df = build_daily_dataframe(all_data)
    df = add_rolling_features(df)
    df = add_intraday_labels(df, all_data)
    
    log.info(f"Final DataFrame: {len(df)} rows, {len(df.columns)} columns")
    log.info(f"Feature columns: {len(get_feature_cols())}")
    log.info(f"Labels: {LABEL_COLS}")
    
    # Save features
    df.to_parquet(OUTPUT_DIR / "daily_features_refined.parquet", index=False)
    log.info(f"Saved features to {OUTPUT_DIR / 'daily_features_refined.parquet'}")
    
    # ── Feature ablation: do new features help? ──
    log.info("\n" + "=" * 50)
    log.info("FEATURE ABLATION: original vs original+new")
    log.info("=" * 50)
    ablation = run_feature_ablation(df, 'fwd_return_halfday')
    
    # ── Walk-forward training ──
    log.info("\n" + "=" * 50)
    log.info("WALK-FORWARD TRAINING (all features, all labels)")
    log.info("=" * 50)
    results, importance = walk_forward_train(df)
    
    # ── IC metrics ──
    log.info("\n" + "=" * 50)
    log.info("IC METRICS")
    log.info("=" * 50)
    ic_report = compute_ic_metrics(results)
    for label, metrics in ic_report.items():
        log.info(f"  {label}: IC={metrics.get('ic_overall')}, IC_Sharpe={metrics.get('ic_sharpe')}, "
                 f"DirAcc={metrics.get('dir_accuracy')}, N={metrics.get('n_predictions')}")
    
    # ── Overnight gap analysis ──
    log.info("\n" + "=" * 50)
    log.info("OVERNIGHT GAP ANALYSIS")
    log.info("=" * 50)
    gap_analysis = analyze_overnight_gap(results, 'fwd_return_halfday')
    
    # ── Trading simulation ──
    log.info("\n" + "=" * 50)
    log.info("TRADING SIMULATION")
    log.info("=" * 50)
    
    strategy_results = {}
    for label in LABEL_COLS:
        exit_type = 'halfday' if 'halfday' in label else 'session'
        strat = simulate_intraday(results, all_data, df, label, exit_type)
        if strat:
            strategy_results[label] = strat
            for thresh_key, metrics in strat.items():
                log.info(f"  {label} | {thresh_key}: Sharpe={metrics.get('sharpe')}, "
                         f"PF={metrics.get('profit_factor')}, WR={metrics.get('win_rate')}, "
                         f"N={metrics.get('n_trades')}, PnL={metrics.get('total_pnl_ticks')}t")
                
                # Regime gate
                gate = regime_gate(metrics)
                log.info(f"    Regime gate: {'PASS' if gate['pass'] else 'FAIL'} "
                         f"(gap={gate.get('regime_gap')}, green={gate.get('sharpe_green')}, red={gate.get('sharpe_red')})")
    
    # ── Find best strategy ──
    best = None
    best_sharpe = -999
    for label, strats in strategy_results.items():
        for thresh_key, metrics in strats.items():
            s = metrics.get('sharpe')
            if s is not None and s > best_sharpe and metrics.get('n_trades', 0) >= 10:
                best_sharpe = s
                best = {
                    'label': label,
                    'threshold': thresh_key,
                    'metrics': metrics,
                    'regime_gate': regime_gate(metrics),
                }
    
    if best:
        log.info(f"\n{'='*50}")
        log.info(f"BEST STRATEGY: {best['label']} / {best['threshold']}")
        log.info(f"  Sharpe: {best['metrics'].get('sharpe')}")
        log.info(f"  Sortino: {best['metrics'].get('sortino')}")
        log.info(f"  PF: {best['metrics'].get('profit_factor')}")
        log.info(f"  WR: {best['metrics'].get('win_rate')}")
        log.info(f"  Trades: {best['metrics'].get('n_trades')}")
        log.info(f"  Total PnL: {best['metrics'].get('total_pnl_ticks')}t (${best['metrics'].get('total_pnl_dollars')})")
        log.info(f"  Regime gate: {'PASS' if best['regime_gate']['pass'] else 'FAIL'} (gap={best['regime_gate'].get('regime_gap')})")
    
    # ── Feature importance (top 15 for each label) ──
    log.info(f"\n{'='*50}")
    log.info("FEATURE IMPORTANCE (top 15)")
    log.info("=" * 50)
    for label, imp in importance.items():
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:15]
        log.info(f"  {label}:")
        for feat, val in sorted_imp:
            log.info(f"    {feat}: {val:.0f}")
    
    # ── Save results ──
    elapsed = time.time() - t0
    
    output = {
        'run_time': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'data': {
            'n_days': len(df),
            'date_range': [str(df['date'].min()), str(df['date'].max())],
            'n_features': len(get_feature_cols()),
            'new_features': [
                'overnight_gap_ticks', 'first_15min_return', 'first_30min_ofi',
                'first_30min_volume_vs_avg', 'prev_session_last_hour_ofi',
                'prev_day_range_percentile', 'days_since_high', 'days_since_low',
                'overnight_gap_percentile', 'opening_flow_agreement',
            ],
        },
        'walk_forward': {
            'train_days': TRAIN_DAYS,
            'oot_days': OOT_DAYS,
            'slide_days': SLIDE_DAYS,
        },
        'feature_ablation': ablation,
        'ic_report': ic_report,
        'overnight_gap_analysis': gap_analysis,
        'strategy_results': strategy_results,
        'best_strategy': best,
        'feature_importance': {k: dict(sorted(v.items(), key=lambda x: x[1], reverse=True)[:15]) for k, v in importance.items()},
    }
    
    results_path = OUTPUT_DIR / "results_refined_v1.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {results_path}")
    
    # Save predictions for each label
    for label, recs in results.items():
        if recs:
            pred_df = pd.DataFrame(recs)
            pred_df.to_parquet(OUTPUT_DIR / f"predictions_{label}.parquet", index=False)
            log.info(f"Saved predictions for {label}: {len(pred_df)} rows")
    
    log.info(f"\nTotal elapsed: {elapsed:.1f}s")
    log.info("DONE")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)
