#!/usr/bin/env python3
"""
long_horizon_trading_v1.py — Practical long-horizon trading system using accumulated flow.

Building on validated results:
- 3-day flow validation PASSED: Sharpe ~1.7-1.9 (NW corrected), WR 55.3%, PF 1.35
- Multi-signal ensemble AUC 0.737, WR 77.8% at meta > 0.6
- HC #647: multi-hour to multi-day holds where execution cost is trivial

Strategy Architecture:
- SIGNAL: LightGBM classifier on accumulated OFI features (3d/5d/10d rolling)
- ENTRY: When flow signal crosses threshold, enter at session open via limit order
- EXIT: (a) Flow reversal, (b) Time limit (4h intraday / 3d multi-day), (c) Stop loss
- HORIZONS: Intraday 4h-8h holds + multi-day 1-3d holds
- COST: Passive entry (0.376 ticks) + aggressive exit (1.376 ticks) = 1.752 ticks RT

Walk-forward: 40-day sliding train, 1-day OOT, 1-day slide (per HC #0).
Regime gate: HC #428 R1 (gap <= 0.50).

MLflow logging to Jupiter (http://jupiter:5000), experiment "long_horizon_trading_v1".
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
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/long_horizon_trading_v1")
LOG_FILE = Path("/home/nick/Lvl3Quant/logs/long_horizon_trading_v1.log")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

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
SPREAD_CROSSING_TICKS = 1.0   # 1 tick to cross spread

# Realistic cost model:
# Entry: passive limit at bid/ask = 0.376/2 = 0.188 ticks (half RT commission)
# Exit: passive when time allows, aggressive when forced
COST_PASSIVE_ENTRY_AGGRESSIVE_EXIT = COMMISSION_RT_TICKS + SPREAD_CROSSING_TICKS  # 1.376 ticks
COST_PASSIVE_BOTH = COMMISSION_RT_TICKS  # 0.376 ticks (best case)
COST_AGGRESSIVE_BOTH = COMMISSION_RT_TICKS + 2 * SPREAD_CROSSING_TICKS  # 2.376 ticks (worst case)

# Use realistic blend: passive entry (limit at open), aggressive exit (market at time limit)
COST_RT_TICKS = COST_PASSIVE_ENTRY_AGGRESSIVE_EXIT  # 1.376 ticks

# Walk-forward params (HC #0: SLIDING only)
TRAIN_DAYS = 40
OOT_DAYS = 1   # 1-day OOT for maximum granularity
SLIDE_DAYS = 1  # slide 1 day

# Session timing
SESSION_MINUTES = 450   # ~7.5 hours (9:30 - 16:45 ET for ES)
FIRST_HOUR_MIN = 60
HALF_SESSION_MIN = 225  # ~3.75 hours
FOUR_HOUR_MIN = 240
SIX_HOUR_MIN = 360
EXIT_BUFFER_MIN = 10    # exit 10 min before session end

# Strategy parameters
INTRADAY_HOLD_MINUTES = 240     # 4-hour default hold
MAX_INTRADAY_HOLD_MINUTES = 360 # 6-hour max
MULTIDAY_MAX_HOLD = 3           # days
STOP_LOSS_TICKS = 80            # 20 points = 80 ticks (wide for long horizon)
TRAILING_STOP_TICKS = 40        # 10 points trailing stop

# MLflow
MLFLOW_TRACKING_URI = "http://jupiter:5000"
MLFLOW_EXPERIMENT = "long_horizon_trading_v1"

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    HAS_MLFLOW = True
    log.info(f"MLflow connected: {MLFLOW_TRACKING_URI}, experiment={MLFLOW_EXPERIMENT}")
except Exception as e:
    HAS_MLFLOW = False
    log.warning(f"MLflow unavailable: {e}. Results will be saved locally only.")


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_all_minute_bars():
    """Load all parquet files, return dict of date_str -> DataFrame."""
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


def compute_session_features(day_df):
    """Compute session-level features from minute bars for ONE day."""
    n = len(day_df)
    am_end = min(225, n)

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

    # Volume concentration
    vol_30min = day_df['volume'].rolling(30, min_periods=1).sum()
    avg_30min_vol = day_df['volume'].sum() / max(1, n / 30)
    vol_concentration = vol_30min.max() / max(1, avg_30min_vol)

    # Momentum
    first_2h_end = min(120, n)
    momentum_am = (day_df['close'].iloc[first_2h_end - 1] - o) / TICK_SIZE if first_2h_end > 0 else 0
    last_2h_start = max(0, n - 120)
    momentum_pm = (c - day_df['open'].iloc[last_2h_start]) / TICK_SIZE if last_2h_start < n else 0

    # OFI by hour (first 4 hours, then rest)
    h1_end = min(60, n)
    h2_end = min(120, n)
    h3_end = min(180, n)
    h4_end = min(240, n)

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
        'spread_mean': day_df['spread_mean'].mean() if 'spread_mean' in day_df.columns else 1.0,
        'high_close_pct': (c - l) / max(TICK_SIZE, h - l),
        'momentum_am': momentum_am,
        'momentum_pm': momentum_pm,
        'trade_count': day_df['trade_count'].sum() if 'trade_count' in day_df.columns else n,
        # Hourly OFI breakdown
        'ofi_h1': day_df['ofi_1min'].iloc[:h1_end].sum(),
        'ofi_h2': day_df['ofi_1min'].iloc[h1_end:h2_end].sum() if h2_end > h1_end else 0,
        'ofi_h3': day_df['ofi_1min'].iloc[h2_end:h3_end].sum() if h3_end > h2_end else 0,
        'ofi_h4': day_df['ofi_1min'].iloc[h3_end:h4_end].sum() if h4_end > h3_end else 0,
        'ofi_late': day_df['ofi_1min'].iloc[h4_end:].sum() if n > h4_end else 0,
        # Price location
        'open': o,
        'close': c,
        'high': h,
        'low': l,
        'session_vwap': session_vwap,
    }

    return feats


def build_daily_dataframe(all_data):
    """Build daily-level DataFrame with session features."""
    records = []
    dates = sorted(all_data.keys())

    for date_str in dates:
        day_df = all_data[date_str]
        if len(day_df) < 60:
            log.warning(f"Skipping {date_str}: only {len(day_df)} bars")
            continue
        feats = compute_session_features(day_df)
        feats['date'] = date_str
        records.append(feats)

    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)
    log.info(f"Built daily DataFrame: {len(df)} rows")
    return df


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def add_rolling_features(df):
    """Add multi-day rolling features. Rolling windows use only past data."""

    # Core OFI rolling sums
    for w in [3, 5, 10, 20]:
        df[f'ofi_{w}d'] = df['session_ofi'].rolling(w, min_periods=w).sum()
        df[f'signed_vol_{w}d'] = df['session_signed_volume'].rolling(w, min_periods=w).sum()

    # Rolling returns (close-to-close)
    df['cc_return_ticks'] = df['close'].diff() / TICK_SIZE
    for w in [3, 5, 10, 20]:
        df[f'cc_return_{w}d'] = df['cc_return_ticks'].rolling(w, min_periods=w).sum()

    # Volatility regime
    for w in [5, 10, 20]:
        df[f'vol_regime_{w}d'] = df['cc_return_ticks'].rolling(w, min_periods=w).std()

    # OFI z-scores (normalized intensity)
    for w in [5, 10, 20]:
        roll_mean = df['session_ofi'].rolling(w, min_periods=w).mean()
        roll_std = df['session_ofi'].rolling(w, min_periods=w).std()
        df[f'ofi_zscore_{w}d'] = (df['session_ofi'] - roll_mean) / roll_std.clip(lower=1e-6)

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

    # OFI vs price divergence (3d and 5d)
    df['ofi_vs_price_3d'] = np.sign(df['ofi_3d']) * np.sign(df['cc_return_3d'])
    df['ofi_vs_price_5d'] = np.sign(df['ofi_5d']) * np.sign(df['cc_return_5d'])

    # OFI intensity (per unit volume)
    df['ofi_intensity'] = df['session_ofi'] / df['session_volume'].clip(lower=1)
    df['ofi_intensity_3d'] = df['ofi_intensity'].rolling(3, min_periods=3).mean()
    df['ofi_intensity_5d'] = df['ofi_intensity'].rolling(5, min_periods=5).mean()

    # OFI acceleration
    df['ofi_accel_3d'] = df['ofi_3d'].diff()
    df['ofi_accel_5d'] = df['ofi_5d'].diff()

    # AM/PM consistency
    am_sign_3d = np.sign(df['am_ofi']).rolling(3, min_periods=3).sum()
    pm_sign_3d = np.sign(df['pm_ofi']).rolling(3, min_periods=3).sum()
    df['am_pm_consistency_3d'] = am_sign_3d * pm_sign_3d

    # Range expansion
    df['range_expansion_3d'] = df['price_range_ticks'].pct_change(3)

    # Volume trend
    df['volume_trend_5d'] = df['session_volume'].rolling(5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=False
    )

    # Close position in range (where did we close relative to high/low)
    df['close_position_3d'] = df['high_close_pct'].rolling(3, min_periods=3).mean()

    # Hourly OFI patterns
    df['late_ofi_dominance'] = df['ofi_late'] / (df['session_ofi'].abs().clip(lower=1))
    df['h1_ofi_dominance'] = df['ofi_h1'] / (df['session_ofi'].abs().clip(lower=1))

    # Cumulative OFI (running total — represents institutional positioning)
    df['cum_ofi'] = df['session_ofi'].cumsum()
    df['cum_ofi_detrend_10d'] = df['cum_ofi'] - df['cum_ofi'].rolling(10, min_periods=10).mean()

    log.info(f"Added rolling features. Total columns: {len(df.columns)}")
    return df


def add_forward_labels(df, all_data):
    """Add forward return labels for multiple horizons."""
    dates_list = sorted(all_data.keys())
    date_to_idx = {d: i for i, d in enumerate(dates_list)}

    # Close-to-close forward returns
    df['fwd_return_1d'] = (df['close'].shift(-1) - df['close']) / TICK_SIZE
    df['fwd_return_2d'] = (df['close'].shift(-2) - df['close']) / TICK_SIZE
    df['fwd_return_3d'] = (df['close'].shift(-3) - df['close']) / TICK_SIZE

    # Direction labels (binary)
    df['fwd_dir_1d'] = (df['fwd_return_1d'] > 0).astype(float)
    df['fwd_dir_3d'] = (df['fwd_return_3d'] > 0).astype(float)
    # Set NaN where return is NaN
    df.loc[df['fwd_return_1d'].isna(), 'fwd_dir_1d'] = np.nan
    df.loc[df['fwd_return_3d'].isna(), 'fwd_dir_3d'] = np.nan

    # Intraday forward labels (using next-day minute bars)
    fwd_4h = []
    fwd_halfday = []
    fwd_fullday = []

    for _, row in df.iterrows():
        date_str = row['date'].strftime('%Y%m%d')
        idx = date_to_idx.get(date_str)

        if idx is None or idx + 1 >= len(dates_list):
            fwd_4h.append(np.nan)
            fwd_halfday.append(np.nan)
            fwd_fullday.append(np.nan)
            continue

        next_date = dates_list[idx + 1]
        next_df = all_data[next_date]

        if len(next_df) < 60:
            fwd_4h.append(np.nan)
            fwd_halfday.append(np.nan)
            fwd_fullday.append(np.nan)
            continue

        next_open = next_df['open'].iloc[0]

        # 4-hour return
        h4_idx = min(FOUR_HOUR_MIN - 1, len(next_df) - 1)
        fwd_4h.append((next_df['close'].iloc[h4_idx] - next_open) / TICK_SIZE)

        # Half-day return
        hd_idx = min(HALF_SESSION_MIN - 1, len(next_df) - 1)
        fwd_halfday.append((next_df['close'].iloc[hd_idx] - next_open) / TICK_SIZE)

        # Full day return (open to 10 min before close)
        exit_idx = max(0, len(next_df) - EXIT_BUFFER_MIN - 1)
        fwd_fullday.append((next_df['close'].iloc[exit_idx] - next_open) / TICK_SIZE)

    df['fwd_return_4h'] = fwd_4h
    df['fwd_return_halfday'] = fwd_halfday
    df['fwd_return_fullday'] = fwd_fullday

    log.info(f"Forward labels added. Valid counts: 1d={df['fwd_return_1d'].notna().sum()}, "
             f"4h={df['fwd_return_4h'].notna().sum()}, halfday={df['fwd_return_halfday'].notna().sum()}")
    return df


# ═══════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════

def get_feature_cols(df):
    """Return feature columns (exclude metadata, prices, labels)."""
    exclude_prefixes = ['fwd_', 'date', 'open', 'close', 'high', 'low', 'session_vwap']
    return [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]


# ═══════════════════════════════════════════════════════════════════════
# WALK-FORWARD ENGINE
# ═══════════════════════════════════════════════════════════════════════

def walk_forward_predict(df, feature_cols, target_col, use_classifier=True):
    """
    Sliding walk-forward prediction. HC #0: SLIDING window only.

    Returns DataFrame with columns: date, pred, actual, fold
    """
    n = len(df)
    predictions = []
    fold_id = 0
    importances = {}

    lgb_params_reg = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.03,
        'num_leaves': 12,
        'max_depth': 4,
        'min_child_samples': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    lgb_params_cls = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.03,
        'num_leaves': 12,
        'max_depth': 4,
        'min_child_samples': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    start_idx = TRAIN_DAYS
    while start_idx + OOT_DAYS <= n:
        train_start = start_idx - TRAIN_DAYS
        train_end = start_idx
        oot_end = min(start_idx + OOT_DAYS, n)

        train_df = df.iloc[train_start:train_end]
        oot_df = df.iloc[start_idx:oot_end]

        fold_id += 1

        # Valid masks
        train_mask = train_df[target_col].notna() & train_df[feature_cols].notna().all(axis=1)
        oot_mask = oot_df[target_col].notna() & oot_df[feature_cols].notna().all(axis=1)

        X_train = train_df.loc[train_mask, feature_cols].values
        y_train = train_df.loc[train_mask, target_col].values
        X_oot = oot_df.loc[oot_mask, feature_cols].values
        y_oot = oot_df.loc[oot_mask, target_col].values

        if len(X_train) < 20 or len(X_oot) < 1:
            start_idx += SLIDE_DAYS
            continue

        if use_classifier:
            params = lgb_params_cls
        else:
            params = lgb_params_reg

        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, dtrain, num_boost_round=200)

        preds = model.predict(X_oot)

        for i in range(len(preds)):
            oot_idx = oot_df.index[oot_mask][i]
            predictions.append({
                'fold': fold_id,
                'date': df.loc[oot_idx, 'date'],
                'pred': preds[i],
                'actual': y_oot[i],
                'close': df.loc[oot_idx, 'close'],
                'cc_return_ticks': df.loc[oot_idx, 'cc_return_ticks'] if 'cc_return_ticks' in df.columns else 0,
            })

        # Track feature importance from last model
        if fold_id == 1 or fold_id % 30 == 0:
            imp = model.feature_importance(importance_type='gain')
            for fname, fval in zip(feature_cols, imp):
                importances[fname] = importances.get(fname, 0) + fval

        if fold_id % 30 == 0:
            log.info(f"  Fold {fold_id}: train {train_df['date'].iloc[0].strftime('%Y-%m-%d')} "
                     f"to {train_df['date'].iloc[-1].strftime('%Y-%m-%d')}, "
                     f"OOT {oot_df['date'].iloc[0].strftime('%Y-%m-%d')}")

        start_idx += SLIDE_DAYS

    log.info(f"Walk-forward complete: {fold_id} folds, {len(predictions)} predictions")
    return pd.DataFrame(predictions), importances


# ═══════════════════════════════════════════════════════════════════════
# INTRADAY SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def simulate_intraday_trades(pred_df, all_data, hold_minutes=240, threshold_pct=55,
                              stop_loss_ticks=80, use_trailing_stop=True,
                              trailing_stop_ticks=40, cost_rt=COST_RT_TICKS):
    """
    Simulate intraday trades with proper minute-bar execution.

    - Enter at session open when previous day's prediction crosses threshold
    - Hold for hold_minutes or until stop/trailing stop hit
    - Exit at hold_minutes or session end - 10 min, whichever comes first
    - Track MFE/MAE for each trade
    """
    dates_list = sorted(all_data.keys())
    date_to_idx = {d: i for i, d in enumerate(dates_list)}

    trades = []

    for _, row in pred_df.iterrows():
        prob = row['pred']
        date_str = row['date'].strftime('%Y%m%d')
        idx = date_to_idx.get(date_str)

        if idx is None or idx + 1 >= len(dates_list):
            continue

        # Decision: go long if prob > threshold, short if prob < (1 - threshold)
        threshold = threshold_pct / 100.0
        if prob >= threshold:
            direction = 1
        elif prob <= (1 - threshold):
            direction = -1
        else:
            continue

        # Execute on NEXT day
        next_date_str = dates_list[idx + 1]
        next_df = all_data[next_date_str]

        if len(next_df) < 60:
            continue

        entry_price = next_df['open'].iloc[0]
        entry_bar = 0

        # Simulate bar-by-bar
        max_bar = min(hold_minutes, len(next_df) - EXIT_BUFFER_MIN)
        if max_bar <= 0:
            continue

        mfe = 0  # max favorable excursion in ticks
        mae = 0  # max adverse excursion in ticks
        exit_price = None
        exit_bar = None
        exit_reason = 'time'

        running_peak = 0  # for trailing stop

        for bar in range(1, max_bar + 1):
            bar_high = next_df['high'].iloc[bar]
            bar_low = next_df['low'].iloc[bar]
            bar_close = next_df['close'].iloc[bar]

            # Favorable/adverse excursion
            if direction == 1:
                fav = (bar_high - entry_price) / TICK_SIZE
                adv = (entry_price - bar_low) / TICK_SIZE
                unrealized = (bar_close - entry_price) / TICK_SIZE
            else:
                fav = (entry_price - bar_low) / TICK_SIZE
                adv = (bar_high - entry_price) / TICK_SIZE
                unrealized = (entry_price - bar_close) / TICK_SIZE

            mfe = max(mfe, fav)
            mae = max(mae, adv)
            running_peak = max(running_peak, unrealized)

            # Stop loss check
            if adv >= stop_loss_ticks:
                exit_price = entry_price - direction * stop_loss_ticks * TICK_SIZE
                exit_bar = bar
                exit_reason = 'stop_loss'
                break

            # Trailing stop check
            if use_trailing_stop and running_peak > trailing_stop_ticks:
                if running_peak - unrealized >= trailing_stop_ticks:
                    exit_price = bar_close
                    exit_bar = bar
                    exit_reason = 'trailing_stop'
                    break

        # If no stop hit, exit at time limit
        if exit_price is None:
            exit_bar = max_bar
            exit_price = next_df['close'].iloc[exit_bar]
            exit_reason = 'time'

        pnl_ticks = direction * (exit_price - entry_price) / TICK_SIZE - cost_rt

        # Determine regime from the signal day (not trade day)
        regime = classify_regime(row['cc_return_ticks'])

        trades.append({
            'signal_date': row['date'],
            'trade_date': pd.Timestamp(next_date_str),
            'direction': direction,
            'pred': prob,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'hold_minutes': exit_bar,
            'pnl_ticks': pnl_ticks,
            'pnl_dollars': pnl_ticks * TICK_VALUE,
            'mfe_ticks': mfe,
            'mae_ticks': mae,
            'exit_reason': exit_reason,
            'regime': regime,
            'cost_ticks': cost_rt,
        })

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════
# MULTI-DAY SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def simulate_multiday_trades(pred_df, daily_df, threshold_pct=55,
                               max_hold_days=3, cost_rt=COST_RT_TICKS):
    """
    Simulate multi-day trend-following trades.

    - Enter when prediction + flow indicators agree
    - Hold until flow reverses, prediction flips, or max hold reached
    - Cost is trivial at multi-day horizons
    """
    pred_map = {row['date']: row['pred'] for _, row in pred_df.iterrows()}
    threshold = threshold_pct / 100.0

    trades = []
    position = 0
    entry_date = None
    entry_price = None
    hold_days = 0

    for i, row in daily_df.iterrows():
        pred = pred_map.get(row['date'])
        if pred is None:
            # If we're in a position, count the hold day
            if position != 0:
                hold_days += 1
                if hold_days >= max_hold_days:
                    pnl_ticks = position * (row['close'] - entry_price) / TICK_SIZE - cost_rt
                    regime = classify_regime(row['cc_return_ticks'])
                    trades.append({
                        'signal_date': entry_date,
                        'exit_date': row['date'],
                        'direction': position,
                        'entry_price': entry_price,
                        'exit_price': row['close'],
                        'hold_days': hold_days,
                        'pnl_ticks': pnl_ticks,
                        'pnl_dollars': pnl_ticks * TICK_VALUE,
                        'exit_reason': 'max_hold',
                        'regime': regime,
                    })
                    position = 0
            continue

        if position == 0:
            # Entry
            ofi_3d = row.get('ofi_3d', 0)
            ofi_5d = row.get('ofi_5d', 0)

            if pred >= threshold and ofi_3d > 0 and ofi_5d > 0:
                position = 1
                entry_date = row['date']
                entry_price = row['close']
                hold_days = 0
            elif pred <= (1 - threshold) and ofi_3d < 0 and ofi_5d < 0:
                position = -1
                entry_date = row['date']
                entry_price = row['close']
                hold_days = 0
        else:
            hold_days += 1

            # Exit conditions
            ofi_reversed = (position == 1 and row['session_ofi'] < 0) or \
                           (position == -1 and row['session_ofi'] > 0)
            signal_flipped = (position == 1 and pred < 0.45) or \
                             (position == -1 and pred > 0.55)
            max_hold_reached = hold_days >= max_hold_days

            if ofi_reversed or signal_flipped or max_hold_reached:
                pnl_ticks = position * (row['close'] - entry_price) / TICK_SIZE - cost_rt
                regime = classify_regime(row['cc_return_ticks'])

                trades.append({
                    'signal_date': entry_date,
                    'exit_date': row['date'],
                    'direction': position,
                    'pred': pred,
                    'entry_price': entry_price,
                    'exit_price': row['close'],
                    'hold_days': hold_days,
                    'pnl_ticks': pnl_ticks,
                    'pnl_dollars': pnl_ticks * TICK_VALUE,
                    'exit_reason': 'ofi_reverse' if ofi_reversed else ('signal_flip' if signal_flipped else 'max_hold'),
                    'regime': regime,
                })
                position = 0

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════

def classify_regime(cc_return_ticks):
    """Classify day as green/red/flat."""
    if isinstance(cc_return_ticks, (float, int, np.floating, np.integer)):
        if cc_return_ticks > 4:
            return 'green'
        elif cc_return_ticks < -4:
            return 'red'
    return 'flat'


def compute_newey_west_sharpe(daily_pnl, max_lag=5):
    """Compute Sharpe ratio with Newey-West standard errors for autocorrelation correction."""
    n = len(daily_pnl)
    if n < 10:
        return np.nan, np.nan

    mean_pnl = daily_pnl.mean()
    resids = daily_pnl - mean_pnl

    # Gamma_0 (variance)
    gamma_0 = (resids ** 2).sum() / n

    # Newey-West HAC estimator
    nw_var = gamma_0
    for lag in range(1, min(max_lag + 1, n)):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = (resids[lag:].values * resids[:-lag].values).sum() / n
        nw_var += 2 * weight * gamma_lag

    nw_std = np.sqrt(max(nw_var, 1e-10))
    nw_sharpe = mean_pnl / nw_std * np.sqrt(252)

    # Standard Sharpe for comparison
    std_sharpe = mean_pnl / daily_pnl.std(ddof=1) * np.sqrt(252) if daily_pnl.std() > 0 else np.nan

    return nw_sharpe, std_sharpe


def compute_trading_metrics(trades_df, name=""):
    """Comprehensive trading metrics with regime analysis."""
    if len(trades_df) < 3:
        return {'name': name, 'n_trades': len(trades_df), 'error': 'too_few_trades'}

    pnl = trades_df['pnl_ticks'].values
    n_trades = len(pnl)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    wr = len(wins) / n_trades
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Daily P&L aggregation
    date_col = 'trade_date' if 'trade_date' in trades_df.columns else 'exit_date'
    if date_col in trades_df.columns:
        daily_pnl = trades_df.groupby(trades_df[date_col].dt.date)['pnl_ticks'].sum()
    else:
        daily_pnl = pd.Series(pnl)

    # Newey-West corrected Sharpe
    nw_sharpe, std_sharpe = compute_newey_west_sharpe(daily_pnl)

    # Sortino
    downside = daily_pnl[daily_pnl < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (daily_pnl.mean() / downside_std * np.sqrt(252)) if downside_std and downside_std > 0 else np.nan

    # Drawdown
    cum_pnl = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd = dd.max() if len(dd) > 0 else 0

    # Per-regime analysis (HC #428)
    regime_metrics = {}
    if 'regime' in trades_df.columns:
        for regime in ['green', 'red', 'flat']:
            rdf = trades_df[trades_df['regime'] == regime]
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
                    'total_pnl': round(r_pnl.sum(), 1),
                }

    # Regime gap (HC #428 R1)
    regime_sharpes = {k: v['sharpe'] for k, v in regime_metrics.items()
                      if v.get('sharpe') is not None}
    regime_gap = None
    regime_pass = None
    if len(regime_sharpes) >= 2:
        vals = list(regime_sharpes.values())
        max_s = max(abs(v) for v in vals)
        if max_s > 0:
            regime_gap = round(abs(max(vals) - min(vals)) / max_s, 3)
            regime_pass = regime_gap <= 0.50

    # Long vs Short
    long_trades = trades_df[trades_df['direction'] == 1]['pnl_ticks']
    short_trades = trades_df[trades_df['direction'] == -1]['pnl_ticks']

    # MFE/MAE if available
    mfe_stats = {}
    if 'mfe_ticks' in trades_df.columns:
        mfe_stats = {
            'avg_mfe': round(trades_df['mfe_ticks'].mean(), 1),
            'avg_mae': round(trades_df['mae_ticks'].mean(), 1),
            'mfe_mae_ratio': round(trades_df['mfe_ticks'].mean() / max(1, trades_df['mae_ticks'].mean()), 2),
            'p90_mfe': round(trades_df['mfe_ticks'].quantile(0.9), 1),
        }

    # Exit reason distribution
    exit_reasons = {}
    if 'exit_reason' in trades_df.columns:
        exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

    # Trades per month
    date_col_val = trades_df.get(date_col)
    if date_col_val is not None and n_trades > 1:
        span_months = max(1, (date_col_val.max() - date_col_val.min()).days / 30)
        trades_per_month = n_trades / span_months
    else:
        trades_per_month = n_trades

    return {
        'name': name,
        'n_trades': n_trades,
        'wr': round(wr, 3),
        'pf': round(pf, 2) if pf != float('inf') else 999,
        'sharpe_nw': round(nw_sharpe, 2) if not np.isnan(nw_sharpe) else None,
        'sharpe_std': round(std_sharpe, 2) if not np.isnan(std_sharpe) else None,
        'sortino': round(sortino, 2) if not np.isnan(sortino) else None,
        'total_pnl_ticks': round(sum(pnl), 1),
        'total_pnl_dollars': round(sum(pnl) * TICK_VALUE, 0),
        'avg_pnl_ticks': round(np.mean(pnl), 2),
        'avg_win_ticks': round(wins.mean(), 1) if len(wins) > 0 else 0,
        'avg_loss_ticks': round(abs(losses.mean()), 1) if len(losses) > 0 else 0,
        'max_dd_ticks': round(max_dd, 1),
        'max_dd_dollars': round(max_dd * TICK_VALUE, 0),
        'trades_per_month': round(trades_per_month, 1),
        'n_long': len(long_trades),
        'n_short': len(short_trades),
        'long_mean_pnl': round(long_trades.mean(), 2) if len(long_trades) > 0 else 0,
        'short_mean_pnl': round(short_trades.mean(), 2) if len(short_trades) > 0 else 0,
        'regime': regime_metrics,
        'regime_gap': regime_gap,
        'regime_pass': regime_pass,
        'mfe_stats': mfe_stats,
        'exit_reasons': exit_reasons,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 80)
    log.info("LONG HORIZON TRADING v1 — Practical trading system from accumulated flow")
    log.info("=" * 80)
    t0 = time.time()

    run_params = {
        'train_days': TRAIN_DAYS,
        'oot_days': OOT_DAYS,
        'slide_days': SLIDE_DAYS,
        'cost_rt_ticks': COST_RT_TICKS,
        'intraday_hold_min': INTRADAY_HOLD_MINUTES,
        'max_intraday_hold_min': MAX_INTRADAY_HOLD_MINUTES,
        'multiday_max_hold': MULTIDAY_MAX_HOLD,
        'stop_loss_ticks': STOP_LOSS_TICKS,
        'trailing_stop_ticks': TRAILING_STOP_TICKS,
    }

    # Start MLflow run
    mlflow_run = None
    if HAS_MLFLOW:
        try:
            mlflow_run = mlflow.start_run(run_name=f"lh_trading_v1_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.log_params(run_params)
            log.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow run start failed: {e}")

    try:
        # ── 1. Load data ──
        all_data = load_all_minute_bars()

        # ── 2. Build daily features ──
        daily_df = build_daily_dataframe(all_data)

        # ── 3. Rolling features ──
        daily_df = add_rolling_features(daily_df)

        # ── 4. Forward labels ──
        daily_df = add_forward_labels(daily_df, all_data)

        feature_cols = get_feature_cols(daily_df)
        log.info(f"\nFeatures ({len(feature_cols)}): {feature_cols[:10]}...")

        # ── 5. Data summary ──
        log.info(f"\n{'='*60}")
        log.info(f"DATA SUMMARY")
        log.info(f"{'='*60}")
        log.info(f"  Days: {len(daily_df)}, Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
        log.info(f"  Features: {len(feature_cols)}")

        cc = daily_df['cc_return_ticks'].dropna()
        log.info(f"  CC returns: mean={cc.mean():.1f}, std={cc.std():.1f}, "
                 f"range=[{cc.min():.0f}, {cc.max():.0f}] ticks")

        if HAS_MLFLOW:
            try:
                mlflow.log_metric("n_days", len(daily_df))
                mlflow.log_metric("n_features", len(feature_cols))
            except:
                pass

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 1: Intraday hold using 3-day direction classifier
        # ══════════════════════════════════════════════════════════════
        log.info(f"\n{'='*60}")
        log.info(f"STRATEGY 1: INTRADAY HOLD (3-Day Direction Classifier)")
        log.info(f"{'='*60}")

        # Train classifier for 3-day direction
        pred_3d, imp_3d = walk_forward_predict(
            daily_df, feature_cols, 'fwd_dir_3d', use_classifier=True
        )

        if len(pred_3d) > 0:
            # IC / AUC metrics
            from sklearn.metrics import roc_auc_score
            try:
                auc_3d = roc_auc_score(pred_3d['actual'], pred_3d['pred'])
            except:
                auc_3d = np.nan
            dir_acc = ((pred_3d['pred'] > 0.5) == (pred_3d['actual'] > 0.5)).mean()
            log.info(f"  3d Direction: AUC={auc_3d:.4f}, DirAcc={dir_acc:.4f}, N={len(pred_3d)}")

            if HAS_MLFLOW:
                try:
                    mlflow.log_metric("auc_3d_direction", round(auc_3d, 4))
                    mlflow.log_metric("dir_acc_3d", round(dir_acc, 4))
                except:
                    pass

            # Simulate intraday trades at various thresholds and hold periods
            all_intraday_results = {}
            best_intraday = None
            best_intraday_sharpe = -999

            for threshold_pct in [52, 55, 58, 60, 65]:
                for hold_min in [240, 360]:
                    for trailing in [True, False]:
                        label = f"intra_t{threshold_pct}_h{hold_min}_trail{trailing}"

                        trades = simulate_intraday_trades(
                            pred_3d, all_data,
                            hold_minutes=hold_min,
                            threshold_pct=threshold_pct,
                            stop_loss_ticks=STOP_LOSS_TICKS,
                            use_trailing_stop=trailing,
                            trailing_stop_ticks=TRAILING_STOP_TICKS,
                        )

                        if len(trades) < 5:
                            continue

                        metrics = compute_trading_metrics(trades, label)
                        all_intraday_results[label] = metrics

                        sharpe = metrics.get('sharpe_nw') or -999
                        if sharpe > best_intraday_sharpe:
                            best_intraday_sharpe = sharpe
                            best_intraday = (label, metrics, trades)

                        log.info(f"\n  {label}:")
                        log.info(f"    Trades={metrics['n_trades']}, WR={metrics['wr']}, PF={metrics['pf']}")
                        log.info(f"    Sharpe(NW)={metrics['sharpe_nw']}, Sortino={metrics['sortino']}")
                        log.info(f"    PnL={metrics['total_pnl_ticks']}t (${metrics['total_pnl_dollars']})")
                        if metrics.get('regime_gap') is not None:
                            status = "PASS" if metrics['regime_pass'] else "FAIL"
                            log.info(f"    Regime gap={metrics['regime_gap']} — {status}")
                        if metrics.get('exit_reasons'):
                            log.info(f"    Exits: {metrics['exit_reasons']}")
                        if metrics.get('mfe_stats'):
                            log.info(f"    MFE/MAE: {metrics['mfe_stats']}")

            if best_intraday:
                log.info(f"\n  >>> BEST INTRADAY: {best_intraday[0]}")
                log.info(f"      Sharpe(NW)={best_intraday[1]['sharpe_nw']}, "
                         f"WR={best_intraday[1]['wr']}, PF={best_intraday[1]['pf']}")

                if HAS_MLFLOW:
                    try:
                        bm = best_intraday[1]
                        mlflow.log_metric("best_intraday_sharpe_nw", bm.get('sharpe_nw') or 0)
                        mlflow.log_metric("best_intraday_wr", bm['wr'])
                        mlflow.log_metric("best_intraday_pf", bm['pf'] if bm['pf'] != 999 else 0)
                        mlflow.log_metric("best_intraday_n_trades", bm['n_trades'])
                        mlflow.log_metric("best_intraday_total_pnl", bm['total_pnl_ticks'])
                        if bm.get('regime_gap') is not None:
                            mlflow.log_metric("best_intraday_regime_gap", bm['regime_gap'])
                    except:
                        pass

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 2: Multi-day trend following
        # ══════════════════════════════════════════════════════════════
        log.info(f"\n{'='*60}")
        log.info(f"STRATEGY 2: MULTI-DAY TREND FOLLOWING")
        log.info(f"{'='*60}")

        all_multiday_results = {}
        best_multiday = None
        best_multiday_sharpe = -999

        for threshold_pct in [52, 55, 58, 60]:
            for max_hold in [2, 3, 5]:
                label = f"multiday_t{threshold_pct}_h{max_hold}d"

                trades = simulate_multiday_trades(
                    pred_3d, daily_df,
                    threshold_pct=threshold_pct,
                    max_hold_days=max_hold,
                    cost_rt=COST_RT_TICKS,
                )

                if len(trades) < 5:
                    continue

                metrics = compute_trading_metrics(trades, label)
                all_multiday_results[label] = metrics

                sharpe = metrics.get('sharpe_nw') or -999
                if sharpe > best_multiday_sharpe:
                    best_multiday_sharpe = sharpe
                    best_multiday = (label, metrics, trades)

                avg_hold = trades['hold_days'].mean() if 'hold_days' in trades.columns else 0
                log.info(f"\n  {label}:")
                log.info(f"    Trades={metrics['n_trades']}, WR={metrics['wr']}, PF={metrics['pf']}")
                log.info(f"    Sharpe(NW)={metrics['sharpe_nw']}, Sortino={metrics['sortino']}")
                log.info(f"    PnL={metrics['total_pnl_ticks']}t (${metrics['total_pnl_dollars']})")
                log.info(f"    Avg hold={avg_hold:.1f}d")
                if metrics.get('regime_gap') is not None:
                    status = "PASS" if metrics['regime_pass'] else "FAIL"
                    log.info(f"    Regime gap={metrics['regime_gap']} — {status}")
                if metrics.get('exit_reasons'):
                    log.info(f"    Exits: {metrics['exit_reasons']}")

        if best_multiday:
            log.info(f"\n  >>> BEST MULTI-DAY: {best_multiday[0]}")
            log.info(f"      Sharpe(NW)={best_multiday[1]['sharpe_nw']}, "
                     f"WR={best_multiday[1]['wr']}, PF={best_multiday[1]['pf']}")

            if HAS_MLFLOW:
                try:
                    bm = best_multiday[1]
                    mlflow.log_metric("best_multiday_sharpe_nw", bm.get('sharpe_nw') or 0)
                    mlflow.log_metric("best_multiday_wr", bm['wr'])
                    mlflow.log_metric("best_multiday_pf", bm['pf'] if bm['pf'] != 999 else 0)
                    mlflow.log_metric("best_multiday_n_trades", bm['n_trades'])
                    mlflow.log_metric("best_multiday_total_pnl", bm['total_pnl_ticks'])
                    if bm.get('regime_gap') is not None:
                        mlflow.log_metric("best_multiday_regime_gap", bm['regime_gap'])
                except:
                    pass

        # ══════════════════════════════════════════════════════════════
        # STRATEGY 3: Regression-based with 1d returns
        # ══════════════════════════════════════════════════════════════
        log.info(f"\n{'='*60}")
        log.info(f"STRATEGY 3: REGRESSION-BASED (1d forward return)")
        log.info(f"{'='*60}")

        pred_1d_reg, imp_1d = walk_forward_predict(
            daily_df, feature_cols, 'fwd_return_1d', use_classifier=False
        )

        if len(pred_1d_reg) > 0:
            ic = stats.spearmanr(pred_1d_reg['pred'], pred_1d_reg['actual'])[0]
            log.info(f"  1d Return Regression: IC={ic:.4f}, N={len(pred_1d_reg)}")

            if HAS_MLFLOW:
                try:
                    mlflow.log_metric("ic_1d_regression", round(ic, 4))
                except:
                    pass

            # Simulate with regression predictions
            pred_std = pred_1d_reg['pred'].std()
            for mult in [0.5, 1.0, 1.5, 2.0]:
                thresh = mult * pred_std
                label = f"reg_1d_thresh_{mult}x"

                # Convert regression to directional trades
                mask_long = pred_1d_reg['pred'] > thresh
                mask_short = pred_1d_reg['pred'] < -thresh

                if mask_long.sum() + mask_short.sum() < 5:
                    continue

                # Simple next-day trade
                trades_list = []
                for _, row in pred_1d_reg.iterrows():
                    if row['pred'] > thresh:
                        direction = 1
                    elif row['pred'] < -thresh:
                        direction = -1
                    else:
                        continue

                    pnl = direction * row['actual'] - COST_RT_TICKS
                    trades_list.append({
                        'trade_date': row['date'],
                        'direction': direction,
                        'pred': row['pred'],
                        'pnl_ticks': pnl,
                        'pnl_dollars': pnl * TICK_VALUE,
                        'regime': classify_regime(row['cc_return_ticks']),
                    })

                if len(trades_list) < 5:
                    continue

                tdf = pd.DataFrame(trades_list)
                metrics = compute_trading_metrics(tdf, label)

                log.info(f"\n  {label}:")
                log.info(f"    Trades={metrics['n_trades']}, WR={metrics['wr']}, PF={metrics['pf']}")
                log.info(f"    Sharpe(NW)={metrics['sharpe_nw']}, Sortino={metrics['sortino']}")
                log.info(f"    PnL={metrics['total_pnl_ticks']}t (${metrics['total_pnl_dollars']})")
                if metrics.get('regime_gap') is not None:
                    status = "PASS" if metrics['regime_pass'] else "FAIL"
                    log.info(f"    Regime gap={metrics['regime_gap']} — {status}")

        # ══════════════════════════════════════════════════════════════
        # FEATURE IMPORTANCE SUMMARY
        # ══════════════════════════════════════════════════════════════
        log.info(f"\n{'='*60}")
        log.info(f"FEATURE IMPORTANCE (3d classifier, accumulated gain)")
        log.info(f"{'='*60}")
        if imp_3d:
            sorted_imp = sorted(imp_3d.items(), key=lambda x: x[1], reverse=True)
            for fname, fval in sorted_imp[:20]:
                log.info(f"  {fname}: {fval:.1f}")

        # ══════════════════════════════════════════════════════════════
        # SAVE ALL RESULTS
        # ══════════════════════════════════════════════════════════════
        elapsed = time.time() - t0
        log.info(f"\n{'='*60}")
        log.info(f"COMPLETE — {elapsed:.0f}s elapsed")
        log.info(f"{'='*60}")

        # Comprehensive output
        output = {
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed, 1),
            'data': {
                'n_days': len(daily_df),
                'date_range': [str(daily_df['date'].min()), str(daily_df['date'].max())],
                'n_features': len(feature_cols),
                'feature_list': feature_cols,
            },
            'walk_forward': {
                'train_days': TRAIN_DAYS,
                'oot_days': OOT_DAYS,
                'slide_days': SLIDE_DAYS,
                'method': 'SLIDING (HC #0)',
            },
            'cost_model': {
                'cost_rt_ticks': COST_RT_TICKS,
                'description': 'Passive entry + aggressive exit',
                'commission_rt_ticks': COMMISSION_RT_TICKS,
                'spread_crossing_ticks': SPREAD_CROSSING_TICKS,
            },
            'predictions_3d': {
                'n_predictions': len(pred_3d),
                'auc': round(auc_3d, 4) if not np.isnan(auc_3d) else None,
                'dir_accuracy': round(dir_acc, 4),
            },
            'intraday_results': all_intraday_results,
            'multiday_results': all_multiday_results,
            'best_intraday': best_intraday[1] if best_intraday else None,
            'best_multiday': best_multiday[1] if best_multiday else None,
            'feature_importance': dict(sorted(imp_3d.items(), key=lambda x: x[1], reverse=True)[:20]) if imp_3d else {},
        }

        # Save results
        results_path = OUTPUT_DIR / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        log.info(f"Results saved to {results_path}")

        # Save daily features
        daily_df.to_parquet(OUTPUT_DIR / 'daily_features.parquet', index=False)

        # Save predictions
        if len(pred_3d) > 0:
            pred_3d.to_parquet(OUTPUT_DIR / 'predictions_3d_direction.parquet', index=False)
        if len(pred_1d_reg) > 0:
            pred_1d_reg.to_parquet(OUTPUT_DIR / 'predictions_1d_regression.parquet', index=False)

        # Save best trade logs
        if best_intraday and len(best_intraday[2]) > 0:
            best_intraday[2].to_parquet(OUTPUT_DIR / 'best_intraday_trades.parquet', index=False)
        if best_multiday and len(best_multiday[2]) > 0:
            best_multiday[2].to_parquet(OUTPUT_DIR / 'best_multiday_trades.parquet', index=False)

        # Log artifacts to MLflow
        if HAS_MLFLOW:
            try:
                mlflow.log_artifact(str(results_path))
                mlflow.log_metric("elapsed_seconds", round(elapsed, 1))
                mlflow.log_metric("n_intraday_configs", len(all_intraday_results))
                mlflow.log_metric("n_multiday_configs", len(all_multiday_results))
            except Exception as e:
                log.warning(f"MLflow artifact logging failed: {e}")

        # ══════════════════════════════════════════════════════════════
        # SUMMARY
        # ══════════════════════════════════════════════════════════════
        log.info(f"\n{'='*60}")
        log.info(f"SUMMARY")
        log.info(f"{'='*60}")

        if best_intraday:
            bi = best_intraday[1]
            log.info(f"\nBest Intraday: {best_intraday[0]}")
            log.info(f"  {bi['n_trades']} trades, WR {bi['wr']}, PF {bi['pf']}")
            log.info(f"  Sharpe(NW) {bi['sharpe_nw']}, Sortino {bi['sortino']}")
            log.info(f"  Total: {bi['total_pnl_ticks']}t / ${bi['total_pnl_dollars']}")
            log.info(f"  Regime gap: {bi.get('regime_gap')} ({'PASS' if bi.get('regime_pass') else 'FAIL'})")

        if best_multiday:
            bm = best_multiday[1]
            log.info(f"\nBest Multi-day: {best_multiday[0]}")
            log.info(f"  {bm['n_trades']} trades, WR {bm['wr']}, PF {bm['pf']}")
            log.info(f"  Sharpe(NW) {bm['sharpe_nw']}, Sortino {bm['sortino']}")
            log.info(f"  Total: {bm['total_pnl_ticks']}t / ${bm['total_pnl_dollars']}")
            log.info(f"  Regime gap: {bm.get('regime_gap')} ({'PASS' if bm.get('regime_pass') else 'FAIL'})")

        log.info(f"\nAll outputs in {OUTPUT_DIR}")

    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        if HAS_MLFLOW:
            try:
                mlflow.log_param("error", str(e)[:250])
            except:
                pass
        raise
    finally:
        if HAS_MLFLOW and mlflow_run:
            try:
                mlflow.end_run()
            except:
                pass


if __name__ == '__main__':
    main()
