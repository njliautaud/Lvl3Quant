#!/usr/bin/env python3
"""
deep_strategy_analysis_v1.py — Deep Analysis of Champion Strategy (30-min LGB + Daily OFI)

Reconstructs all 210 trades from multi_scale_combo_v1.py and performs:
  1. Trade clustering (time-of-day, DOW, confidence, volatility, momentum)
  2. Win rate by entry quality (z-score quantiles, OFI strength, cross-tabs)
  3. Loss analysis (SL speed, MFE before SL, near-miss detection)
  4. Opportunity cost (blocked trades, alternative thresholds)

FIFO fills only. HC #0: SLIDING windows. HC #428: Regime-agnostic.
"""

import os, sys, json, logging, warnings, time
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths (same as multi_scale_combo_v1) ──
ROOT = Path("/home/nick/Lvl3Quant")
DATA_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v1" / "daily_features.parquet"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
OUTPUT_DIR = ROOT / "output" / "deep_strategy_analysis_v1"
LOG_FILE = ROOT / "logs" / "deep_strategy_analysis_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DEEP-ANALYSIS] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants (from multi_scale_combo_v1) ──
TICK_SIZE = 1.0
TICK_SIZE_PTS = 0.25
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0

DAILY_TRAIN_DAYS = 40
DAILY_SLIDE_DAYS = 5

TP_LONG = 25
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD = 60
ENTRY_THRESHOLD = 0.05
CANCEL_WINDOW = 10
ENTRY_BAR_SIZE = 30

DAILY_FEATURE_COLS = [
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

LGB_PARAMS = {
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


# ──────────────────────────────────────────────────────────────────────
# COPIED FROM multi_scale_combo_v1.py (unchanged logic)
# ──────────────────────────────────────────────────────────────────────

def load_daily_features():
    df = pd.read_parquet(FEATURES_PATH)
    log.info(f"Loaded daily features: {len(df)} rows")
    return df

def load_30min_predictions():
    data = np.load(str(ENTRY_PREDS_PATH), allow_pickle=True)
    preds = data['entry_preds']
    dates = data['dates']
    log.info(f"Loaded 30-min predictions: {len(preds)} entries, {np.sum(~np.isnan(preds))} non-NaN")
    return preds, dates

def load_minute_bars():
    files = sorted(DATA_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df['date'] = f.stem
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined['ts_minute'] = pd.to_datetime(combined['ts_minute'], utc=True)
    combined = combined.sort_values('ts_minute').reset_index(drop=True)
    log.info(f"Loaded {len(combined):,} minute bars across {len(frames)} days")
    return combined

def aggregate_to_30min_bars(minute_df):
    df = minute_df.copy()
    df['bar_key'] = df['ts_minute'].dt.floor('30min')
    records = []
    for (date_str, bar_key), grp in df.groupby(['date', 'bar_key']):
        if len(grp) < 2:
            continue
        close_arr = grp['close'].values
        vol_arr = grp['volume'].values
        ofi_arr = grp['ofi_1min'].values
        rec = {
            'date': date_str,
            'bar_key': bar_key,
            'ts': grp['ts_minute'].iloc[0],
            'open': close_arr[0],
            'high': close_arr.max(),
            'low': close_arr.min(),
            'close': close_arr[-1],
            'total_volume': vol_arr.sum(),
            'ofi_sum': ofi_arr.sum(),
        }
        records.append(rec)
    result = pd.DataFrame(records)
    log.info(f"Aggregated {len(result):,} 30min bars")
    return result

def walk_forward_daily_contrarian(daily_df):
    n = len(daily_df)
    predictions = {}
    fold_id = 0
    start_idx = 0
    while start_idx + DAILY_TRAIN_DAYS < n:
        train_end = start_idx + DAILY_TRAIN_DAYS
        oot_end = min(train_end + DAILY_SLIDE_DAYS, n)
        train_df = daily_df.iloc[start_idx:train_end]
        oot_df = daily_df.iloc[train_end:oot_end]
        if len(oot_df) == 0:
            break
        fold_id += 1
        label = 'fwd_return_1d'
        train_mask = train_df[label].notna() & train_df[DAILY_FEATURE_COLS].notna().all(axis=1)
        oot_mask = oot_df[label].notna() & oot_df[DAILY_FEATURE_COLS].notna().all(axis=1)
        X_train = train_df.loc[train_mask, DAILY_FEATURE_COLS].values
        y_train = -train_df.loc[train_mask, label].values
        X_oot = oot_df.loc[oot_mask, DAILY_FEATURE_COLS].values
        if len(X_train) < 15 or len(X_oot) < 1:
            start_idx += DAILY_SLIDE_DAYS
            continue
        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=200)
        preds = model.predict(X_oot)
        for i in range(len(preds)):
            oot_idx = oot_df.index[oot_mask][i]
            row = daily_df.loc[oot_idx]
            date_str = row['date'].strftime('%Y%m%d')
            predictions[date_str] = float(preds[i])
        start_idx += DAILY_SLIDE_DAYS
    log.info(f"Daily contrarian WF: {fold_id} folds, {len(predictions)} date predictions")
    return predictions


# ──────────────────────────────────────────────────────────────────────
# ENHANCED TRADE RECONSTRUCTION (captures more metadata per trade)
# ──────────────────────────────────────────────────────────────────────

def reconstruct_entry_fills_enhanced(bars_30m, minute_df, pred_30m, confidence_pct, cancel_window_min):
    """Reconstruct FIFO entry fills with extra context for deep analysis."""
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]
    if len(valid_preds) < 20:
        return [], [], {}

    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - confidence_pct)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], confidence_pct)

    # Also compute z-scores for all predictions
    pred_mean = np.nanmean(pred_30m[valid_mask])
    pred_std = np.nanstd(pred_30m[valid_mask])

    minute_lookup = {}
    for date_str, grp in minute_df.groupby('date'):
        minute_lookup[date_str] = grp.sort_values('ts_minute').reset_index(drop=True)

    trades = []
    blocked_signals = []  # signals that were valid but didn't fill
    n_signals = 0
    n_filled = 0
    n_cancelled = 0

    bars_ts = bars_30m['ts'].values
    bars_dates = bars_30m['date'].values
    bars_close = bars_30m['close'].values
    bars_ofi = bars_30m['ofi_sum'].values
    bars_volume = bars_30m['total_volume'].values
    bars_high = bars_30m['high'].values
    bars_low = bars_30m['low'].values

    for i in range(len(bars_30m)):
        if np.isnan(pred_30m[i]):
            continue
        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1
        elif pred_30m[i] <= lower_thresh:
            direction = -1
        else:
            continue

        n_signals += 1
        date_str = bars_dates[i]
        signal_ts = pd.Timestamp(bars_ts[i])
        signal_price = bars_close[i]

        # Compute prediction z-score
        pred_zscore = (pred_30m[i] - pred_mean) / pred_std if pred_std > 0 else 0.0

        # Compute bar-level volatility context (range of the signal bar)
        bar_range = bars_high[i] - bars_low[i]

        # Compute local momentum (last 5 bars close-to-close)
        lookback = min(i, 5)
        if lookback > 1:
            local_momentum = bars_close[i] - bars_close[i - lookback]
        else:
            local_momentum = 0.0

        # Pre-entry volatility: avg range of last 5 minute bars before signal
        pre_entry_vol = np.nan
        if date_str in minute_lookup:
            day_minutes = minute_lookup[date_str]
            day_ts = day_minutes['ts_minute'].values
            signal_ts_np = np.datetime64(signal_ts)
            before_mask = day_ts < signal_ts_np
            if before_mask.sum() >= 5:
                pre_bars = day_minutes[before_mask].tail(5)
                pre_entry_vol = (pre_bars['high'] - pre_bars['low']).mean()

        if date_str not in minute_lookup:
            n_cancelled += 1
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes['ts_minute'].values
        limit_price = signal_price
        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE)
        signal_bar_end_np = np.datetime64(signal_bar_end)
        cancel_ts = signal_ts + pd.Timedelta(minutes=cancel_window_min + ENTRY_BAR_SIZE)
        cancel_ts_np = np.datetime64(cancel_ts)

        fill_mask = (day_ts >= signal_bar_end_np) & (day_ts <= cancel_ts_np)
        fill_candidates = day_minutes[fill_mask]

        if len(fill_candidates) == 0:
            n_cancelled += 1
            blocked_signals.append({
                'idx': i, 'date': date_str, 'signal_ts': signal_ts,
                'direction': direction, 'pred_30m': float(pred_30m[i]),
                'pred_zscore': float(pred_zscore), 'reason': 'no_fill_candidates',
            })
            continue

        filled = False
        fill_price = None
        fill_ts = None
        fill_delay = 0

        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                if mbar['low'] <= limit_price - TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    fill_delay = j + 1
                    break
            else:
                if mbar['high'] >= limit_price + TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    fill_delay = j + 1
                    break

        if not filled:
            n_cancelled += 1
            blocked_signals.append({
                'idx': i, 'date': date_str, 'signal_ts': signal_ts,
                'direction': direction, 'pred_30m': float(pred_30m[i]),
                'pred_zscore': float(pred_zscore), 'reason': 'no_fill_in_window',
            })
            continue

        n_filled += 1
        fill_ts_np = np.datetime64(fill_ts)
        remaining_mask = day_ts >= fill_ts_np
        remaining_minutes = day_minutes[remaining_mask]

        if len(remaining_minutes) < 2:
            n_filled -= 1
            n_cancelled += 1
            continue

        # Time-of-day features
        fill_ts_pd = pd.Timestamp(fill_ts)
        if fill_ts_pd.tzinfo is not None:
            fill_hour = fill_ts_pd.tz_convert('US/Eastern').hour
            fill_minute = fill_ts_pd.tz_convert('US/Eastern').minute
        else:
            fill_hour = fill_ts_pd.hour
            fill_minute = fill_ts_pd.minute

        # Day of week
        dow = fill_ts_pd.dayofweek  # 0=Mon, 4=Fri

        trade = {
            'idx': i,
            'date': date_str,
            'signal_ts': signal_ts,
            'fill_ts': pd.Timestamp(fill_ts),
            'fill_price': fill_price,
            'fill_delay_minutes': fill_delay,
            'direction': direction,
            'pred_30m': float(pred_30m[i]),
            'pred_zscore': float(pred_zscore),
            'bar_range': float(bar_range),
            'bar_ofi': float(bars_ofi[i]),
            'bar_volume': float(bars_volume[i]),
            'pre_entry_vol': float(pre_entry_vol) if not np.isnan(pre_entry_vol) else None,
            'local_momentum': float(local_momentum),
            'fill_hour_et': fill_hour,
            'fill_minute_et': fill_minute,
            'day_of_week': dow,
            'prices_close': remaining_minutes['close'].values.copy(),
            'prices_high': remaining_minutes['high'].values.copy(),
            'prices_low': remaining_minutes['low'].values.copy(),
            'times': remaining_minutes['ts_minute'].values.copy(),
            'n_remaining_minutes': len(remaining_minutes),
        }
        trades.append(trade)

    fill_stats = {
        'n_signals': n_signals,
        'n_filled': n_filled,
        'n_cancelled': n_cancelled,
        'fill_rate': n_filled / max(n_signals, 1),
        'upper_thresh': float(upper_thresh),
        'lower_thresh': float(lower_thresh),
    }
    log.info(f"  Entry fills: signals={n_signals}, filled={n_filled} ({fill_stats['fill_rate']:.1%}), cancelled={n_cancelled}")
    return trades, blocked_signals, fill_stats


def simulate_fifo_exits_detailed(trades, tp_long, tp_short, sl_long, sl_short, max_hold_minutes):
    """Simulate FIFO exits returning detailed per-trade results."""
    results = []

    for trade in trades:
        direction = trade['direction']
        fill_price = trade['fill_price']
        prices_close = trade['prices_close']
        prices_high = trade['prices_high']
        prices_low = trade['prices_low']
        n_remaining = trade['n_remaining_minutes']

        tp_ticks = tp_long if direction == 1 else tp_short
        sl_ticks = sl_long if direction == 1 else sl_short

        tp_price = fill_price + direction * tp_ticks * TICK_SIZE
        sl_price = fill_price - direction * sl_ticks * TICK_SIZE

        # Track MFE and MAE
        mfe = 0.0  # max favorable excursion in ticks
        mae = 0.0  # max adverse excursion in ticks

        exit_type = 'time'
        exit_minute = 0
        trade_pnl = 0.0
        exit_found = False
        max_check = min(max_hold_minutes, n_remaining)

        for m in range(1, max_check):
            bar_high = prices_high[m]
            bar_low = prices_low[m]
            bar_close = prices_close[m]

            # Update MFE/MAE
            if direction == 1:
                fav = (bar_high - fill_price) / TICK_SIZE
                adv = (fill_price - bar_low) / TICK_SIZE
            else:
                fav = (fill_price - bar_low) / TICK_SIZE
                adv = (bar_high - fill_price) / TICK_SIZE

            mfe = max(mfe, fav)
            mae = max(mae, adv)

            sl_hit = tp_hit = False
            if direction == 1:
                if bar_low <= sl_price:
                    sl_hit = True
                if bar_high >= tp_price + TICK_SIZE:
                    tp_hit = True
            else:
                if bar_high >= sl_price:
                    sl_hit = True
                if bar_low <= tp_price - TICK_SIZE:
                    tp_hit = True

            if sl_hit and tp_hit:
                sl_hit = True
                tp_hit = False

            if sl_hit:
                trade_pnl = -(sl_ticks + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                exit_type = 'sl'
                exit_minute = m
                exit_found = True
                break

            if tp_hit:
                trade_pnl = tp_ticks - RT_COMMISSION_TICKS
                exit_type = 'tp'
                exit_minute = m
                exit_found = True
                break

        if not exit_found:
            exit_minute_val = min(max_hold_minutes, n_remaining - 1)
            exit_minute_val = max(exit_minute_val, 1)
            exit_close = prices_close[exit_minute_val]
            if direction == 1:
                exit_fill = exit_close - TICK_SIZE
            else:
                exit_fill = exit_close + TICK_SIZE
            raw_pnl_ticks = (exit_fill - fill_price) / TICK_SIZE * direction
            trade_pnl = raw_pnl_ticks - RT_COMMISSION_TICKS
            exit_type = 'time'
            exit_minute = exit_minute_val

        result = {
            'date': trade['date'],
            'signal_ts': trade['signal_ts'],
            'fill_ts': trade['fill_ts'],
            'fill_price': trade['fill_price'],
            'direction': trade['direction'],
            'pred_30m': trade['pred_30m'],
            'pred_zscore': trade['pred_zscore'],
            'bar_range': trade['bar_range'],
            'bar_ofi': trade['bar_ofi'],
            'bar_volume': trade['bar_volume'],
            'pre_entry_vol': trade['pre_entry_vol'],
            'local_momentum': trade['local_momentum'],
            'fill_hour_et': trade['fill_hour_et'],
            'fill_minute_et': trade.get('fill_minute_et', 0),
            'day_of_week': trade['day_of_week'],
            'fill_delay_minutes': trade['fill_delay_minutes'],
            'pnl_ticks': float(trade_pnl),
            'exit_type': exit_type,
            'exit_minute': exit_minute,
            'mfe_ticks': float(mfe),
            'mae_ticks': float(mae),
            'winner': trade_pnl > 0,
        }
        results.append(result)

    return results


# ──────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def analyze_trade_clustering(df):
    """Section 1: Cluster winners vs losers by various dimensions."""
    log.info("\n" + "=" * 70)
    log.info("SECTION 1: TRADE CLUSTERING — Winners vs Losers")
    log.info("=" * 70)

    results = {}

    # 1a. Time of day
    log.info("\n--- 1a. Time of Day ---")
    df['time_slot'] = df['fill_hour_et'].astype(str).str.zfill(2) + ':' + (df['fill_minute_et'] // 30 * 30).astype(str).str.zfill(2)
    time_stats = df.groupby('time_slot').agg(
        n_trades=('pnl_ticks', 'count'),
        n_wins=('winner', 'sum'),
        mean_pnl=('pnl_ticks', 'mean'),
        total_pnl=('pnl_ticks', 'sum'),
        mean_mfe=('mfe_ticks', 'mean'),
    ).reset_index()
    time_stats['wr'] = time_stats['n_wins'] / time_stats['n_trades']
    time_stats = time_stats.sort_values('time_slot')

    log.info(f"{'Time':>8} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'AvgMFE':>8}")
    log.info("-" * 60)
    for _, row in time_stats.iterrows():
        log.info(f"{row['time_slot']:>8} {row['n_trades']:5.0f} {row['n_wins']:5.0f} "
                 f"{row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['total_pnl']:10.1f} {row['mean_mfe']:8.1f}")
    results['time_of_day'] = time_stats.to_dict('records')

    # 1b. Day of week
    log.info("\n--- 1b. Day of Week ---")
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    df['dow_name'] = df['day_of_week'].map(dow_names)
    dow_stats = df.groupby('dow_name').agg(
        n_trades=('pnl_ticks', 'count'),
        n_wins=('winner', 'sum'),
        mean_pnl=('pnl_ticks', 'mean'),
        total_pnl=('pnl_ticks', 'sum'),
    ).reset_index()
    dow_stats['wr'] = dow_stats['n_wins'] / dow_stats['n_trades']

    log.info(f"{'Day':>5} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'TotalPnL':>10}")
    log.info("-" * 45)
    for _, row in dow_stats.iterrows():
        log.info(f"{row['dow_name']:>5} {row['n_trades']:5.0f} {row['n_wins']:5.0f} "
                 f"{row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['total_pnl']:10.1f}")
    results['day_of_week'] = dow_stats.to_dict('records')

    # 1c. Prediction confidence buckets
    log.info("\n--- 1c. Prediction Confidence ---")
    abs_zscore = df['pred_zscore'].abs()
    df['confidence_bucket'] = pd.cut(abs_zscore,
        bins=[0, abs_zscore.quantile(0.33), abs_zscore.quantile(0.66), abs_zscore.quantile(0.90), abs_zscore.quantile(0.99), np.inf],
        labels=['Bottom 33%', 'Mid 33-66%', 'High 66-90%', 'Top 10%', 'Top 1%'],
        include_lowest=True
    )
    conf_stats = df.groupby('confidence_bucket', observed=True).agg(
        n_trades=('pnl_ticks', 'count'),
        n_wins=('winner', 'sum'),
        mean_pnl=('pnl_ticks', 'mean'),
        total_pnl=('pnl_ticks', 'sum'),
        mean_mfe=('mfe_ticks', 'mean'),
        mean_mae=('mae_ticks', 'mean'),
    ).reset_index()
    conf_stats['wr'] = conf_stats['n_wins'] / conf_stats['n_trades']

    log.info(f"{'Confidence':>15} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'MFE':>6} {'MAE':>6}")
    log.info("-" * 60)
    for _, row in conf_stats.iterrows():
        log.info(f"{row['confidence_bucket']:>15} {row['n_trades']:5.0f} {row['n_wins']:5.0f} "
                 f"{row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['mean_mfe']:6.1f} {row['mean_mae']:6.1f}")
    results['confidence'] = conf_stats.to_dict('records')

    # 1d. Market volatility context
    log.info("\n--- 1d. Pre-Entry Volatility ---")
    vol_valid = df[df['pre_entry_vol'].notna()].copy()
    if len(vol_valid) > 10:
        vol_valid['vol_bucket'] = pd.qcut(vol_valid['pre_entry_vol'], q=4,
            labels=['Low Vol', 'Med-Low', 'Med-High', 'High Vol'])
        vol_stats = vol_valid.groupby('vol_bucket', observed=True).agg(
            n_trades=('pnl_ticks', 'count'),
            n_wins=('winner', 'sum'),
            mean_pnl=('pnl_ticks', 'mean'),
            total_pnl=('pnl_ticks', 'sum'),
        ).reset_index()
        vol_stats['wr'] = vol_stats['n_wins'] / vol_stats['n_trades']

        log.info(f"{'Volatility':>12} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'TotalPnL':>10}")
        log.info("-" * 50)
        for _, row in vol_stats.iterrows():
            log.info(f"{row['vol_bucket']:>12} {row['n_trades']:5.0f} {row['n_wins']:5.0f} "
                     f"{row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['total_pnl']:10.1f}")
        results['volatility'] = vol_stats.to_dict('records')

    # 1e. Momentum context
    log.info("\n--- 1e. Momentum Context (trending vs mean-reverting) ---")
    df['momentum_bucket'] = pd.qcut(df['local_momentum'], q=4,
        labels=['Strong Down', 'Mild Down', 'Mild Up', 'Strong Up'],
        duplicates='drop')
    mom_stats = df.groupby('momentum_bucket', observed=True).agg(
        n_trades=('pnl_ticks', 'count'),
        n_wins=('winner', 'sum'),
        mean_pnl=('pnl_ticks', 'mean'),
        n_long=('direction', lambda x: (x == 1).sum()),
        n_short=('direction', lambda x: (x == -1).sum()),
    ).reset_index()
    mom_stats['wr'] = mom_stats['n_wins'] / mom_stats['n_trades']

    log.info(f"{'Momentum':>15} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'Long':>5} {'Short':>5}")
    log.info("-" * 60)
    for _, row in mom_stats.iterrows():
        log.info(f"{row['momentum_bucket']:>15} {row['n_trades']:5.0f} {row['n_wins']:5.0f} "
                 f"{row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['n_long']:5.0f} {row['n_short']:5.0f}")
    results['momentum'] = mom_stats.to_dict('records')

    return results


def analyze_entry_quality(df, daily_bias):
    """Section 2: Win rate by entry quality cross-tabulations."""
    log.info("\n" + "=" * 70)
    log.info("SECTION 2: WIN RATE BY ENTRY QUALITY")
    log.info("=" * 70)

    results = {}

    # 2a. Z-score quantiles
    log.info("\n--- 2a. Prediction Z-Score Quantiles ---")
    df['zscore_quantile'] = pd.qcut(df['pred_zscore'].abs(), q=5,
        labels=['Q1 (weakest)', 'Q2', 'Q3', 'Q4', 'Q5 (strongest)'],
        duplicates='drop')
    zq_stats = df.groupby('zscore_quantile', observed=True).agg(
        n_trades=('pnl_ticks', 'count'),
        n_wins=('winner', 'sum'),
        mean_pnl=('pnl_ticks', 'mean'),
        total_pnl=('pnl_ticks', 'sum'),
        mean_mfe=('mfe_ticks', 'mean'),
        mean_mae=('mae_ticks', 'mean'),
    ).reset_index()
    zq_stats['wr'] = zq_stats['n_wins'] / zq_stats['n_trades']

    log.info(f"{'Z-Score Q':>15} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'MFE':>6} {'MAE':>6}")
    log.info("-" * 60)
    for _, row in zq_stats.iterrows():
        log.info(f"{row['zscore_quantile']:>15} {row['n_trades']:5.0f} {row['n_wins']:5.0f} "
                 f"{row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['mean_mfe']:6.1f} {row['mean_mae']:6.1f}")
    results['zscore_quantiles'] = zq_stats.to_dict('records')

    # 2b. Daily OFI strength
    log.info("\n--- 2b. Daily OFI Bias Strength ---")
    df_bias = df.copy()
    df_bias['daily_bias'] = df_bias['date'].map(daily_bias)
    df_with_bias = df_bias[df_bias['daily_bias'].notna()].copy()

    if len(df_with_bias) > 20:
        df_with_bias['bias_strength'] = pd.qcut(df_with_bias['daily_bias'].abs(), q=4,
            labels=['Weak Bias', 'Mild Bias', 'Moderate Bias', 'Strong Bias'],
            duplicates='drop')
        bias_stats = df_with_bias.groupby('bias_strength', observed=True).agg(
            n_trades=('pnl_ticks', 'count'),
            n_wins=('winner', 'sum'),
            mean_pnl=('pnl_ticks', 'mean'),
            total_pnl=('pnl_ticks', 'sum'),
        ).reset_index()
        bias_stats['wr'] = bias_stats['n_wins'] / bias_stats['n_trades']

        log.info(f"{'Bias Strength':>15} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'TotalPnL':>10}")
        log.info("-" * 60)
        for _, row in bias_stats.iterrows():
            log.info(f"{row['bias_strength']:>15} {row['n_trades']:5.0f} {row['n_wins']:5.0f} "
                     f"{row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['total_pnl']:10.1f}")
        results['bias_strength'] = bias_stats.to_dict('records')

        # 2c. Cross-tab: prediction strength x bias strength
        log.info("\n--- 2c. Cross-Tab: Prediction Confidence x Daily Bias Strength ---")
        # Use 2x2 for clarity: high/low pred x high/low bias
        pred_median = df_with_bias['pred_zscore'].abs().median()
        bias_median = df_with_bias['daily_bias'].abs().median()

        df_with_bias['pred_level'] = np.where(df_with_bias['pred_zscore'].abs() >= pred_median, 'HighPred', 'LowPred')
        df_with_bias['bias_level'] = np.where(df_with_bias['daily_bias'].abs() >= bias_median, 'StrongBias', 'WeakBias')

        cross_stats = df_with_bias.groupby(['pred_level', 'bias_level']).agg(
            n_trades=('pnl_ticks', 'count'),
            n_wins=('winner', 'sum'),
            mean_pnl=('pnl_ticks', 'mean'),
            total_pnl=('pnl_ticks', 'sum'),
            mean_mfe=('mfe_ticks', 'mean'),
        ).reset_index()
        cross_stats['wr'] = cross_stats['n_wins'] / cross_stats['n_trades']

        log.info(f"{'Pred':>10} {'Bias':>12} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'TotalPnL':>10}")
        log.info("-" * 65)
        for _, row in cross_stats.iterrows():
            log.info(f"{row['pred_level']:>10} {row['bias_level']:>12} {row['n_trades']:5.0f} "
                     f"{row['n_wins']:5.0f} {row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['total_pnl']:10.1f}")
        results['cross_tab'] = cross_stats.to_dict('records')

        # 2d. Alignment analysis: does bias direction match trade direction?
        log.info("\n--- 2d. Bias-Trade Alignment ---")
        # bias > 0 = expect DOWN = favors shorts; bias < 0 = expect UP = favors longs
        df_with_bias['aligned'] = (
            ((df_with_bias['daily_bias'] > 0) & (df_with_bias['direction'] == -1)) |
            ((df_with_bias['daily_bias'] < 0) & (df_with_bias['direction'] == 1))
        )
        align_stats = df_with_bias.groupby('aligned').agg(
            n_trades=('pnl_ticks', 'count'),
            n_wins=('winner', 'sum'),
            mean_pnl=('pnl_ticks', 'mean'),
            total_pnl=('pnl_ticks', 'sum'),
            mean_mfe=('mfe_ticks', 'mean'),
            mean_mae=('mae_ticks', 'mean'),
        ).reset_index()
        align_stats['wr'] = align_stats['n_wins'] / align_stats['n_trades']
        align_stats['aligned'] = align_stats['aligned'].map({True: 'ALIGNED', False: 'AGAINST'})

        log.info(f"{'Alignment':>10} {'N':>5} {'Wins':>5} {'WR':>7} {'AvgPnL':>8} {'TotalPnL':>10} {'MFE':>6} {'MAE':>6}")
        log.info("-" * 70)
        for _, row in align_stats.iterrows():
            log.info(f"{row['aligned']:>10} {row['n_trades']:5.0f} {row['n_wins']:5.0f} "
                     f"{row['wr']:7.1%} {row['mean_pnl']:8.2f} {row['total_pnl']:10.1f} "
                     f"{row['mean_mfe']:6.1f} {row['mean_mae']:6.1f}")
        results['alignment'] = align_stats.to_dict('records')

    return results


def analyze_losses(df):
    """Section 3: Deep analysis of losing trades."""
    log.info("\n" + "=" * 70)
    log.info("SECTION 3: LOSS ANALYSIS")
    log.info("=" * 70)

    results = {}
    losers = df[~df['winner']].copy()
    winners = df[df['winner']].copy()
    sl_trades = df[df['exit_type'] == 'sl'].copy()
    tp_trades = df[df['exit_type'] == 'tp'].copy()
    time_trades = df[df['exit_type'] == 'time'].copy()

    log.info(f"\nTotal trades: {len(df)}")
    log.info(f"  Winners: {len(winners)} ({len(winners)/len(df):.1%})")
    log.info(f"  Losers:  {len(losers)} ({len(losers)/len(df):.1%})")
    log.info(f"\nExit type breakdown:")
    log.info(f"  TP exits:   {len(tp_trades):4d} ({len(tp_trades)/len(df):.1%})")
    log.info(f"  SL exits:   {len(sl_trades):4d} ({len(sl_trades)/len(df):.1%})")
    log.info(f"  Time exits: {len(time_trades):4d} ({len(time_trades)/len(df):.1%})")

    # 3a. How quickly do SL trades hit SL?
    log.info("\n--- 3a. Speed to Stop Loss ---")
    if len(sl_trades) > 0:
        sl_minutes = sl_trades['exit_minute']
        log.info(f"  SL exit minute distribution:")
        log.info(f"    Min:    {sl_minutes.min():.0f}")
        log.info(f"    p10:    {sl_minutes.quantile(0.10):.0f}")
        log.info(f"    p25:    {sl_minutes.quantile(0.25):.0f}")
        log.info(f"    Median: {sl_minutes.median():.0f}")
        log.info(f"    p75:    {sl_minutes.quantile(0.75):.0f}")
        log.info(f"    p90:    {sl_minutes.quantile(0.90):.0f}")
        log.info(f"    Max:    {sl_minutes.max():.0f}")

        # Distribution buckets
        sl_speed_buckets = pd.cut(sl_minutes, bins=[0, 2, 5, 10, 20, 30, 60, np.inf],
            labels=['0-2m', '2-5m', '5-10m', '10-20m', '20-30m', '30-60m', '60m+'])
        speed_dist = sl_speed_buckets.value_counts().sort_index()
        log.info(f"\n  SL Speed Distribution:")
        for bucket, count in speed_dist.items():
            pct = count / len(sl_trades)
            bar = '#' * int(pct * 40)
            log.info(f"    {bucket:>8}: {count:4d} ({pct:5.1%}) {bar}")
        results['sl_speed'] = {str(k): int(v) for k, v in speed_dist.items()}

    # 3b. MFE before SL (how close did losers come to winning?)
    log.info("\n--- 3b. Maximum Favorable Excursion Before Stop Loss ---")
    if len(sl_trades) > 0:
        sl_mfe = sl_trades['mfe_ticks']
        log.info(f"  MFE of SL trades (ticks):")
        log.info(f"    Mean:   {sl_mfe.mean():.1f}")
        log.info(f"    Median: {sl_mfe.median():.1f}")
        log.info(f"    p75:    {sl_mfe.quantile(0.75):.1f}")
        log.info(f"    p90:    {sl_mfe.quantile(0.90):.1f}")
        log.info(f"    Max:    {sl_mfe.max():.1f}")

        # What fraction had significant MFE?
        for mfe_thresh in [1, 2, 3, 5, 10]:
            count = (sl_mfe >= mfe_thresh).sum()
            pct = count / len(sl_trades)
            log.info(f"    MFE >= {mfe_thresh:2d} ticks: {count:4d} ({pct:5.1%})")
        results['sl_mfe_stats'] = {
            'mean': float(sl_mfe.mean()),
            'median': float(sl_mfe.median()),
            'p75': float(sl_mfe.quantile(0.75)),
            'p90': float(sl_mfe.quantile(0.90)),
        }

    # 3c. Near-miss analysis: SL trades with high MFE
    log.info("\n--- 3c. Near-Miss Losers (SL hit but had MFE >= 3 ticks) ---")
    if len(sl_trades) > 0:
        near_miss = sl_trades[sl_trades['mfe_ticks'] >= 3.0]
        log.info(f"  Near-miss SL trades (MFE >= 3): {len(near_miss)} / {len(sl_trades)} SL trades ({len(near_miss)/max(len(sl_trades),1):.1%})")
        if len(near_miss) > 0:
            log.info(f"  Their avg MFE: {near_miss['mfe_ticks'].mean():.1f} ticks")
            log.info(f"  Their avg SL exit minute: {near_miss['exit_minute'].mean():.1f}")
            log.info(f"  These trades saw {near_miss['mfe_ticks'].mean():.1f} ticks favorable before reversing to SL")

            # Could a trailing stop have saved them?
            # If MFE >= SL + 1, a breakeven move would have saved them
            sl_longs = near_miss[near_miss['direction'] == 1]
            sl_shorts = near_miss[near_miss['direction'] == -1]
            be_saved_long = (sl_longs['mfe_ticks'] >= SL_LONG).sum() if len(sl_longs) > 0 else 0
            be_saved_short = (sl_shorts['mfe_ticks'] >= SL_SHORT).sum() if len(sl_shorts) > 0 else 0
            total_saveable = be_saved_long + be_saved_short
            log.info(f"  Saveable by breakeven stop (MFE >= SL): {total_saveable} trades")

        results['near_miss_count'] = len(near_miss) if len(sl_trades) > 0 else 0

    # 3d. Time exits analysis
    log.info("\n--- 3d. Time Exit Analysis ---")
    if len(time_trades) > 0:
        time_pnl = time_trades['pnl_ticks']
        log.info(f"  Time exits: {len(time_trades)}")
        log.info(f"  Time exit PnL: mean={time_pnl.mean():.2f}, median={time_pnl.median():.2f}")
        log.info(f"  Time exit winners: {(time_pnl > 0).sum()} ({(time_pnl > 0).mean():.1%})")
        log.info(f"  Time exit MFE: mean={time_trades['mfe_ticks'].mean():.1f}, max={time_trades['mfe_ticks'].max():.1f}")
        results['time_exits'] = {
            'count': len(time_trades),
            'mean_pnl': float(time_pnl.mean()),
            'wr': float((time_pnl > 0).mean()),
            'mean_mfe': float(time_trades['mfe_ticks'].mean()),
        }

    # 3e. SL sensitivity: what if SL was wider?
    log.info("\n--- 3e. Stop Loss Sensitivity ---")
    log.info("  What if SL were wider? (checking MFE vs MAE of SL-exited trades)")
    if len(sl_trades) > 0:
        for alt_sl in [5, 6, 8, 10, 15]:
            # Trades that hit SL_LONG=4/SL_SHORT=3 but had MFE >= alt_sl - original SL
            # i.e., with wider SL they might have survived and reached TP
            # Check: how many SL trades had MAE < alt_sl? They'd survive with wider SL
            longs_sl = sl_trades[sl_trades['direction'] == 1]
            shorts_sl = sl_trades[sl_trades['direction'] == -1]
            survive_long = (longs_sl['mae_ticks'] < alt_sl).sum() if len(longs_sl) > 0 else 0
            survive_short = (shorts_sl['mae_ticks'] < alt_sl).sum() if len(shorts_sl) > 0 else 0
            total_survive = survive_long + survive_short
            # Of those that survive, how many would eventually hit TP?
            would_tp_long = ((longs_sl['mae_ticks'] < alt_sl) & (longs_sl['mfe_ticks'] >= TP_LONG)).sum() if len(longs_sl) > 0 else 0
            would_tp_short = ((shorts_sl['mae_ticks'] < alt_sl) & (shorts_sl['mfe_ticks'] >= TP_SHORT)).sum() if len(shorts_sl) > 0 else 0
            would_tp = would_tp_long + would_tp_short
            log.info(f"  SL={alt_sl}: {total_survive}/{len(sl_trades)} survive ({total_survive/max(len(sl_trades),1):.1%}), "
                     f"{would_tp} would hit TP ({would_tp/max(len(sl_trades),1):.1%})")

    return results


def analyze_opportunity_cost(df, all_trades_unfiltered, blocked_signals, daily_bias, minute_df,
                            bars_30m, pred_30m):
    """Section 4: What happened to blocked/filtered trades?"""
    log.info("\n" + "=" * 70)
    log.info("SECTION 4: OPPORTUNITY COST ANALYSIS")
    log.info("=" * 70)

    results = {}

    # 4a. Blocked by daily bias filter
    log.info("\n--- 4a. Trades Blocked by Daily Bias Filter ---")
    # Identify which trades from unfiltered set are NOT in the filtered set
    filtered_dates = set(df['date'].astype(str) + '_' + df['direction'].astype(str) + '_' + df['fill_price'].astype(str))
    unfiltered_df = pd.DataFrame(all_trades_unfiltered)
    unfiltered_df['key'] = unfiltered_df['date'].astype(str) + '_' + unfiltered_df['direction'].astype(str) + '_' + unfiltered_df['fill_price'].astype(str)

    blocked = unfiltered_df[~unfiltered_df['key'].isin(filtered_dates)]
    log.info(f"  Total unfiltered trades: {len(unfiltered_df)}")
    log.info(f"  Filtered (champion): {len(df)}")
    log.info(f"  Blocked by filter: {len(blocked)}")

    if len(blocked) > 0:
        blocked_pnl = blocked['pnl_ticks']
        blocked_wr = (blocked_pnl > 0).mean()
        log.info(f"  Blocked trade stats:")
        log.info(f"    WR: {blocked_wr:.1%}")
        log.info(f"    Mean PnL: {blocked_pnl.mean():.2f} ticks")
        log.info(f"    Total PnL: {blocked_pnl.sum():.1f} ticks")
        log.info(f"    Filter was {'CORRECT (blocked bad trades)' if blocked_pnl.mean() < df['pnl_ticks'].mean() else 'WRONG (blocked good trades)'}")

        # Breakdown by direction
        blocked_longs = blocked[blocked['direction'] == 1]
        blocked_shorts = blocked[blocked['direction'] == -1]
        if len(blocked_longs) > 0:
            log.info(f"    Blocked longs:  {len(blocked_longs)}, WR={blocked_longs['pnl_ticks'].gt(0).mean():.1%}, avg={blocked_longs['pnl_ticks'].mean():.2f}")
        if len(blocked_shorts) > 0:
            log.info(f"    Blocked shorts: {len(blocked_shorts)}, WR={blocked_shorts['pnl_ticks'].gt(0).mean():.1%}, avg={blocked_shorts['pnl_ticks'].mean():.2f}")

        results['blocked_trades'] = {
            'count': len(blocked),
            'wr': float(blocked_wr),
            'mean_pnl': float(blocked_pnl.mean()),
            'total_pnl': float(blocked_pnl.sum()),
        }

    # 4b. What if we used different confidence thresholds?
    log.info("\n--- 4b. Alternative Confidence Thresholds ---")
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]

    for alt_pct in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        alt_upper = np.nanquantile(pred_30m[valid_mask], 1 - alt_pct)
        alt_lower = np.nanquantile(pred_30m[valid_mask], alt_pct)
        n_signals = 0
        for i in range(len(pred_30m)):
            if np.isnan(pred_30m[i]):
                continue
            if pred_30m[i] >= alt_upper or pred_30m[i] <= alt_lower:
                n_signals += 1
        log.info(f"  Top {alt_pct*100:.0f}%: ~{n_signals} raw signals (before fill/filter)")

    # 4c. Cancelled signals analysis (signals that didn't fill)
    log.info("\n--- 4c. Cancelled Signals (no fill within window) ---")
    if len(blocked_signals) > 0:
        bs_df = pd.DataFrame(blocked_signals)
        log.info(f"  Total cancelled signals: {len(bs_df)}")
        reason_counts = bs_df['reason'].value_counts()
        for reason, count in reason_counts.items():
            log.info(f"    {reason}: {count}")

        # What was the distribution of cancelled signal confidence?
        if 'pred_zscore' in bs_df.columns:
            log.info(f"  Cancelled signal z-score: mean={bs_df['pred_zscore'].abs().mean():.2f}, "
                     f"max={bs_df['pred_zscore'].abs().max():.2f}")

        # What would have happened if these trades filled at the next available price?
        # (Hypothetical market order fill)
        log.info(f"  Cancelled signal directions: Long={len(bs_df[bs_df['direction']==1])}, Short={len(bs_df[bs_df['direction']==-1])}")
        results['cancelled_signals'] = {
            'count': len(bs_df),
            'reasons': {str(k): int(v) for k, v in reason_counts.items()},
        }

    return results


def generate_executive_summary(df, clustering, entry_quality, losses, opportunity, daily_bias):
    """Generate a clean executive summary."""
    log.info("\n" + "=" * 70)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 70)

    n = len(df)
    winners = df[df['winner']]
    losers = df[~df['winner']]
    wr = len(winners) / n

    log.info(f"\n  STRATEGY: 30-min LGB + Daily OFI Contrarian Filter")
    log.info(f"  Total Trades: {n}")
    log.info(f"  Winners: {len(winners)} ({wr:.1%})")
    log.info(f"  Losers:  {len(losers)} ({1-wr:.1%})")
    log.info(f"  Total PnL: {df['pnl_ticks'].sum():.1f} ticks (${df['pnl_ticks'].sum() * TICK_VALUE:,.0f})")

    # Key findings
    log.info(f"\n  KEY FINDINGS:")

    # Best time of day
    time_wr = df.groupby('time_slot').agg(n=('winner', 'count'), wr=('winner', 'mean')).reset_index()
    time_wr = time_wr[time_wr['n'] >= 5]  # minimum sample
    if len(time_wr) > 0:
        best_time = time_wr.loc[time_wr['wr'].idxmax()]
        worst_time = time_wr.loc[time_wr['wr'].idxmin()]
        log.info(f"    Best time slot:  {best_time['time_slot']} (WR={best_time['wr']:.1%}, N={best_time['n']:.0f})")
        log.info(f"    Worst time slot: {worst_time['time_slot']} (WR={worst_time['wr']:.1%}, N={worst_time['n']:.0f})")

    # Best day
    dow_wr = df.groupby('dow_name').agg(n=('winner', 'count'), wr=('winner', 'mean')).reset_index()
    dow_wr = dow_wr[dow_wr['n'] >= 5]
    if len(dow_wr) > 0:
        best_dow = dow_wr.loc[dow_wr['wr'].idxmax()]
        worst_dow = dow_wr.loc[dow_wr['wr'].idxmin()]
        log.info(f"    Best day:  {best_dow['dow_name']} (WR={best_dow['wr']:.1%}, N={best_dow['n']:.0f})")
        log.info(f"    Worst day: {worst_dow['dow_name']} (WR={worst_dow['wr']:.1%}, N={worst_dow['n']:.0f})")

    # Long vs short
    longs = df[df['direction'] == 1]
    shorts = df[df['direction'] == -1]
    log.info(f"    Long WR:  {longs['winner'].mean():.1%} (N={len(longs)})")
    log.info(f"    Short WR: {shorts['winner'].mean():.1%} (N={len(shorts)})")

    # SL analysis
    sl_trades = df[df['exit_type'] == 'sl']
    if len(sl_trades) > 0:
        log.info(f"    Median time to SL: {sl_trades['exit_minute'].median():.0f} minutes")
        near_miss_pct = (sl_trades['mfe_ticks'] >= 3).mean()
        log.info(f"    SL trades with MFE >= 3 ticks (near-misses): {near_miss_pct:.1%}")

    # MFE/MAE
    log.info(f"    Winner avg MFE: {winners['mfe_ticks'].mean():.1f} ticks")
    log.info(f"    Loser  avg MFE: {losers['mfe_ticks'].mean():.1f} ticks")
    log.info(f"    Loser  avg MAE: {losers['mae_ticks'].mean():.1f} ticks")

    # Actionable insights
    log.info(f"\n  ACTIONABLE INSIGHTS FOR BOOSTING WR:")
    insights = []

    # Check if high-confidence trades have better WR
    if 'zscore_quantile' in df.columns:
        top_q = df[df['zscore_quantile'] == 'Q5 (strongest)']
        bottom_q = df[df['zscore_quantile'] == 'Q1 (weakest)']
        if len(top_q) > 3 and len(bottom_q) > 3:
            top_wr = top_q['winner'].mean()
            bottom_wr = bottom_q['winner'].mean()
            if top_wr > bottom_wr + 0.05:
                insights.append(f"Tighter confidence filter: Top quintile WR={top_wr:.1%} vs Bottom={bottom_wr:.1%}. "
                               f"Raising threshold could boost WR by {top_wr - wr:.1%} but reduces N to {len(top_q)}.")
            else:
                insights.append(f"Confidence filter has little WR impact (Top={top_wr:.1%} vs Bottom={bottom_wr:.1%}). Edge is broad-based.")

    # Check aligned vs unaligned
    if 'aligned' in df.columns:
        aligned = df[df['aligned'] == True]
        unaligned = df[df['aligned'] == False]
        if len(aligned) > 5 and len(unaligned) > 5:
            if aligned['winner'].mean() > unaligned['winner'].mean() + 0.05:
                insights.append(f"Bias alignment works: Aligned WR={aligned['winner'].mean():.1%} vs Against={unaligned['winner'].mean():.1%}. "
                               f"Stricter alignment could improve WR.")

    # Check near-miss potential
    if len(sl_trades) > 0:
        near_miss_count = (sl_trades['mfe_ticks'] >= SL_SHORT).sum()
        if near_miss_count > 5:
            pct = near_miss_count / len(sl_trades)
            insights.append(f"Trailing stop opportunity: {near_miss_count} SL trades ({pct:.0%}) had MFE >= SL. "
                           f"A breakeven trail could recover these as scratch/small wins.")

    # Check time-of-day filter potential
    if len(time_wr) > 0:
        bad_times = time_wr[time_wr['wr'] < wr * 0.7]
        if len(bad_times) > 0:
            bad_n = df[df['time_slot'].isin(bad_times['time_slot'])].shape[0]
            insights.append(f"Time filter: {len(bad_times)} time slots with WR < {wr*0.7:.0%}. "
                           f"Blocking them removes {bad_n} trades and may boost WR.")

    for i, insight in enumerate(insights, 1):
        log.info(f"    {i}. {insight}")

    if not insights:
        log.info(f"    No clear single-factor confluence found. Edge appears broad-based.")

    log.info("\n" + "=" * 70)


def main():
    log.info("=" * 70)
    log.info("DEEP STRATEGY ANALYSIS v1 — Champion 30-min LGB + Daily OFI")
    log.info("=" * 70)
    t0 = time.time()

    # ── 1. Load data ──
    log.info("\n--- Loading data ---")
    daily_df = load_daily_features()
    entry_preds_30m, entry_dates_30m = load_30min_predictions()
    minute_df = load_minute_bars()

    # ── 2. Aggregate to 30-min bars ──
    log.info("\n--- Aggregating to 30-min bars ---")
    bars_30m = aggregate_to_30min_bars(minute_df)
    bars_30m = bars_30m.sort_values('ts').reset_index(drop=True)

    if len(bars_30m) != len(entry_preds_30m):
        log.warning(f"Bar count ({len(bars_30m)}) != prediction count ({len(entry_preds_30m)})")
        min_len = min(len(bars_30m), len(entry_preds_30m))
        bars_30m = bars_30m.iloc[:min_len].reset_index(drop=True)
        entry_preds_30m = entry_preds_30m[:min_len]

    # ── 3. Train daily contrarian model (walk-forward) ──
    log.info("\n--- Training daily contrarian model ---")
    daily_preds = walk_forward_daily_contrarian(daily_df)

    # Map to next trading day
    dates_list = sorted(daily_df['date'].dt.strftime('%Y%m%d').values)
    daily_bias = {}
    for i, d in enumerate(dates_list):
        if d in daily_preds and i + 1 < len(dates_list):
            next_d = dates_list[i + 1]
            daily_bias[next_d] = daily_preds[d]
    bias_std = np.std(list(daily_bias.values()))

    # ── 4. Reconstruct ALL trades (unfiltered) with enhanced metadata ──
    log.info("\n--- Reconstructing ALL trades (unfiltered, top 5%) ---")
    all_trades_raw, blocked_signals, fill_stats = reconstruct_entry_fills_enhanced(
        bars_30m, minute_df, entry_preds_30m,
        confidence_pct=ENTRY_THRESHOLD,
        cancel_window_min=CANCEL_WINDOW,
    )

    # Simulate exits for unfiltered trades
    log.info(f"\n--- Simulating FIFO exits (unfiltered) ---")
    unfiltered_results = simulate_fifo_exits_detailed(
        all_trades_raw, TP_LONG, TP_SHORT, SL_LONG, SL_SHORT, MAX_HOLD
    )
    unfiltered_df = pd.DataFrame(unfiltered_results)
    log.info(f"  Unfiltered: {len(unfiltered_df)} trades, WR={unfiltered_df['winner'].mean():.1%}")

    # ── 5. Apply daily bias filter (champion strategy logic) ──
    log.info("\n--- Applying daily bias filter (champion strategy) ---")
    bias_thresh = 0.5 * bias_std  # using 0.5x threshold as in the champion

    filtered_trades = []
    for trade in all_trades_raw:
        trade_date = trade['date']
        direction = trade['direction']

        if trade_date not in daily_bias:
            filtered_trades.append(trade)
            continue

        bias = daily_bias[trade_date]
        if abs(bias) < bias_thresh:
            filtered_trades.append(trade)
            continue

        # Strong bias: only aligned trades pass
        if bias > 0 and direction == -1:
            filtered_trades.append(trade)
        elif bias < 0 and direction == 1:
            filtered_trades.append(trade)
        # else: blocked

    log.info(f"  After filter: {len(filtered_trades)} / {len(all_trades_raw)} trades passed")

    # Simulate exits for filtered (champion) trades
    log.info(f"\n--- Simulating FIFO exits (filtered = champion) ---")
    filtered_results = simulate_fifo_exits_detailed(
        filtered_trades, TP_LONG, TP_SHORT, SL_LONG, SL_SHORT, MAX_HOLD
    )
    df = pd.DataFrame(filtered_results)
    log.info(f"  Champion trades: {len(df)}, WR={df['winner'].mean():.1%}")

    # ── 6. Run all analyses ──
    clustering_results = analyze_trade_clustering(df)
    entry_quality_results = analyze_entry_quality(df, daily_bias)
    loss_results = analyze_losses(df)
    opportunity_results = analyze_opportunity_cost(
        df, unfiltered_results, blocked_signals, daily_bias, minute_df, bars_30m, entry_preds_30m
    )

    # ── 7. Executive Summary ──
    generate_executive_summary(df, clustering_results, entry_quality_results,
                               loss_results, opportunity_results, daily_bias)

    # ── 8. Save all results ──
    elapsed = time.time() - t0
    log.info(f"\n=== COMPLETE ({elapsed:.0f}s) ===")

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Interval):
            return str(obj)
        return obj

    output = {
        'run_time': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'n_trades_champion': len(df),
        'n_trades_unfiltered': len(unfiltered_df),
        'champion_wr': float(df['winner'].mean()),
        'champion_total_pnl': float(df['pnl_ticks'].sum()),
        'clustering': make_serializable(clustering_results),
        'entry_quality': make_serializable(entry_quality_results),
        'loss_analysis': make_serializable(loss_results),
        'opportunity_cost': make_serializable(opportunity_results),
        'fill_stats': make_serializable(fill_stats),
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # Save per-trade details
    trade_details = df.drop(columns=['prices_close', 'prices_high', 'prices_low', 'times'], errors='ignore')
    trade_cols = [c for c in trade_details.columns if c not in ['prices_close', 'prices_high', 'prices_low', 'times']]
    trade_details_save = trade_details[trade_cols].copy()
    for col in trade_details_save.columns:
        if trade_details_save[col].dtype == 'object':
            trade_details_save[col] = trade_details_save[col].astype(str)
    trade_details_save.to_parquet(OUTPUT_DIR / 'trade_details.parquet', index=False)
    log.info(f"Trade details saved to {OUTPUT_DIR / 'trade_details.parquet'}")


if __name__ == '__main__':
    main()
