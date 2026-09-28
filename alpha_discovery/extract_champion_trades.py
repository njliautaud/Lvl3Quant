#!/usr/bin/env python3
"""Extract per-trade details from champion strategy (FILTERED_bias_1.50x).
Replicates multi_scale_combo_v1.py logic exactly, dumps CSV."""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT = Path("/home/nick/Lvl3Quant")
DATA_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v1" / "daily_features.parquet"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
OUTPUT_DIR = ROOT / "output" / "multi_scale_combo_v1"

TICK_SIZE = 1.0
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0

DAILY_TRAIN_DAYS = 40
DAILY_SLIDE_DAYS = 5

TP_LONG = 25; TP_SHORT = 25
SL_LONG = 4; SL_SHORT = 3
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
    return combined

def aggregate_to_30min_bars(minute_df):
    df = minute_df.copy()
    df['bar_key'] = df['ts_minute'].dt.floor('30min')
    df['return_1m'] = df.groupby('date')['close'].pct_change()
    records = []
    for (date_str, bar_key), grp in df.groupby(['date', 'bar_key']):
        if len(grp) < 2:
            continue
        close_arr = grp['close'].values
        vol_arr = grp['volume'].values
        ofi_arr = grp['ofi_1min'].values
        rec = {
            'date': date_str, 'bar_key': bar_key,
            'ts': grp['ts_minute'].iloc[0],
            'open': close_arr[0], 'high': close_arr.max(),
            'low': close_arr.min(), 'close': close_arr[-1],
            'total_volume': vol_arr.sum(), 'ofi_sum': ofi_arr.sum(),
        }
        records.append(rec)
    return pd.DataFrame(records)

def reconstruct_entry_fills(bars_30m, minute_df, pred_30m, confidence_pct, cancel_window_min):
    valid_mask = ~np.isnan(pred_30m)
    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - confidence_pct)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], confidence_pct)

    minute_lookup = {}
    for date_str, grp in minute_df.groupby('date'):
        minute_lookup[date_str] = grp.sort_values('ts_minute').reset_index(drop=True)

    trades = []
    n_signals = n_filled = n_cancelled = 0
    fill_delays = []

    bars_ts = bars_30m['ts'].values
    bars_dates = bars_30m['date'].values
    bars_close = bars_30m['close'].values

    for i in range(len(bars_30m)):
        if np.isnan(pred_30m[i]):
            continue
        direction = 0
        if pred_30m[i] >= upper_thresh: direction = 1
        elif pred_30m[i] <= lower_thresh: direction = -1
        else: continue

        n_signals += 1
        date_str = bars_dates[i]
        signal_ts = pd.Timestamp(bars_ts[i])
        signal_price = bars_close[i]
        if date_str not in minute_lookup: continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes['ts_minute'].values
        limit_price = signal_price
        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE)
        cancel_ts = signal_ts + pd.Timedelta(minutes=cancel_window_min + ENTRY_BAR_SIZE)

        fill_mask = (day_ts >= np.datetime64(signal_bar_end)) & (day_ts <= np.datetime64(cancel_ts))
        fill_candidates = day_minutes[fill_mask]
        if len(fill_candidates) == 0:
            n_cancelled += 1; continue

        filled = False
        fill_price = fill_ts = None
        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                if mbar['low'] <= limit_price - TICK_SIZE:
                    filled = True; fill_price = limit_price; fill_ts = mbar['ts_minute']
                    fill_delays.append(j+1); break
            else:
                if mbar['high'] >= limit_price + TICK_SIZE:
                    filled = True; fill_price = limit_price; fill_ts = mbar['ts_minute']
                    fill_delays.append(j+1); break

        if not filled:
            n_cancelled += 1; continue

        n_filled += 1
        remaining_mask = day_ts >= np.datetime64(fill_ts)
        remaining_minutes = day_minutes[remaining_mask]
        if len(remaining_minutes) < 2:
            n_filled -= 1; n_cancelled += 1; continue

        trade = {
            'idx': i, 'date': date_str,
            'signal_ts': signal_ts, 'fill_ts': pd.Timestamp(fill_ts),
            'fill_price': fill_price, 'fill_delay_minutes': fill_delays[-1],
            'direction': direction, 'pred_30m': float(pred_30m[i]),
            'prices_close': remaining_minutes['close'].values.copy(),
            'prices_high': remaining_minutes['high'].values.copy(),
            'prices_low': remaining_minutes['low'].values.copy(),
            'times': remaining_minutes['ts_minute'].values.copy(),
            'n_remaining_minutes': len(remaining_minutes),
        }
        trades.append(trade)

    return trades, {'n_signals': n_signals, 'n_filled': n_filled, 'n_cancelled': n_cancelled}

def simulate_fifo_exits_detailed(trades):
    results = []
    for trade in trades:
        direction = trade['direction']
        fill_price = trade['fill_price']
        prices_close = trade['prices_close']
        prices_high = trade['prices_high']
        prices_low = trade['prices_low']
        n_remaining = trade['n_remaining_minutes']

        tp_ticks = TP_LONG if direction == 1 else TP_SHORT
        sl_ticks = SL_LONG if direction == 1 else SL_SHORT
        tp_price = fill_price + direction * tp_ticks * TICK_SIZE
        sl_price = fill_price - direction * sl_ticks * TICK_SIZE

        max_check = min(MAX_HOLD, n_remaining)
        exit_type = 'time'; exit_minute = max_check; pnl_ticks = 0
        mfe = 0; mae = 0
        for m in range(1, max_check):
            if direction == 1:
                mfe = max(mfe, (prices_high[m] - fill_price) / TICK_SIZE)
                mae = max(mae, (fill_price - prices_low[m]) / TICK_SIZE)
            else:
                mfe = max(mfe, (fill_price - prices_low[m]) / TICK_SIZE)
                mae = max(mae, (prices_high[m] - fill_price) / TICK_SIZE)

        exit_found = False
        for m in range(1, max_check):
            bar_high = prices_high[m]; bar_low = prices_low[m]
            sl_hit = tp_hit = False
            if direction == 1:
                if bar_low <= sl_price: sl_hit = True
                if bar_high >= tp_price + TICK_SIZE: tp_hit = True
            else:
                if bar_high >= sl_price: sl_hit = True
                if bar_low <= tp_price - TICK_SIZE: tp_hit = True
            if sl_hit and tp_hit: sl_hit = True; tp_hit = False
            if sl_hit:
                pnl_ticks = -(sl_ticks + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                exit_type = 'sl'; exit_minute = m; exit_found = True; break
            if tp_hit:
                pnl_ticks = tp_ticks - RT_COMMISSION_TICKS
                exit_type = 'tp'; exit_minute = m; exit_found = True; break

        if not exit_found:
            exit_minute = min(MAX_HOLD, n_remaining - 1)
            exit_minute = max(exit_minute, 1)
            exit_close = prices_close[exit_minute]
            if direction == 1: exit_fill = exit_close - TICK_SIZE
            else: exit_fill = exit_close + TICK_SIZE
            raw_pnl = (exit_fill - fill_price) / TICK_SIZE * direction
            pnl_ticks = raw_pnl - RT_COMMISSION_TICKS

        results.append({
            'date': trade['date'],
            'signal_ts': str(trade['signal_ts']),
            'fill_ts': str(trade['fill_ts']),
            'fill_price': trade['fill_price'],
            'direction': trade['direction'],
            'pred_30m': trade['pred_30m'],
            'pnl_ticks': round(pnl_ticks, 4),
            'exit_type': exit_type,
            'exit_minute': exit_minute,
            'mfe_ticks': round(mfe, 2),
            'mae_ticks': round(mae, 2),
        })
    return results

def walk_forward_daily_contrarian(daily_df):
    n = len(daily_df)
    predictions = {}
    start_idx = 0
    while start_idx + DAILY_TRAIN_DAYS < n:
        train_end = start_idx + DAILY_TRAIN_DAYS
        oot_end = min(train_end + DAILY_SLIDE_DAYS, n)
        train_df = daily_df.iloc[start_idx:train_end]
        oot_df = daily_df.iloc[train_end:oot_end]
        if len(oot_df) == 0: break

        label = 'fwd_return_1d'
        train_mask = train_df[label].notna() & train_df[DAILY_FEATURE_COLS].notna().all(axis=1)
        oot_mask = oot_df[label].notna() & oot_df[DAILY_FEATURE_COLS].notna().all(axis=1)
        X_train = train_df.loc[train_mask, DAILY_FEATURE_COLS].values
        y_train = -train_df.loc[train_mask, label].values  # CONTRARIAN
        X_oot = oot_df.loc[oot_mask, DAILY_FEATURE_COLS].values

        if len(X_train) < 15 or len(X_oot) < 1:
            start_idx += DAILY_SLIDE_DAYS; continue

        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=200)
        preds = model.predict(X_oot)

        for i in range(len(preds)):
            oot_idx = oot_df.index[oot_mask][i]
            row = daily_df.loc[oot_idx]
            date_str = row['date'].strftime('%Y%m%d')
            predictions[date_str] = float(preds[i])

        start_idx += DAILY_SLIDE_DAYS

    print(f"Daily contrarian WF: {len(predictions)} date predictions")
    return predictions


def main():
    t0 = time.time()
    print("Loading data...")
    minute_df = load_minute_bars()
    bars_30m = aggregate_to_30min_bars(minute_df)
    print(f"  {len(minute_df):,} minute bars -> {len(bars_30m):,} 30m bars")

    daily_df = pd.read_parquet(FEATURES_PATH)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    print(f"  {len(daily_df)} daily features")

    pred_data = np.load(str(ENTRY_PREDS_PATH), allow_pickle=True)
    entry_preds_30m = pred_data['entry_preds']
    print(f"  {len(entry_preds_30m)} entry predictions ({np.sum(~np.isnan(entry_preds_30m))} non-NaN)")

    # Reconstruct fills
    all_trades, fill_stats = reconstruct_entry_fills(
        bars_30m, minute_df, entry_preds_30m,
        confidence_pct=ENTRY_THRESHOLD, cancel_window_min=CANCEL_WINDOW)
    print(f"Filled trades: {len(all_trades)} (signals={fill_stats['n_signals']}, cancelled={fill_stats['n_cancelled']})")

    # Train daily model
    print("Training daily contrarian model (walk-forward)...")
    daily_preds = walk_forward_daily_contrarian(daily_df)

    dates_list = sorted(daily_df['date'].dt.strftime('%Y%m%d').values)
    daily_bias = {}
    for i, d in enumerate(dates_list):
        if d in daily_preds and i + 1 < len(dates_list):
            daily_bias[dates_list[i + 1]] = daily_preds[d]

    bias_values = np.array(list(daily_bias.values()))
    bias_std = bias_values.std()
    bias_thresh = 1.5 * bias_std
    print(f"Bias threshold (1.5x): {bias_thresh:.2f} (std={bias_std:.2f})")

    # Filter trades (FILTERED_bias_1.50x)
    filtered_trades = []
    n_blocked = 0
    for trade in all_trades:
        td = trade['date']; d = trade['direction']
        if td not in daily_bias:
            filtered_trades.append(trade); continue
        bias = daily_bias[td]
        if abs(bias) < bias_thresh:
            filtered_trades.append(trade); continue
        if (bias > 0 and d == -1) or (bias < 0 and d == 1):
            filtered_trades.append(trade)
        else:
            n_blocked += 1

    print(f"Champion filtered: {len(filtered_trades)} trades (blocked {n_blocked})")

    # Simulate
    trade_details = simulate_fifo_exits_detailed(filtered_trades)
    df = pd.DataFrame(trade_details)

    # Regime info
    regime_map = {}
    for _, row in daily_df.iterrows():
        d_str = row['date'].strftime('%Y%m%d')
        cc = row.get('cc_return_ticks', 0)
        if cc > 4: regime_map[d_str] = 'green'
        elif cc < -4: regime_map[d_str] = 'red'
        else: regime_map[d_str] = 'flat'
    df['regime'] = df['date'].map(regime_map).fillna('flat')

    # Time features (fill_ts is UTC string with tz)
    df['fill_ts_parsed'] = pd.to_datetime(df['fill_ts'])
    df['fill_hour_et'] = df['fill_ts_parsed'].dt.tz_convert('America/New_York').dt.hour
    df['fill_minute_et'] = df['fill_ts_parsed'].dt.tz_convert('America/New_York').dt.minute
    df['day_of_week'] = df['fill_ts_parsed'].dt.tz_convert('America/New_York').dt.dayofweek
    df['dow_name'] = df['fill_ts_parsed'].dt.tz_convert('America/New_York').dt.day_name()

    outpath = OUTPUT_DIR / 'champion_trade_details.csv'
    df.drop(columns=['fill_ts_parsed']).to_csv(outpath, index=False)

    elapsed = time.time() - t0
    print(f"\nSaved {len(df)} trades to {outpath} ({elapsed:.1f}s)")
    print(f"PnL={df['pnl_ticks'].sum():.1f}t, WR={len(df[df['pnl_ticks']>0])/len(df):.3f}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}, {df['date'].nunique()} unique dates")
    print(f"\nDay-of-week:\n{df['dow_name'].value_counts().to_string()}")
    print(f"\nHour (ET):\n{df['fill_hour_et'].value_counts().sort_index().to_string()}")
    print(f"\nRegime:\n{df['regime'].value_counts().to_string()}")

if __name__ == '__main__':
    main()
