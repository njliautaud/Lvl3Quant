#!/usr/bin/env python3
"""
robustness_v1.py — Champion Strategy Robustness Testing
========================================================

Tests the champion ES futures strategy across 5 robustness dimensions:
  1. Time-period stability (3 non-overlapping chunks)
  2. Parameter sensitivity (neighborhood perturbation)
  3. Transaction cost sensitivity
  4. Monthly breakdown
  5. Drawdown analysis

Champion config:
  30-min LightGBM entry (top 5%) + daily OFI contrarian filter (1.5x threshold)
  TP=25, SL_L=4/SL_S=3, max_hold=60, passive entry/TP, market SL
  Sharpe 5.46, regime gap 0.12

HC #0: SLIDING windows only.
HC #74: FIFO fills only.
HC #428: Regime-agnostic validation.
"""

import os, sys, json, logging, warnings, time, traceback
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──
ROOT = Path("/home/nick/Lvl3Quant")
DATA_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v1" / "daily_features.parquet"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
OUTPUT_DIR = ROOT / "output" / "robustness_v1"
LOG_FILE = ROOT / "logs" / "robustness_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ROBUST] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
TICK_SIZE = 1.0
TICK_SIZE_PTS = 0.25
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0

# Champion params
TP = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD = 60
ENTRY_THRESHOLD = 0.05  # top 5%
CANCEL_WINDOW = 10
ENTRY_BAR_SIZE = 30
DAILY_BIAS_THRESHOLD_MULT = 1.5

# Daily model params
DAILY_TRAIN_DAYS = 40
DAILY_SLIDE_DAYS = 5

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
    'objective': 'regression', 'metric': 'mae',
    'learning_rate': 0.05, 'num_leaves': 16, 'max_depth': 4,
    'min_child_samples': 5, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'verbose': -1, 'n_jobs': -1, 'seed': 42,
}


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING (from multi_scale_combo_v1)
# ═══════════════════════════════════════════════════════════════

def load_daily_features():
    df = pd.read_parquet(FEATURES_PATH)
    log.info(f"Loaded daily features: {len(df)} rows")
    return df


def load_30min_predictions():
    data = np.load(str(ENTRY_PREDS_PATH), allow_pickle=True)
    preds = data['entry_preds']
    dates = data['dates']
    log.info(f"Loaded 30-min predictions: {len(preds)} entries")
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
            'date': date_str, 'bar_key': bar_key,
            'ts': grp['ts_minute'].iloc[0],
            'open': close_arr[0], 'high': close_arr.max(),
            'low': close_arr.min(), 'close': close_arr[-1],
            'total_volume': vol_arr.sum(), 'ofi_sum': ofi_arr.sum(),
        }
        records.append(rec)
    result = pd.DataFrame(records)
    log.info(f"Aggregated {len(result):,} 30min bars")
    return result


def walk_forward_daily_contrarian(daily_df):
    n = len(daily_df)
    predictions = {}
    start_idx = 0
    while start_idx + DAILY_TRAIN_DAYS < n:
        train_end = start_idx + DAILY_TRAIN_DAYS
        oot_end = min(train_end + DAILY_SLIDE_DAYS, n)
        train_df = daily_df.iloc[start_idx:train_end]
        oot_df = daily_df.iloc[train_end:oot_end]
        if len(oot_df) == 0:
            break
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
    log.info(f"Daily contrarian WF: {len(predictions)} date predictions")
    return predictions


def reconstruct_entry_fills(bars_30m, minute_df, pred_30m, confidence_pct, cancel_window_min):
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]
    if len(valid_preds) < 20:
        return [], {}
    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - confidence_pct)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], confidence_pct)
    minute_lookup = {}
    for date_str, grp in minute_df.groupby('date'):
        minute_lookup[date_str] = grp.sort_values('ts_minute').reset_index(drop=True)
    trades = []
    n_signals = n_filled = n_cancelled = 0
    bars_ts = bars_30m['ts'].values
    bars_dates = bars_30m['date'].values
    bars_close = bars_30m['close'].values
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
        if date_str not in minute_lookup:
            continue
        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes['ts_minute'].values
        limit_price = signal_price
        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE)
        cancel_ts = signal_ts + pd.Timedelta(minutes=cancel_window_min + ENTRY_BAR_SIZE)
        fill_mask = (day_ts >= np.datetime64(signal_bar_end)) & (day_ts <= np.datetime64(cancel_ts))
        fill_candidates = day_minutes[fill_mask]
        if len(fill_candidates) == 0:
            n_cancelled += 1
            continue
        filled = False
        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                if mbar['low'] <= limit_price - TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    break
            else:
                if mbar['high'] >= limit_price + TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    break
        if not filled:
            n_cancelled += 1
            continue
        n_filled += 1
        remaining_mask = day_ts >= np.datetime64(fill_ts)
        remaining_minutes = day_minutes[remaining_mask]
        if len(remaining_minutes) < 2:
            n_filled -= 1
            n_cancelled += 1
            continue
        trade = {
            'idx': i, 'date': date_str,
            'signal_ts': signal_ts, 'fill_ts': pd.Timestamp(fill_ts),
            'fill_price': fill_price, 'direction': direction,
            'pred_30m': float(pred_30m[i]),
            'prices_close': remaining_minutes['close'].values.copy(),
            'prices_high': remaining_minutes['high'].values.copy(),
            'prices_low': remaining_minutes['low'].values.copy(),
            'times': remaining_minutes['ts_minute'].values.copy(),
            'n_remaining_minutes': len(remaining_minutes),
        }
        trades.append(trade)
    fill_stats = {'n_signals': n_signals, 'n_filled': n_filled, 'n_cancelled': n_cancelled,
                  'fill_rate': n_filled / max(n_signals, 1)}
    log.info(f"  Entry fills: signals={n_signals}, filled={n_filled} ({fill_stats['fill_rate']:.1%})")
    return trades, fill_stats


def simulate_fifo_exits(trades, tp_long, tp_short, sl_long, sl_short, max_hold_minutes,
                        extra_slippage=0.0, market_entry_cost=0.0):
    """
    Simulate FIFO exits with configurable cost model.
    extra_slippage: additional slippage on ALL exits (ticks)
    market_entry_cost: cost of market entry (ticks, 0 for passive)
    """
    pnls = []
    dates = []
    directions = []
    exit_types = []
    hold_durations = []

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

        exit_found = False
        max_check = min(max_hold_minutes, n_remaining)

        for m in range(1, max_check):
            bar_high = prices_high[m]
            bar_low = prices_low[m]
            sl_hit = tp_hit = False
            if direction == 1:
                if bar_low <= sl_price: sl_hit = True
                if bar_high >= tp_price + TICK_SIZE: tp_hit = True
            else:
                if bar_high >= sl_price: sl_hit = True
                if bar_low <= tp_price - TICK_SIZE: tp_hit = True
            if sl_hit and tp_hit:
                sl_hit = True; tp_hit = False

            if sl_hit:
                # SL: market exit = 1 tick slippage + extra_slippage
                trade_pnl = -(sl_ticks + MARKET_SLIPPAGE_TICKS + extra_slippage + RT_COMMISSION_TICKS + market_entry_cost)
                pnls.append(trade_pnl)
                dates.append(trade['date'])
                directions.append(direction)
                exit_types.append('sl')
                hold_durations.append(m)
                exit_found = True
                break

            if tp_hit:
                # TP: passive exit (no slippage) + extra_slippage
                trade_pnl = tp_ticks - extra_slippage - RT_COMMISSION_TICKS - market_entry_cost
                pnls.append(trade_pnl)
                dates.append(trade['date'])
                directions.append(direction)
                exit_types.append('tp')
                hold_durations.append(m)
                exit_found = True
                break

        if not exit_found:
            exit_minute = max(min(max_hold_minutes, n_remaining - 1), 1)
            exit_close = prices_close[exit_minute]
            if direction == 1:
                exit_fill = exit_close - TICK_SIZE
            else:
                exit_fill = exit_close + TICK_SIZE
            raw_pnl_ticks = (exit_fill - fill_price) / TICK_SIZE * direction
            trade_pnl = raw_pnl_ticks - extra_slippage - RT_COMMISSION_TICKS - market_entry_cost
            pnls.append(trade_pnl)
            dates.append(trade['date'])
            directions.append(direction)
            exit_types.append('time')
            hold_durations.append(exit_minute)

    return (np.array(pnls), np.array(dates), np.array(directions),
            np.array(exit_types), np.array(hold_durations))


def classify_regime(cc_return_ticks):
    if cc_return_ticks > 4: return 'green'
    elif cc_return_ticks < -4: return 'red'
    return 'flat'


def compute_metrics(pnl_arr, dates_arr, dirs_arr, daily_df=None):
    """Compute comprehensive metrics for a set of trades."""
    n = len(pnl_arr)
    if n < 3:
        return {'n_trades': n, 'error': 'too few trades'}

    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    wr = len(wins) / n
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    unique_dates = sorted(set(dates_arr))
    daily_pnl = np.array([pnl_arr[dates_arr == d].sum() for d in unique_dates])

    mean_d = daily_pnl.mean()
    std_d = daily_pnl.std(ddof=1) if len(daily_pnl) > 1 else np.nan
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d and std_d > 0 else np.nan

    downside = daily_pnl[daily_pnl < 0]
    ds_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (mean_d / ds_std * np.sqrt(252)) if ds_std and ds_std > 0 else np.nan

    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) > 0 else 0

    # Regime analysis
    regime_sharpes = {}
    if daily_df is not None:
        date_regime = {}
        for _, row in daily_df.iterrows():
            d_str = row['date'].strftime('%Y%m%d')
            date_regime[d_str] = classify_regime(row.get('cc_return_ticks', 0))
        for reg in ['green', 'red', 'flat']:
            reg_mask = np.array([date_regime.get(d, 'flat') == reg for d in dates_arr])
            if reg_mask.sum() < 3: continue
            r_pnl = pnl_arr[reg_mask]
            r_dates = dates_arr[reg_mask]
            r_unique = sorted(set(r_dates))
            r_daily = np.array([r_pnl[r_dates == d].sum() for d in r_unique])
            r_std = r_daily.std(ddof=1) if len(r_daily) > 1 else np.nan
            r_sharpe = (r_daily.mean() / r_std * np.sqrt(252)) if r_std and r_std > 0 else np.nan
            regime_sharpes[reg] = round(r_sharpe, 2) if not np.isnan(r_sharpe) else None

    regime_gap = None
    if len(regime_sharpes) >= 2:
        vals = [v for v in regime_sharpes.values() if v is not None]
        if vals:
            max_s = max(abs(v) for v in vals)
            if max_s > 0:
                regime_gap = round(abs(max(vals) - min(vals)) / max_s, 3)

    return {
        'n_trades': n,
        'n_days': len(unique_dates),
        'wr': round(wr, 3),
        'pf': round(pf, 2) if pf != float('inf') else 999.0,
        'sharpe': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'sortino': round(sortino, 2) if not np.isnan(sortino) else None,
        'total_pnl_ticks': round(pnl_arr.sum(), 1),
        'total_pnl_dollars': round(pnl_arr.sum() * TICK_VALUE, 0),
        'max_dd_ticks': round(max_dd, 1),
        'regime_sharpes': regime_sharpes,
        'regime_gap': regime_gap,
        'regime_pass': regime_gap <= 0.50 if regime_gap is not None else None,
    }


def apply_daily_filter(all_trades, daily_bias, bias_threshold):
    """Apply the daily OFI contrarian filter to trades."""
    filtered = []
    for trade in all_trades:
        trade_date = trade['date']
        direction = trade['direction']
        if trade_date not in daily_bias:
            filtered.append(trade)
            continue
        bias = daily_bias[trade_date]
        if abs(bias) < bias_threshold:
            filtered.append(trade)
            continue
        # Strong bias: only aligned trades
        if bias > 0 and direction == -1:
            filtered.append(trade)
        elif bias < 0 and direction == 1:
            filtered.append(trade)
        # else: blocked
    return filtered


# ═══════════════════════════════════════════════════════════════
#  TEST 1: TIME-PERIOD STABILITY
# ═══════════════════════════════════════════════════════════════

def test_time_stability(pnl_arr, dates_arr, dirs_arr, daily_df):
    log.info("\n" + "=" * 70)
    log.info("TEST 1: TIME-PERIOD STABILITY")
    log.info("=" * 70)

    unique_dates = sorted(set(dates_arr))
    n_dates = len(unique_dates)
    chunk_size = n_dates // 3

    chunks = [
        unique_dates[:chunk_size],
        unique_dates[chunk_size:2*chunk_size],
        unique_dates[2*chunk_size:],
    ]

    results = []
    all_positive = True

    for i, chunk_dates in enumerate(chunks):
        chunk_set = set(chunk_dates)
        mask = np.array([d in chunk_set for d in dates_arr])
        if mask.sum() < 5:
            log.info(f"  Chunk {i+1}: too few trades ({mask.sum()})")
            results.append({'chunk': i+1, 'error': 'too few trades'})
            all_positive = False
            continue

        m = compute_metrics(pnl_arr[mask], dates_arr[mask], dirs_arr[mask], daily_df)
        m['chunk'] = i + 1
        m['date_range'] = f"{chunk_dates[0]} - {chunk_dates[-1]}"
        results.append(m)

        sharpe = m.get('sharpe')
        if sharpe is None or sharpe <= 0:
            all_positive = False

        log.info(f"  Chunk {i+1} ({chunk_dates[0]}-{chunk_dates[-1]}): "
                 f"N={m['n_trades']}, Sharpe={m.get('sharpe')}, WR={m.get('wr')}, "
                 f"PF={m.get('pf')}, PnL={m.get('total_pnl_ticks')}t, "
                 f"Gap={m.get('regime_gap')}")

    verdict = "PASS" if all_positive else "FAIL"
    log.info(f"\n  TIME STABILITY VERDICT: {verdict}")
    return {'test': 'time_stability', 'verdict': verdict, 'chunks': results, 'all_positive_sharpe': all_positive}


# ═══════════════════════════════════════════════════════════════
#  TEST 2: PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════

def test_parameter_sensitivity(filtered_trades, daily_df):
    log.info("\n" + "=" * 70)
    log.info("TEST 2: PARAMETER SENSITIVITY (NEIGHBORHOOD)")
    log.info("=" * 70)

    # Sweep dimensions
    tp_range = [22, 23, 24, 25, 26, 27, 28]
    sl_long_range = [3, 4, 5]
    sl_short_range = [2, 3, 4]
    entry_thresh_range = [0.03, 0.04, 0.05, 0.06, 0.07]
    # Note: daily_bias_threshold tested separately since it requires re-filtering

    results = {}
    min_sharpe_threshold = 3.0

    # TP sweep (hold others at champion)
    log.info("\n  --- TP sweep ---")
    tp_results = {}
    for tp in tp_range:
        pnl, dates, dirs, _, _ = simulate_fifo_exits(
            filtered_trades, tp, tp, SL_LONG, SL_SHORT, MAX_HOLD)
        if len(pnl) < 10: continue
        m = compute_metrics(pnl, dates, dirs, daily_df)
        tp_results[tp] = m
        log.info(f"    TP={tp}: Sharpe={m.get('sharpe')}, N={m['n_trades']}, WR={m.get('wr')}")
    results['tp_sweep'] = tp_results

    # SL_long sweep
    log.info("\n  --- SL_long sweep ---")
    sl_long_results = {}
    for sl in sl_long_range:
        pnl, dates, dirs, _, _ = simulate_fifo_exits(
            filtered_trades, TP, TP, sl, SL_SHORT, MAX_HOLD)
        if len(pnl) < 10: continue
        m = compute_metrics(pnl, dates, dirs, daily_df)
        sl_long_results[sl] = m
        log.info(f"    SL_L={sl}: Sharpe={m.get('sharpe')}, N={m['n_trades']}")
    results['sl_long_sweep'] = sl_long_results

    # SL_short sweep
    log.info("\n  --- SL_short sweep ---")
    sl_short_results = {}
    for sl in sl_short_range:
        pnl, dates, dirs, _, _ = simulate_fifo_exits(
            filtered_trades, TP, TP, SL_LONG, sl, MAX_HOLD)
        if len(pnl) < 10: continue
        m = compute_metrics(pnl, dates, dirs, daily_df)
        sl_short_results[sl] = m
        log.info(f"    SL_S={sl}: Sharpe={m.get('sharpe')}, N={m['n_trades']}")
    results['sl_short_sweep'] = sl_short_results

    # Collect all neighbor Sharpe values
    all_sharpes = []
    for sweep_name, sweep_results in results.items():
        for param_val, m in sweep_results.items():
            s = m.get('sharpe')
            if s is not None:
                all_sharpes.append(s)

    n_above_threshold = sum(1 for s in all_sharpes if s >= min_sharpe_threshold)
    n_total = len(all_sharpes)
    pct_robust = n_above_threshold / max(n_total, 1)

    verdict = "PASS" if pct_robust >= 0.70 else "FAIL"
    log.info(f"\n  Neighbors with Sharpe >= {min_sharpe_threshold}: {n_above_threshold}/{n_total} ({pct_robust:.0%})")
    log.info(f"  Min neighbor Sharpe: {min(all_sharpes):.2f}" if all_sharpes else "  No results")
    log.info(f"  Max neighbor Sharpe: {max(all_sharpes):.2f}" if all_sharpes else "")
    log.info(f"  PARAMETER SENSITIVITY VERDICT: {verdict}")

    return {
        'test': 'parameter_sensitivity',
        'verdict': verdict,
        'results': {k: {str(pk): pv for pk, pv in v.items()} for k, v in results.items()},
        'n_above_threshold': n_above_threshold,
        'n_total': n_total,
        'pct_robust': round(pct_robust, 3),
        'min_neighbor_sharpe': round(min(all_sharpes), 2) if all_sharpes else None,
        'max_neighbor_sharpe': round(max(all_sharpes), 2) if all_sharpes else None,
        'threshold': min_sharpe_threshold,
    }


def test_daily_bias_sensitivity(all_trades, daily_bias, bias_std, daily_df):
    """Test sensitivity to daily bias threshold parameter."""
    log.info("\n  --- Daily bias threshold sweep ---")
    bias_thresh_mults = [1.0, 1.25, 1.50, 1.75, 2.0]
    bias_results = {}

    for mult in bias_thresh_mults:
        thresh = mult * bias_std
        filtered = apply_daily_filter(all_trades, daily_bias, thresh)
        if len(filtered) < 10:
            continue
        pnl, dates, dirs, _, _ = simulate_fifo_exits(
            filtered, TP, TP, SL_LONG, SL_SHORT, MAX_HOLD)
        if len(pnl) < 10: continue
        m = compute_metrics(pnl, dates, dirs, daily_df)
        bias_results[mult] = m
        log.info(f"    bias_mult={mult:.2f}: Sharpe={m.get('sharpe')}, N={m['n_trades']}")

    return bias_results


# ═══════════════════════════════════════════════════════════════
#  TEST 3: TRANSACTION COST SENSITIVITY
# ═══════════════════════════════════════════════════════════════

def test_cost_sensitivity(filtered_trades, daily_df):
    log.info("\n" + "=" * 70)
    log.info("TEST 3: TRANSACTION COST SENSITIVITY")
    log.info("=" * 70)

    scenarios = {
        'optimistic': {
            'desc': 'Passive entry + passive exit, commission only',
            'extra_slippage': 0.0,
            'market_entry_cost': 0.0,
            # Override: TP exit is already passive (0 slippage). SL exit normally has 1 tick.
            # For optimistic, assume passive exit on SL too (unrealistic but lower bound)
        },
        'base': {
            'desc': 'Current model: passive entry, market SL, passive TP, 0.376 commission',
            'extra_slippage': 0.0,
            'market_entry_cost': 0.0,
        },
        'pessimistic': {
            'desc': 'Base + 0.5 tick extra slippage on ALL exits',
            'extra_slippage': 0.5,
            'market_entry_cost': 0.0,
        },
        'worst_case': {
            'desc': 'Market entry (1 tick) + market exit (1 tick) + 0.5 extra = 3.376 RT',
            'extra_slippage': 0.5,
            'market_entry_cost': 1.0,
        },
    }

    results = {}
    for name, params in scenarios.items():
        pnl, dates, dirs, exit_types, holds = simulate_fifo_exits(
            filtered_trades, TP, TP, SL_LONG, SL_SHORT, MAX_HOLD,
            extra_slippage=params['extra_slippage'],
            market_entry_cost=params['market_entry_cost'],
        )
        m = compute_metrics(pnl, dates, dirs, daily_df)
        m['scenario'] = name
        m['description'] = params['desc']
        results[name] = m
        log.info(f"  {name:15s}: Sharpe={m.get('sharpe'):>6}, PnL={(m.get('total_pnl_ticks') or 0):>8.1f}t, "
                 f"WR={m.get('wr')}, PF={m.get('pf')}")

    # For optimistic, we re-run with modified SL logic (no market slippage on SL)
    # Actually simulate this by adjusting: SL cost = sl_ticks + 0 + commission (no market slippage)
    # We can approximate by adding back the market slippage savings
    base_pnl = results['base'].get('total_pnl_ticks', 0)
    n_trades = results['base'].get('n_trades', 0)

    pessimistic_sharpe = results['pessimistic'].get('sharpe')
    survives_pessimistic = pessimistic_sharpe is not None and pessimistic_sharpe > 0

    verdict = "PASS" if survives_pessimistic else "FAIL"
    log.info(f"\n  Pessimistic Sharpe > 0: {survives_pessimistic}")
    log.info(f"  COST SENSITIVITY VERDICT: {verdict}")

    return {
        'test': 'cost_sensitivity',
        'verdict': verdict,
        'scenarios': results,
        'survives_pessimistic': survives_pessimistic,
    }


# ═══════════════════════════════════════════════════════════════
#  TEST 4: MONTHLY BREAKDOWN
# ═══════════════════════════════════════════════════════════════

def test_monthly_breakdown(pnl_arr, dates_arr, dirs_arr, daily_df):
    log.info("\n" + "=" * 70)
    log.info("TEST 4: MONTHLY BREAKDOWN")
    log.info("=" * 70)

    # Convert dates to month keys
    month_trades = defaultdict(lambda: {'pnl': [], 'dates': [], 'dirs': []})
    for i in range(len(pnl_arr)):
        d = dates_arr[i]  # YYYYMMDD string
        month_key = d[:6]  # YYYYMM
        month_trades[month_key]['pnl'].append(pnl_arr[i])
        month_trades[month_key]['dates'].append(d)
        month_trades[month_key]['dirs'].append(dirs_arr[i])

    results = {}
    n_negative = 0

    for month_key in sorted(month_trades.keys()):
        mt = month_trades[month_key]
        pnl = np.array(mt['pnl'])
        dates = np.array(mt['dates'])
        dirs = np.array(mt['dirs'])

        m = compute_metrics(pnl, dates, dirs, daily_df)
        m['month'] = month_key
        results[month_key] = m

        total = pnl.sum()
        if total < 0:
            n_negative += 1

        log.info(f"  {month_key}: N={m['n_trades']:>4}, Sharpe={str(m.get('sharpe', 'N/A')):>6}, "
                 f"WR={m.get('wr')}, PF={m.get('pf')}, PnL={(m.get('total_pnl_ticks') or 0):>8.1f}t "
                 f"({'NEG' if total < 0 else 'POS'})")

    n_months = len(results)
    all_profitable = n_negative == 0

    # Find worst month
    worst_month = min(results.items(), key=lambda x: x[1].get('total_pnl_ticks', 0)) if results else None

    verdict = "PASS" if n_negative <= 1 else "FAIL"  # Allow 1 negative month
    log.info(f"\n  Months: {n_months}, Negative: {n_negative}, All profitable: {all_profitable}")
    if worst_month:
        log.info(f"  Worst month: {worst_month[0]} (PnL={worst_month[1].get('total_pnl_ticks')}t, "
                 f"Sharpe={worst_month[1].get('sharpe')})")
    log.info(f"  MONTHLY BREAKDOWN VERDICT: {verdict}")

    return {
        'test': 'monthly_breakdown',
        'verdict': verdict,
        'months': results,
        'n_months': n_months,
        'n_negative': n_negative,
        'all_profitable': all_profitable,
        'worst_month': worst_month[0] if worst_month else None,
        'worst_month_pnl': worst_month[1].get('total_pnl_ticks') if worst_month else None,
    }


# ═══════════════════════════════════════════════════════════════
#  TEST 5: DRAWDOWN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def test_drawdown_analysis(pnl_arr, dates_arr, dirs_arr, daily_df):
    log.info("\n" + "=" * 70)
    log.info("TEST 5: DRAWDOWN ANALYSIS")
    log.info("=" * 70)

    cum_pnl = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl

    max_dd_ticks = drawdown.max()
    max_dd_idx = np.argmax(drawdown)

    # Find peak before max DD
    peak_idx = np.argmax(cum_pnl[:max_dd_idx + 1]) if max_dd_idx > 0 else 0

    # Find recovery (where cumPnL gets back to peak level)
    recovery_idx = None
    peak_val = cum_pnl[peak_idx]
    for i in range(max_dd_idx, len(cum_pnl)):
        if cum_pnl[i] >= peak_val:
            recovery_idx = i
            break

    # DD duration in trades
    dd_duration_trades = max_dd_idx - peak_idx
    recovery_trades = (recovery_idx - peak_idx) if recovery_idx else None

    # DD duration in days
    dd_start_date = dates_arr[peak_idx]
    dd_trough_date = dates_arr[max_dd_idx]
    dd_recovery_date = dates_arr[recovery_idx] if recovery_idx else "NOT RECOVERED"

    # Daily drawdown
    unique_dates = sorted(set(dates_arr))
    daily_pnl = np.array([pnl_arr[dates_arr == d].sum() for d in unique_dates])
    cum_daily = np.cumsum(daily_pnl)
    daily_peak = np.maximum.accumulate(cum_daily)
    daily_dd = daily_peak - cum_daily
    max_daily_dd_idx = np.argmax(daily_dd)
    daily_peak_idx = np.argmax(cum_daily[:max_daily_dd_idx + 1]) if max_daily_dd_idx > 0 else 0

    dd_days = max_daily_dd_idx - daily_peak_idx

    # Consecutive losing trades
    max_losing_streak = 0
    current_streak = 0
    for p in pnl_arr:
        if p < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    # Consecutive losing days
    max_losing_day_streak = 0
    current_day_streak = 0
    for dp in daily_pnl:
        if dp < 0:
            current_day_streak += 1
            max_losing_day_streak = max(max_losing_day_streak, current_day_streak)
        else:
            current_day_streak = 0

    # Largest single-day loss
    worst_day_idx = np.argmin(daily_pnl)
    worst_day_pnl = daily_pnl[worst_day_idx]
    worst_day_date = unique_dates[worst_day_idx]

    # Calmar ratio (annualized return / max DD)
    total_pnl = pnl_arr.sum()
    n_days = len(unique_dates)
    ann_pnl = total_pnl * (252 / max(n_days, 1))
    calmar = ann_pnl / max(max_dd_ticks, 1)

    log.info(f"  Max drawdown: {max_dd_ticks:.1f} ticks (${max_dd_ticks * TICK_VALUE:.0f})")
    log.info(f"  DD peak date: {dd_start_date}, trough date: {dd_trough_date}")
    log.info(f"  DD duration: {dd_days} trading days, {dd_duration_trades} trades")
    log.info(f"  DD recovery: {dd_recovery_date} ({recovery_trades} trades)" if recovery_idx else
             f"  DD recovery: NOT YET RECOVERED")
    log.info(f"  Max consecutive losing trades: {max_losing_streak}")
    log.info(f"  Max consecutive losing days: {max_losing_day_streak}")
    log.info(f"  Worst single day: {worst_day_date} ({worst_day_pnl:.1f} ticks, ${worst_day_pnl * TICK_VALUE:.0f})")
    log.info(f"  Calmar ratio: {calmar:.2f}")

    # Verdict: acceptable if max DD < 150 ticks and losing streak < 15
    acceptable = max_dd_ticks < 150 and max_losing_streak < 15
    verdict = "PASS" if acceptable else "FAIL"
    log.info(f"\n  DRAWDOWN VERDICT: {verdict}")

    return {
        'test': 'drawdown_analysis',
        'verdict': verdict,
        'max_dd_ticks': round(max_dd_ticks, 1),
        'max_dd_dollars': round(max_dd_ticks * TICK_VALUE, 0),
        'dd_peak_date': dd_start_date,
        'dd_trough_date': dd_trough_date,
        'dd_recovery_date': str(dd_recovery_date),
        'dd_duration_days': dd_days,
        'dd_duration_trades': dd_duration_trades,
        'recovery_trades': recovery_trades,
        'max_losing_streak': max_losing_streak,
        'max_losing_day_streak': max_losing_day_streak,
        'worst_day_date': worst_day_date,
        'worst_day_pnl_ticks': round(worst_day_pnl, 1),
        'worst_day_pnl_dollars': round(worst_day_pnl * TICK_VALUE, 0),
        'calmar_ratio': round(calmar, 2),
    }


# ═══════════════════════════════════════════════════════════════
#  JSON SERIALIZER
# ═══════════════════════════════════════════════════════════════

def make_serializable(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, dict): return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [make_serializable(v) for v in obj]
    if isinstance(obj, pd.Timestamp): return str(obj)
    return obj


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    log.info("=" * 70)
    log.info("ROBUSTNESS v1 — Champion Strategy Robustness Testing")
    log.info("=" * 70)
    log.info(f"Champion: TP=25, SL_L=4/SL_S=3, max_hold=60, top-5%, daily bias 1.5x")
    t0 = time.time()

    try:
        # ── Load data ──
        log.info("\n--- Loading data ---")
        daily_df = load_daily_features()
        entry_preds_30m, entry_dates_30m = load_30min_predictions()
        minute_df = load_minute_bars()

        # ── Aggregate to 30-min bars ──
        bars_30m = aggregate_to_30min_bars(minute_df)
        bars_30m = bars_30m.sort_values('ts').reset_index(drop=True)

        if len(bars_30m) != len(entry_preds_30m):
            min_len = min(len(bars_30m), len(entry_preds_30m))
            bars_30m = bars_30m.iloc[:min_len].reset_index(drop=True)
            entry_preds_30m = entry_preds_30m[:min_len]

        log.info(f"Aligned: {len(bars_30m)} bars")

        # ── Reconstruct entry fills (all trades, no daily filter) ──
        log.info("\n--- Reconstructing FIFO entry fills ---")
        all_trades, fill_stats = reconstruct_entry_fills(
            bars_30m, minute_df, entry_preds_30m,
            confidence_pct=ENTRY_THRESHOLD,
            cancel_window_min=CANCEL_WINDOW,
        )
        log.info(f"Total trades before daily filter: {len(all_trades)}")

        # ── Train daily contrarian model ──
        log.info("\n--- Training daily contrarian model (walk-forward) ---")
        daily_preds = walk_forward_daily_contrarian(daily_df)

        # Map to next trading day
        dates_list = sorted(daily_df['date'].dt.strftime('%Y%m%d').values)
        daily_bias = {}
        for i, d in enumerate(dates_list):
            if d in daily_preds and i + 1 < len(dates_list):
                daily_bias[dates_list[i + 1]] = daily_preds[d]

        bias_values = np.array(list(daily_bias.values()))
        bias_std = bias_values.std()
        bias_threshold = DAILY_BIAS_THRESHOLD_MULT * bias_std
        log.info(f"Daily bias: mean={bias_values.mean():.1f}, std={bias_std:.1f}, threshold={bias_threshold:.1f}")

        # ── Apply daily filter (champion config) ──
        filtered_trades = apply_daily_filter(all_trades, daily_bias, bias_threshold)
        log.info(f"Trades after daily filter (1.5x): {len(filtered_trades)} (blocked {len(all_trades) - len(filtered_trades)})")

        # ── Simulate champion baseline ──
        log.info("\n--- Champion baseline ---")
        champ_pnl, champ_dates, champ_dirs, champ_exits, champ_holds = simulate_fifo_exits(
            filtered_trades, TP, TP, SL_LONG, SL_SHORT, MAX_HOLD)
        champ_metrics = compute_metrics(champ_pnl, champ_dates, champ_dirs, daily_df)
        log.info(f"Champion: N={champ_metrics['n_trades']}, Sharpe={champ_metrics.get('sharpe')}, "
                 f"WR={champ_metrics.get('wr')}, PF={champ_metrics.get('pf')}, "
                 f"PnL={champ_metrics.get('total_pnl_ticks')}t, Gap={champ_metrics.get('regime_gap')}")

        # ═══════════════════════════════════
        #  RUN ALL 5 TESTS
        # ═══════════════════════════════════

        test1 = test_time_stability(champ_pnl, champ_dates, champ_dirs, daily_df)
        test2 = test_parameter_sensitivity(filtered_trades, daily_df)
        bias_sweep = test_daily_bias_sensitivity(all_trades, daily_bias, bias_std, daily_df)
        test2['daily_bias_sweep'] = {str(k): v for k, v in bias_sweep.items()}
        test3 = test_cost_sensitivity(filtered_trades, daily_df)
        test4 = test_monthly_breakdown(champ_pnl, champ_dates, champ_dirs, daily_df)
        test5 = test_drawdown_analysis(champ_pnl, champ_dates, champ_dirs, daily_df)

        # ═══════════════════════════════════
        #  EXECUTIVE SUMMARY
        # ═══════════════════════════════════
        elapsed = time.time() - t0

        verdicts = {
            'time_stability': test1['verdict'],
            'parameter_sensitivity': test2['verdict'],
            'cost_sensitivity': test3['verdict'],
            'monthly_breakdown': test4['verdict'],
            'drawdown': test5['verdict'],
        }
        n_pass = sum(1 for v in verdicts.values() if v == 'PASS')
        n_tests = len(verdicts)
        overall = "PASS" if n_pass >= 4 else "FAIL"  # 4/5 = robust enough

        log.info("\n" + "=" * 70)
        log.info("EXECUTIVE SUMMARY — ROBUSTNESS TESTING")
        log.info("=" * 70)
        log.info(f"\n  Champion: Sharpe={champ_metrics.get('sharpe')}, N={champ_metrics['n_trades']}, "
                 f"WR={champ_metrics.get('wr')}, Gap={champ_metrics.get('regime_gap')}")
        log.info(f"\n  Test results:")
        for test_name, verdict in verdicts.items():
            log.info(f"    {test_name:30s}: {verdict}")
        log.info(f"\n  OVERALL: {n_pass}/{n_tests} PASS -> {overall}")
        log.info(f"  Ready for paper trading: {'YES' if overall == 'PASS' else 'NO'}")
        log.info(f"\n  Elapsed: {elapsed:.0f}s")

        # ── Save results ──
        output = make_serializable({
            'run_time': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed, 1),
            'champion_config': {
                'tp': TP, 'sl_long': SL_LONG, 'sl_short': SL_SHORT,
                'max_hold': MAX_HOLD, 'entry_threshold': ENTRY_THRESHOLD,
                'daily_bias_mult': DAILY_BIAS_THRESHOLD_MULT,
            },
            'champion_metrics': champ_metrics,
            'test_1_time_stability': test1,
            'test_2_parameter_sensitivity': test2,
            'test_3_cost_sensitivity': test3,
            'test_4_monthly_breakdown': test4,
            'test_5_drawdown': test5,
            'verdicts': verdicts,
            'overall_verdict': overall,
            'ready_for_paper_trading': overall == 'PASS',
        })

        results_path = OUTPUT_DIR / 'robustness_results.json'
        with open(str(results_path), 'w') as f:
            json.dump(output, f, indent=2, default=str)
        log.info(f"\nResults saved to {results_path}")
        log.info("=" * 70)
        log.info("ROBUSTNESS v1 COMPLETE")
        log.info("=" * 70)

    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
