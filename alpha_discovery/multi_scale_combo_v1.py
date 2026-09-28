#!/usr/bin/env python3
"""
multi_scale_combo_v1.py — Multi-Scale Signal Combination

Combines TWO complementary signals at different timescales:
  1. Daily contrarian: OFI reversal (negative IC at 1d horizon)
  2. 30-min continuation: OFI predicts short-term flow (positive IC at 30m)

HYPOTHESIS: The daily contrarian signal tells us WHICH SIDE is more likely
to work today. If we filter 30-min signals by this daily bias, we should:
  - Improve win rate (only trade aligned signals)
  - Reduce regime gap (adapt to daily conditions)
  - Maintain trade frequency (still using 30-min entries)

BASE STRATEGY (from regime_balanced_v1):
  TP=25, SL_L=4/SL_S=3, max_hold=60, top-5% confidence
  Passive entry/TP, market SL. Sharpe 5.17 baseline.

FILTER LOGIC:
  - Daily bias SHORT -> only take short signals from 30-min model
  - Daily bias LONG -> only take long signals from 30-min model
  - Daily bias NEUTRAL (low conviction) -> take both sides

Walk-forward: Load cached 30-min predictions, train daily model WF, combine.
HC #0: SLIDING windows only.
HC #428: Regime-agnostic, MFE-within-horizon.
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
ROOT = Path("/home/nick/Lvl3Quant")
DATA_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v1" / "daily_features.parquet"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
OUTPUT_DIR = ROOT / "output" / "multi_scale_combo_v1"
LOG_FILE = ROOT / "logs" / "multi_scale_combo_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MULTI-SCALE] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
TICK_SIZE = 1.0  # in data units (1 tick = 1 unit in minute bars, like regime_balanced)
TICK_SIZE_PTS = 0.25
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0

# Daily model params
DAILY_TRAIN_DAYS = 40
DAILY_SLIDE_DAYS = 5

# 30-min FIFO strategy params (from the Sharpe 5.17 baseline)
TP_LONG = 25
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD = 60
ENTRY_THRESHOLD = 0.05  # top 5% confidence
CANCEL_WINDOW = 10  # minutes
ENTRY_BAR_SIZE = 30  # minutes

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


def load_daily_features():
    """Load daily features from long_horizon_flow_v1 output."""
    df = pd.read_parquet(FEATURES_PATH)
    log.info(f"Loaded daily features: {len(df)} rows")
    return df


def load_30min_predictions():
    """Load cached 30-min entry predictions."""
    data = np.load(str(ENTRY_PREDS_PATH), allow_pickle=True)
    preds = data['entry_preds']
    dates = data['dates']
    log.info(f"Loaded 30-min predictions: {len(preds)} entries, "
             f"{np.sum(~np.isnan(preds))} non-NaN")
    return preds, dates


def load_minute_bars():
    """Load all minute bars."""
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


def _safe_polyfit_slope(arr):
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except:
        return 0.0


def aggregate_to_30min_bars(minute_df):
    """Aggregate minute bars to 30-min bars (matching regime_balanced_v1)."""
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
    """
    Walk-forward: train daily contrarian model (predict -fwd_return_1d),
    produce per-date predictions.
    Returns: dict of {date_str: contrarian_prediction}
    """
    n = len(daily_df)
    predictions = {}  # date_str -> contrarian pred
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
        y_train = -train_df.loc[train_mask, label].values  # CONTRARIAN target

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


def reconstruct_entry_fills(bars_30m, minute_df, pred_30m, confidence_pct, cancel_window_min):
    """Reconstruct FIFO entry fills (from regime_balanced_v1)."""
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
    n_signals = 0
    n_filled = 0
    n_cancelled = 0
    fill_delays = []

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
        signal_bar_end_np = np.datetime64(signal_bar_end)
        cancel_ts = signal_ts + pd.Timedelta(minutes=cancel_window_min + ENTRY_BAR_SIZE)
        cancel_ts_np = np.datetime64(cancel_ts)

        fill_mask = (day_ts >= signal_bar_end_np) & (day_ts <= cancel_ts_np)
        fill_candidates = day_minutes[fill_mask]

        if len(fill_candidates) == 0:
            n_cancelled += 1
            continue

        filled = False
        fill_price = None
        fill_ts = None

        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                if mbar['low'] <= limit_price - TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    fill_delays.append(j + 1)
                    break
            else:
                if mbar['high'] >= limit_price + TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar['ts_minute']
                    fill_delays.append(j + 1)
                    break

        if not filled:
            n_cancelled += 1
            continue

        n_filled += 1
        fill_ts_np = np.datetime64(fill_ts)
        remaining_mask = day_ts >= fill_ts_np
        remaining_minutes = day_minutes[remaining_mask]

        if len(remaining_minutes) < 2:
            n_filled -= 1
            n_cancelled += 1
            continue

        trade = {
            'idx': i,
            'date': date_str,
            'signal_ts': signal_ts,
            'fill_ts': pd.Timestamp(fill_ts),
            'fill_price': fill_price,
            'fill_delay_minutes': fill_delays[-1],
            'direction': direction,
            'pred_30m': float(pred_30m[i]),
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
    }
    log.info(f"  Entry fills: signals={n_signals}, filled={n_filled} ({fill_stats['fill_rate']:.1%})")
    return trades, fill_stats


def simulate_fifo_exits(trades, tp_long, tp_short, sl_long, sl_short, max_hold_minutes):
    """Simulate FIFO TP/SL exits (from regime_balanced_v1)."""
    pnls = []
    dates = []
    directions = []
    n_tp = 0
    n_sl = 0
    n_time = 0

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
                pnls.append(trade_pnl)
                dates.append(trade['date'])
                directions.append(direction)
                n_sl += 1
                exit_found = True
                break

            if tp_hit:
                trade_pnl = tp_ticks - RT_COMMISSION_TICKS
                pnls.append(trade_pnl)
                dates.append(trade['date'])
                directions.append(direction)
                n_tp += 1
                exit_found = True
                break

        if not exit_found:
            exit_minute = min(max_hold_minutes, n_remaining - 1)
            exit_minute = max(exit_minute, 1)
            exit_close = prices_close[exit_minute]
            if direction == 1:
                exit_fill = exit_close - TICK_SIZE
            else:
                exit_fill = exit_close + TICK_SIZE
            raw_pnl_ticks = (exit_fill - fill_price) / TICK_SIZE * direction
            trade_pnl = raw_pnl_ticks - RT_COMMISSION_TICKS
            pnls.append(trade_pnl)
            dates.append(trade['date'])
            directions.append(direction)
            n_time += 1

    total = n_tp + n_sl + n_time
    exit_stats = {
        'n_tp': n_tp, 'n_sl': n_sl, 'n_time': n_time,
        'tp_rate': n_tp / max(total, 1),
        'sl_rate': n_sl / max(total, 1),
        'time_rate': n_time / max(total, 1),
    }
    return np.array(pnls), np.array(dates), np.array(directions), exit_stats


def classify_regime(cc_return_ticks):
    if cc_return_ticks > 4:
        return 'green'
    elif cc_return_ticks < -4:
        return 'red'
    return 'flat'


def compute_strategy_metrics(pnl_arr, dates_arr, dirs_arr, label, daily_df=None):
    """Compute full strategy metrics including regime analysis."""
    n = len(pnl_arr)
    if n < 5:
        return {'name': label, 'n_trades': n, 'error': 'too few trades'}

    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    wr = len(wins) / n
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Daily aggregation
    unique_dates = sorted(set(dates_arr))
    daily_pnl = []
    for d in unique_dates:
        mask = dates_arr == d
        daily_pnl.append(pnl_arr[mask].sum())
    daily_pnl = np.array(daily_pnl)

    mean_d = daily_pnl.mean()
    std_d = daily_pnl.std(ddof=1) if len(daily_pnl) > 1 else np.nan
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d and std_d > 0 else np.nan

    downside = daily_pnl[daily_pnl < 0]
    ds_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (mean_d / ds_std * np.sqrt(252)) if ds_std and ds_std > 0 else np.nan

    # Max DD
    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) > 0 else 0

    # Day concentration
    daily_pnl_abs = np.abs(daily_pnl)
    day_conc = daily_pnl_abs.max() / daily_pnl_abs.sum() if daily_pnl_abs.sum() > 0 else 1.0

    # Regime analysis
    regime_metrics = {}
    if daily_df is not None:
        # Build date -> regime mapping
        date_regime = {}
        for _, row in daily_df.iterrows():
            d_str = row['date'].strftime('%Y%m%d')
            date_regime[d_str] = classify_regime(row.get('cc_return_ticks', 0))

        for reg in ['green', 'red', 'flat']:
            reg_mask = np.array([date_regime.get(d, 'flat') == reg for d in dates_arr])
            if reg_mask.sum() < 3:
                continue
            r_pnl = pnl_arr[reg_mask]
            r_dates = dates_arr[reg_mask]
            r_unique = sorted(set(r_dates))
            r_daily = np.array([r_pnl[r_dates == d].sum() for d in r_unique])
            r_mean = r_daily.mean()
            r_std = r_daily.std(ddof=1) if len(r_daily) > 1 else np.nan
            r_sharpe = (r_mean / r_std * np.sqrt(252)) if r_std and r_std > 0 else np.nan
            regime_metrics[reg] = {
                'n_trades': int(reg_mask.sum()),
                'n_days': len(r_unique),
                'sharpe': round(r_sharpe, 2) if not np.isnan(r_sharpe) else None,
                'wr': round(len(r_pnl[r_pnl > 0]) / len(r_pnl), 3),
                'mean_pnl': round(r_pnl.mean(), 2),
                'total_pnl': round(r_pnl.sum(), 1),
            }

    # Regime gap
    regime_sharpes = {k: v['sharpe'] for k, v in regime_metrics.items() if v.get('sharpe') is not None}
    regime_gap = None
    regime_pass = None
    if len(regime_sharpes) >= 2:
        vals = list(regime_sharpes.values())
        max_s = max(abs(v) for v in vals)
        if max_s > 0:
            regime_gap = round(abs(max(vals) - min(vals)) / max_s, 3)
            regime_pass = regime_gap <= 0.50

    # Long/short breakdown
    long_pnl = pnl_arr[dirs_arr == 1]
    short_pnl = pnl_arr[dirs_arr == -1]

    return {
        'name': label,
        'n_trades': n,
        'n_trading_days': len(unique_dates),
        'wr': round(wr, 3),
        'pf': round(pf, 2) if pf != float('inf') else 'inf',
        'sharpe': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'sortino': round(sortino, 2) if not np.isnan(sortino) else None,
        'total_pnl_ticks': round(pnl_arr.sum(), 1),
        'total_pnl_dollars': round(pnl_arr.sum() * TICK_VALUE, 0),
        'max_dd_ticks': round(max_dd, 1),
        'max_dd_dollars': round(max_dd * TICK_VALUE, 0),
        'day_concentration': round(day_conc, 3),
        'n_long': int((dirs_arr == 1).sum()),
        'n_short': int((dirs_arr == -1).sum()),
        'long_mean_pnl': round(long_pnl.mean(), 2) if len(long_pnl) > 0 else 0,
        'short_mean_pnl': round(short_pnl.mean(), 2) if len(short_pnl) > 0 else 0,
        'long_wr': round(len(long_pnl[long_pnl > 0]) / max(len(long_pnl), 1), 3),
        'short_wr': round(len(short_pnl[short_pnl > 0]) / max(len(short_pnl), 1), 3),
        'regime': regime_metrics,
        'regime_gap': regime_gap,
        'regime_pass': regime_pass,
    }


def main():
    log.info("=" * 70)
    log.info("MULTI-SCALE COMBO v1 — Daily Contrarian + 30-min Continuation")
    log.info("=" * 70)
    t0 = time.time()

    # ── 1. Load all data ──
    log.info("\n--- Loading data ---")
    daily_df = load_daily_features()
    entry_preds_30m, entry_dates_30m = load_30min_predictions()
    minute_df = load_minute_bars()

    # ── 2. Aggregate to 30-min bars ──
    log.info("\n--- Aggregating to 30-min bars ---")
    bars_30m = aggregate_to_30min_bars(minute_df)
    bars_30m = bars_30m.sort_values('ts').reset_index(drop=True)

    # Verify alignment with predictions
    if len(bars_30m) != len(entry_preds_30m):
        log.warning(f"Bar count ({len(bars_30m)}) != prediction count ({len(entry_preds_30m)})")
        min_len = min(len(bars_30m), len(entry_preds_30m))
        bars_30m = bars_30m.iloc[:min_len].reset_index(drop=True)
        entry_preds_30m = entry_preds_30m[:min_len]

    log.info(f"Aligned: {len(bars_30m)} bars, {np.sum(~np.isnan(entry_preds_30m))} non-NaN preds")

    # ── 3. Reconstruct FIFO entry fills for BASELINE (no daily filter) ──
    log.info("\n--- Reconstructing FIFO entry fills (baseline, top 5%) ---")
    all_trades, fill_stats = reconstruct_entry_fills(
        bars_30m, minute_df, entry_preds_30m,
        confidence_pct=ENTRY_THRESHOLD,
        cancel_window_min=CANCEL_WINDOW,
    )

    if len(all_trades) < 20:
        log.error(f"Only {len(all_trades)} trades reconstructed. Need >= 20.")
        sys.exit(1)

    log.info(f"Total filled trades: {len(all_trades)}")

    # ── 4. BASELINE: simulate without daily filter ──
    log.info("\n--- BASELINE: No daily filter ---")
    base_pnl, base_dates, base_dirs, base_exit = simulate_fifo_exits(
        all_trades, TP_LONG, TP_SHORT, SL_LONG, SL_SHORT, MAX_HOLD
    )
    baseline_metrics = compute_strategy_metrics(
        base_pnl, base_dates, base_dirs, "BASELINE_no_filter", daily_df
    )

    log.info(f"  BASELINE: n={baseline_metrics['n_trades']}, "
             f"Sharpe={baseline_metrics.get('sharpe')}, "
             f"WR={baseline_metrics.get('wr')}, "
             f"PF={baseline_metrics.get('pf')}, "
             f"PnL={baseline_metrics.get('total_pnl_ticks')}t, "
             f"Gap={baseline_metrics.get('regime_gap')}")

    # ── 5. Train daily contrarian model walk-forward ──
    log.info("\n" + "=" * 70)
    log.info("TRAINING DAILY CONTRARIAN MODEL (walk-forward)")
    log.info("=" * 70)

    daily_preds = walk_forward_daily_contrarian(daily_df)

    # Map predictions to NEXT trading day (the day we actually trade)
    # daily_preds[date_str] = contrarian pred for date_str
    # This prediction is ABOUT fwd_return from date_str to date_str+1
    # So the bias applies to the NEXT trading date
    dates_list = sorted(daily_df['date'].dt.strftime('%Y%m%d').values)
    daily_bias = {}  # trade_date -> bias direction
    for i, d in enumerate(dates_list):
        if d in daily_preds and i + 1 < len(dates_list):
            next_d = dates_list[i + 1]
            pred = daily_preds[d]
            daily_bias[next_d] = pred  # positive = expect DOWN -> short bias

    log.info(f"Daily bias computed for {len(daily_bias)} trading dates")

    # ── 6. Compute bias thresholds ──
    bias_values = np.array(list(daily_bias.values()))
    bias_std = bias_values.std()
    log.info(f"Daily bias stats: mean={bias_values.mean():.2f}, std={bias_std:.2f}")

    # ── 7. FILTERED strategies with different bias thresholds ──
    log.info("\n" + "=" * 70)
    log.info("FILTERED STRATEGIES — Daily bias applied to 30-min trades")
    log.info("=" * 70)

    bias_threshold_mults = [0.0, 0.25, 0.5, 1.0, 1.5]
    filter_results = {}

    for mult in bias_threshold_mults:
        bias_thresh = mult * bias_std

        # Filter trades by daily bias
        filtered_trades = []
        n_blocked = 0
        n_passed = 0
        n_no_bias = 0

        for trade in all_trades:
            trade_date = trade['date']
            direction = trade['direction']

            if trade_date not in daily_bias:
                # No daily bias available -> take trade as-is
                filtered_trades.append(trade)
                n_no_bias += 1
                continue

            bias = daily_bias[trade_date]

            if abs(bias) < bias_thresh:
                # NEUTRAL: bias too weak -> take both sides
                filtered_trades.append(trade)
                n_passed += 1
                continue

            # Strong bias: only take aligned trades
            # bias > 0 means expect DOWN -> only take shorts
            # bias < 0 means expect UP -> only take longs
            if bias > 0 and direction == -1:
                filtered_trades.append(trade)
                n_passed += 1
            elif bias < 0 and direction == 1:
                filtered_trades.append(trade)
                n_passed += 1
            else:
                n_blocked += 1

        if len(filtered_trades) < 10:
            log.info(f"  bias_thresh={mult:.2f}x: too few trades after filter ({len(filtered_trades)})")
            continue

        # Simulate exits on filtered trades
        f_pnl, f_dates, f_dirs, f_exit = simulate_fifo_exits(
            filtered_trades, TP_LONG, TP_SHORT, SL_LONG, SL_SHORT, MAX_HOLD
        )

        label = f"FILTERED_bias_{mult:.2f}x"
        metrics = compute_strategy_metrics(f_pnl, f_dates, f_dirs, label, daily_df)
        metrics['bias_threshold_mult'] = mult
        metrics['bias_threshold_abs'] = round(bias_thresh, 2)
        metrics['n_blocked'] = n_blocked
        metrics['n_passed'] = n_passed
        metrics['n_no_bias'] = n_no_bias
        metrics['exit_stats'] = f_exit

        filter_results[label] = metrics

        log.info(f"  {label}: n={metrics['n_trades']} (blocked {n_blocked}, no_bias {n_no_bias}), "
                 f"Sharpe={metrics.get('sharpe')}, WR={metrics.get('wr')}, "
                 f"PF={metrics.get('pf')}, PnL={metrics.get('total_pnl_ticks')}t, "
                 f"Gap={metrics.get('regime_gap')}")

    # ── 8. DIRECTIONAL FILTER: only take shorts when bias=short, only longs when bias=long ──
    log.info("\n--- DIRECTIONAL-ONLY FILTER (no neutral pass-through) ---")
    for mult in [0.0, 0.5, 1.0]:
        bias_thresh = mult * bias_std
        filtered_trades = []
        n_blocked = 0

        for trade in all_trades:
            trade_date = trade['date']
            direction = trade['direction']

            if trade_date not in daily_bias:
                continue  # skip if no bias
            
            bias = daily_bias[trade_date]

            if abs(bias) < bias_thresh:
                continue  # skip neutral days entirely

            if bias > 0 and direction == -1:
                filtered_trades.append(trade)
            elif bias < 0 and direction == 1:
                filtered_trades.append(trade)
            else:
                n_blocked += 1

        if len(filtered_trades) < 10:
            log.info(f"  dir_only_{mult:.1f}x: too few ({len(filtered_trades)})")
            continue

        f_pnl, f_dates, f_dirs, f_exit = simulate_fifo_exits(
            filtered_trades, TP_LONG, TP_SHORT, SL_LONG, SL_SHORT, MAX_HOLD
        )
        label = f"DIR_ONLY_bias_{mult:.1f}x"
        metrics = compute_strategy_metrics(f_pnl, f_dates, f_dirs, label, daily_df)
        metrics['bias_threshold_mult'] = mult
        metrics['n_blocked'] = n_blocked
        filter_results[label] = metrics

        log.info(f"  {label}: n={metrics['n_trades']}, "
                 f"Sharpe={metrics.get('sharpe')}, WR={metrics.get('wr')}, "
                 f"PnL={metrics.get('total_pnl_ticks')}t, Gap={metrics.get('regime_gap')}")

    # ── 9. LEADERBOARD ──
    log.info("\n" + "=" * 70)
    log.info("LEADERBOARD — BASELINE vs FILTERED")
    log.info("=" * 70)

    all_configs = {'BASELINE': baseline_metrics}
    all_configs.update(filter_results)

    ranked = sorted(all_configs.items(), key=lambda x: x[1].get('sharpe') or -999, reverse=True)

    log.info(f"{'Rk':>3} {'Config':>30} {'N':>5} {'Days':>5} {'WR':>6} {'Sharpe':>8} "
             f"{'Sort':>8} {'PF':>6} {'PnL_t':>8} {'Gap':>6} {'Pass':>5}")
    log.info("-" * 100)

    for i, (name, m) in enumerate(ranked):
        gap_str = f"{m['regime_gap']:.3f}" if m.get('regime_gap') is not None else "N/A"
        pass_str = "PASS" if m.get('regime_pass') else ("FAIL" if m.get('regime_pass') is False else "N/A")
        sharpe_str = f"{m['sharpe']:.2f}" if m.get('sharpe') is not None else "N/A"
        sortino_str = f"{m['sortino']:.2f}" if m.get('sortino') is not None else "N/A"
        pf_str = f"{m['pf']}" if m.get('pf') is not None else "N/A"

        log.info(f"{i+1:3d} {name:>30} {m['n_trades']:5d} {m.get('n_trading_days', 0):5d} "
                 f"{m['wr']:6.3f} {sharpe_str:>8} {sortino_str:>8} {pf_str:>6} "
                 f"{m['total_pnl_ticks']:8.1f} {gap_str:>6} {pass_str:>5}")

    # ── 10. Compare filtered vs baseline ──
    log.info("\n" + "=" * 70)
    log.info("COMPARISON: BEST FILTERED vs BASELINE")
    log.info("=" * 70)

    base_sharpe = baseline_metrics.get('sharpe') or 0
    best_filtered = None
    best_filtered_name = None
    for name, m in filter_results.items():
        s = m.get('sharpe') or -999
        if best_filtered is None or s > (best_filtered.get('sharpe') or -999):
            best_filtered = m
            best_filtered_name = name

    if best_filtered:
        f_sharpe = best_filtered.get('sharpe') or 0
        log.info(f"\n  BASELINE:")
        log.info(f"    Sharpe={base_sharpe}, WR={baseline_metrics['wr']}, "
                 f"PF={baseline_metrics['pf']}, N={baseline_metrics['n_trades']}")
        log.info(f"    Regime: {baseline_metrics.get('regime', {})}")
        log.info(f"    Regime gap: {baseline_metrics.get('regime_gap')}")

        log.info(f"\n  BEST FILTERED ({best_filtered_name}):")
        log.info(f"    Sharpe={f_sharpe}, WR={best_filtered['wr']}, "
                 f"PF={best_filtered['pf']}, N={best_filtered['n_trades']}")
        log.info(f"    Regime: {best_filtered.get('regime', {})}")
        log.info(f"    Regime gap: {best_filtered.get('regime_gap')}")

        improvement = f_sharpe - base_sharpe if base_sharpe and f_sharpe else None
        if improvement is not None:
            log.info(f"\n  Sharpe improvement: {improvement:+.2f} ({improvement / abs(base_sharpe) * 100:+.1f}%)"
                     if base_sharpe != 0 else f"\n  Sharpe improvement: {improvement:+.2f}")

        # Win rate comparison
        if baseline_metrics['wr'] and best_filtered['wr']:
            wr_diff = best_filtered['wr'] - baseline_metrics['wr']
            log.info(f"  WR improvement: {wr_diff:+.3f}")

        # Regime gap comparison
        bg = baseline_metrics.get('regime_gap')
        fg = best_filtered.get('regime_gap')
        if bg is not None and fg is not None:
            log.info(f"  Regime gap: {bg:.3f} -> {fg:.3f} ({'improved' if fg < bg else 'worsened'})")

    # ── 11. Save results ──
    elapsed = time.time() - t0
    log.info(f"\n=== COMPLETE ({elapsed:.0f}s) ===")

    # Convert for JSON
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    output = {
        'run_time': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'hypothesis': 'Daily contrarian signal (OFI reversal) filters 30-min continuation trades. '
                      'Only take 30-min signals aligned with daily bias.',
        'baseline': make_serializable(baseline_metrics),
        'filtered_results': make_serializable(filter_results),
        'best_filtered': make_serializable(best_filtered) if best_filtered else None,
        'daily_bias_stats': {
            'n_dates': len(daily_bias),
            'bias_mean': round(float(bias_values.mean()), 4),
            'bias_std': round(float(bias_std), 4),
        },
        'fill_stats': fill_stats,
        'strategy_params': {
            'tp_long': TP_LONG, 'tp_short': TP_SHORT,
            'sl_long': SL_LONG, 'sl_short': SL_SHORT,
            'max_hold': MAX_HOLD,
            'entry_threshold': ENTRY_THRESHOLD,
            'cancel_window': CANCEL_WINDOW,
        },
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # Save daily bias
    bias_df = pd.DataFrame([
        {'date': d, 'bias': b} for d, b in sorted(daily_bias.items())
    ])
    bias_df.to_parquet(OUTPUT_DIR / 'daily_bias.parquet', index=False)

    # ── EXECUTIVE SUMMARY ──
    log.info("\n" + "=" * 70)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 70)
    log.info(f"  BASELINE (no filter): Sharpe={base_sharpe}, N={baseline_metrics['n_trades']}, "
             f"Gap={baseline_metrics.get('regime_gap')}")
    if best_filtered:
        log.info(f"  BEST FILTERED ({best_filtered_name}): Sharpe={best_filtered.get('sharpe')}, "
                 f"N={best_filtered['n_trades']}, Gap={best_filtered.get('regime_gap')}")
        improvement = (best_filtered.get('sharpe') or 0) - base_sharpe if base_sharpe else None
        if improvement is not None:
            verdict = "IMPROVEMENT" if improvement > 0 else "NO IMPROVEMENT"
            log.info(f"  Sharpe delta: {improvement:+.2f} -> {verdict}")
    log.info(f"  Daily contrarian adds value: {'YES' if best_filtered and (best_filtered.get('sharpe') or 0) > base_sharpe else 'INCONCLUSIVE'}")
    log.info("=" * 70)


if __name__ == '__main__':
    main()
