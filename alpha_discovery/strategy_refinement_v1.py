#!/usr/bin/env python3
"""
strategy_refinement_v1.py — Test Refinements to Champion Strategy

Based on deep analysis findings:
1. TIME FILTER: Block entries after 1 PM (13:00) — afternoon WR is 10-14%
2. DAY FILTER: Block Fridays — 7% WR
3. ADAPTIVE TP: Lower TP from 25 to [18, 20, 22] for near-miss recovery
4. COMBINATIONS: Test all combinations with regime gate (HC #428)

Uses same FIFO simulation as multi_scale_combo_v1.py.

HC #0: SLIDING windows only.
HC #428: Regime-agnostic (gap < 0.50), MFE-within-horizon.
HC #433: Plain English summaries.
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

# ── Paths ──
ROOT = Path("/home/nick/Lvl3Quant")
DATA_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
FEATURES_PATH = ROOT / "output" / "long_horizon_flow_v1" / "daily_features.parquet"
ENTRY_PREDS_PATH = ROOT / "output" / "mfe_mae_analysis" / "entry_predictions.npz"
OUTPUT_DIR = ROOT / "output" / "strategy_refinement_v1"
LOG_FILE = ROOT / "logs" / "strategy_refinement_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REFINE] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
TICK_SIZE = 1.0
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0

# Base strategy params
TP_LONG = 25
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3
MAX_HOLD = 60
ENTRY_THRESHOLD = 0.05
CANCEL_WINDOW = 10
ENTRY_BAR_SIZE = 30
DAILY_BIAS_THRESHOLD_MULT = 1.5

# Refinement configs to test
CONFIGS = []

# Base (no changes)
CONFIGS.append({
    'name': 'BASE',
    'time_filter': None,
    'day_filter': None,
    'tp_override': None,
})

# Time filters
for cutoff_hour in [12, 13, 14]:
    CONFIGS.append({
        'name': f'TIME_before_{cutoff_hour}',
        'time_filter': cutoff_hour,
        'day_filter': None,
        'tp_override': None,
    })

# Day filters
CONFIGS.append({
    'name': 'NO_FRIDAY',
    'time_filter': None,
    'day_filter': [4],  # Friday = 4
    'tp_override': None,
})
CONFIGS.append({
    'name': 'NO_FRI_THU',
    'time_filter': None,
    'day_filter': [3, 4],  # Thu, Fri
    'tp_override': None,
})
CONFIGS.append({
    'name': 'MON_TUE_WED_ONLY',
    'time_filter': None,
    'day_filter': [3, 4],  # same as NO_FRI_THU
    'tp_override': None,
})

# Adaptive TP
for tp in [18, 20, 22]:
    CONFIGS.append({
        'name': f'TP_{tp}',
        'time_filter': None,
        'day_filter': None,
        'tp_override': tp,
    })

# Combinations
for cutoff in [12, 13]:
    CONFIGS.append({
        'name': f'TIME_{cutoff}_NO_FRI',
        'time_filter': cutoff,
        'day_filter': [4],
        'tp_override': None,
    })
    for tp in [18, 20]:
        CONFIGS.append({
            'name': f'TIME_{cutoff}_NO_FRI_TP_{tp}',
            'time_filter': cutoff,
            'day_filter': [4],
            'tp_override': tp,
        })

# Morning only + adaptive TP
for tp in [18, 20, 22]:
    CONFIGS.append({
        'name': f'MORNING_TP_{tp}',
        'time_filter': 13,
        'day_filter': None,
        'tp_override': tp,
    })

# Best combo: morning, no Fri, adaptive TP
for tp in [18, 20, 22]:
    CONFIGS.append({
        'name': f'BEST_COMBO_TP_{tp}',
        'time_filter': 13,
        'day_filter': [4],
        'tp_override': tp,
    })


def aggregate_to_30min_bars(minute_df):
    """Aggregate minute bars to 30-min bars (matching multi_scale_combo_v1)."""
    df = minute_df.copy()
    df['bar_key'] = df['ts_minute'].dt.floor('30min')

    records = []
    for (date_str, bar_key), grp in df.groupby(['date', 'bar_key']):
        if len(grp) < 2:
            continue
        close_arr = grp['close'].values
        vol_arr = grp['volume'].values if 'volume' in grp.columns else np.zeros(len(grp))
        ofi_arr = grp['ofi_1min'].values if 'ofi_1min' in grp.columns else np.zeros(len(grp))

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


def load_data():
    """Load all required data."""
    # Minute bars
    files = sorted(DATA_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        df['date'] = f.stem
        frames.append(df)
    minute_df = pd.concat(frames, ignore_index=True)
    minute_df['ts_minute'] = pd.to_datetime(minute_df['ts_minute'], utc=True)
    minute_df = minute_df.sort_values('ts_minute').reset_index(drop=True)
    log.info(f"Loaded {len(minute_df):,} minute bars across {len(files)} days")

    # Build 30-min bars (MUST match multi_scale_combo_v1 aggregation)
    bars_30m = aggregate_to_30min_bars(minute_df)

    # 30-min predictions
    data = np.load(str(ENTRY_PREDS_PATH), allow_pickle=True)
    pred_30m = data['entry_preds']
    log.info(f"Loaded {len(pred_30m)} predictions, {np.sum(~np.isnan(pred_30m))} non-NaN")
    log.info(f"30-min bars: {len(bars_30m)}, predictions: {len(pred_30m)}")

    # Daily features for bias filter
    daily_df = pd.read_parquet(FEATURES_PATH)
    if 'date' not in daily_df.columns:
        daily_df = daily_df.reset_index()
        daily_df.columns = ['date'] + list(daily_df.columns[1:])

    if 'session_ofi' in daily_df.columns:
        ofi_col = daily_df['session_ofi']
        ofi_std = ofi_col.expanding(min_periods=20).std()
        ofi_mean = ofi_col.expanding(min_periods=20).mean()
        daily_df['ofi_z'] = (ofi_col - ofi_mean) / ofi_std.clip(lower=1e-6)

    return minute_df, bars_30m, pred_30m, daily_df


def run_config(config, minute_df, bars_30m, pred_30m, daily_df):
    """Run a single configuration through the full FIFO pipeline."""
    time_filter = config['time_filter']
    day_filter = config['day_filter']
    tp_override = config['tp_override']

    tp_long = tp_override if tp_override else TP_LONG
    tp_short = tp_override if tp_override else TP_SHORT

    # Build daily bias lookup
    daily_bias_lookup = {}
    if 'ofi_z' in daily_df.columns:
        for _, row in daily_df.iterrows():
            d = str(row['date'])
            z = row['ofi_z']
            if pd.isna(z):
                daily_bias_lookup[d] = 'neutral'
            elif z > DAILY_BIAS_THRESHOLD_MULT:
                daily_bias_lookup[d] = 'short'
            elif z < -DAILY_BIAS_THRESHOLD_MULT:
                daily_bias_lookup[d] = 'long'
            else:
                daily_bias_lookup[d] = 'neutral'

    # Thresholds
    valid_mask = ~np.isnan(pred_30m)
    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - ENTRY_THRESHOLD)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], ENTRY_THRESHOLD)

    minute_lookup = {}
    for date_str, grp in minute_df.groupby('date'):
        minute_lookup[date_str] = grp.sort_values('ts_minute').reset_index(drop=True)

    ts_col = 'ts' if 'ts' in bars_30m.columns else 'ts_minute'
    bars_ts = bars_30m[ts_col].values
    bars_dates = bars_30m['date'].values
    bars_close = bars_30m['close'].values

    pnls = []
    dates = []
    directions = []
    n_filtered_time = 0
    n_filtered_day = 0

    for i in range(len(bars_30m)):
        if i >= len(pred_30m) or np.isnan(pred_30m[i]):
            continue

        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1
        elif pred_30m[i] <= lower_thresh:
            direction = -1
        else:
            continue

        date_str = str(bars_dates[i])
        signal_ts = pd.Timestamp(bars_ts[i])

        # Daily bias filter
        bias = daily_bias_lookup.get(date_str, 'neutral')
        if bias == 'short' and direction == 1:
            continue
        if bias == 'long' and direction == -1:
            continue

        # TIME FILTER
        if time_filter is not None:
            signal_hour = signal_ts.hour
            # Convert UTC to ET (subtract 4 for EDT, 5 for EST)
            # Approximate: if hour > 17, it's likely morning ET
            et_hour = signal_hour - 4  # EDT approximation
            if et_hour < 0:
                et_hour += 24
            if et_hour >= time_filter:
                n_filtered_time += 1
                continue

        # DAY FILTER
        if day_filter is not None:
            dow = signal_ts.dayofweek
            if dow in day_filter:
                n_filtered_day += 1
                continue

        signal_price = bars_close[i]
        if date_str not in minute_lookup:
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes['ts_minute'].values

        limit_price = signal_price
        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE)
        cancel_ts = signal_ts + pd.Timedelta(minutes=CANCEL_WINDOW + ENTRY_BAR_SIZE)

        fill_mask = (day_ts >= np.datetime64(signal_bar_end)) & (day_ts <= np.datetime64(cancel_ts))
        fill_candidates = day_minutes[fill_mask]
        if len(fill_candidates) == 0:
            continue

        filled = False
        fill_price = None
        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                if mbar['low'] <= limit_price - TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    break
            else:
                if mbar['high'] >= limit_price + TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    break

        if not filled:
            continue

        fill_ts = mbar['ts_minute']
        fill_ts_np = np.datetime64(fill_ts)
        remaining_mask = day_ts >= fill_ts_np
        remaining_minutes = day_minutes[remaining_mask]
        if len(remaining_minutes) < 2:
            continue

        # Simulate exit
        prices_close = remaining_minutes['close'].values
        prices_high = remaining_minutes['high'].values
        prices_low = remaining_minutes['low'].values

        tp_ticks = tp_long if direction == 1 else tp_short
        sl_ticks = SL_LONG if direction == 1 else SL_SHORT
        tp_price = fill_price + direction * tp_ticks * TICK_SIZE
        sl_price = fill_price - direction * sl_ticks * TICK_SIZE

        exit_pnl = 0.0
        exit_type = 'time'
        max_check = min(MAX_HOLD, len(remaining_minutes))

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

            if sl_hit and tp_hit: sl_hit, tp_hit = True, False

            if sl_hit:
                exit_pnl = -(sl_ticks + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                exit_type = 'sl'
                break
            if tp_hit:
                exit_pnl = tp_ticks - RT_COMMISSION_TICKS
                exit_type = 'tp'
                break

        if exit_type == 'time':
            em = min(MAX_HOLD, len(remaining_minutes) - 1)
            em = max(em, 1)
            ec = prices_close[em]
            ef = ec - TICK_SIZE if direction == 1 else ec + TICK_SIZE
            exit_pnl = (ef - fill_price) / TICK_SIZE * direction - RT_COMMISSION_TICKS

        pnls.append(exit_pnl)
        dates.append(date_str)
        directions.append(direction)

    if len(pnls) < 10:
        return {
            'name': config['name'],
            'n_trades': len(pnls),
            'error': 'too few trades',
        }

    pnl_arr = np.array(pnls)
    dates_arr = np.array(dates)
    dirs_arr = np.array(directions)

    # Compute metrics
    n = len(pnl_arr)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    wr = len(wins) / n
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

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

    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    max_dd = (peak - cum).max()

    # Regime analysis
    # Load daily returns for regime classification
    daily_df_with_regime = daily_df.copy()
    if 'close_vs_open_ticks' in daily_df_with_regime.columns:
        regime_col = 'close_vs_open_ticks'
    elif 'cc_return_3d' in daily_df_with_regime.columns:
        regime_col = None  # compute from daily PnL context
    else:
        regime_col = None

    # Compute regime from ES daily returns
    date_regime = {}
    for _, row in daily_df.iterrows():
        d = str(row['date'])
        ret = row.get('close_vs_open_ticks', 0)
        if pd.isna(ret):
            ret = 0
        if ret > 4:
            date_regime[d] = 'green'
        elif ret < -4:
            date_regime[d] = 'red'
        else:
            date_regime[d] = 'flat'

    green_pnl = [pnl_arr[dates_arr == d].sum() for d in unique_dates if date_regime.get(d, 'flat') == 'green']
    red_pnl = [pnl_arr[dates_arr == d].sum() for d in unique_dates if date_regime.get(d, 'flat') == 'red']

    green_sharpe = np.nan
    red_sharpe = np.nan
    if len(green_pnl) > 3:
        gp = np.array(green_pnl)
        if gp.std() > 0:
            green_sharpe = gp.mean() / gp.std() * np.sqrt(252)
    if len(red_pnl) > 3:
        rp = np.array(red_pnl)
        if rp.std() > 0:
            red_sharpe = rp.mean() / rp.std() * np.sqrt(252)

    regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 1e-6)
    regime_pass = regime_gap < 0.50

    result = {
        'name': config['name'],
        'n_trades': n,
        'n_trading_days': len(unique_dates),
        'wr': wr,
        'pf': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'total_pnl_ticks': float(pnl_arr.sum()),
        'total_pnl_dollars': float(pnl_arr.sum() * TICK_VALUE),
        'max_dd_ticks': float(max_dd),
        'n_long': int((dirs_arr == 1).sum()),
        'n_short': int((dirs_arr == -1).sum()),
        'green_sharpe': green_sharpe,
        'red_sharpe': red_sharpe,
        'regime_gap': regime_gap,
        'regime_pass': regime_pass,
        'filtered_time': n_filtered_time,
        'filtered_day': n_filtered_day,
    }
    return result


def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("STRATEGY REFINEMENT v1 — Testing Time/Day/TP Filters")
    log.info(f"Configs to test: {len(CONFIGS)}")
    log.info("=" * 70)

    minute_df, bars_30m, pred_30m, daily_df = load_data()

    results = []
    for i, config in enumerate(CONFIGS):
        log.info(f"\n[{i+1}/{len(CONFIGS)}] Testing: {config['name']}")
        result = run_config(config, minute_df, bars_30m, pred_30m, daily_df)
        results.append(result)

        if 'error' not in result:
            log.info(f"  Trades={result['n_trades']}, WR={result['wr']:.1%}, "
                     f"Sharpe={result['sharpe']:.2f}, PF={result['pf']:.2f}, "
                     f"Regime gap={result['regime_gap']:.3f} {'PASS' if result['regime_pass'] else 'FAIL'}")

    # Sort by Sharpe (regime-passing only first)
    passing = [r for r in results if r.get('regime_pass', False) and 'error' not in r]
    failing = [r for r in results if not r.get('regime_pass', True) or 'error' in r]
    passing.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

    log.info("\n" + "=" * 70)
    log.info("RESULTS — REGIME-PASSING CONFIGS (sorted by Sharpe)")
    log.info("=" * 70)
    log.info(f"{'Config':<30s} {'Trades':>6s} {'WR':>6s} {'Sharpe':>7s} {'PF':>5s} "
             f"{'PnL($)':>8s} {'Gap':>5s}")
    log.info("-" * 70)
    for r in passing:
        log.info(f"{r['name']:<30s} {r['n_trades']:>6d} {r['wr']:>6.1%} "
                 f"{r['sharpe']:>7.2f} {r['pf']:>5.2f} "
                 f"${r['total_pnl_dollars']:>7.0f} {r['regime_gap']:>5.3f}")

    if failing:
        log.info(f"\nRegime-FAILING configs: {len(failing)}")
        for r in failing[:10]:
            if 'error' in r:
                log.info(f"  {r['name']}: {r['error']}")
            else:
                log.info(f"  {r['name']}: Sharpe={r.get('sharpe', 'N/A'):.2f}, "
                         f"gap={r.get('regime_gap', 'N/A'):.3f}")

    # Save
    elapsed = time.time() - start_time
    summary = {
        'run_time': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'n_configs_tested': len(CONFIGS),
        'n_regime_passing': len(passing),
        'results': results,
        'best_config': passing[0] if passing else None,
    }
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\nCompleted in {elapsed:.0f}s")
    if passing:
        best = passing[0]
        log.info(f"\nBEST CONFIG: {best['name']}")
        log.info(f"  Sharpe: {best['sharpe']:.2f} (base: see BASE row)")
        log.info(f"  WR: {best['wr']:.1%}")
        log.info(f"  Trades: {best['n_trades']}")
        log.info(f"  PnL: ${best['total_pnl_dollars']:.0f}")
        log.info(f"  Regime gap: {best['regime_gap']:.3f}")


if __name__ == '__main__':
    main()
