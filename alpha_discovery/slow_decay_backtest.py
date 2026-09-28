#!/usr/bin/env python3
"""
Realistic Trading Backtest — Slow-Decay Composite Signal at 120s Horizon
========================================================================

KEY TEST: Can the slow-decay composite signal (total_depth_log, bid_pressure,
ask_pressure, total_ask_vol, ask_L3_orders) generate profit after market order costs?

Design:
  - Walk-forward: expanding-window z-score normalization (no look-ahead)
  - Non-overlapping 120s windows (~19 trades/day max)
  - Market order entry + exit: 1.24 ticks RT cost
  - Signal direction: SHORT when composite is HIGH (negative IC)
  - Multiple signal thresholds tested: all signals, top 50%, top 30%, top 20%
  - Holdout: first 50 days for developing, last 50 for validation
  - Detrended variant: remove intraday linear trend from features

Output: PnL curve, Sharpe, max drawdown, per-day breakdown, threshold comparison.
"""

import argparse
import gc
import json
import logging
import numpy as np
import sys
import time
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(format='%(asctime)s [backtest] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('backtest')

EXCLUDE_FEATURES = [0, 3, 8, 9]

# Cost structure
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
SPREAD_TICKS = 1.0
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24
TOTAL_COST_TICKS = SPREAD_TICKS + COMMISSION_TICKS  # 1.24

# Composite features (from IC decay analysis)
COMPOSITE_FEATURES = [
    'total_depth_log', 'total_ask_vol', 'ask_pressure',
    'bid_pressure', 'ask_L3_orders',
]

# Extended composite (top 10)
EXTENDED_FEATURES = COMPOSITE_FEATURES + [
    'ask_L2_orders', 'ask_L4_orders', 'ask_L5_orders',
    'rvol_10', 'event_int_50',
]


def load_data(data_dir, n_days, horizon_bars):
    """Load MBO feature data."""
    files = sorted(data_dir.glob('*_mbo_features.npz'))[:n_days]

    # Get feature names
    feature_names = None
    try:
        from alpha_discovery.mbo_features import get_feature_names
        all_names = get_feature_names()
        keep = [i for i in range(len(all_names)) if i not in EXCLUDE_FEATURES]
        feature_names = [all_names[i] for i in keep]
    except Exception:
        feature_names = [f'f{i}' for i in range(336)]

    name_to_idx = {n: i for i, n in enumerate(feature_names)}

    day_data = []
    for i, fpath in enumerate(files):
        date = fpath.stem.replace('_mbo_features', '')
        data = np.load(str(fpath))
        raw = data['mbo_features']
        mid = raw[:, 0].copy()

        # Forward-fill NaN mid prices
        mask = np.isnan(mid)
        if mask.any():
            fv = np.argmax(~mask)
            mid[:fv] = mid[fv]

        keep_cols = [j for j in range(raw.shape[1]) if j not in EXCLUDE_FEATURES]
        features = raw[:, keep_cols]
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        N = len(mid)
        if N <= horizon_bars + 100:
            continue

        day_data.append({
            'date': date,
            'mid': mid,
            'features': features,
            'N': N,
        })

        if (i + 1) % 20 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(files)}] {date}: {N:,} bars")

        del raw
        gc.collect()

    return day_data, feature_names, name_to_idx


def compute_composite_signal(features, feature_indices, detrend=False):
    """
    Compute composite signal from z-scored features.
    Uses WITHIN-DAY expanding window z-score (no look-ahead).
    The signal lives in within-day rank variation, not cross-day levels.
    """
    N = features.shape[0]
    n_feat = len(feature_indices)
    signals = np.full(N, np.nan)

    # Extract raw values for composite features
    raw = features[:, feature_indices].astype(np.float64)

    if detrend:
        # Expanding-window linear detrend (no look-ahead)
        # At time t, fit trend using only bars [0, t], subtract predicted value
        t_arr = np.arange(N, dtype=np.float64)

        for j in range(n_feat):
            x = raw[:, j].copy()
            # Compute expanding linear regression coefficients
            # Using online formula: slope = (n*sum(tx) - sum(t)*sum(x)) / (n*sum(t2) - sum(t)^2)
            cum_t = np.cumsum(t_arr)
            cum_x = np.cumsum(x)
            cum_tx = np.cumsum(t_arr * x)
            cum_t2 = np.cumsum(t_arr ** 2)
            n = np.arange(1, N + 1, dtype=np.float64)

            denom = n * cum_t2 - cum_t ** 2
            # Avoid division by zero for first few bars
            safe_denom = np.where(np.abs(denom) > 1e-12, denom, 1.0)
            slope = (n * cum_tx - cum_t * cum_x) / safe_denom
            intercept = (cum_x - slope * cum_t) / n

            # Detrend: x_detrended[t] = x[t] - (slope[t] * t + intercept[t])
            predicted = slope * t_arr + intercept
            raw[:, j] = x - predicted

    # Within-day expanding window z-score (no look-ahead)
    # At time t, z-score uses mean/std from bars [0, t-1]
    WARMUP = 1000

    # Vectorized: use data up to t-1 for normalization (strict no look-ahead)
    cumsum = np.cumsum(raw, axis=0)
    cumsum2 = np.cumsum(raw ** 2, axis=0)

    # Shift cumsums by 1 so cumsum_prev[t] = sum of raw[0:t]
    cumsum_prev = np.zeros_like(cumsum)
    cumsum_prev[1:] = cumsum[:-1]
    cumsum2_prev = np.zeros_like(cumsum2)
    cumsum2_prev[1:] = cumsum2[:-1]
    n_prev = np.arange(0, N, dtype=np.float64).reshape(-1, 1)  # count of past obs

    # Only valid for t >= WARMUP (need enough past data)
    with np.errstate(divide='ignore', invalid='ignore'):
        means = cumsum_prev / np.maximum(n_prev, 1)
        variances = cumsum2_prev / np.maximum(n_prev, 1) - means ** 2
        stds = np.sqrt(np.maximum(variances, 1e-12))
        z_scores = (raw - means) / stds

    signals[WARMUP:] = np.mean(z_scores[WARMUP:], axis=1)

    return signals


def compute_percentile_signal(features, feature_indices, trade_bars):
    """
    Compute within-day expanding percentile rank signal at trade bars.
    At each trade bar t, rank current feature value vs all past values
    within this day (bars 0 to t-1). Returns average percentile across features.
    This matches the per-day Spearman IC methodology.
    No look-ahead: only uses past data.
    """
    n_feat = len(feature_indices)
    raw = features[:, feature_indices].astype(np.float64)
    signals = np.full(len(trade_bars), np.nan)

    for i, t in enumerate(trade_bars):
        if t < 100:  # Need some warmup
            continue
        percentiles = np.empty(n_feat)
        for j in range(n_feat):
            past_vals = raw[:t, j]  # all values before t
            current = raw[t, j]
            # Percentile: fraction of past values <= current
            percentiles[j] = np.mean(past_vals <= current)
        signals[i] = np.mean(percentiles)  # Average percentile across features

    return signals


def run_backtest(day_data, name_to_idx, feature_list, horizon_bars,
                 thresholds, detrend=False, label=''):
    """
    Run walk-forward backtest for composite signal.

    Returns dict of results per threshold.
    """
    feature_indices = [name_to_idx[f] for f in feature_list if f in name_to_idx]
    actual_names = [f for f in feature_list if f in name_to_idx]
    logger.info(f"\n{'='*70}")
    logger.info(f"BACKTEST: {label} ({len(actual_names)} features, {horizon_bars/100:.0f}s horizon)")
    logger.info(f"Features: {actual_names}")
    logger.info(f"Detrend: {detrend}")
    logger.info(f"{'='*70}")

    # Results per threshold
    results = {}
    for thresh_name in thresholds:
        results[thresh_name] = {
            'daily_pnl': [],  # (date, pnl_ticks, n_trades, n_long, n_short)
            'all_trades': [],  # (date, entry_bar, direction, entry_mid, exit_mid, pnl_ticks)
        }

    horizon_sec = horizon_bars / 100
    avg_price = 5800  # approximate ES price
    tick_frac = TICK_SIZE / avg_price

    for day_idx, day in enumerate(day_data):
        date = day['date']
        mid = day['mid']
        features = day['features']
        N = day['N']

        # Compute composite signal (expanding window z-score)
        signal = compute_composite_signal(features, feature_indices, detrend=detrend)

        # Non-overlapping 120s windows
        # Start after warmup period (first 1000 bars = 10 seconds)
        trade_bars = np.arange(1000, N - horizon_bars, horizon_bars)

        if len(trade_bars) < 2:
            continue

        # Get signal values at trade points
        sig_values = signal[trade_bars]
        valid = np.isfinite(sig_values)

        if valid.sum() < 2:
            continue

        valid_bars = trade_bars[valid]
        valid_sigs = sig_values[valid]

        # Forward returns at each trade point (in ticks)
        fwd_ret_pct = np.array([
            (mid[b + horizon_bars] - mid[b]) / mid[b] if mid[b] > 0 else 0
            for b in valid_bars
        ])
        fwd_ret_ticks = fwd_ret_pct / tick_frac

        for thresh_name, (quantile_lo, quantile_hi) in thresholds.items():
            # Determine signal thresholds for this day
            # Use ONLY past signals for threshold computation (walk-forward)
            if day_idx == 0:
                # First day: use expanding threshold within the day
                # Use the first half to set thresholds for second half
                half = len(valid_sigs) // 2
                if half < 3:
                    continue
                past_sigs = valid_sigs[:half]
            else:
                # Use all past days' signals for threshold computation
                past_sigs = []
                for prev_idx in range(day_idx):
                    prev_day = day_data[prev_idx]
                    prev_signal = compute_composite_signal(
                        prev_day['features'], feature_indices, detrend=detrend)
                    prev_trade_bars = np.arange(1000, prev_day['N'] - horizon_bars, horizon_bars)
                    prev_vals = prev_signal[prev_trade_bars]
                    past_sigs.extend(prev_vals[np.isfinite(prev_vals)])
                past_sigs = np.array(past_sigs)

            if len(past_sigs) < 5:
                continue

            # Compute thresholds from past data
            thresh_short = np.percentile(past_sigs, quantile_hi)  # SHORT when signal is high
            thresh_long = np.percentile(past_sigs, quantile_lo)   # LONG when signal is low

            # Generate trades
            day_pnl_ticks = 0
            n_trades = 0
            n_long = 0
            n_short = 0

            for i, (bar, sig, ret) in enumerate(zip(valid_bars, valid_sigs, fwd_ret_ticks)):
                direction = 0
                if sig >= thresh_short:
                    direction = -1  # SHORT (high composite → negative return)
                elif sig <= thresh_long:
                    direction = +1  # LONG (low composite → positive return)

                if direction != 0:
                    # PnL = direction * return - cost
                    trade_pnl = direction * ret - TOTAL_COST_TICKS
                    day_pnl_ticks += trade_pnl
                    n_trades += 1
                    if direction > 0:
                        n_long += 1
                    else:
                        n_short += 1

                    results[thresh_name]['all_trades'].append({
                        'date': date,
                        'bar': int(bar),
                        'direction': direction,
                        'signal': float(sig),
                        'return_ticks': float(ret),
                        'pnl_ticks': float(trade_pnl),
                    })

            results[thresh_name]['daily_pnl'].append({
                'date': date,
                'pnl_ticks': float(day_pnl_ticks),
                'n_trades': n_trades,
                'n_long': n_long,
                'n_short': n_short,
            })

        if (day_idx + 1) % 10 == 0:
            logger.info(f"  [{day_idx+1}/{len(day_data)}] {date}")

    return results


def run_backtest_fast(day_data, name_to_idx, feature_list, horizon_bars,
                      detrend=False, label=''):
    """
    Fast backtest: compute signal once per day, trade at non-overlapping windows.
    Uses expanding-window z-score within each day (no cross-day normalization needed
    since we're using rank correlation direction, not absolute z-score thresholds).

    Simpler approach: trade ALL signals, just use direction (sign of composite).
    Also test with expanding-window percentile thresholds.
    """
    feature_indices = [name_to_idx[f] for f in feature_list if f in name_to_idx]
    actual_names = [f for f in feature_list if f in name_to_idx]

    logger.info(f"\n{'='*70}")
    logger.info(f"FAST BACKTEST: {label}")
    logger.info(f"Features: {actual_names}")
    logger.info(f"Horizon: {horizon_bars/100:.0f}s, Detrend: {detrend}")
    logger.info(f"{'='*70}")

    avg_price = 5800
    tick_frac = TICK_SIZE / avg_price

    # Collect all results
    daily_results = []
    all_signals = []  # (day_idx, bar, signal_value, fwd_ret_ticks)

    t0 = time.time()

    for day_idx, day in enumerate(day_data):
        date = day['date']
        mid = day['mid']
        features = day['features']
        N = day['N']

        # Compute composite signal
        signal = compute_composite_signal(features, feature_indices, detrend=detrend)

        # Non-overlapping trade windows
        trade_bars = np.arange(1000, N - horizon_bars, horizon_bars)
        if len(trade_bars) < 2:
            continue

        sig_values = signal[trade_bars]
        valid = np.isfinite(sig_values)
        if valid.sum() < 2:
            continue

        valid_bars = trade_bars[valid]
        valid_sigs = sig_values[valid]

        # Forward returns
        for b, s in zip(valid_bars, valid_sigs):
            if mid[b] > 0:
                ret_pct = (mid[b + horizon_bars] - mid[b]) / mid[b]
                ret_ticks = ret_pct / tick_frac
                all_signals.append((day_idx, date, int(b), float(s), float(ret_ticks)))

        if (day_idx + 1) % 10 == 0:
            logger.info(f"  [{day_idx+1}/{len(day_data)}] {date} ({time.time()-t0:.0f}s)")

    logger.info(f"Total trade opportunities: {len(all_signals)}")

    # Convert to arrays
    if not all_signals:
        logger.error("No trade signals generated!")
        return {}

    day_indices = np.array([s[0] for s in all_signals])
    dates = [s[1] for s in all_signals]
    bars = np.array([s[2] for s in all_signals])
    signals = np.array([s[3] for s in all_signals])
    returns = np.array([s[4] for s in all_signals])

    # IC check
    ic, pval = spearmanr(signals, returns)
    logger.info(f"Overall Spearman IC: {ic:+.4f} (p={pval:.2e})")

    # Strategies to test
    strategies = {
        'all_short': 'Trade all: SHORT when signal > median (walk-forward)',
        'all_long': 'Trade all: LONG when signal < median (walk-forward)',
        'top50_short': 'SHORT when signal > 50th percentile (walk-forward)',
        'top30_short': 'SHORT when signal > 70th percentile (walk-forward)',
        'top20_short': 'SHORT when signal > 80th percentile (walk-forward)',
        'top50_both': 'SHORT top 50% + LONG bottom 50% (walk-forward)',
        'top30_both': 'SHORT top 30% + LONG bottom 30% (walk-forward)',
        'top20_both': 'SHORT top 20% + LONG bottom 20% (walk-forward)',
    }

    results = {}
    unique_days = sorted(set(day_indices))
    MIN_WARMUP_DAYS = 10

    for strat_name, strat_desc in strategies.items():
        trades = []  # (date, direction, signal, return, pnl)
        daily_pnl = {}

        for day_idx in unique_days:
            if day_idx < MIN_WARMUP_DAYS:
                continue

            # Past signals for threshold computation
            past_mask = day_indices < day_idx
            past_sigs = signals[past_mask]
            if len(past_sigs) < 20:
                continue

            # Current day signals
            today_mask = day_indices == day_idx
            today_sigs = signals[today_mask]
            today_rets = returns[today_mask]
            today_dates = [dates[i] for i, m in enumerate(today_mask) if m]

            if len(today_sigs) == 0:
                continue

            date = today_dates[0]
            day_pnl = 0
            n_trades = 0

            # Compute thresholds from past data
            median = np.median(past_sigs)
            p70 = np.percentile(past_sigs, 70)
            p80 = np.percentile(past_sigs, 80)
            p30 = np.percentile(past_sigs, 30)
            p20 = np.percentile(past_sigs, 20)

            for sig, ret in zip(today_sigs, today_rets):
                direction = 0

                if strat_name == 'all_short':
                    if sig > median:
                        direction = -1
                elif strat_name == 'all_long':
                    if sig < median:
                        direction = +1
                elif strat_name == 'top50_short':
                    if sig > median:
                        direction = -1
                elif strat_name == 'top30_short':
                    if sig > p70:
                        direction = -1
                elif strat_name == 'top20_short':
                    if sig > p80:
                        direction = -1
                elif strat_name == 'top50_both':
                    if sig > median:
                        direction = -1
                    elif sig < median:
                        direction = +1
                elif strat_name == 'top30_both':
                    if sig > p70:
                        direction = -1
                    elif sig < p30:
                        direction = +1
                elif strat_name == 'top20_both':
                    if sig > p80:
                        direction = -1
                    elif sig < p20:
                        direction = +1

                if direction != 0:
                    trade_pnl = direction * ret - TOTAL_COST_TICKS
                    day_pnl += trade_pnl
                    n_trades += 1
                    trades.append({
                        'date': date,
                        'direction': direction,
                        'signal': float(sig),
                        'return_ticks': float(ret),
                        'pnl_ticks': float(trade_pnl),
                    })

            if n_trades > 0:
                daily_pnl[date] = {
                    'pnl_ticks': day_pnl,
                    'n_trades': n_trades,
                    'pnl_dollars': day_pnl * TICK_VALUE,
                }

        if not trades:
            results[strat_name] = {'n_trades': 0, 'description': strat_desc}
            continue

        # Compute stats
        trade_pnls = np.array([t['pnl_ticks'] for t in trades])
        daily_pnls = np.array([v['pnl_ticks'] for v in daily_pnl.values()])
        daily_trades = np.array([v['n_trades'] for v in daily_pnl.values()])

        total_pnl_ticks = np.sum(trade_pnls)
        total_pnl_dollars = total_pnl_ticks * TICK_VALUE
        n_trading_days = len(daily_pnl)
        avg_daily_pnl = np.mean(daily_pnls) if n_trading_days > 0 else 0
        std_daily_pnl = np.std(daily_pnls) if n_trading_days > 1 else 1
        sharpe = avg_daily_pnl / std_daily_pnl * np.sqrt(252) if std_daily_pnl > 0 else 0
        avg_trade_pnl = np.mean(trade_pnls)
        win_rate = np.mean(trade_pnls > 0) * 100
        avg_trades_per_day = np.mean(daily_trades)

        # Max drawdown
        cum_pnl = np.cumsum(daily_pnls)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdown = cum_pnl - running_max
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0

        # Profit factor
        gross_profit = np.sum(trade_pnls[trade_pnls > 0])
        gross_loss = abs(np.sum(trade_pnls[trade_pnls < 0]))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Holdout split
        sorted_dates = sorted(daily_pnl.keys())
        half = len(sorted_dates) // 2
        first_half_pnl = sum(daily_pnl[d]['pnl_ticks'] for d in sorted_dates[:half])
        second_half_pnl = sum(daily_pnl[d]['pnl_ticks'] for d in sorted_dates[half:])

        results[strat_name] = {
            'description': strat_desc,
            'n_trades': len(trades),
            'n_trading_days': n_trading_days,
            'total_pnl_ticks': float(total_pnl_ticks),
            'total_pnl_dollars': float(total_pnl_dollars),
            'avg_daily_pnl_ticks': float(avg_daily_pnl),
            'std_daily_pnl_ticks': float(std_daily_pnl),
            'sharpe': float(sharpe),
            'avg_trade_pnl_ticks': float(avg_trade_pnl),
            'win_rate': float(win_rate),
            'avg_trades_per_day': float(avg_trades_per_day),
            'profit_factor': float(pf),
            'max_drawdown_ticks': float(max_dd),
            'first_half_pnl': float(first_half_pnl),
            'second_half_pnl': float(second_half_pnl),
            'daily_pnl': daily_pnl,
        }

    return results


def print_results(results, label):
    """Print formatted results table."""
    logger.info(f"\n{'='*100}")
    logger.info(f"RESULTS: {label}")
    logger.info(f"{'='*100}")

    header = (f"{'Strategy':20s} {'Trades':>7s} {'Days':>5s} {'Tr/Day':>6s} "
              f"{'PnL($)':>9s} {'$/Day':>8s} {'Sharpe':>7s} {'WR%':>5s} "
              f"{'PF':>5s} {'MaxDD':>8s} {'1stHalf':>8s} {'2ndHalf':>8s}")
    logger.info(header)
    logger.info("-" * 100)

    for name, r in sorted(results.items()):
        if r.get('n_trades', 0) == 0:
            logger.info(f"  {name:20s}  NO TRADES")
            continue

        pnl_d = r['total_pnl_dollars']
        daily_d = r['avg_daily_pnl_ticks'] * TICK_VALUE
        first_d = r['first_half_pnl'] * TICK_VALUE
        second_d = r['second_half_pnl'] * TICK_VALUE
        dd_d = r['max_drawdown_ticks'] * TICK_VALUE

        logger.info(
            f"  {name:20s} {r['n_trades']:7d} {r['n_trading_days']:5d} "
            f"{r['avg_trades_per_day']:6.1f} "
            f"{'${:,.0f}'.format(pnl_d):>9s} {'${:,.0f}'.format(daily_d):>8s} "
            f"{r['sharpe']:+7.2f} {r['win_rate']:5.1f} "
            f"{r['profit_factor']:5.2f} {'${:,.0f}'.format(dd_d):>8s} "
            f"{'${:,.0f}'.format(first_d):>8s} {'${:,.0f}'.format(second_d):>8s}"
        )

    # Highlight best
    profitable = {k: v for k, v in results.items()
                  if v.get('n_trades', 0) > 0 and v.get('total_pnl_dollars', 0) > 0}
    if profitable:
        best = max(profitable, key=lambda k: profitable[k]['sharpe'])
        r = profitable[best]
        logger.info(f"\n  BEST: {best}")
        logger.info(f"    Sharpe: {r['sharpe']:+.2f}")
        logger.info(f"    Total PnL: ${r['total_pnl_dollars']:,.0f}")
        logger.info(f"    Avg PnL/day: ${r['avg_daily_pnl_ticks'] * TICK_VALUE:,.0f}")
        logger.info(f"    Win rate: {r['win_rate']:.1f}%")
        logger.info(f"    1st half: ${r['first_half_pnl'] * TICK_VALUE:,.0f}")
        logger.info(f"    2nd half: ${r['second_half_pnl'] * TICK_VALUE:,.0f}")

        if r['second_half_pnl'] > 0 and r['first_half_pnl'] > 0:
            logger.info(f"    >>> HOLDOUT STABLE! Both halves profitable.")
        elif r['second_half_pnl'] <= 0:
            logger.info(f"    >>> WARNING: Second half is negative. Signal may be decaying.")
    else:
        logger.info(f"\n  >>> NO PROFITABLE STRATEGY FOUND.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=str, default='120s', choices=['60s', '120s', '300s'])
    parser.add_argument('--n-days', type=int, default=100)
    parser.add_argument('--detrend', action='store_true', help='Detrend features within each day')
    args = parser.parse_args()

    horizon_bars = {'60s': 6000, '120s': 12000, '300s': 30000}[args.horizon]

    data_dir = ROOT / 'data' / 'processed' / 'mbo_features_cache'
    output_dir = ROOT / 'alpha_discovery' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f'slow_decay_backtest_{args.horizon}_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [backtest] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    logger.info(f"Slow-Decay Composite Backtest")
    logger.info(f"Horizon: {args.horizon} ({horizon_bars/100:.0f}s)")
    logger.info(f"Days: {args.n_days}")
    logger.info(f"Cost: {TOTAL_COST_TICKS:.2f} ticks RT (${TOTAL_COST_TICKS * TICK_VALUE:.2f})")
    logger.info(f"Detrend: {args.detrend}")

    # Load data
    logger.info("\nLoading data...")
    t0 = time.time()
    day_data, feature_names, name_to_idx = load_data(data_dir, args.n_days, horizon_bars)
    logger.info(f"Loaded {len(day_data)} days ({time.time()-t0:.1f}s)")

    # Run backtests
    all_results = {}

    # 1. Top-5 composite
    label = f'Top5 Composite @ {args.horizon}'
    if args.detrend:
        label += ' (detrended)'
    r = run_backtest_fast(day_data, name_to_idx, COMPOSITE_FEATURES,
                          horizon_bars, detrend=args.detrend, label=label)
    all_results[f'top5{"_detrend" if args.detrend else ""}'] = r
    print_results(r, label)

    # 2. Top-10 composite
    label = f'Top10 Composite @ {args.horizon}'
    if args.detrend:
        label += ' (detrended)'
    r = run_backtest_fast(day_data, name_to_idx, EXTENDED_FEATURES,
                          horizon_bars, detrend=args.detrend, label=label)
    all_results[f'top10{"_detrend" if args.detrend else ""}'] = r
    print_results(r, label)

    # 3. Individual features (top 3 only)
    for feat_name in ['total_depth_log', 'bid_pressure', 'ask_pressure']:
        if feat_name in name_to_idx:
            label = f'{feat_name} @ {args.horizon}'
            if args.detrend:
                label += ' (detrended)'
            r = run_backtest_fast(day_data, name_to_idx, [feat_name],
                                  horizon_bars, detrend=args.detrend, label=label)
            all_results[f'{feat_name}{"_detrend" if args.detrend else ""}'] = r
            print_results(r, label)

    # If not detrending, also run detrended versions
    if not args.detrend:
        label = f'Top5 Composite @ {args.horizon} (detrended)'
        r = run_backtest_fast(day_data, name_to_idx, COMPOSITE_FEATURES,
                              horizon_bars, detrend=True, label=label)
        all_results['top5_detrend'] = r
        print_results(r, label)

        label = f'total_depth_log @ {args.horizon} (detrended)'
        r = run_backtest_fast(day_data, name_to_idx, ['total_depth_log'],
                              horizon_bars, detrend=True, label=label)
        all_results['total_depth_log_detrend'] = r
        print_results(r, label)

    # 4. Within-day percentile rank approach (matches per-day IC methodology)
    logger.info(f"\n{'='*70}")
    logger.info(f"PERCENTILE RANK APPROACH (within-day, no look-ahead)")
    logger.info(f"{'='*70}")

    avg_price = 5800
    tick_frac = TICK_SIZE / avg_price

    for feature_list, group_name in [
        (COMPOSITE_FEATURES, 'pctile_top5'),
        (['total_depth_log'], 'pctile_depth'),
        (['bid_pressure'], 'pctile_bidpres'),
    ]:
        feature_indices = [name_to_idx[f] for f in feature_list if f in name_to_idx]
        actual_names = [f for f in feature_list if f in name_to_idx]
        label = f'Percentile {group_name} @ {args.horizon}'
        logger.info(f"\n  Computing {label}...")

        pctile_signals = []  # (day_idx, date, signal, fwd_ret_ticks)

        for day_idx, day in enumerate(day_data):
            mid = day['mid']
            features = day['features']
            N = day['N']
            trade_bars = np.arange(1000, N - horizon_bars, horizon_bars)

            if len(trade_bars) < 2:
                continue

            # Compute within-day percentile signal
            pctile = compute_percentile_signal(features, feature_indices, trade_bars)

            for i, b in enumerate(trade_bars):
                if np.isfinite(pctile[i]) and mid[b] > 0:
                    ret_pct = (mid[b + horizon_bars] - mid[b]) / mid[b]
                    ret_ticks = ret_pct / tick_frac
                    pctile_signals.append((day_idx, day['date'], float(pctile[i]), float(ret_ticks)))

        if not pctile_signals:
            logger.info(f"  No signals for {group_name}")
            continue

        day_indices = np.array([s[0] for s in pctile_signals])
        p_signals = np.array([s[2] for s in pctile_signals])
        p_returns = np.array([s[3] for s in pctile_signals])
        p_dates = [s[1] for s in pctile_signals]

        ic, pval = spearmanr(p_signals, p_returns)
        logger.info(f"  {group_name} IC: {ic:+.4f} (p={pval:.2e}), n={len(p_signals)}")

        # Per-day IC
        unique_days = sorted(set(day_indices))
        daily_ics = []
        for di in unique_days:
            mask = day_indices == di
            if mask.sum() >= 5:
                d_ic, _ = spearmanr(p_signals[mask], p_returns[mask])
                if np.isfinite(d_ic):
                    daily_ics.append(d_ic)
        if daily_ics:
            mean_ic = np.mean(daily_ics)
            t_stat = mean_ic / (np.std(daily_ics) / np.sqrt(len(daily_ics))) if np.std(daily_ics) > 0 else 0
            logger.info(f"  Per-day IC: {mean_ic:+.4f} (t={t_stat:.2f}, n={len(daily_ics)} days)")

        # Trade using percentile thresholds (directly, no cross-day normalization needed)
        # Percentile is already 0-1, so threshold is natural
        pctile_strats = {
            'short_above_60': (0.60, None),  # SHORT when percentile > 0.60
            'short_above_70': (0.70, None),
            'short_above_80': (0.80, None),
            'both_60': (0.60, 0.40),         # SHORT > 0.60, LONG < 0.40
            'both_70': (0.70, 0.30),
            'both_80': (0.80, 0.20),
        }

        results_pctile = {}
        MIN_WARMUP_DAYS = 10

        for strat_name, (short_thresh, long_thresh) in pctile_strats.items():
            trades_list = []
            daily_pnl = {}

            for di in unique_days:
                if di < MIN_WARMUP_DAYS:
                    continue
                mask = day_indices == di
                today_sigs = p_signals[mask]
                today_rets = p_returns[mask]
                today_date = p_dates[np.where(mask)[0][0]]

                day_pnl_val = 0
                n_tr = 0

                for sig, ret in zip(today_sigs, today_rets):
                    direction = 0
                    if sig >= short_thresh:
                        direction = -1  # SHORT when depth is high
                    elif long_thresh is not None and sig <= long_thresh:
                        direction = +1  # LONG when depth is low

                    if direction != 0:
                        trade_pnl = direction * ret - TOTAL_COST_TICKS
                        day_pnl_val += trade_pnl
                        n_tr += 1
                        trades_list.append(trade_pnl)

                if n_tr > 0:
                    daily_pnl[today_date] = {'pnl_ticks': day_pnl_val, 'n_trades': n_tr}

            if not trades_list:
                results_pctile[strat_name] = {'n_trades': 0}
                continue

            trade_pnls = np.array(trades_list)
            daily_pnls = np.array([v['pnl_ticks'] for v in daily_pnl.values()])
            daily_trades_arr = np.array([v['n_trades'] for v in daily_pnl.values()])
            n_tdays = len(daily_pnl)
            avg_dpnl = np.mean(daily_pnls) if n_tdays > 0 else 0
            std_dpnl = np.std(daily_pnls) if n_tdays > 1 else 1
            sharpe = avg_dpnl / std_dpnl * np.sqrt(252) if std_dpnl > 0 else 0

            cum = np.cumsum(daily_pnls)
            rmax = np.maximum.accumulate(cum)
            max_dd = np.min(cum - rmax) if len(cum) > 0 else 0

            gp = np.sum(trade_pnls[trade_pnls > 0])
            gl = abs(np.sum(trade_pnls[trade_pnls < 0]))
            pf = gp / gl if gl > 0 else float('inf')

            sorted_d = sorted(daily_pnl.keys())
            h = len(sorted_d) // 2
            fh_pnl = sum(daily_pnl[d]['pnl_ticks'] for d in sorted_d[:h])
            sh_pnl = sum(daily_pnl[d]['pnl_ticks'] for d in sorted_d[h:])

            results_pctile[strat_name] = {
                'n_trades': len(trades_list),
                'n_trading_days': n_tdays,
                'total_pnl_ticks': float(np.sum(trade_pnls)),
                'total_pnl_dollars': float(np.sum(trade_pnls) * TICK_VALUE),
                'avg_daily_pnl_ticks': float(avg_dpnl),
                'std_daily_pnl_ticks': float(std_dpnl),
                'sharpe': float(sharpe),
                'avg_trade_pnl_ticks': float(np.mean(trade_pnls)),
                'win_rate': float(np.mean(trade_pnls > 0) * 100),
                'avg_trades_per_day': float(np.mean(daily_trades_arr)),
                'profit_factor': float(pf),
                'max_drawdown_ticks': float(max_dd),
                'first_half_pnl': float(fh_pnl),
                'second_half_pnl': float(sh_pnl),
                'daily_pnl': daily_pnl,
            }

        all_results[group_name] = results_pctile
        print_results(results_pctile, label)

    # Save results (without daily_pnl details for each trade)
    save_results = {}
    for group_name, group_results in all_results.items():
        save_results[group_name] = {}
        for strat_name, r in group_results.items():
            save_r = {k: v for k, v in r.items() if k != 'daily_pnl'}
            save_results[group_name][strat_name] = save_r

    results_path = output_dir / f'slow_decay_backtest_{args.horizon}_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {results_path}")
    logger.info(f"Log saved: {log_path}")

    # Final summary
    logger.info(f"\n{'='*100}")
    logger.info(f"FINAL SUMMARY")
    logger.info(f"{'='*100}")

    best_sharpe = -999
    best_name = ''
    for group_name, group_results in all_results.items():
        for strat_name, r in group_results.items():
            if r.get('n_trades', 0) > 0:
                s = r.get('sharpe', -999)
                if s > best_sharpe:
                    best_sharpe = s
                    best_name = f"{group_name}/{strat_name}"

    if best_sharpe > 0:
        logger.info(f"  BEST OVERALL: {best_name} (Sharpe={best_sharpe:+.2f})")
    else:
        logger.info(f"  NO PROFITABLE COMBINATION FOUND")
        logger.info(f"  Best Sharpe: {best_sharpe:+.2f} ({best_name})")

    logger.info(f"\nDone.")


if __name__ == '__main__':
    main()
