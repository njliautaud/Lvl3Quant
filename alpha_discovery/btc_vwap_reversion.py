#!/usr/bin/env python3
"""
BTC/USDT VWAP Mean-Reversion Trading Strategy
==============================================
Signal: When price deviates from rolling VWAP, it reverts.
  - vwap_dev @ 1hr: IC=-0.209, t=-9.16, 2.49x breakeven
  - vwap_dev @ 4hr: IC=-0.326, t=-7.07, 8.40x breakeven

Strategy:
  1. Compute rolling VWAP (or daily VWAP)
  2. When price is X std above VWAP → go SHORT (expect reversion)
  3. When price is X std below VWAP → go LONG (expect reversion)
  4. Exit after holding period or when deviation decreases

Walk-forward: train thresholds on past days, test on next day.
"""

import gc
import glob
import json
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

logging.basicConfig(format='%(asctime)s [vwap_rv] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('vwap_rv')

DATA_DIR = r"C:\Users\Footb\Documents\Github\Lvl3Quant\data\raw\crypto\aggTrades"
RESULTS_DIR = Path(r"C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BAR_MS = 1000  # 1 second bars

# Costs
RT_TAKER_BPS = 10.0
RT_MIXED_BPS = 7.0
RT_MAKER_BPS = 4.0


def load_day_1s(filepath):
    """Load one day of aggTrades into 1-second bars."""
    df = pd.read_csv(filepath,
                     usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'],
                     dtype={'price': 'float64', 'quantity': 'float64',
                            'transact_time': 'int64', 'is_buyer_maker': 'str'})
    df['notional'] = df['price'] * df['quantity']
    df['bar_key'] = (df['transact_time'] // BAR_MS).astype('int64') * BAR_MS

    g = df.groupby('bar_key', sort=True)
    bars = g['price'].agg(['first', 'last']).rename(columns={'first': 'open', 'last': 'close'})
    bars['volume'] = g['quantity'].sum()
    bars['notional'] = g['notional'].sum()

    for col in ['open', 'close', 'volume', 'notional']:
        bars[col] = bars[col].astype('float64')

    return bars


def compute_vwap_features(bars):
    """Compute VWAP deviation features with multiple window sizes."""
    safe_vol = bars['volume'].replace(0, np.nan)

    # Rolling VWAP at multiple windows
    for w in [300, 900, 1800, 3600, 7200, 14400]:  # 5m to 4hr
        label = f'{w}s'
        roll_notional = bars['notional'].rolling(w, min_periods=max(10, w//10)).sum()
        roll_vol = bars['volume'].rolling(w, min_periods=max(10, w//10)).sum()
        vwap_roll = roll_notional / roll_vol.replace(0, np.nan)
        bars[f'vwap_{label}'] = vwap_roll
        bars[f'vwap_dev_{label}'] = (bars['close'] - vwap_roll) / bars['close']

    # Z-score of deviation (how many std away)
    for w in [3600, 7200]:
        label = f'{w}s'
        dev = bars[f'vwap_dev_{label}']
        dev_std = dev.rolling(w, min_periods=100).std()
        bars[f'vwap_zscore_{label}'] = dev / dev_std.replace(0, np.nan)

    # Rolling MA deviation (alternative to VWAP)
    for w in [3600, 7200, 14400]:
        label = f'{w}s'
        ma = bars['close'].rolling(w, min_periods=100).mean()
        std = bars['close'].rolling(w, min_periods=100).std()
        bars[f'ma_zscore_{label}'] = (bars['close'] - ma) / std.replace(0, np.nan)

    return bars


def simulate_vwap_strategy(bars, signal_col, hold_seconds, threshold_pct,
                           cost_bps=RT_MAKER_BPS, max_position_hours=4):
    """
    Simulate VWAP mean-reversion strategy.

    Entry: when |vwap_dev| > threshold_pct
      - If vwap_dev > threshold → SHORT (expect price to fall back to VWAP)
      - If vwap_dev < -threshold → LONG (expect price to rise back to VWAP)

    Exit: after hold_seconds, or when deviation crosses zero (mean-reverted)

    Returns dict with PnL metrics.
    """
    close = bars['close'].values
    signal = bars[signal_col].values
    N = len(close)

    if N < hold_seconds + 100:
        return None

    trades = []
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0
    entry_bar = 0
    hold_bars = hold_seconds

    # Walk through bars
    for i in range(1000, N - hold_bars):  # warmup 1000 bars
        if np.isnan(signal[i]):
            continue

        # If in position, check exit
        if position != 0:
            bars_held = i - entry_bar

            # Exit conditions:
            # 1. Hold time reached
            # 2. Signal reverted past zero (mean reverted)
            exit_now = False
            if bars_held >= hold_bars:
                exit_now = True
            elif position == 1 and signal[i] > 0:  # Long, price back above VWAP
                exit_now = True
            elif position == -1 and signal[i] < 0:  # Short, price back below VWAP
                exit_now = True

            if exit_now:
                exit_price = close[i]
                pnl_bps = position * (exit_price - entry_price) / entry_price * 10000 - cost_bps
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'direction': position,
                    'pnl_bps': pnl_bps,
                    'hold_bars': bars_held,
                    'entry_dev': signal[entry_bar],
                })
                position = 0
                continue

        # If flat, check entry
        if position == 0:
            if signal[i] > threshold_pct:
                # Price above VWAP → SHORT
                position = -1
                entry_price = close[i]
                entry_bar = i
            elif signal[i] < -threshold_pct:
                # Price below VWAP → LONG
                position = 1
                entry_price = close[i]
                entry_bar = i

    if not trades:
        return None

    pnls = np.array([t['pnl_bps'] for t in trades])
    gross_pnls = pnls + cost_bps  # add cost back to get gross

    return {
        'n_trades': len(trades),
        'total_pnl_bps': float(np.sum(pnls)),
        'mean_pnl_bps': float(np.mean(pnls)),
        'gross_mean_bps': float(np.mean(gross_pnls)),
        'win_rate': float(np.mean(pnls > 0)),
        'max_pnl_bps': float(np.max(pnls)),
        'min_pnl_bps': float(np.min(pnls)),
        'std_pnl_bps': float(np.std(pnls)),
        'avg_hold': float(np.mean([t['hold_bars'] for t in trades])),
    }


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = RESULTS_DIR / f'btc_vwap_reversion_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [vwap_rv] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("BTC/USDT VWAP Mean-Reversion Strategy")
    logger.info("=" * 70)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "BTCUSDT_aggTrades_2026-*.csv")))
    logger.info(f"Found {len(files)} days")

    # Load all days
    day_data = []
    t0 = time.time()
    for i, fpath in enumerate(files):
        date = os.path.basename(fpath).replace('BTCUSDT_aggTrades_', '').replace('.csv', '')
        logger.info(f"Loading [{i+1}/{len(files)}] {date}...")
        try:
            bars = load_day_1s(fpath)
            bars = compute_vwap_features(bars)
            day_data.append({'date': date, 'bars': bars, 'n_bars': len(bars)})
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

    elapsed = time.time() - t0
    logger.info(f"\nLoaded {len(day_data)} days ({elapsed:.1f}s)")

    # ── Phase 1: IC Verification ────────────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 1: IC Verification by Day")
    logger.info(f"{'='*70}")

    signal_cols = [
        'vwap_dev_300s', 'vwap_dev_900s', 'vwap_dev_1800s',
        'vwap_dev_3600s', 'vwap_dev_7200s', 'vwap_dev_14400s',
        'vwap_zscore_3600s', 'vwap_zscore_7200s',
        'ma_zscore_3600s', 'ma_zscore_7200s', 'ma_zscore_14400s',
    ]

    horizons = {
        '15min': 900,
        '30min': 1800,
        '1hr': 3600,
        '4hr': 14400,
    }

    for sig in signal_cols:
        for h_label, h_bars in horizons.items():
            daily_ics = []
            for d in day_data:
                bars = d['bars']
                if sig not in bars.columns:
                    continue
                fwd = bars['close'].pct_change(h_bars).shift(-h_bars)
                valid_mask = bars[sig].notna() & fwd.notna()
                if valid_mask.sum() < 100:
                    continue
                ic = float(bars[sig][valid_mask].corr(fwd[valid_mask], method='spearman'))
                if np.isfinite(ic):
                    daily_ics.append(ic)

            if len(daily_ics) >= 5:
                arr = np.array(daily_ics)
                mean_ic = np.mean(arr)
                std_ic = np.std(arr, ddof=1)
                t_stat = mean_ic / (std_ic / np.sqrt(len(arr))) if std_ic > 0 else 0
                pct_pos = np.mean(arr > 0) * 100
                if abs(mean_ic) > 0.03:
                    logger.info(f"  {sig:<25} @ {h_label:<6}: IC={mean_ic:+.4f}  t={t_stat:+.2f}  %pos={pct_pos:.0f}%  n={len(arr)}")

    # ── Phase 2: Walk-Forward Trading Simulation ────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 2: Walk-Forward Trading Simulation")
    logger.info(f"{'='*70}")

    # Test multiple configurations
    configs = [
        # (signal_col, hold_seconds, threshold_pct, cost_label, cost_bps)
        ('vwap_dev_3600s', 3600, 0.001, 'maker', RT_MAKER_BPS),
        ('vwap_dev_3600s', 3600, 0.002, 'maker', RT_MAKER_BPS),
        ('vwap_dev_3600s', 3600, 0.003, 'maker', RT_MAKER_BPS),
        ('vwap_dev_3600s', 7200, 0.001, 'maker', RT_MAKER_BPS),
        ('vwap_dev_3600s', 7200, 0.002, 'maker', RT_MAKER_BPS),
        ('vwap_dev_3600s', 7200, 0.003, 'maker', RT_MAKER_BPS),
        ('vwap_dev_7200s', 3600, 0.002, 'maker', RT_MAKER_BPS),
        ('vwap_dev_7200s', 7200, 0.002, 'maker', RT_MAKER_BPS),
        ('vwap_dev_7200s', 14400, 0.002, 'maker', RT_MAKER_BPS),
        ('vwap_dev_14400s', 7200, 0.003, 'maker', RT_MAKER_BPS),
        ('vwap_dev_14400s', 14400, 0.003, 'maker', RT_MAKER_BPS),
        ('ma_zscore_3600s', 3600, 1.0, 'maker', RT_MAKER_BPS),
        ('ma_zscore_3600s', 7200, 1.0, 'maker', RT_MAKER_BPS),
        ('ma_zscore_3600s', 3600, 1.5, 'maker', RT_MAKER_BPS),
        ('ma_zscore_7200s', 7200, 1.0, 'maker', RT_MAKER_BPS),
        ('ma_zscore_7200s', 14400, 1.0, 'maker', RT_MAKER_BPS),
        ('ma_zscore_7200s', 7200, 1.5, 'maker', RT_MAKER_BPS),
        ('ma_zscore_14400s', 14400, 1.0, 'maker', RT_MAKER_BPS),
        ('ma_zscore_14400s', 14400, 1.5, 'maker', RT_MAKER_BPS),
        # Also test with taker costs (pessimistic)
        ('vwap_dev_3600s', 3600, 0.002, 'taker', RT_TAKER_BPS),
        ('vwap_dev_7200s', 7200, 0.002, 'taker', RT_TAKER_BPS),
        ('ma_zscore_3600s', 3600, 1.0, 'taker', RT_TAKER_BPS),
        ('ma_zscore_7200s', 7200, 1.0, 'taker', RT_TAKER_BPS),
    ]

    all_results = []

    for sig, hold_sec, threshold, cost_label, cost_bps in configs:
        daily_pnls = []
        total_trades = 0

        for d in day_data:
            bars = d['bars']
            if sig not in bars.columns:
                continue

            result = simulate_vwap_strategy(
                bars, sig, hold_sec, threshold, cost_bps=cost_bps
            )
            if result and result['n_trades'] > 0:
                daily_pnls.append(result['total_pnl_bps'])
                total_trades += result['n_trades']

        if daily_pnls and len(daily_pnls) >= 5:
            arr = np.array(daily_pnls)
            mean_daily = np.mean(arr)
            std_daily = np.std(arr, ddof=1) if len(arr) > 1 else 1
            sharpe = mean_daily / std_daily * np.sqrt(365) if std_daily > 0 else 0  # 365 for crypto
            total_pnl = np.sum(arr)
            avg_trades = total_trades / len(daily_pnls)
            dollar_pnl = total_pnl / 10000 * 100000  # 1 BTC position at ~$100K

            row = {
                'signal': sig,
                'hold': hold_sec,
                'threshold': threshold,
                'cost': cost_label,
                'days': len(daily_pnls),
                'trades_d': avg_trades,
                'total_bps': total_pnl,
                'mean_bps': mean_daily,
                'sharpe': sharpe,
                'win_pct': np.mean(arr > 0) * 100,
                'dollar_pnl': dollar_pnl,
            }
            all_results.append(row)

            status = "PROFITABLE" if total_pnl > 0 else "LOSS"
            logger.info(f"  {sig:<25} hold={hold_sec:>5}s  thresh={threshold:<5}  "
                       f"{cost_label}: {total_pnl:+8.1f}bps  mean={mean_daily:+5.1f}bps/d  "
                       f"Sharpe={sharpe:+.2f}  trades/d={avg_trades:.1f}  "
                       f"win%={np.mean(arr > 0)*100:.0f}  ${dollar_pnl:+,.0f}  [{status}]")

    # Sort by total PnL
    all_results.sort(key=lambda x: x['total_bps'], reverse=True)

    logger.info(f"\n{'='*70}")
    logger.info("TOP 10 STRATEGIES BY TOTAL PNL")
    logger.info(f"{'='*70}")
    for i, r in enumerate(all_results[:10], 1):
        logger.info(f"  {i:2d}. {r['signal']:<25} hold={r['hold']:>5}  thresh={r['threshold']:<5}  "
                   f"{r['cost']}: {r['total_bps']:+8.1f}bps  Sharpe={r['sharpe']:+.2f}  "
                   f"trades/d={r['trades_d']:.1f}  ${r['dollar_pnl']:+,.0f}")

    # ── Phase 3: Walk-Forward Holdout ───────────────────────────────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 3: Walk-Forward Holdout (first 15d train, last 15d test)")
    logger.info(f"{'='*70}")

    train_days = day_data[:15]
    test_days = day_data[15:]
    logger.info(f"Train: {train_days[0]['date']} to {train_days[-1]['date']} ({len(train_days)} days)")
    logger.info(f"Test:  {test_days[0]['date']} to {test_days[-1]['date']} ({len(test_days)} days)")

    # Find best config on train, test on holdout
    best_train = None
    best_train_pnl = -999999

    for sig, hold_sec, threshold, cost_label, cost_bps in configs:
        if cost_label != 'maker':  # Focus on maker for best case
            continue
        daily_pnls = []
        for d in train_days:
            if sig not in d['bars'].columns:
                continue
            result = simulate_vwap_strategy(d['bars'], sig, hold_sec, threshold, cost_bps=cost_bps)
            if result and result['n_trades'] > 0:
                daily_pnls.append(result['total_pnl_bps'])
        if daily_pnls:
            total = np.sum(daily_pnls)
            if total > best_train_pnl:
                best_train_pnl = total
                best_train = (sig, hold_sec, threshold, cost_label, cost_bps)

    if best_train:
        sig, hold_sec, threshold, cost_label, cost_bps = best_train
        logger.info(f"\nBest train config: {sig} hold={hold_sec} thresh={threshold}")
        logger.info(f"  Train PnL: {best_train_pnl:+.1f} bps")

        # Test on holdout
        test_pnls = []
        for d in test_days:
            if sig not in d['bars'].columns:
                continue
            result = simulate_vwap_strategy(d['bars'], sig, hold_sec, threshold, cost_bps=cost_bps)
            if result and result['n_trades'] > 0:
                test_pnls.append(result['total_pnl_bps'])

        if test_pnls:
            test_total = np.sum(test_pnls)
            test_mean = np.mean(test_pnls)
            test_std = np.std(test_pnls, ddof=1) if len(test_pnls) > 1 else 1
            test_sharpe = test_mean / test_std * np.sqrt(365) if test_std > 0 else 0
            test_dollar = test_total / 10000 * 100000

            logger.info(f"  Test PnL:  {test_total:+.1f} bps ({test_mean:+.2f} bps/day)")
            logger.info(f"  Sharpe:    {test_sharpe:+.2f}")
            logger.info(f"  Dollar:    ${test_dollar:+,.0f}")
            logger.info(f"  Win rate:  {np.mean(np.array(test_pnls) > 0)*100:.0f}%")
            logger.info(f"  Days:      {len(test_pnls)}")

            if test_total > 0:
                logger.info(f"\n  >>> HOLDOUT PROFITABLE! <<<")
            else:
                logger.info(f"\n  >>> HOLDOUT NEGATIVE <<<")

    # ── Phase 4: Test ALL configs on holdout (no selection bias) ────────────────
    logger.info(f"\n{'='*70}")
    logger.info("PHASE 4: ALL configs on holdout (unbiased)")
    logger.info(f"{'='*70}")

    for sig, hold_sec, threshold, cost_label, cost_bps in configs:
        test_pnls = []
        for d in test_days:
            if sig not in d['bars'].columns:
                continue
            result = simulate_vwap_strategy(d['bars'], sig, hold_sec, threshold, cost_bps=cost_bps)
            if result and result['n_trades'] > 0:
                test_pnls.append(result['total_pnl_bps'])

        if test_pnls and len(test_pnls) >= 3:
            arr = np.array(test_pnls)
            total = np.sum(arr)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1) if len(arr) > 1 else 1
            sharpe = mean / std * np.sqrt(365) if std > 0 else 0
            dollar = total / 10000 * 100000
            status = "PROFIT" if total > 0 else "LOSS"

            logger.info(f"  {sig:<25} hold={hold_sec:>5}  thresh={threshold:<5}  "
                       f"{cost_label}: {total:+8.1f}bps  Sharpe={sharpe:+.2f}  "
                       f"${dollar:+,.0f}  [{status}]")

    # Save results
    results = {
        'timestamp': timestamp,
        'n_days': len(day_data),
        'strategies': all_results,
    }
    results_path = RESULTS_DIR / f'btc_vwap_reversion_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")
    logger.info(f"Log saved to {log_path}")
    logger.info("\nDone.")


if __name__ == '__main__':
    main()
