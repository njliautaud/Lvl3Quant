#!/usr/bin/env python3
"""
VIX Term Structure Contango + Sector Oversold Contrarian Signal Backtest
========================================================================
Thesis: When VIX is in steep contango (front-month VIX << VIX futures => fear priced
into the curve but spot calming) AND a sector ETF is oversold (RSI < 30), institutions
have already hedged and the sector is poised for a short-term bounce.

Conversely, when VIX is in backwardation (spot > futures => panic NOW) AND a sector
is overbought (RSI > 70), complacency is overdone => sell/avoid.

This is NOT: VIX standalone, VIX term structure standalone, sector dispersion, or VRP.
It is the INTERACTION of term structure slope with sector-specific technical oversold.

Signal construction:
  1. VIX contango slope = (VIX3M - VIX) / VIX  (positive = contango, negative = backwardation)
  2. Sector RSI(14)
  3. LONG signal: contango_slope > threshold AND RSI < 30
  4. SHORT signal: contango_slope < -threshold AND RSI > 70 (optional)
  5. Hold 5 trading days

Walk-forward: 60-day sliding window to set contango threshold adaptively.
5-gate validation: Sharpe>0.5, regime gap<0.5, perm p<0.05, max DD reasonable, trades>30.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
import sys

# ============================================================
# CONFIG
# ============================================================
SECTOR_ETFS = ['XLE', 'XLU', 'XLP', 'XLK', 'XLY', 'XLF', 'XLRE', 'XLV', 'XLI', 'XLB', 'XLC']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'
VIX3M_TICKER = '^VIX3M'  # CBOE 3-month VIX (for term structure)
HOLD_DAYS = 5
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
TRAIN_WINDOW = 60  # sliding window days
MIN_TRADES = 30
START_DATE = '2018-01-01'
END_DATE = '2026-08-15'
N_PERMUTATIONS = 1000


def download_data():
    """Download all required data."""
    print("Downloading data...")
    all_tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER, VIX3M_TICKER]

    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: INSUFFICIENT DATA ({len(df)} days)")
        except Exception as e:
            print(f"  {ticker}: DOWNLOAD FAILED - {e}")

    # Check VIX3M availability - if not available, construct proxy from VIX
    if VIX3M_TICKER not in data:
        print(f"\n  WARNING: {VIX3M_TICKER} not available. Constructing contango proxy from VIX momentum.")
        # Use VIX 20-day MA as proxy for "longer-term" VIX level
        # contango proxy = (VIX_20MA - VIX) / VIX
        # When VIX spikes above its MA => backwardation-like (acute fear)
        # When VIX is below its MA => contango-like (complacency / hedges in place)

    return data


def compute_rsi(prices, period=14):
    """Compute RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_contango_slope(data):
    """
    Compute VIX term structure slope.
    If VIX3M available: (VIX3M - VIX) / VIX
    Otherwise: proxy from VIX vs its 20-day MA.
    """
    vix = data[VIX_TICKER]['Close']

    if VIX3M_TICKER in data:
        vix3m = data[VIX3M_TICKER]['Close']
        # Align indices
        common_idx = vix.index.intersection(vix3m.index)
        vix = vix.loc[common_idx]
        vix3m = vix3m.loc[common_idx]
        contango = (vix3m - vix) / vix
        print(f"\n  Using real VIX3M term structure. Contango mean={contango.mean():.4f}")
    else:
        # Proxy: VIX relative to its own 20-day MA
        vix_ma20 = vix.rolling(20).mean()
        contango = (vix_ma20 - vix) / vix
        print(f"\n  Using VIX-MA20 proxy for term structure. Proxy mean={contango.mean():.4f}")

    return contango


def compute_signals(data, contango_slope, rsi_os=RSI_OVERSOLD, rsi_ob=RSI_OVERBOUGHT):
    """
    Generate signals for each sector ETF.
    Returns DataFrame with columns: date, ticker, signal, rsi, contango, fwd_return
    """
    spy_close = data[BENCHMARK]['Close']
    spy_ret = spy_close.pct_change()

    records = []

    for ticker in SECTOR_ETFS:
        if ticker not in data:
            continue

        close = data[ticker]['Close']
        rsi = compute_rsi(close, RSI_PERIOD)

        # Forward 5-day return
        fwd_ret = close.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)

        # SPY forward return for regime classification
        spy_fwd = spy_close.pct_change(1)  # daily for regime

        # Align all series
        common_idx = close.index.intersection(contango_slope.index).intersection(rsi.dropna().index)

        for dt in common_idx:
            if dt not in fwd_ret.index or pd.isna(fwd_ret.loc[dt]):
                continue
            if pd.isna(rsi.loc[dt]) or pd.isna(contango_slope.loc[dt]):
                continue

            # Regime: green day = SPY up, red day = SPY down
            regime = 'green' if (dt in spy_ret.index and spy_ret.loc[dt] > 0) else 'red'

            records.append({
                'date': dt,
                'ticker': ticker,
                'rsi': rsi.loc[dt],
                'contango': contango_slope.loc[dt],
                'fwd_return': fwd_ret.loc[dt],
                'regime': regime
            })

    df = pd.DataFrame(records)
    print(f"\n  Total observation rows: {len(df)}")
    return df


def walk_forward_backtest(signals_df, rsi_os=RSI_OVERSOLD, rsi_ob=RSI_OVERBOUGHT):
    """
    Walk-forward sliding window backtest.

    In each 60-day training window, find the optimal contango threshold
    that maximizes Sharpe when combined with RSI oversold.
    Then apply to next day's signals.
    """
    signals_df = signals_df.sort_values('date').reset_index(drop=True)
    dates = sorted(signals_df['date'].unique())

    if len(dates) < TRAIN_WINDOW + 20:
        print("ERROR: Not enough dates for walk-forward")
        return pd.DataFrame()

    all_trades = []

    # Candidate contango thresholds to search over
    contango_thresholds = np.arange(0.02, 0.20, 0.01)

    for i in range(TRAIN_WINDOW, len(dates)):
        test_date = dates[i]
        train_start = dates[max(0, i - TRAIN_WINDOW)]
        train_end = dates[i - 1]

        # Training window
        train = signals_df[(signals_df['date'] >= train_start) & (signals_df['date'] <= train_end)]

        # Test day
        test = signals_df[signals_df['date'] == test_date]

        if len(train) < 50 or len(test) == 0:
            continue

        # Find best contango threshold in training window
        best_sharpe = -999
        best_thresh = 0.05  # default

        for thresh in contango_thresholds:
            # Long signals: contango > thresh AND RSI < oversold
            longs = train[(train['contango'] > thresh) & (train['rsi'] < rsi_os)]

            if len(longs) < 5:
                continue

            ret_mean = longs['fwd_return'].mean()
            ret_std = longs['fwd_return'].std()

            if ret_std > 0:
                sharpe = ret_mean / ret_std * np.sqrt(252 / HOLD_DAYS)
            else:
                sharpe = 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_thresh = thresh

        # Apply to test day
        # LONG: contango > best_thresh AND RSI < oversold
        long_signals = test[(test['contango'] > best_thresh) & (test['rsi'] < rsi_os)]

        # SHORT (contrarian sell): backwardation AND RSI > overbought
        short_signals = test[(test['contango'] < -0.02) & (test['rsi'] > rsi_ob)]

        for _, row in long_signals.iterrows():
            all_trades.append({
                'date': row['date'],
                'ticker': row['ticker'],
                'direction': 'LONG',
                'rsi': row['rsi'],
                'contango': row['contango'],
                'return': row['fwd_return'],
                'regime': row['regime'],
                'threshold_used': best_thresh
            })

        for _, row in short_signals.iterrows():
            all_trades.append({
                'date': row['date'],
                'ticker': row['ticker'],
                'direction': 'SHORT',
                'rsi': row['rsi'],
                'contango': row['contango'],
                'return': -row['fwd_return'],  # short => negate
                'regime': row['regime'],
                'threshold_used': best_thresh
            })

    trades = pd.DataFrame(all_trades)
    print(f"\n  Walk-forward generated {len(trades)} trades")
    return trades


def compute_metrics(returns, dates=None):
    """Compute trading metrics from a series of trade returns."""
    if len(returns) == 0:
        return {}

    returns = returns.reset_index(drop=True)
    mean_ret = returns.mean()
    std_ret = returns.std()

    # Annualize (trades are 5-day holds)
    trades_per_year = 252 / HOLD_DAYS
    ann_ret = mean_ret * trades_per_year
    ann_std = std_ret * np.sqrt(trades_per_year)

    sharpe = ann_ret / ann_std if ann_std > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(trades_per_year) if len(downside) > 0 else 1e-6
    sortino = ann_ret / downside_std if downside_std > 0 else 0

    # Win rate
    wr = (returns > 0).mean()

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown (cumulative) - sort by date if available
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    return {
        'n_trades': len(returns),
        'mean_return': mean_ret,
        'ann_return': ann_ret,
        'ann_std': ann_std,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': wr,
        'profit_factor': pf,
        'max_drawdown': max_dd,
        'total_return': (1 + returns).prod() - 1
    }


def permutation_test(trade_returns, all_returns, n_perms=N_PERMUTATIONS):
    """
    Permutation test for statistical significance.
    Randomly sample len(trade_returns) from all_returns (the full universe of
    possible returns), and compare mean to the observed mean.
    This tests whether the signal selects above-average returns.
    """
    n_trades = len(trade_returns)
    if n_trades < 10:
        return 1.0

    observed_mean = trade_returns.mean()
    all_vals = all_returns.values

    count_exceed = 0
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        # Random sample of same size from all possible returns
        idx = rng.choice(len(all_vals), size=n_trades, replace=True)
        perm_mean = all_vals[idx].mean()
        if perm_mean >= observed_mean:
            count_exceed += 1

    return count_exceed / n_perms


def regime_analysis(trades_df):
    """Stratify performance by regime (green/red SPY days)."""
    results = {}
    for regime in ['green', 'red']:
        subset = trades_df[trades_df['regime'] == regime]
        if len(subset) > 0:
            results[regime] = compute_metrics(subset['return'])
        else:
            results[regime] = {'sharpe': 0, 'n_trades': 0}

    # Regime gap
    s_green = results.get('green', {}).get('sharpe', 0)
    s_red = results.get('red', {}).get('sharpe', 0)
    max_abs = max(abs(s_green), abs(s_red), 1e-6)
    regime_gap = abs(s_green - s_red) / max_abs

    return results, regime_gap


def five_gate_validation(metrics, regime_gap, perm_p):
    """Apply 5-gate validation."""
    gates = {}
    gates['sharpe_gt_0.5'] = metrics.get('sharpe', 0) > 0.5
    gates['regime_gap_lt_0.5'] = regime_gap < 0.5
    gates['perm_p_lt_0.05'] = perm_p < 0.05
    gates['max_dd_gt_neg30pct'] = metrics.get('max_drawdown', -1) > -0.30
    gates['trades_gt_30'] = metrics.get('n_trades', 0) > MIN_TRADES

    gates['ALL_PASS'] = all(gates.values())
    return gates


def print_results(metrics, regime_results, regime_gap, perm_p, gates, trades_df):
    """Print comprehensive results."""
    print("\n" + "=" * 70)
    print("VIX CONTANGO + SECTOR OVERSOLD CONTRARIAN BACKTEST RESULTS")
    print("=" * 70)

    print(f"\n{'OVERALL PERFORMANCE':=^50}")
    print(f"  Total Trades:      {metrics.get('n_trades', 0)}")
    print(f"  Annualized Return: {metrics.get('ann_return', 0)*100:.2f}%")
    print(f"  Annualized Std:    {metrics.get('ann_std', 0)*100:.2f}%")
    print(f"  Sharpe Ratio:      {metrics.get('sharpe', 0):.3f}")
    print(f"  Sortino Ratio:     {metrics.get('sortino', 0):.3f}")
    print(f"  Win Rate:          {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor:     {metrics.get('profit_factor', 0):.3f}")
    print(f"  Max Drawdown:      {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"  Total Return:      {metrics.get('total_return', 0)*100:.2f}%")

    print(f"\n{'PER-REGIME STRATIFICATION':=^50}")
    for regime in ['green', 'red']:
        r = regime_results.get(regime, {})
        print(f"\n  {regime.upper()} days (SPY {'up' if regime=='green' else 'down'}):")
        print(f"    Trades:  {r.get('n_trades', 0)}")
        print(f"    Sharpe:  {r.get('sharpe', 0):.3f}")
        print(f"    Sortino: {r.get('sortino', 0):.3f}")
        print(f"    WR:      {r.get('win_rate', 0)*100:.1f}%")
        print(f"    PF:      {r.get('profit_factor', 0):.3f}")

    print(f"\n  Regime Gap: {regime_gap:.3f} (threshold: < 0.50)")

    print(f"\n{'STATISTICAL SIGNIFICANCE':=^50}")
    print(f"  Permutation p-value: {perm_p:.4f} (threshold: < 0.05)")
    print(f"  Permutations run:    {N_PERMUTATIONS}")

    print(f"\n{'5-GATE VALIDATION':=^50}")
    for gate, passed in gates.items():
        if gate == 'ALL_PASS':
            continue
        status = 'PASS' if passed else 'FAIL'
        print(f"  [{status}] {gate}")

    overall = 'PASS - SIGNAL IS VIABLE' if gates['ALL_PASS'] else 'FAIL - SIGNAL REJECTED'
    print(f"\n  >>> OVERALL: {overall} <<<")

    # Breakdown by direction
    if len(trades_df) > 0:
        print(f"\n{'BY DIRECTION':=^50}")
        for direction in ['LONG', 'SHORT']:
            subset = trades_df[trades_df['direction'] == direction]
            if len(subset) > 0:
                m = compute_metrics(subset['return'])
                print(f"\n  {direction}:")
                print(f"    Trades: {m['n_trades']}, Sharpe: {m['sharpe']:.3f}, WR: {m['win_rate']*100:.1f}%, PF: {m['profit_factor']:.3f}")

    # Top/bottom sectors
    if len(trades_df) > 0:
        print(f"\n{'BY SECTOR':=^50}")
        for ticker in sorted(trades_df['ticker'].unique()):
            subset = trades_df[trades_df['ticker'] == ticker]
            if len(subset) >= 3:
                m = compute_metrics(subset['return'])
                print(f"  {ticker:5s}: {m['n_trades']:3d} trades, Sharpe={m['sharpe']:+.3f}, WR={m['win_rate']*100:.1f}%, PF={m['profit_factor']:.3f}")

    # Yearly breakdown
    if len(trades_df) > 0:
        print(f"\n{'BY YEAR':=^50}")
        trades_df['year'] = trades_df['date'].dt.year
        for year in sorted(trades_df['year'].unique()):
            subset = trades_df[trades_df['year'] == year]
            if len(subset) >= 3:
                m = compute_metrics(subset['return'])
                print(f"  {year}: {m['n_trades']:3d} trades, Sharpe={m['sharpe']:+.3f}, WR={m['win_rate']*100:.1f}%, Ret={m['total_return']*100:+.2f}%")


def main():
    print("=" * 70)
    print("VIX TERM STRUCTURE CONTANGO + SECTOR OVERSOLD CONTRARIAN BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Hold: {HOLD_DAYS} days | RSI period: {RSI_PERIOD}")
    print(f"Oversold: RSI<{RSI_OVERSOLD} | Overbought: RSI>{RSI_OVERBOUGHT}")
    print(f"Walk-forward: {TRAIN_WINDOW}-day sliding window")
    print("=" * 70)

    # 1. Download data
    data = download_data()

    if VIX_TICKER not in data:
        print("FATAL: VIX data not available")
        sys.exit(1)

    # 2. Compute VIX term structure slope
    contango_slope = compute_contango_slope(data)

    # 3. Build signal observations
    signals_df = compute_signals(data, contango_slope)

    if len(signals_df) == 0:
        print("FATAL: No signal observations generated")
        sys.exit(1)

    # Summary stats on the signal components
    print(f"\n  Contango slope stats: mean={contango_slope.mean():.4f}, std={contango_slope.std():.4f}")
    print(f"  Pct oversold obs (RSI<30): {(signals_df['rsi'] < RSI_OVERSOLD).mean()*100:.2f}%")
    print(f"  Pct overbought obs (RSI>70): {(signals_df['rsi'] > RSI_OVERBOUGHT).mean()*100:.2f}%")

    # How often both conditions align
    both_long = ((signals_df['contango'] > 0.05) & (signals_df['rsi'] < RSI_OVERSOLD))
    print(f"  Pct LONG signal (contango>0.05 & RSI<30): {both_long.mean()*100:.3f}%")

    both_short = ((signals_df['contango'] < -0.02) & (signals_df['rsi'] > RSI_OVERBOUGHT))
    print(f"  Pct SHORT signal (contango<-0.02 & RSI>70): {both_short.mean()*100:.3f}%")

    # 4. Walk-forward backtest (with adaptive RSI relaxation if too few trades)
    rsi_os_used = RSI_OVERSOLD
    rsi_ob_used = RSI_OVERBOUGHT
    trades_df = walk_forward_backtest(signals_df, rsi_os=rsi_os_used, rsi_ob=rsi_ob_used)

    if len(trades_df) < 5:
        print(f"\nOnly {len(trades_df)} trades generated. Relaxing RSI thresholds...")
        for rsi_os_try, rsi_ob_try in [(35, 65), (40, 60)]:
            print(f"\n--- Retrying with RSI<{rsi_os_try} for long, RSI>{rsi_ob_try} for short ---")
            signals_df = compute_signals(data, contango_slope, rsi_os=rsi_os_try, rsi_ob=rsi_ob_try)
            trades_df = walk_forward_backtest(signals_df, rsi_os=rsi_os_try, rsi_ob=rsi_ob_try)
            rsi_os_used = rsi_os_try
            rsi_ob_used = rsi_ob_try
            if len(trades_df) >= 5:
                break

    if len(trades_df) < 5:
        print(f"\nFATAL: Cannot generate enough trades even with relaxed thresholds.")
        sys.exit(1)

    print(f"\n  Final RSI thresholds used: oversold<{rsi_os_used}, overbought>{rsi_ob_used}")

    # Full universe of 5-day returns (for permutation test baseline)
    all_fwd_returns = signals_df['fwd_return'].dropna()

    # Sort trades by date for correct cumulative calculations
    trades_df = trades_df.sort_values('date').reset_index(drop=True)

    # 5. Compute metrics for ALL trades
    metrics = compute_metrics(trades_df['return'])
    regime_results, regime_gap = regime_analysis(trades_df)
    print("\n  Running permutation test (1000 permutations) - ALL trades...")
    perm_p = permutation_test(trades_df['return'], all_fwd_returns)
    gates = five_gate_validation(metrics, regime_gap, perm_p)
    print_results(metrics, regime_results, regime_gap, perm_p, gates, trades_df)

    # 6. LONG-ONLY analysis (the interesting part based on initial results)
    long_trades = trades_df[trades_df['direction'] == 'LONG'].copy().sort_values('date').reset_index(drop=True)
    if len(long_trades) >= 10:
        print("\n" + "=" * 70)
        print("LONG-ONLY ANALYSIS (Contango + RSI Oversold)")
        print("=" * 70)
        long_metrics = compute_metrics(long_trades['return'])
        long_regime, long_regime_gap = regime_analysis(long_trades)
        print("\n  Running permutation test (1000 permutations) - LONG only...")
        long_perm_p = permutation_test(long_trades['return'], all_fwd_returns)
        long_gates = five_gate_validation(long_metrics, long_regime_gap, long_perm_p)
        print_results(long_metrics, long_regime, long_regime_gap, long_perm_p, long_gates, long_trades)

    # 7. SHORT-ONLY analysis
    short_trades = trades_df[trades_df['direction'] == 'SHORT'].copy().sort_values('date').reset_index(drop=True)
    if len(short_trades) >= 10:
        print("\n" + "=" * 70)
        print("SHORT-ONLY ANALYSIS (Backwardation + RSI Overbought)")
        print("=" * 70)
        short_metrics = compute_metrics(short_trades['return'])
        short_regime, short_regime_gap = regime_analysis(short_trades)
        print("\n  Running permutation test (1000 permutations) - SHORT only...")
        short_perm_p = permutation_test(short_trades['return'], all_fwd_returns)
        short_gates = five_gate_validation(short_metrics, short_regime_gap, short_perm_p)
        print_results(short_metrics, short_regime, short_regime_gap, short_perm_p, short_gates, short_trades)

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
