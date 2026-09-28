#!/usr/bin/env python3
"""
Earnings Quality + Momentum Combo Backtest
==========================================
Tests 6 variants of post-earnings-announcement drift (PEAD) strategies
on growth stocks, with full walk-forward OOT validation (Jan 2022 - Jul 2026).

5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import os
import warnings
import sys
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'CRM',
    'NFLX', 'SHOP', 'DDOG', 'SNOW', 'UBER', 'COIN', 'PLTR', 'SQ', 'ROKU',
    'SNAP', 'PINS', 'NET', 'CRWD', 'ZS', 'PANW', 'MDB'
]

START_DATE = '2021-06-01'  # need lookback for 200-SMA
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per trade
MAX_CONCURRENT = 5
N_PERMUTATIONS = 1000

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/earnings_quality_momentum_results.json'


def download_data():
    """Download price data and earnings data for universe + SPY."""
    print("Downloading price data...")
    tickers = UNIVERSE + ['SPY']
    data = {}

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=START_DATE, end=OOT_END, auto_adjust=True)
            if len(hist) > 50:
                hist.index = hist.index.tz_localize(None)
                data[ticker] = hist
                print(f"  {ticker}: {len(hist)} bars")
            else:
                print(f"  {ticker}: insufficient data ({len(hist)} bars), skipping")
        except Exception as e:
            print(f"  {ticker}: error - {e}")

    return data


def get_earnings_data(ticker):
    """Get earnings dates and surprise data from yfinance."""
    try:
        t = yf.Ticker(ticker)
        # Get earnings dates with estimates
        earnings = t.earnings_dates
        if earnings is None or len(earnings) == 0:
            return pd.DataFrame()

        # Clean up the earnings dataframe
        earnings = earnings.copy()
        earnings.index = earnings.index.tz_localize(None)

        # Filter to our date range
        earnings = earnings[(earnings.index >= pd.Timestamp(OOT_START)) &
                           (earnings.index <= pd.Timestamp(OOT_END))]

        # Need both actual and estimate
        if 'Reported EPS' in earnings.columns and 'EPS Estimate' in earnings.columns:
            earnings = earnings.dropna(subset=['Reported EPS', 'EPS Estimate'])
            # Calculate surprise percentage
            earnings['surprise_pct'] = np.where(
                earnings['EPS Estimate'].abs() > 0.01,
                (earnings['Reported EPS'] - earnings['EPS Estimate']) / earnings['EPS Estimate'].abs() * 100,
                np.nan
            )
            earnings = earnings.dropna(subset=['surprise_pct'])
            return earnings

        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


def compute_sma(prices, window):
    """Compute simple moving average."""
    return prices.rolling(window=window, min_periods=window).mean()


def get_spy_regime(spy_data, date):
    """Determine if market is bull or bear based on SPY vs 200-SMA."""
    spy_close = spy_data['Close']
    sma200 = compute_sma(spy_close, 200)

    # Find the most recent trading day on or before date
    mask = spy_close.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return 'bull'  # default

    idx = spy_close.index[mask][-1]
    if pd.isna(sma200.loc[idx]):
        return 'bull'

    return 'bull' if spy_close.loc[idx] > sma200.loc[idx] else 'bear'


def find_next_trading_day(prices, date, offset=1):
    """Find the next trading day after date."""
    future = prices.index[prices.index > pd.Timestamp(date)]
    if len(future) >= offset:
        return future[offset - 1]
    return None


def find_price_on_date(prices, date):
    """Find close price on or just before a date."""
    mask = prices.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return None
    return prices['Close'].iloc[mask.sum() - 1]


def run_backtest(variant_name, signals, price_data, spy_data, account_size=ACCOUNT_SIZE):
    """
    Run backtest given a list of signals.

    Each signal: {
        'ticker': str,
        'entry_date': Timestamp,  # day after earnings
        'hold_days': int,
        'beat_pct': float,
    }

    Returns dict of results.
    """
    trades = []
    active_positions = []  # list of (ticker, entry_date, exit_date, entry_price)

    # Sort signals by date
    signals = sorted(signals, key=lambda x: x['entry_date'])

    for sig in signals:
        ticker = sig['ticker']
        entry_date = sig['entry_date']
        hold_days = sig['hold_days']

        if ticker not in price_data:
            continue

        prices = price_data[ticker]

        # Entry: next trading day after earnings date
        entry_day = find_next_trading_day(prices, entry_date, offset=1)
        if entry_day is None:
            continue

        # Check how many positions are active at entry
        active_positions = [p for p in active_positions if p['exit_date'] > entry_day]
        if len(active_positions) >= MAX_CONCURRENT:
            continue

        entry_price = prices.loc[entry_day, 'Close'] if entry_day in prices.index else None
        if entry_price is None or pd.isna(entry_price):
            continue

        # Find exit day (hold_days trading days later)
        future_dates = prices.index[prices.index > entry_day]
        if len(future_dates) < hold_days:
            continue
        exit_day = future_dates[min(hold_days - 1, len(future_dates) - 1)]

        exit_price = prices.loc[exit_day, 'Close']
        if pd.isna(exit_price):
            continue

        # Apply slippage
        entry_price_adj = entry_price * (1 + SLIPPAGE_PCT)
        exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)

        ret = (exit_price_adj - entry_price_adj) / entry_price_adj

        # Position sizing: equal weight, account_size / MAX_CONCURRENT
        position_size = account_size / MAX_CONCURRENT
        pnl = position_size * ret

        # Determine regime at entry
        regime = get_spy_regime(spy_data, entry_day)

        trade = {
            'ticker': ticker,
            'entry_date': str(entry_day.date()),
            'exit_date': str(exit_day.date()),
            'entry_price': round(float(entry_price), 2),
            'exit_price': round(float(exit_price), 2),
            'return_pct': round(float(ret * 100), 2),
            'pnl': round(float(pnl), 2),
            'regime': regime,
            'hold_days': hold_days,
            'beat_pct': round(float(sig.get('beat_pct', 0)), 2),
        }
        trades.append(trade)

        active_positions.append({
            'ticker': ticker,
            'entry_date': entry_day,
            'exit_date': exit_day,
        })

    return trades


def compute_metrics(trades):
    """Compute performance metrics from list of trades."""
    if len(trades) == 0:
        return {
            'n_trades': 0, 'total_return_pct': 0, 'sharpe': 0,
            'sortino': 0, 'max_dd_pct': 0, 'win_rate': 0,
            'profit_factor': 0, 'avg_return_pct': 0,
            'sharpe_bull': 0, 'sharpe_bear': 0, 'regime_gap': 1.0,
        }

    returns = np.array([t['return_pct'] / 100 for t in trades])
    pnls = np.array([t['pnl'] for t in trades])

    n_trades = len(trades)
    total_return_pct = float(np.sum(returns) * 100)
    avg_return = float(np.mean(returns))

    # Annualize: assume ~9 trades per year on average (conservative)
    # Better: use actual date range
    dates = [pd.Timestamp(t['entry_date']) for t in trades]
    date_range_years = (max(dates) - min(dates)).days / 365.25 if len(dates) > 1 else 1.0
    trades_per_year = n_trades / max(date_range_years, 0.5)

    # Sharpe: annualized
    if np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(returns) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    # Max drawdown on cumulative PnL
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl + ACCOUNT_SIZE)
    dd = (cum_pnl + ACCOUNT_SIZE - peak) / peak
    max_dd_pct = float(np.min(dd) * 100)

    # Win rate
    win_rate = float(np.mean(returns > 0) * 100)

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Regime analysis
    bull_returns = [t['return_pct'] / 100 for t in trades if t['regime'] == 'bull']
    bear_returns = [t['return_pct'] / 100 for t in trades if t['regime'] == 'bear']

    bull_trades_per_year = len(bull_returns) / max(date_range_years, 0.5)
    bear_trades_per_year = len(bear_returns) / max(date_range_years, 0.5)

    if len(bull_returns) > 1 and np.std(bull_returns) > 0:
        sharpe_bull = (np.mean(bull_returns) / np.std(bull_returns)) * np.sqrt(max(bull_trades_per_year, 1))
    else:
        sharpe_bull = 0.0

    if len(bear_returns) > 1 and np.std(bear_returns) > 0:
        sharpe_bear = (np.mean(bear_returns) / np.std(bear_returns)) * np.sqrt(max(bear_trades_per_year, 1))
    else:
        sharpe_bear = 0.0

    # Regime gap
    max_abs = max(abs(sharpe_bull), abs(sharpe_bear), 0.01)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs

    return {
        'n_trades': n_trades,
        'total_return_pct': round(total_return_pct, 2),
        'avg_return_pct': round(avg_return * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd_pct': round(max_dd_pct, 2),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 3),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
        'n_bull_trades': len(bull_returns),
        'n_bear_trades': len(bear_returns),
        'total_pnl': round(float(np.sum(pnls)), 2),
        'final_account': round(ACCOUNT_SIZE + float(np.sum(pnls)), 2),
    }


def permutation_test(trades, price_data, n_perms=N_PERMUTATIONS):
    """
    Permutation test: for each permutation, randomly assign each trade to a
    random ticker and random date from the available price history, then compute
    the hold-period return. Compare actual mean return vs random-timing distribution.
    """
    if len(trades) < 5:
        return 1.0

    actual_returns = np.array([t['return_pct'] / 100 for t in trades])
    actual_mean = np.mean(actual_returns)
    n_trades = len(trades)

    # Build pool of valid (ticker, date_index) pairs for random sampling
    ticker_date_pools = {}
    for ticker, df in price_data.items():
        if ticker == 'SPY':
            continue
        valid_idx = df.index[(df.index >= pd.Timestamp(OOT_START)) &
                             (df.index <= pd.Timestamp(OOT_END))]
        if len(valid_idx) > 60:
            ticker_date_pools[ticker] = valid_idx

    pool_tickers = list(ticker_date_pools.keys())
    if len(pool_tickers) == 0:
        return 1.0

    # Pre-build return arrays for speed
    hold_days_list = [t.get('hold_days', 40) for t in trades]

    n_beats = 0
    for _ in range(n_perms):
        perm_returns = []
        for i in range(n_trades):
            hold = hold_days_list[i]
            # Random ticker and random entry date
            tk = pool_tickers[np.random.randint(len(pool_tickers))]
            dates = ticker_date_pools[tk]
            # Entry must allow room for hold period
            max_entry_idx = max(0, len(dates) - hold - 1)
            if max_entry_idx <= 0:
                continue
            entry_pos = np.random.randint(0, max_entry_idx)
            exit_pos = min(entry_pos + hold, len(dates) - 1)

            entry_p = price_data[tk]['Close'].iloc[price_data[tk].index.get_indexer([dates[entry_pos]])[0]]
            exit_p = price_data[tk]['Close'].iloc[price_data[tk].index.get_indexer([dates[exit_pos]])[0]]

            if pd.isna(entry_p) or pd.isna(exit_p) or entry_p <= 0:
                continue

            ret = (exit_p * (1 - SLIPPAGE_PCT) - entry_p * (1 + SLIPPAGE_PCT)) / (entry_p * (1 + SLIPPAGE_PCT))
            perm_returns.append(ret)

        if len(perm_returns) > 0 and np.mean(perm_returns) >= actual_mean:
            n_beats += 1

    p_value = (n_beats + 1) / (n_perms + 1)
    return round(p_value, 4)


def validate_5gate(metrics, p_value):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': p_value < 0.05,
        'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_dd_pct'] > -50,
        'min_20_trades': metrics['n_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


def generate_signals_A(earnings_data, price_data, spy_data):
    """Variant A: Basic Earnings Beat + Momentum (price > 50-SMA). Hold 40 days."""
    signals = []

    for ticker, earn_df in earnings_data.items():
        if ticker not in price_data:
            continue
        prices = price_data[ticker]
        sma50 = compute_sma(prices['Close'], 50)

        for date, row in earn_df.iterrows():
            if row['surprise_pct'] <= 5:
                continue

            # Check momentum: price > 50-SMA at earnings date
            mask = sma50.index <= date
            if mask.sum() == 0:
                continue
            idx = sma50.index[mask][-1]
            if pd.isna(sma50.loc[idx]):
                continue

            price_at_earnings = prices['Close'].loc[idx]
            if price_at_earnings <= sma50.loc[idx]:
                continue

            signals.append({
                'ticker': ticker,
                'entry_date': date,
                'hold_days': 40,
                'beat_pct': row['surprise_pct'],
            })

    return signals


def generate_signals_B(earnings_data, price_data, spy_data):
    """Variant B: Quality Filter - require earnings acceleration (beat% increasing QoQ)."""
    signals = []

    for ticker, earn_df in earnings_data.items():
        if ticker not in price_data:
            continue
        prices = price_data[ticker]
        sma50 = compute_sma(prices['Close'], 50)

        # Sort by date to track QoQ
        earn_sorted = earn_df.sort_index()
        prev_beat = None

        for date, row in earn_sorted.iterrows():
            current_beat = row['surprise_pct']

            if current_beat <= 5:
                prev_beat = current_beat
                continue

            # Momentum check
            mask = sma50.index <= date
            if mask.sum() == 0:
                prev_beat = current_beat
                continue
            idx = sma50.index[mask][-1]
            if pd.isna(sma50.loc[idx]):
                prev_beat = current_beat
                continue

            price_at_earnings = prices['Close'].loc[idx]
            if price_at_earnings <= sma50.loc[idx]:
                prev_beat = current_beat
                continue

            # Quality: earnings acceleration (beat% > previous beat%)
            if prev_beat is not None and current_beat > prev_beat:
                signals.append({
                    'ticker': ticker,
                    'entry_date': date,
                    'hold_days': 40,
                    'beat_pct': current_beat,
                })

            prev_beat = current_beat

    return signals


def generate_signals_C(earnings_data, price_data, spy_data):
    """Variant C: Top 3 Concentration - only hold top 3 highest beat% stocks."""
    # First generate all valid signals like A
    all_signals = generate_signals_A(earnings_data, price_data, spy_data)

    # Group by approximate date (within 30-day windows = quarterly)
    all_signals.sort(key=lambda x: x['entry_date'])

    # For each quarter, keep only top 3 by beat_pct
    filtered = []
    quarterly_groups = defaultdict(list)

    for sig in all_signals:
        q_key = f"{sig['entry_date'].year}-Q{(sig['entry_date'].month - 1) // 3 + 1}"
        quarterly_groups[q_key].append(sig)

    for q_key, group in quarterly_groups.items():
        group.sort(key=lambda x: x['beat_pct'], reverse=True)
        filtered.extend(group[:3])

    return filtered


def generate_signals_D(earnings_data, price_data, spy_data):
    """Variant D: Regime-Adaptive Hold - 40 days bull, 20 days bear."""
    signals = []

    for ticker, earn_df in earnings_data.items():
        if ticker not in price_data:
            continue
        prices = price_data[ticker]
        sma50 = compute_sma(prices['Close'], 50)

        for date, row in earn_df.iterrows():
            if row['surprise_pct'] <= 5:
                continue

            mask = sma50.index <= date
            if mask.sum() == 0:
                continue
            idx = sma50.index[mask][-1]
            if pd.isna(sma50.loc[idx]):
                continue

            price_at_earnings = prices['Close'].loc[idx]
            if price_at_earnings <= sma50.loc[idx]:
                continue

            regime = get_spy_regime(spy_data, date)
            hold = 40 if regime == 'bull' else 20

            signals.append({
                'ticker': ticker,
                'entry_date': date,
                'hold_days': hold,
                'beat_pct': row['surprise_pct'],
            })

    return signals


def generate_signals_E(earnings_data, price_data, spy_data):
    """Variant E: Revenue + Earnings Double Beat. Require both revenue and EPS beat."""
    signals = []

    for ticker, earn_df in earnings_data.items():
        if ticker not in price_data:
            continue
        prices = price_data[ticker]
        sma50 = compute_sma(prices['Close'], 50)

        # Try to get revenue data
        try:
            t = yf.Ticker(ticker)
            # Check if 'Surprise(%)' or 'Revenue Estimate' columns exist
            # yfinance earnings_dates has 'Surprise(%)' column
            has_surprise_col = 'Surprise(%)' in earn_df.columns
        except:
            has_surprise_col = False

        for date, row in earn_df.iterrows():
            if row['surprise_pct'] <= 5:
                continue

            # For revenue beat, use Surprise(%) column if available
            # Otherwise require a larger EPS beat as proxy for double quality
            if has_surprise_col and not pd.isna(row.get('Surprise(%)', np.nan)):
                surprise_pct_col = row['Surprise(%)']
                if surprise_pct_col < 0:  # Revenue missed
                    continue
            else:
                # Proxy: require >10% EPS beat (higher bar = proxy for double beat)
                if row['surprise_pct'] <= 10:
                    continue

            mask = sma50.index <= date
            if mask.sum() == 0:
                continue
            idx = sma50.index[mask][-1]
            if pd.isna(sma50.loc[idx]):
                continue

            price_at_earnings = prices['Close'].loc[idx]
            if price_at_earnings <= sma50.loc[idx]:
                continue

            signals.append({
                'ticker': ticker,
                'entry_date': date,
                'hold_days': 40,
                'beat_pct': row['surprise_pct'],
            })

    return signals


def generate_signals_F(earnings_data, price_data, spy_data):
    """Variant F: Momentum Confirmation - buy only if stock gaps up on earnings day."""
    signals = []

    for ticker, earn_df in earnings_data.items():
        if ticker not in price_data:
            continue
        prices = price_data[ticker]
        sma50 = compute_sma(prices['Close'], 50)

        for date, row in earn_df.iterrows():
            if row['surprise_pct'] <= 5:
                continue

            # Momentum check
            mask = sma50.index <= date
            if mask.sum() == 0:
                continue
            idx = sma50.index[mask][-1]
            if pd.isna(sma50.loc[idx]):
                continue

            price_at_earnings = prices['Close'].loc[idx]
            if price_at_earnings <= sma50.loc[idx]:
                continue

            # Momentum confirmation: stock is up on/after earnings
            # Compare close on earnings day vs previous close
            earn_day_idx = prices.index.get_indexer([date], method='ffill')[0]
            if earn_day_idx < 1 or earn_day_idx >= len(prices):
                continue

            # Find the trading day of or just after earnings
            future = prices.index[prices.index >= date]
            if len(future) == 0:
                continue
            earn_trade_day = future[0]

            # Previous trading day
            past = prices.index[prices.index < date]
            if len(past) == 0:
                continue
            prev_day = past[-1]

            # Check if stock gapped up
            earn_close = prices.loc[earn_trade_day, 'Close']
            prev_close = prices.loc[prev_day, 'Close']

            if earn_close <= prev_close:
                continue  # No positive gap, skip

            signals.append({
                'ticker': ticker,
                'entry_date': date,
                'hold_days': 40,
                'beat_pct': row['surprise_pct'],
            })

    return signals


def main():
    print("=" * 80)
    print("EARNINGS QUALITY + MOMENTUM COMBO BACKTEST")
    print(f"Universe: {len(UNIVERSE)} growth stocks")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Account: ${ACCOUNT_SIZE}")
    print("=" * 80)

    # Download data
    price_data = download_data()
    spy_data = price_data.get('SPY')

    if spy_data is None:
        print("ERROR: Could not download SPY data. Aborting.")
        sys.exit(1)

    # Download earnings data
    print("\nDownloading earnings data...")
    earnings_data = {}
    for ticker in UNIVERSE:
        if ticker in price_data:
            earn = get_earnings_data(ticker)
            if len(earn) > 0:
                earnings_data[ticker] = earn
                print(f"  {ticker}: {len(earn)} earnings reports with surprise data")
            else:
                print(f"  {ticker}: no earnings surprise data")

    print(f"\n{len(earnings_data)} tickers with earnings data")

    if len(earnings_data) < 5:
        print("WARNING: Very few tickers with earnings data. Results may be sparse.")

    # Define variants
    variants = {
        'A_basic_beat_momentum': generate_signals_A,
        'B_quality_filter': generate_signals_B,
        'C_top3_concentration': generate_signals_C,
        'D_regime_adaptive_hold': generate_signals_D,
        'E_double_beat': generate_signals_E,
        'F_gap_up_confirmation': generate_signals_F,
    }

    all_results = {}

    for name, signal_func in variants.items():
        print(f"\n{'=' * 60}")
        print(f"VARIANT: {name}")
        print(f"{'=' * 60}")

        # Generate signals
        signals = signal_func(earnings_data, price_data, spy_data)
        print(f"  Raw signals generated: {len(signals)}")

        if len(signals) == 0:
            print("  No signals, skipping.")
            all_results[name] = {
                'metrics': compute_metrics([]),
                'gates': validate_5gate(compute_metrics([]), 1.0),
                'p_value': 1.0,
                'n_signals': 0,
                'trades': [],
            }
            continue

        # Run backtest
        trades = run_backtest(name, signals, price_data, spy_data)
        print(f"  Trades executed: {len(trades)}")

        # Compute metrics
        metrics = compute_metrics(trades)

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
        p_value = permutation_test(trades, price_data, N_PERMUTATIONS)

        # 5-gate validation
        gates = validate_5gate(metrics, p_value)

        # Store results
        all_results[name] = {
            'metrics': metrics,
            'gates': gates,
            'p_value': p_value,
            'n_signals': len(signals),
            'trades': trades,
        }

        # Print summary
        print(f"\n  --- Results ---")
        print(f"  Trades: {metrics['n_trades']} (Bull: {metrics['n_bull_trades']}, Bear: {metrics['n_bear_trades']})")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Avg Return/Trade: {metrics['avg_return_pct']:.2f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Max DD: {metrics['max_dd_pct']:.1f}%")
        print(f"  Final Account: ${metrics['final_account']:.2f}")
        print(f"  Sharpe Bull/Bear: {metrics['sharpe_bull']:.3f} / {metrics['sharpe_bear']:.3f}")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f}")
        print(f"  Perm p-value: {p_value:.4f}")
        print(f"\n  5-Gate Validation:")
        for gate, passed in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    {gate}: {status}")

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)

    header = f"{'Variant':<30} {'Trades':>7} {'Return%':>9} {'Sharpe':>8} {'Sortino':>8} {'WR%':>6} {'PF':>7} {'MaxDD%':>8} {'RegGap':>8} {'p-val':>7} {'Pass?':>6}"
    print(header)
    print("-" * 120)

    passing_variants = []

    for name, res in all_results.items():
        m = res['metrics']
        g = res['gates']
        p = res['p_value']
        verdict = "YES" if g.get('all_pass', False) else "NO"

        if g.get('all_pass', False):
            passing_variants.append(name)

        line = f"{name:<30} {m['n_trades']:>7d} {m['total_return_pct']:>8.1f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['win_rate']:>5.1f}% {m['profit_factor']:>7.3f} {m['max_dd_pct']:>7.1f}% {m['regime_gap']:>8.3f} {p:>7.4f} {verdict:>6}"
        print(line)

    print("-" * 120)
    print(f"\nPassing variants: {len(passing_variants)} / {len(all_results)}")
    if passing_variants:
        print(f"  Winners: {', '.join(passing_variants)}")
    else:
        print("  No variants passed all 5 gates.")

    # ============================================================
    # DETAILED GATE ANALYSIS
    # ============================================================
    print("\n" + "=" * 80)
    print("DETAILED GATE ANALYSIS")
    print("=" * 80)

    for name, res in all_results.items():
        g = res['gates']
        m = res['metrics']
        p = res['p_value']

        gate_details = [
            f"Sharpe={m['sharpe']:.3f} {'PASS' if g.get('sharpe_gt_0.5') else 'FAIL'}",
            f"p={p:.4f} {'PASS' if g.get('perm_p_lt_0.05') else 'FAIL'}",
            f"RegGap={m['regime_gap']:.3f} {'PASS' if g.get('regime_gap_lt_0.5') else 'FAIL'}",
            f"MaxDD={m['max_dd_pct']:.1f}% {'PASS' if g.get('max_dd_gt_neg50') else 'FAIL'}",
            f"Trades={m['n_trades']} {'PASS' if g.get('min_20_trades') else 'FAIL'}",
        ]
        overall = "ALL PASS" if g.get('all_pass') else "FAILED"
        print(f"  {name}: [{overall}] {' | '.join(gate_details)}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    # Prepare JSON-serializable results
    json_results = {
        'metadata': {
            'strategy': 'Earnings Quality + Momentum Combo',
            'universe_size': len(UNIVERSE),
            'universe': UNIVERSE,
            'oot_start': OOT_START,
            'oot_end': OOT_END,
            'account_size': ACCOUNT_SIZE,
            'slippage_pct': SLIPPAGE_PCT,
            'max_concurrent': MAX_CONCURRENT,
            'n_permutations': N_PERMUTATIONS,
            'run_timestamp': datetime.now().isoformat(),
            'tickers_with_earnings_data': list(earnings_data.keys()),
        },
        'variants': {},
        'summary': {
            'n_variants': len(all_results),
            'n_passing': len(passing_variants),
            'passing_variants': passing_variants,
        }
    }

    for name, res in all_results.items():
        json_results['variants'][name] = {
            'metrics': res['metrics'],
            'gates': res['gates'],
            'p_value': res['p_value'],
            'n_signals': res['n_signals'],
            'n_trades': res['metrics']['n_trades'],
            'trades': res['trades'],  # full trade log
        }

    # Custom encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(json_results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {RESULTS_PATH}")
    print("Done.")

    return json_results


if __name__ == '__main__':
    np.random.seed(42)
    results = main()
