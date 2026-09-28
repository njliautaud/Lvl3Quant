#!/usr/bin/env python3
"""
Bear Market Alpha Strategy Backtest
====================================
Tests 6 equity-only strategies designed to generate positive returns
during bear markets (SPY below 200-SMA).

Constraints: $669 starting capital, cash account, no shorting, no margin,
fractional shares OK, $0 commission, 0.02% slippage.

Strategies:
  A) Inverse ETF Momentum
  B) VIX Spike Fade
  C) Quality Flight
  D) Bear Rally Surfer
  E) Gold + Dollar Bear Hedge
  F) Defensive Sector Rotation
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
START_DATE = '2021-01-01'  # extra lookback for 200-SMA
END_DATE = '2026-07-30'
TRADE_START = '2022-01-01'
STARTING_CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02%
PERMUTATION_ITERS = 1000
np.random.seed(42)

# ── Data Download ───────────────────────────────────────────────────────
ALL_TICKERS = [
    'SPY', '^VIX',
    # Strategy A
    'SH', 'PSQ', 'TLT', 'GLD', 'UUP',
    # Strategy C
    'MSFT', 'AAPL', 'GOOGL',
    # Strategy D
    'QQQ',
    # Strategy F
    'XLU', 'XLP', 'XLV', 'XLRE',
]

print("Downloading data...")
raw = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)

# Handle multi-level columns from yfinance
if isinstance(raw.columns, pd.MultiIndex):
    close = raw['Close'].copy()
else:
    close = raw[['Close']].copy()
    close.columns = ['SPY']

# Flatten column names if needed
if isinstance(close.columns, pd.MultiIndex):
    close.columns = [c[-1] if isinstance(c, tuple) else c for c in close.columns]

close = close.ffill()

# VIX column name handling
vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
if vix_col not in close.columns:
    # Try downloading VIX separately
    vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
    if isinstance(vix_data.columns, pd.MultiIndex):
        close['^VIX'] = vix_data['Close'].values
    else:
        close['^VIX'] = vix_data['Close'].values
    vix_col = '^VIX'

print(f"Data: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}, {len(close)} days")

# ── Regime Classification ──────────────────────────────────────────────
spy = close['SPY'].copy()
spy_sma200 = spy.rolling(200).mean()
regime = (spy >= spy_sma200).astype(int)  # 1=bull, 0=bear
regime.name = 'regime'

# Trim to trade period
trade_mask = close.index >= TRADE_START
trade_dates = close.index[trade_mask]
print(f"Trade period: {trade_dates[0].strftime('%Y-%m-%d')} to {trade_dates[-1].strftime('%Y-%m-%d')}")
bear_days = (regime[trade_mask] == 0).sum()
bull_days = (regime[trade_mask] == 1).sum()
print(f"Bear days: {bear_days}, Bull days: {bull_days}")

# ── Utility Functions ──────────────────────────────────────────────────

def compute_returns(prices):
    return prices.pct_change().fillna(0)

def apply_slippage(price, direction='buy'):
    """Apply slippage to execution price."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)

def sharpe_ratio(returns, annual_factor=252):
    """Annualized Sharpe from daily returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(annual_factor)

def max_drawdown(equity_curve):
    """Maximum drawdown from equity curve."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return dd.min()

def win_rate(trade_returns):
    """Win rate from list of trade returns."""
    if len(trade_returns) == 0:
        return 0.0
    return sum(1 for r in trade_returns if r > 0) / len(trade_returns)

def rsi(prices, period=14):
    """RSI indicator."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ── Strategy Backtester ────────────────────────────────────────────────

class StrategyResult:
    def __init__(self, name):
        self.name = name
        self.equity_curve = None
        self.daily_returns = None
        self.trades = []  # list of {entry_date, exit_date, return, ticker}
        self.bear_returns = None

def run_strategy_A():
    """Inverse ETF Momentum: buy strongest inverse/defensive ETF when bear."""
    result = StrategyResult("A) Inverse ETF Momentum")
    etfs = ['SH', 'PSQ', 'TLT', 'GLD', 'UUP']

    equity = STARTING_CAPITAL
    equity_series = {}
    trades = []

    holding = None
    hold_ticker = None
    entry_price = None
    entry_date = None
    shares = 0

    # Monthly rebalance dates (first trading day of each month)
    monthly_dates = trade_dates.to_series().groupby([trade_dates.year, trade_dates.month]).first()
    rebal_set = set(monthly_dates.values)

    for i, date in enumerate(trade_dates):
        current_regime = regime.loc[date]

        if current_regime == 1:  # Bull - go to cash
            if holding is not None:
                exit_price = apply_slippage(close.loc[date, hold_ticker], 'sell')
                trade_ret = (exit_price / entry_price) - 1
                trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(date.date()),
                              'return': trade_ret, 'ticker': hold_ticker})
                equity = shares * exit_price
                holding = None
                hold_ticker = None
                shares = 0
            equity_series[date] = equity
            continue

        # Bear regime
        if date in rebal_set or holding is None:
            # Rank ETFs by 1-month return
            lookback = 21
            idx = close.index.get_loc(date)
            if idx < lookback:
                equity_series[date] = equity
                continue

            returns_1m = {}
            for etf in etfs:
                if etf in close.columns:
                    past = close[etf].iloc[idx - lookback]
                    curr = close[etf].iloc[idx]
                    if past > 0:
                        returns_1m[etf] = curr / past - 1

            if not returns_1m:
                equity_series[date] = equity
                continue

            best_etf = max(returns_1m, key=returns_1m.get)

            # Sell current if different
            if holding is not None and hold_ticker != best_etf:
                exit_price = apply_slippage(close.loc[date, hold_ticker], 'sell')
                trade_ret = (exit_price / entry_price) - 1
                trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(date.date()),
                              'return': trade_ret, 'ticker': hold_ticker})
                equity = shares * exit_price
                holding = None
                hold_ticker = None
                shares = 0

            if holding is None:
                buy_price = apply_slippage(close.loc[date, best_etf], 'buy')
                shares = equity / buy_price
                entry_price = buy_price
                entry_date = date
                hold_ticker = best_etf
                holding = True

        if holding is not None:
            current_val = shares * close.loc[date, hold_ticker]
            equity_series[date] = current_val
        else:
            equity_series[date] = equity

    # Close any remaining position
    if holding is not None:
        last_date = trade_dates[-1]
        exit_price = apply_slippage(close.loc[last_date, hold_ticker], 'sell')
        trade_ret = (exit_price / entry_price) - 1
        trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(last_date.date()),
                      'return': trade_ret, 'ticker': hold_ticker})
        equity_series[last_date] = shares * exit_price

    eq = pd.Series(equity_series).reindex(trade_dates).ffill().fillna(STARTING_CAPITAL)
    result.equity_curve = eq
    result.daily_returns = eq.pct_change().fillna(0)
    result.trades = trades
    return result


def run_strategy_B():
    """VIX Spike Fade: buy SPY when VIX spikes >30 and >50% above 20d SMA."""
    result = StrategyResult("B) VIX Spike Fade")

    vix = close[vix_col].copy()
    vix_sma20 = vix.rolling(20).mean()

    equity = STARTING_CAPITAL
    equity_series = {}
    trades = []

    holding = False
    entry_price = None
    entry_date = None
    shares = 0
    hold_days = 0

    for date in trade_dates:
        idx = close.index.get_loc(date)
        v = vix.iloc[idx]
        v_sma = vix_sma20.iloc[idx]

        if holding:
            hold_days += 1
            current_val = shares * close.loc[date, 'SPY']
            equity_series[date] = current_val

            # Exit: VIX < 20 or held 10 days
            if v < 20 or hold_days >= 10:
                exit_price = apply_slippage(close.loc[date, 'SPY'], 'sell')
                trade_ret = (exit_price / entry_price) - 1
                trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(date.date()),
                              'return': trade_ret, 'ticker': 'SPY'})
                equity = shares * exit_price
                holding = False
                shares = 0
                hold_days = 0
        else:
            equity_series[date] = equity
            # Entry: VIX > 30 and > 50% above 20d SMA
            if pd.notna(v_sma) and v > 30 and v > v_sma * 1.5:
                buy_price = apply_slippage(close.loc[date, 'SPY'], 'buy')
                shares = equity / buy_price
                entry_price = buy_price
                entry_date = date
                holding = True
                hold_days = 0

    if holding:
        last_date = trade_dates[-1]
        exit_price = apply_slippage(close.loc[last_date, 'SPY'], 'sell')
        trade_ret = (exit_price / entry_price) - 1
        trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(last_date.date()),
                      'return': trade_ret, 'ticker': 'SPY'})
        equity_series[last_date] = shares * exit_price

    eq = pd.Series(equity_series).reindex(trade_dates).ffill().fillna(STARTING_CAPITAL)
    result.equity_curve = eq
    result.daily_returns = eq.pct_change().fillna(0)
    result.trades = trades
    return result


def run_strategy_C():
    """Quality Flight: in bear, buy top 2 most oversold quality names (MSFT, AAPL, GOOGL)."""
    result = StrategyResult("C) Quality Flight")
    quality = ['MSFT', 'AAPL', 'GOOGL']

    # Precompute RSI
    rsi_data = {}
    for t in quality:
        rsi_data[t] = rsi(close[t], 14)

    equity = STARTING_CAPITAL
    equity_series = {}
    trades = []

    holding = False
    positions = {}  # ticker -> {shares, entry_price, entry_date}
    hold_days = 0

    for date in trade_dates:
        current_regime = regime.loc[date]

        if holding:
            hold_days += 1
            current_val = sum(pos['shares'] * close.loc[date, t] for t, pos in positions.items())
            equity_series[date] = current_val

            # Exit: 20 days or regime switch to bull
            if hold_days >= 20 or current_regime == 1:
                total_exit = 0
                for t, pos in positions.items():
                    exit_price = apply_slippage(close.loc[date, t], 'sell')
                    trade_ret = (exit_price / pos['entry_price']) - 1
                    trades.append({'entry_date': str(pos['entry_date'].date()), 'exit_date': str(date.date()),
                                  'return': trade_ret, 'ticker': t})
                    total_exit += pos['shares'] * exit_price
                equity = total_exit
                positions = {}
                holding = False
                hold_days = 0
        else:
            equity_series[date] = equity

            # Entry: bear regime only
            if current_regime == 0:
                # Rank by RSI (lowest = most oversold)
                rsi_vals = {}
                for t in quality:
                    r = rsi_data[t].loc[date] if date in rsi_data[t].index else None
                    if pd.notna(r):
                        rsi_vals[t] = r

                if len(rsi_vals) >= 2:
                    sorted_tickers = sorted(rsi_vals, key=rsi_vals.get)[:2]
                    alloc = equity / 2
                    positions = {}
                    for t in sorted_tickers:
                        buy_price = apply_slippage(close.loc[date, t], 'buy')
                        positions[t] = {
                            'shares': alloc / buy_price,
                            'entry_price': buy_price,
                            'entry_date': date
                        }
                    holding = True
                    hold_days = 0

    if holding:
        last_date = trade_dates[-1]
        total_exit = 0
        for t, pos in positions.items():
            exit_price = apply_slippage(close.loc[last_date, t], 'sell')
            trade_ret = (exit_price / pos['entry_price']) - 1
            trades.append({'entry_date': str(pos['entry_date'].date()), 'exit_date': str(last_date.date()),
                          'return': trade_ret, 'ticker': t})
            total_exit += pos['shares'] * exit_price
        equity_series[last_date] = total_exit

    eq = pd.Series(equity_series).reindex(trade_dates).ffill().fillna(STARTING_CAPITAL)
    result.equity_curve = eq
    result.daily_returns = eq.pct_change().fillna(0)
    result.trades = trades
    return result


def run_strategy_D():
    """Bear Rally Surfer: buy QQQ/SPY when deeply oversold in bear regime."""
    result = StrategyResult("D) Bear Rally Surfer")

    spy_rsi5 = rsi(close['SPY'], 5)

    equity = STARTING_CAPITAL
    equity_series = {}
    trades = []

    holding = False
    entry_price = None
    entry_date = None
    shares = 0
    hold_days = 0
    buy_ticker = 'QQQ'

    for date in trade_dates:
        current_regime = regime.loc[date]
        r5 = spy_rsi5.loc[date] if date in spy_rsi5.index else None

        if holding:
            hold_days += 1
            current_val = shares * close.loc[date, buy_ticker]
            equity_series[date] = current_val

            # Exit: RSI(5) > 60 or 5 days
            curr_rsi5 = spy_rsi5.loc[date] if date in spy_rsi5.index else 50
            if curr_rsi5 > 60 or hold_days >= 5:
                exit_price = apply_slippage(close.loc[date, buy_ticker], 'sell')
                trade_ret = (exit_price / entry_price) - 1
                trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(date.date()),
                              'return': trade_ret, 'ticker': buy_ticker})
                equity = shares * exit_price
                holding = False
                shares = 0
                hold_days = 0
        else:
            equity_series[date] = equity

            # Entry: bear regime + RSI(5) < 25
            if current_regime == 0 and pd.notna(r5) and r5 < 25:
                # Alternate between QQQ and SPY
                buy_ticker = 'QQQ' if np.random.random() > 0.5 else 'SPY'
                buy_price = apply_slippage(close.loc[date, buy_ticker], 'buy')
                shares = equity / buy_price
                entry_price = buy_price
                entry_date = date
                holding = True
                hold_days = 0

    if holding:
        last_date = trade_dates[-1]
        exit_price = apply_slippage(close.loc[last_date, buy_ticker], 'sell')
        trade_ret = (exit_price / entry_price) - 1
        trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(last_date.date()),
                      'return': trade_ret, 'ticker': buy_ticker})
        equity_series[last_date] = shares * exit_price

    eq = pd.Series(equity_series).reindex(trade_dates).ffill().fillna(STARTING_CAPITAL)
    result.equity_curve = eq
    result.daily_returns = eq.pct_change().fillna(0)
    result.trades = trades
    return result


def run_strategy_E():
    """Gold + Dollar Bear Hedge: 50% GLD + 50% UUP when SPY below 200-SMA."""
    result = StrategyResult("E) Gold + Dollar Bear Hedge")

    equity = STARTING_CAPITAL
    equity_series = {}
    trades = []

    holding = False
    positions = {}  # ticker -> {shares, entry_price, entry_date}

    prev_regime = None

    for date in trade_dates:
        current_regime = regime.loc[date]

        if holding:
            if current_regime == 1:  # Regime switch to bull - exit
                total_exit = 0
                for t, pos in positions.items():
                    exit_price = apply_slippage(close.loc[date, t], 'sell')
                    trade_ret = (exit_price / pos['entry_price']) - 1
                    trades.append({'entry_date': str(pos['entry_date'].date()), 'exit_date': str(date.date()),
                                  'return': trade_ret, 'ticker': t})
                    total_exit += pos['shares'] * exit_price
                equity = total_exit
                positions = {}
                holding = False
            else:
                current_val = sum(pos['shares'] * close.loc[date, t] for t, pos in positions.items())
                equity_series[date] = current_val

        if not holding:
            equity_series[date] = equity

            # Entry: regime crosses below 200-SMA (bear starts)
            if current_regime == 0 and (prev_regime == 1 or prev_regime is None):
                alloc = equity / 2
                positions = {}
                for t in ['GLD', 'UUP']:
                    buy_price = apply_slippage(close.loc[date, t], 'buy')
                    positions[t] = {
                        'shares': alloc / buy_price,
                        'entry_price': buy_price,
                        'entry_date': date
                    }
                holding = True

        if holding and current_regime == 0:
            current_val = sum(pos['shares'] * close.loc[date, t] for t, pos in positions.items())
            equity_series[date] = current_val

        prev_regime = current_regime

    if holding:
        last_date = trade_dates[-1]
        total_exit = 0
        for t, pos in positions.items():
            exit_price = apply_slippage(close.loc[last_date, t], 'sell')
            trade_ret = (exit_price / pos['entry_price']) - 1
            trades.append({'entry_date': str(pos['entry_date'].date()), 'exit_date': str(last_date.date()),
                          'return': trade_ret, 'ticker': t})
            total_exit += pos['shares'] * exit_price
        equity_series[last_date] = total_exit

    eq = pd.Series(equity_series).reindex(trade_dates).ffill().fillna(STARTING_CAPITAL)
    result.equity_curve = eq
    result.daily_returns = eq.pct_change().fillna(0)
    result.trades = trades
    return result


def run_strategy_F():
    """Defensive Sector Rotation: monthly rotation among XLU/XLP/XLV/XLRE in bear."""
    result = StrategyResult("F) Defensive Sector Rotation")
    sectors = ['XLU', 'XLP', 'XLV', 'XLRE']

    equity = STARTING_CAPITAL
    equity_series = {}
    trades = []

    holding = None
    hold_ticker = None
    entry_price = None
    entry_date = None
    shares = 0

    monthly_dates = trade_dates.to_series().groupby([trade_dates.year, trade_dates.month]).first()
    rebal_set = set(monthly_dates.values)

    for date in trade_dates:
        current_regime = regime.loc[date]

        if current_regime == 1:  # Bull - cash
            if holding is not None:
                exit_price = apply_slippage(close.loc[date, hold_ticker], 'sell')
                trade_ret = (exit_price / entry_price) - 1
                trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(date.date()),
                              'return': trade_ret, 'ticker': hold_ticker})
                equity = shares * exit_price
                holding = None
                hold_ticker = None
                shares = 0
            equity_series[date] = equity
            continue

        # Bear regime
        if date in rebal_set or holding is None:
            lookback = 21
            idx = close.index.get_loc(date)
            if idx < lookback:
                equity_series[date] = equity
                continue

            returns_1m = {}
            for s in sectors:
                if s in close.columns:
                    past = close[s].iloc[idx - lookback]
                    curr = close[s].iloc[idx]
                    if past > 0:
                        returns_1m[s] = curr / past - 1

            if not returns_1m:
                equity_series[date] = equity
                continue

            best_sector = max(returns_1m, key=returns_1m.get)

            if holding is not None and hold_ticker != best_sector:
                exit_price = apply_slippage(close.loc[date, hold_ticker], 'sell')
                trade_ret = (exit_price / entry_price) - 1
                trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(date.date()),
                              'return': trade_ret, 'ticker': hold_ticker})
                equity = shares * exit_price
                holding = None
                hold_ticker = None
                shares = 0

            if holding is None:
                buy_price = apply_slippage(close.loc[date, best_sector], 'buy')
                shares = equity / buy_price
                entry_price = buy_price
                entry_date = date
                hold_ticker = best_sector
                holding = True

        if holding is not None:
            current_val = shares * close.loc[date, hold_ticker]
            equity_series[date] = current_val
        else:
            equity_series[date] = equity

    if holding is not None:
        last_date = trade_dates[-1]
        exit_price = apply_slippage(close.loc[last_date, hold_ticker], 'sell')
        trade_ret = (exit_price / entry_price) - 1
        trades.append({'entry_date': str(entry_date.date()), 'exit_date': str(last_date.date()),
                      'return': trade_ret, 'ticker': hold_ticker})
        equity_series[last_date] = shares * exit_price

    eq = pd.Series(equity_series).reindex(trade_dates).ffill().fillna(STARTING_CAPITAL)
    result.equity_curve = eq
    result.daily_returns = eq.pct_change().fillna(0)
    result.trades = trades
    return result


# ── Validation Gates ───────────────────────────────────────────────────

def permutation_test(result, n_iters=PERMUTATION_ITERS):
    """Permutation test: shuffle entry timing within bear periods, compute p-value."""
    if len(result.trades) < 3:
        return 1.0  # Not enough trades

    actual_mean = np.mean([t['return'] for t in result.trades])

    # Get bear-period daily returns for the asset(s) traded
    bear_mask = regime[trade_dates] == 0
    bear_dates = trade_dates[bear_mask]

    if len(bear_dates) < 10:
        return 1.0

    # Compute SPY bear-period returns as baseline
    spy_bear_rets = close['SPY'].reindex(bear_dates).pct_change().dropna().values

    if len(spy_bear_rets) < 5:
        return 1.0

    avg_hold = max(1, int(np.mean([
        (pd.Timestamp(t['exit_date']) - pd.Timestamp(t['entry_date'])).days
        for t in result.trades
    ])))

    count_better = 0
    n_trades = len(result.trades)

    for _ in range(n_iters):
        shuffled_returns = []
        for _ in range(n_trades):
            start = np.random.randint(0, max(1, len(spy_bear_rets) - avg_hold))
            end = min(start + avg_hold, len(spy_bear_rets))
            chunk = spy_bear_rets[start:end]
            shuffled_returns.append(np.prod(1 + chunk) - 1)

        if np.mean(shuffled_returns) >= actual_mean:
            count_better += 1

    return count_better / n_iters


def validate_strategy(result):
    """Apply 5-gate validation. Returns dict with pass/fail for each gate."""
    trades = result.trades
    trade_rets = [t['return'] for t in trades]

    # Bear-period-only returns
    bear_mask = regime[trade_dates] == 0
    bear_daily_returns = result.daily_returns[bear_mask]
    bear_daily_returns = bear_daily_returns[bear_daily_returns != 0]  # Remove pure cash days

    # If no bear returns with activity, use all non-zero returns
    if len(bear_daily_returns) < 5:
        bear_daily_returns = result.daily_returns[result.daily_returns != 0]

    bear_sharpe = sharpe_ratio(bear_daily_returns) if len(bear_daily_returns) > 1 else 0.0
    mdd = max_drawdown(result.equity_curve)
    wr = win_rate(trade_rets) if trade_rets else 0.0
    n_trades = len(trades)

    perm_p = permutation_test(result)

    gates = {
        'sharpe_bear': {'value': round(bear_sharpe, 3), 'threshold': 0.5, 'pass': bear_sharpe > 0.5},
        'permutation_p': {'value': round(perm_p, 4), 'threshold': 0.05, 'pass': perm_p < 0.05},
        'max_drawdown': {'value': round(mdd, 4), 'threshold': -0.30, 'pass': mdd > -0.30},
        'n_trades': {'value': n_trades, 'threshold': 10, 'pass': n_trades >= 10},
        'win_rate': {'value': round(wr, 4), 'threshold': 0.45, 'pass': wr > 0.45},
    }

    gates['all_pass'] = all(g['pass'] for g in gates.values() if isinstance(g, dict))

    return gates, bear_sharpe, mdd, wr, perm_p


def compute_bear_stats(result):
    """Compute bear-period-specific statistics."""
    bear_mask = regime[trade_dates] == 0
    bear_eq = result.equity_curve[bear_mask]

    if len(bear_eq) < 2:
        return {'bear_return': 0, 'bear_sharpe': 0, 'time_invested_pct': 0}

    bear_daily_rets = result.daily_returns[bear_mask]
    active_days = (bear_daily_rets != 0).sum()

    # Bear period return: compare equity at start vs end of bear periods
    bear_return = (bear_eq.iloc[-1] / bear_eq.iloc[0]) - 1 if bear_eq.iloc[0] > 0 else 0

    bear_sharpe = sharpe_ratio(bear_daily_rets[bear_daily_rets != 0]) if (bear_daily_rets != 0).sum() > 1 else 0

    time_invested = active_days / len(bear_eq) if len(bear_eq) > 0 else 0

    return {
        'bear_return_pct': round(float(bear_return * 100), 2),
        'bear_sharpe': round(float(bear_sharpe), 3),
        'time_invested_during_bear_pct': round(float(time_invested * 100), 1),
        'bear_days_total': int(len(bear_eq)),
        'bear_days_active': int(active_days),
    }


# ── Run All Strategies ─────────────────────────────────────────────────

print("\n" + "="*70)
print("RUNNING 6 BEAR MARKET ALPHA STRATEGIES")
print("="*70)

strategies = {
    'A': run_strategy_A,
    'B': run_strategy_B,
    'C': run_strategy_C,
    'D': run_strategy_D,
    'E': run_strategy_E,
    'F': run_strategy_F,
}

results = {}
all_output = {}

for key, func in strategies.items():
    print(f"\nRunning Strategy {key}...")
    res = func()
    gates, bear_sharpe, mdd, wr, perm_p = validate_strategy(res)
    bear_stats = compute_bear_stats(res)

    total_return = (res.equity_curve.iloc[-1] / STARTING_CAPITAL - 1) * 100
    final_equity = res.equity_curve.iloc[-1]

    results[key] = {
        'name': res.name,
        'final_equity': round(float(final_equity), 2),
        'total_return_pct': round(float(total_return), 2),
        'n_trades': len(res.trades),
        'win_rate': round(float(wr), 4),
        'max_drawdown_pct': round(float(mdd * 100), 2),
        'bear_sharpe': round(float(bear_sharpe), 3),
        'permutation_p': round(float(perm_p), 4),
        'validation_gates': gates,
        'bear_stats': bear_stats,
        'trades': res.trades,
    }

    passed = "PASS" if gates['all_pass'] else "FAIL"
    gates_passed = sum(1 for g in gates.values() if isinstance(g, dict) and g['pass'])

    print(f"  {res.name}")
    print(f"  Final equity: ${final_equity:.2f} ({total_return:+.1f}%)")
    print(f"  Trades: {len(res.trades)}, WR: {wr:.1%}, MaxDD: {mdd:.1%}")
    print(f"  Bear Sharpe: {bear_sharpe:.3f}, Perm p: {perm_p:.4f}")
    print(f"  Bear return: {bear_stats['bear_return_pct']:+.1f}%, Invested: {bear_stats['time_invested_during_bear_pct']:.0f}%")
    print(f"  Validation: {passed} ({gates_passed}/5 gates)")

    all_output[key] = res  # Save for portfolio analysis

# ── Portfolio Complement Analysis ──────────────────────────────────────

print("\n" + "="*70)
print("PORTFOLIO COMPLEMENT ANALYSIS")
print("="*70)

# Simulate combined portfolio: use bear strategy during bear, cash during bull
# Compare with SPY buy-and-hold as baseline
spy_eq = close['SPY'].reindex(trade_dates)
spy_eq_normalized = spy_eq / spy_eq.iloc[0] * STARTING_CAPITAL
spy_returns = spy_eq.pct_change().fillna(0)
spy_sharpe = sharpe_ratio(spy_returns)

print(f"\nSPY Buy & Hold Sharpe: {spy_sharpe:.3f}")

# For each passing strategy, compute what combined Sharpe would be
# Combined = strategy returns during bear + SPY returns during bull
bull_mask = regime[trade_dates] == 1
bear_mask = regime[trade_dates] == 0

for key, res in all_output.items():
    combined_returns = pd.Series(0.0, index=trade_dates)
    combined_returns[bull_mask] = spy_returns[bull_mask]
    combined_returns[bear_mask] = res.daily_returns[bear_mask]

    combined_sharpe = sharpe_ratio(combined_returns)
    combined_total_ret = (np.cumprod(1 + combined_returns).iloc[-1] - 1) * 100

    results[key]['combined_with_spy'] = {
        'combined_sharpe': round(float(combined_sharpe), 3),
        'combined_total_return_pct': round(float(combined_total_ret), 2),
        'spy_only_sharpe': round(float(spy_sharpe), 3),
        'improvement': round(float(combined_sharpe - spy_sharpe), 3),
    }

    improvement = "+" if combined_sharpe > spy_sharpe else ""
    print(f"  {results[key]['name']}: Combined Sharpe = {combined_sharpe:.3f} ({improvement}{combined_sharpe - spy_sharpe:.3f} vs SPY-only)")


# ── Summary ────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

passing = []
failing = []

for key in sorted(results.keys()):
    r = results[key]
    gates = r['validation_gates']
    if gates['all_pass']:
        passing.append(key)
    else:
        failing.append(key)

    status = "PASS" if gates['all_pass'] else "FAIL"
    gate_details = []
    for gname, gval in gates.items():
        if isinstance(gval, dict):
            mark = "+" if gval['pass'] else "X"
            gate_details.append(f"{gname}({mark})")

    print(f"  [{status}] {r['name']}")
    print(f"         Return: {r['total_return_pct']:+.1f}% | Bear Sharpe: {r['bear_sharpe']:.3f} | "
          f"WR: {r['win_rate']:.1%} | MaxDD: {r['max_drawdown_pct']:.1f}% | "
          f"Trades: {r['n_trades']} | Perm p: {r['permutation_p']:.4f}")
    print(f"         Gates: {' | '.join(gate_details)}")
    if 'combined_with_spy' in r:
        print(f"         Combined w/SPY: Sharpe={r['combined_with_spy']['combined_sharpe']:.3f} "
              f"(delta={r['combined_with_spy']['improvement']:+.3f})")

print(f"\nPassing strategies: {passing if passing else 'None'}")
print(f"Failing strategies: {failing if failing else 'None'}")

if passing:
    best = max(passing, key=lambda k: results[k]['bear_sharpe'])
    print(f"\nBest bear-market strategy: {results[best]['name']}")
    print(f"  Bear Sharpe: {results[best]['bear_sharpe']:.3f}")
    print(f"  Combined w/SPY Sharpe: {results[best]['combined_with_spy']['combined_sharpe']:.3f}")

# ── Save Results ───────────────────────────────────────────────────────

output_path = '/home/jupiter/Lvl3Quant/data/bear_market_alpha_results.json'

output = {
    'metadata': {
        'run_date': datetime.now().isoformat(),
        'period': f"{TRADE_START} to {END_DATE}",
        'starting_capital': STARTING_CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'bear_days': int(bear_days),
        'bull_days': int(bull_days),
        'spy_sharpe': round(float(spy_sharpe), 3),
    },
    'strategies': results,
    'passing_strategies': passing,
    'failing_strategies': failing,
}

# Convert any remaining numpy types
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

output = convert_numpy(output)

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
