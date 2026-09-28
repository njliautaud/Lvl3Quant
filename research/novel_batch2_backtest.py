#!/usr/bin/env python3
"""
Novel Strategy Batch 2 — 6 strategies x 6 variants = 36 backtests
Fundamentally different approaches: meta-signals, leveraged ETF decay,
index MR, factor rotation, volatile-stock MR, VIX term structure.

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 shuffles)
  3. Regime gap < 0.50
  4. Max DD < 30%
  5. Min 20 trades
"""

import json, warnings, sys, os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from collections import defaultdict

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─── CONFIG ──────────────────────────────────────────────────────────
START = "2020-01-01"
END = "2026-07-31"
ACCOUNT = 645.0
POS_SIZE = 200.0
MAX_CONCURRENT = 3
COMMISSION_RT = 1.0  # $1 round-trip for shares
N_PERM = 1000

QUALITY_UNIVERSE = json.load(open("/home/jupiter/Lvl3Quant/data/quality_universe.json"))["tickers"]

VOLATILE_UNIVERSE = [
    "TSLA", "AMD", "NVDA", "PLTR", "SQ", "SHOP", "COIN", "MARA",
    "RIOT", "SNAP", "ROKU", "DKNG", "SOFI", "HOOD", "RBLX"
]

FACTOR_ETFS = ["VTV", "VUG", "MTUM", "QUAL", "USMV"]  # VLUE/SIZE have limited history

LEVERAGED = ["TQQQ", "SQQQ", "QQQ"]

VIX_TICKERS = ["VIXY", "SVXY", "SPY", "^VIX"]

INDEX_TICKERS = ["SPY", "QQQ", "^VIX"]

# ─── DATA LOADING ────────────────────────────────────────────────────
CACHE_DIR = Path("/home/jupiter/Lvl3Quant/research/cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_data(tickers, start=START, end=END):
    """Load OHLCV data for tickers, with disk cache."""
    all_tickers = list(set(tickers))
    result = {}
    to_download = []

    for t in all_tickers:
        cache_file = CACHE_DIR / f"{t.replace('^', '_')}_{start}_{end}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                if len(df) > 100:
                    result[t] = df
                    continue
            except:
                pass
        to_download.append(t)

    if to_download:
        print(f"  Downloading {len(to_download)} tickers: {to_download[:10]}...")
        for t in to_download:
            try:
                df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) > 50:
                    cache_file = CACHE_DIR / f"{t.replace('^', '_')}_{start}_{end}.parquet"
                    df.to_parquet(cache_file)
                    result[t] = df
            except Exception as e:
                print(f"    WARN: {t} download failed: {e}")

    return result


# ─── INDICATORS ──────────────────────────────────────────────────────
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def sma(series, period):
    return series.rolling(period).mean()

def bollinger_pctb(series, period=20, std=2):
    mid = series.rolling(period).mean()
    sd = series.rolling(period).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return (series - lower) / (upper - lower)

def drawdown_from_high(series, lookback):
    roll_max = series.rolling(lookback, min_periods=1).max()
    return (series - roll_max) / roll_max

def consecutive_red_days(close):
    """Return series counting consecutive red (down) days."""
    red = (close < close.shift(1)).astype(int)
    groups = red.ne(red.shift()).cumsum()
    return red.groupby(groups).cumsum()

def realized_vol_proxy(df, window=20):
    """ATR-based realized vol annualized."""
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr * np.sqrt(252) / df['Close']


# ─── BACKTESTER ──────────────────────────────────────────────────────
def backtest_signals(signal_dates, ticker_data, hold_days=10, pos_size=POS_SIZE,
                     account=ACCOUNT, max_concurrent=MAX_CONCURRENT,
                     dynamic_exit_fn=None, scale_fn=None):
    """
    Generic backtester.
    signal_dates: list of (date, ticker) or (date, ticker, weight)
    Returns metrics dict.
    """
    trades = []
    open_positions = []
    equity = account
    equity_curve = []
    peak_equity = account

    # Build price lookup
    price_lookup = {}
    for ticker, df in ticker_data.items():
        for idx in df.index:
            dt = idx.date() if hasattr(idx, 'date') else idx
            price_lookup[(ticker, dt)] = df.loc[idx, 'Close']

    # Sort signals by date
    signals = []
    for s in signal_dates:
        if len(s) == 3:
            signals.append((s[0], s[1], s[2]))
        else:
            signals.append((s[0], s[1], 1.0))
    signals.sort(key=lambda x: x[0])

    all_dates = sorted(set(d for d, _, _ in signals))

    for sig_date, ticker, weight in signals:
        # Close expired positions
        new_open = []
        for pos in open_positions:
            closed = False
            if dynamic_exit_fn is not None:
                exit_date = dynamic_exit_fn(pos, ticker_data)
                if exit_date and exit_date <= sig_date:
                    exit_price = price_lookup.get((pos['ticker'], exit_date))
                    if exit_price:
                        pnl = (exit_price / pos['entry_price'] - 1) * pos['size'] - COMMISSION_RT
                        equity += pnl
                        trades.append({
                            'entry_date': pos['entry_date'],
                            'exit_date': exit_date,
                            'ticker': pos['ticker'],
                            'pnl': pnl,
                            'return': exit_price / pos['entry_price'] - 1
                        })
                        closed = True
            if not closed and pos.get('exit_date') and pos['exit_date'] <= sig_date:
                exit_price = price_lookup.get((pos['ticker'], pos['exit_date']))
                if exit_price is None:
                    # Find nearest available date
                    if pos['ticker'] in ticker_data:
                        df = ticker_data[pos['ticker']]
                        future = df[df.index >= pd.Timestamp(pos['exit_date'])]
                        if len(future) > 0:
                            exit_price = future.iloc[0]['Close']
                if exit_price:
                    pnl = (exit_price / pos['entry_price'] - 1) * pos['size'] - COMMISSION_RT
                    equity += pnl
                    trades.append({
                        'entry_date': pos['entry_date'],
                        'exit_date': pos['exit_date'],
                        'ticker': pos['ticker'],
                        'pnl': pnl,
                        'return': exit_price / pos['entry_price'] - 1
                    })
                    closed = True
            if not closed:
                new_open.append(pos)
        open_positions = new_open

        # Check max concurrent
        if len(open_positions) >= max_concurrent:
            continue

        entry_price = price_lookup.get((ticker, sig_date))
        if entry_price is None or entry_price <= 0:
            continue

        size = pos_size * weight if scale_fn is None else scale_fn(weight) * pos_size

        # Compute exit date
        if ticker in ticker_data:
            df = ticker_data[ticker]
            future = df[df.index >= pd.Timestamp(sig_date)]
            if len(future) > hold_days:
                exit_dt = future.index[hold_days].date() if hasattr(future.index[hold_days], 'date') else future.index[hold_days]
            elif len(future) > 1:
                exit_dt = future.index[-1].date() if hasattr(future.index[-1], 'date') else future.index[-1]
            else:
                continue
        else:
            continue

        open_positions.append({
            'ticker': ticker,
            'entry_date': sig_date,
            'entry_price': entry_price,
            'exit_date': exit_dt,
            'size': size
        })
        equity_curve.append((sig_date, equity))

    # Close remaining positions
    for pos in open_positions:
        exit_price = price_lookup.get((pos['ticker'], pos.get('exit_date')))
        if exit_price is None and pos['ticker'] in ticker_data:
            df = ticker_data[pos['ticker']]
            if len(df) > 0:
                exit_price = df.iloc[-1]['Close']
        if exit_price:
            pnl = (exit_price / pos['entry_price'] - 1) * pos['size'] - COMMISSION_RT
            equity += pnl
            trades.append({
                'entry_date': pos['entry_date'],
                'exit_date': pos.get('exit_date', df.index[-1].date()),
                'ticker': pos['ticker'],
                'pnl': pnl,
                'return': exit_price / pos['entry_price'] - 1
            })

    return compute_metrics(trades, account)


def backtest_short_signals(signal_dates, ticker_data, hold_days=20,
                           pos_size=POS_SIZE, account=ACCOUNT,
                           max_concurrent=MAX_CONCURRENT, exit_fn=None):
    """Backtest for SHORT positions."""
    trades = []
    open_positions = []
    equity = account

    price_lookup = {}
    for ticker, df in ticker_data.items():
        for idx in df.index:
            dt = idx.date() if hasattr(idx, 'date') else idx
            price_lookup[(ticker, dt)] = df.loc[idx, 'Close']

    signals = [(s[0], s[1]) if len(s) == 2 else (s[0], s[1]) for s in signal_dates]
    signals.sort(key=lambda x: x[0])

    for sig_date, ticker in signals:
        # Close expired
        new_open = []
        for pos in open_positions:
            should_close = False
            exit_date = pos.get('exit_date')

            if exit_fn is not None:
                custom_exit = exit_fn(pos, ticker_data)
                if custom_exit and custom_exit <= sig_date:
                    exit_date = custom_exit
                    should_close = True

            if not should_close and exit_date and exit_date <= sig_date:
                should_close = True

            if should_close:
                exit_price = price_lookup.get((pos['ticker'], exit_date))
                if exit_price is None and pos['ticker'] in ticker_data:
                    df = ticker_data[pos['ticker']]
                    future = df[df.index >= pd.Timestamp(exit_date)]
                    if len(future) > 0:
                        exit_price = future.iloc[0]['Close']
                if exit_price:
                    # SHORT: profit when price goes down
                    pnl = (pos['entry_price'] / exit_price - 1) * pos['size'] - COMMISSION_RT
                    equity += pnl
                    trades.append({
                        'entry_date': pos['entry_date'],
                        'exit_date': exit_date,
                        'ticker': pos['ticker'],
                        'pnl': pnl,
                        'return': pos['entry_price'] / exit_price - 1,
                        'direction': 'short'
                    })
                    continue
            new_open.append(pos)
        open_positions = new_open

        if len(open_positions) >= max_concurrent:
            continue

        entry_price = price_lookup.get((ticker, sig_date))
        if entry_price is None or entry_price <= 0:
            continue

        if ticker in ticker_data:
            df = ticker_data[ticker]
            future = df[df.index >= pd.Timestamp(sig_date)]
            if len(future) > hold_days:
                exit_dt = future.index[hold_days].date()
            elif len(future) > 1:
                exit_dt = future.index[-1].date()
            else:
                continue
        else:
            continue

        open_positions.append({
            'ticker': ticker,
            'entry_date': sig_date,
            'entry_price': entry_price,
            'exit_date': exit_dt,
            'size': pos_size
        })

    # Close remaining
    for pos in open_positions:
        exit_price = price_lookup.get((pos['ticker'], pos.get('exit_date')))
        if exit_price is None and pos['ticker'] in ticker_data:
            df = ticker_data[pos['ticker']]
            if len(df) > 0:
                exit_price = df.iloc[-1]['Close']
        if exit_price:
            pnl = (pos['entry_price'] / exit_price - 1) * pos['size'] - COMMISSION_RT
            equity += pnl
            trades.append({
                'entry_date': pos['entry_date'],
                'exit_date': pos.get('exit_date'),
                'ticker': pos['ticker'],
                'pnl': pnl,
                'return': pos['entry_price'] / exit_price - 1,
                'direction': 'short'
            })

    return compute_metrics(trades, account)


def backtest_rotation(allocation_signals, ticker_data, pos_size=POS_SIZE,
                      account=ACCOUNT):
    """
    Backtest for rotation strategies.
    allocation_signals: list of (date, {ticker: weight, ...})
    """
    trades = []
    equity = account
    current_holdings = {}  # ticker -> {entry_price, entry_date, weight}

    price_lookup = {}
    for ticker, df in ticker_data.items():
        for idx in df.index:
            dt = idx.date() if hasattr(idx, 'date') else idx
            price_lookup[(ticker, dt)] = df.loc[idx, 'Close']

    allocation_signals.sort(key=lambda x: x[0])

    for sig_date, new_alloc in allocation_signals:
        # Close positions not in new allocation
        for ticker in list(current_holdings.keys()):
            if ticker not in new_alloc:
                exit_price = price_lookup.get((ticker, sig_date))
                if exit_price and current_holdings[ticker]['entry_price'] > 0:
                    ret = exit_price / current_holdings[ticker]['entry_price'] - 1
                    sz = current_holdings[ticker]['weight'] * pos_size
                    pnl = ret * sz - COMMISSION_RT
                    equity += pnl
                    trades.append({
                        'entry_date': current_holdings[ticker]['entry_date'],
                        'exit_date': sig_date,
                        'ticker': ticker,
                        'pnl': pnl,
                        'return': ret
                    })
                del current_holdings[ticker]

        # Open new positions
        for ticker, weight in new_alloc.items():
            if ticker not in current_holdings:
                entry_price = price_lookup.get((ticker, sig_date))
                if entry_price and entry_price > 0:
                    current_holdings[ticker] = {
                        'entry_price': entry_price,
                        'entry_date': sig_date,
                        'weight': weight
                    }

    # Close all remaining
    for ticker, pos in current_holdings.items():
        if ticker in ticker_data:
            df = ticker_data[ticker]
            if len(df) > 0:
                exit_price = df.iloc[-1]['Close']
                ret = exit_price / pos['entry_price'] - 1
                sz = pos['weight'] * pos_size
                pnl = ret * sz - COMMISSION_RT
                equity += pnl
                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': df.index[-1].date(),
                    'ticker': ticker,
                    'pnl': pnl,
                    'return': ret
                })

    return compute_metrics(trades, account)


def compute_metrics(trades, account):
    """Compute strategy metrics from trades list."""
    if not trades:
        return {
            'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
            'max_dd': 1.0, 'trade_count': 0, 'total_return': 0,
            'trades': [], 'daily_returns': []
        }

    returns = [t['return'] for t in trades]
    pnls = [t['pnl'] for t in trades]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    win_rate = len(winners) / len(pnls) if pnls else 0
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 1e-10
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_pnl = sum(pnls)
    total_return = total_pnl / account

    # Compute equity curve for drawdown
    eq = account
    eq_curve = [account]
    for p in pnls:
        eq += p
        eq_curve.append(eq)

    eq_arr = np.array(eq_curve)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0

    # Sharpe / Sortino (annualized, assume ~1 trade per day average)
    ret_arr = np.array(returns)
    if len(ret_arr) > 1 and ret_arr.std() > 0:
        # Approximate annualization based on trade frequency
        avg_hold = 10  # approximate
        trades_per_year = 252 / avg_hold
        sharpe = (ret_arr.mean() / ret_arr.std()) * np.sqrt(trades_per_year)
        downside = ret_arr[ret_arr < 0]
        downside_std = downside.std() if len(downside) > 1 else ret_arr.std()
        sortino = (ret_arr.mean() / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else sharpe
    else:
        sharpe = 0
        sortino = 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(min(profit_factor, 99.0), 3),
        'max_dd': round(max_dd, 4),
        'trade_count': len(trades),
        'total_return': round(total_return, 4),
        'total_pnl': round(total_pnl, 2),
        'trades': trades,
        'daily_returns': returns
    }


# ─── PERMUTATION TEST ───────────────────────────────────────────────
def permutation_test(trades, all_ticker_data=None, n_perm=N_PERM):
    """
    Proper permutation test: shuffle SIGNAL DATES against price series.
    For each permutation, pick random entry dates from the same ticker's
    available trading days, compute forward returns, and compare Sharpe.
    """
    if len(trades) < 5:
        return 1.0

    returns = np.array([t['return'] for t in trades])
    actual_sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0

    if all_ticker_data is None:
        # Fallback: block bootstrap (shuffle blocks of returns)
        block_size = max(3, len(returns) // 10)
        count_better = 0
        for _ in range(n_perm):
            # Circular block bootstrap
            n = len(returns)
            perm_returns = []
            while len(perm_returns) < n:
                start = np.random.randint(0, n)
                block = [returns[(start + j) % n] for j in range(block_size)]
                perm_returns.extend(block)
            perm_returns = np.array(perm_returns[:n])
            np.random.shuffle(perm_returns)  # break any remaining order
            sh = perm_returns.mean() / perm_returns.std() if perm_returns.std() > 0 else 0
            if sh >= actual_sharpe:
                count_better += 1
        return count_better / n_perm

    # Build per-ticker forward return lookup
    ticker_fwd = {}
    for t in trades:
        ticker = t['ticker']
        if ticker not in ticker_fwd and ticker in all_ticker_data:
            df = all_ticker_data[ticker]
            close = df['Close'].values
            # Compute 10d forward returns for all available dates
            fwd = np.full(len(close), np.nan)
            hold = 10  # approximate
            for i in range(len(close) - hold):
                fwd[i] = close[i + hold] / close[i] - 1
            ticker_fwd[ticker] = fwd[~np.isnan(fwd)]

    # Group trades by ticker
    trades_by_ticker = defaultdict(int)
    for t in trades:
        trades_by_ticker[t['ticker']] += 1

    count_better = 0
    for _ in range(n_perm):
        perm_returns = []
        for ticker, n_trades in trades_by_ticker.items():
            if ticker in ticker_fwd and len(ticker_fwd[ticker]) > 0:
                # Random sample of forward returns from this ticker
                sampled = np.random.choice(ticker_fwd[ticker], size=n_trades, replace=True)
                perm_returns.extend(sampled.tolist())
            else:
                # Use original returns shuffled
                orig = [t['return'] for t in trades if t['ticker'] == ticker]
                np.random.shuffle(orig)
                perm_returns.extend(orig)

        perm_arr = np.array(perm_returns)
        if len(perm_arr) > 1 and perm_arr.std() > 0:
            sh = perm_arr.mean() / perm_arr.std()
        else:
            sh = 0
        if sh >= actual_sharpe:
            count_better += 1

    return count_better / n_perm


def regime_gap(trades, spy_data):
    """Compute regime gap: |Sharpe_bull - Sharpe_bear| / max."""
    if len(trades) < 10:
        return 1.0

    spy_returns = spy_data['Close'].pct_change()

    bull_trades = []
    bear_trades = []

    for t in trades:
        entry = t['entry_date']
        if isinstance(entry, str):
            entry = pd.Timestamp(entry).date()
        ts = pd.Timestamp(entry)
        if ts in spy_returns.index:
            if spy_returns.loc[ts] >= 0:
                bull_trades.append(t['return'])
            else:
                bear_trades.append(t['return'])
        else:
            # Find nearest
            idx = spy_returns.index.get_indexer([ts], method='nearest')[0]
            if idx >= 0 and spy_returns.iloc[idx] >= 0:
                bull_trades.append(t['return'])
            else:
                bear_trades.append(t['return'])

    if len(bull_trades) < 3 or len(bear_trades) < 3:
        return 1.0

    bull_arr = np.array(bull_trades)
    bear_arr = np.array(bear_trades)

    bull_sharpe = bull_arr.mean() / bull_arr.std() if bull_arr.std() > 0 else 0
    bear_sharpe = bear_arr.mean() / bear_arr.std() if bear_arr.std() > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    if max_abs == 0:
        return 1.0

    return abs(bull_sharpe - bear_sharpe) / max_abs


def validate_5gates(metrics, trades, spy_data, all_ticker_data=None):
    """Run 5-gate validation."""
    gates = {}
    gates['sharpe_pass'] = metrics['sharpe'] > 0.5
    gates['sharpe'] = metrics['sharpe']

    if len(trades) >= 5:
        p_val = permutation_test(trades, all_ticker_data=all_ticker_data)
    else:
        p_val = 1.0
    gates['perm_p'] = round(p_val, 4)
    gates['perm_pass'] = p_val < 0.05

    rg = regime_gap(trades, spy_data)
    gates['regime_gap'] = round(rg, 4)
    gates['regime_pass'] = rg < 0.50

    gates['mdd_pass'] = metrics['max_dd'] < 0.30
    gates['mdd'] = metrics['max_dd']

    gates['trades_pass'] = metrics['trade_count'] >= 20
    gates['trade_count'] = metrics['trade_count']

    gates['all_pass'] = all([
        gates['sharpe_pass'], gates['perm_pass'], gates['regime_pass'],
        gates['mdd_pass'], gates['trades_pass']
    ])

    return gates


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 1: MULTI-STRATEGY CONFLUENCE META-SIGNAL
# ═══════════════════════════════════════════════════════════════════
def strategy1_confluence(data, spy_data, vix_data, tlt_data):
    """Generate confluence signals for quality universe."""
    print("\n=== STRATEGY 1: Multi-Strategy Confluence ===")
    signals_by_stock = {}

    for ticker, df in data.items():
        if ticker not in QUALITY_UNIVERSE:
            continue
        if len(df) < 60:
            continue

        close = df['Close']
        high = df['High']
        low = df['Low']
        r = rsi(close, 14)
        sma20 = sma(close, 20)
        sma50 = sma(close, 50)
        high20 = close.rolling(20).max()
        consec_red = consecutive_red_days(close)
        rv = realized_vol_proxy(df, 20)
        hl_spread = high - low
        hl_avg60 = hl_spread.rolling(60).mean()
        weekly_rsi = rsi(close.rolling(5).mean(), 14)
        weekly_dd = drawdown_from_high(close, 5)

        for i in range(60, len(df)):
            dt = df.index[i].date()
            sig_count = 0
            sig_names = []

            # Signal_MR: >5% below 20d high + RSI<35
            if close.iloc[i] < high20.iloc[i] * 0.95 and r.iloc[i] < 35:
                sig_count += 1; sig_names.append('MR')

            # Signal_Recovery: first green day after 3+ red
            if i > 0 and consec_red.iloc[i] == 0 and consec_red.iloc[i-1] >= 3:
                sig_count += 1; sig_names.append('Recovery')

            # Signal_RSI_Div: RSI higher low while price lower low (20d lookback)
            if i >= 20:
                price_window = close.iloc[i-20:i+1]
                rsi_window = r.iloc[i-20:i+1]
                if (not price_window.isna().all() and not rsi_window.isna().all()):
                    price_min_idx = price_window.idxmin()
                    if price_min_idx == df.index[i]:  # current is price low
                        prev_lows = price_window.iloc[:-5]
                        if len(prev_lows) > 0:
                            prev_min_idx = prev_lows.idxmin()
                            if (close.iloc[i] < close.loc[prev_min_idx] and
                                r.iloc[i] > r.loc[prev_min_idx]):
                                sig_count += 1; sig_names.append('RSI_Div')

            # Signal_Bond: TLT rises >1% in 5d + stock >5% below 20-SMA
            if tlt_data is not None and len(tlt_data) > 5:
                ts = df.index[i]
                if ts in tlt_data.index:
                    tlt_idx = tlt_data.index.get_loc(ts)
                    if tlt_idx >= 5:
                        tlt_ret5 = tlt_data['Close'].iloc[tlt_idx] / tlt_data['Close'].iloc[tlt_idx-5] - 1
                        if tlt_ret5 > 0.01 and close.iloc[i] < sma20.iloc[i] * 0.95:
                            sig_count += 1; sig_names.append('Bond')

            # Signal_IV_RV: VIX > realized vol + 5pts
            if vix_data is not None:
                ts = df.index[i]
                if ts in vix_data.index:
                    vix_val = vix_data.loc[ts, 'Close']
                    rv_val = rv.iloc[i] * 100 if not np.isnan(rv.iloc[i]) else 999
                    if vix_val > rv_val + 5:
                        sig_count += 1; sig_names.append('IV_RV')

            # Signal_Liquidity: H-L spread < 60d avg
            if not np.isnan(hl_avg60.iloc[i]) and hl_spread.iloc[i] < hl_avg60.iloc[i]:
                sig_count += 1; sig_names.append('Liquidity')

            # Signal_MultiTF: daily RSI<35 + weekly RSI<40 + weekly DD>7%
            if (r.iloc[i] < 35 and
                not np.isnan(weekly_rsi.iloc[i]) and weekly_rsi.iloc[i] < 40 and
                not np.isnan(weekly_dd.iloc[i]) and weekly_dd.iloc[i] < -0.07):
                sig_count += 1; sig_names.append('MultiTF')

            if sig_count >= 2:
                if ticker not in signals_by_stock:
                    signals_by_stock[ticker] = []
                signals_by_stock[ticker].append((dt, sig_count, sig_names))

    return signals_by_stock


def run_strategy1(data, spy_data, vix_data, tlt_data):
    sigs = strategy1_confluence(data, spy_data, vix_data, tlt_data)
    results = {}

    # Flatten signals
    all_sigs = []
    for ticker, sig_list in sigs.items():
        for dt, count, names in sig_list:
            all_sigs.append((dt, ticker, count))

    # Variant A: >= 2 signals, hold 10d
    sig_a = [(dt, t) for dt, t, c in all_sigs if c >= 2]
    results['1A'] = backtest_signals(sig_a, data, hold_days=10)
    print(f"  1A (>=2, 10d): {results['1A']['trade_count']} trades, Sharpe={results['1A']['sharpe']}")

    # Variant B: >= 3 signals, hold 10d
    sig_b = [(dt, t) for dt, t, c in all_sigs if c >= 3]
    results['1B'] = backtest_signals(sig_b, data, hold_days=10)
    print(f"  1B (>=3, 10d): {results['1B']['trade_count']} trades, Sharpe={results['1B']['sharpe']}")

    # Variant C: >= 4 signals, hold 10d
    sig_c = [(dt, t) for dt, t, c in all_sigs if c >= 4]
    results['1C'] = backtest_signals(sig_c, data, hold_days=10)
    print(f"  1C (>=4, 10d): {results['1C']['trade_count']} trades, Sharpe={results['1C']['sharpe']}")

    # Variant D: >= 2 signals, scale by count
    sig_d = [(dt, t, 1.0 if c == 2 else 1.5 if c == 3 else 2.0) for dt, t, c in all_sigs if c >= 2]
    results['1D'] = backtest_signals(sig_d, data, hold_days=10)
    print(f"  1D (scaled, 10d): {results['1D']['trade_count']} trades, Sharpe={results['1D']['sharpe']}")

    # Variant E: >= 2 signals + below 50-SMA
    sig_e = []
    for dt, t, c in all_sigs:
        if c >= 2 and t in data:
            df = data[t]
            ts = pd.Timestamp(dt)
            if ts in df.index:
                idx = df.index.get_loc(ts)
                s50 = df['Close'].rolling(50).mean()
                if idx < len(s50) and not np.isnan(s50.iloc[idx]) and df['Close'].iloc[idx] < s50.iloc[idx]:
                    sig_e.append((dt, t))
    results['1E'] = backtest_signals(sig_e, data, hold_days=10)
    print(f"  1E (50SMA filter, 10d): {results['1E']['trade_count']} trades, Sharpe={results['1E']['sharpe']}")

    # Variant F: >= 3 signals, hold until RSI>50 or 15d max
    sig_f = [(dt, t) for dt, t, c in all_sigs if c >= 3]
    results['1F'] = backtest_signals(sig_f, data, hold_days=15)  # simplified: use 15d hold
    print(f"  1F (>=3, dynamic exit): {results['1F']['trade_count']} trades, Sharpe={results['1F']['sharpe']}")

    return results


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 2: LEVERAGED ETF DECAY EXPLOITATION
# ═══════════════════════════════════════════════════════════════════
def run_strategy2(lev_data, vix_data):
    print("\n=== STRATEGY 2: Leveraged ETF Decay ===")
    results = {}

    qqq = lev_data.get('QQQ')
    tqqq = lev_data.get('TQQQ')
    sqqq = lev_data.get('SQQQ')

    if qqq is None or tqqq is None or sqqq is None:
        print("  WARN: Missing leveraged ETF data")
        return {f'2{v}': compute_metrics([], ACCOUNT) for v in 'ABCDEF'}

    qqq_rsi = rsi(qqq['Close'], 14)
    sqqq_rsi = rsi(sqqq['Close'], 14)
    qqq_consec_red = consecutive_red_days(qqq['Close'])
    qqq_mom20 = qqq['Close'].pct_change(20)

    # 2A: Short SQQQ when VIX < 20, hold 20d
    sig_a = []
    for i in range(20, len(sqqq)):
        ts = sqqq.index[i]
        if ts in vix_data.index:
            if vix_data.loc[ts, 'Close'] < 20:
                sig_a.append((ts.date(), 'SQQQ'))
    # Subsample: only take signals every 20 days
    filtered_a = []
    last_date = None
    for dt, t in sig_a:
        if last_date is None or (dt - last_date).days >= 20:
            filtered_a.append((dt, t))
            last_date = dt
    results['2A'] = backtest_short_signals(filtered_a, lev_data, hold_days=20)
    print(f"  2A (Short SQQQ VIX<20): {results['2A']['trade_count']} trades, Sharpe={results['2A']['sharpe']}")

    # 2B: Long TQQQ when QQQ RSI<30, sell RSI>50 (approx 10d hold)
    sig_b = []
    for i in range(20, len(qqq)):
        if qqq_rsi.iloc[i] < 30:
            sig_b.append((qqq.index[i].date(), 'TQQQ'))
    results['2B'] = backtest_signals(sig_b, lev_data, hold_days=10)
    print(f"  2B (Long TQQQ RSI<30): {results['2B']['trade_count']} trades, Sharpe={results['2B']['sharpe']}")

    # 2C: Pairs long TQQQ + short SQQQ, monthly rebalance
    sig_c = []
    for i in range(0, len(tqqq), 21):  # ~monthly
        dt = tqqq.index[i].date()
        sig_c.append((dt, {'TQQQ': 0.5}))  # long only component (simplified)
    results['2C'] = backtest_rotation(sig_c, lev_data)
    print(f"  2C (Pairs TQQQ/SQQQ): {results['2C']['trade_count']} trades, Sharpe={results['2C']['sharpe']}")

    # 2D: Short SQQQ when SQQQ RSI>60
    sig_d = []
    for i in range(20, len(sqqq)):
        if sqqq_rsi.iloc[i] > 60:
            sig_d.append((sqqq.index[i].date(), 'SQQQ'))
    # Subsample every 10d
    filtered_d = []
    last_date = None
    for dt, t in sig_d:
        if last_date is None or (dt - last_date).days >= 10:
            filtered_d.append((dt, t))
            last_date = dt
    results['2D'] = backtest_short_signals(filtered_d, lev_data, hold_days=10)
    print(f"  2D (Short SQQQ RSI>60): {results['2D']['trade_count']} trades, Sharpe={results['2D']['sharpe']}")

    # 2E: Long TQQQ after 3+ QQQ red days, hold 5d
    sig_e = []
    for i in range(20, len(qqq)):
        if qqq_consec_red.iloc[i] >= 3:
            sig_e.append((qqq.index[i].date(), 'TQQQ'))
    results['2E'] = backtest_signals(sig_e, lev_data, hold_days=5)
    print(f"  2E (Long TQQQ 3+ red days): {results['2E']['trade_count']} trades, Sharpe={results['2E']['sharpe']}")

    # 2F: Rotate TQQQ/SQQQ on 20d QQQ momentum
    sig_f = []
    for i in range(21, len(qqq), 21):
        dt = qqq.index[i].date()
        if not np.isnan(qqq_mom20.iloc[i]):
            if qqq_mom20.iloc[i] > 0:
                sig_f.append((dt, {'TQQQ': 1.0}))
            else:
                sig_f.append((dt, {'SQQQ': 1.0}))
    results['2F'] = backtest_rotation(sig_f, lev_data)
    print(f"  2F (Rotate TQQQ/SQQQ): {results['2F']['trade_count']} trades, Sharpe={results['2F']['sharpe']}")

    return results


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 3: SPY/INDEX MEAN REVERSION
# ═══════════════════════════════════════════════════════════════════
def run_strategy3(idx_data, vix_data):
    print("\n=== STRATEGY 3: SPY/Index Mean Reversion ===")
    results = {}

    spy = idx_data.get('SPY')
    qqq = idx_data.get('QQQ')
    if spy is None:
        return {f'3{v}': compute_metrics([], ACCOUNT) for v in 'ABCDEF'}

    spy_rsi = rsi(spy['Close'], 14)
    spy_sma20 = sma(spy['Close'], 20)
    spy_consec_red = consecutive_red_days(spy['Close'])
    spy_ret5 = spy['Close'].pct_change(5)

    # 3A: Buy SPY RSI<30, hold 10d
    sig_a = []
    for i in range(20, len(spy)):
        if spy_rsi.iloc[i] < 30:
            sig_a.append((spy.index[i].date(), 'SPY'))
    results['3A'] = backtest_signals(sig_a, idx_data, hold_days=10)
    print(f"  3A (SPY RSI<30): {results['3A']['trade_count']} trades, Sharpe={results['3A']['sharpe']}")

    # 3B: SPY >3% below 20-SMA + RSI<35, hold 10d
    sig_b = []
    for i in range(20, len(spy)):
        if (not np.isnan(spy_sma20.iloc[i]) and
            spy['Close'].iloc[i] < spy_sma20.iloc[i] * 0.97 and
            spy_rsi.iloc[i] < 35):
            sig_b.append((spy.index[i].date(), 'SPY'))
    results['3B'] = backtest_signals(sig_b, idx_data, hold_days=10)
    print(f"  3B (SPY -3% SMA + RSI<35): {results['3B']['trade_count']} trades, Sharpe={results['3B']['sharpe']}")

    # 3C: SPY 3+ red days with >3% total drop, hold 5d
    sig_c = []
    for i in range(20, len(spy)):
        if spy_consec_red.iloc[i] >= 3:
            ret_3d = spy['Close'].iloc[i] / spy['Close'].iloc[i-3] - 1
            if ret_3d < -0.03:
                sig_c.append((spy.index[i].date(), 'SPY'))
    results['3C'] = backtest_signals(sig_c, idx_data, hold_days=5)
    print(f"  3C (SPY 3+ red >3%): {results['3C']['trade_count']} trades, Sharpe={results['3C']['sharpe']}")

    # 3D: QQQ RSI<30 + VIX>25, hold 10d
    if qqq is not None:
        qqq_rsi = rsi(qqq['Close'], 14)
        sig_d = []
        for i in range(20, len(qqq)):
            ts = qqq.index[i]
            if ts in vix_data.index:
                if qqq_rsi.iloc[i] < 30 and vix_data.loc[ts, 'Close'] > 25:
                    sig_d.append((ts.date(), 'QQQ'))
        results['3D'] = backtest_signals(sig_d, idx_data, hold_days=10)
    else:
        results['3D'] = compute_metrics([], ACCOUNT)
    print(f"  3D (QQQ RSI<30 + VIX>25): {results['3D']['trade_count']} trades, Sharpe={results['3D']['sharpe']}")

    # 3E: SPY intraday drop >2% (H-to-C), overnight bounce (hold 1d)
    sig_e = []
    for i in range(20, len(spy)):
        intraday_drop = (spy['High'].iloc[i] - spy['Close'].iloc[i]) / spy['High'].iloc[i]
        if intraday_drop > 0.02:
            sig_e.append((spy.index[i].date(), 'SPY'))
    results['3E'] = backtest_signals(sig_e, idx_data, hold_days=1)
    print(f"  3E (SPY intraday drop >2%): {results['3E']['trade_count']} trades, Sharpe={results['3E']['sharpe']}")

    # 3F: SPY 5d ret <-4% + VIX spike >20% from 5d ago, hold 10d
    sig_f = []
    vix_ret5 = vix_data['Close'].pct_change(5) if vix_data is not None else None
    for i in range(20, len(spy)):
        if not np.isnan(spy_ret5.iloc[i]) and spy_ret5.iloc[i] < -0.04:
            ts = spy.index[i]
            if vix_ret5 is not None and ts in vix_ret5.index:
                if vix_ret5.loc[ts] > 0.20:
                    sig_f.append((ts.date(), 'SPY'))
    results['3F'] = backtest_signals(sig_f, idx_data, hold_days=10)
    print(f"  3F (SPY -4%/5d + VIX spike): {results['3F']['trade_count']} trades, Sharpe={results['3F']['sharpe']}")

    return results


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 4: FACTOR ETF ROTATION
# ═══════════════════════════════════════════════════════════════════
def run_strategy4(factor_data, vix_data):
    print("\n=== STRATEGY 4: Factor ETF Rotation ===")
    results = {}

    avail_factors = [t for t in FACTOR_ETFS if t in factor_data]
    if len(avail_factors) < 3:
        print("  WARN: Not enough factor ETFs")
        return {f'4{v}': compute_metrics([], ACCOUNT) for v in 'ABCDEF'}

    # Build aligned returns
    factor_closes = {}
    for t in avail_factors:
        factor_closes[t] = factor_data[t]['Close']

    # Common index
    common_idx = factor_closes[avail_factors[0]].index
    for t in avail_factors[1:]:
        common_idx = common_idx.intersection(factor_closes[t].index)
    common_idx = common_idx.sort_values()

    # 4A: Top 2 by 60d momentum, monthly rebalance
    sig_a = []
    for i in range(63, len(common_idx), 21):
        dt = common_idx[i].date()
        rets = {}
        for t in avail_factors:
            r60 = factor_closes[t].loc[common_idx[i]] / factor_closes[t].loc[common_idx[i-60]] - 1
            rets[t] = r60
        top2 = sorted(rets, key=rets.get, reverse=True)[:2]
        alloc = {t: 0.5 for t in top2}
        sig_a.append((dt, alloc))
    results['4A'] = backtest_rotation(sig_a, factor_data)
    print(f"  4A (Top 2 momentum): {results['4A']['trade_count']} trades, Sharpe={results['4A']['sharpe']}")

    # 4B: Bottom 2 by 20d return, hold 20d
    sig_b = []
    for i in range(25, len(common_idx), 21):
        dt = common_idx[i].date()
        rets = {}
        for t in avail_factors:
            r20 = factor_closes[t].loc[common_idx[i]] / factor_closes[t].loc[common_idx[i-20]] - 1
            rets[t] = r20
        bot2 = sorted(rets, key=rets.get)[:2]
        alloc = {t: 0.5 for t in bot2}
        sig_b.append((dt, alloc))
    results['4B'] = backtest_rotation(sig_b, factor_data)
    print(f"  4B (Bottom 2 MR): {results['4B']['trade_count']} trades, Sharpe={results['4B']['sharpe']}")

    # 4C: USMV when VIX>25, MTUM when VIX<18
    sig_c = []
    last_alloc = None
    for i in range(20, len(common_idx), 5):  # weekly check
        ts = common_idx[i]
        dt = ts.date()
        if ts in vix_data.index:
            vix_val = vix_data.loc[ts, 'Close']
            if vix_val > 25 and 'USMV' in avail_factors:
                new_alloc = {'USMV': 1.0}
            elif vix_val < 18 and 'MTUM' in avail_factors:
                new_alloc = {'MTUM': 1.0}
            else:
                continue
            if new_alloc != last_alloc:
                sig_c.append((dt, new_alloc))
                last_alloc = new_alloc
    results['4C'] = backtest_rotation(sig_c, factor_data)
    print(f"  4C (VIX regime): {results['4C']['trade_count']} trades, Sharpe={results['4C']['sharpe']}")

    # 4D: Highest 20d Sharpe, monthly rotation
    sig_d = []
    for i in range(25, len(common_idx), 21):
        dt = common_idx[i].date()
        sharpes = {}
        for t in avail_factors:
            rets = factor_closes[t].pct_change().loc[common_idx[i-20]:common_idx[i]]
            if len(rets) > 5 and rets.std() > 0:
                sharpes[t] = rets.mean() / rets.std()
            else:
                sharpes[t] = 0
        best = max(sharpes, key=sharpes.get)
        sig_d.append((dt, {best: 1.0}))
    results['4D'] = backtest_rotation(sig_d, factor_data)
    print(f"  4D (Best 20d Sharpe): {results['4D']['trade_count']} trades, Sharpe={results['4D']['sharpe']}")

    # 4E: Equal weight all, monthly rebalance (benchmark)
    sig_e = []
    w = 1.0 / len(avail_factors)
    for i in range(5, len(common_idx), 21):
        dt = common_idx[i].date()
        alloc = {t: w for t in avail_factors}
        sig_e.append((dt, alloc))
    results['4E'] = backtest_rotation(sig_e, factor_data)
    print(f"  4E (Equal weight): {results['4E']['trade_count']} trades, Sharpe={results['4E']['sharpe']}")

    # 4F: Dual momentum top 2 if >0 else cash
    sig_f = []
    for i in range(63, len(common_idx), 21):
        dt = common_idx[i].date()
        rets = {}
        for t in avail_factors:
            r60 = factor_closes[t].loc[common_idx[i]] / factor_closes[t].loc[common_idx[i-60]] - 1
            if r60 > 0:
                rets[t] = r60
        if rets:
            top2 = sorted(rets, key=rets.get, reverse=True)[:2]
            alloc = {t: 1.0 / len(top2) for t in top2}
            sig_f.append((dt, alloc))
        # else: cash (no allocation)
    results['4F'] = backtest_rotation(sig_f, factor_data)
    print(f"  4F (Dual momentum): {results['4F']['trade_count']} trades, Sharpe={results['4F']['sharpe']}")

    return results


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 5: MEAN REVERSION ON VOLATILE STOCKS
# ═══════════════════════════════════════════════════════════════════
def run_strategy5(vol_data):
    print("\n=== STRATEGY 5: Volatile Stock Mean Reversion ===")
    results = {}

    for variant, label, cond_fn, hold in [
        ('5A', 'RSI<25 + -10% from 20d high, 10d', None, 10),
        ('5B', 'RSI<20, 5d', None, 5),
        ('5C', '5+ red days, 5d', None, 5),
        ('5D', '-20% from 50d high + vol spike, 10d', None, 10),
        ('5E', 'BB %B<0 + RSI<30, 10d', None, 10),
        ('5F', '-15% from 20d high + RSI<25 + green, 10d', None, 10),
    ]:
        sigs = []
        for ticker, df in vol_data.items():
            if ticker not in VOLATILE_UNIVERSE or len(df) < 60:
                continue
            close = df['Close']
            r = rsi(close, 14)
            high20 = close.rolling(20).max()
            high50 = close.rolling(50).max()
            consec = consecutive_red_days(close)
            bb = bollinger_pctb(close)
            vol = df['Volume']
            vol_avg = vol.rolling(20).mean()

            for i in range(60, len(df)):
                dt = df.index[i].date()

                if variant == '5A':
                    if r.iloc[i] < 25 and close.iloc[i] < high20.iloc[i] * 0.90:
                        sigs.append((dt, ticker))
                elif variant == '5B':
                    if r.iloc[i] < 20:
                        sigs.append((dt, ticker))
                elif variant == '5C':
                    if consec.iloc[i] >= 5:
                        sigs.append((dt, ticker))
                elif variant == '5D':
                    if (not np.isnan(high50.iloc[i]) and
                        close.iloc[i] < high50.iloc[i] * 0.80 and
                        vol.iloc[i] > vol_avg.iloc[i] * 2):
                        sigs.append((dt, ticker))
                elif variant == '5E':
                    if (not np.isnan(bb.iloc[i]) and bb.iloc[i] < 0 and r.iloc[i] < 30):
                        sigs.append((dt, ticker))
                elif variant == '5F':
                    if (close.iloc[i] < high20.iloc[i] * 0.85 and
                        r.iloc[i] < 25 and
                        i > 0 and close.iloc[i] > close.iloc[i-1]):  # green day
                        sigs.append((dt, ticker))

        results[variant] = backtest_signals(sigs, vol_data, hold_days=hold)
        print(f"  {variant} ({label}): {results[variant]['trade_count']} trades, Sharpe={results[variant]['sharpe']}")

    return results


# ═══════════════════════════════════════════════════════════════════
# STRATEGY 6: VIX TERM STRUCTURE
# ═══════════════════════════════════════════════════════════════════
def run_strategy6(vix_etf_data, vix_data, spy_data):
    print("\n=== STRATEGY 6: VIX Term Structure ===")
    results = {}

    vixy = vix_etf_data.get('VIXY')
    svxy = vix_etf_data.get('SVXY')
    spy = spy_data

    if vixy is None:
        print("  WARN: VIXY data missing")
        return {f'6{v}': compute_metrics([], ACCOUNT) for v in 'ABCDEF'}

    vix_sma20 = sma(vix_data['Close'], 20)
    vixy_rsi = rsi(vixy['Close'], 14)

    # 6A: Short VIXY when VIX < 20d avg (contango proxy), hold 20d
    sig_a = []
    last_date = None
    for i in range(25, len(vixy)):
        ts = vixy.index[i]
        if ts in vix_data.index and ts in vix_sma20.index:
            vix_val = vix_data.loc[ts, 'Close']
            if not np.isnan(vix_sma20.loc[ts]):
                if vix_val < vix_sma20.loc[ts]:
                    dt = ts.date()
                    if last_date is None or (dt - last_date).days >= 20:
                        sig_a.append((dt, 'VIXY'))
                        last_date = dt
    results['6A'] = backtest_short_signals(sig_a, vix_etf_data, hold_days=20)
    print(f"  6A (Short VIXY contango): {results['6A']['trade_count']} trades, Sharpe={results['6A']['sharpe']}")

    # 6B: Long VIXY when VIX > 20d avg by 20%, sell when drops below
    sig_b = []
    for i in range(25, len(vixy)):
        ts = vixy.index[i]
        if ts in vix_data.index and ts in vix_sma20.index:
            vix_val = vix_data.loc[ts, 'Close']
            avg = vix_sma20.loc[ts]
            if not np.isnan(avg) and vix_val > avg * 1.20:
                sig_b.append((ts.date(), 'VIXY'))
    results['6B'] = backtest_signals(sig_b, vix_etf_data, hold_days=10)
    print(f"  6B (Long VIXY backwardation): {results['6B']['trade_count']} trades, Sharpe={results['6B']['sharpe']}")

    # 6C: Short VIXY continuously, flat when VIX>30
    sig_c = []
    last_date = None
    for i in range(25, len(vixy)):
        ts = vixy.index[i]
        if ts in vix_data.index:
            if vix_data.loc[ts, 'Close'] <= 30:
                dt = ts.date()
                if last_date is None or (dt - last_date).days >= 20:
                    sig_c.append((dt, 'VIXY'))
                    last_date = dt
    results['6C'] = backtest_short_signals(sig_c, vix_etf_data, hold_days=20)
    print(f"  6C (Short VIXY <VIX30): {results['6C']['trade_count']} trades, Sharpe={results['6C']['sharpe']}")

    # 6D: Long SVXY when VIX<20, exit VIX>25
    if svxy is not None:
        sig_d = []
        last_date = None
        for i in range(25, len(svxy)):
            ts = svxy.index[i]
            if ts in vix_data.index:
                if vix_data.loc[ts, 'Close'] < 20:
                    dt = ts.date()
                    if last_date is None or (dt - last_date).days >= 15:
                        sig_d.append((dt, 'SVXY'))
                        last_date = dt
        results['6D'] = backtest_signals(sig_d, vix_etf_data, hold_days=15)
    else:
        results['6D'] = compute_metrics([], ACCOUNT)
    print(f"  6D (Long SVXY VIX<20): {results['6D']['trade_count']} trades, Sharpe={results['6D']['sharpe']}")

    # 6E: Mean reversion on VIXY: buy when VIXY RSI<20, hold 5d
    sig_e = []
    for i in range(20, len(vixy)):
        if vixy_rsi.iloc[i] < 20:
            sig_e.append((vixy.index[i].date(), 'VIXY'))
    results['6E'] = backtest_signals(sig_e, vix_etf_data, hold_days=5)
    print(f"  6E (VIXY RSI<20 MR): {results['6E']['trade_count']} trades, Sharpe={results['6E']['sharpe']}")

    # 6F: Pairs short VIXY + long SPY, weekly rebalance
    combined_data = {**vix_etf_data}
    if 'SPY' not in combined_data and spy is not None:
        combined_data['SPY'] = spy
    sig_f = []
    for i in range(5, min(len(vixy), len(spy) if spy is not None else 9999), 5):
        ts = vixy.index[i]
        dt = ts.date()
        sig_f.append((dt, {'SPY': 1.0}))  # long SPY component (simplified)
    results['6F'] = backtest_rotation(sig_f, combined_data)
    print(f"  6F (Pairs VIXY/SPY): {results['6F']['trade_count']} trades, Sharpe={results['6F']['sharpe']}")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("NOVEL BATCH 2 — 6 Strategies x 6 Variants = 36 Backtests")
    print("=" * 70)

    # Load all required data
    all_tickers = list(set(
        QUALITY_UNIVERSE + VOLATILE_UNIVERSE + FACTOR_ETFS +
        LEVERAGED + ['SPY', 'QQQ', '^VIX', 'TLT', 'VIXY', 'SVXY']
    ))

    print(f"\nLoading data for {len(all_tickers)} tickers...")
    all_data = load_data(all_tickers)
    print(f"  Loaded {len(all_data)} tickers successfully")

    spy_data = all_data.get('SPY')
    vix_data = all_data.get('^VIX')
    tlt_data = all_data.get('TLT')

    if spy_data is None or vix_data is None:
        print("FATAL: SPY or VIX data missing")
        return

    # Run all strategies
    all_results = {}

    # Strategy 1: Confluence
    quality_data = {t: all_data[t] for t in QUALITY_UNIVERSE if t in all_data}
    r1 = run_strategy1(quality_data, spy_data, vix_data, tlt_data)
    all_results.update(r1)

    # Strategy 2: Leveraged ETF
    lev_data = {t: all_data[t] for t in LEVERAGED if t in all_data}
    r2 = run_strategy2(lev_data, vix_data)
    all_results.update(r2)

    # Strategy 3: Index MR
    idx_data = {t: all_data[t] for t in ['SPY', 'QQQ'] if t in all_data}
    r3 = run_strategy3(idx_data, vix_data)
    all_results.update(r3)

    # Strategy 4: Factor Rotation
    factor_data = {t: all_data[t] for t in FACTOR_ETFS if t in all_data}
    r4 = run_strategy4(factor_data, vix_data)
    all_results.update(r4)

    # Strategy 5: Volatile MR
    vol_data = {t: all_data[t] for t in VOLATILE_UNIVERSE if t in all_data}
    r5 = run_strategy5(vol_data)
    all_results.update(r5)

    # Strategy 6: VIX Term Structure
    vix_etf_data = {t: all_data[t] for t in ['VIXY', 'SVXY', 'SPY'] if t in all_data}
    r6 = run_strategy6(vix_etf_data, vix_data, spy_data)
    all_results.update(r6)

    # ─── VALIDATION ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5-GATE VALIDATION RESULTS")
    print("=" * 70)

    strategy_names = {
        '1': 'CONFLUENCE META-SIGNAL',
        '2': 'LEVERAGED ETF DECAY',
        '3': 'INDEX MEAN REVERSION',
        '4': 'FACTOR ETF ROTATION',
        '5': 'VOLATILE STOCK MR',
        '6': 'VIX TERM STRUCTURE'
    }

    variant_descriptions = {
        '1A': '>=2 signals, 10d', '1B': '>=3 signals, 10d', '1C': '>=4 signals, 10d',
        '1D': 'scaled size, 10d', '1E': '50-SMA filter, 10d', '1F': '>=3, dynamic exit',
        '2A': 'Short SQQQ VIX<20', '2B': 'Long TQQQ RSI<30', '2C': 'Pairs TQQQ/SQQQ',
        '2D': 'Short SQQQ RSI>60', '2E': 'TQQQ 3+ red days', '2F': 'Rotate TQQQ/SQQQ',
        '3A': 'SPY RSI<30', '3B': 'SPY -3%SMA+RSI<35', '3C': 'SPY 3+ red >3%',
        '3D': 'QQQ RSI<30+VIX>25', '3E': 'SPY intraday drop', '3F': 'SPY -4%+VIX spike',
        '4A': 'Top 2 momentum', '4B': 'Bottom 2 MR', '4C': 'VIX regime switch',
        '4D': 'Best 20d Sharpe', '4E': 'Equal weight', '4F': 'Dual momentum',
        '5A': 'RSI<25 -10%, 10d', '5B': 'RSI<20, 5d', '5C': '5+ red days, 5d',
        '5D': '-20% + vol spike', '5E': 'BB+RSI, 10d', '5F': '-15% + recovery',
        '6A': 'Short VIXY contango', '6B': 'Long VIXY backwdn', '6C': 'Short VIXY <VIX30',
        '6D': 'Long SVXY VIX<20', '6E': 'VIXY RSI<20 MR', '6F': 'Pairs VIXY/SPY'
    }

    output = {
        'run_date': datetime.now().isoformat(),
        'config': {
            'start': START, 'end': END, 'account': ACCOUNT,
            'pos_size': POS_SIZE, 'max_concurrent': MAX_CONCURRENT,
            'commission_rt': COMMISSION_RT, 'n_permutations': N_PERM
        },
        'results': {}
    }

    passes = []
    fails = []

    header = f"{'Variant':<8} {'Description':<25} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD':>7} {'#Trades':>7} {'TotRet':>8} {'Perm_p':>7} {'RGap':>6} {'Result':>12}"
    print(header)
    print("-" * len(header))

    for key in sorted(all_results.keys()):
        m = all_results[key]
        trades = m.get('trades', [])
        gates = validate_5gates(m, trades, spy_data, all_ticker_data=all_data)

        strat_num = key[0]
        desc = variant_descriptions.get(key, '')

        status = "PASS" if gates['all_pass'] else "FAIL"
        fail_reasons = []
        if not gates['sharpe_pass']: fail_reasons.append('Sharpe')
        if not gates['perm_pass']: fail_reasons.append('Perm')
        if not gates['regime_pass']: fail_reasons.append('Regime')
        if not gates['mdd_pass']: fail_reasons.append('MDD')
        if not gates['trades_pass']: fail_reasons.append('Trades')

        if gates['all_pass']:
            status_str = "PASS ***"
            passes.append(key)
        else:
            status_str = f"FAIL({','.join(fail_reasons)})"
            fails.append(key)

        print(f"{key:<8} {desc:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['win_rate']:>6.1%} {m['profit_factor']:>6.2f} {m['max_dd']:>7.2%} "
              f"{m['trade_count']:>7d} {m['total_return']:>8.2%} "
              f"{gates['perm_p']:>7.4f} {gates['regime_gap']:>6.3f} {status_str:>12}")

        # Save to output (without raw trades for JSON size)
        output['results'][key] = {
            'strategy': strategy_names.get(strat_num, ''),
            'variant': desc,
            'metrics': {
                'sharpe': m['sharpe'], 'sortino': m['sortino'],
                'win_rate': m['win_rate'], 'profit_factor': m['profit_factor'],
                'max_dd': m['max_dd'], 'trade_count': m['trade_count'],
                'total_return': m['total_return'],
                'total_pnl': m.get('total_pnl', 0)
            },
            'gates': {
                'sharpe_pass': gates['sharpe_pass'],
                'perm_pass': gates['perm_pass'], 'perm_p': gates['perm_p'],
                'regime_pass': gates['regime_pass'], 'regime_gap': gates['regime_gap'],
                'mdd_pass': gates['mdd_pass'],
                'trades_pass': gates['trades_pass'],
                'all_pass': gates['all_pass']
            }
        }

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(passes)}/{len(all_results)} PASSED all 5 gates")
    if passes:
        print(f"  PASSES: {', '.join(passes)} — ADVERSARIAL AUDIT NEEDED")
    print(f"  FAILS:  {', '.join(fails)}")
    print("=" * 70)

    # Save results
    out_path = "/home/jupiter/Lvl3Quant/data/novel_batch2_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
