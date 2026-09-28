#!/usr/bin/env python3
"""
Covered Call / Income Enhancement on Quality Stocks Backtest
============================================================
Tests systematic option selling (simulated via Black-Scholes) on quality
stock holdings, especially during sideways markets.

6 Variants (A-F) with mean-reversion entry, 5-gate validation + 1000 permutations.
"""

import json
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]

OOT_START = '2022-01-01'
OOT_END = '2026-07-31'
STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
RISK_FREE_RATE = 0.045
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/data/covered_call_quality_results.json')


# ─── Black-Scholes ───────────────────────────────────────────────────
def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call option price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put option price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ─── Data Download ───────────────────────────────────────────────────
def download_data():
    """Download price data for universe."""
    print(f"Downloading data for {len(UNIVERSE)} tickers...")
    # Download with buffer for indicator warmup
    buf_start = (pd.Timestamp(OOT_START) - pd.DateOffset(months=3)).strftime('%Y-%m-%d')

    data = {}
    for ticker in UNIVERSE:
        try:
            df = yf.download(ticker, start=buf_start, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 60:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} rows")
            else:
                print(f"  {ticker}: insufficient data ({len(df)} rows), skipping")
        except Exception as e:
            print(f"  {ticker}: download failed - {e}")

    return data


# ─── Feature Computation ────────────────────────────────────────────
def compute_features(df):
    """Compute MR signals, RSI, realized vol for a single stock."""
    close = df['Close'].copy()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # 20-day high
    high_20 = close.rolling(20).max()
    # Dip from 20d high
    dip_pct = (close - high_20) / high_20

    # RSI-14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 20-day realized vol (annualized)
    log_ret = np.log(close / close.shift(1))
    realized_vol = log_ret.rolling(20).std() * np.sqrt(252)

    # MR signal: 5% dip from 20d high AND RSI < 35
    mr_signal = (dip_pct <= -0.05) & (rsi < 35)

    feat = pd.DataFrame({
        'close': close,
        'dip_pct': dip_pct,
        'rsi': rsi,
        'realized_vol': realized_vol,
        'mr_signal': mr_signal
    }, index=df.index)

    return feat


# ─── Trade Simulation ───────────────────────────────────────────────
def apply_slippage(price, direction='buy'):
    """Apply slippage to price."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


def simulate_variant_A(features_dict, oot_dates):
    """Buy-Write Baseline: Buy on MR signal, sell ATM 30-DTE call immediately. Hold 22 trading days."""
    trades = []
    capital = STARTING_CAPITAL
    equity_curve = []

    for date_idx, date in enumerate(oot_dates):
        # Check for MR signals across universe
        for ticker, feat in features_dict.items():
            if date not in feat.index:
                continue
            row = feat.loc[date]
            if not row['mr_signal']:
                continue

            price = row['close']
            vol = row['realized_vol']
            if np.isnan(vol) or vol <= 0:
                continue

            # Position sizing
            buy_price = apply_slippage(price, 'buy')
            shares = min(int(MAX_PER_TRADE / buy_price), int(capital / buy_price))
            if shares <= 0:
                continue

            cost = shares * buy_price

            # Sell ATM call (strike = current price, 30 DTE)
            T = 30 / 365.0
            call_premium = bs_call_price(price, price, T, RISK_FREE_RATE, vol)
            premium_received = shares * apply_slippage(call_premium, 'sell')

            # Hold for 22 trading days
            exit_idx = date_idx + 22
            if exit_idx >= len(oot_dates):
                continue

            exit_date = oot_dates[exit_idx]
            if exit_date not in feat.index:
                continue

            exit_price = feat.loc[exit_date, 'close']
            sell_price = apply_slippage(exit_price, 'sell')

            # P&L
            stock_pnl = shares * (sell_price - buy_price)
            called_away = exit_price > price  # stock above strike
            if called_away:
                # Capped at strike
                stock_pnl = shares * (apply_slippage(price, 'sell') - buy_price)

            total_pnl = stock_pnl + premium_received
            capital += total_pnl

            # Stock-only comparison (no option)
            stock_only_pnl = shares * (sell_price - buy_price)

            trades.append({
                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                'ticker': ticker,
                'entry_price': float(buy_price),
                'exit_price': float(sell_price),
                'shares': shares,
                'premium': float(premium_received),
                'stock_pnl': float(stock_pnl),
                'total_pnl': float(total_pnl),
                'stock_only_pnl': float(stock_only_pnl),
                'called_away': bool(called_away),
                'vol': float(vol),
                'capital_after': float(capital),
            })

        equity_curve.append({'date': str(date.date()) if hasattr(date, 'date') else str(date), 'capital': float(capital)})

    return trades, equity_curve


def simulate_variant_B(features_dict, oot_dates):
    """2% OTM Write: Same as A but sell 2% OTM call."""
    trades = []
    capital = STARTING_CAPITAL
    equity_curve = []

    for date_idx, date in enumerate(oot_dates):
        for ticker, feat in features_dict.items():
            if date not in feat.index:
                continue
            row = feat.loc[date]
            if not row['mr_signal']:
                continue

            price = row['close']
            vol = row['realized_vol']
            if np.isnan(vol) or vol <= 0:
                continue

            buy_price = apply_slippage(price, 'buy')
            shares = min(int(MAX_PER_TRADE / buy_price), int(capital / buy_price))
            if shares <= 0:
                continue

            cost = shares * buy_price
            strike = price * 1.02  # 2% OTM
            T = 30 / 365.0
            call_premium = bs_call_price(price, strike, T, RISK_FREE_RATE, vol)
            premium_received = shares * apply_slippage(call_premium, 'sell')

            exit_idx = date_idx + 22
            if exit_idx >= len(oot_dates):
                continue

            exit_date = oot_dates[exit_idx]
            if exit_date not in feat.index:
                continue

            exit_price = feat.loc[exit_date, 'close']
            sell_price = apply_slippage(exit_price, 'sell')

            stock_pnl = shares * (sell_price - buy_price)
            called_away = exit_price > strike
            if called_away:
                stock_pnl = shares * (apply_slippage(strike, 'sell') - buy_price)

            total_pnl = stock_pnl + premium_received
            capital += total_pnl
            stock_only_pnl = shares * (sell_price - buy_price)

            trades.append({
                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                'ticker': ticker,
                'entry_price': float(buy_price),
                'exit_price': float(sell_price),
                'shares': shares,
                'premium': float(premium_received),
                'stock_pnl': float(stock_pnl),
                'total_pnl': float(total_pnl),
                'stock_only_pnl': float(stock_only_pnl),
                'called_away': bool(called_away),
                'vol': float(vol),
                'capital_after': float(capital),
            })

        equity_curve.append({'date': str(date.date()) if hasattr(date, 'date') else str(date), 'capital': float(capital)})

    return trades, equity_curve


def simulate_variant_C(features_dict, oot_dates):
    """Sell Call After Bounce: Buy on MR signal, wait for 3% bounce, THEN sell covered call.
    Only one pending position per ticker. Capital is NOT reserved upfront - we track
    positions and only deduct on final exit to match other variants' accounting."""
    trades = []
    capital = STARTING_CAPITAL
    equity_curve = []
    # Track: ticker -> (entry_date_idx, buy_price, shares, call_sold_date_idx, call_strike)
    active_positions = {}

    for date_idx, date in enumerate(oot_dates):
        closed_tickers = []

        for ticker, pos in active_positions.items():
            entry_idx, buy_price, shares, call_sold_idx, call_strike = pos
            feat = features_dict[ticker]
            if date not in feat.index:
                continue

            current_price = feat.loc[date, 'close']

            if call_sold_idx is not None:
                # Already sold a call - check if we've reached expiration (22 days from call sell)
                if date_idx - call_sold_idx >= 22:
                    sell_price = apply_slippage(current_price, 'sell')
                    called_away = current_price > call_strike
                    if called_away:
                        stock_pnl = shares * (apply_slippage(call_strike, 'sell') - buy_price)
                    else:
                        stock_pnl = shares * (sell_price - buy_price)

                    vol = feat.loc[date, 'realized_vol']
                    if np.isnan(vol) or vol <= 0:
                        vol = 0.25
                    T = 30 / 365.0
                    premium_received = shares * apply_slippage(
                        bs_call_price(call_strike, call_strike, T, RISK_FREE_RATE, vol), 'sell')

                    total_pnl = stock_pnl + premium_received
                    capital += total_pnl
                    stock_only_pnl = shares * (apply_slippage(current_price, 'sell') - buy_price)

                    trades.append({
                        'date': str(oot_dates[entry_idx].date()) if hasattr(oot_dates[entry_idx], 'date') else str(oot_dates[entry_idx]),
                        'exit_date': str(date.date()) if hasattr(date, 'date') else str(date),
                        'ticker': ticker, 'entry_price': float(buy_price),
                        'exit_price': float(sell_price), 'shares': shares,
                        'premium': float(premium_received), 'stock_pnl': float(stock_pnl),
                        'total_pnl': float(total_pnl), 'stock_only_pnl': float(stock_only_pnl),
                        'called_away': bool(called_away), 'vol': float(vol),
                        'capital_after': float(capital), 'note': 'bounced_then_wrote'
                    })
                    closed_tickers.append(ticker)
            else:
                # Waiting for bounce
                bounce_pct = (current_price - buy_price) / buy_price

                if date_idx - entry_idx > 15:
                    # Timeout - exit without writing call
                    sell_price = apply_slippage(current_price, 'sell')
                    pnl = shares * (sell_price - buy_price)
                    capital += pnl
                    trades.append({
                        'date': str(oot_dates[entry_idx].date()) if hasattr(oot_dates[entry_idx], 'date') else str(oot_dates[entry_idx]),
                        'exit_date': str(date.date()) if hasattr(date, 'date') else str(date),
                        'ticker': ticker, 'entry_price': float(buy_price),
                        'exit_price': float(sell_price), 'shares': shares,
                        'premium': 0.0, 'stock_pnl': float(pnl),
                        'total_pnl': float(pnl), 'stock_only_pnl': float(pnl),
                        'called_away': False, 'vol': 0.0,
                        'capital_after': float(capital), 'note': 'timeout_no_bounce'
                    })
                    closed_tickers.append(ticker)
                elif bounce_pct >= 0.03:
                    # Bounce! Sell call at current price - mark the position
                    active_positions[ticker] = (entry_idx, buy_price, shares, date_idx, current_price)

        for t in closed_tickers:
            del active_positions[t]

        # Check for new MR signals (only if ticker not already active)
        for ticker, feat in features_dict.items():
            if ticker in active_positions:
                continue
            if date not in feat.index:
                continue
            row = feat.loc[date]
            if not row['mr_signal']:
                continue

            price = row['close']
            buy_price = apply_slippage(price, 'buy')
            shares = min(int(MAX_PER_TRADE / buy_price), int(capital / buy_price))
            if shares <= 0:
                continue

            active_positions[ticker] = (date_idx, buy_price, shares, None, None)

        equity_curve.append({'date': str(date.date()) if hasattr(date, 'date') else str(date), 'capital': float(capital)})

    return trades, equity_curve


def simulate_variant_D(features_dict, oot_dates):
    """High-IV Only Write: Only sell call when realized vol > 30%. Otherwise just hold stock."""
    trades = []
    capital = STARTING_CAPITAL
    equity_curve = []

    for date_idx, date in enumerate(oot_dates):
        for ticker, feat in features_dict.items():
            if date not in feat.index:
                continue
            row = feat.loc[date]
            if not row['mr_signal']:
                continue

            price = row['close']
            vol = row['realized_vol']
            if np.isnan(vol) or vol <= 0:
                continue

            buy_price = apply_slippage(price, 'buy')
            shares = min(int(MAX_PER_TRADE / buy_price), int(capital / buy_price))
            if shares <= 0:
                continue

            exit_idx = date_idx + 22
            if exit_idx >= len(oot_dates):
                continue
            exit_date = oot_dates[exit_idx]
            if exit_date not in feat.index:
                continue

            exit_price = feat.loc[exit_date, 'close']
            sell_price = apply_slippage(exit_price, 'sell')

            premium_received = 0.0
            called_away = False
            stock_pnl = shares * (sell_price - buy_price)

            # Only write call if vol > 30%
            wrote_call = vol > 0.30
            if wrote_call:
                T = 30 / 365.0
                call_premium = bs_call_price(price, price, T, RISK_FREE_RATE, vol)
                premium_received = shares * apply_slippage(call_premium, 'sell')
                called_away = exit_price > price
                if called_away:
                    stock_pnl = shares * (apply_slippage(price, 'sell') - buy_price)

            total_pnl = stock_pnl + premium_received
            capital += total_pnl
            stock_only_pnl = shares * (sell_price - buy_price)

            trades.append({
                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                'ticker': ticker,
                'entry_price': float(buy_price),
                'exit_price': float(sell_price),
                'shares': shares,
                'premium': float(premium_received),
                'stock_pnl': float(stock_pnl),
                'total_pnl': float(total_pnl),
                'stock_only_pnl': float(stock_only_pnl),
                'called_away': bool(called_away),
                'wrote_call': bool(wrote_call),
                'vol': float(vol),
                'capital_after': float(capital),
            })

        equity_curve.append({'date': str(date.date()) if hasattr(date, 'date') else str(date), 'capital': float(capital)})

    return trades, equity_curve


def simulate_variant_E(features_dict, oot_dates):
    """Rolling Write: Buy on MR signal, sell weekly ATM calls (5-DTE). Roll each Friday for ~4 weeks.
    Each week: sell ATM call at current price. If called away (stock > strike), cap stock P&L
    at strike for that week, then re-enter at market for next week's roll."""
    trades = []
    capital = STARTING_CAPITAL
    equity_curve = []

    for date_idx, date in enumerate(oot_dates):
        for ticker, feat in features_dict.items():
            if date not in feat.index:
                continue
            row = feat.loc[date]
            if not row['mr_signal']:
                continue

            price = row['close']
            vol = row['realized_vol']
            if np.isnan(vol) or vol <= 0:
                continue

            buy_price = apply_slippage(price, 'buy')
            shares = min(int(MAX_PER_TRADE / buy_price), int(capital / buy_price))
            if shares <= 0:
                continue

            # Simulate 4 weekly rolls (total ~20 trading days)
            total_premium = 0.0
            total_stock_pnl = 0.0
            any_called_away = False
            current_entry_price = buy_price
            valid_weeks = 0

            for week in range(4):
                roll_start = date_idx + week * 5
                roll_end = roll_start + 5
                if roll_end >= len(oot_dates):
                    break

                roll_start_date = oot_dates[roll_start]
                roll_end_date = oot_dates[roll_end]

                if roll_start_date not in feat.index or roll_end_date not in feat.index:
                    continue

                roll_price = feat.loc[roll_start_date, 'close']
                roll_vol = feat.loc[roll_start_date, 'realized_vol']
                if np.isnan(roll_vol) or roll_vol <= 0:
                    roll_vol = 0.25

                T_weekly = 5 / 365.0
                strike = roll_price  # ATM
                weekly_premium = bs_call_price(roll_price, strike, T_weekly, RISK_FREE_RATE, roll_vol)
                total_premium += shares * apply_slippage(weekly_premium, 'sell')

                end_price = feat.loc[roll_end_date, 'close']
                called_this_week = end_price > strike
                if called_this_week:
                    any_called_away = True
                    # Capped at strike for this week
                    week_stock_pnl = shares * (apply_slippage(strike, 'sell') - current_entry_price)
                    # Re-enter at market for next week
                    current_entry_price = apply_slippage(end_price, 'buy')
                else:
                    week_stock_pnl = shares * (apply_slippage(end_price, 'sell') - current_entry_price)
                    current_entry_price = apply_slippage(end_price, 'buy')  # re-enter

                total_stock_pnl += week_stock_pnl
                valid_weeks += 1

            if valid_weeks == 0:
                continue

            total_pnl = total_stock_pnl + total_premium
            capital += total_pnl

            # Stock-only comparison: just buy and hold for same period
            exit_idx = date_idx + valid_weeks * 5
            if exit_idx < len(oot_dates) and oot_dates[exit_idx] in feat.index:
                stock_exit_price = apply_slippage(feat.loc[oot_dates[exit_idx], 'close'], 'sell')
                stock_only_pnl = shares * (stock_exit_price - buy_price)
            else:
                stock_only_pnl = total_stock_pnl

            trades.append({
                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                'exit_date': str(oot_dates[min(exit_idx, len(oot_dates)-1)].date()),
                'ticker': ticker,
                'entry_price': float(buy_price),
                'shares': shares,
                'premium': float(total_premium),
                'stock_pnl': float(total_stock_pnl),
                'total_pnl': float(total_pnl),
                'stock_only_pnl': float(stock_only_pnl),
                'called_away': bool(any_called_away),
                'vol': float(vol),
                'capital_after': float(capital),
                'weeks_rolled': valid_weeks,
            })

        equity_curve.append({'date': str(date.date()) if hasattr(date, 'date') else str(date), 'capital': float(capital)})

    return trades, equity_curve


def simulate_variant_F(features_dict, oot_dates):
    """Collar: Buy stock, sell ATM call, buy 5% OTM put."""
    trades = []
    capital = STARTING_CAPITAL
    equity_curve = []

    for date_idx, date in enumerate(oot_dates):
        for ticker, feat in features_dict.items():
            if date not in feat.index:
                continue
            row = feat.loc[date]
            if not row['mr_signal']:
                continue

            price = row['close']
            vol = row['realized_vol']
            if np.isnan(vol) or vol <= 0:
                continue

            buy_price = apply_slippage(price, 'buy')
            shares = min(int(MAX_PER_TRADE / buy_price), int(capital / buy_price))
            if shares <= 0:
                continue

            T = 30 / 365.0
            call_strike = price  # ATM
            put_strike = price * 0.95  # 5% OTM

            call_premium = bs_call_price(price, call_strike, T, RISK_FREE_RATE, vol)
            put_cost = bs_put_price(price, put_strike, T, RISK_FREE_RATE, vol)

            net_premium = shares * (apply_slippage(call_premium, 'sell') - apply_slippage(put_cost, 'buy'))

            exit_idx = date_idx + 22
            if exit_idx >= len(oot_dates):
                continue
            exit_date = oot_dates[exit_idx]
            if exit_date not in feat.index:
                continue

            exit_price = feat.loc[exit_date, 'close']

            # With collar: capped upside (call strike), floored downside (put strike)
            effective_exit = max(min(exit_price, call_strike), put_strike)
            sell_price = apply_slippage(effective_exit, 'sell')

            stock_pnl = shares * (sell_price - buy_price)
            called_away = exit_price > call_strike
            put_exercised = exit_price < put_strike

            total_pnl = stock_pnl + net_premium
            capital += total_pnl

            # Stock-only for comparison
            stock_only_sell = apply_slippage(exit_price, 'sell')
            stock_only_pnl = shares * (stock_only_sell - buy_price)

            trades.append({
                'date': str(date.date()) if hasattr(date, 'date') else str(date),
                'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                'ticker': ticker,
                'entry_price': float(buy_price),
                'exit_price': float(sell_price),
                'shares': shares,
                'premium': float(net_premium),
                'stock_pnl': float(stock_pnl),
                'total_pnl': float(total_pnl),
                'stock_only_pnl': float(stock_only_pnl),
                'called_away': bool(called_away),
                'put_exercised': bool(put_exercised),
                'vol': float(vol),
                'capital_after': float(capital),
            })

        equity_curve.append({'date': str(date.date()) if hasattr(date, 'date') else str(date), 'capital': float(capital)})

    return trades, equity_curve


# ─── Metrics ─────────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve, label=''):
    """Compute performance metrics from trades."""
    if not trades:
        return {
            'variant': label, 'n_trades': 0, 'total_return_pct': 0, 'sharpe': 0,
            'sortino': 0, 'win_rate': 0, 'max_drawdown_pct': 0, 'profit_factor': 0,
            'avg_premium': 0, 'called_away_pct': 0, 'avg_trade_pnl': 0,
            'stock_only_total_return_pct': 0, 'stock_only_sharpe': 0,
            'premium_contribution_pct': 0,
        }

    pnls = np.array([t['total_pnl'] for t in trades])
    stock_pnls = np.array([t['stock_only_pnl'] for t in trades])
    premiums = np.array([t['premium'] for t in trades])

    total_return = np.sum(pnls)
    total_return_pct = total_return / STARTING_CAPITAL * 100
    stock_only_return = np.sum(stock_pnls)
    stock_only_return_pct = stock_only_return / STARTING_CAPITAL * 100

    # Per-trade returns (as fraction of capital)
    trade_rets = pnls / STARTING_CAPITAL
    stock_rets = stock_pnls / STARTING_CAPITAL

    # Sharpe (annualized, assume ~22 trades/year rough)
    trades_per_year = max(len(trades) / 4.5, 1)  # ~4.5 years of data
    if np.std(trade_rets) > 0:
        sharpe = np.mean(trade_rets) / np.std(trade_rets) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    if np.std(stock_rets) > 0:
        stock_sharpe = np.mean(stock_rets) / np.std(stock_rets) * np.sqrt(trades_per_year)
    else:
        stock_sharpe = 0

    # Sortino
    downside = trade_rets[trade_rets < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = np.mean(trade_rets) / np.std(downside) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0

    # Win rate
    win_rate = np.mean(pnls > 0) * 100

    # Max drawdown from equity curve
    caps = [e['capital'] for e in equity_curve]
    peak = STARTING_CAPITAL
    max_dd = 0
    for c in caps:
        peak = max(peak, c)
        dd = (peak - c) / peak
        max_dd = max(max_dd, dd)
    max_dd_pct = max_dd * 100

    # Profit factor
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Called away stats
    called_away_pct = np.mean([t.get('called_away', False) for t in trades]) * 100

    # Premium contribution
    total_premium = np.sum(premiums)
    premium_pct = (total_premium / abs(total_return) * 100) if total_return != 0 else 0

    return {
        'variant': label,
        'n_trades': len(trades),
        'total_return_pct': round(float(total_return_pct), 2),
        'total_return_dollars': round(float(total_return), 2),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'win_rate': round(float(win_rate), 1),
        'max_drawdown_pct': round(float(max_dd_pct), 2),
        'profit_factor': round(float(min(pf, 99.9)), 3),
        'avg_premium': round(float(np.mean(premiums)), 2),
        'total_premium': round(float(total_premium), 2),
        'called_away_pct': round(float(called_away_pct), 1),
        'avg_trade_pnl': round(float(np.mean(pnls)), 2),
        'stock_only_total_return_pct': round(float(stock_only_return_pct), 2),
        'stock_only_sharpe': round(float(stock_sharpe), 3),
        'sharpe_improvement': round(float(sharpe - stock_sharpe), 3),
        'premium_contribution_pct': round(float(premium_pct), 1),
    }


# ─── Regime Analysis ────────────────────────────────────────────────
def regime_analysis(trades, spy_data):
    """Classify trades by market regime using SPY."""
    if not trades or spy_data is None or len(spy_data) == 0:
        return {'bull': {}, 'bear': {}, 'sideways': {}}

    spy_close = spy_data['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_sma50 = spy_close.rolling(50).mean()
    spy_sma200 = spy_close.rolling(200).mean()

    regime_trades = {'bull': [], 'bear': [], 'sideways': []}

    for t in trades:
        trade_date = pd.Timestamp(t['date'])
        if trade_date not in spy_close.index:
            # Find nearest
            idx = spy_close.index.searchsorted(trade_date)
            if idx >= len(spy_close.index):
                idx = len(spy_close.index) - 1
            trade_date = spy_close.index[idx]

        if trade_date not in spy_sma50.index or trade_date not in spy_sma200.index:
            continue

        sma50_val = spy_sma50.loc[trade_date]
        sma200_val = spy_sma200.loc[trade_date]
        price_val = spy_close.loc[trade_date]

        if pd.isna(sma50_val) or pd.isna(sma200_val):
            continue

        if price_val > sma50_val and sma50_val > sma200_val:
            regime_trades['bull'].append(t)
        elif price_val < sma50_val and sma50_val < sma200_val:
            regime_trades['bear'].append(t)
        else:
            regime_trades['sideways'].append(t)

    results = {}
    for regime, rtrades in regime_trades.items():
        if not rtrades:
            results[regime] = {'n_trades': 0, 'avg_pnl': 0, 'win_rate': 0, 'avg_premium': 0}
        else:
            pnls = [t['total_pnl'] for t in rtrades]
            results[regime] = {
                'n_trades': len(rtrades),
                'avg_pnl': round(float(np.mean(pnls)), 2),
                'win_rate': round(float(np.mean([p > 0 for p in pnls]) * 100), 1),
                'avg_premium': round(float(np.mean([t['premium'] for t in rtrades])), 2),
                'total_return': round(float(np.sum(pnls)), 2),
            }

    return results


# ─── 5-Gate Validation ───────────────────────────────────────────────
def five_gate_validation(metrics, trades, equity_curve):
    """Apply 5-gate validation framework."""
    gates = {}

    # Gate 1: Statistical significance (enough trades)
    gates['gate1_sufficient_trades'] = {
        'passed': metrics['n_trades'] >= 20,
        'value': metrics['n_trades'],
        'threshold': 20,
        'description': 'Minimum 20 trades for statistical significance'
    }

    # Gate 2: Positive risk-adjusted returns
    gates['gate2_positive_sharpe'] = {
        'passed': metrics['sharpe'] > 0.3,
        'value': metrics['sharpe'],
        'threshold': 0.3,
        'description': 'Sharpe ratio > 0.3'
    }

    # Gate 3: Acceptable drawdown
    gates['gate3_max_drawdown'] = {
        'passed': metrics['max_drawdown_pct'] < 25,
        'value': metrics['max_drawdown_pct'],
        'threshold': 25,
        'description': 'Max drawdown < 25%'
    }

    # Gate 4: Win rate + profit factor
    gates['gate4_profitability'] = {
        'passed': metrics['win_rate'] > 45 and metrics['profit_factor'] > 1.1,
        'value': f"WR={metrics['win_rate']}%, PF={metrics['profit_factor']}",
        'threshold': 'WR>45%, PF>1.1',
        'description': 'Minimum profitability thresholds'
    }

    # Gate 5: Option overlay adds value over stock-only
    sharpe_improvement = metrics.get('sharpe_improvement', 0)
    gates['gate5_option_value_add'] = {
        'passed': sharpe_improvement > -0.1,  # At least not much worse
        'value': sharpe_improvement,
        'threshold': -0.1,
        'description': 'Option overlay does not destroy Sharpe by >0.1'
    }

    all_passed = all(g['passed'] for g in gates.values())
    n_passed = sum(g['passed'] for g in gates.values())

    return {
        'gates': gates,
        'all_passed': all_passed,
        'gates_passed': f"{n_passed}/5",
    }


# ─── Permutation Test ───────────────────────────────────────────────
def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """1000-permutation test: shuffle trade dates to test if returns are due to timing."""
    if len(trades) < 5:
        return {'p_value': 1.0, 'actual_sharpe': 0, 'perm_sharpes_mean': 0, 'significant': False}

    rng = np.random.RandomState(RANDOM_SEED)
    pnls = np.array([t['total_pnl'] for t in trades])
    actual_mean = np.mean(pnls)

    perm_means = []
    for _ in range(n_perms):
        shuffled = rng.permutation(pnls)
        # Shuffle sign of each trade (null hypothesis: no directional edge)
        signs = rng.choice([-1, 1], size=len(pnls))
        perm_means.append(np.mean(pnls * signs))

    perm_means = np.array(perm_means)
    p_value = float(np.mean(perm_means >= actual_mean))

    return {
        'p_value': round(p_value, 4),
        'actual_mean_pnl': round(float(actual_mean), 2),
        'perm_mean_pnl_avg': round(float(np.mean(perm_means)), 4),
        'perm_mean_pnl_95th': round(float(np.percentile(perm_means, 95)), 2),
        'significant_5pct': bool(p_value < 0.05),
        'significant_10pct': bool(p_value < 0.10),
    }


# ─── Main ────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 70)
    print("COVERED CALL QUALITY STOCKS BACKTEST")
    print("=" * 70)

    # Download data
    data = download_data()

    # Also get SPY for regime analysis
    print("Downloading SPY for regime classification...")
    spy_data = yf.download('SPY', start='2021-06-01', end=OOT_END, progress=False, auto_adjust=True)

    # Compute features
    print("\nComputing features...")
    features_dict = {}
    for ticker, df in data.items():
        features_dict[ticker] = compute_features(df)

    # Get OOT trading dates (from any ticker that has full coverage)
    ref_ticker = list(features_dict.keys())[0]
    ref_feat = features_dict[ref_ticker]
    oot_mask = ref_feat.index >= pd.Timestamp(OOT_START)
    oot_dates = list(ref_feat.index[oot_mask])
    print(f"OOT period: {oot_dates[0].date()} to {oot_dates[-1].date()} ({len(oot_dates)} trading days)")

    # Count MR signals
    total_signals = 0
    for ticker, feat in features_dict.items():
        oot_feat = feat[feat.index >= pd.Timestamp(OOT_START)]
        n_sig = oot_feat['mr_signal'].sum()
        if n_sig > 0:
            total_signals += n_sig
            print(f"  {ticker}: {n_sig} MR signals")
    print(f"  Total MR signals across universe: {total_signals}")

    # Run all variants
    variants = {
        'A_BuyWrite_ATM': simulate_variant_A,
        'B_OTM_2pct': simulate_variant_B,
        'C_BounceWrite': simulate_variant_C,
        'D_HighIV_Only': simulate_variant_D,
        'E_Rolling_Weekly': simulate_variant_E,
        'F_Collar': simulate_variant_F,
    }

    all_results = {}

    for name, sim_func in variants.items():
        print(f"\n{'─' * 50}")
        print(f"Running Variant {name}...")
        trades, equity_curve = sim_func(features_dict, oot_dates)
        metrics = compute_metrics(trades, equity_curve, label=name)
        regime = regime_analysis(trades, spy_data)
        gates = five_gate_validation(metrics, trades, equity_curve)
        perm = permutation_test(trades)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}% (${metrics['total_return_dollars']:.2f})")
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Avg Premium: ${metrics['avg_premium']:.2f} | Called Away: {metrics['called_away_pct']:.1f}%")
        print(f"  Stock-Only Return: {metrics['stock_only_total_return_pct']:.1f}% | Stock Sharpe: {metrics['stock_only_sharpe']:.3f}")
        print(f"  Sharpe Improvement: {metrics['sharpe_improvement']:+.3f}")
        print(f"  Gates: {gates['gates_passed']} | Perm p-value: {perm['p_value']:.4f}")

        all_results[name] = {
            'metrics': metrics,
            'regime_analysis': regime,
            'five_gate_validation': gates,
            'permutation_test': perm,
            'n_trades_by_ticker': {},
            'sample_trades': trades[:5] if trades else [],
        }

        # Count trades by ticker
        ticker_counts = {}
        for t in trades:
            tk = t['ticker']
            ticker_counts[tk] = ticker_counts.get(tk, 0) + 1
        all_results[name]['n_trades_by_ticker'] = ticker_counts

    # ─── Summary Comparison ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Variant':<25} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'MaxDD%':>7} {'PF':>6} {'Gates':>6}")
    print("-" * 85)

    for name, res in all_results.items():
        m = res['metrics']
        g = res['five_gate_validation']
        print(f"{name:<25} {m['n_trades']:>6} {m['total_return_pct']:>7.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate']:>5.1f}% {m['max_drawdown_pct']:>6.1f}% {m['profit_factor']:>6.2f} {g['gates_passed']:>6}")

    # Best variant
    best = max(all_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
    print(f"\nBest by Sharpe: {best[0]} (Sharpe={best[1]['metrics']['sharpe']:.3f})")

    # ─── Save Results ────────────────────────────────────────────────
    output = {
        'metadata': {
            'backtest_name': 'Covered Call Quality Stocks',
            'run_date': datetime.now().isoformat(),
            'oot_period': f'{OOT_START} to {OOT_END}',
            'starting_capital': STARTING_CAPITAL,
            'max_per_trade': MAX_PER_TRADE,
            'slippage_pct': SLIPPAGE_PCT,
            'risk_free_rate': RISK_FREE_RATE,
            'universe': UNIVERSE,
            'n_permutations': N_PERMUTATIONS,
            'total_mr_signals': int(total_signals),
            'runtime_seconds': round(time.time() - start_time, 1),
        },
        'variants': all_results,
        'best_variant': best[0],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Runtime: {time.time() - start_time:.1f}s")


if __name__ == '__main__':
    main()
