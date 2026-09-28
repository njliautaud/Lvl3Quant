#!/usr/bin/env python3
"""
TRUE Out-of-Sample Validation: Cross-Asset Macro Sector Rotation
================================================================
Tests the AVO-evolved strategy on 2026 data that was NEVER seen during evolution.
The strategy evolved on 2022-2025 walk-forward folds. This is the lockbox test.
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add the strategy to path
STRATEGY_DIR = '/home/jupiter/teleclaude-main/runs/cross_asset_macro-20260823-194958/work'
sys.path.insert(0, STRATEGY_DIR)
import strategy

import yfinance as yf

# Download 2026 data with warmup
print("Downloading 2026 data (with 2025 warmup)...")
SECTOR_ETFS = strategy.SECTOR_ETFS
MACRO_TICKERS = ['TLT', 'IEF', 'GLD', 'USO', 'UUP']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'

tickers = SECTOR_ETFS + MACRO_TICKERS + [BENCHMARK, VIX_TICKER]
raw = yf.download(tickers, start='2025-06-01', end='2026-08-23', auto_adjust=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    prices = raw['Close']
else:
    prices = raw

if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = [c[0] if isinstance(c, tuple) else c for c in prices.columns]

prices = prices.ffill().dropna(how='all')

spy = prices[BENCHMARK]
vix_cols = [c for c in prices.columns if 'VIX' in str(c).upper()]
vix = prices[vix_cols[0]] if vix_cols else pd.Series(20.0, index=prices.index)
macro_data = prices[[t for t in MACRO_TICKERS if t in prices.columns]].copy()

# Generate signals
print("Generating signals...")
signals = strategy.generate_signals(prices, spy, vix, macro_data)

# Simulate trades on 2026 data only
OOS_START = '2026-01-01'
OOS_END = '2026-08-22'

oos_mask = (prices.index >= OOS_START) & (prices.index <= OOS_END)
dates = prices.index[oos_mask]

CAPITAL = 10000.0
capital = CAPITAL
slippage_pct = strategy.SLIPPAGE_PCT
max_per_trade = strategy.MAX_PER_TRADE
max_concurrent = strategy.MAX_CONCURRENT

trades = []
open_positions = []
daily_pnl = pd.Series(0.0, index=dates)
equity_curve = pd.Series(CAPITAL, index=dates)
peak_equity = CAPITAL

for i, date in enumerate(dates):
    curr_equity = equity_curve.iloc[max(0, i-1)] if i > 0 else CAPITAL
    portfolio_dd = (curr_equity - peak_equity) / peak_equity if peak_equity > 0 else 0

    # Check exits
    still_open = []
    for pos in open_positions:
        curr_price = prices.loc[date, pos['ticker']]
        if pd.isna(curr_price):
            still_open.append(pos)
            continue

        do_exit = strategy.should_exit(pos, curr_price, date, portfolio_dd)

        if do_exit:
            exit_price = curr_price * (1.0 - slippage_pct)
            pnl = (exit_price - pos['entry_price_adj']) * pos['shares']
            capital += pos['size_dollars'] + pnl
            daily_pnl.iloc[i] += pnl
            days_held = int(np.busday_count(
                np.datetime64(pos['entry_date'], 'D'),
                np.datetime64(date, 'D')))
            exit_reason = 'normal'
            entry_p = pos['entry_price_adj']
            pnl_pct = (curr_price - entry_p) / entry_p
            if pnl_pct >= strategy.TAKE_PROFIT_PCT:
                exit_reason = 'take_profit'
            elif days_held >= strategy.MAX_HOLD_DAYS:
                exit_reason = 'max_hold'
            elif pnl_pct < 0 and days_held >= strategy.UNDERWATER_EXIT_DAYS:
                exit_reason = 'underwater_exit'
            else:
                exit_reason = 'trailing_stop'

            trades.append({
                'ticker': pos['ticker'],
                'entry_date': str(pos['entry_date'])[:10],
                'exit_date': str(date)[:10],
                'entry_price': round(float(pos['entry_price_adj']), 2),
                'exit_price': round(float(exit_price), 2),
                'shares': pos['shares'],
                'pnl': round(float(pnl), 2),
                'return_pct': round(float(pnl / pos['size_dollars'] * 100), 2),
                'days_held': days_held,
                'exit_reason': exit_reason,
            })
        else:
            if i > 0:
                prev_price = prices.loc[dates[i-1], pos['ticker']]
                if not pd.isna(prev_price):
                    daily_pnl.iloc[i] += (curr_price - prev_price) * pos['shares']
            still_open.append(pos)

    open_positions = still_open

    # Check entries
    if len(open_positions) < max_concurrent and date in signals.index:
        for ticker in SECTOR_ETFS:
            if len(open_positions) >= max_concurrent:
                break
            if ticker not in signals.columns:
                continue
            if not signals.loc[date, ticker]:
                continue
            if any(p['ticker'] == ticker for p in open_positions):
                continue

            price = prices.loc[date, ticker]
            if pd.isna(price) or price <= 0:
                continue

            size = min(max_per_trade, capital * 0.90)
            if size < 50:
                continue

            entry_price = price * (1.0 + slippage_pct)
            shares = int(size / entry_price)
            if shares < 1:
                continue

            actual_cost = shares * entry_price
            capital -= actual_cost
            open_positions.append({
                'ticker': ticker,
                'entry_date': date,
                'entry_price_adj': entry_price,
                'shares': shares,
                'size_dollars': actual_cost,
                'hwm': entry_price,
            })

    if i > 0:
        equity_curve.iloc[i] = equity_curve.iloc[i-1] + daily_pnl.iloc[i]
    else:
        equity_curve.iloc[i] = CAPITAL + daily_pnl.iloc[i]
    peak_equity = max(peak_equity, equity_curve.iloc[i])

# Force close remaining
last_date = dates[-1]
for pos in open_positions:
    price = prices.loc[last_date, pos['ticker']]
    if pd.isna(price):
        continue
    exit_price = price * (1.0 - slippage_pct)
    pnl = (exit_price - pos['entry_price_adj']) * pos['shares']
    days_held = int(np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(last_date, 'D')))
    trades.append({
        'ticker': pos['ticker'],
        'entry_date': str(pos['entry_date'])[:10],
        'exit_date': str(last_date)[:10],
        'entry_price': round(float(pos['entry_price_adj']), 2),
        'exit_price': round(float(exit_price), 2),
        'shares': pos['shares'],
        'pnl': round(float(pnl), 2),
        'return_pct': round(float(pnl / pos['size_dollars'] * 100), 2),
        'days_held': days_held,
        'exit_reason': 'force_close',
    })

# Compute metrics
print("\n" + "="*60)
print("TRUE OUT-OF-SAMPLE RESULTS: Cross-Asset Macro (2026)")
print("="*60)

n = len(trades)
if n > 0:
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / n
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    daily_ret = daily_pnl / CAPITAL
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0]
    ds = downside.std() if len(downside) > 5 else daily_ret.std()
    sortino = float(daily_ret.mean() / ds * np.sqrt(252)) if ds > 0 else 0

    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    max_dd = float(dd.min()) * 100

    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity - CAPITAL) / CAPITAL * 100

    print(f"Total trades:    {n}")
    print(f"Win rate:        {win_rate:.1%}")
    print(f"Profit factor:   {pf:.2f}")
    print(f"Sharpe ratio:    {sharpe:.2f}")
    print(f"Sortino ratio:   {sortino:.2f}")
    print(f"Total P&L:       ${total_pnl:,.2f}")
    print(f"Total return:    {total_return:.1f}%")
    print(f"Max drawdown:    {max_dd:.1f}%")
    print(f"Final equity:    ${final_equity:,.2f}")

    # By sector
    print("\nBy sector:")
    for etf in sorted(set(t['ticker'] for t in trades)):
        etf_trades = [t for t in trades if t['ticker'] == etf]
        etf_pnl = sum(t['pnl'] for t in etf_trades)
        etf_wins = len([t for t in etf_trades if t['pnl'] > 0])
        print(f"  {etf}: {len(etf_trades)} trades, {etf_wins}/{len(etf_trades)} wins, P&L ${etf_pnl:.2f}")

    # By exit reason
    print("\nBy exit reason:")
    for reason in sorted(set(t['exit_reason'] for t in trades)):
        r_trades = [t for t in trades if t['exit_reason'] == reason]
        r_pnl = sum(t['pnl'] for t in r_trades)
        print(f"  {reason}: {len(r_trades)} trades, P&L ${r_pnl:.2f}")

    # Monthly breakdown
    print("\nMonthly equity:")
    for month in sorted(set(str(d)[:7] for d in equity_curve.index)):
        month_mask = [str(d)[:7] == month for d in equity_curve.index]
        month_vals = equity_curve[month_mask]
        if len(month_vals) > 0:
            print(f"  {month}: ${month_vals.iloc[-1]:,.2f}")

    # Save trades
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv('/home/jupiter/Lvl3Quant/validation/cross_asset_macro_2026_oos_trades.csv', index=False)
    print(f"\nTrades saved to validation/cross_asset_macro_2026_oos_trades.csv")
else:
    print("No trades generated in 2026 OOS period!")
