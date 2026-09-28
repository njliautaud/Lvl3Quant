#!/usr/bin/env python3
"""
Sequential Chain Paper Engine — Strategy #15 (6/6 Adversarial PASS)
====================================================================
RSI Divergence setup → Bond Yield Drop trigger within 5 days.

Logic:
  1. Detect RSI Divergence (price lower low, RSI higher low, RSI<40) = SETUP
  2. Within 5 trading days, if 10Y yield drops >0.10 = TRIGGER → BUY
  3. Exit: +10% TP, -15% SL, 21-day max hold

Validated performance: Sharpe 1.528, Sortino 2.257, WR 65.9%, PF 2.23, 135 trades
6/6 adversarial pass. Inverse -0.025, 100% param robustness (108/108).
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

STATE_FILE = Path('/home/jupiter/Lvl3Quant/state/sequential_chain_paper.json')
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
SETUP_WINDOW = 5  # Days for trigger after setup


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'positions': [],
        'closed_trades': [],
        'active_setups': [],  # RSI divergence setups waiting for trigger
        'created': str(datetime.now()),
        'total_pnl': 0,
        'n_trades': 0,
    }


def save_state(state):
    state['last_updated'] = str(datetime.now())
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_market_data():
    """Download latest data for signal generation."""
    import yfinance as yf
    tickers = UNIVERSE + ['SPY', '^TNX']
    end = datetime.now()
    start = end - timedelta(days=120)
    raw = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                      end=end.strftime('%Y-%m-%d'),
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw
    if hasattr(close.columns, 'droplevel'):
        try: close.columns = close.columns.droplevel(1)
        except: pass
    close = close.ffill().dropna(how='all')
    return close


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_rsi_divergence(close_series):
    """Detect RSI bullish divergence: price lower low, RSI higher low, RSI < 40."""
    if len(close_series) < 15:
        return False
    rsi = _rsi(close_series, 14)
    if rsi.isna().iloc[-1]:
        return False
    # Compare today vs 10 days ago
    wc = close_series.iloc[-11:]
    wr = rsi.iloc[-11:]
    if len(wc) < 11 or wr.isna().any():
        return False
    # Price lower low, RSI higher low, RSI < 40
    return (close_series.iloc[-1] < wc.iloc[0] and
            rsi.iloc[-1] > wr.iloc[0] and
            rsi.iloc[-1] < 40)


def detect_bond_yield_drop(close_df):
    """Detect 10Y yield dropping >0.10 in 5 days."""
    if '^TNX' not in close_df.columns:
        return False
    tnx = close_df['^TNX'].dropna()
    if len(tnx) < 6:
        return False
    change_5d = tnx.iloc[-1] - tnx.iloc[-6]
    return change_5d < -0.10


def main():
    print(f"Sequential Chain Paper Engine — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    state = load_state()
    close = get_market_data()
    today = close.index[-1]
    today_str = str(today.date())

    # --- Check existing positions for exits ---
    active = []
    for pos in state['positions']:
        ticker = pos['ticker']
        entry_price = pos['entry_price']
        entry_date = pd.Timestamp(pos['entry_date'])
        days_held = (today - entry_date).days

        try:
            current_price = close.at[today, ticker]
        except:
            active.append(pos)
            continue

        if pd.isna(current_price):
            active.append(pos)
            continue

        ret = (current_price - entry_price) / entry_price
        exit_reason = None

        if ret >= PROFIT_TARGET:
            exit_reason = 'profit_target'
        elif ret <= STOP_LOSS:
            exit_reason = 'stop_loss'
        elif days_held >= HOLD_DAYS:
            exit_reason = 'time_expiry'

        if exit_reason:
            pnl = POS_SIZE * (ret - 0.001)
            trade = {
                'ticker': ticker, 'entry_date': str(entry_date.date()),
                'exit_date': today_str, 'entry_price': round(entry_price, 2),
                'exit_price': round(current_price, 2), 'return': round(ret * 100, 2),
                'pnl': round(pnl, 2), 'exit_reason': exit_reason,
                'hold_days': days_held,
            }
            state['closed_trades'].append(trade)
            state['total_pnl'] = round(state['total_pnl'] + pnl, 2)
            state['n_trades'] += 1
            print(f"  EXIT {ticker}: {exit_reason} | ret={ret*100:+.1f}% | pnl=${pnl:.2f}")
        else:
            pos['current_price'] = round(current_price, 2)
            pos['unrealized_pnl'] = round(POS_SIZE * ret, 2)
            pos['days_held'] = days_held
            active.append(pos)
            print(f"  HOLD {ticker}: {ret*100:+.1f}% | day {days_held}/{HOLD_DAYS}")

    state['positions'] = active

    # --- Step 1: Detect new RSI Divergence setups ---
    new_setups = []
    for setup in state.get('active_setups', []):
        setup_date = pd.Timestamp(setup['setup_date'])
        days_since = (today - setup_date).days
        if days_since <= SETUP_WINDOW:
            new_setups.append(setup)
        else:
            print(f"  EXPIRED setup: {setup['ticker']} (setup {setup['setup_date']}, {days_since}d ago)")

    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        # Skip if already a setup for this ticker
        if any(s['ticker'] == ticker for s in new_setups):
            continue
        if detect_rsi_divergence(close[ticker].dropna()):
            setup = {'ticker': ticker, 'setup_date': today_str}
            new_setups.append(setup)
            print(f"  NEW SETUP: {ticker} RSI divergence detected")

    state['active_setups'] = new_setups

    # --- Step 2: Check for bond yield trigger on active setups ---
    bond_trigger = detect_bond_yield_drop(close)

    if bond_trigger and len(active) < MAX_CONCURRENT:
        print(f"  BOND YIELD TRIGGER FIRED — 10Y dropped >0.10 in 5 days")
        held_tickers = {p['ticker'] for p in active}
        slots = MAX_CONCURRENT - len(active)

        # Sort setups by freshness (newest first)
        eligible = [s for s in new_setups if s['ticker'] not in held_tickers]
        eligible.sort(key=lambda x: x['setup_date'], reverse=True)

        for setup in eligible[:slots]:
            ticker = setup['ticker']
            try:
                price = close.at[today, ticker]
            except:
                continue
            if pd.isna(price):
                continue

            pos = {
                'ticker': ticker,
                'entry_price': round(float(price), 2),
                'entry_date': today_str,
                'setup_date': setup['setup_date'],
                'signal': 'sequential_chain_rsi_div_bond_yield',
            }
            state['positions'].append(pos)
            print(f"  BUY {ticker} @ ${price:.2f} (setup {setup['setup_date']})")

            # Remove from active setups
            new_setups = [s for s in new_setups if s['ticker'] != ticker]
            slots -= 1
            if slots <= 0:
                break

        state['active_setups'] = new_setups
    elif bond_trigger:
        print(f"  Bond yield trigger fired but max positions reached ({len(active)}/{MAX_CONCURRENT})")
    else:
        print(f"  No bond yield trigger today. {len(new_setups)} active setups waiting.")

    # --- Summary ---
    print(f"\nSummary:")
    print(f"  Positions: {len(state['positions'])}/{MAX_CONCURRENT}")
    print(f"  Active setups: {len(state['active_setups'])}")
    print(f"  Closed trades: {state['n_trades']}")
    print(f"  Total P&L: ${state['total_pnl']:.2f}")

    for p in state['positions']:
        upnl = p.get('unrealized_pnl', 0)
        print(f"    {p['ticker']}: entry ${p['entry_price']} | unrealized ${upnl:+.2f}")

    save_state(state)
    print(f"\nState saved. Engine complete.")


if __name__ == '__main__':
    main()
