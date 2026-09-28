#!/usr/bin/env python3
"""
Cross-Asset Trend Following Paper Engine — DualMom VolTarget
=============================================================

Validated: 4/4 adversarial gates passed
- Sharpe 0.91, CAGR 6.0%, MaxDD -9.2%, WR 65.1%
- Corr(SPY) 0.60

Strategy: Dual momentum (price > SMA200 AND 12-1 mom > 0) with
volatility targeting (10% annual portfolio vol target).

Monthly rebalance, broad cross-asset universe.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

STATE_DIR = Path('/home/jupiter/Lvl3Quant/state')
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / 'cross_asset_trend_paper_state.json'
LOG_FILE = STATE_DIR / 'cross_asset_trend_paper_log.json'

UNIVERSE = [
    'SPY', 'QQQ', 'IWM', 'EFA', 'EEM',  # Equities
    'TLT', 'IEF', 'HYG', 'LQD', 'TIP',  # Bonds
    'GLD', 'SLV', 'DBC', 'USO',           # Commodities
    'VNQ', 'IYR',                           # REITs
    'XLE', 'XLU',                           # Sectors
]

PAPER_CAPITAL = 100_000
VOL_TARGET = 0.10  # 10% annual vol target


def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'nav': PAPER_CAPITAL,
        'cash': PAPER_CAPITAL,
        'holdings': {},
        'last_rebalance': None,
        'trade_log': [],
        'created': datetime.now().isoformat(),
    }


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(entry):
    log = []
    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            log = json.load(f)
    log.append(entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2, default=str)


def run():
    import yfinance as yf

    now = datetime.now()
    print(f"Cross-Asset Trend Paper Engine — {now.strftime('%Y-%m-%d %H:%M')}")

    state = load_state()

    # Check if rebalance needed (monthly)
    if state['last_rebalance']:
        last = datetime.fromisoformat(state['last_rebalance'])
        days_since = (now - last).days
        if days_since < 25:  # Monthly = ~21 trading days
            print(f"  Last rebalance {days_since} days ago. Next in ~{25-days_since} days.")

            # Just update NAV
            raw = yf.download(list(state['holdings'].keys()) or ['SPY'],
                             period='5d', progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw['Close']
            else:
                close = raw

            nav = state['cash']
            for ticker, shares in state['holdings'].items():
                if ticker in close.columns:
                    price = float(close[ticker].dropna().iloc[-1])
                    nav += shares * price
                elif isinstance(close, pd.Series):
                    price = float(close.dropna().iloc[-1])
                    nav += shares * price

            state['nav'] = round(nav, 2)
            save_state(state)
            print(f"  NAV: ${nav:,.2f} | Holdings: {len(state['holdings'])} assets")
            return

    # Download data for signals
    print("  Downloading data for signal generation...")
    raw = yf.download(UNIVERSE, start='2024-01-01', end=now.strftime('%Y-%m-%d'),
                      progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    # Compute signals
    sma200 = close.rolling(200).mean()
    daily_rets = close.pct_change()
    rolling_vol = daily_rets.rolling(63).std() * np.sqrt(252)

    # 12-1 momentum
    monthly = close.resample('ME').last()
    mom_12 = monthly.pct_change(12)
    mom_1 = monthly.pct_change(1)

    latest = close.iloc[-1]
    latest_sma200 = sma200.iloc[-1]
    latest_vol = rolling_vol.iloc[-1]

    # Determine long positions
    longs = []
    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue

        price = latest[ticker]
        sma = latest_sma200[ticker]

        if pd.isna(price) or pd.isna(sma):
            continue

        # Dual momentum: above SMA200 AND positive 12-1 momentum
        above_sma = price > sma

        # Get latest monthly 12-1 momentum
        if ticker in mom_12.columns and ticker in mom_1.columns:
            m12 = mom_12[ticker].dropna()
            m1 = mom_1[ticker].dropna()
            if len(m12) > 0 and len(m1) > 0:
                mom = float(m12.iloc[-1]) - float(m1.iloc[-1])
            else:
                mom = 0
        else:
            mom = 0

        pos_mom = mom > 0

        if above_sma and pos_mom:
            vol = latest_vol[ticker]
            if pd.isna(vol) or vol < 0.01:
                vol = 0.15
            longs.append({
                'ticker': ticker,
                'vol': float(vol),
                'price': float(price),
                'mom': float(mom),
            })

    print(f"  Dual momentum filter: {len(longs)} assets trending up")

    if not longs:
        # All flat — go to cash
        print("  No assets trending up. Moving to 100% cash.")
        state['holdings'] = {}
        state['cash'] = state['nav']
        state['last_rebalance'] = now.isoformat()
        save_state(state)
        return

    # Volatility-targeting weights
    inv_vols = {a['ticker']: 1.0 / a['vol'] for a in longs}
    total_inv = sum(inv_vols.values())
    raw_weights = {t: iv / total_inv for t, iv in inv_vols.items()}

    # Estimate portfolio vol and scale to target
    port_vol = sum(raw_weights[t] * next(a['vol'] for a in longs if a['ticker'] == t)
                   for t in raw_weights)
    scale = min(VOL_TARGET / (port_vol + 1e-10), 1.5)
    weights = {t: w * scale for t, w in raw_weights.items()}

    # Ensure weights sum to <= 1.0
    total_weight = sum(weights.values())
    if total_weight > 1.0:
        weights = {t: w / total_weight for t, w in weights.items()}

    # Allocate capital
    nav = state['nav']
    new_holdings = {}
    invested = 0

    for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
        alloc = nav * weight
        price = next(a['price'] for a in longs if a['ticker'] == ticker)
        shares = int(alloc / price)
        if shares > 0:
            new_holdings[ticker] = shares
            invested += shares * price

    state['cash'] = round(nav - invested, 2)
    old_holdings = state.get('holdings', {})
    state['holdings'] = new_holdings
    state['last_rebalance'] = now.isoformat()

    # Log trades
    all_tickers = set(list(old_holdings.keys()) + list(new_holdings.keys()))
    trades = []
    for t in all_tickers:
        old_shares = old_holdings.get(t, 0)
        new_shares = new_holdings.get(t, 0)
        if old_shares != new_shares:
            trades.append({
                'date': now.isoformat(),
                'ticker': t,
                'action': 'BUY' if new_shares > old_shares else 'SELL',
                'shares': abs(new_shares - old_shares),
                'signals': 'DualMom+VolTarget',
            })

    state['trade_log'].extend(trades)
    save_state(state)

    log_trade({
        'date': now.isoformat(),
        'nav': nav,
        'n_long': len(new_holdings),
        'holdings': new_holdings,
        'trades': trades,
    })

    print(f"\n  REBALANCED — NAV: ${nav:,.2f}")
    print(f"  Long: {len(new_holdings)} assets | Cash: ${state['cash']:,.2f}")
    for t, s in sorted(new_holdings.items(), key=lambda x: -x[1]):
        price = next(a['price'] for a in longs if a['ticker'] == t)
        value = s * price
        pct = value / nav * 100
        print(f"    {t}: {s} shares (${value:,.0f}, {pct:.1f}%)")


if __name__ == '__main__':
    run()
