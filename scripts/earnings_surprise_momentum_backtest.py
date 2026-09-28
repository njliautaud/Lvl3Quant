#!/usr/bin/env python3
"""
Earnings Surprise Momentum Backtest
====================================
Academic basis: Bernard & Thomas (1989), Chan et al (1996)
- Post-earnings announcement drift (PEAD) persists 60-90 days
- Firms that beat tend to beat again next quarter

6 Variants:
  A) Beat-and-Hold-to-Next: Buy after >3% gap, hold ~60 days
  B) Beat-Chain: Only if also beat previous quarter
  C) Big Beat Only: Gap >7%, hold 60 days
  D) Sector Leaders: Also buy sector ETF for 20 days
  E) Portfolio Rotation: Always hold 3 most recent beaters equally weighted
  F) Regime-Adjusted: Only enter when SPY > 50-SMA

Walk-forward OOT: Jan 2022 - Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
CAPITAL = 645.0
MAX_POSITIONS = 5
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'UBER',
    'LYFT', 'COIN', 'RBLX', 'DDOG', 'TTD', 'SHOP', 'NET', 'ROKU'
]

SECTOR_ETF_MAP = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'AMD': 'XLK',
    'CRM': 'XLK', 'DDOG': 'XLK', 'NET': 'XLK', 'SHOP': 'XLK',
    'TTD': 'XLK',
    'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC', 'SNAP': 'XLC',
    'PINS': 'XLC', 'ROKU': 'XLC',
    'AMZN': 'XLY', 'TSLA': 'XLY', 'RBLX': 'XLY',
    'PLTR': 'XLK', 'SOFI': 'XLF', 'HOOD': 'XLF',
    'UBER': 'XLY', 'LYFT': 'XLY', 'COIN': 'XLF',
}

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')
HOLD_DAYS = 60  # ~1 quarter minus buffer
PERM_ITERATIONS = 1000

# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download price data and detect earnings events via gap days."""
    print("Downloading price data...")
    all_tickers = list(set(UNIVERSE + list(set(SECTOR_ETF_MAP.values())) + ['SPY']))

    prices = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start='2021-06-01', end=OOT_END.strftime('%Y-%m-%d'),
                           progress=False, auto_adjust=True)
            # Flatten multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                prices[ticker] = df
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    print(f"  Downloaded {len(prices)} tickers")
    return prices


def detect_earnings_events(prices):
    """
    Detect earnings events using overnight gap as proxy.
    A gap > 3% on high volume likely = earnings report.
    We filter to only keep ~4 events per year per stock (quarterly).
    """
    events = {}

    for ticker in UNIVERSE:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['abs_gap'] = df['gap_pct'].abs()

        # Also check volume spike as confirmation
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma20']

        # Earnings = large gap (>3%) with volume spike (>1.5x)
        earnings_mask = (df['abs_gap'] > 0.03) & (df['vol_ratio'] > 1.5)
        earnings_days = df[earnings_mask].copy()

        # Deduplicate: keep only 1 event per 60-day window
        if len(earnings_days) == 0:
            continue

        deduped = []
        last_date = None
        for date, row in earnings_days.iterrows():
            if last_date is None or (date - last_date).days > 60:
                deduped.append({
                    'date': date,
                    'gap_pct': row['gap_pct'],
                    'is_beat': row['gap_pct'] > 0,  # positive gap = beat
                    'gap_abs': row['abs_gap'],
                })
                last_date = date

        events[ticker] = deduped

    total = sum(len(v) for v in events.values())
    beats = sum(sum(1 for e in v if e['is_beat']) for v in events.values())
    print(f"  Detected {total} earnings events ({beats} beats) across {len(events)} stocks")
    return events


# ── Trade Simulation Helpers ───────────────────────────────────────────────
def get_price_at(prices_df, date, field='Open', offset_days=0):
    """Get price at date + offset, handling weekends/holidays."""
    if date not in prices_df.index:
        # Find next available trading day
        mask = prices_df.index >= date
        if not mask.any():
            return None
        date = prices_df.index[mask][0]

    idx = prices_df.index.get_loc(date)
    target_idx = idx + offset_days
    if target_idx < 0 or target_idx >= len(prices_df):
        return None

    val = prices_df.iloc[target_idx][field]
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return float(val)


def get_date_at_offset(prices_df, date, offset_days):
    """Get the trading date at offset from given date."""
    if date not in prices_df.index:
        mask = prices_df.index >= date
        if not mask.any():
            return None
        date = prices_df.index[mask][0]

    idx = prices_df.index.get_loc(date)
    target_idx = idx + offset_days
    if target_idx < 0 or target_idx >= len(prices_df):
        return None
    return prices_df.index[target_idx]


def compute_spy_sma(prices, window=50):
    """Compute SPY SMA for regime detection."""
    if 'SPY' not in prices:
        return pd.Series(dtype=float)
    spy = prices['SPY']['Close'].copy()
    return spy.rolling(window).mean()


def compute_spy_sma200(prices):
    """Compute SPY 200-SMA for bull/bear regime."""
    if 'SPY' not in prices:
        return pd.Series(dtype=float)
    spy = prices['SPY']['Close'].copy()
    return spy.rolling(200).mean()


# ── Variant Backtests ──────────────────────────────────────────────────────

def run_variant_a(prices, events):
    """Beat-and-Hold-to-Next: Buy after >3% gap, hold ~60 trading days."""
    trades = []

    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]

        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < 0.03:
                continue
            if evt['date'] < OOT_START:
                continue

            # Entry: open of next day after earnings
            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=1)
            if entry_price is None:
                continue
            entry_date = get_date_at_offset(df, evt['date'], 1)
            if entry_date is None:
                continue

            # Exit: after HOLD_DAYS trading days
            exit_price = get_price_at(df, evt['date'], 'Close', offset_days=HOLD_DAYS)
            exit_date = get_date_at_offset(df, evt['date'], HOLD_DAYS)
            if exit_price is None or exit_date is None:
                continue

            # Apply slippage
            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': HOLD_DAYS,
            })

    return trades


def run_variant_b(prices, events):
    """Beat-Chain: Only hold if also beat previous quarter (consecutive beats)."""
    trades = []

    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]

        for i, evt in enumerate(evts):
            if not evt['is_beat'] or evt['gap_abs'] < 0.03:
                continue
            if evt['date'] < OOT_START:
                continue

            # Check if previous event was also a beat
            if i == 0:
                continue
            prev_evt = evts[i - 1]
            if not prev_evt['is_beat']:
                continue

            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(df, evt['date'], 'Close', offset_days=HOLD_DAYS)
            exit_date = get_date_at_offset(df, evt['date'], HOLD_DAYS)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': HOLD_DAYS,
                'prev_gap_pct': round(prev_evt['gap_pct'] * 100, 2),
            })

    return trades


def run_variant_c(prices, events):
    """Big Beat Only: Gap >7%, hold 60 days."""
    trades = []

    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]

        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < 0.07:
                continue
            if evt['date'] < OOT_START:
                continue

            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(df, evt['date'], 'Close', offset_days=HOLD_DAYS)
            exit_date = get_date_at_offset(df, evt['date'], HOLD_DAYS)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': HOLD_DAYS,
            })

    return trades


def run_variant_d(prices, events):
    """Sector Leaders: After a beat, also buy sector ETF for 20 days."""
    trades = []

    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        etf = SECTOR_ETF_MAP.get(ticker)

        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < 0.03:
                continue
            if evt['date'] < OOT_START:
                continue

            # Stock trade (same as variant A)
            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(df, evt['date'], 'Close', offset_days=HOLD_DAYS)
            exit_date = get_date_at_offset(df, evt['date'], HOLD_DAYS)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': HOLD_DAYS,
                'type': 'stock',
            })

            # Sector ETF trade (20-day hold)
            if etf and etf in prices:
                etf_df = prices[etf]
                etf_entry = get_price_at(etf_df, evt['date'], 'Open', offset_days=1)
                etf_entry_date = get_date_at_offset(etf_df, evt['date'], 1)
                if etf_entry is None or etf_entry_date is None:
                    continue

                etf_exit = get_price_at(etf_df, evt['date'], 'Close', offset_days=20)
                etf_exit_date = get_date_at_offset(etf_df, evt['date'], 20)
                if etf_exit is None or etf_exit_date is None:
                    continue

                etf_entry *= (1 + SLIPPAGE_PCT)
                etf_exit *= (1 - SLIPPAGE_PCT)

                etf_ret = (etf_exit - etf_entry) / etf_entry
                trades.append({
                    'ticker': etf,
                    'entry_date': str(etf_entry_date.date()),
                    'exit_date': str(etf_exit_date.date()),
                    'entry_price': round(etf_entry, 2),
                    'exit_price': round(etf_exit, 2),
                    'return_pct': round(etf_ret * 100, 2),
                    'gap_pct': round(evt['gap_pct'] * 100, 2),
                    'hold_days': 20,
                    'type': 'sector_etf',
                    'trigger_stock': ticker,
                })

    return trades


def run_variant_e(prices, events):
    """Portfolio Rotation: Always hold 3 most recent beaters equally weighted."""
    # Collect all beat events sorted by date
    all_beats = []
    for ticker, evts in events.items():
        for evt in evts:
            if evt['is_beat'] and evt['gap_abs'] >= 0.03 and evt['date'] >= OOT_START:
                all_beats.append({'ticker': ticker, **evt})

    all_beats.sort(key=lambda x: x['date'])

    if not all_beats:
        return []

    # Build daily portfolio: always hold top 3 most recent beaters
    # Rebalance whenever a new beat comes in
    trades = []
    portfolio = []  # list of (ticker, entry_date, entry_price)

    for i, beat in enumerate(all_beats):
        ticker = beat['ticker']
        if ticker not in prices:
            continue
        df = prices[ticker]

        entry_date = get_date_at_offset(df, beat['date'], 1)
        entry_price = get_price_at(df, beat['date'], 'Open', offset_days=1)
        if entry_date is None or entry_price is None:
            continue

        entry_price *= (1 + SLIPPAGE_PCT)

        # If portfolio full, close oldest position
        if len(portfolio) >= 3:
            old = portfolio.pop(0)
            old_ticker = old['ticker']
            if old_ticker in prices:
                old_df = prices[old_ticker]
                exit_price = get_price_at(old_df, entry_date, 'Close', offset_days=0)
                if exit_price is not None:
                    exit_price *= (1 - SLIPPAGE_PCT)
                    ret = (exit_price - old['entry_price']) / old['entry_price']
                    trades.append({
                        'ticker': old_ticker,
                        'entry_date': str(old['entry_date'].date()) if hasattr(old['entry_date'], 'date') else old['entry_date'],
                        'exit_date': str(entry_date.date()),
                        'entry_price': round(old['entry_price'], 2),
                        'exit_price': round(exit_price, 2),
                        'return_pct': round(ret * 100, 2),
                        'gap_pct': round(old['gap_pct'] * 100, 2),
                        'hold_days': (entry_date - old['entry_date']).days if hasattr(old['entry_date'], 'days') else 0,
                    })

        portfolio.append({
            'ticker': ticker,
            'entry_date': entry_date,
            'entry_price': entry_price,
            'gap_pct': beat['gap_pct'],
        })

    # Close remaining positions at end
    for old in portfolio:
        old_ticker = old['ticker']
        if old_ticker in prices:
            old_df = prices[old_ticker]
            exit_price = get_price_at(old_df, old_df.index[-1], 'Close', offset_days=0)
            if exit_price is not None:
                exit_price *= (1 - SLIPPAGE_PCT)
                ret = (exit_price - old['entry_price']) / old['entry_price']
                exit_date = old_df.index[-1]
                hold = (exit_date - old['entry_date']).days if hasattr(old['entry_date'], 'date') else 0
                trades.append({
                    'ticker': old_ticker,
                    'entry_date': str(old['entry_date'].date()) if hasattr(old['entry_date'], 'date') else str(old['entry_date']),
                    'exit_date': str(exit_date.date()),
                    'entry_price': round(old['entry_price'], 2),
                    'exit_price': round(exit_price, 2),
                    'return_pct': round(ret * 100, 2),
                    'gap_pct': round(old['gap_pct'] * 100, 2),
                    'hold_days': hold,
                })

    return trades


def run_variant_f(prices, events):
    """Regime-Adjusted: Only enter beats when SPY > 50-SMA."""
    spy_sma50 = compute_spy_sma(prices, 50)
    spy_close = prices.get('SPY', pd.DataFrame())
    if 'Close' in spy_close.columns:
        spy_close = spy_close['Close']
    else:
        return []

    trades = []

    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]

        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < 0.03:
                continue
            if evt['date'] < OOT_START:
                continue

            # Check regime: SPY > 50-SMA
            if evt['date'] in spy_close.index and evt['date'] in spy_sma50.index:
                spy_val = float(spy_close.loc[evt['date']].iloc[0]) if isinstance(spy_close.loc[evt['date']], pd.Series) else float(spy_close.loc[evt['date']])
                sma_val = float(spy_sma50.loc[evt['date']].iloc[0]) if isinstance(spy_sma50.loc[evt['date']], pd.Series) else float(spy_sma50.loc[evt['date']])
                if spy_val < sma_val:
                    continue  # Skip in risk-off regime
            else:
                continue

            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(df, evt['date'], 'Close', offset_days=HOLD_DAYS)
            exit_date = get_date_at_offset(df, evt['date'], HOLD_DAYS)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': HOLD_DAYS,
            })

    return trades


# ── Analytics ──────────────────────────────────────────────────────────────

def compute_equity_curve(trades, capital=CAPITAL):
    """Build equity curve from sequential trades (simplified: non-overlapping allocation)."""
    if not trades:
        return pd.Series(dtype=float), []

    sorted_trades = sorted(trades, key=lambda t: t['entry_date'])
    equity = capital
    curve = [{'date': sorted_trades[0]['entry_date'], 'equity': capital}]

    for t in sorted_trades:
        # Position size: equal weight across max positions
        pos_size = equity / MAX_POSITIONS
        pnl = pos_size * (t['return_pct'] / 100)
        equity += pnl
        curve.append({
            'date': t['exit_date'],
            'equity': round(equity, 2),
        })

    return equity, curve


def compute_metrics(trades, capital=CAPITAL):
    """Compute strategy metrics with 5-gate validation."""
    if not trades:
        return {
            'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'profit_factor': 0,
            'win_rate': 0, 'max_dd_pct': 0, 'total_return_pct': 0,
            'avg_return_pct': 0, 'median_return_pct': 0,
            'passes_gates': False, 'gate_details': {},
        }

    returns = np.array([t['return_pct'] / 100 for t in trades])
    n = len(returns)

    # Basic stats
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-6

    # Annualize: assume ~4 trades/quarter = ~16/year for most variants
    trades_per_year = max(n / 4.5, 1)  # OOT is ~4.5 years

    # Sharpe (annualized)
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win rate
    win_rate = (returns > 0).sum() / n

    # Max drawdown from equity curve
    equity = capital
    peak = capital
    max_dd = 0
    for r in returns:
        pos_size = equity / MAX_POSITIONS
        equity += pos_size * r
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    final_equity = equity
    total_return = (final_equity - capital) / capital

    # 5-Gate validation
    gate_sharpe = sharpe > 0.5
    gate_maxdd = max_dd > -0.50
    gate_trades = n >= 20

    metrics = {
        'n_trades': n,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'win_rate': round(win_rate * 100, 1),
        'max_dd_pct': round(max_dd * 100, 2),
        'total_return_pct': round(total_return * 100, 2),
        'final_equity': round(final_equity, 2),
        'avg_return_pct': round(avg_ret * 100, 2),
        'median_return_pct': round(np.median(returns) * 100, 2),
        'std_return_pct': round(std_ret * 100, 2),
        'avg_hold_days': round(np.mean([t.get('hold_days', HOLD_DAYS) for t in trades]), 1),
        'gate_sharpe': gate_sharpe,
        'gate_maxdd': gate_maxdd,
        'gate_trades': gate_trades,
    }

    return metrics


def regime_analysis(trades, prices):
    """Stratify returns by bull/bear regime (SPY vs 200-SMA)."""
    sma200 = compute_spy_sma200(prices)
    spy_close = prices.get('SPY', pd.DataFrame())
    if 'Close' in spy_close.columns:
        spy_close = spy_close['Close']
    else:
        return {'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 1.0}

    bull_returns = []
    bear_returns = []

    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        # Find closest date in spy
        mask = spy_close.index <= entry_date
        if not mask.any():
            continue
        closest = spy_close.index[mask][-1]

        if closest in sma200.index:
            spy_val = float(spy_close.loc[closest].iloc[0]) if isinstance(spy_close.loc[closest], pd.Series) else float(spy_close.loc[closest])
            sma_val = float(sma200.loc[closest].iloc[0]) if isinstance(sma200.loc[closest], pd.Series) else float(sma200.loc[closest])

            if np.isnan(sma_val):
                continue

            r = t['return_pct'] / 100
            if spy_val > sma_val:
                bull_returns.append(r)
            else:
                bear_returns.append(r)

    def calc_sharpe(rets):
        if len(rets) < 2:
            return 0
        rets = np.array(rets)
        std = np.std(rets, ddof=1)
        if std == 0:
            return 0
        return np.mean(rets) / std * np.sqrt(max(len(rets) / 4.5, 1))

    bull_sharpe = calc_sharpe(bull_returns)
    bear_sharpe = calc_sharpe(bear_returns)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        'bull_trades': len(bull_returns),
        'bear_trades': len(bear_returns),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'gate_regime': regime_gap < 0.5,
        'bull_avg_ret': round(np.mean(bull_returns) * 100, 2) if bull_returns else 0,
        'bear_avg_ret': round(np.mean(bear_returns) * 100, 2) if bear_returns else 0,
    }


def permutation_test(trades, n_iter=PERM_ITERATIONS):
    """Shuffle entry selections to test significance."""
    if len(trades) < 5:
        return {'perm_p_value': 1.0, 'gate_perm': False}

    returns = np.array([t['return_pct'] / 100 for t in trades])
    actual_mean = np.mean(returns)

    rng = np.random.RandomState(42)
    count_better = 0

    for _ in range(n_iter):
        # Shuffle returns (breaks any temporal signal)
        shuffled = rng.permutation(returns)
        # Also randomly flip signs (null hypothesis: no directional edge)
        signs = rng.choice([-1, 1], size=len(returns))
        null_returns = shuffled * signs
        if np.mean(null_returns) >= actual_mean:
            count_better += 1

    p_value = (count_better + 1) / (n_iter + 1)

    return {
        'perm_p_value': round(p_value, 4),
        'gate_perm': p_value < 0.05,
    }


def validate_5_gates(metrics, regime, perm):
    """Check all 5 gates."""
    gates = {
        'sharpe_gt_0.5': metrics.get('gate_sharpe', False),
        'perm_p_lt_0.05': perm.get('gate_perm', False),
        'regime_gap_lt_0.5': regime.get('gate_regime', False),
        'maxdd_gt_neg50': metrics.get('gate_maxdd', False),
        'trades_gte_20': metrics.get('gate_trades', False),
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EARNINGS SURPRISE MOMENTUM BACKTEST")
    print(f"OOT: {OOT_START.date()} to {OOT_END.date()}")
    print(f"Capital: ${CAPITAL}  |  Max positions: {MAX_POSITIONS}")
    print(f"Universe: {len(UNIVERSE)} stocks")
    print("=" * 70)

    # Download and prep data
    prices = download_data()
    events = detect_earnings_events(prices)

    # Run all 6 variants
    variants = {
        'A_beat_hold_next': run_variant_a,
        'B_beat_chain': run_variant_b,
        'C_big_beat': run_variant_c,
        'D_sector_leaders': run_variant_d,
        'E_portfolio_rotation': run_variant_e,
        'F_regime_adjusted': run_variant_f,
    }

    results = {}

    for name, func in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {name}")
        print(f"{'─' * 60}")

        trades = func(prices, events)
        metrics = compute_metrics(trades)
        regime = regime_analysis(trades, prices)
        perm = permutation_test(trades)
        gates = validate_5_gates(metrics, regime, perm)

        final_equity, curve = compute_equity_curve(trades)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%  |  PF: {metrics['profit_factor']:.2f}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}%  |  Final: ${metrics.get('final_equity', CAPITAL):.2f}")
        print(f"  Max DD: {metrics['max_dd_pct']:.1f}%")
        print(f"  Avg Return/trade: {metrics['avg_return_pct']:.2f}%  |  Median: {metrics['median_return_pct']:.2f}%")
        print(f"  Regime: Bull Sharpe={regime['bull_sharpe']:.3f} ({regime.get('bull_trades',0)} trades), "
              f"Bear Sharpe={regime['bear_sharpe']:.3f} ({regime.get('bear_trades',0)} trades)")
        print(f"  Regime Gap: {regime['regime_gap']:.3f}")
        print(f"  Perm p-value: {perm['perm_p_value']:.4f}")
        print(f"  Gates: {sum(gates[k] for k in gates if k != 'all_pass')}/5 passed  |  "
              f"{'PASS' if gates['all_pass'] else 'FAIL'}")

        for g, v in gates.items():
            if g != 'all_pass':
                print(f"    {'[X]' if v else '[ ]'} {g}")

        # Top trades
        if trades:
            sorted_by_ret = sorted(trades, key=lambda x: x['return_pct'], reverse=True)
            print(f"  Top 3 trades:")
            for t in sorted_by_ret[:3]:
                print(f"    {t['ticker']} {t['entry_date']}: +{t['return_pct']:.1f}% (gap: {t['gap_pct']:.1f}%)")
            if len(sorted_by_ret) > 3:
                print(f"  Worst 3 trades:")
                for t in sorted_by_ret[-3:]:
                    print(f"    {t['ticker']} {t['entry_date']}: {t['return_pct']:.1f}% (gap: {t['gap_pct']:.1f}%)")

        results[name] = {
            'metrics': metrics,
            'regime': regime,
            'permutation': perm,
            'gates': gates,
            'n_trades': metrics['n_trades'],
            'trades': trades[:10] if trades else [],  # Save first 10 for inspection
            'equity_curve_sample': curve[:20] if curve else [],
        }

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY: 5-GATE VALIDATION")
    print("=" * 70)
    print(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'WR%':>6} {'PF':>6} {'MaxDD':>7} {'Perm p':>7} {'Gates':>6} {'Result':>7}")
    print("-" * 85)

    passing = []
    for name, r in results.items():
        m = r['metrics']
        p = r['permutation']
        g = r['gates']
        n_pass = sum(g[k] for k in g if k != 'all_pass')
        status = 'PASS' if g['all_pass'] else 'FAIL'
        print(f"{name:<25} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['win_rate']:>5.1f}% {m['profit_factor']:>6.2f} "
              f"{m['max_dd_pct']:>6.1f}% {p['perm_p_value']:>7.4f} {n_pass:>3}/5  {status:>7}")
        if g['all_pass']:
            passing.append(name)

    if passing:
        print(f"\nPASSING VARIANTS: {', '.join(passing)}")
        best = max(passing, key=lambda n: results[n]['metrics']['sharpe'])
        print(f"BEST: {best} (Sharpe={results[best]['metrics']['sharpe']:.3f})")
    else:
        print("\nNo variants passed all 5 gates.")
        # Find best partial
        best_name = max(results.keys(),
                       key=lambda n: sum(results[n]['gates'][k] for k in results[n]['gates'] if k != 'all_pass'))
        print(f"Best partial: {best_name} ({sum(results[best_name]['gates'][k] for k in results[best_name]['gates'] if k != 'all_pass')}/5 gates)")

    # ── Save Results ─────────────────────────────────────────────────────
    output = {
        'strategy': 'earnings_surprise_momentum',
        'description': 'Post-earnings drift: stocks that beat earnings continue outperforming 60-90 days',
        'academic_basis': 'Bernard & Thomas (1989), Chan et al (1996)',
        'oot_period': f'{OOT_START.date()} to {OOT_END.date()}',
        'capital': CAPITAL,
        'universe_size': len(UNIVERSE),
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION,
        'hold_days': HOLD_DAYS,
        'timestamp': dt.datetime.now().isoformat(),
        'variants': results,
        'passing_variants': passing,
        'best_variant': passing[0] if passing else None,
    }

    out_path = Path('/home/jupiter/Lvl3Quant/data/earnings_surprise_momentum_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
