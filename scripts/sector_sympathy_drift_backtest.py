#!/usr/bin/env python3
"""
Sector Sympathy Earnings Drift Backtest
========================================
Thesis: When a bellwether stock beats earnings (gaps >3% up), its sector ETF
drifts positively via sympathy. Buy the ETF (cheaper, diversified) not the stock.

Builds on Sector Leaders variant (D) from earnings_surprise_momentum_backtest.py
which showed Sharpe 1.792, perm p=0.001.

6 Variants:
  A) 5-day sympathy: Buy sector ETF day after bellwether gaps >3% up. Hold 5 days.
  B) 10-day sympathy: Same, hold 10 days.
  C) 20-day sympathy: Same, hold 20 days.
  D) Multi-bellwether: Only enter if 2+ bellwethers from same sector gap >3% within 5 days.
  E) Regime-filtered: Only enter when SPY > 50-SMA (risk-on).
  F) Affordable options: ATM calls on sector ETF, 3-week expiry, modeled premium.

Walk-forward OOT: Jan 2022 - Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
CAPITAL = 645.0
MAX_POSITIONS = 3
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0       # $0 on ETF shares

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')
PERM_ITERATIONS = 1000
GAP_THRESHOLD = 0.03   # 3% gap = earnings beat proxy

# ── Bellwether -> Sector ETF Mapping ──────────────────────────────────────
SECTOR_MAP = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ADBE', 'ORCL', 'INTC', 'AMD'],
    'XLC': ['META', 'GOOGL', 'NFLX', 'DIS', 'SNAP', 'PINS', 'RBLX', 'RDDT', 'SPOT'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'UBER', 'LYFT', 'ABNB'],
    'XLF': ['JPM', 'BAC', 'GS', 'MS', 'AXP', 'SOFI', 'HOOD', 'COIN'],
    'XLV': ['UNH', 'JNJ', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
    'XLI': ['CAT', 'HON', 'UNP', 'RTX', 'DE', 'BA', 'GE'],
}

# Inverse map: stock -> ETF
STOCK_TO_ETF = {}
for etf, stocks in SECTOR_MAP.items():
    for s in stocks:
        STOCK_TO_ETF[s] = etf

ALL_BELLWETHERS = list(STOCK_TO_ETF.keys())
ALL_ETFS = list(SECTOR_MAP.keys())


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download price data for bellwethers, sector ETFs, and SPY."""
    print("Downloading price data...")
    all_tickers = list(set(ALL_BELLWETHERS + ALL_ETFS + ['SPY']))

    prices = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start='2021-06-01', end=OOT_END.strftime('%Y-%m-%d'),
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 50:
                prices[ticker] = df
        except Exception as e:
            print(f"  Failed {ticker}: {e}")

    print(f"  Downloaded {len(prices)} tickers")
    return prices


def detect_gap_events(prices):
    """
    Detect large gap-up days (>3%) for each bellwether as earnings beat proxy.
    Uses volume spike confirmation and deduplication (1 event per 60-day window).
    """
    events = {}

    for ticker in ALL_BELLWETHERS:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['abs_gap'] = df['gap_pct'].abs()
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma20']

        # Earnings proxy: large positive gap with volume spike
        mask = (df['gap_pct'] > GAP_THRESHOLD) & (df['vol_ratio'] > 1.5)
        gap_days = df[mask].copy()

        if len(gap_days) == 0:
            continue

        # Deduplicate: 1 event per 60-day window per stock
        deduped = []
        last_date = None
        for date, row in gap_days.iterrows():
            if last_date is None or (date - last_date).days > 60:
                deduped.append({
                    'date': date,
                    'gap_pct': float(row['gap_pct']),
                    'gap_abs': float(row['abs_gap']),
                    'vol_ratio': float(row['vol_ratio']),
                    'etf': STOCK_TO_ETF[ticker],
                })
                last_date = date

        events[ticker] = deduped

    total = sum(len(v) for v in events.values())
    print(f"  Detected {total} gap-up events across {len(events)} bellwethers")
    return events


# ── Trade Helpers ─────────────────────────────────────────────────────────
def get_price_at(prices_df, date, field='Open', offset_days=0):
    """Get price at date + offset, handling weekends/holidays."""
    if date not in prices_df.index:
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


def get_spy_val(spy_close, sma, date):
    """Safely get SPY close and SMA value at a date."""
    if date not in spy_close.index or date not in sma.index:
        # find closest prior date
        mask = spy_close.index <= date
        if not mask.any():
            return None, None
        closest = spy_close.index[mask][-1]
        if closest not in sma.index:
            return None, None
        date = closest

    spy_val = spy_close.loc[date]
    if isinstance(spy_val, pd.Series):
        spy_val = spy_val.iloc[0]
    sma_val = sma.loc[date]
    if isinstance(sma_val, pd.Series):
        sma_val = sma_val.iloc[0]
    return float(spy_val), float(sma_val)


def simulate_position_limits(trades, max_concurrent=MAX_POSITIONS):
    """Filter trades to respect max concurrent position limit.
    First-come-first-served: skip trades that would exceed limit."""
    sorted_trades = sorted(trades, key=lambda t: t['entry_date'])
    active = []  # (exit_date,)
    accepted = []

    for t in sorted_trades:
        entry = t['entry_date']
        # Clear expired positions
        active = [a for a in active if a > entry]
        if len(active) < max_concurrent:
            active.append(t['exit_date'])
            accepted.append(t)

    return accepted


# ── Variant Backtests ────────────────────────────────────────────────────

def _run_sympathy(prices, events, hold_days, label):
    """Core sympathy logic: buy sector ETF day after bellwether gaps up."""
    raw_trades = []

    for ticker, evts in events.items():
        etf = STOCK_TO_ETF.get(ticker)
        if etf is None or etf not in prices:
            continue
        etf_df = prices[etf]

        for evt in evts:
            if evt['date'] < OOT_START:
                continue

            entry_price = get_price_at(etf_df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(etf_df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(etf_df, evt['date'], 'Close', offset_days=hold_days)
            exit_date = get_date_at_offset(etf_df, evt['date'], hold_days)
            if exit_price is None or exit_date is None:
                continue

            # Apply slippage
            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            raw_trades.append({
                'ticker': etf,
                'trigger_stock': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': hold_days,
            })

    # Apply position limits
    trades = simulate_position_limits(raw_trades)
    return trades


def run_variant_a(prices, events):
    """5-day sympathy: Buy sector ETF day after bellwether gaps >3%. Hold 5 days."""
    return _run_sympathy(prices, events, hold_days=5, label='A')


def run_variant_b(prices, events):
    """10-day sympathy: Hold 10 days."""
    return _run_sympathy(prices, events, hold_days=10, label='B')


def run_variant_c(prices, events):
    """20-day sympathy: Hold 20 days (capture longer drift)."""
    return _run_sympathy(prices, events, hold_days=20, label='C')


def run_variant_d(prices, events):
    """Multi-bellwether confirmation: Only enter if 2+ bellwethers from same sector
    gap >3% within 5 trading days."""
    # Group events by sector ETF
    etf_events = defaultdict(list)
    for ticker, evts in events.items():
        etf = STOCK_TO_ETF.get(ticker)
        if etf is None:
            continue
        for evt in evts:
            if evt['date'] >= OOT_START:
                etf_events[etf].append({
                    'stock': ticker,
                    'date': evt['date'],
                    'gap_pct': evt['gap_pct'],
                })

    # Sort by date within each sector
    for etf in etf_events:
        etf_events[etf].sort(key=lambda x: x['date'])

    raw_trades = []

    for etf, evts in etf_events.items():
        if etf not in prices:
            continue
        etf_df = prices[etf]

        # Sliding window: for each event, check if another event from a DIFFERENT
        # bellwether in same sector happened within prior 5 trading days
        for i, evt in enumerate(evts):
            confirming = []
            for j in range(max(0, i - 10), i):
                other = evts[j]
                if other['stock'] != evt['stock']:
                    day_diff = (evt['date'] - other['date']).days
                    if 0 < day_diff <= 7:  # ~5 trading days
                        confirming.append(other['stock'])

            if len(confirming) == 0:
                continue  # No confirmation

            entry_price = get_price_at(etf_df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(etf_df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            # Hold 10 days for confirmed signals
            exit_price = get_price_at(etf_df, evt['date'], 'Close', offset_days=10)
            exit_date = get_date_at_offset(etf_df, evt['date'], 10)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            raw_trades.append({
                'ticker': etf,
                'trigger_stock': evt['stock'],
                'confirming_stocks': confirming,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': 10,
            })

    trades = simulate_position_limits(raw_trades)
    return trades


def run_variant_e(prices, events):
    """Regime-filtered: Only enter when SPY > 50-SMA (risk-on)."""
    spy_sma50 = compute_spy_sma(prices, 50)
    spy_close = prices.get('SPY', pd.DataFrame())
    if 'Close' in spy_close.columns:
        spy_close = spy_close['Close']
    else:
        return []

    raw_trades = []

    for ticker, evts in events.items():
        etf = STOCK_TO_ETF.get(ticker)
        if etf is None or etf not in prices:
            continue
        etf_df = prices[etf]

        for evt in evts:
            if evt['date'] < OOT_START:
                continue

            # Regime filter
            spy_val, sma_val = get_spy_val(spy_close, spy_sma50, evt['date'])
            if spy_val is None or sma_val is None or np.isnan(sma_val):
                continue
            if spy_val < sma_val:
                continue  # risk-off, skip

            entry_price = get_price_at(etf_df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(etf_df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(etf_df, evt['date'], 'Close', offset_days=10)
            exit_date = get_date_at_offset(etf_df, evt['date'], 10)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price
            raw_trades.append({
                'ticker': etf,
                'trigger_stock': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': 10,
            })

    trades = simulate_position_limits(raw_trades)
    return trades


def run_variant_f(prices, events):
    """Affordable options: Buy ATM calls (3-week expiry) on sector ETF.
    Premium model: 2% of ETF price, $0.65 commission, 5% bid-ask spread.
    Only enter if option cost < $200."""
    raw_trades = []

    for ticker, evts in events.items():
        etf = STOCK_TO_ETF.get(ticker)
        if etf is None or etf not in prices:
            continue
        etf_df = prices[etf]

        for evt in evts:
            if evt['date'] < OOT_START:
                continue

            entry_date = get_date_at_offset(etf_df, evt['date'], 1)
            if entry_date is None:
                continue
            etf_price_at_entry = get_price_at(etf_df, evt['date'], 'Open', offset_days=1)
            if etf_price_at_entry is None:
                continue

            # Model ATM call premium: 2% of ETF price * 100 shares per contract
            premium_per_share = etf_price_at_entry * 0.02
            option_cost = premium_per_share * 100  # 1 contract = 100 shares

            # Only enter if affordable
            if option_cost > 200:
                continue

            # Total cost: premium + commission + bid-ask slippage (5% of premium)
            total_cost = option_cost + 0.65 + (option_cost * 0.05)

            # Exit after 15 trading days (~3 weeks)
            exit_price = get_price_at(etf_df, evt['date'], 'Close', offset_days=15)
            exit_date = get_date_at_offset(etf_df, evt['date'], 15)
            if exit_price is None or exit_date is None:
                continue

            # Intrinsic value at exit (ATM call, strike = entry price)
            strike = etf_price_at_entry
            intrinsic = max(0, exit_price - strike) * 100

            # P&L = intrinsic - total_cost (ignore remaining time value for simplicity)
            pnl = intrinsic - total_cost
            ret = pnl / total_cost if total_cost > 0 else 0

            raw_trades.append({
                'ticker': etf,
                'trigger_stock': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(total_cost, 2),
                'exit_price': round(intrinsic, 2),
                'option_cost': round(total_cost, 2),
                'intrinsic_at_exit': round(intrinsic, 2),
                'etf_entry_price': round(etf_price_at_entry, 2),
                'etf_exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': 15,
            })

    trades = simulate_position_limits(raw_trades)
    return trades


# ── Analytics ────────────────────────────────────────────────────────────

def compute_equity_curve(trades, capital=CAPITAL):
    """Build equity curve from sequential trades with position sizing."""
    if not trades:
        return capital, []

    sorted_trades = sorted(trades, key=lambda t: t['entry_date'])
    equity = capital
    curve = [{'date': sorted_trades[0]['entry_date'], 'equity': capital}]

    for t in sorted_trades:
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

    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-6

    # Annualize: OOT is ~4.5 years
    trades_per_year = max(n / 4.5, 1)

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

    # Max drawdown
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

    gate_sharpe = sharpe > 0.5
    gate_maxdd = max_dd > -0.50
    gate_trades = n >= 20

    return {
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
        'avg_hold_days': round(np.mean([t.get('hold_days', 10) for t in trades]), 1),
        'gate_sharpe': gate_sharpe,
        'gate_maxdd': gate_maxdd,
        'gate_trades': gate_trades,
    }


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
        mask = spy_close.index <= entry_date
        if not mask.any():
            continue
        closest = spy_close.index[mask][-1]

        spy_val, sma_val = get_spy_val(spy_close, sma200, closest)
        if spy_val is None or sma_val is None or np.isnan(sma_val):
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


def permutation_test(trades, prices, n_iter=PERM_ITERATIONS):
    """Shuffle which sector ETF gets bought (random sector instead of triggered sector).
    This tests whether the SECTOR SELECTION matters, not just being long any ETF."""
    if len(trades) < 5:
        return {'perm_p_value': 1.0, 'gate_perm': False}

    actual_returns = np.array([t['return_pct'] / 100 for t in trades])
    actual_mean = np.mean(actual_returns)

    # Get all ETF price data for random assignment
    etf_list = [etf for etf in ALL_ETFS if etf in prices]
    if not etf_list:
        return {'perm_p_value': 1.0, 'gate_perm': False}

    rng = np.random.RandomState(42)
    count_better = 0

    for _ in range(n_iter):
        perm_returns = []
        for t in trades:
            # Randomly assign a different sector ETF
            random_etf = rng.choice(etf_list)
            if random_etf not in prices:
                perm_returns.append(0)
                continue
            etf_df = prices[random_etf]

            entry_date = pd.Timestamp(t['entry_date'])
            exit_date = pd.Timestamp(t['exit_date'])

            entry_p = get_price_at(etf_df, entry_date, 'Open', 0)
            exit_p = get_price_at(etf_df, exit_date, 'Close', 0)

            if entry_p is None or exit_p is None or entry_p == 0:
                perm_returns.append(0)
                continue

            perm_ret = (exit_p - entry_p) / entry_p
            perm_returns.append(perm_ret)

        if np.mean(perm_returns) >= actual_mean:
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


def sector_breakdown(trades):
    """Show performance by sector ETF."""
    by_sector = defaultdict(list)
    for t in trades:
        by_sector[t['ticker']].append(t['return_pct'])

    breakdown = {}
    for etf, rets in sorted(by_sector.items()):
        rets_arr = np.array(rets)
        breakdown[etf] = {
            'n_trades': len(rets),
            'avg_return_pct': round(np.mean(rets_arr), 2),
            'win_rate': round((rets_arr > 0).sum() / len(rets_arr) * 100, 1),
            'total_return_pct': round(np.sum(rets_arr), 2),
        }
    return breakdown


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("SECTOR SYMPATHY EARNINGS DRIFT BACKTEST")
    print(f"OOT: {OOT_START.date()} to {OOT_END.date()}")
    print(f"Capital: ${CAPITAL}  |  Max concurrent: {MAX_POSITIONS}")
    print(f"Bellwethers: {len(ALL_BELLWETHERS)}  |  Sector ETFs: {len(ALL_ETFS)}")
    print(f"Gap threshold: {GAP_THRESHOLD*100:.0f}%  |  Slippage: {SLIPPAGE_PCT*100:.3f}%")
    print("=" * 70)

    prices = download_data()
    events = detect_gap_events(prices)

    # Show event distribution by sector
    print("\nEvents by sector:")
    sector_counts = defaultdict(int)
    for ticker, evts in events.items():
        etf = STOCK_TO_ETF.get(ticker)
        oot_evts = [e for e in evts if e['date'] >= OOT_START]
        sector_counts[etf] += len(oot_evts)
    for etf, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"  {etf}: {count} trigger events")

    variants = {
        'A_5day_sympathy': run_variant_a,
        'B_10day_sympathy': run_variant_b,
        'C_20day_sympathy': run_variant_c,
        'D_multi_bellwether': run_variant_d,
        'E_regime_filtered': run_variant_e,
        'F_affordable_options': run_variant_f,
    }

    results = {}

    for name, func in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {name}")
        print(f"{'─' * 60}")

        trades = func(prices, events)
        metrics = compute_metrics(trades)
        regime = regime_analysis(trades, prices)
        perm = permutation_test(trades, prices)
        gates = validate_5_gates(metrics, regime, perm)
        final_equity, curve = compute_equity_curve(trades)
        sectors = sector_breakdown(trades) if trades else {}

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%  |  PF: {metrics['profit_factor']:.2f}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}%  |  Final: ${metrics.get('final_equity', CAPITAL):.2f}")
        print(f"  Max DD: {metrics['max_dd_pct']:.1f}%")
        print(f"  Avg Return/trade: {metrics['avg_return_pct']:.2f}%  |  Median: {metrics['median_return_pct']:.2f}%")
        print(f"  Regime: Bull Sharpe={regime['bull_sharpe']:.3f} ({regime.get('bull_trades',0)}), "
              f"Bear Sharpe={regime['bear_sharpe']:.3f} ({regime.get('bear_trades',0)})")
        print(f"  Regime Gap: {regime['regime_gap']:.3f}")
        print(f"  Perm p-value: {perm['perm_p_value']:.4f}")
        print(f"  Gates: {sum(gates[k] for k in gates if k != 'all_pass')}/5 passed  |  "
              f"{'PASS' if gates['all_pass'] else 'FAIL'}")

        for g, v in gates.items():
            if g != 'all_pass':
                print(f"    {'[X]' if v else '[ ]'} {g}")

        if sectors:
            print(f"  Sector breakdown:")
            for etf, s in sorted(sectors.items(), key=lambda x: -x[1]['avg_return_pct']):
                print(f"    {etf}: {s['n_trades']} trades, avg {s['avg_return_pct']:+.2f}%, WR {s['win_rate']:.0f}%")

        if trades:
            sorted_by_ret = sorted(trades, key=lambda x: x['return_pct'], reverse=True)
            print(f"  Top 3 trades:")
            for t in sorted_by_ret[:3]:
                print(f"    {t['ticker']} (via {t.get('trigger_stock','?')}) {t['entry_date']}: +{t['return_pct']:.1f}%")
            if len(sorted_by_ret) > 3:
                print(f"  Worst 3 trades:")
                for t in sorted_by_ret[-3:]:
                    print(f"    {t['ticker']} (via {t.get('trigger_stock','?')}) {t['entry_date']}: {t['return_pct']:.1f}%")

        results[name] = {
            'metrics': metrics,
            'regime': regime,
            'permutation': perm,
            'gates': gates,
            'n_trades': metrics['n_trades'],
            'sector_breakdown': sectors,
            'trades': trades[:15] if trades else [],
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
        best_name = max(results.keys(),
                        key=lambda n: sum(results[n]['gates'][k] for k in results[n]['gates'] if k != 'all_pass'))
        best_sharpe = max(results.keys(), key=lambda n: results[n]['metrics']['sharpe'])
        print(f"Best partial (gates): {best_name} ({sum(results[best_name]['gates'][k] for k in results[best_name]['gates'] if k != 'all_pass')}/5)")
        print(f"Best Sharpe: {best_sharpe} ({results[best_sharpe]['metrics']['sharpe']:.3f})")

    # ── Save Results ─────────────────────────────────────────────────────
    output = {
        'strategy': 'sector_sympathy_earnings_drift',
        'description': 'When bellwether stock beats earnings (gaps >3%), buy sector ETF for sympathy drift',
        'thesis': 'Bellwether beat signals sector health; ETF captures diversified sympathy drift cheaply',
        'builds_on': 'Sector Leaders variant (D) from earnings_surprise_momentum: Sharpe 1.792, perm p=0.001',
        'oot_period': f'{OOT_START.date()} to {OOT_END.date()}',
        'capital': CAPITAL,
        'max_concurrent_positions': MAX_POSITIONS,
        'bellwethers': len(ALL_BELLWETHERS),
        'sector_etfs': ALL_ETFS,
        'gap_threshold': GAP_THRESHOLD,
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION,
        'timestamp': dt.datetime.now().isoformat(),
        'variants': results,
        'passing_variants': passing,
        'best_variant': passing[0] if passing else None,
    }

    out_path = Path('/home/jupiter/Lvl3Quant/data/sector_sympathy_drift_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
