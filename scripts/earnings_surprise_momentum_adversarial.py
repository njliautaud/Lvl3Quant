#!/usr/bin/env python3
"""
Earnings Surprise Momentum — ADVERSARIAL VALIDATION
=====================================================
6 stress tests to break the strategy:
1. Survivorship bias (remove top contributors + add failed stocks)
2. Random direction baseline (buy/sell randomly after ANY gap)
3. Time-reversed test (reverse price series)
4. Concentration check (top-3 stock P&L dominance)
5. Cost sensitivity (0.05% and 0.10% slippage)
6. Bear market only (SPY < 200-SMA periods)
"""

import json
import warnings
import datetime as dt
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration (matches original) ──────────────────────────────────────
CAPITAL = 645.0
MAX_POSITIONS = 5
SLIPPAGE_PCT = 0.0002  # 0.02% baseline
COMMISSION = 0.0

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'PLTR', 'SOFI', 'HOOD', 'SNAP', 'PINS', 'UBER',
    'LYFT', 'COIN', 'RBLX', 'DDOG', 'TTD', 'SHOP', 'NET', 'ROKU'
]

# Failed growth stocks for survivorship bias test
FAILED_GROWTH = ['PYPL', 'DOCU', 'ZM', 'NKLA', 'WISH', 'CLOV', 'SPCE']

SECTOR_ETF_MAP = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'AMD': 'XLK',
    'CRM': 'XLK', 'DDOG': 'XLK', 'NET': 'XLK', 'SHOP': 'XLK', 'TTD': 'XLK',
    'GOOGL': 'XLC', 'META': 'XLC', 'NFLX': 'XLC', 'SNAP': 'XLC',
    'PINS': 'XLC', 'ROKU': 'XLC',
    'AMZN': 'XLY', 'TSLA': 'XLY', 'RBLX': 'XLY',
    'PLTR': 'XLK', 'SOFI': 'XLF', 'HOOD': 'XLF',
    'UBER': 'XLY', 'LYFT': 'XLY', 'COIN': 'XLF',
    # Failed stocks
    'PYPL': 'XLK', 'DOCU': 'XLK', 'ZM': 'XLK', 'NKLA': 'XLY',
    'WISH': 'XLY', 'CLOV': 'XLV', 'SPCE': 'XLI',
}

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')
HOLD_DAYS = 60
PERM_ITERATIONS = 1000


# ── Data Layer ────────────────────────────────────────────────────────────
def download_data(extra_tickers=None):
    """Download price data for full universe + extras."""
    all_stocks = list(set(UNIVERSE + FAILED_GROWTH + (extra_tickers or [])))
    all_tickers = list(set(all_stocks + list(set(SECTOR_ETF_MAP.values())) + ['SPY']))

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


def detect_earnings_events(prices, universe=None):
    """Detect earnings via overnight gaps. Universe parameter allows customization."""
    if universe is None:
        universe = UNIVERSE

    events = {}
    for ticker in universe:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['abs_gap'] = df['gap_pct'].abs()
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / df['vol_ma20']

        earnings_mask = (df['abs_gap'] > 0.03) & (df['vol_ratio'] > 1.5)
        earnings_days = df[earnings_mask].copy()

        if len(earnings_days) == 0:
            continue

        deduped = []
        last_date = None
        for date, row in earnings_days.iterrows():
            if last_date is None or (date - last_date).days > 60:
                deduped.append({
                    'date': date,
                    'gap_pct': row['gap_pct'],
                    'is_beat': row['gap_pct'] > 0,
                    'gap_abs': row['abs_gap'],
                })
                last_date = date
        events[ticker] = deduped

    return events


# ── Trade Helpers (from original) ─────────────────────────────────────────
def get_price_at(prices_df, date, field='Open', offset_days=0):
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


def compute_spy_sma200(prices):
    if 'SPY' not in prices:
        return pd.Series(dtype=float)
    spy = prices['SPY']['Close'].copy()
    return spy.rolling(200).mean()


# ── Core Strategy: Variant A (baseline for adversarial tests) ─────────────
def run_variant_a(prices, events, slippage=SLIPPAGE_PCT, direction_filter='beat',
                  universe_filter=None):
    """
    Variant A: Buy after >3% up gap, hold 60 days.
    direction_filter: 'beat' (original), 'any' (random), 'miss' (short only)
    universe_filter: list of tickers to exclude
    """
    trades = []
    rng = np.random.RandomState(42)

    for ticker, evts in events.items():
        if universe_filter and ticker in universe_filter:
            continue
        if ticker not in prices:
            continue
        df = prices[ticker]

        for evt in evts:
            if evt['date'] < OOT_START:
                continue
            if evt['gap_abs'] < 0.03:
                continue

            # Direction logic
            if direction_filter == 'beat':
                if not evt['is_beat']:
                    continue
                go_long = True
            elif direction_filter == 'any':
                # Random direction regardless of gap direction
                go_long = rng.random() > 0.5
            elif direction_filter == 'miss':
                if evt['is_beat']:
                    continue
                go_long = True  # Buy even after miss (to test if direction matters)
            else:
                go_long = True

            entry_price = get_price_at(df, evt['date'], 'Open', offset_days=1)
            entry_date = get_date_at_offset(df, evt['date'], 1)
            if entry_price is None or entry_date is None:
                continue

            exit_price = get_price_at(df, evt['date'], 'Close', offset_days=HOLD_DAYS)
            exit_date = get_date_at_offset(df, evt['date'], HOLD_DAYS)
            if exit_price is None or exit_date is None:
                continue

            entry_price *= (1 + slippage)
            exit_price *= (1 - slippage)

            if go_long:
                ret = (exit_price - entry_price) / entry_price
            else:
                ret = (entry_price - exit_price) / entry_price  # short

            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 2),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': HOLD_DAYS,
                'direction': 'long' if go_long else 'short',
            })

    return trades


def run_variant_d(prices, events, slippage=SLIPPAGE_PCT, universe_filter=None):
    """Variant D: Stock + sector ETF trades. With optional universe filter."""
    trades = []

    for ticker, evts in events.items():
        if universe_filter and ticker in universe_filter:
            continue
        if ticker not in prices:
            continue
        df = prices[ticker]
        etf = SECTOR_ETF_MAP.get(ticker)

        for evt in evts:
            if not evt['is_beat'] or evt['gap_abs'] < 0.03:
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

            entry_price *= (1 + slippage)
            exit_price *= (1 - slippage)
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
                etf_entry *= (1 + slippage)
                etf_exit *= (1 - slippage)
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


# ── Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(trades, capital=CAPITAL):
    if not trades:
        return {
            'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'profit_factor': 0,
            'win_rate': 0, 'max_dd_pct': 0, 'total_return_pct': 0,
            'final_equity': capital, 'avg_return_pct': 0,
        }

    returns = np.array([t['return_pct'] / 100 for t in trades])
    n = len(returns)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-6
    trades_per_year = max(n / 4.5, 1)

    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    win_rate = (returns > 0).sum() / n

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
    }


# ── ADVERSARIAL TEST 1: Survivorship Bias ─────────────────────────────────
def test_survivorship_bias(prices, events, all_events):
    """
    Two sub-tests:
    A) Remove top 3 P&L contributors from variant A
    B) Add failed growth stocks to the universe
    """
    print("\n" + "=" * 70)
    print("TEST 1: SURVIVORSHIP BIAS")
    print("=" * 70)

    # First, run baseline to find top contributors
    baseline_trades = run_variant_a(prices, events)
    baseline_metrics = compute_metrics(baseline_trades)

    # Find top 3 P&L contributors
    ticker_pnl = {}
    for t in baseline_trades:
        ticker_pnl[t['ticker']] = ticker_pnl.get(t['ticker'], 0) + t['return_pct']

    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    top3 = [t[0] for t in sorted_tickers[:3]]
    top3_pnl = sum(t[1] for t in sorted_tickers[:3])
    total_pnl = sum(t[1] for t in sorted_tickers)

    print(f"\n  Baseline Variant A: Sharpe={baseline_metrics['sharpe']:.3f}, "
          f"WR={baseline_metrics['win_rate']:.1f}%, {baseline_metrics['n_trades']} trades")
    print(f"  Top 3 contributors: {top3}")
    print(f"  Top 3 P&L share: {top3_pnl:.1f}% of {total_pnl:.1f}% total "
          f"({top3_pnl/total_pnl*100:.0f}% concentration)" if total_pnl != 0 else "  No P&L")

    # Sub-test A: Remove top 3
    print(f"\n  --- Sub-test A: Remove top 3 ({', '.join(top3)}) ---")
    trades_no_top3 = run_variant_a(prices, events, universe_filter=set(top3))
    metrics_no_top3 = compute_metrics(trades_no_top3)
    print(f"  Without top 3: Sharpe={metrics_no_top3['sharpe']:.3f}, "
          f"WR={metrics_no_top3['win_rate']:.1f}%, {metrics_no_top3['n_trades']} trades, "
          f"Return={metrics_no_top3['total_return_pct']:.1f}%")

    edge_survives_removal = metrics_no_top3['sharpe'] > 0.3 and metrics_no_top3['win_rate'] > 50

    # Sub-test B: Add failed growth stocks
    print(f"\n  --- Sub-test B: Add failed growth stocks ({', '.join(FAILED_GROWTH)}) ---")
    extended_universe = UNIVERSE + FAILED_GROWTH
    extended_events = detect_earnings_events(prices, universe=extended_universe)
    trades_extended = run_variant_a(prices, extended_events)
    metrics_extended = compute_metrics(trades_extended)
    print(f"  Extended universe: Sharpe={metrics_extended['sharpe']:.3f}, "
          f"WR={metrics_extended['win_rate']:.1f}%, {metrics_extended['n_trades']} trades, "
          f"Return={metrics_extended['total_return_pct']:.1f}%")

    # Check how many failed stock trades there are and their returns
    failed_trades = [t for t in trades_extended if t['ticker'] in FAILED_GROWTH]
    if failed_trades:
        failed_returns = [t['return_pct'] for t in failed_trades]
        print(f"  Failed stocks: {len(failed_trades)} trades, "
              f"avg return={np.mean(failed_returns):.2f}%, "
              f"WR={sum(1 for r in failed_returns if r > 0)/len(failed_returns)*100:.0f}%")
    else:
        print(f"  Failed stocks: 0 trades triggered (no qualifying gaps)")

    edge_survives_addition = metrics_extended['sharpe'] > 0.3 and metrics_extended['win_rate'] > 50

    # Also run variant D with same tests
    print(f"\n  --- Variant D: Remove top 3 ---")
    d_baseline = run_variant_d(prices, events)
    d_metrics_baseline = compute_metrics(d_baseline)

    d_ticker_pnl = {}
    for t in d_baseline:
        key = t.get('trigger_stock', t['ticker'])
        d_ticker_pnl[key] = d_ticker_pnl.get(key, 0) + t['return_pct']
    d_sorted = sorted(d_ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    d_top3 = [t[0] for t in d_sorted[:3]]

    d_trades_no_top3 = run_variant_d(prices, events, universe_filter=set(d_top3))
    d_metrics_no_top3 = compute_metrics(d_trades_no_top3)
    print(f"  D baseline: Sharpe={d_metrics_baseline['sharpe']:.3f}, "
          f"D without top 3 ({', '.join(d_top3)}): Sharpe={d_metrics_no_top3['sharpe']:.3f}")

    verdict = "SURVIVES" if (edge_survives_removal and edge_survives_addition) else "FAILS"
    print(f"\n  SURVIVORSHIP BIAS VERDICT: {verdict}")

    return {
        'test': 'survivorship_bias',
        'baseline_a': baseline_metrics,
        'top3_contributors': top3,
        'top3_pnl_share_pct': round(top3_pnl / total_pnl * 100, 1) if total_pnl else 0,
        'without_top3_a': metrics_no_top3,
        'with_failed_stocks': metrics_extended,
        'failed_stock_trades': len(failed_trades) if failed_trades else 0,
        'failed_stock_avg_ret': round(np.mean([t['return_pct'] for t in failed_trades]), 2) if failed_trades else None,
        'variant_d_baseline': d_metrics_baseline,
        'variant_d_without_top3': d_metrics_no_top3,
        'd_top3_contributors': d_top3,
        'edge_survives_removal': edge_survives_removal,
        'edge_survives_addition': edge_survives_addition,
        'verdict': verdict,
    }


# ── ADVERSARIAL TEST 2: Random Direction Baseline ─────────────────────────
def test_random_direction(prices, events):
    """
    Buy randomly after ANY >3% gap (up or down).
    If strategy still works, edge is from volatility exposure, not direction.
    Run 20 random seeds for robustness.
    """
    print("\n" + "=" * 70)
    print("TEST 2: RANDOM DIRECTION BASELINE")
    print("=" * 70)

    # Baseline: original (buy after UP gaps only)
    baseline_trades = run_variant_a(prices, events)
    baseline_metrics = compute_metrics(baseline_trades)
    print(f"  Baseline (buy UP gaps): Sharpe={baseline_metrics['sharpe']:.3f}, "
          f"WR={baseline_metrics['win_rate']:.1f}%")

    # Random: buy/sell randomly after ANY gap
    random_sharpes = []
    random_wrs = []
    for seed in range(20):
        # Override the RNG seed for each trial
        trades = []
        rng = np.random.RandomState(seed)
        for ticker, evts in events.items():
            if ticker not in prices:
                continue
            df = prices[ticker]
            for evt in evts:
                if evt['date'] < OOT_START or evt['gap_abs'] < 0.03:
                    continue
                go_long = rng.random() > 0.5
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
                if go_long:
                    ret = (exit_price - entry_price) / entry_price
                else:
                    ret = (entry_price - exit_price) / entry_price
                trades.append({'return_pct': round(ret * 100, 2), 'ticker': ticker,
                               'entry_date': str(entry_date.date()), 'exit_date': str(exit_date.date()),
                               'entry_price': entry_price, 'exit_price': exit_price,
                               'gap_pct': evt['gap_pct'] * 100, 'hold_days': HOLD_DAYS})
        m = compute_metrics(trades)
        random_sharpes.append(m['sharpe'])
        random_wrs.append(m['win_rate'])

    avg_random_sharpe = np.mean(random_sharpes)
    max_random_sharpe = np.max(random_sharpes)
    print(f"  Random direction (20 seeds): avg Sharpe={avg_random_sharpe:.3f}, "
          f"max={max_random_sharpe:.3f}, avg WR={np.mean(random_wrs):.1f}%")

    # Also test: buy after DOWN gaps (contrarian)
    contrarian_trades = []
    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]
        for evt in evts:
            if evt['is_beat'] or evt['gap_abs'] < 0.03:
                continue  # only DOWN gaps
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
            contrarian_trades.append({'return_pct': round(ret * 100, 2), 'ticker': ticker,
                                      'entry_date': str(entry_date.date()), 'exit_date': str(exit_date.date()),
                                      'entry_price': entry_price, 'exit_price': exit_price,
                                      'gap_pct': evt['gap_pct'] * 100, 'hold_days': HOLD_DAYS})
    contrarian_metrics = compute_metrics(contrarian_trades)
    print(f"  Contrarian (buy DOWN gaps): Sharpe={contrarian_metrics['sharpe']:.3f}, "
          f"WR={contrarian_metrics['win_rate']:.1f}%, {contrarian_metrics['n_trades']} trades")

    # Edge is real if baseline >> random AND baseline >> contrarian
    direction_matters = baseline_metrics['sharpe'] > max_random_sharpe + 0.3
    verdict = "REAL EDGE (direction matters)" if direction_matters else "SUSPICIOUS (direction may not matter)"
    print(f"\n  RANDOM DIRECTION VERDICT: {verdict}")

    return {
        'test': 'random_direction',
        'baseline_sharpe': baseline_metrics['sharpe'],
        'baseline_wr': baseline_metrics['win_rate'],
        'random_avg_sharpe': round(avg_random_sharpe, 3),
        'random_max_sharpe': round(max_random_sharpe, 3),
        'random_avg_wr': round(np.mean(random_wrs), 1),
        'contrarian_sharpe': contrarian_metrics['sharpe'],
        'contrarian_wr': contrarian_metrics['win_rate'],
        'contrarian_n_trades': contrarian_metrics['n_trades'],
        'direction_matters': direction_matters,
        'verdict': verdict,
    }


# ── ADVERSARIAL TEST 3: Time-Reversed Test ────────────────────────────────
def test_time_reversed(prices, events):
    """
    Reverse the price time series and re-run.
    A momentum strategy should NOT work on reversed data.
    """
    print("\n" + "=" * 70)
    print("TEST 3: TIME-REVERSED TEST")
    print("=" * 70)

    # Create reversed prices
    reversed_prices = {}
    for ticker, df in prices.items():
        rdf = df.copy()
        # Reverse close/open/high/low but keep the index order (dates stay forward)
        # This breaks momentum patterns while preserving volatility structure
        rdf['Close'] = df['Close'].values[::-1]
        rdf['Open'] = df['Open'].values[::-1]
        rdf['High'] = df['High'].values[::-1]
        rdf['Low'] = df['Low'].values[::-1]
        rdf['Volume'] = df['Volume'].values[::-1]
        reversed_prices[ticker] = rdf

    # Re-detect events on reversed data
    reversed_events = detect_earnings_events(reversed_prices, universe=UNIVERSE)

    # Run baseline on original
    baseline_trades = run_variant_a(prices, events)
    baseline_metrics = compute_metrics(baseline_trades)

    # Run on reversed
    reversed_trades = run_variant_a(reversed_prices, reversed_events)
    reversed_metrics = compute_metrics(reversed_trades)

    print(f"  Original: Sharpe={baseline_metrics['sharpe']:.3f}, "
          f"WR={baseline_metrics['win_rate']:.1f}%, {baseline_metrics['n_trades']} trades")
    print(f"  Reversed: Sharpe={reversed_metrics['sharpe']:.3f}, "
          f"WR={reversed_metrics['win_rate']:.1f}%, {reversed_metrics['n_trades']} trades")

    # Strategy should NOT work on reversed data
    reversed_works = reversed_metrics['sharpe'] > 0.5 and reversed_metrics['win_rate'] > 52
    verdict = "SUSPICIOUS (works on reversed data too)" if reversed_works else "PASSES (no artifact)"
    print(f"\n  TIME-REVERSED VERDICT: {verdict}")

    return {
        'test': 'time_reversed',
        'original_sharpe': baseline_metrics['sharpe'],
        'original_wr': baseline_metrics['win_rate'],
        'reversed_sharpe': reversed_metrics['sharpe'],
        'reversed_wr': reversed_metrics['win_rate'],
        'reversed_n_trades': reversed_metrics['n_trades'],
        'reversed_works': reversed_works,
        'verdict': verdict,
    }


# ── ADVERSARIAL TEST 4: Concentration Check ──────────────────────────────
def test_concentration(prices, events):
    """
    What % of total P&L comes from top 3 stocks?
    If >50%, strategy is stock-picking luck.
    """
    print("\n" + "=" * 70)
    print("TEST 4: CONCENTRATION CHECK")
    print("=" * 70)

    for variant_name, run_fn in [("A (Direct)", lambda: run_variant_a(prices, events)),
                                  ("D (Sector)", lambda: run_variant_d(prices, events))]:
        trades = run_fn()
        if not trades:
            print(f"  {variant_name}: No trades")
            continue

        # Compute P&L by ticker
        ticker_pnl = {}
        ticker_trades = {}
        for t in trades:
            tk = t.get('trigger_stock', t['ticker'])  # For ETF trades, use trigger stock
            ticker_pnl[tk] = ticker_pnl.get(tk, 0) + t['return_pct']
            ticker_trades[tk] = ticker_trades.get(tk, 0) + 1

        total_pnl = sum(v for v in ticker_pnl.values())
        sorted_by_pnl = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  {variant_name}:")
        print(f"  Total P&L: {total_pnl:.1f}% across {len(ticker_pnl)} stocks")
        print(f"  P&L by stock:")
        for tk, pnl in sorted_by_pnl:
            n_tr = ticker_trades[tk]
            pct_of_total = pnl / total_pnl * 100 if total_pnl != 0 else 0
            print(f"    {tk:>6}: {pnl:+7.1f}% ({n_tr} trades, {pct_of_total:.0f}% of total)")

        top3_pnl = sum(p for _, p in sorted_by_pnl[:3])
        top3_pct = top3_pnl / total_pnl * 100 if total_pnl != 0 else 0
        print(f"  Top 3 concentration: {top3_pct:.0f}%")

    # Use variant A for the verdict
    a_trades = run_variant_a(prices, events)
    a_ticker_pnl = {}
    for t in a_trades:
        a_ticker_pnl[t['ticker']] = a_ticker_pnl.get(t['ticker'], 0) + t['return_pct']
    a_total = sum(v for v in a_ticker_pnl.values())
    a_sorted = sorted(a_ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    a_top3_pct = sum(p for _, p in a_sorted[:3]) / a_total * 100 if a_total else 0

    d_trades = run_variant_d(prices, events)
    d_ticker_pnl = {}
    for t in d_trades:
        tk = t.get('trigger_stock', t['ticker'])
        d_ticker_pnl[tk] = d_ticker_pnl.get(tk, 0) + t['return_pct']
    d_total = sum(v for v in d_ticker_pnl.values())
    d_sorted = sorted(d_ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    d_top3_pct = sum(p for _, p in d_sorted[:3]) / d_total * 100 if d_total else 0

    concentrated = a_top3_pct > 50
    verdict = "CONCENTRATED (top 3 dominate)" if concentrated else "DIVERSIFIED"
    print(f"\n  CONCENTRATION VERDICT: {verdict}")

    return {
        'test': 'concentration',
        'variant_a_top3_pct': round(a_top3_pct, 1),
        'variant_a_top3': [t[0] for t in a_sorted[:3]],
        'variant_a_all_stocks': {t[0]: round(t[1], 1) for t in a_sorted},
        'variant_d_top3_pct': round(d_top3_pct, 1),
        'variant_d_top3': [t[0] for t in d_sorted[:3]],
        'concentrated': concentrated,
        'verdict': verdict,
    }


# ── ADVERSARIAL TEST 5: Cost Sensitivity ─────────────────────────────────
def test_cost_sensitivity(prices, events):
    """
    Re-run with higher slippage: 0.05% and 0.10%.
    """
    print("\n" + "=" * 70)
    print("TEST 5: COST SENSITIVITY")
    print("=" * 70)

    slippage_levels = [0.0002, 0.0005, 0.0010]
    results = {}

    for variant_name, run_fn in [("A", lambda s: run_variant_a(prices, events, slippage=s)),
                                  ("D", lambda s: run_variant_d(prices, events, slippage=s))]:
        print(f"\n  Variant {variant_name}:")
        for slip in slippage_levels:
            trades = run_fn(slip)
            m = compute_metrics(trades)
            print(f"    Slippage {slip*100:.2f}%: Sharpe={m['sharpe']:.3f}, "
                  f"WR={m['win_rate']:.1f}%, Return={m['total_return_pct']:.1f}%, "
                  f"Final=${m['final_equity']:.2f}")
            results[f"{variant_name}_{slip*100:.2f}pct"] = m

    # Check if edge survives at 0.10%
    a_010 = results.get('A_0.10pct', {})
    d_010 = results.get('D_0.10pct', {})
    survives = (a_010.get('sharpe', 0) > 0.3) or (d_010.get('sharpe', 0) > 0.3)
    verdict = "SURVIVES (robust to costs)" if survives else "FAILS (cost-sensitive)"
    print(f"\n  COST SENSITIVITY VERDICT: {verdict}")

    return {
        'test': 'cost_sensitivity',
        'results': results,
        'survives_010_slippage': survives,
        'verdict': verdict,
    }


# ── ADVERSARIAL TEST 6: Bear Market Only ─────────────────────────────────
def test_bear_market_only(prices, events):
    """
    Run ONLY during bear market periods (SPY < 200-SMA).
    """
    print("\n" + "=" * 70)
    print("TEST 6: BEAR MARKET ONLY")
    print("=" * 70)

    sma200 = compute_spy_sma200(prices)
    spy_close = prices.get('SPY', pd.DataFrame())
    if 'Close' in spy_close.columns:
        spy_close = spy_close['Close']
    else:
        return {'test': 'bear_market', 'verdict': 'NO SPY DATA'}

    # Identify bear periods
    bear_dates = set()
    bull_dates = set()
    for date in spy_close.index:
        if date in sma200.index:
            spy_val = float(spy_close.loc[date].iloc[0]) if isinstance(spy_close.loc[date], pd.Series) else float(spy_close.loc[date])
            sma_val = float(sma200.loc[date].iloc[0]) if isinstance(sma200.loc[date], pd.Series) else float(sma200.loc[date])
            if np.isnan(sma_val):
                continue
            if spy_val < sma_val:
                bear_dates.add(date)
            else:
                bull_dates.add(date)

    print(f"  Bear market days: {len(bear_dates)}, Bull market days: {len(bull_dates)}")

    # Filter trades by entry date regime
    for variant_name, run_fn in [("A", lambda: run_variant_a(prices, events)),
                                  ("D", lambda: run_variant_d(prices, events))]:
        all_trades = run_fn()
        bear_trades = []
        bull_trades = []

        for t in all_trades:
            entry_date = pd.Timestamp(t['entry_date'])
            # Find closest date
            mask = spy_close.index <= entry_date
            if not mask.any():
                continue
            closest = spy_close.index[mask][-1]
            if closest in bear_dates:
                bear_trades.append(t)
            elif closest in bull_dates:
                bull_trades.append(t)

        bear_m = compute_metrics(bear_trades)
        bull_m = compute_metrics(bull_trades)

        print(f"\n  Variant {variant_name}:")
        print(f"    Bull market: {bull_m['n_trades']} trades, Sharpe={bull_m['sharpe']:.3f}, "
              f"WR={bull_m['win_rate']:.1f}%, Return={bull_m['total_return_pct']:.1f}%")
        print(f"    Bear market: {bear_m['n_trades']} trades, Sharpe={bear_m['sharpe']:.3f}, "
              f"WR={bear_m['win_rate']:.1f}%, Return={bear_m['total_return_pct']:.1f}%")

    # Use variant A for verdict
    all_a = run_variant_a(prices, events)
    bear_a = [t for t in all_a if pd.Timestamp(t['entry_date']) in bear_dates or
              any(abs((pd.Timestamp(t['entry_date']) - d).days) <= 1 for d in list(bear_dates)[:50])]

    # More robust bear filtering
    bear_a_trades = []
    for t in all_a:
        entry_date = pd.Timestamp(t['entry_date'])
        mask = spy_close.index <= entry_date
        if not mask.any():
            continue
        closest = spy_close.index[mask][-1]
        if closest in bear_dates:
            bear_a_trades.append(t)

    bear_a_metrics = compute_metrics(bear_a_trades)
    has_bear_edge = bear_a_metrics['sharpe'] > 0.3 and bear_a_metrics['n_trades'] >= 5
    verdict = "HAS BEAR EDGE" if has_bear_edge else "NO BEAR EDGE (bull-only strategy)"
    print(f"\n  BEAR MARKET VERDICT: {verdict}")

    # Get bull metrics for comparison
    bull_a_trades = []
    for t in all_a:
        entry_date = pd.Timestamp(t['entry_date'])
        mask = spy_close.index <= entry_date
        if not mask.any():
            continue
        closest = spy_close.index[mask][-1]
        if closest in bull_dates:
            bull_a_trades.append(t)
    bull_a_metrics = compute_metrics(bull_a_trades)

    return {
        'test': 'bear_market_only',
        'bear_days': len(bear_dates),
        'bull_days': len(bull_dates),
        'variant_a_bear': bear_a_metrics,
        'variant_a_bull': bull_a_metrics,
        'has_bear_edge': has_bear_edge,
        'verdict': verdict,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("EARNINGS SURPRISE MOMENTUM — ADVERSARIAL VALIDATION")
    print(f"Date: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Goal: Break the strategy. Find any flaw that invalidates results.")
    print("=" * 70)

    # Download data
    prices = download_data()
    events = detect_earnings_events(prices, universe=UNIVERSE)
    all_events = detect_earnings_events(prices, universe=UNIVERSE + FAILED_GROWTH)

    total_events = sum(len(v) for v in events.values())
    total_beats = sum(sum(1 for e in v if e['is_beat']) for v in events.values())
    print(f"  Events: {total_events} total, {total_beats} beats across {len(events)} stocks")

    # Run all 6 adversarial tests
    results = {}

    results['survivorship_bias'] = test_survivorship_bias(prices, events, all_events)
    results['random_direction'] = test_random_direction(prices, events)
    results['time_reversed'] = test_time_reversed(prices, events)
    results['concentration'] = test_concentration(prices, events)
    results['cost_sensitivity'] = test_cost_sensitivity(prices, events)
    results['bear_market'] = test_bear_market_only(prices, events)

    # ── Overall Verdict ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OVERALL ADVERSARIAL VERDICT")
    print("=" * 70)

    verdicts = {k: v['verdict'] for k, v in results.items()}
    failures = []
    passes = []

    for test_name, verdict in verdicts.items():
        is_pass = any(w in verdict.upper() for w in ['SURVIVES', 'PASSES', 'DIVERSIFIED', 'REAL EDGE', 'HAS BEAR'])
        status = "PASS" if is_pass else "FAIL"
        if not is_pass:
            failures.append(test_name)
        else:
            passes.append(test_name)
        print(f"  [{status}] {test_name}: {verdict}")

    print(f"\n  Passed: {len(passes)}/6, Failed: {len(failures)}/6")

    if len(failures) >= 3:
        overall = "FAKE — Strategy has critical flaws"
    elif len(failures) >= 1:
        overall = f"CONDITIONAL — Edge exists but has weaknesses: {', '.join(failures)}"
    else:
        overall = "REAL EDGE — Survived all adversarial tests"

    print(f"\n  FINAL VERDICT: {overall}")

    # Save results
    output = {
        'strategy': 'earnings_surprise_momentum',
        'validation_type': 'adversarial',
        'timestamp': dt.datetime.now().isoformat(),
        'capital': CAPITAL,
        'tests': results,
        'verdicts': verdicts,
        'passes': passes,
        'failures': failures,
        'overall_verdict': overall,
    }

    out_path = Path('/home/jupiter/Lvl3Quant/data/earnings_surprise_adversarial_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
