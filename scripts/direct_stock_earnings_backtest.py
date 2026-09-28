#!/usr/bin/env python3
"""
Direct Stock Earnings Beat Backtest
====================================
Tests buying THE ACTUAL STOCK after earnings beats vs misses.
Prior work showed sector ETF approach was just beta — this tests individual stock alpha.

Beat proxy: gap up >3% on earnings day with volume >1.5x 20d avg
Miss proxy: gap down >3% on earnings day with volume >1.5x 20d avg

6 Variants:
A) Beat Stock Hold 40d
B) Beat Stock Short-Hold 10d
C) Beat Stock $10-$200 filter, Hold 40d
D) Beat vs Miss SPREAD (long beat + short miss), Hold 20d
E) Concentrated Beat (gap >5%), Hold 40d
F) ADVERSARIAL: Buy MISS stocks, Hold 40d (null hypothesis killer)

5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'PLTR', 'SOFI', 'HOOD', 'UBER', 'COIN', 'RBLX',
    'SNAP', 'ROKU', 'DDOG', 'SQ', 'SHOP', 'ABNB', 'NET', 'MELI'
]

# Sector mapping for spread variant
SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'GOOGL': 'Tech', 'AMZN': 'ConsDisc',
    'META': 'Tech', 'NVDA': 'Tech', 'TSLA': 'ConsDisc', 'AMD': 'Tech',
    'NFLX': 'ConsDisc', 'CRM': 'Tech', 'PLTR': 'Tech', 'SOFI': 'Fintech',
    'HOOD': 'Fintech', 'UBER': 'ConsDisc', 'COIN': 'Fintech', 'RBLX': 'ConsDisc',
    'SNAP': 'Tech', 'ROKU': 'ConsDisc', 'DDOG': 'Tech', 'SQ': 'Fintech',
    'SHOP': 'Tech', 'ABNB': 'ConsDisc', 'NET': 'Tech', 'MELI': 'ConsDisc'
}

INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0       # $0 on Robinhood
MAX_CONCURRENT = 3
OOT_START = '2022-01-01'
OOT_END = '2026-07-25'
DATA_START = '2021-06-01'  # Extra history for SMA/volume calcs
PERM_ITERATIONS = 1000
BEAT_GAP_THRESHOLD = 0.03   # 3%
MISS_GAP_THRESHOLD = -0.03  # -3%
STRONG_BEAT_GAP = 0.05      # 5% for concentrated variant
VOLUME_MULTIPLIER = 1.5     # Volume must be 1.5x 20d avg

# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data():
    """Download price data for universe + SPY."""
    tickers = UNIVERSE + ['SPY']
    print(f"Downloading data for {len(tickers)} tickers...")

    all_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=DATA_START, end=OOT_END, progress=False, auto_adjust=False)
            if len(df) > 100:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: SKIPPED (only {len(df)} days)")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    return all_data


# ============================================================
# DETECT EARNINGS EVENTS (GAP + VOLUME PROXY)
# ============================================================
def detect_earnings_events(all_data):
    """
    Detect earnings events using gap + volume proxy.
    Beat: gap up >3% with high volume
    Miss: gap down >3% with high volume
    """
    events = []

    for ticker in UNIVERSE:
        if ticker not in all_data:
            continue

        df = all_data[ticker].copy()
        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['vol_20d_avg'] = df['Volume'].rolling(20).mean().shift(1)
        df['vol_ratio'] = df['Volume'] / df['vol_20d_avg']

        # Filter to OOT period
        oot_mask = df.index >= pd.Timestamp(OOT_START)
        df_oot = df[oot_mask].copy()

        for idx, row in df_oot.iterrows():
            if pd.isna(row['gap_pct']) or pd.isna(row['vol_ratio']):
                continue

            # High volume filter
            if row['vol_ratio'] < VOLUME_MULTIPLIER:
                continue

            gap = row['gap_pct']

            if gap > BEAT_GAP_THRESHOLD:
                events.append({
                    'date': idx,
                    'ticker': ticker,
                    'type': 'beat',
                    'gap_pct': gap,
                    'vol_ratio': row['vol_ratio'],
                    'open_price': row['Open'],
                    'sector': SECTOR_MAP.get(ticker, 'Other')
                })
            elif gap < MISS_GAP_THRESHOLD:
                events.append({
                    'date': idx,
                    'ticker': ticker,
                    'type': 'miss',
                    'gap_pct': gap,
                    'vol_ratio': row['vol_ratio'],
                    'open_price': row['Open'],
                    'sector': SECTOR_MAP.get(ticker, 'Other')
                })

    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values('date').reset_index(drop=True)

    print(f"\nDetected {len(events_df)} earnings events:")
    if len(events_df) > 0:
        beats = (events_df['type'] == 'beat').sum()
        misses = (events_df['type'] == 'miss').sum()
        print(f"  Beats: {beats}, Misses: {misses}")
        print(f"  Tickers with events: {events_df['ticker'].nunique()}")

    return events_df


# ============================================================
# REGIME DETECTION
# ============================================================
def compute_spy_regime(spy_data):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    spy = spy_data.copy()
    spy['sma200'] = spy['Close'].rolling(200).mean()
    spy['regime'] = np.where(spy['Close'] > spy['sma200'], 'bull', 'bear')
    return spy[['regime']].dropna()


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(events_df, all_data, spy_regime, variant_name,
                 hold_days=40, event_type='beat', gap_threshold=0.03,
                 price_filter=None, is_spread=False):
    """
    Run a single variant backtest.

    For spread variant: long beat + short miss from same sector, simultaneous.
    For others: buy stock, hold N days, sell.
    """
    if len(events_df) == 0:
        return empty_result(variant_name)

    # Filter events by type
    if is_spread:
        beats = events_df[events_df['type'] == 'beat'].copy()
        misses = events_df[events_df['type'] == 'miss'].copy()
    else:
        if event_type == 'beat':
            filtered = events_df[events_df['gap_pct'] > gap_threshold].copy()
        else:  # miss (adversarial)
            filtered = events_df[events_df['gap_pct'] < -gap_threshold].copy()

        # Price filter for variant C
        if price_filter:
            lo, hi = price_filter
            filtered = filtered[(filtered['open_price'] >= lo) & (filtered['open_price'] <= hi)]

    equity = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    active_positions = []  # list of dicts with entry info

    # Get all trading dates
    spy_dates = sorted(all_data['SPY'].index)
    spy_dates = [d for d in spy_dates if d >= pd.Timestamp(OOT_START)]

    if is_spread:
        return run_spread_backtest(beats, misses, all_data, spy_regime,
                                   variant_name, hold_days)

    for date in spy_dates:
        # Check for exits
        new_active = []
        for pos in active_positions:
            if date >= pos['exit_date']:
                # Exit position
                ticker = pos['ticker']
                if ticker in all_data and date in all_data[ticker].index:
                    exit_price = float(all_data[ticker].loc[date, 'Open'])
                else:
                    # Find next available date
                    ticker_dates = all_data[ticker].index
                    future = ticker_dates[ticker_dates >= date]
                    if len(future) > 0:
                        exit_price = float(all_data[ticker].loc[future[0], 'Open'])
                    else:
                        exit_price = pos['entry_price']  # fallback

                exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price_adj - pos['entry_price_adj']) * pos['shares']
                equity += pos['position_value'] + pnl

                regime_at_entry = 'unknown'
                if pos['entry_date'] in spy_regime.index:
                    regime_at_entry = spy_regime.loc[pos['entry_date'], 'regime']

                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'ticker': pos['ticker'],
                    'entry_price': pos['entry_price_adj'],
                    'exit_price': exit_price_adj,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'ret': pnl / pos['position_value'],
                    'regime': regime_at_entry,
                    'gap_pct': pos['gap_pct']
                })
            else:
                new_active.append(pos)
        active_positions = new_active

        # Check for entries
        if not is_spread:
            candidates = filtered[filtered['date'] == date]
            for _, event in candidates.iterrows():
                if len(active_positions) >= MAX_CONCURRENT:
                    break

                ticker = event['ticker']
                # Don't double up on same ticker
                if any(p['ticker'] == ticker for p in active_positions):
                    continue

                entry_price = float(event['open_price'])
                entry_price_adj = entry_price * (1 + SLIPPAGE_PCT)

                # Position sizing: equal weight, but can't exceed available capital
                available = equity - sum(p['position_value'] for p in active_positions)
                position_value = min(available / (MAX_CONCURRENT - len(active_positions)),
                                     available)

                if position_value < 10:  # Skip tiny positions
                    continue

                shares = int(position_value / entry_price_adj)
                if shares < 1:
                    # Fractional shares on Robinhood
                    shares = position_value / entry_price_adj

                actual_value = shares * entry_price_adj
                equity -= actual_value

                # Calculate exit date (hold_days trading days forward)
                ticker_dates = all_data[ticker].index
                future = ticker_dates[ticker_dates > date]
                if len(future) >= hold_days:
                    exit_date = future[hold_days - 1]
                elif len(future) > 0:
                    exit_date = future[-1]
                else:
                    equity += actual_value  # Can't trade, refund
                    continue

                active_positions.append({
                    'ticker': ticker,
                    'entry_date': date,
                    'exit_date': exit_date,
                    'entry_price': entry_price,
                    'entry_price_adj': entry_price_adj,
                    'shares': shares,
                    'position_value': actual_value,
                    'gap_pct': event['gap_pct']
                })

        # Mark-to-market for equity curve
        mtm = equity
        for pos in active_positions:
            ticker = pos['ticker']
            if ticker in all_data and date in all_data[ticker].index:
                current_price = float(all_data[ticker].loc[date, 'Close'])
                mtm += current_price * pos['shares']
            else:
                mtm += pos['position_value']

        equity_curve.append({'date': date, 'equity': mtm})

    # Close any remaining positions at last date
    for pos in active_positions:
        ticker = pos['ticker']
        last_date = spy_dates[-1]
        if ticker in all_data and last_date in all_data[ticker].index:
            exit_price = float(all_data[ticker].loc[last_date, 'Close']) * (1 - SLIPPAGE_PCT)
        else:
            exit_price = pos['entry_price_adj']

        pnl = (exit_price - pos['entry_price_adj']) * pos['shares']

        regime_at_entry = 'unknown'
        if pos['entry_date'] in spy_regime.index:
            regime_at_entry = spy_regime.loc[pos['entry_date'], 'regime']

        trades.append({
            'entry_date': pos['entry_date'],
            'exit_date': last_date,
            'ticker': pos['ticker'],
            'entry_price': pos['entry_price_adj'],
            'exit_price': exit_price,
            'shares': pos['shares'],
            'pnl': pnl,
            'ret': pnl / pos['position_value'],
            'regime': regime_at_entry,
            'gap_pct': pos['gap_pct']
        })

    return compute_metrics(trades, equity_curve, variant_name, events_df,
                           all_data, spy_regime, event_type, gap_threshold)


def run_spread_backtest(beats, misses, all_data, spy_regime, variant_name, hold_days=20):
    """
    Spread variant: Long beat + Short miss from same sector.
    Simulate short via buying ATM puts (simplified as inverse return with 1.5x cost for premium decay).
    """
    equity = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    active_positions = []

    spy_dates = sorted(all_data['SPY'].index)
    spy_dates = [d for d in spy_dates if d >= pd.Timestamp(OOT_START)]

    # Build pairs: match beats and misses within ±5 trading days, same sector
    pairs = []
    used_misses = set()

    for _, beat in beats.iterrows():
        sector = beat['sector']
        nearby_misses = misses[
            (misses['sector'] == sector) &
            (abs((misses['date'] - beat['date']).dt.days) <= 10) &
            (~misses.index.isin(used_misses))
        ]
        if len(nearby_misses) > 0:
            miss = nearby_misses.iloc[0]
            pairs.append({'beat': beat, 'miss': miss, 'date': max(beat['date'], miss['date'])})
            used_misses.add(miss.name)

    pairs.sort(key=lambda x: x['date'])

    for date in spy_dates:
        # Check exits
        new_active = []
        for pos in active_positions:
            if date >= pos['exit_date']:
                # Exit long leg
                long_ticker = pos['long_ticker']
                if long_ticker in all_data and date in all_data[long_ticker].index:
                    long_exit = float(all_data[long_ticker].loc[date, 'Open']) * (1 - SLIPPAGE_PCT)
                else:
                    long_exit = pos['long_entry']

                # Exit short leg (put position)
                short_ticker = pos['short_ticker']
                if short_ticker in all_data and date in all_data[short_ticker].index:
                    short_exit = float(all_data[short_ticker].loc[date, 'Open'])
                else:
                    short_exit = pos['short_entry']

                # Long leg PnL
                long_pnl = (long_exit - pos['long_entry']) * pos['long_shares']
                # Short leg PnL (via puts: profit when stock drops, cost = premium)
                short_ret = (pos['short_entry'] - short_exit) / pos['short_entry']
                # Put premium cost ~3% of notional, theta decay ~0.5%/day
                premium_cost = pos['short_value'] * 0.03
                theta_decay = pos['short_value'] * 0.005 * hold_days
                short_pnl = short_ret * pos['short_value'] - premium_cost - theta_decay

                total_pnl = long_pnl + short_pnl
                equity += pos['total_value'] + total_pnl

                regime_at_entry = 'unknown'
                if pos['entry_date'] in spy_regime.index:
                    regime_at_entry = spy_regime.loc[pos['entry_date'], 'regime']

                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'ticker': f"{pos['long_ticker']}/{pos['short_ticker']}",
                    'entry_price': pos['long_entry'],
                    'exit_price': long_exit,
                    'shares': pos['long_shares'],
                    'pnl': total_pnl,
                    'ret': total_pnl / pos['total_value'],
                    'regime': regime_at_entry,
                    'gap_pct': pos['gap_pct']
                })
            else:
                new_active.append(pos)
        active_positions = new_active

        # Check entries
        for pair in pairs:
            if pair['date'] != date:
                continue
            if len(active_positions) >= MAX_CONCURRENT:
                break

            beat_event = pair['beat']
            miss_event = pair['miss']

            long_ticker = beat_event['ticker']
            short_ticker = miss_event['ticker']

            available = equity - sum(p['total_value'] for p in active_positions)
            half_value = available / (2 * (MAX_CONCURRENT - len(active_positions)))

            if half_value < 10:
                continue

            long_entry = float(beat_event['open_price']) * (1 + SLIPPAGE_PCT)
            short_entry = float(miss_event['open_price'])

            long_shares = half_value / long_entry

            # Exit date
            ticker_dates = all_data[long_ticker].index
            future = ticker_dates[ticker_dates > date]
            if len(future) >= hold_days:
                exit_date = future[hold_days - 1]
            elif len(future) > 0:
                exit_date = future[-1]
            else:
                continue

            total_value = half_value * 2
            equity -= total_value

            active_positions.append({
                'long_ticker': long_ticker,
                'short_ticker': short_ticker,
                'entry_date': date,
                'exit_date': exit_date,
                'long_entry': long_entry,
                'short_entry': short_entry,
                'long_shares': long_shares,
                'long_value': half_value,
                'short_value': half_value,
                'total_value': total_value,
                'gap_pct': beat_event['gap_pct']
            })

        # MTM
        mtm = equity
        for pos in active_positions:
            lt = pos['long_ticker']
            st = pos['short_ticker']
            if lt in all_data and date in all_data[lt].index:
                long_val = float(all_data[lt].loc[date, 'Close']) * pos['long_shares']
            else:
                long_val = pos['long_value']

            if st in all_data and date in all_data[st].index:
                short_current = float(all_data[st].loc[date, 'Close'])
                short_ret = (pos['short_entry'] - short_current) / pos['short_entry']
                short_val = pos['short_value'] * (1 + short_ret)
            else:
                short_val = pos['short_value']

            mtm += long_val + short_val

        equity_curve.append({'date': date, 'equity': mtm})

    # Close remaining
    for pos in active_positions:
        last_date = spy_dates[-1]
        lt = pos['long_ticker']
        if lt in all_data and last_date in all_data[lt].index:
            long_exit = float(all_data[lt].loc[last_date, 'Close']) * (1 - SLIPPAGE_PCT)
        else:
            long_exit = pos['long_entry']

        st = pos['short_ticker']
        if st in all_data and last_date in all_data[st].index:
            short_exit = float(all_data[st].loc[last_date, 'Close'])
        else:
            short_exit = pos['short_entry']

        long_pnl = (long_exit - pos['long_entry']) * pos['long_shares']
        short_ret = (pos['short_entry'] - short_exit) / pos['short_entry']
        premium_cost = pos['short_value'] * 0.03
        theta_decay = pos['short_value'] * 0.005 * 20
        short_pnl = short_ret * pos['short_value'] - premium_cost - theta_decay

        total_pnl = long_pnl + short_pnl

        regime_at_entry = 'unknown'
        if pos['entry_date'] in spy_regime.index:
            regime_at_entry = spy_regime.loc[pos['entry_date'], 'regime']

        trades.append({
            'entry_date': pos['entry_date'],
            'exit_date': last_date,
            'ticker': f"{pos['long_ticker']}/{pos['short_ticker']}",
            'entry_price': pos['long_entry'],
            'exit_price': long_exit,
            'shares': pos['long_shares'],
            'pnl': total_pnl,
            'ret': total_pnl / pos['total_value'],
            'regime': regime_at_entry,
            'gap_pct': pos['gap_pct']
        })

    events_combined = pd.concat([beats, misses])
    return compute_metrics(trades, equity_curve, variant_name, events_combined,
                           all_data, spy_regime, 'spread', BEAT_GAP_THRESHOLD)


# ============================================================
# METRICS
# ============================================================
def compute_metrics(trades, equity_curve, variant_name, events_df,
                    all_data, spy_regime, event_type, gap_threshold):
    """Compute all metrics + permutation test."""

    if len(trades) == 0:
        return empty_result(variant_name)

    trades_df = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_curve)

    # Basic metrics
    n_trades = len(trades_df)
    winners = (trades_df['pnl'] > 0).sum()
    losers = (trades_df['pnl'] <= 0).sum()
    wr = winners / n_trades if n_trades > 0 else 0

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    final_equity = eq_df['equity'].iloc[-1] if len(eq_df) > 0 else INITIAL_CAPITAL
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Daily returns from equity curve
    eq_df['daily_ret'] = eq_df['equity'].pct_change()
    daily_rets = eq_df['daily_ret'].dropna()

    if len(daily_rets) > 10:
        ann_factor = np.sqrt(252)
        sharpe = (daily_rets.mean() / daily_rets.std()) * ann_factor if daily_rets.std() > 0 else 0

        downside = daily_rets[daily_rets < 0]
        downside_std = downside.std() if len(downside) > 0 else daily_rets.std()
        sortino = (daily_rets.mean() / downside_std) * ann_factor if downside_std > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    # Max drawdown
    eq_df['cum_max'] = eq_df['equity'].cummax()
    eq_df['drawdown'] = (eq_df['equity'] - eq_df['cum_max']) / eq_df['cum_max']
    max_dd = eq_df['drawdown'].min()

    # Regime analysis
    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']

    def regime_sharpe(regime_trades):
        if len(regime_trades) < 3:
            return 0.0
        rets = regime_trades['ret']
        if rets.std() == 0:
            return 0.0
        return float((rets.mean() / rets.std()) * np.sqrt(252 / 40))  # Adjust for hold period

    sharpe_bull = regime_sharpe(bull_trades)
    sharpe_bear = regime_sharpe(bear_trades)

    max_regime = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_regime if max_regime > 0 else 0

    # Permutation test: shuffle beat/miss labels, recompute mean return
    observed_mean_ret = trades_df['ret'].mean()
    perm_p = run_permutation_test(events_df, all_data, spy_regime,
                                   observed_mean_ret, event_type, gap_threshold,
                                   hold_days=40)

    # 5-gate validation
    gates = {
        'sharpe_gt_0.5': sharpe > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'max_dd_gt_neg50': max_dd > -0.50,
        'min_20_trades': n_trades >= 20
    }
    gates_passed = sum(gates.values())
    all_gates_pass = all(gates.values())

    result = {
        'variant': variant_name,
        'total_trades': n_trades,
        'winners': int(winners),
        'losers': int(losers),
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 3),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown': round(max_dd, 4),
        'total_return': round(total_return, 4),
        'final_equity': round(final_equity, 2),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
        'perm_p_value': round(perm_p, 4),
        'gates': gates,
        'gates_passed': f"{gates_passed}/5",
        'all_gates_pass': all_gates_pass,
        'avg_trade_return': round(trades_df['ret'].mean(), 4),
        'avg_trade_pnl': round(trades_df['pnl'].mean(), 2),
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
    }

    return result


def run_permutation_test(events_df, all_data, spy_regime, observed_mean_ret,
                          event_type, gap_threshold, hold_days=40):
    """
    Permutation test: shuffle beat/miss labels among events, recompute mean return.
    Tests whether BEAT stocks specifically outperform, controlling for earnings timing.
    """
    if len(events_df) == 0 or 'type' not in events_df.columns:
        return 1.0

    # Pre-compute forward returns for all events
    fwd_returns = {}
    for _, event in events_df.iterrows():
        ticker = event['ticker']
        date = event['date']
        if ticker not in all_data:
            continue

        df = all_data[ticker]
        future = df.index[df.index > date]

        if len(future) >= hold_days:
            entry_price = float(event['open_price']) * (1 + SLIPPAGE_PCT)
            exit_price = float(df.loc[future[hold_days - 1], 'Open']) * (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price
            fwd_returns[(ticker, date)] = ret

    if len(fwd_returns) == 0:
        return 1.0

    # Build arrays for permutation
    keys = list(fwd_returns.keys())
    returns = np.array([fwd_returns[k] for k in keys])

    # Original labels
    original_labels = []
    for k in keys:
        ticker, date = k
        match = events_df[(events_df['ticker'] == ticker) & (events_df['date'] == date)]
        if len(match) > 0:
            original_labels.append(match.iloc[0]['type'])
        else:
            original_labels.append('unknown')

    original_labels = np.array(original_labels)

    # Observed: mean return of selected type
    if event_type == 'beat':
        mask = original_labels == 'beat'
    elif event_type == 'miss':
        mask = original_labels == 'miss'
    else:
        # For spread, test beat - miss difference
        beat_mask = original_labels == 'beat'
        miss_mask = original_labels == 'miss'
        if beat_mask.sum() > 0 and miss_mask.sum() > 0:
            observed = returns[beat_mask].mean() - returns[miss_mask].mean()
        else:
            return 1.0

        count_exceed = 0
        for _ in range(PERM_ITERATIONS):
            shuffled = np.random.permutation(original_labels)
            b_mask = shuffled == 'beat'
            m_mask = shuffled == 'miss'
            if b_mask.sum() > 0 and m_mask.sum() > 0:
                perm_stat = returns[b_mask].mean() - returns[m_mask].mean()
                if perm_stat >= observed:
                    count_exceed += 1
        return (count_exceed + 1) / (PERM_ITERATIONS + 1)

    if mask.sum() == 0:
        return 1.0

    observed = returns[mask].mean()

    # Permutation: shuffle labels, recompute
    count_exceed = 0
    n_selected = mask.sum()

    for _ in range(PERM_ITERATIONS):
        # Randomly select same number of events
        perm_idx = np.random.choice(len(returns), size=n_selected, replace=False)
        perm_mean = returns[perm_idx].mean()
        if perm_mean >= observed:
            count_exceed += 1

    return (count_exceed + 1) / (PERM_ITERATIONS + 1)


def empty_result(variant_name):
    return {
        'variant': variant_name,
        'total_trades': 0,
        'error': 'No trades generated',
        'all_gates_pass': False,
        'gates_passed': '0/5'
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("DIRECT STOCK EARNINGS BEAT BACKTEST")
    print("Testing individual stock alpha from earnings beats vs misses")
    print("=" * 70)

    # Download data
    all_data = download_data()
    if 'SPY' not in all_data:
        print("ERROR: Could not download SPY data")
        sys.exit(1)

    # Compute regime
    spy_regime = compute_spy_regime(all_data['SPY'])

    # Detect earnings events
    events_df = detect_earnings_events(all_data)
    if len(events_df) == 0:
        print("ERROR: No earnings events detected")
        sys.exit(1)

    # Print event summary
    print("\nSample events:")
    for _, e in events_df.head(10).iterrows():
        print(f"  {e['date'].strftime('%Y-%m-%d')} {e['ticker']:5s} {e['type']:4s} gap={e['gap_pct']:+.1%} vol_ratio={e['vol_ratio']:.1f}x")

    # Run all 6 variants
    results = {}

    print("\n" + "=" * 70)
    print("VARIANT A: Beat Stock Hold 40d")
    print("=" * 70)
    results['A'] = run_backtest(events_df, all_data, spy_regime,
                                 'A_Beat_Hold_40d', hold_days=40,
                                 event_type='beat', gap_threshold=BEAT_GAP_THRESHOLD)
    print_result(results['A'])

    print("\n" + "=" * 70)
    print("VARIANT B: Beat Stock Short-Hold 10d")
    print("=" * 70)
    results['B'] = run_backtest(events_df, all_data, spy_regime,
                                 'B_Beat_Hold_10d', hold_days=10,
                                 event_type='beat', gap_threshold=BEAT_GAP_THRESHOLD)
    print_result(results['B'])

    print("\n" + "=" * 70)
    print("VARIANT C: Beat Stock $10-$200 Filter, Hold 40d")
    print("=" * 70)
    results['C'] = run_backtest(events_df, all_data, spy_regime,
                                 'C_Beat_PriceFilter_40d', hold_days=40,
                                 event_type='beat', gap_threshold=BEAT_GAP_THRESHOLD,
                                 price_filter=(10, 200))
    print_result(results['C'])

    print("\n" + "=" * 70)
    print("VARIANT D: Beat vs Miss SPREAD (Long Beat + Short Miss), Hold 20d")
    print("=" * 70)
    results['D'] = run_backtest(events_df, all_data, spy_regime,
                                 'D_Spread_20d', hold_days=20,
                                 is_spread=True)
    print_result(results['D'])

    print("\n" + "=" * 70)
    print("VARIANT E: Concentrated Beat (gap >5%), Hold 40d")
    print("=" * 70)
    results['E'] = run_backtest(events_df, all_data, spy_regime,
                                 'E_Concentrated_Beat_40d', hold_days=40,
                                 event_type='beat', gap_threshold=STRONG_BEAT_GAP)
    print_result(results['E'])

    print("\n" + "=" * 70)
    print("VARIANT F: ADVERSARIAL — Buy MISS Stocks, Hold 40d")
    print("=" * 70)
    results['F'] = run_backtest(events_df, all_data, spy_regime,
                                 'F_Adversarial_Miss_40d', hold_days=40,
                                 event_type='miss', gap_threshold=BEAT_GAP_THRESHOLD)
    print_result(results['F'])

    # ============================================================
    # HEAD-TO-HEAD: A (Beats) vs F (Misses)
    # ============================================================
    print("\n" + "=" * 70)
    print("CRITICAL: HEAD-TO-HEAD COMPARISON — A (Beats) vs F (Misses)")
    print("=" * 70)

    a = results['A']
    f = results['F']

    if a.get('total_trades', 0) > 0 and f.get('total_trades', 0) > 0:
        sharpe_spread = a.get('sharpe', 0) - f.get('sharpe', 0)
        wr_spread = a.get('win_rate', 0) - f.get('win_rate', 0)
        return_spread = a.get('total_return', 0) - f.get('total_return', 0)

        print(f"  A (Beats)  Sharpe: {a.get('sharpe', 0):.3f}  WR: {a.get('win_rate', 0):.1%}  Return: {a.get('total_return', 0):.1%}")
        print(f"  F (Misses) Sharpe: {f.get('sharpe', 0):.3f}  WR: {f.get('win_rate', 0):.1%}  Return: {f.get('total_return', 0):.1%}")
        print(f"  SPREAD     Sharpe: {sharpe_spread:+.3f}  WR: {wr_spread:+.1%}  Return: {return_spread:+.1%}")

        if f.get('sharpe', 0) > 0.5:
            print("\n  *** ADVERSARIAL KILL: Miss stocks also profitable (Sharpe > 0.5)")
            print("  *** The earnings beat signal is likely BETA, not ALPHA")
            print("  *** Individual stock approach suffers same flaw as sector ETF approach")
            verdict = "KILLED_BY_ADVERSARIAL"
        elif sharpe_spread > 0.3 and a.get('sharpe', 0) > 0.5:
            print("\n  *** SIGNAL CONFIRMED: Beats meaningfully outperform Misses")
            print(f"  *** Sharpe spread of {sharpe_spread:+.3f} suggests genuine beat alpha")
            verdict = "SIGNAL_CONFIRMED"
        else:
            print("\n  *** INCONCLUSIVE: Spread too small or base Sharpe too low")
            verdict = "INCONCLUSIVE"

        head_to_head = {
            'sharpe_spread': round(sharpe_spread, 3),
            'wr_spread': round(wr_spread, 4),
            'return_spread': round(return_spread, 4),
            'verdict': verdict,
            'adversarial_sharpe': f.get('sharpe', 0),
            'beat_sharpe': a.get('sharpe', 0)
        }
    else:
        head_to_head = {'error': 'Insufficient trades for comparison'}
        verdict = "INSUFFICIENT_DATA"

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Return':>8} {'Perm-p':>7} {'Gates':>6}")
    print("-" * 100)
    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = results[key]
        if r.get('total_trades', 0) > 0:
            print(f"{r['variant']:<30} {r['total_trades']:>6} {r.get('sharpe',0):>7.3f} {r.get('sortino',0):>8.3f} {r.get('profit_factor',0):>6.2f} {r.get('win_rate',0):>5.1%} {r.get('max_drawdown',0):>7.1%} {r.get('total_return',0):>7.1%} {r.get('perm_p_value',1):>7.4f} {r.get('gates_passed','0/5'):>6}")
        else:
            print(f"{r['variant']:<30} {'NO TRADES':>6}")

    # Save results
    output = {
        'metadata': {
            'run_date': datetime.now().isoformat(),
            'oot_period': f"{OOT_START} to {OOT_END}",
            'universe_size': len(UNIVERSE),
            'initial_capital': INITIAL_CAPITAL,
            'max_concurrent': MAX_CONCURRENT,
            'beat_gap_threshold': BEAT_GAP_THRESHOLD,
            'miss_gap_threshold': MISS_GAP_THRESHOLD,
            'slippage_pct': SLIPPAGE_PCT,
            'permutation_iterations': PERM_ITERATIONS,
            'total_events_detected': len(events_df),
            'beats_detected': int((events_df['type'] == 'beat').sum()),
            'misses_detected': int((events_df['type'] == 'miss').sum()),
        },
        'variants': {k: v for k, v in results.items()},
        'head_to_head_A_vs_F': head_to_head,
        'verdict': verdict,
        'five_gate_summary': {
            k: {
                'pass': results[k].get('all_gates_pass', False),
                'gates': results[k].get('gates_passed', '0/5')
            } for k in results
        }
    }

    # Convert any numpy/pandas types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    output_path = '/home/jupiter/Lvl3Quant/data/direct_stock_earnings_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\nResults saved to {output_path}")
    print(f"\nFINAL VERDICT: {verdict}")

    return output


def print_result(result):
    if result.get('total_trades', 0) == 0:
        print("  NO TRADES GENERATED")
        return

    print(f"  Trades: {result['total_trades']} (W:{result['winners']} L:{result['losers']})")
    print(f"  Win Rate: {result['win_rate']:.1%}")
    print(f"  Sharpe: {result['sharpe']:.3f}  Sortino: {result['sortino']:.3f}  PF: {result['profit_factor']:.2f}")
    print(f"  Max DD: {result['max_drawdown']:.1%}")
    print(f"  Total Return: {result['total_return']:.1%}  Final Equity: ${result['final_equity']:.2f}")
    print(f"  Regime: Bull Sharpe={result['sharpe_bull']:.3f} ({result['bull_trades']} trades) | Bear Sharpe={result['sharpe_bear']:.3f} ({result['bear_trades']} trades) | Gap={result['regime_gap']:.3f}")
    print(f"  Permutation p-value: {result['perm_p_value']:.4f}")
    print(f"  Gates: {result['gates_passed']} {'PASS' if result['all_gates_pass'] else 'FAIL'}")
    for gate, passed in result['gates'].items():
        status = 'PASS' if passed else 'FAIL'
        print(f"    {gate}: {status}")


if __name__ == '__main__':
    np.random.seed(42)
    main()
