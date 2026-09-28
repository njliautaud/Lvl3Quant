#!/usr/bin/env python3
"""
PEAD Variant A — Adversarial Validation (5 Tests)
==================================================
Tests whether the PEAD strategy's edge is real or decorative.

1. INVERSE DIRECTION: Buy gap-DOWN stocks instead of gap-UP
2. RANDOM TIMING: Randomize entry dates, keep same stocks (100 iterations)
3. SUB-PERIOD STABILITY: Split OOT into 3 equal sub-periods, all must be Sharpe > 0
4. TOP-TRADE REMOVAL: Remove best 5% of trades, Sharpe must stay > 0.5
5. TICKER CONCENTRATION: Remove top 3 tickers by P&L, Sharpe must stay > 0.5
"""

import json
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Configuration (must match original backtest) ─────────────────────────
CAPITAL = 645.0
MAX_POSITIONS = 5
SLIPPAGE_PCT = 0.0002
HOLD_DAYS = 20

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')

UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'BRK-B',
    'LLY', 'AVGO', 'JPM', 'XOM', 'UNH', 'V', 'MA', 'COST', 'PG', 'JNJ',
    'HD', 'WMT', 'NFLX', 'CRM', 'ABBV', 'BAC', 'ORCL', 'CVX', 'MRK',
    'KO', 'PEP', 'AMD', 'ACN', 'ADBE', 'TMO', 'CSCO', 'LIN', 'MCD',
    'ABT', 'WFC', 'GE', 'DHR', 'PM', 'QCOM', 'TXN', 'ISRG', 'INTU',
    'AMGN', 'CAT', 'AMAT', 'BX', 'NOW',
]

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULTS_PATH = DATA_DIR / 'pead_adversarial_results.json'
ORIGINAL_RESULTS_PATH = DATA_DIR / 'pead_quality_results.json'

RANDOM_TIMING_ITERATIONS = 100


# ── Data Download ────────────────────────────────────────────────────────
def download_data():
    print("Downloading price data...")
    all_tickers = list(set(UNIVERSE + ['SPY']))
    prices = {}
    for ticker in all_tickers:
        try:
            df = yf.download(
                ticker, start='2021-01-01',
                end=(OOT_END + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                prices[ticker] = df
        except Exception as e:
            print(f"  Failed {ticker}: {e}")
    print(f"  Downloaded {len(prices)} tickers")
    return prices


# ── Detect Earnings Events (same logic as original) ─────────────────────
def detect_earnings_events(prices, gap_threshold=0.02):
    events = []
    spy = prices.get('SPY')

    for ticker in UNIVERSE:
        if ticker not in prices or ticker == 'SPY':
            continue
        df = prices[ticker].copy()
        if len(df) < 60:
            continue

        df['prev_close'] = df['Close'].shift(1)
        df['gap_pct'] = (df['Open'] - df['prev_close']) / df['prev_close']
        df['vol_20d'] = df['Close'].pct_change().rolling(20).std()
        df['sma_200'] = df['Close'].rolling(200).mean()
        df['close_t_plus_1'] = df['Close'].shift(-1)
        df['reaction_2d'] = (df['close_t_plus_1'] / df['prev_close']) - 1

        for i in range(21, len(df) - 1):
            row = df.iloc[i]
            date = df.index[i]
            gap = row['gap_pct']
            if pd.isna(gap) or pd.isna(row['vol_20d']):
                continue
            abs_gap = abs(gap)
            if abs_gap < gap_threshold:
                continue
            if abs_gap < 2.0 * row['vol_20d']:
                continue

            if spy is not None and date in spy.index:
                spy_ret = abs(spy.loc[date, 'Close'] / spy['Close'].shift(1).loc[date] - 1)
                if pd.notna(spy_ret) and spy_ret > 0.02:
                    continue

            direction = 'up' if gap > 0 else 'down'
            reaction = row['reaction_2d'] if pd.notna(row['reaction_2d']) else gap

            events.append({
                'ticker': ticker,
                'date': date,
                'gap_pct': gap,
                'reaction_2d': reaction,
                'direction': direction,
                'above_sma200': row['Close'] > row['sma_200'] if pd.notna(row['sma_200']) else False,
                'vol_20d': row['vol_20d'],
            })

    events.sort(key=lambda x: (x['ticker'], x['date']))
    filtered = []
    last_event = {}
    for ev in events:
        tk = ev['ticker']
        if tk in last_event:
            if (ev['date'] - last_event[tk]).days < 60:
                continue
        last_event[tk] = ev['date']
        filtered.append(ev)

    return filtered


# ── Backtest Engine (reusable for both directions) ───────────────────────
def backtest_trades(events, prices, direction_filter='up'):
    """
    Run backtest on events filtered by direction.
    direction_filter: 'up' for gap-up longs, 'down' for gap-down longs.
    """
    valid_events = []
    for ev in events:
        if ev['date'] < OOT_START:
            continue
        if abs(ev['gap_pct']) < 0.02:
            continue
        if ev['direction'] == direction_filter:
            valid_events.append({**ev, 'side': 'long'})

    if not valid_events:
        return None, []

    valid_events.sort(key=lambda x: x['date'])

    spy = prices.get('SPY')
    if spy is None:
        return None, []
    trading_days = spy.loc[OOT_START:OOT_END].index

    event_by_date = {}
    for ev in valid_events:
        if ev['date'] not in event_by_date:
            event_by_date[ev['date']] = []
        event_by_date[ev['date']].append(ev)

    trades = []
    equity = CAPITAL
    equity_curve = [(OOT_START, CAPITAL)]
    open_positions = []

    for day in trading_days:
        still_open = []
        for pos in open_positions:
            days_held = len(spy.loc[pos['entry_date']:day].index) - 1
            if days_held >= HOLD_DAYS:
                ticker = pos['ticker']
                if ticker in prices and day in prices[ticker].index:
                    exit_price = prices[ticker].loc[day, 'Close']
                    slippage = exit_price * SLIPPAGE_PCT
                    exit_price -= slippage
                    pnl_pct = (exit_price / pos['entry_price']) - 1
                    position_value = pos['alloc']
                    pnl_dollar = position_value * pnl_pct
                    equity += pnl_dollar
                    trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': day,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'gap_pct': pos['gap_pct'],
                        'return_pct': pnl_pct * 100,
                        'pnl_dollar': pnl_dollar,
                        'hold_days': days_held,
                    })
                else:
                    still_open.append(pos)
                    continue
            else:
                still_open.append(pos)
        open_positions = still_open

        if day in event_by_date:
            n_open = len(open_positions)
            for ev in event_by_date[day]:
                if n_open >= MAX_POSITIONS:
                    break
                ticker = ev['ticker']
                if any(p['ticker'] == ticker for p in open_positions):
                    continue
                if ticker not in prices or day not in prices[ticker].index:
                    continue
                day_idx = list(prices[ticker].index).index(day)
                if day_idx + 1 >= len(prices[ticker]):
                    continue
                entry_date = prices[ticker].index[day_idx + 1]
                entry_price = prices[ticker].iloc[day_idx + 1]['Open']
                slippage = entry_price * SLIPPAGE_PCT
                entry_price += slippage
                alloc = equity / MAX_POSITIONS
                open_positions.append({
                    'ticker': ticker,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'gap_pct': ev['gap_pct'],
                    'alloc': alloc,
                })
                n_open += 1

        equity_curve.append((day, equity))

    # Close remaining
    for pos in open_positions:
        ticker = pos['ticker']
        if ticker in prices:
            last_price = prices[ticker].iloc[-1]['Close']
            slippage = last_price * SLIPPAGE_PCT
            last_price -= slippage
            pnl_pct = (last_price / pos['entry_price']) - 1
            pnl_dollar = pos['alloc'] * pnl_pct
            equity += pnl_dollar
            trades.append({
                'ticker': ticker,
                'entry_date': pos['entry_date'],
                'exit_date': pd.Timestamp('now'),
                'entry_price': pos['entry_price'],
                'exit_price': last_price,
                'gap_pct': pos['gap_pct'],
                'return_pct': pnl_pct * 100,
                'pnl_dollar': pnl_dollar,
                'hold_days': -1,
            })

    return equity, trades


# ── Compute Sharpe from trade returns ────────────────────────────────────
def compute_sharpe(returns_pct, avg_hold_days=20.0):
    """Annualized Sharpe from array of per-trade return percentages."""
    if len(returns_pct) < 2:
        return 0.0
    avg = np.mean(returns_pct)
    std = np.std(returns_pct)
    if std == 0:
        return 0.0
    trades_per_year = 252 / avg_hold_days
    return (avg / std) * np.sqrt(trades_per_year)


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE DIRECTION
# ══════════════════════════════════════════════════════════════════════════
def test_inverse_direction(events, prices):
    print("\n" + "=" * 60)
    print("TEST 1: INVERSE DIRECTION (buy gap-DOWN instead of gap-UP)")
    print("=" * 60)

    # Original: gap-UP
    _, up_trades = backtest_trades(events, prices, direction_filter='up')
    up_rets = np.array([t['return_pct'] for t in up_trades])
    up_sharpe = compute_sharpe(up_rets)

    # Inverse: gap-DOWN
    _, down_trades = backtest_trades(events, prices, direction_filter='down')
    down_rets = np.array([t['return_pct'] for t in down_trades])
    down_sharpe = compute_sharpe(down_rets)

    down_avg = np.mean(down_rets) if len(down_rets) > 0 else 0
    down_wr = np.mean(down_rets > 0) * 100 if len(down_rets) > 0 else 0

    # PASS if inverse performs meaningfully worse than original
    # If inverse also has positive Sharpe > 0.5, signal is decorative
    inverse_also_works = down_sharpe > 0.5
    passed = not inverse_also_works

    print(f"  Original (gap-UP):  {len(up_trades)} trades, Sharpe={up_sharpe:.3f}, "
          f"avg_ret={np.mean(up_rets):.2f}%, WR={np.mean(up_rets > 0)*100:.1f}%")
    print(f"  Inverse (gap-DOWN): {len(down_trades)} trades, Sharpe={down_sharpe:.3f}, "
          f"avg_ret={down_avg:.2f}%, WR={down_wr:.1f}%")
    print(f"  {'PASS' if passed else 'FAIL'}: Inverse Sharpe={down_sharpe:.3f} "
          f"{'< 0.5 (direction matters)' if passed else '>= 0.5 (signal is decorative — any earnings event works)'}")

    return {
        'test': 'inverse_direction',
        'passed': passed,
        'original_sharpe': round(up_sharpe, 3),
        'original_trades': len(up_trades),
        'original_avg_ret': round(float(np.mean(up_rets)), 2),
        'inverse_sharpe': round(down_sharpe, 3),
        'inverse_trades': len(down_trades),
        'inverse_avg_ret': round(float(down_avg), 2),
        'inverse_wr': round(float(down_wr), 1),
        'verdict': 'Direction matters — gap-UP only' if passed else 'Signal is decorative — any earnings event works',
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM TIMING
# ══════════════════════════════════════════════════════════════════════════
def test_random_timing(events, prices):
    print("\n" + "=" * 60)
    print("TEST 2: RANDOM TIMING (same stocks, random entry dates)")
    print("=" * 60)

    # Get the actual trades from original backtest
    _, real_trades = backtest_trades(events, prices, direction_filter='up')
    real_rets = np.array([t['return_pct'] for t in real_trades])
    real_avg = np.mean(real_rets)

    # For each real trade, we know the ticker. Randomize the entry date.
    spy = prices.get('SPY')
    oot_days = spy.loc[OOT_START:OOT_END].index
    rng = np.random.default_rng(42)

    # Extract the list of tickers from real trades (preserving order/count)
    trade_tickers = [t['ticker'] for t in real_trades]

    perm_avgs = []
    for iteration in range(RANDOM_TIMING_ITERATIONS):
        perm_rets = []
        for ticker in trade_tickers:
            if ticker not in prices:
                continue
            df = prices[ticker]
            valid_days = df.index[(df.index >= OOT_START) & (df.index <= OOT_END)]
            if len(valid_days) < HOLD_DAYS + 2:
                continue

            idx = rng.integers(0, len(valid_days) - HOLD_DAYS - 1)
            entry_date = valid_days[idx]
            exit_date = valid_days[min(idx + HOLD_DAYS, len(valid_days) - 1)]

            entry_p = df.loc[entry_date, 'Open'] * (1 + SLIPPAGE_PCT)
            exit_p = df.loc[exit_date, 'Close'] * (1 - SLIPPAGE_PCT)
            ret = (exit_p / entry_p - 1) * 100
            perm_rets.append(ret)

        if perm_rets:
            perm_avgs.append(np.mean(perm_rets))

    perm_avgs = np.array(perm_avgs)
    p_value = float(np.mean(perm_avgs >= real_avg))

    # PASS if real timing is significantly better than random (p < 0.05)
    passed = p_value < 0.05

    print(f"  Real avg return per trade: {real_avg:.2f}%")
    print(f"  Random timing avg: {np.mean(perm_avgs):.2f}% +/- {np.std(perm_avgs):.2f}%")
    print(f"  p-value: {p_value:.4f}")
    print(f"  {'PASS' if passed else 'FAIL'}: p={p_value:.4f} "
          f"{'< 0.05 (timing matters)' if passed else '>= 0.05 (stock selection is the edge, not earnings timing)'}")

    return {
        'test': 'random_timing',
        'passed': passed,
        'real_avg_ret': round(float(real_avg), 2),
        'random_avg_ret': round(float(np.mean(perm_avgs)), 2),
        'random_std_ret': round(float(np.std(perm_avgs)), 2),
        'p_value': round(p_value, 4),
        'n_iterations': RANDOM_TIMING_ITERATIONS,
        'verdict': 'Earnings timing adds edge beyond stock selection' if passed else 'Stock selection drives returns, not earnings timing',
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
def test_subperiod_stability(events, prices):
    print("\n" + "=" * 60)
    print("TEST 3: SUB-PERIOD STABILITY (3 equal sub-periods)")
    print("=" * 60)

    _, all_trades = backtest_trades(events, prices, direction_filter='up')
    if not all_trades:
        return {'test': 'subperiod_stability', 'passed': False, 'verdict': 'No trades'}

    # Split OOT into 3 equal sub-periods by entry date
    total_days = (OOT_END - OOT_START).days
    period_len = total_days // 3

    boundaries = [
        OOT_START,
        OOT_START + pd.Timedelta(days=period_len),
        OOT_START + pd.Timedelta(days=2 * period_len),
        OOT_END,
    ]

    period_results = []
    all_positive = True

    for i in range(3):
        start = boundaries[i]
        end = boundaries[i + 1]
        period_trades = [t for t in all_trades if start <= t['entry_date'] < end]
        rets = np.array([t['return_pct'] for t in period_trades])

        if len(rets) < 3:
            sharpe = 0.0
        else:
            sharpe = compute_sharpe(rets)

        avg_ret = float(np.mean(rets)) if len(rets) > 0 else 0.0
        wr = float(np.mean(rets > 0) * 100) if len(rets) > 0 else 0.0

        if sharpe <= 0:
            all_positive = False

        period_results.append({
            'period': f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
            'n_trades': len(period_trades),
            'sharpe': round(sharpe, 3),
            'avg_ret': round(avg_ret, 2),
            'win_rate': round(wr, 1),
        })

        print(f"  Period {i+1} ({start.strftime('%Y-%m')} to {end.strftime('%Y-%m')}): "
              f"{len(period_trades)} trades, Sharpe={sharpe:.3f}, avg_ret={avg_ret:.2f}%, WR={wr:.1f}%")

    passed = all_positive
    print(f"  {'PASS' if passed else 'FAIL'}: {'All 3 sub-periods have positive Sharpe' if passed else 'At least one sub-period has non-positive Sharpe'}")

    return {
        'test': 'subperiod_stability',
        'passed': passed,
        'periods': period_results,
        'verdict': 'Edge is stable across time' if passed else 'Edge is concentrated in specific time periods',
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: TOP-TRADE REMOVAL
# ══════════════════════════════════════════════════════════════════════════
def test_top_trade_removal(events, prices):
    print("\n" + "=" * 60)
    print("TEST 4: TOP-TRADE REMOVAL (remove best 5% of trades)")
    print("=" * 60)

    _, all_trades = backtest_trades(events, prices, direction_filter='up')
    if not all_trades:
        return {'test': 'top_trade_removal', 'passed': False, 'verdict': 'No trades'}

    rets = np.array([t['return_pct'] for t in all_trades])
    original_sharpe = compute_sharpe(rets)

    # Remove top 5% by return
    n_remove = max(1, int(len(rets) * 0.05))
    threshold = np.sort(rets)[-n_remove]
    mask = rets < threshold
    # If multiple trades at threshold, keep some
    at_threshold = np.sum(rets == threshold)
    need_to_keep_at_threshold = np.sum(mask) + at_threshold - (len(rets) - n_remove)
    # Simple: sort and drop top n_remove
    sorted_indices = np.argsort(rets)
    keep_indices = sorted_indices[:len(rets) - n_remove]
    trimmed_rets = rets[keep_indices]

    trimmed_sharpe = compute_sharpe(trimmed_rets)
    trimmed_avg = float(np.mean(trimmed_rets))

    # Find what was removed
    removed_indices = sorted_indices[len(rets) - n_remove:]
    removed_trades = [all_trades[i] for i in removed_indices]
    removed_tickers = [t['ticker'] for t in removed_trades]
    removed_rets_vals = rets[removed_indices]

    passed = trimmed_sharpe > 0.5

    print(f"  Original: {len(rets)} trades, Sharpe={original_sharpe:.3f}")
    print(f"  Removed {n_remove} best trades (top 5%): {', '.join(removed_tickers)}")
    print(f"  Removed trade returns: {[f'{r:.1f}%' for r in sorted(removed_rets_vals, reverse=True)]}")
    print(f"  After removal: {len(trimmed_rets)} trades, Sharpe={trimmed_sharpe:.3f}, avg_ret={trimmed_avg:.2f}%")
    print(f"  {'PASS' if passed else 'FAIL'}: Trimmed Sharpe={trimmed_sharpe:.3f} "
          f"{'> 0.5 (not dependent on outliers)' if passed else '<= 0.5 (edge driven by few lucky trades)'}")

    return {
        'test': 'top_trade_removal',
        'passed': passed,
        'original_sharpe': round(original_sharpe, 3),
        'original_n_trades': len(rets),
        'n_removed': n_remove,
        'removed_tickers': removed_tickers,
        'removed_returns': [round(float(r), 2) for r in sorted(removed_rets_vals, reverse=True)],
        'trimmed_sharpe': round(trimmed_sharpe, 3),
        'trimmed_avg_ret': round(trimmed_avg, 2),
        'trimmed_n_trades': len(trimmed_rets),
        'verdict': 'Edge survives outlier removal' if passed else 'Edge depends on few lucky trades',
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: TICKER CONCENTRATION
# ══════════════════════════════════════════════════════════════════════════
def test_ticker_concentration(events, prices):
    print("\n" + "=" * 60)
    print("TEST 5: TICKER CONCENTRATION (remove top 3 tickers by P&L)")
    print("=" * 60)

    _, all_trades = backtest_trades(events, prices, direction_filter='up')
    if not all_trades:
        return {'test': 'ticker_concentration', 'passed': False, 'verdict': 'No trades'}

    rets = np.array([t['return_pct'] for t in all_trades])
    original_sharpe = compute_sharpe(rets)

    # Compute per-ticker total P&L contribution
    ticker_pnl = defaultdict(float)
    ticker_trades = defaultdict(int)
    ticker_avg_ret = defaultdict(list)
    for t in all_trades:
        ticker_pnl[t['ticker']] += t['pnl_dollar']
        ticker_trades[t['ticker']] += 1
        ticker_avg_ret[t['ticker']].append(t['return_pct'])

    # Sort by total P&L (descending)
    sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)

    print(f"  Ticker P&L breakdown (top 10):")
    for ticker, pnl in sorted_tickers[:10]:
        n = ticker_trades[ticker]
        avg = np.mean(ticker_avg_ret[ticker])
        print(f"    {ticker:6s}: ${pnl:8.2f} ({n} trades, avg ret {avg:.2f}%)")

    # Remove top 3 tickers
    top3_tickers = [t for t, _ in sorted_tickers[:3]]
    remaining_trades = [t for t in all_trades if t['ticker'] not in top3_tickers]
    remaining_rets = np.array([t['return_pct'] for t in remaining_trades])

    if len(remaining_rets) < 5:
        remaining_sharpe = 0.0
    else:
        remaining_sharpe = compute_sharpe(remaining_rets)

    remaining_avg = float(np.mean(remaining_rets)) if len(remaining_rets) > 0 else 0
    total_pnl = sum(t['pnl_dollar'] for t in all_trades)
    top3_pnl = sum(ticker_pnl[tk] for tk in top3_tickers)
    concentration_pct = (top3_pnl / total_pnl * 100) if total_pnl > 0 else 0

    passed = remaining_sharpe > 0.5

    print(f"\n  Top 3 tickers by P&L: {top3_tickers}")
    print(f"  Top 3 P&L: ${top3_pnl:.2f} / ${total_pnl:.2f} total ({concentration_pct:.1f}% concentration)")
    print(f"  After removing top 3: {len(remaining_rets)} trades, Sharpe={remaining_sharpe:.3f}, avg_ret={remaining_avg:.2f}%")
    print(f"  {'PASS' if passed else 'FAIL'}: Remaining Sharpe={remaining_sharpe:.3f} "
          f"{'> 0.5 (diversified edge)' if passed else '<= 0.5 (strategy is just buying {}, not PEAD)'.format('/'.join(top3_tickers))}")

    # Also check: unique tickers used
    unique_tickers = len(set(t['ticker'] for t in all_trades))
    print(f"  Total unique tickers traded: {unique_tickers}")

    return {
        'test': 'ticker_concentration',
        'passed': passed,
        'original_sharpe': round(original_sharpe, 3),
        'top3_tickers': top3_tickers,
        'top3_pnl': round(top3_pnl, 2),
        'total_pnl': round(total_pnl, 2),
        'concentration_pct': round(concentration_pct, 1),
        'remaining_sharpe': round(remaining_sharpe, 3),
        'remaining_avg_ret': round(remaining_avg, 2),
        'remaining_n_trades': len(remaining_rets),
        'unique_tickers': unique_tickers,
        'ticker_breakdown': {tk: {'pnl': round(pnl, 2), 'trades': ticker_trades[tk],
                                   'avg_ret': round(float(np.mean(ticker_avg_ret[tk])), 2)}
                              for tk, pnl in sorted_tickers},
        'verdict': 'Edge is diversified across tickers' if passed else f'Strategy is just buying {"/".join(top3_tickers)} repeatedly',
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("PEAD Variant A — Adversarial Validation (5 Tests)")
    print("=" * 70)

    prices = download_data()
    events = detect_earnings_events(prices, gap_threshold=0.02)
    print(f"  Total events detected: {len(events)}")

    oot_events = [e for e in events if e['date'] >= OOT_START]
    up_events = sum(1 for e in oot_events if e['direction'] == 'up')
    down_events = sum(1 for e in oot_events if e['direction'] == 'down')
    print(f"  OOT events: {len(oot_events)} (up: {up_events}, down: {down_events})")

    # Run all 5 tests
    results = {
        'strategy': 'pead_variant_A_adversarial',
        'timestamp': dt.datetime.now().isoformat(),
        'oot_period': f'{OOT_START.strftime("%Y-%m-%d")} to {OOT_END.strftime("%Y-%m-%d")}',
        'tests': {},
    }

    t1 = test_inverse_direction(events, prices)
    results['tests']['1_inverse_direction'] = t1

    t2 = test_random_timing(events, prices)
    results['tests']['2_random_timing'] = t2

    t3 = test_subperiod_stability(events, prices)
    results['tests']['3_subperiod_stability'] = t3

    t4 = test_top_trade_removal(events, prices)
    results['tests']['4_top_trade_removal'] = t4

    t5 = test_ticker_concentration(events, prices)
    results['tests']['5_ticker_concentration'] = t5

    # Summary
    tests = [t1, t2, t3, t4, t5]
    n_passed = sum(1 for t in tests if t['passed'])

    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Score: {n_passed}/5")
    print()
    for i, t in enumerate(tests, 1):
        status = "PASS" if t['passed'] else "FAIL"
        print(f"  {i}. {t['test']:25s} [{status}] — {t['verdict']}")

    results['summary'] = {
        'score': f'{n_passed}/5',
        'n_passed': n_passed,
        'n_tests': 5,
        'overall': 'STRONG' if n_passed >= 4 else 'MODERATE' if n_passed >= 3 else 'WEAK' if n_passed >= 2 else 'FAIL',
    }

    # Key concern check
    if not t5['passed']:
        results['summary']['key_concern'] = (
            f"Strategy is ticker-concentrated — top 3 tickers ({', '.join(t5['top3_tickers'])}) "
            f"account for {t5['concentration_pct']:.0f}% of P&L. "
            f"This may be 'buy mega-cap tech' with extra steps."
        )
    if not t2['passed']:
        results['summary']['timing_concern'] = (
            f"Random timing p={t2['p_value']:.4f} — earnings timing doesn't add "
            f"significant edge beyond being long these stocks."
        )

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
