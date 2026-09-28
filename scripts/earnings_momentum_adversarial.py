#!/usr/bin/env python3
"""
Earnings Surprise Momentum Variant B (Beat-Chain) — ADVERSARIAL VALIDATION
===========================================================================
6 adversarial tests with clear PASS/FAIL criteria:

1. Inverse Direction: Short beat-chain stocks → PASS if inverse Sharpe < 0
2. Buy-and-Hold Comparison: vs equal-weight portfolio → PASS if strategy beats by >= 0.3 Sharpe
3. Random Timing Percentile: 1000 permutations → PASS if real >= 90th percentile
4. Remove Key Component: single-beat vs chain → PASS if chain has >= 0.3 Sharpe advantage
5. Cost Sensitivity: slippage sweep → PASS if Sharpe > 0.5 at 0.20%
6. Sub-Period Stability: 4 sub-periods → PASS if >= 3/4 have positive Sharpe

Universe: 24 growth stocks
OOT: 2022-01-01 to 2026-07-29
Costs: $0 commission (Robinhood), 0.02% baseline slippage
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 10000.0
MAX_POSITIONS = 5
SLIPPAGE_PCT = 0.0002  # 0.02% baseline
COMMISSION = 0.0
HOLD_DAYS = 60

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
    'CRM', 'ADBE', 'NFLX', 'AVGO', 'COST', 'PEP', 'LLY', 'UNH',
    'V', 'MA', 'JPM', 'HD', 'INTC', 'MU', 'QCOM', 'PYPL'
]

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')
PERM_ITERATIONS = 1000


# ── Data Download ─────────────────────────────────────────────────────────
def download_data():
    """Download price data for universe + SPY."""
    print("Downloading price data...")
    all_tickers = list(set(UNIVERSE + ['SPY']))
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


def detect_earnings_events(prices):
    """Detect earnings via overnight gap (>3%) + volume spike (>1.5x 20d avg)."""
    events = {}
    for ticker in UNIVERSE:
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

    total = sum(len(v) for v in events.values())
    beats = sum(sum(1 for e in v if e['is_beat']) for v in events.values())
    print(f"  Detected {total} earnings events ({beats} beats) across {len(events)} stocks")
    return events


# ── Price Helpers ─────────────────────────────────────────────────────────
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


# ── Core Strategy: Beat-Chain (Variant B) ─────────────────────────────────
def run_beat_chain(prices, events, slippage=SLIPPAGE_PCT, direction='long',
                   require_chain=True):
    """
    Beat-Chain strategy.
    direction='long': buy after consecutive beats
    direction='short': short after consecutive beats (inverse test)
    require_chain=True: need previous beat too (variant B)
    require_chain=False: any single beat suffices (variant A-like)
    """
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

            # Chain requirement
            if require_chain:
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

            # Apply slippage
            entry_price *= (1 + slippage)
            exit_price *= (1 - slippage)

            if direction == 'long':
                ret = (exit_price - entry_price) / entry_price
            else:
                # Short: profit when price falls
                ret = (entry_price - exit_price) / entry_price

            trades.append({
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'return_pct': round(ret * 100, 4),
                'gap_pct': round(evt['gap_pct'] * 100, 2),
                'hold_days': HOLD_DAYS,
            })

    return trades


def run_inverse_miss(prices, events, slippage=SLIPPAGE_PCT):
    """Buy stocks that MISSED earnings (negative gap) with chain of misses."""
    trades = []
    for ticker, evts in events.items():
        if ticker not in prices:
            continue
        df = prices[ticker]

        for i, evt in enumerate(evts):
            # Look for misses instead of beats
            if evt['is_beat'] or evt['gap_abs'] < 0.03:
                continue
            if evt['date'] < OOT_START:
                continue

            # Chain: previous was also a miss
            if i == 0:
                continue
            prev_evt = evts[i - 1]
            if prev_evt['is_beat']:
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
                'return_pct': round(ret * 100, 4),
            })

    return trades


# ── Metrics ───────────────────────────────────────────────────────────────
def compute_sharpe(trades, annualize_factor=None):
    """Compute annualized Sharpe from trade returns."""
    if not trades or len(trades) < 2:
        return 0.0
    returns = np.array([t['return_pct'] / 100 for t in trades])
    n = len(returns)
    avg = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std < 1e-10:
        return 0.0
    if annualize_factor is None:
        trades_per_year = max(n / 4.5, 1)  # ~4.5 year OOT
    else:
        trades_per_year = annualize_factor
    return float((avg / std) * np.sqrt(trades_per_year))


def compute_full_metrics(trades):
    """Compute Sharpe, Sortino, WR, PF, MaxDD."""
    if not trades:
        return {'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'win_rate': 0,
                'profit_factor': 0, 'max_dd_pct': 0, 'total_return_pct': 0,
                'avg_return_pct': 0}

    returns = np.array([t['return_pct'] / 100 for t in trades])
    n = len(returns)
    avg = np.mean(returns)
    std = np.std(returns, ddof=1) if n > 1 else 1e-6
    trades_per_year = max(n / 4.5, 1)

    sharpe = (avg / std) * np.sqrt(trades_per_year) if std > 1e-10 else 0

    downside = returns[returns < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (avg / down_std) * np.sqrt(trades_per_year) if down_std > 1e-10 else 0

    win_rate = (returns > 0).sum() / n * 100
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0
    for r in returns:
        pos_size = equity / MAX_POSITIONS
        equity += pos_size * r
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    total_ret = (equity - CAPITAL) / CAPITAL * 100

    return {
        'n_trades': n,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(pf, 3),
        'max_dd_pct': round(max_dd * 100, 2),
        'total_return_pct': round(total_ret, 2),
        'avg_return_pct': round(avg * 100, 2),
    }


# ── Test 1: Inverse Direction ─────────────────────────────────────────────
def test_inverse_direction(prices, events):
    """
    Short beat-chain stocks OR buy miss-chain stocks.
    PASS = inverse Sharpe < 0 (negative).
    """
    print("\n" + "=" * 60)
    print("TEST 1: INVERSE DIRECTION")
    print("=" * 60)

    # Approach A: Short beat-chain stocks
    short_trades = run_beat_chain(prices, events, direction='short')
    short_sharpe = compute_sharpe(short_trades)

    # Approach B: Buy miss-chain stocks
    miss_trades = run_inverse_miss(prices, events)
    miss_sharpe = compute_sharpe(miss_trades)

    # Use the best inverse Sharpe for the test
    best_inverse_sharpe = max(short_sharpe, miss_sharpe)

    passed = best_inverse_sharpe < 0

    print(f"  Short beat-chain: {len(short_trades)} trades, Sharpe={short_sharpe:.3f}")
    print(f"  Buy miss-chain:   {len(miss_trades)} trades, Sharpe={miss_sharpe:.3f}")
    print(f"  Best inverse Sharpe: {best_inverse_sharpe:.3f}")
    print(f"  PASS criteria: best inverse Sharpe < 0")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'inverse_direction',
        'passed': passed,
        'short_beat_chain': {'n_trades': len(short_trades), 'sharpe': round(short_sharpe, 3)},
        'buy_miss_chain': {'n_trades': len(miss_trades), 'sharpe': round(miss_sharpe, 3)},
        'best_inverse_sharpe': round(best_inverse_sharpe, 3),
        'criteria': 'best inverse Sharpe < 0',
    }


# ── Test 2: Buy-and-Hold Comparison ───────────────────────────────────────
def test_buy_and_hold(prices, events):
    """
    Compare strategy to equal-weight buy-hold of same 24 stocks.
    PASS = strategy Sharpe beats buy-hold by >= 0.3.
    """
    print("\n" + "=" * 60)
    print("TEST 2: BUY-AND-HOLD COMPARISON")
    print("=" * 60)

    # Strategy Sharpe
    chain_trades = run_beat_chain(prices, events, require_chain=True)
    strategy_sharpe = compute_sharpe(chain_trades)

    # Buy-and-hold: equal-weight, compute daily returns then annualized Sharpe
    daily_returns_list = []
    for ticker in UNIVERSE:
        if ticker not in prices:
            continue
        df = prices[ticker]
        mask = (df.index >= OOT_START) & (df.index <= OOT_END)
        oot_df = df[mask]
        if len(oot_df) < 10:
            continue
        rets = oot_df['Close'].pct_change().dropna()
        daily_returns_list.append(rets)

    if daily_returns_list:
        combined = pd.concat(daily_returns_list, axis=1).fillna(0)
        ew_daily = combined.mean(axis=1)
        bh_annual_ret = ew_daily.mean() * 252
        bh_annual_vol = ew_daily.std() * np.sqrt(252)
        bh_sharpe = bh_annual_ret / bh_annual_vol if bh_annual_vol > 0 else 0
    else:
        bh_sharpe = 0

    advantage = strategy_sharpe - bh_sharpe
    passed = advantage >= 0.3

    print(f"  Strategy Sharpe (beat-chain): {strategy_sharpe:.3f}")
    print(f"  Buy-and-Hold EW Sharpe:       {bh_sharpe:.3f}")
    print(f"  Advantage:                    {advantage:.3f}")
    print(f"  PASS criteria: advantage >= 0.3")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'buy_and_hold_comparison',
        'passed': passed,
        'strategy_sharpe': round(strategy_sharpe, 3),
        'buyhold_sharpe': round(float(bh_sharpe), 3),
        'advantage': round(advantage, 3),
        'criteria': 'strategy Sharpe - BH Sharpe >= 0.3',
    }


# ── Test 3: Random Timing Percentile ─────────────────────────────────────
def test_random_timing(prices, events):
    """
    Shuffle entry dates randomly (same stocks, same hold period).
    PASS = real strategy >= 90th percentile of random timings.
    """
    print("\n" + "=" * 60)
    print("TEST 3: RANDOM TIMING PERCENTILE")
    print("=" * 60)

    # Real strategy
    chain_trades = run_beat_chain(prices, events, require_chain=True)
    real_sharpe = compute_sharpe(chain_trades)
    n_trades = len(chain_trades)

    if n_trades < 5:
        print("  Not enough trades for permutation test")
        return {
            'test': 'random_timing_percentile',
            'passed': False,
            'real_sharpe': round(real_sharpe, 3),
            'percentile': 0,
            'criteria': '>= 90th percentile',
            'note': 'insufficient trades',
        }

    # Get the tickers used in real trades
    trade_tickers = [t['ticker'] for t in chain_trades]

    # Build pool of valid random entry dates for each ticker
    ticker_dates = {}
    for ticker in set(trade_tickers):
        if ticker not in prices:
            continue
        df = prices[ticker]
        mask = (df.index >= OOT_START) & (df.index <= OOT_END)
        valid_dates = df.index[mask]
        if len(valid_dates) > HOLD_DAYS:
            ticker_dates[ticker] = valid_dates[:-HOLD_DAYS]

    rng = np.random.RandomState(42)
    random_sharpes = []

    print(f"  Running {PERM_ITERATIONS} permutations...")
    for perm_i in range(PERM_ITERATIONS):
        random_trades = []
        for ticker in trade_tickers:
            if ticker not in ticker_dates or len(ticker_dates[ticker]) == 0:
                continue
            df = prices[ticker]
            rand_idx = rng.randint(0, len(ticker_dates[ticker]))
            rand_date = ticker_dates[ticker][rand_idx]

            entry_price = get_price_at(df, rand_date, 'Open', offset_days=1)
            exit_price = get_price_at(df, rand_date, 'Close', offset_days=HOLD_DAYS)
            if entry_price is None or exit_price is None:
                continue

            entry_price *= (1 + SLIPPAGE_PCT)
            exit_price *= (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price
            random_trades.append({'return_pct': ret * 100})

        if len(random_trades) >= 3:
            random_sharpes.append(compute_sharpe(random_trades))

    if not random_sharpes:
        percentile = 0
    else:
        percentile = float(np.mean([1 for s in random_sharpes if real_sharpe > s]) / len(random_sharpes) * 100)

    passed = percentile >= 90

    print(f"  Real Sharpe:    {real_sharpe:.3f}")
    if random_sharpes:
        print(f"  Median random:  {np.median(random_sharpes):.3f}")
    print(f"  Percentile:     {percentile:.1f}th")
    print(f"  PASS criteria:  >= 90th percentile")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'random_timing_percentile',
        'passed': passed,
        'real_sharpe': round(real_sharpe, 3),
        'percentile': round(percentile, 1),
        'random_median_sharpe': round(float(np.median(random_sharpes)), 3) if random_sharpes else 0,
        'random_p10_sharpe': round(float(np.percentile(random_sharpes, 10)), 3) if random_sharpes else 0,
        'random_p90_sharpe': round(float(np.percentile(random_sharpes, 90)), 3) if random_sharpes else 0,
        'n_permutations': len(random_sharpes),
        'criteria': '>= 90th percentile',
    }


# ── Test 4: Remove Key Component (Chain vs Single Beat) ──────────────────
def test_remove_chain(prices, events):
    """
    Run without chain requirement (single beats).
    PASS = chain version has >= 0.3 Sharpe advantage over single-beat.
    """
    print("\n" + "=" * 60)
    print("TEST 4: REMOVE KEY COMPONENT (chain vs single beat)")
    print("=" * 60)

    chain_trades = run_beat_chain(prices, events, require_chain=True)
    single_trades = run_beat_chain(prices, events, require_chain=False)

    chain_sharpe = compute_sharpe(chain_trades)
    single_sharpe = compute_sharpe(single_trades)
    chain_metrics = compute_full_metrics(chain_trades)
    single_metrics = compute_full_metrics(single_trades)

    advantage = chain_sharpe - single_sharpe
    passed = advantage >= 0.3

    print(f"  Chain (Variant B): {len(chain_trades)} trades, Sharpe={chain_sharpe:.3f}, WR={chain_metrics['win_rate']:.1f}%")
    print(f"  Single beat:       {len(single_trades)} trades, Sharpe={single_sharpe:.3f}, WR={single_metrics['win_rate']:.1f}%")
    print(f"  Chain advantage:   {advantage:.3f}")
    print(f"  PASS criteria: chain Sharpe - single Sharpe >= 0.3")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'remove_chain_component',
        'passed': passed,
        'chain_sharpe': round(chain_sharpe, 3),
        'chain_n_trades': len(chain_trades),
        'chain_win_rate': chain_metrics['win_rate'],
        'single_sharpe': round(single_sharpe, 3),
        'single_n_trades': len(single_trades),
        'single_win_rate': single_metrics['win_rate'],
        'chain_advantage': round(advantage, 3),
        'criteria': 'chain Sharpe - single Sharpe >= 0.3',
    }


# ── Test 5: Cost Sensitivity ─────────────────────────────────────────────
def test_cost_sensitivity(prices, events):
    """
    Test at 0.02%, 0.05%, 0.10%, 0.20% slippage.
    PASS = Sharpe > 0.5 at 0.20% slippage.
    """
    print("\n" + "=" * 60)
    print("TEST 5: COST SENSITIVITY")
    print("=" * 60)

    slippage_levels = [0.0002, 0.0005, 0.001, 0.002]
    results_by_slip = {}

    for slip in slippage_levels:
        trades = run_beat_chain(prices, events, slippage=slip, require_chain=True)
        metrics = compute_full_metrics(trades)
        sharpe = compute_sharpe(trades)
        label = f"{slip*100:.2f}%"
        results_by_slip[label] = {
            'slippage_pct': slip * 100,
            'n_trades': len(trades),
            'sharpe': round(sharpe, 3),
            'win_rate': metrics['win_rate'],
            'total_return_pct': metrics['total_return_pct'],
        }
        print(f"  Slippage {label}: Sharpe={sharpe:.3f}, WR={metrics['win_rate']:.1f}%, Return={metrics['total_return_pct']:.1f}%")

    worst_sharpe = results_by_slip['0.20%']['sharpe']
    passed = worst_sharpe > 0.5

    print(f"  Sharpe at 0.20% slippage: {worst_sharpe:.3f}")
    print(f"  PASS criteria: Sharpe > 0.5 at 0.20%")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'cost_sensitivity',
        'passed': passed,
        'slippage_results': results_by_slip,
        'sharpe_at_020pct': worst_sharpe,
        'criteria': 'Sharpe > 0.5 at 0.20% slippage',
    }


# ── Test 6: Sub-Period Stability ──────────────────────────────────────────
def test_sub_period_stability(prices, events):
    """
    Split OOT into 4 equal sub-periods. PASS = >= 3/4 have positive Sharpe.
    """
    print("\n" + "=" * 60)
    print("TEST 6: SUB-PERIOD STABILITY")
    print("=" * 60)

    all_trades = run_beat_chain(prices, events, require_chain=True)

    # 4 sub-periods (~1.1 year each)
    sub_periods = [
        ('2022-01-01', '2023-02-28', 'H1 2022-Q1 2023'),
        ('2023-03-01', '2024-04-30', 'Q2 2023-Q1 2024'),
        ('2024-05-01', '2025-07-31', 'Q2 2024-Q2 2025'),
        ('2025-08-01', '2026-07-29', 'Q3 2025-Q3 2026'),
    ]

    sub_results = []
    positive_count = 0

    for start, end, label in sub_periods:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        period_trades = [t for t in all_trades
                         if start_dt <= pd.Timestamp(t['entry_date']) <= end_dt]

        if len(period_trades) < 2:
            sharpe = 0.0
        else:
            sharpe = compute_sharpe(period_trades)

        is_positive = sharpe > 0
        if is_positive:
            positive_count += 1

        sub_results.append({
            'period': label,
            'start': start,
            'end': end,
            'n_trades': len(period_trades),
            'sharpe': round(sharpe, 3),
            'positive': is_positive,
        })

        status = "+" if is_positive else "-"
        print(f"  [{status}] {label}: {len(period_trades)} trades, Sharpe={sharpe:.3f}")

    passed = positive_count >= 3

    print(f"  Positive periods: {positive_count}/4")
    print(f"  PASS criteria: >= 3/4 sub-periods with positive Sharpe")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'sub_period_stability',
        'passed': passed,
        'positive_periods': positive_count,
        'total_periods': 4,
        'sub_periods': sub_results,
        'criteria': '>= 3/4 sub-periods with positive Sharpe',
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Earnings Surprise Momentum - Variant B (Beat-Chain)")
    print(f"OOT: {OOT_START.date()} to {OOT_END.date()}")
    print(f"Universe: {len(UNIVERSE)} stocks")
    print(f"Hold: {HOLD_DAYS} days | Baseline slippage: {SLIPPAGE_PCT*100:.2f}%")
    print("=" * 70)

    # Download data
    prices = download_data()
    events = detect_earnings_events(prices)

    # Baseline metrics for reference
    baseline_trades = run_beat_chain(prices, events, require_chain=True)
    baseline_metrics = compute_full_metrics(baseline_trades)
    print(f"\nBaseline Beat-Chain: {baseline_metrics['n_trades']} trades, "
          f"Sharpe={baseline_metrics['sharpe']:.3f}, "
          f"Sortino={baseline_metrics['sortino']:.3f}, "
          f"WR={baseline_metrics['win_rate']:.1f}%")

    # Run 6 adversarial tests
    results = {}
    results['test_1_inverse'] = test_inverse_direction(prices, events)
    results['test_2_buyhold'] = test_buy_and_hold(prices, events)
    results['test_3_random_timing'] = test_random_timing(prices, events)
    results['test_4_remove_chain'] = test_remove_chain(prices, events)
    results['test_5_cost_sensitivity'] = test_cost_sensitivity(prices, events)
    results['test_6_sub_period'] = test_sub_period_stability(prices, events)

    # Summary
    n_passed = sum(1 for r in results.values() if r['passed'])

    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)

    for key, r in results.items():
        status = "PASS" if r['passed'] else "FAIL"
        print(f"  [{status}] {r['test']}")

    print(f"\n  OVERALL SCORE: {n_passed}/6 tests passed")

    if n_passed >= 5:
        verdict = "STRONG - strategy has genuine edge"
    elif n_passed >= 3:
        verdict = "MODERATE - some concerns, investigate failures"
    else:
        verdict = "WEAK - likely no real edge, mostly noise or beta"
    print(f"  VERDICT: {verdict}")

    # Save results
    output = {
        'strategy': 'earnings_surprise_momentum_variant_B_beat_chain',
        'timestamp': dt.datetime.now().isoformat(),
        'oot_period': f'{OOT_START.date()} to {OOT_END.date()}',
        'universe': UNIVERSE,
        'hold_days': HOLD_DAYS,
        'baseline_slippage': SLIPPAGE_PCT,
        'baseline_metrics': baseline_metrics,
        'adversarial_tests': results,
        'overall_score': f'{n_passed}/6',
        'n_passed': n_passed,
        'verdict': verdict,
    }

    out_path = Path('/home/jupiter/Lvl3Quant/data/earnings_momentum_adversarial_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    return output


if __name__ == '__main__':
    main()
