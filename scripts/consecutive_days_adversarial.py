#!/usr/bin/env python3
"""
Adversarial Validation Battery for Consecutive Days Reversal Strategy
=====================================================================
6 tests to confirm the strategy edge is real, not curve-fit or luck.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────
START_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = '2022-01-01'
OOT_END = '2026-07-28'
HOLD_DAYS = 5
SPY_CONSEC = 4
QQQ_CONSEC = 5
MAX_CONCURRENT = 3
N_RANDOM = 1000

RANDOM_INSTRUMENTS = ['IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV',
                      'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/consecutive_days_adversarial_results.json'


def download_data(tickers, start, end):
    """Download adjusted close data for all tickers."""
    all_tickers = list(set(tickers))
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data[['Close']].copy()
        close.columns = all_tickers
    close = close.ffill()
    return close


def get_consecutive_days(close_series, direction='down'):
    """Count consecutive up or down days."""
    if direction == 'down':
        streak = (close_series < close_series.shift(1)).astype(int)
    else:
        streak = (close_series > close_series.shift(1)).astype(int)

    # Count consecutive streaks
    consec = pd.Series(0, index=close_series.index)
    for i in range(1, len(streak)):
        if streak.iloc[i] == 1:
            consec.iloc[i] = consec.iloc[i-1] + 1
        else:
            consec.iloc[i] = 0
    return consec


def run_strategy(close_df, spy_ticker='SPY', qqq_ticker='QQQ',
                 spy_consec=SPY_CONSEC, qqq_consec=QQQ_CONSEC,
                 hold_days=HOLD_DAYS, direction='down',
                 start_date=OOT_START, end_date=OOT_END):
    """
    Run the consecutive days reversal strategy.
    Returns list of trades with entry/exit dates and returns.
    """
    # Filter to OOT period
    mask = (close_df.index >= start_date) & (close_df.index <= end_date)
    df = close_df[mask].copy()

    trades = []
    active_positions = []  # list of (exit_idx, ticker)

    # Calculate consecutive days for each ticker
    consec = {}
    thresholds = {}
    if spy_ticker in df.columns:
        consec[spy_ticker] = get_consecutive_days(df[spy_ticker], direction=direction)
        thresholds[spy_ticker] = spy_consec
    if qqq_ticker in df.columns:
        consec[qqq_ticker] = get_consecutive_days(df[qqq_ticker], direction=direction)
        thresholds[qqq_ticker] = qqq_consec

    dates = df.index.tolist()

    for i, date in enumerate(dates):
        # Remove expired positions
        active_positions = [(ei, t) for ei, t in active_positions if ei > i]

        for ticker in consec:
            if consec[ticker].iloc[i] >= thresholds[ticker]:
                if len(active_positions) < MAX_CONCURRENT:
                    exit_idx = min(i + hold_days, len(dates) - 1)
                    if exit_idx > i:
                        entry_price = df[ticker].iloc[i] * (1 + SLIPPAGE_PCT)  # buy slippage
                        exit_price = df[ticker].iloc[exit_idx] * (1 - SLIPPAGE_PCT)  # sell slippage
                        ret = (exit_price - entry_price) / entry_price
                        trades.append({
                            'ticker': ticker,
                            'entry_date': str(dates[i].date()),
                            'exit_date': str(dates[exit_idx].date()),
                            'entry_price': float(entry_price),
                            'exit_price': float(exit_price),
                            'return': float(ret),
                            'pnl_dollars': float(ret * START_CAPITAL / MAX_CONCURRENT)
                        })
                        active_positions.append((exit_idx, ticker))

    return trades


def calc_metrics(trades):
    """Calculate strategy metrics from trade list."""
    if not trades:
        return {'sharpe': 0.0, 'pf': 0.0, 'wr': 0.0, 'max_dd': 0.0,
                'total_pnl': 0.0, 'n_trades': 0, 'avg_return': 0.0}

    returns = np.array([t['return'] for t in trades])
    pnls = np.array([t['pnl_dollars'] for t in trades])

    n_trades = len(returns)
    avg_ret = float(np.mean(returns))
    wr = float(np.sum(returns > 0) / n_trades) if n_trades > 0 else 0.0

    # Sharpe (annualized, assuming ~50 trades/year is generous, use daily-ish)
    if np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252 / HOLD_DAYS))
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.0
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.any(pnls < 0) else 0.001
    pf = gross_profit / gross_loss

    # Max drawdown on equity curve
    equity = np.cumsum(pnls) + START_CAPITAL
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    total_pnl = float(np.sum(pnls))

    return {
        'sharpe': round(sharpe, 3),
        'pf': round(pf, 3),
        'wr': round(wr, 4),
        'max_dd': round(max_dd, 4),
        'total_pnl': round(total_pnl, 2),
        'n_trades': int(n_trades),
        'avg_return': round(avg_ret, 6)
    }


def test_1_inverse_direction(close_df):
    """Test 1: Inverse Direction - buy after consecutive UP days instead of DOWN."""
    print("\n" + "="*70)
    print("TEST 1: INVERSE DIRECTION (buy after UP streaks)")
    print("="*70)

    trades = run_strategy(close_df, direction='up')
    metrics = calc_metrics(trades)

    passed = metrics['sharpe'] < 0.3

    print(f"  Inverse trades: {metrics['n_trades']}")
    print(f"  Inverse Sharpe: {metrics['sharpe']}")
    print(f"  Inverse PF: {metrics['pf']}")
    print(f"  Inverse WR: {metrics['wr']:.1%}")
    print(f"  Threshold: Sharpe < 0.3")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'Inverse Direction',
        'passed': bool(passed),
        'inverse_sharpe': metrics['sharpe'],
        'inverse_pf': metrics['pf'],
        'inverse_wr': metrics['wr'],
        'inverse_n_trades': metrics['n_trades'],
        'threshold': 'Sharpe < 0.3'
    }


def test_2_random_entry_timing(close_df, actual_trades):
    """Test 2: Random entry timing - same # trades, random dates."""
    print("\n" + "="*70)
    print("TEST 2: RANDOM ENTRY TIMING (1000 random portfolios)")
    print("="*70)

    n_actual = len(actual_trades)
    actual_pnl = sum(t['pnl_dollars'] for t in actual_trades)

    # Get valid trading dates in OOT
    mask = (close_df.index >= OOT_START) & (close_df.index <= OOT_END)
    oot_df = close_df[mask]
    dates = oot_df.index.tolist()
    tickers = ['SPY', 'QQQ']

    random_pnls = []
    rng = np.random.RandomState(42)

    for sim in range(N_RANDOM):
        sim_pnl = 0.0
        for _ in range(n_actual):
            ticker = rng.choice(tickers)
            idx = rng.randint(0, len(dates) - HOLD_DAYS - 1)
            entry_price = oot_df[ticker].iloc[idx] * (1 + SLIPPAGE_PCT)
            exit_price = oot_df[ticker].iloc[idx + HOLD_DAYS] * (1 - SLIPPAGE_PCT)
            ret = (exit_price - entry_price) / entry_price
            sim_pnl += float(ret * START_CAPITAL / MAX_CONCURRENT)
        random_pnls.append(sim_pnl)

    random_pnls = np.array(random_pnls)
    p_value = float(np.mean(random_pnls >= actual_pnl))

    passed = p_value < 0.05

    print(f"  Actual total P&L: ${actual_pnl:.2f}")
    print(f"  Random mean P&L: ${np.mean(random_pnls):.2f}")
    print(f"  Random median P&L: ${np.median(random_pnls):.2f}")
    print(f"  Random 95th pct P&L: ${np.percentile(random_pnls, 95):.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Threshold: p < 0.05")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'Random Entry Timing',
        'passed': bool(passed),
        'p_value': round(float(p_value), 4),
        'actual_pnl': round(float(actual_pnl), 2),
        'random_mean_pnl': round(float(np.mean(random_pnls)), 2),
        'random_median_pnl': round(float(np.median(random_pnls)), 2),
        'random_95th_pnl': round(float(np.percentile(random_pnls, 95)), 2),
        'threshold': 'p < 0.05'
    }


def test_3_random_instrument(close_df, actual_trades):
    """Test 3: Keep same entry dates but randomly substitute instruments."""
    print("\n" + "="*70)
    print("TEST 3: RANDOM INSTRUMENT SUBSTITUTION (1000 sims)")
    print("="*70)

    actual_pnl = sum(t['pnl_dollars'] for t in actual_trades)

    # Get entry/exit date pairs from actual trades
    trade_dates = [(t['entry_date'], t['exit_date']) for t in actual_trades]

    mask = (close_df.index >= OOT_START) & (close_df.index <= OOT_END)
    oot_df = close_df[mask]

    random_pnls = []
    rng = np.random.RandomState(42)

    for sim in range(N_RANDOM):
        sim_pnl = 0.0
        for entry_date, exit_date in trade_dates:
            ticker = rng.choice(RANDOM_INSTRUMENTS)
            if ticker not in oot_df.columns:
                continue
            try:
                entry_price = oot_df.loc[entry_date, ticker] * (1 + SLIPPAGE_PCT)
                exit_price = oot_df.loc[exit_date, ticker] * (1 - SLIPPAGE_PCT)
                if pd.isna(entry_price) or pd.isna(exit_price):
                    continue
                ret = (exit_price - entry_price) / entry_price
                sim_pnl += float(ret * START_CAPITAL / MAX_CONCURRENT)
            except (KeyError, IndexError):
                continue
        random_pnls.append(sim_pnl)

    random_pnls = np.array(random_pnls)
    p_value = float(np.mean(random_pnls >= actual_pnl))

    passed = p_value < 0.05

    print(f"  Actual total P&L: ${actual_pnl:.2f}")
    print(f"  Random instrument mean P&L: ${np.mean(random_pnls):.2f}")
    print(f"  Random instrument median P&L: ${np.median(random_pnls):.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Threshold: p < 0.05")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'Random Instrument',
        'passed': bool(passed),
        'p_value': round(float(p_value), 4),
        'actual_pnl': round(float(actual_pnl), 2),
        'random_mean_pnl': round(float(np.mean(random_pnls)), 2),
        'random_median_pnl': round(float(np.median(random_pnls)), 2),
        'threshold': 'p < 0.05'
    }


def test_4_sub_period_stability(close_df):
    """Test 4: Split OOT in half, both halves must have positive Sharpe."""
    print("\n" + "="*70)
    print("TEST 4: SUB-PERIOD STABILITY")
    print("="*70)

    mid_date = '2024-03-31'

    trades_h1 = run_strategy(close_df, start_date=OOT_START, end_date=mid_date)
    trades_h2 = run_strategy(close_df, start_date='2024-04-01', end_date=OOT_END)

    m1 = calc_metrics(trades_h1)
    m2 = calc_metrics(trades_h2)

    passed = m1['sharpe'] > 0 and m2['sharpe'] > 0

    print(f"  Half 1 ({OOT_START} to {mid_date}):")
    print(f"    Trades: {m1['n_trades']}, Sharpe: {m1['sharpe']}, PF: {m1['pf']}, WR: {m1['wr']:.1%}")
    print(f"  Half 2 (2024-04-01 to {OOT_END}):")
    print(f"    Trades: {m2['n_trades']}, Sharpe: {m2['sharpe']}, PF: {m2['pf']}, WR: {m2['wr']:.1%}")
    print(f"  Threshold: Both halves Sharpe > 0")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'Sub-period Stability',
        'passed': bool(passed),
        'half1_sharpe': m1['sharpe'],
        'half1_pf': m1['pf'],
        'half1_wr': m1['wr'],
        'half1_n_trades': m1['n_trades'],
        'half2_sharpe': m2['sharpe'],
        'half2_pf': m2['pf'],
        'half2_wr': m2['wr'],
        'half2_n_trades': m2['n_trades'],
        'threshold': 'Both halves Sharpe > 0'
    }


def test_5_top_trade_removal(actual_trades):
    """Test 5: Remove best 3 trades, recalculate."""
    print("\n" + "="*70)
    print("TEST 5: TOP TRADE REMOVAL (remove best 3)")
    print("="*70)

    sorted_trades = sorted(actual_trades, key=lambda t: t['return'], reverse=True)

    print(f"  Top 3 trades being removed:")
    for i, t in enumerate(sorted_trades[:3]):
        print(f"    #{i+1}: {t['ticker']} {t['entry_date']} -> {t['exit_date']}, return: {t['return']:.4%}")

    remaining = sorted_trades[3:]
    m = calc_metrics(remaining)

    passed = m['sharpe'] > 0.3

    print(f"  Remaining trades: {m['n_trades']}")
    print(f"  Remaining Sharpe: {m['sharpe']}")
    print(f"  Remaining PF: {m['pf']}")
    print(f"  Remaining WR: {m['wr']:.1%}")
    print(f"  Threshold: Sharpe > 0.3")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'Top Trade Removal',
        'passed': bool(passed),
        'remaining_sharpe': m['sharpe'],
        'remaining_pf': m['pf'],
        'remaining_wr': m['wr'],
        'remaining_n_trades': m['n_trades'],
        'removed_trades': [
            {'ticker': t['ticker'], 'entry': t['entry_date'],
             'exit': t['exit_date'], 'return': round(t['return'], 6)}
            for t in sorted_trades[:3]
        ],
        'threshold': 'Sharpe > 0.3'
    }


def test_6_alternative_thresholds(close_df):
    """Test 6: Sensitivity to parameters."""
    print("\n" + "="*70)
    print("TEST 6: ALTERNATIVE THRESHOLDS")
    print("="*70)

    alternatives = [
        {'name': 'SPY 3-day (instead of 4)', 'spy_consec': 3, 'qqq_consec': QQQ_CONSEC, 'hold': HOLD_DAYS},
        {'name': 'SPY 6-day (instead of 4)', 'spy_consec': 6, 'qqq_consec': QQQ_CONSEC, 'hold': HOLD_DAYS},
        {'name': 'Hold 3 days (instead of 5)', 'spy_consec': SPY_CONSEC, 'qqq_consec': QQQ_CONSEC, 'hold': 3},
        {'name': 'Hold 10 days (instead of 5)', 'spy_consec': SPY_CONSEC, 'qqq_consec': QQQ_CONSEC, 'hold': 10},
    ]

    alt_results = []
    n_pass = 0

    for alt in alternatives:
        trades = run_strategy(close_df, spy_consec=alt['spy_consec'],
                             qqq_consec=alt['qqq_consec'], hold_days=alt['hold'])
        m = calc_metrics(trades)
        alt_pass = m['sharpe'] > 0.3
        if alt_pass:
            n_pass += 1

        print(f"  {alt['name']}: Sharpe={m['sharpe']}, PF={m['pf']}, WR={m['wr']:.1%}, "
              f"trades={m['n_trades']}, {'PASS' if alt_pass else 'fail'}")

        alt_results.append({
            'name': alt['name'],
            'sharpe': m['sharpe'],
            'pf': m['pf'],
            'wr': m['wr'],
            'n_trades': m['n_trades'],
            'passed': bool(alt_pass)
        })

    passed = n_pass >= 2

    print(f"  Alternatives with Sharpe > 0.3: {n_pass}/4")
    print(f"  Threshold: >= 2/4 must pass")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'Alternative Thresholds',
        'passed': bool(passed),
        'n_alternatives_passed': int(n_pass),
        'alternatives': alt_results,
        'threshold': '>= 2/4 alternatives Sharpe > 0.3'
    }


def main():
    print("="*70)
    print("ADVERSARIAL VALIDATION BATTERY")
    print("Strategy: Consecutive Days Reversal")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print("="*70)

    # Download all needed data
    all_tickers = ['SPY', 'QQQ'] + RANDOM_INSTRUMENTS
    # Download extra history for streak calculation
    close_df = download_data(all_tickers, start='2021-06-01', end=OOT_END)

    print(f"\nData shape: {close_df.shape}")
    print(f"Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")

    # Run the actual strategy first
    print("\n" + "="*70)
    print("BASELINE: Original Strategy")
    print("="*70)
    actual_trades = run_strategy(close_df)
    baseline = calc_metrics(actual_trades)

    print(f"  Trades: {baseline['n_trades']}")
    print(f"  Sharpe: {baseline['sharpe']}")
    print(f"  PF: {baseline['pf']}")
    print(f"  WR: {baseline['wr']:.1%}")
    print(f"  Max DD: {baseline['max_dd']:.2%}")
    print(f"  Total P&L: ${baseline['total_pnl']:.2f}")

    # Run all 6 tests
    results = {}

    r1 = test_1_inverse_direction(close_df)
    results['test_1_inverse'] = r1

    r2 = test_2_random_entry_timing(close_df, actual_trades)
    results['test_2_random_timing'] = r2

    r3 = test_3_random_instrument(close_df, actual_trades)
    results['test_3_random_instrument'] = r3

    r4 = test_4_sub_period_stability(close_df)
    results['test_4_sub_period'] = r4

    r5 = test_5_top_trade_removal(actual_trades)
    results['test_5_top_removal'] = r5

    r6 = test_6_alternative_thresholds(close_df)
    results['test_6_alt_thresholds'] = r6

    # Overall verdict
    n_passed = sum(1 for r in results.values() if r['passed'])

    if n_passed >= 5:
        verdict = 'CONFIRMED'
    elif n_passed >= 3:
        verdict = 'SUSPICIOUS'
    else:
        verdict = 'DEAD'

    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)
    for key, r in results.items():
        status = 'PASS' if r['passed'] else 'FAIL'
        print(f"  {r['test']}: {status}")
    print(f"\n  Tests passed: {n_passed}/6")
    print(f"  VERDICT: {verdict}")
    print("="*70)

    # Save results
    output = {
        'strategy': 'Consecutive Days Reversal',
        'oot_period': f'{OOT_START} to {OOT_END}',
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline,
        'baseline_trades': actual_trades,
        'tests': results,
        'n_passed': int(n_passed),
        'verdict': verdict
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")

    return output


if __name__ == '__main__':
    main()
