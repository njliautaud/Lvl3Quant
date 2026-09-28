#!/usr/bin/env python3
"""
Survivorship Bias Test for Cross-Sectional Momentum & Short-Term Reversal
=========================================================================
Tests whether strategies backtested on today's large caps are just artifacts
of survivorship bias (these stocks survived and thrived → long bias looks like edge).

Tests:
1. SPY benchmark (hold 5d/10d weekly) vs reversal strategy Sharpe
2. Equal-weight all 30 stocks vs "bottom 3" reversal selection
3. Random 3-stock weekly selection (200 permutations) vs reversal Sharpe 1.05
4. Random 5-stock monthly selection (200 permutations) vs momentum Sharpe 1.27
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')
np.random.seed(42)

# --- Config ---
UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'UNH',
    'JNJ', 'WMT', 'PG', 'HD', 'MA', 'BAC', 'XOM', 'DIS', 'NFLX', 'AMD',
    'CRM', 'AVGO', 'COST', 'ABBV', 'LLY', 'PFE', 'KO', 'PEP', 'MRK', 'CSCO'
]
START = '2014-01-01'
END = '2026-07-01'
N_PERMUTATIONS = 200
REVERSAL_SHARPE_CLAIMED = 1.05
MOMENTUM_SHARPE_CLAIMED = 1.27

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/survivorship_bias_test'


def annualized_sharpe(returns, periods_per_year=52):
    """Annualized Sharpe from periodic returns."""
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def download_data():
    """Download all price data."""
    tickers = UNIVERSE + ['SPY']
    print(f"Downloading {len(tickers)} tickers from {START} to {END}...")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    print(f"Got {len(close)} trading days, {close.shape[1]} tickers")
    # Drop tickers with too much missing data
    missing = close.isnull().sum() / len(close)
    bad = missing[missing > 0.1].index.tolist()
    if bad:
        print(f"Warning: dropping tickers with >10% missing data: {bad}")
        close = close.drop(columns=bad)
    close = close.ffill().bfill()
    return close


def test_spy_benchmark(close):
    """Test 1: SPY buy-and-hold weekly (5d and 10d) as benchmark."""
    spy = close['SPY']
    results = {}

    for hold_days, label in [(5, '5d'), (10, '10d')]:
        # Weekly entry, hold for hold_days
        weekly_dates = spy.resample('W-FRI').last().dropna().index
        returns = []
        for i, date in enumerate(weekly_dates):
            if date not in spy.index:
                continue
            entry_idx = spy.index.get_loc(date)
            exit_idx = min(entry_idx + hold_days, len(spy) - 1)
            ret = spy.iloc[exit_idx] / spy.iloc[entry_idx] - 1
            returns.append(ret)

        returns = pd.Series(returns).dropna()
        sharpe = annualized_sharpe(returns, periods_per_year=52)
        results[f'spy_hold_{label}'] = {
            'sharpe': round(sharpe, 3),
            'mean_weekly_return_pct': round(returns.mean() * 100, 4),
            'n_trades': len(returns),
            'win_rate': round((returns > 0).mean() * 100, 1)
        }

    return results


def test_equal_weight(close):
    """Test 2: Equal-weight all 30 stocks weekly vs bottom-3 reversal."""
    stocks = [c for c in close.columns if c != 'SPY']
    stock_close = close[stocks]

    results = {}

    for hold_days, label in [(5, '5d'), (10, '10d')]:
        weekly_dates = stock_close.resample('W-FRI').last().dropna(how='all').index

        ew_returns = []  # equal weight all 30
        bottom3_returns = []  # bottom 3 by prior week return (reversal)

        for i in range(1, len(weekly_dates)):
            date = weekly_dates[i]
            prev_date = weekly_dates[i - 1]

            if date not in stock_close.index or prev_date not in stock_close.index:
                continue

            entry_idx = stock_close.index.get_loc(date)
            exit_idx = min(entry_idx + hold_days, len(stock_close) - 1)

            # Forward returns for all stocks
            fwd_ret = stock_close.iloc[exit_idx] / stock_close.iloc[entry_idx] - 1
            fwd_ret = fwd_ret.dropna()

            if len(fwd_ret) < 10:
                continue

            # Equal weight return
            ew_returns.append(fwd_ret.mean())

            # Prior week returns for reversal ranking
            prior_ret = stock_close.loc[date] / stock_close.loc[prev_date] - 1
            prior_ret = prior_ret.dropna()
            valid = prior_ret.index.intersection(fwd_ret.index)
            if len(valid) < 10:
                continue

            # Bottom 3 (worst performers = reversal candidates)
            bottom3 = prior_ret[valid].nsmallest(3).index
            bottom3_returns.append(fwd_ret[bottom3].mean())

        ew_returns = pd.Series(ew_returns).dropna()
        bottom3_returns = pd.Series(bottom3_returns).dropna()

        results[f'equal_weight_{label}'] = {
            'sharpe': round(annualized_sharpe(ew_returns), 3),
            'mean_weekly_return_pct': round(ew_returns.mean() * 100, 4),
            'n_trades': len(ew_returns)
        }
        results[f'bottom3_reversal_{label}'] = {
            'sharpe': round(annualized_sharpe(bottom3_returns), 3),
            'mean_weekly_return_pct': round(bottom3_returns.mean() * 100, 4),
            'n_trades': len(bottom3_returns)
        }

    return results


def test_random_reversal(close):
    """Test 3: Random 3-stock weekly selection, 200 permutations."""
    stocks = [c for c in close.columns if c != 'SPY']
    stock_close = close[stocks]

    results = {}

    for hold_days, label in [(5, '5d'), (10, '10d')]:
        weekly_dates = stock_close.resample('W-FRI').last().dropna(how='all').index

        perm_sharpes = []

        for perm in range(N_PERMUTATIONS):
            rng = np.random.RandomState(perm)
            returns = []

            for i in range(1, len(weekly_dates)):
                date = weekly_dates[i]

                if date not in stock_close.index:
                    continue

                entry_idx = stock_close.index.get_loc(date)
                exit_idx = min(entry_idx + hold_days, len(stock_close) - 1)

                fwd_ret = stock_close.iloc[exit_idx] / stock_close.iloc[entry_idx] - 1
                fwd_ret = fwd_ret.dropna()

                if len(fwd_ret) < 3:
                    continue

                # Random 3 stocks
                picks = rng.choice(fwd_ret.index.tolist(), size=min(3, len(fwd_ret)), replace=False)
                returns.append(fwd_ret[picks].mean())

            returns = pd.Series(returns).dropna()
            perm_sharpes.append(annualized_sharpe(returns))

        perm_sharpes = np.array(perm_sharpes)
        results[f'random_3_weekly_{label}'] = {
            'mean_sharpe': round(float(perm_sharpes.mean()), 3),
            'median_sharpe': round(float(np.median(perm_sharpes)), 3),
            'std_sharpe': round(float(perm_sharpes.std()), 3),
            'p5_sharpe': round(float(np.percentile(perm_sharpes, 5)), 3),
            'p95_sharpe': round(float(np.percentile(perm_sharpes, 95)), 3),
            'pct_above_1_0': round(float((perm_sharpes > 1.0).mean() * 100), 1),
            'pct_above_0_8': round(float((perm_sharpes > 0.8).mean() * 100), 1),
            'claimed_reversal_sharpe': REVERSAL_SHARPE_CLAIMED,
            'percentile_of_claimed': round(float((perm_sharpes < REVERSAL_SHARPE_CLAIMED).mean() * 100), 1)
        }

    return results


def test_random_momentum(close):
    """Test 4: Random 5-stock monthly selection, 200 permutations vs top-5 momentum."""
    stocks = [c for c in close.columns if c != 'SPY']
    stock_close = close[stocks]

    # Get last trading day of each month
    monthly_last = stock_close.groupby(stock_close.index.to_period('M')).apply(lambda x: x.index[-1])
    monthly_dates = monthly_last.values

    # Top-5 momentum (6m lookback, 1m hold)
    momentum_returns = []
    for i in range(6, len(monthly_dates)):
        date = monthly_dates[i]
        prev_6m = monthly_dates[i - 6]
        prev_1m = monthly_dates[i - 1]

        # rank at prev month end by 6m return, hold for current month
        mom_rank = stock_close.loc[prev_1m] / stock_close.loc[prev_6m] - 1
        mom_rank = mom_rank.dropna()

        # Forward 1-month return
        fwd_ret = stock_close.loc[date] / stock_close.loc[prev_1m] - 1
        fwd_ret = fwd_ret.dropna()

        valid = mom_rank.index.intersection(fwd_ret.index)
        if len(valid) < 10:
            continue

        top5 = mom_rank[valid].nlargest(5).index
        momentum_returns.append(fwd_ret[top5].mean())

    momentum_returns = pd.Series(momentum_returns).dropna()
    momentum_sharpe = annualized_sharpe(momentum_returns, periods_per_year=12)

    # Random 5-stock monthly, 200 permutations
    perm_sharpes = []
    for perm in range(N_PERMUTATIONS):
        rng = np.random.RandomState(perm + 1000)
        returns = []

        for i in range(6, len(monthly_dates)):
            date = monthly_dates[i]
            prev_1m = monthly_dates[i - 1]

            fwd_ret = stock_close.loc[date] / stock_close.loc[prev_1m] - 1
            fwd_ret = fwd_ret.dropna()

            if len(fwd_ret) < 5:
                continue

            picks = rng.choice(fwd_ret.index.tolist(), size=min(5, len(fwd_ret)), replace=False)
            returns.append(fwd_ret[picks].mean())

        returns = pd.Series(returns).dropna()
        perm_sharpes.append(annualized_sharpe(returns, periods_per_year=12))

    perm_sharpes = np.array(perm_sharpes)

    return {
        'top5_momentum_6m': {
            'sharpe': round(float(momentum_sharpe), 3),
            'mean_monthly_return_pct': round(momentum_returns.mean() * 100, 4),
            'n_trades': len(momentum_returns)
        },
        'random_5_monthly': {
            'mean_sharpe': round(float(perm_sharpes.mean()), 3),
            'median_sharpe': round(float(np.median(perm_sharpes)), 3),
            'std_sharpe': round(float(perm_sharpes.std()), 3),
            'p5_sharpe': round(float(np.percentile(perm_sharpes, 5)), 3),
            'p95_sharpe': round(float(np.percentile(perm_sharpes, 95)), 3),
            'pct_above_1_0': round(float((perm_sharpes > 1.0).mean() * 100), 1),
            'pct_above_0_8': round(float((perm_sharpes > 0.8).mean() * 100), 1),
            'claimed_momentum_sharpe': MOMENTUM_SHARPE_CLAIMED,
            'percentile_of_claimed': round(float((perm_sharpes < MOMENTUM_SHARPE_CLAIMED).mean() * 100), 1)
        }
    }


def make_verdict(results):
    """Determine if survivorship bias is the driver."""
    verdicts = []

    # Check 1: SPY benchmark
    spy_5d = results['spy_benchmark']['spy_hold_5d']['sharpe']
    spy_10d = results['spy_benchmark']['spy_hold_10d']['sharpe']
    if spy_5d > 0.8 or spy_10d > 0.8:
        verdicts.append(f"FAIL: SPY weekly hold produces Sharpe {max(spy_5d, spy_10d):.2f} — long bias alone explains much of the reversal edge")
    else:
        verdicts.append(f"PASS: SPY weekly hold Sharpe ({spy_5d:.2f}/{spy_10d:.2f}) is well below reversal claim of {REVERSAL_SHARPE_CLAIMED}")

    # Check 2: Equal weight vs selection
    ew_5d = results['equal_weight_vs_selection']['equal_weight_5d']['sharpe']
    b3_5d = results['equal_weight_vs_selection']['bottom3_reversal_5d']['sharpe']
    if abs(ew_5d - b3_5d) < 0.15:
        verdicts.append(f"FAIL: Equal-weight Sharpe ({ew_5d:.2f}) ≈ bottom-3 reversal ({b3_5d:.2f}) — selection doesn't matter, it's the universe")
    else:
        verdicts.append(f"MIXED: Equal-weight ({ew_5d:.2f}) vs bottom-3 ({b3_5d:.2f}) — difference is {abs(ew_5d-b3_5d):.2f}")

    # Check 3: Random reversal
    rand_5d = results['random_reversal']['random_3_weekly_5d']
    if rand_5d['pct_above_0_8'] > 50:
        verdicts.append(f"FAIL: {rand_5d['pct_above_0_8']}% of random 3-stock picks get Sharpe >0.8 — no stock selection edge, just universe bias")
    elif rand_5d['pct_above_0_8'] > 20:
        verdicts.append(f"WEAK: {rand_5d['pct_above_0_8']}% of random picks get Sharpe >0.8 — some universe bias present")
    else:
        verdicts.append(f"PASS: Only {rand_5d['pct_above_0_8']}% of random picks get Sharpe >0.8 — reversal selection has real edge")

    # Check 4: Random momentum
    rand_mom = results['random_momentum']['random_5_monthly']
    if rand_mom['pct_above_0_8'] > 50:
        verdicts.append(f"FAIL: {rand_mom['pct_above_0_8']}% of random 5-stock monthly picks get Sharpe >0.8 — momentum edge is universe bias")
    elif rand_mom['pct_above_0_8'] > 20:
        verdicts.append(f"WEAK: {rand_mom['pct_above_0_8']}% of random monthly picks get Sharpe >0.8 — some universe bias")
    else:
        verdicts.append(f"PASS: Only {rand_mom['pct_above_0_8']}% of random picks get Sharpe >0.8 — momentum selection has real edge")

    # Overall
    fail_count = sum(1 for v in verdicts if v.startswith('FAIL'))
    if fail_count >= 2:
        overall = "SURVIVORSHIP BIAS CONFIRMED — strategies are primarily artifacts of testing on a biased universe. Need to retest with point-in-time S&P 500 constituents or use ETFs only."
    elif fail_count == 1:
        overall = "PARTIAL SURVIVORSHIP BIAS — some edges may be real but the universe introduces significant long bias. Proceed with caution, consider ETF-only strategies."
    else:
        overall = "SURVIVORSHIP BIAS MINIMAL — stock selection strategies show genuine edge beyond universe bias."

    return {
        'individual_tests': verdicts,
        'overall_verdict': overall,
        'fail_count': fail_count,
        'total_tests': len(verdicts)
    }


def main():
    close = download_data()

    print("\n=== Test 1: SPY Benchmark ===")
    spy_results = test_spy_benchmark(close)
    for k, v in spy_results.items():
        print(f"  {k}: Sharpe={v['sharpe']}, WR={v.get('win_rate', 'N/A')}%")

    print("\n=== Test 2: Equal-Weight vs Bottom-3 Reversal ===")
    ew_results = test_equal_weight(close)
    for k, v in ew_results.items():
        print(f"  {k}: Sharpe={v['sharpe']}")

    print("\n=== Test 3: Random 3-Stock Weekly (200 perms) ===")
    rand_rev = test_random_reversal(close)
    for k, v in rand_rev.items():
        print(f"  {k}: mean_sharpe={v['mean_sharpe']}, pct>0.8={v['pct_above_0_8']}%")

    print("\n=== Test 4: Random 5-Stock Monthly (200 perms) ===")
    rand_mom = test_random_momentum(close)
    print(f"  top5_momentum: Sharpe={rand_mom['top5_momentum_6m']['sharpe']}")
    print(f"  random_5_monthly: mean_sharpe={rand_mom['random_5_monthly']['mean_sharpe']}, pct>0.8={rand_mom['random_5_monthly']['pct_above_0_8']}%")

    # Compile results
    results = {
        'test_date': datetime.now().isoformat(),
        'universe': UNIVERSE,
        'period': f'{START} to {END}',
        'n_permutations': N_PERMUTATIONS,
        'spy_benchmark': spy_results,
        'equal_weight_vs_selection': ew_results,
        'random_reversal': rand_rev,
        'random_momentum': rand_mom,
    }

    results['verdict'] = make_verdict(results)

    print(f"\n{'='*60}")
    print("VERDICT:")
    for v in results['verdict']['individual_tests']:
        print(f"  {v}")
    print(f"\n  OVERALL: {results['verdict']['overall_verdict']}")
    print(f"{'='*60}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
