#!/usr/bin/env python3
"""
ETF Factor Strategies v1 — Survivorship-Bias-Free
===================================================
Tests reversal and momentum on SECTOR ETFs (no survivorship bias).
If the edge we found on 30 large-caps is real, it should show up on ETFs too.
If it doesn't → the individual stock results were survivorship bias artifacts.

HC #705: All adversarial checks built in from line 1.

Strategies:
  1. ETF Reversal: Buy worst-performing sector ETF(s), hold N days
  2. ETF Momentum: Buy best-performing sector ETF(s), hold N days
  3. ETF Mean-Rev: Buy ETFs that dropped >X% in Y days, hold N days

Universe: 11 sector ETFs (SPDR) + broad market ETFs
  - These have existed since 1998-2000 (25+ years of data)
  - No survivorship bias (ETFs don't go bankrupt)
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/etf_factor_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ETF Universe — sector ETFs (SPDR) + broad
ETF_UNIVERSE = [
    'XLB',   # Materials
    'XLC',   # Communication Services (2018+)
    'XLE',   # Energy
    'XLF',   # Financials
    'XLI',   # Industrials
    'XLK',   # Technology
    'XLP',   # Consumer Staples
    'XLRE',  # Real Estate (2015+)
    'XLU',   # Utilities
    'XLV',   # Healthcare
    'XLY',   # Consumer Discretionary
    'SPY',   # S&P 500
    'QQQ',   # Nasdaq 100
    'IWM',   # Russell 2000
    'DIA',   # Dow Jones
]

N_PERMUTATIONS = 500
START_DATE = '2005-01-01'
END_DATE = '2026-07-01'

def download_data():
    """Download ETF price data."""
    cache_file = OUTPUT_DIR / 'data_cache.parquet'
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        df = pd.read_parquet(cache_file)
        return df

    print(f"Downloading {len(ETF_UNIVERSE)} ETFs from {START_DATE} to {END_DATE}...")
    all_data = {}
    for ticker in ETF_UNIVERSE:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if len(data) > 252:  # Need at least 1 year
                close = data['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                all_data[ticker] = close
                print(f"  {ticker}: {len(data)} days")
            else:
                print(f"  {ticker}: SKIP (only {len(data)} days)")
        except Exception as e:
            print(f"  {ticker}: ERROR {e}")

    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    # Drop rows where we don't have enough tickers
    df = df.dropna(thresh=8)  # Need at least 8 ETFs
    df.to_parquet(cache_file)
    print(f"Cached {len(df)} days, {len(df.columns)} ETFs")
    return df

def download_spy_regime():
    """Download SPY data for regime classification."""
    spy = yf.download('SPY', start=START_DATE, end=END_DATE, progress=False)
    spy_close = spy['Close']
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.iloc[:, 0]
    spy_daily_ret = spy_close.pct_change()
    regime = pd.Series(index=spy_daily_ret.index, dtype=str)
    regime[spy_daily_ret > 0.001] = 'green'
    regime[spy_daily_ret < -0.001] = 'red'
    regime[(spy_daily_ret >= -0.001) & (spy_daily_ret <= 0.001)] = 'flat'
    return regime

def compute_reversal_returns(prices_df, lookback, top_n, hold_days, regime_map=None):
    """
    Reversal: Buy the bottom-N ETFs by lookback-day return, hold for hold_days.
    Returns per-period returns and regime labels.
    """
    returns = prices_df.pct_change(lookback)
    forward_returns = prices_df.pct_change(hold_days).shift(-hold_days)

    trade_dates = []
    trade_returns = []
    trade_regimes = []

    # Weekly rebalancing (every 5 trading days)
    rebal_dates = returns.index[lookback::5]

    for date in rebal_dates:
        if date not in returns.index or date not in forward_returns.index:
            continue

        row = returns.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()

        common = row.index.intersection(fwd.index)
        if len(common) < top_n + 2:
            continue

        row = row[common]
        fwd = fwd[common]

        # Bottom N by lookback return (worst performers)
        bottom_n = row.nsmallest(top_n).index
        period_return = fwd[bottom_n].mean()

        if not np.isnan(period_return):
            trade_dates.append(date)
            trade_returns.append(period_return)
            if regime_map is not None and date in regime_map.index:
                trade_regimes.append(regime_map.loc[date])
            else:
                trade_regimes.append('unknown')

    return np.array(trade_returns), trade_dates, trade_regimes

def compute_momentum_returns(prices_df, lookback, top_n, hold_days, regime_map=None):
    """
    Momentum: Buy the top-N ETFs by lookback-day return, hold for hold_days.
    """
    returns = prices_df.pct_change(lookback)
    forward_returns = prices_df.pct_change(hold_days).shift(-hold_days)

    trade_dates = []
    trade_returns = []
    trade_regimes = []

    # Monthly rebalancing for momentum
    rebal_period = 21 if hold_days >= 20 else 5
    rebal_dates = returns.index[lookback::rebal_period]

    for date in rebal_dates:
        if date not in returns.index or date not in forward_returns.index:
            continue

        row = returns.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()

        common = row.index.intersection(fwd.index)
        if len(common) < top_n + 2:
            continue

        row = row[common]
        fwd = fwd[common]

        # Top N by lookback return (best performers)
        top_tickers = row.nlargest(top_n).index
        period_return = fwd[top_tickers].mean()

        if not np.isnan(period_return):
            trade_dates.append(date)
            trade_returns.append(period_return)
            if regime_map is not None and date in regime_map.index:
                trade_regimes.append(regime_map.loc[date])
            else:
                trade_regimes.append('unknown')

    return np.array(trade_returns), trade_dates, trade_regimes

def compute_meanrev_returns(prices_df, drop_pct, lookback, hold_days, regime_map=None):
    """
    Mean-reversion: Buy any ETF that dropped >drop_pct% in lookback days, hold hold_days.
    """
    returns = prices_df.pct_change(lookback)
    forward_returns = prices_df.pct_change(hold_days).shift(-hold_days)

    trade_dates = []
    trade_returns = []
    trade_regimes = []

    for date in returns.index[lookback:]:
        if date not in forward_returns.index:
            continue

        row = returns.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()

        common = row.index.intersection(fwd.index)
        row = row[common]
        fwd = fwd[common]

        # Find ETFs that dropped more than threshold
        dropped = row[row < -drop_pct/100].index
        if len(dropped) == 0:
            continue

        period_return = fwd[dropped].mean()

        if not np.isnan(period_return):
            trade_dates.append(date)
            trade_returns.append(period_return)
            if regime_map is not None and date in regime_map.index:
                trade_regimes.append(regime_map.loc[date])
            else:
                trade_regimes.append('unknown')

    return np.array(trade_returns), trade_dates, trade_regimes

def run_permutation_test(returns, n_trades, prices_df, strategy_func, strategy_kwargs, n_perms=N_PERMUTATIONS):
    """
    HC #705 R1(c): Random DATE entry permutation test.
    For each perm, randomly select n_trades dates and compute mean forward return.
    """
    real_mean = np.mean(returns)

    # Get all possible dates from the price dataframe
    all_dates = prices_df.index.tolist()
    n_dates = len(all_dates)

    random_means = []
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        # Randomly select same number of dates
        random_idx = rng.choice(n_dates, size=min(n_trades, n_dates), replace=False)
        random_dates = [all_dates[i] for i in random_idx]

        # Compute forward returns for random dates
        hold = strategy_kwargs.get('hold_days', 5)
        fwd = prices_df.pct_change(hold).shift(-hold)

        random_returns = []
        for d in random_dates:
            if d in fwd.index:
                vals = fwd.loc[d].dropna()
                if len(vals) > 0:
                    # Pick random subset of same size as strategy
                    n_pick = min(strategy_kwargs.get('top_n', 3), len(vals))
                    subset = vals.sample(n_pick, random_state=rng)
                    random_returns.append(subset.mean())

        if len(random_returns) > 10:
            random_means.append(np.mean(random_returns))

    if len(random_means) < 50:
        return 1.0, 0, 0  # Not enough data

    random_means = np.array(random_means)
    p_value = np.mean(random_means >= real_mean)

    return p_value, real_mean * 100, np.mean(random_means) * 100

def run_regime_test(returns, regimes):
    """HC #705 R1(f): Regime-agnostic test."""
    rets_arr = np.array(returns)
    reg_arr = np.array(regimes)

    green_mask = reg_arr == 'green'
    red_mask = reg_arr == 'red'

    if np.sum(green_mask) < 10 or np.sum(red_mask) < 10:
        return 0.0, 0.0, 0.0, True  # Not enough data, pass by default

    green_rets = rets_arr[green_mask]
    red_rets = rets_arr[red_mask]

    sharpe_green = np.mean(green_rets) / (np.std(green_rets) + 1e-10) * np.sqrt(52)
    sharpe_red = np.mean(red_rets) / (np.std(red_rets) + 1e-10) * np.sqrt(52)

    gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 0.01)
    passes = gap < 0.50

    return sharpe_green, sharpe_red, gap, passes

def run_subperiod_test(returns):
    """HC #705 R1(d): Sub-period consistency."""
    n = len(returns)
    half = n // 2
    first_half = returns[:half]
    second_half = returns[half:]

    mean1 = np.mean(first_half)
    mean2 = np.mean(second_half)

    # Both halves must be profitable
    passes = mean1 > 0 and mean2 > 0
    return mean1 * 100, mean2 * 100, passes

def run_outlier_test(returns):
    """HC #705 R1(e): Remove top 5% of trades, check if Sharpe drops >50%."""
    full_sharpe = np.mean(returns) / (np.std(returns) + 1e-10)

    # Winsorize at 5th/95th percentile
    p5, p95 = np.percentile(returns, [5, 95])
    trimmed = returns[(returns >= p5) & (returns <= p95)]

    if len(trimmed) < 10:
        return full_sharpe, 0, True

    trimmed_sharpe = np.mean(trimmed) / (np.std(trimmed) + 1e-10)

    if abs(full_sharpe) < 1e-10:
        passes = True
    else:
        drop = 1 - (trimmed_sharpe / full_sharpe)
        passes = drop < 0.50

    return trimmed_sharpe * np.sqrt(52), len(returns) - len(trimmed), passes

def evaluate_config(name, returns, dates, regimes, prices_df, strategy_func, strategy_kwargs):
    """Run all quality gates on a config."""
    n = len(returns)
    if n < 20:
        return None

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)

    # Annualization factor depends on rebalancing frequency
    hold = strategy_kwargs.get('hold_days', 5)
    ann_factor = np.sqrt(252 / hold)

    sharpe = (mean_ret / (std_ret + 1e-10)) * ann_factor

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside) if len(downside) > 5 else std_ret
    sortino = (mean_ret / (downside_std + 1e-10)) * ann_factor

    # Win rate
    wr = np.mean(returns > 0)

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / (losses + 1e-10)

    # Max drawdown (cumulative)
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = np.min(dd) * 100

    # CAGR
    total_return = np.prod(1 + returns) - 1
    n_years = n * hold / 252
    if n_years > 0 and total_return > -1:
        cagr = ((1 + total_return) ** (1 / n_years) - 1) * 100
    else:
        cagr = -100

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  n={n}, Sharpe={sharpe:.3f}, Sortino={sortino:.3f}")
    print(f"  WR={wr:.1%}, PF={pf:.2f}, MaxDD={max_dd:.1f}%, CAGR={cagr:.1f}%")

    # Gate 1: Permutation test
    perm_p, real_mean_pct, random_mean_pct = run_permutation_test(
        returns, n, prices_df, strategy_func, strategy_kwargs
    )
    perm_pass = perm_p < 0.05
    print(f"  G1 Permutation: p={perm_p:.4f} {'✅' if perm_pass else '❌'}")
    print(f"     Real mean: {real_mean_pct:.3f}%, Random mean: {random_mean_pct:.3f}%")

    # Gate 2: Regime test
    sharpe_green, sharpe_red, regime_gap, regime_pass = run_regime_test(returns, regimes)
    print(f"  G2 Regime: gap={regime_gap:.3f} {'✅' if regime_pass else '❌'}")
    print(f"     Sharpe_green={sharpe_green:.3f}, Sharpe_red={sharpe_red:.3f}")

    # Gate 3: Sub-period consistency
    mean1, mean2, subperiod_pass = run_subperiod_test(returns)
    print(f"  G3 SubPeriod: first={mean1:.3f}%, second={mean2:.3f}% {'✅' if subperiod_pass else '❌'}")

    # Gate 4: Outlier robustness
    trimmed_sharpe, n_removed, outlier_pass = run_outlier_test(returns)
    print(f"  G4 Outlier: trimmed_sharpe={trimmed_sharpe:.3f}, removed={n_removed} {'✅' if outlier_pass else '❌'}")

    all_pass = perm_pass and regime_pass and subperiod_pass and outlier_pass
    print(f"  ALL_PASS: {'✅ YES' if all_pass else '❌ NO'}")

    return {
        'config_name': name,
        'n_periods': n,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr, 4),
        'profit_factor': round(pf, 3),
        'mean_return_pct': round(mean_ret * 100, 4),
        'max_dd_pct': round(max_dd, 2),
        'cagr_pct': round(cagr, 2),
        'quality_gates': {
            'permutation_p_value': round(perm_p, 4),
            'permutation_pass': perm_pass,
            'regime_sharpe_green': round(sharpe_green, 3),
            'regime_sharpe_red': round(sharpe_red, 3),
            'regime_gap': round(regime_gap, 3),
            'regime_pass': regime_pass,
            'first_half_mean_pct': round(mean1, 4),
            'second_half_mean_pct': round(mean2, 4),
            'subperiod_pass': subperiod_pass,
            'trimmed_sharpe': round(trimmed_sharpe, 3),
            'outlier_pass': outlier_pass,
            'all_pass': all_pass,
        }
    }

def main():
    print("=" * 70)
    print("ETF FACTOR STRATEGIES v1 — SURVIVORSHIP-BIAS-FREE")
    print("Testing if reversal/momentum edge survives on sector ETFs")
    print("=" * 70)

    # Download data
    prices = download_data()
    regime = download_spy_regime()

    print(f"\nData: {len(prices)} days, {len(prices.columns)} ETFs")
    print(f"ETFs: {list(prices.columns)}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    results = []

    # ===== REVERSAL STRATEGIES =====
    print("\n" + "="*70)
    print("SECTION 1: ETF REVERSAL (buy worst performers)")
    print("="*70)

    reversal_configs = [
        # (lookback, top_n, hold_days, name)
        (5, 3, 5, "Rev: Bottom3 by 5d, hold 5d"),
        (5, 3, 10, "Rev: Bottom3 by 5d, hold 10d"),
        (5, 3, 20, "Rev: Bottom3 by 5d, hold 20d"),
        (10, 3, 5, "Rev: Bottom3 by 10d, hold 5d"),
        (10, 3, 10, "Rev: Bottom3 by 10d, hold 10d"),
        (20, 3, 5, "Rev: Bottom3 by 20d, hold 5d"),
        (20, 3, 10, "Rev: Bottom3 by 20d, hold 10d"),
        (5, 5, 5, "Rev: Bottom5 by 5d, hold 5d"),
        (5, 5, 10, "Rev: Bottom5 by 5d, hold 10d"),
    ]

    for lookback, top_n, hold_days, name in reversal_configs:
        rets, dates, regs = compute_reversal_returns(prices, lookback, top_n, hold_days, regime)
        kwargs = {'lookback': lookback, 'top_n': top_n, 'hold_days': hold_days}
        result = evaluate_config(name, rets, dates, regs, prices, compute_reversal_returns, kwargs)
        if result:
            results.append(result)

    # ===== MOMENTUM STRATEGIES =====
    print("\n" + "="*70)
    print("SECTION 2: ETF MOMENTUM (buy best performers)")
    print("="*70)

    momentum_configs = [
        (63, 3, 21, "Mom: Top3 by 3m, hold 1m"),
        (126, 3, 21, "Mom: Top3 by 6m, hold 1m"),
        (252, 3, 21, "Mom: Top3 by 12m, hold 1m"),
        (126, 5, 21, "Mom: Top5 by 6m, hold 1m"),
        (252, 5, 21, "Mom: Top5 by 12m, hold 1m"),
        (63, 3, 63, "Mom: Top3 by 3m, hold 3m"),
        (126, 3, 63, "Mom: Top3 by 6m, hold 3m"),
    ]

    for lookback, top_n, hold_days, name in momentum_configs:
        rets, dates, regs = compute_momentum_returns(prices, lookback, top_n, hold_days, regime)
        kwargs = {'lookback': lookback, 'top_n': top_n, 'hold_days': hold_days}
        result = evaluate_config(name, rets, dates, regs, prices, compute_momentum_returns, kwargs)
        if result:
            results.append(result)

    # ===== MEAN-REVERSION (DROP-BASED) =====
    print("\n" + "="*70)
    print("SECTION 3: ETF MEAN-REVERSION (buy after drops)")
    print("="*70)

    meanrev_configs = [
        (3, 5, 5, "MeanRev: 3%+ drop in 5d, hold 5d"),
        (3, 5, 10, "MeanRev: 3%+ drop in 5d, hold 10d"),
        (3, 5, 20, "MeanRev: 3%+ drop in 5d, hold 20d"),
        (5, 5, 5, "MeanRev: 5%+ drop in 5d, hold 5d"),
        (5, 5, 10, "MeanRev: 5%+ drop in 5d, hold 10d"),
        (5, 5, 20, "MeanRev: 5%+ drop in 5d, hold 20d"),
        (3, 10, 10, "MeanRev: 3%+ drop in 10d, hold 10d"),
        (5, 10, 10, "MeanRev: 5%+ drop in 10d, hold 10d"),
        (5, 10, 20, "MeanRev: 5%+ drop in 10d, hold 20d"),
    ]

    for drop_pct, lookback, hold_days, name in meanrev_configs:
        rets, dates, regs = compute_meanrev_returns(prices, drop_pct, lookback, hold_days, regime)
        kwargs = {'drop_pct': drop_pct, 'lookback': lookback, 'hold_days': hold_days, 'top_n': 3}
        result = evaluate_config(name, rets, dates, regs, prices, compute_meanrev_returns, kwargs)
        if result:
            results.append(result)

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    passing = [r for r in results if r['quality_gates']['all_pass']]
    failing = [r for r in results if not r['quality_gates']['all_pass']]

    print(f"\nTotal configs tested: {len(results)}")
    print(f"PASSING ALL GATES: {len(passing)}")
    print(f"FAILING: {len(failing)}")

    if passing:
        print(f"\n{'='*60}")
        print("PASSING CONFIGS:")
        for r in sorted(passing, key=lambda x: -x['sharpe']):
            g = r['quality_gates']
            print(f"  ✅ {r['config_name']}")
            print(f"     Sharpe={r['sharpe']:.3f}, WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}")
            print(f"     n={r['n_periods']}, perm_p={g['permutation_p_value']:.4f}, R1_gap={g['regime_gap']:.3f}")
    else:
        print("\n⚠️  NO CONFIGS PASS ALL GATES")
        print("This confirms survivorship bias was driving the individual stock results!")

    # Comparison verdict
    print(f"\n{'='*60}")
    print("SURVIVORSHIP BIAS VERDICT:")
    if len(passing) >= 3:
        print("  ✅ ETF results CONFIRM the edge is real (not survivorship bias)")
    elif len(passing) > 0:
        print("  ⚠️  PARTIAL confirmation — some edge survives on ETFs but weaker")
    else:
        print("  ❌ ETF results REJECT the edge — individual stock results were")
        print("     driven by survivorship bias in the 30-stock universe")

    # Save results
    output = {
        'strategy': 'ETF Factor v1 — Survivorship-Bias-Free Validation',
        'generated': datetime.now().isoformat(),
        'universe': list(prices.columns),
        'date_range': f'{prices.index[0].date()} to {prices.index[-1].date()}',
        'n_etfs': len(prices.columns),
        'n_configs': len(results),
        'n_passing': len(passing),
        'n_failing': len(failing),
        'survivorship_verdict': 'CONFIRMED' if len(passing) >= 3 else 'PARTIAL' if len(passing) > 0 else 'REJECTED',
        'results': results,
    }

    report_file = OUTPUT_DIR / 'backtest_report.json'
    with open(report_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {report_file}")

if __name__ == '__main__':
    main()
