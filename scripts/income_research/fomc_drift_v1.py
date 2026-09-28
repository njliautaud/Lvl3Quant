#!/usr/bin/env python3
"""
Pre-FOMC Announcement Drift Backtest v1
========================================
Tests the well-documented anomaly (Lucca & Moench, 2015) that SPY/equities
drift upward 1-3 days BEFORE FOMC rate decisions.

Adversarial checks BUILT IN (HC #705):
  1. Permutation test (random DATE entry, not return shuffling)
  2. Per-year breakdown (not just aggregate)
  3. Drawdown analysis
  4. Transaction cost sensitivity
  5. Comparison across tickers (market-wide vs SPY-specific)
  6. Out-of-sample split (train 2019-2023, OOS 2024-2026)

Output: /home/jupiter/Lvl3Quant/output/fomc_drift_v1/
"""

import os
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/fomc_drift_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10_000
TRADE_SIZE = 200  # per trade
N_PERMUTATIONS = 10_000

TICKERS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'GS']

# All FOMC meeting dates (announcement day)
FOMC_DATES = sorted([
    # 2019
    '2019-01-30','2019-03-20','2019-05-01','2019-06-19','2019-07-31','2019-09-18','2019-10-30','2019-12-11',
    # 2020
    '2020-01-29','2020-03-03','2020-03-15','2020-03-23','2020-04-29','2020-06-10','2020-07-29','2020-09-16','2020-11-05','2020-12-16',
    # 2021
    '2021-01-27','2021-03-17','2021-04-28','2021-06-16','2021-07-28','2021-09-22','2021-11-03','2021-12-15',
    # 2022
    '2022-01-26','2022-03-16','2022-05-04','2022-06-15','2022-07-27','2022-09-21','2022-11-02','2022-12-14',
    # 2023
    '2023-02-01','2023-03-22','2023-05-03','2023-06-14','2023-07-26','2023-09-20','2023-11-01','2023-12-13',
    # 2024
    '2024-01-31','2024-03-20','2024-05-01','2024-06-12','2024-07-31','2024-09-18','2024-11-07','2024-12-18',
    # 2025
    '2025-01-29','2025-03-19','2025-05-07','2025-06-18','2025-07-30','2025-09-17','2025-10-29','2025-12-17',
    # 2026
    '2026-01-28','2026-03-18','2026-04-29','2026-06-17','2026-07-29',
])

# Entry rules: (name, entry_offset, exit_offset)
# offset is trading days relative to FOMC day (0 = FOMC day)
# entry_offset: buy at close N days before FOMC
# exit_offset:  sell at close on this day
ENTRY_RULES = [
    ('Pre-FOMC 1d', -1, 0),   # buy T-1 close, sell FOMC close
    ('Pre-FOMC 2d', -2, 0),   # buy T-2 close, sell FOMC close
    ('Pre-FOMC 3d', -3, 0),   # buy T-3 close, sell FOMC close
    ('Post-FOMC 1d', 0, 1),   # buy FOMC close, sell T+1 close
]

# Transaction cost (round-trip) as fraction of notional
# For a $200 trade on Robinhood: $0 commission but ~0.01-0.03% spread
# For options: wider spreads. We test multiple levels.
COST_LEVELS_BPS = [0, 5, 10, 20, 50]  # basis points round-trip


def download_data(tickers, start='2018-12-01', end='2026-07-15'):
    """Download daily close prices for all tickers."""
    cache_file = OUTPUT_DIR / 'price_cache.parquet'
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        # Check if we have all tickers
        missing = [t for t in tickers if t not in df.columns]
        if not missing:
            print(f"  Loaded cached prices: {df.shape}")
            return df

    print(f"  Downloading {len(tickers)} tickers from yfinance...")
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    df.to_parquet(cache_file)
    print(f"  Downloaded: {df.shape}")
    return df


def get_trading_day_offset(dates_index, target_date, offset):
    """Get the trading day that is `offset` trading days from target_date."""
    target = pd.Timestamp(target_date)
    # Find nearest trading day to target
    idx = dates_index.searchsorted(target)
    if idx >= len(dates_index):
        return None
    # If target isn't a trading day, use the nearest one at or after
    if dates_index[idx] != target:
        # Try the day before too
        if idx > 0 and abs((dates_index[idx-1] - target).days) <= abs((dates_index[idx] - target).days):
            idx = idx - 1

    result_idx = idx + offset
    if result_idx < 0 or result_idx >= len(dates_index):
        return None
    return dates_index[result_idx]


def compute_fomc_returns(prices, fomc_dates, entry_offset, exit_offset):
    """
    Compute returns for a given entry/exit rule around FOMC dates.
    Returns DataFrame with columns: fomc_date, entry_date, exit_date, entry_price, exit_price, return_pct
    """
    dates_index = prices.index
    records = []

    for fomc_str in fomc_dates:
        fomc_date = pd.Timestamp(fomc_str)

        entry_date = get_trading_day_offset(dates_index, fomc_date, entry_offset)
        exit_date = get_trading_day_offset(dates_index, fomc_date, exit_offset)

        if entry_date is None or exit_date is None:
            continue
        if entry_date not in prices.index or exit_date not in prices.index:
            continue

        entry_price = prices.loc[entry_date]
        exit_price = prices.loc[exit_date]

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
            continue

        ret = (exit_price - entry_price) / entry_price
        records.append({
            'fomc_date': fomc_str,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return_pct': ret,
            'year': fomc_date.year,
        })

    return pd.DataFrame(records)


def permutation_test_random_dates(prices, observed_returns, n_fomc_events, hold_days, n_perms=10000):
    """
    Permutation test using RANDOM DATE ENTRY (not return shuffling).
    For each permutation:
      - Randomly pick n_fomc_events entry dates from the full trading calendar
      - Compute forward returns over hold_days
      - Compare mean to observed mean

    This is the correct test: FOMC dates are specific calendar events.
    Random dates shouldn't capture the same drift if the effect is real.
    """
    all_dates = prices.index
    n_dates = len(all_dates)
    observed_mean = observed_returns.mean()

    # Pre-compute all forward returns for efficiency
    # For each date i, return from date i to date i+hold_days
    fwd_returns = np.full(n_dates, np.nan)
    for i in range(n_dates - hold_days):
        if prices.iloc[i] > 0:
            fwd_returns[i] = (prices.iloc[i + hold_days] - prices.iloc[i]) / prices.iloc[i]

    valid_mask = ~np.isnan(fwd_returns)
    valid_indices = np.where(valid_mask)[0]
    valid_returns = fwd_returns[valid_mask]

    if len(valid_indices) < n_fomc_events:
        return np.nan, np.nan

    # Run permutations
    rng = np.random.default_rng(42)
    perm_means = np.empty(n_perms)

    for p in range(n_perms):
        sample_idx = rng.choice(len(valid_returns), size=n_fomc_events, replace=False)
        perm_means[p] = valid_returns[sample_idx].mean()

    # p-value: fraction of random date selections that beat observed
    p_value = np.mean(perm_means >= observed_mean)

    return p_value, perm_means


def simulate_equity_curve(returns_series, initial_capital, trade_size, cost_bps=0):
    """
    Simulate equity curve with fixed $trade_size per trade.
    Returns equity curve array.
    """
    equity = initial_capital
    curve = [equity]
    cost_frac = cost_bps / 10000.0

    for ret in returns_series:
        # Number of shares (fractional OK for backtest)
        shares_value = min(trade_size, equity)  # Can't bet more than we have
        if shares_value <= 0:
            curve.append(equity)
            continue

        pnl = shares_value * (ret - cost_frac)
        equity += pnl
        curve.append(equity)

    return np.array(curve)


def compute_metrics(returns, cost_bps=0):
    """Compute strategy metrics from return series."""
    if len(returns) == 0:
        return {}

    cost_frac = cost_bps / 10000.0
    net_returns = returns - cost_frac

    mean_ret = net_returns.mean()
    std_ret = net_returns.std()

    # Annualize: ~8 FOMC meetings per year
    ann_factor = 8
    ann_return = mean_ret * ann_factor
    ann_vol = std_ret * np.sqrt(ann_factor)

    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = net_returns[net_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = ann_return / (downside_std * np.sqrt(ann_factor)) if downside_std > 0 else 0

    # Win rate
    wr = (net_returns > 0).mean()

    # Profit factor
    gains = net_returns[net_returns > 0].sum()
    losses = abs(net_returns[net_returns < 0].sum())
    pf = gains / losses if losses > 0 else np.inf

    # Max drawdown
    equity = (1 + net_returns).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # t-stat
    t_stat = mean_ret / (std_ret / np.sqrt(len(returns))) if std_ret > 0 else 0

    return {
        'n_trades': len(returns),
        'mean_ret_bps': mean_ret * 10000,
        'std_ret_bps': std_ret * 10000,
        'win_rate': wr,
        'profit_factor': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd_pct': max_dd * 100,
        't_stat': t_stat,
        'total_return_pct': ((1 + net_returns).prod() - 1) * 100,
    }


def run_backtest():
    """Main backtest runner with all adversarial checks."""
    print("=" * 80)
    print("PRE-FOMC ANNOUNCEMENT DRIFT BACKTEST v1")
    print("=" * 80)

    # ─── 1. DOWNLOAD DATA ────────────────────────────────────────────────
    print("\n[1/7] Downloading price data...")
    prices_df = download_data(TICKERS)

    # Filter to only dates we need (some buffer before first FOMC)
    results = {}
    all_summaries = []

    # ─── 2. MAIN BACKTEST: ALL TICKERS x ALL ENTRY RULES ────────────────
    print("\n[2/7] Computing FOMC returns for all tickers and entry rules...")

    for ticker in TICKERS:
        if ticker not in prices_df.columns:
            print(f"  WARNING: {ticker} not in data, skipping")
            continue

        prices = prices_df[ticker].dropna()
        if len(prices) < 100:
            print(f"  WARNING: {ticker} has only {len(prices)} prices, skipping")
            continue

        results[ticker] = {}

        for rule_name, entry_off, exit_off in ENTRY_RULES:
            trades_df = compute_fomc_returns(prices, FOMC_DATES, entry_off, exit_off)

            if len(trades_df) < 5:
                continue

            returns = trades_df['return_pct'].values
            hold_days = abs(exit_off - entry_off)

            # Base metrics (no cost)
            metrics = compute_metrics(pd.Series(returns), cost_bps=0)

            # Permutation test (random date entry)
            p_value, perm_means = permutation_test_random_dates(
                prices, returns, len(returns), hold_days, n_perms=N_PERMUTATIONS
            )
            metrics['perm_p_value'] = p_value

            # Per-year breakdown
            year_metrics = {}
            for year in sorted(trades_df['year'].unique()):
                yr_rets = trades_df[trades_df['year'] == year]['return_pct'].values
                if len(yr_rets) >= 2:
                    year_metrics[int(year)] = {
                        'n': len(yr_rets),
                        'mean_bps': float(yr_rets.mean() * 10000),
                        'wr': float((yr_rets > 0).mean()),
                    }
            metrics['per_year'] = year_metrics

            # Cost sensitivity
            cost_metrics = {}
            for cost in COST_LEVELS_BPS:
                cm = compute_metrics(pd.Series(returns), cost_bps=cost)
                cost_metrics[cost] = {
                    'sharpe': cm['sharpe'],
                    'total_return_pct': cm['total_return_pct'],
                    'win_rate': cm['win_rate'],
                }
            metrics['cost_sensitivity'] = cost_metrics

            # OOS split: train 2019-2023, OOS 2024-2026
            is_mask = trades_df['year'] <= 2023
            oos_mask = trades_df['year'] >= 2024

            is_rets = trades_df[is_mask]['return_pct'].values
            oos_rets = trades_df[oos_mask]['return_pct'].values

            if len(is_rets) >= 3:
                metrics['in_sample'] = compute_metrics(pd.Series(is_rets))
                metrics['in_sample']['n_trades'] = len(is_rets)
            if len(oos_rets) >= 3:
                metrics['out_of_sample'] = compute_metrics(pd.Series(oos_rets))
                metrics['out_of_sample']['n_trades'] = len(oos_rets)

            results[ticker][rule_name] = metrics

            # Summary row
            all_summaries.append({
                'ticker': ticker,
                'rule': rule_name,
                'n_trades': metrics['n_trades'],
                'mean_ret_bps': metrics['mean_ret_bps'],
                'win_rate': metrics['win_rate'],
                'sharpe': metrics['sharpe'],
                'sortino': metrics['sortino'],
                'profit_factor': metrics['profit_factor'],
                'max_dd_pct': metrics['max_dd_pct'],
                't_stat': metrics['t_stat'],
                'perm_p_value': p_value,
            })

    summary_df = pd.DataFrame(all_summaries)

    # ─── 3. PRINT RESULTS ────────────────────────────────────────────────
    print("\n[3/7] Results Summary")
    print("=" * 120)
    print(f"{'Ticker':<8} {'Rule':<16} {'N':>4} {'Mean(bps)':>10} {'WR':>6} {'Sharpe':>8} {'Sortino':>8} "
          f"{'PF':>6} {'MaxDD%':>8} {'t-stat':>7} {'Perm-p':>8}")
    print("-" * 120)

    for _, row in summary_df.sort_values(['ticker', 'rule']).iterrows():
        flag = " ***" if row['perm_p_value'] < 0.05 else ""
        pf_str = f"{row['profit_factor']:.2f}" if row['profit_factor'] < 100 else "inf"
        print(f"{row['ticker']:<8} {row['rule']:<16} {row['n_trades']:>4} {row['mean_ret_bps']:>10.1f} "
              f"{row['win_rate']:>6.1%} {row['sharpe']:>8.2f} {row['sortino']:>8.2f} "
              f"{pf_str:>6} {row['max_dd_pct']:>7.1f}% {row['t_stat']:>7.2f} {row['perm_p_value']:>8.4f}{flag}")

    # ─── 4. ADVERSARIAL CHECK: PERMUTATION TEST DETAIL ───────────────────
    print("\n[4/7] Permutation Test Results (p < 0.05 = survives)")
    print("-" * 80)
    sig_count = 0
    total_tests = 0
    for _, row in summary_df.iterrows():
        total_tests += 1
        if row['perm_p_value'] < 0.05:
            sig_count += 1
            print(f"  PASS: {row['ticker']} / {row['rule']}: p={row['perm_p_value']:.4f}")

    fail_count = total_tests - sig_count
    print(f"\n  {sig_count}/{total_tests} pass permutation test at p<0.05")
    print(f"  Expected by chance at 5%: ~{total_tests * 0.05:.1f}")
    print(f"  {'ANOMALY LIKELY REAL' if sig_count > total_tests * 0.10 else 'ANOMALY QUESTIONABLE — few survive permutation'}")

    # ─── 5. ADVERSARIAL CHECK: PER-YEAR CONSISTENCY ──────────────────────
    print("\n[5/7] Per-Year Consistency (SPY, Pre-FOMC 2d)")
    print("-" * 60)

    spy_2d = results.get('SPY', {}).get('Pre-FOMC 2d', {})
    if 'per_year' in spy_2d:
        pos_years = 0
        total_years = 0
        for year, ym in sorted(spy_2d['per_year'].items()):
            total_years += 1
            if ym['mean_bps'] > 0:
                pos_years += 1
            marker = "+" if ym['mean_bps'] > 0 else "-"
            print(f"  {year}: {ym['n']} trades, mean={ym['mean_bps']:+.1f} bps, WR={ym['wr']:.0%} [{marker}]")

        consistency = pos_years / total_years if total_years > 0 else 0
        print(f"\n  Positive years: {pos_years}/{total_years} ({consistency:.0%})")
        if consistency < 0.60:
            print("  WARNING: Less than 60% of years positive — WEAK consistency")

    # ─── 6. ADVERSARIAL CHECK: IN-SAMPLE vs OOS ─────────────────────────
    print("\n[6/7] In-Sample (2019-2023) vs Out-of-Sample (2024-2026)")
    print("-" * 80)
    print(f"{'Ticker':<8} {'Rule':<16} {'IS Mean(bps)':>12} {'IS WR':>6} {'OOS Mean(bps)':>14} {'OOS WR':>7} {'Decay?':>8}")
    print("-" * 80)

    for ticker in ['SPY', 'QQQ']:
        for rule_name, _, _ in ENTRY_RULES:
            m = results.get(ticker, {}).get(rule_name, {})
            is_m = m.get('in_sample', {})
            oos_m = m.get('out_of_sample', {})
            if is_m and oos_m:
                is_mean = is_m.get('mean_ret_bps', 0)
                oos_mean = oos_m.get('mean_ret_bps', 0)
                decay = "YES" if oos_mean < is_mean * 0.5 else "no"
                if oos_mean < 0:
                    decay = "DEAD"
                print(f"{ticker:<8} {rule_name:<16} {is_mean:>12.1f} {is_m.get('win_rate',0):>6.1%} "
                      f"{oos_mean:>14.1f} {oos_m.get('win_rate',0):>7.1%} {decay:>8}")

    # ─── 7. COST SENSITIVITY ────────────────────────────────────────────
    print("\n[7/7] Cost Sensitivity (SPY, Pre-FOMC 2d)")
    print("-" * 60)

    if 'cost_sensitivity' in spy_2d:
        for cost, cm in sorted(spy_2d['cost_sensitivity'].items()):
            print(f"  Cost={cost:>3} bps: Sharpe={cm['sharpe']:>6.2f}, "
                  f"Total Return={cm['total_return_pct']:>6.1f}%, WR={cm['win_rate']:>5.1%}")

    # ─── 8. EQUITY CURVES (SPY) ─────────────────────────────────────────
    print("\n\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pre-FOMC Announcement Drift — SPY Backtest', fontsize=16, fontweight='bold')

    spy_prices = prices_df['SPY'].dropna()

    for idx, (rule_name, entry_off, exit_off) in enumerate(ENTRY_RULES):
        ax = axes[idx // 2][idx % 2]
        trades_df = compute_fomc_returns(spy_prices, FOMC_DATES, entry_off, exit_off)

        if len(trades_df) == 0:
            ax.set_title(f'{rule_name}: No data')
            continue

        returns = trades_df['return_pct'].values

        # Equity curve at 0 cost
        eq = simulate_equity_curve(returns, INITIAL_CAPITAL, TRADE_SIZE, cost_bps=0)
        ax.plot(range(len(eq)), eq, 'b-', linewidth=1.5, label='0 bps cost')

        # Equity curve at 20 bps cost
        eq20 = simulate_equity_curve(returns, INITIAL_CAPITAL, TRADE_SIZE, cost_bps=20)
        ax.plot(range(len(eq20)), eq20, 'r--', linewidth=1.5, label='20 bps cost')

        ax.axhline(y=INITIAL_CAPITAL, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'{rule_name} (n={len(returns)})')
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Equity ($)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spy_equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 9. PERMUTATION DISTRIBUTION PLOT ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Permutation Test: Observed vs Random Date Returns (SPY)', fontsize=16, fontweight='bold')

    for idx, (rule_name, entry_off, exit_off) in enumerate(ENTRY_RULES):
        ax = axes[idx // 2][idx % 2]
        trades_df = compute_fomc_returns(spy_prices, FOMC_DATES, entry_off, exit_off)

        if len(trades_df) == 0:
            continue

        returns = trades_df['return_pct'].values
        hold_days = abs(exit_off - entry_off)

        p_val, perm_means = permutation_test_random_dates(
            spy_prices, returns, len(returns), hold_days, n_perms=N_PERMUTATIONS
        )

        if perm_means is not None and not np.isnan(p_val):
            ax.hist(perm_means * 10000, bins=80, density=True, alpha=0.7, color='steelblue',
                    label='Random dates')
            obs_mean = returns.mean() * 10000
            ax.axvline(obs_mean, color='red', linewidth=2, linestyle='--',
                      label=f'Observed: {obs_mean:.1f} bps')
            ax.set_title(f'{rule_name} (p={p_val:.4f})')
            ax.set_xlabel('Mean Return (bps)')
            ax.set_ylabel('Density')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'permutation_tests.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 10. CROSS-TICKER HEATMAP ────────────────────────────────────────
    # Create heatmap of mean returns across tickers and rules
    heatmap_data = []
    for ticker in TICKERS:
        row = {'ticker': ticker}
        for rule_name, _, _ in ENTRY_RULES:
            m = results.get(ticker, {}).get(rule_name, {})
            row[rule_name] = m.get('mean_ret_bps', np.nan)
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data).set_index('ticker')

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_df.values, cmap='RdYlGn', aspect='auto')

    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    # Add text annotations
    for i in range(len(heatmap_df.index)):
        for j in range(len(heatmap_df.columns)):
            val = heatmap_df.values[i, j]
            if not np.isnan(val):
                color = 'black' if abs(val) < 30 else 'white'
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9, color=color)

    ax.set_title('Pre-FOMC Mean Returns by Ticker & Rule (bps)', fontsize=14, fontweight='bold')
    plt.colorbar(im, label='Mean Return (bps)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cross_ticker_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    # ─── 11. SAVE FULL RESULTS ───────────────────────────────────────────
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    # Clean results for JSON
    clean_results = {}
    for ticker, ticker_results in results.items():
        clean_results[ticker] = {}
        for rule, metrics in ticker_results.items():
            clean = {}
            for k, v in metrics.items():
                if k == 'per_year':
                    clean[k] = v
                elif k == 'cost_sensitivity':
                    clean[k] = {str(kk): vv for kk, vv in v.items()}
                elif k in ('in_sample', 'out_of_sample'):
                    clean[k] = {kk: convert(vv) for kk, vv in v.items()}
                else:
                    clean[k] = convert(v)
            clean_results[ticker][rule] = clean

    with open(OUTPUT_DIR / 'full_results.json', 'w') as f:
        json.dump(clean_results, f, indent=2, default=convert)

    summary_df.to_csv(OUTPUT_DIR / 'summary.csv', index=False)

    # ─── 12. OPTIONS P&L SIMULATION ─────────────────────────────────────
    # Simulate call debit spread P&L for the best-performing rule
    print("\n\n" + "=" * 80)
    print("OPTIONS SIMULATION: Call Debit Spread (SPY)")
    print("=" * 80)

    # Find best rule for SPY
    spy_results = {r: m for r, m in results.get('SPY', {}).items()}
    if spy_results:
        best_rule = max(spy_results.items(), key=lambda x: x[1].get('sharpe', 0))
        best_name = best_rule[0]
        best_metrics = best_rule[1]

        print(f"\nBest rule: {best_name}")
        print(f"  Mean return: {best_metrics['mean_ret_bps']:.1f} bps per trade")
        print(f"  Win rate: {best_metrics['win_rate']:.1%}")
        print(f"  Sharpe: {best_metrics['sharpe']:.2f}")

        # For a call debit spread:
        # Max profit = (spread width - debit paid)
        # Max loss = debit paid
        # Typical ATM / ATM+2% spread on SPY ~5-8 points wide
        # Cost: ~$150-300 for near-expiry
        # If SPY moves up by the mean return, estimate P&L

        spy_price = spy_prices.iloc[-1] if len(spy_prices) > 0 else 500
        spread_width_pct = 0.02  # 2% OTM
        spread_width_pts = spy_price * spread_width_pct

        # Rough options pricing for 1-3 day hold
        # ATM call cost ~1-2% of SPY for weekly, short leg offsets
        # Net debit for ATM/ATM+2% spread: ~$200-400
        debit_per_spread = 200  # ~$200 for the spread
        max_profit = (spread_width_pts * 100) - debit_per_spread  # per contract, 100 multiplier

        print(f"\n  Call Debit Spread Setup (rough estimate):")
        print(f"    SPY current: ${spy_price:.0f}")
        print(f"    Spread: ATM / ATM+{spread_width_pct:.0%} ({spread_width_pts:.0f} pts wide)")
        print(f"    Max debit: ~${debit_per_spread}")
        print(f"    Max profit: ~${max_profit:.0f}")
        print(f"    Trade size: ${TRADE_SIZE} (1 spread)")

        # Simulate with $440 account
        print(f"\n  $440 Account Simulation:")
        print(f"    Trades per year: ~8 (FOMC meetings)")

        # For each trade, if SPY move > 0, profit is proportional
        # Debit spread has non-linear payoff, but for small moves:
        # P&L ≈ delta * move * 100 - theta_decay
        # With 1-2 day hold and ATM spread, delta ~0.3-0.5 on net
        net_delta = 0.35
        avg_move_pts = best_metrics['mean_ret_bps'] / 10000 * spy_price
        avg_option_pnl = net_delta * avg_move_pts * 100  # per contract

        print(f"    Avg SPY move: {avg_move_pts:.2f} pts ({best_metrics['mean_ret_bps']:.0f} bps)")
        print(f"    Avg option P&L (est): ${avg_option_pnl:.0f} per contract")
        print(f"    Theta cost (1-2d): ~${debit_per_spread * 0.15:.0f}")

        net_per_trade = avg_option_pnl - debit_per_spread * 0.15  # rough theta
        print(f"    Net per trade (est): ${net_per_trade:.0f}")
        print(f"    Annual est P&L (8 trades): ${net_per_trade * 8:.0f}")
        print(f"    Annual ROI on $440: {net_per_trade * 8 / 440:.0%}")

        # Reality check
        print(f"\n  REALITY CHECK:")
        print(f"    - Options spreads have WIDE bid-ask (~$0.10-0.30 per leg)")
        print(f"    - Theta decay eats 10-20% of premium per day on weeklies")
        print(f"    - With $440 account, max 1 spread per FOMC = ~$200 risk")
        print(f"    - Win rate of {best_metrics['win_rate']:.0%} means ~{(1-best_metrics['win_rate'])*8:.0f} losers per year")
        print(f"    - Each loser costs full debit (~$200)")
        if best_metrics['win_rate'] < 0.55:
            print(f"    - WARNING: Win rate below 55% makes options strategy UNPROFITABLE")
            print(f"      (options need higher WR than stock due to non-linear payoff)")

    # ─── 13. FINAL VERDICT ───────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Check SPY Pre-FOMC 2d (the classic anomaly)
    spy_2d_perm = results.get('SPY', {}).get('Pre-FOMC 2d', {}).get('perm_p_value', 1.0)
    spy_2d_sharpe = results.get('SPY', {}).get('Pre-FOMC 2d', {}).get('sharpe', 0)
    spy_2d_oos = results.get('SPY', {}).get('Pre-FOMC 2d', {}).get('out_of_sample', {})

    checks = {
        'permutation_survives': spy_2d_perm < 0.05,
        'positive_sharpe': spy_2d_sharpe > 0,
        'oos_positive': spy_2d_oos.get('mean_ret_bps', 0) > 0 if spy_2d_oos else False,
        'oos_wr_above_50': spy_2d_oos.get('win_rate', 0) > 0.50 if spy_2d_oos else False,
        'cross_ticker_consistent': sum(1 for t in ['SPY', 'QQQ', 'AAPL', 'MSFT']
                                       if results.get(t, {}).get('Pre-FOMC 2d', {}).get('mean_ret_bps', 0) > 0) >= 3,
    }

    print("\nAdversarial Checks (SPY Pre-FOMC 2d):")
    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check}")

    n_sig = sum(1 for _, row in summary_df.iterrows() if row['perm_p_value'] < 0.05)
    print(f"\n  Total ticker/rule combos surviving permutation: {n_sig}/{len(summary_df)}")

    if all_pass:
        print("\n  VERDICT: Pre-FOMC drift appears REAL and tradeable.")
        print("  HOWEVER: Options execution on $440 account is challenging due to")
        print("  wide spreads, theta decay, and small edge magnitude.")
        print("  RECOMMENDATION: Trade SPY shares (fractional) if possible,")
        print("  or use tight call debit spreads only when edge is strongest.")
    elif checks['permutation_survives']:
        print("\n  VERDICT: Statistical edge EXISTS but may not be practically tradeable.")
        print("  Some adversarial checks failed — edge may be decaying or inconsistent.")
    else:
        print("\n  VERDICT: Pre-FOMC drift does NOT survive rigorous permutation testing.")
        print("  The anomaly may have been arbitraged away or was never as strong as reported.")
        print("  DO NOT TRADE this strategy.")

    print(f"\n  Output saved to: {OUTPUT_DIR}")
    print(f"  Files: summary.csv, full_results.json, spy_equity_curves.png,")
    print(f"         permutation_tests.png, cross_ticker_heatmap.png")

    return results, summary_df


if __name__ == '__main__':
    results, summary = run_backtest()
