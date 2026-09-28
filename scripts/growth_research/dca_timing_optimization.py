#!/usr/bin/env python3
"""
DCA Timing Optimization Study
==============================
Tests whether timing dollar-cost averaging entries can improve returns
vs fixed-schedule DCA for UPRO (3x S&P 500 ETF) and SPY.

Strategies tested:
  1. Fixed DCA (baseline) - $100/week every Friday
  2. Value DCA - invest more when price < SMA(50), less when above
  3. VIX-triggered DCA - accumulate cash, deploy on VIX spikes
  4. Drawdown DCA - double down when UPRO is down >10% from 20-day high
  5. Momentum DCA - invest more during confirmed uptrends (SPY > SMA200)
  6. Combined - VIX + trend signals determine accumulate vs deploy

Methodology:
  - Weekly DCA, $100/week baseline ($5,200/year)
  - Walk-forward: 2013-2019 (in-sample) and 2020-2026 (out-of-sample)
  - Permutation test: 100 shuffles of timing signal, p < 0.05 required
  - Metrics: IRR, total return, Sharpe, MaxDD, avg cost basis
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/dca_timing'
WEEKLY_BUDGET = 100.0  # $100/week baseline
ANNUAL_BUDGET = WEEKLY_BUDGET * 52  # $5,200/year

# ─────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────

def download_data():
    """Download SPY, UPRO, ^VIX from 2013 to present."""
    cache_path = os.path.join(OUTPUT_DIR, '_data_cache.parquet')
    if os.path.exists(cache_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if (datetime.now() - mod_time).days < 1:
            print("Using cached data...")
            return pd.read_parquet(cache_path)

    print("Downloading data from yfinance...")
    tickers = ['SPY', 'UPRO', '^VIX']
    data = {}
    for t in tickers:
        df = yf.download(t, start='2012-12-01', end=datetime.now().strftime('%Y-%m-%d'),
                         progress=False, auto_adjust=True)
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df

    # Build combined dataframe
    combined = pd.DataFrame(index=data['SPY'].index)
    combined['spy_close'] = data['SPY']['Close']
    combined['upro_close'] = data['UPRO']['Close']
    combined['vix_close'] = data['^VIX']['Close']
    combined = combined.dropna()

    # Compute features
    combined['spy_sma50'] = combined['spy_close'].rolling(50).mean()
    combined['spy_sma200'] = combined['spy_close'].rolling(200).mean()
    combined['upro_high20'] = combined['upro_close'].rolling(20).max()
    combined['upro_dd_from_high'] = (combined['upro_close'] / combined['upro_high20'] - 1) * 100

    combined = combined.dropna()
    combined.to_parquet(cache_path)
    print(f"Data: {combined.index[0].date()} to {combined.index[-1].date()}, {len(combined)} days")
    return combined


def get_weekly_dates(data, day_of_week=4):
    """Get weekly rebalance dates (default: Friday=4)."""
    data_copy = data.copy()
    data_copy['dow'] = data_copy.index.dayofweek
    # Pick one day per week — the last trading day of each week
    data_copy['year_week'] = data_copy.index.isocalendar().year.astype(str) + '-' + \
                              data_copy.index.isocalendar().week.astype(str).str.zfill(2)
    weekly = data_copy.groupby('year_week').last()
    weekly.index = data_copy.groupby('year_week').apply(lambda x: x.index[-1])
    return weekly.index.sort_values()


# ─────────────────────────────────────────────────────────────────────
# DCA STRATEGIES
# ─────────────────────────────────────────────────────────────────────

def run_fixed_dca(data, weekly_dates, ticker='upro_close'):
    """Baseline: invest exactly $100 every week."""
    units = 0.0
    total_invested = 0.0
    equity_curve = []

    for dt in weekly_dates:
        if dt not in data.index:
            continue
        price = data.loc[dt, ticker]
        invest_amount = WEEKLY_BUDGET
        units += invest_amount / price
        total_invested += invest_amount
        equity_curve.append({
            'date': dt, 'units': units, 'invested': total_invested,
            'portfolio_value': units * price, 'price': price,
            'invest_this_week': invest_amount
        })

    return pd.DataFrame(equity_curve).set_index('date')


def run_value_dca(data, weekly_dates, ticker='upro_close'):
    """Invest $150 when price < SMA50, $50 when above. Same annual total."""
    units = 0.0
    total_invested = 0.0
    equity_curve = []

    for dt in weekly_dates:
        if dt not in data.index:
            continue
        row = data.loc[dt]
        price = row[ticker]
        # Use SPY's SMA50 as the signal (more stable than UPRO's)
        below_sma = row['spy_close'] < row['spy_sma50']
        invest_amount = 150.0 if below_sma else 50.0
        units += invest_amount / price
        total_invested += invest_amount
        equity_curve.append({
            'date': dt, 'units': units, 'invested': total_invested,
            'portfolio_value': units * price, 'price': price,
            'invest_this_week': invest_amount
        })

    return pd.DataFrame(equity_curve).set_index('date')


def run_vix_dca(data, weekly_dates, ticker='upro_close'):
    """Accumulate cash when VIX < 20, deploy entire pile when VIX > 25."""
    units = 0.0
    total_invested = 0.0
    cash_pile = 0.0
    equity_curve = []

    for dt in weekly_dates:
        if dt not in data.index:
            continue
        row = data.loc[dt]
        price = row[ticker]
        vix = row['vix_close']

        cash_pile += WEEKLY_BUDGET  # Always add weekly budget to pile

        if vix > 25:
            # Deploy entire cash pile
            invest_amount = cash_pile
            cash_pile = 0.0
        elif vix < 20:
            # Save — invest only a small amount to avoid missing rallies entirely
            invest_amount = min(25.0, cash_pile)
            cash_pile -= invest_amount
        else:
            # Normal zone: invest normal amount
            invest_amount = min(WEEKLY_BUDGET, cash_pile)
            cash_pile -= invest_amount

        units += invest_amount / price
        total_invested += invest_amount
        equity_curve.append({
            'date': dt, 'units': units, 'invested': total_invested,
            'portfolio_value': units * price, 'price': price,
            'invest_this_week': invest_amount, 'cash_pile': cash_pile
        })

    return pd.DataFrame(equity_curve).set_index('date')


def run_drawdown_dca(data, weekly_dates, ticker='upro_close'):
    """Invest $200 when UPRO is down >10% from 20-day high, $100 otherwise.
    Budget-neutral: save $50/week during non-drawdown, deploy extra during drawdown."""
    units = 0.0
    total_invested = 0.0
    savings_buffer = 0.0
    equity_curve = []

    for dt in weekly_dates:
        if dt not in data.index:
            continue
        row = data.loc[dt]
        price = row[ticker]
        dd = row['upro_dd_from_high']

        if dd < -10:
            # Drawdown: invest double + any savings
            invest_amount = 200.0 + savings_buffer
            savings_buffer = 0.0
        else:
            # Normal: invest $75, save $25
            invest_amount = 75.0
            savings_buffer += 25.0

        units += invest_amount / price
        total_invested += invest_amount
        equity_curve.append({
            'date': dt, 'units': units, 'invested': total_invested,
            'portfolio_value': units * price, 'price': price,
            'invest_this_week': invest_amount
        })

    return pd.DataFrame(equity_curve).set_index('date')


def run_momentum_dca(data, weekly_dates, ticker='upro_close'):
    """Invest $133 when SPY > SMA200 (uptrend), $67 when below."""
    units = 0.0
    total_invested = 0.0
    equity_curve = []

    for dt in weekly_dates:
        if dt not in data.index:
            continue
        row = data.loc[dt]
        price = row[ticker]
        uptrend = row['spy_close'] > row['spy_sma200']
        invest_amount = 133.0 if uptrend else 67.0
        units += invest_amount / price
        total_invested += invest_amount
        equity_curve.append({
            'date': dt, 'units': units, 'invested': total_invested,
            'portfolio_value': units * price, 'price': price,
            'invest_this_week': invest_amount
        })

    return pd.DataFrame(equity_curve).set_index('date')


def run_combined_dca(data, weekly_dates, ticker='upro_close'):
    """Combined signal: VIX + trend.
    Both green (VIX < 17 + SPY > SMA50) = normal $100
    One yellow = save $50, invest $50
    Both red (VIX > 25 + SPY < SMA50) = deploy all accumulated cash
    """
    units = 0.0
    total_invested = 0.0
    cash_pile = 0.0
    equity_curve = []

    for dt in weekly_dates:
        if dt not in data.index:
            continue
        row = data.loc[dt]
        price = row[ticker]
        vix = row['vix_close']
        trend_up = row['spy_close'] > row['spy_sma50']
        vix_low = vix < 17
        vix_high = vix > 25

        cash_pile += WEEKLY_BUDGET

        if vix_high and not trend_up:
            # Both red: deploy everything
            invest_amount = cash_pile
            cash_pile = 0.0
        elif vix_low and trend_up:
            # Both green: normal DCA
            invest_amount = min(WEEKLY_BUDGET, cash_pile)
            cash_pile -= invest_amount
        else:
            # Mixed: save more
            invest_amount = min(50.0, cash_pile)
            cash_pile -= invest_amount

        units += invest_amount / price
        total_invested += invest_amount
        equity_curve.append({
            'date': dt, 'units': units, 'invested': total_invested,
            'portfolio_value': units * price, 'price': price,
            'invest_this_week': invest_amount, 'cash_pile': cash_pile
        })

    return pd.DataFrame(equity_curve).set_index('date')


# ─────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(curve, label=''):
    """Compute performance metrics for a DCA equity curve."""
    if len(curve) < 10:
        return {}

    # Average cost basis
    final_units = curve['units'].iloc[-1]
    total_invested = curve['invested'].iloc[-1]
    avg_cost = total_invested / final_units if final_units > 0 else np.nan
    final_value = curve['portfolio_value'].iloc[-1]
    total_return = (final_value / total_invested - 1) * 100

    # IRR (annualized)
    cashflows = []
    dates = []
    prev_invested = 0
    for dt, row in curve.iterrows():
        cf = -(row['invested'] - prev_invested)  # negative = outflow
        cashflows.append(cf)
        dates.append(dt)
        prev_invested = row['invested']
    # Add final value as inflow
    cashflows.append(final_value)
    dates.append(curve.index[-1])

    # Simple annualized return instead of XIRR (more robust)
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    ann_return = ((final_value / total_invested) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Sharpe of portfolio growth path
    curve_returns = curve['portfolio_value'].pct_change().dropna()
    if len(curve_returns) > 1 and curve_returns.std() > 0:
        # Annualize: weekly returns -> annual
        sharpe = (curve_returns.mean() / curve_returns.std()) * np.sqrt(52)
        sortino_downside = curve_returns[curve_returns < 0].std()
        sortino = (curve_returns.mean() / sortino_downside) * np.sqrt(52) if sortino_downside > 0 else np.nan
    else:
        sharpe = np.nan
        sortino = np.nan

    # MaxDD of equity curve
    peak = curve['portfolio_value'].expanding().max()
    dd = (curve['portfolio_value'] / peak - 1) * 100
    max_dd = dd.min()

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    return {
        'strategy': label,
        'total_invested': round(total_invested, 2),
        'final_value': round(final_value, 2),
        'total_return_pct': round(total_return, 2),
        'ann_return_pct': round(ann_return, 2),
        'avg_cost_basis': round(avg_cost, 4),
        'total_units': round(final_units, 4),
        'sharpe': round(sharpe, 3) if not np.isnan(sharpe) else None,
        'sortino': round(sortino, 3) if not np.isnan(sortino) else None,
        'max_dd_pct': round(max_dd, 2),
        'calmar': round(calmar, 3) if not np.isnan(calmar) else None,
        'n_weeks': len(curve),
        'years': round(years, 1),
    }


# ─────────────────────────────────────────────────────────────────────
# PERMUTATION TEST
# ─────────────────────────────────────────────────────────────────────

def permutation_test(data, weekly_dates, strategy_func, ticker='upro_close',
                     metric='avg_cost_basis', n_perms=100, lower_is_better=True):
    """
    Shuffle the timing signal (which weeks get extra $) and compare.
    For avg_cost_basis, lower is better. For total_return, higher is better.
    """
    # Run actual strategy
    actual_curve = strategy_func(data, weekly_dates, ticker)
    actual_metrics = compute_metrics(actual_curve)
    actual_val = actual_metrics.get(metric, np.nan)

    if np.isnan(actual_val):
        return actual_val, np.nan, np.nan

    # Get the investment amounts from actual strategy
    actual_amounts = actual_curve['invest_this_week'].values.copy()

    # Permutation: shuffle which weeks get which investment amounts
    perm_vals = []
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled_amounts = actual_amounts.copy()
        rng.shuffle(shuffled_amounts)

        # Reconstruct equity curve with shuffled amounts
        units = 0.0
        total_invested = 0.0
        perm_curve_data = []
        valid_dates = [dt for dt in weekly_dates if dt in data.index]

        for i, dt in enumerate(valid_dates[:len(shuffled_amounts)]):
            price = data.loc[dt, ticker]
            invest_amount = shuffled_amounts[i]
            units += invest_amount / price
            total_invested += invest_amount
            perm_curve_data.append({
                'date': dt, 'units': units, 'invested': total_invested,
                'portfolio_value': units * price, 'price': price,
                'invest_this_week': invest_amount
            })

        perm_curve = pd.DataFrame(perm_curve_data).set_index('date')
        perm_metrics = compute_metrics(perm_curve)
        perm_val = perm_metrics.get(metric, np.nan)
        if not np.isnan(perm_val):
            perm_vals.append(perm_val)

    if not perm_vals:
        return actual_val, np.nan, np.nan

    perm_vals = np.array(perm_vals)

    if lower_is_better:
        p_value = np.mean(perm_vals <= actual_val)
    else:
        p_value = np.mean(perm_vals >= actual_val)

    return actual_val, np.mean(perm_vals), p_value


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("DCA TIMING OPTIMIZATION STUDY")
    print("=" * 80)
    print()

    # Download data
    data = download_data()
    weekly_dates = get_weekly_dates(data)
    print(f"Weekly dates: {len(weekly_dates)} weeks from {weekly_dates[0].date()} to {weekly_dates[-1].date()}")
    print()

    strategies = {
        'Fixed DCA (Baseline)': run_fixed_dca,
        'Value DCA (SMA50)': run_value_dca,
        'VIX-Triggered DCA': run_vix_dca,
        'Drawdown DCA': run_drawdown_dca,
        'Momentum DCA (SMA200)': run_momentum_dca,
        'Combined (VIX+Trend)': run_combined_dca,
    }

    tickers = {
        'SPY': 'spy_close',
        'UPRO': 'upro_close',
    }

    # Define sub-periods for walk-forward
    periods = {
        'Full (2013-2026)': (None, None),
        'In-Sample (2013-2019)': (None, '2019-12-31'),
        'Out-of-Sample (2020-2026)': ('2020-01-01', None),
    }

    all_results = []
    all_curves = {}

    for ticker_name, ticker_col in tickers.items():
        print(f"\n{'='*80}")
        print(f"  TICKER: {ticker_name}")
        print(f"{'='*80}")

        for period_name, (start, end) in periods.items():
            print(f"\n--- {period_name} ---")

            # Filter data
            mask = pd.Series(True, index=data.index)
            if start:
                mask &= data.index >= start
            if end:
                mask &= data.index <= end
            period_data = data[mask]
            period_weekly = get_weekly_dates(period_data)

            if len(period_weekly) < 10:
                print(f"  Skipping: only {len(period_weekly)} weeks")
                continue

            print(f"  Period: {period_weekly[0].date()} to {period_weekly[-1].date()} ({len(period_weekly)} weeks)")

            baseline_metrics = None

            for strat_name, strat_func in strategies.items():
                curve = strat_func(period_data, period_weekly, ticker_col)
                metrics = compute_metrics(curve, strat_name)
                metrics['ticker'] = ticker_name
                metrics['period'] = period_name

                if strat_name == 'Fixed DCA (Baseline)':
                    baseline_metrics = metrics.copy()

                # Compute improvement vs baseline
                if baseline_metrics:
                    cost_improvement = 0
                    if baseline_metrics['avg_cost_basis'] > 0:
                        cost_improvement = (baseline_metrics['avg_cost_basis'] - metrics['avg_cost_basis']) / \
                                           baseline_metrics['avg_cost_basis'] * 100
                    return_improvement = metrics['total_return_pct'] - baseline_metrics['total_return_pct']
                    metrics['cost_improvement_pct'] = round(cost_improvement, 3)
                    metrics['return_improvement_pct'] = round(return_improvement, 2)
                else:
                    metrics['cost_improvement_pct'] = 0
                    metrics['return_improvement_pct'] = 0

                all_results.append(metrics)

                # Save full-period curves
                if period_name.startswith('Full'):
                    key = f"{ticker_name}_{strat_name}"
                    all_curves[key] = curve

            # Print comparison table
            period_results = [r for r in all_results
                              if r['ticker'] == ticker_name and r['period'] == period_name]

            print(f"\n  {'Strategy':<28s} {'Invested':>10s} {'Final Val':>12s} {'Tot Ret%':>9s} "
                  f"{'Ann Ret%':>9s} {'Sharpe':>7s} {'MaxDD%':>8s} {'AvgCost':>10s} {'Cost Imp%':>10s}")
            print(f"  {'-'*28} {'-'*10} {'-'*12} {'-'*9} {'-'*9} {'-'*7} {'-'*8} {'-'*10} {'-'*10}")

            for r in period_results:
                print(f"  {r['strategy']:<28s} "
                      f"${r['total_invested']:>9,.0f} "
                      f"${r['final_value']:>11,.0f} "
                      f"{r['total_return_pct']:>8.1f}% "
                      f"{r['ann_return_pct']:>8.1f}% "
                      f"{r.get('sharpe', 'N/A'):>7} "
                      f"{r['max_dd_pct']:>7.1f}% "
                      f"${r['avg_cost_basis']:>9.2f} "
                      f"{r.get('cost_improvement_pct', 0):>9.3f}%")

    # ─────────────────────────────────────────────────────────────────
    # PERMUTATION TESTS (full period only)
    # ─────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  PERMUTATION TESTS (100 shuffles, full period)")
    print(f"{'='*80}\n")

    perm_results = {}
    for ticker_name, ticker_col in tickers.items():
        print(f"  {ticker_name}:")
        perm_results[ticker_name] = {}

        for strat_name, strat_func in strategies.items():
            if strat_name == 'Fixed DCA (Baseline)':
                continue  # Can't shuffle a constant

            # Test avg cost basis (lower is better)
            actual_cost, perm_mean_cost, p_cost = permutation_test(
                data, weekly_dates, strat_func, ticker_col,
                metric='avg_cost_basis', n_perms=100, lower_is_better=True
            )

            # Test total return (higher is better)
            actual_ret, perm_mean_ret, p_ret = permutation_test(
                data, weekly_dates, strat_func, ticker_col,
                metric='total_return_pct', n_perms=100, lower_is_better=False
            )

            sig_cost = "***" if p_cost < 0.01 else ("**" if p_cost < 0.05 else ("*" if p_cost < 0.10 else ""))
            sig_ret = "***" if p_ret < 0.01 else ("**" if p_ret < 0.05 else ("*" if p_ret < 0.10 else ""))

            perm_results[ticker_name][strat_name] = {
                'actual_cost': actual_cost,
                'perm_mean_cost': perm_mean_cost,
                'p_cost': p_cost,
                'actual_return': actual_ret,
                'perm_mean_return': perm_mean_ret,
                'p_return': p_ret,
            }

            print(f"    {strat_name:<28s}  "
                  f"AvgCost: {actual_cost:>8.2f} vs perm {perm_mean_cost:>8.2f} (p={p_cost:.3f}{sig_cost})  "
                  f"Return: {actual_ret:>7.1f}% vs perm {perm_mean_ret:>7.1f}% (p={p_ret:.3f}{sig_ret})")

        print()

    # ─────────────────────────────────────────────────────────────────
    # WALK-FORWARD CONSISTENCY CHECK
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  WALK-FORWARD CONSISTENCY (strategy must beat baseline in BOTH sub-periods)")
    print(f"{'='*80}\n")

    wf_pass = {}
    for ticker_name in tickers:
        wf_pass[ticker_name] = {}
        for strat_name in strategies:
            if strat_name == 'Fixed DCA (Baseline)':
                continue
            is_results = [r for r in all_results
                          if r['ticker'] == ticker_name and r['period'].startswith('In-Sample')
                          and r['strategy'] == strat_name]
            oos_results = [r for r in all_results
                           if r['ticker'] == ticker_name and r['period'].startswith('Out-of-Sample')
                           and r['strategy'] == strat_name]

            if is_results and oos_results:
                is_better = is_results[0].get('cost_improvement_pct', 0) > 0
                oos_better = oos_results[0].get('cost_improvement_pct', 0) > 0
                both_pass = is_better and oos_better

                # Also check return improvement
                is_ret_better = is_results[0].get('return_improvement_pct', 0) > 0
                oos_ret_better = oos_results[0].get('return_improvement_pct', 0) > 0
                ret_pass = is_ret_better and oos_ret_better

                wf_pass[ticker_name][strat_name] = {
                    'cost_is': is_results[0].get('cost_improvement_pct', 0),
                    'cost_oos': oos_results[0].get('cost_improvement_pct', 0),
                    'cost_pass': both_pass,
                    'ret_is': is_results[0].get('return_improvement_pct', 0),
                    'ret_oos': oos_results[0].get('return_improvement_pct', 0),
                    'ret_pass': ret_pass,
                }

                status_cost = "PASS" if both_pass else "FAIL"
                status_ret = "PASS" if ret_pass else "FAIL"
                print(f"  {ticker_name} | {strat_name:<28s} | "
                      f"CostBasis: IS={is_results[0].get('cost_improvement_pct', 0):+.3f}% "
                      f"OOS={oos_results[0].get('cost_improvement_pct', 0):+.3f}% [{status_cost}] | "
                      f"Return: IS={is_results[0].get('return_improvement_pct', 0):+.1f}% "
                      f"OOS={oos_results[0].get('return_improvement_pct', 0):+.1f}% [{status_ret}]")

    # ─────────────────────────────────────────────────────────────────
    # FINAL VERDICT
    # ─────────────────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print("  FINAL VERDICT")
    print(f"{'='*80}\n")

    for ticker_name in tickers:
        print(f"  {ticker_name}:")
        any_winner = False
        for strat_name in strategies:
            if strat_name == 'Fixed DCA (Baseline)':
                continue

            # Check all gates
            wf = wf_pass.get(ticker_name, {}).get(strat_name, {})
            perm = perm_results.get(ticker_name, {}).get(strat_name, {})

            cost_wf_pass = wf.get('cost_pass', False)
            ret_wf_pass = wf.get('ret_pass', False)
            perm_cost_sig = perm.get('p_cost', 1.0) < 0.05 if perm else False
            perm_ret_sig = perm.get('p_return', 1.0) < 0.05 if perm else False

            gates_passed = sum([cost_wf_pass, ret_wf_pass, perm_cost_sig, perm_ret_sig])

            if gates_passed >= 3:
                verdict = "STRONG WINNER"
                any_winner = True
            elif gates_passed >= 2:
                verdict = "MARGINAL"
                any_winner = True
            elif gates_passed >= 1:
                verdict = "WEAK/UNRELIABLE"
            else:
                verdict = "NO EDGE"

            full_results = [r for r in all_results
                            if r['ticker'] == ticker_name and r['period'].startswith('Full')
                            and r['strategy'] == strat_name]
            ann_imp = full_results[0].get('return_improvement_pct', 0) if full_results else 0
            cost_imp = full_results[0].get('cost_improvement_pct', 0) if full_results else 0

            print(f"    {strat_name:<28s}: {verdict:<18s} "
                  f"(WF-cost:{'+' if cost_wf_pass else '-'} WF-ret:{'+' if ret_wf_pass else '-'} "
                  f"Perm-cost:{'+' if perm_cost_sig else '-'} Perm-ret:{'+' if perm_ret_sig else '-'}) "
                  f"Ann +{ann_imp:.1f}% | Cost {cost_imp:+.3f}%")

        if not any_winner:
            print(f"    -> No strategy reliably beats fixed DCA for {ticker_name}.")
            print(f"       Fixed weekly DCA is the recommended approach.")
        print()

    # ─────────────────────────────────────────────────────────────────
    # SAVE OUTPUTS
    # ─────────────────────────────────────────────────────────────────

    # Save CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, 'dca_timing_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Save equity curves
    for key, curve in all_curves.items():
        curve_path = os.path.join(OUTPUT_DIR, f'equity_curve_{key.replace(" ", "_").replace("(", "").replace(")", "")}.csv')
        curve.to_csv(curve_path)

    # Save JSON summary
    summary = {
        'study': 'DCA Timing Optimization',
        'date_run': datetime.now().isoformat(),
        'data_range': f"{data.index[0].date()} to {data.index[-1].date()}",
        'baseline': f'${WEEKLY_BUDGET}/week fixed DCA',
        'annual_budget': ANNUAL_BUDGET,
        'n_permutations': 100,
        'results': all_results,
        'permutation_tests': {
            ticker: {
                strat: {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in vals.items()}
                for strat, vals in strats.items()
            }
            for ticker, strats in perm_results.items()
        },
        'walk_forward': {
            ticker: {
                strat: {k: float(v) if isinstance(v, (np.floating, float, np.bool_)) else v
                        for k, v in vals.items()}
                for strat, vals in strats.items()
            }
            for ticker, strats in wf_pass.items()
        },
    }

    json_path = os.path.join(OUTPUT_DIR, 'dca_timing_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    print(f"\nAll equity curves saved to: {OUTPUT_DIR}/")
    print("\nDone.")


if __name__ == '__main__':
    main()
