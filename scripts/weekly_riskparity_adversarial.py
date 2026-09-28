#!/usr/bin/env python3
"""
Adversarial validation for Weekly Risk Parity strategy.
Assets: GLD, TLT, UUP — inverse-vol weighted, weekly rebalance, 8% target vol.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Config ───
ASSETS = ['GLD', 'TLT', 'UUP']
START = '2022-01-01'
END = '2026-07-29'
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
VOL_LOOKBACK = 20  # 4 weeks
TARGET_VOL = 0.08  # 8% annualized
REBAL_DAY = 4  # Friday (Monday=0)
ANNUALIZE = 252


def download_data():
    """Download price data for all needed tickers."""
    tickers = ASSETS + ['SPY', 'QQQ', '^VIX']
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    # yfinance returns MultiIndex columns: (Price, Ticker)
    close = data['Close']
    # Rename ^VIX
    close = close.rename(columns={'^VIX': 'VIX'})
    close = close.dropna()
    return close


def compute_returns(prices, assets=None):
    """Daily returns for given assets."""
    if assets is None:
        assets = ASSETS
    return prices[assets].pct_change().dropna()


def risk_parity_weights(returns, lookback, inverse=False):
    """Compute risk parity weights: inverse vol allocation."""
    vol = returns.rolling(lookback).std() * np.sqrt(ANNUALIZE)
    # For inverse=True, give MORE weight to HIGHER vol (opposite of risk parity)
    if inverse:
        raw = vol
    else:
        raw = 1.0 / vol
    weights = raw.div(raw.sum(axis=1), axis=0)
    return weights


def scale_to_target_vol(weights, returns, target_vol, lookback):
    """Scale total position to target portfolio vol."""
    # Portfolio vol using current weights
    cov = returns.rolling(lookback).cov()
    port_vol = pd.Series(index=weights.index, dtype=float)

    for dt in weights.index:
        if dt not in cov.index.get_level_values(0):
            port_vol[dt] = np.nan
            continue
        try:
            c = cov.loc[dt]
            w = weights.loc[dt].values
            pv = np.sqrt(w @ c.values @ w) * np.sqrt(ANNUALIZE)
            port_vol[dt] = pv
        except:
            port_vol[dt] = np.nan

    scale = target_vol / port_vol
    scale = scale.clip(0.1, 3.0)  # Safety bounds
    return scale


def run_strategy(prices, assets=None, lookback=20, target_vol=0.08,
                 rebal_day=4, slippage=0.0002, inverse=False,
                 rebal_dates=None, random_weights=False,
                 lag_days=0, rebal_freq='weekly'):
    """
    Run risk parity strategy.

    Parameters:
    - inverse: if True, give MORE weight to higher vol assets
    - rebal_dates: override rebalance dates (for random timing test)
    - random_weights: if True, randomize weights at each rebalance
    - lag_days: shift vol calculation by N days (look-ahead bias check)
    - rebal_freq: 'weekly', 'biweekly', 'monthly'
    """
    if assets is None:
        assets = ASSETS

    rets = compute_returns(prices, assets)

    # Compute weights with optional lag
    if lag_days > 0:
        # Use T-lag data for T's allocation
        shifted_rets = rets.shift(lag_days)
        weights = risk_parity_weights(shifted_rets, lookback, inverse=inverse)
    else:
        weights = risk_parity_weights(rets, lookback, inverse=inverse)

    # Determine rebalance dates
    if rebal_dates is not None:
        rebal_mask = rets.index.isin(rebal_dates)
    else:
        if rebal_freq == 'weekly':
            rebal_mask = rets.index.to_series().dt.dayofweek == rebal_day
        elif rebal_freq == 'biweekly':
            fridays = rets.index[rets.index.to_series().dt.dayofweek == rebal_day]
            biweekly = fridays[::2]
            rebal_mask = rets.index.isin(biweekly)
        elif rebal_freq == 'monthly':
            # Last trading day of each month
            monthly = rets.index.to_series().groupby(rets.index.to_period('M')).last()
            rebal_mask = rets.index.isin(monthly.values)
        else:
            rebal_mask = rets.index.to_series().dt.dayofweek == rebal_day

    # Drop rows where weights are NaN (warmup period)
    valid_start = weights.dropna().index[0] if not weights.dropna().empty else rets.index[-1]

    # Simulate
    capital = INITIAL_CAPITAL
    current_weights = pd.Series(1.0/len(assets), index=assets)
    equity = []
    trade_count = 0

    for dt in rets.index:
        if dt < valid_start:
            equity.append(capital)
            continue

        # Rebalance?
        if rebal_mask[dt] if isinstance(rebal_mask, pd.Series) else rebal_mask[rets.index.get_loc(dt)]:
            if random_weights:
                raw = np.random.dirichlet(np.ones(len(assets)))
                new_weights = pd.Series(raw, index=assets)
            else:
                w = weights.loc[dt]
                if w.isna().any():
                    new_weights = current_weights
                else:
                    new_weights = w

            # Scale to target vol
            if dt in weights.index and not weights.loc[dt].isna().any():
                # Simple vol scaling
                recent_rets = rets.loc[:dt].tail(lookback)
                if len(recent_rets) >= lookback:
                    port_ret = (recent_rets * new_weights).sum(axis=1)
                    port_vol_realized = port_ret.std() * np.sqrt(ANNUALIZE)
                    if port_vol_realized > 0.001:
                        scale = min(max(target_vol / port_vol_realized, 0.1), 3.0)
                        new_weights = new_weights * scale

            # Slippage on weight changes
            weight_change = (new_weights - current_weights).abs().sum()
            cost = capital * weight_change * slippage
            capital -= cost
            if weight_change > 0.01:
                trade_count += 1

            current_weights = new_weights

        # Daily return
        day_ret = (rets.loc[dt][assets] * current_weights).sum()
        capital *= (1 + day_ret)
        equity.append(capital)

    equity = pd.Series(equity, index=rets.index[:len(equity)])
    return equity, trade_count


def compute_metrics(equity):
    """Compute strategy metrics from equity curve."""
    rets = equity.pct_change().dropna()
    if len(rets) < 10:
        return {'sharpe': 0, 'sortino': 0, 'total_return': 0, 'max_dd': -1, 'vol': 1}

    mu = rets.mean() * ANNUALIZE
    sigma = rets.std() * np.sqrt(ANNUALIZE)
    sharpe = mu / sigma if sigma > 0 else 0

    downside = rets[rets < 0].std() * np.sqrt(ANNUALIZE)
    sortino = mu / downside if downside > 0 else 0

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min()

    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    return {
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'total_return': round(total_return, 4),
        'max_drawdown': round(max_dd, 4),
        'annual_vol': round(sigma, 4)
    }


def qqq_correlation(equity, prices):
    """Compute correlation with QQQ."""
    strat_rets = equity.pct_change().dropna()
    qqq_rets = prices['QQQ'].pct_change().dropna()
    common = strat_rets.index.intersection(qqq_rets.index)
    if len(common) < 20:
        return 0
    return round(strat_rets.loc[common].corr(qqq_rets.loc[common]), 4)


def check1_inverse_direction(prices, baseline_sharpe):
    """Inverse allocation: give HIGHEST weight to HIGHEST vol."""
    print("  Running Check 1: Inverse Direction...")
    equity, trades = run_strategy(prices, inverse=True)
    metrics = compute_metrics(equity)
    inverse_sharpe = metrics['sharpe']

    passed = (inverse_sharpe < 0) and (baseline_sharpe > 2 * abs(inverse_sharpe))

    return {
        'name': 'Inverse Direction Test',
        'passed': passed,
        'inverse_sharpe': inverse_sharpe,
        'baseline_sharpe': baseline_sharpe,
        'ratio': round(baseline_sharpe / abs(inverse_sharpe), 4) if inverse_sharpe != 0 else float('inf'),
        'criteria': 'inverse_sharpe < 0 AND baseline > 2x |inverse|',
        'inverse_metrics': metrics
    }


def check2_random_timing(prices, baseline_sharpe):
    """Randomize rebalance timing and weights — 1000 iterations."""
    print("  Running Check 2: Random Timing (1000 iters)...")
    rets = compute_returns(prices)

    # Count real rebalance events
    fridays = rets.index[rets.index.to_series().dt.dayofweek == REBAL_DAY]
    n_rebals = len(fridays)
    all_dates = rets.index.tolist()

    random_sharpes = []
    for i in range(1000):
        # Random rebalance dates (same count)
        rand_dates = sorted(np.random.choice(all_dates, size=min(n_rebals, len(all_dates)), replace=False))
        equity, _ = run_strategy(prices, rebal_dates=rand_dates, random_weights=True)
        m = compute_metrics(equity)
        random_sharpes.append(m['sharpe'])
        if (i+1) % 200 == 0:
            print(f"    ... {i+1}/1000")

    percentile = np.mean([1 if baseline_sharpe > s else 0 for s in random_sharpes]) * 100
    passed = percentile >= 90

    return {
        'name': 'Random Timing Test',
        'passed': passed,
        'baseline_sharpe': baseline_sharpe,
        'percentile': round(percentile, 2),
        'random_sharpe_mean': round(np.mean(random_sharpes), 4),
        'random_sharpe_std': round(np.std(random_sharpes), 4),
        'random_sharpe_p5': round(np.percentile(random_sharpes, 5), 4),
        'random_sharpe_p50': round(np.percentile(random_sharpes, 50), 4),
        'random_sharpe_p95': round(np.percentile(random_sharpes, 95), 4),
        'criteria': 'real strategy >= 90th percentile of random'
    }


def check3_lookahead_bias(prices, baseline_sharpe):
    """Use T-1 data for T's allocation."""
    print("  Running Check 3: Look-Ahead Bias...")
    equity, trades = run_strategy(prices, lag_days=1)
    metrics = compute_metrics(equity)
    lagged_sharpe = metrics['sharpe']

    within_pct = abs(lagged_sharpe - baseline_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 1
    passed = (lagged_sharpe > 0.5) and (within_pct <= 0.30)

    return {
        'name': 'Look-Ahead Bias Check',
        'passed': passed,
        'lagged_sharpe': lagged_sharpe,
        'baseline_sharpe': baseline_sharpe,
        'degradation_pct': round(within_pct * 100, 2),
        'criteria': 'lagged_sharpe > 0.5 AND within 30% of baseline',
        'lagged_metrics': metrics
    }


def check4_cost_sensitivity(prices):
    """Run at various slippage levels."""
    print("  Running Check 4: Cost Sensitivity...")
    slippage_levels = [0.0005, 0.001, 0.0015, 0.002]  # 0.05%, 0.10%, 0.15%, 0.20%
    results = {}

    for slip in slippage_levels:
        equity, trades = run_strategy(prices, slippage=slip)
        metrics = compute_metrics(equity)
        label = f"{slip*100:.2f}%"
        results[label] = metrics

    passed = results['0.10%']['sharpe'] > 0.5

    return {
        'name': 'Cost Sensitivity Test',
        'passed': passed,
        'results_by_slippage': results,
        'criteria': 'Sharpe > 0.5 at 0.10% slippage',
        'sharpe_at_10bps': results['0.10%']['sharpe']
    }


def check5_subperiod_stability(prices):
    """Split into 4 equal sub-periods."""
    print("  Running Check 5: Sub-Period Stability...")
    equity, _ = run_strategy(prices)

    n = len(equity)
    quarter = n // 4
    sub_results = []

    for i in range(4):
        start_idx = i * quarter
        end_idx = (i + 1) * quarter if i < 3 else n
        sub_equity = equity.iloc[start_idx:end_idx]
        metrics = compute_metrics(sub_equity)
        period_start = sub_equity.index[0].strftime('%Y-%m-%d')
        period_end = sub_equity.index[-1].strftime('%Y-%m-%d')
        sub_results.append({
            'period': f"{period_start} to {period_end}",
            'sharpe': metrics['sharpe'],
            'total_return': metrics['total_return'],
            'max_drawdown': metrics['max_drawdown']
        })

    positive_sharpe_count = sum(1 for r in sub_results if r['sharpe'] > 0)
    passed = positive_sharpe_count >= 3

    return {
        'name': 'Sub-Period Stability Test',
        'passed': passed,
        'sub_periods': sub_results,
        'positive_sharpe_periods': positive_sharpe_count,
        'criteria': '>=3 of 4 periods with Sharpe > 0'
    }


def check6_parameter_sensitivity(prices):
    """Grid search: rebal freq × vol lookback × target vol × asset combos."""
    print("  Running Check 6: Parameter Sensitivity (360 combos)...")

    rebal_freqs = ['weekly', 'biweekly', 'monthly']
    vol_lookbacks = [10, 15, 20, 30, 40, 60]
    target_vols = [0.06, 0.08, 0.10, 0.12, 0.15]
    asset_combos = [
        ['GLD', 'TLT', 'UUP'],
        ['GLD', 'TLT'],
        ['GLD', 'UUP'],
        ['TLT', 'UUP']
    ]

    total = len(rebal_freqs) * len(vol_lookbacks) * len(target_vols) * len(asset_combos)
    sharpes = []
    all_results = []
    count = 0

    for freq in rebal_freqs:
        for lb in vol_lookbacks:
            for tv in target_vols:
                for combo in asset_combos:
                    count += 1
                    try:
                        equity, _ = run_strategy(
                            prices, assets=combo, lookback=lb,
                            target_vol=tv, rebal_freq=freq
                        )
                        m = compute_metrics(equity)
                        sharpes.append(m['sharpe'])
                        all_results.append({
                            'freq': freq, 'lookback': lb, 'target_vol': tv,
                            'assets': '+'.join(combo), 'sharpe': m['sharpe'],
                            'return': m['total_return']
                        })
                    except Exception as e:
                        sharpes.append(0)

                    if count % 60 == 0:
                        print(f"    ... {count}/{total}")

    above_threshold = sum(1 for s in sharpes if s > 0.3)
    pct_above = above_threshold / len(sharpes)
    passed = pct_above >= 0.30

    # Best and worst combos
    if all_results:
        sorted_results = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
        best5 = sorted_results[:5]
        worst5 = sorted_results[-5:]
    else:
        best5, worst5 = [], []

    return {
        'name': 'Parameter Sensitivity Test',
        'passed': passed,
        'total_combinations': total,
        'combinations_sharpe_above_0_3': above_threshold,
        'pct_above_threshold': round(pct_above * 100, 2),
        'median_sharpe': round(np.median(sharpes), 4),
        'mean_sharpe': round(np.mean(sharpes), 4),
        'best_5': best5,
        'worst_5': worst5,
        'criteria': '>=30% of combos with Sharpe > 0.3'
    }


def main():
    print("=" * 60)
    print("ADVERSARIAL VALIDATION: Weekly Risk Parity")
    print("Assets: GLD + TLT + UUP, Inverse-Vol Weighted")
    print("=" * 60)

    # Download data
    print("\nDownloading data...")
    prices = download_data()
    print(f"  Data: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  {len(prices)} trading days")

    # Run baseline
    print("\nRunning baseline strategy...")
    equity, trade_count = run_strategy(prices)
    baseline = compute_metrics(equity)
    baseline['trade_count'] = trade_count
    qqq_corr = qqq_correlation(equity, prices)

    print(f"  Sharpe: {baseline['sharpe']}")
    print(f"  Sortino: {baseline['sortino']}")
    print(f"  Return: {baseline['total_return']*100:.1f}%")
    print(f"  Max DD: {baseline['max_drawdown']*100:.1f}%")
    print(f"  Trades: {trade_count}")
    print(f"  QQQ Corr: {qqq_corr}")

    # Also run equal-weight baseline for comparison
    print("\nRunning equal-weight baseline (no vol weighting)...")
    # Equal weight = just hold equal portions, same rebalance schedule
    # We'll use the strategy with random_weights=False but override to equal
    eq_rets = compute_returns(prices)
    eq_capital = INITIAL_CAPITAL
    eq_weights = pd.Series(1.0/3, index=ASSETS)
    eq_equity_list = []
    rebal_mask = eq_rets.index.to_series().dt.dayofweek == REBAL_DAY
    for dt in eq_rets.index:
        if rebal_mask[dt]:
            eq_weights = pd.Series(1.0/3, index=ASSETS)
        day_ret = (eq_rets.loc[dt][ASSETS] * eq_weights).sum()
        eq_capital *= (1 + day_ret)
        eq_equity_list.append(eq_capital)
    eq_equity = pd.Series(eq_equity_list, index=eq_rets.index)
    eq_metrics = compute_metrics(eq_equity)
    print(f"  Equal-weight Sharpe: {eq_metrics['sharpe']}")
    print(f"  Equal-weight Return: {eq_metrics['total_return']*100:.1f}%")

    # Run all checks
    print("\n" + "=" * 60)
    print("ADVERSARIAL CHECKS")
    print("=" * 60)

    c1 = check1_inverse_direction(prices, baseline['sharpe'])
    print(f"  → {'PASS' if c1['passed'] else 'FAIL'}: inverse Sharpe={c1['inverse_sharpe']}")

    c2 = check2_random_timing(prices, baseline['sharpe'])
    print(f"  → {'PASS' if c2['passed'] else 'FAIL'}: {c2['percentile']}th percentile")

    c3 = check3_lookahead_bias(prices, baseline['sharpe'])
    print(f"  → {'PASS' if c3['passed'] else 'FAIL'}: lagged Sharpe={c3['lagged_sharpe']}, degradation={c3['degradation_pct']:.1f}%")

    c4 = check4_cost_sensitivity(prices)
    print(f"  → {'PASS' if c4['passed'] else 'FAIL'}: Sharpe@10bps={c4['sharpe_at_10bps']}")

    c5 = check5_subperiod_stability(prices)
    print(f"  → {'PASS' if c5['passed'] else 'FAIL'}: {c5['positive_sharpe_periods']}/4 positive")

    c6 = check6_parameter_sensitivity(prices)
    print(f"  → {'PASS' if c6['passed'] else 'FAIL'}: {c6['pct_above_threshold']:.1f}% above 0.3 Sharpe")

    checks = [c1, c2, c3, c4, c5, c6]
    checks_passed = sum(1 for c in checks if c['passed'])
    overall_pass = checks_passed >= 5  # Need 5/6 for overall pass

    # Build results
    results = {
        'strategy': 'Weekly Risk Parity',
        'description': 'GLD+TLT+UUP inverse-vol weighted, weekly rebalance, 8% target vol',
        'data_period': f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}",
        'initial_capital': INITIAL_CAPITAL,
        'baseline': baseline,
        'equal_weight_baseline': eq_metrics,
        'risk_parity_vs_equal_weight': {
            'sharpe_diff': round(baseline['sharpe'] - eq_metrics['sharpe'], 4),
            'return_diff': round(baseline['total_return'] - eq_metrics['total_return'], 4),
            'vol_weighting_adds_alpha': baseline['sharpe'] > eq_metrics['sharpe']
        },
        'qqq_correlation': qqq_corr,
        'check_1_inverse_direction': c1,
        'check_2_random_timing': c2,
        'check_3_lookahead_bias': c3,
        'check_4_cost_sensitivity': c4,
        'check_5_subperiod_stability': c5,
        'check_6_parameter_sensitivity': c6,
        'checks_passed': f"{checks_passed}/6",
        'overall_pass': overall_pass,
        'timestamp': datetime.now().isoformat()
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Baseline Sharpe: {baseline['sharpe']}")
    print(f"  Baseline Sortino: {baseline['sortino']}")
    print(f"  QQQ Correlation: {qqq_corr}")
    print(f"  Equal-Weight Sharpe: {eq_metrics['sharpe']}")
    print(f"  Vol Weighting Adds Alpha: {baseline['sharpe'] > eq_metrics['sharpe']}")
    print(f"  Checks Passed: {checks_passed}/6")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    # Save
    output_path = '/home/jupiter/Lvl3Quant/data/weekly_riskparity_adversarial_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
