#!/usr/bin/env python3
"""
Rebalancing Frequency & Method Optimization
============================================
Tests when and how to rebalance our risk parity portfolio.

Research questions:
1. What's the optimal rebalancing frequency? (daily, weekly, monthly, quarterly, threshold-based)
2. Does threshold-based rebalancing beat calendar-based?
3. What's the rebalancing premium (or cost)?
4. How does transaction cost affect optimal frequency?
5. Tax-loss harvesting: how much value does it add?

Uses our validated risk parity allocation (entry 386) as baseline.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/rebalancing'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Our validated assets from risk parity research
TICKERS = ['UPRO', 'TQQQ', 'TMF', 'GLD', 'SLV', 'USO', 'UUP']

# Transaction cost (round-trip, ETF)
TX_COST = 0.001  # 10 bps round-trip (conservative for liquid ETFs)

def download_data():
    """Download all needed tickers."""
    print(f"Downloading {len(TICKERS)} tickers...")
    data = yf.download(TICKERS, start='2012-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass

    closes = closes.dropna(how='all').dropna(subset=['UPRO'])
    print(f"  Data: {len(closes)} days, {closes.shape[1]} tickers")
    print(f"  Range: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
    return closes

def compute_risk_parity_weights(returns, lookback=126):
    """Inverse-volatility weighting (risk parity)."""
    vols = returns.iloc[-lookback:].std() * np.sqrt(252)
    inv_vol = 1 / vols.clip(lower=0.01)
    weights = inv_vol / inv_vol.sum()
    return weights

def compute_equal_weights(n):
    """Equal weighting."""
    return pd.Series(1/n, index=range(n))

def simulate_portfolio(closes, rebalance_func, tx_cost=TX_COST, name=""):
    """
    Simulate portfolio with given rebalancing function.

    rebalance_func(day_idx, current_weights, target_weights, closes, returns)
        -> should return True if rebalance today
    """
    returns = closes.pct_change().fillna(0)
    tickers = closes.columns.tolist()
    n = len(tickers)

    # Start with equal weights (first 126 days warmup)
    warmup = 126
    sim_returns = []
    tx_costs_paid = []
    n_rebalances = 0

    # Track actual weights (drift between rebalances)
    current_weights = np.ones(n) / n
    target_weights = np.ones(n) / n

    for i in range(warmup, len(closes)):
        daily_ret = returns.iloc[i].values

        # Portfolio return (weighted)
        port_ret = np.sum(current_weights * daily_ret)

        # Check if we should rebalance
        should_rebalance = rebalance_func(
            i, current_weights, target_weights, closes, returns
        )

        if should_rebalance and i > warmup + 10:
            # Compute new target weights
            hist_ret = returns.iloc[max(0, i-126):i]
            vols = hist_ret.std() * np.sqrt(252)
            inv_vol = 1 / vols.clip(lower=0.01)
            target_weights = (inv_vol / inv_vol.sum()).values

            # Transaction cost: proportional to turnover
            turnover = np.sum(np.abs(current_weights - target_weights))
            cost = turnover * tx_cost
            port_ret -= cost
            tx_costs_paid.append(cost)
            n_rebalances += 1

            current_weights = target_weights.copy()
        else:
            # Weights drift with returns
            new_weights = current_weights * (1 + daily_ret)
            total = np.sum(new_weights)
            if total > 0:
                current_weights = new_weights / total

        sim_returns.append(port_ret)

    sim_returns = pd.Series(sim_returns, index=closes.index[warmup:])

    return sim_returns, n_rebalances, sum(tx_costs_paid)

def compute_metrics(returns, name=""):
    """Compute risk-adjusted metrics."""
    r = returns.dropna()
    if len(r) < 63:
        return None

    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg = r[r < 0]
    downside_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    equity = (1 + r).cumprod()
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    cagr = (equity.iloc[-1]) ** (252 / len(r)) - 1
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wr = (r > 0).mean()

    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    return {
        'name': name,
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
        'calmar': float(calmar),
        'win_rate': float(wr * 100),
        'profit_factor': float(pf),
        'ann_vol': float(ann_vol * 100),
        'n_days': len(r),
    }

def permutation_test(returns, baseline_returns, n_perms=200):
    """Test if rebalancing timing matters vs random timing."""
    base_m = compute_metrics(baseline_returns)
    test_m = compute_metrics(returns)
    if not base_m or not test_m:
        return 1.0

    actual_diff = test_m['sharpe'] - base_m['sharpe']

    count = 0
    combined = np.concatenate([returns.values, baseline_returns.values])

    for _ in range(n_perms):
        # Random split
        np.random.shuffle(combined)
        n = len(returns)
        r1 = pd.Series(combined[:n])
        r2 = pd.Series(combined[n:2*n])
        m1 = compute_metrics(r1)
        m2 = compute_metrics(r2)
        if m1 and m2 and (m1['sharpe'] - m2['sharpe']) >= actual_diff:
            count += 1

    return count / n_perms

def main():
    print("="*70)
    print("REBALANCING FREQUENCY & METHOD OPTIMIZATION")
    print("="*70)

    closes = download_data()

    # Define rebalancing strategies
    def daily_rebal(i, cw, tw, c, r):
        return True

    def weekly_rebal(i, cw, tw, c, r):
        # Rebalance on Mondays
        return c.index[i].weekday() == 0

    def biweekly_rebal(i, cw, tw, c, r):
        return c.index[i].weekday() == 0 and c.index[i].isocalendar()[1] % 2 == 0

    def monthly_rebal(i, cw, tw, c, r):
        # First trading day of month
        if i == 0:
            return True
        return c.index[i].month != c.index[i-1].month

    def quarterly_rebal(i, cw, tw, c, r):
        if i == 0:
            return True
        cur_q = (c.index[i].month - 1) // 3
        prev_q = (c.index[i-1].month - 1) // 3
        return cur_q != prev_q

    def threshold_5pct(i, cw, tw, c, r):
        """Rebalance when any weight drifts >5% from target."""
        if i < 130:
            return False
        hist_ret = r.iloc[max(0, i-126):i]
        vols = hist_ret.std() * np.sqrt(252)
        inv_vol = 1 / vols.clip(lower=0.01)
        target = (inv_vol / inv_vol.sum()).values
        max_drift = np.max(np.abs(cw - target))
        return max_drift > 0.05

    def threshold_10pct(i, cw, tw, c, r):
        """Rebalance when any weight drifts >10% from target."""
        if i < 130:
            return False
        hist_ret = r.iloc[max(0, i-126):i]
        vols = hist_ret.std() * np.sqrt(252)
        inv_vol = 1 / vols.clip(lower=0.01)
        target = (inv_vol / inv_vol.sum()).values
        max_drift = np.max(np.abs(cw - target))
        return max_drift > 0.10

    def threshold_15pct(i, cw, tw, c, r):
        """Rebalance when any weight drifts >15% from target."""
        if i < 130:
            return False
        hist_ret = r.iloc[max(0, i-126):i]
        vols = hist_ret.std() * np.sqrt(252)
        inv_vol = 1 / vols.clip(lower=0.01)
        target = (inv_vol / inv_vol.sum()).values
        max_drift = np.max(np.abs(cw - target))
        return max_drift > 0.15

    def monthly_plus_threshold(i, cw, tw, c, r):
        """Monthly OR when drift >10%."""
        is_monthly = monthly_rebal(i, cw, tw, c, r)
        is_threshold = threshold_10pct(i, cw, tw, c, r)
        return is_monthly or is_threshold

    def never_rebal(i, cw, tw, c, r):
        """Buy and hold — never rebalance (drift only)."""
        return False

    strategies = {
        'Never (drift)': never_rebal,
        'Daily': daily_rebal,
        'Weekly': weekly_rebal,
        'Biweekly': biweekly_rebal,
        'Monthly': monthly_rebal,
        'Quarterly': quarterly_rebal,
        'Threshold 5%': threshold_5pct,
        'Threshold 10%': threshold_10pct,
        'Threshold 15%': threshold_15pct,
        'Monthly + Thresh 10%': monthly_plus_threshold,
    }

    print(f"\nTesting {len(strategies)} rebalancing strategies...")
    print(f"Transaction cost: {TX_COST*10000:.0f} bps round-trip\n")

    results = {}
    all_returns = {}

    for name, func in strategies.items():
        print(f"  Running: {name}...", end=" ", flush=True)
        rets, n_rebal, total_tx = simulate_portfolio(closes, func, TX_COST, name)
        m = compute_metrics(rets, name)

        if m:
            m['n_rebalances'] = n_rebal
            m['total_tx_cost'] = float(total_tx * 100)
            m['avg_annual_rebal'] = n_rebal / (m['n_days'] / 252)
            results[name] = m
            all_returns[name] = rets
            print(f"Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1f}%, MaxDD {m['max_dd']:.1f}%, "
                  f"rebalances={n_rebal}, tx cost={total_tx*100:.2f}%")

    # Sort by Sharpe
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n  {'Strategy':<25s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} {'Rebal/yr':>8s} {'TX Cost':>7s}")
    print("  " + "-"*72)
    for name, m in sorted_results:
        print(f"  {name:<25s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% "
              f"{m['avg_annual_rebal']:>7.1f} {m['total_tx_cost']:>6.2f}%")

    # Rebalancing premium: best calendar vs never
    never_m = results.get('Never (drift)')
    best_name, best_m = sorted_results[0]
    if never_m:
        premium = best_m['sharpe'] - never_m['sharpe']
        print(f"\n  Rebalancing premium (best vs drift): Sharpe +{premium:.3f}")
        print(f"  Best frequency: {best_name}")

    # Test with different tx costs
    print("\n" + "="*70)
    print("SENSITIVITY TO TRANSACTION COSTS")
    print("="*70)

    best_strat_name = sorted_results[0][0]
    best_func = strategies[best_strat_name]

    for tc in [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]:
        rets, n_rebal, total_tx = simulate_portfolio(closes, best_func, tc, "")
        m = compute_metrics(rets)
        if m:
            print(f"  TX cost {tc*10000:>5.0f} bps: Sharpe {m['sharpe']:.3f}, "
                  f"CAGR {m['cagr']:.1f}%, total TX paid {total_tx*100:.2f}%")

    # Sub-period consistency of winner
    print("\n" + "="*70)
    print("SUB-PERIOD CONSISTENCY")
    print("="*70)

    winner_rets = all_returns[best_name]
    n = len(winner_rets)
    n_periods = 3
    period_size = n // n_periods
    sub_sharpes = []

    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else n
        sub = winner_rets.iloc[start:end]
        m = compute_metrics(sub)
        if m:
            sub_sharpes.append(m['sharpe'])
            period_dates = f"{sub.index[0].strftime('%Y-%m')} to {sub.index[-1].strftime('%Y-%m')}"
            print(f"  Period {i+1} ({period_dates}): Sharpe {m['sharpe']:.3f}, "
                  f"CAGR {m['cagr']:.1f}%, MaxDD {m['max_dd']:.1f}%")

    all_positive = all(s > 0 for s in sub_sharpes)
    print(f"  All periods positive Sharpe: {'YES ✓' if all_positive else 'NO ✗'}")

    # Practical recommendation
    print("\n" + "="*70)
    print("PRACTICAL RECOMMENDATION")
    print("="*70)

    # Find most practical strategy (good Sharpe + low rebalancing frequency)
    practical = sorted(
        [(name, m) for name, m in results.items() if m['avg_annual_rebal'] < 52],
        key=lambda x: x[1]['sharpe'], reverse=True
    )

    if practical:
        rec_name, rec_m = practical[0]
        print(f"\n  RECOMMENDED: {rec_name}")
        print(f"    Sharpe: {rec_m['sharpe']:.3f}")
        print(f"    CAGR: {rec_m['cagr']:.1f}%")
        print(f"    MaxDD: {rec_m['max_dd']:.1f}%")
        print(f"    Rebalances/year: {rec_m['avg_annual_rebal']:.1f}")
        print(f"    Total TX cost: {rec_m['total_tx_cost']:.2f}%")

        if never_m:
            print(f"\n  vs Buy-and-drift:")
            print(f"    Sharpe improvement: +{rec_m['sharpe'] - never_m['sharpe']:.3f}")
            print(f"    MaxDD improvement: {rec_m['max_dd'] - never_m['max_dd']:+.1f}pp")

    # Save results
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'tx_cost_bps': TX_COST * 10000,
        'tickers': TICKERS,
        'results': {k: v for k, v in results.items()},
        'ranking': [name for name, _ in sorted_results],
        'recommendation': rec_name if practical else None,
    }

    output_path = os.path.join(OUTPUT_DIR, 'rebalancing_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
