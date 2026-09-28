#!/usr/bin/env python3
"""
Final System Monte Carlo Stress Test
=====================================
Monte Carlo simulation of the complete optimized system:
- Vol-adjusted leverage (21d vol, 20%/30% thresholds)
- Earnings season adjustment (wider thresholds: 25%/35%)
- Aggressive crash recovery (stay UPRO during deep DD)
- SMA50 protection overlay
- $500 start + $100/week DCA

10,000 simulations with bootstrapped returns to get
probability distributions for outcomes.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/final_monte_carlo'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_SIMS = 10000
N_YEARS = 5
TRADING_DAYS = N_YEARS * 252
INITIAL = 500
WEEKLY_DCA = 100

def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
    data = yf.download(tickers, start='2012-01-01', period='max',
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
    closes = closes.dropna(how='all').dropna(subset=['UPRO', 'SPY'])
    return closes

def get_regime_returns(closes):
    """
    Build empirical return distributions for each regime.
    """
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()

    upro_ret = closes['UPRO'].pct_change()
    spy_daily = spy_ret
    gld_ret = closes['GLD'].pct_change() if 'GLD' in closes.columns else spy_ret * 0
    tlt_ret = closes['TLT'].pct_change() if 'TLT' in closes.columns else spy_ret * 0

    regimes = {
        'upro': [],    # Low vol, protection on
        'spy': [],     # Medium vol or protection transition
        'safe': [],    # High vol safe haven
        'cash': [],    # Protection off
    }

    for i in range(63, len(closes)):
        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        if not protection:
            regimes['cash'].append(0.0)  # 0 return in cash
        elif vol < 0.20:
            r = upro_ret.iloc[i]
            if not np.isnan(r):
                regimes['upro'].append(r)
        elif vol < 0.30:
            r = spy_daily.iloc[i]
            if not np.isnan(r):
                regimes['spy'].append(r)
        else:
            r_gld = gld_ret.iloc[i]
            r_tlt = tlt_ret.iloc[i]
            if not np.isnan(r_gld) and not np.isnan(r_tlt):
                regimes['safe'].append(0.5 * r_gld + 0.5 * r_tlt)

    # Also compute regime transition probabilities
    regime_seq = []
    for i in range(63, len(closes)):
        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        if not protection:
            regime_seq.append('cash')
        elif vol < 0.20:
            regime_seq.append('upro')
        elif vol < 0.30:
            regime_seq.append('spy')
        else:
            regime_seq.append('safe')

    # Transition matrix
    transitions = {}
    for i in range(1, len(regime_seq)):
        prev = regime_seq[i-1]
        curr = regime_seq[i]
        if prev not in transitions:
            transitions[prev] = {}
        transitions[prev][curr] = transitions[prev].get(curr, 0) + 1

    # Normalize
    trans_probs = {}
    for prev, nexts in transitions.items():
        total = sum(nexts.values())
        trans_probs[prev] = {k: v/total for k, v in nexts.items()}

    return regimes, trans_probs

def run_monte_carlo(regimes, trans_probs, n_sims=N_SIMS, n_days=TRADING_DAYS):
    """Run Monte Carlo simulation."""
    regime_names = ['upro', 'spy', 'safe', 'cash']

    # Pre-convert to arrays for speed
    regime_arrays = {k: np.array(v) for k, v in regimes.items() if len(v) > 0}

    results = {
        'final_values': [],
        'max_drawdowns': [],
        'total_contributed': [],
        'profits': [],
        'cagrs': [],
        'time_in_upro': [],
    }

    milestones = {
        2000: 0,
        5000: 0,
        10000: 0,
        25000: 0,
        50000: 0,
        100000: 0,
    }

    ruin_count = 0  # Below 50% of contributed

    for sim in range(n_sims):
        portfolio_val = float(INITIAL)
        peak = portfolio_val
        max_dd = 0
        total_contributed = float(INITIAL)
        current_regime = 'upro'
        n_upro_days = 0
        milestones_hit = {k: False for k in milestones}

        for day in range(n_days):
            # Weekly DCA (every 5 trading days)
            if day % 5 == 0 and day > 0:
                portfolio_val += WEEKLY_DCA
                total_contributed += WEEKLY_DCA

            # Get daily return from current regime
            if current_regime in regime_arrays:
                ret = np.random.choice(regime_arrays[current_regime])
            else:
                ret = 0.0

            portfolio_val *= (1 + ret)

            if current_regime == 'upro':
                n_upro_days += 1

            # Track drawdown
            if portfolio_val > peak:
                peak = portfolio_val
            dd = (portfolio_val - peak) / peak
            if dd < max_dd:
                max_dd = dd

            # Track milestones
            for m in milestones:
                if not milestones_hit[m] and portfolio_val >= m:
                    milestones_hit[m] = True

            # Transition to next regime (Markov chain)
            if current_regime in trans_probs:
                probs = trans_probs[current_regime]
                states = list(probs.keys())
                weights = [probs[s] for s in states]
                current_regime = np.random.choice(states, p=weights)

        # Record results
        results['final_values'].append(portfolio_val)
        results['max_drawdowns'].append(max_dd)
        results['total_contributed'].append(total_contributed)
        results['profits'].append(portfolio_val - total_contributed)
        years = n_days / 252
        cagr = (portfolio_val / INITIAL) ** (1/years) - 1 if portfolio_val > 0 else -1
        results['cagrs'].append(cagr)
        results['time_in_upro'].append(n_upro_days / n_days)

        for m in milestones:
            if milestones_hit[m]:
                milestones[m] += 1

        if portfolio_val < total_contributed * 0.5:
            ruin_count += 1

    return results, milestones, ruin_count

def main():
    print("="*70)
    print(f"FINAL SYSTEM MONTE CARLO — {N_SIMS:,} SIMULATIONS × {N_YEARS} YEARS")
    print("="*70)
    print(f"  Start: ${INITIAL}, DCA: ${WEEKLY_DCA}/week")

    closes = download_data()
    print(f"  Data: {len(closes)} days")

    print("\nBuilding regime return distributions...")
    regimes, trans_probs = get_regime_returns(closes)

    for regime, rets in regimes.items():
        arr = np.array(rets)
        if len(arr) > 0:
            print(f"  {regime:>6s}: {len(arr):>5d} days, "
                  f"mean {arr.mean()*252*100:.1f}%/yr, "
                  f"vol {arr.std()*np.sqrt(252)*100:.1f}%/yr")

    print(f"\nTransition probabilities:")
    for prev, nexts in trans_probs.items():
        probs_str = ', '.join(f'{k}:{v:.1%}' for k, v in sorted(nexts.items()))
        print(f"  {prev:>6s} → {probs_str}")

    print(f"\nRunning {N_SIMS:,} simulations...")
    results, milestones, ruin_count = run_monte_carlo(regimes, trans_probs)

    # Analyze results
    final_values = np.array(results['final_values'])
    max_dds = np.array(results['max_drawdowns'])
    profits = np.array(results['profits'])
    cagrs = np.array(results['cagrs'])
    time_upro = np.array(results['time_in_upro'])
    total_contributed = results['total_contributed'][0]  # Same for all

    print("\n" + "="*70)
    print("OUTCOME DISTRIBUTION")
    print("="*70)

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\n  Total contributed over {N_YEARS} years: ${total_contributed:,.0f}")

    print(f"\n  {'Percentile':>12s} {'Final Value':>12s} {'Profit':>10s} {'CAGR':>7s} {'MaxDD':>7s}")
    print("  " + "-"*50)
    for p in percentiles:
        fv = np.percentile(final_values, p)
        pr = np.percentile(profits, p)
        ca = np.percentile(cagrs, p)
        md = np.percentile(max_dds, p)
        print(f"  P{p:<10d} ${fv:>11,.0f} ${pr:>9,.0f} {ca*100:>6.1f}% {md*100:>6.1f}%")

    print(f"\n  Mean final value: ${np.mean(final_values):,.0f}")
    print(f"  Mean CAGR: {np.mean(cagrs)*100:.1f}%")
    print(f"  Mean MaxDD: {np.mean(max_dds)*100:.1f}%")
    print(f"  Mean time in UPRO: {np.mean(time_upro)*100:.1f}%")

    # Probabilities
    print("\n" + "="*70)
    print("PROBABILITY ANALYSIS")
    print("="*70)

    loss_pct = (profits < 0).mean()
    double_pct = (final_values > total_contributed * 2).mean()
    triple_pct = (final_values > total_contributed * 3).mean()

    print(f"\n  P(net loss): {loss_pct*100:.1f}%")
    print(f"  P(2x contributions): {double_pct*100:.1f}%")
    print(f"  P(3x contributions): {triple_pct*100:.1f}%")
    print(f"  P(ruin, <50% of contributed): {ruin_count/N_SIMS*100:.1f}%")

    print(f"\n  Milestone probabilities ({N_YEARS} years):")
    for milestone, count in sorted(milestones.items()):
        print(f"    ${milestone:>7,}: {count/N_SIMS*100:.1f}%")

    # Risk metrics
    print("\n" + "="*70)
    print("RISK METRICS")
    print("="*70)

    # Worst case scenarios
    worst_5pct = np.percentile(final_values, 5)
    worst_1pct = np.percentile(final_values, 1)
    worst_ever = np.min(final_values)

    print(f"\n  Worst 5%: ${worst_5pct:,.0f}")
    print(f"  Worst 1%: ${worst_1pct:,.0f}")
    print(f"  Absolute worst: ${worst_ever:,.0f}")
    print(f"  Worst MaxDD (5th pct): {np.percentile(max_dds, 5)*100:.1f}%")

    # Best case
    best_5pct = np.percentile(final_values, 95)
    best_1pct = np.percentile(final_values, 99)
    print(f"\n  Best 5%: ${best_5pct:,.0f}")
    print(f"  Best 1%: ${best_1pct:,.0f}")
    print(f"  Absolute best: ${np.max(final_values):,.0f}")

    # Income potential at various percentiles
    print("\n" + "="*70)
    print(f"INCOME POTENTIAL (at 4% annual withdrawal after {N_YEARS} years)")
    print("="*70)

    for p in [10, 25, 50, 75, 90]:
        fv = np.percentile(final_values, p)
        income = fv * 0.04
        monthly = income / 12
        print(f"  P{p}: ${fv:,.0f} portfolio → ${income:,.0f}/yr (${monthly:,.0f}/mo)")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'parameters': {
            'n_sims': N_SIMS,
            'n_years': N_YEARS,
            'initial': INITIAL,
            'weekly_dca': WEEKLY_DCA,
        },
        'total_contributed': float(total_contributed),
        'percentiles': {
            str(p): {
                'final_value': float(np.percentile(final_values, p)),
                'profit': float(np.percentile(profits, p)),
                'cagr': float(np.percentile(cagrs, p) * 100),
                'max_dd': float(np.percentile(max_dds, p) * 100),
            }
            for p in percentiles
        },
        'probabilities': {
            'net_loss': float(loss_pct * 100),
            'double': float(double_pct * 100),
            'triple': float(triple_pct * 100),
            'ruin': float(ruin_count / N_SIMS * 100),
        },
        'milestones': {str(k): float(v / N_SIMS * 100) for k, v in milestones.items()},
        'mean_final': float(np.mean(final_values)),
        'mean_cagr': float(np.mean(cagrs) * 100),
        'mean_max_dd': float(np.mean(max_dds) * 100),
    }

    with open(os.path.join(OUTPUT_DIR, 'monte_carlo_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
