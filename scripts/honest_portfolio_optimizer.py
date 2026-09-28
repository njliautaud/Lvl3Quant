#!/usr/bin/env python3
"""
Honest Combined Portfolio Optimizer
====================================
Uses CONSERVATIVE estimates for all 3 strategies:
- V5 CSP (d25): from delta ladder study
- IC Condors: permutation-based Sharpe 2.05 (NOT inflated 4.93/5.68)
- ETF Rotation v3: from hedged backtest (R1-passing config)

Tests portfolio allocations at multiple leverage levels with borrowing costs.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

ROOT = '/home/jupiter/Lvl3Quant'
OUT = f'{ROOT}/output/honest_portfolio_optimizer'
os.makedirs(OUT, exist_ok=True)

LEVERAGE_COST = 0.05  # 5% annual borrowing rate on leveraged portion
STARTING_CAPITAL = 100_000
RISK_FREE = 0.045  # current T-bill rate for Sharpe calc


# ============================================================
# 1. LOAD AND ALIGN DAILY RETURN SERIES
# ============================================================

def load_v5_csp_d25():
    """V5 CSP at delta-25 from delta ladder study."""
    df = pd.read_parquet(f'{ROOT}/output/delta_ladder_study/eq_d25.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    rets = df['equity'].pct_change().dropna()
    rets.name = 'V5_CSP_d25'
    return rets


def load_ic_condors_honest():
    """
    IC Condors using permutation-based Sharpe of 2.05.

    Method: Take the hedged IC equity curve's daily return SHAPE (normalized),
    then scale BOTH mean and vol to capacity-capped realistic levels.

    The raw backtest compounds to $441M which is fantasy. At capacity cap ($500K),
    the normalized comparison showed ~7-8% annual vol. We use that vol with the
    permutation-validated Sharpe of 2.05 to get honest return estimates.

    We preserve the CORRELATION structure and DRAWDOWN TIMING from the raw curve
    (when drawdowns happen relative to other strategies) but scale magnitudes
    to capacity-realistic levels.
    """
    df = pd.read_parquet(f'{ROOT}/output/ic_honest_recalc/corrected_equity_curves.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    raw_rets = df['ic_hedged_corrected'].pct_change().dropna()

    raw_sharpe = raw_rets.mean() / raw_rets.std() * np.sqrt(252)

    # CAPACITY-CAPPED SCALING: target 8% annual vol (from normalized comparison at 500K cap)
    # with permutation-validated Sharpe of 2.05
    target_vol_annual = 0.08
    target_vol_daily = target_vol_annual / np.sqrt(252)
    target_sharpe = 2.05

    # Standardize returns (preserve shape/timing), then rescale
    standardized = (raw_rets - raw_rets.mean()) / raw_rets.std()
    # New returns: mean = sharpe * vol_daily / sqrt(252)...
    # Actually: Sharpe = mean_daily / std_daily * sqrt(252)
    # So mean_daily = Sharpe * std_daily / sqrt(252)
    target_mean_daily = target_sharpe * target_vol_daily / np.sqrt(252)
    scaled_rets = standardized * target_vol_daily + target_mean_daily

    # Verify
    check_sharpe = scaled_rets.mean() / scaled_rets.std() * np.sqrt(252)
    check_vol = scaled_rets.std() * np.sqrt(252)
    check_cagr = (1 + scaled_rets.mean()) ** 252 - 1
    print(f"  IC Condors: raw Sharpe={raw_sharpe:.2f}, raw vol={raw_rets.std()*np.sqrt(252)*100:.1f}%")
    print(f"  -> Honest: Sharpe={check_sharpe:.2f}, vol={check_vol*100:.1f}%, CAGR~{check_cagr*100:.1f}%")
    print(f"  (capacity-capped at $500K, permutation-validated edge)")

    scaled_rets.name = 'IC_Condors'
    return scaled_rets


def load_etf_rotation_v3():
    """ETF Rotation v3 - R1-passing beta-scaled config."""
    df = pd.read_parquet(f'{ROOT}/output/etf_v3_hedge/equity_curves.parquet')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    # Use hedged version (R1 passing)
    rets = df['equity_hedged'].pct_change().dropna()
    rets.name = 'ETF_Rotation_v3'
    return rets


def align_return_series(series_list):
    """Align all return series to common dates."""
    combined = pd.concat(series_list, axis=1)
    # Use intersection of dates (all strategies must have data)
    aligned = combined.dropna()
    print(f"\n  Aligned date range: {aligned.index[0].date()} to {aligned.index[-1].date()}")
    print(f"  Common trading days: {len(aligned)}")
    return aligned


# ============================================================
# 2. PORTFOLIO METRICS
# ============================================================

def compute_metrics(daily_returns, label="", starting_cap=STARTING_CAPITAL):
    """Compute comprehensive risk-adjusted metrics from daily return series."""
    n_days = len(daily_returns)
    years = n_days / 252

    # Annualized return & vol
    cum_return = (1 + daily_returns).prod() - 1
    cagr = (1 + cum_return) ** (1 / years) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)

    # Sharpe (excess over risk-free)
    excess_daily = daily_returns - RISK_FREE / 252
    sharpe = excess_daily.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    # Sortino (downside deviation)
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (cagr - RISK_FREE) / downside_std

    # Max drawdown
    equity = (1 + daily_returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = equity / rolling_max - 1
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    # Win rate
    wr = (daily_returns > 0).sum() / len(daily_returns) * 100

    # Profit factor
    gains = daily_returns[daily_returns > 0].sum()
    losses = abs(daily_returns[daily_returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Monthly returns for VaR
    equity_series = pd.Series(
        (1 + daily_returns).cumprod().values,
        index=daily_returns.index
    )
    monthly = equity_series.resample('ME').last().pct_change().dropna()
    var_5 = monthly.quantile(0.05) if len(monthly) > 0 else 0

    # Worst and best year
    yearly_rets = equity_series.resample('YE').last().pct_change().dropna()
    worst_year = yearly_rets.min() if len(yearly_rets) > 0 else 0
    best_year = yearly_rets.max() if len(yearly_rets) > 0 else 0
    worst_year_label = str(yearly_rets.idxmin().year) if len(yearly_rets) > 0 else "N/A"
    best_year_label = str(yearly_rets.idxmax().year) if len(yearly_rets) > 0 else "N/A"

    # Final equity
    final_eq = starting_cap * (1 + cum_return)

    return {
        'label': label,
        'CAGR%': round(cagr * 100, 2),
        'AnnVol%': round(ann_vol * 100, 2),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'MaxDD%': round(max_dd * 100, 2),
        'Calmar': round(calmar, 2),
        'WR%': round(wr, 1),
        'PF': round(pf, 2),
        'Monthly_VaR_5%': round(var_5 * 100, 2),
        'Worst_Year%': round(worst_year * 100, 2),
        'Worst_Year': worst_year_label,
        'Best_Year%': round(best_year * 100, 2),
        'Best_Year': best_year_label,
        'Final_Equity': round(final_eq, 0),
        'Years': round(years, 1),
        'N_Days': n_days,
    }


def apply_leverage(daily_returns, leverage, leverage_cost=LEVERAGE_COST):
    """
    Apply leverage with borrowing costs.
    At leverage L:
    - Daily return = L * r - (L-1) * borrowing_cost / 252
    - Borrowing cost only applies to the leveraged portion (L-1)
    """
    if leverage <= 1.0:
        return daily_returns  # No borrowing needed at 1x or below

    daily_borrow = leverage_cost / 252
    leveraged = leverage * daily_returns - (leverage - 1) * daily_borrow
    return leveraged


# ============================================================
# 3. PORTFOLIO ALLOCATION METHODS
# ============================================================

def equal_weight(returns_df):
    """Equal weight: 1/N allocation."""
    n = returns_df.shape[1]
    weights = np.array([1/n] * n)
    return weights, "Equal Weight (33/33/33)"


def risk_parity(returns_df):
    """Inverse-volatility weighted."""
    vols = returns_df.std() * np.sqrt(252)
    inv_vol = 1 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.values, "Risk Parity (inv-vol)"


def concentrated_v5(returns_df):
    """60% V5, 20% IC, 20% ETF."""
    cols = returns_df.columns.tolist()
    weights = np.zeros(len(cols))
    for i, c in enumerate(cols):
        if 'V5' in c or 'CSP' in c:
            weights[i] = 0.60
        elif 'IC' in c:
            weights[i] = 0.20
        elif 'ETF' in c:
            weights[i] = 0.20
    return weights, "Concentrated (60% V5, 20% IC, 20% ETF)"


def max_sharpe(returns_df):
    """Mean-variance optimal (max Sharpe ratio)."""
    mu = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252
    n = len(mu)

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_ret - RISK_FREE) / port_vol if port_vol > 0 else 0

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.05, 0.80)] * n  # Min 5%, max 80% per strategy
    x0 = np.array([1/n] * n)

    result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    weights = result.x if result.success else x0
    return weights, "Max Sharpe (MVO)"


def kelly_optimal(returns_df):
    """Kelly criterion weights (half-Kelly for safety)."""
    mu = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252

    try:
        cov_inv = np.linalg.inv(cov)
        excess = mu - RISK_FREE
        full_kelly = cov_inv @ excess

        # Half-Kelly for practical safety
        half_kelly = full_kelly / 2

        # Normalize to sum to 1, clip negatives
        half_kelly = np.maximum(half_kelly, 0.05)  # min 5%
        half_kelly = np.minimum(half_kelly, 0.80)  # max 80%
        half_kelly = half_kelly / half_kelly.sum()

        return half_kelly, "Half-Kelly"
    except np.linalg.LinAlgError:
        return np.array([1/len(mu)] * len(mu)), "Half-Kelly (fallback equal)"


# ============================================================
# 4. MAIN ANALYSIS
# ============================================================

def run_analysis():
    print("=" * 70)
    print("HONEST COMBINED PORTFOLIO OPTIMIZER")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load strategies
    print("\n[1] Loading strategy return series...")
    v5 = load_v5_csp_d25()
    ic = load_ic_condors_honest()
    etf = load_etf_rotation_v3()

    # Align to common dates
    aligned = align_return_series([v5, ic, etf])

    # Individual strategy stats (at 1x)
    print("\n" + "=" * 70)
    print("INDIVIDUAL STRATEGY METRICS (1x leverage, honest estimates)")
    print("=" * 70)

    individual_metrics = {}
    for col in aligned.columns:
        m = compute_metrics(aligned[col], label=col)
        individual_metrics[col] = m
        print(f"\n  {col}:")
        for k, v in m.items():
            if k != 'label':
                print(f"    {k}: {v}")

    # Correlation matrix
    print("\n" + "=" * 70)
    print("PAIRWISE CORRELATIONS (daily returns)")
    print("=" * 70)
    corr = aligned.corr()
    print(f"\n{corr.round(4).to_string()}")

    # Portfolio allocations
    print("\n" + "=" * 70)
    print("PORTFOLIO ALLOCATION WEIGHTS")
    print("=" * 70)

    allocation_methods = [equal_weight, risk_parity, concentrated_v5, max_sharpe, kelly_optimal]
    leverage_levels = [1.0, 1.5, 2.0, 2.5, 3.0]

    all_results = []

    for alloc_fn in allocation_methods:
        weights, name = alloc_fn(aligned)
        print(f"\n  {name}:")
        for i, col in enumerate(aligned.columns):
            print(f"    {col}: {weights[i]*100:.1f}%")

        # Portfolio daily returns (weighted sum)
        port_rets = (aligned * weights).sum(axis=1)

        for lev in leverage_levels:
            lev_rets = apply_leverage(port_rets, lev)
            label = f"{name} @ {lev}x"
            m = compute_metrics(lev_rets, label=label)
            m['allocation'] = name
            m['leverage'] = lev
            m['weights'] = {col: round(w, 4) for col, w in zip(aligned.columns, weights)}
            all_results.append(m)

    # Display results table
    print("\n" + "=" * 70)
    print("PORTFOLIO RESULTS BY ALLOCATION & LEVERAGE")
    print("=" * 70)

    # Group by allocation
    for alloc_fn in allocation_methods:
        _, name = alloc_fn(aligned)
        print(f"\n--- {name} ---")
        print(f"{'Leverage':>8} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} {'VaR5%':>7} {'WorstYr%':>9} {'Final$':>12}")

        for r in all_results:
            if r['allocation'] == name:
                print(f"{r['leverage']:>7.1f}x {r['CAGR%']:>7.1f} {r['Sharpe']:>7.2f} {r['Sortino']:>8.2f} {r['MaxDD%']:>7.1f} {r['Calmar']:>7.2f} {r['Monthly_VaR_5%']:>7.2f} {r['Worst_Year%']:>9.1f} {r['Final_Equity']:>12,.0f}")

    # Find best options
    print("\n" + "=" * 70)
    print("TOP RECOMMENDATIONS (ranked by risk-adjusted return)")
    print("=" * 70)

    # Filter: MaxDD > -25%, Sharpe > 1.5
    viable = [r for r in all_results if r['MaxDD%'] > -25 and r['Sharpe'] > 1.5]
    viable.sort(key=lambda x: x['Sharpe'], reverse=True)

    print(f"\nFiltered to: MaxDD > -25%, Sharpe > 1.5 ({len(viable)} of {len(all_results)} options)")
    print(f"\n{'Rank':>4} {'Allocation':<35} {'Lev':>4} {'CAGR%':>7} {'Sharpe':>7} {'MaxDD%':>7} {'Calmar':>7} {'Final$':>12}")
    print("-" * 95)

    for i, r in enumerate(viable[:10], 1):
        print(f"{i:>4} {r['allocation']:<35} {r['leverage']:>3.1f}x {r['CAGR%']:>7.1f} {r['Sharpe']:>7.2f} {r['MaxDD%']:>7.1f} {r['Calmar']:>7.2f} {r['Final_Equity']:>12,.0f}")

    # Deployment recommendations — pick DIFFERENT leverage tiers
    print("\n" + "=" * 70)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("=" * 70)

    # Conservative: best Sharpe at 1x leverage (no borrowing risk)
    conservative = [r for r in viable if r['leverage'] == 1.0]
    conservative.sort(key=lambda x: x['Sharpe'], reverse=True)

    # Balanced: best CAGR at 2x leverage
    balanced = [r for r in viable if r['leverage'] == 2.0]
    balanced.sort(key=lambda x: (x['Sharpe'], x['CAGR%']), reverse=True)

    # Aggressive: best CAGR at 3x leverage
    aggressive = [r for r in all_results if r['leverage'] == 3.0]
    aggressive.sort(key=lambda x: x['CAGR%'], reverse=True)

    for profile_name, profile in [("CONSERVATIVE (1x, no leverage)", conservative),
                                   ("BALANCED (2x leverage)", balanced),
                                   ("AGGRESSIVE (3x leverage)", aggressive)]:
        if profile:
            best = profile[0]
            w = best['weights']
            print(f"\n  {profile_name}:")
            print(f"    Allocation: {best['allocation']} at {best['leverage']}x leverage")
            for strat, wt in w.items():
                print(f"      {strat}: {wt*100:.1f}%")
            print(f"    Expected CAGR: {best['CAGR%']:.1f}%")
            print(f"    Expected Sharpe: {best['Sharpe']:.2f}")
            print(f"    Max Drawdown: {best['MaxDD%']:.1f}%")
            print(f"    Calmar: {best['Calmar']:.2f}")
            print(f"    $100K -> ${best['Final_Equity']:,.0f} over {best['Years']:.1f} years")

    # Dollar deployment table
    print("\n" + "=" * 70)
    print("DOLLAR DEPLOYMENT TABLE (using best allocation per tier)")
    print("=" * 70)

    for capital in [100_000, 250_000, 500_000]:
        print(f"\n  === Starting Capital: ${capital:,} ===")
        for tier_name, tier in [("Conservative (1x)", conservative),
                                 ("Balanced (2x)", balanced),
                                 ("Aggressive (3x)", aggressive)]:
            if tier:
                best = tier[0]
                w = best['weights']
                final = capital * (1 + best['CAGR%']/100) ** best['Years']
                worst_dd = capital * abs(best['MaxDD%'])/100
                print(f"\n    {tier_name}: {best['allocation']}")
                for strat, wt in w.items():
                    alloc = capital * wt * best['leverage']
                    print(f"      {strat}: ${alloc:,.0f} notional")
                print(f"      CAGR {best['CAGR%']:.1f}% | Sharpe {best['Sharpe']:.2f} | MaxDD {best['MaxDD%']:.1f}%")
                print(f"      ${capital:,} -> ${final:,.0f} over {best['Years']:.0f}yr | Worst DD: ${worst_dd:,.0f}")

    # Year-by-year analysis for best balanced option
    if balanced:
        best_b = balanced[0]
        w_arr = np.array([best_b['weights'][c] for c in aligned.columns])
        port_rets = (aligned * w_arr).sum(axis=1)
        lev_rets = apply_leverage(port_rets, best_b['leverage'])

        equity = (1 + lev_rets).cumprod()
        yearly = equity.resample('YE').last()
        yearly_rets = yearly.pct_change().dropna()

        print("\n" + "=" * 70)
        print(f"YEAR-BY-YEAR: {best_b['allocation']} @ {best_b['leverage']}x")
        print("=" * 70)
        print(f"\n{'Year':>6} {'Return%':>8} {'Equity':>12}")

        equity_dollar = equity * STARTING_CAPITAL
        yearly_eq = equity_dollar.resample('YE').last()
        yearly_rets2 = yearly_eq.pct_change()

        for dt in yearly_eq.index:
            ret_val = yearly_rets2.get(dt, 0)
            if pd.isna(ret_val):
                # First year: compute from start
                ret_val = yearly_eq.iloc[0] / STARTING_CAPITAL - 1
            print(f"  {dt.year:>4}   {ret_val*100:>7.1f}   ${yearly_eq.loc[dt]:>11,.0f}")

    # Save results
    output = {
        'generated': datetime.now().isoformat(),
        'methodology': {
            'ic_sharpe_source': 'Permutation-based (2.05), NOT inflated backtest (4.93/5.68)',
            'ic_scaling': 'Daily returns scaled to match permutation Sharpe, preserving correlation structure',
            'v5_source': 'Delta ladder study d25 (BS synthetic pricing, acknowledged overestimate)',
            'etf_source': 'ETF Rotation v3 hedged (R1-passing beta-scaled config)',
            'leverage_cost': f'{LEVERAGE_COST*100}% annual on leveraged portion',
            'risk_free_rate': f'{RISK_FREE*100}%',
            'common_period': f'{aligned.index[0].date()} to {aligned.index[-1].date()}',
            'common_days': len(aligned),
        },
        'individual_strategies': individual_metrics,
        'correlations': corr.to_dict(),
        'portfolio_results': all_results,
        'recommendations': {
            'conservative': conservative[0] if conservative else None,
            'balanced': balanced[0] if balanced else None,
            'aggressive': aggressive[0] if aggressive else None,
        }
    }

    with open(f'{OUT}/results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save equity curves for each recommended portfolio
    for profile_name, profile in [('conservative', conservative),
                                   ('balanced', balanced),
                                   ('aggressive', aggressive)]:
        if profile:
            best = profile[0]
            w_arr = np.array([best['weights'][c] for c in aligned.columns])
            port_rets = (aligned * w_arr).sum(axis=1)
            lev_rets = apply_leverage(port_rets, best['leverage'])
            equity = (1 + lev_rets).cumprod() * STARTING_CAPITAL
            eq_df = pd.DataFrame({'date': equity.index, 'equity': equity.values})
            eq_df.to_parquet(f'{OUT}/equity_{profile_name}.parquet', index=False)

    print(f"\n\nResults saved to {OUT}/")
    print("=" * 70)

    return output


if __name__ == '__main__':
    results = run_analysis()
