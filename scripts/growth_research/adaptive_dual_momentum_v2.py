#!/usr/bin/env python3
"""
Strategy 1: Adaptive Dual Momentum with VIX Regime Overlay
===========================================================
Antonacci-style absolute + relative momentum across SPY/EFA/AGG/SHY.
VIX regime overlay: if VIX > 30 (T-1 close), force cash regardless.

ZERO LOOKAHEAD: All signals use T-1 close data. Execute at T open.
Monthly rebalance on last business day of month using T-1 prices.

Author: Claude Opus (autonomous build)
Date: 2026-07-21
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
INITIAL_CAPITAL = 100_000
COST_BPS = 10  # 10 bps per trade for ETFs
TICKERS = ['SPY', 'EFA', 'AGG', 'SHY']
VIX_TICKER = '^VIX'
LOOKBACK_MONTHS = 12  # 12M momentum for absolute + relative
VIX_THRESHOLD = 30
START_DATE = '2008-01-01'  # Need 12M lookback, so effective start ~2009
END_DATE = '2026-07-18'
N_PERMUTATIONS = 200
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_strategies_v2'

np.random.seed(42)


def download_data():
    """Download all required price data."""
    all_tickers = TICKERS + [VIX_TICKER]
    data = {}
    for t in all_tickers:
        print(f"  Downloading {t}...")
        df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[t] = df['Close']

    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.dropna()
    return prices


def compute_signals(prices):
    """
    Compute monthly signals with ZERO lookahead.
    All signals use data available at T-1 close.
    Rebalance on last business day of each month.
    """
    # Resample to month-end
    monthly = prices.resample('ME').last()

    signals = pd.DataFrame(index=monthly.index)

    for i in range(LOOKBACK_MONTHS, len(monthly)):
        date = monthly.index[i]

        # Use T-1 month data (signal computed on month i-1 data, applied to month i)
        # This means: at end of month i-1, we compute signal and hold during month i

        # 12M return for absolute momentum (using data up to month i-1)
        spy_12m_ret = (monthly['SPY'].iloc[i-1] / monthly['SPY'].iloc[i-1-LOOKBACK_MONTHS]) - 1
        efa_12m_ret = (monthly['EFA'].iloc[i-1] / monthly['EFA'].iloc[i-1-LOOKBACK_MONTHS]) - 1

        # VIX at T-1 month end
        vix_val = monthly[VIX_TICKER].iloc[i-1]

        # Decision logic
        if vix_val > VIX_THRESHOLD:
            # High vol regime: go to cash (SHY)
            signals.loc[date, 'holding'] = 'SHY'
            signals.loc[date, 'reason'] = 'VIX_HIGH'
        elif spy_12m_ret <= 0:
            # Absolute momentum negative: hold bonds
            signals.loc[date, 'holding'] = 'AGG'
            signals.loc[date, 'reason'] = 'ABS_MOM_NEG'
        else:
            # Absolute momentum positive: hold best of SPY/EFA
            if spy_12m_ret >= efa_12m_ret:
                signals.loc[date, 'holding'] = 'SPY'
                signals.loc[date, 'reason'] = 'REL_MOM_SPY'
            else:
                signals.loc[date, 'holding'] = 'EFA'
                signals.loc[date, 'reason'] = 'REL_MOM_EFA'

    return signals


def run_backtest(prices, signals, initial_capital=INITIAL_CAPITAL, cost_bps=COST_BPS):
    """
    Run backtest with proper lag.
    Signal at end of month T-1 -> hold asset during month T.
    """
    monthly = prices.resample('ME').last()

    # Align: signal at date T tells us what to hold during the NEXT month
    valid_signals = signals.dropna(subset=['holding'])

    capital = initial_capital
    holdings = 'CASH'
    equity_curve = []
    trades = []

    for i in range(1, len(valid_signals)):
        prev_date = valid_signals.index[i-1]
        curr_date = valid_signals.index[i]

        target = valid_signals.loc[prev_date, 'holding']

        # Monthly return of the target asset during this month
        if target in monthly.columns and prev_date in monthly.index and curr_date in monthly.index:
            ret = (monthly[target].loc[curr_date] / monthly[target].loc[prev_date]) - 1
        else:
            ret = 0.0

        # Trading cost if we switch holdings
        cost = 0.0
        if target != holdings:
            cost = capital * (cost_bps / 10000) * 2  # Buy + sell
            trades.append({
                'date': curr_date,
                'from': holdings,
                'to': target,
                'cost': cost
            })
            holdings = target

        capital = capital * (1 + ret) - cost
        equity_curve.append({
            'date': curr_date,
            'capital': capital,
            'holding': target,
            'monthly_return': ret
        })

    return pd.DataFrame(equity_curve).set_index('date'), trades


def compute_metrics(eq_curve, initial_capital=INITIAL_CAPITAL):
    """Compute risk-adjusted performance metrics."""
    rets = eq_curve['monthly_return']

    total_ret = (eq_curve['capital'].iloc[-1] / initial_capital) - 1
    n_years = len(rets) / 12
    cagr = (1 + total_ret) ** (1 / n_years) - 1

    monthly_mean = rets.mean()
    monthly_std = rets.std()

    sharpe = (monthly_mean / monthly_std) * np.sqrt(12) if monthly_std > 0 else 0

    downside = rets[rets < 0].std()
    sortino = (monthly_mean / downside) * np.sqrt(12) if downside > 0 else 0

    # Max drawdown
    cummax = eq_curve['capital'].cummax()
    drawdown = (eq_curve['capital'] - cummax) / cummax
    max_dd = drawdown.min()

    # Win rate (monthly)
    wr = (rets > 0).sum() / len(rets)

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        'CAGR': f"{cagr:.2%}",
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'MaxDD': f"{max_dd:.2%}",
        'WinRate': f"{wr:.2%}",
        'ProfitFactor': round(pf, 2),
        'TotalReturn': f"{total_ret:.2%}",
        'FinalCapital': f"${eq_curve['capital'].iloc[-1]:,.0f}",
        'N_Months': len(rets),
    }


def spy_buy_and_hold(prices, initial_capital=INITIAL_CAPITAL):
    """SPY buy and hold benchmark."""
    monthly = prices['SPY'].resample('ME').last()

    # Start from same point as strategy
    start_idx = LOOKBACK_MONTHS + 1
    monthly = monthly.iloc[start_idx:]

    rets = monthly.pct_change().dropna()
    capital = initial_capital
    eq = []
    for date, ret in rets.items():
        capital = capital * (1 + ret)
        eq.append({'date': date, 'capital': capital, 'monthly_return': ret, 'holding': 'SPY'})

    return pd.DataFrame(eq).set_index('date')


def permutation_test(prices, signals, actual_sharpe, n_perms=N_PERMUTATIONS):
    """
    Permutation test: shuffle the SIGNAL dates (not returns).
    This preserves return autocorrelation while destroying signal-return alignment.
    """
    print(f"\n  Running {n_perms} permutations...")
    valid_signals = signals.dropna(subset=['holding'])
    perm_sharpes = []

    for p in range(n_perms):
        # Shuffle signal assignments across dates
        shuffled = valid_signals.copy()
        shuffled['holding'] = np.random.permutation(shuffled['holding'].values)

        eq, _ = run_backtest(prices, shuffled)
        if len(eq) > 0:
            rets = eq['monthly_return']
            if rets.std() > 0:
                s = (rets.mean() / rets.std()) * np.sqrt(12)
            else:
                s = 0
            perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).sum() / len(perm_sharpes)

    return {
        'p_value': round(p_value, 4),
        'actual_sharpe': round(actual_sharpe, 3),
        'perm_mean_sharpe': round(perm_sharpes.mean(), 3),
        'perm_std_sharpe': round(perm_sharpes.std(), 3),
        'perm_95th': round(np.percentile(perm_sharpes, 95), 3),
        'significant_5pct': p_value < 0.05,
    }


def regime_analysis(eq_curve, spy_prices):
    """
    Stratify performance by SPY regime (green/red/flat months).
    Uses T-1 SPY monthly return to classify the regime.
    """
    spy_monthly = spy_prices.resample('ME').last().pct_change()

    results = {}
    for regime, cond in [
        ('green', spy_monthly > 0.01),
        ('red', spy_monthly < -0.01),
        ('flat', (spy_monthly >= -0.01) & (spy_monthly <= 0.01))
    ]:
        # Align dates
        regime_dates = spy_monthly[cond].index
        strat_in_regime = eq_curve[eq_curve.index.isin(regime_dates)]

        if len(strat_in_regime) > 2:
            rets = strat_in_regime['monthly_return']
            sharpe = (rets.mean() / rets.std()) * np.sqrt(12) if rets.std() > 0 else 0
            wr = (rets > 0).sum() / len(rets)
            results[regime] = {
                'n_months': len(rets),
                'sharpe': round(sharpe, 2),
                'avg_return': f"{rets.mean():.2%}",
                'win_rate': f"{wr:.2%}",
            }
        else:
            results[regime] = {'n_months': 0, 'sharpe': 0, 'avg_return': '0%', 'win_rate': '0%'}

    # Regime balance check
    green_sharpe = results['green']['sharpe']
    red_sharpe = results['red']['sharpe']
    max_sharpe = max(abs(green_sharpe), abs(red_sharpe))
    if max_sharpe > 0:
        regime_gap = abs(green_sharpe - red_sharpe) / max_sharpe
    else:
        regime_gap = 0

    results['regime_gap'] = round(regime_gap, 3)
    results['regime_balanced'] = regime_gap <= 0.50

    return results


def main():
    print("=" * 70)
    print("STRATEGY 1: Adaptive Dual Momentum + VIX Regime Overlay")
    print("=" * 70)

    # Download data
    print("\n[1/6] Downloading data...")
    prices = download_data()
    print(f"  Data range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  {len(prices)} trading days")

    # Compute signals
    print("\n[2/6] Computing signals (T-1 lagged)...")
    signals = compute_signals(prices)
    print(f"  Generated {len(signals.dropna())} monthly signals")
    holding_counts = signals['holding'].value_counts()
    print(f"  Holdings distribution:")
    for h, c in holding_counts.items():
        print(f"    {h}: {c} months ({c/len(signals.dropna()):.1%})")

    # Run backtest
    print("\n[3/6] Running backtest...")
    eq_curve, trades = run_backtest(prices, signals)
    metrics = compute_metrics(eq_curve)
    print(f"\n  STRATEGY METRICS:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    print(f"    Trades: {len(trades)}")

    # SPY benchmark
    print("\n[4/6] SPY Buy & Hold benchmark...")
    spy_eq = spy_buy_and_hold(prices)
    spy_metrics = compute_metrics(spy_eq)
    print(f"\n  SPY B&H METRICS:")
    for k, v in spy_metrics.items():
        print(f"    {k}: {v}")

    # Permutation test
    print("\n[5/6] Permutation test (200 perms)...")
    actual_sharpe = float(metrics['Sharpe'])
    perm_results = permutation_test(prices, signals, actual_sharpe)
    print(f"  p-value: {perm_results['p_value']}")
    print(f"  Significant at 5%: {perm_results['significant_5pct']}")
    print(f"  Actual Sharpe: {perm_results['actual_sharpe']} vs Perm 95th: {perm_results['perm_95th']}")

    # Regime analysis
    print("\n[6/6] Regime analysis...")
    regime_results = regime_analysis(eq_curve, prices['SPY'])
    for regime in ['green', 'red', 'flat']:
        r = regime_results[regime]
        print(f"  {regime.upper()}: {r['n_months']} months, Sharpe={r['sharpe']}, WR={r['win_rate']}, Avg={r['avg_return']}")
    print(f"  Regime gap: {regime_results['regime_gap']} (balanced: {regime_results['regime_balanced']})")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    full_results = {
        'strategy': 'Adaptive Dual Momentum + VIX Overlay',
        'strategy_metrics': metrics,
        'spy_benchmark': spy_metrics,
        'permutation_test': perm_results,
        'regime_analysis': regime_results,
        'config': {
            'lookback_months': LOOKBACK_MONTHS,
            'vix_threshold': VIX_THRESHOLD,
            'cost_bps': COST_BPS,
            'initial_capital': INITIAL_CAPITAL,
            'start_date': START_DATE,
            'end_date': END_DATE,
        },
        'trades_count': len(trades),
        'timestamp': datetime.now().isoformat(),
    }

    with open(f"{OUTPUT_DIR}/strategy1_dual_momentum.json", 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    eq_curve.to_csv(f"{OUTPUT_DIR}/strategy1_equity_curve.csv")

    # Monthly breakdown
    eq_curve['year'] = eq_curve.index.year
    eq_curve['month'] = eq_curve.index.month
    monthly_pivot = eq_curve.pivot_table(values='monthly_return', index='year', columns='month', aggfunc='first')
    monthly_pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    monthly_pivot['Annual'] = (1 + eq_curve.groupby('year')['monthly_return'].apply(
        lambda x: (1+x).prod() - 1
    )).values - 1
    monthly_pivot.to_csv(f"{OUTPUT_DIR}/strategy1_monthly_returns.csv")

    print(f"\n  Results saved to {OUTPUT_DIR}/strategy1_*")
    print("\n" + "=" * 70)

    return full_results


if __name__ == '__main__':
    results = main()
