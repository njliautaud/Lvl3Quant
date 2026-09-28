#!/usr/bin/env python3
"""
REAL Combined Portfolio Backtest — No Synthetic Data
=====================================================
Takes actual monthly return series from each validated strategy,
finds common overlap period, computes REAL correlations, and runs
walk-forward portfolio optimization.

NO synthetic returns. NO assumed correlations. NO hand-coded inputs.
Only actual backtest output from individual strategy scripts.

Output: True combined portfolio metrics with honest caveats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/home/jupiter/Lvl3Quant/output/growth_research')
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/real_combined_portfolio')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_strategy_returns():
    """Load actual return series from each strategy's backtest output."""
    strategies = {}

    # 1. CTA Trend Following — CSV with daily returns (index is date, col is '0')
    try:
        df = pd.read_csv(BASE / 'trend_following_returns.csv', index_col=0, parse_dates=True)
        df = df.iloc[:, 0]
        # Resample to monthly
        monthly = (1 + df).resample('ME').prod() - 1
        strategies['CTA_Trend'] = monthly
        print(f"  CTA Trend: {len(monthly)} months ({monthly.index[0].strftime('%Y-%m')} to {monthly.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  CTA Trend: FAILED ({e})")

    # 2. Commodity Trend — parquet with monthly returns
    try:
        df = pd.read_parquet(BASE.parent / 'ml_commodity_trend' / 'portfolio_returns.parquet')
        if 'date' in df.columns:
            df = df.set_index('date')
        ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
        if ret_col:
            s = df[ret_col[0]]
        else:
            s = df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        strategies['Commodity_Trend'] = s
        print(f"  Commodity Trend: {len(s)} months ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  Commodity Trend: FAILED ({e})")

    # 3. Sector Rotation — CSV (index is date, col is '0')
    try:
        df = pd.read_csv(BASE / 'sector_momentum_returns.csv', index_col=0, parse_dates=True)
        df = df.iloc[:, 0]
        if len(df) > 300:  # Daily data, resample
            monthly = (1 + df).resample('ME').prod() - 1
        else:
            monthly = df
            monthly.index = pd.to_datetime(monthly.index)
        strategies['Sector_Rotation'] = monthly
        print(f"  Sector Rotation: {len(monthly)} months ({monthly.index[0].strftime('%Y-%m')} to {monthly.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  Sector Rotation: FAILED ({e})")

    # 4. Currency Carry — parquet
    try:
        df = pd.read_parquet(BASE.parent / 'ml_currency_carry' / 'portfolio_returns.parquet')
        if 'date' in df.columns:
            df = df.set_index('date')
        ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
        s = df[ret_col[0]] if ret_col else df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        strategies['Currency_Carry'] = s
        print(f"  Currency Carry: {len(s)} months ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  Currency Carry: FAILED ({e})")

    # 5. Tail Risk Hedging — parquet
    try:
        df = pd.read_parquet(BASE.parent / 'ml_tail_risk_hedging' / 'portfolio_returns.parquet')
        if 'date' in df.columns:
            df = df.set_index('date')
        ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
        s = df[ret_col[0]] if ret_col else df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        strategies['Tail_Risk'] = s
        print(f"  Tail Risk: {len(s)} months ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  Tail Risk: FAILED ({e})")

    # 6. Bond Duration — parquet
    try:
        df = pd.read_parquet(BASE.parent / 'ml_bond_duration_timing' / 'portfolio_returns.parquet')
        if 'date' in df.columns:
            df = df.set_index('date')
        ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
        s = df[ret_col[0]] if ret_col else df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        strategies['Bond_Duration'] = s
        print(f"  Bond Duration: {len(s)} months ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  Bond Duration: FAILED ({e})")

    # 7. Carry+Momentum — parquet
    try:
        df = pd.read_parquet(BASE.parent / 'ml_carry_momentum' / 'portfolio_returns.parquet')
        if 'date' in df.columns:
            df = df.set_index('date')
        ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
        s = df[ret_col[0]] if ret_col else df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        strategies['Carry_Momentum'] = s
        print(f"  Carry+Momentum: {len(s)} months ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  Carry+Momentum: FAILED ({e})")

    # 8. Vol Breakout — CSV equity curve
    try:
        df = pd.read_csv(BASE.parent / 'ml_vol_breakout' / 'equity_curve.csv', parse_dates=['date'])
        df = df.set_index('date')
        # Equity curve → returns
        if 'equity' in df.columns:
            eq = df['equity']
        elif 'value' in df.columns:
            eq = df['value']
        else:
            eq = df.iloc[:, 0]
        daily_ret = eq.pct_change().dropna()
        monthly = (1 + daily_ret).resample('ME').prod() - 1
        strategies['Vol_Breakout'] = monthly
        print(f"  Vol Breakout: {len(monthly)} months ({monthly.index[0].strftime('%Y-%m')} to {monthly.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  Vol Breakout: FAILED ({e})")

    # 9. Thematic Rotation — parquet
    try:
        df = pd.read_parquet(BASE.parent / 'ml_thematic_rotation' / 'portfolio_returns.parquet')
        if 'date' in df.columns:
            df = df.set_index('date')
        ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
        s = df[ret_col[0]] if ret_col else df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        strategies['Thematic_Rotation'] = s
        print(f"  Thematic Rotation: {len(s)} months ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
    except Exception as e:
        print(f"  Thematic Rotation: FAILED ({e})")

    # 10. Gold/Silver — check for results
    try:
        p = BASE.parent / 'ml_gold_silver'
        files = list(p.glob('*.parquet')) + list(p.glob('*.csv'))
        if files:
            for f in files:
                if 'return' in f.name.lower() or 'portfolio' in f.name.lower():
                    if f.suffix == '.parquet':
                        df = pd.read_parquet(f)
                    else:
                        df = pd.read_csv(f, parse_dates=[0])
                    if 'date' in df.columns:
                        df = df.set_index('date')
                    ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
                    s = df[ret_col[0]] if ret_col else df.iloc[:, 0]
                    s.index = pd.to_datetime(s.index)
                    strategies['Gold_Silver'] = s
                    print(f"  Gold/Silver: {len(s)} months ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
                    break
        if 'Gold_Silver' not in strategies:
            print(f"  Gold/Silver: NO RETURN DATA FOUND")
    except Exception as e:
        print(f"  Gold/Silver: FAILED ({e})")

    # 11. Yield Curve — check for results
    try:
        p = BASE.parent / 'ml_yield_curve'
        files = list(p.glob('*.parquet')) + list(p.glob('*.csv'))
        found = False
        for f in sorted(files, key=lambda x: x.stat().st_size, reverse=True):
            if 'return' in f.name.lower() or 'portfolio' in f.name.lower():
                if f.suffix == '.parquet':
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f, parse_dates=[0])
                if 'date' in df.columns:
                    df = df.set_index('date')
                ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
                if ret_col:
                    s = df[ret_col[0]]
                    s.index = pd.to_datetime(s.index)
                    strategies['Yield_Curve'] = s
                    print(f"  Yield Curve: {len(s)} months ({s.index[0].strftime('%Y-%m')} to {s.index[-1].strftime('%Y-%m')})")
                    found = True
                    break
        if not found:
            print(f"  Yield Curve: NO RETURN DATA FOUND (files: {[f.name for f in files[:5]]})")
    except Exception as e:
        print(f"  Yield Curve: FAILED ({e})")

    # 12. Stat Arb — check for results
    try:
        p = BASE.parent / 'ml_stat_arb'
        if not p.exists():
            p = BASE / 'stat_arb'
        files = list(p.glob('*.parquet')) + list(p.glob('*.csv')) if p.exists() else []
        found = False
        for f in files:
            if 'return' in f.name.lower() or 'portfolio' in f.name.lower():
                if f.suffix == '.parquet':
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f, parse_dates=[0])
                if 'date' in df.columns:
                    df = df.set_index('date')
                ret_col = [c for c in df.columns if 'return' in c.lower() or 'ret' in c.lower()]
                if ret_col:
                    s = df[ret_col[0]]
                    s.index = pd.to_datetime(s.index)
                    strategies['Stat_Arb'] = s
                    print(f"  Stat Arb: {len(s)} months")
                    found = True
                    break
        if not found:
            print(f"  Stat Arb: NO RETURN DATA FOUND")
    except Exception as e:
        print(f"  Stat Arb: FAILED ({e})")

    return strategies


def compute_real_metrics(returns_series, name="Strategy"):
    """Compute honest metrics from actual return series."""
    r = returns_series.dropna()
    if len(r) < 12:
        return None

    n_months = len(r)
    n_years = n_months / 12

    # CAGR
    cum = (1 + r).prod()
    cagr = cum ** (12 / n_months) - 1

    # Annualized return and vol (monthly → annual)
    mean_monthly = r.mean()
    std_monthly = r.std()
    ann_ret = mean_monthly * 12
    ann_vol = std_monthly * np.sqrt(12)

    # Sharpe (annualized, no risk-free for simplicity)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = r[r < 0]
    down_vol = downside.std() * np.sqrt(12) if len(downside) > 0 else 0.001
    sortino = ann_ret / down_vol

    # Max Drawdown
    cum_ret = (1 + r).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = cum_ret / rolling_max - 1
    max_dd = drawdown.min()

    # Win rate
    wr = (r > 0).mean()

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'name': name,
        'n_months': n_months,
        'n_years': round(n_years, 1),
        'CAGR': round(cagr * 100, 1),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'MaxDD': round(max_dd * 100, 1),
        'WinRate': round(wr * 100, 1),
        'ProfitFactor': round(pf, 2),
        'Calmar': round(calmar, 2),
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
    }


def walk_forward_portfolio(returns_df, lookback=36, rebal_freq=3):
    """
    Walk-forward portfolio optimization.
    - lookback: months of history for weight estimation
    - rebal_freq: rebalance every N months
    Returns: combined portfolio monthly return series + weight history
    """
    dates = returns_df.index
    n_strats = returns_df.shape[1]

    portfolio_returns = []
    weight_history = []

    for i in range(lookback, len(dates)):
        # Only rebalance every rebal_freq months
        if (i - lookback) % rebal_freq == 0:
            # Use last `lookback` months to estimate weights
            train = returns_df.iloc[i-lookback:i]

            # Drop strategies with insufficient data in training window
            valid = train.dropna(axis=1, thresh=int(lookback * 0.8))
            if valid.shape[1] < 2:
                continue

            mu = valid.mean() * 12  # annualized
            cov = valid.cov() * 12  # annualized

            # Method 1: Risk Parity (more robust than MVO)
            vols = np.sqrt(np.diag(cov.values))
            inv_vol = 1.0 / (vols + 1e-8)
            weights_rp = inv_vol / inv_vol.sum()

            # Method 2: Min Variance
            try:
                cov_inv = np.linalg.pinv(cov.values)
                ones = np.ones(len(valid.columns))
                weights_mv = cov_inv @ ones / (ones @ cov_inv @ ones)
                weights_mv = np.maximum(weights_mv, 0)  # No shorts
                weights_mv = weights_mv / weights_mv.sum() if weights_mv.sum() > 0 else weights_rp
            except:
                weights_mv = weights_rp

            # Method 3: Equal Weight (honest benchmark)
            weights_ew = np.ones(len(valid.columns)) / len(valid.columns)

            current_weights = {
                'rp': dict(zip(valid.columns, weights_rp)),
                'mv': dict(zip(valid.columns, weights_mv)),
                'ew': dict(zip(valid.columns, weights_ew)),
            }

        # Apply weights to this month's returns
        month_ret = returns_df.iloc[i]

        for method in ['rp', 'mv', 'ew']:
            w = current_weights[method]
            port_ret = sum(month_ret.get(s, 0) * w.get(s, 0) for s in w.keys())
            portfolio_returns.append({
                'date': dates[i],
                'method': method,
                'return': port_ret,
            })

        weight_history.append({
            'date': dates[i],
            **{f'w_{k}': round(v, 3) for k, v in current_weights['rp'].items()},
        })

    return pd.DataFrame(portfolio_returns), pd.DataFrame(weight_history)


def main():
    print("=" * 70)
    print("REAL COMBINED PORTFOLIO BACKTEST — NO SYNTHETIC DATA")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # 1. Load actual return series
    print("\n[1/5] Loading strategy return series...")
    strategies = load_strategy_returns()

    if len(strategies) < 3:
        print(f"\nERROR: Only {len(strategies)} strategies loaded. Need at least 3.")
        return

    print(f"\nLoaded {len(strategies)} strategies")

    # 2. Find common overlap period
    print("\n[2/5] Finding common overlap period...")

    # Normalize all indices to month-period (YYYY-MM) to fix alignment issues
    # Different sources use different month-end dates (Jan 31 vs Jan 30 etc.)
    normalized = {}
    for name, s in strategies.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        # Convert to period (month) then back to timestamp (month-end)
        s.index = s.index.to_period('M').to_timestamp('M')
        # De-duplicate in case of multiple values in same month
        s = s.groupby(s.index).first()
        normalized[name] = s

    combined = pd.DataFrame(normalized)
    combined = combined.sort_index()

    # Show coverage
    print("\nData coverage by strategy:")
    for col in combined.columns:
        valid = combined[col].dropna()
        if len(valid) > 0:
            print(f"  {col}: {valid.index[0].strftime('%Y-%m')} to {valid.index[-1].strftime('%Y-%m')} ({len(valid)} months)")

    # Find max overlap (strategies that have data at same time)
    print("\nOverlap analysis:")
    for min_strats in [len(combined.columns), max(3, len(combined.columns)-2), 3]:
        overlap = combined.dropna(thresh=min_strats)
        if len(overlap) >= 24:
            print(f"  {min_strats}+ strategies: {len(overlap)} months overlap ({overlap.index[0].strftime('%Y-%m')} to {overlap.index[-1].strftime('%Y-%m')})")
            break

    # Find the period where we have the MOST strategies overlapping
    # Count how many strategies have data at each date
    coverage = combined.notna().sum(axis=1)
    print(f"\n  Coverage by date range:")
    for thresh in sorted(set(coverage.values), reverse=True):
        mask = coverage >= thresh
        if mask.sum() >= 12:
            idx = combined.index[mask]
            print(f"    {thresh}+ strategies: {mask.sum()} months ({idx[0].strftime('%Y-%m')} to {idx[-1].strftime('%Y-%m')})")

    # Use period with at least 5 strategies (or max available)
    min_thresh = min(5, len(combined.columns))
    good_mask = coverage >= min_thresh
    if good_mask.sum() < 24:
        min_thresh = 3
        good_mask = coverage >= min_thresh
    combined_focus = combined[good_mask].copy()
    # Only keep strategies that have data in this focused period
    active_strats = [c for c in combined_focus.columns if combined_focus[c].notna().sum() >= 12]
    combined_focus = combined_focus[active_strats]
    print(f"\n  FOCUSED ANALYSIS: {len(active_strats)} strategies over {len(combined_focus)} months")
    print(f"  Period: {combined_focus.index[0].strftime('%Y-%m')} to {combined_focus.index[-1].strftime('%Y-%m')}")
    print("  Missing values in focused period filled with 0 (cash — conservative)")

    # 3. Compute REAL correlation matrix
    print("\n[3/5] Real correlation matrix (from actual overlapping returns)...")

    # Only compute correlation where data actually overlaps
    corr_matrix = combined_focus.corr(min_periods=12)
    print("\nPairwise correlations (measured, not assumed):")
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            c = corr_matrix.iloc[i, j]
            overlap_count = combined_focus[[cols[i], cols[j]]].dropna().shape[0]
            if not np.isnan(c):
                print(f"  {cols[i]:20s} ↔ {cols[j]:20s}: {c:+.3f} ({overlap_count} months overlap)")

    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    avg_corr = avg_corr[~np.isnan(avg_corr)]
    print(f"\n  Average pairwise correlation: {np.mean(avg_corr):.3f}")
    print(f"  Max correlation: {np.max(avg_corr):.3f}")
    print(f"  Min correlation: {np.min(avg_corr):.3f}")

    # 4. Individual strategy metrics (RE-COMPUTED from actual data)
    print("\n[4/5] Individual strategy metrics (re-computed from actual returns)...")
    print(f"\n{'Strategy':<22s} {'Months':>6s} {'CAGR':>7s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'WR':>6s} {'PF':>6s}")
    print("-" * 75)

    individual_metrics = {}
    for name, returns in strategies.items():
        m = compute_real_metrics(returns, name)
        if m:
            individual_metrics[name] = m
            print(f"{name:<22s} {m['n_months']:>6d} {m['CAGR']:>6.1f}% {m['Sharpe']:>7.2f} {m['Sortino']:>8.2f} {m['MaxDD']:>6.1f}% {m['WinRate']:>5.1f}% {m['ProfitFactor']:>5.2f}")

    # 5. Walk-forward combined portfolio
    print("\n[5/5] Walk-forward portfolio optimization (36-month lookback, 3-month rebalance)...")

    # Fill NaN with 0 for portfolio construction (conservative — missing = cash)
    combined_filled = combined_focus.fillna(0)

    port_returns, weight_hist = walk_forward_portfolio(combined_filled, lookback=36, rebal_freq=3)

    if len(port_returns) == 0:
        print("ERROR: Insufficient data for walk-forward optimization")
        return

    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO RESULTS — WALK-FORWARD (REAL DATA)")
    print("=" * 70)

    for method, label in [('ew', 'Equal Weight'), ('rp', 'Risk Parity'), ('mv', 'Min Variance')]:
        pr = port_returns[port_returns['method'] == method].set_index('date')['return']
        m = compute_real_metrics(pr, label)
        if m:
            print(f"\n  {label}:")
            print(f"    Months: {m['n_months']} ({m['n_years']} years)")
            print(f"    CAGR: {m['CAGR']:.1f}%")
            print(f"    Sharpe: {m['Sharpe']:.2f}")
            print(f"    Sortino: {m['Sortino']:.2f}")
            print(f"    MaxDD: {m['MaxDD']:.1f}%")
            print(f"    WinRate: {m['WinRate']:.1f}%")
            print(f"    ProfitFactor: {m['ProfitFactor']:.2f}")
            print(f"    Calmar: {m['Calmar']:.2f}")

    # Caveats
    print("\n" + "=" * 70)
    print("CAVEATS (read before trusting any number above)")
    print("=" * 70)
    print("""
  1. OVERLAP: Not all strategies cover the same dates. Missing months
     are filled with 0% return (cash assumption). This is conservative
     but means earlier periods have fewer active strategies.

  2. LOOKBACK BIAS: Walk-forward optimization uses past 36 months to
     set weights. In real life, you wouldn't have known which strategies
     to include — survivorship bias in strategy SELECTION is not removed.

  3. REBALANCING COSTS: No transaction costs for monthly rebalancing.
     Real costs would reduce returns by ~0.5-1% annually.

  4. LEVERAGE: All strategies are 1x (no leverage). The Sharpe IS
     achievable at 1x but the CAGR assumes full capital allocation.

  5. CORRELATION REGIME: Correlations can spike during crises. The
     walk-forward approach partially addresses this but 36 months
     may not capture tail events.

  6. INDIVIDUAL STRATEGY RISK: Each strategy was individually validated
     but some have short histories (Bond Duration: ~49 months).
     Short histories = less confidence in their metrics.

  7. THIS IS NOT LIVE PERFORMANCE. This is a backtest of backtests.
     Real performance will be worse due to execution, slippage,
     costs, and the strategies not perfectly replicating their
     backtested signals in real-time.
""")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_strategies': len(strategies),
        'strategies_used': list(strategies.keys()),
        'individual_metrics': individual_metrics,
        'correlation_matrix': corr_matrix.to_dict(),
        'avg_correlation': round(float(np.mean(avg_corr)), 3),
    }

    # Add portfolio results
    for method in ['ew', 'rp', 'mv']:
        pr = port_returns[port_returns['method'] == method].set_index('date')['return']
        m = compute_real_metrics(pr, method)
        if m:
            results[f'portfolio_{method}'] = m

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save returns
    port_returns.to_csv(OUTPUT_DIR / 'portfolio_returns.csv', index=False)
    combined_focus.to_csv(OUTPUT_DIR / 'strategy_returns.csv')
    weight_hist.to_csv(OUTPUT_DIR / 'weight_history.csv', index=False)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
