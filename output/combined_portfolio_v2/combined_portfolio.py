#!/usr/bin/env python3
"""
Combined Portfolio Builder v2
=============================
Combines validated strategies with fixed allocations, monthly rebalance,
T-1 signals, 10 bps transaction costs. No optimized weights (anti-lookahead).

Strategies:
1. Trend CTA (6m momentum, 8 ETFs) - from portfolio_v2.csv
2. UPRO + 200SMA with Regime Overlay - reconstructed from v2 signals
3. VIX Spike Buying - constructed: long SPY when VIX > 30 (T-1), hold 3 months
4. Wheel Income - from equity_t1.parquet

Combos:
A. Growth: 50% CTA + 30% UPRO/Overlay + 20% VIX Spike
B. Balanced: 40% CTA + 25% UPRO/Overlay + 15% VIX Spike + 20% Wheel
C. Conservative: 60% CTA + 40% Wheel
D. Two-strategy: 60% CTA + 40% UPRO/Overlay
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

BASE = '/home/jupiter/Lvl3Quant'
OUTPUT_DIR = f'{BASE}/output/combined_portfolio_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COST_BPS = 10  # 10 bps transaction costs


# ============================================================
# STRATEGY 1: Trend CTA (load from file)
# ============================================================
def load_trend_cta():
    """Load trend CTA portfolio value series."""
    df = pd.read_csv(f'{BASE}/output/trend_cta_v1/portfolio_v2_6m_mom_eqwt.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df.columns = ['portfolio_value']
    # Convert to daily returns
    df['returns'] = df['portfolio_value'].pct_change()
    print(f"Trend CTA: {df.index.min().date()} to {df.index.max().date()}, {len(df)} days")
    return df['returns'].dropna()


# ============================================================
# STRATEGY 2: UPRO + Regime Overlay (reconstruct from signals + prices)
# ============================================================
def load_upro_overlay():
    """Reconstruct UPRO+Overlay(SHY) daily returns from v2 signals."""
    signals = pd.read_parquet(f'{BASE}/output/regime_overlay_v1/v2_regime_signals.parquet')
    prices = pd.read_parquet(f'{BASE}/output/regime_overlay_v1/raw_prices.parquet')

    # Daily returns for UPRO and SHY
    upro_ret = prices['UPRO'].pct_change()
    shy_ret = prices['SHY'].pct_change()

    # equity_exposure is T-1 lagged already in the signal file
    # Use it directly: equity_exposure * UPRO_ret + bond_exposure * SHY_ret
    equity_exp = signals['equity_exposure']
    bond_exp = signals['bond_exposure']

    # Align
    common = upro_ret.index.intersection(equity_exp.index)
    common = common[common >= '2010-06-30']  # UPRO inception ~June 2009, give it a year

    upro_r = upro_ret.loc[common]
    shy_r = shy_ret.loc[common]
    eq_exp = equity_exp.loc[common]
    bd_exp = bond_exp.loc[common]

    # Apply transaction costs on rebalance days (when exposure changes)
    exp_change = eq_exp.diff().abs()
    tc = exp_change * (COST_BPS / 10000)

    overlay_ret = eq_exp * upro_r + bd_exp * shy_r - tc
    overlay_ret = overlay_ret.dropna()

    print(f"UPRO+Overlay: {overlay_ret.index.min().date()} to {overlay_ret.index.max().date()}, {len(overlay_ret)} days")
    return overlay_ret


# ============================================================
# STRATEGY 3: VIX Spike Buying (construct from scratch)
# ============================================================
def construct_vix_spike(prices_df=None):
    """
    Buy SPY when VIX crosses above 30 (T-1 signal), hold 3 months (~63 trading days).
    If VIX stays above 30, stay in. Exit after 63 days if VIX < 30.
    When not in a spike trade, hold cash (SHY).
    """
    if prices_df is None:
        print("Downloading VIX and SPY data for VIX spike strategy...")
        spy = yf.download('SPY', start='2006-01-01', end='2026-07-21', progress=False)
        vix = yf.download('^VIX', start='2006-01-01', end='2026-07-21', progress=False)
        shy = yf.download('SHY', start='2006-01-01', end='2026-07-21', progress=False)
        for d in [spy, vix, shy]:
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)

        prices_df = pd.DataFrame({
            'SPY': spy['Close'],
            'VIX': vix['Close'],
            'SHY': shy['Close']
        }).dropna()

    spy_ret = prices_df['SPY'].pct_change()
    shy_ret = prices_df['SHY'].pct_change()
    vix_level = prices_df['VIX']

    # T-1 signal: use yesterday's VIX
    vix_t1 = vix_level.shift(1)

    # Track position: 1 = long SPY (spike trade), 0 = cash (SHY)
    position = pd.Series(0.0, index=prices_df.index)
    entry_date = None
    hold_days = 63  # ~3 months

    for i in range(1, len(prices_df)):
        date = prices_df.index[i]
        prev_vix = vix_t1.iloc[i]

        if pd.isna(prev_vix):
            continue

        if position.iloc[i-1] == 0:
            # Not in trade - check for entry
            if prev_vix >= 30:
                position.iloc[i] = 1.0
                entry_date = date
        else:
            # In trade - check for exit
            days_held = (date - entry_date).days if entry_date else 999
            if days_held >= hold_days * 1.5 and prev_vix < 30:  # ~95 calendar days
                position.iloc[i] = 0.0
                entry_date = None
            else:
                position.iloc[i] = 1.0

    # Returns: SPY when in trade, SHY when not
    strat_ret = position * spy_ret + (1 - position) * shy_ret

    # Transaction costs on entries/exits
    trades = position.diff().abs()
    tc = trades * (COST_BPS / 10000)
    strat_ret = strat_ret - tc

    strat_ret = strat_ret.dropna()

    n_events = (position.diff() == 1).sum()
    print(f"VIX Spike: {strat_ret.index.min().date()} to {strat_ret.index.max().date()}, "
          f"{len(strat_ret)} days, {n_events} spike events")
    return strat_ret


# ============================================================
# STRATEGY 4: Wheel Income (load from file)
# ============================================================
def load_wheel():
    """Load wheel strategy T-1 equity curve."""
    df = pd.read_parquet(f'{BASE}/output/wheel_backtest_v2/equity_t1.parquet')
    returns = df['returns'].dropna()
    print(f"Wheel: {returns.index.min().date()} to {returns.index.max().date()}, {len(returns)} days")
    return returns


# ============================================================
# SPY Benchmark
# ============================================================
def load_spy_benchmark():
    """Load SPY as benchmark."""
    print("Downloading SPY benchmark...")
    spy = yf.download('SPY', start='2006-01-01', end='2026-07-21', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy_ret = spy['Close'].pct_change().dropna()
    spy_ret.index = pd.to_datetime(spy_ret.index)
    if spy_ret.index.tz is not None:
        spy_ret.index = spy_ret.index.tz_localize(None)
    print(f"SPY: {spy_ret.index.min().date()} to {spy_ret.index.max().date()}, {len(spy_ret)} days")
    return spy_ret


# ============================================================
# Portfolio Construction with Monthly Rebalancing
# ============================================================
def build_combined_portfolio(strategy_returns: dict, weights: dict, name: str):
    """
    Combine strategy return series with fixed weights, monthly rebalance.
    T-1 safe: weights are fixed (no optimization), rebalance is calendar-based.
    10 bps costs applied at each monthly rebalance.
    """
    # Find common date range
    all_indices = [r.index for r in strategy_returns.values()]
    common_start = max(idx.min() for idx in all_indices)
    common_end = min(idx.max() for idx in all_indices)

    print(f"\n{name}: common period {common_start.date()} to {common_end.date()}")

    # Align all series to common dates
    aligned = {}
    for sname, rets in strategy_returns.items():
        s = rets.loc[common_start:common_end].copy()
        aligned[sname] = s

    # Get union of all dates
    all_dates = sorted(set().union(*[set(s.index) for s in aligned.values()]))
    all_dates = pd.DatetimeIndex(all_dates)

    # Reindex all to common dates, fill missing with 0
    for sname in aligned:
        aligned[sname] = aligned[sname].reindex(all_dates).fillna(0)

    # Monthly rebalance: on first trading day of each month, reset to target weights
    # Between rebalances, weights drift with returns
    portfolio_value = pd.Series(1.0, index=all_dates)
    strategy_names = list(weights.keys())
    n_strats = len(strategy_names)

    # Track allocation values
    alloc = {s: weights[s] for s in strategy_names}  # $ allocation per unit

    for i in range(1, len(all_dates)):
        date = all_dates[i]
        prev_date = all_dates[i-1]

        # Grow each allocation by its strategy return
        total = 0
        for s in strategy_names:
            ret = aligned[s].loc[date]
            alloc[s] = alloc[s] * (1 + ret)
            total += alloc[s]

        # Monthly rebalance on first day of month
        is_rebal = date.month != prev_date.month

        if is_rebal:
            # Rebalance cost: proportional to turnover
            turnover = sum(abs(alloc[s]/total - weights[s]) for s in strategy_names) / 2
            rebal_cost = turnover * (COST_BPS / 10000)
            total *= (1 - rebal_cost)

            # Reset to target weights
            for s in strategy_names:
                alloc[s] = weights[s] * total

        portfolio_value.iloc[i] = total

    return portfolio_value


# ============================================================
# Metrics
# ============================================================
def compute_metrics(equity_curve, name='Strategy'):
    """Compute standard risk-adjusted metrics from equity curve."""
    returns = equity_curve.pct_change().dropna()
    n_years = len(returns) / 252

    total_ret = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside if downside > 0 else 0

    # Max drawdown
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Profit factor
    pos_ret = returns[returns > 0].sum()
    neg_ret = abs(returns[returns < 0].sum())
    pf = pos_ret / neg_ret if neg_ret > 0 else float('inf')

    win_rate = (returns > 0).sum() / len(returns) * 100

    return {
        'name': name,
        'total_return': f"{total_ret*100:.1f}%",
        'cagr': f"{cagr*100:.1f}%",
        'ann_vol': f"{ann_vol*100:.1f}%",
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_dd': f"{max_dd*100:.1f}%",
        'calmar': round(calmar, 2),
        'profit_factor': round(pf, 2),
        'win_rate': f"{win_rate:.1f}%",
        'years': round(n_years, 1),
        # Raw values for comparison
        '_cagr': cagr,
        '_sharpe': sharpe,
        '_sortino': sortino,
        '_max_dd': max_dd,
        '_calmar': calmar,
        '_ann_vol': ann_vol,
    }


def worst_drawdowns(equity_curve, n=5):
    """Find the N worst drawdown periods."""
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max

    dds = []
    dd_series = drawdown.copy()

    for _ in range(n):
        trough_idx = dd_series.idxmin()
        trough_val = dd_series.min()

        if trough_val >= 0:
            break

        # Find peak before trough
        peak_idx = equity_curve.loc[:trough_idx].idxmax()

        # Find recovery after trough
        recovery = equity_curve.loc[trough_idx:]
        peak_val = equity_curve.loc[peak_idx]
        recovered = recovery[recovery >= peak_val]
        if len(recovered) > 0:
            recovery_idx = recovered.index[0]
            duration = (recovery_idx - peak_idx).days
        else:
            recovery_idx = 'Not recovered'
            duration = (equity_curve.index[-1] - peak_idx).days

        dds.append({
            'peak': str(peak_idx.date()),
            'trough': str(trough_idx.date()),
            'recovery': str(recovery_idx.date()) if isinstance(recovery_idx, pd.Timestamp) else recovery_idx,
            'depth': f"{trough_val*100:.1f}%",
            'duration_days': duration,
        })

        # Mask this drawdown region to find next one
        mask_start = peak_idx
        mask_end = recovery_idx if isinstance(recovery_idx, pd.Timestamp) else equity_curve.index[-1]
        dd_series.loc[mask_start:mask_end] = 0

    return dds


def annual_returns_table(equity_curves: dict):
    """Build annual returns table for all strategies."""
    years = sorted(set(y for eq in equity_curves.values() for y in eq.index.year.unique()))
    table = {}

    for name, eq in equity_curves.items():
        annual = {}
        for yr in years:
            yr_data = eq[eq.index.year == yr]
            if len(yr_data) > 5:
                annual[yr] = (yr_data.iloc[-1] / yr_data.iloc[0] - 1) * 100
        table[name] = annual

    df = pd.DataFrame(table)
    return df


def correlation_matrix(strategy_returns: dict):
    """Compute cross-strategy correlation matrix."""
    df = pd.DataFrame(strategy_returns)
    # Align to common dates
    df = df.dropna()
    return df.corr()


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("COMBINED PORTFOLIO BUILDER v2")
    print("=" * 70)

    # Load all strategies
    print("\n--- Loading Strategy Return Series ---\n")
    cta_ret = load_trend_cta()
    upro_ret = load_upro_overlay()
    vix_ret = construct_vix_spike()
    wheel_ret = load_wheel()
    spy_ret = load_spy_benchmark()

    # Strategy returns dict for correlation
    strat_rets = {
        'Trend CTA': cta_ret,
        'UPRO+Overlay': upro_ret,
        'VIX Spike': vix_ret,
        'Wheel': wheel_ret,
        'SPY B&H': spy_ret,
    }

    # Define portfolios
    portfolios = {
        'A. Growth': {
            'weights': {'Trend CTA': 0.50, 'UPRO+Overlay': 0.30, 'VIX Spike': 0.20},
            'returns': {'Trend CTA': cta_ret, 'UPRO+Overlay': upro_ret, 'VIX Spike': vix_ret},
        },
        'B. Balanced': {
            'weights': {'Trend CTA': 0.40, 'UPRO+Overlay': 0.25, 'VIX Spike': 0.15, 'Wheel': 0.20},
            'returns': {'Trend CTA': cta_ret, 'UPRO+Overlay': upro_ret, 'VIX Spike': vix_ret, 'Wheel': wheel_ret},
        },
        'C. Conservative': {
            'weights': {'Trend CTA': 0.60, 'Wheel': 0.40},
            'returns': {'Trend CTA': cta_ret, 'Wheel': wheel_ret},
        },
        'D. Two-Strategy': {
            'weights': {'Trend CTA': 0.60, 'UPRO+Overlay': 0.40},
            'returns': {'Trend CTA': cta_ret, 'UPRO+Overlay': upro_ret},
        },
    }

    # Build all portfolios
    print("\n--- Building Combined Portfolios ---")
    equity_curves = {}
    all_metrics = {}

    for pname, pconfig in portfolios.items():
        eq = build_combined_portfolio(pconfig['returns'], pconfig['weights'], pname)
        equity_curves[pname] = eq
        metrics = compute_metrics(eq, pname)
        all_metrics[pname] = metrics

    # SPY benchmark equity curve
    spy_common_start = max(eq.index.min() for eq in equity_curves.values())
    spy_common_end = min(eq.index.max() for eq in equity_curves.values())
    spy_aligned = spy_ret.loc[spy_common_start:spy_common_end]
    spy_eq = (1 + spy_aligned).cumprod()
    spy_eq.iloc[0] = 1.0
    equity_curves['SPY B&H'] = spy_eq
    all_metrics['SPY B&H'] = compute_metrics(spy_eq, 'SPY B&H')

    # Also compute individual strategy metrics on common period
    print("\n--- Individual Strategy Metrics (common period) ---")
    indiv_metrics = {}
    for sname, sret in strat_rets.items():
        if sname == 'SPY B&H':
            continue
        common = sret.loc[spy_common_start:spy_common_end].dropna()
        if len(common) > 100:
            eq = (1 + common).cumprod()
            eq.iloc[0] = 1.0
            m = compute_metrics(eq, sname)
            indiv_metrics[sname] = m

    # ============================================================
    # RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("PORTFOLIO METRICS COMPARISON")
    print("=" * 70)

    header = f"{'Portfolio':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>7} {'Vol':>7} {'WR':>6}"
    print(header)
    print("-" * 80)

    for pname in list(portfolios.keys()) + ['SPY B&H']:
        m = all_metrics[pname]
        print(f"{m['name']:<25} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>8} {m['max_dd']:>8} {m['calmar']:>7.2f} {m['ann_vol']:>7} {m['win_rate']:>6}")

    print("\n\nINDIVIDUAL STRATEGY METRICS (common period):")
    print("-" * 80)
    for sname, m in indiv_metrics.items():
        print(f"{m['name']:<25} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['cagr']:>8} {m['max_dd']:>8} {m['calmar']:>7.2f} {m['ann_vol']:>7} {m['win_rate']:>6}")

    # Worst drawdowns
    print("\n" + "=" * 70)
    print("WORST 5 DRAWDOWNS PER PORTFOLIO")
    print("=" * 70)

    worst_dds = {}
    for pname, eq in equity_curves.items():
        dds = worst_drawdowns(eq)
        worst_dds[pname] = dds
        print(f"\n{pname}:")
        for i, dd in enumerate(dds):
            print(f"  {i+1}. {dd['depth']:>8} | {dd['peak']} to {dd['trough']} | "
                  f"Recovery: {dd['recovery']} | Duration: {dd['duration_days']}d")

    # Annual returns
    print("\n" + "=" * 70)
    print("ANNUAL RETURNS (%)")
    print("=" * 70)
    ann_ret = annual_returns_table(equity_curves)
    pd.set_option('display.float_format', lambda x: f'{x:.1f}')
    print(ann_ret.to_string())

    # Correlation matrix
    print("\n" + "=" * 70)
    print("CROSS-STRATEGY CORRELATION MATRIX (daily returns)")
    print("=" * 70)
    corr = correlation_matrix(strat_rets)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')
    print(corr.to_string())

    # ============================================================
    # PLOTS
    # ============================================================

    # 1. Equity curves
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    for pname, eq in equity_curves.items():
        label = pname
        lw = 2 if pname != 'SPY B&H' else 1.5
        ls = '-' if pname != 'SPY B&H' else '--'
        ax.plot(eq.index, eq.values, label=label, linewidth=lw, linestyle=ls)
    ax.set_title('Combined Portfolio Equity Curves (log scale)', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Portfolio Value (starting = 1.0)')

    # 2. Drawdowns
    ax = axes[1]
    for pname, eq in equity_curves.items():
        cum_max = eq.cummax()
        dd = (eq - cum_max) / cum_max * 100
        lw = 2 if pname != 'SPY B&H' else 1.5
        ls = '-' if pname != 'SPY B&H' else '--'
        ax.plot(dd.index, dd.values, label=pname, linewidth=lw, linestyle=ls)
    ax.set_title('Drawdowns', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Drawdown (%)')
    ax.set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/equity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved equity_curves.png")

    # 3. Annual returns heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    ann_display = ann_ret.fillna(0)
    im = ax.imshow(ann_display.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=40)

    ax.set_xticks(range(len(ann_display.columns)))
    ax.set_xticklabels(ann_display.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(ann_display.index)))
    ax.set_yticklabels(ann_display.index, fontsize=9)

    for i in range(len(ann_display.index)):
        for j in range(len(ann_display.columns)):
            val = ann_display.values[i, j]
            if val != 0:
                color = 'black' if abs(val) < 20 else 'white'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)

    ax.set_title('Annual Returns (%) by Portfolio', fontsize=14)
    plt.colorbar(im, ax=ax, label='Return (%)')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/annual_returns_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved annual_returns_heatmap.png")

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='coolwarm', vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=10)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.values[i, j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title('Cross-Strategy Correlation Matrix', fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved correlation_matrix.png")

    # 5. Rolling Sharpe comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    window = 252
    for pname, eq in equity_curves.items():
        rets = eq.pct_change().dropna()
        if len(rets) > window:
            rolling_sharpe = rets.rolling(window).mean() / rets.rolling(window).std() * np.sqrt(252)
            lw = 2 if pname != 'SPY B&H' else 1.5
            ls = '-' if pname != 'SPY B&H' else '--'
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, label=pname, linewidth=lw, linestyle=ls)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Sharpe Ratio')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/rolling_sharpe.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved rolling_sharpe.png")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    # Save metrics
    results = {
        'generated': datetime.now().isoformat(),
        'parameters': {
            'transaction_cost_bps': COST_BPS,
            'rebalance': 'monthly (first trading day)',
            'signal_lag': 'T-1 (all strategies)',
            'weight_method': 'fixed (no optimization)',
        },
        'portfolio_metrics': {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                             for k, v in all_metrics.items()},
        'individual_strategy_metrics': {k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                                        for k, v in indiv_metrics.items()},
        'worst_drawdowns': worst_dds,
        'correlation_matrix': corr.to_dict(),
        'annual_returns': ann_ret.fillna('N/A').to_dict(),
        'portfolio_definitions': {
            'A. Growth': '50% Trend CTA + 30% UPRO+Overlay + 20% VIX Spike',
            'B. Balanced': '40% Trend CTA + 25% UPRO+Overlay + 15% VIX Spike + 20% Wheel',
            'C. Conservative': '60% Trend CTA + 40% Wheel',
            'D. Two-Strategy': '60% Trend CTA + 40% UPRO+Overlay',
        },
    }

    with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results.json")

    # Save equity curves as CSV
    eq_df = pd.DataFrame(equity_curves)
    eq_df.to_csv(f'{OUTPUT_DIR}/equity_curves.csv')
    print("Saved equity_curves.csv")

    # Summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    best = max(all_metrics.items(), key=lambda x: x[1]['_sharpe'] if x[0] != 'SPY B&H' else -999)
    spy_m = all_metrics['SPY B&H']

    print(f"\nBest risk-adjusted portfolio: {best[0]}")
    print(f"  Sharpe: {best[1]['sharpe']} vs SPY {spy_m['sharpe']}")
    print(f"  Sortino: {best[1]['sortino']} vs SPY {spy_m['sortino']}")
    print(f"  MaxDD: {best[1]['max_dd']} vs SPY {spy_m['max_dd']}")
    print(f"  CAGR: {best[1]['cagr']} vs SPY {spy_m['cagr']}")

    # Does adding CTA improve things?
    print(f"\nTrend CTA impact (comparing D vs individual UPRO+Overlay):")
    if 'UPRO+Overlay' in indiv_metrics:
        upro_m = indiv_metrics['UPRO+Overlay']
        d_m = all_metrics['D. Two-Strategy']
        print(f"  UPRO alone: Sharpe={upro_m['sharpe']}, MaxDD={upro_m['max_dd']}")
        print(f"  D (60/40 CTA+UPRO): Sharpe={d_m['sharpe']}, MaxDD={d_m['max_dd']}")
        print(f"  Sharpe improvement: {d_m['_sharpe'] - upro_m['_sharpe']:+.2f}")
        print(f"  MaxDD improvement: {(d_m['_max_dd'] - upro_m['_max_dd'])*100:+.1f}pp")


if __name__ == '__main__':
    main()
