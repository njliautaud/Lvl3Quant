#!/usr/bin/env python3
"""
Multi-Asset Trend Following / Dual Momentum CTA Strategy
=========================================================
Simple rules-based strategy: absolute + relative momentum with monthly rebalancing.

Variants:
  V1: 12-month lookback, equal weight top 3
  V2: 6-month lookback, equal weight top 3
  V3: 12-month lookback, inverse-vol weighted top 3
  V4: 12-month lookback, equal weight top 3, VIX filter (>25 → 100% SHY)

Validation (on best variant):
  - Permutation test (200 perms)
  - Regime test (green vs red months)
  - Sub-period stability (4 blocks, CV of Sharpe)
  - Lag sensitivity (T-0, T-1, T-2)
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
import matplotlib.dates as mdates
from scipy import stats
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

OUT_DIR = Path('/home/jupiter/Lvl3Quant/output/trend_cta_v1')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ───────────────────────────────────────────────────────────
UNIVERSE = ['SPY', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'DBC', 'VNQ']
CASH = 'SHY'
TOP_N = 3
COST_BPS = 10  # per leg
COMMISSION_PER_TRADE = 5.0
INITIAL_CAPITAL = 100_000
START_DATE = '2005-01-01'
END_DATE = '2026-07-18'

# ─── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download adjusted close prices for all tickers."""
    tickers = UNIVERSE + [CASH, '^VIX']
    print(f"Downloading data for {tickers} from {START_DATE} to {END_DATE}...")

    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[t] = df['Close']
            print(f"  {t}: {len(df)} rows, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"  ERROR downloading {t}: {e}")

    prices = pd.DataFrame(data)
    # Rename VIX column
    if '^VIX' in prices.columns:
        prices.rename(columns={'^VIX': 'VIX'}, inplace=True)

    # Forward-fill then drop rows with any NaN in universe
    prices = prices.ffill()
    first_valid = prices[UNIVERSE + [CASH]].dropna().index[0]
    prices = prices.loc[first_valid:]

    prices.to_parquet(OUT_DIR / 'prices.parquet')
    print(f"Prices saved: {prices.shape[0]} days, {prices.shape[1]} columns")
    print(f"Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    return prices


# ─── Monthly Rebalance Dates ─────────────────────────────────────────────────
def get_rebalance_dates(prices):
    """Get first trading day of each month."""
    monthly = prices.resample('MS').first()  # month start
    # Get actual trading days closest to month starts
    rebal_dates = []
    for ms in monthly.index:
        # Find first trading day on or after month start
        mask = prices.index >= ms
        if mask.any():
            rebal_dates.append(prices.index[mask][0])
    return sorted(set(rebal_dates))


# ─── Strategy Engine ─────────────────────────────────────────────────────────
def run_strategy(prices, lookback_months=12, vol_weight=False, vix_filter=False,
                 signal_lag=1, top_n=TOP_N, label='V1'):
    """
    Run the dual momentum strategy.

    signal_lag: 0=T-0 (use today's close, look-ahead bias), 1=T-1 (honest), 2=T-2
    """
    rebal_dates = get_rebalance_dates(prices)
    lookback_days = lookback_months * 21  # approx trading days

    # Need enough history for lookback
    valid_rebal = [d for d in rebal_dates if prices.index.get_loc(d) >= lookback_days + signal_lag]

    # Track portfolio
    portfolio_value = INITIAL_CAPITAL
    holdings = {CASH: 1.0}  # start in cash

    records = []
    daily_values = []
    turnover_history = []

    # Daily returns for all assets
    asset_returns = prices[UNIVERSE + [CASH]].pct_change()

    prev_rebal_idx = None
    current_weights = {CASH: 1.0}

    for i, date in enumerate(valid_rebal):
        date_idx = prices.index.get_loc(date)
        signal_idx = date_idx - signal_lag  # T-1 signal

        if signal_idx < lookback_days:
            continue

        signal_date = prices.index[signal_idx]

        # ── VIX Filter ──
        if vix_filter and 'VIX' in prices.columns:
            vix_val = prices['VIX'].iloc[signal_idx]
            if vix_val > 25:
                new_weights = {CASH: 1.0}
                # Calculate turnover
                turnover = sum(abs(new_weights.get(a, 0) - current_weights.get(a, 0))
                              for a in set(list(new_weights.keys()) + list(current_weights.keys())))
                turnover_history.append(turnover)
                current_weights = new_weights
                records.append({
                    'date': date, 'signal_date': signal_date,
                    'selected': ['SHY (VIX filter)'], 'vix': vix_val,
                    'turnover': turnover
                })
                continue

        # ── Absolute Momentum ──
        # 12-month (or 6-month) return for each asset
        mom_returns = {}
        for asset in UNIVERSE:
            p_now = prices[asset].iloc[signal_idx]
            p_past = prices[asset].iloc[signal_idx - lookback_days]
            mom_returns[asset] = (p_now / p_past) - 1

        # Filter: only assets with positive momentum
        passed = {a: r for a, r in mom_returns.items() if r > 0}

        # ── Relative Momentum Ranking ──
        ranked = sorted(passed.items(), key=lambda x: x[1], reverse=True)
        selected = [a for a, _ in ranked[:top_n]]

        # ── Position Sizing ──
        if vol_weight and len(selected) > 0:
            # Inverse volatility weighting
            vols = {}
            for asset in selected:
                ret_slice = asset_returns[asset].iloc[signal_idx-63:signal_idx]
                vol = ret_slice.std() * np.sqrt(252)
                vols[asset] = max(vol, 0.01)  # floor

            inv_vols = {a: 1.0/v for a, v in vols.items()}
            total_inv_vol = sum(inv_vols.values())
            new_weights = {a: iv/total_inv_vol for a, iv in inv_vols.items()}
        elif len(selected) > 0:
            w = 1.0 / top_n
            new_weights = {a: w for a in selected}
        else:
            new_weights = {}

        # Fill remaining with SHY
        allocated = sum(new_weights.values())
        if allocated < 1.0:
            new_weights[CASH] = 1.0 - allocated

        # ── Turnover & Costs ──
        all_assets = set(list(new_weights.keys()) + list(current_weights.keys()))
        turnover = sum(abs(new_weights.get(a, 0) - current_weights.get(a, 0)) for a in all_assets)
        n_trades = sum(1 for a in all_assets
                      if abs(new_weights.get(a, 0) - current_weights.get(a, 0)) > 0.01)

        turnover_history.append(turnover)

        records.append({
            'date': date, 'signal_date': signal_date,
            'selected': selected if selected else ['SHY (all negative)'],
            'mom_returns': {a: round(r, 4) for a, r in sorted(mom_returns.items(), key=lambda x: x[1], reverse=True)[:5]},
            'turnover': round(turnover, 4),
            'n_trades': n_trades
        })

        current_weights = new_weights.copy()

    # ── Simulate Daily Returns ──
    # Build weight timeseries
    weight_changes = {}
    for rec in records:
        date = rec['date']
        selected = rec['selected']
        # Reconstruct weights from records
        if any('VIX filter' in s or 'all negative' in s for s in selected):
            weight_changes[date] = {CASH: 1.0}
        else:
            if vol_weight:
                # Re-derive (stored in current_weights logic above)
                # For simplicity, just equal weight in the simulation path
                pass
            weight_changes[date] = {}

    # Better approach: simulate using the strategy logic directly
    # Re-run but track daily portfolio value

    portfolio_values = _simulate_daily(prices, records, valid_rebal, lookback_months,
                                        vol_weight, vix_filter, signal_lag, top_n, asset_returns)

    return portfolio_values, records, turnover_history


def _simulate_daily(prices, records, rebal_dates, lookback_months, vol_weight,
                    vix_filter, signal_lag, top_n, asset_returns):
    """Simulate daily portfolio values with proper weight tracking."""
    lookback_days = lookback_months * 21

    # Build rebalance schedule: date -> weights
    weight_schedule = {}

    for rec in records:
        date = rec['date']
        signal_idx = prices.index.get_loc(date) - signal_lag
        selected = rec['selected']

        if any('VIX filter' in str(s) or 'all negative' in str(s) for s in selected):
            weight_schedule[date] = {CASH: 1.0}
            continue

        clean_selected = [s for s in selected if s in UNIVERSE]

        if vol_weight and len(clean_selected) > 0:
            vols = {}
            for asset in clean_selected:
                ret_slice = asset_returns[asset].iloc[max(0, signal_idx-63):signal_idx]
                vol = ret_slice.std() * np.sqrt(252)
                vols[asset] = max(vol, 0.01)
            inv_vols = {a: 1.0/v for a, v in vols.items()}
            total = sum(inv_vols.values())
            weights = {a: iv/total for a, iv in inv_vols.items()}
        elif len(clean_selected) > 0:
            w = 1.0 / top_n
            weights = {a: w for a in clean_selected}
        else:
            weights = {}

        allocated = sum(weights.values())
        if allocated < 1.0:
            weights[CASH] = 1.0 - allocated

        weight_schedule[date] = weights

    # Simulate daily
    all_dates = prices.index
    first_rebal = min(weight_schedule.keys()) if weight_schedule else all_dates[-1]

    portfolio = pd.Series(index=all_dates, dtype=float)
    portfolio[:] = np.nan

    # Start at initial capital on first rebal date
    start_idx = all_dates.get_loc(first_rebal)
    portfolio.iloc[:start_idx+1] = INITIAL_CAPITAL

    current_weights = weight_schedule.get(first_rebal, {CASH: 1.0})
    prev_weights = {CASH: 1.0}

    rebal_dates_set = set(weight_schedule.keys())

    for idx in range(start_idx + 1, len(all_dates)):
        date = all_dates[idx]
        prev_val = portfolio.iloc[idx - 1]

        # Check for rebalance
        if date in rebal_dates_set:
            new_weights = weight_schedule[date]
            # Apply transaction costs
            turnover = sum(abs(new_weights.get(a, 0) - current_weights.get(a, 0))
                          for a in set(list(new_weights.keys()) + list(current_weights.keys())))
            n_trades = sum(1 for a in set(list(new_weights.keys()) + list(current_weights.keys()))
                          if abs(new_weights.get(a, 0) - current_weights.get(a, 0)) > 0.01)

            cost = prev_val * turnover * COST_BPS / 10000 + n_trades * COMMISSION_PER_TRADE
            prev_val -= cost
            current_weights = new_weights

        # Daily return
        day_ret = 0.0
        for asset, w in current_weights.items():
            r = asset_returns[asset].iloc[idx] if not pd.isna(asset_returns[asset].iloc[idx]) else 0.0
            day_ret += w * r

        portfolio.iloc[idx] = prev_val * (1 + day_ret)

    return portfolio.dropna()


# ─── Metrics ─────────────────────────────────────────────────────────────────
def calc_metrics(portfolio_values, label='Strategy'):
    """Calculate risk-adjusted performance metrics."""
    returns = portfolio_values.pct_change().dropna()
    monthly = portfolio_values.resample('ME').last().pct_change().dropna()

    years = len(returns) / 252
    total_ret = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / years) - 1

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (monthly)
    win_rate = (monthly > 0).mean()

    # SPY correlation
    spy_ret = None

    metrics = {
        'label': label,
        'CAGR': f"{cagr:.2%}",
        'Ann_Vol': f"{ann_vol:.2%}",
        'Sharpe': round(sharpe, 3),
        'Sortino': round(sortino, 3),
        'MaxDD': f"{max_dd:.2%}",
        'Calmar': round(calmar, 3),
        'WR_monthly': f"{win_rate:.1%}",
        'Total_Return': f"{total_ret:.1%}",
        'Years': round(years, 1),
        'Final_Value': f"${portfolio_values.iloc[-1]:,.0f}",
    }

    return metrics, returns, monthly


def calc_spy_corr(strat_returns, prices):
    """Calculate correlation with SPY."""
    spy_ret = prices['SPY'].pct_change().dropna()
    common = strat_returns.index.intersection(spy_ret.index)
    if len(common) > 50:
        corr = strat_returns.loc[common].corr(spy_ret.loc[common])
        return round(corr, 3)
    return None


# ─── Validation ──────────────────────────────────────────────────────────────
def permutation_test(prices, best_portfolio, n_perms=200, lookback_months=12,
                     vol_weight=False, vix_filter=False):
    """Shuffle asset rankings across dates, compare Sharpe."""
    real_returns = best_portfolio.pct_change().dropna()
    real_sharpe = real_returns.mean() / real_returns.std() * np.sqrt(252)

    print(f"\nRunning {n_perms} permutations...")
    perm_sharpes = []

    for i in range(n_perms):
        if (i+1) % 50 == 0:
            print(f"  Permutation {i+1}/{n_perms}")

        # Shuffle: for each rebalance date, randomly pick 3 assets instead of momentum-ranked
        perm_portfolio = _run_permuted(prices, lookback_months, vol_weight, vix_filter)
        if perm_portfolio is not None and len(perm_portfolio) > 50:
            perm_ret = perm_portfolio.pct_change().dropna()
            perm_sharpe = perm_ret.mean() / perm_ret.std() * np.sqrt(252) if perm_ret.std() > 0 else 0
            perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    return {
        'real_sharpe': round(real_sharpe, 3),
        'perm_mean_sharpe': round(perm_sharpes.mean(), 3),
        'perm_std_sharpe': round(perm_sharpes.std(), 3),
        'p_value': round(p_value, 4),
        'perm_95th': round(np.percentile(perm_sharpes, 95), 3),
        'significant': p_value < 0.05
    }, perm_sharpes


def _run_permuted(prices, lookback_months, vol_weight, vix_filter):
    """Run strategy with randomly shuffled asset selection."""
    rebal_dates = get_rebalance_dates(prices)
    lookback_days = lookback_months * 21
    asset_returns = prices[UNIVERSE + [CASH]].pct_change()

    valid_rebal = [d for d in rebal_dates if prices.index.get_loc(d) >= lookback_days + 1]
    if not valid_rebal:
        return None

    all_dates = prices.index
    start_idx = all_dates.get_loc(valid_rebal[0])

    portfolio = pd.Series(index=all_dates[start_idx:], dtype=float)
    portfolio.iloc[0] = INITIAL_CAPITAL

    # Build random weight schedule
    weight_schedule = {}
    rng = np.random.default_rng()

    for date in valid_rebal:
        # Random selection of TOP_N assets (or fewer)
        n_pass = rng.integers(0, len(UNIVERSE) + 1)  # random number passing abs momentum
        if n_pass > 0:
            selected = list(rng.choice(UNIVERSE, size=min(n_pass, TOP_N), replace=False))
            w = 1.0 / TOP_N
            weights = {a: w for a in selected}
            allocated = sum(weights.values())
            if allocated < 1.0:
                weights[CASH] = 1.0 - allocated
        else:
            weights = {CASH: 1.0}
        weight_schedule[date] = weights

    current_weights = {CASH: 1.0}
    rebal_set = set(weight_schedule.keys())

    for idx in range(1, len(portfolio)):
        date = portfolio.index[idx]
        prev_val = portfolio.iloc[idx - 1]

        if date in rebal_set:
            new_weights = weight_schedule[date]
            turnover = sum(abs(new_weights.get(a, 0) - current_weights.get(a, 0))
                          for a in set(list(new_weights.keys()) + list(current_weights.keys())))
            n_trades = sum(1 for a in set(list(new_weights.keys()) + list(current_weights.keys()))
                          if abs(new_weights.get(a, 0) - current_weights.get(a, 0)) > 0.01)
            cost = prev_val * turnover * COST_BPS / 10000 + n_trades * COMMISSION_PER_TRADE
            prev_val -= cost
            current_weights = new_weights

        day_ret = 0.0
        for asset, w in current_weights.items():
            abs_idx = all_dates.get_loc(date)
            r = asset_returns[asset].iloc[abs_idx] if not pd.isna(asset_returns[asset].iloc[abs_idx]) else 0.0
            day_ret += w * r

        portfolio.iloc[idx] = prev_val * (1 + day_ret)

    return portfolio


def regime_test(portfolio_values, prices):
    """Test performance in green vs red months (SPY monthly return)."""
    strat_monthly = portfolio_values.resample('ME').last().pct_change().dropna()
    spy_monthly = prices['SPY'].resample('ME').last().pct_change().dropna()

    common = strat_monthly.index.intersection(spy_monthly.index)
    strat_monthly = strat_monthly.loc[common]
    spy_monthly = spy_monthly.loc[common]

    green = spy_monthly > 0
    red = spy_monthly <= 0

    green_ret = strat_monthly[green]
    red_ret = strat_monthly[red]

    green_sharpe = green_ret.mean() / green_ret.std() * np.sqrt(12) if green_ret.std() > 0 else 0
    red_sharpe = red_ret.mean() / red_ret.std() * np.sqrt(12) if red_ret.std() > 0 else 0

    regime_skew = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

    return {
        'green_months': len(green_ret),
        'red_months': len(red_ret),
        'green_avg_ret': f"{green_ret.mean():.3%}",
        'red_avg_ret': f"{red_ret.mean():.3%}",
        'green_sharpe': round(green_sharpe, 3),
        'red_sharpe': round(red_sharpe, 3),
        'regime_skew': round(regime_skew, 3),
        'passes_regime_test': regime_skew <= 0.50,
    }


def subperiod_test(portfolio_values, n_blocks=4):
    """Split into n_blocks equal periods, compute CV of Sharpe."""
    returns = portfolio_values.pct_change().dropna()
    block_size = len(returns) // n_blocks

    sharpes = []
    results = []
    for i in range(n_blocks):
        start = i * block_size
        end = (i + 1) * block_size if i < n_blocks - 1 else len(returns)
        block = returns.iloc[start:end]
        s = block.mean() / block.std() * np.sqrt(252) if block.std() > 0 else 0
        sharpes.append(s)
        results.append({
            'block': i + 1,
            'start': block.index[0].strftime('%Y-%m-%d'),
            'end': block.index[-1].strftime('%Y-%m-%d'),
            'sharpe': round(s, 3),
            'cagr': f"{((1 + block.sum()) ** (252/len(block)) - 1):.2%}",
        })

    cv = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else float('inf')

    return {
        'blocks': results,
        'sharpes': [round(s, 3) for s in sharpes],
        'cv_of_sharpe': round(cv, 3),
        'mean_sharpe': round(np.mean(sharpes), 3),
        'stable': cv < 1.0,
    }


def lag_sensitivity(prices, lookback_months=12, vol_weight=False, vix_filter=False):
    """Test T-0, T-1, T-2 signal lag."""
    results = {}
    for lag in [0, 1, 2]:
        pv, _, _ = run_strategy(prices, lookback_months=lookback_months,
                                vol_weight=vol_weight, vix_filter=vix_filter,
                                signal_lag=lag, label=f'Lag-T{lag}')
        metrics, _, _ = calc_metrics(pv, f'T-{lag}')
        results[f'T-{lag}'] = {
            'sharpe': metrics['Sharpe'],
            'cagr': metrics['CAGR'],
            'max_dd': metrics['MaxDD'],
        }
    return results


# ─── Plotting ────────────────────────────────────────────────────────────────
def plot_equity_curves(results_dict, prices, out_path):
    """Plot equity curves for all variants + SPY benchmark."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Equity curves
    ax = axes[0]
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9E9E9E']

    # SPY benchmark (buy and hold)
    spy_start = None
    for label, pv in results_dict.items():
        if spy_start is None:
            spy_start = pv.index[0]

    spy_bh = prices['SPY'].loc[spy_start:] / prices['SPY'].loc[spy_start] * INITIAL_CAPITAL
    ax.plot(spy_bh.index, spy_bh.values, color=colors[4], linewidth=1.5, label='SPY B&H', alpha=0.7, linestyle='--')

    for i, (label, pv) in enumerate(results_dict.items()):
        ax.plot(pv.index, pv.values, color=colors[i % len(colors)], linewidth=1.5, label=label)

    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Multi-Asset Trend Following: Equity Curves (2005-2026)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # Drawdown for best variant
    ax2 = axes[1]
    best_label = list(results_dict.keys())[0]
    best_pv = results_dict[best_label]
    cummax = best_pv.cummax()
    dd = (best_pv - cummax) / cummax
    ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
    ax2.plot(dd.index, dd.values, color='red', linewidth=0.5)
    ax2.set_ylabel('Drawdown', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title(f'Drawdown ({best_label})', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Equity curve plot saved: {out_path}")


def plot_permutation(perm_sharpes, real_sharpe, out_path):
    """Plot permutation test distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(perm_sharpes, bins=30, color='#90CAF9', edgecolor='#1565C0', alpha=0.8)
    ax.axvline(real_sharpe, color='red', linewidth=2, linestyle='--', label=f'Real Sharpe: {real_sharpe:.3f}')
    ax.axvline(np.percentile(perm_sharpes, 95), color='orange', linewidth=1.5, linestyle=':',
               label=f'95th perm: {np.percentile(perm_sharpes, 95):.3f}')
    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Permutation Test: Real vs Random Asset Selection (200 perms)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Permutation plot saved: {out_path}")


def plot_annual_returns(portfolio_values, prices, out_path):
    """Bar chart of annual returns vs SPY."""
    strat_annual = portfolio_values.resample('YE').last().pct_change().dropna()

    spy_start = portfolio_values.index[0]
    spy_bh = prices['SPY'].loc[spy_start:]
    spy_annual = spy_bh.resample('YE').last().pct_change().dropna()

    common_years = strat_annual.index.intersection(spy_annual.index)
    strat_annual = strat_annual.loc[common_years]
    spy_annual = spy_annual.loc[common_years]

    years = [d.year for d in common_years]
    x = np.arange(len(years))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, strat_annual.values, width, label='CTA Strategy', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + width/2, spy_annual.values, width, label='SPY B&H', color='#9E9E9E', alpha=0.7)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Return', fontsize=12)
    ax.set_title('Annual Returns: CTA Strategy vs SPY', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.axhline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Annual returns plot saved: {out_path}")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MULTI-ASSET TREND FOLLOWING / DUAL MOMENTUM CTA")
    print("=" * 70)

    # 1. Download data
    prices = download_data()

    # 2. Run all variants
    print("\n" + "=" * 70)
    print("RUNNING STRATEGY VARIANTS")
    print("=" * 70)

    variants = {
        'V1: 12m Mom EqWt': {'lookback_months': 12, 'vol_weight': False, 'vix_filter': False},
        'V2: 6m Mom EqWt': {'lookback_months': 6, 'vol_weight': False, 'vix_filter': False},
        'V3: 12m Mom VolWt': {'lookback_months': 12, 'vol_weight': True, 'vix_filter': False},
        'V4: 12m Mom VIX': {'lookback_months': 12, 'vol_weight': False, 'vix_filter': True},
    }

    all_results = {}
    all_metrics = []

    for label, params in variants.items():
        print(f"\n--- {label} ---")
        pv, records, turnover = run_strategy(prices, signal_lag=1, label=label, **params)
        metrics, returns, monthly = calc_metrics(pv, label)
        spy_corr = calc_spy_corr(returns, prices)
        metrics['SPY_Corr'] = spy_corr
        metrics['Avg_Turnover'] = f"{np.mean(turnover):.2%}" if turnover else 'N/A'

        all_results[label] = pv
        all_metrics.append(metrics)

        print(f"  Sharpe: {metrics['Sharpe']}, Sortino: {metrics['Sortino']}, "
              f"CAGR: {metrics['CAGR']}, MaxDD: {metrics['MaxDD']}, "
              f"Calmar: {metrics['Calmar']}, WR: {metrics['WR_monthly']}, "
              f"SPY Corr: {spy_corr}")

    # Also compute SPY B&H metrics for comparison
    spy_start = list(all_results.values())[0].index[0]
    spy_bh = prices['SPY'].loc[spy_start:] / prices['SPY'].loc[spy_start] * INITIAL_CAPITAL
    spy_metrics, spy_ret, _ = calc_metrics(spy_bh, 'SPY B&H')
    spy_metrics['SPY_Corr'] = 1.0
    spy_metrics['Avg_Turnover'] = '0.00%'
    all_metrics.append(spy_metrics)

    # 3. Results table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    metrics_df = pd.DataFrame(all_metrics).set_index('label')
    print(metrics_df.to_string())
    metrics_df.to_csv(OUT_DIR / 'metrics_comparison.csv')

    # 4. Identify best variant (by Sharpe)
    best_idx = np.argmax([m['Sharpe'] for m in all_metrics[:-1]])  # exclude SPY
    best_label = all_metrics[best_idx]['label']
    best_pv = all_results[best_label]
    best_params = variants[best_label]
    print(f"\nBest variant: {best_label} (Sharpe: {all_metrics[best_idx]['Sharpe']})")

    # 5. Validation on best variant
    print("\n" + "=" * 70)
    print(f"VALIDATION: {best_label}")
    print("=" * 70)

    # 5a. Permutation test
    perm_results, perm_sharpes = permutation_test(
        prices, best_pv, n_perms=200,
        lookback_months=best_params['lookback_months'],
        vol_weight=best_params['vol_weight'],
        vix_filter=best_params['vix_filter']
    )
    print(f"\nPermutation Test:")
    for k, v in perm_results.items():
        print(f"  {k}: {v}")

    # 5b. Regime test
    regime_results = regime_test(best_pv, prices)
    print(f"\nRegime Test (Green vs Red months):")
    for k, v in regime_results.items():
        print(f"  {k}: {v}")

    # 5c. Sub-period stability
    subperiod_results = subperiod_test(best_pv)
    print(f"\nSub-Period Test (4 blocks):")
    for block in subperiod_results['blocks']:
        print(f"  Block {block['block']}: {block['start']} to {block['end']} | Sharpe: {block['sharpe']}")
    print(f"  CV of Sharpe: {subperiod_results['cv_of_sharpe']}")
    print(f"  Stable: {subperiod_results['stable']}")

    # 5d. Lag sensitivity
    lag_results = lag_sensitivity(prices, **best_params)
    print(f"\nLag Sensitivity:")
    for lag_label, lr in lag_results.items():
        print(f"  {lag_label}: Sharpe={lr['sharpe']}, CAGR={lr['cagr']}, MaxDD={lr['max_dd']}")

    # 6. Plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_equity_curves(all_results, prices, OUT_DIR / 'equity_curves.png')
    plot_permutation(perm_sharpes, perm_results['real_sharpe'], OUT_DIR / 'permutation_test.png')
    plot_annual_returns(best_pv, prices, OUT_DIR / 'annual_returns.png')

    # 7. Save full results
    full_results = {
        'strategy': 'Multi-Asset Trend Following / Dual Momentum CTA',
        'universe': UNIVERSE,
        'cash': CASH,
        'top_n': TOP_N,
        'cost_bps': COST_BPS,
        'commission': COMMISSION_PER_TRADE,
        'backtest_period': f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}",
        'metrics': {m['label']: m for m in all_metrics},
        'best_variant': best_label,
        'validation': {
            'permutation_test': perm_results,
            'regime_test': regime_results,
            'subperiod_test': {k: v for k, v in subperiod_results.items() if k != 'blocks'},
            'subperiod_blocks': subperiod_results['blocks'],
            'lag_sensitivity': lag_results,
        }
    }

    with open(OUT_DIR / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2, default=str)

    # 8. Save portfolio values
    for label, pv in all_results.items():
        safe_label = label.replace(':', '').replace(' ', '_').lower()
        pv.to_csv(OUT_DIR / f'portfolio_{safe_label}.csv')

    print(f"\nAll results saved to {OUT_DIR}")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return full_results


if __name__ == '__main__':
    results = main()
