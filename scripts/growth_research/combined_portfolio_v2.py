#!/usr/bin/env python3
"""
Combined Portfolio Backtest v2 — REAL DATA ONLY
================================================
Loads 9+ validated strategy returns from disk, constructs portfolios
with 3 weighting methods, and applies dynamic leverage overlay using
real macro signals.

HC #717: No synthetic returns. Every number traces to a real backtest file.
"""

import json
import warnings
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/combined_portfolio_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PART 1: Strategy Data Loading
# =============================================================================

def load_csv_returns(path, daily=True, name=""):
    """Load CSV with date index and single return column."""
    df = pd.read_csv(path)
    # Find date column
    date_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'unnamed' in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    # Find return column
    ret_col = None
    for c in df.columns:
        if c == date_col:
            continue
        if 'return' in c.lower():
            ret_col = c
            break
    if ret_col is None:
        # Use first non-date column
        ret_col = [c for c in df.columns if c != date_col][0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    series = df[ret_col].astype(float)

    if daily:
        # Resample daily returns to monthly by compounding
        monthly = (1 + series).resample('ME').prod() - 1
    else:
        monthly = series

    monthly.index = monthly.index.to_period('M').to_timestamp()
    monthly.name = name
    return monthly.dropna()


def load_parquet_returns(path, name=""):
    """Load parquet with ml_return column (already monthly)."""
    df = pd.read_parquet(path)

    # Set date index if needed
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

    # Find return column — prefer ml_return
    ret_col = None
    for c in df.columns:
        if c == 'ml_return':
            ret_col = c
            break
    if ret_col is None:
        for c in df.columns:
            if 'return' in c.lower():
                ret_col = c
                break
    if ret_col is None:
        ret_col = df.columns[0]

    series = df[ret_col].astype(float)
    # Normalize to month-end
    series.index = series.index.to_period('M').to_timestamp()
    series.name = name
    # If there are duplicate months (shouldn't be), take last
    series = series[~series.index.duplicated(keep='last')]
    return series.dropna()


def load_equity_curve(path, name=""):
    """Load equity curve CSV → pct_change → monthly."""
    df = pd.read_csv(path)
    date_col = df.columns[0]
    val_col = [c for c in df.columns if c != date_col][0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df = df[[val_col]].astype(float)
    # Remove duplicates, keep last per date
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()

    # Get month-end values, then pct_change
    monthly_vals = df[val_col].resample('ME').last()
    monthly_rets = monthly_vals.pct_change().dropna()
    monthly_rets.index = monthly_rets.index.to_period('M').to_timestamp()
    monthly_rets.name = name
    return monthly_rets.dropna()


def load_all_strategies():
    """Load all strategy return series from disk."""
    strategies = {}

    # 1. CTA Trend Following (daily CSV)
    path = "/home/jupiter/Lvl3Quant/output/growth_research/trend_following_returns.csv"
    if os.path.exists(path):
        strategies['CTA_Trend'] = load_csv_returns(path, daily=True, name='CTA_Trend')
        print(f"  CTA Trend: {len(strategies['CTA_Trend'])} months [{strategies['CTA_Trend'].index[0]:%Y-%m} to {strategies['CTA_Trend'].index[-1]:%Y-%m}]")

    # 2. Commodity Trend (monthly parquet)
    path = "/home/jupiter/Lvl3Quant/output/ml_commodity_trend/portfolio_returns.parquet"
    if os.path.exists(path):
        strategies['Commodity_Trend'] = load_parquet_returns(path, name='Commodity_Trend')
        print(f"  Commodity Trend: {len(strategies['Commodity_Trend'])} months [{strategies['Commodity_Trend'].index[0]:%Y-%m} to {strategies['Commodity_Trend'].index[-1]:%Y-%m}]")

    # 3. Sector Rotation (daily CSV)
    path = "/home/jupiter/Lvl3Quant/output/growth_research/sector_momentum_returns.csv"
    if os.path.exists(path):
        strategies['Sector_Rotation'] = load_csv_returns(path, daily=True, name='Sector_Rotation')
        print(f"  Sector Rotation: {len(strategies['Sector_Rotation'])} months [{strategies['Sector_Rotation'].index[0]:%Y-%m} to {strategies['Sector_Rotation'].index[-1]:%Y-%m}]")

    # 4. Currency Carry (monthly parquet)
    path = "/home/jupiter/Lvl3Quant/output/ml_currency_carry/portfolio_returns.parquet"
    if os.path.exists(path):
        strategies['Currency_Carry'] = load_parquet_returns(path, name='Currency_Carry')
        print(f"  Currency Carry: {len(strategies['Currency_Carry'])} months [{strategies['Currency_Carry'].index[0]:%Y-%m} to {strategies['Currency_Carry'].index[-1]:%Y-%m}]")

    # 5. Tail Risk Hedging (monthly parquet)
    path = "/home/jupiter/Lvl3Quant/output/ml_tail_risk_hedging/portfolio_returns.parquet"
    if os.path.exists(path):
        strategies['Tail_Risk'] = load_parquet_returns(path, name='Tail_Risk')
        print(f"  Tail Risk: {len(strategies['Tail_Risk'])} months [{strategies['Tail_Risk'].index[0]:%Y-%m} to {strategies['Tail_Risk'].index[-1]:%Y-%m}]")

    # 6. Bond Duration Timing (monthly parquet)
    path = "/home/jupiter/Lvl3Quant/output/ml_bond_duration_timing/portfolio_returns.parquet"
    if os.path.exists(path):
        strategies['Bond_Duration'] = load_parquet_returns(path, name='Bond_Duration')
        print(f"  Bond Duration: {len(strategies['Bond_Duration'])} months [{strategies['Bond_Duration'].index[0]:%Y-%m} to {strategies['Bond_Duration'].index[-1]:%Y-%m}]")

    # 7. Carry + Momentum (monthly parquet)
    path = "/home/jupiter/Lvl3Quant/output/ml_carry_momentum/portfolio_returns.parquet"
    if os.path.exists(path):
        strategies['Carry_Momentum'] = load_parquet_returns(path, name='Carry_Momentum')
        print(f"  Carry+Momentum: {len(strategies['Carry_Momentum'])} months [{strategies['Carry_Momentum'].index[0]:%Y-%m} to {strategies['Carry_Momentum'].index[-1]:%Y-%m}]")

    # 8. Vol Breakout (daily equity curve)
    path = "/home/jupiter/Lvl3Quant/output/ml_vol_breakout/equity_curve.csv"
    if os.path.exists(path):
        strategies['Vol_Breakout'] = load_equity_curve(path, name='Vol_Breakout')
        print(f"  Vol Breakout: {len(strategies['Vol_Breakout'])} months [{strategies['Vol_Breakout'].index[0]:%Y-%m} to {strategies['Vol_Breakout'].index[-1]:%Y-%m}]")

    # 9. Thematic Rotation (monthly parquet)
    path = "/home/jupiter/Lvl3Quant/output/ml_thematic_rotation/portfolio_returns.parquet"
    if os.path.exists(path):
        strategies['Thematic_Rotation'] = load_parquet_returns(path, name='Thematic_Rotation')
        print(f"  Thematic Rotation: {len(strategies['Thematic_Rotation'])} months [{strategies['Thematic_Rotation'].index[0]:%Y-%m} to {strategies['Thematic_Rotation'].index[-1]:%Y-%m}]")

    # 10. Hybrid Momentum (daily CSV)
    path = "/home/jupiter/Lvl3Quant/output/growth_research/hybrid_momentum/returns_A_full_hybrid.csv"
    if os.path.exists(path):
        strategies['Hybrid_Momentum'] = load_csv_returns(path, daily=True, name='Hybrid_Momentum')
        print(f"  Hybrid Momentum: {len(strategies['Hybrid_Momentum'])} months [{strategies['Hybrid_Momentum'].index[0]:%Y-%m} to {strategies['Hybrid_Momentum'].index[-1]:%Y-%m}]")

    # 11. Vol Harvesting — try best config
    vol_dir = "/home/jupiter/Lvl3Quant/output/growth_research/vol_harvesting/"
    if os.path.exists(vol_dir):
        csv_files = sorted([f for f in os.listdir(vol_dir) if f.endswith('.csv')])
        if csv_files:
            # Use first file (any config)
            path = os.path.join(vol_dir, csv_files[0])
            strategies['Vol_Harvesting'] = load_csv_returns(path, daily=True, name='Vol_Harvesting')
            print(f"  Vol Harvesting: {len(strategies['Vol_Harvesting'])} months [{strategies['Vol_Harvesting'].index[0]:%Y-%m} to {strategies['Vol_Harvesting'].index[-1]:%Y-%m}] (file: {csv_files[0]})")

    return strategies


# =============================================================================
# PART 2: Portfolio Construction
# =============================================================================

def compute_metrics(returns, name=""):
    """Compute standard performance metrics from monthly return series."""
    if len(returns) < 2:
        return {}

    total_return = (1 + returns).prod() - 1
    n_years = len(returns) / 12
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown from monthly returns
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max) - 1
    max_dd = dd.min()

    win_rate = (returns > 0).sum() / len(returns)

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    return {
        'name': name,
        'cagr': cagr,
        'ann_ret': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': pf,
        'total_return': total_return,
        'n_months': len(returns),
    }


def equal_weight_portfolio(returns_df, lookback=36, rebal_freq=3, tcost=0.001):
    """Equal weight 1/N portfolio with quarterly rebalance and transaction costs."""
    n_strats = returns_df.shape[1]
    weights = np.ones(n_strats) / n_strats

    port_returns = []
    prev_weights = weights.copy()

    for i, date in enumerate(returns_df.index):
        row = returns_df.iloc[i].fillna(0).values

        # Rebalance quarterly
        if i % rebal_freq == 0 and i > 0:
            # Transaction cost: proportional to turnover
            turnover = np.sum(np.abs(prev_weights - weights))
            tc = turnover * tcost
        else:
            tc = 0

        port_ret = np.dot(weights, row) - tc
        port_returns.append(port_ret)

        # Drift weights based on returns
        new_vals = prev_weights * (1 + row)
        prev_weights = new_vals / new_vals.sum() if new_vals.sum() > 0 else weights.copy()

        # Reset to equal weight on rebalance
        if (i + 1) % rebal_freq == 0:
            prev_weights = weights.copy()

    return pd.Series(port_returns, index=returns_df.index, name='EqualWeight')


def risk_parity_portfolio(returns_df, lookback=36, rebal_freq=3, tcost=0.001):
    """Inverse-volatility (risk parity) portfolio, walk-forward."""
    n_strats = returns_df.shape[1]
    port_returns = []
    prev_weights = np.ones(n_strats) / n_strats

    for i, date in enumerate(returns_df.index):
        row = returns_df.iloc[i].fillna(0).values

        # Compute weights from lookback window
        if i >= lookback:
            window = returns_df.iloc[i - lookback:i].fillna(0)
            vols = window.std()
            vols = vols.replace(0, vols[vols > 0].min() if (vols > 0).any() else 1)
            inv_vol = 1.0 / vols
            weights = (inv_vol / inv_vol.sum()).values
        elif i >= 6:
            # Shorter lookback if not enough history
            window = returns_df.iloc[:i].fillna(0)
            vols = window.std()
            vols = vols.replace(0, vols[vols > 0].min() if (vols > 0).any() else 1)
            inv_vol = 1.0 / vols
            weights = (inv_vol / inv_vol.sum()).values
        else:
            weights = np.ones(n_strats) / n_strats

        # Rebalance quarterly
        if i % rebal_freq == 0 and i > 0:
            turnover = np.sum(np.abs(prev_weights - weights))
            tc = turnover * tcost
        else:
            tc = 0
            weights = prev_weights  # Don't change weights outside rebalance

        port_ret = np.dot(weights, row) - tc
        port_returns.append(port_ret)

        # Drift weights
        new_vals = weights * (1 + row)
        prev_weights = new_vals / new_vals.sum() if new_vals.sum() > 0 else weights.copy()

        # Store new target weights on rebalance
        if (i + 1) % rebal_freq == 0:
            # Will be recalculated next rebalance
            pass

    return pd.Series(port_returns, index=returns_df.index, name='RiskParity')


def min_variance_portfolio(returns_df, lookback=36, rebal_freq=3, tcost=0.001):
    """Minimum variance portfolio via constrained optimization, walk-forward."""
    n_strats = returns_df.shape[1]
    port_returns = []
    prev_weights = np.ones(n_strats) / n_strats
    current_weights = prev_weights.copy()

    for i, date in enumerate(returns_df.index):
        row = returns_df.iloc[i].fillna(0).values

        # Rebalance quarterly with optimization
        if i % rebal_freq == 0 and i >= lookback:
            window = returns_df.iloc[i - lookback:i].fillna(0)
            cov = window.cov().values

            # Add small regularization for numerical stability
            cov += np.eye(n_strats) * 1e-8

            def portfolio_variance(w):
                return w @ cov @ w

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 0.30)] * n_strats
            x0 = np.ones(n_strats) / n_strats

            result = minimize(portfolio_variance, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-12})

            if result.success:
                current_weights = result.x
                current_weights = np.maximum(current_weights, 0)
                current_weights /= current_weights.sum()

            turnover = np.sum(np.abs(prev_weights - current_weights))
            tc = turnover * tcost
        elif i % rebal_freq == 0 and i >= 6:
            # Use shorter lookback
            window = returns_df.iloc[:i].fillna(0)
            cov = window.cov().values
            cov += np.eye(n_strats) * 1e-8

            def portfolio_variance(w):
                return w @ cov @ w

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0, 0.30)] * n_strats
            x0 = np.ones(n_strats) / n_strats

            result = minimize(portfolio_variance, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': 1000, 'ftol': 1e-12})

            if result.success:
                current_weights = result.x
                current_weights = np.maximum(current_weights, 0)
                current_weights /= current_weights.sum()

            turnover = np.sum(np.abs(prev_weights - current_weights))
            tc = turnover * tcost
        else:
            tc = 0

        port_ret = np.dot(current_weights, row) - tc
        port_returns.append(port_ret)

        # Drift weights
        new_vals = current_weights * (1 + row)
        if new_vals.sum() > 0:
            prev_weights = new_vals / new_vals.sum()
        else:
            prev_weights = current_weights.copy()

        # Reset to target on rebalance
        if (i + 1) % rebal_freq == 0:
            prev_weights = current_weights.copy()

    return pd.Series(port_returns, index=returns_df.index, name='MinVariance')


# =============================================================================
# PART 3: Dynamic Leverage Overlay
# =============================================================================

def download_macro_data(start_date, end_date):
    """Download real macro data using yfinance."""
    import yfinance as yf

    print("\nDownloading macro data...")
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # VIX
    try:
        vix = yf.download('^VIX', start=start_str, end=end_str, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix_monthly = vix['Close'].resample('ME').last()
        print(f"  VIX: {len(vix_monthly)} months")
    except Exception as e:
        print(f"  VIX download failed: {e}")
        vix_monthly = pd.Series(dtype=float)

    # SPY for breadth + trend signals
    try:
        spy = yf.download('SPY', start=start_str, end=end_str, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy_close = spy['Close']

        # 200-day SMA ratio (monthly)
        spy_sma200 = spy_close.rolling(200).mean()
        spy_breadth = (spy_close / spy_sma200).resample('ME').last()

        # 10-month SMA (CTA positioning proxy)
        spy_monthly_close = spy_close.resample('ME').last()
        spy_sma10m = spy_monthly_close.rolling(10).mean()
        spy_trend = spy_monthly_close / spy_sma10m

        print(f"  SPY: {len(spy_breadth)} months (breadth), {len(spy_trend)} months (trend)")
    except Exception as e:
        print(f"  SPY download failed: {e}")
        spy_breadth = pd.Series(dtype=float)
        spy_trend = pd.Series(dtype=float)

    # HYG vs LQD for credit spread
    try:
        hyg = yf.download('HYG', start=start_str, end=end_str, progress=False)
        lqd = yf.download('LQD', start=start_str, end=end_str, progress=False)
        if isinstance(hyg.columns, pd.MultiIndex):
            hyg.columns = hyg.columns.get_level_values(0)
        if isinstance(lqd.columns, pd.MultiIndex):
            lqd.columns = lqd.columns.get_level_values(0)

        hyg_monthly = hyg['Close'].resample('ME').last().pct_change()
        lqd_monthly = lqd['Close'].resample('ME').last().pct_change()
        # 3-month rolling spread
        credit_spread = (hyg_monthly - lqd_monthly).rolling(3).sum()
        print(f"  Credit (HYG-LQD): {len(credit_spread.dropna())} months")
    except Exception as e:
        print(f"  Credit spread download failed: {e}")
        credit_spread = pd.Series(dtype=float)

    # T-bill rate for margin cost
    try:
        irx = yf.download('^IRX', start=start_str, end=end_str, progress=False)
        if isinstance(irx.columns, pd.MultiIndex):
            irx.columns = irx.columns.get_level_values(0)
        tbill_rate = irx['Close'].resample('ME').last() / 100  # Convert from percentage
        print(f"  T-Bill rate: {len(tbill_rate.dropna())} months")
    except Exception as e:
        print(f"  T-Bill rate download failed, using 5% default: {e}")
        tbill_rate = pd.Series(dtype=float)

    return vix_monthly, spy_breadth, spy_trend, credit_spread, tbill_rate


def compute_leverage_score(dates, vix_monthly, spy_breadth, spy_trend, credit_spread):
    """
    Compute dynamic leverage score for each month.
    Base = 1.0x, signals add/subtract, clipped to [0.5, 2.0].
    """
    leverage_records = []

    for date in dates:
        base = 1.0
        vix_signal = 0.0
        breadth_signal = 0.0
        credit_signal = 0.0
        trend_signal = 0.0

        # VIX signal
        if date in vix_monthly.index:
            vix_val = vix_monthly.loc[date]
        else:
            # Try nearest prior date
            prior = vix_monthly[vix_monthly.index <= date]
            vix_val = prior.iloc[-1] if len(prior) > 0 else 20  # neutral default

        if vix_val < 15:
            vix_signal = 0.25
        elif vix_val <= 25:
            vix_signal = 0.0
        elif vix_val <= 35:
            vix_signal = -0.25
        else:
            vix_signal = -0.50

        # SPY > 200SMA signal
        if date in spy_breadth.index:
            breadth_val = spy_breadth.loc[date]
        else:
            prior = spy_breadth[spy_breadth.index <= date]
            breadth_val = prior.iloc[-1] if len(prior) > 0 else 1.0

        breadth_signal = 0.25 if breadth_val > 1.0 else -0.25

        # Credit spread signal (HYG outperforming LQD = risk-on)
        if date in credit_spread.index:
            credit_val = credit_spread.loc[date]
        else:
            prior = credit_spread[credit_spread.index <= date]
            credit_val = prior.iloc[-1] if len(prior) > 0 else 0.0

        if pd.notna(credit_val):
            credit_signal = 0.25 if credit_val > 0 else -0.25

        # CTA trend signal (SPY > 10-month SMA)
        if date in spy_trend.index:
            trend_val = spy_trend.loc[date]
        else:
            prior = spy_trend[spy_trend.index <= date]
            trend_val = prior.iloc[-1] if len(prior) > 0 else 1.0

        trend_signal = 0.25 if trend_val > 1.0 else -0.25

        total_leverage = np.clip(base + vix_signal + breadth_signal + credit_signal + trend_signal, 0.5, 2.0)

        leverage_records.append({
            'date': date,
            'vix_level': float(vix_val) if not pd.isna(vix_val) else np.nan,
            'vix_signal': vix_signal,
            'breadth_ratio': float(breadth_val) if not pd.isna(breadth_val) else np.nan,
            'breadth_signal': breadth_signal,
            'credit_spread_3m': float(credit_val) if not pd.isna(credit_val) else np.nan,
            'credit_signal': credit_signal,
            'trend_ratio': float(trend_val) if not pd.isna(trend_val) else np.nan,
            'trend_signal': trend_signal,
            'leverage': total_leverage,
        })

    return pd.DataFrame(leverage_records).set_index('date')


def apply_leverage(port_returns, leverage_df, tbill_rate):
    """Apply dynamic leverage to portfolio returns, deducting margin cost on borrowed portion."""
    leveraged = []

    for date in port_returns.index:
        ret = port_returns.loc[date]

        if date in leverage_df.index:
            lev = leverage_df.loc[date, 'leverage']
        else:
            lev = 1.0

        # Margin cost on borrowed portion (annualized → monthly)
        if lev > 1.0:
            borrowed = lev - 1.0
            if date in tbill_rate.index:
                annual_rate = tbill_rate.loc[date]
            else:
                prior = tbill_rate[tbill_rate.index <= date]
                annual_rate = prior.iloc[-1] if len(prior) > 0 else 0.05

            if pd.isna(annual_rate):
                annual_rate = 0.05

            monthly_cost = borrowed * annual_rate / 12
        else:
            monthly_cost = 0.0

        leveraged_ret = ret * lev - monthly_cost
        leveraged.append(leveraged_ret)

    return pd.Series(leveraged, index=port_returns.index, name=port_returns.name)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("COMBINED PORTFOLIO BACKTEST v2 -- REAL DATA ONLY")
    print("=" * 70)

    # --- Load strategies ---
    print("\nLoading strategy returns from disk...")
    strategies = load_all_strategies()

    if len(strategies) < 2:
        print("ERROR: Need at least 2 strategies. Exiting.")
        sys.exit(1)

    print(f"\nLoaded {len(strategies)} strategies")

    # Build aligned return matrix
    returns_df = pd.DataFrame(strategies)
    # Align to common monthly index
    returns_df.index = returns_df.index.to_period('M').to_timestamp()
    returns_df = returns_df.sort_index()

    # Find overlap where ALL strategies have data
    overlap = returns_df.dropna(how='any')

    # Also build a version using all available data (fill missing with 0 = cash)
    full_df = returns_df.fillna(0)

    print(f"\nFull period: {returns_df.index[0]:%Y-%m} to {returns_df.index[-1]:%Y-%m} ({len(returns_df)} months)")
    print(f"All-overlap period: {overlap.index[0]:%Y-%m} to {overlap.index[-1]:%Y-%m} ({len(overlap)} months)" if len(overlap) > 0 else "No full overlap period!")

    # Use full data with fillna(0) for strategies not yet live
    # This is conservative — missing strategy = cash position
    use_df = full_df
    print(f"\nUsing FULL period with missing = cash: {len(use_df)} months")

    # Data coverage per strategy
    print("\nStrategy data coverage:")
    for col in returns_df.columns:
        valid = returns_df[col].dropna()
        print(f"  {col:25s}: {len(valid):4d} months  [{valid.index[0]:%Y-%m} to {valid.index[-1]:%Y-%m}]")

    # Correlation matrix
    corr = returns_df.corr()
    print(f"\nPairwise correlation matrix:")
    print(corr.round(3).to_string())

    # Average pairwise correlation (excluding diagonal)
    mask = np.ones(corr.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    avg_corr = corr.values[mask].mean()
    print(f"\nAverage pairwise correlation: {avg_corr:.3f}")

    # Save correlation matrix
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Save aligned strategy returns
    returns_df.to_csv(OUTPUT_DIR / "strategy_returns.csv")

    # --- Build 1x portfolios ---
    print("\n" + "=" * 70)
    print("PORTFOLIO CONSTRUCTION (1x LEVERAGE)")
    print("=" * 70)

    ew_1x = equal_weight_portfolio(use_df)
    rp_1x = risk_parity_portfolio(use_df)
    mv_1x = min_variance_portfolio(use_df)

    port_1x = pd.DataFrame({
        'EqualWeight': ew_1x,
        'RiskParity': rp_1x,
        'MinVariance': mv_1x,
    })
    port_1x.to_csv(OUTPUT_DIR / "portfolio_1x.csv")

    metrics_1x = {
        'EqualWeight': compute_metrics(ew_1x, 'EqualWeight'),
        'RiskParity': compute_metrics(rp_1x, 'RiskParity'),
        'MinVariance': compute_metrics(mv_1x, 'MinVariance'),
    }

    # --- Download macro data and build leverage overlay ---
    print("\n" + "=" * 70)
    print("DYNAMIC LEVERAGE OVERLAY")
    print("=" * 70)

    # Need buffer for lookback calculations
    macro_start = use_df.index[0] - pd.DateOffset(months=15)
    macro_end = use_df.index[-1] + pd.DateOffset(months=1)

    vix_monthly, spy_breadth, spy_trend, credit_spread, tbill_rate = download_macro_data(macro_start, macro_end)

    # Normalize all macro indices to month-end
    for s in [vix_monthly, spy_breadth, spy_trend, credit_spread, tbill_rate]:
        if len(s) > 0:
            s.index = s.index.to_period('M').to_timestamp()

    leverage_df = compute_leverage_score(use_df.index, vix_monthly, spy_breadth, spy_trend, credit_spread)
    leverage_df.to_csv(OUTPUT_DIR / "leverage_history.csv")

    print(f"\nLeverage stats:")
    print(f"  Mean: {leverage_df['leverage'].mean():.2f}x")
    print(f"  Range: [{leverage_df['leverage'].min():.1f}, {leverage_df['leverage'].max():.1f}]")
    print(f"  % time > 1x: {(leverage_df['leverage'] > 1.0).mean() * 100:.1f}%")
    print(f"  % time < 1x: {(leverage_df['leverage'] < 1.0).mean() * 100:.1f}%")

    # Apply leverage to each portfolio method
    ew_lev = apply_leverage(ew_1x, leverage_df, tbill_rate)
    rp_lev = apply_leverage(rp_1x, leverage_df, tbill_rate)
    mv_lev = apply_leverage(mv_1x, leverage_df, tbill_rate)

    port_lev = pd.DataFrame({
        'EqualWeight': ew_lev,
        'RiskParity': rp_lev,
        'MinVariance': mv_lev,
    })
    port_lev.to_csv(OUTPUT_DIR / "portfolio_leveraged.csv")

    metrics_lev = {
        'EqualWeight': compute_metrics(ew_lev, 'EqualWeight_Leveraged'),
        'RiskParity': compute_metrics(rp_lev, 'RiskParity_Leveraged'),
        'MinVariance': compute_metrics(mv_lev, 'MinVariance_Leveraged'),
    }

    # --- Individual strategy metrics ---
    strat_metrics = {}
    for name, series in strategies.items():
        strat_metrics[name] = compute_metrics(series, name)

    # --- Save results JSON ---
    all_results = {
        'portfolio_1x': metrics_1x,
        'portfolio_leveraged': metrics_lev,
        'individual_strategies': strat_metrics,
        'leverage_stats': {
            'mean': float(leverage_df['leverage'].mean()),
            'min': float(leverage_df['leverage'].min()),
            'max': float(leverage_df['leverage'].max()),
            'pct_above_1x': float((leverage_df['leverage'] > 1.0).mean()),
        },
        'avg_pairwise_correlation': float(avg_corr),
        'n_strategies': len(strategies),
        'period': f"{use_df.index[0]:%Y-%m} to {use_df.index[-1]:%Y-%m}",
        'n_months': len(use_df),
    }

    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("COMBINED PORTFOLIO -- REAL DATA ONLY (HC #717)")
    print("=" * 70)

    print("\n1x LEVERAGE:")
    for method in ['EqualWeight', 'RiskParity', 'MinVariance']:
        m = metrics_1x[method]
        label = {'EqualWeight': 'Equal Weight', 'RiskParity': 'Risk Parity', 'MinVariance': 'Min Variance'}[method]
        print(f"  {label:14s}: CAGR {m['cagr']*100:5.1f}% | Sharpe {m['sharpe']:.2f} | Sortino {m['sortino']:.2f} | MaxDD {m['max_dd']*100:6.1f}% | WR {m['win_rate']*100:.1f}%")

    print(f"\nDYNAMIC LEVERAGE (VIX + Credit + Breadth + CTA):")
    for method in ['EqualWeight', 'RiskParity', 'MinVariance']:
        m = metrics_lev[method]
        label = {'EqualWeight': 'Equal Weight', 'RiskParity': 'Risk Parity', 'MinVariance': 'Min Variance'}[method]
        print(f"  {label:14s}: CAGR {m['cagr']*100:5.1f}% | Sharpe {m['sharpe']:.2f} | Sortino {m['sortino']:.2f} | MaxDD {m['max_dd']*100:6.1f}% | WR {m['win_rate']*100:.1f}%")

    lev = leverage_df['leverage']
    print(f"\nLeverage stats: Mean {lev.mean():.2f}x, Range [{lev.min():.1f}, {lev.max():.1f}], % time > 1x: {(lev > 1.0).mean()*100:.0f}%")
    print(f"Period: {use_df.index[0]:%Y-%m} to {use_df.index[-1]:%Y-%m} ({len(use_df)} months)")
    print(f"Strategies: {len(strategies)}")
    print(f"Avg pairwise correlation: {avg_corr:.3f}")

    # --- Yearly returns for best 1x method ---
    best_1x_name = max(metrics_1x.keys(), key=lambda k: metrics_1x[k]['sharpe'])
    best_1x_label = {'EqualWeight': 'Equal Weight', 'RiskParity': 'Risk Parity', 'MinVariance': 'Min Variance'}[best_1x_name]
    best_1x = port_1x[best_1x_name]

    print(f"\n{'='*70}")
    print(f"YEARLY RETURNS -- {best_1x_label} (best 1x Sharpe)")
    print(f"{'='*70}")

    yearly = best_1x.groupby(best_1x.index.year).apply(lambda x: (1 + x).prod() - 1)
    print(f"{'Year':>6s}  {'Return':>8s}  {'Months':>6s}")
    print("-" * 24)
    for year, ret in yearly.items():
        n_months = (best_1x.index.year == year).sum()
        print(f"{year:>6d}  {ret*100:>7.1f}%  {n_months:>6d}")

    # --- Individual strategy metrics ---
    print(f"\n{'='*70}")
    print("INDIVIDUAL STRATEGY METRICS")
    print(f"{'='*70}")
    print(f"{'Strategy':25s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'Sortino':>8s}  {'MaxDD':>7s}  {'Months':>6s}")
    print("-" * 70)
    for name in sorted(strat_metrics.keys()):
        m = strat_metrics[name]
        print(f"{name:25s}  {m['cagr']*100:6.1f}%  {m['sharpe']:7.2f}  {m['sortino']:8.2f}  {m['max_dd']*100:6.1f}%  {m['n_months']:6d}")

    # --- Caveats ---
    print(f"\n{'='*70}")
    print("CAVEATS")
    print(f"{'='*70}")
    print("- Survivorship bias: these strategies were selected from 36+ tested")
    print("- Bull market period bias (backtest largely covers 2012-2026)")
    print("- Individual strategy Sharpes may be inflated (ML walk-forward on ETF data)")
    print("- Transaction costs assumed (0.1%/rebal) but may understate real friction")
    print("- Dynamic leverage uses margin -- real margin rates may differ from T-bill proxy")
    print("- Missing strategy data filled with 0 (cash) -- conservative but understates diversification in early period")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  results.json, strategy_returns.csv, portfolio_1x.csv,")
    print("  portfolio_leveraged.csv, leverage_history.csv, correlation_matrix.csv")


if __name__ == "__main__":
    main()
